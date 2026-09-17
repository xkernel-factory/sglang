from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields, replace
from typing import Any, List, Optional

import torch

from sglang.kernels.ops.attention.dsv4.logits_budget import (
    indexer_logits_budget_bytes,
    paged_logits_rows_per_chunk,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_hip, is_sm120_supported, is_xpu

_IS_SM120 = is_sm120_supported()

"""
Some comments on the common terms used in DeepSeekV4Backend:

topk_lengths:
    NOTE: TL;DR: topk_lengths == seq_lens
    The FlashMLA sparse decode kernel will attend to `k` tokens for each query.
    `topk_lengths` indicates how many tokens each query will attend to.
    This should be named as `seq_lens`, but we simply follow the naming convention.

page_table:
    The page table indicates which pages each request is assigned to.
    Each value in the page table is the page index in the TokenToKVPool.
    This page index is irrelevant to the actual `page_size`.

page_indices:
    The real indices used to index into the KV cache.
    This can be computed from the `page_table` and `page_size`.
    e.g. page_indices[i, j] = page_table[i, j // page_size] * page_size + (j % page_size)
    For sparse C4 top-512 attention, the indices will be selected from the C4 page indices.
    In implementation, we don't materialize the full C4 `page_indices`,
    but calculate them from `page_table` on-the-fly in the attention kernel.

positions:
    The position of the last token for each request.
    For compress token, the positions must be times of compress ratio.
    For example, for C4, raw_position=11 will trigger a compression,
    But the RoPE's position, during compression, must be 8 instead of 11.

Some other notes:
    c4_ / c128_: means "compressed by 4" / "compressed by 128".
    compressed_page_size: physical indexer pool page size
    compressed_seq_lens: seq_lens // 4, but bounded by at least 1, due to flash_mla requirement.
    c4_sparse: means "compressed by 4" but only attend to top-512 tokens.
               all related length will be clipped to 512.
"""
_LARGE_INDEXER_QUERY_THRESHOLD = 11673

_SM120_INDEXER_M_CHUNK = 4096


def copy_metadata(
    *,
    src,
    dst,
    check_eq_fields: List[str],
    copy_fields: List[str],
    assign_fields: Optional[List[str]] = None,
):
    assign_fields = assign_fields or []

    for field_name in check_eq_fields:
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)
        assert src_val == dst_val, f"{field_name=} {src_val=} {dst_val=}"

    for field_name in copy_fields:
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)
        if src_val is None and dst_val is None:
            continue
        assert dst_val is not None, f"{field_name=} {src_val=} {dst_val=}"
        if hasattr(dst_val, "copy_"):
            dst_val.copy_(src_val)
        elif isinstance(dst_val, list) and isinstance(src_val, list):
            # Captured kernels retain these tensor addresses. Rebinding a list
            # leaves graph replays reading the previous forward's plans.
            assert len(dst_val) == len(src_val), f"{field_name}: chunk count changed"
            for dst_item, src_item in zip(dst_val, src_val):
                dst_item.copy_(src_item)
        else:
            warnings.warn(
                f"{field_name=} {type(dst_val)=} does not have copy_, use setattr"
            )
            setattr(dst, field_name, src_val)

    for field_name in assign_fields:
        setattr(dst, field_name, getattr(src, field_name))

    provided_fields = check_eq_fields + copy_fields + assign_fields
    provided_fields_unique = set(provided_fields)
    assert len(provided_fields) == len(provided_fields_unique), (
        f"{provided_fields=} has dup"
    )
    all_fields = {f.name for f in fields(src)}
    provided_fields = set(provided_fields)
    assert provided_fields == all_fields, (
        f"{provided_fields - all_fields=}, {all_fields - provided_fields=}"
    )


@dataclass
class NonPagedIndexerPlan:
    page_table: torch.Tensor
    gather_seq_lens: torch.Tensor
    ks: torch.Tensor
    ke: torch.Tensor
    seq_len_sum: int
    max_seq_len: int
    max_seqlen_k: int
    query_rows: int


@dataclass
class PagedIndexerMetadata:
    page_size: int
    compressed_page_size: int
    page_table: torch.Tensor
    compressed_seq_lens: torch.Tensor
    use_topk_v2: bool
    force_deep_gemm_metadata: bool = False
    use_prefill_cuda_graph: bool = False
    is_prefill: bool = False
    logits_chunk_rows: int = field(init=False, default=0)
    chunk_topk_metadata: Optional[List[torch.Tensor]] = field(
        init=False, repr=False, default=None
    )
    deep_gemm_metadata: Any = field(init=False, repr=False)
    topk_metadata: torch.Tensor = field(init=False, repr=False)
    nonpaged_plan: Optional[NonPagedIndexerPlan] = field(
        init=False, repr=False, default=None
    )
    _aligned_query_cache: dict[int, PagedIndexerMetadata] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self):
        if (
            is_hip() or is_xpu() or envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
        ) and not self.force_deep_gemm_metadata:
            self.deep_gemm_metadata = None
        else:
            import deep_gemm

            use_jit_indexer = not self.force_deep_gemm_metadata and (
                envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.get()
                or self.compressed_seq_lens.numel() > _LARGE_INDEXER_QUERY_THRESHOLD
            )
            if use_jit_indexer:
                from sglang.kernels.ops.attention.dsv4 import (
                    get_paged_mqa_logits_metadata,
                )
            else:
                from deep_gemm import get_paged_mqa_logits_metadata

            compressed_seq_lens = self.compressed_seq_lens.to(torch.int32)
            if compressed_seq_lens.dim() == 1:
                compressed_seq_lens = compressed_seq_lens.unsqueeze(-1)
            num_rows = compressed_seq_lens.shape[0]
            chunk_rows = max(1, num_rows)
            if self.is_prefill and not self.use_prefill_cuda_graph:
                chunk_rows = min(
                    chunk_rows,
                    paged_logits_rows_per_chunk(
                        self.max_compressed_seq_len, indexer_logits_budget_bytes()
                    ),
                )
            if _IS_SM120:
                chunk_rows = min(chunk_rows, _SM120_INDEXER_M_CHUNK)
            self.logits_chunk_rows = chunk_rows if num_rows > chunk_rows else 0
            if num_rows > chunk_rows:
                # Chunk metadata is shared by all indexer layers in this forward.
                self.deep_gemm_metadata = [
                    get_paged_mqa_logits_metadata(
                        compressed_seq_lens[_s : _s + chunk_rows],
                        self.compressed_page_size,
                        deep_gemm.get_num_sms(),
                    )
                    for _s in range(0, num_rows, chunk_rows)
                ]
            else:
                self.deep_gemm_metadata = get_paged_mqa_logits_metadata(
                    compressed_seq_lens,
                    self.compressed_page_size,
                    deep_gemm.get_num_sms(),
                )

            assert isinstance(self.deep_gemm_metadata, (torch.Tensor, list))

        if self.use_topk_v2:
            from sglang.kernels.ops.attention.dsv4 import plan_topk_v2

            self.topk_metadata = plan_topk_v2(self.compressed_seq_lens)
            if isinstance(self.deep_gemm_metadata, list):
                self.chunk_topk_metadata = [
                    plan_topk_v2(
                        self.compressed_seq_lens[start : start + self.logits_chunk_rows]
                    )
                    for start in range(
                        0, self.compressed_seq_lens.shape[0], self.logits_chunk_rows
                    )
                ]
        else:
            self.topk_metadata = torch.empty((0,))

        assert self.page_size == 256, "the system hardcodes page_size=256"

    @property
    def max_seq_len(self) -> int:
        return self.page_table.shape[1] * self.page_size

    @property
    def max_compressed_seq_len(self) -> int:
        return self.page_table.shape[1] * self.compressed_page_size

    def for_query_rows(self, query_rows: int) -> PagedIndexerMetadata:
        """Reuse eager-prefill alignment plans across layers of this forward."""
        if (
            query_rows == self.compressed_seq_lens.shape[0]
            or not isinstance(self.deep_gemm_metadata, list)
            or not self.is_prefill
            or self.use_prefill_cuda_graph
        ):
            return self
        if query_rows not in self._aligned_query_cache:
            from torch.nn.functional import pad

            def align(tensor, value):
                if tensor.shape[0] >= query_rows:
                    return tensor[:query_rows]
                padding = (0, 0) * (tensor.dim() - 1) + (
                    0,
                    query_rows - tensor.shape[0],
                )
                return pad(tensor, padding, value=value)

            # replace reruns __post_init__, building both schedules once for
            # these rows. Its init=False cache starts empty, without cycles.
            self._aligned_query_cache[query_rows] = replace(
                self,
                compressed_seq_lens=align(self.compressed_seq_lens, 1),
                page_table=align(self.page_table, 0),
            )
        return self._aligned_query_cache[query_rows]

    def copy_(self, other: PagedIndexerMetadata):
        if is_hip():
            copy_fields = ["page_table", "compressed_seq_lens"]
            assign_fields = ["deep_gemm_metadata", "nonpaged_plan"]
        else:
            copy_fields = ["page_table", "compressed_seq_lens", "deep_gemm_metadata"]
            assign_fields = ["nonpaged_plan"]
        copy_fields += ["topk_metadata", "chunk_topk_metadata"]
        assign_fields += ["_aligned_query_cache"]
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=[
                "page_size",
                "compressed_page_size",
                "force_deep_gemm_metadata",
                "use_prefill_cuda_graph",
                "use_topk_v2",
                "is_prefill",
                "logits_chunk_rows",
            ],
            copy_fields=copy_fields,
            assign_fields=assign_fields,
        )
        self.nonpaged_plan = None
        # copy_ refreshes this object for another forward. Never reuse (or
        # clear in place) the source forward's derived alignment cache.
        self._aligned_query_cache = {}


def maybe_copy_inplace(dst, *, src) -> None:
    assert type(src) == type(dst)
    if dst is not None:
        dst.copy_(src)
