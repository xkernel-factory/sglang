"""Host-side planning for bounded DSV4 indexer logits allocations."""

import os


def indexer_logits_budget_bytes(*, legacy_fp4: bool = False) -> int:
    name = "SGLANG_DSV4_INDEXER_LOGITS_BUDGET_MB"
    value = os.environ.get(name)
    if value is None:
        if legacy_fp4:
            name = "SGLANG_DSV4_FP4_LOGITS_BUDGET_MB"
            value = os.environ.get(name, "2048")
        else:
            value = "512"
    try:
        budget_mb = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if budget_mb <= 0:
        raise ValueError(f"{name} must be positive, got {budget_mb}")
    return budget_mb * 2**20


def paged_logits_rows_per_chunk(max_context_len: int, budget_bytes: int) -> int:
    # DeepGEMM FP8 paged logits use FP32 output and a 256-element aligned
    # row stride (split_kv=256 and 1024-byte alignment).
    aligned_width = max(256, (max_context_len + 255) // 256 * 256)
    bytes_per_row = aligned_width * 4
    rows = budget_bytes // bytes_per_row
    if rows < 1:
        raise ValueError(
            f"Indexer logits budget {budget_bytes} bytes cannot fit one "
            f"aligned row ({bytes_per_row} bytes)"
        )
    # Limit row-count variants; width and tail chunks still vary, so this is
    # not a fixed-size pool and does not guarantee allocator reuse.
    return 1 << (rows.bit_length() - 1)
