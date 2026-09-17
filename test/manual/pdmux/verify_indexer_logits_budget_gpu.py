"""Real CUDA validation of the production DSV4 paged indexer dispatch.

Run from the repository root. No model weights needed. This is a single-GPU
kernel check, not a TP8/HiCache or CUDA Green Context integration test.
"""

import argparse
import ast
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def production_dispatch():
    # Execute the actual nested production functions with real GPU kernels;
    # avoid fabricating a complete model/compressor just to test this boundary.
    path = Path(__file__).resolve().parents[3] / (
        "python/sglang/srt/layers/attention/dsv4/indexer.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forward = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "forward_c4_indexer"
    )
    start = next(
        i
        for i, n in enumerate(forward.body)
        if isinstance(n, ast.FunctionDef) and n.name == "match_num_queries"
    )
    stop = next(
        i
        for i, n in enumerate(forward.body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "c4_sparse_page_indices"
            for t in n.targets
        )
    )
    prepare = compile(
        ast.Module(body=forward.body[start:stop], type_ignores=[]), str(path), "exec"
    )
    topk = next(
        n
        for n in ast.walk(forward)
        if isinstance(n, ast.FunctionDef) and n.name == "run_topk_transform"
    )
    branch = next(
        n.orelse
        for n in ast.walk(forward)
        if isinstance(n, ast.If)
        and any(
            isinstance(c, ast.FunctionDef) and c.name == "run_paged_indexer"
            for c in n.orelse
        )
    )
    first = next(i for i, n in enumerate(branch) if isinstance(n, ast.FunctionDef))
    dispatch = compile(
        ast.Module(body=[topk, *branch[first:]], type_ignores=[]), str(path), "exec"
    )
    return prepare, dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=259)
    parser.add_argument(
        "--width",
        type=int,
        default=8192,
        help="Compressed context width, a multiple of 256",
    )
    parser.add_argument(
        "--budget-mb",
        type=int,
        default=4,
        help="Small default deliberately exercises splitting",
    )
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", default="indexer-budget-gpu.json")
    args = parser.parse_args()
    if args.width < 1024 or args.width % 256 or args.rows < 1 or args.repeat < 1:
        parser.error(
            "width must be a multiple of 256 >= 1024; rows/repeat must be positive"
        )

    import deep_gemm
    import torch
    import torch.nn.functional as F
    from sglang.kernels.ops.attention.dsv4 import (
        plan_topk_v2,
        topk_transform_paged,
        topk_transform_paged_v2,
    )
    from sglang.kernels.ops.attention.dsv4.logits_budget import (
        paged_logits_rows_per_chunk,
    )
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata

    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError("Requires NVIDIA CUDA and the deployment's DeepGEMM build")
    if torch.cuda.get_device_capability()[0] not in (9, 10):
        raise RuntimeError("This validator targets SM90/SM100 FP8; not SM120 FP4")
    torch.manual_seed(1234)
    chunk = paged_logits_rows_per_chunk(args.width, args.budget_mb * 2**20)
    if args.rows <= chunk:
        parser.error(f"rows must exceed chunk rows ({chunk}) to exercise the fix")
    prepare, dispatch = production_dispatch()
    max_rows = max(5, args.rows, chunk * 2 + 1)
    if max_rows * args.width * 4 > torch.cuda.mem_get_info()[0] // 2:
        raise RuntimeError("Whole-batch reference is too large; reduce --rows/--width")
    pages = args.width // 64
    # Shared identity page table makes transformed indices directly comparable
    # to logical reference columns. Each query still has independent scores.
    table = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(max_rows, 1)
    lens = torch.randint(
        1, args.width + 1, (max_rows,), device="cuda", dtype=torch.int32
    )
    lens[:5] = torch.tensor([1, 511, 512, 513, args.width], device="cuda")
    q = torch.randn(max_rows, 1, 32, 128, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.rand(max_rows, 32, device="cuda", dtype=torch.float32)
    values = torch.randn(pages, 64, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.full((pages, 64), 0.125, device="cuda", dtype=torch.float32)
    cache = torch.cat(
        (
            values.view(torch.uint8).reshape(pages, -1),
            scales.view(torch.uint8).reshape(pages, -1),
        ),
        dim=1,
    )
    cache = cache.view(torch.float8_e4m3fn).reshape(pages, 64, 1, 132)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "deep_gemm": getattr(deep_gemm, "__version__", "unknown"),
        "args": vars(args),
        "chunk_rows": chunk,
        "cases": [],
    }

    def build(rows, budget_mb, v2):
        with patch.dict(
            os.environ, {"SGLANG_DSV4_INDEXER_LOGITS_BUDGET_MB": str(budget_mb)}
        ):
            return PagedIndexerMetadata(
                page_size=256,
                compressed_page_size=64,
                page_table=table[:rows],
                compressed_seq_lens=lens[:rows],
                use_topk_v2=v2,
                is_prefill=True,
            )

    def namespace(m, rows, v2):
        ns = dict(
            torch=torch,
            F=F,
            replace=replace,
            query_rows=rows,
            indexer_metadata=m,
            use_aiter_fp4=False,
            q=q[:rows],
            weights=weights[:rows],
            c4_indexer_kv_cache=cache,
            raw_indices=None,
            all_rows=slice(0, rows),
            c4_sparse_page_indices=torch.empty(
                rows, 512, device="cuda", dtype=torch.int32
            ),
            plan_topk_v2=plan_topk_v2,
            topk_transform_paged=topk_transform_paged,
            topk_transform_paged_v2=topk_transform_paged_v2,
            is_pdmux_enabled=lambda: True,
            self=SimpleNamespace(
                dsa_topk_backend=SimpleNamespace(
                    is_torch=lambda: False,
                    is_flashinfer=lambda: False,
                    should_use_topk_v2=lambda: v2,
                )
            ),
        )
        with patch.dict(
            os.environ, {"SGLANG_DSV4_INDEXER_LOGITS_BUDGET_MB": str(args.budget_mb)}
        ):
            exec(prepare, ns)
        ns["_c4sl"] = ns["c4_seq_lens"].unsqueeze(-1)
        ns["fn"] = deep_gemm.fp8_paged_mqa_logits
        return ns

    def correctness(ns):
        captured = []

        def capture(*a):
            logits = deep_gemm.fp8_paged_mqa_logits(*a)
            # Invalid columns are intentionally uninitialized (clean=False).
            valid = torch.arange(args.width, device="cuda")[None, :] < a[3].reshape(
                -1, 1
            )
            captured.append(
                logits[:, : args.width].masked_fill(~valid, -float("inf")).cpu()
            )
            return logits

        ns["fn"] = capture
        exec(dispatch, ns)
        torch.cuda.synchronize()
        ns["fn"] = deep_gemm.fp8_paged_mqa_logits
        return torch.cat(captured), ns["c4_sparse_page_indices"].cpu()

    def measure(ns):
        for _ in range(3):
            exec(dispatch, ns)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        for _ in range(args.repeat):
            exec(dispatch, ns)
        end.record()
        end.synchronize()
        return {
            "kernel_ms": start.elapsed_time(end) / args.repeat,
            "peak_extra_allocated_mib": (torch.cuda.max_memory_allocated() - baseline)
            / 2**20,
        }

    cases = [
        ("exact", chunk, chunk),
        ("tail_one", chunk + 1, chunk + 1),
        ("tail", args.rows, args.rows),
        ("pad_cross_chunk", chunk + 1, chunk * 2 + 1),
        ("crop_tail", chunk * 2 + 1, chunk + 1),
        ("crop_to_single", chunk + 1, chunk),
    ]
    for v2 in (False, True):
        for name, original_rows, actual_rows in cases:
            # Build a large-budget single-pass baseline over the SAME final rows
            # and values, including padded sequence lengths of one.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            m = build(original_rows, args.budget_mb, v2)
            if original_rows > chunk and not isinstance(m.deep_gemm_metadata, list):
                raise AssertionError(
                    "Chunking inactive; check indexer backend environment overrides"
                )
            ns = namespace(m, actual_rows, v2)
            torch.cuda.synchronize()
            plan_ms = (time.perf_counter() - t0) * 1000
            final = ns["indexer_metadata"]
            with patch.dict(
                os.environ, {"SGLANG_DSV4_INDEXER_LOGITS_BUDGET_MB": "65536"}
            ):
                ref = replace(final)
            ref_ns = namespace(ref, actual_rows, v2)
            reference, ref_indices = correctness(ref_ns)
            scores, indices = correctness(ns)
            torch.testing.assert_close(scores, reference, rtol=1e-4, atol=1e-4)
            # Order and tied indices may differ. Check valid unique indices and
            # selected score multiset against a torch TopK reference.
            lengths = ns["c4_seq_lens"].cpu()
            for output in (indices, ref_indices):
                for row, length in enumerate(lengths.tolist()):
                    chosen = output[row][output[row] >= 0].long()
                    assert chosen.numel() == min(length, 512), (name, row, "count")
                    assert chosen.unique().numel() == chosen.numel(), (
                        name,
                        row,
                        "duplicates",
                    )
                    assert bool((chosen < length).all()), (name, row, "invalid index")
                    expected = (
                        reference[row, :length]
                        .topk(min(length, 512))
                        .values.sort()
                        .values
                    )
                    actual = reference[row, chosen].sort().values
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            result = {
                "case": name,
                "topk": 2 if v2 else 1,
                "metadata_rows": original_rows,
                "query_rows": actual_rows,
                "plan_wall_ms_including_first_use_jit": plan_ms,
                "whole": measure(ref_ns),
                "chunked": measure(ns),
            }
            result["kernel_time_ratio"] = (
                result["chunked"]["kernel_ms"] / result["whole"]["kernel_ms"]
            )
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: all logits and TopK checks passed; report: {args.output}")


if __name__ == "__main__":
    main()
