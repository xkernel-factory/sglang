# DSV4 paged indexer logits budget

NVIDIA eager prefill (including PDMux layer-split prefill) now splits paged
indexer query rows to bound each FP32 logits allocation. This targets OOMs
inside `deep_gemm.fp8_paged_mqa_logits`; it does not establish or fix a HiCache
synchronization root cause.

```bash
export SGLANG_DSV4_INDEXER_LOGITS_BUDGET_MB=512
# Run the existing launch command and workload with the same HiCache settings.
```

The value is a positive integer in MiB. NVIDIA defaults to 512 MiB. For AMD
FP4, the new setting overrides `SGLANG_DSV4_FP4_LOGITS_BUDGET_MB`; without the
new setting, the old variable and its 2048 MiB default remain effective.

Rows are budgeted against DeepGEMM's aligned FP32 output stride and rounded
down to a power of two. SM120 also retains its 4096-row metadata limit. At
1,048,576 original context tokens, C4 width is 262,144: the default allows
512 rows (512 MiB), instead of 4096 rows (4 GiB). A tail chunk is smaller.
DeepGEMM and TopK v2 plans use matching row slices and are built once per
forward, then reused across layers.

This bounds one logits output, not total GPU memory or every workspace.
It is not a fixed-size NVIDIA memory pool: context width and tail sizes still
vary. Decode, target verify, CUDA Graph prefill and the separate nonpaged
indexer path do not acquire the new budget-based split. Existing SM120
row-limit splitting is preserved, including matching chunk-local TopK plans.

## Validation

CPU regression tests (kernel planners/scoring use test doubles):

```bash
python -m pytest -q test/registered/unit/layers/test_dsv4_indexer_logits_budget.py
```

GPU validation remains required on the deployment's actual DeepGEMM build:

1. Compare paged indexer logits/TopK results for a small input under a large
   budget (one chunk) and a small budget (multiple chunks, including a tail).
   Exercise TopK v1/v2 and any selected alternative TopK backend.
2. Replay the failing TP8 workload at 512 MiB, recording peak allocated and
   reserved memory on every rank, throughput, TTFT and decode latency.
3. Keep request data and concurrency fixed while comparing HiCache off,
   write-through with fresh prefixes, and repeated host-cache hits.
4. Test long-context prefills, decode graph replay after prefill, and SM120
   separately. Absence of OOM in one run is not proof of deadlock freedom.

The original GPU failure has not been reproduced in the CPU-only development
environment. These tests do not verify CUDA kernel numerics or TP8 liveness.
