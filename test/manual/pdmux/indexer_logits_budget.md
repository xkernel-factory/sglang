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

### Real SM90/SM100 FP8 kernel regression

From the repository root, on an idle NVIDIA GPU with the deployment's
SGLang/DeepGEMM environment (no model weights needed):

```bash
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
python test/manual/pdmux/verify_indexer_logits_budget_gpu.py \
  --output indexer-budget-gpu.json
```

The default 4 MiB budget deliberately forces small chunks. It is a TEST
setting, not a recommendation to change the server's 512 MiB default.
The script executes production row alignment and paged dispatch extracted
from `indexer.py`, with real metadata planners, DeepGEMM and TopK v1/v2.
It checks exact chunk boundaries, one-row tails, larger tails, padding across
a chunk boundary, cropping the tail, and cropping back to a single chunk.
Only valid logits columns are compared; TopK is checked against torch by
selected score multiset, allowing equivalent orderings/ties while rejecting
invalid/duplicate indices or incorrect counts. Tolerance is 1e-4 absolute
and relative. Any failure exits nonzero; success ends with `PASS`.

This also covers an eager-prefill fallback: if query row alignment changes
the rows of an existing chunked plan, rebuild the local DeepGEMM and TopK
plans together. The original shared metadata stays unchanged. Matching row
counts reuse the existing plan. Aligned tensors and plans are cached on the
source metadata by query row count and shared across indexer layers of the
same forward. A new metadata object starts with an empty cache; `copy_`
invalidates the destination cache without sharing the source cache. The
CPU regression simulates 43 layer calls and checks both reuse and invalidation.
Decode/graph paths are outside this fallback. The fused KV cache is passed as
`uint8` bytes, matching production and DeepGEMM's dtype contract.

For a larger kernel comparison with the deployment budget:

```bash
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
python test/manual/pdmux/verify_indexer_logits_budget_gpu.py \
  --rows 4097 --width 65536 --budget-mb 512 --repeat 20 \
  --output indexer-budget-gpu-512.json
```

`width` means compressed context length (65536 corresponds to 262144 original
tokens). The whole-batch reference needs about 1 GiB of logits in this example;
use an idle GPU. The script rejects reference sizes exceeding half the reported
free memory, but this is not a guarantee that all workspaces will fit.

JSON reports kernel time and peak extra allocated memory for whole vs chunked
execution. `kernel_time_ratio > 1` means chunking took longer for that case.
Plan wall time includes possible first-use JIT and is diagnostic only. Kernel
timing is warmed up; metadata preparation is excluded. This is not a full
model, TP8, HiCache, CUDA Graph or Green Context integration test. The script
disables TopK v2 cluster dispatch like PDMux, but does not create Green Contexts.

### Service throughput validation

More chunks mean more kernel launches and metadata plans. Smaller budgets can
reduce GPU work per launch and increase prefill latency; lower allocation
pressure may offset this under memory stress. No fixed throughput percentage
can be inferred from this kernel benchmark. Compare identical request data,
concurrency, prompt/output lengths, warmup and HiCache residency, with repeated
runs at 512 MiB and (if memory allows) 1024 MiB. Record tokens/s, TTFT p50/p95,
TPOT p50/p95, errors and peak memory on every rank. Keep the original failing
revision comparison separate and stop it if it OOMs; an OOM run is not a valid
steady-state throughput baseline.

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
