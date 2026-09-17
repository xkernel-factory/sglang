"""CPU regressions for rank-uniform PDMux split boundaries under uneven DP load."""

from __future__ import annotations

import ast
import multiprocessing
import runpy
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PATH = ROOT / "python/sglang/srt/multiplex/multiplexing_mixin.py"
TREE = ast.parse(PATH.read_text(encoding="utf-8"))
METHOD = next(
    node
    for node in ast.walk(TREE)
    if isinstance(node, ast.FunctionDef) and node.name == "_get_split_forward_count"
)
NS = dict(torch=torch, dist=dist)
exec(compile(ast.Module(body=[METHOD], type_ignores=[]), str(PATH), "exec"), NS)
get_count = NS["_get_split_forward_count"]


class _Collective:
    """Two-rank CPU collective double; completion requires both ranks to wait."""

    def __init__(self, size):
        self.barrier = threading.Barrier(size, timeout=5)
        self.values = [None] * size
        self.calls = [0] * size

    def group(self, rank):
        def allreduce(tensor, op):
            assert tensor.device.type == "cpu"
            assert op == dist.ReduceOp.MIN
            self.values[rank] = tensor
            self.calls[rank] += 1

            def wait():
                self.barrier.wait()
                tensor.fill_(min(value.item() for value in self.values))
                self.barrier.wait()

            return SimpleNamespace(wait=wait)

        return SimpleNamespace(allreduce=allreduce)


def scheduler(tokens, *, decode_empty=False, dp_size=2, group=None):
    return SimpleNamespace(
        ps=SimpleNamespace(attn_dp_size=dp_size),
        tp_cpu_group=group,
        model_config=SimpleNamespace(num_hidden_layers=61),
        pdmux_config=SimpleNamespace(split_forward_token_budget=65536),
        running_batch=SimpleNamespace(is_empty=lambda: decode_empty),
        split_prefill_batch=SimpleNamespace(
            split_index=0, extend_num_tokens=tokens, global_num_tokens=None
        ),
    )


def _run_gloo_rank(rank, store_path, results, gathered_counts=False):
    store = dist.FileStore(store_path, 2)
    group = dist.ProcessGroupGloo(store, rank, 2, timeout=timedelta(seconds=20))
    split_sync_calls = 0

    def allreduce(*args):
        nonlocal split_sync_calls
        split_sync_calls += 1
        return group.allreduce(*args)

    current = scheduler(
        2048 if rank == 0 else 0,
        decode_empty=rank == 1,
        group=SimpleNamespace(allreduce=allreduce),
    )
    decode_batch = None
    if gathered_counts:
        current.split_prefill_batch.global_num_tokens = [2048, 0]
        decode_batch = SimpleNamespace(global_num_tokens=[1, 0])
    counts = []
    while current.split_prefill_batch.split_index < 61:
        count = get_count(current, decode_batch)
        counts.append(count)
        current.split_prefill_batch.split_index += count
    # Mirror the loop's collective when all layers have completed.
    done = torch.ones(1, dtype=torch.int32)
    group.allreduce(done, dist.ReduceOp.SUM).wait()
    results.put((rank, counts, done.item(), split_sync_calls))


class TestPDMuxDPLayerSplit(unittest.TestCase):
    def test_uneven_dp_ranks_finish_each_segment_together(self):
        for tokens, empty, expected in (
            ((2048, 0), (False, True), [32, 29]),
            ((16384, 2048), (False, False), [4] * 15 + [1]),
            ((2048, 2048), (False, True), [32, 29]),
            ((2048, 0), (True, True), [61]),
        ):
            with self.subTest(tokens=tokens, decode_empty=empty):
                collective = _Collective(2)
                schedulers = [
                    scheduler(
                        tokens[i], decode_empty=empty[i], group=collective.group(i)
                    )
                    for i in range(2)
                ]

                def run(rank):
                    current = schedulers[rank]
                    counts = []
                    while current.split_prefill_batch.split_index < 61:
                        count = get_count(current)
                        counts.append(count)
                        current.split_prefill_batch.split_index += count
                    return counts

                with ThreadPoolExecutor(max_workers=2) as executor:
                    counts = list(executor.map(run, range(2)))
                self.assertEqual(counts, [expected, expected])
                self.assertEqual(collective.calls, [len(expected)] * 2)

    def test_non_dp_keeps_local_policy_without_collectives(self):
        group = Mock()
        self.assertEqual(get_count(scheduler(2048, dp_size=1, group=group)), 32)
        self.assertEqual(
            get_count(scheduler(2048, decode_empty=True, dp_size=1, group=group)), 61
        )
        group.allreduce.assert_not_called()

    def test_gathered_counts_bound_global_work_without_another_collective(self):
        for prefill, decode in (
            ([2048, 0], [1, 0]),
            ([16384, 2048], [1, 1]),
            ([2048, 2048], [1, 0]),
            ([2048, 0], [0, 0]),
            ([2048, 0], [0, 1]),
            ([0, 2048, 256, 1923, 2048, 0, 1, 1024], [1, 2, 0, 1, 0, 0, 1, 2]),
        ):
            for budget in (8192, 16384, 65536):
                for split_index in (0, 32, 60):
                    with self.subTest(
                        prefill=prefill, decode=decode, budget=budget, index=split_index
                    ):
                        remaining = 61 - split_index
                        expected = (
                            min(remaining, max(1, budget // sum(prefill)))
                            if sum(prefill) and any(decode)
                            else remaining
                        )
                        group = Mock(side_effect=AssertionError("redundant collective"))
                        for rank in range(len(prefill)):
                            current = scheduler(
                                prefill[rank],
                                decode_empty=decode[rank] == 0,
                                dp_size=len(prefill),
                                group=group,
                            )
                            current.pdmux_config.split_forward_token_budget = budget
                            current.split_prefill_batch.split_index = split_index
                            current.split_prefill_batch.global_num_tokens = prefill
                            self.assertEqual(
                                get_count(
                                    current, SimpleNamespace(global_num_tokens=decode)
                                ),
                                expected,
                            )
                        group.allreduce.assert_not_called()

    def test_dp8_preserves_tp_layer_granularity_with_full_chunks(self):
        group = Mock()
        for rank in range(8):
            current = scheduler(2048, dp_size=8, group=group)
            current.model_config.num_hidden_layers = 43
            current.split_prefill_batch.global_num_tokens = [2048] * 8
            decode = SimpleNamespace(global_num_tokens=[1] * 8)
            segments = []
            while current.split_prefill_batch.split_index < 43:
                count = get_count(current, decode)
                segments.append(count)
                current.split_prefill_batch.split_index += count
            self.assertEqual(segments, [4] * 10 + [3])
        group.allreduce.assert_not_called()

    def test_sparse_dp_load_uses_actual_tokens_instead_of_dp_multiplier(self):
        current = scheduler(2048, dp_size=8, group=Mock())
        current.split_prefill_batch.global_num_tokens = [2048] + [0] * 7
        decode = SimpleNamespace(global_num_tokens=[0, 1] + [0] * 6)
        self.assertEqual(get_count(current, decode), 32)
        current.tp_cpu_group.allreduce.assert_not_called()

    def test_local_only_metadata_keeps_collective_fallback(self):
        collective = _Collective(2)

        def run(rank):
            current = scheduler(
                2048 if rank == 0 else 0,
                decode_empty=rank == 1,
                group=collective.group(rank),
            )
            # A2A configurations can publish only local counts; these are not
            # rank-invariant and must never select the collective-free path.
            current.split_prefill_batch.global_num_tokens = [2048 if rank == 0 else 0]
            return get_count(
                current, SimpleNamespace(global_num_tokens=[1 if rank == 0 else 0])
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            self.assertEqual(list(executor.map(run, range(2))), [32, 32])
        self.assertEqual(collective.calls, [1, 1])

    @unittest.skipUnless(dist.is_gloo_available(), "Gloo is unavailable")
    def test_two_process_gloo_keeps_completion_collective_aligned(self):
        self._check_gloo_completion(gathered_counts=False)

    @unittest.skipUnless(dist.is_gloo_available(), "Gloo is unavailable")
    def test_two_process_gloo_gathered_counts_skip_split_collectives(self):
        self._check_gloo_completion(gathered_counts=True)

    def _check_gloo_completion(self, gathered_counts):
        ctx = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="pdmux-dp-split-") as tmp:
            results = ctx.Queue()
            processes = [
                ctx.Process(
                    target=_run_gloo_rank,
                    args=(rank, str(Path(tmp) / "store"), results, gathered_counts),
                )
                for rank in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                actual = sorted(results.get(timeout=30) for _ in processes)
                expected_sync_calls = 0 if gathered_counts else 2
                self.assertEqual(
                    actual,
                    [
                        (0, [32, 29], 2, expected_sync_calls),
                        (1, [32, 29], 2, expected_sync_calls),
                    ],
                )
                for process in processes:
                    process.join(timeout=5)
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                results.close()


if __name__ == "__main__":
    unittest.main()
