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
        split_prefill_batch=SimpleNamespace(split_index=0, extend_num_tokens=tokens),
    )


def _run_gloo_rank(rank, store_path, results):
    store = dist.FileStore(store_path, 2)
    group = dist.ProcessGroupGloo(store, rank, 2, timeout=timedelta(seconds=20))
    current = scheduler(
        2048 if rank == 0 else 0,
        decode_empty=rank == 1,
        group=group,
    )
    counts = []
    while current.split_prefill_batch.split_index < 61:
        count = get_count(current)
        counts.append(count)
        current.split_prefill_batch.split_index += count
    # Mirror the loop's collective when all layers have completed.
    done = torch.ones(1, dtype=torch.int32)
    group.allreduce(done, dist.ReduceOp.SUM).wait()
    results.put((rank, counts, done.item()))


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

    @unittest.skipUnless(dist.is_gloo_available(), "Gloo is unavailable")
    def test_two_process_gloo_keeps_completion_collective_aligned(self):
        ctx = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="pdmux-dp-split-") as tmp:
            results = ctx.Queue()
            processes = [
                ctx.Process(
                    target=_run_gloo_rank,
                    args=(rank, str(Path(tmp) / "store"), results),
                )
                for rank in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                actual = sorted(results.get(timeout=30) for _ in processes)
                self.assertEqual(actual, [(0, [32, 29], 2), (1, [32, 29], 2)])
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
