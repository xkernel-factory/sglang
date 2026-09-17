"""CPU-only budget/metadata tests; kernel planners are replaced with test doubles."""

import ast
import importlib.util
import os
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations against the defining module.
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


budget = load(
    "_dsv4_logits_budget_test",
    "python/sglang/kernels/ops/attention/dsv4/logits_budget.py",
)
load(
    "_dsv4_logits_budget_ci",
    "python/sglang/test/ci/ci_register.py",
).register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Tensor:
    """Minimal CPU tensor for testing host-side planning, not kernel numerics."""

    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    def numel(self):
        return self.value.size

    def dim(self):
        return self.value.ndim

    def to(self, dtype):
        return Tensor(self.value.astype(dtype))

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.value, axis))

    def __getitem__(self, key):
        return Tensor(self.value[key])

    def copy_(self, other):
        assert self.shape == other.shape
        self.value[...] = other.value


class TestIndexerLogitsBudget(unittest.TestCase):
    def test_defaults_and_legacy_precedence(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(budget.indexer_logits_budget_bytes(), 512 * 2**20)
            self.assertEqual(
                budget.indexer_logits_budget_bytes(legacy_fp4=True), 2048 * 2**20
            )
            os.environ["SGLANG_DSV4_FP4_LOGITS_BUDGET_MB"] = "128"
            self.assertEqual(
                budget.indexer_logits_budget_bytes(legacy_fp4=True), 128 * 2**20
            )
            self.assertEqual(budget.indexer_logits_budget_bytes(), 512 * 2**20)
            os.environ["SGLANG_DSV4_INDEXER_LOGITS_BUDGET_MB"] = "256"
            self.assertEqual(
                budget.indexer_logits_budget_bytes(legacy_fp4=True), 256 * 2**20
            )

    def test_invalid_budgets(self):
        for value in ("0", "-1", "bad", "1.5"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"SGLANG_DSV4_INDEXER_LOGITS_BUDGET_MB": value}
            ):
                with self.assertRaises(ValueError):
                    budget.indexer_logits_budget_bytes()

    def test_alignment_long_context_and_power_of_two(self):
        limit = 512 * 2**20
        for width in (0, 64, 256, 257, 65536, 262144, 262145):
            with self.subTest(width=width):
                rows = budget.paged_logits_rows_per_chunk(width, limit)
                aligned = max(256, (width + 255) // 256 * 256)
                self.assertEqual(rows & (rows - 1), 0)
                self.assertLessEqual(rows * aligned * 4, limit)
                self.assertGreater(rows * 2 * aligned * 4, limit)
        self.assertEqual(budget.paged_logits_rows_per_chunk(262144, limit), 512)
        with self.assertRaises(ValueError):
            budget.paged_logits_rows_per_chunk(262144, 1024)


class TestChunkMetadata(unittest.TestCase):
    def setUp(self):
        self.dg_calls = []
        self.topk_calls = []

        def dg_plan(lens, page_size, sms):
            self.dg_calls.append(lens.value.copy())
            return Tensor(np.full((sms + 1, 2), int(lens.value.sum())))

        def topk_plan(lens):
            self.topk_calls.append(lens.value.copy())
            return Tensor(np.full((lens.shape[0] + 1, 2), int(lens.value.sum())))

        def pad(tensor, padding, value):
            widths = list(zip(padding[::2], padding[1::2]))[::-1]
            return Tensor(np.pad(tensor.value, widths, constant_values=value))

        envs = types.SimpleNamespace(
            SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=types.SimpleNamespace(get=lambda: False),
            SGLANG_OPT_USE_JIT_INDEXER_METADATA=types.SimpleNamespace(
                get=lambda: False
            ),
        )
        stubs = {
            "torch.nn.functional": types.SimpleNamespace(pad=pad),
            "torch": types.SimpleNamespace(
                Tensor=Tensor,
                int32=np.int32,
                empty=lambda shape: Tensor(np.empty(shape)),
            ),
            "sglang.srt.environ": types.SimpleNamespace(envs=envs),
            "sglang.srt.utils": types.SimpleNamespace(
                is_hip=lambda: False,
                is_xpu=lambda: False,
                is_sm120_supported=lambda: False,
            ),
            "deep_gemm": types.SimpleNamespace(
                get_num_sms=lambda: 8, get_paged_mqa_logits_metadata=dg_plan
            ),
            "sglang.kernels.ops.attention.dsv4": types.SimpleNamespace(
                get_paged_mqa_logits_metadata=dg_plan, plan_topk_v2=topk_plan
            ),
            "sglang.kernels.ops.attention.dsv4.logits_budget": budget,
        }
        self.modules = patch.dict(sys.modules, stubs)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.env = patch.dict(os.environ, {"SGLANG_DSV4_INDEXER_LOGITS_BUDGET_MB": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.metadata = load(
            "_dsv4_chunk_metadata_test",
            "python/sglang/srt/layers/attention/dsv4/metadata.py",
        )

    def make(self, rows=2050, *, prefill=True, graph=False, value=1):
        return self.metadata.PagedIndexerMetadata(
            page_size=256,
            compressed_page_size=64,
            page_table=Tensor(np.zeros((rows, 4), dtype=np.int32)),
            compressed_seq_lens=Tensor(np.full(rows, value, dtype=np.int32)),
            use_topk_v2=True,
            is_prefill=prefill,
            use_prefill_cuda_graph=graph,
        )

    def test_tail_and_matching_topk_plans(self):
        m = self.make()
        self.assertEqual(m.logits_chunk_rows, 1024)
        self.assertEqual([x.shape[0] for x in self.dg_calls], [1024, 1024, 2])
        self.assertEqual([x.shape[0] for x in m.chunk_topk_metadata], [1025, 1025, 3])
        np.testing.assert_array_equal(
            np.concatenate(self.dg_calls).reshape(-1), m.compressed_seq_lens.value
        )
        for dg, tk in zip(self.dg_calls, self.topk_calls[1:]):
            np.testing.assert_array_equal(dg.reshape(-1), tk)

    def test_decode_and_prefill_graph_remain_unchunked(self):
        for prefill, graph in ((False, False), (False, True), (True, True)):
            with self.subTest(prefill=prefill, graph=graph):
                m = self.make(prefill=prefill, graph=graph)
                self.assertIsInstance(m.deep_gemm_metadata, Tensor)
                self.assertIsNone(m.chunk_topk_metadata)
                self.assertEqual(m.logits_chunk_rows, 0)

    def test_empty_small_and_exact_budget_boundary(self):
        for rows in (0, 1, 1024):
            with self.subTest(rows=rows):
                m = self.make(rows=rows)
                self.assertIsInstance(m.deep_gemm_metadata, Tensor)
                self.assertIsNone(m.chunk_topk_metadata)
        m = self.make(rows=1025)
        self.assertEqual(m.logits_chunk_rows, 1024)
        self.assertEqual([x.shape[0] for x in m.chunk_topk_metadata], [1025, 2])

    def test_sm120_respects_budget_below_hardware_cap(self):
        self.metadata._IS_SM120 = True
        m = self.make(rows=9000)
        self.assertEqual(m.logits_chunk_rows, 1024)
        self.assertEqual(sum(x.shape[0] for x in self.dg_calls), 9000)

    def test_sm120_keeps_hard_row_cap(self):
        self.metadata._IS_SM120 = True
        m = self.make(rows=9000, prefill=False)
        self.assertEqual(m.logits_chunk_rows, 4096)
        self.assertEqual([x.shape[0] for x in self.dg_calls], [4096, 4096, 808])

    def test_copy_preserves_plan_addresses_and_updates_contents(self):
        dst, src = self.make(value=1), self.make(value=2)
        dg_ids = [id(x) for x in dst.deep_gemm_metadata]
        tk_ids = [id(x) for x in dst.chunk_topk_metadata]
        dst.copy_(src)
        self.assertEqual(dg_ids, [id(x) for x in dst.deep_gemm_metadata])
        self.assertEqual(tk_ids, [id(x) for x in dst.chunk_topk_metadata])
        for a, b in zip(dst.chunk_topk_metadata, src.chunk_topk_metadata):
            np.testing.assert_array_equal(a.value, b.value)
        for a, b in zip(dst.deep_gemm_metadata, src.deep_gemm_metadata):
            np.testing.assert_array_equal(a.value, b.value)

    def test_runtime_padding_and_cropping_rebuild_chunk_plans(self):
        validator = load(
            "_dsv4_gpu_validator",
            "test/manual/pdmux/verify_indexer_logits_budget_gpu.py",
        )
        prepare, _ = validator.production_dispatch()

        def pad(tensor, padding, value):
            widths = list(zip(padding[::2], padding[1::2]))[::-1]
            return Tensor(np.pad(tensor.value, widths, constant_values=value))

        for target_rows in (2050, 3073, 1025, 1024):
            with self.subTest(target_rows=target_rows):
                original = self.make(value=2)
                ns = dict(
                    torch=types.SimpleNamespace(Tensor=Tensor),
                    F=types.SimpleNamespace(pad=pad),
                    replace=replace,
                    query_rows=target_rows,
                    indexer_metadata=original,
                    use_aiter_fp4=False,
                )
                exec(prepare, ns)
                actual = ns["indexer_metadata"]
                plan_count = len(self.dg_calls), len(self.topk_calls)
                for _ in range(42):
                    ns["indexer_metadata"] = original
                    exec(prepare, ns)
                    self.assertIs(ns["indexer_metadata"], actual)
                self.assertEqual(plan_count, (len(self.dg_calls), len(self.topk_calls)))
                self.assertEqual(actual.compressed_seq_lens.shape[0], target_rows)
                self.assertEqual(actual.page_table.shape[0], target_rows)
                self.assertEqual(original.compressed_seq_lens.shape[0], 2050)
                if target_rows == 2050:
                    self.assertIs(actual, original)
                else:
                    self.assertIsNot(actual, original)
                if target_rows > 2050:
                    np.testing.assert_array_equal(
                        actual.compressed_seq_lens.value[2050:], 1
                    )
                    np.testing.assert_array_equal(actual.page_table.value[2050:], 0)
                if target_rows <= 1024:
                    self.assertEqual(actual.logits_chunk_rows, 0)
                    self.assertIsNone(actual.chunk_topk_metadata)
                else:
                    expected = [
                        min(1024, target_rows - start)
                        for start in range(0, target_rows, 1024)
                    ]
                    self.assertEqual(
                        [p.shape[0] - 1 for p in actual.chunk_topk_metadata], expected
                    )
                    for idx, start in enumerate(range(0, target_rows, 1024)):
                        length_sum = int(
                            actual.compressed_seq_lens.value[start : start + 1024].sum()
                        )
                        self.assertEqual(
                            int(actual.deep_gemm_metadata[idx].value[0, 0]), length_sum
                        )
                        self.assertEqual(
                            int(actual.chunk_topk_metadata[idx].value[0, 0]), length_sum
                        )

    def test_alignment_cache_is_per_forward_and_per_row_count(self):
        original = self.make(value=2)
        first = original.for_query_rows(3073)
        second = original.for_query_rows(1025)
        count = len(self.dg_calls), len(self.topk_calls)
        self.assertIs(original.for_query_rows(3073), first)
        self.assertIs(original.for_query_rows(1025), second)
        self.assertEqual(count, (len(self.dg_calls), len(self.topk_calls)))
        self.assertIsNot(first, second)
        fresh = self.make(value=3)
        fresh_aligned = fresh.for_query_rows(3073)
        self.assertIsNot(fresh_aligned, first)
        original.copy_(fresh)
        self.assertEqual(original._aligned_query_cache, {})
        self.assertIs(fresh.for_query_rows(3073), fresh_aligned)
        rebuilt = original.for_query_rows(3073)
        self.assertIsNot(rebuilt, first)
        self.assertIsNot(rebuilt, fresh_aligned)
        np.testing.assert_array_equal(rebuilt.compressed_seq_lens.value[:2050], 3)
        np.testing.assert_array_equal(rebuilt.compressed_seq_lens.value[2050:], 1)

    def test_alignment_leaves_graph_and_decode_metadata_unchanged(self):
        self.metadata._IS_SM120 = True
        for prefill, graph in ((False, False), (True, True)):
            m = self.make(rows=5000, prefill=prefill, graph=graph)
            count = len(self.dg_calls), len(self.topk_calls)
            self.assertIs(m.for_query_rows(4097), m)
            self.assertEqual(m._aligned_query_cache, {})
            self.assertEqual(count, (len(self.dg_calls), len(self.topk_calls)))

    def test_indexer_dispatch_matches_whole_topk_and_reuses_plans(self):
        # Execute the production dispatch with deterministic CPU scoring and
        # topk doubles. This checks row offsets, the tail and both metadata
        # lists together without importing CUDA dependencies on CPU CI.
        path = ROOT / "python/sglang/srt/layers/attention/dsv4/indexer.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        topk = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_topk_transform"
        )
        branch = next(
            n.orelse
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and any(
                isinstance(child, ast.FunctionDef) and child.name == "run_paged_indexer"
                for child in n.orelse
            )
        )
        first = next(i for i, n in enumerate(branch) if isinstance(n, ast.FunctionDef))
        code = ast.Module(body=[topk, *branch[first:]], type_ignores=[])
        compiled = compile(ast.fix_missing_locations(code), str(path), "exec")
        m = self.make()
        rows = m.compressed_seq_lens.shape[0]
        output = Tensor(np.zeros((rows, 4), dtype=np.int64))
        seen = []

        def scores(row_q, cache, weights, lens, pages, plan, width, clean):
            seen.append(row_q.shape[0])
            self.assertEqual(int(plan.value[0, 0]), int(lens.value.sum()))
            values = (row_q.value.reshape(-1, 1) * 13 + np.arange(17) * 7) % 257
            return Tensor(values)

        def reduce(scores, lens, pages, out, page_size, plan, enable_cluster):
            self.assertEqual(plan.shape[0], scores.shape[0] + 1)
            self.assertEqual(int(plan.value[0, 0]), int(lens.value.sum()))
            out.value[...] = np.argsort(-scores.value, axis=1)[:, :4]

        ns = dict(
            torch=types.SimpleNamespace(Tensor=Tensor),
            self=types.SimpleNamespace(
                dsa_topk_backend=types.SimpleNamespace(
                    is_torch=lambda: False,
                    is_flashinfer=lambda: False,
                    should_use_topk_v2=lambda: True,
                )
            ),
            q=Tensor(np.arange(rows)),
            weights=Tensor(np.ones(rows)),
            c4_indexer_kv_cache=None,
            _c4sl=m.compressed_seq_lens.unsqueeze(-1),
            c4_seq_lens=m.compressed_seq_lens,
            page_table=m.page_table,
            c4_sparse_page_indices=output,
            raw_indices=None,
            indexer_metadata=m,
            all_rows=slice(0, rows),
            fn=scores,
            topk_transform_paged_v2=reduce,
            is_pdmux_enabled=lambda: True,
        )
        plan_count = len(self.dg_calls), len(self.topk_calls)
        for _ in range(2):
            exec(compiled, ns)
        self.assertEqual(seen, [1024, 1024, 2] * 2)
        expected = (np.arange(rows)[:, None] * 13 + np.arange(17) * 7) % 257
        np.testing.assert_array_equal(
            output.value, np.argsort(-expected, axis=1)[:, :4]
        )
        self.assertEqual(plan_count, (len(self.dg_calls), len(self.topk_calls)))


if __name__ == "__main__":
    unittest.main()
