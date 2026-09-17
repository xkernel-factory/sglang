"""Exercise DSv4 host metadata dispatch with real CPU tensors and mocked planners."""

from __future__ import annotations

import ast
import runpy
import unittest
from enum import IntEnum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _load_definitions(path, names, namespace):
    # Load the actual host methods without importing CUDA/Triton dependencies.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


NS = dict(torch=torch, IntEnum=IntEnum, auto=auto)
_load_definitions(
    ROOT / "python/sglang/srt/model_executor/forward_batch_info.py",
    ["ForwardMode"],
    NS,
)
_load_definitions(
    ROOT / "python/sglang/srt/layers/attention/deepseek_v4_backend.py",
    ["_get_logical_forward_mode", "_get_target_verify_bs", "_build_forward_metadata"],
    NS,
)
NS["SWA_WINDOW"] = 128
ForwardMode = NS["ForwardMode"]
build_metadata = NS["_build_forward_metadata"]


class TestDSV4EmptyBatchMetadata(unittest.TestCase):
    def setUp(self):
        req_to_token = object()
        self.backend = SimpleNamespace(
            req_to_token=req_to_token,
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            swa_page_size=128,
            page_size=256,
            MAX_SEQ_LEN_FOR_CAPTURE=8192,
            online_c128_mtp=SimpleNamespace(prepare_forward=Mock(return_value=0)),
            topk=0,
            is_dspark_draft=False,
            init_forward_metadata_decode=Mock(),
            init_forward_metadata_prefill=Mock(),
        )

    def batch(self, mode, lengths):
        lengths = torch.tensor(lengths, dtype=torch.int64)
        return SimpleNamespace(
            forward_mode=mode,
            batch_size=len(lengths),
            req_pool_indices=torch.arange(len(lengths)),
            seq_lens=lengths,
            seq_lens_cpu=lengths.clone(),
            out_cache_loc=torch.empty(0, dtype=torch.int64),
            extend_seq_lens=lengths.clone(),
            extend_seq_lens_cpu=lengths.tolist(),
            extend_start_loc=torch.zeros(len(lengths), dtype=torch.int32),
        )

    def test_idle_dp_rank_still_prepares_sync_metadata(self):
        batch = self.batch(ForwardMode.IDLE, [])
        # A reused idle batch must not take a stale speculative verify path.
        batch._original_forward_mode = ForwardMode.TARGET_VERIFY
        result = build_metadata(self.backend, batch)
        planner = self.backend.init_forward_metadata_decode
        self.assertIs(result, planner.return_value)
        self.assertEqual(planner.call_args.kwargs["max_seq_len"], 0)
        self.assertEqual(planner.call_args.kwargs["seq_lens"].numel(), 0)
        self.backend.online_c128_mtp.prepare_forward.assert_called_once()
        self.assertEqual(
            self.backend.online_c128_mtp.prepare_forward.call_args.args[0],
            ForwardMode.IDLE,
        )
        self.backend.init_forward_metadata_prefill.assert_not_called()

    def test_empty_split_prefill(self):
        batch = self.batch(ForwardMode.SPLIT_PREFILL, [])
        result = build_metadata(self.backend, batch)
        planner = self.backend.init_forward_metadata_prefill
        self.assertIs(result, planner.return_value)
        self.assertEqual(planner.call_args.kwargs["max_seq_len"], 0)
        self.assertEqual(planner.call_args.kwargs["num_tokens"], 0)
        self.assertEqual(planner.call_args.kwargs["seq_lens_cpu"], [])

    def test_nonempty_prefill_uses_cpu_maximum(self):
        batch = self.batch(ForwardMode.SPLIT_PREFILL, [7, 19, 3])
        build_metadata(self.backend, batch)
        self.assertEqual(
            self.backend.init_forward_metadata_prefill.call_args.kwargs["max_seq_len"],
            19,
        )

    def test_missing_cpu_lengths_keep_capture_bound(self):
        batch = self.batch(ForwardMode.DECODE, [7])
        batch.seq_lens_cpu = None
        build_metadata(self.backend, batch)
        self.assertEqual(
            self.backend.init_forward_metadata_decode.call_args.kwargs["max_seq_len"],
            self.backend.MAX_SEQ_LEN_FOR_CAPTURE,
        )

    def test_overrides_take_precedence_for_empty_and_nonempty_batches(self):
        for lengths in ([], [7]):
            for explicit in (None, 0, 1024):
                with self.subTest(lengths=lengths, explicit=explicit):
                    batch = self.batch(ForwardMode.SPLIT_PREFILL, lengths)
                    batch.max_seq_len_override = 512
                    build_metadata(self.backend, batch, max_seq_len_override=explicit)
                    self.assertEqual(
                        self.backend.init_forward_metadata_prefill.call_args.kwargs[
                            "max_seq_len"
                        ],
                        512 if explicit is None else explicit,
                    )


if __name__ == "__main__":
    unittest.main()
