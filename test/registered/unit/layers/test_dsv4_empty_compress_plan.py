"""Empty compressor plans must not compile or launch kernels on idle DP ranks.

Run directly with a CUDA PyTorch environment to also exercise device tensors.
Production Python definitions are loaded without the Linux JIT dependencies;
nonempty launches use doubles, while all tensor operations are real PyTorch.
"""

from __future__ import annotations

import ast
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _load(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


class TestEmptyCompressPlan(unittest.TestCase):
    def setUp(self):
        self.ns = dict(
            __name__=__name__, torch=torch, NamedTuple=NamedTuple, _is_xpu=False
        )
        _load(
            ROOT / "python/sglang/kernels/ops/attention/dsv4/compress.py",
            [
                "CompressorDecodePlan",
                "CompressorPrefillPlan",
                "compress_forward",
                "compress_norm_rope_store",
            ],
            self.ns,
        )
        _load(
            ROOT / "python/sglang/srt/layers/attention/dsv4/compressor_v2.py",
            ["create_paged_compressor_data", "_create_online_paged_compressor_data"],
            self.ns,
        )
        self.no_launch = Mock(
            side_effect=AssertionError("empty input reached GPU/JIT planner")
        )
        for name in (
            "_jit_compress_plan_module",
            "_jit_compress_128_online_module",
            "_jit_compress_module",
            "_jit_compress_norm_rope_module",
            "plan_compress_decode",
            "plan_compress_decode_legacy",
            "plan_compress_prefill",
            "plan_compress_prefill_legacy",
        ):
            self.ns[name] = self.no_launch
        self.decode = self.ns["CompressorDecodePlan"]
        self.prefill = self.ns["CompressorPrefillPlan"]
        self.devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            self.devices.append(torch.device("cuda", torch.cuda.current_device()))

    def inputs(self, device, rows=0):
        return dict(
            req_pool_indices=torch.zeros(rows, dtype=torch.int64, device=device),
            seq_lens=torch.ones(rows, dtype=torch.int64, device=device),
            req_to_token=torch.zeros((1, 256), dtype=torch.int64, device=device),
            full_to_state=torch.zeros(256, dtype=torch.int64, device=device),
            swa_page_size=128,
            ring_size=256,
        )

    def check_plan(self, plan, device, ratio, write_width=8):
        self.assertEqual(plan.compress_ratio, ratio)
        tensors = [plan.plan_d] if plan.is_decode else [plan.plan_c, plan.plan_w]
        widths = [16] if plan.is_decode else [16, write_width]
        for tensor, width in zip(tensors, widths):
            self.assertEqual(tensor.shape, (0, width))
            self.assertEqual(tensor.dtype, torch.uint8)
            self.assertEqual(tensor.device, device)
        self.no_launch.assert_not_called()

    def test_empty_decode_variants(self):
        for device in self.devices:
            args = self.inputs(device)
            for xpu in (False, True):
                self.ns["_is_xpu"] = xpu
                for ratio in (4, 128):
                    with self.subTest(device=device, xpu=xpu, ratio=ratio):
                        self.check_plan(
                            self.decode.generate(ratio, **args), device, ratio
                        )
                        self.check_plan(
                            self.decode.generate_legacy(
                                ratio, args["req_pool_indices"], args["seq_lens"]
                            ),
                            device,
                            ratio,
                        )
            self.check_plan(
                self.decode.generate_online(
                    args["seq_lens"], args["req_pool_indices"], args["req_to_token"]
                ),
                device,
                128,
            )

    def test_empty_prefill_variants(self):
        for device in self.devices:
            args = self.inputs(device)
            for xpu in (False, True):
                self.ns["_is_xpu"] = xpu
                for ratio in (4, 128):
                    for use_graph in (False, True):
                        with self.subTest(
                            device=device, xpu=xpu, ratio=ratio, use_graph=use_graph
                        ):
                            plan = self.prefill.generate(
                                ratio,
                                **args,
                                extend_lens=args["seq_lens"],
                                num_q_tokens=0,
                                use_cuda_graph=use_graph,
                            )
                            self.check_plan(plan, device, ratio)
                            legacy = self.prefill.generate_legacy(
                                ratio,
                                args["req_pool_indices"],
                                args["seq_lens"],
                                args["seq_lens"],
                                num_q_tokens=0,
                                device=device,
                                use_cuda_graph=use_graph,
                            )
                            self.check_plan(legacy, device, ratio)
            plan = self.prefill.generate_online(
                args["seq_lens"],
                args["seq_lens"],
                args["req_pool_indices"],
                args["req_to_token"],
                num_q_tokens=0,
            )
            self.check_plan(plan, device, 128, write_width=16)

    def test_backend_compressor_factory_with_empty_inputs(self):
        for device in self.devices:
            args = self.inputs(device)
            pool = SimpleNamespace(
                swa_page_size=128,
                get_ring_size=lambda **kwargs: 256,
                _unified_kv=True,
                full_to_swa_index_mapping=args["full_to_state"],
            )
            for online in (False, True):
                self.ns["_use_online_compress"] = lambda ratio: online and ratio == 128
                for prefill in (False, True):
                    for ratio in (4, 128):
                        with self.subTest(
                            device=device, online=online, prefill=prefill, ratio=ratio
                        ):
                            plan = self.ns["create_paged_compressor_data"](
                                ratio,
                                is_prefill=prefill,
                                token_to_kv_pool=pool,
                                req_to_token=args["req_to_token"],
                                req_pool_indices=args["req_pool_indices"],
                                seq_lens=args["seq_lens"],
                                extend_lens=args["seq_lens"],
                                seq_lens_cpu=[],
                                extend_lens_cpu=[],
                            )
                            width = 16 if online and ratio == 128 else 8
                            self.check_plan(plan, device, ratio, write_width=width)

    def test_empty_plan_consumers_do_not_launch_kernels(self):
        for device in self.devices:
            args = self.inputs(device)
            for ratio in (4, 128):
                plan = self.decode.generate(ratio, **args)
                empty = torch.empty((0, 512), device=device)
                for online in (False, True) if ratio == 128 else (False,):
                    output = self.ns["compress_forward"](
                        empty,
                        empty,
                        empty,
                        plan,
                        head_dim=512,
                        compress_ratio=ratio,
                        is_online=online,
                    )
                    self.assertEqual(output.shape, (0, 512))
                    self.assertEqual(output.device, device)
                self.ns["compress_norm_rope_store"](
                    empty,
                    plan,
                    norm_weight=None,
                    norm_eps=1e-6,
                    freq_cis=None,
                    out_loc=None,
                    kvcache=None,
                    page_size=256,
                )
                self.no_launch.assert_not_called()

    def test_nonempty_decode_still_dispatches_and_preserves_plan(self):
        args = self.inputs(torch.device("cpu"), rows=1)
        expected = torch.arange(16, dtype=torch.uint8).reshape(1, 16)
        module = SimpleNamespace(
            plan_decode=Mock(return_value=expected),
            plan_decode_legacy=Mock(return_value=expected),
        )
        self.ns["_jit_compress_plan_module"] = Mock(return_value=module)
        actual = self.decode.generate(4, **args, use_req_ring=True)
        torch.testing.assert_close(actual.plan_d, expected)
        self.assertEqual(actual.plan_d.data_ptr(), expected.data_ptr())
        module.plan_decode.assert_called_once_with(
            args["req_pool_indices"],
            args["req_to_token"],
            args["full_to_state"],
            args["seq_lens"],
            4,
            128,
            256,
            True,
        )
        legacy = self.decode.generate_legacy(
            4, args["req_pool_indices"], args["seq_lens"]
        )
        torch.testing.assert_close(legacy.plan_d, expected)
        module.plan_decode_legacy.assert_called_once()

    def test_nonempty_online_decode_still_dispatches(self):
        args = self.inputs(torch.device("cpu"), rows=1)

        def fill_plan(seq_lens, req_indices, req_to_token, plan, offset):
            plan.fill_(offset)

        module = SimpleNamespace(plan_decode=Mock(side_effect=fill_plan))
        self.ns["_jit_compress_128_online_module"] = Mock(return_value=module)
        plan = self.decode.generate_online(
            args["seq_lens"],
            args["req_pool_indices"],
            args["req_to_token"],
            state_slot_offset=3,
        )
        torch.testing.assert_close(
            plan.plan_d, torch.full((1, 16), 3, dtype=torch.uint8)
        )
        module.plan_decode.assert_called_once()


if __name__ == "__main__":
    unittest.main()
