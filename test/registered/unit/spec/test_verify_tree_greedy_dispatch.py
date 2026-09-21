"""CPU tests for NPU greedy dispatch and contract inspection."""

from __future__ import annotations

import importlib.machinery
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

if "torchvision" not in sys.modules:
    _tv = types.ModuleType("torchvision")
    _tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", None)
    _tv_io = types.ModuleType("torchvision.io")
    _tv_io.__spec__ = importlib.machinery.ModuleSpec("torchvision.io", None)
    _tv_io.decode_jpeg = lambda *a, **k: None
    _tv.io = _tv_io
    sys.modules["torchvision"] = _tv
    sys.modules["torchvision.io"] = _tv_io
if "sgl_kernel" not in sys.modules:
    sys.modules["sgl_kernel"] = MagicMock()

from sglang.srt.speculative.eagle_utils import (
    greedy_verify_path_info,
    verify_tree_greedy_func,
)
from sglang.srt.speculative.tree_verify_npu import inspect_greedy_npu_contract
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")

_REPO = Path(__file__).resolve().parents[4]


def _cpu_case(bs=1, width=3, path_cap=2, extra=0):
    predicts = torch.full((bs * width + extra,), 777, dtype=torch.int32)
    accept_index = torch.full((bs, path_cap), -1, dtype=torch.int32)
    accept_token_num = torch.zeros((bs,), dtype=torch.int32)
    candidates = torch.arange(bs * width, dtype=torch.int64).reshape(bs, width)
    retrive_index = torch.arange(bs * width, dtype=torch.int64).reshape(bs, width)
    retrive_next_token = torch.full((bs, width), -1, dtype=torch.int64)
    retrive_next_sibling = torch.full((bs, width), -1, dtype=torch.int64)
    target_predict = torch.zeros((bs, width), dtype=torch.int64)
    if width > 1:
        retrive_next_token[:, 0] = 1
        retrive_next_sibling[:, 1] = 2 if width > 2 else -1
        candidates[:, 1] = 10
        if width > 2:
            candidates[:, 2] = 20
        target_predict[:, 0] = 20 if width > 2 else 10
    return dict(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )


class TestGreedyNpuContract(CustomTestCase):
    def test_cpu_tensors_are_cpu_decision(self):
        decision, reason = inspect_greedy_npu_contract(**_cpu_case())
        self.assertEqual(decision, "cpu")
        self.assertEqual(reason, "cpu tensors")

    def test_bs_zero_ok_after_meta(self):
        case = _cpu_case(bs=0, width=3, path_cap=2)
        decision, _reason = inspect_greedy_npu_contract(**case)
        self.assertEqual(decision, "cpu")

    def test_illegal_dtype_raises(self):
        case = _cpu_case()
        case["predicts"] = case["predicts"].float()
        with self.assertRaises(ValueError):
            inspect_greedy_npu_contract(**case)

    def test_illegal_shape_raises(self):
        case = _cpu_case()
        case["accept_index"] = torch.full((2, 2), -1, dtype=torch.int32)
        with self.assertRaises(ValueError):
            inspect_greedy_npu_contract(**case)

    def test_noncontiguous_npu_marked_unsupported(self):
        case = _cpu_case()
        for name, tensor in list(case.items()):
            fake = tensor
            fake = fake.as_strided(fake.shape, tuple(s * 1 for s in fake.stride()))
            case[name] = fake
        # CPU noncontiguous still reports cpu first (device type wins).
        decision, reason = inspect_greedy_npu_contract(**case)
        self.assertEqual(decision, "cpu")

    def test_wrapper_source_has_no_host_sync(self):
        src = (
            _REPO / "python/sglang/srt/speculative/tree_verify_npu.py"
        ).read_text()
        self.assertNotIn(".item(", src)
        self.assertNotIn(".cpu()", src)
        self.assertNotIn("synchronize(", src)


class TestGreedyDispatch(CustomTestCase):
    def setUp(self):
        import sglang.srt.speculative.eagle_utils as eu

        eu._logged_greedy_paths.clear()
        eu._npu_greedy_deps = None
        eu._last_greedy_verify_path = None
        eu._last_greedy_verify_reason = None

    def test_host_path_calls_reference_inplace(self):
        case = _cpu_case()
        predicts = case["predicts"]
        with patch("sglang.srt.speculative.eagle_utils._is_cuda", False), patch(
            "sglang.srt.speculative.eagle_utils._is_hip", False
        ), patch("sglang.srt.speculative.eagle_utils._is_npu", False), patch(
            "sglang.srt.speculative.tree_verify.verify_tree_greedy_ref",
            side_effect=lambda **kw: kw["predicts"],
        ) as ref:
            out_p, out_a, out_n = verify_tree_greedy_func(**case)
        self.assertIs(out_p, predicts)
        self.assertIs(out_a, case["accept_index"])
        self.assertIs(out_n, case["accept_token_num"])
        ref.assert_called_once()
        path, reason = greedy_verify_path_info()
        self.assertEqual(path, "cpu_reference")
        self.assertEqual(reason, "non-npu host path")

    def test_npu_cpu_tensors_fallback_reference(self):
        case = _cpu_case()
        with patch("sglang.srt.speculative.eagle_utils._is_cuda", False), patch(
            "sglang.srt.speculative.eagle_utils._is_hip", False
        ), patch("sglang.srt.speculative.eagle_utils._is_npu", True), patch(
            "sglang.srt.speculative.tree_verify.verify_tree_greedy_ref"
        ) as ref, patch(
            "sglang.srt.speculative.tree_verify_npu.verify_tree_greedy_npu"
        ) as kern:
            verify_tree_greedy_func(**case)
        ref.assert_called_once()
        kern.assert_not_called()
        path, reason = greedy_verify_path_info()
        self.assertEqual(path, "cpu_reference")
        self.assertEqual(reason, "cpu tensors")

    def test_npu_kernel_error_does_not_retry_reference(self):
        case = _cpu_case()
        with patch("sglang.srt.speculative.eagle_utils._is_cuda", False), patch(
            "sglang.srt.speculative.eagle_utils._is_hip", False
        ), patch("sglang.srt.speculative.eagle_utils._is_npu", True), patch(
            "sglang.srt.speculative.eagle_utils.npu_greedy_optional_deps",
            return_value=("ok", None),
        ), patch(
            "sglang.srt.speculative.tree_verify_npu.inspect_greedy_npu_contract",
            return_value=("ok", None),
        ), patch(
            "sglang.srt.speculative.tree_verify_npu.verify_tree_greedy_npu",
            side_effect=RuntimeError("launch failed"),
        ), patch(
            "sglang.srt.speculative.tree_verify.verify_tree_greedy_ref"
        ) as ref:
            with self.assertRaises(RuntimeError):
                verify_tree_greedy_func(**case)
        ref.assert_not_called()

    def test_missing_optional_dep_uses_reference(self):
        case = _cpu_case()
        with patch("sglang.srt.speculative.eagle_utils._is_cuda", False), patch(
            "sglang.srt.speculative.eagle_utils._is_hip", False
        ), patch("sglang.srt.speculative.eagle_utils._is_npu", True), patch(
            "sglang.srt.speculative.eagle_utils.npu_greedy_optional_deps",
            return_value=("missing", "triton not installed"),
        ), patch(
            "sglang.srt.speculative.tree_verify_npu.inspect_greedy_npu_contract",
            return_value=("ok", None),
        ), patch(
            "sglang.srt.speculative.tree_verify.verify_tree_greedy_ref"
        ) as ref, patch(
            "sglang.srt.speculative.tree_verify_npu.verify_tree_greedy_npu"
        ) as kern:
            verify_tree_greedy_func(**case)
        ref.assert_called_once()
        kern.assert_not_called()
        path, reason = greedy_verify_path_info()
        self.assertEqual(path, "cpu_reference")
        self.assertIn("triton", reason)

    def test_unsupported_layout_fallback_does_not_disable_deps(self):
        import sglang.srt.speculative.eagle_utils as eu

        case = _cpu_case()
        eu._npu_greedy_deps = ("ok", None)
        with patch("sglang.srt.speculative.eagle_utils._is_cuda", False), patch(
            "sglang.srt.speculative.eagle_utils._is_hip", False
        ), patch("sglang.srt.speculative.eagle_utils._is_npu", True), patch(
            "sglang.srt.speculative.tree_verify_npu.inspect_greedy_npu_contract",
            return_value=("unsupported_layout", "noncontiguous layout"),
        ), patch(
            "sglang.srt.speculative.tree_verify.verify_tree_greedy_ref"
        ) as ref:
            verify_tree_greedy_func(**case)
        ref.assert_called_once()
        self.assertEqual(eu._npu_greedy_deps, ("ok", None))
        path, reason = greedy_verify_path_info()
        self.assertEqual(path, "cpu_reference")
        self.assertEqual(reason, "noncontiguous layout")

    def test_npu_ok_calls_device_helper_inplace(self):
        case = _cpu_case()
        predicts = case["predicts"]

        def _fake_npu(**kwargs):
            kwargs["predicts"].fill_(3)
            return kwargs["predicts"], kwargs["accept_index"], kwargs["accept_token_num"]

        with patch("sglang.srt.speculative.eagle_utils._is_cuda", False), patch(
            "sglang.srt.speculative.eagle_utils._is_hip", False
        ), patch("sglang.srt.speculative.eagle_utils._is_npu", True), patch(
            "sglang.srt.speculative.eagle_utils.npu_greedy_optional_deps",
            return_value=("ok", None),
        ), patch(
            "sglang.srt.speculative.tree_verify_npu.inspect_greedy_npu_contract",
            return_value=("ok", None),
        ), patch(
            "sglang.srt.speculative.tree_verify_npu.verify_tree_greedy_npu",
            side_effect=_fake_npu,
        ) as kern, patch(
            "sglang.srt.speculative.tree_verify.verify_tree_greedy_ref"
        ) as ref:
            out_p, out_a, out_n = verify_tree_greedy_func(**case)
        kern.assert_called_once()
        ref.assert_not_called()
        self.assertIs(out_p, predicts)
        self.assertEqual(int(out_p[0]), 3)
        self.assertEqual(greedy_verify_path_info()[0], "npu_kernel")

    def test_source_does_not_import_sample_chain_kernel(self):
        src = (
            _REPO / "python/sglang/srt/speculative/eagle_utils.py"
        ).read_text()
        self.assertNotIn("sgl_kernel_npu", src)
        self.assertIn("tree_verify_npu", src)
        self.assertIn("verify_tree_greedy_ref", src)
        npu_src = (
            _REPO / "python/sglang/srt/speculative/tree_verify_npu.py"
        ).read_text()
        self.assertNotIn("sgl_kernel_npu.sample.verify_tree_greedy", npu_src)
        start = npu_src.index("_verify_tree_greedy_kernel[(bs,)]")
        launch = npu_src[
            start : npu_src.index(
                "return predicts, accept_index, accept_token_num", start
            )
        ]
        self.assertIn("PATH_CAP=path_cap", launch)
        self.assertNotIn("num_warps", launch)


if __name__ == "__main__":
    unittest.main()
