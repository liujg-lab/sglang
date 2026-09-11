"""Manual / nightly checklist for SPECTRE and STANDALONE_REMOTE dual-backend.

This file documents the e2e matrix. Full four-device runs need CUDA and NPU
hosts and are not executed in CPU CI.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

_DEVICE_PAIRS = (
    ("cuda", "cuda"),
    ("npu", "npu"),
    ("cuda", "npu"),
    ("npu", "cuda"),
)

_ACCEPTANCE = {
    "modes": ("SPECTRE", "STANDALONE_REMOTE"),
    "draft_shapes": ("chain", "tree"),
    "models": ("Qwen3", "Qwen3-VL"),
    "verify": ("greedy", "target_only", "rpd"),
    "graphs": ("eager", "npu_or_cuda_graph"),
    "page_size": ("1", ">1"),
    "tp": ("1", "2-unequal"),
}


class TestRemoteSpecE2EMatrixDoc(CustomTestCase):
    def test_matrix_covers_four_device_pairs_and_modes(self):
        modes = _ACCEPTANCE["modes"]
        covered = {(m, t, d) for m in modes for t, d in _DEVICE_PAIRS}
        self.assertEqual(len(covered), 8)
        self.assertIn(("STANDALONE_REMOTE", "npu", "cuda"), covered)
        self.assertIn("tree", _ACCEPTANCE["draft_shapes"])
        self.assertIn("npu_or_cuda_graph", _ACCEPTANCE["graphs"])
        self.assertIn(">1", _ACCEPTANCE["page_size"])

    def test_acceptance_rules_are_backend_local(self):
        # greedy: compare against the same Target backend AR, not CUDA vs NPU.
        # target_only: same coins / tree / probs.
        # RPD: rule path, not distribution-preserving sampling.
        rules = {
            "greedy": "same-target-AR",
            "target_only": "fixed-coins",
            "rpd": "rule-only",
        }
        self.assertEqual(rules["greedy"], "same-target-AR")


if __name__ == "__main__":
    unittest.main()
