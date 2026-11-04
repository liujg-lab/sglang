"""CI entry for the lookahead template versus ``init_new`` check.

The comparison lives in the manual file. This wrapper selects the small CI
model and registers the CUDA suite.
"""

import importlib.util
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small")

_MANUAL = (
    Path(__file__).resolve().parents[2]
    / "manual"
    / "test_opt_lookahead_template_equiv.py"
)
_spec = importlib.util.spec_from_file_location(
    "opt_lookahead_template_equiv", _MANUAL
)
_manual = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_manual)


class TestOptLookaheadTemplateEquiv(_manual.OptLookaheadTemplateEquivTest):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("lookahead template check needs CUDA")
        from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        cls.model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        super().setUpClass()


if __name__ == "__main__":
    unittest.main()
