"""CPU checks for the CLEAR_TREE_REMOTE shell.

The shell is a renamed copy of fixed-width OPT. It must not be recognized as
OPT or STANDALONE_REMOTE, and OPT must not pick up ClearTree fields.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.speculative.clear_tree_remote.clear_tree_args import (
    apply_clear_tree_settings,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm, remote_spec_role
from sglang.srt.speculative.standalone_remote.sr_protocol import local_spec_shape
from sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept import (
    static_disable_reason,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _args(**overrides):
    values = dict(
        speculative_algorithm="CLEAR_TREE_REMOTE",
        clear_tree_role="target",
        clear_tree_width=4,
        clear_tree_tau=0.5,
        clear_tree_max_depth=8,
        clear_tree_addr="127.0.0.1",
        clear_tree_port="30039",
        opt_tree_role=None,
        opt_tree_width=None,
        opt_tree_max_depth=None,
        standalone_remote_role=None,
        speculative_num_draft_tokens=None,
        speculative_num_steps=None,
        speculative_verify_mode=None,
        speculative_eagle_topk=None,
        max_running_requests=1,
        page_size=1,
        disable_overlap_schedule=False,
        enable_mixed_chunk=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class ClearTreeRemoteShellTest(CustomTestCase):
    def test_algorithm_is_distinct(self):
        clear = SpeculativeAlgorithm.from_string("CLEAR_TREE_REMOTE")
        opt = SpeculativeAlgorithm.from_string("OPT_TREE_REMOTE")
        sr = SpeculativeAlgorithm.from_string("STANDALONE_REMOTE")

        self.assertTrue(clear.is_clear_tree_remote())
        self.assertFalse(clear.is_opt_tree_remote())
        self.assertFalse(clear.is_standalone_remote())
        self.assertTrue(opt.is_opt_tree_remote())
        self.assertFalse(opt.is_clear_tree_remote())
        self.assertFalse(sr.is_clear_tree_remote())

    def test_role_and_draft_hooks(self):
        clear = SpeculativeAlgorithm.from_string("CLEAR_TREE_REMOTE")
        draft = _args(clear_tree_role="draft")
        target = _args(clear_tree_role="target")

        self.assertTrue(clear.skips_colocated_draft_worker(draft))
        self.assertFalse(clear.skips_colocated_draft_worker(target))
        self.assertTrue(clear.uses_draft_paused_reqs(draft))
        self.assertTrue(clear.uses_ordinary_decode_prepare(draft))
        self.assertFalse(clear.uses_draft_paused_reqs(target))
        self.assertEqual(clear.guess_tokens_per_step(draft), 4)
        self.assertTrue(clear.reports_remote_spec_metrics())
        self.assertEqual(remote_spec_role(draft), "draft")

        sr_wins = _args(standalone_remote_role="target", clear_tree_role="draft")
        self.assertEqual(remote_spec_role(sr_wins), "target")
        opt_wins = _args(opt_tree_role="draft", clear_tree_role="target")
        self.assertEqual(remote_spec_role(opt_wins), "draft")

    def test_omitted_draft_tokens_follow_role(self):
        target = _args(clear_tree_role="target")
        apply_clear_tree_settings(target, on_npu=False)
        self.assertEqual(target.speculative_num_draft_tokens, 5)
        self.assertEqual(target.speculative_num_steps, 8)
        self.assertTrue(target.disable_overlap_schedule)
        self.assertFalse(target.enable_mixed_chunk)

        draft = _args(clear_tree_role="draft")
        apply_clear_tree_settings(draft, on_npu=False)
        self.assertEqual(draft.speculative_num_draft_tokens, 4)

    def test_explicit_draft_tokens_must_match(self):
        draft = _args(clear_tree_role="draft", speculative_num_draft_tokens=5)
        with self.assertRaises(ValueError):
            apply_clear_tree_settings(draft, on_npu=False)

    def test_wire_shape_uses_clear_fields(self):
        clear = _args(
            speculative_num_draft_tokens=99,
            speculative_num_steps=2,
            opt_tree_width=7,
            opt_tree_max_depth=3,
        )
        self.assertEqual(local_spec_shape(clear), (4, 8, 4))

        opt = _args(
            speculative_algorithm="OPT_TREE_REMOTE",
            opt_tree_width=7,
            opt_tree_max_depth=3,
            clear_tree_width=4,
            clear_tree_max_depth=8,
        )
        self.assertEqual(local_spec_shape(opt), (7, 3, 7))

    def test_fixed_accept_stays_off(self):
        clear = SimpleNamespace(speculative_algorithm="CLEAR_TREE_REMOTE")
        opt = SimpleNamespace(speculative_algorithm="OPT_TREE_REMOTE")
        self.assertEqual(static_disable_reason(clear), "clear_tree")
        self.assertEqual(static_disable_reason(opt), "opt_tree")


if __name__ == "__main__":
    unittest.main()
