"""CPU checks for OPT cascade attention, plus a CUDA numerical check when present."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.speculative.opt_tree_remote.opt_tree_attention import (
    OptTreeAttention,
    PrefixPlan,
    attention_layer_accepts,
    ensure_workspace,
    model_supports,
    prefix_split_count,
    split_range,
    stage2_live_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=30, suite="stage-a-test-cpu")

_BACKEND_SRC = (
    Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "triton_backend.py"
)


def _cpu_backend(context=64, heads=4, dim=8, draft_tokens=4):
    return SimpleNamespace(
        device="cpu",
        num_draft_tokens=draft_tokens,
        num_head=heads,
        v_head_dim=dim,
        max_context_len=context,
    )


def _layer(**overrides):
    values = dict(
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        sliding_window_size=-1,
        k_scale=None,
        qk_head_dim=64,
        v_head_dim=64,
        xai_temperature_len=-1,
        logit_cap=0.0,
        tp_q_head_num=8,
        tp_k_head_num=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class OptTreeCascadeTest(CustomTestCase):
    def test_prefix_splits_cover_the_prefix(self):
        for length in (1, 7, 31, 100, 4096):
            splits = prefix_split_count(length)
            self.assertGreater(splits, 0)
            self.assertLessEqual(splits, 31)
            covered = []
            for split in range(splits):
                start, end = split_range(split, length, splits)
                self.assertLess(start, end)
                covered.extend(range(start, end))
            self.assertEqual(covered, list(range(length)))

    def test_stage2_keeps_the_tree_slot(self):
        for prefix_splits in range(32):
            self.assertEqual(stage2_live_slots(prefix_splits), prefix_splits + 1)

    def test_prefix_plan_is_copied_once(self):
        backend = _cpu_backend()
        plan = PrefixPlan(torch.arange(6), 6)
        attn = OptTreeAttention(plan, active=True)
        tree = torch.eye(4, dtype=torch.uint8)
        indices = torch.arange(4)
        attn.note_layer(tree, indices)
        self.assertTrue(attn.prepare(backend, bs=1))
        self.assertEqual(plan.prefix_copies, 1)
        attn.note_layer(torch.ones(4, 4, dtype=torch.uint8), indices + 10)
        self.assertTrue(attn.prepare(backend, bs=1))
        self.assertEqual(plan.prefix_copies, 1)
        workspace = ensure_workspace(backend, 4)
        self.assertTrue(torch.equal(workspace.prefix_indices[:6], plan.indices))
        self.assertTrue(torch.equal(workspace.tree_indices[:4], indices + 10))

    def test_prepare_rejects_inactive_and_batched(self):
        plan = PrefixPlan(torch.arange(3), 3)
        inactive = OptTreeAttention(plan, active=False)
        self.assertFalse(inactive.prepare(None, bs=1))
        active = OptTreeAttention(plan, active=True)
        self.assertFalse(active.prepare(None, bs=2))

    def test_layer_gate(self):
        self.assertTrue(attention_layer_accepts(_layer()))
        self.assertFalse(attention_layer_accepts(_layer(sliding_window_size=128)))
        self.assertFalse(attention_layer_accepts(_layer(k_scale=torch.tensor(1.0))))
        self.assertFalse(attention_layer_accepts(_layer(qk_head_dim=64, v_head_dim=32)))
        self.assertFalse(attention_layer_accepts(_layer(logit_cap=30.0)))
        self.assertFalse(attention_layer_accepts(_layer(xai_temperature_len=1024)))

    def test_page_size_disables_model(self):
        runner = SimpleNamespace(
            server_args=SimpleNamespace(page_size=128),
            attn_backend=None,
            model=None,
        )
        self.assertFalse(model_supports(runner))

    def test_shared_backend_does_not_name_opt(self):
        text = _BACKEND_SRC.read_text()
        self.assertNotIn("opt_tree_remote", text)
        self.assertNotIn("is_opt_tree_remote", text)
        self.assertNotIn("is_standalone_remote", text)
        self.assertNotIn("is_spectre", text)
        self.assertIn("target_verify_attention", text)
        self.assertIn("def forward_extend", text)

    def test_spec_without_the_attribute_stays_on_extend(self):
        spec = SimpleNamespace(custom_mask=torch.ones(4, dtype=torch.bool))
        self.assertIsNone(getattr(spec, "target_verify_attention", None))


def _reference(q, k_cache, v_cache, prefix_idx, tree_idx, tree_mask, prefix_rows, scale):
    queries, heads, _ = q.shape
    kv_heads = k_cache.shape[1]
    group = heads // kv_heads
    out = torch.empty_like(q)
    for head in range(heads):
        kv_head = head // group
        key_prefix = k_cache[prefix_idx, kv_head].float()
        value_prefix = v_cache[prefix_idx, kv_head].float()
        key_tree = k_cache[tree_idx, kv_head].float()
        value_tree = v_cache[tree_idx, kv_head].float()
        prefix_scores = torch.matmul(q[:, head].float(), key_prefix.T) * scale
        tree_scores = torch.matmul(q[:, head].float(), key_tree.T) * scale
        tree_scores = tree_scores.masked_fill(~tree_mask, float("-inf"))
        if prefix_rows is not None:
            prefix_scores = prefix_scores.clone()
            prefix_scores[~prefix_rows] = float("-inf")
        scores = torch.cat((prefix_scores, tree_scores), dim=-1)
        weights = torch.softmax(scores, dim=-1)
        values = torch.cat((value_prefix, value_tree), dim=0)
        out[:, head] = torch.matmul(weights, values).to(dtype=q.dtype)
    return out


class OptTreeCascadeCudaTest(CustomTestCase):
    def test_matches_torch_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("cuda")
        try:
            import triton  # noqa: F401
        except Exception as exc:
            self.skipTest(f"triton: {exc}")
        from sglang.srt.speculative.opt_tree_remote.opt_tree_attention import (
            run_cascade_attention,
        )

        torch.manual_seed(7)
        device = "cuda"
        queries, kv_heads, group, dim = 5, 2, 2, 32
        heads = kv_heads * group
        prefix_len, tree_cols = 40, 9
        scale = dim**-0.5
        q = torch.randn(queries, heads, dim, device=device)
        tokens = prefix_len + tree_cols + 3
        k_cache = torch.randn(tokens, kv_heads, dim, device=device)
        v_cache = torch.randn(tokens, kv_heads, dim, device=device)
        prefix_idx = torch.arange(prefix_len, device=device)
        tree_idx = torch.arange(prefix_len, prefix_len + tree_cols, device=device)
        tree_mask = torch.zeros(queries, tree_cols, dtype=torch.bool, device=device)
        tree_mask[:, 0] = True
        tree_mask[torch.arange(queries, device=device), torch.arange(queries, device=device) % tree_cols] = True
        prefix_rows = torch.ones(queries, dtype=torch.bool, device=device)
        prefix_rows[-1] = False
        got = run_cascade_attention(
            q, k_cache, v_cache, prefix_idx, tree_idx, tree_mask, prefix_rows, scale
        )
        ref = _reference(
            q, k_cache, v_cache, prefix_idx, tree_idx, tree_mask, prefix_rows, scale
        )
        torch.testing.assert_close(got, ref, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
