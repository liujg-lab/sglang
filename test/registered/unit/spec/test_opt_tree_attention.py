"""CPU checks for OPT cascade attention, plus a CUDA numerical check when present."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.speculative.opt_tree_remote.opt_tree_attention import (
    OptTreeAttention,
    PrefixPlan,
    _captured_cascade,
    attention_layer_accepts,
    bind_opt_cascade,
    cascade_query_block,
    cdiv,
    ensure_workspace,
    tree_limit,
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
    def setUp(self):
        super().setUp()
        _captured_cascade.clear()

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

    def test_layer_slots_append_without_rewriting_prefix_metadata(self):
        backend = _cpu_backend()
        plan = PrefixPlan(torch.arange(6), 6)
        attn = OptTreeAttention(plan, active=True)
        first = torch.arange(2)
        second = torch.arange(2) + 10
        attn.note_layer(torch.eye(2, dtype=torch.uint8), first)
        self.assertTrue(attn.prepare(backend, bs=1))
        workspace = ensure_workspace(backend, 2)
        meta_writes = workspace.prefix_meta_writes
        stage2_writes = workspace.stage2_writes
        attn.note_layer(torch.ones(2, 4, dtype=torch.uint8), second)
        self.assertTrue(attn.prepare(backend, bs=1))
        self.assertEqual(workspace.prefix_meta_writes, meta_writes)
        self.assertEqual(workspace.stage2_writes, stage2_writes)
        self.assertEqual(plan.prefix_copies, 1)
        self.assertTrue(
            torch.equal(workspace.tree_indices[:4], torch.cat((first, second)))
        )
        attn.note_layer(torch.ones(3, 4, dtype=torch.uint8), torch.arange(4) + 20)
        self.assertTrue(attn.prepare(backend, bs=1))
        self.assertEqual(workspace.prefix_meta_writes, meta_writes)
        self.assertEqual(workspace.stage2_writes, stage2_writes + 1)

    def test_single_runner_after_batch_rewrites_tree_columns(self):
        backend = _cpu_backend(draft_tokens=2)
        backend.speculative_num_steps = 4
        pool = torch.full((1, 64), -1, dtype=torch.int32)
        pool[0, :6] = torch.arange(6, dtype=torch.int32)
        pool[0, 6:8] = torch.tensor([10, 11], dtype=torch.int32)
        attn = OptTreeAttention(PrefixPlan(pool[0, :6], 6), active=True)
        runner = SimpleNamespace(
            prefix_len=6,
            cached_len=8,
            req=SimpleNamespace(req_pool_idx=0),
            scheduler=SimpleNamespace(
                req_to_token_pool=SimpleNamespace(req_to_token=pool)
            ),
        )
        module = "sglang.srt.speculative.opt_tree_remote.opt_tree_attention"
        with mock.patch(f"{module}._attention_for", return_value=attn):
            bind_opt_cascade(
                SimpleNamespace(),
                runner,
                torch.tensor([20, 21]),
                torch.ones(2, 4, dtype=torch.uint8),
                append_slots=True,
            )
        self.assertTrue(attn.prepare(backend, bs=1))
        workspace = ensure_workspace(backend, 2)
        self.assertEqual(workspace.tree_indices[:4].tolist(), [10, 11, 20, 21])

    def test_capture_resets_prefix_generation(self):
        backend = _cpu_backend()
        plan = PrefixPlan(torch.arange(6), 6)
        attn = OptTreeAttention(plan, active=True)
        attn.note_layer(torch.eye(2, dtype=torch.uint8), torch.arange(2))
        self.assertTrue(attn.prepare(backend, bs=1))
        workspace = ensure_workspace(backend, 2)
        self.assertEqual(workspace.prefix_generation, plan.generation)
        captured = OptTreeAttention(plan=None, active=True, capture_query_width=2)
        self.assertTrue(captured.prepare(backend, bs=1))
        self.assertIsNone(workspace.prefix_generation)
        copies = plan.prefix_copies
        attn.note_layer(torch.eye(2, dtype=torch.uint8), torch.arange(2))
        self.assertTrue(attn.prepare(backend, bs=1))
        self.assertEqual(plan.prefix_copies, copies + 1)
        self.assertEqual(int(workspace.prefix_len.item()), 6)
        self.assertTrue(torch.equal(workspace.prefix_indices[:6], plan.indices))

    def test_tree_workspace_is_not_the_context_length(self):
        backend = _cpu_backend(context=4096, draft_tokens=4)
        backend.speculative_num_steps = 10
        self.assertEqual(tree_limit(backend), 41)
        workspace = ensure_workspace(backend, 4)
        self.assertEqual(workspace.max_prefix, 4096)
        self.assertEqual(workspace.max_tree, 41)
        self.assertLess(workspace.max_tree, workspace.max_prefix)

    def test_prepare_rejects_inactive_and_batched(self):
        plan = PrefixPlan(torch.arange(3), 3)
        inactive = OptTreeAttention(plan, active=False)
        self.assertFalse(inactive.prepare(None, bs=1))
        active = OptTreeAttention(plan, active=True)
        self.assertFalse(active.prepare(None, bs=2))

    def test_eager_prepare_failure_stays_on_extend(self):
        backend = _cpu_backend()
        plan = PrefixPlan(torch.arange(4), 4)
        attn = OptTreeAttention(plan, active=True)
        batch = SimpleNamespace(batch_size=1)
        self.assertFalse(attn.prepare(backend, batch=batch))
        self.assertFalse(attn.ready)

    def test_replay_prepare_failure_on_captured_graph_raises(self):
        backend = _cpu_backend()
        captured = OptTreeAttention(plan=None, active=True, capture_query_width=4)
        self.assertTrue(captured.prepare(backend, bs=1))
        self.assertIn((id(backend), 4), _captured_cascade)

        plan = PrefixPlan(torch.arange(4), 4)
        runtime = OptTreeAttention(plan, active=True)
        runtime.note_layer(
            torch.eye(4, dtype=torch.uint8),
            torch.arange(4),
            prefix_rows=torch.ones(2, dtype=torch.int32),
        )
        with self.assertRaises(RuntimeError) as ctx:
            runtime.prepare(backend, bs=1)
        self.assertIn(
            "OPT cascade prepare failed without a usable fallback",
            str(ctx.exception),
        )
        self.assertFalse(runtime.ready)

        incomplete = OptTreeAttention(plan, active=True)
        incomplete.note_layer(
            torch.eye(4, dtype=torch.uint8),
            torch.arange(4),
            prefix_rows=torch.ones(2, dtype=torch.int32),
            spec_mask_complete=False,
        )
        with self.assertRaises(RuntimeError) as eager_ctx:
            incomplete.prepare(backend, batch=SimpleNamespace(batch_size=1))
        self.assertIn(
            "OPT cascade prepare failed without a usable fallback",
            str(eager_ctx.exception),
        )
        self.assertFalse(incomplete.ready)

        missing = OptTreeAttention(plan, active=True)
        with self.assertRaises(RuntimeError):
            missing.prepare(backend, bs=1)
        self.assertFalse(missing.ready)

        other = OptTreeAttention(plan, active=True)
        other.note_layer(
            torch.eye(5, dtype=torch.uint8),
            torch.arange(5),
            prefix_rows=torch.ones(1, dtype=torch.int32),
        )
        self.assertFalse(other.prepare(backend, bs=1))
        self.assertFalse(other.ready)

        eager = OptTreeAttention(plan, active=True)
        self.assertFalse(eager.prepare(backend, batch=SimpleNamespace(batch_size=1)))
        self.assertFalse(eager.ready)

    def test_failed_capture_does_not_bind_the_graph(self):
        backend = _cpu_backend()
        captured = OptTreeAttention(plan=None, active=True, capture_query_width=0)
        self.assertFalse(captured.prepare(backend, bs=1))
        self.assertFalse(_captured_cascade)
        runtime = OptTreeAttention(PrefixPlan(torch.arange(4), 4), active=True)
        self.assertFalse(runtime.prepare(backend, bs=1))
        self.assertFalse(runtime.ready)

    def test_n19_query_group_is_one_tile(self):
        rows = 19 * 2
        block = cascade_query_block(rows)
        self.assertEqual(block, 64)
        self.assertEqual(cdiv(rows, block), 1)
        self.assertEqual(cdiv(rows, 32), 2)

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
        start = text.index("def forward_extend")
        body = text[start : text.index("\n    def ", start + 1)]
        self.assertIn('getattr(_attn, "ready", False)', body)
        self.assertNotIn('getattr(_attn, "active", False)', body)

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


def _require_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("cuda")
    try:
        import triton  # noqa: F401
    except Exception as exc:
        raise unittest.SkipTest(f"triton: {exc}")


class OptTreeCascadeCudaTest(CustomTestCase):
    def test_matches_torch_reference(self):
        _require_cuda()
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

    def test_n19_bf16_long_prefix_matches_reference(self):
        _require_cuda()
        from sglang.srt.speculative.opt_tree_remote.opt_tree_attention import (
            run_cascade_attention,
        )

        torch.manual_seed(11)
        device = "cuda"
        queries, kv_heads, group, dim = 19, 2, 2, 32
        heads = kv_heads * group
        prefix_len, tree_cols = 4096, 19
        scale = dim**-0.5
        q = torch.randn(queries, heads, dim, device=device, dtype=torch.bfloat16)
        tokens = prefix_len + tree_cols + 3
        k_cache = torch.randn(
            tokens, kv_heads, dim, device=device, dtype=torch.bfloat16
        )
        v_cache = torch.randn(
            tokens, kv_heads, dim, device=device, dtype=torch.bfloat16
        )
        prefix_idx = torch.arange(prefix_len, device=device)
        tree_idx = torch.arange(prefix_len, prefix_len + tree_cols, device=device)
        tree_mask = torch.zeros(queries, tree_cols, dtype=torch.bool, device=device)
        tree_mask[:, 0] = True
        columns = torch.arange(queries, device=device) % tree_cols
        tree_mask[torch.arange(queries, device=device), columns] = True
        prefix_rows = torch.ones(queries, dtype=torch.bool, device=device)
        prefix_rows[-1] = False
        got = run_cascade_attention(
            q, k_cache, v_cache, prefix_idx, tree_idx, tree_mask, prefix_rows, scale
        )
        ref = _reference(
            q, k_cache, v_cache, prefix_idx, tree_idx, tree_mask, prefix_rows, scale
        )
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)

    def test_replay_uses_the_replaced_prefix(self):
        _require_cuda()
        from sglang.srt.speculative.opt_tree_remote.opt_tree_attention import (
            launch_workspace,
        )

        torch.manual_seed(13)
        device = "cuda"
        queries, kv_heads, group, dim = 4, 2, 2, 32
        heads = kv_heads * group
        tree_cols = 6
        scale = dim**-0.5
        backend = SimpleNamespace(
            device=device,
            num_draft_tokens=queries,
            speculative_num_steps=2,
            num_head=heads,
            v_head_dim=dim,
            max_context_len=256,
        )
        _captured_cascade.clear()
        captured = OptTreeAttention(
            plan=None, active=True, capture_query_width=queries
        )
        self.assertTrue(captured.prepare(backend, bs=1))

        tokens = 256
        k_cache = torch.randn(tokens, kv_heads, dim, device=device)
        v_cache = torch.randn(tokens, kv_heads, dim, device=device)
        q = torch.randn(queries, heads, dim, device=device)
        tree_idx = torch.arange(200, 200 + tree_cols, device=device)
        tree_mask = torch.zeros(queries, tree_cols, dtype=torch.bool, device=device)
        tree_mask[:, 0] = True
        columns = torch.arange(queries, device=device) % tree_cols
        tree_mask[torch.arange(queries, device=device), columns] = True
        prefix_rows = torch.ones(queries, dtype=torch.bool, device=device)
        prefix_rows[-1] = False
        prefix_a = torch.arange(40, device=device)
        prefix_b = torch.arange(50, 130, device=device)
        attn = OptTreeAttention(PrefixPlan(prefix_a, int(prefix_a.shape[0])), active=True)
        attn.note_layer(tree_mask, tree_idx, prefix_rows)

        def launch(prefix):
            attn.plan = PrefixPlan(prefix, int(prefix.shape[0]))
            self.assertTrue(attn.prepare(backend, bs=1))
            out = torch.empty_like(q)
            launch_workspace(q, out, k_cache, v_cache, scale, kv_heads)
            return out

        got_a = launch(prefix_a)
        ref_a = _reference(
            q, k_cache, v_cache, prefix_a, tree_idx, tree_mask, prefix_rows, scale
        )
        torch.testing.assert_close(got_a, ref_a, atol=2e-3, rtol=2e-3)
        got_b = launch(prefix_b)
        ref_b = _reference(
            q, k_cache, v_cache, prefix_b, tree_idx, tree_mask, prefix_rows, scale
        )
        torch.testing.assert_close(got_b, ref_b, atol=2e-3, rtol=2e-3)
        self.assertFalse(torch.allclose(got_b, ref_a, atol=2e-3, rtol=2e-3))
        self.assertGreater(float((got_b - ref_a).abs().max()), 1e-2)


if __name__ == "__main__":
    unittest.main()
