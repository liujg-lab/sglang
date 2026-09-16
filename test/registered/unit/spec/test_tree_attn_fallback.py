"""CPU tests for Target tree-verify slot-gather attention fallback.

Avoids importing the NPU attention backend (torch_npu). Math helpers come
from ``tree_attn_fallback``; wiring is source-guarded via AST.
"""

import ast
import math
import pathlib
import types
import unittest

import torch

from sglang.srt.speculative.tree_attn_fallback import (
    build_tree_verify_kv_slots,
    chunked_attend,
    flatten_paged_kv,
    gather_kv_by_slots,
    resolve_backend_topks,
    should_skip_npu_target_verify_graph,
    tree_verify_attention,
    use_tree_verify_fallback,
    verify_tree_topk_from_server_args,
)
from sglang.srt.speculative.tree_attn_mask import (
    iter_full_mask_rows,
    visible_token_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_ASCEND_BACKEND = (
    _REPO_ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
)
_CUDA_GRAPH_RUNNER = (
    _REPO_ROOT / "python/sglang/srt/model_executor/cuda_graph_runner.py"
)
_NPU_GRAPH_RUNNER = (
    _REPO_ROOT
    / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py"
)
_FALLBACK = _REPO_ROOT / "python/sglang/srt/speculative/tree_attn_fallback.py"


def _function_source(path: pathlib.Path, name: str) -> str:
    text = path.read_text()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node) or ast.unparse(node)
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    return ast.get_source_segment(text, child) or ast.unparse(child)
    raise AssertionError(f"{name} not found in {path}")


def _class_method_source(path: pathlib.Path, class_name: str, method_name: str) -> str:
    text = path.read_text()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.get_source_segment(text, child) or ast.unparse(child)
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def _dense_attend(q, k, v, scale, q_rope=None, k_rope=None):
    q = q.reshape(-1, q.shape[-1]).float()
    n_q = q.shape[0]
    n_rep = n_q // k.shape[1]
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        if k_rope is not None:
            k_rope = k_rope.repeat_interleave(n_rep, dim=1)
    scores = torch.einsum("hd,shd->hs", q, k.float())
    if q_rope is not None:
        scores = scores + torch.einsum(
            "hd,shd->hs", q_rope.reshape(-1, q_rope.shape[-1]).float(), k_rope.float()
        )
    probs = torch.softmax(scores * float(scale), dim=-1)
    return torch.einsum("hs,shd->hd", probs, v.float())


def _tree_verify_attention_per_query(
    query,
    k_cache,
    v_cache,
    *,
    custom_mask,
    seq_lens,
    req_to_token,
    req_pool_indices,
    out_cache_loc,
    num_draft,
    scale,
    n_q_heads,
    n_kv_heads,
    qk_head_dim,
    v_head_dim,
):
    query = query.reshape(-1, n_q_heads, qk_head_dim)
    flat_k = flatten_paged_kv(k_cache, n_kv_heads, qk_head_dim)
    flat_v = flatten_paged_kv(v_cache, n_kv_heads, v_head_dim)
    seq_list = [int(x) for x in seq_lens]
    req_pool = req_pool_indices.reshape(-1).to(dtype=torch.int64)
    draft_locs_all = out_cache_loc.reshape(-1)
    outputs = []
    for b, t, attend_row in iter_full_mask_rows(custom_mask, seq_list, num_draft):
        q_idx = b * num_draft + t
        if q_idx >= query.shape[0]:
            break
        seq_len = seq_list[b]
        req = int(req_pool[b].item())
        prefix_locs = req_to_token[req, :seq_len]
        draft_locs = draft_locs_all[b * num_draft : (b + 1) * num_draft]
        slots = visible_token_indices(attend_row, prefix_locs, draft_locs)
        k_vis, v_vis = gather_kv_by_slots(flat_k, flat_v, slots)
        outputs.append(
            chunked_attend(query[q_idx], k_vis, v_vis, scale, chunk_size=2)
        )
    if not outputs:
        return query.new_zeros(query.shape[0], n_q_heads * v_head_dim)
    stacked = torch.stack(outputs, dim=0)
    return stacked.reshape(stacked.shape[0], n_q_heads * v_head_dim)


class TestTreeAttnFallback(CustomTestCase):
    def test_dispatch_topk_and_mask(self):
        mask = torch.ones(4, dtype=torch.bool)
        self.assertTrue(use_tree_verify_fallback(True, 3, mask))
        self.assertFalse(use_tree_verify_fallback(True, 1, mask))
        self.assertFalse(use_tree_verify_fallback(False, 3, mask))
        self.assertFalse(use_tree_verify_fallback(True, 1, None))
        with self.assertRaises(RuntimeError) as ctx:
            use_tree_verify_fallback(True, 3, None)
        self.assertIn("custom_mask", str(ctx.exception))
        self.assertIn("linear FIA", str(ctx.exception))
        with self.assertRaises(RuntimeError):
            use_tree_verify_fallback(True, 3, torch.empty(0))

    def test_target_factory_wiring_keeps_draft_topk_one(self):
        args = types.SimpleNamespace(speculative_eagle_topk=3)
        self.assertEqual(verify_tree_topk_from_server_args(args), 3)
        self.assertEqual(verify_tree_topk_from_server_args(types.SimpleNamespace()), 1)
        draft_topk, verify_topk = resolve_backend_topks(1, args)
        self.assertEqual(draft_topk, 1)
        self.assertEqual(verify_topk, 3)
        mask = torch.ones(4, dtype=torch.bool)
        self.assertTrue(use_tree_verify_fallback(True, verify_topk, mask))
        self.assertFalse(use_tree_verify_fallback(True, draft_topk, mask))
        draft_topk, verify_topk = resolve_backend_topks(3, args)
        self.assertEqual(draft_topk, 3)
        self.assertEqual(verify_topk, 3)

    def test_skip_verify_graph_npu_tree_only(self):
        self.assertFalse(should_skip_npu_target_verify_graph("npu", 3))
        self.assertFalse(should_skip_npu_target_verify_graph("npu:0", 3))
        self.assertFalse(should_skip_npu_target_verify_graph("npu", 1))
        self.assertFalse(should_skip_npu_target_verify_graph("cuda", 3))

    def test_sibling_slots_in_same_page_are_not_gathered(self):
        page_size = 4
        prefix_locs = torch.tensor([0], dtype=torch.int64)
        # Page 1 holds draft slots 4 (root), 5 (child), 6 (sibling of 5).
        draft_locs = torch.tensor([4, 5, 6], dtype=torch.int64)
        attend_q1 = torch.tensor([True, True, True, False], dtype=torch.bool)
        vis = visible_token_indices(attend_q1, prefix_locs, draft_locs)
        self.assertEqual(vis.tolist(), [0, 4, 5])
        self.assertNotIn(6, vis.tolist())
        self.assertEqual(int(vis[1] // page_size), int(6 // page_size))

        n_heads, dim = 2, 4
        k = torch.zeros(2, page_size, n_heads, dim)
        v = torch.zeros(2, page_size, n_heads, dim)
        for slot in range(8):
            k.view(-1, n_heads, dim)[slot, :, 0] = float(slot)
            v.view(-1, n_heads, dim)[slot, :, 0] = float(slot + 100)
        flat_k = flatten_paged_kv(k, n_heads, dim)
        flat_v = flatten_paged_kv(v, n_heads, dim)
        g_k, g_v = gather_kv_by_slots(flat_k, flat_v, vis)
        self.assertEqual(g_k[:, 0, 0].tolist(), [0.0, 4.0, 5.0])
        self.assertEqual(g_v[:, 0, 0].tolist(), [100.0, 104.0, 105.0])
        self.assertFalse(torch.any(g_k[:, 0, 0] == 6))

    def test_flatten_packed_heads_matches_token_slots(self):
        page_size, n_heads, dim = 4, 2, 3
        packed = torch.arange(2 * page_size * n_heads * dim, dtype=torch.float32).view(
            2, page_size, n_heads * dim
        )
        flat = flatten_paged_kv(packed, n_heads, dim)
        self.assertEqual(tuple(flat.shape), (8, n_heads, dim))
        self.assertTrue(torch.equal(flat[5], packed[1, 1].view(n_heads, dim)))

    def test_chunked_attend_matches_dense_and_chunk_size(self):
        torch.manual_seed(0)
        h, s, d = 4, 7, 8
        kv = 2
        q = torch.randn(h, d)
        k = torch.randn(s, kv, d)
        v = torch.randn(s, kv, d)
        scale = 1.0 / math.sqrt(d)
        ref = _dense_attend(q, k, v, scale)
        full = chunked_attend(q, k, v, scale, chunk_size=256)
        chunked = chunked_attend(q, k, v, scale, chunk_size=2)
        self.assertTrue(torch.allclose(full.float(), ref, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(chunked.float(), ref, atol=1e-5, rtol=1e-5))

    def test_chunked_attend_empty_visible_is_zero(self):
        q = torch.ones(2, 4)
        k = torch.zeros(0, 2, 4)
        v = torch.zeros(0, 2, 4)
        out = chunked_attend(q, k, v, 1.0)
        self.assertEqual(tuple(out.shape), (2, 4))
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_tree_verify_hides_sibling_value(self):
        n_q, n_kv, d = 2, 2, 4
        page_size = 4
        num_pages = 2
        k_cache = torch.zeros(num_pages, page_size, n_kv * d)
        v_cache = torch.zeros(num_pages, page_size, n_kv * d)
        flat_k = k_cache.view(-1, n_kv, d)
        flat_v = v_cache.view(-1, n_kv, d)
        # Distinct keys so softmax is not uniform; sibling V is a spike.
        for slot, val in ((0, 1.0), (4, 1.0), (5, 1.0), (6, 1.0)):
            flat_k[slot, :, 0] = val
        flat_v[0, :, 0] = 1.0
        flat_v[4, :, 0] = 2.0
        flat_v[5, :, 0] = 3.0
        flat_v[6, :, 0] = 1000.0

        query = torch.ones(3, n_q, d)
        custom = torch.tensor(
            [
                True, True, False, False,
                True, True, True, False,
                True, True, False, True,
            ],
            dtype=torch.bool,
        )
        req_to_token = torch.zeros((1, 8), dtype=torch.int64)
        req_to_token[0, 0] = 0
        out_cache_loc = torch.tensor([4, 5, 6], dtype=torch.int64)
        out = tree_verify_attention(
            query,
            k_cache,
            v_cache,
            custom_mask=custom,
            seq_lens=[1],
            req_to_token=req_to_token,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            out_cache_loc=out_cache_loc,
            num_draft=3,
            scale=1.0,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
            qk_head_dim=d,
            v_head_dim=d,
            chunk_size=2,
        )
        # Query 1 attends slots 0,4,5 — not sibling 6.
        q1 = out[1].view(n_q, d)
        self.assertLess(float(q1[0, 0]), 10.0)
        vis1 = visible_token_indices(
            torch.tensor([True, True, True, False]),
            torch.tensor([0]),
            out_cache_loc,
        )
        k_vis, v_vis = gather_kv_by_slots(flat_k, flat_v, vis1)
        ref = _dense_attend(query[1], k_vis, v_vis, 1.0)
        self.assertTrue(torch.allclose(q1.float(), ref, atol=1e-5, rtol=1e-5))

    def test_chain_mask_matches_causal_dense(self):
        n_q, n_kv, d = 2, 1, 4
        torch.manual_seed(1)
        seq_len, num_draft = 2, 2
        page_size = 4
        k_cache = torch.randn(1, page_size, n_kv * d)
        v_cache = torch.randn(1, page_size, n_kv * d)
        query = torch.randn(num_draft, n_q, d)
        # Causal over prefix + drafts: q0 sees 2 prefix + self; q1 sees all 4.
        custom = torch.tensor(
            [True, True, True, False, True, True, True, True], dtype=torch.bool
        )
        req_to_token = torch.arange(page_size, dtype=torch.int64).unsqueeze(0)
        out_cache_loc = torch.tensor([2, 3], dtype=torch.int64)
        scale = 1.0 / math.sqrt(d)
        out = tree_verify_attention(
            query,
            k_cache,
            v_cache,
            custom_mask=custom,
            seq_lens=[seq_len],
            req_to_token=req_to_token,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            out_cache_loc=out_cache_loc,
            num_draft=num_draft,
            scale=scale,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
            qk_head_dim=d,
            v_head_dim=d,
            chunk_size=2,
        )
        flat_k = flatten_paged_kv(k_cache, n_kv, d)
        flat_v = flatten_paged_kv(v_cache, n_kv, d)
        packed_k = torch.cat([flat_k[:seq_len], flat_k[seq_len : seq_len + num_draft]], 0)
        packed_v = torch.cat([flat_v[:seq_len], flat_v[seq_len : seq_len + num_draft]], 0)
        ref0 = _dense_attend(query[0], packed_k[:3], packed_v[:3], scale)
        ref1 = _dense_attend(query[1], packed_k[:4], packed_v[:4], scale)
        self.assertTrue(
            torch.allclose(out[0].view(n_q, d).float(), ref0, atol=1e-5, rtol=1e-5)
        )
        self.assertTrue(
            torch.allclose(out[1].view(n_q, d).float(), ref1, atol=1e-5, rtol=1e-5)
        )

    def test_mla_rope_scores_match_dense(self):
        torch.manual_seed(2)
        n_q, n_kv, d, rd = 2, 2, 4, 2
        q = torch.randn(n_q, d)
        k = torch.randn(5, n_kv, d)
        v = torch.randn(5, n_kv, d)
        q_rope = torch.randn(n_q, rd)
        k_rope = torch.randn(5, n_kv, rd)
        scale = 0.5
        ref = _dense_attend(q, k, v, scale, q_rope=q_rope, k_rope=k_rope)
        out = chunked_attend(
            q, k, v, scale, chunk_size=2, q_rope=q_rope, k_rope=k_rope
        )
        self.assertTrue(torch.allclose(out.float(), ref, atol=1e-5, rtol=1e-5))

    def test_forward_mtp_source_uses_fallback_not_fia_tree_mask(self):
        src = _function_source(_ASCEND_BACKEND, "forward_mtp")
        self.assertIn("use_tree_verify_fallback", src)
        self.assertIn("_run_tree_verify_slot_gather", src)
        self.assertIn("self.verify_tree_topk", src)
        self.assertIn("atten_mask=self.mtp_mask", src)
        self.assertIn("sparse_mode=3", src)
        self.assertNotIn("sparse_mode=0", src)
        self.assertNotIn("tree_mask if use_tree", src)
        self.assertNotIn("locs_to_page_ids", src)
        self.assertEqual(src.count("self.draft_topk"), 0)
        self.assertGreaterEqual(src.count("self.verify_tree_topk"), 2)

        helper_src = _class_method_source(
            _ASCEND_BACKEND, "AscendAttnBackend", "_run_tree_verify_slot_gather"
        )
        self.assertIn("tree_verify_attention", helper_src)
        self.assertIn("log_tree_verify_fallback_once", helper_src)
        self.assertIn("kv_slots=self.forward_metadata.tree_verify_kv_slots", helper_src)

        fallback_src = _FALLBACK.read_text()
        self.assertNotIn("locs_to_page_ids", fallback_src)
        self.assertNotIn("// page_size", fallback_src)
        self.assertIn("refusing to fall back to linear FIA", fallback_src)

        init_src = _class_method_source(_ASCEND_BACKEND, "AscendAttnBackend", "__init__")
        self.assertIn("self.verify_tree_topk = verify_tree_topk_from_server_args", init_src)
        self.assertIn("self.draft_topk = max(int(draft_topk), 1)", init_src)

    def test_capture_does_not_bind_tree_mask_to_fia(self):
        capture_src = _function_source(
            _ASCEND_BACKEND, "init_forward_metadata_capture_cuda_graph"
        )
        self.assertNotIn(
            "metadata.tree_attn_mask = self.cuda_graph_tree_attn_mask",
            capture_src,
        )
        self.assertIn("_bind_graph_verify_slot_views", capture_src)
        self.assertIn("_bind_graph_draft_slot_views", capture_src)

    def test_verify_graph_slot_buffers_wired(self):
        state_src = _function_source(_ASCEND_BACKEND, "init_cuda_graph_state")
        self.assertIn("cuda_graph_kv_slots", state_src)
        self.assertIn("cuda_graph_kv_lens", state_src)
        init_src = _function_source(_ASCEND_BACKEND, "init_forward_metadata")
        self.assertIn("_fill_tree_verify_kv_slots", init_src)
        replay_src = _function_source(
            _ASCEND_BACKEND, "init_forward_metadata_replay_cuda_graph"
        )
        self.assertIn("_fill_tree_verify_kv_slots", replay_src)
        self.assertIn("_restore_graph_verify_slot_views", replay_src)

    def test_cuda_graph_runner_captures_npu_tree_verify_ntpb(self):
        init_src = _function_source(_CUDA_GRAPH_RUNNER, "__init__")
        self.assertNotIn("should_skip_npu_target_verify_graph", init_src)
        self.assertNotIn("eager slot-gather fallback", init_src)
        self.assertIn("self.spectre_ntpb_options", init_src)

    def test_npu_graph_runner_skips_fia_update_for_tree_verify(self):
        replay_src = _function_source(_NPU_GRAPH_RUNNER, "replay")
        self.assertIn("skip_fia_update", replay_src)
        self.assertIn("speculative_eagle_topk", replay_src)

    def test_build_tree_verify_kv_slots_matches_visible(self):
        custom = torch.tensor(
            [
                True, True, False, False,
                True, True, True, False,
                True, True, False, True,
            ],
            dtype=torch.bool,
        )
        req_to_token = torch.zeros((1, 8), dtype=torch.int64)
        req_to_token[0, 0] = 0
        out_cache_loc = torch.tensor([4, 5, 6], dtype=torch.int64)
        slots, lens = build_tree_verify_kv_slots(
            custom,
            [1],
            req_to_token,
            torch.tensor([0], dtype=torch.int64),
            out_cache_loc,
            num_draft=3,
        )
        prefix = torch.tensor([0], dtype=torch.int64)
        expected = [
            visible_token_indices(
                torch.tensor([True, True, False, False]), prefix, out_cache_loc
            ),
            visible_token_indices(
                torch.tensor([True, True, True, False]), prefix, out_cache_loc
            ),
            visible_token_indices(
                torch.tensor([True, True, False, True]), prefix, out_cache_loc
            ),
        ]
        self.assertEqual(tuple(slots.shape), (3, 4))
        for i, vis in enumerate(expected):
            n = int(vis.numel())
            self.assertEqual(int(lens[i]), n)
            self.assertEqual(slots[i, :n].tolist(), vis.tolist())

    def test_batched_verify_matches_per_query_oracle(self):
        n_q, n_kv, d = 4, 2, 4
        page_size = 4
        k_cache = torch.randn(2, page_size, n_kv * d)
        v_cache = torch.randn(2, page_size, n_kv * d)
        query = torch.randn(3, n_q, d)
        custom = torch.tensor(
            [
                True, True, False, False,
                True, True, True, False,
                True, True, False, True,
            ],
            dtype=torch.bool,
        )
        req_to_token = torch.zeros((1, 8), dtype=torch.int64)
        req_to_token[0, 0] = 0
        out_cache_loc = torch.tensor([4, 5, 6], dtype=torch.int64)
        kwargs = dict(
            custom_mask=custom,
            seq_lens=[1],
            req_to_token=req_to_token,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            out_cache_loc=out_cache_loc,
            num_draft=3,
            scale=1.0 / math.sqrt(d),
            n_q_heads=n_q,
            n_kv_heads=n_kv,
            qk_head_dim=d,
            v_head_dim=d,
        )
        batched = tree_verify_attention(query, k_cache, v_cache, **kwargs)
        oracle = _tree_verify_attention_per_query(query, k_cache, v_cache, **kwargs)
        self.assertTrue(
            torch.allclose(batched.float(), oracle.float(), atol=1e-4, rtol=1e-4)
        )


if __name__ == "__main__":
    unittest.main()
