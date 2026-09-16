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
    ATTN_CHUNK_ALIGN,
    MAX_ATTN_CHUNK_WIDTH,
    MAX_ATTN_CHUNKS,
    build_tree_verify_kv_slots,
    build_tree_verify_kv_slots_ref,
    chunked_attend,
    flatten_paged_kv,
    gather_kv_by_slots,
    gather_kv_into,
    parse_tree_draft_capture_bs,
    parse_tree_graph_kv_buckets,
    resolve_backend_topks,
    select_tree_kv_bucket,
    should_skip_npu_target_verify_graph,
    tree_attn_chunk_width,
    tree_compact_fia_layout_supported,
    tree_draft_attention,
    tree_fia_actual_seq_lengths_kv,
    tree_verify_attention,
    use_tree_verify_fallback,
    verify_tree_topk_from_server_args,
    zero_gathered_kv_padding,
)
from sglang.srt.speculative.tree_attn_mask import (
    full_mask_numel,
    iter_full_mask_rows,
    resolve_tree_verify_mask_seq_lens,
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
        self.assertIn("_tree_verify_mask_layout", replay_src)
        self.assertIn("raw_bs", replay_src)
        self.assertIn("int(raw_bs) * num_draft", replay_src)
        fill_slots_src = _function_source(_ASCEND_BACKEND, "_fill_tree_verify_kv_slots")
        self.assertIn("_tree_verify_mask_layout", fill_slots_src)
        self.assertIn("log_tree_verify_kv_slot_layout_once", fill_slots_src)
        self.assertIn("rows_limit", fill_slots_src)
        self.assertIn("ValueError", fill_slots_src)
        layout_src = _function_source(_ASCEND_BACKEND, "_tree_verify_mask_layout")
        self.assertIn("resolve_tree_verify_mask_seq_lens", layout_src)
        self.assertIn("seq_lens_cpu", layout_src)
        max_kv_src = _function_source(_ASCEND_BACKEND, "_slot_gather_graph_max_kv")
        self.assertIn("_slot_gather_kv_pool_size", max_kv_src)
        self.assertIn("min(cap, pool)", max_kv_src)
        self.assertNotIn("return int(self.max_context_len) + extra", max_kv_src)
        # The slot table must be sized by the KV pool. req_to_token is
        # context_len wide, so it caps nothing.
        pool_src = _function_source(_ASCEND_BACKEND, "_slot_gather_kv_pool_size")
        self.assertIn("token_to_kv_pool", pool_src)
        self.assertIn("max_total_num_tokens", pool_src)
        self.assertNotIn("self.req_to_token", pool_src)
        state_src = _function_source(_ASCEND_BACKEND, "init_cuda_graph_state")
        self.assertIn("mask_max_kv", state_src)
        self.assertIn("slot_max_kv", state_src)
        self.assertIn("_slot_gather_graph_slot_max_kv", state_src)
        self.assertIn("max_q * mask_max_kv", state_src)
        self.assertIn("(max_q, slot_max_kv)", state_src)

    def test_graph_state_drops_dead_tree_mask_for_slot_gather(self):
        state_src = _function_source(_ASCEND_BACKEND, "init_cuda_graph_state")
        self.assertIn("_slot_gather_tree_attn", state_src)
        self.assertIn("self.cuda_graph_tree_attn_mask = None", state_src)
        # custom_mask is still consumed when building the slot table.
        self.assertIn("cuda_graph_custom_mask", state_src)
        fill_src = _function_source(_ASCEND_BACKEND, "_fill_tree_verify_mask")
        self.assertIn("_slot_gather_tree_attn", fill_src)
        self.assertIn("_tree_verify_mask_layout", fill_src)

    def test_chunked_attention_matches_dense_slot_gather(self):
        torch.manual_seed(3)
        page_size, num_pages = 128, 8
        for n_q, n_kv in ((4, 4), (8, 2), (8, 1)):
            rows, s_pad, d = 5, 600, 16
            q = torch.randn(rows, n_q, d)
            k_cache = torch.randn(num_pages, page_size, n_kv, d)
            v_cache = torch.randn(num_pages, page_size, n_kv, d)
            kv_slots = torch.randint(0, num_pages * page_size, (rows, s_pad))
            kv_lens = torch.randint(1, s_pad + 1, (rows,), dtype=torch.int32)
            scale = 1.0 / math.sqrt(d)
            out = tree_draft_attention(
                q,
                k_cache,
                v_cache,
                kv_slots=kv_slots,
                kv_lens=kv_lens,
                scale=scale,
                n_q_heads=n_q,
                n_kv_heads=n_kv,
                qk_head_dim=d,
                v_head_dim=d,
            )
            self.assertEqual(tuple(out.shape), (rows, n_q * d))
            flat_k = flatten_paged_kv(k_cache, n_kv, d)
            flat_v = flatten_paged_kv(v_cache, n_kv, d)
            for r in range(rows):
                n = int(kv_lens[r])
                k_vis, v_vis = gather_kv_by_slots(flat_k, flat_v, kv_slots[r, :n])
                ref = _dense_attend(q[r], k_vis, v_vis, scale)
                self.assertTrue(
                    torch.allclose(
                        out[r].view(n_q, d).float(), ref, atol=1e-4, rtol=1e-4
                    ),
                    msg=f"n_q={n_q} n_kv={n_kv} row={r}",
                )

    def test_chunked_attention_ignores_padding_columns(self):
        torch.manual_seed(4)
        rows, n_q, n_kv, d = 3, 4, 2, 8
        page_size, num_pages = 128, 4
        q = torch.randn(rows, n_q, d)
        k_cache = torch.randn(num_pages, page_size, n_kv, d)
        v_cache = torch.randn(num_pages, page_size, n_kv, d)
        kv_lens = torch.tensor([0, 7, 300], dtype=torch.int32)
        base = torch.randint(0, num_pages * page_size, (rows, 300))

        def run(s_pad):
            slots = torch.zeros(rows, s_pad, dtype=torch.int64)
            slots[:, :300] = base
            if s_pad > 300:
                # Junk in the padding columns must not reach the output.
                slots[:, 300:] = torch.randint(
                    0, num_pages * page_size, (rows, s_pad - 300)
                )
            return tree_draft_attention(
                q,
                k_cache,
                v_cache,
                kv_slots=slots,
                kv_lens=kv_lens,
                scale=0.25,
                n_q_heads=n_q,
                n_kv_heads=n_kv,
                qk_head_dim=d,
                v_head_dim=d,
            )

        tight = run(300)
        padded = run(4096)
        self.assertTrue(torch.allclose(tight, padded, atol=1e-5, rtol=1e-5))
        # kv_lens == 0 contributes nothing.
        self.assertTrue(torch.equal(padded[0], torch.zeros_like(padded[0])))

    def test_chunked_attention_kv_bound_matches_full_width(self):
        torch.manual_seed(5)
        rows, n_q, n_kv, d = 4, 4, 4, 8
        page_size, num_pages = 128, 4
        q = torch.randn(rows, n_q, d)
        k_cache = torch.randn(num_pages, page_size, n_kv, d)
        v_cache = torch.randn(num_pages, page_size, n_kv, d)
        slots = torch.randint(0, num_pages * page_size, (rows, 2048))
        kv_lens = torch.full((rows,), 130, dtype=torch.int32)
        kwargs = dict(
            kv_slots=slots,
            kv_lens=kv_lens,
            scale=0.25,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
            qk_head_dim=d,
            v_head_dim=d,
        )
        full = tree_draft_attention(q, k_cache, v_cache, **kwargs)
        bounded = tree_draft_attention(q, k_cache, v_cache, kv_bound=130, **kwargs)
        self.assertTrue(torch.allclose(full, bounded, atol=1e-6, rtol=1e-6))

    def test_chunk_width_bounds_graph_node_count(self):
        for s_pad in (1, 17, 128, 384, 2063, 4096, 65536, 262151):
            width = tree_attn_chunk_width(s_pad)
            self.assertEqual(width % ATTN_CHUNK_ALIGN, 0)
            self.assertLessEqual(width, MAX_ATTN_CHUNK_WIDTH)
            chunks = -(-s_pad // width)
            if width < MAX_ATTN_CHUNK_WIDTH:
                self.assertLessEqual(chunks, MAX_ATTN_CHUNKS, msg=f"s_pad={s_pad}")
        # A pool-sized table stays within the node budget.
        self.assertLessEqual(-(-2063 // tree_attn_chunk_width(2063)), MAX_ATTN_CHUNKS)

    def test_attention_never_gathers_the_whole_slot_table(self):
        src = _function_source(_FALLBACK, "tree_draft_attention")
        # A dense gather is S_pad-sized even when every kv_len is zero, which
        # is exactly what graph capture feeds in.
        self.assertNotIn("safe_slots.reshape(-1)", src)
        self.assertNotIn(".repeat_interleave(", src)
        self.assertIn("tree_attn_chunk_width", src)
        self.assertIn("for start in range(0, max_kv, chunk)", src)
        self.assertIn("kv_slots[:, start : start + width]", src)
        # Trip count must be capture-time constant: no host sync inside.
        self.assertNotIn(".item()", src)

    def test_cuda_graph_runner_captures_npu_tree_verify_ntpb(self):
        init_src = _function_source(_CUDA_GRAPH_RUNNER, "__init__")
        self.assertNotIn("should_skip_npu_target_verify_graph", init_src)
        self.assertNotIn("eager slot-gather fallback", init_src)
        self.assertIn("self.spectre_ntpb_options", init_src)

    def test_npu_graph_runner_updates_fia_for_tree_verify(self):
        replay_src = _function_source(_NPU_GRAPH_RUNNER, "replay")
        self.assertIn("skip_fia_update", replay_src)
        self.assertIn("is_deepseek_nsa", replay_src)
        self.assertIn("compact_fia", replay_src)
        self.assertIn("tree_fia_kv_lens_cpu", replay_src)
        self.assertIn("tree_fia_actual_seq_lengths_kv", replay_src)
        self.assertIn("_replay_tree_s_cap", replay_src)

    def test_npu_graph_runner_can_run_defers_to_slot_width(self):
        can_src = _class_method_source(_NPU_GRAPH_RUNNER, "NPUGraphRunner", "can_run")
        self.assertIn("super().can_run", can_src)
        self.assertIn("tree_slot_graph_can_run", can_src)
        self.assertIn("_s", can_src)
        self.assertIn("tree_verify_eager_fallback_count", can_src)

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

    def test_log_config_all_true_mask_kv_lens(self):
        seq_len, num_draft = 128, 15
        custom = torch.ones(
            full_mask_numel([seq_len], num_draft), dtype=torch.bool
        )
        req_to_token = torch.arange(seq_len + num_draft, dtype=torch.int64).unsqueeze(0)
        out_cache_loc = torch.arange(seq_len, seq_len + num_draft, dtype=torch.int64)
        slots, lens = build_tree_verify_kv_slots(
            custom,
            [seq_len],
            req_to_token,
            torch.tensor([0], dtype=torch.int64),
            out_cache_loc,
            num_draft,
        )
        self.assertEqual(tuple(slots.shape), (num_draft, seq_len + num_draft))
        self.assertTrue(torch.equal(lens, torch.full((num_draft,), seq_len + num_draft)))
        self.assertEqual(slots[0, :seq_len].tolist(), list(range(seq_len)))
        self.assertEqual(
            slots[0, seq_len:].tolist(), list(range(seq_len, seq_len + num_draft))
        )

    def test_short_mask_raises_value_error_not_index_error(self):
        seq_len, num_draft = 128, 15
        expected = full_mask_numel([seq_len], num_draft)
        custom = torch.ones(expected - 9, dtype=torch.bool)
        req_to_token = torch.arange(seq_len + num_draft, dtype=torch.int64).unsqueeze(0)
        out_cache_loc = torch.arange(num_draft, dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            build_tree_verify_kv_slots(
                custom,
                [seq_len],
                req_to_token,
                torch.tensor([0], dtype=torch.int64),
                out_cache_loc,
                num_draft,
            )
        msg = str(ctx.exception)
        self.assertIn("FULL_MASK layout mismatch", msg)
        self.assertIn(str(expected), msg)
        self.assertIn(str(expected - 9), msg)
        self.assertIn("num_draft=15", msg)
        self.assertIn("seq_lens_sum=128", msg)

    def test_long_mask_raises_value_error(self):
        seq_len, num_draft = 128, 15
        expected = full_mask_numel([seq_len], num_draft)
        custom = torch.ones(expected + 10, dtype=torch.bool)
        req_to_token = torch.arange(seq_len + num_draft, dtype=torch.int64).unsqueeze(0)
        out_cache_loc = torch.arange(num_draft, dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            build_tree_verify_kv_slots(
                custom,
                [seq_len],
                req_to_token,
                torch.tensor([0], dtype=torch.int64),
                out_cache_loc,
                num_draft,
            )
        msg = str(ctx.exception)
        self.assertIn("FULL_MASK layout mismatch", msg)
        self.assertIn(str(expected), msg)
        self.assertIn(str(expected + 10), msg)

    def test_padded_replay_walks_only_raw_bs_rows(self):
        raw_bs, padded_bs, seq_len, num_draft = 1, 2, 128, 15
        custom = torch.ones(full_mask_numel([seq_len], num_draft), dtype=torch.bool)
        req_to_token = torch.arange(
            seq_len + num_draft, dtype=torch.int64
        ).unsqueeze(0).expand(padded_bs, -1).contiguous()
        out_cache_loc = torch.arange(padded_bs * num_draft, dtype=torch.int64)
        padded_seq = torch.tensor([seq_len, 0], dtype=torch.int32)
        with self.assertRaises(ValueError):
            build_tree_verify_kv_slots(
                custom,
                padded_seq,
                req_to_token,
                torch.arange(padded_bs, dtype=torch.int64),
                out_cache_loc,
                num_draft,
            )
        eager_slots, eager_lens = build_tree_verify_kv_slots(
            custom,
            [seq_len],
            req_to_token[:raw_bs],
            torch.tensor([0], dtype=torch.int64),
            out_cache_loc[: raw_bs * num_draft],
            num_draft,
        )
        slots, lens = build_tree_verify_kv_slots(
            custom,
            padded_seq,
            req_to_token,
            torch.arange(padded_bs, dtype=torch.int64),
            out_cache_loc,
            num_draft,
            rows_limit=raw_bs * num_draft,
        )
        self.assertEqual(tuple(slots.shape), (raw_bs * num_draft, seq_len + num_draft))
        self.assertTrue(torch.equal(slots, eager_slots))
        self.assertTrue(torch.equal(lens, eager_lens))
        dest_slots = torch.zeros(
            (padded_bs * num_draft, seq_len + num_draft), dtype=torch.int64
        )
        dest_lens = torch.zeros((padded_bs * num_draft,), dtype=torch.int32)
        dest_slots[: slots.shape[0]].copy_(slots)
        dest_lens[: lens.shape[0]].copy_(lens)
        self.assertTrue(torch.equal(dest_lens[: raw_bs * num_draft], eager_lens))
        self.assertTrue((dest_lens[raw_bs * num_draft :] == 0).all())

    def test_spec_info_seq_lens_preferred_over_padded_fallback(self):
        spec = types.SimpleNamespace(
            seq_lens_cpu=torch.tensor([128], dtype=torch.int32),
            seq_lens_sum=128,
        )
        seq_list, raw_bs, source = resolve_tree_verify_mask_seq_lens(
            spec, torch.tensor([128, 0], dtype=torch.int32)
        )
        self.assertEqual(source, "spec_info")
        self.assertEqual(raw_bs, 1)
        self.assertEqual(seq_list, [128])

        mismatch = types.SimpleNamespace(
            seq_lens_cpu=torch.tensor([128], dtype=torch.int32),
            seq_lens_sum=999,
        )
        seq_list, raw_bs, source = resolve_tree_verify_mask_seq_lens(
            mismatch, torch.tensor([50], dtype=torch.int32)
        )
        self.assertEqual(source, "fallback")
        self.assertEqual(raw_bs, 1)
        self.assertEqual(seq_list, [50])

        missing = types.SimpleNamespace(seq_lens_cpu=None, seq_lens_sum=None)
        seq_list, raw_bs, source = resolve_tree_verify_mask_seq_lens(
            missing, torch.tensor([128, 0], dtype=torch.int32)
        )
        self.assertEqual(source, "fallback")
        self.assertEqual(raw_bs, 2)
        self.assertEqual(seq_list, [128, 0])

    def test_visible_token_indices_rejects_short_row(self):
        prefix = torch.arange(128, dtype=torch.int64)
        draft = torch.arange(15, dtype=torch.int64)
        short = torch.ones(128 + 6, dtype=torch.bool)
        with self.assertRaises(ValueError) as ctx:
            visible_token_indices(short, prefix, draft)
        msg = str(ctx.exception)
        self.assertIn("attend_row=134", msg)
        self.assertIn("prefix=128", msg)
        self.assertIn("draft=15", msg)


class TestTreeGraphSlotCap(CustomTestCase):
    """Single S_cap + overflow eager: mask stays orig-sized, slots are capped."""

    def test_parse_tree_graph_max_kv(self):
        from sglang.srt.speculative.tree_attn_fallback import parse_tree_graph_max_kv

        self.assertEqual(parse_tree_graph_max_kv(None), 1024)
        self.assertEqual(parse_tree_graph_max_kv("2048"), 2048)
        for bad in ("0", "-1", "abc", "", "  ", "1.5"):
            with self.assertRaises(ValueError):
                parse_tree_graph_max_kv(bad)

    def test_slot_cap_mins_orig_bound(self):
        from sglang.srt.speculative.tree_attn_fallback import tree_graph_slot_max_kv

        self.assertEqual(tree_graph_slot_max_kv(8207, 1024), 1024)
        self.assertEqual(tree_graph_slot_max_kv(256, 1024), 256)

    def test_chunk_width_1024_is_eight_trips(self):
        width = tree_attn_chunk_width(1024)
        self.assertEqual(width, 128)
        self.assertEqual(-(-1024 // width), 8)

    def _admit(self, **kwargs):
        from sglang.srt.speculative.tree_attn_fallback import (
            tree_slot_graph_can_run_batch,
        )

        defaults = dict(
            slot_width=1024,
            slot_gather_enabled=True,
            is_target_verify=True,
            is_tree_draft=False,
            spec_info=None,
            fallback_seq_lens=[1],
            draft_token_num_fallback=15,
            draft_num_steps=5,
            seq_lens=[1],
        )
        defaults.update(kwargs)
        return tree_slot_graph_can_run_batch(**defaults)

    def _verify_info(self, seq_lens, num_draft=15, pad=None):
        info = types.SimpleNamespace(
            seq_lens_cpu=list(seq_lens),
            seq_lens_sum=sum(seq_lens),
            draft_token_num=num_draft,
        )
        fallback = list(seq_lens) if pad is None else list(seq_lens) + list(pad)
        return info, fallback

    def test_needed_equal_to_cap_admits(self):
        from sglang.srt.speculative.tree_attn_fallback import tree_verify_needed_kv

        info, fallback = self._verify_info([1009])
        needed = tree_verify_needed_kv([1009], 15)
        self.assertEqual(needed, 1024)
        self.assertTrue(
            self._admit(spec_info=info, fallback_seq_lens=fallback, slot_width=1024)
        )

    def test_needed_cap_plus_one_rejects_before_replay(self):
        from sglang.srt.speculative.tree_attn_fallback import (
            tree_slot_graph_fits,
            tree_verify_needed_kv,
        )

        needed = tree_verify_needed_kv([1010], 15)
        self.assertEqual(needed, 1025)
        self.assertFalse(tree_slot_graph_fits(needed, 1024))
        info, fallback = self._verify_info([1010])
        self.assertFalse(
            self._admit(spec_info=info, fallback_seq_lens=fallback, slot_width=1024)
        )
        dest_cols = 1024
        self.assertGreater(needed, dest_cols)

    def test_short_long_short_rehits_graph(self):
        path = []
        for seq in (100, 2000, 100):
            info, fallback = self._verify_info([seq])
            path.append(
                self._admit(
                    spec_info=info, fallback_seq_lens=fallback, slot_width=1024
                )
            )
        self.assertEqual(path, [True, False, True])

    def test_padded_batch_uses_producer_seq_lens(self):
        from sglang.srt.speculative.tree_attn_fallback import tree_verify_needed_kv

        info, fallback = self._verify_info([10, 20], pad=[9999])
        needed = tree_verify_needed_kv(
            resolve_tree_verify_mask_seq_lens(info, fallback)[0], 15
        )
        self.assertEqual(needed, 35)
        self.assertTrue(
            self._admit(spec_info=info, fallback_seq_lens=fallback, slot_width=1024)
        )
        self.assertFalse(
            self._admit(
                spec_info=info, fallback_seq_lens=fallback, slot_width=34
            )
        )

    def test_draft_bound_is_seq_plus_steps(self):
        from sglang.srt.speculative.tree_attn_fallback import tree_draft_needed_kv

        self.assertEqual(tree_draft_needed_kv([100], 5), 105)
        self.assertTrue(
            self._admit(
                is_target_verify=False,
                is_tree_draft=True,
                seq_lens=[100],
                draft_num_steps=5,
                slot_width=105,
            )
        )
        self.assertFalse(
            self._admit(
                is_target_verify=False,
                is_tree_draft=True,
                seq_lens=[100],
                draft_num_steps=5,
                slot_width=104,
            )
        )

    def test_buckets_admit_smallest_fit(self):
        info, fallback = self._verify_info([200])
        self.assertTrue(
            self._admit(
                spec_info=info,
                fallback_seq_lens=fallback,
                slot_width=1,
                buckets=[256, 512, 1024],
            )
        )
        self.assertFalse(
            self._admit(
                spec_info=info,
                fallback_seq_lens=fallback,
                slot_width=1,
                buckets=[64, 128],
            )
        )

    def test_parent_can_run_false_stays_false(self):
        self.assertTrue(self._admit(slot_gather_enabled=False, slot_width=1))

        class _Runner:
            def __init__(self, parent_ok):
                self.parent_ok = parent_ok

            def can_run(self, info):
                if not self.parent_ok:
                    return False
                return TestTreeGraphSlotCap()._admit(
                    spec_info=info,
                    fallback_seq_lens=info.seq_lens_cpu,
                    slot_width=1024,
                )

        info, _ = self._verify_info([8])
        self.assertFalse(_Runner(False).can_run(info))
        self.assertTrue(_Runner(True).can_run(info))

    def test_equal_cap_attention_matches_eager_reference(self):
        torch.manual_seed(6)
        rows, n_q, n_kv, d = 2, 4, 2, 8
        cap, page_size, num_pages = 64, 8, 16
        q = torch.randn(rows, n_q, d)
        k_cache = torch.randn(num_pages, page_size, n_kv, d)
        v_cache = torch.randn(num_pages, page_size, n_kv, d)
        slots = torch.randint(0, num_pages * page_size, (rows, cap))
        kv_lens = torch.full((rows,), cap, dtype=torch.int32)
        kwargs = dict(
            kv_slots=slots,
            kv_lens=kv_lens,
            scale=0.25,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
            qk_head_dim=d,
            v_head_dim=d,
        )
        graph_shaped = tree_draft_attention(q, k_cache, v_cache, **kwargs)
        eager = tree_draft_attention(q, k_cache, v_cache, kv_bound=cap, **kwargs)
        self.assertTrue(torch.allclose(graph_shaped, eager, atol=1e-6, rtol=1e-6))

    def test_overlap_verify_mask_keeps_orig_capacity(self):
        max_q, orig, s_cap = 15, 8207, 1024
        from sglang.srt.speculative.tree_attn_fallback import tree_graph_slot_max_kv

        slot_cols = tree_graph_slot_max_kv(orig, s_cap)
        self.assertEqual(slot_cols, 1024)
        mask_numel = max_q * orig
        long_mask = full_mask_numel([2000], 15)
        self.assertLessEqual(long_mask, mask_numel)
        self.assertGreater(long_mask, max_q * slot_cols)

    def test_init_cuda_graph_state_splits_mask_and_slots(self):
        state_src = _function_source(_ASCEND_BACKEND, "init_cuda_graph_state")
        self.assertIn("mask_max_kv = self._slot_gather_graph_max_kv()", state_src)
        self.assertIn(
            "slot_max_kv = self._slot_gather_graph_slot_max_kv()", state_src
        )
        self.assertIn("cuda_graph_custom_mask", state_src)
        self.assertIn("max_q * mask_max_kv", state_src)
        self.assertIn("cuda_graph_kv_slots", state_src)
        self.assertIn("(max_q, slot_max_kv)", state_src)
        self.assertIn("parse_tree_graph_kv_buckets", state_src)
        self.assertIn("tree_kv_buckets", state_src)
        self.assertIn("parse_tree_graph_max_kv", _ASCEND_BACKEND.read_text())


class TestTreeCompactFillAndBuckets(CustomTestCase):
    def test_vectorized_slots_match_ref(self):
        torch.manual_seed(0)
        for seq_lens, num_draft in (([1], 3), ([2, 1], 2), ([4, 8, 3], 4)):
            bs = len(seq_lens)
            rows = []
            for seq in seq_lens:
                width = seq + num_draft
                for _t in range(num_draft):
                    rows.append(torch.randint(0, 2, (width,), dtype=torch.bool))
            custom = torch.cat(rows)
            req = torch.arange(bs * 32).view(bs, 32)
            pool = torch.arange(bs)
            draft = torch.arange(100, 100 + bs * num_draft)
            a, la = build_tree_verify_kv_slots(
                custom, seq_lens, req, pool, draft, num_draft
            )
            b, lb = build_tree_verify_kv_slots_ref(
                custom, seq_lens, req, pool, draft, num_draft
            )
            self.assertTrue(torch.equal(a, b), msg=str(seq_lens))
            self.assertTrue(torch.equal(la, lb), msg=str(seq_lens))

    def test_hot_path_has_no_per_query_python_filter(self):
        src = _function_source(_FALLBACK, "build_tree_verify_kv_slots")
        self.assertNotIn("visible_token_indices", src)
        self.assertNotIn("iter_full_mask_rows", src)
        self.assertNotIn(".item()", src)
        self.assertIn("cumsum", src)
        self.assertIn("scatter_", src)

    def test_invalid_columns_do_not_clobber_compact_slots(self):
        custom = torch.tensor(
            [
                True, False, True, False,
                False, False, False, False,
            ],
            dtype=torch.bool,
        )
        req = torch.tensor([[10, 11, 12, 13]], dtype=torch.int64)
        draft = torch.tensor([20, 21], dtype=torch.int64)
        slots, lens = build_tree_verify_kv_slots(
            custom, [2], req, torch.tensor([0]), draft, 2
        )
        self.assertEqual(int(lens[0]), 2)
        self.assertEqual(slots[0, :2].tolist(), [10, 20])
        self.assertEqual(slots[0, 2:].tolist(), [0, 0])

    def test_overflow_width_raises_before_truncating(self):
        custom = torch.ones(full_mask_numel([8], 2), dtype=torch.bool)
        req = torch.arange(16).view(1, 16)
        draft = torch.tensor([100, 101])
        with self.assertRaises(RuntimeError):
            build_tree_verify_kv_slots(
                custom, [8], req, torch.tensor([0]), draft, 2, max_kv=4
            )

    def test_gather_kv_into_matches_index_select(self):
        torch.manual_seed(1)
        k = torch.randn(16, 2, 4)
        v = torch.randn(16, 2, 4)
        slots = torch.randint(0, 16, (3, 5))
        k_out = torch.zeros(3, 5, 2, 4)
        v_out = torch.zeros(3, 5, 2, 4)
        gather_kv_into(k, v, slots, k_out, v_out)
        ref_k, ref_v = gather_kv_by_slots(k, v, slots)
        self.assertTrue(torch.equal(k_out.reshape_as(ref_k), ref_k))
        self.assertTrue(torch.equal(v_out.reshape_as(ref_v), ref_v))
        src = _function_source(_FALLBACK, "gather_kv_into")
        self.assertIn("out=k_view", src)
        self.assertIn("out=v_view", src)
        self.assertNotIn("scratch.copy_", src)

    def test_zero_padding_clears_empty_rows(self):
        k_out = torch.ones(2, 3, 1, 2)
        v_out = torch.ones(2, 3, 1, 2)
        zero_gathered_kv_padding(k_out, v_out, torch.tensor([2, 0]))
        self.assertEqual(float(k_out[1].abs().sum()), 0.0)
        self.assertEqual(float(k_out[0, 2].abs().sum()), 0.0)
        self.assertGreater(float(k_out[0, :2].abs().sum()), 0.0)

    def test_buckets_clip_select_and_fia_lens(self):
        self.assertEqual(
            parse_tree_graph_kv_buckets(None, orig_max_kv=1024),
            [256, 512, 1024],
        )
        self.assertEqual(
            parse_tree_graph_kv_buckets("256,512,1024,2048", orig_max_kv=300),
            [256, 300],
        )
        with self.assertRaises(ValueError):
            parse_tree_graph_kv_buckets("0,256", orig_max_kv=1024)
        self.assertEqual(select_tree_kv_bucket(200, [256, 512, 1024]), 256)
        self.assertIsNone(select_tree_kv_bucket(2048, [256, 512]))
        self.assertEqual(
            tree_fia_actual_seq_lengths_kv(torch.tensor([0, 3]), capture_rows=4),
            [1, 3, 1, 1],
        )
        self.assertEqual(parse_tree_draft_capture_bs(None), [1, 2])
        self.assertTrue(tree_compact_fia_layout_supported(use_mla=False, has_rope_split=False))
        self.assertFalse(tree_compact_fia_layout_supported(use_mla=True, has_rope_split=False))

    def test_fia_update_list_matches_full_records(self):
        spec_utils = _REPO_ROOT / "python/sglang/srt/speculative/spec_utils.py"
        src = _function_source(spec_utils, "expand_fia_cpu_update_inputs")
        ns = {}
        exec(src, ns)
        updates = ns["expand_fia_cpu_update_inputs"](
            [[2, 2, 2], [3, 3, 3]], 4, "actual_seq_lengths_kv"
        )
        self.assertEqual(len(updates), 8)
        self.assertEqual(updates[0]["actual_seq_lengths_kv"], [2, 2, 2])
        self.assertEqual(updates[4]["actual_seq_lengths_kv"], [3, 3, 3])
        replay_src = _function_source(
            _REPO_ROOT
            / "python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py",
            "_replay",
        )
        self.assertIn("len(cpu_update_input) != n_records", replay_src)

    def test_cuda_graph_capture_has_optional_extra_dim(self):
        extra_src = _function_source(_CUDA_GRAPH_RUNNER, "_capture_extra_keys")
        self.assertIn("return [None]", extra_src)
        key_src = _function_source(_CUDA_GRAPH_RUNNER, "_make_graph_key")
        self.assertIn("extra", key_src)
        captured_src = _function_source(_CUDA_GRAPH_RUNNER, "_graph_key_captured")
        self.assertIn("_s", captured_src)
        can_src = _function_source(_CUDA_GRAPH_RUNNER, "can_run")
        self.assertIn("_graph_key_captured", can_src)
        capture_src = _function_source(_CUDA_GRAPH_RUNNER, "capture")
        self.assertIn("_capture_extra_keys", capture_src)
        self.assertIn("_active_capture_extra", capture_src)

    def test_compact_fia_backend_wiring(self):
        compact_src = _function_source(_ASCEND_BACKEND, "_run_tree_compact_fia")
        self.assertIn("gather_kv_into", compact_src)
        self.assertIn('input_layout="BSND"', compact_src)
        self.assertIn("sparse_mode=0", compact_src)
        self.assertIn("get_max_workspace", compact_src)
        self.assertIn("npu_fused_infer_attention_score.out", compact_src)
        self.assertNotIn("block_table", compact_src)
        verify_src = _function_source(_ASCEND_BACKEND, "_run_tree_verify_slot_gather")
        self.assertIn("_run_tree_compact_fia", verify_src)
        self.assertIn("tree_verify_attention", verify_src)
        draft_src = _function_source(_ASCEND_BACKEND, "_run_tree_draft_slot_gather")
        self.assertIn("_run_tree_compact_fia", draft_src)


if __name__ == "__main__":
    unittest.main()
