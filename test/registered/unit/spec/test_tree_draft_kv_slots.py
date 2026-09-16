"""CPU tests for token-level tree-draft KV slots and batched draft attention.

Helpers are AST-loaded from ``spec_utils.py`` so this file does not import
the full runtime (Triton / torchvision). Attention math comes from
``tree_attn_fallback``.
"""

from __future__ import annotations

import ast
import math
import pathlib
import unittest

import torch

from sglang.srt.speculative.tree_attn_fallback import (
    chunked_attend,
    flatten_paged_kv,
    tree_draft_attention,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_SPEC_UTILS = _REPO_ROOT / "python/sglang/srt/speculative/spec_utils.py"
_ASCEND_BACKEND = (
    _REPO_ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
)
_NPU_GRAPH = (
    _REPO_ROOT
    / "python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py"
)
_SR_TREE = (
    _REPO_ROOT
    / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
)
_HELPER_NAMES = (
    "_raise_tree_draft_pos_overflow",
    "build_tree_draft_kv_slots",
    "build_paged_draft_cache_locs",
    "split_draft_cache_locs",
)


def _load_helpers():
    tree = ast.parse(_SPEC_UTILS.read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _HELPER_NAMES
    ]
    names = {node.name for node in body}
    missing = set(_HELPER_NAMES) - names
    if missing:
        raise AssertionError(f"missing helpers in spec_utils.py: {sorted(missing)}")
    future = ast.ImportFrom(
        module="__future__",
        names=[ast.alias(name="annotations", asname=None)],
        level=0,
    )
    mod = ast.Module(body=[future] + body, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"torch": torch}
    exec(compile(mod, str(_SPEC_UTILS), "exec"), ns)
    return ns


_HELPERS = _load_helpers()
build_tree_draft_kv_slots = _HELPERS["build_tree_draft_kv_slots"]
build_paged_draft_cache_locs = _HELPERS["build_paged_draft_cache_locs"]
split_draft_cache_locs = _HELPERS["split_draft_cache_locs"]


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


def _ref_draft_pos(seq, k, draft_i, page_size, topk, num_steps):
    if topk == 1 or page_size == 1:
        return seq + k * num_steps + draft_i
    last_page_len = seq % page_size
    prefix_base = seq - last_page_len
    nnp = (last_page_len + num_steps + page_size - 1) // page_size
    return prefix_base + k * nnp * page_size + last_page_len + draft_i


def _ref_tree_draft_kv_slots(
    req_to_token, pool_idx, seq_lens, page_size, topk, step_id, num_steps
):
    rows = []
    lens = []
    for b, seq in enumerate(seq_lens):
        seq = int(seq)
        pool = int(pool_idx[b])
        kv_len = seq + step_id + 1
        for k in range(topk):
            slots = []
            for j in range(kv_len):
                if j < seq:
                    pos = j
                else:
                    pos = _ref_draft_pos(
                        seq, k, j - seq, page_size, topk, num_steps
                    )
                slots.append(int(req_to_token[pool, pos]))
            rows.append(slots)
            lens.append(kv_len)
    return rows, lens


def _ref_paged_draft_cache_locs(
    req_to_token, pool_idx, seq_lens, num_new_pages, topk, num_steps, page_size
):
    out = []
    for b, seq in enumerate(seq_lens):
        seq = int(seq)
        pool = int(pool_idx[b])
        last_page_len = seq % page_size
        prefix_base = seq - last_page_len
        nnp = int(num_new_pages[b])
        for k in range(topk):
            for i in range(num_steps):
                pos = prefix_base + k * nnp * page_size + last_page_len + i
                out.append(int(req_to_token[pool, pos]))
    return out


class TestTreeDraftKvSlots(CustomTestCase):
    def test_slots_match_generate_draft_decode_formula(self):
        cases = []
        for topk in (1, 3):
            for page_size in (1, 16, 128):
                for seq in (0, 1, page_size - 1, page_size, page_size + 1):
                    if page_size == 1 and seq > 4:
                        continue
                    cases.append((topk, page_size, seq))
        num_steps = 2
        for topk, page_size, seq in cases:
            pool_len = max(paged_end(seq, page_size, topk, num_steps), seq + topk * num_steps) + 8
            req_to_token = torch.arange(pool_len, dtype=torch.int32).unsqueeze(0)
            pool_idx = torch.tensor([0], dtype=torch.int64)
            seq_t = torch.tensor([seq], dtype=torch.int64)
            for step_id in range(num_steps):
                slots, lens = build_tree_draft_kv_slots(
                    req_to_token,
                    pool_idx,
                    seq_t,
                    page_size,
                    topk,
                    step_id,
                    num_steps,
                )
                ref_rows, ref_lens = _ref_tree_draft_kv_slots(
                    req_to_token, pool_idx, [seq], page_size, topk, step_id, num_steps
                )
                self.assertEqual(lens.tolist(), ref_lens)
                for i, ref in enumerate(ref_rows):
                    self.assertEqual(
                        slots[i, : len(ref)].tolist(),
                        ref,
                        f"topk={topk} page={page_size} seq={seq} step={step_id} row={i}",
                    )

    def test_page128_last_page_almost_full_crosses_page(self):
        page_size, topk, num_steps, seq = 128, 3, 2, 127
        nnp = (seq % page_size + num_steps + page_size - 1) // page_size
        self.assertEqual(nnp, 2)
        pool_len = seq - (seq % page_size) + topk * nnp * page_size + 4
        req_to_token = torch.arange(pool_len, dtype=torch.int32).unsqueeze(0)
        slots, lens = build_tree_draft_kv_slots(
            req_to_token,
            torch.tensor([0]),
            torch.tensor([seq]),
            page_size,
            topk,
            step_id=1,
            num_steps=num_steps,
        )
        self.assertEqual(lens.tolist(), [seq + 2] * topk)
        ref_rows, _ = _ref_tree_draft_kv_slots(
            req_to_token, [0], [seq], page_size, topk, 1, num_steps
        )
        for i, ref in enumerate(ref_rows):
            self.assertEqual(slots[i, : len(ref)].tolist(), ref)

    def test_prefix_reads_original_slots_not_branch_page(self):
        page_size, topk, num_steps, seq = 4, 3, 2, 5
        last = seq % page_size
        prefix_base = seq - last
        nnp = (last + num_steps + page_size - 1) // page_size
        pool_len = prefix_base + topk * nnp * page_size + 4
        req_to_token = torch.full((1, pool_len), -7, dtype=torch.int32)
        req_to_token[0, :seq] = torch.arange(100, 100 + seq)
        req_to_token[0, seq:] = torch.arange(500, 500 + pool_len - seq)
        slots, _ = build_tree_draft_kv_slots(
            req_to_token,
            torch.tensor([0]),
            torch.tensor([seq]),
            page_size,
            topk,
            step_id=0,
            num_steps=num_steps,
        )
        for k in range(topk):
            self.assertEqual(slots[k, :seq].tolist(), list(range(100, 100 + seq)))

    def test_overflow_raises(self):
        req_to_token = torch.arange(8, dtype=torch.int32).unsqueeze(0)
        with self.assertRaises(RuntimeError) as ctx:
            build_tree_draft_kv_slots(
                req_to_token,
                torch.tensor([0]),
                torch.tensor([6]),
                page_size=4,
                topk=3,
                step_id=1,
                num_steps=2,
            )
        self.assertIn("out of req_to_token range", str(ctx.exception))


def paged_end(seq, page_size, topk, num_steps):
    if page_size <= 1 or topk <= 1:
        return seq + topk * num_steps
    last = seq % page_size
    nnp = (last + num_steps + page_size - 1) // page_size
    return seq - last + topk * nnp * page_size


class TestBuildPagedDraftCacheLocs(CustomTestCase):
    def test_matches_part3_python_reference(self):
        cases = [
            (1, 4, 5, 7, 8),
            (3, 4, 5, 7, 8),
            (3, 4, 2, 8, 5),
            (3, 16, 4, 12, 1),
            (3, 128, 2, 127, 1),
            (3, 128, 2, 128, 1),
            (3, 128, 2, 1, 1),
        ]
        for topk, page_size, num_steps, seq, bs in cases:
            seqs = [seq + b for b in range(bs)]
            nnp = [
                (s % page_size + num_steps + page_size - 1) // page_size for s in seqs
            ]
            pool_len = max(paged_end(s, page_size, topk, num_steps) for s in seqs) + 4
            req_to_token = torch.arange(
                bs * pool_len, dtype=torch.int32
            ).view(bs, pool_len)
            pool_idx = torch.arange(bs, dtype=torch.int64)
            seq_t = torch.tensor(seqs, dtype=torch.int64)
            nnp_t = torch.tensor(nnp, dtype=torch.int64)
            got = build_paged_draft_cache_locs(
                req_to_token, pool_idx, seq_t, nnp_t, topk, num_steps, page_size
            )
            ref = _ref_paged_draft_cache_locs(
                req_to_token, pool_idx, seqs, nnp, topk, num_steps, page_size
            )
            self.assertEqual(
                got.tolist(),
                ref,
                f"topk={topk} page={page_size} steps={num_steps} seqs={seqs}",
            )

    def test_split_draft_cache_locs_fills_minus_one(self):
        raw = torch.arange(16, dtype=torch.int32)
        raw_out, draft = split_draft_cache_locs(raw, 1, 3, 2, page_size=128)
        self.assertTrue(torch.equal(raw_out, raw))
        self.assertEqual(draft.numel(), 6)
        self.assertTrue(torch.equal(draft, torch.full_like(draft, -1)))


class TestTreeDraftAttention(CustomTestCase):
    def test_matches_chunked_attend_gqa_and_padding(self):
        torch.manual_seed(0)
        n_q, n_kv, d = 4, 2, 8
        rows, max_kv = 3, 6
        lens = [4, 6, 2]
        k_cache = torch.randn(2, 8, n_kv * d)
        v_cache = torch.randn(2, 8, n_kv * d)
        query = torch.randn(rows, n_q, d)
        kv_slots = torch.tensor(
            [
                [0, 1, 2, 3, 0, 0],
                [1, 2, 3, 4, 5, 6],
                [7, 8, 0, 0, 0, 0],
            ],
            dtype=torch.int64,
        )
        scale = 1.0 / math.sqrt(d)
        out = tree_draft_attention(
            query,
            k_cache,
            v_cache,
            kv_slots=kv_slots,
            kv_lens=lens,
            scale=scale,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
            qk_head_dim=d,
            v_head_dim=d,
        )
        flat_k = flatten_paged_kv(k_cache, n_kv, d)
        flat_v = flatten_paged_kv(v_cache, n_kv, d)
        for i, kv_len in enumerate(lens):
            slots = kv_slots[i, :kv_len]
            k_vis = flat_k.index_select(0, slots)
            v_vis = flat_v.index_select(0, slots)
            ref = chunked_attend(query[i], k_vis, v_vis, scale, chunk_size=2)
            self.assertTrue(
                torch.allclose(out[i].view(n_q, d).float(), ref.float(), atol=1e-4, rtol=1e-4),
                f"row {i} mismatch",
            )

    def test_gpu_kv_lens_matches_list(self):
        torch.manual_seed(0)
        n_q, n_kv, d = 4, 2, 8
        rows = 3
        lens = [4, 6, 2]
        k_cache = torch.randn(2, 8, n_kv * d)
        v_cache = torch.randn(2, 8, n_kv * d)
        query = torch.randn(rows, n_q, d)
        kv_slots = torch.tensor(
            [
                [0, 1, 2, 3, 0, 0],
                [1, 2, 3, 4, 5, 6],
                [7, 8, 0, 0, 0, 0],
            ],
            dtype=torch.int64,
        )
        scale = 1.0 / math.sqrt(d)
        kwargs = dict(
            kv_slots=kv_slots,
            scale=scale,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
            qk_head_dim=d,
            v_head_dim=d,
        )
        out_list = tree_draft_attention(query, k_cache, v_cache, kv_lens=lens, **kwargs)
        out_t = tree_draft_attention(
            query,
            k_cache,
            v_cache,
            kv_lens=torch.tensor(lens, dtype=torch.int32),
            **kwargs,
        )
        self.assertTrue(
            torch.allclose(out_list.float(), out_t.float(), atol=1e-5, rtol=1e-5)
        )

    def test_equal_kv_lens_no_padding(self):
        torch.manual_seed(3)
        n_q, n_kv, d = 2, 2, 4
        rows, kv_len = 2, 3
        k_cache = torch.randn(1, 8, n_kv * d)
        v_cache = torch.randn(1, 8, n_kv * d)
        query = torch.randn(rows, n_q, d)
        kv_slots = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.int64)
        scale = 0.5
        out = tree_draft_attention(
            query,
            k_cache,
            v_cache,
            kv_slots=kv_slots,
            kv_lens=[kv_len, kv_len],
            scale=scale,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
            qk_head_dim=d,
            v_head_dim=d,
        )
        flat_k = flatten_paged_kv(k_cache, n_kv, d)
        flat_v = flatten_paged_kv(v_cache, n_kv, d)
        for i in range(rows):
            ref = chunked_attend(
                query[i],
                flat_k.index_select(0, kv_slots[i]),
                flat_v.index_select(0, kv_slots[i]),
                scale,
            )
            self.assertTrue(
                torch.allclose(out[i].view(n_q, d).float(), ref.float(), atol=1e-4, rtol=1e-4)
            )


class TestTreeDraftSlotGatherWiring(CustomTestCase):
    def test_forward_decode_short_circuits_to_slot_gather(self):
        src = _function_source(_ASCEND_BACKEND, "forward_decode")
        self.assertIn("_use_tree_draft_slot_gather", src)
        self.assertIn("_run_tree_draft_slot_gather", src)
        self.assertIn("tree_draft_kv_slots", _ASCEND_BACKEND.read_text())

    def test_init_metadata_fills_slots(self):
        src = _function_source(_ASCEND_BACKEND, "init_forward_metadata")
        self.assertIn("_fill_tree_draft_kv_slots", src)

    def test_npu_graph_slot_gather_for_paged_tree(self):
        src = _function_source(_NPU_GRAPH, "__init__")
        self.assertIn("token-level slot gather", src)
        self.assertIn("page_size > 1 and topk > 1", src)
        self.assertIn("_slot_gather_graph", src)
        self.assertNotIn("npu tree draft uses token-level slot gather (eager)", src)
        capture_src = _function_source(_NPU_GRAPH, "capture")
        self.assertIn("tree_graph_disabled_reason", capture_src)
        self.assertIn("_slot_gather_graph", capture_src)
        replay_src = _function_source(_NPU_GRAPH, "_replay")
        self.assertIn("_slot_gather_graph", replay_src)
        state_src = _function_source(_ASCEND_BACKEND, "init_cuda_graph_state")
        self.assertIn("cuda_graph_kv_slots", state_src)
        replay_meta = _function_source(
            _ASCEND_BACKEND, "init_forward_metadata_replay_cuda_graph"
        )
        self.assertIn("_fill_tree_draft_kv_slots", replay_meta)
        self.assertIn("_copy_into_graph_slot_buffers", _ASCEND_BACKEND.read_text())

    def test_sr_tree_drafter_skips_last_page_copy(self):
        src = _function_source(_SR_TREE, "_alloc_tree_kv")
        self.assertIn("build_paged_draft_cache_locs", src)
        self.assertIn("skip the copy", src.lower())
        self.assertNotIn("token_to_kv_pool.move_kv_cache", src)


if __name__ == "__main__":
    unittest.main()
