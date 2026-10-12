"""CPU contracts for MHA FIA KV writes reusing the ATB paged slot layout.

Scatter's flat token index and ``_npu_reshape_and_cache``'s
``slot // page_size`` / ``slot % page_size`` address the same ND bytes.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

_REPO = pathlib.Path(__file__).resolve().parents[4]
_POOL = _REPO / "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py"
_BACKEND = _REPO / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
_DRAFTER = (
    _REPO / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
)


def _fn_source_at(path: pathlib.Path, class_name: str, name: str) -> str:
    text = path.read_text()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    return ast.get_source_segment(text, child) or ast.unparse(child)
    raise AssertionError(f"{class_name}.{name} not found in {path}")


def _fn_source(class_name: str, name: str) -> str:
    return _fn_source_at(_POOL, class_name, name)


class TestFiaPagedSlotLayout(CustomTestCase):
    def test_flat_slot_matches_page_offset_and_bsnd_view(self):
        page_size, heads, dim = 128, 4, 8
        num_pages = 3
        base = torch.zeros((num_pages + 1, page_size, heads, dim))
        fia = base.view(-1, 1, heads, dim)
        paged = fia.view(-1, page_size, heads, dim)
        attn = fia.view(-1, page_size, heads * dim)

        self.assertEqual(tuple(fia.shape), ((num_pages + 1) * page_size, 1, heads, dim))
        self.assertEqual(tuple(paged.shape), tuple(base.shape))
        self.assertEqual(paged.data_ptr(), base.data_ptr())
        self.assertEqual(attn.data_ptr(), base.data_ptr())
        self.assertEqual(tuple(attn.shape), (num_pages + 1, page_size, heads * dim))

        slots = torch.tensor([0, 1, 127, 128, 130, (num_pages + 1) * page_size - 1])
        src = torch.arange(slots.numel() * heads * dim, dtype=torch.float32).reshape(
            slots.numel(), heads, dim
        )
        flat = torch.zeros_like(base).view(-1, 1, heads, dim)
        flat[slots] = src.view(-1, 1, heads, dim)

        paged_write = torch.zeros_like(base)
        for i, slot in enumerate(slots.tolist()):
            paged_write[slot // page_size, slot % page_size] = src[i]

        self.assertTrue(torch.equal(flat.view_as(base), paged_write))
        self.assertTrue(
            torch.equal(
                flat.view(-1, page_size, heads * dim),
                paged_write.view(-1, page_size, heads * dim),
            )
        )

    def test_flattened_key_reshape_matches_slot_bytes(self):
        page_size, heads, dim = 16, 2, 4
        num_tokens = 5
        key = torch.arange(num_tokens * heads * dim, dtype=torch.float32).reshape(
            num_tokens, heads * dim
        )
        slots = torch.tensor([1, 16, 17, 31, 32], dtype=torch.int64)
        cache = torch.zeros((4, page_size, heads, dim))
        packed = key.reshape(-1, heads, dim)
        self.assertTrue(packed.is_contiguous())
        for i, slot in enumerate(slots.tolist()):
            cache[slot // page_size, slot % page_size] = packed[i]
        flat = cache.view(-1, 1, heads, dim)
        for i, slot in enumerate(slots.tolist()):
            self.assertTrue(torch.equal(flat[slot, 0], key[i].view(heads, dim)))


class TestFiaKvWriteSource(CustomTestCase):
    def test_mha_fia_uses_reshape_and_cache_for_nd_fp16(self):
        create = _fn_source("NPUMHATokenToKVPool", "_create_buffers")
        self.assertIn("view(\n                        -1, 1, self.head_num, self.head_dim", create)
        self.assertIn("_fia_kv_nd", create)
        self.assertIn("ACL_FORMAT_FRACTAL_NZ", create)

        gate = _fn_source("NPUMHATokenToKVPool", "_fia_kv_use_reshape_and_cache")
        self.assertIn("k_scale is not None or v_scale is not None", gate)
        self.assertIn("self.store_dtype != self.dtype", gate)
        self.assertIn("torch.float16, torch.bfloat16", gate)
        self.assertIn("_fia_kv_nd", gate)

        write = _fn_source("NPUMHATokenToKVPool", "set_kv_buffer")
        self.assertIn("_fia_kv_use_reshape_and_cache(k_scale, v_scale)", write)
        self.assertIn("_reshape_and_cache_fia_kv", write)
        self.assertIn("_scatter_fia_kv", write)

        reshape = _fn_source("NPUMHATokenToKVPool", "_reshape_and_cache_fia_kv")
        self.assertIn("_npu_reshape_and_cache", reshape)
        self.assertIn("reshape(-1, self.head_num, self.head_dim)", reshape)
        self.assertIn(
            "view(\n                -1, self.page_size, self.head_num, self.head_dim",
            reshape,
        )
        self.assertIn("slots.dtype != torch.int32", reshape)
        self.assertLess(reshape.find("slots.dtype != torch.int32"), reshape.find(".to("))

        scatter = _fn_source("NPUMHATokenToKVPool", "_scatter_fia_kv")
        self.assertIn("npu_scatter_nd_update_", scatter)
        self.assertNotIn("_npu_reshape_and_cache", scatter)

    def test_mla_pool_keeps_scatter(self):
        mla = _fn_source("NPUMLATokenToKVPool", "set_kv_buffer")
        self.assertIn("npu_scatter_nd_update_", mla)
        self.assertNotIn("_npu_reshape_and_cache", mla)

    def test_draft_tree_fia_reuses_workspace_out(self):
        buffers = _fn_source_at(
            _BACKEND, "AscendAttnBackend", "_sr_tree_paged_fia_buffers"
        )
        self.assertIn(
            "_npu_fused_infer_attention_score_get_max_workspace", buffers
        )
        self.assertIn("torch.empty(1,", buffers)
        self.assertIn("if not self.graph_mode", buffers)
        run = _fn_source_at(_BACKEND, "AscendAttnBackend", "_run_sr_tree_paged_fia")
        self.assertIn("npu_fused_infer_attention_score.out", run)
        self.assertIn("out=[output, softmax_lse]", run)
        self.assertIn('input_layout="BSND"', run)
        self.assertNotIn("torch.ops.npu.npu_fused_infer_attention_score(", run)

    def test_draft_forward_casts_slots_once(self):
        src = _fn_source_at(_DRAFTER, "SRTreeDrafter", "_draft_forward")
        cast_at = src.find("kv_slots.to(dtype=torch.int32)")
        loop_at = src.find("for i in range(self.speculative_num_steps)")
        self.assertGreaterEqual(cast_at, 0)
        self.assertLess(cast_at, loop_at)
        self.assertIn(
            "_remap_tree_kv_to_parents(out_cache_loc, parent_rows, i)", src
        )
        self.assertIn("forward_batch.out_cache_loc = kv_slots[i]", src)
        self.assertIn("self._last_out_cache_loc = out_cache_loc", src)


if __name__ == "__main__":
    unittest.main()
