"""CPU tests for paged-tree draft cache mapping and buffer split.

This file avoids importing the full sglang runtime (Triton kernels, torchvision).
Helpers are executed from the live ``spec_utils.py`` source; prefix-sum and
kernel-signature checks are source-guards on the listed files.
"""

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

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_SPEC_UTILS = _REPO_ROOT / "python/sglang/srt/speculative/spec_utils.py"
_ALLOCATOR = _REPO_ROOT / "python/sglang/srt/mem_cache/allocator.py"
_HELPER_NAMES = (
    "paged_tree_mapping_end",
    "paged_tree_mapping_extra",
    "req_to_token_extra_context_len",
    "paged_tree_mapping_fits",
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
    ns = {"torch": torch}
    exec(compile(ast.Module(body, type_ignores=[]), str(_SPEC_UTILS), "exec"), ns)
    return ns


_HELPERS = _load_helpers()
paged_tree_mapping_end = _HELPERS["paged_tree_mapping_end"]
req_to_token_extra_context_len = _HELPERS["req_to_token_extra_context_len"]
split_draft_cache_locs = _HELPERS["split_draft_cache_locs"]


def _function_source(path: pathlib.Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(path.read_text(), node) or ast.unparse(node)
    raise AssertionError(f"{name} not found in {path}")


class TestPagedTreeDraftCache(CustomTestCase):
    def test_req_to_token_extra_is_branch_pages_not_draft_tokens(self):
        extra = req_to_token_extra_context_len(15, 128, 3, 5)
        self.assertGreaterEqual(extra, 3 * 2 * 128)
        self.assertNotEqual(extra, 4 + 15)

    def test_paged_tree_mapping_end_fits_widened_table(self):
        self.assertLessEqual(paged_tree_mapping_end(512, 128, 3, 5), 262163)
        near_limit = paged_tree_mapping_end(262016, 128, 3, 5)
        self.assertGreater(near_limit, 262144 + 19)
        self.assertLessEqual(near_limit, 262144 + 768)

    def test_prefix_sum_loads_use_other_zero(self):
        assign_src = _function_source(_SPEC_UTILS, "assign_draft_cache_locs")
        self.assertIn("raw_cache_loc", assign_src)
        self.assertIn("draft_cache_loc", assign_src)
        self.assertIn("mask=bs_offset < pid, other=0", assign_src)

        target_src = _function_source(_SPEC_UTILS, "get_target_cache_loc")
        self.assertIn("mask=bs_offset < bid, other=0", target_src)

        filter_src = _function_source(_SPEC_UTILS, "filter_finished_cache_loc_kernel")
        self.assertIn("mask=bs_offset < bid, other=0", filter_src)

        extend_src = _function_source(_ALLOCATOR, "alloc_extend_kernel")
        self.assertIn("mask=load_offset <= pid, other=0", extend_src)

        decode_src = _function_source(_ALLOCATOR, "alloc_decode_kernel")
        self.assertIn("mask=load_offset <= pid, other=0", decode_src)

    def test_shared_buffer_compact_write_corrupts_pid0_raw(self):
        topk, steps = 3, 5
        extend0 = 256
        raw = torch.arange(extend0 + 256, dtype=torch.int32)
        snapshot = raw.clone()
        ptr = 1 * topk * steps
        payload = torch.tensor([-1, -2, -3], dtype=torch.int32)
        raw[ptr : ptr + payload.numel()] = payload
        self.assertFalse(torch.equal(raw[:extend0], snapshot[:extend0]))

        raw_split = snapshot.clone()
        _, draft = split_draft_cache_locs(raw_split, 2, topk, steps, page_size=128)
        draft[ptr : ptr + payload.numel()] = payload
        self.assertTrue(torch.equal(raw_split, snapshot))
        self.assertEqual(draft[ptr : ptr + payload.numel()].tolist(), [-1, -2, -3])


if __name__ == "__main__":
    unittest.main()
