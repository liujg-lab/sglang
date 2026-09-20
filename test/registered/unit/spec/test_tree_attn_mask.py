"""CPU tests for Ascend tree-mask polarity conversion."""

import ast
import pathlib
import unittest

import torch

from sglang.srt.speculative.tree_attn_mask import (
    FIA_TREE_MASK_CONTRACT,
    FIA_TREE_MASK_CONTRACT_BSND,
    custom_mask_to_ascend_masked,
    inplace_update_graph_tree_attn_mask,
    iter_full_mask_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_ASCEND_BACKEND = (
    _REPO_ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
)


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


class TestTreeAttnMask(CustomTestCase):
    def test_cuda_true_attend_becomes_ascend_masked_false(self):
        # One request, seq_len=2, num_draft=2. Flattened FULL_MASK rows of length 4.
        # Query 0 attends prefix+self: [T, T, T, F]
        # Query 1 attends prefix+self+ancestor: [T, T, T, T]
        custom = torch.tensor(
            [True, True, True, False, True, True, True, True], dtype=torch.bool
        )
        seq_lens = torch.tensor([2], dtype=torch.int32)
        masked = custom_mask_to_ascend_masked(custom, seq_lens, num_draft=2)
        self.assertEqual(tuple(masked.shape), (2, 4))
        # Ascend True = masked
        self.assertEqual(masked[0].tolist(), [False, False, False, True])
        self.assertEqual(masked[1].tolist(), [False, False, False, False])

    def test_sibling_column_stays_masked_for_other_branch(self):
        # seq_len=1, draft=3. Query 1 cannot see draft token 2 (sibling).
        custom = torch.tensor(
            [
                True, True, False, False,  # q0: prefix + self
                True, True, True, False,  # q1: prefix + root + self, not sibling
                True, True, False, True,  # q2: prefix + root + self, not sibling 1
            ],
            dtype=torch.bool,
        )
        seq_lens = torch.tensor([1], dtype=torch.int32)
        masked = custom_mask_to_ascend_masked(custom, seq_lens, num_draft=3)
        self.assertTrue(masked[1, 3].item())  # sibling masked
        self.assertTrue(masked[2, 2].item())  # other sibling masked
        rows = list(iter_full_mask_rows(custom, [1], 3))
        self.assertEqual(len(rows), 3)

    def test_fia_contract_and_padding_is_masked(self):
        self.assertEqual(FIA_TREE_MASK_CONTRACT["ascend_true"], "masked")
        self.assertEqual(FIA_TREE_MASK_CONTRACT["sparse_mode"], 0)
        self.assertFalse(FIA_TREE_MASK_CONTRACT["fia_consumes_mask"])
        self.assertEqual(FIA_TREE_MASK_CONTRACT["layout"], "TND")
        self.assertEqual(FIA_TREE_MASK_CONTRACT_BSND["layout"], "BSND")
        self.assertEqual(FIA_TREE_MASK_CONTRACT_BSND["sparse_mode"], 0)
        self.assertTrue(FIA_TREE_MASK_CONTRACT_BSND["fia_consumes_mask"])
        self.assertIn("[B, 1, Q, S]", FIA_TREE_MASK_CONTRACT_BSND["shape"])
        # seq_len=1, num_draft=2 → two rows of length 3.
        custom = torch.tensor(
            [True, True, True, True, True, True], dtype=torch.bool
        )
        seq_lens = torch.tensor([1], dtype=torch.int32)
        masked = custom_mask_to_ascend_masked(
            custom, seq_lens, num_draft=2, max_kv=5
        )
        self.assertEqual(tuple(masked.shape), (2, 5))
        self.assertTrue(masked[:, 3:].all())

    def test_inplace_graph_mask_keeps_storage_and_sibling_padding(self):
        custom = torch.tensor(
            [
                True, True, False, False,
                True, True, True, False,
                True, True, False, True,
            ],
            dtype=torch.bool,
        )
        converted = custom_mask_to_ascend_masked(
            custom, torch.tensor([1], dtype=torch.int32), num_draft=3
        )
        buf = torch.zeros((8, 8), dtype=torch.bool)
        ptr = buf.data_ptr()
        shape = tuple(buf.shape)
        out = inplace_update_graph_tree_attn_mask(buf, converted)
        self.assertIs(out, buf)
        self.assertEqual(out.data_ptr(), ptr)
        self.assertEqual(tuple(out.shape), shape)
        self.assertTrue(out[1, 3].item())
        self.assertTrue(out[2, 2].item())
        self.assertTrue(out[:, converted.shape[1] :].all())
        self.assertTrue(out[converted.shape[0] :, :].all())

    def test_capture_does_not_bind_tree_mask_for_fia(self):
        capture_src = _function_source(
            _ASCEND_BACKEND, "init_forward_metadata_capture_cuda_graph"
        )
        self.assertNotIn(
            "metadata.tree_attn_mask = self.cuda_graph_tree_attn_mask",
            capture_src,
        )
        fill_src = _function_source(_ASCEND_BACKEND, "_fill_tree_verify_mask")
        self.assertIn("inplace_update_graph_tree_attn_mask", fill_src)
        self.assertNotIn(
            "tree_mask = self.cuda_graph_tree_attn_mask[",
            fill_src,
        )


if __name__ == "__main__":
    unittest.main()
