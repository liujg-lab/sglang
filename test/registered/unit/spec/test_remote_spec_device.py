"""Device-agnostic helpers for SPECTRE / STANDALONE_REMOTE dual-backend."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")

_REPO = Path(__file__).resolve().parents[4]


class TestRemoteSpecDevice(CustomTestCase):
    def _import_or_skip(self, fn):
        try:
            return fn()
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")
    def test_idle_input_uses_explicit_device(self):
        try:
            from sglang.srt.speculative.eagle_info import EagleVerifyInput
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")
        idle = EagleVerifyInput.create_idle_input(
            topk=2, spec_steps=3, num_verify_tokens=4, device="cpu"
        )
        self.assertEqual(idle.draft_token.device.type, "cpu")
        self.assertEqual(idle.custom_mask.device.type, "cpu")
        self.assertEqual(idle.retrive_index.shape[1], 4)

    def test_mm_serialize_moves_non_cpu_fake_via_cpu_tensor(self):
        try:
            from sglang.srt.speculative.spec_utils import tensor_on_accelerator
            from sglang.srt.speculative.spectre.spectre_mm_transport import (
                _tensor_gpu_bytes,
                _to_cpu_contiguous_tensor,
            )
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")

        tensor = torch.arange(4, dtype=torch.float32)
        out = _to_cpu_contiguous_tensor(tensor)
        self.assertEqual(out.device.type, "cpu")
        self.assertTrue(out.is_contiguous())
        self.assertEqual(_tensor_gpu_bytes(tensor), 0)
        self.assertFalse(tensor_on_accelerator(tensor))

        fake_npu = MagicMock(spec=torch.Tensor)
        fake_npu.device = SimpleNamespace(type="npu")
        fake_npu.numel.return_value = 8
        fake_npu.element_size.return_value = 4
        self.assertEqual(_tensor_gpu_bytes(fake_npu), 32)
        self.assertTrue(tensor_on_accelerator(fake_npu))

        fake_cuda = MagicMock(spec=torch.Tensor)
        fake_cuda.device = SimpleNamespace(type="cuda")
        fake_cuda.numel.return_value = 2
        fake_cuda.element_size.return_value = 2
        self.assertEqual(_tensor_gpu_bytes(fake_cuda), 4)

    def test_sr_mm_serialize_cpu(self):
        try:
            from sglang.srt.speculative.standalone_remote.sr_mm_payload import (
                _to_cpu_contiguous_tensor as sr_to_cpu,
            )
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")

        tensor = torch.ones(2, 2)
        out = sr_to_cpu(tensor)
        self.assertEqual(out.device.type, "cpu")

    def test_capability_table(self):
        try:
            from sglang.srt.speculative.spec_utils import (
                device_backend_key,
                tree_verify_method_available,
            )
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")

        self.assertTrue(tree_verify_method_available("greedy", "npu"))
        self.assertTrue(tree_verify_method_available("target_only", "npu"))
        self.assertTrue(tree_verify_method_available("target_only", "cuda"))
        self.assertFalse(tree_verify_method_available("target_only", "hip"))
        self.assertTrue(tree_verify_method_available("rpd", "npu"))
        self.assertEqual(device_backend_key("npu:0"), "npu")
        self.assertEqual(device_backend_key(torch.device("cpu")), "cpu")

    def test_page_aligned_rollback(self):
        try:
            from sglang.srt.speculative.standalone_remote.sr_kv_rollbacker import (
                SRKVRollbacker,
            )
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")

        pool = MagicMock()
        req_to_token = torch.arange(16, dtype=torch.int32).view(1, 16)
        pool.req_to_token = req_to_token
        allocator = MagicMock()
        rb = SRKVRollbacker(
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=pool,
            tree_cache=MagicMock(),
            page_size=4,
            tp_rank=0,
        )
        req = SimpleNamespace(
            prefix_indices=[],
            req_pool_idx=0,
            kv_allocated_len=12,
            kv_committed_len=12,
            rid="r0",
        )
        self.assertFalse(rb.can_local_rollback(req, fork_point=5))
        self.assertTrue(rb.can_local_rollback(req, fork_point=8))
        ok = rb.local_rollback(req, fork_point=8, current_kv_len=12)
        self.assertTrue(ok)
        allocator.free.assert_called_once()
        freed = allocator.free.call_args[0][0]
        self.assertEqual(freed.tolist(), [8, 9, 10, 11])
        self.assertEqual(req.kv_committed_len, 8)

    def test_device_context_error_classifier(self):
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
                _sr_is_device_context_error,
            )
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")

        class NPUError(Exception):
            pass

        self.assertTrue(_sr_is_device_context_error(NPUError("illegal memory access")))
        self.assertTrue(_sr_is_device_context_error(RuntimeError("NPU error: illegal")))
        self.assertFalse(_sr_is_device_context_error(RuntimeError("rpc timeout")))

    def test_npu_graph_runner_replay_uses_make_graph_key(self):
        src = (
            _REPO
            / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py"
        ).read_text()
        self.assertIn("self._make_graph_key", src)
        self.assertIn("NPU graph miss", src)
        self.assertIn("actual_ntpb", src)
        self.assertNotIn("self.graphs[self.bs].replay()", src)

    def test_page_physical_kv_copy_matches_slot_to_page_offset(self):
        page_size = 4
        buf = torch.arange(2 * 1 * 3 * 4 * 1 * 2, dtype=torch.float32).reshape(
            2, 1, 3, 4, 1, 2
        )
        src = torch.tensor([5, 6], dtype=torch.int32)
        tgt = torch.tensor([9, 10], dtype=torch.int32)
        src_page = torch.div(src, page_size, rounding_mode="floor")
        src_off = src % page_size
        tgt_page = torch.div(tgt, page_size, rounding_mode="floor")
        tgt_off = tgt % page_size
        expected = buf[:, :, src_page, src_off, :, :].clone()
        buf[:, :, tgt_page, tgt_off, :, :] = buf[:, :, src_page, src_off, :, :]
        torch.testing.assert_close(buf[:, :, tgt_page, tgt_off, :, :], expected)
        npu_src = (
            _REPO / "python/sglang/srt/hardware_backend/npu/memory_pool_npu.py"
        ).read_text()
        self.assertIn("def move_kv_cache", npu_src)
        self.assertIn("tgt_page", npu_src)
        self.assertNotIn("copy_all_layer_kv_cache_tiled", npu_src)
        self.assertIn("enable_kv_cache_copy=False", npu_src)
        self.assertNotIn("enable_kv_cache_copy=enable_kv_cache_copy", npu_src)
        self.assertIn("def _init_kv_copy_and_warmup", npu_src)

    def test_npu_tree_draft_triton_and_kv_restore_source_guards(self):
        spec_src = (
            _REPO / "python/sglang/srt/speculative/spec_utils.py"
        ).read_text()
        self.assertNotIn(
            "if page_size != 1 and topk != 1 and duplicate_cache_len > 0:",
            spec_src,
        )
        self.assertIn(
            "if ((page_size != 1) and (topk != 1)) and (duplicate_cache_len > 0):",
            spec_src,
        )
        drafter_src = (
            _REPO
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
        ).read_text()
        self.assertIn("def _alloc_tree_kv", drafter_src)
        self.assertIn("if token_to_kv_pool_state_backup is not None:", drafter_src)
        self.assertIn("except Exception:", drafter_src)

    def test_expand_seq_lens_for_spec_topk(self):
        try:
            from sglang.srt.speculative.spec_utils import (
                expand_seq_lens_for_spec_topk,
            )
        except Exception as e:
            self.skipTest(f"sglang runtime deps missing: {e}")

        self.assertEqual(
            expand_seq_lens_for_spec_topk([10, 20], 6),
            [10, 10, 10, 20, 20, 20],
        )
        self.assertEqual(expand_seq_lens_for_spec_topk([10, 20], 2), [10, 20])
        self.assertEqual(expand_seq_lens_for_spec_topk([7], 4), [7])

    def test_eagle_verify_refuses_silent_greedy_for_remote_spec(self):
        eagle_src = (
            _REPO / "python/sglang/srt/speculative/eagle_info.py"
        ).read_text()
        self.assertIn("Refusing to fall back to greedy", eagle_src)
        self.assertIn("is_remote_spec_algorithm", eagle_src)

    def test_npu_does_not_call_chain_sgl_kernel_npu_greedy(self):
        src = (_REPO / "python/sglang/srt/speculative/eagle_utils.py").read_text()
        self.assertIn("verify_tree_greedy_ref", src)
        self.assertNotIn("sgl_kernel_npu", src)


if __name__ == "__main__":
    unittest.main()
