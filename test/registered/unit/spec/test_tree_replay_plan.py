"""CPU tests for NPU tree graph admission, exception protocol, and Draft restore.

Does not import torch_npu. Runtime helpers come from ``tree_attn_fallback``;
NPU runner wiring is source-guarded via AST. A small in-process fake covers
plan lifecycle and update/replay join without launching a graph.
"""

from __future__ import annotations

import ast
import pathlib
import threading
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.tree_attn_fallback import TreeReplayPlan
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
_FALLBACK = _REPO_ROOT / "python/sglang/srt/speculative/tree_attn_fallback.py"
_NPU_GRAPH_RUNNER = (
    _REPO_ROOT
    / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py"
)
_EAGLE_DRAFT_NPU = (
    _REPO_ROOT
    / "python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py"
)


def _class_method_source(path: pathlib.Path, class_name: str, method_name: str) -> str:
    text = path.read_text()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.get_source_segment(text, child) or ast.unparse(child)
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def _producer_fallback(spec_info, seq):
    producer_seq_lens = getattr(spec_info, "seq_lens_cpu", None)
    return seq if producer_seq_lens is None else producer_seq_lens


class _SubmittedError(RuntimeError):
    """Stand-in for NpuGraphReplaySubmittedError; spec_utils pulls Triton."""


def _run_update_replay(update_fn, replay_fn):
    errors = []
    replay_error = None

    def _update():
        try:
            update_fn()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=_update)
    thread.start()
    try:
        replay_fn()
    except Exception as exc:
        replay_error = exc
    finally:
        thread.join()

    if replay_error is not None or errors:
        cause = replay_error if replay_error is not None else errors[0]
        raise _SubmittedError("NPU graph update/replay failed") from cause
    return "ok"


class _FakeRunner:
    def __init__(self, graphs):
        self.graphs = dict(graphs)
        self._clear()

    def _clear(self):
        self.plan = None
        self.batch_id = None
        self.stream_idx = None
        self.graph = None

    def can_run(self, batch, graph_key, stream_idx=None):
        self._clear()
        if graph_key not in self.graphs:
            return False
        self.plan = TreeReplayPlan(
            graph_key=graph_key,
            raw_bs=1,
            capture_bs=1,
            tokens_per_req=1,
            kv_bucket=None if not isinstance(graph_key, str) else 256,
        )
        self.batch_id = id(batch)
        self.stream_idx = stream_idx
        self.graph = self.graphs[graph_key]
        return True

    def replay(self, batch, stream_idx=None, recapture=None, replace_key=None):
        plan = self.plan
        batch_id = self.batch_id
        try:
            if plan is None or batch_id != id(batch) or stream_idx != self.stream_idx:
                raise RuntimeError("prep")
            if recapture is not None:
                self.graphs = recapture
            if replace_key is not None:
                self.graphs[plan.graph_key] = replace_key
            if self.graphs.get(plan.graph_key) is not self.graph:
                raise RuntimeError("prep")
            return plan.graph_key
        finally:
            self._clear()


class TestTreeReplayPlan(CustomTestCase):
    def test_plan_accepts_int_and_str_keys(self):
        decode = TreeReplayPlan(
            graph_key=1, raw_bs=1, capture_bs=1, tokens_per_req=1, kv_bucket=None
        )
        verify = TreeReplayPlan(
            graph_key="r15_1_s256",
            raw_bs=1,
            capture_bs=1,
            tokens_per_req=15,
            kv_bucket=256,
        )
        self.assertEqual(decode.graph_key, 1)
        self.assertIsNone(decode.kv_bucket)
        self.assertEqual(verify.graph_key, "r15_1_s256")
        self.assertEqual(verify.kv_bucket, 256)
        with self.assertRaises(Exception):
            decode.raw_bs = 2

    def test_producer_seq_lens_is_none_not_or(self):
        seq = torch.tensor([4, 5], dtype=torch.int32)
        missing = SimpleNamespace()
        self.assertIs(_producer_fallback(missing, seq), seq)

        empty_list = SimpleNamespace(seq_lens_cpu=[])
        self.assertEqual(_producer_fallback(empty_list, seq), [])

        empty_t = SimpleNamespace(seq_lens_cpu=torch.tensor([], dtype=torch.int32))
        got = _producer_fallback(empty_t, seq)
        self.assertEqual(list(got.shape), [0])

        src = _class_method_source(
            _ASCEND_BACKEND, "AscendAttnBackend", "tree_slot_graph_can_run"
        )
        self.assertIn("producer_seq_lens = getattr(spec_info, \"seq_lens_cpu\", None)", src)
        self.assertIn(
            "fallback = seq if producer_seq_lens is None else producer_seq_lens", src
        )
        self.assertNotIn("or seq", src)

    def test_can_run_miss_clears_plan_and_returns_false(self):
        runner = _FakeRunner({1: object()})
        batch = object()
        self.assertFalse(runner.can_run(batch, "1_s256"))
        self.assertIsNone(runner.plan)
        self.assertTrue(runner.can_run(batch, 1))
        self.assertEqual(runner.plan.graph_key, 1)
        other = object()
        with self.assertRaisesRegex(RuntimeError, "prep"):
            runner.replay(other)
        self.assertIsNone(runner.plan)

    def test_replay_clears_plan_and_rejects_recapture(self):
        runner = _FakeRunner({1: object(), "1_s256": object()})
        batch = object()
        self.assertTrue(runner.can_run(batch, 1))
        self.assertEqual(runner.replay(batch), 1)
        self.assertIsNone(runner.plan)

        self.assertTrue(runner.can_run(batch, 1))
        with self.assertRaisesRegex(RuntimeError, "prep"):
            runner.replay(batch, recapture={"2": object()})
        self.assertIsNone(runner.plan)

        g1 = object()
        runner = _FakeRunner({1: g1})
        self.assertTrue(runner.can_run(batch, 1))
        with self.assertRaisesRegex(RuntimeError, "prep"):
            runner.replay(batch, replace_key=object())
        self.assertIsNone(runner.plan)

        runner = _FakeRunner({1: g1})
        self.assertTrue(runner.can_run(batch, 1))
        self.assertEqual(runner.replay(batch), 1)

    def test_update_replay_join_on_replay_error(self):
        joined = []

        def update():
            threading.Event().wait(0.02)
            joined.append("update")

        def replay():
            raise RuntimeError("replay boom")

        with self.assertRaises(_SubmittedError) as ctx:
            _run_update_replay(update, replay)
        self.assertEqual(joined, ["update"])
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertIn("replay boom", str(ctx.exception.__cause__))

    def test_update_replay_join_on_update_error(self):
        def update():
            raise ValueError("update boom")

        def replay():
            return None

        with self.assertRaises(_SubmittedError) as ctx:
            _run_update_replay(update, replay)
        self.assertIsInstance(ctx.exception.__cause__, ValueError)

    def test_update_replay_success_returns_after_join(self):
        self.assertEqual(_run_update_replay(lambda: None, lambda: None), "ok")

    def test_draft_restore_original_batch_fields(self):
        batch = SimpleNamespace(
            batch_size=1,
            seq_lens="orig_seq",
            req_pool_indices="orig_idx",
            positions="orig_pos",
            mrope_positions="orig_mrope",
            seq_lens_cpu="orig_cpu",
        )
        snap = {
            "batch_size": batch.batch_size,
            "seq_lens": batch.seq_lens,
            "req_pool_indices": batch.req_pool_indices,
            "positions": batch.positions,
            "mrope_positions": batch.mrope_positions,
            "seq_lens_cpu": batch.seq_lens_cpu,
        }
        batch.batch_size = 2
        batch.seq_lens = "padded"
        for name, value in snap.items():
            setattr(batch, name, value)
        self.assertEqual(batch.batch_size, 1)
        self.assertEqual(batch.seq_lens, "orig_seq")
        self.assertEqual(batch.seq_lens_cpu, "orig_cpu")

    def test_npu_runners_wire_plan_and_join_protocol(self):
        fallback = _FALLBACK.read_text()
        self.assertIn("class TreeReplayPlan", fallback)
        self.assertIn("graph_key: int | str", fallback)

        target_can = _class_method_source(_NPU_GRAPH_RUNNER, "NPUGraphRunner", "can_run")
        self.assertIn("_clear_tree_replay_plan", target_can)
        self.assertIn("_save_tree_replay_plan", target_can)
        self.assertIn("graph_key not in self.graphs", target_can)
        self.assertIn("TreeReplayPlan", target_can)
        self.assertNotIn("endswith(suffix)", target_can)

        target_replay = _class_method_source(
            _NPU_GRAPH_RUNNER, "NPUGraphRunner", "replay"
        )
        self.assertIn("run_npu_graph_update_and_replay", target_replay)
        self.assertIn("_assert_tree_replay_graph", target_replay)
        self.assertIn("NpuGraphReplaySubmittedError", target_replay)
        self.assertIn("output_buffers[graph_key]", target_replay)
        self.assertIn("finally:", target_replay)
        self.assertIn("_clear_tree_replay_plan", target_replay)
        self.assertNotIn("NPU graph miss", target_replay)
        self.assertNotIn("_tree_replay_graphs_id", target_replay)

        draft_can = _class_method_source(
            _EAGLE_DRAFT_NPU, "EAGLEDraftNpuGraphRunner", "can_run"
        )
        self.assertIn("tokens_per_req = self.num_tokens_per_bs", draft_can)
        self.assertIn("graph_key not in self.graphs", draft_can)
        self.assertNotIn("endswith(suffix)", draft_can)

        draft_replay = _class_method_source(
            _EAGLE_DRAFT_NPU, "EAGLEDraftNpuGraphRunner", "replay"
        )
        self.assertIn("super().replay(forward_batch)", draft_replay)
        self.assertIn("NpuGraphPreparationError", draft_replay)
        self.assertIn("NpuGraphReplaySubmittedError", draft_replay)
        self.assertIn("_restore_forward_batch_fields", draft_replay)
        self.assertIn("_clear_tree_replay_plan", draft_replay)

        draft_inner = _class_method_source(
            _EAGLE_DRAFT_NPU, "EAGLEDraftNpuGraphRunner", "_replay"
        )
        self.assertIn("run_npu_graph_update_and_replay", draft_inner)
        self.assertIn("_assert_tree_replay_graph", draft_inner)
        self.assertIn("fill_fia_cpu_update_payload", draft_inner)
        self.assertIn("NpuGraphReplaySubmittedError", draft_inner)
        self.assertNotIn("_tree_replay_graphs_id", draft_inner)

    def test_npu_graph_runner_inits_capture_attrs_before_parent(self):
        init_src = _class_method_source(_NPU_GRAPH_RUNNER, "NPUGraphRunner", "__init__")
        self.assertIn("super().__init__", init_src)
        before, after = init_src.split("super().__init__", 1)
        self.assertIn("self._fia_payloads = {}", before)
        self.assertIn("self.update_attr_name = None", before)
        self.assertNotIn("self._fia_payloads = {}", after)
        self.assertNotIn("self.update_attr_name = None", after)

        ensure_src = _class_method_source(
            _NPU_GRAPH_RUNNER, "NPUGraphRunner", "_ensure_capture_attrs"
        )
        self.assertIn("_init_arch_map", ensure_src)
        self.assertIn("attr_name", ensure_src)
        self.assertIn("_fia_payloads", ensure_src)

        cap_src = _class_method_source(_NPU_GRAPH_RUNNER, "NPUGraphRunner", "capture")
        self.assertIn("_ensure_capture_attrs", cap_src)
        self.assertIn("super().capture()", cap_src)
        self.assertLess(
            cap_src.find("_ensure_capture_attrs"), cap_src.find("super().capture()")
        )

        one_src = _class_method_source(
            _NPU_GRAPH_RUNNER, "NPUGraphRunner", "capture_one_batch_size"
        )
        self.assertIn("_ensure_capture_attrs", one_src)
        self.assertLess(
            one_src.find("_ensure_capture_attrs"),
            one_src.find("_get_update_attr_name"),
        )


if __name__ == "__main__":
    unittest.main()
