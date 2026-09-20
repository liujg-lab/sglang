"""CPU tests for NPU tree graph admission, exception protocol, and Draft restore.

Does not import torch_npu. Runtime helpers come from ``tree_attn_fallback``;
NPU runner wiring is source-guarded via AST. A small in-process fake covers
plan lifecycle and sequential update-then-replay without launching a graph.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import threading
import unittest
from types import MethodType, SimpleNamespace

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
_SPEC_UTILS = _REPO_ROOT / "python/sglang/srt/speculative/spec_utils.py"


def _class_method_source(path: pathlib.Path, class_name: str, method_name: str) -> str:
    text = path.read_text()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.get_source_segment(text, child) or ast.unparse(child)
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def _extract_class_methods(path: pathlib.Path, class_name: str, names, ns):
    tree = ast.parse(path.read_text())
    klass = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    nodes = [
        n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    if len(nodes) != len(names):
        missing = set(names) - {n.name for n in nodes}
        raise AssertionError(f"missing {missing} in {class_name}")
    ns = dict(ns)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        ns,
    )
    return {name: ns[name] for name in names}


def _producer_fallback(spec_info, seq):
    producer_seq_lens = getattr(spec_info, "seq_lens_cpu", None)
    return seq if producer_seq_lens is None else producer_seq_lens


def _load_spec_utils_nodes(*names):
    tree = ast.parse(_SPEC_UTILS.read_text())
    wanted = set(names)
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted
    ]
    found = {n.name for n in nodes}
    if found != wanted:
        raise AssertionError(f"missing {sorted(wanted - found)} in spec_utils.py")
    ns = {"threading": threading}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_SPEC_UTILS), "exec"), ns)
    return ns


_HELPER_NS = _load_spec_utils_nodes(
    "NpuGraphPreparationError",
    "NpuGraphReplaySubmittedError",
    "run_npu_graph_update_and_replay",
)
_PrepError = _HELPER_NS["NpuGraphPreparationError"]
_SubmittedError = _HELPER_NS["NpuGraphReplaySubmittedError"]
_run_update_replay = _HELPER_NS["run_npu_graph_update_and_replay"]


class _FakeLogits:
    def __init__(self, next_token_logits=None, full_logits=None, hidden_states=None, **kwargs):
        self.next_token_logits = next_token_logits
        self.full_logits = full_logits
        self.hidden_states = hidden_states


class _FakePP:
    def __init__(self, tensors):
        self.tensors = tensors


class _FakeGraph:
    def __init__(self, name):
        self.name = name
        self.updates = []
        self.replay_count = 0

    def update(self, cpu_update_input=None):
        rec = cpu_update_input[0]
        snap = {}
        for key, value in rec.items():
            snap[key] = list(value) if not torch.is_tensor(value) else value.clone()
        self.updates.append(snap)

    def replay(self):
        self.replay_count += 1


class _HelperBox:
    def __init__(self):
        self.calls = []
        self.impl = _run_update_replay

    def __call__(self, update_fn, replay_fn, overlap=False):
        self.calls.append({"overlap": overlap, "replay_fn": replay_fn})
        return self.impl(update_fn, replay_fn, overlap=overlap)

    def reset(self):
        self.calls.clear()


def _fill_fia_stub(payload, step_lens_list, step_ids, attr_name):
    for rec, step in zip(payload, step_ids):
        rec[attr_name][:] = list(step_lens_list[step])


def _install_fake_npu(devices=None):
    devices = [] if devices is None else devices

    def set_device(gpu_id):
        devices.append(gpu_id)

    torch.npu = SimpleNamespace(set_device=set_device)
    return devices


def _thread_idents():
    return {t.ident for t in threading.enumerate() if t.ident is not None}


def _extract_npu_replay(helper_fn):
    ns = {
        "torch": torch,
        "np": SimpleNamespace(array=lambda x: x, int32="int32"),
        "logger": logging.getLogger("test.npu_replay"),
        "TreeReplayPlan": TreeReplayPlan,
        "NpuGraphPreparationError": _PrepError,
        "NpuGraphReplaySubmittedError": _SubmittedError,
        "run_npu_graph_update_and_replay": helper_fn,
        "get_current_stream_idx": lambda: None,
        "is_deepseek_nsa": lambda hf: False,
        "tree_fia_actual_seq_lengths_kv": lambda lens, rows: list(lens)
        + [0] * (int(rows) - len(list(lens))),
        "fill_fia_cpu_update_payload": _fill_fia_stub,
        "LogitsProcessorOutput": _FakeLogits,
        "PPProxyTensors": _FakePP,
        "AttentionArch": SimpleNamespace(MLA="mla"),
    }
    methods = _extract_class_methods(
        _NPU_GRAPH_RUNNER,
        "NPUGraphRunner",
        [
            "_clear_tree_replay_plan",
            "_assert_tree_replay_graph",
            "_is_tree_verify_batch",
            "_current_tree_attention_impl",
            "_get_update_attr_name",
            "_get_update_attr_type",
            "_update_inputs",
            "_update_decode_inputs",
            "_update_target_tree_fia_inputs",
            "replay",
        ],
        ns,
    )
    return methods


def _bind_methods(runner, methods):
    for name, fn in methods.items():
        setattr(runner, name, MethodType(fn, runner))


def _make_replay_runner(
    methods,
    *,
    raw_bs,
    capture_bs,
    seq_lens,
    plain_ar=True,
    decode=True,
    target_verify=False,
    tail=False,
    gpu_id=3,
    target_fia=False,
    compact_fia=False,
    topk=1,
    graph_key=None,
    graphs=None,
    payloads=None,
):
    key = capture_bs if graph_key is None else graph_key
    graphs = {} if graphs is None else graphs
    payloads = {} if payloads is None else payloads
    graph = graphs.get(key) or _FakeGraph(key)
    graphs[key] = graph
    backend = SimpleNamespace(
        _use_target_tree_paged_fia=lambda: target_fia,
        _use_tree_compact_fia=lambda: compact_fia,
        _use_tree_shared_prefix=lambda: False,
        _replay_tree_s_cap=None,
        forward_metadata=SimpleNamespace(
            sr_target_tree_fia=None, tree_verify_kv_lens_t=None
        ),
        tree_fia_kv_lens_cpu=None,
        tree_attention_impl="compact_fia",
    )
    padded = list(seq_lens) + [0] * (int(capture_bs) - int(raw_bs))
    if target_fia:
        backend.forward_metadata.sr_target_tree_fia = SimpleNamespace(
            kv_lens_cpu=list(padded)
        )
    runner = SimpleNamespace(
        _plain_ar_update_overlap=plain_ar,
        model_runner=SimpleNamespace(
            attn_backend=backend,
            server_args=SimpleNamespace(speculative_eagle_topk=topk),
            model_config=SimpleNamespace(hf_config=None),
            gpu_id=gpu_id,
        ),
        graphs=graphs,
        _fia_payloads=payloads,
        _target_fia_maps={},
        _tree_attention_impls={},
        output_buffers={},
        enable_pdmux=False,
        is_dllm=False,
        tree_verify_replay_count=0,
        tree_verify_eager_fallback_count=0,
        bs=capture_bs,
        raw_bs=raw_bs,
        raw_num_token=raw_bs,
        num_tokens_per_bs=1,
        actual_ntpb=1,
        update_attr_name="actual_seq_lengths_kv",
        update_attr_type=[],
        attr_name={"mla": "actual_seq_lengths_kv"},
        attr_type={"mla": []},
    )
    runner.output_buffers[key] = _FakeLogits(
        next_token_logits=torch.arange(capture_bs, dtype=torch.float32) + 100,
        hidden_states=torch.arange(capture_bs, dtype=torch.float32) + 200,
    )
    if target_fia:
        payload = [{"actual_seq_lengths_kv": [1] * int(capture_bs)}]
        payloads[key] = payload
        runner._target_fia_maps[key] = {
            "n_records": 1,
            "num_layers": 1,
            "step_ids": [0],
            "bs": int(capture_bs),
            "payload": payload,
            "attr_name": "actual_seq_lengths_kv",
        }
    plan = TreeReplayPlan(
        graph_key=key,
        raw_bs=int(raw_bs),
        capture_bs=int(capture_bs),
        tokens_per_req=1,
        kv_bucket=None,
    )
    runner._tree_replay_plan = plan
    runner._tree_replay_graph = graph
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: decode,
            is_target_verify=lambda: target_verify,
        ),
        is_sr_tail_extend=tail,
        seq_lens=torch.tensor(list(seq_lens), dtype=torch.int32),
        input_ids=torch.zeros(raw_bs, dtype=torch.int64),
        positions=torch.zeros(raw_bs, dtype=torch.int64),
    )
    runner._tree_replay_batch_id = id(batch)
    runner._tree_replay_stream_idx = None
    runner.replay_prepare = MethodType(
        lambda self, forward_batch, pp_proxy_tensors=None: None, runner
    )
    _bind_methods(runner, methods)
    return runner, batch, graph


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

    def test_update_then_replay_skips_replay_on_update_error(self):
        order = []

        def update():
            order.append("update")
            raise ValueError("update boom")

        def replay():
            order.append("replay")

        with self.assertRaises(_SubmittedError) as ctx:
            _run_update_replay(update, replay)
        self.assertEqual(order, ["update"])
        self.assertIsInstance(ctx.exception.__cause__, ValueError)

    def test_update_then_replay_wraps_replay_error(self):
        order = []

        def update():
            order.append("update")

        def replay():
            order.append("replay")
            raise RuntimeError("replay boom")

        with self.assertRaises(_SubmittedError) as ctx:
            _run_update_replay(update, replay)
        self.assertEqual(order, ["update", "replay"])
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertIn("replay boom", str(ctx.exception.__cause__))

    def test_update_replay_success_is_sequential(self):
        order = []
        self.assertIsNone(
            _run_update_replay(
                lambda: order.append("update"), lambda: order.append("replay")
            )
        )
        self.assertEqual(order, ["update", "replay"])

    def test_spec_utils_update_replay_has_no_side_thread(self):
        src = _SPEC_UTILS.read_text()
        start = src.index("def run_npu_graph_update_and_replay")
        end = src.index("\ndef normalize_fia_op_name")
        helper = src[start:end]
        self.assertIn("overlap=False", helper)
        serial = helper.split("if not overlap:", 1)[1].split("return", 1)[0]
        self.assertNotIn("threading.Thread", serial)
        self.assertIn("update_fn()", serial)
        self.assertIn("replay_fn()", serial)
        self.assertLess(serial.index("update_fn()"), serial.index("replay_fn()"))
        overlap = helper.split("if not overlap:", 1)[1].split("return", 1)[1]
        self.assertIn("threading.Thread", overlap)
        self.assertIn("finally:", overlap)
        self.assertIn("thread.join()", overlap)
        self.assertLess(overlap.index("finally:"), overlap.index("thread.join()"))

    def test_overlap_replay_starts_before_update_finishes(self):
        started = threading.Event()
        release = threading.Event()
        order = []
        before = _thread_idents()

        def update():
            order.append("update_start")
            started.set()
            self.assertTrue(release.wait(timeout=2))
            order.append("update_end")

        def replay():
            self.assertTrue(started.wait(timeout=2))
            order.append("replay")
            release.set()

        self.assertIsNone(_run_update_replay(update, replay, overlap=True))
        self.assertEqual(order[0], "update_start")
        self.assertLess(order.index("replay"), order.index("update_end"))
        self.assertIn("update_end", order)
        self.assertTrue(_thread_idents() <= before)

    def test_overlap_update_error_after_replay_is_submitted(self):
        started = threading.Event()
        release = threading.Event()
        order = []
        before = _thread_idents()

        def update():
            order.append("update_start")
            started.set()
            self.assertTrue(release.wait(timeout=2))
            raise ValueError("update boom")

        def replay():
            self.assertTrue(started.wait(timeout=2))
            order.append("replay")
            release.set()

        with self.assertRaises(_SubmittedError) as ctx:
            _run_update_replay(update, replay, overlap=True)
        self.assertIn("replay", order)
        self.assertIsInstance(ctx.exception.__cause__, ValueError)
        self.assertTrue(_thread_idents() <= before)

    def test_overlap_replay_error_has_priority_and_joins(self):
        counts = {"update": 0, "replay": 0}
        before = _thread_idents()

        def update():
            counts["update"] += 1
            raise ValueError("update boom")

        def replay():
            counts["replay"] += 1
            raise RuntimeError("replay boom")

        with self.assertRaises(_SubmittedError) as ctx:
            _run_update_replay(update, replay, overlap=True)
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertEqual(counts, {"update": 1, "replay": 1})
        self.assertTrue(_thread_idents() <= before)

    def test_overlap_replay_baseexception_joins_then_reraises(self):
        started = threading.Event()
        release = threading.Event()
        finished = []
        before = _thread_idents()

        def update():
            started.set()
            self.assertTrue(release.wait(timeout=2))
            finished.append("update")

        def replay():
            self.assertTrue(started.wait(timeout=2))
            release.set()
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            _run_update_replay(update, replay, overlap=True)
        self.assertEqual(finished, ["update"])
        self.assertTrue(_thread_idents() <= before)

        class Boom(BaseException):
            pass

        started.clear()
        release.clear()
        finished.clear()

        def replay_base():
            self.assertTrue(started.wait(timeout=2))
            release.set()
            raise Boom()

        with self.assertRaises(Boom) as ctx:
            _run_update_replay(update, replay_base, overlap=True)
        self.assertNotIsInstance(ctx.exception, _SubmittedError)
        self.assertEqual(finished, ["update"])
        self.assertTrue(_thread_idents() <= before)

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
        self.assertIn("capture_bs is None", target_can)
        self.assertLess(
            target_can.find("capture_bs is None"),
            target_can.find("_save_tree_replay_plan"),
        )
        self.assertLess(
            target_can.find("_clear_tree_replay_plan"),
            target_can.find("capture_bs is None"),
        )
        self.assertIn("tree_verify_eager_fallback_count", target_can)
        self.assertIn("_last_can_run_reject", target_can)
        self.assertIn("bs_over_max_capture_bs", target_can)
        self.assertLess(
            target_can.find("if not super().can_run"),
            target_can.find("tree_verify_eager_fallback_count += 1"),
        )

        target_replay = _class_method_source(
            _NPU_GRAPH_RUNNER, "NPUGraphRunner", "replay"
        )
        self.assertIn("run_npu_graph_update_and_replay", target_replay)
        self.assertIn("overlap=False", target_replay)
        self.assertIn("overlap_decode", target_replay)
        self.assertIn("overlap=overlap_decode", target_replay)
        self.assertIn("_plain_ar_update_overlap", target_replay)
        self.assertIn("_update_decode_inputs", target_replay)
        self.assertIn("is_sr_tail_extend", target_replay)
        self.assertNotIn("overlap=True", target_replay)
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
        self.assertIn("capture_bs is None", draft_can)
        self.assertLess(
            draft_can.find("capture_bs is None"),
            draft_can.find("_save_tree_replay_plan"),
        )
        self.assertLess(
            draft_can.find("_clear_tree_replay_plan"),
            draft_can.find("capture_bs is None"),
        )
        self.assertIn("tree_eager_fallback_count", draft_can)
        self.assertIn("_last_can_run_reject", draft_can)
        self.assertIn("bs_over_max_capture_bs", draft_can)
        self.assertIn("empty_capture_bs", draft_can)
        self.assertLess(
            draft_can.find("not is_bs_supported or not self.capture_bs"),
            draft_can.find("tree_eager_fallback_count += 1"),
        )

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
        self.assertIn("overlap=", draft_inner)
        self.assertIn("SGLANG_NPU_TREE_FIA_SERIAL_UPDATE", draft_inner)
        self.assertIn("_assert_tree_replay_graph", draft_inner)
        self.assertIn("fill_fia_cpu_update_payload", draft_inner)
        self.assertIn("NpuGraphReplaySubmittedError", draft_inner)
        self.assertNotIn("_tree_replay_graphs_id", draft_inner)

        eager_src = _class_method_source(
            _ASCEND_BACKEND, "AscendAttnMultiStepDraftBackend", "prepare_sr_tree_paged_eager"
        )
        can_src = _class_method_source(
            _ASCEND_BACKEND, "AscendAttnBackend", "tree_slot_graph_can_run"
        )
        self.assertIn("quantize_page_width", eager_src)
        self.assertIn("max_pages=max_pages", eager_src)
        self.assertIn("metrics=None", eager_src)
        self.assertIn("tree_paged_view", eager_src)
        self.assertIn("tree_paged_copy", eager_src)
        self.assertIn("tree_paged_bind", eager_src)
        self.assertIn("select_page_bucket", can_src)
        self.assertNotIn("quantize_page_width", can_src)
        self.assertIn("tree_slot_graph_can_run", draft_can)

    def test_npu_graph_runner_inits_capture_attrs_before_parent(self):
        init_src = _class_method_source(_NPU_GRAPH_RUNNER, "NPUGraphRunner", "__init__")
        self.assertIn("super().__init__", init_src)
        before, after = init_src.split("super().__init__", 1)
        self.assertIn("self._fia_payloads = {}", before)
        self.assertIn("self.update_attr_name = None", before)
        self.assertIn("self._plain_ar_update_overlap", before)
        self.assertIn("standalone_remote_role", before)
        self.assertIn("spectre_role", before)
        self.assertNotIn("self._fia_payloads = {}", after)
        self.assertNotIn("self.update_attr_name = None", after)
        self.assertNotIn("self._plain_ar_update_overlap", after)

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

    def test_padded_capture_bs_over_max_returns_none(self):
        import bisect

        draft_fn = _extract_class_methods(
            _EAGLE_DRAFT_NPU,
            "EAGLEDraftNpuGraphRunner",
            ["_padded_capture_bs"],
            {"bisect": bisect},
        )["_padded_capture_bs"]
        target_fn = _extract_class_methods(
            _NPU_GRAPH_RUNNER,
            "NPUGraphRunner",
            ["_padded_capture_bs"],
            {"bisect": bisect},
        )["_padded_capture_bs"]
        draft = SimpleNamespace(capture_bs=[1, 2], require_mlp_tp_gather=False)
        draft._padded_capture_bs = MethodType(draft_fn, draft)
        target = SimpleNamespace(capture_bs=[1, 2, 4], require_mlp_tp_gather=False)
        target._padded_capture_bs = MethodType(target_fn, target)

        raw, cap = draft._padded_capture_bs(SimpleNamespace(batch_size=4))
        self.assertEqual(raw, 4)
        self.assertIsNone(cap)
        raw, cap = draft._padded_capture_bs(SimpleNamespace(batch_size=1))
        self.assertEqual((raw, cap), (1, 1))
        raw, cap = draft._padded_capture_bs(SimpleNamespace(batch_size=2))
        self.assertEqual((raw, cap), (2, 2))

        raw, cap = target._padded_capture_bs(SimpleNamespace(batch_size=8), actual_ntpb=1)
        self.assertEqual(raw, 8)
        self.assertIsNone(cap)
        raw, cap = target._padded_capture_bs(SimpleNamespace(batch_size=3), actual_ntpb=1)
        self.assertEqual((raw, cap), (3, 4))
        empty = SimpleNamespace(capture_bs=[], require_mlp_tp_gather=False)
        empty._padded_capture_bs = MethodType(draft_fn, empty)
        raw, cap = empty._padded_capture_bs(SimpleNamespace(batch_size=1))
        self.assertEqual(raw, 1)
        self.assertIsNone(cap)

    def test_can_run_over_max_bs_counts_and_records_reason(self):
        import bisect

        fns = _extract_class_methods(
            _EAGLE_DRAFT_NPU,
            "EAGLEDraftNpuGraphRunner",
            [
                "can_run",
                "_padded_capture_bs",
                "_clear_tree_replay_plan",
                "_make_graph_key",
                "_save_tree_replay_plan",
            ],
            {"bisect": bisect, "TreeReplayPlan": TreeReplayPlan},
        )
        runner = SimpleNamespace(
            require_mlp_tp_gather=False,
            require_mlp_sync=False,
            disable_padding=False,
            capture_bs=[1, 2],
            max_bs=2,
            _slot_gather_graph=True,
            _tree_paged=True,
            tree_eager_fallback_count=0,
            _last_can_run_reject=None,
            graphs={},
            _tree_fia_maps={},
            _tree_attention_impls={},
            num_tokens_per_bs=3,
        )
        for name, fn in fns.items():
            setattr(runner, name, MethodType(fn, runner))
        self.assertFalse(runner.can_run(SimpleNamespace(batch_size=4)))
        self.assertEqual(runner.tree_eager_fallback_count, 1)
        self.assertIn("bs_over_max_capture_bs", runner._last_can_run_reject)
        self.assertIn("bs=4", runner._last_can_run_reject)
        self.assertIn("max_bs=2", runner._last_can_run_reject)

    def test_record_tree_expand_admission_reasons(self):
        path = (
            _REPO_ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
        )
        tree = ast.parse(path.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "record_tree_expand_admission"
        )
        ns = {}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
            ns,
        )
        record = ns["record_tree_expand_admission"]
        from collections import Counter

        graph_metrics = SimpleNamespace(counts=Counter())
        self.assertFalse(record(graph_metrics, True, None))
        self.assertEqual(graph_metrics.counts["tree_graph_batches"], 1)
        self.assertEqual(graph_metrics.counts["tree_eager_batches"], 0)

        eager_metrics = SimpleNamespace(counts=Counter())
        runner = SimpleNamespace(
            _last_can_run_reject="bs_over_max_capture_bs bs=4 max_bs=2"
        )
        self.assertTrue(record(eager_metrics, False, runner))
        self.assertEqual(eager_metrics.counts["tree_eager_batches"], 1)
        self.assertEqual(eager_metrics.counts["tree_eager_bs_over_max_capture_bs"], 1)

        missing = SimpleNamespace(counts=Counter())
        self.assertTrue(record(missing, False, SimpleNamespace()))
        self.assertEqual(missing.counts["tree_eager_graph_unavailable"], 1)
        self.assertFalse(record(None, False, None))

    def test_filter_capture_bs_follows_cuda_graph_unless_env(self):
        import logging
        import os

        from sglang.srt.speculative.tree_attn_fallback import (
            TREE_DRAFT_CAPTURE_BS_ENV,
            parse_tree_draft_capture_bs,
        )

        fn = _extract_class_methods(
            _EAGLE_DRAFT_NPU,
            "EAGLEDraftNpuGraphRunner",
            ["filter_capture_batch_sizes"],
            {
                "os": os,
                "parse_tree_draft_capture_bs": parse_tree_draft_capture_bs,
                "TREE_DRAFT_CAPTURE_BS_ENV": TREE_DRAFT_CAPTURE_BS_ENV,
                "logger": logging.getLogger("test.filter_capture_bs"),
            },
        )["filter_capture_batch_sizes"]
        runner = SimpleNamespace(
            _slot_gather_graph=True, tree_graph_disabled_reason=None
        )
        runner.filter_capture_batch_sizes = MethodType(fn, runner)
        old = os.environ.pop(TREE_DRAFT_CAPTURE_BS_ENV, None)
        try:
            got, compile_bs = runner.filter_capture_batch_sizes([1, 2, 4], [1, 2, 4])
            self.assertEqual(got, [1, 2, 4])
            self.assertEqual(compile_bs, [1, 2, 4])
            self.assertIsNone(runner.tree_graph_disabled_reason)

            os.environ[TREE_DRAFT_CAPTURE_BS_ENV] = "1,2"
            got, compile_bs = runner.filter_capture_batch_sizes([1, 2, 4], [1, 2, 4])
            self.assertEqual(got, [1, 2])
            self.assertEqual(compile_bs, [1, 2])

            os.environ[TREE_DRAFT_CAPTURE_BS_ENV] = "8"
            got, compile_bs = runner.filter_capture_batch_sizes([1, 2, 4], [1, 2, 4])
            self.assertEqual(got, [])
            self.assertEqual(compile_bs, [])
            self.assertEqual(
                runner.tree_graph_disabled_reason,
                "tree draft capture_bs filter is empty",
            )
        finally:
            if old is None:
                os.environ.pop(TREE_DRAFT_CAPTURE_BS_ENV, None)
            else:
                os.environ[TREE_DRAFT_CAPTURE_BS_ENV] = old

        filter_src = _class_method_source(
            _EAGLE_DRAFT_NPU, "EAGLEDraftNpuGraphRunner", "filter_capture_batch_sizes"
        )
        self.assertIn("follow --cuda-graph-bs", filter_src)
        self.assertIn("os.environ.get(TREE_DRAFT_CAPTURE_BS_ENV)", filter_src)

    def test_update_decode_inputs_binds_worker_gpu_id(self):
        decode_src = _class_method_source(
            _NPU_GRAPH_RUNNER, "NPUGraphRunner", "_update_decode_inputs"
        )
        self.assertIn("torch.npu.set_device(self.model_runner.gpu_id)", decode_src)
        self.assertIn("self._update_inputs(seq_lens, graph_key)", decode_src)
        devices = _install_fake_npu()
        helper = _HelperBox()
        methods = _extract_npu_replay(helper)
        graph = _FakeGraph(1)
        runner = SimpleNamespace(
            model_runner=SimpleNamespace(gpu_id=2),
            update_attr_type=[],
            update_attr_name="actual_seq_lengths_kv",
            _fia_payloads={},
            bs=1,
            graphs={1: graph},
            _tree_replay_graph=graph,
        )
        runner._update_inputs = MethodType(methods["_update_inputs"], runner)
        runner._update_decode_inputs = MethodType(
            methods["_update_decode_inputs"], runner
        )
        runner._update_decode_inputs([10], 1)
        self.assertEqual(devices, [2])
        self.assertEqual(graph.updates[-1]["actual_seq_lengths_kv"], [10])

    def test_overlap_set_device_error_wraps_and_joins(self):
        devices = []

        def set_device(gpu_id):
            devices.append(gpu_id)
            raise RuntimeError("set_device boom")

        torch.npu = SimpleNamespace(set_device=set_device)
        helper = _HelperBox()
        methods = _extract_npu_replay(helper)
        graph = _FakeGraph(1)
        runner = SimpleNamespace(
            model_runner=SimpleNamespace(gpu_id=2),
            update_attr_type=[],
            update_attr_name="actual_seq_lengths_kv",
            _fia_payloads={},
            bs=1,
            graphs={1: graph},
            _tree_replay_graph=graph,
        )
        runner._update_inputs = MethodType(methods["_update_inputs"], runner)
        runner._update_decode_inputs = MethodType(
            methods["_update_decode_inputs"], runner
        )
        counts = {"replay": 0}
        before = _thread_idents()

        def replay():
            counts["replay"] += 1

        with self.assertRaises(_SubmittedError) as ctx:
            _run_update_replay(
                lambda: runner._update_decode_inputs([10], 1),
                replay,
                overlap=True,
            )
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertEqual(devices, [2])
        self.assertEqual(counts["replay"], 1)
        self.assertEqual(graph.updates, [])
        self.assertTrue(_thread_idents() <= before)

    def _run_replay_path(self, helper, **kwargs):
        _install_fake_npu()
        methods = _extract_npu_replay(helper)
        runner, batch, graph = _make_replay_runner(methods, **kwargs)
        out = runner.replay(batch)
        return helper, runner, batch, graph, out

    def test_plain_ar_decode_overlap_uses_this_round_graph_and_payload(self):
        helper = _HelperBox()
        _, runner, _, graph, out = self._run_replay_path(
            helper, raw_bs=1, capture_bs=1, seq_lens=[17]
        )
        self.assertEqual(len(helper.calls), 1)
        self.assertIs(helper.calls[0]["overlap"], True)
        self.assertIs(helper.calls[0]["replay_fn"].__self__, graph)
        self.assertEqual(graph.replay_count, 1)
        self.assertEqual(graph.updates[-1]["actual_seq_lengths_kv"], [17])
        self.assertEqual(runner._fia_payloads[1][0]["actual_seq_lengths_kv"], [17])
        self.assertEqual(out.next_token_logits.tolist(), [100.0])

    def test_replay_paths_keep_existing_serial_overlap(self):
        cases = [
            dict(plain_ar=False, raw_bs=1, capture_bs=1, seq_lens=[8]),
            dict(decode=False, target_verify=True, raw_bs=1, capture_bs=1, seq_lens=[8]),
            dict(tail=True, raw_bs=1, capture_bs=1, seq_lens=[8]),
            dict(
                target_fia=True,
                target_verify=True,
                topk=2,
                raw_bs=1,
                capture_bs=1,
                seq_lens=[8],
            ),
            dict(
                compact_fia=True,
                target_verify=True,
                topk=2,
                raw_bs=1,
                capture_bs=1,
                seq_lens=[8],
            ),
        ]
        for kwargs in cases:
            helper = _HelperBox()
            self._run_replay_path(helper, **kwargs)
            self.assertTrue(helper.calls, kwargs)
            self.assertIs(helper.calls[0]["overlap"], False, kwargs)

    def test_decode_batch_sizes_select_key_pad_and_crop(self):
        helper = _HelperBox()
        graphs = {}
        payloads = {}
        shapes = ((1, 1, [11]), (2, 2, [21, 22]), (4, 4, [41, 42, 43, 44]), (3, 4, [31, 32, 33]))
        for raw_bs, capture_bs, seq_lens in shapes:
            helper.reset()
            _, runner, _, graph, out = self._run_replay_path(
                helper,
                raw_bs=raw_bs,
                capture_bs=capture_bs,
                seq_lens=seq_lens,
                graphs=graphs,
                payloads=payloads,
            )
            padded = list(seq_lens) + [0] * (capture_bs - raw_bs)
            self.assertIs(helper.calls[0]["overlap"], True)
            self.assertIs(helper.calls[0]["replay_fn"].__self__, graph)
            self.assertIs(graph, graphs[capture_bs])
            self.assertEqual(
                graph.updates[-1]["actual_seq_lengths_kv"], padded
            )
            self.assertEqual(
                payloads[capture_bs][0]["actual_seq_lengths_kv"], padded
            )
            self.assertEqual(
                out.next_token_logits.tolist(),
                [100.0 + i for i in range(raw_bs)],
            )
            self.assertEqual(
                out.hidden_states.tolist(),
                [200.0 + i for i in range(raw_bs)],
            )

    def test_dynamic_batch_reuses_payload_after_prior_thread_exits(self):
        helper = _HelperBox()
        graphs = {}
        payloads = {}
        payload_ids = {}
        before = _thread_idents()
        sequence = (
            (1, 1, [10]),
            (2, 2, [20, 21]),
            (4, 4, [40, 41, 42, 43]),
            (3, 4, [30, 31, 32]),
            (1, 1, [11]),
        )
        for raw_bs, capture_bs, seq_lens in sequence:
            helper.reset()
            _, runner, _, graph, out = self._run_replay_path(
                helper,
                raw_bs=raw_bs,
                capture_bs=capture_bs,
                seq_lens=seq_lens,
                graphs=graphs,
                payloads=payloads,
            )
            padded = list(seq_lens) + [0] * (capture_bs - raw_bs)
            self.assertIs(helper.calls[0]["overlap"], True)
            self.assertTrue(_thread_idents() <= before)
            current = payloads[capture_bs][0]["actual_seq_lengths_kv"]
            self.assertEqual(current, padded)
            self.assertEqual(graph.updates[-1]["actual_seq_lengths_kv"], padded)
            if capture_bs in payload_ids:
                self.assertIs(payloads[capture_bs], payload_ids[capture_bs])
            else:
                payload_ids[capture_bs] = payloads[capture_bs]
            self.assertEqual(
                out.next_token_logits.tolist(),
                [100.0 + i for i in range(raw_bs)],
            )
        self.assertEqual(payloads[4][0]["actual_seq_lengths_kv"], [30, 31, 32, 0])
        self.assertEqual(payloads[1][0]["actual_seq_lengths_kv"], [11])
        self.assertNotEqual(payloads[1][0]["actual_seq_lengths_kv"], [10])


if __name__ == "__main__":
    unittest.main()
