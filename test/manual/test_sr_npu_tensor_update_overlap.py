"""Real-NPU regression probe for captured CPU tensor lengths (no model).

SR_TENSOR_ROLE=draft|tail; SR_TENSOR_BACKEND=fia|atb;
SR_TENSOR_DTYPE=float16|bfloat16; SR_TENSOR_MODE=serial|overlap|
delayed_overlap|mid_update_overlap|queued_serial|queued_overlap.
SR_TENSOR_ITERS defaults to 100; SR_TENSOR_DEVICE defaults to 0.
Run configurations in separate processes with an external timeout.
Production attention/fill/helper function bodies are loaded via AST to avoid
initializing the full SGLang dependency stack. The worker module is loaded whole.
"""

import ast
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest

import torch
import torch_npu


ROOT = Path(__file__).resolve().parents[2]
SRT = ROOT / "python/sglang/srt"
SOURCE_HASHES = {}


def load_functions(path, names):
    source = path.read_text()
    SOURCE_HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(source.encode()).hexdigest()
    nodes = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError(f"Missing production definitions: {names}")
    ns = dict(torch=torch, torch_npu=torch_npu, threading=threading)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


HELPER = load_functions(SRT / "speculative/spec_utils.py",
                        {"run_npu_graph_update_and_replay", "NpuGraphReplaySubmittedError"})
ATTENTION = load_functions(SRT / "hardware_backend/npu/attention/ascend_backend.py",
                          {"_run_sr_tree_paged_attention", "_run_sr_tail_paged"})
DRAFT_FILL = load_functions(SRT / "speculative/standalone_remote/drafter/sr_tree_paged_layout.py",
                           {"fill_paged_cpu_update_payload"})["fill_paged_cpu_update_payload"]
TAIL_FILL = load_functions(SRT / "hardware_backend/npu/graph_runner/sr_tail_extend_npu_graph_runner.py",
                          {"fill_tail_graph_cpu_update_payload"})["fill_tail_graph_cpu_update_payload"]
WORKER_PATH = SRT / "speculative/standalone_remote/sr_graph_update.py"
SOURCE_HASHES[str(WORKER_PATH.relative_to(ROOT))] = hashlib.sha256(WORKER_PATH.read_bytes()).hexdigest()
SPEC = importlib.util.spec_from_file_location("sr_tensor_probe_worker", WORKER_PATH)
WORKER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER_MODULE)


class TensorAttentionGraph:
    def __init__(self, role, backend, dtype, device):
        self.role, self.backend = role, backend
        self.device, self.dtype = device, dtype
        self.steps = 3 if role == "draft" else 1
        self.groups = [0, 0, 1, 1] if role == "draft" else [0, 0, 1, 1, 1, 2]
        self.rows = len(self.groups)
        self.step_ids = [step for step in range(self.steps) for _ in range(2)]
        self.attr = "actual_seq_lengths_kv" if backend == "fia" else "context_lens"
        self.stream = torch.npu.current_stream(device)
        self.capture_stream = torch.npu.Stream(device=device)
        self.q = torch.zeros((self.rows, 2, 64), device=device, dtype=dtype)
        self.k = torch.zeros((7, 128, 2, 64), device=device, dtype=dtype)
        cpu_v = torch.zeros((7, 128, 2, 64), dtype=torch.float32)
        # Exactly representable BF16 values; uniform attention has a CPU oracle.
        for req in range(2):
            values = torch.arange(384).float().div(16, rounding_mode="floor") / 16 + req * 2
            cpu_v[1 + req * 3:4 + req * 3] = values.reshape(3, 128, 1, 1)
        self.cpu_v = cpu_v
        self.v = cpu_v.to(device=device, dtype=dtype)
        pages = [[1, 2, 3], [4, 5, 6], [0, 0, 0]]
        self.tables = torch.tensor([pages[g] for g in self.groups], dtype=torch.int32, device=device)
        self.lens = [torch.ones(self.rows, dtype=torch.int32) for _ in range(self.steps)]
        self.identities = [(id(x), x.data_ptr()) for x in self.lens]
        self.metadata = [SimpleNamespace(block_tables=self.tables, context_lens_cpu=x,
                                         context_lens_list=[1] * self.rows) for x in self.lens]
        self.owner = SimpleNamespace(use_fia=backend == "fia", page_size=128,
                                     sr_tail_attention_paths=set())
        self.layer = SimpleNamespace(tp_q_head_num=2, tp_k_head_num=2, tp_v_head_num=2,
                                     qk_head_dim=64, v_head_dim=64, scaling=0.125)
        self.payload = ([{self.attr: self.lens[s]} for s in self.step_ids]
                        if role == "draft" else [{self.attr: self.lens[0]}])
        self.fill([32, 129])
        self.capture_stream.wait_stream(self.stream)
        with torch.npu.stream(self.capture_stream):
            for _ in range(3):
                self.forward()
        self.capture_stream.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=self.capture_stream, auto_dispatch_capture=True):
            self.outputs = self.forward()
        self.records = self.graph.graph_dispatch_mode.graph_dispatch_records
        if len(self.records) != len(self.step_ids):
            raise AssertionError(f"Dynamic records: {len(self.records)} != {len(self.step_ids)}")
        self.update()
        self.graph.replay()
        torch.npu.synchronize(device)

    def lengths(self, base, step):
        a, b = base
        return ([a + step, a + step, b + step, b + step] if self.role == "draft"
                else [a - 1, a, b - 2, b - 1, b, 1])

    def fill(self, base):
        lengths = [self.lengths(base, step) for step in range(self.steps)]
        if self.role == "draft":
            DRAFT_FILL(self.payload, lengths, self.step_ids, self.attr)
        else:
            TAIL_FILL(self.payload, lengths[0], use_fia=self.backend == "fia")
        assert self.identities == [(id(x), x.data_ptr()) for x in self.lens]
        for step, md in enumerate(self.metadata):
            md.context_lens_list[:] = lengths[step]
        assert all(rec[self.attr] is self.lens[step]
                   for rec, step in zip(self.payload, self.step_ids))

    def forward(self):
        outputs = []
        for step in self.step_ids:
            md = self.metadata[step]
            self.owner.forward_metadata = SimpleNamespace(sr_tree_paged=md, sr_tail=md)
            name = "_run_sr_tree_paged_attention" if self.role == "draft" else "_run_sr_tail_paged"
            outputs.append(ATTENTION[name](self.owner, self.q, self.k, self.v, self.layer))
        return outputs

    def update(self):
        self.graph.update(cpu_update_input=self.payload)

    def expected(self, base):
        outputs = []
        for step in self.step_ids:
            rows = []
            for group, length in zip(self.groups, self.lengths(base, step)):
                if group == 2:
                    rows.append(torch.zeros(128))
                else:
                    data = self.cpu_v[1 + group * 3:4 + group * 3].reshape(384, 128)
                    rows.append(data[:length].mean(dim=0))
            outputs.append(torch.stack(rows))
        return outputs


class TestCapturedTensorOverlap(unittest.TestCase):
    def test_tensor_lengths(self):
        role = os.environ.get("SR_TENSOR_ROLE", "draft")
        backend = os.environ.get("SR_TENSOR_BACKEND", "fia")
        dtype_name = os.environ.get("SR_TENSOR_DTYPE", "float16")
        mode = os.environ.get("SR_TENSOR_MODE", "serial")
        iterations = int(os.environ.get("SR_TENSOR_ITERS", "100"))
        device = int(os.environ.get("SR_TENSOR_DEVICE", "0"))
        self.assertIn(role, ("draft", "tail"))
        self.assertIn(backend, ("fia", "atb"))
        self.assertIn(dtype_name, ("float16", "bfloat16"))
        self.assertIn(mode, ("serial", "overlap", "delayed_overlap", "mid_update_overlap",
                             "queued_serial", "queued_overlap"))
        torch.npu.set_device(device)
        g = TensorAttentionGraph(role, backend, getattr(torch, dtype_name), device)
        tolerance = 0.003 if dtype_name == "float16" else 0.02
        bases = ([32, 129], [257, 65], [127, 381], [191, 17])
        eager_refs = {}
        eager_error = 0.0
        for base in bases:
            g.fill(base)
            actuals = [x.cpu().float() for x in g.forward()]
            eager_refs[tuple(base)] = actuals
            for actual, expected in zip(actuals, g.expected(base)):
                eager_error = max(eager_error, (actual - expected).abs().max().item())
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0)
        # Positive detection control: a correct replay for A must fail B's oracle.
        g.fill(bases[0]); g.update(); g.graph.replay()
        torch.npu.synchronize(device)
        negative_gap = max((x.cpu().float() - y).abs().max().item()
                           for x, y in zip(g.outputs, g.expected(bases[1])))
        self.assertGreater(negative_gap, tolerance * 10)
        worker = WORKER_MODULE.SRGraphUpdateWorker(device)
        self.addCleanup(worker.close)
        saved, overlap_count = [], 0
        queued = mode.startswith("queued_")
        overlap = mode not in ("serial", "queued_serial")
        for iteration in range(iterations):
            base = bases[(iteration + 1) % len(bases)]
            g.fill(base)  # Always caller-side; same captured tensors every round.
            entered, release = threading.Event(), threading.Event()
            times = {}
            original = None
            if mode == "mid_update_overlap":
                original = g.records[1].op_cache_entry

                @functools.wraps(original)
                def gated(*args, **kwargs):
                    entered.set()
                    if not release.wait(10):
                        raise TimeoutError("mid-update gate")
                    return original(*args, **kwargs)

                g.records[1].op_cache_entry = gated

            def update():
                times["u0"] = time.perf_counter()
                if mode == "delayed_overlap":
                    entered.set()
                    if not release.wait(10):
                        raise TimeoutError("before-update gate")
                try:
                    g.update()
                finally:
                    times["u1"] = time.perf_counter()

            def replay():
                timer = None
                if mode in ("delayed_overlap", "mid_update_overlap"):
                    if not entered.wait(10):
                        raise TimeoutError("update did not reach gate")
                    timer = threading.Timer(0.02, release.set)
                    timer.start()
                times["r0"] = time.perf_counter()
                try:
                    g.graph.replay()
                finally:
                    times["r1"] = time.perf_counter()
                    release.set()
                    if timer is not None:
                        timer.join()

            try:
                HELPER["run_npu_graph_update_and_replay"](
                    update, replay, overlap=overlap, update_worker=worker if overlap else None)
            finally:
                release.set()
                if original is not None:
                    g.records[1].op_cache_entry = original
            overlap_count += int(max(times["u0"], times["r0"]) < min(times["u1"], times["r1"]))
            snapshots = [output.clone() for output in g.outputs]
            if not queued:
                torch.npu.synchronize(device)
                snapshots = [x.cpu().float() for x in snapshots]
            saved.append((list(base), snapshots))
        torch.npu.synchronize(device)
        mismatches = stale = 0
        max_error = max_eager_difference = 0.0
        previous = bases[0]
        for iteration, (base, outputs) in enumerate(saved):
            bad = False
            for index, (actual, expected, old, eager) in enumerate(zip(
                    outputs, g.expected(base), g.expected(previous), eager_refs[tuple(base)])):
                actual = actual.cpu().float()
                finite = bool(torch.isfinite(actual).all())
                error = (actual - expected).abs().max().item() if finite else float("inf")
                max_error = max(max_error, error)
                max_eager_difference = max(max_eager_difference, (actual - eager).abs().max().item())
                if error > tolerance:
                    bad = True
                    old_match = bool(torch.allclose(actual, old, atol=tolerance, rtol=0))
                    stale += int(old_match)
                    if mismatches < 2:
                        print("MISMATCH " + json.dumps(dict(iteration=iteration, record=index,
                              base=base, actual=actual[:, 0].tolist(), expected=expected[:, 0].tolist(),
                              matches_previous=old_match)), flush=True)
            mismatches += int(bad)
            previous = base
        result = dict(role=role, backend=backend, dtype=dtype_name, mode=mode, iterations=iterations,
                      records=len(g.records), captured_cpu_tensor=True, pointer_identity_preserved=True,
                      mismatched_rounds=mismatches, stale_records=stale, max_abs_error=max_error,
                      max_eager_difference=max_eager_difference, eager_oracle_error=eager_error,
                      negative_control_gap=negative_gap, host_overlap_rounds=overlap_count,
                      inter_round_device_sync=not queued, source_hashes=SOURCE_HASHES)
        print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
        self.assertEqual(mismatches, 0)
        if mode in ("delayed_overlap", "mid_update_overlap"):
            self.assertGreater(overlap_count, 0)


if __name__ == "__main__":
    unittest.main()
