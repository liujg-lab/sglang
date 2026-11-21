"""CPU checks for ClearTree H + Q prefit.

These tests stay on the protocol, the ledger split, and the fit file.
They do not start a server.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import zmq

from sglang.srt.speculative.clear_tree_remote import clear_tree_fit as fit
from sglang.srt.speculative.clear_tree_remote import clear_tree_policy as policy
from sglang.srt.speculative.clear_tree_remote import clear_tree_prefit as prefit
from sglang.srt.speculative.clear_tree_remote import clear_tree_transport as transport
from sglang.srt.speculative.clear_tree_remote.clear_tree_config import ClearTreeConfig
from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
    ClearLedger,
    _new_layer_model,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_ops import (
    ClearTreeState,
    finish,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
    keep_by_net_benefit,
    make_round,
    remember_choice,
    reset_policy,
    select_k,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_protocol import (
    ClearAction,
    ClearBatchReply,
    ClearBatchRequest,
    ClearDraftRequest,
    clear_batch_reply,
    parse_target_info,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")


class _Sock:
    def __init__(self, *responses):
        self.recv_multipart = Mock(side_effect=list(responses))
        self.setsockopt = Mock()
        self.send_multipart = Mock()


def _reply(session, rpc):
    batch = ClearBatchReply(session_id=session, rpc_seq=rpc)
    return transport._pack_reply(batch)


def _args():
    return SimpleNamespace(
        host="0.0.0.0",
        port=30000,
        clear_tree_max_depth=8,
        clear_tree_ks=[2, 4, 8, 16, 32, 64],
        clear_tree_widths=[2, 4, 8, 16, 32, 64],
        clear_tree_width=8,
        max_running_requests=4,
        device="cpu",
        model_path="draft-model",
        served_model_name="draft-model",
        dtype="auto",
        quantization=None,
        page_size=1,
    )


class _Thread:
    def __init__(self, target, name, daemon):
        self.target = target
        self.name = name
        self.daemon = daemon
        self._alive = False

    def start(self):
        self._alive = True

    def is_alive(self):
        return self._alive


class ClearTreeFitProtocolTest(CustomTestCase):
    def test_residence_and_fit_ready_round_trip(self):
        request = ClearBatchRequest(
            session_id="s",
            rpc_seq=2,
            action=ClearAction.STEP,
            target_residence_ns=20,
            target_residence_rpc_seq=1,
            target_queued=False,
            target_info={"backend": "cpu", "tp": 1, "extra": "drop", "device": 1.5},
        )
        decoded = ClearBatchRequest.from_dict(request.to_dict())
        self.assertEqual(decoded.target_residence_ns, 20)
        self.assertEqual(decoded.target_residence_rpc_seq, 1)
        self.assertFalse(decoded.target_queued)
        self.assertEqual(decoded.target_info["backend"], "cpu")
        self.assertEqual(decoded.target_info["tp"], 1)
        self.assertNotIn("extra", decoded.target_info)
        self.assertNotIn("device", decoded.target_info)

        illegal = ClearBatchRequest(
            session_id="s",
            rpc_seq=1,
            action=ClearAction.STEP,
            target_residence_ns=-1,
            target_queued=True,
        )
        self.assertNotIn("target_residence_ns", illegal.to_dict())
        dropped = ClearBatchRequest.from_dict(
            {
                "session_id": "s",
                "rpc_seq": 1,
                "action": "step",
                "target_residence_ns": True,
            }
        )
        self.assertIsNone(dropped.target_residence_ns)
        old = ClearBatchRequest.from_dict(
            {"session_id": "s", "rpc_seq": 1, "action": "step", "reqs": []}
        )
        self.assertIsNone(old.target_residence_ns)
        self.assertIsNone(old.target_info)

        reply = clear_batch_reply(
            ClearBatchReply(session_id="s", rpc_seq=1, fit_ready=None)
        )
        reply.fit_ready = False
        again = ClearBatchReply.from_dict(reply.to_dict())
        self.assertIs(reply, clear_batch_reply(reply))
        self.assertFalse(again.fit_ready)
        self.assertIsNone(
            ClearBatchReply.from_dict(
                {"session_id": "s", "rpc_seq": 1, "reqs": [], "fit_ready": "yes"}
            ).fit_ready
        )
        self.assertEqual(
            parse_target_info({"kv_buckets": [256, True], "model": None}),
            {"model": None},
        )
        header = parse_target_info(
            {
                "disable_cuda_graph": True,
                "attention_backend": "ascend",
                "page_size": 128,
                "cuda_graph_bs": [1, 2, 4],
                "tp": True,
            }
        )
        self.assertIs(header["disable_cuda_graph"], True)
        self.assertEqual(header["attention_backend"], "ascend")
        self.assertEqual(header["page_size"], 128)
        self.assertEqual(header["cuda_graph_bs"], [1, 2, 4])
        self.assertIs(header["tp"], True)
        request = ClearBatchRequest(
            session_id="s",
            rpc_seq=3,
            action=ClearAction.PREFILL,
            target_info={
                "disable_cuda_graph": False,
                "attention_backend": "triton",
                "page_size": 128,
                "cuda_graph_bs": [1, 4],
            },
        )
        decoded = ClearBatchRequest.from_dict(request.to_dict())
        self.assertIs(decoded.target_info["disable_cuda_graph"], False)
        self.assertEqual(decoded.target_info["attention_backend"], "triton")
        self.assertEqual(decoded.target_info["page_size"], 128)
        self.assertEqual(decoded.target_info["cuda_graph_bs"], [1, 4])


class ClearTreeTargetResidenceTest(CustomTestCase):
    def _client(self, *responses):
        client = transport.ClearTargetClient.__new__(transport.ClearTargetClient)
        client._socket = _Sock(*responses)
        client._comm_pending = None
        client.comm_metrics = Mock()
        client.timeout_ms = 1000
        client.drop_callback = None
        client.rebuild_on_timeout = False
        client._last_reply_ns = None
        client._last_reply_key = None
        client._last_reply_queued = False
        return client

    def _step(self, session="sess", rpc=2):
        return ClearBatchRequest(session_id=session, rpc_seq=rpc, action=ClearAction.STEP)

    def test_recv_then_send_uses_one_reply(self):
        client = self._client(_reply("sess", 1), zmq.Again())
        reply = client.recv_batch("sess", 1)
        self.assertEqual(reply.rpc_seq, 1)
        self.assertTrue(client._last_reply_queued)
        first = self._step()
        client.send_batch(first)
        self.assertEqual(first.target_residence_rpc_seq, 1)
        self.assertIsInstance(first.target_residence_ns, int)
        self.assertGreaterEqual(first.target_residence_ns, 0)
        self.assertTrue(first.target_queued)
        self.assertIsNone(client._last_reply_ns)
        second = self._step(rpc=3)
        client.send_batch(second)
        self.assertIsNone(second.target_residence_ns)
        self.assertIsNone(second.target_residence_rpc_seq)

    def test_other_session_does_not_fill_residence(self):
        client = self._client(zmq.Again())
        client._last_reply_ns = 10
        client._last_reply_key = ("other", 1)
        client._last_reply_queued = True
        batch = self._step()
        client.send_batch(batch)
        self.assertIsNone(batch.target_residence_ns)
        self.assertIsNone(client._last_reply_key)

    def test_prefill_consumes_the_reply(self):
        client = self._client(zmq.Again(), zmq.Again())
        client._last_reply_ns = 10
        client._last_reply_key = ("sess", 1)
        client._last_reply_queued = False
        prefill = ClearBatchRequest(
            session_id="sess", rpc_seq=2, action=ClearAction.PREFILL
        )
        client.send_batch(prefill)
        self.assertIsNone(prefill.target_residence_ns)
        step = self._step(rpc=3)
        client.send_batch(step)
        self.assertIsNone(step.target_residence_ns)

    def test_stale_probe_does_not_mark_the_match_queued(self):
        client = self._client(_reply("sess", 1), _reply("sess", 2))
        reply = client.recv_batch("sess", 2)
        self.assertEqual(reply.rpc_seq, 2)
        self.assertFalse(client._last_reply_queued)

    def test_draft_records_packet_bytes(self):
        batch = ClearBatchRequest(
            session_id="session",
            rpc_seq=3,
            action=ClearAction.STEP,
            reqs=[ClearDraftRequest(rid="r", committed_ids=[1, 2])],
        )
        frames = [b"target", *transport._pack_request(batch)]
        server = transport.ClearDraftServer.__new__(transport.ClearDraftServer)
        server._socket = _Sock(frames)
        server._comm_pending = None
        server.comm_metrics = Mock()
        server._last_identity = None
        server.last_recv_queued = False
        server.recv_batch(timeout_ms=17)
        self.assertGreater(server.last_request_bytes, 0)


class ClearTreeHoldCurveTest(CustomTestCase):
    def setUp(self):
        reset_policy()

    def test_spikes_do_not_move_a_full_window(self):
        for _ in range(14):
            policy.observe_verify(1, 8, 256, 50.0)
        policy.observe_verify(1, 8, 256, 230.0)
        policy.observe_verify(1, 8, 256, 230.0)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 8, 256), 50.0)

    def test_pava_pools_by_sample_weight(self):
        policy.observe_verify(1, 8, 256, 49.8)
        policy.observe_verify(1, 16, 256, 238.0)
        policy.observe_verify(1, 32, 256, 81.4)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 8, 256), 49.8)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 16, 256), 159.7)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 32, 256), 159.7)

        reset_policy()
        for _ in range(3):
            policy.observe_verify(1, 8, 256, 30.0)
        policy.observe_verify(1, 4, 256, 20.0)
        policy.observe_verify(1, 32, 256, 10.0)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 4, 256), 20.0)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 8, 256), 25.0)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 32, 256), 25.0)

    def test_q_shift_does_not_change_the_choice(self):
        def fill(shift):
            reset_policy()
            policy.observe_verify(1, 8, 256, 20.0)
            policy.observe_verify(1, 16, 256, 40.0)
            for k in (8, 16, 8, 16, 8, 16, 8, 16):
                policy.observe_q(1, k, 5.0 + 0.1 * k + shift)
            return (
                policy.predict_verify_ms(1, 8, 256),
                policy.predict_verify_ms(1, 16, 256),
            )

        base = fill(0.0)
        shifted = fill(10.0)
        self.assertAlmostEqual(shifted[0] - base[0], 10.0, places=4)
        self.assertAlmostEqual(shifted[1] - base[1], 10.0, places=4)
        request = {
            "e": {8: 2.0, 16: 2.2},
            "g": {8: 0.4, 16: 0.5},
            "ell": 2,
            "c": 1.0,
            "item": "a",
            "growing": True,
        }

        def choice(times):
            picked = select_k({k: request["e"][k] - 0.05 * times[k] for k in times})
            kept = keep_by_net_benefit(
                [request],
                theta=0.05,
                verify_times=times,
                layer_cost=lambda _m, _ell: 1.0,
                kappa=0.0,
            )
            return picked, kept

        self.assertEqual(choice({8: base[0], 16: base[1]}), choice({8: shifted[0], 16: shifted[1]}))

    def test_q_outlier_stays_out_of_the_regression(self):
        policy.observe_q(1, 8, 20.0)
        policy.observe_q(1, 8, 20.0)
        fitted = policy._q_fit[1].n
        policy.observe_q(1, 8, 200.0)
        self.assertEqual(policy._q_fit[1].n, fitted)
        self.assertLess(policy.predict_q_ms(1, 8), 100.0)

    def test_regressor_snapshot_continues(self):
        model = _new_layer_model()
        for length, target in ((4.0, 1.0), (8.0, 1.5), (12.0, 2.0), (16.0, 2.4)):
            model.update([1.0, length], target)
        other = _new_layer_model()
        other.from_dict(model.to_dict())
        self.assertIsNone(other._coef)
        model.update([1.0, 20.0], 3.0)
        other.update([1.0, 20.0], 3.0)
        self.assertAlmostEqual(
            model.predict([1.0, 20.0]), other.predict([1.0, 20.0])
        )


class ClearTreePrefitPlanTest(CustomTestCase):
    def _scheduler(self):
        return SimpleNamespace(
            server_args=_args(),
            _clear_fit_ready=False,
            _clear_prefit=fit.PrefitState(),
            _clear_target_prefix={},
        )

    def test_verify_group_locks_depth_and_pads_to_k(self):
        scheduler = self._scheduler()
        reqs = [
            SimpleNamespace(
                rid="clearfit-v-1-128-64-0",
                origin_input_ids=[1] * 128,
                output_ids=[],
            )
        ]
        phases = []
        plan = None
        for _ in range(3):
            plan = make_round(scheduler, reqs)
            phases.append(plan.prefit["phase"])
        self.assertEqual(phases, ["warmup", "warmup", "keep"])
        self.assertEqual(plan.fixed_depth, 7)
        self.assertEqual(plan.locked_k, 64)
        self.assertEqual(plan.replay_kind, "prefit")
        self.assertFalse(plan.replay_eligible)
        self.assertFalse(plan.fallback)
        state = SimpleNamespace(depth=6, config=SimpleNamespace(max_depth=8))
        self.assertTrue(plan.should_grow(state))
        state.depth = 7
        self.assertFalse(plan.should_grow(state))

        config = ClearTreeConfig(width=4, tau=0.1, max_depth=8)
        tree = ClearTreeState.create(config)
        tree.depth = 1
        tree.weight[0] = torch.tensor([0.4, 0.3, 0.2, 0.1])
        tree.tokens[0] = torch.tensor([11, 12, 13, 14])
        verified = finish(tree, keep=8)
        self.assertEqual(len(verified.tokens), 8)
        self.assertIn(-2, [int(item) for item in verified.parents.tolist()])
        ledger = ClearLedger()
        ledger.open = SimpleNamespace(k=0, w=0)
        remember_choice(SimpleNamespace(_clear_ledger=ledger), len(verified.tokens), 4)
        self.assertEqual(ledger.open.k, 8)

    def test_width_group_rotates_every_four_rounds(self):
        scheduler = self._scheduler()
        reqs = [SimpleNamespace(rid="clearfit-w-1-128-0")]
        widths = []
        phases = []
        for _ in range(5):
            info = fit.prefit_plan(scheduler, reqs, 1, 0, (256, 512))
            widths.append(info["width"])
            phases.append(info["phase"])
        self.assertEqual(widths, [4, 4, 4, 4, 16])
        self.assertEqual(phases, ["warmup", "keep", "keep", "keep", "warmup"])

    def test_ready_file_does_not_lock_clearfit_rids(self):
        scheduler = self._scheduler()
        scheduler._clear_fit_ready = True
        reqs = [SimpleNamespace(rid="clearfit-warm-1-0")]
        self.assertIsNone(fit.prefit_plan(scheduler, reqs, 1, 0, (256,)))

    def test_schedule_covers_the_tight_cell(self):
        cfg = prefit.PrefitConfig(
            url="http://127.0.0.1:30000",
            buckets=(256, 512, 1024, 2048, 4096),
            max_depth=8,
            ks=(2, 4, 8, 16, 32, 64),
            widths=(2, 4, 8, 16, 32, 64),
            batches=(1, 2, 4),
            token_range=(1000, 30000),
        )
        groups = prefit.build_schedule(cfg)
        self.assertEqual(len(groups), 109)
        self.assertEqual(groups[0]["kind"], "warm")
        self.assertEqual(groups[1]["kind"], "v")
        self.assertEqual(groups[1]["k"], 16)
        self.assertEqual(groups[-1]["kind"], "done")
        verify = [item for item in groups if item["kind"] == "v"]
        for item in verify:
            self.assertGreaterEqual(item["max_new"], 7 * (item["depth"] + 1))
        tight = next(
            item
            for item in verify
            if item["b"] == 1 and item["bucket"] == 256 and item["k"] == 64
        )
        self.assertEqual(tight["max_new"], 60)
        self.assertEqual(tight["depth"], 7)
        self.assertGreaterEqual(tight["max_new"], 56)
        roomy = next(
            item
            for item in verify
            if item["b"] == 4 and item["bucket"] == 4096 and item["k"] == 2
        )
        self.assertEqual(roomy["max_new"], 8 * (roomy["depth"] + 1))
        width_groups = [item for item in groups if item["kind"] == "w" and item["b"] == 1]
        self.assertEqual(width_groups[0]["max_new"], 160)

    def test_retry_attempt_restarts_warmup(self):
        scheduler = self._scheduler()
        reqs = [SimpleNamespace(rid="clearfit-v-1-128-16-0")]
        phases = [
            fit.prefit_plan(scheduler, reqs, 1, 0, (256,))["phase"] for _ in range(3)
        ]
        self.assertEqual(phases, ["warmup", "warmup", "keep"])
        retry = [SimpleNamespace(rid="clearfit-v-1-128-16-1")]
        info = fit.prefit_plan(scheduler, retry, 1, 0, (256,))
        self.assertEqual(info["phase"], "warmup")

    def test_body_error_still_sends_done(self):
        cfg = prefit.PrefitConfig(
            url="http://127.0.0.1:9",
            buckets=(256,),
            max_depth=8,
            ks=(8,),
            widths=(8,),
            batches=(1,),
            token_range=(1, 2),
        )
        groups = [
            {"kind": "v", "b": 1, "max_new": 8, "prefix": 8},
            {"kind": "done", "b": 1, "max_new": 1, "prefix": 8},
        ]
        seen = []

        def fake_group(group_cfg, group, rng):
            del group_cfg, rng
            seen.append(group["kind"])
            if group["kind"] == "v":
                raise RuntimeError("boom")
            return {"kind": group["kind"]}

        with patch.object(prefit, "build_schedule", return_value=groups), patch.object(
            prefit, "_wait_until"
        ), patch.object(prefit, "_run_group", side_effect=fake_group):
            with self.assertLogs(prefit.logger, level="INFO") as logs:
                with self.assertRaises(RuntimeError):
                    prefit.run_prefit(cfg)
        self.assertEqual(seen, ["v", "done"])
        self.assertTrue(any("elapsed_s=" in line for line in logs.output))

    def test_wait_timeout_does_not_send_done(self):
        cfg = prefit.PrefitConfig(
            url="http://127.0.0.1:9",
            buckets=(256,),
            max_depth=8,
            ks=(8,),
            widths=(8,),
            batches=(1,),
            token_range=(1, 2),
        )
        seen = []

        def fail_wait(url, timeout_s, ready):
            del url, timeout_s, ready
            raise TimeoutError("not ready")

        with patch.object(prefit, "_wait_until", side_effect=fail_wait), patch.object(
            prefit, "_run_group", side_effect=lambda *args, **kwargs: seen.append("ran")
        ):
            with self.assertRaises(TimeoutError):
                prefit.run_prefit(cfg)
        self.assertEqual(seen, [])

    def test_request_sends_bearer_only_when_set(self):
        captured = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b"{}"

        def fake_urlopen(req, timeout):
            captured["headers"] = {key.lower(): value for key, value in req.header_items()}
            captured["timeout"] = timeout
            return _Resp()

        with patch.object(prefit.urllib.request, "urlopen", fake_urlopen):
            prefit._request("http://127.0.0.1/generate", "POST", {"rid": "a"}, 1.0, "secret")
        self.assertEqual(captured["headers"].get("authorization"), "Bearer secret")
        with patch.object(prefit.urllib.request, "urlopen", fake_urlopen):
            prefit._request("http://127.0.0.1/model_info", "GET", None, 1.0, "")
        self.assertNotIn("authorization", captured["headers"])


class ClearTreeFitFileTest(CustomTestCase):
    def setUp(self):
        reset_policy()
        self._saved = {
            name: os.environ.get(name)
            for name in ("SGLANG_CLEAR_LEDGER", "SGLANG_CLEAR_FIT_PATH", "SGLANG_CLEAR_AUTO_PREFIT")
        }
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        self._dir = tempfile.TemporaryDirectory()
        self._path = str(Path(self._dir.name) / "clear_tree_fit.json")
        os.environ["SGLANG_CLEAR_FIT_PATH"] = self._path

    def tearDown(self):
        self._dir.cleanup()
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        reset_policy()

    def _scheduler(self):
        ledger = ClearLedger()
        return SimpleNamespace(
            server_args=_args(),
            tp_size=1,
            tp_rank=0,
            _clear_ledger=ledger,
            _clear_layer_capture_bs=(),
            model_worker=None,
            max_running_requests=4,
            model_config=SimpleNamespace(vocab_size=32000),
        )

    def test_file_round_trip_keeps_h_and_layers(self):
        scheduler = self._scheduler()
        for _ in range(4):
            policy.observe_verify(1, 8, 256, 12.5)
        model = _new_layer_model()
        for length, target in ((4.0, 1.0), (8.0, 1.4), (12.0, 1.8), (16.0, 2.2)):
            model.update([1.0, length], target)
        scheduler._clear_ledger.layer_models[(8, "graph", 1)] = model
        scheduler._clear_target_header = fit.target_info(scheduler)
        held = policy.predict_hold_ms(1, 8, 256)
        layer = model.predict([1.0, 16.0])
        self.assertTrue(fit.save_prefit(scheduler))
        payload = json.loads(Path(self._path).read_text())
        self.assertNotIn("q", payload)
        self.assertIn("h", payload)
        self.assertIn("layers", payload)

        reset_policy()
        scheduler._clear_ledger.layer_models = {}
        scheduler._clear_fit_installed = False
        fit.load_draft_part(scheduler)
        fit.on_target_info(scheduler, scheduler._clear_target_header)
        self.assertTrue(scheduler._clear_fit_ready)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 8, 256), held)
        loaded = scheduler._clear_ledger.layer_models[(8, "graph", 1)]
        self.assertAlmostEqual(loaded.predict([1.0, 16.0]), layer)

        loaded.update([1.0, 20.0], 3.0)
        fresh = _new_layer_model()
        fresh.from_dict(model.to_dict())
        fresh.update([1.0, 20.0], 3.0)
        self.assertAlmostEqual(loaded.predict([1.0, 20.0]), fresh.predict([1.0, 20.0]))

    def test_header_mismatch_refuses_the_file(self):
        scheduler = self._scheduler()
        policy.observe_verify(1, 8, 256, 12.5)
        scheduler._clear_target_header = fit.target_info(scheduler)
        fit.save_prefit(scheduler)
        payload = json.loads(Path(self._path).read_text())
        payload["header"]["ks"] = [2]
        Path(self._path).write_text(json.dumps(payload))
        reset_policy()
        fit.load_draft_part(scheduler)
        self.assertIsNone(scheduler._clear_fit_pending)
        fit.on_target_info(scheduler, fit.target_info(scheduler))
        self.assertFalse(scheduler._clear_fit_ready)
        self.assertIsNotNone(scheduler._clear_prefit)
        self.assertEqual(policy._clock, {})

    def test_failed_replace_keeps_the_old_file(self):
        Path(self._path).write_text('{"old": true}\n')
        scheduler = self._scheduler()
        policy.observe_verify(1, 8, 256, 12.5)
        scheduler._clear_target_header = fit.target_info(scheduler)
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_fit.os.replace",
            side_effect=OSError("disk"),
        ):
            with self.assertRaises(OSError):
                fit.save_prefit(scheduler)
        self.assertEqual(json.loads(Path(self._path).read_text()), {"old": True})

    def test_online_updates_do_not_write(self):
        scheduler = self._scheduler()
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_fit.os.replace"
        ) as repl:
            policy.observe_verify(1, 8, 256, 12.5)
            fit.on_target_info(scheduler, fit.target_info(scheduler))
            self.assertFalse(scheduler._clear_fit_ready)
            repl.assert_not_called()

    def test_finish_keeps_fits_and_writes_once(self):
        scheduler = self._scheduler()
        scheduler._clear_prefit = fit.PrefitState()
        policy.observe_verify(1, 8, 256, 12.5)
        policy.observe_price(1, 4.0, 2.0)
        model = _new_layer_model()
        model.update([1.0, 4.0], 1.0)
        scheduler._clear_ledger.layer_models[(8, "graph", 1)] = model
        scheduler._clear_target_header = fit.target_info(scheduler)
        self.assertIsNotNone(policy.theta(1))
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_fit.os.replace",
            wraps=os.replace,
        ) as repl:
            fit.maybe_finish(scheduler, "clearfit-done")
            fit.maybe_finish(scheduler, "clearfit-done")
            self.assertEqual(repl.call_count, 1)
        self.assertIsNone(policy.theta(1))
        self.assertEqual(policy._clock[(1, 8, 256)], [12.5])
        self.assertIn((8, "graph", 1), scheduler._clear_ledger.layer_models)
        self.assertTrue(scheduler._clear_fit_ready)
        self.assertIsNone(scheduler._clear_prefit)
        pending = scheduler._clear_fit_pending
        self.assertEqual(pending["target"], scheduler._clear_target_header)
        self.assertTrue(pending["h"])
        self.assertEqual(pending["layers"][0]["w"], 8)

    def test_full_grid_waits_for_done(self):
        scheduler = self._scheduler()
        scheduler._clear_prefit = fit.PrefitState()
        scheduler._clear_target_header = fit.target_info(scheduler)
        policy.observe_verify(1, 8, 256, 12.5)
        state = scheduler._clear_prefit
        cells = fit.expected_cells()
        for cell in cells:
            state.kept[cell] = 4
        last = cells[-1]
        state.kept[last] = 3
        with self.assertLogs(fit.logger, level="INFO") as logs:
            fit.note_kept(scheduler, *last)
            fit.note_kept(scheduler, *last)
        self.assertEqual(sum("cells complete" in line for line in logs.output), 1)
        self.assertTrue(state.cells_announced)
        self.assertFalse(state.finished)
        self.assertFalse(Path(self._path).is_file())
        fit.maybe_finish(scheduler, "cells")
        self.assertIs(scheduler._clear_prefit, state)
        self.assertFalse(Path(self._path).is_file())
        fit.maybe_finish(scheduler, "clearfit-done")
        self.assertTrue(Path(self._path).is_file())
        self.assertTrue(scheduler._clear_fit_ready)
        self.assertIsNone(scheduler._clear_prefit)

    def test_save_error_still_closes_the_sweep(self):
        scheduler = self._scheduler()
        scheduler._clear_prefit = fit.PrefitState()
        scheduler._clear_fit_pending = {"h": [], "layers": [], "target": {"backend": "old"}}
        policy.observe_verify(1, 8, 256, 12.5)
        policy.observe_price(1, 4.0, 2.0)
        scheduler._clear_target_header = fit.target_info(scheduler)
        with patch.object(fit, "save_prefit", side_effect=OSError("disk")):
            fit.maybe_finish(scheduler, "clearfit-done")
        self.assertIsNone(scheduler._clear_prefit)
        self.assertTrue(scheduler._clear_fit_ready)
        self.assertTrue(scheduler._clear_fit_installed)
        self.assertEqual(scheduler._clear_fit_pending["target"], {"backend": "old"})
        self.assertIsNone(policy.theta(1))
        self.assertEqual(policy._clock[(1, 8, 256)], [12.5])

    def test_old_target_header_reopens_prefit(self):
        scheduler = self._scheduler()
        scheduler.server_args.attention_backend = "triton"
        scheduler.server_args.disable_cuda_graph = True
        scheduler.server_args.cuda_graph_bs = [1, 2, 4]
        policy.observe_verify(1, 8, 256, 12.5)
        scheduler._clear_target_header = fit.target_info(scheduler)
        self.assertIsNotNone(fit.save_prefit(scheduler))
        payload = json.loads(Path(self._path).read_text())
        self.assertIn("disable_cuda_graph", payload["header"]["target"])
        self.assertIn("attention_backend", payload["header"]["draft"])
        self.assertTrue(payload["header"]["target"]["disable_cuda_graph"])
        for key in (
            "disable_cuda_graph",
            "attention_backend",
            "page_size",
            "cuda_graph_bs",
        ):
            payload["header"]["target"].pop(key, None)
        Path(self._path).write_text(json.dumps(payload))
        reset_policy()
        fit.load_draft_part(scheduler)
        self.assertIsNotNone(scheduler._clear_fit_pending)
        fit.on_target_info(scheduler, fit.target_info(scheduler))
        self.assertFalse(scheduler._clear_fit_ready)
        self.assertIsNotNone(scheduler._clear_prefit)

    def test_prefit_config_copies_api_key(self):
        scheduler = self._scheduler()
        scheduler.server_args.api_key = "secret"
        self.assertEqual(fit.prefit_config(scheduler).api_key, "secret")
        scheduler.server_args.api_key = "  "
        self.assertIsNone(fit.prefit_config(scheduler).api_key)

    def test_same_target_keeps_fits(self):
        scheduler = self._scheduler()
        header = fit.target_info(scheduler)
        scheduler._clear_target_header = header
        policy.observe_verify(1, 8, 256, 12.5)
        model = _new_layer_model()
        scheduler._clear_ledger.layer_models[(8, "graph", 1)] = model
        scheduler._clear_ledger.open = object()
        policy.on_external_event(
            scheduler, SimpleNamespace(target_info=dict(header))
        )
        self.assertEqual(policy._clock[(1, 8, 256)], [12.5])
        self.assertIn((8, "graph", 1), scheduler._clear_ledger.layer_models)
        self.assertIsNone(scheduler._clear_ledger.open)

        policy.on_external_event(
            scheduler, SimpleNamespace(target_info={"backend": "other", "tp": 2})
        )
        self.assertEqual(policy._clock, {})
        self.assertEqual(scheduler._clear_ledger.layer_models, {})

    def test_auto_prefit_starts_at_most_twice(self):
        scheduler = self._scheduler()
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_fit.threading.Thread",
            _Thread,
        ):
            fit.maybe_start_prefit(scheduler, True)
            fit.maybe_start_prefit(scheduler, None)
            self.assertFalse(hasattr(scheduler, "_clear_prefit_thread"))
            os.environ["SGLANG_CLEAR_AUTO_PREFIT"] = "0"
            fit.maybe_start_prefit(scheduler, False)
            self.assertFalse(hasattr(scheduler, "_clear_prefit_thread"))
            os.environ.pop("SGLANG_CLEAR_AUTO_PREFIT", None)
            fit.maybe_start_prefit(scheduler, False)
            self.assertTrue(scheduler._clear_prefit_thread.is_alive())
            self.assertEqual(scheduler._clear_prefit_runs, 1)
            fit.maybe_start_prefit(scheduler, False)
            self.assertEqual(scheduler._clear_prefit_runs, 1)
            scheduler._clear_prefit_thread._alive = False
            fit.maybe_start_prefit(scheduler, False)
            self.assertEqual(scheduler._clear_prefit_runs, 2)
            scheduler._clear_prefit_thread._alive = False
            fit.maybe_start_prefit(scheduler, False)
            self.assertEqual(scheduler._clear_prefit_runs, 2)


if __name__ == "__main__":
    unittest.main()
