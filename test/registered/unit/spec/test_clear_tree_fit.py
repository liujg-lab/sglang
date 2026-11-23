"""CPU checks for ClearTree H + Q prefit.

These tests stay on the protocol, the ledger split, and the fit file.
They do not start a server.
"""

import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError

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
    complete_verify_report,
    note_packet,
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
    attach_verify_report,
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


def _begin_scan(scheduler, run, session="s"):
    """Open a run the way CONTROL.begin does, so sampling has a registered group."""
    scheduler.sr_session_id = session
    plan = fit.draft_plan(scheduler)
    status = fit.begin_prefit(scheduler, run, plan.fingerprint, session, 0)
    if status.get("reason") != "begun":
        raise AssertionError(status)
    return status


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
        shaped = ClearBatchRequest(
            session_id="s",
            rpc_seq=3,
            action=ClearAction.STEP,
            target_verify_rpc_seq=2,
            target_verify_b=1,
            target_verify_k=16,
            target_verify_bucket=512,
            target_verify_fallback=False,
            target_verify_path="graph",
        )
        shaped_wire = shaped.to_dict()
        self.assertEqual(shaped_wire["target_verify_k"], 16)
        self.assertEqual(shaped_wire["target_verify_path"], "graph")
        decoded_shape = ClearBatchRequest.from_dict(shaped_wire)
        self.assertEqual(decoded_shape.target_verify_rpc_seq, 2)
        self.assertEqual(decoded_shape.target_verify_b, 1)
        self.assertEqual(decoded_shape.target_verify_bucket, 512)
        self.assertFalse(decoded_shape.target_verify_fallback)
        self.assertEqual(decoded_shape.target_verify_path, "graph")
        dropped_shape = ClearBatchRequest.from_dict(
            {
                "session_id": "s",
                "rpc_seq": 3,
                "action": "step",
                "target_verify_rpc_seq": 1.5,
                "target_verify_b": True,
                "target_verify_fallback": "no",
            }
        )
        self.assertIsNone(dropped_shape.target_verify_rpc_seq)
        self.assertIsNone(dropped_shape.target_verify_b)
        self.assertIsNone(dropped_shape.target_verify_fallback)
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
        self._curve_env = os.environ.pop("SGLANG_CLEAR_H_CURVE", None)

    def tearDown(self):
        if self._curve_env is None:
            os.environ.pop("SGLANG_CLEAR_H_CURVE", None)
        else:
            os.environ["SGLANG_CLEAR_H_CURVE"] = self._curve_env
        reset_policy()

    @staticmethod
    def _row(batch, k, bucket, hold, ell, n=4):
        return {
            "b": batch,
            "k": k,
            "bucket": bucket,
            "samples_ms": [float(hold)] * n,
            "ells": [int(ell)] * n,
        }

    def _two_buckets(self):
        policy.replace_hold_cells(
            [
                self._row(1, 16, 512, 40.0, 320),
                self._row(1, 16, 4096, 48.0, 2560),
            ]
        )

    def test_curve_interpolates_between_bucket_anchors(self):
        self._two_buckets()
        self.assertEqual(
            policy._hold_anchors[(1, 16)], [(320.0, 40.0), (2560.0, 48.0)]
        )
        expected = 40.0 + 8.0 * (2200 - 320) / (2560 - 320)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 16, 4096, 2200), expected)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 16, 4096, 2560), 48.0)
        # Without a prefix the bucket mean is unchanged.
        self.assertAlmostEqual(policy.predict_hold_ms(1, 16, 4096), 48.0)

    def test_switch_off_keeps_the_bucket_mean(self):
        self._two_buckets()
        os.environ["SGLANG_CLEAR_H_CURVE"] = "0"
        self.assertAlmostEqual(policy.predict_hold_ms(1, 16, 4096, 2200), 48.0)
        self.assertAlmostEqual(policy.predict_verify_ms(1, 16, 4096, 2200), 48.0)

    def test_curve_extrapolates_with_a_non_negative_end_slope(self):
        policy.replace_hold_cells(
            [
                self._row(1, 8, 512, 37.0, 320),
                self._row(1, 8, 4096, 42.0, 2560),
                self._row(1, 4, 512, 40.0, 320),
                self._row(1, 4, 4096, 38.0, 2560),
                self._row(1, 2, 4096, 35.0, 2560),
            ]
        )
        self.assertAlmostEqual(
            policy.predict_hold_ms(1, 8, 4096, 3000), 42.0 + 5.0 * 440 / 2240
        )
        rising = policy._hold_anchors[(1, 8)]
        self.assertAlmostEqual(
            policy._interp_anchor(rising, 100), 37.0 - 5.0 * 220 / 2240
        )
        # A falling end segment does not extrapolate downward or upward.
        falling = policy._hold_anchors[(1, 4)]
        self.assertAlmostEqual(policy._interp_anchor(falling, 3000), 38.0)
        self.assertAlmostEqual(policy._interp_anchor(falling, 100), 40.0)
        # One anchor is a constant.
        single = policy._hold_anchors[(1, 2)]
        self.assertAlmostEqual(policy._interp_anchor(single, 500), 35.0)

    def test_online_offset_moves_only_its_own_cell(self):
        self._two_buckets()
        base = policy.predict_hold_ms(1, 16, 4096, 2200)
        for _ in range(16):
            policy.observe_verify(1, 16, 4096, base - 1.0, 2200)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 16, 4096, 2200), base - 1.0)
        # The other bucket still prices from the curve alone.
        expected = 40.0 + 8.0 * (372 - 320) / (2560 - 320)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 16, 512, 372), expected)
        # Anchors do not move with online samples.
        self.assertEqual(
            policy._hold_anchors[(1, 16)], [(320.0, 40.0), (2560.0, 48.0)]
        )

    def test_offset_uses_the_spike_filter(self):
        self._two_buckets()
        base = policy.predict_hold_ms(1, 16, 4096, 2200)
        for _ in range(15):
            policy.observe_verify(1, 16, 4096, base, 2200)
        policy.observe_verify(1, 16, 4096, 400.0, 2200)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 16, 4096, 2200), base)

    def test_samples_without_prefix_do_not_move_the_offset(self):
        self._two_buckets()
        base = policy.predict_hold_ms(1, 16, 4096, 2200)
        for _ in range(16):
            policy.observe_verify(1, 16, 4096, base + 5.0)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 16, 4096, 2200), base)

    def test_k_stays_monotone_at_the_queried_prefix(self):
        policy.replace_hold_cells(
            [
                self._row(1, 8, 512, 40.0, 320),
                self._row(1, 8, 4096, 50.0, 2560),
                self._row(1, 16, 512, 41.0, 320),
                self._row(1, 16, 4096, 45.0, 2560),
            ]
        )
        self.assertAlmostEqual(policy.predict_hold_ms(1, 8, 4096, 2560), 47.5)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 16, 4096, 2560), 47.5)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 8, 512, 320), 40.0)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 16, 512, 320), 41.0)

    def test_a_cell_without_samples_is_not_priced(self):
        self._two_buckets()
        self.assertIsNone(policy.predict_hold_ms(1, 16, 1024, 700))

    def test_rows_without_prefixes_give_no_curve(self):
        policy.replace_hold_cells(
            [
                {"b": 1, "k": 16, "bucket": 512, "samples_ms": [40.0] * 4},
                {
                    "b": 1,
                    "k": 16,
                    "bucket": 4096,
                    "samples_ms": [48.0] * 4,
                    "ells": [None] * 4,
                },
            ]
        )
        self.assertEqual(policy._hold_anchors, {})
        self.assertAlmostEqual(policy.predict_hold_ms(1, 16, 4096, 2200), 48.0)

    def test_external_reset_drops_prefixes_and_anchors(self):
        self._two_buckets()
        policy.observe_verify(1, 16, 4096, 47.0, 2200)
        reset_policy()
        self.assertEqual(policy._clock_ell, {})
        self.assertEqual(policy._hold_anchors, {})

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

    def test_q_fits_on_observe_not_on_query(self):
        with patch.object(policy, "_fit_q_line", wraps=policy._fit_q_line) as fit_q:
            self.assertEqual(policy.predict_q_ms(1, 8), 0.0)
            policy.observe_q(1, 0, 20.0)
            policy.observe_q(1, 8, None)
            fit_q.assert_not_called()

            policy.observe_q(1, 8, 20.0)
            fit_q.assert_called_once()
            policy.observe_q(4, 8, 40.0)
            self.assertEqual(fit_q.call_count, 2)
            self.assertEqual(policy.predict_q_ms(1, 16), 20.0)

            policy.observe_q(1, 16, 24.0)
            self.assertEqual(fit_q.call_count, 3)
            self.assertGreater(policy.predict_q_ms(1, 16), 20.0)
            self.assertLess(policy.predict_q_ms(1, 8), policy.predict_q_ms(1, 16))
            self.assertEqual(policy.predict_q_ms(4, 16), 40.0)
            self.assertEqual(fit_q.call_count, 3)

    def test_q_prediction_matches_window_after_eviction(self):
        policy.observe_q(1, 8, 1.0)
        for idx in range(policy.Q_RES_WINDOW):
            policy.observe_q(1, 8, 5.0 + idx / 32.0)
            policy.observe_q(1, 16, 7.0 + idx / 32.0)
        policy.observe_q(1, 16, 100.0)

        self.assertNotIn(1.0, policy._q_raw[(1, 8)])
        kept = policy._q_kept(1)
        _, slope = policy._fit_q_line(kept)
        residuals = [ms - slope * float(k) for k, ms in kept]
        for k in (2, 8, 16, 64):
            self.assertEqual(
                policy.predict_q_ms(1, k), policy._median(residuals) + slope * k
            )

    def test_q_prediction_follows_reset_policy(self):
        policy.observe_q(1, 8, 20.0)
        reset_policy(keep_fits=True)
        self.assertEqual(policy.predict_q_ms(1, 8), 20.0)
        self.assertEqual(list(policy._q_raw[(1, 8)]), [20.0])
        reset_policy()
        self.assertEqual(policy.predict_q_ms(1, 8), 0.0)
        self.assertEqual(policy._q_raw, {})

    def test_q_outlier_stays_out_of_the_regression(self):
        policy.observe_q(1, 8, 20.0)
        policy.observe_q(1, 8, 20.0)
        policy.observe_q(1, 8, 200.0)
        self.assertIn(200.0, list(policy._q_raw[(1, 8)]))
        self.assertAlmostEqual(policy.predict_q_ms(1, 8), 20.0, places=3)

    def test_q_keeps_a_stable_k_gap(self):
        for _ in range(32):
            policy.observe_q(1, 2, 1.0)
            policy.observe_q(1, 64, 10.0)
        self.assertAlmostEqual(policy.predict_q_ms(1, 2), 1.0, places=2)
        self.assertAlmostEqual(policy.predict_q_ms(1, 64), 10.0, places=2)

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
            _clear_prefit=None,
            _clear_target_prefix={},
        )

    def test_verify_group_locks_depth_and_pads_to_k(self):
        scheduler = self._scheduler()
        _begin_scan(scheduler, "abcd12341")
        reqs = [
            SimpleNamespace(
                rid="clearfit-abcd12341-v-1-128-64-0",
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
        _begin_scan(scheduler, "abcd12341")
        reqs = [SimpleNamespace(rid="clearfit-abcd12341-w-1-128-0")]
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
        reqs = [SimpleNamespace(rid="clearfit-abcd12341-warm-1-0")]
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
        plan = prefit.build_plan(
            prefit.PlanInputs(
                batches=cfg.batches,
                ks=cfg.ks,
                widths=cfg.widths,
                buckets=cfg.buckets,
                max_depth=cfg.max_depth,
            )
        )
        groups = plan.groups
        self.assertEqual(len(groups), 108)
        self.assertEqual(groups[0]["kind"], "warm")
        self.assertEqual(groups[1]["kind"], "v")
        self.assertEqual(groups[1]["k"], 16)
        self.assertNotEqual(groups[-1]["kind"], "done")
        self.assertEqual(len(plan.required_h), 90)
        verify = [item for item in groups if item["kind"] == "v"]
        for item in verify:
            self.assertGreater(item["max_new"], 1 + 7 * (item["depth"] + 1))
            previous = max((b for b in cfg.buckets if b < item["bucket"]), default=0)
            lo, hi = prefit.recorded_span(item["prefix"], item["k"], item["max_new"])
            self.assertLess(previous, lo)
            self.assertLessEqual(hi, item["bucket"])
        tight = next(
            item
            for item in verify
            if item["b"] == 1 and item["bucket"] == 256 and item["k"] == 64
        )
        self.assertEqual(tight["prefix"], 128)
        self.assertEqual(tight["max_new"], 60)
        self.assertEqual(tight["depth"], 7)
        roomy = next(
            item
            for item in verify
            if item["b"] == 4 and item["bucket"] == 4096 and item["k"] == 2
        )
        self.assertEqual(roomy["max_new"], 8 * (roomy["depth"] + 1))
        width_groups = [item for item in groups if item["kind"] == "w" and item["b"] == 1]
        self.assertEqual(width_groups[0]["max_new"], 160)

    def test_verify_group_moves_prefix_down_to_fit_budget(self):
        plan = prefit.build_plan(prefit.PlanInputs((1,), (8,), (8,), (52,), 8))
        group = next(group for group in plan.groups if group["kind"] == "v")
        self.assertEqual(plan.required_h, [(1, 8, 52)])
        self.assertEqual((group["prefix"], group["max_new"]), (24, 16))
        self.assertEqual(prefit.recorded_span(24, 8, 16), (33, 49))

    def test_verify_group_retries_until_every_request_has_eight_verifies(self):
        cfg = prefit.PrefitConfig(
            url="http://127.0.0.1:9",
            buckets=(256,),
            max_depth=8,
            ks=(8,),
            widths=(4,),
            batches=(2,),
            token_range=(1, 2),
        )
        group = next(g for g in prefit.plan_from_config(cfg).groups if g["kind"] == "v")
        responses = [
            [{"meta_info": {"completion_tokens": 20}}] * 2,
            [{"meta_info": {"spec_verify_ct": count}} for count in (8, 7)],
            [{"meta_info": {"spec_verify_ct": 8}}] * 2,
        ]
        with patch.object(prefit, "_request", side_effect=responses) as request:
            result = prefit._run_group(cfg, group, random.Random(0), "run", 0)
        self.assertEqual((result["steps"], result["retries"]), (8, 3))
        sent = [call.args[2] for call in request.call_args_list]
        rids = [rid for payload in sent for rid in payload["rid"]]
        self.assertEqual(len(set(rids)), 6)
        self.assertTrue(
            all(p["sampling_params"]["max_new_tokens"] == group["max_new"] for p in sent)
        )

    def test_retry_attempt_restarts_warmup(self):
        scheduler = self._scheduler()
        _begin_scan(scheduler, "abcd12341")
        reqs = [SimpleNamespace(rid="clearfit-abcd12341-v-1-128-16-0")]
        phases = [
            fit.prefit_plan(scheduler, reqs, 1, 0, (256,))["phase"] for _ in range(3)
        ]
        self.assertEqual(phases, ["warmup", "warmup", "keep"])
        retry = [SimpleNamespace(rid="clearfit-abcd12341-v-1-128-16-1")]
        info = fit.prefit_plan(scheduler, retry, 1, 0, (256,))
        self.assertEqual(info["phase"], "warmup")

    def test_body_error_does_not_commit(self):
        cfg = prefit.PrefitConfig(
            url="http://127.0.0.1:9",
            buckets=(256,),
            max_depth=8,
            ks=(8,),
            widths=(8,),
            batches=(1,),
            token_range=(1, 2),
        )
        plan = prefit.PrefitPlan(
            inputs=prefit.PlanInputs((1,), (8,), (8,), (256,), 8),
            groups=[{"kind": "v", "b": 1, "max_new": 8, "prefix": 8, "k": 8, "bucket": 256}],
            required_h=[(1, 8, 256)],
            skipped_h=[],
            fingerprint="abcd1234",
        )
        seen = []

        def fake_group(group_cfg, group, rng, run, n0):
            del group_cfg, rng, run, n0
            seen.append(group["kind"])
            raise RuntimeError("boom")

        def fake_control(group_cfg, run, kind, seq, rng, fingerprint=""):
            del group_cfg, run, seq, rng, fingerprint
            seen.append(kind)
            return True

        with patch.object(prefit, "_wait_until"), patch.object(
            prefit, "_run_group", side_effect=fake_group
        ), patch.object(prefit, "_post_control", side_effect=fake_control):
            result = prefit.run_prefit(cfg, plan=plan, wait_status=lambda *args: None)
        self.assertEqual(result, "request_error")
        self.assertEqual(seen, ["begin", "v", "abandon", "begin", "v", "abandon"])

    def test_prefit_waits_for_health_then_idle(self):
        cfg = prefit.PrefitConfig(
            url="http://127.0.0.1:9",
            buckets=(256,),
            max_depth=8,
            ks=(8,),
            widths=(8,),
            batches=(1,),
            token_range=(1, 2),
            ready_timeout_s=2.0,
            api_key="test-key",
        )
        idle = {"num_running_reqs": 0, "num_waiting_reqs": 0}
        for eventually_ready in (False, True):
            with self.subTest(eventually_ready=eventually_ready):
                events = []
                now = 0.0
                loads = iter(
                    [dict(idle, num_running_reqs=1), dict(idle, num_waiting_reqs=1), idle]
                )

                def request(url, method, _payload, _timeout, api_key):
                    self.assertEqual((method, api_key), ("GET", "test-key"))
                    path = url.removeprefix(cfg.url)
                    events.append(path)
                    if path == "/model_info":
                        return {}
                    if path == "/health":
                        if not eventually_ready or events.count(path) == 1:
                            raise HTTPError(url, 503, "Starting", None, None)
                        return {}
                    return {"loads": [next(loads) if eventually_ready else idle]}

                def sleep(delay):
                    nonlocal now
                    now += delay

                def exchange(_cfg, _plan, _run, op, *_args, **_kwargs):
                    events.append(op)
                    return {"reason": "begun"} if op == "begin" else {"ready": True}

                def group(*_args):
                    events.append("group")
                    return {"retries": 1}

                with patch.object(prefit, "_request", side_effect=request), patch.object(
                    prefit.time, "monotonic", side_effect=lambda: now
                ), patch.object(prefit.time, "sleep", side_effect=sleep), patch.object(
                    prefit, "_exchange", side_effect=exchange
                ), patch.object(prefit, "_run_group", side_effect=group):
                    if eventually_ready:
                        self.assertEqual(prefit.run_prefit(cfg), "ready")
                    else:
                        with self.assertRaises(TimeoutError):
                            prefit.run_prefit(cfg)
                if eventually_ready:
                    self.assertEqual(
                        events,
                        ["/health"] * 2 + ["/v1/loads"] * 3 + ["begin", "group", "group", "commit"],
                    )
                else:
                    self.assertEqual(events, ["/health"] * 4)

    def test_idle_requires_valid_counts_from_every_load_shard(self):
        idle = {"num_running_reqs": 0, "num_waiting_reqs": 0}
        cases = [
            ({}, False),
            ({"loads": []}, False),
            ({"loads": [idle, {}]}, False),
            ({"loads": [idle, "invalid"]}, False),
            ({"loads": [dict(idle, num_waiting_reqs=None)]}, False),
            ({"loads": [dict(idle, num_running_reqs=False)]}, False),
            ({"loads": [idle, dict(idle, num_waiting_reqs=1)]}, False),
            ({"loads": [idle, idle]}, True),
        ]
        for body, expected in cases:
            with self.subTest(body=body), patch.object(prefit, "_request", return_value=body):
                self.assertEqual(prefit._server_idle("http://127.0.0.1:9"), expected)

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
            for name in (
                "SGLANG_CLEAR_LEDGER",
                "SGLANG_CLEAR_FIT_PATH",
                "SGLANG_CLEAR_AUTO_PREFIT",
                "SGLANG_CLEAR_NPU_KV_BUCKETS",
                "SGLANG_CLEAR_LAYER_SLOPE_PRIOR",
            )
        }
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        os.environ.pop("SGLANG_CLEAR_LAYER_SLOPE_PRIOR", None)
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

    def _tiny(self, scheduler, *, batches=1, ks=None, capture=()):
        os.environ["SGLANG_CLEAR_NPU_KV_BUCKETS"] = "256"
        scheduler.max_running_requests = int(batches)
        scheduler.server_args.clear_tree_ks = list(ks or [8])
        scheduler.server_args.clear_tree_widths = [8]
        scheduler._clear_layer_capture_bs = tuple(capture)
        scheduler._clear_layer_capture_pairs = tuple(
            (int(width), int(bs))
            for bs in capture
            for width in scheduler.server_args.clear_tree_widths
        )
        scheduler._clear_target_header = fit.target_info(scheduler)
        scheduler._clear_fit_ready = False
        scheduler._clear_fit_installed = False
        scheduler._clear_prefit = None
        scheduler._clear_prefit_seen = []
        return fit.draft_plan(scheduler).fingerprint + "1"

    def _fill(self, scheduler, run):
        state, stale = fit._adopt_run(scheduler, run)
        self.assertFalse(stale)
        for cell in state.required_h:
            for _ in range(4):
                fit.prefit_record_h(scheduler, *cell, 12.5)
        for cell in state.required_layers:
            for length, target in ((4.0, 1.0), (8.0, 1.4), (12.0, 1.8), (16.0, 2.2)):
                fit.prefit_record_layer(
                    scheduler, cell[0], cell[1], cell[2], [1.0, length], target
                )
        return state

    def test_file_round_trip_keeps_h_and_layers(self):
        scheduler = self._scheduler()
        run = self._tiny(scheduler, ks=[16], capture=(1,))
        self._fill(scheduler, run)
        model = scheduler._clear_prefit.layers[(8, "graph", 1)]
        layer = model.predict([1.0, 16.0])
        status = fit.commit_prefit(scheduler, run, 1)
        self.assertTrue(status["ready"])
        self.assertTrue(status["written"])
        payload = json.loads(Path(self._path).read_text())
        self.assertEqual(payload["schema"], 3)
        self.assertNotIn("q", payload)
        for row in payload["h"]:
            self.assertEqual(len(row["ells"]), len(row["samples_ms"]))
        self.assertEqual(payload["run"], run)
        self.assertIn("plan", payload)
        self.assertIn("coverage", payload)
        held = policy.predict_hold_ms(1, 16, 256)

        reset_policy()
        scheduler._clear_ledger.layer_models = {}
        scheduler._clear_fit_installed = False
        scheduler._clear_fit_ready = False
        fit.load_draft_part(scheduler)
        fit.on_target_info(scheduler, scheduler._clear_target_header)
        self.assertTrue(scheduler._clear_fit_ready)
        self.assertAlmostEqual(policy.predict_hold_ms(1, 16, 256), held)
        loaded = scheduler._clear_ledger.layer_models[(8, "graph", 1)]
        self.assertAlmostEqual(loaded.predict([1.0, 16.0]), layer)

    def test_file_round_trip_keeps_prefixes_and_anchors(self):
        scheduler = self._scheduler()
        run = self._tiny(scheduler, ks=[16], capture=(1,))
        state, stale = fit._adopt_run(scheduler, run)
        self.assertFalse(stale)
        for cell in state.required_h:
            for step in range(4):
                fit.prefit_record_h(scheduler, *cell, 12.5, ell=100 + 2 * step)
        for cell in state.required_layers:
            for length, target in ((4.0, 1.0), (8.0, 1.4), (12.0, 1.8), (16.0, 2.2)):
                fit.prefit_record_layer(
                    scheduler, cell[0], cell[1], cell[2], [1.0, length], target
                )
        status = fit.commit_prefit(scheduler, run, 1)
        self.assertTrue(status["ready"])
        payload = json.loads(Path(self._path).read_text())
        row = payload["h"][0]
        self.assertEqual(row["ells"], [100, 102, 104, 106])
        anchors = dict(policy._hold_anchors)
        self.assertEqual(anchors[(1, 16)], [(103.0, 12.5)])

        reset_policy()
        scheduler._clear_ledger.layer_models = {}
        scheduler._clear_fit_installed = False
        scheduler._clear_fit_ready = False
        fit.load_draft_part(scheduler)
        fit.on_target_info(scheduler, scheduler._clear_target_header)
        self.assertTrue(scheduler._clear_fit_ready)
        self.assertEqual(policy._hold_anchors, anchors)
        self.assertEqual(policy._clock_ell[(1, 16, 256)], [100, 102, 104, 106])

    def test_schema_three_needs_matching_ells(self):
        scheduler = self._scheduler()
        run = self._tiny(scheduler, ks=[16], capture=(1,))
        self._fill(scheduler, run)
        fit.commit_prefit(scheduler, run, 1)
        payload = json.loads(Path(self._path).read_text())
        for broken in (None, [1]):
            if broken is None:
                payload["h"][0].pop("ells", None)
            else:
                payload["h"][0]["ells"] = broken
            Path(self._path).write_text(json.dumps(payload))
            reset_policy()
            scheduler._clear_fit_installed = False
            scheduler._clear_fit_ready = False
            scheduler._clear_fit_pending = None
            fit.load_draft_part(scheduler)
            fit.on_target_info(scheduler, scheduler._clear_target_header)
            self.assertFalse(scheduler._clear_fit_ready)

    def test_online_samples_replayed_after_commit_keep_prefixes(self):
        scheduler = self._scheduler()
        run = self._tiny(scheduler, ks=[16], capture=(1,))
        self._fill(scheduler, run)
        fit.note_online_h(scheduler, 1, 16, 256, 30.0, ell=150)
        self.assertTrue(fit.commit_prefit(scheduler, run, 1)["ready"])
        self.assertEqual(policy._clock[(1, 16, 256)][-1], 30.0)
        self.assertEqual(policy._clock_ell[(1, 16, 256)][-1], 150)

    def test_commit_and_file_install_freeze_the_prefit_slope(self):
        scheduler = self._scheduler()
        run = self._tiny(scheduler, ks=[16], capture=(1,))
        self._fill(scheduler, run)
        key = (8, "graph", 1)
        prefit_slope = scheduler._clear_prefit.layers[key].coefficients()[1]
        fit.note_online_layer(scheduler, 8, "graph", 1, [1.0, 20.0], 3.0)
        self.assertTrue(fit.commit_prefit(scheduler, run, 1)["ready"])
        committed = scheduler._clear_ledger.layer_models[key]
        self.assertEqual(committed.n, 5)
        self.assertIsNotNone(committed.slope_prior)
        self.assertAlmostEqual(committed.slope_prior[1], prefit_slope)
        self.assertGreater(committed.slope_prior[2], 0.0)

        reset_policy()
        scheduler._clear_ledger.layer_models = {}
        scheduler._clear_fit_installed = False
        scheduler._clear_fit_ready = False
        fit.load_draft_part(scheduler)
        fit.on_target_info(scheduler, scheduler._clear_target_header)
        self.assertTrue(scheduler._clear_fit_ready)
        loaded = scheduler._clear_ledger.layer_models[key]
        self.assertIsNotNone(loaded.slope_prior)
        self.assertAlmostEqual(loaded.slope_prior[1], prefit_slope)

    def test_slope_prior_switch_off_installs_plain_fits(self):
        os.environ["SGLANG_CLEAR_LAYER_SLOPE_PRIOR"] = "0"
        scheduler = self._scheduler()
        run = self._tiny(scheduler, ks=[16], capture=(1,))
        self._fill(scheduler, run)
        self.assertTrue(fit.commit_prefit(scheduler, run, 1)["ready"])
        for model in scheduler._clear_ledger.layer_models.values():
            self.assertIsNone(model.slope_prior)

    def test_header_mismatch_refuses_the_file(self):
        scheduler = self._scheduler()
        run = self._tiny(scheduler)
        self._fill(scheduler, run)
        self.assertTrue(fit.commit_prefit(scheduler, run, 1)["ready"])
        payload = json.loads(Path(self._path).read_text())
        payload["header"]["ks"] = [2]
        Path(self._path).write_text(json.dumps(payload))
        reset_policy()
        scheduler._clear_fit_installed = False
        scheduler._clear_fit_pending = None
        fit.load_draft_part(scheduler)
        self.assertIsNone(scheduler._clear_fit_pending)
        fit.on_target_info(scheduler, fit.target_info(scheduler))
        self.assertFalse(scheduler._clear_fit_ready)
        self.assertEqual(policy._clock, {})

    def test_failed_replace_keeps_the_old_file(self):
        Path(self._path).write_text('{"old": true}\n')
        scheduler = self._scheduler()
        run = self._tiny(scheduler)
        self._fill(scheduler, run)
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_fit.os.replace",
            side_effect=OSError("disk"),
        ):
            status = fit.commit_prefit(scheduler, run, 1)
        self.assertFalse(status["ready"])
        self.assertEqual(status["reason"], "write_failed")
        self.assertFalse(scheduler._clear_fit_ready)
        self.assertEqual(json.loads(Path(self._path).read_text()), {"old": True})
        self.assertFalse(os.path.exists(self._path + ".tmp"))

    def test_online_updates_do_not_write(self):
        scheduler = self._scheduler()
        run = self._tiny(scheduler)
        self._fill(scheduler, run)
        policy.observe_verify(1, 2, 256, 99.0)
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_fit.os.replace",
            wraps=os.replace,
        ) as repl:
            status = fit.commit_prefit(scheduler, run, 1)
            self.assertEqual(repl.call_count, 1)
        self.assertTrue(status["ready"])
        payload = json.loads(Path(self._path).read_text())
        cells = {(row["b"], row["k"], row["bucket"]) for row in payload["h"]}
        self.assertNotIn((1, 2, 256), cells)
        self.assertNotIn(99.0, payload["h"][0]["samples_ms"])

    def test_repeat_commit_writes_once(self):
        scheduler = self._scheduler()
        run = self._tiny(scheduler)
        self._fill(scheduler, run)
        policy.observe_price(1, 4.0, 2.0)
        self.assertIsNotNone(policy.theta(1))
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_fit.os.replace",
            wraps=os.replace,
        ) as repl:
            first = fit.commit_prefit(scheduler, run, 1)
            second = fit.commit_prefit(scheduler, run, 2)
            self.assertEqual(repl.call_count, 1)
        self.assertTrue(first["ready"])
        self.assertTrue(first["written"])
        self.assertTrue(second["ready"])
        self.assertEqual(second["seq"], 2)
        self.assertIsNone(policy.theta(1))
        self.assertEqual(policy._clock[(1, 8, 256)], [12.5, 12.5, 12.5, 12.5])
        self.assertTrue(scheduler._clear_fit_ready)

    def test_missing_cell_does_not_write(self):
        scheduler = self._scheduler()
        run = self._tiny(scheduler, ks=[16], capture=(1,))
        state, _stale = fit._adopt_run(scheduler, run)
        for cell in state.required_h:
            for _ in range(3):
                fit.prefit_record_h(scheduler, *cell, 12.5)
        status = fit.commit_prefit(scheduler, run, 1)
        self.assertFalse(status["ready"])
        self.assertEqual(status["reason"], "missing")
        self.assertTrue(status["missing_h"] or status["missing_layers"])
        self.assertFalse(Path(self._path).is_file())
        self.assertFalse(scheduler._clear_fit_ready)
        for cell in state.required_h:
            fit.prefit_record_h(scheduler, *cell, 12.5)
        self._fill_layers = None
        for cell in state.required_layers:
            for length, target in ((4.0, 1.0), (8.0, 1.4), (12.0, 1.8), (16.0, 2.2)):
                fit.prefit_record_layer(
                    scheduler, cell[0], cell[1], cell[2], [1.0, length], target
                )
        layer_key = state.required_layers[0]
        state.layers[layer_key].n = 3
        short = fit.commit_prefit(scheduler, run, 2)
        self.assertEqual(short["reason"], "missing")
        self.assertTrue(short["missing_layers"])
        self.assertFalse(Path(self._path).is_file())
        self.assertFalse(scheduler._clear_fit_ready)

    def test_eight_verifies_still_refill_when_draft_is_short(self):
        buckets = (256, 320)
        with patch.dict(os.environ, {"SGLANG_NPU_TREE_GRAPH_KV_BUCKETS": "256,320"}):
            scheduler = self._scheduler()
            self._tiny(scheduler)
            scheduler.sr_session_id = "sess"
            cfg = fit.prefit_config(scheduler)
            scanned = []
            commits = []

            def generate(_url, _method, payload, _timeout, _api_key):
                rid = payload["rid"][0]
                parsed = fit.parse_rid(rid)
                if parsed["kind"] == "v":
                    scan = fit.prefit_plan(
                        scheduler, [SimpleNamespace(rid=rid)], 1, 0, buckets
                    )
                    self.assertEqual(scan["phase"], "warmup")
                    bucket = scan["bucket"]
                    self.assertEqual(parsed["prefix"], 248 if bucket == 320 else 128)
                    self.assertFalse(Path(self._path).exists())
                    count = 4 if bucket == 256 else 1 if bucket in scanned else 3
                    scanned.append(bucket)
                    for _ in range(count):
                        fit.prefit_record_h(scheduler, 1, 8, bucket, 12.5, scan)
                return [{"meta_info": {"spec_verify_ct": 8}}]

            def control(op, run, seq, fingerprint):
                if op == "begin":
                    return fit.begin_prefit(scheduler, run, fingerprint, "sess", seq)
                self.assertEqual(op, "commit")
                status = fit.commit_prefit(scheduler, run, seq, "sess", fingerprint)
                commits.append(status)
                return status

            with (
                patch.object(prefit, "_wait_until"),
                patch.object(prefit, "_request", side_effect=generate),
                patch.object(fit, "_store_fit", wraps=fit._store_fit) as store,
            ):
                result = prefit.run_prefit(cfg, submit_control=control, session_id="sess")
            store.assert_called_once()

        self.assertEqual(result, "ready")
        self.assertEqual(scanned, [256, 320, 320])
        self.assertEqual([status["reason"] for status in commits], ["missing", "ready"])
        self.assertEqual(commits[0]["missing_h"], [[1, 8, 320]])
        self.assertEqual(commits[0]["refill_groups"], [["v", 1, 248, 8]])

    def test_old_target_header_reopens_prefit(self):
        scheduler = self._scheduler()
        scheduler.server_args.attention_backend = "triton"
        scheduler.server_args.disable_cuda_graph = True
        scheduler.server_args.cuda_graph_bs = [1, 2, 4]
        run = self._tiny(scheduler)
        self._fill(scheduler, run)
        self.assertTrue(fit.commit_prefit(scheduler, run, 1)["ready"])
        payload = json.loads(Path(self._path).read_text())
        self.assertIn("disable_cuda_graph", payload["header"]["target"])
        self.assertIn("attention_backend", payload["header"]["draft"])
        self.assertTrue(payload["header"]["target"]["disable_cuda_graph"])
        self.assertNotIn("prefit_batches", payload["header"]["target"])
        for key in (
            "disable_cuda_graph",
            "attention_backend",
            "page_size",
            "cuda_graph_bs",
        ):
            payload["header"]["target"].pop(key, None)
        Path(self._path).write_text(json.dumps(payload))
        reset_policy()
        scheduler._clear_fit_installed = False
        fit.load_draft_part(scheduler)
        self.assertIsNotNone(scheduler._clear_fit_pending)
        fit.on_target_info(scheduler, fit.target_info(scheduler))
        self.assertFalse(scheduler._clear_fit_ready)

    def test_prefit_config_copies_api_key(self):
        scheduler = self._scheduler()
        scheduler.server_args.api_key = "secret"
        self.assertEqual(fit.prefit_config(scheduler).api_key, "secret")
        scheduler.server_args.api_key = "  "
        self.assertIsNone(fit.prefit_config(scheduler).api_key)

    def test_prefit_driver_publishes_terminal_state_for_its_run(self):
        class ImmediateThread(_Thread):
            def start(self):
                self.target()

        for current_session in ("sess", "new-session"):
            with self.subTest(session=current_session):
                scheduler = self._scheduler()
                scheduler.sr_session_id = "sess"

                def run_prefit(_cfg, *, submit_control, **_kwargs):
                    submit_control("begin", "run", 0, "fingerprint")
                    scheduler.sr_session_id = current_session
                    return "ready"

                with patch.object(fit.threading, "Thread", ImmediateThread), patch.object(
                    prefit, "run_prefit", side_effect=run_prefit
                ), patch.dict(os.environ, {"SGLANG_CLEAR_AUTO_PREFIT": "1"}):
                    self.assertTrue(fit.start_prefit_driver(scheduler))

                expected = (
                    ("sess", "ready", "prefit", "run")
                    if current_session == "sess"
                    else ("sess", "prefitting", "need_prefit", None)
                )
                self.assertEqual(scheduler._clear_fit_snapshot, expected)

    def test_no_pending_fit_keeps_header_without_building_plan(self):
        scheduler = self._scheduler()
        header = fit.target_info(scheduler)
        scheduler._clear_fit_ready = True
        scheduler._clear_fit_installed = True
        scheduler._clear_fit_plan_fp = "old"
        active_run = object()
        scheduler._clear_prefit = active_run
        model = _new_layer_model()
        scheduler._clear_ledger.layer_models[(8, "graph", 1)] = model
        policy.observe_verify(1, 8, 256, 12.5)
        policy.observe_q(1, 8, 3.0)

        with patch.object(fit, "draft_plan") as plan, patch.object(fit.logger, "info") as log:
            fit.on_target_info(scheduler, header)
            fit.on_target_info(scheduler, dict(header, page_size=128))
        plan.assert_not_called()
        log.assert_not_called()
        self.assertEqual(scheduler._clear_target_header["page_size"], 128)
        self.assertFalse(scheduler._clear_fit_ready)
        self.assertFalse(scheduler._clear_fit_installed)
        self.assertIsNone(scheduler._clear_fit_plan_fp)
        self.assertIs(scheduler._clear_prefit, active_run)
        self.assertIs(scheduler._clear_ledger.layer_models[(8, "graph", 1)], model)
        self.assertEqual(policy.predict_hold_ms(1, 8, 256), 12.5)
        self.assertEqual(policy.predict_q_ms(1, 8), 3.0)

    def test_pending_fit_is_revalidated_with_one_plan_and_logs_new_rejections(self):
        scheduler = self._scheduler()
        header = fit.target_info(scheduler)
        scheduler._clear_fit_pending = {"payload": {"schema": 2}, "target": header}
        with patch.object(fit, "draft_plan", wraps=fit.draft_plan) as plan, patch.object(
            fit.logger, "info"
        ) as log:
            fit.on_target_info(scheduler, header)
            fit.on_target_info(scheduler, dict(header))
            self.assertEqual(log.call_count, 1)
            fit.on_target_info(scheduler, dict(header, page_size=128))
        self.assertEqual(plan.call_count, 3)
        self.assertEqual(log.call_count, 2)
        self.assertFalse(scheduler._clear_fit_ready)

    def test_same_target_keeps_fits(self):
        scheduler = self._scheduler()
        header = fit.target_info(scheduler)
        scheduler._clear_target_header = header
        policy.observe_verify(1, 8, 256, 12.5)
        policy.observe_q(1, 8, 3.0)
        model = _new_layer_model()
        scheduler._clear_ledger.layer_models[(8, "graph", 1)] = model
        scheduler._clear_ledger.open = object()
        policy.on_external_event(
            scheduler, SimpleNamespace(target_info=dict(header))
        )
        self.assertEqual(policy._clock[(1, 8, 256)], [12.5])
        self.assertEqual(policy.predict_q_ms(1, 8), 3.0)
        self.assertIn((8, "graph", 1), scheduler._clear_ledger.layer_models)
        self.assertIsNone(scheduler._clear_ledger.open)

        policy.on_external_event(
            scheduler, SimpleNamespace(target_info={"backend": "other", "tp": 2})
        )
        self.assertEqual(policy._clock, {})
        self.assertEqual(policy._q_raw, {})
        self.assertEqual(policy.predict_q_ms(1, 8), 0.0)
        self.assertEqual(scheduler._clear_ledger.layer_models, {})

    def test_auto_prefit_starts_from_the_handshake(self):
        scheduler = self._scheduler()
        calls = []

        def _query():
            calls.append(scheduler._clear_fit_state)
            return SimpleNamespace(fit_status={"state": "need_prefit", "ready": False})

        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_fit.threading.Thread",
            _Thread,
        ):
            fit.fit_idle(scheduler, 0.0, _query)
            self.assertEqual(scheduler._clear_fit_state, "prefitting")
            self.assertTrue(scheduler._clear_prefit_thread.is_alive())
            fit.fit_idle(scheduler, 1.0, _query)
            self.assertEqual(calls, ["querying"])
            os.environ["SGLANG_CLEAR_AUTO_PREFIT"] = "0"
            scheduler._clear_fit_state = "querying"
            scheduler._clear_prefit_thread = None
            fit.apply_fit_query(
                scheduler,
                SimpleNamespace(fit_status={"state": "need_prefit"}),
                2.0,
            )
        self.assertEqual(scheduler._clear_fit_state, "failed")
        self.assertEqual(scheduler._clear_fit_reason, "auto_prefit_off")

    def test_stale_and_user_rounds_stay_out_of_the_run(self):
        scheduler = self._scheduler()
        run = self._tiny(scheduler)
        fit._adopt_run(scheduler, run)
        mixed = fit.prefit_plan(
            scheduler,
            [
                SimpleNamespace(rid=f"clearfit-{run}-v-1-128-8-0"),
                SimpleNamespace(rid="user-1"),
            ],
            2,
            0,
            (256,),
        )
        self.assertEqual(mixed["phase"], "mixed")
        other = fit.draft_plan(scheduler).fingerprint + "2"
        fit._adopt_run(scheduler, other)
        stale = fit.prefit_plan(
            scheduler,
            [SimpleNamespace(rid=f"clearfit-{run}-v-1-128-8-0")],
            1,
            0,
            (256,),
        )
        self.assertEqual(stale["phase"], "stale")
        self.assertEqual(scheduler._clear_prefit.h, {})
        scheduler._clear_fit_ready = True
        fit.abandon_prefit(scheduler, run)
        self.assertTrue(scheduler._clear_fit_ready)
        self.assertEqual(scheduler._clear_prefit.run, other)
        policy.observe_verify(1, 2, 256, 99.0)
        self._fill(scheduler, other)
        fit.commit_prefit(scheduler, other, 1)
        payload = json.loads(Path(self._path).read_text())
        self.assertNotIn(99.0, payload["h"][0]["samples_ms"])

    def test_schema_one_and_narrow_cache_need_prefit(self):
        scheduler = self._scheduler()
        run = self._tiny(scheduler, batches=1)
        self._fill(scheduler, run)
        fit.commit_prefit(scheduler, run, 1)
        payload = json.loads(Path(self._path).read_text())
        payload["schema"] = 1
        payload["header"]["schema"] = 1
        Path(self._path).write_text(json.dumps(payload))
        reset_policy()
        scheduler._clear_fit_installed = False
        scheduler._clear_fit_pending = None
        fit.load_draft_part(scheduler)
        fit.on_target_info(scheduler, fit.target_info(scheduler))
        self.assertFalse(scheduler._clear_fit_ready)

        payload["schema"] = fit._FIT_SCHEMA
        payload["header"]["schema"] = fit._FIT_SCHEMA
        Path(self._path).write_text(json.dumps(payload))
        scheduler.max_running_requests = 4
        scheduler._clear_target_header = fit.target_info(scheduler)
        scheduler._clear_fit_installed = False
        fit.load_draft_part(scheduler)
        fit.on_target_info(scheduler, scheduler._clear_target_header)
        self.assertFalse(scheduler._clear_fit_ready)

    def test_plan_mismatch_and_batch_limit(self):
        scheduler = self._scheduler()
        self._tiny(scheduler, batches=1)
        status = fit.commit_prefit(scheduler, "ffffffff1", 1)
        self.assertEqual(status["reason"], "plan_mismatch")
        self.assertFalse(Path(self._path).is_file())
        plan = prefit.build_plan(
            prefit.PlanInputs(
                batches=(1,),
                ks=(2, 4, 8, 16, 32, 64),
                widths=(2, 4, 8, 16, 32, 64),
                buckets=(256, 512, 1024, 2048, 4096),
                max_depth=8,
            )
        )
        self.assertEqual(len(plan.required_h), 30)
        self.assertTrue(all(cell[0] == 1 for cell in plan.required_h))
        self.assertTrue(any(reason == "batch_limit" for _cell, reason in plan.skipped_h))

    def test_driver_retries_missing_and_abandons_timeouts(self):
        cfg = prefit.PrefitConfig(
            url="http://127.0.0.1:9",
            buckets=(256,),
            max_depth=8,
            ks=(8, 16),
            widths=(2, 8),
            batches=(1,),
            token_range=(1, 2),
            fit_timeout_s=0.01,
        )
        plan = prefit.plan_from_config(cfg)
        seen = []
        posts = []

        def fake_group(group_cfg, group, rng, run, n0):
            del group_cfg, rng, run
            seen.append((group["kind"], group.get("k"), n0))
            return {"steps": 8, "retries": 1, "step_target_met": True}

        def fake_control(group_cfg, run, kind, seq, rng, fingerprint=""):
            del group_cfg, rng, fingerprint
            posts.append((kind, run, seq))
            return True

        answers = [
            {
                "ready": False,
                "reason": "missing",
                "missing_h": [[1, 8, 256]],
                "missing_layers": [],
                "refill_groups": [["v", 1, 128, 8]],
            },
            {"ready": True, "reason": "ready"},
        ]

        def wait_status(run, seq, timeout):
            del run, seq, timeout
            return answers.pop(0)

        with patch.object(prefit, "_wait_until"), patch.object(
            prefit, "_run_group", side_effect=fake_group
        ), patch.object(prefit, "_post_control", side_effect=fake_control), self.assertLogs(
            prefit.logger, level="INFO"
        ) as logs:
            result = prefit.run_prefit(cfg, plan=plan, wait_status=wait_status)
        self.assertEqual(result, "ready")
        commits = [
            json.loads(entry.split("[CLEAR prefit_commit] ", 1)[1])
            for entry in logs.output
            if "[CLEAR prefit_commit]" in entry
        ]
        refills = [
            json.loads(entry.split("[CLEAR prefit_refill_group] ", 1)[1])
            for entry in logs.output
            if "[CLEAR prefit_refill_group]" in entry
        ]
        self.assertEqual([event["reason"] for event in commits], ["missing", "ready"])
        self.assertEqual(commits[0]["missing_h"], [[1, 8, 256]])
        self.assertEqual(commits[0]["refill_groups"], [["v", 1, 128, 8]])
        self.assertEqual(commits[1]["refill_groups"], [])
        self.assertEqual(
            refills,
            [
                {
                    "run": commits[0]["run"],
                    "commit_seq": 1,
                    "group_id": ["v", 1, 128, 8],
                    "attempts": 1,
                    "steps": 8,
                    "step_target_met": True,
                }
            ],
        )
        kinds = [item[0] for item in seen]
        self.assertIn("v", kinds)
        rerun = [item for item in seen if item[1] == 8]
        self.assertGreaterEqual(len(rerun), 2)
        self.assertGreater(rerun[-1][2], rerun[0][2])
        self.assertEqual([item[0] for item in posts], ["begin", "commit", "commit"])

        posts.clear()

        def wait_timeout(run, seq, timeout):
            del run, seq, timeout
            return None

        with patch.object(prefit, "_wait_until"), patch.object(
            prefit, "_run_group", side_effect=fake_group
        ), patch.object(prefit, "_post_control", side_effect=fake_control):
            failed = prefit.run_prefit(cfg, plan=plan, wait_status=wait_timeout)
        self.assertEqual(failed, "timeout")
        self.assertEqual([item[0] for item in posts], ["begin", "commit", "commit", "status"])
        self.assertNotIn("abandon", [item[0] for item in posts])

    def test_fit_status_and_idle_handshake(self):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_protocol import (
            parse_fit_status,
        )

        request = ClearBatchRequest(
            session_id="s",
            rpc_seq=4,
            action=ClearAction.FIT,
            target_info={"backend": "cpu", "tp": 1, "prefit_batches": [1, 2]},
        )
        decoded = ClearBatchRequest.from_dict(request.to_dict())
        self.assertEqual(decoded.action, ClearAction.FIT)
        self.assertEqual(decoded.target_info["prefit_batches"], [1, 2])
        reply = ClearBatchReply(
            session_id="s",
            rpc_seq=4,
            fit_status={
                "run": "abcd12341",
                "seq": 1,
                "ready": False,
                "reason": "missing",
                "state": "need_prefit",
                "missing_h": [[1, 8, 256]],
                "written": False,
            },
        )
        again = ClearBatchReply.from_dict(reply.to_dict())
        self.assertEqual(again.fit_status["state"], "need_prefit")
        self.assertEqual(again.fit_status["missing_h"], [[1, 8, 256]])
        self.assertIsNone(parse_fit_status("nope"))
        self.assertTrue(
            fit.target_headers_match(
                {"backend": "cpu", "tp": 1, "prefit_batches": [1]},
                {"backend": "cpu", "tp": 1, "prefit_batches": [1, 4]},
            )
        )

        scheduler = self._scheduler()
        scheduler.sr_speculation_stopped = False
        queries = []

        def query():
            queries.append(scheduler._clear_fit_state)
            if len(queries) < 3:
                return None
            return SimpleNamespace(fit_status={"state": "ready", "reason": "cached"})

        fit.fit_idle(scheduler, 0.0, query)
        self.assertEqual(scheduler._clear_fit_state, "retry")
        fit.fit_idle(scheduler, 4.0, query)
        self.assertEqual(len(queries), 1)
        fit.fit_idle(scheduler, 5.0, query)
        self.assertEqual(scheduler._clear_fit_state, "retry")
        fit.fit_idle(scheduler, 5.0 + 10.0, query)
        self.assertEqual(scheduler._clear_fit_state, "ready")

        quiet = self._scheduler()
        quiet.sr_speculation_stopped = False

        def time_out():
            return None

        fit.fit_idle(quiet, 0.0, time_out)
        fit.fit_idle(quiet, 5.0, time_out)
        fit.fit_idle(quiet, 15.0, time_out)
        self.assertEqual(quiet._clear_fit_state, "no_reply")
        fit.fit_idle(quiet, 20.0, time_out)
        self.assertEqual(quiet._clear_fit_state, "no_reply")
        fit.fit_idle(quiet, 15.0 + 60.0, time_out)
        self.assertEqual(quiet._clear_fit_state, "no_reply")

        scheduler._clear_fit_state = "ready"
        fit.note_prefill_fit(scheduler, SimpleNamespace(fit_ready=None, fit_status=None))
        self.assertEqual(scheduler._clear_fit_state, "legacy")
        scheduler._clear_fit_state = "ready"
        fit.note_prefill_fit(scheduler, SimpleNamespace(fit_ready=False, fit_status={"ready": False}))
        self.assertEqual(scheduler._clear_fit_state, "querying")

        scheduler.sr_speculation_stopped = True
        fit.fit_idle(scheduler, 100.0, query)
        self.assertEqual(scheduler._clear_fit_state, "breaker")
        scheduler.sr_speculation_stopped = False
        fit.fit_idle(
            scheduler,
            100.0,
            lambda: SimpleNamespace(fit_status={"state": "ledger_off"}),
        )
        self.assertEqual(scheduler._clear_fit_state, "disabled")

    def test_width_sixty_four_is_two_layers_and_skips_uncaptured(self):
        self.assertEqual(prefit.depth_for_width(16, 64, 8), 2)
        self.assertLess(prefit.depth_for_width(16, 64, 2), 2)
        scheduler = SimpleNamespace(
            server_args=_args(),
            _clear_fit_ready=False,
            _clear_prefit=None,
            _clear_layer_capture_bs=(),
        )
        scheduler.server_args.clear_tree_widths = [2, 4, 8, 16, 32, 64]
        run = "abcd12341"
        _begin_scan(scheduler, run)
        reqs = [SimpleNamespace(rid=f"clearfit-{run}-w-1-128-0")]
        info = None
        for _ in range(17):
            info = fit.prefit_plan(scheduler, reqs, 1, 0, (256,))
        self.assertEqual(info["width"], 64)
        self.assertEqual(info["depth"], 2)
        plan = fit.draft_plan(scheduler)
        required, skipped = fit.layer_requirements(scheduler, plan)
        self.assertEqual(required, [])
        self.assertTrue(skipped)
        self.assertTrue(all(reason == "no_graph" for _cell, reason in skipped))
        scheduler._clear_layer_capture_bs = (1,)
        scheduler._clear_prefit = None
        scheduler._clear_prefit_seen = []
        required, skipped = fit.layer_requirements(scheduler, plan)
        self.assertEqual(required, [])
        self.assertTrue(all(reason == "no_graph" for _cell, reason in skipped))
        scheduler._clear_layer_capture_pairs = tuple(
            (int(width), 1) for width in scheduler.server_args.clear_tree_widths
        )
        required, skipped = fit.layer_requirements(scheduler, plan)
        self.assertTrue(any(cell[0] == 64 and cell[2] == 1 for cell in required))
        self.assertFalse(any(reason == "no_graph" for _cell, reason in skipped))


class ClearTreeFitSyncTest(CustomTestCase):
    def test_control_round_trip_and_rid_has_no_control_meaning(self):
        request = ClearBatchRequest(
            session_id="sess",
            rpc_seq=3,
            action=ClearAction.CONTROL,
            control={
                "op": "commit",
                "run": "sess_abcd1234_1",
                "seq": 2,
                "fingerprint": "abcd1234",
            },
        )
        decoded = ClearBatchRequest.from_dict(request.to_dict())
        self.assertEqual(decoded.action, ClearAction.CONTROL)
        self.assertEqual(decoded.control["op"], "commit")
        self.assertIsNone(fit.parse_rid("clearfit-abcd12341-commit-1"))
        self.assertIsNone(fit.parse_rid("clearfit-abcd12341-abandon-1"))

    def test_rid_commit_does_not_change_ready(self):
        reset_policy()
        scheduler = ClearTreeFitFileTest()._scheduler()
        scheduler._clear_fit_ready = True
        scheduler._clear_fit_installed = True
        note_packet(
            scheduler,
            ClearBatchRequest(
                session_id="sess",
                rpc_seq=1,
                action=ClearAction.PREFILL,
                reqs=[ClearDraftRequest(rid="clearfit-abcd12341-commit-1")],
            ),
        )
        self.assertTrue(scheduler._clear_fit_ready)
        self.assertTrue(scheduler._clear_fit_installed)
        fit.abandon_prefit(scheduler, "missing-run")
        self.assertTrue(scheduler._clear_fit_ready)
        self.assertTrue(scheduler._clear_fit_installed)

    def test_control_commit_checks_session_and_refuses_cancel_after_write(self):
        case = ClearTreeFitFileTest()
        case.setUp()
        try:
            scheduler = case._scheduler()
            run = case._tiny(scheduler)
            case._fill(scheduler, run)
            fingerprint = fit.draft_plan(scheduler).fingerprint
            begun = fit.execute_control(
                scheduler,
                ClearBatchRequest(
                    session_id="sess",
                    rpc_seq=1,
                    action=ClearAction.CONTROL,
                    control={
                        "op": "begin",
                        "run": run,
                        "seq": 0,
                        "fingerprint": fingerprint,
                    },
                ),
            )
            self.assertEqual(begun["reason"], "begun")
            wrong = fit.execute_control(
                scheduler,
                ClearBatchRequest(
                    session_id="",
                    rpc_seq=2,
                    action=ClearAction.CONTROL,
                    control={
                        "op": "commit",
                        "run": run,
                        "seq": 1,
                        "fingerprint": fingerprint,
                    },
                ),
            )
            self.assertEqual(wrong["reason"], "bad_control")
            self.assertFalse(Path(case._path).is_file())
            other = fit.execute_control(
                scheduler,
                ClearBatchRequest(
                    session_id="sess",
                    rpc_seq=3,
                    action=ClearAction.CONTROL,
                    control={
                        "op": "commit",
                        "run": "other-run",
                        "seq": 1,
                        "fingerprint": fingerprint,
                    },
                ),
            )
            self.assertEqual(other["reason"], "stale_run")
            status = fit.execute_control(
                scheduler,
                ClearBatchRequest(
                    session_id="sess",
                    rpc_seq=4,
                    action=ClearAction.CONTROL,
                    control={
                        "op": "commit",
                        "run": run,
                        "seq": 1,
                        "fingerprint": fingerprint,
                    },
                ),
            )
            self.assertTrue(status["ready"])
            cancelled = fit.abandon_prefit(scheduler, run, session="sess")
            self.assertEqual(cancelled["reason"], "committed")
            self.assertTrue(scheduler._clear_fit_ready)
            self.assertTrue(scheduler._clear_fit_installed)
        finally:
            case.tearDown()

    def test_bucket_span_rpc_limit_and_graph_refill(self):
        plan = prefit.build_plan(
            prefit.PlanInputs(
                batches=(1,),
                ks=(8,),
                widths=(8,),
                buckets=(256, 320),
                max_depth=8,
            )
        )
        self.assertIn((1, 8, 320), plan.required_h)
        moved = next(g for g in plan.groups if g.get("bucket") == 320)
        self.assertEqual((moved["prefix"], moved["max_new"]), (248, 16))
        self.assertEqual(prefit.recorded_span(248, 8, 16), (257, 273))
        narrow = prefit.build_plan(prefit.PlanInputs((1,), (8,), (8,), (256, 272), 8))
        self.assertNotIn((1, 8, 272), narrow.required_h)
        self.assertIn(((1, 8, 272), "bucket_span"), narrow.skipped_h)
        limited = prefit.build_plan(
            prefit.PlanInputs(
                batches=(1, 2, 4),
                ks=(8,),
                widths=(8,),
                buckets=(256,),
                max_depth=8,
                rpc_batch_limit=2,
            )
        )
        self.assertTrue(limited.groups)
        self.assertTrue(all(int(group["b"]) <= 2 for group in limited.groups))
        self.assertTrue(all(int(cell[0]) <= 2 for cell in limited.required_h))
        for group in plan.groups:
            if int(group["b"]) == 1 and group["kind"] == "v":
                group["produces_layers"] = [(8, "graph", 4)]
                break
        refill = prefit.groups_for_missing(
            plan, [], [[8, "graph", 4]], b_to_g=[[1, 4]]
        )
        self.assertTrue(refill)
        self.assertTrue(all(int(group["b"]) == 1 for group in refill))
        self.assertEqual(prefit.groups_for_missing(plan, [], [[8, "graph", 1]]), [])

    def test_same_hardware_wider_plan_is_not_ready(self):
        case = ClearTreeFitFileTest()
        case.setUp()
        try:
            scheduler = case._scheduler()
            run = case._tiny(scheduler, batches=1)
            case._fill(scheduler, run)
            self.assertTrue(fit.commit_prefit(scheduler, run, 1)["ready"])
            scheduler.max_running_requests = 4
            info = fit.target_info(scheduler)
            policy.on_external_event(scheduler, SimpleNamespace(target_info=info))
            self.assertFalse(scheduler._clear_fit_installed)
            self.assertFalse(scheduler._clear_fit_ready)
        finally:
            case.tearDown()

    def test_corrupt_layer_file_does_not_replace_memory(self):
        case = ClearTreeFitFileTest()
        case.setUp()
        try:
            scheduler = case._scheduler()
            run = case._tiny(scheduler, ks=[16], capture=(1,))
            case._fill(scheduler, run)
            self.assertTrue(fit.commit_prefit(scheduler, run, 1)["ready"])
            payload = json.loads(Path(case._path).read_text())
            payload["layers"][0]["xtx"] = [[1.0]]
            Path(case._path).write_text(json.dumps(payload))
            policy.observe_verify(1, 16, 256, 4.5)
            kept = list(policy._clock[(1, 16, 256)])
            reset_policy()
            policy.observe_verify(1, 8, 256, 9.0)
            scheduler._clear_fit_installed = False
            scheduler._clear_fit_pending = None
            scheduler._clear_ledger.layer_models = {"keep": object()}
            fit.load_draft_part(scheduler)
            fit.on_target_info(scheduler, scheduler._clear_target_header)
            self.assertFalse(scheduler._clear_fit_ready)
            self.assertFalse(scheduler._clear_fit_installed)
            self.assertEqual(policy._clock[(1, 8, 256)], [9.0])
            self.assertNotIn((1, 16, 256), policy._clock)
            self.assertEqual(list(scheduler._clear_ledger.layer_models), ["keep"])
            self.assertNotEqual(kept, [])
        finally:
            case.tearDown()

    def test_timeout_queries_status_and_keeps_a_written_run(self):
        cfg = prefit.PrefitConfig(
            url="http://127.0.0.1:9",
            buckets=(256,),
            max_depth=8,
            ks=(8,),
            widths=(8,),
            batches=(1,),
            token_range=(1, 2),
            fit_timeout_s=0.01,
        )
        plan = prefit.plan_from_config(cfg)
        posts = []

        def fake_group(*args, **kwargs):
            del args, kwargs
            return {"kind": "v", "b": 1, "steps": 8, "retries": 1}

        def fake_control(group_cfg, run, kind, seq, rng, fingerprint=""):
            del group_cfg, rng, fingerprint
            posts.append(kind)
            if kind == "begin":
                return {"reason": "begun", "ready": False, "run": run, "seq": seq}
            if kind == "commit":
                return None
            if kind == "status":
                return {
                    "ready": True,
                    "written": True,
                    "reason": "ready",
                    "run": run,
                    "seq": seq,
                }
            return {"reason": kind, "ready": False, "run": run, "seq": seq}

        with patch.object(prefit, "_wait_until"), patch.object(
            prefit, "_run_group", side_effect=fake_group
        ), patch.object(prefit, "_post_control", side_effect=fake_control):
            result = prefit.run_prefit(
                cfg,
                plan=plan,
                wait_status=lambda run, seq, timeout: None,
                session_id="cli1",
            )
        self.assertEqual(result, "ready")
        self.assertEqual(posts, ["begin", "commit", "commit", "status"])
        self.assertNotIn("abandon", posts)

    def test_fit_then_next_rpc_keeps_every_rank_in_step(self):
        from collections import deque

        pipe = deque()

        def _broadcast(rank, value):
            if rank == 0:
                pipe.append(value)
                return value
            if not pipe:
                raise AssertionError("rank entered a collective the leader skipped")
            return pipe.popleft()

        def _rank(rank):
            return SimpleNamespace(
                tp_size=2,
                tp_rank=rank,
                sr_rpc_seq=0,
                sr_session_id="sess",
                sr_speculation_stopped=False,
                server_args=_args(),
                max_running_requests=1,
                _sr_broadcast_obj=lambda value, rank=rank: _broadcast(rank, value),
                _clear_fit_state="unknown",
                _clear_fit_reason=None,
                _clear_fit_run=None,
                _clear_fit_timeouts=0,
                _clear_fit_next_at=0.0,
            )

        leader = _rank(0)
        follower = _rank(1)
        # Same order as ClearTreeSchedulerMixin._clear_service_idle.
        control = None
        self.assertIsNone(_broadcast(0, control))
        self.assertIsNone(_broadcast(1, None))
        now = 0.0
        due = fit.fit_query_due(leader, now)
        self.assertTrue(_broadcast(0, due))
        self.assertTrue(_broadcast(1, None))
        fit_seq = fit.allocate_rpc_seq(leader)
        self.assertEqual(fit.allocate_rpc_seq(follower), fit_seq)
        reply = SimpleNamespace(
            fit_status={"state": "need_prefit", "ready": False, "reason": "need_prefit"},
            rpc_seq=fit_seq,
        )
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_fit.threading.Thread",
            _Thread,
        ):
            fit.apply_fit_query(leader, reply, now)
            fit.mirror_fit_reply(follower, reply)
        self.assertEqual(leader.sr_rpc_seq, follower.sr_rpc_seq)
        self.assertEqual(reply.rpc_seq, leader.sr_rpc_seq)
        self.assertIsNotNone(leader._clear_prefit_thread)
        self.assertTrue(leader._clear_prefit_thread.is_alive())
        self.assertFalse(hasattr(follower, "_clear_prefit_thread"))
        self.assertEqual(follower._clear_fit_state, "prefitting")
        nxt = fit.allocate_rpc_seq(leader)
        self.assertEqual(fit.allocate_rpc_seq(follower), nxt)
        self.assertEqual(nxt, fit_seq + 1)
        self.assertEqual(leader.sr_rpc_seq, follower.sr_rpc_seq)

    def test_handshake_plan_begins_from_the_target_constructor(self):
        saved = {
            name: os.environ.get(name)
            for name in (
                "SGLANG_CLEAR_LEDGER",
                "SGLANG_CLEAR_NPU_KV_BUCKETS",
                "SGLANG_CLEAR_FIT_PATH",
            )
        }
        directory = tempfile.TemporaryDirectory()
        try:
            os.environ["SGLANG_CLEAR_LEDGER"] = "1"
            os.environ["SGLANG_CLEAR_NPU_KV_BUCKETS"] = "256,512,1024,2048,4096"
            os.environ["SGLANG_CLEAR_FIT_PATH"] = str(Path(directory.name) / "none.json")
            target = ClearTreeFitFileTest()._scheduler()
            draft = ClearTreeFitFileTest()._scheduler()
            target.server_args.standalone_remote_max_batch_size = 32
            draft.server_args.standalone_remote_max_batch_size = 32
            fit.on_target_info(draft, fit.target_info(target))
            status = fit.fit_handshake_status(draft)
            self.assertEqual(status["state"], "need_prefit")
            self.assertEqual(status["prefit_batches"], [1, 2, 4])
            with patch(
                "sglang.srt.speculative.clear_tree_remote.clear_tree_fit.threading.Thread",
                _Thread,
            ):
                fit.apply_fit_query(
                    target, SimpleNamespace(fit_status=dict(status)), 0.0
                )
            plan = prefit.plan_from_config(fit.prefit_config(target))
            run = prefit.make_run_id("sess", plan.fingerprint, "1")
            begun = fit.begin_prefit(draft, run, plan.fingerprint, "sess", 0)
            self.assertEqual(begun["reason"], "begun")
            self.assertEqual(len(plan.required_h), 90)
            self.assertEqual(fit.draft_plan(draft).fingerprint, plan.fingerprint)
            self.assertEqual(len(fit.draft_plan(draft).required_h), 90)

            draft.max_running_requests = 1
            draft.server_args.max_running_requests = 1
            draft._clear_prefit = None
            draft._clear_prefit_seen = []
            fit.on_target_info(draft, fit.target_info(target))
            narrow = fit.fit_handshake_status(draft)
            self.assertEqual(narrow["prefit_batches"], [1])
            fit.note_joint_batches(target, narrow)
            narrow_plan = prefit.plan_from_config(fit.prefit_config(target))
            self.assertTrue(narrow_plan.required_h)
            self.assertTrue(all(cell[0] == 1 for cell in narrow_plan.required_h))
            narrow_run = prefit.make_run_id("sess", narrow_plan.fingerprint, "2")
            narrow_begun = fit.begin_prefit(
                draft, narrow_run, narrow_plan.fingerprint, "sess", 0
            )
            self.assertEqual(narrow_begun["reason"], "begun")
            self.assertEqual(fit.draft_plan(draft).fingerprint, narrow_plan.fingerprint)
        finally:
            directory.cleanup()
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_samples_stay_on_the_run_that_opened_them(self):
        scheduler = ClearTreeFitFileTest()._scheduler()
        scheduler._clear_target_header = fit.target_info(scheduler)
        untouched = fit.prefit_plan(
            scheduler,
            [SimpleNamespace(rid="clearfit-norun-v-1-128-8-0")],
            1,
            0,
            (256,),
        )
        self.assertEqual(untouched["phase"], "stale")
        self.assertIsNone(getattr(scheduler, "_clear_prefit", None))
        fingerprint = fit.draft_plan(scheduler).fingerprint
        first = prefit.make_run_id("sess", fingerprint, "1")
        scheduler.sr_session_id = "sess"
        self.assertEqual(
            fit.begin_prefit(scheduler, first, fingerprint, "sess", 0)["reason"],
            "begun",
        )
        key = next(iter(scheduler._clear_prefit.groups))
        owner = {"run": first, "session": "sess", "group_key": list(key)}
        fit.prefit_record_h(scheduler, 1, 8, 256, 3.25, owner)
        self.assertEqual(scheduler._clear_prefit.h[(1, 8, 256)], [3.25])
        second = prefit.make_run_id("sess", fingerprint, "2")
        self.assertEqual(
            fit.begin_prefit(scheduler, second, fingerprint, "sess", 0)["reason"],
            "begun",
        )
        fit.prefit_record_h(scheduler, 1, 8, 256, 9.5, owner)
        self.assertEqual(scheduler._clear_prefit.run, second)
        self.assertEqual(scheduler._clear_prefit.h, {})

    def test_control_shape_and_abandon_keep_the_request_seq(self):
        self.assertTrue(fit.shape_check_covers(ClearAction.CONTROL))
        self.assertFalse(fit.shape_check_covers(ClearAction.FIT))
        self.assertFalse(fit.shape_check_covers(ClearAction.FINISH))
        batch = ClearBatchRequest(
            session_id="sess",
            rpc_seq=3,
            action=ClearAction.CONTROL,
            spec_shape=(1, 2, 3),
            control={
                "op": "begin",
                "run": "sess_abcd1234_1",
                "seq": 7,
                "fingerprint": "abcd1234",
            },
        )
        status = fit.control_shape_status(batch)
        self.assertEqual(status["seq"], 7)
        self.assertEqual(status["reason"], "spec_shape_mismatch")
        self.assertFalse(status["ready"])

        case = ClearTreeFitFileTest()
        case.setUp()
        try:
            scheduler = case._scheduler()
            run = case._tiny(scheduler)
            fingerprint = fit.draft_plan(scheduler).fingerprint
            begun = fit.begin_prefit(scheduler, run, fingerprint, "sess", 0)
            self.assertEqual(begun["reason"], "begun")
            cancelled = fit.execute_control(
                scheduler,
                ClearBatchRequest(
                    session_id="sess",
                    rpc_seq=8,
                    action=ClearAction.CONTROL,
                    control={
                        "op": "abandon",
                        "run": run,
                        "seq": 4,
                        "fingerprint": fingerprint,
                    },
                ),
            )
            self.assertEqual(cancelled["reason"], "abandon")
            self.assertEqual(cancelled["seq"], 4)
            fit.publish_fit_status(scheduler, cancelled)
            found = fit.wait_fit_status(scheduler, run, 4, 0.2)
            self.assertEqual(found["seq"], 4)
            self.assertEqual(found["reason"], "abandon")
            self.assertIsNone(fit.wait_fit_status(scheduler, run, 0, 0.05))
        finally:
            case.tearDown()

    def test_unproducible_layer_cell_is_skipped(self):
        saved = os.environ.get("SGLANG_CLEAR_NPU_KV_BUCKETS")
        os.environ["SGLANG_CLEAR_NPU_KV_BUCKETS"] = "64"
        try:
            plan = prefit.build_plan(
                prefit.PlanInputs(
                    batches=(1,),
                    ks=(8,),
                    widths=(4, 8),
                    buckets=(64,),
                    max_depth=8,
                )
            )
            scheduler = SimpleNamespace(
                server_args=_args(),
                _clear_layer_capture_bs=(1,),
                _clear_layer_capture_pairs=((4, 1), (8, 1)),
            )
            required, skipped = fit.layer_requirements(scheduler, plan)
            required = [tuple(cell) for cell in required]
            skipped = [(tuple(cell), reason) for cell, reason in skipped]
            self.assertNotIn((8, "graph", 1), required)
            self.assertIn(((8, "graph", 1), "no_group"), skipped)
            self.assertIn((4, "graph", 1), required)
            self.assertEqual(
                prefit.groups_for_missing(plan, [], [[8, "graph", 1]]), []
            )
        finally:
            if saved is None:
                os.environ.pop("SGLANG_CLEAR_NPU_KV_BUCKETS", None)
            else:
                os.environ["SGLANG_CLEAR_NPU_KV_BUCKETS"] = saved

    def test_prefit_keep_replays_only_ordinary_samples(self):
        case = ClearTreeFitFileTest()
        case.setUp()
        try:
            scheduler = case._scheduler()
            run = case._tiny(scheduler)
            fingerprint = fit.draft_plan(scheduler).fingerprint
            begun = fit.begin_prefit(scheduler, run, fingerprint, "sess", 0)
            self.assertEqual(begun["reason"], "begun")
            ledger = scheduler._clear_ledger
            ledger.active_rids = {"user"}
            layer = {
                "layer": 0,
                "active": 1,
                "graph_bucket": 1,
                "w": 8,
                "sum_ell": 32,
                "host_us": 1000,
                "gpu_us": 4000,
                "path": "graph",
                "post_reply": False,
            }
            ledger.on_packet("step", ["user"], "sess", 1, [32], 8, now=1.0)
            ledger.mark_stop(1.0)
            ledger.mark_send(1.1)
            ledger.queue_layer(dict(layer))
            ledger.on_packet(
                "step",
                ["user"],
                "sess",
                2,
                [32],
                8,
                now=1.4,
                scheduler=scheduler,
            )
            self.assertEqual(policy._clock[(1, 8, 256)], [400.0])
            self.assertEqual(policy._clock_ell[(1, 8, 256)], [32])
            self.assertEqual(scheduler._clear_prefit.h, {})
            self.assertEqual(scheduler._clear_prefit.online_h, [(1, 8, 256, 400.0, 32)])
            self.assertEqual(ledger.layer_models[(8, "graph", 1)].n, 1)
            self.assertEqual(len(scheduler._clear_prefit.online_layers), 1)
            self.assertEqual(policy._q_raw, {})

            key = next(iter(scheduler._clear_prefit.groups))
            ledger.mark_stop(1.4)
            ledger.mark_send(1.5)
            ledger.open.prefit = {
                "phase": "keep",
                "group": "v",
                "bucket": 256,
                "run": run,
                "session": "sess",
                "group_key": list(key),
            }
            kept_layer = dict(layer)
            kept_layer["host_us"] = 2000
            kept_layer["gpu_us"] = 2000
            ledger.queue_layer(kept_layer)
            kept = ledger.on_packet(
                "step",
                ["user"],
                "sess",
                3,
                [32],
                8,
                now=1.8,
                hold_ns=100_000_000,
                hold_rpc_seq=2,
                verify={
                    "rpc_seq": 2,
                    "fallback": False,
                    "b": 1,
                    "k": 8,
                    "bucket": 256,
                },
                scheduler=scheduler,
            )
            self.assertNotIn("prefit_bucket", kept["exclude_reason"])
            self.assertEqual(policy._clock[(1, 8, 256)], [400.0])
            self.assertEqual(policy._q_raw, {})
            self.assertEqual(scheduler._clear_prefit.h[(1, 8, 256)], [100.0])
            self.assertEqual(scheduler._clear_prefit.h_ell[(1, 8, 256)], [32])
            self.assertEqual(ledger.layer_models[(8, "graph", 1)].n, 1)
            self.assertEqual(scheduler._clear_prefit.layers[(8, "graph", 1)].n, 1)

            ledger.mark_stop(1.8)
            ledger.mark_send(1.9)
            ordinary = ledger.on_packet(
                "step",
                ["user"],
                "sess",
                4,
                [32],
                8,
                now=2.2,
                hold_ns=50_000_000,
                hold_rpc_seq=3,
                verify={
                    "rpc_seq": 3,
                    "fallback": False,
                    "b": 1,
                    "k": 8,
                    "bucket": 256,
                },
                scheduler=scheduler,
            )
            self.assertFalse(ordinary["excluded"])
            self.assertEqual(policy._clock[(1, 8, 256)], [400.0, 50.0])
            self.assertEqual(policy._clock_ell[(1, 8, 256)], [32, 32])
            self.assertEqual(list(policy._q_raw[(1, 8)]), [350.0])
            case._fill(scheduler, run)
            status = fit.commit_prefit(scheduler, run, 1)
            self.assertTrue(status["ready"], status)
            self.assertEqual(
                policy._clock[(1, 8, 256)],
                [100.0, 12.5, 12.5, 12.5, 12.5, 400.0, 50.0],
            )
            self.assertEqual(
                policy._clock_ell[(1, 8, 256)],
                [32, None, None, None, None, 32, 32],
            )
            self.assertEqual(ledger.layer_models[(8, "graph", 1)].n, 2)
            self.assertEqual(list(policy._q_raw[(1, 8)]), [350.0])
            self.assertEqual(scheduler._clear_prefit.online_h, [])
            payload = json.loads(Path(case._path).read_text())
            saved = []
            for row in payload["h"]:
                saved.extend(float(value) for value in row["samples_ms"])
            self.assertNotIn(400.0, saved)
            self.assertNotIn(50.0, saved)
            self.assertIn(100.0, saved)
        finally:
            case.tearDown()

    def test_abandon_drops_the_run_and_keeps_online_policy(self):
        case = ClearTreeFitFileTest()
        case.setUp()
        try:
            scheduler = case._scheduler()
            run = case._tiny(scheduler)
            fingerprint = fit.draft_plan(scheduler).fingerprint
            self.assertEqual(
                fit.begin_prefit(scheduler, run, fingerprint, "sess", 0)["reason"],
                "begun",
            )
            scheduler._clear_fit_ready = True
            scheduler._clear_fit_installed = True
            policy._rounds = 5
            policy._probe_rounds = 2
            policy.observe_verify(1, 8, 256, 3.0)
            policy.observe_q(1, 8, 4.0)
            model = _new_layer_model()
            for length, target in ((4.0, 1.0), (8.0, 1.2), (12.0, 1.4), (16.0, 1.6)):
                model.update([1.0, length], target)
            scheduler._clear_ledger.layer_models[(8, "graph", 1)] = model
            status = fit.abandon_prefit(scheduler, run, session="sess", seq=4)
            self.assertEqual(status["reason"], "abandon")
            self.assertIsNone(scheduler._clear_prefit)
            self.assertTrue(scheduler._clear_fit_ready)
            self.assertTrue(scheduler._clear_fit_installed)
            self.assertEqual(policy._rounds, 5)
            self.assertEqual(policy._probe_rounds, 2)
            self.assertEqual(policy._clock[(1, 8, 256)], [3.0])
            self.assertEqual(list(policy._q_raw[(1, 8)]), [4.0])
            self.assertEqual(scheduler._clear_ledger.layer_models[(8, "graph", 1)].n, 4)
        finally:
            case.tearDown()


class ClearTreeVerifyFreezeTest(CustomTestCase):
    def test_report_keeps_the_pre_accept_bucket_and_rpc(self):
        frozen = {
            "session": "sess",
            "rpc_seq": 7,
            "b": 1,
            "k": 8,
            "seq_lens": [247],
        }
        report = complete_verify_report(frozen, True)
        self.assertEqual(report["rpc_seq"], 7)
        self.assertEqual(report["bucket"], 256)
        self.assertEqual(report["path"], "graph")
        self.assertFalse(report["fallback"])
        sent = ClearBatchRequest(session_id="sess", rpc_seq=9, action=ClearAction.STEP)
        self.assertTrue(attach_verify_report(sent, report, "sess"))
        self.assertEqual(sent.target_verify_rpc_seq, 7)
        self.assertEqual(sent.target_verify_bucket, 256)
        grown = complete_verify_report(
            {"session": "sess", "rpc_seq": 8, "b": 1, "k": 8, "seq_lens": [248]},
            True,
        )
        self.assertEqual(grown["bucket"], 512)
        eager = complete_verify_report(frozen, False)
        self.assertEqual(eager["path"], "eager")
        self.assertFalse(eager["fallback"])
        self.assertEqual(eager["bucket"], 256)
        unused = complete_verify_report({**frozen, "k": 0}, False)
        self.assertEqual(unused["path"], "fallback")
        self.assertTrue(unused["fallback"])
        bare = ClearBatchRequest(session_id="sess", rpc_seq=10, action=ClearAction.STEP)
        self.assertFalse(attach_verify_report(bare, {"b": 1, "k": 8}, "sess"))
        self.assertIsNone(bare.target_verify_rpc_seq)

    def test_refill_ids_select_the_negotiated_group(self):
        saved = os.environ.get("SGLANG_CLEAR_NPU_KV_BUCKETS")
        os.environ["SGLANG_CLEAR_NPU_KV_BUCKETS"] = "256"
        case = ClearTreeFitFileTest()
        case.setUp()
        try:
            scheduler = case._scheduler()
            run = case._tiny(scheduler, ks=[16], capture=(1,))
            fingerprint = fit.draft_plan(scheduler).fingerprint
            self.assertEqual(
                fit.begin_prefit(scheduler, run, fingerprint, "sess", 0)["reason"],
                "begun",
            )
            case._fill(scheduler, run)
            state = scheduler._clear_prefit
            layer = state.required_layers[0]
            state.layers[layer].n = 0
            status = fit.commit_prefit(scheduler, run, 1)
            self.assertEqual(status["reason"], "missing")
            self.assertTrue(status["refill_groups"])
            bare = prefit.build_plan(state.plan.inputs)
            self.assertEqual(
                prefit.groups_for_missing(bare, [], status["missing_layers"]), []
            )
            chosen = prefit.groups_for_refill(bare, status)
            self.assertTrue(chosen)
            self.assertTrue(
                all(prefit._group_id(group)[0] == "v" for group in chosen)
            )
            self.assertEqual(prefit.groups_for_refill(bare, {}), [])
        finally:
            case.tearDown()
            if saved is None:
                os.environ.pop("SGLANG_CLEAR_NPU_KV_BUCKETS", None)
            else:
                os.environ["SGLANG_CLEAR_NPU_KV_BUCKETS"] = saved

    def test_graph_off_handshake_does_not_scan(self):
        saved = os.environ.get("SGLANG_CLEAR_LEDGER")
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        try:
            draft = ClearTreeFitFileTest()._scheduler()
            draft.sr_session_id = "sess"
            header = dict(fit.target_info(draft))
            header["disable_cuda_graph"] = True
            fit.on_target_info(draft, header)
            status = fit.fit_handshake_status(draft)
            self.assertEqual(status["state"], "graph_unsupported")
            fit.apply_fit_query(draft, SimpleNamespace(fit_status=dict(status)), 0.0)
            self.assertEqual(draft._clear_fit_state, "disabled")
            self.assertFalse(hasattr(draft, "_clear_prefit_thread"))
            draft._clear_target_header = None
            self.assertEqual(fit.fit_handshake_status(draft)["state"], "need_prefit")
        finally:
            if saved is None:
                os.environ.pop("SGLANG_CLEAR_LEDGER", None)
            else:
                os.environ["SGLANG_CLEAR_LEDGER"] = saved

    def test_commit_merges_a_late_graph_report(self):
        case = ClearTreeFitFileTest()
        case.setUp()
        try:
            scheduler = case._scheduler()
            run = case._tiny(scheduler)
            fingerprint = fit.draft_plan(scheduler).fingerprint
            self.assertEqual(
                fit.begin_prefit(scheduler, run, fingerprint, "sess", 0)["reason"],
                "begun",
            )
            ledger = scheduler._clear_ledger
            key = ("sess", 4)
            ledger._pending_rounds[key] = {
                "hold_ms": 11.0,
                "q_ms": 3.0,
                "prefit": None,
            }
            ledger._pending_reports[key] = {
                "rpc_seq": 4,
                "fallback": False,
                "path": "graph",
                "b": 1,
                "k": 8,
                "bucket": 256,
            }
            ledger._pending_order = [key]
            status = fit.commit_prefit(scheduler, run, 1)
            self.assertEqual(status["reason"], "missing")
            self.assertEqual(policy._clock[(1, 8, 256)], [11.0])
            self.assertEqual(list(policy._q_raw[(1, 8)]), [3.0])
            self.assertEqual(ledger._pending_rounds, {})
        finally:
            case.tearDown()


if __name__ == "__main__":
    unittest.main()
