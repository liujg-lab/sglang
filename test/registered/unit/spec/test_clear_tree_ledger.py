"""CPU checks for the CLEAR_TREE_REMOTE shadow ledger.

The ledger must not change width or tau. These tests stay on the pure
accounting object.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.clear_tree_remote.clear_tree_layout import (
    align_draft_row,
    build_tree_from_parents,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_config import (
    ClearTreeConfig,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_forward import (
    _cut_deferred,
    grow_active_layers,
)
from sglang.srt.speculative.clear_tree_remote import clear_tree_policy as policy
from sglang.srt.speculative.clear_tree_remote import clear_tree_ledger as ledger_module
from sglang.srt.speculative.clear_tree_remote import clear_tree_replay as replay
from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
    RoundPlan,
    accept_ratio,
    attach_round,
    choose_build_width,
    flat_width,
    _marginal_gain,
    _raw_marginal,
    gain_ratio,
    ingest_verify_samples,
    install_fallback_plan,
    compare_price,
    keep_by_net_benefit,
    layer_should_continue,
    make_round,
    note_verified,
    on_external_event,
    observe_accept,
    observe_price,
    observe_verify,
    persistent_width,
    predict_verify_ms,
    request_terms,
    reset_gain_calibration,
    reset_policy,
    select_continuing,
    settle_verified,
    theta,
    truncate_columns,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_replay import (
    _freeze_layer_rows,
    _schedule_ms,
    _select_replay_k,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_ops import (
    ClearTreeState,
    acceptance_mass,
    depth_rule,
    step_after_layer,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
    assign_persistent_width,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_args import (
    apply_clear_tree_settings,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_protocol import (
    ClearAction,
    ClearBatchRequest,
    ClearDraftRequest,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_transport import (
    _pack_request,
    _unpack_request,
)
from sglang.srt.speculative.standalone_remote.sr_protocol import (
    SRAction,
    SRBatchRequest,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
    LAYER_MIN_SAMPLES,
    LAYER_RIDGE,
    VERIFY_MIN_SAMPLES,
    ClearLedger,
    predict_layer_ms,
    note_packet,
    push_verify_sample,
    restore_verify_samples,
    take_verify_samples,
    FORGET,
    ForgettingRegressor,
    consume_verify_device,
    emit_verify_device,
    graph_bucket,
    layer_graph_bs,
    layer_sum_ell_slope,
    note_target_prefill,
    read_verify_host_us,
    reset_target_verify_model,
    runtime_kv_buckets,
    smallest_bucket,
    target_verify_record,
    target_verify_samples,
)
from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
    SRRoundMetrics,
)
from sglang.srt.speculative.tree_attn_fallback import select_tree_kv_bucket
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

_LEDGER_ENV = (
    "SGLANG_CLEAR_LEDGER",
    "SGLANG_CLEAR_TARGET_BACKEND",
    "SGLANG_CLEAR_NPU_KV_BUCKETS",
)


def _step(ledger, rids, prefix, rpc, now, width=8, recv_queued=False):
    return ledger.on_packet(
        "step",
        rids,
        "sess",
        rpc,
        prefix,
        width,
        now=now,
        recv_queued=recv_queued,
    )


def _pin_verify_ks(test, ks=(8, 16, 32)):
    """Keep one probe scenario on the previous verify tiers, then restore."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        install_candidates,
    )

    install_candidates(None, ks)
    test.addCleanup(lambda: install_candidates(None, None))


class ClearTreeLedgerTest(CustomTestCase):
    def setUp(self):
        self._saved = {name: os.environ.get(name) for name in _LEDGER_ENV}
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        reset_target_verify_model()
        reset_policy()
        reset_gain_calibration()

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_time_ledger_and_floor(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [32], 1, 10.0)
        ledger.mark_stop(10.0)
        ledger.mark_send(10.25)
        ledger.mark_post(10.30)
        record = _step(ledger, ["a"], [32], 2, 10.40, recv_queued=True)

        self.assertEqual(record["t_k_us"], 400000)
        self.assertTrue(record["draft_floor"])
        self.assertFalse(record["excluded"])
        self.assertEqual(ledger.verify_samples, 0)

    def test_floor_follows_receive_queue_not_idle_or_post_time(self):
        for recv_queued in (False, True):
            for post_time in (None, 1.2):
                with self.subTest(recv_queued=recv_queued, post_time=post_time):
                    ledger = ClearLedger(active_rids=["a"])
                    _step(ledger, ["a"], [8], 1, 1.0)
                    ledger.mark_stop(1.0)
                    ledger.mark_send(1.1)
                    if post_time is not None:
                        ledger.mark_post(post_time)
                    ledger.note_idle()
                    record = _step(
                        ledger,
                        ["a"],
                        [8],
                        2,
                        1.5,
                        recv_queued=recv_queued,
                    )
                    self.assertEqual(record["draft_floor"], recv_queued)
                    self.assertFalse(record["excluded"])
                    self.assertEqual(ledger.verify_samples, int(not recv_queued))
                    self.assertEqual(record["post_ms"], 0.0)
                    self.assertEqual(record["policy_ms"], 0.0)
                    self.assertAlmostEqual(
                        record["wait_ms"], 0.0 if post_time is None else 300.0
                    )

    def test_round_logs_are_deferred_ordered_and_flushed_once(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [8], 1, 1.0)
        ledger.queue_layer(
            {
                "layer": 0,
                "active": 1,
                "graph_bucket": 1,
                "w": 8,
                "sum_ell": 8,
                "host_us": 1000,
                "gpu_us": 1000,
                "path": "graph",
                "post_reply": False,
            }
        )
        ledger.mark_stop(1.1)
        ledger.mark_send(1.2)
        with patch.object(ledger_module, "_emit") as emit:
            closed = _step(ledger, ["a"], [9], 2, 1.5)
            emit.assert_not_called()
            self.assertEqual(
                [kind for kind, _record in ledger.pending_logs],
                ["d_round", "d_layer"],
            )
            replay_record = {"rounds": 1}
            ledger.queue_log("d_replay", replay_record)
            ledger.flush_logs()
            self.assertEqual(
                [call.args[0] for call in emit.call_args_list],
                ["d_round", "d_layer", "d_replay"],
            )
            self.assertIs(emit.call_args_list[0].args[1], closed)
            self.assertIs(emit.call_args_list[-1].args[1], replay_record)
            self.assertFalse(ledger.pending_logs)
            ledger.flush_logs()
            self.assertEqual(emit.call_count, 3)
        self.assertEqual(ledger.layer_models[(8, "graph", 1)].n, 1)

    def test_idle_flushes_logs_without_dropping_open_round(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [8], 1, 1.0)
        opened = ledger.open
        record = {"rpc_seq": 0}
        ledger.queue_log("d_round", record)
        with patch.object(ledger_module, "_emit") as emit:
            ledger.note_idle()
            ledger.note_idle()
        emit.assert_called_once_with("d_round", record)
        self.assertFalse(ledger.pending_logs)
        self.assertIs(ledger.open, opened)
        self.assertEqual(ledger.active_rids, {"a"})

    def test_flush_continues_after_a_diagnostic_failure(self):
        ledger = ClearLedger()
        ledger.queue_log("d_round", {"rpc_seq": 1})
        ledger.queue_log("d_replay", {"rounds": 1})
        with patch.object(
            ledger_module, "_emit", side_effect=[RuntimeError("log failed"), None]
        ) as emit:
            ledger.flush_logs()
            ledger.flush_logs()
        self.assertEqual(emit.call_count, 2)
        self.assertFalse(ledger.pending_logs)

    def test_nested_policy_timing_and_post_accounting(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [8], 1, 1.0)
        scheduler = SimpleNamespace(_clear_ledger=ledger)
        owner = SimpleNamespace(_scheduler=scheduler)

        @ledger_module.policy_timed
        def inner(_owner):
            return 7

        @ledger_module.policy_timed
        def outer(_scheduler):
            return inner(owner)

        with patch.object(
            ledger_module.time, "perf_counter", side_effect=[2.0, 2.005, 3.0, 3.002]
        ) as clock:
            self.assertEqual(outer(scheduler), 7)
            self.assertEqual(inner(owner), 7)
        self.assertEqual(clock.call_count, 4)
        self.assertEqual(scheduler._clear_policy_depth, 0)
        self.assertAlmostEqual(ledger.open.policy_ms, 7.0)
        ledger_module.add_post_ms(scheduler, 2.0)
        ledger_module.add_post_ms(scheduler, 3.0)
        ledger_module.add_post_ms(scheduler, -100.0)
        ledger.mark_stop(1.1)
        ledger.mark_send(1.2)
        ledger.mark_post(1.3)
        record = _step(ledger, ["a"], [9], 2, 1.5)
        self.assertAlmostEqual(record["policy_ms"], 7.0)
        self.assertAlmostEqual(record["post_ms"], 5.0)
        self.assertAlmostEqual(record["wait_ms"], 200.0)
        self.assertEqual(ledger.open.policy_ms, 0.0)
        self.assertEqual(ledger.open.post_ms, 0.0)

    def test_disabled_policy_timing_keeps_controller_behavior_without_clock(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [8], 1, 1.0)
        scheduler = SimpleNamespace(_clear_ledger=ledger)

        @ledger_module.policy_timed
        def controller(_scheduler):
            return "kept"

        with patch.dict(os.environ, {"SGLANG_CLEAR_LEDGER": "0"}), patch.object(
            ledger_module.time, "perf_counter"
        ) as clock:
            self.assertEqual(controller(scheduler), "kept")
        clock.assert_not_called()
        self.assertEqual(ledger.open.policy_ms, 0.0)

    def test_policy_timing_restores_depth_after_failure(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [8], 1, 1.0)
        scheduler = SimpleNamespace(_clear_ledger=ledger)

        @ledger_module.policy_timed
        def fail(_scheduler):
            raise ValueError("controller failed")

        with patch.object(ledger_module.time, "perf_counter", side_effect=[2.0, 2.004]):
            with self.assertRaisesRegex(ValueError, "controller failed"):
                fail(scheduler)
        self.assertEqual(scheduler._clear_policy_depth, 0)
        self.assertAlmostEqual(ledger.open.policy_ms, 4.0)

    def test_price_uses_open_to_recv_minus_prefill(self):
        reset_policy()
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [10], 1, 10.0)
        ledger.add_draft_prefill(0.02)
        ledger.mark_stop(10.10)
        ledger.mark_send(10.12)
        ledger.mark_post(10.13)
        closed = _step(ledger, ["a"], [14], 2, 10.40)
        self.assertTrue(closed["price_kept"])
        self.assertEqual(closed["price_tokens"], 4)
        self.assertAlmostEqual(closed["price_ms"], 380.0)
        self.assertAlmostEqual(theta(1), 4.0 / 380.0)

        ledger.mark_stop(10.50)
        ledger.mark_send(10.55)
        ledger.mark_post(10.56)
        ledger.note_idle()
        later = _step(ledger, ["a"], [40], 3, 11.40)
        self.assertTrue(later["price_kept"])
        self.assertFalse(later["excluded"])
        self.assertAlmostEqual(theta(1), 15.0 / 690.0)

    def test_wait_is_not_an_empty_server(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [8], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.0)
        ledger.mark_post(1.01)
        ledger.note_idle()
        record = _step(ledger, ["a"], [8], 2, 1.05)

        self.assertFalse(record["draft_floor"])
        self.assertFalse(record["excluded"])
        self.assertEqual(ledger.verify_samples, 1)
        self.assertEqual(ledger.active_rids, {"a"})

    def test_three_exclusions(self):
        fresh = ClearLedger()
        _step(fresh, ["a"], [4], 1, 1.0)
        fresh.mark_stop(1.0)
        fresh.mark_send(1.1)
        fresh.mark_post(1.2)
        fresh.note_idle()
        before = theta(1)
        idle = _step(fresh, ["a"], [4], 2, 1.4)
        self.assertIn("idle_start", idle["exclude_reason"])
        self.assertEqual(fresh.verify_samples, 0)
        self.assertFalse(idle["price_kept"])
        self.assertEqual(idle["price_tokens"], 0)
        self.assertAlmostEqual(idle["price_ms"], 400.0)
        self.assertEqual(idle["gap_rpc"], [])
        self.assertEqual(theta(1), before)

        prefill = ClearLedger(active_rids=["a"])
        _step(prefill, ["a"], [4], 1, 1.0)
        prefill.mark_stop(1.0)
        prefill.mark_send(1.1)
        prefill.mark_post(1.2)
        self.assertIsNone(
            prefill.on_packet("prefill", ["b"], "sess", 2, [9], 8, now=1.3)
        )
        prefill.note_idle()
        held = _step(prefill, ["a"], [4], 3, 1.5)
        self.assertIn("prefill", held["exclude_reason"])
        self.assertEqual(prefill.verify_samples, 0)
        self.assertFalse(held["price_kept"])
        self.assertEqual(held["price_tokens"], 0)
        self.assertAlmostEqual(held["price_ms"], 500.0)
        self.assertEqual(held["gap_rpc"], [2])
        self.assertEqual(theta(1), before)

        grown = ClearLedger(active_rids=["a"])
        _step(grown, ["a"], [4], 1, 1.0)
        grown.mark_stop(1.0)
        grown.mark_send(1.1)
        grown.mark_post(1.2)
        grown.note_idle()
        added = _step(grown, ["a", "b"], [4, 8], 2, 1.5)
        self.assertIn("new_rid", added["exclude_reason"])
        self.assertEqual(grown.verify_samples, 0)
        self.assertFalse(added["price_kept"])
        self.assertEqual(added["price_tokens"], 0)
        self.assertAlmostEqual(added["price_ms"], 500.0)
        self.assertEqual(added["gap_rpc"], [])
        self.assertEqual(theta(1), before)

    def test_target_prefill_log_stays_off_the_step(self):
        record = note_target_prefill("sess", 7, ["a"], 10000)
        self.assertEqual(
            record,
            {
                "action": "prefill",
                "session_id": "sess",
                "rpc_seq": 7,
                "rids": ["a"],
                "wall_us": 10000,
            },
        )
        self.assertIsNone(note_target_prefill("sess", 8, ["a"], 10000, 1))

    def test_bonus_hit_skips_layer_regression(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [16], 1, 1.0)
        ledger.note_bonus_hit("a")
        ledger.queue_layer(
            {
                "layer": 0,
                "active": 1,
                "graph_bucket": 1,
                "w": 8,
                "sum_ell": 16,
                "host_us": 1000,
                "gpu_us": 4000,
                "path": "graph",
                "post_reply": False,
            }
        )
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        ledger.mark_post(1.2)
        ledger.note_idle()
        record = _step(ledger, ["a"], [16], 2, 1.4)

        self.assertEqual(record["bonus_hit_rids"], ["a"])
        self.assertEqual(ledger.layer_models[(8, "graph", 1)].n, 1)

    def test_layer_line_and_graph_bucket(self):
        self.assertEqual(graph_bucket(1), 1)
        self.assertEqual(graph_bucket(3), 4)
        self.assertEqual(graph_bucket(4), 4)

        reg = ForgettingRegressor(3, 4)
        points = [(1, 1), (2, 1), (1, 2), (2, 3), (4, 1), (1, 4), (3, 3), (2, 2)]
        for x1, x2 in points:
            reg.update([1.0, float(x1), float(x2)], 1.0 + 2.0 * x1 + 3.0 * x2)
        pred = reg.predict([1.0, 5.0, 5.0])
        self.assertIsNotNone(pred)
        self.assertAlmostEqual(pred, 1.0 + 10.0 + 15.0, places=3)

    def test_verify_prediction_waits_for_six_samples(self):
        ledger = ClearLedger(active_rids=["a"])
        closed = []
        for index in range(VERIFY_MIN_SAMPLES + 1):
            if index == 0:
                _step(ledger, ["a"], [20], 1, 0.0)
            ledger.mark_stop(float(index))
            ledger.mark_send(float(index))
            ledger.mark_post(float(index) + 0.002)
            ledger.note_idle()
            nxt = index + 1
            closed.append(
                _step(ledger, ["a"], [20 + nxt], nxt + 1, float(index) + 0.01)
            )
        self.assertIsNone(closed[VERIFY_MIN_SAMPLES - 1]["verify_pred_ms"])
        self.assertIsNotNone(closed[VERIFY_MIN_SAMPLES]["verify_pred_ms"])
        self.assertEqual(ledger.verify_samples, VERIFY_MIN_SAMPLES + 1)

    def test_npu_draft_keeps_t_k_out_of_the_verify_model(self):
        ledger = ClearLedger(backend="npu", active_rids=["a"])
        _step(ledger, ["a"], [8], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.0)
        ledger.mark_post(1.01)
        ledger.note_idle()
        record = _step(ledger, ["a"], [8], 2, 1.05)

        self.assertFalse(record["draft_floor"])
        self.assertFalse(record["excluded"])
        self.assertEqual(record["t_k_us"], 50000)
        self.assertEqual(record["kv_bucket"], 256)
        self.assertFalse(record["over_bucket"])
        self.assertFalse(record["verify_fit"])
        self.assertIsNone(record["verify_pred_ms"])
        self.assertEqual(ledger.verify_samples, 0)
        self.assertIsNone(ledger.verify_model)

    def test_target_device_us_fits_graph_rounds_only(self):
        closed = []
        for index in range(VERIFY_MIN_SAMPLES + 1):
            closed.append(
                target_verify_record(
                    rank=0,
                    batch_size=1,
                    k=4,
                    seq_lens=[20 + index],
                    ran_graph=True,
                    kv_bucket=256,
                    device_us=33000,
                    verify_id=index + 1,
                )
            )
        self.assertIsNone(closed[VERIFY_MIN_SAMPLES - 1]["verify_pred_ms"])
        self.assertTrue(closed[VERIFY_MIN_SAMPLES - 1]["verify_fit"])
        self.assertIsNotNone(closed[VERIFY_MIN_SAMPLES]["verify_pred_ms"])
        self.assertAlmostEqual(
            closed[VERIFY_MIN_SAMPLES]["verify_pred_ms"], 33.0, places=2
        )
        self.assertEqual(target_verify_samples(), VERIFY_MIN_SAMPLES + 1)

        before = target_verify_samples()
        eager = target_verify_record(
            rank=0,
            batch_size=1,
            k=4,
            seq_lens=[40],
            ran_graph=False,
            kv_bucket=256,
            device_us=33000,
        )
        over = target_verify_record(
            rank=0,
            batch_size=1,
            k=4,
            seq_lens=[5000],
            ran_graph=True,
            kv_bucket=None,
            device_us=60000,
        )
        pending = target_verify_record(
            rank=0,
            batch_size=1,
            k=4,
            seq_lens=[80],
            ran_graph=True,
            kv_bucket=256,
        )
        self.assertEqual(eager["path"], "eager")
        self.assertFalse(eager["verify_fit"])
        self.assertEqual(over["path"], "eager")
        self.assertFalse(over["verify_fit"])
        self.assertFalse(pending["verify_fit"])
        self.assertIsNone(pending["device_us"])
        self.assertEqual(target_verify_samples(), before)

    def test_target_verify_over_bucket_is_eager(self):
        buckets = (256, 512)
        self.assertEqual(
            smallest_bucket(100, buckets), select_tree_kv_bucket(100, buckets)
        )
        self.assertEqual(
            smallest_bucket(300, buckets), select_tree_kv_bucket(300, buckets)
        )
        self.assertIsNone(smallest_bucket(900, buckets))
        self.assertIsNone(select_tree_kv_bucket(900, buckets))

        inside = target_verify_record(
            rank=0,
            batch_size=2,
            k=8,
            seq_lens=[100, 200],
            ran_graph=True,
            kv_bucket=256,
        )
        self.assertEqual(inside["path"], "graph")
        self.assertFalse(inside["over_bucket"])
        self.assertIsNone(inside["verify_host_us"])
        self.assertIsNone(inside["device_us"])

        timed = target_verify_record(
            rank=0,
            batch_size=1,
            k=4,
            seq_lens=[300],
            ran_graph=True,
            kv_bucket=512,
            verify_host_us=1234,
            device_us=1100,
            verify_id=7,
            rpc_seq=9,
        )
        self.assertEqual(timed["verify_host_us"], 1234)
        self.assertEqual(timed["device_us"], 1100)
        self.assertEqual(timed["sum_ell"], 300)
        self.assertEqual(timed["verify_id"], 7)
        self.assertEqual(timed["rpc_seq"], 9)

        outside = target_verify_record(
            rank=0,
            batch_size=1,
            k=8,
            seq_lens=[4000],
            ran_graph=True,
            kv_bucket=None,
        )
        self.assertEqual(outside["path"], "eager")
        self.assertTrue(outside["over_bucket"])
        self.assertIsNone(
            target_verify_record(
                rank=1,
                batch_size=1,
                k=8,
                seq_lens=[1],
                ran_graph=True,
                kv_bucket=256,
            )
        )

    def test_verify_host_us_reads_the_existing_counter(self):
        metrics = SimpleNamespace(_round_host={"verify_forward": 0.00125})
        self.assertEqual(read_verify_host_us(metrics), 1250)
        self.assertIsNone(read_verify_host_us(SimpleNamespace(_round_host={})))

    def test_ledger_device_event_is_every_round_and_polled_without_sync(self):
        start, end = Mock(), Mock()
        end.query.side_effect = [False, True]
        start.elapsed_time.return_value = 1.5
        metrics = SRRoundMetrics(
            "Target", SimpleNamespace(Event=Mock(side_effect=[start, end]))
        )
        metrics._clear_ledger_device = True
        metrics._clear_device_emit = emit_verify_device
        self.assertNotEqual(metrics.rounds, 15)
        with metrics.round():
            with metrics.phase("verify_forward", device=True):
                pass
        self.assertFalse(metrics.pending)
        self.assertEqual(len(metrics._clear_device_open), 1)
        meta = {"verify_id": 3, "rpc_seq": 8, "sum_ell": 400, "kv_bucket": 512}
        self.assertIsNone(consume_verify_device(metrics, meta))
        self.assertFalse(metrics._clear_device_open)
        self.assertEqual(len(metrics._clear_device_pending), 1)
        logged = []

        def capture(fmt, kind, payload):
            logged.append((kind, payload))

        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.logger.info",
            side_effect=capture,
        ):
            metrics.poll()
        self.assertFalse(metrics._clear_device_pending)
        self.assertFalse(metrics.device_samples)
        end.synchronize.assert_not_called()
        kind, payload = logged[0]
        self.assertEqual(kind, "t_verify_device")
        self.assertIn('"device_us":1500', payload)
        self.assertIn('"sum_ell":400', payload)
        self.assertIn('"kv_bucket":512', payload)
        self.assertIn('"rpc_seq":8', payload)
        self.assertIn('"verify_fit":false', payload)
        self.assertEqual(target_verify_samples(), 0)

    def test_late_device_event_fits_without_sync(self):
        start, end = Mock(), Mock()
        end.query.side_effect = [False, True]
        start.elapsed_time.return_value = 39.0
        metrics = SRRoundMetrics(
            "Target", SimpleNamespace(Event=Mock(side_effect=[start, end]))
        )
        metrics._clear_ledger_device = True
        metrics._clear_device_emit = emit_verify_device
        with metrics.round():
            with metrics.phase("verify_forward", device=True):
                pass
        meta = {
            "verify_id": 4,
            "rpc_seq": 9,
            "sum_ell": 200,
            "kv_bucket": 256,
            "b": 1,
            "k": 16,
            "ran_graph": True,
        }
        self.assertIsNone(consume_verify_device(metrics, meta))
        logged = []

        def capture(fmt, kind, payload):
            logged.append((kind, payload))

        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.logger.info",
            side_effect=capture,
        ):
            metrics.poll()
        end.synchronize.assert_not_called()
        self.assertEqual(target_verify_samples(), 1)
        kind, payload = logged[0]
        self.assertEqual(kind, "t_verify_device")
        self.assertIn('"device_us":39000', payload)
        self.assertIn('"verify_fit":true', payload)


class TestFlatWidthOnlineK(CustomTestCase):
    def setUp(self):
        reset_policy()

    def test_price_ema_floor_leaves_the_cumulative_ratio(self):
        for _ in range(20):
            observe_price(1, 10.0, 10.0)
        self.assertAlmostEqual(theta(1), 1.0)
        observe_price(1, 0.0, 10.0)
        self.assertAlmostEqual(theta(1), 0.95)
        self.assertNotAlmostEqual(theta(1), 200.0 / 210.0)

    def test_unseen_batch_uses_the_other_batch_prior(self):
        observe_price(1, 10.0, 10.0)
        self.assertAlmostEqual(theta(2), 2.0)
        sched = SimpleNamespace(_clear_ledger=False)
        reqs = [
            SimpleNamespace(rid="a", origin_input_ids=[1], output_ids=[]),
            SimpleNamespace(rid="b", origin_input_ids=[1], output_ids=[]),
        ]
        plan = make_round(sched, reqs)
        self.assertAlmostEqual(plan.theta_value, 2.0)
        observe_price(4, 20.0, 40.0)
        self.assertAlmostEqual(theta(2), 2.0**-0.5)

    def test_subtracting_prefill_differs_from_dropping_the_round(self):
        report = compare_price(
            [
                {
                    "b": 1,
                    "price_tokens": 10,
                    "price_ms": 10.0,
                    "price_kept": True,
                    "exclude_reason": [],
                    "gap_rpc": [],
                },
                {
                    "b": 1,
                    "price_tokens": 2,
                    "price_ms": 20.0,
                    "price_kept": False,
                    "exclude_reason": ["prefill"],
                    "gap_rpc": [7],
                },
                {
                    "b": 1,
                    "price_tokens": 100,
                    "price_ms": 10.0,
                    "price_kept": False,
                    "exclude_reason": ["finish"],
                    "gap_rpc": [8],
                },
                {
                    "b": 1,
                    "price_tokens": 4,
                    "price_ms": 10.0,
                    "price_kept": False,
                    "exclude_reason": ["idle_start"],
                    "gap_rpc": [],
                },
            ],
            [{"rpc_seq": 7, "wall_us": 10000, "action": "prefill"}],
        )
        self.assertAlmostEqual(report[1]["discard"], 1.0)
        self.assertAlmostEqual(report[1]["deduct"], 16.0 / 30.0)
        self.assertAlmostEqual(report[1]["relative"], 14.0 / 30.0)

    def test_flat_width(self):
        self.assertEqual(flat_width(1), 8)
        self.assertEqual(flat_width(2), 4)
        self.assertEqual(flat_width(8), 4)

    def test_layer_forward_drops_one_request(self):
        width = 4
        vocab = 8
        config = ClearTreeConfig(width=width, tau=0.4, max_depth=8)
        cached = [0, 0]
        calls = []

        class _DropSecond:
            def begin_layer(self, states):
                self.peers = list(states)

            def should_grow(self, state):
                if int(state.depth) >= int(state.config.max_depth) - 1:
                    return False
                if int(state.depth) <= 1:
                    return True
                if int(state.depth) >= 3:
                    return False
                return state is self.peers[0]

            def verify_k(self, state):
                return width

            def note(self, state, keep):
                return None

        def forward_active(steps, indices):
            calls.append(list(indices))
            for index in indices:
                cached[index] += width
            rows = torch.full((len(indices), width, vocab), -8.0)
            for rank in range(min(width + 1, vocab)):
                rows[:, :, rank] = float(6 - rank)
            return rows

        tail = torch.full((vocab,), -8.0)
        for rank in range(min(width + 1, vocab)):
            tail[rank] = float(6 - rank)
        scheduler = SimpleNamespace(
            _clear_round_plan=_DropSecond(),
            _clear_round_rids=["a", "b"],
            _clear_round_prefixes=[4, 9],
        )
        windows = grow_active_layers(
            config, [tail.clone(), tail.clone()], forward_active, scheduler
        )
        self.assertEqual(calls[0], [0, 1])
        self.assertEqual(calls[1], [0])
        stopped = cached[1]
        self.assertEqual(stopped, width)
        self.assertGreater(cached[0], stopped)
        self.assertEqual(len(windows), 2)
        self.assertTrue(windows[0][0])
        self.assertTrue(windows[1][0])

    def test_each_layer_shape_is_priced_once_across_k(self):
        times = {8: 1.0, 16: 1.0, 32: 1.0}
        rows = [
            {
                "item": name,
                "e": dict.fromkeys(times, 1.0),
                "g": dict.fromkeys(times, gain),
                "c": 1.0,
                "ell": prefix,
            }
            for name, gain, prefix in (("good", 2.0, 4), ("poor", 0.0, 8))
        ]
        for active in (1, 2):
            with self.subTest(active=active):
                cost = Mock(side_effect=lambda m, _ell: 0.1 if m == 1 else None)
                slope = Mock(return_value=0.0)
                kept = keep_by_net_benefit(
                    rows[:active],
                    theta=1.0,
                    verify_times=times,
                    layer_cost=cost,
                    kappa=slope,
                )
                self.assertEqual(kept, ["good"])
                expected = [(1, 4)] if active == 1 else [(1, 4), (2, 12)]
                self.assertEqual([call.args for call in cost.call_args_list], expected)
                self.assertEqual(slope.call_count, 0 if active == 1 else 2)

    def test_round_reuses_scores_until_the_next_layer(self):
        state = _rows_state([[0.5, 0.25], [0.125, 0.0625]], "live", 4)
        state.depth = 1
        plan = _value_plan()
        plan._time = lambda _k: 1.0
        plan._layer_cost = lambda _active, _ell: 0.1
        plan.bind_states([state])
        with patch.object(policy, "_path_scores", wraps=policy._path_scores) as scores:
            plan.begin_layer([state])
            plan.should_grow(state)
            self.assertEqual(plan._values([state], include_gain=False)[8], -0.25)
            keep = plan.verify_k(state)
            plan.note(state, keep)
            scores.assert_called_once_with(state)

            state.depth = 2
            plan.begin_layer([state])
            plan.should_grow(state)
            self.assertEqual(plan._values([state], include_gain=False)[8], -0.0625)
            self.assertEqual(scores.call_count, 2)

    def test_only_the_positive_net_request_continues(self):
        good = _rows_state([[0.5, 0, 0, 0, 0, 0, 0, 0]], "good", 1)
        poor = _rows_state([[0.05, 0, 0, 0, 0, 0, 0, 0]], "poor", 100)
        kept = select_continuing(
            [good, poor],
            [good, poor],
            theta=1.0,
            verify_times={8: 1.0, 16: 1.0, 32: 1.0},
            layer_cost=lambda active, _ell: 0.05 if int(active) == 1 else 0.5,
            kappa=0.1,
        )
        self.assertEqual([item.rid for item in kept], ["good"])

    def test_a_step_in_layer_cost_can_keep_fewer_requests(self):
        first = _rows_state([[0.5, 0, 0, 0, 0, 0, 0, 0]], "first", 4)
        second = _rows_state([[0.4, 0, 0, 0, 0, 0, 0, 0]], "second", 4)
        times = {8: 1.0, 16: 1.0, 32: 1.0}
        flat = select_continuing(
            [first, second],
            [first, second],
            theta=1.0,
            verify_times=times,
            layer_cost=lambda _active, _ell: 0.01,
            kappa=0.0,
        )
        stepped = select_continuing(
            [first, second],
            [first, second],
            theta=1.0,
            verify_times=times,
            layer_cost=lambda active, _ell: 0.01 if int(active) == 1 else 10.0,
            kappa=0.0,
        )
        self.assertEqual({item.rid for item in flat}, {"first", "second"})
        self.assertEqual([item.rid for item in stepped], ["first"])

    def test_a_stopped_request_still_changes_later_j(self):
        live = _rows_state(
            [
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            "live",
            4,
        )
        stopped = _rows_state([[0.125] * 8] * 4, "stopped", 4)
        times = {8: 0.0, 16: 3.0, 32: 3.0}
        alone = select_continuing(
            [live],
            [live],
            theta=1.0,
            verify_times=times,
            layer_cost=lambda _active, _ell: 0.3,
            kappa=0.0,
        )
        joint = select_continuing(
            [stopped, live],
            [live],
            theta=1.0,
            verify_times=times,
            layer_cost=lambda _active, _ell: 0.3,
            kappa=0.0,
        )
        self.assertEqual(alone, [])
        self.assertEqual([item.rid for item in joint], ["live"])
        mass, _gain = request_terms(stopped, 32)
        self.assertGreater(mass, request_terms(stopped, 8)[0])

    def test_deferred_windows_share_one_k(self):
        width = 4
        vocab = 8
        config = ClearTreeConfig(width=width, tau=0.4, max_depth=8)
        plan = RoundPlan(
            width=width,
            k=8,
            use_tau=False,
            locked_k=None,
            theta_value=1.0,
            layer_ms=1.0,
            max_ell=10,
            batch=2,
            buckets=(256,),
        )

        def choose(eligible):
            if not eligible or int(eligible[0].depth) >= 3:
                return []
            if int(eligible[0].depth) <= 1:
                return list(eligible)
            return [eligible[0]]

        plan._choose_keep = choose
        plan._select = lambda states: 8

        def forward_active(steps, indices):
            rows = torch.full((len(indices), width, vocab), -8.0)
            for rank in range(min(width + 1, vocab)):
                rows[:, :, rank] = float(6 - rank)
            return rows

        tail = torch.full((vocab,), -8.0)
        for rank in range(min(width + 1, vocab)):
            tail[rank] = float(6 - rank)
        scheduler = SimpleNamespace(
            _clear_round_plan=plan,
            _clear_round_rids=["a", "b"],
            _clear_round_prefixes=[4, 9],
        )
        windows = grow_active_layers(
            config, [tail.clone(), tail.clone()], forward_active, scheduler
        )
        self.assertEqual(plan._frozen_k, 8)
        self.assertEqual(len(windows[0][0]), 8)
        self.assertEqual(len(windows[1][0]), 8)
        self.assertEqual(len(windows[0][1]), 8)
        self.assertEqual(len(windows[1][1]), 8)

    def test_single_request_matches_the_whole_batch(self):
        state = _score_state([0.4, 0.2, 0.1, 0.05])
        for k in (8, 16, 32):
            observe_verify(1, k, 256, 10.0)
        for _ in range(3):
            observe_verify(1, 8, 256, 10.0)
        for layer_ms in (0.0, 1000.0):
            plan = _value_plan()
            plan.layer_ms = layer_ms
            plan.bind_states([state])
            plan.begin_layer([state])
            current = plan._values([state], include_gain=False)
            nxt = plan._values([state], include_gain=True)
            old = layer_should_continue(
                max(current.values()),
                max(nxt.values()) - float(plan.theta_value) * float(layer_ms),
            )
            self.assertEqual(plan.should_grow(state), old)
            plan._at = None
            plan._layer_cost = lambda _active, _ell: float(layer_ms)
            plan._kappa = lambda: 0.0
            self.assertEqual(plan.should_grow(state), old)

    def test_equal_verify_time_keeps_the_largest_k(self):
        reset_policy()
        weight = torch.zeros(4, 8)
        weight[0, :4] = torch.tensor([0.4, 0.2, 0.1, 0.05])
        state = SimpleNamespace(
            depth=4,
            weight=weight,
            rid="r",
            prefix=4,
            config=SimpleNamespace(max_depth=8, tau=0.4, width=8),
        )
        for k in (8, 16, 32):
            observe_verify(1, k, 256, 10.0)
        for _ in range(3):
            observe_verify(1, 8, 256, 10.0)
        plan = _value_plan()
        plan.begin_layer([state])
        self.assertEqual(plan.verify_k(state), 32)

    def test_slower_long_graph_can_pick_a_shorter_k(self):
        state = _score_state([0.4, 0.2, 0.1, 0.05])
        observe_verify(1, 8, 256, 10.0)
        observe_verify(1, 16, 256, 40.0)
        observe_verify(1, 32, 256, 50.0)
        for _ in range(3):
            observe_verify(1, 8, 256, 10.0)
        plan = _value_plan()
        plan.begin_layer([state])
        self.assertEqual(plan.verify_k(state), 8)

    def test_short_sample_and_probe_stay_on_tau(self):
        _pin_verify_ks(self)
        reset_policy()
        sched = SimpleNamespace(_clear_ledger=False)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        locked = [make_round(sched, [req]).locked_k for _ in range(6)]
        self.assertEqual(locked, [16, 16, 16, 16, 16, 16])
        warmed = make_round(sched, [req])
        self.assertTrue(warmed.use_tau)
        self.assertEqual(warmed.locked_k, 16)
        self.assertEqual(policy._probe_rounds, 0)

    def test_new_batch_restarts_probes_at_k8(self):
        _pin_verify_ks(self)
        reset_policy()
        sched = SimpleNamespace(_clear_ledger=False)
        one = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        two = [
            one,
            SimpleNamespace(rid="b", origin_input_ids=[1] * 10, output_ids=[]),
        ]
        locked = [make_round(sched, [one]).locked_k for _ in range(6)]
        self.assertEqual(locked, [16, 16, 16, 16, 16, 16])
        again = make_round(sched, two)
        self.assertTrue(again.use_tau)
        self.assertEqual(again.locked_k, 8)
        self.assertEqual(policy._probe_rounds, 0)

    def test_missing_tier_stays_on_tau_until_every_k_is_timed(self):
        _pin_verify_ks(self)
        reset_policy()
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        self.addCleanup(lambda: os.environ.pop("SGLANG_CLEAR_LEDGER", None))
        sched = SimpleNamespace(_clear_ledger=False)
        reqs = [
            SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[]),
            SimpleNamespace(rid="b", origin_input_ids=[1] * 10, output_ids=[]),
        ]
        for k in (16, 32):
            observe_verify(2, k, 256, 10.0)
        for _ in range(4):
            observe_verify(2, 16, 256, 10.0)
        observe_price(2, 2.0, 1.0)
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            for _ in range(10):
                make_round(sched, reqs)
            held = make_round(sched, reqs)
        self.assertTrue(held.use_tau)
        self.assertEqual(held.locked_k, 8)
        self.assertEqual(policy._probe_rounds, 0)
        observe_verify(2, 8, 256, 10.0)
        observe_verify(2, 8, 256, 10.0)
        observe_verify(2, 32, 256, 10.0)
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            left = make_round(sched, reqs)
        self.assertFalse(left.use_tau)
        self.assertIsNone(left.locked_k)

    def test_flat_mass_selects_k8_when_it_is_not_slower(self):
        state = _score_state([0.4, 0.2, 0.1, 0.05])
        observe_verify(2, 8, 256, 10.0)
        observe_verify(2, 16, 256, 12.0)
        observe_verify(2, 32, 256, 14.0)
        for _ in range(3):
            observe_verify(2, 8, 256, 10.0)
        plan = _value_plan()
        plan.batch = 2
        plan.begin_layer([state])
        values = plan._values([state], include_gain=False)
        self.assertEqual(set(values), {8, 16, 32})
        self.assertGreater(values[8], values[16])
        self.assertGreater(values[8], values[32])
        self.assertEqual(plan.verify_k(state), 8)

    def test_ready_sample_leaves_tau(self):
        reset_policy()
        sched = SimpleNamespace(_clear_ledger=False)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        for k in (2, 4, 8, 16, 32, 64):
            observe_verify(1, k, 256, 10.0)
            observe_verify(1, k, 256, 10.0)
        for _ in range(3):
            observe_verify(1, 8, 256, 10.0)
        observe_price(1, 2.0, 1.0)
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.ledger_enabled",
            return_value=True,
        ), patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            plan = make_round(sched, [req])
        self.assertFalse(plan.use_tau)
        self.assertIsNone(plan.locked_k)
        self.assertEqual(plan.width, 4)
        self.assertEqual(plan.replay_kind, "probe_w")
        self.assertFalse(plan.replay_eligible)

    def test_accept_ratio_shrinks_toward_the_pool(self):
        self.assertEqual(accept_ratio("a"), 1.0)
        observe_accept("a", 1.0, 4.0)
        self.assertAlmostEqual(accept_ratio("a"), 1.75 / 7.0)

    def test_pad_parent_is_not_linked(self):
        layout = build_tree_from_parents(
            torch.tensor([[-1, -2]]),
            torch.tensor([3]),
        )
        self.assertEqual(int(layout.retrive_next_token[0, 0]), 1)
        self.assertEqual(int(layout.retrive_next_sibling[0, 1]), -1)
        self.assertEqual(int(layout.retrive_next_token[0, 2]), -1)

    def test_verify_samples_round_trip_and_stay_off_sr(self):
        req = ClearDraftRequest(rid="a", step_id=1, base_committed_len=4)
        batch = ClearBatchRequest(
            session_id="s",
            rpc_seq=3,
            action=ClearAction.STEP,
            reqs=[req],
            verify_samples=[[1, 8, 256, 33000], [1, 16, 256, 40000]],
        )
        decoded, _mm = _unpack_request(_pack_request(batch))
        self.assertEqual(
            decoded.verify_samples, [[1, 8, 256, 33000], [1, 16, 256, 40000]]
        )
        self.assertEqual(decoded.reqs[0].rid, "a")
        empty = ClearBatchRequest(session_id="s", rpc_seq=1, action=ClearAction.STEP)
        self.assertNotIn("verify_samples", empty.to_dict())
        shared = dict(batch.to_dict())
        sr = SRBatchRequest.from_dict(shared)
        self.assertFalse(hasattr(sr, "verify_samples"))
        opt = SRBatchRequest(session_id="s", rpc_seq=1, action=SRAction.STEP)
        self.assertNotIn("verify_samples", opt.to_dict())
        self.assertEqual(
            set(opt.to_dict()),
            {"session_id", "rpc_seq", "action", "reqs", "protocol_version"},
        )

    def test_spike_does_not_make_the_short_graph_look_slower(self):
        observe_verify(1, 8, 256, 10.0)
        observe_verify(1, 8, 256, 40.0)
        for _ in range(4):
            observe_verify(1, 32, 256, 12.0)
        self.assertAlmostEqual(predict_verify_ms(1, 8, 256), 10.0)
        state = _score_state([0.4, 0.2, 0.1, 0.05])
        plan = _value_plan()
        plan.begin_layer([state])
        self.assertEqual(plan.verify_k(state), 8)

    def test_probe_skips_k_that_does_not_fit(self):
        _pin_verify_ks(self)
        reset_policy()
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        self.addCleanup(lambda: os.environ.pop("SGLANG_CLEAR_LEDGER", None))
        sched = SimpleNamespace(_clear_ledger=False)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 4070, output_ids=[])
        saved = os.environ.get("SGLANG_NPU_TREE_GRAPH_KV_BUCKETS")
        os.environ["SGLANG_NPU_TREE_GRAPH_KV_BUCKETS"] = "256,512,1024,2048,4096"
        try:
            observe_verify(1, 16, 4096, 10.0)
            probed = [make_round(sched, [req]).locked_k for _ in range(2)]
            observe_verify(1, 8, 4096, 10.0)
            observe_verify(1, 8, 4096, 10.0)
            rested = [make_round(sched, [req]).locked_k for _ in range(2)]
        finally:
            if saved is None:
                os.environ.pop("SGLANG_NPU_TREE_GRAPH_KV_BUCKETS", None)
            else:
                os.environ["SGLANG_NPU_TREE_GRAPH_KV_BUCKETS"] = saved
        self.assertEqual(probed, [8, 8])
        self.assertEqual(rested, [16, 16])
        self.assertNotIn(32, probed + rested)

    def test_ingested_sample_is_visible_before_the_round_plan(self):
        for _ in range(4):
            observe_verify(1, 8, 256, 10.0)
        observe_verify(1, 16, 256, 10.0)
        observe_verify(1, 32, 256, 10.0)
        self.assertEqual(predict_verify_ms(1, 8, 256), 10.0)
        before = {key: list(rows) for key, rows in policy._clock.items()}
        ingest_verify_samples(
            [[1, 8, 256, 10000], [1, 16, 256, 10000], [1, 32, 256, 10000]]
        )
        self.assertEqual(
            {key: list(rows) for key, rows in policy._clock.items()}, before
        )

    def test_lock_k_does_not_consume_probes(self):
        _pin_verify_ks(self)
        reset_policy()
        sched = SimpleNamespace(_clear_ledger=False)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        os.environ["SGLANG_CLEAR_LOCK_K"] = "32"
        try:
            plan = make_round(sched, [req])
            self.assertTrue(plan.use_tau)
            self.assertEqual(plan.locked_k, 32)
        finally:
            os.environ.pop("SGLANG_CLEAR_LOCK_K", None)
            nxt = make_round(sched, [req])
        self.assertEqual(nxt.locked_k, 16)
        self.assertEqual(policy._probe_rounds, 0)

    def test_truncate_drops_the_weaker_branch(self):
        layers = [
            [
                {"token": 1, "parent": 0, "score": 0.9},
                {"token": 2, "parent": 1, "score": 0.1},
            ],
            [
                {"token": 3, "parent": 0, "score": 0.4},
                {"token": 4, "parent": 1, "score": 0.8},
            ],
        ]
        kept = truncate_columns(layers, 1)
        self.assertEqual(kept, [[0], [0]])

    def test_batch_one_builds_the_next_wider_width(self):
        reset_policy()
        ready = lambda _width: True
        chosen = [
            choose_build_width(1, ledger_on=True, eligible=True, ready=ready)
            for _ in range(9)
        ]
        self.assertEqual(chosen[:4], [(4, "probe_w")] * 4)
        self.assertEqual(chosen[4:8], [(8, "ordinary")] * 4)
        self.assertEqual(chosen[8], (16, "superset"))
        assign_persistent_width(1, 4)
        later = [
            choose_build_width(1, ledger_on=True, eligible=True, ready=ready)
            for _ in range(9)
        ]
        self.assertEqual(later[:4], [(2, "probe_w")] * 4)
        self.assertEqual(later[4:8], [(4, "ordinary")] * 4)
        self.assertEqual(later[8], (8, "superset"))

    def test_widest_width_does_not_leave(self):
        reset_policy()
        assign_persistent_width(1, 64)
        chosen = [
            choose_build_width(1, ledger_on=True, eligible=True, ready=lambda _w: True)
            for _ in range(9)
        ]
        self.assertEqual(chosen[:4], [(32, "probe_w")] * 4)
        self.assertTrue(all(item == (64, "ordinary") for item in chosen[4:]))

    def test_wider_batch_builds_a_superset_only_while_the_ledger_is_on(self):
        reset_policy()
        self.assertEqual(
            choose_build_width(2, ledger_on=False, eligible=True), (4, "ordinary")
        )
        cold = [
            choose_build_width(2, ledger_on=True, eligible=False) for _ in range(9)
        ]
        self.assertTrue(all(item == (4, "ordinary") for item in cold))
        hot = [
            choose_build_width(2, ledger_on=True, eligible=True, ready=lambda _w: True)
            for _ in range(9)
        ]
        self.assertEqual(hot[:4], [(2, "probe_w")] * 4)
        self.assertEqual(hot[4:8], [(4, "ordinary")] * 4)
        self.assertEqual(hot[8], (8, "superset"))

    def test_probe_cover_follows_served_width_at_depth_eight(self):
        for batch, width, expected in (
            (4, 4, [4, 16, 2, 32]),
            (1, 8, [8, 32, 4, 64, 2]),
            (1, 4, [8, 32, 4, 2]),
        ):
            with self.subTest(batch=batch, width=width):
                reset_policy()
                assign_persistent_width(batch, width)
                scheduler = SimpleNamespace(
                    server_args=SimpleNamespace(clear_tree_max_depth=8)
                )
                reqs = [
                    SimpleNamespace(
                        rid=str(index), origin_input_ids=[1] * 10, output_ids=[]
                    )
                    for index in range(batch)
                ]
                flat = 8 if batch == 4 else 16
                observe_verify(batch, flat, 256, 10.0)
                probed = []
                with patch.object(ledger_module, "ledger_enabled", return_value=True):
                    for k in expected:
                        for _ in range(policy.PROBE_RUNS):
                            plan = make_round(scheduler, reqs)
                            self.assertTrue(plan.use_tau)
                            self.assertEqual(plan.width, width)
                            self.assertEqual(plan.locked_k, k)
                            self.assertFalse(plan.replay_eligible)
                            observe_verify(batch, k, 256, 10.0)
                        probed.append(k)
                    done = make_round(scheduler, reqs)
                self.assertEqual(probed, expected)
                self.assertEqual(done.locked_k, flat)
                self.assertEqual(
                    policy._probe_rounds, len(expected) * policy.PROBE_RUNS
                )
                if width == 4:
                    self.assertNotIn(64, probed)
                else:
                    self.assertIn(64, probed)

    def test_rewritten_zero_temperature_builds_a_superset(self):
        reset_policy()
        sched = SimpleNamespace(_clear_ledger=False)
        params = SamplingParams(temperature=0)
        reqs = [
            SimpleNamespace(
                rid=rid,
                origin_input_ids=[1] * 10,
                output_ids=[],
                sampling_params=params,
            )
            for rid in ("a", "b")
        ]
        for k in (2, 4, 8, 16, 32, 64):
            observe_verify(2, k, 256, 5.0)
            observe_verify(2, k, 256, 5.0)
        observe_price(2, 2.0, 1.0)
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.ledger_enabled",
            return_value=True,
        ), patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            eligible = []
            for _ in range(9):
                plan = make_round(sched, reqs)
                eligible.append((plan.width, plan.replay_kind, plan.replay_eligible))
        self.assertEqual(
            eligible,
            [(2, "probe_w", False)] * 4
            + [(4, "ordinary", True)] * 4
            + [(8, "superset", True)],
        )

        reset_policy()
        warm = SimpleNamespace(
            rid="a",
            origin_input_ids=[1] * 10,
            output_ids=[],
            sampling_params=SamplingParams(temperature=0.7),
        )
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.ledger_enabled",
            return_value=True,
        ):
            widths = [make_round(sched, [warm]).width for _ in range(40)]
        self.assertTrue(all(width == 8 for width in widths))

    def test_nonzero_temperature_stays_on_the_flat_width(self):
        sched = SimpleNamespace(_clear_ledger=False)
        req = SimpleNamespace(
            rid="a",
            origin_input_ids=[1] * 10,
            output_ids=[],
            sampling_params=SimpleNamespace(temperature=0.7),
        )
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.ledger_enabled",
            return_value=True,
        ):
            widths = [make_round(sched, [req]).width for _ in range(40)]
        self.assertTrue(all(width == 8 for width in widths))

    def test_verify_sample_queue_is_sent_once(self):
        metrics = SimpleNamespace(tp_rank=0)
        push_verify_sample(metrics, 1, 8, 256, 1000, True)
        push_verify_sample(metrics, 1, 16, 256, 2000, True)
        push_verify_sample(metrics, 1, 3, 256, 3000, True)
        push_verify_sample(metrics, 1, 8, None, 1000, True)
        push_verify_sample(metrics, 1, 8, 256, 1000, False)
        self.assertEqual(
            take_verify_samples(metrics),
            [[1, 8, 256, 1000], [1, 16, 256, 2000]],
        )
        self.assertEqual(take_verify_samples(metrics), [])
        for index in range(10):
            push_verify_sample(metrics, 1, 8, 256, 1000 + index, True)
        queued = take_verify_samples(metrics)
        self.assertEqual(len(queued), 8)
        self.assertEqual(queued[0][3], 1002)

    def test_accept_count_drops_the_bonus(self):
        reset_policy()
        note_verified("a", 10, 2.0)
        settle_verified(["a"], [13])
        self.assertAlmostEqual(accept_ratio("a"), 1.0)

    def test_snapshot_prefix_does_not_add_the_prompt(self):
        reset_policy()
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        self.addCleanup(lambda: os.environ.pop("SGLANG_CLEAR_LEDGER", None))
        sched = SimpleNamespace(
            server_args=SimpleNamespace(clear_tree_width=8),
            tp_rank=0,
        )

        def packet(base, committed, rpc):
            req = SimpleNamespace(
                rid="a",
                base_committed_len=base,
                committed_ids=committed,
            )
            return SimpleNamespace(
                reqs=[req],
                action="step",
                session_id="s",
                rpc_seq=rpc,
            )

        note_packet(sched, packet(1003, [1, 2, 3], 1))
        ledger = sched._clear_ledger
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        note_packet(sched, packet(1006, None, 2))
        ledger_module.run_deferred_close(sched)
        closed = next(
            record
            for kind, record in ledger.pending_logs
            if kind == "d_round"
        )
        self.assertEqual(closed["price_tokens"], 3)
        self.assertEqual(sched._clear_target_prefix, {"a": 1006})
        other = SimpleNamespace(
            reqs=[SimpleNamespace(rid="b", base_committed_len=4, committed_ids=None)],
            action="step",
            session_id="s",
            rpc_seq=3,
        )
        note_packet(sched, other)
        self.assertEqual(sched._clear_target_prefix, {"b": 4})

    def test_target_prefix_sizes_the_verify_bucket(self):
        reset_policy()
        sched = SimpleNamespace(
            _clear_build_token=7,
            _clear_target_prefix={"a": 500},
        )
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        plan = attach_round(sched, [req])
        self.assertEqual(plan.max_ell, 500)
        self.assertEqual(sched._clear_round_prefixes, [10])

    def test_frozen_rows_omit_the_unforwarded_layer(self):
        state = SimpleNamespace(depth=3, prefix=10)

        def predict(_scheduler, width, _active, _sum_ell, depth=None):
            return 7.0 if int(width) == 8 else 5.0

        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            side_effect=predict,
        ):
            rows = _freeze_layer_rows(SimpleNamespace(), [state])
        self.assertEqual(len(rows), 2)
        verify = {8: 10.0, 16: 10.0, 32: 10.0}
        actual = _schedule_ms(rows, 8, len(rows), verify, 8)
        narrower = _schedule_ms(rows, 4, 2, verify, 8)
        self.assertAlmostEqual(actual - narrower, 4.0)

    def test_probe_uses_the_sample_bucket_and_can_retry(self):
        _pin_verify_ks(self)
        reset_policy()
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        self.addCleanup(lambda: os.environ.pop("SGLANG_CLEAR_LEDGER", None))
        sched = SimpleNamespace()
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 240, output_ids=[])
        saved = os.environ.get("SGLANG_NPU_TREE_GRAPH_KV_BUCKETS")
        os.environ["SGLANG_NPU_TREE_GRAPH_KV_BUCKETS"] = "256,512,1024,2048,4096"
        try:
            observe_verify(1, 16, 512, 10.0)
            self.assertEqual(make_round(sched, [req]).locked_k, 8)
            self.assertEqual(make_round(sched, [req]).locked_k, 8)
            observe_verify(1, 8, 512, 10.0)
            self.assertEqual(make_round(sched, [req]).locked_k, 8)
            observe_verify(1, 8, 256, 10.0)
            observe_verify(1, 8, 256, 10.0)
            self.assertEqual(make_round(sched, [req]).locked_k, 32)
        finally:
            if saved is None:
                os.environ.pop("SGLANG_NPU_TREE_GRAPH_KV_BUCKETS", None)
            else:
                os.environ["SGLANG_NPU_TREE_GRAPH_KV_BUCKETS"] = saved

    def test_probe_limit_follows_rounds_and_tier_count(self):
        reset_policy()
        policy._rounds = 100
        self.assertEqual(policy._probe_limit(), 12)
        policy._rounds = 1000
        self.assertEqual(policy._probe_limit(), 50)

    def test_flat_sample_alone_leaves_tau_when_probes_are_spent(self):
        reset_policy()
        policy._probe_rounds = 12
        observe_verify(1, 16, 256, 10.0)
        observe_price(1, 2.0, 1.0)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.ledger_enabled",
            return_value=True,
        ), patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            plan = make_round(SimpleNamespace(), [req])
        self.assertFalse(plan.use_tau)
        self.assertIsNone(plan.locked_k)
        self.assertEqual(policy._probe_rounds, 12)
        state = _score_state([0.4, 0.2, 0.1, 0.05])
        plan.begin_layer([state])
        self.assertEqual(plan.verify_k(state), 16)

    def test_nonflat_samples_stay_on_tau_and_do_not_probe(self):
        reset_policy()
        for k in (8, 32):
            observe_verify(1, k, 256, 10.0)
            observe_verify(1, k, 256, 10.0)
        observe_price(1, 2.0, 1.0)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.ledger_enabled",
            return_value=True,
        ), patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            plan = make_round(SimpleNamespace(), [req])
        self.assertTrue(plan.use_tau)
        self.assertEqual(plan.locked_k, 16)
        self.assertEqual(policy._probe_rounds, 0)

    def test_batch_one_probes_the_nearest_tier_first(self):
        reset_policy()
        saved = os.environ.get("SGLANG_CLEAR_LEDGER")
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        try:
            sched = SimpleNamespace()
            req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
            observe_verify(1, 16, 256, 10.0)
            order = []
            for expected in (8, 32, 4, 64, 2):
                for _ in range(policy.PROBE_RUNS):
                    plan = make_round(sched, [req])
                    self.assertTrue(plan.use_tau)
                    self.assertEqual(plan.locked_k, expected)
                    observe_verify(1, expected, 256, 10.0)
                order.append(expected)
            self.assertEqual(order, [8, 32, 4, 64, 2])
            done = make_round(sched, [req])
            self.assertEqual(done.locked_k, 16)
        finally:
            if saved is None:
                os.environ.pop("SGLANG_CLEAR_LEDGER", None)
            else:
                os.environ["SGLANG_CLEAR_LEDGER"] = saved

    def test_failed_send_puts_verify_samples_back(self):
        metrics = SimpleNamespace(_clear_verify_samples=[[1, 8, 256, 1000]])
        taken = take_verify_samples(metrics)
        self.assertEqual(metrics._clear_verify_samples, [])
        restore_verify_samples(metrics, taken)
        self.assertEqual(take_verify_samples(metrics), [[1, 8, 256, 1000]])

    def test_fallback_plan_does_not_count_or_probe(self):
        reset_policy()
        sched = SimpleNamespace(_clear_build_token=3, _clear_target_prefix={})
        reqs = [
            SimpleNamespace(rid="a", origin_input_ids=[1] * 4, output_ids=[]),
            SimpleNamespace(rid="b", origin_input_ids=[1] * 4, output_ids=[]),
        ]
        plan = install_fallback_plan(sched, reqs)
        self.assertTrue(plan.fallback)
        self.assertTrue(plan.use_tau)
        self.assertEqual(plan.locked_k, 8)
        self.assertFalse(plan.replay_eligible)
        self.assertEqual(policy._rounds, 0)
        self.assertEqual(policy._probe_rounds, 0)
        again = attach_round(sched, [reqs[0]])
        self.assertTrue(again.fallback)
        self.assertEqual(policy._rounds, 0)
        self.assertEqual(sched._clear_round_prefixes, [4])

    def test_mixed_k_rows_are_padded_not_dropped(self):
        tokens, parents, missing = align_draft_row([1] * 8, [-1] + [0] * 7, 16)
        self.assertFalse(missing)
        self.assertEqual(len(tokens), 16)
        self.assertEqual(parents[8:], [-2] * 8)
        _tokens, _parents, dropped = align_draft_row([1, 2, 3], [-1, 0, 0], 8)
        self.assertTrue(dropped)

    def test_ledger_off_does_not_probe(self):
        reset_policy()
        saved = os.environ.get("SGLANG_CLEAR_LEDGER")
        os.environ["SGLANG_CLEAR_LEDGER"] = "0"
        try:
            sched = SimpleNamespace()
            req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
            plan = make_round(sched, [req])
            self.assertTrue(plan.use_tau)
            self.assertEqual(plan.locked_k, 16)
            self.assertEqual(policy._probe_rounds, 0)
        finally:
            if saved is None:
                os.environ.pop("SGLANG_CLEAR_LEDGER", None)
            else:
                os.environ["SGLANG_CLEAR_LEDGER"] = saved

    def test_nonzero_temperature_locks_the_largest_k(self):
        reset_policy()
        for k in (8, 16, 32):
            observe_verify(1, k, 256, 10.0)
            observe_verify(1, k, 256, 10.0)
        observe_price(1, 2.0, 1.0)
        policy._rounds = 11
        req = SimpleNamespace(
            rid="a",
            origin_input_ids=[1] * 10,
            output_ids=[],
            sampling_params=SamplingParams(temperature=0.7),
        )
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            plan = make_round(SimpleNamespace(), [req])
        self.assertTrue(plan.use_tau)
        self.assertEqual(plan.locked_k, 16)
        self.assertFalse(plan.replay_eligible)

    def test_verify_window_keeps_the_latest_sixteen(self):
        reset_policy()
        for _ in range(20):
            observe_verify(1, 8, 256, 10.0)
        self.assertEqual(len(policy._clock[(1, 8, 256)]), 16)
        self.assertAlmostEqual(predict_verify_ms(1, 8, 256), 10.0)

    def test_tau_mass_uses_the_locked_k(self):
        weight = torch.tensor([[0.5, 0.4, 0.3, 0.2, 0.05, 0.04, 0.03, 0.02]])
        state = SimpleNamespace(
            depth=1,
            weight=weight,
            prev_e=torch.tensor(1.4),
            tokens=torch.zeros(1, 8, dtype=torch.long),
            anc_mask=torch.eye(8, dtype=torch.int8),
            config=SimpleNamespace(width=4, tau=0.05, max_depth=8),
        )
        self.assertFalse(bool(depth_rule(state)))
        self.assertTrue(bool(depth_rule(state, 8)))
        state.decider = RoundPlan(
            width=4,
            k=8,
            use_tau=True,
            locked_k=8,
            theta_value=None,
            layer_ms=None,
            max_ell=4,
            batch=1,
            buckets=(256,),
        )
        step = step_after_layer(state)
        self.assertFalse(step.is_final)
        self.assertAlmostEqual(float(state.prev_e), float(acceptance_mass(state, 8)))

    def test_finish_drops_the_request_totals(self):
        reset_policy()
        observe_accept("z", 2.0, 2.0)
        note_verified("z", 4, 1.0)
        ClearLedger(active_rids=["z"])._apply_gap("finish", ["z"])
        self.assertNotIn("z", policy._sum_a)
        self.assertNotIn("z", policy._pending)
        self.assertGreater(policy._total_a, 0.0)

    def test_draft_tp_above_one_is_rejected(self):
        args = SimpleNamespace(
            clear_tree_role="draft",
            clear_tree_width=4,
            clear_tree_tau=0.4,
            clear_tree_max_depth=8,
            speculative_num_draft_tokens=None,
            speculative_num_steps=8,
            speculative_verify_mode=None,
            speculative_eagle_topk=None,
            max_running_requests=1,
            page_size=1,
            disable_overlap_schedule=False,
            enable_mixed_chunk=False,
            tp_size=2,
        )
        with self.assertRaises(ValueError):
            apply_clear_tree_settings(args, on_npu=False)
        args.tp_size = 1
        apply_clear_tree_settings(args, on_npu=False)

    def test_layer_width_follows_the_candidate_list(self):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
            install_candidates,
            layer_width_supported,
        )

        self.assertTrue(layer_width_supported(16))
        self.assertFalse(layer_width_supported(7))
        install_candidates([4, 8], [8, 16])
        self.addCleanup(lambda: install_candidates(None, None))
        self.assertTrue(layer_width_supported(4))
        self.assertFalse(layer_width_supported(16))

    def test_frozen_rows_skip_unforwarded_requests_and_missing_widths(self):
        deep = SimpleNamespace(depth=3, prefix=10)
        shallow = SimpleNamespace(depth=2, prefix=4)

        def predict(_scheduler, width, _active, _sum_ell, depth=None):
            if int(width) != 8:
                return None
            return 7.0

        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            side_effect=predict,
        ):
            rows = _freeze_layer_rows(SimpleNamespace(), [deep, shallow])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["active"], 2)
        self.assertEqual(rows[0]["sum_ell"], 14)
        self.assertEqual(rows[1]["active"], 1)
        self.assertEqual(rows[1]["sum_ell"], 10)
        self.assertEqual(rows[0]["pred"][8], 7.0)
        self.assertIsNone(rows[0]["pred"][4])
        self.assertIn(16, rows[0]["pred"])
        self.assertIsNone(rows[0]["pred"][16])

    def test_tau_round_stays_out_of_replay(self):
        reset_policy()
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        self.addCleanup(lambda: os.environ.pop("SGLANG_CLEAR_LEDGER", None))
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        for k in (2, 4, 8, 16, 32, 64):
            observe_verify(1, k, 256, 10.0)
            observe_verify(1, k, 256, 10.0)
        observe_price(1, 2.0, 1.0)
        policy._rounds = 11
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=None,
        ):
            plan = make_round(SimpleNamespace(), [req])
        self.assertTrue(plan.use_tau)
        self.assertFalse(plan.replay_eligible)
        self.assertEqual(plan.locked_k, 16)
        self.assertEqual(plan.width, 4)
        self.assertEqual(plan.replay_kind, "probe_w")

    def test_foreign_verify_bucket_warns_once(self):
        reset_policy()
        with self.assertLogs(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_policy",
            level="WARNING",
        ) as logs:
            ingest_verify_samples([[1, 8, 99999, 10000], [1, 8, 99999, 11000]])
            ingest_verify_samples([[1, 8, 256, 10000]])
        foreign = [line for line in logs.output if "99999" in line]
        self.assertEqual(len(foreign), 1)
        self.assertEqual(policy._clock, {})

    def test_probe_round_cap_stays_on_the_flat_k(self):
        reset_policy()
        policy._probe_rounds = 54
        policy._rounds = 20
        for k in (2, 4, 8, 16, 32, 64):
            observe_verify(1, k, 256, 10.0)
            observe_verify(1, k, 256, 10.0)
        observe_price(1, 2.0, 1.0)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.ledger_enabled",
            return_value=True,
        ), patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            plan = make_round(SimpleNamespace(), [req])
        self.assertFalse(plan.use_tau)
        self.assertIsNone(plan.locked_k)
        self.assertEqual(plan.replay_kind, "probe_w")
        self.assertFalse(plan.replay_eligible)
        self.assertEqual(policy._probe_rounds, 54)

        reset_policy()
        policy._probe_rounds = 54
        policy._rounds = 20
        for k in (2, 4, 8, 16, 32):
            observe_verify(1, k, 256, 10.0)
            observe_verify(1, k, 256, 10.0)
        observe_price(1, 2.0, 1.0)
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.ledger_enabled",
            return_value=True,
        ), patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            short = make_round(SimpleNamespace(), [req])
        self.assertFalse(short.use_tau)
        self.assertIsNone(short.locked_k)
        self.assertFalse(short.replay_eligible)
        self.assertEqual(policy._probe_rounds, 54)

    def test_replay_k_stays_inside_retained_nodes(self):
        self.assertEqual(
            _select_replay_k(
                [[0.4, 0.3]], [1.0], 1.0, {32: 1.0}, 32, cap=2, flat_k_value=16
            ),
            16,
        )
        self.assertEqual(
            _select_replay_k(
                [[0.4, 0.3]],
                [1.0],
                1.0,
                {2: 1.0, 32: 5.0},
                32,
                cap=2,
                flat_k_value=16,
            ),
            2,
        )

    def test_layer_prefix_ridge_stays_inside_the_seen_span(self):
        model = ForgettingRegressor(
            2, LAYER_MIN_SAMPLES, ridge=LAYER_RIDGE, span_feature=1
        )
        for prefix, cost in (
            (900.0, 5.8),
            (900.0, 6.2),
            (903.0, 5.7),
            (903.0, 6.3),
        ):
            model.update([1.0, prefix], cost)
        self.assertLess(abs(model.coefficients()[1]), 1e-3)
        ledger = ClearLedger()
        ledger.layer_models[(4, "graph", 1)] = model
        sched = SimpleNamespace(_clear_ledger=ledger, tp_rank=0)
        near = predict_layer_ms(sched, 4, 1, 1000)
        self.assertIsNotNone(near)
        self.assertGreater(near, 0.0)
        self.assertLess(abs(near - 6.0), 1.0)
        self.assertIsNone(predict_layer_ms(sched, 4, 1, 2500))

        negative = ForgettingRegressor(
            2, LAYER_MIN_SAMPLES, ridge=LAYER_RIDGE, span_feature=1
        )
        for prefix in (900.0, 900.0, 903.0, 903.0):
            negative.update([1.0, prefix], -1.0)
        ledger.layer_models[(8, "graph", 1)] = negative
        self.assertIsNone(predict_layer_ms(sched, 8, 1, 901))

    def test_layer_cells_keep_their_own_slope_and_span(self):
        def fit(pairs):
            model = ForgettingRegressor(
                2, LAYER_MIN_SAMPLES, ridge=LAYER_RIDGE, span_feature=1
            )
            for prefix, cost in pairs:
                model.update([1.0, float(prefix)], float(cost))
            return model

        small = [(200.0, 4.2), (400.0, 6.0), (600.0, 8.1), (800.0, 10.0)]
        large = [(200.0, 14.0), (400.0, 24.2), (600.0, 34.0), (800.0, 44.1)]
        wide = [(4000.0, 14.0), (5000.0, 15.0), (6000.0, 16.0), (7000.0, 17.0)]
        ledger = ClearLedger()
        ledger.layer_models[(8, "graph", 1)] = fit(small)
        ledger.layer_models[(8, "graph", 4)] = fit(large)
        sched = SimpleNamespace(_clear_ledger=ledger, tp_rank=0)
        low = predict_layer_ms(sched, 8, 1, 500)
        high = predict_layer_ms(sched, 8, 3, 500)
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertLess(abs(low - 7.0), 1.0)
        self.assertGreater(high, low + 10.0)
        slope_low = layer_sum_ell_slope(sched, 8, 1)
        slope_high = layer_sum_ell_slope(sched, 8, 3)
        self.assertIsNotNone(slope_low)
        self.assertIsNotNone(slope_high)
        self.assertGreater(slope_high, slope_low + 0.02)
        self.assertIsNone(predict_layer_ms(sched, 8, 2, 500))

        ledger.layer_models[(8, "graph", 4)] = fit(wide)
        self.assertIsNotNone(predict_layer_ms(sched, 8, 3, 2000))
        self.assertIsNone(predict_layer_ms(sched, 8, 1, 2000))
        self.assertLess(abs(predict_layer_ms(sched, 8, 1, 500) - low), 1e-6)

    def test_layer_graph_bs_uses_the_captured_list(self):
        padded = SimpleNamespace(_clear_layer_capture_bs=(1, 2, 4))
        exact = SimpleNamespace(_clear_layer_capture_bs=(1, 2, 3, 4))
        self.assertEqual(layer_graph_bs(padded, 3), 4)
        self.assertEqual(layer_graph_bs(exact, 3), 3)
        self.assertEqual(layer_graph_bs(SimpleNamespace(), 3), 4)
        strict = SimpleNamespace(
            _clear_layer_capture_bs=(1, 2, 4),
            _clear_layer_disable_padding=True,
        )
        self.assertEqual(layer_graph_bs(strict, 2), 2)
        self.assertEqual(layer_graph_bs(strict, 3), 4)

    def test_keep_uses_each_count_its_own_order(self):
        long = {
            "e": {8: 0.0},
            "g": {8: 10.0},
            "ell": 100,
            "c": 1.0,
            "item": "long",
            "growing": True,
        }
        shorts = [
            {
                "e": {8: 0.0},
                "g": {8: 3.0},
                "ell": 1,
                "c": 1.0,
                "item": name,
                "growing": True,
            }
            for name in ("b", "c", "d", "e")
        ]
        requests = [long, *shorts]

        def kappa(m):
            return 0.0 if int(m) < 4 else 100.0

        one = keep_by_net_benefit(
            requests,
            theta=1.0,
            verify_times={8: 1.0},
            layer_cost=lambda m, _ell: 0.0 if int(m) == 1 else None,
            kappa=kappa,
        )
        four = keep_by_net_benefit(
            requests,
            theta=1.0,
            verify_times={8: 1.0},
            layer_cost=lambda m, _ell: 0.0 if int(m) == 4 else None,
            kappa=kappa,
        )
        self.assertEqual(one, ["long"])
        self.assertEqual(four, ["b", "c", "d", "e"])

    def test_narrow_probe_counts_samples_at_this_graph_batch(self):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_replay import (
            PROBE_W_IDLE,
            _narrow_idle,
            _probe_w_fill,
        )

        reset_policy()
        ledger = ClearLedger()
        other = ForgettingRegressor(
            2, LAYER_MIN_SAMPLES, ridge=LAYER_RIDGE, span_feature=1
        )
        other.n = LAYER_MIN_SAMPLES
        ledger.layer_models[(4, "graph", 4)] = other
        sched = SimpleNamespace(
            _clear_ledger=ledger,
            _clear_layer_capture_bs=(1, 2, 4),
        )
        _narrow_idle[(1, 4)] = PROBE_W_IDLE
        self.assertEqual(
            choose_build_width(1, ledger_on=True, eligible=True, scheduler=sched),
            (4, "probe_w"),
        )
        self.assertTrue(_probe_w_fill[(1, 4)])
        ready = ForgettingRegressor(
            2, LAYER_MIN_SAMPLES, ridge=LAYER_RIDGE, span_feature=1
        )
        ready.n = LAYER_MIN_SAMPLES
        ledger.layer_models[(4, "graph", 1)] = ready
        self.assertEqual(
            choose_build_width(1, ledger_on=True, eligible=True, scheduler=sched),
            (8, "ordinary"),
        )

    def test_probe_w_stops_once_four_layer_samples_exist(self):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_replay import (
            PROBE_W_IDLE,
            _narrow_idle,
        )

        reset_policy()
        ledger = ClearLedger()
        sched = SimpleNamespace(_clear_ledger=ledger)
        self.assertEqual(
            choose_build_width(1, ledger_on=True, eligible=True, scheduler=sched),
            (4, "probe_w"),
        )
        filled = ForgettingRegressor(
            2, LAYER_MIN_SAMPLES, ridge=LAYER_RIDGE, span_feature=1
        )
        filled.n = LAYER_MIN_SAMPLES
        ledger.layer_models[(4, "graph", 1)] = filled
        self.assertEqual(
            choose_build_width(1, ledger_on=True, eligible=True, scheduler=sched),
            (8, "ordinary"),
        )
        _narrow_idle[(1, 4)] = PROBE_W_IDLE
        again = [
            choose_build_width(1, ledger_on=True, eligible=True, scheduler=sched)
            for _ in range(3)
        ]
        self.assertEqual(again, [(4, "probe_w"), (4, "probe_w"), (8, "ordinary")])

    def test_narrow_width_probes_early_when_the_prefix_is_unpriced(self):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_replay import (
            _narrow_idle,
        )

        def prepared(idle):
            reset_policy()
            ledger = ClearLedger()
            filled = ForgettingRegressor(
                2, LAYER_MIN_SAMPLES, ridge=LAYER_RIDGE, span_feature=1
            )
            filled.n = LAYER_MIN_SAMPLES
            ledger.layer_models[(4, "graph", 1)] = filled
            _narrow_idle[(1, 4)] = idle
            return SimpleNamespace(_clear_ledger=ledger)

        due = prepared(policy.EXPLORE_EVERY)
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=None,
        ):
            self.assertEqual(
                choose_build_width(
                    1, ledger_on=True, eligible=True, scheduler=due, sum_ell=5000
                ),
                (4, "probe_w"),
            )
        early = prepared(policy.EXPLORE_EVERY - 1)
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=None,
        ):
            self.assertEqual(
                choose_build_width(
                    1, ledger_on=True, eligible=True, scheduler=early, sum_ell=5000
                ),
                (8, "ordinary"),
            )
        covered = prepared(policy.EXPLORE_EVERY)
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            self.assertEqual(
                choose_build_width(
                    1,
                    ledger_on=True,
                    eligible=True,
                    scheduler=covered,
                    sum_ell=5000,
                ),
                (8, "ordinary"),
            )

    def test_external_event_clears_fits_and_keeps_width_and_theta(self):
        reset_policy()
        assign_persistent_width(1, 16)
        observe_price(1, 4.0, 2.0)
        note_verified("kept", 10, 2.0)
        settle_verified(["kept"], [13])
        kept = accept_ratio("kept")
        note_verified("pending", 10, 1.0, 3)
        observe_verify(1, 8, 256, 10.0)
        from sglang.srt.speculative.clear_tree_remote import clear_tree_wpair as wpair

        wpair.superset_due(1, 64, None)
        self.assertGreater(wpair._blocks[1]["virt"], 0.0)
        ledger = ClearLedger()
        fitted = ForgettingRegressor(
            2, LAYER_MIN_SAMPLES, ridge=LAYER_RIDGE, span_feature=1
        )
        fitted.n = 4
        ledger.layer_models[(8, "graph", 1)] = fitted
        ledger.verify_model = ForgettingRegressor(4, 6)
        ledger.verify_samples = 3
        ledger.open = object()
        sched = SimpleNamespace(
            _clear_ledger=ledger,
            _clear_replay_hold={"tree": 1},
            _clear_replay_closed={"excluded": False},
            _clear_replay_pending={"old": True},
            _clear_replay_to_hold=[object()],
        )
        price = theta(1)
        on_external_event(sched)
        self.assertEqual(persistent_width(1), 16)
        self.assertAlmostEqual(theta(1), price)
        self.assertAlmostEqual(accept_ratio("kept"), kept)
        self.assertEqual(policy._pending, {})
        self.assertEqual(policy._clock, {})
        self.assertEqual(policy._probe_rounds, 0)
        self.assertEqual(ledger.layer_models, {})
        self.assertIsNone(ledger.verify_model)
        self.assertEqual(ledger.verify_samples, 0)
        self.assertIsNone(ledger.open)
        self.assertEqual(sched._clear_replay_hold, {"tree": 1})
        self.assertEqual(sched._clear_replay_closed, {"excluded": False})
        self.assertEqual(sched._clear_replay_pending, {"old": True})
        self.assertFalse(sched._clear_replay_to_hold)
        self.assertEqual(wpair._blocks, {})

    def test_target_gap_skips_price_accept_and_layer_fit(self):
        reset_policy()
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [100], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        ledger.mark_post(1.2)
        ledger.note_idle()
        ledger.queue_layer(
            {
                "layer": 0,
                "active": 1,
                "graph_bucket": 1,
                "w": 8,
                "sum_ell": 100,
                "host_us": 1000,
                "gpu_us": 4000,
                "path": "graph",
                "post_reply": False,
            }
        )
        note_verified("a", 100, 1.5, 3)
        before = theta(1)
        jumped = _step(ledger, ["a"], [110], 2, 1.5)
        self.assertIn("target_gap", jumped["exclude_reason"])
        self.assertTrue(jumped["excluded"])
        self.assertFalse(jumped["price_kept"])
        self.assertEqual(theta(1), before)
        self.assertEqual(ledger.layer_models[(8, "graph", 1)].n, 0)
        self.assertNotIn("a", policy._sum_a)
        self.assertEqual(policy._pending, {})

        ledger.mark_stop(2.0)
        ledger.mark_send(2.1)
        ledger.mark_post(2.2)
        ledger.note_idle()
        note_verified("a", 110, 1.5, 3)
        kept = _step(ledger, ["a"], [114], 4, 2.5)
        self.assertNotIn("target_gap", kept["exclude_reason"])
        self.assertTrue(kept["price_kept"])
        self.assertAlmostEqual(policy._sum_a["a"], 3.0)

    def test_prefix_jump_without_depth_is_kept(self):
        reset_policy()
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [100], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        ledger.mark_post(1.2)
        ledger.note_idle()
        note_verified("a", 100, 1.5)
        record = _step(ledger, ["a"], [200], 2, 1.5)
        self.assertNotIn("target_gap", record["exclude_reason"])
        self.assertTrue(record["price_kept"])
        self.assertAlmostEqual(policy._sum_a["a"], 99.0)

    def test_close_writes_draft_t_k_into_the_verify_table(self):
        reset_policy()
        ledger = ClearLedger(active_rids=["a", "b", "c", "d"])
        _step(ledger, ["a", "b", "c", "d"], [100, 100, 100, 100], 1, 0.0)
        ledger.open.k = 16
        ledger.mark_stop(1.0)
        ledger.mark_send(1.02)
        ledger.mark_post(1.03)
        ledger.note_idle()
        record = _step(ledger, ["a", "b", "c", "d"], [100, 100, 100, 100], 2, 1.08)
        bucket = smallest_bucket(100 + 16 + 1, runtime_kv_buckets())
        self.assertEqual(record["t_k_us"], 80000)
        self.assertFalse(record["excluded"])
        self.assertIsNone(record["hold_ms"])
        self.assertEqual(policy._clock[(4, 16, bucket)], [80.0])
        self.assertEqual(policy._clock_ell[(4, 16, bucket)], [100])

        ledger.open.k = 16
        ledger.mark_stop(2.0)
        ledger.mark_send(2.02)
        ledger.mark_post(2.03)
        second = _step(ledger, ["a", "b", "c", "d"], [100, 100, 100, 100], 3, 2.08)
        self.assertFalse(second["excluded"])
        self.assertEqual(policy._clock[(4, 16, bucket)], [80.0, 80.0])
        self.assertEqual(policy._clock_ell[(4, 16, bucket)], [100, 100])

        idle = ClearLedger(active_rids=[])
        _step(idle, ["a", "b", "c", "d"], [100, 100, 100, 100], 4, 3.0)
        idle.open.k = 16
        idle.mark_stop(3.0)
        idle.mark_send(3.0)
        idle.mark_post(3.01)
        idle.note_idle()
        skipped = _step(idle, ["a", "b", "c", "d"], [100, 100, 100, 100], 5, 3.08)
        self.assertIn("idle_start", skipped["exclude_reason"])
        self.assertEqual(policy._clock[(4, 16, bucket)], [80.0, 80.0])

    def test_layer_wall_uses_the_gap_between_forwards(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [16], 1, 0.0)
        samples = []
        for t0, layer in ((0.0, 0), (0.012, 1), (0.025, 2)):
            sample = {
                "layer": layer,
                "active": 1,
                "graph_bucket": 1,
                "w": 8,
                "sum_ell": 16,
                "host_us": 1,
                "gpu_us": 1,
                "path": "graph",
                "post_reply": False,
                "t0": t0,
            }
            ledger.queue_layer(sample)
            samples.append(sample)
        ledger.mark_stop(0.040)
        ledger.mark_send(0.040)
        ledger.mark_post(0.041)
        ledger.note_idle()
        record = _step(ledger, ["a"], [16], 2, 0.050)
        self.assertEqual([item["wall_us"] for item in samples], [12000, 13000, 15000])
        model = ledger.layer_models[(8, "graph", 1)]
        self.assertEqual(model.n, 3)
        expected = ((12.0 * FORGET + 13.0) * FORGET) + 15.0
        self.assertAlmostEqual(model.xty[0], expected)
        self.assertEqual(record["seed_ms"], 0.0)
        self.assertAlmostEqual(record["layer_wall_ms"], 40.0)

    def test_post_reply_layer_does_not_fit(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [16], 1, 0.0)
        ledger.queue_layer(
            {
                "layer": 0,
                "active": 1,
                "graph_bucket": 1,
                "w": 8,
                "sum_ell": 16,
                "host_us": 1000,
                "gpu_us": 1000,
                "path": "graph",
                "post_reply": False,
            }
        )
        ledger.queue_layer(
            {
                "layer": 1,
                "active": 1,
                "graph_bucket": 1,
                "w": 8,
                "sum_ell": 16,
                "host_us": 9000,
                "gpu_us": 9000,
                "path": "graph",
                "post_reply": True,
                "t0": 1.0,
            }
        )
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        ledger.mark_post(1.2)
        ledger.note_idle()
        _step(ledger, ["a"], [16], 2, 1.4)
        self.assertEqual(ledger.layer_models[(8, "graph", 1)].n, 1)

    def test_post_reply_sample_does_not_count_for_prefit(self):
        from sglang.srt.speculative.clear_tree_remote import clear_tree_fit as fit
        from sglang.srt.speculative.clear_tree_remote import clear_tree_prefit as prefit

        plan = prefit.build_plan(prefit.PlanInputs((1,), (8,), (8,), (256,), 8))
        state = fit.PrefitState("abcd12341", plan)
        scheduler = SimpleNamespace(_clear_prefit=state)
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [16], 1, 0.0)
        ledger.open.prefit = {"phase": "keep", "group": "w", "run": "abcd12341"}
        ledger.queue_layer(
            {
                "layer": 0,
                "active": 1,
                "graph_bucket": 1,
                "w": 8,
                "sum_ell": 16,
                "host_us": 1000,
                "gpu_us": 1000,
                "path": "graph",
                "post_reply": False,
            }
        )
        ledger.queue_layer(
            {
                "layer": 1,
                "active": 1,
                "graph_bucket": 1,
                "w": 8,
                "sum_ell": 16,
                "host_us": 9000,
                "gpu_us": 9000,
                "path": "graph",
                "post_reply": True,
                "t0": 1.0,
            }
        )
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        ledger.mark_post(1.2)
        ledger.note_idle()
        ledger.on_packet(
            "step",
            ["a"],
            "sess",
            2,
            [16],
            8,
            now=1.4,
            scheduler=scheduler,
        )
        self.assertEqual(state.layers[(8, "graph", 1)].n, 1)
        self.assertEqual(getattr(ledger.layer_models.get((8, "graph", 1)), "n", 0), 0)

    def test_mark_stop_keeps_the_first_time(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [8], 1, 0.0)
        ledger.mark_stop(1.5)
        ledger.mark_stop(9.0)
        self.assertEqual(ledger.open.t_stop, 1.5)

    def test_fallback_round_stays_out_of_the_fits(self):
        saved = os.environ.get("SGLANG_CLEAR_LEDGER")
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"

        def _restore_ledger_flag():
            if saved is None:
                os.environ.pop("SGLANG_CLEAR_LEDGER", None)
            else:
                os.environ["SGLANG_CLEAR_LEDGER"] = saved

        self.addCleanup(_restore_ledger_flag)
        ledger = ClearLedger(active_rids=["a"])
        sched = SimpleNamespace(
            _clear_ledger=ledger,
            _clear_build_token=1,
            _clear_target_prefix={},
        )
        _step(ledger, ["a"], [32], 1, 1.0)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 4, output_ids=[])
        install_fallback_plan(sched, [req])
        self.assertTrue(ledger.open.fallback)
        ledger.open.k = 16
        ledger.queue_layer(
            {
                "layer": 0,
                "active": 1,
                "graph_bucket": 1,
                "w": 8,
                "sum_ell": 32,
                "host_us": 1000,
                "gpu_us": 1000,
                "path": "graph",
                "post_reply": False,
            }
        )
        ledger.mark_stop(1.0)
        ledger.mark_send(1.0)
        ledger.mark_post(1.01)
        ledger.note_idle()
        note_verified("a", 32, 5.0, 2)
        record = _step(ledger, ["a"], [34], 2, 1.08)
        self.assertIn("fallback", record["exclude_reason"])
        self.assertFalse(record["price_kept"])
        self.assertEqual(policy._clock, {})
        self.assertEqual(ledger.layer_models[(8, "graph", 1)].n, 0)
        self.assertNotIn("a", policy._sum_a)
        self.assertEqual(policy._pending, {})

    def test_stop_rule_keeps_the_same_request_and_refits(self):
        good = _rows_state([[0.5, 0, 0, 0, 0, 0, 0, 0]], "good", 1)
        poor = _rows_state([[0.05, 0, 0, 0, 0, 0, 0, 0]], "poor", 100)
        kept = select_continuing(
            [good, poor],
            [good, poor],
            theta=1.0,
            verify_times={8: 1.0, 16: 1.0, 32: 1.0},
            layer_cost=lambda active, _ell: 0.05 if int(active) == 1 else 0.5,
            kappa=0.1,
        )
        self.assertEqual([item.rid for item in kept], ["good"])
        reg = ForgettingRegressor(2, 1)
        reg.update([1.0, 0.0], 1.0)
        first = list(reg.coefficients())
        self.assertEqual(reg.coefficients(), first)
        reg.update([1.0, 1.0], 3.0)
        self.assertNotEqual(reg.coefficients(), first)

    def test_gain_ratio_shrinks_the_upper_bound(self):
        self.assertEqual(gain_ratio(), 1.0)
        policy._gain_pred = 10.0
        policy._gain_actual = 6.0
        self.assertAlmostEqual(gain_ratio(), (6.0 + 3.0) / (10.0 + 3.0))
        state = _rows_state([[0.4, 0.2, 0.1, 0.0]], "r", 4)
        raw = _raw_marginal(state, 2)
        self.assertAlmostEqual(_marginal_gain(state, 2), raw * gain_ratio())
        policy._gain_pred = 10.0
        policy._gain_actual = 20.0
        self.assertEqual(gain_ratio(), 1.0)
        on_external_event(SimpleNamespace())
        self.assertEqual(policy._gain_pred, 10.0)
        self.assertEqual(policy._gain_actual, 20.0)
        self.assertEqual(gain_ratio(), 1.0)


class TestPacketWidthPair(CustomTestCase):
    def setUp(self):
        reset_policy()
        environment = patch.dict(os.environ, {"SGLANG_CLEAR_LEDGER": "1"})
        environment.start()
        self.addCleanup(environment.stop)
        self.ledger = ClearLedger(active_rids=["a"])
        self.scheduler = SimpleNamespace(
            _clear_ledger=self.ledger,
            _clear_build_token=1,
            _clear_build_serial=1,
            _clear_replay_packet=("sess", 1),
            server_args=SimpleNamespace(clear_tree_width=8, clear_tree_max_depth=8),
            tp_rank=0,
            tp_size=1,
        )
        self.req = SimpleNamespace(rid="a", output_ids=[11])
        self.state = SimpleNamespace(
            depth=1,
            tokens=torch.arange(1, 9).reshape(1, 8),
            parents=torch.arange(8).reshape(1, 8),
            weight=torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.01, 0.01, 0.01, 0.01]]),
            prefix=4,
            rid="a",
        )
        self.plan = _value_plan()
        self.plan.replay_kind = "ordinary"
        self.plan.replay_eligible = True
        self.plan._frozen_k = 8
        self.plan._time = lambda _k: 5.0

    def test_post_reply_freezes_then_logs_d_wpair(self):
        scheduler = self.scheduler
        with patch.object(replay, "_layers_of", wraps=replay._layers_of) as freeze:
            replay.queue_hold_replay(scheduler, [self.req], [self.state], self.plan)
            freeze.assert_not_called()
        record = {"decision": "stay", "target": 8, "bucket": 1}
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_wpair.observe_pair",
            return_value=record,
        ) as observe, patch.object(ledger_module, "_emit") as emit:
            replay.run_post_reply(scheduler)
            replay.run_post_reply(scheduler)
        observe.assert_called_once()
        frozen = observe.call_args.args[0]
        self.assertEqual(frozen["width"], 8)
        self.assertTrue(frozen["eligible"])
        self.assertNotIn("output_len", frozen["trees"]["a"])
        self.assertIsNone(getattr(scheduler, "_clear_replay_hold", None))
        self.assertIn("d_wpair", [call.args[0] for call in emit.call_args_list])
        self.assertFalse(scheduler._clear_replay_to_hold)

    def test_ineligible_fallback_and_switched_width_do_not_observe(self):
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_wpair.observe_pair"
        ) as observe:
            self.plan.replay_eligible = False
            replay.queue_hold_replay(
                self.scheduler, [self.req], [self.state], self.plan
            )
            replay.run_post_reply(self.scheduler)
            observe.assert_not_called()

            self.scheduler._clear_replay_post_done = None
            self.plan.replay_eligible = True
            self.scheduler._clear_plan_fallback = True
            replay.queue_hold_replay(
                self.scheduler, [self.req], [self.state], self.plan
            )
            replay.run_post_reply(self.scheduler)
            observe.assert_not_called()

            self.scheduler._clear_replay_post_done = None
            self.scheduler._clear_plan_fallback = False
            replay.queue_hold_replay(
                self.scheduler, [self.req], [self.state], self.plan
            )
            assign_persistent_width(1, 16)
            replay.run_post_reply(self.scheduler)
            observe.assert_not_called()


def _rows_state(rows, rid, prefix):
    weight = torch.tensor(rows, dtype=torch.float32)
    return SimpleNamespace(
        depth=int(weight.shape[0]),
        weight=weight,
        rid=rid,
        prefix=int(prefix),
        config=SimpleNamespace(max_depth=8, tau=0.4, width=int(weight.shape[1])),
    )


def _score_state(scores):
    weight = torch.zeros(1, 8)
    weight[0, : len(scores)] = torch.tensor(scores, dtype=torch.float32)
    return SimpleNamespace(
        depth=1,
        weight=weight,
        rid="r",
        prefix=4,
        config=SimpleNamespace(max_depth=8, tau=0.4, width=8),
    )


def _value_plan():
    return RoundPlan(
        width=8,
        k=8,
        use_tau=False,
        locked_k=None,
        theta_value=1.0,
        layer_ms=0.0,
        max_ell=10,
        batch=1,
        buckets=(256, 512, 1024, 2048, 4096),
    )


class TestHoldAndPrefitExclusions(CustomTestCase):
    def setUp(self):
        reset_policy()

    def _close(self, ledger, rpc, now, **kwargs):
        return ledger.on_packet(
            "step",
            ["a"],
            "sess",
            rpc,
            [32],
            8,
            now=now,
            **kwargs,
        )

    def test_matching_hold_splits_h_and_q(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [32], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        record = self._close(
            ledger,
            2,
            1.4,
            hold_ns=100_000_000,
            hold_rpc_seq=1,
            target_queued=True,
            step_bytes=40,
            reply_bytes=80,
            verify={
                "rpc_seq": 1,
                "fallback": False,
                "b": 1,
                "k": 8,
                "bucket": 256,
            },
        )
        self.assertEqual(record["t_k_us"], 400000)
        self.assertAlmostEqual(record["hold_ms"], 100.0)
        self.assertAlmostEqual(record["q_ms"], 300.0)
        self.assertTrue(record["target_queued"])
        self.assertEqual(record["step_bytes"], 40)
        self.assertEqual(record["reply_bytes"], 80)
        self.assertEqual(policy._clock[(1, 8, 256)], [100.0])
        self.assertTrue(policy._q_raw[(1, 8)])

    def test_hold_above_t_k_is_dropped(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [32], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        record = self._close(ledger, 2, 1.4, hold_ns=500_000_000, hold_rpc_seq=1)
        self.assertIsNone(record["hold_ms"])
        self.assertEqual(ledger.hold_invalid, 1)
        self.assertEqual(policy._clock, {})

    def test_mismatched_rpc_does_not_write_hold(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [32], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        record = self._close(ledger, 2, 1.4, hold_ns=100_000_000, hold_rpc_seq=99)
        self.assertIsNone(record["hold_ms"])
        self.assertEqual(ledger.hold_invalid, 0)
        self.assertEqual(policy._clock, {})

    def test_draft_floor_still_writes_q(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [32], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        record = self._close(
            ledger,
            2,
            1.4,
            recv_queued=True,
            hold_ns=100_000_000,
            hold_rpc_seq=1,
            verify={
                "rpc_seq": 1,
                "fallback": False,
                "b": 1,
                "k": 8,
                "bucket": 256,
            },
        )
        self.assertTrue(record["draft_floor"])
        self.assertFalse(record["excluded"])
        self.assertEqual(policy._clock[(1, 8, 256)], [100.0])
        self.assertTrue(policy._q_raw[(1, 8)])

    def test_prefit_rounds_skip_theta_and_accepts(self):
        for phase, group, reason in (
            ("warmup", "warm", "warmup"),
            ("mixed", "mixed", "prefit_mixed"),
            ("keep", "v", "prefit_bucket"),
        ):
            with self.subTest(phase=phase):
                reset_policy()
                ledger = ClearLedger(active_rids=["a"])
                _step(ledger, ["a"], [32], 1, 1.0)
                ledger.open.prefit = {
                    "phase": phase,
                    "group": group,
                    "bucket": 512,
                }
                ledger.mark_stop(1.0)
                ledger.mark_send(1.1)
                note_verified("a", 32, 1.0, 1)
                before = theta(1)
                record = self._close(ledger, 2, 1.4)
                self.assertIn(reason, record["exclude_reason"])
                self.assertFalse(record["price_kept"])
                self.assertEqual(theta(1), before)
                self.assertEqual(policy._pending, {})
                self.assertEqual(policy._clock, {})

    def test_width_sweep_skips_hold_and_q(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [32], 1, 1.0)
        ledger.open.prefit = {"phase": "keep", "group": "w"}
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        record = self._close(ledger, 2, 1.4, hold_ns=100_000_000, hold_rpc_seq=1)
        self.assertNotIn("warmup", record["exclude_reason"])
        self.assertNotIn("prefit_bucket", record["exclude_reason"])
        self.assertFalse(record["price_kept"])
        self.assertEqual(policy._clock, {})
        self.assertEqual(policy._q_raw, {})

    def test_unseen_target_writes_t_k_on_the_first_round(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [32], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        record = self._close(ledger, 2, 1.4)
        self.assertIsNone(record["hold_ms"])
        self.assertFalse(ledger.split_capable)
        self.assertEqual(policy._clock[(1, 8, 256)], [400.0])

    def test_capable_target_does_not_store_t_k(self):
        ledger = ClearLedger(active_rids=["a"])
        ledger.split_capable = True
        _step(ledger, ["a"], [32], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        record = self._close(ledger, 2, 1.4)
        self.assertIsNone(record["hold_ms"])
        self.assertEqual(policy._clock, {})
        self.assertTrue(ledger.split_capable)

    def test_over_bucket_writes_neither_h_nor_q(self):
        saved = os.environ.get("SGLANG_NPU_TREE_GRAPH_KV_BUCKETS")
        os.environ["SGLANG_NPU_TREE_GRAPH_KV_BUCKETS"] = "256"
        try:
            ledger = ClearLedger(active_rids=["a"])
            ledger.split_capable = True
            _step(ledger, ["a"], [300], 1, 1.0)
            ledger.mark_stop(1.0)
            ledger.mark_send(1.1)
            record = self._close(
                ledger, 2, 1.4, hold_ns=100_000_000, hold_rpc_seq=1
            )
            self.assertAlmostEqual(record["q_ms"], 300.0)
            self.assertEqual(policy._clock, {})
            self.assertEqual(policy._q_raw, {})
        finally:
            if saved is None:
                os.environ.pop("SGLANG_NPU_TREE_GRAPH_KV_BUCKETS", None)
            else:
                os.environ["SGLANG_NPU_TREE_GRAPH_KV_BUCKETS"] = saved

    def test_target_fallback_skips_h_and_q(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [32], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        record = self._close(
            ledger,
            2,
            1.4,
            hold_ns=100_000_000,
            hold_rpc_seq=1,
            verify={"rpc_seq": 1, "fallback": True, "b": 1, "k": 8, "bucket": 256},
        )
        self.assertIn("target_fallback", record["exclude_reason"])
        self.assertEqual(policy._clock, {})
        self.assertEqual(policy._q_raw, {})

    def test_target_shape_labels_h_and_q(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [32], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        record = self._close(
            ledger,
            2,
            1.4,
            hold_ns=100_000_000,
            hold_rpc_seq=1,
            verify={"rpc_seq": 1, "fallback": False, "b": 1, "k": 16, "bucket": 512},
        )
        self.assertEqual(record["k"], 8)
        self.assertEqual(policy._clock[(1, 16, 512)], [100.0])
        self.assertNotIn((1, 8, 256), policy._clock)
        self.assertTrue(policy._q_raw[(1, 16)])
        self.assertNotIn((1, 8), policy._q_raw)

    def test_excluded_gap_does_not_make_a_capable_target_legacy(self):
        saved = os.environ.get("SGLANG_CLEAR_LEDGER")
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        try:
            ledger = ClearLedger(active_rids=["a"])
            scheduler = SimpleNamespace(
                server_args=SimpleNamespace(clear_tree_width=8),
                tp_rank=0,
                tp_size=1,
                _clear_ledger=ledger,
                sr_server=None,
            )
            ledger_module.note_packet(
                scheduler,
                SimpleNamespace(
                    action="prefill",
                    reqs=[SimpleNamespace(rid="a", base_committed_len=32)],
                    session_id="sess",
                    rpc_seq=1,
                    target_info={"backend": "npu", "tp": 2},
                    target_residence_ns=None,
                    target_residence_rpc_seq=None,
                    target_queued=None,
                ),
            )
            self.assertTrue(ledger.split_capable)
            _step(ledger, ["a"], [32], 2, 1.0)
            ledger.mark_stop(1.0)
            ledger.mark_send(1.1)
            ledger.on_packet("prefill", ["extra"], "sess", 3, [8], 8, now=1.2)
            record = self._close(ledger, 4, 1.5)
            self.assertIn("prefill", record["exclude_reason"])
            self.assertTrue(ledger.split_capable)
            self.assertEqual(policy._clock, {})
            ledger.mark_stop(1.5)
            ledger.mark_send(1.6)
            record = self._close(ledger, 5, 1.9)
            self.assertEqual(record["exclude_reason"], [])
            self.assertEqual(policy._clock, {})
            self.assertTrue(ledger.split_capable)
            ledger.drop_open()
            self.assertFalse(ledger.split_capable)
            ledger_module.note_packet(
                scheduler,
                SimpleNamespace(
                    action="step",
                    reqs=[SimpleNamespace(rid="a", base_committed_len=32)],
                    session_id="sess",
                    rpc_seq=6,
                    target_info=None,
                    target_residence_ns=100_000_000,
                    target_residence_rpc_seq=5,
                    target_queued=None,
                ),
            )
            self.assertTrue(ledger.split_capable)
        finally:
            if saved is None:
                os.environ.pop("SGLANG_CLEAR_LEDGER", None)
            else:
                os.environ["SGLANG_CLEAR_LEDGER"] = saved


    def test_late_report_merges_by_the_original_rpc(self):
        ledger = ClearLedger(active_rids=["a"])
        ledger.split_capable = True
        _step(ledger, ["a"], [32], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        record = self._close(
            ledger, 2, 1.4, hold_ns=100_000_000, hold_rpc_seq=1
        )
        self.assertEqual(policy._clock, {})
        self.assertIn(("sess", 1), ledger._pending_rounds)
        self.assertEqual(ledger._pending_rounds[("sess", 1)]["ell"], 32)
        ledger.mark_stop(1.4)
        ledger.mark_send(1.5)
        self._close(
            ledger,
            3,
            1.8,
            hold_ns=80_000_000,
            hold_rpc_seq=2,
            verify={
                "rpc_seq": 1,
                "fallback": False,
                "path": "graph",
                "b": 1,
                "k": 8,
                "bucket": 256,
            },
        )
        self.assertEqual(policy._clock[(1, 8, 256)], [100.0])
        self.assertEqual(policy._clock_ell[(1, 8, 256)], [32])
        self.assertTrue(policy._q_raw[(1, 8)])
        self.assertNotIn(("sess", 1), ledger._pending_rounds)
        self.assertIn(("sess", 2), ledger._pending_rounds)
        self.assertNotIn("target_fallback", record["exclude_reason"])

    def test_eager_report_does_not_exclude_or_write(self):
        ledger = ClearLedger(active_rids=["a"])
        ledger.split_capable = True
        _step(ledger, ["a"], [32], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        record = self._close(
            ledger,
            2,
            1.4,
            hold_ns=100_000_000,
            hold_rpc_seq=1,
            verify={
                "rpc_seq": 1,
                "fallback": False,
                "path": "eager",
                "b": 1,
                "k": 8,
                "bucket": 256,
            },
        )
        self.assertNotIn("target_fallback", record["exclude_reason"])
        self.assertEqual(policy._clock, {})
        self.assertEqual(policy._q_raw, {})
        self.assertEqual(ledger._pending_rounds, {})

    def test_unmatched_report_does_not_label_the_closed_round(self):
        ledger = ClearLedger(active_rids=["a"])
        ledger.split_capable = True
        _step(ledger, ["a"], [32], 1, 1.0)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.1)
        self._close(
            ledger,
            2,
            1.4,
            hold_ns=100_000_000,
            hold_rpc_seq=1,
            verify={
                "rpc_seq": 99,
                "fallback": False,
                "path": "graph",
                "b": 1,
                "k": 64,
                "bucket": 512,
            },
        )
        self.assertEqual(policy._clock, {})
        self.assertIn(("sess", 1), ledger._pending_rounds)
        self.assertEqual(ledger._pending_reports[("sess", 99)]["k"], 64)


class TestRoundQuoteAndPolicySplit(CustomTestCase):
    def setUp(self):
        reset_policy()
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        self.addCleanup(lambda: os.environ.pop("SGLANG_CLEAR_LEDGER", None))

    def test_one_quote_covers_layers_and_the_final_k(self):
        state = _rows_state([[0.5, 0.25, 0.12, 0.08, 0.04, 0.02, 0.01, 0.005]], "live", 4)
        plan = _value_plan()
        plan.cover_k = 32
        plan._verify_quotes = {}
        plan._layer_cost = lambda _active, _ell: 0.01
        plan.bind_states([state])
        prices = {8: 1.0, 16: 1.0}
        calls = []

        def fake(_batch, k, _bucket, _ell=None):
            calls.append(int(k))
            return prices.get(int(k))

        with patch.object(policy, "predict_verify_ms", side_effect=fake):
            plan.begin_layer([state])
            grew = plan.should_grow(state)
            times = {
                int(k): float(spent)
                for k, spent in plan._verify_quotes.items()
                if spent is not None and int(k) <= int(plan.cover_k)
            }
            kept = select_continuing(
                [state],
                [state],
                theta=float(plan.theta_value),
                verify_times=times,
                layer_cost=plan._layer_cost,
                kappa=plan._kappa(),
                scores_for=plan._scores_for,
            )
            self.assertEqual(grew, state in list(kept or []))
            self.assertEqual(calls.count(8), 1)
            self.assertEqual(calls.count(16), 1)
            self.assertEqual(calls.count(32), 1)
            self.assertIsNone(plan._verify_quotes[32])
            self.assertNotIn(64, calls)
            quoted = list(calls)

            state.weight = torch.cat(
                [state.weight, torch.tensor([[0.2, 0, 0, 0, 0, 0, 0, 0]])], dim=0
            )
            state.depth = 2
            plan.begin_layer([state])
            plan.should_grow(state)
            keep = plan.verify_k(state)
            plan.note(state, keep)
            self.assertEqual(calls, quoted)
            _, ordered, _ = plan._scores_for(state)
            ratio = accept_ratio(str(state.rid))
            values = {
                int(k): ratio * float(sum(ordered[: int(k)]))
                - float(plan.theta_value) * float(spent)
                for k, spent in plan._verify_quotes.items()
                if spent is not None and int(k) <= int(plan.cover_k)
            }
            self.assertEqual(keep, 16)
            self.assertEqual(keep, policy.select_k(values))
            self.assertAlmostEqual(policy._pending["live"]["e"], float(sum(ordered[:keep])))

            prices[16] = 100.0
            self.assertEqual(plan.verify_k(state), keep)
            self.assertEqual(calls, quoted)
            self.assertEqual(plan._quoted(16), 1.0)
            width = int(state.weight.shape[1])
            state.tokens = torch.zeros(state.depth, width, dtype=torch.long)
            state.parents = torch.zeros(state.depth, width, dtype=torch.long)
            plan._frozen_k = keep
            frozen = replay.hold_replay(
                SimpleNamespace(),
                [SimpleNamespace(rid="live")],
                [state],
                plan,
            )
            self.assertEqual(frozen["k"], keep)
            self.assertEqual(frozen["verify_ms"][16], 100.0)
            self.assertEqual(frozen["verify_ms"][8], 1.0)
            fresh = _value_plan()
            fresh.cover_k = 32
            fresh._verify_quotes = {}
            fresh.bind_states([state])
            self.assertEqual(fresh.verify_k(state), 8)
            self.assertEqual(plan._quoted(16), 1.0)

    def test_make_round_keeps_the_readiness_quote_for_the_online_plan(self):
        policy._probe_rounds = 12
        observe_verify(1, 16, 256, 10.0)
        observe_price(1, 2.0, 1.0)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.ledger_enabled",
            return_value=True,
        ), patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            plan = make_round(SimpleNamespace(), [req])
        self.assertIsNone(plan.locked_k)
        self.assertIn(16, plan._verify_quotes)
        seen = []
        real = policy.predict_verify_ms

        def spy(batch, k, bucket, ell=None):
            seen.append(int(k))
            return real(batch, k, bucket, ell)

        state = _score_state([0.4, 0.2, 0.1, 0.05])
        plan.bind_states([state])
        with patch.object(policy, "predict_verify_ms", side_effect=spy):
            plan.begin_layer([state])
            plan.should_grow(state)
            plan.verify_k(state)
        self.assertNotIn(16, seen)
        self.assertEqual(seen.count(8), 1)
        self.assertEqual(len(seen), len(set(seen)))

    def test_locked_prefit_and_fallback_do_not_hang_a_quote_table(self):
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        seen = []
        real = policy.predict_verify_ms

        def spy(batch, k, bucket, ell=None):
            seen.append(int(k))
            return real(batch, k, bucket, ell)

        with patch.object(policy, "predict_verify_ms", side_effect=spy):
            os.environ["SGLANG_CLEAR_LOCK_K"] = "32"
            try:
                locked = make_round(SimpleNamespace(_clear_ledger=False), [req])
            finally:
                os.environ.pop("SGLANG_CLEAR_LOCK_K", None)
            self.assertTrue(locked.use_tau)
            self.assertIsNone(locked._verify_quotes)
            self.assertEqual(seen, [16])
            seen.clear()
            fallback = policy.make_fallback_round(
                SimpleNamespace(_clear_target_prefix={}), [req]
            )
            self.assertTrue(fallback.fallback)
            self.assertIsNone(fallback._verify_quotes)
            self.assertEqual(seen, [])
            with patch(
                "sglang.srt.speculative.clear_tree_remote.clear_tree_fit.prefit_plan",
                return_value={"use_tau": True, "width": 8, "k": 8},
            ):
                prefit = make_round(SimpleNamespace(), [req])
            self.assertEqual(prefit.prefit["k"], 8)
            self.assertIsNone(prefit._verify_quotes)
            self.assertEqual(seen, [])

    def test_cut_and_note_share_one_selected_k(self):
        prices = {8: 1.0, 16: 1.0}

        def fake(_batch, k, _bucket, _ell=None):
            return prices.get(int(k))

        def counted(plan):
            calls = {"n": 0}
            real = plan._select

            def wrapped(states):
                calls["n"] += 1
                return real(states)

            plan._select = wrapped
            return calls

        def tree(rid, scores):
            state = ClearTreeState.create(ClearTreeConfig(width=8, tau=0.4, max_depth=2))
            state.depth = 1
            state.weight[0, : len(scores)] = torch.tensor(scores, dtype=torch.float32)
            state.tokens[0] = torch.arange(8)
            state.rid = rid
            state.prefix = 4
            return state

        with patch.object(policy, "predict_verify_ms", side_effect=fake):
            single = tree("one", [0.5, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01])
            plan = _value_plan()
            plan._verify_quotes = {}
            plan.bind_states([single])
            calls = counted(plan)
            single.decider = plan
            step = step_after_layer(single)
            keep = int(plan._frozen_k)
            ordered = sorted(float(v) for v in single.weight[0].tolist())
            ordered.reverse()
            self.assertEqual(calls["n"], 1)
            self.assertTrue(step.is_final)
            self.assertEqual(int(step.verify.tokens.shape[0]), keep)
            self.assertAlmostEqual(
                policy._pending["one"]["e"], float(sum(ordered[:keep]))
            )

            left = tree("left", [0.6, 0.2, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005])
            right = tree("right", [0.3, 0.2, 0.1, 0.08, 0.04, 0.02, 0.01, 0.005])
            batch = _value_plan()
            batch.batch = 2
            batch.layer_ms = 0.1
            batch._verify_quotes = {}
            batch.bind_states([left, right])
            self.assertTrue(batch.defer_finish)
            calls = counted(batch)
            pending = {0, 1}
            results = {}
            _cut_deferred(batch, [left, right], pending, results)
            shared = int(batch._frozen_k)
            self.assertEqual(calls["n"], 1)
            self.assertFalse(pending)
            self.assertEqual(len(results[0][0]), shared)
            self.assertEqual(len(results[1][0]), shared)
            for rid, state in (("left", left), ("right", right)):
                ranked = sorted(float(v) for v in state.weight[0].tolist())
                ranked.reverse()
                self.assertAlmostEqual(
                    policy._pending[rid]["e"], float(sum(ranked[:shared]))
                )

    def test_policy_stages_sum_to_the_outer_sample(self):
        ledger = ClearLedger(active_rids=["a"])
        _step(ledger, ["a"], [8], 1, 1.0)
        scheduler = SimpleNamespace(_clear_ledger=ledger)
        owner = SimpleNamespace(_scheduler=scheduler)

        @ledger_module.policy_timed(stage="final_k")
        def inner(_owner):
            return 7

        @ledger_module.policy_timed(stage="grow")
        def outer(_scheduler):
            return inner(owner)

        @ledger_module.policy_timed(stage="plan")
        def planned(_scheduler):
            return 1

        @ledger_module.policy_timed(stage="recv")
        def received(_scheduler):
            return 1

        with patch.object(
            ledger_module.time,
            "perf_counter",
            side_effect=[2.0, 2.005, 3.0, 3.004, 4.0, 4.001],
        ) as clock:
            self.assertEqual(outer(scheduler), 7)
            self.assertEqual(planned(scheduler), 1)
            self.assertEqual(received(scheduler), 1)
        self.assertEqual(clock.call_count, 6)
        self.assertAlmostEqual(ledger.open.policy_grow_ms, 5.0)
        self.assertEqual(ledger.open.policy_final_k_ms, 0.0)
        self.assertAlmostEqual(ledger.open.policy_plan_ms, 4.0)
        self.assertAlmostEqual(ledger.open.policy_recv_ms, 1.0)
        staged = (
            ledger.open.policy_recv_ms
            + ledger.open.policy_plan_ms
            + ledger.open.policy_grow_ms
            + ledger.open.policy_final_k_ms
        )
        self.assertAlmostEqual(staged, ledger.open.policy_ms)
        self.assertAlmostEqual(ledger.open.policy_ms, 10.0)

        scheduler._clear_in_post_reply = True
        with patch.object(ledger_module.time, "perf_counter") as clock:
            self.assertEqual(outer(scheduler), 7)
        clock.assert_not_called()
        self.assertAlmostEqual(ledger.open.policy_ms, 10.0)
        scheduler._clear_in_post_reply = False
        with patch.dict(os.environ, {"SGLANG_CLEAR_LEDGER": "0"}), patch.object(
            ledger_module.time, "perf_counter"
        ) as clock:
            self.assertEqual(outer(scheduler), 7)
        clock.assert_not_called()

    def test_recv_time_lands_on_the_round_opened_by_that_packet(self):
        sched = SimpleNamespace(
            server_args=SimpleNamespace(clear_tree_width=8),
            tp_rank=0,
        )

        def packet(base, rpc):
            return SimpleNamespace(
                reqs=[
                    SimpleNamespace(
                        rid="a", base_committed_len=base, committed_ids=None
                    )
                ],
                action="step",
                session_id="s",
                rpc_seq=rpc,
            )

        ticks = {"n": 10.0}

        def clock():
            ticks["n"] += 0.002
            return ticks["n"]

        with patch.object(ledger_module.time, "perf_counter", side_effect=clock):
            note_packet(sched, packet(8, 1))
            first = sched._clear_ledger.open
            recv = first.policy_recv_ms
            self.assertGreater(recv, 0.0)
            self.assertAlmostEqual(
                recv,
                first.policy_recv_ms
                + first.policy_plan_ms
                + first.policy_grow_ms
                + first.policy_final_k_ms,
            )
            first.t_send = 1.0
            note_packet(sched, packet(9, 2))
            ledger_module.run_deferred_close(sched)
        closed = next(
            record
            for kind, record in sched._clear_ledger.pending_logs
            if kind == "d_round"
        )
        opened = sched._clear_ledger.open
        self.assertAlmostEqual(closed["policy_recv_ms"], recv)
        self.assertAlmostEqual(closed["policy_ms"], recv)
        self.assertGreater(opened.policy_recv_ms, 0.0)
        self.assertIsNot(opened, first)
        staged = (
            closed["policy_recv_ms"]
            + closed["policy_plan_ms"]
            + closed["policy_grow_ms"]
            + closed["policy_final_k_ms"]
        )
        self.assertAlmostEqual(staged, closed["policy_ms"])


class TestDeferredClose(CustomTestCase):
    """The previous round's accounting runs after the reply, with the same result."""

    def setUp(self):
        reset_policy()
        self._env = {
            name: os.environ.get(name)
            for name in ("SGLANG_CLEAR_LEDGER", "SGLANG_CLEAR_DEFER_CLOSE")
        }
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        os.environ.pop("SGLANG_CLEAR_DEFER_CLOSE", None)

    def tearDown(self):
        for name, value in self._env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        reset_policy()

    @staticmethod
    def _sched():
        return SimpleNamespace(
            server_args=SimpleNamespace(clear_tree_width=8),
            tp_rank=0,
        )

    @staticmethod
    def _packet(base, rpc, session="s"):
        return SimpleNamespace(
            reqs=[
                SimpleNamespace(rid="a", base_committed_len=base, committed_ids=None)
            ],
            action="step",
            session_id=session,
            rpc_seq=rpc,
        )

    def _drive(self, deferred):
        """Four STEP packets, each followed by one build and one reply."""
        if deferred:
            os.environ.pop("SGLANG_CLEAR_DEFER_CLOSE", None)
        else:
            os.environ["SGLANG_CLEAR_DEFER_CLOSE"] = "0"
        reset_policy()
        sched = self._sched()
        now = {"t": 100.0}
        with patch.object(ledger_module.time, "perf_counter", side_effect=lambda: now["t"]):
            for rpc, base in enumerate((100, 103, 107, 110), start=1):
                now["t"] = 100.0 + 0.07 * rpc
                ledger_module.note_packet(sched, self._packet(base, rpc))
                ledger = sched._clear_ledger
                ledger.open.k = 16
                # What RoundPlan.note writes while this round's tree is built.
                note_verified("a", base, 2.5 + rpc, 8)
                ledger.mark_stop(now["t"] + 0.020)
                ledger.mark_send(now["t"] + 0.021)
                # The Draft loop, after the reply and before the post step.
                ledger_module.run_deferred_close(sched)
        closed = [
            dict(record)
            for kind, record in sched._clear_ledger.pending_logs
            if kind == "d_round"
        ]
        state = {
            "clock": {key: list(rows) for key, rows in policy._clock.items()},
            "sum_a": dict(policy._sum_a),
            "sum_e": dict(policy._sum_e),
            "price_ms": dict(policy._price_ms),
            "price_tok": dict(policy._price_tok),
            "pending": {key: dict(item) for key, item in policy._pending.items()},
        }
        return closed, state

    def test_close_waits_for_the_post_step(self):
        sched = self._sched()
        ledger_module.note_packet(sched, self._packet(100, 1))
        ledger = sched._clear_ledger
        ledger.open.k = 16
        note_verified("a", 100, 2.5, 8)
        ledger.mark_stop()
        ledger.mark_send()
        ledger_module.note_packet(sched, self._packet(103, 2))
        self.assertIsNotNone(ledger.deferred)
        self.assertFalse([1 for kind, _ in ledger.pending_logs if kind == "d_round"])
        self.assertEqual(policy._pending, {})
        ledger_module.run_deferred_close(sched)
        self.assertIsNone(ledger.deferred)
        closed = [record for kind, record in ledger.pending_logs if kind == "d_round"]
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["rpc_seq"], 1)
        self.assertTrue(closed[0]["close_deferred"])
        self.assertGreaterEqual(closed[0]["close_ms"], 0.0)
        self.assertEqual(closed[0]["price_tokens"], 3)

    def test_the_next_build_does_not_leak_into_the_settlement(self):
        sched = self._sched()
        ledger_module.note_packet(sched, self._packet(100, 1))
        ledger = sched._clear_ledger
        ledger.open.k = 16
        note_verified("a", 100, 2.5, 8)
        ledger.mark_stop()
        ledger.mark_send()
        ledger_module.note_packet(sched, self._packet(103, 2))
        # Round 2 is built before round 1 is accounted.
        note_verified("a", 103, 9.0, 8)
        ledger_module.run_deferred_close(sched)
        self.assertAlmostEqual(policy._sum_a["a"], 2.0)
        self.assertAlmostEqual(policy._sum_e["a"], 2.5)
        self.assertEqual(policy._pending["a"]["prefix"], 103)
        self.assertAlmostEqual(policy._pending["a"]["e"], 9.0)

    def test_deferred_and_receive_close_give_the_same_accounting(self):
        sync_rounds, sync_state = self._drive(deferred=False)
        late_rounds, late_state = self._drive(deferred=True)
        self.assertEqual(len(sync_rounds), 3)
        self.assertEqual(len(late_rounds), 3)
        for sync, late in zip(sync_rounds, late_rounds):
            self.assertFalse(sync.pop("close_deferred"))
            self.assertTrue(late.pop("close_deferred"))
            sync.pop("close_ms")
            late.pop("close_ms")
            self.assertEqual(sync, late)
        self.assertEqual(sync_state, late_state)
        self.assertTrue(sync_state["clock"])
        self.assertTrue(sync_state["price_ms"])

    def test_session_change_closes_at_receive(self):
        sched = self._sched()
        ledger_module.note_packet(sched, self._packet(100, 1, session="s"))
        ledger = sched._clear_ledger
        ledger.open.k = 16
        ledger.mark_stop()
        ledger.mark_send()
        ledger_module.note_packet(sched, self._packet(103, 2, session="t"))
        self.assertIsNone(ledger.deferred)
        closed = [record for kind, record in ledger.pending_logs if kind == "d_round"]
        self.assertEqual(len(closed), 1)
        self.assertFalse(closed[0]["close_deferred"])


class TestLayerDepthAndExpect(CustomTestCase):
    """Per-depth layer correction and the accepted-versus-E diagnostics."""

    def setUp(self):
        reset_policy()
        self._env = {
            name: os.environ.get(name)
            for name in ("SGLANG_CLEAR_LEDGER", "SGLANG_CLEAR_LAYER_DEPTH")
        }
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        os.environ.pop("SGLANG_CLEAR_LAYER_DEPTH", None)

    def tearDown(self):
        for name, value in self._env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        reset_policy()

    @staticmethod
    def _fitted_ledger():
        model = ForgettingRegressor(2, LAYER_MIN_SAMPLES, ridge=LAYER_RIDGE, span_feature=1)
        for prefix in (100.0, 200.0, 300.0, 400.0):
            model.update([1.0, prefix], 6.0)
        ledger = ClearLedger()
        ledger.layer_models[(8, "graph", 1)] = model
        return ledger

    @staticmethod
    def _sample(layer, *, last=False, path="graph", wall_us=5400):
        return {
            "layer": layer,
            "active": 1,
            "graph_bucket": 1,
            "w": 8,
            "sum_ell": 200,
            "host_us": 1000,
            "gpu_us": 4000,
            "wall_us": wall_us,
            "path": path,
            "post_reply": False,
            "last": last,
        }

    def test_last_pre_reply_layer_is_marked(self):
        ledger = ClearLedger()
        rnd = SimpleNamespace(
            layers=[
                {"t0": 1.000},
                {"t0": 1.006},
                {"t0": 1.012},
                {"t0": 1.030, "post_reply": True},
            ],
            t_stop=1.018,
            t_open=0.990,
        )
        ledger._assign_layer_walls(rnd)
        self.assertEqual([item.get("last") for item in rnd.layers[:3]], [False, False, True])
        self.assertNotIn("last", rnd.layers[3])

    def test_only_middle_graph_layers_teach_the_depth_ratio(self):
        ledger = self._fitted_ledger()
        rnd = SimpleNamespace(prefit=None, rpc_seq=1, session_id="s")
        ledger._finish_layer(rnd, self._sample(3), False)
        self.assertEqual(ledger.layer_depth[(8, 1, 3)][2], 1)
        self.assertEqual(ledger.layer_depth_pooled[(1, 3)][2], 1)
        for sample, excluded in (
            (self._sample(3, last=True), False),
            (self._sample(3, path="eager"), False),
            # Without a wall the fallback host plus GPU is a different span.
            (self._sample(3, wall_us=None), False),
            (self._sample(3), True),
        ):
            ledger._finish_layer(rnd, sample, excluded)
        self.assertEqual(ledger.layer_depth[(8, 1, 3)][2], 1)
        logged = [record for kind, record in ledger.pending_logs if kind == "d_layer"]
        self.assertEqual(len(logged), 5)
        self.assertEqual(logged[0]["depth_src"], "none")
        self.assertAlmostEqual(logged[0]["depth_ratio"], 1.0)
        self.assertTrue(logged[1]["last"])

    def test_depth_ratio_scales_prices_once_it_has_samples(self):
        ledger = self._fitted_ledger()
        sched = SimpleNamespace(_clear_ledger=ledger, tp_rank=0)
        base = predict_layer_ms(sched, 8, 1, 200)
        slope = layer_sum_ell_slope(sched, 8, 1)
        for _ in range(ledger_module.DEPTH_MIN_SAMPLES - 1):
            ledger._learn_depth({"layer": 4, "graph_bucket": 1, "w": 8}, 9.0, 10.0)
        self.assertAlmostEqual(predict_layer_ms(sched, 8, 1, 200, depth=4), base)
        ledger._learn_depth({"layer": 4, "graph_bucket": 1, "w": 8}, 9.0, 10.0)
        self.assertEqual(ledger_module.layer_depth_ratio(ledger, 8, 1, 4)[1], "cell")
        self.assertAlmostEqual(predict_layer_ms(sched, 8, 1, 200, depth=4), base * 0.9)
        self.assertAlmostEqual(layer_sum_ell_slope(sched, 8, 1, depth=4), slope * 0.9)
        # Without a depth the price is the base fit.
        self.assertAlmostEqual(predict_layer_ms(sched, 8, 1, 200), base)
        # Another width at the same capture batch borrows the pooled row.
        ratio, source = ledger_module.layer_depth_ratio(ledger, 16, 1, 4)
        self.assertEqual(source, "pooled")
        self.assertAlmostEqual(ratio, 0.9)
        # Deep forwards share the capped bucket.
        self.assertEqual(ledger_module._depth_key(12), ledger_module.DEPTH_CAP)
        os.environ["SGLANG_CLEAR_LAYER_DEPTH"] = "0"
        self.assertAlmostEqual(predict_layer_ms(sched, 8, 1, 200, depth=4), base)
        # Installing fits replaces the regressions the ratios were measured on.
        from sglang.srt.speculative.clear_tree_remote import clear_tree_fit

        clear_tree_fit._install_layers(sched, [])
        self.assertEqual(ledger.layer_depth, {})
        self.assertEqual(ledger.layer_depth_pooled, {})

    def test_round_records_accepts_against_expected_mass(self):
        ledger = ClearLedger(active_rids=["a", "b"])
        sched = SimpleNamespace(_clear_ledger=ledger, tp_rank=0)
        _step(ledger, ["a", "b"], [100, 200], 1, 1.0)
        ledger_module.note_round_expect(sched, "a", 0.5, 4.0, 3)
        ledger_module.note_round_expect(sched, "b", 0.8, 2.0, 2)
        ledger.mark_stop(1.0)
        ledger.mark_send(1.01)
        # "b" finished, so only "a" is compared.
        record = _step(ledger, ["a"], [104], 2, 1.08)
        self.assertEqual(record["accepted"], 3)
        self.assertAlmostEqual(record["e_pred"], 2.0)
        self.assertAlmostEqual(record["e_mass"], 4.0)
        self.assertEqual(record["e_n"], 1)
        self.assertEqual(record["tree_depth"], 3)

        ledger.mark_stop(1.1)
        ledger.mark_send(1.11)
        empty = _step(ledger, ["a"], [106], 3, 1.2)
        self.assertIsNone(empty["e_pred"])
        self.assertIsNone(empty["accepted"])

    def test_stop_decision_and_replay_price_the_forwarded_depth(self):
        state = SimpleNamespace(depth=3, config=SimpleNamespace(max_depth=8), prefix=100)
        plan = policy.RoundPlan.__new__(policy.RoundPlan)
        plan._scheduler = None
        plan.fixed_depth = None
        plan.use_tau = False
        plan.theta_value = 1.0
        plan.layer_ms = 1.0
        plan._at = None
        plan._peers = [state]
        plan._keep_ids = set()
        plan._cost_depth = None
        plan.width = 8
        with patch.object(policy.RoundPlan, "_choose_keep", return_value=[]):
            plan.should_grow(state)
        self.assertEqual(plan._cost_depth, 3)

        seen = []
        factors = []

        def capture(*_args, **kwargs):
            seen.append(kwargs.get("depth"))
            return 2.0

        def factor(_scheduler, _width, active, depth):
            factors.append((active, depth))
            return 0.5

        plan._scheduler = SimpleNamespace()
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            side_effect=capture,
        ), patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.layer_depth_factor",
            side_effect=factor,
        ):
            # The stop decision prices the forward at the current depth, and
            # reads that depth's multiplier once for the decision.
            self.assertAlmostEqual(plan._layer_cost(1, 100), 1.0)
            self.assertAlmostEqual(plan._layer_cost(1, 200), 1.0)
            self.assertEqual(factors, [(1, 3)])
            self.assertEqual(seen, [None, None])
            seen.clear()
            rows = replay._freeze_layer_rows(SimpleNamespace(), [state])
        self.assertEqual(len(rows), 2)
        self.assertEqual(sorted(set(seen)), [1, 2])


if __name__ == "__main__":
    unittest.main()
