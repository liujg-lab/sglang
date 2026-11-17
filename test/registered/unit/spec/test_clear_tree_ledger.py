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
    grow_active_layers,
)
from sglang.srt.speculative.clear_tree_remote import clear_tree_policy as policy
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
    layer_should_continue,
    hold_replay,
    make_round,
    matched_depth,
    note_verified,
    on_external_event,
    observe_accept,
    observe_price,
    observe_replay,
    observe_verify,
    persistent_width,
    predict_verify_ms,
    request_terms,
    reset_gain_calibration,
    reset_policy,
    select_continuing,
    settle_replay,
    settle_verified,
    theta,
    truncate_columns,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_replay import (
    _freeze_layer_rows,
    _schedule_ms,
    batch_stop_depth,
    replay_accept_k,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_ops import (
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


def _step(ledger, rids, prefix, rpc, now, width=8):
    return ledger.on_packet("step", rids, "sess", rpc, prefix, width, now=now)


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
        record = _step(ledger, ["a"], [32], 2, 10.40)

        self.assertEqual(record["t_k_us"], 400000)
        self.assertTrue(record["draft_floor"])
        self.assertFalse(record["excluded"])
        self.assertEqual(ledger.verify_samples, 0)

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
        self.assertIsNone(prefill.on_packet("prefill", ["b"], "sess", 2, [9], 8, now=1.3))
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
        self.assertEqual(ledger.layer_models[(8, "graph")].n, 1)

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
        self.assertEqual(smallest_bucket(100, buckets), select_tree_kv_bucket(100, buckets))
        self.assertEqual(smallest_bucket(300, buckets), select_tree_kv_bucket(300, buckets))
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
        self.assertAlmostEqual(theta(2), 2.0 ** -0.5)

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

            def note(self, state):
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
        self.assertEqual(matched_depth(layers, kept, [1, 3]), 2)
        self.assertEqual(matched_depth(layers, kept, [2, 4]), 0)

    def test_batch_one_explores_the_next_wider_width(self):
        reset_policy()
        chosen = [
            choose_build_width(1, ledger_on=True, eligible=True) for _ in range(20)
        ]
        self.assertEqual(chosen[:4], [(4, "probe_w")] * 4)
        self.assertTrue(all(item == (8, "ordinary") for item in chosen[4:-1]))
        self.assertEqual(chosen[-1], (16, "explore"))
        assign_persistent_width(1, 4)
        later = [
            choose_build_width(1, ledger_on=True, eligible=True) for _ in range(20)
        ]
        self.assertEqual(later[:4], [(2, "probe_w")] * 4)
        self.assertTrue(all(item == (4, "ordinary") for item in later[4:-1]))
        self.assertEqual(later[-1], (8, "explore"))

    def test_widest_width_does_not_leave(self):
        assign_persistent_width(1, 64)
        chosen = [
            choose_build_width(1, ledger_on=True, eligible=True) for _ in range(20)
        ]
        self.assertEqual(chosen[:4], [(32, "probe_w")] * 4)
        self.assertTrue(all(item == (64, "ordinary") for item in chosen[4:]))

    def test_wider_batch_explores_only_while_the_ledger_is_on(self):
        self.assertEqual(
            choose_build_width(2, ledger_on=False, eligible=True), (4, "ordinary")
        )
        cold = [
            choose_build_width(2, ledger_on=True, eligible=False) for _ in range(20)
        ]
        self.assertTrue(all(item == (4, "ordinary") for item in cold))
        hot = [
            choose_build_width(2, ledger_on=True, eligible=True) for _ in range(20)
        ]
        self.assertEqual(hot[:4], [(2, "probe_w")] * 4)
        self.assertTrue(all(item == (4, "ordinary") for item in hot[4:-1]))
        self.assertEqual(hot[-1], (8, "explore"))

    def test_warmup_round_does_not_enter_the_replay_sum(self):
        skipped = observe_replay(
            1, 8, "ordinary", 10.0, 10.0, {4: (14.0, 10.0)}, counted=False
        )
        self.assertEqual(skipped["rounds"], 0)
        counted = observe_replay(1, 8, "ordinary", 10.0, 10.0, {4: (14.0, 10.0)})
        self.assertEqual(counted["rounds"], 1)

    def test_large_replay_bias_does_not_change_width(self):
        for _ in range(30):
            observe_replay(1, 4, "measure", 10.0, 10.0)
        last = None
        for _ in range(40):
            last = observe_replay(
                1, 8, "ordinary", 10.0, 10.0, {4: (14.0, 10.0)}
            )
        self.assertGreater(last["gain"], 0.03)
        self.assertNotIn("gate", last)
        self.assertGreater(abs(last["bias"]), 0.03)
        self.assertTrue(last["applied"])
        self.assertEqual(last["would_w"], 4)
        self.assertLess(last["rounds"], 200)
        self.assertNotIn("se", last)
        sched = SimpleNamespace(_clear_ledger=False)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        self.assertEqual(make_round(sched, [req]).width, 4)

    def test_bias_inside_three_percent_switches_the_next_round(self):
        for _ in range(30):
            observe_replay(1, 4, "measure", 14.0, 10.0)
        last = None
        for _ in range(40):
            last = observe_replay(
                1, 8, "ordinary", 10.0, 10.0, {4: (14.0, 10.0)}
            )
        self.assertNotIn("gate", last)
        self.assertAlmostEqual(last["bias"], 0.0)
        self.assertTrue(last["applied"])
        self.assertEqual(last["would_w"], 4)
        self.assertLess(last["rounds"], 200)
        self.assertNotIn("se", last)
        sched = SimpleNamespace(_clear_ledger=False)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        self.assertEqual(make_round(sched, [req]).width, 4)

    def test_open_pair_bias_serves_the_next_wider_width(self):
        reset_policy()
        for _ in range(30):
            observe_replay(1, 8, "ordinary", 10.0, 10.0)
        closed = None
        for _ in range(10):
            closed = observe_replay(
                1, 16, "explore", 12.0, 10.0, {8: (10.0, 10.0)}
            )
        self.assertFalse(closed["applied"])
        self.assertEqual(persistent_width(1), 8)
        for _ in range(20):
            closed = observe_replay(
                1, 16, "explore", 12.0, 10.0, {8: (10.0, 10.0)}
            )
        self.assertEqual(closed["would_w"], 16)
        self.assertNotIn("gate", closed)
        self.assertTrue(closed["applied"])
        self.assertEqual(persistent_width(1), 16)

    def test_narrow_move_stops_at_the_adjacent_width(self):
        reset_policy()
        for _ in range(30):
            observe_replay(1, 4, "measure", 11.0, 10.0)
        for _ in range(30):
            observe_replay(1, 2, "measure", 14.0, 10.0)
        last = None
        for _ in range(40):
            last = observe_replay(
                1,
                8,
                "ordinary",
                10.0,
                10.0,
                {4: (11.0, 10.0), 2: (14.0, 10.0)},
            )
        self.assertEqual(last["would_w"], 2)
        self.assertTrue(last["applied"])
        self.assertEqual(persistent_width(1), 4)

    def test_far_narrow_gate_does_not_skip_the_adjacent_width(self):
        reset_policy()
        for _ in range(30):
            observe_replay(1, 2, "measure", 14.0, 10.0)
        last = None
        for _ in range(40):
            last = observe_replay(
                1,
                8,
                "ordinary",
                10.0,
                10.0,
                {4: (11.0, 10.0), 2: (14.0, 10.0)},
            )
        self.assertEqual(last["would_w"], 2)
        self.assertTrue(last["applied"])
        self.assertEqual(persistent_width(1), 4)

    def test_width_two_never_becomes_the_served_width(self):
        reset_policy()
        for _ in range(30):
            observe_replay(1, 4, "measure", 10.2, 10.0)
        last = None
        for _ in range(40):
            last = observe_replay(
                1,
                8,
                "ordinary",
                10.0,
                10.0,
                {4: (10.2, 10.0), 2: (40.0, 10.0)},
            )
        self.assertEqual(last["would_w"], 2)
        self.assertFalse(last["applied"])
        self.assertEqual(persistent_width(1), 8)
        sched = SimpleNamespace(_clear_ledger=False)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        self.assertEqual(make_round(sched, [req]).width, 8)

    def test_settle_counts_only_the_kept_branch(self):
        for k in (2, 4, 8, 16, 32):
            for _ in range(6):
                observe_verify(1, k, 256, 5.0)
        scores = [0.9, 0.8, 0.7, 0.6, 0.01, 0.01, 0.01, 0.01]
        state = SimpleNamespace(
            depth=1,
            tokens=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            parents=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]),
            weight=torch.tensor([scores]),
            prefix=4,
            rid="a",
        )
        plan = _value_plan()
        plan.replay_kind = "ordinary"
        plan.replay_eligible = True
        sched = SimpleNamespace(_clear_ledger=False, tp_rank=0, tp_size=1)
        req = SimpleNamespace(rid="a", output_ids=[11])
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            hold_replay(sched, [req], [state], plan)
        req.output_ids = [11, 8]
        sched._clear_replay_closed = {
            "excluded": False,
            "price_kept": True,
            "price_ms": 12.0,
        }
        missed = settle_replay(sched, [req])
        self.assertEqual(missed["candidates"]["4"]["tokens"], 1.0)
        req.output_ids = [11]
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            hold_replay(sched, [req], [state], plan)
        req.output_ids = [11, 1, 5]
        sched._clear_replay_closed = {
            "excluded": False,
            "price_kept": True,
            "price_ms": 12.0,
        }
        hit = settle_replay(sched, [req])
        self.assertEqual(hit["candidates"]["4"]["tokens"], 2.0)
        self.assertIsNotNone(hit["candidates"]["4"]["ms"])
        req.output_ids = [11]
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            hold_replay(sched, [req], [state], plan)
        sched._clear_replay_closed = {
            "excluded": True,
            "price_kept": False,
            "price_ms": 12.0,
        }
        self.assertIsNone(settle_replay(sched, [req]))

    def test_rewritten_zero_temperature_explores_the_twentieth_round(self):
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
            for _ in range(80):
                plan = make_round(sched, reqs)
                eligible.append(
                    (plan.width, plan.replay_kind, plan.replay_eligible)
                )
                if len(eligible) == 20:
                    break
        self.assertEqual(
            eligible,
            [(2, "probe_w", False)] * 4
            + [(4, "ordinary", True)] * 15
            + [(8, "explore", True)],
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
        closed = sched._clear_replay_closed
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

        def predict(_scheduler, width, _active, _sum_ell):
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

    def test_ordinary_rounds_judge_every_twenty(self):
        reset_policy()
        last = None
        for _ in range(20):
            last = observe_replay(2, 4, "ordinary", 10.0, 10.0, {2: (12.0, None)})
        self.assertEqual(last["rounds"], 20)

    def test_width_two_does_not_block_a_servable_switch(self):
        reset_policy()
        last = None
        for n in range(1, 41):
            last = observe_replay(
                2,
                8,
                "explore",
                10.6,
                10.0,
                {4: (10.0, 10.0), 2: (11.0, 10.0)},
                apply=n > 20,
            )
        self.assertEqual(last["rounds"], 40)
        self.assertEqual(last["would_w"], 2)
        self.assertTrue(last["applied"])
        self.assertEqual(persistent_width(2), 2)

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

    def test_replay_accept_stops_at_k_prime(self):
        layers = [
            [
                {"token": 1, "parent": -1, "score": 0.9},
                {"token": 2, "parent": -1, "score": 0.2},
            ],
            [
                {"token": 3, "parent": 0, "score": 0.4},
                {"token": 4, "parent": 1, "score": 0.05},
            ],
        ]
        kept = truncate_columns(layers, 2)
        self.assertEqual(replay_accept_k(layers, kept, [2, 4], 2, 1), 1)
        self.assertEqual(replay_accept_k(layers, kept, [2, 4], 2, 4), 2)

    def test_joint_stop_uses_the_latest_written_layer(self):
        def tree():
            layers = [
                [{"token": 1, "parent": -1, "score": 0.2}],
                [{"token": 2, "parent": 0, "score": 0.04}],
                [{"token": 3, "parent": 0, "score": 0.9}],
            ]
            return {
                "layers": layers,
                "kept": truncate_columns(layers, 1),
                "ratio": 1.0,
                "prefix": 1,
            }

        rows = [{"pred": {4: 0.05}}, {"pred": {4: 0.05}}]
        verify = {8: 0.0, 16: 0.0, 32: 0.0}
        both = [tree(), tree()]
        self.assertEqual(
            batch_stop_depth(
                both, theta=1.0, verify_ms=verify, layer_rows=rows, width=4
            ),
            3,
        )
        self.assertEqual(
            batch_stop_depth(
                both[:1], theta=1.0, verify_ms=verify, layer_rows=rows, width=4
            ),
            2,
        )

    def test_mixed_k_rows_are_padded_not_dropped(self):
        tokens, parents, missing = align_draft_row(
            [1] * 8, [-1] + [0] * 7, 16
        )
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

        def predict(_scheduler, width, _active, _sum_ell):
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
            ingest_verify_samples(
                [[1, 8, 99999, 10000], [1, 8, 99999, 11000]]
            )
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
        from sglang.srt.speculative.clear_tree_remote.clear_tree_replay import (
            _replayed_counts,
            _select_replay_k,
        )

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
        layers = [
            [
                {"token": 0, "parent": -1, "score": 0.4},
                {"token": 1, "parent": -1, "score": 0.3},
                {"token": 2, "parent": -1, "score": 0.2},
                {"token": 3, "parent": -1, "score": 0.1},
            ]
        ]
        hold = {
            "width": 4,
            "theta": 1.0,
            "verify_ms": {
                2: 100.0,
                4: 1.0,
                8: 1.0,
                16: 1.0,
                32: 1.0,
                64: 1.0,
            },
            "layers": [],
            "k": 2,
            "measured_ms": 50.0,
            "trees": {
                "a": {"layers": layers, "ratio": 1.0, "prefix": 1},
            },
        }
        replayed = _replayed_counts(hold, {"a": [0]})
        self.assertEqual(replayed[2][1], 50.0)

    def test_finished_tree_does_not_keep_the_batch_growing(self):
        def nodes(scores):
            return [
                [
                    {"token": index, "parent": -1, "score": score}
                    for index, score in enumerate(scores)
                ]
            ]

        def tree(scores):
            layers = nodes(scores)
            return {
                "layers": layers,
                "kept": truncate_columns(layers, 2),
                "ratio": 1.0,
                "prefix": 1,
            }

        short = tree([0.2, 0.1])
        short["layers"] = short["layers"] + [
            [
                {"token": 10, "parent": 0, "score": 9.0},
                {"token": 11, "parent": 1, "score": 9.0},
            ]
        ]
        short["kept"] = truncate_columns(short["layers"], 2)
        long_layers = nodes([0.2, 0.1])
        for depth in range(3):
            long_layers.append(
                [
                    {"token": 20 + depth, "parent": 0, "score": 0.0},
                    {"token": 30 + depth, "parent": 1, "score": 0.0},
                ]
            )
        long = {
            "layers": long_layers,
            "kept": truncate_columns(long_layers, 2),
            "ratio": 1.0,
            "prefix": 1,
        }
        rows = [{"pred": {2: 5.0}} for _ in range(4)]
        self.assertEqual(
            batch_stop_depth(
                [short, long],
                theta=1.0,
                verify_ms={64: 1.0},
                layer_rows=rows,
                width=2,
            ),
            2,
        )

    def test_layer_prefix_ridge_stays_inside_the_seen_span(self):
        model = ForgettingRegressor(
            3, LAYER_MIN_SAMPLES, ridge=LAYER_RIDGE, span_feature=2
        )
        for prefix, cost in (
            (900.0, 5.8),
            (900.0, 6.2),
            (903.0, 5.7),
            (903.0, 6.3),
        ):
            model.update([1.0, 1.0, prefix], cost)
        self.assertLess(abs(model.coefficients()[2]), 1e-3)
        ledger = ClearLedger()
        ledger.layer_models[(4, "graph")] = model
        sched = SimpleNamespace(_clear_ledger=ledger, tp_rank=0)
        near = predict_layer_ms(sched, 4, 1, 1000)
        self.assertIsNotNone(near)
        self.assertGreater(near, 0.0)
        self.assertLess(abs(near - 6.0), 1.0)
        self.assertIsNone(predict_layer_ms(sched, 4, 1, 2500))

        negative = ForgettingRegressor(
            3, LAYER_MIN_SAMPLES, ridge=LAYER_RIDGE, span_feature=2
        )
        for prefix in (900.0, 900.0, 903.0, 903.0):
            negative.update([1.0, 1.0, prefix], -1.0)
        ledger.layer_models[(8, "graph")] = negative
        self.assertIsNone(predict_layer_ms(sched, 8, 1, 901))

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
            3, LAYER_MIN_SAMPLES, ridge=LAYER_RIDGE, span_feature=2
        )
        filled.n = LAYER_MIN_SAMPLES
        ledger.layer_models[(4, "graph")] = filled
        self.assertEqual(
            choose_build_width(1, ledger_on=True, eligible=True, scheduler=sched),
            (8, "ordinary"),
        )
        _narrow_idle[(1, 4)] = PROBE_W_IDLE
        again = [
            choose_build_width(1, ledger_on=True, eligible=True, scheduler=sched)
            for _ in range(3)
        ]
        self.assertEqual(
            again, [(4, "probe_w"), (4, "probe_w"), (8, "ordinary")]
        )

    def test_narrow_width_probes_early_when_the_prefix_is_unpriced(self):
        from sglang.srt.speculative.clear_tree_remote.clear_tree_replay import (
            _narrow_idle,
        )

        def prepared(idle):
            reset_policy()
            ledger = ClearLedger()
            filled = ForgettingRegressor(
                3, LAYER_MIN_SAMPLES, ridge=LAYER_RIDGE, span_feature=2
            )
            filled.n = LAYER_MIN_SAMPLES
            ledger.layer_models[(4, "graph")] = filled
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
        observe_replay(1, 16, "ordinary", 10.0, 10.0, {8: (12.0, 9.0)})
        ledger = ClearLedger()
        fitted = ForgettingRegressor(
            3, LAYER_MIN_SAMPLES, ridge=LAYER_RIDGE, span_feature=2
        )
        fitted.n = 4
        ledger.layer_models[(8, "graph")] = fitted
        ledger.verify_model = ForgettingRegressor(4, 6)
        ledger.verify_samples = 3
        ledger.open = object()
        sched = SimpleNamespace(
            _clear_ledger=ledger,
            _clear_replay_hold={"tree": 1},
            _clear_replay_closed={"excluded": False},
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
        self.assertIsNone(sched._clear_replay_hold)
        self.assertIsNone(sched._clear_replay_closed)
        fresh = observe_replay(1, 16, "ordinary", 10.0, 10.0, {8: (12.0, 9.0)})
        self.assertEqual(fresh["rounds"], 1)

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
        self.assertEqual(ledger.layer_models[(8, "graph")].n, 0)
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
        self.assertEqual(policy._clock[(4, 16, bucket)], [80.0])

        idle = ClearLedger(active_rids=[])
        _step(idle, ["a", "b", "c", "d"], [100, 100, 100, 100], 3, 2.0)
        idle.open.k = 16
        idle.mark_stop(2.0)
        idle.mark_send(2.0)
        idle.mark_post(2.01)
        idle.note_idle()
        skipped = _step(idle, ["a", "b", "c", "d"], [100, 100, 100, 100], 4, 2.08)
        self.assertIn("idle_start", skipped["exclude_reason"])
        self.assertEqual(policy._clock[(4, 16, bucket)], [80.0])

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
        model = ledger.layer_models[(8, "graph")]
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
        self.assertEqual(ledger.layer_models[(8, "graph")].n, 1)

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
        self.assertEqual(ledger.layer_models[(8, "graph")].n, 0)
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


if __name__ == "__main__":
    unittest.main()
