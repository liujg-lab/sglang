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
    build_tree_from_parents,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_config import (
    ClearTreeConfig,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_forward import (
    grow_active_layers,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
    RoundPlan,
    accept_ratio,
    choose_build_width,
    flat_width,
    ingest_verify_samples,
    compare_price,
    layer_should_continue,
    hold_replay,
    make_round,
    matched_depth,
    observe_accept,
    observe_price,
    observe_replay,
    observe_verify,
    persistent_width,
    predict_verify_ms,
    request_terms,
    reset_policy,
    select_continuing,
    settle_replay,
    theta,
    truncate_columns,
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
    VERIFY_MIN_SAMPLES,
    ClearLedger,
    push_verify_sample,
    take_verify_samples,
    ForgettingRegressor,
    consume_verify_device,
    emit_verify_device,
    graph_bucket,
    note_target_prefill,
    read_verify_host_us,
    reset_target_verify_model,
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


class ClearTreeLedgerTest(CustomTestCase):
    def setUp(self):
        self._saved = {name: os.environ.get(name) for name in _LEDGER_ENV}
        os.environ["SGLANG_CLEAR_LEDGER"] = "1"
        reset_target_verify_model()

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
        self.assertAlmostEqual(theta(2), 2.0 ** 0.8)
        sched = SimpleNamespace(_clear_ledger=False)
        reqs = [
            SimpleNamespace(rid="a", origin_input_ids=[1], output_ids=[]),
            SimpleNamespace(rid="b", origin_input_ids=[1], output_ids=[]),
        ]
        plan = make_round(sched, reqs)
        self.assertAlmostEqual(plan.theta_value, 2.0 ** 0.8)
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
        state = _score_state([0.4, 0.2, 0.1, 0.05])
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
        sched = SimpleNamespace(_clear_ledger=False)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        locked = [make_round(sched, [req]).locked_k for _ in range(6)]
        self.assertEqual(locked, [8, 8, 16, 16, 32, 32])
        warmed = make_round(sched, [req])
        self.assertTrue(warmed.use_tau)
        self.assertEqual(warmed.locked_k, 32)

    def test_new_batch_restarts_probes_at_k8(self):
        sched = SimpleNamespace(_clear_ledger=False)
        one = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        two = [
            one,
            SimpleNamespace(rid="b", origin_input_ids=[1] * 10, output_ids=[]),
        ]
        locked = [make_round(sched, [one]).locked_k for _ in range(6)]
        self.assertEqual(locked, [8, 8, 16, 16, 32, 32])
        again = make_round(sched, two)
        self.assertTrue(again.use_tau)
        self.assertEqual(again.locked_k, 8)

    def test_missing_tier_stays_on_tau_until_every_k_is_timed(self):
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
        self.assertEqual(held.locked_k, 32)
        observe_verify(2, 8, 256, 10.0)
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
        sched = SimpleNamespace(_clear_ledger=False)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        for _ in range(10):
            make_round(sched, [req])
        for k, ms in ((8, 10.0), (16, 10.0), (32, 10.0)):
            observe_verify(1, k, 256, ms)
        for _ in range(3):
            observe_verify(1, 8, 256, 10.0)
        observe_price(1, 2.0, 1.0)
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.predict_layer_ms",
            return_value=1.0,
        ):
            plan = make_round(sched, [req])
        self.assertFalse(plan.use_tau)
        self.assertIsNone(plan.locked_k)
        self.assertEqual(plan.width, 8)

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
        sched = SimpleNamespace(_clear_ledger=False)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 4070, output_ids=[])
        saved = os.environ.get("SGLANG_CLEAR_NPU_KV_BUCKETS")
        os.environ["SGLANG_CLEAR_NPU_KV_BUCKETS"] = "256,512,1024,2048,4096"
        try:
            locked = [make_round(sched, [req]).locked_k for _ in range(6)]
        finally:
            if saved is None:
                os.environ.pop("SGLANG_CLEAR_NPU_KV_BUCKETS", None)
            else:
                os.environ["SGLANG_CLEAR_NPU_KV_BUCKETS"] = saved
        self.assertEqual(locked, [8, 8, 16, 16, 16, 16])
        self.assertNotIn(32, locked)

    def test_ingested_sample_is_visible_before_the_round_plan(self):
        ingest_verify_samples(
            [[1, 8, 256, 10000], [1, 16, 256, 10000], [1, 32, 256, 10000]]
            + [[1, 8, 256, 10000]] * 3
        )
        self.assertEqual(predict_verify_ms(1, 8, 256), 10.0)

    def test_lock_k_does_not_consume_probes(self):
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
        self.assertEqual(nxt.locked_k, 8)

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

    def test_batch_one_measures_four_and_never_builds_wider(self):
        chosen = [
            choose_build_width(1, ledger_on=True, eligible=True) for _ in range(20)
        ]
        self.assertEqual(chosen[-1], (4, "measure"))
        self.assertTrue(all(width <= 8 for width, _kind in chosen))
        self.assertTrue(all(width != 2 for width, _kind in chosen))
        self.assertTrue(all(kind != "explore" for _width, kind in chosen))

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
        self.assertEqual(hot[-1], (8, "explore"))
        self.assertEqual(hot[0], (4, "ordinary"))

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
        self.assertFalse(last["gate"])
        self.assertFalse(last["applied"])
        self.assertEqual(last["would_w"], 4)
        self.assertLess(last["rounds"], 200)
        self.assertNotIn("se", last)
        sched = SimpleNamespace(_clear_ledger=False)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        self.assertEqual(make_round(sched, [req]).width, 8)

    def test_bias_inside_three_percent_switches_the_next_round(self):
        for _ in range(30):
            observe_replay(1, 4, "measure", 14.0, 10.0)
        last = None
        for _ in range(40):
            last = observe_replay(
                1, 8, "ordinary", 10.0, 10.0, {4: (14.0, 10.0)}
            )
        self.assertTrue(last["gate"])
        self.assertTrue(last["applied"])
        self.assertEqual(last["would_w"], 4)
        self.assertLess(last["rounds"], 200)
        self.assertNotIn("se", last)
        sched = SimpleNamespace(_clear_ledger=False)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        self.assertEqual(make_round(sched, [req]).width, 4)

    def test_width_two_never_becomes_the_served_width(self):
        for _ in range(30):
            observe_replay(1, 4, "measure", 11.0, 10.0)
        last = None
        for _ in range(40):
            last = observe_replay(
                1,
                8,
                "ordinary",
                10.0,
                10.0,
                {4: (11.0, 10.0), 2: (40.0, 10.0)},
            )
        self.assertTrue(last["gate"])
        self.assertEqual(last["would_w"], 2)
        self.assertFalse(last["applied"])
        self.assertEqual(persistent_width(1), 8)
        sched = SimpleNamespace(_clear_ledger=False)
        req = SimpleNamespace(rid="a", origin_input_ids=[1] * 10, output_ids=[])
        self.assertEqual(make_round(sched, [req]).width, 8)

    def test_settle_counts_only_the_kept_branch(self):
        for k in (8, 16, 32):
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
        with patch(
            "sglang.srt.speculative.clear_tree_remote.clear_tree_ledger.ledger_enabled",
            return_value=True,
        ):
            eligible = []
            for _ in range(80):
                plan = make_round(sched, reqs)
                if plan.replay_eligible:
                    eligible.append((plan.width, plan.replay_kind))
                if len(eligible) == 20:
                    break
        self.assertEqual(eligible, [(4, "ordinary")] * 19 + [(8, "explore")])

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
        push_verify_sample(metrics, 1, 4, 256, 3000, True)
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
