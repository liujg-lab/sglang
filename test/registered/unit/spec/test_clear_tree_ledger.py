"""CPU checks for the CLEAR_TREE_REMOTE shadow ledger.

The ledger must not change width or tau. These tests stay on the pure
accounting object.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative.clear_tree_remote.clear_tree_layout import (
    build_tree_from_parents,
)
from sglang.srt.speculative.clear_tree_remote.clear_tree_policy import (
    RoundPlan,
    accept_ratio,
    flat_width,
    ingest_verify_samples,
    make_round,
    observe_accept,
    observe_price,
    observe_verify,
    predict_verify_ms,
    reset_policy,
    theta,
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
        _step(ledger, ["a"], [14], 2, 10.40)
        self.assertAlmostEqual(theta(1), 4.0 / 380.0)

        ledger.mark_stop(10.50)
        ledger.mark_send(10.55)
        ledger.mark_post(10.56)
        ledger.note_idle()
        _step(ledger, ["a"], [40], 3, 11.40)
        self.assertAlmostEqual(theta(1), 4.0 / 380.0)

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
        idle = _step(fresh, ["a"], [4], 2, 1.4)
        self.assertIn("idle_start", idle["exclude_reason"])
        self.assertEqual(fresh.verify_samples, 0)

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

        grown = ClearLedger(active_rids=["a"])
        _step(grown, ["a"], [4], 1, 1.0)
        grown.mark_stop(1.0)
        grown.mark_send(1.1)
        grown.mark_post(1.2)
        grown.note_idle()
        added = _step(grown, ["a", "b"], [4, 8], 2, 1.5)
        self.assertIn("new_rid", added["exclude_reason"])
        self.assertEqual(grown.verify_samples, 0)

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

    def test_flat_width(self):
        self.assertEqual(flat_width(1), 8)
        self.assertEqual(flat_width(2), 4)
        self.assertEqual(flat_width(8), 4)

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
