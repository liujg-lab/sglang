#!/usr/bin/env python3
"""Client-clock stream metrics for the serving eval. No server required."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_spectre_rsteller_serving as serving
from eval_spectre_rsteller import extract_perf, spec_condition_columns

STOP = {"type": "stop"}


def chunk(
    count: int | None,
    text: str = "",
    ids: list[int] | None = None,
    *,
    finish: dict | None = None,
    e2e: float | None = None,
    spec_accept_rate: float | None = None,
    omit_count: bool = False,
    count_value: object | None = None,
) -> dict:
    meta: dict = {}
    if not omit_count:
        meta["completion_tokens"] = count if count_value is None else count_value
    if finish is not None:
        meta["finish_reason"] = finish
    if e2e is not None:
        meta["e2e_latency"] = e2e
    if spec_accept_rate is not None:
        meta["spec_accept_rate"] = spec_accept_rate
    data: dict = {"text": text, "meta_info": meta}
    if ids is not None:
        data["output_ids"] = ids
    return data


class ObserveStreamTest(unittest.TestCase):
    def test_ar_excludes_first_token_and_done_wait(self) -> None:
        obs = serving.observe_stream(
            [
                (0.10, chunk(1, "a", [1])),
                (0.20, chunk(2, "ab", [1, 2])),
                (0.40, chunk(4, "abcd", [1, 2, 3, 4], finish=STOP, e2e=9.0)),
                (0.55, "[DONE]"),
            ],
            0.60,
        )
        self.assertIsNone(obs.error)
        self.assertEqual(obs.ttft_s, 0.10)
        self.assertEqual(obs.first_text_s, 0.10)
        self.assertEqual(obs.first_chunk_tokens, 1)
        self.assertEqual(obs.completion_tokens, 4)
        self.assertEqual(obs.decode_tokens, 3)
        self.assertAlmostEqual(obs.decode_s, 0.30)
        self.assertAlmostEqual(obs.decode_tok_s, 10.0)
        self.assertEqual(obs.client_e2e_s, 0.60)
        self.assertEqual(obs.server_e2e_s, 9.0)
        self.assertEqual(obs.text, "abcd")
        self.assertEqual(obs.output_ids, [1, 2, 3, 4])
        self.assertEqual(
            obs.token_events,
            [
                {"t_rel_s": 0.10, "completion_tokens": 1, "delta_tokens": 1},
                {"t_rel_s": 0.20, "completion_tokens": 2, "delta_tokens": 1},
                {"t_rel_s": 0.40, "completion_tokens": 4, "delta_tokens": 2},
            ],
        )
        self.assertEqual(obs.itl, [0.10, 0.10, 0.10])

    def test_sr_first_chunk_drops_k_and_keeps_text_delay(self) -> None:
        obs = serving.observe_stream(
            [
                (0.20, chunk(3, "", [10, 11, 12])),
                (0.25, chunk(3, "hi", [10, 11, 12])),
                (0.50, chunk(5, "hello", [10, 11, 12, 13, 14], finish=STOP)),
                (0.70, "[DONE]"),
            ],
            0.80,
        )
        self.assertIsNone(obs.error)
        self.assertEqual(obs.ttft_s, 0.20)
        self.assertEqual(obs.first_text_s, 0.25)
        self.assertEqual(obs.first_chunk_tokens, 3)
        self.assertEqual(obs.decode_tokens, 2)
        self.assertAlmostEqual(obs.decode_s, 0.30)
        self.assertAlmostEqual(obs.decode_tok_s, 2 / 0.30)
        self.assertEqual(obs.itl, [0.15, 0.15])
        self.assertEqual(obs.text, "hello")
        self.assertEqual(obs.output_ids, [10, 11, 12, 13, 14])

    def test_single_token_event_has_zero_decode_and_blank_speed(self) -> None:
        obs = serving.observe_stream(
            [
                (0.15, chunk(1, "a", [1], finish=STOP)),
                (0.40, "[DONE]"),
            ],
            0.45,
        )
        self.assertEqual(obs.decode_s, 0.0)
        self.assertEqual(obs.decode_tokens, 0)
        self.assertIsNone(obs.decode_tok_s)
        self.assertEqual(obs.itl, [])
        self.assertEqual(obs.client_e2e_s, 0.45)

    def test_zero_gap_leaves_speed_blank(self) -> None:
        obs = serving.observe_stream(
            [
                (0.20, chunk(1, "a", [1])),
                (0.20, chunk(3, "abc", [1, 2, 3], finish=STOP)),
            ],
            0.50,
        )
        self.assertEqual(obs.decode_tokens, 2)
        self.assertEqual(obs.decode_s, 0.0)
        self.assertIsNone(obs.decode_tok_s)

    def test_duplicate_text_finish_and_done_do_not_extend_decode(self) -> None:
        obs = serving.observe_stream(
            [
                (0.10, chunk(1, "a", [1])),
                (0.40, chunk(1, "a", [1])),
                (0.50, chunk(2, "ab", [1, 2])),
                (0.80, chunk(2, "ab", [1, 2], finish=STOP)),
                (0.95, "[DONE]"),
            ],
            1.10,
        )
        self.assertIsNone(obs.error)
        self.assertAlmostEqual(obs.decode_s, 0.40)
        self.assertEqual(obs.decode_tokens, 1)
        self.assertAlmostEqual(obs.ttft_s, 0.10)
        self.assertEqual(obs.client_e2e_s, 1.10)

    def test_missing_or_regressing_counts_are_unavailable(self) -> None:
        missing = serving.observe_stream(
            [
                (0.10, chunk(1, "a", [1])),
                (0.20, chunk(None, "ab", [1, 2], omit_count=True)),
                (0.30, chunk(2, "ab", [1, 2], finish=STOP)),
            ],
            0.40,
        )
        self.assertIsNone(missing.error)
        self.assertFalse(missing.token_metrics_available)
        self.assertIsNone(missing.completion_tokens)
        self.assertIsNone(missing.decode_s)
        self.assertIsNone(missing.decode_tokens)
        self.assertIsNone(missing.decode_tok_s)
        self.assertEqual(missing.itl, [])
        self.assertEqual(missing.text, "ab")

        regressed = serving.observe_stream(
            [
                (0.10, chunk(2, "ab", [1, 2])),
                (0.30, chunk(1, "a", [1], finish=STOP)),
            ],
            0.40,
        )
        self.assertFalse(regressed.token_metrics_available)
        self.assertIsNone(regressed.decode_tok_s)
        self.assertEqual(regressed.text, "ab")

        fractional = serving.observe_stream(
            [
                (
                    0.10,
                    chunk(
                        None,
                        "a",
                        [1],
                        finish=STOP,
                        count_value=1.5,
                    ),
                )
            ],
            0.20,
        )
        self.assertFalse(fractional.token_metrics_available)
        self.assertEqual(fractional.text, "a")
        self.assertIsNone(fractional.completion_tokens)

    def test_incremental_and_cumulative_text(self) -> None:
        incremental = serving.observe_stream(
            [
                (0.10, chunk(1, "hel", [1])),
                (0.30, chunk(2, "lo", [2], finish=STOP)),
            ],
            0.40,
        )
        self.assertEqual(incremental.text, "hello")
        self.assertEqual(incremental.output_ids, [1, 2])

        kept = serving.observe_stream(
            [
                (0.10, chunk(2, "hi", [1, 2])),
                (0.20, chunk(3, "", [1, 2, 3], finish=STOP)),
            ],
            0.30,
        )
        self.assertEqual(kept.text, "hi")
        self.assertEqual(kept.output_ids, [1, 2, 3])

    def test_truncated_stream_is_a_failure(self) -> None:
        obs = serving.observe_stream(
            [
                (0.10, chunk(1, "a", [1])),
                (0.20, "[DONE]"),
            ],
            0.25,
        )
        self.assertEqual(obs.error, "stream ended without finish_reason")
        self.assertEqual(obs.text, "a")

    def test_server_error_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            serving.observe_stream([(0.1, {"error": "boom"})], 0.2)

    def test_parse_sse_payload(self) -> None:
        self.assertEqual(serving.parse_sse_payload("data: [DONE]\n"), "[DONE]")
        self.assertEqual(serving.parse_sse_payload('data: {"a": 1}')["a"], 1)
        self.assertIsNone(serving.parse_sse_payload("\n"))


class ApplyAndSummaryTest(unittest.TestCase):
    def test_apply_replaces_extract_perf_times(self) -> None:
        obs = serving.observe_stream(
            [
                (0.10, chunk(1, "a", [1])),
                (0.40, chunk(4, "abcd", [1, 2, 3, 4], finish=STOP, e2e=9.0, spec_accept_rate=0.5)),
            ],
            0.60,
        )
        rec = extract_perf(
            {"image": "a.jpg", "question": "q", "ground_truth": "g"},
            {"text": obs.text, "output_ids": obs.output_ids, "meta_info": obs.meta_info},
            ttft_s=1.0,
            client_e2e_s=9.0,
        )
        self.assertEqual(rec["decode_s"], 8.0)
        self.assertEqual(rec["decode_tok_s"], 0.5)
        serving.apply_stream_observation(rec, obs)
        self.assertIsNone(rec["prefill_s"])
        self.assertIsNone(rec["e2e_tok_s"])
        self.assertAlmostEqual(rec["decode_s"], 0.30)
        self.assertEqual(rec["decode_tokens"], 3)
        self.assertAlmostEqual(rec["decode_tok_s"], 10.0)
        self.assertEqual(rec["ttft_s"], 0.10)
        self.assertEqual(rec["client_e2e_s"], 0.60)
        self.assertEqual(rec["server_e2e_s"], 9.0)
        self.assertEqual(rec["spec_accept_rate"], 0.5)
        self.assertEqual(rec["itl_kind"], serving.ITL_KIND)
        self.assertTrue(rec["token_ids_complete"])

    def test_weighted_decode_rate_and_tpot_split(self) -> None:
        records = [
            {
                "completion_tokens": 4,
                "first_chunk_tokens": 1,
                "decode_tokens": 3,
                "decode_s": 1.0,
                "ttft_s": 0.2,
                "client_e2e_s": 1.5,
                "server_e2e_s": 9.0,
                "itl": [0.5, 0.5],
            },
            {
                "completion_tokens": 5,
                "first_chunk_tokens": 3,
                "decode_tokens": 2,
                "decode_s": 0.5,
                "ttft_s": 0.4,
                "client_e2e_s": 1.0,
                "server_e2e_s": None,
                "itl": [0.25, 0.25],
            },
            {
                "completion_tokens": 1,
                "first_chunk_tokens": 1,
                "decode_tokens": 0,
                "decode_s": 0.0,
                "ttft_s": 0.2,
                "client_e2e_s": 0.2,
                "itl": [],
            },
            {
                "completion_tokens": 3,
                "first_chunk_tokens": 1,
                "decode_tokens": 2,
                "decode_s": 0.0,
                "ttft_s": 0.1,
                "client_e2e_s": 0.3,
                "itl": [0.0, 0.0],
            },
            {
                "error": "stream ended without finish_reason",
                "completion_tokens": 100,
                "decode_tokens": 99,
                "decode_s": 0.1,
                "client_e2e_s": 50.0,
                "server_e2e_s": 80.0,
                "ttft_s": 1.0,
            },
            {
                "completion_tokens": None,
                "decode_s": None,
                "decode_tokens": None,
                "token_metrics_available": False,
                "client_e2e_s": 0.4,
                "ttft_s": None,
            },
        ]
        stats = serving.summarize_serving(records, wall_s=10.0)
        self.assertEqual(stats["decode_samples"], 2)
        self.assertEqual(stats["total_decode_tokens"], 5)
        self.assertAlmostEqual(stats["total_decode_time_s"], 1.5)
        self.assertAlmostEqual(stats["decode_tok_s"], 5 / 1.5)
        self.assertAlmostEqual(stats["output_tok_s"], 13 / 10)
        self.assertAlmostEqual(stats["mean_decode_ms"], 375.0)
        self.assertAlmostEqual(stats["mean_tpot_ms"], 1000.0 / 3)
        self.assertAlmostEqual(stats["mean_post_first_ms"], 250.0)
        self.assertAlmostEqual(stats["mean_ttft_ms"], 225.0)
        self.assertAlmostEqual(stats["mean_e2e_ms"], 680.0)
        self.assertAlmostEqual(stats["mean_server_e2e_ms"], 9000.0)
        self.assertAlmostEqual(stats["concurrency"], 3.4 / 10)
        self.assertEqual(stats["n_ok"], 5)
        self.assertEqual(stats["n_total"], 6)

    def test_no_measurable_decode_samples_leave_rate_empty(self) -> None:
        stats = serving.summarize_serving(
            [
                {
                    "completion_tokens": 1,
                    "first_chunk_tokens": 1,
                    "decode_tokens": 0,
                    "decode_s": 0.0,
                    "ttft_s": 0.2,
                    "client_e2e_s": 0.2,
                }
            ],
            wall_s=1.0,
        )
        self.assertEqual(stats["decode_samples"], 0)
        self.assertIsNone(stats["decode_tok_s"])
        self.assertIsNone(stats["total_decode_tokens"])
        self.assertIsNone(stats["total_decode_time_s"])
        self.assertEqual(stats["mean_decode_ms"], 0.0)
        self.assertIsNone(stats["mean_tpot_ms"])
        self.assertIsNone(stats["mean_post_first_ms"])

    def test_weighted_accept_length_removes_prefill_and_fallback(self) -> None:
        stats = serving.summarize_serving(
            [
                {
                    "completion_tokens": 5,
                    "spec_verify_ct": 2,
                    "spec_fallback_ct": 1,
                    "spec_accept_token_num": 3,
                    "spec_draft_token_num": 14,
                    "spec_accept_length": 2.5,
                    "spec_accept_rate": 3 / 14,
                },
                {
                    "completion_tokens": 4,
                    "spec_verify_ct": 0,
                    "spec_fallback_ct": 3,
                    "spec_accept_token_num": 0,
                    "spec_draft_token_num": 0,
                },
            ],
            wall_s=1.0,
        )
        # (5-1-1) + (4-1-3) = 3, over 2 verify steps.
        self.assertAlmostEqual(stats["accept_length_weighted"], 1.5)
        self.assertAlmostEqual(stats["tokens_per_target_step"], 7 / 6)
        self.assertAlmostEqual(stats["accepted_draft_per_verify"], 1.5)
        self.assertAlmostEqual(stats["accept_rate_weighted"], 3 / 14)
        self.assertEqual(stats["total_fallback_steps"], 4)
        self.assertAlmostEqual(stats["fallback_ratio"], 4 / 6)
        self.assertEqual(stats["spec_samples"], 2)
        self.assertEqual(stats["fallback_known"], "yes")
        # The legacy column is still the equal-weight mean of the biased ratio.
        self.assertAlmostEqual(stats["avg_alen"], 2.5)

    def test_missing_fallback_field_is_an_upper_bound(self) -> None:
        rec = extract_perf(
            {},
            {
                "text": "a",
                "meta_info": {
                    "completion_tokens": 3,
                    "spec_verify_ct": 1,
                    "spec_accept_token_num": 1,
                    "spec_draft_token_num": 7,
                    "spec_accept_length": 3.0,
                },
            },
            ttft_s=None,
            client_e2e_s=1.0,
        )
        self.assertIsNone(rec["spec_fallback_ct"])
        stats = serving.summarize_serving([rec], wall_s=1.0)
        self.assertAlmostEqual(stats["accept_length_weighted"], 2.0)
        self.assertEqual(stats["fallback_known"], "no")
        self.assertEqual(stats["total_fallback_steps"], 0)

    def test_verify_zero_request_stays_in_the_totals(self) -> None:
        stats = serving.summarize_serving(
            [
                {
                    "completion_tokens": 2,
                    "spec_verify_ct": 0,
                    "spec_fallback_ct": 1,
                    "spec_accept_token_num": 0,
                    "spec_draft_token_num": 0,
                }
            ],
            wall_s=1.0,
        )
        self.assertEqual(stats["spec_samples"], 1)
        self.assertIsNone(stats["accept_length_weighted"])
        self.assertAlmostEqual(stats["tokens_per_target_step"], 1.0)
        self.assertEqual(stats["fallback_known"], "yes")
        self.assertEqual(serving.metric_csv_values(stats)["Spec Samples"], "1")

    def test_csv_header_mismatch_refuses_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.csv"
            serving.write_csv_row(path, {field: "1" for field in serving.CSV_FIELDS})
            with path.open(newline="", encoding="utf-8") as f:
                header = next(csv.reader(f))
            self.assertEqual(header, serving.CSV_FIELDS)
            serving.write_csv_row(path, {field: "2" for field in serving.CSV_FIELDS})
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["Decode Samples"], "2")

            stale = Path(tmp) / "old.csv"
            stale.write_text("Test Time,Avg Prefill (ms)\n1,2\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                serving.write_csv_row(stale, {field: "" for field in serving.CSV_FIELDS})


class AlignedClientsTest(unittest.TestCase):
    def test_text_serving_uses_shared_csv_columns(self) -> None:
        import eval_spectre_text_serving as text_serving

        self.assertEqual(text_serving.CSV_FIELDS, serving.CSV_FIELDS)
        self.assertIn("Decode tok/s", text_serving.CSV_FIELDS)
        self.assertNotIn("Avg Prefill (ms)", text_serving.CSV_FIELDS)

    def test_closed_loop_k1_uses_tpot_and_weighted_decode(self) -> None:
        import eval_spectre_rsteller as closed

        obs = closed.consume_sse_lines(
            [
                (
                    0.10,
                    'data: {"text":"a","output_ids":[1],'
                    '"meta_info":{"completion_tokens":1}}',
                ),
                (
                    0.40,
                    'data: {"text":"abcd","output_ids":[1,2,3,4],'
                    '"meta_info":{"completion_tokens":4,'
                    '"finish_reason":{"type":"stop"}}}',
                ),
                (0.55, "data: [DONE]"),
            ],
            0.60,
        )
        self.assertIsNone(obs.error)
        self.assertEqual(obs.first_chunk_tokens, 1)
        self.assertEqual(obs.decode_tokens, 3)
        self.assertAlmostEqual(obs.decode_s, 0.30)
        k1 = {
            "completion_tokens": obs.completion_tokens,
            "first_chunk_tokens": obs.first_chunk_tokens,
            "decode_tokens": obs.decode_tokens,
            "decode_s": obs.decode_s,
            "ttft_s": obs.ttft_s,
            "client_e2e_s": obs.client_e2e_s,
            "itl": obs.itl,
        }
        only_k1 = serving.summarize_serving([k1], wall_s=1.0)
        self.assertAlmostEqual(only_k1["mean_tpot_ms"], 100.0)
        self.assertIsNone(only_k1["mean_post_first_ms"])

        # Two overlapped closed-loop requests still share one weighted rate.
        other = {
            "completion_tokens": 5,
            "first_chunk_tokens": 3,
            "decode_tokens": 2,
            "decode_s": 0.5,
            "ttft_s": 0.2,
            "client_e2e_s": 0.8,
            "itl": [0.25, 0.25],
        }
        mixed = serving.summarize_serving([k1, other], wall_s=2.0)
        self.assertAlmostEqual(mixed["decode_tok_s"], (3 + 2) / (0.30 + 0.5))
        self.assertAlmostEqual(mixed["mean_post_first_ms"], 250.0)
        fields = closed.closed_csv_fields()
        self.assertEqual(fields[:3], ["Test Time", "batch_size", "mode"])
        self.assertIn("Algorithm", fields)
        self.assertIn("Accept Length (weighted)", fields)
        self.assertIn("Decode tok/s", fields)
        self.assertIn("Mean Post-first Delivery (ms)", fields)
        self.assertNotIn("Avg Prefill (ms)", fields)
        self.assertNotIn("End-to-end tok/s", fields)


class SpecConditionColumnsTest(unittest.TestCase):
    def test_unavailable_snapshot_is_unknown(self) -> None:
        columns = spec_condition_columns(
            {"status": "unavailable", "error": "connection refused"}
        )
        self.assertEqual(columns, {"Algorithm": "unknown", "Spec Config": ""})

    def test_available_without_speculation_is_ar(self) -> None:
        columns = spec_condition_columns(
            {"status": "available", "settings": {"speculative_algorithm": None}}
        )
        self.assertEqual(columns, {"Algorithm": "AR", "Spec Config": ""})

    def test_standalone_remote_records_spec_shape(self) -> None:
        columns = spec_condition_columns(
            {
                "status": "available",
                "settings": {
                    "speculative_algorithm": "STANDALONE_REMOTE",
                    "speculative_num_steps": 4,
                    "speculative_eagle_topk": 2,
                    "speculative_num_draft_tokens": 8,
                },
            }
        )
        self.assertEqual(columns["Algorithm"], "STANDALONE_REMOTE")
        self.assertEqual(columns["Spec Config"], "steps=4/topk=2/ndt=8")


if __name__ == "__main__":
    unittest.main()
