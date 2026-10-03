"""CPU-only checks for evaluation evidence and repeatability comparisons."""

import csv
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "sr_rsteller_eval",
    ROOT / "test_sglang_liujg/evaluation/eval_spectre_rsteller.py",
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def result(ids, count=None, text="caption", finished=True):
    return {
        "text": text,
        "output_ids": ids,
        "meta_info": {
            "completion_tokens": len(ids) if count is None else count,
            "finish_reason": {"type": "stop"} if finished else None,
        },
    }


class TestRSTellerEvidence(unittest.TestCase):
    def stream(self, responses):
        payload = "".join("data: " + json.dumps(r) + "\n\n" for r in responses)
        payload += "data: [DONE]\n"
        with patch.object(
            evaluation.urllib.request,
            "urlopen",
            return_value=io.BytesIO(payload.encode()),
        ):
            return evaluation.post_generate_stream("http://localhost/generate", {}, 1)[
                0
            ]

    def test_cumulative_and_incremental_streams_preserve_repeated_ids(self):
        cumulative = self.stream(
            [
                result([7], text="a", finished=False),
                result([7, 7], text="ab", finished=False),
                result([7, 7], text="ab"),
            ]
        )
        incremental = self.stream(
            [
                result([7], count=1, text="a", finished=False),
                result([7], count=2, text="b", finished=False),
                result([], count=2, text=""),
            ]
        )
        self.assertEqual(cumulative, incremental)
        rec = evaluation.extract_perf({}, incremental, ttft_s=0.1, client_e2e_s=1)
        self.assertEqual(rec["output_ids"], [7, 7])
        self.assertTrue(rec["token_ids_complete"])
        self.assertEqual(rec["finish_reason"], {"type": "stop"})

    def test_missing_and_gapped_tokens_are_not_claimed_complete(self):
        streamed = self.stream([result([4], count=3)])
        rec = evaluation.extract_perf({}, streamed, ttft_s=None, client_e2e_s=1)
        self.assertFalse(rec["token_ids_complete"])
        self.assertIsNone(rec["decode_s"])
        self.assertIsNone(rec["decode_tok_s"])
        self.assertEqual(rec["e2e_tok_s"], 3)
        missing = evaluation.extract_perf(
            {},
            {"text": "caption", "meta_info": {"completion_tokens": 2}},
            ttft_s=None,
            client_e2e_s=1,
        )
        self.assertIsNone(missing["output_ids"])
        self.assertFalse(missing["token_ids_complete"])

    def test_stream_error_is_not_hidden_by_later_chunk(self):
        with self.assertRaisesRegex(RuntimeError, "broken"):
            self.stream([{"error": "broken"}, result([1])])

    def test_comparison_first_difference_length_and_missing_evidence(self):
        def row(ids, **overrides):
            return dict(
                global_index=0,
                request_fingerprint="same",
                output_ids=ids,
                token_ids_complete=True,
                **overrides,
            )

        changed = evaluation.compare_records([row([1, 2, 3])], [row([1, 4, 3])])[0]
        self.assertEqual(changed["first_difference_index"], 1)
        self.assertEqual(changed["baseline_token"], 2)
        shorter = evaluation.compare_records([row([1, 2])], [row([1])])[0]
        self.assertEqual(shorter["first_difference_index"], 1)
        self.assertIsNone(shorter["current_token"])
        different_context = evaluation.compare_records(
            [row([1], batch_size=1)], [row([1], batch_size=2)]
        )[0]
        self.assertEqual(different_context["status"], "equal_tokens")
        self.assertIn("batch_size", different_context["context_differences"])
        old = row([1])
        old["token_ids_complete"] = False
        self.assertEqual(
            evaluation.compare_records([old], [row([1])])[0]["status"],
            "token_ids_unavailable",
        )
        old["request_fingerprint"] = "different"
        self.assertEqual(
            evaluation.compare_records([old], [row([1])])[0]["status"],
            "request_mismatch",
        )
        self.assertEqual(
            evaluation.compare_records([], [row([1])])[0]["status"], "missing_request"
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluation.compare_records([row([1]), row([2])], [])

    def test_main_keeps_runs_separate_and_routes_accuracy_to_own_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data.json"
            data.write_text(json.dumps([{"image": "a.jpg", "question": "q"}]))
            (root / "a.jpg").write_bytes(b"private test image")
            jsonl, csv_path = root / "result.jsonl", root / "stats.csv"
            jsonl.write_text("previous result")
            csv_path.write_text("previous summary")
            args = [
                "eval",
                "--data-file",
                str(data),
                "--img-dir",
                str(root),
                "--output-jsonl",
                str(jsonl),
                "--output-csv",
                str(csv_path),
                "--flush-cache",
                "--max-items",
                "1",
            ]

            def post(url, *a, **kw):
                if url.endswith("/server_info"):
                    return {"tp_size": 2, "api_key": "do not persist"}
                if url.endswith("/flush_cache"):
                    raise RuntimeError("flush rejected")
                return result([7, 8])

            with (
                patch.object(evaluation, "post_json", side_effect=post),
                patch.object(
                    evaluation,
                    "post_generate_stream",
                    return_value=(result([7, 8]), 0.01),
                ),
                patch.object(evaluation, "call_metrics_script") as score,
                redirect_stdout(io.StringIO()),
            ):
                with patch.object(evaluation.sys, "argv", args):
                    self.assertEqual(evaluation.main(), 0)
                first = score.call_args.args[0]
                with patch.object(
                    evaluation.sys, "argv", args + ["--compare-jsonl", str(first)]
                ):
                    self.assertEqual(evaluation.main(), 0)
                second = score.call_args.args[0]
            self.assertNotEqual(first, second)
            self.assertEqual(jsonl.read_text(), "previous result")
            self.assertEqual(csv_path.read_text(), "previous summary")
            self.assertEqual(len(list(root.glob("*.run.json"))), 2)
            saved = json.loads(second.read_text())
            self.assertEqual(saved["output_ids"], [7, 8])
            self.assertEqual(saved["global_index"], 0)
            self.assertEqual(saved["cache_flush"]["status"], "failed")
            self.assertIn("request_started_at", saved)
            manifest = json.loads(second.with_suffix(".run.json").read_text())
            self.assertEqual(manifest["server"]["settings"], {"tp_size": 2})
            self.assertEqual(
                manifest["comparison"]["requests"][0]["status"], "equal_tokens"
            )
            self.assertEqual(manifest["warmup_result"]["output_ids"], [7, 8])
            for call in score.call_args_list:
                with call.args[1].open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                self.assertIn("End-to-end tok/s", rows[0])
                self.assertNotIn("Decode tok/s", rows[0])

    def test_summary_uses_full_chunk_wall_time(self):
        summary = evaluation.summarize([{"completion_tokens": 100}], 2)
        self.assertEqual(summary["e2e_tok_s"], 50)
        self.assertNotIn("decode_tok_s", summary)

    def test_concurrent_requests_keep_dataset_order_and_nonstream_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.jpg").write_bytes(b"image")
            items = [(i, {"image": "a.jpg", "question": str(i)}) for i in range(3)]

            def post(url, payload, timeout):
                token = next(
                    i for i in range(3) if f">{i}<|im_end|>" in payload["text"]
                )
                self.assertNotIn("stream", payload)
                return result([token, 9])

            with patch.object(evaluation, "post_json", side_effect=post):
                _, rows = evaluation.run_chunk_concurrent(
                    "http://localhost/generate", items, root, {"temperature": 0}, 1
                )
            self.assertEqual([i for i, _ in rows], [0, 1, 2])
            self.assertEqual(
                [r["output_ids"] for _, r in rows], [[0, 9], [1, 9], [2, 9]]
            )
            self.assertTrue(all(r["token_ids_complete"] for _, r in rows))
            self.assertTrue(all(r["decode_tok_s"] is None for _, r in rows))
            self.assertEqual(len({r["request_fingerprint"] for _, r in rows}), 3)

    def test_server_metadata_failure_does_not_abort_evaluation(self):
        with patch.object(evaluation, "post_json", side_effect=RuntimeError("denied")):
            self.assertEqual(
                evaluation.server_snapshot("http://localhost", 1)["status"],
                "unavailable",
            )


if __name__ == "__main__":
    unittest.main()
