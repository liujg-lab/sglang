#!/usr/bin/env python3
"""GSM8K serving 评测客户端（开放到达 + 流式百分位 + 第一版准确率）。

功能
    HTTP 客户端，打已启动的 Target（默认 :30000），不拉起服务。
    时间与投机列与 eval_spectre_rsteller_serving.py 相同：Poisson / burst
    到达、客户端 TTFT / Decode / E2E、服务端投机 meta_info。
    本脚本是纯文本，不传 image_data，也不计算 BLEU / ROUGE。

    一次运行对应一个已经启动的服务。自回归（不带 speculative-algorithm）
    和 STANDALONE_REMOTE 都走同一条请求路径；Algorithm / Spec Config 从
    /server_info 写入 CSV。要对两种方式各评一次，分别启动服务后各跑一遍，
    两次可以追加到同一个 --output-csv。

COT
    few-shot 只从 train.jsonl 取，默认 4 条，不使用 test 题，避免泄漏。
    示例答案去掉 <<...>> 计算器标注，保留逐步推理和结尾的 #### 数字。
    当前题以 "Answer:" 结尾，模型续写解答。stop 为 "\\nQuestion:"。

准确率
    取输出里最后一次 #### 后的数字。#### 与数字之间可以有空白，数字前可以有 $，
    数字里可以有千分位逗号。用十进制比较，所以 70000、70,000、70000.0 相同。
    没有可解析的 #### 时 extracted_pred 为 [invalid]，answer_status=invalid。
    抽到数字但与金标不同为 wrong。不回退到正文里的最后一个数字。
    HTTP 失败和流中断不进分母，也不写 answer_status。
    Accuracy = Correct / Scored，Scored 是无 error 的成功条数。
    finish_reason 为 length 的成功条记入 Truncated，仍算进分母。
    流式输出里同一条至少 40 字的非空行出现 3 次时，停止读取并 POST /abort_request。
    已收到的文本仍按 #### 计分，stopped_repeat 为 true，不写成 error。
    jsonl 保存 model_output、extracted_pred、extracted_gold、correct、answer_status、
    stopped_repeat。
    --rescore-jsonl 只重算已有 jsonl，不访问服务。

运行（仓库根目录，Target 已启动；AR 与 SR 各跑一次）
    python test_sglang_liujg/evaluation/eval_spectre_gsm8k_serving.py \\
      --port 30000 --max-items 20 --max-concurrency 1
    python test_sglang_liujg/evaluation/eval_spectre_gsm8k_serving.py \\
      --rescore-jsonl test_sglang_liujg/result/GSM8K_spectre_serving_results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from eval_spectre_rsteller import (  # noqa: E402
    _fmt,
    append_jsonl,
    create_run_paths,
    extract_perf,
    flush_cache,
    server_snapshot,
    spec_condition_columns,
    warn_eval_conditions,
)
from eval_spectre_rsteller_serving import (  # noqa: E402
    AIOHTTP_READ_BUFSIZE,
    CSV_FIELDS as SERVING_CSV_FIELDS,
    StreamAccumulator,
    _blank_stream_fields,
    _csv_num,
    apply_stream_observation,
    format_serving_summary,
    parse_sse_payload,
    get_request,
    metric_csv_values,
    print_serving_item,
    summarize_serving,
)

GSM8K_DATA_DIR = Path(
    "/home_18T/liujg/grade-school-math-master/grade_school_math/data"
)
DEFAULT_TEST = GSM8K_DATA_DIR / "test.jsonl"
DEFAULT_TRAIN = GSM8K_DATA_DIR / "train.jsonl"
DEFAULT_RESULT_DIR = HERE.parent / "result"

_DROPPED_METRIC_FIELDS = (
    "Avg BLEU",
    "Avg ROUGE-1",
    "Avg ROUGE-2",
    "Avg ROUGE-L",
)
_ACCURACY_FIELDS = (
    "Accuracy",
    "Correct",
    "Wrong",
    "Invalid Answers",
    "Truncated",
    "Scored",
)
REPEAT_LINE_MIN_CHARS = 40
REPEAT_LINE_LIMIT = 3
CSV_FIELDS = [
    name for name in SERVING_CSV_FIELDS if name not in _DROPPED_METRIC_FIELDS
] + list(_ACCURACY_FIELDS)

INVALID_ANS = "[invalid]"
# Last #### wins. Allow whitespace and a leading $, and thousand separators.
ANS_RE = re.compile(r"####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)")
CALC_RE = re.compile(r"<<[^<>]*>>")
STOP_STRING = "\nQuestion:"
COT_INSTRUCTION = (
    "Solve the grade-school math problem. Write a short step-by-step solution, "
    "then put the final numeric answer on its own line as `#### <number>`."
)


def load_jsonl(path: Path, max_items: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}")
            rows.append(row)
            if max_items is not None and max_items > 0 and len(rows) >= max_items:
                break
    return rows


def extract_answer(completion: str) -> str:
    """Last #### number, with commas removed. Unparseable markers stay invalid."""
    matches = ANS_RE.findall(completion or "")
    if not matches:
        return INVALID_ANS
    raw = matches[-1].replace(",", "")
    try:
        Decimal(raw)
    except InvalidOperation:
        return INVALID_ANS
    return raw


def answers_equal(pred: str, gold: str) -> bool:
    """Decimal equality: 70000, 70,000 and 70000.0 match after extraction."""
    if pred == INVALID_ANS or gold == INVALID_ANS:
        return False
    try:
        return Decimal(pred) == Decimal(gold)
    except InvalidOperation:
        return False


def strip_calculator(answer: str) -> str:
    return CALC_RE.sub("", answer).strip()


def format_example(question: str, answer: str | None) -> str:
    body = f"Question: {question.strip()}\nAnswer:"
    if answer is not None:
        body += " " + strip_calculator(answer)
    return body


def build_prompt(question: str, shots: list[dict[str, Any]]) -> str:
    parts = [COT_INSTRUCTION, ""]
    for shot in shots:
        parts.append(format_example(shot["question"], shot["answer"]))
        parts.append("")
    parts.append(format_example(question, None))
    user = "\n".join(parts)
    return (
        "<|im_start|>user\n"
        f"{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_score(rec: dict[str, Any]) -> dict[str, Any]:
    gold = extract_answer(rec.get("answer") or "")
    pred = extract_answer(rec.get("model_output") or "")
    rec["extracted_gold"] = gold
    rec["extracted_pred"] = pred
    if rec.get("error"):
        rec["correct"] = False
        rec.pop("answer_status", None)
        return rec
    if pred == INVALID_ANS:
        rec["correct"] = False
        rec["answer_status"] = "invalid"
        return rec
    rec["correct"] = answers_equal(pred, gold)
    rec["answer_status"] = "correct" if rec["correct"] else "wrong"
    return rec


def finish_type(rec: dict[str, Any]) -> str | None:
    reason = rec.get("finish_reason")
    if isinstance(reason, dict):
        value = reason.get("type")
        return value if isinstance(value, str) else None
    return None


def repeated_long_line(text: str) -> bool:
    """True when one non-empty line of at least 40 characters appears 3 times."""
    counts: dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if len(line) < REPEAT_LINE_MIN_CHARS:
            continue
        counts[line] = counts.get(line, 0) + 1
        if counts[line] >= REPEAT_LINE_LIMIT:
            return True
    return False


def summarize_accuracy(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [rec for rec in records if not rec.get("error")]
    correct = sum(1 for rec in ok if rec.get("answer_status") == "correct")
    wrong = sum(1 for rec in ok if rec.get("answer_status") == "wrong")
    invalid = sum(1 for rec in ok if rec.get("answer_status") == "invalid")
    truncated = sum(1 for rec in ok if finish_type(rec) == "length")
    scored = len(ok)
    accuracy = (correct / scored) if scored else None
    return {
        "accuracy": accuracy,
        "correct": correct,
        "wrong": wrong,
        "invalid": invalid,
        "truncated": truncated,
        "scored": scored,
    }


def format_accuracy(accuracy: dict[str, Any]) -> str:
    return (
        f"accuracy={_fmt(accuracy['accuracy'], 4)}  "
        f"correct={accuracy['correct']}/{accuracy['scored']}  "
        f"wrong={accuracy['wrong']}  "
        f"invalid={accuracy['invalid']}  "
        f"truncated={accuracy['truncated']}"
    )


def rescore_jsonl(src: Path, dest: Path) -> int:
    """Rewrite score fields of an existing jsonl. Does not call the server."""
    records = load_jsonl(src, None)
    if not records:
        raise SystemExit(f"no items in {src}")
    for rec in records:
        apply_score(rec)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    accuracy = summarize_accuracy(records)
    print(f"rescored {len(records)} items from {src}")
    print(format_accuracy(accuracy))
    print(f"wrote {dest}")
    return 0


def gsm8k_csv_values(
    stats: dict[str, Any], accuracy: dict[str, Any]
) -> dict[str, str]:
    values = metric_csv_values(stats)
    for name in _DROPPED_METRIC_FIELDS:
        values.pop(name, None)
    values["Accuracy"] = _csv_num(accuracy["accuracy"], 4)
    values["Correct"] = str(accuracy["correct"])
    values["Wrong"] = str(accuracy["wrong"])
    values["Invalid Answers"] = str(accuracy["invalid"])
    values["Truncated"] = str(accuracy["truncated"])
    values["Scored"] = str(accuracy["scored"])
    return values


def write_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    if file_exists:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), None)
        if header != CSV_FIELDS:
            raise SystemExit(
                f"CSV header in {csv_path} does not match the current columns. "
                "Refusing to append. Use a new --output-csv."
            )
    mode = "a" if file_exists else "w"
    with csv_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_gsm8k_item(idx: int, rec: dict[str, Any]) -> None:
    print_serving_item(idx, rec)
    if rec.get("error"):
        return
    flags = ""
    if finish_type(rec) == "length":
        flags += "  finish=length"
    if rec.get("stopped_repeat"):
        flags += "  stopped_repeat=yes"
    print(
        f"       pred={rec.get('extracted_pred')!r}  "
        f"gold={rec.get('extracted_gold')!r}  "
        f"status={rec.get('answer_status')}{flags}"
    )


def prepare_items(
    rows: list[dict[str, Any]], shots: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        question = item.get("question") or ""
        item["prompt"] = build_prompt(question, shots)
        prepared.append(item)
    return prepared


async def abort_generate(session: Any, abort_url: str, rid: str) -> None:
    """Best-effort. A failed abort must not turn the sample into an error."""
    try:
        async with session.post(abort_url, json={"rid": rid, "abort_all": False}) as resp:
            await resp.read()
    except Exception:
        return


async def async_post_generate_stream(
    session: Any,
    url: str,
    payload: dict[str, Any],
    abort_url: str,
) -> tuple[Any, bool]:
    """Stream /generate. Stop and abort when one long line has repeated 3 times."""
    acc = StreamAccumulator()
    stopped_repeat = False
    rid: str | None = None
    st = time.perf_counter()
    async with session.post(url=url, json=payload) as response:
        if response.status != 200:
            detail = await response.text()
            raise RuntimeError(
                f"HTTP {response.status} from {url}: {response.reason or ''}: {detail}"
            )
        async for chunk_bytes in response.content:
            chunk_bytes = chunk_bytes.strip()
            if not chunk_bytes:
                continue
            chunk = chunk_bytes.decode("utf-8", errors="replace")
            payload_obj = parse_sse_payload(chunk)
            if payload_obj is None:
                continue
            acc.add_event(time.perf_counter() - st, payload_obj)
            meta_id = acc.last_meta.get("id")
            if isinstance(meta_id, str) and meta_id:
                rid = meta_id
            if repeated_long_line(acc.output_text):
                stopped_repeat = True
                break
    obs = acc.finish(time.perf_counter() - st)
    if stopped_repeat and rid is not None:
        await abort_generate(session, abort_url, rid)
    return obs, stopped_repeat


async def async_run_one(
    session: Any,
    generate_url: str,
    abort_url: str,
    item: dict[str, Any],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "text": item["prompt"],
        "sampling_params": sampling,
        "stream": True,
    }
    t0 = time.perf_counter()
    try:
        obs, stopped_repeat = await async_post_generate_stream(
            session, generate_url, payload, abort_url
        )
        result = {
            "text": obs.text,
            "output_ids": obs.output_ids,
            "meta_info": obs.meta_info,
        }
        rec = extract_perf(item, result, ttft_s=None, client_e2e_s=None)
        rec = apply_stream_observation(rec, obs)
        rec["stopped_repeat"] = stopped_repeat
        if stopped_repeat:
            # Closing the stream before finish_reason is not a failed request.
            rec.pop("error", None)
    except Exception as exc:  # noqa: BLE001 — keep eval going
        rec = dict(item)
        rec["error"] = str(exc)
        rec["model_output"] = ""
        rec["stopped_repeat"] = False
        elapsed = time.perf_counter() - t0
        rec["e2e_latency"] = elapsed
        _blank_stream_fields(rec)
        rec["client_e2e_s"] = elapsed
    return apply_score(rec)


async def run_benchmark(args: argparse.Namespace) -> int:
    import aiohttp

    generate_url = f"http://{args.host}:{args.port}/generate"
    base_url = f"http://{args.host}:{args.port}"
    abort_url = f"{base_url}/abort_request"
    sampling = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "stop": [STOP_STRING],
    }
    if args.num_shots:
        shots = load_jsonl(args.train_file, args.num_shots)
        if len(shots) < args.num_shots:
            raise SystemExit(
                f"{args.train_file} has {len(shots)} examples, "
                f"fewer than --num-shots {args.num_shots}"
            )
    else:
        shots = []
    dataset = prepare_items(load_jsonl(args.data_file, args.max_items), shots)
    if not dataset:
        raise SystemExit(f"no items in {args.data_file}")
    indexed = list(enumerate(dataset))
    conc = args.max_concurrency if args.max_concurrency else "not set"
    rate_s = "inf" if args.request_rate == float("inf") else str(args.request_rate)
    print(
        f"loaded {len(dataset)} items from {args.data_file}  "
        f"shots={len(shots)} from {args.train_file}"
    )
    print(
        f"target={generate_url}  traffic=open  request_rate={rate_s}  "
        f"max_concurrency={conc}  max_new_tokens={args.max_new_tokens}"
    )
    snapshot = server_snapshot(base_url, args.timeout)
    warn_eval_conditions(snapshot, args.max_concurrency, bool(args.flush_cache))
    conditions = spec_condition_columns(snapshot)
    run_id, args.output_jsonl, args.output_csv, manifest_path = create_run_paths(
        args.output_jsonl, args.output_csv
    )
    print(f"run_id={run_id}  manifest={manifest_path}")

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(
        timeout=timeout, read_bufsize=AIOHTTP_READ_BUFSIZE
    ) as session:
        if not args.no_warmup:
            print("=== warmup (first item, discarded) ===")
            warmup = await async_run_one(
                session, generate_url, abort_url, dataset[0], sampling
            )
            if warmup.get("error"):
                raise RuntimeError(f"Warmup failed: {warmup['error']}")
            print("warmup ok")
            if args.flush_cache:
                flush_cache(base_url, args.timeout)

        semaphore = (
            asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
        )

        async def limited(idx: int, item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            if semaphore is None:
                rec = await async_run_one(
                    session, generate_url, abort_url, item, sampling
                )
            else:
                async with semaphore:
                    rec = await async_run_one(
                        session, generate_url, abort_url, item, sampling
                    )
            return idx, rec

        print("=== main run ===")
        t0 = time.perf_counter()
        tasks: list[asyncio.Task[tuple[int, dict[str, Any]]]] = []
        async for idx, item in get_request(indexed, args.request_rate):
            tasks.append(asyncio.create_task(limited(idx, item)))
        paired = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

    paired_sorted = sorted(paired, key=lambda pair: pair[0])
    records: list[dict[str, Any]] = []
    for idx, rec in paired_sorted:
        rec["global_index"] = idx
        print_gsm8k_item(idx, rec)
        append_jsonl(args.output_jsonl, rec)
        records.append(rec)

    stats = summarize_serving(records, wall)
    accuracy = summarize_accuracy(records)
    test_time = time.strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "Test Time": test_time,
        "request_rate": rate_s,
        "max_concurrency": args.max_concurrency if args.max_concurrency else "",
        "traffic": "open",
        **conditions,
    }
    row.update(gsm8k_csv_values(stats, accuracy))
    write_csv_row(args.output_csv, row)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "server": snapshot,
                "num_shots": args.num_shots,
                "train_file": str(args.train_file),
                "data_file": str(args.data_file),
                "results_jsonl": str(args.output_jsonl),
                "summary_csv": str(args.output_csv),
                "summary": stats,
                "accuracy": accuracy,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(format_serving_summary(stats, wall))
    print(format_accuracy(accuracy))
    print(f"wrote {args.output_jsonl}")
    print(f"wrote {args.output_csv}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSM8K serving evaluation (open-loop, COT, #### accuracy)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Arrivals per second. inf sends all requests immediately (burst). "
        "Finite values use a Poisson process. Default: inf.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Cap on in-flight requests (asyncio.Semaphore). Default: unlimited.",
    )
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--num-shots", type=int, default=4)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_RESULT_DIR / "GSM8K_spectre_serving_results.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULT_DIR / "GSM8K_spectre_serving_statistics.csv",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        help="POST /flush_cache after warmup, once before the main run",
    )
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for Poisson arrivals")
    parser.add_argument(
        "--rescore-jsonl",
        type=Path,
        default=None,
        help="Rescore an existing jsonl without calling the server. "
        "Writes <stem>.rescored.jsonl beside the input.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rescore_jsonl is not None:
        src = args.rescore_jsonl
        dest = src.with_name(f"{src.stem}.rescored{src.suffix}")
        return rescore_jsonl(src, dest)
    if args.max_items < 1:
        raise SystemExit("--max-items must be >= 1")
    if args.max_concurrency is not None and args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")
    if args.request_rate <= 0:
        raise SystemExit("--request-rate must be > 0")
    if args.num_shots < 0:
        raise SystemExit("--num-shots must be >= 0")
    random.seed(args.seed)
    return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    raise SystemExit(main())
