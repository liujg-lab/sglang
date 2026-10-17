#!/usr/bin/env python3
"""SPECTRE 纯文本 serving-style 评测客户端（开放到达 + 流式百分位）。

功能
    HTTP 客户端，打已启动的 SPECTRE Target（默认 :30000），不拉起服务。
    到达模型和客户端计时与 eval_spectre_rsteller_serving.py / WaterCaption 相同，
    数据改为 ShareGPT jsonl，无 image_data。结束后调用 eval_rsteller_metrics.py
    补 BLEU / ROUGE。不设置 ignore_eos。

    Decode tok/s 只覆盖客户端输出阶段，分子是 N-K。串行时适合比较 AR/SR；
    并发时它是请求级合并速度，不是服务器聚合吞吐。整场墙钟吞吐看
    Output tok/s (wall)。不要和闭式脚本的 chunk 墙钟之和直接横比。

到达模型
    --request-rate inf（默认）  立刻把全部请求发出（burst）。
    --request-rate R            Poisson：相邻请求间隔 ~ Exp(R)。
    --max-concurrency N         asyncio.Semaphore，只限制在飞请求数。

单条时间轴（全部使用客户端 time.perf_counter）
    t_send              t_first                 t_last           t_end
    发起 HTTP ────────── 首次 token 事件 ──────── 最后新增 token ── 响应结束
              TTFT                  Decode Time
    TTFT          t_first - t_send。按 completion_tokens 增长计，不要求 text 非空。
    Decode Time   t_last - t_first。不含 TTFT，不含结束通知到 t_end 的等待。
    Client E2E    t_end - t_send。
    Server E2E    meta_info.e2e_latency，单独保存，不与客户端 TTFT 相减。

数据集
    JSONL。每条 ShareGPT 对话至少两轮；只用前两轮：
      question       第一条 human.value
      ground_truth   第一条 gpt.value
    结果 jsonl 不回写完整 conversations。

jsonl 每条字段
    id / difficulty / question / ground_truth，外加共享 serving 的
    ttft_s / first_text_s / first_chunk_tokens / decode_s / decode_tokens /
    decode_tok_s / client_e2e_s / server_e2e_s / token_events / itl，
    以及 extract_perf 的投机字段和 global_index。
    error HTTP 失败或缺 prompt 时有；精度脚本会跳过。

CSV 列与 WaterCaption serving 相同。旧表头的统计文件会拒绝追加。

运行（仓库根目录，Target + Draft 已启动）
    python test_sglang_liujg/evaluation/eval_spectre_text_serving.py \
      --port 30000 --max-items 100 --request-rate inf --max-concurrency 4
    python test_sglang_liujg/evaluation/eval_spectre_text_serving.py \
      --port 30000 --max-items 100 --request-rate 8 --max-concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from eval_spectre_rsteller import (
    DEFAULT_RESULT_DIR,
    append_jsonl,
    call_metrics_script,
    create_run_paths,
    extract_perf,
    flush_cache,
    server_snapshot,
    spec_condition_columns,
    warn_eval_conditions,
)
from eval_spectre_rsteller_serving import (
    AIOHTTP_READ_BUFSIZE,
    CSV_FIELDS,
    _blank_stream_fields,
    apply_stream_observation,
    async_post_generate_stream,
    format_serving_summary,
    get_request,
    metric_csv_values,
    print_serving_item,
    summarize_serving,
    write_csv_row,
)

DEFAULT_DATA = Path(
    "/home/liujg/workspace/spec_qwen_tree_socket/cascade_pilot/data/text_only.jsonl"
)


def build_text_prompt(question: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _turn_text(turn: Any) -> str | None:
    if not isinstance(turn, dict):
        return None
    value = turn.get("value", turn.get("content"))
    return value if isinstance(value, str) else None


def _first_turn(conversations: list[Any], role: str) -> str | None:
    for turn in conversations:
        if isinstance(turn, dict) and turn.get("from") == role:
            text = _turn_text(turn)
            if text is not None:
                return text
    return None


def parse_text_record(raw: dict[str, Any]) -> dict[str, Any] | None:
    conversations = raw.get("conversations", raw.get("conversation"))
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None
    question = _first_turn(conversations, "human")
    if question is None or not question.strip():
        return None
    ground_truth = _first_turn(conversations, "gpt") or ""
    rec: dict[str, Any] = {
        "question": question,
        "ground_truth": ground_truth,
    }
    if "id" in raw:
        rec["id"] = raw["id"]
    if "difficulty" in raw:
        rec["difficulty"] = raw["difficulty"]
    return rec


def load_text_dataset(path: Path, max_items: int | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if max_items is not None and max_items > 0 and len(items) >= max_items:
                break
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                skipped += 1
                print(
                    f"Warning: line {line_number}: invalid JSON ({exc.msg}); skipped.",
                    file=sys.stderr,
                )
                continue
            if not isinstance(obj, dict):
                skipped += 1
                continue
            rec = parse_text_record(obj)
            if rec is None:
                skipped += 1
                continue
            items.append(rec)
    if skipped:
        print(f"skipped {skipped} invalid jsonl rows in {path}", file=sys.stderr)
    return items


async def async_run_one(
    session: Any,
    generate_url: str,
    item: dict[str, Any],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    question = item.get("question") or ""
    if not str(question).strip():
        rec = dict(item)
        rec["error"] = "empty question"
        rec["model_output"] = ""
        _blank_stream_fields(rec)
        return rec
    payload = {
        "text": build_text_prompt(str(question)),
        "sampling_params": sampling,
        "stream": True,
    }
    t0 = time.perf_counter()
    try:
        obs = await async_post_generate_stream(session, generate_url, payload)
        result = {
            "text": obs.text,
            "output_ids": obs.output_ids,
            "meta_info": obs.meta_info,
        }
        rec = extract_perf(item, result, ttft_s=None, client_e2e_s=None)
        return apply_stream_observation(rec, obs)
    except Exception as e:  # noqa: BLE001 — keep eval going
        rec = dict(item)
        rec["error"] = str(e)
        rec["model_output"] = ""
        elapsed = time.perf_counter() - t0
        rec["e2e_latency"] = elapsed
        _blank_stream_fields(rec)
        rec["client_e2e_s"] = elapsed
        return rec


async def run_benchmark(args: argparse.Namespace) -> int:
    import aiohttp

    generate_url = f"http://{args.host}:{args.port}/generate"
    base_url = f"http://{args.host}:{args.port}"
    sampling = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    dataset = load_text_dataset(args.data_file, args.max_items)
    if not dataset:
        raise SystemExit(f"no valid items loaded from {args.data_file}")
    indexed = list(enumerate(dataset))
    conc = args.max_concurrency if args.max_concurrency else "not set"
    rate_s = "inf" if args.request_rate == float("inf") else str(args.request_rate)
    print(f"loaded {len(dataset)} items from {args.data_file}")
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
        if not args.no_warmup and dataset:
            print("=== warmup (first item, discarded) ===")
            warmup = await async_run_one(session, generate_url, dataset[0], sampling)
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
                rec = await async_run_one(session, generate_url, item, sampling)
            else:
                async with semaphore:
                    rec = await async_run_one(session, generate_url, item, sampling)
            return idx, rec

        print("=== main run ===")
        t0 = time.perf_counter()
        tasks: list[asyncio.Task[tuple[int, dict[str, Any]]]] = []
        async for idx, item in get_request(indexed, args.request_rate):
            tasks.append(asyncio.create_task(limited(idx, item)))
        paired = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

    paired_sorted = sorted(paired, key=lambda x: x[0])
    records: list[dict[str, Any]] = []
    for idx, rec in paired_sorted:
        rec["global_index"] = idx
        print_serving_item(idx, rec)
        append_jsonl(args.output_jsonl, rec)
        records.append(rec)

    stats = summarize_serving(records, wall)
    row = {
        "Test Time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "request_rate": rate_s,
        "max_concurrency": args.max_concurrency if args.max_concurrency else "",
        "traffic": "open",
        **conditions,
    }
    row.update(metric_csv_values(stats))
    write_csv_row(args.output_csv, row)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "server": snapshot,
                "results_jsonl": str(args.output_jsonl),
                "summary_csv": str(args.output_csv),
                "summary": stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(format_serving_summary(stats, wall))
    print(f"wrote {args.output_jsonl}")
    print(f"wrote {args.output_csv}")

    if not args.skip_metrics:
        call_metrics_script(args.output_jsonl, args.output_csv)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPECTRE text-only serving-style evaluation (open-loop)"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Arrivals per second. inf sends all requests immediately (burst). "
        "Finite values use a Poisson process. Default: inf.",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Cap on in-flight requests (asyncio.Semaphore). Default: unlimited.",
    )
    p.add_argument("--max-items", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_RESULT_DIR / "text_only_spectre_serving_results.jsonl",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULT_DIR / "text_only_spectre_serving_statistics.csv",
    )
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument(
        "--flush-cache",
        action="store_true",
        help="POST /flush_cache after warmup, once before the main run",
    )
    p.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Do not call eval_rsteller_metrics.py after generation",
    )
    p.add_argument("--seed", type=int, default=1, help="RNG seed for Poisson arrivals")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_items < 1:
        raise SystemExit("--max-items must be >= 1")
    if args.max_concurrency is not None and args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")
    if args.request_rate <= 0:
        raise SystemExit("--request-rate must be > 0")
    if not args.data_file.is_file():
        raise SystemExit(f"data file not found: {args.data_file}")
    random.seed(args.seed)
    return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    raise SystemExit(main())
