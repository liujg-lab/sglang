#!/usr/bin/env python3
"""SPECTRE 纯文本 serving-style 评测客户端（开放到达 + 流式百分位）。

功能
    HTTP 客户端，打已启动的 SPECTRE Target（默认 :30000），不拉起服务。
    与 eval_spectre_rsteller_serving.py 相同的开放到达（Poisson / burst）、
    全程 stream 计时，以及 E2E/TTFT/ITL/TPOT 百分位；数据改为 ShareGPT
    jsonl，无 image_data。结束后调用 eval_rsteller_metrics.py 补 BLEU / ROUGE。

    不要与闭式脚本的 Decode tok/s 直接横比：本脚本吞吐分母是整场墙钟。

到达模型
    --request-rate inf（默认）  立刻把全部请求发出（burst）。
    --request-rate R            Poisson：相邻请求间隔 ~ Exp(R)。
    --max-concurrency N         asyncio.Semaphore，只限制在飞请求数。

流式计时
    始终 stream=true。首个非空 text chunk 的墙钟为 TTFT（prefill_s）。
    后续按 completion_tokens 增量均摊 ITL（与 sglang.bench_serving native 相同）。
    最后一个 chunk 的 meta_info 交给 extract_perf，保留投机字段。
    不设置 ignore_eos。

数据集
    JSONL。每条 ShareGPT 对话至少两轮；只用前两轮：
      question       第一条 human.value
      ground_truth   第一条 gpt.value
    结果 jsonl 不回写完整 conversations。

jsonl 每条字段
    id / difficulty / question / ground_truth，外加 extract_perf 字段以及
      itl / ttft / global_index
    error HTTP 失败或缺 prompt 时有；精度脚本会跳过。

单条时间轴（客户端；E2E 优先用服务端 meta_info.e2e_latency）
    st 发出 HTTP
      |-- TTFT / Prefill：到第一个非空 text chunk --|-- decode（若干 ITL）--|
      |<---------------- E2E ------------------------------------------------>|
    ITL  从第二个 token 起；一个 SSE chunk 含多个 token 时，间隔按 Δcompletion_tokens 均摊
    TPOT  (E2E - TTFT) / (tokens - 1)，该条去掉首 token 后的平均每 token 时间

CSV 汇总列（只统计无 error 的成功条；百分位为排序后线性插值）
    Test Time                     评测结束墙钟
    request_rate                  发起速率；inf 为 burst
    max_concurrency               在飞上限；空表示不限制
    traffic                       固定为 open
    Total Items                   实际跑的条数（含失败）
    Total Generated Tokens        成功条 completion_tokens 之和

    Avg Prefill (ms)              各条客户端 TTFT 的算术平均 ×1000
                                  inf+concurrency 时含 semaphore 排队，不只是 GPU prefill
    Decode tok/s (wall)           sum(completion_tokens) / 整场墙钟
                                  墙钟 = 主跑 create_task 起到 gather 结束（不含 warmup）
                                  有限 request-rate 时分母含空档，会低于饱和吞吐
    Request/s                     成功条数 / 整场墙钟（完成 QPS，不是 --request-rate）
    Concurrency                   sum(E2E) / 整场墙钟（Little's Law，平均在途请求数）

    Mean/Median/P90/P99 E2E (ms)  各条 e2e_latency 的均值 / 中位数 / 90 / 99 分位 ×1000
                                  优先服务端 meta_info.e2e_latency，缺则用客户端墙钟
    Mean/P99 TTFT (ms)            各条 prefill_s（客户端首 token）的均值 / 99 分位 ×1000
    Mean/P99 ITL (ms)             所有成功条的所有 token 间隔摊平后均值 / 99 分位 ×1000
                                  长请求权重大；与 TPOT（按条等权）不同
    Mean TPOT (ms)                各条 (E2E - TTFT) / (tokens - 1) 再算术平均 ×1000
                                  仅 tokens>1；E2E 用服务端、TTFT 用客户端时时钟可能略有偏差

    Avg Accept Length             spec_accept_length 均值（toks / verify）
    Avg Accept Rate               spec_accept_rate 均值
    Avg Draft Tokens per Verify   draft_tokens_per_verify 均值
    Total Verify                  spec_verify_ct 之和
    Avg BLEU / ROUGE-1/2/L        本脚本先留空，精度脚本补上

运行（仓库根目录，Target + Draft 已启动）
    python test_sglang_liujg/evaluation/eval_spectre_text_serving.py \
      --port 30000 --max-items 100 --request-rate inf --max-concurrency 4
    python test_sglang_liujg/evaluation/eval_spectre_text_serving.py \
      --port 30000 --max-items 100 --request-rate 8 --max-concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator

import aiohttp

from eval_spectre_rsteller import (
    DEFAULT_RESULT_DIR,
    _fmt,
    append_jsonl,
    call_metrics_script,
    extract_perf,
    flush_cache,
    mean,
    print_item,
)

DEFAULT_DATA = Path(
    "/home/liujg/workspace/spec_qwen_tree_socket/cascade_pilot/data/text_only.jsonl"
)

CSV_FIELDS = [
    "Test Time",
    "request_rate",
    "max_concurrency",
    "traffic",
    "Total Items",
    "Total Generated Tokens",
    "Avg Prefill (ms)",
    "Decode tok/s (wall)",
    "Request/s",
    "Concurrency",
    "Mean E2E (ms)",
    "Median E2E (ms)",
    "P90 E2E (ms)",
    "P99 E2E (ms)",
    "Mean TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean ITL (ms)",
    "P99 ITL (ms)",
    "Mean TPOT (ms)",
    "Avg Accept Length",
    "Avg Accept Rate",
    "Avg Draft Tokens per Verify",
    "Total Verify",
    "Avg BLEU",
    "Avg ROUGE-1",
    "Avg ROUGE-2",
    "Avg ROUGE-L",
]

AIOHTTP_READ_BUFSIZE = 10 * 1024**2  # 10 MB, same as bench_serving


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


def _remove_sse_prefix(chunk: str) -> str:
    if chunk.startswith("data: "):
        return chunk[6:]
    if chunk.startswith("data:"):
        return chunk[5:].lstrip()
    return chunk


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (numpy.percentile default style)."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _ms(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 1000.0


def _csv_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


async def get_request(
    items: list[tuple[int, dict[str, Any]]],
    request_rate: float,
) -> AsyncGenerator[tuple[int, dict[str, Any]], None]:
    """Open-loop arrivals: burst if rate is inf, else Poisson Exp(rate)."""
    for idx, item in items:
        yield idx, item
        if request_rate == float("inf"):
            continue
        await asyncio.sleep(random.expovariate(request_rate))


async def async_post_generate_stream(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], float, list[float], float]:
    """Stream /generate. Return (result, ttft_s, itl_list, client_e2e_s)."""
    generated_text = ""
    last_meta: dict[str, Any] = {}
    ttft = 0.0
    itl: list[float] = []
    st = time.perf_counter()
    most_recent = st
    last_output_len = 0

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
            chunk = _remove_sse_prefix(chunk_bytes.decode("utf-8", errors="replace"))
            if chunk == "[DONE]":
                continue
            data = json.loads(chunk)
            if "error" in data:
                raise RuntimeError(f"server error: {data['error']}")
            if data.get("meta_info"):
                last_meta = data["meta_info"]
            if not ("text" in data and data["text"]):
                continue
            timestamp = time.perf_counter()
            generated_text = data["text"]
            output_len = int((last_meta or {}).get("completion_tokens") or 0)
            if ttft == 0.0:
                ttft = timestamp - st
            else:
                num_new_tokens = output_len - last_output_len
                if num_new_tokens <= 0:
                    continue
                chunk_gap = timestamp - most_recent
                adjust_itl = chunk_gap / num_new_tokens
                itl.extend([adjust_itl] * num_new_tokens)
            most_recent = timestamp
            last_output_len = output_len

    client_e2e = time.perf_counter() - st
    if not generated_text and not last_meta:
        raise RuntimeError("empty streaming generate response")
    result = {"text": generated_text, "meta_info": last_meta}
    return result, ttft if ttft else client_e2e, itl, client_e2e


async def async_run_one(
    session: aiohttp.ClientSession,
    generate_url: str,
    item: dict[str, Any],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    question = item.get("question") or ""
    if not str(question).strip():
        rec = dict(item)
        rec["error"] = "empty question"
        rec["model_output"] = ""
        rec["itl"] = []
        rec["ttft"] = None
        return rec
    payload = {
        "text": build_text_prompt(str(question)),
        "sampling_params": sampling,
        "stream": True,
    }
    t0 = time.perf_counter()
    try:
        result, ttft, itl, client_e2e = await async_post_generate_stream(
            session, generate_url, payload
        )
        rec = extract_perf(item, result, ttft_s=ttft, client_e2e_s=client_e2e)
        rec["itl"] = itl
        rec["ttft"] = ttft
        return rec
    except Exception as e:  # noqa: BLE001 — keep eval going
        rec = dict(item)
        rec["error"] = str(e)
        rec["model_output"] = ""
        rec["e2e_latency"] = time.perf_counter() - t0
        rec["itl"] = []
        rec["ttft"] = None
        return rec


def summarize_serving(records: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    ok = [r for r in records if not r.get("error")]
    toks = [
        r["completion_tokens"]
        for r in ok
        if isinstance(r.get("completion_tokens"), (int, float))
    ]
    prefills = [r["prefill_s"] for r in ok if isinstance(r.get("prefill_s"), (int, float))]
    e2es = [r["e2e_latency"] for r in ok if isinstance(r.get("e2e_latency"), (int, float))]
    ttfts = [r["prefill_s"] for r in ok if isinstance(r.get("prefill_s"), (int, float))]
    itls: list[float] = []
    tpots: list[float] = []
    for r in ok:
        chunk_itl = r.get("itl") or []
        if isinstance(chunk_itl, list):
            itls.extend(x for x in chunk_itl if isinstance(x, (int, float)))
        output_len = r.get("completion_tokens")
        latency = r.get("e2e_latency")
        ttft = r.get("prefill_s")
        if (
            isinstance(output_len, (int, float))
            and output_len > 1
            and isinstance(latency, (int, float))
            and isinstance(ttft, (int, float))
        ):
            tpots.append((latency - ttft) / (output_len - 1))
    alens = [
        r["spec_accept_length"]
        for r in ok
        if isinstance(r.get("spec_accept_length"), (int, float))
    ]
    rates = [
        r["spec_accept_rate"]
        for r in ok
        if isinstance(r.get("spec_accept_rate"), (int, float))
    ]
    drafts = [
        r["draft_tokens_per_verify"]
        for r in ok
        if isinstance(r.get("draft_tokens_per_verify"), (int, float))
    ]
    verifies = [
        r["spec_verify_ct"]
        for r in ok
        if isinstance(r.get("spec_verify_ct"), (int, float))
    ]
    total_toks = sum(toks)
    n_ok = len(ok)
    return {
        "n_ok": n_ok,
        "n_total": len(records),
        "total_tokens": total_toks,
        "avg_prefill_ms": _ms(mean(prefills)),
        "decode_tok_s": (total_toks / wall_s) if wall_s > 0 and total_toks else None,
        "request_s": (n_ok / wall_s) if wall_s > 0 and n_ok else None,
        "concurrency": (sum(e2es) / wall_s) if wall_s > 0 and e2es else None,
        "mean_e2e_ms": _ms(mean(e2es)),
        "median_e2e_ms": _ms(statistics.median(e2es) if e2es else None),
        "p90_e2e_ms": _ms(percentile(e2es, 90)),
        "p99_e2e_ms": _ms(percentile(e2es, 99)),
        "mean_ttft_ms": _ms(mean(ttfts)),
        "p99_ttft_ms": _ms(percentile(ttfts, 99)),
        "mean_itl_ms": _ms(mean(itls)),
        "p99_itl_ms": _ms(percentile(itls, 99)),
        "mean_tpot_ms": _ms(mean(tpots)),
        "avg_alen": mean(alens),
        "avg_accept": mean(rates),
        "avg_draft_per_verify": mean(drafts),
        "total_verify": sum(verifies) if verifies else 0.0,
    }


async def run_benchmark(args: argparse.Namespace) -> int:
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

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("", encoding="utf-8")

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
        print_item(idx, rec)
        append_jsonl(args.output_jsonl, rec)
        records.append(rec)

    stats = summarize_serving(records, wall)
    test_time = time.strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "Test Time": test_time,
        "request_rate": rate_s,
        "max_concurrency": args.max_concurrency if args.max_concurrency else "",
        "traffic": "open",
        "Total Items": stats["n_total"],
        "Total Generated Tokens": f"{stats['total_tokens']:.0f}",
        "Avg Prefill (ms)": _csv_num(stats["avg_prefill_ms"]),
        "Decode tok/s (wall)": _csv_num(stats["decode_tok_s"]),
        "Request/s": _csv_num(stats["request_s"]),
        "Concurrency": _csv_num(stats["concurrency"]),
        "Mean E2E (ms)": _csv_num(stats["mean_e2e_ms"]),
        "Median E2E (ms)": _csv_num(stats["median_e2e_ms"]),
        "P90 E2E (ms)": _csv_num(stats["p90_e2e_ms"]),
        "P99 E2E (ms)": _csv_num(stats["p99_e2e_ms"]),
        "Mean TTFT (ms)": _csv_num(stats["mean_ttft_ms"]),
        "P99 TTFT (ms)": _csv_num(stats["p99_ttft_ms"]),
        "Mean ITL (ms)": _csv_num(stats["mean_itl_ms"]),
        "P99 ITL (ms)": _csv_num(stats["p99_itl_ms"]),
        "Mean TPOT (ms)": _csv_num(stats["mean_tpot_ms"]),
        "Avg Accept Length": _csv_num(stats["avg_alen"], 4),
        "Avg Accept Rate": _csv_num(stats["avg_accept"], 4),
        "Avg Draft Tokens per Verify": _csv_num(stats["avg_draft_per_verify"], 4),
        "Total Verify": f"{stats['total_verify']:.0f}",
        "Avg BLEU": "",
        "Avg ROUGE-1": "",
        "Avg ROUGE-2": "",
        "Avg ROUGE-L": "",
    }
    write_csv_row(args.output_csv, row)
    print(
        f"summary items={stats['n_ok']}/{stats['n_total']}  "
        f"toks={stats['total_tokens']:.0f}  wall={wall:.3f}s  "
        f"decode_tok/s={_fmt(stats['decode_tok_s'])}  "
        f"req/s={_fmt(stats['request_s'])}  "
        f"concurrency={_fmt(stats['concurrency'])}  "
        f"prefill_ms={_fmt(stats['avg_prefill_ms'], 2)}  "
        f"p99_e2e_ms={_fmt(stats['p99_e2e_ms'], 2)}  "
        f"p99_ttft_ms={_fmt(stats['p99_ttft_ms'], 2)}  "
        f"mean_itl_ms={_fmt(stats['mean_itl_ms'], 2)}  "
        f"accept={_fmt(stats['avg_accept'])}  "
        f"alen={_fmt(stats['avg_alen'])}"
    )
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
