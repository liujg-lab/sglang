#!/usr/bin/env python3
"""SPECTRE 图文 caption 评测客户端（默认 RSTeller）。

功能
    HTTP 客户端，打已启动的 SPECTRE Target（默认 :30000），不拉起服务。
    闭式并发：每个 chunk 同时发出 batch_size 条独立 POST，等全部返回再发下一批。
    逐条把结果追加到 jsonl；全部结束后写/追加汇总 CSV，并 subprocess 调用
    eval_rsteller_metrics.py 补 BLEU / ROUGE。

支持的数据集
    JSON 数组。每条至少包含：
      image          相对 img_dir 的图片路径
      question       用户问题；其中的 <image> 会被去掉
      ground_truth   参考 caption（精度脚本用；本脚本只原样写入 jsonl）
    默认 RSTeller：
      --data-file  .../USFM_UAV_training_data/json/test/test_RSTeller.json
      --img-dir    .../USFM_UAV_training_data
    其它同结构的 caption JSON 也可通过上述两个参数切换。
    Prompt 套 Qwen3-VL chat 模板：
      <|im_start|>user
      <|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>
      <|im_start|>assistant

并发
    batch-size=1  串行；stream=true，第一条 token 的墙钟为 prefill（TTFT）。
    batch-size>=2 线程池同时发 N 条单请求 POST；服务端 continuous batching
                  收成 GPU batch。不是一条 HTTP 里塞 text[]。

jsonl 每条字段（在原始 sample 上追加）
    model_output              生成文本
    completion_tokens         生成 token 数
    e2e_latency               服务端端到端时延（秒）；缺则用客户端墙钟
    prefill_s                 TTFT（秒）；仅 serial stream 有值
    decode_s                  e2e - prefill（无 TTFT 则为 e2e）
    decode_tok_s              该条 completion_tokens / decode_s
    spec_accept_rate          投机接受率（不含 Target bonus）
    spec_accept_length        平均接受长度 = toks / verify
    spec_verify_ct            Target verify 次数
    spec_draft_token_num      草稿 token 总数（verify × (num_draft_tokens-1)）
    draft_tokens_per_verify   每步草稿长度 = spec_draft_token_num / verify
    batch_wall_s / chunk_index / global_index
    error                     缺图或 HTTP 失败时有；精度脚本会跳过

CSV 汇总列含义
    Test Time                     评测结束墙钟
    batch_size / mode             并发度；serial 或 concurrent
    Total Items                   实际跑的条数（含失败）
    Total Generated Tokens        成功条 completion_tokens 之和
    Avg Prefill (ms)              prefill_s 均值×1000；bs>1 无 TTFT 时常为空
    Decode tok/s                  sum(completion_tokens) / 各 chunk 墙钟之和
    Avg Accept Length             spec_accept_length 均值（toks/verify）
    Avg Accept Rate               spec_accept_rate 均值
                                  = 猜中 draft / (verify × (num_draft_tokens-1))
    Avg Draft Tokens per Verify   draft_tokens_per_verify 均值
    Total Verify                  spec_verify_ct 之和
    Avg BLEU / ROUGE-1/2/L        本脚本先留空，精度脚本补上

运行（仓库根目录，Target + Draft 已启动）
    python test_sglang_liujg/evaluation/eval_spectre_rsteller.py \
      --port 30000 --batch-size 1 --max-items 100
    python test_sglang_liujg/evaluation/eval_spectre_rsteller.py \
      --port 30000 --batch-size 4 --max-items 100

参数
    --host / --port           Target HTTP，默认 127.0.0.1:30000
    --batch-size              闭式并发度，默认 1
    --max-items               最多评几条，默认 100
    --max-new-tokens          默认 256
    --temperature             默认 0.0
    --timeout                 单条 HTTP 超时秒，默认 600
    --data-file / --img-dir   数据集 JSON 与图片根目录
    --output-jsonl            默认 test_sglang_liujg/result/RSTeller_spectre_results.jsonl
    --output-csv              默认 test_sglang_liujg/result/RSTeller_spectre_statistics.csv
    --no-warmup               跳过用第一条做的丢弃 warmup
    --flush-cache             warmup 后以及每个 chunk 前 POST /flush_cache
    --skip-metrics            生成结束后不调用精度脚本
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import json
import mimetypes
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = Path(
    "/home_18T/liujg/Hugging_Face/model_and_data_of_project_1/"
    "USFM_UAV_training_data/json/test/test_RSTeller.json"
)
DEFAULT_IMG_DIR = Path(
    "/home_18T/liujg/Hugging_Face/model_and_data_of_project_1/USFM_UAV_training_data"
)
DEFAULT_RESULT_DIR = HERE.parent / "result"
CSV_FIELDS = [
    "Test Time",
    "batch_size",
    "mode",
    "Total Items",
    "Total Generated Tokens",
    "Avg Prefill (ms)",
    "Decode tok/s",
    "Avg Accept Length",
    "Avg Accept Rate",
    "Avg Draft Tokens per Verify",
    "Total Verify",
    "Avg BLEU",
    "Avg ROUGE-1",
    "Avg ROUGE-2",
    "Avg ROUGE-L",
]


def strip_image_tag(question: str) -> str:
    text = question.replace("<image>", "")
    return text.strip()


def build_prompt(question: str) -> str:
    user_text = strip_image_tag(question)
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def image_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def post_json(
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
    method: str = "POST",
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if not raw.strip():
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e


def _as_result_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "error" in raw:
        raise RuntimeError(f"server error: {raw['error']}")
    return [raw]


def parse_sse_chunk(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    return json.loads(data)


def post_generate_stream(
    url: str, payload: dict[str, Any], timeout: float
) -> tuple[dict[str, Any], float]:
    """Return (final generate result, TTFT seconds)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft: float | None = None
    last: dict[str, Any] | None = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace")
            parsed = parse_sse_chunk(line)
            if parsed is None:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            last = parsed
    if last is None:
        raise RuntimeError("empty streaming generate response")
    if "error" in last:
        raise RuntimeError(f"server error: {last['error']}")
    return last, float(ttft if ttft is not None else time.perf_counter() - t0)


def post_generate(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    return _as_result_list(post_json(url, payload, timeout))[0]


def flush_cache(base_url: str, timeout: float) -> None:
    url = f"{base_url}/flush_cache"
    print(f"  flush_cache -> {url}")
    try:
        post_json(url, None, timeout, method="POST")
    except Exception as e:  # noqa: BLE001 — cache flush is best-effort
        print(f"  flush_cache failed: {e}")


def load_dataset(path: Path, max_items: int | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(raw)}")
    if max_items is not None and max_items > 0:
        return raw[:max_items]
    return raw


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _num(meta: dict[str, Any], key: str) -> float | None:
    value = meta.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def extract_perf(
    item: dict[str, Any],
    generate_result: dict[str, Any],
    *,
    ttft_s: float | None,
    client_e2e_s: float | None,
) -> dict[str, Any]:
    meta = generate_result.get("meta_info") or {}
    text = generate_result.get("text") or ""
    toks = _num(meta, "completion_tokens")
    e2e = _num(meta, "e2e_latency")
    if e2e is None:
        e2e = client_e2e_s
    prefill_s = ttft_s
    decode_s: float | None = None
    if e2e is not None and prefill_s is not None and e2e > prefill_s:
        decode_s = e2e - prefill_s
    elif e2e is not None:
        decode_s = e2e
    decode_tok_s: float | None = None
    if toks is not None and decode_s and decode_s > 0:
        decode_tok_s = toks / decode_s

    draft_num = _num(meta, "spec_draft_token_num")
    verify_ct = _num(meta, "spec_verify_ct")
    draft_per_verify: float | None = None
    if draft_num is not None and verify_ct and verify_ct > 0:
        draft_per_verify = draft_num / verify_ct

    rec = dict(item)
    rec["model_output"] = text
    rec["completion_tokens"] = toks
    rec["e2e_latency"] = e2e
    rec["prefill_s"] = prefill_s
    rec["decode_s"] = decode_s
    rec["decode_tok_s"] = decode_tok_s
    rec["spec_accept_rate"] = _num(meta, "spec_accept_rate")
    rec["spec_accept_length"] = _num(meta, "spec_accept_length")
    rec["spec_verify_ct"] = verify_ct
    rec["spec_draft_token_num"] = draft_num
    rec["draft_tokens_per_verify"] = draft_per_verify
    rec["spec_accept_token_num"] = _num(meta, "spec_accept_token_num")
    return rec


def one_payload(prompt: str, data_uri: str, sampling: dict[str, Any], stream: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": prompt,
        "image_data": data_uri,
        "sampling_params": sampling,
    }
    if stream:
        payload["stream"] = True
    return payload


def prepare_request(
    item: dict[str, Any], img_dir: Path
) -> tuple[str, str] | tuple[None, str]:
    image_rel = item.get("image") or ""
    img_path = img_dir / image_rel
    if not image_rel or not img_path.is_file():
        return None, f"image not found: {img_path}"
    question = item.get("question") or ""
    return build_prompt(question), image_data_uri(img_path)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_item(idx: int, rec: dict[str, Any]) -> None:
    label = rec.get("image") or rec.get("id") or str(idx)
    text = (rec.get("model_output") or "").replace("\n", " ").strip()
    if len(text) > 80:
        text = text[:77] + "..."
    err = rec.get("error")
    if err:
        print(f"  [{idx}] {label}  ERROR {err}")
        return
    print(
        f"  [{idx}] toks={_fmt(rec.get('completion_tokens'), 0):>4s}  "
        f"prefill={_fmt(rec.get('prefill_s')):>6s}s  "
        f"e2e={_fmt(rec.get('e2e_latency')):>6s}s  "
        f"accept={_fmt(rec.get('spec_accept_rate')):>6s}  "
        f"alen={_fmt(rec.get('spec_accept_length')):>6s}  "
        f"verify={_fmt(rec.get('spec_verify_ct'), 0):>4s}  "
        f"text={text!r}"
    )


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def run_one(
    generate_url: str,
    item: dict[str, Any],
    img_dir: Path,
    sampling: dict[str, Any],
    timeout: float,
    stream: bool,
) -> dict[str, Any]:
    prompt, extra = prepare_request(item, img_dir)
    if prompt is None:
        rec = dict(item)
        rec["error"] = extra
        rec["model_output"] = ""
        return rec
    payload = one_payload(prompt, extra, sampling, stream=stream)
    t0 = time.perf_counter()
    try:
        if stream:
            result, ttft = post_generate_stream(generate_url, payload, timeout)
            client_e2e = time.perf_counter() - t0
            return extract_perf(item, result, ttft_s=ttft, client_e2e_s=client_e2e)
        result = post_generate(generate_url, payload, timeout)
        client_e2e = time.perf_counter() - t0
        return extract_perf(item, result, ttft_s=None, client_e2e_s=client_e2e)
    except Exception as e:  # noqa: BLE001 — keep eval going
        rec = dict(item)
        rec["error"] = str(e)
        rec["model_output"] = ""
        rec["e2e_latency"] = time.perf_counter() - t0
        return rec


def run_chunk_concurrent(
    generate_url: str,
    chunk: list[tuple[int, dict[str, Any]]],
    img_dir: Path,
    sampling: dict[str, Any],
    timeout: float,
) -> tuple[float, list[tuple[int, dict[str, Any]]]]:
    n = len(chunk)
    out: list[tuple[int, dict[str, Any]] | None] = [None] * n

    def work(local_i: int) -> tuple[int, dict[str, Any]]:
        idx, item = chunk[local_i]
        rec = run_one(generate_url, item, img_dir, sampling, timeout, stream=False)
        return local_i, rec

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(work, i) for i in range(n)]
        for fut in concurrent.futures.as_completed(futs):
            local_i, rec = fut.result()
            out[local_i] = (chunk[local_i][0], rec)
    wall = time.perf_counter() - t0
    results = [item for item in out if item is not None]
    return wall, results


def write_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def summarize(records: list[dict[str, Any]], total_wall_s: float) -> dict[str, Any]:
    ok = [r for r in records if not r.get("error")]
    toks = [r["completion_tokens"] for r in ok if isinstance(r.get("completion_tokens"), (int, float))]
    prefills = [r["prefill_s"] for r in ok if isinstance(r.get("prefill_s"), (int, float))]
    alens = [r["spec_accept_length"] for r in ok if isinstance(r.get("spec_accept_length"), (int, float))]
    rates = [r["spec_accept_rate"] for r in ok if isinstance(r.get("spec_accept_rate"), (int, float))]
    drafts = [
        r["draft_tokens_per_verify"]
        for r in ok
        if isinstance(r.get("draft_tokens_per_verify"), (int, float))
    ]
    verifies = [r["spec_verify_ct"] for r in ok if isinstance(r.get("spec_verify_ct"), (int, float))]
    total_toks = sum(toks)
    avg_prefill_ms = (mean(prefills) * 1000.0) if prefills else None
    decode_tok_s = (total_toks / total_wall_s) if total_wall_s > 0 and total_toks else None
    return {
        "n_ok": len(ok),
        "n_total": len(records),
        "total_tokens": total_toks,
        "avg_prefill_ms": avg_prefill_ms,
        "decode_tok_s": decode_tok_s,
        "avg_alen": mean(alens),
        "avg_accept": mean(rates),
        "avg_draft_per_verify": mean(drafts),
        "total_verify": sum(verifies) if verifies else 0.0,
    }


def call_metrics_script(jsonl: Path, csv_path: Path) -> None:
    script = HERE / "eval_rsteller_metrics.py"
    cmd = [sys.executable, str(script), "--jsonl", str(jsonl), "--csv", str(csv_path)]
    print(f"running accuracy script: {' '.join(cmd)}")
    try:
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            print(f"accuracy script exited {completed.returncode}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 — perf CSV already written
        print(f"accuracy script failed: {e}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPECTRE RSTeller evaluation client")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=1, help="Closed-loop concurrency")
    p.add_argument("--max-items", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    p.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_RESULT_DIR / "RSTeller_spectre_results.jsonl",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULT_DIR / "RSTeller_spectre_statistics.csv",
    )
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument(
        "--flush-cache",
        action="store_true",
        help="POST /flush_cache after warmup and before each chunk",
    )
    p.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Do not call eval_rsteller_metrics.py after generation",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.max_items < 1:
        raise SystemExit("--max-items must be >= 1")

    mode = "serial" if args.batch_size == 1 else "concurrent"
    generate_url = f"http://{args.host}:{args.port}/generate"
    base_url = f"http://{args.host}:{args.port}"
    sampling = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }

    dataset = load_dataset(args.data_file, args.max_items)
    print(f"loaded {len(dataset)} items from {args.data_file}")
    print(
        f"target={generate_url}  batch_size={args.batch_size}  mode={mode}  "
        f"max_new_tokens={args.max_new_tokens}"
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("", encoding="utf-8")

    if not args.no_warmup and dataset:
        print("=== warmup (first item, discarded) ===")
        run_one(
            generate_url,
            dataset[0],
            args.img_dir,
            sampling,
            args.timeout,
            stream=False,
        )
        if args.flush_cache:
            flush_cache(base_url, args.timeout)

    records: list[dict[str, Any]] = []
    total_wall = 0.0
    indexed = list(enumerate(dataset))
    groups = chunks(indexed, args.batch_size)
    n_groups = len(groups)

    for g_i, group in enumerate(groups):
        print(f"=== chunk {g_i + 1}/{n_groups}  n={len(group)} ===")
        if args.flush_cache:
            flush_cache(base_url, args.timeout)

        if args.batch_size == 1:
            idx, item = group[0]
            t0 = time.perf_counter()
            rec = run_one(
                generate_url,
                item,
                args.img_dir,
                sampling,
                args.timeout,
                stream=True,
            )
            wall = time.perf_counter() - t0
            chunk_results = [(idx, rec)]
        else:
            wall, chunk_results = run_chunk_concurrent(
                generate_url, group, args.img_dir, sampling, args.timeout
            )

        total_wall += wall
        chunk_recs = []
        for idx, rec in chunk_results:
            rec["batch_wall_s"] = wall
            rec["chunk_index"] = g_i
            rec["global_index"] = idx
            print_item(idx, rec)
            append_jsonl(args.output_jsonl, rec)
            records.append(rec)
            chunk_recs.append(rec)
        toks = sum(
            r["completion_tokens"]
            for r in chunk_recs
            if isinstance(r.get("completion_tokens"), (int, float))
        )
        print(f"  chunk wall={wall:.3f}s  toks={toks}  tok/s={toks / wall if wall else 0:.2f}")

    stats = summarize(records, total_wall)
    test_time = time.strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "Test Time": test_time,
        "batch_size": args.batch_size,
        "mode": mode,
        "Total Items": stats["n_total"],
        "Total Generated Tokens": f"{stats['total_tokens']:.0f}",
        "Avg Prefill (ms)": (
            f"{stats['avg_prefill_ms']:.2f}" if stats["avg_prefill_ms"] is not None else ""
        ),
        "Decode tok/s": (
            f"{stats['decode_tok_s']:.2f}" if stats["decode_tok_s"] is not None else ""
        ),
        "Avg Accept Length": (
            f"{stats['avg_alen']:.4f}" if stats["avg_alen"] is not None else ""
        ),
        "Avg Accept Rate": (
            f"{stats['avg_accept']:.4f}" if stats["avg_accept"] is not None else ""
        ),
        "Avg Draft Tokens per Verify": (
            f"{stats['avg_draft_per_verify']:.4f}"
            if stats["avg_draft_per_verify"] is not None
            else ""
        ),
        "Total Verify": f"{stats['total_verify']:.0f}",
        "Avg BLEU": "",
        "Avg ROUGE-1": "",
        "Avg ROUGE-2": "",
        "Avg ROUGE-L": "",
    }
    write_csv_row(args.output_csv, row)
    print(
        f"summary items={stats['n_ok']}/{stats['n_total']}  "
        f"toks={stats['total_tokens']:.0f}  wall={total_wall:.3f}s  "
        f"decode_tok/s={_fmt(stats['decode_tok_s'])}  "
        f"prefill_ms={_fmt(stats['avg_prefill_ms'], 2)}  "
        f"accept={_fmt(stats['avg_accept'])}  "
        f"alen={_fmt(stats['avg_alen'])}  "
        f"draft/verify={_fmt(stats['avg_draft_per_verify'])}"
    )
    print(f"wrote {args.output_jsonl}")
    print(f"wrote {args.output_csv}")

    if not args.skip_metrics:
        call_metrics_script(args.output_jsonl, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
