#!/usr/bin/env python3
"""纯文本 batch 冒烟：打已启动的 SGLang / SPECTRE Target（默认 :30000）。

Qwen chat 模板、无 vision token。测 scheduler 是否把多路请求打成一批，
并用 sequential 作无 batch 基线。

模式（默认 --mode all）：
  1. single       第一条 prompt，JIT / 编译 warmup
  2. sequential   N 次串行 POST，wall ≈ 各 lat 之和
  3. http-batch   一次 POST，payload 为 text[]
  4. concurrent   N 路并行 POST（更接近真实 serving）

默认在 warmup 之后、每个 batch 模式之前 POST /flush_cache（--no-flush-cache 可关）。
每条 prompt 要求写一段含唯一水果词的短文，用 hit/match 查串台。

每条输出：
  expect   期望出现的关键词
  hit      生成文本是否包含 expect（大小写不敏感）
  toks     completion_tokens，生成 token 数
  lat      该请求服务端 e2e 时延（秒）
  accept   投机草稿接受率 = 猜中的 draft token / (verify × (num-draft-tokens-1))
           不含 Target bonus；全中为 1.0
  alen     平均接受长度 = toks / verify；上限约等于 --speculative-num-draft-tokens
           分子含 prefill 首 token 和 SPECTRE AR fallback，可能略高于上限
  verify   Target verify 次数（spec_verify_ct）
  text     生成文本

summary：
  returned / wall / avg_accept / avg_alen / avg_e2e / match
  http-batch 与 concurrent 另打 speedup = sequential_wall / wall

Usage（Target + Draft 已启动）：
  python test_text_batch.py --port 30000 --batch-size 4
  python test_text_batch.py --mode sequential,http-batch --batch-size 8 --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from typing import Any

WORDS: list[str] = [
    "apple",
    "banana",
    "cherry",
    "date",
    "elderberry",
    "fig",
    "grape",
    "honeydew",
    "kiwi",
    "lemon",
    "mango",
    "nectarine",
    "orange",
    "papaya",
    "quince",
    "raspberry",
]

VALID_MODES: tuple[str, ...] = ("single", "sequential", "http-batch", "concurrent")


def build_prompt(word: str) -> str:
    question = (
        f'Write a short paragraph (about 80-120 words) about {word}. '
        f'Start with the word "{word}" and mention it at least twice. '
        f"Do not mention other fruit names."
    )
    return (
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def expected_word(index: int) -> str:
    return WORDS[index % len(WORDS)]


def text_matches(item: dict[str, Any], expect: str) -> bool:
    text = (item.get("text") or "").lower()
    return expect.lower() in text


def post_json(url: str, payload: dict[str, Any] | None, timeout: float, method: str = "POST") -> Any:
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


def post_generate(url: str, payload: dict[str, Any], timeout: float) -> Any:
    return post_json(url, payload, timeout)


def flush_cache(base_url: str, timeout: float) -> None:
    url = f"{base_url}/flush_cache"
    print(f"  flush_cache -> {url}")
    try:
        post_json(url, None, timeout, method="POST")
    except Exception as e:  # noqa: BLE001 — cache flush is best-effort
        print(f"  flush_cache failed: {e}")


def maybe_flush(base_url: str, timeout: float, enabled: bool) -> None:
    if enabled:
        flush_cache(base_url, timeout)


def _as_result_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "error" in raw:
        raise RuntimeError(f"server error: {raw['error']}")
    return [raw]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_item(idx: int, expect: str, item: dict[str, Any]) -> None:
    meta = item.get("meta_info") or {}
    text = (item.get("text") or "").replace("\n", " ").strip()
    hit = "Y" if text_matches(item, expect) else "N"
    print(
        f"  [{idx}] expect={expect:12s}  hit={hit}  "
        f"toks={_fmt(meta.get('completion_tokens')):>4s}  "
        f"lat={_fmt(meta.get('e2e_latency')):>7s}s  "
        f"accept={_fmt(meta.get('spec_accept_rate')):>6s}  "
        f"alen={_fmt(meta.get('spec_accept_length')):>6s}  "
        f"verify={_fmt(meta.get('spec_verify_ct')):>4s}  "
        f"text={text!r}"
    )


def summarize(
    mode: str,
    wall_s: float,
    items: list[dict[str, Any]],
    expects: list[str],
    n_expected: int,
    sequential_wall: float | None = None,
) -> None:
    metas = [it.get("meta_info") or {} for it in items]
    rates = [m["spec_accept_rate"] for m in metas if isinstance(m.get("spec_accept_rate"), (int, float))]
    alens = [
        m["spec_accept_length"]
        for m in metas
        if isinstance(m.get("spec_accept_length"), (int, float))
    ]
    lats = [m["e2e_latency"] for m in metas if isinstance(m.get("e2e_latency"), (int, float))]
    n_match = sum(
        1
        for it, exp in zip(items, expects)
        if text_matches(it, exp)
    )
    speedup = ""
    if sequential_wall is not None and sequential_wall > 0 and wall_s > 0:
        speedup = f"  speedup={sequential_wall / wall_s:.3f}x"
    print(
        f"  summary {mode}: returned={len(items)}/{n_expected}  wall={wall_s:.3f}s  "
        f"avg_accept={_fmt(statistics.mean(rates) if rates else None)}  "
        f"avg_alen={_fmt(statistics.mean(alens) if alens else None)}  "
        f"avg_e2e={_fmt(statistics.mean(lats) if lats else None)}s  "
        f"match={n_match}/{n_expected}{speedup}"
    )
    print()


def run_single(
    url: str,
    expects: list[str],
    prompts: list[str],
    sampling: dict[str, Any],
    timeout: float,
) -> None:
    print("=== single (warmup, first prompt) ===")
    payload = {
        "text": prompts[0],
        "sampling_params": sampling,
    }
    t0 = time.perf_counter()
    raw = post_generate(url, payload, timeout)
    wall = time.perf_counter() - t0
    items = _as_result_list(raw)
    print_item(0, expects[0], items[0])
    summarize("single", wall, items, expects[:1], 1)


def run_sequential(
    url: str,
    expects: list[str],
    prompts: list[str],
    sampling: dict[str, Any],
    timeout: float,
) -> float:
    n = len(prompts)
    print(f"=== sequential ({n} serial POSTs) ===")
    t0 = time.perf_counter()
    items: list[dict[str, Any]] = []
    for i in range(n):
        payload = {
            "text": prompts[i],
            "sampling_params": sampling,
        }
        items.append(_as_result_list(post_generate(url, payload, timeout))[0])
    wall = time.perf_counter() - t0
    for i, item in enumerate(items):
        print_item(i, expects[i], item)
    summarize("sequential", wall, items, expects, n)
    return wall


def run_http_batch(
    url: str,
    expects: list[str],
    prompts: list[str],
    sampling: dict[str, Any],
    timeout: float,
    sequential_wall: float | None,
) -> None:
    n = len(prompts)
    print(f"=== http-batch (1 POST, batch_size={n}) ===")
    payload = {
        "text": prompts,
        "sampling_params": sampling,
    }
    t0 = time.perf_counter()
    raw = post_generate(url, payload, timeout)
    wall = time.perf_counter() - t0
    items = _as_result_list(raw)
    for i, item in enumerate(items):
        print_item(i, expects[i] if i < len(expects) else "?", item)
    summarize("http-batch", wall, items, expects, n, sequential_wall=sequential_wall)


def run_concurrent(
    url: str,
    expects: list[str],
    prompts: list[str],
    sampling: dict[str, Any],
    timeout: float,
    sequential_wall: float | None,
) -> None:
    n = len(prompts)
    print(f"=== concurrent ({n} parallel POSTs) ===")

    def one(i: int) -> tuple[int, dict[str, Any]]:
        payload = {
            "text": prompts[i],
            "sampling_params": sampling,
        }
        return i, _as_result_list(post_generate(url, payload, timeout))[0]

    t0 = time.perf_counter()
    items: list[dict[str, Any] | None] = [None] * n
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(one, i) for i in range(n)]
        for fut in concurrent.futures.as_completed(futs):
            try:
                i, item = fut.result()
                items[i] = item
            except Exception as e:  # noqa: BLE001 — report and keep other requests
                errors.append(str(e))
    wall = time.perf_counter() - t0
    ok = [it for it in items if it is not None]
    ok_expects = [expects[i] for i, it in enumerate(items) if it is not None]
    for i, item in enumerate(items):
        if item is None:
            print(f"  [{i}] expect={expects[i]:12s}  hit=N  FAILED")
        else:
            print_item(i, expects[i], item)
    for err in errors:
        print(f"  error: {err}")
    summarize("concurrent", wall, ok, ok_expects, n, sequential_wall=sequential_wall)


def parse_modes(raw: str) -> list[str]:
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens or tokens == ["all"]:
        return list(VALID_MODES)
    if "all" in tokens:
        return list(VALID_MODES)
    unknown = [t for t in tokens if t not in VALID_MODES]
    if unknown:
        raise SystemExit(
            f"unknown --mode {unknown}; choose all or a comma-separated subset of {list(VALID_MODES)}"
        )
    return tokens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pure-text batch smoke test for SGLang / SPECTRE Target"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--mode",
        default="all",
        help="all, or comma-separated: single,sequential,http-batch,concurrent",
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--min-new-tokens",
        type=int,
        default=64,
        help="Minimum generated tokens (avoids early EOS; must be <= --max-new-tokens)",
    )
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--no-flush-cache",
        action="store_true",
        help="Do not POST /flush_cache between modes (default: flush after warmup and before each batch mode)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.min_new_tokens < 0:
        raise SystemExit("--min-new-tokens must be >= 0")
    if args.min_new_tokens > args.max_new_tokens:
        raise SystemExit("--min-new-tokens must be <= --max-new-tokens")

    modes = parse_modes(args.mode)
    flush = not args.no_flush_cache
    base_url = f"http://{args.host}:{args.port}"
    url = f"{base_url}/generate"
    sampling = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
    }

    expects = [expected_word(i) for i in range(args.batch_size)]
    prompts = [build_prompt(w) for w in expects]

    print(f"target: {url}")
    print(
        f"batch-size={args.batch_size}  modes={','.join(modes)}  flush_cache={flush}  "
        f"min_new_tokens={args.min_new_tokens}  max_new_tokens={args.max_new_tokens}"
    )
    for i, word in enumerate(expects):
        print(f"  [{i}] expect={word}")
    print()

    if "single" in modes:
        run_single(url, expects, prompts, sampling, args.timeout)

    sequential_wall: float | None = None
    if "sequential" in modes:
        maybe_flush(base_url, args.timeout, flush)
        sequential_wall = run_sequential(url, expects, prompts, sampling, args.timeout)

    if "http-batch" in modes:
        maybe_flush(base_url, args.timeout, flush)
        run_http_batch(
            url, expects, prompts, sampling, args.timeout, sequential_wall
        )

    if "concurrent" in modes:
        maybe_flush(base_url, args.timeout, flush)
        run_concurrent(
            url, expects, prompts, sampling, args.timeout, sequential_wall
        )


if __name__ == "__main__":
    main()
