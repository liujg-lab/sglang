#!/usr/bin/env python3
"""VL 图文串行长生成：打已启动的 SGLang / SPECTRE Target（默认 :30000）。

每次 POST 一路（batch size=1）+ 一张图，N 次串行。用来看单请求 SPECTRE
（accept / alen / verify），不被 HTTP batch 或并发打混。无 speedup / hit。

流程（默认）：
  1. warmup     第一张图 JIT / 编译
  2. flush_cache
  3. sequential N 次串行 POST，每次一条 text + 一张 image_data

prompt 要求多句描述颜色和字母，默认 max_new_tokens=128，避免短答过早 EOS。
不默认 min_new_tokens（SPECTRE AR 与 penalizer 可能不兼容）。

每条输出：
  expect   图上字母/颜色，对照 text 看是否串台
  toks     completion_tokens
  lat      该请求服务端 e2e 时延（秒）
  accept   投机草稿接受率 = 猜中的 draft token / (verify × (num-draft-tokens-1))
  alen     toks / verify；上限约等于 --speculative-num-draft-tokens
  verify   Target verify 次数（spec_verify_ct）
  text     生成文本

summary：returned / wall / avg_accept / avg_alen / avg_e2e

Usage（Target + Draft 已启动）：
  python test_vl_serial.py --port 30000
  python test_vl_serial.py --num-requests 1 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROMPT_QUESTION = (
    "Describe this image in several sentences. Name the background color and "
    "the letter, then add two more sentences about the composition. Do not "
    "stop after the first sentence."
)
QWEN3_VL_PROMPT = (
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    f"{PROMPT_QUESTION}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("red", (220, 40, 40)),
    ("green", (40, 180, 60)),
    ("blue", (40, 80, 220)),
    ("yellow", (240, 210, 40)),
    ("orange", (240, 140, 30)),
    ("purple", (150, 50, 200)),
    ("cyan", (40, 200, 210)),
    ("pink", (230, 90, 160)),
]

IMAGE_DIR = Path("/tmp/sglang_vl_serial")


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_image(index: int, image_size: int) -> tuple[str, str, str]:
    """Return (label, file_path, data_uri) for a solid-color lettered PNG."""
    color_name, rgb = COLORS[index % len(COLORS)]
    letter = chr(ord("A") + (index % 26))
    label = f"{color_name}+{letter}"

    img = Image.new("RGB", (image_size, image_size), rgb)
    draw = ImageDraw.Draw(img)
    font = _font(max(image_size // 2, 16))
    text_bbox = draw.textbbox((0, 0), letter, font=font)
    tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    xy = ((image_size - tw) / 2 - text_bbox[0], (image_size - th) / 2 - text_bbox[1])
    ink = (20, 20, 20) if sum(rgb) > 420 else (250, 250, 250)
    draw.text(xy, letter, fill=ink, font=font)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGE_DIR / f"{index:02d}_{color_name}_{letter}.png"
    img.save(path, format="PNG")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return label, str(path), f"data:image/png;base64,{b64}"


def build_prompt() -> str:
    return QWEN3_VL_PROMPT


def post_json(
    url: str, payload: dict[str, Any] | None, timeout: float, method: str = "POST"
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


def print_item(idx: int, label: str, item: dict[str, Any]) -> None:
    meta = item.get("meta_info") or {}
    text = (item.get("text") or "").replace("\n", " ").strip()
    print(
        f"  [{idx}] expect={label:12s}  "
        f"toks={_fmt(meta.get('completion_tokens')):>4s}  "
        f"lat={_fmt(meta.get('e2e_latency')):>7s}s  "
        f"accept={_fmt(meta.get('spec_accept_rate')):>6s}  "
        f"alen={_fmt(meta.get('spec_accept_length')):>6s}  "
        f"verify={_fmt(meta.get('spec_verify_ct')):>4s}  "
        f"text={text!r}"
    )


def summarize(mode: str, wall_s: float, items: list[dict[str, Any]], n_expected: int) -> None:
    metas = [it.get("meta_info") or {} for it in items]
    rates = [m["spec_accept_rate"] for m in metas if isinstance(m.get("spec_accept_rate"), (int, float))]
    alens = [
        m["spec_accept_length"]
        for m in metas
        if isinstance(m.get("spec_accept_length"), (int, float))
    ]
    lats = [m["e2e_latency"] for m in metas if isinstance(m.get("e2e_latency"), (int, float))]
    print(
        f"  summary {mode}: returned={len(items)}/{n_expected}  wall={wall_s:.3f}s  "
        f"avg_accept={_fmt(statistics.mean(rates) if rates else None)}  "
        f"avg_alen={_fmt(statistics.mean(alens) if alens else None)}  "
        f"avg_e2e={_fmt(statistics.mean(lats) if lats else None)}s"
    )
    print()


def one_payload(data_uri: str, sampling: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": build_prompt(),
        "image_data": data_uri,
        "sampling_params": sampling,
    }


def run_warmup(
    url: str,
    labels: list[str],
    data_uris: list[str],
    sampling: dict[str, Any],
    timeout: float,
) -> None:
    print("=== warmup (first image, batch size=1) ===")
    t0 = time.perf_counter()
    raw = post_generate(url, one_payload(data_uris[0], sampling), timeout)
    wall = time.perf_counter() - t0
    items = _as_result_list(raw)
    print_item(0, labels[0], items[0])
    summarize("warmup", wall, items, 1)


def run_sequential(
    url: str,
    labels: list[str],
    data_uris: list[str],
    sampling: dict[str, Any],
    timeout: float,
) -> None:
    n = len(labels)
    print(f"=== sequential ({n} serial POSTs, batch size=1) ===")
    t0 = time.perf_counter()
    items: list[dict[str, Any]] = []
    for i in range(n):
        items.append(
            _as_result_list(post_generate(url, one_payload(data_uris[i], sampling), timeout))[0]
        )
    wall = time.perf_counter() - t0
    for i, item in enumerate(items):
        print_item(i, labels[i], item)
    summarize("sequential", wall, items, n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VL serial long-gen smoke test for SGLang / SPECTRE Target (batch size=1)"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument(
        "--num-requests",
        type=int,
        default=4,
        help="Number of serial POSTs, each with one image (default: 4)",
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--min-new-tokens",
        type=int,
        default=0,
        help="Optional floor on generated tokens (0 = unset; SPECTRE AR may not honor this)",
    )
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-warmup", action="store_true", help="Skip the first-image JIT warmup")
    p.add_argument(
        "--no-flush-cache",
        action="store_true",
        help="Do not POST /flush_cache after warmup / before sequential",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_requests < 1:
        raise SystemExit("--num-requests must be >= 1")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be >= 1")
    if args.min_new_tokens < 0:
        raise SystemExit("--min-new-tokens must be >= 0")
    if args.min_new_tokens > args.max_new_tokens:
        raise SystemExit("--min-new-tokens must be <= --max-new-tokens")

    generate_url = f"http://{args.host}:{args.port}/generate"
    base_url = f"http://{args.host}:{args.port}"
    sampling: dict[str, Any] = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.min_new_tokens > 0:
        sampling["min_new_tokens"] = args.min_new_tokens

    labels: list[str] = []
    data_uris: list[str] = []
    paths: list[str] = []
    for i in range(args.num_requests):
        label, path, uri = make_image(i, args.image_size)
        labels.append(label)
        paths.append(path)
        data_uris.append(uri)

    print(f"target: {generate_url}")
    print(
        f"num_requests={args.num_requests}  batch_size=1  "
        f"min_new_tokens={args.min_new_tokens}  max_new_tokens={args.max_new_tokens}"
    )
    print(f"images ({args.image_size}x{args.image_size}) saved under {IMAGE_DIR}:")
    for i, (label, path) in enumerate(zip(labels, paths)):
        print(f"  [{i}] {label}  {path}")
    print()

    do_flush = not args.no_flush_cache
    if not args.no_warmup:
        run_warmup(generate_url, labels, data_uris, sampling, args.timeout)
        maybe_flush(base_url, args.timeout, do_flush)
    else:
        maybe_flush(base_url, args.timeout, do_flush)

    run_sequential(generate_url, labels, data_uris, sampling, args.timeout)


if __name__ == "__main__":
    main()
