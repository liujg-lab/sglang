#!/usr/bin/env python3
"""VL 图文 batch 冒烟：打已启动的 SGLang / SPECTRE Target（默认 :30000）。

生成带彩色字母的小 PNG，测图文配对（不串台）以及 SPECTRE 投机指标。
无 sequential 基线，故无 speedup / hit 字段。

模式（默认 --mode all）：
  1. single       第一张图 warmup
  2. http-batch   一次 POST，payload 为 text[] + image_data[]
  3. concurrent   N 路并行 POST

默认在 warmup 之后、每个 batch 模式之前 POST /flush_cache（--no-flush-cache 可关）。
VL tokenize / pad 较慢，HTTP 一次打 N 条不等于 GPU 上一次 prefill=N。

每条输出（无 hit；对照 expect 与 text 看是否串台）：
  expect   图上字母，期望出现在生成文本中
  toks     completion_tokens，生成 token 数
  lat      该请求服务端 e2e 时延（秒）
  accept   投机草稿接受率 = 猜中的 draft token / (verify × (num-draft-tokens-1))
           不含 Target bonus；全中为 1.0
  alen     平均接受长度 = toks / verify；上限约等于 --speculative-num-draft-tokens
           分子含 prefill 首 token 和 SPECTRE AR fallback，可能略高于上限
  verify   Target verify 次数（spec_verify_ct）
  text     生成文本

summary：
  returned / wall / avg_accept / avg_alen / avg_e2e
  （无 sequential，故无 speedup / match）

Usage（Target + Draft 已启动）：
  python test_vl_batch.py --port 30000 --batch-size 4
  python test_vl_batch.py --mode concurrent --batch-size 4 --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
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
    "What is the background color and the letter in this image? Answer briefly."
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

IMAGE_DIR = Path("/tmp/sglang_vl_batch")


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


def post_generate(url: str, payload: dict[str, Any], timeout: float) -> Any:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e


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


def run_single(
    url: str,
    labels: list[str],
    data_uris: list[str],
    sampling: dict[str, Any],
    timeout: float,
) -> None:
    print("=== single (warmup, first image) ===")
    payload = {
        "text": build_prompt(),
        "image_data": data_uris[0],
        "sampling_params": sampling,
    }
    t0 = time.perf_counter()
    raw = post_generate(url, payload, timeout)
    wall = time.perf_counter() - t0
    items = _as_result_list(raw)
    print_item(0, labels[0], items[0])
    summarize("single", wall, items, 1)


def run_http_batch(
    url: str,
    labels: list[str],
    data_uris: list[str],
    sampling: dict[str, Any],
    timeout: float,
) -> None:
    n = len(labels)
    print(f"=== http-batch (1 POST, batch_size={n}) ===")
    payload = {
        "text": [build_prompt() for _ in range(n)],
        "image_data": data_uris,
        "sampling_params": sampling,
    }
    t0 = time.perf_counter()
    raw = post_generate(url, payload, timeout)
    wall = time.perf_counter() - t0
    items = _as_result_list(raw)
    for i, item in enumerate(items):
        print_item(i, labels[i] if i < len(labels) else "?", item)
    summarize("http-batch", wall, items, n)


def run_concurrent(
    url: str,
    labels: list[str],
    data_uris: list[str],
    sampling: dict[str, Any],
    timeout: float,
) -> None:
    n = len(labels)
    print(f"=== concurrent ({n} parallel POSTs) ===")

    def one(i: int) -> tuple[int, dict[str, Any]]:
        payload = {
            "text": build_prompt(),
            "image_data": data_uris[i],
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
    for i, item in enumerate(items):
        if item is None:
            print(f"  [{i}] expect={labels[i]:12s}  FAILED")
        else:
            print_item(i, labels[i], item)
    for err in errors:
        print(f"  error: {err}")
    summarize("concurrent", wall, ok, n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VL image batch smoke test for SGLang / SPECTRE Target")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--mode",
        choices=("all", "single", "http-batch", "concurrent"),
        default="all",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    url = f"http://{args.host}:{args.port}/generate"
    sampling = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }

    labels: list[str] = []
    data_uris: list[str] = []
    paths: list[str] = []
    for i in range(args.batch_size):
        label, path, uri = make_image(i, args.image_size)
        labels.append(label)
        paths.append(path)
        data_uris.append(uri)

    print(f"target: {url}")
    print(f"images ({args.image_size}x{args.image_size}) saved under {IMAGE_DIR}:")
    for i, (label, path) in enumerate(zip(labels, paths)):
        print(f"  [{i}] {label}  {path}")
    print()

    if args.mode in ("all", "single"):
        run_single(url, labels, data_uris, sampling, args.timeout)
    if args.mode in ("all", "http-batch"):
        run_http_batch(url, labels, data_uris, sampling, args.timeout)
    if args.mode in ("all", "concurrent"):
        run_concurrent(url, labels, data_uris, sampling, args.timeout)


if __name__ == "__main__":
    main()
