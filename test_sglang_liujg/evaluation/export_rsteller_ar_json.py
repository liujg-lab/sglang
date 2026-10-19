#!/usr/bin/env python3
"""把已启动的 AR 服务对 RSTeller 的生成结果写成同结构 JSON。

功能
    HTTP 客户端，不拉起服务。服务需已用自回归模式听在 --host/--port
    （不要加 --speculative-algorithm）。
    请求复用 eval_spectre_rsteller_serving.async_run_one，prompt、图片和
    流式 /generate 与现有 serving 评测一致。
    不写 CSV，不计算 BLEU / ROUGE。

输出
    JSON 数组，每条只保留与 test_RSTeller.json 相同的三个键，顺序一致：
      image / question    原样拷贝
      ground_truth        模型输出去掉首尾空白
    不保留原始参考句，也不写入耗时、token 或投机字段。
    条目顺序与输入切片一致。error 非空的条 ground_truth 写成空字符串，
    进程以退出码 1 结束。spec_verify_ct > 0 时警告当前服务不是纯 AR。

运行（仓库根目录，AR 服务已启动）
    python test_sglang_liujg/evaluation/export_rsteller_ar_json.py \
      --host 127.0.0.1 --port 30000
    python test_sglang_liujg/evaluation/export_rsteller_ar_json.py \
      --max-items 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from eval_spectre_rsteller import (
    DEFAULT_DATA,
    DEFAULT_IMG_DIR,
    DEFAULT_RESULT_DIR,
    load_dataset,
)
from eval_spectre_rsteller_serving import AIOHTTP_READ_BUFSIZE, async_run_one

DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "test_RSTeller_ar.json"


def to_rsteller_item(item: dict[str, Any], rec: dict[str, Any]) -> dict[str, str]:
    """Keep source keys only. Failed requests store an empty ground_truth."""
    if rec.get("error"):
        text = ""
    else:
        text = str(rec.get("model_output") or "").strip()
    return {
        "image": str(item.get("image") or ""),
        "question": str(item.get("question") or ""),
        "ground_truth": text,
    }


def _spec_verify_ct(rec: dict[str, Any]) -> float:
    value = rec.get("spec_verify_ct")
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


async def export_ar(args: argparse.Namespace) -> int:
    import aiohttp

    generate_url = f"http://{args.host}:{args.port}/generate"
    max_items = None if args.max_items == 0 else args.max_items
    dataset = load_dataset(args.data_file, max_items)
    sampling = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    print(f"loaded {len(dataset)} items from {args.data_file}")
    print(
        f"target={generate_url}  max_concurrency={args.max_concurrency}  "
        f"max_new_tokens={args.max_new_tokens}  temperature={args.temperature}"
    )

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(
        timeout=timeout, read_bufsize=AIOHTTP_READ_BUFSIZE
    ) as session:
        if not args.no_warmup and dataset:
            print("=== warmup (first item, discarded) ===")
            warmup = await async_run_one(
                session, generate_url, dataset[0], args.img_dir, sampling
            )
            if warmup.get("error"):
                print(f"warmup failed: {warmup['error']}", file=sys.stderr)
                return 1
            print("warmup ok")

        semaphore = asyncio.Semaphore(args.max_concurrency)

        async def limited(idx: int, item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            async with semaphore:
                rec = await async_run_one(
                    session, generate_url, item, args.img_dir, sampling
                )
            return idx, rec

        tasks = [
            asyncio.create_task(limited(idx, item))
            for idx, item in enumerate(dataset)
        ]
        paired = await asyncio.gather(*tasks)

    paired.sort(key=lambda pair: pair[0])
    exported: list[dict[str, str]] = []
    n_error = 0
    warned_spec = False
    for idx, rec in paired:
        item = dataset[idx]
        if rec.get("error"):
            n_error += 1
            print(f"  [{idx}] ERROR {rec['error']}", file=sys.stderr)
        verify_ct = _spec_verify_ct(rec)
        if verify_ct > 0 and not warned_spec:
            warned_spec = True
            print(
                f"warning: spec_verify_ct={verify_ct:g} > 0 at item {idx}; "
                "server is not pure AR",
                file=sys.stderr,
            )
        exported.append(to_rsteller_item(item, rec))
        if not rec.get("error"):
            preview = exported[-1]["ground_truth"].replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:77] + "..."
            print(f"  [{idx}] {preview!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(exported, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(exported)} items ({n_error} errors) to {args.output}")
    return 1 if n_error else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export AR RSTeller generations as a ground_truth JSON file"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Cap on in-flight requests. Default: 1.",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Use the first N items. 0 means the full dataset. Default: 0.",
    )
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    p.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the discarded warmup request on the first item",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_items < 0:
        raise SystemExit("--max-items must be >= 0")
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be >= 1")
    return asyncio.run(export_ar(args))


if __name__ == "__main__":
    raise SystemExit(main())
