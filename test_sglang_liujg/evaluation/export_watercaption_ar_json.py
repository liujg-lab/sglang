#!/usr/bin/env python3
"""把已启动的 AR 服务对 WaterCaption 的生成结果写成同结构 JSON。

功能
    HTTP 客户端，不拉起服务。行为与 export_rsteller_ar_json.py 相同，
    只换 WaterCaption 默认数据、图片目录和输出路径。
    请求仍走 export_ar：async_run_one、丢弃式 warmup、失败条 ground_truth
    留空并退出码 1、spec_verify_ct > 0 时警告当前服务不是纯 AR。
    不写 CSV，不计算 BLEU / ROUGE。

数据集（4039 条 JSON 数组）
    --data-file  .../test_WaterCaption/json/test/test_WaterCaption.json
    --img-dir    .../test_WaterCaption
    每条字段与 RSTeller 相同：image / question / ground_truth。
    ground_truth 换成去掉首尾空白的模型输出。

运行（仓库根目录，AR 服务已启动）
    python test_sglang_liujg/evaluation/export_watercaption_ar_json.py \
      --host 127.0.0.1 --port 30000
    python test_sglang_liujg/evaluation/export_watercaption_ar_json.py \
      --max-items 2
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from eval_spectre_rsteller import DEFAULT_RESULT_DIR
from eval_spectre_watercaption_serving import DEFAULT_DATA, DEFAULT_IMG_DIR
from export_rsteller_ar_json import export_ar

DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "test_WaterCaption_ar.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export AR WaterCaption generations as a ground_truth JSON file"
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
