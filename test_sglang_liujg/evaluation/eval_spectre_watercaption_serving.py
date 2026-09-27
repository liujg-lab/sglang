#!/usr/bin/env python3
"""SPECTRE WaterCaption serving-style 评测客户端（开放到达 + 流式百分位）。

功能
    HTTP 客户端，打已启动的 SPECTRE Target（默认 :30000），不拉起服务。
    到达、流式计时、投机统计和 BLEU / ROUGE 与
    eval_spectre_rsteller_serving.py 相同，只换 WaterCaption 默认数据与输出路径。
    请求构造复用 eval_spectre_rsteller.prepare_request（Qwen3-VL chat 模板 + image_data）。
    结束后调用 eval_rsteller_metrics.py，按 model_output / ground_truth 补 BLEU / ROUGE。

数据集（4039 条 JSON 数组）
    --data-file  .../test_WaterCaption/json/test/test_WaterCaption.json
    --img-dir    .../test_WaterCaption
    每条字段：
      image          相对 img_dir 的图片路径（已含 source/water/.../images/）
      question       用户问题；其中的 <image> 会被去掉
      ground_truth   英文参考 caption

运行（仓库根目录，Target + Draft 已启动）
    python test_sglang_liujg/evaluation/eval_spectre_watercaption_serving.py \
      --port 30000 --max-items 100 --request-rate inf --max-concurrency 4
    python test_sglang_liujg/evaluation/eval_spectre_watercaption_serving.py \
      --port 30000 --max-items 4039 --request-rate 8 --max-concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

from eval_spectre_rsteller import DEFAULT_RESULT_DIR
from eval_spectre_rsteller_serving import run_benchmark

DEFAULT_DATA = Path(
    "/home_18T/liujg/Hugging_Face/model_and_data_of_project_1/"
    "test_WaterCaption/json/test/test_WaterCaption.json"
)
DEFAULT_IMG_DIR = Path(
    "/home_18T/liujg/Hugging_Face/model_and_data_of_project_1/test_WaterCaption"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPECTRE WaterCaption serving-style evaluation (open-loop)"
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
    p.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_RESULT_DIR / "WaterCaption_spectre_serving_results.jsonl",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULT_DIR / "WaterCaption_spectre_serving_statistics.csv",
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
    random.seed(args.seed)
    return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    raise SystemExit(main())
