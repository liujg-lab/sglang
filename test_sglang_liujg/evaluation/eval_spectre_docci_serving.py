#!/usr/bin/env python3
"""Evaluate prepared DOCCI captions against a running Qwen3-VL Target.

在仓库根目录运行。先用 prepare_docci_sample.py 转换 test Parquet，再启动 Target。
--host / --port 指定 Target 地址，默认 127.0.0.1:30000。
--data-file / --img-dir 可覆盖转换后的 samples.json 和 images/ 路径。
--max-items 默认取前 100 条；--max-concurrency 默认 1（串行）。
--request-rate 默认 inf（突发到达，仍受并发上限约束）；有限值为平均每秒到达数。
--max-new-tokens 默认 1024；--temperature 默认 0。
--output-jsonl / --output-csv 指定结果路径；--skip-metrics 仅跳过 BLEU/ROUGE。

转换：python3 -I test_sglang_liujg/evaluation/prepare_docci_sample.py
冒烟：python3 test_sglang_liujg/evaluation/eval_spectre_docci_serving.py --max-items 5 --skip-metrics
评测 120 条：python3 test_sglang_liujg/evaluation/eval_spectre_docci_serving.py --max-items 120
并发 4、平均每秒 2 条：python3 test_sglang_liujg/evaluation/eval_spectre_docci_serving.py --max-concurrency 4 --request-rate 2
"""

import argparse
import asyncio
import random
from pathlib import Path

from eval_spectre_rsteller import DEFAULT_RESULT_DIR
from eval_spectre_rsteller_serving import run_benchmark

DATASET_ROOT = Path(
    "/home_18T/liujg/Hugging_Face/gokaygokay/random_instruct_docci_sample_120"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SPECTRE DOCCI serving-style caption evaluation (open-loop)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Arrivals per second. Finite values use a Poisson process; inf sends a burst.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of in-flight requests. Default: 1.",
    )
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--data-file", type=Path, default=DATASET_ROOT / "samples.json")
    parser.add_argument("--img-dir", type=Path, default=DATASET_ROOT / "images")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_RESULT_DIR / "DOCCI_spectre_serving_results.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULT_DIR / "DOCCI_spectre_serving_statistics.csv",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--flush-cache", action="store_true", help="POST /flush_cache after warmup"
    )
    parser.add_argument(
        "--skip-metrics", action="store_true", help="Skip BLEU / ROUGE scoring"
    )
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for Poisson arrivals")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_items < 1:
        raise SystemExit("--max-items must be >= 1")
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")
    if not args.request_rate > 0:
        raise SystemExit("--request-rate must be > 0")
    random.seed(args.seed)
    return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    raise SystemExit(main())
