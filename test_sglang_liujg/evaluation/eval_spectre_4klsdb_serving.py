#!/usr/bin/env python3
"""
Evaluate prepared 4KLSDB image captions against a running Qwen3-VL Target.

在脚本所在目录运行；先启动 Target，用 --host / --port 指定地址（默认 127.0.0.1:30000）。

--data-file / --img-dir 指定样本清单和图片目录；--max-items 取清单前 N 条（默认 100）。

--max-concurrency 限制客户端同时在途的请求数（默认 1，即串行）。

--request-rate 指定平均每秒到达数；默认 inf（一次性提交，仍受并发上限约束）。

--max-new-tokens 限制单条输出长度（默认 1024）；--temperature 默认 0。

--output-jsonl / --output-csv 指定明细和汇总；--skip-metrics 只跳过 BLEU/ROUGE。

冒烟：python3 eval_spectre_4klsdb_serving.py --max-items 5 --skip-metrics

并发 4、平均每秒到达 2 条：python3 test_sglang_liujg/evaluation/eval_spectre_4klsdb_serving.py --max-concurrency 4 --request-rate 2

串行评测 100 条并计算 BLEU/ROUGE：python3 eval_spectre_4klsdb_serving.py
"""

import argparse
import asyncio
import random
from pathlib import Path

from eval_spectre_rsteller import DEFAULT_RESULT_DIR
from eval_spectre_rsteller_serving import run_benchmark

DATASET_ROOT = Path("/home_18T/liujg/Hugging_Face/pandixing/4KLSDB_sample_100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SPECTRE 4KLSDB serving-style evaluation (open-loop)"
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
        default=DEFAULT_RESULT_DIR / "4KLSDB_spectre_serving_results.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULT_DIR / "4KLSDB_spectre_serving_statistics.csv",
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
    if args.request_rate <= 0:
        raise SystemExit("--request-rate must be > 0")
    random.seed(args.seed)
    return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    raise SystemExit(main())
