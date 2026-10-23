#!/usr/bin/env python3
"""SPECTRE RSVQA serving-style 评测客户端（开放到达 + 流式百分位）。

功能
    HTTP 客户端，打已启动的 SPECTRE Target（默认 :30000），不拉起服务。
    到达、流式计时和投机统计与 eval_spectre_rsteller_serving.py 相同，只换
    RSVQA 默认数据与输出路径。
    请求构造复用 eval_spectre_rsteller.prepare_request（Qwen3-VL chat 模板 + image_data）。
    不调用 eval_rsteller_metrics.py，不计算 Avg BLEU / ROUGE-1 / ROUGE-2 / ROUGE-L。
    共享 CSV 表头仍保留这四列，单元格留空，避免改动共用的 CSV_FIELDS。
    时间口径在共享 serving 里计算，全部使用客户端 time.perf_counter：

        t_send                 t_first                t_last          t_end
        发起 HTTP ──────────── 首次 token 事件 ────── 最后新增 token ── 响应结束
                  TTFT                   Decode Time

    TTFT = t_first - t_send。Decode Time = t_last - t_first。
    Client E2E = t_end - t_send。Server E2E 单独保存，不与 TTFT 相减。
    吞吐分子是 N-K：K 是首个 token 事件的累计 token 数，N 是最后一次增长的累计数。
    串行时 Decode tok/s 适合比较 AR/SR 的输出阶段；并发时它是请求级合并速度，
    不是服务器聚合吞吐。整场墙钟吞吐看 Output tok/s (wall)。

数据集（222684 条 JSON 数组，按图排列，每张图约 100 题）
    --data-file  .../USFM_UAV_training_data/json/test/test_RSVQA.json
    --img-dir    .../USFM_UAV_training_data
    每条字段：
      image          相对 img_dir 的图片路径
      question       用户问题；其中的 <image> 会被去掉
      ground_truth   参考答案（原样写入 jsonl，不参与打分）

    默认按 --seed 打乱图片，取 --max-items 张，每张抽 1 题。
    相同 seed 和 max-items 两次运行选中同一批题、同一顺序。
    抽中的列表写到 result/RSVQA_sample_seed{seed}_n{n}.json，再交给评测。
    抽样用独立的 Random，不改变 Poisson 到达所用的全局随机序列。
    --no-sample 恢复为取数据文件的前 max-items 条。

运行（仓库根目录，Target + Draft 已启动）
    python test_sglang_liujg/evaluation/eval_spectre_rsvqa_serving.py \
      --port 30000 --max-items 20 --max-concurrency 1
    python test_sglang_liujg/evaluation/eval_spectre_rsvqa_serving.py \
      --port 30000 --max-items 20 --seed 1 --request-rate 8 --max-concurrency 4
    python test_sglang_liujg/evaluation/eval_spectre_rsvqa_serving.py \
      --port 30000 --max-items 20 --no-sample
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any

from eval_spectre_rsteller import DEFAULT_RESULT_DIR, load_dataset
from eval_spectre_rsteller_serving import run_benchmark

DEFAULT_DATA = Path(
    "/home_18T/liujg/Hugging_Face/model_and_data_of_project_1/"
    "USFM_UAV_training_data/json/test/test_RSVQA.json"
)
DEFAULT_IMG_DIR = Path(
    "/home_18T/liujg/Hugging_Face/model_and_data_of_project_1/USFM_UAV_training_data"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPECTRE RSVQA serving-style evaluation (open-loop)"
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
        default=1,
        help="Cap on in-flight requests (asyncio.Semaphore). Default: 1.",
    )
    p.add_argument("--max-items", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    p.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_RESULT_DIR / "RSVQA_spectre_serving_results.jsonl",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULT_DIR / "RSVQA_spectre_serving_statistics.csv",
    )
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument(
        "--flush-cache",
        action="store_true",
        help="POST /flush_cache after warmup, once before the main run",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for image sampling and Poisson arrivals. "
        "The same seed and --max-items select the same questions.",
    )
    p.add_argument(
        "--no-sample",
        action="store_true",
        help="Use the first --max-items records instead of one question per image.",
    )
    return p.parse_args()


def group_by_image(
    items: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    """Group items by image, keeping the dataset order of both images and questions."""
    order: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        image = str(item.get("image") or "")
        if image not in groups:
            order.append(image)
            groups[image] = []
        groups[image].append(item)
    return order, groups


def sample_one_per_image(
    items: list[dict[str, Any]], max_items: int, seed: int
) -> tuple[list[dict[str, Any]], int]:
    """Pick one question from each of max_items images.

    Image order and the question inside each image come from Random(seed),
    not from the global random stream used for Poisson arrivals.
    """
    order, groups = group_by_image(items)
    rng = random.Random(seed)
    images = list(order)
    rng.shuffle(images)
    if max_items < len(images):
        images = images[:max_items]
    chosen = [groups[image][rng.randrange(len(groups[image]))] for image in images]
    return chosen, len(order)


def sample_json_path(seed: int, n_items: int) -> Path:
    return DEFAULT_RESULT_DIR / f"RSVQA_sample_seed{seed}_n{n_items}.json"


def prepare_dataset(args: argparse.Namespace) -> None:
    """Point --data-file at a reproducible per-image sample, unless --no-sample."""
    if args.no_sample:
        print(f"no-sample: first {args.max_items} items from {args.data_file}")
        return
    items = load_dataset(args.data_file, None)
    chosen, n_images = sample_one_per_image(items, args.max_items, args.seed)
    if len(chosen) < args.max_items:
        print(
            f"only {n_images} images in {args.data_file}; "
            f"sampling {len(chosen)} instead of {args.max_items}"
        )
    path = sample_json_path(args.seed, len(chosen))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(chosen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    n_unique = len({item.get("image") for item in chosen})
    print(
        f"sampled {len(chosen)} questions from {n_unique} images "
        f"seed={args.seed} -> {path}"
    )
    args.data_file = path
    args.max_items = len(chosen)


def main() -> int:
    args = parse_args()
    if args.max_items < 1:
        raise SystemExit("--max-items must be >= 1")
    if args.max_concurrency is not None and args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")
    if args.request_rate <= 0:
        raise SystemExit("--request-rate must be > 0")
    prepare_dataset(args)
    random.seed(args.seed)
    args.skip_metrics = True
    return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    raise SystemExit(main())
