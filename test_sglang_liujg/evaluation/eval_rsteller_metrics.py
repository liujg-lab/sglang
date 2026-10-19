#!/usr/bin/env python3
"""RSTeller（及同结构 jsonl）生成结果精度统计：BLEU / ROUGE-1/2/L。

功能
    读生成脚本写出的 jsonl，对无 error（或 error 为空）且 ground_truth
    分词后非空的条目打分。重跑时先清掉旧 metrics。
    没有 error 但 model_output 分词后为空的条目记 0，并计入平均分母。
    ground_truth 分词后为空的条目跳过、不计入分母，并单独计数（no_ref）。
    带 error 的条目跳过，不计入分母。
    把 per-item 字典 metrics={bleu, rouge1, rouge2, rougeL} 写回 jsonl。
    若指定 --csv，把平均值写入该 CSV 的指定数据行（默认最后一行）的
    Avg BLEU / Avg ROUGE-1 / Avg ROUGE-2 / Avg ROUGE-L 列（缺列会补上）。
    不直接读原始 RSTeller JSON；输入必须是 eval_spectre_rsteller.py 的产出。
    本分词与分母规则改动后的数值与历史 CSV 不能直接比较。

指标含义
    bleu 与 rouge1/2/L 共用同一套分词：转小写；中日韩字符按单字切；
    其余按词字符切；标点丢掉；ASCII 词长度大于 3 时做 Porter 词干化
    （与原先 use_stemmer=True 对 ASCII 词的处理一致）。
    bleu        nltk sentence_bleu，smoothing=method1
    rouge1/2/L  rouge-score F-measure，使用上述 tokenizer
    CSV 四列为上述字段在有效条目上的算术平均。分母含空输出，
    不含 error 与无参考的条目。

依赖
    pip install nltk "rouge-score>=0.1.2"

运行
    python test_sglang_liujg/evaluation/eval_rsteller_metrics.py \\
      --jsonl test_sglang_liujg/result/RSTeller_spectre_results.jsonl \\
      --csv test_sglang_liujg/result/RSTeller_spectre_statistics.csv

参数
    --jsonl            必填。生成结果 jsonl 路径（读写）
    --csv              可选。汇总 CSV；省略则只更新 jsonl
    --csv-row-index    要改的数据行（0-based，不含表头）。默认最后一行。
                       负数按 Python 切片从末尾计数。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

ACCURACY_FIELDS = ("Avg BLEU", "Avg ROUGE-1", "Avg ROUGE-2", "Avg ROUGE-L")

# CJK (incl. kana / hangul) as single characters; other word chars as tokens.
# Punctuation and underscores are dropped.
_TOKEN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]|[^\W_]+")
_stemmer: Any = None


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [_stemmer.stem(t) if t.isascii() and len(t) > 3 else t for t in tokens]


class _Tokenizer:
    def tokenize(self, text: str) -> list[str]:
        return tokenize(text)


def _import_scorers():
    global _stemmer
    try:
        from nltk.stem.porter import PorterStemmer
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        from rouge_score import rouge_scorer
    except ImportError:
        print(
            '缺少评估库，请安装: pip install nltk "rouge-score>=0.1.2"',
            file=sys.stderr,
        )
        raise
    _stemmer = PorterStemmer()
    return sentence_bleu, SmoothingFunction().method1, rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], tokenizer=_Tokenizer()
    )


def compute_metrics(
    hypothesis: str,
    reference: str,
    sentence_bleu: Callable[..., float],
    smooth_fn: Any,
    scorer: Any,
) -> dict[str, float]:
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)
    if not hyp_tokens:
        return {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth_fn)
    scores = scorer.score(reference, hypothesis)
    return {
        "bleu": float(bleu),
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def score_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    sentence_bleu, smooth_fn, scorer = _import_scorers()
    totals = {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n_valid = 0
    n_no_ref = 0
    n_empty_output = 0
    for rec in records:
        rec.pop("metrics", None)
        if rec.get("error"):
            continue
        reference = str(rec.get("ground_truth") or "")
        if not tokenize(reference):
            n_no_ref += 1
            continue
        hypothesis = str(rec.get("model_output") or "")
        metrics = compute_metrics(hypothesis, reference, sentence_bleu, smooth_fn, scorer)
        if not tokenize(hypothesis):
            n_empty_output += 1
        rec["metrics"] = metrics
        for key in totals:
            totals[key] += metrics[key]
        n_valid += 1

    avgs = {key: (totals[key] / n_valid if n_valid else 0.0) for key in totals}
    avgs["valid_metric_count"] = float(n_valid)
    avgs["n_no_ref"] = float(n_no_ref)
    avgs["n_empty_output"] = float(n_empty_output)
    return records, avgs


def update_csv(csv_path: Path, avgs: dict[str, float], row_index: int | None) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for name in ACCURACY_FIELDS:
        if name not in fieldnames:
            fieldnames.append(name)

    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")

    idx = row_index if row_index is not None else len(rows) - 1
    if idx < 0:
        idx = len(rows) + idx
    if idx < 0 or idx >= len(rows):
        raise IndexError(f"csv row index {idx} out of range (n={len(rows)})")

    rows[idx]["Avg BLEU"] = f"{avgs['bleu']:.4f}"
    rows[idx]["Avg ROUGE-1"] = f"{avgs['rouge1']:.4f}"
    rows[idx]["Avg ROUGE-2"] = f"{avgs['rouge2']:.4f}"
    rows[idx]["Avg ROUGE-L"] = f"{avgs['rougeL']:.4f}"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute BLEU/ROUGE for RSTeller SPECTRE jsonl")
    p.add_argument("--jsonl", type=Path, required=True, help="Generation jsonl to score")
    p.add_argument("--csv", type=Path, default=None, help="Summary CSV to update")
    p.add_argument(
        "--csv-row-index",
        type=int,
        default=None,
        help="0-based data-row index to update (default: last row)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.jsonl.exists():
        print(f"jsonl not found: {args.jsonl}", file=sys.stderr)
        return 1

    records = load_jsonl(args.jsonl)
    records, avgs = score_records(records)
    write_jsonl(args.jsonl, records)

    n_valid = int(avgs["valid_metric_count"])
    n_no_ref = int(avgs["n_no_ref"])
    n_empty_output = int(avgs["n_empty_output"])
    print(
        f"scored {n_valid}/{len(records)} items  "
        f"no_ref={n_no_ref}  "
        f"empty_output={n_empty_output}  "
        f"BLEU={avgs['bleu']:.4f}  "
        f"ROUGE-1={avgs['rouge1']:.4f}  "
        f"ROUGE-2={avgs['rouge2']:.4f}  "
        f"ROUGE-L={avgs['rougeL']:.4f}"
    )

    if args.csv is not None:
        update_csv(args.csv, avgs, args.csv_row_index)
        print(f"updated accuracy columns in {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
