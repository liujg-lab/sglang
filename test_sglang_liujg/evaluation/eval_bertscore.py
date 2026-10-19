#!/usr/bin/env python3
"""用本地 roberta-large 计算 SR 输出相对 AR 参考的 BERTScore。

功能
    按 image 把 SR jsonl 的 model_output 配到参考 JSON 的 ground_truth。
    RSTeller 与 WaterCaption 使用同一入口，两条路径都由参数指定。
    SR jsonl 自带的 ground_truth 是原始短 caption，不参与打分。
    只把平均精确率、召回率和 F1 打印到终端，不写文件，不改 jsonl。

    跳过 error 非空、model_output 为空、参考 ground_truth 为空，
    或 image 在参考中找不到的条。这些条不进入平均分母。

    每条文本按最多 480 个子词切成不重叠的连续段，再交给 bert-score。
    库会加上 CLS/SEP，总长仍低于 512。段序号对齐；候选多出的段记
    P=0、F1=0，参考多出的段记 R=0、F1=0。段长为两侧子词数之和，
    单侧多余段用它自己的子词数。每条样本先按段长加权，再对样本等权平均。

    前向只跑一次且不做库内缩放。原始分打印后，再用安装包内
    rescale_baseline/en/roberta-large.tsv 的第 17 层自行缩放。

指标
    候选是 SR 的 model_output，参考是同一张图的 32B 自回归 ground_truth。
    精确率：候选里的子词能否在参考里找到相近向量。
    召回率：参考里的子词能否被候选覆盖。
    F1：精确率和召回率的调和平均。
    参考比候选长时，多出的段把召回率和 F1 记为 0，精确率不受这些段影响。

    原始分是贪心匹配后的余弦相似度。无关英文句子也常在 0.83 附近，
    所以原始 F1 为 0.82 并不表示和参考很像。

    缩放分使用 roberta-large 第 17 层基线 b，按 (s - b) / (1 - b) 换算。
    b 约为 0.83，对应随机英文句对。缩放后 0 表示与随机句对相同，
    1 表示完全相同，负数表示低于随机句对。比较不同 SR 配置时看缩放分。

    例：原始 P/R/F1 为 0.8878 / 0.8115 / 0.8193 时，缩放后为
    0.3339 / -0.1189 / -0.0708。召回率只比基线低约 0.02，
    除以 (1 - b) 后变成 -0.12。

依赖
    pip install bert-score

运行
    python test_sglang_liujg/evaluation/eval_bertscore.py \
      --reference /path/to/test_RSTeller_ar_qwen3-vl-32B.json \
      --jsonl test_sglang_liujg/result/RSTeller_spectre_serving_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = Path("/home_18T/liujg/Hugging_Face/FacebookAI/roberta-large")
ROBERTA_LARGE_LAYERS = 17
CHUNK_TOKENS = 480


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_reference(path: Path) -> dict[str, str]:
    """Map image path to AR ground_truth. The first copy of an image wins."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(raw)}")
    refs: dict[str, str] = {}
    duplicate = 0
    for item in raw:
        image = str(item.get("image") or "")
        if not image:
            continue
        if image in refs:
            duplicate += 1
            continue
        refs[image] = str(item.get("ground_truth") or "")
    if duplicate:
        print(f"warning: ignored {duplicate} duplicate reference images", file=sys.stderr)
    return refs


def pair_records(
    records: list[dict[str, Any]], refs: dict[str, str]
) -> tuple[list[str], list[str], dict[str, int]]:
    """Return candidate texts, reference texts, and skip counts."""
    cands: list[str] = []
    references: list[str] = []
    skipped = {"error": 0, "empty": 0, "unmatched": 0}
    for rec in records:
        if rec.get("error"):
            skipped["error"] += 1
            continue
        hypothesis = str(rec.get("model_output") or "").strip()
        image = str(rec.get("image") or "")
        if not hypothesis or not image:
            skipped["empty"] += 1
            continue
        if image not in refs:
            skipped["unmatched"] += 1
            continue
        reference = refs[image].strip()
        if not reference:
            skipped["empty"] += 1
            continue
        cands.append(hypothesis)
        references.append(reference)
    return cands, references, skipped


def chunk_ids(ids: list[int], size: int = CHUNK_TOKENS) -> list[list[int]]:
    """Split token ids into contiguous non-overlapping chunks."""
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def encode_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def chunk_texts(tokenizer: Any, text: str, size: int = CHUNK_TOKENS) -> list[tuple[str, int]]:
    """Return decoded chunks and their subword counts. Empty pieces are dropped."""
    chunks: list[tuple[str, int]] = []
    for part in chunk_ids(encode_ids(tokenizer, text), size):
        piece = tokenizer.decode(part, skip_special_tokens=True).strip()
        if not piece:
            continue
        chunks.append((piece, len(part)))
    return chunks


def weighted_mean(parts: list[tuple[float, int]]) -> float:
    weight = sum(item_weight for _, item_weight in parts)
    if weight <= 0:
        return 0.0
    return sum(value * item_weight for value, item_weight in parts) / weight


def example_metrics(
    cand_lengths: list[int],
    ref_lengths: list[int],
    pair_metrics: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Length-weighted P, R, F1 for one example.

    Paired chunks use cand_len + ref_len. A leftover candidate chunk contributes
    P=0 and F1=0. A leftover reference chunk contributes R=0 and F1=0.
    """
    n_pair = min(len(cand_lengths), len(ref_lengths))
    if len(pair_metrics) != n_pair:
        raise ValueError(
            f"expected {n_pair} chunk scores, got {len(pair_metrics)}"
        )
    precision_parts: list[tuple[float, int]] = []
    recall_parts: list[tuple[float, int]] = []
    f1_parts: list[tuple[float, int]] = []
    for index in range(n_pair):
        precision, recall, f1 = pair_metrics[index]
        weight = cand_lengths[index] + ref_lengths[index]
        precision_parts.append((precision, weight))
        recall_parts.append((recall, weight))
        f1_parts.append((f1, weight))
    for length in cand_lengths[n_pair:]:
        precision_parts.append((0.0, length))
        f1_parts.append((0.0, length))
    for length in ref_lengths[n_pair:]:
        recall_parts.append((0.0, length))
        f1_parts.append((0.0, length))
    return (
        weighted_mean(precision_parts),
        weighted_mean(recall_parts),
        weighted_mean(f1_parts),
    )


def macro_average(rows: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    count = len(rows)
    if count == 0:
        return 0.0, 0.0, 0.0
    return (
        sum(row[0] for row in rows) / count,
        sum(row[1] for row in rows) / count,
        sum(row[2] for row in rows) / count,
    )


def rescale_scores(
    raw: tuple[float, float, float], baseline: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Apply BERTScore baseline rescaling (s - b) / (1 - b)."""
    scaled: list[float] = []
    for value, base in zip(raw, baseline):
        denom = 1.0 - base
        if denom == 0.0:
            raise ZeroDivisionError(f"baseline {base} leaves a zero denominator")
        scaled.append((value - base) / denom)
    return scaled[0], scaled[1], scaled[2]


def baseline_tsv() -> Path:
    import bert_score

    path = (
        Path(bert_score.__file__).resolve().parent
        / "rescale_baseline"
        / "en"
        / "roberta-large.tsv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"roberta-large baseline not found at {path}. "
            "Reinstall bert-score or check the package data files."
        )
    return path


def _first_baseline_line(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def layer_baseline(path: Path, layer: int) -> tuple[float, float, float]:
    """Copy the P/R/F baseline for one transformer layer.

    Newer bert-score files are CSV with a ``LAYER,P,R,F`` header. Older files
    are whitespace-separated numbers, one row per layer.
    """
    import numpy as np

    header = _first_baseline_line(path)
    if header.upper().startswith("LAYER"):
        loaded = np.loadtxt(path, delimiter=",", skiprows=1)
        rows = np.array(loaded, dtype=np.float64, copy=True)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != 4:
            raise ValueError(f"baseline shape {rows.shape} is not LAYER,P,R,F")
        matched = rows[np.isclose(rows[:, 0], layer)]
        if matched.shape[0] != 1:
            raise ValueError(f"baseline has no unique layer {layer}")
        return float(matched[0, 1]), float(matched[0, 2]), float(matched[0, 3])

    loaded = np.loadtxt(path)
    rows = np.array(loaded, dtype=np.float64, copy=True)
    if rows.ndim != 2 or rows.shape[1] != 3 or not 0 <= layer < rows.shape[0]:
        raise ValueError(f"baseline shape {getattr(rows, 'shape', None)} has no layer {layer}")
    return float(rows[layer, 0]), float(rows[layer, 1]), float(rows[layer, 2])


def quiet_transformers() -> None:
    """Hide the encoder load report. Loading itself is unchanged."""
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()


def prepare_chunk_pairs(
    tokenizer: Any, cands: list[str], refs: list[str], skipped: dict[str, int]
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    flat_cands: list[str] = []
    flat_refs: list[str] = []
    examples: list[dict[str, Any]] = []
    for cand, ref in zip(cands, refs):
        cand_chunks = chunk_texts(tokenizer, cand)
        ref_chunks = chunk_texts(tokenizer, ref)
        if not cand_chunks or not ref_chunks:
            skipped["empty"] += 1
            continue
        n_pair = min(len(cand_chunks), len(ref_chunks))
        for index in range(n_pair):
            flat_cands.append(cand_chunks[index][0])
            flat_refs.append(ref_chunks[index][0])
        examples.append(
            {
                "cand_lengths": [length for _, length in cand_chunks],
                "ref_lengths": [length for _, length in ref_chunks],
                "n_pair": n_pair,
            }
        )
    return flat_cands, flat_refs, examples


def score_bertscore(
    cands: list[str],
    refs: list[str],
    model_path: Path,
    batch_size: int,
    skipped: dict[str, int],
) -> tuple[int, tuple[float, float, float], tuple[float, float, float]]:
    try:
        from bert_score import score
        from transformers import AutoTokenizer
    except ImportError:
        print("缺少评估库，请安装: pip install bert-score", file=sys.stderr)
        raise
    if not model_path.is_dir():
        raise FileNotFoundError(f"model path not found: {model_path}")

    quiet_transformers()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    # Encoding the full caption only measures chunk boundaries. The model still
    # sees pieces of at most 480 subwords.
    tokenizer.model_max_length = int(1e9)
    flat_cands, flat_refs, examples = prepare_chunk_pairs(tokenizer, cands, refs, skipped)
    if not examples:
        return 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    precision, recall, f1 = score(
        flat_cands,
        flat_refs,
        model_type=str(model_path),
        num_layers=ROBERTA_LARGE_LAYERS,
        lang="en",
        idf=False,
        rescale_with_baseline=False,
        batch_size=batch_size,
        verbose=False,
    )
    cursor = 0
    per_example: list[tuple[float, float, float]] = []
    for example in examples:
        n_pair = example["n_pair"]
        pair_metrics = [
            (float(precision[cursor + offset]), float(recall[cursor + offset]), float(f1[cursor + offset]))
            for offset in range(n_pair)
        ]
        cursor += n_pair
        per_example.append(
            example_metrics(example["cand_lengths"], example["ref_lengths"], pair_metrics)
        )
    if cursor != len(flat_cands):
        raise RuntimeError(f"consumed {cursor} chunk scores, expected {len(flat_cands)}")

    raw = macro_average(per_example)
    scaled = rescale_scores(raw, layer_baseline(baseline_tsv(), ROBERTA_LARGE_LAYERS))
    return len(examples), raw, scaled


def format_scores(label: str, scores: tuple[float, float, float]) -> str:
    precision, recall, f1 = scores
    return f"BERTScore {label:<8} P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BERTScore F1 of SR outputs against AR reference captions"
    )
    p.add_argument("--reference", type=Path, required=True, help="AR JSON with image and ground_truth")
    p.add_argument("--jsonl", type=Path, required=True, help="SR jsonl with image and model_output")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="bert-score encoder batch size. Default: 16.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if not args.reference.is_file():
        print(f"reference not found: {args.reference}", file=sys.stderr)
        return 1
    if not args.jsonl.is_file():
        print(f"jsonl not found: {args.jsonl}", file=sys.stderr)
        return 1

    refs = load_reference(args.reference)
    records = load_jsonl(args.jsonl)
    cands, paired_refs, skipped = pair_records(records, refs)
    if not cands:
        print("no pairs to score", file=sys.stderr)
        return 1
    try:
        n_scored, raw, scaled = score_bertscore(
            cands, paired_refs, args.model_path, args.batch_size, skipped
        )
    except ImportError:
        return 1
    print(
        f"scored {n_scored}/{len(records)}  "
        f"skipped_error={skipped['error']}  "
        f"skipped_empty={skipped['empty']}  "
        f"skipped_unmatched={skipped['unmatched']}"
    )
    if n_scored == 0:
        print("no pairs to score", file=sys.stderr)
        return 1
    print(format_scores("raw", raw))
    print(format_scores("rescaled", scaled))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
