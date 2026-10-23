#!/usr/bin/env python3
"""RSVQA 生成结果的大模型正确性评价。

功能
    读取 eval_spectre_rsvqa_serving.py 写出的 jsonl，不重新生成答案。
    每条把图片、问题、参考答案和模型输出交给另一台已启动的视觉模型。
    默认裁判地址是 127.0.0.1:31000 的 /generate，不打被评测的 Target :30000。
    带 error 的生成失败条不送给裁判。

    图片用 image_data URI 和 post_generate 发送。prompt 只在本脚本组装，
    不走 eval_spectre_rsteller.build_prompt。用户轮按 Qwen3.8 的 chat template
    把图像标记直接接在文本前。
    思考默认关闭：助手开头是空的思考块。用 --thinking 开启，此时 system
    写入模板里的 xhigh 说明，助手开头停在未闭合的 <think>。
    开启时只解析最后一个 </think> 之后的 JSON。--max-new-tokens 默认 4096。
    图片路径用 --img-dir 加上条目里的相对 image。

打分
    提示词为英文。correctness 只能是 0 或 1。
    1 与参考答案同义。忽略大小写、空白和 m2 / m²。
    519 m2 对 519m2 为 1。0、0.0、0m2 为 1。
    0 是 yes/no 答反、数字不对或面积数值不对。
    7 对 2、yes 对 no、519m2 对 1000000 都是 0。
    其它整数记解析失败。图片仍会送给裁判。

    缺图、HTTP 失败和解析失败写入 judge_error，不进准确率分母。

汇总
    总条数、生成失败、缺图、裁判失败、成功打分条数。
    accuracy = 答对条数 / 成功打分条数。
    按问题首句分成 presence / comparison / count / area / other。

启动裁判（NPU，端口 31000，不占用被评测 Target 的 30000）
    权重是 Qwen3_5ForConditionalGeneration（model_type=qwen3_5），带视觉塔。
    export ASCEND_RT_VISIBLE_DEVICES=2,3
    python -m sglang.launch_server \
      --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3.8-27B \
      --device npu --tp 2 \
      --host 127.0.0.1 --port 30000 \
      --attention-backend ascend \
      --mm-attention-backend ascend_attn \
      --page-size 128 \
      --disable-cuda-graph
    该模型含线性注意力。NPU 能否跑通是服务启动问题，不在本客户端里改。

运行（仓库根目录，裁判已在 :30000 就绪；默认关闭思考）
    python test_sglang_liujg/evaluation/eval_rsvqa_judge.py \
      --jsonl test_sglang_liujg/result/RSVQA_spectre_serving_results.<run_id>.jsonl \
      --judge-port 30000 --max-items 1
    python test_sglang_liujg/evaluation/eval_rsvqa_judge.py \
      --jsonl test_sglang_liujg/result/RSVQA_spectre_serving_results.<run_id>.jsonl \
      --judge-port 30000 --thinking --max-items 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from eval_spectre_rsteller import (
    image_data_uri,
    mean,
    post_generate,
    strip_image_tag,
)
from eval_spectre_rsvqa_serving import DEFAULT_IMG_DIR

QUESTION_TYPES = ("presence", "comparison", "count", "area", "other")
SCORE_LIMITS = {
    "correctness": (0, 1),
}

# Copied from Qwen3.8-27B chat_template when enable_thinking is true and
# reasoning_effort is the default xhigh.
REASONING_XHIGH = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)
THINK_END = "</think>"

JUDGE_INSTRUCTION = """You are a judge for remote-sensing visual question answering.
Look at the image, the question, the reference answer, and the model output.
Decide only whether the model output means the same thing as the reference answer.
Return one JSON object and no other text. correctness must be the integer 0 or 1.

Ignore case, whitespace, and the difference between m2 and m².
1: same meaning. Reference 519m2 and output 519 m2 are 1. The values 0, 0.0, and 0m2 are 1.
0: the yes/no answer is opposite, the number is different, or the area value is different. Reference 7 and output 2 are 0. Reference yes and output no are 0. Reference 519m2 and output 1000000 are 0.

brief is one English sentence giving the reason.
JSON format:
{"correctness":1,"brief":"..."}
"""


def question_type(question: str) -> str:
    """Classify a RSVQA question from its first line."""
    first = strip_image_tag(question).split("\n", 1)[0].strip().lower()
    if first.startswith("what is the area"):
        return "area"
    if first.startswith(("how many", "what is the number", "what is the amount")):
        return "count"
    if first.startswith(("are there more", "are there less", "is the number")):
        return "comparison"
    if first.startswith(("is there", "is a", "are there")):
        return "presence"
    return "other"


def _int_score(value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"score {value!r} is not an integer in [{low}, {high}]")
    return value


def parse_judge_text(text: str, *, thinking: bool = False) -> dict[str, Any]:
    """Parse one judge JSON object. Non-integers and out-of-range scores fail.

    With thinking on, only the text after the last ``</think>`` is scored.
    Braces inside the reasoning are ignored. A missing close tag means the
    generation was cut off before the answer.
    """
    body = text
    if thinking:
        mark = text.rfind(THINK_END)
        if mark < 0:
            raise ValueError("thinking did not finish: missing </think>")
        body = text[mark + len(THINK_END) :]
    start = body.find("{")
    if start < 0:
        raise ValueError("judge output has no JSON object")
    obj, _ = json.JSONDecoder().raw_decode(body[start:])
    if not isinstance(obj, dict):
        raise ValueError("judge JSON is not an object")
    scores = {
        key: _int_score(obj.get(key), low, high)
        for key, (low, high) in SCORE_LIMITS.items()
    }
    brief = obj.get("brief") or ""
    if not isinstance(brief, str):
        brief = str(brief)
    scores["judge_brief"] = brief.strip()
    return scores


def build_judge_prompt(user_text: str, *, thinking: bool) -> str:
    """Qwen3.8-27B wrapper used only by this judge.

    The image placeholder is glued to the following text. Thinking on matches
    enable_thinking=true: an xhigh system line and an unclosed ``<think>``.
    Thinking off matches enable_thinking=false: an empty think block and no
    reasoning system line. RSVQA generation still uses
    eval_spectre_rsteller.build_prompt.
    """
    text = strip_image_tag(user_text)
    user = (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{text}<|im_end|>\n"
    )
    if thinking:
        system = f"<|im_start|>system\n{REASONING_XHIGH}<|im_end|>\n"
        return system + user + "<|im_start|>assistant\n<think>\n"
    return user + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def judge_user_text(question: str, ground_truth: Any, model_output: Any) -> str:
    return (
        f"{JUDGE_INSTRUCTION}\n"
        f"问题：\n{strip_image_tag(question)}\n\n"
        f"参考答案：\n{ground_truth if ground_truth is not None else ''}\n\n"
        f"模型输出：\n{model_output if model_output is not None else ''}\n"
    )


def iter_jsonl(path: Path, max_items: int | None) -> Iterator[dict[str, Any]]:
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path} line {seen + 1} is not a JSON object")
            yield item
            seen += 1
            if max_items is not None and seen >= max_items:
                return


def _blank_scores(rec: dict[str, Any], judge_error: str | None) -> dict[str, Any]:
    rec["correctness"] = None
    rec["judge_brief"] = ""
    rec["judge_error"] = judge_error
    return rec


def judge_one(
    rec: dict[str, Any],
    generate_url: str,
    img_dir: Path,
    sampling: dict[str, Any],
    timeout: float,
    *,
    thinking: bool,
) -> dict[str, Any]:
    rec = dict(rec)
    rec["question_type"] = question_type(str(rec.get("question") or ""))
    if rec.get("error"):
        return _blank_scores(rec, None)
    image_rel = rec.get("image") or ""
    img_path = img_dir / str(image_rel)
    if not image_rel or not img_path.is_file():
        return _blank_scores(rec, f"image not found: {img_path}")
    prompt = build_judge_prompt(
        judge_user_text(
            str(rec.get("question") or ""),
            rec.get("ground_truth"),
            rec.get("model_output"),
        ),
        thinking=thinking,
    )
    payload = {
        "text": prompt,
        "image_data": image_data_uri(img_path),
        "sampling_params": sampling,
    }
    raw = ""
    try:
        result = post_generate(generate_url, payload, timeout)
        raw = result.get("text") if isinstance(result, dict) else ""
        if not isinstance(raw, str):
            raw = str(raw or "")
        scores = parse_judge_text(raw, thinking=thinking)
    except Exception as exc:  # noqa: BLE001 — keep the remaining items going
        failed = _blank_scores(rec, str(exc))
        failed["judge_raw"] = raw
        return failed
    rec.update(scores)
    rec["judge_error"] = None
    rec["judge_raw"] = raw
    return rec


def _accuracy_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for rec in records if rec.get("correctness") == 1)
    return {
        "n": len(records),
        "correct": correct,
        "accuracy": mean([1.0 if rec.get("correctness") == 1 else 0.0 for rec in records]),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [rec for rec in records if rec.get("judge_error") is None and not rec.get("error")]
    by_type = {
        name: _accuracy_block([rec for rec in scored if rec.get("question_type") == name])
        for name in QUESTION_TYPES
    }
    totals = _accuracy_block(scored)
    return {
        "n_total": len(records),
        "gen_error": sum(1 for rec in records if rec.get("error")),
        "missing_image": sum(
            1
            for rec in records
            if isinstance(rec.get("judge_error"), str)
            and rec["judge_error"].startswith("image not found:")
        ),
        "judge_fail": sum(
            1
            for rec in records
            if rec.get("judge_error")
            and not str(rec["judge_error"]).startswith("image not found:")
        ),
        "scored": totals["n"],
        "correct": totals["correct"],
        "accuracy": totals["accuracy"],
        "by_type": by_type,
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def format_summary(stats: dict[str, Any]) -> str:
    lines = [
        (
            f"items={stats['scored']}/{stats['n_total']}  "
            f"gen_error={stats['gen_error']}  "
            f"missing_image={stats['missing_image']}  "
            f"judge_fail={stats['judge_fail']}"
        ),
        (
            f"accuracy={_fmt(stats['accuracy'])}  "
            f"correct={stats['correct']}/{stats['scored']}"
        ),
    ]
    for name in QUESTION_TYPES:
        block = stats["by_type"][name]
        lines.append(
            f"  {name:<12} n={block['n']:<5} "
            f"accuracy={_fmt(block['accuracy'])}  "
            f"correct={block['correct']}/{block['n']}"
        )
    return "\n".join(lines)


def print_item(idx: int, rec: dict[str, Any]) -> None:
    if rec.get("error"):
        print(f"  [{idx}] {rec.get('question_type')}  GEN_ERROR {rec['error']}")
        return
    if rec.get("judge_error"):
        print(f"  [{idx}] {rec.get('question_type')}  JUDGE_ERROR {rec['judge_error']}")
        return
    print(
        f"  [{idx}] {rec.get('question_type'):<12} "
        f"correct={rec['correctness']}"
    )


def run_judge(args: argparse.Namespace) -> int:
    if args.output_jsonl.resolve() == args.jsonl.resolve():
        raise SystemExit("--output-jsonl must be a different file from --jsonl")
    generate_url = f"http://{args.judge_host}:{args.judge_port}/generate"
    sampling = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"judge={generate_url}  img_dir={args.img_dir}  "
        f"thinking={args.thinking}  max_new_tokens={args.max_new_tokens}"
    )
    print(f"input={args.jsonl}")

    records: list[dict[str, Any]] = []
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(iter_jsonl(args.jsonl, args.max_items)):
            rec = judge_one(
                item,
                generate_url,
                args.img_dir,
                sampling,
                args.timeout,
                thinking=args.thinking,
            )
            print_item(idx, rec)
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(rec)
    if not records:
        raise SystemExit(f"no items in {args.jsonl}")

    stats = summarize(records)
    payload = {
        "source_jsonl": str(args.jsonl),
        "judge_url": generate_url,
        "img_dir": str(args.img_dir),
        "results_jsonl": str(args.output_jsonl),
        "thinking": args.thinking,
        **stats,
    }
    args.output_summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(format_summary(stats))
    print(f"wrote {args.output_jsonl}")
    print(f"wrote {args.output_summary}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge RSVQA serving jsonl with a larger VL model")
    p.add_argument("--jsonl", type=Path, required=True, help="Serving result jsonl to score")
    p.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    p.add_argument("--judge-host", default="127.0.0.1")
    p.add_argument("--judge-port", type=int, default=30000)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Completion budget. Default 4096 so xhigh thinking can finish before the JSON.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Qwen3.8 thinking. Default: off. --thinking leaves <think> open.",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Score only the first N jsonl records. Default: all.",
    )
    p.add_argument("--output-jsonl", type=Path, default=None)
    p.add_argument("--output-summary", type=Path, default=None)
    args = p.parse_args()
    if args.output_jsonl is None:
        args.output_jsonl = args.jsonl.with_name(args.jsonl.stem + ".judge.jsonl")
    if args.output_summary is None:
        args.output_summary = args.jsonl.with_name(args.jsonl.stem + ".judge_summary.json")
    return args


def main() -> int:
    args = parse_args()
    if args.max_items is not None and args.max_items < 1:
        raise SystemExit("--max-items must be >= 1")
    if not args.jsonl.is_file():
        raise SystemExit(f"jsonl not found: {args.jsonl}")
    return run_judge(args)


if __name__ == "__main__":
    raise SystemExit(main())
