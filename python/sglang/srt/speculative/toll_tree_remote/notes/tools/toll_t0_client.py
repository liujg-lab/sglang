"""Closed-loop client for the OPT T0 cells.

Paths, URLs, and token files come from the command line. This file is not a
CI test. ``serve`` keeps ``concurrency`` requests in flight. ``filter`` keeps
prompts whose target-only generation reaches ``--min-output`` tokens.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def _post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _load_prompts(path: str) -> list:
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            if "input_ids" in item:
                prompts.append(list(item["input_ids"]))
            elif "text" in item:
                prompts.append(item["text"])
            elif "prompt" in item:
                prompts.append(item["prompt"])
            else:
                raise SystemExit(f"prompt row has no text or input_ids: {list(item)[:6]}")
        elif isinstance(item, list):
            prompts.append([int(token) for token in item])
        else:
            prompts.append(str(item))
    return prompts


def _fit_length(ids: list[int], ell: int, pad_token: int) -> list[int]:
    ids = [int(token) for token in ids]
    if len(ids) >= ell:
        return ids[:ell]
    if not ids:
        ids = [pad_token]
    out = list(ids)
    while len(out) < ell:
        out.append(out[len(out) % len(ids)])
    return out


def _payload(prompt, max_new_tokens: int, ignore_eos: bool, return_logprob: bool, top_logprobs_num: int) -> dict:
    sampling = {"temperature": 0, "max_new_tokens": int(max_new_tokens)}
    if ignore_eos:
        sampling["ignore_eos"] = True
    body = {"sampling_params": sampling}
    if isinstance(prompt, list):
        body["input_ids"] = prompt
    else:
        body["text"] = prompt
    if return_logprob:
        body["return_logprob"] = True
        body["top_logprobs_num"] = int(top_logprobs_num)
        body["return_text_in_logprobs"] = False
    return body


def _one(url: str, prompt, max_new_tokens: int, ignore_eos: bool, timeout: float, return_logprob: bool = False, top_logprobs_num: int = 0) -> dict:
    started = time.perf_counter()
    prompt_len = len(prompt) if isinstance(prompt, list) else None
    try:
        data = _post(
            url,
            _payload(prompt, max_new_tokens, ignore_eos, return_logprob, top_logprobs_num),
            timeout,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "error": str(exc),
            "prompt_len": prompt_len,
            "latency_s": time.perf_counter() - started,
        }
    output = data.get("output_ids") or []
    meta = data.get("meta_info") or {}
    return {
        "ok": True,
        "prompt_len": prompt_len if prompt_len is not None else meta.get("prompt_tokens"),
        "output_len": len(output),
        "finish_reason": meta.get("finish_reason"),
        "latency_s": time.perf_counter() - started,
        "output_ids": output,
        "output_top_logprobs": meta.get("output_top_logprobs"),
    }


def _serve(args) -> None:
    prompts = _load_prompts(args.prompts) if args.prompts else []
    if args.ell:
        if not prompts:
            prompts = [[args.pad_token]]
        seed = prompts[0] if isinstance(prompts[0], list) else [args.pad_token]
        prompts = [_fit_length(seed, args.ell, args.pad_token) for _ in range(args.count)]
    elif args.count and len(prompts) < args.count:
        if not prompts:
            raise SystemExit("--prompts is required unless --ell builds them")
        prompts = [prompts[i % len(prompts)] for i in range(args.count)]
    elif args.count:
        prompts = prompts[: args.count]
    if not prompts:
        raise SystemExit("no prompts")

    work: queue.Queue = queue.Queue()
    for index, prompt in enumerate(prompts):
        work.put((index, prompt))
    rows = [None] * len(prompts)
    errors = []

    def worker() -> None:
        while True:
            try:
                index, prompt = work.get_nowait()
            except queue.Empty:
                return
            row = _one(
                args.url,
                prompt,
                args.max_new_tokens,
                args.ignore_eos,
                args.timeout,
                args.return_logprob,
                args.top_logprobs_num,
            )
            row["index"] = index
            rows[index] = row
            if not row["ok"]:
                errors.append(row["error"])
            work.task_done()

    threads = [
        threading.Thread(target=worker, daemon=True) for _ in range(max(1, args.concurrency))
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    ok = [row for row in rows if row and row["ok"]]
    produced = sum(int(row["output_len"] or 0) for row in ok)
    summary = {
        "requests": len(rows),
        "ok": len(ok),
        "errors": len(errors),
        "elapsed_s": elapsed,
        "output_tokens": produced,
        "tok_s": produced / elapsed if elapsed else None,
        "concurrency": args.concurrency,
        "error_sample": errors[:3],
    }
    print(json.dumps(summary))
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if errors and not args.allow_errors:
        raise SystemExit(f"{len(errors)} requests failed")


def _filter(args) -> None:
    prompts = _load_prompts(args.prompts)
    kept = []
    for prompt in prompts:
        row = _one(args.url, prompt, args.min_output, False, args.timeout, False, 0)
        if not row["ok"]:
            continue
        if int(row["output_len"] or 0) >= args.min_output:
            kept.append(prompt if isinstance(prompt, list) else {"text": prompt})
        if args.limit and len(kept) >= args.limit:
            break
    Path(args.out).write_text(
        "\n".join(json.dumps(item) for item in kept) + ("\n" if kept else ""),
        encoding="utf-8",
    )
    print(json.dumps({"seen": len(prompts), "kept": len(kept)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--mode", choices=("serve", "filter"), default="serve")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--ell", type=int, default=0)
    parser.add_argument("--pad-token", type=int, default=100)
    parser.add_argument("--min-output", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--return-logprob", action="store_true")
    parser.add_argument("--top-logprobs-num", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "filter":
        _filter(args)
    else:
        _serve(args)


if __name__ == "__main__":
    main()
