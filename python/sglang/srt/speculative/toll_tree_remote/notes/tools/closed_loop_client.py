#!/usr/bin/env python3
"""Closed-loop client for SGLang /generate. Keeps C requests in flight.

Prompt file: JSONL, one {"idx": int, "input_ids": [int, ...]} per line.
Output: JSONL, one record per finished request, plus <out>.args.json.

  python3 closed_loop_client.py --url http://<target>:<port> \
      --prompts p.jsonl --concurrency 4 --max-new-tokens 256 --ignore-eos \
      --out run.jsonl [--top-logprobs 5] [--stream] [--duration-s 600 --cycle]

Field names follow SGLang's /generate response (meta_info.id,
completion_tokens, output_token_logprobs, output_top_logprobs). Check them
against your server version on one request before a long run.
"""
import argparse
import itertools
import json
import random
import threading
import time

import requests


def load_prompts(path):
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                items.append({"idx": int(rec["idx"]), "input_ids": list(rec["input_ids"])})
    return items


def _ids_from(ret):
    if isinstance(ret.get("output_ids"), list):
        return [int(x) for x in ret["output_ids"]]
    lps = (ret.get("meta_info") or {}).get("output_token_logprobs") or []
    return [int(x[1]) for x in lps]


def _top_from(meta):
    tops = meta.get("output_top_logprobs") or []
    out = []
    for pos in tops:
        out.append([[float(x[0]), int(x[1])] for x in (pos or [])])
    return out


def run_one(session, url, item, args, max_new):
    sp = {"temperature": 0, "max_new_tokens": max_new, "ignore_eos": args.ignore_eos}
    payload = {"input_ids": item["input_ids"], "sampling_params": sp}
    if args.top_logprobs:
        payload.update(
            return_logprob=True, top_logprobs_num=args.top_logprobs, logprob_start_len=-1
        )
    rec = {"idx": item["idx"], "prompt_len": len(item["input_ids"]), "max_new": max_new}
    t0 = time.time()
    rec["t_start"] = t0
    if args.stream:
        payload["stream"] = True
        arrivals = []
        last = None
        with session.post(url + "/generate", json=payload, stream=True, timeout=args.timeout) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                last = json.loads(body)
                meta = last.get("meta_info") or {}
                arrivals.append([time.time(), int(meta.get("completion_tokens", 0))])
        meta = (last or {}).get("meta_info") or {}
        rec["arrivals"] = arrivals
    else:
        r = session.post(url + "/generate", json=payload, timeout=args.timeout)
        r.raise_for_status()
        last = r.json()
        meta = last.get("meta_info") or {}
        rec["output_ids"] = _ids_from(last)
        if args.top_logprobs:
            rec["top"] = _top_from(meta)
    rec["t_end"] = time.time()
    rec["rid"] = meta.get("id")
    rec["completion_tokens"] = int(meta.get("completion_tokens", 0))
    rec["cached_tokens"] = meta.get("cached_tokens")
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-new-jitter", type=int, default=0,
                   help="uniform +-jitter on max_new_tokens to avoid synchronized finishes")
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--top-logprobs", type=int, default=0)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--cycle", action="store_true", help="reuse prompts until --duration-s")
    p.add_argument("--duration-s", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    prompts = load_prompts(args.prompts)
    source = itertools.cycle(prompts) if args.cycle else iter(prompts)
    rng = random.Random(args.seed)
    lock = threading.Lock()
    stop_at = time.time() + args.duration_s if args.duration_s > 0 else None
    out = open(args.out, "w")
    json.dump(vars(args) | {"n_prompts": len(prompts), "wall_start": time.time()},
              open(args.out + ".args.json", "w"), indent=1)

    def next_job():
        with lock:
            if stop_at is not None and time.time() >= stop_at:
                return None
            item = next(source, None)
            if item is None:
                return None
            j = args.max_new_jitter
            return item, args.max_new_tokens + (rng.randint(-j, j) if j else 0)

    errors = []

    def worker():
        session = requests.Session()
        while True:
            job = next_job()
            if job is None:
                return
            try:
                rec = run_one(session, args.url, job[0], args, job[1])
            except Exception as exc:  # keep the loop alive, record the failure
                rec = {"idx": job[0]["idx"], "error": repr(exc), "t_end": time.time()}
                errors.append(rec)
            with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()

    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out.close()
    print(f"done, errors={len(errors)}")


if __name__ == "__main__":
    main()
