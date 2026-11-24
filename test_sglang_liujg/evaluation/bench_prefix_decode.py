#!/usr/bin/env python3
import argparse
import json
import random
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


SEED_TEXT = (
    "A systematic benchmark changes one variable at a time. Stable measurements "
    "require an explicit warmup, randomized trial order, independent samples, "
    "and careful separation of input processing from output generation. "
    "Record the hardware, model, request sizes, cache settings, latency, and "
    "throughput for every run. Compare the observations before drawing conclusions.\n"
)


def request(base_url, route, payload=None):
    if payload is None:
        return urlopen(base_url + route, timeout=180)
    return urlopen(
        Request(
            base_url + route,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=180,
    )


def read_json(base_url, *routes, payload=None):
    for route in routes[:-1]:
        try:
            with request(base_url, route, payload) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code != 404:
                raise
    with request(base_url, routes[-1], payload) as response:
        return json.load(response)


def measure(base_url, ids, output_tokens):
    payload = {
        "input_ids": ids,
        "stream": True,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
            "stream_interval": 1,
        },
    }
    first = last = None
    first_count = count = 0
    meta = {}
    done = False
    start = time.perf_counter()
    with request(base_url, "/generate", payload) as response:
        for line in response:
            arrived = time.perf_counter()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            meta = event.get("meta_info", {})
            new_count = meta.get("completion_tokens", 0)
            if new_count > count:
                if first is None:
                    first, first_count = arrived, new_count
                last, count = arrived, new_count

    prompt_tokens = meta.get("prompt_tokens")
    cached_tokens = meta.get("cached_tokens")
    if not done or (prompt_tokens, cached_tokens, first_count, count) != (
        len(ids), 0, 1, output_tokens
    ):
        raise RuntimeError(
            f"Unexpected response: done={done}, prompt={prompt_tokens}/{len(ids)}, "
            f"cached={cached_tokens}, first_chunk={first_count}, output={count}/{output_tokens}"
        )
    return {
        "prefix_tokens": prompt_tokens,
        "completion_tokens": count,
        "cached_tokens": cached_tokens,
        "ttft_ms": 1e3 * (first - start),
        "decode_tokens": count - first_count,
        "decode_s": last - first,
    }


def main():
    parser = argparse.ArgumentParser(description="Serial AR prefix-length sweep")
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--prefixes", nargs="+", type=int, default=[512, 2048, 3072, 4096, 8192, 12288])
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.prefixes) < 2 or args.output_tokens < 2 or args.repeats < 1:
        parser.error("prefixes and output-tokens must be >= 2; repeats must be >= 1")
    if len(set(args.prefixes)) != len(args.prefixes):
        parser.error("prefixes must be distinct")

    url = args.url.rstrip("/")
    info = read_json(url, "/server_info", "/get_server_info")
    if info.get("speculative_algorithm") not in (None, "NONE"):
        raise SystemExit("Expected an autoregressive server, not speculative decoding")
    config = ("model_path", "tp_size", "attention_backend", "page_size", "kv_cache_dtype", "disable_cuda_graph")
    print(json.dumps({"server": {k: info.get(k) for k in config}, "args": vars(args)}), flush=True)

    tokenized = read_json(
        url, "/tokenize", "/v1/tokenize",
        payload={"prompt": SEED_TEXT, "add_special_tokens": False},
    )
    tokens = tokenized["tokens"]
    markers = list(dict.fromkeys(tokens))
    if len(markers) < 1 + len(args.prefixes) * args.repeats:
        parser.error("too many trials for unique cache-avoiding first tokens")
    if max(args.prefixes) + args.output_tokens > tokenized["max_model_len"]:
        parser.error("largest prefix plus output exceeds the model context")
    markers = iter(markers)

    def run(length):
        # Different first tokens avoid radix hits; equal suffixes keep the generation prompt comparable.
        body = (tokens * (1 + length // len(tokens)))[-(length - 1):]
        return measure(url, [next(markers)] + body, args.output_tokens)

    print(json.dumps({"warmup": run(max(args.prefixes))}), flush=True)
    rng = random.Random(0)
    results = {length: [] for length in args.prefixes}
    for repeat in range(1, args.repeats + 1):
        order = rng.sample(args.prefixes, len(args.prefixes))
        for length in order:
            row = run(length)
            results[length].append(row)
            print(json.dumps({"repeat": repeat, **row}), flush=True)

    for length, rows in sorted(results.items()):
        decode_tokens = sum(row["decode_tokens"] for row in rows)
        decode_s = sum(row["decode_s"] for row in rows)
        print(json.dumps({"summary": {
            "prefix_tokens": length,
            "samples": len(rows),
            "decode_tok_s": round(decode_tokens / decode_s, 2),
            "decode_ms_per_token": round(1e3 * decode_s / decode_tokens, 2),
            "median_ttft_ms": round(statistics.median(row["ttft_ms"] for row in rows), 2),
        }}), flush=True)


if __name__ == "__main__":
    main()
