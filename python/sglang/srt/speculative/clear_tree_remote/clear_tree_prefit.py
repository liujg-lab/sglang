"""HTTP sweep that fills ClearTree H cells before user traffic.

The Target starts this from a daemon thread. It talks only to the local
HTTP server. Draft decides which rounds to keep and writes the fit file.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PrefitConfig:
    url: str
    buckets: Sequence[int]
    max_depth: int
    ks: Sequence[int]
    widths: Sequence[int]
    batches: Sequence[int]
    token_range: Tuple[int, int]
    retries: int = 3
    seed: int = 0
    ready_timeout_s: float = 600.0
    api_key: Optional[str] = None


def prefixes_for(buckets: Sequence[int]) -> list:
    """First bucket uses half its width. Every later bucket uses five eighths."""
    prefixes = []
    for index, bucket in enumerate(buckets):
        fraction = 0.5 if index == 0 else 0.625
        prefixes.append(int(int(bucket) * fraction))
    return prefixes


def depth_for(k: int, width: int, max_depth: int) -> int:
    if int(width) <= 0:
        return 0
    return max(0, min(math.ceil(int(k) / float(width)), int(max_depth) - 1))


def max_new_for_verify(bucket: int, prefix: int, k: int) -> int:
    """Tokens that still leave this verify inside the planned KV bucket."""
    return int(bucket) - int(prefix) - int(k) - 4


def max_new_for_width(d_max: int) -> int:
    return 20 * (int(d_max) + 1)


def _ordered(values, center: int, *, include_center: bool) -> list:
    center = int(center)
    pool = []
    for value in values:
        width = int(value)
        if width <= 0:
            continue
        if width == center and not include_center:
            continue
        pool.append(width)
    return sorted(pool, key=lambda value: (abs(math.log2(value) - math.log2(center)), value))


def build_schedule(cfg: PrefitConfig) -> list:
    """Warm, then verify cells, then one width sweep per prefix, then done."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        flat_k,
        flat_width,
    )

    groups = []
    prefixes = prefixes_for(cfg.buckets)
    for batch in cfg.batches:
        size = int(batch)
        groups.append({"kind": "warm", "b": size, "max_new": 8, "prefix": 32})
        flat_verify = int(flat_k(size))
        flat_w = int(flat_width(size))
        for bucket, prefix in zip(cfg.buckets, prefixes):
            for k in _ordered(cfg.ks, flat_verify, include_center=True):
                depth = depth_for(k, flat_w, cfg.max_depth)
                room = max_new_for_verify(bucket, prefix, k)
                need = 7 * (depth + 1)
                if room < need:
                    logger.info(
                        "[CLEAR] skip prefit cell b=%s bucket=%s k=%s room=%s need=%s",
                        size,
                        bucket,
                        k,
                        room,
                        need,
                    )
                    continue
                groups.append(
                    {
                        "kind": "v",
                        "b": size,
                        "prefix": int(prefix),
                        "k": int(k),
                        "bucket": int(bucket),
                        "max_new": int(min(room, 8 * (depth + 1))),
                        "depth": int(depth),
                    }
                )
            others = _ordered(cfg.widths, flat_w, include_center=False)
            if not others:
                continue
            d_max = max(depth_for(flat_verify, width, cfg.max_depth) for width in others)
            groups.append(
                {
                    "kind": "w",
                    "b": size,
                    "prefix": int(prefix),
                    "bucket": int(bucket),
                    "max_new": int(max_new_for_width(d_max)),
                }
            )
    groups.append({"kind": "done", "b": 1, "max_new": 1, "prefix": 8})
    return groups


def _rid(group: dict, n: int) -> str:
    kind = group["kind"]
    if kind == "warm":
        return f"clearfit-warm-{int(group['b'])}-{int(n)}"
    if kind == "v":
        return (
            f"clearfit-v-{int(group['b'])}-{int(group['prefix'])}-"
            f"{int(group['k'])}-{int(n)}"
        )
    if kind == "w":
        return f"clearfit-w-{int(group['b'])}-{int(group['prefix'])}-{int(n)}"
    return f"clearfit-done-{int(n)}"


def _request(
    url: str,
    method: str,
    payload: Optional[dict],
    timeout: float,
    api_key: Optional[str] = None,
):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    if not body:
        return {}
    return json.loads(body)


def _wait_until(url: str, timeout_s: float, ready) -> None:
    deadline = time.monotonic() + float(timeout_s)
    last = None
    while time.monotonic() < deadline:
        try:
            if ready(url):
                return
        except Exception as exc:
            last = exc
        time.sleep(0.5)
    raise TimeoutError(f"prefit wait failed for {url}: {last}")


def _model_ready(url: str, api_key: Optional[str] = None) -> bool:
    _request(url + "/model_info", "GET", None, 5.0, api_key)
    return True


def _server_idle(url: str, api_key: Optional[str] = None) -> bool:
    body = _request(url + "/v1/loads", "GET", None, 5.0, api_key)
    loads = body.get("loads") if isinstance(body, dict) else None
    if not isinstance(loads, list):
        return False
    running = sum(int(item.get("num_running_reqs") or 0) for item in loads)
    waiting = sum(int(item.get("num_waiting_reqs") or 0) for item in loads)
    return running == 0 and waiting == 0


def _steps(payload) -> int:
    meta = payload.get("meta_info") if isinstance(payload, dict) else None
    if not isinstance(meta, dict):
        meta = {}
    if meta.get("spec_verify_ct") is not None:
        return int(meta["spec_verify_ct"])
    if meta.get("completion_tokens") is not None:
        return int(meta["completion_tokens"])
    if isinstance(payload, dict) and payload.get("completion_tokens") is not None:
        return int(payload["completion_tokens"])
    return 0


def _tokens(rng: random.Random, length: int, token_range: Tuple[int, int]) -> list:
    lo, hi = int(token_range[0]), int(token_range[1])
    if hi <= lo:
        hi = lo + 1
    return [rng.randrange(lo, hi) for _ in range(max(int(length), 1))]


def _post_payload(url: str, payload: dict, api_key: Optional[str] = None) -> int:
    try:
        body = _request(url + "/generate", "POST", payload, 600.0, api_key)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        logger.warning(
            "[CLEAR] prefit request failed rid=%s", payload.get("rid"), exc_info=True
        )
        return 0
    return _steps(body)


def _run_group(cfg: PrefitConfig, group: dict, rng: random.Random) -> dict:
    need = 7 if group["kind"] == "v" else 20 if group["kind"] == "w" else 0
    size = int(group["b"])
    attempts = 0
    steps = []
    limit = max(int(cfg.retries), 1)
    while attempts < limit:
        attempts += 1
        base = (attempts - 1) * size
        payloads = [
            {
                "input_ids": _tokens(rng, int(group.get("prefix") or 8), cfg.token_range),
                "rid": _rid(group, base + index),
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": int(group["max_new"]),
                    "ignore_eos": True,
                },
            }
            for index in range(size)
        ]
        with ThreadPoolExecutor(max_workers=size) as pool:
            futures = [
                pool.submit(_post_payload, cfg.url, payload, cfg.api_key)
                for payload in payloads
            ]
            steps = [int(future.result()) for future in futures]
        if need == 0 or (steps and min(steps) >= need):
            break
    record = {
        "kind": group["kind"],
        "b": size,
        "steps": steps,
        "retries": attempts,
    }
    logger.info("[CLEAR] prefit group %s", record)
    return record


def run_prefit(cfg: PrefitConfig) -> list:
    """Wait until the server is idle, then run the schedule. Do not abort traffic.

    The done request is sent after the server is ready, including when an
    earlier group raises. A wait timeout does not send it.
    """
    logger.info("[CLEAR] prefit waiting for %s", cfg.url)
    _wait_until(
        cfg.url, cfg.ready_timeout_s, partial(_model_ready, api_key=cfg.api_key)
    )
    _wait_until(
        cfg.url, cfg.ready_timeout_s, partial(_server_idle, api_key=cfg.api_key)
    )
    started = time.monotonic()
    rng = random.Random(int(cfg.seed))
    groups = build_schedule(cfg)
    body = [group for group in groups if group["kind"] != "done"]
    done = [group for group in groups if group["kind"] == "done"]
    summary = []
    try:
        for group in body:
            summary.append(_run_group(cfg, group, rng))
    finally:
        for group in done:
            try:
                summary.append(_run_group(cfg, group, rng))
            except Exception:
                logger.warning("[CLEAR] prefit done request failed", exc_info=True)
        elapsed = time.monotonic() - started
        logger.info(
            "[CLEAR] prefit driver finished groups=%d elapsed_s=%.3f",
            len(summary),
            elapsed,
        )
    return summary


def _parse_args(argv=None) -> PrefitConfig:
    parser = argparse.ArgumentParser(description="Rerun a ClearTree prefit sweep")
    parser.add_argument("--url", required=True)
    parser.add_argument("--buckets", nargs="+", type=int, required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--ks", nargs="+", type=int, required=True)
    parser.add_argument("--widths", nargs="+", type=int, required=True)
    parser.add_argument("--batches", nargs="+", type=int, required=True)
    parser.add_argument("--token-lo", type=int, required=True)
    parser.add_argument("--token-hi", type=int, required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ready-timeout-s", type=float, default=600.0)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args(argv)
    return PrefitConfig(
        url=args.url,
        buckets=tuple(args.buckets),
        max_depth=int(args.max_depth),
        ks=tuple(args.ks),
        widths=tuple(args.widths),
        batches=tuple(args.batches),
        token_range=(int(args.token_lo), int(args.token_hi)),
        retries=int(args.retries),
        seed=int(args.seed),
        ready_timeout_s=float(args.ready_timeout_s),
        api_key=args.api_key or None,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_prefit(_parse_args())
