"""One ClearTree layer forward at a fixed batch, width, and prefix.

Runs inside the draft scheduler after CUDA graphs exist. The timed interval
is one ``_run_batch``. Width 1 has no layer graph and stays eager.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import traceback

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.clear_tree_remote.clear_tree_forward import (
    OptLayerForward,
    _extend_span_batch,
    _run_batch,
    _verify_batch_many,
    layer_positions,
    layer_verify_mask,
    opt_extend_chunk_limit,
    pack_verify_attention,
)

BATCHES = (1, 4)
WIDTHS = (1, 2, 4, 8, 16, 32, 64)
GRAPH_WIDTHS = (2, 4, 8, 16, 32, 64)
PREFIXES = (256, 1024, 2048, 4096, 8192)
WARMUP = 3
MEASURE = 20
TOKEN = 16


def run(scheduler) -> None:
    out = os.environ.get("SGLANG_CLEAR_LAYER_OUT", "").strip()
    if not out:
        raise RuntimeError("SGLANG_CLEAR_LAYER_OUT is empty")
    payload = {
        "warmup": WARMUP,
        "measure": MEASURE,
        "batches": list(BATCHES),
        "widths": list(WIDTHS),
        "prefixes": list(PREFIXES),
        "cells": [],
        "error": None,
    }
    try:
        _require_graphs(scheduler)
        for prefix in PREFIXES:
            for batch in BATCHES:
                _sweep_prefix(scheduler, prefix, batch, payload["cells"])
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        print(f"LAYER_BENCH_ERROR {payload['error']}", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, out)
    print(f"LAYER_BENCH_WROTE {out} cells={len(payload['cells'])}", flush=True)


def _require_graphs(scheduler) -> None:
    graph = _graph(scheduler)
    missing = []
    for batch in BATCHES:
        for width in GRAPH_WIDTHS:
            key = graph._make_graph_key(batch, None, width)
            if key not in graph.graphs:
                missing.append(str(key))
    if missing:
        have = sorted(str(key) for key in graph.graphs)
        raise RuntimeError(
            "missing layer graphs: " + ", ".join(missing) + "; have " + ", ".join(have)
        )
    print("LAYER_BENCH_GRAPHS_OK", flush=True)


def _graph(scheduler):
    worker = getattr(scheduler, "model_worker", None) or scheduler.tp_worker
    runner = worker.model_runner
    graph = getattr(runner, "graph_runner", None)
    if graph is None:
        raise RuntimeError("draft CUDA graph runner is missing")
    return graph


def _sweep_prefix(scheduler, prefix: int, batch: int, cells: list) -> None:
    limit = scheduler.req_to_token_pool.req_to_token.shape[1]
    if int(prefix) + max(WIDTHS) > int(limit):
        raise RuntimeError(
            f"prefix {prefix} plus width {max(WIDTHS)} exceeds context {limit}"
        )
    runners = _make_runners(scheduler, batch, prefix)
    try:
        _fill_prefix(scheduler, runners, prefix)
        for width in WIDTHS:
            cell = _time_width(scheduler, runners, prefix, batch, width)
            cells.append(cell)
            print(
                "LAYER_BENCH_CELL"
                f" batch={batch} width={width} prefix={prefix}"
                f" median_us={cell.get('median_us')} path={cell.get('path')}",
                flush=True,
            )
    finally:
        _release_runners(scheduler, runners)


def _make_runners(scheduler, batch: int, prefix: int):
    reqs = []
    for index in range(int(batch)):
        req = Req(
            rid=f"layer-bench-{prefix}-{batch}-{index}",
            origin_input_text="",
            origin_input_ids=[TOKEN] * int(prefix),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.fill_ids = list(req.origin_input_ids)
        reqs.append(req)
    allocated = scheduler.req_to_token_pool.alloc(reqs)
    if allocated is None:
        raise RuntimeError(f"request pool cannot hold batch {batch}")
    return [OptLayerForward(scheduler, req, int(prefix)) for req in reqs]


def _fill_prefix(scheduler, runners, prefix: int) -> None:
    limit = opt_extend_chunk_limit(
        getattr(scheduler.server_args, "chunked_prefill_size", None)
    )
    if limit is None:
        limit = int(prefix)
    for runner in runners:
        while int(runner.req.kv_committed_len or 0) < int(prefix):
            start = int(runner.req.kv_committed_len or 0)
            end = min(int(prefix), start + int(limit))
            _extend_span_batch(scheduler, [runner], [end])
        if int(runner.cached_len) != int(prefix) or int(runner.prefix_len) != int(prefix):
            raise RuntimeError(
                f"prefix fill cached={runner.cached_len} prefix_len={runner.prefix_len} "
                f"wanted {prefix}"
            )


def _time_width(scheduler, runners, prefix: int, batch: int, width: int) -> dict:
    key = f"r{int(width)}_{int(batch)}" if int(width) > 1 else None
    locs = []
    try:
        for runner in runners:
            if int(runner.cached_len) != int(prefix):
                raise RuntimeError(
                    f"cached length {runner.cached_len} left the prefix {prefix}"
                )
            loc = runner._alloc_layer_slots(int(width))
            runner.req.kv_allocated_len = int(prefix) + int(width)
            locs.append(loc)
        forward = _layer_batch(scheduler, runners, prefix, width, locs)
        path = "unknown"
        samples = []
        for step in range(WARMUP + MEASURE):
            _restore(forward)
            torch.cuda.synchronize()
            forward["ev0"].record()
            if step == 0:
                path = _probe(scheduler, forward["batch"])
            else:
                _run_batch(scheduler, forward["batch"])
            forward["ev1"].record()
            forward["ev1"].synchronize()
            samples.append(float(forward["ev0"].elapsed_time(forward["ev1"])) * 1000.0)
        measured = samples[WARMUP:]
        return {
            "batch": int(batch),
            "width": int(width),
            "prefix": int(prefix),
            "sum_ell": int(batch) * int(prefix),
            "median_us": statistics.median(measured),
            "min_us": min(measured),
            "path": path,
            "graph_key": key,
            "samples_us": measured,
        }
    finally:
        for runner in runners:
            runner.layer_slots.clear()
            if int(getattr(runner.req, "kv_allocated_len", 0) or 0) > int(prefix):
                scheduler._clear_rollback(runner.req, int(prefix))
            runner.req.kv_allocated_len = int(prefix)
            runner.req.kv_committed_len = int(prefix)
            runner.cached_len = int(prefix)


def _layer_batch(scheduler, runners, prefix: int, width: int, locs):
    device = scheduler.req_to_token_pool.device
    ancestor = torch.eye(int(width), dtype=torch.bool)
    tokens = [
        torch.full((int(width),), TOKEN, dtype=torch.long, device=device)
        for _ in runners
    ]
    if len(runners) == 1:
        positions = layer_positions(int(prefix), 1, int(width)).to(device=device)
        mask = layer_verify_mask(runners[0], ancestor, device)
    else:
        mask, positions = pack_verify_attention(
            [int(prefix)] * len(runners),
            [ancestor for _ in runners],
            1,
            int(width),
            device,
        )
    batch = _verify_batch_many(
        scheduler,
        runners,
        tokens,
        positions,
        mask,
        torch.cat([loc.reshape(-1) for loc in locs]),
        [int(prefix)] * len(runners),
    )
    if len(runners) == 1:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_attention import (
            bind_opt_cascade,
        )

        bind_opt_cascade(
            batch.spec_info,
            runners[0],
            locs[0],
            ancestor,
            cached_len=int(prefix),
            append_slots=True,
        )
    return {
        "batch": batch,
        "input_ids": batch.input_ids,
        "seq_lens": batch.seq_lens.clone(),
        "seq_lens_cpu": batch.seq_lens_cpu.clone(),
        "orig_seq_lens": batch.orig_seq_lens.clone(),
        "out_cache_loc": batch.out_cache_loc,
        "seq_lens_sum": int(batch.seq_lens_sum),
        "forward_mode": batch.forward_mode,
        "global_forward_mode": batch.global_forward_mode,
        "spec_info": batch.spec_info,
        "ev0": torch.cuda.Event(enable_timing=True),
        "ev1": torch.cuda.Event(enable_timing=True),
    }


def _restore(forward) -> None:
    batch = forward["batch"]
    batch.input_ids = forward["input_ids"]
    batch.seq_lens = forward["seq_lens"]
    batch.seq_lens_cpu = forward["seq_lens_cpu"]
    batch.orig_seq_lens = forward["orig_seq_lens"]
    batch.out_cache_loc = forward["out_cache_loc"]
    batch.seq_lens_sum = forward["seq_lens_sum"]
    batch.output_ids = None
    batch.forward_mode = forward["forward_mode"]
    batch.global_forward_mode = forward["global_forward_mode"]
    batch.spec_info = forward["spec_info"]


def _probe(scheduler, batch) -> str:
    scheduler.cur_batch = batch
    result = scheduler.run_batch(batch)
    logits = getattr(getattr(result, "logits_output", None), "next_token_logits", None)
    if logits is None:
        raise RuntimeError("layer bench forward returned no logits")
    flag = getattr(result, "can_run_cuda_graph", None)
    if flag is None:
        return "unknown"
    return "graph" if bool(flag) else "eager"


def _release_runners(scheduler, runners) -> None:
    for runner in runners:
        req = runner.req
        try:
            if req.req_pool_idx is not None and int(getattr(req, "kv_allocated_len", 0) or 0) > 0:
                scheduler._clear_rollback(req, 0)
        finally:
            if req.req_pool_idx is not None:
                scheduler.req_to_token_pool.free(req)


def main() -> None:
    print("bench_layer.py runs inside the draft scheduler", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
