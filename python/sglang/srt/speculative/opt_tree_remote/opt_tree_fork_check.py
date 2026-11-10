"""Host checks for one OPT verify round. Does not change acceptance."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import torch

HEAD = (2441, 38891, 891, 220)
KV_LEN = 4172
_DONE = False


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _log(line: str) -> None:
    logging.warning("FORKCHECK %s", line)


def _parents_of(req) -> tuple[list[int], list[int]]:
    packed = req.draft_tokens_and_logits or {}
    tokens = [int(token) for token in (packed.get("draft_tokens") or [])]
    parents = [int(parent) for parent in (packed.get("parent_list") or [])]
    return tokens, parents


def _accepted(res, row: int, num_verify: int):
    lengths = [int(item) for item in res.accept_length_per_req_cpu]
    accepted = [int(item) for item in res.accepted_indices.detach().cpu().tolist()]
    verified = [int(item) for item in res.verified_id.detach().cpu().tolist()]
    cursor = 0
    for index, count in enumerate(lengths):
        span = count + 1
        if index == row:
            return (
                count,
                [item - row * num_verify for item in accepted[cursor : cursor + span]],
                verified[cursor : cursor + span],
            )
        cursor += span
    return 0, [], []


def _hole_line(spec_info, prefix: int, sequence, covered, pool, table, pool_idx: int, page: int) -> str:
    missing = [index for index, token in enumerate(covered) if token is None]
    positions = getattr(spec_info, "positions", None)
    if positions is None:
        head = []
    else:
        head = [int(item) for item in positions.detach().cpu().reshape(-1)[:9].tolist()]
    shown = "none" if not missing else ",".join(str(item) for item in missing[:8])
    line = f"hole p={shown} seq={prefix} n_ids={len(sequence)} positions={head}"
    visible = [item for item in missing if item <= 4167]
    if not visible:
        return line
    pos = visible[0]
    try:
        layer_id = int(getattr(pool, "start_layer", 0))
        value = pool.get_value_buffer(layer_id)
        slot = table[pool_idx, pos].to(device=value.device, dtype=torch.long)
        cached = _cached_values(value, slot.reshape(1), page).float().reshape(-1)
        peak = float(cached.abs().max().item())
        zero = int(bool(torch.count_nonzero(cached).item() == 0))
        return f"{line} v0_p={pos} v0_max={peak:.6g} v0_zero={zero}"
    except Exception as exc:
        return f"{line} v0_err={exc}"


def _layer_summary() -> str:
    try:
        from sglang.srt.speculative.opt_tree_remote.opt_tree_layer_check import (
            write_spec_values,
        )

        return write_spec_values()
    except Exception as exc:
        return f"k err {exc}"


def _check_h(tokens: Sequence[int], parents: Sequence[int], count: int, slots, ids) -> str:
    if count == 0:
        return "h pass nodes=0"
    for step in range(count):
        slot = slots[step + 1]
        draft_i = slot - 1
        if draft_i < 0 or draft_i >= len(tokens) or draft_i >= len(parents):
            return f"h fail step={step} slot={slot} outside draft"
        if tokens[draft_i] != ids[step]:
            return (
                f"h fail step={step} node={draft_i} token={tokens[draft_i]} "
                f"committed={ids[step]}"
            )
        previous = slots[step]
        expect = -1 if previous == 0 else previous - 1
        if parents[draft_i] != expect:
            return (
                f"h fail step={step} node={draft_i} parent={parents[draft_i]} "
                f"expect={expect}"
            )
    return f"h pass nodes={count}"


def _ancestors(parents: Sequence[int], slot: int) -> set[int]:
    verify_parent = [-1] + [0 if parent < 0 else parent + 1 for parent in parents]
    found = set()
    node = slot
    while node >= 0:
        if node in found:
            break
        found.add(node)
        if node >= len(verify_parent):
            break
        node = verify_parent[node]
    return found


def _check_i(md, row: int, prefix: int, queries: int, slot: int, parents) -> str:
    if md is None or row >= md.blocked_mask.shape[0]:
        return "i fail mask missing"
    mask = md.blocked_mask
    if mask.ndim != 4 or not (0 <= slot < mask.shape[2]):
        return f"i fail query={slot} shape={tuple(mask.shape)}"
    end = prefix + queries
    if end > mask.shape[-1]:
        return f"i fail prefix={prefix} queries={queries} width={mask.shape[-1]}"
    flags = mask[row, 0, slot, prefix:end].detach().cpu().tolist()
    visible = _ancestors(parents, slot)
    for column, blocked in enumerate(flags):
        attend = column in visible
        if bool(blocked) == attend:
            return (
                f"i fail query={slot} column={prefix + column} "
                f"blocked={int(bool(blocked))} ancestor={int(attend)}"
            )
    return f"i pass query={slot} columns={queries}"


def _check_f(table, pool: int, kv_len: int, page: int) -> str:
    if kv_len > table.shape[1]:
        return f"f fail kv={kv_len} width={table.shape[1]}"
    slots = [int(item) for item in table[pool, :kv_len].detach().cpu().tolist()]
    for pos, slot in enumerate(slots):
        start = (pos // page) * page
        page_id = slots[start] // page
        expect = page_id * page + (pos % page)
        if slot != expect:
            return f"f fail p={pos} slot={slot} expect={expect} page={page_id}"
    return f"f pass len={kv_len}"


def _text_model(model):
    current = model
    for _ in range(4):
        layers = getattr(current, "layers", None)
        embed = getattr(current, "embed_tokens", None)
        if layers is not None and embed is not None:
            return current
        nxt = getattr(current, "model", None)
        if nxt is None:
            break
        current = nxt
    raise RuntimeError("text model layers are not on this module")


def _cached_values(value, slots, page: int):
    if value.ndim == 4 and value.shape[1] == page and page != 1:
        return value[slots // page, slots % page]
    return value[slots]


def _over_two_ulps(got: torch.Tensor, cached: torch.Tensor) -> torch.Tensor:
    """True where the absolute gap exceeds two bf16 steps at that magnitude."""
    diff = (got.float() - cached.float()).abs()
    mag = torch.maximum(got.float().abs(), cached.float().abs())
    _, exponent = torch.frexp(mag)
    step = torch.ldexp(torch.ones_like(diff), exponent - 8)
    return diff > (step * 2)


def _check_j(
    model, pool, table, pool_idx: int, tokens: List[Optional[int]], page: int
) -> str:
    text = _text_model(model)
    layer = text.layers[0]
    attn = layer.self_attn
    known = [index for index, token in enumerate(tokens) if token is not None]
    if not known:
        return "j fail no tokens"
    layer_id = int(getattr(pool, "start_layer", 0))
    value = pool.get_value_buffer(layer_id)
    ids = torch.tensor([tokens[index] for index in known], dtype=torch.long, device=value.device)
    worst = 0.0
    worst_at = known[0]
    mismatch = None
    with torch.inference_mode():
        for start in range(0, ids.shape[0], 128):
            chunk_ids = ids[start : start + 128]
            hidden = layer.input_layernorm(text.embed_tokens(chunk_ids))
            qkv, _ = attn.qkv_proj(hidden)
            got = qkv[:, attn.q_size + attn.kv_size : attn.q_size + 2 * attn.kv_size]
            positions = known[start : start + chunk_ids.shape[0]]
            slots = table[
                pool_idx, torch.tensor(positions, dtype=torch.long, device=table.device)
            ].to(device=value.device, dtype=torch.long)
            cached = _cached_values(value, slots, page)
            flat = got.reshape(got.shape[0], -1).to(dtype=cached.dtype)
            cached_flat = cached.reshape(cached.shape[0], -1)
            if flat.shape != cached_flat.shape:
                return (
                    f"j fail shape got={tuple(flat.shape)} cache={tuple(cached_flat.shape)} "
                    f"buf={tuple(value.shape)}"
                )
            bad = _over_two_ulps(flat, cached_flat)
            amount = float((flat.float() - cached_flat.float()).abs().max().item())
            if amount > worst:
                worst = amount
                worst_at = positions[int(torch.argmax(
                    (flat.float() - cached_flat.float()).abs().reshape(flat.shape[0], -1).max(dim=1).values
                ).item())]
            if mismatch is None and bool(bad.any().item()):
                token_bad = bad.reshape(flat.shape[0], -1).any(dim=1)
                mismatch = positions[int(torch.argmax(token_bad.int()).item())]
    if mismatch is None:
        return f"j pass n={len(known)} max_abs={worst:.6g} rank={_rank()}"
    return (
        f"j fail p={mismatch} max_abs={worst:.6g} at={worst_at} "
        f"n={len(known)} rank={_rank()}"
    )


def fork_check(batch, spec_info, res, seq_lens_cpu_pre, model, pool, page_size: int) -> None:
    """One round: parent chain, mask columns, page slots, and layer-0 V."""
    global _DONE
    if _DONE or seq_lens_cpu_pre is None:
        return
    reqs = list(getattr(batch, "reqs", []) or [])
    md = getattr(batch, "_fork_dump_fia", None)
    if md is None or not reqs:
        return
    row = None
    for index, req in enumerate(reqs):
        origin = tuple(int(token) for token in list(getattr(req, "origin_input_ids", []) or [])[:4])
        kv = int(md.kv_lens_cpu[index]) if index < len(md.kv_lens_cpu) else -1
        if origin == HEAD and kv == KV_LEN:
            row = index
            break
    if row is None:
        return
    _DONE = True
    req = reqs[row]
    tokens, parents = _parents_of(req)
    num_verify = int(spec_info.draft_token_num)
    count, slots, ids = _accepted(res, row, num_verify)
    prefix = int(seq_lens_cpu_pre[row])
    rank = _rank()
    if rank == 0:
        _log(_check_h(tokens, parents, count, slots, ids))
        slot = 0
        for index, token in enumerate(ids):
            if token == 56056:
                slot = slots[index]
                break
        _log(_check_i(md, row, prefix, num_verify, slot, parents))
        table = batch.req_to_token_pool.req_to_token
        pool_idx = int(batch.req_pool_indices[row].item())
        _log(_check_f(table, pool_idx, KV_LEN, int(page_size)))
    else:
        table = batch.req_to_token_pool.req_to_token
        pool_idx = int(batch.req_pool_indices[row].item())
    sequence = [int(token) for token in list(req.origin_input_ids) + list(req.output_ids)]
    covered: List[Optional[int]] = [None] * KV_LEN
    for pos, token in enumerate(sequence[: min(prefix, KV_LEN)]):
        covered[pos] = token
    try:
        from sglang.srt.speculative.opt_tree_remote.opt_tree_layout import (
            _validate_parents,
        )

        for draft_i, hop in enumerate(_validate_parents(parents)):
            pos = prefix + hop + 1
            if 0 <= pos < KV_LEN and draft_i < len(tokens):
                covered[pos] = tokens[draft_i]
    except Exception as exc:
        if rank == 0:
            _log(f"j fail parents {exc}")
        return
    if rank == 0:
        _log(_hole_line(spec_info, prefix, sequence, covered, pool, table, pool_idx, int(page_size)))
    _log(_check_j(model, pool, table, pool_idx, covered, int(page_size)))
    summary = _layer_summary()
    if rank == 0:
        _log(summary)
