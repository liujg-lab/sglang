"""One-round checks for the nine tree slots. Does not change the forward."""

from __future__ import annotations

import logging
import os

import torch

PREFIX = 4163
QUERIES = 9
KV_LEN = 4172

_written: list = []
_cos: list = []
_mask = None
_parents = None
_loc = None
_table = None
_done = False


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _log(line: str) -> None:
    logging.warning("FORKTREE %s", line)


def _gather(buf: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    page = 128
    if buf.ndim == 4 and int(buf.shape[1]) == page:
        return buf[slots // page, slots % page]
    if buf.ndim == 4 and int(buf.shape[1]) == 1:
        return buf[slots, 0]
    return buf[slots]


def note_written(forward_batch, k: torch.Tensor, v: torch.Tensor, layer) -> None:
    """Device-side equality of the just-written rows. No host sync."""
    if _done or k is None or k.shape[0] < QUERIES:
        return
    loc = getattr(forward_batch, "out_cache_loc", None)
    if loc is None or loc.numel() < QUERIES:
        return
    try:
        slots = loc[:QUERIES].to(dtype=torch.long)
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        got_k = _gather(k_cache, slots).reshape(QUERIES, -1)
        got_v = _gather(v_cache, slots).reshape(QUERIES, -1)
        key = k[:QUERIES].reshape(QUERIES, -1)
        value = v[:QUERIES].reshape(QUERIES, -1)
        if got_k.shape[-1] != key.shape[-1] or got_v.shape[-1] != value.shape[-1]:
            _written.append((int(layer.layer_id), None))
            return
        same = (got_k == key).all(dim=-1) & (got_v == value).all(dim=-1)
        _written.append((int(layer.layer_id), same.detach()))
    except Exception as exc:
        if _rank() == 0:
            _log(f"e2 err layer={int(layer.layer_id)} {exc}")


def note_rows(layer_id: int, cosines: list) -> None:
    if _done:
        return
    _cos.append((int(layer_id), [float(item) for item in cosines]))


def note_mask(mask: torch.Tensor, parents, forward_batch=None) -> None:
    global _mask, _parents, _loc, _table
    if _mask is None and mask is not None:
        _mask = mask
        _parents = [int(item) for item in parents]
    if _loc is None and forward_batch is not None:
        loc = getattr(forward_batch, "out_cache_loc", None)
        if loc is not None and loc.numel() >= QUERIES:
            pool = int(forward_batch.req_pool_indices.reshape(-1)[0])
            table = forward_batch.req_to_token_pool.req_to_token
            _loc = loc[:QUERIES].detach().to(dtype=torch.long).cpu()
            _table = table[pool, PREFIX : PREFIX + QUERIES].detach().to(dtype=torch.long).cpu()


def _ancestors(parents: list, slot: int) -> set:
    verify_parent = [-1] + [0 if parent < 0 else parent + 1 for parent in parents]
    found = set()
    node = slot
    while node >= 0 and node not in found:
        found.add(node)
        if node >= len(verify_parent):
            break
        node = verify_parent[node]
    return found


def _flush_e(forward) -> None:
    loc = _loc
    want = _table
    if loc is None or want is None:
        _log(f"e absent rank={_rank()}")
        return
    same = bool(torch.equal(loc, want))
    unique = int(torch.unique(loc).numel()) == QUERIES
    if same and unique:
        _log(f"e pass unique={QUERIES} rank={_rank()}")
        return
    _log(
        f"e fail same={int(same)} unique={int(torch.unique(loc).numel())} "
        f"loc={loc.tolist()} table={want.tolist()} rank={_rank()}"
    )


def _flush_e2() -> None:
    if not _written:
        _log(f"e2 absent rank={_rank()}")
        return
    for layer_id, same in _written:
        if same is None:
            _log(f"e2 fail layer={layer_id} shape rank={_rank()}")
            return
        bad = (~same).nonzero(as_tuple=False)
        if bad.numel():
            _log(f"e2 fail layer={layer_id} row={int(bad.reshape(-1)[0])} rank={_rank()}")
            return
    _log(f"e2 pass layers={len(_written)} rank={_rank()}")


def _flush_i() -> None:
    if _mask is None or not _parents:
        _log(f"i absent rank={_rank()}")
        return
    got = _mask[0, 0].detach()
    if got.device.type != "cpu":
        got = got.cpu()
    width = int(got.shape[-1])
    rows = min(QUERIES, int(got.shape[0]))
    for row in range(rows):
        ancestors = _ancestors(_parents, row)
        for col in range(width):
            if col < PREFIX:
                expect = False
            elif col >= KV_LEN:
                expect = True
            else:
                expect = (col - PREFIX) not in ancestors
            if bool(got[row, col]) != expect:
                _log(
                    f"i fail row={row} col={col} got={int(bool(got[row, col]))} "
                    f"expect={int(expect)} rank={_rank()}"
                )
                return
    _log(f"i pass rows={rows} width={width} rank={_rank()}")


def _flush_k() -> None:
    if not _cos:
        _log(f"k absent rank={_rank()}")
        return
    for layer_id, cosines in _cos:
        for row, cos in enumerate(cosines[:5]):
            if cos < 0.9999:
                _log(f"k fail layer={layer_id} row={row} cos={cos:.6f} rank={_rank()}")
                return
    _log(f"k pass layers={len(_cos)} rows=0-4 rank={_rank()}")


def flush_tree(forward) -> None:
    global _done
    if _done or _rank() != 0 and not _written and not _cos:
        return
    try:
        if forward is not None and _rank() == 0:
            _flush_e(forward)
        _flush_e2()
        if _rank() == 0:
            _flush_i()
        _flush_k()
    except Exception as exc:
        _log(f"err {exc}")
    _done = True
