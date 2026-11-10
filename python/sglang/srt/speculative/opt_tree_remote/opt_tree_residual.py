"""Save one residual stream and a KV window. Does not change the forward result."""

from __future__ import annotations

import logging
import os

import torch

SPEC_DIR = "/mnt/user/liujg/t0-run"
POS = 4167
WINDOW = (4096, 4164)

_row = None
_hidden: list = []
_pending = None
_forward = None
_pos = None
_seen = 0
_shapes = set()
_done = False
_err = False
_ctrl_pos = None
_ctrl = {}


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _positions(forward_batch, n_tokens: int) -> torch.Tensor | None:
    pos = getattr(forward_batch, "positions", None)
    if pos is None or pos.numel() == 0:
        return None
    if pos.ndim == 2 and int(pos.shape[0]) == 3:
        pos = pos[0]
    elif pos.ndim > 1:
        pos = pos.reshape(-1)
    return pos[:n_tokens]


def _watch_row(forward_batch, hidden: torch.Tensor):
    mode = os.environ.get("SGLANG_FORK_RESID", "")
    pos = _positions(forward_batch, hidden.shape[0])
    if pos is None:
        return None
    if mode == "spec":
        if pos.numel() < 9 or int(pos[0]) != 4163 or int(pos[8]) != 4171:
            return None
        hits = (pos[: hidden.shape[0]] == POS).nonzero(as_tuple=False)
        if hits.numel() == 0:
            return None
        return int(hits[0])
    if mode == "prefill":
        if hidden.shape[0] <= POS or pos.numel() <= POS or int(pos[POS]) != POS:
            return None
        return POS
    if mode == "decode":
        if int(pos.reshape(-1)[0]) != POS:
            return None
        seq = getattr(forward_batch, "seq_lens", None)
        if seq is not None and int(seq.reshape(-1)[0]) not in (POS, POS + 1):
            return None
        return 0
    return None


def _gather(buf: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    page = 128
    if buf.ndim == 4 and int(buf.shape[1]) == page:
        return buf[slots // page, slots % page]
    if buf.ndim == 4 and int(buf.shape[1]) == 1:
        return buf[slots, 0]
    return buf[slots]


def _window(forward_batch, layer_idx: int):
    pool = int(forward_batch.req_pool_indices[0])
    table = forward_batch.req_to_token_pool.req_to_token
    slots = table[pool, WINDOW[0] : WINDOW[1]].to(torch.int64)
    if slots.device.type == "cpu" and int(slots.min()) < 0:
        raise RuntimeError(f"slot min={int(slots.min())}")
    k_buf, v_buf = forward_batch.token_to_kv_pool.get_kv_buffer(layer_idx)
    slots = slots.to(k_buf.device)
    width = WINDOW[1] - WINDOW[0]
    return (
        _gather(k_buf, slots).detach().reshape(width, -1),
        _gather(v_buf, slots).detach().reshape(width, -1),
    )


def flush_pending(spec_info=None) -> None:
    _flush(spec_info)


def _flush(spec_info) -> None:
    global _pending, _done, _hidden, _row, _forward, _pos, _seen
    if _done or spec_info is None:
        return
    pos = getattr(spec_info, "positions", None)
    if not _hidden or _forward is None or pos is None:
        _hidden = []
        return
    flat = pos.reshape(-1)
    if flat.numel() < 9 or int(flat[0]) != 4163 or int(flat[8]) != 4171:
        _hidden = []
        _row = None
        _pending = None
        return
    hits = (flat[: _hidden[0].shape[0]] == POS).nonzero(as_tuple=False)
    if hits.numel() == 0:
        _hidden = []
        return
    row = int(hits.reshape(-1)[0])
    if os.environ.get("SGLANG_FORK_TREE"):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_slot_check import flush_tree

        flush_tree(_forward)
        _save_tree("spec", [item[:5].detach() for item in _hidden], _forward)
    mode, hiddens, norm, forward = "spec", list(_hidden), _pending, _forward
    _hidden = []
    _pending = None
    _forward = None
    _pos = None
    if norm is None:
        return
    hiddens = [item[row] for item in hiddens]
    norm = norm[row]
    payload = {
        "h": torch.stack(hiddens).float().cpu(),
        "norm": norm.float().cpu(),
        "row": row,
    }
    path = f"{SPEC_DIR}/fork-resid-{mode}-r{_rank()}.pt"
    torch.save(payload, path)
    _done = True
    kv = 0
    try:
        keys, values = [], []
        for layer_idx in range(len(hiddens)):
            key, value = _window(forward, layer_idx)
            keys.append(key.float().cpu())
            values.append(value.float().cpu())
        payload["k"] = torch.stack(keys)
        payload["v"] = torch.stack(values)
        torch.save(payload, path)
        kv = 1
    except Exception as exc:
        if _rank() == 0:
            logging.warning("FORKRESID_KV %s", exc)
    if _rank() == 0:
        logging.warning(
            "FORKRESID saved mode=%s layers=%s row=%s kv=%s ptrs=%s",
            mode,
            len(hiddens),
            row,
            kv,
            len({int(item.data_ptr()) for item in hiddens}),
        )


def _tree_kv(forward, layers: range):
    pool = int(forward.req_pool_indices.reshape(-1)[0])
    table = forward.req_to_token_pool.req_to_token
    slots = table[pool, 4164:4168].to(torch.int64)
    keys, values = [], []
    for layer_idx in layers:
        k_buf, v_buf = forward.token_to_kv_pool.get_kv_buffer(layer_idx)
        slots_dev = slots.to(k_buf.device)
        keys.append(_gather(k_buf, slots_dev).reshape(4, -1).float().cpu())
        values.append(_gather(v_buf, slots_dev).reshape(4, -1).float().cpu())
    return torch.stack(keys), torch.stack(values)


def _save_tree(mode: str, rows: list, forward) -> None:
    if not rows:
        return
    payload = {"h": torch.stack([item.float().cpu() for item in rows])}
    try:
        payload["k"], payload["v"] = _tree_kv(forward, range(30, 44))
    except Exception as exc:
        if _rank() == 0:
            logging.warning("FORKTREE_KV %s", exc)
    path = f"{SPEC_DIR}/fork-tree-{mode}-r{_rank()}.pt"
    torch.save(payload, path)
    if _rank() == 0:
        logging.warning(
            "FORKTREE saved mode=%s layers=%s kv=%s",
            mode,
            len(rows),
            int("k" in payload),
        )


def _control_layer(layer_idx: int, hidden: torch.Tensor, residual, forward_batch) -> None:
    global _ctrl_pos, _forward, _err
    mode = os.environ.get("SGLANG_FORK_RESID", "")
    stream = hidden if residual is None else hidden + residual
    if mode == "prefill":
        if stream.shape[0] <= 4167:
            return
        if layer_idx == 0:
            _hidden.clear()
            _forward = forward_batch
        _hidden.append(stream[4163:4168].detach())
        return
    if stream.shape[0] != 1:
        return
    if layer_idx == 0:
        pos = int(forward_batch.positions.reshape(-1)[0])
        _ctrl_pos = pos if 4163 <= pos <= 4167 else None
        _hidden.clear()
        _forward = forward_batch
    if _ctrl_pos is None:
        return
    _hidden.append(stream[:1].detach())


def note_layer(layer_idx: int, hidden: torch.Tensor, residual, forward_batch) -> None:
    global _row, _hidden, _forward, _pos, _err
    if os.environ.get("SGLANG_FORK_TREE") and os.environ.get("SGLANG_FORK_RESID") in (
        "prefill",
        "decode",
    ):
        try:
            _control_layer(layer_idx, hidden, residual, forward_batch)
        except Exception as exc:
            _err = True
            if _rank() == 0:
                logging.warning("FORKTREE_ERR layer=%s %s", layer_idx, exc)
        return
    if _done or _err or os.environ.get("SGLANG_FORK_RESID") != "spec":
        return
    if hidden is None or hidden.ndim != 2 or hidden.shape[0] != 9:
        return
    try:
        if layer_idx == 0:
            _hidden = []
            _forward = forward_batch
        stream = hidden if residual is None else hidden + residual
        _hidden.append(stream.detach())
    except Exception as exc:
        _err = True
        if _rank() == 0:
            logging.warning("FORKRESID_ERR layer=%s %s", layer_idx, exc)


def note_norm(hidden: torch.Tensor, forward_batch) -> None:
    global _pending, _ctrl_pos
    if os.environ.get("SGLANG_FORK_TREE") and os.environ.get("SGLANG_FORK_RESID") in (
        "prefill",
        "decode",
    ):
        mode = os.environ.get("SGLANG_FORK_RESID", "")
        if mode == "prefill" and len(_hidden) >= 64:
            _save_tree("prefill", list(_hidden), forward_batch)
            _hidden.clear()
            return
        if mode == "decode" and _ctrl_pos is not None and _hidden:
            _ctrl[_ctrl_pos] = [item.float().cpu() for item in _hidden]
            _hidden.clear()
            if all(pos in _ctrl for pos in range(4163, 4168)):
                rows = []
                for layer_idx in range(len(_ctrl[4163])):
                    rows.append(torch.cat([_ctrl[pos][layer_idx] for pos in range(4163, 4168)], 0))
                _save_tree("decode", rows, forward_batch)
            _ctrl_pos = None
            return
        return
    if _done or _err or not _hidden:
        return
    if hidden is None or hidden.ndim != 2 or not _hidden:
        return
    if hidden.shape[0] != _hidden[-1].shape[0]:
        return
    _pending = hidden.detach()
