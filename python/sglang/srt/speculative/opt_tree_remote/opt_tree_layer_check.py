"""One-round fp32 GQA against tree FIA. Does not change the returned attention."""

from __future__ import annotations

import logging
import os

import torch

KV_LEN = 4172
QUERIES = 9
SLOT = 4
PAGE_END = 4224
SPEC_DIR = "/mnt/user/liujg/t0-run"

_active = False
_failed = None
_layers = 0
_saved: dict = {}


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _log(line: str) -> None:
    logging.warning("FORKCHECK %s", line)


def _gather(buf: torch.Tensor, slots: torch.Tensor, page: int) -> torch.Tensor:
    if buf.ndim == 4 and int(buf.shape[1]) == page and page != 1:
        return buf[slots // page, slots % page]
    if buf.ndim == 4 and int(buf.shape[1]) == 1:
        return buf[slots, 0]
    return buf[slots]


def _ancestors(parents: list[int], slot: int) -> set[int]:
    verify_parent = [-1] + [0 if parent < 0 else parent + 1 for parent in parents]
    found = set()
    node = slot
    while node >= 0 and node not in found:
        found.add(node)
        if node >= len(verify_parent):
            break
        node = verify_parent[node]
    return found


def _attend(parents: list[int], prefix: int, queries: int, device) -> torch.Tensor:
    width = prefix + queries
    mask = torch.zeros((queries, width), dtype=torch.bool, device=device)
    if prefix > 0:
        mask[:, :prefix] = True
    for slot in range(queries):
        for node in _ancestors(parents, slot):
            col = prefix + node
            if 0 <= col < width:
                mask[slot, col] = True
    return mask


def _gqa(q, k, v, attend, scale: float) -> torch.Tensor:
    heads = int(q.shape[1])
    kv_heads = int(k.shape[1])
    group = heads // kv_heads
    qf = q.float()
    kf = k.float()
    vf = v.float()
    out = torch.empty(q.shape[0], heads, vf.shape[-1], device=q.device, dtype=torch.float32)
    floor = torch.finfo(torch.float32).min
    for kv_head in range(kv_heads):
        start = kv_head * group
        qs = qf[:, start : start + group, :]
        score = torch.einsum("qgd,sd->qgs", qs, kf[:, kv_head, :]) * scale
        score = score.masked_fill(~attend[:, None, :], floor)
        prob = torch.softmax(score, dim=-1)
        out[:, start : start + group, :] = torch.einsum("qgs,sd->qgd", prob, vf[:, kv_head, :])
    return out


def _metrics(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    left = got.reshape(-1).float()
    right = ref.reshape(-1).float()
    gap = float((left - right).abs().max().item())
    denom = max(float(left.abs().max().item()), 1e-6)
    rel = gap / denom
    norm = float(left.norm().item() * right.norm().item())
    cos = float(torch.dot(left, right).item() / norm) if norm > 0 else 0.0
    return cos, rel


def _row_passes(cos: float, rel: float) -> bool:
    return cos >= 0.999 and rel <= 0.05


def _stash_values(layer_id: int, values: torch.Tensor) -> None:
    cpu = values.detach().to("cpu")
    if layer_id == 10:
        _saved["10"] = cpu
    if layer_id == 20:
        _saved["20"] = cpu
    if layer_id >= 20:
        _saved["last"] = cpu
        _saved["last_id"] = int(layer_id)


def _fia(q, k_cache, v_cache, layer, block_table, mask, kv_lens, q_lens, page: int):
    bs = int(q.shape[0])
    queries = int(q.shape[1])
    output, _ = torch.ops.npu.npu_fused_infer_attention_score(
        q,
        k_cache.view(-1, page, layer.tp_k_head_num * layer.qk_head_dim),
        v_cache.view(-1, page, layer.tp_v_head_num * layer.v_head_dim),
        input_layout="BSND",
        num_heads=layer.tp_q_head_num,
        num_key_value_heads=layer.tp_k_head_num,
        scale=layer.scaling,
        block_table=block_table,
        block_size=page,
        atten_mask=mask,
        actual_seq_lengths=q_lens,
        actual_seq_lengths_kv=kv_lens,
        sparse_mode=0,
    )
    return output.reshape(bs, queries, layer.tp_q_head_num, layer.v_head_dim)


def _probes(q_rows, k_cache, v_cache, layer, md, page, ref) -> None:
    bs = int(q_rows.shape[0])
    width = int(md.blocked_mask.shape[-1])
    base_kv = list(md.kv_lens_cpu)
    base_q = list(md.q_lens_cpu)

    page_mask = md.blocked_mask.clone()
    if PAGE_END < width:
        page_mask[:, :, :, PAGE_END:] = True
    short = md.blocked_mask.clone()
    if KV_LEN < width:
        short[:, :, :, KV_LEN:] = True
    short_kv = list(base_kv)
    short_kv[0] = KV_LEN
    wide_kv = list(base_kv)
    wide_kv[0] = PAGE_END

    variants = (
        ("a", q_rows, page_mask, base_kv, base_q),
        ("a_short", q_rows, short, short_kv, base_q),
        ("b", q_rows, md.blocked_mask.clone(), wide_kv, base_q),
    )
    for name, query, mask, kv_lens, q_lens in variants:
        try:
            out = _fia(query, k_cache, v_cache, layer, md.block_tables, mask, kv_lens, q_lens, page)
            cos, rel = _metrics(out[0, SLOT], ref[SLOT])
            _log(
                f"k probe {name} layer={int(layer.layer_id)} "
                f"cos={cos:.6f} rel={rel:.6g} pass={int(_row_passes(cos, rel))}"
            )
        except Exception as exc:
            _log(f"k probe {name} err {exc}")
    try:
        mask_row = md.blocked_mask[:, :, SLOT : SLOT + 1, :].clone()
        one = _fia(
            q_rows[:, SLOT : SLOT + 1].contiguous(),
            k_cache,
            v_cache,
            layer,
            md.block_tables,
            mask_row,
            base_kv,
            [1] * bs,
            page,
        )
        cos, rel = _metrics(one[0, 0], ref[SLOT])
        _log(
            f"k probe c layer={int(layer.layer_id)} "
            f"cos={cos:.6f} rel={rel:.6g} pass={int(_row_passes(cos, rel))}"
        )
    except Exception as exc:
        _log(f"k probe c err {exc}")


def maybe_layer_check(forward_batch, q, k_cache, v_cache, layer, output, page_size: int, md) -> None:
    """Compare this layer's FIA rows with fp32 GQA. The caller keeps `output`."""
    global _active, _failed, _layers
    if md is None or not getattr(md, "kv_lens_cpu", None):
        return
    kv = int(md.kv_lens_cpu[0])
    queries = int(md.blocked_mask.shape[2])
    if kv != KV_LEN or queries != QUERIES:
        return
    parents_rows = getattr(forward_batch.spec_info, "_fork_parents", None)
    if not parents_rows:
        _log(f"k skip layer={int(layer.layer_id)} parents missing")
        return
    _active = True
    prefix = kv - queries
    bs = int(md.block_tables.shape[0])
    q_rows = q.reshape(bs, queries, layer.tp_q_head_num, layer.qk_head_dim)
    fia = output.reshape(bs, queries, layer.tp_q_head_num, layer.v_head_dim)
    table = forward_batch.req_to_token_pool.req_to_token
    pool = int(forward_batch.req_pool_indices[0].item())
    page = int(page_size)
    device = q.device
    slots = table[pool, :KV_LEN].to(device=device, dtype=torch.long)
    keys = _gather(k_cache, slots, page)
    values = _gather(v_cache, slots, page)
    attend = _attend([int(item) for item in parents_rows[0]], prefix, queries, device)
    ref = _gqa(q_rows[0], keys, values, attend, float(layer.scaling))
    worst_row = 0
    worst_rel = -1.0
    slot_cos, slot_rel = 0.0, 0.0
    early = []
    for row in range(queries):
        cos, rel = _metrics(fia[0, row], ref[row])
        if row <= 4:
            early.append(cos)
        if row == SLOT:
            slot_cos, slot_rel = cos, rel
        if rel > worst_rel:
            worst_rel = rel
            worst_row = row
    if os.environ.get("SGLANG_FORK_TREE"):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_slot_check import (
            note_mask,
            note_rows,
        )

        note_rows(int(layer.layer_id), early)
        note_mask(md.blocked_mask, parents_rows[0], forward_batch)
    _layers += 1
    passed = _row_passes(slot_cos, slot_rel)
    _log(
        f"k layer={int(layer.layer_id)} slot4 cos={slot_cos:.6f} rel={slot_rel:.6g} "
        f"worst_row={worst_row} worst_rel={worst_rel:.6g} pass={int(passed)} rank={_rank()}"
    )
    if not passed and _failed is None:
        _failed = int(layer.layer_id)
        _probes(q_rows, k_cache, v_cache, layer, md, page, ref)
    tail = table[pool, 4096:4164].to(device=device, dtype=torch.long)
    _stash_values(int(layer.layer_id), _gather(v_cache, tail, page))


def write_spec_values() -> str:
    rank = _rank()
    if not _active:
        return "k absent"
    path = os.path.join(SPEC_DIR, f"fork-v-spec-r{rank}.pt")
    if _failed is None and _saved:
        torch.save(_saved, path)
    failed = "none" if _failed is None else str(_failed)
    return f"k layers={_layers} failed={failed} saved={int(_failed is None and bool(_saved))} rank={rank}"


def maybe_dump_prefill(forward_batch, layer, page_size: int) -> None:
    """Save the same positions from a Target-only prefill."""
    if not os.environ.get("SGLANG_FORK_PREFILL"):
        return
    cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if cpu is None or int(cpu[0]) < 4164:
        return
    layer_id = int(layer.layer_id)
    if layer_id not in (10, 20) and layer_id < 20:
        return
    table = forward_batch.req_to_token_pool.req_to_token
    pool = int(forward_batch.req_pool_indices[0].item())
    value = forward_batch.token_to_kv_pool.get_value_buffer(layer_id)
    slots = table[pool, 4096:4164].to(device=value.device, dtype=torch.long)
    _stash_values(layer_id, _gather(value, slots, int(page_size)))
    if layer_id < 20 and layer_id != 10:
        return
    path = os.path.join(SPEC_DIR, f"fork-v-prefill-r{_rank()}.pt")
    torch.save(_saved, path)
