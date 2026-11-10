"""Shadow req_to_token rows for OPT Draft tree layers.

b>1 TARGET_VERIFY reads KV indices from ``req_to_token[req_pool_indices,
:seq_lens]`` and appends the new layer through ``out_cache_loc``. A shadow
row keeps the sink, a recent window, and the tree columns already written.
The batch then points at that row and a shorter ``seq_lens``. Positions stay
on the real prefix length.

The eager allocator and the CUDA-graph replay both rebuild those indices
from the row and ``seq_lens``. Replay copies the custom mask into the
attention buffer sized by ``max_num_tokens * max_context_len``. The length-1
captured mask buffer is installed only for IDLE, so a window does not write
a long mask into it. A window mask is shorter than the full-prefix mask the
same batch already replays.

Page size other than 1 leaves the feature off. The shadow index is borrowed
from ``ReqToTokenPool.free_slots`` and is not attached to a ``Req``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from sglang.srt.speculative.opt_tree_remote.opt_tree_config import (
    opt_draft_kv_sink,
    opt_draft_kv_window,
    opt_draft_kv_window_check,
    opt_draft_kv_window_min_len,
)

logger = logging.getLogger(__name__)

_CHECK_EVERY = 100


@dataclass(frozen=True)
class WindowLayout:
    """Index layout of one shadow row before this layer's new slots.

    ``seq_len`` is the KV length the verify kernel reads from the row.
    ``new_at`` is where this layer's slots are mirrored. ``position_prefix``
    is the real prefix, used only for RoPE.
    """

    mask_prefix: int
    seq_len: int
    position_prefix: int
    skipped: int
    copies: Tuple[Tuple[int, int, int], ...]
    new_at: int


def window_layout(
    prefix_len: int, cached_len: int, sink: int, window
) -> WindowLayout:
    """Map a real prefix onto sink + recent tokens + tree columns.

    ``window`` is ``"full"`` or a positive int. A window that covers the
    whole prefix copies it without a hole.
    """
    prefix = int(prefix_len)
    cached = int(cached_len)
    if cached < prefix:
        raise ValueError(f"cached length {cached} is shorter than prefix {prefix}")
    tree = cached - prefix
    if window == "full" or prefix <= int(sink) + int(window):
        copies = ((0, 0, cached),) if cached else ()
        return WindowLayout(prefix, cached, prefix, 0, copies, cached)
    sink_n = int(sink)
    recent = int(window)
    copies = (
        (0, 0, sink_n),
        (sink_n, prefix - recent, recent),
    )
    if tree:
        copies = copies + ((sink_n + recent, prefix, tree),)
    visible = sink_n + recent
    return WindowLayout(
        visible,
        visible + tree,
        prefix,
        prefix - visible,
        copies,
        visible + tree,
    )


def fill_shadow(table: torch.Tensor, real_idx: int, shadow_idx: int, layout: WindowLayout) -> None:
    """Copy the layout's slices from the real row into the shadow row."""
    real = table[int(real_idx)]
    shadow = table[int(shadow_idx)]
    for dst, src, count in layout.copies:
        if count:
            shadow[dst : dst + count].copy_(real[src : src + count])


def fill_shadow_delta(
    table: torch.Tensor,
    real_idx: int,
    shadow_idx: int,
    layout: WindowLayout,
    copied_through: int,
) -> None:
    """Copy only the real-row span past ``copied_through``.

    The sink and the recent window stay where the previous layer put them.
    A new tree column is appended. ``copied_through`` is a real-row index.
    """
    real = table[int(real_idx)]
    shadow = table[int(shadow_idx)]
    prefix = int(layout.position_prefix)
    through = int(copied_through)
    for dst, src, count in layout.copies:
        if count <= 0:
            continue
        src_end = src + count
        if src_end <= prefix:
            if through < prefix:
                shadow[dst : dst + count].copy_(real[src:src_end])
            continue
        start = max(src, through)
        if start >= src_end:
            continue
        offset = start - src
        span = src_end - start
        shadow[dst + offset : dst + offset + span].copy_(real[start:src_end])


def index_mismatch(
    real: torch.Tensor, shadow: torch.Tensor, layout: WindowLayout, slots: torch.Tensor
) -> Tuple[int, Optional[int]]:
    """Count shadow positions that differ from the kept real-row span.

    The reference is computed on the host: the layout's copies, then this
    layer's new slots at ``new_at``. The returned index is the first shadow
    column that differs, or None when the span matches.
    """
    mismatch = 0
    first: Optional[int] = None

    def add(dst: int, got: torch.Tensor, expect: torch.Tensor) -> None:
        nonlocal mismatch, first
        if got.shape != expect.shape:
            mismatch += max(int(got.numel()), int(expect.numel()))
            if first is None:
                first = int(dst)
            return
        diff = got != expect
        count = int(diff.sum().item()) if diff.numel() else 0
        mismatch += count
        if count and first is None:
            first = int(dst) + int(diff.nonzero()[0].item())

    for dst, src, count in layout.copies:
        if count <= 0:
            continue
        got = shadow[dst : dst + count].detach().to(device="cpu")
        expect = real[src : src + count].detach().to(device="cpu")
        add(dst, got, expect)
    new = slots.detach().reshape(-1).to(device="cpu", dtype=shadow.dtype)
    width = int(new.shape[0])
    if width:
        start = int(layout.new_at)
        got = shadow[start : start + width].detach().to(device="cpu")
        add(start, got, new)
    return mismatch, first


def write_new_slots(
    table: torch.Tensor, shadow_idx: int, layout: WindowLayout, slots: torch.Tensor
) -> None:
    """Mirror this layer's KV slots into the shadow row after the visible prefix."""
    count = int(slots.reshape(-1).shape[0])
    if count == 0:
        return
    start = int(layout.new_at)
    table[int(shadow_idx), start : start + count] = slots.reshape(-1).to(
        dtype=table.dtype
    )


@dataclass
class RequestWindow:
    pool_index: Optional[int]
    layout: Optional[WindowLayout]
    seq_len: int
    mask_prefix: int
    position_prefix: int
    skipped: int
    fallback: bool


@dataclass
class LayerWindow:
    requests: List[RequestWindow]

    @property
    def active(self) -> bool:
        return any(row.pool_index is not None for row in self.requests)

    def seq_lens(self) -> List[int]:
        return [row.seq_len for row in self.requests]

    def mask_prefixes(self) -> List[int]:
        return [row.mask_prefix for row in self.requests]

    def position_prefixes(self) -> List[int]:
        return [row.position_prefix for row in self.requests]

    def pool_indices(self, runners) -> List[int]:
        out = []
        for row, runner in zip(self.requests, runners):
            if row.pool_index is None:
                out.append(int(runner.req.req_pool_idx))
            else:
                out.append(int(row.pool_index))
        return out


def _count(scheduler, name: str, n: int) -> None:
    if n <= 0:
        return
    from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import _draft_count

    _draft_count(scheduler, name, n)


def log_request_accept(req) -> None:
    """One line per finished request, so two runs can be paired by prompt."""
    if req is None:
        return
    rounds = int(getattr(req, "opt_accept_rounds", 0) or 0)
    if rounds <= 0:
        return
    total = int(getattr(req, "opt_accept_sum", 0) or 0)
    ids = list(getattr(req, "origin_input_ids", None) or [])
    head = ",".join(str(int(token)) for token in ids[:8])
    logger.info(
        "[OPT accept] rid=%s rounds=%d accept_sum=%d mean=%.4f prompt_len=%d head=%s",
        getattr(req, "rid", None),
        rounds,
        total,
        total / rounds,
        len(ids),
        head,
    )


def release_kv_window(scheduler, req) -> None:
    """Return a borrowed shadow row. A missing row is a no-op."""
    if req is None:
        return
    idx = getattr(req, "opt_kv_window_idx", None)
    if idx is None:
        return
    req.opt_kv_window_idx = None
    req.opt_kv_window_prefix = None
    req.opt_kv_window_cached = None
    pool = getattr(scheduler, "req_to_token_pool", None)
    slots = getattr(pool, "free_slots", None)
    if isinstance(slots, list) and int(idx) not in slots:
        slots.append(int(idx))


def _borrow(scheduler, req) -> Optional[int]:
    held = getattr(req, "opt_kv_window_idx", None)
    pool = scheduler.req_to_token_pool
    slots = getattr(pool, "free_slots", None)
    if not isinstance(slots, list):
        return None
    if held is not None:
        if int(held) == int(req.req_pool_idx) or not (0 <= int(held) < pool.size):
            req.opt_kv_window_idx = None
        else:
            return int(held)
    if not slots:
        return None
    idx = int(slots.pop(0))
    if idx == int(req.req_pool_idx):
        slots.append(idx)
        return None
    req.opt_kv_window_idx = idx
    req.opt_kv_window_prefix = None
    req.opt_kv_window_cached = None
    return idx


def note_windowed_layer(scheduler) -> bool:
    """Count one windowed layer. True on every 100th when the check is on."""
    if not opt_draft_kv_window_check():
        return False
    done = int(getattr(scheduler, "_opt_kv_window_checked", 0) or 0) + 1
    scheduler._opt_kv_window_checked = done
    return done % _CHECK_EVERY == 0


def prepare_layer_window(scheduler, runners, page_size: int) -> Optional[LayerWindow]:
    """Fill shadow rows for this tree layer. ``None`` keeps the real rows.

    Page size other than 1, an empty window, or a prefix at or below the
    minimum leaves every request on its real row. A borrow failure falls
    that request back for this layer and counts ``opt_kv_window_fallbacks``.
    """
    spec = opt_draft_kv_window()
    if spec is None or len(runners) <= 1 or int(page_size) != 1:
        return None
    sink = opt_draft_kv_sink()
    minimum = opt_draft_kv_window_min_len()
    window = None if spec == "full" else int(spec)
    rows: List[RequestWindow] = []
    any_long = False
    for runner in runners:
        runner._opt_shadow_dest = None
        prefix = int(runner.prefix_len)
        cached = int(runner.cached_len)
        if prefix <= minimum:
            rows.append(
                RequestWindow(None, None, cached, prefix, prefix, 0, False)
            )
            continue
        any_long = True
        layout = window_layout(prefix, cached, sink, spec if window is None else window)
        shadow = _borrow(scheduler, runner.req)
        if shadow is None:
            rows.append(RequestWindow(None, None, cached, prefix, prefix, 0, True))
            continue
        req = runner.req
        copied = getattr(req, "opt_kv_window_cached", None)
        same_prefix = getattr(req, "opt_kv_window_prefix", None) == prefix
        table = scheduler.req_to_token_pool.req_to_token
        # A new round gets a full copy. Commit rewrites real-row indices, so
        # the previous tree columns are not a valid prefix for the next round.
        if same_prefix and copied is not None and 0 <= int(copied) <= cached:
            fill_shadow_delta(table, int(req.req_pool_idx), shadow, layout, int(copied))
        else:
            fill_shadow(table, int(req.req_pool_idx), shadow, layout)
        req.opt_kv_window_prefix = prefix
        req.opt_kv_window_cached = cached
        runner._opt_shadow_dest = (int(shadow), int(layout.new_at))
        rows.append(
            RequestWindow(
                shadow,
                layout,
                layout.seq_len,
                layout.mask_prefix,
                layout.position_prefix,
                layout.skipped,
                False,
            )
        )
    if not any_long:
        return None
    plan = LayerWindow(rows)
    _count(scheduler, "opt_kv_window_layers", sum(row.pool_index is not None for row in rows))
    _count(scheduler, "opt_kv_window_fallbacks", sum(row.fallback for row in rows))
    _count(scheduler, "opt_kv_window_tokens_skipped", sum(row.skipped for row in rows if row.pool_index is not None))
    return plan


def mirror_new_slots(scheduler, runners, plan: LayerWindow, locs: Sequence[torch.Tensor]) -> None:
    """Write this layer's real slots into each active shadow row."""
    table = scheduler.req_to_token_pool.req_to_token
    for row, runner, loc in zip(plan.requests, runners, locs):
        if row.pool_index is None or row.layout is None:
            continue
        if getattr(runner, "_opt_shadow_written", False):
            runner._opt_shadow_written = False
            continue
        write_new_slots(table, row.pool_index, row.layout, loc)
        count = int(loc.reshape(-1).shape[0])
        runner.req.opt_kv_window_cached = int(runner.cached_len) + count


def layer_is_truncated(plan: LayerWindow) -> bool:
    """True when an active shadow row drops a hole out of the prefix."""
    return any(
        row.pool_index is not None and row.skipped > 0 for row in plan.requests
    )


def maybe_check_indices(scheduler, runners, plan: LayerWindow, locs: Sequence[torch.Tensor]) -> None:
    """Compare kept shadow columns with the real row. Sampling stays on the shadow.

    Used when the window truncates. A second full-prefix forward would not
    match, and it sits inside the layer timer.
    """
    table = scheduler.req_to_token_pool.req_to_token
    checked = 0
    mismatch = 0
    first = None
    prefix = None
    for row, runner, loc in zip(plan.requests, runners, locs):
        if row.pool_index is None or row.layout is None or row.skipped <= 0:
            continue
        count, pos = index_mismatch(
            table[int(runner.req.req_pool_idx)],
            table[int(row.pool_index)],
            row.layout,
            loc,
        )
        checked += 1
        mismatch += count
        if count and first is None:
            first = pos
            prefix = int(row.position_prefix)
    done = int(getattr(scheduler, "_opt_kv_window_checked", 0) or 0)
    logger.info(
        "[OPT kv window index] layer=%d checked=%d mismatch=%d first_shadow=%s prefix=%s",
        done,
        checked,
        mismatch,
        first,
        prefix,
    )
    _count(scheduler, "opt_kv_window_index_checks", checked)
    _count(scheduler, "opt_kv_window_index_mismatch", mismatch)


def maybe_check_logits(scheduler, shadow_logits: torch.Tensor, real_logits: torch.Tensor, width: int) -> None:
    """Compare one checked layer. Sampling keeps ``shadow_logits``."""
    done = int(getattr(scheduler, "_opt_kv_window_checked", 0) or 0)
    left = shadow_logits.detach().float().reshape(-1, shadow_logits.shape[-1])
    right = real_logits.detach().float().reshape(-1, real_logits.shape[-1])
    rows = min(int(left.shape[0]), int(right.shape[0]))
    left = left[:rows]
    right = right[:rows]
    max_abs = float((left - right).abs().max().item()) if rows else 0.0
    k = min(int(width), int(left.shape[-1]))
    if rows and k > 0:
        left_ids = torch.topk(left, k, dim=-1).indices.sort(dim=-1).values
        right_ids = torch.topk(right, k, dim=-1).indices.sort(dim=-1).values
        match = int((left_ids == right_ids).all(dim=-1).sum().item())
    else:
        match = 0
    worst = max(float(getattr(scheduler, "_opt_kv_window_max_abs", 0.0) or 0.0), max_abs)
    scheduler._opt_kv_window_max_abs = worst
    matched = int(getattr(scheduler, "_opt_kv_window_topn_match", 0) or 0) + match
    total = int(getattr(scheduler, "_opt_kv_window_topn_rows", 0) or 0) + rows
    scheduler._opt_kv_window_topn_match = matched
    scheduler._opt_kv_window_topn_rows = total
    agree = (matched / total) if total else 1.0
    logger.info(
        "[OPT kv window check] layer=%d max_abs=%.6g worst=%.6g topn=%.6f rows=%d",
        done,
        max_abs,
        worst,
        agree,
        rows,
    )
    _count(scheduler, "opt_kv_window_check_rows", rows)
    _count(scheduler, "opt_kv_window_check_match", match)
    micro = int(round(float(scheduler._opt_kv_window_max_abs) * 1_000_000.0))
    # Running worst, emitted as a delta so a window sum stays the worst.
    previous = int(getattr(scheduler, "_opt_kv_window_max_abs_micro", 0) or 0)
    if micro > previous:
        _count(scheduler, "opt_kv_window_check_max_abs_micro", micro - previous)
        scheduler._opt_kv_window_max_abs_micro = micro


def check_enabled() -> bool:
    return opt_draft_kv_window_check()
