"""SR-only RPD host plans. The shared RPD device-output API stays unchanged.

Topology comes from the normalized verify packet, never from device readback.
Statistics keep their original dtypes; both D2H copies share one completion
event. CPU selection deliberately uses the existing RPD reference arithmetic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.speculative.rpd_verify import (
    _rpd_compact_edges,
    _rpd_compact_select,
    rpd_gap_max,
)
from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
    SRTransferUnresolved,
    alloc_host,
    event_supported,
    pin_supported,
    wait_event,
)

logger = logging.getLogger(__name__)


def rpd_batch_key(reqs):
    return tuple(
        (id(req), len(req.output_ids), getattr(req, "sr_step_id", None)) for req in reqs
    )


@dataclass
class SRRPDHostInput:
    tree: torch.Tensor  # owned CPU int64 [4, B, W]
    edges: list
    edge_index_cpu: torch.Tensor
    vocab: int
    owner: object
    generation: int
    batch_key: tuple
    edge_index: Optional[torch.Tensor] = None
    consumed: bool = False

    def current(self) -> bool:
        return (
            not self.consumed
            and self.owner.generation == self.generation
            and self.owner.rpd_input is self
            and not self.owner.unresolved
        )


@dataclass
class SRRPDHostPlan:
    rows: list[list[int]]
    tokens: list[list[int]]
    pre_lengths: list[int]  # draft count, excludes bonus
    vocab: int
    batch_key: tuple
    consumed: bool = False


def build_sr_rpd_input(
    verified,
    tokens,
    parents,
    selected,
    *,
    topk,
    steps,
    width,
    vocab,
    owner,
    generation,
    batch_key=(),
):
    """Build only topology, matching build_tree_efficient's reverse insertion.

    Unsupported/malformed layouts decline the optimization before upload. Missing
    selected parents stay disconnected, exactly as in the existing tree builder.
    """
    bs = int(verified.numel())
    # Parent width follows the normalized packet, including the chain template
    # (``spec_steps`` columns) and zero-filled empty rows. Only an index that
    # addresses past that row declines the batch.
    if (
        topk <= 1
        or width < 1
        or vocab < 1
        or int(steps) < 1
        or tokens.shape != (bs, width - 1)
        or selected.shape != (bs, width - 1)
        or parents.dim() != 2
        or parents.shape[0] != bs
        or int(parents.shape[1]) < 1
        or any(t.device.type != "cpu" for t in (verified, tokens, parents, selected))
    ):
        return None
    cand = torch.cat((verified.reshape(bs, 1), tokens), dim=1)
    retr = torch.arange(bs * width, dtype=torch.int64).reshape(bs, width)
    nxt = torch.full((bs, width), -1, dtype=torch.int64)
    sibling = torch.full_like(nxt, -1)
    for b, (pp, ss) in enumerate(zip(parents.tolist(), selected.tolist())):
        parent_of = [-1] * width
        for i in range(width - 1, 0, -1):
            if ss[i - 1] < 0:
                return None
            parent_tb = ss[i - 1] // topk
            pos = 0
            if parent_tb > 0:
                if parent_tb >= len(pp):
                    return None
                try:
                    pos = ss.index(pp[parent_tb]) + 1
                except ValueError:
                    continue
            parent_of[i] = pos
            sibling[b, i] = nxt[b, pos]
            nxt[b, pos] = i
        # Never let a malformed input turn the CPU DFS into an infinite loop.
        for i in range(width):
            seen = set()
            u = i
            while u >= 0:
                if u in seen:
                    return None
                seen.add(u)
                u = parent_of[u]
    tree = torch.stack((cand, retr, nxt, sibling))
    edges, index = _rpd_compact_edges(tree, bs * width, vocab)
    if index is None:
        index = torch.empty((2, 0), dtype=torch.int64)
    return SRRPDHostInput(tree, edges, index, int(vocab), owner, generation, batch_key)


class SRRPDWorkspace:
    """One synchronous Target's reusable statistics and source lifetime."""

    def __init__(self):
        self.key = None
        self.rows_cap = self.edges_cap = 0
        self.event = None
        self.unresolved = False
        self.holds = None
        self.logged = False
        self.metrics = None

    def count(self, name, value=1):
        if self.metrics is not None:
            self.metrics.counts[name] += value

    def prepare(self, logits, context, metrics=None):
        """Metadata/capability admission only; no statistic or request writes."""
        if self.unresolved:
            raise SRTransferUnresolved("RPD statistics workspace is unresolved")
        self.metrics = metrics
        if not isinstance(context, SRRPDHostInput) or not context.current():
            return "context"
        bs, width = context.tree.shape[1:]
        rows = bs * width
        if (
            logits.dim() != 2
            or tuple(logits.shape) != (rows, context.vocab)
            or not logits.is_contiguous()
            or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            return "logits_layout"
        index = context.edge_index
        if (
            index is None
            or index.device != logits.device
            or index.dtype != torch.int64
            or tuple(index.shape) != (2, len(context.edges))
        ):
            return "edge_layout"
        device = logits.device
        # CPU supports deterministic contract tests, but production admission is
        # restricted to NPU by SRFixedAcceptState.static_disable_reason.
        if device.type not in ("cpu", "npu"):
            return "device"
        if device.type != "cpu" and (
            not pin_supported(device) or not event_supported(device)
        ):
            return "staging"
        key = (device, logits.dtype)
        edges = len(context.edges)
        if key != self.key or rows > self.rows_cap or edges > self.edges_cap:
            rc = max(rows, self.rows_cap, 1)
            ec = max(edges, self.edges_cap, 1)
            rc, ec = 1 << (rc - 1).bit_length(), 1 << (ec - 1).bit_length()
            star_host, star_pinned = alloc_host((rc,), torch.int64, device)
            stats_host, stats_pinned = alloc_host((2, ec), logits.dtype, device)
            if device.type != "cpu" and not (star_pinned and stats_pinned):
                return "staging"
            self.values = torch.empty(rc, dtype=logits.dtype, device=device)
            self.argmax = torch.empty(rc, dtype=torch.int64, device=device)
            self.edge_values = torch.empty((2, ec), dtype=logits.dtype, device=device)
            self.star_host, self.stats_host = star_host, stats_host
            self.rows_cap, self.edges_cap, self.key = rc, ec, key
            self.event = (
                torch.get_device_module(device.type).Event()
                if device.type != "cpu"
                else None
            )
            self.count("rpd_host_workspace_grow")
        return None

    def statistics(self, logits, context):
        bs, width = context.tree.shape[1:]
        rows, edges = bs * width, len(context.edges)
        if rows == 0:
            return self.star_host[:0].reshape(bs, width), None
        values, argmax = self.values[:rows], self.argmax[:rows]
        # Edge buffers are flattened contiguous live regions, independent of
        # capacity; this also avoids strided D2H copies when the batch shrinks.
        stats = self.edge_values.reshape(-1)[: 2 * edges].reshape(2, edges)
        host_stats = self.stats_host.reshape(-1)[: 2 * edges].reshape(2, edges)
        host_star = self.star_host[:rows]
        # Reduction and on-device gathers have not queued a host readback.
        # A launch failure here must leave the packet reusable for the next round.
        torch.max(logits, dim=-1, out=(values, argmax))
        if edges:
            parent, token = context.edge_index
            stats[0].copy_(values.index_select(0, parent))
            stats[1].copy_(logits[parent, token])
        async_copy = logits.device.type != "cpu"
        if not async_copy:
            host_star.copy_(argmax, non_blocking=False)
            if edges:
                host_stats.copy_(stats, non_blocking=False)
            return host_star.reshape(bs, width), host_stats if edges else None
        self.holds = (logits, context, values, argmax, stats)
        try:
            host_star.copy_(argmax, non_blocking=True)
            if edges:
                host_stats.copy_(stats, non_blocking=True)
            self.count("rpd_host_stats_d2h_count", 1 + int(edges > 0))
            self.count(
                "rpd_host_stats_d2h_bytes",
                rows * 8 + 2 * edges * logits.element_size(),
            )
            self.event.record()
            self.count("rpd_host_stats_waits")
            wait_event(self.event)
        except BaseException:
            # The D2H or its event may still be running. Keep sources alive.
            self.unresolved = True
            context.owner.unresolved = True
            raise
        self.holds = None
        return host_star.reshape(bs, width), host_stats if edges else None


def verify_sr_rpd_host(logits, context, workspace, tau, path_cap):
    """Consume a prepared context once, returning no intermediate device output."""
    gap = rpd_gap_max(tau)
    if not context.current():
        raise RuntimeError("stale or consumed SR RPD input")
    context.consumed = True
    bs, width = context.tree.shape[1:]
    if bs == 0:
        return SRRPDHostPlan([], [], [], context.vocab, context.batch_key)
    star, stats = workspace.statistics(logits, context)
    accept, lengths, positions, tokens = _rpd_compact_select(
        context.tree,
        context.edges,
        star,
        stats,
        gap,
        float(tau) == 0.0,
        path_cap,
        bs * width,
        bs * width + 1,
    )
    folded = dict(zip(positions.tolist(), tokens.tolist()))
    rows = accept.tolist()
    token_rows = [[folded[i] if i >= 0 else 0 for i in row] for row in rows]
    workspace.count("rpd_host_plan_hit")
    if not workspace.logged:
        logger.info("Speculative RPD verify path: npu_sr_host_plan")
        workspace.logged = True
    return SRRPDHostPlan(
        rows, token_rows, lengths.tolist(), context.vocab, context.batch_key
    )
