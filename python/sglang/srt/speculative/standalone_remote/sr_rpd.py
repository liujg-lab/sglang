"""SR-only RPD host plans. The shared RPD device-output API stays unchanged.

Topology comes from the normalized verify packet, never from device readback.
Statistics keep their original dtypes; both D2H copies share one completion
event. NPU edge logits are gathered into that reused buffer without a device
gap. CPU selection deliberately uses the existing RPD reference arithmetic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.speculative.rpd_verify import (
    _RPDHostTopology,
    _rpd_select_host,
    rpd_gap_max,
)
from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
    SRTransferUnresolved,
    alloc_host,
    device_cache_key,
    event_supported,
    pin_supported,
    wait_event,
)

logger = logging.getLogger(__name__)


def _use_npu_edge_kernel(logits) -> bool:
    """True only for a real NPU tensor.

    Contract tests spoof ``Tensor.device`` with a plain object whose ``type``
    is ``"npu"``. Those tensors stay on the reference gather.
    """
    device = logits.device
    return isinstance(device, torch.device) and device.type == "npu"


def _edge_gather_impl(logits):
    """Resolve backend capability before submitting the reduction."""
    if _use_npu_edge_kernel(logits):
        from sglang.srt.speculative.standalone_remote.sr_rpd_kernels_npu import (
            gather_edge_stats,
        )

        return gather_edge_stats
    return _gather_edge_stats


def _gather_edge_stats(values, logits, edge_index, stats) -> bool:
    """CPU reference gather used by contract tests."""
    parent, token = edge_index
    stats[0].copy_(values.index_select(0, parent))
    stats[1].copy_(logits[parent, token])
    return False


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
    topology: object
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
    parent_slots,
):
    """Build topology from parent slots already chosen by ``plan_draft_rows``.

    Unsupported layouts decline the optimization before upload. ``parent_slots``
    is one row per request, length ``width``. Slot 0 and every unlinked draft
    slot are ``-1``. Insertion still walks slots from high to low so sibling
    order matches ``build_tree_efficient``.
    """
    bs = int(verified.numel())
    # Parent width is the verify-kernel stride ``topk * (steps - 1) + 1``
    # (1 when ``steps == 1``), including chain-template prefixes and -1 fill.
    # It is not the longest raw row. An index that addresses past that stride
    # declines the batch.
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
        or len(parent_slots) != bs
        or any(len(row) != width for row in parent_slots)
    ):
        return None
    cand = torch.cat((verified.reshape(bs, 1), tokens), dim=1)
    retr = torch.arange(bs * width, dtype=torch.int64).reshape(bs, width)
    nxt = torch.full((bs, width), -1, dtype=torch.int64)
    sibling = torch.full_like(nxt, -1)
    cand_rows = cand.tolist()
    edges: list = []
    parent_rows: list = []
    child_tokens: list = []
    batch_children: list = []
    for b, slots in enumerate(parent_slots):
        children = [[] for _ in range(width)]
        for i in range(width - 1, 0, -1):
            pos = int(slots[i])
            if pos < 0:
                continue
            children[pos].append(i)
            sibling[b, i] = nxt[b, pos]
            nxt[b, pos] = i
        # Reverse insertion prepends, so append order is opposite sibling order.
        for kids in children:
            kids.reverse()
        batch_children.append(children)
        row_tokens = cand_rows[b]
        for v, p in enumerate(slots):
            if p < 0:
                continue
            edges.append((b, v, int(p)))
            parent_rows.append(b * width + int(p))
            child_tokens.append(int(row_tokens[v]))
    tree = torch.stack((cand, retr, nxt, sibling))
    if edges:
        index = torch.tensor([parent_rows, child_tokens], dtype=torch.int64)
    else:
        index = torch.empty((2, 0), dtype=torch.int64)
    retrieve = [[b * width + v for v in range(width)] for b in range(bs)]
    topology = _RPDHostTopology(
        cand_rows,
        retrieve,
        batch_children,
        {(b, v): i for i, (b, v, _parent) in enumerate(edges)},
        tuple(edges),
    )
    return SRRPDHostInput(
        tree,
        edges,
        index,
        int(vocab),
        owner,
        generation,
        batch_key,
        topology,
    )


class SRRPDWorkspace:
    """One synchronous Target's reusable statistics and source lifetime."""

    def __init__(self):
        self.key = None
        self.rows_cap = self.edges_cap = 0
        self.event = None
        self.unresolved = False
        self.holds = None
        self._logged_paths = set()
        self.metrics = None
        self.edge_index = None
        self._v1_dev = None
        self._v1_host = None
        self._v1_cap = 0
        self._v1_key = None
        self._v1_pinned = False

    def log_path_once(self, name: str) -> None:
        if name in self._logged_paths:
            return
        self._logged_paths.add(name)
        logger.info("Speculative RPD verify path: %s", name)

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
            or not index.is_contiguous()
        ):
            return "edge_layout"
        device = logits.device
        # CPU stays for contract tests. CUDA and NPU share this workspace;
        # CUDA edge statistics use the reference gather below.
        if device.type not in ("cpu", "npu", "cuda"):
            return "device"
        if device.type != "cpu" and (
            not pin_supported(device) or not event_supported(device)
        ):
            return "staging"
        key = (device, logits.dtype)
        edges = len(context.edges)
        if edges:
            try:
                _edge_gather_impl(logits)
            except (ImportError, AttributeError):
                return "edge_kernel"
        if key != self.key or rows > self.rows_cap or edges > self.edges_cap:
            rc = max(rows, self.rows_cap, 1)
            ec = max(edges, self.edges_cap, 1)
            rc, ec = 1 << (rc - 1).bit_length(), 1 << (ec - 1).bit_length()
            star_host, star_pinned = alloc_host((rc,), torch.int64, device)
            stats_host, stats_pinned = alloc_host((2, ec), logits.dtype, device)
            if device.type != "cpu" and not (star_pinned and stats_pinned):
                return "staging"
            values = torch.empty(rc, dtype=logits.dtype, device=device)
            argmax = torch.empty(rc, dtype=torch.int64, device=device)
            edge_values = torch.empty((2, ec), dtype=logits.dtype, device=device)
            event = (
                torch.get_device_module(device.type).Event()
                if device.type != "cpu"
                else None
            )
            self.values, self.argmax, self.edge_values = values, argmax, edge_values
            self.star_host, self.stats_host = star_host, stats_host
            self.rows_cap, self.edges_cap, self.key = rc, ec, key
            self.event = event
            self.count("rpd_host_workspace_grow")
        self.edge_index = index
        return None

    def statistics(self, logits, context):
        if self.unresolved or context.owner.unresolved:
            raise SRTransferUnresolved("RPD statistics workspace is unresolved")
        bs, width = context.tree.shape[1:]
        rows, edges = bs * width, len(context.edges)
        # Metadata checks and lazy imports must finish before the first device
        # operation. An import failure does not imply an in-flight reduction.
        if self.key is None or rows > self.rows_cap or edges > self.edges_cap:
            raise RuntimeError("RPD statistics logits or prepared capacity mismatch")
        index = context.edge_index
        if index is not self.edge_index:
            raise RuntimeError("RPD statistics edge index changed")
        gather = _edge_gather_impl(logits) if edges else None
        if rows == 0:
            return self.star_host[:0].reshape(bs, width), None
        async_copy = logits.device.type != "cpu"
        if async_copy and self.event is None:
            raise RuntimeError("RPD statistics completion event is missing")
        values, argmax = self.values[:rows], self.argmax[:rows]
        # Edge buffers are flattened contiguous live regions, independent of
        # capacity; this also avoids strided D2H copies when the batch shrinks.
        stats = self.edge_values.reshape(-1)[: 2 * edges].reshape(2, edges)
        host_stats = self.stats_host.reshape(-1)[: 2 * edges].reshape(2, edges)
        host_star = self.star_host[:rows]
        if async_copy:
            self.holds = (
                logits,
                context,
                index,
                values,
                argmax,
                stats,
                host_star,
                host_stats,
            )
        try:
            torch.max(logits, dim=-1, out=(values, argmax))
            if edges and gather(values, logits, index, stats):
                self.count("rpd_host_edge_gather")
            host_star.copy_(argmax, non_blocking=async_copy)
            if edges:
                host_stats.copy_(stats, non_blocking=async_copy)
            if not async_copy:
                return host_star.reshape(bs, width), host_stats if edges else None
            self.count("rpd_host_stats_d2h_count", 1 + int(edges > 0))
            self.count(
                "rpd_host_stats_d2h_bytes",
                rows * 8 + 2 * edges * logits.element_size(),
            )
            self.event.record()
            self.count("rpd_host_stats_waits")
            wait_event(self.event)
        except BaseException as exc:
            if async_copy:
                # Even a reduction/gather launch can have submitted work. No
                # later packet fill, workspace growth or retry is safe yet.
                self.unresolved = True
                context.owner.unresolved = True
                if isinstance(exc, Exception) and not isinstance(
                    exc, SRTransferUnresolved
                ):
                    raise SRTransferUnresolved(
                        "RPD statistics failed after device submission; refuse reuse"
                    ) from exc
            raise
        self.holds = None
        return host_star.reshape(bs, width), host_stats if edges else None


def _host_array(value):
    """Host array for selection.

    ``_rpd_select_host`` converts the array with one ``tolist``, so the edge
    loop does not pay a scalar conversion per value.

    bfloat16 has no NumPy scalar type here. It is the high 16 bits of float32,
    so the cast matches the Python ``float()`` values used by selection.
    """
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.dtype == torch.bfloat16:
            value = value.to(torch.float32)
        return value.detach().cpu().contiguous().numpy()
    return value


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
    rows, lengths, folded = _rpd_select_host(
        context.topology,
        _host_array(star),
        None if stats is None else _host_array(stats),
        gap,
        float(tau) == 0.0,
        path_cap,
        bs * width,
        bs * width + 1,
    )
    token_rows = [[folded[i] if i >= 0 else 0 for i in row] for row in rows]
    workspace.count("rpd_host_plan_hit")
    return SRRPDHostPlan(rows, token_rows, lengths, context.vocab, context.batch_key)


def stage_v1_result(workspace, indices, tokens, lengths, device, select=()):
    """Upload one truncated V1 result on the current stream, without waiting.

    ``indices`` and ``tokens`` are the surviving accept positions and the
    tokens written there, in row order. ``lengths`` is one draft count per
    request, excluding the bonus. ``select`` holds offsets into those flat
    arrays for unfinished requests.

    Device and host storage grow by capacity and stay on ``workspace``.
    ``alloc_host`` runs only when the capacity grows or the device changes.
    This function does not call ``wait_event``. The H2D ``copy_`` is ordered
    on the current stream, so later consumers on that stream observe it
    without another wait.

    The host prefix may be overwritten on the next call. The next
    ``statistics()`` records ``wait_event`` on the same stream after the
    previous H2D, so that copy has finished before the overwrite.

    The four returned views alias the reused device buffer and are valid only
    for this round. Standalone-remote code does not read
    ``spec_info.verified_id`` or ``spec_info.accept_length`` across rounds.
    Same-round consumers are ``eagle_info.verify`` before it returns, and
    ``sr_worker`` handing ``verified_id`` back as ``next_token_ids`` (the
    output processor calls ``tolist()``). The next ``verify()`` replaces
    ``batch.spec_info`` before the next ``stage_v1_result``.
    """
    indices = list(indices)
    tokens = list(tokens)
    lengths = list(lengths)
    select = list(select)
    n_idx = len(indices)
    n_tok = len(tokens)
    n_len = len(lengths)
    n_sel = len(select)
    need = n_idx + n_tok + n_len + n_sel
    dev = device if isinstance(device, torch.device) else torch.device(device)
    key = device_cache_key(dev)
    if workspace._v1_key != key or workspace._v1_cap < max(need, 1):
        cap = max(need, workspace._v1_cap, 1)
        cap = 1 << (cap - 1).bit_length()
        workspace._v1_dev = torch.empty((cap,), dtype=torch.int32, device=dev)
        workspace._v1_host, workspace._v1_pinned = alloc_host(
            (cap,), torch.int32, dev
        )
        workspace._v1_cap = cap
        workspace._v1_key = key
    host = workspace._v1_host[:need]
    if need:
        host.copy_(
            torch.tensor(indices + tokens + lengths + select, dtype=torch.int32)
        )
        workspace._v1_dev[:need].copy_(
            host, non_blocking=workspace._v1_pinned and dev.type != "cpu"
        )
    buf = workspace._v1_dev
    return (
        buf[:n_idx],
        buf[n_idx : n_idx + n_tok],
        buf[n_idx + n_tok : n_idx + n_tok + n_len],
        buf[n_idx + n_tok + n_len : need],
    )


def host_plan_compact_payload(plan, accept_width):
    """Pack a host plan into the four tensors ``_rpd_compact_apply`` already writes.

    Nonnegative accept indices are the folded predict positions. A repeated
    retrieve keeps the last token and the first insertion order, matching
    ``_rpd_select_host``.
    """
    width = int(accept_width)
    if plan.rows:
        accept = torch.tensor(plan.rows, dtype=torch.int64)
    else:
        accept = torch.empty((0, width), dtype=torch.int64)
    lengths = torch.tensor(plan.pre_lengths, dtype=torch.int64)
    folded = {}
    for row, tokens in zip(plan.rows, plan.tokens):
        for idx, tok in zip(row, tokens):
            if idx >= 0:
                folded[int(idx)] = int(tok)
    if folded:
        positions = torch.tensor(list(folded.keys()), dtype=torch.int64)
        values = torch.tensor(list(folded.values()), dtype=torch.int64)
    else:
        positions = torch.empty((0,), dtype=torch.int64)
        values = torch.empty((0,), dtype=torch.int64)
    return accept, lengths, positions, values
