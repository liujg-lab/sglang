"""ClearTree target adapter for the existing RPD host plan.

The topology is built here from ClearTree parent pointers. Its width is this
round's verify length K + 1, not the build width W. Selection, statistics, and
the verify postprocess stay in the shared RPD path. This module does not change
standalone-remote packets or message fields.

``build_clear_rpd_input`` returns the same ``SRRPDHostInput`` as the shared
``build_sr_rpd_input`` but links siblings in Python lists and makes each tensor
once. The shared builder writes two tensor scalars per edge; on the NPU host
that was about 17 us per verified token inside the Draft's H (run 2026-11-26-d).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import torch

from sglang.srt.speculative.clear_tree_remote.clear_tree_layout import opt_parent_slots


class ClearRpdOwner:
    """Lifetime of one ClearTree target's RPD host input.

    ``SRRPDHostInput.current`` only reads ``generation``, ``rpd_input``, and
    ``unresolved``. The ClearTree worker keeps this object and does not rewrite
    the standalone-remote verify packet.
    """

    def __init__(self) -> None:
        self.generation = 0
        self.rpd_input = None
        self.unresolved = False
        # Rows of ``rpd_input`` in build order, for the acceptable-slot masks.
        self.rids: List[str] = []
        self.no_draft: List[bool] = []
        # Host time of the last attach (topology build and edge upload), us.
        self.build_us: Optional[int] = None


def attach_clear_rpd_input(
    owner: Optional[ClearRpdOwner],
    server_args,
    spec_info,
    batch,
    roots: Sequence[int],
    token_rows: Sequence[Sequence[int]],
    parent_rows: Sequence[Sequence[int]],
    no_draft: Sequence[bool],
    device,
    *,
    num_verify: int,
    tree_topk: int,
    max_depth: int,
) -> Optional[ClearRpdOwner]:
    """Attach one host-plan input when this target is verifying with RPD.

    ``num_verify`` is K + 1 for this round. Every token row is already padded
    to K, and every parent-slot row has K + 1 entries. ``tree_topk`` is the
    verify input's topk, which is at least 2, so the shared verify always asks
    for a host topology. A declined layout raises instead of falling through.
    """
    if getattr(server_args, "speculative_verify_mode", None) != "rpd":
        return owner
    from sglang.srt.speculative.standalone_remote.sr_rpd import rpd_batch_key

    started = time.perf_counter_ns()
    if owner is None:
        owner = ClearRpdOwner()
    if owner.unresolved:
        raise RuntimeError("ClearTree RPD verify input is unresolved")
    owner.generation += 1
    owner.rpd_input = None
    owner.build_us = None
    bs = len(roots)
    owner.rids = [str(getattr(req, "rid", "")) for req in list(batch.reqs)[:bs]]
    owner.no_draft = [bool(flag) for flag in no_draft]
    context = None
    # The shared builder declines topk <= 1 and steps < 1 the same way.
    if int(tree_topk) > 1 and int(max_depth) >= 1:
        context = build_clear_rpd_input(
            roots,
            token_rows,
            opt_parent_slots(parent_rows, no_draft),
            width=int(num_verify),
            vocab=int(batch.sampling_info.vocab_size),
            owner=owner,
            generation=owner.generation,
            batch_key=rpd_batch_key(list(batch.reqs)[:bs]),
        )
    if context is None:
        raise RuntimeError(
            f"ClearTree RPD verify input was declined (K+1={int(num_verify)}, "
            f"topk={int(tree_topk)}, depth={int(max_depth)})"
        )
    context.edge_index = context.edge_index_cpu.to(device=device).contiguous()
    owner.rpd_input = context
    spec_info.sr_rpd_input = context
    owner.build_us = (time.perf_counter_ns() - started) // 1000
    return owner


def build_clear_rpd_input(
    roots: Sequence[int],
    token_rows: Sequence[Sequence[int]],
    parent_slots: Sequence[Sequence[int]],
    *,
    width: int,
    vocab: int,
    owner,
    generation: int,
    batch_key=(),
):
    """``build_sr_rpd_input`` for ClearTree rows, without per-edge tensor writes.

    ``width`` is K + 1. ``token_rows`` are the K draft tokens per request and
    ``parent_slots`` the K + 1 verify-slot parents from ``opt_parent_slots``.
    The result equals the shared builder's for the same rows: tree
    ``[cand, retrieve, next_token, next_sibling]``, edges in (row, slot) order,
    the edge index, and a topology whose children keep ``build_tree_efficient``
    sibling order. None declines: a row of the wrong length, a parent slot
    past the row (the shared builder would raise there), or an empty width
    or vocabulary.
    """
    from sglang.srt.speculative.rpd_verify import _RPDHostTopology
    from sglang.srt.speculative.standalone_remote.sr_rpd import SRRPDHostInput

    width = int(width)
    bs = len(roots)
    if (
        width < 1
        or int(vocab) < 1
        or len(token_rows) != bs
        or len(parent_slots) != bs
    ):
        return None
    cand_rows: List[List[int]] = []
    next_rows: List[List[int]] = []
    sibling_rows: List[List[int]] = []
    batch_children: list = []
    edge_of: List[List[int]] = []
    edges: list = []
    parent_index: List[int] = []
    child_tokens: List[int] = []
    for b in range(bs):
        tokens, slots = token_rows[b], parent_slots[b]
        if len(tokens) != width - 1 or len(slots) != width:
            return None
        row = [int(roots[b])] + [int(token) for token in tokens]
        slots = [int(slot) for slot in slots]
        # Negative slots are unlinked, as in the shared builder.
        if any(slot >= width for slot in slots):
            return None
        nxt = [-1] * width
        sibling = [-1] * width
        children: List[List[int]] = [[] for _ in range(width)]
        for i in range(width - 1, 0, -1):
            pos = slots[i]
            if pos < 0:
                continue
            children[pos].append(i)
            sibling[i] = nxt[pos]
            nxt[pos] = i
        # Reverse insertion prepends, so append order is opposite sibling order.
        for kids in children:
            kids.reverse()
        row_edges = [-1] * width
        for v, parent in enumerate(slots):
            if parent < 0:
                continue
            row_edges[v] = len(edges)
            edges.append((b, v, parent))
            parent_index.append(b * width + parent)
            child_tokens.append(row[v])
        cand_rows.append(row)
        next_rows.append(nxt)
        sibling_rows.append(sibling)
        batch_children.append(children)
        edge_of.append(row_edges)
    retrieve = [[b * width + v for v in range(width)] for b in range(bs)]
    shape = (4, bs, width)
    tree = torch.tensor(
        [cand_rows, retrieve, next_rows, sibling_rows], dtype=torch.int64
    ).reshape(shape)
    if edges:
        index = torch.tensor([parent_index, child_tokens], dtype=torch.int64)
    else:
        index = torch.empty((2, 0), dtype=torch.int64)
    topology = _RPDHostTopology(
        cand_rows,
        retrieve,
        batch_children,
        edge_of,
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


# The shared host plan counts this once, right after ``_rpd_select_host``.
PLAN_HIT_COUNT = "rpd_host_plan_hit"
_TIMED_WORKSPACE = None


def clear_rpd_workspace():
    """A shared ``SRRPDWorkspace`` that also stamps its host stages.

    ``statistics`` (edge gather, D2H copies, and the wait for the verify
    forward to finish) and host selection (``statistics`` returning until the
    shared plan counts ``PLAN_HIT_COUNT``) get ``perf_counter_ns`` stamps in
    ``clear_stamps``. Results, counts, and the shared class are unchanged.
    """
    global _TIMED_WORKSPACE
    if _TIMED_WORKSPACE is None:
        from sglang.srt.speculative.standalone_remote.sr_rpd import SRRPDWorkspace

        class ClearRpdWorkspace(SRRPDWorkspace):
            def __init__(self):
                super().__init__()
                self.clear_stamps: Dict[str, int] = {}

            def statistics(self, logits, context):
                self.clear_stamps = {"stats_start": time.perf_counter_ns()}
                try:
                    return super().statistics(logits, context)
                finally:
                    self.clear_stamps["stats_end"] = time.perf_counter_ns()

            def count(self, name, value=1):
                stamps = self.clear_stamps
                if name == PLAN_HIT_COUNT and "stats_end" in stamps:
                    stamps.setdefault("plan", time.perf_counter_ns())
                super().count(name, value)

        _TIMED_WORKSPACE = ClearRpdWorkspace
    return _TIMED_WORKSPACE()


def rpd_timing(owner, workspace, spec_info, since_ns, mask_us=None) -> dict:
    """RPD host stage times of one verify, in microseconds. Never raises.

    ``rpd_build_us`` is the attach of the context this verify used, which ran
    before ``since_ns`` while the verify input was built. ``rpd_stats_us`` and
    ``rpd_select_us`` count only stamps taken after ``since_ns``, so a verify
    that fell back to the device path leaves them null. ``rpd_mask_us`` is the
    caller's timing of ``rpd_valid_masks``.
    """
    out = {
        "rpd_build_us": None,
        "rpd_stats_us": None,
        "rpd_select_us": None,
        "rpd_mask_us": None if mask_us is None else int(mask_us),
    }
    try:
        context = None if owner is None else getattr(owner, "rpd_input", None)
        if context is not None and context is getattr(spec_info, "sr_rpd_input", None):
            build = getattr(owner, "build_us", None)
            out["rpd_build_us"] = None if build is None else int(build)
        stamps = getattr(workspace, "clear_stamps", None) or {}
        start, end = stamps.get("stats_start"), stamps.get("stats_end")
        if start is not None and end is not None and start >= int(since_ns):
            out["rpd_stats_us"] = (int(end) - int(start)) // 1000
            plan = stamps.get("plan")
            if plan is not None and plan >= end:
                out["rpd_select_us"] = (int(plan) - int(end)) // 1000
    except Exception:
        pass
    return out


def rpd_valid_masks(owner, workspace, tau) -> Optional[Dict[str, str]]:
    """Acceptable verify slots per request after a host-plan RPD verify.

    Bit ``v - 1`` is set when slot ``v`` (1..K) is acceptable on its own edge,
    by the same rule as ``_rpd_select_host``: the logit gap is at most
    ``rpd_gap_max(tau)``, or the token equals the parent's argmax when tau is 0.
    A node is accepted only when it and every ancestor are acceptable; the
    Draft applies that, with its ranks, to price smaller windows.

    The statistics are the workspace host buffers filled for this context.
    They stay until the next ``prepare``. None when this round did not run the
    host plan (device fallback), or anything is unresolved or stale.
    """
    context = None if owner is None else getattr(owner, "rpd_input", None)
    if context is None or workspace is None:
        return None
    if (
        not getattr(context, "consumed", False)
        or getattr(owner, "unresolved", False)
        or getattr(workspace, "unresolved", False)
        or getattr(workspace, "holds", None) is not None
        or getattr(workspace, "edge_index", None) is not context.edge_index
    ):
        return None
    from sglang.srt.speculative.rpd_verify import rpd_gap_max

    bs, width = (int(x) for x in context.tree.shape[1:])
    rids = list(getattr(owner, "rids", None) or [])
    no_draft = list(getattr(owner, "no_draft", None) or [])
    if len(rids) != bs or len(no_draft) != bs:
        return None
    edges = context.edges
    count = len(edges)
    use_equality = float(tau) == 0.0
    gap_max = rpd_gap_max(tau)
    if count:
        stats = workspace.stats_host.reshape(-1)[: 2 * count].reshape(2, count).tolist()
    else:
        stats = [[], []]
    star = None
    if use_equality:
        star = workspace.star_host[: bs * width].reshape(bs, width).tolist()
    candidates = context.topology.candidates
    masks = [0] * bs
    for edge_i, (b, v, parent) in enumerate(edges):
        b, v = int(b), int(v)
        if use_equality:
            ok = int(candidates[b][v]) == int(star[b][int(parent)])
        else:
            ok = float(stats[0][edge_i]) - float(stats[1][edge_i]) <= gap_max
        if ok and v >= 1:
            masks[b] |= 1 << (v - 1)
    return {
        rid: format(mask, "x")
        for rid, mask, missing in zip(rids, masks, no_draft)
        if not missing
    }
