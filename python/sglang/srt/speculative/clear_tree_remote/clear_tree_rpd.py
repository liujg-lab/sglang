"""ClearTree target adapter for the existing RPD host plan.

The topology is built here from ClearTree parent pointers. Its width is this
round's verify length K + 1, not the build width W. Selection, statistics, and
the verify postprocess stay in the shared RPD path. This module does not change
standalone-remote packets or message fields.
"""

from __future__ import annotations

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
    from sglang.srt.speculative.standalone_remote.sr_rpd import (
        build_sr_rpd_input,
        rpd_batch_key,
    )

    if owner is None:
        owner = ClearRpdOwner()
    if owner.unresolved:
        raise RuntimeError("ClearTree RPD verify input is unresolved")
    owner.generation += 1
    owner.rpd_input = None
    bs = len(roots)
    owner.rids = [str(getattr(req, "rid", "")) for req in list(batch.reqs)[:bs]]
    owner.no_draft = [bool(flag) for flag in no_draft]
    tokens = torch.tensor(token_rows, dtype=torch.int64).reshape(
        bs, int(num_verify) - 1
    )
    context = build_sr_rpd_input(
        torch.tensor(list(roots), dtype=torch.int64),
        tokens,
        torch.tensor(parent_rows, dtype=torch.int64).reshape(bs, int(num_verify) - 1),
        torch.zeros_like(tokens),
        topk=int(tree_topk),
        steps=int(max_depth),
        width=int(num_verify),
        vocab=int(batch.sampling_info.vocab_size),
        owner=owner,
        generation=owner.generation,
        batch_key=rpd_batch_key(list(batch.reqs)[:bs]),
        parent_slots=opt_parent_slots(parent_rows, no_draft),
    )
    if context is None:
        raise RuntimeError(
            f"ClearTree RPD verify input was declined (K+1={int(num_verify)}, "
            f"topk={int(tree_topk)}, depth={int(max_depth)})"
        )
    context.edge_index = context.edge_index_cpu.to(device=device).contiguous()
    owner.rpd_input = context
    spec_info.sr_rpd_input = context
    return owner


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
