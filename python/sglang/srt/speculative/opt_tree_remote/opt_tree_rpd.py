"""OPT target adapter for the existing RPD host plan.

The topology is built here from OPT parent pointers. Selection, statistics,
and the verify postprocess stay in the shared RPD path. This module does not
change standalone-remote packets or message fields.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from sglang.srt.speculative.opt_tree_remote.opt_tree_layout import opt_parent_slots


class OptRpdOwner:
    """Lifetime of one OPT target's RPD host input.

    ``SRRPDHostInput.current`` only reads ``generation``, ``rpd_input``, and
    ``unresolved``. The OPT worker keeps this object and does not rewrite the
    standalone-remote verify packet.
    """

    def __init__(self) -> None:
        self.generation = 0
        self.rpd_input = None
        self.unresolved = False


def attach_opt_rpd_input(
    owner: Optional[OptRpdOwner],
    server_args,
    spec_info,
    batch,
    roots: Sequence[int],
    token_rows: Sequence[Sequence[int]],
    parent_rows: Sequence[Sequence[int]],
    no_draft: Sequence[bool],
    device,
) -> Optional[OptRpdOwner]:
    """Attach one host-plan input when this target is verifying with RPD.

    ``N == 1`` leaves the input unset. The verify path then uses
    ``verify_tree_rpd`` on the retrieve links. A declined layout raises instead
    of falling through to that compact path.
    """
    if getattr(server_args, "speculative_verify_mode", None) != "rpd":
        return owner
    width = int(server_args.opt_tree_width)
    if width <= 1:
        return owner
    from sglang.srt.speculative.standalone_remote.sr_rpd import (
        build_sr_rpd_input,
        rpd_batch_key,
    )

    if owner is None:
        owner = OptRpdOwner()
    if owner.unresolved:
        raise RuntimeError("OPT RPD verify input is unresolved")
    owner.generation += 1
    owner.rpd_input = None
    bs = len(roots)
    tokens = torch.tensor(token_rows, dtype=torch.int64)
    context = build_sr_rpd_input(
        torch.tensor(list(roots), dtype=torch.int64),
        tokens,
        torch.tensor(parent_rows, dtype=torch.int64),
        torch.zeros_like(tokens),
        topk=width,
        steps=int(server_args.opt_tree_max_depth),
        width=width + 1,
        vocab=int(batch.sampling_info.vocab_size),
        owner=owner,
        generation=owner.generation,
        batch_key=rpd_batch_key(list(batch.reqs)[:bs]),
        parent_slots=opt_parent_slots(parent_rows, no_draft),
    )
    if context is None:
        raise RuntimeError("OPT RPD verify input was declined")
    context.edge_index = context.edge_index_cpu.to(device=device).contiguous()
    owner.rpd_input = context
    spec_info.sr_rpd_input = context
    return owner
