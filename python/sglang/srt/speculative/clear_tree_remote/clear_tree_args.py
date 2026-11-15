"""Argument checks for CLEAR_TREE_REMOTE.

Kept separate from ``ServerArgs`` so the rules can run without importing the
server module.
"""

from __future__ import annotations

import logging

from sglang.srt.speculative.clear_tree_remote.clear_tree_config import ClearTreeConfig

logger = logging.getLogger(__name__)


def apply_clear_tree_settings(server_args, *, on_npu: bool) -> None:
    """Check ClearTree flags and turn off overlap, mixed chunk, and co-located draft.

    ``--clear-tree-width`` stays the wire width shared by the two roles.
    ``--speculative-num-draft-tokens`` is buffer capacity, not the per-round
    length: Target holds the longest verify graph (33) and Draft holds the
    widest layer (8). An omitted value is filled with that capacity; an
    explicit value must already equal it.
    ``--clear-tree-max-depth`` includes the seed layer. When
    ``--speculative-num-steps`` is omitted it is set to that depth; an explicit
    value must already equal it.
    """
    server_args.disable_overlap_schedule = True
    server_args.enable_mixed_chunk = False
    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for CLEAR_TREE_REMOTE. "
            "You can override this by explicitly setting --max-running-requests."
        )
    if server_args.clear_tree_role not in ("target", "draft"):
        raise ValueError("CLEAR_TREE_REMOTE requires --clear-tree-role {target|draft}.")
    if (
        server_args.speculative_verify_mode == "rpd"
        and server_args.clear_tree_role != "target"
    ):
        raise ValueError(
            "CLEAR_TREE_REMOTE accepts --speculative-verify-mode rpd only on the target."
        )
    if server_args.clear_tree_width is None:
        raise ValueError("CLEAR_TREE_REMOTE requires --clear-tree-width.")
    if server_args.clear_tree_max_depth is None:
        raise ValueError("CLEAR_TREE_REMOTE requires --clear-tree-max-depth.")
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        DRAFT_CAPACITY,
        TARGET_CAPACITY,
    )

    width = int(server_args.clear_tree_width)
    depth = int(server_args.clear_tree_max_depth)
    role = server_args.clear_tree_role
    expected_draft_tokens = TARGET_CAPACITY if role == "target" else DRAFT_CAPACITY
    if server_args.speculative_num_draft_tokens is None:
        server_args.speculative_num_draft_tokens = expected_draft_tokens
    elif int(server_args.speculative_num_draft_tokens) != expected_draft_tokens:
        if role == "target":
            raise ValueError(
                "CLEAR_TREE_REMOTE target requires --speculative-num-draft-tokens "
                f"to equal the verify capacity {expected_draft_tokens}."
            )
        raise ValueError(
            "CLEAR_TREE_REMOTE draft requires --speculative-num-draft-tokens "
            f"to equal the layer capacity {expected_draft_tokens}."
        )
    if server_args.speculative_num_steps is None:
        server_args.speculative_num_steps = depth
    elif int(server_args.speculative_num_steps) != depth:
        raise ValueError(
            "CLEAR_TREE_REMOTE requires --speculative-num-steps to equal "
            f"--clear-tree-max-depth ({depth})."
        )
    ClearTreeConfig(
        width=int(server_args.clear_tree_width),
        tau=float(server_args.clear_tree_tau),
        max_depth=int(server_args.clear_tree_max_depth),
    )
    server_args.remote_spec_shape = (width, depth, width)
    page = int(server_args.page_size or 1)
    if on_npu:
        expected_page, where = 128, "NPU"
    else:
        expected_page, where = 1, "CUDA"
    if page != expected_page:
        raise ValueError(
            f"CLEAR_TREE_REMOTE on {where} requires --page-size {expected_page}, "
            f"got {page}."
        )
    tree_topk = max(int(server_args.clear_tree_width), 2)
    if server_args.speculative_eagle_topk not in (None, tree_topk):
        logger.warning(
            "speculative_eagle_topk is set to %s so NPU tree verify keeps "
            "topk greater than 1. It is not the verify count.",
            tree_topk,
        )
    server_args.speculative_eagle_topk = tree_topk
    logger.warning(
        "The overlap scheduler and mixed chunked prefill are disabled "
        "because of using CLEAR_TREE_REMOTE speculative decoding."
    )
