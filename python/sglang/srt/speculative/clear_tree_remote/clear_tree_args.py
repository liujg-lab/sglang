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

    ``--clear-tree-width`` is N. ``--speculative-num-draft-tokens`` follows the
    EAGLE split: Target stores the verify length N + 1 (committed tail plus N
    nodes), Draft stores the per-layer width N. An omitted value is filled
    with that role's length; an explicit value must already equal it.
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
    width = int(server_args.clear_tree_width)
    depth = int(server_args.clear_tree_max_depth)
    role = server_args.clear_tree_role
    expected_draft_tokens = width + 1 if role == "target" else width
    if server_args.speculative_num_draft_tokens is None:
        server_args.speculative_num_draft_tokens = expected_draft_tokens
    elif int(server_args.speculative_num_draft_tokens) != expected_draft_tokens:
        if role == "target":
            raise ValueError(
                "CLEAR_TREE_REMOTE target requires --speculative-num-draft-tokens "
                "to equal --clear-tree-width + 1 "
                f"({expected_draft_tokens})."
            )
        raise ValueError(
            "CLEAR_TREE_REMOTE draft requires --speculative-num-draft-tokens "
            f"to equal --clear-tree-width ({expected_draft_tokens})."
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
    if server_args.speculative_eagle_topk not in (None, int(server_args.clear_tree_width)):
        logger.warning(
            "speculative_eagle_topk is set to clear_tree_width (%s) so NPU "
            "tree verify keeps topk greater than 1.",
            server_args.clear_tree_width,
        )
    server_args.speculative_eagle_topk = int(server_args.clear_tree_width)
    logger.warning(
        "The overlap scheduler and mixed chunked prefill are disabled "
        "because of using CLEAR_TREE_REMOTE speculative decoding."
    )
