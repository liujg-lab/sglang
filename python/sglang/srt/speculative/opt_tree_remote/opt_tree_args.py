"""Argument checks for OPT_TREE_REMOTE.

Kept separate from ``ServerArgs`` so the rules can run without importing the
server module.
"""

from __future__ import annotations

import logging

from sglang.srt.speculative.opt_tree_remote.opt_tree_config import OptTreeConfig

logger = logging.getLogger(__name__)


def apply_opt_tree_settings(server_args, *, on_npu: bool) -> None:
    """Check OPT flags and turn off overlap, mixed chunk, and co-located draft.

    ``--opt-tree-width`` is N. When ``--speculative-num-draft-tokens`` is
    omitted it is set to N; an explicit value must already equal N. The verify
    forward adds the committed tail, so it runs N + 1 tokens.
    ``--opt-tree-max-depth`` includes the seed layer. When
    ``--speculative-num-steps`` is omitted it is set to that depth; an explicit
    value must already equal it.
    """
    server_args.disable_overlap_schedule = True
    server_args.enable_mixed_chunk = False
    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for OPT_TREE_REMOTE. "
            "You can override this by explicitly setting --max-running-requests."
        )
    if server_args.speculative_verify_mode == "rpd":
        raise ValueError(
            "OPT_TREE_REMOTE does not support --speculative-verify-mode rpd."
        )
    if server_args.opt_tree_role not in ("target", "draft"):
        raise ValueError("OPT_TREE_REMOTE requires --opt-tree-role {target|draft}.")
    if server_args.opt_tree_width is None:
        raise ValueError("OPT_TREE_REMOTE requires --opt-tree-width.")
    if server_args.opt_tree_max_depth is None:
        raise ValueError("OPT_TREE_REMOTE requires --opt-tree-max-depth.")
    width = int(server_args.opt_tree_width)
    depth = int(server_args.opt_tree_max_depth)
    if server_args.speculative_num_draft_tokens is None:
        server_args.speculative_num_draft_tokens = width
    elif int(server_args.speculative_num_draft_tokens) != width:
        raise ValueError(
            "OPT_TREE_REMOTE requires --speculative-num-draft-tokens to equal "
            f"--opt-tree-width ({width})."
        )
    if server_args.speculative_num_steps is None:
        server_args.speculative_num_steps = depth
    elif int(server_args.speculative_num_steps) != depth:
        raise ValueError(
            "OPT_TREE_REMOTE requires --speculative-num-steps to equal "
            f"--opt-tree-max-depth ({depth})."
        )
    OptTreeConfig(
        width=int(server_args.opt_tree_width),
        tau=float(server_args.opt_tree_tau),
        max_depth=int(server_args.opt_tree_max_depth),
    )
    page = int(server_args.page_size or 1)
    if on_npu:
        expected_page, where = 128, "NPU"
    else:
        expected_page, where = 1, "CUDA"
    if page != expected_page:
        raise ValueError(
            f"OPT_TREE_REMOTE on {where} requires --page-size {expected_page}, "
            f"got {page}."
        )
    if server_args.speculative_eagle_topk not in (None, int(server_args.opt_tree_width)):
        logger.warning(
            "speculative_eagle_topk is set to opt_tree_width (%s) so both "
            "OPT processes share one RPC spec shape.",
            server_args.opt_tree_width,
        )
    server_args.speculative_eagle_topk = int(server_args.opt_tree_width)
    logger.warning(
        "The overlap scheduler and mixed chunked prefill are disabled "
        "because of using OPT_TREE_REMOTE speculative decoding."
    )
