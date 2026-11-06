"""Choose whether NPU tree attention enters the shared target-FIA path.

Toll does not appear by name. Its target sets ``npu_target_tree_fia``.
OPT keeps the name check it already had.
"""

from sglang.srt.speculative.spec_info import remote_spec_role

TREE_PAGED_FIA = "tree_paged_fia"


def enter_target_tree_fia(args, verify_tree_topk) -> bool:
    """False means the caller leaves the impl chosen before this selection."""
    if int(verify_tree_topk) <= 1:
        return False
    algo = getattr(args, "speculative_algorithm", None)
    role = remote_spec_role(args)
    if algo == "OPT_TREE_REMOTE" or bool(getattr(args, "npu_target_tree_fia", False)):
        return role == "target"
    return algo == "STANDALONE_REMOTE"


def restore_initial_tree_impl(args, selected_impl, initial_impl):
    """OPT and the flagged target keep a non-paged choice from before selection."""
    flagged = getattr(args, "speculative_algorithm", None) == "OPT_TREE_REMOTE" or bool(
        getattr(args, "npu_target_tree_fia", False)
    )
    if flagged and selected_impl != TREE_PAGED_FIA:
        return initial_impl
    return selected_impl
