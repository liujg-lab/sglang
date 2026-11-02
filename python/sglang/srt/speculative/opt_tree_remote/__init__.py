"""Original OPT-Tree for a remote draft/target split.

AdaCost, cost models, and the online width controller are not part of this
package. Draft width N and threshold tau stay fixed for a round.
"""

from sglang.srt.speculative.opt_tree_remote.opt_tree_config import (
    OptTreeConfig,
    longest_draft_chain,
)
from sglang.srt.speculative.opt_tree_remote.opt_tree_layout import (
    OptVerifyLayout,
    build_tree_from_parents,
)
from sglang.srt.speculative.opt_tree_remote.opt_tree_ops import (
    OptTreeState,
    OptTreeStep,
    OptVerifySet,
    ancestor_mask,
    depth_rule,
    expand_layer,
    finish,
    seed,
    topk_probs_from_logits,
    update,
)

__all__ = [
    "OptTreeConfig",
    "OptTreeState",
    "OptTreeStep",
    "OptVerifyLayout",
    "OptVerifySet",
    "ancestor_mask",
    "build_tree_from_parents",
    "depth_rule",
    "expand_layer",
    "finish",
    "longest_draft_chain",
    "seed",
    "topk_probs_from_logits",
    "update",
]
