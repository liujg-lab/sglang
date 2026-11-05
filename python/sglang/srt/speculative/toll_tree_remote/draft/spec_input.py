"""Layer spec read by TARGET_VERIFY attention. It must not use the decode graph."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TollLayerSpec:
    custom_mask: object
    positions: object
    draft_token_num: int
    num_tokens_per_req: int
    capture_hidden_mode: object = None
    apply_prefix_custom_mask: bool = True
    is_bonus: bool = False
    target_verify_attention: object = None

    forbids_decode_graph = True

    def is_draft_input(self) -> bool:
        return False

    def get_spec_adjusted_global_num_tokens(self, batch):
        return batch.global_num_tokens, batch.global_num_tokens_for_logprob
