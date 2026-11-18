"""CUDA-graph capture for CLEAR_TREE_REMOTE.

The shared runner asks ``SpeculativeAlgorithm.graph_capture_delegate`` and
then calls these methods. It does not read ClearTree fields itself.
"""

from __future__ import annotations

from typing import Optional


def _capture_mask(runner, num_tokens: int, query_width: int):
    """Synthetic all-visible mask for this bs/G, retained for graph lifetime.

    Shared input buffers are sized using the layer width, which can be
    smaller than a K+1 bonus graph. Capture metadata reads
    ``bs * G * (seq_len_fill_value + G)`` entries even when the shared mask
    was allocated with fewer. Keep any replacement on the runner rather
    than letting capture refer to a temporary allocation.
    """
    required = int(num_tokens) * (int(runner.seq_len_fill_value) + int(query_width))
    mask = runner.buffers.custom_mask
    if mask.numel() >= required:
        return mask[:required]
    import torch

    masks = getattr(runner, "_clear_tree_capture_masks", None)
    if masks is None:
        masks = {}
        runner._clear_tree_capture_masks = masks
    if required not in masks:
        masks[required] = torch.ones(required, dtype=mask.dtype, device=mask.device)
    return masks[required]


class ClearTreeGraphCapture:
    log_tag = "[ClearTree CudaGraph]"

    @staticmethod
    def dual_ntpb_options(server_args) -> list:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
            capture_ntpb,
        )

        return capture_ntpb(getattr(server_args, "clear_tree_role", None))

    @staticmethod
    def draft_buffer_tokens(server_args) -> Optional[int]:
        """Keep max K + 1 draft capacity; target keeps its verify capacity."""
        if getattr(server_args, "clear_tree_role", None) != "draft":
            return None
        from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
            BONUS_NTPBS,
        )

        return max(BONUS_NTPBS)

    @staticmethod
    def capture_extra_keys(server_args, ntpb) -> list:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_forward import (
            opt_draft_capture_extra_keys,
        )

        return opt_draft_capture_extra_keys(server_args, ntpb)

    @staticmethod
    def replay_graph_extra(server_args, forward_batch):
        if getattr(server_args, "clear_tree_role", None) != "draft":
            return None
        if not forward_batch.forward_mode.is_target_verify():
            return None
        from sglang.srt.speculative.clear_tree_remote.clear_tree_forward import (
            opt_bonus_graph_extra,
        )

        return opt_bonus_graph_extra(forward_batch.spec_info)

    @staticmethod
    def bonus_ntpbs_for_capture(server_args) -> list:
        """No bs=1-only extras: bonus widths are already in ``dual_ntpb_options``.

        The shared runner treats this supplemental list as unconditionally
        bs=1-only. ClearTree captures bonus graphs at every batch size that
        fits the ordinary token buffers instead.
        """
        return []

    @staticmethod
    def is_draft(server_args) -> bool:
        return getattr(server_args, "clear_tree_role", None) == "draft"

    @staticmethod
    def owns_verify_capture(server_args) -> bool:
        return getattr(server_args, "clear_tree_role", None) in ("target", "draft")

    @staticmethod
    def layer_verify_graph_matches(spec_info, actual_ntpb: int) -> bool:
        from sglang.srt.speculative.clear_tree_remote.clear_tree_forward import (
            opt_layer_verify_graph_matches,
        )

        return opt_layer_verify_graph_matches(spec_info, actual_ntpb)

    @staticmethod
    def capture_verify_spec(runner, num_tokens: int, ntpb_override, uses_dual: bool):
        """Spec info recorded into one TARGET_VERIFY graph, or None for DECODE."""
        server_args = runner.model_runner.server_args
        role = getattr(server_args, "clear_tree_role", None)
        draft_token_num = (
            ntpb_override if ntpb_override is not None else runner.num_tokens_per_bs
        )
        if uses_dual and draft_token_num == 1:
            return None
        if role not in ("target", "draft"):
            return None
        if role == "target":
            from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
            from sglang.srt.speculative.eagle_info import EagleVerifyInput

            return EagleVerifyInput(
                draft_token=None,
                custom_mask=runner.buffers.custom_mask,
                positions=None,
                retrive_index=None,
                retrive_next_token=None,
                retrive_next_sibling=None,
                retrive_cum_len=None,
                spec_steps=int(server_args.clear_tree_max_depth),
                topk=max(int(server_args.speculative_eagle_topk or 2), 2),
                draft_token_num=int(draft_token_num),
                capture_hidden_mode=CaptureHiddenMode.FULL,
                seq_lens_sum=None,
                seq_lens_cpu=None,
            )
        if role != "draft":
            return None
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
        from sglang.srt.speculative.clear_tree_remote.clear_tree_attention import (
            capture_attention,
        )
        from sglang.srt.speculative.clear_tree_remote.clear_tree_forward import (
            OPT_BONUS_GRAPH_EXTRA,
            OptLayerSpec,
        )
        from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
            BONUS_NTPBS,
            LAYER_WIDTHS,
        )

        query_width = int(draft_token_num)
        is_bonus = (
            query_width in BONUS_NTPBS
            and getattr(runner, "_active_capture_extra", None) == OPT_BONUS_GRAPH_EXTRA
        )
        if query_width not in LAYER_WIDTHS and not is_bonus:
            raise RuntimeError(
                f"CLEAR_TREE_REMOTE draft TARGET_VERIFY graph ntpb {query_width} "
                f"is not a layer width {LAYER_WIDTHS} or a bonus width {BONUS_NTPBS}"
            )
        spec_info = OptLayerSpec(
            custom_mask=_capture_mask(runner, num_tokens, query_width),
            positions=runner.buffers.positions[:num_tokens],
            draft_token_num=query_width,
            num_tokens_per_req=query_width,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            apply_prefix_custom_mask=True,
            is_bonus=is_bonus,
        )
        capture_bs = num_tokens // query_width if query_width else 0
        spec_info.target_verify_attention = capture_attention(
            runner.model_runner, query_width, capture_bs
        )
        return spec_info
