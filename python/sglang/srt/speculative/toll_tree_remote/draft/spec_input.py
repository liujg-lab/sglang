"""Layer spec read by TARGET_VERIFY attention. It must not use the decode graph."""

from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.speculative.spec_info import SpecInput, SpecInputType


@dataclass
class TollLayerSpec(SpecInput):
    custom_mask: object
    positions: object
    draft_token_num: int
    num_tokens_per_req: int
    capture_hidden_mode: object = None
    apply_prefix_custom_mask: bool = True
    is_bonus: bool = False
    target_verify_attention: object = None

    forbids_decode_graph = True

    def __post_init__(self) -> None:
        SpecInput.__init__(self, SpecInputType.EAGLE_VERIFY)

    def is_draft_input(self) -> bool:
        return False

    def get_spec_adjust_token_coefficient(self):
        return 1, 1

    def get_spec_adjusted_global_num_tokens(self, batch):
        return batch.global_num_tokens, batch.global_num_tokens_for_logprob

    def generate_attn_arg_prefill(
        self,
        req_pool_indices,
        paged_kernel_lens,
        paged_kernel_lens_sum: int,
        req_to_token,
    ):
        """KV indices for one layer, plus the mask this layer already built."""
        import torch

        from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton

        device = req_pool_indices.device
        batch_size = len(req_pool_indices)
        width = int(self.draft_token_num)
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * width,
            step=width,
            dtype=torch.int32,
            device=device,
        )
        kv_lens = paged_kernel_lens + width
        cum_kv_seq_len = torch.zeros((batch_size + 1,), dtype=torch.int32, device=device)
        cum_kv_seq_len[1:] = torch.cumsum(kv_lens, dim=0)
        if paged_kernel_lens_sum is None:
            paged_kernel_lens_sum = int(paged_kernel_lens.sum().item())
        kv_indices = torch.empty(
            int(paged_kernel_lens_sum) + width * batch_size,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            kv_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask = self.custom_mask
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, dtype=torch.bool)
        mask = mask.to(device=device, dtype=torch.bool).reshape(-1)
        mask_numel = int(paged_kernel_lens_sum) * width + (width**2) * batch_size
        if mask.numel() < mask_numel:
            mask = torch.cat(
                [
                    mask,
                    torch.full(
                        (mask_numel - mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=0,
            )
        self.custom_mask = mask
        return kv_indices, cum_kv_seq_len, qo_indptr, mask
