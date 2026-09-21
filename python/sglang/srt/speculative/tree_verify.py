"""Portable tree verification (greedy + target_only) for CUDA/NPU/CPU.

CUDA keeps the existing sgl_kernel implementations. The CPU greedy reference
is the correctness baseline for NPU sibling-walk and the fallback when the
NPU device kernel cannot be launched. target_only and tests still use these
portable algorithms so sibling traversal, threshold accept, and relu(q-p)
bonus sampling match the CUDA kernels given the same tensors and coins.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


def check_tree_verify_tensors(
    *,
    predicts: Optional[torch.Tensor] = None,
    accept_index: Optional[torch.Tensor] = None,
    accept_token_num: Optional[torch.Tensor] = None,
    candidates: Optional[torch.Tensor] = None,
    retrive_index: Optional[torch.Tensor] = None,
    retrive_next_token: Optional[torch.Tensor] = None,
    retrive_next_sibling: Optional[torch.Tensor] = None,
    target_predict: Optional[torch.Tensor] = None,
    target_probs: Optional[torch.Tensor] = None,
    draft_probs: Optional[torch.Tensor] = None,
    uniform_samples: Optional[torch.Tensor] = None,
    uniform_samples_for_final_sampling: Optional[torch.Tensor] = None,
    custom_mask: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
) -> None:
    """Validate shapes and devices of tree-verify tensors."""

    def _require(name: str, tensor: torch.Tensor, dims: int) -> None:
        if tensor.dim() != dims:
            raise ValueError(f"{name} must be {dims}D, got shape {tuple(tensor.shape)}")

    if candidates is None:
        return
    _require("candidates", candidates, 2)
    bs, num_draft = candidates.shape
    device = candidates.device
    if retrive_index is not None:
        _require("retrive_index", retrive_index, 2)
        if retrive_index.shape != (bs, num_draft):
            raise ValueError(
                f"retrive_index shape {tuple(retrive_index.shape)} "
                f"!= candidates {(bs, num_draft)}"
            )
        if retrive_index.device != device:
            raise ValueError("retrive_index device mismatch")
    if retrive_next_token is not None:
        _require("retrive_next_token", retrive_next_token, 2)
        if retrive_next_token.shape != (bs, num_draft):
            raise ValueError("retrive_next_token shape mismatch")
        if retrive_next_token.device != device:
            raise ValueError("retrive_next_token device mismatch")
    if retrive_next_sibling is not None:
        _require("retrive_next_sibling", retrive_next_sibling, 2)
        if retrive_next_sibling.shape != (bs, num_draft):
            raise ValueError("retrive_next_sibling shape mismatch")
        if retrive_next_sibling.device != device:
            raise ValueError("retrive_next_sibling device mismatch")
    if accept_index is not None:
        _require("accept_index", accept_index, 2)
        if accept_index.shape[0] != bs:
            raise ValueError("accept_index batch mismatch")
    if accept_token_num is not None:
        _require("accept_token_num", accept_token_num, 1)
        if accept_token_num.shape[0] != bs:
            raise ValueError("accept_token_num batch mismatch")
    if target_predict is not None:
        if target_predict.reshape(-1).numel() < bs * num_draft:
            raise ValueError("target_predict is smaller than bs * num_draft")
        if target_predict.device != device:
            raise ValueError("target_predict device mismatch")
    if target_probs is not None:
        _require("target_probs", target_probs, 3)
        if target_probs.shape[:2] != (bs, num_draft):
            raise ValueError("target_probs shape mismatch")
        if target_probs.device != device:
            raise ValueError("target_probs device mismatch")
    if draft_probs is not None and target_probs is not None:
        if draft_probs.shape != target_probs.shape:
            raise ValueError("draft_probs shape mismatch")
        if draft_probs.device != device:
            raise ValueError("draft_probs device mismatch")
    if uniform_samples is not None:
        if uniform_samples.numel() != bs * num_draft:
            raise ValueError("uniform_samples shape mismatch")
        if uniform_samples.device != device:
            raise ValueError("uniform_samples device mismatch")
    if uniform_samples_for_final_sampling is not None:
        if uniform_samples_for_final_sampling.numel() != bs:
            raise ValueError("uniform_samples_for_final_sampling shape mismatch")
        if uniform_samples_for_final_sampling.device != device:
            raise ValueError("uniform_samples_for_final_sampling device mismatch")
    if positions is not None and positions.numel() != bs * num_draft:
        raise ValueError(
            f"positions numel {positions.numel()} != bs * num_draft {bs * num_draft}"
        )
    if custom_mask is not None and custom_mask.dim() != 1:
        raise ValueError(
            f"custom_mask must be 1D flattened FULL_MASK, got {tuple(custom_mask.shape)}"
        )


def torch_top_k_renorm_prob(probs: torch.Tensor, top_k) -> torch.Tensor:
    """Match sgl_kernel.top_k_renorm_prob with a torch-only implementation."""
    batch_size, vocab_size = probs.shape
    if not torch.is_tensor(top_k):
        k_val = min(max(int(top_k), 1), vocab_size)
        _, topk_indices = torch.topk(probs, k_val, dim=1, largest=True)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, topk_indices, 1.0)
        masked = probs * mask
        return masked / (masked.sum(dim=1, keepdim=True) + 1e-10)

    k_vec = top_k.reshape(-1)
    if k_vec.numel() == 1 or bool(torch.all(k_vec == k_vec[0])):
        k_val = min(max(int(k_vec[0].item()), 1), vocab_size)
        _, topk_indices = torch.topk(probs, k_val, dim=1, largest=True)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, topk_indices, 1.0)
        masked = probs * mask
        return masked / (masked.sum(dim=1, keepdim=True) + 1e-10)

    out = torch.zeros_like(probs)
    for i in range(batch_size):
        k_val = min(max(int(k_vec[i].item()), 1), vocab_size)
        _, topk_indices = torch.topk(probs[i], k_val, largest=True)
        mask = torch.zeros_like(probs[i])
        mask[topk_indices] = 1.0
        masked = probs[i] * mask
        out[i] = masked / (masked.sum() + 1e-10)
    return out


def torch_top_p_renorm_prob(
    probs: torch.Tensor, top_p, eps: float = 1e-5
) -> torch.Tensor:
    """Match sgl_kernel.top_p_renorm_prob with a torch-only implementation."""
    batch_size, vocab_size = probs.shape
    if not torch.is_tensor(top_p):
        p_val = float(top_p)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=1)
        cutoff_mask = cumsum_probs <= p_val
        cutoff_mask[:, 0] = True
        mask = torch.zeros_like(probs)
        mask.scatter_(1, sorted_indices, cutoff_mask.to(probs.dtype))
        masked = probs * mask
        return masked / (masked.sum(dim=1, keepdim=True) + eps)

    p_vec = top_p.reshape(-1)
    out = torch.zeros_like(probs)
    for i in range(batch_size):
        p_val = float(p_vec[i].item())
        sorted_prob, indices = torch.sort(probs[i], descending=False)
        cdf = torch.cumsum(sorted_prob, dim=-1)
        mask = torch.zeros(vocab_size, dtype=probs.dtype, device=probs.device)
        mask.scatter_(0, indices, (cdf >= (1 - p_val) - eps).to(probs.dtype))
        masked = probs[i] * mask
        out[i] = masked / (masked.sum() + eps)
    return out


def _to_long_list2(t: torch.Tensor) -> list:
    return t.detach().to("cpu").long().tolist()


def verify_tree_greedy_ref(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sibling-walk greedy verify. Matches CUDA ``VerifyTreeGreedy``."""
    check_tree_verify_tensors(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )
    bs, num_draft_tokens = candidates.shape
    num_speculative_tokens = accept_index.shape[1]
    cand = _to_long_list2(candidates)
    ridx = _to_long_list2(retrive_index)
    nxt = _to_long_list2(retrive_next_token)
    sib = _to_long_list2(retrive_next_sibling)
    tgt_flat = target_predict.reshape(-1).detach().to("cpu").long().tolist()

    predict_list = predicts.detach().to("cpu").long().tolist()
    accept_rows = [[-1] * num_speculative_tokens for _ in range(bs)]
    accept_counts = [0] * bs

    for bx in range(bs):
        last_accepted = int(ridx[bx][0])
        accept_rows[bx][0] = last_accepted
        num_accepted = 0
        cur_index = 0
        for _j in range(1, num_speculative_tokens):
            cur_index = int(nxt[bx][cur_index])
            while cur_index != -1:
                draft_index = int(ridx[bx][cur_index])
                draft_token_id = int(cand[bx][cur_index])
                target_token_id = int(tgt_flat[last_accepted])
                if draft_token_id == target_token_id:
                    predict_list[last_accepted] = target_token_id
                    num_accepted += 1
                    accept_rows[bx][num_accepted] = draft_index
                    last_accepted = draft_index
                    break
                cur_index = int(sib[bx][cur_index])
            if cur_index == -1:
                break
        accept_counts[bx] = num_accepted
        predict_list[last_accepted] = int(tgt_flat[last_accepted])

    device = predicts.device
    predicts.copy_(
        torch.tensor(predict_list, dtype=predicts.dtype, device=device)
    )
    accept_index.copy_(
        torch.tensor(accept_rows, dtype=accept_index.dtype, device=device)
    )
    accept_token_num.copy_(
        torch.tensor(accept_counts, dtype=accept_token_num.dtype, device=device)
    )
    return predicts, accept_index, accept_token_num


def _sample_relu_q_minus_p(
    target_row: torch.Tensor,
    draft_row: torch.Tensor,
    coin: float,
    subtract_draft: bool,
) -> int:
    q = target_row.to(dtype=torch.float32)
    if subtract_draft:
        relu = torch.clamp(q - draft_row.to(dtype=torch.float32), min=0)
    else:
        relu = torch.clamp(q, min=0)
    total = float(relu.sum().item())
    vocab = relu.numel()
    if total <= 0.0:
        return vocab - 1
    u = float(coin) * total
    cdf = torch.cumsum(relu, dim=0)
    idx = int(torch.searchsorted(cdf, torch.tensor(u, device=cdf.device), right=True).item())
    if idx >= vocab:
        return vocab - 1
    return idx


def tree_speculative_sampling_target_only_ref(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match CUDA ``TreeSpeculativeSamplingTargetOnly`` given the same coins."""
    check_tree_verify_tensors(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
    )
    del deterministic  # CUDA path always uses deterministic=True from eagle_info
    bs, num_draft_tokens, vocab = target_probs.shape
    num_speculative_tokens = accept_index.shape[1]
    capped_threshold_acc = max(float(threshold_acc), 1e-9)
    threshold_single_f = float(threshold_single)

    cand = _to_long_list2(candidates)
    ridx = _to_long_list2(retrive_index)
    nxt = _to_long_list2(retrive_next_token)
    sib = _to_long_list2(retrive_next_sibling)
    coins = uniform_samples.detach().to("cpu").float().tolist()
    coins_final = uniform_samples_for_final_sampling.detach().to("cpu").float().tolist()
    tp = target_probs.detach()
    dp = draft_probs.detach().clone()

    predict_list = predicts.detach().to("cpu").long().tolist()
    accept_rows = [[-1] * num_speculative_tokens for _ in range(bs)]
    accept_counts = [0] * bs

    for bx in range(bs):
        prob_acc = 0.0
        cur_prob_index = 0  # local draft slot whose target row is current
        coin = float(coins[bx][0] if isinstance(coins[bx], list) else coins[bx])
        last_accepted = ridx[bx][0]
        accept_rows[bx][0] = last_accepted
        num_accepted = 0
        cur_index = 0
        for _j in range(1, num_speculative_tokens):
            cur_index = nxt[bx][cur_index]
            while cur_index != -1:
                draft_index = ridx[bx][cur_index]
                draft_token_id = int(cand[bx][cur_index])
                target_prob_single = float(tp[bx, cur_prob_index, draft_token_id].item())
                prob_acc += target_prob_single
                if (
                    coin <= prob_acc / capped_threshold_acc
                    or target_prob_single >= threshold_single_f
                ):
                    prob_acc = 0.0
                    cur_prob_index = cur_index
                    coin = float(coins[bx][cur_index])
                    predict_list[last_accepted] = draft_token_id
                    num_accepted += 1
                    accept_rows[bx][num_accepted] = draft_index
                    last_accepted = draft_index
                    break
                dp[bx, cur_prob_index, draft_token_id] = tp[bx, cur_prob_index, draft_token_id]
                cur_index = sib[bx][cur_index]
            if cur_index == -1:
                break
        accept_counts[bx] = num_accepted
        subtract_draft = num_accepted != (num_speculative_tokens - 1)
        sampled = _sample_relu_q_minus_p(
            tp[bx, cur_prob_index],
            dp[bx, cur_prob_index],
            float(coins_final[bx]),
            subtract_draft=subtract_draft,
        )
        predict_list[last_accepted] = sampled

    device = predicts.device
    predicts.copy_(torch.tensor(predict_list, dtype=predicts.dtype, device=device))
    accept_index.copy_(
        torch.tensor(accept_rows, dtype=accept_index.dtype, device=device)
    )
    accept_token_num.copy_(
        torch.tensor(accept_counts, dtype=accept_token_num.dtype, device=device)
    )
    draft_probs.copy_(dp.to(device=draft_probs.device, dtype=draft_probs.dtype))
    return predicts, accept_index, accept_token_num


TREE_MASK_FULL = 0
TREE_MASK_QLEN_ONLY = 1


def build_tree_kernel_efficient_ref(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int = TREE_MASK_FULL,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU/torch replica of CUDA ``build_tree_efficient`` FULL_MASK writes.

    Matches ``sgl-kernel/csrc/speculative/eagle_utils.cu``: the mask is filled
    True (prefix columns stay True because the kernel never writes them), then
    each row's trailing ``draft_token_num`` columns are rewritten from the
    ancestor chain. Returns
    ``(tree_mask, positions, retrive_index, retrive_next_token, retrive_next_sibling)``.
    """
    if int(tree_mask_mode) != TREE_MASK_FULL:
        raise NotImplementedError(
            f"build_tree_kernel_efficient_ref only implements FULL_MASK, "
            f"got tree_mask_mode={tree_mask_mode}"
        )
    topk = int(topk)
    depth = int(depth)
    draft = int(draft_token_num)
    seq_lens = [int(x) for x in verified_seq_len.detach().reshape(-1).tolist()]
    bs = len(seq_lens)
    expected_parent_cols = topk * (depth - 1) + 1
    if parent_list.dim() == 2 and int(parent_list.shape[1]) != expected_parent_cols:
        raise ValueError(
            f"parent_list width {tuple(parent_list.shape)} != "
            f"[bs, topk*(depth-1)+1]={expected_parent_cols}"
        )
    device = parent_list.device
    parents = _to_long_list2(parent_list)
    selected = _to_long_list2(selected_index)

    mask_numel = sum(seq_lens) * draft + bs * draft * draft
    mask = [True] * mask_numel
    positions = [0] * (bs * draft)
    retrive_index = [[-1] * draft for _ in range(bs)]
    retrive_next_token = [[-1] * draft for _ in range(bs)]
    retrive_next_sibling = [[-1] * draft for _ in range(bs)]

    for bid in range(bs):
        seq_tree_idx = draft * draft * bid
        for i in range(bid):
            seq_tree_idx += seq_lens[i] * draft
        seq_len = seq_lens[bid]
        parent_row = parents[bid] if bid < len(parents) else []
        selected_row = selected[bid] if bid < len(selected) else []

        retrive_index[bid][0] = bid * draft
        for i in range(draft - 1, 0, -1):
            retrive_index[bid][i] = bid * draft + i
            parent_tb_idx = int(selected_row[i - 1]) // topk
            parent_position = 0
            if parent_tb_idx > 0:
                parent_token_idx = int(parent_row[parent_tb_idx])
                parent_position = 0
                while parent_position < draft:
                    if (
                        parent_position < len(selected_row)
                        and int(selected_row[parent_position]) == parent_token_idx
                    ):
                        parent_position += 1
                        break
                    parent_position += 1
            if parent_position == draft:
                continue
            if retrive_next_token[bid][parent_position] == -1:
                retrive_next_token[bid][parent_position] = i
            else:
                origin = retrive_next_token[bid][parent_position]
                retrive_next_token[bid][parent_position] = i
                retrive_next_sibling[bid][i] = origin

        positions[bid * draft] = seq_len
        for tid in range(draft):
            token_tree_idx = seq_tree_idx + (seq_len + draft) * tid + seq_len + 1
            mask[token_tree_idx - 1] = True
            for i in range(draft - 1):
                mask[token_tree_idx + i] = False
            if tid == 0:
                continue
            cur_position = tid - 1
            position = 0
            while True:
                position += 1
                mask[token_tree_idx + cur_position] = True
                parent_tb_idx = int(selected_row[cur_position]) // topk
                if parent_tb_idx == 0:
                    break
                token_idx = int(parent_row[parent_tb_idx])
                cur_position = 0
                while cur_position < draft:
                    if (
                        cur_position < len(selected_row)
                        and int(selected_row[cur_position]) == token_idx
                    ):
                        break
                    cur_position += 1
            positions[bid * draft + tid] = position + seq_len

    tree_mask = torch.tensor(mask, dtype=torch.bool, device=device)
    pos = torch.tensor(positions, dtype=torch.long, device=device)
    ridx = torch.tensor(retrive_index, dtype=torch.long, device=device)
    rnxt = torch.tensor(retrive_next_token, dtype=torch.long, device=device)
    rsib = torch.tensor(retrive_next_sibling, dtype=torch.long, device=device)
    return tree_mask, pos, ridx, rnxt, rsib


def first_full_mask_mismatch(
    got: torch.Tensor,
    ref: torch.Tensor,
    seq_lens: Sequence[int],
    num_draft: int,
):
    """Return ``(batch, row, col, got, ref, got_row, ref_row)`` or None."""
    got_list = got.detach().to("cpu").reshape(-1).bool().tolist()
    ref_list = ref.detach().to("cpu").reshape(-1).bool().tolist()
    offset = 0
    for b, seq_len in enumerate(seq_lens):
        row_len = int(seq_len) + int(num_draft)
        for t in range(int(num_draft)):
            got_row = got_list[offset : offset + row_len]
            ref_row = ref_list[offset : offset + row_len]
            for c, (g, r) in enumerate(zip(got_row, ref_row)):
                if bool(g) != bool(r):
                    return (b, t, c, bool(g), bool(r), got_row, ref_row)
            offset += row_len
    if len(got_list) != len(ref_list):
        return (
            -1,
            -1,
            -1,
            len(got_list),
            len(ref_list),
            got_list[:8],
            ref_list[:8],
        )
    return None
