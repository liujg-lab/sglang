import logging
import math
from enum import IntEnum
from typing import List, Optional

import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
logger = logging.getLogger(__name__)

_logged_greedy_paths: set[tuple[str, Optional[str]]] = set()
_npu_greedy_deps: Optional[tuple[str, Optional[str]]] = None
_last_greedy_verify_path: Optional[str] = None
_last_greedy_verify_reason: Optional[str] = None

if _is_cuda or _is_hip:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrive_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrive_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    if _is_npu:
        torch.ops.npu.build_tree_kernel_efficient(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )


def greedy_verify_path_info() -> tuple[Optional[str], Optional[str]]:
    """Last greedy dispatch path and fallback reason for this process."""
    return _last_greedy_verify_path, _last_greedy_verify_reason


def npu_greedy_optional_deps() -> tuple[str, Optional[str]]:
    """Cache whether optional Triton is importable. Does not cache layout."""
    global _npu_greedy_deps
    if _npu_greedy_deps is not None:
        return _npu_greedy_deps
    from sglang.srt.speculative.tree_verify_npu import npu_greedy_triton_status

    _npu_greedy_deps = npu_greedy_triton_status()
    return _npu_greedy_deps


def _log_greedy_verify_path(path: str, reason: Optional[str] = None) -> None:
    global _last_greedy_verify_path, _last_greedy_verify_reason
    _last_greedy_verify_path = path
    _last_greedy_verify_reason = reason
    key = (path, reason)
    if key in _logged_greedy_paths:
        return
    _logged_greedy_paths.add(key)
    if reason:
        logger.info("Speculative greedy verify path: %s reason=%s", path, reason)
    else:
        logger.info("Speculative greedy verify path: %s", path)


def _run_greedy_reference(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
    topk,
    reason: str,
):
    from sglang.srt.speculative.tree_verify import verify_tree_greedy_ref

    _log_greedy_verify_path("cpu_reference", reason)
    verify_tree_greedy_ref(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
        topk=topk,
    )
    return predicts, accept_index, accept_token_num


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    if _is_cuda or _is_hip:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )
        _log_greedy_verify_path("cuda_kernel")
        return predicts, accept_index, accept_token_num

    if _is_npu:
        from sglang.srt.speculative.tree_verify_npu import (
            inspect_greedy_npu_contract,
            verify_tree_greedy_npu,
        )

        decision, reason = inspect_greedy_npu_contract(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            target_predict,
        )
        if decision == "cpu":
            return _run_greedy_reference(
                predicts,
                accept_index,
                accept_token_num,
                candidates,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                target_predict,
                topk,
                reason or "cpu tensors",
            )
        if decision == "unsupported_layout":
            return _run_greedy_reference(
                predicts,
                accept_index,
                accept_token_num,
                candidates,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                target_predict,
                topk,
                reason or "unsupported layout",
            )
        dep_status, dep_reason = npu_greedy_optional_deps()
        if dep_status != "ok":
            return _run_greedy_reference(
                predicts,
                accept_index,
                accept_token_num,
                candidates,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                target_predict,
                topk,
                dep_reason or "missing optional dependency",
            )
        _log_greedy_verify_path("npu_kernel")
        verify_tree_greedy_npu(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
            topk=topk,
        )
        return predicts, accept_index, accept_token_num

    return _run_greedy_reference(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        topk,
        "non-npu host path",
    )
