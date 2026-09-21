"""NPU device sibling-walk greedy tree verify.

Matches ``verify_tree_greedy_ref`` and CUDA ``VerifyTreeGreedy``. The Python
wrapper only inspects tensor metadata (shape, dtype, device index, contiguity).
It does not read scalar device values, copy tensors to host, or synchronize.
"""

from __future__ import annotations

from typing import Optional

import torch

_INT_DTYPES = (torch.int32, torch.int64)
_INPUT_NAMES = (
    "predicts",
    "accept_index",
    "accept_token_num",
    "candidates",
    "retrive_index",
    "retrive_next_token",
    "retrive_next_sibling",
    "target_predict",
)

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    _TRITON_IMPORT_ERROR = exc
    triton = None
    tl = None
else:
    _TRITON_IMPORT_ERROR = None

_verify_tree_greedy_kernel = None


def npu_greedy_triton_status() -> tuple[str, Optional[str]]:
    """Return ``("ok", None)`` or ``("missing", reason)`` for optional Triton."""
    if _TRITON_IMPORT_ERROR is None:
        return "ok", None
    name = getattr(_TRITON_IMPORT_ERROR, "name", None) or ""
    if name in ("", "triton") or str(name).startswith("triton"):
        return "missing", f"triton not installed: {_TRITON_IMPORT_ERROR}"
    raise _TRITON_IMPORT_ERROR


def _device_key(tensor: torch.Tensor):
    return (tensor.device.type, tensor.device.index)


def inspect_greedy_npu_contract(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> tuple[str, Optional[str]]:
    """Validate greedy verify tensors without reading device values.

    Returns:
        ``("ok", None)``: launch the device kernel (after ``BS=0`` early return).
        ``("cpu", reason)``: host tensors; caller may use the CPU reference.
        ``("unsupported_layout", reason)``: legal but first-release unsupported
        (non-contiguous). Caller must use the CPU reference *before* submit.

    Raises:
        TypeError / ValueError: mixed devices, illegal shape, or illegal dtype.
    """
    named = {
        "predicts": predicts,
        "accept_index": accept_index,
        "accept_token_num": accept_token_num,
        "candidates": candidates,
        "retrive_index": retrive_index,
        "retrive_next_token": retrive_next_token,
        "retrive_next_sibling": retrive_next_sibling,
        "target_predict": target_predict,
    }
    for name in _INPUT_NAMES:
        tensor = named[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
        if tensor.dtype not in _INT_DTYPES:
            raise ValueError(
                f"{name} dtype must be int32 or int64, got {tensor.dtype}"
            )

    keys = [_device_key(named[name]) for name in _INPUT_NAMES]
    if len(set(keys)) != 1:
        raise ValueError(f"greedy verify tensors are on mixed devices: {keys}")

    if candidates.dim() != 2:
        raise ValueError(f"candidates must be 2D [BS, W], got {tuple(candidates.shape)}")
    bs, width = candidates.shape
    if retrive_index.shape != (bs, width):
        raise ValueError(
            f"retrive_index shape {tuple(retrive_index.shape)} != candidates {(bs, width)}"
        )
    if retrive_next_token.shape != (bs, width):
        raise ValueError("retrive_next_token shape mismatch")
    if retrive_next_sibling.shape != (bs, width):
        raise ValueError("retrive_next_sibling shape mismatch")
    if accept_index.dim() != 2 or accept_index.shape[0] != bs:
        raise ValueError(
            f"accept_index must be [BS, L], got {tuple(accept_index.shape)}"
        )
    path_cap = accept_index.shape[1]
    if accept_token_num.dim() != 1 or accept_token_num.shape[0] != bs:
        raise ValueError(
            f"accept_token_num must be [BS], got {tuple(accept_token_num.shape)}"
        )
    if predicts.dim() != 1:
        raise ValueError(f"predicts must be 1D, got {tuple(predicts.shape)}")
    if target_predict.numel() < bs * width:
        raise ValueError(
            f"target_predict numel {target_predict.numel()} < BS*W {bs * width}"
        )

    if bs > 0:
        if width < 1 or path_cap < 1:
            raise ValueError(
                f"non-empty batch requires W>=1 and L>=1, got W={width} L={path_cap}"
            )
        if predicts.numel() < 1:
            raise ValueError("predicts capacity must be at least 1 for a non-empty batch")

    device = predicts.device
    if device.type == "cpu":
        return "cpu", "cpu tensors"
    if device.type != "npu":
        raise ValueError(f"NPU greedy kernel expected npu tensors, got {device}")

    if not all(named[name].is_contiguous() for name in _INPUT_NAMES):
        return "unsupported_layout", "noncontiguous layout"

    return "ok", None


if tl is not None:

    @triton.jit
    def _verify_tree_greedy_kernel(
        predicts_ptr,
        accept_index_ptr,
        accept_token_num_ptr,
        candidates_ptr,
        retrive_index_ptr,
        retrive_next_token_ptr,
        retrive_next_sibling_ptr,
        target_predict_ptr,
        predicts_numel,
        WIDTH: tl.constexpr,
        PATH_CAP: tl.constexpr,
    ):
        bx = tl.program_id(0)
        row = bx * WIDTH
        path_row = bx * PATH_CAP
        neg_one = tl.full((), -1, tl.int64)

        for slot in range(PATH_CAP):
            tl.store(accept_index_ptr + path_row + slot, -1)

        root_valid = WIDTH > 0
        last_accepted = tl.load(
            retrive_index_ptr + row, mask=root_valid, other=-1
        ).to(tl.int64)
        tl.store(
            accept_index_ptr + path_row,
            last_accepted,
            mask=root_valid,
        )

        num_accepted = tl.full((), 0, tl.int32)
        cur_index = tl.full((), 0, tl.int64)
        active = tl.where(root_valid, 1, 0).to(tl.int32)

        for _depth in range(1, PATH_CAP):
            parent_ok = (active != 0) & (cur_index >= 0) & (cur_index < WIDTH)
            child = tl.load(
                retrive_next_token_ptr + row + cur_index,
                mask=parent_ok,
                other=-1,
            ).to(tl.int64)
            cur_index = tl.where(parent_ok, child, neg_one)
            matched = tl.full((), 0, tl.int32)

            for _sib in range(WIDTH):
                walk = (active != 0) & (matched == 0)
                node_ok = walk & (cur_index >= 0) & (cur_index < WIDTH)
                draft_index = tl.load(
                    retrive_index_ptr + row + cur_index,
                    mask=node_ok,
                    other=-1,
                ).to(tl.int64)
                draft_token = tl.load(
                    candidates_ptr + row + cur_index,
                    mask=node_ok,
                    other=0,
                ).to(tl.int64)
                pred_ok = (
                    node_ok & (last_accepted >= 0) & (last_accepted < predicts_numel)
                )
                target_token = tl.load(
                    target_predict_ptr + last_accepted,
                    mask=pred_ok,
                    other=0,
                ).to(tl.int64)
                hit = node_ok & (draft_token == target_token)
                tl.store(
                    predicts_ptr + last_accepted,
                    draft_token,
                    mask=hit & pred_ok,
                )
                num_accepted = tl.where(hit, num_accepted + 1, num_accepted)
                acc_ok = hit & (num_accepted > 0) & (num_accepted < PATH_CAP)
                tl.store(
                    accept_index_ptr + path_row + num_accepted,
                    draft_index,
                    mask=acc_ok,
                )
                last_accepted = tl.where(hit, draft_index, last_accepted)
                matched = tl.where(hit, 1, matched)
                sibling = tl.load(
                    retrive_next_sibling_ptr + row + cur_index,
                    mask=node_ok & (~hit),
                    other=-1,
                ).to(tl.int64)
                new_cur = cur_index
                new_cur = tl.where(node_ok & (~hit), sibling, new_cur)
                new_cur = tl.where(walk & (~node_ok), neg_one, new_cur)
                cur_index = tl.where(walk, new_cur, cur_index)

            active = tl.where((active != 0) & (matched != 0), 1, 0)

        tl.store(accept_token_num_ptr + bx, num_accepted)
        bonus_ok = (last_accepted >= 0) & (last_accepted < predicts_numel)
        bonus = tl.load(target_predict_ptr + last_accepted, mask=bonus_ok, other=0)
        tl.store(predicts_ptr + last_accepted, bonus, mask=bonus_ok)


def verify_tree_greedy_npu(
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
    """Launch the NPU sibling-walk greedy kernel. In-place; returns the same objects."""
    del topk
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
    if decision != "ok":
        raise ValueError(
            f"verify_tree_greedy_npu requires a launchable NPU contract, got "
            f"{decision}: {reason}"
        )
    status, dep_reason = npu_greedy_triton_status()
    if status != "ok":
        raise ImportError(dep_reason)
    if _verify_tree_greedy_kernel is None:
        raise ImportError("triton is required for verify_tree_greedy_npu")

    bs, width = candidates.shape
    if bs == 0:
        return predicts, accept_index, accept_token_num

    path_cap = accept_index.shape[1]
    target_flat = target_predict.reshape(-1)
    _verify_tree_greedy_kernel[(bs,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_flat,
        predicts.numel(),
        WIDTH=width,
        PATH_CAP=path_cap,
        num_warps=1,
    )
    return predicts, accept_index, accept_token_num
