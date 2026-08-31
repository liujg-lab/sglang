"""Multimodal payload for STANDALONE_REMOTE Target -> Draft.

Vision tensors travel on the same ZMQ RPC as control (multipart), not a
sidecar PUB/SUB. Target already padded input_ids; Draft reuses that sequence
and pad_value and must never call pad_input_ids again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.managers.mm_utils import ShmPointerMMData
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalInputs,
)

logger = logging.getLogger(__name__)

_TENSOR_PLACEHOLDER_PREFIX = "__sr_mm_tensor_"


def _to_cpu_contiguous_tensor(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, ShmPointerMMData):
        value = value.tensor.clone()
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not isinstance(value, torch.Tensor):
        return None
    tensor = value.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _serialize_tensor(value: Any) -> Any:
    tensor = _to_cpu_contiguous_tensor(value)
    if tensor is None:
        return value
    return tensor


def _deserialize_tensor(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, ShmPointerMMData):
        return value.materialize()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return value


def _walk_and_transform(obj: Any, fn) -> Any:
    if isinstance(obj, dict):
        # Placeholder dicts from to_multipart are leaves; recursing into
        # shape/dtype would skip restore() and leave image_grid_thw as a dict.
        if _TENSOR_PLACEHOLDER_PREFIX in obj:
            return fn(obj)
        return {k: _walk_and_transform(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_transform(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_walk_and_transform(v, fn) for v in obj)
    return fn(obj)


def serialize_mm_item(item: MultimodalDataItem) -> Dict[str, Any]:
    modality = item.modality
    if isinstance(modality, Modality):
        modality_name = modality.name
    else:
        modality_name = str(modality)
    fmt = item.format
    if isinstance(fmt, MultimodalInputFormat):
        fmt_name = fmt.name
    else:
        fmt_name = str(fmt)
    return {
        "modality": modality_name,
        "hash": item.hash,
        "pad_value": item.pad_value,
        "offsets": item.offsets,
        "format": fmt_name,
        "feature": _serialize_tensor(item.feature),
        "precomputed_embeddings": _serialize_tensor(item.precomputed_embeddings),
        "model_specific_data": _walk_and_transform(
            item.model_specific_data, _serialize_tensor
        ),
    }


def deserialize_mm_item(data: Dict[str, Any]) -> MultimodalDataItem:
    modality = data.get("modality", "IMAGE")
    if isinstance(modality, str):
        modality = Modality.from_str(modality)
    fmt = data.get("format", "NORMAL")
    if isinstance(fmt, str):
        try:
            fmt = MultimodalInputFormat[fmt]
        except KeyError:
            fmt = MultimodalInputFormat.NORMAL
    model_specific = _walk_and_transform(
        data.get("model_specific_data") or {}, _deserialize_tensor
    )
    return MultimodalDataItem(
        modality=modality,
        hash=data.get("hash"),
        pad_value=data.get("pad_value"),
        offsets=data.get("offsets"),
        format=fmt,
        feature=_deserialize_tensor(data.get("feature")),
        precomputed_embeddings=_deserialize_tensor(data.get("precomputed_embeddings")),
        model_specific_data=model_specific,
    )


def items_to_multimodal_inputs(
    items: List[MultimodalDataItem],
    *,
    im_token_id: Optional[int] = None,
    im_start_id: Optional[int] = None,
    im_end_id: Optional[int] = None,
    video_token_id: Optional[int] = None,
    audio_token_id: Optional[int] = None,
    audio_start_id: Optional[int] = None,
    audio_end_id: Optional[int] = None,
    slice_start_id: Optional[int] = None,
    slice_end_id: Optional[int] = None,
) -> MultimodalInputs:
    mm = MultimodalInputs(mm_items=items)
    mm.im_token_id = im_token_id
    mm.im_start_id = im_start_id
    mm.im_end_id = im_end_id
    mm.video_token_id = video_token_id
    mm.audio_token_id = audio_token_id
    mm.audio_start_id = audio_start_id
    mm.audio_end_id = audio_end_id
    mm.slice_start_id = slice_start_id
    mm.slice_end_id = slice_end_id
    return mm


def ids_for_mrope_compute(
    origin_input_ids: Optional[List[int]],
    mm: Optional[MultimodalInputs],
) -> List[int]:
    ids = list(origin_input_ids or [])
    if mm is None or not ids:
        return ids
    n = len(ids)
    for item in mm.mm_items or []:
        if item.is_image():
            token_id = mm.im_token_id
        elif item.is_video():
            token_id = mm.video_token_id
        elif item.is_audio():
            token_id = mm.audio_token_id
        else:
            continue
        if token_id is None or not item.offsets:
            continue
        for offset in item.offsets:
            if offset is None or len(offset) < 2:
                continue
            start = max(0, int(offset[0]))
            end = min(n - 1, int(offset[1]))
            if start > end:
                continue
            for i in range(start, end + 1):
                ids[i] = token_id
    return ids


def reset_mm_mrope(mm: Optional[MultimodalInputs]) -> None:
    if mm is None:
        return
    mm.mrope_positions = None
    mm.mrope_position_delta = None
    mm.mrope_position_delta_repeated_cache = None


def release_mm_resources(mm: Optional[MultimodalInputs]) -> None:
    if mm is None:
        return
    mm.release_features()
    for item in mm.mm_items:
        item.precomputed_embeddings = None


@dataclass
class SRMMPayload:
    rid: str
    padded_input_ids: List[int]
    mm_items: List[Dict[str, Any]]
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    video_token_id: Optional[int] = None
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None
    mm_inputs: Optional[MultimodalInputs] = field(default=None, repr=False)
    attached: bool = field(default=False, repr=False)

    @classmethod
    def from_req(cls, req) -> "SRMMPayload":
        mm: MultimodalInputs = req.multimodal_inputs
        return cls(
            rid=req.rid,
            padded_input_ids=list(req.origin_input_ids),
            mm_items=[serialize_mm_item(item) for item in mm.mm_items],
            im_token_id=mm.im_token_id,
            im_start_id=mm.im_start_id,
            im_end_id=mm.im_end_id,
            video_token_id=mm.video_token_id,
            audio_token_id=mm.audio_token_id,
            audio_start_id=mm.audio_start_id,
            audio_end_id=mm.audio_end_id,
            slice_start_id=mm.slice_start_id,
            slice_end_id=mm.slice_end_id,
        )

    def to_multimodal_inputs(self) -> MultimodalInputs:
        if self.mm_inputs is not None:
            return self.mm_inputs
        items = [deserialize_mm_item(d) for d in self.mm_items]
        self.mm_inputs = items_to_multimodal_inputs(
            items,
            im_token_id=self.im_token_id,
            im_start_id=self.im_start_id,
            im_end_id=self.im_end_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
            audio_start_id=self.audio_start_id,
            audio_end_id=self.audio_end_id,
            slice_start_id=self.slice_start_id,
            slice_end_id=self.slice_end_id,
        )
        return self.mm_inputs

    def to_pickleable(self) -> Dict[str, Any]:
        mm = self.to_multimodal_inputs()
        items = [serialize_mm_item(item) for item in mm.mm_items]
        return {
            "rid": self.rid,
            "padded_input_ids": self.padded_input_ids,
            "mm_items": items,
            "im_token_id": self.im_token_id,
            "im_start_id": self.im_start_id,
            "im_end_id": self.im_end_id,
            "video_token_id": self.video_token_id,
            "audio_token_id": self.audio_token_id,
            "audio_start_id": self.audio_start_id,
            "audio_end_id": self.audio_end_id,
            "slice_start_id": self.slice_start_id,
            "slice_end_id": self.slice_end_id,
        }

    @classmethod
    def from_pickleable(cls, data: Dict[str, Any]) -> "SRMMPayload":
        items = [
            _walk_and_transform(d, _deserialize_tensor)
            for d in data.get("mm_items", [])
        ]
        payload = cls(
            rid=data["rid"],
            padded_input_ids=list(data.get("padded_input_ids") or []),
            mm_items=items,
            im_token_id=data.get("im_token_id"),
            im_start_id=data.get("im_start_id"),
            im_end_id=data.get("im_end_id"),
            video_token_id=data.get("video_token_id"),
            audio_token_id=data.get("audio_token_id"),
            audio_start_id=data.get("audio_start_id"),
            audio_end_id=data.get("audio_end_id"),
            slice_start_id=data.get("slice_start_id"),
            slice_end_id=data.get("slice_end_id"),
        )
        payload.to_multimodal_inputs()
        return payload

    def to_multipart(self) -> Tuple[Dict[str, Any], List[memoryview]]:
        buffers: List[memoryview] = []
        counter = {"n": 0}

        def replace_tensor(value: Any):
            tensor = _to_cpu_contiguous_tensor(value)
            if tensor is None:
                if isinstance(value, (torch.Tensor, np.ndarray, ShmPointerMMData)):
                    return None
                return value
            idx = counter["n"]
            counter["n"] += 1
            raw = tensor.view(torch.uint8).reshape(-1).numpy()
            buffers.append(memoryview(raw))
            return {
                _TENSOR_PLACEHOLDER_PREFIX: idx,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
            }

        pickled = self.to_pickleable()
        meta = {
            "rid": pickled["rid"],
            "padded_input_ids": pickled["padded_input_ids"],
            "mm_items": _walk_and_transform(pickled["mm_items"], replace_tensor),
            "im_token_id": pickled["im_token_id"],
            "im_start_id": pickled["im_start_id"],
            "im_end_id": pickled["im_end_id"],
            "video_token_id": pickled["video_token_id"],
            "audio_token_id": pickled["audio_token_id"],
            "audio_start_id": pickled["audio_start_id"],
            "audio_end_id": pickled["audio_end_id"],
            "slice_start_id": pickled["slice_start_id"],
            "slice_end_id": pickled["slice_end_id"],
        }
        return meta, buffers

    @classmethod
    def from_multipart(
        cls, meta: Dict[str, Any], frames: List[Any]
    ) -> "SRMMPayload":
        def _frame_to_uint8(raw: Any) -> torch.Tensor:
            try:
                return torch.frombuffer(memoryview(raw), dtype=torch.uint8).clone()
            except (TypeError, ValueError, RuntimeError, BufferError):
                arr = np.frombuffer(memoryview(raw), dtype=np.uint8).copy()
                return torch.from_numpy(arr)

        def restore(value: Any):
            if not isinstance(value, dict) or _TENSOR_PLACEHOLDER_PREFIX not in value:
                return value
            idx = value[_TENSOR_PLACEHOLDER_PREFIX]
            dtype = getattr(torch, value["dtype"])
            tensor = _frame_to_uint8(frames[idx])
            tensor = tensor.view(dtype).reshape(value["shape"])
            return tensor

        restored = {
            **meta,
            "mm_items": _walk_and_transform(meta.get("mm_items") or [], restore),
        }
        return cls.from_pickleable(restored)
