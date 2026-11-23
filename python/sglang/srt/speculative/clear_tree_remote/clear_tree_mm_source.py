"""Send the original image bytes to the Draft instead of pixel tensors.

Target side: ``source_payload(req)`` turns a request whose tokenizer kept the
encoded images (``req.source_image_bytes``) into a payload that carries those
bytes plus the Target's item metadata (offsets, pad_value, hash, grid). The
pixel tensor is not sent.

Draft side: ``materialize(payload, processor)`` decodes the bytes, resizes each
image to the Target's grid, runs the Draft's own image processor without its
resize step, and checks that the grid it gets equals the Target's. On a match
it returns an ``SRMMPayload`` with the features filled in; otherwise None, and
the caller rejects that request so the Target serves it without drafts.

``SRMMPayload`` is only constructed here, never changed.
"""

from __future__ import annotations

import copy
import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.speculative.standalone_remote.sr_mm_payload import (
    SRMMPayload,
    serialize_mm_item,
)

logger = logging.getLogger(__name__)

SRC_KEY = "_ct_src"
SRC_OFFSET_KEY = "_ct_src_offset"
SRC_COUNT_KEY = "_ct_src_count"


def _grid_rows(grid: Any) -> Optional[torch.Tensor]:
    if grid is None:
        return None
    if not isinstance(grid, torch.Tensor):
        try:
            grid = torch.as_tensor(grid)
        except Exception:
            return None
    if grid.ndim == 1:
        grid = grid.unsqueeze(0)
    if grid.ndim != 2 or grid.shape[-1] != 3:
        return None
    return grid.to(torch.long)


def _item_grid(item: Any) -> Optional[torch.Tensor]:
    if isinstance(item, dict):
        data = item.get("model_specific_data") or {}
    else:
        data = getattr(item, "model_specific_data", None) or {}
    return _grid_rows(data.get("image_grid_thw"))


def _is_image_item(item: Any) -> bool:
    if isinstance(item, dict):
        return str(item.get("modality", "")).upper() == "IMAGE"
    is_image = getattr(item, "is_image", None)
    return bool(is_image()) if callable(is_image) else False


@dataclass
class ClearImageSourcePayload:
    """Item metadata without pixels, plus the encoded images in request order."""

    meta: SRMMPayload
    images: List[bytes] = field(default_factory=list)

    @property
    def rid(self) -> str:
        return self.meta.rid

    @property
    def mm_items(self) -> List[Dict[str, Any]]:
        return self.meta.mm_items

    def to_multipart(self) -> Tuple[Dict[str, Any], List[Any]]:
        meta, buffers = self.meta.to_multipart()
        meta[SRC_KEY] = True
        meta[SRC_OFFSET_KEY] = len(buffers)
        meta[SRC_COUNT_KEY] = len(self.images)
        buffers = list(buffers) + [memoryview(raw) for raw in self.images]
        return meta, buffers

    @classmethod
    def from_multipart(
        cls, meta: Dict[str, Any], frames: List[Any]
    ) -> "ClearImageSourcePayload":
        offset = int(meta[SRC_OFFSET_KEY])
        count = int(meta[SRC_COUNT_KEY])
        plain = {
            k: v
            for k, v in meta.items()
            if k not in (SRC_KEY, SRC_OFFSET_KEY, SRC_COUNT_KEY)
        }
        inner = SRMMPayload.from_multipart(plain, list(frames[:offset]))
        images = [
            bytes(memoryview(getattr(frame, "buffer", frame)))
            for frame in frames[offset : offset + count]
        ]
        return cls(meta=inner, images=images)


    def to_pickleable(self) -> Dict[str, Any]:
        """For the Draft's tp broadcast, like ``SRMMPayload.to_pickleable``."""
        return {
            SRC_KEY: True,
            "meta": self.meta.to_pickleable(),
            "images": list(self.images),
        }


def is_source_meta(meta: Optional[Dict[str, Any]]) -> bool:
    return bool(meta) and bool(meta.get(SRC_KEY))


def payload_from_pickleable(data: Dict[str, Any]):
    """Inverse of either payload's ``to_pickleable``."""
    if isinstance(data, dict) and data.get(SRC_KEY):
        return ClearImageSourcePayload(
            meta=SRMMPayload.from_pickleable(data["meta"]),
            images=[bytes(raw) for raw in data.get("images") or []],
        )
    return SRMMPayload.from_pickleable(data)


def source_payload(req) -> Optional[ClearImageSourcePayload]:
    """Target: the image-source payload for ``req``, or None to send tensors.

    Needs kept image bytes, image-only items, and one grid row per image.
    """
    images = getattr(req, "source_image_bytes", None)
    if not images:
        return None
    mm = getattr(req, "multimodal_inputs", None)
    items = list(getattr(mm, "mm_items", None) or [])
    if not items:
        return None
    rows = 0
    for item in items:
        if not _is_image_item(item):
            return None
        grid = _item_grid(item)
        if grid is None:
            return None
        rows += int(grid.shape[0])
    if rows != len(images):
        return None
    metas = []
    for item in items:
        bare = copy.copy(item)
        bare.feature = None
        bare.precomputed_embeddings = None
        metas.append(serialize_mm_item(bare))
    inner = SRMMPayload(
        rid=req.rid,
        mm_items=metas,
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
    return ClearImageSourcePayload(meta=inner, images=list(images))


def _image_processor(processor):
    return getattr(processor, "image_processor", None) or processor


def _decode(raw: bytes, width: int, height: int):
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as img:
        rgb = img.convert("RGB")
    if rgb.size != (width, height):
        rgb = rgb.resize((width, height), Image.BICUBIC)
    return rgb


def materialize(
    payload: ClearImageSourcePayload, processor
) -> Tuple[Optional[SRMMPayload], str]:
    """Draft: features from the Draft's own processor at the Target's grid.

    Returns ``(payload, "ok")`` or ``(None, reason)`` where reason is
    ``mismatch`` or ``error``.
    """
    proc = _image_processor(processor)
    if proc is None:
        return None, "error"
    patch = getattr(proc, "patch_size", None)
    if not patch:
        return None, "error"
    patch = int(patch)
    try:
        mm = payload.meta.to_multimodal_inputs()
        items = list(mm.mm_items)
        if len(items) != len(payload.meta.mm_items):
            return None, "error"
        cursor = 0
        for item, item_meta in zip(items, payload.meta.mm_items):
            grid = _item_grid(item)
            if grid is None:
                return None, "error"
            count = int(grid.shape[0])
            raws = payload.images[cursor : cursor + count]
            cursor += count
            if len(raws) != count:
                return None, "error"
            pics = [
                _decode(raw, int(row[2]) * patch, int(row[1]) * patch)
                for raw, row in zip(raws, grid.tolist())
            ]
            out = proc(images=pics, do_resize=False, return_tensors="pt")
            got = _grid_rows(out["image_grid_thw"])
            if got is None or not torch.equal(got, grid):
                logger.warning(
                    "[CLEAR] image source grid mismatch rid=%s target=%s draft=%s",
                    payload.rid,
                    grid.tolist(),
                    None if got is None else got.tolist(),
                )
                return None, "mismatch"
            feature = out["pixel_values"]
            if not isinstance(feature, torch.Tensor):
                feature = torch.as_tensor(feature)
            item.feature = feature
            item_meta["feature"] = feature
        if cursor != len(payload.images):
            return None, "error"
    except Exception:
        logger.warning(
            "[CLEAR] image source decode failed rid=%s", payload.rid, exc_info=True
        )
        return None, "error"
    return payload.meta, "ok"
