from __future__ import annotations

import copy
import hashlib
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from sglang.srt.speculative.standalone_remote.sr_mm_payload import (
    SRMMPayload,
    serialize_mm_item,
)

RAW_IMAGE_CACHE_NAMESPACE = "sr-image-v3"


def _grid_rows(grid):
    if grid is None:
        return None
    grid = torch.as_tensor(grid)
    if grid.ndim == 1:
        grid = grid.unsqueeze(0)
    if (
        grid.ndim != 2
        or grid.shape[1] != 3
        or not grid.shape[0]
        or grid.dtype not in (torch.int32, torch.int64)
        or bool((grid <= 0).any())
    ):
        return None
    return grid.to(device="cpu", dtype=torch.int64)


@dataclass
class SRImagePayload:
    meta: SRMMPayload
    images: List[bytes]

    @classmethod
    def from_req(cls, req) -> Optional[SRImagePayload]:
        images = getattr(req, "source_image_bytes", None)
        if not images:
            return None
        mm = req.multimodal_inputs
        metas = []
        previous_end = -1
        for item in mm.mm_items:
            if (
                not item.is_image()
                or item.is_precomputed_embedding()
                or item.precomputed_embeddings is not None
                or item.hash is None
                or item.pad_value is None
            ):
                return None
            grid = _grid_rows(item.model_specific_data.get("image_grid_thw"))
            offsets = item.offsets
            if grid is None or not offsets or len(offsets) != len(grid):
                return None
            for span in offsets:
                if not isinstance(span, (tuple, list)) or len(span) != 2:
                    return None
                start, end = span
                if not (previous_end < start <= end < len(req.origin_input_ids)):
                    return None
                if any(
                    token != item.pad_value
                    for token in req.origin_input_ids[start : end + 1]
                ):
                    return None
                previous_end = end
            bare = copy.copy(item)
            bare.feature = None
            bare.precomputed_embeddings = None
            bare.model_specific_data = {"image_grid_thw": grid}
            metas.append(serialize_mm_item(bare))
        if not metas or sum(len(item["offsets"]) for item in metas) != len(images):
            return None
        return cls(
            SRMMPayload(
                rid=req.rid,
                mm_items=metas,
                im_token_id=mm.im_token_id,
                im_start_id=mm.im_start_id,
                im_end_id=mm.im_end_id,
                slice_start_id=mm.slice_start_id,
                slice_end_id=mm.slice_end_id,
            ),
            images,
        )

    def to_multipart(self) -> Tuple[Dict[str, Any], List[Any]]:
        meta, buffers = self.meta.to_multipart()
        meta["_sr_image_offset"] = len(buffers)
        meta["_sr_image_count"] = len(self.images)
        return meta, [*buffers, *(memoryview(image) for image in self.images)]

    @classmethod
    def from_multipart(cls, meta, frames):
        offset = meta["_sr_image_offset"]
        count = meta["_sr_image_count"]
        if (
            type(offset) is not int
            or type(count) is not int
            or offset < 0
            or count < 1
            or offset + count != len(frames)
        ):
            raise ValueError("invalid image frame range")
        if any(
            item.get("feature") is not None or item.get("precomputed_embeddings") is not None
            for item in meta["mm_items"]
        ):
            raise ValueError("image payload contains tensors instead of encoded images")
        images = [bytes(memoryview(getattr(f, "buffer", f))) for f in frames[offset:]]
        if not all(images):
            raise ValueError("empty image frame")
        return cls(SRMMPayload.from_multipart(meta, frames[:offset]), images)

    def to_pickleable(self):
        return self.meta.to_pickleable(), self.images

    @classmethod
    def from_pickleable(cls, data):
        meta, images = data
        return cls(SRMMPayload.from_pickleable(meta), images)

    def materialize(self, processor):
        processor = getattr(processor, "image_processor", None)
        patch = getattr(processor, "patch_size", None)
        merge = getattr(processor, "merge_size", None)
        if not callable(processor) or not isinstance(patch, int) or patch <= 0:
            raise ValueError("unsupported image processor")
        if not isinstance(merge, int) or merge <= 0:
            raise ValueError("unsupported image merge size")
        cursor = 0
        mm = self.meta.to_multimodal_inputs()
        items = mm.mm_items
        if not items:
            raise ValueError("missing image metadata")
        for item, meta in zip(items, self.meta.mm_items):
            grid = _grid_rows(item.model_specific_data.get("image_grid_thw"))
            if (
                not item.is_image()
                or item.feature is not None
                or item.precomputed_embeddings is not None
                or grid is None
                or len(item.offsets or ()) != len(grid)
                or cursor + len(grid) > len(self.images)
            ):
                raise ValueError("invalid image metadata")
            images = []
            key = hashlib.blake2b(digest_size=8, person=b"sr-image-v3")
            key.update(repr(grid.tolist()).encode())
            for (t, h, w), (start, end) in zip(grid.tolist(), item.offsets):
                if h % merge or w % merge or end - start + 1 != t * h * w // merge**2:
                    raise ValueError("image grid does not match its token span")
                raw = self.images[cursor]
                cursor += 1
                key.update(len(raw).to_bytes(8, "little"))
                key.update(raw)
                with Image.open(io.BytesIO(raw)) as image:
                    rgb = image.convert("RGB")
                size = (w * patch, h * patch)
                if rgb.size != size:
                    rgb = rgb.resize(size, Image.Resampling.BICUBIC)
                images.append(rgb)
            out = processor(images=images, do_resize=False, return_tensors="pt")
            got = _grid_rows(out["image_grid_thw"])
            if got is None or not torch.equal(got, grid):
                raise ValueError("image grid mismatch")
            item.feature = meta["feature"] = torch.as_tensor(out["pixel_values"])
            item.hash = meta["hash"] = int.from_bytes(key.digest(), "little")
        if cursor != len(self.images):
            raise ValueError("unused image frames")
        mm.sr_image_cache_key = RAW_IMAGE_CACHE_NAMESPACE + ":" + ".".join(
            str(item.hash) for item in items
        )
        return self.meta
