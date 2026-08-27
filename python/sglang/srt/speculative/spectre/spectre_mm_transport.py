"""[SPECTRE-VL] Out-of-band multimodal payload transport for SPECTRE Target -> Draft.

The C++ msgpack channel is reserved for control + token ids. Vision tensors
travel on a dedicated pyzmq PUB/SUB socket so every Draft TP rank can receive
the same payload without a gloo pixel broadcast:

- Same node (ipc): CPU tensors are wrapped in POSIX SHM pointers.
- Cross node (tcp): metadata pickle + raw tensor frames (multipart).

Pad values and hashes computed on the Target are preserved so Draft never
re-pads input_ids. Draft locates vision tokens by scanning input_ids for
item.pad_value, so re-padding on Draft would desync embeddings.
"""

from __future__ import annotations

import functools
import logging
import os
import pickle
import time
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

_TENSOR_PLACEHOLDER_PREFIX = "__spectre_mm_tensor_"


def _to_cpu_contiguous_tensor(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, ShmPointerMMData):
        # [SPECTRE-VL] 只读拷贝，绝不 materialize()：那会 unlink 别人拥有的 shm 段。
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


def _serialize_tensor(value: Any, use_shm: bool) -> Any:
    tensor = _to_cpu_contiguous_tensor(value)
    if tensor is None:
        return value
    if use_shm:
        return ShmPointerMMData(tensor)
    return tensor


def _deserialize_tensor(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, ShmPointerMMData):
        return value.materialize()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return value


def _collect_shm_pointers(obj: Any) -> List[ShmPointerMMData]:
    # [SPECTRE-VL] to_pickleable(use_shm=True) 创建的段需要在发送失败时手动回收。
    found: List[ShmPointerMMData] = []

    def visit(value: Any) -> None:
        if isinstance(value, ShmPointerMMData):
            found.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                visit(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                visit(v)

    visit(obj)
    return found


def _unlink_shm_pointers(pointers: List[ShmPointerMMData]) -> None:
    """[SPECTRE-VL] 创建端的 ShmPointerMMData.__del__ 只 close 不 unlink（unlink 是接收端
    materialize() 的职责），所以消息没送出去时段会永久留在 /dev/shm，必须显式回收。"""
    from multiprocessing import shared_memory

    for pointer in pointers:
        name = getattr(pointer, "shm_name", None)
        if not name:
            continue
        try:
            handle = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.warning("[SPECTRE][MM] Cannot open shm %s to unlink: %s", name, e)
            continue
        try:
            handle.close()
            handle.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("[SPECTRE][MM] Failed to unlink shm %s: %s", name, e)


def _walk_and_transform(obj: Any, fn) -> Any:
    if isinstance(obj, dict):
        return {k: _walk_and_transform(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_transform(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_walk_and_transform(v, fn) for v in obj)
    return fn(obj)


def serialize_mm_item(item: MultimodalDataItem, use_shm: bool = False) -> Dict[str, Any]:
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
        # [SPECTRE-VL] 显式带 pad_value/hash：接收端不重算，兼容 SGLANG_MM_SKIP_COMPUTE_HASH。
        "pad_value": item.pad_value,
        "offsets": item.offsets,
        "format": fmt_name,
        "feature": _serialize_tensor(item.feature, use_shm),
        "precomputed_embeddings": _serialize_tensor(
            item.precomputed_embeddings, use_shm
        ),
        "model_specific_data": _walk_and_transform(
            item.model_specific_data, lambda v: _serialize_tensor(v, use_shm)
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
    item = MultimodalDataItem(
        modality=modality,
        hash=data.get("hash"),
        pad_value=data.get("pad_value"),
        offsets=data.get("offsets"),
        format=fmt,
        feature=_deserialize_tensor(data.get("feature")),
        precomputed_embeddings=_deserialize_tensor(data.get("precomputed_embeddings")),
        model_specific_data=model_specific,
    )
    return item


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
    """Copy origin_input_ids with pad_value slots restored to image/video/audio ids.

    Target pad_input_ids replaces vision tokens with hash pad_value so embeddings
    can scatter by pad_value. get_rope_index still looks for image_token_id.
    Draft recomputes M-RoPE on this restored copy; KV/embedding keep scanning
    pad_value on the original sequence. Length is unchanged; re-prefill suffixes
    stay as generated token ids.
    """
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
    # [SPECTRE-VL] re-prefill 后序列变长，旧 mrope / delta / cache 全部失效。
    if mm is None:
        return
    mm.mrope_positions = None
    mm.mrope_position_delta = None
    mm.mrope_position_delta_repeated_cache = None


def release_mm_resources(mm: Optional[MultimodalInputs]) -> None:
    # [SPECTRE-VL] 同时清 pixel 和预热 embedding；Draft req 不会自然 finished()。
    if mm is None:
        return
    mm.release_features()
    for item in mm.mm_items:
        item.precomputed_embeddings = None


@dataclass
class SpectreMMPayload:
    # [SPECTRE-VL] padded_input_ids 让 Draft 在 DRAFT_REQUEST 到达前就能预跑 ViT，并做一致性校验。
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
    prewarm_failed: bool = field(default=False, repr=False)
    # [SPECTRE-VL] 已预热的 payload 不再进入预热候选，否则每轮都被重扫一遍。
    prewarmed: bool = field(default=False, repr=False)
    created_at: float = field(default_factory=time.time, repr=False)

    @classmethod
    def from_req(cls, req) -> "SpectreMMPayload":
        mm: MultimodalInputs = req.multimodal_inputs
        return cls(
            rid=req.rid,
            padded_input_ids=list(req.origin_input_ids),
            mm_items=[serialize_mm_item(item, use_shm=False) for item in mm.mm_items],
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

    def to_pickleable(self, use_shm: bool) -> Dict[str, Any]:
        # [SPECTRE-VL] 同机：tensor 换成 POSIX SHM 指针，避免 pyzmq pickle 大拷贝。
        mm = self.to_multimodal_inputs()
        items = [serialize_mm_item(item, use_shm=use_shm) for item in mm.mm_items]
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
    def from_pickleable(
        cls, data: Dict[str, Any], materialize_shm: bool = True
    ) -> "SpectreMMPayload":
        # [SPECTRE-VL] PUB/SUB + SHM：先让每个 rank shm_open，barrier 后再 materialize。
        if materialize_shm:
            items = [
                _walk_and_transform(d, _deserialize_tensor)
                for d in data.get("mm_items", [])
            ]
        else:
            items = list(data.get("mm_items") or [])
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
        if materialize_shm:
            payload.to_multimodal_inputs()
        return payload

    def to_multipart(self) -> Tuple[Dict[str, Any], List[memoryview]]:
        # [SPECTRE-VL] 跨机：metadata pickle + raw uint8 frames，不走 C++ 单帧 std::string。
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

        pickled = self.to_pickleable(use_shm=False)
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
        cls, meta: Dict[str, Any], frames: List[bytes]
    ) -> "SpectreMMPayload":
        def _frame_to_uint8(raw: Any) -> torch.Tensor:
            # [SPECTRE-VL] 只 clone 一次。只读 memoryview 被 frombuffer 拒绝时退 numpy copy。
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


class SpectreMMSender:
    def __init__(self, addr: str, use_shm: bool, bind: bool = True):
        import zmq

        self.addr = addr
        self.use_shm = use_shm
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 128)
        if bind:
            self._socket.bind(addr)
        else:
            self._socket.connect(addr)
        logger.info("[SPECTRE][MM] Sender %s on %s (shm=%s)", "bound" if bind else "connected", addr, use_shm)

    def send(self, payload: SpectreMMPayload) -> str:
        import zmq

        # [SPECTRE-VL] shm 段在序列化时就已创建；一旦发送失败必须自己 unlink，否则泄漏 /dev/shm。
        shm_pointers: List[ShmPointerMMData] = []
        try:
            if self.use_shm:
                pickled = payload.to_pickleable(use_shm=True)
                shm_pointers = _collect_shm_pointers(pickled)
                self._socket.send_pyobj(pickled, flags=zmq.NOBLOCK)
            else:
                meta, buffers = payload.to_multipart()
                # [SPECTRE-VL] copy=False 直接送 memoryview，避免再包一层 bytes。
                frames = [pickle.dumps(meta), *buffers]
                self._socket.send_multipart(frames, flags=zmq.NOBLOCK, copy=False)
            return "sent"
        except zmq.Again:
            _unlink_shm_pointers(shm_pointers)
            logger.warning("[SPECTRE][MM] Sender queue full, dropping payload rid=%s", payload.rid)
            return "queue_full"
        except Exception as e:
            _unlink_shm_pointers(shm_pointers)
            logger.warning("[SPECTRE][MM] Failed to send payload rid=%s: %s", payload.rid, e)
            return "error"

    def close(self) -> None:
        try:
            self._socket.close(linger=0)
        except Exception:
            pass


class SpectreMMReceiver:
    def __init__(self, addr: str, use_shm: bool, bind: bool = False):
        import zmq

        self.addr = addr
        self.use_shm = use_shm
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 128)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        if bind:
            self._socket.bind(addr)
        else:
            self._socket.connect(addr)
        logger.info(
            "[SPECTRE][MM] Receiver %s on %s (shm=%s)",
            "bound" if bind else "connected",
            addr,
            use_shm,
        )

    def recv_all(
        self, defer_shm_materialize: bool = False
    ) -> Tuple[List[SpectreMMPayload], int]:
        import zmq

        payloads: List[SpectreMMPayload] = []
        n_errors = 0
        while True:
            try:
                if self.use_shm:
                    data = self._socket.recv_pyobj(flags=zmq.NOBLOCK)
                    payloads.append(
                        SpectreMMPayload.from_pickleable(
                            data, materialize_shm=not defer_shm_materialize
                        )
                    )
                else:
                    frames = self._socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
                    meta_raw = frames[0]
                    meta = pickle.loads(
                        meta_raw if isinstance(meta_raw, (bytes, bytearray)) else bytes(meta_raw)
                    )
                    payloads.append(SpectreMMPayload.from_multipart(meta, frames[1:]))
            except zmq.Again:
                break
            except Exception as e:
                logger.warning("[SPECTRE][MM] Failed to recv payload: %s", e)
                n_errors += 1
                break
        return payloads, n_errors

    def close(self) -> None:
        try:
            self._socket.close(linger=0)
        except Exception:
            pass


def clear_spectre_mm_env_cache() -> None:
    """Reset cached SPECTRE_MM_* env lookups (unit tests patch os.environ)."""
    spectre_mm_prewarm_max.cache_clear()
    spectre_mm_prewarm_bytes.cache_clear()
    spectre_mm_stale_s.cache_clear()
    spectre_mm_wait_ms.cache_clear()


@functools.lru_cache(maxsize=1)
def spectre_mm_prewarm_max() -> int:
    # [SPECTRE-VL] 单轮最多预热 N 个请求，避免 ViT 抢占正在进行的 draft decode。
    try:
        return max(0, int(os.environ.get("SPECTRE_MM_PREWARM_MAX", "2")))
    except ValueError:
        return 2


@functools.lru_cache(maxsize=1)
def spectre_mm_prewarm_bytes() -> int:
    """[SPECTRE-VL] pending mm 预热 embedding 的 GPU 水位。0 表示不按字节限流。"""
    try:
        return max(0, int(os.environ.get("SPECTRE_MM_PREWARM_BYTES", str(512 * 1024 * 1024))))
    except ValueError:
        return 512 * 1024 * 1024


@functools.lru_cache(maxsize=1)
def spectre_mm_stale_s() -> float:
    """[SPECTRE-VL] 无 draft state 的孤儿 payload 超时（秒）。FINISH/ABORT 已即时回收。"""
    try:
        return max(0.0, float(os.environ.get("SPECTRE_MM_STALE_S", "5")))
    except ValueError:
        return 5.0


def _tensor_gpu_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor) and value.is_cuda:
        return int(value.numel() * value.element_size())
    return 0


def _tensor_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    return 0


def payload_gpu_bytes(payload: Any) -> int:
    mm = getattr(payload, "mm_inputs", None)
    if mm is None:
        return 0
    total = 0
    for item in getattr(mm, "mm_items", None) or []:
        total += _tensor_gpu_bytes(getattr(item, "precomputed_embeddings", None))
        total += _tensor_gpu_bytes(getattr(item, "feature", None))
    return total


def payload_resident_bytes(payload: Any) -> int:
    """CPU+GPU bytes of pending feature / embedding tensors."""
    mm = getattr(payload, "mm_inputs", None)
    if mm is None:
        to_mm = getattr(payload, "to_multimodal_inputs", None)
        if callable(to_mm):
            try:
                mm = to_mm()
            except Exception:
                mm = None
    if mm is None:
        return 0
    total = 0
    for item in getattr(mm, "mm_items", None) or []:
        total += _tensor_nbytes(getattr(item, "precomputed_embeddings", None))
        total += _tensor_nbytes(getattr(item, "feature", None))
    return total


@functools.lru_cache(maxsize=1)
def spectre_mm_wait_ms() -> float:
    """[SPECTRE-VL] 等 mm payload 的墙钟预算（毫秒）。

    默认取 Target recv 超时的一半：等更久没有意义，Target 那一步早已放弃。
    跨机传输大图时可以通过 SPECTRE_MM_WAIT_MS 调大。
    """
    raw = os.environ.get("SPECTRE_MM_WAIT_MS")
    if raw is not None:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    try:
        recv_ms = float(os.environ.get("SPECTRE_RECV_TIMEOUT_MS", "5000"))
    except ValueError:
        recv_ms = 5000.0
    return max(0.0, recv_ms / 2.0)
