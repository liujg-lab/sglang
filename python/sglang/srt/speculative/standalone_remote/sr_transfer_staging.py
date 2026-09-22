"""Reusable host/device staging for SR accept and tree-result transfers.

Pin support is cached per backend. Event support is cached per device.
A failed ``record()`` after a copy has been issued is not treated as
"events are unsupported".
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang.srt.speculative.standalone_remote.sr_align import is_device_context_error

logger = logging.getLogger(__name__)

STATE_FREE = "free"
STATE_IN_FLIGHT = "in_flight"
STATE_CONSUMING = "consuming"
STATE_UNRESOLVED = "unresolved"

_pin_by_backend = {}
_event_by_device = {}
_pin_logged = set()
_event_logged = set()


class SRTransferUnresolved(RuntimeError):
    """A transfer may still be running and completion cannot be confirmed."""


def _backend_name(device) -> str:
    if device is None:
        return "cpu"
    return torch.device(device).type


def device_cache_key(device) -> Tuple[str, Optional[int]]:
    """Cache key for one accelerator. CPU stays separate from every device."""
    if device is None:
        return ("cpu", None)
    dev = torch.device(device)
    if dev.type == "cpu":
        return ("cpu", None)
    index = dev.index
    if index is None:
        try:
            index = int(torch.get_device_module(dev.type).current_device())
        except Exception as exc:
            if is_device_context_error(exc):
                raise
            index = None
    return (dev.type, index)


def pin_supported(device) -> bool:
    """Whether this backend can allocate a pinned CPU buffer. Cached per backend."""
    backend = _backend_name(device)
    if backend == "cpu":
        return False
    cached = _pin_by_backend.get(backend)
    if cached is not None:
        return cached[0]
    reason = None
    try:
        probe = torch.empty(1, dtype=torch.int64, device="cpu", pin_memory=True)
        ok = bool(probe.is_pinned())
        if not ok:
            reason = "allocation is not pinned"
    except Exception as exc:
        if is_device_context_error(exc):
            raise
        ok = False
        reason = str(exc)
    _pin_by_backend[backend] = (ok, reason)
    if not ok and backend not in _pin_logged:
        _pin_logged.add(backend)
        logger.info(
            "[SR] pinned staging unavailable backend=%s reason=%s", backend, reason
        )
    return ok


def event_supported(device) -> bool:
    """Whether ``record()`` works before any transfer. Cached per device.

    This probe does not submit a copy. A later ``record()`` failure after
    ``copy_()`` is not cached as "events unsupported".
    """
    key = device_cache_key(device)
    if key[0] == "cpu":
        return False
    cached = _event_by_device.get(key)
    if cached is not None:
        return cached[0]
    reason = None
    try:
        event = torch.get_device_module(key[0]).Event()
        event.record()
        ok = True
    except Exception as exc:
        if is_device_context_error(exc):
            raise
        ok = False
        reason = str(exc)
    _event_by_device[key] = (ok, reason)
    if not ok and key not in _event_logged:
        _event_logged.add(key)
        logger.info("[SR] staging events unavailable device=%s reason=%s", key, reason)
    return ok


def alloc_host(shape, dtype, device) -> Tuple[torch.Tensor, bool]:
    """CPU buffer. Pinned only when this backend's probe succeeded."""
    if pin_supported(device):
        buf = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        if bool(buf.is_pinned()):
            return buf, True
    return torch.empty(shape, dtype=dtype, device="cpu"), False


def _accel_device(dst: torch.Tensor, src: torch.Tensor):
    if src.device.type != "cpu":
        return src.device
    if dst.device.type != "cpu":
        return dst.device
    return None


def _use_async(dst: torch.Tensor, src: torch.Tensor) -> bool:
    accel = _accel_device(dst, src)
    if accel is None or dst.device.type == src.device.type:
        return False
    cpu = dst if dst.device.type == "cpu" else src
    if not bool(cpu.is_pinned()):
        return False
    return event_supported(accel)


def submit_copy(dst: torch.Tensor, src: torch.Tensor):
    """Copy ``src`` into ``dst`` on the current stream.

    Returns an event when the copy is async. ``None`` means the copy finished
    before this function returned. Raises ``SRTransferUnresolved`` when a copy
    may already have been queued and completion cannot be confirmed.
    """
    if int(dst.numel()) != int(src.numel()):
        raise RuntimeError("staging copy length mismatch")
    if int(src.numel()) == 0:
        return None
    try:
        async_copy = _use_async(dst, src)
    except Exception as exc:
        if is_device_context_error(exc):
            raise
        async_copy = False
    if not async_copy:
        try:
            dst.copy_(src, non_blocking=False)
        except Exception as exc:
            if is_device_context_error(exc):
                raise
            raise SRTransferUnresolved("staging copy failed") from exc
        return None
    try:
        dst.copy_(src, non_blocking=True)
    except Exception as exc:
        raise SRTransferUnresolved("async staging copy failed") from exc
    try:
        accel = _accel_device(dst, src)
        event = torch.get_device_module(accel.type).Event()
        event.record()
    except Exception as exc:
        raise SRTransferUnresolved("event record failed after staging copy") from exc
    return event


def wait_event(event) -> None:
    """Wait for one submitted transfer. A failure stays unresolved."""
    if event is None:
        return
    try:
        event.synchronize()
    except Exception as exc:
        raise SRTransferUnresolved("staging wait failed") from exc


def cross_device_d2h(src: torch.Tensor, dst: torch.Tensor) -> bool:
    return src.device.type != "cpu" and dst.device.type == "cpu"


class ResultSlot:
    """One tree-result staging slot. ``wait`` does not make it reusable."""

    def __init__(self) -> None:
        self.state = STATE_FREE
        self.capacity = 0
        self.device_buf: Optional[torch.Tensor] = None
        self.host_buf: Optional[torch.Tensor] = None
        self.pinned = False
        self.event = None
        self.src_hold = None
        self.segments = []
        self.used = 0

    def grow(self, needed: int, device) -> bool:
        """Grow only while FREE. Returns whether storage was replaced."""
        if self.state != STATE_FREE:
            raise RuntimeError("refuse to grow a tree staging slot that is not free")
        needed = int(needed)
        if self.capacity >= needed:
            return False
        cap = 1
        while cap < max(needed, 1):
            cap *= 2
        self.device_buf = torch.empty((cap,), dtype=torch.int64, device=device)
        self.host_buf, self.pinned = alloc_host((cap,), torch.int64, device)
        self.capacity = cap
        self.event = None
        self.src_hold = None
        self.segments = []
        self.used = 0
        return True
