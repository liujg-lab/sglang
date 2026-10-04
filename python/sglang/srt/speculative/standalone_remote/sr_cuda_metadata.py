"""Stream-owned eager multi-step index storage. No model/attention imports."""

import torch

from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
    SRTransferUnresolved,
)


class SRDraftDecodeMetadataWorkspace:
    def __init__(self, device, steps, max_context):
        self.device = torch.device(device)
        self.steps, self.max_context = steps, max_context
        self.indices = self.indptr = None
        self.rows = 0
        self.retired = []
        self.unresolved = False
        self.used = False
        self.holds = None

    def reserve(self, rows):
        if self.unresolved:
            raise SRTransferUnresolved("Draft eager metadata completion is unresolved")
        rows = int(rows)
        if rows < 1:
            raise ValueError("Draft eager metadata requires nonempty rows")
        self._collect()
        if rows <= self.rows:
            return self.indices, self.indptr
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Draft metadata cannot grow during graph capture")
        cap = 1 << (rows - 1).bit_length()
        indices = torch.empty(
            (self.steps, cap * self.max_context), dtype=torch.int64, device=self.device
        )
        indptr = torch.zeros(
            (self.steps, cap + 1), dtype=torch.int32, device=self.device
        )
        if self.used and self.device.type != "cpu":
            try:
                event = torch.get_device_module(self.device.type).Event()
                event.record()
            except BaseException as exc:
                self.unresolved = True
                self.holds = (self.holds, indices, indptr)
                raise SRTransferUnresolved(
                    "Draft metadata retirement event failed"
                ) from exc
            self.retired.append((event, self.indices, self.indptr, self.holds))
        self.indices, self.indptr, self.rows = indices, indptr, cap
        return indices, indptr

    def _collect(self):
        try:
            self.retired = [item for item in self.retired if not item[0].query()]
        except BaseException as exc:
            self.unresolved = True
            raise SRTransferUnresolved(
                "Draft metadata retirement query failed"
            ) from exc

    def submit(self, inputs, function):
        if self.unresolved:
            raise SRTransferUnresolved("Draft metadata completion is unresolved")
        self.holds = (inputs, self.indices, self.indptr)
        self.used = True
        try:
            return function()
        except BaseException as exc:
            self.unresolved = True
            if isinstance(exc, SRTransferUnresolved):
                raise
            raise SRTransferUnresolved(
                "Draft metadata generation failed after submission"
            ) from exc
