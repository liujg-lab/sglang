"""NPU graph runner for STANDALONE_REMOTE tail EXTEND."""

from __future__ import annotations

import logging

import torch

from sglang.srt.speculative.spec_utils import run_npu_graph_update_and_replay
from sglang.srt.speculative.standalone_remote.drafter.sr_tail_extend_graph import (
    SRTailExtendGraphRunner,
)
from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
    read_sr_tail_update_overlap_env,
)
from sglang.srt.speculative.standalone_remote.sr_align import is_device_context_error
from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
    begin_graph_host_sample,
    mark_graph_host_failed,
    measure_call,
    record_graph_host_sample_safely,
)

logger = logging.getLogger(__name__)


def _tail_graph_uses_fia(runner) -> bool:
    backend = getattr(runner, "attn_backend", None)
    return bool(getattr(backend, "use_fia", False))


def make_tail_graph_cpu_update_payload(token_cap: int, *, use_fia: bool):
    """Captured ATB/FIA payload. Replay mutates it in place."""
    token_cap = int(token_cap)
    if use_fia:
        return [{"actual_seq_lengths_kv": [1] * token_cap}]
    return [{"context_lens": torch.ones((token_cap,), dtype=torch.int32, device="cpu")}]


def fill_tail_graph_cpu_update_payload(payload, seq_lens_kv, *, use_fia: bool):
    """Write replay lengths into a captured payload without replacing storage."""
    if use_fia:
        dest = payload[0]["actual_seq_lengths_kv"]
        if torch.is_tensor(seq_lens_kv):
            values = [int(v) for v in seq_lens_kv.detach().tolist()]
        else:
            values = [int(v) for v in seq_lens_kv]
        dest[:] = values
        return payload
    dest = payload[0]["context_lens"]
    dest.fill_(1)
    if torch.is_tensor(seq_lens_kv):
        src = seq_lens_kv.detach()
        count = min(int(dest.numel()), int(src.numel()))
        if count:
            if src.dtype != torch.int32 or src.device.type != "cpu":
                for i in range(count):
                    dest[i] = int(src[i].item())
            else:
                dest[:count].copy_(src[:count])
        return payload
    values = list(seq_lens_kv)
    count = min(int(dest.numel()), len(values))
    for i in range(count):
        dest[i] = int(values[i])
    return payload


def tail_graph_cpu_update_payload(seq_lens_kv, *, use_fia: bool):
    """FIA updates actual_seq_lengths_kv lists; ATB/MHA updates context_lens tensors."""
    if torch.is_tensor(seq_lens_kv):
        n = int(seq_lens_kv.numel())
    else:
        seq_lens_kv = list(seq_lens_kv)
        n = len(seq_lens_kv)
    payload = make_tail_graph_cpu_update_payload(n, use_fia=use_fia)
    return fill_tail_graph_cpu_update_payload(payload, seq_lens_kv, use_fia=use_fia)


class SRTailExtendNpuGraphRunner(SRTailExtendGraphRunner):
    def __init__(self, drafter) -> None:
        self.update_payloads = {}
        self._npu_sr_tail_update_overlap = False
        self._npu_graph_device_id = None
        self._logged_sr_tail_overlap_submit = False
        self._npu_sr_tail_update_overlap_requested = read_sr_tail_update_overlap_env()
        super().__init__(drafter)
        self._maybe_enable_sr_tail_update_overlap()

    def _maybe_enable_sr_tail_update_overlap(self) -> None:
        """Enable overlap only after tail graphs and a device id exist."""
        requested = bool(getattr(self, "_npu_sr_tail_update_overlap_requested", False))
        graphs_ok = bool(getattr(self, "graphs", None)) and not getattr(
            self, "disabled_reason", None
        )
        reason = None
        if requested and graphs_ok:
            try:
                device_id = int(torch.npu.current_device())
            except Exception as exc:
                if is_device_context_error(exc):
                    raise
                reason = f"current_device failed: {exc}"
            else:
                self._npu_graph_device_id = device_id
                self._npu_sr_tail_update_overlap = True
        elif requested and getattr(self, "disabled_reason", None):
            reason = self.disabled_reason
        elif requested:
            reason = "no graphs"
        if not requested:
            return
        impl = "fia" if _tail_graph_uses_fia(self.model_runner) else "atb"
        extra = f" reason={reason}" if reason else ""
        logger.info(
            "NPU SR tail update/replay overlap requested=%s effective=%s "
            "implementation=%s device=%s%s",
            True,
            bool(self._npu_sr_tail_update_overlap),
            impl,
            self._npu_graph_device_id,
            extra,
        )

    def _tail_graph_host_metrics(self):
        drafter = getattr(self, "drafter", None)
        scheduler = getattr(drafter, "scheduler", None)
        metrics = getattr(scheduler, "_sr_round_metrics", None)
        tail = getattr(getattr(scheduler, "sr_tree_drafter", None), "tail_graph_runner", None)
        if (
            metrics is None
            or tail is not self
            or getattr(metrics, "role", None) != "Draft"
            or not getattr(metrics, "active", False)
        ):
            return None
        return metrics

    def _create_graph(self):
        return torch.npu.NPUGraph()

    def _capture_context(self, graph, pool, stream):
        return torch.npu.graph(
            graph, pool=pool, stream=stream, auto_dispatch_capture=True
        )

    def _device_synchronize(self):
        torch.npu.synchronize()

    def _capture_bucket(self, bucket) -> None:
        super()._capture_bucket(bucket)
        if bucket not in self.graphs:
            return
        self.update_payloads[bucket] = make_tail_graph_cpu_update_payload(
            bucket[1], use_fia=_tail_graph_uses_fia(self.model_runner)
        )

    def _replay_graph(self, graph, seq_lens_kv, bucket=None):
        use_fia = _tail_graph_uses_fia(self.model_runner)
        payload = self.update_payloads.get(bucket) if bucket is not None else None
        if payload is None:
            if torch.is_tensor(seq_lens_kv):
                n = int(seq_lens_kv.numel())
            else:
                n = len(list(seq_lens_kv))
            payload = make_tail_graph_cpu_update_payload(n, use_fia=use_fia)
            if bucket is not None:
                self.update_payloads[bucket] = payload
        overlap = bool(getattr(self, "_npu_sr_tail_update_overlap", False))
        device_id = getattr(self, "_npu_graph_device_id", None)
        metrics = None
        if hasattr(self, "_tail_graph_host_metrics"):
            metrics = self._tail_graph_host_metrics()
        sample = None
        if metrics is not None:
            try:
                sample = begin_graph_host_sample(
                    {
                        "graph_phase": "tail_extend",
                        "round_id": int(metrics.rounds) + 1,
                        "graph_key": bucket,
                        "implementation": "fia" if use_fia else "atb",
                        "raw_bs": None,
                        "capture_bs": None if bucket is None else bucket[0],
                        "kv_bucket": None if bucket is None else bucket[1],
                        "overlap": overlap,
                    }
                )
            except Exception:
                sample = None

        def _fill():
            fill_tail_graph_cpu_update_payload(payload, seq_lens_kv, use_fia=use_fia)

        def update():
            if overlap:
                torch.npu.set_device(device_id)
            measure_call(
                sample, "update_call", lambda: graph.update(cpu_update_input=payload)
            )

        def replay():
            measure_call(sample, "replay_call", graph.replay)

        def _submit():
            if getattr(self, "_npu_sr_tail_update_overlap_requested", False) and not getattr(
                self, "_logged_sr_tail_overlap_submit", False
            ):
                self._logged_sr_tail_overlap_submit = True
                logger.info(
                    "NPU SR tail update/replay submit effective=%s implementation=%s "
                    "device=%s overlap=%s",
                    overlap,
                    "fia" if use_fia else "atb",
                    device_id,
                    overlap,
                )
            run_npu_graph_update_and_replay(update, replay, overlap=overlap)

        def _prepare():
            measure_call(sample, "payload_fill", _fill)
            measure_call(sample, "submit_envelope", _submit)

        try:
            measure_call(sample, "prepare_submit", _prepare)
        except BaseException:
            mark_graph_host_failed(sample)
            raise
        finally:
            record_graph_host_sample_safely(metrics, sample)
