"""NPU graph runner for STANDALONE_REMOTE tail EXTEND."""

from __future__ import annotations

import torch

from sglang.srt.speculative.spec_utils import run_npu_graph_update_and_replay
from sglang.srt.speculative.standalone_remote.drafter.sr_tail_extend_graph import (
    SRTailExtendGraphRunner,
)


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
        super().__init__(drafter)

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
        fill_tail_graph_cpu_update_payload(payload, seq_lens_kv, use_fia=use_fia)

        def update():
            graph.update(cpu_update_input=payload)

        run_npu_graph_update_and_replay(update, graph.replay)
