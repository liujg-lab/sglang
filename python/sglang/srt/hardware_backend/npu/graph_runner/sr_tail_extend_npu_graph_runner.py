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


def tail_graph_cpu_update_payload(seq_lens_kv, *, use_fia: bool):
    """FIA updates actual_seq_lengths_kv lists; ATB/MHA updates context_lens tensors."""
    if use_fia:
        return [{"actual_seq_lengths_kv": list(seq_lens_kv)}]
    if torch.is_tensor(seq_lens_kv):
        lens = seq_lens_kv.detach().to(dtype=torch.int32, device="cpu")
    else:
        lens = torch.tensor(list(seq_lens_kv), dtype=torch.int32, device="cpu")
    return [{"context_lens": lens}]


class SRTailExtendNpuGraphRunner(SRTailExtendGraphRunner):
    def _create_graph(self):
        return torch.npu.NPUGraph()

    def _capture_context(self, graph, pool, stream):
        return torch.npu.graph(
            graph, pool=pool, stream=stream, auto_dispatch_capture=True
        )

    def _device_synchronize(self):
        torch.npu.synchronize()

    def _replay_graph(self, graph, seq_lens_kv):
        payload = tail_graph_cpu_update_payload(
            seq_lens_kv, use_fia=_tail_graph_uses_fia(self.model_runner)
        )

        def update():
            graph.update(cpu_update_input=payload)

        run_npu_graph_update_and_replay(update, graph.replay)
