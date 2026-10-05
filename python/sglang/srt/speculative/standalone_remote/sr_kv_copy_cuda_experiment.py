"""Contiguous CUDA MHA/GQA tree remap. Gather every group, then scatter.

prepare_kv_move selects this workspace only for tree_eager and tree_graph.
Other domains stay on torch_out. Capacity is fixed: a larger eager request
builds a new workspace and retains the old one until its stream completes.
"""

import torch

from sglang.srt.speculative.standalone_remote import sr_kv_copy as kv


def continuous_groups(layout):
    """Validate the entire layout before allocation; group static geometry."""
    if layout.kind != "token":
        raise kv.UnsupportedKVMoveLayout("experiment requires token-major KV")
    groups = {}
    for i, b in enumerate(layout.buffers):
        t = b.tensor
        if (
            b.axis != 0
            or t.ndim != 3
            or t.requires_grad
            or t.stride(2) != 1
            or t.stride(1) != t.shape[2]
            or t.stride(0) < t.shape[1] * t.shape[2]
            or t.element_size() not in (1, 2, 4, 8)
        ):
            raise kv.UnsupportedKVMoveLayout(
                "experiment requires disjoint contiguous [H,D] token rows"
            )
        key = (t.dtype, t.shape[1] * t.shape[2], tuple(t.stride()))
        groups.setdefault(key, []).append(i)
    return tuple((key, tuple(indices)) for key, indices in groups.items())


class CudaKVExperiment(kv.KVMoveWorkspace):
    """Fixed-capacity scratch with the shared submission/poisoning contract."""

    def __init__(self, pool, capacity, *, graph=False):
        if int(capacity) <= 0:
            raise ValueError("experiment capacity must be positive")
        layout = kv.KVMoveLayout.from_pool(pool)
        groups = continuous_groups(layout)
        if layout.device.type != "cuda":
            raise kv.UnsupportedKVMoveLayout("experiment requires CUDA")
        if kv._capturing(layout.device):
            raise RuntimeError("prepare CUDA experiment before capture")
        if (
            not isinstance(getattr(pool, "k_buffer", None), (list, tuple))
            or not isinstance(getattr(pool, "v_buffer", None), (list, tuple))
            or getattr(pool, "index_k_buffer", None) is not None
        ):
            raise kv.UnsupportedKVMoveLayout(
                "experiment requires separate MHA K/V lists"
            )
        from sglang.srt.speculative.standalone_remote import (
            sr_kv_copy_kernels_cuda_experiment,
        )

        super().__init__(pool, layout, graph=graph)
        self.reserve(capacity)
        self.backend = "cuda_contiguous_experiment"
        self.tables = []
        self._pointer_hosts = []
        self.launcher = sr_kv_copy_kernels_cuda_experiment.launch
        # Retain partial uploads too if a later pointer upload/record fails.
        try:
            for (_, width, strides), indices in groups:
                host = torch.tensor(
                    [
                        [
                            layout.buffers[i].tensor.data_ptr(),
                            self.scratch[i].data_ptr(),
                        ]
                        for i in indices
                    ],
                    dtype=torch.int64,
                ).pin_memory()
                table = torch.empty_like(host, device=layout.device)
                self._pointer_hosts.append(host)
                self.tables.append(
                    (
                        table,
                        width,
                        strides[0],
                        layout.buffers[indices[0]].tensor.element_size(),
                    )
                )
                table.copy_(host, non_blocking=True)
        except BaseException as exc:
            self._poison()
            # The failed constructor cannot otherwise return ownership.
            pool._kv_experiment_unresolved_hold = self
            raise kv.KVMoveSubmittedError(
                "CUDA pointer upload completion unknown"
            ) from exc
        self.frozen = True
        self._scratch_identity = tuple(kv._source_signature(t) for t in self.scratch)

    def check(self, *, check_stream=True):
        super().check(check_stream=check_stream)
        expected = getattr(self, "_scratch_identity", None)
        if expected is not None and expected != tuple(
            kv._source_signature(t) for t in self.scratch
        ):
            raise RuntimeError("experiment scratch storage changed")

    def _poison(self):
        super()._poison()
        # Keep pointer tables, indices and scratch reachable even if a test's
        # local workspace variable unwinds after a submitted failure.
        self.pool._kv_experiment_unresolved_hold = self

    def replay(self, graph):
        """Use this wrapper to validate storage/stream before test graph replay."""
        self.check()
        try:
            graph.replay()
        except BaseException as exc:
            self._poison()
            self._failed_graph = graph
            raise kv.KVMoveSubmittedError(
                "experiment graph completion unknown"
            ) from exc

    def _submit(self, inputs, **kwargs):
        tables = tuple(t[0] for t in self.tables)
        self.submitted(inputs + tables, lambda: self.launcher(self, **kwargs))

    def move(self, src, dst):
        if src.ndim != 1 or dst.ndim != 1:
            raise ValueError("experiment slot indices must be vectors")
        src, dst = kv._indices(self, src, dst)
        if src is None:
            return
        self._submit((src, dst), src=src, dst=dst)
        self.account(src.numel())

    def remap(self, slots, parents, depth):
        self.check()
        if (
            slots.ndim != 2
            or parents.ndim != 1
            or slots.shape[1] != parents.numel()
            or not 0 <= depth <= slots.shape[0]
            or depth * parents.numel() > self.capacity
        ):
            raise ValueError("invalid tree geometry or scratch capacity")
        for t in (slots, parents):
            if t.device != self.layout.device or t.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError("tree indices must be device-local integers")
        if not depth or not parents.numel():
            return
        self._submit((slots, parents), slots=slots, parents=parents, depth=depth)
        self.account(depth * parents.numel())
