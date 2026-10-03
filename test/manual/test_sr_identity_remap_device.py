"""Device checks for the Draft identity remap kernel. Run with PYTHONPATH=python.

CPU unit tests cover the portable snapshot. This file launches the Triton kernel.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
    IdentityRemapWorkspace,
    remap_slot_node_ids,
)

try:
    import torch_npu
except ImportError:
    torch_npu = None


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch_npu is not None:
        try:
            if torch.npu.is_available():
                return torch.device("npu")
        except Exception:
            return None
    return None


def _gold(ids, parents, depth):
    out = ids.clone()
    rows = int(parents.numel())
    idx = parents.to(dtype=torch.int64).reshape(-1)
    for step in range(int(depth)):
        out[step, :rows] = out[step, :rows].index_select(0, idx)
    return out


@unittest.skipUnless(_device() is not None, "requires CUDA or NPU")
class TestIdentityRemapDevice(unittest.TestCase):
    def test_warmup_does_not_write_live_table(self):
        device = _device()
        live = torch.arange(4, dtype=torch.int64, device=device).reshape(2, 2)
        before = live.clone()
        workspace = IdentityRemapWorkspace(
            device, 2, 2, graph=True, domain="tree_graph"
        )
        workspace.warm()
        torch.testing.assert_close(live, before)
        self.assertFalse(workspace.unresolved)

    def test_swap_cycle_duplicate_and_padding_tail(self):
        device = _device()
        ids = torch.tensor(
            [
                [10, 20, 30, 40, 7],
                [11, 21, 31, 41, 8],
                [12, 22, 32, 42, 9],
            ],
            dtype=torch.int64,
            device=device,
        )
        parents = torch.tensor([1, 2, 0, 1], dtype=torch.int32, device=device)
        scratch = torch.empty((3, 8), dtype=torch.int64, device=device)
        expected = _gold(ids.cpu(), parents.cpu(), 2)
        remap_slot_node_ids(ids, parents, 2, scratch)
        self.assertEqual(ids.cpu().tolist(), expected.tolist())
        self.assertEqual(ids[:, 4].tolist(), [7, 8, 9])

    def test_wide_strided_table_and_noncontiguous_parents(self):
        device = _device()
        cases = (
            torch.tensor([1, 0], dtype=torch.int32),
            torch.tensor([1, 2, 0], dtype=torch.int32),
            torch.tensor([0, 0, 0], dtype=torch.int32),
            torch.tensor([2, 2, 0, 1], dtype=torch.int32),
        )
        for logical in cases:
            rows = int(logical.numel())
            width = rows + 2
            step_stride = width + 4
            depth_cap = 4
            backing = torch.arange(
                depth_cap * step_stride, dtype=torch.int64, device=device
            )
            ids = backing.as_strided((depth_cap, width), (step_stride, 1))
            before = backing.clone()
            spaced = torch.empty(rows * 2, dtype=torch.int32, device=device)
            spaced[::2] = logical.to(device)
            parents = spaced.as_strided((rows,), (2,))
            self.assertFalse(parents.is_contiguous())
            scratch = torch.empty(
                (depth_cap, width + 3), dtype=torch.int64, device=device
            )
            expected = _gold(ids.detach().cpu(), parents.detach().cpu(), 2)
            remap_slot_node_ids(ids, parents, 2, scratch)
            self.assertEqual(ids.cpu().tolist(), expected.tolist())
            for step in range(depth_cap):
                gap = slice(step * step_stride + width, (step + 1) * step_stride)
                self.assertEqual(backing[gap].tolist(), before[gap].tolist())

    def test_chunk_boundary_gathers_before_scatter(self):
        device = _device()
        rows, depth = 40000, 2
        self.assertGreater(depth * rows, 65535)
        ids = torch.arange(depth * rows, dtype=torch.int64, device=device).reshape(
            depth, rows
        )
        parents = torch.arange(rows, dtype=torch.int64, device=device)
        parents[0] = 1
        parents[1] = 0
        parents[4] = 2
        parents[5] = 2
        cross = 65535
        step, row = divmod(cross, rows)
        parents[row - 1] = row
        parents[row] = row - 1
        before = ids.clone()
        scratch = torch.empty((depth, rows + 1), dtype=torch.int64, device=device)
        from sglang.srt.speculative.standalone_remote.drafter import (
            sr_identity_remap_kernels as identity_kernels,
        )

        launches = []
        jit_fn = identity_kernels._remap_identity

        class _RecordScatter:
            def __getitem__(self, grid):
                launch = jit_fn[grid]

                def run(*args, **kwargs):
                    launches.append(bool(kwargs["Scatter"]))
                    return launch(*args, **kwargs)

                return run

        with patch.object(identity_kernels, "_remap_identity", _RecordScatter()):
            remap_slot_node_ids(ids, parents, depth, scratch)
        limit = identity_kernels._GRID_LIMIT
        chunks = (depth * rows + limit - 1) // limit
        self.assertGreater(chunks, 1)
        self.assertEqual(launches, [False] * chunks + [True] * chunks)
        self.assertEqual(int(ids[0, 0].item()), int(before[0, 1].item()))
        self.assertEqual(int(ids[0, 1].item()), int(before[0, 0].item()))
        self.assertEqual(int(ids[0, 4].item()), int(before[0, 2].item()))
        self.assertEqual(int(ids[0, 5].item()), int(before[0, 2].item()))
        for table_step in range(depth):
            self.assertEqual(
                int(ids[table_step, row - 1].item()),
                int(before[table_step, row].item()),
            )
            self.assertEqual(
                int(ids[table_step, row].item()),
                int(before[table_step, row - 1].item()),
            )
        self.assertEqual(step, 1)

    def test_out_of_range_parent_keeps_its_row(self):
        device = _device()
        ids = torch.tensor(
            [
                [10, 20, 30, 40, 7],
                [11, 21, 31, 41, 8],
                [12, 22, 32, 42, 9],
            ],
            dtype=torch.int64,
            device=device,
        )
        parents = torch.tensor([1, 99, 0], dtype=torch.int64, device=device)
        before = ids.clone()
        scratch = torch.empty((3, 8), dtype=torch.int64, device=device)
        remap_slot_node_ids(ids, parents, 2, scratch)
        self.assertEqual(ids[0, :3].tolist(), [20, 20, 10])
        self.assertEqual(ids[1, :3].tolist(), [21, 21, 11])
        self.assertEqual(ids[2].tolist(), before[2].tolist())
        self.assertEqual(ids[:, 3:].tolist(), before[:, 3:].tolist())
