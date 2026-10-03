"""SR fused integer preparation: real-device oracles, no model or new runner."""

import pathlib
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
    ALLOC_LEASE,
    ALLOC_ORDINARY,
    PrefixTailCopyBuffers,
    fill_prefix_tail_copy_slots,
    materialize_prefix_tail_copy_slots,
    plan_prefix_tail_copy_indices,
)
from sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept_kernels import (
    gather_commit_slots,
    pack_accept,
)
from sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept import (
    apply_free_unique_pages,
)

try:
    import torch_npu
except ImportError:
    torch_npu = None


class LaunchRecorder:
    def __init__(self, kernel):
        self.kernel = kernel
        self.calls = 0

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.calls += 1
            return self.kernel[grid](*args, **kwargs)

        return run


def _sort_storage_probe(variant):
    """Run only in a child process: a failing legacy sort may poison its NPU."""
    if variant == "independent":
        index = torch.arange(6, device="npu", dtype=torch.int64)
        cache = torch.arange(15, device="npu", dtype=torch.int64) + 248
        pages = gather_commit_slots(
            cache, index, index, torch.tensor([8], device="npu"), 128
        )[2]
    else:
        backing = torch.zeros(13, device="npu", dtype=torch.int64)
        backing[12] = 2
        pages = backing[12:]
        if variant == "contiguous":
            pages = pages.contiguous()
        elif variant == "clone":
            pages = pages.clone()
        elif variant != "view":
            raise ValueError(variant)
    print(
        variant,
        "shape",
        tuple(pages.shape),
        "offset",
        pages.storage_offset(),
        "logical_bytes",
        pages.numel() * 8,
        "storage_bytes",
        pages.untyped_storage().nbytes(),
        flush=True,
    )
    torch.npu.synchronize()
    print("sort_begin", flush=True)
    values, _ = torch.sort(pages)
    torch.npu.synchronize()
    assert values.cpu().tolist() == [2]
    print("sort_complete", flush=True)


@unittest.skipUnless(torch_npu is not None, "requires torch_npu and NPU")
class TestNPUSmallKernels(unittest.TestCase):
    def test_sort_storage_variants_in_isolated_processes(self):
        for variant in ("view", "contiguous", "clone", "independent"):
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import runpy,sys; runpy.run_path(sys.argv[1])['_sort_storage_probe'](sys.argv[2])",
                    str(pathlib.Path(__file__).resolve()),
                    variant,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            print(
                f"sort probe {variant}: exit={child.returncode}\n{child.stdout}\n{child.stderr}",
                flush=True,
            )
            # Legacy outcomes are diagnostic: compatible runtimes may succeed.
            if variant == "independent":
                self.assertEqual(child.returncode, 0, child.stderr)
                self.assertIn("sort_complete", child.stdout)

    def test_pack_exact_cpu_oracle(self):
        from sglang.srt.speculative.standalone_remote import (
            sr_small_kernels_npu as kernels,
        )

        for dtype in (torch.int32, torch.int64):
            for bs in (0, 1, 2, 3, 4):
                for length in (0, 1, 6, 7, 17):
                    for n_predict in (0, 31):
                        backing = (
                            torch.arange(bs * length * 2, dtype=dtype) % 37 - 3
                        ).view(bs, length * 2)
                        index = backing[:, ::2]
                        pred = torch.arange(n_predict, dtype=dtype) + 900
                        counts = torch.arange(bs, dtype=dtype)
                        cpu = torch.full((bs, 4 * length + 8), -99, dtype=torch.int64)
                        dev = cpu.to("npu")
                        pack_accept(index, pred, counts, cpu[:, ::2])
                        with patch.object(
                            kernels, "_pack", LaunchRecorder(kernels._pack)
                        ) as launch:
                            pack_accept(
                                backing.to("npu")[:, ::2],
                                pred.to("npu"),
                                counts.to("npu"),
                                dev[:, ::2],
                            )
                            self.assertEqual(launch.calls, int(bs > 0))
                        self.assertTrue(torch.equal(cpu, dev.cpu()))

    def test_commit_single_launch_and_owned_results(self):
        from sglang.srt.speculative.standalone_remote import (
            sr_small_kernels_npu as kernels,
        )

        for dtype in (torch.int32, torch.int64):
            for n, f in ((0, 0), (0, 3), (1, 0), (1, 1), (6, 1), (6, 3), (513, 129)):
                cache_back = torch.arange(2048, dtype=dtype) * 129
                cache = cache_back[::2]
                cache_device = cache_back.to("npu")[::2]
                src = torch.arange(2 * n)[::2] % 1024
                dst = src.flip(0)
                pages = torch.arange(2 * f)[::2]
                expected = gather_commit_slots(cache, src, dst, pages, 128)
                with patch.object(
                    kernels, "_commit", LaunchRecorder(kernels._commit)
                ) as launch:
                    actual = gather_commit_slots(
                        cache_device,
                        src.to("npu"),
                        dst.to("npu"),
                        pages.to("npu"),
                        128,
                    )
                    self.assertEqual(launch.calls, int(max(n, f) > 0))
                for a, e in zip(actual, expected):
                    self.assertTrue(torch.equal(a.cpu(), e))
                    self.assertEqual(a.storage_offset(), 0)
                    self.assertEqual(a.untyped_storage().nbytes(), a.numel() * 8)
                nonempty = [t for t in actual if t.numel()]
                self.assertEqual(
                    len({t.untyped_storage().data_ptr() for t in nonempty}),
                    len(nonempty),
                )
                for need_sort in (False, True):
                    holder = SimpleNamespace(
                        is_not_in_free_group=True,
                        need_sort=need_sort,
                        debug_mode=False,
                        free_pages=torch.tensor([9000], device="npu"),
                        release_pages=torch.tensor([9001], device="npu"),
                    )
                    apply_free_unique_pages(holder, actual[2])
                    torch.npu.synchronize()
                    sorted_pages = sorted(expected[2].tolist())
                    self.assertEqual(
                        holder.free_pages.cpu().tolist(),
                        sorted_pages + [9000] if not need_sort else [9000],
                    )
                    self.assertEqual(
                        holder.release_pages.cpu().tolist(),
                        sorted_pages + [9001] if need_sort else [9001],
                    )
                # A later call must not overwrite a retained earlier output.
                gather_commit_slots(
                    (cache + 7).to("npu"),
                    src.to("npu"),
                    dst.to("npu"),
                    pages.to("npu"),
                    128,
                )
                for a, e in zip(actual, expected):
                    self.assertTrue(torch.equal(a.cpu(), e))

    def test_prefix_strides_mixed_rows_and_growth(self):
        from sglang.srt.speculative.standalone_remote import (
            sr_small_kernels_npu as kernels,
        )

        for dtype in (torch.int32, torch.int64):
            req_back = torch.arange(8 * 1024, dtype=dtype).view(8, 1024)
            pool_back = torch.tensor([3, 0, 1, 0, 2, 0, 0, 0], dtype=dtype)
            branch_back = torch.arange(4 * 6 * 2, dtype=dtype).view(4, 6, 2) + 100
            req, pool, branch = (
                req_back[:, ::2],
                pool_back[::2],
                branch_back[:, ::2, ::2],
            )
            rd, pd, bd = (
                req_back.to("npu")[:, ::2],
                pool_back.to("npu")[::2],
                branch_back.to("npu")[:, ::2, ::2],
            )
            buffers = PrefixTailCopyBuffers()
            before = rd.cpu()
            for prefixes in ([127], [127, 128, 50], [1, 255, 129, 127], [128], [7]):
                for kind in (ALLOC_ORDINARY, ALLOC_LEASE):
                    plan = plan_prefix_tail_copy_indices(prefixes, kind, 3, 128)
                    expected = materialize_prefix_tail_copy_slots(
                        req, pool, branch, plan, 128
                    )
                    with patch.object(
                        kernels, "_prefix", LaunchRecorder(kernels._prefix)
                    ) as launch:
                        actual = fill_prefix_tail_copy_slots(
                            torch.tensor(prefixes, device="npu"),
                            rd,
                            pd,
                            bd,
                            prefixes,
                            kind,
                            3,
                            128,
                            buffers,
                        )
                        self.assertEqual(launch.calls, int(len(plan) > 0))
                    for a, e in zip(actual, expected):
                        self.assertTrue(torch.equal(a.cpu(), e.long()))
            self.assertTrue(torch.equal(before, rd.cpu()))

    def test_graph_reads_changed_device_mappings(self):
        # CPU preparation stays outside graph; this exercises only the captured
        # integer kernel with live metadata and padded B=3 mapping storage.
        from sglang.srt.speculative.standalone_remote import (
            sr_small_kernels_npu as kernels,
        )

        req = torch.arange(4 * 256, device="npu", dtype=torch.int64).view(4, 256)
        pool = torch.arange(4, device="npu", dtype=torch.int64)
        branch = (torch.arange(12, device="npu", dtype=torch.int64) + 10).view(4, 3, 1)
        meta = torch.tensor([[0, 7, 0], [128, 0, 14], [0, 3, 14]], device="npu")
        src = torch.empty(32, device="npu", dtype=torch.int64)
        dst = torch.empty_like(src)
        kernels.prefix_tail(req, pool, branch, meta, src, dst, 3, 1, 3, 128)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            kernels.prefix_tail(req, pool, branch, meta, src, dst, 3, 1, 3, 128)
        for prefixes, order in (([7, 128, 3], [0, 1, 2, 3]), ([1, 3, 5], [2, 0, 1, 3])):
            pool.copy_(torch.tensor(order, device="npu"))
            branch.add_(2)
            offset, rows = 0, []
            for prefix in prefixes:
                rem = prefix % 128
                rows.append([prefix - rem, rem, offset])
                offset += rem * 2
            meta.copy_(torch.tensor(rows, device="npu"))
            graph.replay()
            plan = plan_prefix_tail_copy_indices(prefixes, ALLOC_ORDINARY, 3, 128)
            expected = materialize_prefix_tail_copy_slots(
                req.cpu(), pool.cpu(), branch.cpu(), plan, 128
            )
            self.assertTrue(torch.equal(src[:offset].cpu(), expected[0]))
            self.assertTrue(torch.equal(dst[:offset].cpu(), expected[1]))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCUDAGenericPreparation(unittest.TestCase):
    def test_pack_and_commit(self):
        idx = torch.tensor([[0, 1, -1], [3, 4, 5]])
        pred, counts = torch.arange(6) + 10, torch.tensor([1, 2])
        cpu, dev = torch.empty(2, 8, dtype=torch.int64), torch.empty(
            2, 8, dtype=torch.int64, device="cuda"
        )
        pack_accept(idx, pred, counts, cpu)
        pack_accept(idx.cuda(), pred.cuda(), counts.cuda(), dev)
        self.assertTrue(torch.equal(cpu, dev.cpu()))
        inputs = (
            torch.arange(32),
            torch.tensor([1, 0, 1]),
            torch.tensor([0, 1, 2]),
            torch.tensor([20]),
        )
        expected = gather_commit_slots(*inputs, 8)
        actual = gather_commit_slots(*(t.cuda() for t in inputs), 8)
        for a, e in zip(actual, expected):
            self.assertTrue(torch.equal(a.cpu(), e))
            self.assertEqual(a.storage_offset(), 0)
        holder = SimpleNamespace(
            is_not_in_free_group=True,
            need_sort=False,
            free_pages=torch.tensor([99], device="cuda"),
            release_pages=torch.empty(0, dtype=torch.int64, device="cuda"),
        )
        apply_free_unique_pages(holder, actual[2])
        self.assertEqual(
            holder.free_pages.cpu().tolist(), sorted(expected[2].tolist()) + [99]
        )


if __name__ == "__main__":
    unittest.main()
