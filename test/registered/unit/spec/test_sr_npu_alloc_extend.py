"""CPU checks for NPU paged extend and sort storage size."""

from __future__ import annotations

import unittest
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

# Execute actual allocator host methods with a CPU-only base and launch oracle.
# This verifies dispatch/ownership, not the Ascend kernel implementation.
_ROOT = Path(__file__).resolve().parents[4]


class _CPUBase:
    def __init__(self, size, page_size, dtype, device, kvcache, need_sort):
        self.page_size, self.device = page_size, device
        self.need_sort, self.debug_mode = need_sort, False
        self.free_pages = torch.arange(1, size // page_size + 1)
        self.release_pages = torch.empty(0, dtype=torch.int64)


def _load_allocator():
    src = _ROOT / "python/sglang/srt/hardware_backend/npu/allocator_npu.py"
    nodes = [
        n
        for n in ast.parse(src.read_text(encoding="utf-8")).body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
    ]
    base_src = _ROOT / "python/sglang/srt/mem_cache/allocator.py"
    merge = next(
        n
        for n in ast.walk(ast.parse(base_src.read_text(encoding="utf-8")))
        if isinstance(n, ast.FunctionDef) and n.name == "merge_and_sort_free"
    )
    future = ast.parse("from __future__ import annotations").body[0]
    env = dict(
        torch=torch,
        PagedTokenToKVPoolAllocator=_CPUBase,
        next_power_of_2=lambda n: 1 << max(n - 1, 0).bit_length(),
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, merge, *nodes], type_ignores=[])
            ),
            str(src),
            "exec",
        ),
        env,
    )
    _CPUBase.merge_and_sort_free = env["merge_and_sort_free"]
    return env["NPUPagedTokenToKVPoolAllocator"]


NPUPagedTokenToKVPoolAllocator = _load_allocator()


class _Launch:
    def __init__(self):
        self.calls = 0

    def __getitem__(self, grid):
        def run(prefix, seq, last, pages, output, batch_cap, page_size, token_cap):
            self.calls += 1
            free_offset, output_offset = 0, 0
            for p, end, tail in zip(prefix.tolist(), seq.tolist(), last.tolist()):
                for pos in range(p, end):
                    if pos // page_size == p // page_size and p % page_size:
                        output[output_offset] = tail + pos - p + 1
                    else:
                        page = (
                            free_offset
                            + pos // page_size
                            - (p + page_size - 1) // page_size
                        )
                        output[output_offset] = (
                            pages[page] * page_size + pos % page_size
                        )
                    output_offset += 1
                free_offset += (end + page_size - 1) // page_size - (
                    p + page_size - 1
                ) // page_size

        return run


from sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept import (
    apply_free_unique_pages,
)
from sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept_kernels import (
    gather_commit_slots,
    owned_span,
)


def _gather(cache, src_index, tgt_index, page_index, page_size):
    device = cache.device
    outs = tuple(
        torch.empty(int(index.numel()), dtype=torch.int64, device=device)
        for index in (src_index, tgt_index, page_index)
    )
    return gather_commit_slots(
        cache, src_index, tgt_index, page_index, page_size, *outs
    )
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=15, suite="stage-a-test-cpu")


def _allocator(need_sort: bool = False) -> NPUPagedTokenToKVPoolAllocator:
    return NPUPagedTokenToKVPoolAllocator(
        size=8192,
        page_size=128,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


class NPUAllocExtendReleaseTest(CustomTestCase):
    def setUp(self):
        self.kernel = _Launch()
        ctx = patch.dict(
            sys.modules,
            {
                "sgl_kernel_npu.mem_cache.allocator": SimpleNamespace(
                    alloc_extend_kernel=self.kernel
                )
            },
        )
        ctx.start()
        self.addCleanup(ctx.stop)

    def test_release_pages_satisfy_page_aligned_verify(self):
        alloc = _allocator(need_sort=False)
        alloc.release_pages = alloc.free_pages.flip(0).clone()
        alloc.free_pages = torch.empty((0,), dtype=torch.int64)
        prefix = torch.tensor([2176], dtype=torch.int64)
        seq = torch.tensor([2196], dtype=torch.int64)
        last_loc = torch.tensor([2175], dtype=torch.int64)

        out = alloc.alloc_extend(prefix, prefix, seq, seq, last_loc, 20)

        self.assertEqual(out.tolist(), list(range(128, 148)))
        self.assertEqual(len(alloc.free_pages), 63)
        self.assertEqual(len(alloc.release_pages), 0)
        self.assertEqual(self.kernel.calls, 1)
        # Tail views remain views; no full remaining-pool clone per allocation.
        self.assertGreater(
            alloc.free_pages.untyped_storage().nbytes(), alloc.free_pages.numel() * 8
        )

    def test_shortfall_preserves_none_contract_without_launch(self):
        alloc = _allocator(need_sort=False)
        alloc.free_pages = torch.empty((0,), dtype=torch.int64)
        alloc.release_pages = torch.empty((0,), dtype=torch.int64)
        prefix = torch.tensor([2176], dtype=torch.int64)
        seq = torch.tensor([2196], dtype=torch.int64)
        last_loc = torch.tensor([2175], dtype=torch.int64)

        self.assertIsNone(alloc.alloc_extend(prefix, prefix, seq, seq, last_loc, 20))
        self.assertEqual(self.kernel.calls, 0)

    def test_mixed_page_ranges_match_host_oracle_without_pool_clone(self):
        alloc = _allocator()
        prefix = torch.tensor([127, 128, 129, 0])
        seq = torch.tensor([130, 131, 132, 2])
        last = torch.tensor([3966, 127, 512, -1])
        storage = alloc.free_pages.untyped_storage().data_ptr()
        out = alloc.alloc_extend(prefix, prefix, seq, seq, last, 11)
        self.assertEqual(
            out.tolist(), [3967, 128, 129, 256, 257, 258, 513, 514, 515, 384, 385]
        )
        self.assertEqual(alloc.free_pages.untyped_storage().data_ptr(), storage)
        self.assertEqual(self.kernel.calls, 1)


class NPUSortStorageTest(CustomTestCase):
    def test_independent_outputs_and_sort_storage_match_logical_length(self):
        parent = torch.arange(768, dtype=torch.int64)
        pages = _gather(
            torch.arange(8, dtype=torch.int64),
            torch.arange(8),
            torch.arange(8),
            torch.arange(8),
            1,
        )[2]
        self.assertEqual(pages.numel(), 8)
        self.assertEqual(pages.untyped_storage().nbytes(), 8 * pages.element_size())
        self.assertNotEqual(
            pages.untyped_storage().data_ptr(), parent.untyped_storage().data_ptr()
        )

        view = parent[:8]
        self.assertGreater(view.untyped_storage().nbytes(), view.numel() * 8)
        compact = owned_span(view)
        self.assertEqual(compact.tolist(), list(range(8)))
        self.assertEqual(compact.untyped_storage().nbytes(), 8 * compact.element_size())

        holder = type("Holder", (), {})()
        holder.is_not_in_free_group = True
        holder.need_sort = False
        holder.debug_mode = False
        holder.free_pages = torch.tensor([11], dtype=torch.int64)
        holder.release_pages = torch.empty((0,), dtype=torch.int64)
        real_sort = torch.sort

        def _spy(tensor, *args, **kwargs):
            self.assertEqual(tensor.numel(), 8)
            self.assertEqual(
                tensor.untyped_storage().nbytes(),
                tensor.numel() * tensor.element_size(),
            )
            return real_sort(tensor, *args, **kwargs)

        with patch("torch.sort", side_effect=_spy):
            apply_free_unique_pages(holder, view)
        self.assertEqual(holder.free_pages.tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 11])

    def test_merge_sorts_a_tail_view_from_its_own_storage(self):
        alloc = _allocator(need_sort=False)
        parent = torch.arange(768, dtype=torch.int64)
        alloc.free_pages = parent[760:]
        alloc.release_pages = torch.tensor([900, 901], dtype=torch.int64)
        real_sort = torch.sort
        seen = {}

        def _spy(tensor, *args, **kwargs):
            seen["numel"] = tensor.numel()
            seen["storage"] = tensor.untyped_storage().nbytes()
            return real_sort(tensor, *args, **kwargs)

        with patch("torch.sort", side_effect=_spy):
            alloc.merge_and_sort_free()

        self.assertEqual(seen["numel"], 10)
        self.assertEqual(seen["storage"], 10 * 8)
        self.assertEqual(len(alloc.release_pages), 0)
        self.assertEqual(alloc.free_pages.numel(), 10)


if __name__ == "__main__":
    unittest.main()
