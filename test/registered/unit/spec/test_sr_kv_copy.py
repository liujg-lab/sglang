"""CPU contracts for pool-owned KV scratch and two-stage movement."""

from __future__ import annotations

import ast
import sys
import unittest
from collections import Counter, namedtuple
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.speculative.standalone_remote import sr_kv_copy as kv
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def paged(dtype=torch.float32, width=7):
    return NS(
        kv_buffer=torch.arange(2 * 3 * 4 * 4 * width)
        .reshape(2, 3, 4, 4, 1, width)
        .to(dtype)
    )


def snapshot_move(layout, src, dst):
    originals = [b.tensor.clone() for b in layout.buffers]
    for b, original in zip(layout.buffers, originals):
        b.tensor.index_copy_(
            b.axis, dst.long(), original.index_select(b.axis, src.long())
        )


MetadataDevice = namedtuple("MetadataDevice", "type index")


class MetadataNPUTensor(torch.Tensor):
    """CPU payload with NPU metadata only; never claims device execution."""

    @property
    def device(self):
        return MetadataDevice("npu", 0)


@contextmanager
def npu_metadata_runtime():
    formats = {}
    allocations = []
    api = NS()

    def empty_with_format(shape, *, dtype, device, acl_format):
        tensor = torch.empty(shape, dtype=dtype).as_subclass(MetadataNPUTensor)
        allocations.append((shape, dtype, device, acl_format))
        formats[tensor.data_ptr()] = 2
        return tensor

    def get_format(tensor):
        return formats.get(tensor.data_ptr(), 2)

    def factory(original):
        def allocate(*args, **kwargs):
            kwargs.pop("device", None)
            return original(*args, **kwargs).as_subclass(MetadataNPUTensor)

        return allocate

    api.empty_with_format = empty_with_format
    api.get_npu_format = get_format
    stream = NS(stream_id=0, synchronize=lambda: None)
    kernels = NS()
    with patch.dict(
        sys.modules,
        {
            "torch_npu": api,
            "sglang.srt.speculative.standalone_remote.sr_kv_copy_kernels": kernels,
        },
    ), patch.object(kv, "_stream", return_value=stream), patch.object(
        kv, "_capturing", return_value=False
    ), patch.object(
        torch, "zeros", factory(torch.zeros)
    ), patch.object(
        torch, "arange", factory(torch.arange)
    ):
        yield api, formats, allocations


class TestKVMove(unittest.TestCase):
    def test_real_kernel_body_identity_masks_and_cross_chunk_snapshot(self):
        """Execute the actual Triton function body with CPU pointer semantics.

        This checks masks/index arithmetic/order, not Ascend compilation or
        hardware scheduling. Device tests cover those separately.
        """
        path = Path(kv.__file__).with_name("sr_kv_copy_kernels.py")
        tree = ast.parse(path.read_text())
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_paged_move"
        )
        function.decorator_list = []
        for arg in function.args.args:
            arg.annotation = None
        counts = Counter()

        class Pointer:
            def __init__(self, tensor, name, offset=0):
                self.tensor = torch.as_strided(
                    tensor,
                    (tensor.untyped_storage().nbytes() // tensor.element_size(),),
                    (1,),
                    storage_offset=0,
                )
                self.name, self.offset = name, offset + tensor.storage_offset()

            def __add__(self, value):
                return Pointer(self.tensor, self.name, self.offset + value)

        class Language:
            int64, int1 = torch.int64, torch.bool
            pid = (0, 0)

            def program_id(self, axis):
                return torch.tensor(self.pid[axis])

            def arange(self, lo, hi):
                return torch.arange(lo, hi)

            def device_assert(self, condition, message):
                if not bool(torch.as_tensor(condition).all()):
                    raise AssertionError(message)

            def load(self, pointer, mask=True, other=0):
                indices = torch.as_tensor(pointer.offset)
                mask = torch.as_tensor(mask).expand_as(indices)
                safe = torch.where(mask, indices, torch.zeros_like(indices))
                counts[(pointer.name, "read")] += int(mask.sum())
                return torch.where(mask, pointer.tensor[safe], other)

            def store(self, pointer, value, mask=True):
                offsets, values, masks = torch.broadcast_tensors(
                    torch.as_tensor(pointer.offset), value, torch.as_tensor(mask)
                )
                counts[(pointer.name, "write")] += int(masks.sum())
                pointer.tensor[offsets[masks]] = values[masks]

        language = Language()
        ns = {"tl": language}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                str(path),
                "exec",
            ),
            ns,
        )
        kernel = ns["_paged_move"]

        class Launch:
            def __getitem__(self, grid):
                def call(
                    pool, scratch, src, dst, slots, parents, active, *args, **kwargs
                ):
                    tensors = [pool, scratch, src, dst, slots, parents, active]
                    names = [
                        "pool",
                        "scratch",
                        "src",
                        "dst",
                        "slots",
                        "parents",
                        "active",
                    ]
                    pointers = [Pointer(t, name) for t, name in zip(tensors, names)]
                    for x in range(grid[0]):
                        for y in range(grid[1]):
                            language.pid = x, y
                            kernel(*pointers, *args, **kwargs)

                return call

        launch_ns = {"_paged_move": Launch()}
        funcs = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("_launch", "move", "remap")
        ]
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])),
                str(path),
                "exec",
            ),
            launch_ns,
        )
        for src in (
            torch.arange(17),
            torch.tensor([0] * 17),
            torch.roll(torch.arange(17), 6),
            torch.tensor([16] * 17),
        ):
            counts.clear()
            pool = torch.arange(2 * 3 * 20 * 259).reshape(2, 3, 20, 259).float()
            initial = pool.clone()
            expected = initial.clone()
            dst = torch.arange(17)
            expected.index_copy_(2, dst, initial.index_select(2, src))
            with patch.object(kv, "_PAGED_GRID_LIMIT", 63):
                launch_ns["move"](pool, torch.empty(2, 3, 32, 259), src, dst)
            torch.testing.assert_close(pool, expected, rtol=0, atol=0)
            moved = int((src != dst).sum()) * 6 * 259
            self.assertEqual(counts["pool", "read"], moved)
            self.assertEqual(counts["pool", "write"], moved)
        # Strided slots, repeated parents and a padded inactive row whose
        # dummy slot is also read by an active row.
        slots = torch.tensor([[0, 1], [2, 3], [4, 5], [0, 0]]).T
        active = torch.tensor([True, True, True, False])
        parents = torch.tensor([0, 0, 1, 3])
        pool = torch.arange(2 * 3 * 20 * 259).reshape(2, 3, 20, 259).float()
        initial = pool.clone()
        expected = pool.clone()
        src = slots[:, parents[:3]].reshape(-1)
        dst = slots[:, :3].reshape(-1)
        expected.index_copy_(2, dst, initial.index_select(2, src))
        identity_slots = set()
        moved = reread = 0
        depth, rows = 2, int(parents.shape[0])
        for step in range(depth):
            for row in range(rows):
                if not bool(active[row]):
                    continue
                src_slot = int(slots[step, int(parents[row])])
                dst_slot = int(slots[step, row])
                if src_slot == dst_slot:
                    identity_slots.add((step, dst_slot))
                else:
                    moved += 1
        for step in range(depth):
            for row in range(rows):
                if not bool(active[row]):
                    continue
                src_slot = int(slots[step, int(parents[row])])
                dst_slot = int(slots[step, row])
                if src_slot != dst_slot and (step, src_slot) in identity_slots:
                    reread += 1
        self.assertGreater(len(identity_slots), 0)
        self.assertGreater(reread, 0)
        counts.clear()
        with patch.object(kv, "_PAGED_GRID_LIMIT", 63):
            launch_ns["remap"](
                pool, torch.empty(2, 3, 32, 259), slots, parents, 2, active
            )
        torch.testing.assert_close(pool, expected, rtol=0, atol=0)
        # Identity tasks do not touch the pool. Another active row may still
        # gather that same slot, and that gather stays in the read count.
        access = moved * pool.shape[0] * pool.shape[1] * pool.shape[3]
        self.assertEqual(counts["pool", "read"], access)
        self.assertEqual(counts["pool", "write"], access)

    def test_nd_backing_allocation_and_metadata_failures(self):
        with npu_metadata_runtime() as (api, _, allocations):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                view, backing = kv._empty_nd(
                    (2, 3, 4, 7),
                    dtype=dtype,
                    device=NS(type="npu", index=0),
                    role="scratch",
                )
                self.assertEqual(allocations[-1][0], (168,))
                self.assertEqual(allocations[-1][3], 2)
                self.assertEqual(view.data_ptr(), backing.data_ptr())
                self.assertEqual(view.stride(), (84, 28, 7, 1))
                self.assertEqual(view.dtype, dtype)
            for rejected_rank, role in ((1, "scratch.backing"), (4, "scratch")):
                with patch.object(
                    api,
                    "get_npu_format",
                    side_effect=lambda t: 0 if t.ndim == rejected_rank else 2,
                ), self.assertRaisesRegex(
                    kv.UnsupportedKVMoveLayout, "role=" + role + ".*actual_format=0"
                ):
                    kv._empty_nd(
                        (2, 3, 4, 7),
                        dtype=torch.float32,
                        device=NS(type="npu", index=0),
                        role="scratch",
                    )
            original = api.empty_with_format

            def wrong_dtype(shape, **kwargs):
                kwargs["dtype"] = torch.float16
                return original(shape, **kwargs)

            with patch.object(
                api, "empty_with_format", wrong_dtype
            ), self.assertRaisesRegex(kv.UnsupportedKVMoveLayout, "metadata mismatch"):
                kv._empty_nd(
                    (2, 3, 4, 7),
                    dtype=torch.float32,
                    device=NS(type="npu", index=0),
                    role="scratch",
                )
            original_view = MetadataNPUTensor.view

            def detached_view(tensor, shape):
                return original_view(tensor, shape).clone()

            with patch.object(
                MetadataNPUTensor, "view", detached_view
            ), self.assertRaisesRegex(kv.UnsupportedKVMoveLayout, "metadata mismatch"):
                kv._empty_nd(
                    (2, 3, 4, 7),
                    dtype=torch.float32,
                    device=NS(type="npu", index=0),
                    role="scratch",
                )

    def test_nd_shared_allocator_warmup_growth_and_retirement(self):
        pool = paged(torch.bfloat16)
        pool.kv_buffer = pool.kv_buffer.as_subclass(MetadataNPUTensor)
        before, address = pool.kv_buffer.clone(), pool.kv_buffer.data_ptr()
        with npu_metadata_runtime() as (api, _, allocations), patch.object(
            kv, "move_kv_slots_"
        ) as move, patch.object(kv, "_empty_nd", wraps=kv._empty_nd) as allocate:
            kv.warm_private_slot_move(pool)
            self.assertEqual(
                [c.kwargs["role"] for c in allocate.call_args_list],
                ["private_warmup", "scratch"],
            )
            private_ws = move.call_args.args[0]
            self.assertIsNotNone(private_ws.layout.buffers[0].backing)
            self.assertEqual(len(private_ws.scratch_backings), 1)
            self.assertTrue(next(iter(pool._kv_move_warmups.values())).complete)
            count = len(allocations)
            kv.warm_private_slot_move(pool)
            self.assertEqual(len(allocations), count)
            ws = kv.prepare_kv_move(pool, 2)
            old, backings = ws.scratch, ws.scratch_backings
            capacity, size = ws.capacity, ws.scratch_bytes
            with patch.object(
                api, "get_npu_format", side_effect=lambda t: 0 if t.ndim == 1 else 2
            ):
                with self.assertRaisesRegex(
                    kv.UnsupportedKVMoveLayout, "scratch.backing"
                ):
                    ws.reserve(8)
            self.assertIs(ws.scratch, old)
            self.assertIs(ws.scratch_backings, backings)
            self.assertEqual((ws.capacity, ws.scratch_bytes), (capacity, size))
            self.assertFalse(ws.unresolved)
            ws._used = True
            event = NS(query=lambda: False)
            with patch.object(ws, "_record_event", return_value=event):
                ws.reserve(8)
            self.assertIs(ws.retired[0][-1], backings)
            self.assertIs(ws.retired[0][1], old)
            self.assertEqual(ws.scratch_bytes, ws.scratch[0].nbytes)
            count = len(allocations)
            ws.reserve(8)
            self.assertEqual(len(allocations), count)
            self.assertEqual(ws.retired_bytes, size)
            event.query = lambda: True
            ws.reserve(8)
            self.assertEqual(ws.retired, [])
            self.assertTrue(torch.equal(pool.kv_buffer, before))
            self.assertEqual(pool.kv_buffer.data_ptr(), address)

    def test_nd_live_pool_and_cached_format_change_rejected_before_work(self):
        pool = paged()
        pool.kv_buffer = pool.kv_buffer.as_subclass(MetadataNPUTensor)
        before = pool.kv_buffer.clone()
        with npu_metadata_runtime() as (api, formats, allocations), patch.object(
            kv, "move_kv_slots_"
        ) as move:
            formats[pool.kv_buffer.data_ptr()] = 0
            with self.assertRaisesRegex(
                kv.UnsupportedKVMoveLayout, "role=live_pool.*actual_format=0"
            ):
                kv.warm_private_slot_move(pool)
            self.assertEqual(allocations, [])
            formats[pool.kv_buffer.data_ptr()] = 2
            # The normalized view must be checked too, even if the source is ND.
            with patch.object(
                api, "get_npu_format", side_effect=lambda t: 0 if t.ndim == 4 else 2
            ):
                with self.assertRaisesRegex(
                    kv.UnsupportedKVMoveLayout, "live_pool.view"
                ):
                    kv.prepare_kv_move(pool, 2)
            self.assertEqual(allocations, [])
            ws = kv.prepare_kv_move(pool, 2)
            signature = kv._source_signature(pool.kv_buffer)
            formats[pool.kv_buffer.data_ptr()] = 0
            self.assertNotEqual(kv._source_signature(pool.kv_buffer), signature)
            with self.assertRaises(kv.UnsupportedKVMoveLayout):
                ws.check()
            with self.assertRaises(kv.UnsupportedKVMoveLayout):
                kv.warm_private_slot_move(pool)
            self.assertEqual(ws.counts["calls"], 0)
            self.assertTrue(torch.equal(pool.kv_buffer, before))
            move.assert_not_called()

    def test_nd_private_failure_does_not_mark_warmup_complete(self):
        pool = paged()
        pool.kv_buffer = pool.kv_buffer.as_subclass(MetadataNPUTensor)
        with npu_metadata_runtime() as (api, _, _), patch.object(
            kv, "move_kv_slots_"
        ) as move:
            with patch.object(
                api, "get_npu_format", side_effect=lambda t: 0 if t.ndim == 1 else 2
            ):
                with self.assertRaisesRegex(
                    kv.UnsupportedKVMoveLayout, "private_warmup.backing"
                ):
                    kv.warm_private_slot_move(pool)
            self.assertEqual(pool._kv_move_warmups, {})
            move.assert_not_called()
            # A pre-submission allocation error does not poison a live pool.
            kv.warm_private_slot_move(pool)
            self.assertTrue(next(iter(pool._kv_move_warmups.values())).complete)

    def test_overlap_all_dtypes_and_round_sizes(self):
        mappings = [
            ([1, 0], [0, 1]),
            ([1, 2, 3, 0], [0, 1, 2, 3]),
            ([2, 2, 2], [0, 1, 2]),
            ([4, 5], [8, 9]),
            ([0, 1], [0, 1]),
        ]
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.uint8):
            pool, gold = paged(dtype), paged(dtype)
            ws = kv.prepare_kv_move(pool, 4)
            addresses = [x.data_ptr() for x in ws.scratch]
            for sr, ds in mappings:
                for index_dtype in (torch.int32, torch.int64):
                    src, dst = torch.tensor(sr, dtype=index_dtype), torch.tensor(
                        ds, dtype=index_dtype
                    )
                    snapshot_move(kv.KVMoveLayout.from_pool(gold), src, dst)
                    kv.move_kv_slots_(ws, src, dst)
                    self.assertTrue(torch.equal(pool.kv_buffer, gold.kv_buffer))
                    self.assertEqual(addresses, [x.data_ptr() for x in ws.scratch])
            self.assertEqual(ws.counts["grow"], 1)

    def test_all_layouts_and_index_storage(self):
        variants = [
            NS(
                k_buffer=[torch.randn(16, 2, 7) for _ in range(3)],
                v_buffer=[torch.randn(16, 2, 7) for _ in range(3)],
            ),
            NS(
                k_buffer=torch.randn(3, 4, 4, 1, 7),
                v_buffer=torch.randn(3, 4, 4, 1, 5),
                index_k_buffer=torch.randn(3, 4, 4, 1, 3),
            ),
            NS(
                k_buffer=torch.randn(16, 7),
                v_buffer=torch.randn(16, 5),
                index_k_buffer=torch.randn(16, 3),
            ),
            NS(kv_buffer=[torch.randn(16, 7) for _ in range(3)]),
        ]
        src, dst = torch.tensor([1, 2, 0, 1]), torch.tensor([0, 1, 2, 3])
        for pool in variants:
            ws = kv.prepare_kv_move(pool, 4)
            gold = [b.tensor.clone() for b in ws.layout.buffers]
            kv.move_kv_slots_(ws, src, dst)
            for b, initial in zip(ws.layout.buffers, gold):
                expected = initial.clone()
                expected.index_copy_(b.axis, dst, initial.index_select(b.axis, src))
                self.assertTrue(torch.equal(expected, b.tensor))

    def test_all_gathers_precede_first_scatter(self):
        pool = NS(
            k_buffer=[torch.randn(8, 3), torch.randn(8, 3)],
            v_buffer=[torch.randn(8, 3), torch.randn(8, 3)],
        )
        ws = kv.prepare_kv_move(pool, 2)
        original_select, original_copy = torch.index_select, torch.Tensor.index_copy_
        sequence = []

        def gather(*args, **kwargs):
            self.assertIn("out", kwargs)
            sequence.append("gather")
            return original_select(*args, **kwargs)

        def scatter(*args, **kwargs):
            sequence.append("scatter")
            return original_copy(*args, **kwargs)

        with patch.object(torch, "index_select", gather), patch.object(
            torch.Tensor, "index_copy_", scatter
        ):
            kv.move_kv_slots_(ws, torch.tensor([1, 0]), torch.tensor([0, 1]))
        self.assertEqual(sequence, ["gather"] * 4 + ["scatter"] * 4)

    def test_full_validation_before_mutation(self):
        k = torch.randn(3, 4, 4, 1, 7)
        for pool in (
            NS(k_buffer=k, v_buffer=None),
            NS(k_buffer=k, v_buffer=k.clone(), index_k_buffer=torch.zeros(16)),
            NS(k_buffer=[k], v_buffer=[k, k]),
        ):
            before = k.clone()
            with self.assertRaises(kv.UnsupportedKVMoveLayout):
                kv.prepare_kv_move(pool, 3)
            self.assertTrue(torch.equal(before, k))
        ws = kv.prepare_kv_move(paged(), 2)
        for src, dst in (
            (torch.tensor([0]), torch.tensor([1, 2])),
            (torch.tensor([-1]), torch.tensor([0])),
            (torch.tensor([16]), torch.tensor([0])),
        ):
            before = ws.layout.buffers[0].tensor.clone()
            with self.assertRaises((RuntimeError, IndexError)):
                kv.move_kv_slots_(ws, src, dst)
            self.assertTrue(torch.equal(before, ws.layout.buffers[0].tensor))
            self.assertFalse(ws.unresolved)

    def test_stacked_axes_validation_and_different_feature_widths(self):
        k = torch.randn(2, 4, 2, 1, 7)
        for v, index in (
            (torch.randn(3, 4, 2, 1, 5), None),
            (torch.randn(2, 2, 4, 1, 5), None),
            (torch.randn(2, 4, 2, 1, 5), torch.randn(3, 4, 2, 1, 3)),
            (torch.randn(2, 4, 2, 1, 5), [torch.randn(2, 4, 2, 1, 3)]),
        ):
            pool = NS(k_buffer=k, v_buffer=v, index_k_buffer=index)
            with self.assertRaises(kv.UnsupportedKVMoveLayout), patch.object(
                torch, "empty"
            ) as alloc:
                kv.prepare_kv_move(pool, 2)
            alloc.assert_not_called()
        good = NS(
            k_buffer=k,
            v_buffer=torch.randn(2, 4, 2, 1, 5),
            index_k_buffer=torch.randn(2, 4, 2, 1, 3),
        )
        self.assertEqual(len(kv.prepare_kv_move(good, 2).scratch), 3)

    def test_steady_state_uses_cached_layout_and_rejects_storage_changes(self):
        pool = paged()
        with patch.object(
            kv.KVMoveLayout, "from_pool", wraps=kv.KVMoveLayout.from_pool
        ) as parse:
            ws = kv.prepare_kv_move(pool, 2)
            for _ in range(3):
                self.assertIs(ws, kv.prepare_kv_move(pool, 2))
                kv.move_kv_slots_(ws, torch.tensor([1, 0]), torch.tensor([0, 1]))
            kv.prepare_kv_move(pool, 2, domain="graph", graph=True)
            self.assertEqual(parse.call_count, 1)
        pool.kv_buffer.set_(pool.kv_buffer.clone())
        with self.assertRaises(kv.UnsupportedKVMoveLayout):
            ws.check()
        pool = NS(k_buffer=[torch.zeros(8, 3)], v_buffer=[torch.zeros(8, 3)])
        ws = kv.prepare_kv_move(pool, 2)
        pool.k_buffer.append(torch.zeros(8, 3))
        with self.assertRaises(kv.UnsupportedKVMoveLayout):
            kv.prepare_kv_move(pool, 2)

    def test_exact_alias_and_disjoint_layer_views(self):
        base = torch.randn(2, 8, 3)
        pool = NS(k_buffer=[base[0], base[1]], v_buffer=[base[0], base[1]])
        self.assertEqual(len(kv.prepare_kv_move(pool, 2).scratch), 2)
        with self.assertRaises(kv.UnsupportedKVMoveLayout):
            kv.prepare_kv_move(NS(k_buffer=base[0, :6], v_buffer=base[0, 2:]), 2)

    def test_fp8_storage_bits(self):
        pool = NS(
            k_buffer=torch.arange(64).reshape(16, 4).to(torch.float8_e4m3fn),
            v_buffer=torch.arange(64).reshape(16, 4).to(torch.float8_e4m3fn),
        )
        ws = kv.prepare_kv_move(pool, 2)
        before = pool.k_buffer.view(torch.uint8).clone()
        kv.move_kv_slots_(ws, torch.tensor([1, 0]), torch.tensor([0, 1]))
        self.assertTrue(torch.equal(pool.k_buffer.view(torch.uint8)[0], before[1]))
        self.assertEqual(ws.scratch[0].dtype, torch.uint8)

    def test_domains_capacity_and_storage_generation(self):
        pool = paged()
        a = kv.prepare_kv_move(pool, 3)
        self.assertIs(a, kv.prepare_kv_move(pool, 2))
        b = kv.prepare_kv_move(pool, 3, domain="other")
        self.assertIs(a, b)
        a.reserve(5)
        self.assertEqual(a.capacity, 8)
        self.assertEqual(a.counts["grow"], 2)
        pool.kv_buffer = pool.kv_buffer.clone()
        with self.assertRaises(kv.UnsupportedKVMoveLayout):
            a.check()

    def test_stream_isolation(self):
        pool = paged()
        with patch.object(kv, "_stream", return_value=NS(stream_id=11)):
            a = kv.prepare_kv_move(pool, 2)
        with patch.object(kv, "_stream", return_value=NS(stream_id=12)):
            b = kv.prepare_kv_move(pool, 2)
            self.assertIsNot(a, b)
            with self.assertRaisesRegex(RuntimeError, "another execution stream"):
                a.check()

    def test_ordinary_timing_failure_does_not_poison_or_repeat_move(self):
        from unittest.mock import Mock

        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            SRRoundMetrics,
        )

        metrics = SRRoundMetrics(
            "Draft", NS(Event=Mock(side_effect=RuntimeError("timing unavailable")))
        )
        pool = paged()
        ws = kv.prepare_kv_move(pool, 2, metrics=metrics)
        before = pool.kv_buffer.clone().view(2, 3, 16, 7)
        with patch.object(kv, "_portable_move", wraps=kv._portable_move) as move:
            with metrics.round():
                kv.move_kv_slots_(ws, torch.tensor([1, 0]), torch.tensor([0, 1]))
            move.assert_called_once()
        self.assertFalse(ws.unresolved)
        self.assertTrue(
            torch.equal(pool.kv_buffer.view_as(before)[:, :, 0], before[:, :, 1])
        )
        self.assertEqual(metrics.counts["kv_move_eager_calls"], 1)
        self.assertEqual(metrics.counts["kv_move_graph_calls"], 0)
        self.assertEqual(
            ws.scratch_bytes,
            sum(t.nbytes for t in ws.scratch) + ws.index_scratch.nbytes,
        )

    def test_failed_growth_keeps_previous_capacity_usable(self):
        pool = paged()
        ws = kv.prepare_kv_move(pool, 2)
        old = ws.scratch
        old_index = ws.index_scratch
        empty = torch.empty

        def fail_index(shape, **kwargs):
            if tuple(shape) == (2, 8) and kwargs.get("dtype") == torch.int64:
                raise RuntimeError("integer scratch allocation failed")
            return empty(shape, **kwargs)

        with patch.object(torch, "empty", fail_index):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                ws.reserve(8)
        self.assertIs(ws.scratch, old)
        self.assertIs(ws.index_scratch, old_index)
        self.assertEqual(ws.capacity, 2)
        kv.move_kv_slots_(
            ws,
            torch.tensor([1, 0], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
        )

    def test_graph_capacity_and_replay_accounting(self):
        pool = paged()
        metrics = NS(counts=Counter())
        ws = kv.prepare_kv_move(pool, 4, domain="graph", graph=True, metrics=metrics)
        with ws.capture_scope():
            kv.move_kv_slots_(ws, torch.tensor([1, 0]), torch.tensor([0, 1]))
        self.assertEqual(ws.counts["calls"], 0)
        ws.account(6, calls=3)
        self.assertEqual(ws.counts["calls"], 3)
        self.assertEqual(metrics.counts["kv_move_graph_calls"], 3)
        self.assertEqual(metrics.counts["kv_move_eager_calls"], 0)
        self.assertEqual(metrics.counts["kv_move_scratch_bytes"], ws.scratch_bytes)
        with self.assertRaisesRegex(RuntimeError, "cannot grow"):
            ws.reserve(5)
        self.assertIsNot(ws, kv.prepare_kv_move(pool, 8))

    def test_tree_live_mapping_strides_padding_and_growth(self):
        pool = paged()
        ws = kv.prepare_kv_move(pool, 12)
        backing = torch.tensor(
            [
                [1, 99, 2, 99, 3, 99, 0, 99],
                [5, 99, 6, 99, 7, 99, 0, 99],
                [9, 99, 10, 99, 11, 99, 0, 99],
            ]
        )
        slots = backing[:, ::2]
        active = torch.tensor([True, True, True, False])
        for parents in (
            torch.tensor([1, 2, 0, 3]),
            torch.tensor([2, 2, 2, 3]),
            torch.tensor([0, 2, 1, 3]),
        ):
            initial = pool.kv_buffer.clone().view(2, 3, 16, 7)
            expected = initial.clone()
            for step in range(3):
                for row in range(3):
                    expected[:, :, slots[step, row]].copy_(
                        initial[:, :, slots[step, parents[row]]]
                    )
            kv.remap_tree_kv_(ws, slots, parents, 3, active)
            self.assertTrue(torch.equal(expected, pool.kv_buffer.view_as(expected)))
        ws.reserve(32)
        self.assertEqual(ws.tree_indices.shape[1], 32)
        kv.remap_tree_kv_(ws, slots, torch.arange(4), 3, active)
        kv.remap_tree_kv_(ws, slots.to(torch.int32), torch.arange(4), 3, active)
        self.assertEqual(ws.tree_indices.dtype, torch.int32)

    def test_tree_bad_metadata_fails_before_copy(self):
        ws = kv.prepare_kv_move(paged(), 4)
        for slots, parents, depth in (
            (torch.zeros(2, 2, dtype=torch.int64), torch.tensor([0]), 1),
            (torch.zeros(2, 2, dtype=torch.int64), torch.tensor([0, 1]), 3),
        ):
            with self.assertRaises(RuntimeError):
                kv.remap_tree_kv_(ws, slots, parents, depth)
        self.assertEqual(ws.counts["calls"], 0)

    def test_submitted_failure_poison_and_no_second_move(self):
        pool = paged()
        ws = kv.prepare_kv_move(pool, 2)
        with patch.object(
            torch.Tensor, "index_copy_", side_effect=RuntimeError("scatter failed")
        ) as spy:
            with self.assertRaises(kv.KVMoveSubmittedError):
                kv.move_kv_slots_(ws, torch.tensor([1, 0]), torch.tensor([0, 1]))
            first = spy.call_count
            with self.assertRaises(kv.KVMoveSubmittedError):
                kv.move_kv_slots_(ws, torch.tensor([1, 0]), torch.tensor([0, 1]))
            self.assertEqual(spy.call_count, first)
        self.assertIsNotNone(ws.hold)
        with self.assertRaises(kv.KVMoveSubmittedError):
            kv.prepare_kv_move(pool, 3, domain="other")

    def test_retired_consumer_query_failure_retains_storage(self):
        ws = kv.prepare_kv_move(paged(), 2)
        old = ws.scratch
        bad_event = NS(
            query=lambda: (_ for _ in ()).throw(RuntimeError("event failed"))
        )
        ws.retired = [(bad_event, old)]
        with self.assertRaises(kv.KVMoveSubmittedError):
            ws.reserve(4)
        self.assertIs(ws.retired[0][1], old)
        self.assertIs(ws.scratch, old)

    def test_memory_accounting_includes_indices_and_pending_retirement(self):
        metrics = NS(counts=Counter())
        pool = paged()
        ws = kv.prepare_kv_move(pool, 2, metrics=metrics)
        self.assertEqual(
            ws.scratch_bytes,
            sum(t.nbytes for t in ws.scratch) + ws.index_scratch.nbytes,
        )
        old = ws.scratch
        size = ws.scratch_bytes
        event = NS(query=lambda: False)
        ws.retired = [(event, old, ws.index_scratch)]
        ws._retired_sizes[id(event)] = size
        ws.retired_bytes = size
        ws.reserve(8)
        self.assertEqual(metrics.counts["kv_move_retired_bytes"], size)
        peak = ws.scratch_bytes + size
        self.assertEqual(metrics.counts["kv_move_peak_bytes"], peak)
        event.query = lambda: True
        ws.reserve(8)
        self.assertEqual(metrics.counts["kv_move_retired_bytes"], 0)
        self.assertEqual(metrics.counts["kv_move_peak_bytes"], peak)
        kv.prepare_kv_move(pool, 8)
        self.assertIsNone(ws.metrics)
        ws.reserve(16)
        self.assertGreater(pool._kv_move_peak_bytes, peak)

    def test_no_new_workspace_or_growth_inside_capture(self):
        pool = paged()
        ws = kv.prepare_kv_move(pool, 2)
        with patch.object(kv, "_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "before graph capture"):
                kv.prepare_kv_move(pool, 2, domain="new", graph=True)
            with self.assertRaisesRegex(RuntimeError, "cannot grow"):
                ws.reserve(3)

    def test_kernel_pointers_keep_mode_flags(self):
        src = torch.arange(4)
        dst = src + 1
        slots = torch.arange(8).reshape(2, 4)
        parents = torch.tensor([1, 0, 2, 3])
        tree, has_active, strides, pointers = kv.kernel_pointer_launch(
            src, dst, None, None, None
        )
        self.assertFalse(tree)
        self.assertFalse(has_active)
        self.assertEqual(strides[:2], (src.stride(0), dst.stride(0)))
        self.assertTrue(all(torch.is_tensor(pointer) for pointer in pointers))
        self.assertTrue(all(pointer is src or pointer is dst for pointer in pointers))
        tree, has_active, strides, pointers = kv.kernel_pointer_launch(
            None, None, slots, parents, None
        )
        self.assertTrue(tree)
        self.assertFalse(has_active)
        self.assertEqual(strides[2:4], (slots.stride(0), slots.stride(1)))
        self.assertIs(pointers[2], slots)
        self.assertIs(pointers[3], parents)
        self.assertTrue(all(torch.is_tensor(pointer) for pointer in pointers))

    def test_npu_paged_dtype_rejection(self):
        with self.assertRaises(kv.UnsupportedKVMoveLayout):
            kv.require_npu_paged_dtype(torch.int8)
        kv.require_npu_paged_dtype(torch.float16)
        kv.require_npu_paged_dtype(torch.uint8)

    def test_submitted_error_is_device_context(self):
        from sglang.srt.speculative.standalone_remote.sr_align import (
            is_device_context_error,
        )

        self.assertTrue(is_device_context_error(kv.KVMoveSubmittedError("submitted")))
        self.assertFalse(is_device_context_error(kv.UnsupportedKVMoveLayout("layout")))

    def test_paged_launch_chunks_stay_under_grid_limit(self):
        groups = 56
        width = 1024
        tiles = (width + 255) // 256
        chunks = kv.paged_launch_chunks(381, groups, width)
        self.assertGreater(len(chunks), 1)
        covered = {}
        for slot_base, slot_count, tile_base, tile_count in chunks:
            self.assertLess(slot_count * groups * tile_count, 65536)
            for slot in range(slot_base, slot_base + slot_count):
                covered.setdefault(slot, set()).update(
                    range(tile_base, tile_base + tile_count)
                )
        self.assertEqual(set(covered), set(range(381)))
        for cols in covered.values():
            self.assertEqual(cols, set(range(tiles)))
        self.assertEqual(kv.paged_launch_chunks(36, groups, width), [(0, 36, 0, tiles)])
        with self.assertRaises(RuntimeError):
            kv.paged_launch_chunks(1, 65536, width)

    def test_private_warmup_does_not_need_live_storage(self):
        pool = paged(torch.bfloat16, width=259)
        before = pool.kv_buffer.clone()
        with patch.object(kv, "move_kv_slots_", wraps=kv.move_kv_slots_) as moves:
            kv.warm_private_slot_move(pool, 3)
            self.assertEqual(moves.call_count, 2)
            kv.warm_private_slot_move(pool, 381)
            self.assertEqual(moves.call_count, 2)
            private = next(iter(pool._kv_move_warmups.values()))
            warmed_workspace = moves.call_args.args[0]
            self.assertEqual(
                warmed_workspace.layout.buffers[0].tensor.shape, (2, 3, 3, 259)
            )
            self.assertEqual(
                warmed_workspace.layout.buffers[0].tensor.dtype, torch.bfloat16
            )
            self.assertIsNone(private.workspace)
            self.assertTrue(private.complete)
            kv.warm_private_slot_move(pool, 1, index_dtype=torch.int32, src_stride=2)
            self.assertEqual(moves.call_count, 4)
            kv.warm_private_slot_move(pool, dst_index_dtype=torch.int32)
            self.assertEqual(moves.call_count, 6)
            self.assertEqual(moves.call_args.args[1].dtype, torch.int64)
            self.assertEqual(moves.call_args.args[2].dtype, torch.int32)
        self.assertTrue(torch.equal(pool.kv_buffer, before))
        self.assertEqual(len(pool._kv_move_warmups), 3)
        other = paged(torch.float16, width=67)
        kv.warm_private_slot_move(other)
        self.assertEqual(len(other._kv_move_warmups), 1)

    def test_private_warmup_failure_retains_storage_and_refuses_retry(self):
        pool = paged()
        with patch.object(
            kv, "move_kv_slots_", side_effect=RuntimeError("warm failed")
        ) as move:
            with self.assertRaisesRegex(RuntimeError, "warm failed"):
                kv.warm_private_slot_move(pool)
            with self.assertRaises(kv.KVMoveSubmittedError):
                kv.warm_private_slot_move(pool)
            self.assertEqual(move.call_count, 1)
        entry = next(iter(pool._kv_move_warmups.values()))
        self.assertFalse(entry.complete)
        self.assertTrue(entry.workspace.unresolved)

    def test_chunk_launches_have_runtime_scalars_and_snapshot_order(self):
        path = Path(kv.__file__).with_name("sr_kv_copy_kernels.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        kernel = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_paged_move"
        )
        dynamic = {"SlotBase", "TileBase", "ValidSlots", "PoolSlots", "Capacity"}
        self.assertEqual(
            set(ast.literal_eval(kernel.decorator_list[0].keywords[0].value)), dynamic
        )
        self.assertTrue(
            all(a.annotation is None for a in kernel.args.args if a.arg in dynamic)
        )
        funcs = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("_launch", "move")
        ]
        calls = []

        class Kernel:
            def __getitem__(self, grid):
                def launch(
                    pool,
                    scratch,
                    src,
                    dst,
                    slots,
                    parents,
                    active,
                    slot_base,
                    tile_base,
                    valid,
                    **args,
                ):
                    calls.append((args["Scatter"], grid))
                    lo, hi = tile_base * 256, min(
                        (tile_base + grid[1]) * 256, args["Width"]
                    )
                    source = scratch if args["Scatter"] else pool
                    target = pool if args["Scatter"] else scratch
                    src_idx = (
                        torch.arange(slot_base, slot_base + valid)
                        if args["Scatter"]
                        else src[slot_base : slot_base + valid]
                    )
                    dst_idx = (
                        dst[slot_base : slot_base + valid]
                        if args["Scatter"]
                        else torch.arange(slot_base, slot_base + valid)
                    )
                    target[..., lo:hi].index_copy_(
                        2, dst_idx, source[..., lo:hi].index_select(2, src_idx)
                    )

                return launch

        ns = {"_paged_move": Kernel()}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), "exec"), ns)
        for mapping in (torch.roll(torch.arange(17), 6), torch.tensor([16] * 17)):
            pool = torch.randn(2, 3, 20, 259)
            initial = pool.clone()
            src, dst = mapping, torch.arange(17)
            expected = initial.clone()
            expected.index_copy_(2, dst, initial.index_select(2, src))
            calls.clear()
            with patch.object(kv, "_PAGED_GRID_LIMIT", 63):
                ns["move"](pool, torch.empty(2, 3, 32, 259), src, dst)
            phases = [scatter for scatter, _ in calls]
            self.assertEqual(phases, sorted(phases))
            self.assertGreater(len(calls), 2)
            torch.testing.assert_close(pool, expected)
        calls.clear()
        ns["move"](
            pool, torch.empty(2, 3, 32, 259), torch.tensor([1, 0]), torch.tensor([0, 1])
        )
        self.assertEqual(len(calls), 2)
        self.assertNotIn("fill_", path.read_text(encoding="utf-8"))

    def test_production_old_methods_removed(self):
        root = Path(__file__).resolve().parents[4] / "python/sglang/srt"
        names = (
            "copy_kv_pool_by_slot",
            "copy_paged_kv_buffer_by_slot",
            "copy_mha_kv_by_slot",
            "_copy_token_slots",
            "SGLANG_NPU_SR_TREE_KV_SCRATCH",
        )
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for name in names:
                self.assertNotIn(name, text, str(path))


if __name__ == "__main__":
    unittest.main()
