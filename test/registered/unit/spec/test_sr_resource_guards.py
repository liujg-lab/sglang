"""CPU fault injection for SR upload and quarantine."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative.standalone_remote import sr_transfer_staging as staging
from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
    KVQuarantine,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]


class Event:
    def __init__(self):
        self.done = True
        self.fail_record = self.fail_wait = self.fail_query = False
        self.records = 0

    def record(self, stream):
        self.records += 1
        self.done = False
        if self.fail_record:
            raise RuntimeError("record")

    def query(self):
        if self.fail_query:
            raise RuntimeError("query")
        return self.done

    def synchronize(self):
        if self.fail_wait:
            raise RuntimeError("wait")
        self.done = True


class UploadTest(unittest.TestCase):
    def setUp(self):
        self.owner = NS()
        self.stream = NS(cuda_stream=7)
        self.event = Event()
        self.module = NS(
            current_device=lambda: 1,
            current_stream=lambda dev: self.stream,
            Event=lambda: self.event,
        )
        empty = torch.empty

        def cpu_empty(*args, **kwargs):
            kwargs.pop("device", None)
            return empty(*args, **kwargs)

        for ctx in (
            patch.object(torch, "get_device_module", return_value=self.module),
            patch.object(torch, "empty", side_effect=cpu_empty),
            patch.object(torch.Tensor, "is_pinned", return_value=True),
            patch.object(
                staging,
                "alloc_host",
                side_effect=lambda shape, dtype, dev: (empty(shape, dtype=dtype), True),
            ),
        ):
            ctx.start()
            self.addCleanup(ctx.stop)

    def upload(self, values=(1, 2), device="cuda"):
        return staging.owned_ints(self.owner, "rows", values, torch.int64, device)

    def test_owns_outputs_and_reuses_host(self):
        first = self.upload()
        slot = next(iter(self.owner._sr_int_staging.values()))
        host = slot["host"]
        self.upload((3, 4))
        self.assertEqual(first.tolist(), [1, 2])
        self.assertIs(host, slot["host"])

    def test_record_failure_keeps_buffers_and_blocks_reuse(self):
        self.event.fail_record = True
        with self.assertRaises(staging.SRTransferUnresolved) as caught:
            self.upload()
        self.assertEqual(str(caught.exception.__cause__), "record")
        slot = next(iter(self.owner._sr_int_staging.values()))
        self.assertIsNotNone(slot["holds"])
        with self.assertRaises(staging.SRTransferUnresolved):
            self.upload((8, 9))
        self.assertEqual(slot["host"][:2].tolist(), [1, 2])

    def test_copy_failure_is_not_retried(self):
        with patch.object(
            torch.Tensor, "copy_", side_effect=RuntimeError("copy")
        ) as copy:
            with self.assertRaises(staging.SRTransferUnresolved):
                self.upload()
            with self.assertRaises(staging.SRTransferUnresolved):
                self.upload()
            self.assertEqual(copy.call_count, 1)

    def test_event_creation_is_before_copy(self):
        self.module.Event = lambda: (_ for _ in ()).throw(RuntimeError("create"))
        with patch.object(
            torch.Tensor, "copy_", side_effect=AssertionError("submitted")
        ):
            with self.assertRaisesRegex(RuntimeError, "create"):
                self.upload()
        self.assertFalse(next(iter(self.owner._sr_int_staging.values()))["unresolved"])

    def test_wait_failure_keeps_old_host(self):
        self.upload()
        self.event.fail_wait = True
        with self.assertRaises(staging.SRTransferUnresolved):
            self.upload((3, 4))
        self.assertEqual(
            next(iter(self.owner._sr_int_staging.values()))["host"][:2].tolist(), [1, 2]
        )

    def test_device_and_stream_isolation(self):
        self.upload(device="cuda:0")
        self.upload(device="cuda:1")
        self.stream.cuda_stream = 9
        self.upload(device="cuda:1")
        self.assertEqual(len(self.owner._sr_int_staging), 3)

    def test_unpinned_copy_is_synchronous(self):
        with patch.object(torch.Tensor, "is_pinned", return_value=False), patch.object(
            torch.Tensor, "copy_", autospec=True
        ) as copy:
            self.upload()
            self.assertFalse(copy.call_args.kwargs["non_blocking"])
        self.assertEqual(self.event.records, 0)


class QuarantineTest(unittest.TestCase):
    def test_full_reset_confirms_before_invalidating_all_retirement_records(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_draft_scheduler_mixin.py"
        )
        fn = next(
            n
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name == "_sr_reset_scheduler_caches"
        )
        env = dict(
            torch=torch,
            __name__=__name__,
            __package__="sglang.srt.speculative.standalone_remote.drafter",
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                str(path),
                "exec",
            ),
            env,
        )
        for fails in (False, True):
            calls = []
            pending, holds = [object()], [object()]
            owner = NS(
                device="cuda:1",
                token_to_kv_pool_allocator=NS(
                    clear=lambda: calls.append("reset"),
                    _sr_tree_lease_pending_frees=pending,
                ),
                _sr_pending_copy_holds=holds,
            )
            bag = quarantine(owner)
            bag.retain(object(), [1, 2])

            def synchronize(device):
                self.assertEqual(device, "cuda:1")
                calls.append("confirm")
                if fails:
                    raise RuntimeError("device failed")

            with patch.object(
                torch, "get_device_module", return_value=NS(synchronize=synchronize)
            ):
                if fails:
                    with self.assertRaises(staging.SRTransferUnresolved):
                        env["_sr_reset_scheduler_caches"](owner)
                    self.assertEqual(calls, ["confirm"])
                    self.assertTrue(bag.records and pending and holds)
                else:
                    env["_sr_reset_scheduler_caches"](owner)
                    self.assertEqual(calls, ["confirm", "reset"])
                    self.assertFalse(bag.records or pending or holds)

    def test_session_wipe_keeps_http_and_quarantines_unknown_resources(self):
        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_draft_scheduler_mixin.py"
        )
        fn = next(
            n
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name == "_sr_wipe_all"
        )
        env = dict(logger=logging.getLogger(__name__))
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                str(path),
                "exec",
            ),
            env,
        )
        for unknown in (False, True):
            with self.subTest(unknown=unknown):
                req, http = object(), object()
                state = NS(req_id="r", req_object=req)
                released, held = [], []

                def forbidden():
                    raise AssertionError("must not reset shared pools")

                owner = NS(
                    sr_state=NS(clear=lambda: [state]),
                    sr_waiting=[req, http],
                    _sr_int_staging={"slot": {"unresolved": unknown}},
                    _sr_finish_rid=lambda rid, state: released.append(state.req_object),
                    _sr_abandon_poisoned_kv=held.append,
                    _sr_is_http_req=lambda r: r is http,
                    _sr_http_alive=lambda: True,
                    _sr_reset_scheduler_caches=forbidden,
                    sr_server=None,
                )
                env["_sr_wipe_all"](owner)
                self.assertEqual(released, [] if unknown else [req])
                self.assertEqual(held, [req] if unknown else [])
                self.assertIn(http, owner.sr_waiting)

    def test_wipe_without_http_resets_quarantine_only_after_sync(self):
        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_draft_scheduler_mixin.py"
        )
        names = {"_sr_wipe_all", "_sr_reset_scheduler_caches"}
        funcs = [
            n
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        env = dict(
            logger=logging.getLogger(__name__),
            torch=torch,
            __name__=__name__,
            __package__="sglang.srt.speculative.standalone_remote.drafter",
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=funcs, type_ignores=[])
                ),
                str(path),
                "exec",
            ),
            env,
        )
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        for fails in (False, True):
            with self.subTest(fails=fails):
                freed = []
                cleared = []
                owner = NS(
                    device="cuda:1",
                    sr_state=NS(clear=lambda: []),
                    sr_waiting=[],
                    draft_paused_reqs=[],
                    _sr_is_http_req=lambda r: False,
                    _sr_http_alive=lambda: False,
                    sr_server=None,
                    sr_tree_leases=NS(
                        release_all=lambda alloc: freed.append("release_all")
                    ),
                    token_to_kv_pool_allocator=NS(
                        clear=lambda: cleared.append("clear"),
                        free=lambda page: freed.append(page),
                    ),
                )
                owner._sr_reset_scheduler_caches = env[
                    "_sr_reset_scheduler_caches"
                ].__get__(owner)
                quarantine(owner).retain(object(), ["stale-page"])

                def synchronize(device):
                    if fails:
                        raise RuntimeError("device failed")

                with patch.object(
                    torch,
                    "get_device_module",
                    return_value=NS(synchronize=synchronize),
                ):
                    if fails:
                        with self.assertRaises(staging.SRTransferUnresolved):
                            env["_sr_wipe_all"](owner)
                        self.assertEqual(cleared, [])
                        self.assertTrue(owner._sr_kv_quarantine.records)
                        self.assertTrue(owner._sr_device_poisoned)
                    else:
                        env["_sr_wipe_all"](owner)
                        self.assertEqual(cleared, ["clear"])
                        self.assertFalse(owner._sr_kv_quarantine.records)
                        self.assertNotIn("stale-page", freed)
                        self.assertNotIn("release_all", freed)

    def test_abandon_records_consumer_event_or_poisons_without_free(self):
        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
        )
        wanted = {
            "_remember_abandoned_lease",
            "_capture_abandon_event",
            "_capture_abandoned_spec_image",
        }
        funcs = [
            n
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name in wanted
        ]
        env = dict(
            __name__="sglang.srt.speculative.standalone_remote.drafter.sr_tree_drafter",
            __package__="sglang.srt.speculative.standalone_remote.drafter",
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=funcs, type_ignores=[])
                ),
                str(path),
                "exec",
            ),
            env,
        )
        from sglang.srt.speculative.standalone_remote.drafter import (
            sr_tree_kv_lease,
        )

        lease = {"page_slots": []}
        done = Event()
        scheduler = NS(_sr_device_poisoned=False)
        drafter = NS(scheduler=scheduler, device=NS(type="cuda"))
        drafter._remember_abandoned_lease = env["_remember_abandoned_lease"].__get__(
            drafter
        )
        drafter._capture_abandon_event = env["_capture_abandon_event"].__get__(drafter)
        drafter._capture_abandoned_spec_image = env[
            "_capture_abandoned_spec_image"
        ].__get__(drafter)
        with patch.object(sr_tree_kv_lease, "record_device_event", return_value=done):
            drafter._remember_abandoned_lease(lease)
        self.assertIs(lease["completion_event"], done)
        self.assertIs(drafter._sr_kv_quarantine.records[0].event, done)
        self.assertFalse(scheduler._sr_device_poisoned)

        failed = {"page_slots": []}
        poisoned = NS(_sr_device_poisoned=False)
        other = NS(scheduler=poisoned, device=NS(type="cuda"))
        other._remember_abandoned_lease = env["_remember_abandoned_lease"].__get__(other)
        other._capture_abandon_event = env["_capture_abandon_event"].__get__(other)
        other._capture_abandoned_spec_image = env[
            "_capture_abandoned_spec_image"
        ].__get__(other)

        def explode(*args, **kwargs):
            raise RuntimeError("cannot record")

        with patch.object(sr_tree_kv_lease, "record_device_event", side_effect=explode):
            other._remember_abandoned_lease(failed)
        self.assertTrue(poisoned._sr_device_poisoned)
        self.assertIsNone(other._sr_kv_quarantine.records[0].event)
        self.assertNotIn("completion_event", failed)
        self.assertEqual(other._sr_kv_quarantine.reclaim(NS(free=lambda u: None)), 0)

    def test_partial_release_failure_never_retries_a_unit(self):
        bag, event, calls = KVQuarantine(), Event(), []
        owner = object()
        record = bag.retain(owner, ["a", "b"], event)

        def free(unit):
            calls.append(unit)
            if unit == "b":
                raise RuntimeError("allocator submitted")

        with self.assertRaises(staging.SRTransferUnresolved):
            bag.reclaim(NS(free=free))
        self.assertEqual(record.states, ["done", "unknown"])
        self.assertEqual(bag.reclaim(NS(free=free)), 0)
        self.assertEqual(calls, ["a", "b"])
        self.assertIs(record.owner, owner)

    def test_unknown_release_blocks_the_rest_of_the_record(self):
        bag, event, calls = KVQuarantine(), Event(), []
        record = bag.retain(object(), ["a", "b", "c"], event)

        def free(unit):
            calls.append(unit)
            if unit == "b":
                raise RuntimeError("allocator submitted")

        with self.assertRaises(staging.SRTransferUnresolved):
            bag.reclaim(NS(free=free))
        self.assertEqual(record.states, ["done", "unknown", "unsubmitted"])
        self.assertIsInstance(record.release_error, RuntimeError)
        owner = NS(token_to_kv_pool_allocator=NS(free=free), _sr_kv_quarantine=bag)
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            has_unresolved_resources,
        )

        self.assertTrue(has_unresolved_resources(owner))
        self.assertTrue(has_unresolved_resources(owner))
        self.assertTrue(has_unresolved_resources(owner, include_quarantine=False))
        self.assertEqual(calls, ["a", "b"])
        self.assertEqual(record.states, ["done", "unknown", "unsubmitted"])
        self.assertIn(record, bag.records)
        with self.assertRaises(staging.SRTransferUnresolved):
            bag.after_allocator_reset(completion_confirmed=False)
        self.assertIn(record, bag.records)
        bag.after_allocator_reset(completion_confirmed=True)
        self.assertFalse(bag.records)

    def test_completed_event_does_not_block_other_requests(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            has_unresolved_resources,
            quarantine,
        )

        calls = []
        owner = NS(token_to_kv_pool_allocator=NS(free=calls.append))
        quarantine(owner).retain(object(), ["page"], Event())
        self.assertFalse(has_unresolved_resources(owner))
        self.assertEqual(calls, ["page"])
        self.assertFalse(owner._sr_kv_quarantine.records)

    def test_missing_event_blocks_without_freeing(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            has_unresolved_resources,
            quarantine,
        )

        calls = []
        owner = NS(token_to_kv_pool_allocator=NS(free=calls.append))
        quarantine(owner).retain(object(), ["page"])
        self.assertTrue(has_unresolved_resources(owner))
        self.assertEqual(calls, [])
        self.assertEqual(len(owner._sr_kv_quarantine.records), 1)

    def test_finish_check_ignores_other_requests_records(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            has_unresolved_resources,
            quarantine,
        )

        calls = []
        drafter = NS()
        owner = NS(
            token_to_kv_pool_allocator=NS(free=calls.append),
            sr_tree_drafter=drafter,
        )
        quarantine(owner).retain(object(), ["page"])
        quarantine(drafter).retain(object(), ["tree-page"])
        self.assertTrue(has_unresolved_resources(owner))
        self.assertFalse(has_unresolved_resources(owner, include_quarantine=False))
        self.assertEqual(calls, [])

    def test_finish_check_still_sees_global_faults(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            has_unresolved_resources,
        )

        poisoned = NS(_sr_device_poisoned=True)
        staging_slot = NS(_sr_int_staging={"rows": {"unresolved": True}})
        copying = NS(sr_tree_leases=NS(_leases={"r": NS(copy_unresolved=True)}))
        pending = NS(
            token_to_kv_pool_allocator=NS(
                _sr_tree_lease_pending_frees=(NS(copy_unresolved=True),)
            )
        )
        for owner in (poisoned, staging_slot, copying, pending):
            self.assertTrue(has_unresolved_resources(owner, include_quarantine=False))

    def test_unknown_or_pending_consumer_cannot_free(self):
        bag, event = KVQuarantine(), Event()
        bag.retain(object(), [1])
        event.done = False
        bag.retain(object(), [2], event)
        with patch.object(NS(), "free", create=True) as free:
            self.assertEqual(bag.reclaim(NS(free=free)), 0)
            free.assert_not_called()

    def test_confirmed_event_reclaims_once_and_reset_invalidates(self):
        bag, calls = KVQuarantine(), []
        bag.retain(object(), [1, 2], Event())
        self.assertEqual(bag.reclaim(NS(free=calls.append)), 1)
        bag.retain(object(), [3])
        with self.assertRaises(staging.SRTransferUnresolved):
            bag.after_allocator_reset(completion_confirmed=False)
        bag.after_allocator_reset(completion_confirmed=True)
        bag.reclaim(NS(free=calls.append))
        self.assertEqual(calls, [1, 2])
        self.assertEqual(bag.generation, 1)

    def test_poisoned_finish_does_not_call_device_cleanup(self):
        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_draft_scheduler_mixin.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            "_sr_finish_rid",
            "_sr_abandon_poisoned_kv",
            "_sr_release_poisoned_req_row",
        }
        funcs = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        metrics = NS(counts={})
        env = dict(
            __name__="sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin",
            __package__="sglang.srt.speculative.standalone_remote.drafter",
            get_sr_round_metrics=lambda *a: metrics,
            logger=logging.getLogger(__name__),
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[
                            ast.parse("from __future__ import annotations").body[0],
                            *funcs,
                        ],
                        type_ignores=[],
                    )
                ),
                str(path),
                "exec",
            ),
            env,
        )

        def forbidden(*a, **kw):
            raise AssertionError("device cleanup")

        req = NS(
            rid="r",
            req_pool_idx=3,
            kv_allocated_len=8,
            kv_committed_len=7,
            finished=lambda: True,
            multimodal_inputs=object(),
        )
        row_pool = NS(freed=[])

        def free_row(request):
            row_pool.freed.append(request.req_pool_idx)
            request.req_pool_idx = None

        row_pool.free = free_row
        owner = NS(
            _sr_device_poisoned=True,
            _sr_remove_req=forbidden,
            _sr_release_tree_lease=forbidden,
            sr_state=NS(delete=lambda rid: NS(req_object=req)),
            sr_tree_leases=NS(pop=lambda rid: "lease"),
            req_to_token_pool=row_pool,
            token_to_kv_pool_allocator=NS(
                free=lambda *_a, **_k: (_ for _ in ()).throw(
                    AssertionError("device cleanup")
                )
            ),
        )
        owner._sr_release_poisoned_req_row = env[
            "_sr_release_poisoned_req_row"
        ].__get__(owner)
        owner._sr_abandon_poisoned_kv = env["_sr_abandon_poisoned_kv"].__get__(owner)
        env["_sr_finish_rid"](owner, "r")
        self.assertIs(owner._sr_kv_quarantine.records[0].owner[0], req)
        self.assertEqual(metrics.counts["abandoned_kv_slots"], 8)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(row_pool.freed, [3])

    def test_abandon_matches_dict_owner_without_indexing(self):
        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_draft_scheduler_mixin.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        funcs = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name in {"_sr_abandon_poisoned_kv", "_sr_release_poisoned_req_row"}
        ]
        metrics = NS(counts={})
        env = dict(
            __name__="sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin",
            __package__="sglang.srt.speculative.standalone_remote.drafter",
            get_sr_round_metrics=lambda *a: metrics,
            logger=logging.getLogger(__name__),
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[
                            ast.parse("from __future__ import annotations").body[0],
                            *funcs,
                        ],
                        type_ignores=[],
                    )
                ),
                str(path),
                "exec",
            ),
            env,
        )
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        req = NS(rid="r", req_pool_idx=1, kv_allocated_len=2, kv_committed_len=2)
        freed = []
        popped = []
        row_pool = NS(freed=[])

        def free_row(request):
            row_pool.freed.append(request.req_pool_idx)
            request.req_pool_idx = None

        row_pool.free = free_row
        owner = NS(
            token_to_kv_pool_allocator=NS(free=freed.append),
            req_to_token_pool=row_pool,
            sr_tree_leases=NS(pop=lambda rid: popped.append(rid) or "lease"),
        )
        quarantine(owner).retain({"reqs": [req]}, [torch.tensor([8])])
        owner._sr_release_poisoned_req_row = env[
            "_sr_release_poisoned_req_row"
        ].__get__(owner)
        owner._sr_abandon_poisoned_kv = env["_sr_abandon_poisoned_kv"].__get__(owner)
        owner._sr_abandon_poisoned_kv(req)
        self.assertEqual(freed, [])
        self.assertEqual(popped, ["r"])
        self.assertEqual(len(owner._sr_kv_quarantine.records), 1)
        self.assertIsInstance(owner._sr_kv_quarantine.records[0].owner, dict)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(row_pool.freed, [1])
        self.assertEqual(metrics.counts["abandoned_kv_slots"], 2)

    def test_later_finish_releases_after_an_abandoned_request(self):
        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_draft_scheduler_mixin.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            "_sr_finish_rid",
            "_sr_abandon_poisoned_kv",
            "_sr_release_poisoned_req_row",
        }
        funcs = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        metrics = NS(counts={})
        env = dict(
            __name__="sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin",
            __package__="sglang.srt.speculative.standalone_remote.drafter",
            get_sr_round_metrics=lambda *a: metrics,
            logger=logging.getLogger(__name__),
            FINISH_ABORT=lambda msg: msg,
            release_mm_resources=lambda mm: None,
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[
                            ast.parse("from __future__ import annotations").body[0],
                            *funcs,
                        ],
                        type_ignores=[],
                    )
                ),
                str(path),
                "exec",
            ),
            env,
        )
        first = NS(
            rid="r",
            req_pool_idx=3,
            kv_allocated_len=8,
            kv_committed_len=7,
            finished=lambda: True,
            multimodal_inputs=None,
        )
        second = NS(
            rid="s",
            req_pool_idx=4,
            kv_allocated_len=2,
            kv_committed_len=2,
            finished=lambda: True,
            multimodal_inputs=None,
        )
        released = []
        removed = []
        states = {"r": NS(req_object=first), "s": NS(req_object=second)}
        owner = NS(
            _sr_device_poisoned=True,
            _sr_remove_req=removed.append,
            _sr_release_tree_lease=lambda rid: None,
            sr_state=NS(delete=lambda rid: states.pop(rid)),
            sr_tree_leases=NS(pop=lambda rid: "lease"),
            sr_kv=NS(release_all_kv_for_finished_req=released.append),
        )
        owner._sr_release_poisoned_req_row = env[
            "_sr_release_poisoned_req_row"
        ].__get__(owner)
        owner._sr_abandon_poisoned_kv = env["_sr_abandon_poisoned_kv"].__get__(owner)
        env["_sr_finish_rid"](owner, "r")
        owner._sr_device_poisoned = False
        env["_sr_finish_rid"](owner, "s")
        self.assertEqual(removed, [second])
        self.assertEqual(released, [second])
        self.assertEqual(len(owner._sr_kv_quarantine.records), 1)
        self.assertIs(owner._sr_kv_quarantine.records[0].owner[0], first)

    def _load_reclaim(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
        )
        names = {
            "_restore_tree_mapping",
            "_restore_abandoned_mapping",
            "_partition_abandoned_pages",
            "_reclaim_abandoned_leases",
        }
        funcs = [
            n
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        env = dict(
            torch=torch,
            SimpleNamespace=NS,
            __name__="sglang.srt.speculative.standalone_remote.drafter.sr_tree_drafter",
            __package__="sglang.srt.speculative.standalone_remote.drafter",
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[
                            ast.parse("from __future__ import annotations").body[0],
                            *funcs,
                        ],
                        type_ignores=[],
                    )
                ),
                str(path),
                "exec",
            ),
            env,
        )
        return env, quarantine

    def _drafter(self, mapping, free):
        env, _quarantine = self._load_reclaim()
        drafter = NS(
            req_to_token_pool=NS(req_to_token=mapping),
            token_to_kv_pool_allocator=NS(free=free),
            scheduler=None,
            draft_attn_backend=NS(),
        )
        for name in (
            "_restore_tree_mapping",
            "_restore_abandoned_mapping",
            "_partition_abandoned_pages",
            "_reclaim_abandoned_leases",
        ):
            setattr(drafter, name, env[name].__get__(drafter))
        return drafter

    def test_completed_abandon_restores_mapping_before_free(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        mapping = torch.tensor([[4, 9, 9, 5]], dtype=torch.int64)
        pages = torch.tensor([9, 9], dtype=torch.int64)
        seen = []
        freed = []

        def free(unit):
            seen.append(mapping[0, 1:3].tolist())
            freed.append(unit)

        drafter = self._drafter(mapping, free)
        lease = {
            "reqs": [NS(req_pool_idx=0)],
            "starts": [1],
            "mapping_snaps": [torch.tensor([1, 2], dtype=torch.int64)],
            "spec_snaps": [torch.tensor([9, 9], dtype=torch.int64)],
            "page_slots": [pages],
        }
        quarantine(drafter).retain(lease, lease["page_slots"], Event())
        self.assertEqual(drafter._reclaim_abandoned_leases(), 1)
        self.assertEqual(mapping[0].tolist(), [4, 1, 2, 5])
        self.assertEqual(seen, [[1, 2]])
        self.assertEqual(len(freed), 1)
        self.assertIs(freed[0], pages)
        self.assertEqual(drafter._reclaim_abandoned_leases(), 0)
        self.assertEqual(freed, [pages])
        self.assertFalse(drafter._sr_kv_quarantine.records)

    def test_missing_abandon_event_does_not_restore_or_free(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        mapping = torch.tensor([[4, 9, 9, 5]], dtype=torch.int64)
        pages = torch.tensor([9, 9], dtype=torch.int64)
        freed = []
        drafter = self._drafter(mapping, freed.append)
        lease = {
            "reqs": [NS(req_pool_idx=0)],
            "starts": [1],
            "mapping_snaps": [torch.tensor([1, 2], dtype=torch.int64)],
            "page_slots": [pages],
        }
        quarantine(drafter).retain(lease, lease["page_slots"])
        self.assertEqual(drafter._reclaim_abandoned_leases(), 0)
        self.assertEqual(mapping[0].tolist(), [4, 9, 9, 5])
        self.assertEqual(freed, [])
        self.assertEqual(len(drafter._sr_kv_quarantine.records), 1)
        self.assertIsNone(drafter._sr_kv_quarantine.records[0].event)

    def _load_note_abandoned(self):
        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_draft_scheduler_mixin.py"
        )
        func = next(
            n
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name == "_sr_note_abandoned_expand"
        )
        env = {}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[
                            ast.parse("from __future__ import annotations").body[0],
                            func,
                        ],
                        type_ignores=[],
                    )
                ),
                str(path),
                "exec",
            ),
            env,
        )
        return env["_sr_note_abandoned_expand"]

    def test_recorded_abandon_stays_speculative_and_reclaim_restores_once(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            has_unresolved_resources,
            quarantine,
        )

        note = self._load_note_abandoned()
        mapping = torch.tensor([[4, 9, 9, 5]], dtype=torch.int64)
        pages = torch.tensor([9, 9], dtype=torch.int64)
        freed = []
        cleared = []
        drafter = self._drafter(mapping, freed.append)
        drafter._sr_last_abandon_has_event = True
        scheduler = NS(
            _sr_device_poisoned=False,
            sr_tree_drafter=drafter,
            _sr_clear_stamps_after_poison=lambda: cleared.append(1),
        )
        drafter.scheduler = scheduler
        note(scheduler)
        self.assertFalse(scheduler._sr_device_poisoned)
        self.assertEqual(cleared, [])
        lease = {
            "reqs": [NS(req_pool_idx=0)],
            "starts": [1],
            "mapping_snaps": [torch.tensor([1, 2], dtype=torch.int64)],
            "spec_snaps": [torch.tensor([9, 9], dtype=torch.int64)],
            "page_slots": [pages],
        }
        quarantine(drafter).retain(lease, lease["page_slots"], Event())
        self.assertEqual(drafter._reclaim_abandoned_leases(), 1)
        self.assertEqual(mapping[0].tolist(), [4, 1, 2, 5])
        self.assertEqual(freed, [pages])
        self.assertTrue(lease.get("mapping_restored"))
        self.assertFalse(drafter._sr_kv_quarantine.records)

        mapping2 = torch.tensor([[4, 9, 9, 5]], dtype=torch.int64)
        pages2 = torch.tensor([9, 9], dtype=torch.int64)
        freed2 = []
        drafter2 = self._drafter(mapping2, freed2.append)
        drafter2._sr_last_abandon_has_event = False
        scheduler2 = NS(
            _sr_device_poisoned=False,
            sr_tree_drafter=drafter2,
            _sr_clear_stamps_after_poison=lambda: cleared.append(2),
        )
        drafter2.scheduler = scheduler2
        note(scheduler2)
        self.assertTrue(scheduler2._sr_device_poisoned)
        self.assertEqual(cleared, [2])
        lease2 = {
            "reqs": [NS(req_pool_idx=0)],
            "starts": [1],
            "mapping_snaps": [torch.tensor([1, 2], dtype=torch.int64)],
            "page_slots": [pages2],
        }
        quarantine(drafter2).retain(lease2, lease2["page_slots"])
        self.assertEqual(drafter2._reclaim_abandoned_leases(), 0)
        self.assertEqual(mapping2[0].tolist(), [4, 9, 9, 5])
        self.assertEqual(freed2, [])
        self.assertTrue(has_unresolved_resources(scheduler2))
        self.assertEqual(mapping2[0].tolist(), [4, 9, 9, 5])
        self.assertEqual(freed2, [])

    def test_tail_written_front_keeps_new_page(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        mapping = torch.tensor([[4, 7, 9, 5]], dtype=torch.int64)
        pages = torch.tensor([8, 9], dtype=torch.int64)
        freed = []
        drafter = self._drafter(mapping, freed.append)
        lease = {
            "reqs": [NS(req_pool_idx=0, kv_committed_len=2)],
            "starts": [1],
            "mapping_snaps": [torch.tensor([1, 2], dtype=torch.int64)],
            "spec_snaps": [torch.tensor([8, 9], dtype=torch.int64)],
            "page_slots": [pages],
        }
        quarantine(drafter).retain(lease, lease["page_slots"], Event())
        self.assertEqual(drafter._reclaim_abandoned_leases(), 1)
        self.assertEqual(mapping[0].tolist(), [4, 7, 2, 5])
        self.assertEqual([unit.tolist() for unit in freed], [[8, 9]])
        self.assertFalse(drafter._sr_kv_quarantine.records)

        mapping_held = torch.tensor([[8, 7, 9, 5]], dtype=torch.int64)
        pages_held = torch.tensor([8, 9], dtype=torch.int64)
        freed_held = []
        drafter_held = self._drafter(mapping_held, freed_held.append)
        lease_held = {
            "reqs": [NS(req_pool_idx=0, kv_committed_len=2)],
            "starts": [1],
            "mapping_snaps": [torch.tensor([1, 2], dtype=torch.int64)],
            "spec_snaps": [torch.tensor([8, 9], dtype=torch.int64)],
            "page_slots": [pages_held],
        }
        quarantine(drafter_held).retain(
            lease_held, lease_held["page_slots"], Event()
        )
        self.assertEqual(drafter_held._reclaim_abandoned_leases(), 0)
        self.assertEqual(mapping_held[0].tolist(), [8, 7, 2, 5])
        self.assertEqual([unit.tolist() for unit in freed_held], [[9]])
        held = [
            unit
            for unit in drafter_held._sr_kv_quarantine.records[0].units
            if unit is not None
        ]
        self.assertEqual([unit.tolist() for unit in held], [[8]])

        mapping_undo = torch.tensor([[4, 8, 9, 5]], dtype=torch.int64)
        pages_undo = torch.tensor([8, 9], dtype=torch.int64)
        freed_undo = []
        drafter_undo = self._drafter(mapping_undo, freed_undo.append)
        lease_undo = {
            "reqs": [NS(req_pool_idx=0, kv_committed_len=1)],
            "starts": [1],
            "mapping_snaps": [torch.tensor([1, 2], dtype=torch.int64)],
            "spec_snaps": [torch.tensor([8, 9], dtype=torch.int64)],
            "page_slots": [pages_undo],
        }
        quarantine(drafter_undo).retain(
            lease_undo, lease_undo["page_slots"], Event()
        )
        self.assertEqual(drafter_undo._reclaim_abandoned_leases(), 1)
        self.assertEqual(mapping_undo[0].tolist(), [4, 1, 2, 5])
        self.assertEqual([unit.tolist() for unit in freed_undo], [[8, 9]])

    def test_committed_prefix_inside_speculative_window_restores_snap(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        mapping = torch.tensor([[4, 9, 9, 5]], dtype=torch.int64)
        pages = torch.tensor([9, 9], dtype=torch.int64)
        freed = []
        drafter = self._drafter(mapping, freed.append)
        lease = {
            "reqs": [NS(req_pool_idx=0, kv_committed_len=2)],
            "starts": [1],
            "mapping_snaps": [torch.tensor([1, 2], dtype=torch.int64)],
            "spec_snaps": [torch.tensor([9, 9], dtype=torch.int64)],
            "page_slots": [pages],
        }
        quarantine(drafter).retain(lease, lease["page_slots"], Event())
        self.assertEqual(drafter._reclaim_abandoned_leases(), 1)
        self.assertEqual(mapping[0].tolist(), [4, 1, 2, 5])
        self.assertEqual([unit.tolist() for unit in freed], [[9, 9]])
        self.assertFalse(drafter._sr_kv_quarantine.records)

    def test_second_query_does_not_free_before_restore(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        class _DoneOnSecondQuery:
            def __init__(self):
                self.calls = 0

            def query(self):
                self.calls += 1
                return self.calls >= 2

        mapping = torch.tensor([[4, 9, 9, 5]], dtype=torch.int64)
        pages = torch.tensor([9, 9], dtype=torch.int64)
        freed = []
        drafter = self._drafter(mapping, freed.append)
        lease = {
            "reqs": [NS(req_pool_idx=0)],
            "starts": [1],
            "mapping_snaps": [torch.tensor([1, 2], dtype=torch.int64)],
            "spec_snaps": [torch.tensor([9, 9], dtype=torch.int64)],
            "page_slots": [pages],
        }
        event = _DoneOnSecondQuery()
        quarantine(drafter).retain(lease, lease["page_slots"], event)
        self.assertEqual(drafter._reclaim_abandoned_leases(), 0)
        self.assertEqual(mapping[0].tolist(), [4, 9, 9, 5])
        self.assertEqual(freed, [])
        self.assertFalse(lease.get("mapping_restored"))
        self.assertEqual(drafter._reclaim_abandoned_leases(), 1)
        self.assertEqual(mapping[0].tolist(), [4, 1, 2, 5])
        self.assertEqual(freed, [pages])
        self.assertTrue(lease.get("mapping_restored"))
        self.assertFalse(drafter._sr_kv_quarantine.records)

    def test_missing_spec_image_stays_quarantined(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        mapping = torch.tensor([[4, 9, 9, 5]], dtype=torch.int64)
        pages = torch.tensor([9, 9], dtype=torch.int64)
        freed = []
        drafter = self._drafter(mapping, freed.append)
        lease = {
            "reqs": [NS(req_pool_idx=0)],
            "starts": [1],
            "mapping_snaps": [torch.tensor([1, 2], dtype=torch.int64)],
            "page_slots": [pages],
        }
        quarantine(drafter).retain(lease, lease["page_slots"], Event())
        self.assertEqual(drafter._reclaim_abandoned_leases(), 0)
        self.assertEqual(mapping[0].tolist(), [4, 9, 9, 5])
        self.assertEqual(freed, [])
        self.assertEqual(len(drafter._sr_kv_quarantine.records), 1)

    def test_paged_abandon_keeps_the_whole_physical_page(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        mapping = torch.tensor([[4, 128, 200, 201]], dtype=torch.int64)
        pages = torch.tensor([128, 129, 256], dtype=torch.int64)
        freed = []
        drafter = self._drafter(mapping, freed.append)
        drafter.token_to_kv_pool_allocator.page_size = 128
        lease = {
            "reqs": [NS(req_pool_idx=0, kv_committed_len=2)],
            "starts": [2],
            "mapping_snaps": [torch.tensor([3, 4], dtype=torch.int64)],
            "spec_snaps": [torch.tensor([200, 201], dtype=torch.int64)],
            "page_slots": [pages],
        }
        quarantine(drafter).retain(lease, lease["page_slots"], Event())
        self.assertEqual(drafter._reclaim_abandoned_leases(), 0)
        self.assertEqual(mapping[0].tolist(), [4, 128, 3, 4])
        self.assertEqual([unit.tolist() for unit in freed], [[256]])
        held = [
            unit
            for unit in drafter._sr_kv_quarantine.records[0].units
            if unit is not None
        ]
        self.assertEqual([unit.tolist() for unit in held], [[128, 129]])

        page0 = torch.tensor([1, 2, 256], dtype=torch.int64)
        mapping0 = torch.tensor([[4, 0, 200, 201]], dtype=torch.int64)
        freed0 = []
        drafter0 = self._drafter(mapping0, freed0.append)
        drafter0.token_to_kv_pool_allocator.page_size = 128
        req0 = NS(req_pool_idx=0, kv_committed_len=1)
        lease0 = {
            "reqs": [req0],
            "starts": [2],
            "mapping_snaps": [torch.tensor([3, 4], dtype=torch.int64)],
            "spec_snaps": [torch.tensor([200, 201], dtype=torch.int64)],
            "page_slots": [page0],
        }
        quarantine(drafter0).retain(lease0, lease0["page_slots"], Event())
        self.assertEqual(drafter0._reclaim_abandoned_leases(), 0)
        self.assertEqual([unit.tolist() for unit in freed0], [[256]])
        held0 = [
            unit
            for unit in drafter0._sr_kv_quarantine.records[0].units
            if unit is not None
        ]
        self.assertEqual([unit.tolist() for unit in held0], [[1, 2]])

    def test_stale_generation_or_row_does_not_free(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        mapping = torch.tensor([[4, 9, 9, 5]], dtype=torch.int64)
        pages = torch.tensor([9, 9], dtype=torch.int64)
        freed = []
        drafter = self._drafter(mapping, freed.append)
        drafter.token_to_kv_pool_allocator.generation = 1
        req = NS(req_pool_idx=0)
        lease = {
            "reqs": [req],
            "starts": [1],
            "mapping_snaps": [torch.tensor([1, 2], dtype=torch.int64)],
            "spec_snaps": [torch.tensor([9, 9], dtype=torch.int64)],
            "page_slots": [pages],
            "alloc_generation": 0,
            "row_identity": [(id(req), 0)],
        }
        quarantine(drafter).retain(lease, lease["page_slots"], Event())
        self.assertEqual(drafter._reclaim_abandoned_leases(), 0)
        self.assertEqual(freed, [])
        self.assertEqual(mapping[0].tolist(), [4, 9, 9, 5])

        mapping2 = torch.tensor([[4, 9, 9, 5]], dtype=torch.int64)
        pages2 = torch.tensor([9, 9], dtype=torch.int64)
        freed2 = []
        drafter2 = self._drafter(mapping2, freed2.append)
        req2 = NS(req_pool_idx=0)
        lease2 = {
            "reqs": [req2],
            "starts": [1],
            "mapping_snaps": [torch.tensor([1, 2], dtype=torch.int64)],
            "spec_snaps": [torch.tensor([9, 9], dtype=torch.int64)],
            "page_slots": [pages2],
            "row_identity": [(id(req2), 0)],
        }
        quarantine(drafter2).retain(lease2, lease2["page_slots"], Event())
        req2.req_pool_idx = 3
        self.assertEqual(drafter2._reclaim_abandoned_leases(), 0)
        self.assertEqual(freed2, [])
        self.assertEqual(mapping2[0].tolist(), [4, 9, 9, 5])

    def test_finish_handoff_keeps_quarantine_from_freeing_owned_pages(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_draft_scheduler_mixin.py"
        )
        names = {"_sr_finish_rid"}
        funcs = [
            n
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        env = dict(
            __name__="sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin",
            __package__="sglang.srt.speculative.standalone_remote.drafter",
            logger=logging.getLogger(__name__),
            FINISH_ABORT=lambda msg: msg,
            release_mm_resources=lambda mm: None,
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[
                            ast.parse("from __future__ import annotations").body[0],
                            *funcs,
                        ],
                        type_ignores=[],
                    )
                ),
                str(path),
                "exec",
            ),
            env,
        )
        mapping = torch.tensor([[4, 128, 0, 0]], dtype=torch.int64)
        pages = torch.tensor([128, 129, 256], dtype=torch.int64)
        req = NS(
            rid="r",
            req_pool_idx=0,
            kv_allocated_len=2,
            kv_committed_len=2,
            finished=lambda: True,
            multimodal_inputs=None,
        )
        freed = []
        released = []
        drafter = NS(
            req_to_token_pool=NS(req_to_token=mapping),
            token_to_kv_pool_allocator=NS(page_size=128, free=freed.append),
        )
        lease = {
            "reqs": [req],
            "page_slots": [pages],
        }
        quarantine(drafter).retain(lease, lease["page_slots"], Event())
        owner = NS(
            sr_tree_drafter=drafter,
            req_to_token_pool=drafter.req_to_token_pool,
            token_to_kv_pool_allocator=drafter.token_to_kv_pool_allocator,
            _sr_remove_req=lambda item: None,
            _sr_release_tree_lease=lambda rid: None,
            sr_state=NS(delete=lambda rid: NS(req_object=req)),
            sr_kv=NS(release_all_kv_for_finished_req=released.append),
        )
        env["_sr_finish_rid"](owner, "r")
        self.assertEqual(released, [req])
        self.assertEqual(freed, [])
        held = [
            unit
            for unit in drafter._sr_kv_quarantine.records[0].units
            if unit is not None
        ]
        self.assertEqual([unit.tolist() for unit in held], [[256]])

    def test_abandon_retains_before_a_failing_clone(self):
        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
        )
        func = next(
            n
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name == "_remember_abandoned_lease"
        )
        env = dict(
            __name__="sglang.srt.speculative.standalone_remote.drafter.sr_tree_drafter",
            __package__="sglang.srt.speculative.standalone_remote.drafter",
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])),
                str(path),
                "exec",
            ),
            env,
        )
        lease = {"page_slots": [torch.tensor([9], dtype=torch.int64)], "reqs": []}
        drafter = NS(scheduler=NS(_sr_device_poisoned=False), device=NS(type="cpu"))
        drafter._remember_abandoned_lease = env["_remember_abandoned_lease"].__get__(
            drafter
        )

        def explode(*args, **kwargs):
            raise RuntimeError("clone")

        drafter._capture_abandoned_spec_image = explode
        drafter._capture_abandon_event = lambda: Event()
        original = RuntimeError("expand failed")
        try:
            try:
                raise original
            finally:
                drafter._remember_abandoned_lease(lease)
        except RuntimeError as exc:
            self.assertIs(exc, original)
        self.assertEqual(len(drafter._sr_kv_quarantine.records), 1)
        self.assertEqual(drafter._sr_kv_quarantine.records[0].alloc_generation, 0)
        self.assertEqual(lease["row_identity"], [])

    def _load_finish(self):
        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_draft_scheduler_mixin.py"
        )
        names = {
            "_sr_finish_rid",
            "_sr_rollback_else_reprefill",
            "_sr_poll_deferred_finishes",
            "_sr_deferred_step_gate",
        }
        funcs = [
            n
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        env = dict(
            __name__="sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin",
            __package__="sglang.srt.speculative.standalone_remote.drafter",
            logger=logging.getLogger(__name__),
            FINISH_ABORT=lambda msg: msg,
            release_mm_resources=lambda mm: None,
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[
                            ast.parse("from __future__ import annotations").body[0],
                            *funcs,
                        ],
                        type_ignores=[],
                    )
                ),
                str(path),
                "exec",
            ),
            env,
        )
        return env

    def test_submitted_rollback_is_not_a_reprefill(self):
        from sglang.srt.speculative.standalone_remote.sr_kv_rollbacker import (
            SRKVRollbacker,
        )

        mapping = torch.arange(16, dtype=torch.int64).view(1, 16)
        freed = []

        def free(unit):
            freed.append([int(x) for x in unit.view(-1).tolist()])
            raise staging.SRTransferUnresolved("allocator submitted")

        req = NS(
            rid="r",
            req_pool_idx=0,
            kv_committed_len=8,
            kv_allocated_len=8,
            prefix_indices=[],
        )
        rollbacker = SRKVRollbacker(
            NS(free=free), NS(req_to_token=mapping), NS(), page_size=1
        )
        with self.assertRaises(staging.SRTransferUnresolved):
            rollbacker.local_rollback(req, 4, 8)
        self.assertEqual(freed, [[4, 5, 6, 7]])
        self.assertEqual(req.kv_committed_len, 8)
        self.assertEqual(req.kv_allocated_len, 8)

        plain = NS(
            rid="r",
            req_pool_idx=0,
            kv_committed_len=8,
            kv_allocated_len=8,
            prefix_indices=[],
        )

        def plain_free(unit):
            raise RuntimeError("maybe submitted")

        rollbacker.token_to_kv_pool_allocator = NS(free=plain_free)
        with self.assertRaises(staging.SRTransferUnresolved) as caught:
            rollbacker.local_rollback(plain, 4, 8)
        self.assertIs(caught.exception.req, plain)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(plain.kv_committed_len, 8)

        refused = NS(
            rid="r",
            req_pool_idx=0,
            kv_committed_len=8,
            kv_allocated_len=8,
            prefix_indices=[],
        )
        calls = []
        rollbacker.page_size = 2
        rollbacker.token_to_kv_pool_allocator = NS(free=calls.append)
        self.assertFalse(rollbacker.local_rollback(refused, 1, 8))
        self.assertEqual(calls, [])
        self.assertEqual(refused.kv_committed_len, 8)
        self.assertEqual(refused.kv_allocated_len, 8)

        class RaisingMap:
            shape = (1, 16)

            def __getitem__(self, key):
                raise staging.SRTransferUnresolved("map read")

        class IllegalMap:
            shape = (1, 16)

            def __getitem__(self, key):
                raise RuntimeError("illegal memory access")

        class PlainMap:
            shape = (1, 16)

            def __getitem__(self, key):
                raise RuntimeError("host bug")

        mapped = NS(
            rid="map",
            req_pool_idx=0,
            kv_committed_len=8,
            kv_allocated_len=8,
            prefix_indices=[],
        )
        map_frees = []
        rollbacker.page_size = 1
        rollbacker.req_to_token_pool = NS(req_to_token=RaisingMap())
        rollbacker.token_to_kv_pool_allocator = NS(free=map_frees.append)
        with self.assertRaises(staging.SRTransferUnresolved):
            rollbacker.local_rollback(mapped, 4, 8)
        self.assertEqual(map_frees, [])
        self.assertEqual(mapped.kv_committed_len, 8)

        illegal = NS(
            rid="illegal",
            req_pool_idx=0,
            kv_committed_len=8,
            kv_allocated_len=8,
            prefix_indices=[],
        )
        rollbacker.req_to_token_pool = NS(req_to_token=IllegalMap())
        with self.assertRaisesRegex(RuntimeError, "illegal memory access"):
            rollbacker.local_rollback(illegal, 4, 8)
        self.assertEqual(map_frees, [])
        self.assertEqual(illegal.kv_committed_len, 8)

        ordinary = NS(
            rid="ordinary",
            req_pool_idx=0,
            kv_committed_len=8,
            kv_allocated_len=8,
            prefix_indices=[],
        )
        rollbacker.req_to_token_pool = NS(req_to_token=PlainMap())
        self.assertFalse(rollbacker.local_rollback(ordinary, 4, 8))
        self.assertEqual(map_frees, [])

        reprefill = []
        env = self._load_finish()
        rollbacker.req_to_token_pool = NS(req_to_token=RaisingMap())
        owner = NS(sr_kv=rollbacker)
        helper = env["_sr_rollback_else_reprefill"].__get__(owner)
        with self.assertRaises(staging.SRTransferUnresolved):
            helper(mapped, 4, 8, lambda: reprefill.append("map"))
        self.assertEqual(reprefill, [])

        submitted = NS(
            rid="submitted",
            req_pool_idx=0,
            kv_committed_len=8,
            kv_allocated_len=8,
            prefix_indices=[],
        )
        submitted_frees = []

        def submitted_free(unit):
            submitted_frees.append([int(x) for x in unit.view(-1).tolist()])
            raise staging.SRTransferUnresolved("allocator submitted")

        rollbacker.req_to_token_pool = NS(req_to_token=mapping)
        rollbacker.token_to_kv_pool_allocator = NS(free=submitted_free)
        with self.assertRaises(staging.SRTransferUnresolved):
            helper(submitted, 4, 8, lambda: reprefill.append("free"))
        self.assertEqual(reprefill, [])
        self.assertEqual(submitted_frees, [[4, 5, 6, 7]])

        structural = []
        rollbacker.page_size = 1
        rollbacker.req_to_token_pool = NS(req_to_token=mapping)
        rollbacker.token_to_kv_pool_allocator = NS(free=structural.append)
        noop = NS(
            rid="noop",
            req_pool_idx=0,
            kv_committed_len=8,
            kv_allocated_len=8,
            prefix_indices=[],
        )
        self.assertFalse(rollbacker.local_rollback(noop, 8, 8))
        self.assertEqual(noop.kv_committed_len, 8)
        self.assertEqual(noop.kv_allocated_len, 8)
        missing = NS(
            rid="none",
            req_pool_idx=None,
            kv_committed_len=8,
            kv_allocated_len=8,
            prefix_indices=[],
        )
        self.assertFalse(rollbacker.local_rollback(missing, 4, 8))
        prefixed = NS(
            rid="prefix",
            req_pool_idx=0,
            kv_committed_len=8,
            kv_allocated_len=8,
            prefix_indices=[1, 2, 3, 4, 5],
        )
        self.assertFalse(rollbacker.local_rollback(prefixed, 4, 8))
        self.assertEqual(structural, [])
        rollbacker.page_size = 2
        owner = NS(sr_kv=rollbacker)
        helper = env["_sr_rollback_else_reprefill"].__get__(owner)
        misaligned = NS(
            rid="page",
            req_pool_idx=0,
            kv_committed_len=8,
            kv_allocated_len=8,
            prefix_indices=[],
        )
        self.assertFalse(
            helper(misaligned, 1, 8, lambda: structural.append("reprefill"))
        )
        self.assertEqual(structural, ["reprefill"])
        self.assertEqual(misaligned.kv_committed_len, 8)
        self.assertEqual(misaligned.kv_allocated_len, 8)

    def test_finish_waits_for_this_request_consumer_only(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        env = self._load_finish()
        req = NS(
            rid="r",
            req_pool_idx=3,
            kv_allocated_len=2,
            kv_committed_len=2,
            finished=lambda: True,
            multimodal_inputs=None,
        )
        other = NS(rid="s", req_pool_idx=4)
        pending = Event()
        pending.done = False
        other_pending = Event()
        other_pending.done = False
        mapping = torch.zeros(4, 4, dtype=torch.int64)
        mapping[3, :2] = torch.tensor([4, 5])
        pages = torch.tensor([4, 9], dtype=torch.int64)
        drafter = NS(
            req_to_token_pool=NS(req_to_token=mapping),
            token_to_kv_pool_allocator=NS(page_size=1, free=lambda unit: None),
        )
        quarantine(drafter).retain(
            {"reqs": [req], "page_slots": [pages]}, [pages], pending
        )
        released = []
        deleted = []
        state = NS(req_object=req)
        owner = NS(
            sr_tree_drafter=drafter,
            req_to_token_pool=drafter.req_to_token_pool,
            token_to_kv_pool_allocator=drafter.token_to_kv_pool_allocator,
            sr_state=NS(get=lambda rid: state, delete=lambda rid: deleted.append(rid)),
            _sr_remove_req=lambda item: None,
            _sr_release_tree_lease=lambda rid: None,
            sr_kv=NS(release_all_kv_for_finished_req=released.append),
        )
        env["_sr_finish_rid"](owner, "r")
        self.assertEqual(released, [])
        self.assertEqual(deleted, [])
        self.assertEqual(req.req_pool_idx, 3)
        self.assertEqual(len(drafter._sr_kv_quarantine.records), 1)

        quarantine(owner).retain(
            {"reqs": [other], "page_slots": [torch.tensor([7])]},
            [torch.tensor([7])],
            other_pending,
        )
        pending.done = True
        owner._sr_finish_rid = env["_sr_finish_rid"].__get__(owner)
        owner._sr_poll_deferred_finishes = env["_sr_poll_deferred_finishes"].__get__(
            owner
        )
        owner._sr_poll_deferred_finishes()
        self.assertEqual(released, [req])
        self.assertEqual(deleted, ["r"])
        self.assertNotIn("r", owner._sr_deferred_finish)
        self.assertTrue(owner._sr_kv_quarantine.records)

    def test_deferred_finish_stays_outside_sr_state_until_the_event_is_ready(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )
        from sglang.srt.speculative.standalone_remote.sr_protocol import SRAction

        env = self._load_finish()
        env["SRAction"] = SRAction
        req = NS(
            rid="r",
            req_pool_idx=0,
            kv_allocated_len=1,
            kv_committed_len=1,
            finished=lambda: True,
            multimodal_inputs=None,
        )
        pending = Event()
        pending.done = False
        mapping = torch.tensor([[4, 0]], dtype=torch.int64)
        pages = torch.tensor([4], dtype=torch.int64)
        drafter = NS(
            req_to_token_pool=NS(req_to_token=mapping),
            token_to_kv_pool_allocator=NS(page_size=1, generation=3, free=lambda unit: None),
        )
        quarantine(drafter).retain(
            {"reqs": [req], "page_slots": [pages]}, [pages], pending
        )
        released = []
        removed = []
        store = {}
        state = NS(req_object=req, req_id="r")
        owner = NS(
            sr_tree_drafter=drafter,
            req_to_token_pool=drafter.req_to_token_pool,
            token_to_kv_pool_allocator=drafter.token_to_kv_pool_allocator,
            sr_state=NS(
                get=lambda rid: store.get(rid),
                delete=lambda rid: store.pop(rid, None),
            ),
            _sr_remove_req=removed.append,
            _sr_release_tree_lease=lambda rid: None,
            sr_kv=NS(release_all_kv_for_finished_req=released.append),
        )
        owner._sr_finish_rid = env["_sr_finish_rid"].__get__(owner)
        owner._sr_poll_deferred_finishes = env["_sr_poll_deferred_finishes"].__get__(
            owner
        )
        owner._sr_deferred_step_gate = env["_sr_deferred_step_gate"].__get__(owner)
        env["_sr_finish_rid"](owner, "r", True, state)
        self.assertEqual(released, [])
        self.assertEqual(removed, [req])
        self.assertEqual(req.req_pool_idx, 0)
        self.assertEqual(set(owner._sr_deferred_finish), {"r"})
        self.assertIs(owner._sr_deferred_finish["r"]["state"], state)
        self.assertEqual(owner._sr_deferred_finish["r"]["generation"], 3)
        queued = owner._sr_deferred_finish
        owner.sr_state = NS(get=lambda rid: None, delete=lambda rid: None)
        self.assertIs(owner._sr_deferred_finish, queued)

        env["_sr_finish_rid"](owner, "r")
        self.assertEqual(
            owner._sr_deferred_step_gate("r", SRAction.STEP), "reject"
        )
        self.assertEqual(
            owner._sr_deferred_step_gate("r", SRAction.ABORT), "control"
        )
        self.assertEqual(released, [])
        self.assertEqual(set(owner._sr_deferred_finish), {"r"})

        owner._sr_poll_deferred_finishes()
        self.assertEqual(released, [])
        pending.fail_query = True
        owner._sr_poll_deferred_finishes()
        self.assertEqual(released, [])
        self.assertIn("r", owner._sr_deferred_finish)
        self.assertIsInstance(
            drafter._sr_kv_quarantine.records[0].release_error, RuntimeError
        )

    def test_deferred_finish_releases_once_after_the_consumer_completes(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        env = self._load_finish()
        req = NS(
            rid="r",
            req_pool_idx=0,
            kv_allocated_len=1,
            kv_committed_len=1,
            finished=lambda: True,
            multimodal_inputs=None,
        )
        pending = Event()
        pending.done = False
        mapping = torch.tensor([[4, 0]], dtype=torch.int64)
        pages = torch.tensor([4], dtype=torch.int64)
        drafter = NS(
            token_to_kv_pool_allocator=NS(page_size=1, generation=1, free=lambda unit: None),
        )
        quarantine(drafter).retain(
            {"reqs": [req], "page_slots": [pages]}, [pages], pending
        )
        released = []
        state = NS(req_object=req, req_id="r")
        owner = NS(
            sr_tree_drafter=drafter,
            req_to_token_pool=NS(req_to_token=mapping),
            token_to_kv_pool_allocator=drafter.token_to_kv_pool_allocator,
            sr_state=NS(get=lambda rid: None, delete=lambda rid: None),
            _sr_remove_req=lambda item: None,
            _sr_release_tree_lease=lambda rid: None,
            sr_kv=NS(release_all_kv_for_finished_req=released.append),
        )
        owner._sr_finish_rid = env["_sr_finish_rid"].__get__(owner)
        owner._sr_poll_deferred_finishes = env["_sr_poll_deferred_finishes"].__get__(
            owner
        )
        env["_sr_finish_rid"](owner, "r", True, state)
        pending.done = True
        owner._sr_poll_deferred_finishes()
        owner._sr_poll_deferred_finishes()
        self.assertEqual(released, [req])
        self.assertNotIn("r", getattr(owner, "_sr_deferred_finish", {}))

    def test_deferred_finish_does_not_release_a_reused_pool_row(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            deferred_release_allowed,
            drop_confirmed_deferred_finishes,
            freeze_deferred_finishes,
            quarantine,
        )
        from sglang.srt.speculative.standalone_remote.sr_protocol import SRAction

        env = self._load_finish()
        env["SRAction"] = SRAction
        req = NS(
            rid="r",
            req_pool_idx=0,
            kv_allocated_len=1,
            kv_committed_len=1,
            finished=lambda: True,
            multimodal_inputs=None,
        )
        pending = Event()
        pending.done = False
        mapping = torch.tensor([[4, 0]], dtype=torch.int64)
        pages = torch.tensor([4], dtype=torch.int64)

        class CountingMap:
            def __init__(self, inner):
                self.inner = inner
                self.reads = 0
                self.shape = inner.shape

            def __getitem__(self, key):
                self.reads += 1
                return self.inner[key]

        freed = []
        alloc = NS(page_size=1, generation=2, free=freed.append)
        drafter = NS(
            token_to_kv_pool_allocator=alloc,
            req_to_token_pool=NS(req_to_token=mapping),
        )
        quarantine(drafter).retain(
            {"reqs": [req], "page_slots": [pages]}, [pages], pending
        )
        released = []
        leases = []
        state = NS(req_object=req, req_id="r", session_id="s0")
        counted = CountingMap(mapping)
        owner = NS(
            sr_tree_drafter=drafter,
            req_to_token_pool=NS(req_to_token=counted),
            token_to_kv_pool_allocator=alloc,
            sr_state=NS(
                session_id="s0",
                get=lambda rid: None,
                delete=lambda rid: None,
            ),
            sr_waiting=[],
            _sr_remove_req=lambda item: None,
            _sr_release_tree_lease=leases.append,
            sr_kv=NS(release_all_kv_for_finished_req=released.append),
        )
        owner._sr_finish_rid = env["_sr_finish_rid"].__get__(owner)
        owner._sr_poll_deferred_finishes = env["_sr_poll_deferred_finishes"].__get__(
            owner
        )
        owner._sr_deferred_step_gate = env["_sr_deferred_step_gate"].__get__(owner)
        env["_sr_finish_rid"](owner, "r", True, state)
        self.assertEqual(counted.reads, 0)
        entry = owner._sr_deferred_finish["r"]
        self.assertEqual(entry["allocator_generation"], 2)
        self.assertEqual(entry["session_id"], "s0")
        self.assertEqual(entry["row_identity"][1], 0)
        self.assertEqual(entry["pages"], [4])

        alloc.generation = 3
        mapping[0, 0] = 99
        newbie = NS(rid="new", req_pool_idx=0)
        owner.sr_waiting = [newbie]
        self.assertFalse(deferred_release_allowed(owner, entry))
        owner._sr_poll_deferred_finishes()
        self.assertEqual(counted.reads, 0)
        self.assertEqual(released, [])
        self.assertEqual(leases, [])
        self.assertEqual(freed, [])
        self.assertTrue(entry["stale"])
        self.assertIn("r", owner._sr_deferred_finish)
        stale = [
            record
            for record in owner._sr_kv_quarantine.records
            if getattr(record, "stale", False)
        ]
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0].alloc_generation, 2)
        self.assertEqual(owner._sr_kv_quarantine.reclaim(alloc), 0)
        self.assertEqual(freed, [])
        self.assertEqual(
            owner._sr_deferred_step_gate("r", SRAction.FINISH, "s1"),
            "deferred_finish",
        )
        self.assertEqual(released, [])
        owner.sr_state.session_id = "s1"
        owner._sr_poll_deferred_finishes()
        self.assertEqual(counted.reads, 0)
        self.assertEqual(released, [])

        queued = owner._sr_deferred_finish
        freeze_deferred_finishes(owner)
        self.assertIs(owner._sr_deferred_finish, queued)
        self.assertTrue(queued["r"]["stale"])
        mixin = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_draft_scheduler_mixin.py"
        ).read_text(encoding="utf-8")
        self.assertLess(
            mixin.find("freeze_deferred_finishes(self)"),
            mixin.find("alloc.clear()"),
        )
        self.assertLess(
            mixin.find("alloc.clear()"),
            mixin.find("drop_confirmed_deferred_finishes(self)"),
        )
        drop_confirmed_deferred_finishes(owner)
        self.assertEqual(owner._sr_deferred_finish, {})
        self.assertTrue(any(getattr(record, "stale", False) for record in owner._sr_kv_quarantine.records))

    def test_unallocated_abort_does_not_leave_an_empty_hold(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            has_unresolved_resources,
        )

        env = self._load_finish()
        mm = []
        env["release_mm_resources"] = mm.append
        req = NS(
            rid="r",
            req_pool_idx=None,
            kv_allocated_len=0,
            kv_committed_len=0,
            finished=lambda: False,
            multimodal_inputs="mm",
        )
        store = {"r": NS(req_object=req)}
        released = []
        leases = []
        owner = NS(
            sr_state=NS(
                get=lambda rid: store.get(rid),
                delete=lambda rid: store.pop(rid, None),
            ),
            _sr_remove_req=lambda item: None,
            _sr_release_tree_lease=leases.append,
            sr_kv=NS(release_all_kv_for_finished_req=released.append),
        )
        env["_sr_finish_rid"](owner, "r")
        env["_sr_finish_rid"](owner, "r")
        self.assertEqual(released, [])
        self.assertEqual(leases, [])
        self.assertEqual(mm, ["mm"])
        self.assertIsNone(req.multimodal_inputs)
        self.assertTrue(req.to_abort)
        self.assertFalse(hasattr(owner, "_sr_kv_quarantine"))
        self.assertFalse(has_unresolved_resources(owner))
        self.assertEqual(store, {})

    def test_finish_with_only_a_lease_still_commits_the_hold(self):
        env = self._load_finish()
        req = NS(
            rid="r",
            req_pool_idx=None,
            kv_allocated_len=0,
            kv_committed_len=0,
            finished=lambda: True,
            multimodal_inputs=None,
        )
        released = []
        leases = []
        owner = NS(
            sr_tree_leases=NS(_leases={"r": NS(copy_unresolved=False)}),
            sr_state=NS(
                get=lambda rid: NS(req_object=req),
                delete=lambda rid: NS(req_object=req),
            ),
            _sr_remove_req=lambda item: None,
            _sr_release_tree_lease=leases.append,
            sr_kv=NS(release_all_kv_for_finished_req=released.append),
        )
        env["_sr_finish_rid"](owner, "r")
        self.assertEqual(released, [])
        self.assertEqual(leases, ["r"])
        self.assertFalse(
            any(
                not record.units and record.event is None
                for record in owner._sr_kv_quarantine.records
            )
        )

    def test_finish_without_pool_row_keeps_quarantined_pages(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        env = self._load_finish()
        req = NS(
            rid="r",
            req_pool_idx=None,
            kv_allocated_len=0,
            kv_committed_len=0,
            finished=lambda: True,
            multimodal_inputs=None,
        )
        pages = torch.tensor([9], dtype=torch.int64)
        drafter = NS(token_to_kv_pool_allocator=NS(page_size=1, generation=0))
        quarantine(drafter).retain(
            {"reqs": [req], "page_slots": [pages]}, [pages], Event()
        )
        released = []
        owner = NS(
            sr_tree_drafter=drafter,
            token_to_kv_pool_allocator=drafter.token_to_kv_pool_allocator,
            sr_state=NS(
                get=lambda rid: NS(req_object=req),
                delete=lambda rid: NS(req_object=req),
            ),
            _sr_remove_req=lambda item: None,
            _sr_release_tree_lease=lambda rid: None,
            sr_kv=NS(release_all_kv_for_finished_req=released.append),
        )
        env["_sr_finish_rid"](owner, "r")
        self.assertEqual(released, [])
        left = []
        for record in drafter._sr_kv_quarantine.records:
            for unit in record.units:
                left.extend(int(x) for x in unit.view(-1).tolist())
        self.assertEqual(left, [9])
        owner_bag = getattr(owner, "_sr_kv_quarantine", None)
        self.assertIsNotNone(owner_bag)
        self.assertFalse(
            any(not record.units and record.event is None for record in owner_bag.records)
        )

    def test_finish_release_failure_keeps_the_hold(self):
        env = self._load_finish()
        req = NS(
            rid="r",
            req_pool_idx=0,
            kv_allocated_len=1,
            kv_committed_len=1,
            finished=lambda: True,
            multimodal_inputs="mm",
        )
        mapping = torch.tensor([[4, 0]], dtype=torch.int64)
        freed = []
        state = {"current": NS(req_object=req)}

        def get(rid):
            return state["current"]

        def delete(rid):
            current = state["current"]
            state["current"] = None
            return current

        def release(item):
            freed.append(item)
            raise RuntimeError("release failed")

        owner = NS(
            req_to_token_pool=NS(req_to_token=mapping),
            token_to_kv_pool_allocator=NS(page_size=1, free=lambda unit: None),
            sr_state=NS(get=get, delete=delete),
            _sr_remove_req=lambda item: None,
            _sr_release_tree_lease=lambda rid: None,
            sr_kv=NS(release_all_kv_for_finished_req=release),
        )
        with self.assertRaises(staging.SRTransferUnresolved) as caught:
            env["_sr_finish_rid"](owner, "r")
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertIsNone(state["current"])
        self.assertEqual(req.multimodal_inputs, "mm")
        hold = owner._sr_kv_quarantine.records[0]
        self.assertEqual(hold.states, ["unknown"])
        self.assertIsInstance(hold.release_error, RuntimeError)
        env["_sr_finish_rid"](owner, "r")
        self.assertEqual(freed, [req])
        self.assertEqual(len(owner._sr_kv_quarantine.records), 1)

    def test_sealed_speculative_pages_survive_pool_reuse(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            commit_finish_pages,
            prepare_finish_pages,
            quarantine,
            record_may_release,
        )

        req_a = NS(req_pool_idx=0, kv_committed_len=1, kv_allocated_len=1)
        req_b = NS(req_pool_idx=1, kv_committed_len=1, kv_allocated_len=1)
        pages_a = torch.tensor([8, 9], dtype=torch.int64)
        pages_b = torch.tensor([10, 11], dtype=torch.int64)
        drafter = NS(token_to_kv_pool_allocator=NS(page_size=1, generation=0))
        lease = {
            "reqs": [req_a, req_b],
            "page_slots": [pages_a, pages_b],
            "row_identity": [(id(req_a), 0), (id(req_b), 1)],
        }
        record = quarantine(drafter).retain(lease, lease["page_slots"], Event())
        record.alloc_generation = 0
        owner = NS(
            sr_tree_drafter=drafter,
            token_to_kv_pool_allocator=drafter.token_to_kv_pool_allocator,
        )
        prepare_finish_pages(owner, req_a, {8}, 1, 0)
        commit_finish_pages(owner, req_a, {8}, 1)
        req_a.req_pool_idx = None
        self.assertTrue(drafter._sr_kv_quarantine.records)
        self.assertTrue(
            all(
                not record_may_release(item, NS(generation=1))
                for item in drafter._sr_kv_quarantine.records
            )
        )
        self.assertEqual(
            drafter._sr_kv_quarantine.reclaim(NS(free=lambda unit: None, generation=1)),
            0,
        )
        freed = []
        self.assertEqual(
            drafter._sr_kv_quarantine.reclaim(NS(free=freed.append, generation=0)),
            2,
        )
        released = []
        for unit in freed:
            released.extend(int(x) for x in unit.view(-1).tolist())
        self.assertEqual(released.count(9), 1)
        self.assertNotIn(8, released)
        self.assertIn(10, released)
        self.assertIn(11, released)


class SeqLensCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/verifier/sr_worker.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            "_req_committed_len",
            "_host_len_cache",
            "_write_host_values",
            "_copy_committed_lens",
            "_align_seq_lens_to_committed",
        }
        funcs = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        env = {"torch": torch}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[
                            ast.parse("from __future__ import annotations").body[0],
                            *funcs,
                        ],
                        type_ignores=[],
                    )
                ),
                str(path),
                "exec",
            ),
            env,
        )
        cls.env = env

    def test_shape_mismatch_does_not_alias_the_cache(self):
        copy_lens = self.env["_copy_committed_lens"]
        host = torch.tensor([3, 4, 9], dtype=torch.int64)
        dest = torch.zeros(1, dtype=torch.int64)
        out = copy_lens(dest, host, 2)
        self.assertEqual(out.tolist(), [3, 4])
        self.assertNotEqual(out.data_ptr(), host.data_ptr())
        out.add_(10)
        self.assertEqual(host.tolist(), [3, 4, 9])

        matched = torch.zeros(2, dtype=torch.int64)
        copied = copy_lens(matched, host, 2)
        self.assertEqual(copied.data_ptr(), matched.data_ptr())
        self.assertEqual(matched.tolist(), [3, 4])

    def test_align_keeps_cache_independent_of_seq_lens(self):
        batch = NS(
            reqs=[NS(kv_committed_len=5), NS(kv_committed_len=6)],
            seq_lens=torch.zeros(1, dtype=torch.int64),
            seq_lens_cpu=torch.zeros(1, dtype=torch.int64),
            orig_seq_lens=torch.zeros(1, dtype=torch.int32),
        )
        self.env["_align_seq_lens_to_committed"](batch)
        batch.seq_lens.add_(1)
        batch.seq_lens_cpu.add_(1)
        batch.orig_seq_lens.add_(1)
        self.assertEqual(batch._sr_committed_len_host[:2].tolist(), [5, 6])
        self.assertEqual(batch._sr_committed_len_cpu[:2].tolist(), [5, 6])
        self.assertEqual(batch._sr_committed_len_orig[:2].tolist(), [5, 6])


class CandidateGuardsTest(unittest.TestCase):
    def test_fatal_preflight_is_retained_and_not_cached_as_decline(self):
        from sglang.srt.speculative.standalone_remote.drafter import (
            sr_tree_candidates as c,
        )

        for error in (
            staging.SRTransferUnresolved("submitted"),
            RuntimeError("device-side assert triggered"),
        ):
            with self.subTest(error=type(error)):
                worker = NS(
                    topk=2,
                    speculative_num_steps=2,
                    speculative_num_draft_tokens=3,
                    model_config=NS(vocab_size=8),
                    scheduler=None,
                )
                seed = NS(device=NS(type="npu"), dtype=torch.float32)
                ws = c.SRTreeCandidateWorkspace("cpu", 1, 2, 2, 3, 8, torch.float32)
                held = object()
                ws.holds = held
                with patch.object(
                    torch,
                    "get_device_module",
                    return_value=NS(current_stream=lambda d: NS()),
                ), patch(
                    "sglang.srt.speculative.standalone_remote.sr_kv_copy._capturing",
                    return_value=False,
                ), patch(
                    "sglang.srt.speculative.standalone_remote.sr_kv_copy._stream_key",
                    return_value=1,
                ), patch.object(
                    c, "SRTreeCandidateWorkspace", return_value=ws
                ), patch.object(
                    ws, "warm", side_effect=error
                ) as warm:
                    with self.assertRaises(type(error)) as raised:
                        c.candidate_workspace(worker, 1, seed)
                    self.assertIs(raised.exception, error)
                    self.assertTrue(ws.unresolved)
                    self.assertIs(ws.holds, held)
                    self.assertIn(ws, worker._candidate_workspaces.values())
                    with self.assertRaises(staging.SRTransferUnresolved):
                        c.candidate_workspace(worker, 1, seed)
                    self.assertEqual(warm.call_count, 1)

    def test_cuda_candidate_preflight_decline_is_cached(self):
        from sglang.srt.speculative.standalone_remote.drafter import (
            sr_tree_candidates as c,
        )

        calls = {"init": 0, "warm": 0}
        seen = {}

        def init(self, device, batch, topk, steps, width, vocab, dtype, graph=False):
            calls["init"] += 1
            seen["device"] = device
            self.batch = batch
            self.device = torch.device("cpu")
            self.unresolved = False

        def warm(self):
            calls["warm"] += 1
            raise RuntimeError("declined")

        worker = NS(
            topk=2,
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            model_config=NS(vocab_size=8),
            scheduler=None,
        )
        seed = NS(device=NS(type="cuda"), dtype=torch.float32)
        with patch.object(
            torch,
            "get_device_module",
            return_value=NS(current_stream=lambda d: NS()),
        ), patch(
            "sglang.srt.speculative.standalone_remote.sr_kv_copy._capturing",
            return_value=False,
        ), patch(
            "sglang.srt.speculative.standalone_remote.sr_kv_copy._stream_key",
            return_value=1,
        ), patch.object(
            c.SRTreeCandidateWorkspace, "__init__", init
        ), patch.object(
            c.SRTreeCandidateWorkspace, "warm", warm
        ), patch.object(
            c.SRTreeCandidateWorkspace,
            "check",
            side_effect=AssertionError("declined workspace reused"),
        ):
            self.assertIsNone(c.candidate_workspace(worker, 2, seed, graph=False))
            self.assertIsNone(c.candidate_workspace(worker, 2, seed, graph=False))
        self.assertEqual(calls["init"], 1)
        self.assertEqual(calls["warm"], 1)
        self.assertEqual(seen["device"].type, "cuda")
        self.assertEqual(list(worker._candidate_workspaces.values()), [None])

    def test_graph_decline_never_selects_eager_workspace(self):
        path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
        )
        fn = next(
            n
            for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name == "_draft_forward"
        )
        start = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "candidates" for t in n.targets)
        )
        fragment = ast.fix_missing_locations(
            ast.Module(body=fn.body[start : start + 2], type_ignores=[])
        )
        for selected in (True, False):
            calls = []
            env = dict(
                self=NS(
                    _capture_candidate_selected=selected,
                    _capture_candidate_workspace=None,
                ),
                forward_batch=NS(batch_size=1),
                topk_p=object(),
                candidate_workspace=lambda *a: calls.append(a) or "eager",
            )
            exec(compile(fragment, str(path), "exec"), env)
            self.assertEqual(len(calls), int(not selected))
            self.assertEqual(env["candidates"], None if selected else "eager")


if __name__ == "__main__":
    unittest.main()
