"""CPU contracts for Target tree_paged_fia metadata, selection, and graph wiring."""

from __future__ import annotations

import ast
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.standalone_remote.verifier.sr_target_tree_fia import (
    IMPL_TREE_PAGED_FIA,
    SR_TARGET_TREE_FIA_ENV,
    SRTargetTreeFiaMetadata,
    TARGET_TREE_FIA_KV_ATTR,
    dense_target_tree_fia_reference,
    fill_target_tree_fia_metadata_,
    maybe_select_target_tree_fia,
    pages_for_s_cap,
    plan_target_tree_fia_lengths,
    prefix_columns_visible,
    prime_target_tree_fia_capture_,
    read_sr_target_tree_fia_env,
    target_tree_fia_blocked_extra_combos,
    validate_target_tree_fia_inputs,
    validate_target_tree_fia_records,
)
from sglang.srt.speculative.tree_attn_fallback import build_tree_verify_kv_slots_ref
from sglang.srt.speculative.tree_attn_mask import full_mask_numel
from sglang.srt.speculative.tree_shared_prefix import SHARED_PREFIX_IMPL
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

_REPO = pathlib.Path(__file__).resolve().parents[4]
_BACKEND = _REPO / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
_RUNNER = _REPO / "python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py"


def _fn_source(path: pathlib.Path, class_name: str, name: str) -> str:
    tree = ast.parse(path.read_text())
    text = path.read_text()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    return ast.get_source_segment(text, child) or ast.unparse(child)
    raise AssertionError(f"{class_name}.{name} not found in {path}")


def _req_to_token(page_ids, page_size: int, extra: int = 16) -> torch.Tensor:
    bs = len(page_ids)
    n_pages = max((len(p) for p in page_ids), default=1)
    ctx = n_pages * page_size + extra
    req = torch.zeros((bs, ctx), dtype=torch.int64)
    for b, pages in enumerate(page_ids):
        for j, pid in enumerate(pages):
            start = j * page_size
            req[b, start : start + page_size] = int(pid) * page_size + torch.arange(
                page_size, dtype=torch.int64
            )
    return req


def _full_mask(prefixes, queries: int, *, siblings=True) -> torch.Tensor:
    tree = torch.eye(queries, dtype=torch.bool)
    tree[:, 0] = True
    if siblings:
        for q in range(3, queries):
            tree[q, 1] = True
    return torch.cat(
        [
            torch.cat(
                (torch.ones(queries, int(p), dtype=torch.bool), tree), 1
            ).flatten()
            for p in prefixes
        ]
    )


def _visible_from_pages(md, prefixes, queries, page_size):
    rows = []
    for b, prefix in enumerate(prefixes):
        kv_len = int(prefix) + queries
        for q in range(queries):
            slots = []
            for s in range(kv_len):
                if bool(md.blocked_mask[b, 0, q, s]):
                    continue
                page_id = int(md.block_tables[b, s // page_size])
                slots.append(page_id * page_size + (s % page_size))
            rows.append(torch.tensor(slots, dtype=torch.int64))
    return rows


class TestTargetTreeFiaEnv(CustomTestCase):
    def test_default_on_when_unset(self):
        self.assertTrue(read_sr_target_tree_fia_env({}))

    def test_explicit_off(self):
        for raw in ("", "0", "false", "no", "off", "maybe"):
            self.assertFalse(read_sr_target_tree_fia_env({SR_TARGET_TREE_FIA_ENV: raw}))

    def test_explicit_on(self):
        for raw in ("1", "true", "YES", "On"):
            self.assertTrue(read_sr_target_tree_fia_env({SR_TARGET_TREE_FIA_ENV: raw}))


class TestTargetTreeFiaSelection(CustomTestCase):
    def test_default_upgrades_capable_target(self):
        impl, reason = maybe_select_target_tree_fia(
            SHARED_PREFIX_IMPL,
            None,
            requested=True,
            capable=True,
            extra_reason=None,
            page_size=128,
            role="target",
            verify_topk=3,
        )
        self.assertEqual(impl, IMPL_TREE_PAGED_FIA)
        self.assertIsNone(reason)

    def test_env_off_keeps_shared_prefix(self):
        impl, reason = maybe_select_target_tree_fia(
            SHARED_PREFIX_IMPL,
            None,
            requested=False,
            capable=True,
            extra_reason=None,
            page_size=128,
            role="target",
            verify_topk=3,
        )
        self.assertEqual(impl, SHARED_PREFIX_IMPL)
        self.assertIn(SR_TARGET_TREE_FIA_ENV, reason)

    def test_capability_or_combo_or_page_keeps_prior(self):
        impl, reason = maybe_select_target_tree_fia(
            "compact_fia",
            "unsupported attention layer",
            requested=True,
            capable=False,
            extra_reason=None,
            page_size=128,
            role="target",
            verify_topk=3,
        )
        self.assertEqual(impl, "compact_fia")
        self.assertEqual(reason, "unsupported attention layer")
        impl, reason = maybe_select_target_tree_fia(
            SHARED_PREFIX_IMPL,
            None,
            requested=True,
            capable=True,
            extra_reason="dp attention",
            page_size=128,
            role="target",
            verify_topk=3,
        )
        self.assertEqual(impl, SHARED_PREFIX_IMPL)
        self.assertEqual(reason, "dp attention")
        impl, reason = maybe_select_target_tree_fia(
            SHARED_PREFIX_IMPL,
            None,
            requested=True,
            capable=True,
            extra_reason=None,
            page_size=1,
            role="target",
            verify_topk=3,
        )
        self.assertEqual(impl, SHARED_PREFIX_IMPL)
        self.assertIn("page_size=1", reason)

    def test_draft_role_untouched(self):
        impl, reason = maybe_select_target_tree_fia(
            "paged_atb",
            None,
            requested=True,
            capable=True,
            extra_reason=None,
            page_size=128,
            role="draft",
            verify_topk=3,
        )
        self.assertEqual(impl, "paged_atb")
        self.assertIsNone(reason)

    def test_extra_combos(self):
        self.assertIsNone(target_tree_fia_blocked_extra_combos(SimpleNamespace()))
        self.assertEqual(
            target_tree_fia_blocked_extra_combos(
                SimpleNamespace(enable_dp_attention=True)
            ),
            "dp attention",
        )
        self.assertEqual(
            target_tree_fia_blocked_extra_combos(SimpleNamespace(attn_cp_size=2)),
            "context parallel",
        )
        self.assertEqual(
            target_tree_fia_blocked_extra_combos(SimpleNamespace(pp_size=2)),
            "pipeline parallel",
        )
        self.assertEqual(
            target_tree_fia_blocked_extra_combos(
                SimpleNamespace(enable_two_batch_overlap=True)
            ),
            "two batch overlap",
        )
        self.assertEqual(
            target_tree_fia_blocked_extra_combos(SimpleNamespace(enable_pdmux=True)),
            "pdmux",
        )


class TestTargetTreeFiaLayout(CustomTestCase):
    def _fill(self, prefixes, queries, page_ids, *, capture_bs=None, page=128):
        raw_bs = len(prefixes)
        capture_bs = raw_bs if capture_bs is None else capture_bs
        needed = max((p + queries for p in prefixes), default=queries)
        pages = pages_for_s_cap(max(needed, 256), page)
        md = SRTargetTreeFiaMetadata.allocate(capture_bs, queries, pages, page, "cpu")
        table = _req_to_token(page_ids, page)
        pool = torch.arange(raw_bs, dtype=torch.int64)
        if capture_bs > raw_bs:
            pool = torch.cat((pool, torch.zeros(capture_bs - raw_bs, dtype=torch.int64)))
        mask = _full_mask(prefixes, queries)
        fill_target_tree_fia_metadata_(
            md, table, pool, mask, prefixes, queries, raw_bs
        )
        return md, table, mask

    def test_prefix_boundaries_and_unequal_batch(self):
        for prefixes, queries in (
            ([0], 4),
            ([127], 5),
            ([128], 6),
            ([129], 4),
            ([0, 129], 7),
            ([127, 128], 8),
        ):
            page_ids = [[3, 9, 1] for _ in prefixes]
            md, _table, mask = self._fill(prefixes, queries, page_ids)
            self.assertEqual(int(mask.numel()), full_mask_numel(prefixes, queries))
            self.assertTrue(prefix_columns_visible(md.blocked_mask, prefixes))
            self.assertEqual(md.q_lens_cpu, [queries] * len(prefixes))
            self.assertEqual(
                md.kv_lens_cpu, [int(p) + queries for p in prefixes]
            )
            self.assertEqual(len(md.kv_lens_cpu), md.block_tables.shape[0])

    def test_noncontiguous_pages_and_dummy_padding(self):
        prefixes = [10, 200]
        queries = 5
        md, table, _ = self._fill(
            prefixes, queries, [[7], [2, 11, 4]], capture_bs=3, page=128
        )
        self.assertTrue(bool(md.active_rows[0]) and bool(md.active_rows[1]))
        self.assertFalse(bool(md.active_rows[2]))
        self.assertEqual(md.kv_lens_cpu[2], 1)
        self.assertFalse(bool(md.blocked_mask[2, 0, 0, 0]))
        self.assertTrue(bool(md.blocked_mask[2, 0, 0, 1:].all()))
        self.assertTrue((md.block_tables[2] == 0).all())
        self.assertEqual(int(md.block_tables[0, 0]), 7)
        self.assertEqual(int(md.block_tables[1, 0]), 2)
        self.assertEqual(int(md.block_tables[1, 1]), 11)

    def test_mask_polarity_siblings_and_self(self):
        prefixes = [4, 6]
        queries = 4
        md, _table, _ = self._fill(prefixes, queries, [[1], [3]])
        for b, prefix in enumerate(prefixes):
            for q in range(queries):
                self.assertFalse(bool(md.blocked_mask[b, 0, q, prefix + q]))
                self.assertFalse(bool(md.blocked_mask[b, 0, q, 0]))
            self.assertTrue(bool(md.blocked_mask[b, 0, 1, prefix + 2]))
            self.assertTrue(bool(md.blocked_mask[b, 0, 2, prefix + 1]))

    def test_matches_slot_gather_and_dense_reference(self):
        prefixes = [3, 130]
        queries = 5
        page = 128
        page_ids = [[4], [6, 1]]
        md, table, mask = self._fill(prefixes, queries, page_ids, page=page)
        loc = torch.stack(
            [
                table[b, int(p) : int(p) + queries]
                for b, p in enumerate(prefixes)
            ]
        ).reshape(-1)
        slots, lens = build_tree_verify_kv_slots_ref(
            mask,
            prefixes,
            table,
            torch.arange(len(prefixes)),
            loc,
            queries,
        )
        visible = _visible_from_pages(md, prefixes, queries, page)
        for i, row in enumerate(visible):
            n = int(lens[i])
            self.assertEqual(row.tolist(), slots[i, :n].tolist())
        q = torch.randn(2, queries, 4, 64)
        n_pages = 16
        k = torch.randn(n_pages, page, 2 * 64)
        v = torch.randn_like(k)
        ref = dense_target_tree_fia_reference(
            q,
            k,
            v,
            md.block_tables,
            md.kv_lens_cpu,
            md.blocked_mask,
            scale=64**-0.5,
            page_size=page,
            n_kv_heads=2,
        )
        self.assertEqual(tuple(ref.shape), (2 * queries, 4 * 64))

    def test_buffer_stable_across_shrink_grow(self):
        page = 128
        queries = 6
        pages = pages_for_s_cap(512, page)
        md = SRTargetTreeFiaMetadata.allocate(2, queries, pages, page, "cpu")
        ptrs = (md.block_tables.data_ptr(), md.blocked_mask.data_ptr())
        table = _req_to_token([[3, 8], [1, 5, 9]], page)
        pool = torch.tensor([0, 1], dtype=torch.int64)
        for prefixes, raw_bs, pool_now in (
            ([200, 129], 2, pool),
            ([10], 1, pool[:1]),
            ([250, 40], 2, pool),
        ):
            fill_target_tree_fia_metadata_(
                md,
                table,
                pool_now,
                _full_mask(prefixes, queries),
                prefixes,
                queries,
                raw_bs,
            )
            self.assertEqual(
                (md.block_tables.data_ptr(), md.blocked_mask.data_ptr()), ptrs
            )
            if raw_bs == 1:
                self.assertEqual(md.kv_lens_cpu[1], 1)
                self.assertTrue((md.block_tables[1] == 0).all())
                self.assertFalse(bool(md.blocked_mask[1, 0, 0, 0]))
                self.assertTrue(bool(md.blocked_mask[1, 0, :, 1:].all()))

    def test_rejects_before_any_write(self):
        page = 128
        md = SRTargetTreeFiaMetadata.allocate(2, 4, 1, page, "cpu")
        md.block_tables.fill_(7)
        md.blocked_mask.fill_(False)
        table = _req_to_token([[1], [2]], page)
        pool = torch.arange(2)
        with self.assertRaises(ValueError):
            fill_target_tree_fia_metadata_(
                md,
                table,
                pool,
                _full_mask([10], 4),
                [10, 10],
                4,
                2,
            )
        self.assertTrue((md.block_tables == 7).all())
        self.assertFalse(bool(md.blocked_mask.any()))
        with self.assertRaises(ValueError):
            validate_target_tree_fia_inputs(
                torch.tensor([10, 10]),
                _full_mask([10, 10], 4)[:3],
                4,
                2,
                2,
                1,
                page,
            )
        with self.assertRaises(ValueError):
            validate_target_tree_fia_inputs(
                [400, 400],
                _full_mask([400, 400], 4),
                4,
                2,
                2,
                1,
                page,
            )
        with self.assertRaises(ValueError):
            validate_target_tree_fia_inputs(
                torch.tensor([[8, 8]]),
                _full_mask([8, 8], 3),
                3,
                2,
                2,
                2,
                128,
            )


class TestTargetTreeFiaRecords(CustomTestCase):
    def test_payload_length_is_batch_not_tokens(self):
        prefixes, kv, q_lens = plan_target_tree_fia_lengths([10, 20], 15, 2, 3)
        self.assertEqual(prefixes, [10, 20])
        self.assertEqual(kv, [25, 35, 1])
        self.assertEqual(q_lens, [15, 15, 15])
        spec = _REPO / "python/sglang/srt/speculative/spec_utils.py"
        tree = ast.parse(spec.read_text())
        text = spec.read_text()
        names = {
            "expand_fia_cpu_update_inputs",
            "fill_fia_cpu_update_payload",
        }
        nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        ns = {}
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(spec), "exec"),
            ns,
        )
        payload = ns["expand_fia_cpu_update_inputs"]([kv], 4, TARGET_TREE_FIA_KV_ATTR)
        self.assertEqual(len(payload), 4)
        self.assertEqual(len(payload[0][TARGET_TREE_FIA_KV_ATTR]), 3)
        ns["fill_fia_cpu_update_payload"](
            payload, [[9, 8, 1]], [0, 0, 0, 0], TARGET_TREE_FIA_KV_ATTR
        )
        self.assertEqual(payload[3][TARGET_TREE_FIA_KV_ATTR], [9, 8, 1])

    def test_records_must_be_fia_and_match_layers(self):
        class Err(Exception):
            def __init__(self, msg, scope="graph"):
                super().__init__(msg)
                self.scope = scope

        fake = types.ModuleType("sglang.srt.speculative.spec_utils")
        fake.NpuGraphPreparationError = Err
        fake.inspect_dispatch_record = lambda rec, attr: (rec.name, rec.has)
        fake.normalize_fia_op_name = lambda name: name
        recs = [
            SimpleNamespace(name="npu_fused_infer_attention_score", has=True)
            for _ in range(3)
        ]
        with patch.dict(sys.modules, {"sglang.srt.speculative.spec_utils": fake}):
            n, step_ids = validate_target_tree_fia_records(recs, 3)
            self.assertEqual(n, 3)
            self.assertEqual(step_ids, [0, 0, 0])
            with self.assertRaises(Err):
                validate_target_tree_fia_records(recs[:2], 3)
            bad = list(recs)
            bad[1] = SimpleNamespace(name="npu_paged_attention", has=True)
            with self.assertRaises(Err):
                validate_target_tree_fia_records(bad, 3)


class TestTargetTreeFiaWiring(CustomTestCase):
    def test_independent_of_ascend_use_fia(self):
        module = (
            _REPO
            / "python/sglang/srt/speculative/standalone_remote/verifier/sr_target_tree_fia.py"
        ).read_text()
        self.assertNotIn("ASCEND_USE_FIA", module)
        select = _fn_source(_BACKEND, "AscendAttnBackend", "_init_tree_shared_prefix")
        after = select.split("maybe_select_target_tree_fia", 1)[1]
        self.assertNotIn("use_fia", after)
        self.assertNotIn("ASCEND_USE_FIA", after)

    def test_backend_source_guards(self):
        init_src = _fn_source(_BACKEND, "AscendAttnBackend", "_init_tree_shared_prefix")
        self.assertIn("maybe_select_target_tree_fia", init_src)
        self.assertIn("read_sr_target_tree_fia_env", init_src)
        self.assertNotIn("_paged_impl_selected()", init_src.split("maybe_select_target_tree_fia")[1][:80])
        use_src = _fn_source(_BACKEND, "AscendAttnBackend", "_use_target_tree_paged_fia")
        self.assertIn("IMPL_TREE_PAGED_FIA", use_src)
        paged_src = _fn_source(_BACKEND, "AscendAttnBackend", "_paged_impl_selected")
        self.assertNotIn("tree_paged_fia", paged_src)
        width = _fn_source(_BACKEND, "AscendAttnBackend", "tree_slot_graph_width")
        self.assertIn("_use_target_tree_paged_fia", width)
        eager = _fn_source(_BACKEND, "AscendAttnBackend", "init_forward_metadata")
        self.assertLess(
            eager.find("_prepare_target_tree_fia_eager"),
            eager.find("_use_tree_shared_prefix"),
        )
        self.assertLess(
            eager.find("return"),
            eager.find("_fill_tree_verify_kv_slots"),
        )
        mtp = _fn_source(_BACKEND, "AscendAttnBackend", "forward_mtp")
        self.assertLess(
            mtp.find("_run_sr_target_tree_fia"),
            mtp.find("_run_tree_shared_prefix_attention"),
        )
        self.assertLess(
            mtp.find("_run_sr_target_tree_fia"),
            mtp.find("_run_tree_verify_slot_gather"),
        )
        self.assertIn("input_layout=\"BSND\"", _fn_source(_BACKEND, "AscendAttnBackend", "_run_sr_target_tree_fia"))
        self.assertIn("sparse_mode=0", _fn_source(_BACKEND, "AscendAttnBackend", "_run_sr_target_tree_fia"))
        cap = _fn_source(
            _BACKEND, "AscendAttnBackend", "init_forward_metadata_capture_cuda_graph"
        )
        self.assertIn("_shared_capture_width", cap)
        self.assertNotIn("graph_runner", cap.split("_use_target_tree_paged_fia")[1][:400])
        replay = _fn_source(
            _BACKEND, "AscendAttnBackend", "init_forward_metadata_replay_cuda_graph"
        )
        self.assertIn("fill_target_tree_fia_metadata_", replay)
        state = _fn_source(_BACKEND, "AscendAttnBackend", "init_cuda_graph_state")
        self.assertIn("skip_slots", state)
        self.assertIn("_use_target_tree_paged_fia", state)

    def test_graph_runner_source_guards(self):
        one = _fn_source(_RUNNER, "NPUGraphRunner", "capture_one_batch_size")
        self.assertIn("_bind_target_tree_fia_payload", one)
        self.assertIn("expand_fia_cpu_update_inputs", _fn_source(_RUNNER, "NPUGraphRunner", "_bind_target_tree_fia_payload"))
        replay = _fn_source(_RUNNER, "NPUGraphRunner", "replay")
        self.assertIn("not compact_fia and not target_fia", replay)
        self.assertIn("overlap=False", replay)
        target_call = replay.split("if is_tree_verify and target_fia:", 1)[1]
        target_call = target_call.split("elif is_tree_verify and compact_fia:", 1)[0]
        self.assertIn("overlap=False", target_call)
        self.assertNotIn("overlap=True", target_call)
        self.assertNotIn("overlap=True", replay)
        self.assertIn("raw_bs", replay)
        self.assertIn("capture_bs", replay)
        self.assertIn("_update_target_tree_fia_inputs", replay)
        can = _fn_source(_RUNNER, "NPUGraphRunner", "can_run")
        self.assertIn("_target_fia_maps", can)

    def test_replay_error_does_not_rerun_forward(self):
        spec = _REPO / "python/sglang/srt/speculative/spec_utils.py"
        tree = ast.parse(spec.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "run_npu_graph_update_and_replay"
        )
        err = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef)
            and n.name == "NpuGraphReplaySubmittedError"
        )
        ns = {"threading": __import__("threading")}
        exec(compile(ast.Module(body=[err, node], type_ignores=[]), str(spec), "exec"), ns)
        calls = []

        def update():
            calls.append("update")
            raise RuntimeError("update boom")

        def replay():
            calls.append("replay")

        with self.assertRaises(ns["NpuGraphReplaySubmittedError"]):
            ns["run_npu_graph_update_and_replay"](update, replay, overlap=False)
        self.assertEqual(calls, ["update"])

    def test_target_fia_replay_keeps_serial_overlap(self):
        replay = _fn_source(_RUNNER, "NPUGraphRunner", "replay")
        calls = []

        def helper(update_fn, replay_fn, overlap=False):
            calls.append({"overlap": overlap, "replay_fn": replay_fn})
            update_fn()
            replay_fn()

        class _Graph:
            def replay(self):
                return None

        graph = _Graph()
        loc = {
            "self": SimpleNamespace(
                _target_fia_maps={1: {"payload": []}},
                bs=1,
                _update_target_tree_fia_inputs=lambda info, kv_lens: None,
            ),
            "run_npu_graph_update_and_replay": helper,
            "is_tree_verify": True,
            "target_fia": True,
            "graph_key": 1,
            "graph": graph,
            "backend": SimpleNamespace(
                forward_metadata=SimpleNamespace(
                    sr_target_tree_fia=SimpleNamespace(kv_lens_cpu=[4])
                )
            ),
            "NpuGraphPreparationError": RuntimeError,
        }
        tree = ast.parse(replay)
        fn = tree.body[0]
        target_if = None
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.If)
                and ast.unparse(node.test) == "is_tree_verify and target_fia"
            ):
                target_if = node
                break
        self.assertIsNotNone(target_if)
        exec(
            compile(
                ast.Module(body=list(target_if.body), type_ignores=[]),
                str(_RUNNER),
                "exec",
            ),
            loc,
            loc,
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["overlap"], False)
        self.assertIs(calls[0]["replay_fn"].__self__, graph)


class TestPrimeCapture(CustomTestCase):
    def test_capture_prime_is_legal_dummy(self):
        md = SRTargetTreeFiaMetadata.allocate(2, 4, 2, 128, "cpu")
        prime_target_tree_fia_capture_(md, dummy_page=0)
        self.assertTrue((md.block_tables == 0).all())
        self.assertFalse(bool(md.blocked_mask[:, :, :, 0].any()))
        self.assertTrue(bool(md.blocked_mask[:, :, :, 1:].all()))
        self.assertEqual(md.kv_lens_cpu, [1, 1])


if __name__ == "__main__":
    unittest.main()
