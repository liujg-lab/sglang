"""SR RPD host topology, transfers, selection and commit contracts."""

import copy
import re
import unittest
from collections import Counter
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch

import sglang.srt.speculative.rpd_verify as rpd
from sglang.srt.speculative.standalone_remote.sr_rpd import (
    SRRPDHostPlan,
    SRRPDWorkspace,
    build_sr_rpd_input,
    rpd_batch_key,
    verify_sr_rpd_host,
)
from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
    SRTransferUnresolved,
)
from sglang.srt.speculative.standalone_remote.sr_verify_layout import VerifyInputPacket
from sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept import (
    SRFixedAcceptState,
)
from sglang.srt.speculative.tree_verify import build_tree_kernel_efficient_ref
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-test-cpu")


def reference_tree(parents, selected, seq_lens, topk, steps, width):
    """Pad a short normalized parent row so the CPU reference accepts it.

    Missing columns are the same ``-1`` fill the packet uses. In-range parent
    lookups do not read the padding, so the tree matches the unpadded packet.
    """
    expected = topk * (steps - 1) + 1
    if int(parents.shape[1]) < expected:
        pad = torch.full(
            (int(parents.shape[0]), expected - int(parents.shape[1])),
            -1,
            dtype=parents.dtype,
        )
        parents = torch.cat((parents, pad), dim=1)
    return build_tree_kernel_efficient_ref(
        parents, selected, seq_lens, topk, steps, width
    )


def packet_input(bs=2, packet=None, reqs=None, vocab=32):
    packet = packet or VerifyInputPacket()
    pp = [[-1, 0, 1, 2, 3] for _ in range(bs)]
    ss = [list(range(5)) for _ in range(bs)]
    result = packet.load(
        [2] * bs,
        [[1, 2, 3, 4, 5] for _ in range(bs)],
        pp,
        ss,
        2,
        3,
        6,
        "cpu",
        rpd_vocab=vocab,
        rpd_batch_key=rpd_batch_key(reqs) if reqs is not None else (),
    )
    return packet, result


class SRRPDTest(unittest.TestCase):
    def test_topology_matches_reference_and_survives_packet_reuse(self):
        packet, (_, parents, indices, _) = packet_input()
        context = packet.rpd_input
        reference = build_tree_kernel_efficient_ref(
            parents, indices, torch.tensor([0, 5]), 2, 3, 6
        )
        for actual, expected in zip(context.tree[1:], reference[2:]):
            torch.testing.assert_close(actual, expected)
        self.assertTrue(context.current())
        saved = context.tree.clone()
        packet_input(1, packet)
        self.assertFalse(context.current())
        torch.testing.assert_close(saved, context.tree)
        self.assertEqual(packet.rpd_input.edge_index.shape[0], 2)

    def test_chain_missing_topology_padding_and_no_full_mask(self):
        trees = {}
        cases = (
            ("missing", None, None),
            ("empty", [], []),
            ("short", [-1, 0, 1], [0, 1, 2, 3, 4]),
            ("padded", [-1, 0, 1, 2, 3], [0, 1, 2]),
        )
        for name, parents, selected in cases:
            packet = VerifyInputPacket()
            _, pp, ss, _tokens = packet.load(
                [2],
                [[3]],
                [parents],
                [selected],
                2,
                3,
                6,
                "cpu",
                rpd_vocab=32,
            )
            context = packet.rpd_input
            self.assertIsNotNone(context)
            ref = reference_tree(pp, ss, torch.tensor([0]), 2, 3, 6)
            for actual, expected in zip(context.tree[1:], ref[2:]):
                torch.testing.assert_close(actual, expected)
            self.assertEqual(context.tree[0, 0].tolist(), [2, 3, 0, 0, 0, 0])
            trees[name] = context.tree[2, 0].tolist()
        self.assertNotEqual(trees["missing"], trees["empty"])

    def test_cycle_declines_and_missing_parent_is_disconnected(self):
        owner = SimpleNamespace(generation=1, unresolved=False)
        args = dict(topk=2, steps=3, width=6, vocab=32, owner=owner, generation=1)
        verified, tokens = torch.tensor([0]), torch.tensor([[1, 2, 3, 4, 5]])
        selected = torch.tensor([[0, 1, 2, 3, 4]])
        bad = build_sr_rpd_input(
            verified, tokens, torch.tensor([[-1, 2, 1, 2, 3]]), selected, **args
        )
        self.assertIsNone(bad)
        missing = build_sr_rpd_input(
            verified, tokens, torch.tensor([[-1, 99, 1, 2, 3]]), selected, **args
        )
        self.assertIsNotNone(missing)
        self.assertEqual(missing.tree[2, 0, 1].item(), -1)

    def test_host_plan_matches_existing_compact_all_dtypes_and_thresholds(self):
        torch.manual_seed(11)
        workspace = SRRPDWorkspace()
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for tau in (0.0, 0.2, 0.5):
                for special in (False, True):
                    packet, _ = packet_input()
                    context = packet.rpd_input
                    logits = torch.randn(12, 32, dtype=dtype)
                    if special:
                        logits[0, 2] = float("nan")
                        logits[6, :] = float("-inf")
                        logits[2, 1] = float("inf")
                    ref_pred = torch.full((13,), -1, dtype=torch.int32)
                    ref_idx = torch.full((2, 4), -1, dtype=torch.int32)
                    ref_len = torch.zeros(2, dtype=torch.int32)
                    rpd._verify_tree_rpd_compact(
                        ref_pred,
                        ref_idx,
                        ref_len,
                        *context.tree,
                        logits,
                        rpd.rpd_gap_max(tau),
                        float(tau) == 0.0,
                    )
                    self.assertIsNone(workspace.prepare(logits, context))
                    with patch.object(
                        rpd, "_rpd_compact_tree", side_effect=AssertionError("tree D2H")
                    ), patch.object(
                        rpd,
                        "_rpd_compact_apply",
                        side_effect=AssertionError("result H2D"),
                    ):
                        plan = verify_sr_rpd_host(logits, context, workspace, tau, 4)
                    self.assertEqual(plan.rows, ref_idx.tolist())
                    self.assertEqual(plan.pre_lengths, ref_len.tolist())
                    for row, tokens in zip(plan.rows, plan.tokens):
                        self.assertEqual(
                            tokens, [int(ref_pred[i]) if i >= 0 else 0 for i in row]
                        )
                    with self.assertRaisesRegex(RuntimeError, "consumed"):
                        verify_sr_rpd_host(logits, context, workspace, tau, 4)

    def test_one_input_upload_includes_edges_and_generation_rejects_old_input(self):
        packet = VerifyInputPacket()
        with patch.object(packet, "on_accelerator", return_value=True), patch(
            "sglang.srt.speculative.standalone_remote.sr_verify_layout.submit_copy"
        ) as submit:

            def transfer(dst, src):
                dst.copy_(src)
                return None

            submit.side_effect = transfer
            packet_input(packet=packet)
            self.assertEqual(submit.call_count, 1)
            context = packet.rpd_input
            torch.testing.assert_close(context.edge_index, context.edge_index_cpu)
            self.assertGreater(submit.call_args.args[0].numel(), 2 * (1 + 5 + 5 + 5))
        packet_input(packet=packet)
        workspace = SRRPDWorkspace()
        self.assertEqual(workspace.prepare(torch.zeros(12, 32), context), "context")

    def test_workspace_reuse_failure_and_empty_edges(self):
        workspace = SRRPDWorkspace()
        packet, _ = packet_input()
        logits = torch.zeros(12, 32)
        self.assertIsNone(workspace.prepare(logits, packet.rpd_input))
        ptr = workspace.star_host.data_ptr()
        verify_sr_rpd_host(logits, packet.rpd_input, workspace, 0, 4)
        packet_input(1, packet)
        self.assertIsNone(workspace.prepare(logits[:6], packet.rpd_input))
        self.assertEqual(ptr, workspace.star_host.data_ptr())
        with patch("torch.max", side_effect=RuntimeError("launch failed")):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                verify_sr_rpd_host(logits[:6], packet.rpd_input, workspace, 0, 4)
        self.assertFalse(workspace.unresolved)
        self.assertIsNone(workspace.holds)
        self.assertFalse(packet.unresolved)
        packet_input(packet=packet)
        self.assertIsNotNone(packet.rpd_input)
        self.assertIsNone(workspace.prepare(logits, packet.rpd_input))

    def test_statistics_have_one_wait_and_failure_poison_both_owners(self):
        # Exercise async scheduling on CPU storage; real NPU copies are covered
        # separately by the manual hardware test.
        for fail in (None, "record", "wait"):
            packet, _ = packet_input()
            context = packet.rpd_input
            metrics = SimpleNamespace(counts=Counter())
            workspace = SRRPDWorkspace()
            logits = torch.zeros(12, 32, dtype=torch.bfloat16)
            self.assertIsNone(workspace.prepare(logits, context, metrics))
            workspace.event = Mock()
            if fail == "record":
                workspace.event.record.side_effect = RuntimeError("record")
            if fail == "wait":
                workspace.event.synchronize.side_effect = RuntimeError("wait")
            with patch.object(
                torch.Tensor,
                "device",
                new_callable=PropertyMock,
                return_value=SimpleNamespace(type="npu"),
            ):
                if fail:
                    with self.assertRaises(RuntimeError):
                        verify_sr_rpd_host(logits, context, workspace, 0.2, 4)
                else:
                    verify_sr_rpd_host(logits, context, workspace, 0.2, 4)
            workspace.event.record.assert_called_once()
            if fail:
                self.assertTrue(workspace.unresolved)
                self.assertTrue(packet.unresolved)
                self.assertIsNotNone(workspace.holds)
            else:
                workspace.event.synchronize.assert_called_once()
                self.assertEqual(metrics.counts["rpd_host_stats_waits"], 1)
                self.assertEqual(metrics.counts["rpd_host_stats_d2h_count"], 2)
                self.assertEqual(
                    metrics.counts["rpd_host_stats_d2h_bytes"],
                    12 * 8 + 4 * len(context.edges),
                )

    def test_empty_batch_and_root_only(self):
        for bs in (0, 2):
            owner = SimpleNamespace(generation=1, unresolved=False)
            context = build_sr_rpd_input(
                torch.zeros(bs, dtype=torch.int64),
                torch.empty((bs, 0), dtype=torch.int64),
                torch.full((bs, 1), -1, dtype=torch.int64),
                torch.empty((bs, 0), dtype=torch.int64),
                topk=2,
                steps=1,
                width=1,
                vocab=3,
                owner=owner,
                generation=1,
            )
            owner.rpd_input = context
            context.edge_index = context.edge_index_cpu
            workspace = SRRPDWorkspace()
            logits = torch.zeros(bs, 3)
            self.assertIsNone(workspace.prepare(logits, context))
            with patch.object(
                workspace, "statistics", wraps=workspace.statistics
            ) as stats:
                plan = verify_sr_rpd_host(logits, context, workspace, 0, 2)
            self.assertEqual(stats.call_count, int(bs > 0))
            self.assertEqual(plan.pre_lengths, [0] * bs)
            self.assertEqual(len(context.edges), 0)

    def test_threshold_ties_and_duplicate_tokens(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for tau in (0.0, 0.2, 0.5):
                for delta in (-1e-3, 0.0, 1e-3):
                    packet = VerifyInputPacket()
                    packet.load(
                        [0], [[1, 1]], [[-1]], [[0, 1]], 2, 1, 3, "cpu", rpd_vocab=4
                    )
                    context = packet.rpd_input
                    logits = torch.zeros(3, 4, dtype=dtype)
                    logits[0, 1] = -rpd.rpd_gap_max(tau) + delta
                    pred, idx, length = (
                        torch.zeros(4, dtype=torch.int32),
                        torch.full((1, 2), -1, dtype=torch.int32),
                        torch.zeros(1, dtype=torch.int32),
                    )
                    rpd._verify_tree_rpd_compact(
                        pred,
                        idx,
                        length,
                        *context.tree,
                        logits,
                        rpd.rpd_gap_max(tau),
                        tau == 0,
                    )
                    workspace = SRRPDWorkspace()
                    workspace.prepare(logits, context)
                    plan = verify_sr_rpd_host(logits, context, workspace, tau, 2)
                    self.assertEqual(plan.rows, idx.tolist())
                    self.assertEqual(plan.pre_lengths, length.tolist())
                    if plan.pre_lengths[0]:
                        self.assertEqual(plan.rows[0][1], 1)  # first sibling wins

    def test_rpd_admission_keeps_shared_gates(self):
        import test_sr_fixed_accept as fixed

        state = SRFixedAcceptState(2, 4, 6, 4, "cpu")
        packet, _ = packet_input()
        base = fixed._admit_kwargs(
            state,
            verify_mode="rpd",
            logits=torch.zeros(12, 32),
            rpd_host_input=packet.rpd_input,
        )
        self.assertIsNone(state.reject_before_alloc(**base))
        for field, value, reason in (
            ("rpd_host_input", None, "rpd_context"),
            ("has_grammar", True, "grammar"),
            ("return_logprob", True, "logprob"),
            ("prepare_hidden", True, "hidden"),
            ("has_custom_logit_processor", True, "logit_processor"),
            ("simulate_acc_len", 1, "simulate"),
            ("sampling_rows", 1, "sampling_rows"),
        ):
            args = dict(base, **{field: value})
            self.assertEqual(state.reject_before_alloc(**args), reason)
        self.assertEqual(
            state.reject_before_alloc(
                **dict(base, verify_mode="target_only", is_all_greedy=False)
            ),
            "mode",
        )
        reqs = [fixed._Req(), fixed._Req()]
        keyed, _ = packet_input(reqs=reqs)
        live = rpd_batch_key(reqs)
        matched = dict(base, rpd_host_input=keyed.rpd_input, rpd_live_batch_key=live)
        self.assertIsNone(state.reject_before_alloc(**matched))
        swapped = rpd_batch_key(list(reversed(reqs)))
        self.assertEqual(
            state.reject_before_alloc(**dict(matched, rpd_live_batch_key=swapped)),
            "rpd_batch_key",
        )

    def test_eagle_dispatch_runs_penalty_once_and_skips_greedy_buffers(self):
        import test_sr_fixed_accept as fixed

        reqs = [fixed._Req(), fixed._Req()]
        packet, _ = packet_input(reqs=reqs)
        context = packet.rpd_input
        state = SRFixedAcceptState(2, 4, 6, 4, "cpu")
        penalizer = Mock(is_required=True)

        class Sampling:
            is_all_greedy = False
            has_custom_logit_processor = False
            logit_bias = torch.zeros(2, 32)
            penalizer_orchestrator = penalizer

            def __len__(self):
                return 2

        batch = SimpleNamespace(
            reqs=reqs,
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            sampling_info=Sampling(),
            seq_lens_cpu=torch.tensor([1, 1]),
            out_cache_loc=torch.arange(12),
            device=torch.device("cpu"),
        )
        spec = SimpleNamespace(
            retrive_index=context.tree[1],
            draft_token=context.tree[0].flatten(),
            draft_token_num=6,
            spec_steps=3,
            topk=2,
            sr_rpd_input=context,
        )
        namespace = dict(
            torch=torch,
            SIMULATE_ACC_LEN=0,
            get_global_server_args=lambda: SimpleNamespace(
                speculative_verify_mode="rpd", speculative_rpd_tau=0.2
            ),
            tree_verify_backend=lambda: "cpu",
            is_remote_spec_algorithm=lambda _: True,
            _log_verify_method_once=lambda *a, **kw: None,
            _eagle_output_from_fixed=lambda raw, *args: raw,
        )
        verify = fixed._load_functions(
            fixed._REPO / "python/sglang/srt/speculative/eagle_info.py",
            ["verify"],
            namespace,
        )["verify"]
        with patch.object(
            state, "bind_verify_buffers", side_effect=AssertionError("greedy buffers")
        ), patch.object(
            state, "finalize_from_host", return_value="committed"
        ) as finalize:
            result = verify(
                spec,
                batch,
                SimpleNamespace(next_token_logits=torch.zeros(12, 32)),
                SimpleNamespace(is_not_in_free_group=True),
                4,
                prepare_local_draft_hidden=False,
                sr_accept_state=state,
            )
        self.assertEqual(result, "committed")
        penalizer.apply.assert_called_once()
        finalize.assert_called_once()

        class LegacyExit(RuntimeError):
            pass

        legacy = Mock(side_effect=LegacyExit)
        namespace["verify_tree_rpd"] = legacy
        namespace["apply_custom_logit_processor"] = Mock()
        for case in (
            "off",
            "missing",
            "stale",
            "logprob",
            "hidden",
            "grammar",
            "processor",
            "simulate",
            "noncontiguous",
        ):
            packet, _ = packet_input(reqs=reqs)
            ctx = packet.rpd_input
            spec.sr_rpd_input = ctx
            spec.retrive_next_token = ctx.tree[2]
            spec.retrive_next_sibling = ctx.tree[3]
            if case == "missing":
                spec.sr_rpd_input = None
            if case == "stale":
                packet_input(packet=packet, reqs=reqs)
            batch.return_logprob = case == "logprob"
            batch.has_grammar = case == "grammar"
            batch.sampling_info.has_custom_logit_processor = case == "processor"
            namespace["SIMULATE_ACC_LEN"] = 1 if case == "simulate" else 0
            logits = (
                torch.zeros(32, 12).t()
                if case == "noncontiguous"
                else torch.zeros(12, 32)
            )
            legacy.reset_mock()
            penalizer.apply.reset_mock()
            with patch.object(
                state,
                "finalize_from_host",
                side_effect=AssertionError("host fallback replay"),
            ), patch.object(
                state,
                "bind_verify_buffers",
                side_effect=AssertionError("greedy buffers"),
            ):
                with self.assertRaises(LegacyExit):
                    verify(
                        spec,
                        batch,
                        SimpleNamespace(next_token_logits=logits),
                        SimpleNamespace(is_not_in_free_group=True),
                        4,
                        prepare_local_draft_hidden=case == "hidden",
                        sr_accept_state=None if case == "off" else state,
                    )
            legacy.assert_called_once()
            penalizer.apply.assert_called_once()

    def test_host_finalizer_matches_device_finalizer_and_never_reads_back(self):
        import test_sr_fixed_accept as fixed
        from sglang.srt.speculative.standalone_remote import sr_kv_copy

        rows = [[0, 2, -1, -1], [4, 5, 6, -1]]
        tokens = [[10, 12, 0, 0], [20, 21, 22, 0]]
        for finish in ([None, None], [1, None], [1, 2]):
            expected = fixed.FixedAcceptFinalizeTest()._run(
                rows, tokens, [1, 2], [1, 1], finish
            )
            reqs = [fixed._Req(finish_at=x) for x in finish]
            alloc = fixed.NPUPagedTokenToKVPoolAllocator(4)
            alloc.kv_buffer = torch.arange(24, dtype=torch.float32).reshape(
                2, 1, 3, 4, 1, 1
            )
            batch, _ = fixed.FixedAcceptFinalizeTest()._batch(
                reqs, [1, 1], torch.arange(8), alloc.kv_buffer
            )
            state = SRFixedAcceptState(2, 4, 4, 4, "cpu")
            plan = SRRPDHostPlan(
                copy.deepcopy(rows),
                copy.deepcopy(tokens),
                [1, 2],
                32,
                rpd_batch_key(reqs),
            )
            with patch.object(
                state, "_pack", side_effect=AssertionError("pack")
            ), patch.object(
                state, "_wait_accept_readback", side_effect=AssertionError("readback")
            ), patch.object(
                state, "_submit_commit_packet", wraps=state._submit_commit_packet
            ) as submit, patch.object(
                sr_kv_copy, "prepare_kv_move", wraps=sr_kv_copy.prepare_kv_move
            ) as prepare:
                result = state.finalize_from_host(
                    batch, SimpleNamespace(), 4, 2, alloc, plan
                )
                self.assertEqual(submit.call_count, 1)
                self.assertEqual(prepare.call_count, 1)
            self.assertEqual(
                [q.output_ids for q in reqs], [q.output_ids for q in expected[3]]
            )
            self.assertEqual(
                result.accept_length_per_req_cpu, expected[4].accept_length_per_req_cpu
            )
            self.assertEqual(result.tree_paths, expected[4].tree_paths)
            torch.testing.assert_close(result.verified_id, expected[4].verified_id)
            torch.testing.assert_close(alloc.kv_buffer, expected[5])
            self.assertEqual(
                [q.histogram for q in reqs], [q.histogram for q in expected[3]]
            )
            with self.assertRaisesRegex(RuntimeError, "consumed"):
                state.finalize_from_host(batch, SimpleNamespace(), 4, 2, alloc, plan)

    def test_host_validation_rejects_whole_batch_before_append(self):
        import test_sr_fixed_accept as fixed

        for kind in ("tokens", "indices", "lengths", "order"):
            reqs = [fixed._Req(), fixed._Req()]
            state = SRFixedAcceptState(2, 4, 4, 4, "cpu")
            alloc = fixed.NPUPagedTokenToKVPoolAllocator(4)
            batch, _ = fixed.FixedAcceptFinalizeTest()._batch(
                reqs, [1, 1], torch.arange(8), alloc.kv_buffer
            )
            plan = SRRPDHostPlan(
                [[0, -1, -1, -1], [4, -1, -1, -1]],
                [[1, 0, 0, 0], [2, 0, 0, 0]],
                [0, 0],
                32,
                rpd_batch_key(reqs),
            )
            if kind == "tokens":
                plan.tokens[1] = []
            if kind == "indices":
                plan.rows[1][0] = 8
            if kind == "lengths":
                plan.pre_lengths[1] = 2
            if kind == "order":
                batch.reqs.reverse()
            with self.assertRaises(RuntimeError):
                state.finalize_from_host(batch, SimpleNamespace(), 4, 2, alloc, plan)
            self.assertEqual([r.output_ids for r in reqs], [[], []])
            self.assertEqual(alloc.freed, [])

    def test_kv_scratch_preflight_and_submitted_failure(self):
        import test_sr_fixed_accept as fixed

        from sglang.srt.speculative.standalone_remote.sr_kv_copy import (
            KVMoveSubmittedError,
            UnsupportedKVMoveLayout,
        )

        for preflight in (True, False):
            reqs = [fixed._Req(), fixed._Req()]
            state = SRFixedAcceptState(2, 4, 4, 4, "cpu")
            alloc = fixed.NPUPagedTokenToKVPoolAllocator(4)
            batch, _ = fixed.FixedAcceptFinalizeTest()._batch(
                reqs, [1, 1], torch.arange(8), alloc.kv_buffer
            )

            def plan():
                return SRRPDHostPlan(
                    [[0, -1, -1, -1], [4, -1, -1, -1]],
                    [[1, 0, 0, 0], [2, 0, 0, 0]],
                    [0, 0],
                    32,
                    rpd_batch_key(reqs),
                )

            if preflight:
                alloc.kv_buffer = torch.zeros(8)
                host_plan = plan()
                with self.assertRaises(UnsupportedKVMoveLayout):
                    state.finalize_from_host(
                        batch, SimpleNamespace(), 4, 2, alloc, host_plan
                    )
                self.assertFalse(host_plan.consumed)
                self.assertEqual([r.output_ids for r in reqs], [[], []])
            else:
                with patch.object(
                    torch,
                    "index_select",
                    side_effect=RuntimeError("gather launch failed"),
                ):
                    with self.assertRaises(KVMoveSubmittedError):
                        state.finalize_from_host(
                            batch, SimpleNamespace(), 4, 2, alloc, plan()
                        )
                once = [list(r.output_ids) for r in reqs]
                # A later forward would provide a fresh full verification
                # window. Do not let the prior publish's compact view hide
                # the pool poison behind an unrelated capacity rejection.
                batch.out_cache_loc = torch.arange(8)
                with self.assertRaises(KVMoveSubmittedError):
                    state.finalize_from_host(
                        batch, SimpleNamespace(), 4, 2, alloc, plan()
                    )
                self.assertEqual([r.output_ids for r in reqs], once)
            self.assertEqual(alloc.freed, [])

    def test_commit_failure_does_not_append_twice_or_free(self):
        import test_sr_fixed_accept as fixed

        reqs = [fixed._Req(), fixed._Req()]
        state = SRFixedAcceptState(2, 4, 4, 4, "cpu")
        alloc = fixed.NPUPagedTokenToKVPoolAllocator(4)
        batch, _ = fixed.FixedAcceptFinalizeTest()._batch(
            reqs, [1, 1], torch.arange(8), alloc.kv_buffer
        )
        plan = SRRPDHostPlan(
            [[0, -1, -1, -1], [4, -1, -1, -1]],
            [[1, 0, 0, 0], [2, 0, 0, 0]],
            [0, 0],
            32,
            rpd_batch_key(reqs),
        )
        state.inject_error = "move"
        with self.assertRaisesRegex(RuntimeError, "kv move failed"):
            state.finalize_from_host(batch, SimpleNamespace(), 4, 2, alloc, plan)
        once = [list(req.output_ids) for req in reqs]
        self.assertEqual(once, [[1], [2]])
        self.assertEqual(alloc.freed, [])
        with self.assertRaisesRegex(RuntimeError, "consumed"):
            state.finalize_from_host(batch, SimpleNamespace(), 4, 2, alloc, plan)
        self.assertEqual([list(req.output_ids) for req in reqs], once)
        self.assertEqual(alloc.freed, [])

    def test_page_sort_failure_after_host_commit_is_not_retried(self):
        import test_sr_fixed_accept as fixed

        reqs = [fixed._Req()]
        state = SRFixedAcceptState(1, 6, 15, 128, "cpu")
        state.metrics = SimpleNamespace(counts=Counter(), paths=Counter(), active=False)
        alloc = fixed.NPUPagedTokenToKVPoolAllocator(128)
        batch, _ = fixed.FixedAcceptFinalizeTest()._batch(
            reqs, [248], torch.arange(15) + 248, alloc.kv_buffer
        )
        plan = SRRPDHostPlan(
            [list(range(6))], [list(range(10, 16))], [5], 32, rpd_batch_key(reqs)
        )
        before = alloc.free_pages.clone()
        failure = RuntimeError("aclnnSort failed")
        with patch.object(
            alloc,
            "free_unique_pages",
            side_effect=lambda pages: fixed.apply_free_unique_pages(alloc, pages),
        ) as release, patch("torch.sort", side_effect=failure) as sort:
            with self.assertRaises(RuntimeError) as caught:
                state.finalize_from_host(batch, SimpleNamespace(), 128, 3, alloc, plan)
            self.assertIs(caught.exception, failure)
            self.assertTrue(plan.consumed)
            self.assertEqual(reqs[0].output_ids, list(range(10, 16)))
            self.assertTrue(torch.equal(alloc.free_pages, before))
            self.assertEqual(state.metrics.counts["fixed_accept_slot_output_alloc"], 3)
            self.assertEqual(
                state.metrics.counts["fixed_accept_slot_output_bytes"], 104
            )
            with self.assertRaisesRegex(RuntimeError, "consumed"):
                state.finalize_from_host(batch, SimpleNamespace(), 128, 3, alloc, plan)
            sort.assert_called_once()
            release.assert_called_once()
            self.assertEqual(reqs[0].output_ids, list(range(10, 16)))
            self.assertTrue(torch.equal(alloc.free_pages, before))

    def test_non_rpd_packet_has_no_edge_tail(self):
        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
            SRRoundMetrics,
        )

        plain_metrics = SRRoundMetrics("Target")
        edged_metrics = SRRoundMetrics("Target")
        plain = VerifyInputPacket()
        with plain_metrics.round():
            plain.load(
                [2],
                [[1, 2, 3, 4, 5]],
                [[-1, 0, 1, 2, 3]],
                [list(range(5))],
                2,
                3,
                6,
                "cpu",
                metrics=plain_metrics,
            )
        edged = VerifyInputPacket()
        with edged_metrics.round():
            edged.load(
                [2],
                [[1, 2, 3, 4, 5]],
                [[-1, 0, 1, 2, 3]],
                [list(range(5))],
                2,
                3,
                6,
                "cpu",
                metrics=edged_metrics,
                rpd_vocab=32,
            )
        self.assertIsNone(plain.rpd_input)
        self.assertNotIn("rpd_input_edge_bytes", plain_metrics.counts)
        self.assertIsNotNone(edged.rpd_input)
        self.assertGreater(edged_metrics.counts["rpd_input_edge_bytes"], 0)
        self.assertEqual(
            edged_metrics.counts["rpd_input_edge_bytes"],
            edged.rpd_input.edge_index_cpu.numel() * 8,
        )
        for metrics in (plain_metrics, edged_metrics):
            self.assertGreater(metrics.host["verify_packet_fill"], 0)
            self.assertEqual(
                metrics.host_max["verify_packet_fill"],
                metrics.host["verify_packet_fill"],
            )

    def test_logged_layout_reordering_and_capacity_changes(self):
        # The pasted NPU run uses K=3, S=5, W=15. Select complete ancestors
        # from the flattened beam tree, with three siblings at each level.
        packet, workspace = VerifyInputPacket(), SRRPDWorkspace()
        parents = [-1, 0, 1, 2, 3, 4, 5, 12, 13, 14, 21, 22, 23]
        selected = [0, 1, 2, 3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 31]
        previous = None
        for order in ([0], [2, 0, 1], [3, 1, 0, 2], [1, 0], [0]):
            bs = len(order)
            _, pp, ss, _ = packet.load(
                [b + 1 for b in order],
                [[(i + b) % 32 for i in range(14)] for b in order],
                [parents] * bs,
                [selected] * bs,
                3,
                5,
                15,
                "cpu",
                rpd_vocab=32,
            )
            context = packet.rpd_input
            reference = build_tree_kernel_efficient_ref(
                pp, ss, torch.tensor([b * 7 for b in order]), 3, 5, 15
            )
            for actual, expected in zip(context.tree[1:], reference[2:]):
                torch.testing.assert_close(actual, expected)
            if previous is not None:
                self.assertFalse(previous[0].current())
                torch.testing.assert_close(previous[0].tree, previous[1])
            logits = torch.full((bs * 15, 32), -4.0)
            for b in range(bs):
                for slot in range(15):
                    child = int(context.tree[2, b, slot])
                    token = int(context.tree[0, b, child]) if child >= 0 else 0
                    logits[b * 15 + slot, token] = 0
            pred = torch.zeros(bs * 15 + 1, dtype=torch.int32)
            index = torch.full((bs, 6), -1, dtype=torch.int32)
            length = torch.zeros(bs, dtype=torch.int32)
            rpd._verify_tree_rpd_compact(
                pred, index, length, *context.tree, logits, 0, True
            )
            self.assertIsNone(workspace.prepare(logits, context))
            plan = verify_sr_rpd_host(logits, context, workspace, 0, 6)
            self.assertEqual(plan.rows, index.tolist())
            self.assertEqual(plan.pre_lengths, [5] * bs)
            previous = (context, context.tree.clone())

    def test_unknown_commit_completion_blocks_a_new_plan(self):
        import test_sr_fixed_accept as fixed

        reqs = [fixed._Req(), fixed._Req()]
        state = SRFixedAcceptState(2, 4, 4, 4, "cpu")
        alloc = fixed.NPUPagedTokenToKVPoolAllocator(4)
        batch, _ = fixed.FixedAcceptFinalizeTest()._batch(
            reqs, [1, 1], torch.arange(8), alloc.kv_buffer
        )

        def new_plan():
            return SRRPDHostPlan(
                [[0, -1, -1, -1], [4, -1, -1, -1]],
                [[1, 0, 0, 0], [2, 0, 0, 0]],
                [0, 0],
                32,
                rpd_batch_key(reqs),
            )

        module = "sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept"
        with patch(
            module + ".submit_copy",
            side_effect=SRTransferUnresolved("unknown commit copy"),
        ):
            with self.assertRaises(SRTransferUnresolved):
                state.finalize_from_host(
                    batch, SimpleNamespace(), 4, 2, alloc, new_plan()
                )
        once = [list(req.output_ids) for req in reqs]
        self.assertEqual(once, [[1], [2]])
        self.assertEqual(alloc.freed, [])
        # A different plan also must not overwrite the possibly live H2D
        # source or append tokens. Same-plan consumption alone is insufficient.
        with self.assertRaises(SRTransferUnresolved):
            state.finalize_from_host(batch, SimpleNamespace(), 4, 2, alloc, new_plan())
        self.assertEqual([req.output_ids for req in reqs], once)
        self.assertEqual(alloc.freed, [])

    def test_production_stop_checks_reasoning_and_truncated_statistics(self):
        import test_sr_fixed_accept as fixed

        names = [
            "check_finished",
            "_check_token_based_finish",
            "_check_vocab_boundary_finish",
            "_check_str_based_finish",
            "update_reasoning_tokens",
        ]
        namespace = dict(re=re)
        for name in (
            "FINISH_LENGTH",
            "FINISH_MATCHED_TOKEN",
            "FINISH_MATCHED_STR",
            "FINISHED_MATCHED_REGEX",
        ):
            namespace[name] = lambda **kw: SimpleNamespace(**kw)
        methods = fixed._load_functions(
            fixed._REPO / "python/sglang/srt/managers/schedule_batch.py",
            names,
            namespace,
        )

        def request(mode):
            req = fixed._Req()
            req.kv_committed_len = req.kv_allocated_len = 1
            req.to_finish = None
            req.suppress_local_finish = False
            req.grammar = req.tokenizer = None
            req.vocab_size = 32
            req.eos_token_ids = {8} if mode == "eos" else set()
            req.decoded_text = ""
            req.tail_str = lambda: "".join(
                {7: "A", 8: "B", 9: "C"}.get(t, "x") for t in req.output_ids
            )
            req.sampling_params = SimpleNamespace(
                max_new_tokens=2 if mode == "length" else 100,
                ignore_eos=mode == "ignore_eos",
                stop_token_ids={8} if mode in ("token", "ignore_eos") else set(),
                stop_strs=["AB"] if mode == "string" else [],
                stop_regex_strs=["A."] if mode == "regex" else [],
            )
            req.require_reasoning = True
            req._is_reasoning_over = False
            req.reasoning_tokens = 0
            for name, method in methods.items():
                setattr(req, name, MethodType(method, req))
            return req

        rows, tokens = [[0, 2, 1, -1], [4, 6, 5, -1]], [[7, 8, 9, 0], [10, 11, 12, 0]]
        for mode in (
            "eos",
            "token",
            "length",
            "string",
            "regex",
            "ignore_eos",
            "continue",
        ):
            results = []
            for host in (False, True):
                reqs = [request(mode), request("continue")]
                alloc = fixed.NPUPagedTokenToKVPoolAllocator(4)
                batch, _ = fixed.FixedAcceptFinalizeTest()._batch(
                    reqs, [1, 1], torch.arange(8), alloc.kv_buffer
                )
                batch.model_config.think_end_id = 8
                state = SRFixedAcceptState(2, 4, 4, 4, "cpu")
                if host:
                    plan = SRRPDHostPlan(rows, tokens, [2, 2], 32, rpd_batch_key(reqs))
                    result = state.finalize_from_host(
                        batch, SimpleNamespace(), 4, 2, alloc, plan
                    )
                else:
                    predict, index, length = state.bind_verify_buffers(2)
                    predict.zero_()
                    index.copy_(torch.tensor(rows, dtype=torch.int32))
                    length.fill_(2)
                    for row, token_row in zip(rows, tokens):
                        for idx, tok in zip(row, token_row):
                            if idx >= 0:
                                predict[idx] = tok
                    result = state.finalize(
                        batch, SimpleNamespace(), 4, 2, alloc, index, predict, length
                    )
                count = 3 if mode in ("ignore_eos", "continue") else 2
                self.assertEqual(reqs[0].output_ids, tokens[0][:count])
                self.assertEqual(reqs[0].kv_committed_len, 1 + count)
                self.assertEqual(reqs[0].spec_accepted_tokens, 2)
                self.assertEqual(reqs[0].histogram, [2])
                self.assertEqual(reqs[0].reasoning_tokens, 2)
                results.append((result, alloc, reqs))
            old, new = results
            torch.testing.assert_close(new[0].verified_id, old[0].verified_id)
            torch.testing.assert_close(new[1].kv_buffer, old[1].kv_buffer)
            self.assertEqual(
                new[0].accept_length_per_req_cpu, old[0].accept_length_per_req_cpu
            )
            self.assertEqual(new[0].tree_paths, old[0].tree_paths)
            torch.testing.assert_close(
                new[0].draft_verified_id, old[0].draft_verified_id
            )


if __name__ == "__main__":
    unittest.main()
