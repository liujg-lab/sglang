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
    def test_host_selection_uses_owned_topology_and_no_tensor_results(self):
        packet, _ = packet_input(2)
        context = packet.rpd_input
        logits = torch.randn(12, 32)
        workspace = SRRPDWorkspace()
        workspace.prepare(logits, context)
        with patch.object(rpd, "_rpd_host_topology", side_effect=AssertionError("rebuilt topology")), patch.object(
            rpd, "_rpd_compact_select", side_effect=AssertionError("intermediate tensors")
        ), patch.object(torch, "tensor", side_effect=AssertionError("tensor allocation")), patch.object(
            torch, "full", side_effect=AssertionError("tensor allocation")
        ):
            plan = verify_sr_rpd_host(logits, context, workspace, .2, 4)
        self.assertTrue(all(isinstance(row, list) for row in plan.rows))
        before = copy.deepcopy(context.topology)
        packet_input(1, packet)
        self.assertEqual(context.topology.candidates, before.candidates)
        self.assertEqual(context.topology.children, before.children)
        self.assertEqual(context.topology.edge_pos, before.edge_pos)

    def test_python_selection_fold_and_nan_match_independent_reference(self):
        # Duplicate retrieve writes across requests are folded globally, in
        # request/path order, before any token rows are materialized.
        tree = torch.tensor(
            [
                [[1, 3, 3], [2, 4, 4]],
                [[0, 1, 2], [0, 4, 5]],
                [[1, -1, -1], [1, -1, -1]],
                [[-1, 2, -1], [-1, 2, -1]],
            ]
        )
        edges = [(0, 1, 0), (0, 2, 0), (1, 1, 0), (1, 2, 0)]
        topology = rpd._rpd_host_topology(tree, edges)
        star = [[3, 8, 9], [4, 10, 11]]
        stats = [[float("nan"), 0.0, float("inf"), 0.0], [0.0, 1.0, float("inf"), 1.0]]
        rows, lengths, folded = rpd._rpd_select_host(
            topology, star, stats, 0.0, True, 3, 6, 7
        )
        self.assertEqual(rows, [[0, 1, -1], [0, 4, -1]])
        self.assertEqual(lengths, [1, 1])
        self.assertEqual(folded, {0: 4, 1: 8, 4: 10})

    def test_select_host_rejects_overlong_path_and_retrieve_outside_rows(self):
        tree = torch.tensor(
            [
                [[1, 3, 3], [2, 4, 4]],
                [[0, 1, 2], [0, 4, 5]],
                [[1, -1, -1], [1, -1, -1]],
                [[-1, 2, -1], [-1, 2, -1]],
            ]
        )
        edges = [(0, 1, 0), (0, 2, 0), (1, 1, 0), (1, 2, 0)]
        topology = rpd._rpd_host_topology(tree, edges)
        star = [[3, 8, 9], [4, 10, 11]]
        stats = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        with self.assertRaises(RuntimeError):
            rpd._rpd_select_host(topology, star, stats, 0.0, True, 1, 6, 7)
        topology.retrieve[0][0] = 6
        with self.assertRaises(ValueError):
            rpd._rpd_select_host(topology, star, stats, 0.0, True, 3, 6, 7)

    def test_topology_matches_reference_and_survives_packet_reuse(self):
        packet, (_, parents, indices, _) = packet_input()
        context = packet.rpd_input
        reference = build_tree_kernel_efficient_ref(
            parents, indices, torch.tensor([0, 5]), 2, 3, 6
        )
        for actual, expected in zip(context.tree[1:], reference[2:]):
            torch.testing.assert_close(actual, expected)
        rows = int(context.tree.shape[1] * context.tree.shape[2])
        ref_edges, ref_index = rpd._rpd_compact_edges(context.tree, rows, context.vocab)
        ref_topo = rpd._rpd_host_topology(context.tree, ref_edges)
        self.assertEqual(context.topology.candidates, ref_topo.candidates)
        self.assertEqual(context.topology.retrieve, ref_topo.retrieve)
        self.assertEqual(context.topology.children, ref_topo.children)
        self.assertEqual(context.topology.edges, ref_topo.edges)
        self.assertEqual(context.topology.edge_pos, ref_topo.edge_pos)
        if ref_index is None:
            self.assertEqual(tuple(context.edge_index_cpu.shape), (2, 0))
        else:
            torch.testing.assert_close(context.edge_index_cpu, ref_index)
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
            _, pp, ss, tokens = packet.load(
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
            self.assertEqual(packet.no_draft_rows, (True,))
            self.assertEqual(pp.tolist(), [[-1, -1, -1, -1, -1]])
            self.assertEqual(ss.tolist(), [[0, 0, 0, 0, 0]])
            self.assertEqual(context.tree[2, 0].tolist(), [-1, -1, -1, -1, -1, -1])
            if name == "missing":
                self.assertEqual(tokens.tolist(), [[3, 0, 0, 0, 0]])
                self.assertEqual(context.tree[0, 0].tolist(), [2, 3, 0, 0, 0, 0])
            else:
                self.assertEqual(tokens.tolist(), [[0, 0, 0, 0, 0]])
                self.assertEqual(context.tree[0, 0].tolist(), [2, 0, 0, 0, 0, 0])
            logits = torch.zeros(6, 32)
            logits[:, 3] = 10
            workspace = SRRPDWorkspace()
            workspace.prepare(logits, context)
            plan = verify_sr_rpd_host(logits, context, workspace, 0.0, 4)
            self.assertEqual(plan.pre_lengths, [0])
            trees[name] = context.tree[2, 0].tolist()
        self.assertEqual(trees["missing"], trees["empty"])
        self.assertEqual(trees["empty"], trees["short"])
        self.assertEqual(trees["short"], trees["padded"])

    def test_cycle_declines_and_missing_parent_is_disconnected(self):
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            plan_draft_rows,
        )

        tokens = [[1, 2, 3, 4, 5]]
        selected = [[0, 1, 2, 3, 4]]
        cycled = plan_draft_rows(
            tokens, [[-1, 2, 1, 2, 3]], selected, 2, 3, 6, vocab=32
        )
        self.assertEqual(cycled.invalid_tree_rows, 1)
        self.assertEqual(cycled.no_draft, [True])
        self.assertEqual(cycled.parent_slots, [[-1, -1, -1, -1, -1, -1]])
        owner = SimpleNamespace(generation=1, unresolved=False)
        verified = torch.tensor([0])
        token_row = torch.tensor(tokens)
        cycled_tree = build_sr_rpd_input(
            verified,
            token_row,
            torch.tensor([[-1, 2, 1, 2, 3]]),
            torch.tensor(selected),
            topk=2,
            steps=3,
            width=6,
            vocab=32,
            owner=owner,
            generation=1,
            parent_slots=cycled.parent_slots,
        )
        self.assertEqual(cycled_tree.edges, [])
        self.assertEqual(cycled_tree.tree[2, 0].tolist(), [-1, -1, -1, -1, -1, -1])

        missing = plan_draft_rows(
            tokens, [[-1, 99, 1, 2, 3]], selected, 2, 3, 6, vocab=32
        )
        self.assertEqual(missing.invalid_tree_rows, 0)
        self.assertEqual(missing.no_draft, [False])
        self.assertEqual(missing.disconnected, [[2, 3]])
        context = build_sr_rpd_input(
            verified,
            token_row,
            torch.tensor([[-1, 99, 1, 2, 3]]),
            torch.tensor(selected),
            topk=2,
            steps=3,
            width=6,
            vocab=32,
            owner=owner,
            generation=1,
            parent_slots=missing.parent_slots,
        )
        self.assertIsNotNone(context)
        self.assertEqual(context.tree[2, 0, 1].item(), -1)

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
                    with self.assertRaises(SRTransferUnresolved):
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

    def test_statistics_submission_failures_retain_all_sources_and_refuse_reuse(self):
        import sglang.srt.speculative.standalone_remote.sr_rpd as module

        for stage in ("max", "edge", "first_copy", "second_copy", "record", "wait"):
            with self.subTest(stage=stage):
                packet, _ = packet_input()
                context = packet.rpd_input
                ws = SRRPDWorkspace()
                logits = torch.zeros(12, 32)
                self.assertIsNone(ws.prepare(logits, context))
                ws.event = Mock()
                error = RuntimeError(stage)
                real_max, real_copy = torch.max, torch.Tensor.copy_
                real_gather = module._gather_edge_stats
                copy_calls = []

                def reduction(*args, **kwargs):
                    self.assertIsNotNone(ws.holds)
                    self.assertTrue(any(t is context.edge_index for t in ws.holds))
                    if stage == "max":
                        raise error
                    return real_max(*args, **kwargs)

                def gather(*args):
                    if stage == "edge":
                        raise error
                    return real_gather(*args)

                def copy_into(dest, src, **kwargs):
                    if dest.data_ptr() in (
                        ws.star_host.data_ptr(),
                        ws.stats_host.data_ptr(),
                    ):
                        copy_calls.append(dest)
                        if stage == (
                            "first_copy" if len(copy_calls) == 1 else "second_copy"
                        ):
                            raise error
                    return real_copy(dest, src, **kwargs)

                if stage == "record":
                    ws.event.record.side_effect = error
                if stage == "wait":
                    ws.event.synchronize.side_effect = error
                with (
                    patch.object(
                        torch.Tensor,
                        "device",
                        new_callable=PropertyMock,
                        return_value=SimpleNamespace(type="npu"),
                    ),
                    patch.object(torch, "max", side_effect=reduction) as maximum,
                    patch.object(
                        module, "_gather_edge_stats", side_effect=gather
                    ) as edge,
                    patch.object(torch.Tensor, "copy_", copy_into),
                ):
                    with self.assertRaises(SRTransferUnresolved) as caught:
                        verify_sr_rpd_host(logits, context, ws, 0.2, 4)
                self.assertIs(caught.exception.__cause__, error)
                self.assertEqual(maximum.call_count, 1)
                self.assertEqual(edge.call_count, int(stage != "max"))
                self.assertTrue(ws.unresolved)
                self.assertTrue(packet.unresolved)
                self.assertIsNotNone(ws.holds)
                with self.assertRaises(SRTransferUnresolved):
                    ws.prepare(logits, context)
                with self.assertRaisesRegex(RuntimeError, "unresolved"):
                    packet_input(packet=packet)
                with self.assertRaises(SRTransferUnresolved):
                    ws.statistics(logits, context)

    def test_statistics_preflight_failures_do_not_submit_or_poison(self):
        import sglang.srt.speculative.standalone_remote.sr_rpd as module

        for case in ("capacity", "buffer", "kernel"):
            with self.subTest(case=case):
                packet, _ = packet_input()
                context = packet.rpd_input
                ws = SRRPDWorkspace()
                logits = torch.zeros(12, 32)
                ws.prepare(logits, context)
                if case == "capacity":
                    ws.rows_cap = 1
                elif case == "buffer":
                    context.edge_index = context.edge_index.clone()
                elif case == "kernel":
                    ws._edge_probe.clear()
                with (
                    patch.object(torch, "max") as maximum,
                    patch.object(
                        module,
                        "_edge_gather_impl",
                        side_effect=ImportError("kernel unavailable"),
                    ),
                ):
                    with self.assertRaises((RuntimeError, ImportError)):
                        verify_sr_rpd_host(logits, context, ws, 0.2, 4)
                maximum.assert_not_called()
                self.assertFalse(ws.unresolved)
                self.assertFalse(packet.unresolved)
                self.assertIsNone(ws.holds)
                packet_input(packet=packet)
                self.assertIsNotNone(packet.rpd_input)

    def test_missing_edge_kernel_declines_before_workspace_allocation(self):
        packet, _ = packet_input()
        ws = SRRPDWorkspace()
        with (
            patch(
                "sglang.srt.speculative.standalone_remote.sr_rpd._edge_gather_impl",
                side_effect=ImportError("missing Triton"),
            ),
            patch("torch.empty") as allocate,
        ):
            self.assertEqual(
                ws.prepare(torch.zeros(12, 32), packet.rpd_input), "edge_kernel"
            )
        allocate.assert_not_called()
        self.assertIsNone(ws.key)

    def test_prepare_caches_edge_kernel_probe_by_device_and_dtype(self):
        import sglang.srt.speculative.standalone_remote.sr_rpd as module

        packet, _ = packet_input()
        context = packet.rpd_input
        self.assertGreater(len(context.edges), 0)
        ws = SRRPDWorkspace()
        real = module._edge_gather_impl
        calls = []

        def spy(logits):
            calls.append((logits.device.type, logits.dtype))
            return real(logits)

        with patch.object(module, "_edge_gather_impl", side_effect=spy):
            logits = torch.zeros(12, 32)
            self.assertIsNone(ws.prepare(logits, context))
            self.assertIsNone(ws.prepare(logits, context))
            self.assertEqual(calls, [("cpu", torch.float32)])
            logits_f16 = torch.zeros(12, 32, dtype=torch.float16)
            self.assertIsNone(ws.prepare(logits_f16, context))
            self.assertEqual(calls, [("cpu", torch.float32), ("cpu", torch.float16)])

        failed = SRRPDWorkspace()
        with (
            patch.object(
                module,
                "_edge_gather_impl",
                side_effect=ImportError("missing Triton"),
            ) as probe,
            patch("torch.empty") as allocate,
        ):
            self.assertEqual(failed.prepare(logits, context), "edge_kernel")
            self.assertEqual(failed.prepare(logits, context), "edge_kernel")
        self.assertEqual(probe.call_count, 1)
        allocate.assert_not_called()
        self.assertIsNone(failed.key)

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
                parent_slots=[[-1] for _ in range(bs)],
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
                        [0], [[1, 1]], [[]], [[0, 1]], 2, 1, 3, "cpu", rpd_vocab=4
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

    def test_special_logits_host_plan_matches_compact_at_tau0(self):
        workspace = SRRPDWorkspace()
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for tau in (0.0, 0.2):
                packet, _ = packet_input()
                context = packet.rpd_input
                self.assertGreater(len(context.edges), 0)
                logits = torch.zeros(12, 32, dtype=dtype)
                logits[0, 0] = float("nan")
                logits[0, 1] = float("inf")
                logits[1, :] = float("-inf")
                logits[2, 3] = float("nan")
                logits[4, 5] = float("inf")
                logits[6, 2] = float("-inf")
                ref_idx = torch.full((2, 4), -1, dtype=torch.int32)
                ref_len = torch.zeros(2, dtype=torch.int32)
                rpd._verify_tree_rpd_compact(
                    torch.full((13,), -1, dtype=torch.int32),
                    ref_idx,
                    ref_len,
                    *context.tree,
                    logits,
                    rpd.rpd_gap_max(tau),
                    tau == 0.0,
                )
                metrics = SimpleNamespace(counts=Counter())
                self.assertIsNone(workspace.prepare(logits, context, metrics))
                plan = verify_sr_rpd_host(logits, context, workspace, tau, 4)
                self.assertEqual(plan.rows, ref_idx.tolist())
                self.assertEqual(plan.pre_lengths, ref_len.tolist())
                self.assertEqual(metrics.counts["rpd_host_edge_gather"], 0)

    def test_tau0_nan_gap_uses_python_comparison(self):
        """An equal-length tau=0 tie keeps the NaN gap instead of treating it as 0."""
        cand = torch.tensor([[0, 7, 7]], dtype=torch.int64)
        retr = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        nxt = torch.tensor([[1, -1, -1]], dtype=torch.int64)
        sibling = torch.tensor([[-1, 2, -1]], dtype=torch.int64)
        tree = torch.stack((cand, retr, nxt, sibling))
        edges = [(0, 1, 0), (0, 2, 0)]
        star = torch.tensor([[7, 3, 4]], dtype=torch.int64)
        # Both children match the root argmax. The first gap is NaN; the
        # second is -1. Python ``-1 < nan`` is false, so the first sibling
        # stays. Replacing the NaN with 0 makes ``-1 < 0`` and selects the other.
        stats = torch.tensor([[float("nan"), 0.0], [0.0, 1.0]], dtype=torch.float32)
        accept, lengths, _, _ = rpd._rpd_compact_select(
            tree, edges, star, stats, 0.0, True, 3, 3, 4
        )
        self.assertEqual(lengths.tolist(), [1])
        self.assertEqual(accept[0, :2].tolist(), [0, 1])
        zeroed = stats.clone()
        zeroed[0, 0] = 0.0
        accept_zero, _, _, _ = rpd._rpd_compact_select(
            tree, edges, star, zeroed, 0.0, True, 3, 3, 4
        )
        self.assertEqual(accept_zero[0, :2].tolist(), [0, 2])

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

    def test_prepare_admits_cuda_without_npu_edge_kernel(self):
        import sys

        import sglang.srt.speculative.standalone_remote.sr_rpd as module

        packet, _ = packet_input()
        context = packet.rpd_input
        logits = torch.zeros(12, 32)
        workspace = SRRPDWorkspace()
        npu_mod = "sglang.srt.speculative.standalone_remote.sr_rpd_kernels_npu"
        sys.modules.pop(npu_mod, None)
        cuda = torch.device("cuda")
        real_empty = torch.empty
        real_get = torch.get_device_module

        def empty(*args, **kwargs):
            device = kwargs.get("device", None)
            if getattr(device, "type", None) == "cuda":
                kwargs["device"] = "cpu"
            return real_empty(*args, **kwargs)

        class _Event:
            def record(self):
                return None

        class _CudaMod:
            @staticmethod
            def Event():
                return _Event()

        def get_device_module(name):
            if name == "cuda":
                return _CudaMod
            return real_get(name)

        with patch.object(
            torch.Tensor, "device", new_callable=PropertyMock, return_value=cuda
        ), patch.object(module, "pin_supported", return_value=True), patch.object(
            module, "event_supported", return_value=True
        ), patch.object(
            module,
            "alloc_host",
            side_effect=lambda shape, dtype, device: (real_empty(shape, dtype=dtype), True),
        ), patch.object(torch, "empty", side_effect=empty), patch.object(
            torch, "get_device_module", side_effect=get_device_module
        ):
            self.assertIs(module._edge_gather_impl(logits), module._gather_edge_stats)
            self.assertIsNone(workspace.prepare(logits, context))
        self.assertNotIn(npu_mod, sys.modules)

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
        workspace = SRRPDWorkspace()
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
                sr_rpd_workspace=workspace,
            )
        self.assertEqual(result, "committed")
        penalizer.apply.assert_called_once()
        finalize.assert_called_once()

        class StopAfterStage(Exception):
            pass

        legacy = Mock(side_effect=AssertionError("shared compact path"))
        namespace["verify_tree_rpd"] = legacy
        namespace["apply_custom_logit_processor"] = Mock()
        batch.model_config = SimpleNamespace(think_end_id=None)
        apply_cases = ("off", "logprob", "hidden", "grammar", "processor", "simulate")
        import sglang.srt.speculative.standalone_remote.sr_rpd as sr_rpd_mod

        real_stage = sr_rpd_mod.stage_v1_result
        for case in apply_cases + ("missing", "stale", "noncontiguous"):
            case_reqs = [fixed._Req(), fixed._Req()]
            for req in case_reqs:
                req.grammar = None
            packet, _ = packet_input(reqs=case_reqs)
            ctx = packet.rpd_input
            batch.reqs = case_reqs
            spec.sr_rpd_input = ctx
            spec.retrive_index = ctx.tree[1]
            spec.retrive_next_token = ctx.tree[2]
            spec.retrive_next_sibling = ctx.tree[3]
            spec.draft_token = ctx.tree[0].flatten()
            if case == "missing":
                spec.sr_rpd_input = None
            if case == "stale":
                packet_input(packet=packet, reqs=case_reqs)
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
            staged = []

            def watch_stage(workspace, indices, tokens, lengths, device, select=()):
                staged.append((list(indices), list(tokens), list(lengths)))
                real_stage(workspace, indices, tokens, lengths, device, select)
                raise StopAfterStage

            def forbid_compact(*_args, **_kwargs):
                raise AssertionError("compact transfer")

            with patch.object(
                state,
                "finalize_from_host",
                side_effect=AssertionError("host fallback replay"),
            ), patch.object(
                state,
                "bind_verify_buffers",
                side_effect=AssertionError("greedy buffers"),
            ), patch.object(
                rpd, "_compact_await_stats", forbid_compact
            ), patch.object(
                rpd, "_compact_host_read", forbid_compact
            ), patch.object(
                rpd, "_compact_device_write", forbid_compact
            ), patch.object(sr_rpd_mod, "stage_v1_result", watch_stage):
                if case in apply_cases:
                    with self.assertRaises(StopAfterStage):
                        verify(
                            spec,
                            batch,
                            SimpleNamespace(next_token_logits=logits),
                            SimpleNamespace(is_not_in_free_group=True),
                            4,
                            prepare_local_draft_hidden=case == "hidden",
                            sr_accept_state=None if case == "off" else state,
                            sr_rpd_workspace=workspace,
                        )
                elif case == "noncontiguous":
                    with self.assertRaisesRegex(AssertionError, "shared compact path"):
                        verify(
                            spec,
                            batch,
                            SimpleNamespace(next_token_logits=logits),
                            SimpleNamespace(is_not_in_free_group=True),
                            4,
                            prepare_local_draft_hidden=False,
                            sr_accept_state=state,
                            sr_rpd_workspace=workspace,
                        )
                else:
                    with self.assertRaises(RuntimeError):
                        verify(
                            spec,
                            batch,
                            SimpleNamespace(next_token_logits=logits),
                            SimpleNamespace(is_not_in_free_group=True),
                            4,
                            prepare_local_draft_hidden=False,
                            sr_accept_state=state,
                            sr_rpd_workspace=workspace,
                        )
            if case == "noncontiguous":
                legacy.assert_called_once()
                penalizer.apply.assert_called_once()
                self.assertIn("host_plan_fallback_logits_layout", workspace._logged_paths)
            elif case in ("missing", "stale"):
                legacy.assert_not_called()
                penalizer.apply.assert_not_called()
            else:
                legacy.assert_not_called()
                penalizer.apply.assert_called_once()
            if case not in apply_cases:
                continue
            ref_predict = torch.empty((13,), dtype=torch.int32)
            ref_index = torch.full((2, 4), -1, dtype=torch.int32)
            ref_length = torch.empty((2,), dtype=torch.int32)
            rpd.verify_tree_rpd(
                ref_predict,
                ref_index,
                ref_length,
                ctx.tree[0],
                ctx.tree[1],
                ctx.tree[2],
                ctx.tree[3],
                logits,
                0.2,
            )
            got_index, got_tokens, got_length = staged[0]
            ref_flat = [idx for idx in ref_index.reshape(-1).tolist() if idx >= 0]
            self.assertEqual(got_index, ref_flat)
            self.assertEqual(got_length, ref_length.tolist())
            self.assertEqual(got_tokens, [int(ref_predict[idx]) for idx in ref_flat])

    def test_host_plan_prepare_failure_falls_back_to_verify_tree_rpd(self):
        import test_sr_fixed_accept as fixed

        reqs = [fixed._Req(), fixed._Req()]
        for req in reqs:
            req.grammar = None
        packet, _ = packet_input(reqs=reqs)
        context = packet.rpd_input
        state = SRFixedAcceptState(2, 4, 6, 4, "cpu")
        state.metrics = SimpleNamespace(paths=Counter())

        class Sampling:
            is_all_greedy = True
            has_custom_logit_processor = False
            logit_bias = None
            penalizer_orchestrator = SimpleNamespace(is_required=False)

            def __len__(self):
                return 2

        class _Bucket:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        batch = SimpleNamespace(
            reqs=reqs,
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            sampling_info=Sampling(),
            seq_lens=torch.tensor([1, 1], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([1, 1], dtype=torch.int32),
            out_cache_loc=torch.arange(12),
            req_pool_indices=torch.tensor([0, 1]),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((2, 16), dtype=torch.int32)
            ),
            spec_info=SimpleNamespace(hidden_states=None),
            model_config=SimpleNamespace(
                think_end_id=None, hidden_size=4, dtype=torch.float32
            ),
            device=torch.device("cpu"),
            return_logprob=False,
            has_grammar=False,
        )
        spec = SimpleNamespace(
            retrive_index=context.tree[1],
            retrive_next_token=context.tree[2],
            retrive_next_sibling=context.tree[3],
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
            verify_tree_rpd=rpd.verify_tree_rpd,
            select_hidden_states_for_draft=lambda *a, **kw: None,
            assign_req_to_token_pool_func=lambda *a, **kw: None,
            EagleDraftInput=_Bucket,
            EagleVerifyOutput=_Bucket,
        )
        verify = fixed._load_functions(
            fixed._REPO / "python/sglang/srt/speculative/eagle_info.py",
            ["verify"],
            namespace,
        )["verify"]
        workspace = SRRPDWorkspace()
        logits = torch.randn(12, 32)
        ref_logits = logits.clone()
        with patch.object(
            workspace, "prepare", return_value="edge_kernel"
        ) as prepare, patch.object(
            state,
            "finalize_from_host",
            side_effect=AssertionError("host plan must not commit"),
        ) as finalize, patch.object(
            state,
            "bind_verify_buffers",
            side_effect=AssertionError("greedy buffers"),
        ):
            result = verify(
                spec,
                batch,
                SimpleNamespace(next_token_logits=logits),
                SimpleNamespace(is_not_in_free_group=True, free=lambda slots: None),
                1,
                prepare_local_draft_hidden=False,
                sr_accept_state=state,
                sr_rpd_workspace=workspace,
            )
        prepare.assert_called_once()
        finalize.assert_not_called()
        self.assertIn("host_plan_fallback_edge_kernel", workspace._logged_paths)
        self.assertEqual(state.metrics.paths["fixed_accept_rpd_prepare"], 1)
        ref_predict = torch.empty((13,), dtype=torch.int32)
        ref_index = torch.full((2, 4), -1, dtype=torch.int32)
        ref_length = torch.empty((2,), dtype=torch.int32)
        rpd.verify_tree_rpd(
            ref_predict,
            ref_index,
            ref_length,
            context.tree[0],
            context.tree[1],
            context.tree[2],
            context.tree[3],
            ref_logits,
            0.2,
        )
        ref_flat = [idx for idx in ref_index.reshape(-1).tolist() if idx >= 0]
        self.assertEqual(result.accepted_indices.tolist(), ref_flat)
        self.assertEqual(result.accept_length_per_req_cpu, ref_length.tolist())
        self.assertEqual(
            result.verified_id.tolist(), [int(ref_predict[idx]) for idx in ref_flat]
        )

    def test_stage_v1_reuses_buffer_without_pinned_alloc_or_wait(self):
        import sglang.srt.speculative.standalone_remote.sr_rpd as module

        workspace = SRRPDWorkspace()
        allocs = {"host": 0}
        real_alloc = module.alloc_host

        def counting_alloc(*args, **kwargs):
            allocs["host"] += 1
            return real_alloc(*args, **kwargs)

        def forbid_wait(*_args, **_kwargs):
            raise AssertionError("stage must not wait")

        def forbid_empty(*_args, **_kwargs):
            raise AssertionError("stage must not allocate")

        with patch.object(module, "alloc_host", counting_alloc), patch.object(
            module, "wait_event", forbid_wait
        ):
            first_idx, first_tok, first_len, first_sel = module.stage_v1_result(
                workspace, [3, 4], [8, 9], [1], torch.device("cpu"), [0]
            )
            self.assertEqual(first_idx.tolist(), [3, 4])
            self.assertEqual(first_tok.tolist(), [8, 9])
            self.assertEqual(first_len.tolist(), [1])
            self.assertEqual(first_sel.tolist(), [0])
            self.assertEqual(allocs["host"], 1)
            held_dev = workspace._v1_dev
            held_host = workspace._v1_host
            with patch.object(module.torch, "empty", forbid_empty):
                second_idx, second_tok, second_len, second_sel = module.stage_v1_result(
                    workspace, [1], [7], [0, 0], torch.device("cpu")
                )
        self.assertEqual(allocs["host"], 1)
        self.assertIs(workspace._v1_dev, held_dev)
        self.assertIs(workspace._v1_host, held_host)
        self.assertEqual(second_idx.tolist(), [1])
        self.assertEqual(second_tok.tolist(), [7])
        self.assertEqual(second_len.tolist(), [0, 0])
        self.assertEqual(second_sel.tolist(), [])

    def test_device_cache_key_equates_unnumbered_and_indexed_device(self):
        from sglang.srt.speculative.standalone_remote import sr_transfer_staging

        class _Mod:
            def current_device(self):
                return 0

        # ``npu`` is not a builtin device type here. ``cuda`` has the same
        # unnumbered-versus-indexed mismatch that breaks buffer reuse.
        unnumbered = torch.device("cuda")
        indexed = torch.device("cuda", 0)
        self.assertNotEqual(unnumbered, indexed)
        with patch.object(
            sr_transfer_staging.torch, "get_device_module", return_value=_Mod()
        ):
            self.assertEqual(
                sr_transfer_staging.device_cache_key(unnumbered),
                sr_transfer_staging.device_cache_key(indexed),
            )

    def test_stage_v1_reuses_unnumbered_device_with_same_cache_key(self):
        import sglang.srt.speculative.standalone_remote.sr_rpd as module

        workspace = SRRPDWorkspace()

        def accel_key(device):
            dev = device if isinstance(device, torch.device) else torch.device(device)
            if dev.type in ("cpu", "cuda"):
                return ("cuda", 0)
            raise AssertionError(f"unexpected device {device}")

        def forbid_empty(*_args, **_kwargs):
            raise AssertionError("stage must reuse the device buffer")

        # The buffer stays on CPU, so ``tensor.device == torch.device("cuda")``
        # is false. Both that buffer and the unnumbered ``"cuda"`` request share
        # one cache key, which is what production resolves for ``"cuda"``.
        with patch.object(module, "device_cache_key", side_effect=accel_key):
            module.stage_v1_result(
                workspace, [3, 4], [8, 9], [1], torch.device("cpu"), [0]
            )
            self.assertEqual(workspace._v1_dev.device.type, "cpu")
            self.assertNotEqual(workspace._v1_dev.device, torch.device("cuda"))
            self.assertEqual(workspace._v1_key, ("cuda", 0))
            held_dev = workspace._v1_dev
            held_host = workspace._v1_host
            with patch.object(module.torch, "empty", forbid_empty):
                second_idx, second_tok, second_len, second_sel = module.stage_v1_result(
                    workspace, [1], [7], [0, 0], "cuda"
                )
        self.assertIs(workspace._v1_dev, held_dev)
        self.assertIs(workspace._v1_host, held_host)
        self.assertEqual(second_idx.tolist(), [1])
        self.assertEqual(second_tok.tolist(), [7])
        self.assertEqual(second_len.tolist(), [0, 0])
        self.assertEqual(second_sel.tolist(), [])

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
        plan.pre_lengths[1] = 2
        with self.assertRaises(RuntimeError):
            state.finalize_from_host(batch, SimpleNamespace(), 4, 2, alloc, plan)
        self.assertEqual([r.output_ids for r in reqs], [[], []])
        self.assertEqual(alloc.freed, [])

        reqs = [fixed._Req(), fixed._Req()]
        state = SRFixedAcceptState(2, 4, 6, 4, "cpu")
        alloc = fixed.NPUPagedTokenToKVPoolAllocator(4)
        packet, _ = packet_input(reqs=reqs)
        swapped = rpd_batch_key(list(reversed(reqs)))
        reason = state.reject_before_alloc(
            **fixed._admit_kwargs(
                state,
                verify_mode="rpd",
                logits=torch.zeros(12, 32),
                allocator=alloc,
                rpd_host_input=packet.rpd_input,
                rpd_live_batch_key=swapped,
            )
        )
        self.assertEqual(reason, "rpd_batch_key")
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
            self.assertEqual(state.metrics.counts["fixed_accept_slot_output_alloc"], 1)
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
