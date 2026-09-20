"""CPU contracts for SR tail ingest, using the real NPU dispatch with op spies.

AST-loading backend/scheduler methods avoids importing torch_npu and Triton;
the methods themselves execute, rather than being checked as source strings.
"""

import ast
import copy
import math
import os
import logging
import time
from dataclasses import dataclass, replace
from types import MethodType
from pathlib import Path
from types import SimpleNamespace as NS
from typing import List, Optional, Sequence, Tuple
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative.standalone_remote.drafter import sr_tail_extend as tail
from sglang.srt.speculative.standalone_remote.sr_round_metrics import SRRoundMetrics
from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
    get_sr_round_metrics,
)
from sglang.srt.speculative.standalone_remote.sr_tail_attention import (
    SRTailAttentionMetadata,
    build_tail_attention_metadata,
    copy_tail_attention_metadata_,
    fill_tail_attention_metadata_,
    pad_tail_attention_metadata,
    tail_graph_fits_pages,
    tail_graph_max_pages,
    validate_tail_forward_batch,
    widen_tail_block_tables,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
BACKEND = ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
SCHEDULER = (
    ROOT
    / "python/sglang/srt/speculative/standalone_remote/drafter/sr_draft_scheduler_mixin.py"
)


def load_functions(path, names, namespace, *, strip_imports=False, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if class_name is not None:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    functions = [
        copy.deepcopy(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != set(names):
        raise AssertionError("missing production methods")
    if strip_imports:
        for fn in functions:
            fn.body = [
                n for n in fn.body if not isinstance(n, (ast.Import, ast.ImportFrom))
            ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future] + functions, type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def paged_reference(query, keys, values, tables, lengths, scale):
    """Independent per-query oracle; physical page order is deliberately arbitrary."""
    page_size = keys.shape[1]
    keys = keys.reshape(-1, keys.shape[-2], keys.shape[-1])
    values = values.reshape(-1, values.shape[-2], values.shape[-1])
    result = []
    for row, length in enumerate(lengths):
        pos = torch.arange(length)
        slots = tables[row, pos // page_size].long() * page_size + pos % page_size
        k, v = keys[slots], values[slots]
        repeat = query.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        score = torch.einsum("hd,shd->hs", query[row], k) * scale
        result.append(torch.einsum("hs,shd->hd", score.softmax(-1), v))
    return torch.stack(result)


def make_backend(prefix, lengths, tables, *, fia=False, page_size=128):
    def atb(**kwargs):
        kwargs["out"].copy_(
            paged_reference(
                kwargs["query"],
                kwargs["key_cache"],
                kwargs["value_cache"],
                kwargs["block_table"],
                kwargs["context_lens"].tolist(),
                kwargs["scale_value"],
            )
        )

    def fia_op(query, keys, values, **kwargs):
        h = kwargs["num_key_value_heads"]
        out = paged_reference(
            query[:, 0],
            keys.reshape(keys.shape[0], page_size, h, -1),
            values.reshape(values.shape[0], page_size, h, -1),
            kwargs["block_table"],
            kwargs["actual_seq_lengths_kv"],
            kwargs["scale"],
        )
        return out.unsqueeze(1), None

    atb_spy, fia_spy = Mock(side_effect=atb), Mock(side_effect=fia_op)
    torch_proxy = NS(ops=NS(npu=NS(npu_fused_infer_attention_score=fia_spy)))
    namespace = dict(
        torch=torch_proxy,
        torch_npu=NS(_npu_paged_attention=atb_spy),
        AttentionType=NS(ENCODER_ONLY="encoder"),
        is_mla_preprocess_enabled=lambda: False,
    )
    methods = load_functions(
        BACKEND,
        ["_can_run_sr_tail_paged", "_run_sr_tail_paged", "forward_extend"],
        namespace,
    )
    backend = type("CPUAscendSpy", (), methods)()
    backend.use_fia, backend.use_mla, backend.use_alibi = fia, False, False
    backend.is_dllm_model = False
    backend.page_size = page_size
    backend.sr_tail_attention_paths = set()
    backend.forward_metadata = NS(
        sr_tail=build_tail_attention_metadata(prefix, lengths, tables)
    )
    return backend, atb_spy, fia_spy


def layer_info(layer_id=0):
    return NS(
        layer_id=layer_id,
        tp_q_head_num=2,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=2,
        v_head_dim=2,
        scaling=1 / math.sqrt(2),
        is_cross_attention=False,
        attn_type="decoder",
        sliding_window_size=-1,
        logit_cap=0,
    )


class FakePool:
    def __init__(self, page_size=128, layers=2):
        self.keys = [torch.zeros(12, page_size, 1, 2) for _ in range(layers)]
        self.values = [torch.zeros_like(k) for k in self.keys]
        self.writes = 0

    def set_kv_buffer(self, layer, slots, k, v):
        self.keys[layer.layer_id].view(-1, 1, 2)[slots.long()] = k.reshape(-1, 1, 2)
        self.values[layer.layer_id].view(-1, 1, 2)[slots.long()] = v.reshape(-1, 1, 2)
        self.writes += 1

    def get_key_buffer(self, layer_id):
        return self.keys[layer_id]

    def get_value_buffer(self, layer_id):
        return self.values[layer_id]


def forward_batch(prefix, lengths, slots, pool):
    positions = [p + j for p, n in zip(prefix, lengths) for j in range(n)]
    return NS(
        is_sr_tail_extend=True,
        batch_size=len(prefix),
        extend_prefix_lens_cpu=prefix,
        extend_seq_lens_cpu=lengths,
        extend_num_tokens=sum(lengths),
        input_ids=torch.ones(sum(lengths), dtype=torch.long),
        positions=torch.tensor(positions),
        out_cache_loc=slots,
        seq_lens_cpu=torch.tensor([p + n for p, n in zip(prefix, lengths)]),
        mrope_positions=None,
        token_to_kv_pool=pool,
        encoder_lens=None,
        forward_mode=NS(
            is_target_verify=lambda: False,
            is_draft_extend=lambda: False,
            is_draft_extend_v2=lambda: False,
        ),
    )


class TestTailAttention(unittest.TestCase):
    def test_real_metadata_initialization_uses_physical_page_starts(self):
        backend, _, _ = make_backend(
            [127, 129], [2, 1], torch.zeros(2, 2, dtype=torch.int32)
        )
        backend.draft_topk = 3
        backend.is_hybrid_swa = False
        backend._use_tree_shared_prefix = lambda: False
        backend._use_target_tree_paged_fia = lambda: False
        backend._paged_impl_selected = lambda: False
        backend._sr_tree_paged_meta = None
        backend._use_tree_draft_slot_gather = lambda batch: False
        mapping = torch.empty(2, 256, dtype=torch.int64)
        mapping[0] = torch.cat((torch.arange(512, 640), torch.arange(128, 256)))
        mapping[1] = torch.cat((torch.arange(896, 1024), torch.arange(256, 384)))
        batch = forward_batch(
            [127, 129], [2, 1], torch.tensor([639, 128, 257]), FakePool()
        )
        batch.req_to_token_pool = NS(req_to_token=mapping)
        batch.req_pool_indices = torch.tensor([0, 1])
        batch.seq_lens = batch.seq_lens_cpu.clone()
        batch.extend_seq_lens = torch.tensor([2, 1])
        batch.forward_mode.is_decode_or_idle = lambda: False
        method = load_functions(
            BACKEND,
            ["init_forward_metadata"],
            dict(
                torch=torch,
                ForwardMetadata=NS,
                build_tail_attention_metadata=build_tail_attention_metadata,
                np=NS(cumsum=lambda xs: torch.tensor(xs).cumsum(0).tolist()),
            ),
            class_name="AscendAttnBackend",
        )["init_forward_metadata"]
        method(backend, batch)
        md = backend.forward_metadata
        self.assertEqual(md.sr_tail.block_tables.tolist(), [[4, 1], [4, 1], [7, 2]])
        self.assertEqual(md.sr_tail.context_lens_list, [128, 129, 130])
        self.assertEqual(md.extend_seq_lens_cpu_int.tolist(), [2, 1])
        self.assertFalse(backend.graph_mode)

    def test_packed_contract_and_operator_dispatch(self):
        tables = torch.tensor([[4, 1], [7, 2]], dtype=torch.int32)
        generator = torch.Generator().manual_seed(7)
        for env_value in (None, "0", "1"):
            with self.subTest(env=env_value), patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ASCEND_USE_FIA", None)
                if env_value is not None:
                    os.environ["ASCEND_USE_FIA"] = env_value
                backend, atb, fia = make_backend(
                    [3, 5], [2, 1], tables, fia=os.getenv("ASCEND_USE_FIA", "0") == "1"
                )
                pool = FakePool()
                for k, v in zip(pool.keys, pool.values):
                    k.normal_(generator=generator)
                    v.normal_(generator=generator)
                slots = torch.tensor([4 * 128 + 3, 4 * 128 + 4, 7 * 128 + 5])
                batch = forward_batch([3, 5], [2, 1], slots, pool)
                validate_tail_forward_batch(batch)
                self.assertEqual(batch.positions.tolist(), [3, 4, 5])
                md = backend.forward_metadata.sr_tail
                self.assertEqual(md.context_lens_list, [4, 5, 6])
                self.assertEqual(md.block_tables.tolist(), [[4, 1], [4, 1], [7, 2]])
                q = torch.randn(3, 2, 2, generator=generator)
                k = torch.randn(3, 1, 2, generator=generator)
                v = torch.randn(3, 1, 2, generator=generator)
                before = pool.keys[0][4, :3].clone()
                actual = backend.forward_extend(q, k, v, layer_info(), batch)
                expected = paged_reference(
                    q,
                    pool.keys[0],
                    pool.values[0],
                    md.block_tables,
                    [4, 5, 6],
                    1 / math.sqrt(2),
                )
                torch.testing.assert_close(actual, expected.flatten(1))
                torch.testing.assert_close(pool.keys[0][4, :3], before)
                self.assertEqual(pool.writes, 1)
                self.assertEqual(batch.batch_size, 2)
                self.assertEqual(atb.call_count + fia.call_count, 1)
                if env_value == "1":
                    self.assertEqual(
                        fia.call_args.kwargs["actual_seq_lengths_kv"], [4, 5, 6]
                    )
                    self.assertEqual(
                        fia.call_args.args[1].data_ptr(), pool.keys[0].data_ptr()
                    )
                else:
                    self.assertIs(atb.call_args.kwargs["key_cache"], pool.keys[0])

    def test_special_attention_not_dispatched(self):
        backend, _, _ = make_backend([3], [1], torch.tensor([[4]]))
        batch = forward_batch([3], [1], torch.tensor([515]), FakePool())
        for name, value in (
            ("is_cross_attention", True),
            ("sliding_window_size", 4),
            ("attn_type", "encoder"),
            ("logit_cap", 2),
        ):
            layer = layer_info()
            setattr(layer, name, value)
            self.assertFalse(backend._can_run_sr_tail_paged(layer, batch, None, None))
        self.assertFalse(
            backend._can_run_sr_tail_paged(layer_info(), batch, object(), None)
        )
        backend.use_alibi = True
        self.assertFalse(
            backend._can_run_sr_tail_paged(layer_info(), batch, None, None)
        )

    def test_reject_invalid_forward_shapes(self):
        batch = forward_batch([3, 5], [2, 1], torch.arange(3), FakePool())
        for field, value in (
            ("positions", torch.arange(2)),
            ("out_cache_loc", torch.arange(4)),
            ("mrope_positions", torch.zeros(3, 1)),
            ("seq_lens_cpu", torch.tensor([4, 6])),
        ):
            broken = copy.copy(batch)
            setattr(broken, field, value)
            with self.assertRaises(ValueError):
                validate_tail_forward_batch(broken)

    def test_causal_prefix_and_mutation_detection(self):
        keys, values = torch.zeros(1, 8, 1, 2), torch.zeros(1, 8, 1, 2)
        values[0, :, 0, 0] = torch.arange(8.0)
        q = torch.zeros(2, 2, 2)
        tables = torch.zeros(2, 1, dtype=torch.int32)
        correct = paged_reference(q, keys, values, tables, [4, 5], 1.0)
        future = values.clone()
        future[0, 4] += 100
        changed = paged_reference(q, keys, future, tables, [4, 5], 1.0)
        torch.testing.assert_close(correct[0], changed[0])
        self.assertFalse(torch.allclose(correct[1], changed[1]))
        prefix = values.clone()
        prefix[0, 0] += 10
        self.assertFalse(
            torch.allclose(
                correct, paged_reference(q, keys, prefix, tables, [4, 5], 1.0)
            )
        )
        # Deliberately wrong implementations must fail the same numerical assertion.
        for wrong in (
            paged_reference(q, keys, values, tables, [5, 5], 1.0),
            paged_reference(q, keys[:, 3:], values[:, 3:], tables, [1, 2], 1.0),
        ):
            with self.assertRaises(AssertionError):
                torch.testing.assert_close(correct, wrong)

    def test_two_layer_full_sequential_and_packed_equivalence(self):
        gen = torch.Generator().manual_seed(42)
        embed = torch.randn(32, 4, generator=gen) * 0.2
        weights = [
            (
                torch.randn(4, 4, generator=gen) * 0.3,
                torch.randn(4, 2, generator=gen) * 0.3,
                torch.randn(4, 2, generator=gen) * 0.3,
            )
            for _ in range(2)
        ]
        head = torch.randn(4, 32, generator=gen)
        tokens = [torch.arange(9) % 32, torch.arange(5, 14) % 32]
        tables = torch.tensor([[4, 1, 8], [7, 2, 9]], dtype=torch.int32)
        page_size = 4

        def run(pool, prefix, tails, *, reference=False):
            lens = [len(t) for t in tails]
            slots = torch.cat(
                [
                    tables[r, torch.arange(p, p + n) // page_size].long() * page_size
                    + torch.arange(p, p + n) % page_size
                    for r, (p, n) in enumerate(zip(prefix, lens))
                ]
            )
            batch = forward_batch(prefix, lens, slots, pool)
            backend, _, _ = make_backend(prefix, lens, tables, page_size=page_size)
            hidden = embed[torch.cat(tails)]
            for i, (wq, wk, wv) in enumerate(weights):
                q, k, v = hidden @ wq, hidden @ wk, hidden @ wv
                if reference:
                    pool.set_kv_buffer(layer_info(i), slots, k, v)
                    pieces, start = [], 0
                    # Full causal oracle uses a triangular mask, independently of
                    # the production per-query context-length builder.
                    for n in lens:
                        qr = q[start : start + n].reshape(n, 2, 2)
                        kr = k[start : start + n].reshape(n, 1, 2).expand(-1, 2, -1)
                        vr = v[start : start + n].reshape(n, 1, 2).expand(-1, 2, -1)
                        score = torch.einsum("qhd,khd->hqk", qr, kr) / math.sqrt(2)
                        score.masked_fill_(
                            torch.ones(n, n, dtype=torch.bool).triu(1), -torch.inf
                        )
                        pieces.append(
                            torch.einsum("hqk,khd->qhd", score.softmax(-1), vr).flatten(
                                1
                            )
                        )
                        start += n
                    attention = torch.cat(pieces)
                else:
                    attention = backend.forward_extend(q, k, v, layer_info(i), batch)
                hidden = torch.tanh(hidden + attention)
            last = torch.tensor(lens).cumsum(0) - 1
            return hidden[last] @ head

        full_pool = FakePool(page_size)
        expected = run(full_pool, [0, 0], tokens, reference=True)
        packed_pool = FakePool(page_size)
        run(packed_pool, [0, 0], [tokens[0][:6], tokens[1][:8]])
        actual = run(packed_pool, [6, 8], [tokens[0][6:], tokens[1][8:]])
        sequential_pool = FakePool(page_size)
        for j in range(9):
            sequential = run(sequential_pool, [j, j], [t[j : j + 1] for t in tokens])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(sequential, expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(
            actual.topk(3).indices.tolist(), expected.topk(3).indices.tolist()
        )
        for expected_kv, packed_kv, sequential_kv in zip(
            full_pool.keys + full_pool.values,
            packed_pool.keys + packed_pool.values,
            sequential_pool.keys + sequential_pool.values,
        ):
            torch.testing.assert_close(packed_kv, expected_kv, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(sequential_kv, expected_kv, atol=1e-6, rtol=1e-5)


class FakeAllocator:
    def __init__(self, page_size, used):
        self.page_size = page_size
        self.free_pages = [p for p in range(1, 16) if p not in used]

    def available_size(self):
        return len(self.free_pages) * self.page_size

    def backup_state(self):
        return list(self.free_pages)

    def restore_state(self, state):
        self.free_pages = list(state)

    def alloc(self, count):
        result = self.free_pages[:count]
        del self.free_pages[:count]
        return torch.tensor(result)

    def alloc_extend(self, prefix, prefix_cpu, end, end_cpu, last, count):
        slots = []
        for p, e, loc in zip(prefix_cpu.tolist(), end_cpu.tolist(), last.tolist()):
            for pos in range(p, e):
                loc = (
                    self.free_pages.pop(0) * self.page_size
                    if pos % self.page_size == 0
                    else loc + 1
                )
                slots.append(loc)
        return torch.tensor(slots)

    def free(self, slots):
        raise AssertionError(
            "tail rollback must not free slots from a live prefix page"
        )


def request(prefix=3, output=(7, 8), slot=0):
    return NS(
        origin_input_ids=[2] * prefix,
        output_ids=list(output),
        kv_committed_len=prefix,
        kv_allocated_len=prefix,
        req_pool_idx=slot,
        prefix_indices=[],
        cache_protected_len=0,
        multimodal_inputs=None,
        grammar=None,
        rid=str(slot),
        draft_tokens_target=15,
    )


def transaction_fixture(reqs, page_size):
    mapping = torch.full((len(reqs), 512), -1, dtype=torch.int64)
    used = set()
    for r, req in enumerate(reqs):
        pages = [4, 1, 7] if r == 0 else [8, 6, 9]
        for pos in range(req.kv_committed_len):
            page = pages[pos // page_size]
            mapping[r, pos] = page * page_size + pos % page_size
            used.add(page)
    allocator = FakeAllocator(page_size, used)
    scheduler = NS(
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=NS(req_to_token=mapping),
        tree_cache=object(),
        device_module=NS(synchronize=Mock()),
        server_args=NS(speculative_eagle_topk=3),
    )
    plans = [
        tail.plan_tail_extend(r, vocab_size=32, model_is_mrope=False) for r in reqs
    ]
    return scheduler, plans


def _tail_plans_for(reqs):
    plans = []
    for req in reqs:
        plan = tail.plan_tail_extend(req, vocab_size=32, model_is_mrope=False)
        if plan is not None:
            plans.append(plan)
    return plans


def seed_output(rows):
    return NS(
        tree_seed_topk_p=torch.ones(rows, 3) / 3,
        tree_seed_topk_index=torch.tensor([[1, 2, 3]] * rows),
        hidden_states=torch.ones(rows, 4),
    )


class TestTailTransaction(unittest.TestCase):
    def setUp(self):
        self.evict = patch.object(tail, "_evict_tail_capacity")
        self.evict.start()
        self.addCleanup(self.evict.stop)

    def test_page_append_commit_and_rollback(self):
        for page_size, prefix in ((1, 2), (128, 127), (128, 128), (128, 129)):
            for fail in (False, True):
                with self.subTest(page_size=page_size, prefix=prefix, fail=fail):
                    req = request(prefix)
                    scheduler, plans = transaction_fixture([req], page_size)
                    txn = tail.SRTailExtendTransaction(scheduler, plans)
                    mapping = scheduler.req_to_token_pool.req_to_token
                    before, state = mapping.clone(), txn.allocator.backup_state()
                    batch = NS(device="cpu")
                    txn.allocate(batch)
                    self.assertEqual(req.kv_committed_len, prefix)
                    self.assertEqual(batch.out_cache_loc.numel(), 2)
                    torch.testing.assert_close(mapping[0, :prefix], before[0, :prefix])
                    if prefix % page_size:
                        self.assertEqual(
                            batch.out_cache_loc[0], before[0, prefix - 1] + 1
                        )
                    if fail:
                        txn.submitted = True
                        txn.rollback()
                        torch.testing.assert_close(mapping, before)
                        self.assertEqual(txn.allocator.backup_state(), state)
                        scheduler.device_module.synchronize.assert_called_once()
                        self.assertEqual(req.kv_committed_len, prefix)
                    else:
                        out = seed_output(1)
                        txn.commit(out)
                        self.assertEqual(req.kv_committed_len, prefix + 2)
                        self.assertEqual(req.draft_tokens_target, 15)
                        self.assertTrue(tail.tree_seed_is_current(req))
                        self.assertIsNone(req.sr_tree_seed[2])
                        out.tree_seed_topk_p.zero_()
                        self.assertTrue(req.sr_tree_seed[0].any())

    def test_allocation_and_seed_failure_are_atomic(self):
        reqs = [request(slot=0), request(slot=1)]
        scheduler, plans = transaction_fixture(reqs, 128)
        txn = tail.SRTailExtendTransaction(scheduler, plans)
        before = txn.mapping.clone()
        state = txn.allocator.backup_state()
        txn.allocate(NS(device="cpu"))
        with self.assertRaises(RuntimeError):
            txn.commit(seed_output(1))
        self.assertTrue(all(getattr(r, "sr_tree_seed", None) is None for r in reqs))
        reqs[1].sr_prefix_revision = 1
        with self.assertRaises(RuntimeError):
            txn.commit(seed_output(2))
        self.assertTrue(all(getattr(r, "sr_tree_seed", None) is None for r in reqs))
        reqs[1].sr_prefix_revision = 0
        wide = NS(
            tree_seed_topk_p=torch.ones(2, 4),
            tree_seed_topk_index=torch.ones(2, 4, dtype=torch.int64),
            hidden_states=torch.ones(2, 4),
        )
        with self.assertRaises(RuntimeError):
            txn.commit(wide)
        self.assertTrue(all(getattr(r, "sr_tree_seed", None) is None for r in reqs))
        txn.rollback()
        torch.testing.assert_close(txn.mapping, before)
        self.assertEqual(txn.allocator.backup_state(), state)
        req = request(128)
        scheduler, plans = transaction_fixture([req], 128)
        scheduler.token_to_kv_pool_allocator.free_pages.clear()
        txn = tail.SRTailExtendTransaction(scheduler, plans)
        with self.assertRaises(RuntimeError):
            txn.allocate(NS(device="cpu"))
        txn.rollback()
        self.assertEqual(req.kv_committed_len, 128)

    def test_empty_tail_seed_revision_and_recapture(self):
        req = request(3, ())
        scheduler, plans = transaction_fixture([req], 128)
        self.assertTrue(plans[0].recapture)
        txn = tail.SRTailExtendTransaction(scheduler, plans)
        state = txn.allocator.backup_state()
        batch = NS(device="cpu")
        txn.allocate(batch)
        self.assertEqual(batch.out_cache_loc.tolist(), [4 * 128 + 2])
        self.assertEqual(txn.allocator.backup_state(), state)
        txn.commit(seed_output(1))
        self.assertIsNone(
            tail.plan_tail_extend(req, vocab_size=32, model_is_mrope=False)
        )
        old_seed = req.sr_tree_seed
        tail.invalidate_tree_seed(req)
        req.sr_tree_seed = old_seed
        req.sr_tree_seed_boundary = 3
        self.assertFalse(tail.tree_seed_is_current(req))
        req.cache_protected_len = 3
        with self.assertRaises(tail.TailExtendRecoveryRequired):
            tail.plan_tail_extend(req, vocab_size=32, model_is_mrope=False)

    def test_copy_plus_recapture_allocates_from_original(self):
        from dataclasses import replace

        req = request(10, (7, 8))
        plan = tail.plan_tail_extend(
            req, vocab_size=32, model_is_mrope=False, materialized_len=12
        )
        self.assertTrue(plan.recapture)
        self.assertEqual(plan.prefix_len, 11)
        plan = replace(
            plan,
            materialized_len=10,
            original_len=10,
            copy_src_slots=[99, 100],
        )
        self.assertEqual(plan.alloc_start, 10)
        self.assertTrue(plan.needs_suffix_alloc)
        scheduler, _ = transaction_fixture([req], 128)
        txn = tail.SRTailExtendTransaction(scheduler, [plan])
        batch = NS(device="cpu")
        txn.allocate(batch)
        mapping = scheduler.req_to_token_pool.req_to_token
        self.assertTrue(bool((mapping[0, 10:12] >= 0).all()))
        self.assertEqual(int(batch.out_cache_loc.numel()), 1)
        self.assertEqual(int(batch.out_cache_loc[0]), int(mapping[0, 11]))
        txn.commit(seed_output(1))
        self.assertEqual(req.kv_committed_len, 12)

    def test_mrope_stored_slice_and_generated_delta(self):
        stored = torch.tensor([[0, 1, 2, 3], [0, 1, 1, 2], [0, 1, 2, 2]])
        mm = NS(mrope_positions=stored, mrope_position_delta=torch.tensor([-1]))
        actual = tail.tail_mrope_positions(mm, 2, 4)
        expected = torch.cat((stored[:, 2:], torch.tensor([[3, 4]]).expand(3, -1)), 1)
        torch.testing.assert_close(actual, expected)
        mm.mrope_position_delta = None
        with self.assertRaises(tail.TailExtendRecoveryRequired):
            tail.tail_mrope_positions(mm, 4, 2)

    def test_scheduler_one_forward_then_reuse_and_failure(self):
        # Execute the production batch construction and scheduler transaction;
        # only runtime dependencies and the model execution are substituted.
        from sglang.srt.speculative.standalone_remote.sr_align import (
            apply_tree_seed_topk,
        )

        def init_batch(**kwargs):
            return NS(**kwargs, device="cpu", get_model_worker_batch=lambda: built[0])

        build = load_functions(
            Path(tail.__file__),
            ["make_tail_extend_batch"],
            dict(
                torch=torch,
                ScheduleBatch=NS(init_new=init_batch),
                ForwardMode=NS(EXTEND="extend"),
                SamplingBatchInfo=NS(
                    from_schedule_batch=lambda *a: NS(
                        penalizer_orchestrator=NS(is_required=False)
                    )
                ),
                apply_tree_seed_topk=apply_tree_seed_topk,
                _restore_committed_penalties=tail._restore_committed_penalties,
            ),
            strip_imports=True,
        )["make_tail_extend_batch"]
        method = load_functions(
            SCHEDULER,
            ["_sr_execute_tree_tails"],
            dict(
                time=time,
                logger=logging.getLogger(__name__),
                get_sr_round_metrics=get_sr_round_metrics,
                plan_tail_extend=tail.plan_tail_extend,
                TailExtendRecoveryRequired=tail.TailExtendRecoveryRequired,
                tree_seed_is_current=tail.tree_seed_is_current,
                SRTailExtendTransaction=tail.SRTailExtendTransaction,
                _sr_is_device_context_error=lambda exc: False,
                NpuGraphReplaySubmittedError=type("Submitted", (Exception,), {}),
                NpuGraphPreparationError=type("Prep", (Exception,), {}),
            ),
        )["_sr_execute_tree_tails"]
        for fail in (False, True):
            with self.subTest(fail=fail):
                reqs = [request(3, (7, 8), 0), request(5, (9,), 1)]
                scheduler, _ = transaction_fixture(reqs, 128)
                before = scheduler.req_to_token_pool.req_to_token.clone()
                built = []

                def make(plans):
                    batch = build(scheduler, plans)
                    built.append(batch)
                    return batch

                def execute(batch, *, seed_only):
                    self.assertTrue(seed_only)
                    self.assertTrue(batch.is_sr_tail_extend)
                    self.assertEqual(batch.input_ids.tolist(), [7, 8, 9])
                    self.assertEqual(batch.prefix_lens, [3, 5])
                    self.assertEqual(batch.extend_lens, [2, 1])
                    self.assertEqual([r.kv_committed_len for r in reqs], [3, 5])
                    if fail:
                        raise RuntimeError("injected forward failure")
                    return NS(logits_output=seed_output(2))

                scheduler.model_config = NS(vocab_size=32)
                scheduler.server_args = NS(speculative_eagle_topk=3)
                scheduler.spec_algorithm = "STANDALONE_REMOTE"
                scheduler.tp_size = 1
                scheduler._sr_ensure_window_budget = Mock()
                scheduler._sr_is_degraded = lambda rid: False
                scheduler._sr_enable_tree_seed_hidden = Mock()
                scheduler._sr_replay_grammars = Mock()
                scheduler._sr_pause_req = Mock()
                scheduler._sr_mark_degraded = Mock()
                scheduler._sr_make_tail_extend_batch = make
                scheduler.tp_worker = NS(
                    model_runner=NS(model_is_mrope=False, attn_backend=NS()),
                    forward_batch_generation=Mock(side_effect=execute),
                )
                method(scheduler, _tail_plans_for(reqs))
                scheduler.tp_worker.forward_batch_generation.assert_called_once()
                self.assertEqual(scheduler._sr_pause_req.call_count, 2)
                self.assertEqual([r.output_ids for r in reqs], [[7, 8], [9]])
                if fail:
                    torch.testing.assert_close(
                        scheduler.req_to_token_pool.req_to_token, before
                    )
                    self.assertEqual([r.kv_committed_len for r in reqs], [3, 5])
                    self.assertEqual(scheduler._sr_mark_degraded.call_count, 2)
                else:
                    self.assertEqual([r.kv_committed_len for r in reqs], [5, 6])
                    self.assertEqual(_tail_plans_for(reqs), [])
                    method(scheduler, _tail_plans_for(reqs))
                    scheduler.tp_worker.forward_batch_generation.assert_called_once()


class TestTailGraphBuckets(unittest.TestCase):
    def setUp(self):
        self.evict = patch.object(tail, "_evict_tail_capacity")
        self.evict.start()
        self.addCleanup(self.evict.stop)

    def _helpers(self):
        GRAPH = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tail_extend_graph.py"
        )
        tree = ast.parse(GRAPH.read_text(encoding="utf-8"))
        names = {
            "default_tail_token_caps",
            "trim_capture_batch_sizes",
            "select_tail_graph_bucket",
            "real_seed_rows",
            "default_tail_graph_buckets",
            "tail_graph_attn_caps",
            "tail_graph_kv_tokens",
            "tail_token_caps_for_bs",
            "packed_tail_graph_buckets",
            "tail_max_per_request",
            "tail_graph_page_kv_tokens",
            "build_tail_graph_plan",
        }
        body = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "TailGraphPlan":
                body.append(copy.deepcopy(node))
            if isinstance(node, ast.FunctionDef) and node.name in names:
                fn = copy.deepcopy(node)
                fn.body = [
                    n
                    for n in fn.body
                    if not isinstance(n, (ast.Import, ast.ImportFrom))
                ]
                body.append(fn)
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[future] + body, type_ignores=[])
        )
        ns = dict(
            torch=torch,
            dataclass=dataclass,
            Optional=Optional,
            Sequence=Sequence,
            Tuple=Tuple,
            List=List,
            tail_graph_fits_pages=tail_graph_fits_pages,
        )
        exec(compile(module, str(GRAPH), "exec"), ns)
        return ns

    def test_select_bucket_requires_dummy_request_for_token_padding(self):
        helpers = self._helpers()
        buckets = helpers["default_tail_graph_buckets"]([1, 2], [2, 4], (None,))
        select = helpers["select_tail_graph_bucket"]
        self.assertEqual(select(1, 2, 8, buckets), (1, 2, None))
        self.assertEqual(select(1, 3, 8, buckets), (2, 4, None))
        self.assertIsNone(select(1, 3, 8, [(1, 4, None)]))
        self.assertIsNone(select(2, 4, 8, [(4, 4, None)]))
        self.assertIsNone(select(1, 5, 8, buckets))
        self.assertIsNone(select(1, 2, 64, [(1, 2, 32)]))

    def test_packed_caps_cover_concurrent_tails(self):
        helpers = self._helpers()
        buckets = helpers["packed_tail_graph_buckets"]([1, 2], 5, (None,))
        select = helpers["select_tail_graph_bucket"]
        self.assertEqual(select(1, 3, 8, buckets), (2, 4, None))
        self.assertEqual(select(1, 6, 8, buckets), (1, 6, None))
        self.assertEqual(select(2, 6, 8, buckets), (2, 6, None))
        self.assertEqual(select(2, 8, 8, buckets), (2, 8, None))
        self.assertEqual(select(2, 7, 8, buckets), (2, 7, None))
        self.assertEqual(select(2, 7, 8, [(3, 8, None)]), (3, 8, None))
        self.assertIn((2, 8, None), buckets)
        self.assertIn((3, 8, None), buckets)
        self.assertIn((3, 4, None), buckets)

    def test_plan_miss_reasons_pages_and_no_bucket(self):
        helpers = self._helpers()
        plan_fn = helpers["build_tail_graph_plan"]
        graphs = {(1, 6, None): object(), (2, 8, None): object()}
        buckets = [(1, 6, None), (2, 8, None)]
        empty, reason = plan_fn(NS(), {}, buckets, 8, 128)
        self.assertIsNone(empty)
        self.assertEqual(reason, "no_graphs")
        too_long, reason = plan_fn(
            NS(extend_lens=[1], prefix_lens=[128], extend_num_tokens=1),
            graphs,
            buckets,
            captured_pages=1,
            page_size=128,
        )
        self.assertIsNone(too_long)
        self.assertEqual(reason, "pages")
        miss, reason = plan_fn(
            NS(extend_lens=[4, 4], prefix_lens=[3, 3], extend_num_tokens=8),
            {(1, 6, None): object()},
            [(1, 6, None)],
            captured_pages=8,
            page_size=128,
        )
        self.assertIsNone(miss)
        self.assertEqual(reason, "no_bucket")
        hit, reason = plan_fn(
            NS(extend_lens=[4, 4], prefix_lens=[3, 3], extend_num_tokens=8),
            graphs,
            buckets,
            captured_pages=8,
            page_size=128,
        )
        self.assertEqual(reason, "ok")
        self.assertEqual(hit.bucket, (2, 8, None))
        self.assertEqual(hit.raw_bs, 2)
        self.assertEqual(hit.dummy_tokens, 0)

    def test_page_kv_tokens_caps_to_tree_buckets(self):
        helpers = self._helpers()
        runner = NS(
            token_to_kv_pool=NS(size=8192),
            max_total_num_tokens=8192,
            attn_backend=NS(tree_kv_buckets=[256, 512, 1024]),
        )
        self.assertEqual(helpers["tail_graph_page_kv_tokens"](runner), 1024)
        self.assertEqual(
            helpers["tail_graph_page_kv_tokens"](
                NS(
                    token_to_kv_pool=NS(size=8192),
                    max_total_num_tokens=8192,
                    attn_backend=NS(tree_kv_buckets=None),
                )
            ),
            1024,
        )

    def test_seed_rows_are_last_real_tokens(self):
        helpers = self._helpers()
        self.assertEqual(helpers["real_seed_rows"]([2, 1]), [1, 2])
        self.assertEqual(helpers["real_seed_rows"]([1]), [0])
        with self.assertRaises(ValueError):
            helpers["real_seed_rows"]([2, 0])

    def test_dummy_queries_use_slot_zero_and_cannot_see_real_prefix(self):
        tables = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
        real = build_tail_attention_metadata([3, 5], [2, 1], tables)
        padded = pad_tail_attention_metadata(real, 6)
        self.assertEqual(padded.block_tables.shape[0], 6)
        self.assertEqual(padded.context_lens_list[:3], real.context_lens_list)
        self.assertTrue(torch.equal(padded.block_tables[:3], real.block_tables))
        self.assertTrue(torch.equal(padded.block_tables[3:], torch.zeros(3, 2, dtype=torch.int32)))
        self.assertEqual(padded.context_lens_list[3:], [1, 1, 1])
        self.assertNotIn(7, padded.block_tables[3:].reshape(-1).tolist())
        with self.assertRaises(ValueError):
            pad_tail_attention_metadata(real, 6, dummy_slot=3)

    def test_widen_right_pads_zero_pages(self):
        tables = torch.tensor([[7], [9]], dtype=torch.int32)
        wide = widen_tail_block_tables(tables, 4)
        self.assertEqual(tuple(wide.shape), (2, 4))
        self.assertEqual(wide[:, 0].tolist(), [7, 9])
        self.assertEqual(wide[:, 1:].tolist(), [[0, 0, 0], [0, 0, 0]])
        self.assertEqual(tail_graph_max_pages(8192, 128), 64)
        self.assertIs(widen_tail_block_tables(wide, 4), wide)

    def test_copy_keeps_captured_storage(self):
        dst = SRTailAttentionMetadata(
            torch.zeros((4, 4), dtype=torch.int32),
            torch.ones(4, dtype=torch.int32),
            [1, 1, 1, 1],
        )
        src = SRTailAttentionMetadata(
            torch.tensor([[5, 6], [7, 8], [0, 0], [0, 0]], dtype=torch.int32),
            torch.tensor([129, 130, 1, 1], dtype=torch.int32),
            [129, 130, 1, 1],
        )
        tables_id = id(dst.block_tables)
        lens_id = id(dst.context_lens_cpu)
        copy_tail_attention_metadata_(dst, src)
        self.assertEqual(id(dst.block_tables), tables_id)
        self.assertEqual(id(dst.context_lens_cpu), lens_id)
        self.assertEqual(dst.block_tables.tolist(), [[5, 6, 0, 0], [7, 8, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(dst.context_lens_cpu.tolist(), [129, 130, 1, 1])
        self.assertEqual(dst.context_lens_list, [129, 130, 1, 1])

    def test_fill_tail_attention_keeps_captured_storage(self):
        dst = SRTailAttentionMetadata(
            torch.zeros((6, 2), dtype=torch.int32),
            torch.ones(6, dtype=torch.int32),
            [1, 1, 1, 1, 1, 1],
        )
        tables = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
        tables_id = id(dst.block_tables)
        lens_id = id(dst.context_lens_cpu)
        fill_tail_attention_metadata_(dst, [3, 5], [2, 1], tables)
        self.assertEqual(id(dst.block_tables), tables_id)
        self.assertEqual(id(dst.context_lens_cpu), lens_id)
        padded = pad_tail_attention_metadata(
            build_tail_attention_metadata([3, 5], [2, 1], tables), 6
        )
        self.assertEqual(dst.block_tables.tolist(), padded.block_tables.tolist())
        self.assertEqual(dst.context_lens_cpu.tolist(), padded.context_lens_cpu.tolist())
        self.assertEqual(dst.context_lens_list, padded.context_lens_list)
        with self.assertRaises(ValueError):
            fill_tail_attention_metadata_(dst, [3, 5], [2, 1], tables, dummy_slot=3)

    def test_copy_and_plan_reject_too_many_pages(self):
        dst = SRTailAttentionMetadata(
            torch.zeros((2, 1), dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
            [1, 1],
        )
        src = SRTailAttentionMetadata(
            torch.tensor([[5, 6], [7, 8]], dtype=torch.int32),
            torch.tensor([129, 130], dtype=torch.int32),
            [129, 130],
        )
        with self.assertRaises(ValueError):
            copy_tail_attention_metadata_(dst, src)
        self.assertTrue(tail_graph_fits_pages(129, 64, 128))
        self.assertFalse(tail_graph_fits_pages(129, 1, 128))
        helpers = self._helpers()
        self.assertEqual(
            helpers["tail_graph_kv_tokens"](
                NS(token_to_kv_pool=NS(size=8192), max_total_num_tokens=8192)
            ),
            8192,
        )

    def test_npu_tail_runner_uses_wrapped_update(self):
        src = (
            ROOT
            / "python/sglang/srt/hardware_backend/npu/graph_runner/sr_tail_extend_npu_graph_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn("run_npu_graph_update_and_replay", src)
        self.assertNotIn("overlap=True", src)
        self.assertNotIn("threading.Thread", src)
        self.assertIn("context_lens", src)
        self.assertIn("actual_seq_lengths_kv", src)
        helper = (
            ROOT / "python/sglang/srt/speculative/spec_utils.py"
        ).read_text(encoding="utf-8")
        start = helper.index("def run_npu_graph_update_and_replay")
        end = helper.index("\ndef normalize_fia_op_name")
        body = helper[start:end]
        self.assertIn("overlap=False", body)
        serial = body.split("if not overlap:", 1)[1].split("return", 1)[0]
        self.assertNotIn("threading.Thread", serial)
        self.assertLess(serial.index("update_fn()"), serial.index("replay_fn()"))

    def test_wait_copy_event_before_replay_only_when_copy(self):
        class Event:
            def __init__(self):
                self.syncs = 0

            def synchronize(self):
                self.syncs += 1

        event = Event()
        tail.wait_copy_event(event)
        self.assertEqual(event.syncs, 0)
        tail.wait_copy_event(None)
        self.assertEqual(event.syncs, 0)
        req = request(3)
        scheduler, plans = transaction_fixture([req], 128)
        txn = tail.SRTailExtendTransaction(scheduler, plans)
        txn.wait_copy_done()
        txn.copy_done_event = event
        txn.wait_copy_done()
        self.assertEqual(event.syncs, 0)
        ingest_src = SCHEDULER.read_text(encoding="utf-8")
        copy_at = ingest_src.index("transaction.copy_reused_tree_kv()")
        wait_at = ingest_src.index("transaction.wait_copy_done()")
        replay_at = ingest_src.index("tail_runner.replay_filled(plan)")
        eager_at = ingest_src.index("self.tp_worker.forward_batch_generation(")
        fill_at = ingest_src.index("tail_runner.fill(forward_batch, plan)")
        init_at = ingest_src.index("tail_runner.init_forward_batch(worker_batch)")
        worker_batch_at = ingest_src.index("batch.get_model_worker_batch()")
        self.assertLess(copy_at, wait_at)
        self.assertLess(worker_batch_at, wait_at)
        self.assertLess(init_at, wait_at)
        self.assertLess(fill_at, wait_at)
        self.assertLess(wait_at, replay_at)
        self.assertLess(wait_at, eager_at)
        self.assertNotIn("event.synchronize", ingest_src)
        src = tail.wait_copy_event.__doc__ or ""
        body = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tail_extend.py"
        ).read_text(encoding="utf-8")
        start = body.index("def wait_copy_event")
        end = body.index("\ndef get_kv_copy_stream")
        helper = body[start:end]
        self.assertNotIn(".synchronize", helper)
        self.assertNotIn("except Exception", helper)
        self.assertIn("wait_event", helper)
        self.assertIn("def get_kv_copy_stream", body)

    def test_wait_copy_event_uses_stream_wait_event(self):
        class Stream:
            def __init__(self):
                self.seen = []

            def wait_event(self, ev):
                self.seen.append(ev)

        stream = Stream()

        class Mod:
            def current_stream(self, device=None):
                return stream

        event = NS(device=NS(type="npu"))
        with patch.object(torch, "get_device_module", return_value=Mod()):
            tail.wait_copy_event(event)
        self.assertEqual(stream.seen, [event])

    def test_wait_copy_event_propagates_wait_failure(self):
        class Stream:
            def wait_event(self, ev):
                raise RuntimeError("wait failed")

        class Mod:
            def current_stream(self, device=None):
                return Stream()

        event = NS(device=NS(type="npu"))
        with patch.object(torch, "get_device_module", return_value=Mod()):
            with self.assertRaisesRegex(RuntimeError, "wait failed"):
                tail.wait_copy_event(event)

    def test_wait_copy_event_raises_when_wait_event_missing(self):
        class Stream:
            pass

        class Mod:
            def current_stream(self, device=None):
                return Stream()

        event = NS(device=NS(type="npu"))
        with patch.object(torch, "get_device_module", return_value=Mod()):
            with self.assertRaisesRegex(RuntimeError, "wait_event missing"):
                tail.wait_copy_event(event)

    def test_get_kv_copy_stream_reuses_persistent_stream(self):
        class Stream:
            def wait_event(self, ev):
                return None

        created = []

        class Mod:
            def Stream(self):
                stream = Stream()
                created.append(stream)
                return stream

            def stream(self, s):
                return NS()

            def Event(self):
                return NS()

            def current_stream(self, device=None):
                return Stream()

        scheduler = NS()
        with patch.object(torch, "get_device_module", return_value=Mod()):
            first = tail.get_kv_copy_stream(scheduler, NS(type="npu"))
            second = tail.get_kv_copy_stream(scheduler, NS(type="npu"))
        self.assertIs(first, second)
        self.assertEqual(created, [first])
        self.assertIsNone(tail.get_kv_copy_stream(NS(), NS(type="cpu")))

    def test_get_kv_copy_stream_falls_back_before_submit(self):
        class Mod:
            pass

        scheduler = NS()
        with patch.object(torch, "get_device_module", return_value=Mod()):
            self.assertIsNone(tail.get_kv_copy_stream(scheduler, NS(type="npu")))
        self.assertTrue(scheduler._sr_kv_copy_stream_unsupported)

        class Capable:
            def Stream(self):
                raise AssertionError("must not retry Stream after unsupported")

        with patch.object(torch, "get_device_module", return_value=Capable()):
            self.assertIsNone(tail.get_kv_copy_stream(scheduler, NS(type="npu")))

    def test_copy_reused_tree_kv_skips_empty_and_identical_slots(self):
        from dataclasses import replace

        req = request(10, (7, 8))
        plan = tail.plan_tail_extend(
            req, vocab_size=32, model_is_mrope=False, materialized_len=12
        )
        scheduler, _ = transaction_fixture([req], 128)
        copies = []

        def spy_mha(*args, **kwargs):
            copies.append("mha")

        pool = NS(
            kv_buffer=None,
            k_buffer=torch.zeros(2, 32, 1, 2),
            v_buffer=torch.zeros(2, 32, 1, 2),
            move_kv_cache=None,
            _kv_copy_config=None,
        )
        scheduler.tp_worker = NS(model_runner=NS(token_to_kv_pool=pool))
        layout = "sglang.srt.speculative.standalone_remote.sr_verify_layout"

        empty = replace(
            plan, materialized_len=10, original_len=10, copy_src_slots=None
        )
        txn = tail.SRTailExtendTransaction(scheduler, [empty])
        txn.allocate(NS(device="cpu"))
        with patch(layout + ".copy_mha_kv_by_slot", spy_mha):
            txn.copy_reused_tree_kv()
        self.assertEqual(copies, [])
        self.assertIsNone(txn.copy_done_event)

        ident = replace(
            plan, materialized_len=10, original_len=10, copy_src_slots=[99, 100]
        )
        txn = tail.SRTailExtendTransaction(scheduler, [ident])
        txn.allocate(NS(device="cpu"))
        mapping = scheduler.req_to_token_pool.req_to_token
        mapping[0, ident.alloc_start : ident.alloc_start + 2] = torch.tensor([99, 100])
        with patch(layout + ".copy_mha_kv_by_slot", spy_mha):
            txn.copy_reused_tree_kv()
        self.assertEqual(copies, ["mha"])

        real = replace(
            plan, materialized_len=10, original_len=10, copy_src_slots=[99, 100]
        )
        txn = tail.SRTailExtendTransaction(scheduler, [real])
        txn.allocate(NS(device="cpu"))
        with patch(layout + ".copy_mha_kv_by_slot", spy_mha):
            txn.copy_reused_tree_kv()
        self.assertEqual(copies, ["mha", "mha"])
        copy_src = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tail_extend.py"
        ).read_text(encoding="utf-8")
        fn = copy_src[
            copy_src.index("def copy_reused_tree_kv") : copy_src.index(
                "def wait_copy_done"
            )
        ]
        self.assertNotIn(".tolist()", fn)

    def _copy_job_txn(self):
        from dataclasses import replace

        req = request(10, (7, 8))
        plan = replace(
            tail.plan_tail_extend(
                req, vocab_size=32, model_is_mrope=False, materialized_len=12
            ),
            materialized_len=10,
            original_len=10,
            copy_src_slots=[99, 100],
        )
        scheduler, _ = transaction_fixture([req], 128)
        scheduler.tp_worker = NS(
            model_runner=NS(
                token_to_kv_pool=NS(
                    kv_buffer=None,
                    k_buffer=torch.zeros(2, 32, 1, 2),
                    v_buffer=torch.zeros(2, 32, 1, 2),
                    move_kv_cache=None,
                    _kv_copy_config=None,
                )
            )
        )
        txn = tail.SRTailExtendTransaction(scheduler, [plan])
        txn.allocate(NS(device="cpu"))
        return scheduler, txn, plan, req

    def test_copy_reused_tree_kv_side_stream_handshake(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            SRTreeKVLease,
            SRTreeLeaseStore,
        )

        scheduler, txn, plan, req = self._copy_job_txn()
        order = []
        events = []

        class CopyStream:
            def wait_event(self, ev):
                order.append(("copy_wait", ev))

        class Ctx:
            def __enter__(self):
                order.append("ctx_in")
                return self

            def __exit__(self, *exc):
                order.append("ctx_out")
                return False

        copy_stream = CopyStream()
        scheduler._sr_kv_copy_ctx = lambda s: Ctx()
        store = SRTreeLeaseStore()
        lease = SRTreeKVLease(
            rid=req.rid,
            version=1,
            revision=0,
            base_committed_len=10,
            prefix_tokens=(1,),
            page_ids=[1],
            page_slots=torch.arange(4),
            candidate_slots=[0],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
        )
        store.register(lease)
        scheduler.sr_tree_leases = store
        from dataclasses import replace as _replace

        txn.plans = [_replace(txn.plans[0], copy_lease=lease)]

        def fake_record(device, stream=None, *, required=False):
            ev = NS(device=NS(type="npu"), required=required)
            events.append(ev)
            order.append(("record", required))
            return ev

        def spy_mha(*args, **kwargs):
            order.append("copy")

        layout = "sglang.srt.speculative.standalone_remote.sr_verify_layout"
        with patch.object(tail, "get_kv_copy_stream", return_value=copy_stream), patch.object(
            tail, "_is_cpu_device", return_value=False
        ), patch(
            "sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease.record_device_event",
            fake_record,
        ), patch(
            layout + ".copy_mha_kv_by_slot", spy_mha
        ):
            txn.copy_reused_tree_kv()

        self.assertEqual(
            [step if isinstance(step, str) else step[0] for step in order],
            ["record", "copy_wait", "ctx_in", "copy", "record", "ctx_out"],
        )
        self.assertIs(order[1][1], events[0])
        self.assertTrue(events[0].required)
        self.assertTrue(events[1].required)
        self.assertIs(txn.copy_done_event, events[1])
        self.assertIs(lease.pending_free_event, events[1])
        self.assertIsNotNone(txn._copy_hold)
        self.assertEqual(len(txn._copy_hold), 2)
        self.assertTrue(torch.is_tensor(txn._copy_hold[0]))
        self.assertTrue(torch.is_tensor(txn._copy_hold[1]))
        compute = NS(seen=[])

        class ComputeMod:
            def current_stream(self, device=None):
                return NS(wait_event=lambda ev: compute.seen.append(ev))

        with patch.object(torch, "get_device_module", return_value=ComputeMod()):
            txn.wait_copy_done()
        self.assertEqual(compute.seen, [events[1]])
        self.assertIsNone(txn._copy_hold)

    def test_copy_reused_tree_kv_main_stream_records_event(self):
        scheduler, txn, _, _ = self._copy_job_txn()
        order = []

        def fake_record(device, stream=None, *, required=False):
            ev = NS(device=NS(type="npu"), required=required)
            order.append(("record", required))
            return ev

        def spy_mha(*args, **kwargs):
            order.append("copy")

        layout = "sglang.srt.speculative.standalone_remote.sr_verify_layout"
        with patch.object(tail, "get_kv_copy_stream", return_value=None), patch.object(
            tail, "_is_cpu_device", return_value=False
        ), patch(
            "sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease.record_device_event",
            fake_record,
        ), patch(
            layout + ".copy_mha_kv_by_slot", spy_mha
        ):
            txn.copy_reused_tree_kv()
        self.assertEqual(order, ["copy", ("record", True)])
        self.assertIsNotNone(txn.copy_done_event)
        self.assertTrue(txn.copy_done_event.required)

    def test_copy_record_failure_keeps_lease_pages(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            SRTreeKVLease,
            SRTreeLeaseStore,
        )

        scheduler, txn, _, req = self._copy_job_txn()
        store = SRTreeLeaseStore()
        lease = SRTreeKVLease(
            rid=req.rid,
            version=1,
            revision=0,
            base_committed_len=10,
            prefix_tokens=(1,),
            page_ids=[1],
            page_slots=torch.arange(4),
            candidate_slots=[0],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
        )
        store.register(lease)
        scheduler.sr_tree_leases = store
        from dataclasses import replace as _replace

        txn.plans = [_replace(txn.plans[0], copy_lease=lease)]
        records = []

        def fake_record(device, stream=None, *, required=False):
            records.append(required)
            if len(records) == 1:
                return NS(device=NS(type="npu"))
            raise RuntimeError("record failed")

        class CopyStream:
            def wait_event(self, ev):
                return None

        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        scheduler._sr_kv_copy_ctx = lambda s: Ctx()
        layout = "sglang.srt.speculative.standalone_remote.sr_verify_layout"
        with patch.object(tail, "get_kv_copy_stream", return_value=CopyStream()), patch.object(
            tail, "_is_cpu_device", return_value=False
        ), patch(
            "sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease.record_device_event",
            fake_record,
        ), patch(
            layout + ".copy_mha_kv_by_slot", lambda *a, **k: None
        ):
            with self.assertRaisesRegex(RuntimeError, "record failed"):
                txn.copy_reused_tree_kv()
        self.assertIsInstance(lease.pending_free_event, tail._UnfinishedCopyEvent)

        class Alloc:
            def __init__(self):
                self.freed = []

            def free(self, slots):
                self.freed.append(int(slots.numel()))

        alloc = Alloc()
        store.release(lease, allocator=alloc, event=lease.pending_free_event)
        self.assertEqual(alloc.freed, [])
        store.poll_pending_frees(alloc)
        self.assertEqual(alloc.freed, [])

    def _record_fail_copy_txn(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            SRTreeKVLease,
            SRTreeLeaseStore,
        )

        scheduler, txn, _, req = self._copy_job_txn()
        store = SRTreeLeaseStore()
        lease = SRTreeKVLease(
            rid=req.rid,
            version=1,
            revision=0,
            base_committed_len=10,
            prefix_tokens=(1,),
            page_ids=[1],
            page_slots=torch.arange(4),
            candidate_slots=[0],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
        )
        store.register(lease)
        scheduler.sr_tree_leases = store
        from dataclasses import replace as _replace

        txn.plans = [_replace(txn.plans[0], copy_lease=lease)]
        records = []

        def fake_record(device, stream=None, *, required=False):
            records.append(required)
            if len(records) == 1:
                return NS(device=NS(type="npu"))
            raise RuntimeError("record failed")

        class CopyStream:
            def wait_event(self, ev):
                return None

        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        scheduler._sr_kv_copy_ctx = lambda s: Ctx()
        layout = "sglang.srt.speculative.standalone_remote.sr_verify_layout"
        with patch.object(tail, "get_kv_copy_stream", return_value=CopyStream()), patch.object(
            tail, "_is_cpu_device", return_value=False
        ), patch(
            "sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease.record_device_event",
            fake_record,
        ), patch(
            layout + ".copy_kv_pool_by_slot", lambda *a, **k: None
        ):
            with self.assertRaisesRegex(RuntimeError, "record failed"):
                txn.copy_reused_tree_kv()
        return scheduler, txn, lease, store

    def test_rollback_after_record_failure_frees_lease_pages_once(self):
        scheduler, txn, lease, store = self._record_fail_copy_txn()
        self.assertIsInstance(lease.pending_free_event, tail._UnfinishedCopyEvent)
        self.assertTrue(txn.copy_submitted)
        txn.rollback()
        self.assertIsNone(lease.pending_free_event)
        self.assertIsNone(txn._copy_hold)
        scheduler.device_module.synchronize.assert_called()

        class Alloc:
            def __init__(self):
                self.freed = []

            def free(self, slots):
                self.freed.append(int(slots.numel()))

        alloc = Alloc()
        scheduler.token_to_kv_pool_allocator = alloc
        release = load_functions(SCHEDULER, ["_sr_release_held_leases"], {})[
            "_sr_release_held_leases"
        ]
        MethodType(release, scheduler)([lease])
        self.assertEqual(alloc.freed, [4])
        store.poll_pending_frees(alloc)
        self.assertEqual(alloc.freed, [4])

    def test_rollback_sync_failure_keeps_unfinished_lease_pages(self):
        scheduler, txn, lease, store = self._record_fail_copy_txn()
        hold = txn._copy_hold
        scheduler.device_module.synchronize = Mock(
            side_effect=RuntimeError("synchronize failed")
        )
        with self.assertRaisesRegex(RuntimeError, "synchronize failed"):
            txn.rollback()
        self.assertIsInstance(lease.pending_free_event, tail._UnfinishedCopyEvent)
        self.assertIs(txn._copy_hold, hold)

        class Alloc:
            def __init__(self):
                self.freed = []

            def free(self, slots):
                self.freed.append(int(slots.numel()))

        alloc = Alloc()
        store.release(lease, allocator=alloc, event=lease.pending_free_event)
        self.assertEqual(alloc.freed, [])
        store.poll_pending_frees(alloc)
        self.assertEqual(alloc.freed, [])

    def test_wait_copy_done_keeps_hold_when_wait_fails(self):
        scheduler, txn, _, _ = self._copy_job_txn()
        txn._copy_hold = [("src", "dst")]
        txn.copy_done_event = NS(device=NS(type="npu"))

        class Stream:
            def wait_event(self, ev):
                raise RuntimeError("wait failed")

        class Mod:
            def current_stream(self, device=None):
                return Stream()

        with patch.object(torch, "get_device_module", return_value=Mod()):
            with self.assertRaisesRegex(RuntimeError, "wait failed"):
                txn.wait_copy_done()
        self.assertEqual(txn._copy_hold, [("src", "dst")])

    def test_capture_buffers_keep_cpu_seq_lens_and_mrope_shape(self):
        GRAPH = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tail_extend_graph.py"
        )
        tree = ast.parse(GRAPH.read_text(encoding="utf-8"))
        wanted = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "_TailGraphBuffers":
                wanted.append(copy.deepcopy(node))
            if isinstance(node, ast.FunctionDef) and node.name == "make_tail_graph_buffers":
                wanted.append(copy.deepcopy(node))
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[future] + wanted, type_ignores=[])
        )
        ns = {"torch": torch, "Optional": Optional, "dataclass": dataclass}
        exec(compile(module, str(GRAPH), "exec"), ns)
        buffers = ns["make_tail_graph_buffers"](
            bs_cap=2,
            token_cap=6,
            device="cpu",
            vocab=8,
            hidden=4,
            dtype=torch.bfloat16,
            loc_dtype=torch.int32,
        )
        self.assertEqual(str(buffers.seq_lens_cpu.device), "cpu")
        self.assertEqual(tuple(buffers.mrope_positions.shape), (3, 6))
        self.assertEqual(tuple(buffers.input_ids.shape), (6,))
        self.assertEqual(str(buffers.input_ids.device), "cpu")
        self.assertEqual(buffers.next_token_logits.dtype, torch.float32)
        self.assertIsNone(buffers.hidden_states)
        self.assertEqual(str(buffers.extend_lens_cpu.device), "cpu")
        self.assertEqual(tuple(buffers.req_page_tables.shape), (2, 1))
        wide = ns["make_tail_graph_buffers"](
            bs_cap=2,
            token_cap=6,
            device="cpu",
            vocab=8,
            hidden=4,
            dtype=torch.bfloat16,
            loc_dtype=torch.int32,
            pages=8,
        )
        self.assertEqual(tuple(wide.req_page_tables.shape), (2, 8))

    def test_non_fia_buckets_do_not_multiply_tree_kv_caps(self):
        helpers = self._helpers()
        atb = NS(use_fia=False, tree_kv_buckets=[256, 512, 1024])
        fia = NS(use_fia=True, tree_kv_buckets=[256, 512, 1024])
        self.assertEqual(helpers["tail_graph_attn_caps"](atb), (None,))
        self.assertEqual(helpers["tail_graph_attn_caps"](None), (None,))
        self.assertEqual(helpers["tail_graph_attn_caps"](fia), [1024, 512, 256])
        buckets = helpers["default_tail_graph_buckets"](
            [1, 2], [1, 2, 4, 5, 6], helpers["tail_graph_attn_caps"](atb)
        )
        self.assertTrue(all(cap is None for _, _, cap in buckets))
        self.assertEqual(len(buckets), 10)

    def test_npu_atb_update_uses_cpu_context_lens(self):
        npu_runner = (
            ROOT
            / "python/sglang/srt/hardware_backend/npu/graph_runner/sr_tail_extend_npu_graph_runner.py"
        )
        helpers = load_functions(
            npu_runner,
            [
                "make_tail_graph_cpu_update_payload",
                "fill_tail_graph_cpu_update_payload",
                "tail_graph_cpu_update_payload",
            ],
            dict(torch=torch),
            strip_imports=True,
        )
        payload_fn = helpers["tail_graph_cpu_update_payload"]
        fia = payload_fn([3, 4, 1], use_fia=True)
        self.assertEqual(fia, [{"actual_seq_lengths_kv": [3, 4, 1]}])
        atb = payload_fn([3, 4, 1], use_fia=False)
        self.assertEqual(list(atb[0]), ["context_lens"])
        self.assertEqual(str(atb[0]["context_lens"].device), "cpu")
        self.assertEqual(atb[0]["context_lens"].tolist(), [3, 4, 1])
        captured = helpers["make_tail_graph_cpu_update_payload"](6, use_fia=False)
        lens = captured[0]["context_lens"]
        helpers["fill_tail_graph_cpu_update_payload"](
            captured, [3, 4, 1, 1, 1, 1], use_fia=False
        )
        self.assertIs(captured[0]["context_lens"], lens)
        self.assertEqual(lens.tolist(), [3, 4, 1, 1, 1, 1])
        helpers["fill_tail_graph_cpu_update_payload"](
            captured, [9, 8, 7, 1, 1, 1], use_fia=False
        )
        self.assertIs(captured[0]["context_lens"], lens)
        self.assertEqual(lens.tolist(), [9, 8, 7, 1, 1, 1])
        fia_payload = helpers["make_tail_graph_cpu_update_payload"](3, use_fia=True)
        fia_list = fia_payload[0]["actual_seq_lengths_kv"]
        helpers["fill_tail_graph_cpu_update_payload"](
            fia_payload, [9, 8, 1], use_fia=True
        )
        self.assertIs(fia_payload[0]["actual_seq_lengths_kv"], fia_list)
        self.assertEqual(fia_list, [9, 8, 1])

    def test_skip_bucket_log_includes_exception_type(self):
        src = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tail_extend_graph.py"
        ).read_text(encoding="utf-8")
        self.assertIn("type(e).__name__", src)
        self.assertIn("skip tail graph bucket %s: %s: %s", src)
        self.assertIn("copy_tail_attention_metadata_", src)
        self.assertIn("fill_tail_attention_metadata_", src)
        self.assertIn("widen_tail_block_tables", src)
        self.assertIn("tail_graph_fits_pages", src)
        self.assertIn("packed_tail_graph_buckets", src)

    def test_scheduler_graph_replay_then_submitted_error_skips_rollback(self):
        submitted = type("Submitted", (Exception,), {})
        method = load_functions(
            SCHEDULER,
            ["_sr_execute_tree_tails"],
            dict(
                time=time,
                logger=logging.getLogger(__name__),
                get_sr_round_metrics=get_sr_round_metrics,
                plan_tail_extend=tail.plan_tail_extend,
                TailExtendRecoveryRequired=tail.TailExtendRecoveryRequired,
                tree_seed_is_current=tail.tree_seed_is_current,
                SRTailExtendTransaction=tail.SRTailExtendTransaction,
                _sr_is_device_context_error=lambda exc: False,
                NpuGraphReplaySubmittedError=submitted,
                NpuGraphPreparationError=type("Prep", (Exception,), {}),
            ),
        )["_sr_execute_tree_tails"]
        reqs = [request(3, (7, 8), 0)]
        scheduler, _ = transaction_fixture(reqs, 128)
        batch = NS(
            is_sr_tail_extend=True,
            device="cpu",
            extend_lens=[2],
            prefix_lens=[3],
            extend_num_tokens=2,
            get_model_worker_batch=lambda: NS(),
        )
        plan = NS(bucket=(1, 2, None))
        runner = NS(
            plan=lambda _batch: plan,
            plan_with_reason=lambda _batch: (plan, "ok"),
            init_forward_batch=Mock(return_value=NS(sampling_info=NS())),
            model_runner=NS(
                capture_tree_seed_only=Mock(),
            ),
            fill=Mock(),
            replay_filled=Mock(side_effect=submitted("graph submitted")),
            eager_fallback_count=0,
        )
        scheduler.model_config = NS(vocab_size=32)
        scheduler.server_args = NS(speculative_eagle_topk=3)
        scheduler.spec_algorithm = "STANDALONE_REMOTE"
        scheduler.tp_size = 1
        scheduler._sr_ensure_window_budget = Mock()
        scheduler._sr_is_degraded = lambda rid: False
        scheduler._sr_enable_tree_seed_hidden = Mock()
        scheduler._sr_replay_grammars = Mock()
        scheduler._sr_pause_req = Mock()
        scheduler._sr_mark_degraded = Mock()
        scheduler._sr_make_tail_extend_batch = lambda plans: batch
        scheduler.sr_tree_drafter = NS(tail_graph_runner=runner)
        scheduler.tp_worker = NS(
            model_runner=NS(model_is_mrope=False, attn_backend=NS()),
            forward_batch_generation=Mock(side_effect=AssertionError("eager")),
        )
        with self.assertRaises(submitted):
            method(scheduler, _tail_plans_for(reqs))
        scheduler.tp_worker.forward_batch_generation.assert_not_called()
        scheduler._sr_mark_degraded.assert_not_called()
        self.assertEqual(reqs[0].kv_committed_len, 3)
        scheduler._sr_pause_req.assert_not_called()

    def test_scheduler_uses_graph_logits_and_skips_eager(self):
        method = load_functions(
            SCHEDULER,
            ["_sr_execute_tree_tails"],
            dict(
                time=time,
                logger=logging.getLogger(__name__),
                get_sr_round_metrics=get_sr_round_metrics,
                plan_tail_extend=tail.plan_tail_extend,
                TailExtendRecoveryRequired=tail.TailExtendRecoveryRequired,
                tree_seed_is_current=tail.tree_seed_is_current,
                SRTailExtendTransaction=tail.SRTailExtendTransaction,
                _sr_is_device_context_error=lambda exc: False,
                NpuGraphReplaySubmittedError=type("Submitted", (Exception,), {}),
                NpuGraphPreparationError=type("Prep", (Exception,), {}),
            ),
        )["_sr_execute_tree_tails"]
        reqs = [request(3, (7, 8), 0)]
        scheduler, _ = transaction_fixture(reqs, 128)
        seed = seed_output(1)
        batch = NS(
            is_sr_tail_extend=True,
            device="cpu",
            extend_lens=[2],
            prefix_lens=[3],
            extend_num_tokens=2,
            get_model_worker_batch=lambda: NS(),
        )
        plan = NS(bucket=(1, 2, None))
        runner = NS(
            plan=lambda _batch: plan,
            plan_with_reason=lambda _batch: (plan, "ok"),
            init_forward_batch=Mock(return_value=NS(sampling_info=NS())),
            model_runner=NS(capture_tree_seed_only=Mock()),
            fill=Mock(),
            replay_filled=Mock(return_value=seed),
            eager_fallback_count=0,
        )
        scheduler.model_config = NS(vocab_size=32)
        scheduler.server_args = NS(speculative_eagle_topk=3)
        scheduler.spec_algorithm = "STANDALONE_REMOTE"
        scheduler.tp_size = 1
        scheduler._sr_ensure_window_budget = Mock()
        scheduler._sr_is_degraded = lambda rid: False
        scheduler._sr_enable_tree_seed_hidden = Mock()
        scheduler._sr_replay_grammars = Mock()
        scheduler._sr_pause_req = Mock()
        scheduler._sr_mark_degraded = Mock()
        scheduler._sr_make_tail_extend_batch = lambda plans: batch
        scheduler.sr_tree_drafter = NS(tail_graph_runner=runner)
        scheduler.tp_worker = NS(
            model_runner=NS(model_is_mrope=False, attn_backend=NS()),
            forward_batch_generation=Mock(side_effect=AssertionError("eager")),
        )
        method(scheduler, _tail_plans_for(reqs))
        runner.replay_filled.assert_called_once()
        runner.model_runner.capture_tree_seed_only.assert_called_once()
        scheduler.tp_worker.forward_batch_generation.assert_not_called()
        self.assertEqual(reqs[0].kv_committed_len, 5)
        self.assertTrue(tail.tree_seed_is_current(reqs[0]))
        self.assertEqual(runner.eager_fallback_count, 0)
        metrics = get_sr_round_metrics(scheduler, "Draft")
        self.assertEqual(metrics.paths["tail_extend_graph"], 1)
        self.assertEqual(metrics.paths["ordinary_extend"], 0)

    def test_scheduler_records_plan_miss_and_falls_back(self):
        method = load_functions(
            SCHEDULER,
            ["_sr_execute_tree_tails"],
            dict(
                time=time,
                logger=logging.getLogger(__name__),
                get_sr_round_metrics=get_sr_round_metrics,
                plan_tail_extend=tail.plan_tail_extend,
                TailExtendRecoveryRequired=tail.TailExtendRecoveryRequired,
                tree_seed_is_current=tail.tree_seed_is_current,
                SRTailExtendTransaction=tail.SRTailExtendTransaction,
                _sr_is_device_context_error=lambda exc: False,
                NpuGraphReplaySubmittedError=type("Submitted", (Exception,), {}),
                NpuGraphPreparationError=type("Prep", (Exception,), {}),
            ),
        )["_sr_execute_tree_tails"]
        reqs = [request(3, (7, 8), 0)]
        scheduler, _ = transaction_fixture(reqs, 128)
        seed = seed_output(1)
        batch = NS(
            is_sr_tail_extend=True,
            device="cpu",
            extend_lens=[2],
            prefix_lens=[3],
            extend_num_tokens=2,
            get_model_worker_batch=lambda: NS(),
        )
        runner = NS(
            plan=lambda _batch: None,
            plan_with_reason=lambda _batch: (None, "no_bucket"),
            init_forward_batch=Mock(side_effect=AssertionError("graph")),
            fill=Mock(side_effect=AssertionError("graph")),
            replay_filled=Mock(side_effect=AssertionError("graph")),
            eager_fallback_count=0,
        )
        scheduler.model_config = NS(vocab_size=32)
        scheduler.server_args = NS(speculative_eagle_topk=3)
        scheduler.spec_algorithm = "STANDALONE_REMOTE"
        scheduler.tp_size = 1
        scheduler._sr_ensure_window_budget = Mock()
        scheduler._sr_is_degraded = lambda rid: False
        scheduler._sr_enable_tree_seed_hidden = Mock()
        scheduler._sr_replay_grammars = Mock()
        scheduler._sr_pause_req = Mock()
        scheduler._sr_mark_degraded = Mock()
        scheduler._sr_make_tail_extend_batch = lambda plans: batch
        scheduler.sr_tree_drafter = NS(tail_graph_runner=runner)
        scheduler.tp_worker = NS(
            model_runner=NS(model_is_mrope=False, attn_backend=NS()),
            forward_batch_generation=Mock(return_value=NS(logits_output=seed)),
        )
        method(scheduler, _tail_plans_for(reqs))
        scheduler.tp_worker.forward_batch_generation.assert_called_once()
        self.assertEqual(runner.eager_fallback_count, 1)
        metrics = get_sr_round_metrics(scheduler, "Draft")
        self.assertEqual(metrics.counts["tail_graph_miss_no_bucket"], 1)
        self.assertEqual(metrics.paths["ordinary_extend"], 1)
        self.assertEqual(metrics.paths["tail_extend_graph"], 0)
        self.assertEqual(reqs[0].kv_committed_len, 5)


class TestSeedOnlyForward(unittest.TestCase):
    def test_last_logits_bias_custom_temperature_and_no_sampling(self):
        srt = ROOT / "python/sglang/srt"
        prune = load_functions(
            srt / "layers/logits_processor.py",
            ["_get_pruned_states"],
            dict(torch=torch),
        )["_get_pruned_states"]
        mode = NS(
            is_decode_or_idle=lambda: False,
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
            is_extend=lambda: True,
        )
        metadata = NS(
            forward_mode=mode,
            extend_return_logprob=False,
            padded_static_len=-1,
            extend_seq_lens=torch.tensor([2, 1]),
        )
        hidden = torch.arange(12.0).reshape(3, 4)
        pruned = prune(NS(), hidden, None, None, metadata)[0]
        torch.testing.assert_close(pruned, hidden[[1, 2]])

        def custom(logits, info):
            logits[:, 1] += 2

        sampler_methods = load_functions(
            srt / "layers/sampler.py",
            ["_preprocess_logits", "_capture_tree_seed", "capture_tree_seed_only"],
            dict(torch=torch, apply_custom_logit_processor=custom),
        )
        sampler = type("CPUSampler", (), sampler_methods)()
        sampler.use_nan_detection = False
        runner_methods = load_functions(
            srt / "model_executor/model_runner.py",
            ["_preprocess_logits", "capture_tree_seed_only"],
            {},
        )
        runner = type("CPURunner", (), runner_methods)()
        runner.sampler = sampler
        runner.sample = Mock(side_effect=AssertionError("seed-only must not sample"))

        def bias(logits):
            logits[:, 0] += 3
            logits[:, 3] = -torch.inf  # stand-in grammar/logits bias

        info = NS(
            update_regex_vocab_mask=Mock(),
            apply_logits_bias=bias,
            has_custom_logit_processor=True,
            is_all_greedy=False,
            tree_seed_topk=3,
            temperatures=torch.tensor([[0.5], [2.0]]),
            vocab_mask=None,
        )
        output = NS(next_token_logits=pruned.clone())
        runner.forward = Mock(
            return_value=NS(
                logits_output=output,
                can_run_graph=False,
                expert_distribution_metrics=None,
            )
        )
        forward = load_functions(
            srt / "managers/tp_worker.py",
            ["forward_batch_generation"],
            dict(GenerationBatchResult=NS),
            class_name="TpModelWorker",
        )["forward_batch_generation"]
        worker = NS(
            model_runner=runner, is_dllm=lambda: False, pp_group=NS(is_last_rank=True)
        )
        rng = torch.random.get_rng_state().clone()
        result = forward(
            worker, None, forward_batch=NS(sampling_info=info), seed_only=True
        )
        expected = pruned.clone()
        bias(expected)
        custom(expected, info)
        vals, indices = (expected / info.temperatures).softmax(-1).topk(3)
        torch.testing.assert_close(result.logits_output.tree_seed_topk_p, vals)
        torch.testing.assert_close(result.logits_output.tree_seed_topk_index, indices)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        runner.sample.assert_not_called()
        info.update_regex_vocab_mask.assert_called_once()
        self.assertFalse(hasattr(result, "next_token_ids"))


class TestRoundMetrics(unittest.TestCase):
    def test_target_round_preserves_rpc_and_fallback_order(self):
        target = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/verifier/sr_target_scheduler_mixin.py"
        )
        run = load_functions(
            target,
            ["_sr_run_verify_round"],
            dict(get_sr_round_metrics=get_sr_round_metrics),
        )["_sr_run_verify_round"]
        for has_draft in (False, True):
            with self.subTest(has_draft=has_draft):
                trace = []
                req = NS(
                    cur_drafts={"tree": True} if has_draft else None,
                    finished=lambda: False,
                )
                batch = NS()
                owner = NS(
                    server_args=NS(speculative_num_draft_tokens=15),
                    _sr_select_reqs=lambda batch: [req],
                    rpc_next_draft=lambda reqs: trace.append("rpc") or {},
                    _sr_maybe_align_chain_replies=lambda *a, **k: None,
                    _sr_attach_replies=lambda *a: 0,
                    run_batch=lambda batch: trace.append(
                        ("forward", batch.draft_num_tokens)
                    ),
                    process_batch_result=lambda *a: trace.append("process"),
                )
                run(owner, batch)
                self.assertEqual(
                    trace,
                    (
                        [("forward", 15), "process", "rpc"]
                        if has_draft
                        else ["rpc", ("forward", 1), "process", "rpc"]
                    ),
                )
                self.assertIsNone(batch.sr_round_metrics)

    def test_device_samples_are_polled_without_synchronize(self):
        events = []

        class Event:
            def __init__(self, **kwargs):
                self.ready = False
                events.append(self)

            def record(self):
                pass

            def query(self):
                return self.ready

            def elapsed_time(self, other):
                return 2.5

        metrics = SRRoundMetrics("test", NS(Event=Event))
        with metrics.round(), metrics.phase("forward", device=True):
            pass
        metrics.poll()
        self.assertEqual(len(metrics.pending), 1)
        events[-1].ready = True
        metrics.poll()
        self.assertEqual(metrics.device_ms["forward"], 2.5)
        self.assertEqual(len(metrics.pending), 0)


@dataclass(frozen=True)
class _FakeTailPlan:
    req: object
    copy_src_slots: object = None
    copy_lease: object = None
    length: int = 1
    recapture: bool = False


@dataclass
class _SRTreePlanDraft:
    req: object
    kind: str
    plan: object = None
    lease: object = None
    miss: object = None


class TestTreeIngestLifecycle(unittest.TestCase):
    def setUp(self):
        self.evict = patch.object(tail, "_evict_tail_capacity")
        self.evict.start()
        self.addCleanup(self.evict.stop)

    def _lease(self, rid, version=1):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            SRTreeKVLease,
        )

        return SRTreeKVLease(
            rid=rid,
            version=version,
            revision=0,
            base_committed_len=1,
            prefix_tokens=(1,),
            page_ids=[1],
            page_slots=torch.arange(2),
            candidate_slots=[0],
            parent_list=[],
            top_scores_index=[],
            draft_tokens=[],
        )

    def _bind(self, scheduler, *, plan_tail_extend=None, execute=None):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            live_accept_prefix,
            validate_lease_commit,
        )
        from sglang.srt.speculative.standalone_remote.sr_align import (
            draft_needed_max_new_tokens,
        )

        fns = load_functions(
            SCHEDULER,
            [
                "_sr_tree_req_alive",
                "_sr_prepare_tree_reqs",
                "_sr_inspect_lease_copy",
                "_sr_inspect_tree_plans",
                "_sr_record_inspect_misses",
                "_sr_acquire_tree_plans",
                "_sr_release_unused_leases",
                "_sr_release_held_leases",
                "_sr_record_committed_reuse",
                "_sr_run_tree_ingest",
                "_sr_mark_degraded",
                "_sr_is_degraded",
                "_sr_ensure_window_budget",
                "_sr_release_tree_lease",
                "_sr_execute_tree_tails",
            ],
            dict(
                time=time,
                logger=logging.getLogger(__name__),
                get_sr_round_metrics=get_sr_round_metrics,
                plan_tail_extend=plan_tail_extend or tail.plan_tail_extend,
                TailExtendRecoveryRequired=tail.TailExtendRecoveryRequired,
                tree_seed_is_current=tail.tree_seed_is_current,
                SRTailExtendTransaction=tail.SRTailExtendTransaction,
                _sr_is_device_context_error=lambda exc: False,
                NpuGraphReplaySubmittedError=type("Submitted", (Exception,), {}),
                NpuGraphPreparationError=type("Prep", (Exception,), {}),
                _SRTreePlanDraft=_SRTreePlanDraft,
                replace=replace,
                draft_needed_max_new_tokens=draft_needed_max_new_tokens,
                validate_lease_commit=validate_lease_commit,
                live_accept_prefix=live_accept_prefix,
            ),
            class_name="StandaloneRemoteDraftSchedulerMixin",
        )
        for name, fn in fns.items():
            setattr(scheduler, name, MethodType(fn, scheduler))
        if execute is not None:
            scheduler._sr_execute_tree_tails = execute
        scheduler._sr_tree_req_alive = lambda req: True
        if not hasattr(scheduler, "model_config"):
            scheduler.model_config = NS(vocab_size=32)
        if not hasattr(scheduler, "tp_worker"):
            scheduler.tp_worker = NS(model_runner=NS(model_is_mrope=False))
        if not hasattr(scheduler, "token_to_kv_pool_allocator"):
            scheduler.token_to_kv_pool_allocator = NS(free=lambda slots: None)
        elif not callable(getattr(scheduler.token_to_kv_pool_allocator, "free", None)):
            scheduler.token_to_kv_pool_allocator.free = lambda slots: None
        return scheduler

    def test_expand_and_execute_are_unidirectional(self):
        src = SCHEDULER.read_text(encoding="utf-8")
        execute = src[
            src.index("def _sr_execute_tree_tails") : src.index(
                "def _sr_ingest_committed_batch"
            )
        ]
        self.assertNotIn("_sr_run_tree_ingest", execute)
        self.assertNotIn("plan_tail_extend", execute)
        expand = src[
            src.index("def _sr_tree_expand_batch") : src.index("def _sr_reprefill(")
        ]
        self.assertEqual(expand.count("_sr_run_tree_ingest"), 1)
        self.assertNotIn("_sr_ingest_committed_batch", expand)
        inspect = src[
            src.index("def _sr_inspect_tree_plans") : src.index(
                "def _sr_record_inspect_misses"
            )
        ]
        self.assertNotIn("pin_lease", inspect)
        self.assertNotIn("release_rid", inspect)
        self.assertNotIn("_sr_ensure_window_budget", inspect)
        self.assertNotIn("_sr_mark_degraded", inspect)
        prepare = src[
            src.index("def _sr_prepare_tree_reqs") : src.index(
                "def _sr_inspect_lease_copy"
            )
        ]
        self.assertIn("_sr_ensure_window_budget", prepare)
        acquire = src[
            src.index("def _sr_acquire_tree_plans") : src.index(
                "def _sr_release_unused_leases"
            )
        ]
        self.assertIn("held.append", acquire)
        self.assertLess(acquire.index("pin_lease"), acquire.index("held.append"))
        ingest = src[
            src.index("def _sr_ingest_committed_batch") : src.index(
                "def _sr_ingest_committed_chain_batch"
            )
        ]
        self.assertIn("_sr_run_tree_ingest", ingest)
        copy_inspect = src[
            src.index("def _sr_inspect_lease_copy") : src.index(
                "def _sr_inspect_tree_plans"
            )
        ]
        self.assertNotIn("pin_lease", copy_inspect)
        run = src[
            src.index("def _sr_run_tree_ingest") : src.index("def _sr_ingest_tree_tails")
        ]
        self.assertLess(
            run.index("_sr_release_unused_leases"), run.index("_sr_execute_tree_tails")
        )

    def test_mixed_hit_and_ordinary_one_execute(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            SRTreeLeaseStore,
        )

        store = SRTreeLeaseStore()
        lease = self._lease("a")
        store.register(lease)
        a = NS(rid="a")
        b = NS(rid="b")
        plan_a = _FakeTailPlan(req=a, copy_src_slots=[7], copy_lease=lease, length=1)
        plan_b = _FakeTailPlan(req=b, length=2)
        calls = []
        scheduler = self._bind(NS(sr_tree_leases=store), execute=lambda plans: calls.append(list(plans)) or True)
        scheduler._sr_prepare_tree_reqs = lambda reqs: reqs
        scheduler._sr_inspect_tree_plans = lambda reqs: [
            _SRTreePlanDraft(a, "copy", plan=plan_a, lease=lease),
            _SRTreePlanDraft(b, "ordinary", plan=plan_b),
        ]
        self.assertTrue(scheduler._sr_run_tree_ingest([a, b]))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 2)
        self.assertTrue(lease.released)
        self.assertEqual(store.counts["tree_kv_commit_hit"], 1)
        self.assertEqual(store.counts["tree_kv_reused_tokens"], 1)

    def test_valid_empty_tail_stays_out_of_batch(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            SRTreeLeaseStore,
        )

        store = SRTreeLeaseStore()
        seed = NS(rid="s")
        calls = []
        scheduler = self._bind(NS(sr_tree_leases=store), execute=lambda plans: calls.append(plans) or True)
        scheduler._sr_prepare_tree_reqs = lambda reqs: reqs
        scheduler._sr_inspect_tree_plans = lambda reqs: [_SRTreePlanDraft(seed, "skip")]
        self.assertTrue(scheduler._sr_run_tree_ingest([seed]))
        self.assertEqual(calls, [])

    def test_second_pin_failure_releases_first_held_lease(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            SRTreeLeaseStore,
        )

        store = SRTreeLeaseStore()
        first = self._lease("a", 1)
        second = self._lease("b", 1)
        store.register(first)
        store.register(second)
        a = NS(rid="a")
        b = NS(rid="b")
        plan_a = _FakeTailPlan(req=a, copy_src_slots=[1], length=1)
        plan_b = _FakeTailPlan(req=b, copy_src_slots=[2], length=1)
        n = {"n": 0}
        real_pin = store.pin_lease
        held_during_fail = []

        def pin_once(lease):
            n["n"] += 1
            if n["n"] > 1:
                held_during_fail.append((first.in_use, first.released, store.get("a") is first))
                return None
            return real_pin(lease)

        store.pin_lease = pin_once
        calls = []

        def no_recovery(*_a, **_k):
            raise tail.TailExtendRecoveryRequired("need recovery")

        scheduler = self._bind(
            NS(sr_tree_leases=store),
            plan_tail_extend=no_recovery,
            execute=lambda plans: calls.append(list(plans)) or True,
        )
        scheduler._sr_prepare_tree_reqs = lambda reqs: reqs
        scheduler._sr_inspect_tree_plans = lambda reqs: [
            _SRTreePlanDraft(a, "copy", plan=plan_a, lease=first),
            _SRTreePlanDraft(b, "copy", plan=plan_b, lease=second),
        ]
        scheduler._sr_mark_degraded = Mock()
        scheduler._sr_run_tree_ingest([a, b])
        self.assertEqual(held_during_fail, [(True, False, True)])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0].req is a, True)
        self.assertTrue(first.released)
        self.assertFalse(first.in_use)
        replacement = self._lease("a", 2)
        store.register(replacement)
        store.release(first)
        self.assertIs(store.get("a"), replacement)
        self.assertFalse(replacement.released)

    def test_mid_pin_exception_releases_already_held_lease(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            SRTreeLeaseStore,
        )

        store = SRTreeLeaseStore()
        first = self._lease("a", 1)
        second = self._lease("b", 1)
        store.register(first)
        store.register(second)
        a = NS(rid="a")
        b = NS(rid="b")
        plan_a = _FakeTailPlan(req=a, copy_src_slots=[1], length=1)
        plan_b = _FakeTailPlan(req=b, copy_src_slots=[2], length=1)
        n = {"n": 0}
        real_pin = store.pin_lease

        def pin_then_raise(lease):
            n["n"] += 1
            if n["n"] > 1:
                raise RuntimeError("pin boom")
            return real_pin(lease)

        store.pin_lease = pin_then_raise
        calls = []
        scheduler = self._bind(
            NS(sr_tree_leases=store),
            execute=lambda plans: calls.append(list(plans)) or True,
        )
        scheduler._sr_prepare_tree_reqs = lambda reqs: reqs
        scheduler._sr_inspect_tree_plans = lambda reqs: [
            _SRTreePlanDraft(a, "copy", plan=plan_a, lease=first),
            _SRTreePlanDraft(b, "copy", plan=plan_b, lease=second),
        ]
        with self.assertRaisesRegex(RuntimeError, "pin boom"):
            scheduler._sr_run_tree_ingest([a, b])
        self.assertEqual(calls, [])
        self.assertTrue(first.released)
        self.assertFalse(first.in_use)
        self.assertIsNone(store.get("a"))
        self.assertIs(store.get("b"), second)

    def test_unused_lease_released_before_execute_and_no_hit_on_rollback(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            SRTreeLeaseStore,
        )

        store = SRTreeLeaseStore()
        lease = self._lease("a")
        store.register(lease)
        a = NS(rid="a")
        b = NS(rid="b")
        plan_b = _FakeTailPlan(req=b, length=1)
        seen = []
        freed = []
        allocator = NS(free=lambda slots: freed.append(slots))

        def execute(plans):
            seen.append((lease.released, store.get("a") is None, list(freed)))
            return False

        scheduler = self._bind(
            NS(sr_tree_leases=store, token_to_kv_pool_allocator=allocator),
            execute=execute,
        )
        scheduler._sr_prepare_tree_reqs = lambda reqs: reqs
        scheduler._sr_inspect_tree_plans = lambda reqs: [
            _SRTreePlanDraft(a, "ordinary", lease=lease, miss="version"),
            _SRTreePlanDraft(b, "ordinary", plan=plan_b),
        ]
        self.assertFalse(scheduler._sr_run_tree_ingest([a, b]))
        self.assertEqual(seen[0][0], True)
        self.assertTrue(seen[0][1])
        self.assertEqual(len(seen[0][2]), 1)
        self.assertEqual(store.counts["tree_kv_commit_hit"], 0)
        self.assertIsNone(store.get("a"))

    def test_budget_degrade_releases_lease_before_inspect(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_draft_state import (
            SRDraftState,
            SRDraftStateManager,
        )
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            SRTreeLeaseStore,
        )

        store = SRTreeLeaseStore()
        lease = self._lease("a")
        store.register(lease)
        req = NS(
            rid="a",
            origin_input_ids=[1, 2, 3],
            output_ids=[],
            draft_tokens_target=8,
            sampling_params=NS(max_new_tokens=1),
        )
        sr_state = SRDraftStateManager()
        sr_state.set("a", SRDraftState(req_id="a", session_id="s", req_object=req))
        scheduler = self._bind(
            NS(
                sr_tree_leases=store,
                sr_state=sr_state,
                server_args=NS(speculative_num_steps=4),
                max_req_input_len=4,
            )
        )
        scheduler._sr_tree_req_alive = MethodType(
            load_functions(
                SCHEDULER,
                ["_sr_tree_req_alive"],
                dict(),
                class_name="StandaloneRemoteDraftSchedulerMixin",
            )["_sr_tree_req_alive"],
            scheduler,
        )
        inspects = []
        scheduler._sr_inspect_tree_plans = lambda reqs: inspects.append(list(reqs)) or []
        alive = scheduler._sr_prepare_tree_reqs([req])
        self.assertEqual(alive, [])
        self.assertTrue(scheduler._sr_is_degraded("a"))
        self.assertTrue(lease.released)
        self.assertIsNone(store.get("a"))
        scheduler._sr_run_tree_ingest([req])
        self.assertEqual(inspects, [[]])

    def test_recovery_once_then_retract_original_hit(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            SRTreeLeaseStore,
        )

        store = SRTreeLeaseStore()
        lease = self._lease("a")
        store.register(lease)
        a = NS(rid="a")
        b = NS(rid="b")
        copy_plan = _FakeTailPlan(req=a, copy_src_slots=[3], copy_lease=lease, length=1)
        ordinary_a = _FakeTailPlan(req=a, length=1)
        ordinary_b = _FakeTailPlan(req=b, length=1)
        inspects = []
        recovered = []
        executed = []

        def inspect(reqs):
            inspects.append([r.rid for r in reqs])
            if len(inspects) == 1:
                return [
                    _SRTreePlanDraft(a, "copy", plan=copy_plan, lease=lease),
                    _SRTreePlanDraft(b, "recover", lease=None),
                ]
            return [
                _SRTreePlanDraft(a, "ordinary", plan=ordinary_a, lease=lease, miss="revision"),
                _SRTreePlanDraft(b, "ordinary", plan=ordinary_b),
            ]

        scheduler = self._bind(
            NS(sr_tree_leases=store),
            execute=lambda plans: executed.append([p.req.rid for p in plans]) or True,
        )
        scheduler._sr_prepare_tree_reqs = lambda reqs: reqs
        scheduler._sr_inspect_tree_plans = inspect
        scheduler._sr_reprefill_committed = lambda reqs: recovered.append([r.rid for r in reqs])
        self.assertTrue(scheduler._sr_run_tree_ingest([a, b]))
        self.assertEqual(recovered, [["b"]])
        self.assertEqual(len(inspects), 2)
        self.assertEqual(executed, [["a", "b"]])
        self.assertTrue(lease.released)
        self.assertEqual(store.counts["tree_kv_commit_hit"], 0)

    def test_second_recovery_is_degraded_not_retried(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            SRTreeLeaseStore,
        )

        store = SRTreeLeaseStore()
        a = NS(rid="a")
        recovered = []
        executed = []
        degraded = []
        scheduler = self._bind(
            NS(sr_tree_leases=store),
            execute=lambda plans: executed.append(list(plans)) or True,
        )
        scheduler._sr_prepare_tree_reqs = lambda reqs: reqs
        scheduler._sr_inspect_tree_plans = lambda reqs: [_SRTreePlanDraft(a, "recover")]
        scheduler._sr_reprefill_committed = lambda reqs: recovered.append([r.rid for r in reqs])
        scheduler._sr_mark_degraded = lambda rid, reason: degraded.append((rid, reason))
        self.assertTrue(scheduler._sr_run_tree_ingest([a]))
        self.assertEqual(recovered, [["a"]])
        self.assertEqual(executed, [])
        self.assertEqual(degraded[0][0], "a")

    def test_seed_fail_does_not_partial_commit_or_count_hits(self):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            SRTreeLeaseStore,
        )

        reqs = [request(3, (7, 8), 0), request(5, (9,), 1)]
        scheduler, plans = transaction_fixture(reqs, 128)
        store = SRTreeLeaseStore()
        lease = self._lease(reqs[0].rid)
        store.register(lease)
        plans[0] = replace(plans[0], copy_src_slots=[7], copy_lease=lease)
        scheduler.sr_tree_leases = store
        scheduler.model_config = NS(vocab_size=32)
        scheduler.server_args = NS(speculative_eagle_topk=3)
        scheduler.spec_algorithm = "STANDALONE_REMOTE"
        scheduler.tp_size = 1
        built = []

        def make(ps):
            batch = NS(
                is_sr_tail_extend=True,
                device="cpu",
                get_model_worker_batch=lambda: built[0] if built else NS(),
            )
            built.append(batch)
            return batch

        self._bind(scheduler)
        scheduler.token_to_kv_pool_allocator.free = lambda slots: None
        scheduler._sr_replay_grammars = Mock()
        scheduler._sr_pause_req = Mock()
        scheduler._sr_mark_degraded = Mock()
        scheduler._sr_make_tail_extend_batch = make
        scheduler.tp_worker = NS(
            model_runner=NS(model_is_mrope=False, attn_backend=NS()),
            forward_batch_generation=Mock(return_value=NS(logits_output=seed_output(1))),
        )
        before = scheduler.req_to_token_pool.req_to_token.clone()
        with patch.object(tail.SRTailExtendTransaction, "copy_reused_tree_kv"):
            committed = scheduler._sr_execute_tree_tails(plans)
        self.assertFalse(committed)
        torch.testing.assert_close(scheduler.req_to_token_pool.req_to_token, before)
        self.assertEqual([r.kv_committed_len for r in reqs], [3, 5])
        self.assertEqual(scheduler._sr_mark_degraded.call_count, 2)
        scheduler._sr_prepare_tree_reqs = lambda reqs: reqs
        scheduler._sr_inspect_tree_plans = lambda reqs: [
            _SRTreePlanDraft(reqs[0], "copy", plan=plans[0], lease=lease),
            _SRTreePlanDraft(reqs[1], "ordinary", plan=plans[1]),
        ]
        with patch.object(tail.SRTailExtendTransaction, "copy_reused_tree_kv"):
            self.assertFalse(scheduler._sr_run_tree_ingest(reqs))
        self.assertEqual(store.counts["tree_kv_commit_hit"], 0)
        scheduler.tp_worker.forward_batch_generation.assert_called()
        self.assertEqual(scheduler.tp_worker.forward_batch_generation.call_count, 2)

    def test_graph_bucket_miss_is_one_eager_forward(self):
        reqs = [request(3, (7, 8), 0), request(5, (9,), 1)]
        scheduler, plans = transaction_fixture(reqs, 128)
        seed = seed_output(2)
        batch = NS(
            is_sr_tail_extend=True,
            device="cpu",
            extend_lens=[2, 1],
            prefix_lens=[3, 5],
            extend_num_tokens=3,
            get_model_worker_batch=lambda: NS(),
        )
        runner = NS(
            plan=lambda _batch: None,
            plan_with_reason=lambda _batch: (None, "no_bucket"),
            init_forward_batch=Mock(side_effect=AssertionError("graph")),
            fill=Mock(side_effect=AssertionError("graph")),
            replay_filled=Mock(side_effect=AssertionError("graph")),
            eager_fallback_count=0,
        )
        scheduler.model_config = NS(vocab_size=32)
        scheduler.server_args = NS(speculative_eagle_topk=3)
        scheduler.spec_algorithm = "STANDALONE_REMOTE"
        scheduler.tp_size = 1
        scheduler._sr_replay_grammars = Mock()
        scheduler._sr_pause_req = Mock()
        scheduler._sr_mark_degraded = Mock()
        scheduler._sr_make_tail_extend_batch = lambda ps: batch
        scheduler.sr_tree_drafter = NS(tail_graph_runner=runner)
        scheduler.tp_worker = NS(
            model_runner=NS(model_is_mrope=False, attn_backend=NS()),
            forward_batch_generation=Mock(return_value=NS(logits_output=seed)),
        )
        method = load_functions(
            SCHEDULER,
            ["_sr_execute_tree_tails"],
            dict(
                time=time,
                logger=logging.getLogger(__name__),
                get_sr_round_metrics=get_sr_round_metrics,
                plan_tail_extend=tail.plan_tail_extend,
                TailExtendRecoveryRequired=tail.TailExtendRecoveryRequired,
                tree_seed_is_current=tail.tree_seed_is_current,
                SRTailExtendTransaction=tail.SRTailExtendTransaction,
                _sr_is_device_context_error=lambda exc: False,
                NpuGraphReplaySubmittedError=type("Submitted", (Exception,), {}),
                NpuGraphPreparationError=type("Prep", (Exception,), {}),
            ),
            class_name="StandaloneRemoteDraftSchedulerMixin",
        )["_sr_execute_tree_tails"]
        method(scheduler, plans)
        scheduler.tp_worker.forward_batch_generation.assert_called_once()
        self.assertEqual(runner.eager_fallback_count, 1)
        self.assertEqual([r.kv_committed_len for r in reqs], [5, 6])


class TestBatchedKvCopy(unittest.TestCase):
    def setUp(self):
        self.evict = patch.object(tail, "_evict_tail_capacity")
        self.evict.start()
        self.addCleanup(self.evict.stop)

    def _drafter(self, pool, copies, node_ids=None):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            remap_slot_node_ids,
        )
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            copy_kv_pool_by_slot,
        )

        fns = load_functions(
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py",
            ["_remap_tree_kv_to_parents", "_copy_tree_kv_slots"],
            dict(
                torch=torch,
                remap_slot_node_ids=remap_slot_node_ids,
                copy_kv_pool_by_slot=lambda *a, **k: copies.append(
                    (a[1].detach().clone(), a[2].detach().clone())
                )
                or copy_kv_pool_by_slot(*a, **k),
            ),
            class_name="SRTreeDrafter",
        )
        obj = NS(
            draft_model_runner=NS(token_to_kv_pool=pool),
            _slot_node_ids=(
                torch.arange(16).reshape(4, 4)
                if node_ids is None
                else node_ids.clone()
            ),
            _slot_node_id_tmp=torch.empty(8, dtype=torch.int64),
        )
        obj._copy_tree_kv_slots = MethodType(fns["_copy_tree_kv_slots"], obj)
        obj._remap_tree_kv_to_parents = MethodType(
            fns["_remap_tree_kv_to_parents"], obj
        )
        return obj

    @staticmethod
    def _staged_gold(buf, loc, parent, n_prev):
        gold = buf.clone()
        for step in range(int(n_prev)):
            src = loc[step].index_select(0, parent)
            tgt = loc[step]
            staged = gold.index_select(0, src)
            gold = gold.clone()
            gold.index_copy_(0, tgt, staged)
        return gold

    def test_remap_flattens_steps_and_matches_staged_gold(self):
        k = torch.arange(16, dtype=torch.float32).reshape(16, 1)
        v = k.clone() + 1
        pool = NS(kv_buffer=None, k_buffer=k.clone(), v_buffer=v.clone())
        copies = []
        drafter = self._drafter(pool, copies)
        loc = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64)
        parent = torch.tensor([1, 0], dtype=torch.int64)
        drafter._remap_tree_kv_to_parents(loc, parent, 2)
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0][0].tolist(), [6, 5, 8, 7])
        self.assertEqual(copies[0][1].tolist(), [5, 6, 7, 8])
        torch.testing.assert_close(pool.k_buffer, self._staged_gold(k, loc, parent, 2))
        torch.testing.assert_close(pool.v_buffer, self._staged_gold(v, loc, parent, 2))
        self.assertEqual(drafter._slot_node_ids[0, :2].tolist(), [1, 0])
        self.assertEqual(drafter._slot_node_ids[1, :2].tolist(), [5, 4])
        torch.testing.assert_close(pool.k_buffer[5], k[6])
        torch.testing.assert_close(pool.k_buffer[7], k[8])

    def test_remap_duplicate_parent_and_multi_request_offsets(self):
        k = torch.arange(32, dtype=torch.float32).reshape(32, 1)
        pool = NS(kv_buffer=None, k_buffer=k.clone(), v_buffer=k.clone() + 1)
        copies = []
        drafter = self._drafter(pool, copies)
        loc = torch.tensor([[5, 6]], dtype=torch.int64)
        parent = torch.tensor([0, 0], dtype=torch.int64)
        drafter._remap_tree_kv_to_parents(loc, parent, 1)
        self.assertEqual(len(copies), 1)
        torch.testing.assert_close(pool.k_buffer[5], k[5])
        torch.testing.assert_close(pool.k_buffer[6], k[5])
        self.assertEqual(drafter._slot_node_ids[0, :2].tolist(), [0, 0])

        pool = NS(kv_buffer=None, k_buffer=k.clone(), v_buffer=k.clone() + 1)
        copies = []
        drafter = self._drafter(pool, copies)
        loc = torch.tensor([[10, 11, 20, 21]], dtype=torch.int64)
        parent = torch.tensor([1, 0, 3, 2], dtype=torch.int64)
        drafter._remap_tree_kv_to_parents(loc, parent, 1)
        self.assertEqual(copies[0][0].tolist(), [11, 10, 21, 20])
        torch.testing.assert_close(pool.k_buffer[10], k[11])
        torch.testing.assert_close(pool.k_buffer[20], k[21])
        self.assertEqual(drafter._slot_node_ids[0, :4].tolist(), [1, 0, 3, 2])

    def test_remap_dummy_rows_stay_dummy_to_dummy(self):
        dummy = 0
        k = torch.arange(16, dtype=torch.float32).reshape(16, 1)
        dummy_val = k[dummy].clone()
        pool = NS(kv_buffer=None, k_buffer=k.clone(), v_buffer=k.clone() + 1)
        copies = []
        drafter = self._drafter(pool, copies)
        loc = torch.tensor([[5, 6, dummy, dummy]], dtype=torch.int64)
        parent = torch.tensor([1, 0, 2, 3], dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "parent_rows width"):
            drafter._remap_tree_kv_to_parents(loc, torch.tensor([1, 0]), 1)
        drafter._remap_tree_kv_to_parents(loc, parent, 1)
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0][0].tolist(), [6, 5, dummy, dummy])
        torch.testing.assert_close(pool.k_buffer[dummy], dummy_val)
        after_live = pool.k_buffer[5].clone()
        loc[:, 2] = 9
        copies.clear()
        drafter._remap_tree_kv_to_parents(loc, parent, 1)
        self.assertEqual(int(copies[0][0].shape[0]), 4)
        torch.testing.assert_close(pool.k_buffer[dummy], dummy_val)
        torch.testing.assert_close(pool.k_buffer[5], k[5])
        loc[:, 2] = dummy
        copies.clear()
        drafter._remap_tree_kv_to_parents(loc, parent, 1)
        self.assertEqual(int(copies[0][0].shape[0]), 4)
        torch.testing.assert_close(pool.k_buffer[dummy], dummy_val)
        torch.testing.assert_close(pool.k_buffer[5], after_live)

    def test_remap_n_prev_steps_and_noncontiguous_loc(self):
        k = torch.arange(16, dtype=torch.float32).reshape(16, 1)
        pool = NS(kv_buffer=None, k_buffer=k.clone(), v_buffer=k.clone() + 1)
        copies = []
        drafter = self._drafter(pool, copies)
        loc = torch.tensor([[5, 6], [7, 8]], dtype=torch.int64)
        parent = torch.tensor([1, 0], dtype=torch.int64)
        drafter._remap_tree_kv_to_parents(loc, parent, 0)
        self.assertEqual(copies, [])
        torch.testing.assert_close(pool.k_buffer, k)

        drafter._remap_tree_kv_to_parents(loc, parent, 1)
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0][0].tolist(), [6, 5])
        torch.testing.assert_close(pool.k_buffer, self._staged_gold(k, loc, parent, 1))

        loc_nc = torch.tensor([[5, 7], [6, 8]], dtype=torch.int64).t()
        self.assertFalse(loc_nc.is_contiguous())
        pool = NS(kv_buffer=None, k_buffer=k.clone(), v_buffer=k.clone() + 1)
        copies = []
        drafter = self._drafter(pool, copies)
        drafter._remap_tree_kv_to_parents(loc_nc, parent, 2)
        self.assertEqual(copies[0][0].tolist(), [6, 5, 8, 7])
        torch.testing.assert_close(
            pool.k_buffer, self._staged_gold(k, loc_nc.contiguous(), parent, 2)
        )

        copies.clear()
        parent2 = torch.tensor([0, 1], dtype=torch.int64)
        drafter._remap_tree_kv_to_parents(loc_nc, parent2, 2)
        self.assertEqual(copies[0][0].tolist(), [5, 6, 7, 8])

    def test_seq_lens_sum_from_cpu_and_mismatch(self):
        from sglang.srt.speculative.standalone_remote.sr_align import (
            seq_lens_sum_from_batch,
            sr_decode_seq_len,
        )

        reqs = [NS(kv_committed_len=3, origin_input_ids=[1], output_ids=[2])]
        batch = NS(
            reqs=reqs,
            seq_lens=torch.tensor([3, 4]),
            seq_lens_cpu=torch.tensor([3, 4]),
        )
        with self.assertRaisesRegex(RuntimeError, "request count"):
            seq_lens_sum_from_batch(batch)
        batch.reqs = [
            reqs[0],
            NS(kv_committed_len=4, origin_input_ids=[], output_ids=[]),
        ]
        self.assertEqual(seq_lens_sum_from_batch(batch), 7)
        batch.seq_lens = torch.tensor([7])
        with self.assertRaisesRegex(RuntimeError, "seq_lens rows"):
            seq_lens_sum_from_batch(batch)
        missing = NS(reqs=batch.reqs, seq_lens=None, seq_lens_cpu=None)
        self.assertEqual(seq_lens_sum_from_batch(missing), 7)
        self.assertEqual(sr_decode_seq_len(reqs[0]), 3)
        empty = NS(
            kv_committed_len=0, origin_input_ids=[1, 2, 3], output_ids=[4, 5]
        )
        self.assertEqual(sr_decode_seq_len(empty), 4)
        mixin = SCHEDULER.read_text(encoding="utf-8")
        self.assertIn("sr_decode_seq_len(r)", mixin)
        try:
            meta = torch.zeros(2, device="meta")
        except Exception:
            meta = None
        if meta is not None:
            with self.assertRaisesRegex(RuntimeError, "must stay on CPU"):
                seq_lens_sum_from_batch(
                    NS(
                        reqs=batch.reqs,
                        seq_lens=torch.tensor([3, 4]),
                        seq_lens_cpu=meta,
                    )
                )

    def test_paged_mapping_fits_reads_cpu_not_device_tolist(self):
        from sglang.srt.speculative.standalone_remote.sr_align import (
            seq_lens_cpu_for_host,
        )

        helpers = load_functions(
            ROOT / "python/sglang/srt/speculative/spec_utils.py",
            ["paged_tree_mapping_end", "paged_tree_mapping_fits"],
            {"torch": torch},
        )
        paged_tree_mapping_end = helpers["paged_tree_mapping_end"]
        paged_tree_mapping_fits = helpers["paged_tree_mapping_fits"]

        class Boom:
            shape = (2,)

            def tolist(self):
                raise AssertionError("device seq_lens.tolist")

        reqs = [
            NS(kv_committed_len=123, origin_input_ids=[], output_ids=[]),
            NS(kv_committed_len=124, origin_input_ids=[], output_ids=[]),
        ]
        cpu = torch.tensor([123, 124], dtype=torch.int64)
        batch = NS(reqs=reqs, seq_lens=Boom(), seq_lens_cpu=cpu)
        host = seq_lens_cpu_for_host(batch)
        page, topk, steps = 128, 3, 5
        end = max(paged_tree_mapping_end(int(x), page, topk, steps) for x in cpu)
        self.assertTrue(paged_tree_mapping_fits(host, page, topk, steps, end))
        self.assertFalse(paged_tree_mapping_fits(host, page, topk, steps, end - 1))
        self.assertTrue(paged_tree_mapping_fits(host, 1, topk, steps, 1))
        self.assertTrue(paged_tree_mapping_fits(host, page, 1, steps, 1))
        with self.assertRaisesRegex(RuntimeError, "request count"):
            seq_lens_cpu_for_host(
                NS(reqs=reqs[:1], seq_lens=Boom(), seq_lens_cpu=cpu)
            )
        with self.assertRaisesRegex(RuntimeError, "seq_lens rows"):
            seq_lens_cpu_for_host(
                NS(
                    reqs=reqs,
                    seq_lens=NS(shape=(1,), tolist=Boom.tolist),
                    seq_lens_cpu=cpu,
                )
            )

    def test_merged_lease_copy_one_helper_and_hold(self):
        from dataclasses import replace

        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            SRTreeKVLease,
            SRTreeLeaseStore,
        )

        reqs = [request(10, (7, 8), 0), request(10, (9,), 1)]
        plans = []
        leases = []
        store = SRTreeLeaseStore()
        for req, src in zip(reqs, ([99, 100], [101])):
            lease = SRTreeKVLease(
                rid=req.rid,
                version=1,
                revision=0,
                base_committed_len=10,
                prefix_tokens=(1,),
                page_ids=[1],
                page_slots=torch.arange(4),
                candidate_slots=[0],
                parent_list=[],
                top_scores_index=[],
                draft_tokens=[],
            )
            store.register(lease)
            leases.append(lease)
            plans.append(
                replace(
                    tail.plan_tail_extend(
                        req, vocab_size=32, model_is_mrope=False
                    ),
                    materialized_len=int(req.kv_committed_len),
                    original_len=int(req.kv_committed_len),
                    copy_src_slots=src,
                    copy_lease=lease,
                )
            )
        scheduler, _ = transaction_fixture(reqs, 128)
        scheduler.sr_tree_leases = store
        pool = NS(
            kv_buffer=None,
            k_buffer=torch.zeros(128, 1, 2),
            v_buffer=torch.zeros(128, 1, 2),
        )
        scheduler.tp_worker = NS(model_runner=NS(token_to_kv_pool=pool))
        txn = tail.SRTailExtendTransaction(scheduler, plans)
        txn.allocate(NS(device="cpu"))
        calls = []
        as_calls = []
        real_as = torch.as_tensor
        order = []

        def spy_as(*args, **kwargs):
            as_calls.append(args[0] if args else None)
            return real_as(*args, **kwargs)

        def spy_copy(*args, **kwargs):
            calls.append(args)
            order.append("copy")

        def fake_record(device, stream=None, *, required=False):
            self.assertIsNotNone(txn._copy_hold)
            self.assertEqual(len(txn._copy_hold), 2)
            ev = NS(device=NS(type="npu"), required=required)
            order.append(("record", required))
            return ev

        class CopyStream:
            def wait_event(self, ev):
                order.append(("copy_wait", ev))

        class Ctx:
            def __enter__(self):
                order.append("ctx_in")
                return self

            def __exit__(self, *exc):
                order.append("ctx_out")
                return False

        scheduler._sr_kv_copy_ctx = lambda s: Ctx()
        layout = "sglang.srt.speculative.standalone_remote.sr_verify_layout"
        with patch.object(torch, "as_tensor", spy_as), patch.object(
            tail, "get_kv_copy_stream", return_value=CopyStream()
        ), patch.object(tail, "_is_cpu_device", return_value=False), patch(
            "sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease.record_device_event",
            fake_record,
        ), patch(
            layout + ".copy_kv_pool_by_slot", spy_copy
        ):
            txn.copy_reused_tree_kv()
        host_lists = [item for item in as_calls if isinstance(item, list)]
        self.assertEqual(host_lists, [[99, 100, 101]])
        self.assertEqual(len(calls), 1)
        self.assertEqual(int(calls[0][1].numel()), 3)
        self.assertIs(txn._copy_hold[0], calls[0][1])
        self.assertIs(txn._copy_hold[1], calls[0][2])
        self.assertTrue(txn.copy_submitted)
        self.assertIs(leases[0].pending_free_event, txn.copy_done_event)
        self.assertIs(leases[1].pending_free_event, txn.copy_done_event)
        self.assertEqual(
            [step if isinstance(step, str) else step[0] for step in order],
            ["record", "copy_wait", "ctx_in", "copy", "record", "ctx_out"],
        )

    def test_copy_record_failure_after_submit_keeps_hold(self):
        scheduler, txn, _, _ = self._copy_job_txn()
        records = []

        def fake_record(device, stream=None, *, required=False):
            records.append(required)
            if len(records) == 1:
                return NS(device=NS(type="npu"))
            raise RuntimeError("record failed")

        class CopyStream:
            def wait_event(self, ev):
                return None

        class Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        scheduler._sr_kv_copy_ctx = lambda s: Ctx()
        layout = "sglang.srt.speculative.standalone_remote.sr_verify_layout"
        with patch.object(tail, "get_kv_copy_stream", return_value=CopyStream()), patch.object(
            tail, "_is_cpu_device", return_value=False
        ), patch(
            "sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease.record_device_event",
            fake_record,
        ), patch(
            layout + ".copy_kv_pool_by_slot", lambda *a, **k: None
        ):
            with self.assertRaisesRegex(RuntimeError, "record failed"):
                txn.copy_reused_tree_kv()
        self.assertTrue(txn.copy_submitted)
        self.assertIsNotNone(txn._copy_hold)
        self.assertEqual(len(txn._copy_hold), 2)

    def test_execute_copy_failure_classes(self):
        hold = (torch.tensor([1]), torch.tensor([2]))

        class BaseTxn:
            def __init__(self, scheduler, plans):
                self.scheduler = scheduler
                self.submitted = False
                self.copy_submitted = False
                self._copy_hold = hold
                self.committed = False

            def allocate(self, batch):
                return None

            def wait_copy_done(self):
                raise AssertionError("should not wait after copy failure")

            def commit(self, logits):
                return None

        class NotSubmitted(BaseTxn):
            def copy_reused_tree_kv(self):
                raise RuntimeError("wait_event missing")

            def rollback(self):
                self._copy_hold = None

        class SubmittedSyncOk(BaseTxn):
            def copy_reused_tree_kv(self):
                self.copy_submitted = True
                raise RuntimeError("copy helper failed")

            def rollback(self):
                self.scheduler.device_module.synchronize()
                self._copy_hold = None

        class SubmittedUnconfirmed(BaseTxn):
            def copy_reused_tree_kv(self):
                self.copy_submitted = True
                raise RuntimeError("copy helper failed")

            def rollback(self):
                raise RuntimeError("synchronize failed")

        cases = [
            (NotSubmitted, False, False, "wait_event missing"),
            (SubmittedSyncOk, True, False, "copy helper failed"),
            (SubmittedUnconfirmed, True, True, "synchronize failed"),
        ]
        for txn_cls, submitted, fatal, msg in cases:
            with self.subTest(txn=txn_cls.__name__):
                execute, scheduler, reqs, _submitted = self._execute(txn_cls)
                if fatal:
                    with self.assertRaisesRegex(RuntimeError, msg):
                        execute(scheduler, _tail_plans_for(reqs))
                    self.assertEqual(scheduler._sr_pending_copy_holds, [hold])
                    scheduler._sr_mark_degraded.assert_not_called()
                    scheduler._sr_pause_req.assert_not_called()
                else:
                    self.assertFalse(execute(scheduler, _tail_plans_for(reqs)))
                    self.assertFalse(getattr(scheduler, "_sr_pending_copy_holds", []))
                    scheduler._sr_mark_degraded.assert_called()
                    self.assertIsNone(
                        getattr(scheduler, "_last_txn", None)
                    )
                self.assertEqual(bool(submitted), txn_cls is not NotSubmitted)

    def _execute(self, txn_cls):
        fns = load_functions(
            SCHEDULER,
            ["_sr_execute_tree_tails", "_sr_park_copy_hold"],
            dict(
                time=time,
                logger=logging.getLogger(__name__),
                get_sr_round_metrics=get_sr_round_metrics,
                plan_tail_extend=tail.plan_tail_extend,
                TailExtendRecoveryRequired=tail.TailExtendRecoveryRequired,
                tree_seed_is_current=tail.tree_seed_is_current,
                SRTailExtendTransaction=txn_cls,
                _sr_is_device_context_error=lambda exc: False,
                NpuGraphReplaySubmittedError=type("Submitted", (Exception,), {}),
                NpuGraphPreparationError=type("Prep", (Exception,), {}),
            ),
        )
        reqs = [request(3, (7, 8), 0)]
        scheduler, _ = transaction_fixture(reqs, 128)
        scheduler.model_config = NS(vocab_size=32)
        scheduler.server_args = NS(speculative_eagle_topk=3)
        scheduler.spec_algorithm = "STANDALONE_REMOTE"
        scheduler.tp_size = 1
        scheduler._sr_replay_grammars = Mock()
        scheduler._sr_pause_req = Mock()
        scheduler._sr_mark_degraded = Mock()
        scheduler._sr_make_tail_extend_batch = lambda plans: NS(
            get_model_worker_batch=lambda: NS()
        )
        scheduler.sr_tree_drafter = None
        scheduler.tp_worker = NS(
            model_runner=NS(model_is_mrope=False, attn_backend=NS()),
            forward_batch_generation=Mock(
                return_value=NS(logits_output=seed_output(1))
            ),
        )
        scheduler._sr_park_copy_hold = MethodType(
            fns["_sr_park_copy_hold"], scheduler
        )
        return fns["_sr_execute_tree_tails"], scheduler, reqs, None

    def _copy_job_txn(self):
        from dataclasses import replace

        req = request(10, (7, 8))
        plan = replace(
            tail.plan_tail_extend(
                req, vocab_size=32, model_is_mrope=False, materialized_len=12
            ),
            materialized_len=10,
            original_len=10,
            copy_src_slots=[99, 100],
        )
        scheduler, _ = transaction_fixture([req], 128)
        scheduler.tp_worker = NS(
            model_runner=NS(
                token_to_kv_pool=NS(
                    kv_buffer=None,
                    k_buffer=torch.zeros(128, 1, 2),
                    v_buffer=torch.zeros(128, 1, 2),
                    move_kv_cache=None,
                    _kv_copy_config=None,
                )
            )
        )
        txn = tail.SRTailExtendTransaction(scheduler, [plan])
        txn.allocate(NS(device="cpu"))
        return scheduler, txn, plan, req


if __name__ == "__main__":
    unittest.main()
