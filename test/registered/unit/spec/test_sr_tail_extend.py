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
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative.standalone_remote.drafter import sr_tail_extend as tail
from sglang.srt.speculative.standalone_remote.sr_round_metrics import SRRoundMetrics
from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
    get_sr_round_metrics,
)
from sglang.srt.speculative.standalone_remote.sr_tail_attention import (
    build_tail_attention_metadata,
    validate_tail_forward_batch,
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
    )
    plans = [
        tail.plan_tail_extend(r, vocab_size=32, model_is_mrope=False) for r in reqs
    ]
    return scheduler, plans


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
                        out.hidden_states.zero_()
                        self.assertTrue(req.sr_tree_seed[2].any())

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
            ["_sr_ingest_tree_tails"],
            dict(
                time=time,
                logger=logging.getLogger(__name__),
                get_sr_round_metrics=get_sr_round_metrics,
                plan_tail_extend=tail.plan_tail_extend,
                TailExtendRecoveryRequired=tail.TailExtendRecoveryRequired,
                tree_seed_is_current=tail.tree_seed_is_current,
                SRTailExtendTransaction=tail.SRTailExtendTransaction,
                _sr_is_device_context_error=lambda exc: False,
            ),
        )["_sr_ingest_tree_tails"]
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
                method(scheduler, reqs)
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
                    method(scheduler, reqs)
                    scheduler.tp_worker.forward_batch_generation.assert_called_once()


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


if __name__ == "__main__":
    unittest.main()
