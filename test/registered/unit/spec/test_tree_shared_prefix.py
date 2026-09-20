"""CPU execution of shared-prefix production math, metadata and NPU dispatch."""

import ast
import logging
import math
import os
import threading
import time
import unittest
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.speculative import tree_shared_prefix as shared
from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
    SR_TREE_PAGED_ENV,
    SRTreePagedMetadata,
    build_step_context_lens,
    context_lens_list,
    read_sr_tree_paged_env,
)
from sglang.srt.speculative.standalone_remote.verifier.sr_target_tree_fia import (
    IMPL_TREE_PAGED_FIA,
    SR_TARGET_TREE_FIA_ENV,
    maybe_select_target_tree_fia,
    read_sr_target_tree_fia_env,
    target_tree_fia_blocked_extra_combos,
)
from sglang.srt.speculative.standalone_remote.sr_align import is_device_context_error
from sglang.srt.speculative.standalone_remote.sr_round_metrics import SRRoundMetrics
from sglang.srt.speculative.tree_attn_fallback import (
    build_tree_verify_kv_slots_ref,
    tree_fia_actual_seq_lengths_kv,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
NPU = ROOT / "python/sglang/srt/hardware_backend/npu"


@contextmanager
def isolated_sr_tree_paged_env(value=None, target_fia=None):
    """Isolate Draft/Target tree env vars. value=None unsets the variable."""
    extra = {}
    cleaned = {
        k: v
        for k, v in os.environ.items()
        if k not in (SR_TREE_PAGED_ENV, SR_TARGET_TREE_FIA_ENV)
    }
    if value is not None:
        extra[SR_TREE_PAGED_ENV] = value
    if target_fia is not None:
        extra[SR_TARGET_TREE_FIA_ENV] = target_fia
    cleaned.update(extra)
    with patch.dict(os.environ, cleaned, clear=True):
        yield


def methods(path, cls, names, ns=None):
    """Execute production methods without loading accelerator-only imports."""
    ns = dict(ns or {})
    tree = ast.parse(path.read_text(encoding="utf-8"))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    nodes = [
        n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(nodes) == len(names)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        ns,
    )
    return {name: ns[name] for name in names}


def mask_for(lengths, queries, device="cpu"):
    # Root 0; siblings 1,2; subsequent nodes extend the first sibling's path.
    tree = torch.eye(queries, dtype=torch.bool)
    tree[:, 0] = True
    for q in range(3, queries):
        tree[q, 1] = True
    return torch.cat(
        [
            torch.cat((torch.ones(queries, p, dtype=torch.bool), tree), 1).flatten()
            for p in lengths
        ]
    ).to(device)


def visible_slots(md, b, q):
    p, a = int(md.prefix_lens[b]), int(md.path_lens[b, q])
    return torch.cat(
        (md.prefix_slots[b, :p], md.node_slots[b, md.ancestor_indices[b, q, :a]])
    )


def dense_reference(q, k, v, md):
    bs, queries, heads, dim = q.shape
    kv_heads = k.shape[-2]
    k, v = k.reshape(-1, kv_heads, dim).float(), v.reshape(-1, kv_heads, dim).float()
    out = torch.zeros_like(q, dtype=torch.float32)
    for b in range(bs):
        for row in range(queries):
            slots = visible_slots(md, b, row)
            if slots.numel():
                keys = k[slots].repeat_interleave(heads // kv_heads, 1)
                vals = v[slots].repeat_interleave(heads // kv_heads, 1)
                probs = (
                    torch.einsum("hd,shd->hs", q[b, row].float(), keys)
                    .mul(dim**-0.5)
                    .softmax(-1)
                )
                out[b, row] = torch.einsum("hs,shd->hd", probs, vals)
    return out.reshape(bs * queries, heads * dim)


class TestSharedPrefix(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.set_num_threads(1)

    def fixture(
        self,
        lengths=(3, 5),
        queries=4,
        cap=512,
        dtype=torch.float32,
        heads=4,
        kv_heads=2,
        dim=64,
    ):
        table = (torch.randperm(2048).view(2, 1024) + 1).int()
        pool = torch.tensor([1, 0])
        md = shared.SharedPrefixMetadata.allocate(
            3, queries, cap, queries, queries, "cpu"
        )
        nodes = torch.arange(2100, 2100 + len(lengths) * queries)
        shared.fill_shared_verify_(
            md, table, pool, list(lengths), nodes, mask_for(lengths, queries), queries
        )
        q = torch.randn(3, queries, heads, dim).to(dtype)
        k = torch.randn(18, 128, kv_heads, dim).to(dtype)
        v = torch.randn_like(k)
        return md, q, k, v, table, pool, nodes

    def test_numerics_and_padding(self):
        for heads, kv_heads in ((2, 2), (4, 2)):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for lengths in ((0, 0), (3, 5), (257, 511)):
                    md, q, k, v, *_ = self.fixture(
                        lengths, dtype=dtype, heads=heads, kv_heads=kv_heads
                    )
                    # Unused slots may contain garbage: padding must remain zero.
                    k.view(-1, kv_heads, 64)[0].fill_(float("nan"))
                    v.view(-1, kv_heads, 64)[0].fill_(float("nan"))
                    actual = shared.shared_prefix_attention(
                        q, k, v, md, scale=1 / 8, kv_heads=kv_heads
                    )
                    expected = dense_reference(q, k, v, md).to(dtype)
                    # The last cast can straddle a rounding boundary by one ULP.
                    tolerance = (
                        (2e-5, 2e-6)
                        if dtype == torch.float32
                        else ((1e-2, 1e-3) if dtype == torch.bfloat16 else (2e-3, 1e-3))
                    )
                    torch.testing.assert_close(
                        actual, expected, rtol=tolerance[0], atol=tolerance[1]
                    )
                    self.assertEqual(torch.count_nonzero(actual[-q.shape[1] :]), 0)

    def test_union_normalization_not_sum_of_outputs(self):
        md = shared.SharedPrefixMetadata.allocate(1, 1, 1, 1, 1, "cpu")
        md.prefix_slots.fill_(1)
        md.node_slots.fill_(2)
        md.prefix_lens.fill_(1)
        md.path_lens.fill_(1)
        q = torch.ones(1, 1, 1, 64)
        k = torch.zeros(3, 1, 64)
        k[1].fill_(100)
        k[2].fill_(-100)
        v = torch.zeros_like(k)
        v[1].fill_(2)
        v[2].fill_(7)
        out = shared.shared_prefix_attention(q, k, v, md, scale=1 / 8, kv_heads=1)
        torch.testing.assert_close(out, torch.full_like(out, 2))
        self.assertFalse(torch.allclose(out, torch.full_like(out, 9)))
        md.path_lens.zero_()
        torch.testing.assert_close(
            shared.shared_prefix_attention(q, k, v, md, scale=1 / 8, kv_heads=1), out
        )
        md.prefix_lens.zero_()
        self.assertEqual(
            shared.shared_prefix_attention(
                q, k, v, md, scale=1 / 8, kv_heads=1
            ).count_nonzero(),
            0,
        )

    def test_sibling_isolation_and_prefix_visibility(self):
        md, q, k, v, *_ = self.fixture()
        run = lambda: shared.shared_prefix_attention(
            q, k, v, md, scale=1 / 8, kv_heads=2
        )
        before = run()
        v.view(-1, 2, 64)[md.node_slots[0, 2]] += 20
        after = run()
        torch.testing.assert_close(before[1], after[1])
        self.assertFalse(torch.allclose(before[2], after[2]))
        v.view(-1, 2, 64)[md.prefix_slots[0, 0]] += 20
        self.assertFalse(torch.allclose(after[1], run()[1]))

    def test_verify_layout_and_short_replay(self):
        md, q, k, v, table, pool, nodes = self.fixture((127, 129))
        pointers = [x.data_ptr() for x in vars(md).values()]
        for lengths in ([127, 129], [128, 0], [3]):
            mask = mask_for(lengths, 4)
            shared.fill_shared_verify_(md, table, pool, lengths, nodes, mask, 4)
            slots, lens = build_tree_verify_kv_slots_ref(
                mask, lengths, table, pool, nodes, 4
            )
            for b in range(len(lengths)):
                for row in range(4):
                    torch.testing.assert_close(
                        visible_slots(md, b, row),
                        slots[b * 4 + row, : lens[b * 4 + row]],
                    )
            self.assertEqual(pointers, [x.data_ptr() for x in vars(md).values()])
            self.assertEqual(md.path_lens[len(lengths) :].count_nonzero(), 0)
            actual = shared.shared_prefix_attention(
                q, k, v, md, scale=1 / 8, kv_heads=2
            )
            torch.testing.assert_close(
                actual, dense_reference(q, k, v, md), rtol=2e-5, atol=2e-6
            )

    def test_draft_page_layout_and_remapped_values(self):
        # Load the old production slot builder as an independent layout oracle.
        path = ROOT / "python/sglang/srt/speculative/spec_utils.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "build_tree_draft_kv_slots"
        )
        ns = {"torch": torch}
        mod = ast.parse("from __future__ import annotations")
        mod.body.append(node)
        exec(compile(mod, str(path), "exec"), ns)
        table = torch.randperm(4096).view(2, 2048).int()
        pool = torch.tensor([1, 0])
        for page in (1, 128):
            for prefix in (127, 128, 129):
                lengths = [prefix, 7]
                md = shared.SharedPrefixMetadata.allocate(3, 3, 256, 15, 5, "cpu")
                for step in (0, 3, 1):
                    shared.fill_shared_draft_(
                        md,
                        table,
                        pool,
                        lengths,
                        page_size=page,
                        topk=3,
                        steps=5,
                        step=step,
                    )
                    slots, lens = ns["build_tree_draft_kv_slots"](
                        table, pool, torch.tensor(lengths), page, 3, step, 5
                    )
                    for b in range(2):
                        for row in range(3):
                            torch.testing.assert_close(
                                visible_slots(md, b, row),
                                slots[b * 3 + row, : lens[b * 3 + row]],
                            )
                    self.assertEqual(md.path_lens[2].count_nonzero(), 0)
                # The allocator/remapper changes physical contents, not paths.
                q = torch.randn(3, 3, 4, 64)
                k, v = torch.randn(32, 128, 2, 64), torch.randn(32, 128, 2, 64)
                flat = v.view(-1, 2, 64)
                flat[md.node_slots[0, 5]] = flat[md.node_slots[0, 0]].clone()
                actual = shared.shared_prefix_attention(
                    q, k, v, md, scale=1 / 8, kv_heads=2
                )
                torch.testing.assert_close(
                    actual, dense_reference(q, k, v, md), rtol=2e-5, atol=2e-6
                )

    def test_prefix_gather_is_per_request_per_chunk(self):
        md, q, k, v, *_ = self.fixture((257, 511))
        calls = []
        original = shared._gather

        def gather(cache, slots):
            result = original(cache, slots)
            calls.append((tuple(slots.shape), tuple(result.shape)))
            return result

        with patch.object(shared, "_gather", gather):
            shared.shared_prefix_attention(q, k, v, md, scale=1 / 8, kv_heads=2)
        self.assertEqual([x[0] for x in calls], [(3, 256)] * 4 + [(3, 4, 4)] * 2)
        self.assertEqual(calls[0][1], (3, 256, 2, 64))

    def test_draft_mapping_dtypes_and_fixed_buffers(self):
        class NoIndexPut(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if "index_put" in str(func):
                    raise AssertionError(f"metadata must use slice copy, got {func}")
                return func(*args, **(kwargs or {}))

        for dtype in (torch.int32, torch.int64):
            for page in (1, 128):
                for prefix in (127, 128, 129):
                    with self.subTest(dtype=dtype, page=page, prefix=prefix):
                        table = torch.randperm(4096).view(2, 2048).to(dtype)
                        pool = torch.tensor([1, 0])
                        md = shared.SharedPrefixMetadata.allocate(
                            2, 3, 256, 15, 5, "cpu"
                        )
                        pointers = [x.data_ptr() for x in vars(md).values()]
                        for step, lengths in (
                            (0, [prefix, 7]),
                            (3, [prefix]),
                            (1, [7, prefix]),
                        ):
                            pool = pool.flip(0)
                            with NoIndexPut():
                                shared.fill_shared_draft_(
                                    md,
                                    table,
                                    pool,
                                    lengths,
                                    page_size=page,
                                    topk=3,
                                    steps=5,
                                    step=step,
                                )
                            for b, p in enumerate(lengths):
                                stride = (
                                    5
                                    if page == 1
                                    else ((p % page + 5 + page - 1) // page) * page
                                )
                                for branch in range(3):
                                    expected = torch.cat(
                                        (
                                            table[pool[b], :p],
                                            table[
                                                pool[b],
                                                p
                                                + branch * stride : p
                                                + branch * stride
                                                + step
                                                + 1,
                                            ],
                                        )
                                    ).long()
                                    torch.testing.assert_close(
                                        visible_slots(md, b, branch), expected
                                    )
                                    self.assertEqual(
                                        md.node_slots[
                                            b, branch * 5 + step + 1 : (branch + 1) * 5
                                        ].count_nonzero(),
                                        0,
                                    )
                                self.assertEqual(
                                    md.ancestor_indices[
                                        b, :, step + 1 :
                                    ].count_nonzero(),
                                    0,
                                )
                            for tensor in vars(md).values():
                                self.assertEqual(
                                    tensor[len(lengths) :].count_nonzero(), 0
                                )
                            self.assertEqual(
                                pointers, [x.data_ptr() for x in vars(md).values()]
                            )
                            self.assertEqual(md.node_slots.dtype, torch.int64)
                            self.assertEqual(md.path_lens.dtype, torch.int32)

    def test_invalid_metadata_and_sparse_prefix_rejected(self):
        md, _, _, _, table, pool, nodes = self.fixture()
        mask = mask_for([3, 5], 4)
        mask[0] = False
        with self.assertRaises(RuntimeError):
            shared.fill_shared_verify_(md, table, pool, [3, 5], nodes, mask, 4)
        with self.assertRaises(ValueError):
            shared.fill_shared_verify_(
                md, table, pool, [513], nodes, mask_for([513], 4), 4
            )

    def test_head_dim_128_and_empty_draft_batch(self):
        md, q, k, v, *_ = self.fixture((0, 257), dim=128)
        result = shared.shared_prefix_attention(
            q, k, v, md, scale=128**-0.5, kv_heads=2
        )
        torch.testing.assert_close(
            result, dense_reference(q, k, v, md), rtol=2e-5, atol=2e-6
        )
        md = shared.SharedPrefixMetadata.allocate(2, 3, 256, 15, 5, "cpu")
        md.path_lens.fill_(4)
        shared.fill_shared_draft_(
            md,
            torch.zeros(1, 512, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            [],
            page_size=128,
            topk=3,
            steps=5,
            step=1,
        )
        self.assertEqual(md.path_lens.count_nonzero(), 0)

    def test_selection_checks_all_layers_and_cache(self):
        class Layer:
            attn_type = NS(value="decoder")
            qk_head_dim = v_head_dim = 64
            tp_q_head_num, tp_k_head_num, tp_v_head_num = 4, 2, 2
            is_cross_attention = False
            sliding_window_size = -1
            logit_cap = 0
            layer_id = 0

        layer = Layer()
        cache = torch.zeros(2, 128, 2, 64, dtype=torch.float16)
        pool = NS(get_key_buffer=lambda i: cache, get_value_buffer=lambda i: cache)
        runner = NS(
            server_args=NS(
                speculative_algorithm="STANDALONE_REMOTE",
                standalone_remote_role="target",
            ),
            model=NS(modules=lambda: [Layer(), layer]),
            token_to_kv_pool=pool,
        )
        backend = NS(
            model_runner=runner,
            _use_tree_compact_fia=lambda: True,
            verify_tree_topk=3,
            draft_topk=3,
            use_fia=False,
            use_mla=False,
            use_alibi=False,
            is_hybrid_swa=False,
            is_dllm_model=False,
            model_dtype=torch.float16,
            page_size=128,
        )
        fn = methods(
            NPU / "attention/ascend_backend.py",
            "AscendAttnBackend",
            ["_init_tree_shared_prefix"],
            dict(
                vars(shared),
                torch_npu=NS(get_npu_format=lambda x: 0),
                logger=logging.getLogger(__name__),
                read_sr_tree_paged_env=read_sr_tree_paged_env,
                read_sr_target_tree_fia_env=read_sr_target_tree_fia_env,
                target_tree_fia_blocked_extra_combos=target_tree_fia_blocked_extra_combos,
                maybe_select_target_tree_fia=maybe_select_target_tree_fia,
            ),
        )["_init_tree_shared_prefix"]

        def make_step(model_runner, **kwargs):
            inner = NS(**vars(backend))
            fn(inner)
            inner._use_tree_shared_prefix = (
                lambda: inner.tree_attention_impl == shared.SHARED_PREFIX_IMPL
            )
            return inner

        def init_multi_steps():
            init_steps = methods(
                NPU / "attention/ascend_backend.py",
                "AscendAttnMultiStepDraftBackend",
                ["__init__"],
                dict(AscendAttnBackend=make_step, AttnGraphRole=NS(TREE_DRAFT="draft")),
            )["__init__"]
            multi = NS()
            init_steps(multi, runner, 3, 5)
            return multi

        with patch.dict(
            "sys.modules",
            {"sglang.srt.layers.radix_attention": NS(RadixAttention=Layer)},
        ):
            with isolated_sr_tree_paged_env(None):
                fn(backend)
                self.assertEqual(backend.tree_attention_impl, IMPL_TREE_PAGED_FIA)
                for role, page, expected in (
                    ("draft", 128, "paged_atb"),
                    ("draft", 1, shared.SHARED_PREFIX_IMPL),
                    ("target", 128, IMPL_TREE_PAGED_FIA),
                    ("target", 1, shared.SHARED_PREFIX_IMPL),
                ):
                    runner.server_args.standalone_remote_role = role
                    backend.page_size = page
                    # A remote Draft is not necessarily a local draft worker.
                    for local_draft in (False, True):
                        runner.is_draft_worker = local_draft
                        fn(backend)
                        self.assertEqual(backend.tree_attention_impl, expected)
                runner.server_args.standalone_remote_role = "draft"
                runner.page_size = backend.page_size = 128
                backend.use_fia = True
                fn(backend)
                self.assertEqual(backend.tree_attention_impl, "paged_fia")
                backend.use_fia = False
                backend.draft_topk = 1
                fn(backend)
                self.assertEqual(backend.tree_attention_impl, "compact_fia")
                backend.draft_topk = 3
                multi = init_multi_steps()
                self.assertEqual(
                    [b.tree_attention_impl for b in multi.attn_backends],
                    ["paged_atb"] * 5,
                )
                self.assertTrue(multi._central_tree_draft_fill)
                self.assertTrue(
                    all(b._central_tree_draft_fill for b in multi.attn_backends)
                )
            with isolated_sr_tree_paged_env(None, target_fia="0"):
                runner.server_args.standalone_remote_role = "target"
                backend.page_size = 128
                backend.use_fia = False
                fn(backend)
                self.assertEqual(backend.tree_attention_impl, shared.SHARED_PREFIX_IMPL)
            with isolated_sr_tree_paged_env("0"):
                runner.server_args.standalone_remote_role = "draft"
                backend.page_size = 128
                backend.draft_topk = 3
                backend.use_fia = False
                fn(backend)
                self.assertEqual(backend.tree_attention_impl, "compact_fia")
                multi = init_multi_steps()
                self.assertEqual(
                    [b.tree_attention_impl for b in multi.attn_backends],
                    ["compact_fia"] * 5,
                )
            for env_value in (None, "1"):
                with isolated_sr_tree_paged_env(env_value):
                    for role in ("draft", "target"):
                        runner.server_args.standalone_remote_role = role
                        backend.page_size = 128
                        backend.draft_topk = 3
                        backend.use_fia = False
                        backend.use_mla = False
                        backend._use_tree_compact_fia = lambda: True
                        for attr, bad in (
                            ("sliding_window_size", 128),
                            ("is_cross_attention", True),
                            ("logit_cap", 1),
                            ("qk_head_dim", 256),
                        ):
                            original = getattr(layer, attr)
                            setattr(layer, attr, bad)
                            fn(backend)
                            self.assertEqual(
                                backend.tree_attention_impl, "compact_fia"
                            )
                            setattr(layer, attr, original)
                        backend.use_mla = True
                        backend._use_tree_compact_fia = lambda: False
                        fn(backend)
                        self.assertEqual(backend.tree_attention_impl, "chunked")
                        backend.use_mla = False
                        backend._use_tree_compact_fia = lambda: True
                        quantized = cache.to(torch.uint8)
                        pool.get_key_buffer = lambda i, _c=quantized: _c
                        pool.get_value_buffer = lambda i, _c=quantized: _c
                        fn(backend)
                        self.assertEqual(backend.tree_attention_impl, "compact_fia")
                        pool.get_key_buffer = lambda i: cache
                        pool.get_value_buffer = lambda i: cache
            with isolated_sr_tree_paged_env(None):
                runner.server_args.speculative_algorithm = "EAGLE"
                runner.server_args.standalone_remote_role = "draft"
                fn(backend)
                self.assertEqual(backend.tree_attention_impl, "compact_fia")

    def test_real_backend_dispatch_and_fallback(self):
        fn = methods(
            NPU / "attention/ascend_backend.py",
            "AscendAttnBackend",
            [
                "_run_tree_shared_prefix_attention",
                "_run_tree_draft_slot_gather",
                "_run_tree_verify_slot_gather",
            ],
            {
                "shared_prefix_attention": shared.shared_prefix_attention,
                "shared_prefix_layer_supported": shared.shared_prefix_layer_supported,
                "log_tree_draft_slot_gather_once": lambda *a: None,
                "log_tree_verify_fallback_once": lambda *a: None,
            },
        )
        md, q, k, v, *_ = self.fixture()
        layer = NS(
            attn_type=NS(value="decoder"),
            qk_head_dim=64,
            v_head_dim=64,
            tp_q_head_num=4,
            tp_k_head_num=2,
            tp_v_head_num=2,
            is_cross_attention=False,
            sliding_window_size=-1,
            logit_cap=0,
            scaling=1 / 8,
        )
        backend = NS(
            forward_metadata=NS(tree_shared=md), _use_tree_shared_prefix=lambda: True
        )
        backend._run_tree_shared_prefix_attention = MethodType(
            fn["_run_tree_shared_prefix_attention"], backend
        )
        for name in ("_run_tree_draft_slot_gather", "_run_tree_verify_slot_gather"):
            extra = (NS(), None) if "verify" in name else ()
            result = fn[name](
                backend, q, k, v, layer, *extra, qk_head_dim=64, v_head_dim=64
            )
            torch.testing.assert_close(
                result, dense_reference(q, k, v, md), rtol=2e-5, atol=2e-6
            )
        layer.sliding_window_size = 128
        self.assertFalse(shared.shared_prefix_layer_supported(layer))
        with self.assertRaises(RuntimeError):
            backend._run_tree_shared_prefix_attention(q, k, v, layer)
        backend._use_tree_shared_prefix = lambda: False
        backend._use_tree_compact_fia = lambda *args: True
        backend.is_hybrid_swa, backend.draft_topk, backend.page_size = False, 3, 128
        backend.forward_metadata.tree_draft_kv_slots = torch.zeros(1, 1)
        backend.forward_metadata.tree_draft_kv_lens_t = torch.ones(1)
        backend._run_tree_compact_fia = Mock(return_value="fallback")
        self.assertEqual(
            fn["_run_tree_draft_slot_gather"](
                backend, q, k, v, layer, qk_head_dim=64, v_head_dim=64
            ),
            "fallback",
        )


class TestTreeFailureHandling(unittest.TestCase):
    def setUp(self):
        # Load the actual exception type without importing accelerator modules.
        path = ROOT / "python/sglang/srt/speculative/spec_utils.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "NpuGraphReplaySubmittedError"
        )
        ns = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
        self.submitted_error = ns[node.name]
        self.logger = Mock()
        names = [
            "expand_batch",
            "_expand_one",
            "_log_tree_failure",
            "_init_cuda_graphs",
            "_pack_tree_windows",
            "_d2h_tree_outputs",
            "_acquire_host_staging",
        ]
        src_path = (
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py"
        )
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        helper_ns = dict(torch=torch)
        helper_nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in {"_as_2d", "_wait_d2h_event", "_ensure_host_staging_slot"}
        ]
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=helper_nodes, type_ignores=[])
                ),
                str(src_path),
                "exec",
            ),
            helper_ns,
        )
        functions = methods(
            src_path,
            "SRTreeDrafter",
            names,
            dict(
                logger=self.logger,
                is_device_context_error=is_device_context_error,
                NpuGraphReplaySubmittedError=self.submitted_error,
                device_backend_key=lambda _: "cuda",
                nullcontext=nullcontext,
                torch=torch,
                _as_2d=helper_ns["_as_2d"],
                _wait_d2h_event=helper_ns["_wait_d2h_event"],
                _ensure_host_staging_slot=helper_ns["_ensure_host_staging_slot"],
            ),
        )
        md = shared.SharedPrefixMetadata.allocate(1, 3, 256, 15, 5, "cpu")
        inner = NS(
            tree_attention_impl=shared.SHARED_PREFIX_IMPL,
            forward_metadata=NS(tree_shared=md),
        )
        self.drafter = NS(
            _tree_failure_counts={},
            _tree_batch_isolate_count=0,
            draft_attn_backend=NS(attn_backends=[inner]),
            req_to_token_pool=NS(req_to_token=torch.zeros(1, 512, dtype=torch.int32)),
            _stack_seeds=Mock(return_value=(None,) * 4),
            _expand_tree=Mock(side_effect=RuntimeError("metadata dtype mismatch")),
            server_args=NS(disable_cuda_graph=False),
            speculative_num_steps=5,
            draft_model_runner=NS(draft_attn_backend="original"),
            device="cpu",
        )
        for name, fn in functions.items():
            setattr(self.drafter, name, MethodType(fn, self.drafter))
        self.drafter._publish_tree_leases = lambda *a, **k: None
        self.drafter._free_lease_alloc = lambda *a, **k: None
        self.drafter._pending_lease_state = None
        self.req = NS(rid="request", req_pool_idx=0, sr_tree_seed=(None,) * 4)

    def test_single_eligible_request_does_not_retry(self):
        self.drafter._expand_one = Mock(side_effect=AssertionError("duplicate retry"))
        skipped = NS(req_pool_idx=None)
        result = self.drafter.expand_batch([skipped, self.req])
        self.assertEqual(result, [([], None, None)] * 2)
        self.drafter._expand_tree.assert_called_once()
        self.drafter._expand_one.assert_not_called()
        self.assertEqual(self.drafter._tree_batch_isolate_count, 0)

    def test_multi_request_failure_still_isolates_requests(self):
        output = ([9], [0], [0])
        self.drafter._expand_one = Mock(return_value=output)
        self.assertEqual(
            self.drafter.expand_batch([self.req, self.req]), [output, output]
        )
        self.assertEqual(self.drafter._expand_one.call_count, 2)
        self.assertEqual(self.drafter._tree_batch_isolate_count, 1)
        self.assertEqual(self.logger.warning.call_args.args[5], 1)

    def test_expand_batch_copies_each_tree_tensor_once(self):
        parent = torch.tensor([[-1, 0], [-1, 1]], dtype=torch.int64)
        index = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
        tokens = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int64)
        self.drafter._expand_tree = Mock(return_value=(parent, index, tokens))
        expand_one = Mock(side_effect=AssertionError("isolate"))
        self.drafter._expand_one = expand_one
        self.drafter.scheduler = NS(_sr_round_metrics=None)
        copies = {"n": 0}
        orig = torch.Tensor.copy_

        def counting_copy(self, *args, **kwargs):
            copies["n"] += 1
            return orig(self, *args, **kwargs)

        with patch.object(torch.Tensor, "copy_", counting_copy):
            windows = self.drafter.expand_batch([self.req, self.req])
        self.assertEqual(windows[0][0], [10, 11, 12])
        self.assertEqual(windows[1][0], [20, 21, 22])
        self.assertEqual(copies["n"], 3)
        self.assertEqual(self.drafter._tree_batch_isolate_count, 0)
        self.drafter._expand_one.assert_not_called()

    def test_submitted_and_device_errors_propagate(self):
        for exc in (
            self.submitted_error("submitted"),
            RuntimeError("illegal memory access"),
        ):
            for name in ("expand_batch", "_expand_one"):
                self.drafter._expand_tree.reset_mock(side_effect=True)
                self.drafter._expand_tree.side_effect = exc
                with self.assertRaises(type(exc)) as caught:
                    getattr(self.drafter, name)(
                        [self.req] if name == "expand_batch" else self.req
                    )
                self.assertIs(caught.exception, exc)
                self.drafter._expand_tree.assert_called_once()
        self.logger.warning.assert_not_called()
        self.assertEqual(self.drafter._tree_batch_isolate_count, 0)

    def test_submitted_multi_request_does_not_isolate(self):
        self.drafter._expand_tree.side_effect = self.submitted_error("submitted")
        expand_one = Mock(side_effect=AssertionError("isolate"))
        self.drafter._expand_one = expand_one
        with self.assertRaises(self.submitted_error):
            self.drafter.expand_batch([self.req, self.req])
        expand_one.assert_not_called()
        self.assertEqual(self.drafter._tree_batch_isolate_count, 0)

    def test_failures_log_traceback_once_then_totals(self):
        for _ in range(32):
            self.drafter.expand_batch([self.req])
        self.assertEqual(self.drafter._expand_tree.call_count, 32)
        self.assertEqual(self.logger.warning.call_count, 2)
        first, repeated = self.logger.warning.call_args_list
        self.assertEqual(first.args[1:3], ("expand_batch", shared.SHARED_PREFIX_IMPL))
        self.assertEqual(first.args[3]["req_to_token"], "torch.int32")
        self.assertEqual(first.args[3]["node_slots"], "torch.int64")
        self.assertIsNotNone(first.kwargs["exc_info"][2])
        self.assertEqual(repeated.args[4], 32)
        self.assertIsNone(repeated.kwargs["exc_info"])

    def test_capture_status_does_not_claim_configured_graph_is_ready(self):
        runner_cls = Mock()
        module = NS(EAGLEDraftCudaGraphRunner=runner_cls)
        with patch.dict(
            "sys.modules",
            {"sglang.srt.speculative.eagle_draft_cuda_graph_runner": module},
        ):
            for graphs, ready in (({}, False), ({1: object()}, True)):
                runner_cls.return_value = NS(graphs=graphs)
                self.drafter._init_cuda_graphs()
                self.assertEqual(self.drafter.tree_graph_capture_succeeded, ready)
                self.assertEqual(
                    self.drafter.tree_graph_disabled_reason,
                    None if ready else "no graphs",
                )
                self.assertEqual(
                    self.drafter.draft_model_runner.draft_attn_backend, "original"
                )
            runner_cls.side_effect = RuntimeError("capture failed")
            self.drafter._init_cuda_graphs()
            self.assertFalse(self.drafter.tree_graph_capture_succeeded)
            self.assertIsNone(self.drafter.cuda_graph_runner)
            self.assertIn("capture failed", self.drafter.tree_graph_disabled_reason)
            for exc in (
                self.submitted_error("submitted"),
                RuntimeError("illegal memory access"),
            ):
                runner_cls.side_effect = exc
                with self.assertRaises(type(exc)):
                    self.drafter._init_cuda_graphs()
                self.assertEqual(
                    self.drafter.draft_model_runner.draft_attn_backend, "original"
                )

    def test_target_normal_decode_counts_only_fallback(self):
        fn = methods(
            ROOT
            / "python/sglang/srt/speculative/standalone_remote/verifier/sr_worker.py",
            "StandaloneRemoteWorker",
            ["_forward_normal_decode"],
            dict(
                torch=torch,
                ForwardMode=NS(DECODE="decode"),
                GenerationBatchResult=NS,
                _align_seq_lens_to_committed=lambda batch: None,
                alloc_for_decode=lambda batch, **kw: torch.tensor([3]),
                _is_health_check=lambda req: False,
                _default_draft=lambda: {},
            ),
        )["_forward_normal_decode"]
        req = NS(
            output_ids=[2],
            kv_committed_len=3,
            kv_allocated_len=3,
            grammar=None,
            check_finished=Mock(),
        )
        metrics = NS(counts=Counter())
        batch = NS(
            reqs=[req],
            batch_size=lambda: 1,
            seq_lens=torch.tensor([3]),
            seq_lens_cpu=torch.tensor([3]),
            orig_seq_lens=None,
            seq_lens_sum=3,
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            return_logprob=False,
            get_model_worker_batch=lambda: NS(),
            sampling_info=NS(penalizer_orchestrator=NS(is_required=False)),
            sr_round_metrics=metrics,
        )
        worker = NS(
            _forward_target_eager=Mock(
                return_value=NS(next_token_ids=torch.tensor([4]), logits_output=None)
            )
        )
        result = fn(worker, batch)
        self.assertEqual(req.output_ids, [2, 4])
        self.assertEqual(result.accept_length_per_req_cpu, [1])
        self.assertFalse(result.can_run_cuda_graph)
        self.assertEqual(
            metrics.counts,
            Counter(
                normal_decode_fallback_batches=1, normal_decode_fallback_requests=1
            ),
        )

    def test_forward_device_event_ends_before_result_wait(self):
        for graph_mode in (True, False):
            trace = []
            events = []

            class Event:
                def __init__(self, **kwargs):
                    self.index = len(events)
                    events.append(self)

                def record(self):
                    trace.append(f"event{self.index}")

                def query(self):
                    return True

                def elapsed_time(self, end):
                    return 7.0

            class Result:
                def __getitem__(self, row):
                    return self

                def detach(self):
                    return self

                def to(self, device):
                    trace.append("result_d2h")
                    return self

                def tolist(self):
                    return [1]

            def forward(*args):
                trace.append("forward")
                return (Result(),) * 3

            metrics = SRRoundMetrics("Draft", NS(Event=Event))
            batch = NS(get_model_worker_batch=lambda: NS(), out_cache_loc=None)
            self.drafter.scheduler = NS(
                _sr_round_metrics=metrics, _sr_make_decode_batch=lambda reqs: batch
            )
            self.drafter.topk = 3
            self.drafter._alloc_tree_kv = Mock(return_value=("snapshot", None))
            self.drafter.token_to_kv_pool_allocator = NS(restore_state=Mock())
            self.drafter.draft_attn_backend.init_forward_metadata = Mock()
            self.drafter.cuda_graph_runner = NS(
                can_run=lambda batch: graph_mode,
                replay=Mock(side_effect=forward),
                tree_graph_replay_count=0,
            )
            self.drafter._draft_forward = forward
            fn = methods(
                ROOT
                / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tree_drafter.py",
                "SRTreeDrafter",
                ["_expand_tree"],
                dict(
                    time=time,
                    nullcontext=nullcontext,
                    logger=self.logger,
                    EagleDraftInput=NS,
                    CaptureHiddenMode=NS(LAST="last"),
                    ForwardBatch=NS(
                        init_new=lambda *args: NS(
                            forward_mode=NS(is_idle=lambda: False)
                        )
                    ),
                    NpuGraphReplaySubmittedError=self.submitted_error,
                    NpuGraphPreparationError=ValueError,
                    record_tree_expand_admission=lambda *args, **kwargs: False,
                ),
            )["_expand_tree"]
            self.drafter._expand_tree = MethodType(fn, self.drafter)
            with metrics.round():
                self.assertEqual(
                    self.drafter.expand_batch([self.req]), [([1], [1], [1])]
                )
            self.assertEqual(
                trace, ["event0", "forward", "event1"] + ["result_d2h"] * 3
            )
            self.assertIn("tree_forward", metrics.host)
            self.assertIn("tree_result_wait_pack", metrics.host)
            self.assertIn("tree_make_batch", metrics.host)
            self.assertIn("tree_alloc_kv", metrics.host)
            self.assertIn("tree_prepare_meta", metrics.host)
            self.assertEqual(len(metrics.pending), 1)
            metrics.poll()
            self.assertEqual(metrics.device_ms["tree_forward"], 7.0)
            self.drafter.token_to_kv_pool_allocator.restore_state.assert_called_once_with(
                "snapshot"
            )
            if graph_mode:
                # Never record an end event or restore memory after a submitted
                # replay failure; preserve the worker's fatal-error path.
                metrics.rounds = 32
                self.drafter.cuda_graph_runner.replay.side_effect = (
                    self.submitted_error("submitted")
                )
                self.drafter.token_to_kv_pool_allocator.restore_state.reset_mock()
                with self.assertRaises(self.submitted_error), metrics.round():
                    self.drafter.expand_batch([self.req])
                self.assertEqual(len(events), 4)
                self.assertEqual(trace[-1], "event2")
                self.assertEqual(len(metrics.pending), 0)
                self.drafter.token_to_kv_pool_allocator.restore_state.assert_not_called()


class TestGraphDispatch(unittest.TestCase):
    def test_compact_draft_updates_existing_fia_payload(self):
        path = ROOT / "python/sglang/srt/speculative/spec_utils.py"
        names = {
            "NpuGraphPreparationError",
            "NpuGraphReplaySubmittedError",
            "fill_fia_cpu_update_payload",
            "run_npu_graph_update_and_replay",
        }
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
        ]
        ns = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
        ns.update(
            logger=logging.getLogger(__name__),
            is_deepseek_nsa=lambda _: False,
            tree_fia_actual_seq_lengths_kv=tree_fia_actual_seq_lengths_kv,
        )
        fn = methods(
            NPU / "graph_runner/eagle_draft_npu_graph_runner.py",
            "EAGLEDraftNpuGraphRunner",
            ["_replay", "_assert_tree_replay_graph"],
            ns,
        )
        batch = NS(seq_lens_cpu=torch.tensor([127]))
        inners = [
            NS(_replay_tree_s_cap=256, tree_fia_kv_lens_cpu=[128 + i] * 3 + [0] * 3)
            for i in range(4)
        ]
        payload = [{"actual_seq_lengths_kv": [1] * 6} for _ in range(8)]
        addresses = [id(x["actual_seq_lengths_kv"]) for x in payload]
        graph = NS(update=Mock(), replay=Mock())
        plan = NS(
            graph_key="2_s256", capture_bs=2, raw_bs=1, tokens_per_req=3, kv_bucket=256
        )
        runner = NS(
            _tree_replay_plan=plan,
            _tree_replay_batch_id=id(batch),
            _tree_replay_graph=graph,
            graphs={plan.graph_key: graph},
            _tree_attention_impls={plan.graph_key: "compact_fia"},
            _current_tree_attention_impl=lambda: "compact_fia",
            model_runner=NS(
                draft_attn_backend=NS(attn_backends=inners),
                model_config=NS(hf_config=NS()),
            ),
            bs=2,
            raw_bs=1,
            num_tokens_per_bs=3,
            topk=3,
            _get_update_attr_name=lambda: "actual_seq_lengths_kv",
            _get_update_attr_type=lambda: [],
            output_buffers={plan.graph_key: object()},
            _slot_gather_graph=True,
            _tree_compact_fia=True,
            _tree_fia_maps={
                plan.graph_key: dict(
                    n_steps=4,
                    n_records=8,
                    num_layers=2,
                    step_ids=[0, 0, 1, 1, 2, 2, 3, 3],
                    payload=payload,
                )
            },
            _logged_tree_fia_update_bs=set(),
            tree_graph_replay_count=0,
            tree_eager_fallback_count=0,
        )
        runner._assert_tree_replay_graph = MethodType(
            fn["_assert_tree_replay_graph"], runner
        )
        for prefix in (127, 3):
            batch.seq_lens_cpu.fill_(prefix)
            for i, inner in enumerate(inners):
                inner.tree_fia_kv_lens_cpu[:] = [prefix + i + 1] * 3 + [0] * 3
            fn["_replay"](runner, batch)
            graph.update.assert_called_with(cpu_update_input=payload)
            for i, record in enumerate(payload):
                # FIA clamps padding to one safe slot; the slot-length tensor
                # remains zero and masks the padded outputs in the backend.
                self.assertEqual(
                    record["actual_seq_lengths_kv"], [prefix + i // 2 + 1] * 3 + [1] * 3
                )
            self.assertEqual(
                addresses, [id(x["actual_seq_lengths_kv"]) for x in payload]
            )
        self.assertEqual(graph.replay.call_count, 2)
        runner._tree_attention_impls[plan.graph_key] = shared.SHARED_PREFIX_IMPL
        with self.assertRaises(ns["NpuGraphPreparationError"]):
            fn["_replay"](runner, batch)
        self.assertEqual(graph.replay.call_count, 2)

    def test_oversized_batch_keeps_eager_compact_fallback(self):
        fn = methods(
            NPU / "attention/ascend_backend.py",
            "AscendAttnBackend",
            ["_prepare_shared_eager"],
            dict(vars(shared)),
        )["_prepare_shared_eager"]
        backend = NS(
            _is_tree_draft=lambda batch: True,
            draft_topk=3,
            draft_num_steps=5,
            tree_kv_buckets=[256, 512],
            _log_tree_fallback_once=Mock(),
            _shared_metadata=Mock(
                side_effect=AssertionError("must use old eager metadata")
            ),
        )
        self.assertFalse(
            fn(backend, NS(seq_lens_cpu=torch.tensor([510]), batch_size=1))
        )
        backend._shared_metadata.assert_not_called()

    def test_real_metadata_capture_rebinds_bucket_and_padding(self):
        names = [
            "_shared_metadata",
            "_sync_active_tree_s_cap",
            "_fill_shared_metadata",
            "init_forward_metadata_capture_cuda_graph",
            "init_forward_metadata_replay_cuda_graph",
        ]
        functions = methods(
            NPU / "attention/ascend_backend.py",
            "AscendAttnBackend",
            names,
            dict(vars(shared), ForwardMetadata=NS),
        )
        backend = NS(
            _use_tree_shared_prefix=lambda: True,
            _use_target_tree_paged_fia=lambda: False,
            draft_topk=1,
            draft_num_steps=5,
            verify_tree_topk=3,
            _shared_graph_metadata={},
            graph_metadata={},
            _replay_tree_s_cap=None,
            device="cpu",
            tree_kv_buckets=[256, 512],
            req_to_token=torch.arange(2048, dtype=torch.int32).view(2, 1024),
            _tree_verify_mask_layout=lambda spec, seq: (
                spec.seq_lens_cpu,
                len(spec.seq_lens_cpu),
            ),
        )
        for name, fn in functions.items():
            setattr(backend, name, MethodType(fn, backend))
        mode = NS(is_target_verify=lambda: True, is_decode_or_idle=lambda: False)
        pool = torch.tensor([0, 1])
        for width in (512, 256):
            backend._shared_capture_width = width
            backend.init_forward_metadata_capture_cuda_graph(
                2, 6, pool, torch.zeros(2), None, mode, NS()
            )
        addresses = {
            key: [v.data_ptr() for v in vars(md).values()]
            for key, md in backend._shared_graph_metadata.items()
        }
        nodes = torch.arange(1600, 1606)
        backend._graph_out_cache_loc = lambda n: nodes[:n]
        backend._shared_capture_width = None
        for width, lengths in ((256, [127, 3]), (512, [300]), (256, [3])):
            backend._replay_tree_s_cap = width
            spec = NS(
                draft_token_num=3,
                seq_lens_cpu=lengths,
                custom_mask=mask_for(lengths, 3),
            )
            backend.init_forward_metadata_replay_cuda_graph(
                2, pool, torch.zeros(2), 0, None, mode, spec, torch.tensor(lengths)
            )
            md = backend.forward_metadata.tree_shared
            self.assertEqual(md.prefix_slots.shape, (2, width))
            self.assertEqual(
                md.prefix_lens.tolist(), lengths + [0] * (2 - len(lengths))
            )
            self.assertEqual(
                addresses[(2, 3, width, False)],
                [v.data_ptr() for v in vars(md).values()],
            )

    def test_both_runners_skip_fia_and_propagate_submitted_failure(self):
        class PreparationError(RuntimeError):
            def __init__(self, message, **kwargs):
                super().__init__(message)

        class SubmittedError(RuntimeError):
            pass

        class Output(NS):
            pass

        for draft in (True, False):
            file = "eagle_draft_npu_graph_runner.py" if draft else "npu_graph_runner.py"
            klass = "EAGLEDraftNpuGraphRunner" if draft else "NPUGraphRunner"
            method = "_replay" if draft else "replay"
            fail = Mock(side_effect=AssertionError("FIA must not be used"))
            ns = {
                "is_deepseek_nsa": lambda cfg: False,
                "logger": logging.getLogger(__name__),
                "NpuGraphPreparationError": PreparationError,
                "NpuGraphReplaySubmittedError": SubmittedError,
                "LogitsProcessorOutput": Output,
                "tree_fia_actual_seq_lengths_kv": fail,
                "run_npu_graph_update_and_replay": fail,
            }
            fn = methods(
                NPU / "graph_runner" / file,
                klass,
                [method, "_assert_tree_replay_graph"],
                ns,
            )
            batch = NS()
            graph = NS(replay=Mock(), update=fail)
            backend = NS(
                _replay_tree_s_cap=256,
                _use_tree_compact_fia=lambda: True,
                _use_tree_shared_prefix=lambda: True,
                _use_target_tree_paged_fia=lambda: False,
            )
            plan = NS(
                graph_key="tree",
                capture_bs=1,
                raw_bs=1,
                tokens_per_req=3,
                kv_bucket=256,
            )
            runner = NS(
                _tree_replay_plan=plan,
                _tree_replay_batch_id=id(batch),
                _tree_replay_stream_idx=None,
                _tree_replay_graph=graph,
                graphs={"tree": graph},
                bs=1,
                raw_bs=1,
                raw_num_token=3,
                num_tokens_per_bs=3,
                output_buffers={
                    "tree": Output(
                        next_token_logits=torch.ones(3, 5), hidden_states=None
                    )
                },
                model_runner=NS(attn_backend=backend, model_config=NS(hf_config=NS())),
                _get_update_attr_name=lambda: "unused",
                _get_update_attr_type=lambda: [],
                _slot_gather_graph=True,
                _tree_compact_fia=False,
                tree_graph_replay_count=0,
                tree_eager_fallback_count=0,
                tree_verify_replay_count=0,
                tree_verify_eager_fallback_count=0,
                _current_tree_attention_impl=lambda: shared.SHARED_PREFIX_IMPL,
                _tree_attention_impls={"tree": shared.SHARED_PREFIX_IMPL},
                _clear_tree_replay_plan=Mock(),
                replay_prepare=Mock(),
                _is_tree_verify_batch=lambda b: True,
                is_dllm=False,
            )
            runner._assert_tree_replay_graph = MethodType(
                fn["_assert_tree_replay_graph"], runner
            )
            fn[method](runner, batch)
            graph.replay.assert_called_once()
            fail.assert_not_called()
            graph.replay.side_effect = RuntimeError("device error")
            with self.assertRaises(SubmittedError):
                fn[method](runner, batch)
            self.assertEqual(graph.replay.call_count, 2)
            runner._tree_attention_impls["tree"] = "compact_fia"
            with self.assertRaises(PreparationError):
                fn[method](runner, batch)
            self.assertEqual(graph.replay.call_count, 2)


def _load_spec_utils_errors():
    path = ROOT / "python/sglang/srt/speculative/spec_utils.py"
    names = {"NpuGraphPreparationError", "NpuGraphReplaySubmittedError"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name in names
    ]
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns["NpuGraphPreparationError"], ns["NpuGraphReplaySubmittedError"]


class _ForwardMetadata:
    def __init__(self):
        self.sr_tree_paged = None
        self.block_tables = None


class TestPagedReplayRestore(unittest.TestCase):
    def setUp(self):
        self.PrepError, self.SubmittedError = _load_spec_utils_errors()
        path = NPU / "graph_runner/eagle_draft_npu_graph_runner.py"
        names = [
            "_snapshot_forward_batch_fields",
            "_restore_forward_batch_fields",
            "_snapshot_paged_eager_metadata",
            "_paged_eager_restore_valid",
            "_restore_paged_eager_metadata",
            "_clear_tree_replay_plan",
            "_assert_tree_replay_graph",
            "replay",
        ]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        klass = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "EAGLEDraftNpuGraphRunner"
        )
        nodes = [
            n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        self.assertEqual({n.name for n in nodes}, set(names))

        class SuperReplay:
            def replay(self, forward_batch):
                return self._super_replay(forward_batch)

        ns = {
            "SuperReplay": SuperReplay,
            "NpuGraphPreparationError": self.PrepError,
            "NpuGraphReplaySubmittedError": self.SubmittedError,
            "ForwardBatch": object,
        }
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        class_def = ast.ClassDef(
            name="PagedReplayRunner",
            bases=[ast.Name(id="SuperReplay", ctx=ast.Load())],
            keywords=[],
            body=nodes,
            decorator_list=[],
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[future, class_def], type_ignores=[])
                ),
                str(path),
                "exec",
            ),
            ns,
        )
        self.Runner = ns["PagedReplayRunner"]
        bind_ns = dict(
            torch=torch,
            NpuGraphPreparationError=self.PrepError,
            build_step_context_lens=build_step_context_lens,
            context_lens_list=context_lens_list,
            SRTreePagedMetadata=SRTreePagedMetadata,
            ForwardMetadata=_ForwardMetadata,
        )
        fns = methods(
            NPU / "attention/ascend_backend.py",
            "AscendAttnMultiStepDraftBackend",
            [
                "_paged_graph_table_view",
                "_validate_sr_tree_paged_replay",
                "bind_sr_tree_paged_replay",
            ],
            bind_ns,
        )
        self.bind_fn = fns["bind_sr_tree_paged_replay"]
        self.validate_fn = fns["_validate_sr_tree_paged_replay"]
        self.view_fn = fns["_paged_graph_table_view"]

    def _eager_meta(self, tables, active, dummy=0):
        lens = torch.arange(int(tables.shape[0]), dtype=torch.int32) + 3
        return SRTreePagedMetadata(
            block_tables=tables,
            active_rows=active,
            context_lens_cpu=lens,
            context_lens_list=context_lens_list(lens),
            dummy_page=dummy,
            max_pages=int(tables.shape[1]),
            impl="paged_atb",
        )

    def _make(self, raw_bs=2, capture_bs=2, topk=2, max_pages=4, dummy=7, had_fm=True):
        src = torch.arange(raw_bs * topk * max_pages, dtype=torch.int32).reshape(
            raw_bs * topk, max_pages
        )
        src_act = torch.ones(raw_bs * topk, dtype=torch.int32)
        dest = torch.full((capture_bs * topk, max_pages), 111, dtype=torch.int32)
        dest_act = torch.full((capture_bs * topk,), 5, dtype=torch.int32)
        eager = self._eager_meta(src, src_act, dummy=dummy)
        rows = int(dest.shape[0])
        pages = int(dest.shape[1])
        inners = []
        for step in range(2):
            inner = NS(
                speculative_step_id=step,
                tree_attention_impl="paged_atb",
                device=dest.device,
                cuda_graph_paged_tables={(rows, pages): dest},
                cuda_graph_paged_actives={(rows, pages): dest_act},
                _sr_tree_paged_meta=eager,
                forward_metadata=_ForwardMetadata() if had_fm else None,
            )
            if had_fm:
                inner.forward_metadata.sr_tree_paged = eager
                inner.forward_metadata.block_tables = eager.block_tables
            inner.bind_sr_tree_paged_metadata = MethodType(
                lambda self, meta: setattr(self, "_sr_tree_paged_meta", meta),
                inner,
            )
            inners.append(inner)
        prefix = torch.zeros(raw_bs, dtype=torch.int32)
        backend = NS(
            topk=topk,
            speculative_num_steps=3,
            attn_backends=inners,
            _tree_replay_raw_bs=raw_bs,
            _paged_round_prefix=prefix,
            _paged_round_tables=src,
            _paged_round_active=src_act,
            _paged_round_dummy=dummy,
            _paged_round_impl="paged_atb",
            paged_impl_selected=lambda: True,
        )
        backend.bind_sr_tree_paged_replay = MethodType(self.bind_fn, backend)
        backend._validate_sr_tree_paged_replay = MethodType(self.validate_fn, backend)
        backend._paged_graph_table_view = MethodType(self.view_fn, backend)
        batch = NS(
            batch_size=raw_bs,
            seq_lens=torch.tensor([1] * raw_bs),
            req_pool_indices=torch.arange(raw_bs),
            positions=torch.arange(raw_bs),
            mrope_positions=None,
            seq_lens_cpu=torch.tensor([8] * raw_bs),
        )
        graph = NS()
        plan = NS(
            graph_key="2_s4",
            raw_bs=raw_bs,
            capture_bs=capture_bs,
            tokens_per_req=topk,
            kv_bucket=max_pages,
        )
        runner = self.Runner()
        runner._tree_paged = True
        runner._tree_shared_prefix = False
        runner._tree_replay_plan = plan
        runner._tree_replay_graph = graph
        runner._tree_replay_batch_id = id(batch)
        runner._tree_replay_stream_idx = None
        runner.graphs = {plan.graph_key: graph}
        runner._tree_attention_impls = {plan.graph_key: "paged_atb"}
        runner._current_tree_attention_impl = lambda: "paged_atb"
        runner.model_runner = NS(draft_attn_backend=backend, attn_backend=None)
        runner._super_replay = Mock(return_value="ok")
        return runner, backend, batch, eager, inners, dest, dest_act, src

    def _assert_eager_restored(self, inners, eager, had_fm=True):
        for inner in inners:
            self.assertIs(inner._sr_tree_paged_meta, eager)
            self.assertEqual(
                list(inner._sr_tree_paged_meta.context_lens_list),
                list(eager.context_lens_list),
            )
            self.assertIs(inner._sr_tree_paged_meta.block_tables, eager.block_tables)
            torch.testing.assert_close(
                inner._sr_tree_paged_meta.block_tables, eager.block_tables
            )
            if had_fm:
                self.assertIs(inner.forward_metadata.sr_tree_paged, eager)
                self.assertIs(
                    inner.forward_metadata.block_tables, eager.block_tables
                )
            else:
                self.assertIsNone(inner.forward_metadata)

    def test_prep_error_after_bind_restores_eager_meta(self):
        runner, backend, batch, eager, inners, dest, _, src = self._make()
        runner._super_replay.side_effect = self.PrepError("after bind", scope="graph")
        with self.assertRaises(self.PrepError):
            runner.replay(batch)
        self._assert_eager_restored(inners, eager)
        self.assertIsNone(runner._tree_replay_plan)
        self.assertIs(inners[0]._sr_tree_paged_meta.block_tables, src)
        self.assertIsNot(inners[0]._sr_tree_paged_meta.block_tables, dest)
        torch.testing.assert_close(dest[: src.shape[0]], src)

    def test_mid_bind_failure_restores_first_step(self):
        runner, backend, batch, eager, inners, dest, _, src = self._make()
        calls = {"n": 0}
        orig = inners[0].bind_sr_tree_paged_metadata

        def boom(meta):
            calls["n"] += 1
            if calls["n"] == 2:
                raise self.PrepError("second step", scope="graph")
            return orig(meta)

        for inner in inners:
            inner.bind_sr_tree_paged_metadata = boom
        with self.assertRaises(self.PrepError):
            runner.replay(batch)
        self.assertEqual(calls["n"], 2)
        self._assert_eager_restored(inners, eager)
        self.assertIsNone(runner._tree_replay_plan)

    def test_raw_one_capture_two_restores_true_rows(self):
        runner, backend, batch, eager, inners, dest, _, src = self._make(
            raw_bs=1, capture_bs=2
        )
        self.assertEqual(batch.batch_size, 1)

        def pad_then_fail(fb):
            fb.batch_size = 2
            fb.seq_lens = torch.tensor([1, 0])
            raise self.PrepError("pad fail", scope="graph")

        runner._super_replay.side_effect = pad_then_fail
        with self.assertRaises(self.PrepError):
            runner.replay(batch)
        self.assertEqual(batch.batch_size, 1)
        self.assertEqual(int(batch.seq_lens.numel()), 1)
        self.assertEqual(int(eager.block_tables.shape[0]), 2)
        self._assert_eager_restored(inners, eager)
        self.assertIs(inners[0]._sr_tree_paged_meta.block_tables, src)
        self.assertIsNone(runner._tree_replay_plan)

    def test_invalid_eager_meta_does_not_reraise_prep_error(self):
        runner, backend, batch, eager, inners, _, _, _ = self._make()
        inners[0]._sr_tree_paged_meta = None
        inners[1]._sr_tree_paged_meta = None
        runner._super_replay.side_effect = self.PrepError("no eager", scope="graph")
        with self.assertRaises(RuntimeError) as caught:
            runner.replay(batch)
        self.assertNotIsInstance(caught.exception, self.PrepError)
        self.assertIsNone(runner._tree_replay_plan)

    def test_submitted_does_not_restore_for_eager(self):
        runner, backend, batch, eager, inners, dest, _, src = self._make()
        restored = []
        runner._restore_paged_eager_metadata = lambda snap: restored.append(snap)
        runner._super_replay.side_effect = self.SubmittedError("submitted")
        with self.assertRaises(self.SubmittedError):
            runner.replay(batch)
        self.assertEqual(restored, [])
        self.assertIsNone(runner._tree_replay_plan)
        self.assertIsNot(inners[0]._sr_tree_paged_meta, eager)

    def test_none_forward_metadata_restored_to_none(self):
        runner, backend, batch, eager, inners, _, _, _ = self._make(had_fm=False)
        runner._super_replay.side_effect = self.PrepError("after bind", scope="graph")
        with self.assertRaises(self.PrepError):
            runner.replay(batch)
        self._assert_eager_restored(inners, eager, had_fm=False)


if __name__ == "__main__":
    unittest.main()
