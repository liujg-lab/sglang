"""CPU contracts for the default-on SR fixed-capacity accept path."""

from __future__ import annotations

import ast
import copy
import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.standalone_remote.drafter.sr_tail_extend import (
    tail_mrope_positions,
)
from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
    copy_paged_kv_buffer_by_slot,
    export_accepted_tree_candidate_indices,
)
from sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept import (
    SR_FIXED_ACCEPT_ENV,
    SRFixedAcceptState,
    accept_control_decision,
    apply_cpu_acceptance,
    apply_free_unique_pages,
    build_fixed_accept_state,
    detach_verify_output,
    first_free_pos,
    free_page_row_offsets,
    length_update_mode,
    multimodal_accept_reject_reason,
    num_free_pages,
    read_sr_fixed_accept_env,
    run_accept_stages,
    static_disable_reason,
    validate_packed_rows,
)
from sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept_kernels import (
    gather_commit_slots,
    pack_accept,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=20, suite="stage-a-test-cpu")

_REPO = pathlib.Path(__file__).resolve().parents[4]
_KERNEL_MODULE = (
    "sglang.srt.speculative.standalone_remote.verifier.sr_fixed_accept_kernels"
)


class NPUPagedTokenToKVPoolAllocator:
    """Name matches the static admission check. Not the production allocator."""

    def __init__(self, page_size=4):
        self.page_size = page_size
        self.is_not_in_free_group = True
        self.need_sort = False
        self.debug_mode = False
        self.free_pages = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        self.release_pages = torch.tensor([], dtype=torch.int64)
        self.kv_buffer = torch.zeros((2, 1, 4, page_size, 1, 1))
        self.freed = []

    def free_unique_pages(self, page_ids):
        self.freed.append(page_ids.detach().clone())
        apply_free_unique_pages(self, page_ids)

    def get_kvcache(self):
        return self


class _Req:
    def __init__(self, finish_at=None, cancel_at=None):
        self.output_ids = []
        self.require_reasoning = False
        self.kv_committed_len = 5
        self.kv_allocated_len = 5
        self.spec_verify_ct = 0
        self.spec_accepted_tokens = 0
        self.histogram = []
        self.finish_at = finish_at
        self.cancel_at = cancel_at
        self.finished_reason = None
        self.sr_accepted_tree_candidate_indices = None

    def finished(self):
        return self.finished_reason is not None

    def check_finished(self):
        if self.finished():
            return
        if self.cancel_at is not None and len(self.output_ids) >= self.cancel_at:
            self.finished_reason = "cancel"
            return
        if self.finish_at is not None and len(self.output_ids) >= self.finish_at:
            self.finished_reason = "stop"

    def update_spec_acceptance_histogram(self, count):
        self.histogram.append(count)


def _worker(page_size=128, device_type="cpu"):
    alloc = NPUPagedTokenToKVPoolAllocator(page_size)
    alloc.kv_buffer = torch.zeros((2, 1, 4, page_size, 1, 1))
    if device_type == "npu":
        alloc.kv_buffer = SimpleNamespace(dim=lambda: 6, shape=(2, 1, 4, page_size, 1, 1), device=SimpleNamespace(type="npu"))
    return SimpleNamespace(
        topk=2,
        page_size=page_size,
        speculative_num_draft_tokens=4,
        speculative_num_steps=3,
        _verify_max_bs=2,
        _hybrid_needs_hidden=False,
        token_to_kv_pool_allocator=alloc,
        device=device_type,
    )


def _admit_kwargs(state, **overrides):
    bs = 2
    logits = torch.zeros((bs * state.W, 3), dtype=torch.float32)
    cache = torch.arange(bs * state.W, dtype=torch.int64)
    seq = torch.tensor([4, 4], dtype=torch.int64)
    values = dict(
        bs=bs,
        verify_mode="greedy",
        is_all_greedy=True,
        has_grammar=False,
        vocab_mask=None,
        return_logprob=False,
        prepare_hidden=False,
        has_custom_logit_processor=False,
        multimodal_reject_reason=None,
        simulate_acc_len=0.0,
        sampling_rows=bs,
        seq_lens_cpu=seq,
        allocator=SimpleNamespace(is_not_in_free_group=True),
        draft_token_num=state.W,
        spec_steps=state.L - 1,
        logits=logits,
        out_cache_loc=cache,
    )
    values.update(overrides)
    return values


class FixedAcceptEnvTest(CustomTestCase):
    def test_unset_and_explicit_one_match_and_zero_is_off(self):
        self.assertTrue(read_sr_fixed_accept_env({}))
        self.assertTrue(read_sr_fixed_accept_env({SR_FIXED_ACCEPT_ENV: "1"}))
        self.assertTrue(read_sr_fixed_accept_env({SR_FIXED_ACCEPT_ENV: "true"}))
        self.assertTrue(read_sr_fixed_accept_env({SR_FIXED_ACCEPT_ENV: "YES"}))
        self.assertTrue(read_sr_fixed_accept_env({SR_FIXED_ACCEPT_ENV: "on"}))
        self.assertFalse(read_sr_fixed_accept_env({SR_FIXED_ACCEPT_ENV: "0"}))
        self.assertFalse(read_sr_fixed_accept_env({SR_FIXED_ACCEPT_ENV: "false"}))
        self.assertFalse(read_sr_fixed_accept_env({SR_FIXED_ACCEPT_ENV: "no"}))
        self.assertFalse(read_sr_fixed_accept_env({SR_FIXED_ACCEPT_ENV: "off"}))
        for enabled in (read_sr_fixed_accept_env({}), read_sr_fixed_accept_env({SR_FIXED_ACCEPT_ENV: "1"})):
            self.assertEqual(
                accept_control_decision(enabled, "greedy", False), "finalizer"
            )
        self.assertEqual(
            accept_control_decision(read_sr_fixed_accept_env({SR_FIXED_ACCEPT_ENV: "0"}), "greedy", False),
            "fresh_v1",
        )

    def test_env_off_and_static_reject_skip_workspace_and_kernel_import(self):
        sys.modules.pop(_KERNEL_MODULE, None)
        state = build_fixed_accept_state(
            _worker(), env={SR_FIXED_ACCEPT_ENV: "0"}
        )
        self.assertIsNone(state)
        self.assertNotIn(_KERNEL_MODULE, sys.modules)

        sys.modules.pop(_KERNEL_MODULE, None)
        state = build_fixed_accept_state(_worker(), env={})
        self.assertIsNone(state)
        self.assertEqual(static_disable_reason(_worker()), "not npu")
        self.assertNotIn(_KERNEL_MODULE, sys.modules)

        missing = _worker()
        missing.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=128, is_not_in_free_group=True, get_kvcache=lambda: None
        )
        self.assertEqual(
            static_disable_reason(missing), "allocator missing free_unique_pages"
        )
        mismatch = _worker(page_size=128)
        mismatch.page_size = 64
        self.assertEqual(static_disable_reason(mismatch), "page size mismatch")
        too_long = _worker()
        too_long.speculative_num_steps = 4
        self.assertEqual(static_disable_reason(too_long), "capacity")

    def test_warmup_scratch_resets_private_buffers(self):
        state = SRFixedAcceptState(2, 3, 4, 4, "cpu")
        state.warmup_scratch()
        self.assertTrue(torch.all(state.accept_index == -1))
        self.assertTrue(torch.all(state.accept_length == 0))
        self.assertTrue(torch.all(state.predict == 0))


class FixedAcceptRouteTest(CustomTestCase):
    def test_two_exits_and_defensive_branch_runs_each_stage_once(self):
        self.assertEqual(accept_control_decision(False, "greedy", False), "fresh_v1")
        self.assertEqual(accept_control_decision(True, "target_only", False), "v1_workspace")
        self.assertEqual(accept_control_decision(True, "rpd", False), "v1_workspace")
        calls = []

        def penalty():
            calls.append("penalty")

        def verify():
            calls.append("verify")
            return "target_only"

        def v1():
            calls.append("append")
            return "v1"

        def finalizer():
            calls.append("finalizer")
            return "fast"

        self.assertEqual(
            run_accept_stages(
                admitted=True,
                simulate=False,
                penalty=penalty,
                verify=verify,
                v1=v1,
                finalizer=finalizer,
            ),
            "v1",
        )
        self.assertEqual(calls, ["penalty", "verify", "append"])

    def test_pre_alloc_reject_reasons(self):
        state = SRFixedAcceptState(2, 4, 4, 4, "cpu")
        self.assertIsNone(state.reject_before_alloc(**_admit_kwargs(state)))
        self.assertEqual(
            state.reject_before_alloc(**_admit_kwargs(state, verify_mode="rpd")),
            "mode",
        )
        self.assertEqual(
            state.reject_before_alloc(**_admit_kwargs(state, verify_mode="target_only")),
            "mode",
        )
        self.assertEqual(
            state.reject_before_alloc(
                **_admit_kwargs(state, verify_mode="auto", is_all_greedy=False)
            ),
            "mode",
        )
        self.assertIsNone(
            state.reject_before_alloc(
                **_admit_kwargs(state, verify_mode="auto", is_all_greedy=True)
            )
        )
        for key, value, reason in (
            ("has_grammar", True, "grammar"),
            ("return_logprob", True, "logprob"),
            ("prepare_hidden", True, "hidden"),
            ("has_custom_logit_processor", True, "logit_processor"),
            ("multimodal_reject_reason", "multimodal_model", "multimodal_model"),
            ("simulate_acc_len", 1.0, "simulate"),
            ("sampling_rows", 1, "sampling_rows"),
            ("draft_token_num", 3, "width"),
            ("spec_steps", 9, "path_cap"),
        ):
            self.assertEqual(
                state.reject_before_alloc(**_admit_kwargs(state, **{key: value})),
                reason,
                key,
            )
        wide_logits = torch.zeros((3 * state.W, 3), dtype=torch.float32)
        wide_cache = torch.arange(3 * state.W, dtype=torch.int64)
        self.assertEqual(
            state.reject_before_alloc(
                **_admit_kwargs(
                    state,
                    bs=3,
                    sampling_rows=3,
                    seq_lens_cpu=torch.tensor([1, 1, 1]),
                    logits=wide_logits,
                    out_cache_loc=wide_cache,
                )
            ),
            "batch_cap",
        )
        self.assertEqual(
            state.reject_before_alloc(
                **_admit_kwargs(state, out_cache_loc=torch.arange(7, dtype=torch.int64))
            ),
            "cache_loc",
        )
        wide = torch.arange(16, dtype=torch.int64)
        sliced = wide[::2]
        self.assertFalse(sliced.is_contiguous())
        self.assertEqual(int(sliced.numel()), 8)
        self.assertEqual(
            state.reject_before_alloc(**_admit_kwargs(state, out_cache_loc=sliced)),
            "noncontiguous",
        )
        predict, accept_index, accept_length = state.bind_verify_buffers(2)
        self.assertTrue(predict.is_contiguous())
        self.assertTrue(accept_index.is_contiguous())
        self.assertTrue(accept_length.is_contiguous())
        self.assertEqual(tuple(accept_index.shape), (2, 4))
        self.assertEqual(int(predict.numel()), 2 * 4 + 1)


class FixedAcceptPageTest(CustomTestCase):
    def test_integer_page_formula_and_zero_release_does_not_offer_a_slot(self):
        self.assertEqual(num_free_pages(100, 3, 15, 128), 0)
        self.assertEqual(first_free_pos(103, 128), 128)
        self.assertEqual(free_page_row_offsets(100, 3, 15, 128), [])
        # extended_end=115 sits inside the page; the representative slot would be 28.
        self.assertNotIn(28, free_page_row_offsets(100, 3, 15, 128))

        offsets = free_page_row_offsets(100, 40, 200, 128)
        self.assertEqual(offsets, [156])
        self.assertEqual(num_free_pages(0, 10, 300, 128), 2)
        self.assertEqual(free_page_row_offsets(0, 10, 300, 128), [128, 256])

        prefix_page = 3
        row = torch.full((200,), prefix_page * 128, dtype=torch.int64)
        row[156] = 9 * 128 + 3
        pages = [int(row[off]) // 128 for off in offsets]
        self.assertEqual(pages, [9])
        self.assertNotIn(prefix_page, pages)

    def test_length_modes_are_not_the_unfinished_filter(self):
        self.assertEqual(length_update_mode([False, False]), "all")
        self.assertEqual(length_update_mode([True, False]), "all")
        self.assertEqual(length_update_mode([True, True]), "none")
        self.assertEqual(length_update_mode([]), "none")


class FixedAcceptCpuLoopTest(CustomTestCase):
    def test_finish_positions_and_raw_histogram(self):
        rows = [[0, 5, 2, -1], [1, -1, -1, -1], [3, 4, 6, 7]]
        tokens = [[10, 11, 12, 0], [20, 0, 0, 0], [30, 31, 32, 33]]
        reqs = [_Req(finish_at=1), _Req(), _Req(finish_at=4)]
        accepted, finished, truncated = apply_cpu_acceptance(reqs, rows, tokens, None)
        self.assertEqual(accepted, [1, 1, 4])
        self.assertEqual(finished, [True, False, True])
        self.assertEqual(truncated[0], [0, -1, -1, -1])
        self.assertEqual(reqs[0].output_ids, [10])
        self.assertEqual(reqs[0].histogram, [2])
        self.assertEqual(reqs[1].output_ids, [20])
        self.assertEqual(reqs[1].histogram, [0])
        self.assertEqual(reqs[2].output_ids, [30, 31, 32, 33])
        self.assertEqual(reqs[0].kv_committed_len, 6)
        self.assertEqual(reqs[0].spec_verify_ct, 1)

        cancel = [_Req(cancel_at=2), _Req(finish_at=3)]
        accepted, finished, _ = apply_cpu_acceptance(
            cancel,
            [[1, 2, 3, -1], [4, 5, 6, -1]],
            [[1, 2, 3, 0], [4, 5, 6, 0]],
            None,
        )
        self.assertEqual(accepted, [2, 3])
        self.assertEqual(finished, [True, True])
        self.assertEqual(cancel[0].finished_reason, "cancel")

    def test_validate_rejects_the_batch_before_any_append(self):
        with self.assertRaises(RuntimeError):
            validate_packed_rows([[0, -1], [5, 1]], [0, 1], [0, 1])
        with self.assertRaises(RuntimeError):
            validate_packed_rows([[0, 1, -1, 2]], [2], [0])
        validate_packed_rows([[0, 2, -1], [-1, -1, -1]], [1, 0], [0, 0])


class FixedAcceptFinalizeTest(CustomTestCase):
    def _batch(self, reqs, prefixes, cache, kv):
        bs = len(reqs)
        return SimpleNamespace(
            reqs=reqs,
            seq_lens=torch.tensor(prefixes, dtype=torch.int64),
            seq_lens_cpu=torch.tensor(prefixes, dtype=torch.int64),
            out_cache_loc=cache.clone(),
            req_pool_indices=torch.arange(bs, dtype=torch.int64),
            device=torch.device("cpu"),
            topk=2,
            spec_algorithm=SimpleNamespace(is_standalone_remote=lambda: True),
            model_config=SimpleNamespace(
                think_end_id=None, hidden_size=4, dtype=torch.float32
            ),
        ), kv

    def _run(self, rows, tokens, pre_lengths, prefixes, finish_at, cache=None):
        bs = len(rows)
        path = max(len(row) for row in rows)
        width = 4
        state = SRFixedAcceptState(max(bs, 1), path, width, 4, "cpu")
        predict, accept_index, accept_length = state.bind_verify_buffers(bs)
        predict.zero_()
        accept_index.fill_(-1)
        for i, row in enumerate(rows):
            for j, idx in enumerate(row):
                accept_index[i, j] = idx
                if idx >= 0:
                    predict[idx] = tokens[i][j]
            accept_length[i] = pre_lengths[i]
        if cache is None:
            cache = torch.arange(bs * width, dtype=torch.int64)
        pages = int(cache.max()) // 4 + 2
        kv = torch.arange(2 * pages * 4, dtype=torch.float32).reshape(2, 1, pages, 4, 1, 1)
        alloc = NPUPagedTokenToKVPoolAllocator(4)
        alloc.kv_buffer = kv
        reqs = [_Req(finish_at=item) for item in finish_at]
        batch, kv = self._batch(reqs, prefixes, cache, kv)
        logits = SimpleNamespace()
        result = state.finalize(
            batch, logits, 4, 2, alloc, accept_index, predict, accept_length
        )
        return state, batch, alloc, reqs, result, kv

    def test_continue_partial_and_all_finished_lengths(self):
        _, batch, _, reqs, result, _ = self._run(
            rows=[[0, 1, -1, -1], [4, -1, -1, -1]],
            tokens=[[10, 11, 0, 0], [20, 0, 0, 0]],
            pre_lengths=[1, 0],
            prefixes=[1, 1],
            finish_at=[None, None],
        )
        self.assertEqual(result.accept_length_per_req_cpu, [1, 0])
        self.assertEqual(batch.seq_lens.tolist(), [3, 2])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [3, 2])
        self.assertFalse(result.idle)
        self.assertEqual(result.draft_accept_length.tolist(), [1, 0])
        self.assertEqual(reqs[0].output_ids, [10, 11])

        _, batch, _, _, result, _ = self._run(
            rows=[[0, 1, -1, -1], [4, 5, -1, -1]],
            tokens=[[10, 11, 0, 0], [20, 21, 0, 0]],
            pre_lengths=[1, 1],
            prefixes=[1, 1],
            finish_at=[1, None],
        )
        self.assertEqual(length_update_mode([True, False]), "all")
        self.assertEqual(batch.seq_lens.tolist(), [2, 3])
        self.assertEqual(result.accept_length_per_req_cpu, [0, 1])
        self.assertEqual(result.draft_accept_length_cpu, [1])
        self.assertEqual(result.req_pool_indices.tolist(), [1])

        _, batch, _, reqs, result, _ = self._run(
            rows=[[0, -1, -1, -1], [4, -1, -1, -1]],
            tokens=[[10, 0, 0, 0], [20, 0, 0, 0]],
            pre_lengths=[0, 0],
            prefixes=[8, 9],
            finish_at=[1, 1],
        )
        self.assertTrue(result.idle)
        self.assertEqual(batch.seq_lens.tolist(), [8, 9])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [8, 9])
        self.assertEqual(result.accept_length_per_req_cpu, [0, 0])
        self.assertEqual(
            [pre + (length + 1) for pre, length in zip([8, 9], result.accept_length_per_req_cpu)],
            [9, 10],
        )
        self.assertEqual(reqs[0].kv_committed_len, 6)

    def test_paths_pages_and_overlap_move(self):
        state, batch, alloc, reqs, result, kv = self._run(
            rows=[[0, 2, -1, -1], [4, -1, -1, -1]],
            tokens=[[10, 12, 0, 0], [20, 0, 0, 0]],
            pre_lengths=[1, 0],
            prefixes=[0, 0],
            finish_at=[None, None],
        )
        self.assertEqual(
            reqs[0].sr_accepted_tree_candidate_indices,
            export_accepted_tree_candidate_indices([[0, 2, -1, -1]])[0],
        )
        self.assertEqual(
            result.tree_paths[0],
            reqs[0].sr_accepted_tree_candidate_indices,
        )
        reqs[0].sr_accepted_tree_candidate_indices = [99]
        batch.spec_algorithm = None
        state._export_paths(batch, [[0, 2, -1, -1], [4, -1, -1, -1]])
        self.assertEqual(state._tree_paths, [])
        self.assertEqual(reqs[0].sr_accepted_tree_candidate_indices, [99])
        self.assertEqual(result.verified_id.tolist(), [10, 12, 20])
        self.assertTrue(alloc.freed)
        flat = kv.view(2, 1, -1, 1, 1)
        src_before = [0, 2, 4]
        # Destination is the row prefix of length A: [0, 1] and [4].
        self.assertEqual(int(flat[0, 0, 1, 0, 0]), int(torch.arange(flat.shape[2])[2]))

        overlap = torch.zeros((2, 1, 2, 4, 1, 1))
        token = overlap.view(2, 1, -1, 1, 1)
        token[0, 0, :, 0, 0] = torch.arange(8)
        copy_paged_kv_buffer_by_slot(
            overlap,
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([1, 2], dtype=torch.int64),
        )
        self.assertEqual(token[0, 0, :4, 0, 0].tolist(), [0, 0, 1, 3])

    def test_zero_page_gather_and_batch_error_appends_nothing(self):
        state = SRFixedAcceptState(1, 4, 15, 128, "cpu")
        predict, accept_index, accept_length = state.bind_verify_buffers(1)
        accept_index.fill_(-1)
        accept_index[0, 0] = 0
        accept_index[0, 1] = 1
        accept_index[0, 2] = 2
        accept_length[0] = 2
        predict.zero_()
        predict[0] = 1
        predict[1] = 2
        predict[2] = 3
        cache = torch.arange(15, dtype=torch.int64)
        reads = []
        original = cache.index_select

        def _select(dim, index):
            reads.append(index.detach().cpu().tolist())
            return original(dim, index)

        cache.index_select = _select
        src_buf = torch.empty((8,), dtype=torch.int64)
        tgt_buf = torch.empty((8,), dtype=torch.int64)
        page_buf = torch.empty((4,), dtype=torch.int64)
        _, _, pages = gather_commit_slots(
            cache,
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 1, 2]),
            torch.tensor([], dtype=torch.int64),
            128,
            src_buf,
            tgt_buf,
            page_buf,
        )
        self.assertEqual(pages.numel(), 0)
        self.assertEqual(src_buf.device, cache.device)
        with self.assertRaises(RuntimeError):
            gather_commit_slots(
                cache,
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([0, 1, 2]),
                torch.tensor([], dtype=torch.int64),
                128,
                src_buf,
                tgt_buf,
                page_buf,
            )
        self.assertEqual(free_page_row_offsets(100, 3, 15, 128), [])
        self.assertTrue(all(28 not in group for group in reads))

        accept_index[0, 1] = 10**6
        alloc = NPUPagedTokenToKVPoolAllocator(128)
        pages_n = 4
        alloc.kv_buffer = torch.zeros((2, 1, pages_n, 128, 1, 1))
        reqs = [_Req(), _Req()]
        # Rebuild a 2-request error on the second row only.
        state = SRFixedAcceptState(2, 3, 4, 4, "cpu")
        predict, accept_index, accept_length = state.bind_verify_buffers(2)
        predict.zero_()
        accept_index.fill_(-1)
        accept_index[0, 0] = 0
        accept_index[1, 0] = 10**6
        accept_length[0] = 0
        accept_length[1] = 0
        predict[0] = 7
        batch = SimpleNamespace(
            reqs=reqs,
            seq_lens=torch.tensor([1, 1]),
            seq_lens_cpu=torch.tensor([1, 1]),
            out_cache_loc=torch.arange(8),
            req_pool_indices=torch.arange(2),
            device=torch.device("cpu"),
            spec_algorithm=None,
            model_config=SimpleNamespace(think_end_id=None, hidden_size=1, dtype=torch.float32),
        )
        with self.assertRaises(RuntimeError):
            state.finalize(batch, None, 4, 2, alloc, accept_index, predict, accept_length)
        self.assertEqual(reqs[0].output_ids, [])
        self.assertEqual(reqs[1].output_ids, [])
        self.assertEqual(alloc.freed, [])

    def test_failures_do_not_append_twice_or_free_early(self):
        def once(inject):
            state = SRFixedAcceptState(1, 2, 4, 4, "cpu")
            predict, accept_index, accept_length = state.bind_verify_buffers(1)
            accept_index.fill_(-1)
            accept_index[0, 0] = 0
            accept_length[0] = 0
            predict.zero_()
            predict[0] = 5
            alloc = NPUPagedTokenToKVPoolAllocator(4)
            alloc.kv_buffer = torch.zeros((2, 1, 2, 4, 1, 1))
            req = _Req()
            batch = SimpleNamespace(
                reqs=[req],
                seq_lens=torch.tensor([1]),
                seq_lens_cpu=torch.tensor([1]),
                out_cache_loc=torch.arange(4),
                req_pool_indices=torch.tensor([0]),
                device=torch.device("cpu"),
                spec_algorithm=None,
                model_config=SimpleNamespace(think_end_id=None, hidden_size=1, dtype=torch.float32),
            )
            state.inject_error = inject
            with self.assertRaises(RuntimeError):
                state.finalize(
                    batch, None, 4, 2, alloc, accept_index, predict, accept_length
                )
            return req, alloc

        for inject in ("pack", "readback", "d2h_submit", "d2h_wait", "h2d_reuse_wait"):
            req, alloc = once(inject)
            self.assertEqual(req.output_ids, [])
            self.assertEqual(alloc.freed, [])
        for inject in ("move", "h2d_submit"):
            req, alloc = once(inject)
            self.assertEqual(req.output_ids, [5])
            self.assertEqual(alloc.freed, [])
            self.assertEqual(req.spec_verify_ct, 1)

    def test_two_rounds_do_not_overwrite_returned_storage(self):
        def run(token):
            state = shared["state"]
            predict, accept_index, accept_length = state.bind_verify_buffers(1)
            accept_index.fill_(-1)
            accept_index[0, 0] = 0
            accept_length[0] = 0
            predict.zero_()
            predict[0] = token
            alloc = NPUPagedTokenToKVPoolAllocator(4)
            alloc.kv_buffer = torch.zeros((2, 1, 2, 4, 1, 1))
            batch = SimpleNamespace(
                reqs=[_Req()],
                seq_lens=torch.tensor([1]),
                seq_lens_cpu=torch.tensor([1]),
                out_cache_loc=torch.arange(4),
                req_pool_indices=torch.tensor([3]),
                device=torch.device("cpu"),
                spec_algorithm=None,
                model_config=SimpleNamespace(think_end_id=None, hidden_size=1, dtype=torch.float32),
            )
            result = state.finalize(
                batch, None, 4, 2, alloc, accept_index, predict, accept_length
            )
            return result

        shared = {"state": SRFixedAcceptState(1, 2, 4, 4, "cpu")}
        first = run(11)
        second = run(99)
        self.assertEqual(first.verified_id.tolist(), [11])
        self.assertEqual(first.accepted_indices.tolist(), [0])
        self.assertEqual(first.draft_accept_length.tolist(), [0])
        self.assertEqual(first.seq_lens_for_draft.tolist(), [2])
        self.assertEqual(first.req_pool_indices.tolist(), [3])
        self.assertEqual(second.verified_id.tolist(), [99])

        workspace = shared["state"].accept_length
        workspace.fill_(0)
        workspace[0] = 7
        draft = SimpleNamespace(
            verified_id=workspace[:1],
            accept_length=workspace[:1],
            seq_lens_for_draft_extend=workspace[:1],
            seq_lens_for_draft_extend_cpu=workspace[:1].clone(),
            req_pool_indices_for_draft_extend=workspace[:1].clone(),
        )
        output = SimpleNamespace(
            verified_id=workspace[:1],
            accepted_indices=workspace[:1],
            draft_input=draft,
        )
        detach_verify_output(output)
        workspace.fill_(123)
        self.assertEqual(int(output.verified_id[0]), 7)
        self.assertEqual(int(output.draft_input.accept_length[0]), 7)
        self.assertEqual(int(output.draft_input.seq_lens_for_draft_extend[0]), 7)
        self.assertEqual(int(output.draft_input.req_pool_indices_for_draft_extend[0]), 7)

    def test_page_capacity_matches_new_bound(self):
        state = SRFixedAcceptState(2, 4, 128, 128, "cpu")
        self.assertEqual(state.F_cap, 4)
        self.assertEqual(state.page_buf.numel(), 4)
        self.assertEqual(state.N_cap, 8)
        self.assertEqual(state.packet_cap, 3 * 8 + 4 + 4)

    def test_rounds_keep_storage_when_batch_shrinks_and_grows(self):
        state = SRFixedAcceptState(2, 2, 4, 4, "cpu")
        held = []

        def run(bs, token):
            predict, accept_index, accept_length = state.bind_verify_buffers(bs)
            accept_index.fill_(-1)
            accept_length.zero_()
            predict.zero_()
            for i in range(bs):
                accept_index[i, 0] = i * 4
                predict[i * 4] = token + i
            alloc = NPUPagedTokenToKVPoolAllocator(4)
            alloc.kv_buffer = torch.zeros((2, 1, 4, 4, 1, 1))
            batch = SimpleNamespace(
                reqs=[_Req() for _ in range(bs)],
                seq_lens=torch.ones(bs, dtype=torch.int64),
                seq_lens_cpu=torch.ones(bs, dtype=torch.int64),
                out_cache_loc=torch.arange(bs * 4),
                req_pool_indices=torch.arange(bs),
                device=torch.device("cpu"),
                spec_algorithm=None,
                model_config=SimpleNamespace(
                    think_end_id=None, hidden_size=1, dtype=torch.float32
                ),
            )
            result = state.finalize(
                batch, None, 4, 2, alloc, accept_index, predict, accept_length
            )
            held.append((result, batch.out_cache_loc.clone(), [r.output_ids[:] for r in batch.reqs]))
            return result

        run(2, 10)
        run(1, 30)
        run(2, 50)
        self.assertEqual(state._d2h_count, 3)
        self.assertEqual(state._h2d_count, 3)
        state.commit_device.fill_(-1)
        state.commit_host.fill_(-1)
        state.accept_host.fill_(-1)
        self.assertEqual(held[0][0].verified_id.tolist(), [10, 11])
        self.assertEqual(held[0][0].accepted_indices.tolist(), [0, 4])
        self.assertEqual(held[0][1].tolist(), [0, 4])
        self.assertEqual(held[1][0].verified_id.tolist(), [30])
        self.assertEqual(held[2][0].verified_id.tolist(), [50, 51])
        self.assertEqual(held[0][2], [[10], [11]])

    def test_empty_batch_submits_no_transfer(self):
        state = SRFixedAcceptState(1, 2, 4, 4, "cpu")
        predict, accept_index, accept_length = state.bind_verify_buffers(0)
        alloc = NPUPagedTokenToKVPoolAllocator(4)
        batch = SimpleNamespace(
            reqs=[],
            seq_lens=torch.zeros(0, dtype=torch.int64),
            seq_lens_cpu=torch.zeros(0, dtype=torch.int64),
            out_cache_loc=torch.arange(4),
            req_pool_indices=torch.zeros(0, dtype=torch.int64),
            device=torch.device("cpu"),
            spec_algorithm=None,
            model_config=SimpleNamespace(
                think_end_id=None, hidden_size=1, dtype=torch.float32
            ),
        )
        result = state.finalize(
            batch, None, 4, 2, alloc, accept_index, predict, accept_length
        )
        self.assertEqual(state._d2h_count, 0)
        self.assertEqual(state._h2d_count, 0)
        self.assertEqual(result.verified_id.numel(), 0)
        self.assertTrue(result.idle)

    def test_publish_runs_before_kv_move(self):
        state = SRFixedAcceptState(1, 2, 4, 4, "cpu")
        state.control_buf.fill_(-7)
        predict, accept_index, accept_length = state.bind_verify_buffers(1)
        accept_index.fill_(-1)
        accept_index[0, 0] = 0
        accept_index[0, 1] = 1
        accept_length[0] = 1
        predict.zero_()
        predict[0] = 10
        predict[1] = 11
        alloc = NPUPagedTokenToKVPoolAllocator(4)
        alloc.kv_buffer = torch.zeros((2, 1, 2, 4, 1, 1))
        batch = SimpleNamespace(
            reqs=[_Req()],
            seq_lens=torch.tensor([1]),
            seq_lens_cpu=torch.tensor([1]),
            out_cache_loc=torch.arange(4),
            req_pool_indices=torch.tensor([3]),
            device=torch.device("cpu"),
            spec_algorithm=None,
            model_config=SimpleNamespace(think_end_id=None, hidden_size=1, dtype=torch.float32),
        )
        seen = {}

        def wrapped(kv_buffer, src, tgt):
            seen["seq"] = batch.seq_lens.tolist()
            seen["cache"] = batch.out_cache_loc.tolist()
            return copy_paged_kv_buffer_by_slot(kv_buffer, src, tgt)

        with unittest.mock.patch(
            "sglang.srt.speculative.standalone_remote.sr_verify_layout.copy_paged_kv_buffer_by_slot",
            wrapped,
        ):
            result = state.finalize(
                batch, None, 4, 2, alloc, accept_index, predict, accept_length
            )
        self.assertEqual(seen["seq"], [3])
        self.assertEqual(seen["cache"], [0, 1])
        self.assertEqual(result.verified_id.tolist(), [10, 11])
        state.src_buf.fill_(123)
        self.assertEqual(result.verified_id.tolist(), [10, 11])
        self.assertTrue(torch.equal(state.control_buf, torch.full_like(state.control_buf, -7)))

    def test_pack_masks_invalid_indexes(self):
        accept_index = torch.tensor([[0, -1], [4, 1]], dtype=torch.int32)
        predict = torch.tensor([8, 9, 0, 0], dtype=torch.int32)
        accept_length = torch.tensor([0, 1], dtype=torch.int32)
        out = torch.empty((2, 6), dtype=torch.int64)
        pack_accept(accept_index, predict, accept_length, out)
        self.assertEqual(out[0, 2:4].tolist(), [8, 0])
        self.assertEqual(int(out[0, 5]), 0)
        self.assertEqual(int(out[1, 5]), 1)


class FixedAcceptSourceTest(CustomTestCase):
    def test_verify_keeps_the_two_exits_and_allocator_does_not_read_back(self):
        eagle = (_REPO / "python/sglang/srt/speculative/eagle_info.py").read_text()
        worker = (
            _REPO / "python/sglang/srt/speculative/standalone_remote/verifier/sr_worker.py"
        ).read_text()
        alloc = (
            _REPO / "python/sglang/srt/hardware_backend/npu/allocator_npu.py"
        ).read_text()
        self.assertIn("sr_accept_state=None", eagle)
        self.assertIn("accept_control_decision", eagle)
        self.assertIn("multimodal_accept_reject_reason", eagle)
        self.assertNotIn("has_multimodal=", eagle)
        self.assertIn("sr_accept_state=fixed_state", worker)
        self.assertEqual(eagle.count("penalizer_orchestrator.apply"), 1)
        method = alloc.split("def free_unique_pages", 1)[1].split("def ", 1)[0]
        self.assertNotIn(".cpu(", method)
        self.assertNotIn("torch.unique", method)
        self.assertIn("apply_free_unique_pages", method)
        self.assertIn("def free(", alloc)

        holder = SimpleNamespace(
            is_not_in_free_group=True,
            need_sort=False,
            debug_mode=False,
            free_pages=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            release_pages=torch.empty((0,), dtype=torch.int64),
        )
        apply_free_unique_pages(holder, torch.tensor([8, 5], dtype=torch.int64))
        self.assertEqual(holder.free_pages.tolist(), [5, 8, 1, 2, 3, 4])
        holder.need_sort = True
        before = holder.free_pages.clone()
        apply_free_unique_pages(holder, torch.tensor([9, 6], dtype=torch.int64))
        self.assertEqual(holder.release_pages.tolist(), [6, 9])
        self.assertEqual(holder.free_pages.tolist(), before.tolist())
        free_before = holder.free_pages.clone()
        release_before = holder.release_pages.clone()
        apply_free_unique_pages(holder, torch.empty((0,), dtype=torch.int64))
        self.assertEqual(holder.free_pages.tolist(), free_before.tolist())
        self.assertEqual(holder.release_pages.tolist(), release_before.tolist())
        holder.is_not_in_free_group = False
        with self.assertRaises(RuntimeError):
            apply_free_unique_pages(holder, torch.tensor([1], dtype=torch.int64))
        self.assertEqual(holder.free_pages.tolist(), free_before.tolist())


class _NoValueDelta(torch.Tensor):
    """Integer ``[1, 1]`` delta whose numeric reads fail. Shape and dtype stay usable."""

    @classmethod
    def wrap(cls, value: int):
        return torch.tensor([[int(value)]], dtype=torch.int64).as_subclass(cls)

    def cpu(self, *args, **kwargs):
        raise RuntimeError("delta device read")

    def item(self):
        raise RuntimeError("delta device read")

    def __int__(self):
        raise RuntimeError("delta device read")

    def tolist(self):
        raise RuntimeError("delta device read")


class _PathCounter(dict):
    def __missing__(self, key):
        return 0


def _verify_mode():
    return SimpleNamespace(is_target_verify=lambda: True)


def _qwen_batch(reqs, rows, seq, arch="Qwen3VLForConditionalGeneration"):
    return SimpleNamespace(
        reqs=reqs,
        multimodal_inputs=rows,
        seq_lens_cpu=seq,
        forward_mode=_verify_mode(),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[arch]),
            think_end_id=None,
            hidden_size=4,
            dtype=torch.float32,
        ),
    )


def _mm_input(delta, modality, *, features=False):
    item = SimpleNamespace(
        modality=modality,
        feature=torch.ones(2, 3) if features else None,
        precomputed_embeddings=torch.ones(2, 4) if features else None,
    )
    return SimpleNamespace(
        modality=modality,
        mrope_position_delta=delta,
        mrope_positions=torch.arange(12, dtype=torch.int64).reshape(3, 4),
        mm_items=[item],
    )


def _load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        copy.deepcopy(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != set(names):
        raise AssertionError("missing production methods")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future] + functions, type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


class FixedAcceptMultimodalAdmitTest(CustomTestCase):
    def test_qwen3_vl_image_video_text_and_mixed_batches_pass(self):
        state = SRFixedAcceptState(3, 4, 4, 4, "cpu")
        state.accept_index.fill_(3)
        for arch in (
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
        ):
            for value in (0, 4, -3):
                image = _mm_input(_NoValueDelta.wrap(value), "image", features=True)
                video = _mm_input(_NoValueDelta.wrap(value), "video", features=True)
                text = _Req()
                text.origin_input_ids = [9, 9, 9, 9, 9]
                text.multimodal_inputs = None
                image_req = _Req()
                image_req.origin_input_ids = [1, 2, 3, 4]
                image_req.multimodal_inputs = image
                video_req = _Req()
                video_req.origin_input_ids = [1, 2, 3]
                video_req.multimodal_inputs = video
                reqs = [text, image_req, video_req]
                rows = [None, image, video]
                # Text row is still inside prefill. Multimodal rows are past the padded prompt.
                seq = torch.tensor([0, 4, 8], dtype=torch.int32)
                batch = _qwen_batch(reqs, rows, seq, arch)
                before = (
                    id(image),
                    id(video),
                    id(image_req.origin_input_ids),
                    id(image.mrope_positions),
                    id(image.mm_items[0].feature),
                )
                self.assertIsNone(multimodal_accept_reject_reason(batch, bs=3))
                self.assertIsNone(
                    state.reject_before_alloc(
                        **_admit_kwargs(
                            state,
                            bs=3,
                            sampling_rows=3,
                            seq_lens_cpu=seq,
                            logits=torch.zeros((3 * state.W, 3)),
                            out_cache_loc=torch.arange(3 * state.W),
                            multimodal_reject_reason=None,
                        )
                    )
                )
                self.assertEqual(
                    (
                        id(image),
                        id(video),
                        id(image_req.origin_input_ids),
                        id(image.mrope_positions),
                        id(image.mm_items[0].feature),
                    ),
                    before,
                )
                self.assertTrue(torch.all(state.accept_index == 3))

        text_only = [_Req(), _Req()]
        self.assertIsNone(
            multimodal_accept_reject_reason(
                SimpleNamespace(reqs=text_only), bs=2
            )
        )

    def test_rejections_leave_requests_and_workspace_unchanged(self):
        state = SRFixedAcceptState(2, 4, 4, 4, "cpu")
        state.accept_index.fill_(3)
        image = _mm_input(_NoValueDelta.wrap(2), "image", features=True)
        other = _mm_input(torch.tensor([[2]]), "image")
        req = _Req()
        req.origin_input_ids = [1, 2, 3, 4]
        req.multimodal_inputs = image
        text = _Req()
        text.multimodal_inputs = None
        text.origin_input_ids = [7]
        seq = torch.tensor([4, 1], dtype=torch.int64)

        def reject(batch, bs, reason):
            self.assertEqual(multimodal_accept_reject_reason(batch, bs=bs), reason)
            self.assertEqual(
                state.reject_before_alloc(
                    **_admit_kwargs(state, multimodal_reject_reason=reason)
                ),
                reason,
            )

        bare = _qwen_batch([req], [image], seq[:1])
        bare.model_config.hf_config.architectures = ["Qwen2VLForConditionalGeneration"]
        reject(bare, 1, "multimodal_model")
        bare.model_config.hf_config.architectures = []
        reject(bare, 1, "multimodal_model")
        bare.model_config = SimpleNamespace(hf_config=None)
        reject(bare, 1, "multimodal_model")

        phased = _qwen_batch([req], [image], seq[:1])
        phased.forward_mode = SimpleNamespace(is_target_verify=lambda: False)
        reject(phased, 1, "multimodal_phase")
        phased.forward_mode = None
        reject(phased, 1, "multimodal_phase")

        short = _qwen_batch([req], [image], seq[:1])
        short.multimodal_inputs = [other]
        reject(short, 1, "multimodal_rows")
        short.multimodal_inputs = [image, image]
        reject(short, 1, "multimodal_rows")
        short.multimodal_inputs = None
        reject(short, 1, "multimodal_rows")

        prefix = _qwen_batch([req], [image], seq[:1])
        prefix.seq_lens_cpu = torch.zeros((1, 1), dtype=torch.int64)
        reject(prefix, 1, "multimodal_prefix")
        prefix.seq_lens_cpu = torch.tensor([4.0])
        reject(prefix, 1, "multimodal_prefix")
        prefix.seq_lens_cpu = None
        reject(prefix, 1, "multimodal_prefix")

        early = _qwen_batch([req], [image], torch.tensor([3], dtype=torch.int64))
        reject(early, 1, "multimodal_prefill")
        bare_origin = _Req()
        bare_origin.multimodal_inputs = image
        bare_origin.origin_input_ids = None
        reject(
            _qwen_batch([bare_origin], [image], seq[:1]),
            1,
            "multimodal_prefill",
        )

        for bad in (
            None,
            torch.tensor([2], dtype=torch.int64),
            torch.tensor([[2.0]]),
            torch.zeros((1, 1, 1), dtype=torch.int64),
        ):
            bad_mm = _mm_input(bad, "video")
            bad_req = _Req()
            bad_req.origin_input_ids = [1]
            bad_req.multimodal_inputs = bad_mm
            reject(
                _qwen_batch([bad_req], [bad_mm], torch.tensor([1], dtype=torch.int64)),
                1,
                "multimodal_mrope",
            )

        self.assertEqual(id(req.multimodal_inputs), id(image))
        self.assertEqual(req.origin_input_ids, [1, 2, 3, 4])
        self.assertTrue(torch.all(state.accept_index == 3))
        self.assertEqual(
            state.reject_before_alloc(
                **_admit_kwargs(
                    state,
                    has_grammar=True,
                    multimodal_reject_reason="multimodal_mrope",
                )
            ),
            "grammar",
        )

    def test_env_off_still_skips_the_workspace(self):
        self.assertIsNone(
            build_fixed_accept_state(_worker(), env={SR_FIXED_ACCEPT_ENV: "0"})
        )


def _clone_req(req, mm):
    clone = _Req(finish_at=req.finish_at, cancel_at=req.cancel_at)
    clone.kv_committed_len = req.kv_committed_len
    clone.kv_allocated_len = req.kv_allocated_len
    clone.origin_input_ids = req.origin_input_ids
    clone.multimodal_inputs = mm
    return clone


def _mm_copy(mm):
    if mm is None:
        return None
    return _mm_input(mm.mrope_position_delta.clone(), mm.modality, features=True)


class FixedAcceptMultimodalFinalizeTest(CustomTestCase):
    def _compare(self, prefixes, rows, tokens, finish, cache, page_size, width):
        bs = len(rows)
        path = max(len(row) for row in rows)
        image = _mm_input(torch.tensor([[5]]), "image", features=True)
        video = _mm_input(torch.tensor([[-2]]), "video", features=True)
        mm_rows = [None if i % 2 == 0 else (image if i == 1 else video) for i in range(bs)]
        if bs == 1:
            mm_rows = [image]
        base_reqs = []
        for i, stop in enumerate(finish):
            if stop == "cancel":
                req = _Req(cancel_at=2)
            elif stop is None:
                req = _Req()
            else:
                req = _Req(finish_at=stop)
            req.kv_committed_len = prefixes[i]
            req.kv_allocated_len = prefixes[i]
            req.origin_input_ids = [0] * prefixes[i]
            req.multimodal_inputs = mm_rows[i]
            base_reqs.append(req)
        ref_mm = [_mm_copy(mm) for mm in mm_rows]
        ref_reqs = [_clone_req(req, mm) for req, mm in zip(base_reqs, ref_mm)]
        accepted, finished, truncated = apply_cpu_acceptance(
            ref_reqs, rows, tokens, None
        )
        ref_seq = torch.tensor(prefixes, dtype=torch.int64)
        if length_update_mode(finished) == "all":
            ref_seq = ref_seq + torch.tensor(accepted, dtype=torch.int64)
        src_index, tgt_index, page_index = [], [], []
        for i, (count, row, prefix) in enumerate(zip(accepted, truncated, prefixes)):
            for idx in row:
                if int(idx) < 0:
                    break
                src_index.append(int(idx))
            for j in range(int(count)):
                tgt_index.append(i * width + j)
            for offset in free_page_row_offsets(prefix, count, width, page_size):
                page_index.append(i * width + offset)
        pages = int(cache.max()) // page_size + 2
        ref_kv = torch.arange(2 * pages * page_size, dtype=torch.float32).reshape(
            2, 1, pages, page_size, 1, 1
        )
        fast_kv = ref_kv.clone()
        src_slots = cache[torch.tensor(src_index, dtype=torch.int64)] if src_index else torch.empty(0, dtype=torch.int64)
        tgt_slots = cache[torch.tensor(tgt_index, dtype=torch.int64)] if tgt_index else torch.empty(0, dtype=torch.int64)
        if src_slots.numel():
            flat = ref_kv.view(2, 1, -1, 1, 1)
            staged = flat.index_select(2, src_slots).clone()
            flat.index_copy_(2, tgt_slots, staged)
        page_slots = (
            cache[torch.tensor(page_index, dtype=torch.int64)]
            if page_index
            else torch.empty(0, dtype=torch.int64)
        )
        ref_pages = sorted(int(slot) // page_size for slot in page_slots.tolist())

        state = SRFixedAcceptState(max(bs, 1), path, width, page_size, "cpu")
        metrics = SimpleNamespace(paths=_PathCounter(), active=False)
        state.metrics = metrics
        predict, accept_index, accept_length = state.bind_verify_buffers(bs)
        predict.zero_()
        accept_index.fill_(-1)
        for i, row in enumerate(rows):
            for j, idx in enumerate(row):
                accept_index[i, j] = idx
                if idx >= 0:
                    predict[idx] = tokens[i][j]
            accept_length[i] = sum(1 for idx in row if idx >= 0) - 1
        alloc = NPUPagedTokenToKVPoolAllocator(page_size)
        alloc.kv_buffer = fast_kv
        batch = SimpleNamespace(
            reqs=base_reqs,
            seq_lens=torch.tensor(prefixes, dtype=torch.int64),
            seq_lens_cpu=torch.tensor(prefixes, dtype=torch.int64),
            out_cache_loc=cache.clone(),
            req_pool_indices=torch.arange(bs, dtype=torch.int64),
            device=torch.device("cpu"),
            topk=2,
            spec_algorithm=SimpleNamespace(is_standalone_remote=lambda: True),
            model_config=SimpleNamespace(think_end_id=None, hidden_size=4, dtype=torch.float32),
            multimodal_inputs=list(mm_rows),
        )
        result = state.finalize(
            batch, SimpleNamespace(), page_size, 2, alloc, accept_index, predict, accept_length
        )
        self.assertEqual(
            [req.output_ids for req in base_reqs],
            [req.output_ids for req in ref_reqs],
        )
        self.assertEqual(result.accepted_indices.tolist(), src_index)
        self.assertEqual(
            [req.histogram for req in base_reqs],
            [req.histogram for req in ref_reqs],
        )
        self.assertEqual(
            [req.spec_accepted_tokens for req in base_reqs],
            [req.spec_accepted_tokens for req in ref_reqs],
        )
        self.assertEqual(result.accept_length_per_req_cpu, [count - 1 for count in accepted])
        self.assertEqual(batch.seq_lens.tolist(), ref_seq.tolist())
        self.assertEqual(batch.seq_lens_cpu.tolist(), ref_seq.tolist())
        self.assertEqual(
            [req.kv_committed_len for req in base_reqs],
            [prefix + count for prefix, count in zip(prefixes, accepted)],
        )
        self.assertEqual(
            [prefix + count for prefix, count in zip(prefixes, accepted)],
            [
                prefix + (length + 1)
                for prefix, length in zip(prefixes, result.accept_length_per_req_cpu)
            ],
        )
        got_pages = sorted(int(page) for page in alloc.freed[-1].tolist())
        self.assertEqual(got_pages, ref_pages)
        self.assertEqual(fast_kv.view(-1).tolist(), ref_kv.view(-1).tolist())
        for req, mm in zip(base_reqs, mm_rows):
            self.assertIs(req.multimodal_inputs, mm)
            if mm is not None:
                self.assertEqual(tuple(mm.mrope_position_delta.shape), (1, 1))
                self.assertEqual(tuple(mm.mrope_positions.shape), (3, 4))
        self.assertEqual(metrics.paths["fixed_accept_hit"], 1)
        self.assertEqual(metrics.paths["fixed_accept_multimodal_hit"], 1)
        return result, base_reqs

    def test_outputs_kv_and_metadata_match_the_v1_accept_contract(self):
        self._compare(
            prefixes=[100, 100],
            rows=[[0, 1, 2, -1], [4, 5, -1, -1]],
            tokens=[[10, 11, 12, 0], [20, 21, 0, 0]],
            finish=[None, None],
            cache=torch.arange(30, dtype=torch.int64),
            page_size=128,
            width=15,
        )
        partial, reqs = self._compare(
            prefixes=[8, 9],
            rows=[[0, 1, 2, -1], [4, 5, 6, -1]],
            tokens=[[10, 11, 12, 0], [20, 21, 22, 0]],
            finish=[1, None],
            cache=torch.arange(16, dtype=torch.int64),
            page_size=4,
            width=8,
        )
        self.assertEqual(partial.draft_accept_length_cpu, [2])
        self.assertEqual(reqs[0].finished_reason, "stop")
        self._compare(
            prefixes=[8, 9],
            rows=[[0, -1, -1, -1], [4, -1, -1, -1]],
            tokens=[[10, 0, 0, 0], [20, 0, 0, 0]],
            finish=[1, 1],
            cache=torch.arange(8, dtype=torch.int64),
            page_size=4,
            width=4,
        )
        cancelled, reqs = self._compare(
            prefixes=[6],
            rows=[[0, 1, 2, -1]],
            tokens=[[10, 11, 12, 0]],
            finish=["cancel"],
            cache=torch.tensor([0, 1, 2, 3, 4, 5, 40, 7], dtype=torch.int64),
            page_size=4,
            width=8,
        )
        self.assertEqual(reqs[0].finished_reason, "cancel")
        self.assertEqual(cancelled.accept_length_per_req_cpu, [1])
        overlap = torch.arange(8, dtype=torch.int64)
        self._compare(
            prefixes=[0],
            rows=[[1, 2, -1, -1]],
            tokens=[[11, 12, 0, 0]],
            finish=[None],
            cache=overlap,
            page_size=4,
            width=4,
        )

    def test_two_rounds_keep_results_and_multimodal_metadata(self):
        state = SRFixedAcceptState(1, 2, 4, 4, "cpu")
        metrics = SimpleNamespace(paths=_PathCounter(), active=False)
        state.metrics = metrics
        image = _mm_input(torch.tensor([[7]]), "image", features=True)
        positions = image.mrope_positions.clone()
        position_id = id(image.mrope_positions)
        delta_id = id(image.mrope_position_delta)
        delta = image.mrope_position_delta.clone()
        feature = image.mm_items[0].feature.clone()

        def run(token, pool):
            predict, accept_index, accept_length = state.bind_verify_buffers(1)
            accept_index.fill_(-1)
            accept_index[0, 0] = 0
            accept_length[0] = 0
            predict.zero_()
            predict[0] = token
            alloc = NPUPagedTokenToKVPoolAllocator(4)
            alloc.kv_buffer = torch.zeros((2, 1, 2, 4, 1, 1))
            req = _Req()
            req.kv_committed_len = 1
            req.multimodal_inputs = image
            req.origin_input_ids = [1]
            batch = SimpleNamespace(
                reqs=[req],
                seq_lens=torch.tensor([1]),
                seq_lens_cpu=torch.tensor([1]),
                out_cache_loc=torch.arange(4),
                req_pool_indices=torch.tensor([pool]),
                device=torch.device("cpu"),
                spec_algorithm=None,
                model_config=SimpleNamespace(think_end_id=None, hidden_size=1, dtype=torch.float32),
            )
            return state.finalize(
                batch, None, 4, 2, alloc, accept_index, predict, accept_length
            )

        first = run(11, 3)
        second = run(99, 4)
        self.assertEqual(first.verified_id.tolist(), [11])
        self.assertEqual(first.accepted_indices.tolist(), [0])
        self.assertEqual(first.req_pool_indices.tolist(), [3])
        self.assertEqual(second.verified_id.tolist(), [99])
        self.assertEqual(id(image.mrope_positions), position_id)
        self.assertEqual(id(image.mrope_position_delta), delta_id)
        self.assertEqual(image.mrope_positions.tolist(), positions.tolist())
        self.assertEqual(image.mrope_position_delta.tolist(), delta.tolist())
        self.assertEqual(image.mm_items[0].feature.tolist(), feature.tolist())
        self.assertEqual(metrics.paths["fixed_accept_hit"], 2)
        self.assertEqual(metrics.paths["fixed_accept_multimodal_hit"], 2)

        text_state = SRFixedAcceptState(1, 2, 4, 4, "cpu")
        text_metrics = SimpleNamespace(paths=_PathCounter(), active=False)
        text_state.metrics = text_metrics
        predict, accept_index, accept_length = text_state.bind_verify_buffers(1)
        accept_index.fill_(-1)
        accept_index[0, 0] = 0
        accept_length[0] = 0
        predict.zero_()
        predict[0] = 3
        text_alloc = NPUPagedTokenToKVPoolAllocator(4)
        text_alloc.kv_buffer = torch.zeros((2, 1, 2, 4, 1, 1))
        text_req = _Req()
        text_req.multimodal_inputs = None
        text_state.finalize(
            SimpleNamespace(
                reqs=[text_req],
                seq_lens=torch.tensor([1]),
                seq_lens_cpu=torch.tensor([1]),
                out_cache_loc=torch.arange(4),
                req_pool_indices=torch.tensor([1]),
                device=torch.device("cpu"),
                spec_algorithm=None,
                model_config=SimpleNamespace(
                    think_end_id=None, hidden_size=1, dtype=torch.float32
                ),
            ),
            None,
            4,
            2,
            text_alloc,
            accept_index,
            predict,
            accept_length,
        )
        self.assertEqual(text_metrics.paths["fixed_accept_hit"], 1)
        self.assertEqual(text_metrics.paths["fixed_accept_multimodal_hit"], 0)


class FixedAcceptMultimodalPositionTest(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        namespace = {"torch": torch, "_has_foreach_copy": hasattr(torch, "_foreach_copy_")}
        graph = _load_functions(
            _REPO / "python/sglang/srt/model_executor/cuda_graph_runner.py",
            ["_grouped_foreach_copy_", "populate_from_forward_batch"],
            namespace,
        )
        positions = _load_functions(
            _REPO / "python/sglang/srt/model_executor/forward_batch_info.py",
            ["_compute_spec_mrope_positions"],
            {"torch": torch},
        )
        filtered = _load_functions(
            _REPO / "python/sglang/srt/managers/schedule_batch.py",
            ["filter_batch"],
            {
                "torch": torch,
                "is_pin_memory_available": lambda device: False,
                "Req": type("Req", (), {}),
            },
        )
        tail_graph = _load_functions(
            _REPO
            / "python/sglang/srt/speculative/standalone_remote/drafter/sr_tail_extend_graph.py",
            ["_fill_mrope_positions"],
            {"torch": torch},
        )
        encoder_ns = {"torch": torch}
        embed = _load_functions(
            _REPO / "python/sglang/srt/managers/mm_utils.py",
            ["general_mm_embed_routine"],
            encoder_ns,
        )
        cls.prod = {
            "populate": graph["populate_from_forward_batch"],
            "positions": positions["_compute_spec_mrope_positions"],
            "filter_batch": filtered["filter_batch"],
            "fill_tail": tail_graph["_fill_mrope_positions"],
            "embed": embed["general_mm_embed_routine"],
            "encoder_ns": encoder_ns,
        }

    def _positions(self, seq_rows, mm_inputs):
        holder = SimpleNamespace(seq_lens=torch.ones(len(mm_inputs), dtype=torch.int64))
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_draft_extend=lambda: False),
            multimodal_inputs=mm_inputs,
            spec_info=SimpleNamespace(positions=seq_rows),
        )
        self.prod["positions"](
            holder, SimpleNamespace(device=torch.device("cpu")), batch
        )
        return holder.mrope_positions

    def test_filter_keeps_each_request_position_and_tail_matches_v1(self):
        text_positions = torch.tensor([[10, 11]], dtype=torch.int64)
        image = _mm_input(torch.tensor([[5]]), "image")
        video = _mm_input(torch.tensor([[-2]]), "video")
        stored = torch.tensor(
            [[0, 1, 8], [0, 2, 8], [0, 3, 8]], dtype=torch.int64
        )
        video.mrope_positions = stored
        v1_video = _mm_copy(video)
        v1_video.mrope_positions = stored.clone()
        rows = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int64)
        mm_inputs = [None, image, video]
        original = self._positions(rows, mm_inputs)
        self.assertEqual(original[0].tolist(), [10, 11, 25, 26, 28, 29])

        class _Row:
            def __init__(self, done, mm):
                self._done = done
                self.multimodal_inputs = mm
                self.draft_is_paused = False
                self.return_logprob = False
                self.stream = False
                self.grammar = None

            def finished(self):
                return self._done

        schedule = SimpleNamespace(
            reqs=[_Row(False, None), _Row(True, image), _Row(False, video)],
            multimodal_inputs=list(mm_inputs),
            model_config=SimpleNamespace(is_encoder_decoder=False),
            device="cpu",
            req_pool_indices=torch.tensor([4, 5, 6]),
            seq_lens=torch.tensor([10, 12, 14]),
            seq_lens_cpu=torch.tensor([10, 12, 14]),
            orig_seq_lens=torch.tensor([10, 12, 14]),
            output_ids=None,
            sampling_info=SimpleNamespace(filter_batch=lambda *args, **kwargs: None),
            is_spec_v2=False,
            spec_info=None,
            maybe_wait_verify_done=lambda: None,
        )
        self.prod["filter_batch"](schedule)
        self.assertEqual([req.multimodal_inputs for req in schedule.reqs], [None, video])
        survivor_rows = torch.tensor([[10, 11], [30, 31]], dtype=torch.int64)
        survived = self._positions(survivor_rows, [None, video])
        expected = torch.cat((original[:, 0:2], original[:, 4:6]), dim=1)
        self.assertEqual(survived.tolist(), expected.tolist())
        self.assertNotEqual(survived[0, 2:].tolist(), original[0, 2:4].tolist())

        tail_after = tail_mrope_positions(video, 2, 3)
        tail_v1 = tail_mrope_positions(v1_video, 2, 3)
        self.assertEqual(tail_after.tolist(), tail_v1.tolist())
        self.assertEqual(tail_after[:, 0].tolist(), stored[:, 2].tolist())
        self.assertEqual(tail_after[0, 1:].tolist(), [1, 2])

    def test_target_graph_updates_the_valid_region_without_clearing_padding(self):
        image = _mm_input(torch.tensor([[5]]), "image")
        video = _mm_input(torch.tensor([[-2]]), "video")
        round1 = self._positions(
            torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int64),
            [image, None],
        )
        round2 = self._positions(
            torch.tensor([[40, 41]], dtype=torch.int64),
            [video],
        )
        self.assertNotEqual(round1[0, :2].tolist(), round2[0].tolist())
        cap = 10
        buffers = SimpleNamespace(
            input_ids=torch.zeros(cap, dtype=torch.int64),
            req_pool_indices=torch.zeros(4, dtype=torch.int64),
            seq_lens=torch.zeros(4, dtype=torch.int64),
            out_cache_loc=torch.zeros(cap, dtype=torch.int64),
            positions=torch.zeros(cap, dtype=torch.int64),
            mrope_positions=torch.full((3, cap), -99, dtype=torch.int64),
            seq_lens_cpu=torch.zeros(4, dtype=torch.int64),
            ngram_embedding_info=None,
            mamba_track_indices=None,
            mamba_track_mask=None,
            encoder_lens=None,
            pp_proxy_tensors=None,
        )

        def replay(mrope, raw_bs, raw_tokens):
            forward = SimpleNamespace(
                input_ids=torch.arange(raw_tokens, dtype=torch.int64),
                req_pool_indices=torch.arange(raw_bs, dtype=torch.int64),
                seq_lens=torch.ones(raw_bs, dtype=torch.int64),
                out_cache_loc=torch.arange(raw_tokens, dtype=torch.int64),
                positions=torch.arange(raw_tokens, dtype=torch.int64),
                mrope_positions=mrope,
                seq_lens_cpu=torch.ones(raw_bs, dtype=torch.int64),
                mamba_track_indices=None,
                mamba_track_mask=None,
                encoder_lens=None,
            )
            self.prod["populate"](
                buffers,
                forward_batch=forward,
                raw_bs=raw_bs,
                raw_num_token=raw_tokens,
                bs=raw_bs,
                seq_len_fill_value=0,
                require_gathered_buffer=False,
                num_tokens_per_bs=raw_tokens // raw_bs,
                nsa_enable_prefill_cp=False,
                enable_num_token_non_padded_flag=False,
            )

        replay(round1, 2, 6)
        self.assertEqual(buffers.mrope_positions[:, :6].tolist(), round1.tolist())
        replay(round2, 1, 2)
        self.assertEqual(buffers.mrope_positions[:, :2].tolist(), round2.tolist())
        self.assertEqual(buffers.mrope_positions[:, 2:6].tolist(), round1[:, 2:].tolist())
        self.assertTrue(torch.all(buffers.mrope_positions[:, 6:] == -99))

    def test_tail_graph_zeros_padding_before_copy(self):
        buffers = SimpleNamespace(
            mrope_positions=torch.full((3, 8), 7, dtype=torch.int64),
            positions=torch.arange(8, dtype=torch.int64),
        )
        src = torch.tensor([[4, 5], [6, 7], [8, 9]], dtype=torch.int64)
        self.prod["fill_tail"](None, buffers, SimpleNamespace(mrope_positions=src), 2)
        self.assertEqual(buffers.mrope_positions[:, :2].tolist(), src.tolist())
        self.assertTrue(torch.all(buffers.mrope_positions[:, 2:] == 0))

    def test_verify_and_tail_do_not_call_the_vision_encoder(self):
        calls = []

        def embed_mm_inputs(**kwargs):
            calls.append("encoder")
            return torch.zeros(1, 2), {}

        def get_global_server_args():
            return SimpleNamespace(
                enable_adaptive_dispatch_to_encoder=False, language_only=False
            )

        self.prod["encoder_ns"]["embed_mm_inputs"] = embed_mm_inputs
        self.prod["encoder_ns"]["get_global_server_args"] = get_global_server_args
        mm = _mm_input(torch.tensor([[1]]), "image", features=True)

        class _Language:
            def get_input_embeddings(self):
                return lambda ids: torch.zeros(int(ids.shape[0]), 2)

            def __call__(self, **kwargs):
                return "hidden"

        def run(verify, tail):
            batch = SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_decode=lambda: False,
                    is_target_verify=lambda: verify,
                ),
                is_sr_tail_extend=tail,
                contains_mm_inputs=lambda: True,
                mm_inputs=[mm],
                input_embeds=None,
                extend_prefix_lens_cpu=[0],
                extend_seq_lens_cpu=[4],
            )
            self.prod["embed"](
                torch.tensor([1, 2, 3, 4]),
                batch,
                _Language(),
            )
            return batch

        verify_batch = run(True, False)
        tail_batch = run(False, True)
        self.assertEqual(calls, [])
        self.assertIs(verify_batch.mm_inputs[0], mm)
        self.assertIs(tail_batch.mm_inputs[0], mm)
        prefill = run(False, False)
        self.assertEqual(calls, ["encoder"])
        self.assertIsNone(prefill.mm_inputs)


if __name__ == "__main__":
    unittest.main()
