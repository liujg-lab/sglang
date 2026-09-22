"""CPU contracts for the default-on SR fixed-capacity accept path."""

from __future__ import annotations

import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch

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
        has_multimodal=False,
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
            ("has_multimodal", True, "multimodal"),
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

        for inject in ("pack", "readback"):
            req, alloc = once(inject)
            self.assertEqual(req.output_ids, [])
            self.assertEqual(alloc.freed, [])
        req, alloc = once("move")
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


if __name__ == "__main__":
    unittest.main()
