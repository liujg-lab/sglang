"""Lookahead template ForwardBatch must match per-layer ``init_new``.

CI collects the registered wrapper, not this file. A local run needs CUDA
and draft weights:

    OPT_DRAFT_MODEL=/path/to/model \\
        PYTHONPATH=python python3 test/manual/test_opt_lookahead_template_equiv.py

Batch size is 1. Depths are 1, 2, and 3. The template is admitted once.
Each layer builds the old batch with ``_verify_batch`` plus ``init_new``
and the template batch with ``_refresh_lookahead_forward``.
"""

import dataclasses
import os
import unittest
from types import SimpleNamespace

import torch

try:
    from sglang.test.test_utils import CustomTestCase
except Exception:
    CustomTestCase = unittest.TestCase

WIDTH = 4
PREFIX = 8
DEPTHS = (1, 2, 3)
_REFERENCE_FIELDS = {
    "attn_backend",
    "req_to_token_pool",
    "token_to_kv_pool",
    "sampling_info",
}


def _draft_host_cls():
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.standalone_remote.drafter.sr_draft_scheduler_mixin import (
        StandaloneRemoteDraftSchedulerMixin,
    )

    class _DraftHost:
        """Just enough of a Draft scheduler for one lookahead request."""

        def __init__(self, model_runner, server_args, tree_cache):
            self.model_worker = SimpleNamespace(model_runner=model_runner)
            self.server_args = server_args
            self.req_to_token_pool = model_runner.req_to_token_pool
            self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
            self.tree_cache = tree_cache
            self.device = model_runner.device
            self.model_config = model_runner.model_config
            self.enable_overlap = False
            self.spec_algorithm = SpeculativeAlgorithm.OPT_TREE_REMOTE
            self.tp_worker = SimpleNamespace(model_runner=model_runner)

        _sr_make_decode_batch = (
            StandaloneRemoteDraftSchedulerMixin._sr_make_decode_batch
        )

    return _DraftHost


def _describe(value) -> str:
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device})"
        )
    text = repr(value)
    if len(text) > 160:
        text = text[:157] + "..."
    return text


def _walk(left, right, path, mismatches) -> None:
    if left is right:
        return
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        same = False
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            try:
                same = bool(torch.equal(left, right))
            except Exception:
                same = False
        if not same:
            mismatches.append(f"{path}: {_describe(left)} != {_describe(right)}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            mismatches.append(f"{path}: len {len(left)} != {len(right)}")
            return
        for index, (item_left, item_right) in enumerate(zip(left, right)):
            _walk(item_left, item_right, f"{path}[{index}]", mismatches)
        return
    if (
        dataclasses.is_dataclass(left)
        and dataclasses.is_dataclass(right)
        and type(left) is type(right)
    ):
        _compare_mapping(vars(left), vars(right), path, mismatches, whitelist=False)
        return
    try:
        same = left == right
    except Exception:
        same = False
    if isinstance(same, torch.Tensor):
        try:
            same = bool(same)
        except Exception:
            same = False
    if same is not True:
        mismatches.append(f"{path}: {_describe(left)} != {_describe(right)}")


def _compare_mapping(left_vars, right_vars, path, mismatches, whitelist: bool) -> None:
    for key in sorted(set(left_vars) | set(right_vars)):
        child = key if not path else f"{path}.{key}"
        if whitelist and key in _REFERENCE_FIELDS:
            continue
        if key not in left_vars or key not in right_vars:
            mismatches.append(f"{child}: missing on one side")
            continue
        _walk(left_vars[key], right_vars[key], child, mismatches)


def forward_batch_mismatches(reference, fresh) -> list:
    """Field paths whose values differ. Reference objects are skipped."""
    mismatches = []
    _compare_mapping(vars(reference), vars(fresh), "", mismatches, whitelist=True)
    return mismatches


def _ancestor(width: int, depth: int) -> torch.Tensor:
    mask = torch.zeros((width, depth * width), dtype=torch.bool)
    for token in range(width):
        for layer in range(depth):
            mask[token, layer * width + token] = True
    return mask


class ForwardBatchCompareTest(CustomTestCase):
    def test_tensors_match_by_value_and_references_are_skipped(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import OptLayerSpec

        spec = OptLayerSpec(
            custom_mask=torch.tensor([True, False]),
            positions=torch.tensor([3, 3]),
            draft_token_num=2,
            num_tokens_per_req=2,
        )
        reference = SimpleNamespace(
            input_ids=torch.tensor([1, 2]),
            seq_lens_sum=4,
            attn_backend=object(),
            sampling_info=object(),
            req_to_token_pool=object(),
            token_to_kv_pool=object(),
            spec_info=spec,
        )
        fresh = SimpleNamespace(
            input_ids=torch.tensor([1, 2]),
            seq_lens_sum=4,
            attn_backend=object(),
            sampling_info=object(),
            req_to_token_pool=object(),
            token_to_kv_pool=object(),
            spec_info=OptLayerSpec(
                custom_mask=torch.tensor([True, False]),
                positions=torch.tensor([3, 3]),
                draft_token_num=2,
                num_tokens_per_req=2,
            ),
        )
        self.assertEqual(forward_batch_mismatches(reference, fresh), [])
        fresh.seq_lens_sum = 5
        self.assertEqual(
            forward_batch_mismatches(reference, fresh),
            ["seq_lens_sum: 4 != 5"],
        )

    def test_nested_tensor_mismatch_names_the_field(self):
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import OptLayerSpec

        reference = SimpleNamespace(
            spec_info=OptLayerSpec(
                custom_mask=torch.tensor([True]),
                positions=torch.tensor([1]),
                draft_token_num=1,
                num_tokens_per_req=1,
            )
        )
        fresh = SimpleNamespace(
            spec_info=OptLayerSpec(
                custom_mask=torch.tensor([False]),
                positions=torch.tensor([1]),
                draft_token_num=1,
                num_tokens_per_req=1,
            )
        )
        mismatches = forward_batch_mismatches(reference, fresh)
        self.assertEqual(len(mismatches), 1)
        self.assertTrue(mismatches[0].startswith("spec_info.custom_mask:"))


class OptLookaheadTemplateEquivTest(CustomTestCase):
    """One admitted template against ``init_new`` at every layer."""

    model_path = None

    @classmethod
    def setUpClass(cls):
        model_path = cls.model_path or os.environ.get("OPT_DRAFT_MODEL")
        if not model_path:
            raise unittest.SkipTest("set OPT_DRAFT_MODEL to the draft weights")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("lookahead template check needs CUDA")
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.server_args import PortArgs, ServerArgs

        cls.server_args = ServerArgs(
            model_path=model_path,
            tokenizer_path=model_path,
            host="127.0.0.1",
            port=30000,
            tp_size=1,
            mem_fraction_static=0.6,
            trust_remote_code=True,
            disable_cuda_graph=False,
            disable_piecewise_cuda_graph=True,
            cuda_graph_bs=[1],
            attention_backend="triton",
            page_size=1,
            speculative_algorithm="OPT_TREE_REMOTE",
            opt_tree_role="draft",
            opt_tree_width=WIDTH,
            opt_tree_tau=0.5,
            opt_tree_max_depth=4,
        )
        cls.port_args = PortArgs.init_new(cls.server_args)
        cls.model_config = ModelConfig.from_server_args(cls.server_args)
        cls.model_runner = ModelRunner(
            model_config=cls.model_config,
            mem_fraction_static=cls.server_args.mem_fraction_static,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=cls.port_args.nccl_port,
            server_args=cls.server_args,
            moe_ep_rank=0,
            moe_ep_size=1,
        )

    @classmethod
    def tearDownClass(cls):
        runner = getattr(cls, "model_runner", None)
        if runner is not None:
            del cls.model_runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_template_matches_init_new_for_each_layer(self):
        runner = self.model_runner
        req, tree_cache = self._prefill()
        host = _draft_host_cls()(runner, self.server_args, tree_cache)
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            OptLayerForward,
            _admit_lookahead,
            prepare_lookahead_runtime,
        )

        layer = OptLayerForward(host, req, PREFIX)
        runtime = prepare_lookahead_runtime(host, layer)
        _admit_lookahead(host, layer, runtime[0])
        template = layer._la_forward_template
        self.assertIsNotNone(template)
        for depth in DEPTHS:
            with self.subTest(depth=depth):
                self._compare_layer(layer, template, tree_cache, req, depth)

    def _compare_layer(self, layer, template, tree_cache, req, depth: int):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            _refresh_lookahead_forward,
            layer_attention_mask,
            layer_positions,
        )

        runner = self.model_runner
        device = runner.device
        cached = PREFIX + (depth - 1) * WIDTH
        layer.cached_len = cached
        tokens = torch.arange(20, 20 + WIDTH, dtype=torch.long, device=device)
        positions = layer_positions(PREFIX, depth, WIDTH)
        mask = layer_attention_mask(PREFIX, _ancestor(WIDTH, depth), device)
        out_cache_loc = self._alloc_layer(tree_cache, req, cached)
        schedule = layer._verify_batch(tokens, positions, mask, out_cache_loc)
        reference = ForwardBatch.init_new(schedule.get_model_worker_batch(), runner)
        fresh = _refresh_lookahead_forward(template, schedule, layer._la_mrope_delta)
        mismatches = forward_batch_mismatches(reference, fresh)
        self.assertEqual(mismatches, [])
        with torch.no_grad():
            reference_out = runner.forward(reference)
            fresh_out = runner.forward(fresh)
        self.assertTrue(reference_out.can_run_graph)
        self.assertTrue(fresh_out.can_run_graph)
        self.assertTrue(
            torch.equal(
                reference_out.logits_output.next_token_logits,
                fresh_out.logits_output.next_token_logits,
            )
        )

    def _prefill(self):
        from sglang.bench_one_batch import TreeCacheNamespace
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        runner = self.model_runner
        token_ids = [10 + (index % 100) for index in range(PREFIX)]
        req = Req(
            rid="opt-lookahead-equiv",
            origin_input_text="",
            origin_input_ids=token_ids,
            sampling_params=SamplingParams(temperature=0.0, max_new_tokens=1),
        )
        req.fill_ids = list(req.origin_input_ids)
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        tree_cache = TreeCacheNamespace(
            page_size=1,
            device=runner.device,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        )
        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.OPT_TREE_REMOTE,
        )
        batch.prepare_for_extend()
        forward_batch = ForwardBatch.init_new(batch.get_model_worker_batch(), runner)
        with torch.no_grad():
            runner.forward_extend(forward_batch)
        self.assertIsNotNone(req.req_pool_idx)
        req.kv_committed_len = PREFIX
        req.kv_allocated_len = PREFIX
        return req, tree_cache

    def _alloc_layer(self, tree_cache, req, cached: int):
        from sglang.srt.mem_cache.common import alloc_token_slots
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        runner = self.model_runner
        device = runner.device
        loc = alloc_token_slots(tree_cache, WIDTH)
        pool_index = torch.tensor(
            [int(req.req_pool_idx)], dtype=torch.int64, device=device
        )
        start = torch.tensor([cached], dtype=torch.int64, device=device)
        assign_req_to_token_pool_func(
            pool_index,
            runner.req_to_token_pool.req_to_token,
            start,
            start + WIDTH,
            loc,
            1,
        )
        return loc.reshape(-1)


if __name__ == "__main__":
    unittest.main()
