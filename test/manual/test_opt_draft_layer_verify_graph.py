"""OPT draft layer: eager logits must match the TARGET_VERIFY CUDA graph.

CI does not collect this file. It needs a draft GPU and model weights:

    OPT_DRAFT_LAYER_GRAPH=1 OPT_DRAFT_MODEL=/path/to/model \\
        PYTHONPATH=python python3 test/manual/test_opt_draft_layer_verify_graph.py

Batch sizes are 1 and 2. Depths are 1, 2, and 3. Eager is the same forward
with ``can_run`` forced false. One request carries ``mrope_position_delta``;
replay must copy those 3-axis positions instead of leaving the capture zeros.
"""

import os
import unittest

import torch

from sglang.bench_one_batch import TreeCacheNamespace
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.sampler import SamplingBatchInfo
from sglang.srt.managers.schedule_batch import MultimodalInputs, Req, ScheduleBatch
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
    OptLayerSpec,
    _keep_raw_logits,
    layer_attention_mask,
    layer_positions,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.test.test_utils import CustomTestCase

WIDTH = 4
BASE_PREFIX = 8
_RUN = os.environ.get("OPT_DRAFT_LAYER_GRAPH") == "1"


def _ancestor(width: int, depth: int) -> torch.Tensor:
    mask = torch.zeros((width, depth * width), dtype=torch.bool)
    for token in range(width):
        for layer in range(depth):
            mask[token, layer * width + token] = True
    return mask


def _expected_mrope(positions: torch.Tensor, deltas, batch_size: int) -> torch.Tensor:
    seq = positions.view(batch_size, -1)
    delta = torch.stack([item.reshape(-1)[:1] for item in deltas]).to(
        device=seq.device, dtype=seq.dtype
    )
    return (seq + delta).flatten().unsqueeze(0).repeat(3, 1)


@unittest.skipUnless(_RUN, "set OPT_DRAFT_LAYER_GRAPH=1 to run this manual check")
class TestOptDraftLayerVerifyGraph(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("OPT draft layer graph check needs CUDA")
        model_path = os.environ.get("OPT_DRAFT_MODEL")
        if not model_path:
            raise unittest.SkipTest("set OPT_DRAFT_MODEL to the draft weights")
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
            cuda_graph_bs=[1, 2],
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

    def test_layer_eager_matches_verify_graph(self):
        runner = self.model_runner
        self.assertTrue(runner.graph_runner.dual_ntpb)
        self.assertEqual(
            set(runner.graph_runner.spectre_ntpb_options), {1, WIDTH}
        )
        for batch_size in (1, 2):
            for depth in (1, 2, 3):
                with self.subTest(batch_size=batch_size, depth=depth):
                    self._compare_one(batch_size, depth)

    def _compare_one(self, batch_size: int, depth: int):
        cached = BASE_PREFIX + (depth - 1) * WIDTH
        reqs, tree_cache = self._prefill(batch_size, cached)
        deltas = [
            torch.tensor([[3 + i]], dtype=torch.int64) for i in range(batch_size)
        ]
        out_cache_loc = self._alloc_layer(reqs, tree_cache, cached)
        graph_batch = self._verify_forward_batch(
            reqs, tree_cache, cached, depth, deltas, out_cache_loc
        )
        eager_batch = self._verify_forward_batch(
            reqs, tree_cache, cached, depth, deltas, out_cache_loc
        )
        self._assert_mrope_replay(graph_batch, deltas)

        runner = self.model_runner
        self.assertTrue(runner.graph_runner.can_run(graph_batch))
        graph_out = runner.forward(graph_batch)
        self.assertTrue(graph_out.can_run_graph)
        n = int(graph_batch.input_ids.shape[0])
        graph_logits = graph_out.logits_output.next_token_logits[:n].float()

        if runner.model_is_mrope:
            copied = runner.graph_runner.buffers.mrope_positions[:, :n]
            self.assertTrue(torch.equal(copied, graph_batch.mrope_positions))
            self.assertFalse(torch.all(copied == 0))

        original = runner.graph_runner.can_run
        runner.graph_runner.can_run = lambda _batch: False
        try:
            eager_out = runner.forward(eager_batch)
        finally:
            runner.graph_runner.can_run = original
        self.assertFalse(eager_out.can_run_graph)
        eager_logits = eager_out.logits_output.next_token_logits[:n].float()
        torch.testing.assert_close(graph_logits, eager_logits, rtol=1e-2, atol=1e-2)

    def _assert_mrope_replay(self, batch, deltas):
        expected = _expected_mrope(
            batch.spec_info.positions, deltas, batch.batch_size
        )
        self.assertFalse(torch.all(expected == 0))
        if self.model_runner.model_is_mrope:
            self.assertTrue(torch.equal(batch.mrope_positions, expected))
        else:
            batch.mrope_positions = expected
        buffers = self.model_runner.graph_runner.buffers
        buffers.mrope_positions.zero_()
        self.model_runner.graph_runner.replay_prepare(batch)
        copied = buffers.mrope_positions[:, : expected.shape[1]]
        self.assertTrue(torch.equal(copied, expected))
        self.assertFalse(torch.all(copied == 0))

    def _prefill(self, batch_size: int, length: int):
        runner = self.model_runner
        sampling = SamplingParams(temperature=0.0, max_new_tokens=1)
        reqs = []
        for i in range(batch_size):
            token_ids = [10 + ((i * 17 + j) % 100) for j in range(length)]
            req = Req(
                rid=f"opt-layer-{i}",
                origin_input_text="",
                origin_input_ids=token_ids,
                sampling_params=sampling,
            )
            req.fill_ids = list(req.origin_input_ids)
            req.logprob_start_len = -1
            reqs.append(req)
        tree_cache = TreeCacheNamespace(
            page_size=1,
            device=runner.device,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        )
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.OPT_TREE_REMOTE,
        )
        batch.prepare_for_extend()
        forward_batch = ForwardBatch.init_new(
            batch.get_model_worker_batch(), runner
        )
        runner.forward_extend(forward_batch)
        for req in reqs:
            self.assertIsNotNone(req.req_pool_idx)
        return reqs, tree_cache

    def _alloc_layer(self, reqs, tree_cache, cached: int):
        runner = self.model_runner
        device = runner.device
        locs = []
        for req in reqs:
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
            locs.append(loc.reshape(-1))
        return torch.cat(locs)

    def _verify_forward_batch(
        self, reqs, tree_cache, cached: int, depth: int, deltas, out_cache_loc
    ):
        runner = self.model_runner
        device = runner.device
        tokens = []
        masks = []
        positions = []
        for index, _req in enumerate(reqs):
            row = torch.arange(
                20 + index * WIDTH, 20 + (index + 1) * WIDTH, dtype=torch.long
            )
            tokens.append(row)
            masks.append(
                layer_attention_mask(BASE_PREFIX, _ancestor(WIDTH, depth), device)
            )
            positions.append(layer_positions(BASE_PREFIX, depth, WIDTH))
        flat_tokens = torch.cat(tokens).to(device=device, dtype=torch.int64)
        flat_positions = torch.cat(positions).to(device=device, dtype=torch.int64)
        flat_mask = torch.cat(masks)
        spec = OptLayerSpec(
            custom_mask=flat_mask.reshape(-1),
            positions=flat_positions,
            draft_token_num=WIDTH,
            num_tokens_per_req=WIDTH,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        batch = ScheduleBatch(
            reqs=list(reqs),
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.OPT_TREE_REMOTE,
            forward_mode=ForwardMode.TARGET_VERIFY,
            device=str(device),
        )
        batch.req_pool_indices = torch.tensor(
            [int(req.req_pool_idx) for req in reqs],
            dtype=torch.int64,
            device=device,
        )
        batch.seq_lens = torch.full(
            (len(reqs),), cached, dtype=torch.int64, device=device
        )
        batch.seq_lens_cpu = torch.full((len(reqs),), cached, dtype=torch.int64)
        batch.orig_seq_lens = torch.full(
            (len(reqs),), cached, dtype=torch.int32, device=device
        )
        batch.seq_lens_sum = cached * len(reqs)
        batch.input_ids = flat_tokens
        batch.output_ids = None
        batch.out_cache_loc = out_cache_loc
        batch.extend_lens = [WIDTH] * len(reqs)
        batch.prefix_lens = [cached] * len(reqs)
        batch.extend_num_tokens = WIDTH * len(reqs)
        batch.extend_logprob_start_lens = [cached] * len(reqs)
        batch.global_forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec
        batch.multimodal_inputs = [
            MultimodalInputs(mm_items=[], mrope_position_delta=delta)
            for delta in deltas
        ]
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )
        _keep_raw_logits(batch)
        return ForwardBatch.init_new(batch.get_model_worker_batch(), runner)


if __name__ == "__main__":
    unittest.main()
