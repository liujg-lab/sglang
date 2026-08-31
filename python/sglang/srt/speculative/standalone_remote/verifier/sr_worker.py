import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import alloc_for_decode
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleVerifyInput, EagleVerifyOutput
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
)
from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
    assemble_draft_rows,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import maybe_detect_nan
from sglang.srt.speculative.standalone_remote.sr_protocol import (
    is_health_check_req as _is_health_check,
)

logger = logging.getLogger(__name__)

_DEFAULT_DRAFT = {
    "draft_tokens": torch.tensor([0], dtype=torch.int64, device="cpu"),
}


def _req_committed_len(req) -> int:
    committed = int(getattr(req, "kv_committed_len", 0) or 0)
    if committed > 0:
        return committed
    return max(0, len(req.origin_input_ids) + len(req.output_ids))


def _sync_kv_from_seq_lens(batch: ScheduleBatch) -> None:
    seq_lens = batch.seq_lens.tolist()
    for req, sl in zip(batch.reqs, seq_lens):
        if _is_health_check(req):
            continue
        sl_i = int(sl)
        req.kv_committed_len = sl_i
        req.kv_allocated_len = sl_i


def _align_seq_lens_to_committed(batch: ScheduleBatch) -> None:
    device = batch.seq_lens.device
    committed = [_req_committed_len(req) for req in batch.reqs]
    new_lens = torch.tensor(committed, dtype=batch.seq_lens.dtype, device=device)
    if batch.seq_lens.shape == new_lens.shape:
        batch.seq_lens.copy_(new_lens)
    else:
        batch.seq_lens = new_lens
    cpu_lens = torch.tensor(committed, dtype=torch.int64)
    if batch.seq_lens_cpu is not None and batch.seq_lens_cpu.shape == cpu_lens.shape:
        batch.seq_lens_cpu.copy_(cpu_lens)
    else:
        batch.seq_lens_cpu = cpu_lens
    if batch.orig_seq_lens is not None:
        orig = torch.tensor(committed, dtype=batch.orig_seq_lens.dtype, device=device)
        if batch.orig_seq_lens.shape == orig.shape:
            batch.orig_seq_lens.copy_(orig)
        else:
            batch.orig_seq_lens = orig
    batch.seq_lens_sum = int(sum(committed))


class StandaloneRemoteWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.enable_nan_detection = server_args.enable_nan_detection
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.tp_group = getattr(target_worker.model_runner, "tp_group", None)
        self.tp_size = target_worker.tp_size

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

    @property
    def draft_model_runner(self):
        return None

    @property
    def model_runner(self):
        return self.target_worker.model_runner

    @property
    def model_config(self):
        return self.target_worker.model_config

    def clear_cache_pool(self):
        pass

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            logits_output, next_token_ids, _ = self.forward_target_extend(batch)
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )

        draft_num_tokens = getattr(batch, "draft_num_tokens", None)
        if draft_num_tokens is None:
            draft_num_tokens = self.speculative_num_draft_tokens

        if draft_num_tokens == 1 and not batch.forward_mode.is_idle():
            return self._forward_normal_decode(batch)

        spec_steps = self.speculative_num_steps
        spec_info = self.construct_draft_input(batch, draft_num_tokens, spec_steps)
        logits_output, verify_output, _, can_run_cuda_graph = self.verify(
            batch, spec_info
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
            accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, Optional[torch.Tensor]]:
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        return (
            batch_result.logits_output,
            batch_result.next_token_ids,
            model_worker_batch.seq_lens_cpu,
        )

    def construct_draft_input(
        self,
        batch: ScheduleBatch,
        draft_num_tokens: Optional[int] = None,
        spec_steps: Optional[int] = None,
    ) -> EagleVerifyInput:
        num_draft_tokens = (
            draft_num_tokens
            if draft_num_tokens is not None
            else self.speculative_num_draft_tokens
        )
        spec_steps = (
            spec_steps if spec_steps is not None else self.speculative_num_steps
        )

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk, spec_steps, num_draft_tokens
            )

        bs = batch.batch_size()
        device = batch.device
        topk = self.topk
        token_width = max(num_draft_tokens - 1, 1)

        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        if (
            batch.seq_lens_cpu is None
            or batch.seq_lens_cpu.sum().item() != batch.seq_lens_sum
        ):
            batch.seq_lens_cpu = batch.seq_lens.cpu()

        verified_id, parent_list, top_scores_index, draft_tokens = (
            self._assemble_draft_tensors(
                batch, bs, device, topk, spec_steps, num_draft_tokens, token_width
            )
        )

        if batch.sampling_info.penalizer_orchestrator.is_required:
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                verified_id
            )

        (
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            final_draft_tokens,
        ) = build_tree_kernel_efficient(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens,
            seq_lens=batch.seq_lens,
            seq_lens_sum=batch.seq_lens_sum,
            topk=topk,
            spec_steps=spec_steps,
            num_verify_tokens=num_draft_tokens,
        )

        return EagleVerifyInput(
            draft_token=final_draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=spec_steps,
            topk=topk,
            draft_token_num=num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        spec_info.prepare_for_verify(batch, self.page_size)
        spec_info.num_tokens_per_req = spec_info.draft_token_num
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode

        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        if self.enable_nan_detection:
            maybe_detect_nan(
                logits_output.next_token_logits,
                "StandaloneRemoteWorker verify logits",
            )

        spec_info.hidden_states = logits_output.hidden_states
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask=None,
        )
        if not batch.forward_mode.is_idle():
            _sync_kv_from_seq_lens(batch)

        for req in batch.reqs:
            if _is_health_check(req):
                continue
            req.spec_cnt = int(getattr(req, "spec_cnt", 0) or 0) + 1
            req.len_output_ids = len(req.output_ids)
            req.sr_step_id = int(getattr(req, "sr_step_id", 0) or 0) + 1

        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input
        return logits_output, res, model_worker_batch, can_run_cuda_graph

    def _forward_normal_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        bs = batch.batch_size()
        last_token_ids_cpu = [
            req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]
            for req in batch.reqs
        ]
        device = batch.seq_lens.device

        if batch.sampling_info.penalizer_orchestrator.is_required:
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                torch.tensor(last_token_ids_cpu, dtype=torch.int64, device=device)
            )

        batch.input_ids = torch.tensor(
            last_token_ids_cpu, dtype=torch.int32, device=device
        )
        batch.output_ids = None
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_info = None

        if batch.global_num_tokens is not None:
            batch.global_num_tokens = [bs]
        if batch.global_num_tokens_for_logprob is not None:
            batch.global_num_tokens_for_logprob = [bs]

        _align_seq_lens_to_committed(batch)
        batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)

        for req in batch.reqs:
            req.kv_committed_len += 1
            req.kv_allocated_len += 1

        batch.seq_lens.add_(1)
        batch.seq_lens_cpu.add_(1)
        if batch.orig_seq_lens is not None:
            batch.orig_seq_lens.add_(1)
        batch.seq_lens_sum += bs

        model_worker_batch = batch.get_model_worker_batch()
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)

        next_token_ids_list = batch_result.next_token_ids.tolist()
        for i, req in enumerate(batch.reqs):
            if _is_health_check(req):
                continue
            token = next_token_ids_list[i] if i < len(next_token_ids_list) else None
            if token is not None:
                req.output_ids.append(token)
                if req.grammar is not None and not req.finished():
                    try:
                        req.grammar.accept_token(token)
                    except ValueError:
                        logger.error(
                            "[SR] grammar.accept_token failed for %s token %s",
                            req.rid,
                            token,
                        )
                req.check_finished()
            req.cur_drafts = []
            req.draft_tokens_and_logits = _default_draft()
            req.spec_cnt = int(getattr(req, "spec_cnt", 0) or 0) + 1
            req.sr_step_id = int(getattr(req, "sr_step_id", 0) or 0) + 1
            req.len_output_ids = len(req.output_ids)

        return GenerationBatchResult(
            logits_output=batch_result.logits_output,
            next_token_ids=batch_result.next_token_ids,
            num_accepted_tokens=0,
            accept_length_per_req_cpu=[1] * bs,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
        )

    def _assemble_draft_tensors(
        self,
        batch: ScheduleBatch,
        bs: int,
        device,
        topk: int,
        spec_steps: int,
        num_draft_tokens: int,
        token_width: int,
    ):
        """Build verify tensors. Never synthesizes a SPECTRE-style fake bush."""
        verified_id_buf = torch.empty(bs, dtype=torch.int64)
        token_rows = []
        parent_rows: List[Optional[list]] = []
        index_rows: List[Optional[list]] = []

        for i, req in enumerate(batch.reqs):
            verified_id_buf[i] = (
                req.output_ids[-1]
                if len(req.output_ids) > 0
                else req.origin_input_ids[-1]
            )
            dtl = req.draft_tokens_and_logits
            dt = None if dtl is None else dtl.get("draft_tokens")
            pl = None if dtl is None else dtl.get("parent_list")
            ix = None if dtl is None else dtl.get("top_scores_index")
            token_rows.append(dt)
            parent_rows.append(pl)
            index_rows.append(ix)

        verified_id = verified_id_buf.to(device=device, non_blocking=True)
        parents, indices, draft_tokens = assemble_draft_rows(
            token_rows,
            parent_rows,
            index_rows,
            topk=topk,
            spec_steps=spec_steps,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )
        if draft_tokens.shape[1] != token_width:
            if draft_tokens.shape[1] > token_width:
                draft_tokens = draft_tokens[:, :token_width].contiguous()
            else:
                pad = torch.zeros(
                    bs,
                    token_width - draft_tokens.shape[1],
                    dtype=draft_tokens.dtype,
                    device=draft_tokens.device,
                )
                draft_tokens = torch.cat([draft_tokens, pad], dim=1)
        return verified_id, parents, indices, draft_tokens


def _default_draft() -> dict:
    return {
        "draft_tokens": _DEFAULT_DRAFT["draft_tokens"].clone(),
        "parent_list": None,
        "top_scores_index": None,
    }
