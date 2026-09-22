import logging
import time
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
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
from sglang.srt.speculative.spec_utils import generate_token_bitmask, maybe_detect_nan
from sglang.srt.speculative.standalone_remote.sr_protocol import (
    is_health_check_req as _is_health_check,
)
from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
    bind_graph_host_metrics,
    restore_graph_host_metrics,
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


def _snapshot_seq_lens_cpu(batch: ScheduleBatch) -> Optional[torch.Tensor]:
    cpu = getattr(batch, "seq_lens_cpu", None)
    if cpu is None:
        return None
    return cpu.clone()


def _sync_kv_from_cpu_lengths(
    batch: ScheduleBatch,
    seq_lens_cpu_pre,
    accept_length_per_req_cpu,
) -> None:
    """SET KV bounds from CPU pre-verify lengths plus accepted draft + bonus.

    ``accept_length`` is draft tokens only; +1 is the Target bonus. Finished
    requests use the same formula after eagle_info truncates accept_index.
    Do not read device ``seq_lens``: paged topk>1 may leave it unchanged.
    """
    if seq_lens_cpu_pre is None:
        pre_list = [_req_committed_len(req) for req in batch.reqs]
    elif isinstance(seq_lens_cpu_pre, torch.Tensor):
        pre_list = seq_lens_cpu_pre.tolist()
    else:
        pre_list = list(seq_lens_cpu_pre)
    acc = list(accept_length_per_req_cpu or [])
    for i, req in enumerate(batch.reqs):
        if _is_health_check(req):
            continue
        accepted = int(acc[i]) if i < len(acc) else 0
        pre = int(pre_list[i]) if i < len(pre_list) else _req_committed_len(req)
        committed = pre + accepted + 1
        req.kv_committed_len = committed
        req.kv_allocated_len = committed


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
        self._verify_max_bs = max(
            int(getattr(server_args, "cuda_graph_max_bs", 0) or 0),
            int(getattr(server_args, "max_running_requests", 0) or 0),
            int(getattr(server_args, "standalone_remote_max_batch_size", 0) or 0),
            1,
        )
        self._verify_tokens_buf = None
        self._verify_parents_buf = None
        self._verify_indices_buf = None
        self._verify_id_buf = None
        self._verify_mask_buf = None
        self._verify_pos_buf = None
        runner = getattr(target_worker, "model_runner", None)
        self._hybrid_needs_hidden = bool(
            runner is not None
            and (
                getattr(runner, "hybrid_gdn_config", None) is not None
                or getattr(runner, "mamba2_config", None) is not None
                or getattr(runner, "hybrid_lightning_config", None) is not None
            )
        )
        from sglang.srt.speculative.standalone_remote.verifier.sr_target_warmup import (
            warm_sr_target_kernels,
        )

        warm_sr_target_kernels(self)

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

    def _need_target_hidden(self, batch: Optional[ScheduleBatch] = None) -> bool:
        """Keep Target hidden states for HTTP return or hybrid models.

        Ordinary SR Target does not send hidden states to remote Draft.
        """
        if getattr(self.server_args, "enable_return_hidden_states", False):
            return True
        if self._hybrid_needs_hidden:
            return True
        if batch is None:
            return False
        if getattr(batch, "return_hidden_states", False):
            return True
        for req in getattr(batch, "reqs", None) or []:
            if getattr(req, "return_hidden_states", False):
                return True
        return False

    def _target_capture_hidden_mode(
        self, batch: Optional[ScheduleBatch] = None
    ) -> CaptureHiddenMode:
        if self._need_target_hidden(batch):
            return CaptureHiddenMode.FULL
        return CaptureHiddenMode.NULL

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
        metrics = getattr(batch, "sr_round_metrics", None)
        with metrics.phase("construct_tree") if metrics else nullcontext():
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
        model_worker_batch.capture_hidden_mode = self._target_capture_hidden_mode(batch)
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
                self.topk, spec_steps, num_draft_tokens, device=batch.device
            )

        bs = batch.batch_size()
        device = batch.device
        topk = self.topk
        token_width = max(num_draft_tokens - 1, 1)

        if batch.seq_lens_cpu is not None:
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
        else:
            batch.seq_lens_cpu = batch.seq_lens.cpu()
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())

        verified_id, parent_list, top_scores_index, draft_tokens = (
            self._assemble_draft_tensors(
                batch, bs, device, topk, spec_steps, num_draft_tokens, token_width
            )
        )

        if batch.sampling_info.penalizer_orchestrator.is_required:
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                verified_id
            )

        needed_mask = (
            int(batch.seq_lens_sum) * num_draft_tokens
            + num_draft_tokens * num_draft_tokens * bs
        )
        needed_pos = bs * num_draft_tokens
        tree_mask_buf, position_buf, wrote_graph_mask, wrote_graph_pos = (
            self._verify_mask_position_bufs(needed_mask, needed_pos, device)
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
            tree_mask_buf=tree_mask_buf,
            position_buf=position_buf,
        )

        spec_info = EagleVerifyInput(
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
            capture_hidden_mode=self._target_capture_hidden_mode(batch),
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )
        if not wrote_graph_mask or not wrote_graph_pos:
            self._maybe_update_graph_verify_buffers(spec_info)
        return spec_info

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        metrics = getattr(batch, "sr_round_metrics", None)
        prepare_start = time.perf_counter()
        prepare_hidden = self._need_target_hidden(batch)
        seq_lens_pre_verify = batch.seq_lens.clone()
        seq_lens_cpu_pre = _snapshot_seq_lens_cpu(batch)
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

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrive_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrive_next_token.shape
            ).cpu()

        if metrics:
            metrics.host["verify_prepare"] += time.perf_counter() - prepare_start
        graph_runner = getattr(
            getattr(self.target_worker, "model_runner", None), "graph_runner", None
        )
        metrics_token = bind_graph_host_metrics(graph_runner, metrics)
        try:
            with metrics.phase("verify_forward", device=True) if metrics else nullcontext():
                batch_result = self.target_worker.forward_batch_generation(
                    model_worker_batch, is_verify=True
                )
        finally:
            restore_graph_host_metrics(metrics_token)
        accept_start = time.perf_counter()
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        vocab_mask = None
        if batch.has_grammar:
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )
            if vocab_mask is not None:
                assert spec_info.grammar is not None
                vocab_mask = vocab_mask.to(spec_info.retrive_next_token.device)
                batch.sampling_info.vocab_mask = None

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
            vocab_mask,
            prepare_local_draft_hidden=prepare_hidden,
        )
        if not batch.forward_mode.is_idle():
            _sync_kv_from_cpu_lengths(
                batch, seq_lens_cpu_pre, res.accept_length_per_req_cpu
            )

        for req in batch.reqs:
            if _is_health_check(req):
                continue
            req.spec_cnt = int(getattr(req, "spec_cnt", 0) or 0) + 1
            req.len_output_ids = len(req.output_ids)
            req.sr_step_id = int(getattr(req, "sr_step_id", 0) or 0) + 1

        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states[
                res.accepted_indices
            ]

        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
            or self.target_worker.model_runner.hybrid_lightning_config is not None
        ):
            self._mamba_verify_update(
                batch, res, logits_output, spec_info, seq_lens_pre_verify
            )

        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input
        if metrics:
            # Verification is asynchronous; acceptance can wait on its results.
            metrics.host["accept_commit_including_wait"] += (
                time.perf_counter() - accept_start
            )
            lengths = res.accept_length_per_req_cpu
            metrics.counts["verify_batches"] += 1
            metrics.counts["verify_requests"] += len(lengths)
            metrics.counts["accepted_draft_tokens"] += sum(lengths)
            metrics.counts["accepted_tokens_including_bonus"] += sum(lengths) + len(lengths)
            metrics.counts["first_level_hits"] += sum(n > 0 for n in lengths)
            metrics.counts["verify_graph_batches"] += int(can_run_cuda_graph)
        return logits_output, res, model_worker_batch, can_run_cuda_graph

    def _mamba_verify_update(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
        spec_info: EagleVerifyInput,
        seq_lens_pre_verify: torch.Tensor,
    ):
        if batch.forward_mode.is_idle():
            return

        hidden = logits_output.hidden_states
        device = (
            hidden.device
            if hidden is not None
            else getattr(logits_output.next_token_logits, "device", batch.device)
        )
        accepted_length = (
            torch.tensor(
                res.accept_length_per_req_cpu,
                device=device,
                dtype=torch.int64,
            )
            + 1
        )
        cumulative_accepted_lengths = torch.cumsum(accepted_length, dim=0)
        accepted_indices_start = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=cumulative_accepted_lengths.dtype,
                    device=cumulative_accepted_lengths.device,
                ),
                cumulative_accepted_lengths[:-1],
            ]
        )
        accepted_indices_offset = torch.arange(
            0,
            len(batch.seq_lens) * batch.spec_info.draft_token_num,
            step=batch.spec_info.draft_token_num,
            dtype=accepted_indices_start.dtype,
            device=accepted_indices_start.device,
        )

        if spec_info.topk > 1 and res.accepted_indices.shape[0] > 0:
            accepted_steps = (
                res.accepted_indices[cumulative_accepted_lengths - 1]
                - accepted_indices_offset
            )
        else:
            accepted_steps = accepted_length - 1

        if batch.mamba_track_indices is not None:
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            mamba_steps_to_track = torch.where(
                to_track_mask,
                res.accepted_indices[to_track_ith + accepted_indices_start]
                - accepted_indices_offset,
                -1,
            )
        else:
            mamba_steps_to_track = None

        self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

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
        batch_result = self._forward_target_eager(model_worker_batch)

        next_token_ids_list = batch_result.next_token_ids.tolist()
        if batch.return_logprob:
            _add_ar_fallback_logprobs(
                batch, batch_result.logits_output, next_token_ids_list
            )
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

        metrics = getattr(batch, "sr_round_metrics", None)
        if metrics:
            metrics.counts["normal_decode_fallback_batches"] += 1
            metrics.counts["normal_decode_fallback_requests"] += bs
        return GenerationBatchResult(
            logits_output=batch_result.logits_output,
            next_token_ids=batch_result.next_token_ids,
            num_accepted_tokens=0,
            accept_length_per_req_cpu=[1] * bs,
            can_run_cuda_graph=False,
        )

    def _forward_target_eager(self, model_worker_batch):
        """1-token AR fallback must not replay TARGET_VERIFY CUDA graphs.

        Dual-ntpb DECODE graphs exist for safety if graph_runner is invoked
        elsewhere; this path stays eager.
        """
        runner = self.target_worker.model_runner
        graph_runner = getattr(runner, "graph_runner", None)
        runner.graph_runner = None
        try:
            return self.target_worker.forward_batch_generation(model_worker_batch)
        finally:
            runner.graph_runner = graph_runner

    def _attn_backend(self):
        runner = getattr(self.target_worker, "model_runner", None)
        return getattr(runner, "attn_backend", None)

    def _grow_int64_buf(self, name: str, rows: int, cols: int, device):
        buf = getattr(self, name, None)
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        cols = max(int(cols), 1)
        rows = max(int(rows), 1)
        if (
            buf is None
            or buf.device != dev
            or buf.dtype != torch.int64
            or buf.dim() != 2
            or buf.shape[0] < rows
            or buf.shape[1] < cols
        ):
            buf = torch.zeros((rows, cols), dtype=torch.int64, device=dev)
            setattr(self, name, buf)
        return buf

    def _verify_input_bufs(self, bs: int, token_width: int, parent_w: int, index_w: int, device):
        cap = max(int(getattr(self, "_verify_max_bs", 1) or 1), bs)
        tokens = self._grow_int64_buf("_verify_tokens_buf", cap, token_width, device)
        parents = self._grow_int64_buf("_verify_parents_buf", cap, max(parent_w, 1), device)
        indices = self._grow_int64_buf("_verify_indices_buf", cap, max(index_w, 1), device)
        verified = getattr(self, "_verify_id_buf", None)
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if (
            verified is None
            or verified.device != dev
            or verified.dtype != torch.int64
            or verified.numel() < cap
        ):
            verified = torch.empty((cap,), dtype=torch.int64, device=dev)
            self._verify_id_buf = verified
        return verified[:bs], parents[:bs], indices[:bs], tokens[:bs]

    def _verify_mask_position_bufs(self, needed_mask: int, needed_pos: int, device):
        backend = self._attn_backend()
        graph_mask = graph_pos = None
        getter = getattr(backend, "get_verify_buffers_to_fill_after_draft", None)
        if getter is not None:
            bufs = getter() or [None, None]
            graph_mask = bufs[0] if len(bufs) > 0 else None
            graph_pos = bufs[1] if len(bufs) > 1 else None

        wrote_graph_mask = False
        if graph_mask is not None:
            if graph_mask.numel() < needed_mask:
                raise RuntimeError(
                    f"SR verify tree_mask ({needed_mask}) exceeds graph buffer "
                    f"({graph_mask.numel()})"
                )
            mask_buf = graph_mask.reshape(-1)[:needed_mask]
            wrote_graph_mask = True
        else:
            mask_buf = self._ensure_own_mask(needed_mask, device)

        wrote_graph_pos = False
        if graph_pos is not None:
            if graph_pos.numel() < needed_pos:
                raise RuntimeError(
                    f"SR verify positions ({needed_pos}) exceeds graph buffer "
                    f"({graph_pos.numel()})"
                )
            pos_buf = graph_pos.reshape(-1)[:needed_pos]
            wrote_graph_pos = True
        else:
            pos_buf = self._ensure_own_pos(needed_pos, device)
        return mask_buf, pos_buf, wrote_graph_mask, wrote_graph_pos

    def _ensure_own_mask(self, needed: int, device):
        buf = getattr(self, "_verify_mask_buf", None)
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if buf is None or buf.device != dev or buf.numel() < needed:
            buf = torch.empty((needed,), dtype=torch.bool, device=dev)
            self._verify_mask_buf = buf
        return buf.reshape(-1)[:needed]

    def _ensure_own_pos(self, needed: int, device):
        buf = getattr(self, "_verify_pos_buf", None)
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if buf is None or buf.device != dev or buf.numel() < needed:
            buf = torch.empty((needed,), dtype=torch.int64, device=dev)
            self._verify_pos_buf = buf
        return buf.reshape(-1)[:needed]

    def _maybe_update_graph_verify_buffers(self, spec_info: EagleVerifyInput) -> None:
        backend = self._attn_backend()
        updater = getattr(backend, "update_verify_buffers_to_fill_after_draft", None)
        if updater is None:
            return
        updater(spec_info, None)

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
        verified_cpu = torch.empty(bs, dtype=torch.int64)
        token_rows = []
        parent_rows: List[Optional[list]] = []
        index_rows: List[Optional[list]] = []

        for i, req in enumerate(batch.reqs):
            verified_cpu[i] = (
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

        parent_w = max(spec_steps, 1)
        index_w = token_width
        verified_id, out_parents, out_indices, out_tokens = self._verify_input_bufs(
            bs, token_width, parent_w, index_w, device
        )
        verified_id.copy_(verified_cpu, non_blocking=True)
        parents, indices, draft_tokens = assemble_draft_rows(
            token_rows,
            parent_rows,
            index_rows,
            topk=topk,
            spec_steps=spec_steps,
            num_draft_tokens=num_draft_tokens,
            device=device,
            out_tokens=out_tokens,
            out_parents=out_parents,
            out_indices=out_indices,
        )
        if draft_tokens.shape[1] != token_width:
            if draft_tokens.shape[1] > token_width:
                draft_tokens = draft_tokens[:, :token_width]
            else:
                out_tokens[:, : draft_tokens.shape[1]].copy_(draft_tokens)
                out_tokens[:, draft_tokens.shape[1] : token_width].zero_()
                draft_tokens = out_tokens[:, :token_width]
        return verified_id, parents, indices, draft_tokens


def _row_to_list(container, i):
    if not container or i >= len(container):
        return None
    item = container[i]
    if isinstance(item, torch.Tensor):
        return item.tolist()
    return item


def _add_ar_fallback_logprobs(
    batch: ScheduleBatch,
    logits_output: Optional[LogitsProcessorOutput],
    next_token_ids_list,
) -> None:
    """Fill 1-token output logprobs. Scheduler treats SR as spec v1."""
    if logits_output is None or logits_output.next_token_logprobs is None:
        return
    logprobs_list = logits_output.next_token_logprobs.tolist()
    top_val = logits_output.next_token_top_logprobs_val
    top_idx = logits_output.next_token_top_logprobs_idx
    tok_val = logits_output.next_token_token_ids_logprobs_val
    tok_idx = logits_output.next_token_token_ids_logprobs_idx
    for i, req in enumerate(batch.reqs):
        if _is_health_check(req) or not req.return_logprob:
            continue
        if i >= len(next_token_ids_list) or i >= len(logprobs_list):
            continue
        token = next_token_ids_list[i]
        if token is None:
            continue
        req.output_token_logprobs_val.append(logprobs_list[i])
        req.output_token_logprobs_idx.append(token)
        if req.top_logprobs_num > 0 and top_val:
            req.output_top_logprobs_val.append(_row_to_list(top_val, i))
            req.output_top_logprobs_idx.append(_row_to_list(top_idx, i))
        if req.token_ids_logprob is not None and tok_val:
            req.output_token_ids_logprobs_val.append(_row_to_list(tok_val, i))
            req.output_token_ids_logprobs_idx.append(_row_to_list(tok_idx, i))


def _default_draft() -> dict:
    return {
        "draft_tokens": _DEFAULT_DRAFT["draft_tokens"].clone(),
        "parent_list": None,
        "top_scores_index": None,
    }
