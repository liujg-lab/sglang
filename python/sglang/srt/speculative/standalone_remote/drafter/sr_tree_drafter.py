"""STANDALONE-style top-k tree expansion on the remote Draft process.

Uses the Draft server's existing TpModelWorker (no second weight load).
Linear KV stays the Target committed prefix; tree KV is ephemeral.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.common import alloc_paged_token_slots_extend, alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_utils import organize_draft_results
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    device_backend_key,
    fast_topk,
    get_last_loc_large_page_size_large_top_k,
    maybe_detect_nan,
    maybe_detect_oob,
    select_top_k_tokens,
)
from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
    advance_tree_draft_positions,
)
from sglang.srt.utils import next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)

SRTreeWindow = Tuple[List[int], Optional[List[int]], Optional[List[int]]]


class SRTreeDrafter:
    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler
        self.server_args = scheduler.server_args
        self.draft_model_runner = scheduler.tp_worker.model_runner
        self.device = getattr(scheduler, "device", None) or self.draft_model_runner.device
        self.topk = max(1, int(self.server_args.speculative_eagle_topk or 1))
        self.speculative_num_steps = max(
            1, int(self.server_args.speculative_num_steps or 1)
        )
        self.speculative_num_draft_tokens = int(
            self.server_args.speculative_num_draft_tokens
            or (self.speculative_num_steps + 1)
        )
        self.page_size = int(self.server_args.page_size or 1)
        self.token_to_kv_pool_allocator = scheduler.token_to_kv_pool_allocator
        self.req_to_token_pool = scheduler.req_to_token_pool
        self.model_config = scheduler.model_config
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
        self.draft_attn_backend = None
        self.cuda_graph_runner = None
        self._init_attention_backend()
        self._init_cuda_graphs()

    @property
    def model_runner(self):
        """Alias for EAGLEDraftCudaGraphRunner, which reads ``eagle_worker.model_runner``."""
        return self.draft_model_runner

    def draft_forward(self, forward_batch: ForwardBatch):
        """Alias for CUDA-graph capture, which calls ``eagle_worker.draft_forward``."""
        return self._draft_forward(forward_batch)

    def _init_attention_backend(self) -> None:
        if self.speculative_num_steps <= 1:
            return
        try:
            factory = DraftBackendFactory(
                self.server_args,
                self.draft_model_runner,
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_attn_backend = factory.create_decode_backend()
        except Exception as e:
            logger.warning("[SR] tree draft attention backend unavailable: %s", e)
            self.draft_attn_backend = None

    def _init_cuda_graphs(self) -> None:
        """Capture v1 EAGLE draft-tree graphs. Skip draft-extend graphs (not used by SR)."""
        self.cuda_graph_runner = None
        if getattr(self.server_args, "disable_cuda_graph", False):
            return
        if self.speculative_num_steps <= 1 or self.draft_attn_backend is None:
            return
        prev_draft_backend = getattr(self.draft_model_runner, "draft_attn_backend", None)
        try:
            from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
                EAGLEDraftCudaGraphRunner,
            )

            backend = device_backend_key(self.device)
            # Import the NPU runner only on NPU so CUDA hosts never need torch_npu.
            if backend == "npu":
                from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
                    EAGLEDraftNpuGraphRunner,
                )

                runner_cls = EAGLEDraftNpuGraphRunner
            else:
                runner_cls = EAGLEDraftCudaGraphRunner
            self.draft_model_runner.draft_attn_backend = self.draft_attn_backend
            logger.info("[SR] Capture tree draft graph begin (backend=%s).", backend)
            self.cuda_graph_runner = runner_cls(self)
            logger.info("[SR] Capture tree draft graph end.")
        except Exception as e:
            logger.warning("[SR] tree draft graph capture failed: %s", e)
            self.cuda_graph_runner = None
        finally:
            self.draft_model_runner.draft_attn_backend = prev_draft_backend

    def expand_batch(self, reqs: List["Req"]) -> List[SRTreeWindow]:
        """Fused tree expand for every req that has a pool slot and a seed."""
        empty: SRTreeWindow = ([], None, None)
        if not reqs:
            return []
        windows: List[SRTreeWindow] = [empty] * len(reqs)
        keep: List["Req"] = []
        keep_idx: List[int] = []
        for i, req in enumerate(reqs):
            if req.req_pool_idx is None:
                continue
            if getattr(req, "sr_tree_seed", None) is None:
                continue
            keep.append(req)
            keep_idx.append(i)
        if not keep:
            return windows
        try:
            parent_list, top_scores_index, draft_tokens = self._expand_tree(
                keep, *self._stack_seeds(keep)
            )
        except Exception as e:
            logger.warning(
                "[SR] tree expand_batch failed for %s: %s; falling back per-req",
                [r.rid for r in keep],
                e,
            )
            for j, req in enumerate(keep):
                windows[keep_idx[j]] = self._expand_one(req)
            return windows
        for j, idx in enumerate(keep_idx):
            windows[idx] = (
                draft_tokens[j].detach().to("cpu").tolist(),
                parent_list[j].detach().to("cpu").tolist(),
                top_scores_index[j].detach().to("cpu").tolist(),
            )
        return windows

    def _expand_one(self, req: "Req") -> SRTreeWindow:
        empty: SRTreeWindow = ([], None, None)
        seed = getattr(req, "sr_tree_seed", None)
        if req.req_pool_idx is None or seed is None:
            return empty
        try:
            parent_list, top_scores_index, draft_tokens = self._expand_tree(
                [req], *self._stack_seeds([req])
            )
        except Exception as e:
            logger.warning("[SR] tree expand failed for %s: %s", req.rid, e)
            return empty
        return (
            draft_tokens[0].detach().to("cpu").tolist(),
            parent_list[0].detach().to("cpu").tolist(),
            top_scores_index[0].detach().to("cpu").tolist(),
        )

    def _stack_seeds(
        self, reqs: List["Req"]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ps: List[torch.Tensor] = []
        ixs: List[torch.Tensor] = []
        hs: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        for req in reqs:
            topk_p, topk_index, hidden_states, verified_id = req.sr_tree_seed
            if topk_p.dim() == 1:
                topk_p = topk_p.unsqueeze(0)
            if topk_index.dim() == 1:
                topk_index = topk_index.unsqueeze(0)
            if hidden_states.dim() == 1:
                hidden_states = hidden_states.unsqueeze(0)
            elif hidden_states.dim() == 3:
                hidden_states = hidden_states[:, -1, :]
            if verified_id.dim() == 0:
                verified_id = verified_id.unsqueeze(0)
            ps.append(topk_p[:1])
            ixs.append(topk_index[:1])
            hs.append(hidden_states[:1])
            vs.append(verified_id.reshape(-1)[:1])
        return (
            torch.cat(ps, dim=0),
            torch.cat(ixs, dim=0),
            torch.cat(hs, dim=0),
            torch.cat(vs, dim=0),
        )

    def _expand_tree(
        self,
        reqs: List["Req"],
        topk_p: torch.Tensor,
        topk_index: torch.Tensor,
        hidden_states: torch.Tensor,
        verified_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scheduler = self.scheduler
        batch = scheduler._sr_make_decode_batch(reqs)
        spec_info = EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
            verified_id=verified_id,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        batch.spec_info = spec_info
        batch.return_hidden_states = False
        token_to_kv_pool_state_backup = self._alloc_tree_kv(batch)
        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        prev_draft_backend = getattr(self.draft_model_runner, "draft_attn_backend", None)
        try:
            if self.draft_attn_backend is not None:
                self.draft_model_runner.draft_attn_backend = self.draft_attn_backend
            forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.draft_model_runner
            )
            can_cuda_graph = (
                self.cuda_graph_runner is not None
                and self.cuda_graph_runner.can_run(forward_batch)
            )
            if can_cuda_graph:
                parent_list, top_scores_index, draft_tokens = (
                    self.cuda_graph_runner.replay(forward_batch)
                )
            else:
                if (
                    self.draft_attn_backend is not None
                    and self.speculative_num_steps > 1
                    and not forward_batch.forward_mode.is_idle()
                ):
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                parent_list, top_scores_index, draft_tokens = self._draft_forward(
                    forward_batch
                )
        finally:
            self.draft_model_runner.draft_attn_backend = prev_draft_backend
            self.token_to_kv_pool_allocator.restore_state(
                token_to_kv_pool_state_backup
            )
        return parent_list, top_scores_index, draft_tokens

    def _alloc_tree_kv(self, batch: "ScheduleBatch"):
        from sglang.srt.speculative.eagle_worker import (
            get_last_loc_large_page_size_top_k_1,
        )

        num_seqs = batch.batch_size()
        token_to_kv_pool_state_backup = None
        if self.page_size == 1:
            alloc_len = self.speculative_num_steps * self.topk
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
                batch.tree_cache,
                num_seqs * alloc_len,
                backup_state=True,
            )
            duplicate_cache_len = 0
            source_cache_loc = target_cache_loc = last_page_lens_cumsum = None
        else:
            if self.topk == 1:
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                seq_lens_cpu = batch.seq_lens_cpu + self.speculative_num_steps
                extend_num_tokens = num_seqs * self.speculative_num_steps
                last_page_lens = None
            else:
                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.num_new_pages_per_topk,
                    self.extend_lens,
                    last_page_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                    self.topk,
                    self.page_size,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                last_page_lens_cpu = prefix_lens_cpu % self.page_size
                num_new_pages_per_topk = (
                    last_page_lens_cpu + self.speculative_num_steps + self.page_size - 1
                ) // self.page_size
                seq_lens_cpu = (
                    prefix_lens_cpu // self.page_size * self.page_size
                    + num_new_pages_per_topk * (self.page_size * self.topk)
                )
                extend_num_tokens = torch.sum((seq_lens_cpu - prefix_lens_cpu)).item()

            out_cache_loc, token_to_kv_pool_state_backup = (
                alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )
            if self.page_size > 1 and self.topk > 1:
                last_page_lens_cpu = batch.seq_lens_cpu % self.page_size
                last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
                duplicate_cache_len = torch.sum(last_page_lens_cpu).item() * (
                    self.topk - 1
                )
                target_cache_loc = torch.zeros(
                    duplicate_cache_len, dtype=torch.int32, device=self.device
                )
                source_cache_loc = torch.zeros(
                    duplicate_cache_len, dtype=torch.int32, device=self.device
                )
            else:
                duplicate_cache_len = 0
                source_cache_loc = target_cache_loc = last_page_lens_cumsum = None

        try:
            assign_draft_cache_locs[(num_seqs,)](
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                batch.seq_lens,
                self.extend_lens,
                self.num_new_pages_per_topk,
                out_cache_loc,
                source_cache_loc,
                target_cache_loc,
                last_page_lens_cumsum,
                duplicate_cache_len,
                batch.req_to_token_pool.req_to_token.shape[1],
                self.topk,
                self.speculative_num_steps,
                self.page_size,
                next_power_of_2(num_seqs),
                next_power_of_2(self.speculative_num_steps + self.page_size),
            )
            if self.page_size > 1 and self.topk > 1:
                if duplicate_cache_len > 0:
                    self.draft_model_runner.token_to_kv_pool.move_kv_cache(
                        target_cache_loc, source_cache_loc
                    )
                # Remove padded slots (including last_page_len == 0).
                out_cache_loc = out_cache_loc[
                    : num_seqs * self.topk * self.speculative_num_steps
                ]
            batch.out_cache_loc = out_cache_loc
            batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
            batch.spec_info.positions = batch.seq_lens.repeat_interleave(
                self.topk, dim=0
            )
            return token_to_kv_pool_state_backup
        except Exception:
            if token_to_kv_pool_state_backup is not None:
                self.token_to_kv_pool_allocator.restore_state(
                    token_to_kv_pool_state_backup
                )
            raise

    def _draft_forward(self, forward_batch: ForwardBatch):
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        maybe_detect_nan(topk_p, "SR draft_forward: NaN in seed topk_p")
        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])
            if i == self.speculative_num_steps - 1:
                break
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            advance_tree_draft_positions(
                forward_batch.positions,
                getattr(forward_batch, "mrope_positions", None),
            )
            if self.draft_attn_backend is not None:
                forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states
            # Tree steps are bs*topk, not 1-token AR. DECODE graphs must not replay
            # here (capture would hit unset raw_num_token; runtime layout is wrong).
            prev_graph_runner = getattr(self.draft_model_runner, "graph_runner", None)
            self.draft_model_runner.graph_runner = None
            try:
                logits_output = self.draft_model_runner.forward(
                    forward_batch, skip_attn_backend_init=True
                ).logits_output
            finally:
                self.draft_model_runner.graph_runner = prev_graph_runner
            maybe_detect_nan(logits_output.next_token_logits, f"SR draft_forward step {i}")
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"SR draft_forward step {i}: topk_index OOB",
            )
            hidden_states = logits_output.hidden_states
        return organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )
