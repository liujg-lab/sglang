"""STANDALONE-style top-k tree expansion on the remote Draft process.

Uses the Draft server's existing TpModelWorker (no second weight load).
Linear KV stays the Target committed prefix; tree KV is ephemeral.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
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
    NpuGraphPreparationError,
    NpuGraphReplaySubmittedError,
    assign_draft_cache_locs,
    build_paged_draft_cache_locs,
    device_backend_key,
    fast_topk,
    get_last_loc_large_page_size_large_top_k,
    maybe_detect_nan,
    maybe_detect_oob,
    paged_tree_mapping_fits,
    select_top_k_tokens,
    split_draft_cache_locs,
)
from sglang.srt.speculative.standalone_remote.sr_align import is_device_context_error
from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
    advance_tree_draft_positions_for_step,
    copy_mha_kv_by_slot,
    copy_paged_kv_buffer_by_slot,
)
from sglang.srt.utils import next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)

SRTreeWindow = Tuple[List[int], Optional[List[int]], Optional[List[int]]]


def _as_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


def _wait_d2h_event(device, host: dict) -> None:
    """Block until the batched D2H into ``host`` is visible on CPU."""
    host["event"] = None
    if device is None:
        return
    dev_type = getattr(device, "type", None) or str(device)
    if dev_type == "cpu":
        return
    try:
        module = torch.get_device_module(dev_type)
        event = module.Event()
        event.record()
        event.synchronize()
        host["event"] = event
    except Exception:
        # CPU tests and missing device modules still have finished copy_ data.
        return


def _ensure_host_staging_slot(slots, idx: int, bs: int, token_w: int, parent_w: int, index_w: int):
    host = slots[idx]
    token_w = max(int(token_w), 0)
    parent_w = max(int(parent_w), 0)
    index_w = max(int(index_w), 0)
    bs = max(int(bs), 1)
    need_new = host is None
    if not need_new:
        need_new = (
            host["tokens"].shape[0] < bs
            or host["tokens"].shape[1] < token_w
            or host["parents"].shape[1] < parent_w
            or host["indices"].shape[1] < index_w
        )
    if need_new:
        host = {
            "tokens": torch.zeros((bs, max(token_w, 1)), dtype=torch.int64),
            "parents": torch.zeros((bs, max(parent_w, 1)), dtype=torch.int64),
            "indices": torch.zeros((bs, max(index_w, 1)), dtype=torch.int64),
            "event": None,
            "in_use": False,
        }
        slots[idx] = host
    return host


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
        self._tree_failure_counts = {}
        self.tree_graph_capture_succeeded = False
        self.tree_graph_disabled_reason = None
        # Two host stagings so a later replay cannot overwrite a D2H that has
        # not yet been split into RPC lists.
        self._host_staging = [None, None]
        self._staging_index = 0
        self._init_attention_backend()
        self._init_cuda_graphs()

    @property
    def model_runner(self):
        """Alias for EAGLEDraftCudaGraphRunner, which reads ``eagle_worker.model_runner``."""
        return self.draft_model_runner

    def draft_forward(self, forward_batch: ForwardBatch):
        """Alias for CUDA-graph capture, which calls ``eagle_worker.draft_forward``."""
        return self._draft_forward(forward_batch)

    def _log_tree_failure(self, stage: str, exc: Exception) -> None:
        """One traceback per failure signature; repeated failures report totals."""
        # NPU errors append timestamps/PIDs on subsequent lines. Do not include
        # request IDs in this key, otherwise every new request logs a traceback.
        message = str(exc).splitlines()
        summary = message[0] if message else type(exc).__name__
        key = (stage, type(exc), summary)
        count = self._tree_failure_counts.get(key, 0) + 1
        self._tree_failure_counts[key] = count
        if count != 1 and count % 32:
            return
        backend = self.draft_attn_backend
        inners = getattr(backend, "attn_backends", None)
        if inners:
            backend = inners[0]
        metadata = getattr(getattr(backend, "forward_metadata", None), "tree_shared", None)
        dtypes = (
            {name: str(value.dtype) for name, value in vars(metadata).items()}
            if metadata is not None
            else {}
        )
        dtypes["req_to_token"] = str(self.req_to_token_pool.req_to_token.dtype)
        logger.warning(
            "[SR] tree failure stage=%s implementation=%s metadata_dtypes=%s "
            "occurrences=%s error=%s",
            stage,
            getattr(backend, "tree_attention_impl", type(backend).__name__),
            dtypes,
            count,
            summary,
            exc_info=(type(exc), exc, exc.__traceback__) if count == 1 else None,
        )

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
        self.tree_graph_capture_succeeded = False
        self.tree_graph_disabled_reason = None
        if getattr(self.server_args, "disable_cuda_graph", False):
            self.tree_graph_disabled_reason = "disabled by configuration"
            return
        if self.speculative_num_steps <= 1 or self.draft_attn_backend is None:
            self.tree_graph_disabled_reason = "no multi-step attention backend"
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
            reason = getattr(
                self.cuda_graph_runner, "tree_graph_disabled_reason", None
            )
            graphs = getattr(self.cuda_graph_runner, "graphs", None)
            if reason or not graphs:
                self.tree_graph_disabled_reason = reason or "no graphs"
                logger.warning(
                    "[SR] tree draft graphs disabled after capture: %s "
                    "(tree_graph_replay_count=%s tree_eager_fallback_count=%s)",
                    reason or "no graphs",
                    getattr(self.cuda_graph_runner, "tree_graph_replay_count", 0),
                    getattr(self.cuda_graph_runner, "tree_eager_fallback_count", 0),
                )
                self.cuda_graph_runner = None
            else:
                self.tree_graph_capture_succeeded = True
            logger.info("[SR] Capture tree draft graph end.")
        except NpuGraphReplaySubmittedError:
            raise
        except Exception as e:
            if is_device_context_error(e):
                raise
            self._log_tree_failure("capture", e)
            self.tree_graph_disabled_reason = "capture failed; see tree failure traceback"
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
        except NpuGraphReplaySubmittedError:
            raise
        except Exception as e:
            if is_device_context_error(e):
                raise
            self._log_tree_failure("expand_batch", e)
            # Retrying the very same single-request batch cannot isolate a bad
            # request. Preserve the empty-window fallback without a second run.
            # Do not pack the failed batch's graph outputs before per-req replay.
            if len(keep) == 1:
                return windows
            for j, req in enumerate(keep):
                windows[keep_idx[j]] = self._expand_one(req)
            return windows
        metrics = getattr(self.scheduler, "_sr_round_metrics", None)
        # Batched D2H waits for the asynchronously submitted tree forward.
        with metrics.phase("tree_result_wait_pack") if metrics else nullcontext():
            self._pack_tree_windows(
                parent_list, top_scores_index, draft_tokens, keep_idx, windows
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
        except NpuGraphReplaySubmittedError:
            raise
        except Exception as e:
            if is_device_context_error(e):
                raise
            self._log_tree_failure("expand_one", e)
            return empty
        windows: List[SRTreeWindow] = [empty]
        metrics = getattr(self.scheduler, "_sr_round_metrics", None)
        with metrics.phase("tree_result_wait_pack") if metrics else nullcontext():
            self._pack_tree_windows(
                parent_list, top_scores_index, draft_tokens, [0], windows
            )
        return windows[0]

    def _pack_tree_windows(
        self,
        parent_list,
        top_scores_index,
        draft_tokens,
        keep_idx: List[int],
        windows: List[SRTreeWindow],
    ) -> None:
        """One D2H per tree tensor, then CPU row splits into RPC lists."""
        tokens_cpu, parents_cpu, indices_cpu = self._d2h_tree_outputs(
            draft_tokens, parent_list, top_scores_index
        )
        for j, idx in enumerate(keep_idx):
            windows[idx] = (
                tokens_cpu[j].tolist(),
                parents_cpu[j].tolist(),
                indices_cpu[j].tolist(),
            )

    def _d2h_tree_outputs(self, draft_tokens, parent_list, top_scores_index):
        tensors = (draft_tokens, parent_list, top_scores_index)
        if all(isinstance(t, torch.Tensor) for t in tensors):
            tokens = _as_2d(draft_tokens.detach())
            parents = _as_2d(parent_list.detach())
            indices = _as_2d(top_scores_index.detach())
            n = int(tokens.shape[0])
            token_w = int(tokens.shape[1]) if tokens.dim() > 1 else 1
            parent_w = int(parents.shape[1]) if parents.dim() > 1 else 1
            index_w = int(indices.shape[1]) if indices.dim() > 1 else 1
            host = self._acquire_host_staging(n, token_w, parent_w, index_w)
            non_blocking = tokens.device.type != "cpu"
            host["tokens"][:n, :token_w].copy_(tokens, non_blocking=non_blocking)
            host["parents"][:n, :parent_w].copy_(parents, non_blocking=non_blocking)
            host["indices"][:n, :index_w].copy_(indices, non_blocking=non_blocking)
            _wait_d2h_event(tokens.device, host)
            host["in_use"] = False
            return (
                host["tokens"][:n, :token_w],
                host["parents"][:n, :parent_w],
                host["indices"][:n, :index_w],
            )
        # Test doubles expose detach().to("cpu") on the whole batch object.
        return (
            draft_tokens.detach().to("cpu"),
            parent_list.detach().to("cpu"),
            top_scores_index.detach().to("cpu"),
        )

    def _acquire_host_staging(self, bs: int, token_w: int, parent_w: int, index_w: int):
        slots = getattr(self, "_host_staging", None)
        if not slots:
            self._host_staging = [None, None]
            self._staging_index = 0
            slots = self._host_staging
        idx = int(getattr(self, "_staging_index", 0) or 0) % 2
        other = slots[idx]
        if other is not None and other.get("in_use"):
            event = other.get("event")
            if event is not None:
                event.synchronize()
            other["in_use"] = False
        host = _ensure_host_staging_slot(slots, idx, bs, token_w, parent_w, index_w)
        host["in_use"] = True
        self._staging_index = (idx + 1) % 2
        return host

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
        graph_submitted = False
        metrics = getattr(scheduler, "_sr_round_metrics", None)
        t_prep = time.perf_counter()
        try:
            if self.draft_attn_backend is not None:
                self.draft_model_runner.draft_attn_backend = self.draft_attn_backend
            forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.draft_model_runner
            )
            prep_s = time.perf_counter() - t_prep
            can_cuda_graph = (
                self.cuda_graph_runner is not None
                and self.cuda_graph_runner.can_run(forward_batch)
            )
            t_exec = time.perf_counter()
            # End the device event before returning tensors for D2H/packing.
            # No synchronize here: SRRoundMetrics polls completed samples later.
            with (
                metrics.phase("tree_forward", device=True) if metrics else nullcontext()
            ):
                if can_cuda_graph:
                    try:
                        parent_list, top_scores_index, draft_tokens = (
                            self.cuda_graph_runner.replay(forward_batch)
                        )
                    except NpuGraphReplaySubmittedError:
                        self.cuda_graph_runner = None
                        graph_submitted = True
                        raise
                    except NpuGraphPreparationError as e:
                        logger.warning(
                            "[SR] tree draft graph replay prep failed: %s; "
                            "falling back to eager",
                            e,
                        )
                        if getattr(e, "scope", "graph") == "format":
                            self.cuda_graph_runner = None
                        else:
                            runner = self.cuda_graph_runner
                            if runner is not None:
                                runner.tree_eager_fallback_count = (
                                    getattr(runner, "tree_eager_fallback_count", 0) + 1
                                )
                        can_cuda_graph = False
                if not can_cuda_graph:
                    if (
                        self.draft_attn_backend is not None
                        and self.speculative_num_steps > 1
                        and not forward_batch.forward_mode.is_idle()
                    ):
                        self.draft_attn_backend.init_forward_metadata(forward_batch)
                    parent_list, top_scores_index, draft_tokens = self._draft_forward(
                        forward_batch
                    )
            exec_s = time.perf_counter() - t_exec
        finally:
            self.draft_model_runner.draft_attn_backend = prev_draft_backend
            if not graph_submitted:
                self.token_to_kv_pool_allocator.restore_state(
                    token_to_kv_pool_state_backup
                )
        runner = self.cuda_graph_runner
        replay_n = getattr(runner, "tree_graph_replay_count", 0) if runner else 0
        if replay_n <= 1 or replay_n % 8 == 0:
            logger.info(
                "[SR] tree draft timings: prepare_host=%.3fs "
                "forward_call_host=%.3fs graph=%s "
                "replay=%s eager_fallback=%s",
                prep_s,
                exec_s,
                can_cuda_graph,
                replay_n,
                getattr(runner, "tree_eager_fallback_count", 0) if runner else 0,
            )
        return parent_list, top_scores_index, draft_tokens

    def _alloc_tree_kv(self, batch: "ScheduleBatch"):
        from sglang.srt.speculative.eagle_worker import (
            get_last_loc_large_page_size_top_k_1,
        )

        num_seqs = batch.batch_size()
        token_to_kv_pool_state_backup = None
        pool_len = batch.req_to_token_pool.req_to_token.shape[1]
        if not paged_tree_mapping_fits(
            batch.seq_lens,
            self.page_size,
            self.topk,
            self.speculative_num_steps,
            pool_len,
        ):
            raise RuntimeError(
                "tree draft mapping exceeds req_to_token width: "
                f"pool_len={pool_len} page_size={self.page_size} "
                f"topk={self.topk} steps={self.speculative_num_steps}"
            )
        if self.page_size == 1:
            alloc_len = self.speculative_num_steps * self.topk
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
                batch.tree_cache,
                num_seqs * alloc_len,
                backup_state=True,
            )
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
            else:
                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.num_new_pages_per_topk,
                    self.extend_lens,
                    _last_page_lens,
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
        try:
            raw_cache_loc, draft_cache_loc = split_draft_cache_locs(
                out_cache_loc,
                num_seqs,
                self.topk,
                self.speculative_num_steps,
                self.page_size,
            )
            assign_draft_cache_locs[(num_seqs,)](
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                batch.seq_lens,
                self.extend_lens,
                raw_cache_loc,
                batch.req_to_token_pool.req_to_token.shape[1],
                self.topk,
                self.speculative_num_steps,
                self.page_size,
                next_power_of_2(num_seqs),
            )
            if self.page_size > 1 and self.topk > 1:
                # Compact draft slots in PyTorch (former Triton Part 3).
                # Last-page KV duplication (former Part 2 / move_kv_cache) is
                # only required for page-level attention. Token-level slot
                # gather reads the original prefix slots, so skip the copy.
                draft_cache_loc = build_paged_draft_cache_locs(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.num_new_pages_per_topk,
                    self.topk,
                    self.speculative_num_steps,
                    self.page_size,
                )
            batch.out_cache_loc = draft_cache_loc
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
            input_ids, hidden_states, scores, tree_info, parent_rows = (
                select_top_k_tokens(
                    i, topk_p, topk_index, hidden_states, scores, self.topk
                )
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])
            if i == self.speculative_num_steps - 1:
                break
            if i > 0 and self.topk > 1 and parent_rows is not None:
                self._remap_tree_kv_to_parents(out_cache_loc, parent_rows, i)
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            advance_tree_draft_positions_for_step(
                i,
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

    def _remap_tree_kv_to_parents(
        self,
        out_cache_loc: torch.Tensor,
        parent_rows: torch.Tensor,
        n_prev_steps: int,
    ) -> None:
        kv_pool = getattr(self.draft_model_runner, "token_to_kv_pool", None)
        if kv_pool is None:
            return
        for s in range(n_prev_steps):
            src = out_cache_loc[s][parent_rows]
            tgt = out_cache_loc[s]
            self._copy_tree_kv_slots(kv_pool, src, tgt)

    def _copy_tree_kv_slots(self, kv_pool, src: torch.Tensor, tgt: torch.Tensor) -> None:
        kv_buffer = getattr(kv_pool, "kv_buffer", None)
        if torch.is_tensor(kv_buffer) and kv_buffer.dim() == 6:
            copy_paged_kv_buffer_by_slot(kv_buffer, src, tgt)
            return
        mover = getattr(kv_pool, "move_kv_cache", None)
        if callable(mover) and getattr(kv_pool, "_kv_copy_config", None) is not None:
            mover(tgt, src)
            return
        k_buffer = getattr(kv_pool, "k_buffer", None)
        v_buffer = getattr(kv_pool, "v_buffer", None)
        if k_buffer is not None:
            copy_mha_kv_by_slot(
                k_buffer,
                v_buffer,
                src,
                tgt,
                getattr(kv_pool, "index_k_buffer", None),
            )
            return
        if isinstance(kv_buffer, (list, tuple)):
            copy_mha_kv_by_slot(kv_buffer, None, src, tgt)
