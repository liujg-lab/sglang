import logging
import time
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.sampler import SamplingBatchInfo
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    Req,
    ScheduleBatch,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.standalone_remote.drafter.sr_draft_state import (
    SRDraftState,
    SRDraftStateManager,
    SRWindow,
)
from sglang.srt.speculative.standalone_remote.sr_align import (
    DEFAULT_MAX_INGEST_DECODE_STEPS,
    DraftDecision,
    broadcast_sr_obj,
    classify_prefix_alignment,
    committed_tail_not_in_kv,
    decide_draft_action,
    draft_needed_max_new_tokens,
    find_fork_point,
    ingest_active_indices,
    plan_committed_ingest,
    replay_grammar_from_committed,
)
from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
    slice_decode_batch_row,
)
from sglang.srt.speculative.standalone_remote.sr_kv_rollbacker import SRKVRollbacker
from sglang.srt.speculative.standalone_remote.sr_mm_payload import (
    SRMMPayload,
    release_mm_resources,
    reset_mm_mrope,
)
from sglang.srt.speculative.standalone_remote.sr_protocol import (
    SRAction,
    SRBatchReply,
    SRBatchRequest,
    SRDraftReply,
    SRDraftRequest,
    SRReplyStatus,
)
from sglang.srt.speculative.standalone_remote.sr_transport import (
    SRDraftServer,
    make_transport_from_server_args,
)
from sglang.srt.speculative.standalone_remote.drafter.sr_tree_drafter import (
    SRTreeDrafter,
)
from sglang.srt.utils import DynamicGradMode

logger = logging.getLogger(__name__)


def _fix_sampling_params_stop_strs(sp) -> None:
    if not hasattr(sp, "stop_strs") or sp.stop_strs is None:
        sp.stop_strs = []
    elif isinstance(sp.stop_strs, str):
        sp.stop_strs = [sp.stop_strs]
    if not hasattr(sp, "stop_regex_strs") or sp.stop_regex_strs is None:
        sp.stop_regex_strs = []
    elif isinstance(sp.stop_regex_strs, str):
        sp.stop_regex_strs = [sp.stop_regex_strs]


def _padded_ids_mismatch(a: List[int], b: List[int]) -> Optional[int]:
    if len(a) != len(b):
        return 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def _sr_is_device_context_error(exc: BaseException) -> bool:
    """True when the accelerator context is already poisoned (do not keep running)."""
    name = type(exc).__name__
    if name in ("AcceleratorError", "CUDAError", "NPUError", "XPUError"):
        return True
    accel = getattr(torch, "AcceleratorError", None)
    if accel is not None and isinstance(exc, accel):
        return True
    msg = str(exc).lower()
    return (
        "illegal memory access" in msg
        or "cudaerrorillegaladdress" in msg
        or "npu error" in msg
        or ("ascend" in msg and "illegal" in msg)
    )


# Backward-compatible alias.
_sr_is_cuda_context_error = _sr_is_device_context_error


class StandaloneRemoteDraftSchedulerMixin:
    def _init_sr_draft(self) -> None:
        ttl = float(
            getattr(self.server_args, "standalone_remote_draft_ttl_s", 60.0) or 0.0
        )
        self.sr_state = SRDraftStateManager(timeout_threshold=ttl)
        self.sr_kv = SRKVRollbacker(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            tree_cache=self.tree_cache,
            page_size=self.server_args.page_size or 1,
            tp_rank=self.tp_rank,
        )
        self.sr_server: Optional[SRDraftServer] = None
        if self.tp_size == 1 or self.tp_rank == 0:
            server = make_transport_from_server_args(self.server_args)
            assert isinstance(server, SRDraftServer)
            self.sr_server = server
        self.sr_waiting: List[Req] = []
        self.draft_paused_reqs: List[Req] = []
        self.paused_reqs = self.draft_paused_reqs
        self.sr_tree_drafter: Optional[SRTreeDrafter] = None
        topk = int(self.server_args.speculative_eagle_topk or 1)
        if topk > 1:
            self.sr_tree_drafter = SRTreeDrafter(self)
        logger.info(
            "[SR] Draft scheduler ready (tree=%s topk=%s)",
            self.sr_tree_drafter is not None,
            topk,
        )

    def _sr_tree_mode(self) -> bool:
        return getattr(self, "sr_tree_drafter", None) is not None

    def _sr_enable_tree_seed_hidden(self, batch: ScheduleBatch) -> None:
        """Request last-token hidden for tree seed without HTTP FULL capture."""
        batch.return_hidden_states = False
        batch.capture_hidden_mode = CaptureHiddenMode.LAST

    def _sr_is_http_req(self, req: Req) -> bool:
        """HTTP generate reqs are unmarked; Target RPC reqs set is_sr_draft."""
        return getattr(req, "is_sr_draft", False) is not True

    def _sr_http_alive(self) -> bool:
        waiting = getattr(self, "waiting_queue", None) or []
        paused = getattr(self, "draft_paused_reqs", None) or []
        running = getattr(self, "running_batch", None)
        running_reqs = (
            list(running.reqs)
            if running is not None and not running.is_empty()
            else []
        )
        chunked = getattr(self, "chunked_req", None)
        candidates = list(waiting) + list(paused) + running_reqs
        if chunked is not None:
            candidates.append(chunked)
        return any(self._sr_is_http_req(r) for r in candidates if r is not None)

    def _sr_draft_busy(self) -> bool:
        """True when HTTP / leftover GPU work is too large to start a new RPC window.

        Any alive HTTP req would REJECT every Target RPC during warmup, so this
        only trips when the running batch (or its HTTP subset) exceeds the cap.
        """
        cap = int(
            getattr(
                getattr(self, "server_args", None),
                "standalone_remote_max_batch_size",
                32,
            )
            or 32
        )
        running = getattr(self, "running_batch", None)
        if running is None:
            return False
        is_empty = getattr(running, "is_empty", None)
        if callable(is_empty) and is_empty() is True:
            return False
        bsz = getattr(running, "batch_size", 0)
        bsz = bsz() if callable(bsz) else bsz
        try:
            bsz = int(bsz or 0)
        except (TypeError, ValueError):
            bsz = 0
        if bsz > cap:
            return True
        reqs = getattr(running, "reqs", None) or []
        http_n = sum(1 for r in reqs if r is not None and self._sr_is_http_req(r))
        return http_n > cap

    def _sr_cleanup_stale_drafts(self, keep_rids=None) -> None:
        manager = getattr(self, "sr_state", None)
        if manager is None:
            return
        popped = manager.cleanup_stale_states(keep_rids=keep_rids)
        for state in popped:
            logger.info(
                "[SR] Draft TTL expired rid=%s idle=%.1fs",
                state.req_id,
                time.time() - state.last_updated_time,
            )
            self._sr_finish_rid(state.req_id, state=state)

    def _sr_wipe_all(self) -> None:
        """Drop previous-session Draft RPC state. Target flush_cache never
        reaches this process; a new session_id is the only signal.

        HTTP generate reqs on this Draft process must survive. Full radix /
        KV / embedding_cache reset only runs when no HTTP req is alive
        (needed for VL leftover 128 vs 64 embeddings).
        """
        for state in self.sr_state.clear():
            req = state.req_object
            if req is None:
                continue
            self._sr_remove_req(req)
            release_mm_resources(req.multimodal_inputs)
            req.multimodal_inputs = None
            req.req_pool_idx = None
            if not req.finished():
                req.to_abort = True
                req.finished_reason = FINISH_ABORT("Target session reset")
        self.sr_waiting = [
            r for r in self.sr_waiting if self._sr_is_http_req(r)
        ]
        if getattr(self, "draft_paused_reqs", None) is not None:
            self.draft_paused_reqs = [
                r for r in self.draft_paused_reqs if self._sr_is_http_req(r)
            ]
        http_alive = self._sr_http_alive()
        if http_alive:
            self.cur_batch = None
            self.last_batch = None
        else:
            self._sr_reset_scheduler_caches()
        if self.sr_server is not None:
            self.sr_server.last_rpc_seq = -1
            drain = getattr(self.sr_server, "drain", None)
            if callable(drain):
                drain()
        logger.info(
            "[SR] Draft wiped RPC state for new session (http_alive=%s)",
            http_alive,
        )

    def _sr_reset_scheduler_caches(self) -> None:
        waiting = getattr(self, "waiting_queue", None)
        if waiting is not None:
            waiting.clear()
        running = getattr(self, "running_batch", None)
        if running is not None and not running.is_empty():
            running.filter_batch(keep_indices=[])
            running.batch_is_full = False
        self.cur_batch = None
        self.last_batch = None
        if getattr(self, "chunked_req", None) is not None:
            self.chunked_req = None
        tree = getattr(self, "tree_cache", None)
        if tree is not None and hasattr(tree, "reset"):
            tree.reset()
        pool = getattr(self, "req_to_token_pool", None)
        if pool is not None and hasattr(pool, "clear"):
            pool.clear()
        alloc = getattr(self, "token_to_kv_pool_allocator", None)
        if alloc is not None and hasattr(alloc, "clear"):
            alloc.clear()
        gm = getattr(self, "grammar_manager", None)
        if gm is not None and hasattr(gm, "clear"):
            gm.clear()
        try:
            from sglang.srt.managers.mm_utils import embedding_cache

            if embedding_cache is not None:
                embedding_cache.clear()
        except Exception:
            pass

    def _sr_remove_req(self, req: Req) -> None:
        if req in self.sr_waiting:
            self.sr_waiting.remove(req)
        if req in self.draft_paused_reqs:
            self.draft_paused_reqs.remove(req)
        waiting = getattr(self, "waiting_queue", None)
        if waiting is not None and req in waiting:
            waiting.remove(req)
        running = getattr(self, "running_batch", None)
        if running is not None and not running.is_empty():
            keep = [i for i, r in enumerate(running.reqs) if r is not req]
            if len(keep) != len(running.reqs):
                running.filter_batch(keep_indices=keep)

    def _sr_pause_req(self, req: Req) -> None:
        req.draft_is_paused = True
        if req not in self.draft_paused_reqs:
            self.draft_paused_reqs.append(req)
        waiting = getattr(self, "waiting_queue", None)
        if waiting is not None and req in waiting:
            waiting.remove(req)
        running = getattr(self, "running_batch", None)
        if running is not None and not running.is_empty() and req in running.reqs:
            keep = [i for i, r in enumerate(running.reqs) if r is not req]
            running.filter_batch(keep_indices=keep)

    def _sr_resume_req(self, req: Req) -> None:
        req.draft_is_paused = False
        if req in self.draft_paused_reqs:
            self.draft_paused_reqs.remove(req)

    def _sr_resume_http_reqs(self) -> None:
        """Requeue HTTP reqs parked by isolate_need after an RPC tick."""
        paused = getattr(self, "draft_paused_reqs", None)
        if not paused:
            return
        to_resume = [r for r in list(paused) if self._sr_is_http_req(r)]
        waiting = getattr(self, "waiting_queue", None)
        if waiting is None:
            waiting = []
            self.waiting_queue = waiting
        for req in to_resume:
            self._sr_resume_req(req)
            if req.req_pool_idx is None:
                if req not in waiting:
                    waiting.append(req)
            else:
                self._sr_park_in_running_many([req])

    def _sr_park_sr_reqs(self) -> None:
        """Keep Target RPC reqs out of the HTTP get_next_batch_to_run path."""
        running = getattr(self, "running_batch", None)
        if running is not None and not running.is_empty():
            keep: List[int] = []
            to_pause: List[Req] = []
            for i, r in enumerate(running.reqs):
                if getattr(r, "is_sr_draft", False) is True:
                    to_pause.append(r)
                else:
                    keep.append(i)
            if len(keep) != len(running.reqs):
                running.filter_batch(keep_indices=keep)
            for r in to_pause:
                r.draft_is_paused = True
                if r not in self.draft_paused_reqs:
                    self.draft_paused_reqs.append(r)
        waiting = getattr(self, "waiting_queue", None)
        if waiting:
            stay = []
            for r in list(waiting):
                if getattr(r, "is_sr_draft", False) is True:
                    r.draft_is_paused = True
                    if r not in self.draft_paused_reqs:
                        self.draft_paused_reqs.append(r)
                else:
                    stay.append(r)
            waiting[:] = stay
        last = getattr(self, "last_batch", None)
        if last is not None and not getattr(last, "is_empty", lambda: True)():
            if any(getattr(r, "is_sr_draft", False) is True for r in last.reqs):
                self.last_batch = None

    def _sr_run_http_batch(self) -> None:
        """One ordinary generate tick for Draft HTTP reqs (no RPC this loop)."""
        self._sr_park_sr_reqs()
        batch = self.get_next_batch_to_run()
        if batch is not None and batch.reqs:
            keep = [
                i for i, r in enumerate(batch.reqs) if self._sr_is_http_req(r)
            ]
            if keep:
                if len(keep) != len(batch.reqs):
                    batch.filter_batch(keep_indices=keep)
                self.cur_batch = batch
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
                self.last_batch = batch
                return
        self.cur_batch = None
        self.self_check_during_idle()

    def _sr_isolate_need(self, need_rids: set) -> None:
        """Park every scheduler req that is not in this RPC's need set.

        `_sr_run_until_ready` uses the global get_next_batch_to_run; without
        this, leftover waiting/running reqs mix into the GPU batch and blow up
        VL M-RoPE / KV gather. Sibling rids of the **same** RPC belong in
        ``need_rids`` so they share one fused forward.
        """
        running = getattr(self, "running_batch", None)
        if running is not None and not running.is_empty():
            keep: List[int] = []
            to_pause: List[Req] = []
            for i, r in enumerate(running.reqs):
                if r.rid in need_rids:
                    keep.append(i)
                else:
                    to_pause.append(r)
            if len(keep) != len(running.reqs):
                running.filter_batch(keep_indices=keep)
            for r in to_pause:
                r.draft_is_paused = True
                if r not in self.draft_paused_reqs:
                    self.draft_paused_reqs.append(r)
        waiting = getattr(self, "waiting_queue", None)
        if waiting:
            stay = []
            for r in list(waiting):
                if r.rid in need_rids:
                    stay.append(r)
                else:
                    r.draft_is_paused = True
                    if r not in self.draft_paused_reqs:
                        self.draft_paused_reqs.append(r)
            waiting[:] = stay
        last = getattr(self, "last_batch", None)
        if last is not None and not getattr(last, "is_empty", lambda: True)():
            if any(getattr(r, "rid", None) not in need_rids for r in last.reqs):
                self.last_batch = None

    def _sr_make_decode_batch(self, reqs: List[Req]) -> ScheduleBatch:
        device = getattr(self, "device", None)
        if device is None:
            try:
                device = self.tp_worker.model_runner.device
            except Exception:
                from sglang.srt.utils import get_device

                device = get_device()

        def _seq_len(r: Req) -> int:
            committed = int(getattr(r, "kv_committed_len", 0) or 0)
            if committed > 0:
                return committed
            return max(0, len(r.origin_input_ids) + len(r.output_ids or []) - 1)

        seq_lens_list = [_seq_len(r) for r in reqs]
        batch = ScheduleBatch(
            reqs=list(reqs),
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
            forward_mode=ForwardMode.DECODE,
            device=device,
        )
        batch.req_pool_indices = torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int64, device=device
        )
        batch.seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(
            seq_lens_list, dtype=torch.int32, device=device
        )
        batch.out_cache_loc = None
        batch.seq_lens_sum = sum(seq_lens_list)
        batch.output_ids = torch.tensor(
            [
                r.output_ids[-1] if r.output_ids else r.origin_input_ids[-1]
                for r in reqs
            ],
            dtype=torch.int64,
            device=device,
        )
        batch.return_logprob = any(r.return_logprob for r in reqs)
        batch.top_logprobs_nums = [
            r.top_logprobs_num if r.return_logprob else 0 for r in reqs
        ]
        batch.token_ids_logprobs = [
            r.token_ids_logprob if r.return_logprob else None for r in reqs
        ]
        batch.multimodal_inputs = [r.multimodal_inputs for r in reqs]
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )
        return batch

    def _sr_park_in_running_many(self, reqs: List[Req]) -> None:
        running = getattr(self, "running_batch", None)
        in_running = set()
        if running is not None and not running.is_empty():
            in_running = set(running.reqs)
        fresh = [
            r
            for r in reqs
            if r.req_pool_idx is not None and r not in in_running
        ]
        if not fresh:
            return
        decode_batch = self._sr_make_decode_batch(fresh)
        if running is None or running.is_empty():
            self.running_batch = decode_batch
        else:
            running.merge_batch(decode_batch)

    def _sr_finish_rid(
        self,
        rid: str,
        release_mm: bool = True,
        state: Optional[SRDraftState] = None,
    ) -> None:
        if state is None:
            state = self.sr_state.delete(rid)
        if state is None:
            return
        req = state.req_object
        if req is None:
            return
        self._sr_remove_req(req)
        if not req.finished():
            req.to_abort = True
            req.finished_reason = FINISH_ABORT("Target request finished")
        if req.req_pool_idx is not None:
            self.sr_kv.release_all_kv_for_finished_req(req)
        if release_mm:
            release_mm_resources(req.multimodal_inputs)
        req.multimodal_inputs = None

    def _sr_is_finished(self, req: Req) -> bool:
        """Target FINISH/ABORT is the only way a draft req finishes."""
        return req.finished()

    def _sr_mm_embed_error(self, req: Req) -> Optional[str]:
        """Vision items must still carry features to survive a re-prefill."""
        mm = getattr(req, "multimodal_inputs", None)
        if mm is None:
            return None
        for item in getattr(mm, "mm_items", None) or []:
            is_image = getattr(item, "is_image", None)
            is_video = getattr(item, "is_video", None)
            if not (
                (callable(is_image) and is_image())
                or (callable(is_video) and is_video())
            ):
                continue
            if item.precomputed_embeddings is None and item.feature is None:
                return "mm item has neither feature nor precomputed_embeddings"
        return None

    def _sr_mark_degraded(self, rid: str, reason: str) -> None:
        """Give up speculation for this rid until Target FINISHes it.

        Half-ingested KV / broken VL tensors cannot be repaired mid-flight, and
        letting them reach the GPU is what triggers the gather OOB. Target
        treats a non-OK reply as an empty window and falls back to 1-token AR.
        """
        state = self.sr_state.get(rid)
        if state is not None:
            if state.degraded:
                return
            state.degraded = True
        logger.warning("[SR] degrading %s to AR (no speculation): %s", rid, reason)

    def _sr_is_degraded(self, rid: str) -> bool:
        state = self.sr_state.get(rid)
        return state is not None and state.degraded is True

    def _sr_ensure_window_budget(
        self, req: Req, num_draft_tokens: Optional[int]
    ) -> None:
        """Raise max_new_tokens so PrefillAdder can budget this window.

        Local FINISH_LENGTH is suppressed on draft reqs; this still lifts the
        cap so the scheduler does not stall, and degrades if the next window
        would exceed the context.
        """
        sp = req.sampling_params
        already = len(req.output_ids or [])
        need = draft_needed_max_new_tokens(
            already,
            num_draft_tokens,
            self.server_args.speculative_num_steps,
            getattr(sp, "max_new_tokens", None),
        )
        if getattr(sp, "max_new_tokens", 0) < need:
            sp.max_new_tokens = need
        cap = int(getattr(self, "max_req_input_len", 0) or 0)
        if cap > 0:
            horizon = len(req.origin_input_ids or []) + need
            if horizon >= cap:
                self._sr_mark_degraded(req.rid, "context horizon exhausted")

    def _sr_create_req(
        self,
        dreq: SRDraftRequest,
        mm: Optional[SRMMPayload],
        session_id: str,
    ) -> Optional[Req]:
        rid = dreq.rid
        if self.sr_state.exists(rid):
            self._sr_finish_rid(rid, release_mm=False)

        input_ids = list(dreq.padded_input_ids or [])
        if mm is not None and mm.padded_input_ids:
            mismatch = _padded_ids_mismatch(input_ids, list(mm.padded_input_ids))
            if mismatch is not None:
                logger.warning(
                    "[SR] padded_input_ids mismatch for %s at %s; skip spec",
                    rid,
                    mismatch,
                )
                return None
        if not input_ids:
            logger.warning("[SR] PREFILL without padded_input_ids for %s", rid)
            return None

        sampling_params = dreq.sampling_params
        if sampling_params is None:
            from sglang.srt.sampling.sampling_params import SamplingParams

            sampling_params = SamplingParams()
        if hasattr(sampling_params, "normalize"):
            try:
                sampling_params.normalize(self.tokenizer)
            except Exception:
                _fix_sampling_params_stop_strs(sampling_params)
        else:
            _fix_sampling_params_stop_strs(sampling_params)

        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=False,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            lora_id=None,
            input_embeds=None,
            custom_logit_processor=None,
            return_hidden_states=False,
            eos_token_ids=self.model_config.hf_eos_token_id,
            bootstrap_host=None,
            bootstrap_port=8998,
            bootstrap_room=None,
            vocab_size=self.model_config.vocab_size,
        )
        req.tokenizer = self.tokenizer
        req.sr_padded_ids = list(input_ids)
        committed = list(dreq.committed_ids or [])
        req.origin_input_ids = list(input_ids) + committed
        req.origin_input_ids_unpadded = list(req.origin_input_ids)
        req.fill_ids = list(req.origin_input_ids)
        req.extend_input_len = len(req.fill_ids)
        req.output_ids = []
        req.logprob_start_len = len(req.origin_input_ids) - 1
        req.draft_tokens_target = dreq.num_draft_tokens
        req.draft_generation_start_len = 0
        req.sr_step_id = dreq.step_id
        req.draft_is_paused = False
        req.is_sr_draft = True
        req.suppress_local_finish = True
        self._sr_ensure_window_budget(req, dreq.num_draft_tokens)

        if mm is not None:
            mm_inputs = mm.to_multimodal_inputs()
            req.extend_image_inputs(mm_inputs)
            mm.attached = True
            self._maybe_compute_mrope_positions(req)

        self._sr_attach_grammar(req)

        self.sr_waiting.append(req)
        self.sr_state.set(
            rid,
            SRDraftState(
                req_id=rid,
                session_id=session_id,
                last_step_id=dreq.step_id,
                last_base_committed_len=dreq.base_committed_len,
                req_object=req,
            ),
        )
        return req

    def _sr_attach_grammar(self, req: Req) -> None:
        """Compile Target's schema on Draft. Failure leaves grammar unset."""
        from concurrent.futures import Future

        from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject

        gm = getattr(self, "grammar_manager", None)
        if gm is None:
            return
        try:
            added = gm.process_req_with_grammar(req)
        except Exception as e:
            logger.warning("[SR] Draft grammar compile failed for %s: %s", req.rid, e)
            req.grammar = None
            return
        if added:
            gm.grammar_queue = [r for r in gm.grammar_queue if r is not req]
            grammar = req.grammar
            if isinstance(grammar, Future):
                try:
                    grammar = grammar.result()
                    req.grammar = grammar
                    if (
                        getattr(req, "grammar_key", None) is not None
                        and gm.grammar_backend is not None
                    ):
                        gm.grammar_backend.set_cache(
                            req.grammar_key, grammar.copy()
                        )
                except Exception as e:
                    logger.warning(
                        "[SR] Draft grammar future failed for %s: %s", req.rid, e
                    )
                    req.grammar = None
                    return
        if req.grammar is None or isinstance(req.grammar, InvalidGrammarObject):
            if isinstance(req.grammar, InvalidGrammarObject):
                logger.warning(
                    "[SR] invalid grammar for %s: %s",
                    req.rid,
                    getattr(req.grammar, "error_message", ""),
                )
            req.grammar = None
            return
        try:
            req.sr_grammar_template = req.grammar.copy()
        except Exception as e:
            logger.warning(
                "[SR] Draft grammar copy failed for %s: %s", req.rid, e
            )
            req.grammar = None

    def _sr_replay_grammars(self, reqs: List[Req]) -> None:
        for req in reqs:
            template = getattr(req, "sr_grammar_template", None)
            if template is None:
                continue
            try:
                req.grammar = replay_grammar_from_committed(
                    template, req.output_ids or []
                )
            except Exception as e:
                logger.warning("[SR] grammar replay failed for %s: %s", req.rid, e)
                req.grammar = None

    def _sr_align(
        self, req: Req, dreq: SRDraftRequest, state: SRDraftState
    ) -> None:
        padded = list(getattr(req, "sr_padded_ids", None) or req.origin_input_ids)
        local = list(req.origin_input_ids) + list(req.output_ids or [])
        target = list(padded) + list(dreq.committed_ids or [])
        if local == target:
            req.draft_generation_start_len = len(req.output_ids or [])
            req.draft_tokens_target = dreq.num_draft_tokens
            return

        prefix_len = self.sr_kv.get_prefix_len(req)
        kind = classify_prefix_alignment(local, target, prefix_len)
        current_kv = int(getattr(req, "kv_allocated_len", 0) or 0)
        if current_kv <= 0:
            current_kv = max(0, len(local) - 1)
        _, fork = find_fork_point(local, target)

        if kind == "replace_tail":
            req.output_ids[-1] = target[-1]
            req.draft_generation_start_len = len(req.output_ids)
            req.draft_tokens_target = dreq.num_draft_tokens
            return
        if kind in ("append_one", "append_n"):
            req.output_ids.extend(target[len(local) :])
            req.draft_generation_start_len = len(req.output_ids)
            req.draft_tokens_target = dreq.num_draft_tokens
            return
        if kind == "local_rollback" and fork == len(target) and self.sr_kv.rollback(
            req, fork, current_kv
        ):
            extra = len(local) - fork
            if extra > 0 and req.output_ids:
                keep = max(0, len(req.output_ids) - extra)
                req.output_ids = req.output_ids[:keep]
            req.draft_generation_start_len = len(req.output_ids)
            req.draft_tokens_target = dreq.num_draft_tokens
            return
        self._sr_reprefill(req, target, dreq, state)

    def _sr_run_window_batch(
        self, pairs: List[Tuple[SRDraftRequest, Req]]
    ) -> None:
        reqs: List[Req] = []
        for dreq, req in pairs:
            self._sr_ensure_window_budget(req, dreq.num_draft_tokens)
            req.draft_tokens_target = dreq.num_draft_tokens
            reqs.append(req)
        if reqs:
            self._sr_run_until_ready(reqs)

    def _sr_token_lens_for_seed(self, reqs: List[Req], batch) -> Optional[List[int]]:
        if batch is None:
            return None
        lens = getattr(batch, "extend_lens", None)
        if lens is not None and len(lens) == len(reqs):
            return [int(x) for x in lens]
        fallback = [int(getattr(r, "extend_input_len", 0) or 0) for r in reqs]
        if fallback and all(x > 0 for x in fallback):
            return fallback
        return None

    def _sr_cache_tree_seeds(
        self, reqs: List[Req], result, batch=None
    ) -> None:
        """Cache one tree seed per req. ``reqs`` must match the GPU batch rows."""
        logits_output = getattr(result, "logits_output", None)
        if logits_output is None or logits_output.next_token_logits is None:
            return
        hidden = getattr(logits_output, "hidden_states", None)
        if hidden is None:
            return
        logits = logits_output.next_token_logits
        n = len(reqs)
        token_lens = self._sr_token_lens_for_seed(reqs, batch)
        try:
            from sglang.srt.speculative.spec_utils import fast_topk

            topk = max(1, int(self.server_args.speculative_eagle_topk or 1))
            skipped: List[str] = []
            for i, req in enumerate(reqs):
                row_logits = slice_decode_batch_row(logits, i, n, token_lens)
                row_hidden = slice_decode_batch_row(hidden, i, n, token_lens)
                if row_logits is None or row_hidden is None:
                    skipped.append(req.rid)
                    continue
                probs = torch.softmax(row_logits, dim=-1)
                topk_p, topk_index = fast_topk(probs, topk, dim=-1)
                token_id = (
                    req.output_ids[-1]
                    if req.output_ids
                    else req.origin_input_ids[-1]
                )
                verified_id = torch.tensor(
                    [token_id], dtype=torch.int64, device=row_logits.device
                )
                req.sr_tree_seed = (topk_p, topk_index, row_hidden, verified_id)
            if skipped:
                logger.warning(
                    "[SR] skip tree seed for %s: logits/hidden %s/%s "
                    "batch=%s extend_lens=%s",
                    skipped,
                    tuple(logits.shape),
                    tuple(hidden.shape),
                    n,
                    token_lens,
                )
        except Exception as e:
            if _sr_is_device_context_error(e):
                logger.error(
                    "[SR] cache tree seed device context error for %s: %s",
                    [r.rid for r in reqs],
                    e,
                )
                raise
            logger.warning(
                "[SR] cache tree seed failed for %s: %s",
                [r.rid for r in reqs],
                e,
            )

    def _sr_intercept_retract_enqueue(self, req: Req) -> bool:
        """Handle scheduler KV retract for Target-RPC drafts.

        Generic retract requeues onto ``waiting_queue`` and re-prefills
        ``origin + output`` with the prompt-length M-RoPE tensor. That path
        illegal-memory-accesses Qwen-VL. Return True to skip enqueue.
        """
        if getattr(req, "is_sr_draft", False) is not True:
            return False
        self._sr_on_scheduler_retract(req)
        return True

    def _sr_on_scheduler_retract(self, req: Req) -> None:
        fill_ids = list(req.origin_input_ids or []) + list(req.output_ids or [])
        if not fill_ids:
            fill_ids = list(getattr(req, "sr_padded_ids", None) or [])
        self._sr_reset_linear_kv_state(req, fill_ids)
        self._sr_pause_req(req)

    def _sr_reset_linear_kv_state(self, req: Req, fill_ids: List[int]) -> None:
        """Drop linear KV bookkeeping and rebuild fill/origin for a full re-prefill."""
        self._sr_remove_req(req)
        if req.req_pool_idx is not None:
            kv = getattr(self, "sr_kv", None)
            if kv is not None:
                kv.release_all_kv_for_finished_req(req)
        req.fill_ids = list(fill_ids)
        req.origin_input_ids = list(fill_ids)
        req.output_ids = []
        req.prefix_indices = []
        req.extend_input_len = len(req.fill_ids)
        req.draft_generation_start_len = 0
        req.last_node = None
        req.kv_committed_len = 0
        req.kv_committed_freed = False
        req.kv_overallocated_freed = False
        req.logprob_start_len = max(0, len(req.origin_input_ids) - 1)
        req.sr_tree_seed = None
        if req.multimodal_inputs is not None:
            # origin_input_ids was rewritten, so M-RoPE must be recomputed. If
            # that fails or the vision tensors are already gone, prefilling
            # would index past the embedding table and take down the scheduler.
            reset_mm_mrope(req.multimodal_inputs)
            maybe = getattr(self, "_maybe_compute_mrope_positions", None)
            if callable(maybe):
                try:
                    maybe(req)
                except Exception as e:
                    self._sr_mark_degraded(
                        req.rid, f"compute_mrope_positions failed: {e}"
                    )
                    return
            embed_err = self._sr_mm_embed_error(req)
            if embed_err:
                self._sr_mark_degraded(req.rid, embed_err)
                return

    def _sr_enqueue_for_reprefill(self, req: Req, fill_ids: List[int]) -> bool:
        """The only way a draft req may re-enter waiting_queue.

        Rebuilds all length-derived state (M-RoPE included) so EXTEND cannot
        read past a stale prompt-width tensor.
        """
        self._sr_reset_linear_kv_state(req, fill_ids)
        if self._sr_is_degraded(req.rid):
            return False
        mm = req.multimodal_inputs
        if mm is not None and getattr(mm, "mrope_positions", None) is not None:
            if mm.mrope_positions.shape[1] != len(fill_ids):
                self._sr_mark_degraded(req.rid, "mrope width != fill_ids")
                return False
        self._sr_resume_req(req)
        if req not in self.sr_waiting:
            self.sr_waiting.append(req)
        waiting = getattr(self, "waiting_queue", None)
        if waiting is not None and req not in waiting:
            waiting.append(req)
        return True

    def _sr_materialize_prefix_batch(self, reqs: List[Req]) -> None:
        """Prefill until every req has a pool slot; drop any sampled extras."""
        need = [r for r in reqs if r.req_pool_idx is None]
        if not need:
            return
        live: List[Req] = []
        for req in need:
            if req.output_ids:
                fill_ids = list(req.origin_input_ids or []) + list(req.output_ids)
                if not self._sr_enqueue_for_reprefill(req, fill_ids):
                    continue
            else:
                self._sr_resume_req(req)
                if req not in self.sr_waiting:
                    self.sr_waiting.append(req)
                waiting = getattr(self, "waiting_queue", None)
                if waiting is not None and req not in waiting:
                    waiting.append(req)
            live.append(req)
        need = live
        if not need:
            return
        need_rids = {r.rid for r in need}
        self._sr_isolate_need(need_rids)
        for _ in range(8):
            still = [r for r in need if r.req_pool_idx is None]
            if not still:
                break
            still_rids = {r.rid for r in still}
            self._sr_isolate_need(still_rids)
            batch = self.get_next_batch_to_run()
            if batch is None or not batch.reqs:
                break
            keep = [i for i, r in enumerate(batch.reqs) if r.rid in still_rids]
            if not keep:
                break
            if len(keep) != len(batch.reqs):
                batch.filter_batch(keep_indices=keep)
            self._sr_enable_tree_seed_hidden(batch)
            self.cur_batch = batch
            result = self.run_batch(batch)
            self._sr_cache_tree_seeds(list(batch.reqs), result, batch)
            self.process_batch_result(batch, result)
            self.last_batch = batch
            is_extend = (
                batch.forward_mode is not None and batch.forward_mode.is_extend()
            ) or getattr(batch, "is_extend_in_batch", False)
            if is_extend and len(keep) == len(still):
                break
        for req in need:
            req.output_ids = []
            req.draft_generation_start_len = 0
            if not self._sr_is_finished(req):
                self._sr_pause_req(req)
        self.last_batch = None

    def _sr_kv_len(self, req: Req) -> int:
        committed = int(getattr(req, "kv_committed_len", 0) or 0)
        if committed > 0:
            return committed
        allocated = int(getattr(req, "kv_allocated_len", 0) or 0)
        if allocated > 0:
            return allocated
        return max(0, len(req.origin_input_ids))

    def _sr_max_ingest_decode_steps(self) -> int:
        """Per-token teacher forcing budget before one extend is cheaper."""
        server_args = getattr(self, "server_args", None)
        n = int(getattr(server_args, "speculative_num_draft_tokens", 0) or 0)
        if n <= 0:
            n = int(getattr(server_args, "speculative_num_steps", 0) or 0) + 1
        return max(DEFAULT_MAX_INGEST_DECODE_STEPS, 2 * n)

    def _sr_reprefill_committed(self, reqs: List[Req]) -> None:
        """Fold a long committed tail into one extend instead of N decodes."""
        rebuilt: List[Req] = []
        for req in reqs:
            padded = list(getattr(req, "sr_padded_ids", None) or [])
            committed = list(req.output_ids or [])
            fill_ids = (padded + committed) if padded else (
                list(req.origin_input_ids) + committed
            )
            logger.info(
                "[SR] ingest tail %s tokens for %s: one reprefill of %s ids",
                len(committed),
                req.rid,
                len(fill_ids),
            )
            if not self._sr_enqueue_for_reprefill(req, fill_ids):
                continue
            self._sr_ensure_window_budget(req, req.draft_tokens_target)
            rebuilt.append(req)
        if rebuilt:
            self._sr_materialize_prefix_batch(rebuilt)

    def _sr_ingest_committed_batch(self, reqs: List[Req]) -> None:
        """Teacher-force committed tails into linear KV.

        Align only appends Target ids onto ``output_ids``. Chain then forwards
        that last token with ``prepare_for_decode``. Tree used to skip that and
        expand on prompt KV, so only layer-0 matched (accept len stuck at 2).
        Short tails go one decode per token (tails may differ in length: each
        step only runs reqs that still have a token at that offset). A long
        tail -- Target ran ahead autoregressively -- is folded into a single
        reprefill so the RPC cannot blow its deadline.
        """
        if not reqs:
            return
        max_decode_steps = self._sr_max_ingest_decode_steps()
        snapshots = []
        long_tail: List[Req] = []
        for req in reqs:
            committed = list(req.output_ids or [])
            mode, tail = plan_committed_ingest(
                len(req.origin_input_ids),
                committed,
                self._sr_kv_len(req),
                max_decode_steps=max_decode_steps,
            )
            if mode == "reprefill":
                long_tail.append(req)
                continue
            snapshots.append(
                (req, committed, tail, len(committed) - len(tail))
            )
        if long_tail:
            self._sr_reprefill_committed(long_tail)
        for t, active_idx in enumerate(
            ingest_active_indices(
                [len(tail) for _, _, tail, _ in snapshots]
            )
        ):
            active: List[Req] = []
            for i in active_idx:
                req, committed, _tail, already = snapshots[i]
                req.output_ids = committed[: already + t + 1]
                req.draft_generation_start_len = len(req.output_ids)
                req.draft_tokens_target = 1
                self._sr_ensure_window_budget(req, 1)
                active.append(req)
            if active:
                self._sr_run_until_ready(active, capture_tree_seed=True)
        for req, committed, _, _ in snapshots:
            req.output_ids = committed
            req.draft_generation_start_len = len(committed)

    def _sr_tree_expand_batch(self, reqs: List[Req]) -> List[SRWindow]:
        empty: SRWindow = ([], None, None)
        if not reqs:
            return []
        self._sr_materialize_prefix_batch(reqs)
        self._sr_ingest_committed_batch(reqs)
        self._sr_replay_grammars(reqs)
        windows: List[SRWindow] = [empty] * len(reqs)
        ready: List[Req] = []
        ready_idx: List[int] = []
        for i, req in enumerate(reqs):
            leftover = committed_tail_not_in_kv(
                len(req.origin_input_ids),
                req.output_ids,
                self._sr_kv_len(req),
            )
            if leftover:
                logger.warning(
                    "[SR] tree ingest left %s token(s) out of KV for %s "
                    "(finished_reason=%s kv_committed_len=%s prompt_len=%s "
                    "output_len=%s)",
                    len(leftover),
                    req.rid,
                    getattr(req, "finished_reason", None),
                    getattr(req, "kv_committed_len", None),
                    len(req.origin_input_ids),
                    len(req.output_ids or []),
                )
                self._sr_mark_degraded(req.rid, "tree ingest left tokens out of KV")
                continue
            if req.req_pool_idx is None or self.sr_tree_drafter is None:
                continue
            if getattr(req, "sr_tree_seed", None) is None:
                continue
            ready.append(req)
            ready_idx.append(i)
        if not ready:
            return windows
        for req in ready:
            self._sr_resume_req(req)
        self._sr_park_in_running_many(ready)
        try:
            got = self.sr_tree_drafter.expand_batch(ready)
        except Exception as e:
            if _sr_is_device_context_error(e):
                logger.error(
                    "[SR] tree expand device context error for %s: %s",
                    [r.rid for r in ready],
                    e,
                )
                raise
            logger.warning(
                "[SR] tree expand failed for %s: %s",
                [r.rid for r in ready],
                e,
            )
            got = [empty] * len(ready)
        finally:
            for req in ready:
                if not self._sr_is_finished(req):
                    self._sr_pause_req(req)
            self.last_batch = None
        for j, idx in enumerate(ready_idx):
            windows[idx] = got[j]
        return windows

    def _sr_reprefill(
        self,
        req: Req,
        target_fill_ids: List[int],
        dreq: SRDraftRequest,
        state: SRDraftState,
    ) -> None:
        req.draft_tokens_target = dreq.num_draft_tokens
        self._sr_enqueue_for_reprefill(req, target_fill_ids)

    def _sr_run_until_ready(
        self,
        reqs: List[Req],
        capture_tree_seed: bool = False,
    ) -> None:
        need = {r.rid: r for r in reqs}
        need_rids = set(need.keys())
        self._sr_isolate_need(need_rids)
        parked: List[Req] = []
        for req in reqs:
            self._sr_resume_req(req)
            if req.req_pool_idx is None:
                if req.output_ids:
                    logger.error(
                        "[SR] req %s lost pool slot with %s output tokens; "
                        "rebuilding prefix (suppress_local_finish invariant broken)",
                        req.rid,
                        len(req.output_ids),
                    )
                    if not self._sr_enqueue_for_reprefill(
                        req,
                        list(req.origin_input_ids or []) + list(req.output_ids),
                    ):
                        need.pop(req.rid, None)
                        continue
                else:
                    if req not in self.sr_waiting:
                        self.sr_waiting.append(req)
                    waiting = getattr(self, "waiting_queue", None)
                    if waiting is not None and req not in waiting:
                        waiting.append(req)
            else:
                parked.append(req)
        self._sr_park_in_running_many(parked)

        guard = 0
        max_steps = max(
            int(self.server_args.speculative_num_steps or 1) + 4, 8
        ) * max(1, len(reqs))
        while need and guard < max_steps:
            guard += 1
            still = []
            for rid, req in list(need.items()):
                produced = len(req.output_ids) - int(
                    getattr(req, "draft_generation_start_len", 0) or 0
                )
                target = int(getattr(req, "draft_tokens_target", 0) or 0)
                if target <= 0:
                    target = int(self.server_args.speculative_num_steps or 1)
                if produced >= target or self._sr_is_finished(req):
                    still.append(rid)
            for rid in still:
                need.pop(rid, None)
            if not need:
                break
            self._sr_isolate_need(set(need.keys()))
            batch = self.get_next_batch_to_run()
            if batch is not None and batch.reqs:
                keep = [i for i, r in enumerate(batch.reqs) if r.rid in need]
                if not keep:
                    break
                if len(keep) != len(batch.reqs):
                    batch.filter_batch(keep_indices=keep)
            self.cur_batch = batch
            if batch:
                if capture_tree_seed:
                    self._sr_enable_tree_seed_hidden(batch)
                result = self.run_batch(batch)
                if capture_tree_seed:
                    self._sr_cache_tree_seeds(list(batch.reqs), result, batch)
                self.process_batch_result(batch, result)
                self.last_batch = batch
            else:
                break

        for req in reqs:
            if not self._sr_is_finished(req):
                self._sr_pause_req(req)
        self.last_batch = None

    def _sr_extract_window(self, req: Req, dreq: SRDraftRequest) -> SRWindow:
        start = int(getattr(req, "draft_generation_start_len", 0) or 0)
        tokens = list(req.output_ids[start:])
        n = dreq.num_draft_tokens or len(tokens)
        return tokens[:n], None, None

    def _sr_empty_reply(
        self, dreq: SRDraftRequest, status: SRReplyStatus = SRReplyStatus.EMPTY
    ) -> SRDraftReply:
        return SRDraftReply(
            rid=dreq.rid,
            step_id=dreq.step_id,
            base_committed_len=dreq.base_committed_len,
            draft_tokens=[],
            status=status,
        )

    def _sr_stamp_window(
        self,
        dreq: SRDraftRequest,
        window: SRWindow,
        rpc_seq: int,
        session_id: str,
        status: Optional[SRReplyStatus] = None,
    ) -> SRDraftReply:
        tokens, pl, ix = window
        st = self.sr_state.get(dreq.rid)
        if st is not None:
            st.last_step_id = dreq.step_id
            st.last_base_committed_len = dreq.base_committed_len
            st.last_rpc_seq = rpc_seq
            st.last_window = window
            st.session_id = session_id
            st.last_updated_time = time.time()
        if status is None:
            status = SRReplyStatus.OK if tokens else SRReplyStatus.EMPTY
        return SRDraftReply(
            rid=dreq.rid,
            step_id=dreq.step_id,
            base_committed_len=dreq.base_committed_len,
            draft_tokens=list(tokens),
            status=status,
            parent_list=pl,
            top_scores_index=ix,
        )

    def _sr_prepare_one(
        self,
        dreq: SRDraftRequest,
        action: SRAction,
        session_id: str,
        rpc_seq: int,
        mm: Optional[SRMMPayload],
        last_session_id: Optional[str],
        last_rpc_seq: int,
    ) -> Tuple[Optional[SRDraftReply], Optional[Req]]:
        """CPU decision + align. Returns (early_reply, None) or (None, live_req)."""
        state = self.sr_state.get(dreq.rid)
        req = state.req_object if state is not None else None
        decision = decide_draft_action(
            action=action,
            session_id=session_id,
            rpc_seq=rpc_seq,
            last_session_id=last_session_id,
            last_rpc_seq=last_rpc_seq,
            last_step_id=state.last_step_id if state is not None else -1,
            last_base_committed_len=(
                state.last_base_committed_len if state is not None else -1
            ),
            step_id=dreq.step_id,
            base_committed_len=dreq.base_committed_len,
            has_state=state is not None and req is not None,
        )
        empty = self._sr_empty_reply(dreq)
        if decision in (
            DraftDecision.DROP_OLD_SESSION,
            DraftDecision.DROP_STALE_SEQ,
        ):
            empty.status = SRReplyStatus.REJECT
            return empty, None
        if decision == DraftDecision.FINISH:
            self._sr_finish_rid(dreq.rid)
            empty.status = SRReplyStatus.OK
            return empty, None
        if decision == DraftDecision.IDEMPOTENT:
            if state is not None and state.last_window is not None:
                tokens, pl, ix = state.last_window
                return (
                    SRDraftReply(
                        rid=dreq.rid,
                        step_id=dreq.step_id,
                        base_committed_len=dreq.base_committed_len,
                        draft_tokens=list(tokens),
                        status=SRReplyStatus.IDEMPOTENT,
                        parent_list=pl,
                        top_scores_index=ix,
                    ),
                    None,
                )
            return empty, None
        if decision == DraftDecision.HARD_RESET:
            # A fresh req replaces the broken KV, so speculation can resume.
            if state is not None:
                state.degraded = False
            req = self._sr_create_req(dreq, mm, session_id)
            if req is None:
                return empty, None
        elif decision == DraftDecision.WIPE_NEW_SESSION:
            # Batch-level wipe already ran for this session_id; do not wipe
            # siblings created earlier in the same RPC.
            if self.sr_state.session_id != session_id:
                self._sr_wipe_all()
                self.sr_state.session_id = session_id
            if dreq.padded_input_ids:
                req = self._sr_create_req(dreq, mm, session_id)
                if req is None:
                    return empty, None
            else:
                return empty, None
        else:
            if req is None:
                if dreq.padded_input_ids:
                    req = self._sr_create_req(dreq, mm, session_id)
                    if req is None:
                        return empty, None
                else:
                    return empty, None
            elif state is not None and state.degraded:
                # Broken Draft state: stay off the GPU, let Target autoregress.
                return empty, None
            else:
                self._sr_align(req, dreq, state)

        if req is None:
            return empty, None
        return None, req

    def _sr_produce_windows(
        self,
        action: SRAction,
        pairs: List[Tuple[SRDraftRequest, Req]],
    ) -> List[SRWindow]:
        if not pairs:
            return []
        reqs = [req for _, req in pairs]
        if self._sr_tree_mode():
            if action == SRAction.PREFILL:
                self._sr_materialize_prefix_batch(reqs)
                return [([], None, None)] * len(pairs)
            return self._sr_tree_expand_batch(reqs)
        # Chain: teacher-force newly appended committed tokens into KV before
        # generating the next window. append_one used to fold that into the
        # first draft decode; append_n cannot (KV would skip the middle ids).
        self._sr_ingest_committed_batch(reqs)
        self._sr_replay_grammars(reqs)
        self._sr_run_window_batch(pairs)
        return [self._sr_extract_window(req, dreq) for dreq, req in pairs]

    def _sr_handle_batch(
        self, batch: SRBatchRequest, mm_by_rid: Dict[str, SRMMPayload]
    ) -> SRBatchReply:
        last_session = self.sr_state.session_id
        last_rpc = self.sr_server.last_rpc_seq if self.sr_server is not None else -1
        wiped_this_batch = (
            last_session is not None and batch.session_id > last_session
        )
        if wiped_this_batch:
            self._sr_wipe_all()
            # New session resets Target rpc_seq to 0. Do not apply the previous
            # session's last_rpc_seq to this batch or PREFILL is DROP_STALE_SEQ.
            last_rpc = -1
        self.sr_state.session_id = batch.session_id
        self._sr_cleanup_stale_drafts(keep_rids={d.rid for d in batch.reqs})
        if (
            batch.action not in (SRAction.FINISH, SRAction.ABORT)
            and self._sr_draft_busy()
        ):
            logger.info(
                "[SR] Draft busy, REJECT rpc_seq=%s n=%s",
                batch.rpc_seq,
                len(batch.reqs),
            )
            return SRBatchReply(
                session_id=batch.session_id,
                rpc_seq=batch.rpc_seq,
                reqs=[
                    self._sr_empty_reply(d, status=SRReplyStatus.REJECT)
                    for d in batch.reqs
                ],
            )
        n = len(batch.reqs)
        replies: List[Optional[SRDraftReply]] = [None] * n
        gpu_pairs: List[Tuple[int, SRDraftRequest, Req]] = []
        # After a batch-level wipe, later rids must see the new session so
        # decide_draft_action does not WIPE_NEW_SESSION again and drop siblings.
        prepare_last_session = (
            batch.session_id if wiped_this_batch else last_session
        )
        prepare_last_rpc = -1 if wiped_this_batch else last_rpc
        for i, dreq in enumerate(batch.reqs):
            reply, req = self._sr_prepare_one(
                dreq,
                batch.action,
                batch.session_id,
                batch.rpc_seq,
                mm_by_rid.get(dreq.rid),
                prepare_last_session,
                prepare_last_rpc,
            )
            if reply is not None:
                replies[i] = reply
            else:
                gpu_pairs.append((i, dreq, req))
        if gpu_pairs:
            windows = self._sr_produce_windows(
                batch.action, [(dreq, req) for _, dreq, req in gpu_pairs]
            )
            for (i, dreq, _req), window in zip(gpu_pairs, windows):
                replies[i] = self._sr_stamp_window(
                    dreq, window, batch.rpc_seq, batch.session_id
                )
        if self.sr_server is not None:
            self.sr_server.remember(batch)
        return SRBatchReply(
            session_id=batch.session_id,
            rpc_seq=batch.rpc_seq,
            reqs=[r if r is not None else self._sr_empty_reply(d) for r, d in zip(replies, batch.reqs)],
        )

    def _sr_recv_packet(self):
        packet = None
        if self.tp_size == 1 or self.tp_rank == 0:
            if self.sr_server is not None:
                packet = self.sr_server.recv_batch(timeout_ms=50)
                if packet is not None:
                    batch, mm = packet
                    mm_pickled = {
                        rid: p.to_pickleable() for rid, p in mm.items()
                    }
                    packet = (batch.to_dict(), mm_pickled)
        if self.tp_size > 1:
            packet = broadcast_sr_obj(
                packet,
                self.tp_size,
                self.tp_rank,
                self.tp_group,
                self.tp_cpu_group,
            )
        if packet is None:
            return None
        batch_d, mm_d = packet
        batch = SRBatchRequest.from_dict(batch_d)
        mm = {
            rid: SRMMPayload.from_pickleable(d) for rid, d in (mm_d or {}).items()
        }
        return batch, mm

    @DynamicGradMode()
    def event_loop_normal_standalone_remote_draft(self) -> None:
        self._init_sr_draft()
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            packet = self._sr_recv_packet()
            if packet is not None:
                batch, mm = packet
                if self.sr_server is not None and self.sr_server.is_stale(batch):
                    if self.tp_rank == 0:
                        logger.info(
                            "[SR] Draft drop stale session=%s rpc_seq=%s",
                            batch.session_id,
                            batch.rpc_seq,
                        )
                else:
                    reply = self._sr_handle_batch(batch, mm)
                    if (
                        (self.tp_size == 1 or self.tp_rank == 0)
                        and self.sr_server is not None
                        and batch.action not in (SRAction.FINISH, SRAction.ABORT)
                    ):
                        self.sr_server.send_batch(reply)
                self.last_batch = None
                self._sr_resume_http_reqs()
                continue
            self._sr_cleanup_stale_drafts()
            self._sr_run_http_batch()
