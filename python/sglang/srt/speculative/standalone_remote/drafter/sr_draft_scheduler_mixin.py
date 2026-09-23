import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.speculative.standalone_remote.sr_round_metrics import get_sr_round_metrics

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
from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
    SRAlignResult,
    SRTreeKVLease,
    SRTreeLeaseStore,
    live_accept_prefix,
    prefix_window_tokens,
    read_token_span,
    snapshot_sr_align,
    validate_lease_commit,
)
from sglang.srt.speculative.standalone_remote.drafter.sr_tail_extend import (
    SRTailExtendTransaction,
    TailExtendRecoveryRequired,
    invalidate_tree_seed,
    make_tail_extend_batch,
    plan_tail_extend,
    stamp_tree_seed,
    tree_seed_is_current,
)
from sglang.srt.speculative.standalone_remote.sr_align import (
    DEFAULT_MAX_INGEST_DECODE_STEPS,
    DraftDecision,
    apply_tree_seed_topk,
    broadcast_sr_obj,
    committed_tail_not_in_kv,
    decide_draft_action,
    draft_needed_max_new_tokens,
    find_fork_point,
    ingest_active_indices,
    is_device_context_error,
    last_token_in_kv,
    plan_committed_ingest,
    plan_tree_seed_recovery,
    replay_grammar_from_committed,
    snapshot_reprefill_fill_ids,
    sr_decode_seq_len,
    tree_seed_matches_prefix,
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
from sglang.srt.speculative.standalone_remote.sr_commit import (
    SR_PROTOCOL_VERSION,
    CommitOutcome,
    CommittedPrefixView,
    PendingCommit,
    RecoveryRoute,
    candidate_stamp_current,
    commit_fingerprint,
    delta_fast_allowed,
    grammar_committed_ids,
    inspect_commit,
    make_prefix_stamp,
    mm_items_complete,
    note_commit_result,
    rebuild_committed_output,
    retarget_stamp_version,
    route_snapshot_recovery,
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
from sglang.srt.speculative.spec_utils import (
    NpuGraphPreparationError,
    NpuGraphReplaySubmittedError,
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


@dataclass
class _SRTreePlanDraft:
    req: Req
    kind: str
    plan: object = None
    lease: Optional[SRTreeKVLease] = None
    miss: Optional[str] = None


def _sr_is_device_context_error(exc: BaseException) -> bool:
    return is_device_context_error(exc)


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
        self.sr_tree_leases = SRTreeLeaseStore()
        self._sr_device_poisoned = False
        topk = int(self.server_args.speculative_eagle_topk or 1)
        if topk > 1:
            self.sr_tree_drafter = SRTreeDrafter(self)
        logger.info(
            "[SR] Draft scheduler ready (tree_configured=%s topk=%s "
            "tree_graph_captured=%s tree_graph_disabled_reason=%s)",
            self.sr_tree_drafter is not None,
            topk,
            getattr(self.sr_tree_drafter, "tree_graph_capture_succeeded", False),
            getattr(
                self.sr_tree_drafter, "tree_graph_disabled_reason", "tree not configured"
            ),
        )

    def _sr_tree_mode(self) -> bool:
        return getattr(self, "sr_tree_drafter", None) is not None

    def _sr_enable_tree_seed_hidden(self, batch: ScheduleBatch) -> None:
        """Request tree-seed top-k without recurrent hidden capture."""
        batch.return_hidden_states = False
        batch.capture_hidden_mode = CaptureHiddenMode.NULL
        apply_tree_seed_topk(
            batch,
            getattr(self.server_args, "speculative_eagle_topk", 1),
        )

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
            self._sr_release_tree_lease(state.req_id)
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
            store = getattr(self, "sr_tree_leases", None)
            if store is not None:
                store.release_all(self.token_to_kv_pool_allocator)
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

        seq_lens_list = [sr_decode_seq_len(r) for r in reqs]
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
            self._sr_release_tree_lease(rid)
            if getattr(self, "_sr_device_poisoned", False):
                logger.error(
                    "[SR] NPU context already poisoned; skip KV release for %s",
                    rid,
                )
            else:
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
            state.local_prefix_stamp = None
            state.pending_commit = None
            if state.degraded:
                return
            state.degraded = True
            store = getattr(self, "sr_tree_leases", None)
            lease = store.get(rid) if store is not None else None
            if lease is None or not lease.in_use:
                self._sr_release_tree_lease(rid)
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

    def _sr_replay_grammars(self, reqs: List[Req], *, strict: bool = False) -> None:
        for req in reqs:
            template = getattr(req, "sr_grammar_template", None)
            if template is None:
                continue
            try:
                committed_ids = grammar_committed_ids(
                    getattr(req, "sr_commit_output_ids", None),
                    req.output_ids,
                    req.origin_input_ids,
                    getattr(req, "sr_padded_ids", None),
                    tree_mode=self._sr_tree_mode(),
                )
                req.grammar = replay_grammar_from_committed(
                    template, committed_ids
                )
            except Exception as e:
                if strict:
                    raise RuntimeError(f"grammar replay failed for {req.rid}") from e
                logger.warning("[SR] grammar replay failed for %s: %s", req.rid, e)
                req.grammar = None

    def _sr_clear_stamps_after_poison(self) -> None:
        manager = getattr(self, "sr_state", None)
        active = getattr(manager, "active", None)
        if not isinstance(active, dict):
            return
        for state in active.values():
            state.local_prefix_stamp = None
            state.pending_commit = None

    def _sr_clear_prefix_stamp(self, state: Optional[SRDraftState]) -> None:
        if state is not None:
            state.local_prefix_stamp = None

    def _sr_drop_unconfirmed_commit(self, state: Optional[SRDraftState]) -> None:
        """Forget a commit that never reached finalize. Do not extend history."""
        if state is None or state.pending_commit is None:
            return
        state.pending_commit = None
        state.local_prefix_stamp = None

    def _sr_align(
        self, req: Req, dreq: SRDraftRequest, state: SRDraftState
    ) -> None:
        prefix_len = self.sr_kv.get_prefix_len(req)
        result = snapshot_sr_align(req, dreq, prefix_len)
        req.sr_align_result = result
        req.sr_prefix_proven = False
        req.sr_pending_dreq = dreq
        padded = list(getattr(req, "sr_padded_ids", None) or req.origin_input_ids)
        local = list(req.origin_input_ids) + list(req.output_ids or [])
        target = list(padded) + list(dreq.committed_ids or [])
        if result.kind == "equal":
            req.sr_prefix_proven = True
            req.draft_generation_start_len = len(req.output_ids or [])
            req.draft_tokens_target = dreq.num_draft_tokens
            return

        invalidate_tree_seed(req)

        kind = result.kind
        allocated = int(getattr(req, "kv_allocated_len", 0) or 0)
        if allocated <= 0:
            allocated = max(0, len(local) - 1)
        committed = int(result.old_kv_committed_len)
        fork = result.fork

        if kind == "replace_tail":
            self._sr_clear_prefix_stamp(state)
            req.output_ids[-1] = target[-1]
            req.sr_tree_seed = None
            req.draft_generation_start_len = len(req.output_ids)
            req.draft_tokens_target = dreq.num_draft_tokens
            if last_token_in_kv(fork, committed):
                if not self.sr_kv.rollback(req, fork, allocated):
                    self._sr_reprefill(req, target, dreq, state)
            return
        if kind in ("append_one", "append_n"):
            req.output_ids.extend(target[len(local) :])
            req.sr_prefix_proven = True
            req.draft_generation_start_len = len(req.output_ids)
            req.draft_tokens_target = dreq.num_draft_tokens
            return
        if kind == "local_rollback" and fork == len(target) and self.sr_kv.rollback(
            req, fork, allocated
        ):
            extra = len(local) - fork
            if extra > 0 and req.output_ids:
                keep = max(0, len(req.output_ids) - extra)
                req.output_ids = req.output_ids[:keep]
            self._sr_clear_prefix_stamp(state)
            req.sr_tree_seed = None
            req.draft_generation_start_len = len(req.output_ids)
            req.draft_tokens_target = dreq.num_draft_tokens
            return
        self._sr_clear_prefix_stamp(state)
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
        if logits_output is None:
            return
        topk_p_all = getattr(logits_output, "tree_seed_topk_p", None)
        topk_index_all = getattr(logits_output, "tree_seed_topk_index", None)
        topk = max(1, int(getattr(self.server_args, "speculative_eagle_topk", 1) or 1))
        n = len(reqs)
        if (
            topk_p_all is None
            or topk_index_all is None
            or topk_p_all.ndim != 2
            or topk_index_all.ndim != 2
            or topk_p_all.shape != topk_index_all.shape
            or topk_p_all.shape != (n, topk)
        ):
            logger.warning(
                "[SR] skip tree seed for %s: missing pre-sample top-k "
                "(topk_p=%s topk_index=%s want=(%s, %s))",
                [r.rid for r in reqs],
                None if topk_p_all is None else tuple(topk_p_all.shape),
                None if topk_index_all is None else tuple(topk_index_all.shape),
                n,
                topk,
            )
            return
        try:
            skipped: List[str] = []
            for i, req in enumerate(reqs):
                row_p = slice_decode_batch_row(topk_p_all, i, n, None)
                row_ix = slice_decode_batch_row(topk_index_all, i, n, None)
                if row_p is None or row_ix is None:
                    skipped.append(req.rid)
                    continue
                token_id = (
                    req.output_ids[-1]
                    if req.output_ids
                    else req.origin_input_ids[-1]
                )
                verified_id = torch.tensor(
                    [token_id], dtype=torch.int64, device=row_ix.device
                )
                req.sr_tree_seed = (
                    row_p.detach().clone(),
                    row_ix.detach().clone(),
                    None,
                    verified_id,
                )
                stamp_tree_seed(
                    req, len(req.origin_input_ids or []) + len(req.output_ids or [])
                )
            if skipped:
                logger.warning(
                    "[SR] skip tree seed for %s: topk %s/%s batch=%s",
                    skipped,
                    tuple(topk_p_all.shape),
                    tuple(topk_index_all.shape),
                    n,
                )
        except Exception as e:
            if _sr_is_device_context_error(e):
                self._sr_device_poisoned = True
                self._sr_clear_stamps_after_poison()
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
        invalidate_tree_seed(req)
        self._sr_clear_prefix_stamp(self.sr_state.get(req.rid))
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
            if self._sr_tree_mode():
                self._sr_replay_grammars(list(batch.reqs), strict=True)
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
            self._sr_release_tree_lease(req.rid)
            fill_ids = snapshot_reprefill_fill_ids(
                req.origin_input_ids, req.output_ids
            )
            logger.info(
                "[SR] ingest tail %s tokens for %s: one reprefill of %s ids",
                len(req.output_ids or []),
                req.rid,
                len(fill_ids),
            )
            if not self._sr_enqueue_for_reprefill(req, fill_ids):
                continue
            self._sr_ensure_window_budget(req, req.draft_tokens_target)
            rebuilt.append(req)
        if rebuilt:
            self._sr_materialize_prefix_batch(rebuilt)

    def _sr_make_tail_extend_batch(self, plans) -> ScheduleBatch:
        return make_tail_extend_batch(self, plans)

    def _sr_tree_req_alive(self, req: Req) -> bool:
        if req is None:
            return False
        finished = getattr(req, "finished", None)
        if callable(finished):
            try:
                if finished():
                    return False
            except TypeError:
                pass
        if self._sr_is_degraded(req.rid):
            return False
        sr_state = getattr(self, "sr_state", None)
        if sr_state is not None:
            st = sr_state.get(req.rid)
            if st is None:
                return False
            obj = getattr(st, "req_object", None)
            if obj is not None and obj is not req:
                return False
        return True

    def _sr_prepare_tree_reqs(self, reqs: List[Req]) -> List[Req]:
        for req in reqs:
            if not self._sr_tree_req_alive(req):
                continue
            self._sr_ensure_window_budget(req, getattr(req, "draft_tokens_target", None))
        return [req for req in reqs if self._sr_tree_req_alive(req)]

    def _sr_inspect_lease_copy(self, req: Req, lease: SRTreeKVLease, runner):
        align = getattr(req, "sr_align_result", None)
        if align is None or align.kind not in ("append_one", "append_n"):
            return _SRTreePlanDraft(req, "ordinary", lease=lease, miss="fields")
        dreq = getattr(req, "sr_pending_dreq", None)
        indices = list(getattr(dreq, "commit_candidate_indices", None) or [])
        old_len = int(align.old_local_len)
        path_tokens = read_token_span(
            req.origin_input_ids, req.output_ids, old_len, len(indices)
        )
        miss = validate_lease_commit(
            lease,
            commit_tree_version=getattr(dreq, "commit_tree_version", None),
            commit_tree_base_committed_len=getattr(
                dreq, "commit_tree_base_committed_len", None
            ),
            commit_candidate_indices=indices,
            align=align,
            path_tokens=path_tokens,
        )
        if miss:
            return _SRTreePlanDraft(req, "ordinary", lease=lease, miss=miss)
        src_slots = live_accept_prefix(lease.candidate_slots, indices)
        original = int(align.old_kv_committed_len)
        effective = original + len(src_slots)
        try:
            plan = plan_tail_extend(
                req,
                vocab_size=self.model_config.vocab_size,
                model_is_mrope=runner.model_is_mrope,
                materialized_len=effective,
            )
        except TailExtendRecoveryRequired:
            return _SRTreePlanDraft(req, "recover", lease=lease)
        if plan is None:
            return _SRTreePlanDraft(req, "skip", lease=lease)
        plan = replace(
            plan,
            materialized_len=original,
            original_len=original,
            copy_src_slots=list(src_slots),
            copy_lease=lease,
        )
        return _SRTreePlanDraft(req, "copy", plan=plan, lease=lease)

    def _sr_inspect_tree_plans(self, reqs: List[Req]) -> List[_SRTreePlanDraft]:
        store = getattr(self, "sr_tree_leases", None)
        runner = self.tp_worker.model_runner
        drafts: List[_SRTreePlanDraft] = []
        for req in reqs:
            lease = store.get(req.rid) if store is not None else None
            if tree_seed_is_current(req):
                drafts.append(_SRTreePlanDraft(req, "skip", lease=lease))
                continue
            if lease is not None:
                draft = self._sr_inspect_lease_copy(req, lease, runner)
                if draft.kind == "copy":
                    drafts.append(draft)
                    continue
                if draft.kind == "recover":
                    drafts.append(draft)
                    continue
                ordinary_from_lease = draft
            else:
                ordinary_from_lease = None
            try:
                plan = plan_tail_extend(
                    req,
                    vocab_size=self.model_config.vocab_size,
                    model_is_mrope=runner.model_is_mrope,
                )
            except TailExtendRecoveryRequired:
                drafts.append(
                    _SRTreePlanDraft(
                        req,
                        "recover",
                        lease=lease,
                        miss=getattr(ordinary_from_lease, "miss", None),
                    )
                )
                continue
            if plan is None:
                drafts.append(
                    _SRTreePlanDraft(
                        req,
                        "skip",
                        lease=lease,
                        miss=getattr(ordinary_from_lease, "miss", None),
                    )
                )
                continue
            drafts.append(
                _SRTreePlanDraft(
                    req,
                    "ordinary",
                    plan=plan,
                    lease=lease,
                    miss=getattr(ordinary_from_lease, "miss", None),
                )
            )
        return drafts

    def _sr_record_inspect_misses(self, drafts: List[_SRTreePlanDraft]) -> None:
        store = getattr(self, "sr_tree_leases", None)
        if store is None:
            return
        for d in drafts:
            if not d.miss:
                continue
            store.counts[f"tree_kv_commit_miss_{d.miss}"] += 1
            if d.miss == "depth":
                store.counts["tree_kv_reuse_skip_depth"] += 1

    def _sr_acquire_tree_plans(
        self, drafts: List[_SRTreePlanDraft], held: List[SRTreeKVLease]
    ):
        store = getattr(self, "sr_tree_leases", None)
        runner = self.tp_worker.model_runner
        plans = []
        for d in drafts:
            if not self._sr_tree_req_alive(d.req):
                continue
            if d.kind == "skip":
                continue
            if d.kind == "recover":
                self._sr_mark_degraded(d.req.rid, "prefix still invalid after recovery")
                continue
            if d.kind == "copy":
                pinned = store.pin_lease(d.lease) if store is not None else None
                if pinned is not None:
                    held.append(pinned)
                    plans.append(replace(d.plan, copy_lease=pinned))
                    continue
                try:
                    plan = plan_tail_extend(
                        d.req,
                        vocab_size=self.model_config.vocab_size,
                        model_is_mrope=runner.model_is_mrope,
                    )
                except TailExtendRecoveryRequired:
                    self._sr_mark_degraded(
                        d.req.rid, "lease pin failed and prefix needs recovery"
                    )
                    continue
                if plan is not None:
                    plans.append(plan)
                continue
            if d.plan is not None:
                plans.append(d.plan)
        inspected = []
        seen = set()
        for d in drafts:
            if d.lease is None:
                continue
            key = id(d.lease)
            if key in seen:
                continue
            seen.add(key)
            inspected.append(d.lease)
        held_ids = {id(lease) for lease in held}
        unused = [
            lease
            for lease in inspected
            if id(lease) not in held_ids and not lease.released
        ]
        return plans, unused

    def _sr_release_unused_leases(self, unused: List[SRTreeKVLease]) -> None:
        store = getattr(self, "sr_tree_leases", None)
        if store is None:
            return
        allocator = self.token_to_kv_pool_allocator
        for lease in unused:
            store.release(lease, allocator=allocator)

    def _sr_release_held_leases(self, held: List[SRTreeKVLease]) -> None:
        store = getattr(self, "sr_tree_leases", None)
        if store is None:
            return
        allocator = self.token_to_kv_pool_allocator
        for lease in held:
            event = getattr(lease, "pending_free_event", None)
            store.release(lease, allocator=allocator, event=event)

    def _sr_record_committed_reuse(self, plans) -> None:
        store = getattr(self, "sr_tree_leases", None)
        if store is None:
            return
        for p in plans:
            src = p.copy_src_slots or []
            if not src or getattr(p, "copy_lease", None) is None:
                continue
            store.counts["tree_kv_commit_hit"] += 1
            store.counts["tree_kv_reused_tokens"] += len(src)
            store.counts["tree_kv_unmaterialized_leaf_tokens"] += max(p.length, 0)

    def _sr_run_tree_ingest(self, reqs: List[Req]) -> bool:
        """Prepare, inspect, recover once, pin, then execute one packed EXTEND."""
        if not reqs:
            return True
        metrics = get_sr_round_metrics(self, "Draft")
        plan_start = time.perf_counter()
        store = getattr(self, "sr_tree_leases", None)
        if store is not None:
            store.poll_pending_frees(self.token_to_kv_pool_allocator)
        prepared = self._sr_prepare_tree_reqs(reqs)
        drafts = self._sr_inspect_tree_plans(prepared)
        recover = [d.req for d in drafts if d.kind == "recover"]
        if recover:
            with metrics.phase("prefix_recovery", device=True):
                self._sr_reprefill_committed(recover)
            metrics.counts["prefix_recovered"] += len(recover)
            prepared = self._sr_prepare_tree_reqs(reqs)
            drafts = self._sr_inspect_tree_plans(prepared)
        self._sr_record_inspect_misses(drafts)
        if metrics.active:
            metrics.host["tail_plan_including_recovery"] += (
                time.perf_counter() - plan_start
            )
        held: List[SRTreeKVLease] = []
        try:
            plans, unused = self._sr_acquire_tree_plans(drafts, held)
            self._sr_release_unused_leases(unused)
            skip_count = sum(
                1
                for d in drafts
                if d.kind == "skip" and self._sr_tree_req_alive(d.req)
            )
            if skip_count:
                metrics.counts["seed_reused"] += skip_count
            if not plans:
                return True
            committed = self._sr_execute_tree_tails(plans)
            if committed:
                self._sr_record_committed_reuse(plans)
            return committed
        finally:
            self._sr_release_held_leases(held)

    def _sr_ingest_tree_tails(self, reqs: List[Req], plans=None):
        if plans is None:
            return self._sr_run_tree_ingest(reqs)
        return self._sr_execute_tree_tails(plans)

    def _sr_execute_tree_tails(self, plans) -> bool:
        """Run one transaction on final plans. Does not plan, recover, or pin."""
        if not plans:
            return True
        metrics = get_sr_round_metrics(self, "Draft")
        runner = self.tp_worker.model_runner
        transaction = SRTailExtendTransaction(self, plans)
        active = [p.req for p in plans]
        worker_failed = False
        try:
            with metrics.phase("tail_prepare_allocate"):
                self._sr_replay_grammars(active, strict=True)
                batch = self._sr_make_tail_extend_batch(plans)
                transaction.allocate(batch)
                transaction.copy_reused_tree_kv()
                worker_batch = batch.get_model_worker_batch()
            metrics.counts["tail_requests"] += len(plans)
            metrics.counts["tail_tokens"] += sum(p.length for p in plans)
            metrics.counts["seed_recaptured"] += sum(p.recapture for p in plans)
            for p in plans:
                metrics.counts[f"tail_len_{p.length if p.length <= 16 else 'gt16'}"] += 1
            with metrics.phase("tail_forward_seed_commit", device=True):
                drafter = getattr(self, "sr_tree_drafter", None)
                tail_runner = (
                    getattr(drafter, "tail_graph_runner", None) if drafter else None
                )
                plan = None
                miss_reason = "no_runner"
                logits_output = None
                forward_batch = None
                if tail_runner is not None:
                    if hasattr(tail_runner, "plan_with_reason"):
                        plan, miss_reason = tail_runner.plan_with_reason(batch)
                    else:
                        plan = tail_runner.plan(batch)
                        miss_reason = "ok" if plan is not None else "no_bucket"
                    if plan is not None:
                        try:
                            forward_batch = tail_runner.init_forward_batch(worker_batch)
                            tail_runner.fill(forward_batch, plan)
                        except NpuGraphPreparationError as e:
                            logger.warning(
                                "[SR] tail graph prep failed: %s; falling back to eager",
                                e,
                            )
                            miss_reason = "prep"
                            plan = None
                transaction.wait_copy_done()
                if plan is not None:
                    transaction.submitted = True
                    logits_output = tail_runner.replay_filled(plan)
                    tail_runner.model_runner.capture_tree_seed_only(
                        logits_output, forward_batch
                    )
                    metrics.paths["tail_extend_graph"] += 1
                    paths = getattr(
                        runner.attn_backend, "sr_tail_attention_paths", None
                    )
                    if paths:
                        for path in paths:
                            metrics.paths[path] += 1
                else:
                    if tail_runner is not None:
                        tail_runner.eager_fallback_count = (
                            getattr(tail_runner, "eager_fallback_count", 0) + 1
                        )
                    metrics.counts[f"tail_graph_miss_{miss_reason}"] += 1
                    transaction.submitted = True
                    result = self.tp_worker.forward_batch_generation(
                        worker_batch, seed_only=True
                    )
                    logits_output = result.logits_output
                    paths = getattr(
                        runner.attn_backend, "sr_tail_attention_paths", None
                    )
                    for path in paths or ("ordinary_extend",):
                        metrics.paths[path] += 1
                transaction.commit(logits_output)
                return True
        except Exception as e:
            copy_submitted = bool(getattr(transaction, "copy_submitted", False))
            if _sr_is_device_context_error(e):
                self._sr_device_poisoned = True
                self._sr_clear_stamps_after_poison()
            if (
                getattr(self, "_sr_device_poisoned", False)
                or isinstance(e, NpuGraphReplaySubmittedError)
                or (self.tp_size > 1 and (transaction.submitted or copy_submitted))
            ):
                if copy_submitted:
                    self._sr_park_copy_hold(getattr(transaction, "_copy_hold", None))
                worker_failed = True
                raise
            try:
                transaction.rollback()
            except Exception:
                if copy_submitted:
                    self._sr_park_copy_hold(getattr(transaction, "_copy_hold", None))
                worker_failed = True
                raise
            if self.tp_size > 1:
                worker_failed = True
                raise
            metrics.counts["tail_failed_requests"] += len(active)
            for req in active:
                self._sr_mark_degraded(req.rid, f"tail extend failed: {e}")
            return False
        finally:
            if not worker_failed:
                for req in active:
                    self._sr_pause_req(req)
            self.last_batch = None

    def _sr_ingest_committed_batch(self, reqs: List[Req]) -> None:
        if self._sr_tree_mode():
            self._sr_run_tree_ingest(reqs)
            return
        self._sr_ingest_committed_chain_batch(reqs)

    def _sr_ingest_committed_chain_batch(self, reqs: List[Req]) -> None:
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

    def _sr_ensure_tree_seeds(self, reqs: List[Req]) -> None:
        """Rebuild seed when prefix KV is complete but last-token logits are gone."""
        if self._sr_tree_mode():
            # Ingest already planned both missing tails and seed recapture.
            # Do not retry a failed transaction or run per-request decodes here.
            for req in reqs:
                if not self._sr_is_degraded(req.rid) and not tree_seed_is_current(req):
                    self._sr_mark_degraded(req.rid, "tree seed recovery failed")
            return
        need_ingest: List[Req] = []
        recapture: List[Req] = []
        reprefill: List[Req] = []
        for req in reqs:
            committed = int(getattr(req, "kv_committed_len", 0) or 0)
            seed_ok = tree_seed_matches_prefix(
                getattr(req, "sr_tree_seed", None),
                req.origin_input_ids or [],
                req.output_ids or [],
            )
            can_last = False
            kv = getattr(self, "sr_kv", None)
            if kv is not None and committed > 0:
                can_last = bool(kv.can_local_rollback(req, committed - 1))
            action = plan_tree_seed_recovery(
                len(req.origin_input_ids or []),
                req.output_ids,
                committed,
                seed_ok,
                can_last,
            )
            if action == "ingest":
                need_ingest.append(req)
            elif action == "recapture_last":
                recapture.append(req)
            elif action == "reprefill":
                reprefill.append(req)
        if need_ingest:
            self._sr_ingest_committed_batch(need_ingest)
        failed_recapture: List[Req] = []
        for req in recapture:
            committed = int(getattr(req, "kv_committed_len", 0) or 0)
            allocated = int(getattr(req, "kv_allocated_len", 0) or 0)
            if committed <= 0 or not self.sr_kv.rollback(
                req, committed - 1, allocated
            ):
                failed_recapture.append(req)
                continue
            self._sr_ingest_committed_batch([req])
        need_reprefill = reprefill + failed_recapture
        if need_reprefill:
            self._sr_reprefill_committed(need_reprefill)
        for req in reqs:
            if tree_seed_matches_prefix(
                getattr(req, "sr_tree_seed", None),
                req.origin_input_ids or [],
                req.output_ids or [],
            ):
                continue
            self._sr_mark_degraded(req.rid, "tree seed recovery failed")

    def _sr_park_copy_hold(self, hold) -> None:
        if hold is None:
            return
        bag = getattr(self, "_sr_pending_copy_holds", None)
        if bag is None:
            self._sr_pending_copy_holds = [hold]
            return
        bag.append(hold)

    def _sr_release_tree_lease(self, rid: str, event=None) -> None:
        store = getattr(self, "sr_tree_leases", None)
        if store is None:
            return
        store.release_rid(
            rid, allocator=self.token_to_kv_pool_allocator, event=event
        )

    def _sr_tree_expand_batch(self, reqs: List[Req]) -> List[SRWindow]:
        empty: SRWindow = ([], None, None)
        if not reqs:
            return []
        metrics = get_sr_round_metrics(self, "Draft")
        with metrics.phase("prefix_materialize", device=True):
            self._sr_materialize_prefix_batch(reqs)
        self._sr_run_tree_ingest(reqs)
        self._sr_ensure_tree_seeds(reqs)
        self._sr_replay_grammars(reqs)
        windows: List[SRWindow] = [empty] * len(reqs)
        ready: List[Req] = []
        ready_idx: List[int] = []
        for i, req in enumerate(reqs):
            if self._sr_is_degraded(req.rid):
                continue
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
            with metrics.phase("tree_expand_pack"):
                got = self.sr_tree_drafter.expand_batch(ready)
        except NpuGraphReplaySubmittedError:
            raise
        except Exception as e:
            if _sr_is_device_context_error(e):
                self._sr_device_poisoned = True
                self._sr_clear_stamps_after_poison()
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
            # An empty window is not a completed commit. The KV may be partial.
            for req in ready:
                req.sr_commit_incomplete = True
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
        self._sr_release_tree_lease(req.rid)
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
            tree_version=self._sr_reply_tree_version(dreq.rid, status),
        )

    def _sr_reply_tree_version(self, rid: str, status: SRReplyStatus):
        store = getattr(self, "sr_tree_leases", None)
        if store is None:
            return None
        lease = store.get(rid)
        if lease is None:
            return None
        return lease.version

    def _sr_v2_reply(
        self,
        dreq: SRDraftRequest,
        status: SRReplyStatus,
        *,
        reason: Optional[str] = None,
        window: Optional[SRWindow] = None,
        tree_version: Optional[int] = None,
        ack_version: Optional[int] = None,
        ack_len: Optional[int] = None,
    ) -> SRDraftReply:
        tokens, parent_list, top_index = window or ([], None, None)
        return SRDraftReply(
            rid=dreq.rid,
            step_id=dreq.step_id,
            base_committed_len=dreq.base_committed_len,
            draft_tokens=list(tokens),
            status=status,
            parent_list=parent_list,
            top_scores_index=top_index,
            tree_version=tree_version,
            ack_commit_version=ack_version,
            ack_output_len=ack_len,
            reason=reason,
        )

    def _sr_cached_reply(self, dreq: SRDraftRequest, state: SRDraftState) -> Optional[SRDraftReply]:
        cached = state.last_reply
        if not cached:
            return None
        return SRDraftReply(
            rid=dreq.rid,
            step_id=dreq.step_id,
            base_committed_len=dreq.base_committed_len,
            draft_tokens=list(cached.get("draft_tokens") or []),
            status=SRReplyStatus.IDEMPOTENT,
            parent_list=cached.get("parent_list"),
            top_scores_index=cached.get("top_scores_index"),
            tree_version=cached.get("tree_version"),
            ack_commit_version=cached.get("ack_commit_version"),
            ack_output_len=cached.get("ack_output_len"),
        )

    def _sr_remember_commit(
        self,
        state: SRDraftState,
        dreq: SRDraftRequest,
        reply: SRDraftReply,
        fingerprint: Tuple,
    ) -> None:
        state.acked_version = int(dreq.commit_version)
        state.commit_trusted = True
        state.completion_unknown = False
        state.last_commit_fingerprint = fingerprint
        state.last_num_draft_tokens = int(dreq.num_draft_tokens or 0)
        state.last_step_id = dreq.step_id
        state.last_base_committed_len = dreq.base_committed_len
        state.last_reply = {
            "draft_tokens": list(reply.draft_tokens or []),
            "parent_list": reply.parent_list,
            "top_scores_index": reply.top_scores_index,
            "tree_version": reply.tree_version,
            "ack_commit_version": reply.ack_commit_version,
            "ack_output_len": reply.ack_output_len,
        }
        state.last_window = (
            list(reply.draft_tokens or []),
            reply.parent_list,
            reply.top_scores_index,
        )

    def _sr_mm_can_restore(self, req: Optional[Req], dreq: SRDraftRequest, mm) -> bool:
        if not getattr(dreq, "requires_mm", False):
            return True
        if (
            req is not None
            and getattr(req, "multimodal_inputs", None) is not None
            and getattr(req, "sr_commit_kv_trusted", True)
        ):
            return True
        items = []
        if mm is not None:
            items = list(getattr(mm, "mm_items", None) or [])
        return mm_items_complete(items)

    def _sr_reuse_zero_delta(self, dreq, req, state, verdict) -> bool:
        if verdict.outcome != CommitOutcome.APPLY or verdict.delta_ids != ():
            return False
        if state is None or state.last_window is None or state.pending_commit is not None:
            return False
        if (
            not state.commit_trusted
            or state.completion_unknown
            or state.degraded
            or getattr(self, "_sr_device_poisoned", False)
        ):
            return False
        return (
            state.last_step_id == dreq.step_id
            and state.last_base_committed_len == dreq.base_committed_len
            and state.last_num_draft_tokens == int(dreq.num_draft_tokens or 0)
        )

    def _sr_confirm_zero_delta(self, dreq, req, state, reply) -> None:
        fingerprint = commit_fingerprint(dreq)
        state.acked_version = int(dreq.commit_version)
        if req is None:
            state.local_prefix_stamp = None
        else:
            state.local_prefix_stamp = retarget_stamp_version(
                state.local_prefix_stamp,
                int(dreq.commit_version),
                req=req,
                origin=req.origin_input_ids,
                output=req.output_ids,
                revision=int(getattr(req, "sr_prefix_revision", 0) or 0),
            )
            req.sr_commit_output_ids = CommittedPrefixView(
                state.acked_output_ids, len(state.acked_output_ids), ()
            )
        self._sr_remember_commit(state, dreq, reply, fingerprint)

    def _sr_prepare_delta_fast(self, dreq, req, state, verdict, metrics) -> None:
        delta = verdict.delta_ids or ()
        fingerprint = commit_fingerprint(dreq)
        origin = req.origin_input_ids
        output = req.output_ids
        old_kv = int(getattr(req, "kv_committed_len", 0) or 0)
        old_revision = int(getattr(req, "sr_prefix_revision", 0) or 0)
        window = prefix_window_tokens(origin or [], output or [], base=old_kv)
        old_local_len = len(origin or []) + len(output or [])
        if delta:
            invalidate_tree_seed(req)
            if req.output_ids is None:
                req.output_ids = []
            req.output_ids.extend(delta)
        kind = (
            "equal"
            if not delta
            else "append_one"
            if len(delta) == 1
            else "append_n"
        )
        req.sr_align_result = SRAlignResult(
            kind=kind,
            old_kv_committed_len=old_kv,
            old_prefix_revision=old_revision,
            old_local_len=old_local_len,
            old_prefix_window=window,
            fork=old_local_len,
        )
        req.sr_prefix_proven = True
        req.sr_pending_dreq = dreq
        req.sr_pending_wire_request = dreq
        req.draft_generation_start_len = len(req.output_ids or [])
        req.draft_tokens_target = int(dreq.num_draft_tokens or 0)
        base_len = len(state.acked_output_ids)
        req.sr_commit_output_ids = CommittedPrefixView(
            state.acked_output_ids, base_len, delta
        )
        state.pending_commit = PendingCommit(
            old_version=state.acked_version,
            old_output_len=base_len,
            delta_ids=delta,
            expected_len=base_len + len(delta),
            fingerprint=fingerprint,
            candidate_stamp=make_prefix_stamp(req, int(dreq.commit_version)),
        )
        note_commit_result(metrics, "commit_fast_apply")

    def _sr_stage_slow_commit(
        self, dreq, req, state, verdict, output_ids, fingerprint
    ) -> None:
        proven = bool(getattr(req, "sr_prefix_proven", False))
        candidate = (
            make_prefix_stamp(req, int(dreq.commit_version)) if proven else None
        )
        if not proven:
            self._sr_clear_prefix_stamp(state)
        if verdict.delta_ids is not None:
            base = [] if state is None else state.acked_output_ids
            base_len = 0 if state is None else len(state.acked_output_ids)
            pending = PendingCommit(
                old_version=None if state is None else state.acked_version,
                old_output_len=base_len,
                delta_ids=verdict.delta_ids,
                expected_len=base_len + len(verdict.delta_ids),
                fingerprint=fingerprint,
                candidate_stamp=candidate,
            )
            req.sr_commit_output_ids = CommittedPrefixView(
                base, base_len, verdict.delta_ids
            )
        else:
            snap = tuple(int(x) for x in output_ids)
            pending = PendingCommit(
                old_version=None if state is None else state.acked_version,
                old_output_len=0 if state is None else len(state.acked_output_ids),
                delta_ids=(),
                expected_len=len(snap),
                fingerprint=fingerprint,
                candidate_stamp=candidate,
                snapshot_ids=snap,
            )
            req.sr_commit_output_ids = list(snap)
        if state is not None:
            state.pending_commit = pending
        req.sr_pending_wire_request = dreq

    def _sr_prepare_v2(
        self,
        dreq: SRDraftRequest,
        action: SRAction,
        session_id: str,
        mm: Optional[SRMMPayload],
    ) -> Tuple[Optional[SRDraftReply], Optional[Req], Optional[SRDraftRequest]]:
        """Returns (early_reply, live_req, local_request)."""
        metrics = get_sr_round_metrics(self, "Draft")
        state = self.sr_state.get(dreq.rid)
        current_session = self.sr_state.session_id
        if current_session is not None and session_id < current_session:
            return (
                self._sr_v2_reply(dreq, SRReplyStatus.REJECT, reason="stale_session"),
                None,
                None,
            )
        self._sr_drop_unconfirmed_commit(state)
        req = state.req_object if state is not None else None
        if state is not None and state.completion_unknown:
            return (
                self._sr_v2_reply(
                    dreq, SRReplyStatus.REJECT, reason="completion_unknown"
                ),
                None,
                None,
            )
        verdict = inspect_commit(
            dreq,
            action=action,
            current_version=None if state is None else state.acked_version,
            current_output=None if state is None else state.acked_output_ids,
            prompt_len=0 if state is None else state.prompt_len,
            cached_fingerprint=None if state is None else state.last_commit_fingerprint,
            has_state=state is not None and req is not None,
            state_trusted=True if state is None else state.commit_trusted,
        )
        poisoned = bool(getattr(self, "_sr_device_poisoned", False))
        if poisoned and verdict.outcome in (CommitOutcome.APPLY, CommitOutcome.CACHE):
            return (
                self._sr_v2_reply(dreq, SRReplyStatus.REJECT, reason="untrusted_kv"),
                None,
                None,
            )
        if verdict.outcome == CommitOutcome.CONTROL:
            if action in (SRAction.FINISH, SRAction.ABORT):
                self._sr_finish_rid(dreq.rid)
            return self._sr_v2_reply(dreq, SRReplyStatus.OK), None, None
        if verdict.outcome == CommitOutcome.CACHE:
            cached = self._sr_cached_reply(dreq, state)
            note_commit_result(metrics, "commit_idempotent_hits")
            if cached is None:
                return (
                    self._sr_v2_reply(dreq, SRReplyStatus.REJECT, reason="missing_cache"),
                    None,
                    None,
                )
            return cached, None, None
        if verdict.outcome == CommitOutcome.REJECT:
            return (
                self._sr_v2_reply(dreq, SRReplyStatus.REJECT, reason=verdict.reason),
                None,
                None,
            )
        if verdict.outcome == CommitOutcome.NEED_SNAPSHOT:
            note_commit_result(metrics, "commit_need_snapshot")
            return (
                self._sr_v2_reply(
                    dreq, SRReplyStatus.NEED_SNAPSHOT, reason=verdict.reason
                ),
                None,
                None,
            )

        if self._sr_reuse_zero_delta(dreq, req, state, verdict):
            reply = self._sr_v2_reply(
                dreq,
                SRReplyStatus.OK if state.last_window[0] else SRReplyStatus.EMPTY,
                window=state.last_window,
                tree_version=(state.last_reply or {}).get("tree_version"),
                ack_version=int(dreq.commit_version),
                ack_len=len(state.acked_output_ids),
            )
            self._sr_confirm_zero_delta(dreq, req, state, reply)
            return reply, None, None
        if delta_fast_allowed(
            verdict,
            req=req,
            state=state,
            base_output_len=int(dreq.base_output_len or 0),
            poisoned=poisoned,
        ):
            self._sr_prepare_delta_fast(dreq, req, state, verdict, metrics)
            return None, req, dreq

        output_ids = (
            rebuild_committed_output(
                [] if state is None else state.acked_output_ids,
                verdict.delta_ids,
            )
            if verdict.delta_ids is not None
            else list(verdict.output_ids or [])
        )
        padded = getattr(req, "sr_padded_ids", None) if req is not None else None
        prompt = list(padded or dreq.padded_input_ids or [])
        if dreq.commit_mode == "snapshot":
            prompt = list(dreq.padded_input_ids or prompt)
        local = replace(dreq, committed_ids=list(output_ids))
        route = RecoveryRoute.ALIGN
        if verdict.outcome == CommitOutcome.FORCE_RESET:
            route = route_snapshot_recovery(
                kv_trusted=False,
                degraded=bool(state is not None and state.degraded),
                poisoned=poisoned,
                completion_unknown=bool(
                    state is not None and state.completion_unknown
                ),
            )
        if route == RecoveryRoute.BLOCKED:
            return (
                self._sr_v2_reply(dreq, SRReplyStatus.REJECT, reason="untrusted_kv"),
                None,
                None,
            )
        if not self._sr_mm_can_restore(req, dreq, mm) and (
            req is None or route == RecoveryRoute.REPREFILL
        ):
            return (
                self._sr_v2_reply(
                    dreq, SRReplyStatus.REJECT, reason="unrecoverable_mm"
                ),
                None,
                None,
            )
        if req is None and dreq.commit_mode != "snapshot":
            return (
                self._sr_v2_reply(dreq, SRReplyStatus.NEED_SNAPSHOT, reason="no_req"),
                None,
                None,
            )
        note_commit_result(metrics, "commit_slow_apply")
        fingerprint = commit_fingerprint(dreq)

        if req is None:
            req = self._sr_create_req(dreq, mm, session_id)
            if req is None:
                return self._sr_v2_reply(dreq, SRReplyStatus.EMPTY), None, None
            state = self.sr_state.get(dreq.rid)
        elif route == RecoveryRoute.REPREFILL:
            fill = list(prompt) + list(output_ids)
            self._sr_reprefill(req, fill, local, state)
        else:
            self._sr_align(req, local, state)
        self._sr_stage_slow_commit(
            dreq, req, state, verdict, output_ids, fingerprint
        )
        if state is not None and dreq.commit_mode == "snapshot":
            state.prompt_len = len(prompt)
        return None, req, local

    def _sr_finalize_v2_commit(
        self,
        dreq: SRDraftRequest,
        req: Req,
        reply: SRDraftReply,
    ) -> SRDraftReply:
        state = self.sr_state.get(dreq.rid)
        pending = None if state is None else state.pending_commit
        if getattr(req, "sr_commit_incomplete", False) or (
            state is not None and state.completion_unknown
        ):
            if state is not None:
                state.commit_trusted = False
                state.completion_unknown = True
                state.pending_commit = None
                state.local_prefix_stamp = None
            reply.ack_commit_version = None
            reply.ack_output_len = None
            reply.status = SRReplyStatus.EMPTY
            reply.reason = "commit_incomplete"
            reply.draft_tokens = []
            return reply
        if state is None or pending is None or (
            state.acked_version != pending.old_version
            or len(state.acked_output_ids) != pending.old_output_len
        ):
            if state is not None:
                state.pending_commit = None
                state.local_prefix_stamp = None
            reply.ack_commit_version = None
            reply.ack_output_len = None
            reply.status = SRReplyStatus.EMPTY
            reply.reason = "commit_incomplete"
            reply.draft_tokens = []
            return reply
        if pending.snapshot_ids is not None:
            state.acked_output_ids = list(pending.snapshot_ids)
        else:
            state.acked_output_ids.extend(pending.delta_ids)
        if candidate_stamp_current(pending.candidate_stamp, req):
            state.local_prefix_stamp = pending.candidate_stamp
        else:
            state.local_prefix_stamp = None
        fingerprint = pending.fingerprint
        ack_len = pending.expected_len
        state.pending_commit = None
        reply.ack_commit_version = int(dreq.commit_version)
        reply.ack_output_len = ack_len
        self._sr_remember_commit(state, dreq, reply, fingerprint)
        return reply

    def _sr_protocol_mismatch_reply(self, batch: SRBatchRequest) -> SRBatchReply:
        return SRBatchReply(
            session_id=batch.session_id,
            rpc_seq=batch.rpc_seq,
            protocol_version=SR_PROTOCOL_VERSION,
            reqs=[
                SRDraftReply(
                    rid=dreq.rid,
                    step_id=dreq.step_id,
                    base_committed_len=dreq.base_committed_len,
                    draft_tokens=[],
                    status=SRReplyStatus.REJECT,
                    reason="protocol_mismatch",
                )
                for dreq in batch.reqs
            ],
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
                        tree_version=self._sr_reply_tree_version(
                            dreq.rid, SRReplyStatus.IDEMPOTENT
                        ),
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
        if (
            self._sr_tree_mode() and batch.action == SRAction.STEP
            and not get_sr_round_metrics(self, "Draft").active
        ):
            with get_sr_round_metrics(self, "Draft").round():
                return self._sr_handle_batch_impl(batch, mm_by_rid)
        return self._sr_handle_batch_impl(batch, mm_by_rid)

    def _sr_handle_batch_impl(
        self, batch: SRBatchRequest, mm_by_rid: Dict[str, SRMMPayload]
    ) -> SRBatchReply:
        metrics = get_sr_round_metrics(self, "Draft")
        prepare_start = time.perf_counter()
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
        use_v2 = batch.protocol_version == SR_PROTOCOL_VERSION
        # After a batch-level wipe, later rids must see the new session so
        # decide_draft_action does not WIPE_NEW_SESSION again and drop siblings.
        prepare_last_session = (
            batch.session_id if wiped_this_batch else last_session
        )
        prepare_last_rpc = -1 if wiped_this_batch else last_rpc
        for i, dreq in enumerate(batch.reqs):
            if use_v2:
                reply, req, _local = self._sr_prepare_v2(
                    dreq, batch.action, batch.session_id, mm_by_rid.get(dreq.rid)
                )
            else:
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
        if metrics.active:
            metrics.host["align_prepare"] += time.perf_counter() - prepare_start
        if gpu_pairs:
            windows = self._sr_produce_windows(
                batch.action, [(dreq, req) for _, dreq, req in gpu_pairs]
            )
            reply_start = time.perf_counter()
            for (i, dreq, req), window in zip(gpu_pairs, windows):
                wire = getattr(req, "sr_pending_wire_request", None) or dreq
                reply = self._sr_stamp_window(
                    wire, window, batch.rpc_seq, batch.session_id
                )
                if use_v2:
                    reply = self._sr_finalize_v2_commit(wire, req, reply)
                replies[i] = reply
            if metrics.active:
                metrics.host["reply_prepare"] += time.perf_counter() - reply_start
        if self.sr_server is not None:
            self.sr_server.remember(batch)
        self._sr_note_handled(batch)
        return SRBatchReply(
            session_id=batch.session_id,
            rpc_seq=batch.rpc_seq,
            protocol_version=SR_PROTOCOL_VERSION if use_v2 else None,
            reqs=[r if r is not None else self._sr_empty_reply(d) for r, d in zip(replies, batch.reqs)],
        )

    def _sr_note_handled(self, batch: SRBatchRequest) -> None:
        self.sr_handled_session = batch.session_id
        self.sr_handled_rpc = int(batch.rpc_seq)

    def _sr_batch_is_stale(self, batch: SRBatchRequest) -> bool:
        """Same decision on every TP rank. Do not read rank 0's server cursor."""
        last_session = getattr(self, "sr_handled_session", None)
        last_rpc = int(getattr(self, "sr_handled_rpc", -1) or -1)
        if last_session is not None and batch.session_id < last_session:
            return True
        if last_session == batch.session_id and int(batch.rpc_seq) <= last_rpc:
            return True
        return False

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
                if self._sr_batch_is_stale(batch):
                    if self.tp_rank == 0:
                        logger.info(
                            "[SR] Draft drop stale session=%s rpc_seq=%s",
                            batch.session_id,
                            batch.rpc_seq,
                        )
                else:
                    metrics = get_sr_round_metrics(self, "Draft")
                    measure = self._sr_tree_mode() and batch.action == SRAction.STEP
                    with metrics.round() if measure else nullcontext():
                        if (
                            batch.protocol_version is not None
                            and batch.protocol_version != SR_PROTOCOL_VERSION
                        ):
                            reply = self._sr_protocol_mismatch_reply(batch)
                            self._sr_note_handled(batch)
                        else:
                            reply = self._sr_handle_batch(batch, mm)
                        if (
                            (self.tp_size == 1 or self.tp_rank == 0)
                            and self.sr_server is not None
                            and batch.action not in (SRAction.FINISH, SRAction.ABORT)
                        ):
                            with metrics.phase("reply_send"):
                                self.sr_server.send_batch(reply)
                self.last_batch = None
                self._sr_resume_http_reqs()
                continue
            self._sr_cleanup_stale_drafts()
            self._sr_run_http_batch()
