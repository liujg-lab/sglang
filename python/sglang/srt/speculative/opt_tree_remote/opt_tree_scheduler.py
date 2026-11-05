"""OPT_TREE_REMOTE scheduling.

The send/recv loop follows the standalone-remote RPC session: Target asks,
Draft answers, and a missed reply falls back inside that session. Tree growth
does not use the EAGLE top-k drafter. Each kept layer is one masked forward.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.opt_tree_remote.opt_tree_config import OptTreeConfig
from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
    LookaheadFallback,
    OptLayerForward,
    _admit_lookahead,
    _reraise_device_context,
    clear_prefill_tail_logits,
    critical_layer_forwards,
    drain_opt_layer_gpu_us,
    drop_lookahead_request,
    forward_deferred_final_layers,
    grow_active_layers,
    grow_lookahead,
    ingest_tail_logits,
    lookahead_supported,
    opt_other_seconds,
    prepare_lookahead_runtime,
    produce_opt_windows,
    release_uncommitted,
)
from sglang.srt.speculative.standalone_remote.sr_circuit_breaker import SRRpcBreaker
from sglang.srt.speculative.standalone_remote.sr_commit import (
    SR_PROTOCOL_VERSION,
    supported_protocol,
)
from sglang.srt.speculative.standalone_remote.sr_mm_payload import SRMMPayload
from sglang.srt.speculative.standalone_remote.sr_protocol import (
    SRAction,
    SRBatchReply,
    SRBatchRequest,
    SRDraftRequest,
    SRReplyStatus,
)
from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
    get_sr_round_metrics,
    idle_sr_round_metrics,
)
from sglang.srt.speculative.standalone_remote.sr_transport import (
    SRDraftServer,
    SRTargetClient,
)
from sglang.srt.utils import DynamicGradMode

logger = logging.getLogger(__name__)


def make_opt_tree_transport(server_args):
    """Bind or connect the OPT RPC socket. Addresses come from ``--opt-tree-*``."""
    role = server_args.opt_tree_role
    addr = server_args.opt_tree_addr or "127.0.0.1"
    port = str(server_args.opt_tree_port or "30029")
    if role == "draft":
        return SRDraftServer(addr, port)
    timeout_ms = int(
        getattr(server_args, "standalone_remote_rpc_timeout_ms", 5000) or 5000
    )
    return SRTargetClient(addr, port, timeout_ms=timeout_ms)


def run_opt_draft_round(metrics, measure, work) -> None:
    """Run one draft step, then ``work``'s post-reply, before the window flushes.

    ``round`` snapshots and clears ``counts`` on the way out of a full window.
    ``work`` receives that same counter and must finish its adds before return.
    ``host["total"]`` covers this whole call, including the deferred final
    layer and a later layer-event drain. The pre-reply wall clock is
    ``counts["opt_pre_reply_us"]``.
    """
    with metrics.round() if measure else nullcontext():
        work(metrics.counts)


def drain_deferred_layer_events(scheduler, metrics, *, defer_drain, jobs) -> None:
    """Read layer events after the reply when this step still has queued work.

    Lookahead sets ``defer_drain`` because its last layer was submitted before
    the reply. ``grow_batch`` and a non-lookahead expand do not. A deferred
    final layer still queues events inside ``forward_deferred_final_layers``.
    Either case is drained here, after the reply, so the sync stays in this
    step. An empty queue does not synchronize.
    """
    if defer_drain or jobs:
        drain_opt_layer_gpu_us(scheduler, metrics)


class OptTreeSchedulerMixin:
    @property
    def is_remote_spec_draft(self) -> bool:
        if (
            self.spec_algorithm.is_opt_tree_remote()
            and self.server_args.opt_tree_role == "draft"
        ):
            return True
        return super().is_remote_spec_draft

    def _sr_materialize_prefix_batch(
        self,
        reqs,
        on_extend_logits=None,
        logits_survive=None,
        capture_tree_seed: bool = True,
    ):
        """OPT grows from tail logits and does not read ``sr_tree_seed``."""
        if self.spec_algorithm.is_opt_tree_remote():
            capture_tree_seed = False
        return super()._sr_materialize_prefix_batch(
            reqs,
            on_extend_logits=on_extend_logits,
            logits_survive=logits_survive,
            capture_tree_seed=capture_tree_seed,
        )

    def maybe_notify_remote_draft_finished(self, req) -> None:
        if (
            self.spec_algorithm.is_opt_tree_remote()
            and self.server_args.opt_tree_role == "target"
        ):
            self.notify_sr_draft_finished(req, SRAction.FINISH)
            return
        return super().maybe_notify_remote_draft_finished(req)

    def maybe_abort_remote_target(self) -> None:
        """Tell the draft to drop KV before the target session rotates."""
        if not (
            self.spec_algorithm.is_opt_tree_remote()
            and self.server_args.opt_tree_role == "target"
        ):
            parent = getattr(super(), "maybe_abort_remote_target", None)
            if callable(parent):
                parent()
            return
        notify_active = getattr(self, "notify_sr_draft_aborted_active", None)
        if callable(notify_active):
            notify_active()
        reset = getattr(self, "reset_standalone_remote_target_state", None)
        if callable(reset):
            reset()

    def maybe_notify_remote_draft_aborted(self, reqs) -> None:
        """Same single-request draft abort as standalone-remote, for OPT target."""
        if not (
            self.spec_algorithm.is_opt_tree_remote()
            and self.server_args.opt_tree_role == "target"
        ):
            return super().maybe_notify_remote_draft_aborted(reqs)
        if not reqs:
            return
        notify = getattr(self, "notify_sr_draft_aborted", None)
        if callable(notify):
            notify(reqs)

    def init_opt_tree_communication(self) -> None:
        if not self.spec_algorithm.is_opt_tree_remote():
            return
        role = self.server_args.opt_tree_role
        if role not in ("target", "draft"):
            raise ValueError(f"Invalid OPT_TREE_REMOTE role: {role}")
        self.sr_client = None
        self.sr_breaker = None
        self.sr_speculation_stopped = False
        if role == "draft":
            self.paused_reqs: List = []
            self.paused_reqs_lock = threading.RLock()
            self.draft_paused_reqs: List = []

    def _init_opt_tree_target(self) -> None:
        self._sync_sr_session_id()
        self.sr_rpc_seq = 0
        self.sr_client = None
        self._sr_hold_finish_flush = False
        self.sr_breaker = SRRpcBreaker(
            failure_threshold=int(
                getattr(self.server_args, "standalone_remote_breaker_failures", 3)
                or 3
            ),
            cooldown_steps=int(
                getattr(self.server_args, "standalone_remote_breaker_cooldown", 32)
                or 32
            ),
        )
        if self.tp_size == 1 or self.tp_rank == 0:
            client = make_opt_tree_transport(self.server_args)
            assert isinstance(client, SRTargetClient)
            client.drop_callback = self._sr_note_stale
            self.sr_client = client
        if self.tp_rank == 0:
            logger.info(
                "[OPT] Target session_id=%s addr=%s port=%s",
                self.sr_session_id,
                self.server_args.opt_tree_addr,
                self.server_args.opt_tree_port,
            )

    def _init_opt_tree_draft(self) -> None:
        from sglang.srt.speculative.standalone_remote.drafter.sr_draft_state import (
            SRDraftStateManager,
        )
        from sglang.srt.speculative.standalone_remote.sr_kv_rollbacker import (
            SRKVRollbacker,
        )

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
        self.sr_server = None
        if self.tp_size == 1 or self.tp_rank == 0:
            server = make_opt_tree_transport(self.server_args)
            assert isinstance(server, SRDraftServer)
            self.sr_server = server
        self.draft_paused_reqs = []
        self.sr_tree_drafter = None
        self._sr_device_poisoned = False
        self._opt_deferred = []
        if self.tp_rank == 0:
            logger.info(
                "[OPT] Draft scheduler ready width=%s max_depth=%s tau=%s addr=%s port=%s",
                self.server_args.opt_tree_width,
                self.server_args.opt_tree_max_depth,
                self.server_args.opt_tree_tau,
                self.server_args.opt_tree_addr,
                self.server_args.opt_tree_port,
            )

    def _opt_run_verify_round(self, batch) -> None:
        """Same verify round as standalone-remote.

        Target ``--speculative-num-draft-tokens`` is already N + 1, the
        committed tail plus the N draft nodes, so the default forward length
        is the verify length.
        """
        self._sr_run_verify_round(batch)

    def _opt_rollback(self, req, keep_len: int) -> None:
        kv = getattr(self, "sr_kv", None)
        if kv is None:
            return
        allocated = int(getattr(req, "kv_allocated_len", 0) or 0)
        if allocated <= int(keep_len):
            return
        if not kv.rollback(req, int(keep_len), allocated):
            raise RuntimeError(
                f"OPT draft {req.rid} could not roll KV back to {keep_len} "
                f"from {allocated}"
            )

    def _opt_release_uncommitted(self, req) -> int:
        """Free slots past ``kv_committed_len`` before extend reuses that span."""
        return release_uncommitted(req, self._opt_rollback)

    def _opt_draft_lease_active(self) -> bool:
        """True only on an OPT Draft. Missing fields stay inactive."""
        try:
            algo = getattr(self, "spec_algorithm", None)
            is_opt = getattr(algo, "is_opt_tree_remote", None)
            if not callable(is_opt) or not is_opt():
                return False
            args = getattr(self, "server_args", None)
            return getattr(args, "opt_tree_role", None) == "draft"
        except Exception:
            return False

    def _sr_finish_rid(self, rid, release_mm: bool = True, state=None, entry=None):
        """FINISH and ABORT free a page-1 tree lease before the prefix KV.

        TTL cleanup and ``_sr_wipe_all`` both enter here. A poisoned device
        leaves the lease for ``_sr_abandon_poisoned_kv``. Other algorithms
        keep the standalone-remote path unchanged.
        """
        if not self._opt_draft_lease_active():
            return super()._sr_finish_rid(
                rid, release_mm=release_mm, state=state, entry=entry
            )
        from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
            opt_finish_req,
            opt_kv_release_allowed,
            release_opt_bonus_guesses,
            release_opt_tree_lease,
        )

        if opt_kv_release_allowed(self):
            req = opt_finish_req(self, rid, state)
            if req is not None:
                release_opt_tree_lease(self, req)
                release_opt_bonus_guesses(self, req)
        drop_lookahead_request(self, rid)
        return super()._sr_finish_rid(
            rid, release_mm=release_mm, state=state, entry=entry
        )

    def _sr_rollback_else_reprefill(self, req, fork, allocated, on_refused) -> bool:
        """A prefix rollback makes the previous OPT tree stale."""
        if not self._opt_draft_lease_active():
            return super()._sr_rollback_else_reprefill(
                req, fork, allocated, on_refused
            )
        from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
            opt_kv_release_allowed,
            release_opt_bonus_guesses,
            release_opt_tree_lease,
        )

        if opt_kv_release_allowed(self):
            release_opt_tree_lease(self, req)
            release_opt_bonus_guesses(self, req)
        return super()._sr_rollback_else_reprefill(req, fork, allocated, on_refused)

    def _sr_reset_linear_kv_state(self, req, fill_ids) -> None:
        """Reprefill and retract drop OPT tree slots the length no longer covers."""
        if not self._opt_draft_lease_active():
            return super()._sr_reset_linear_kv_state(req, fill_ids)
        from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
            opt_kv_release_allowed,
            release_opt_bonus_guesses,
            release_opt_tree_lease,
        )

        if opt_kv_release_allowed(self):
            release_opt_tree_lease(self, req)
            release_opt_bonus_guesses(self, req)
        return super()._sr_reset_linear_kv_state(req, fill_ids)

    def _sr_abandon_poisoned_kv(self, req) -> None:
        """Keep an OPT Draft lease. Do not return its slots to the allocator."""
        if not self._opt_draft_lease_active():
            return super()._sr_abandon_poisoned_kv(req)
        from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
            detach_opt_tree_lease,
        )
        from sglang.srt.speculative.standalone_remote.drafter.sr_resource_quarantine import (
            quarantine,
        )

        lease = detach_opt_tree_lease(req)
        try:
            return super()._sr_abandon_poisoned_kv(req)
        finally:
            if lease is not None:
                quarantine(self).retain((req, lease))

    def _opt_expand_one(self, req) -> Tuple[List[int], Optional[List[int]], Optional[List[int]]]:
        import time

        from sglang.srt.speculative.opt_tree_remote.opt_tree_forward import (
            OptPhaseClock,
            commit_deferred_expands,
            drain_opt_layer_gpu_us,
            finish_opt_expand_req,
            flush_opt_ingest_phases,
        )
        from sglang.srt.speculative.opt_tree_remote.opt_tree_lease import (
            consume_opt_tree_lease,
        )

        metrics = get_sr_round_metrics(self, "Draft")
        resume_start = time.perf_counter()
        self._sr_resume_req(req)
        self._sr_isolate_need({req.rid})
        metrics.add_host("opt_resume", time.perf_counter() - resume_start)
        committed = len(req.origin_input_ids or []) + len(req.output_ids or [])
        runner = None
        expanded = False
        window = None
        tree_states = []
        try:
            consume_opt_tree_lease(self, req)
            self._opt_release_uncommitted(req)
            config = OptTreeConfig(
                width=int(self.server_args.opt_tree_width),
                tau=float(self.server_args.opt_tree_tau),
                max_depth=int(self.server_args.opt_tree_max_depth),
            )
            runner = OptLayerForward(self, req, committed)
            self._opt_lookahead_active = False
            self._opt_defer_layer_drain = False
            lookahead_runtime = None
            if lookahead_supported(self):
                t_prep = time.perf_counter()
                try:
                    lookahead_runtime = prepare_lookahead_runtime(self, runner)
                except Exception as exc:
                    _reraise_device_context(self, exc)
                    logger.info("[OPT] lookahead setup fell back: %s", exc)
                    metrics.counts["opt_lookahead_fallbacks"] += 1
                    lookahead_runtime = None
                finally:
                    metrics.add_host(
                        "opt_lookahead_prepare", time.perf_counter() - t_prep
                    )
            if lookahead_runtime is not None:
                t_admit = time.perf_counter()
                try:
                    _admit_lookahead(self, runner, lookahead_runtime[0])
                    self._opt_lookahead_active = True
                except LookaheadFallback as exc:
                    logger.info("[OPT] lookahead admit fell back: %s", exc)
                    metrics.counts["opt_lookahead_fallbacks"] += 1
                    lookahead_runtime = None
                except Exception as exc:
                    # Admission has not allocated KV. An ordinary error, including
                    # OOM, can still grow this step one layer at a time.
                    _reraise_device_context(self, exc)
                    logger.info("[OPT] lookahead admit fell back: %s", exc)
                    metrics.counts["opt_lookahead_fallbacks"] += 1
                    lookahead_runtime = None
                finally:
                    if metrics.active:
                        metrics.counts["opt_lookahead_admit_us"] += int(
                            (time.perf_counter() - t_admit) * 1e6
                        )

            def _sync():
                # Metrics sync here waits out the ingest forward and leaves no
                # room to overlap the seed. Lookahead times layers with events.
                if metrics.active and not self._opt_lookahead_active:
                    torch.get_device_module(self.device).synchronize()

            phases = OptPhaseClock()
            t0 = time.perf_counter()
            tail_logits = ingest_tail_logits(self, [runner], phases=phases)[0]
            _sync()
            phases.lap("opt_ingest_forward")
            flush_opt_ingest_phases(
                metrics, phases.parts, time.perf_counter() - t0
            )
            written = int(getattr(req, "kv_committed_len", 0) or 0)
            if written != committed:
                raise RuntimeError(
                    f"OPT draft {req.rid} committed KV {written} != {committed} tokens"
                )
            layers = 0

            def timed_forward(step):
                nonlocal layers
                layers += 1
                return runner.forward_layer(step)

            t1 = time.perf_counter()
            topk_out = [[]]
            windows = None
            seed_logits = tail_logits
            lookahead_launched = None
            if lookahead_runtime is not None:
                try:
                    windows, state, launched = grow_lookahead(
                        self, runner, config, tail_logits, lookahead_runtime
                    )
                    tree_states.append(state)
                    layers += launched
                    lookahead_launched = int(launched)
                    self._opt_defer_layer_drain = True
                except LookaheadFallback:
                    logger.info("[OPT] lookahead fell back for %s", req.rid)
                    windows = None
                    tree_states.clear()
                    saved = getattr(runner, "_lookahead_tail", None)
                    if saved is not None:
                        seed_logits = saved
            if windows is None:
                self._opt_lookahead_active = False
                self._opt_defer_layer_drain = False
                windows = grow_active_layers(
                    config,
                    [seed_logits],
                    lambda steps, _idx: [timed_forward(steps[0])],
                    self,
                    states_out=tree_states,
                    topk_out=topk_out,
                )
                runner.bonus_ids = topk_out[0]
            if not windows or windows[0] is None:
                raise RuntimeError("OPT layer forward failed")
            tokens, parents, cols = windows[0]
            metrics.add_host("opt_grow", time.perf_counter() - t1)
            metrics.counts["opt_layer_forwards"] += layers
            if lookahead_launched is not None:
                metrics.counts["opt_layer_forwards_critical"] += critical_layer_forwards(
                    lookahead_launched
                )
            metrics.counts["opt_rounds"] += 1
            window = (tokens, parents, cols)
            expanded = True
            return tokens, parents, cols
        except Exception as exc:
            _reraise_device_context(self, exc)
            logger.warning("[OPT] tree expand failed for %s: %s", req.rid, exc)
            get_sr_round_metrics(self, "Draft").counts["opt_expand_failures"] += 1
            return [], None, None
        finally:
            import sys

            if not getattr(self, "_opt_defer_layer_drain", False):
                drain_opt_layer_gpu_us(self, metrics)
            finish_start = time.perf_counter()
            try:
                pending = sys.exc_info()[1]
                clear_prefill_tail_logits(req)
                rollback_error = None
                deferred = []
                try:
                    job = finish_opt_expand_req(
                        self,
                        req,
                        runner,
                        committed,
                        window,
                        expanded,
                        tree_states[0] if tree_states else None,
                    )
                    if job is not None:
                        deferred.append(job)
                except Exception as rollback_exc:
                    logger.warning(
                        "[OPT] rollback after expand failed for %s: %s",
                        req.rid,
                        rollback_exc,
                    )
                    rollback_error = rollback_exc
                cancel_error = commit_deferred_expands(
                    self,
                    deferred,
                    keep=pending is None and rollback_error is None,
                )
                if cancel_error is not None and rollback_error is None:
                    rollback_error = cancel_error
                self.last_batch = None
                self.cur_batch = None
                if rollback_error is not None and pending is None:
                    raise rollback_error
            finally:
                metrics.add_host("opt_finish", time.perf_counter() - finish_start)

    def _opt_handle_batch_impl(self, batch: SRBatchRequest, mm_by_rid: Dict[str, SRMMPayload]):
        metrics = get_sr_round_metrics(self, "Draft")
        prepare_start = time.perf_counter()
        last_session = self.sr_state.session_id
        if last_session is not None and batch.session_id > last_session:
            self._sr_wipe_all()
        self.sr_state.session_id = batch.session_id
        self._sr_cleanup_stale_drafts(keep_rids={d.rid for d in batch.reqs})
        if (
            batch.action not in (SRAction.FINISH, SRAction.ABORT)
            and self._sr_draft_busy()
        ):
            logger.info(
                "[OPT] Draft busy, REJECT rpc_seq=%s n=%s",
                batch.rpc_seq,
                len(batch.reqs),
            )
            return SRBatchReply(
                session_id=batch.session_id,
                rpc_seq=batch.rpc_seq,
                protocol_version=SR_PROTOCOL_VERSION,
                reqs=[
                    self._sr_empty_reply(d, status=SRReplyStatus.REJECT)
                    for d in batch.reqs
                ],
            )
        n = len(batch.reqs)
        replies: List[Optional[object]] = [None] * n
        gpu_pairs = []
        vision_jobs = [] if batch.action == SRAction.PREFILL else None
        for i, dreq in enumerate(batch.reqs):
            reply, req = self._sr_prepare_v2(
                dreq,
                batch.action,
                batch.session_id,
                mm_by_rid.get(dreq.rid),
                vision_jobs,
            )
            if reply is not None:
                replies[i] = reply
            else:
                gpu_pairs.append((i, dreq, req))
        if metrics.active:
            metrics.add_host("align_prepare", time.perf_counter() - prepare_start)
        if vision_jobs:
            self._sr_flush_vision_jobs(vision_jobs)
        if gpu_pairs:
            windows = produce_opt_windows(
                self, batch.action, [(dreq, req) for _, dreq, req in gpu_pairs]
            )
            for (i, dreq, req), window in zip(gpu_pairs, windows):
                pending_wire = getattr(req, "sr_pending_wire_request", None)
                wire = pending_wire if isinstance(pending_wire, SRDraftRequest) else dreq
                reply = self._sr_stamp_window(wire, window, batch.session_id)
                if batch.action == SRAction.PREFILL:
                    reply = self._sr_annotate_prompt_compaction(req, reply)
                replies[i] = self._sr_finalize_v2_commit(wire, req, reply)
        self._sr_note_handled(batch)
        return SRBatchReply(
            session_id=batch.session_id,
            rpc_seq=batch.rpc_seq,
            protocol_version=SR_PROTOCOL_VERSION,
            reqs=[
                r if r is not None else self._sr_empty_reply(d)
                for r, d in zip(replies, batch.reqs)
            ],
        )

    @DynamicGradMode()
    def event_loop_normal_opt_tree_target(self):
        self._init_opt_tree_target()
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                is_extend = (
                    batch.forward_mode is not None and batch.forward_mode.is_extend()
                ) or getattr(batch, "is_extend_in_batch", False)

                if is_extend:
                    live = [
                        r for r in self._sr_select_reqs(batch) if not r.finished()
                    ]
                    inflight = (
                        self._sr_send(
                            SRAction.PREFILL, live, include_full_context=True
                        )
                        if live
                        else None
                    )
                    batch.draft_num_tokens = self.server_args.speculative_num_draft_tokens
                    try:
                        result = self.run_batch(batch)
                        self.process_batch_result(batch, result)
                    finally:
                        if inflight is not None:
                            replies, pending = self._sr_recv(inflight)
                            still = [r for r in live if not r.finished()]
                            self._sr_apply_commit_acks(still, replies, pending)
                            self._sr_note_draft_prompt_lens(still, replies)
                            self._sr_maybe_align_chain_replies(
                                still, replies, prefill=True
                            )
                            self._sr_attach_replies(still, replies)
                elif skip_reason := self._sr_skip_draft_rpc(batch):
                    get_sr_round_metrics(self, "Target").note_skip(
                        skip_reason, batch.batch_size()
                    )
                    batch.draft_num_tokens = 1
                    result = self.run_batch(batch)
                    self.process_batch_result(batch, result)
                else:
                    self._opt_run_verify_round(batch)
            else:
                self.self_check_during_idle()
                idle_sr_round_metrics(self)

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    @DynamicGradMode()
    def event_loop_normal_opt_tree_draft(self) -> None:
        self._init_opt_tree_draft()
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            packet = self._sr_recv_packet()
            if packet is not None:
                batch, mm = packet
                if self._sr_batch_is_stale(batch):
                    if self.tp_rank == 0:
                        logger.info(
                            "[OPT] Draft drop stale session=%s rpc_seq=%s",
                            batch.session_id,
                            batch.rpc_seq,
                        )
                else:
                    metrics = get_sr_round_metrics(self, "Draft")
                    measure = batch.action == SRAction.STEP
                    last_session = self.sr_state.session_id
                    if last_session is not None and batch.session_id > last_session:
                        from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
                            flush_sr_round_metrics,
                        )

                        flush_sr_round_metrics(self, "session_reset")
                    timing = False
                    try:
                        def _work(counts):
                            nonlocal timing
                            # Sync stays outside round(), after active is cleared.
                            timing = bool(getattr(metrics, "active", False))
                            try:
                                round_start = time.perf_counter()
                                if not supported_protocol(batch.protocol_version):
                                    reply = self._sr_protocol_mismatch_reply(batch)
                                    self._sr_note_handled(batch)
                                elif self._sr_spec_shape_rejected(batch):
                                    reply = self._sr_spec_shape_mismatch_reply(batch)
                                    self._sr_note_handled(batch)
                                else:
                                    reply = self._opt_handle_batch_impl(batch, mm)
                                if (
                                    (self.tp_size == 1 or self.tp_rank == 0)
                                    and self.sr_server is not None
                                    and batch.action not in (SRAction.FINISH, SRAction.ABORT)
                                ):
                                    with metrics.phase("reply_send"):
                                        self.sr_server.send_batch(reply)
                                if measure and getattr(metrics, "active", False):
                                    # Reply path is done, whether or not this rank sent.
                                    # A failed send skips this stamp.
                                    counts["opt_pre_reply_us"] += int(
                                        (time.perf_counter() - round_start) * 1e6
                                    )
                                    host = getattr(metrics, "_round_host", None)
                                    ingest = 0.0 if host is None else host["opt_ingest"]
                                    grow = 0.0 if host is None else host["opt_grow"]
                                    metrics.add_host(
                                        "opt_other",
                                        opt_other_seconds(
                                            time.perf_counter() - round_start,
                                            ingest,
                                            grow,
                                        ),
                                    )
                            finally:
                                # Still inside round(), so a full-window clear
                                # cannot move these adds into the next window.
                                jobs = self._opt_deferred
                                self._opt_deferred = []
                                defer_drain = bool(
                                    getattr(self, "_opt_defer_layer_drain", False)
                                )
                                self._opt_defer_layer_drain = False
                                if jobs:
                                    started = time.perf_counter()
                                    try:
                                        forward_deferred_final_layers(self, jobs)
                                    finally:
                                        counts["opt_post_reply_us"] += int(
                                            (time.perf_counter() - started) * 1e6
                                        )
                                drain_deferred_layer_events(
                                    self,
                                    metrics,
                                    defer_drain=defer_drain,
                                    jobs=jobs,
                                )

                        run_opt_draft_round(metrics, measure, _work)
                    finally:
                        if timing:
                            torch.get_device_module(self.device).synchronize()
                self.last_batch = None
                self._sr_resume_http_reqs()
                self._sr_poll_deferred_finishes()
                continue
            self._sr_cleanup_stale_drafts()
            self._sr_poll_deferred_finishes()
            self._sr_run_http_batch()
            idle_sr_round_metrics(self)
