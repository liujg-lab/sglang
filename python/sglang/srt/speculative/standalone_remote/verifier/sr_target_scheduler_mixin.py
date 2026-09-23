import logging
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sglang.srt.environ import envs
from sglang.srt.speculative.standalone_remote.sr_round_metrics import get_sr_round_metrics
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.speculative.standalone_remote.sr_align import (
    broadcast_sr_obj,
    shift_overlapped_prefill_drafts,
    drop_duplicate_root_draft,
)
from sglang.srt.speculative.standalone_remote.sr_commit import (
    SR_PROTOCOL_VERSION,
    ack_matches,
    note_commit_result,
    note_commit_send,
    reply_stops_speculation,
    status_advances_cursor,
)
from sglang.srt.speculative.standalone_remote.sr_circuit_breaker import SRRpcBreaker
from sglang.srt.speculative.standalone_remote.sr_mm_payload import SRMMPayload
from sglang.srt.speculative.standalone_remote.sr_protocol import (
    SRAction,
    SRBatchReply,
    SRBatchRequest,
    SRDraftReply,
    SRDraftRequest,
    SRPendingEntry,
    SRReplyStatus,
    is_health_check_req as _is_health_check,
)
from sglang.srt.speculative.standalone_remote.sr_transport import (
    SRTargetClient,
    make_transport_from_server_args,
)
from sglang.srt.utils import DynamicGradMode
import torch

logger = logging.getLogger(__name__)


def _mm_features_alive(req: Req) -> bool:
    mm = getattr(req, "multimodal_inputs", None)
    if mm is None:
        return False
    for item in getattr(mm, "mm_items", None) or []:
        if getattr(item, "feature", None) is not None:
            return True
        if getattr(item, "precomputed_embeddings", None) is not None:
            return True
    return False


def _apply_reply_to_req(req: Req, reply: SRDraftReply) -> None:
    tokens = list(reply.draft_tokens or [])
    req.cur_drafts = list(tokens)
    req.draft_tokens_and_logits = {
        "draft_tokens": torch.tensor(tokens, dtype=torch.int64)
        if tokens
        else torch.tensor([0], dtype=torch.int64),
        "parent_list": reply.parent_list,
        "top_scores_index": reply.top_scores_index,
    }
    req.sr_draft_tree_version = reply.tree_version
    req.sr_draft_tree_base_committed_len = (
        int(reply.base_committed_len) if reply.tree_version is not None else None
    )


def _clear_draft(req: Req) -> None:
    req.cur_drafts = []
    req.draft_tokens_and_logits = {
        "draft_tokens": torch.tensor([0], dtype=torch.int64),
        "parent_list": None,
        "top_scores_index": None,
    }
    req.sr_draft_tree_version = None
    req.sr_draft_tree_base_committed_len = None
    req.sr_accepted_tree_candidate_indices = None


@dataclass
class SRInflight:
    session_id: str
    rpc_seq: int
    pending: Dict[str, SRPendingEntry]
    wait_reply: bool
    send_ok: bool = True


class SchedulerStandaloneRemoteTargetMixin:
    @property
    def is_remote_spec_draft(self) -> bool:
        """Draft process that only produces tokens for a remote verifier."""
        return (
            self.spec_algorithm.is_spectre()
            and self.server_args.spectre_role == "draft"
        ) or (
            self.spec_algorithm.is_standalone_remote()
            and self.server_args.standalone_remote_role == "draft"
        )

    @property
    def is_remote_spec_target(self) -> bool:
        """Target process that verifies tokens from a remote drafter."""
        return (
            self.spec_algorithm.is_spectre()
            and self.server_args.spectre_role == "target"
        ) or (
            self.spec_algorithm.is_standalone_remote()
            and self.server_args.standalone_remote_role == "target"
        )

    def maybe_notify_remote_draft_finished(self, req) -> None:
        if not self.is_remote_spec_target:
            return
        if self.spec_algorithm.is_spectre():
            from sglang.srt.speculative.spectre.spectre_protocol import SpectreAction

            self.notify_draft_request_finished_or_aborted(req, SpectreAction.FINISH)
        elif self.spec_algorithm.is_standalone_remote():
            self.notify_sr_draft_finished(req, SRAction.FINISH)

    def _new_sr_session_id(self) -> str:
        return f"{time.time_ns():020d}_{uuid.uuid4().hex[:8]}"

    def _sr_broadcast_obj(self, obj):
        return broadcast_sr_obj(
            obj, self.tp_size, self.tp_rank, self.tp_group, self.tp_cpu_group
        )

    def _sync_sr_session_id(self) -> None:
        if self.tp_rank == 0:
            self.sr_session_id = self._new_sr_session_id()
        else:
            self.sr_session_id = None
        if self.tp_size > 1:
            self.sr_session_id = self._sr_broadcast_obj(self.sr_session_id)

    def _init_sr_target(self) -> None:
        self._sync_sr_session_id()
        self.sr_rpc_seq = 0
        self.sr_pending: Dict[str, SRPendingEntry] = {}
        self._sr_inflight: Optional[SRInflight] = None
        self.sr_client: Optional[SRTargetClient] = None
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
            client = make_transport_from_server_args(self.server_args)
            assert isinstance(client, SRTargetClient)
            client.drop_callback = self._sr_note_stale
            self.sr_client = client
        if self.tp_rank == 0:
            logger.info(
                "[SR] Target session_id=%s",
                self.sr_session_id,
            )

    def reset_standalone_remote_target_state(self) -> None:
        self._sync_sr_session_id()
        self.sr_rpc_seq = 0
        self.sr_pending = {}
        self._sr_inflight = None
        self.sr_speculation_stopped = False
        self.sr_seen_generation = None
        client = getattr(self, "sr_client", None)
        if client is not None:
            client._drain()
        breaker = getattr(self, "sr_breaker", None)
        if breaker is not None:
            breaker.reset()
        if self.tp_rank == 0:
            logger.info("[SR] Target flushed, new session_id=%s", self.sr_session_id)

    def _sr_note_stale(self, reason: str) -> None:
        from sglang.srt.speculative.standalone_remote.sr_transport import (
            note_stale_drop,
        )

        note_stale_drop(reason)
        metrics = self._sr_metrics()
        if metrics is not None:
            metrics.increment_sr_stale_replies_dropped(reason=reason)

    def _sr_metrics(self):
        if not getattr(self, "current_scheduler_metrics_enabled", False):
            return None
        return getattr(self, "metrics_collector", None)

    def _sr_committed_ids(self, req: Req) -> List[int]:
        return list(req.output_ids or [])

    def _sr_ensure_cursor(self, req: Req) -> None:
        if not isinstance(getattr(req, "sr_next_commit_version", None), int):
            req.sr_next_commit_version = 1
            req.sr_ack_commit_version = None
            req.sr_ack_output_len = 0
            req.sr_force_snapshot = True
            req.sr_stop_spec = False

    def _sr_active_reqs(self) -> List[Req]:
        found = []
        for batch in (
            getattr(self, "running_batch", None),
            getattr(self, "cur_batch", None),
            getattr(self, "last_batch", None),
        ):
            if batch is None:
                continue
            for req in getattr(batch, "reqs", None) or []:
                if req is not None and not _is_health_check(req):
                    found.append(req)
        return found

    def _sr_sync_connection(self) -> None:
        generation = None
        if self.tp_size == 1 or self.tp_rank == 0:
            client = getattr(self, "sr_client", None)
            generation = getattr(client, "connection_generation", None)
        if self.tp_size > 1:
            generation = self._sr_broadcast_obj(generation)
        if generation is None:
            return
        seen = getattr(self, "sr_seen_generation", None)
        self.sr_seen_generation = generation
        if seen is None or seen == generation:
            return
        for req in self._sr_active_reqs():
            self._sr_ensure_cursor(req)
            req.sr_force_snapshot = True

    def _sr_stop_speculation(self, reason: str) -> None:
        if getattr(self, "sr_speculation_stopped", False):
            return
        self.sr_speculation_stopped = True
        logger.warning("[SR] stop speculation for session: %s", reason)
        for req in self._sr_active_reqs():
            self._sr_ensure_cursor(req)
            req.sr_stop_spec = True
            _clear_draft(req)

    def _sr_candidate_path(self, req: Req, delta_ids: Optional[List[int]]):
        if not delta_ids:
            return None, None, None
        base = getattr(req, "sr_draft_tree_base_committed_len", None)
        version = getattr(req, "sr_draft_tree_version", None)
        path = list(getattr(req, "sr_accepted_tree_candidate_indices", None) or [])
        prompt_len = len(req.origin_input_ids or [])
        acked = int(getattr(req, "sr_ack_output_len", 0) or 0)
        delta_start = prompt_len + acked
        if (
            version is None
            or base != delta_start
            or not path
            or len(path) > len(delta_ids)
        ):
            return None, None, None
        return version, base, path

    def _sr_base_committed_len(self, req: Req) -> int:
        return len(req.origin_input_ids) + len(req.output_ids or [])

    def _sr_step_id(self, req: Req) -> int:
        return int(getattr(req, "sr_step_id", 0) or 0)

    def _build_sr_request(
        self, req: Req, action: SRAction, include_full_context: bool
    ) -> Tuple[SRDraftRequest, Optional[SRMMPayload]]:
        self._sr_ensure_cursor(req)
        metrics = get_sr_round_metrics(self, "Target")
        num_draft = (
            int(self.server_args.speculative_num_draft_tokens or 0)
            or int(self.server_args.speculative_num_steps or 0) + 1
        )
        if action in (SRAction.FINISH, SRAction.ABORT):
            return (
                SRDraftRequest(
                    rid=req.rid,
                    step_id=self._sr_step_id(req),
                    base_committed_len=self._sr_base_committed_len(req),
                    num_draft_tokens=0,
                ),
                None,
            )
        snapshot = (
            include_full_context
            or bool(req.sr_force_snapshot)
            or req.sr_ack_commit_version is None
        )
        version = int(req.sr_next_commit_version)
        req.sr_next_commit_version = version + 1
        sent_len = len(req.output_ids or [])
        req.sr_sent_commit_version = version
        req.sr_sent_output_len = sent_len
        mm_items = getattr(getattr(req, "multimodal_inputs", None), "mm_items", None)
        requires_mm = isinstance(mm_items, list) and len(mm_items) > 0
        if snapshot:
            committed = list(req.output_ids or [])
            mm_payload = None
            has_mm = False
            if requires_mm and _mm_features_alive(req):
                has_mm = True
                mm_payload = SRMMPayload.from_req(req)
            note_commit_send(metrics, "snapshot", len(committed))
            return (
                SRDraftRequest(
                    rid=req.rid,
                    step_id=self._sr_step_id(req),
                    base_committed_len=self._sr_base_committed_len(req),
                    committed_ids=committed,
                    num_draft_tokens=num_draft,
                    padded_input_ids=list(req.origin_input_ids),
                    sampling_params=req.sampling_params,
                    has_mm=has_mm,
                    commit_mode="snapshot",
                    commit_version=version,
                    requires_mm=requires_mm,
                ),
                mm_payload,
            )
        acked = int(req.sr_ack_output_len or 0)
        delta_ids = list((req.output_ids or [])[acked:sent_len])
        tree_version, tree_base, path = self._sr_candidate_path(req, delta_ids)
        note_commit_send(metrics, "delta", len(delta_ids))
        return (
            SRDraftRequest(
                rid=req.rid,
                step_id=self._sr_step_id(req),
                base_committed_len=self._sr_base_committed_len(req),
                num_draft_tokens=num_draft,
                commit_mode="delta",
                commit_version=version,
                base_commit_version=req.sr_ack_commit_version,
                base_output_len=acked,
                delta_ids=delta_ids,
                requires_mm=requires_mm,
                commit_tree_version=tree_version,
                commit_tree_base_committed_len=tree_base,
                commit_candidate_indices=path,
            ),
            None,
        )

    def _sr_send(
        self,
        action: SRAction,
        reqs: List[Req],
        include_full_context: bool,
    ) -> Optional[SRInflight]:
        if not reqs:
            return None
        if getattr(self, "sr_speculation_stopped", False) and action == SRAction.STEP:
            return None
        self._sr_sync_connection()

        pending: Dict[str, SRPendingEntry] = {}
        draft_reqs: List[SRDraftRequest] = []
        mm_by_rid: Dict[str, SRMMPayload] = {}
        for req in reqs:
            if _is_health_check(req):
                continue
            dreq, mm = self._build_sr_request(req, action, include_full_context)
            draft_reqs.append(dreq)
            pending[req.rid] = SRPendingEntry(
                step_id=dreq.step_id,
                base_committed_len=dreq.base_committed_len,
                commit_version=dreq.commit_version,
                sent_output_len=getattr(req, "sr_sent_output_len", None),
                protocol_version=SR_PROTOCOL_VERSION,
            )
            if mm is not None:
                mm_by_rid[req.rid] = mm

        if not draft_reqs:
            return None

        self.sr_rpc_seq += 1
        rpc_seq = self.sr_rpc_seq
        self.sr_pending = pending
        wait_reply = action not in (SRAction.FINISH, SRAction.ABORT)
        batch = SRBatchRequest(
            session_id=self.sr_session_id,
            rpc_seq=rpc_seq,
            action=action,
            reqs=draft_reqs,
            protocol_version=SR_PROTOCOL_VERSION,
        )

        send_ok = True
        if self.tp_size == 1 or self.tp_rank == 0:
            if self.sr_client is None:
                logger.warning("[SR] Target has no ZMQ client")
                send_ok = False
            else:
                try:
                    self.sr_client.send_batch(batch, mm_by_rid or None)
                except Exception as e:
                    logger.warning("[SR] Target RPC send failed: %s", e)
                    send_ok = False

        if self.tp_size > 1:
            send_ok = self._sr_broadcast_obj(send_ok)

        inflight = SRInflight(
            session_id=self.sr_session_id,
            rpc_seq=rpc_seq,
            pending=pending,
            wait_reply=wait_reply,
            send_ok=send_ok,
        )
        self._sr_inflight = inflight
        return inflight

    def _sr_recv(self, inflight: Optional[SRInflight]) -> Dict[str, SRDraftReply]:
        if inflight is None:
            return {}
        self._sr_ack_pending = inflight.pending
        if getattr(self, "_sr_inflight", None) is inflight:
            self._sr_inflight = None

        reply: Optional[SRBatchReply] = None
        if inflight.wait_reply and inflight.send_ok:
            if self.tp_size == 1 or self.tp_rank == 0:
                if self.sr_client is not None:
                    try:
                        reply = self.sr_client.recv_batch(
                            inflight.session_id, inflight.rpc_seq
                        )
                    except Exception as e:
                        logger.warning("[SR] Target RPC recv failed: %s", e)
                        reply = None

        if self.tp_size > 1:
            payload = reply.to_dict() if reply is not None else None
            payload = self._sr_broadcast_obj(payload)
            reply = SRBatchReply.from_dict(payload) if payload is not None else None

        result: Dict[str, SRDraftReply] = {}
        got_packet = reply is not None
        if reply is None:
            self._sr_observe_rpc(inflight, got_packet=False)
            return result
        if reply.session_id != self.sr_session_id:
            self._sr_note_stale("session")
            self._sr_observe_rpc(inflight, got_packet=True)
            return result
        if reply.rpc_seq != inflight.rpc_seq:
            self._sr_note_stale("rpc_seq")
            self._sr_observe_rpc(inflight, got_packet=True)
            return result
        if reply_stops_speculation(
            reply.protocol_version, session_rpc_matched=True
        ) or any(
            item.reason == "protocol_mismatch" for item in reply.reqs
        ):
            self._sr_stop_speculation(
                "protocol_mismatch"
                if any(item.reason == "protocol_mismatch" for item in reply.reqs)
                else "protocol_version"
            )
            self._sr_observe_rpc(inflight, got_packet=True)
            return result
        pending = inflight.pending
        for item in reply.reqs:
            pending_entry = pending.get(item.rid)
            if pending_entry is None:
                continue
            ok, reason = item.matches(pending_entry)
            if not ok:
                self._sr_note_stale(reason or "step")
                continue
            result[item.rid] = item
        self._sr_observe_rpc(inflight, got_packet=got_packet)
        return result

    def _sr_observe_rpc(self, inflight: SRInflight, *, got_packet: bool) -> None:
        if not inflight.wait_reply:
            return
        breaker = getattr(self, "sr_breaker", None)
        if breaker is None:
            return
        if inflight.send_ok and got_packet:
            breaker.record_success()
        else:
            breaker.record_failure()

    def _sr_rpc(
        self,
        action: SRAction,
        reqs: List[Req],
        include_full_context: bool,
    ) -> Dict[str, SRDraftReply]:
        inflight = self._sr_send(action, reqs, include_full_context)
        return self._sr_recv(inflight)

    def _sr_apply_commit_acks(
        self, reqs: List[Req], replies: Dict[str, SRDraftReply]
    ) -> List[Req]:
        """Advance cursors. Returns rids that asked for one snapshot retry."""
        pending = getattr(self, "_sr_ack_pending", {}) or {}
        metrics = get_sr_round_metrics(self, "Target")
        retry: List[Req] = []
        for req in reqs:
            self._sr_ensure_cursor(req)
            item = replies.get(req.rid)
            entry = pending.get(req.rid)
            if item is None or entry is None:
                req.sr_force_snapshot = True
                continue
            if item.status == SRReplyStatus.NEED_SNAPSHOT:
                note_commit_result(metrics, "commit_recover_" + (item.reason or "snapshot"))
                retry.append(req)
                req.sr_force_snapshot = True
                continue
            if item.status == SRReplyStatus.REJECT and item.reason in (
                "unrecoverable_mm",
                "completion_unknown",
                "untrusted_kv",
            ):
                req.sr_stop_spec = True
                _clear_draft(req)
                continue
            if item.status == SRReplyStatus.REJECT:
                continue
            if status_advances_cursor(item.status):
                ok, _reason = ack_matches(item, entry)
                if ok:
                    req.sr_ack_commit_version = entry.commit_version
                    req.sr_ack_output_len = entry.sent_output_len
                    req.sr_force_snapshot = False
                    if item.status == SRReplyStatus.IDEMPOTENT:
                        note_commit_result(metrics, "commit_idempotent_hits")
                    continue
            req.sr_force_snapshot = True
        return retry

    def rpc_next_draft(self, reqs: List[Req]) -> Dict[str, SRDraftReply]:
        breaker = getattr(self, "sr_breaker", None)
        if breaker is not None and not breaker.should_send():
            return {}
        if getattr(self, "sr_speculation_stopped", False):
            return {}
        live = [req for req in reqs if not getattr(req, "sr_stop_spec", False)]
        if not live:
            return {}
        replies = self._sr_rpc(SRAction.STEP, live, include_full_context=False)
        if getattr(self, "sr_speculation_stopped", False):
            return {}
        retry = self._sr_apply_commit_acks(live, replies)
        if not retry:
            return replies
        for req in retry:
            req.sr_force_snapshot = True
        snap = self._sr_rpc(SRAction.STEP, retry, include_full_context=True)
        if getattr(self, "sr_speculation_stopped", False):
            return replies
        again = self._sr_apply_commit_acks(retry, snap)
        metrics = get_sr_round_metrics(self, "Target")
        note_commit_result(metrics, "commit_resend_ok", len(retry) - len(again))
        note_commit_result(metrics, "commit_resend_fail", len(again))
        for req in again:
            req.sr_force_snapshot = True
            _clear_draft(req)
        replies.update(snap)
        return replies

    def notify_sr_draft_finished(self, req: Req, action: SRAction = SRAction.FINISH) -> None:
        if _is_health_check(req):
            return
        self._sr_rpc(action, [req], include_full_context=False)

    def _sr_shift_prefill_replies(
        self, reqs: List[Req], replies: Dict[str, SRDraftReply]
    ) -> None:
        """Align overlapped PREFILL drafts to tokens Target already sampled."""
        for req in reqs:
            reply = replies.get(req.rid)
            if reply is None or not reply.draft_tokens:
                continue
            original = list(reply.draft_tokens)
            shifted = shift_overlapped_prefill_drafts(
                req.output_ids, reply.draft_tokens
            )
            if shifted is None:
                reply.draft_tokens = []
                reply.parent_list = None
                reply.top_scores_index = None
                continue
            skipped = len(original) - len(shifted)
            if skipped == 0:
                continue
            reply.draft_tokens = shifted
            # Chain overlap shift indexes from D0; tree topology is left intact.
            reply.parent_list = None
            reply.top_scores_index = None

    def _sr_drop_root_duplicate_replies(
        self, reqs: List[Req], replies: Dict[str, SRDraftReply]
    ) -> None:
        for req in reqs:
            reply = replies.get(req.rid)
            if reply is None or not reply.draft_tokens:
                continue
            last = (req.output_ids or [None])[-1] if req.output_ids else None
            original = list(reply.draft_tokens)
            dropped = drop_duplicate_root_draft(last, original)
            if dropped == original:
                continue
            reply.draft_tokens = dropped
            reply.parent_list = None
            reply.top_scores_index = None

    def _sr_attach_replies(
        self, reqs: List[Req], replies: Dict[str, SRDraftReply]
    ) -> int:
        n_ok = 0
        for req in reqs:
            if _is_health_check(req):
                continue
            reply = replies.get(req.rid)
            if (
                reply is not None
                and reply.status in (SRReplyStatus.OK, SRReplyStatus.IDEMPOTENT)
                and reply.draft_tokens
            ):
                _apply_reply_to_req(req, reply)
                n_ok += 1
            else:
                _clear_draft(req)
        return n_ok

    def _sr_select_reqs(self, batch: ScheduleBatch) -> List[Req]:
        return [req for req in batch.reqs if not _is_health_check(req)]

    def _sr_is_high_overhead(self, batch: ScheduleBatch) -> bool:
        current_bsz = max(batch.batch_size(), self.running_batch.batch_size())
        return current_bsz > self.server_args.standalone_remote_max_batch_size

    def _sr_skip_draft_rpc(self, batch: ScheduleBatch) -> bool:
        if self._sr_is_high_overhead(batch):
            return True
        breaker = getattr(self, "sr_breaker", None)
        if breaker is not None and not breaker.should_send():
            breaker.note_skipped_step()
            return True
        return False

    def _sr_chain_mode(self) -> bool:
        return int(getattr(self.server_args, "speculative_eagle_topk", 1) or 1) <= 1

    def _sr_maybe_align_chain_replies(
        self, reqs: List[Req], replies: Dict[str, SRDraftReply], *, prefill: bool
    ) -> None:
        if not self._sr_chain_mode():
            return
        if prefill:
            self._sr_shift_prefill_replies(reqs, replies)
        self._sr_drop_root_duplicate_replies(reqs, replies)

    @DynamicGradMode()
    def event_loop_normal_standalone_remote_target(self):
        self._init_sr_target()
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
                        r
                        for r in self._sr_select_reqs(batch)
                        if not r.finished()
                    ]
                    inflight = (
                        self._sr_send(
                            SRAction.PREFILL, live, include_full_context=True
                        )
                        if live
                        else None
                    )
                    batch.draft_num_tokens = (
                        self.server_args.speculative_num_draft_tokens
                    )
                    try:
                        result = self.run_batch(batch)
                        self.process_batch_result(batch, result)
                    finally:
                        if inflight is not None:
                            replies = self._sr_recv(inflight)
                            still = [r for r in live if not r.finished()]
                            self._sr_apply_commit_acks(still, replies)
                            self._sr_maybe_align_chain_replies(
                                still, replies, prefill=True
                            )
                            self._sr_attach_replies(still, replies)
                elif self._sr_skip_draft_rpc(batch):
                    batch.draft_num_tokens = 1
                    result = self.run_batch(batch)
                    self.process_batch_result(batch, result)
                else:
                    self._sr_run_verify_round(batch)
            else:
                self.self_check_during_idle()

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    def _sr_run_verify_round(self, batch):
        metrics = get_sr_round_metrics(self, "Target")
        with metrics.round():
            batch.sr_round_metrics = metrics
            try:
                live = self._sr_select_reqs(batch)
                n_ok = sum(1 for r in live if getattr(r, "cur_drafts", None))
                if n_ok == 0:
                    with metrics.phase("rpc_wait"):
                        replies = self.rpc_next_draft(live)
                    self._sr_maybe_align_chain_replies(live, replies, prefill=False)
                    n_ok = self._sr_attach_replies(live, replies)
                batch.draft_num_tokens = (
                    self.server_args.speculative_num_draft_tokens if n_ok else 1
                )
                with metrics.phase("run_batch_including_verify"):
                    result = self.run_batch(batch)
                with metrics.phase("process_result"):
                    self.process_batch_result(batch, result)
                still = [r for r in live if not r.finished()]
                if still:
                    with metrics.phase("rpc_wait"):
                        replies = self.rpc_next_draft(still)
                    self._sr_maybe_align_chain_replies(still, replies, prefill=False)
                    self._sr_attach_replies(still, replies)
            finally:
                batch.sr_round_metrics = None
