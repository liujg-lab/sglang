import logging
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.speculative.standalone_remote.sr_align import (
    shift_overlapped_prefill_drafts,
    drop_duplicate_root_draft,
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


def _clear_draft(req: Req) -> None:
    req.cur_drafts = []
    req.draft_tokens_and_logits = {
        "draft_tokens": torch.tensor([0], dtype=torch.int64),
        "parent_list": None,
        "top_scores_index": None,
    }


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

    def _init_sr_target(self) -> None:
        self.sr_session_id = self._new_sr_session_id()
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
        logger.info(
            "[SR] Target session_id=%s",
            self.sr_session_id,
        )

    def reset_standalone_remote_target_state(self) -> None:
        self.sr_session_id = self._new_sr_session_id()
        self.sr_rpc_seq = 0
        self.sr_pending = {}
        self._sr_inflight = None
        client = getattr(self, "sr_client", None)
        if client is not None:
            client._drain()
        breaker = getattr(self, "sr_breaker", None)
        if breaker is not None:
            breaker.reset()
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

    def _sr_base_committed_len(self, req: Req) -> int:
        return len(req.origin_input_ids) + len(req.output_ids or [])

    def _sr_step_id(self, req: Req) -> int:
        return int(getattr(req, "sr_step_id", 0) or 0)

    def _build_sr_request(
        self, req: Req, action: SRAction, include_full_context: bool
    ) -> Tuple[SRDraftRequest, Optional[SRMMPayload]]:
        mm_payload = None
        has_mm = False
        if (
            include_full_context
            and req.multimodal_inputs is not None
            and _mm_features_alive(req)
        ):
            has_mm = True
            mm_payload = SRMMPayload.from_req(req)
        return (
            SRDraftRequest(
                rid=req.rid,
                step_id=self._sr_step_id(req),
                base_committed_len=self._sr_base_committed_len(req),
                committed_ids=self._sr_committed_ids(req),
                num_draft_tokens=(
                    int(self.server_args.speculative_num_draft_tokens or 0)
                    or int(self.server_args.speculative_num_steps or 0) + 1
                )
                if action != SRAction.FINISH
                else 0,
                padded_input_ids=(
                    list(req.origin_input_ids) if include_full_context else None
                ),
                sampling_params=req.sampling_params if include_full_context else None,
                has_mm=has_mm,
            ),
            mm_payload,
        )

    def _sr_send(
        self,
        action: SRAction,
        reqs: List[Req],
        include_full_context: bool,
    ) -> Optional[SRInflight]:
        if not reqs:
            return None

        pending: Dict[str, SRPendingEntry] = {}
        draft_reqs: List[SRDraftRequest] = []
        mm_by_rid: Dict[str, SRMMPayload] = {}
        for req in reqs:
            if _is_health_check(req):
                continue
            dreq, mm = self._build_sr_request(req, action, include_full_context)
            draft_reqs.append(dreq)
            pending[req.rid] = SRPendingEntry(
                step_id=dreq.step_id, base_committed_len=dreq.base_committed_len
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
            from sglang.srt.utils import broadcast_pyobj

            send_ok = broadcast_pyobj(
                send_ok,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )

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
            from sglang.srt.utils import broadcast_pyobj

            payload = reply.to_dict() if reply is not None else None
            payload = broadcast_pyobj(
                payload,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
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

    def rpc_next_draft(self, reqs: List[Req]) -> Dict[str, SRDraftReply]:
        breaker = getattr(self, "sr_breaker", None)
        if breaker is not None and not breaker.should_send():
            return {}
        return self._sr_rpc(SRAction.STEP, reqs, include_full_context=False)

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
                            self._sr_maybe_align_chain_replies(
                                still, replies, prefill=True
                            )
                            self._sr_attach_replies(still, replies)
                elif self._sr_skip_draft_rpc(batch):
                    batch.draft_num_tokens = 1
                    result = self.run_batch(batch)
                    self.process_batch_result(batch, result)
                else:
                    live = self._sr_select_reqs(batch)
                    n_ok = sum(1 for r in live if getattr(r, "cur_drafts", None))
                    if n_ok == 0:
                        replies = self.rpc_next_draft(live)
                        self._sr_maybe_align_chain_replies(
                            live, replies, prefill=False
                        )
                        n_ok = self._sr_attach_replies(live, replies)
                    if n_ok == 0:
                        batch.draft_num_tokens = 1
                    else:
                        batch.draft_num_tokens = (
                            self.server_args.speculative_num_draft_tokens
                        )
                    result = self.run_batch(batch)
                    self.process_batch_result(batch, result)
                    still = [r for r in live if not r.finished()]
                    if still:
                        replies = self.rpc_next_draft(still)
                        self._sr_maybe_align_chain_replies(
                            still, replies, prefill=False
                        )
                        self._sr_attach_replies(still, replies)
            else:
                self.self_check_during_idle()

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()
