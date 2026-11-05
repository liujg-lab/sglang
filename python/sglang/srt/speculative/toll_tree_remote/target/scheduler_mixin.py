"""Target event loop. Rank 0 owns the socket. Other ranks receive the reply."""

from __future__ import annotations

import time
import uuid

from sglang.srt.speculative.toll_tree_remote.protocol import (
    PROTO_VERSION,
    TollAction,
    TollReply,
    TollReqItem,
    TollRequest,
    TollStatus,
    spec_shape_of,
)
from sglang.srt.speculative.toll_tree_remote.target.breaker import TollRpcBreaker
from sglang.srt.utils import DynamicGradMode


def committed_len(req) -> int:
    return len(list(getattr(req, "origin_input_ids", []) or [])) + len(
        list(getattr(req, "output_ids", []) or [])
    )


def tail_token(req) -> int:
    output = list(getattr(req, "output_ids", []) or [])
    if output:
        return int(output[-1])
    origin = list(getattr(req, "origin_input_ids", []) or [])
    return int(origin[-1]) if origin else 0


def skip_speculative(batch) -> bool:
    """Ordinary decode for health checks and anything that is not plain greedy."""
    for req in getattr(batch, "reqs", None) or []:
        if str(getattr(req, "rid", "")).startswith("HEALTH_CHECK"):
            return True
    info = getattr(batch, "sampling_info", None)
    if info is None or not bool(getattr(info, "is_all_greedy", False)):
        return True
    if bool(getattr(batch, "has_grammar", False)):
        return True
    orchestrator = getattr(info, "penalizer_orchestrator", None)
    if orchestrator is not None and bool(getattr(orchestrator, "is_required", False)):
        return True
    if getattr(info, "logit_bias", None) is not None:
        return True
    return False


class TollTargetSchedulerMixin:
    def init_toll_tree_communication(self) -> None:
        if not self.spec_algorithm.is_toll_tree_remote():
            return
        if getattr(self.server_args, "toll_tree_role", None) != "target":
            return
        from sglang.srt.speculative.toll_tree_remote.config import TollConfig
        from sglang.srt.speculative.toll_tree_remote.transport import TollTargetClient

        self.toll_cfg = TollConfig.from_server_args(self.server_args, "target")
        self.toll_breaker = TollRpcBreaker()
        self.toll_seq = 0
        self.toll_session = uuid.uuid4().hex
        self.toll_sent_len: dict[str, int] = {}
        self.toll_need_snapshot: set[str] = set()
        self._toll_last_reply_ns = time.perf_counter_ns()
        url = f"tcp://{self.server_args.toll_tree_addr}:{self.server_args.toll_tree_port}"
        if self.tp_rank == 0:
            timeout_ms = int(getattr(self.server_args, "toll_tree_rpc_timeout_ms", 5000) or 5000)
            self.toll_client = TollTargetClient(url, self.toll_cfg, timeout_ms=timeout_ms)
        else:
            self.toll_client = None

    def _toll_broadcast(self, payload):
        if int(getattr(self, "tp_size", 1)) <= 1:
            return payload
        from sglang.srt.utils import broadcast_pyobj

        wrapped = [payload] if self.tp_rank == 0 else []
        return broadcast_pyobj(
            wrapped,
            self.tp_group.rank,
            self.tp_cpu_group,
            src=self.tp_group.ranks[0],
        )[0]

    def _toll_delta(self, req) -> tuple[list[int], int, list[int]]:
        origin = list(getattr(req, "origin_input_ids", []) or [])
        output = list(getattr(req, "output_ids", []) or [])
        full = origin + output
        total = len(full)
        if req.rid in self.toll_need_snapshot or req.rid not in self.toll_sent_len:
            return full, total, full
        sent = int(self.toll_sent_len[req.rid])
        return full[sent:], total, full

    def _toll_request(self, action: str, batch) -> TollRequest:
        self.toll_seq += 1
        now = time.perf_counter_ns()
        elapsed = max(0, now - int(self._toll_last_reply_ns))
        reqs = []
        for req in batch.reqs:
            delta, total, full = self._toll_delta(req)
            snapshot = req.rid in self.toll_need_snapshot or req.rid not in self.toll_sent_len
            reqs.append(
                TollReqItem(
                    rid=req.rid,
                    commit_len=total,
                    new_tokens=[int(token) for token in delta],
                    input_ids=[int(token) for token in full] if snapshot else [],
                    sampling={"temperature": 0.0},
                )
            )
        return TollRequest(
            proto=PROTO_VERSION,
            session=self.toll_session,
            seq=self.toll_seq,
            action=action,
            spec_shape=spec_shape_of(self.toll_cfg),
            t_target_ns=elapsed,
            reqs=reqs,
        )

    def _toll_note_reply(self, reply: TollReply | None, timed_out: bool) -> None:
        self._toll_last_reply_ns = time.perf_counter_ns()
        if timed_out:
            self.toll_breaker.record_failure()
            return
        if reply is None:
            return
        status = reply.reqs[0].status if reply.reqs else TollStatus.EMPTY
        if status == TollStatus.REJECT:
            self.toll_breaker.record_failure()
            return
        if status == TollStatus.NEED_SNAPSHOT and reply.reqs:
            self.toll_need_snapshot.add(reply.reqs[0].rid)
        self.toll_breaker.record_success()

    def _toll_exchange(self, request: TollRequest, wait: bool = True) -> TollReply | None:
        if not self.toll_breaker.should_send():
            self.toll_breaker.note_skipped_step()
            return None
        reply = None
        timed_out = False
        if self.tp_rank == 0 and self.toll_client is not None:
            before = self.toll_client.timeouts
            self.toll_client.send_request(request)
            if wait:
                reply = self.toll_client.recv_reply(request.session, request.seq)
                timed_out = self.toll_client.timeouts > before
        if wait:
            outcome = self._toll_broadcast({"reply": reply, "timed_out": timed_out})
            reply = outcome["reply"]
            timed_out = bool(outcome["timed_out"])
            self._toll_note_reply(reply, timed_out)
        return reply

    def _toll_mark_sent(self, batch, reply: TollReply | None) -> None:
        if reply is None or not reply.reqs:
            return
        status = reply.reqs[0].status
        if status in (TollStatus.REJECT, TollStatus.NEED_SNAPSHOT):
            return
        for req in batch.reqs:
            self.toll_sent_len[req.rid] = committed_len(req)
            self.toll_need_snapshot.discard(req.rid)

    def maybe_notify_remote_draft_finished(self, req) -> None:
        if not self.spec_algorithm.is_toll_tree_remote():
            return super().maybe_notify_remote_draft_finished(req)
        if getattr(self.server_args, "toll_tree_role", None) != "target":
            return
        self._toll_send_control(TollAction.FINISH, req)

    def maybe_notify_remote_draft_aborted(self, reqs) -> None:
        if not self.spec_algorithm.is_toll_tree_remote():
            return super().maybe_notify_remote_draft_aborted(reqs)
        if getattr(self.server_args, "toll_tree_role", None) != "target":
            return
        for req in reqs or []:
            self._toll_send_control(TollAction.ABORT, req)

    def maybe_abort_remote_target(self) -> None:
        if not self.spec_algorithm.is_toll_tree_remote():
            return super().maybe_abort_remote_target()
        if getattr(self.server_args, "toll_tree_role", None) != "target":
            return
        for rid in list(getattr(self, "toll_sent_len", {})):
            req = type("Req", (), {"rid": rid, "origin_input_ids": [], "output_ids": []})()
            self._toll_send_control(TollAction.ABORT, req)
        self.toll_session = uuid.uuid4().hex
        self.toll_seq = 0
        self.toll_sent_len.clear()
        self.toll_need_snapshot.clear()

    def _toll_send_control(self, action: str, req) -> None:
        if self.tp_rank != 0 or self.toll_client is None:
            return
        batch = type("Batch", (), {"reqs": [req]})()
        self.toll_client.send_request(self._toll_request(action, batch))
        self.toll_sent_len.pop(req.rid, None)
        self.toll_need_snapshot.discard(req.rid)

    def _toll_run_batch(self, batch) -> None:
        if len(batch.reqs) != 1:
            raise RuntimeError("TOLL_TREE_REMOTE serving currently requires batch size 1")
        req = batch.reqs[0]
        extend = batch.forward_mode is not None and batch.forward_mode.is_extend()
        if not extend and skip_speculative(batch):
            batch.draft_num_tokens = 1
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
            return
        action = TollAction.PREFILL if extend else TollAction.STEP
        request = self._toll_request(action, batch)
        reply = self._toll_exchange(request)
        self._toll_mark_sent(batch, reply)
        useful = (
            reply is not None
            and reply.reqs
            and reply.reqs[0].status == TollStatus.OK
            and reply.reqs[0].tokens
        )
        if not useful:
            batch.draft_num_tokens = 1
            req.toll_reply = None
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
            return
        req.toll_reply = reply
        batch.draft_num_tokens = int(self.server_args.speculative_num_draft_tokens)
        result = self.run_batch(batch)
        self.process_batch_result(batch, result)

    @DynamicGradMode()
    def event_loop_normal_toll_tree_target(self) -> None:
        self.init_toll_tree_communication()
        while True:
            self.process_input_requests(self.recv_requests())
            if getattr(self, "_engine_paused", False):
                continue
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            if batch:
                self._toll_run_batch(batch)
            else:
                self.self_check_during_idle()
            self.last_batch = batch
