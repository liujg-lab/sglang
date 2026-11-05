"""Target event loop. Rank 0 owns the socket. Other ranks receive the reply."""

from __future__ import annotations

import time

from sglang.srt.speculative.toll_tree_remote.target.breaker import TollRpcBreaker
from sglang.srt.speculative.toll_tree_remote.protocol import (
    PROTO_VERSION,
    TollAction,
    TollReply,
    TollReqItem,
    TollRequest,
    spec_shape_of,
)


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
        url = f"tcp://{self.server_args.toll_tree_addr}:{self.server_args.toll_tree_port}"
        if self.tp_rank == 0:
            self.toll_client = TollTargetClient(url, self.toll_cfg)
        else:
            self.toll_client = None

    def _toll_broadcast(self, payload):
        from sglang.srt.utils import broadcast_pyobj

        return broadcast_pyobj(
            [payload] if self.tp_rank == 0 else None,
            self.tp_rank,
            self.dist_port if hasattr(self, "dist_port") else 0,
            self.tp_cpu_group if hasattr(self, "tp_cpu_group") else None,
        )[0]

    def _toll_request(self, action: str, batch) -> TollRequest:
        self.toll_seq += 1
        reqs = []
        for req in batch.reqs:
            reqs.append(
                TollReqItem(
                    rid=req.rid,
                    commit_len=len(getattr(req, "output_ids", []) or []),
                    new_tokens=list(getattr(req, "output_ids", []) or [])[-1:],
                    input_ids=list(getattr(req, "origin_input_ids", []) or []),
                    sampling={"temperature": 0.0},
                )
            )
        return TollRequest(
            proto=PROTO_VERSION,
            session="toll",
            seq=self.toll_seq,
            action=action,
            spec_shape=spec_shape_of(self.toll_cfg),
            t_target_ns=time.perf_counter_ns(),
            reqs=reqs,
        )

    def _toll_exchange(self, request: TollRequest) -> TollReply | None:
        if not self.toll_breaker.should_send():
            self.toll_breaker.note_skipped_step()
            return None
        reply = None
        if self.tp_rank == 0:
            self.toll_client.send_request(request)
            reply = self.toll_client.recv_reply(request.session, request.seq)
        reply = self._toll_broadcast(reply)
        if reply is None or not reply.reqs or not reply.reqs[0].tokens:
            self.toll_breaker.record_failure()
            return None
        self.toll_breaker.record_success()
        return reply

    def _toll_run_batch(self, batch) -> None:
        if len(batch.reqs) != 1:
            raise RuntimeError("TOLL_TREE_REMOTE serving currently requires batch size 1")
        extend = batch.forward_mode is not None and batch.forward_mode.is_extend()
        action = TollAction.PREFILL if extend else TollAction.STEP
        reply = self._toll_exchange(self._toll_request(action, batch))
        if reply is None or not reply.reqs[0].tokens:
            batch.draft_num_tokens = 1
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
            return
        worker = self.tp_worker
        spec = worker.build_verify_input(
            batch,
            [reply.reqs[0].tokens],
            [reply.reqs[0].parents],
            [reply.reqs[0].tokens[0]],
            int(self.tp_worker.model_config.vocab_size),
            int(self.toll_cfg.max_depth),
        )
        if spec is None:
            batch.draft_num_tokens = 1
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
            return
        batch.spec_info = spec
        batch.draft_num_tokens = spec.draft_token_num
        result = self.run_batch(batch)
        self.process_batch_result(batch, result)
        req = batch.reqs[0]
        if req.finished():
            self._toll_exchange(self._toll_request(TollAction.FINISH, batch))

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
