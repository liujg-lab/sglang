"""Draft request handler. The event loop lives on the scheduler mixin."""

from __future__ import annotations

from sglang.srt.speculative.toll_tree_remote.draft.kv import TreeKV
from sglang.srt.speculative.toll_tree_remote.draft.state import DraftStateStore
from sglang.srt.speculative.toll_tree_remote.protocol import (
    TollAction,
    TollReply,
    TollReplyItem,
    TollStatus,
    request_ok,
    spec_shape_of,
)


class TollDraftService:
    """PREFILL stores the prefix. STEP runs one fixed-width round."""

    def __init__(self, cfg, forward, alloc, free, ttl_s: float = 60.0, root_topk=None, layer_forward=None) -> None:
        self.cfg = cfg
        self.forward = forward
        self.alloc = alloc
        self.free = free
        self.root_topk = root_topk
        self.layer_forward = layer_forward
        self.state = DraftStateStore(ttl_s)
        self.kv: dict[str, TreeKV] = {}
        from sglang.srt.speculative.toll_tree_remote.policy.calib import Calibrator
        from sglang.srt.speculative.toll_tree_remote.policy.cost import TierCostModel
        from sglang.srt.speculative.toll_tree_remote.policy.gain import GainTable
        from sglang.srt.speculative.toll_tree_remote.policy.price import PriceTracker
        from sglang.srt.speculative.toll_tree_remote.policy.round import TollRound

        self.round = TollRound(
            cfg, PriceTracker(), TierCostModel(), Calibrator(), GainTable(), "sync"
        )

    def handle(self, request) -> TollReply:
        if not request_ok(request, self.cfg):
            return self._reply(request, [], TollStatus.REJECT)
        if len(request.reqs) != 1:
            return self._reply(request, [], TollStatus.REJECT)
        self.state.begin_session(request.session)
        self.state.expire(keep={request.reqs[0].rid})
        item = request.reqs[0]
        if request.action == TollAction.PREFILL:
            return self._prefill(request, item)
        if request.action == TollAction.STEP:
            return self._step(request, item)
        if request.action in (TollAction.FINISH, TollAction.ABORT):
            self._drop(item.rid)
            return self._reply(request, [], TollStatus.OK)
        return self._reply(request, [], TollStatus.REJECT)

    def _prefill(self, request, item) -> TollReply:
        prefix = int(item.commit_len) or len(item.input_ids)
        self.state.upsert(item.rid, prefix, int(item.commit_len) or prefix)
        self.kv[item.rid] = TreeKV(self.alloc, self.free)
        return self._reply(request, [], TollStatus.OK)

    def _step(self, request, item) -> TollReply:
        stored = self.state.get(item.rid)
        if stored is None or int(stored.commit_len) != int(item.commit_len):
            return self._reply(request, [], TollStatus.NEED_SNAPSHOT)
        kv = self.kv.setdefault(item.rid, TreeKV(self.alloc, self.free))
        width = int(self.cfg.fixed_width)
        slots = kv.alloc_layer(width)
        if slots is None:
            return self._reply(request, [], TollStatus.EMPTY)
        if self.root_topk is not None and self.layer_forward is not None:
            from sglang.srt.speculative.toll_tree_remote.policy.round import RoundReq

            result = self.round.run(
                [RoundReq(item.rid, stored.prefix_len)],
                [self.root_topk(item)],
                self.layer_forward,
            )
            verify = result.verify[0]
            tokens, parents = list(verify.tokens), list(verify.parents)
        else:
            tokens, parents = self.forward(item, slots)
        kv.rollback()
        if not tokens:
            return self._reply(request, [], TollStatus.EMPTY)
        stored.commit_len = int(item.commit_len)
        stored.prefix_len = int(item.commit_len)
        return self._reply(request, [(tokens, parents)], TollStatus.OK)

    def _drop(self, rid: str) -> None:
        kv = self.kv.pop(rid, None)
        if kv is not None:
            kv.rollback()
        self.state.delete(rid)

    def _reply(self, request, windows, status: str) -> TollReply:
        reqs = []
        for item in request.reqs:
            tokens, parents = windows[0] if windows else ([], [])
            if status != TollStatus.OK:
                tokens, parents = [], []
            reqs.append(
                TollReplyItem(
                    rid=item.rid,
                    base_len=int(item.commit_len),
                    tokens=[int(token) for token in tokens] if status == TollStatus.OK else [],
                    parents=[int(parent) for parent in parents] if status == TollStatus.OK else [],
                    status=status,
                )
            )
        if not reqs:
            reqs.append(
                TollReplyItem(rid="", base_len=0, tokens=[], parents=[], status=status)
            )
        return TollReply(
            proto=request.proto,
            session=request.session,
            seq=int(request.seq),
            verify_bucket=int(self.cfg.fixed_width or self.cfg.verify_buckets[0]),
            residence_ns=0,
            reqs=reqs,
        )


class TollDraftSchedulerMixin:
    """Draft event loop. Not-Toll calls fall through to the next mixin."""

    def init_toll_tree_communication(self) -> None:
        if not self.spec_algorithm.is_toll_tree_remote():
            return
        if getattr(self.server_args, "toll_tree_role", None) != "draft":
            super().init_toll_tree_communication()
            return
        from sglang.srt.speculative.toll_tree_remote.config import TollConfig
        from sglang.srt.speculative.toll_tree_remote.transport import TollDraftServer

        cfg = TollConfig.from_server_args(self.server_args, "draft")
        url = f"tcp://{self.server_args.toll_tree_addr}:{self.server_args.toll_tree_port}"
        self.toll_cfg = cfg
        self.toll_server = TollDraftServer(url)
        self.toll_draft = TollDraftService(
            cfg,
            forward=lambda item, slots: ([], []),
            alloc=lambda width: self._toll_alloc(width),
            free=lambda slots: self._toll_free(slots),
            root_topk=lambda item: self._toll_root_topk(item),
            layer_forward=lambda trees, frontiers, width: self._toll_layer_forward(
                trees, frontiers, width
            ),
        )

    def _toll_alloc(self, width: int):
        from sglang.srt.mem_cache.common import alloc_token_slots

        loc = alloc_token_slots(self.tree_cache, int(width))
        if loc is None:
            return []
        return [int(v) for v in loc.reshape(-1).tolist()]

    def _toll_free(self, slots) -> None:
        if not slots:
            return
        import torch

        allocator = self.token_to_kv_pool_allocator
        allocator.free(torch.tensor(list(slots), dtype=torch.int64, device=allocator.device))

    def _toll_root_topk(self, item):
        from sglang.srt.speculative.toll_tree_remote.draft.forward import topc_from_logits

        logits = self.tp_worker.model_runner.forward_extend_logits(item)
        probs, ids = topc_from_logits(logits, int(self.toll_cfg.fixed_width))
        return probs.reshape(-1).tolist(), ids.reshape(-1).tolist()

    def _toll_layer_forward(self, trees, frontiers, width):
        from sglang.srt.speculative.toll_tree_remote.draft.forward import topc_from_logits
        from sglang.srt.speculative.toll_tree_remote.draft.spec_input import TollLayerSpec
        from sglang.srt.speculative.toll_tree_remote.tree.mask import layer_mask

        outputs = []
        for tree, frontier in zip(trees, frontiers):
            slots = {int(node): index for index, node in enumerate(frontier)}
            mask = layer_mask(tree.prefix_len, tree, frontier, int(width), slots)
            spec = TollLayerSpec(
                custom_mask=mask.reshape(-1),
                positions=list(range(int(width))),
                draft_token_num=int(width),
                num_tokens_per_req=int(width),
            )
            logits = self.tp_worker.model_runner.forward_extend_logits(tree, spec)
            probs, ids = topc_from_logits(logits, int(width))
            outputs.append((probs.cpu().numpy(), ids.cpu().numpy(), 0.0))
        return outputs

    @property
    def is_remote_spec_draft(self) -> bool:
        if (
            self.spec_algorithm.is_toll_tree_remote()
            and getattr(self.server_args, "toll_tree_role", None) == "draft"
        ):
            return True
        return super().is_remote_spec_draft

    def event_loop_normal_toll_tree_draft(self) -> None:
        self.init_toll_tree_communication()
        while True:
            self.process_input_requests(self.recv_requests())
            request = self.toll_server.recv_request(timeout_ms=50)
            if request is None:
                continue
            reply = self.toll_draft.handle(request)
            reply.residence_ns = self.toll_server.residence_ns()
            self.toll_server.send_reply(reply)

