"""Draft request handler. The event loop lives on the scheduler mixin."""

from __future__ import annotations

import time

import numpy as np

from sglang.srt.speculative.toll_tree_remote.draft.kv import TreeKV
from sglang.srt.speculative.toll_tree_remote.draft.state import DraftStateStore
from sglang.srt.speculative.toll_tree_remote.protocol import (
    TollAction,
    TollReply,
    TollReplyItem,
    TollStatus,
    request_ok,
)
from sglang.srt.utils import DynamicGradMode


class TollDraftService:
    """PREFILL writes the prefix. STEP appends the commit, then runs one round."""

    def __init__(
        self,
        cfg,
        forward,
        alloc,
        free,
        ttl_s: float = 60.0,
        root_topk=None,
        layer_forward=None,
        ingest=None,
    ) -> None:
        self.cfg = cfg
        self.forward = forward
        self.alloc = alloc
        self.free = free
        self.root_topk = root_topk
        self.layer_forward = layer_forward
        self.ingest = ingest
        self.state = DraftStateStore(ttl_s)
        self.kv: dict[str, TreeKV] = {}
        self.prefix_slots: dict[str, list[int]] = {}
        self.reqs: dict[str, object] = {}
        self.cols: dict[int, int] = {}
        self.slot_of_node: dict[int, int] = {}
        self.current_slots: list[int] = []
        self._rid = ""
        self._budget_ok = True
        from sglang.srt.speculative.toll_tree_remote.policy.calib import Calibrator
        from sglang.srt.speculative.toll_tree_remote.policy.cost import TierCostModel
        from sglang.srt.speculative.toll_tree_remote.policy.gain import GainTable
        from sglang.srt.speculative.toll_tree_remote.policy.price import PriceTracker
        from sglang.srt.speculative.toll_tree_remote.policy.round import TollRound

        self.round = TollRound(
            cfg,
            PriceTracker(alpha=float(cfg.price_alpha)),
            TierCostModel(
                probe_interval=int(cfg.probe_interval),
                warmup_rounds=int(cfg.warmup_rounds),
            ),
            Calibrator(),
            GainTable(),
            "sync",
        )

    def handle(self, request) -> TollReply:
        if not request_ok(request, self.cfg):
            return self._reply(request, [], TollStatus.REJECT)
        if len(request.reqs) != 1:
            return self._reply(request, [], TollStatus.REJECT)
        if self.state.session not in (None, request.session):
            for rid in list(self.prefix_slots):
                self._drop(rid)
            for rid in list(self.kv):
                self._drop(rid)
        self.state.begin_session(request.session)
        item = request.reqs[0]
        for rid in self.state.expire(keep={item.rid}):
            self._drop(rid)
        if request.action == TollAction.PREFILL:
            return self._prefill(request, item)
        if request.action == TollAction.STEP:
            return self._step(request, item)
        if request.action in (TollAction.FINISH, TollAction.ABORT):
            self._drop(item.rid)
            return self._reply(request, [], TollStatus.OK)
        return self._reply(request, [], TollStatus.REJECT)

    def _prefill(self, request, item) -> TollReply:
        from sglang.srt.speculative.toll_tree_remote.draft.reqs import make_draft_req

        ids = [int(token) for token in item.input_ids]
        if not ids:
            ids = [0] * int(item.commit_len)
        req = make_draft_req(item.rid, ids)
        slots = self._alloc_exact(len(ids)) if ids else []
        if ids and slots is None:
            return self._reply(request, [], TollStatus.EMPTY)
        self.prefix_slots[item.rid] = list(slots or [])
        self.reqs[item.rid] = req
        self.kv[item.rid] = TreeKV(self.alloc, self.free)
        if self.ingest is not None:
            self.ingest(req, self.prefix_slots[item.rid])
        prefix = len(ids)
        self.state.upsert(item.rid, prefix, int(item.commit_len) or prefix)
        return self._reply(request, [], TollStatus.OK)

    def _step(self, request, item) -> TollReply:
        from sglang.srt.speculative.toll_tree_remote.policy.round import RoundReq

        stored = self.state.get(item.rid)
        if stored is None:
            return self._reply(request, [], TollStatus.NEED_SNAPSHOT)
        delta = self._take_commit(item, stored)
        if delta is None:
            return self._reply(request, [], TollStatus.NEED_SNAPSHOT)
        self._rid = item.rid
        self.cols.clear()
        self._budget_ok = True
        self.round.on_step(
            time.perf_counter_ns(),
            {item.rid: delta},
            bs_prev=1,
            continuous=True,
            net_ms=0.0,
            net_bytes=0.0,
            target_ms=int(request.t_target_ns) / 1e6,
        )
        top = self._root(item)
        result = self.round.run(
            [RoundReq(item.rid, stored.prefix_len)],
            [top],
            self._forward,
            kv_ok=self._kv_ok,
        )
        self.round.after_reply()
        kv = self.kv.get(item.rid)
        if kv is not None:
            kv.rollback()
        self.cols.clear()
        self.slot_of_node = {}
        verify = result.verify[0]
        tokens, parents = list(verify.tokens), list(verify.parents)
        if not tokens:
            return self._reply(request, [], TollStatus.EMPTY)
        return self._reply(request, [(tokens, parents)], TollStatus.OK)

    def _root(self, item):
        width = int(self.cfg.fixed_width or self.cfg.width_buckets[0])
        if self.root_topk is not None:
            return self.root_topk(item)
        scores = np.full(width, 1.0 / width, dtype=np.float64)
        return scores, np.arange(width)

    def _kv_ok(self, _depth: int = 0) -> bool:
        return bool(self._budget_ok)

    def _forward(self, trees, frontiers, width):
        width = int(width)
        started = time.perf_counter()
        kv = self.kv.get(self._rid)
        allocated = []
        for frontier in frontiers:
            kept = [int(node) for node in list(frontier)]
            if kv is None or not kept:
                allocated.append(None)
                continue
            slots = kv.alloc_layer(width)
            if slots is None:
                self._budget_ok = False
                allocated.append(None)
                continue
            base = sum(len(layer) for layer in kv.layers[:-1])
            for index, node in enumerate(kept):
                self.cols[int(node)] = base + index
            allocated.append(slots)
        self.current_slots = list(allocated[0] or []) if allocated else []
        self.slot_of_node = dict(self.cols)
        if self.layer_forward is None or not self._budget_ok:
            outputs = []
        else:
            outputs = self.layer_forward(trees, frontiers, width)
        elapsed = (time.perf_counter() - started) * 1000.0
        if not outputs:
            outputs = []
            for frontier in frontiers:
                rows = max(len(list(frontier)), 1)
                outputs.append(
                    (
                        np.zeros((rows, width), dtype=np.float64),
                        np.zeros((rows, width), dtype=np.int64),
                        elapsed,
                    )
                )
        fixed = []
        for scores, ids, ms in outputs:
            taken = float(ms)
            fixed.append((scores, ids, taken if taken > 0.0 else elapsed))
        return fixed

    def _take_commit(self, item, stored):
        delta = [int(token) for token in item.new_tokens]
        full = [int(token) for token in item.input_ids]
        expected = int(stored.commit_len) + len(delta)
        if int(item.commit_len) == expected and self._suffix_matches(stored, delta, full):
            if delta:
                slots = self._alloc_exact(len(delta))
                if slots is None:
                    return None
                self.prefix_slots.setdefault(item.rid, []).extend(slots)
                self._append_req(item.rid, delta)
            stored.commit_len = int(item.commit_len)
            stored.prefix_len = int(stored.commit_len)
            return delta
        if full and int(item.commit_len) == len(full):
            if not self._replace_prefix(item.rid, full):
                return None
            stored.commit_len = len(full)
            stored.prefix_len = len(full)
            return full
        return None

    def _suffix_matches(self, stored, delta, full) -> bool:
        if not delta or not full:
            return True
        start = int(stored.commit_len)
        return full[start : start + len(delta)] == delta

    def _append_req(self, rid: str, delta) -> None:
        req = self.reqs.get(rid)
        if req is None:
            return
        req.output_ids.extend(delta)
        req.fill_ids = list(req.origin_input_ids) + list(req.output_ids)

    def _replace_prefix(self, rid: str, full) -> bool:
        from sglang.srt.speculative.toll_tree_remote.draft.reqs import make_draft_req

        slots = self._alloc_exact(len(full)) if full else []
        if full and slots is None:
            return False
        old = self.prefix_slots.pop(rid, [])
        if old:
            self.free(old)
        self.prefix_slots[rid] = list(slots or [])
        req = make_draft_req(rid, full)
        self.reqs[rid] = req
        if self.ingest is not None and slots:
            self.ingest(req, slots)
        return True

    def _alloc_exact(self, width: int):
        slots = list(self.alloc(int(width)) or [])
        if len(slots) < int(width):
            if slots:
                self.free(slots)
            return None
        return slots

    def _drop(self, rid: str) -> None:
        slots = self.prefix_slots.pop(rid, None)
        if slots:
            self.free(slots)
        kv = self.kv.pop(rid, None)
        if kv is not None:
            kv.rollback()
        self.reqs.pop(rid, None)
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
            reqs.append(TollReplyItem(rid="", base_len=0, tokens=[], parents=[], status=status))
        bucket = int(self.cfg.fixed_width or self.cfg.verify_buckets[0])
        return TollReply(
            proto=request.proto,
            session=request.session,
            seq=int(request.seq),
            verify_bucket=bucket,
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
        from sglang.srt.speculative.toll_tree_remote.draft.reqs import ingest_prefix
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
            ingest=lambda req, slots: ingest_prefix(self, req, slots),
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
        from sglang.srt.speculative.toll_tree_remote.draft.reqs import tail_logits

        req = self.toll_draft.reqs[item.rid]
        logits = tail_logits(self, req)
        width = int(self.toll_cfg.fixed_width or self.toll_cfg.width_buckets[0])
        probs, ids = topc_from_logits(logits, width)
        return probs.reshape(-1).detach().cpu().tolist(), ids.reshape(-1).detach().cpu().tolist()

    def _toll_layer_forward(self, trees, frontiers, width):
        import torch

        from sglang.srt.speculative.toll_tree_remote.draft.forward import topc_from_logits
        from sglang.srt.speculative.toll_tree_remote.draft.reqs import (
            layer_positions,
            run_layer_logits,
        )
        from sglang.srt.speculative.toll_tree_remote.tree.mask import layer_mask

        outputs = []
        slots = list(self.toll_draft.current_slots)
        slot_of_node = self.toll_draft.slot_of_node
        for tree, frontier in zip(trees, frontiers):
            kept = [int(node) for node in list(frontier)]
            mask = layer_mask(tree.prefix_len, tree, kept, int(width), slot_of_node)
            depth = int(tree.depth[kept[0]]) if kept else 1
            positions = layer_positions(tree.prefix_len, depth, int(width))
            tokens = [0] * int(width)
            for index, node in enumerate(kept):
                tokens[index] = int(tree.token[node])
            req = self.toll_draft.reqs[self.toll_draft._rid]
            started = time.perf_counter()
            logits = run_layer_logits(self, req, tokens, positions, mask, slots)
            elapsed = (time.perf_counter() - started) * 1000.0
            probs, ids = topc_from_logits(logits, int(width))
            outputs.append(
                (
                    probs.detach().cpu().numpy(),
                    ids.detach().cpu().numpy(),
                    elapsed,
                )
            )
        return outputs

    @property
    def is_remote_spec_draft(self) -> bool:
        if (
            self.spec_algorithm.is_toll_tree_remote()
            and getattr(self.server_args, "toll_tree_role", None) == "draft"
        ):
            return True
        return super().is_remote_spec_draft

    @DynamicGradMode()
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
