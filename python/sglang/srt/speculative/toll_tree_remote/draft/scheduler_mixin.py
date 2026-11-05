"""Draft request handler. The event loop lives on the scheduler mixin."""

from __future__ import annotations

import time

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


class DraftForwardError(RuntimeError):
    """A real draft forward failed. The step returns an empty draft."""


def is_device_context_error(exc: BaseException) -> bool:
    """True when the accelerator context is already dead and must not become an empty draft."""
    name = type(exc).__name__
    if name in ("AcceleratorError", "CUDAError", "NPUError", "XPUError"):
        return True
    try:
        import torch

        accel = getattr(torch, "AcceleratorError", None)
    except Exception:
        accel = None
    if accel is not None and isinstance(exc, accel):
        return True
    message = str(exc).lower()
    return (
        "illegal memory access" in message
        or "device-side assert" in message
        or "cudaerrorillegaladdress" in message
        or "npu error" in message
        or ("ascend" in message and "illegal" in message)
    )


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
        req_factory=None,
        release_req=None,
        extend_new=None,
    ) -> None:
        self.cfg = cfg
        self.forward = forward
        self.alloc = alloc
        self.free = free
        self.root_topk = root_topk
        self.layer_forward = layer_forward
        self.ingest = ingest
        self.req_factory = req_factory
        self.release_req = release_req
        self.extend_new = extend_new
        self.empty_forwards = 0
        self.cached_len = 0
        self._round_open = False
        self._pending_commits: dict[str, list[int]] = {}
        self._pending_target_ms = 0.0
        self._net_ms = 0.0
        self._net_bytes = 0.0
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

    def expire_idle(self) -> None:
        for rid in self.state.expire():
            self._drop(rid)

    def begin_round(self, net_ms: float, net_bytes: float) -> None:
        """Network figures for the RPC that is about to run. The first RPC is zero."""
        self._net_ms = max(0.0, float(net_ms))
        self._net_bytes = max(0.0, float(net_bytes))

    def close_round(self) -> None:
        """Label the previous tree after the reply has been sent."""
        if not self._round_open:
            return
        self.round.after_reply()
        self._round_open = False

    def _empty_after_error(self, request, exc: BaseException) -> TollReply:
        if is_device_context_error(exc):
            raise exc
        self.empty_forwards += 1
        return self._reply(request, [], TollStatus.EMPTY)

    def _prefill(self, request, item) -> TollReply:
        ids = [int(token) for token in item.input_ids]
        if not ids:
            ids = [0] * int(item.commit_len)
        try:
            self._materialize(item.rid, ids, int(item.commit_len) or len(ids))
        except Exception as exc:
            return self._empty_after_error(request, exc)
        return self._reply(request, [], TollStatus.OK)

    def _materialize(self, rid: str, ids: list[int], commit_len: int) -> None:
        from sglang.srt.speculative.toll_tree_remote.draft.reqs import make_draft_req

        factory = self.req_factory or make_draft_req
        req = factory(rid, ids)
        slots = self._alloc_exact(len(ids)) if ids else []
        if ids and slots is None:
            raise DraftForwardError(f"draft {rid} has no prefix slots")
        self.prefix_slots[rid] = list(slots or [])
        self.reqs[rid] = req
        self.kv[rid] = TreeKV(self.alloc, self.free)
        if self.ingest is not None:
            self.ingest(req, self.prefix_slots[rid])
        prefix = len(ids)
        self.state.upsert(rid, prefix, int(commit_len) or prefix)

    def _step(self, request, item) -> TollReply:
        from sglang.srt.speculative.toll_tree_remote.policy.round import RoundReq

        full = [int(token) for token in item.input_ids]
        stored = self.state.get(item.rid)
        if stored is None:
            if not full:
                return self._reply(request, [], TollStatus.NEED_SNAPSHOT)
            try:
                self._materialize(item.rid, full, len(full))
            except Exception as exc:
                return self._empty_after_error(request, exc)
            stored = self.state.get(item.rid)
            if stored is None:
                return self._reply(request, [], TollStatus.EMPTY)
        try:
            if int(item.commit_len) == int(stored.commit_len) and full and int(item.commit_len) == len(full):
                delta = []
            else:
                delta = self._take_commit(item, stored)
        except Exception as exc:
            return self._empty_after_error(request, exc)
        if delta is None:
            return self._reply(request, [], TollStatus.NEED_SNAPSHOT)
        self._rid = item.rid
        self.cols.clear()
        self._budget_ok = True
        self._pending_commits = {item.rid: list(delta)}
        self._pending_target_ms = int(request.t_target_ns) / 1e6
        try:
            self.round.on_step(
                time.perf_counter_ns(),
                {item.rid: list(delta)},
                bs_prev=1,
                continuous=True,
                net_ms=self._net_ms,
                net_bytes=self._net_bytes,
                target_ms=self._pending_target_ms,
            )
            top = self._root(item)
            result = self.round.run(
                [RoundReq(item.rid, stored.prefix_len)],
                [top],
                self._forward,
                kv_ok=self._kv_ok,
            )
        except Exception as exc:
            self._rollback_tree(item.rid)
            self._round_open = False
            return self._empty_after_error(request, exc)
        self._rollback_tree(item.rid)
        self.cols.clear()
        self.slot_of_node = {}
        verify = result.verify[0]
        tokens, parents = list(verify.tokens), list(verify.parents)
        if not tokens:
            self.empty_forwards += 1
            self._round_open = False
            return self._reply(request, [], TollStatus.EMPTY)
        self._round_open = True
        return self._reply(request, [(tokens, parents)], TollStatus.OK)

    def _rollback_tree(self, rid: str) -> None:
        kv = self.kv.get(rid)
        if kv is not None:
            kv.rollback()

    def _root(self, item):
        if self.root_topk is None:
            raise DraftForwardError(f"draft {item.rid} has no root distribution")
        return self.root_topk(item)

    def _kv_ok(self, _depth: int = 0) -> bool:
        return bool(self._budget_ok)

    def _forward(self, trees, frontiers, width):
        width = int(width)
        started = time.perf_counter()
        kv = self.kv.get(self._rid)
        allocated = []
        base = 0
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
        prefix = int(trees[0].prefix_len) if trees else 0
        self.cached_len = prefix + base
        self.current_slots = list(allocated[0] or []) if allocated else []
        self.slot_of_node = dict(self.cols)
        if not self._budget_ok or self.layer_forward is None:
            raise DraftForwardError("draft layer has no KV slot or forward")
        outputs = self.layer_forward(trees, frontiers, width)
        elapsed = (time.perf_counter() - started) * 1000.0
        if not outputs:
            raise DraftForwardError("draft layer returned no logits")
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
                req = self.reqs.get(item.rid)
                if self.extend_new is not None and req is not None:
                    self.extend_new(req, slots)
            stored.commit_len = int(item.commit_len)
            stored.prefix_len = int(stored.commit_len)
            return delta
        if full and int(item.commit_len) == len(full):
            if not self._replace_prefix(item.rid, full):
                return None
            stored.commit_len = len(full)
            stored.prefix_len = len(full)
            return []
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
        previous = self.reqs.get(rid)
        if self.release_req is not None and previous is not None:
            self.release_req(previous)
        self.prefix_slots[rid] = list(slots or [])
        factory = self.req_factory or make_draft_req
        req = factory(rid, full)
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
        req = self.reqs.pop(rid, None)
        if self.release_req is not None and req is not None:
            self.release_req(req)
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
        from sglang.srt.speculative.toll_tree_remote.transport import TollDraftServer

        cfg = TollConfig.from_server_args(self.server_args, "draft")
        url = f"tcp://*:{self.server_args.toll_tree_port}"
        self.toll_cfg = cfg
        self._toll_root_logits = None
        self._toll_last_send_ns = None
        self._toll_rpc_bytes = 0
        self._toll_check_left = 4
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
            ingest=lambda req, slots: self._toll_ingest(req, slots),
            release_req=self._toll_release_req,
            extend_new=self._toll_extend_new,
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

    def _toll_release_req(self, req) -> None:
        pool = getattr(self, "req_to_token_pool", None)
        if pool is not None and getattr(req, "req_pool_idx", None) is not None:
            pool.free(req)

    def _toll_note_logits(self, logits) -> None:
        if logits is not None:
            self._toll_root_logits = logits

    def _toll_ingest(self, req, slots):
        from sglang.srt.speculative.toll_tree_remote.draft.reqs import ingest_prefix

        self._toll_note_logits(ingest_prefix(self, req, slots))

    def _toll_extend_new(self, req, slots) -> None:
        from sglang.srt.speculative.toll_tree_remote.draft.reqs import extend_committed

        start = int(getattr(req, "kv_committed_len", 0) or 0)
        tokens = list(req.fill_ids[start:])
        if len(tokens) != len(list(slots)):
            tokens = list(req.fill_ids[-len(slots) :])
        self._toll_note_logits(extend_committed(self, req, start, tokens, slots))

    def _toll_root_topk(self, item):
        from sglang.srt.speculative.toll_tree_remote.draft.forward import topc_from_logits

        logits = getattr(self, "_toll_root_logits", None)
        if logits is None:
            raise DraftForwardError(f"draft {item.rid} has no root logits")
        row = logits[-1] if getattr(logits, "dim", lambda: 1)() == 2 else logits
        width = int(self.toll_cfg.fixed_width or self.toll_cfg.width_buckets[0])
        probs, ids = topc_from_logits(row, width)
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
            logits = run_layer_logits(
                self,
                req,
                tokens,
                positions,
                mask,
                slots,
                int(self.toll_draft.cached_len),
            )
            self._toll_maybe_check_layer(tree, kept, logits)
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

    def _toll_reject_http(self) -> None:
        waiting = list(getattr(self, "waiting_queue", []) or [])
        if not waiting:
            return
        self.waiting_queue.clear()
        message = "TOLL_TREE_REMOTE draft does not accept HTTP. Send requests to the target."
        for req in waiting:
            req.set_finish_with_abort(message)
            if callable(getattr(req, "check_finished", None)):
                req.check_finished()
            self.stream_output([req], bool(getattr(req, "return_logprob", False)))

    def _toll_maybe_check_layer(self, tree, kept, logits) -> None:
        from sglang.srt.speculative.toll_tree_remote.draft.check_layer import (
            check_enabled,
            reference_and_report,
        )

        if not kept or not check_enabled() or self._toll_check_left <= 0:
            return
        self._toll_check_left -= 1
        reference_and_report(self, self.toll_draft.reqs[self.toll_draft._rid], tree, int(kept[0]), logits)

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
            self._toll_reject_http()
            before = int(self.toll_server.bytes_recv) + int(self.toll_server.bytes_sent)
            request = self.toll_server.recv_request(timeout_ms=50)
            if request is None:
                self.toll_draft.expire_idle()
                continue
            now = time.perf_counter_ns()
            if self._toll_last_send_ns is None:
                net_ms = 0.0
            else:
                gap_ns = now - int(self._toll_last_send_ns) - int(request.t_target_ns)
                net_ms = max(0.0, gap_ns / 1e6)
            self.toll_draft.begin_round(net_ms, self._toll_rpc_bytes)
            reply = self.toll_draft.handle(request)
            reply.residence_ns = self.toll_server.residence_ns()
            self.toll_server.send_reply(reply)
            self._toll_last_send_ns = time.perf_counter_ns()
            after = int(self.toll_server.bytes_recv) + int(self.toll_server.bytes_sent)
            self._toll_rpc_bytes = max(0, after - before)
            self.toll_draft.close_round()
