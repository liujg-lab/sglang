import logging
import os
import time
from collections import OrderedDict
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.sampler import SamplingBatchInfo
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spectre.drafter.spectre_kv_rollbacker import (
    SpectreKVRollbacker,
)
from sglang.srt.speculative.spectre.drafter.spectre_state_manager import (
    SpectreDraftState,
    SpectreDraftStateManager,
)
from sglang.srt.speculative.spectre.spectre_protocol import (
    SpectreAction,
    SpectreRequest,
    SpecType,
)
# [SPECTRE-VL] Draft 自跑 ViT：旁路收 payload、预热 embedding、re-prefill 重算 M-RoPE。
from sglang.srt.speculative.spectre.spectre_mm_transport import (
    payload_gpu_bytes,
    payload_resident_bytes,
    release_mm_resources,
    reset_mm_mrope,
    spectre_mm_prewarm_bytes,
    spectre_mm_prewarm_max,
    spectre_mm_stale_s,
    spectre_mm_wait_ms,
)
from sglang.srt.utils import DynamicGradMode, broadcast_pyobj

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DraftReqLocation(str, Enum):
    DRAFT_WAITING = "draft_waiting"
    DRAFT_BATCH = "draft_batch"
    PAUSED = "paused"


def _fix_sampling_params_stop_strs(sp) -> None:
    if not hasattr(sp, "stop_strs") or sp.stop_strs is None:
        sp.stop_strs = []
    elif isinstance(sp.stop_strs, str):
        sp.stop_strs = [sp.stop_strs]

    if not hasattr(sp, "stop_regex_strs") or sp.stop_regex_strs is None:
        sp.stop_regex_strs = []
    elif isinstance(sp.stop_regex_strs, str):
        sp.stop_regex_strs = [sp.stop_regex_strs]


class SpectreDraftSchedulerMixin:
    def _init_draft_components(self) -> None:
        self.draft_state_manager = SpectreDraftStateManager(timeout_threshold=60.0)
        self.draft_kv_manager = SpectreKVRollbacker(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            tree_cache=self.tree_cache,
            page_size=self.server_args.page_size or 1,
            tp_rank=self.tp_rank,
        )

        self.draft_waiting_queue: List[Req] = []
        self.draft_batch: ScheduleBatch = ScheduleBatch(reqs=[])
        self.draft_paused_reqs: List[Req] = []
        self.last_draft_batch: Optional[ScheduleBatch] = None
        self._draft_batch_pending_adds: List[Req] = []
        self.draft_forward_cycle: int = 0
        self.draft_cleanup_interval: int = int(
            os.environ.get("SGLANG_DRAFT_CLEANUP_INTERVAL", "500")
        )
        self._pending_mm: Dict[str, Any] = {}  # [SPECTRE-VL] rid -> SpectreMMPayload
        # [SPECTRE-VL] 两条通道无序：DRAFT_REQUEST 可能早于 mm payload，暂存重试。
        self._draft_reqs_waiting_mm: List[SpectreRequest] = []
        # [SPECTRE-VL] FINISH 后迟到的 payload 直接丢弃，避免泄漏。O(1) 查找，超 1024 淘汰最旧。
        self._finished_mm_rids: OrderedDict[str, None] = OrderedDict()
        # [SPECTRE-VL] rid -> 等 payload 的截止时间（墙钟，秒）。
        self._mm_wait_deadline: Dict[str, float] = {}
        # [SPECTRE-VL] rid -> 降级时刻。已降级的请求每步都回空 DRAFT 响应，
        # 让 Target 立刻放弃这一步而不是白等 SPECTRE_RECV_TIMEOUT_MS。
        self._mm_unavailable_rids: Dict[str, float] = {}

    def _get_draft_state(self, req_id: str) -> Optional[SpectreDraftState]:
        return self.draft_state_manager.get_state(req_id)

    def _set_draft_state(self, req_id: str, state: SpectreDraftState) -> None:
        self.draft_state_manager.set_state(req_id, state)

    def _delete_draft_state(self, req_id: str) -> bool:
        return self.draft_state_manager.delete(req_id)

    def _exists_draft_state(self, req_id: str) -> bool:
        return self.draft_state_manager.exists(req_id)

    @DynamicGradMode()
    def event_loop_normal_spectre_draft(self) -> None:
        self.last_batch = None
        self._init_draft_components()
        draft_priority: bool = self.server_args.spectre_draft_priority

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            # [SPECTRE-VL] 先收 mm 并预跑 ViT，再处理 DRAFT_REQUEST，避免占满 200ms recv 窗口。
            self._recv_and_store_mm_payloads()
            self._maybe_prewarm_pending_mm()
            self._retry_draft_reqs_waiting_mm()
            self.recv_and_process_draft_requests()
            saved_last_batch = self.last_batch
            if self.draft_waiting_queue:
                self._prefill_draft_reqs()
            self.last_batch = saved_last_batch

            if draft_priority:
                self._run_draft_priority_phase()
            else:
                self._merge_draft_into_running()
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
                self.draft_forward_cycle += 1
            else:
                self.self_check_during_idle()

            self.last_batch = batch

            if not draft_priority:
                self._extract_paused_drafts_from_running()

            if self.draft_forward_cycle % self.draft_cleanup_interval == 0:
                self._cleanup_stale_draft_states()

    def _run_draft_priority_phase(self) -> None:
        saved_last_batch = self.last_batch
        self._filter_draft_batch()

        if self.draft_batch.is_empty():
            self.last_batch = saved_last_batch
            return

        remaining_steps = max(
            (r.draft_tokens_target - (len(r.output_ids) - r.draft_generation_start_len))
            for r in self.draft_batch.reqs
            if not getattr(r, "draft_is_paused", False)
        )
        max_steps = self.server_args.spectre_max_draft_priority_steps
        if max_steps <= 0:
            max_steps = remaining_steps
        steps_taken = max(1, min(remaining_steps, max_steps))

        for _step in range(steps_taken):
            self._filter_draft_batch()
            if self.draft_batch.is_empty():
                break

            if not self.draft_batch.check_decode_mem():
                self._handle_draft_batch_oom()
                break

            self.draft_batch.prepare_for_decode()
            result = self.run_batch(self.draft_batch)
            self._process_draft_decode_result(self.draft_batch, result)

            self._update_draft_batch_after_decode()

        self.last_batch = saved_last_batch

    def _process_draft_decode_result(self, batch: ScheduleBatch, result) -> None:
        self.process_batch_result_decode(batch, result)
        self._filter_draft_batch()

    def _update_draft_batch_after_decode(self) -> None:
        self._filter_draft_batch()

    def _handle_draft_batch_oom(self) -> None:
        if self.tp_rank == 0:
            logger.warning("[Draft] Draft batch OOM, aborting all draft reqs in batch")
        for req in list(self.draft_batch.reqs):
            try:
                self._finish_draft_request(req.rid)
            except Exception as e:
                if self.tp_rank == 0:
                    logger.error(f"[Draft] Failed to finish {req.rid} during OOM: {e}")
        self.draft_batch = ScheduleBatch(reqs=[])

    def _filter_draft_batch(self) -> None:
        if self.draft_batch.is_empty():
            return

        keep_indices: List[int] = []
        for i, req in enumerate(self.draft_batch.reqs):
            if getattr(req, "draft_is_paused", False):
                if req not in self.draft_paused_reqs:
                    self.draft_paused_reqs.append(req)
            elif req.finished():
                pass
            else:
                keep_indices.append(i)

        if len(keep_indices) < len(self.draft_batch.reqs):
            if keep_indices:
                self.draft_batch.filter_batch(keep_indices=keep_indices)
            else:
                self.draft_batch = ScheduleBatch(reqs=[])

    def _merge_draft_into_running(self) -> None:
        self._filter_draft_batch()
        if self.draft_batch.is_empty():
            return

        if self.running_batch.is_empty():
            self.running_batch = self.draft_batch
        else:
            self.running_batch.merge_batch(self.draft_batch)

        self.draft_batch = ScheduleBatch(reqs=[])

    def _extract_paused_drafts_from_running(self) -> None:
        if self.running_batch.is_empty():
            return

        paused_indices: List[int] = []
        for i, req in enumerate(self.running_batch.reqs):
            if not getattr(req, "draft_is_paused", False):
                continue
            if req not in self.draft_paused_reqs:
                self.draft_paused_reqs.append(req)
            state = self._get_draft_state(req.rid)
            if state:
                state.location = DraftReqLocation.PAUSED
            paused_indices.append(i)

        if paused_indices:
            paused_set = set(paused_indices)
            keep = [
                i for i in range(len(self.running_batch.reqs)) if i not in paused_set
            ]
            self.running_batch.filter_batch(keep_indices=keep)

    def _draft_req_uses_precomputed_mm(self, req: Req) -> bool:
        """[SPECTRE-VL] True if a VL prefill of this req would take the precomputed path.

        mm_utils._get_precomputed_embedding cannot mix precomputed and raw-feature
        requests in one batch (NotImplementedError). Text-only / not-yet-prewarmed
        reqs return False so they can share a ViT fallback batch.
        """
        mm = getattr(req, "multimodal_inputs", None)
        if mm is None:
            return False
        found = False
        for item in getattr(mm, "mm_items", None) or []:
            is_image = getattr(item, "is_image", None)
            is_video = getattr(item, "is_video", None)
            if not (
                (callable(is_image) and is_image())
                or (callable(is_video) and is_video())
            ):
                continue
            if item.precomputed_embeddings is None:
                return False
            found = True
        return found

    def _partition_draft_prefill_queue(self, queue: List[Req]) -> List[List[Req]]:
        """[SPECTRE-VL] Split waiting reqs so one extend batch is embedding-homogeneous.

        Busy-loop ViT prewarm only does budget=1, so a later payload can attach
        without precomputed_embeddings while an earlier one already has them.
        """
        if len(queue) <= 1:
            return [list(queue)] if queue else []
        precomputed: List[Req] = []
        others: List[Req] = []
        for req in queue:
            if self._draft_req_uses_precomputed_mm(req):
                precomputed.append(req)
            else:
                others.append(req)
        if not precomputed or not others:
            return [list(queue)]
        if self._draft_req_uses_precomputed_mm(queue[0]):
            return [precomputed, others]
        return [others, precomputed]

    def _prefill_draft_reqs(self) -> None:
        if not self.draft_waiting_queue:
            return

        for req in self.draft_waiting_queue:
            req.init_next_round_input(self.tree_cache)

        original = list(self.draft_waiting_queue)
        admitted_ids = set()
        for group in self._partition_draft_prefill_queue(original):
            pending = [req for req in group if id(req) not in admitted_ids]
            if not pending:
                continue
            for req in self._prefill_admitted_group(pending):
                admitted_ids.add(id(req))
        self.draft_waiting_queue = [
            req for req in original if id(req) not in admitted_ids
        ]

    def _prefill_admitted_group(self, group: List[Req]) -> List[Req]:
        adder = self._build_prefill_adder_for_draft()
        for req in group:
            res = adder.add_one_req(
                req,
                has_chunked_req=False,
                truncation_align_size=getattr(self, "truncation_align_size", None),
            )
            if res != AddReqResult.CONTINUE:
                break

        admitted: List[Req] = adder.can_run_list
        if not admitted:
            return []

        draft_prefill_batch = ScheduleBatch.init_new(
            admitted,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        draft_prefill_batch.prepare_for_extend()
        result = self.run_batch(draft_prefill_batch)
        self._process_draft_prefill_result(draft_prefill_batch, result)

        draft_prefill_batch.filter_batch()

        if not draft_prefill_batch.is_empty():
            if self.draft_batch.is_empty():
                self.draft_batch = draft_prefill_batch
            else:
                self.draft_batch.merge_batch(draft_prefill_batch)

            for req in draft_prefill_batch.reqs:
                state = self._get_draft_state(req.rid)
                if state:
                    state.location = DraftReqLocation.DRAFT_BATCH
        return admitted

    def _process_draft_prefill_result(self, batch: ScheduleBatch, result) -> None:
        self.process_batch_result_prefill(batch, result)

    def _build_prefill_adder_for_draft(self) -> PrefillAdder:
        return PrefillAdder(
            page_size=self.page_size,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            running_batch=self.running_batch,
            new_token_ratio=self.new_token_ratio,
            rem_input_tokens=self.max_prefill_tokens,
            rem_chunk_tokens=None,
            mixed_with_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )

    def _add_req_to_draft_batch(self, req: Req) -> None:
        tmp_batch = self._build_decode_batch_from_reqs([req])
        if self.draft_batch.is_empty():
            self.draft_batch = tmp_batch
        else:
            self.draft_batch.merge_batch(tmp_batch)

    def _build_decode_batch_from_reqs(self, reqs: List[Req]) -> ScheduleBatch:
        try:
            device = self.tp_group.device
        except Exception:
            device = getattr(self, "device", None)
            if device is None:
                try:
                    device = self.tp_worker.model_runner.device
                except Exception:
                    from sglang.srt.utils import get_device

                    device = get_device()

        def _seq_len(r: Req) -> int:
            # prepare_for_decode writes the next slot at seq_lens then increments.
            # Use real KV occupancy so Case 3.1 catch-up tokens (appended to
            # output_ids without an alloc) do not skip a req_to_token index.
            committed = int(getattr(r, "kv_committed_len", 0) or 0)
            if committed > 0:
                return committed
            return max(0, len(r.origin_input_ids) + len(r.output_ids) - 1)

        seq_lens_list = [_seq_len(r) for r in reqs]

        try:
            from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator

            is_hybrid_swa = isinstance(
                self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
            )
        except ImportError:
            is_hybrid_swa = False

        batch = ScheduleBatch(
            reqs=list(reqs),
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            is_hybrid_swa=is_hybrid_swa,
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
        batch.has_stream = any(r.stream for r in reqs)
        batch.has_grammar = any(r.grammar for r in reqs)
        batch.return_hidden_states = any(r.return_hidden_states for r in reqs)
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )

        return batch

    def _spectre_mm_metrics(self):
        if not getattr(self, "current_scheduler_metrics_enabled", False):
            return None
        return getattr(self, "metrics_collector", None)

    def _refresh_pending_mm_bytes(self) -> None:
        metrics = self._spectre_mm_metrics()
        if metrics is None:
            return
        total = sum(payload_resident_bytes(p) for p in self._pending_mm.values())
        metrics.set_spectre_mm_pending_bytes(total)

    def _record_spectre_mm_degrade(self, reason: str) -> None:
        metrics = self._spectre_mm_metrics()
        if metrics is not None:
            metrics.increment_spectre_mm_degrades(reason=reason)

    def _mm_embed_error(self, req) -> Optional[str]:
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
                return (
                    "mm item has neither feature nor precomputed_embeddings"
                )
        return None

    def _recv_and_store_mm_payloads(self) -> None:
        # [SPECTRE-VL] 每个 Draft TP rank 从 PUB/SUB 直收；不再 broadcast 整包 pixel。
        receiver = getattr(self, "spectre_mm_receiver", None)
        payloads = []
        n_recv_errors = 0
        use_shm = receiver is not None and getattr(receiver, "use_shm", False) is True
        if receiver is not None:
            result = receiver.recv_all(
                defer_shm_materialize=(self.tp_size > 1 and use_shm)
            )
            if isinstance(result, tuple):
                payloads, n_recv_errors = result
            else:
                payloads = result or []
        metrics = self._spectre_mm_metrics()
        if metrics is not None and n_recv_errors:
            metrics.increment_spectre_mm_payloads_dropped(
                n_recv_errors, reason="recv_error"
            )
        if self.tp_size > 1 and use_shm:
            # [SPECTRE-VL] 先让每个 rank shm_open，barrier 后再 materialize/unlink。
            has = torch.tensor([1 if payloads else 0], dtype=torch.int32)
            torch.distributed.all_reduce(
                has, op=torch.distributed.ReduceOp.MAX, group=self.tp_cpu_group
            )
            if int(has.item()) > 0:
                torch.distributed.barrier(group=self.tp_cpu_group)
                for payload in payloads:
                    payload.to_multimodal_inputs()
        for payload in payloads:
            if payload.rid in self._finished_mm_rids:
                # [SPECTRE-VL] 真实 FINISH/ABORT 之后的迟到 payload 才丢；
                # 降级 / 孤儿回收不进这个集合，这样才能吃 HALF_OPEN 重发。
                release_mm_resources(payload.mm_inputs)
                if metrics is not None:
                    metrics.increment_spectre_mm_payloads_dropped(reason="finished_rid")
                continue
            old = self._pending_mm.get(payload.rid)
            if old is not None and old is not payload:
                if getattr(old, "attached", False):
                    # [SPECTRE-VL] 活请求已经持有 old.mm_inputs。重发 payload
                    # 若 release 旧对象，re-prefill 会变成 feature=None。
                    release_mm_resources(payload.mm_inputs)
                    if metrics is not None:
                        metrics.increment_spectre_mm_payloads_dropped(
                            reason="duplicate_attached"
                        )
                    continue
                release_mm_resources(old.mm_inputs)
            self._pending_mm[payload.rid] = payload
            # [SPECTRE-VL] 重发把粘性空响应解开，后续 DRAFT_REQUEST 可以重新 attach。
            self._clear_mm_degradation(payload.rid)
            if metrics is not None:
                metrics.increment_spectre_mm_payloads_received()
            if self.tp_rank == 0:
                logger.debug(
                    "[Draft][MM] stored payload rid=%s items=%d",
                    payload.rid,
                    len(payload.mm_items),
                )
        if payloads or n_recv_errors:
            self._refresh_pending_mm_bytes()

    def _retry_draft_reqs_waiting_mm(self) -> None:
        if not self._draft_reqs_waiting_mm:
            return
        still_waiting: List[SpectreRequest] = []
        for draft_req in self._draft_reqs_waiting_mm:
            if not self._create_new_draft_req(draft_req):
                still_waiting.append(draft_req)
        self._draft_reqs_waiting_mm = still_waiting

    def _maybe_prewarm_pending_mm(self) -> None:
        """[SPECTRE-VL] 把 ViT 移出 SPECTRE_RECV_TIMEOUT_MS 关键路径；失败则回落到 prefill 现跑。

        ViT 在 TP 下含 all-reduce，所以各 rank 必须预热同一批 payload、同样的顺序，
        并且提交与否也要一致 —— 否则下一轮的候选集就开始发散，最终挂死在集合通信里。
        """
        # _pending_mm 由各 rank 直收同一份 PUB 消息写入、由广播过的控制消息删除。
        # 空集时直接返回，避免在空闲路径上引入额外集合通信。
        if not self._pending_mm:
            return

        rids = self._decide_prewarm_rids()
        if not rids:
            return

        model = getattr(getattr(self, "tp_worker", None), "model_runner", None)
        model = getattr(model, "model", None) if model is not None else None

        groups: List[Tuple[str, List[Any], List[Any]]] = []
        all_images: List[Any] = []
        all_videos: List[Any] = []
        for rid in rids:
            pending = self._prewarmable_items(self._pending_mm.get(rid))
            images = [item for item in pending if item.is_image()]
            videos = [item for item in pending if item.is_video()]
            groups.append((rid, images, videos))
            all_images.extend(images)
            all_videos.extend(videos)

        results_per_rid: List[List[Tuple[Any, Any, Any, Any]]] = [[] for _ in groups]
        batch_ok = model is not None
        try:
            if batch_ok:
                with torch.no_grad():
                    img_out: List[Any] = []
                    vid_out: List[Any] = []
                    if all_images and hasattr(model, "get_image_feature"):
                        img_out = self._prewarm_items_with_fn(
                            model.get_image_feature, all_images
                        )
                    if all_videos and hasattr(model, "get_video_feature"):
                        vid_out = self._prewarm_items_with_fn(
                            model.get_video_feature, all_videos
                        )
                if all_images and len(img_out) != len(all_images):
                    batch_ok = False
                if all_videos and len(vid_out) != len(all_videos):
                    batch_ok = False
                if not img_out and not vid_out:
                    batch_ok = False
                if batch_ok:
                    img_i = 0
                    vid_i = 0
                    for gidx, (_rid, images, videos) in enumerate(groups):
                        n_img, n_vid = len(images), len(videos)
                        results_per_rid[gidx].extend(img_out[img_i : img_i + n_img])
                        results_per_rid[gidx].extend(vid_out[vid_i : vid_i + n_vid])
                        img_i += n_img
                        vid_i += n_vid
        except Exception as e:
            if self.tp_rank == 0:
                logger.warning(
                    "[Draft] ViT prewarm failed for rids=%s (fallback to prefill ViT): %s",
                    rids,
                    e,
                )
            batch_ok = False
            results_per_rid = [[] for _ in groups]

        failed = [0 if batch_ok else 1] * len(rids)
        failed = self._agree_on_prewarm_failures(failed)

        metrics = self._spectre_mm_metrics()
        for rid, results, is_failed in zip(rids, results_per_rid, failed):
            payload = self._pending_mm.get(rid)
            if payload is None:
                continue
            if is_failed:
                payload.prewarm_failed = True
                self._revert_prewarm_items_to_cpu(payload)
            else:
                self._commit_prewarm_results(results)
                payload.prewarmed = True
                if self.tp_rank == 0:
                    logger.debug("[Draft][MM] ViT prewarmed rid=%s", rid)
            if metrics is not None:
                metrics.increment_spectre_mm_prewarm(
                    result="failure" if is_failed else "success"
                )
        self._refresh_pending_mm_bytes()

    def _decide_prewarm_rids(self) -> List[str]:
        """[SPECTRE-VL] 本轮要预热的 rid 有序列表。TP>1 时由 rank0 拍板并广播。"""
        max_n = spectre_mm_prewarm_max()
        rids: List[str] = []
        if max_n > 0 and (self.tp_size == 1 or self.tp_rank == 0):
            busy = not self.draft_batch.is_empty()
            running = getattr(self, "running_batch", None)
            if running is not None and not running.is_empty():
                busy = True
            budget = 1 if busy else max_n
            cap = spectre_mm_prewarm_bytes()
            used = 0
            if cap > 0:
                used = sum(payload_gpu_bytes(p) for p in self._pending_mm.values())
            for rid, payload in self._pending_mm.items():
                if len(rids) >= budget:
                    break
                if cap > 0 and used >= cap:
                    # [SPECTRE-VL] 已超 GPU 水位：只存 CPU pixel，prefill 现跑 ViT。
                    break
                if payload.attached or payload.prewarm_failed or payload.prewarmed:
                    continue
                if not self._prewarmable_items(payload):
                    continue
                rids.append(rid)

        if self.tp_size > 1:
            rids = self._broadcast_list(rids)
        return rids

    def _broadcast_list(self, items: Optional[List]) -> List:
        """[SPECTRE-VL] 空 list 只广播长度，跳过 pickle 对象本体。"""
        if self.tp_size <= 1:
            return list(items or [])
        src = self.tp_group.ranks[0]
        n = len(items) if (self.tp_rank == 0 and items) else 0
        count = torch.tensor([n], dtype=torch.int32)
        torch.distributed.broadcast(count, src=src, group=self.tp_cpu_group)
        if int(count.item()) == 0:
            return []
        return broadcast_pyobj(
            items if items else [],
            self.tp_group.rank,
            self.tp_cpu_group,
            src=src,
        )

    def _agree_on_prewarm_failures(self, failed: List[int]) -> List[int]:
        """[SPECTRE-VL] 取失败的并集：只要有一个 rank 跑挂了，所有 rank 都必须放弃这个
        payload，否则部分 rank 有 precomputed_embeddings、部分没有，prefill 会形状不一致。"""
        if self.tp_size <= 1 or not failed:
            return failed
        flags = torch.tensor(failed, dtype=torch.int32)
        torch.distributed.all_reduce(
            flags, op=torch.distributed.ReduceOp.MAX, group=self.tp_cpu_group
        )
        return flags.tolist()

    def _prewarmable_items(self, payload) -> List[Any]:
        if payload is None:
            return []
        mm = payload.to_multimodal_inputs()
        return [
            item
            for item in mm.mm_items
            if item.precomputed_embeddings is None
            and item.feature is not None
            and (item.is_image() or item.is_video())
        ]

    def _prewarm_device(self):
        try:
            return self.tp_worker.model_runner.device
        except Exception:
            from sglang.srt.utils import get_device

            return getattr(self, "device", None) or get_device()

    def _prewarm_item_to_device(self, item, device) -> None:
        # [SPECTRE-VL] 只搬 feature，与正常 prefill 路径的 _move_items_to_device 保持一致。
        # model_specific_data（如 grid_thw）在 visual.forward 里就会被拉回 CPU/numpy，
        # 搬上 GPU 纯属来回拷贝。
        if isinstance(item.feature, torch.Tensor) and item.feature.device != device:
            item.feature = item.feature.to(device, non_blocking=True)

    def _revert_prewarm_items_to_cpu(self, payload) -> None:
        """[SPECTRE-VL] 预热失败后 feature 还留在 GPU，而 prewarm_failed 之后永不重试，
        这些像素张量会一直占着 Draft 显存直到 FINISH。搬回 CPU，prefill 会自己再搬上去。"""
        mm = getattr(payload, "mm_inputs", None)
        if mm is None:
            return
        for item in mm.mm_items:
            if isinstance(item.feature, torch.Tensor) and item.feature.device.type != "cpu":
                item.feature = item.feature.cpu()

    def _mm_item_token_count(self, item) -> Optional[int]:
        offsets = getattr(item, "offsets", None)
        if not offsets:
            return None
        total = 0
        for offset in offsets:
            try:
                start, end = offset[0], offset[1]
            except (TypeError, IndexError, KeyError):
                return None
            total += end - start + 1
        return total

    def _prewarm_items_with_fn(self, fn, items):
        """[SPECTRE-VL] 只收集 (item, pad_value, hash, emb)，不写回 item：
        TP 下要等所有 rank 都成功才能提交。"""
        device = self._prewarm_device()
        for item in items:
            self._prewarm_item_to_device(item, device)

        counts = [self._mm_item_token_count(item) for item in items]
        if len(items) > 1 and all(c is not None for c in counts):
            # 合并成一次调用，恢复 get_image_feature 内部 torch.cat 的批处理。
            emb = fn(items)
            if isinstance(emb, torch.Tensor) and emb.shape[0] == sum(counts):
                results = []
                start = 0
                for item, count in zip(items, counts):
                    results.append(
                        (item, item.pad_value, item.hash, emb[start : start + count])
                    )
                    start += count
                return results
            # 行数和 offsets 对不上说明两者语义不一致，退回逐 item 以保证正确性。
            if self.tp_rank == 0:
                logger.debug(
                    "[Draft][MM] batched prewarm row mismatch, falling back to per-item"
                )

        return [(item, item.pad_value, item.hash, fn([item])) for item in items]

    def _commit_prewarm_results(self, results) -> None:
        # [SPECTRE-VL] 全部成功后才改 format / 丢掉 feature，避免半成功导致 mixed embedding 报错。
        from sglang.srt.managers.schedule_batch import MultimodalInputFormat

        for item, pad_value, item_hash, emb in results:
            item.precomputed_embeddings = (
                emb.detach() if isinstance(emb, torch.Tensor) else emb
            )
            item.format = MultimodalInputFormat.PRECOMPUTED_EMBEDDING
            # [SPECTRE-VL] 必须保留 Target 传来的 pad_value；set_pad_value 在已有 pad_value 时早退。
            item.pad_value = pad_value
            item.hash = item_hash
            item.feature = None

    def _prewarm_payload_vit(self, payload) -> Tuple[bool, List]:
        """[SPECTRE-VL] 跑 ViT 但不提交结果，返回 (是否成功, 待提交结果)。

        get_image_feature/get_video_feature 的输出写进 precomputed_embeddings 后，
        prefill 只需要跑 LLM。
        """
        pending = self._prewarmable_items(payload)
        if not pending:
            return False, []
        model = getattr(getattr(self, "tp_worker", None), "model_runner", None)
        model = getattr(model, "model", None) if model is not None else None
        if model is None:
            return False, []
        try:
            with torch.no_grad():
                image_items = [i for i in pending if i.is_image()]
                video_items = [i for i in pending if i.is_video()]
                results = []
                if image_items and hasattr(model, "get_image_feature"):
                    results.extend(
                        self._prewarm_items_with_fn(
                            model.get_image_feature, image_items
                        )
                    )
                if video_items and hasattr(model, "get_video_feature"):
                    results.extend(
                        self._prewarm_items_with_fn(
                            model.get_video_feature, video_items
                        )
                    )
            if not results:
                return False, []
            return True, results
        except Exception as e:
            if self.tp_rank == 0:
                logger.warning(
                    "[Draft] ViT prewarm failed for %s (fallback to prefill ViT): %s",
                    getattr(payload, "rid", None),
                    e,
                )
            return False, []

    def _lookup_mm_payload(self, draft_req: SpectreRequest):
        mm_ref = getattr(draft_req, "mm_ref", None)
        if not mm_ref:
            return None
        return self._pending_mm.get(mm_ref) or self._pending_mm.get(
            draft_req.request_id
        )

    def _drop_waiting_mm_req(self, req_id: str) -> None:
        self._draft_reqs_waiting_mm = [
            r
            for r in self._draft_reqs_waiting_mm
            if getattr(r, "request_id", None) != req_id
        ]

    def _park_draft_req_waiting_mm(self, draft_req: SpectreRequest) -> None:
        req_id = draft_req.request_id
        if not any(
            getattr(x, "request_id", None) == req_id
            for x in self._draft_reqs_waiting_mm
        ):
            self._draft_reqs_waiting_mm.append(draft_req)

    def _mm_wait_expired(self, req_id: str) -> bool:
        """[SPECTRE-VL] 按墙钟预算判断是否还值得等 payload。

        以前是数事件循环轮数（<50），但 Draft 空闲时一轮只有几十微秒，
        50 轮可能才 1~2ms，远早于跨机 payload 到达就放弃了。
        """
        budget_ms = spectre_mm_wait_ms()
        if budget_ms <= 0:
            return True
        now = time.time()
        deadline = self._mm_wait_deadline.get(req_id)
        if deadline is None:
            self._mm_wait_deadline[req_id] = now + budget_ms / 1000.0
            return False
        return now >= deadline

    def _send_empty_draft_response(self, draft_req: SpectreRequest) -> None:
        """[SPECTRE-VL] 明确告诉 Target "这一步我没有 draft"。

        沉默的代价很高：Target 的 _collect_draft_messages 会一直等到
        SPECTRE_RECV_TIMEOUT_MS，整个 batch 每步白等 200ms。空响应则让 Target 把该 rid
        从 pending_rids 里 discard、record_success() 照常（熔断器不受伤），
        _apply_drafts_to_req 把 cur_drafts 置空，该请求当步退化成自回归。
        """
        if self.tp_size > 1 and self.tp_rank != 0:
            return
        communicator = getattr(self, "zmq_communicator", None)
        if communicator is None:
            return
        response = SpectreRequest(
            request_id=draft_req.request_id,
            spec_cnt=draft_req.spec_cnt,
            action=SpectreAction.DRAFT,
            spec_type=SpecType.DRAFT_RESPONSE,
            draft_token_ids=[],
            draft_logprobs=[],
            target_send_time=draft_req.target_send_time,
            draft_recv_time=draft_req.draft_recv_time,
        )
        communicator.send_objs([response])

    def _degrade_draft_req(self, draft_req: SpectreRequest, reason: str) -> None:
        """[SPECTRE-VL] 放弃为该请求投机，并把 rid 标记成粘性降级。

        粘性是必须的：后续的 DRAFT_REQUEST 在 spec_cnt>0 时不带 input_ids，
        走到 _process_draft_requests 会因为「无 state 且无 input_ids」被直接 continue，
        于是 Target 每一步都白等满 200ms。标记后每步都回空响应。
        """
        req_id = draft_req.request_id
        self._mm_unavailable_rids[req_id] = time.time()
        self._mm_wait_deadline.pop(req_id, None)
        self._drop_waiting_mm_req(req_id)
        if self._exists_draft_state(req_id):
            self._finish_draft_request(req_id, mark_finished=False)
        else:
            self._release_mm_for_rid(req_id, mark_finished=False)
        if self.tp_rank == 0:
            logger.warning(
                "[Draft] Degrading %s to autoregressive (no speculation): %s",
                req_id,
                reason,
            )
        self._send_empty_draft_response(draft_req)

    def _clear_mm_degradation(self, req_id: str) -> None:
        self._mm_unavailable_rids.pop(req_id, None)
        self._mm_wait_deadline.pop(req_id, None)

    def _first_token_mismatch(
        self, left: Optional[List[int]], right: Optional[List[int]]
    ) -> Optional[int]:
        if not left or not right:
            return None
        if len(left) != len(right):
            return min(len(left), len(right))
        for idx, (a, b) in enumerate(zip(left, right)):
            if a != b:
                return idx
        return None

    def _release_mm_for_rid(
        self,
        req_id: str,
        req: Optional[Req] = None,
        mark_finished: bool = True,
    ) -> None:
        # [SPECTRE-VL] FINISH/ABORT 才 mark_finished；降级 / 孤儿回收必须能吃 HALF_OPEN 重发。
        payload = self._pending_mm.pop(req_id, None)
        mm = None
        if req is not None:
            mm = req.multimodal_inputs
            req.multimodal_inputs = None
        if mm is None and payload is not None:
            mm = payload.mm_inputs
        release_mm_resources(mm)
        self._drop_waiting_mm_req(req_id)
        if mark_finished:
            self._mark_mm_finished(req_id)
        self._refresh_pending_mm_bytes()

    def _mark_mm_finished(self, req_id: str) -> None:
        self._finished_mm_rids.pop(req_id, None)
        self._finished_mm_rids[req_id] = None
        while len(self._finished_mm_rids) > 1024:
            self._finished_mm_rids.popitem(last=False)

    def recv_and_process_draft_requests(self) -> None:
        if self.tp_size == 1:
            if not hasattr(self, "zmq_communicator") or self.zmq_communicator is None:
                return

        if self._is_self_high_overhead_draft():
            if self.tp_size == 1 or self.tp_rank == 0:
                self._send_reject_message()
            return

        if self.tp_size == 1 or self.tp_rank == 0:
            messages = self._recv_draft_requests()
            if messages:
                logger.debug(
                    f"\033[32m [Draft][Recv] {len(messages)} messages from Target \033[0m"
                )
        else:
            messages = None

        if self.tp_size > 1:
            messages = self._broadcast_list(messages)

        if not messages:
            return

        control_msgs, latest_msgs = self.deduplicate_draft_requests(messages)

        if not control_msgs and not latest_msgs:
            return

        self.token_to_kv_pool_allocator.free_group_begin()
        self._process_control_message(control_msgs)
        self._process_draft_requests(latest_msgs)
        self.token_to_kv_pool_allocator.free_group_end()
        self._flush_draft_batch_pending_adds()

    def _recv_draft_requests(self) -> List[SpectreRequest]:
        try:
            msgs: List[SpectreRequest] = []
            if hasattr(self, "zmq_communicator") and self.zmq_communicator is not None:
                msgs = self.zmq_communicator.recv_all_objs()
                if msgs:
                    more = self.zmq_communicator.recv_all_objs()
                    if more:
                        msgs.extend(more)
            return msgs
        except (ConnectionError, OSError) as e:
            if self.tp_rank == 0:
                logger.error(f"[Draft] Network error in recv: {e}", exc_info=True)
            return []
        except Exception as e:
            if self.tp_rank == 0:
                logger.error(f"[Draft] Unexpected recv error: {e}", exc_info=True)
            return []

    def deduplicate_draft_requests(
        self,
        messages: List[SpectreRequest],
    ) -> Tuple[List[SpectreRequest], Dict[str, SpectreRequest]]:
        latest_msgs: Dict[str, SpectreRequest] = {}
        control_msgs: List[SpectreRequest] = []

        for draft_req in messages:
            req_id = draft_req.request_id
            action = getattr(draft_req, "action", SpectreAction.DRAFT)

            if action in (SpectreAction.FINISH, SpectreAction.ABORT):
                control_msgs.append(draft_req)
                continue

            if (
                req_id not in latest_msgs
                or draft_req.spec_cnt > latest_msgs[req_id].spec_cnt
            ):
                if req_id in latest_msgs and draft_req.input_ids is None:
                    draft_req.input_ids = latest_msgs[req_id].input_ids
                    draft_req.sampling_params = (
                        draft_req.sampling_params or latest_msgs[req_id].sampling_params
                    )
                latest_msgs[req_id] = draft_req

        total = len(messages)
        kept = len(latest_msgs) + len(control_msgs)
        if total > kept and self.tp_rank == 0:
            logger.debug(f"\033[36m [Draft][Dedup] {total} → {kept} \033[0m")

        return control_msgs, latest_msgs

    def _process_control_message(self, control_msgs: List[SpectreRequest]) -> None:
        for draft_req in control_msgs:
            action = draft_req.action
            if action in (SpectreAction.FINISH, SpectreAction.ABORT):
                if self.tp_rank == 0:
                    logger.debug(
                        f"[Draft] Received {action} for {draft_req.request_id}"
                    )
                # [SPECTRE-VL] 请求真正结束，解除降级标记，rid 复用时不会被误判。
                self._clear_mm_degradation(draft_req.request_id)
                self._finish_draft_request(draft_req.request_id)

    def _process_draft_requests(self, latest_msgs: Dict[str, SpectreRequest]) -> None:
        for req_id, draft_req in latest_msgs.items():
            try:
                # [SPECTRE-VL] 已降级的 rid：每一步都要回空响应，否则 Target 白等 200ms。
                if req_id in self._mm_unavailable_rids:
                    self._send_empty_draft_response(draft_req)
                    continue

                state = self._get_draft_state(req_id)

                if state is None:
                    if draft_req.input_ids is None:
                        if self.tp_rank == 0:
                            logger.warning(
                                f"[Draft] {req_id}: no state and no input_ids, skipping"
                            )
                        continue
                    self._create_new_draft_req(draft_req)
                    continue

                state.last_updated_time = time.time()
                req = state.req_object

                if req is None:
                    if self.tp_rank == 0:
                        logger.warning(f"[Draft] {req_id} has None req_object")
                    self._finish_draft_request(req_id)
                    continue

                req.target_send_time = draft_req.target_send_time
                req.draft_recv_time = draft_req.draft_recv_time

                if draft_req.input_ids is not None:
                    target_input = draft_req.input_ids
                    state.target_origin_input_ids = list(target_input)
                else:
                    target_input = state.target_origin_input_ids or []

                target_fill_ids: List[int] = (
                    target_input
                    + (draft_req.output_ids or [])
                    + (draft_req.draft_token_ids or [])
                )
                draft_fill_ids: List[int] = (req.origin_input_ids or []) + (
                    req.output_ids or []
                )

                if not target_fill_ids:
                    continue

                skip = len(target_input)
                if (
                    skip > 0
                    and len(draft_fill_ids) >= skip
                    and len(target_fill_ids) >= skip
                ):
                    is_identical, fork_offset = self._find_fork_point(
                        draft_fill_ids[skip:], target_fill_ids[skip:]
                    )
                    fork_point = skip + fork_offset
                else:
                    is_identical, fork_point = self._find_fork_point(
                        draft_fill_ids, target_fill_ids
                    )

                if is_identical:
                    self._handle_identical_tokens(req, draft_req, state)
                else:
                    self._handle_divergence(
                        req, target_fill_ids, fork_point, draft_req, state
                    )

            except Exception as e:
                if self.tp_rank == 0:
                    logger.error(
                        f"\033[31m [Draft] Error processing {req_id}: {e} \033[0m",
                        exc_info=True,
                    )
                try:
                    self._finish_draft_request(req_id)
                except Exception:
                    pass

    def _find_fork_point(
        self,
        draft_ids: List[int],
        target_ids: List[int],
    ) -> Tuple[bool, int]:
        min_len = min(len(draft_ids), len(target_ids))
        for i in range(min_len):
            if draft_ids[i] != target_ids[i]:
                return (False, i)
        return (len(draft_ids) == len(target_ids), min_len)

    def _handle_identical_tokens(
        self,
        req: Req,
        draft_req: SpectreRequest,
        state: SpectreDraftState,
    ) -> None:
        if self.tp_rank == 0:
            logger.debug(
                f"\033[34m [Draft][NoChange] {req.rid}, "
                f"spec_cnt={draft_req.spec_cnt}, location={state.location} \033[0m"
            )
        # Same prefix, possibly a later spec_cnt (Target recv timeout retry).
        # Keep the in-flight draft window; do not reset start_len.
        req.spec_cnt = draft_req.spec_cnt
        if draft_req.num_draft_tokens:
            req.draft_tokens_target = draft_req.num_draft_tokens
        req.len_output_ids = len(req.output_ids)
        state.last_updated_time = time.time()

        if req.draft_is_paused:
            # Already produced drafts for this prefix; resend under the new spec_cnt.
            self._send_draft_response(req)
            return

        self._resume_or_update(req, state, tokens_changed=False)

    def _handle_divergence(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        draft_req: SpectreRequest,
        state: SpectreDraftState,
    ) -> None:
        state.last_updated_time = time.time()

        current_len = len(req.origin_input_ids) + len(req.output_ids)
        target_len = len(target_fill_ids)
        allocated = int(getattr(req, "kv_allocated_len", 0) or 0)
        token_kv_len = max(0, current_len - 1)
        current_kv_len = allocated if allocated > 0 else token_kv_len
        needs_kv_release = fork_point < current_kv_len

        if current_len == target_len:
            self._handle_equal_length(
                req,
                target_fill_ids,
                fork_point,
                current_len,
                current_kv_len,
                needs_kv_release,
                draft_req,
                state,
            )
        elif current_len > target_len:
            self._handle_draft_ahead(
                req,
                target_fill_ids,
                fork_point,
                current_kv_len,
                needs_kv_release,
                draft_req,
                state,
            )
        else:
            self._handle_target_ahead(
                req,
                target_fill_ids,
                fork_point,
                current_len,
                current_kv_len,
                needs_kv_release,
                draft_req,
                state,
            )

    def _handle_equal_length(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_len: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: SpectreRequest,
        state: SpectreDraftState,
    ) -> None:
        if fork_point == current_len - 1:
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[36m [Case 1.1] {req.rid=}, {draft_req.spec_cnt=}, "
                    f"replace last token \033[0m"
                )
            self._update_tokens(req, fork_point, target_fill_ids[fork_point:])
            self._update_req_state(req, draft_req, state)
            self._resume_or_update(req, state)
        else:
            self._handle_multi_token_divergence(
                req,
                target_fill_ids,
                fork_point,
                current_kv_len,
                needs_kv_release,
                draft_req,
                state,
                "1.2",
            )

    def _handle_draft_ahead(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: SpectreRequest,
        state: SpectreDraftState,
    ) -> None:
        target_len = len(target_fill_ids)

        if fork_point == target_len:
            target_output_len = target_len - len(req.origin_input_ids)
            draft_output_len = len(req.output_ids)
            tokens_ahead = draft_output_len - target_output_len

            if self.tp_rank == 0:
                logger.debug(
                    f"\033[36m [Case 2.1] {req.rid=}, {draft_req.spec_cnt=}, "
                    f"draft ahead by {tokens_ahead} token(s) \033[0m"
                )

            req.draft_generation_start_len = target_output_len
            req.spec_cnt = draft_req.spec_cnt
            if draft_req.num_draft_tokens:
                req.draft_tokens_target = draft_req.num_draft_tokens
            req.len_output_ids = draft_output_len
            state.last_updated_time = time.time()

            target = req.draft_tokens_target or (
                self.server_args.speculative_num_draft_tokens or 1
            )
            if tokens_ahead >= target:
                if self.tp_rank == 0:
                    logger.debug(
                        f"\033[33m [Case 2.1] {req.rid=}: already {tokens_ahead} ahead, "
                        f"sending immediately \033[0m"
                    )
                self._send_draft_response(req)
                self._pause_req(req, state)
            else:
                req.draft_is_paused = False
                self._resume_or_update(req, state)
        else:
            self._handle_multi_token_divergence(
                req,
                target_fill_ids,
                fork_point,
                current_kv_len,
                needs_kv_release,
                draft_req,
                state,
                "2.2",
            )

    def _handle_target_ahead(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_len: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: SpectreRequest,
        state: SpectreDraftState,
    ) -> None:
        if fork_point == current_len:
            delta_tokens = target_fill_ids[fork_point:]
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[36m [Case 3.1] {req.rid=}, {draft_req.spec_cnt=}, "
                    f"append {len(delta_tokens)} token(s) without full re-prefill \033[0m"
                )
            self._update_tokens(req, fork_point, delta_tokens)
            self._update_req_state(req, draft_req, state)
            # +1 catch-up token becomes the next decode query. KV for it is
            # allocated in prepare_for_decode as long as seq_lens comes from
            # kv_committed_len (see _build_decode_batch_from_reqs). Longer
            # gaps still re-prefill.
            if req.req_pool_idx is not None and len(delta_tokens) <= 1:
                self._resume_or_update(req, state)
            else:
                self._prepare_for_reprefill(req, target_fill_ids, draft_req, state)
            return

        if self.tp_rank == 0:
            logger.debug(
                f"\033[36m [Case 3.2] {req.rid=}, {draft_req.spec_cnt=}, "
                f"re-prefill for extend+rollback \033[0m"
            )
        self._prepare_for_reprefill(req, target_fill_ids, draft_req, state)

    def _handle_multi_token_divergence(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: SpectreRequest,
        state: SpectreDraftState,
        case_name: str,
    ) -> None:
        new_len = len(target_fill_ids)
        can_decode_after_rollback = new_len - 1 <= fork_point

        if can_decode_after_rollback and self.draft_kv_manager.can_local_rollback(
            req, fork_point
        ):
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[35m [Case {case_name}] {req.rid=}, {draft_req.spec_cnt=} "
                    f"→ local rollback + decode \033[0m"
                )
            if needs_kv_release:
                self.draft_kv_manager.local_rollback(req, fork_point, current_kv_len)
            self._update_tokens(req, fork_point, target_fill_ids[fork_point:])
            self._update_req_state(req, draft_req, state)
            self._resume_or_update(req, state)
        else:
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[36m [Case {case_name}] {req.rid=}, {draft_req.spec_cnt=} "
                    f"→ re-prefill \033[0m"
                )
            self._prepare_for_reprefill(req, target_fill_ids, draft_req, state)

    def _update_tokens(
        self, req: Req, fork_point: int, delta_tokens: List[int]
    ) -> None:
        truncate_point = fork_point - len(req.origin_input_ids)
        req.output_ids = req.output_ids[: max(0, truncate_point)]
        req.output_ids.extend(delta_tokens)
        req.fill_ids = req.origin_input_ids + req.output_ids

    def _update_req_state(
        self,
        req: Req,
        draft_req: SpectreRequest,
        state: SpectreDraftState,
        reset_generation_start: bool = True,
    ) -> None:
        req.spec_cnt = draft_req.spec_cnt
        if draft_req.num_draft_tokens:
            req.draft_tokens_target = draft_req.num_draft_tokens
        # After a new committed prefix (replace last token, rollback, etc.)
        # start a fresh draft round from the current output_ids length.
        if reset_generation_start:
            req.draft_generation_start_len = len(req.output_ids)
        req.draft_is_paused = False
        req.len_output_ids = len(req.output_ids)
        state.last_updated_time = time.time()

    def _resume_or_update(
        self,
        req: Req,
        state: SpectreDraftState,
        tokens_changed: bool = True,
    ) -> None:
        location = state.location

        if location == DraftReqLocation.PAUSED:
            self._resume_draft_req(req, state)
        elif location == DraftReqLocation.DRAFT_BATCH:
            if tokens_changed:
                self._rebuild_req_in_draft_batch(req, state)

    def _resume_draft_req(self, req: Req, state: SpectreDraftState) -> None:
        if req in self.draft_paused_reqs:
            self.draft_paused_reqs.remove(req)

        if req.req_pool_idx is None:
            if self.tp_rank == 0:
                logger.warning(
                    f"[Draft][Resume] {req.rid} has no KV pool slot, "
                    f"falling back to re-prefill"
                )
            req.draft_is_paused = False
            state.location = DraftReqLocation.DRAFT_WAITING
            if req not in self.draft_waiting_queue:
                self.draft_waiting_queue.append(req)
            return

        req.draft_is_paused = False
        state.location = DraftReqLocation.DRAFT_BATCH
        if req not in self._draft_batch_pending_adds:
            self._draft_batch_pending_adds.append(req)

        if self.tp_rank == 0:
            logger.debug(f"[Draft][Resume] {req.rid} → draft_batch (pending)")

    def _rebuild_req_in_draft_batch(self, req: Req, state: SpectreDraftState) -> None:
        self._filter_req_from_batch(self.draft_batch, req)
        self._filter_req_from_batch(getattr(self, "running_batch", None), req)

        if req.req_pool_idx is not None:
            req.draft_is_paused = False
            state.location = DraftReqLocation.DRAFT_BATCH
            if req not in self._draft_batch_pending_adds:
                self._draft_batch_pending_adds.append(req)
        else:
            state.location = DraftReqLocation.DRAFT_WAITING
            if req not in self.draft_waiting_queue:
                self.draft_waiting_queue.append(req)

    def _flush_draft_batch_pending_adds(self) -> None:
        if not self._draft_batch_pending_adds:
            return

        new_batch = self._build_decode_batch_from_reqs(self._draft_batch_pending_adds)
        self._draft_batch_pending_adds.clear()

        if self.draft_batch.is_empty():
            self.draft_batch = new_batch
        else:
            self.draft_batch.merge_batch(new_batch)

    def _pause_req(self, req: Req, state: SpectreDraftState) -> None:
        self._filter_req_from_batch(self.draft_batch, req)
        self._filter_req_from_batch(getattr(self, "running_batch", None), req)

        req.draft_is_paused = True
        if req not in self.draft_paused_reqs:
            self.draft_paused_reqs.append(req)
        state.location = DraftReqLocation.PAUSED
        state.last_updated_time = time.time()

    def _reset_req_logprob_fields(self, req: Req) -> None:
        req.input_token_logprobs_val = None
        req.input_token_logprobs_idx = None
        req.input_top_logprobs_val = None
        req.input_top_logprobs_idx = None
        req.input_token_ids_logprobs_val = None
        req.input_token_ids_logprobs_idx = None
        req.input_token_logprobs = None
        req.temp_input_top_logprobs_val = None
        req.temp_input_top_logprobs_idx = None
        req.temp_input_token_ids_logprobs_val = None
        req.temp_input_token_ids_logprobs_idx = None
        req.input_logprob_sent = False

        req.output_token_logprobs_val = []
        req.output_token_logprobs_idx = []
        req.output_top_logprobs_val = []
        req.output_top_logprobs_idx = []
        req.output_token_ids_logprobs_val = []
        req.output_token_ids_logprobs_idx = []

    def _filter_req_from_batch(self, batch: Optional[ScheduleBatch], req: Req) -> None:
        if batch is None or batch.is_empty():
            return
        keep = [i for i, r in enumerate(batch.reqs) if r is not req]
        if len(keep) == len(batch.reqs):
            return
        batch.filter_batch(keep_indices=keep)

    def _remove_draft_req(self, req: Req) -> None:
        if req in self.draft_paused_reqs:
            self.draft_paused_reqs.remove(req)

        self._filter_req_from_batch(self.draft_batch, req)
        self._filter_req_from_batch(getattr(self, "running_batch", None), req)

        if req in self.draft_waiting_queue:
            self.draft_waiting_queue.remove(req)
        if req in self._draft_batch_pending_adds:
            self._draft_batch_pending_adds.remove(req)

    def _prepare_for_reprefill(
        self,
        req: Req,
        target_fill_ids: List[int],
        draft_req: SpectreRequest,
        state: SpectreDraftState,
    ) -> None:
        if self.tp_rank == 0:
            logger.debug(
                f"[Draft][RePrefill] {req.rid=}, {draft_req.spec_cnt=}, "
                f"new_len={len(target_fill_ids)}"
            )

        self._remove_draft_req(req)
        if req.req_pool_idx is not None:
            self.draft_kv_manager.release_all_kv_for_reprefill_req(req)

        req.fill_ids = target_fill_ids
        req.origin_input_ids = list(target_fill_ids)
        req.output_ids = []
        req.prefix_indices = []
        req.extend_input_len = len(req.fill_ids)

        req.spec_cnt = draft_req.spec_cnt
        req.draft_tokens_target = draft_req.num_draft_tokens
        req.draft_generation_start_len = 0
        req.draft_is_paused = False
        req.len_output_ids = 0

        req.last_node = None
        req.kv_committed_len = 0
        req.kv_committed_freed = False
        req.kv_overallocated_freed = False

        state.location = DraftReqLocation.DRAFT_WAITING
        req.logprob_start_len = len(req.origin_input_ids) - 1
        self._reset_req_logprob_fields(req)

        if req.multimodal_inputs is not None:
            # [SPECTRE-VL] origin_input_ids 被整体重写后长度变了，必须清空并重算 M-RoPE。
            reset_mm_mrope(req.multimodal_inputs)
            try:
                self._maybe_compute_mrope_positions(req)
            except Exception as e:
                self._degrade_draft_req(
                    draft_req, f"compute_mrope_positions failed: {e}"
                )
                self._record_spectre_mm_degrade("mrope_failed")
                return
            embed_err = self._mm_embed_error(req)
            if embed_err:
                self._degrade_draft_req(draft_req, embed_err)
                self._record_spectre_mm_degrade("mm_tensors_missing")
                return

        if req not in self.draft_waiting_queue:
            self.draft_waiting_queue.append(req)

    def _check_and_pause_draft_req(self, req: Req) -> bool:
        if getattr(req, "spec_type", None) != SpecType.DRAFT_REQUEST:
            return False

        if req.draft_is_paused:
            return True

        target = req.draft_tokens_target
        if not target:
            target = self.server_args.speculative_num_draft_tokens or 1
            req.draft_tokens_target = target

        tokens_generated = len(req.output_ids) - req.draft_generation_start_len

        if tokens_generated >= target:
            self._send_draft_response(req)

            req.draft_is_paused = True
            if req not in self.draft_paused_reqs:
                self.draft_paused_reqs.append(req)

            state = self._get_draft_state(req.rid)
            if state:
                state.location = DraftReqLocation.PAUSED
                state.last_updated_time = time.time()

            return True

        return False

    def _send_draft_response(self, req: Req) -> None:
        draft_tokens = req.output_ids[req.draft_generation_start_len :]

        draft_logits: List[float] = []
        if hasattr(req, "output_token_logprobs_val") and req.output_token_logprobs_val:
            start = req.draft_generation_start_len
            end = start + len(draft_tokens)
            if len(req.output_token_logprobs_val) >= end:
                draft_logits = req.output_token_logprobs_val[start:end]

        response = SpectreRequest(
            request_id=req.rid,
            spec_cnt=req.spec_cnt,
            action=SpectreAction.DRAFT,
            spec_type=SpecType.DRAFT_RESPONSE,
            draft_token_ids=draft_tokens,
            draft_logprobs=draft_logits or [],
            target_send_time=req.target_send_time,
            draft_recv_time=req.draft_recv_time,
        )

        if self.tp_size == 1 or self.tp_rank == 0:
            if hasattr(self, "zmq_communicator") and self.zmq_communicator is not None:
                self.zmq_communicator.send_objs([response])
                logger.info(
                    f"\033[32m [Draft][Send] rid={req.rid} spec_cnt={req.spec_cnt} "
                    f"n={len(draft_tokens)} \033[0m"
                )

        # Keep start_len so a Target timeout retry (NoChange / same prefix)
        # can resend the same draft window instead of an empty slice.

    def _create_new_draft_req(self, draft_req: SpectreRequest) -> bool:
        req_id = draft_req.request_id

        mm_payload = self._lookup_mm_payload(draft_req)
        mm_ref = getattr(draft_req, "mm_ref", None)
        if mm_ref and mm_payload is None:
            # [SPECTRE-VL] mm 旁路与 C++ DRAFT_REQUEST 无顺序保证，payload 可能还在路上。
            # 注意这里不要补收 payload：本函数在 per-request 循环里被调用，
            # SHM 路径的 barrier 不能按请求次数发散。下一轮 event loop 会再收。
            if self._mm_wait_expired(req_id):
                self._degrade_draft_req(
                    draft_req,
                    f"mm payload did not arrive within {spectre_mm_wait_ms():.0f}ms",
                )
                self._record_spectre_mm_degrade("wait_timeout")
                return True
            self._park_draft_req_waiting_mm(draft_req)
            if self.tp_rank == 0:
                logger.debug("[Draft] waiting for mm payload rid=%s", req_id)
            return False

        input_ids: List[int] = (
            (draft_req.input_ids or [])
            + (draft_req.output_ids or [])
            + (draft_req.draft_token_ids or [])
        )

        # [SPECTRE-VL] Target pad 后的序列可能超过 Draft 的窗口。Target 在
        # handle_generate_request 里有对应的 abort，Draft 这边没有，硬建 req 会在 prefill 里崩。
        max_input_len = getattr(self, "max_req_input_len", None)
        if max_input_len is not None and len(input_ids) >= max_input_len:
            self._degrade_draft_req(
                draft_req,
                f"input length {len(input_ids)} >= draft max_req_input_len "
                f"{max_input_len}; increase the draft --context-length",
            )
            self._record_spectre_mm_degrade("oversized")
            return True

        # [SPECTRE-VL] Draft 靠扫 pad_value 定位视觉 token，两侧序列必须逐 token 相同。
        # 不一致（例如 session 请求被 adjust_mm_offsets 改过）时 embedding 行数会和
        # 占位符数量对不上，轻则 shape 报错，重则静默错位，所以宁可不投机。
        if mm_payload is not None:
            mismatch = self._first_token_mismatch(
                draft_req.input_ids, mm_payload.padded_input_ids
            )
            if mismatch is not None:
                self._degrade_draft_req(
                    draft_req,
                    f"padded_input_ids mismatch at index {mismatch} "
                    f"(draft_req={len(draft_req.input_ids)} "
                    f"payload={len(mm_payload.padded_input_ids)})",
                )
                self._record_spectre_mm_degrade("padded_mismatch")
                return True

        if self._exists_draft_state(req_id):
            # [SPECTRE-VL] 重建请求时不要释放 mm 缓存，后面还要 attach 同一份 embedding。
            self._finish_draft_request(req_id, release_mm=False)

        if draft_req.sampling_params is None:
            from sglang.srt.sampling.sampling_params import SamplingParams

            sampling_params = SamplingParams()
        else:
            sampling_params = draft_req.sampling_params

        if hasattr(sampling_params, "normalize"):
            try:
                sampling_params.normalize(self.tokenizer)
            except Exception as e:
                if self.tp_rank == 0:
                    logger.warning(
                        f"[Draft] Failed to normalize SamplingParams for {req_id}: {e}, "
                        f"applying manual fix"
                    )
                _fix_sampling_params_stop_strs(sampling_params)
        else:
            _fix_sampling_params_stop_strs(sampling_params)

        req = Req(
            rid=req_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=True,
            top_logprobs_num=1,
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

        req.spec_cnt = draft_req.spec_cnt
        req.spec_type = SpecType.DRAFT_REQUEST
        req.tokenizer = self.tokenizer
        req.logprob_start_len = len(req.origin_input_ids) - 1
        req.draft_tokens_target = draft_req.num_draft_tokens
        req.draft_generation_start_len = 0
        req.draft_is_paused = False
        req.len_output_ids = 0
        req.target_send_time = draft_req.target_send_time
        req.draft_recv_time = draft_req.draft_recv_time
        # [SPECTRE-VL] 与 origin_input_ids 同为 Target 已 pad 序列；不再 pad。
        req.origin_input_ids_unpadded = list(req.origin_input_ids)

        if mm_payload is not None:
            # [SPECTRE-VL] 复用 Target 已 pad 的 token + pad_value，Draft 绝不能再调
            # pad_input_ids_func。一致性已在上面校验过。
            mm = mm_payload.to_multimodal_inputs()
            req.extend_image_inputs(mm)
            mm_payload.attached = True
            try:
                self._maybe_compute_mrope_positions(req)
            except Exception as e:
                # tcp multipart 反序列化失败时 grid_thw 会是 dict，不能让 scheduler 自杀。
                self._degrade_draft_req(
                    draft_req, f"compute_mrope_positions failed: {e}"
                )
                self._record_spectre_mm_degrade("mrope_failed")
                return True
            embed_err = self._mm_embed_error(req)
            if embed_err:
                self._degrade_draft_req(draft_req, embed_err)
                self._record_spectre_mm_degrade("mm_tensors_missing")
                return True

        self.draft_waiting_queue.append(req)

        self._set_draft_state(
            req_id,
            SpectreDraftState(
                req_id=req_id,
                spec_cnt=draft_req.spec_cnt,
                req_object=req,
                location=DraftReqLocation.DRAFT_WAITING,
                target_origin_input_ids=(
                    list(draft_req.input_ids) if draft_req.input_ids else []
                ),
                last_prefix_length=len(input_ids),
                last_output_length=0,
                mm_ref=mm_ref,
            ),
        )

        if self.tp_rank == 0:
            logger.debug(
                f"[Draft][New] {req_id=}, {req.spec_cnt=}, input_len={len(input_ids)}"
            )
        self._drop_waiting_mm_req(req_id)
        self._mm_wait_deadline.pop(req_id, None)
        return True

    def _finish_draft_request(
        self, req_id: str, release_mm: bool = True, mark_finished: bool = True
    ) -> None:
        # [SPECTRE-VL] DRAFT_REQUEST 永不自然 finish，必须在此显式释放 feature / 预热 embedding。
        state = self._get_draft_state(req_id)
        if state is None:
            if release_mm:
                self._release_mm_for_rid(req_id, mark_finished=mark_finished)
            return

        req = state.req_object
        self._remove_draft_req(req)

        if not req.finished():
            req.to_abort = True
            req.finished_reason = FINISH_ABORT("Target request finished")

        if req.req_pool_idx is not None and not getattr(
            req, "kv_committed_freed", False
        ):
            self.draft_kv_manager.release_all_kv_for_finished_req(req)

        if release_mm:
            self._release_mm_for_rid(req_id, req, mark_finished=mark_finished)
        elif req is not None:
            req.multimodal_inputs = None

        self._delete_draft_state(req_id)
        if self.tp_rank == 0:
            logger.debug(f"[Draft][Finish] {req_id=}")

    def _cleanup_stale_draft_states(self) -> None:
        for req_id in self.draft_state_manager.cleanup_stale_states():
            try:
                # [SPECTRE-VL] 超时不等于 Target FINISH；不进 _finished_mm_rids，HALF_OPEN 重发才能恢复。
                self._finish_draft_request(req_id, mark_finished=False)
            except Exception as e:
                if self.tp_rank == 0:
                    logger.warning(f"[Draft] Cleanup failed for {req_id=}: {e}")
            finally:
                try:
                    self._delete_draft_state(req_id)
                except Exception:
                    pass
        stale_s = spectre_mm_stale_s()
        now = time.time()
        # [SPECTRE-VL] 无 draft state 的孤儿 payload（例如 Target 已 abort 但 FINISH 丢失）短超时回收。
        for rid, payload in list(self._pending_mm.items()):
            if self._exists_draft_state(rid):
                continue
            if now - getattr(payload, "created_at", now) > stale_s:
                self._release_mm_for_rid(rid, mark_finished=False)

        # [SPECTRE-VL] 降级标记正常由 FINISH/ABORT 清除；控制消息丢失时在这里兜底。
        timeout = getattr(self.draft_state_manager, "timeout_threshold", 60.0)
        for rid, marked_at in list(self._mm_unavailable_rids.items()):
            if now - marked_at > timeout:
                self._clear_mm_degradation(rid)
        for rid, deadline in list(self._mm_wait_deadline.items()):
            if now - deadline > timeout:
                self._mm_wait_deadline.pop(rid, None)

    def _is_self_high_overhead_draft(self) -> bool:
        if not hasattr(self, "running_batch") or self.running_batch is None:
            return False
        return self.running_batch.batch_size() > self.server_args.spectre_max_batch_size

    def _send_reject_message(self) -> None:
        if self.tp_size > 1 and self.tp_rank != 0:
            return
        if not hasattr(self, "zmq_communicator") or self.zmq_communicator is None:
            return

        reject_msg = SpectreRequest(
            request_id="system",
            spec_cnt=0,
            action=SpectreAction.REJECT,
            spec_type=SpecType.DRAFT_RESPONSE,
            draft_token_ids=[],
            draft_logprobs=[],
        )
        self.zmq_communicator.send_objs([reject_msg])
        if self.tp_rank == 0:
            logger.debug("[Draft] Sent REJECT to Target (high load)")

    def get_num_allocatable_reqs(self, running_bs: int) -> int:
        paused_id_set: set = set()
        paused_reqs_lock = getattr(self, "paused_reqs_lock", None)
        if paused_reqs_lock is not None and hasattr(self, "paused_reqs"):
            try:
                with paused_reqs_lock:
                    paused_id_set.update(id(r) for r in self.paused_reqs)
            except Exception:
                pass
        if hasattr(self, "draft_paused_reqs"):
            paused_id_set.update(id(r) for r in self.draft_paused_reqs)

        paused_count = len(paused_id_set)
        total_occupied = running_bs + paused_count

        res = self.server_args.pp_max_micro_batch_size - total_occupied
        if self.pp_size > 1:
            res = min(res, self.req_to_token_pool.available_size())
        return res
