"""STANDALONE-style top-k tree expansion on the remote Draft process.

Uses the Draft server's existing TpModelWorker (no second weight load).
Linear KV stays the Target committed prefix. Tree KV is allocated on exclusive
pages and may be leased for the next STEP when topk>1 and page_size>1.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.common import alloc_paged_token_slots_extend, alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_utils import organize_draft_results
from sglang.srt.speculative.spec_utils import (
    NpuGraphPreparationError,
    NpuGraphReplaySubmittedError,
    assign_draft_cache_locs,
    build_paged_draft_cache_locs,
    device_backend_key,
    fast_topk,
    get_last_loc_large_page_size_large_top_k,
    maybe_detect_nan,
    maybe_detect_oob,
    paged_tree_mapping_fits,
    select_top_k_tokens,
    split_draft_cache_locs,
)
from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
    SRTreeKVLease,
    first_forward_node_ids,
    immediate_free_pages,
    later_forward_node_ids,
    lease_budget_ok,
    lookup_candidate_slots,
    plan_paged_tree_layout,
    prefix_window_tokens,
    remap_slot_node_ids,
)
from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
    ALLOC_LEASE,
    ALLOC_ORDINARY,
    IMPL_PAGED_ATB,
    IMPL_PAGED_FIA,
    SR_TREE_WARMUP_ENV,
    SRTreeExpandTxn,
    build_step_context_lens,
    context_lens_list,
    kv_buckets_to_page_buckets,
    materialize_prefix_tail_copy_slots,
    plan_prefix_tail_copy_indices,
    prepare_tree_paged_view,
    read_sr_tree_update_overlap_env,
    read_sr_tree_warmup_env,
    resolve_eager_page_buckets,
    tree_paged_shape_key,
)
from sglang.srt.speculative.standalone_remote.drafter.sr_tree_warmup import (
    draft_warmup_raw_batch_sizes,
    warm_draft_alloc_mapping,
)
from sglang.srt.speculative.standalone_remote.sr_align import (
    SRWarmupFatalError,
    is_device_context_error,
    seq_lens_cpu_for_host,
    seq_lens_sum_from_batch,
)
from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
    advance_tree_draft_positions_for_step,
    copy_kv_pool_by_slot,
)
from sglang.srt.utils import next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)

SRTreeWindow = Tuple[List[int], Optional[List[int]], Optional[List[int]]]


def record_tree_expand_admission(metrics, can_cuda_graph, runner) -> bool:
    """Record one tree expand graph/eager outcome. Returns True if counted eager."""
    if metrics is None:
        return False
    if can_cuda_graph:
        metrics.counts["tree_graph_batches"] += 1
        return False
    metrics.counts["tree_eager_batches"] += 1
    reason = getattr(runner, "_last_can_run_reject", None) or "graph_unavailable"
    metrics.counts[f"tree_eager_{str(reason).split()[0]}"] += 1
    return True


def _as_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


def _wait_d2h_event(device, host: dict) -> None:
    """Block until the batched D2H into ``host`` is visible on CPU."""
    host["event"] = None
    if device is None:
        return
    dev_type = getattr(device, "type", None) or str(device)
    if dev_type == "cpu":
        return
    try:
        module = torch.get_device_module(dev_type)
        event = module.Event()
        event.record()
        event.synchronize()
        host["event"] = event
    except Exception:
        # CPU tests and missing device modules still have finished copy_ data.
        return


def _ensure_host_staging_slot(
    slots, idx: int, bs: int, token_w: int, parent_w: int, index_w: int, slot_w: int = 0
):
    host = slots[idx]
    token_w = max(int(token_w), 0)
    parent_w = max(int(parent_w), 0)
    index_w = max(int(index_w), 0)
    slot_w = max(int(slot_w), 0)
    bs = max(int(bs), 1)
    need_new = host is None
    if not need_new:
        need_new = (
            host["tokens"].shape[0] < bs
            or host["tokens"].shape[1] < token_w
            or host["parents"].shape[1] < parent_w
            or host["indices"].shape[1] < index_w
            or host.get("slots") is None
            or host["slots"].shape[1] < max(slot_w, 1)
        )
    if need_new:
        host = {
            "tokens": torch.zeros((bs, max(token_w, 1)), dtype=torch.int64),
            "parents": torch.zeros((bs, max(parent_w, 1)), dtype=torch.int64),
            "indices": torch.zeros((bs, max(index_w, 1)), dtype=torch.int64),
            "slots": torch.full((bs, max(slot_w, 1)), -1, dtype=torch.int64),
            "event": None,
            "in_use": False,
        }
        slots[idx] = host
    return host


class SRTreeDrafter:
    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler
        self.server_args = scheduler.server_args
        self.draft_model_runner = scheduler.tp_worker.model_runner
        self.device = getattr(scheduler, "device", None) or self.draft_model_runner.device
        self.topk = max(1, int(self.server_args.speculative_eagle_topk or 1))
        self.speculative_num_steps = max(
            1, int(self.server_args.speculative_num_steps or 1)
        )
        self.speculative_num_draft_tokens = int(
            self.server_args.speculative_num_draft_tokens
            or (self.speculative_num_steps + 1)
        )
        self.page_size = int(self.server_args.page_size or 1)
        self.token_to_kv_pool_allocator = scheduler.token_to_kv_pool_allocator
        self.req_to_token_pool = scheduler.req_to_token_pool
        self.model_config = scheduler.model_config
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
        self.draft_attn_backend = None
        self.cuda_graph_runner = None
        self._tree_failure_counts = {}
        self._tree_batch_isolate_count = 0
        self._tree_forward_calls = 0
        self._seen_tree_graph_keys = set()
        self.tree_graph_capture_succeeded = False
        self.tree_graph_disabled_reason = None
        # Two host stagings so a later replay cannot overwrite a D2H that has
        # not yet been split into RPC lists.
        self._host_staging = [None, None]
        self._staging_index = 0
        self._last_out_cache_loc = None
        self._last_candidate_slots = None
        max_bs = max(int(getattr(self.server_args, "cuda_graph_max_bs", 0) or 8), 8)
        self._identity_rows = max(max_bs, 1) * max(self.topk, 1)
        id_dev = self.device
        self._slot_node_ids = torch.full(
            (self.speculative_num_steps, self._identity_rows),
            -1,
            dtype=torch.int64,
            device=id_dev,
        )
        self._slot_node_id_tmp = torch.empty(
            (self._identity_rows,), dtype=torch.int64, device=id_dev
        )
        self._step0_node_ids = first_forward_node_ids(
            self.topk, max_bs, device=id_dev
        )
        self._init_attention_backend()
        self.sr_tree_paged = False
        self._paged_dummy_page = 0
        if self.draft_attn_backend is not None:
            inners = getattr(self.draft_attn_backend, "attn_backends", None) or []
            if inners:
                impl = getattr(inners[0], "tree_attention_impl", None)
                self.sr_tree_paged = impl in (IMPL_PAGED_ATB, IMPL_PAGED_FIA)
        # Graph runner captures in its constructor; set this first.
        self.need_draft_hidden = False
        self.npu_sr_tree_update_overlap_requested = read_sr_tree_update_overlap_env()
        self._init_cuda_graphs()

    def _lease_supported(self) -> bool:
        return self.page_size > 1 and self.topk > 1

    def _ensure_identity_capacity(self, rows: int) -> None:
        rows = max(int(rows), 1)
        if self._slot_node_ids is not None and self._slot_node_ids.shape[1] >= rows:
            return
        device = self._slot_node_ids.device if self._slot_node_ids is not None else self.device
        steps = self.speculative_num_steps
        self._slot_node_ids = torch.full(
            (steps, rows), -1, dtype=torch.int64, device=device
        )
        self._slot_node_id_tmp = torch.empty((rows,), dtype=torch.int64, device=device)
        max_bs = max((rows + self.topk - 1) // max(self.topk, 1), 1)
        self._step0_node_ids = first_forward_node_ids(self.topk, max_bs, device=device)
        self._identity_rows = rows

    @property
    def model_runner(self):
        """Alias for EAGLEDraftCudaGraphRunner, which reads ``eagle_worker.model_runner``."""
        return self.draft_model_runner

    def draft_forward(self, forward_batch: ForwardBatch):
        """Alias for CUDA-graph capture, which calls ``eagle_worker.draft_forward``."""
        return self._draft_forward(forward_batch)

    def _log_tree_failure(self, stage: str, exc: Exception) -> None:
        """One traceback per failure signature; repeated failures report totals."""
        # NPU errors append timestamps/PIDs on subsequent lines. Do not include
        # request IDs in this key, otherwise every new request logs a traceback.
        message = str(exc).splitlines()
        summary = message[0] if message else type(exc).__name__
        key = (stage, type(exc), summary)
        count = self._tree_failure_counts.get(key, 0) + 1
        self._tree_failure_counts[key] = count
        if count != 1 and count % 32:
            return
        backend = self.draft_attn_backend
        inners = getattr(backend, "attn_backends", None)
        if inners:
            backend = inners[0]
        metadata = getattr(getattr(backend, "forward_metadata", None), "tree_shared", None)
        dtypes = (
            {name: str(value.dtype) for name, value in vars(metadata).items()}
            if metadata is not None
            else {}
        )
        dtypes["req_to_token"] = str(self.req_to_token_pool.req_to_token.dtype)
        logger.warning(
            "[SR] tree failure stage=%s implementation=%s metadata_dtypes=%s "
            "occurrences=%s isolate_batches=%s error=%s",
            stage,
            getattr(backend, "tree_attention_impl", type(backend).__name__),
            dtypes,
            count,
            getattr(self, "_tree_batch_isolate_count", 0),
            summary,
            exc_info=(type(exc), exc, exc.__traceback__) if count == 1 else None,
        )

    def _init_attention_backend(self) -> None:
        if self.speculative_num_steps <= 1:
            return
        try:
            factory = DraftBackendFactory(
                self.server_args,
                self.draft_model_runner,
                self.topk,
                self.speculative_num_steps,
            )
            self.draft_attn_backend = factory.create_decode_backend()
        except Exception as e:
            logger.warning("[SR] tree draft attention backend unavailable: %s", e)
            self.draft_attn_backend = None

    def _init_cuda_graphs(self) -> None:
        """Capture v1 EAGLE draft-tree graphs. Skip EAGLE draft-extend graphs."""
        self.cuda_graph_runner = None
        self.tail_graph_runner = None
        self.tree_graph_capture_succeeded = False
        self.tree_graph_disabled_reason = None
        if getattr(self.server_args, "disable_cuda_graph", False):
            self.tree_graph_disabled_reason = "disabled by configuration"
            getattr(self, "_sr_warm_tree_shapes", lambda: None)()
            return
        if self.speculative_num_steps <= 1 or self.draft_attn_backend is None:
            self.tree_graph_disabled_reason = "no multi-step attention backend"
            getattr(self, "_init_tail_graphs", lambda: None)()
            getattr(self, "_sr_warm_tree_shapes", lambda: None)()
            return
        prev_draft_backend = getattr(self.draft_model_runner, "draft_attn_backend", None)
        try:
            from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
                EAGLEDraftCudaGraphRunner,
            )

            backend = device_backend_key(self.device)
            # Import the NPU runner only on NPU so CUDA hosts never need torch_npu.
            if backend == "npu":
                from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
                    EAGLEDraftNpuGraphRunner,
                )

                runner_cls = EAGLEDraftNpuGraphRunner
            else:
                runner_cls = EAGLEDraftCudaGraphRunner
            self.draft_model_runner.draft_attn_backend = self.draft_attn_backend
            self.need_draft_hidden = False
            logger.info("[SR] Capture tree draft graph begin (backend=%s).", backend)
            self.cuda_graph_runner = runner_cls(self)
            reason = getattr(
                self.cuda_graph_runner, "tree_graph_disabled_reason", None
            )
            graphs = getattr(self.cuda_graph_runner, "graphs", None)
            if reason or not graphs:
                self.tree_graph_disabled_reason = reason or "no graphs"
                logger.warning(
                    "[SR] tree draft graphs disabled after capture: %s "
                    "(tree_graph_replay_count=%s tree_eager_fallback_count=%s)",
                    reason or "no graphs",
                    getattr(self.cuda_graph_runner, "tree_graph_replay_count", 0),
                    getattr(self.cuda_graph_runner, "tree_eager_fallback_count", 0),
                )
                self.cuda_graph_runner = None
            else:
                self.tree_graph_capture_succeeded = True
            logger.info("[SR] Capture tree draft graph end.")
        except NpuGraphReplaySubmittedError:
            raise
        except Exception as e:
            if is_device_context_error(e):
                raise
            self._log_tree_failure("capture", e)
            self.tree_graph_disabled_reason = "capture failed; see tree failure traceback"
            self.cuda_graph_runner = None
        finally:
            self.draft_model_runner.draft_attn_backend = prev_draft_backend
        getattr(self, "_init_tail_graphs", lambda: None)()
        getattr(self, "_sr_warm_tree_shapes", lambda: None)()

    def _init_tail_graphs(self) -> None:
        """Capture SR-only tail EXTEND graphs. Keep ordinary EXTEND attention."""
        self.tail_graph_runner = None
        if getattr(self.server_args, "disable_cuda_graph", False):
            return
        try:
            from sglang.srt.speculative.standalone_remote.drafter.sr_tail_extend_graph import (
                create_sr_tail_extend_graph_runner,
            )

            runner = create_sr_tail_extend_graph_runner(self)
            if runner is None or not getattr(runner, "graphs", None):
                reason = getattr(runner, "disabled_reason", None) if runner else None
                logger.info("[SR] tail EXTEND graphs disabled: %s", reason or "no graphs")
                return
            self.tail_graph_runner = runner
            logger.info(
                "[SR] tail EXTEND graphs captured: buckets=%s",
                list(runner.graphs),
            )
        except NpuGraphReplaySubmittedError:
            raise
        except Exception as e:
            if is_device_context_error(e):
                raise
            logger.warning("[SR] tail EXTEND graph init failed: %s", e)
            self.tail_graph_runner = None

    def _sr_warmup_capture_bs(self) -> list[int]:
        runner = self.cuda_graph_runner
        capture_bs = [int(b) for b in (getattr(runner, "capture_bs", None) or [])]
        if not capture_bs:
            raw = getattr(self.server_args, "cuda_graph_bs", None) or [1]
            capture_bs = [int(b) for b in raw]
        return [b for b in capture_bs if b > 0]

    def _sr_warmup_raw_batch_sizes(self) -> list[int]:
        sizes, _skip = draft_warmup_raw_batch_sizes(self)
        return list(sizes)

    def _sr_warmup_page_buckets(self) -> list[int]:
        backend = self.draft_attn_backend
        inners = getattr(backend, "attn_backends", None) if backend is not None else None
        inner = inners[0] if inners else None
        page_buckets = resolve_eager_page_buckets(
            getattr(inner, "tree_kv_buckets", None),
            self.page_size,
            getattr(inner, "_paged_graph_max_pages", None),
        )
        if not page_buckets:
            page_buckets = kv_buckets_to_page_buckets([256, 512, 1024], self.page_size)
        return sorted({max(int(b), 1) for b in page_buckets})

    def _sr_warm_layout_shapes_builders(self) -> list[tuple[int, int, int, int]]:
        """CUDA / no-eager-backend fallback: prime builders without bind state."""
        req_to_token = self.req_to_token_pool.req_to_token
        device = req_to_token.device
        dummy = int(self._paged_dummy_page)
        page_buckets = self._sr_warmup_page_buckets()
        kv_pool = getattr(self.draft_model_runner, "token_to_kv_pool", None)
        pool_cols = int(req_to_token.shape[1])
        page = int(self.page_size)
        max_shared = max(int(b) for b in page_buckets)
        steps = max(int(self.speculative_num_steps), 1)
        rem_choices = sorted({1, max(page - steps + 1, 1), max(page - 1, 1)})
        combos = []
        seen = set()
        for bs in self._sr_warmup_raw_batch_sizes() or self._sr_warmup_capture_bs():
            pool_rows = min(int(bs), int(req_to_token.shape[0]))
            pool = torch.arange(pool_rows, dtype=torch.int64, device=device)
            if pool_rows < int(bs):
                continue
            slots = torch.arange(
                int(bs) * self.topk * self.speculative_num_steps,
                dtype=torch.int64,
                device=device,
            ).reshape(int(bs), self.topk, self.speculative_num_steps)
            for shared in range(max_shared + 1):
                for rem in rem_choices:
                    prefix = shared * page + rem
                    prefixes = [prefix] * int(bs)
                    key = tree_paged_shape_key(
                        prefixes,
                        page,
                        self.topk,
                        self.speculative_num_steps,
                        page_buckets,
                    )
                    if key in seen or prefix >= pool_cols:
                        continue
                    seen.add(key)
                    width = key[3]
                    tables, _shared, branch, active, _n_sh, _n_q = (
                        prepare_tree_paged_view(
                            req_to_token,
                            pool,
                            slots,
                            prefixes,
                            page,
                            self.topk,
                            self.speculative_num_steps,
                            dummy_page=dummy,
                            max_pages=width,
                        )
                    )
                    n_rows = int(tables.shape[0])
                    n_fwd = max(int(self.speculative_num_steps) - 1, 0)
                    for step in range(max(n_fwd, 1)):
                        lens = build_step_context_lens(
                            prefixes, self.topk, step, n_rows
                        )
                        context_lens_list(lens)
                    tables.contiguous()
                    active.contiguous()
                    if rem:
                        indices = plan_prefix_tail_copy_indices(
                            prefixes, ALLOC_ORDINARY, self.topk, page
                        )
                        if len(indices) and kv_pool is not None:
                            src, dst = materialize_prefix_tail_copy_slots(
                                req_to_token,
                                pool,
                                branch,
                                indices,
                                page,
                            )
                            if int(src.numel()) > 0:
                                copy_kv_pool_by_slot(kv_pool, src, dst)
                    combos.append(key)
        return combos

    def _sr_warm_layout_shapes(self) -> list[tuple[int, int, int, int]]:
        backend = self.draft_attn_backend
        prepare = getattr(backend, "prepare_sr_tree_paged_eager", None)
        if prepare is None:
            return self._sr_warm_layout_shapes_builders()
        req_to_token = self.req_to_token_pool.req_to_token
        device = req_to_token.device
        dummy = int(self._paged_dummy_page)
        page_buckets = self._sr_warmup_page_buckets()
        kv_pool = getattr(self.draft_model_runner, "token_to_kv_pool", None)
        pool_cols = int(req_to_token.shape[1])
        pool_rows_max = int(req_to_token.shape[0])
        page = int(self.page_size)
        # Enumerate raw shared page counts, not bucket values: n_query is always
        # shared+1 or shared+2, so bucket values alone never reach the
        # (shared, width) pairs real prefixes produce.
        max_shared = max(int(b) for b in page_buckets)
        # rem=1 gives nnp=1 with n_query=shared+1; page-steps+1 is the smallest
        # rem where nnp=2 while n_query is still shared+1; page-1 gives nnp=2
        # with n_query=shared+2. nnp=1 with shared+2 is unreachable. All three
        # keep remainder != 0 so the prefix-tail copy path is warmed too.
        steps = max(int(self.speculative_num_steps), 1)
        rem_choices = sorted({1, max(page - steps + 1, 1), max(page - 1, 1)})
        combos = []
        seen = set()
        try:
            for bs in self._sr_warmup_raw_batch_sizes() or self._sr_warmup_capture_bs():
                raw_bs = int(bs)
                if raw_bs > pool_rows_max:
                    continue
                pool = torch.arange(raw_bs, dtype=torch.int64, device=device)
                slots = torch.arange(
                    raw_bs * self.topk * self.speculative_num_steps,
                    dtype=torch.int64,
                    device=device,
                ).reshape(raw_bs, self.topk, self.speculative_num_steps)
                dummy_fb = SimpleNamespace(
                    batch_size=raw_bs, req_pool_indices=pool
                )
                for shared in range(max_shared + 1):
                    for rem in rem_choices:
                        prefix = shared * page + rem
                        prefixes = [prefix] * raw_bs
                        key = tree_paged_shape_key(
                            prefixes,
                            page,
                            self.topk,
                            self.speculative_num_steps,
                            page_buckets,
                        )
                        if key in seen or prefix >= pool_cols:
                            continue
                        seen.add(key)
                        seq_lens = torch.tensor(
                            prefixes, dtype=torch.int64, device=device
                        )
                        prepare(
                            dummy_fb,
                            slots,
                            prefixes,
                            ALLOC_ORDINARY,
                            kv_pool,
                            seq_lens=seq_lens,
                            dummy_page=dummy,
                        )
                        prepare(
                            dummy_fb,
                            slots,
                            prefixes,
                            ALLOC_LEASE,
                            kv_pool,
                            seq_lens=seq_lens,
                            dummy_page=dummy,
                        )
                        combos.append(key)
        finally:
            clear = getattr(backend, "_sr_clear_paged_round_state", None)
            if clear is not None:
                clear()
        return combos

    def _sr_warm_allocator_shapes(self) -> list[int]:
        if self.page_size <= 1 or self.topk <= 1:
            return []
        tree_cache = getattr(self.scheduler, "tree_cache", None)
        if tree_cache is None:
            return []
        device = self.device
        req_to_token = self.req_to_token_pool.req_to_token
        warmed = []
        for bs in self._sr_warmup_raw_batch_sizes() or self._sr_warmup_capture_bs():
            prefix = int(self.page_size)
            req_pool = torch.zeros((int(bs),), dtype=torch.int64, device=device)
            seq_lens = torch.full(
                (int(bs),), prefix, dtype=torch.int64, device=device
            )
            seq_lens_cpu = torch.full((int(bs),), prefix, dtype=torch.int64)
            (
                prefix_lens,
                seq_lens_out,
                last_loc,
                _num_new_pages,
                _extend_lens,
                _last_page_lens,
            ) = get_last_loc_large_page_size_large_top_k(
                req_to_token,
                req_pool,
                seq_lens,
                self.speculative_num_steps,
                self.topk,
                self.page_size,
            )
            last_page_lens_cpu = seq_lens_cpu % self.page_size
            num_new_pages = (
                last_page_lens_cpu + self.speculative_num_steps + self.page_size - 1
            ) // self.page_size
            seq_lens_cpu_out = (
                seq_lens_cpu // self.page_size * self.page_size
                + num_new_pages * (self.page_size * self.topk)
            )
            extend_num_tokens = int(torch.sum(seq_lens_cpu_out - seq_lens_cpu).item())
            _out, backup = alloc_paged_token_slots_extend(
                tree_cache,
                prefix_lens,
                seq_lens_cpu,
                seq_lens_out,
                seq_lens_cpu_out,
                last_loc,
                extend_num_tokens,
                backup_state=True,
            )
            if backup is not None:
                self.token_to_kv_pool_allocator.restore_state(backup)
            warmed.append(int(bs))
        return warmed

    def _sr_warm_tree_shapes(self) -> None:
        """Prime layout, alloc, and mapping before the first request."""
        if not read_sr_tree_warmup_env():
            logger.info("[SR] tree shape warmup skipped: %s=0", SR_TREE_WARMUP_ENV)
            return
        if not self.sr_tree_paged:
            return
        t0 = time.perf_counter()
        combos = []
        alloc_keys = []
        mapping_keys = []
        try:
            combos = self._sr_warm_layout_shapes()
            coverage = warm_draft_alloc_mapping(self)
            alloc_keys = coverage.get("allocation") or []
            mapping_keys = coverage.get("mapping") or []
        except (NpuGraphReplaySubmittedError, SRWarmupFatalError):
            raise
        except Exception as e:
            if is_device_context_error(e):
                raise
            logger.warning(
                "[SR] tree shape warmup failed: %s uncovered layout=%s "
                "allocation=%s mapping=%s",
                e,
                combos,
                alloc_keys,
                mapping_keys,
            )
            return
        seen = getattr(self, "_seen_tree_paged_shapes", None)
        if seen is None:
            seen = set()
            self._seen_tree_paged_shapes = seen
        seen.update(combos)
        logger.info(
            "[SR] tree warmup layout=%s allocation=%s mapping=%s elapsed=%.3fs",
            combos,
            alloc_keys,
            mapping_keys,
            time.perf_counter() - t0,
        )

    def expand_batch(self, reqs: List["Req"]) -> List[SRTreeWindow]:
        """Fused tree expand for every req that has a pool slot and a seed."""
        empty: SRTreeWindow = ([], None, None)
        if not reqs:
            return []
        windows: List[SRTreeWindow] = [empty] * len(reqs)
        keep: List["Req"] = []
        keep_idx: List[int] = []
        for i, req in enumerate(reqs):
            if req.req_pool_idx is None:
                continue
            if getattr(req, "sr_tree_seed", None) is None:
                continue
            keep.append(req)
            keep_idx.append(i)
        if not keep:
            return windows
        try:
            parent_list, top_scores_index, draft_tokens = self._expand_tree(
                keep, *self._stack_seeds(keep)
            )
        except NpuGraphReplaySubmittedError:
            raise
        except Exception as e:
            if is_device_context_error(e):
                raise
            if len(keep) > 1:
                self._tree_batch_isolate_count = (
                    getattr(self, "_tree_batch_isolate_count", 0) + 1
                )
            self._log_tree_failure("expand_batch", e)
            # Retrying the very same single-request batch cannot isolate a bad
            # request. Preserve the empty-window fallback without a second run.
            # Do not pack the failed batch's graph outputs before per-req replay.
            if len(keep) == 1:
                return windows
            for j, req in enumerate(keep):
                windows[keep_idx[j]] = self._expand_one(req)
            return windows
        metrics = getattr(self.scheduler, "_sr_round_metrics", None)
        try:
            with metrics.phase("tree_result_wait_pack") if metrics else nullcontext():
                self._pack_tree_windows(
                    parent_list, top_scores_index, draft_tokens, keep_idx, windows, keep
                )
        except Exception:
            lease_state = getattr(self, "_pending_lease_state", None)
            self._pending_lease_state = None
            self._free_lease_alloc(lease_state)
            raise
        return windows

    def _expand_one(self, req: "Req") -> SRTreeWindow:
        empty: SRTreeWindow = ([], None, None)
        seed = getattr(req, "sr_tree_seed", None)
        if req.req_pool_idx is None or seed is None:
            return empty
        try:
            parent_list, top_scores_index, draft_tokens = self._expand_tree(
                [req], *self._stack_seeds([req])
            )
        except NpuGraphReplaySubmittedError:
            raise
        except Exception as e:
            if is_device_context_error(e):
                raise
            self._log_tree_failure("expand_one", e)
            return empty
        windows: List[SRTreeWindow] = [empty]
        metrics = getattr(self.scheduler, "_sr_round_metrics", None)
        try:
            with metrics.phase("tree_result_wait_pack") if metrics else nullcontext():
                self._pack_tree_windows(
                    parent_list, top_scores_index, draft_tokens, [0], windows, [req]
                )
        except Exception:
            lease_state = getattr(self, "_pending_lease_state", None)
            self._pending_lease_state = None
            self._free_lease_alloc(lease_state)
            raise
        return windows[0]

    def _pack_tree_windows(
        self,
        parent_list,
        top_scores_index,
        draft_tokens,
        keep_idx: List[int],
        windows: List[SRTreeWindow],
        reqs: Optional[List["Req"]] = None,
    ) -> None:
        """One D2H per tree tensor, then CPU row splits into RPC lists."""
        tokens_cpu, parents_cpu, indices_cpu, slots_cpu = self._d2h_tree_outputs(
            draft_tokens, parent_list, top_scores_index
        )
        packed = []
        for j, idx in enumerate(keep_idx):
            window = (
                tokens_cpu[j].tolist(),
                parents_cpu[j].tolist(),
                indices_cpu[j].tolist(),
            )
            windows[idx] = window
            packed.append(window)
        self._publish_tree_leases(reqs or [], packed, slots_cpu)

    def _publish_tree_leases(self, reqs, windows, slots_cpu) -> None:
        lease_state = getattr(self, "_pending_lease_state", None)
        self._pending_lease_state = None
        if not lease_state:
            return
        store = getattr(self.scheduler, "sr_tree_leases", None)
        if store is None:
            self._free_lease_alloc(lease_state)
            return
        unpublished = set(range(len(lease_state["page_slots"])))
        try:
            for j, req in enumerate(lease_state["reqs"]):
                tokens, pl, ix = windows[j] if j < len(windows) else ([], None, None)
                if not tokens or pl is None or ix is None:
                    self.token_to_kv_pool_allocator.free(lease_state["page_slots"][j])
                    unpublished.discard(j)
                    continue
                if slots_cpu is not None and j < slots_cpu.shape[0]:
                    cand = [int(x) for x in slots_cpu[j].tolist()]
                    if len(cand) < len(ix):
                        cand = cand + [-1] * (len(ix) - len(cand))
                    cand = cand[: len(ix)]
                else:
                    cand = [-1] * len(ix)
                base = int(getattr(req, "kv_committed_len", 0) or 0)
                if base <= 0:
                    origin = req.origin_input_ids or []
                    output = req.output_ids or []
                    base = len(origin) + len(output)
                lease = SRTreeKVLease(
                    rid=req.rid,
                    version=store.next_version(),
                    revision=int(getattr(req, "sr_prefix_revision", 0) or 0),
                    base_committed_len=base,
                    prefix_tokens=prefix_window_tokens(
                        req.origin_input_ids or (),
                        req.output_ids or (),
                        base,
                    ),
                    page_ids=[],
                    page_slots=lease_state["page_slots"][j],
                    candidate_slots=list(cand),
                    parent_list=list(pl),
                    top_scores_index=list(ix),
                    draft_tokens=list(tokens),
                    page_count=int(lease_state["page_counts"][j]),
                )
                prev = store.pop(req.rid)
                if prev is not None:
                    store.release(prev, allocator=self.token_to_kv_pool_allocator)
                store.register(lease)
                unpublished.discard(j)
                req.sr_tree_version = lease.version
        except Exception:
            for j in unpublished:
                self.token_to_kv_pool_allocator.free(lease_state["page_slots"][j])
            raise

    def _d2h_tree_outputs(self, draft_tokens, parent_list, top_scores_index):
        tensors = (draft_tokens, parent_list, top_scores_index)
        if all(isinstance(t, torch.Tensor) for t in tensors):
            tokens = _as_2d(draft_tokens.detach())
            parents = _as_2d(parent_list.detach())
            indices = _as_2d(top_scores_index.detach())
            n = int(tokens.shape[0])
            token_w = int(tokens.shape[1]) if tokens.dim() > 1 else 1
            parent_w = int(parents.shape[1]) if parents.dim() > 1 else 1
            index_w = int(indices.shape[1]) if indices.dim() > 1 else 1
            slot_w = index_w
            slots_dev = None
            lease_state = getattr(self, "_pending_lease_state", None)
            if lease_state is not None:
                compact = getattr(self, "_lease_compact_slots", None)
                if compact is not None and self._slot_node_ids is not None:
                    phys = compact.reshape(
                        n, self.topk, self.speculative_num_steps
                    ).permute(2, 0, 1).reshape(self.speculative_num_steps, -1)
                    slots_dev = lookup_candidate_slots(
                        self._slot_node_ids, phys, indices, n, self.topk
                    ).clone()
                    slot_w = int(slots_dev.shape[1])
            host = self._acquire_host_staging(n, token_w, parent_w, index_w, slot_w)
            non_blocking = tokens.device.type != "cpu"
            host["tokens"][:n, :token_w].copy_(tokens, non_blocking=non_blocking)
            host["parents"][:n, :parent_w].copy_(parents, non_blocking=non_blocking)
            host["indices"][:n, :index_w].copy_(indices, non_blocking=non_blocking)
            if slots_dev is not None:
                host["slots"][:n, :slot_w].copy_(slots_dev, non_blocking=non_blocking)
            else:
                host["slots"][:n].fill_(-1)
            _wait_d2h_event(tokens.device, host)
            host["in_use"] = False
            slots_cpu = host["slots"][:n, :slot_w] if slots_dev is not None else None
            return (
                host["tokens"][:n, :token_w],
                host["parents"][:n, :parent_w],
                host["indices"][:n, :index_w],
                slots_cpu,
            )
        return (
            draft_tokens.detach().to("cpu"),
            parent_list.detach().to("cpu"),
            top_scores_index.detach().to("cpu"),
            None,
        )

    def _acquire_host_staging(
        self, bs: int, token_w: int, parent_w: int, index_w: int, slot_w: int = 0
    ):
        slots = getattr(self, "_host_staging", None)
        if not slots:
            self._host_staging = [None, None]
            self._staging_index = 0
            slots = self._host_staging
        idx = int(getattr(self, "_staging_index", 0) or 0) % 2
        other = slots[idx]
        if other is not None and other.get("in_use"):
            event = other.get("event")
            if event is not None:
                event.synchronize()
            other["in_use"] = False
        host = _ensure_host_staging_slot(
            slots, idx, bs, token_w, parent_w, index_w, slot_w
        )
        host["in_use"] = True
        self._staging_index = (idx + 1) % 2
        return host

    def _stack_seeds(
        self, reqs: List["Req"]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        ps: List[torch.Tensor] = []
        ixs: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        for req in reqs:
            topk_p, topk_index, _hidden_states, verified_id = req.sr_tree_seed
            if topk_p.dim() == 1:
                topk_p = topk_p.unsqueeze(0)
            if topk_index.dim() == 1:
                topk_index = topk_index.unsqueeze(0)
            if verified_id.dim() == 0:
                verified_id = verified_id.unsqueeze(0)
            ps.append(topk_p[:1])
            ixs.append(topk_index[:1])
            vs.append(verified_id.reshape(-1)[:1])
        return (
            torch.cat(ps, dim=0),
            torch.cat(ixs, dim=0),
            None,
            torch.cat(vs, dim=0),
        )

    def _expand_tree(
        self,
        reqs: List["Req"],
        topk_p: torch.Tensor,
        topk_index: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        verified_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del hidden_states
        scheduler = self.scheduler
        metrics = getattr(scheduler, "_sr_round_metrics", None)
        with metrics.phase("tree_make_batch") if metrics else nullcontext():
            batch = scheduler._sr_make_decode_batch(reqs)
        spec_info = EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=None,
            verified_id=verified_id,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        batch.spec_info = spec_info
        batch.return_hidden_states = False
        txn_cls = globals().get("SRTreeExpandTxn")
        if txn_cls is None:

            class txn_cls:
                def __init__(self):
                    self.allocation_owned = False
                    self.prefix_copy_submitted = False
                    self.tree_compute_submitted = False
                    self.completion_confirmed = False
                    self.rolled_back = False
                    self.lease_state = None
                    self.allocator_backup = None

                def mark_copy_begin(self):
                    self.prefix_copy_submitted = True

                def mark_compute_begin(self):
                    self.tree_compute_submitted = True

                def in_flight(self):
                    return (
                        self.prefix_copy_submitted or self.tree_compute_submitted
                    ) and not self.completion_confirmed

                def may_rollback(self):
                    return (not self.in_flight()) or self.completion_confirmed

        txn = txn_cls()
        with metrics.phase("tree_alloc_kv") if metrics else nullcontext():
            token_to_kv_pool_state_backup, lease_state = self._alloc_tree_kv(batch)
        txn.allocation_owned = True
        txn.lease_state = lease_state
        txn.allocator_backup = token_to_kv_pool_state_backup
        self._pending_lease_state = lease_state
        self._lease_compact_slots = batch.out_cache_loc
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL
        model_worker_batch = batch.get_model_worker_batch()
        prev_draft_backend = getattr(self.draft_model_runner, "draft_attn_backend", None)
        graph_submitted = False
        t_prep = time.perf_counter()
        try:
            if self.draft_attn_backend is not None:
                self.draft_model_runner.draft_attn_backend = self.draft_attn_backend
            forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.draft_model_runner
            )
            init_s = time.perf_counter() - t_prep
            t_paged = time.perf_counter()
            if getattr(self, "sr_tree_paged", False):
                txn.mark_copy_begin()
                prepare = getattr(self, "_prepare_paged_tree_round", None)
                if prepare is not None:
                    prepare(
                        forward_batch, batch, lease_state is not None, metrics
                    )
            paged_s = time.perf_counter() - t_paged
            prep_s = time.perf_counter() - t_prep
            if metrics is not None:
                metrics.add_host("tree_init_forward_batch", init_s)
                metrics.add_host("tree_paged_eager", paged_s)
                metrics.add_host("tree_prepare_meta", prep_s)
            can_fn = getattr(self, "_can_run_tree_graph", None)
            if can_fn is not None:
                can_cuda_graph = can_fn(forward_batch)
            else:
                runner = getattr(self, "cuda_graph_runner", None)
                can_cuda_graph = bool(
                    runner is not None and runner.can_run(forward_batch)
                )
            runner = getattr(self, "cuda_graph_runner", None)
            counted_eager = record_tree_expand_admission(
                metrics, can_cuda_graph, runner
            )
            if metrics is not None and can_cuda_graph and runner is not None:
                plan = getattr(runner, "_tree_replay_plan", None)
                key = getattr(plan, "graph_key", None)
                seen = getattr(self, "_seen_tree_graph_keys", None)
                if seen is None:
                    seen = set()
                    self._seen_tree_graph_keys = seen
                if key is not None and key not in seen:
                    seen.add(key)
                    metrics.counts["tree_graph_key_first_use"] += 1
                    metrics.counts[f"tree_graph_first_{key}"] += 1
            t_exec = time.perf_counter()
            with (
                metrics.phase("tree_forward", device=True) if metrics else nullcontext()
            ):
                txn.mark_compute_begin()
                if can_cuda_graph:
                    try:
                        parent_list, top_scores_index, draft_tokens = (
                            self.cuda_graph_runner.replay(forward_batch)
                        )
                    except NpuGraphReplaySubmittedError:
                        self.cuda_graph_runner = None
                        graph_submitted = True
                        raise
                    except NpuGraphPreparationError as e:
                        logger.warning(
                            "[SR] tree draft graph replay prep failed: %s; "
                            "falling back to eager",
                            e,
                        )
                        if getattr(e, "scope", "graph") == "format":
                            self.cuda_graph_runner = None
                        else:
                            runner = self.cuda_graph_runner
                            if runner is not None:
                                runner.tree_eager_fallback_count = (
                                    getattr(runner, "tree_eager_fallback_count", 0) + 1
                                )
                                if hasattr(runner, "_last_can_run_reject"):
                                    runner._last_can_run_reject = "prep_failed"
                        if metrics is not None and not counted_eager:
                            metrics.counts["tree_eager_batches"] += 1
                            metrics.counts["tree_eager_prep_failed"] += 1
                        can_cuda_graph = False
                if not can_cuda_graph:
                    if (
                        self.draft_attn_backend is not None
                        and self.speculative_num_steps > 1
                        and not forward_batch.forward_mode.is_idle()
                    ):
                        self.draft_attn_backend.init_forward_metadata(forward_batch)
                    parent_list, top_scores_index, draft_tokens = self._draft_forward(
                        forward_batch
                    )
            exec_s = time.perf_counter() - t_exec
            txn.completion_confirmed = True
        except NpuGraphReplaySubmittedError:
            graph_submitted = True
            raise
        except Exception as e:
            ctx_err = globals().get("is_device_context_error")
            if ctx_err is not None and ctx_err(e):
                graph_submitted = True
                raise NpuGraphReplaySubmittedError(
                    "tree expand device context error"
                ) from e
            if txn.in_flight():
                confirm = getattr(self, "_try_confirm_tree_completion", lambda: True)
                if not confirm():
                    graph_submitted = True
                    raise NpuGraphReplaySubmittedError(
                        "tree expand in-flight; refuse rollback"
                    ) from e
                txn.completion_confirmed = True
            if txn.may_rollback() and not graph_submitted:
                rollback = getattr(self, "_rollback_tree_expand", None)
                if rollback is not None:
                    rollback(batch, txn)
                elif token_to_kv_pool_state_backup is not None:
                    self.token_to_kv_pool_allocator.restore_state(
                        token_to_kv_pool_state_backup
                    )
                    txn.rolled_back = True
            raise
        finally:
            self.draft_model_runner.draft_attn_backend = prev_draft_backend
            abandon = graph_submitted or (
                txn.in_flight() and not txn.completion_confirmed
            )
            if abandon:
                self._pending_lease_state = None
            elif txn.rolled_back:
                pass
            elif lease_state is not None and self._pending_lease_state is not None:
                self._restore_tree_mapping(batch, lease_state)
            elif token_to_kv_pool_state_backup is not None:
                self.token_to_kv_pool_allocator.restore_state(
                    token_to_kv_pool_state_backup
                )
        runner = self.cuda_graph_runner
        replay_n = getattr(runner, "tree_graph_replay_count", 0) if runner else 0
        self._tree_forward_calls = getattr(self, "_tree_forward_calls", 0) + 1
        calls = self._tree_forward_calls
        reason = None if can_cuda_graph else (
            getattr(runner, "_last_can_run_reject", None) or "graph_unavailable"
        )
        if calls <= 1 or calls % 8 == 0:
            logger.info(
                "[SR] tree draft timings: prepare_host=%.3fs "
                "forward_call_host=%.3fs graph=%s "
                "replay=%s eager_fallback=%s reason=%s",
                prep_s,
                exec_s,
                can_cuda_graph,
                replay_n,
                getattr(runner, "tree_eager_fallback_count", 0) if runner else 0,
                reason,
            )
        return parent_list, top_scores_index, draft_tokens

    def _can_run_tree_graph(self, forward_batch) -> bool:
        runner = self.cuda_graph_runner
        if runner is None:
            return False
        if self.sr_tree_paged and not getattr(runner, "_tree_paged", False):
            return False
        return bool(runner.can_run(forward_batch))

    def _try_confirm_tree_completion(self) -> bool:
        device = self.device
        if device is None:
            return True
        dev_type = getattr(device, "type", None) or str(device)
        if dev_type == "cpu":
            return True
        try:
            torch.get_device_module(dev_type).synchronize()
            return True
        except Exception:
            return False

    def _rollback_tree_expand(self, batch, txn: SRTreeExpandTxn) -> None:
        if txn.rolled_back or not txn.may_rollback():
            return
        if txn.lease_state is not None:
            self._restore_tree_mapping(batch, txn.lease_state)
            self._free_lease_alloc(txn.lease_state)
            self._pending_lease_state = None
        elif txn.allocator_backup is not None:
            self.token_to_kv_pool_allocator.restore_state(txn.allocator_backup)
        txn.rolled_back = True
        txn.allocation_owned = False

    def _prepare_paged_tree_round(
        self, forward_batch, batch, leased: bool, metrics=None
    ) -> None:
        backend = self.draft_attn_backend
        if backend is None or not hasattr(backend, "prepare_sr_tree_paged_eager"):
            return
        prefix = seq_lens_cpu_for_host(batch)
        if metrics is not None:
            inners = getattr(backend, "attn_backends", None)
            inner = inners[0] if inners else None
            page_buckets = resolve_eager_page_buckets(
                getattr(inner, "tree_kv_buckets", None),
                self.page_size,
                getattr(inner, "_paged_graph_max_pages", None),
            )
            key = tree_paged_shape_key(
                prefix,
                self.page_size,
                self.topk,
                self.speculative_num_steps,
                page_buckets,
            )
            seen = getattr(self, "_seen_tree_paged_shapes", None)
            if seen is None:
                seen = set()
                self._seen_tree_paged_shapes = seen
            if key not in seen:
                seen.add(key)
                metrics.counts["tree_shape_first_use"] += 1
                metrics.counts["tree_shape_{}_{}_{}_{}".format(*key)] += 1
        kind = ALLOC_LEASE if leased else ALLOC_ORDINARY
        compact = batch.out_cache_loc
        kv_pool = getattr(self.draft_model_runner, "token_to_kv_pool", None)
        backend.prepare_sr_tree_paged_eager(
            forward_batch,
            compact,
            prefix,
            kind,
            kv_pool,
            batch.seq_lens,
            dummy_page=self._paged_dummy_page,
            metrics=metrics,
        )

    def _alloc_tree_kv(self, batch: "ScheduleBatch"):
        from sglang.srt.speculative.eagle_worker import (
            get_last_loc_large_page_size_top_k_1,
        )

        num_seqs = batch.batch_size()
        token_to_kv_pool_state_backup = None
        lease_state = None
        pool_len = batch.req_to_token_pool.req_to_token.shape[1]
        if not paged_tree_mapping_fits(
            seq_lens_cpu_for_host(batch),
            self.page_size,
            self.topk,
            self.speculative_num_steps,
            pool_len,
        ):
            raise RuntimeError(
                "tree draft mapping exceeds req_to_token width: "
                f"pool_len={pool_len} page_size={self.page_size} "
                f"topk={self.topk} steps={self.speculative_num_steps}"
            )
        leased = self._try_alloc_lease_tree_kv(batch)
        if leased is not None:
            raw_cache_loc, lease_state = leased
            try:
                self._apply_tree_mapping(batch, raw_cache_loc)
                return None, lease_state
            except Exception:
                self._restore_tree_mapping(batch, lease_state)
                self._free_lease_alloc(lease_state)
                raise
        if self.page_size == 1:
            alloc_len = self.speculative_num_steps * self.topk
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
                batch.tree_cache,
                num_seqs * alloc_len,
                backup_state=True,
            )
        else:
            if self.topk == 1:
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                seq_lens_cpu = batch.seq_lens_cpu + self.speculative_num_steps
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:
                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.num_new_pages_per_topk,
                    self.extend_lens,
                    _last_page_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                    self.topk,
                    self.page_size,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                last_page_lens_cpu = prefix_lens_cpu % self.page_size
                num_new_pages_per_topk = (
                    last_page_lens_cpu + self.speculative_num_steps + self.page_size - 1
                ) // self.page_size
                seq_lens_cpu = (
                    prefix_lens_cpu // self.page_size * self.page_size
                    + num_new_pages_per_topk * (self.page_size * self.topk)
                )
                extend_num_tokens = torch.sum((seq_lens_cpu - prefix_lens_cpu)).item()

            out_cache_loc, token_to_kv_pool_state_backup = (
                alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )
        try:
            self._apply_tree_mapping(batch, out_cache_loc)
            return token_to_kv_pool_state_backup, None
        except Exception:
            if token_to_kv_pool_state_backup is not None:
                self.token_to_kv_pool_allocator.restore_state(
                    token_to_kv_pool_state_backup
                )
            raise

    def _try_alloc_lease_tree_kv(self, batch: "ScheduleBatch"):
        if not self._lease_supported():
            return None
        seq_cpu = [int(x) for x in batch.seq_lens_cpu.tolist()]
        layout = plan_paged_tree_layout(
            seq_cpu, self.page_size, self.topk, self.speculative_num_steps
        )
        need = int(sum(layout.pages_per_req))
        store = getattr(self.scheduler, "sr_tree_leases", None)
        if store is not None:
            store.poll_pending_frees(self.token_to_kv_pool_allocator)
        live = int(store.live_pages()) if store is not None else 0
        free = immediate_free_pages(self.token_to_kv_pool_allocator)
        if not lease_budget_ok(free, live, need):
            if store is not None:
                store.reclaim_idle(self.token_to_kv_pool_allocator)
                live = int(store.live_pages())
                free = immediate_free_pages(self.token_to_kv_pool_allocator)
        if not lease_budget_ok(free, live, need):
            if store is not None:
                store.counts["tree_lease_skip_budget"] += 1
            return None
        allocated = self.token_to_kv_pool_allocator.alloc(need * self.page_size)
        if allocated is None:
            if store is not None:
                store.counts["tree_lease_skip_budget"] += 1
            return None
        try:
            mapping = batch.req_to_token_pool.req_to_token
            raw_parts = []
            lease_state = {
                "reqs": list(batch.reqs),
                "starts": [],
                "extend": layout.extend_lens,
                "mapping_snaps": [],
                "page_counts": [],
                "page_slots": [],
                "remainders": layout.remainders,
            }
            offset = 0
            ps = int(self.page_size)
            for b, req in enumerate(batch.reqs):
                n_pages = layout.pages_per_req[b]
                chunk = allocated[ps * offset : ps * (offset + n_pages)]
                offset += n_pages
                r = layout.remainders[b]
                raw_parts.append(chunk[r:])
                start = layout.prefix_lens[b]
                ext = layout.extend_lens[b]
                snap = mapping[req.req_pool_idx, start : start + ext].clone()
                lease_state["starts"].append(start)
                lease_state["mapping_snaps"].append(snap)
                lease_state["page_counts"].append(n_pages)
                lease_state["page_slots"].append(chunk)
            raw_cache_loc = torch.cat(raw_parts) if raw_parts else allocated[:0]
            self.extend_lens = torch.tensor(
                layout.extend_lens, dtype=torch.int64, device=self.device
            )
            self.num_new_pages_per_topk = torch.tensor(
                layout.pages_per_branch, dtype=torch.int64, device=self.device
            )
            return raw_cache_loc, lease_state
        except Exception:
            self.token_to_kv_pool_allocator.free(allocated)
            raise

    def _apply_tree_mapping(self, batch: "ScheduleBatch", raw_cache_loc):
        num_seqs = batch.batch_size()
        raw_cache_loc, draft_cache_loc = split_draft_cache_locs(
            raw_cache_loc,
            num_seqs,
            self.topk,
            self.speculative_num_steps,
            self.page_size,
        )
        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens,
            raw_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
            self.page_size,
            next_power_of_2(num_seqs),
        )
        if self.page_size > 1 and self.topk > 1:
            draft_cache_loc = build_paged_draft_cache_locs(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                batch.seq_lens,
                self.num_new_pages_per_topk,
                self.topk,
                self.speculative_num_steps,
                self.page_size,
            )
        batch.out_cache_loc = draft_cache_loc
        batch.seq_lens_sum = seq_lens_sum_from_batch(batch)
        batch.spec_info.positions = batch.seq_lens.repeat_interleave(
            self.topk, dim=0
        )

    def _restore_tree_mapping(self, batch: "ScheduleBatch", lease_state) -> None:
        if not lease_state:
            return
        mapping = batch.req_to_token_pool.req_to_token
        for req, start, snap in zip(
            lease_state["reqs"], lease_state["starts"], lease_state["mapping_snaps"]
        ):
            if req.req_pool_idx is None:
                continue
            mapping[req.req_pool_idx, start : start + snap.numel()].copy_(snap)

    def _free_lease_alloc(self, lease_state) -> None:
        if not lease_state:
            return
        for slots in lease_state["page_slots"]:
            self.token_to_kv_pool_allocator.free(slots)

    def _draft_forward(self, forward_batch: ForwardBatch):
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            None,
        )
        maybe_detect_nan(topk_p, "SR draft_forward: NaN in seed topk_p")
        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )
        rows = int(out_cache_loc.shape[1])
        self._ensure_identity_capacity(rows)
        self._slot_node_ids.fill_(-1)
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info, parent_rows = (
                select_top_k_tokens(
                    i, topk_p, topk_index, hidden_states, scores, self.topk
                )
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])
            if i == self.speculative_num_steps - 1:
                break
            if i > 0 and self.topk > 1 and parent_rows is not None:
                self._remap_tree_kv_to_parents(out_cache_loc, parent_rows, i)
            if i == 0:
                self._slot_node_ids[i, :rows].copy_(self._step0_node_ids[:rows])
            else:
                self._slot_node_ids[i, :rows].copy_(
                    later_forward_node_ids(tree_info[2])[:rows]
                )
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            advance_tree_draft_positions_for_step(
                i,
                forward_batch.positions,
                getattr(forward_batch, "mrope_positions", None),
            )
            if self.draft_attn_backend is not None:
                forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = None
            # Tree steps are bs*topk, not 1-token AR. DECODE graphs must not replay
            # here (capture would hit unset raw_num_token; runtime layout is wrong).
            prev_graph_runner = getattr(self.draft_model_runner, "graph_runner", None)
            self.draft_model_runner.graph_runner = None
            try:
                logits_output = self.draft_model_runner.forward(
                    forward_batch, skip_attn_backend_init=True
                ).logits_output
            finally:
                self.draft_model_runner.graph_runner = prev_graph_runner
            maybe_detect_nan(logits_output.next_token_logits, f"SR draft_forward step {i}")
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"SR draft_forward step {i}: topk_index OOB",
            )
            hidden_states = None
        self._last_out_cache_loc = out_cache_loc
        return organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

    def _remap_tree_kv_to_parents(
        self,
        out_cache_loc: torch.Tensor,
        parent_rows: torch.Tensor,
        n_prev_steps: int,
    ) -> None:
        if n_prev_steps <= 0:
            return
        rows = int(out_cache_loc.shape[1])
        if int(parent_rows.numel()) != rows:
            raise RuntimeError("parent_rows width must match out_cache_loc rows")
        remap_slot_node_ids(
            self._slot_node_ids, parent_rows, n_prev_steps, self._slot_node_id_tmp
        )
        kv_pool = getattr(self.draft_model_runner, "token_to_kv_pool", None)
        if kv_pool is None:
            return
        hist = out_cache_loc[:n_prev_steps]
        src = hist[:, parent_rows.to(dtype=torch.int64)].reshape(-1)
        tgt = hist.reshape(-1)
        self._copy_tree_kv_slots(kv_pool, src, tgt)

    def _copy_tree_kv_slots(self, kv_pool, src: torch.Tensor, tgt: torch.Tensor) -> None:
        copy_kv_pool_by_slot(kv_pool, src, tgt)
