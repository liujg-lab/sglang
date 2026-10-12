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

from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
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
    IdentityRemapWorkspace,
    SRTreeKVLease,
    first_forward_node_ids,
    immediate_free_pages,
    later_forward_node_ids,
    lease_budget_ok,
    plan_paged_tree_layout,
    prefix_window_tokens,
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
from sglang.srt.speculative.standalone_remote.sr_kv_copy import (
    KVMoveSubmittedError,
    prepare_kv_move,
    remap_tree_kv_,
    warm_private_slot_move,
)
from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
    SRTransferUnresolved,
)
from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
    advance_tree_draft_positions_for_step,
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
        # Capacity tensors stay here. Public attributes are views of ``[:n]``.
        self._span_caps = {}
        self._span_hosts = {}
        self._span_host_pinned = {}
        self._span_events = {}
        self._span_blocked = set()
        self._span_hold = {}
        self._raw_cache_inflight = False
        self.draft_attn_backend = None
        self.cuda_graph_runner = None
        self._tree_failure_counts = {}
        self._tree_batch_isolate_count = 0
        self._tree_forward_calls = 0
        self._seen_tree_graph_keys = set()
        self.tree_graph_capture_succeeded = False
        self.tree_graph_disabled_reason = None
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
        self._identity_table_frozen = False
        self._identity_table_key = None
        self._identity_graph = None
        self._active_identity_workspace = None
        self._identity_eager = IdentityRemapWorkspace(
            id_dev,
            self.speculative_num_steps,
            self._identity_rows,
            graph=False,
            domain="tree_eager",
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

    def _identity_storage_key(self):
        table = self._slot_node_ids
        return (
            table.data_ptr(),
            tuple(table.shape),
            tuple(table.stride()),
            table.dtype,
            table.device,
        )

    def _freeze_identity_table(self) -> None:
        self._identity_table_frozen = True
        self._identity_table_key = self._identity_storage_key()

    def _check_identity_table(self) -> None:
        if not getattr(self, "_identity_table_frozen", False):
            return
        if self._identity_storage_key() != self._identity_table_key:
            raise RuntimeError(
                "identity table storage was replaced after graph binding"
            )

    def _ensure_identity_capacity(self, rows: int) -> None:
        rows = max(int(rows), 1)
        self._check_identity_table()
        capturing = getattr(self, "_capture_identity_workspace", None) is not None
        table = self._slot_node_ids
        if table is not None and int(table.shape[1]) >= rows:
            if not capturing:
                self._identity_eager.reserve_rows(rows)
            return
        if capturing or getattr(self, "_identity_table_frozen", False):
            raise RuntimeError(
                "identity table cannot grow during capture or after graph binding"
            )
        device = table.device if table is not None else self.device
        # Grow scratch first. A failed allocation leaves the live table in place.
        self._identity_eager.reserve_rows(rows)
        steps = self.speculative_num_steps
        self._slot_node_ids = torch.full(
            (steps, rows), -1, dtype=torch.int64, device=device
        )
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

    def prepare_tree_candidate_graph(self, batch_cap, seed):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_candidates import candidate_workspace

        return candidate_workspace(self, batch_cap, seed, graph=True)

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
                if getattr(self, "_identity_graph", None) is not None:
                    self._freeze_identity_table()
            logger.info("[SR] Capture tree draft graph end.")
        except (NpuGraphReplaySubmittedError, SRTransferUnresolved):
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
        except (NpuGraphReplaySubmittedError, SRTransferUnresolved):
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
                                warm_private_slot_move(
                                    kv_pool,
                                    int(src.numel()),
                                    index_dtype=src.dtype,
                                    dst_index_dtype=dst.dtype,
                                    src_stride=src.stride(0),
                                    dst_stride=dst.stride(0),
                                )
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
                            commit_kv=False,
                        )
                        prepare(
                            dummy_fb,
                            slots,
                            prefixes,
                            ALLOC_LEASE,
                            kv_pool,
                            seq_lens=seq_lens,
                            dummy_page=dummy,
                            commit_kv=False,
                        )
                        combos.append(key)
        except BaseException:
            view = getattr(backend, "_sr_draft_paged_view", None)
            workspace = getattr(view, "workspace", None)
            if workspace is not None and (
                workspace.pending_consumer or workspace.unresolved
            ):
                workspace.poison()
            else:
                clear = getattr(backend, "_sr_clear_paged_round_state", None)
                if clear is not None:
                    try:
                        clear()
                    except BaseException:
                        # Keep the primary warmup error, including interrupts.
                        # Failed cleanup retains the backend's remaining refs.
                        try:
                            logger.exception(
                                "[SR] secondary paged warmup cleanup failure"
                            )
                        except BaseException:
                            pass
            raise
        else:
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
        if not getattr(self, "sr_tree_paged", False):
            return
        t0 = time.perf_counter()
        combos = []
        alloc_keys = []
        mapping_keys = []
        reply_pack_s = 0.0
        try:
            # Lease sources are assembled as int64, while destinations retain
            # the request mapping dtype. Warm that distinct pointer signature.
            warm_private_slot_move(
                self.draft_model_runner.token_to_kv_pool,
                index_dtype=torch.int64,
                dst_index_dtype=self.req_to_token_pool.req_to_token.dtype,
            )
            combos = self._sr_warm_layout_shapes()
            coverage = warm_draft_alloc_mapping(self)
            alloc_keys = coverage.get("allocation") or []
            mapping_keys = coverage.get("mapping") or []
            from sglang.srt.speculative.standalone_remote.drafter.sr_tree_reply_pack import (
                warm_tree_reply_pack,
            )
            from sglang.srt.speculative.standalone_remote.sr_warmup import (
                warmup_synchronize,
            )

            t_pack = time.perf_counter()
            # Hold the sources across the sync. The first launch compiles.
            reply_holds = warm_tree_reply_pack(
                self.device,
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )
            warmup_synchronize(self.device)
            reply_pack_s = time.perf_counter() - t_pack
            del reply_holds
        except (NpuGraphReplaySubmittedError, KVMoveSubmittedError, SRWarmupFatalError):
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
            "[SR] tree warmup layout=%s allocation=%s mapping=%s "
            "reply_pack=%.3fs elapsed=%.3fs",
            combos,
            alloc_keys,
            mapping_keys,
            reply_pack_s,
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
        except (NpuGraphReplaySubmittedError, SRTransferUnresolved):
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
        scheduler = getattr(self, "scheduler", None)
        metrics = (
            getattr(scheduler, "_sr_round_metrics", None) if scheduler is not None else None
        )
        try:
            with metrics.phase("tree_result_wait_pack") if metrics else nullcontext():
                self._pack_tree_windows(
                    parent_list, top_scores_index, draft_tokens, keep_idx, windows, keep
                )
        except Exception as exc:
            self._abort_tree_pack(exc)
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
        except (NpuGraphReplaySubmittedError, SRTransferUnresolved):
            raise
        except Exception as e:
            if is_device_context_error(e):
                raise
            self._log_tree_failure("expand_one", e)
            return empty
        windows: List[SRTreeWindow] = [empty]
        scheduler = getattr(self, "scheduler", None)
        metrics = (
            getattr(scheduler, "_sr_round_metrics", None) if scheduler is not None else None
        )
        try:
            with metrics.phase("tree_result_wait_pack") if metrics else nullcontext():
                self._pack_tree_windows(
                    parent_list, top_scores_index, draft_tokens, [0], windows, [req]
                )
        except Exception as exc:
            self._abort_tree_pack(exc)
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
        """Pack tree tensors on device, one D2H, then CPU row splits into RPC lists."""
        tokens_cpu, parents_cpu, indices_cpu, slots_cpu = self._d2h_tree_outputs(
            draft_tokens, parent_list, top_scores_index
        )
        token_rows = tokens_cpu.tolist()
        parent_rows = parents_cpu.tolist()
        index_rows = indices_cpu.tolist()
        packed = []
        for j, idx in enumerate(keep_idx):
            window = (token_rows[j], parent_rows[j], index_rows[j])
            windows[idx] = window
            packed.append(window)
        self._publish_tree_leases(reqs or [], packed, slots_cpu)
        self._release_tree_slot()

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
        slot_rows = slots_cpu.tolist() if slots_cpu is not None else None
        try:
            for j, req in enumerate(lease_state["reqs"]):
                tokens, pl, ix = windows[j] if j < len(windows) else ([], None, None)
                if not tokens or pl is None or ix is None:
                    self.token_to_kv_pool_allocator.free(lease_state["page_slots"][j])
                    unpublished.discard(j)
                    continue
                if slot_rows is not None and j < len(slot_rows):
                    cand = slot_rows[j]
                    if len(cand) < len(ix):
                        cand = cand + [-1] * (len(ix) - len(cand))
                    elif len(cand) > len(ix):
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
                    candidate_slots=cand,
                    parent_list=pl,
                    top_scores_index=ix,
                    draft_tokens=tokens,
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

    def _tree_phase(self, name, device=False):
        from contextlib import nullcontext

        metrics = getattr(getattr(self, "scheduler", None), "_sr_round_metrics", None)
        if metrics is None or not hasattr(metrics, "phase"):
            return nullcontext()
        return metrics.phase(name, device=device)

    def _count_tree(self, name: str, value: int = 1) -> None:
        current = int(getattr(self, "_" + name, 0) or 0)
        setattr(self, "_" + name, current + int(value))
        metrics = getattr(getattr(self, "scheduler", None), "_sr_round_metrics", None)
        if metrics is not None and hasattr(metrics, "counts"):
            metrics.counts[name] += int(value)

    def _tree_result_slots(self):
        from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
            ResultSlot,
        )

        slots = getattr(self, "_tree_slots", None)
        if not slots:
            self._tree_slots = [ResultSlot(), ResultSlot()]
        return self._tree_slots

    def _acquire_tree_slot(self):
        for slot in self._tree_result_slots():
            if slot.state == "free":
                self._active_tree_slot = slot
                return slot
        raise RuntimeError("no free tree staging slot")

    def _mark_tree_slot(self, slot, state: str) -> None:
        slot.state = state
        trace = getattr(self, "_tree_slot_trace", None)
        if trace is None:
            trace = []
            self._tree_slot_trace = trace
        trace.append(state)

    def _release_tree_slot(self) -> None:
        """Return a consumed slot to FREE. Does not run from ``finally``."""
        slot = getattr(self, "_active_tree_slot", None)
        if slot is None or slot.state != "consuming":
            return
        slot.event = None
        slot.src_hold = None
        slot.segments = []
        slot.used = 0
        self._mark_tree_slot(slot, "free")
        self._active_tree_slot = None

    def _abort_tree_pack(self, exc: BaseException) -> None:
        """Keep the lease when completion is unknown. Confirmed work may clean up."""
        from sglang.srt.speculative.standalone_remote.sr_align import (
            is_device_context_error,
        )

        slot = getattr(self, "_active_tree_slot", None)
        state = getattr(slot, "state", None)
        if state == "unresolved" or is_device_context_error(exc):
            raise exc
        if state == "consuming":
            self._release_tree_slot()
        lease_state = getattr(self, "_pending_lease_state", None)
        self._pending_lease_state = None
        self._free_lease_alloc(lease_state)
        raise exc

    def _d2h_tree_outputs(self, draft_tokens, parent_list, top_scores_index):
        from sglang.srt.speculative.standalone_remote.sr_align import (
            is_device_context_error,
        )
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_reply_pack import (
            pack_tree_reply,
            prepare_tree_reply,
            record_reply_inputs,
        )
        from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
            SRTransferUnresolved,
            cross_device_d2h,
            submit_copy,
            wait_event,
        )

        tensors = (draft_tokens, parent_list, top_scores_index)
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            self._active_tree_slot = None
            return (
                draft_tokens.detach().to("cpu"),
                parent_list.detach().to("cpu"),
                top_scores_index.detach().to("cpu"),
                None,
            )
        tokens = _as_2d(draft_tokens.detach())
        parents = _as_2d(parent_list.detach())
        indices = _as_2d(top_scores_index.detach())
        lease_state = getattr(self, "_pending_lease_state", None)
        compact = None
        node_ids = None
        write_slots = False
        if lease_state is not None:
            compact = getattr(self, "_lease_compact_slots", None)
            node_ids = getattr(self, "_slot_node_ids", None)
            if (
                compact is not None
                and node_ids is not None
                and int(indices.numel()) > 0
            ):
                write_slots = True
        batch = int(tokens.shape[0])
        # No-lease replies never touch the lease tables. Reading topk only when
        # slots are packed keeps callers that do not set those attributes working.
        topk = int(self.topk) if write_slots else 1
        steps = int(self.speculative_num_steps) if write_slots else 1
        segments, needed = prepare_tree_reply(
            tokens,
            parents,
            indices,
            node_ids if write_slots else None,
            compact if write_slots else None,
            batch=batch,
            topk=topk,
            steps=steps,
            write_slots=write_slots,
        )
        slot = self._acquire_tree_slot()
        self._mark_tree_slot(slot, "free")
        if slot.grow(needed, tokens.device):
            self._count_tree("tree_staging_grow")
        if slot.pinned:
            metrics = getattr(getattr(self, "scheduler", None), "_sr_round_metrics", None)
            if metrics is not None and hasattr(metrics, "counts"):
                metrics.counts["tree_pinned"] = 1
        holds = [tokens, parents, indices]
        if write_slots:
            holds.extend((node_ids, compact))
        try:
            with self._tree_phase("tree_pack_device", device=True):
                if needed > 0:
                    pack_tree_reply(
                        slot.device_buf,
                        tokens,
                        parents,
                        indices,
                        node_ids if write_slots else None,
                        compact if write_slots else None,
                        batch=batch,
                        topk=topk,
                        steps=steps,
                        write_slots=write_slots,
                    )
                    self._count_tree("tree_device_pack_copies")
                record_reply_inputs(holds, tokens.device)
        except Exception as exc:
            if is_device_context_error(exc):
                record_reply_inputs(holds, tokens.device)
                slot.src_hold = holds
                slot.segments = segments
                slot.used = needed
                self._mark_tree_slot(slot, "unresolved")
                raise SRTransferUnresolved("device pack failed") from exc
            raise
        slot.src_hold = holds
        slot.segments = segments
        slot.used = needed
        if needed <= 0:
            self._mark_tree_slot(slot, "consuming")
            return tokens[:0], parents[:0], indices[:0], None
        host = slot.host_buf[:needed]
        device_src = slot.device_buf[:needed]
        self._mark_tree_slot(slot, "in_flight")
        try:
            with self._tree_phase("tree_d2h_submit"):
                event = submit_copy(host, device_src)
        except Exception:
            self._mark_tree_slot(slot, "unresolved")
            raise
        self._count_tree("tree_host_payload_submits")
        self._count_tree("tree_d2h_count")
        self._count_tree("tree_d2h_bytes", needed * 8)
        if cross_device_d2h(device_src, host):
            self._count_tree("tree_cross_device_d2h")
        slot.event = event
        try:
            # Recorded after the pack on this stream, so the wait also covers
            # earlier Draft compute. It is not only the copy duration.
            with self._tree_phase("tree_d2h_wait"):
                wait_event(event)
        except Exception:
            self._mark_tree_slot(slot, "unresolved")
            raise
        self._mark_tree_slot(slot, "consuming")
        with self._tree_phase("tree_unpack_cpu"):
            return self._views_from_tree_slot(slot)

    def _views_from_tree_slot(self, slot):
        views = {}
        host = slot.host_buf
        for name, shape, start, length in slot.segments:
            views[name] = host[start : start + length].reshape(shape)
        empty = host[:0].reshape(0, 0)
        return (
            views.get("tokens", empty),
            views.get("parents", empty),
            views.get("indices", empty),
            views.get("candidate_slots"),
        )

    def _stack_seeds(
        self, reqs: List["Req"]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        from sglang.srt.speculative.standalone_remote.drafter.sr_tail_extend import (
            stack_recorded_seeds,
        )

        return stack_recorded_seeds(reqs)

    def _expand_tree(
        self,
        reqs: List["Req"],
        topk_p: torch.Tensor,
        topk_index: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        verified_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del hidden_states
        self._sr_last_abandon_has_event = False
        # A previous abandon may already have a completed consumer event.
        self._reclaim_abandoned_leases()
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
        self._sr_last_abandon_has_event = False
        with metrics.phase("tree_alloc_kv") if metrics else nullcontext():
            token_to_kv_pool_state_backup, lease_state = self._alloc_tree_kv(batch)
        # Captured here so a later private pack cannot change the seal decision.
        # AST tests bind this method onto a namespace that does not carry the helper.
        shared_raw = bool(getattr(self, "_raw_cache_shared", False))
        seal_raw_cache = getattr(self, "_seal_raw_cache", None)
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
            paged_graph_selected = None
            if (
                getattr(self, "sr_tree_paged", False)
                and batch.out_cache_loc.device.type == "npu"
            ):
                # Admission reads CPU lengths and captured geometry only. Select
                # the final output addresses before writing any paged metadata.
                paged_graph_selected = self._can_run_tree_graph(forward_batch)
                runner = getattr(self, "cuda_graph_runner", None)
                plan = (
                    getattr(runner, "_tree_replay_plan", None)
                    if paged_graph_selected
                    else None
                )
                if plan is not None:
                    try:
                        runner._assert_tree_replay_graph(plan)
                        self.draft_attn_backend._paged_graph_table_view(
                            plan.capture_bs, plan.kv_bucket, 0
                        )
                        if (plan.capture_bs, plan.kv_bucket) not in getattr(
                            self.draft_attn_backend, "_sr_paged_graph_views", {}
                        ):
                            raise NpuGraphPreparationError(
                                "missing captured metadata workspace", scope="graph"
                            )
                    except NpuGraphPreparationError:
                        runner._clear_tree_replay_plan()
                        runner._last_can_run_reject = "metadata_preparation"
                        paged_graph_selected, plan = False, None
                        if metrics is not None:
                            metrics.counts[
                                "paged_metadata_draft_admission_fallback"
                            ] += 1
                forward_batch._sr_paged_replay_plan = plan
            if getattr(self, "sr_tree_paged", False):
                txn.mark_copy_begin()
                prepare = getattr(self, "_prepare_paged_tree_round", None)
                if prepare is not None:
                    prepare(forward_batch, batch, lease_state is not None, metrics)
            paged_s = time.perf_counter() - t_paged
            prep_s = time.perf_counter() - t_prep
            if metrics is not None:
                metrics.add_host("tree_init_forward_batch", init_s)
                metrics.add_host("tree_paged_eager", paged_s)
                metrics.add_host("tree_prepare_meta", prep_s)
            can_fn = getattr(self, "_can_run_tree_graph", None)
            if paged_graph_selected is not None:
                can_cuda_graph = paged_graph_selected
            elif can_fn is not None:
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
                    except (NpuGraphReplaySubmittedError, SRTransferUnresolved):
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
                from sglang.srt.speculative.standalone_remote.sr_paged_metadata import (
                    finish_paged_metadata,
                )

                finish_paged_metadata(self.draft_attn_backend)
            exec_s = time.perf_counter() - t_exec
            txn.completion_confirmed = True
            if callable(seal_raw_cache):
                seal_raw_cache(shared_raw)
        except (NpuGraphReplaySubmittedError, SRTransferUnresolved):
            from sglang.srt.speculative.standalone_remote.sr_paged_metadata import (
                poison_paged_metadata,
            )

            poison_paged_metadata(self.draft_attn_backend)
            graph_submitted = True
            raise
        except Exception as e:
            from sglang.srt.speculative.standalone_remote.sr_paged_metadata import (
                finish_paged_metadata,
                poison_paged_metadata,
            )

            ctx_err = globals().get("is_device_context_error")
            if ctx_err is not None and ctx_err(e):
                poison_paged_metadata(self.draft_attn_backend)
                graph_submitted = True
                raise NpuGraphReplaySubmittedError(
                    "tree expand device context error"
                ) from e
            if txn.in_flight():
                confirm = getattr(self, "_try_confirm_tree_completion", lambda: True)
                if not confirm():
                    poison_paged_metadata(self.draft_attn_backend)
                    graph_submitted = True
                    raise SRTransferUnresolved(
                        "tree expand in-flight; refuse rollback"
                    ) from e
                txn.completion_confirmed = True
                # Device sync finished the forward that still reads raw slots.
                if callable(seal_raw_cache):
                    seal_raw_cache(shared_raw)
                # A successful device confirmation also closes metadata use on
                # this recoverable error path, before rollback frees pages.
                finish_paged_metadata(self.draft_attn_backend)
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
            if (
                not abandon
                and not getattr(txn, "tree_compute_submitted", False)
                and getattr(self, "_raw_cache_inflight", False)
            ):
                # Assign already queued, and no forward was submitted.
                # A private pack must not clear an older shared inflight.
                if callable(seal_raw_cache):
                    seal_raw_cache(shared_raw)
            if abandon:
                # A missing helper must not replace the replay exception already
                # in flight. AST tests bind expand onto a namespace without it.
                abandon_lease = getattr(self, "_abandon_tree_lease", None)
                if callable(abandon_lease):
                    abandon_lease(lease_state)
                self._pending_lease_state = None
            elif txn.rolled_back:
                pass
            elif lease_state is not None and self._pending_lease_state is not None:
                self._restore_tree_mapping(batch, lease_state)
            elif token_to_kv_pool_state_backup is not None:
                self.token_to_kv_pool_allocator.restore_state(
                    token_to_kv_pool_state_backup
                )
        if not abandon:
            self._reclaim_abandoned_leases()
        runner = self.cuda_graph_runner
        replay_n = getattr(runner, "tree_graph_replay_count", 0) if runner else 0
        self._tree_forward_calls = getattr(self, "_tree_forward_calls", 0) + 1
        calls = self._tree_forward_calls
        reason = (
            None
            if can_cuda_graph
            else (getattr(runner, "_last_can_run_reject", None) or "graph_unavailable")
        )
        if calls <= 1 or calls % 8 == 0:
            logger.debug(
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
        self._raw_cache_shared = False
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
            shared_raw = bool(getattr(self, "_raw_cache_shared", False))
            if shared_raw:
                self._raw_cache_owner = lease_state
            try:
                self._apply_tree_mapping(batch, raw_cache_loc)
                # The speculative image is saved before any device work. The
                # abandon path must not clone again.
                self._capture_abandoned_spec_image(lease_state)
                return None, lease_state
            except Exception:
                self._restore_tree_mapping(batch, lease_state)
                self._free_lease_alloc(lease_state)
                if getattr(self, "_raw_cache_inflight", False) and getattr(
                    self, "_raw_cache_owner", None
                ) is lease_state:
                    self._seal_raw_cache(shared_raw)
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
        lease_state = {
            "reqs": list(batch.reqs),
            "starts": [],
            "extend": layout.extend_lens,
            "mapping_snaps": [],
            "page_counts": [],
            "page_slots": [],
            "remainders": layout.remainders,
        }
        try:
            mapping = batch.req_to_token_pool.req_to_token
            raw_parts = []
            snap_rows = []
            row_meta = []
            offset = 0
            ps = int(self.page_size)
            for b, req in enumerate(batch.reqs):
                n_pages = layout.pages_per_req[b]
                chunk = allocated[ps * offset : ps * (offset + n_pages)]
                offset += n_pages
                r = layout.remainders[b]
                raw_parts.append(chunk[r:])
                start = int(layout.prefix_lens[b])
                ext = int(layout.extend_lens[b])
                snap_rows.append(mapping[req.req_pool_idx, start : start + ext])
                row_meta.append((start, n_pages, chunk))
            raw_cache_loc, _, _ = self._pack_device_rows(
                "raw_cache_loc",
                raw_parts,
                allocated.device,
                dtype=allocated.dtype,
            )
            _, snap_views, shared = self._pack_device_rows(
                "mapping_snap", snap_rows, mapping.device
            )
            if shared:
                self._span_hold["mapping_snap"] = lease_state
            for (start, n_pages, chunk), snap in zip(row_meta, snap_views):
                lease_state["starts"].append(start)
                lease_state["mapping_snaps"].append(snap)
                lease_state["page_counts"].append(n_pages)
                lease_state["page_slots"].append(chunk)
            self.extend_lens = self._reuse_long(
                "extend_lens", layout.extend_lens, self.device
            )
            self.num_new_pages_per_topk = self._reuse_long(
                "num_new_pages_per_topk", layout.pages_per_branch, self.device
            )
            return raw_cache_loc, lease_state
        except Exception:
            if self._span_hold.get("mapping_snap") is lease_state:
                self._span_hold["mapping_snap"] = None
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
        # Mark the shared raw buffer live before either kernel is queued, so a
        # failure halfway through still refuses to overwrite it.
        if getattr(self, "_raw_cache_shared", False):
            self._raw_cache_inflight = True
        try:
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
        finally:
            # Covers assign/build even when the second kernel raises.
            self._note_span_consumer("extend_lens", "num_new_pages_per_topk")

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
        release = getattr(self, "_release_mapping_snap", None)
        if callable(release):
            release(lease_state)

    def _restore_abandoned_mapping(self, lease_state) -> None:
        """Undo an abandoned window without aliasing committed pages.

        A slot that still names the abandoned speculative page holds draft-tree
        KV, including rejected siblings. Write that slot back to the pre-expand
        snapshot even when it sits below ``kv_committed_len``. A slot that no
        longer matches the speculative image was written by a later tail and
        stays. Without an image, the whole snap is restored.

        ``copy_`` is queued on the mapping device. The copy is finished before
        return so a following free cannot recycle the pages while a reader
        still observes the abandoned ids.
        """
        pool = getattr(self, "req_to_token_pool", None)
        mapping = getattr(pool, "req_to_token", None)
        if mapping is None:
            raise RuntimeError("abandoned tree mapping is missing")
        specs = lease_state.get("spec_snaps") if lease_state else None
        if not specs:
            self._restore_tree_mapping(
                SimpleNamespace(req_to_token_pool=pool), lease_state
            )
        else:
            rows = zip(
                lease_state.get("reqs") or [],
                lease_state.get("starts") or [],
                lease_state.get("mapping_snaps") or [],
                specs,
            )
            for req, start, snap, spec in rows:
                if req.req_pool_idx is None:
                    continue
                n = int(snap.numel())
                view = mapping[req.req_pool_idx, start : start + n]
                image = spec.to(device=view.device, dtype=view.dtype)
                saved = snap.to(device=view.device, dtype=view.dtype)
                if image.numel() != n or saved.numel() != n or view.numel() != n:
                    continue
                speculative = view.eq(image)
                view.copy_(torch.where(speculative, saved, view))
        device = mapping.device
        if getattr(device, "type", "cpu") != "cpu":
            module = torch.get_device_module(device.type)
            try:
                module.synchronize(device)
            except TypeError:
                module.synchronize()
        release = getattr(self, "_release_mapping_snap", None)
        if callable(release):
            release(lease_state)

    def _release_mapping_snap(self, lease_state) -> None:
        """Return the shared snap buffer after its restore copy has been queued."""
        if not lease_state:
            return
        lease_state["mapping_restored"] = True
        if self._span_hold.get("mapping_snap") is lease_state:
            self._span_hold["mapping_snap"] = None
            self._note_span_consumer("mapping_snap")

    def _span_reusable(self, name: str) -> bool:
        if name in self._span_blocked:
            raise SRTransferUnresolved(f"{name} staging is unresolved")
        if name == "raw_cache_loc" and getattr(self, "_raw_cache_inflight", False):
            return False
        hold = self._span_hold.get(name)
        if hold is not None and not hold.get("mapping_restored"):
            return False
        return True

    def _wait_span_consumer(self, name: str) -> None:
        if name in self._span_blocked:
            raise SRTransferUnresolved(f"{name} staging is unresolved")
        event = self._span_events.get(name)
        if event is None:
            return
        try:
            done = bool(event.query())
        except Exception as exc:
            self._span_blocked.add(name)
            raise SRTransferUnresolved(f"{name} consumer query failed") from exc
        if done:
            return
        try:
            event.synchronize()
        except Exception as exc:
            self._span_blocked.add(name)
            raise SRTransferUnresolved(f"{name} consumer wait failed") from exc

    def _note_span_consumer(self, *names: str) -> None:
        from .sr_tree_kv_lease import record_device_event

        event = record_device_event(getattr(self, "device", None), required=False)
        dev = getattr(self, "device", None)
        dev_type = getattr(dev, "type", None) or (str(dev) if dev is not None else "cpu")
        if event is None and dev_type != "cpu" and "cpu" not in str(dev_type):
            for name in names:
                self._span_blocked.add(name)
            return
        for name in names:
            self._span_events[name] = event

    def _seal_raw_cache(self, shared_raw: bool) -> None:
        """Record the raw-slot consumer for this expand.

        ``shared_raw`` is the pack that owns the shared cap, captured before a
        later private pack can clear ``_raw_cache_shared``. A private note must
        leave an older inflight set.
        """
        self._note_span_consumer("raw_cache_loc")
        if shared_raw:
            self._raw_cache_inflight = False

    def _grow_span_cap(self, name: str, n: int, device, dtype) -> torch.Tensor:
        caps = self._span_caps
        cap = caps.get(name)
        dev = torch.device(device)
        if (
            not torch.is_tensor(cap)
            or int(cap.numel()) < int(n)
            or cap.device != dev
            or cap.dtype != dtype
        ):
            cap = torch.empty(max(int(n), 1), dtype=dtype, device=dev)
            caps[name] = cap
        return cap

    def _pack_device_rows(self, name: str, rows, device, dtype=None):
        """Copy device rows into one buffer at offsets. Views share that buffer.

        A shared cap is reused only after the previous consumer event. A lease
        that still owns the cap, or an in-flight raw buffer, gets a private
        tensor so the previous reader is not overwritten.
        """
        rows = list(rows)
        dev = torch.device(device)
        if rows:
            dtype = rows[0].dtype
            dev = rows[0].device
        elif dtype is None:
            dtype = torch.int64
        total = sum(int(row.numel()) for row in rows)
        shared = self._span_reusable(name)
        if name == "raw_cache_loc":
            self._raw_cache_shared = shared
        if shared:
            self._wait_span_consumer(name)
            backing = self._grow_span_cap(name, max(total, 1), dev, dtype)
        else:
            backing = torch.empty(max(total, 1), dtype=dtype, device=dev)
        offset = 0
        views = []
        for row in rows:
            n = int(row.numel())
            if n:
                src = row
                if src.device != dev or src.dtype != dtype:
                    src = src.to(device=dev, dtype=dtype)
                backing[offset : offset + n].copy_(src)
            views.append(backing[offset : offset + n])
            offset += n
        return backing[:total], views, shared

    def _reuse_long(self, name, values, device) -> torch.Tensor:
        """Copy a host list into ``_span_caps[name]`` and return a ``[:n]`` view.

        The capacity tensor is not the returned view. The next call waits for
        the consumer recorded by ``_note_span_consumer`` before writing again.
        """
        from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
            alloc_host,
        )

        self._wait_span_consumer(name)
        n = len(values)
        dev = torch.device(device)
        cap = self._grow_span_cap(name, max(n, 1), dev, torch.int64)
        host = self._span_hosts.get(name)
        pinned = bool(self._span_host_pinned.get(name, False))
        if host is None or int(host.numel()) < n or host.dtype != torch.int64:
            host, pinned = alloc_host((max(n, 1),), torch.int64, dev)
            self._span_hosts[name] = host
            self._span_host_pinned[name] = pinned
        if n:
            try:
                host[:n].numpy()[:] = values
            except Exception:
                for i, value in enumerate(values):
                    host[i] = int(value)
            cap[:n].copy_(host[:n], non_blocking=bool(pinned and dev.type != "cpu"))
        return cap[:n]

    def _free_lease_alloc(self, lease_state) -> None:
        if not lease_state:
            return
        for slots in lease_state["page_slots"]:
            self.token_to_kv_pool_allocator.free(slots)

    def _capture_abandon_event(self):
        """Record the consumer that still owns these pages. Failure poisons."""
        from .sr_resource_quarantine import HostCompletion

        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None and getattr(scheduler, "_sr_device_poisoned", False):
            return None
        device = getattr(self, "device", None)
        dev_type = getattr(device, "type", None) or (
            str(device) if device is not None else "cpu"
        )
        if device is None or "cpu" in str(dev_type):
            return HostCompletion()
        try:
            from .sr_tree_kv_lease import record_device_event

            return record_device_event(device, required=True)
        except BaseException:
            if scheduler is not None:
                scheduler._sr_device_poisoned = True
            return None

    def _capture_abandoned_spec_image(self, lease_state) -> None:
        """Clone the mapping slice the abandoned expand currently occupies."""
        if not lease_state or lease_state.get("spec_snaps") is not None:
            return
        pool = getattr(self, "req_to_token_pool", None)
        mapping = getattr(pool, "req_to_token", None)
        snaps = lease_state.get("mapping_snaps")
        if mapping is None or not snaps:
            return
        images = []
        rows = zip(
            lease_state.get("reqs") or [],
            lease_state.get("starts") or [],
            snaps,
        )
        for req, start, snap in rows:
            n = int(snap.numel())
            idx = getattr(req, "req_pool_idx", None)
            if idx is None:
                images.append(snap.detach().clone())
                continue
            images.append(mapping[idx, start : start + n].detach().clone())
        lease_state["spec_snaps"] = images

    def _abandon_tree_lease(self, lease_state) -> None:
        """Retain this lease, then free earlier ones whose events have completed."""
        import sys

        if lease_state:
            self._remember_abandoned_lease(lease_state)
        unwinding = sys.exc_info()[0] is not None
        try:
            self._reclaim_abandoned_leases()
        except BaseException:
            if not unwinding:
                raise

    def _remember_abandoned_lease(self, lease_state) -> None:
        if not lease_state:
            return
        from .sr_resource_quarantine import quarantine

        # Retain the lease, pages and request row before any further device
        # work. A clone or event failure must not erase this record or replace
        # the exception already in flight.
        bag = quarantine(self)
        before = len(bag.records)
        record = bag.retain(
            lease_state,
            lease_state.get("page_slots") or [],
            lease_state.get("completion_event"),
        )
        alloc = getattr(self, "token_to_kv_pool_allocator", None)
        generation = (
            int(getattr(alloc, "generation", 0) or 0) if alloc is not None else 0
        )
        identity = [
            (id(req), getattr(req, "req_pool_idx", None))
            for req in lease_state.get("reqs") or []
        ]
        lease_state["alloc_generation"] = generation
        lease_state["row_identity"] = identity
        record.alloc_generation = generation
        record.row_identity = identity
        try:
            event = lease_state.get("completion_event")
            if event is None:
                event = self._capture_abandon_event()
                if event is not None:
                    lease_state["completion_event"] = event
            record.event = event
            self._sr_last_abandon_has_event = event is not None
        except BaseException:
            record.event = None
            self._sr_last_abandon_has_event = False
        if len(bag.records) != before:
            metrics = getattr(
                getattr(self, "scheduler", None), "_sr_round_metrics", None
            )
            if metrics is not None:
                metrics.counts["tree_lease_abandoned_pages"] += sum(
                    int(slots.numel()) for slots in lease_state.get("page_slots") or []
                )
        # The quarantine record retains the lease. Do not also pin it here.
        backend = getattr(self, "draft_attn_backend", None)
        view = (
            getattr(backend, "_sr_draft_paged_view", None)
            if backend is not None
            else None
        )
        if view is not None:
            view.workspace.lease_hold = None

    def _partition_abandoned_pages(self, record) -> None:
        """Free abandoned KV by physical page.

        ``page_size == 1`` keeps the slot-level split. A larger page is kept
        or released as a whole: any committed-prefix slot on that page retains
        it, and physical page 0 is never released. The same page is queued
        once. A unit whose pages are all unreferenced stays the original
        object. A failed release is left untouched.
        """
        from .sr_resource_quarantine import release_unknown

        if release_unknown(record):
            return
        owner = record.owner
        if not isinstance(owner, dict) or "mapping_snaps" not in owner:
            return
        pool = getattr(self, "req_to_token_pool", None)
        mapping = getattr(pool, "req_to_token", None)
        if mapping is None:
            return
        alloc = getattr(self, "token_to_kv_pool_allocator", None)
        page_size = max(1, int(getattr(alloc, "page_size", 1) or 1))

        def page_of(slot):
            slot = int(slot)
            return slot if page_size == 1 else slot // page_size

        def flat_ids(unit):
            if torch.is_tensor(unit):
                return [int(x) for x in unit.detach().view(-1).tolist()]
            return [int(x) for x in unit]

        def take_ids(unit, ids):
            if torch.is_tensor(unit):
                return torch.tensor(ids, dtype=unit.dtype, device=unit.device)
            return list(ids)

        live_pages = {0}
        width = int(mapping.shape[1])
        starts = owner.get("starts") or []
        for row, req in enumerate(owner.get("reqs") or []):
            idx = getattr(req, "req_pool_idx", None)
            if idx is None:
                return
            committed = getattr(req, "kv_committed_len", None)
            if committed is None:
                committed = int(starts[row]) if row < len(starts) else width
            committed = max(0, min(int(committed), width))
            if committed:
                for slot in mapping[idx, :committed].detach().view(-1).tolist():
                    live_pages.add(page_of(slot))
        states = list(record.states)
        if len(states) < len(record.units):
            states.extend(["unsubmitted"] * (len(record.units) - len(states)))
        unknown = []
        releasable = []
        retained = []
        queued_pages = set()
        for unit, state in zip(record.units, states):
            if state == "unknown":
                unknown.append(unit)
                continue
            if unit is None or state == "done":
                continue
            ids = flat_ids(unit)
            drop = []
            keep = []
            for slot in ids:
                page = page_of(slot)
                if page in live_pages:
                    keep.append(slot)
                elif page in queued_pages:
                    continue
                else:
                    drop.append(slot)
            for slot in drop:
                queued_pages.add(page_of(slot))
            if keep and drop:
                releasable.append(take_ids(unit, drop))
                retained.append(take_ids(unit, keep))
            elif drop:
                releasable.append(unit)
            elif keep:
                retained.append(unit)
        record.units = unknown + releasable + retained
        record.states = (
            ["unknown"] * len(unknown)
            + ["unsubmitted"] * len(releasable)
            + ["held"] * len(retained)
        )

    def _reclaim_abandoned_leases(self) -> int:
        from .sr_resource_quarantine import quarantine
        from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
            SRTransferUnresolved,
        )

        scheduler = getattr(self, "scheduler", None)
        # A poisoned device still rejects new speculation, but a finished
        # consumer can be rolled back. Skipping this reclaim would leave the
        # request map pointing at pages the allocator is about to reuse.
        bag = quarantine(self)
        for record in list(bag.records):
            event = record.event
            # A missing event means the consumer may still be writing. Do not
            # restore and do not free; poison stays set by the capture path.
            if event is None:
                record.completion_done = None
                continue
            try:
                done = bool(event.query())
            except BaseException as exc:
                record.event = None
                record.completion_done = None
                if scheduler is not None:
                    scheduler._sr_device_poisoned = True
                raise SRTransferUnresolved(
                    "quarantined KV completion unknown"
                ) from exc
            record.completion_done = done
            if not done:
                continue
            if (
                getattr(self, "_raw_cache_inflight", False)
                and record.owner is getattr(self, "_raw_cache_owner", None)
            ):
                # This record's event is the shared cap's consumer. Clear
                # inflight directly; a later private pack has already set
                # ``_raw_cache_shared`` false, so note() must not decide.
                self._raw_cache_inflight = False
            owner = record.owner
            if not isinstance(owner, dict) or "mapping_snaps" not in owner:
                continue
            from .sr_resource_quarantine import record_may_release

            # No trusted speculative image, a failed release, or a stale
            # generation/row: leave the pages quarantined.
            if "spec_snaps" not in owner or not record_may_release(
                record, self.token_to_kv_pool_allocator
            ):
                continue
            if owner.get("mapping_restored"):
                self._partition_abandoned_pages(record)
                continue
            try:
                self._restore_abandoned_mapping(owner)
                self._partition_abandoned_pages(record)
            except BaseException as exc:
                record.event = None
                record.completion_done = None
                if scheduler is not None:
                    scheduler._sr_device_poisoned = True
                raise SRTransferUnresolved(
                    "abandoned tree mapping restore failed"
                ) from exc
            owner["mapping_restored"] = True
        freed = bag.reclaim(self.token_to_kv_pool_allocator)
        backend = getattr(self, "draft_attn_backend", None)
        view = (
            getattr(backend, "_sr_draft_paged_view", None)
            if backend is not None
            else None
        )
        if view is not None:
            hold = getattr(view.workspace, "lease_hold", None)
            if hold is not None and not any(r.owner is hold for r in bag.records):
                view.workspace.lease_hold = None
        return freed

    def _draft_forward(self, forward_batch: ForwardBatch):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_candidates import (
            candidate_workspace,
        )

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
        # One int32 conversion for every layer of every step. Remap and the
        # saved loc stay on the original int64 tensor.
        kv_slots = out_cache_loc
        if kv_slots.dtype != torch.int32:
            kv_slots = kv_slots.to(dtype=torch.int32)
        candidates = getattr(self, "_capture_candidate_workspace", None)
        if not getattr(self, "_capture_candidate_selected", False):
            candidates = candidate_workspace(self, forward_batch.batch_size, topk_p)
        kv_pool = getattr(self.draft_model_runner, "token_to_kv_pool", None)
        self._active_tree_kv_workspace = getattr(
            self, "_capture_tree_kv_workspace", None
        )
        self._active_identity_workspace = getattr(
            self, "_capture_identity_workspace", None
        )
        if self._active_identity_workspace is None:
            self._active_identity_workspace = self._identity_eager
        if kv_pool is not None and self.topk > 1 and self.speculative_num_steps > 2:
            if self._active_tree_kv_workspace is None:
                self._active_tree_kv_workspace = prepare_kv_move(
                    kv_pool,
                    rows * (self.speculative_num_steps - 2),
                    domain="tree_eager",
                    metrics=getattr(
                        getattr(self, "scheduler", None), "_sr_round_metrics", None
                    ),
                )
        self._ensure_identity_capacity(rows)
        self._slot_node_ids.fill_(-1)
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        scores = None
        for i in range(self.speculative_num_steps):
            if candidates is None:
                input_ids, hidden_states, scores, tree_info, parent_rows = (
                    select_top_k_tokens(i, topk_p, topk_index, hidden_states, scores, self.topk)
                )
                score_list.append(tree_info[0])
                token_list.append(tree_info[1])
                parents_list.append(tree_info[2])
                node_ids = tree_info[2]
            else:
                input_ids, scores, node_ids, parent_rows = candidates.select(i, topk_p, topk_index, scores)
            if i == self.speculative_num_steps - 1:
                break
            if i > 0 and self.topk > 1 and parent_rows is not None:
                self._remap_tree_kv_to_parents(out_cache_loc, parent_rows, i)
            if i == 0:
                self._slot_node_ids[i, :rows].copy_(self._step0_node_ids[:rows])
            else:
                self._slot_node_ids[i, :rows].copy_(
                    later_forward_node_ids(node_ids)[:rows]
                )
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = kv_slots[i]
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
            if candidates is None:
                probs = torch.softmax(logits_output.next_token_logits, dim=-1)
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            else:
                topk_p, topk_index = candidates.probabilities(logits_output.next_token_logits, i)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"SR draft_forward step {i}: topk_index OOB",
            )
            hidden_states = None
        self._last_out_cache_loc = out_cache_loc
        if candidates is not None:
            return candidates.finish(forward_batch.batch_size)
        return organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

    def prepare_tree_kv_graph(self, batch_cap):
        if self.topk <= 1 or self.speculative_num_steps <= 2:
            return None
        pool = self.draft_model_runner.token_to_kv_pool
        return prepare_kv_move(
            pool,
            int(batch_cap) * self.topk * (self.speculative_num_steps - 2),
            domain="tree_graph",
            graph=True,
        )

    def prepare_identity_remap_graph(self, batch_cap):
        if self.topk <= 1 or self.speculative_num_steps <= 2:
            return None
        rows = int(batch_cap) * int(self.topk)
        if rows < 1:
            return None
        self._ensure_identity_capacity(rows)
        workspace = getattr(self, "_identity_graph", None)
        if workspace is None:
            workspace = IdentityRemapWorkspace(
                self._slot_node_ids.device,
                int(self.speculative_num_steps),
                rows,
                graph=True,
                domain="tree_graph",
            )
            self._identity_graph = workspace
        else:
            workspace.reserve_rows(rows)
        workspace.warm()
        return workspace

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
        check = getattr(self, "_check_identity_table", None)
        if callable(check):
            check()
        workspace = getattr(self, "_active_identity_workspace", None)
        if workspace is None:
            workspace = getattr(self, "_identity_eager", None)
        if workspace is None:
            raise RuntimeError("identity remap workspace is not prepared")
        workspace.remap(self._slot_node_ids, parent_rows, n_prev_steps)
        kv_pool = getattr(self.draft_model_runner, "token_to_kv_pool", None)
        if kv_pool is None:
            return
        workspace = getattr(self, "_active_tree_kv_workspace", None)
        if workspace is None:
            workspace = prepare_kv_move(
                kv_pool, rows * n_prev_steps, domain="tree_eager"
            )
        active = None
        if getattr(self, "sr_tree_paged", False):
            inners = getattr(self.draft_attn_backend, "attn_backends", ())
            meta = (
                getattr(
                    getattr(inners[0], "forward_metadata", None), "sr_tree_paged", None
                )
                if inners
                else None
            )
            active = getattr(meta, "active_rows", None)
            if active is None:
                raise RuntimeError("paged tree KV remap requires live active_rows")
        remap_tree_kv_(workspace, out_cache_loc, parent_rows, n_prev_steps, active)
