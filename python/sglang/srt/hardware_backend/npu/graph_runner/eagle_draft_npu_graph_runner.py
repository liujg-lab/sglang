# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run the model with npu graph and torch.compile"""

from __future__ import annotations

import bisect
import logging
import os
import threading
from typing import TYPE_CHECKING, Callable, Dict, Union

import torch

from sglang.srt.configs.model_config import AttentionArch, is_deepseek_nsa
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.spec_utils import (
    NpuGraphPreparationError,
    NpuGraphReplaySubmittedError,
    build_draft_graph_step_kv_lens,
    expand_fia_cpu_update_inputs,
    fill_fia_cpu_update_payload,
    run_npu_graph_update_and_replay,
    validate_draft_graph_step_kv_lens,
    validate_tree_draft_fia_records,
)
from sglang.srt.speculative.standalone_remote.sr_align import is_device_context_error
from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
    begin_graph_host_sample,
    mark_graph_host_failed,
    measure_call,
    record_graph_host_sample_safely,
)
from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
    IMPL_PAGED_ATB,
    IMPL_PAGED_FIA,
    build_step_context_lens,
    context_lens_list,
    fill_paged_cpu_update_payload,
    kv_buckets_to_page_buckets,
    validate_tree_draft_paged_records,
)
from sglang.srt.speculative.tree_attn_fallback import (
    TREE_DRAFT_CAPTURE_BS_ENV,
    TreeReplayPlan,
    parse_tree_draft_capture_bs,
    tree_compact_fia_layout_supported,
    tree_fia_actual_seq_lengths_kv,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)

if is_npu():
    torch.cuda.CUDAGraph = torch.npu.NPUGraph
    torch.cuda.synchronize = torch.npu.synchronize
    torch.cuda.graph = torch.npu.graph
    torch.cuda.stream = torch.npu.stream
    torch.cuda.Stream = torch.npu.Stream
    torch.cuda.current_stream = torch.npu.current_stream


def _iter_graph_dispatch_records(graph):
    mode = getattr(graph, "graph_dispatch_mode", None)
    records = getattr(mode, "graph_dispatch_records", None) if mode is not None else None
    if records is None:
        records = getattr(graph, "graph_dispatch_records", None)
    return records


class EAGLEDraftNpuGraphRunner(EAGLEDraftCudaGraphRunner):
    def __init__(self, eagle_worker: EAGLEWorker):
        self.update_attr_name = None
        self.update_attr_type = None
        self._logged_tree_fia_update_bs = set()
        self._tree_fia_maps = {}
        self._tree_attention_impls = {}
        self.tree_graph_replay_count = 0
        self.tree_eager_fallback_count = 0
        self._last_can_run_reject = None
        self.tree_graph_disabled_reason = None
        self._npu_sr_tree_update_overlap = False
        self._npu_graph_device_id = None
        self._init_arch_map()
        page_size = int(getattr(eagle_worker, "page_size", 1) or 1)
        topk = int(getattr(eagle_worker, "topk", 1) or 1)
        self._slot_gather_graph = page_size > 1 and topk > 1
        model = getattr(eagle_worker, "model_runner", None) or getattr(
            eagle_worker, "draft_runner", None
        )
        use_mla = False
        if model is not None:
            cfg = getattr(model, "model_config", None)
            use_mla = getattr(cfg, "attention_arch", None) == AttentionArch.MLA
        self._tree_compact_fia = self._slot_gather_graph and tree_compact_fia_layout_supported(
            use_mla=use_mla, has_rope_split=False
        )
        backend = getattr(model, "draft_attn_backend", None)
        inners = getattr(backend, "attn_backends", [])
        self._tree_shared_prefix = bool(
            inners and getattr(inners[0], "_use_tree_shared_prefix", lambda: False)()
        )
        impl = getattr(inners[0], "tree_attention_impl", None) if inners else None
        self._tree_paged = impl in (IMPL_PAGED_ATB, IMPL_PAGED_FIA)
        self._paged_dummy_page = 0
        if self._tree_paged:
            self._tree_compact_fia = False
            self._slot_gather_graph = page_size > 1 and topk > 1
            multi = getattr(model, "draft_attn_backend", None)
            if multi is not None:
                multi._paged_dummy_page = self._paged_dummy_page
        if self._tree_shared_prefix:
            self._slot_gather_graph = topk > 1
            self._tree_compact_fia = False
        if self._slot_gather_graph:
            logger.info(
                "NPU tree draft graphs use token-level slot gather "
                "page_size=%s topk=%s compact_fia=%s paged=%s",
                page_size,
                topk,
                self._tree_compact_fia,
                getattr(self, "_tree_paged", False),
            )
        requested = bool(
            getattr(eagle_worker, "npu_sr_tree_update_overlap_requested", False)
        )
        super().__init__(eagle_worker)
        self._clear_tree_replay_plan()
        self._maybe_enable_sr_tree_update_overlap(requested)

    def _maybe_enable_sr_tree_update_overlap(self, requested: bool) -> None:
        """Enable overlap only for SR paged trees after graphs and device id exist."""
        graphs_ok = bool(getattr(self, "graphs", None)) and not getattr(
            self, "tree_graph_disabled_reason", None
        )
        reason = None
        if requested and getattr(self, "_tree_paged", False) and graphs_ok:
            try:
                device_id = int(torch.npu.current_device())
            except Exception as exc:
                if is_device_context_error(exc):
                    raise
                reason = f"current_device failed: {exc}"
            else:
                self._npu_graph_device_id = device_id
                self._npu_sr_tree_update_overlap = True
        elif requested and getattr(self, "_tree_paged", False):
            reason = getattr(self, "tree_graph_disabled_reason", None) or "no graphs"
        elif requested:
            reason = "not paged tree"
        worker = getattr(self, "eagle_worker", None)
        if worker is None or not hasattr(
            worker, "npu_sr_tree_update_overlap_requested"
        ):
            return
        extra = f" reason={reason}" if reason else ""
        logger.info(
            "NPU SR tree update/replay overlap requested=%s effective=%s "
            "implementation=%s device=%s%s",
            bool(requested),
            bool(self._npu_sr_tree_update_overlap),
            self._current_tree_attention_impl(),
            self._npu_graph_device_id,
            extra,
        )

    def filter_capture_batch_sizes(self, capture_bs, compile_bs):
        if not getattr(self, "_slot_gather_graph", False):
            return capture_bs, compile_bs
        raw = os.environ.get(TREE_DRAFT_CAPTURE_BS_ENV)
        if raw is None:
            logger.info(
                "NPU tree draft capture_bs follow --cuda-graph-bs %s",
                capture_bs,
            )
            return capture_bs, compile_bs
        allow = set(parse_tree_draft_capture_bs(raw))
        requested = list(capture_bs)
        capture_bs = [b for b in capture_bs if b in allow]
        compile_bs = [b for b in compile_bs if b in capture_bs]
        if not capture_bs:
            self.tree_graph_disabled_reason = (
                "tree draft capture_bs filter is empty"
            )
            logger.warning(
                "NPU tree draft graphs disabled: capture_bs filter is empty"
            )
        else:
            logger.info(
                "NPU tree draft capture_bs restricted by %s to %s",
                TREE_DRAFT_CAPTURE_BS_ENV,
                capture_bs,
            )
            omitted = [b for b in requested if b not in capture_bs]
            if omitted:
                logger.warning(
                    "NPU tree draft capture_bs env omits %s; "
                    "those batch sizes use eager",
                    omitted,
                )
        return capture_bs, compile_bs

    def _capture_extra_keys(self, ntpb=None):
        del ntpb
        if not getattr(self, "_slot_gather_graph", False):
            return [None]
        backend = None
        runner = getattr(self, "model_runner", None)
        if runner is not None:
            backend = getattr(runner, "draft_attn_backend", None) or getattr(
                runner, "attn_backend", None
            )
            inner = getattr(backend, "attn_backends", None)
            if inner:
                backend = inner[0]
        buckets = getattr(backend, "tree_kv_buckets", None) if backend is not None else None
        if getattr(self, "_tree_paged", False):
            page = max(int(getattr(self.eagle_worker, "page_size", 1) or 1), 1)
            if buckets:
                return kv_buckets_to_page_buckets(buckets, page)
            max_pages = getattr(backend, "_paged_graph_max_pages", None)
            if max_pages is not None:
                return [max(int(max_pages), 1)]
            return [1]
        if buckets:
            return list(reversed(list(buckets)))
        return [None]

    def _make_graph_key(
        self,
        bs: int,
        stream_idx=None,
        ntpb=None,
        extra=None,
    ):
        base_key = bs
        if stream_idx is not None:
            base_key = f"{stream_idx}_{base_key}"
        if extra is not None:
            return f"{base_key}_s{int(extra)}"
        return base_key

    def _clear_tree_replay_plan(self):
        self._tree_replay_plan = None
        self._tree_replay_batch_id = None
        self._tree_replay_stream_idx = None
        self._tree_replay_graph = None

    def _save_tree_replay_plan(self, plan: TreeReplayPlan, forward_batch: ForwardBatch):
        self._tree_replay_plan = plan
        self._tree_replay_batch_id = id(forward_batch)
        self._tree_replay_stream_idx = None
        self._tree_replay_graph = self.graphs[plan.graph_key]

    def _assert_tree_replay_graph(self, plan: TreeReplayPlan):
        implementations = getattr(self, "_tree_attention_impls", {})
        if (
            plan.graph_key in implementations
            and implementations[plan.graph_key] != self._current_tree_attention_impl()
        ):
            raise NpuGraphPreparationError(
                "tree attention implementation changed after capture", scope="graph"
            )
        if self.graphs.get(plan.graph_key) is not self._tree_replay_graph:
            raise NpuGraphPreparationError(
                "captured graph changed after admission",
                scope="graph",
            )

    def _current_tree_attention_impl(self):
        backend = getattr(self.model_runner, "draft_attn_backend", None)
        inners = getattr(backend, "attn_backends", [])
        return (
            getattr(inners[0], "tree_attention_impl", "compact_fia")
            if inners
            else "compact_fia"
        )

    def _padded_capture_bs(self, forward_batch: ForwardBatch):
        """Same conversion as ``EAGLEDraftCudaGraphRunner.replay``."""
        raw_bs = forward_batch.batch_size
        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = (
                max_num_tokens // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.uses_spec_topk_cuda_graph_layout()
                else max_num_tokens
            )
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        if index >= len(self.capture_bs):
            return raw_bs, None
        return raw_bs, self.capture_bs[index]

    def _snapshot_forward_batch_fields(self, forward_batch: ForwardBatch):
        return {
            "batch_size": forward_batch.batch_size,
            "seq_lens": forward_batch.seq_lens,
            "req_pool_indices": forward_batch.req_pool_indices,
            "positions": forward_batch.positions,
            "mrope_positions": forward_batch.mrope_positions,
            "seq_lens_cpu": forward_batch.seq_lens_cpu,
        }

    def _restore_forward_batch_fields(self, forward_batch: ForwardBatch, snap):
        for name, value in snap.items():
            setattr(forward_batch, name, value)

    def _snapshot_paged_eager_metadata(self, backend):
        steps = []
        for inner in getattr(backend, "attn_backends", None) or []:
            fm = getattr(inner, "forward_metadata", None)
            steps.append(
                {
                    "inner": inner,
                    "meta": getattr(inner, "_sr_tree_paged_meta", None),
                    "had_forward_metadata": fm is not None,
                }
            )
        return steps

    def _paged_eager_restore_valid(self, backend, snap, raw_bs) -> bool:
        inners = list(getattr(backend, "attn_backends", None) or [])
        if not snap or len(snap) != len(inners):
            return False
        if raw_bs is None:
            return False
        need = int(raw_bs) * int(getattr(backend, "topk", 1) or 1)
        if need <= 0:
            return False
        round_tables = getattr(backend, "_paged_round_tables", None)
        for item, inner in zip(snap, inners):
            if item.get("inner") is not inner:
                return False
            meta = item.get("meta")
            if meta is None:
                return False
            tables = getattr(meta, "block_tables", None)
            active = getattr(meta, "active_rows", None)
            if tables is None or active is None:
                return False
            if int(tables.shape[0]) != need or int(active.numel()) != need:
                return False
            if round_tables is None or tables is not round_tables:
                return False
        return True

    def _restore_paged_eager_metadata(self, snap) -> None:
        for item in snap:
            inner = item["inner"]
            saved_meta = item["meta"]
            inner._sr_tree_paged_meta = saved_meta
            if not item["had_forward_metadata"]:
                inner.forward_metadata = None
                continue
            fm = inner.forward_metadata
            if fm is None:
                continue
            fm.sr_tree_paged = saved_meta
            fm.block_tables = (
                None if saved_meta is None else saved_meta.block_tables
            )

    def replay(self, forward_batch: ForwardBatch):
        snap = self._snapshot_forward_batch_fields(forward_batch)
        backend = getattr(self.model_runner, "draft_attn_backend", None) or getattr(
            self.model_runner, "attn_backend", None
        )
        meta_snap = None
        raw_bs = None
        try:
            plan = self._tree_replay_plan
            if (
                getattr(self, "_tree_paged", False)
                and backend is not None
                and hasattr(backend, "bind_sr_tree_paged_replay")
            ):
                raw_bs = None if plan is None else plan.raw_bs
                meta_snap = self._snapshot_paged_eager_metadata(backend)
            if plan is not None:
                self._assert_tree_replay_graph(plan)
                if backend is not None:
                    backend._tree_replay_raw_bs = plan.raw_bs
                    backend._tree_replay_capture_bs = plan.capture_bs
                    backend._tree_replay_kv_bucket = plan.kv_bucket
                    raw_bs = plan.raw_bs
                    if getattr(self, "_tree_paged", False) and hasattr(
                        backend, "bind_sr_tree_paged_replay"
                    ):
                        pages = int(plan.kv_bucket) if plan.kv_bucket is not None else 1
                        backend.bind_sr_tree_paged_replay(plan.capture_bs, pages)
                    if getattr(self, "_tree_shared_prefix", False):
                        for inner in backend.attn_backends:
                            inner._replay_tree_s_cap = plan.kv_bucket
            return super().replay(forward_batch)
        except NpuGraphPreparationError as exc:
            self._restore_forward_batch_fields(forward_batch, snap)
            if not getattr(self, "_tree_paged", False):
                raise
            if meta_snap is not None and self._paged_eager_restore_valid(
                backend, meta_snap, raw_bs
            ):
                self._restore_paged_eager_metadata(meta_snap)
                raise
            raise RuntimeError(
                "paged tree graph prep failed without restorable eager metadata"
            ) from exc
        except NpuGraphReplaySubmittedError:
            self._restore_forward_batch_fields(forward_batch, snap)
            raise
        finally:
            self._clear_tree_replay_plan()

    def can_run(self, forward_batch: ForwardBatch):
        self._clear_tree_replay_plan()
        self._last_can_run_reject = None
        if self.require_mlp_tp_gather:
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.uses_spec_topk_cuda_graph_layout()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:
            cuda_graph_bs = forward_batch.batch_size
        is_bs_supported = (
            cuda_graph_bs in self.capture_bs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )
        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph
        if not is_bs_supported or not self.capture_bs:
            if not self.capture_bs:
                self._last_can_run_reject = "empty_capture_bs"
            else:
                self._last_can_run_reject = (
                    f"bs_over_max_capture_bs bs={int(cuda_graph_bs)} "
                    f"max_bs={int(getattr(self, 'max_bs', 0) or 0)}"
                )
            if getattr(self, "_slot_gather_graph", False):
                self.tree_eager_fallback_count += 1
            return False
        kv_bucket = None
        if getattr(self, "_slot_gather_graph", False):
            runner = getattr(self, "model_runner", None)
            backend = None
            if runner is not None:
                backend = getattr(runner, "draft_attn_backend", None) or getattr(
                    runner, "attn_backend", None
                )
                inner = (
                    getattr(backend, "attn_backends", None)
                    if backend is not None
                    else None
                )
                if inner:
                    backend = inner[0]
            fn = (
                getattr(backend, "tree_slot_graph_can_run", None)
                if backend is not None
                else None
            )
            if fn is not None:
                ok = bool(fn(forward_batch))
                if not ok:
                    self._last_can_run_reject = "slot_graph_reject"
                    self.tree_eager_fallback_count += 1
                    return False
                kv_bucket = getattr(backend, "_replay_tree_s_cap", None)
        tokens_per_req = self.num_tokens_per_bs  # currently equals topk
        raw_bs, capture_bs = self._padded_capture_bs(forward_batch)
        if capture_bs is None:
            self._last_can_run_reject = (
                f"bs_over_max_capture_bs bs={int(raw_bs)} "
                f"max_bs={int(getattr(self, 'max_bs', 0) or 0)}"
            )
            if getattr(self, "_slot_gather_graph", False):
                self.tree_eager_fallback_count += 1
            return False
        graph_key = self._make_graph_key(capture_bs, extra=kv_bucket)
        if graph_key not in self.graphs:
            self._last_can_run_reject = "graph_key_missing"
            if getattr(self, "_slot_gather_graph", False):
                self.tree_eager_fallback_count += 1
            return False
        if getattr(self, "_tree_paged", False) and graph_key not in getattr(
            self, "_tree_fia_maps", {}
        ):
            self._last_can_run_reject = "fia_map_missing"
            self.tree_eager_fallback_count += 1
            return False
        if (
            self._tree_attention_impls.get(
                graph_key, self._current_tree_attention_impl()
            )
            != self._current_tree_attention_impl()
        ):
            self._last_can_run_reject = "impl_changed"
            return False
        self._last_can_run_reject = None
        self._save_tree_replay_plan(
            TreeReplayPlan(
                graph_key=graph_key,
                raw_bs=int(raw_bs),
                capture_bs=int(capture_bs),
                tokens_per_req=int(tokens_per_req),
                kv_bucket=kv_bucket if kv_bucket is None else int(kv_bucket),
            ),
            forward_batch,
        )
        return True

    def _init_arch_map(self):
        self.attr_name: Dict[str, str] = {
            AttentionArch.MLA: "actual_seq_lengths_kv",
            AttentionArch.MHA: "context_lens",
        }
        self.attr_type: Dict[str, Union[list, torch.Tensor]] = {
            AttentionArch.MLA: [],
            AttentionArch.MHA: torch.Tensor(),
        }

    def _create_graph(self):
        return torch.npu.NPUGraph()

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.npu.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.npu.graph(
            graph, pool=pool, stream=stream, auto_dispatch_capture=True
        ):
            out = run_once_fn()
        return out

    def _get_update_attr_name(self):
        return self.attr_name[AttentionArch.MLA]

    def _get_update_attr_type(self):
        return self.attr_type[AttentionArch.MLA]

    def _num_model_layers(self):
        model = self.model_runner.model
        start = getattr(model, "start_layer", None)
        end = getattr(model, "end_layer", None)
        if start is None or end is None:
            inner = getattr(model, "model", None)
            if start is None:
                start = getattr(inner, "start_layer", 0)
            if end is None:
                end = getattr(inner, "end_layer", None)
        if end is None:
            raise NpuGraphPreparationError(
                "model end_layer is unavailable for NPU tree graph update",
                scope="format",
            )
        return int(end) - int(start or 0)

    def capture_one_batch_size(
        self, num_seqs: int, forward: Callable, stream_idx: int = 0
    ):
        backend = self.model_runner.draft_attn_backend
        inners = getattr(backend, "attn_backends", [])
        if self._tree_shared_prefix:
            for inner in inners:
                inner._shared_capture_width = getattr(
                    self, "_active_capture_extra", None
                )
        if getattr(self, "_tree_paged", False) and hasattr(
            backend, "bind_sr_tree_paged_capture"
        ):
            extra = getattr(self, "_active_capture_extra", None)
            max_pages = int(extra) if extra is not None else 1
            backend._paged_capture_max_pages = max_pages
            backend._paged_dummy_page = getattr(self, "_paged_dummy_page", 0)
            backend.bind_sr_tree_paged_capture(
                int(num_seqs), max_pages, self._paged_dummy_page
            )
        try:
            graph, out = super().capture_one_batch_size(num_seqs, forward, stream_idx)
        finally:
            if self._tree_shared_prefix:
                for inner in inners:
                    inner._shared_capture_width = None
        extra = getattr(self, "_active_capture_extra", None)
        self._tree_attention_impls[self._make_graph_key(int(num_seqs), extra=extra)] = (
            self._current_tree_attention_impl()
        )
        skip_fia = self.tree_graph_disabled_reason or (
            self._slot_gather_graph
            and not self._tree_compact_fia
            and not getattr(self, "_tree_paged", False)
        )
        if skip_fia:
            return graph, out
        impl = self._current_tree_attention_impl()
        if getattr(self, "_tree_paged", False):
            self.update_attr_name = (
                "actual_seq_lengths_kv" if impl == IMPL_PAGED_FIA else "context_lens"
            )
        else:
            self.update_attr_name = self._get_update_attr_name()
        n_steps = max(int(self.speculative_num_steps) - 1, 0)
        extra = getattr(self, "_active_capture_extra", None)
        map_key = self._make_graph_key(int(num_seqs), extra=extra)
        try:
            num_layers = self._num_model_layers()
            records = _iter_graph_dispatch_records(graph)
            if getattr(self, "_tree_paged", False):
                n_records, step_ids = validate_tree_draft_paged_records(
                    records, n_steps, num_layers, self.update_attr_name, impl
                )
            else:
                n_records, step_ids = validate_tree_draft_fia_records(
                    records, n_steps, num_layers, self.update_attr_name
                )
            n_lens = int(num_seqs) * max(int(self.topk), 1)
            placeholder = [1] * n_lens
            step_lens_list = [list(placeholder) for _ in range(n_steps)]
            if getattr(self, "_tree_paged", False) and impl == IMPL_PAGED_ATB:
                payload = [
                    {
                        "context_lens": torch.ones((n_lens,), dtype=torch.int32)
                    }
                    for step_lens in step_lens_list
                    for _ in range(num_layers)
                ]
            else:
                payload = expand_fia_cpu_update_inputs(
                    step_lens_list, num_layers, self.update_attr_name
                )
            self._tree_fia_maps[map_key] = {
                "n_records": n_records,
                "n_steps": n_steps,
                "num_layers": num_layers,
                "step_ids": step_ids,
                "bs": int(num_seqs),
                "extra": extra,
                "payload": payload,
                "impl": impl,
                "attr_name": self.update_attr_name,
            }
        except NpuGraphPreparationError as e:
            if getattr(e, "scope", "graph") == "format":
                self.tree_graph_disabled_reason = str(e)
                logger.warning(
                    "NPU tree graph record format unsupported: %s", e
                )
            else:
                logger.warning("NPU tree graph bs=%s extra=%s rejected: %s", num_seqs, extra, e)
        return graph, out

    def capture(self):
        if not self.capture_bs:
            self.graphs.clear()
            self.max_bs = 0
            if not self.tree_graph_disabled_reason:
                self.tree_graph_disabled_reason = "tree draft capture_bs is empty"
            logger.warning(
                "NPU tree draft graphs disabled: reason=%s "
                "tree_graph_replay_count=%s tree_eager_fallback_count=%s",
                self.tree_graph_disabled_reason,
                self.tree_graph_replay_count,
                self.tree_eager_fallback_count,
            )
            return
        if self.tree_graph_disabled_reason:
            self.graphs.clear()
            self.capture_bs = []
            self.max_bs = 0
            logger.warning(
                "NPU tree draft graphs disabled: reason=%s "
                "tree_graph_replay_count=%s tree_eager_fallback_count=%s",
                self.tree_graph_disabled_reason,
                self.tree_graph_replay_count,
                self.tree_eager_fallback_count,
            )
            return
        super().capture()
        if self._slot_gather_graph and not self._tree_compact_fia:
            logger.info(
                "NPU tree draft %s graphs ready: graphs=%s "
                "tree_graph_replay_count=%s tree_eager_fallback_count=%s",
                self._current_tree_attention_impl(),
                sorted(map(str, self.graphs)),
                self.tree_graph_replay_count,
                self.tree_eager_fallback_count,
            )
            return
        if self._tree_compact_fia:
            self._finalize_tree_fia_maps()

    def _finalize_tree_fia_maps(self):
        if self.tree_graph_disabled_reason or not self._tree_fia_maps:
            if not self.tree_graph_disabled_reason:
                self.tree_graph_disabled_reason = (
                    "no captured tree graph passed FIA record validation"
                )
            self.graphs.clear()
            self.capture_bs = []
            self.max_bs = 0
            logger.warning(
                "NPU tree draft graphs disabled: reason=%s "
                "tree_graph_replay_count=%s tree_eager_fallback_count=%s",
                self.tree_graph_disabled_reason,
                self.tree_graph_replay_count,
                self.tree_eager_fallback_count,
            )
            return
        for key in list(self.graphs):
            if key not in self._tree_fia_maps:
                del self.graphs[key]
                self.output_buffers.pop(key, None)
        kept_bs = sorted(
            {
                int(self._tree_fia_maps[k]["bs"])
                for k in self.graphs
                if k in self._tree_fia_maps
            }
        )
        self.capture_bs = [b for b in self.capture_bs if b in kept_bs]
        self.max_bs = max(self.capture_bs) if self.capture_bs else 0
        logger.info(
            "NPU tree draft FIA maps ready: graphs=%s "
            "tree_graph_replay_count=%s tree_eager_fallback_count=%s "
            "tree_graph_disabled_reason=%s",
            sorted(map(str, self._tree_fia_maps)),
            self.tree_graph_replay_count,
            self.tree_eager_fallback_count,
            self.tree_graph_disabled_reason,
        )

    def _replay_update(self, graph, cpu_update_input, errors):
        try:
            graph.update(cpu_update_input=cpu_update_input)
        except Exception as e:
            errors.append(e)

    def _replay(self, forward_batch: ForwardBatch):
        plan = self._tree_replay_plan
        if plan is None or self._tree_replay_batch_id != id(forward_batch):
            raise NpuGraphPreparationError(
                "tree draft graph replay has no TreeReplayPlan for this batch",
                scope="graph",
            )
        self._assert_tree_replay_graph(plan)
        graph = self._tree_replay_graph
        fia_maps = getattr(self, "_tree_fia_maps", None) or {}
        fia_map_peek = fia_maps.get(plan.graph_key)
        if getattr(self, "_tree_paged", False) and fia_map_peek is not None:
            self.update_attr_name = fia_map_peek.get("attr_name") or (
                "actual_seq_lengths_kv"
                if fia_map_peek.get("impl") == IMPL_PAGED_FIA
                else "context_lens"
            )
        else:
            self.update_attr_name = self._get_update_attr_name()
        self.update_attr_type = self._get_update_attr_type()
        backend = getattr(self.model_runner, "draft_attn_backend", None) or getattr(
            self.model_runner, "attn_backend", None
        )
        inner = (
            getattr(backend, "attn_backends", None) if backend is not None else None
        )
        if inner:
            backend = inner[0]
        kv_bucket = None
        if getattr(self, "_slot_gather_graph", False) and backend is not None:
            kv_bucket = getattr(backend, "_replay_tree_s_cap", None)
        if (
            int(self.bs) != plan.capture_bs
            or int(self.raw_bs) != plan.raw_bs
            or int(self.num_tokens_per_bs) != plan.tokens_per_req
            or kv_bucket != plan.kv_bucket
        ):
            raise NpuGraphPreparationError(
                f"tree draft replay shape != plan: bs={self.bs}/{plan.capture_bs} "
                f"raw_bs={self.raw_bs}/{plan.raw_bs} ntpb={self.num_tokens_per_bs}/"
                f"{plan.tokens_per_req} bucket={kv_bucket}/{plan.kv_bucket}",
                scope="graph",
            )
        graph_key = plan.graph_key
        self.output_buffers[self.bs] = self.output_buffers[graph_key]
        skip_fia = is_deepseek_nsa(
            self.model_runner.model_config.hf_config
        ) or (
            self._slot_gather_graph
            and not self._tree_compact_fia
            and not getattr(self, "_tree_paged", False)
        )
        env_mod = globals().get("os")
        sr_paged_overlap = bool(
            getattr(self, "_tree_paged", False)
            and getattr(self, "_npu_sr_tree_update_overlap", False)
        )
        overlap = False
        if getattr(self, "_tree_paged", False):
            overlap = sr_paged_overlap
        elif (
            not skip_fia
            and env_mod is not None
        ):
            overlap = not env_mod.environ.get("SGLANG_NPU_TREE_FIA_SERIAL_UPDATE")

        worker = getattr(self, "eagle_worker", None)
        scheduler = getattr(worker, "scheduler", None)
        metrics = getattr(scheduler, "_sr_round_metrics", None)
        if (
            metrics is None
            or getattr(scheduler, "sr_tree_drafter", None) is not worker
            or getattr(metrics, "role", None) != "Draft"
            or not getattr(metrics, "active", False)
        ):
            metrics = None
        sample = None
        if metrics is not None:
            try:
                peek = fia_maps.get(graph_key) if fia_maps else None
                num_steps = None
                if isinstance(peek, dict) and peek.get("n_steps") is not None:
                    try:
                        num_steps = int(peek["n_steps"])
                    except (TypeError, ValueError):
                        num_steps = peek.get("n_steps")
                sample = begin_graph_host_sample(
                    {
                        "graph_phase": "draft_tree",
                        "round_id": int(metrics.rounds) + 1,
                        "graph_key": graph_key,
                        "implementation": self._current_tree_attention_impl(),
                        "raw_bs": int(self.raw_bs),
                        "capture_bs": int(self.bs),
                        "kv_bucket": plan.kv_bucket,
                        "topk": int(getattr(self, "topk", 0) or 0),
                        "num_steps": num_steps,
                        "overlap": overlap,
                    }
                )
            except Exception:
                sample = None

        def _measure(stage, fn):
            if sample is None:
                return fn()
            return measure_call(sample, stage, fn)

        try:
            if skip_fia:
                def _direct_submit():
                    replay_error = None
                    try:
                        _measure("replay_call", graph.replay)
                    except Exception as exc:
                        replay_error = exc
                    if replay_error is not None:
                        raise NpuGraphReplaySubmittedError(
                            "NPU graph update/replay failed"
                        ) from replay_error

                _measure("submit_envelope", _direct_submit)
                self.tree_graph_replay_count += 1
                if (
                    self.tree_graph_replay_count == 1
                    or self.tree_graph_replay_count % 32 == 0
                ):
                    logger.info(
                        "NPU tree draft graph replay count=%s key=%s "
                        "raw_bs=%s capture_bs=%s implementation=%s eager_fallback=%s",
                        self.tree_graph_replay_count,
                        graph_key,
                        getattr(self, "raw_bs", None),
                        getattr(self, "bs", None),
                        self._current_tree_attention_impl(),
                        self.tree_eager_fallback_count,
                    )
                return

            if forward_batch.seq_lens_cpu is None:
                raise NpuGraphPreparationError(
                    "tree draft graph replay requires seq_lens_cpu",
                    scope="graph",
                )

            fia_map = fia_maps.get(graph_key)
            if fia_map is None:
                raise NpuGraphPreparationError(
                    f"tree draft graph key={graph_key!r} has no FIA map",
                    scope="graph",
                )

            def _build_lengths():
                prefix_lens = forward_batch.seq_lens_cpu[: self.raw_bs]
                step_lens_list = []
                n_steps = int(fia_map["n_steps"])
                step_backends = inner if inner else None
                capture_rows = int(self.bs) * max(int(self.topk), 1)
                try:
                    for speculative_step_id in range(n_steps):
                        seq_lens = None
                        if getattr(self, "_tree_paged", False):
                            meta = None
                            if step_backends is not None and speculative_step_id < len(
                                step_backends
                            ):
                                meta = getattr(
                                    step_backends[speculative_step_id],
                                    "_sr_tree_paged_meta",
                                    None,
                                )
                            if meta is not None:
                                seq_lens = list(meta.context_lens_list)
                            else:
                                seq_lens = context_lens_list(
                                    build_step_context_lens(
                                        prefix_lens,
                                        self.topk,
                                        speculative_step_id,
                                        capture_rows,
                                    )
                                )
                        elif self._tree_compact_fia and step_backends is not None:
                            if speculative_step_id < len(step_backends):
                                seq_lens = getattr(
                                    step_backends[speculative_step_id],
                                    "tree_fia_kv_lens_cpu",
                                    None,
                                )
                        if seq_lens is None:
                            seq_lens = build_draft_graph_step_kv_lens(
                                prefix_lens, self.bs, self.topk, speculative_step_id
                            )
                            validate_draft_graph_step_kv_lens(
                                seq_lens,
                                self.bs,
                                self.topk,
                                self.raw_bs,
                                prefix_lens,
                                speculative_step_id,
                            )
                            if self._tree_compact_fia:
                                seq_lens = tree_fia_actual_seq_lengths_kv(seq_lens)
                        elif (
                            not getattr(self, "_tree_paged", False)
                            and self._tree_compact_fia
                        ):
                            seq_lens = tree_fia_actual_seq_lengths_kv(
                                seq_lens, capture_rows
                            )
                        step_lens_list.append(seq_lens)
                except NpuGraphPreparationError:
                    raise
                except (TypeError, ValueError) as e:
                    raise NpuGraphPreparationError(
                        f"tree draft step KV lengths invalid: {e}",
                        scope="graph",
                    ) from e
                return step_lens_list

            step_lens_list = _measure("lengths", _build_lengths)

            n_records = int(fia_map["n_records"])
            num_layers = int(fia_map["num_layers"])
            payload = fia_map.get("payload")
            if payload is None or len(payload) != n_records:
                raise NpuGraphPreparationError(
                    f"tree draft graph key={graph_key!r} has no reusable FIA payload",
                    scope="graph",
                )
            attr_name = fia_map.get("attr_name") or self.update_attr_name

            def _fill_payload():
                if getattr(self, "_tree_paged", False):
                    fill_paged_cpu_update_payload(
                        payload, step_lens_list, fia_map["step_ids"], attr_name
                    )
                else:
                    fill_fia_cpu_update_payload(
                        payload, step_lens_list, fia_map["step_ids"], attr_name
                    )

            _measure("payload_fill", _fill_payload)
            log_key = graph_key
            if log_key not in self._logged_tree_fia_update_bs:
                logger.info(
                    "NPU tree draft graph FIA updates: key=%s bs=%s raw_bs=%s "
                    "records=%s steps=%s layers=%s updates=%s bucket=%s",
                    graph_key,
                    self.bs,
                    self.raw_bs,
                    n_records,
                    int(fia_map["n_steps"]),
                    num_layers,
                    len(payload),
                    plan.kv_bucket,
                )
                self._logged_tree_fia_update_bs.add(log_key)

            def _call_update():
                if sr_paged_overlap:
                    torch.npu.set_device(self._npu_graph_device_id)
                if sample is None:
                    graph.update(cpu_update_input=payload)
                else:
                    measure_call(
                        sample,
                        "update_call",
                        lambda: graph.update(cpu_update_input=payload),
                    )

            def _call_replay():
                if sample is None:
                    graph.replay()
                else:
                    measure_call(sample, "replay_call", graph.replay)

            if sample is None:
                run_npu_graph_update_and_replay(
                    _call_update,
                    _call_replay,
                    overlap=overlap,
                )
            else:
                _measure(
                    "submit_envelope",
                    lambda: run_npu_graph_update_and_replay(
                        _call_update,
                        _call_replay,
                        overlap=overlap,
                    ),
                )
            self.tree_graph_replay_count += 1
            if self.tree_graph_replay_count == 1 or self.tree_graph_replay_count % 32 == 0:
                logger.info(
                    "NPU tree draft graph replay count=%s key=%s bucket=%s "
                    "raw_bs=%s capture_bs=%s implementation=%s eager_fallback=%s",
                    self.tree_graph_replay_count,
                    graph_key,
                    plan.kv_bucket,
                    plan.raw_bs,
                    plan.capture_bs,
                    self._current_tree_attention_impl(),
                    self.tree_eager_fallback_count,
                )
        except BaseException:
            mark_graph_host_failed(sample)
            raise
        finally:
            if sample is not None:
                record_graph_host_sample_safely(metrics, sample)

    def _cache_loc_dtype(self):
        return torch.int32
