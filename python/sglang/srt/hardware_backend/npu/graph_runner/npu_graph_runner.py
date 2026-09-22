# Copyright 2023-2024 SGLang Team
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
"""Run the model with npu graph and torch.compile."""

from __future__ import annotations

import bisect
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
import torch

import sglang
from sglang.srt.configs.model_config import AttentionArch, is_deepseek_nsa
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.model_executor.cuda_graph_runner import (
    CudaGraphRunner,
    _is_spectre,
    _uses_dual_ntpb,
)
from sglang.srt.multiplex.pdmux_context import get_current_stream_idx
from sglang.srt.speculative.spec_utils import (
    NpuGraphPreparationError,
    NpuGraphReplaySubmittedError,
    expand_fia_cpu_update_inputs,
    fill_fia_cpu_update_payload,
    run_npu_graph_update_and_replay,
)
from sglang.srt.speculative.standalone_remote.sr_align import is_device_context_error
from sglang.srt.speculative.standalone_remote.sr_round_metrics import (
    begin_graph_host_sample,
    mark_graph_host_failed,
    measure_call,
    record_graph_host_sample_safely,
)
from sglang.srt.speculative.standalone_remote.verifier.sr_target_tree_fia import (
    IMPL_TREE_PAGED_FIA,
    TARGET_TREE_FIA_KV_ATTR,
    read_sr_target_update_overlap_env,
    validate_target_tree_fia_records,
)
from sglang.srt.speculative.tree_attn_fallback import (
    TreeReplayPlan,
    tree_fia_actual_seq_lengths_kv,
)
from sglang.srt.utils import (
    empty_context,
    get_bool_env_var,
    get_compiler_backend,
    is_npu,
)

is_npu = is_npu()

if is_npu:
    import torch_npu
    from torch_npu.profiler import ProfilerActivity, profile

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors


@contextmanager
def patch_model_npu(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    if enable_compile:
        backend = get_compiler_backend("npugraph_ex")
        yield torch.compile(
            torch.no_grad()(model.forward),
            fullgraph=True,
            dynamic=False,
            backend=backend,
        )
    else:
        yield model.forward


class NPUGraphRunner(CudaGraphRunner):
    """A NPUGraphRunner runs the forward pass of a model with npu graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        sglang.srt.model_executor.cuda_graph_runner.patch_model = patch_model_npu
        # Parent CudaGraphRunner.__init__ calls capture() before returning.
        # Payload dict and FIA attr names must exist for capture_one_batch_size.
        self.update_attr_name = None
        self.update_attr_type = None
        self._fia_payloads = {}
        self._tree_attention_impls = {}
        self._target_fia_maps = {}
        self._last_can_run_reject = None
        self.tree_verify_replay_count = 0
        self.tree_verify_eager_fallback_count = 0
        args = model_runner.server_args
        self._plain_ar_update_overlap = (
            model_runner.spec_algorithm.is_none()
            and not model_runner.is_draft_worker
            and args.standalone_remote_role is None
            and args.spectre_role is None
        )
        self._npu_sr_target_update_overlap = False
        self._npu_graph_device_id = None
        self._logged_sr_target_overlap_submit = False
        target_overlap_requested = read_sr_target_update_overlap_env()
        self._npu_sr_target_update_overlap_requested = target_overlap_requested
        super().__init__(model_runner)
        self.model_runner = model_runner
        if not hasattr(self, "attr_name"):
            self._init_arch_map()
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self._clear_tree_replay_plan()
        self._maybe_enable_sr_target_update_overlap(target_overlap_requested)

    def _ensure_capture_attrs(self):
        if not hasattr(self, "attr_name"):
            self._init_arch_map()
        if getattr(self, "_fia_payloads", None) is None:
            self._fia_payloads = {}
        if getattr(self, "_target_fia_maps", None) is None:
            self._target_fia_maps = {}

    def capture(self):
        self._ensure_capture_attrs()
        super().capture()

    def _capture_extra_keys(self, ntpb=None):
        """S_cap buckets only for tree TARGET_VERIFY. DECODE keeps integer / r1 keys."""
        fm = getattr(self, "capture_forward_mode", None)
        is_target_verify = bool(fm is not None and fm.is_target_verify())
        if not is_target_verify:
            return [None]
        if _uses_dual_ntpb(self) and ntpb is not None and int(ntpb) == 1:
            return [None]
        backend = getattr(self.model_runner, "attn_backend", None)
        verify_topk = (
            int(getattr(backend, "verify_tree_topk", 1) or 1)
            if backend is not None
            else 1
        )
        if verify_topk <= 1:
            return [None]
        buckets = getattr(backend, "tree_kv_buckets", None) if backend is not None else None
        if buckets:
            return list(reversed(list(buckets)))
        return [None]

    def _clear_tree_replay_plan(self):
        self._tree_replay_plan = None
        self._tree_replay_batch_id = None
        self._tree_replay_stream_idx = None
        self._tree_replay_graph = None

    def _save_tree_replay_plan(
        self,
        plan: TreeReplayPlan,
        forward_batch: ForwardBatch,
        stream_idx,
    ):
        self._tree_replay_plan = plan
        self._tree_replay_batch_id = id(forward_batch)
        self._tree_replay_stream_idx = stream_idx
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
        return getattr(
            self.model_runner.attn_backend, "tree_attention_impl", "compact_fia"
        )

    def _padded_capture_bs(self, forward_batch: ForwardBatch, actual_ntpb: int):
        """Same conversion as ``CudaGraphRunner.replay_prepare``."""
        raw_bs = forward_batch.batch_size
        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = (
                max_num_tokens / actual_ntpb
                if (
                    self.model_runner.spec_algorithm.is_eagle()
                    or self.model_runner.spec_algorithm.is_standalone()
                    or self.model_runner.spec_algorithm.is_standalone_remote()
                    or _is_spectre(self)
                )
                else max_num_tokens
            )
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        if index >= len(self.capture_bs):
            return raw_bs, None
        return raw_bs, self.capture_bs[index]

    def _is_tree_verify_batch(self, forward_batch: ForwardBatch) -> bool:
        fm = getattr(forward_batch, "forward_mode", None)
        return bool(
            fm is not None
            and fm.is_target_verify()
            and int(
                getattr(
                    self.model_runner.server_args, "speculative_eagle_topk", 1
                )
                or 1
            )
            > 1
        )

    def can_run(self, forward_batch: ForwardBatch):
        self._clear_tree_replay_plan()
        self._last_can_run_reject = None
        is_tree_verify = self._is_tree_verify_batch(forward_batch)
        if not super().can_run(forward_batch):
            if is_tree_verify:
                self.tree_verify_eager_fallback_count += 1
                self._last_can_run_reject = (
                    f"bs_over_max_capture_bs bs={int(forward_batch.batch_size)} "
                    f"max_bs={int(getattr(self, 'max_bs', 0) or 0)}"
                )
            return False
        if not self.capture_bs:
            if is_tree_verify:
                self.tree_verify_eager_fallback_count += 1
                self._last_can_run_reject = "empty_capture_bs"
            return False
        backend = getattr(self.model_runner, "attn_backend", None)
        kv_bucket = None
        fn = getattr(backend, "tree_slot_graph_can_run", None)
        if fn is not None:
            ok = bool(fn(forward_batch))
            if not ok:
                self._last_can_run_reject = "slot_graph_reject"
                self.tree_verify_eager_fallback_count += 1
                return False
            if is_tree_verify:
                kv_bucket = getattr(backend, "_replay_tree_s_cap", None)
        tokens_per_req = self._get_actual_ntpb(forward_batch)
        raw_bs, capture_bs = self._padded_capture_bs(forward_batch, tokens_per_req)
        if capture_bs is None:
            self._last_can_run_reject = (
                f"bs_over_max_capture_bs bs={int(raw_bs)} "
                f"max_bs={int(getattr(self, 'max_bs', 0) or 0)}"
            )
            if is_tree_verify:
                self.tree_verify_eager_fallback_count += 1
            return False
        stream_idx = (
            get_current_stream_idx() if getattr(self, "enable_pdmux", False) else None
        )
        graph_key = self._make_graph_key(
            capture_bs,
            stream_idx,
            tokens_per_req if _uses_dual_ntpb(self) else None,
            extra=kv_bucket,
        )
        if graph_key not in self.graphs:
            self._last_can_run_reject = "graph_key_missing"
            if is_tree_verify:
                self.tree_verify_eager_fallback_count += 1
            return False
        if (
            graph_key in self._tree_attention_impls
            and self._tree_attention_impls[graph_key]
            != self._current_tree_attention_impl()
        ):
            self._last_can_run_reject = "impl_changed"
            return False
        if (
            self._current_tree_attention_impl() == IMPL_TREE_PAGED_FIA
            and getattr(self, "_target_fia_maps", {}).get(graph_key) is None
        ):
            self._last_can_run_reject = "fia_map_missing"
            if is_tree_verify:
                self.tree_verify_eager_fallback_count += 1
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
            stream_idx,
        )
        return True

    def _init_arch_map(self):
        if self.is_dllm:
            self.attr_name: Dict[str, str] = {
                AttentionArch.MLA: "actual_seq_lengths_kv",
                AttentionArch.MHA: "actual_seq_lengths_kv",
            }
        else:
            self.attr_name: Dict[str, str] = {
                AttentionArch.MLA: "actual_seq_lengths_kv",
                AttentionArch.MHA: "context_lens",
            }
        self.attr_type: Dict[str, Union[list, torch.Tensor]] = {
            AttentionArch.MLA: [],
            AttentionArch.MHA: torch.Tensor(),
        }

    def _create_device_graph(self):
        return torch.npu.NPUGraph()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        if self.enable_torch_compile:
            skip_guard_context = torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        else:
            skip_guard_context = empty_context()

        with skip_guard_context, torch.npu.graph(
            graph,
            pool=pool,
            stream=stream,
            auto_dispatch_capture=True,
        ):
            out = run_once_fn()
        return out

    def _get_update_attr_name(self):
        return self.attr_name[AttentionArch.MLA]

    def _get_update_attr_type(self):
        return self.attr_type[AttentionArch.MLA]

    def capture_one_batch_size(
        self,
        bs: int,
        forward,
        stream_idx=None,
        ntpb_override=None,
    ):
        backend = self.model_runner.attn_backend
        shared = getattr(backend, "_use_tree_shared_prefix", lambda: False)()
        target_fia = getattr(backend, "_use_target_tree_paged_fia", lambda: False)()
        if shared or target_fia:
            backend._shared_capture_width = getattr(self, "_active_capture_extra", None)
        try:
            graph, out = super().capture_one_batch_size(
                bs, forward, stream_idx, ntpb_override
            )
        finally:
            if shared or target_fia:
                backend._shared_capture_width = None
        self._ensure_capture_attrs()
        self.update_attr_name = self._get_update_attr_name()
        ntpb = ntpb_override if ntpb_override is not None else self.num_tokens_per_bs
        extra = getattr(self, "_active_capture_extra", None)
        key = self._make_graph_key(
            bs,
            stream_idx,
            ntpb if _uses_dual_ntpb(self) else None,
            extra=extra,
        )
        if extra is not None:
            self._tree_attention_impls[key] = self._current_tree_attention_impl()
            if shared:
                return graph, out
            if target_fia:
                self._bind_target_tree_fia_payload(graph, key, bs)
                return graph, out
        n_lens = int(bs) * int(ntpb) if extra is not None else int(bs)
        self._fia_payloads[key] = [{self.update_attr_name: [1] * n_lens}]
        return graph, out

    def _iter_graph_dispatch_records(self, graph):
        mode = getattr(graph, "graph_dispatch_mode", None)
        records = (
            getattr(mode, "graph_dispatch_records", None) if mode is not None else None
        )
        if records is None:
            records = getattr(graph, "graph_dispatch_records", None)
        return records

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

    def _bind_target_tree_fia_payload(self, graph, key, bs):
        self.update_attr_name = TARGET_TREE_FIA_KV_ATTR
        try:
            num_layers = self._num_model_layers()
            records = self._iter_graph_dispatch_records(graph)
            n_records, step_ids = validate_target_tree_fia_records(
                records, num_layers, self.update_attr_name
            )
            placeholder = [1] * int(bs)
            payload = expand_fia_cpu_update_inputs(
                [placeholder], num_layers, self.update_attr_name
            )
            self._fia_payloads[key] = payload
            self._target_fia_maps[key] = {
                "n_records": n_records,
                "num_layers": num_layers,
                "step_ids": step_ids,
                "bs": int(bs),
                "payload": payload,
                "attr_name": self.update_attr_name,
            }
        except NpuGraphPreparationError:
            self._target_fia_maps[key] = None

    def _maybe_enable_sr_target_update_overlap(self, requested: bool) -> None:
        """Enable overlap only for an SR Target tree_paged_fia graph."""
        runner = self.model_runner
        args = getattr(runner, "server_args", None)
        algo = getattr(runner, "spec_algorithm", None)
        is_sr_target = (
            getattr(args, "standalone_remote_role", None) == "target"
            and callable(getattr(algo, "is_standalone_remote", None))
            and algo.is_standalone_remote()
        )
        backend = getattr(runner, "attn_backend", None)
        target_fia = bool(
            backend is not None
            and getattr(backend, "_use_target_tree_paged_fia", lambda: False)()
        )
        maps = getattr(self, "_target_fia_maps", None) or {}
        has_map = any(info is not None for info in maps.values())
        graphs_ok = bool(getattr(self, "graphs", None)) and has_map
        reason = None
        if requested and is_sr_target and target_fia and graphs_ok:
            try:
                device_id = int(torch.npu.current_device())
            except Exception as exc:
                if is_device_context_error(exc):
                    raise
                reason = f"current_device failed: {exc}"
            else:
                self._npu_graph_device_id = device_id
                self._npu_sr_target_update_overlap = True
        elif requested and not is_sr_target:
            reason = "not sr target"
        elif requested and not target_fia:
            reason = "not tree_paged_fia"
        elif requested and not getattr(self, "graphs", None):
            reason = "no graphs"
        elif requested and not has_map:
            reason = "no target fia map"
        if not requested:
            return
        extra = f" reason={reason}" if reason else ""
        impl = getattr(backend, "tree_attention_impl", None)
        logger.info(
            "NPU SR target update/replay overlap requested=%s effective=%s "
            "implementation=%s device=%s%s",
            True,
            bool(self._npu_sr_target_update_overlap),
            impl,
            self._npu_graph_device_id,
            extra,
        )

    def _update_target_tree_fia_inputs(self, info, kv_lens, sample=None):
        def _fill():
            fill_fia_cpu_update_payload(
                info["payload"],
                [list(kv_lens)],
                info["step_ids"],
                info["attr_name"],
            )

        measure_call(sample, "payload_fill", _fill)
        graph = self._tree_replay_graph
        if graph is None:
            raise NpuGraphPreparationError(
                "target tree FIA replay graph missing", scope="graph"
            )
        payload = info["payload"]
        measure_call(
            sample,
            "update_call",
            lambda: graph.update(cpu_update_input=payload),
        )

    def _target_graph_host_metrics(self):
        metrics = getattr(self, "_sr_graph_host_metrics", None)
        if (
            metrics is None
            or getattr(metrics, "role", None) != "Target"
            or not getattr(metrics, "active", False)
        ):
            return None
        return metrics

    def _log_sr_target_overlap_submit(self, overlap: bool) -> None:
        if not getattr(self, "_npu_sr_target_update_overlap_requested", False):
            return
        if getattr(self, "_logged_sr_target_overlap_submit", False):
            return
        self._logged_sr_target_overlap_submit = True
        logger.info(
            "NPU SR target update/replay submit effective=%s implementation=%s "
            "device=%s overlap=%s",
            bool(getattr(self, "_npu_sr_target_update_overlap", False)),
            self._current_tree_attention_impl(),
            getattr(self, "_npu_graph_device_id", None),
            bool(overlap),
        )

    def _replay_target_tree_fia(self, info, kv_lens, graph, graph_key, kv_bucket):
        """Fill on the caller thread, then update/replay through the shared helper."""
        overlap = bool(getattr(self, "_npu_sr_target_update_overlap", False))
        metrics = self._target_graph_host_metrics()
        sample = None
        if metrics is not None:
            try:
                sample = begin_graph_host_sample(
                    {
                        "graph_phase": "target_verify",
                        "round_id": int(metrics.rounds) + 1,
                        "graph_key": graph_key,
                        "implementation": self._current_tree_attention_impl(),
                        "raw_bs": int(getattr(self, "raw_bs", 0) or 0),
                        "capture_bs": int(self.bs),
                        "kv_bucket": kv_bucket,
                        "overlap": overlap,
                    }
                )
            except Exception:
                sample = None
        payload = info["payload"]
        device_id = getattr(self, "_npu_graph_device_id", None)

        def _serial_submit():
            self._log_sr_target_overlap_submit(False)
            run_npu_graph_update_and_replay(
                lambda: self._update_target_tree_fia_inputs(info, kv_lens, sample),
                lambda: measure_call(sample, "replay_call", graph.replay),
                overlap=False,
            )

        def _overlap_submit():
            def _fill():
                fill_fia_cpu_update_payload(
                    payload,
                    [list(kv_lens)],
                    info["step_ids"],
                    info["attr_name"],
                )

            measure_call(sample, "payload_fill", _fill)

            def update():
                torch.npu.set_device(device_id)
                measure_call(
                    sample,
                    "update_call",
                    lambda: graph.update(cpu_update_input=payload),
                )

            def replay():
                measure_call(sample, "replay_call", graph.replay)

            def _submit():
                self._log_sr_target_overlap_submit(True)
                run_npu_graph_update_and_replay(update, replay, overlap=True)

            measure_call(sample, "submit_envelope", _submit)

        def _prepare():
            if overlap:
                _overlap_submit()
            else:
                measure_call(sample, "submit_envelope", _serial_submit)

        try:
            measure_call(sample, "prepare_submit", _prepare)
        except BaseException:
            mark_graph_host_failed(sample)
            raise
        finally:
            record_graph_host_sample_safely(metrics, sample)

    def _update_decode_inputs(self, seq_lens, graph_key):
        torch.npu.set_device(self.model_runner.gpu_id)
        self._update_inputs(seq_lens, graph_key)

    def _update_inputs(self, seq_lens, graph_key=None):
        if isinstance(self.update_attr_type, torch.Tensor):
            seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
            key = self.bs if graph_key is None else graph_key
            graph = self.graphs[key]
            graph.update(cpu_update_input=[{self.update_attr_name: seq_lens}])
            return

        key = self.bs if graph_key is None else graph_key
        values = list(seq_lens)
        payload = self._fia_payloads.get(key)
        if payload is None:
            payload = [{self.update_attr_name: list(values)}]
            self._fia_payloads[key] = payload
        else:
            dest = payload[0].setdefault(self.update_attr_name, [])
            dest[:] = values
        graph = self._tree_replay_graph
        if graph is None:
            graph = self.graphs[key]
        graph.update(cpu_update_input=payload)

    def _replay_update(self, seq_lens, graph_key, errors):
        try:
            self._update_inputs(seq_lens, graph_key)
        except Exception as e:
            errors.append(e)

    def _cache_loc_dtype(self):
        return torch.int32

    def _init_profile_context_and_memory_record(self):
        output_dir = os.path.join(
            os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp"), "graph_capture_profile"
        )
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Profiling starts for graph capture for NPU. Traces will be saved to: {output_dir}"
        )
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        )
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                output_dir, async_mode=True
            ),
            experimental_config=experimental_config,
        )
        return profile_context

    def _post_process_after_profile(self, prof_context):
        # for NPU, profile data will be saved to disk for further analysis.
        pass

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        plan = self._tree_replay_plan
        batch_id = self._tree_replay_batch_id
        stream_idx_saved = self._tree_replay_stream_idx
        try:
            if plan is None or batch_id != id(forward_batch):
                raise NpuGraphPreparationError(
                    "NPU graph replay has no TreeReplayPlan for this batch",
                    scope="graph",
                )
            stream_idx = (
                get_current_stream_idx()
                if getattr(self, "enable_pdmux", False)
                else None
            )
            if stream_idx != stream_idx_saved:
                raise NpuGraphPreparationError(
                    f"NPU graph stream_idx {stream_idx!r} != plan {stream_idx_saved!r}",
                    scope="graph",
                )
            self._assert_tree_replay_graph(plan)
            if not skip_attn_backend_init:
                self.replay_prepare(forward_batch, pp_proxy_tensors)
            else:
                # In speculative decoding, these two fields are still needed.
                self.buffers.input_ids[: self.raw_num_token].copy_(
                    forward_batch.input_ids
                )
                self.buffers.positions[: self.raw_num_token].copy_(
                    forward_batch.positions
                )
            self._assert_tree_replay_graph(plan)

            backend = getattr(self.model_runner, "attn_backend", None)
            is_tree_verify = self._is_tree_verify_batch(forward_batch)
            actual_ntpb = getattr(self, "actual_ntpb", None) or self.num_tokens_per_bs
            kv_bucket = (
                getattr(backend, "_replay_tree_s_cap", None)
                if is_tree_verify and backend is not None
                else None
            )
            if (
                int(self.bs) != plan.capture_bs
                or int(self.raw_bs) != plan.raw_bs
                or int(actual_ntpb) != plan.tokens_per_req
                or kv_bucket != plan.kv_bucket
            ):
                raise NpuGraphPreparationError(
                    f"NPU graph replay shape != plan: bs={self.bs}/{plan.capture_bs} "
                    f"raw_bs={self.raw_bs}/{plan.raw_bs} ntpb={actual_ntpb}/"
                    f"{plan.tokens_per_req} bucket={kv_bucket}/{plan.kv_bucket}",
                    scope="graph",
                )
            graph_key = plan.graph_key
            graph = self._tree_replay_graph

            self.update_attr_name = self._get_update_attr_name()
            self.update_attr_type = self._get_update_attr_type()
            target_fia = bool(
                backend is not None
                and getattr(backend, "_use_target_tree_paged_fia", lambda: False)()
            )
            compact_fia = bool(
                backend is not None
                and getattr(backend, "_use_tree_compact_fia", lambda: False)()
                and not getattr(backend, "_use_tree_shared_prefix", lambda: False)()
                and not target_fia
            )
            skip_fia_update = is_deepseek_nsa(
                self.model_runner.model_config.hf_config
            ) or (is_tree_verify and not compact_fia and not target_fia)
            seq_lens = None
            if not skip_fia_update:
                if is_tree_verify and target_fia:
                    info = getattr(self, "_target_fia_maps", {}).get(graph_key)
                    if info is None:
                        raise NpuGraphPreparationError(
                            "target tree FIA payload missing", scope="graph"
                        )
                    md = getattr(
                        getattr(backend, "forward_metadata", None),
                        "sr_target_tree_fia",
                        None,
                    )
                    if md is None:
                        raise NpuGraphPreparationError(
                            "target tree FIA metadata missing", scope="graph"
                        )
                    kv_lens = list(md.kv_lens_cpu)
                    if len(kv_lens) != int(self.bs):
                        raise NpuGraphPreparationError(
                            f"target tree FIA kv_lens {len(kv_lens)} != "
                            f"capture_bs {self.bs}",
                            scope="graph",
                        )
                    seq_lens = kv_lens
                    self._replay_target_tree_fia(
                        info, kv_lens, graph, graph_key, kv_bucket
                    )
                elif is_tree_verify and compact_fia:
                    ntpb = actual_ntpb
                    capture_rows = int(self.bs) * int(ntpb)
                    kv_lens = getattr(backend, "tree_fia_kv_lens_cpu", None)
                    if kv_lens is None:
                        dest_lens = getattr(
                            getattr(backend, "forward_metadata", None),
                            "tree_verify_kv_lens_t",
                            None,
                        )
                        kv_lens = tree_fia_actual_seq_lengths_kv(
                            dest_lens if dest_lens is not None else [],
                            capture_rows,
                        )
                    elif len(kv_lens) != capture_rows:
                        kv_lens = tree_fia_actual_seq_lengths_kv(
                            kv_lens, capture_rows
                        )
                    seq_lens = kv_lens
                    run_npu_graph_update_and_replay(
                        lambda: self._update_inputs(seq_lens, graph_key),
                        graph.replay,
                    )
                elif forward_batch.forward_mode.is_target_verify():
                    ntpb = actual_ntpb
                    seq_lens_cpu = forward_batch.seq_lens.cpu() + ntpb
                    seq_lens = seq_lens_cpu.tolist() + [0] * (self.bs - self.raw_bs)
                    run_npu_graph_update_and_replay(
                        lambda: self._update_inputs(seq_lens, graph_key),
                        graph.replay,
                    )
                else:
                    seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (
                        self.bs - self.raw_bs
                    )
                    overlap_decode = (
                        self._plain_ar_update_overlap
                        and forward_batch.forward_mode.is_decode()
                        and not forward_batch.is_sr_tail_extend
                    )
                    run_npu_graph_update_and_replay(
                        lambda: self._update_decode_inputs(seq_lens, graph_key),
                        graph.replay,
                        overlap=overlap_decode,
                    )
            else:
                replay_error = None
                try:
                    graph.replay()
                except Exception as exc:
                    replay_error = exc
                if replay_error is not None:
                    raise NpuGraphReplaySubmittedError(
                        "NPU graph update/replay failed"
                    ) from replay_error
            if is_tree_verify:
                self.tree_verify_replay_count += 1
                if (
                    self.tree_verify_replay_count == 1
                    or self.tree_verify_replay_count % 32 == 0
                ):
                    logger.info(
                        "NPU tree verify graph replay count=%s bucket=%s key=%s "
                        "needed_len_max=%s raw_bs=%s capture_bs=%s "
                        "eager_fallback=%s implementation=%s",
                        self.tree_verify_replay_count,
                        kv_bucket,
                        graph_key,
                        (
                            max(seq_lens)
                            if not skip_fia_update
                            and is_tree_verify
                            and (compact_fia or target_fia)
                            else None
                        ),
                        getattr(self, "raw_bs", None),
                        getattr(self, "bs", None),
                        self.tree_verify_eager_fallback_count,
                        self._current_tree_attention_impl(),
                    )

            output = self.output_buffers[graph_key]
            if isinstance(output, LogitsProcessorOutput):
                if self.is_dllm:
                    next_token_logits = None
                    full_logits = output.full_logits[: self.raw_num_token]
                else:
                    full_logits = None
                    next_token_logits = output.next_token_logits[: self.raw_num_token]
                return LogitsProcessorOutput(
                    next_token_logits=next_token_logits,
                    full_logits=full_logits,
                    hidden_states=(
                        output.hidden_states[: self.raw_num_token]
                        if output.hidden_states is not None
                        else None
                    ),
                )
            else:
                assert isinstance(output, PPProxyTensors)
                return PPProxyTensors(
                    {k: v[: self.bs] for k, v in output.tensors.items()}
                )
        finally:
            self._clear_tree_replay_plan()
