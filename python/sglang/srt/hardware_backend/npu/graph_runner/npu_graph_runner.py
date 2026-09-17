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
    run_npu_graph_update_and_replay,
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
        super().__init__(model_runner)
        self.model_runner = model_runner
        if not hasattr(self, "attr_name"):
            self._init_arch_map()
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        self.tree_verify_replay_count = 0
        self.tree_verify_eager_fallback_count = 0
        self._clear_tree_replay_plan()

    def _ensure_capture_attrs(self):
        if not hasattr(self, "attr_name"):
            self._init_arch_map()
        if getattr(self, "_fia_payloads", None) is None:
            self._fia_payloads = {}

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
        if not super().can_run(forward_batch):
            return False
        if not self.capture_bs:
            return False
        backend = getattr(self.model_runner, "attn_backend", None)
        is_tree_verify = self._is_tree_verify_batch(forward_batch)
        kv_bucket = None
        fn = getattr(backend, "tree_slot_graph_can_run", None)
        if fn is not None:
            ok = bool(fn(forward_batch))
            if not ok:
                self.tree_verify_eager_fallback_count += 1
                return False
            if is_tree_verify:
                kv_bucket = getattr(backend, "_replay_tree_s_cap", None)
        tokens_per_req = self._get_actual_ntpb(forward_batch)
        raw_bs, capture_bs = self._padded_capture_bs(forward_batch, tokens_per_req)
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
            if is_tree_verify:
                self.tree_verify_eager_fallback_count += 1
            return False
        if (
            graph_key in self._tree_attention_impls
            and self._tree_attention_impls[graph_key]
            != self._current_tree_attention_impl()
        ):
            return False
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
        if shared:
            backend._shared_capture_width = getattr(self, "_active_capture_extra", None)
        try:
            graph, out = super().capture_one_batch_size(
                bs, forward, stream_idx, ntpb_override
            )
        finally:
            if shared:
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
        n_lens = int(bs) * int(ntpb) if extra is not None else int(bs)
        self._fia_payloads[key] = [{self.update_attr_name: [1] * n_lens}]
        return graph, out

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
            compact_fia = bool(
                backend is not None
                and getattr(backend, "_use_tree_compact_fia", lambda: False)()
                and not getattr(backend, "_use_tree_shared_prefix", lambda: False)()
            )
            skip_fia_update = is_deepseek_nsa(
                self.model_runner.model_config.hf_config
            ) or (is_tree_verify and not compact_fia)
            seq_lens = None
            if not skip_fia_update:
                if is_tree_verify and compact_fia:
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
                elif forward_batch.forward_mode.is_target_verify():
                    ntpb = actual_ntpb
                    seq_lens_cpu = forward_batch.seq_lens.cpu() + ntpb
                    seq_lens = seq_lens_cpu.tolist() + [0] * (self.bs - self.raw_bs)
                else:
                    seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (
                        self.bs - self.raw_bs
                    )
                run_npu_graph_update_and_replay(
                    lambda: self._update_inputs(seq_lens, graph_key),
                    graph.replay,
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
                        "needed_len_max=%s eager_fallback=%s implementation=%s",
                        self.tree_verify_replay_count,
                        kv_bucket,
                        graph_key,
                        (
                            max(seq_lens)
                            if not skip_fia_update and is_tree_verify and compact_fia
                            else None
                        ),
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
