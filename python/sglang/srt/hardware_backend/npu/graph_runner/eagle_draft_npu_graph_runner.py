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

import logging
import threading
from typing import TYPE_CHECKING, Dict, Union

import torch

from sglang.srt.configs.model_config import AttentionArch, is_deepseek_nsa
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.spec_utils import (
    NpuGraphReplaySubmittedError,
    build_draft_graph_step_kv_lens,
    expand_fia_cpu_update_inputs,
    resolve_fia_update_count,
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


def count_fia_kv_len_records(graph, attr_name):
    """Count captured graph records that take ``actual_seq_lengths_kv``.

    Returns None when the torch_npu dispatch-record API is missing.
    """
    records = _iter_graph_dispatch_records(graph)
    if records is None:
        return None
    n = 0
    for rec in records:
        update_info = getattr(rec, "update_info", None)
        if isinstance(update_info, dict) and attr_name in update_info:
            n += 1
            continue
        attrs = getattr(rec, "attrs", None)
        if isinstance(attrs, dict) and attr_name in attrs:
            n += 1
            continue
        if attr_name in str(rec):
            n += 1
    return n


class EAGLEDraftNpuGraphRunner(EAGLEDraftCudaGraphRunner):
    def __init__(self, eagle_worker: EAGLEWorker):
        super().__init__(eagle_worker)
        self.update_attr_name = None
        self.update_attr_type = None
        self._logged_tree_fia_update = False
        self._init_arch_map()

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
            raise RuntimeError(
                "model end_layer is unavailable for NPU tree graph update"
            )
        return int(end) - int(start or 0)

    def _replay_update(self, graph, cpu_update_input, errors):
        try:
            graph.update(cpu_update_input=cpu_update_input)
        except Exception as e:
            errors.append(e)

    def _replay(self, forward_batch: ForwardBatch):
        self.update_attr_name = self._get_update_attr_name()
        self.update_attr_type = self._get_update_attr_type()
        if is_deepseek_nsa(self.model_runner.model_config.hf_config):
            self.graphs[self.bs].replay()
            return

        if forward_batch.seq_lens_cpu is None:
            raise RuntimeError("tree draft graph replay requires seq_lens_cpu")

        prefix_lens = forward_batch.seq_lens_cpu[: self.raw_bs]
        num_tokens = self.bs * self.num_tokens_per_bs
        step_lens_list = []
        for speculative_step_id in range(self.speculative_num_steps - 1):
            seq_lens = build_draft_graph_step_kv_lens(
                prefix_lens, self.bs, self.topk, speculative_step_id
            )
            if len(seq_lens) != num_tokens:
                raise RuntimeError(
                    f"tree draft step KV lengths length {len(seq_lens)} "
                    f"!= bs*tokens_per_bs={num_tokens}"
                )
            step_lens_list.append(seq_lens)

        graph = self.graphs[self.bs]
        n_steps = len(step_lens_list)
        num_layers = self._num_model_layers()
        n_records = count_fia_kv_len_records(graph, self.update_attr_name)
        n_updates = resolve_fia_update_count(n_records, n_steps, num_layers)
        cpu_update_input = expand_fia_cpu_update_inputs(
            step_lens_list, num_layers, self.update_attr_name
        )
        if len(cpu_update_input) != n_updates:
            raise RuntimeError(
                f"cpu_update_input length {len(cpu_update_input)} != resolved {n_updates}"
            )
        if not self._logged_tree_fia_update:
            logger.info(
                "NPU tree draft graph FIA updates: records=%s steps=%s layers=%s updates=%s",
                n_records,
                n_steps,
                num_layers,
                n_updates,
            )
            self._logged_tree_fia_update = True

        errors = []
        thread = threading.Thread(
            target=self._replay_update, args=(graph, cpu_update_input, errors)
        )
        thread.start()
        try:
            graph.replay()
        except Exception as e:
            thread.join()
            raise NpuGraphReplaySubmittedError(
                "NPU tree draft graph.replay() failed"
            ) from e
        thread.join()
        if errors:
            raise NpuGraphReplaySubmittedError(
                "NPU tree draft graph update failed"
            ) from errors[0]

    def _cache_loc_dtype(self):
        return torch.int32
