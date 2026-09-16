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
    validate_draft_graph_step_kv_lens,
    validate_tree_draft_fia_records,
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
        self.tree_graph_replay_count = 0
        self.tree_eager_fallback_count = 0
        self.tree_graph_disabled_reason = None
        self._init_arch_map()
        page_size = int(getattr(eagle_worker, "page_size", 1) or 1)
        topk = int(getattr(eagle_worker, "topk", 1) or 1)
        if page_size > 1 and topk > 1:
            self.tree_graph_disabled_reason = (
                "npu tree draft uses token-level slot gather (eager)"
            )
            logger.info(
                "%s page_size=%s topk=%s",
                self.tree_graph_disabled_reason,
                page_size,
                topk,
            )
        super().__init__(eagle_worker)

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
        graph, out = super().capture_one_batch_size(num_seqs, forward, stream_idx)
        if self.tree_graph_disabled_reason:
            return graph, out
        self.update_attr_name = self._get_update_attr_name()
        n_steps = max(int(self.speculative_num_steps) - 1, 0)
        try:
            num_layers = self._num_model_layers()
            records = _iter_graph_dispatch_records(graph)
            n_records, step_ids = validate_tree_draft_fia_records(
                records, n_steps, num_layers, self.update_attr_name
            )
            self._tree_fia_maps[num_seqs] = {
                "n_records": n_records,
                "n_steps": n_steps,
                "num_layers": num_layers,
                "step_ids": step_ids,
            }
        except NpuGraphPreparationError as e:
            if getattr(e, "scope", "graph") == "format":
                self.tree_graph_disabled_reason = str(e)
                logger.warning(
                    "NPU tree graph record format unsupported: %s", e
                )
            else:
                logger.warning("NPU tree graph bs=%s rejected: %s", num_seqs, e)
        return graph, out

    def capture(self):
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
        for bs in list(self.graphs):
            if bs not in self._tree_fia_maps:
                del self.graphs[bs]
                self.output_buffers.pop(bs, None)
        self.capture_bs = [b for b in self.capture_bs if b in self.graphs]
        self.max_bs = max(self.capture_bs) if self.capture_bs else 0
        logger.info(
            "NPU tree draft FIA maps ready: graphs=%s "
            "tree_graph_replay_count=%s tree_eager_fallback_count=%s "
            "tree_graph_disabled_reason=%s",
            sorted(self._tree_fia_maps),
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
        self.update_attr_name = self._get_update_attr_name()
        self.update_attr_type = self._get_update_attr_type()
        if is_deepseek_nsa(self.model_runner.model_config.hf_config):
            self.graphs[self.bs].replay()
            return

        if forward_batch.seq_lens_cpu is None:
            raise NpuGraphPreparationError(
                "tree draft graph replay requires seq_lens_cpu",
                scope="graph",
            )

        fia_map = self._tree_fia_maps.get(self.bs)
        if fia_map is None:
            raise NpuGraphPreparationError(
                f"tree draft graph bs={self.bs} has no FIA map",
                scope="graph",
            )

        prefix_lens = forward_batch.seq_lens_cpu[: self.raw_bs]
        step_lens_list = []
        n_steps = int(fia_map["n_steps"])
        try:
            for speculative_step_id in range(n_steps):
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
                step_lens_list.append(seq_lens)
        except NpuGraphPreparationError:
            raise
        except (TypeError, ValueError) as e:
            raise NpuGraphPreparationError(
                f"tree draft step KV lengths invalid: {e}",
                scope="graph",
            ) from e

        graph = self.graphs[self.bs]
        n_records = int(fia_map["n_records"])
        num_layers = int(fia_map["num_layers"])
        cpu_update_input = expand_fia_cpu_update_inputs(
            step_lens_list, num_layers, self.update_attr_name
        )
        if len(cpu_update_input) != n_records:
            raise NpuGraphPreparationError(
                f"cpu_update_input length {len(cpu_update_input)} != records {n_records}",
                scope="graph",
            )
        if self.bs not in self._logged_tree_fia_update_bs:
            logger.info(
                "NPU tree draft graph FIA updates: bs=%s raw_bs=%s records=%s "
                "steps=%s layers=%s updates=%s",
                self.bs,
                self.raw_bs,
                n_records,
                n_steps,
                num_layers,
                len(cpu_update_input),
            )
            self._logged_tree_fia_update_bs.add(self.bs)

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
        self.tree_graph_replay_count += 1

    def _cache_loc_dtype(self):
        return torch.int32
