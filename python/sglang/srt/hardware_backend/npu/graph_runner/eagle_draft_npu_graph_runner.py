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
    validate_draft_graph_step_kv_lens,
    validate_tree_draft_fia_records,
)
from sglang.srt.speculative.tree_attn_fallback import (
    TREE_DRAFT_CAPTURE_BS_ENV,
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
        self.tree_graph_replay_count = 0
        self.tree_eager_fallback_count = 0
        self.tree_graph_disabled_reason = None
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
        if self._slot_gather_graph:
            logger.info(
                "NPU tree draft graphs use token-level slot gather "
                "page_size=%s topk=%s compact_fia=%s",
                page_size,
                topk,
                self._tree_compact_fia,
            )
        super().__init__(eagle_worker)

    def _capture_extra_keys(self):
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

    def can_run(self, forward_batch: ForwardBatch):
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
        if not is_bs_supported:
            return False
        if not getattr(self, "_slot_gather_graph", False):
            return True
        runner = getattr(self, "model_runner", None)
        backend = None
        if runner is not None:
            backend = getattr(runner, "draft_attn_backend", None) or getattr(
                runner, "attn_backend", None
            )
            inner = getattr(backend, "attn_backends", None)
            if inner:
                backend = inner[0]
        fn = getattr(backend, "tree_slot_graph_can_run", None)
        if fn is None:
            return True
        ok = bool(fn(forward_batch))
        if not ok:
            self.tree_eager_fallback_count += 1
            return False
        extra = getattr(backend, "_replay_tree_s_cap", None)
        if extra is not None:
            suffix = f"_s{int(extra)}"
            if not any(
                isinstance(k, str) and str(k).endswith(suffix) for k in self.graphs
            ):
                self.tree_eager_fallback_count += 1
                return False
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
        graph, out = super().capture_one_batch_size(num_seqs, forward, stream_idx)
        skip_fia = self.tree_graph_disabled_reason or (
            self._slot_gather_graph and not self._tree_compact_fia
        )
        if skip_fia:
            return graph, out
        self.update_attr_name = self._get_update_attr_name()
        n_steps = max(int(self.speculative_num_steps) - 1, 0)
        extra = getattr(self, "_active_capture_extra", None)
        map_key = self._make_graph_key(int(num_seqs), extra=extra)
        try:
            num_layers = self._num_model_layers()
            records = _iter_graph_dispatch_records(graph)
            n_records, step_ids = validate_tree_draft_fia_records(
                records, n_steps, num_layers, self.update_attr_name
            )
            self._tree_fia_maps[map_key] = {
                "n_records": n_records,
                "n_steps": n_steps,
                "num_layers": num_layers,
                "step_ids": step_ids,
                "bs": int(num_seqs),
                "extra": extra,
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
        if self._slot_gather_graph:
            allow = set(
                parse_tree_draft_capture_bs(os.environ.get(TREE_DRAFT_CAPTURE_BS_ENV))
            )
            self.capture_bs = [b for b in self.capture_bs if b in allow]
            self.compile_bs = [b for b in self.compile_bs if b in self.capture_bs]
            self.max_bs = max(self.capture_bs) if self.capture_bs else 0
            logger.info(
                "NPU tree draft capture_bs restricted to %s", self.capture_bs
            )
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
                "NPU tree draft slot-gather graphs ready: graphs=%s "
                "tree_graph_replay_count=%s tree_eager_fallback_count=%s",
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
        self.update_attr_name = self._get_update_attr_name()
        self.update_attr_type = self._get_update_attr_type()
        backend = getattr(self.model_runner, "draft_attn_backend", None) or getattr(
            self.model_runner, "attn_backend", None
        )
        inner = getattr(backend, "attn_backends", None)
        if inner:
            backend = inner[0]
        extra = getattr(backend, "_replay_tree_s_cap", None)
        graph_key = self._make_graph_key(self.bs, extra=extra)
        if graph_key not in self.graphs:
            raise NpuGraphPreparationError(
                f"tree draft graph miss key={graph_key!r}",
                scope="graph",
            )
        graph = self.graphs[graph_key]
        self.output_buffers[self.bs] = self.output_buffers[graph_key]
        skip_fia = is_deepseek_nsa(
            self.model_runner.model_config.hf_config
        ) or (self._slot_gather_graph and not self._tree_compact_fia)
        if skip_fia:
            graph.replay()
            self.tree_graph_replay_count += 1
            return

        if forward_batch.seq_lens_cpu is None:
            raise NpuGraphPreparationError(
                "tree draft graph replay requires seq_lens_cpu",
                scope="graph",
            )

        fia_map = self._tree_fia_maps.get(graph_key)
        if fia_map is None:
            raise NpuGraphPreparationError(
                f"tree draft graph key={graph_key!r} has no FIA map",
                scope="graph",
            )

        prefix_lens = forward_batch.seq_lens_cpu[: self.raw_bs]
        step_lens_list = []
        n_steps = int(fia_map["n_steps"])
        step_backends = inner if inner else None
        try:
            for speculative_step_id in range(n_steps):
                seq_lens = None
                if self._tree_compact_fia and step_backends is not None:
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
                else:
                    capture_rows = int(self.bs) * max(int(self.topk), 1)
                    seq_lens = tree_fia_actual_seq_lengths_kv(seq_lens, capture_rows)
                step_lens_list.append(seq_lens)
        except NpuGraphPreparationError:
            raise
        except (TypeError, ValueError) as e:
            raise NpuGraphPreparationError(
                f"tree draft step KV lengths invalid: {e}",
                scope="graph",
            ) from e

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
        log_key = graph_key
        if log_key not in self._logged_tree_fia_update_bs:
            logger.info(
                "NPU tree draft graph FIA updates: key=%s bs=%s raw_bs=%s "
                "records=%s steps=%s layers=%s updates=%s bucket=%s",
                graph_key,
                self.bs,
                self.raw_bs,
                n_records,
                n_steps,
                num_layers,
                len(cpu_update_input),
                extra,
            )
            self._logged_tree_fia_update_bs.add(log_key)

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
        if self.tree_graph_replay_count == 1 or self.tree_graph_replay_count % 32 == 0:
            logger.info(
                "NPU tree draft graph replay count=%s key=%s bucket=%s "
                "eager_fallback=%s",
                self.tree_graph_replay_count,
                graph_key,
                extra,
                self.tree_eager_fallback_count,
            )

    def _cache_loc_dtype(self):
        return torch.int32
