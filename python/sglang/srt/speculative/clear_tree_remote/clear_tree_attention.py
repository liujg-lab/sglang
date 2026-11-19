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
"""Cascade attention metadata for one ClearTree forward.

The kernel and the reference live in ``clear_tree_cascade_kernel``. This
module owns the static buffers those launches read. A forward brings its own
rows; nothing is appended across layers. Startup decides whether every
batch uses this path. A batch that does not fit the buffers fixed before
capture is a program error.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.speculative.clear_tree_remote.clear_tree_cascade_kernel import (
    MAX_SPLITS,
    fill_cascade_meta,
    launch,
    splits_for,
)

_WORKSPACE = None
_support_cache = {}


def attention_layer_accepts(layer) -> bool:
    """True when one attention layer can use the cascade kernel."""
    if getattr(layer, "is_cross_attention", False):
        return False
    attn_type = getattr(layer, "attn_type", None)
    if attn_type is None:
        return False
    from sglang.srt.layers.radix_attention import AttentionType

    if attn_type != AttentionType.DECODER:
        return False
    window = int(getattr(layer, "sliding_window_size", -1) or -1)
    if window > 0:
        return False
    if getattr(layer, "k_scale", None) is not None:
        return False
    if getattr(layer, "sinks", None) is not None:
        return False
    qk_dim = int(getattr(layer, "qk_head_dim", 0) or 0)
    v_dim = int(getattr(layer, "v_head_dim", 0) or 0)
    if qk_dim != v_dim or qk_dim < 16 or qk_dim > 256:
        return False
    if int(getattr(layer, "xai_temperature_len", -1) or -1) > 0:
        return False
    if float(getattr(layer, "logit_cap", 0.0) or 0.0) != 0.0:
        return False
    q_heads = int(getattr(layer, "tp_q_head_num", 0) or 0)
    kv_heads = int(getattr(layer, "tp_k_head_num", 0) or 0)
    if q_heads < 1 or kv_heads < 1 or q_heads % kv_heads != 0:
        return False
    return True


def cascade_enabled(model_runner) -> bool:
    """True when this process uses cascade for every verify and bonus forward."""
    if model_runner is None:
        return False
    key = id(model_runner)
    cached = _support_cache.get(key)
    if cached is not None:
        return cached
    supported = _compute_support(model_runner)
    _support_cache[key] = supported
    return supported


def _compute_support(model_runner) -> bool:
    args = getattr(model_runner, "server_args", None)
    page = getattr(args, "page_size", None) if args is not None else None
    if page is not None and int(page) != 1:
        return False
    backend = getattr(model_runner, "attn_backend", None)
    if backend is None or type(backend).__name__ != "TritonAttnBackend":
        return False
    if getattr(backend, "enable_deterministic", False):
        return False
    if getattr(backend, "use_mla", False):
        return False
    model = getattr(model_runner, "model", None)
    if model is None or not hasattr(model, "modules"):
        return False
    from sglang.srt.layers.radix_attention import RadixAttention

    found = False
    q_heads = None
    v_dim = None
    for module in model.modules():
        if not isinstance(module, RadixAttention):
            continue
        found = True
        if not attention_layer_accepts(module):
            return False
        if q_heads is None:
            q_heads = int(module.tp_q_head_num)
            v_dim = int(module.v_head_dim)
        elif int(module.tp_q_head_num) != q_heads or int(module.v_head_dim) != v_dim:
            return False
        if int(getattr(backend, "num_head", q_heads) or q_heads) != q_heads:
            return False
        if int(getattr(backend, "v_head_dim", v_dim) or v_dim) != v_dim:
            return False
    return found


def _query_capacities(backend) -> tuple:
    """Widest layer and enabled bonus, installed before the first capture."""
    from sglang.srt.speculative.clear_tree_remote.clear_tree_config import (
        bonus_prediction_enabled,
    )
    from sglang.srt.speculative.clear_tree_remote.clear_tree_width import (
        BONUS_NTPBS,
        LAYER_WIDTHS,
    )

    width = max(int(getattr(backend, "num_draft_tokens", 0) or 0), max(LAYER_WIDTHS))
    bonus = max(BONUS_NTPBS) if bonus_prediction_enabled() else 0
    return width, bonus


def tree_limit(backend) -> int:
    """Hold ``(max_depth - 1) * max W`` plus the enabled ``max K + 1`` bonus.

    Without a usable step count, reserve one layer. A disabled bonus does
    not enlarge the tree columns.
    """
    width, bonus = _query_capacities(backend)
    steps = int(getattr(backend, "speculative_num_steps", 0) or 0)
    return max(steps - 1, 1) * width + bonus


def batch_ceiling(server_args) -> int:
    """Largest verify batch this process will capture or run."""
    if server_args is None:
        return 1
    listed = [int(item) for item in (getattr(server_args, "cuda_graph_bs", None) or [])]
    running = int(getattr(server_args, "max_running_requests", None) or 1)
    graph_max = int(getattr(server_args, "cuda_graph_max_bs", None) or 0)
    if graph_max:
        running = min(running, graph_max)
    return max([1, running, *listed])


class _Workspace:
    def __init__(self, device, max_queries, heads, dim, max_prefix, max_tree):
        self.device = torch.device(device)
        self.max_queries = int(max_queries)
        self.heads = int(heads)
        self.dim = int(dim)
        self.max_prefix = int(max_prefix)
        self.max_tree = int(max_tree)
        self.max_bs = 1
        self.max_width = 1
        self.frozen = False
        self.splits = 1
        self.launch_bs = 0
        self.launch_width = 0
        self.prefix_len = torch.zeros(self.max_queries, dtype=torch.int32, device=self.device)
        self.tree_mask = torch.zeros(
            (self.max_queries, self.max_tree), dtype=torch.uint8, device=self.device
        )
        self.prefix_rows = torch.ones(self.max_queries, dtype=torch.int32, device=self.device)
        self.logits = torch.zeros(
            (self.max_queries, self.heads, MAX_SPLITS, self.dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.lse = torch.full(
            (self.max_queries, self.heads, MAX_SPLITS),
            -1.0e20,
            dtype=torch.float32,
            device=self.device,
        )

    @classmethod
    def allocate(cls, device, max_queries, heads, dim, context, tree_limit):
        context = max(int(context), 1)
        tree_limit = max(int(tree_limit), 1)
        return cls(device, max_queries, heads, dim, context, tree_limit)


def ensure_workspace(backend, query_width: int):
    """Allocate the static buffers once. Later calls only reuse them."""
    global _WORKSPACE
    device = torch.device(backend.device)
    width, bonus = _query_capacities(backend)
    max_width = max(int(query_width), width, bonus, 1)
    max_bs = max(int(getattr(backend, "max_capture_bs", 1) or 1), 1)
    max_q = max_width * max_bs
    heads = int(backend.num_head)
    dim = int(backend.v_head_dim)
    context = int(getattr(backend, "max_context_len", 1) or 1)
    columns = tree_limit(backend)
    ws = _WORKSPACE
    need = (
        ws is None
        or ws.device != device
        or ws.max_queries < max_q
        or ws.heads != heads
        or ws.dim != dim
        or ws.max_prefix < context
        or ws.max_tree < columns
        or getattr(ws, "max_bs", 0) < max_bs
    )
    if need:
        if ws is not None and getattr(ws, "frozen", False):
            raise RuntimeError(
                "ClearTree cascade batch exceeds the capacity fixed at startup"
            )
        ws = _Workspace.allocate(device, max_q, heads, dim, context, columns)
        ws.max_bs = max_bs
        ws.max_width = max_width
        _WORKSPACE = ws
    return ws


class ClearTreeAttention:
    """Duck-typed ``target_verify_attention`` for one forward or one capture."""

    def __init__(self, rows=None, capture_bs: int = 0, capture_width: int = 0):
        self.rows = rows
        self.capture_bs = int(capture_bs)
        self.capture_width = int(capture_width)
        self.ready = False

    def prepare(
        self,
        backend,
        batch=None,
        bs=None,
        req_pool_indices=None,
        seq_lens=None,
        spec_info=None,
    ) -> bool:
        """Fill the static buffers. Overflow raises; there is no extend fallback."""
        del req_pool_indices, spec_info
        self.ready = False
        if batch is not None:
            pool = getattr(batch, "req_pool_indices", None)
            if pool is not None:
                bs = int(pool.shape[0])
            else:
                bs = int(getattr(batch, "batch_size", 0) or 0)
            if seq_lens is None:
                seq_lens = batch.seq_lens
        if bs is None:
            raise RuntimeError("ClearTree cascade prepare did not receive a batch size")
        bs = int(bs)
        rows = self.rows
        width = int(self.capture_width if rows is None else rows.width)
        if bs < 1 or width < 1:
            raise RuntimeError(
                f"ClearTree cascade prepare received batch {bs} width {width}"
            )
        if seq_lens is None:
            raise RuntimeError("ClearTree cascade prepare did not receive seq_lens")
        ws = ensure_workspace(backend, width)
        raw = 0 if rows is None else len(rows.prefix_lens)
        mask = None if rows is None else rows.tree_mask
        cols = 0 if mask is None else int(mask.shape[1])
        if (
            bs > int(getattr(ws, "max_bs", bs))
            or bs * width > int(ws.max_queries)
            or width > int(getattr(ws, "max_width", width))
            or cols > int(ws.max_tree)
        ):
            raise RuntimeError(
                f"ClearTree cascade batch {bs} width {width} columns {cols} "
                f"exceeds capacity bs={getattr(ws, 'max_bs', None)} "
                f"width={getattr(ws, 'max_width', None)} queries={ws.max_queries} "
                f"tree={ws.max_tree}"
            )
        fill_cascade_meta(
            ws.prefix_len,
            ws.tree_mask,
            ws.prefix_rows,
            prefix_lens=None if rows is None else rows.prefix_lens,
            tree_mask=mask,
            prefix_rows=None if rows is None else rows.prefix_rows,
            seq_lens=seq_lens,
            width=width,
            padded_bs=bs,
        )
        kv_heads = int(getattr(backend, "num_kv_head", 1) or 1)
        sm = int(getattr(backend, "device_core_count", 1) or 1)
        ws.splits = splits_for(bs, kv_heads, max(sm, 1))
        ws.launch_bs = bs
        ws.launch_width = width
        self.ready = True
        return True

    def forward(self, q, k, v, o, layer, forward_batch, sinks=None):
        del k, v
        if sinks is not None:
            raise RuntimeError("ClearTree cascade does not support attention sinks")
        if not self.ready:
            raise RuntimeError("ClearTree cascade attention forward ran before prepare")
        ws = _WORKSPACE
        if ws is None:
            raise RuntimeError("ClearTree cascade attention was launched before prepare")
        qh = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        oh = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        pool = forward_batch.token_to_kv_pool
        scale = layer.scaling
        if scale is None:
            scale = layer.qk_head_dim**-0.5
        launch(
            qh,
            oh,
            pool.get_key_buffer(layer.layer_id),
            pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            ws.prefix_len,
            ws.tree_mask,
            ws.prefix_rows,
            ws.logits,
            ws.lse,
            float(scale),
            int(layer.tp_k_head_num),
            int(ws.launch_width),
            int(ws.splits),
            int(ws.launch_bs),
        )
        return o


def capture_attention(model_runner, query_width: int, bs: int):
    """Attention object for a captured draft verify graph, or None."""
    if int(bs) < 1 or not cascade_enabled(model_runner):
        return None
    backend = model_runner.attn_backend
    backend.max_capture_bs = batch_ceiling(getattr(model_runner, "server_args", None))
    workspace = ensure_workspace(backend, int(query_width))
    workspace.frozen = True
    return ClearTreeAttention(
        rows=None,
        capture_bs=int(bs),
        capture_width=int(query_width),
    )
