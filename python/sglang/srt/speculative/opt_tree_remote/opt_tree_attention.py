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
"""Cascade attention for one OPT draft tree layer.

The committed prefix is dense and shared by every query, so it is split across
CTAs the way flash-decoding splits a decode. The ancestor columns are one extra
split. Partials are merged by the existing decode stage-2 kernel.

Buffers and the kernel stay in this module. Shared attention code only calls
``prepare`` and ``forward`` on ``spec_info.target_verify_attention``.
"""

from __future__ import annotations

from typing import Optional

import torch

# Must match ``decode_attention._MIN_BLOCK_KV``. Stage 2 drops a split unless
# the synthetic length makes that slot non-empty under this rounding.
STAGE2_MIN_BLOCK_KV = 32
MAX_SPLITS = 32
MAX_PREFIX_SPLITS = MAX_SPLITS - 1
BLOCK_N = 32
_MAX_QUERY_BLOCK = 64
_EMPTY_LSE = -1.0e20

_generation = 0
_WORKSPACE = None
_support_cache = {}
# (id(backend), query_width) of bs=1 graphs whose capture admitted the cascade kernel.
_captured_cascade = set()


def _next_generation() -> int:
    global _generation
    _generation += 1
    return _generation


def cascade_query_block(rows: int) -> int:
    """``min(64, next_power_of_2(rows))``. Width 19 with group 2 stays one tile."""
    rows = max(int(rows), 1)
    return min(_MAX_QUERY_BLOCK, 1 << (rows - 1).bit_length())


def cdiv(value: int, block: int) -> int:
    value = int(value)
    block = int(block)
    if block <= 0:
        raise ValueError(f"block must be positive, got {block}")
    return (value + block - 1) // block


def prefix_split_count(prefix_len: int) -> int:
    """How many dense prefix splits stage 1 writes. The tree uses the next slot."""
    prefix_len = int(prefix_len)
    if prefix_len <= 0:
        return 0
    return min(MAX_PREFIX_SPLITS, prefix_len)


def split_range(split: int, prefix_len: int, prefix_splits: int) -> tuple:
    """Token range of one dense prefix split. Splits cover ``[0, prefix_len)``."""
    split = int(split)
    prefix_len = int(prefix_len)
    prefix_splits = int(prefix_splits)
    if prefix_splits <= 0 or split < 0 or split >= prefix_splits:
        return 0, 0
    base = prefix_len // prefix_splits
    rem = prefix_len % prefix_splits
    start = split * base + min(split, rem)
    length = base + (1 if split < rem else 0)
    return start, start + length


def synthetic_kv_len(prefix_splits: int) -> int:
    """Length stage 2 must see so slots ``0 .. prefix_splits`` are all merged."""
    return (int(prefix_splits) + 1) * STAGE2_MIN_BLOCK_KV


def stage2_live_slots(prefix_splits: int, max_splits: int = MAX_SPLITS) -> int:
    """How many partials ``_fwd_kernel_stage2`` merges for this split count."""
    seq_len = synthetic_kv_len(prefix_splits)
    kv_splits = int(prefix_splits) + 1
    if kv_splits <= 0 or seq_len <= 0:
        return 0
    chunk = cdiv(cdiv(seq_len, kv_splits), STAGE2_MIN_BLOCK_KV) * STAGE2_MIN_BLOCK_KV
    live = 0
    for split in range(int(max_splits)):
        start = chunk * split
        end = min(start + chunk, seq_len)
        if end > start:
            live += 1
    return live


class PrefixPlan:
    """Committed-prefix KV slots for one OPT step. Layers and the bonus reuse it."""

    def __init__(self, indices: torch.Tensor, prefix_len: int):
        if indices.dim() != 1:
            raise ValueError(
                f"prefix indices must be 1-D, got {tuple(indices.shape)}"
            )
        prefix_len = int(prefix_len)
        if int(indices.shape[0]) != prefix_len:
            raise ValueError(
                f"prefix indices {int(indices.shape[0])} != prefix_len {prefix_len}"
            )
        self.indices = indices.to(dtype=torch.int64).contiguous()
        self.prefix_len = prefix_len
        self.generation = _next_generation()
        self.prefix_copies = 0


def build_prefix_plan(req_to_token, pool_idx: int, prefix_len: int) -> PrefixPlan:
    """Copy ``req_to_token[pool, :prefix_len]`` once. No host sync."""
    prefix_len = int(prefix_len)
    if prefix_len < 0:
        raise ValueError(f"prefix_len must be >= 0, got {prefix_len}")
    indices = req_to_token[int(pool_idx), :prefix_len]
    return PrefixPlan(indices, prefix_len)


def tree_kv_indices(req_to_token, pool_idx, prefix_len, cached_len, out_cache_loc):
    """Earlier tree slots plus this layer's new slots, in mask-column order."""
    prefix_len = int(prefix_len)
    cached_len = int(cached_len)
    if cached_len < prefix_len:
        raise ValueError(
            f"cached_len {cached_len} is before prefix_len {prefix_len}"
        )
    current = out_cache_loc.reshape(-1).to(dtype=torch.int64)
    if cached_len == prefix_len:
        return current.contiguous()
    earlier = req_to_token[int(pool_idx), prefix_len:cached_len].reshape(-1)
    earlier = earlier.to(dtype=torch.int64)
    return torch.cat((earlier, current), dim=0)


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


def model_supports(model_runner) -> bool:
    """CUDA Triton, page size 1, and every attention layer accepts the kernel."""
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


class _Workspace:
    def __init__(self, device, max_queries, heads, dim, max_prefix, max_tree):
        self.device = torch.device(device)
        self.max_queries = int(max_queries)
        self.heads = int(heads)
        self.dim = int(dim)
        self.max_prefix = int(max_prefix)
        self.max_tree = int(max_tree)
        self.prefix_generation = None
        self.tree_filled = 0
        self.stage2_queries = None
        self.prefix_meta_writes = 0
        self.stage2_writes = 0
        self.prefix_indices = torch.zeros(
            self.max_prefix, dtype=torch.int64, device=self.device
        )
        self.tree_indices = torch.zeros(
            self.max_tree, dtype=torch.int64, device=self.device
        )
        self.tree_mask = torch.zeros(
            (self.max_queries, self.max_tree), dtype=torch.uint8, device=self.device
        )
        self.prefix_rows = torch.ones(
            self.max_queries, dtype=torch.int32, device=self.device
        )
        self.prefix_len = torch.zeros((), dtype=torch.int32, device=self.device)
        self.prefix_splits = torch.zeros((), dtype=torch.int32, device=self.device)
        self.tree_cols = torch.zeros((), dtype=torch.int32, device=self.device)
        self.logits = torch.zeros(
            (self.max_queries, self.heads, MAX_SPLITS, self.dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.lse = torch.full(
            (self.max_queries, self.heads, MAX_SPLITS),
            _EMPTY_LSE,
            dtype=torch.float32,
            device=self.device,
        )
        self.kv_indptr = torch.zeros(
            self.max_queries + 1, dtype=torch.int32, device=self.device
        )
        self.num_kv_splits = torch.ones(
            self.max_queries, dtype=torch.int32, device=self.device
        )
        self.arange = torch.arange(
            self.max_queries + 1, dtype=torch.int32, device=self.device
        )

    @classmethod
    def allocate(cls, device, max_queries, heads, dim, context, tree_limit):
        context = max(int(context), 1)
        tree_limit = max(int(tree_limit), 1)
        return cls(
            device,
            max_queries,
            heads,
            dim,
            max_prefix=context,
            max_tree=tree_limit,
        )


def tree_limit(backend) -> int:
    """Tree columns the cascade workspace must hold.

    A full draft is ``(max_depth - 1) * N`` slots, and the bonus group adds
    ``N + 1``. That is ``speculative_num_steps * num_draft_tokens + 1``.
    Without a step count, the cap is one layer plus the bonus column, not
    the context length.
    """
    width = int(getattr(backend, "num_draft_tokens", 0) or 0)
    steps = int(getattr(backend, "speculative_num_steps", 0) or 0)
    if steps >= 1 and width >= 1:
        return steps * width + 1
    return width + 1


def ensure_workspace(backend, query_width: int) -> _Workspace:
    """Allocate the static buffers once. Later calls only reuse them."""
    global _WORKSPACE
    device = torch.device(backend.device)
    draft = int(getattr(backend, "num_draft_tokens", 0) or 0)
    max_q = max(int(query_width), draft + 1, 1)
    heads = int(backend.num_head)
    dim = int(backend.v_head_dim)
    context = int(getattr(backend, "max_context_len", 1) or 1)
    columns = tree_limit(backend)
    ws = _WORKSPACE
    if (
        ws is None
        or ws.device != device
        or ws.max_queries < max_q
        or ws.heads != heads
        or ws.dim != dim
        or ws.max_prefix < context
        or ws.max_tree < columns
    ):
        ws = _Workspace.allocate(device, max_q, heads, dim, context, columns)
        _WORKSPACE = ws
    return ws


def write_stage2_metadata(ws: _Workspace, queries: int, prefix_splits: int) -> None:
    queries = int(queries)
    seq = synthetic_kv_len(prefix_splits)
    ws.kv_indptr[: queries + 1].copy_(ws.arange[: queries + 1] * seq)
    ws.num_kv_splits[:queries].fill_(int(prefix_splits) + 1)


def _mask_query_width(tree_mask):
    """Query rows of a noted tree mask, or None when the mask cannot name a graph."""
    if tree_mask is None or not hasattr(tree_mask, "dim") or tree_mask.dim() != 2:
        return None
    queries = int(tree_mask.shape[0])
    if queries < 1:
        return None
    return queries


def _replay_uses_captured_cascade(backend, tree_mask) -> bool:
    """True when this replay would run a cascade kernel already frozen into a graph."""
    backend_id = id(backend)
    width = _mask_query_width(tree_mask)
    if width is None:
        return any(stored == backend_id for stored, _stored_width in _captured_cascade)
    return (backend_id, width) in _captured_cascade


class OptTreeAttention:
    """Duck-typed ``target_verify_attention`` owned by one OPT draft step."""

    def __init__(
        self,
        plan: Optional[PrefixPlan],
        active: bool,
        capture_query_width: int = 0,
    ):
        self.plan = plan
        self.active = bool(active)
        self.capture_query_width = int(capture_query_width)
        self._tree_mask = None
        self._tree_indices = None
        self._prefix_rows = None
        self._spec_mask_complete = True
        self.ready = False

    def note_layer(
        self, tree_mask, tree_indices, prefix_rows=None, spec_mask_complete=True
    ) -> None:
        self._tree_mask = tree_mask
        self._tree_indices = tree_indices
        self._prefix_rows = prefix_rows
        self._spec_mask_complete = bool(spec_mask_complete)

    def prepare(
        self,
        backend,
        batch=None,
        bs=None,
        req_pool_indices=None,
        seq_lens=None,
        spec_info=None,
    ) -> bool:
        """Fill static buffers. False leaves the caller's metadata path unchanged.

        An incomplete spec mask has no ordinary fallback. Replay of a bs=1
        graph that already captured this cascade kernel also raises when the
        fill does not fit.
        """
        del req_pool_indices, seq_lens, spec_info
        self.ready = False
        if not self.active:
            return False
        if batch is not None:
            bs = int(getattr(batch, "batch_size", 0) or 0)
        if bs is None or int(bs) != 1:
            return False
        if self.plan is None:
            ok = self._fill_capture(backend)
            if ok:
                _captured_cascade.add((id(backend), int(self.capture_query_width)))
        else:
            ok = self._fill_runtime(backend)
        if not ok and self.plan is not None and (
            not self._spec_mask_complete
            or (
                batch is None
                and _replay_uses_captured_cascade(backend, self._tree_mask)
            )
        ):
            raise RuntimeError("OPT cascade prepare failed without a usable fallback")
        self.ready = bool(ok)
        return bool(ok)

    def _fill_capture(self, backend) -> bool:
        queries = int(self.capture_query_width)
        if queries < 1:
            return False
        ws = ensure_workspace(backend, queries)
        if queries > ws.max_queries or ws.max_prefix < 1 or ws.max_tree < queries:
            return False
        ws.prefix_len.fill_(1)
        ws.prefix_splits.fill_(1)
        ws.tree_cols.fill_(queries)
        ws.prefix_indices[:1].zero_()
        ws.tree_indices[:queries].zero_()
        ws.tree_mask[:queries, :queries].fill_(1)
        ws.prefix_rows[:queries].fill_(1)
        write_stage2_metadata(ws, queries, 1)
        ws.prefix_generation = None
        return True

    def _fill_runtime(self, backend) -> bool:
        plan = self.plan
        mask = self._tree_mask
        indices = self._tree_indices
        if plan is None or mask is None or indices is None:
            return False
        if mask.dim() != 2 or indices.dim() != 1:
            return False
        queries = int(mask.shape[0])
        cols = int(mask.shape[1])
        n_idx = int(indices.shape[0])
        if queries < 1 or cols < 1 or n_idx < 1:
            return False
        rows = self._prefix_rows
        if rows is not None and int(rows.shape[0]) != queries:
            return False
        ws = ensure_workspace(backend, queries)
        if (
            plan.prefix_len > ws.max_prefix
            or cols > ws.max_tree
            or queries > ws.max_queries
        ):
            return False
        new_gen = ws.prefix_generation != plan.generation
        filled = 0 if new_gen else int(ws.tree_filled)
        if n_idx == cols:
            append = False
        elif cols == filled + n_idx:
            append = True
        else:
            return False
        if new_gen:
            if plan.prefix_len > 0:
                ws.prefix_indices[: plan.prefix_len].copy_(plan.indices)
            ws.prefix_generation = plan.generation
            plan.prefix_copies += 1
            splits = prefix_split_count(plan.prefix_len)
            ws.prefix_len.fill_(int(plan.prefix_len))
            ws.prefix_splits.fill_(int(splits))
            ws.prefix_meta_writes += 1
        else:
            splits = prefix_split_count(plan.prefix_len)
        if append:
            ws.tree_indices[filled:cols].copy_(indices)
        else:
            ws.tree_indices[:cols].copy_(indices)
        ws.tree_filled = cols
        ws.tree_cols.fill_(int(cols))
        ws.tree_mask[:queries, :cols].copy_(mask.to(dtype=torch.uint8))
        if rows is None:
            ws.prefix_rows[:queries].fill_(1)
        else:
            ws.prefix_rows[:queries].copy_(rows.to(dtype=torch.int32))
        if new_gen or ws.stage2_queries != queries:
            write_stage2_metadata(ws, queries, splits)
            ws.stage2_queries = queries
            ws.stage2_writes += 1
        return True

    def forward(self, q, k, v, o, layer, forward_batch, sinks=None):
        del k, v
        if not self.ready:
            raise RuntimeError("OPT cascade attention forward ran before prepare")
        qh = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        oh = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        pool = forward_batch.token_to_kv_pool
        scale = layer.scaling
        if scale is None:
            scale = layer.qk_head_dim**-0.5
        launch_workspace(
            qh,
            oh,
            pool.get_key_buffer(layer.layer_id),
            pool.get_value_buffer(layer.layer_id),
            float(scale),
            int(layer.tp_k_head_num),
            sinks,
        )
        return o


def capture_attention(model_runner, query_width: int, bs: int):
    """Attention object for a captured draft verify graph, or None."""
    if int(bs) != 1 or not model_supports(model_runner):
        return None
    return OptTreeAttention(
        plan=None,
        active=True,
        capture_query_width=int(query_width),
    )


def _attention_for(runner) -> Optional[OptTreeAttention]:
    scheduler = getattr(runner, "scheduler", None)
    worker = getattr(scheduler, "model_worker", None)
    model_runner = getattr(worker, "model_runner", None)
    if not model_supports(model_runner):
        return None
    attn = getattr(runner, "_opt_cascade", None)
    if attn is None:
        pool = scheduler.req_to_token_pool.req_to_token
        plan = build_prefix_plan(pool, runner.req.req_pool_idx, runner.prefix_len)
        attn = OptTreeAttention(plan, active=True)
        runner._opt_cascade = attn
    return attn


def _workspace_holds_tree(attn, tree_cols: int) -> bool:
    """True when this prefix generation's earlier tree columns are already stored."""
    ws = _WORKSPACE
    return (
        ws is not None
        and attn.plan is not None
        and ws.prefix_generation == attn.plan.generation
        and int(ws.tree_filled) == int(tree_cols)
    )


def bind_opt_cascade(
    spec,
    runner,
    out_cache_loc,
    tree_mask,
    prefix_rows=None,
    cached_len=None,
    append_slots=False,
) -> None:
    """Attach this step's cascade object and this layer's tree columns.

    ``append_slots`` records only this layer's new slots when the workspace
    already holds this prefix generation's earlier tree columns. Otherwise the
    call writes every tree column, including slots left by a batched forward.
    """
    attn = _attention_for(runner)
    if attn is None:
        return
    if tree_mask is None or tree_mask.dim() != 2:
        raise RuntimeError(
            f"OPT cascade tree mask must be [queries, cols], got {None if tree_mask is None else tuple(tree_mask.shape)}"
        )
    cached = int(runner.cached_len if cached_len is None else cached_len)
    pool = runner.scheduler.req_to_token_pool.req_to_token
    device = pool.device
    prefix = int(runner.prefix_len)
    appended = append_slots and _workspace_holds_tree(attn, cached - prefix)
    if appended:
        indices = (
            out_cache_loc.reshape(-1).to(dtype=torch.int64, device=device).contiguous()
        )
    else:
        indices = tree_kv_indices(
            pool,
            runner.req.req_pool_idx,
            prefix,
            cached,
            out_cache_loc,
        )
    mask = tree_mask.to(device=device, dtype=torch.uint8).contiguous()
    if not appended and int(indices.shape[0]) != int(mask.shape[1]):
        raise RuntimeError(
            f"OPT cascade tree indices {int(indices.shape[0])} != "
            f"mask columns {int(mask.shape[1])}"
        )
    rows = None
    if prefix_rows is not None:
        rows = prefix_rows.to(device=device, dtype=torch.int32).reshape(-1).contiguous()
        if int(rows.shape[0]) != int(mask.shape[0]):
            raise RuntimeError(
                f"OPT cascade prefix rows {int(rows.shape[0])} != "
                f"queries {int(mask.shape[0])}"
            )
    attn.note_layer(mask, indices, rows, spec_mask_complete=rows is not None)
    spec.target_verify_attention = attn


def launch_workspace(q, o, k_buffer, v_buffer, sm_scale, kv_heads, sinks=None):
    ws = _WORKSPACE
    if ws is None:
        raise RuntimeError("OPT cascade attention was launched before prepare")
    queries = int(q.shape[0])
    _launch(
        q,
        o,
        k_buffer,
        v_buffer,
        ws.prefix_indices,
        ws.tree_indices,
        ws.tree_mask,
        ws.prefix_rows,
        ws.prefix_len,
        ws.prefix_splits,
        ws.tree_cols,
        ws.logits[:queries],
        ws.lse[:queries],
        ws.kv_indptr[: queries + 1],
        ws.num_kv_splits[:queries],
        sm_scale,
        int(kv_heads),
        sinks,
    )


def run_cascade_attention(
    q,
    k_cache,
    v_cache,
    prefix_indices,
    tree_indices,
    tree_mask,
    prefix_rows,
    sm_scale,
):
    """Reference entry used by tests. Does not touch the captured workspace."""
    queries, heads, dim = q.shape
    cols = int(tree_indices.shape[0])
    prefix_len = int(prefix_indices.shape[0])
    splits = prefix_split_count(prefix_len)
    device = q.device
    logits = torch.empty(
        (queries, heads, MAX_SPLITS, dim), dtype=torch.float32, device=device
    )
    lse = torch.full(
        (queries, heads, MAX_SPLITS), _EMPTY_LSE, dtype=torch.float32, device=device
    )
    kv_indptr = torch.arange(queries + 1, dtype=torch.int32, device=device)
    kv_indptr.mul_(synthetic_kv_len(splits))
    num_kv_splits = torch.full(
        (queries,), splits + 1, dtype=torch.int32, device=device
    )
    prefix_len_t = torch.tensor(prefix_len, dtype=torch.int32, device=device)
    prefix_splits_t = torch.tensor(splits, dtype=torch.int32, device=device)
    tree_cols_t = torch.tensor(cols, dtype=torch.int32, device=device)
    if prefix_rows is None:
        rows = torch.ones(queries, dtype=torch.int32, device=device)
    else:
        rows = prefix_rows.to(device=device, dtype=torch.int32)
    mask = tree_mask.to(device=device, dtype=torch.uint8)
    o = torch.empty_like(q)
    kv_heads = int(k_cache.shape[1])
    _launch(
        q,
        o,
        k_cache,
        v_cache,
        prefix_indices.to(dtype=torch.int64),
        tree_indices.to(dtype=torch.int64),
        mask,
        rows,
        prefix_len_t,
        prefix_splits_t,
        tree_cols_t,
        logits,
        lse,
        kv_indptr,
        num_kv_splits,
        float(sm_scale),
        kv_heads,
        None,
    )
    return o


def _launch(
    q,
    o,
    k_buffer,
    v_buffer,
    prefix_indices,
    tree_indices,
    tree_mask,
    prefix_rows,
    prefix_len,
    prefix_splits,
    tree_cols,
    logits,
    lse,
    kv_indptr,
    num_kv_splits,
    sm_scale,
    kv_heads,
    sinks,
):
    import triton

    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        _MIN_BLOCK_KV,
        _decode_softmax_reducev_fwd,
    )

    if int(_MIN_BLOCK_KV) != STAGE2_MIN_BLOCK_KV:
        raise RuntimeError(
            f"decode stage2 block {_MIN_BLOCK_KV} != {STAGE2_MIN_BLOCK_KV}"
        )
    queries, q_heads, dim = q.shape
    if int(kv_heads) < 1 or int(q_heads) % int(kv_heads) != 0:
        raise RuntimeError(f"bad GQA shape heads={q_heads} kv_heads={kv_heads}")
    group = int(q_heads) // int(kv_heads)
    if _stage1 is None:
        raise RuntimeError("triton is required for OPT cascade attention")
    block_d = triton.next_power_of_2(int(dim))
    kernel = _stage1
    rows = int(queries) * group
    block_q = cascade_query_block(rows)
    grid = (
        int(kv_heads),
        MAX_SPLITS,
        cdiv(rows, block_q),
    )
    kernel[grid](
        q,
        k_buffer,
        v_buffer,
        prefix_indices,
        tree_indices,
        tree_mask,
        prefix_rows,
        prefix_len,
        prefix_splits,
        tree_cols,
        logits,
        lse,
        sm_scale,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        tree_mask.stride(0),
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        lse.stride(0),
        lse.stride(1),
        int(queries),
        GROUP=group,
        BLOCK_Q=block_q,
        BLOCK_N=BLOCK_N,
        BLOCK_D=block_d,
        DIM=int(dim),
        MAX_SPLITS=MAX_SPLITS,
        num_warps=4,
        num_stages=1,
    )
    _decode_softmax_reducev_fwd(
        logits,
        lse,
        q,
        o,
        1.0,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        MAX_SPLITS,
        sinks,
    )


try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

if triton is not None:

    @triton.jit
    def _stage1(
        Q,
        K,
        V,
        PrefixIdx,
        TreeIdx,
        TreeMask,
        PrefixRows,
        PrefixLen,
        PrefixSplits,
        TreeCols,
        Out,
        Lse,
        sm_scale,
        stride_q_n,
        stride_q_h,
        stride_k_t,
        stride_k_h,
        stride_v_t,
        stride_v_h,
        stride_mask_n,
        stride_o_n,
        stride_o_h,
        stride_o_s,
        stride_lse_n,
        stride_lse_h,
        n_queries,
        GROUP: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        DIM: tl.constexpr,
        MAX_SPLITS: tl.constexpr,
    ):
        kv_head = tl.program_id(0)
        split = tl.program_id(1)
        q_tile = tl.program_id(2)
        prefix_len = tl.load(PrefixLen)
        prefix_splits = tl.load(PrefixSplits)
        tree_cols = tl.load(TreeCols)

        rows = tl.arange(0, BLOCK_Q)
        linear = q_tile * BLOCK_Q + rows
        row_ok = linear < n_queries * GROUP
        node = tl.where(row_ok, linear // GROUP, 0)
        local_h = tl.where(row_ok, linear % GROUP, 0)
        q_head = kv_head * GROUP + local_h
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < DIM

        q = tl.load(
            Q + node[:, None] * stride_q_n + q_head[:, None] * stride_q_h + offs_d[None, :],
            mask=row_ok[:, None] & mask_d[None, :],
            other=0.0,
        )
        attend = tl.load(PrefixRows + node, mask=row_ok, other=0)

        acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
        e_max = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

        do_prefix = split < prefix_splits
        do_tree = split == prefix_splits
        safe_splits = tl.maximum(prefix_splits, 1)
        base = prefix_len // safe_splits
        rem = prefix_len % safe_splits
        length = base + tl.where(split < rem, 1, 0)
        start = split * base + tl.minimum(split, rem)
        prefix_span = tl.where(do_prefix, length, 0)
        tree_span = tl.where(do_tree, tree_cols, 0)

        for off in range(0, prefix_span, BLOCK_N):
            off = tl.multiple_of(off, BLOCK_N)
            offs_n = off + tl.arange(0, BLOCK_N)
            in_bounds = offs_n < prefix_span
            slots = tl.load(PrefixIdx + start + offs_n, mask=in_bounds, other=0).to(
                tl.int64
            )
            k = tl.load(
                K
                + slots[None, :] * stride_k_t
                + kv_head * stride_k_h
                + offs_d[:, None],
                mask=in_bounds[None, :] & mask_d[:, None],
                other=0.0,
            )
            qk = tl.dot(q.to(k.dtype), k) * sm_scale
            qk = tl.where(in_bounds[None, :], qk, float("-inf"))
            qk = tl.where(attend[:, None] != 0, qk, float("-inf"))
            row_max = tl.max(qk, 1)
            row_max = tl.where(row_max == float("-inf"), -1.0e20, row_max)
            n_e_max = tl.maximum(row_max, e_max)
            rescale = tl.exp(e_max - n_e_max)
            prob = tl.exp(qk - n_e_max[:, None])
            v = tl.load(
                V
                + slots[:, None] * stride_v_t
                + kv_head * stride_v_h
                + offs_d[None, :],
                mask=in_bounds[:, None] & mask_d[None, :],
                other=0.0,
            )
            acc = acc * rescale[:, None] + tl.dot(prob.to(v.dtype), v)
            e_sum = e_sum * rescale + tl.sum(prob, 1)
            e_max = n_e_max

        for off in range(0, tree_span, BLOCK_N):
            off = tl.multiple_of(off, BLOCK_N)
            offs_n = off + tl.arange(0, BLOCK_N)
            in_bounds = offs_n < tree_span
            slots = tl.load(TreeIdx + offs_n, mask=in_bounds, other=0).to(tl.int64)
            k = tl.load(
                K
                + slots[None, :] * stride_k_t
                + kv_head * stride_k_h
                + offs_d[:, None],
                mask=in_bounds[None, :] & mask_d[:, None],
                other=0.0,
            )
            visible = tl.load(
                TreeMask + node[:, None] * stride_mask_n + offs_n[None, :],
                mask=row_ok[:, None] & in_bounds[None, :],
                other=0,
            )
            qk = tl.dot(q.to(k.dtype), k) * sm_scale
            qk = tl.where((visible != 0) & in_bounds[None, :], qk, float("-inf"))
            row_max = tl.max(qk, 1)
            row_max = tl.where(row_max == float("-inf"), -1.0e20, row_max)
            n_e_max = tl.maximum(row_max, e_max)
            rescale = tl.exp(e_max - n_e_max)
            prob = tl.exp(qk - n_e_max[:, None])
            v = tl.load(
                V
                + slots[:, None] * stride_v_t
                + kv_head * stride_v_h
                + offs_d[None, :],
                mask=in_bounds[:, None] & mask_d[None, :],
                other=0.0,
            )
            acc = acc * rescale[:, None] + tl.dot(prob.to(v.dtype), v)
            e_sum = e_sum * rescale + tl.sum(prob, 1)
            e_max = n_e_max

        keep = do_prefix | do_tree
        safe = (e_sum > 0) & keep
        denom = tl.where(safe, e_sum, 1.0)
        stored = tl.where(safe[:, None], acc / denom[:, None], 0.0)
        lse = tl.where(safe, e_max + tl.log(denom), -1.0e20)
        tl.store(
            Out
            + node[:, None] * stride_o_n
            + q_head[:, None] * stride_o_h
            + split * stride_o_s
            + offs_d[None, :],
            stored,
            mask=row_ok[:, None] & mask_d[None, :] & keep,
        )
        tl.store(
            Lse + node * stride_lse_n + q_head * stride_lse_h + split,
            lse,
            mask=row_ok & keep,
        )

else:
    _stage1 = None


try:
    launch_workspace = torch.compiler.disable(launch_workspace)
except Exception:
    pass
