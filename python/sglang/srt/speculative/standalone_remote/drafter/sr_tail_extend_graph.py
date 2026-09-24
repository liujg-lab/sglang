"""Dedicated EXTEND graph helpers and runner for STANDALONE_REMOTE tails.

Keeps ordinary EXTEND + ``is_sr_tail_extend`` attention. Padding queries write
reserved KV slot 0 and are packed after real tokens so they cannot enter a
real prefix or a later real query's visible context. Seed is still taken from
each request's last real token after replay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from sglang.srt.speculative.spec_utils import (
    NpuGraphPreparationError,
    NpuGraphReplaySubmittedError,
    device_backend_key,
)
from sglang.srt.speculative.standalone_remote.sr_align import is_device_context_error
from sglang.srt.speculative.standalone_remote.sr_tail_attention import (
    SRTailAttentionMetadata,
    build_tail_attention_metadata,
    copy_tail_attention_metadata_,
    fill_tail_attention_metadata_,
    pad_tail_attention_metadata,
    tail_graph_fits_pages,
    tail_graph_max_pages,
    widen_tail_block_tables,
)

logger = logging.getLogger(__name__)

TAIL_DUMMY_SLOT = 0
TailGraphBucket = Tuple[int, int, Optional[int]]


def default_tail_token_caps(speculative_num_steps: int) -> Tuple[int, ...]:
    caps = {1, 2, 4}
    steps = max(int(speculative_num_steps or 0), 0)
    if steps:
        caps.add(steps)
        caps.add(steps + 1)
    return tuple(sorted(c for c in caps if c > 0))


def tail_max_per_request(speculative_num_steps: int) -> int:
    return max(int(speculative_num_steps or 0), 0) + 1


def cuda_tail_capture_extend_lens(
    bs_cap: int, token_cap: int, max_per_req: int
) -> Optional[List[int]]:
    """Extend layout whose longest row covers any real tail in the bucket.

    Triton freezes ``max_extend_len`` into the captured kernel grid. The first
    row takes every token that is not required to keep the other rows nonempty,
    which is at least ``min(token_cap, max_per_req)`` whenever that fits.
    """
    bs_cap = int(bs_cap)
    token_cap = int(token_cap)
    if bs_cap <= 0 or token_cap < bs_cap or int(max_per_req) < 0:
        return None
    if bs_cap == 1:
        return [token_cap]
    first = token_cap - (bs_cap - 1)
    rest = token_cap - first
    base, extra = divmod(rest, bs_cap - 1)
    return [first] + [base + (1 if i < extra else 0) for i in range(bs_cap - 1)]


def tail_kv_index_capacity(bs_cap: int, captured_pages: int, page_size: int) -> int:
    """Prefix-token slots a captured tail graph must be able to index."""
    return max(int(bs_cap), 0) * max(int(captured_pages), 1) * max(int(page_size), 1)


def widen_tail_kv_indices(metadata, capacity: int, device) -> torch.Tensor:
    """Point ``metadata.kv_indices`` at a buffer of at least ``capacity`` slots.

    Call this before CUDA graph capture. Replacing the tensor afterwards would
    leave the graph pointing at the old storage.
    """
    capacity = max(int(capacity), 0)
    current = getattr(metadata, "kv_indices", None)
    target = torch.device(device)
    if (
        torch.is_tensor(current)
        and int(current.numel()) >= capacity
        and current.device == target
    ):
        return current
    buf = torch.zeros(capacity, dtype=torch.int64, device=target)
    if torch.is_tensor(current) and int(current.numel()) > 0 and capacity > 0:
        n = min(int(current.numel()), capacity)
        buf[:n].copy_(current[:n].to(device=buf.device, dtype=buf.dtype))
    metadata.kv_indices = buf
    return buf


def build_cuda_tail_attention_metadata(
    prefix_lens,
    extend_lens,
    block_tables: torch.Tensor,
    token_cap: int,
    captured_pages: int,
) -> SRTailAttentionMetadata:
    """SR tail buffers for a backend that does not publish ``forward_metadata.sr_tail``."""
    metadata = build_tail_attention_metadata(prefix_lens, extend_lens, block_tables)
    padded = pad_tail_attention_metadata(
        metadata, int(token_cap), dummy_slot=TAIL_DUMMY_SLOT
    )
    return SRTailAttentionMetadata(
        widen_tail_block_tables(padded.block_tables, max(int(captured_pages), 1)),
        padded.context_lens_cpu,
        list(padded.context_lens_list),
    )


def _write_indptr_(indptr: torch.Tensor, lengths: torch.Tensor) -> None:
    n = int(lengths.shape[0])
    if int(indptr.numel()) < n + 1:
        raise NpuGraphPreparationError(
            "captured SR tail indptr is shorter than the bucket",
            scope="graph",
        )
    values = lengths.to(dtype=indptr.dtype, device=indptr.device)
    indptr.zero_()
    if n:
        indptr[1 : n + 1].copy_(torch.cumsum(values, dim=0))


def refresh_tail_triton_metadata(
    metadata,
    prefix_lens: Sequence[int],
    extend_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
) -> None:
    """Update the Triton extend buffers captured for one tail bucket, in place."""
    kv_indices = getattr(metadata, "kv_indices", None)
    kv_indptr = getattr(metadata, "kv_indptr", None)
    qo_indptr = getattr(metadata, "qo_indptr", None)
    if not (
        torch.is_tensor(kv_indices)
        and torch.is_tensor(kv_indptr)
        and torch.is_tensor(qo_indptr)
    ):
        raise NpuGraphPreparationError(
            "captured SR tail triton metadata is incomplete",
            scope="graph",
        )
    prefix = [int(v) for v in prefix_lens]
    if any(v < 0 for v in prefix):
        raise NpuGraphPreparationError(
            "SR tail prefix length is negative",
            scope="graph",
        )
    if int(extend_lens.shape[0]) != len(prefix):
        raise NpuGraphPreparationError(
            "SR tail extend rows do not match the bucket",
            scope="graph",
        )
    prefix_tokens = sum(prefix)
    if prefix_tokens > int(kv_indices.numel()):
        raise NpuGraphPreparationError(
            "SR tail prefix exceeds captured kv_indices",
            scope="graph",
        )
    prefix_t = torch.tensor(prefix, dtype=kv_indptr.dtype, device=kv_indptr.device)
    _write_indptr_(kv_indptr, prefix_t)
    _write_indptr_(qo_indptr, extend_lens)
    if prefix_tokens <= 0:
        return
    from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton

    create_flashinfer_kv_indices_triton[(len(prefix),)](
        req_to_token,
        req_pool_indices,
        prefix_t,
        kv_indptr,
        None,
        kv_indices,
        req_to_token.stride(0),
    )


def trim_capture_batch_sizes(capture_bs: Sequence[int], limit: int = 6) -> List[int]:
    values = [int(bs) for bs in capture_bs if int(bs) > 0]
    values = sorted(set(values))
    if len(values) <= limit:
        return values
    return values[: limit - 1] + [values[-1]]


def tail_token_caps_for_bs(
    bs: int,
    max_per_req: int,
    *,
    dummy: bool = False,
    max_bs: int = 1,
) -> Tuple[int, ...]:
    """Packed batch token caps. Dummy bs covers inexact concurrent sums."""
    max_per_req = max(int(max_per_req), 1)
    bs = int(bs)
    if dummy:
        packed = max(int(max_bs), 1) * max_per_req
        return tuple(sorted(c for c in {4, 6, 8, packed} if c > 0))
    if bs <= 1:
        return tuple(sorted(c for c in {1, 2, 4, max_per_req} if c > 0))
    return tuple(range(bs, bs * max_per_req + 1))


def default_tail_graph_buckets(
    capture_bs: Sequence[int],
    token_caps: Sequence[int],
    attn_caps: Sequence[Optional[int]] = (None,),
) -> List[TailGraphBucket]:
    buckets: List[TailGraphBucket] = []
    for bs in trim_capture_batch_sizes(capture_bs):
        for token_cap in sorted(set(int(c) for c in token_caps if int(c) > 0)):
            for attn_cap in attn_caps:
                cap = None if attn_cap is None else int(attn_cap)
                buckets.append((int(bs), int(token_cap), cap))
    return buckets


def packed_tail_graph_buckets(
    capture_bs: Sequence[int],
    speculative_num_steps: int,
    attn_caps: Sequence[Optional[int]] = (None,),
) -> List[TailGraphBucket]:
    """Per-bs packed caps plus dummy_bs = max(capture_bs) + 1."""
    sizes = trim_capture_batch_sizes(capture_bs)
    if not sizes:
        return []
    max_bs = max(sizes)
    max_per_req = tail_max_per_request(speculative_num_steps)
    buckets: List[TailGraphBucket] = []
    seen = set()

    def add(bs: int, token_cap: int) -> None:
        for attn_cap in attn_caps:
            cap = None if attn_cap is None else int(attn_cap)
            key = (int(bs), int(token_cap), cap)
            if key in seen:
                continue
            seen.add(key)
            buckets.append(key)

    for bs in sizes:
        for token_cap in tail_token_caps_for_bs(bs, max_per_req, max_bs=max_bs):
            add(bs, token_cap)
    dummy_bs = max_bs + 1
    for token_cap in tail_token_caps_for_bs(
        dummy_bs, max_per_req, dummy=True, max_bs=max_bs
    ):
        add(dummy_bs, token_cap)
    return buckets


def select_tail_graph_bucket(
    bs: int,
    n_tokens: int,
    max_seq: int,
    buckets: Sequence[TailGraphBucket],
) -> Optional[TailGraphBucket]:
    """Pick the smallest fitting bucket.

    Token padding needs a dummy request, so ``bs_cap`` must be at least
    ``bs + 1`` when ``n_tokens < token_cap``. Exact token fills require
    ``bs_cap == bs`` so dummy requests would not have an empty tail.
    """
    best = None
    for bs_cap, token_cap, attn_cap in buckets:
        if n_tokens > token_cap or bs > bs_cap:
            continue
        if attn_cap is not None and max_seq > attn_cap:
            continue
        if n_tokens < token_cap and bs_cap < bs + 1:
            continue
        if n_tokens == token_cap and bs_cap != bs:
            continue
        key = (bs_cap, token_cap, attn_cap if attn_cap is not None else 0)
        if best is None or key < (best[0], best[1], best[2] if best[2] is not None else 0):
            best = (bs_cap, token_cap, attn_cap)
    return best


def build_tail_graph_plan(
    batch,
    graphs,
    buckets: Sequence[TailGraphBucket],
    captured_pages: int,
    page_size: int,
) -> Tuple[Optional[TailGraphPlan], str]:
    """Classify a tail batch against captured graphs. Reasons: ok / no_graphs / pages / no_bucket."""
    if not graphs:
        return None, "no_graphs"
    extend_lens = list(getattr(batch, "extend_lens", None) or [])
    prefix_lens = list(getattr(batch, "prefix_lens", None) or [])
    if not extend_lens:
        extend_lens = list(getattr(batch, "extend_seq_lens_cpu", None) or [])
        prefix_lens = list(getattr(batch, "extend_prefix_lens_cpu", None) or [])
    n_tokens = int(getattr(batch, "extend_num_tokens", 0) or sum(extend_lens))
    raw_bs = len(extend_lens)
    if raw_bs <= 0 or n_tokens <= 0:
        return None, "no_bucket"
    max_seq = max((p + n for p, n in zip(prefix_lens, extend_lens)), default=0)
    if not tail_graph_fits_pages(max_seq, captured_pages, page_size):
        return None, "pages"
    bucket = select_tail_graph_bucket(raw_bs, n_tokens, max_seq, buckets)
    if bucket is None or bucket not in graphs:
        return None, "no_bucket"
    dummy_tokens = bucket[1] - n_tokens
    return (
        TailGraphPlan(
            bucket=bucket,
            raw_bs=raw_bs,
            raw_tokens=n_tokens,
            dummy_tokens=dummy_tokens,
            prefix_lens=[int(v) for v in prefix_lens],
            extend_lens=[int(v) for v in extend_lens],
            max_seq=int(max_seq),
        ),
        "ok",
    )


def real_seed_rows(extend_lens: Sequence[int]) -> List[int]:
    rows = []
    offset = 0
    for length in extend_lens:
        if int(length) <= 0:
            raise ValueError("SR tail seed requires a nonempty real tail")
        offset += int(length)
        rows.append(offset - 1)
    return rows


@dataclass
class TailGraphPlan:
    bucket: TailGraphBucket
    raw_bs: int
    raw_tokens: int
    dummy_tokens: int
    prefix_lens: List[int]
    extend_lens: List[int]
    max_seq: int


@dataclass
class _TailGraphBuffers:
    input_ids: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_lens_cpu: torch.Tensor
    req_page_tables: torch.Tensor
    next_token_logits: torch.Tensor
    hidden_states: Optional[torch.Tensor]
    mrope_positions: torch.Tensor


def tail_graph_attn_caps(backend) -> Sequence[Optional[int]]:
    """FIA graphs may specialize on tree KV buckets; ATB updates lengths at replay."""
    if backend is not None and getattr(backend, "use_fia", False):
        buckets = getattr(backend, "tree_kv_buckets", None)
        if buckets:
            return list(reversed(list(buckets)))
    return (None,)


def tail_graph_kv_tokens(runner) -> int:
    """KV-pool capacity for tail graph page tables, not model context_len."""
    pool = getattr(runner, "token_to_kv_pool", None)
    pool_tokens = int(getattr(pool, "size", 0) or 0) if pool is not None else 0
    total = int(getattr(runner, "max_total_num_tokens", 0) or 0)
    return max(pool_tokens, total, 1)


def tail_graph_page_kv_tokens(runner) -> int:
    """Cap captured page-table width to the tree KV bucket, default 1024 tokens."""
    kv = tail_graph_kv_tokens(runner)
    backend = getattr(runner, "attn_backend", None)
    buckets = getattr(backend, "tree_kv_buckets", None) if backend is not None else None
    tree_max = max(int(v) for v in buckets) if buckets else 1024
    return min(max(int(kv), 1), max(int(tree_max), 1))


def make_tail_graph_buffers(
    *,
    bs_cap: int,
    token_cap: int,
    device,
    vocab: int,
    hidden: int,
    dtype,
    loc_dtype,
    pages: int = 1,
) -> _TailGraphBuffers:
    """Allocate capture buffers.

    ``seq_lens_cpu`` stays on CPU. ``next_token_logits`` is float32 because
    LogitsProcessor copies into ``next_token_logits_buffer`` only when
    ``dtype == torch.float``. ``hidden`` / ``dtype`` stay for callers; SR
    tail graphs do not allocate unused hidden buffers.
    """
    del hidden, dtype
    return _TailGraphBuffers(
        input_ids=torch.zeros((token_cap,), dtype=torch.int64, device=device),
        positions=torch.zeros((token_cap,), dtype=torch.int64, device=device),
        out_cache_loc=torch.zeros((token_cap,), dtype=loc_dtype, device=device),
        req_pool_indices=torch.zeros((bs_cap,), dtype=torch.int64, device=device),
        seq_lens=torch.ones((bs_cap,), dtype=torch.int64, device=device),
        seq_lens_cpu=torch.ones((bs_cap,), dtype=torch.int64, device="cpu"),
        extend_seq_lens=torch.ones((bs_cap,), dtype=torch.int64, device=device),
        extend_lens_cpu=torch.zeros((bs_cap,), dtype=torch.int64, device="cpu"),
        req_page_tables=torch.zeros(
            (bs_cap, max(int(pages), 1)), dtype=torch.int32, device=device
        ),
        next_token_logits=torch.zeros((bs_cap, vocab), dtype=torch.float, device=device),
        hidden_states=None,
        mrope_positions=torch.zeros((3, token_cap), dtype=torch.int64, device=device),
    )


class SRTailExtendGraphRunner:
    """Capture/replay ordinary EXTEND for packed SR tails."""

    def __init__(self, drafter) -> None:
        self.drafter = drafter
        self.model_runner = drafter.draft_model_runner
        self.server_args = drafter.server_args
        self.device = drafter.device
        self.graphs = {}
        self.buffers = {}
        self.output_buffers = {}
        self.attn_metadata = {}
        self.buckets: List[TailGraphBucket] = []
        self.dummy_req_idx = None
        self.disabled_reason = None
        self.replay_count = 0
        self.eager_fallback_count = 0
        self.page_size = 1
        self.captured_pages = 0
        self.triton_metadata = {}
        self._capture()

    def _create_graph(self):
        return torch.cuda.CUDAGraph()

    def _capture_context(self, graph, pool, stream):
        return torch.cuda.graph(graph, pool=pool, stream=stream)

    def _device_synchronize(self):
        torch.cuda.synchronize()

    def _replay_graph(self, graph, seq_lens_kv, bucket=None):
        graph.replay()

    def plan_with_reason(self, batch) -> Tuple[Optional[TailGraphPlan], str]:
        return build_tail_graph_plan(
            batch, self.graphs, self.buckets, self.captured_pages, self.page_size
        )

    def plan(self, batch) -> Optional[TailGraphPlan]:
        planned, _reason = self.plan_with_reason(batch)
        return planned

    def can_run(self, batch) -> bool:
        return self.plan(batch) is not None

    def fill(self, forward_batch, plan: TailGraphPlan) -> None:
        buffers = self.buffers[plan.bucket]
        bs_cap, token_cap, _ = plan.bucket
        raw_bs = plan.raw_bs
        raw_tokens = plan.raw_tokens
        buffers.input_ids.zero_()
        buffers.positions.zero_()
        buffers.out_cache_loc.fill_(TAIL_DUMMY_SLOT)
        buffers.seq_lens.fill_(1)
        buffers.seq_lens_cpu.fill_(1)
        buffers.extend_seq_lens.zero_()
        buffers.req_pool_indices.fill_(int(self.dummy_req_idx or 0))
        buffers.input_ids[:raw_tokens].copy_(forward_batch.input_ids[:raw_tokens])
        buffers.positions[:raw_tokens].copy_(forward_batch.positions[:raw_tokens])
        buffers.out_cache_loc[:raw_tokens].copy_(
            forward_batch.out_cache_loc[:raw_tokens]
        )
        buffers.req_pool_indices[:raw_bs].copy_(
            forward_batch.req_pool_indices[:raw_bs]
        )
        buffers.seq_lens[:raw_bs].copy_(forward_batch.seq_lens[:raw_bs])
        if forward_batch.seq_lens_cpu is not None:
            buffers.seq_lens_cpu[:raw_bs].copy_(
                forward_batch.seq_lens_cpu[:raw_bs].to(device="cpu")
            )
        cpu_extend = buffers.extend_lens_cpu
        cpu_extend.zero_()
        for i, length in enumerate(plan.extend_lens):
            cpu_extend[i] = int(length)
        if plan.dummy_tokens:
            buffers.req_pool_indices[raw_bs:bs_cap] = int(self.dummy_req_idx or 0)
            cpu_extend[raw_bs] = int(plan.dummy_tokens)
            buffers.seq_lens[raw_bs] = plan.dummy_tokens
            buffers.seq_lens_cpu[raw_bs] = plan.dummy_tokens
        buffers.extend_seq_lens.copy_(cpu_extend)
        self._fill_mrope_positions(buffers, forward_batch, raw_tokens)
        self._fill_attention_metadata(forward_batch, plan, buffers)
        self._refresh_captured_triton_metadata(forward_batch, plan, buffers)

    def replay_filled(self, plan: TailGraphPlan):
        graph = self.graphs[plan.bucket]
        buffers = self.buffers[plan.bucket]
        kv_lens = getattr(self, "_padded_context_lens", None) or buffers.seq_lens_cpu[
            : plan.bucket[0]
        ].tolist()
        self._replay_graph(graph, kv_lens, plan.bucket)
        self.replay_count += 1
        output = self.output_buffers[plan.bucket]
        raw_bs = plan.raw_bs
        hidden = output.hidden_states
        return type(output)(
            next_token_logits=output.next_token_logits[:raw_bs],
            hidden_states=None if hidden is None else hidden[:raw_bs],
        )

    def replay(self, forward_batch, plan: Optional[TailGraphPlan] = None):
        if plan is None:
            plan = self.plan(forward_batch)
        if plan is None:
            raise NpuGraphPreparationError("no SR tail graph bucket", scope="graph")
        self.fill(forward_batch, plan)
        return self.replay_filled(plan)

    def init_forward_batch(self, worker_batch):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        return ForwardBatch.init_new(worker_batch, self.model_runner)

    def _fill_mrope_positions(self, buffers, forward_batch, raw_tokens: int) -> None:
        buffers.mrope_positions.zero_()
        src = getattr(forward_batch, "mrope_positions", None)
        if src is not None and src.ndim == 2 and src.shape[0] == 3:
            n = min(raw_tokens, int(src.shape[1]), int(buffers.mrope_positions.shape[1]))
            buffers.mrope_positions[:, :n].copy_(src[:, :n])
            return
        n = min(raw_tokens, int(buffers.positions.shape[0]))
        if n <= 0:
            return
        buffers.mrope_positions[:, :n].copy_(
            buffers.positions[:n].unsqueeze(0).expand(3, -1)
        )

    def _fill_attention_metadata(self, forward_batch, plan, buffers) -> None:
        backend = getattr(self.model_runner, "attn_backend", None)
        token_cap = plan.bucket[1]
        if backend is None or not hasattr(backend, "forward_metadata"):
            self._padded_context_lens = [1] * token_cap
            return
        prefix = list(plan.prefix_lens)
        extend = list(plan.extend_lens)
        seq_max = max(plan.max_seq, 1)
        page_size = int(getattr(backend, "page_size", self.page_size) or 1)
        tables = buffers.req_page_tables
        tables.zero_()
        src = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices[: plan.raw_bs], :seq_max:page_size
        ]
        n_pages = min(int(src.shape[1]), int(tables.shape[1]))
        if n_pages > 0:
            gathered = src[:, :n_pages]
            if page_size != 1:
                gathered = gathered // page_size
            tables[: plan.raw_bs, :n_pages].copy_(gathered.to(dtype=torch.int32))
        captured = self.attn_metadata.get(plan.bucket)
        if captured is None:
            raise NpuGraphPreparationError(
                "missing captured SR tail attention buffers", scope="graph"
            )
        try:
            fill_tail_attention_metadata_(
                captured, prefix, extend, tables[: plan.raw_bs]
            )
        except ValueError as e:
            raise NpuGraphPreparationError(str(e), scope="graph") from e
        if getattr(backend, "forward_metadata", None) is None:
            backend.forward_metadata = type("ForwardMetadata", (), {})()
        backend.forward_metadata.sr_tail = captured
        self._padded_context_lens = captured.context_lens_list

    def _refresh_captured_triton_metadata(self, forward_batch, plan, buffers) -> None:
        meta = getattr(self, "triton_metadata", {}).get(plan.bucket)
        if meta is None:
            return
        bs_cap = int(plan.bucket[0])
        prefix = [int(v) for v in plan.prefix_lens]
        if len(prefix) < bs_cap:
            prefix.extend([0] * (bs_cap - len(prefix)))
        else:
            prefix = prefix[:bs_cap]
        refresh_tail_triton_metadata(
            meta,
            prefix,
            buffers.extend_seq_lens[:bs_cap],
            forward_batch.req_to_token_pool.req_to_token,
            buffers.req_pool_indices[:bs_cap],
        )

    def _capture(self) -> None:
        if getattr(self.server_args, "disable_cuda_graph", False):
            self.disabled_reason = "disabled by configuration"
            return
        try:
            dummy = self._reserve_dummy_req()
            self.dummy_req_idx = dummy
            backend = getattr(self.model_runner, "attn_backend", None)
            self.page_size = int(getattr(backend, "page_size", 1) or 1)
            self.captured_pages = tail_graph_max_pages(
                tail_graph_page_kv_tokens(self.model_runner), self.page_size
            )
            self.buckets = self._build_buckets()
            if not self.buckets:
                self.disabled_reason = "no tail graph buckets"
                return
            for bucket in list(self.buckets):
                self._capture_bucket(bucket)
            if not self.graphs:
                self.disabled_reason = self.disabled_reason or "no graphs"
                self.buckets = []
                self.attn_metadata = {}
                self.triton_metadata = {}
        except NpuGraphReplaySubmittedError:
            raise
        except Exception as e:
            if is_device_context_error(e):
                raise
            logger.warning("[SR] tail EXTEND graph capture failed: %s", e)
            self.disabled_reason = "capture failed"
            self.graphs = {}
            self.buffers = {}
            self.output_buffers = {}
            self.attn_metadata = {}
            self.triton_metadata = {}
            self.buckets = []

    def _build_buckets(self) -> List[TailGraphBucket]:
        try:
            from sglang.srt.model_executor.cuda_graph_runner import (
                get_batch_sizes_to_capture,
            )

            capture_bs, _ = get_batch_sizes_to_capture(self.model_runner)
        except Exception:
            capture_bs = [1, 2, 4]
        backend = getattr(self.model_runner, "attn_backend", None)
        return packed_tail_graph_buckets(
            capture_bs,
            int(getattr(self.server_args, "speculative_num_steps", 0) or 0),
            tail_graph_attn_caps(backend),
        )

    def _reserve_dummy_req(self) -> Optional[int]:
        pool = self.model_runner.req_to_token_pool
        size = int(getattr(pool, "size", 0) or pool.req_to_token.shape[0])
        if size <= 1:
            return None
        return 0

    def _bind_capture_sr_tail(self, backend, token_cap: int):
        sr_tail = getattr(getattr(backend, "forward_metadata", None), "sr_tail", None)
        if sr_tail is None:
            return None
        padded = pad_tail_attention_metadata(
            sr_tail, token_cap, dummy_slot=TAIL_DUMMY_SLOT
        )
        captured = SRTailAttentionMetadata(
            widen_tail_block_tables(padded.block_tables, self.captured_pages),
            padded.context_lens_cpu,
            list(padded.context_lens_list),
        )
        backend.forward_metadata.sr_tail = captured
        return captured

    def _capture_bucket(self, bucket: TailGraphBucket) -> None:
        bs_cap, token_cap, _ = bucket
        runner = self.model_runner
        device = runner.device
        vocab = int(runner.model_config.vocab_size)
        hidden = int(runner.model_config.hidden_size)
        dtype = runner.model_config.dtype
        loc_dtype = torch.int64 if device_backend_key(device) != "npu" else torch.int32
        buffers = make_tail_graph_buffers(
            bs_cap=bs_cap,
            token_cap=token_cap,
            device=device,
            vocab=vocab,
            hidden=hidden,
            dtype=dtype,
            loc_dtype=loc_dtype,
            pages=self.captured_pages,
        )
        dummy = int(self.dummy_req_idx or 0)
        buffers.req_pool_indices.fill_(dummy)
        buffers.out_cache_loc.fill_(TAIL_DUMMY_SLOT)
        is_npu = device_backend_key(device) == "npu"
        if is_npu:
            per = max(token_cap // bs_cap, 1)
            leftover = token_cap - per * (bs_cap - 1)
            extend = [per] * (bs_cap - 1) + [leftover]
            if leftover <= 0:
                return
        else:
            extend = cuda_tail_capture_extend_lens(
                bs_cap,
                token_cap,
                tail_max_per_request(
                    int(getattr(self.server_args, "speculative_num_steps", 0) or 0)
                ),
            )
            if not extend:
                return
        buffers.extend_seq_lens.copy_(
            torch.tensor(extend, dtype=torch.int64, device=buffers.extend_seq_lens.device)
        )
        buffers.seq_lens.copy_(buffers.extend_seq_lens)
        buffers.seq_lens_cpu.copy_(
            torch.tensor(extend, dtype=torch.int64, device="cpu")
        )
        buffers.mrope_positions.copy_(buffers.positions.unsqueeze(0).expand(3, -1))

        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=bs_cap,
            input_ids=buffers.input_ids,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_cpu=buffers.seq_lens_cpu,
            seq_lens_sum=int(sum(extend)),
            out_cache_loc=buffers.out_cache_loc,
            positions=buffers.positions,
            mrope_positions=buffers.mrope_positions,
            extend_seq_lens=buffers.extend_seq_lens,
            extend_seq_lens_cpu=extend,
            extend_prefix_lens=torch.zeros(
                (bs_cap,), dtype=buffers.seq_lens.dtype, device=buffers.seq_lens.device
            ),
            extend_prefix_lens_cpu=[0] * bs_cap,
            extend_num_tokens=token_cap,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool=runner.token_to_kv_pool,
            spec_algorithm=runner.spec_algorithm,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            attn_backend=runner.attn_backend,
            next_token_logits_buffer=buffers.next_token_logits,
        )
        forward_batch.is_sr_tail_extend = True
        backend = runner.attn_backend
        if backend is not None and hasattr(backend, "init_forward_metadata"):
            backend.init_forward_metadata(forward_batch)
            if not is_npu:
                meta = getattr(backend, "forward_metadata", None)
                if meta is not None and hasattr(meta, "kv_indices"):
                    widen_tail_kv_indices(
                        meta,
                        tail_kv_index_capacity(
                            bs_cap, self.captured_pages, self.page_size
                        ),
                        device,
                    )
            captured = self._bind_capture_sr_tail(backend, token_cap)
        else:
            captured = None
        if captured is None and not is_npu:
            captured = build_cuda_tail_attention_metadata(
                [0] * bs_cap,
                extend,
                buffers.req_page_tables,
                token_cap,
                self.captured_pages,
            )
            meta = (
                getattr(backend, "forward_metadata", None)
                if backend is not None
                else None
            )
            if meta is not None:
                meta.sr_tail = captured

        def run_once():
            out = runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )
            if getattr(out, "next_token_logits", None) is not None:
                buffers.next_token_logits.copy_(out.next_token_logits[:bs_cap])
            return out

        try:
            self._device_synchronize()
            run_once()
            self._device_synchronize()
            run_once()
            graph = self._create_graph()
            stream = getattr(runner, "stream", None)
            pool = None
            try:
                from sglang.srt.model_executor.cuda_graph_runner import (
                    get_global_graph_memory_pool,
                    set_global_graph_memory_pool,
                )

                pool = get_global_graph_memory_pool()
            except Exception:
                pool = None
            with self._capture_context(graph, pool, stream):
                out = run_once()
            try:
                from sglang.srt.model_executor.cuda_graph_runner import (
                    set_global_graph_memory_pool,
                )

                if hasattr(graph, "pool"):
                    set_global_graph_memory_pool(graph.pool())
            except Exception:
                pass
        except Exception as e:
            if is_device_context_error(e) or isinstance(e, NpuGraphReplaySubmittedError):
                raise
            logger.warning(
                "[SR] skip tail graph bucket %s: %s: %s",
                bucket,
                type(e).__name__,
                e,
            )
            return
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        self.graphs[bucket] = graph
        self.buffers[bucket] = buffers
        if captured is not None:
            self.attn_metadata[bucket] = captured
        if not is_npu and backend is not None:
            meta = getattr(backend, "forward_metadata", None)
            if (
                meta is not None
                and torch.is_tensor(getattr(meta, "kv_indptr", None))
                and torch.is_tensor(getattr(meta, "kv_indices", None))
                and torch.is_tensor(getattr(meta, "qo_indptr", None))
            ):
                self.triton_metadata[bucket] = meta
        self.output_buffers[bucket] = LogitsProcessorOutput(
            next_token_logits=buffers.next_token_logits,
            hidden_states=None,
        )


def create_sr_tail_extend_graph_runner(drafter):
    if getattr(drafter.server_args, "disable_cuda_graph", False):
        return None
    backend = device_backend_key(drafter.device)
    if backend == "npu":
        from sglang.srt.hardware_backend.npu.graph_runner.sr_tail_extend_npu_graph_runner import (
            SRTailExtendNpuGraphRunner,
        )

        return SRTailExtendNpuGraphRunner(drafter)
    if backend in ("cuda", "hip"):
        return SRTailExtendGraphRunner(drafter)
    return None
