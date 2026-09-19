"""Transactional, prefix-preserving ingestion for standalone remote trees."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


def _evict_tail_capacity(tree_cache, capacity):
    from sglang.srt.mem_cache.common import evict_from_tree_cache

    evict_from_tree_cache(tree_cache, capacity)


class TailExtendRecoveryRequired(ValueError):
    """The existing prefix cannot be used by a text-tail extend."""


def invalidate_tree_seed(req: Req) -> None:
    req.sr_prefix_revision = int(getattr(req, "sr_prefix_revision", 0)) + 1
    req.sr_tree_seed = None
    req.sr_tree_seed_boundary = None


def tree_seed_is_current(req: Req) -> bool:
    seed = getattr(req, "sr_tree_seed", None)
    boundary = len(req.origin_input_ids or []) + len(req.output_ids or [])
    return (
        isinstance(seed, (tuple, list))
        and len(seed) == 4
        and all(value is not None for value in seed)
        and getattr(req, "sr_tree_seed_boundary", None) == boundary
        and getattr(req, "sr_tree_seed_revision", None)
        == int(getattr(req, "sr_prefix_revision", 0))
        and int(req.kv_committed_len) == boundary
    )


def stamp_tree_seed(req: Req, boundary: int) -> None:
    req.sr_tree_seed_boundary = boundary
    req.sr_tree_seed_revision = int(getattr(req, "sr_prefix_revision", 0))


class _UnfinishedCopyEvent:
    """Keep lease pages pinned if copy already submitted but completion record failed."""

    def query(self) -> bool:
        return False


def _device_type(device):
    if device is None:
        return None
    return getattr(device, "type", None) or str(device)


def _is_cpu_device(device) -> bool:
    dev_type = _device_type(device)
    return device is None or dev_type is None or "cpu" in str(dev_type)


def wait_copy_event(event) -> None:
    """Wait for copy on the current device stream. Do not CPU-synchronize."""
    if event is None:
        return
    device = getattr(event, "device", None)
    if device is None:
        return
    dev_type = getattr(device, "type", None)
    if dev_type is None or "cpu" in str(dev_type):
        return
    module = torch.get_device_module(dev_type)
    stream = module.current_stream(device)
    wait = getattr(stream, "wait_event", None)
    if not callable(wait):
        raise RuntimeError("cannot wait for KV copy: stream.wait_event missing")
    wait(event)


def get_kv_copy_stream(scheduler, device):
    """Return a persistent side copy stream, or None to copy on the compute stream.

    Probe failures are cached. After a copy is submitted, callers must not
    fall back through this helper.
    """
    if scheduler is None or _is_cpu_device(device):
        return None
    cached = getattr(scheduler, "_sr_kv_copy_stream", None)
    if cached is not None:
        return cached
    if getattr(scheduler, "_sr_kv_copy_stream_unsupported", False):
        return None
    try:
        module = torch.get_device_module(_device_type(device))
        factory = getattr(module, "Stream", None)
        ctx_fn = getattr(module, "stream", None)
        event_factory = getattr(module, "Event", None)
        current_stream_fn = getattr(module, "current_stream", None)
        if not all(
            callable(fn)
            for fn in (factory, ctx_fn, event_factory, current_stream_fn)
        ):
            scheduler._sr_kv_copy_stream_unsupported = True
            return None
        current = current_stream_fn(device)
        if not callable(getattr(current, "wait_event", None)):
            scheduler._sr_kv_copy_stream_unsupported = True
            return None
        stream = factory()
        if not callable(getattr(stream, "wait_event", None)):
            scheduler._sr_kv_copy_stream_unsupported = True
            return None
    except Exception:
        scheduler._sr_kv_copy_stream_unsupported = True
        return None
    scheduler._sr_kv_copy_stream = stream
    scheduler._sr_kv_copy_ctx = ctx_fn
    return stream


def validate_tail_mrope(mm_input, start: int, length: int) -> None:
    if mm_input is None:
        return
    positions = getattr(mm_input, "mrope_positions", None)
    if positions is None or positions.ndim != 2 or positions.shape[0] != 3:
        raise TailExtendRecoveryRequired("missing prefix M-RoPE positions")
    if start + length > positions.shape[1]:
        delta = getattr(mm_input, "mrope_position_delta", None)
        if delta is None or delta.numel() != 1:
            raise TailExtendRecoveryRequired("missing text continuation M-RoPE delta")


def tail_mrope_positions(mm_input, start: int, length: int) -> torch.Tensor:
    """Use stored positions, then extend every generated text position."""
    validate_tail_mrope(mm_input, start, length)
    end = start + length
    stored = mm_input.mrope_positions
    stored_end = min(end, stored.shape[1])
    parts = []
    if start < stored_end:
        parts.append(stored[:, start:stored_end].to(device="cpu", dtype=torch.int64))
    text_start = max(start, stored.shape[1])
    if text_start < end:
        delta = mm_input.mrope_position_delta.to(device="cpu", dtype=torch.int64)
        positions = torch.arange(text_start, end, dtype=torch.int64)
        parts.append((positions + delta.reshape(1)).unsqueeze(0).expand(3, -1))
    return torch.cat(parts, dim=1)


@dataclass(frozen=True)
class SRTailExtendPlan:
    req: Req
    tokens: List[int]
    materialized_len: int
    prefix_len: int
    revision: int
    recapture: bool = False
    original_len: Optional[int] = None
    copy_src_slots: Optional[List[int]] = None
    copy_lease: Any = None

    @property
    def end(self) -> int:
        return len(self.tokens)

    @property
    def length(self) -> int:
        return self.end - self.prefix_len

    @property
    def alloc_start(self) -> int:
        if self.original_len is None:
            return self.prefix_len
        return int(self.original_len)

    @property
    def reused_tokens(self) -> int:
        if self.copy_src_slots is None:
            return 0
        return len(self.copy_src_slots)

    @property
    def needs_suffix_alloc(self) -> bool:
        return (not self.recapture) or bool(self.reused_tokens)


def plan_tail_extend(
    req: Req,
    *,
    vocab_size: int,
    model_is_mrope: bool,
    materialized_len: Optional[int] = None,
):
    tokens = list(req.origin_input_ids or []) + list(req.output_ids or [])
    materialized = (
        int(req.kv_committed_len)
        if materialized_len is None
        else int(materialized_len)
    )
    allocated = int(req.kv_allocated_len)
    published = int(req.kv_committed_len)
    if (
        req.req_pool_idx is None
        or not tokens
        or materialized < len(req.origin_input_ids or [])
        or materialized > len(tokens)
        or (
            allocated != published
            and allocated != materialized
        )
    ):
        raise TailExtendRecoveryRequired("invalid or overallocated linear KV prefix")
    if materialized == len(tokens) and tree_seed_is_current(req):
        return None
    recapture = materialized == len(tokens)
    prefix_len = materialized - 1 if recapture else materialized
    protected = max(
        len(req.prefix_indices), int(getattr(req, "cache_protected_len", 0) or 0)
    )
    if recapture and prefix_len < protected:
        raise TailExtendRecoveryRequired("last KV token belongs to a protected prefix")
    if any(token < 0 or token >= vocab_size for token in tokens[prefix_len:]):
        raise TailExtendRecoveryRequired(
            "tail contains an unprocessed multimodal token"
        )
    if model_is_mrope:
        validate_tail_mrope(req.multimodal_inputs, prefix_len, len(tokens) - prefix_len)
    return SRTailExtendPlan(
        req=req,
        tokens=tokens,
        materialized_len=materialized,
        prefix_len=prefix_len,
        revision=int(getattr(req, "sr_prefix_revision", 0)),
        recapture=recapture,
    )


def make_tail_extend_batch(scheduler, plans: List[SRTailExtendPlan]) -> ScheduleBatch:
    from sglang.srt.layers.sampler import SamplingBatchInfo
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.standalone_remote.sr_align import apply_tree_seed_topk

    batch = ScheduleBatch.init_new(
        reqs=[p.req for p in plans],
        req_to_token_pool=scheduler.req_to_token_pool,
        token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
        tree_cache=scheduler.tree_cache,
        model_config=scheduler.model_config,
        enable_overlap=False,
        spec_algorithm=scheduler.spec_algorithm,
    )
    device = batch.device
    batch.forward_mode = ForwardMode.EXTEND
    batch.is_sr_tail_extend = True
    batch.return_logprob = False
    batch.is_prefill_only = False
    batch.return_hidden_states = False
    batch.extend_lens = [p.length for p in plans]
    batch.prefix_lens = [p.prefix_len for p in plans]
    batch.extend_num_tokens = sum(batch.extend_lens)
    batch.extend_logprob_start_lens = [0] * len(plans)
    batch.input_ids = torch.tensor(
        [token for p in plans for token in p.tokens[p.prefix_len :]],
        dtype=torch.int64,
        device=device,
    )
    batch.seq_lens_cpu = torch.tensor([p.end for p in plans], dtype=torch.int64)
    batch.seq_lens = batch.seq_lens_cpu.to(device)
    batch.orig_seq_lens = batch.seq_lens.to(dtype=torch.int32)
    batch.seq_lens_sum = sum(p.end for p in plans)
    batch.req_pool_indices = torch.tensor(
        [p.req.req_pool_idx for p in plans], dtype=torch.int64, device=device
    )
    batch.multimodal_inputs = [p.req.multimodal_inputs for p in plans]
    batch.top_logprobs_nums = [0] * len(plans)
    batch.token_ids_logprobs = [None] * len(plans)
    scheduler._sr_enable_tree_seed_hidden(batch)
    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
        batch, scheduler.model_config.vocab_size
    )
    apply_tree_seed_topk(batch, scheduler.server_args.speculative_eagle_topk)
    _restore_committed_penalties(batch, plans)
    return batch


def _restore_committed_penalties(batch, plans) -> None:
    """Rebuild optional sampling penalties without replaying token decodes."""
    orchestrator = batch.sampling_info.penalizer_orchestrator
    if not orchestrator.is_required:
        return
    from sglang.srt.sampling import penaltylib

    histories = [
        p.tokens[len(getattr(p.req, "sr_padded_ids", p.req.origin_input_ids)) :]
        for p in plans
    ]
    vocab_size = orchestrator.vocab_size
    counts = torch.zeros(
        (len(plans), vocab_size), dtype=torch.float32, device=batch.device
    )
    flat_indices = [
        row * vocab_size + token
        for row, history in enumerate(histories)
        for token in history
    ]
    if flat_indices:
        indices = torch.tensor(flat_indices, dtype=torch.int64, device=batch.device)
        counts.view(-1).scatter_add_(
            0, indices, torch.ones_like(indices, dtype=counts.dtype)
        )
    present = counts > 0
    for penalizer in orchestrator.penalizers.values():
        if not penalizer._is_prepared:
            continue
        if isinstance(penalizer, penaltylib.BatchedFrequencyPenalizer):
            penalizer.cumulated_frequency_penalties.copy_(
                counts * penalizer.frequency_penalties
            )
        elif isinstance(penalizer, penaltylib.BatchedPresencePenalizer):
            penalizer.cumulated_presence_penalties.copy_(
                present * penalizer.presence_penalties
            )
        elif isinstance(penalizer, penaltylib.BatchedRepetitionPenalizer):
            penalizer.cumulated_repetition_penalties.copy_(
                torch.where(present, penalizer.repetition_penalties, 1.0)
            )
        elif isinstance(penalizer, penaltylib.BatchedMinNewTokensPenalizer):
            penalizer.len_output_tokens.copy_(
                torch.tensor(
                    [[len(history)] for history in histories],
                    dtype=torch.int32,
                    device=batch.device,
                )
            )


class SRTailExtendTransaction:
    """Own only this synchronous ingest's allocations and mapping changes.

    Cache eviction precedes the allocator snapshot. No scheduler work may run
    between allocate and commit/rollback. Never free individual tail slots:
    some of them can belong to an existing, partially filled prefix page.
    """

    def __init__(self, scheduler, plans: List[SRTailExtendPlan]):
        self.scheduler = scheduler
        self.plans = plans
        self.allocator = scheduler.token_to_kv_pool_allocator
        self.mapping = scheduler.req_to_token_pool.req_to_token
        self.allocator_state = None
        self.mapping_snapshots = []
        self.grammars = [p.req.grammar for p in plans]
        self.submitted = False
        self.copy_submitted = False
        self.committed = False
        self.copy_done_event = None
        self._copy_stream = None
        self._copy_hold = None

    def allocate(self, batch: ScheduleBatch) -> None:
        if any(p.end > self.mapping.shape[1] for p in self.plans):
            raise RuntimeError("SR tail exceeds request-to-token mapping capacity")
        additions = [p for p in self.plans if p.needs_suffix_alloc]
        page_size = self.allocator.page_size
        count = sum(p.end - p.alloc_start for p in additions)
        capacity = sum(
            (
                (p.end + page_size - 1) // page_size
                - (p.alloc_start + page_size - 1) // page_size
            )
            * page_size
            for p in additions
        )
        if count:
            _evict_tail_capacity(self.scheduler.tree_cache, capacity)
            if self.allocator.available_size() < capacity:
                raise RuntimeError("insufficient KV capacity for SR tail extend")
        self.allocator_state = self.allocator.backup_state()
        self.mapping_snapshots = [
            self.mapping[p.req.req_pool_idx, p.alloc_start : p.end].clone()
            for p in self.plans
        ]
        allocated = None
        if additions:
            if page_size == 1:
                allocated = self.allocator.alloc(count)
            else:
                prefix_cpu = torch.tensor(
                    [p.alloc_start for p in additions], dtype=torch.int64
                )
                end_cpu = torch.tensor([p.end for p in additions], dtype=torch.int64)
                last_loc = torch.cat(
                    [
                        (
                            self.mapping[
                                p.req.req_pool_idx,
                                p.alloc_start - 1 : p.alloc_start,
                            ]
                            if p.alloc_start
                            else torch.tensor(
                                [-1], device=batch.device, dtype=torch.int64
                            )
                        )
                        for p in additions
                    ]
                )
                allocated = self.allocator.alloc_extend(
                    prefix_cpu.to(batch.device),
                    prefix_cpu,
                    end_cpu.to(batch.device),
                    end_cpu,
                    last_loc,
                    count,
                )
            if allocated is None:
                raise RuntimeError("insufficient KV capacity for SR tail extend")
        locations = []
        offset = 0
        for p in self.plans:
            if p.recapture and not p.reused_tokens:
                slots = self.mapping[p.req.req_pool_idx, p.prefix_len : p.end].clone()
            else:
                span = p.end - p.alloc_start
                suffix = allocated[offset : offset + span]
                offset += span
                self.mapping[p.req.req_pool_idx, p.alloc_start : p.end] = suffix
                slots = self.mapping[p.req.req_pool_idx, p.prefix_len : p.end]
            locations.append(slots)
        batch.out_cache_loc = torch.cat(locations) if locations else torch.empty(
            0, dtype=torch.int64, device=batch.device
        )

    def copy_reused_tree_kv(self) -> None:
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
            record_device_event,
        )
        from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
            copy_kv_pool_by_slot,
        )

        self.copy_done_event = None
        self._copy_stream = None
        self._copy_hold = None
        self.copy_submitted = False
        kv_pool = getattr(
            getattr(self.scheduler, "tp_worker", None), "model_runner", None
        )
        kv_pool = getattr(kv_pool, "token_to_kv_pool", None) if kv_pool else None
        if kv_pool is None:
            if any(p.copy_src_slots for p in self.plans):
                raise RuntimeError("tree KV copy required but kv pool is missing")
            return
        src_host = []
        dst_parts = []
        copy_plans = []
        for p in self.plans:
            src_list = p.copy_src_slots or []
            if not src_list:
                continue
            dst = self.mapping[
                p.req.req_pool_idx, p.alloc_start : p.alloc_start + len(src_list)
            ]
            src_host.extend(int(x) for x in src_list)
            dst_parts.append(dst)
            copy_plans.append(p)
        if not dst_parts:
            return
        merged_dst = torch.cat(dst_parts)
        merged_src = torch.as_tensor(
            src_host, dtype=torch.int64, device=merged_dst.device
        )
        if int(merged_src.numel()) != int(merged_dst.numel()):
            raise RuntimeError("tree KV copy src/dst length mismatch")
        self._copy_hold = (merged_src, merged_dst)

        def _bind_copy_event(ev) -> None:
            if ev is None:
                return
            for p in copy_plans:
                lease = getattr(p, "copy_lease", None)
                if lease is not None:
                    lease.pending_free_event = ev

        t0 = time.perf_counter()
        store = getattr(self.scheduler, "sr_tree_leases", None)
        device = merged_dst.device
        copy_stream = get_kv_copy_stream(self.scheduler, device)
        self._copy_stream = copy_stream
        use_device_event = not _is_cpu_device(device)
        if copy_stream is not None:
            ready = record_device_event(device, required=True)
            wait = getattr(copy_stream, "wait_event", None)
            if not callable(wait):
                raise RuntimeError(
                    "cannot wait for slot prep: copy stream wait_event missing"
                )
            wait(ready)
            ctx_fn = getattr(self.scheduler, "_sr_kv_copy_ctx", None)
            if not callable(ctx_fn):
                raise RuntimeError("copy stream context missing after probe")
            _bind_copy_event(_UnfinishedCopyEvent())
            self.copy_submitted = True
            with ctx_fn(copy_stream):
                copy_kv_pool_by_slot(kv_pool, merged_src, merged_dst)
                event = record_device_event(device, required=True)
        elif use_device_event:
            _bind_copy_event(_UnfinishedCopyEvent())
            self.copy_submitted = True
            copy_kv_pool_by_slot(kv_pool, merged_src, merged_dst)
            event = record_device_event(device, required=True)
        else:
            self.copy_submitted = True
            copy_kv_pool_by_slot(kv_pool, merged_src, merged_dst)
            event = None
        if store is not None:
            store.counts["tree_kv_copy_submit_ms"] += int(
                (time.perf_counter() - t0) * 1000
            )
        if (use_device_event or copy_stream is not None) and event is None:
            raise RuntimeError("KV copy submitted without a completion event")
        self.copy_done_event = event
        _bind_copy_event(event)

    def wait_copy_done(self) -> None:
        wait_copy_event(self.copy_done_event)
        self._copy_hold = None

    def commit(self, logits_output) -> None:
        count = len(self.plans)
        tensors = (
            logits_output.tree_seed_topk_p,
            logits_output.tree_seed_topk_index,
            logits_output.hidden_states,
        )
        if any(t is None or t.ndim != 2 or t.shape[0] != count for t in tensors):
            raise RuntimeError("tail extend did not produce one seed per request")
        seeds = []
        for i, p in enumerate(self.plans):
            if (
                int(getattr(p.req, "sr_prefix_revision", 0)) != p.revision
                or int(p.req.kv_committed_len) != p.materialized_len
            ):
                raise RuntimeError("request changed during SR tail extend")
            seeds.append(
                tuple(t[i : i + 1].detach().clone() for t in tensors)
                + (
                    torch.tensor(
                        [p.tokens[-1]], dtype=torch.int64, device=tensors[2].device
                    ),
                )
            )
        # All validation and tensor allocations precede publication.
        for p, seed in zip(self.plans, seeds):
            p.req.kv_committed_len = p.end
            p.req.kv_allocated_len = p.end
            p.req.fill_ids = p.tokens
            p.req.sr_tree_seed = seed
            stamp_tree_seed(p.req, p.end)
            p.req.draft_generation_start_len = len(p.req.output_ids or [])
        self.committed = True

    def rollback(self) -> None:
        if self.committed:
            return
        # Exceptional path only: do not recycle slots while kernels can write them.
        # A synchronization failure propagates; the device must then be abandoned.
        device_module = self.scheduler.device_module
        if self.allocator_state is not None:
            device_module.synchronize()
            for p, previous in zip(self.plans, self.mapping_snapshots):
                self.mapping[p.req.req_pool_idx, p.alloc_start : p.end].copy_(previous)
            self.allocator.restore_state(self.allocator_state)
        for p, grammar in zip(self.plans, self.grammars):
            p.req.grammar = grammar
        self._copy_hold = None
