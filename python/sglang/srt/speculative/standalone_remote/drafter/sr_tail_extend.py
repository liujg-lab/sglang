"""Transactional, prefix-preserving ingestion for standalone remote trees."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

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
        and seed[0] is not None
        and seed[1] is not None
        and seed[3] is not None
        and getattr(req, "sr_tree_seed_boundary", None) == boundary
        and getattr(req, "sr_tree_seed_revision", None)
        == int(getattr(req, "sr_prefix_revision", 0))
        and int(req.kv_committed_len) == boundary
    )


def stamp_tree_seed(req: Req, boundary: int) -> None:
    req.sr_tree_seed_boundary = boundary
    req.sr_tree_seed_revision = int(getattr(req, "sr_prefix_revision", 0))


def clear_fill_credential(req: Req) -> None:
    """Drop the incremental fill credential.

    The credential records list identity, published length, and revision. It
    cannot see ``fill_ids[i] = token``. This path is the only writer of a
    registered list; every other writer must clear the credential first.
    ``invalidate_tree_seed`` does not clear it, because a proven append
    increments revision and still extends the same list.
    """
    req.sr_fill_credential = None


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
class SRFillCredential:
    """Identity of the fill list this path published.

    ``revision`` is ``sr_prefix_revision`` at publish time. A later proven
    append chains from ``SRAlignResult.old_prefix_revision``, which was sampled
    before ``invalidate_tree_seed`` incremented the revision.
    """

    owned_id: int
    published_len: int
    revision: int


@dataclass(frozen=True)
class SRTailPlanSnapshot:
    """Request identity frozen while the tail plan is built.

    Commit refuses the plan when any of these change. It does not rebuild a
    fill list from the mutated request and publish that with the old KV plan.
    """

    fill_is_list: bool
    fill_id: int
    fill_len: int
    origin_id: int
    origin_len: int
    output_id: int
    output_len: int
    revision: int
    compute_len: int
    fill_append_len: int


@dataclass(frozen=True)
class SRTailExtendPlan:
    req: Req
    total_len: int
    prefix_len: int
    last_token: int
    compute_tokens: Tuple[int, ...]
    fill_append_tokens: Tuple[int, ...]
    snapshot: SRTailPlanSnapshot
    materialized_len: int
    revision: int
    recapture: bool = False
    original_len: Optional[int] = None
    copy_src_slots: Optional[List[int]] = None
    copy_lease: Any = None
    replay_tokens: Optional[Tuple[int, ...]] = None
    rebuild_fill: Optional[Tuple[int, ...]] = None

    @property
    def end(self) -> int:
        return self.total_len

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


def _tail_needs_full_replay(req: Req) -> bool:
    """Penalty, grammar, and custom processors still need the whole sequence."""
    if getattr(req, "grammar", None) is not None:
        return True
    if getattr(req, "sr_grammar_template", None) is not None:
        return True
    if getattr(req, "custom_logit_processor", None):
        return True
    params = getattr(req, "sampling_params", None)
    if params is None:
        return True
    if float(getattr(params, "repetition_penalty", 1.0)) != 1.0:
        return True
    if float(getattr(params, "frequency_penalty", 0.0)) != 0.0:
        return True
    if float(getattr(params, "presence_penalty", 0.0)) != 0.0:
        return True
    if int(getattr(params, "min_new_tokens", 0) or 0) != 0:
        return True
    return False


def _fill_update_mode(req: Req, total_len: int) -> Optional[str]:
    """Return ``equal`` or ``append`` when the published fill matches this align.

    Identity, published length, and alias checks always apply. ``equal`` keeps
    the list when the credential revision is current and the published length
    is already ``total_len``. ``append`` requires the credential to name this
    align's old prefix: its revision equals ``old_prefix_revision``, its
    length equals ``old_local_len``, the live revision advanced exactly once,
    and ``sr_prefix_proven`` is set. Anything else rebuilds.
    """
    cred = getattr(req, "sr_fill_credential", None)
    fill = getattr(req, "fill_ids", None)
    if cred is None or not isinstance(fill, list):
        return None
    if fill is getattr(req, "origin_input_ids", None) or fill is getattr(
        req, "output_ids", None
    ):
        return None
    if id(fill) != int(cred.owned_id) or len(fill) != int(cred.published_len):
        return None
    align = getattr(req, "sr_align_result", None)
    if align is None:
        return None
    revision = int(getattr(req, "sr_prefix_revision", 0) or 0)
    kind = getattr(align, "kind", None)
    if kind == "equal":
        if int(cred.revision) == revision and int(cred.published_len) == int(total_len):
            return "equal"
        return None
    if kind not in ("append_one", "append_n"):
        return None
    old_revision = int(getattr(align, "old_prefix_revision", -1))
    old_len = int(getattr(align, "old_local_len", -1))
    if (
        bool(getattr(req, "sr_prefix_proven", False))
        and int(cred.revision) == old_revision
        and int(cred.published_len) == old_len
        and revision == old_revision + 1
        and int(total_len) >= old_len
    ):
        return "append"
    return None


def _capture_tail_snapshot(
    req: Req, compute_len: int, fill_append_len: int
) -> SRTailPlanSnapshot:
    fill = getattr(req, "fill_ids", None)
    origin = getattr(req, "origin_input_ids", None)
    output = getattr(req, "output_ids", None)
    fill_is_list = isinstance(fill, list)
    return SRTailPlanSnapshot(
        fill_is_list=fill_is_list,
        fill_id=id(fill) if fill_is_list else 0,
        fill_len=len(fill) if fill_is_list else -1,
        origin_id=id(origin),
        origin_len=len(origin or []),
        output_id=id(output),
        output_len=len(output or []),
        revision=int(getattr(req, "sr_prefix_revision", 0) or 0),
        compute_len=int(compute_len),
        fill_append_len=int(fill_append_len),
    )


def _tail_snapshot_holds(plan: SRTailExtendPlan) -> bool:
    req = plan.req
    snap = plan.snapshot
    fill = getattr(req, "fill_ids", None)
    if snap.fill_is_list:
        if (
            not isinstance(fill, list)
            or id(fill) != snap.fill_id
            or len(fill) != snap.fill_len
        ):
            return False
    elif isinstance(fill, list):
        return False
    origin = getattr(req, "origin_input_ids", None)
    output = getattr(req, "output_ids", None)
    if id(origin) != snap.origin_id or len(origin or []) != snap.origin_len:
        return False
    if id(output) != snap.output_id or len(output or []) != snap.output_len:
        return False
    if int(getattr(req, "sr_prefix_revision", 0) or 0) != snap.revision:
        return False
    if len(plan.compute_tokens) != snap.compute_len:
        return False
    if len(plan.fill_append_tokens) != snap.fill_append_len:
        return False
    if len(plan.compute_tokens) != plan.length:
        return False
    return True


def plan_tail_extend(
    req: Req,
    *,
    vocab_size: int,
    model_is_mrope: bool,
    materialized_len: Optional[int] = None,
):
    from sglang.srt.speculative.standalone_remote.drafter.sr_tree_kv_lease import (
        read_token_span,
    )

    origin = getattr(req, "origin_input_ids", None)
    output = getattr(req, "output_ids", None)
    origin_len = len(origin or [])
    total_len = origin_len + len(output or [])
    materialized = (
        int(req.kv_committed_len)
        if materialized_len is None
        else int(materialized_len)
    )
    allocated = int(req.kv_allocated_len)
    published = int(req.kv_committed_len)
    if (
        req.req_pool_idx is None
        or total_len <= 0
        or materialized < origin_len
        or materialized > total_len
        or (allocated != published and allocated != materialized)
    ):
        raise TailExtendRecoveryRequired("invalid or overallocated linear KV prefix")
    if materialized == total_len and tree_seed_is_current(req):
        return None
    recapture = materialized == total_len
    prefix_len = materialized - 1 if recapture else materialized
    protected = max(
        len(req.prefix_indices), int(getattr(req, "cache_protected_len", 0) or 0)
    )
    if recapture and prefix_len < protected:
        raise TailExtendRecoveryRequired("last KV token belongs to a protected prefix")
    compute_count = total_len - prefix_len
    if _tail_needs_full_replay(req):
        replay_tokens = tuple(read_token_span(origin, output, 0, total_len))
        compute_tokens = replay_tokens[prefix_len:]
    else:
        replay_tokens = None
        compute_tokens = tuple(
            read_token_span(origin, output, prefix_len, compute_count)
        )
    if len(compute_tokens) != compute_count:
        raise TailExtendRecoveryRequired("tail token span is shorter than its length")
    if any(token < 0 or token >= vocab_size for token in compute_tokens):
        raise TailExtendRecoveryRequired(
            "tail contains an unprocessed multimodal token"
        )
    if model_is_mrope:
        validate_tail_mrope(req.multimodal_inputs, prefix_len, compute_count)
    mode = _fill_update_mode(req, total_len)
    if mode == "append":
        cred = req.sr_fill_credential
        fill_count = total_len - int(cred.published_len)
        fill_append_tokens = tuple(
            read_token_span(origin, output, int(cred.published_len), fill_count)
        )
        if len(fill_append_tokens) != fill_count:
            raise TailExtendRecoveryRequired("fill append span is short")
        rebuild_fill = None
    elif mode == "equal":
        fill_append_tokens = ()
        rebuild_fill = None
    else:
        fill_append_tokens = ()
        rebuild_fill = (
            replay_tokens
            if replay_tokens is not None
            else tuple(read_token_span(origin, output, 0, total_len))
        )
        if len(rebuild_fill) != total_len:
            raise TailExtendRecoveryRequired("rebuilt fill span is short")
    return SRTailExtendPlan(
        req=req,
        total_len=total_len,
        prefix_len=prefix_len,
        last_token=int(compute_tokens[-1]),
        compute_tokens=compute_tokens,
        fill_append_tokens=fill_append_tokens,
        snapshot=_capture_tail_snapshot(
            req, len(compute_tokens), len(fill_append_tokens)
        ),
        materialized_len=materialized,
        revision=int(getattr(req, "sr_prefix_revision", 0) or 0),
        recapture=recapture,
        replay_tokens=replay_tokens,
        rebuild_fill=rebuild_fill,
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
        [token for p in plans for token in p.compute_tokens],
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


def _prompt_boundary(req: Req) -> int:
    padded = getattr(req, "sr_padded_ids", None)
    if padded is not None:
        return len(padded)
    return len(getattr(req, "origin_input_ids", None) or [])


def _penalty_output_len(plan: SRTailExtendPlan) -> int:
    boundary = _prompt_boundary(plan.req)
    if plan.replay_tokens is not None:
        return max(0, len(plan.replay_tokens) - boundary)
    return max(0, plan.total_len - boundary)


def _restore_committed_penalties(batch, plans) -> None:
    """Rebuild optional sampling penalties without replaying token decodes.

    A mixed batch still visits every row. Rows without ``replay_tokens`` keep
    frequency and presence neutral and repetition at 1. Their output length
    comes from ``total_len`` and the prompt boundary, not from a token scan.
    """
    orchestrator = batch.sampling_info.penalizer_orchestrator
    if not orchestrator.is_required:
        return
    from sglang.srt.sampling import penaltylib

    histories = []
    output_lens = []
    for plan in plans:
        output_lens.append(_penalty_output_len(plan))
        if plan.replay_tokens is None:
            histories.append(())
            continue
        boundary = _prompt_boundary(plan.req)
        histories.append(plan.replay_tokens[boundary:])
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
                    [[length] for length in output_lens],
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
        self._copy_leases = []

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
        self._copy_leases = []
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
                if lease is None:
                    continue
                lease.pending_free_event = ev
                if lease not in self._copy_leases:
                    self._copy_leases.append(lease)

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
        topk = max(1, int(getattr(self.scheduler.server_args, "speculative_eagle_topk", 1) or 1))
        p = getattr(logits_output, "tree_seed_topk_p", None)
        ix = getattr(logits_output, "tree_seed_topk_index", None)
        if (
            p is None
            or ix is None
            or p.ndim != 2
            or ix.ndim != 2
            or p.shape != ix.shape
            or p.shape != (count, topk)
        ):
            raise RuntimeError("tail extend did not produce one seed per request")
        prepared = []
        for i, plan in enumerate(self.plans):
            if (
                int(getattr(plan.req, "sr_prefix_revision", 0) or 0) != plan.revision
                or int(plan.req.kv_committed_len) != plan.materialized_len
                or not _tail_snapshot_holds(plan)
            ):
                raise RuntimeError("request changed during SR tail extend")
            if plan.rebuild_fill is not None and len(plan.rebuild_fill) != plan.total_len:
                raise RuntimeError("tail fill rebuild does not match the plan")
            seed = (
                p[i : i + 1].detach().clone(),
                ix[i : i + 1].detach().clone(),
                None,
                torch.tensor([plan.last_token], dtype=torch.int64, device=ix.device),
            )
            if plan.rebuild_fill is not None:
                replacement = list(plan.rebuild_fill)
                cred = SRFillCredential(id(replacement), plan.total_len, plan.revision)
                action = ("replace", replacement)
            else:
                fill = plan.req.fill_ids
                cred = SRFillCredential(id(fill), plan.total_len, plan.revision)
                action = ("extend", list(plan.fill_append_tokens))
            prepared.append((plan, seed, cred, action))
        undo = []
        try:
            for plan, _seed, _cred, action in prepared:
                kind, payload = action
                if kind != "extend" or not payload:
                    continue
                fill = plan.req.fill_ids
                old_len = len(fill)
                undo.append((fill, old_len))
                fill.extend(payload)
        except Exception:
            for fill, old_len in undo:
                del fill[old_len:]
            raise
        for plan, seed, cred, action in prepared:
            kind, payload = action
            plan.req.kv_committed_len = plan.end
            plan.req.kv_allocated_len = plan.end
            if kind == "replace":
                plan.req.fill_ids = payload
            plan.req.sr_fill_credential = cred
            plan.req.sr_tree_seed = seed
            stamp_tree_seed(plan.req, plan.end)
            plan.req.draft_generation_start_len = len(plan.req.output_ids or [])
        self.committed = True

    def _clear_installed_copy_markers(self) -> None:
        done = self.copy_done_event
        for lease in self._copy_leases:
            ev = getattr(lease, "pending_free_event", None)
            if ev is None:
                continue
            if isinstance(ev, _UnfinishedCopyEvent) or ev is done:
                lease.pending_free_event = None

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
            self._clear_installed_copy_markers()
        for p, grammar in zip(self.plans, self.grammars):
            p.req.grammar = grammar
        self._copy_hold = None
