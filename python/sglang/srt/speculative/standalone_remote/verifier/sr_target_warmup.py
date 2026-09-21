"""NPU SR Target extra-graph kernel warmup.

Isolated scratch tensors and a tracking allocator. Does not submit real
requests or mutate radix / output tokens.
"""

from __future__ import annotations

import logging
import torch

from sglang.srt.mem_cache.common import alloc_paged_token_slots_extend, get_last_loc_torch
from sglang.srt.speculative.standalone_remote.drafter.sr_tree_paged_layout import (
    SR_TREE_WARMUP_ENV,
    read_sr_tree_warmup_env,
)
from sglang.srt.speculative.standalone_remote.sr_align import SRWarmupFatalError
from sglang.srt.speculative.standalone_remote.sr_verify_layout import (
    copy_paged_kv_buffer_by_slot,
)
from sglang.srt.speculative.standalone_remote.sr_warmup import (
    SRWarmupCacheAdapter,
    SRWarmupTrackingAllocator,
    alloc_prefix_pages,
    allocator_in_free_group,
    clone_page_lists,
    fill_prefix_slots,
    make_scratch_req_pool,
    restore_page_lists,
    slots_to_pages,
    sr_warmup_raw_batch_sizes,
    page_set,
    target_keep_len,
    warmup_should_run_finished_filter,
    warmup_synchronize,
)
from sglang.srt.speculative.spec_utils import (
    assign_req_to_token_pool_func,
    create_accept_length_filter,
    filter_finished_cache_loc_kernel,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
)
from sglang.srt.utils import is_npu, next_power_of_2

logger = logging.getLogger(__name__)


def configured_target_capture_bs(worker) -> list[int]:
    runner = getattr(getattr(worker, "target_worker", None), "model_runner", None)
    graph = getattr(runner, "graph_runner", None) or getattr(runner, "cuda_graph_runner", None)
    capture_bs = [int(b) for b in (getattr(graph, "capture_bs", None) or []) if int(b) > 0]
    if capture_bs:
        return capture_bs
    raw = getattr(getattr(worker, "server_args", None), "cuda_graph_bs", None) or []
    return [int(b) for b in raw if int(b) > 0]


def target_warmup_raw_batch_sizes(worker) -> tuple[tuple[int, ...], str | None]:
    pool = getattr(worker, "req_to_token_pool", None)
    mapping = getattr(pool, "req_to_token", None) if pool is not None else None
    rows = int(mapping.shape[0]) if mapping is not None else 0
    args = getattr(worker, "server_args", None)
    return sr_warmup_raw_batch_sizes(
        configured_target_capture_bs(worker),
        rows,
        getattr(args, "max_running_requests", None),
        getattr(args, "standalone_remote_max_batch_size", None),
    )


def _verify_cross_page_cases(page: int, draft_num: int):
    """(seq, accept, kind) using verify width, not Draft steps."""
    seq = max(int(page) - 2, 1)
    keep_all = target_keep_len(seq, max(draft_num - 1, 0), draft_num, page)
    free_some = target_keep_len(seq, 0, draft_num, page)
    cases = []
    if keep_all >= page and keep_all == seq + draft_num:
        cases.append((seq, max(draft_num - 1, 0), "keep"))
    if free_some < seq + draft_num:
        cases.append((seq, 0, "free"))
    if not cases:
        cases.append((seq, 0, "free"))
    return cases


def _run_cache_loc(bs, draft_num, seq, accept, device, page, out_cache_loc):
    need = int(bs) * int(draft_num)
    if torch.is_tensor(out_cache_loc) and int(out_cache_loc.numel()) >= need:
        out = out_cache_loc.reshape(-1)[:need].to(dtype=torch.int64)
    else:
        out = torch.arange(need, dtype=torch.int64, device=device)
    accept_index = torch.arange(need, dtype=torch.int64, device=device)
    accept_length = torch.full((bs,), int(accept), dtype=torch.int32, device=device)
    seq_lens = torch.full((bs,), int(seq), dtype=torch.int64, device=device)
    src, tgt, to_free_num = get_src_tgt_cache_loc(
        seq_lens, out, accept_index, accept_length, draft_num, page
    )
    to_free_slots = torch.empty(
        (int(to_free_num.sum().item()),), dtype=torch.int64, device=device
    )
    get_target_cache_loc[(bs,)](
        tgt,
        to_free_slots,
        accept_length,
        to_free_num,
        out,
        draft_num,
        next_power_of_2(draft_num),
        next_power_of_2(bs),
    )
    return src, tgt, to_free_num, to_free_slots


def _maybe_filter(kind, bs, draft_num, tgt, accept, device):
    if not warmup_should_run_finished_filter(kind):
        return False
    unfinished = torch.tensor([0], dtype=torch.int64, device=device)
    accept_length = torch.full((bs,), int(accept), dtype=torch.int32, device=device)
    seq_lens = torch.zeros((bs,), dtype=torch.int64, device=device)
    filt = create_accept_length_filter(accept_length, unfinished, seq_lens)
    out = torch.empty(
        int(filt.sum().item()) if int(filt.sum().item()) > 0 else 1,
        dtype=torch.int64,
        device=device,
    )
    filter_finished_cache_loc_kernel[(bs,)](
        out,
        tgt,
        accept_length,
        filt,
        next_power_of_2(bs),
        next_power_of_2(draft_num),
    )
    return True


def _overlap_kv_check(kv_buffer, slots, page):
    if kv_buffer is None or not torch.is_tensor(kv_buffer) or kv_buffer.dim() != 6:
        return False
    if int(slots.numel()) < 2:
        return False
    src = slots[:2].to(dtype=torch.int64)
    tgt = torch.stack((slots[1], slots[0])).to(dtype=torch.int64)
    flat = kv_buffer.view(kv_buffer.shape[0], kv_buffer.shape[1], -1, kv_buffer.shape[4], kv_buffer.shape[5])
    before = flat[:, :, src].clone()
    copy_paged_kv_buffer_by_slot(kv_buffer, src, tgt)
    after = flat[:, :, tgt]
    return bool(torch.equal(after, before))


def warm_sr_target_kernels(worker) -> dict:
    coverage = {
        "allocation": [],
        "mapping": [],
        "kernels": [],
        "skipped": None,
        "filter": [],
    }
    if not read_sr_tree_warmup_env():
        coverage["skipped"] = f"{SR_TREE_WARMUP_ENV}=0"
        logger.info("[SR] target kernel warmup skipped: %s=0", SR_TREE_WARMUP_ENV)
        return coverage
    if not is_npu() or int(worker.page_size) <= 1 or int(worker.topk) <= 1:
        coverage["skipped"] = "not npu tree-paged target"
        return coverage
    raw_bs, skip = target_warmup_raw_batch_sizes(worker)
    if skip:
        coverage["skipped"] = skip
        logger.info("[SR] target kernel warmup skipped: %s", skip)
        return coverage

    inner = worker.token_to_kv_pool_allocator
    if allocator_in_free_group(inner):
        coverage["skipped"] = "allocator is in a free-group"
        return coverage
    page = int(worker.page_size)
    draft_num = int(worker.speculative_num_draft_tokens)
    device = worker.device
    pool = worker.req_to_token_pool.req_to_token
    snapshot = clone_page_lists(inner)
    adapter = SRWarmupTrackingAllocator(inner)
    cache = SRWarmupCacheAdapter(adapter)
    scratch = make_scratch_req_pool(max(raw_bs), int(pool.shape[1]), pool.device, pool.dtype)
    saved_fields = {}
    fatal = None
    try:
        empty_slots = torch.empty((0,), dtype=torch.int64, device=device)
        before_zero = page_set(inner)
        adapter.free(empty_slots)
        if page_set(inner) != before_zero or slots_to_pages(empty_slots, page):
            raise SRWarmupFatalError("zero-free released pages")
        coverage["kernels"].append(("zero_free", 0, []))
        for bs in raw_bs:
            for seq, accept, cross_kind in _verify_cross_page_cases(page, draft_num):
                prefixes = [int(seq)] * int(bs)
                restore_page_lists(inner, snapshot)
                adapter.owned_pages = set()
                adapter._known = True
                scratch.req_to_token.fill_(-1)
                prefix_slots = alloc_prefix_pages(adapter, prefixes)
                if prefix_slots is None:
                    continue
                for row, slots in enumerate(prefix_slots):
                    fill_prefix_slots(scratch.req_to_token, row, prefixes[row], slots, page)
                req_pool_indices = torch.arange(bs, dtype=torch.int64, device=device)
                seq_lens = torch.tensor(prefixes, dtype=torch.int64, device=device)
                seq_lens_cpu = torch.tensor(prefixes, dtype=torch.int64)
                last_loc = get_last_loc_torch(scratch.req_to_token, req_pool_indices, seq_lens)
                end = seq_lens + draft_num
                end_cpu = seq_lens_cpu + draft_num
                try:
                    out_loc, _backup = alloc_paged_token_slots_extend(
                        cache,
                        seq_lens,
                        seq_lens_cpu,
                        end,
                        end_cpu,
                        last_loc,
                        int(bs) * draft_num,
                        backup_state=True,
                    )
                    assign_req_to_token_pool_func(
                        req_pool_indices,
                        scratch.req_to_token,
                        seq_lens,
                        end,
                        out_loc,
                        int(bs),
                    )
                    coverage["allocation"].append((int(bs), tuple(prefixes)))
                    coverage["mapping"].append((int(bs), tuple(prefixes)))
                    src, tgt, to_free_num, to_free_slots = _run_cache_loc(
                        int(bs), draft_num, seq, accept, device, page, out_loc
                    )
                    slot_count = int(to_free_num.sum().item())
                    if slot_count == 0:
                        empty = to_free_slots if int(to_free_slots.numel()) == 0 else to_free_slots[:0]
                        before_pages = page_set(inner)
                        adapter.free(empty)
                        freed_pages = slots_to_pages(empty, page)
                        if freed_pages or page_set(inner) != before_pages:
                            raise SRWarmupFatalError("zero-free released pages")
                    else:
                        freed_pages = slots_to_pages(to_free_slots, page)
                        if freed_pages <= adapter.owned_pages:
                            adapter.free(to_free_slots)
                        else:
                            freed_pages = set()
                    coverage["kernels"].append(
                        (int(bs), cross_kind, slot_count, sorted(freed_pages))
                    )
                    kv = getattr(inner.get_kvcache(), "kv_buffer", None)
                    if kv is not None and int(out_loc.numel()) >= 2:
                        _overlap_kv_check(kv, out_loc[:2], page)
                    for kind in ("continue", "partial", "all_finished"):
                        if kind == "all_finished":
                            _ = torch.empty((0,), dtype=torch.int64, device=device)
                        ran = _maybe_filter(kind, int(bs), draft_num, tgt, accept, device)
                        if ran:
                            coverage["filter"].append((int(bs), kind))
                except SRWarmupFatalError:
                    raise
                except Exception as exc:
                    logger.warning("[SR] target warmup uncovered bs=%s: %s", bs, exc)
                warmup_synchronize(device)
        warmup_synchronize(device)
    except SRWarmupFatalError as exc:
        fatal = exc
    finally:
        _ = saved_fields
        if fatal is not None:
            raise fatal
        restore_page_lists(inner, snapshot)
        adapter.owned_pages = set()
    logger.info(
        "[SR] target warmup allocation=%s mapping=%s kernels=%s filter=%s",
        coverage["allocation"],
        coverage["mapping"],
        coverage["kernels"],
        coverage["filter"],
    )
    return coverage
