"""Draft extra-graph alloc/mapping warmup host.

Binds production SRTreeDrafter methods onto an isolated host. Does not copy
method source and does not mutate the production scheduler or drafter methods.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Optional, Sequence

import torch

from sglang.srt.speculative.standalone_remote.sr_align import SRWarmupFatalError
from sglang.srt.speculative.standalone_remote.sr_warmup import (
    SRWarmupCacheAdapter,
    SRWarmupTrackingAllocator,
    alloc_prefix_pages,
    allocator_in_free_group,
    clone_page_lists,
    fill_prefix_slots,
    make_scratch_req_pool,
    restore_page_lists,
    sr_warmup_raw_batch_sizes,
    warmup_nnp_jump_reachable,
    warmup_prefix_candidates,
    warmup_prefixes_for_bs,
    warmup_synchronize,
)

logger = logging.getLogger(__name__)


class SRDraftWarmupHost:
    def __init__(self, drafter, adapter, scratch_pool, cache, *, lease_supported: bool):
        self.page_size = drafter.page_size
        self.topk = drafter.topk
        self.speculative_num_steps = drafter.speculative_num_steps
        self.device = drafter.device
        self.token_to_kv_pool_allocator = adapter
        self.req_to_token_pool = scratch_pool
        self.scheduler = SimpleNamespace(sr_tree_leases=None, tree_cache=cache)
        self._lease_flag = bool(lease_supported)
        self.extend_lens = None
        self.num_new_pages_per_topk = None

    def _lease_supported(self) -> bool:
        return self._lease_flag

    def _try_alloc_lease_tree_kv(self, batch):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_drafter import (
            SRTreeDrafter,
        )

        return SRTreeDrafter._try_alloc_lease_tree_kv(self, batch)

    def _alloc_tree_kv(self, batch):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_drafter import (
            SRTreeDrafter,
        )

        return SRTreeDrafter._alloc_tree_kv(self, batch)

    def _apply_tree_mapping(self, batch, raw_cache_loc):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_drafter import (
            SRTreeDrafter,
        )

        return SRTreeDrafter._apply_tree_mapping(self, batch, raw_cache_loc)

    def _restore_tree_mapping(self, batch, lease_state):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_drafter import (
            SRTreeDrafter,
        )

        return SRTreeDrafter._restore_tree_mapping(self, batch, lease_state)

    def _free_lease_alloc(self, lease_state):
        from sglang.srt.speculative.standalone_remote.drafter.sr_tree_drafter import (
            SRTreeDrafter,
        )

        return SRTreeDrafter._free_lease_alloc(self, lease_state)


def _dummy_reqs(prefixes: Sequence[int]):
    return [
        SimpleNamespace(req_pool_idx=i, rid=f"sr-warmup-{i}")
        for i in range(len(prefixes))
    ]


def _make_batch(host: SRDraftWarmupHost, prefixes: Sequence[int], cache):
    device = host.device
    reqs = _dummy_reqs(prefixes)
    seq = list(prefixes)
    batch = SimpleNamespace(
        reqs=reqs,
        req_to_token_pool=host.req_to_token_pool,
        token_to_kv_pool_allocator=host.token_to_kv_pool_allocator,
        tree_cache=cache,
        req_pool_indices=torch.arange(len(seq), dtype=torch.int64, device=device),
        seq_lens=torch.tensor(seq, dtype=torch.int64, device=device),
        seq_lens_cpu=torch.tensor(seq, dtype=torch.int64),
        spec_info=SimpleNamespace(positions=None),
        out_cache_loc=None,
        seq_lens_sum=int(sum(seq)),
    )
    batch.batch_size = lambda: len(seq)
    return batch


def _install_prefixes(host, prefixes, prefix_slots) -> None:
    mapping = host.req_to_token_pool.req_to_token
    for row, (prefix, slots) in enumerate(zip(prefixes, prefix_slots)):
        fill_prefix_slots(mapping, row, prefix, slots, host.page_size)


def configured_draft_capture_bs(drafter) -> list[int]:
    runner = getattr(drafter, "cuda_graph_runner", None)
    capture_bs = [int(b) for b in (getattr(runner, "capture_bs", None) or []) if int(b) > 0]
    if capture_bs:
        return capture_bs
    raw = getattr(getattr(drafter, "server_args", None), "cuda_graph_bs", None) or []
    return [int(b) for b in raw if int(b) > 0]


def draft_warmup_raw_batch_sizes(drafter) -> tuple[tuple[int, ...], Optional[str]]:
    pool = getattr(drafter, "req_to_token_pool", None)
    mapping = getattr(pool, "req_to_token", None) if pool is not None else None
    rows = int(mapping.shape[0]) if mapping is not None else 0
    args = getattr(drafter, "server_args", None)
    return sr_warmup_raw_batch_sizes(
        configured_draft_capture_bs(drafter),
        rows,
        getattr(args, "max_running_requests", None),
        getattr(args, "standalone_remote_max_batch_size", None),
    )


def _run_one(host, cache, prefixes, *, lease: bool) -> dict:
    adapter = host.token_to_kv_pool_allocator
    prefix_slots = alloc_prefix_pages(adapter, prefixes)
    if prefix_slots is None:
        return {"ok": False, "reason": "capacity"}
    _install_prefixes(host, prefixes, prefix_slots)
    batch = _make_batch(host, prefixes, cache)
    lease_state = None
    try:
        if lease:
            leased = host._try_alloc_lease_tree_kv(batch)
            if leased is None:
                return {"ok": False, "reason": "lease_skip"}
            raw_cache_loc, lease_state = leased
            host._apply_tree_mapping(batch, raw_cache_loc)
        else:
            _backup, lease_state = host._alloc_tree_kv(batch)
        mapping = batch.out_cache_loc
        return {"ok": True, "lease_state": lease_state, "mapping": mapping}
    except SRWarmupFatalError:
        raise
    except Exception:
        if lease_state is not None and adapter._known:
            try:
                if any(
                    adapter.still_owns(slots)
                    for slots in lease_state.get("page_slots", [])
                ):
                    host._free_lease_alloc(lease_state)
            except SRWarmupFatalError:
                raise
        raise


def _recycle_scenario(host, cache, prefixes, lease_state, adapter) -> None:
    batch = _make_batch(host, prefixes, cache)
    host._restore_tree_mapping(batch, lease_state)
    warmup_synchronize(host.device)
    if lease_state and adapter._known:
        still = [
            slots
            for slots in lease_state.get("page_slots", [])
            if adapter.still_owns(slots)
        ]
        if still:
            host._free_lease_alloc(
                {**lease_state, "page_slots": still, "reqs": lease_state.get("reqs", [])}
            )


def warm_draft_alloc_mapping(drafter) -> dict:
    """Run isolated lease + ordinary alloc/mapping warmup. Returns coverage."""
    coverage = {
        "allocation": [],
        "mapping": [],
        "unreachable_jump": False,
        "skipped": None,
    }
    raw_bs, skip = draft_warmup_raw_batch_sizes(drafter)
    if skip:
        coverage["skipped"] = skip
        logger.info("[SR] tree alloc/mapping warmup skipped: %s", skip)
        return coverage
    inner = drafter.token_to_kv_pool_allocator
    if allocator_in_free_group(inner):
        coverage["skipped"] = "allocator is in a free-group"
        logger.info("[SR] tree alloc/mapping warmup skipped: allocator free-group")
        return coverage
    pool = drafter.req_to_token_pool.req_to_token
    candidates = warmup_prefix_candidates(
        drafter.page_size,
        drafter.speculative_num_steps,
        int(pool.shape[1]),
        drafter.topk,
    )
    if not warmup_nnp_jump_reachable(drafter.page_size, drafter.speculative_num_steps):
        coverage["unreachable_jump"] = True
    if not candidates:
        coverage["skipped"] = "no prefix candidates fit mapping width"
        logger.info("[SR] tree alloc/mapping warmup skipped: no prefix candidates")
        return coverage

    snapshot = clone_page_lists(inner)
    adapter = SRWarmupTrackingAllocator(inner)
    cache = SRWarmupCacheAdapter(adapter)
    scratch = make_scratch_req_pool(
        max(raw_bs), int(pool.shape[1]), pool.device, pool.dtype
    )
    saved = {
        "extend_lens": getattr(drafter, "extend_lens", None),
        "num_new_pages_per_topk": getattr(drafter, "num_new_pages_per_topk", None),
    }
    fatal = None
    try:
        for lease, label in ((True, "lease"), (False, "ordinary")):
            host = SRDraftWarmupHost(
                drafter, adapter, scratch, cache, lease_supported=lease
            )
            for bs in raw_bs:
                for prefixes in warmup_prefixes_for_bs(candidates, bs):
                    key = (label, int(bs), tuple(prefixes))
                    restore_page_lists(inner, snapshot)
                    adapter.owned_pages = set()
                    adapter._known = True
                    scratch.req_to_token.fill_(-1)
                    try:
                        result = _run_one(host, cache, prefixes, lease=lease)
                        if result.get("ok"):
                            coverage["allocation"].append(key)
                            if result.get("mapping") is not None:
                                coverage["mapping"].append(key)
                        _recycle_scenario(
                            host, cache, prefixes, result.get("lease_state"), adapter
                        )
                    except SRWarmupFatalError:
                        raise
                    except Exception as exc:
                        logger.warning(
                            "[SR] tree %s warmup uncovered %s: %s", label, key, exc
                        )
        warmup_synchronize(drafter.device)
    except SRWarmupFatalError as exc:
        fatal = exc
    finally:
        drafter.extend_lens = saved["extend_lens"]
        drafter.num_new_pages_per_topk = saved["num_new_pages_per_topk"]
        if fatal is not None:
            raise fatal
        restore_page_lists(inner, snapshot)
        adapter.owned_pages = set()
    return coverage
