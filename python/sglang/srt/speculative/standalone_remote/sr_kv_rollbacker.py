import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.speculative.standalone_remote.sr_align import (
    kv_release_len,
    rollback_free_range,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


class SRKVRollbacker:
    def __init__(
        self,
        token_to_kv_pool_allocator: "TokenToKVPoolAllocator",
        req_to_token_pool: "ReqToTokenPool",
        tree_cache: "BasePrefixCache",
        page_size: int = 1,
        tp_rank: int = 0,
    ) -> None:
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.req_to_token_pool = req_to_token_pool
        self.tree_cache = tree_cache
        self.page_size = page_size
        self.tp_rank = tp_rank

    def get_prefix_len(self, req: "Req") -> int:
        if hasattr(req, "prefix_indices") and req.prefix_indices is not None:
            try:
                return len(req.prefix_indices)
            except TypeError:
                return 0
        return 0

    def can_local_rollback(self, req: "Req", fork_point: int) -> bool:
        if self.page_size > 1 and (fork_point % self.page_size) != 0:
            return False
        prefix_len = self.get_prefix_len(req)
        if fork_point < prefix_len:
            return False
        return True

    def rollback(
        self, req: "Req", fork_point: int, current_kv_len: Optional[int] = None
    ) -> bool:
        if current_kv_len is None:
            current_kv_len = int(getattr(req, "kv_allocated_len", 0) or 0)
            if current_kv_len <= 0:
                input_ids = getattr(req, "origin_input_ids", [])
                output_ids = getattr(req, "output_ids", [])
                input_len = len(input_ids) if input_ids is not None else 0
                output_len = len(output_ids) if output_ids is not None else 0
                current_kv_len = max(0, input_len + output_len - 1)

        if self.can_local_rollback(req, fork_point):
            return self.local_rollback(req, fork_point, current_kv_len)
        return False

    def local_rollback(
        self,
        req: "Req",
        fork_point: int,
        current_kv_len: int,
    ) -> bool:
        if self.page_size > 1 and (fork_point % self.page_size) != 0:
            return False
        if req.req_pool_idx is None:
            return False

        allocated = int(getattr(req, "kv_allocated_len", 0) or 0)
        end = allocated if allocated > 0 else int(current_kv_len)
        if fork_point > end:
            # KV does not cover the fork. Leave lengths alone so the caller
            # reprefills instead of trimming tokens that were never stored.
            return False
        if fork_point == end:
            # Nothing is allocated past the fork. The prefix may still be the
            # draft pages tree expand wrote, which are not the verified KV.
            # Refuse so the caller reprefills instead of drafting from them.
            req.kv_committed_len = min(int(req.kv_committed_len or 0), fork_point, end)
            req.kv_allocated_len = min(end, fork_point)
            return False

        prefix_len = self.get_prefix_len(req)
        if fork_point < prefix_len:
            return False

        kv_indices = None
        stage = "pre"
        try:
            old_allocated = allocated if allocated > 0 else int(current_kv_len)
            max_len = self.req_to_token_pool.req_to_token.shape[1]
            start, end = rollback_free_range(
                fork_point, old_allocated, max_len
            )
            if start < end:
                stage = "map"
                kv_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, start:end
                ]
                stage = "free"
                self.token_to_kv_pool_allocator.free(kv_indices)
            req.kv_committed_len = fork_point
            req.kv_allocated_len = fork_point
            return True
        except Exception as exc:
            # Device-context failures are independent of which stage raised.
            # Only a free that has been attempted wraps an ordinary failure.
            from sglang.srt.speculative.standalone_remote.sr_align import (
                is_device_context_error,
            )
            from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
                SRTransferUnresolved,
            )

            if isinstance(exc, SRTransferUnresolved) or is_device_context_error(exc):
                raise
            if stage == "free":
                self._raise_submitted_rollback(req, kv_indices, exc)
            if self.tp_rank == 0:
                logger.warning(
                    "[SR] local_rollback failed for %s: %s", req.rid, exc
                )
            return False

    def _raise_submitted_rollback(self, req, slots, exc: BaseException) -> None:
        from sglang.srt.speculative.standalone_remote.sr_align import (
            is_device_context_error,
        )
        from sglang.srt.speculative.standalone_remote.sr_transfer_staging import (
            SRTransferUnresolved,
        )

        if isinstance(exc, SRTransferUnresolved) or is_device_context_error(exc):
            raise exc
        wrapped = SRTransferUnresolved(
            "local rollback release completion unknown for "
            f"{getattr(req, 'rid', None)}"
        )
        wrapped.req = req
        wrapped.slots = slots
        raise wrapped from exc

    def release_all_kv_for_finished_req(self, req: "Req") -> None:
        if req.req_pool_idx is None:
            return
        committed = int(getattr(req, "kv_committed_len", 0) or 0)
        allocated = int(getattr(req, "kv_allocated_len", 0) or 0)
        kv_len = kv_release_len(committed, allocated)
        req.kv_allocated_len = kv_len
        from sglang.srt.speculative.standalone_remote.drafter.sr_tail_extend import (
            clear_fill_credential,
        )

        clear_fill_credential(req)
        ids = list(req.origin_input_ids or []) + list(req.output_ids or [])
        req.fill_ids = ids[:kv_len]
        release_kv_cache(req, self.tree_cache, is_insert=False)
