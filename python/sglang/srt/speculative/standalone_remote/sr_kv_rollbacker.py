import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.common import release_kv_cache

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
        if self.page_size > 1:
            return False
        prefix_len = self.get_prefix_len(req)
        return fork_point >= prefix_len

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
        if self.page_size > 1 or req.req_pool_idx is None:
            return False

        allocated = int(getattr(req, "kv_allocated_len", 0) or 0)
        end = allocated if allocated > 0 else int(current_kv_len)
        if fork_point >= end:
            req.kv_committed_len = min(int(req.kv_committed_len or 0), fork_point, end)
            req.kv_allocated_len = min(end, fork_point)
            return False

        prefix_len = self.get_prefix_len(req)
        if fork_point < prefix_len:
            return False

        try:
            max_len = self.req_to_token_pool.req_to_token.shape[1]
            start = min(fork_point, max_len)
            end = min(end, max_len)
            if start >= end:
                return False
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start:end
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            req.kv_committed_len = fork_point
            req.kv_allocated_len = fork_point
            return True
        except Exception as e:
            if self.tp_rank == 0:
                logger.warning("[SR] local_rollback failed for %s: %s", req.rid, e)
            return False

    def release_all_kv_for_finished_req(self, req: "Req") -> None:
        if req.req_pool_idx is None:
            return
        kv_len = req.kv_committed_len
        req.fill_ids = (req.origin_input_ids + req.output_ids)[:kv_len]
        release_kv_cache(req, self.tree_cache, is_insert=False)
