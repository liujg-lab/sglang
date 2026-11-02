"""KV pool sizing from free memory measured before and after weight load.

The static fraction is a share of device capacity. It is applied to the free
memory measured before weights are loaded. On an empty device that free memory
is close to capacity, and the original formula is kept. When the page-aligned
result is below one page but post-load free memory can still hold one page,
the activation reserve is at least 0.5 GB and NPU graph capture is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# When the static-fraction reserve leaves less than one KV page, keep at least
# this much post-load free memory for activations and the FIA workspace.
KV_FALLBACK_ACTIVATION_RATIO = 0.2
KV_FALLBACK_MIN_HEADROOM_GB = 0.5
# Same 1.5 MB/token heuristic as ServerArgs._handle_gpu_memory_settings.
KV_ACTIVATION_MB_PER_TOKEN = 1.5
KV_FALLBACK_GRAPH_BS_CAP = 8


@dataclass(frozen=True)
class KVTokenBudget:
    """Token-pool size after the static-fraction formula and page alignment."""

    profiled_tokens: int
    aligned_tokens: int
    rest_gb: float
    activation_reserve_gb: float
    used_fallback: bool
    chunked_prefill_size: Optional[int] = None
    cuda_graph_bs: Optional[List[int]] = None
    cuda_graph_max_bs: Optional[int] = None
    disable_cuda_graph: bool = False
    warning: Optional[str] = None


def _align_token_capacity(tokens: int, page_size: int) -> int:
    if page_size <= 1:
        return tokens
    return tokens // page_size * page_size


def _tokens_from_gb(rest_gb: float, cell_size: int) -> int:
    return int(rest_gb * (1 << 30)) // cell_size


def _kv_budget_error(
    *,
    pre_gb: float,
    post_gb: float,
    mem_fraction_static: float,
    rest_gb: float,
    cell_size: int,
    page_size: int,
    aligned_tokens: int,
    total_capacity_gb: Optional[float],
) -> str:
    if total_capacity_gb is None:
        total = "unknown"
    else:
        total = f"{total_capacity_gb:.4f}"
    return (
        "Not enough memory for one KV page. "
        f"mem_fraction_static={mem_fraction_static}, "
        f"pre_free_gb={pre_gb:.4f}, post_free_gb={post_gb:.4f}, "
        f"rest_gb={rest_gb:.4f}, cell_size={cell_size}, page_size={page_size}, "
        f"aligned_tokens={aligned_tokens}, "
        f"total_capacity_gb={total}, available_gb={post_gb:.4f}."
    )


def _fallback_activation_reserve_bytes(
    post_gb: float, extra_reserve_gb: float, page_size: int, cell_size: int
) -> Optional[int]:
    """Bytes to leave after weights. None when one KV page does not fit."""
    post_bytes = int(post_gb * (1 << 30))
    extra_bytes = int(max(extra_reserve_gb, 0.0) * (1 << 30))
    available = post_bytes - extra_bytes
    one_page_bytes = page_size * cell_size
    if available < one_page_bytes:
        return None
    ratio_bytes = int(post_gb * KV_FALLBACK_ACTIVATION_RATIO * (1 << 30))
    headroom_bytes = int(KV_FALLBACK_MIN_HEADROOM_GB * (1 << 30))
    desired_bytes = max(ratio_bytes, headroom_bytes)
    return min(desired_bytes, available - one_page_bytes)


def _shrink_chunked_prefill_for_tight_memory(
    activation_reserve_gb: float, page_size: int, chunked_prefill_size: Optional[int]
) -> int:
    activation_mb = activation_reserve_gb * 1024
    supported = int(activation_mb / KV_ACTIVATION_MB_PER_TOKEN)
    if supported < page_size:
        supported = page_size
    if chunked_prefill_size is None or chunked_prefill_size <= 0:
        return supported
    return max(page_size, min(chunked_prefill_size, supported))


def _shrink_cuda_graph_bs_for_tight_memory(
    cuda_graph_bs: Optional[Sequence[int]],
) -> Tuple[List[int], int]:
    if not cuda_graph_bs:
        return [1], 1
    capped = [int(bs) for bs in cuda_graph_bs if int(bs) <= KV_FALLBACK_GRAPH_BS_CAP]
    if not capped:
        capped = [1]
    return capped, max(capped)


def resolve_kv_token_budget(
    *,
    pre_gb: float,
    post_gb: float,
    mem_fraction_static: float,
    cell_size: int,
    page_size: int,
    chunked_prefill_size: Optional[int] = None,
    cuda_graph_bs: Optional[Sequence[int]] = None,
    total_capacity_gb: Optional[float] = None,
    extra_reserve_gb: float = 0.0,
) -> KVTokenBudget:
    """Return the profiled KV token count and, when needed, tighter runtime limits."""
    if cell_size <= 0:
        raise RuntimeError(f"KV cell size must be positive, got cell_size={cell_size}")
    page = page_size if page_size and page_size > 0 else 1
    rest_gb = post_gb - pre_gb * (1.0 - mem_fraction_static) - extra_reserve_gb
    profiled_tokens = _tokens_from_gb(rest_gb, cell_size)
    aligned_tokens = _align_token_capacity(profiled_tokens, page)
    if aligned_tokens >= page:
        return KVTokenBudget(
            profiled_tokens=profiled_tokens,
            aligned_tokens=aligned_tokens,
            rest_gb=rest_gb,
            activation_reserve_gb=pre_gb * (1.0 - mem_fraction_static)
            + extra_reserve_gb,
            used_fallback=False,
        )

    post_tokens = _tokens_from_gb(post_gb, cell_size)
    if _align_token_capacity(post_tokens, page) < page:
        raise RuntimeError(
            _kv_budget_error(
                pre_gb=pre_gb,
                post_gb=post_gb,
                mem_fraction_static=mem_fraction_static,
                rest_gb=rest_gb,
                cell_size=cell_size,
                page_size=page,
                aligned_tokens=aligned_tokens,
                total_capacity_gb=total_capacity_gb,
            )
        )

    reserve_bytes = _fallback_activation_reserve_bytes(
        post_gb, extra_reserve_gb, page, cell_size
    )
    if reserve_bytes is None:
        raise RuntimeError(
            _kv_budget_error(
                pre_gb=pre_gb,
                post_gb=post_gb,
                mem_fraction_static=mem_fraction_static,
                rest_gb=rest_gb,
                cell_size=cell_size,
                page_size=page,
                aligned_tokens=aligned_tokens,
                total_capacity_gb=total_capacity_gb,
            )
        )
    extra_bytes = int(max(extra_reserve_gb, 0.0) * (1 << 30))
    kv_bytes = int(post_gb * (1 << 30)) - extra_bytes - reserve_bytes
    fallback_tokens = kv_bytes // cell_size
    fallback_aligned = _align_token_capacity(fallback_tokens, page)
    if fallback_aligned < page:
        raise RuntimeError(
            _kv_budget_error(
                pre_gb=pre_gb,
                post_gb=post_gb,
                mem_fraction_static=mem_fraction_static,
                rest_gb=rest_gb,
                cell_size=cell_size,
                page_size=page,
                aligned_tokens=aligned_tokens,
                total_capacity_gb=total_capacity_gb,
            )
        )

    activation_reserve_gb = reserve_bytes / (1 << 30)
    fallback_rest_gb = kv_bytes / (1 << 30)
    new_chunked = _shrink_chunked_prefill_for_tight_memory(
        activation_reserve_gb, page, chunked_prefill_size
    )
    new_graph_bs, new_graph_max_bs = _shrink_cuda_graph_bs_for_tight_memory(
        cuda_graph_bs
    )
    warning = (
        "KV token budget is below one page after "
        f"mem_fraction_static={mem_fraction_static} "
        f"(pre_free_gb={pre_gb:.4f}, post_free_gb={post_gb:.4f}, "
        f"rest_gb={rest_gb:.4f}, cell_size={cell_size}, page_size={page}, "
        f"aligned_tokens={aligned_tokens}). "
        "Free memory is below the device capacity used to size the activation "
        "reserve. "
        f"Keeping {activation_reserve_gb:.4f} GB for activations and the FIA "
        "workspace, and assigning the rest to the KV pool "
        f"(profiled_tokens={fallback_tokens}, aligned_tokens={fallback_aligned}). "
        "Skipping NPU graph capture because the FIA workspace does not fit in "
        "the remaining memory. "
        f"chunked_prefill_size {chunked_prefill_size} -> {new_chunked}, "
        f"cuda_graph_bs capped at {KV_FALLBACK_GRAPH_BS_CAP}: {new_graph_bs}."
    )
    return KVTokenBudget(
        profiled_tokens=fallback_tokens,
        aligned_tokens=fallback_aligned,
        rest_gb=fallback_rest_gb,
        activation_reserve_gb=activation_reserve_gb,
        used_fallback=True,
        chunked_prefill_size=new_chunked,
        cuda_graph_bs=new_graph_bs,
        cuda_graph_max_bs=new_graph_max_bs,
        disable_cuda_graph=True,
        warning=warning,
    )
