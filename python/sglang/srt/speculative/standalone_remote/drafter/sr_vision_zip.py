"""Training-free VisionZip pruning for STANDALONE_REMOTE Draft prompts.

The Draft runs its own ViT, keeps a fraction of the LLM vision tokens, and
enters prefill with precomputed embeddings. Target verification is unchanged.
Any failure leaves the request on the uncompacted path (delta 0).
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.speculative.standalone_remote.sr_prompt_coords import (
    _gather_mrope,
    restore_compacted_mrope,
)

logger = logging.getLogger(__name__)

ALGO_VERSION = 3
MAX_QUERIES = 1024
QUERY_CHUNK = 64
CACHE_MAX = 8

_warned: set = set()
_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_cache_ready: Dict[tuple, torch.cuda.Event] = {}


def validate_draft_vision_config(
    role: Optional[str],
    compress: str,
    keep_ratio: float,
    dominant_ratio: float,
    score_layer: int,
) -> None:
    """Reject Target overrides and ratios outside (0, 1]."""
    if compress not in ("none", "visionzip"):
        raise ValueError(
            "standalone_remote_draft_vision_compress must be none or visionzip, "
            f"got {compress!r}."
        )
    keep = float(keep_ratio)
    dominant = float(dominant_ratio)
    if not 0.0 < keep <= 1.0:
        raise ValueError(
            "standalone_remote_draft_vision_keep_ratio must be in (0, 1], "
            f"got {keep}."
        )
    if not 0.0 < dominant <= 1.0:
        raise ValueError(
            "standalone_remote_draft_vision_dominant_ratio must be in (0, 1], "
            f"got {dominant}."
        )
    nondefault = (
        compress != "none" or keep != 0.25 or dominant != 0.85 or int(score_layer) != -2
    )
    if role != "draft" and nondefault:
        raise ValueError(
            "STANDALONE_REMOTE vision compression flags are draft-only. "
            "Target must leave --standalone-remote-draft-vision-* at the defaults."
        )


def visionzip_radix_salt(
    keep_index: Sequence[int],
    keep_ratio: float,
    dominant_ratio: float,
    score_layer: int,
) -> str:
    """Namespace a compacted prompt so radix cannot reuse another vision KV.

    Image spans are a run of the same pad id. Two keep-sets of different
    lengths share a token-id prefix. The salt is a digest of the selection,
    plus the algorithm version, ratios, and score layer.
    """
    kept = ",".join(str(int(i)) for i in keep_index)
    digest = hashlib.blake2b(kept.encode(), digest_size=16).hexdigest()
    return (
        f"sr-visionzip:v{ALGO_VERSION}:{float(keep_ratio):.8f}:"
        f"{float(dominant_ratio):.8f}:{int(score_layer)}:{digest}"
    )


def append_visionzip_extra_key(
    req,
    keep_index: Sequence[int],
    keep_ratio: float,
    dominant_ratio: float,
    score_layer: int,
) -> str:
    """Append the VisionZip salt. An existing cache salt stays in front."""
    salt = visionzip_radix_salt(keep_index, keep_ratio, dominant_ratio, score_layer)
    prev = getattr(req, "extra_key", None) or ""
    req.extra_key = f"{prev}|{salt}" if prev else salt
    return req.extra_key


def park_precomputed_embeddings(mm) -> None:
    """Move VisionZip rows to CPU after prefill so they do not sit on device.

    Compacted rows already have one host copy on ``sr_parked_rows``. Parking
    swaps to that tensor. Other embeddings still copy here.
    """
    if mm is None:
        return
    from sglang.srt.utils.common import is_pin_memory_available

    pin = is_pin_memory_available()
    for item in getattr(mm, "mm_items", None) or []:
        parked = getattr(item, "sr_parked_rows", None)
        if torch.is_tensor(parked):
            ready = getattr(item, "sr_parked_ready", None)
            if ready is not None:
                ready.synchronize()
            item.precomputed_embeddings = parked
            continue
        emb = getattr(item, "precomputed_embeddings", None)
        if not torch.is_tensor(emb) or emb.device.type == "cpu":
            continue
        cpu = emb.detach().to("cpu")
        if pin and not cpu.is_pinned():
            cpu = cpu.pin_memory()
        item.precomputed_embeddings = cpu


def stage_precomputed_embeddings(mm, device) -> None:
    """Bring parked rows back to the forward device before a re-prefill."""
    if mm is None:
        return
    target = torch.device(device)
    for item in getattr(mm, "mm_items", None) or []:
        emb = getattr(item, "precomputed_embeddings", None)
        if not torch.is_tensor(emb) or emb.device.type != "cpu":
            continue
        if emb.device == target:
            continue
        item.precomputed_embeddings = emb.to(target, non_blocking=emb.is_pinned())


def _warn_once(reason: str) -> None:
    if reason in _warned:
        return
    _warned.add(reason)
    logger.warning("[SR] Draft VisionZip disabled for this request: %s", reason)


def visionzip_compact_rows(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    keep_ratio: float,
    dominant_ratio: float,
    hidden_size: int,
    metric: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """Keep ``ceil(keep_ratio * N)`` rows. Indices are strictly increasing.

    Dominant rows are the highest scores. The remaining budget is a fixed-stride
    sample of context anchors; every other row is averaged into the nearest
    anchor. Cosine similarity uses ``metric`` when it is given (pooled vision
    keys), otherwise the main hidden slice of ``embeddings``. The whole row,
    including deepstack columns, is averaged together.

    ``dominant_ratio == 1`` leaves no anchor budget, so unselected rows are
    dropped instead of merged. ``K >= N`` returns the original rows and
    ``range(N)``.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be rank 2, got {tuple(embeddings.shape)}")
    n = int(embeddings.shape[0])
    if n == 0:
        return embeddings, []
    score = scores.detach().reshape(-1).float()
    if int(score.numel()) != n:
        raise ValueError(f"score length {int(score.numel())} != rows {n}")
    if not (0.0 < float(keep_ratio) <= 1.0):
        raise ValueError(f"keep_ratio {keep_ratio} out of (0, 1]")
    if not (0.0 < float(dominant_ratio) <= 1.0):
        raise ValueError(f"dominant_ratio {dominant_ratio} out of (0, 1]")
    if metric is not None and (
        metric.ndim != 2 or int(metric.shape[0]) != n
    ):
        raise ValueError(
            f"metric rows {tuple(metric.shape)} do not match embeddings {n}"
        )

    k = min(n, int(math.ceil(float(keep_ratio) * n)))
    if k >= n:
        return embeddings, list(range(n))

    k_d = min(k, max(0, int(round(float(dominant_ratio) * k))))
    order = torch.sort(score, descending=True, stable=True).indices
    dominant = [int(i) for i in order[:k_d].tolist()]
    dominant_set = set(dominant)
    remaining = [i for i in range(n) if i not in dominant_set]
    targets = _stride_pick(remaining, k - len(dominant))
    target_set = set(targets)
    keep = sorted(dominant_set | target_set)
    keep_idx = torch.as_tensor(keep, device=embeddings.device, dtype=torch.long)
    rows = embeddings.index_select(0, keep_idx).contiguous()
    if targets:
        averaged = _merge_context(
            embeddings, remaining, targets, hidden_size, metric=metric
        )
        pos = {idx: i for i, idx in enumerate(keep)}
        dest = torch.as_tensor(
            [pos[int(t)] for t in targets], device=rows.device, dtype=torch.long
        )
        rows[dest] = averaged
    return rows, keep


def _stride_pick(remaining: Sequence[int], k: int) -> List[int]:
    if k <= 0 or not remaining:
        return []
    if k >= len(remaining):
        return list(remaining)
    chosen: List[int] = []
    used = set()
    for i in range(k):
        pos = int((i + 0.5) * len(remaining) / k)
        pos = min(max(pos, 0), len(remaining) - 1)
        while pos in used and pos + 1 < len(remaining):
            pos += 1
        if pos in used:
            continue
        used.add(pos)
        chosen.append(int(remaining[pos]))
    return chosen


def _merge_context(
    embeddings: torch.Tensor,
    remaining: Sequence[int],
    targets: Sequence[int],
    hidden_size: int,
    metric: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean each target with the donors whose cosine match is that target.

    Returns one row per target, in ``targets`` order. The mean is the target
    row plus the donors assigned to it, in float, then cast back. ``metric``
    chooses the nearest target; the averaged values always come from
    ``embeddings``.
    """
    target_set = set(int(i) for i in targets)
    donors = [i for i in remaining if i not in target_set]
    width = int(embeddings.shape[-1])
    main = hidden_size if 0 < int(hidden_size) < width else width
    base_src = embeddings[:, :main] if metric is None else metric
    base = torch.nn.functional.normalize(base_src.float(), dim=-1, eps=1e-6)
    t_idx = torch.as_tensor(list(targets), device=embeddings.device, dtype=torch.long)
    pooled_t = embeddings[t_idx].float()
    counts = torch.ones(len(targets), device=embeddings.device)
    if donors:
        d_idx = torch.as_tensor(donors, device=embeddings.device, dtype=torch.long)
        nearest = (base[d_idx] @ base[t_idx].T).argmax(dim=1)
        pooled_t.index_add_(0, nearest, embeddings[d_idx].float())
        counts.index_add_(0, nearest, torch.ones_like(nearest, dtype=counts.dtype))
    return (pooled_t / counts[:, None]).to(dtype=embeddings.dtype)


def rebuild_prompt(
    logical_prompt: Sequence[int],
    spans: Sequence[Tuple[int, int, Sequence[int]]],
) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """Drop vision tokens and record which original prompt positions survive.

    Each span is ``(start, end_inclusive, relative_keep)``. Relative indices are
    into that span. Text outside the spans is kept. Returns the new ids, the
    original indices of those ids, and the rewritten inclusive offsets.
    """
    ids: List[int] = []
    keep: List[int] = []
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    prompt = list(logical_prompt)
    for start, end, relative in spans:
        start = int(start)
        end = int(end)
        if start < cursor or end < start or end >= len(prompt):
            raise ValueError(
                f"vision spans must be increasing and in range, got {(start, end)}"
            )
        for i in range(cursor, start):
            ids.append(int(prompt[i]))
            keep.append(i)
        span_len = end - start + 1
        rel = [int(i) for i in relative]
        if not rel or any(i < 0 or i >= span_len for i in rel):
            raise ValueError(f"keep indices {rel} outside span length {span_len}")
        if rel != sorted(rel) or len(rel) != len(set(rel)):
            raise ValueError("keep indices must be strictly increasing")
        new_start = len(ids)
        pad = int(prompt[start])
        for src in rel:
            ids.append(pad)
            keep.append(start + src)
        offsets.append((new_start, len(ids) - 1))
        cursor = end + 1
    for i in range(cursor, len(prompt)):
        ids.append(int(prompt[i]))
        keep.append(i)
    return ids, keep, offsets


class _ScoreCapture:
    def __init__(self) -> None:
        self.llm_scores: Optional[torch.Tensor] = None
        self.llm_keys: Optional[torch.Tensor] = None


def _layer_index(n_blocks: int, score_layer: int) -> int:
    index = int(score_layer)
    if index < 0:
        index += int(n_blocks)
    if index < 0 or index >= int(n_blocks):
        raise ValueError(f"score layer {score_layer} outside {n_blocks} blocks")
    return index


def _grid_rows(grid) -> List[Tuple[int, int, int]]:
    if grid is None:
        return []
    if not torch.is_tensor(grid):
        grid = torch.as_tensor(grid)
    if grid.ndim == 1:
        grid = grid.view(1, -1)
    rows = []
    for row in grid.detach().cpu().tolist():
        rows.append((int(row[0]), int(row[1]), int(row[2])))
    return rows


def _sample_queries(length: int, device) -> torch.Tensor:
    count = min(MAX_QUERIES, int(length))
    if count <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if count == length:
        return torch.arange(length, dtype=torch.long, device=device)
    idx = torch.linspace(0, length - 1, count, device=device).round().to(torch.long)
    return torch.unique(idx, sorted=True)


def _attention_received(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Mean attention received by each key. q/k are [tokens, heads, dim].

    Query blocks use einsum ``[q, H, d] x [N, H, d] -> [H, q, N]``. Softmax
    stays over keys, matching a per-head matmul.
    """
    tokens, heads, dim = q.shape
    scale = dim ** -0.5
    queries = _sample_queries(tokens, q.device)
    acc = torch.zeros(tokens, dtype=torch.float32, device=q.device)
    if queries.numel() == 0 or heads == 0:
        return acc
    qs = q.index_select(0, queries).float()
    kk = k.float()
    q_count = int(qs.shape[0])
    for q0 in range(0, q_count, QUERY_CHUNK):
        qq = qs[q0 : q0 + QUERY_CHUNK]
        logits = torch.einsum("qhd,nhd->hqn", qq, kk) * scale
        attn = torch.softmax(logits, dim=-1)
        acc += attn.sum(dim=(0, 1))
    acc /= float(q_count * heads)
    return acc


def _pool_merge_keys(k: torch.Tensor, merge_unit: int) -> torch.Tensor:
    """Mean keys inside each merge window, then flatten heads.

    ``k`` is ``[patches, heads, dim]``. Qwen's merge unit is 4, a 2x2 window.
    The result is one vector per LLM vision token, ``[tokens, heads * dim]``.
    """
    patches, heads, dim = k.shape
    unit = int(merge_unit)
    if patches % unit != 0:
        raise ValueError("patch count is not divisible by the merge unit")
    grouped = k.reshape(-1, unit, heads, dim).mean(dim=1)
    return grouped.reshape(grouped.shape[0], heads * dim)


def _llm_scores_from_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    k_content: torch.Tensor,
    grids: Sequence[Tuple[int, int, int]],
    merge_unit: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Score with RoPE q/k. Pool ``k_content`` for merge similarity.

    ``k`` is post-RoPE. ``k_content`` is the same projection before RoPE.
    """
    if int(k_content.shape[0]) != int(q.shape[0]) or int(k.shape[0]) != int(q.shape[0]):
        raise ValueError("q and k lengths disagree")
    score_parts: List[torch.Tensor] = []
    key_parts: List[torch.Tensor] = []
    cursor = 0
    for t, h, w in grids:
        patches = int(t) * int(h) * int(w)
        if patches <= 0 or cursor + patches > q.shape[0]:
            raise ValueError("grid does not match ViT sequence")
        if patches % int(merge_unit) != 0:
            raise ValueError("patch count is not divisible by the merge unit")
        q_slice = q[cursor : cursor + patches]
        k_slice = k[cursor : cursor + patches]
        content_slice = k_content[cursor : cursor + patches]
        received = _attention_received(q_slice, k_slice)
        score_parts.append(received.view(-1, int(merge_unit)).sum(dim=-1))
        key_parts.append(_pool_merge_keys(content_slice, merge_unit))
        cursor += patches
    if cursor != q.shape[0]:
        raise ValueError("ViT sequence has tokens outside image grids")
    if not score_parts:
        return (
            torch.zeros(0, dtype=torch.float32, device=q.device),
            torch.zeros(0, 0, device=q.device),
        )
    return torch.cat(score_parts, dim=0), torch.cat(key_parts, dim=0)


def _project_qk(block, x, cos, sin):
    from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb

    attn = block.attn
    if not getattr(attn, "use_qkv_parallel", False):
        raise RuntimeError("VisionZip scoring expects QKV-parallel vision attention")
    hidden = block.norm1(x)
    if hidden.dim() != 3:
        raise RuntimeError(f"vision block input rank {hidden.dim()}")
    hidden = hidden.permute(1, 0, 2).contiguous()
    bsz, seqlen, _ = hidden.shape
    qkv, _ = attn.qkv_proj(hidden)
    q, k, _v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    head = attn.num_attention_heads_per_partition
    q = q.reshape(bsz * seqlen, head, -1).contiguous()
    k = k.reshape(bsz * seqlen, head, -1).contiguous()
    if getattr(attn, "qk_normalization_by_head_size", False):
        q, k = attn._apply_qk_norm_head_size(q, k)
    k_content = k
    if cos is not None and sin is not None:
        k_content = k.clone()
        q = q.view(-1, head, attn.head_size)
        k = k.view(-1, head, attn.head_size)
        if cos.size(-1) * 2 == attn.head_size:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = q.view(-1, head, attn.head_size)
        k = k.view(-1, head, attn.head_size)
    return q, k, k_content


def _install_score_hook(block, capture: _ScoreCapture, grids, merge_unit: int):
    def hook(module, args, kwargs):
        if capture.llm_scores is not None:
            return
        x = args[0] if args else kwargs.get("x")
        cos = kwargs.get("rotary_pos_emb_cos")
        sin = kwargs.get("rotary_pos_emb_sin")
        with torch.no_grad():
            q, k_rope, k_content = _project_qk(module, x, cos, sin)
            scores, keys = _llm_scores_from_qk(
                q, k_rope, k_content, grids, merge_unit
            )
            capture.llm_scores = scores
            capture.llm_keys = keys

    return block.register_forward_pre_hook(hook, with_kwargs=True)


def _cache_get(key):
    value = _cache.get(key)
    if value is None:
        return None
    _cache.move_to_end(key)
    ready = _cache_ready.get(key)
    if ready is not None:
        ready.synchronize()
    return value


def _cache_put(key, value, ready=None) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    if ready is None:
        _cache_ready.pop(key, None)
    else:
        _cache_ready[key] = ready
    while len(_cache) > CACHE_MAX:
        old, _ = _cache.popitem(last=False)
        _cache_ready.pop(old, None)


def _host_rows(rows: torch.Tensor):
    """Return one CPU copy of compacted rows and an optional ready event.

    A CPU source is detached and shared. A device source is copied once into
    pinned memory when CUDA can pin; the event completes that copy.
    """
    if rows.device.type == "cpu":
        return rows.detach(), None
    from sglang.srt.utils.common import is_pin_memory_available

    pin = bool(rows.is_cuda and is_pin_memory_available())
    cpu_rows = torch.empty(
        rows.shape, dtype=rows.dtype, device="cpu", pin_memory=pin
    )
    cpu_rows.copy_(rows.detach(), non_blocking=pin)
    ready = None
    if pin:
        ready = torch.cuda.Event()
        ready.record()
    return cpu_rows, ready


def _span_lengths(offsets) -> List[int]:
    lengths = []
    for start, end in offsets or []:
        lengths.append(int(end) - int(start) + 1)
    return lengths


def _cache_key(item, keep_ratio, dominant_ratio, layer) -> Optional[tuple]:
    item_hash = getattr(item, "hash", None)
    if item_hash is None:
        return None
    return (
        int(item_hash),
        float(keep_ratio),
        float(dominant_ratio),
        int(layer),
        ALGO_VERSION,
    )


def _split_rows(rows: torch.Tensor, counts: Sequence[int]) -> List[torch.Tensor]:
    parts = []
    cursor = 0
    for count in counts:
        count = int(count)
        parts.append(rows[cursor : cursor + count])
        cursor += count
    if cursor != int(rows.shape[0]):
        raise ValueError("embedding rows do not match image token counts")
    return parts


def _compact_item_rows(
    rows: torch.Tensor,
    counts: Sequence[int],
    scores: torch.Tensor,
    keep_ratio: float,
    dominant_ratio: float,
    hidden_size: int,
    metric: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[List[int]]]:
    pieces = _split_rows(rows, counts)
    score_parts = _split_rows(scores.reshape(-1, 1), counts)
    metric_parts = None if metric is None else _split_rows(metric, counts)
    kept_rows = []
    keeps: List[List[int]] = []
    for i, (piece, score) in enumerate(zip(pieces, score_parts)):
        compact, index = visionzip_compact_rows(
            piece,
            score.reshape(-1),
            keep_ratio,
            dominant_ratio,
            hidden_size,
            metric=None if metric_parts is None else metric_parts[i],
        )
        kept_rows.append(compact)
        keeps.append(index)
    return torch.cat(kept_rows, dim=0), keeps


def _encode_images(model, items, score_layer: int, merge_unit: int):
    visual = model.visual
    grids = []
    for item in items:
        grids.extend(_grid_rows(getattr(item, "image_grid_thw", None)))
    capture = _ScoreCapture()
    layer = _layer_index(len(visual.blocks), score_layer)
    handle = _install_score_hook(visual.blocks[layer], capture, grids, merge_unit)
    try:
        with torch.no_grad():
            embeddings = model.get_image_feature(items)
    finally:
        handle.remove()
    if capture.llm_scores is None or capture.llm_keys is None:
        raise RuntimeError("score hook did not run")
    if int(embeddings.shape[0]) != int(capture.llm_scores.shape[0]):
        raise RuntimeError(
            f"score rows {tuple(capture.llm_scores.shape)} != "
            f"embeddings {tuple(embeddings.shape)}"
        )
    if int(embeddings.shape[0]) != int(capture.llm_keys.shape[0]):
        raise RuntimeError(
            f"key rows {tuple(capture.llm_keys.shape)} != "
            f"embeddings {tuple(embeddings.shape)}"
        )
    return embeddings, capture.llm_scores, capture.llm_keys


def _skip_reason(scheduler, mm) -> Optional[str]:
    from sglang.srt.environ import envs

    args = getattr(scheduler, "server_args", None)
    if getattr(args, "standalone_remote_draft_vision_compress", "none") != "visionzip":
        return "compress is not visionzip"
    if int(getattr(args, "tp_size", 1) or 1) != 1:
        return "draft tp_size is not 1"
    if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
        return "ViT CUDA graph is enabled"
    items = list(getattr(mm, "mm_items", None) or [])
    if any(getattr(item, "is_video", lambda: False)() for item in items):
        return "request contains video"
    images = [item for item in items if getattr(item, "is_image", lambda: False)()]
    if not images:
        return "request has no image"
    if any(
        not getattr(item, "is_image", lambda: False)() and getattr(item, "offsets", None)
        for item in items
    ):
        return "request has non-image multimodal offsets"
    runner = getattr(getattr(scheduler, "tp_worker", None), "model_runner", None)
    model = getattr(runner, "model", None)
    if model is None or not hasattr(model, "get_image_feature"):
        return "draft model has no get_image_feature"
    visual = getattr(model, "visual", None)
    if visual is None or not getattr(visual, "blocks", None):
        return "draft model has no vision blocks"
    if getattr(visual, "use_data_parallel", False) or getattr(model, "use_data_parallel", False):
        return "visual data parallel is enabled"
    return None


def _draft_vision_ratios(scheduler) -> Tuple[float, float, int]:
    args = getattr(scheduler, "server_args", None)
    keep_ratio = float(
        getattr(args, "standalone_remote_draft_vision_keep_ratio", 0.25)
    )
    dominant_ratio = float(
        getattr(args, "standalone_remote_draft_vision_dominant_ratio", 0.85)
    )
    score_layer = int(getattr(args, "standalone_remote_draft_vision_score_layer", -2))
    return keep_ratio, dominant_ratio, score_layer


def _slice_encoded_items(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    keys: torch.Tensor,
    items,
    merge_unit: int,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Split a batched ViT result back into one slice per image item."""
    encoded = {}
    offset = 0
    for item in items:
        n = sum(_item_token_counts(item, merge_unit))
        encoded[id(item)] = (
            embeddings[offset : offset + n],
            scores[offset : offset + n],
            keys[offset : offset + n],
        )
        offset += n
    if (
        offset != int(embeddings.shape[0])
        or offset != int(scores.shape[0])
        or offset != int(keys.shape[0])
    ):
        raise RuntimeError("encoded rows do not match image items")
    return encoded


def _vision_patch_budget(scheduler, merge_unit: int) -> int:
    """Cap one ViT call at ``max_prefill_tokens * merge_unit`` patches."""
    limit = getattr(scheduler, "max_prefill_tokens", None)
    if limit is None:
        args = getattr(scheduler, "server_args", None)
        limit = getattr(args, "max_prefill_tokens", None)
    if limit is None or int(limit) <= 0:
        limit = 16384
    return max(1, int(limit)) * int(merge_unit)


def _item_patch_count(item, merge_unit: int) -> int:
    return sum(_item_token_counts(item, merge_unit)) * int(merge_unit)


def _chunk_by_patches(items, merge_unit: int, patch_budget: int):
    """Group items so each chunk stays within ``patch_budget`` patches.

    An item larger than the budget is its own chunk. One image is never split
    across two ``get_image_feature`` calls.
    """
    chunks = []
    current = []
    current_patches = 0
    budget = max(1, int(patch_budget))
    for item in items:
        patches = _item_patch_count(item, merge_unit)
        if current and current_patches + patches > budget:
            chunks.append(current)
            current = []
            current_patches = 0
        current.append(item)
        current_patches += patches
    if current:
        chunks.append(current)
    return chunks


def _encode_items_chunked(
    model, items, score_layer: int, merge_unit: int, patch_budget: int
):
    """Encode ``items`` in patch-bounded chunks and clone each slice.

    A dim-0 slice shares storage with the chunk tensor. Cloning lets that
    tensor be freed before the next chunk runs.
    """
    encoded = {}
    for chunk in _chunk_by_patches(items, merge_unit, patch_budget):
        embeddings, scores, keys = _encode_images(
            model, chunk, score_layer, merge_unit
        )
        sliced = _slice_encoded_items(
            embeddings, scores, keys, chunk, merge_unit
        )
        for item_id, piece in sliced.items():
            encoded[item_id] = tuple(part.clone() for part in piece)
    return encoded


def _compact_block_reason(scheduler, req) -> Optional[str]:
    mm = getattr(req, "multimodal_inputs", None)
    if mm is None:
        return "request has no multimodal inputs"
    reason = _skip_reason(scheduler, mm)
    if reason is not None:
        return reason
    full = getattr(mm, "mrope_positions", None)
    if full is None or not torch.is_tensor(full) or full.ndim != 2:
        return "mrope positions are missing"
    if int(full.shape[1]) != len(getattr(req, "origin_input_ids", None) or []):
        return "mrope width does not match the logical prompt"
    return None


def preencode_visionzip_misses(scheduler, jobs):
    """Encode one PREFILL batch's cache misses in patch-bounded ViT chunks.

    ``jobs`` is ``[(req, dreq), ...]``. Requests that will not compact are left
    out. Cache hits are snapshotted into ``hits`` before any later ``_cache_put``
    in this batch can evict them. Misses that share a cache key and token shape
    are encoded once; items with no hash stay unique. Each ViT call stays within
    ``max_prefill_tokens * merge_unit`` patches. The return value is
    ``{"hits": {id(item): (rows, keeps)}, "encoded": {id(item): (embeddings, scores, keys)}}``.
    A raised error tells the caller to encode each request on its own.
    """
    empty = {"hits": {}, "encoded": {}}
    ready = [
        req
        for req, _dreq in jobs
        if _compact_block_reason(scheduler, req) is None
    ]
    if not ready:
        return empty
    keep_ratio, dominant_ratio, score_layer = _draft_vision_ratios(scheduler)
    model = scheduler.tp_worker.model_runner.model
    merge_unit = int(getattr(model.visual, "spatial_merge_unit", 4) or 4)
    hits = {}
    groups = {}
    for req in ready:
        for item in req.multimodal_inputs.mm_items:
            if not item.is_image():
                continue
            counts = tuple(_item_token_counts(item, merge_unit))
            key = _cache_key(item, keep_ratio, dominant_ratio, score_layer)
            cached = None if key is None else _cache_get(key)
            if cached is not None:
                rows, keeps = cached
                hits[id(item)] = (rows, [list(part) for part in keeps])
                continue
            dedup = ("id", id(item)) if key is None else (key, counts)
            bucket = groups.get(dedup)
            if bucket is None:
                groups[dedup] = [item]
            else:
                bucket.append(item)
    encoded = {}
    if groups:
        reps = [bucket[0] for bucket in groups.values()]
        encoded = _encode_items_chunked(
            model,
            reps,
            score_layer,
            merge_unit,
            _vision_patch_budget(scheduler, merge_unit),
        )
        for bucket in groups.values():
            piece = encoded[id(bucket[0])]
            for item in bucket[1:]:
                encoded[id(item)] = piece
    return {"hits": hits, "encoded": encoded}


def maybe_compact_draft_prompt(scheduler, req, dreq, preencoded=None) -> bool:
    """Shrink this Draft prompt. False leaves the request unchanged.

    ``preencoded`` is ``{"hits", "encoded"}`` from the shared ViT encode.
    ``hits`` maps ``id(item)`` to a cached compaction, and ``encoded`` maps
    ``id(item)`` to rows for cache misses. ``None`` encodes this request's
    own cache misses.
    """
    mm = getattr(req, "multimodal_inputs", None)
    if mm is None:
        return False
    reason = _compact_block_reason(scheduler, req)
    if reason is not None:
        if reason != "compress is not visionzip":
            _warn_once(reason)
        return False
    full = mm.mrope_positions
    snapshot = _snapshot_request(req, mm)
    try:
        return _apply_visionzip(
            scheduler, req, dreq, mm, full, preencoded=preencoded
        )
    except Exception as exc:
        _restore_request(req, mm, snapshot)
        _warn_once(f"{type(exc).__name__}: {exc}")
        return False


def _snapshot_request(req, mm):
    items = []
    for item in getattr(mm, "mm_items", None) or []:
        offsets = getattr(item, "offsets", None)
        items.append(
            (
                item,
                getattr(item, "precomputed_embeddings", None),
                getattr(item, "format", None),
                getattr(item, "feature", None),
                None if offsets is None else list(offsets),
                getattr(item, "sr_parked_rows", None),
                getattr(item, "sr_parked_ready", None),
            )
        )
    return {
        "sr_logical_mrope": getattr(req, "sr_logical_mrope", None),
        "sr_prompt_keep_index": getattr(req, "sr_prompt_keep_index", None),
        "sr_local_prompt_ids": getattr(req, "sr_local_prompt_ids", None),
        "sr_prompt_delta": getattr(req, "sr_prompt_delta", 0),
        "origin_input_ids": getattr(req, "origin_input_ids", None),
        "origin_input_ids_unpadded": getattr(req, "origin_input_ids_unpadded", None),
        "fill_ids": getattr(req, "fill_ids", None),
        "extend_input_len": getattr(req, "extend_input_len", None),
        "logprob_start_len": getattr(req, "logprob_start_len", None),
        "extra_key": getattr(req, "extra_key", None),
        "mrope_positions": getattr(mm, "mrope_positions", None),
        "mrope_position_delta": getattr(mm, "mrope_position_delta", None),
        "mrope_position_delta_repeated_cache": getattr(
            mm, "mrope_position_delta_repeated_cache", None
        ),
        "items": items,
    }


def _restore_request(req, mm, snapshot) -> None:
    for name in (
        "sr_logical_mrope",
        "sr_prompt_keep_index",
        "sr_local_prompt_ids",
        "sr_prompt_delta",
        "origin_input_ids",
        "origin_input_ids_unpadded",
        "fill_ids",
        "extend_input_len",
        "logprob_start_len",
        "extra_key",
    ):
        setattr(req, name, snapshot[name])
    mm.mrope_positions = snapshot["mrope_positions"]
    mm.mrope_position_delta = snapshot["mrope_position_delta"]
    mm.mrope_position_delta_repeated_cache = snapshot[
        "mrope_position_delta_repeated_cache"
    ]
    for (
        item,
        embeddings,
        fmt,
        feature,
        offsets,
        parked_rows,
        parked_ready,
    ) in snapshot["items"]:
        item.precomputed_embeddings = embeddings
        item.format = fmt
        item.feature = feature
        item.offsets = offsets
        item.sr_parked_rows = parked_rows
        item.sr_parked_ready = parked_ready


def _apply_visionzip(
    scheduler, req, dreq, mm, full_mrope: torch.Tensor, preencoded=None
) -> bool:
    keep_ratio, dominant_ratio, score_layer = _draft_vision_ratios(scheduler)
    model = scheduler.tp_worker.model_runner.model
    hidden = int(getattr(model.config, "hidden_size", 0) or 0)
    merge_unit = int(getattr(model.visual, "spatial_merge_unit", 4) or 4)
    images = [item for item in mm.mm_items if item.is_image()]
    pre_hits = {} if preencoded is None else preencoded.get("hits", {})
    pre_enc = {} if preencoded is None else preencoded.get("encoded", {})

    prepared = []
    misses = []
    for item in images:
        counts = _item_token_counts(item, merge_unit)
        key = _cache_key(item, keep_ratio, dominant_ratio, score_layer)
        cached = pre_hits.get(id(item))
        if cached is None and key is not None:
            cached = _cache_get(key)
        if cached is None and id(item) not in pre_enc:
            misses.append(item)
        prepared.append((item, counts, key, cached))

    encoded = dict(pre_enc)
    if misses:
        encoded.update(
            _encode_items_chunked(
                model,
                misses,
                score_layer,
                merge_unit,
                _vision_patch_budget(scheduler, merge_unit),
            )
        )

    visual_param = next(model.visual.parameters())
    plans = []
    parked_host = {}
    changed = False
    for item, counts, key, cached in prepared:
        if cached is not None:
            cpu_rows, keeps = cached
            rows = cpu_rows.to(
                device=visual_param.device,
                dtype=visual_param.dtype,
                non_blocking=cpu_rows.is_pinned(),
            )
            ready = None
        else:
            rows, score, metric = encoded[id(item)]
            rows, keeps = _compact_item_rows(
                rows,
                counts,
                score,
                keep_ratio,
                dominant_ratio,
                hidden,
                metric=metric,
            )
            cpu_rows, ready = _host_rows(rows)
            if key is not None:
                _cache_put(
                    key,
                    (cpu_rows, [list(part) for part in keeps]),
                    ready=ready,
                )
        if sum(len(part) for part in keeps) != sum(counts):
            changed = True
        parked_host[id(item)] = (cpu_rows, ready)
        plans.append((item, counts, rows, keeps))

    if not changed:
        return False

    logical_prompt = list(req.sr_padded_ids)
    spans = []
    for _item, _counts, _rows, keeps in plans:
        offsets = list(_item.offsets or [])
        if len(offsets) != len(keeps):
            raise RuntimeError("image offsets and grids disagree")
        for (start, end), rel in zip(offsets, keeps):
            spans.append((int(start), int(end), rel))
    _validate_spans(logical_prompt, spans, plans)
    new_prompt, keep_index, new_offsets = rebuild_prompt(logical_prompt, spans)
    if len(new_prompt) >= len(logical_prompt):
        return False
    committed = list(getattr(dreq, "committed_ids", None) or [])
    preview = _gather_mrope(
        full_mrope.detach().cpu(),
        keep_index,
        len(logical_prompt),
        len(committed),
    )
    if preview.shape[1] != len(new_prompt) + len(committed):
        raise RuntimeError("gathered mrope width does not match the local prompt")

    req.sr_logical_mrope = full_mrope.detach().cpu().contiguous()
    req.sr_prompt_keep_index = keep_index
    req.sr_local_prompt_ids = list(new_prompt)
    req.sr_prompt_delta = len(logical_prompt) - len(new_prompt)
    append_visionzip_extra_key(
        req, keep_index, keep_ratio, dominant_ratio, score_layer
    )
    req.origin_input_ids = list(new_prompt) + committed
    req.origin_input_ids_unpadded = list(req.origin_input_ids)
    req.fill_ids = list(req.origin_input_ids)
    req.extend_input_len = len(req.fill_ids)
    req.logprob_start_len = max(0, len(req.origin_input_ids) - 1)

    from sglang.srt.managers.schedule_batch import MultimodalInputFormat

    offset_cursor = 0
    for item, _counts, rows, keeps in plans:
        n_spans = len(keeps)
        cpu_rows, ready = parked_host[id(item)]
        item.precomputed_embeddings = rows.detach()
        item.sr_parked_rows = cpu_rows
        item.sr_parked_ready = ready
        item.format = MultimodalInputFormat.PRECOMPUTED_EMBEDDING
        item.feature = None
        item.offsets = new_offsets[offset_cursor : offset_cursor + n_spans]
        offset_cursor += n_spans
    if not restore_compacted_mrope(req):
        raise RuntimeError("gathered mrope does not match the local prompt")
    return True


def _validate_spans(logical_prompt, spans, plans) -> None:
    expected = []
    for item, counts, rows, keeps in plans:
        offsets = list(item.offsets or [])
        if len(offsets) != len(counts) or len(offsets) != len(keeps):
            raise RuntimeError(
                f"image item has {len(offsets)} offsets, {len(counts)} grids, "
                f"{len(keeps)} keep lists"
            )
        for (start, end), count, rel in zip(offsets, counts, keeps):
            span_len = int(end) - int(start) + 1
            if span_len != int(count):
                raise RuntimeError(
                    f"offset length {span_len} != grid tokens {count}"
                )
            if len(rel) > span_len:
                raise RuntimeError("kept more tokens than the span")
            expected.append((int(start), int(end), rel))
        if int(rows.shape[0]) != sum(len(rel) for rel in keeps):
            raise RuntimeError("compacted rows do not match keep indices")
    if [(a, b, list(c)) for a, b, c in spans] != [
        (a, b, list(c)) for a, b, c in expected
    ]:
        raise RuntimeError("span plan does not match image items")
    if not logical_prompt:
        raise RuntimeError("logical prompt is empty")


def _item_token_counts(item, merge_unit: int) -> List[int]:
    counts = []
    for t, h, w in _grid_rows(getattr(item, "image_grid_thw", None)):
        patches = t * h * w
        if patches % merge_unit != 0:
            raise RuntimeError("patch count is not divisible by the merge unit")
        counts.append(patches // merge_unit)
    lengths = _span_lengths(getattr(item, "offsets", None))
    if counts != lengths:
        raise RuntimeError(f"grid tokens {counts} != offset lengths {lengths}")
    if not counts:
        raise RuntimeError("image item has no grid")
    return counts
