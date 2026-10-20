"""Training-free VisionZip pruning for STANDALONE_REMOTE Draft prompts.

The Draft runs its own ViT, keeps a fraction of the LLM vision tokens, and
enters prefill with precomputed embeddings. Target verification is unchanged.
Any failure leaves the request on the uncompacted path (delta 0).
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import torch

from sglang.srt.speculative.standalone_remote.sr_prompt_coords import (
    _gather_mrope,
    restore_compacted_mrope,
)

logger = logging.getLogger(__name__)

ALGO_VERSION = 1
MAX_QUERIES = 1024
QUERY_CHUNK = 64
CACHE_MAX = 8

_warned: set = set()
_cache: "OrderedDict[tuple, tuple]" = OrderedDict()


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
    lengths share a token-id prefix. The salt is the selection itself.
    """
    kept = ",".join(str(int(i)) for i in keep_index)
    return (
        f"sr-visionzip:v{ALGO_VERSION}:{float(keep_ratio):.8f}:"
        f"{float(dominant_ratio):.8f}:{int(score_layer)}:{kept}"
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
) -> Tuple[torch.Tensor, List[int]]:
    """Keep ``ceil(keep_ratio * N)`` rows. Indices are strictly increasing.

    Dominant rows are the highest scores. The remaining budget is a fixed-stride
    sample of context anchors; every other row is averaged into the nearest
    anchor by cosine similarity of the main hidden slice. The whole row,
    including deepstack columns, is averaged together.

    ``K >= N`` returns the original rows and ``range(N)``.
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
    pooled = embeddings.clone()
    if targets:
        _merge_context(pooled, embeddings, remaining, targets, hidden_size)
    keep = sorted(dominant_set | target_set)
    rows = pooled[keep]
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
    pooled: torch.Tensor,
    embeddings: torch.Tensor,
    remaining: Sequence[int],
    targets: Sequence[int],
    hidden_size: int,
) -> None:
    target_set = set(int(i) for i in targets)
    donors = [i for i in remaining if i not in target_set]
    width = int(embeddings.shape[-1])
    main = hidden_size if 0 < int(hidden_size) < width else width
    base = embeddings[:, :main].float()
    base = torch.nn.functional.normalize(base, dim=-1, eps=1e-6)
    groups = {int(t): [int(t)] for t in targets}
    if donors:
        sim = base[donors] @ base[list(targets)].transpose(0, 1)
        nearest = sim.argmax(dim=1).tolist()
        for donor, slot in zip(donors, nearest):
            groups[int(targets[int(slot)])].append(int(donor))
    for target, members in groups.items():
        pooled[target] = embeddings[members].float().mean(dim=0).to(dtype=embeddings.dtype)


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
    """Mean attention received by each key. q/k are [tokens, heads, dim]."""
    tokens, heads, dim = q.shape
    scale = dim ** -0.5
    queries = _sample_queries(tokens, q.device)
    acc = torch.zeros(tokens, dtype=torch.float32, device=q.device)
    if queries.numel() == 0 or heads == 0:
        return acc
    qs = q.index_select(0, queries).float()
    kk = k.float()
    q_count = int(qs.shape[0])
    for head in range(heads):
        key = kk[:, head, :]
        for q0 in range(0, q_count, QUERY_CHUNK):
            qq = qs[q0 : q0 + QUERY_CHUNK, head, :]
            logits = torch.matmul(qq, key.transpose(0, 1)) * scale
            attn = torch.softmax(logits, dim=-1)
            acc += attn.sum(dim=0)
    acc /= float(q_count * heads)
    return acc


def _llm_scores_from_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    grids: Sequence[Tuple[int, int, int]],
    merge_unit: int,
) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    cursor = 0
    for t, h, w in grids:
        patches = int(t) * int(h) * int(w)
        if patches <= 0 or cursor + patches > q.shape[0]:
            raise ValueError("grid does not match ViT sequence")
        if patches % int(merge_unit) != 0:
            raise ValueError("patch count is not divisible by the merge unit")
        received = _attention_received(q[cursor : cursor + patches], k[cursor : cursor + patches])
        parts.append(received.view(-1, int(merge_unit)).sum(dim=-1))
        cursor += patches
    if cursor != q.shape[0]:
        raise ValueError("ViT sequence has tokens outside image grids")
    if not parts:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(parts, dim=0)


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
    if cos is not None and sin is not None:
        q = q.view(-1, head, attn.head_size)
        k = k.view(-1, head, attn.head_size)
        if cos.size(-1) * 2 == attn.head_size:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = q.view(-1, head, attn.head_size)
        k = k.view(-1, head, attn.head_size)
    return q, k


def _install_score_hook(block, capture: _ScoreCapture, grids, merge_unit: int):
    def hook(module, args, kwargs):
        if capture.llm_scores is not None:
            return
        x = args[0] if args else kwargs.get("x")
        cos = kwargs.get("rotary_pos_emb_cos")
        sin = kwargs.get("rotary_pos_emb_sin")
        with torch.no_grad():
            q, k = _project_qk(module, x, cos, sin)
            capture.llm_scores = _llm_scores_from_qk(q, k, grids, merge_unit)

    return block.register_forward_pre_hook(hook, with_kwargs=True)


def _cache_get(key):
    value = _cache.get(key)
    if value is None:
        return None
    _cache.move_to_end(key)
    return value


def _cache_put(key, value) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)


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
) -> Tuple[torch.Tensor, List[List[int]]]:
    pieces = _split_rows(rows, counts)
    score_parts = _split_rows(scores.reshape(-1, 1), counts)
    kept_rows = []
    keeps: List[List[int]] = []
    for piece, score in zip(pieces, score_parts):
        compact, index = visionzip_compact_rows(
            piece,
            score.reshape(-1),
            keep_ratio,
            dominant_ratio,
            hidden_size,
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
    if capture.llm_scores is None:
        raise RuntimeError("score hook did not run")
    if int(embeddings.shape[0]) != int(capture.llm_scores.shape[0]):
        raise RuntimeError(
            f"score rows {tuple(capture.llm_scores.shape)} != "
            f"embeddings {tuple(embeddings.shape)}"
        )
    return embeddings, capture.llm_scores


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


def maybe_compact_draft_prompt(scheduler, req, dreq) -> bool:
    """Shrink this Draft prompt. False leaves the request unchanged."""
    mm = getattr(req, "multimodal_inputs", None)
    if mm is None:
        return False
    reason = _skip_reason(scheduler, mm)
    if reason is not None:
        if reason != "compress is not visionzip":
            _warn_once(reason)
        return False
    full = getattr(mm, "mrope_positions", None)
    if full is None or not torch.is_tensor(full) or full.ndim != 2:
        _warn_once("mrope positions are missing")
        return False
    if int(full.shape[1]) != len(getattr(req, "origin_input_ids", None) or []):
        _warn_once("mrope width does not match the logical prompt")
        return False
    snapshot = _snapshot_request(req, mm)
    try:
        return _apply_visionzip(scheduler, req, dreq, mm, full)
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
    for item, embeddings, fmt, feature, offsets in snapshot["items"]:
        item.precomputed_embeddings = embeddings
        item.format = fmt
        item.feature = feature
        item.offsets = offsets


def _apply_visionzip(scheduler, req, dreq, mm, full_mrope: torch.Tensor) -> bool:
    args = scheduler.server_args
    keep_ratio = getattr(dreq, "draft_vision_keep_ratio", None)
    if keep_ratio is None:
        keep_ratio = getattr(args, "standalone_remote_draft_vision_keep_ratio", 0.25)
    dominant_ratio = float(
        getattr(args, "standalone_remote_draft_vision_dominant_ratio", 0.85)
    )
    score_layer = int(getattr(args, "standalone_remote_draft_vision_score_layer", -2))
    keep_ratio = float(keep_ratio)
    model = scheduler.tp_worker.model_runner.model
    hidden = int(getattr(model.config, "hidden_size", 0) or 0)
    merge_unit = int(getattr(model.visual, "spatial_merge_unit", 4) or 4)
    images = [item for item in mm.mm_items if item.is_image()]

    prepared = []
    misses = []
    for item in images:
        counts = _item_token_counts(item, merge_unit)
        key = _cache_key(item, keep_ratio, dominant_ratio, score_layer)
        cached = None if key is None else _cache_get(key)
        if cached is None:
            misses.append(item)
        prepared.append((item, counts, key, cached))

    encoded = {}
    if misses:
        embeddings, scores = _encode_images(model, misses, score_layer, merge_unit)
        offset = 0
        for item in misses:
            counts = _item_token_counts(item, merge_unit)
            n = sum(counts)
            encoded[id(item)] = (
                embeddings[offset : offset + n],
                scores[offset : offset + n],
            )
            offset += n
        if offset != int(embeddings.shape[0]):
            raise RuntimeError("encoded rows do not match image items")

    visual_param = next(model.visual.parameters())
    plans = []
    changed = False
    for item, counts, key, cached in prepared:
        if cached is not None:
            rows, keeps = cached
            rows = rows.to(device=visual_param.device, dtype=visual_param.dtype)
        else:
            rows, score = encoded[id(item)]
            rows, keeps = _compact_item_rows(
                rows, counts, score, keep_ratio, dominant_ratio, hidden
            )
            if key is not None:
                _cache_put(
                    key, (rows.detach().cpu(), [list(part) for part in keeps])
                )
        if sum(len(part) for part in keeps) != sum(counts):
            changed = True
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
        item.precomputed_embeddings = rows.detach()
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
