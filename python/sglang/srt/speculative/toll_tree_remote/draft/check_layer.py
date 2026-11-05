"""Optional row-0 check against an ordinary extend. Read only inside this package."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def check_enabled() -> bool:
    return os.environ.get("SGLANG_TOLL_CHECK_LAYER") == "1"


def ancestor_tokens(tree, node: int) -> list[int]:
    chain = []
    cursor = int(node)
    while cursor != -1:
        chain.append(int(tree.token[cursor]))
        cursor = int(tree.parent[cursor])
    chain.reverse()
    return chain


def reference_and_report(scheduler, req, tree, node: int, layer_logits) -> None:
    """EXTEND the ancestor path and log how far row 0 is from that distribution."""
    path = ancestor_tokens(tree, node)
    if not path:
        return
    slots = list(scheduler._toll_alloc(len(path)) or [])
    if len(slots) < len(path):
        if slots:
            scheduler._toll_free(slots)
        logger.info("[TOLL check] skipped, prefix extend has no KV slots")
        return
    saved_committed = int(getattr(req, "kv_committed_len", 0) or 0)
    saved_allocated = int(getattr(req, "kv_allocated_len", 0) or 0)
    try:
        from sglang.srt.speculative.toll_tree_remote.draft.reqs import extend_committed

        logits = extend_committed(scheduler, req, saved_committed, path, slots)
    except Exception as exc:
        from sglang.srt.speculative.toll_tree_remote.draft.scheduler_mixin import (
            is_device_context_error,
        )

        if is_device_context_error(exc):
            raise
        logger.info("[TOLL check] reference extend failed: %s", exc)
        return
    finally:
        req.kv_committed_len = saved_committed
        req.kv_allocated_len = saved_allocated
        scheduler._toll_free(slots)
    if logits is None:
        return
    reference = logits[-1] if getattr(logits, "dim", lambda: 1)() == 2 else logits
    row = layer_logits[0]
    diff = float((row.float() - reference.float()).abs().max())
    top1_match = int(row.argmax()) == int(reference.argmax())
    logger.info("[TOLL check] max_abs=%.6g top1_match=%s", diff, top1_match)
