"""One-launch int64 pack of a STANDALONE_REMOTE tree reply.

Imported only on the CUDA/NPU path. Candidate slots are resolved in this
kernel: each output walks ``step`` then ``branch`` and keeps the first node-id
match. A miss stores ``-1``.
"""

import triton
import triton.language as tl

# Triton-Ascend rejects a grid whose program count is 65536 or larger.
_GRID_LIMIT = 65535


@triton.jit(
    do_not_specialize=[
        "Base",
        "Count",
        "TokEnd",
        "ParEnd",
        "IdxEnd",
        "TokenW",
        "ParentW",
        "IndexW",
        "TopK",
        "Steps",
        "TokS0",
        "TokS1",
        "ParS0",
        "ParS1",
        "IdxS0",
        "IdxS1",
        "NodeS0",
        "NodeS1",
        "CompB",
        "CompK",
        "CompS",
    ]
)
def _pack_tree_reply(
    Out,
    Tokens,
    Parents,
    Indices,
    Nodes,
    Compact,
    Base,
    Count,
    TokEnd,
    ParEnd,
    IdxEnd,
    TokenW,
    ParentW,
    IndexW,
    TopK,
    Steps,
    TokS0,
    TokS1,
    ParS0,
    ParS1,
    IdxS0,
    IdxS1,
    NodeS0,
    NodeS1,
    CompB,
    CompK,
    CompS,
    WRITE_SLOTS: tl.constexpr,
):
    local = tl.program_id(0)
    if local < Count:
        elem = (local.to(tl.int64) + Base).to(tl.int64)
        if elem < TokEnd:
            width = TokenW.to(tl.int64)
            row = elem // width
            col = elem - row * width
            value = tl.load(Tokens + row * TokS0 + col * TokS1).to(tl.int64)
            tl.store(Out + elem, value)
        elif elem < ParEnd:
            width = ParentW.to(tl.int64)
            rel = elem - TokEnd
            row = rel // width
            col = rel - row * width
            value = tl.load(Parents + row * ParS0 + col * ParS1).to(tl.int64)
            tl.store(Out + elem, value)
        elif elem < IdxEnd:
            width = IndexW.to(tl.int64)
            rel = elem - ParEnd
            row = rel // width
            col = rel - row * width
            value = tl.load(Indices + row * IdxS0 + col * IdxS1).to(tl.int64)
            tl.store(Out + elem, value)
        elif WRITE_SLOTS:
            width = IndexW.to(tl.int64)
            rel = elem - IdxEnd
            request = rel // width
            cand = rel - request * width
            target = tl.load(Indices + request * IdxS0 + cand * IdxS1).to(tl.int64)
            topk = TopK.to(tl.int32)
            steps = Steps.to(tl.int32)
            cursor = tl.zeros((), dtype=tl.int32)
            span = steps * topk
            found = tl.full((), -1, tl.int64)
            matched = tl.zeros((), dtype=tl.int32)
            while cursor < span:
                step = cursor // topk
                branch = cursor - step * topk
                col = request.to(tl.int32) * topk + branch
                nid = tl.load(
                    Nodes + step.to(tl.int64) * NodeS0 + col.to(tl.int64) * NodeS1
                ).to(tl.int64)
                slot = tl.load(
                    Compact
                    + request * CompB
                    + branch.to(tl.int64) * CompK
                    + step.to(tl.int64) * CompS
                ).to(tl.int64)
                take = (matched == 0) & (nid == target)
                found = tl.where(take, slot, found)
                matched = tl.where(take, 1, matched)
                cursor += 1
            tl.store(Out + elem, found)


def launch_tree_reply_pack(
    buf,
    tokens,
    parents,
    indices,
    node_ids,
    compact,
    tok_end,
    par_end,
    idx_end,
    total,
    topk,
    steps,
    write_slots,
):
    """Write ``total`` int64 elements into ``buf``. Empty regions are not read."""
    if total <= 0:
        return
    token_w = max(int(tokens.shape[1]), 1)
    parent_w = max(int(parents.shape[1]), 1)
    index_w = max(int(indices.shape[1]), 1)
    tok_s0, tok_s1 = _stride2(tokens)
    par_s0, par_s1 = _stride2(parents)
    idx_s0, idx_s1 = _stride2(indices)
    if write_slots:
        node_s0, node_s1 = _stride2(node_ids)
        comp_b, comp_k, comp_s = (
            int(compact.stride(0)),
            int(compact.stride(1)),
            int(compact.stride(2)),
        )
    else:
        node_ids = tokens
        compact = tokens
        node_s0 = node_s1 = 0
        comp_b = comp_k = comp_s = 0
        topk = 1
        steps = 1
    for base in range(0, int(total), _GRID_LIMIT):
        count = min(_GRID_LIMIT, int(total) - base)
        _pack_tree_reply[(count,)](
            buf,
            tokens,
            parents,
            indices,
            node_ids,
            compact,
            base,
            count,
            int(tok_end),
            int(par_end),
            int(idx_end),
            token_w,
            parent_w,
            index_w,
            int(topk),
            int(steps),
            tok_s0,
            tok_s1,
            par_s0,
            par_s1,
            idx_s0,
            idx_s1,
            node_s0,
            node_s1,
            comp_b,
            comp_k,
            comp_s,
            WRITE_SLOTS=bool(write_slots),
        )


def _stride2(tensor):
    if tensor.ndim != 2:
        raise RuntimeError("tree reply source must be rank 2")
    return int(tensor.stride(0)), int(tensor.stride(1))
