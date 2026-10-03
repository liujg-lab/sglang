"""Integer-only SR preparation kernels. Imported only on the NPU path."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["NPredict"])
def _pack(
    Index,
    Predict,
    Length,
    Out,
    NPredict,
    L: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    LS: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    Block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, Block)
    idx = tl.load(Index + row * IS0 + col * IS1, col < L, other=-1).to(tl.int64)
    valid = (col < L) & (idx >= 0) & (idx < NPredict)
    token = tl.load(Predict + idx, valid, other=0).to(tl.int64)
    # Preserve the existing empty-predict branch exactly. CPU validation also
    # rejects indexes below -1, independently of this flag.
    bad = tl.where(NPredict == 0, idx >= 0, (idx < -1) | (idx >= NPredict))
    error = tl.max(((col < L) & bad).to(tl.int32), 0)
    length = tl.load(Length + row * LS).to(tl.int64)
    tl.store(Out + row * OS0 + col * OS1, idx, col < L)
    tl.store(Out + row * OS0 + (L + col) * OS1, token, col < L)
    tl.store(Out + row * OS0 + 2 * L * OS1, length)
    tl.store(Out + row * OS0 + (2 * L + 1) * OS1, error.to(tl.int64))


def pack_accept(index, predict, length, out):
    if index.shape[0]:
        _pack[(index.shape[0],)](
            index,
            predict,
            length,
            out,
            predict.numel(),
            index.shape[1],
            *index.stride(),
            length.stride(0),
            *out.stride(),
            triton.next_power_of_2(max(index.shape[1], 1)),
        )


@triton.jit(do_not_specialize=["N", "F", "Page"])
def _commit(
    Cache,
    Src,
    Dst,
    Pages,
    OutSrc,
    OutDst,
    OutPages,
    N,
    F,
    Page,
    CS: tl.constexpr,
    SS: tl.constexpr,
    DS: tl.constexpr,
    PS: tl.constexpr,
    Block: tl.constexpr,
):
    i = tl.program_id(0) * Block + tl.arange(0, Block)
    s = tl.load(Src + i * SS, i < N, other=0).to(tl.int64)
    d = tl.load(Dst + i * DS, i < N, other=0).to(tl.int64)
    p = tl.load(Pages + i * PS, i < F, other=0).to(tl.int64)
    sv = tl.load(Cache + s * CS, i < N, other=0).to(tl.int64)
    dv = tl.load(Cache + d * CS, i < N, other=0).to(tl.int64)
    pv = tl.load(Cache + p * CS, i < F, other=0).to(tl.int64)
    tl.store(OutSrc + i, sv, i < N)
    tl.store(OutDst + i, dv, i < N)
    # Cache slots are nonnegative, so integer division equals floor division.
    tl.store(OutPages + i, pv // Page, i < F)


def gather_commit_slots(cache, src, dst, pages, page_size, out_src, out_dst, out_pages):
    n, f = src.numel(), pages.numel()
    if max(n, f):
        _commit[(triton.cdiv(max(n, f), 256),)](
            cache,
            src,
            dst,
            pages,
            out_src,
            out_dst,
            out_pages,
            n,
            f,
            page_size,
            cache.stride(0),
            src.stride(0),
            dst.stride(0),
            pages.stride(0),
            256,
        )


@triton.jit(do_not_specialize=["Page"])
def _prefix(
    Req,
    Pool,
    Branch,
    Meta,
    Src,
    Dst,
    Page,
    RS0: tl.constexpr,
    RS1: tl.constexpr,
    PS: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    BS2: tl.constexpr,
    Start: tl.constexpr,
    Block: tl.constexpr,
):
    b = tl.program_id(0)
    br = tl.program_id(1)
    t = tl.program_id(2) * Block + tl.arange(0, Block)
    base = tl.load(Meta + b * 3).to(tl.int64)
    rem = tl.load(Meta + b * 3 + 1).to(tl.int64)
    offset = tl.load(Meta + b * 3 + 2).to(tl.int64)
    row = tl.load(Pool + b * PS).to(tl.int64)
    page = tl.load(Branch + b * BS0 + (br + Start) * BS1).to(tl.int64)
    src = tl.load(Req + row * RS0 + (base + t) * RS1, t < rem, other=0)
    pos = offset + br * rem + t
    tl.store(Src + pos, src.to(tl.int64), t < rem)
    tl.store(Dst + pos, page * Page + t, t < rem)


def prefix_tail(req, pool, branch, meta, src, dst, bs, start, topk, page):
    if bs and topk > start:
        _prefix[(bs, topk - start, triton.cdiv(page, 256))](
            req,
            pool,
            branch,
            meta,
            src,
            dst,
            page,
            *req.stride(),
            pool.stride(0),
            *branch.stride(),
            start,
            256,
        )
