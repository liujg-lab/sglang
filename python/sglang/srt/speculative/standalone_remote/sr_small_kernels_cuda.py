"""CUDA fixed-capacity integer packing. Imported after CUDA admission only."""

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


@triton.jit(do_not_specialize=["N", "F"])
def _token_slots(
    Cache,
    Kept,
    Free,
    Slots,
    Released,
    N,
    F,
    CS: tl.constexpr,
    KS: tl.constexpr,
    FS: tl.constexpr,
    Block: tl.constexpr,
):
    i = tl.program_id(0) * Block + tl.arange(0, Block)
    k = tl.load(Kept + i * KS, i < N, other=0).to(tl.int64)
    f = tl.load(Free + i * FS, i < F, other=0).to(tl.int64)
    slot = tl.load(Cache + k * CS, i < N, other=0).to(tl.int64)
    released = tl.load(Cache + f * CS, i < F, other=0).to(tl.int64)
    tl.store(Slots + i, slot, i < N)
    tl.store(Released + i, released, i < F)


def gather_token_slots(cache, kept, free, slots, released):
    n, f = kept.numel(), free.numel()
    if max(n, f):
        _token_slots[(triton.cdiv(max(n, f), 256),)](
            cache,
            kept,
            free,
            slots,
            released,
            n,
            f,
            cache.stride(0),
            kept.stride(0),
            free.stride(0),
            256,
        )
