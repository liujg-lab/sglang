"""Small fixtures for tests exercising the new explicit KV workspace API."""

from types import SimpleNamespace

from sglang.srt.speculative.standalone_remote import sr_kv_copy as kv


def move_pool(pool, src, dst):
    if (src is None or src.numel() == 0) and (dst is None or dst.numel() == 0):
        return
    if src is None or dst is None or src.numel() != dst.numel():
        raise RuntimeError("KV move src/dst length mismatch")
    return kv.move_kv_slots_(kv.prepare_kv_move(pool, src.numel()), src, dst)


def move_paged(tensor, src, dst):
    return move_pool(SimpleNamespace(kv_buffer=tensor), src, dst)


def move_buffers(k, v, src, dst, index=None):
    return move_pool(
        SimpleNamespace(k_buffer=k, v_buffer=v, index_k_buffer=index), src, dst
    )
