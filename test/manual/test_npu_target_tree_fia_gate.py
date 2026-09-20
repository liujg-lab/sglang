"""NPU hardware gate for Target BSND + sparse_mode=0 + tree mask.

Compares FIA output to an FP32 dense reference for FP16 and BF16, then
captures a minimal graph and replays after in-place mask/page/length
updates. Failure here means the Target tree_paged_fia path is not viable.

Run with:
  PYTHONPATH=python python test/manual/test_npu_target_tree_fia_gate.py
"""

from __future__ import annotations

import unittest

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

from sglang.srt.speculative.standalone_remote.verifier.sr_target_tree_fia import (
    SRTargetTreeFiaMetadata,
    dense_target_tree_fia_reference,
    fill_target_tree_fia_metadata_,
    pages_for_s_cap,
    prefix_columns_visible,
)
from sglang.srt.speculative.tree_attn_mask import full_mask_numel


def _req_to_token(page_ids, page_size: int, extra: int = 16) -> torch.Tensor:
    bs = len(page_ids)
    n_pages = max(len(p) for p in page_ids)
    ctx = n_pages * page_size + extra
    req = torch.zeros((bs, ctx), dtype=torch.int64)
    for b, pages in enumerate(page_ids):
        for j, pid in enumerate(pages):
            start = j * page_size
            req[b, start : start + page_size] = int(pid) * page_size + torch.arange(
                page_size, dtype=torch.int64
            )
    return req


def _full_mask(prefixes, queries: int) -> torch.Tensor:
    tree = torch.eye(queries, dtype=torch.bool)
    tree[:, 0] = True
    for q in range(3, queries):
        tree[q, 1] = True
        tree[q, : q + 1] = True
    rows = []
    for prefix in prefixes:
        width = int(prefix) + queries
        for q in range(queries):
            row = torch.zeros(width, dtype=torch.bool)
            if prefix:
                row[:prefix] = True
            row[prefix : prefix + queries] = tree[q]
            rows.append(row)
    return torch.cat(rows)


def _call_fia(query, k_cache, v_cache, md, hq, hkv, scale, page_size):
    output, _ = torch.ops.npu.npu_fused_infer_attention_score(
        query,
        k_cache,
        v_cache,
        input_layout="BSND",
        num_heads=hq,
        num_key_value_heads=hkv,
        scale=scale,
        block_table=md.block_tables,
        block_size=page_size,
        atten_mask=md.blocked_mask,
        actual_seq_lengths=md.q_lens_cpu,
        actual_seq_lengths_kv=md.kv_lens_cpu,
        sparse_mode=0,
    )
    return output.reshape(query.shape[0] * query.shape[1], hq * query.shape[-1])


@unittest.skipUnless(torch_npu is not None, "requires torch_npu and NPU hardware")
class TestNpuTargetTreeFiaGate(unittest.TestCase):
    def test_eager_and_minigraph_match_dense_reference(self):
        torch.manual_seed(21)
        page = 128
        hq, hkv, dim = 4, 2, 64
        scale = 1.0 / 8.0
        prefixes = [0, 129]
        queries = 7
        raw_bs = len(prefixes)
        s_cap = 256
        pages = pages_for_s_cap(s_cap, page)
        page_ids = [[3], [5, 1, 8]]
        table = _req_to_token(page_ids, page).npu()
        pool = torch.tensor([0, 1], device="npu")
        n_pages = 16
        k_cache = torch.randn(
            n_pages, page, hkv * dim, device="npu", dtype=torch.float16
        )
        v_cache = torch.randn_like(k_cache)
        mask = _full_mask(prefixes, queries)
        self.assertEqual(int(mask.numel()), full_mask_numel(prefixes, queries))

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype, phase="eager"):
                q = torch.randn(raw_bs, queries, hq, dim, device="npu", dtype=dtype)
                md = SRTargetTreeFiaMetadata.allocate(
                    raw_bs, queries, pages, page, "npu"
                )
                fill_target_tree_fia_metadata_(
                    md, table, pool, mask, prefixes, queries, raw_bs
                )
                self.assertTrue(prefix_columns_visible(md.blocked_mask.cpu(), prefixes))
                got = _call_fia(q, k_cache.to(dtype), v_cache.to(dtype), md, hq, hkv, scale, page)
                ref = dense_target_tree_fia_reference(
                    q.cpu(),
                    k_cache.cpu().to(dtype),
                    v_cache.cpu().to(dtype),
                    md.block_tables.cpu(),
                    md.kv_lens_cpu,
                    md.blocked_mask.cpu(),
                    scale=scale,
                    page_size=page,
                    n_kv_heads=hkv,
                )
                torch.testing.assert_close(
                    got.cpu().float(),
                    ref.float(),
                    rtol=2e-2 if dtype == torch.float16 else 3e-2,
                    atol=2e-2 if dtype == torch.float16 else 3e-2,
                )

        dtype = torch.bfloat16
        q = torch.randn(raw_bs, queries, hq, dim, device="npu", dtype=dtype)
        md = SRTargetTreeFiaMetadata.allocate(raw_bs, queries, pages, page, "npu")
        fill_target_tree_fia_metadata_(
            md, table, pool, mask, prefixes, queries, raw_bs
        )
        k_view = k_cache.to(dtype)
        v_view = v_cache.to(dtype)
        stream = torch.npu.Stream()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            captured = _call_fia(q, k_view, v_view, md, hq, hkv, scale, page)
        first = captured.detach().clone()
        alt_prefixes = [127, 128]
        alt_mask = _full_mask(alt_prefixes, queries)
        fill_target_tree_fia_metadata_(
            md, table, pool, alt_mask, alt_prefixes, queries, raw_bs
        )
        payload = [{"actual_seq_lengths_kv": list(md.kv_lens_cpu)}]
        graph.update(cpu_update_input=payload)
        graph.replay()
        replayed = captured.detach().clone()
        self.assertFalse(torch.equal(first.cpu(), replayed.cpu()))
        ref = dense_target_tree_fia_reference(
            q.cpu(),
            k_view.cpu(),
            v_view.cpu(),
            md.block_tables.cpu(),
            md.kv_lens_cpu,
            md.blocked_mask.cpu(),
            scale=scale,
            page_size=page,
            n_kv_heads=hkv,
        )
        torch.testing.assert_close(
            replayed.cpu().float(),
            ref.float(),
            rtol=3e-2,
            atol=3e-2,
        )


if __name__ == "__main__":
    unittest.main()
