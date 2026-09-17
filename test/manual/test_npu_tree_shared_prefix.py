"""NPU correctness/graph/latency smoke test, without model weights.

Run with PYTHONPATH=python python test/manual/test_npu_tree_shared_prefix.py.
CPU tests do not certify this path; this test requires torch_npu on an NPU host.
"""

import ast
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

from sglang.srt.speculative.tree_attn_fallback import (
    flatten_paged_kv,
    gather_kv_into,
    zero_gathered_kv_padding,
)
from sglang.srt.speculative.tree_shared_prefix import (
    SharedPrefixMetadata,
    fill_shared_draft_,
    fill_shared_verify_,
    shared_prefix_attention,
)


def compact_production():
    path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    klass = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AscendAttnBackend"
    )
    fn = next(
        n
        for n in klass.body
        if isinstance(n, ast.FunctionDef) and n.name == "_run_tree_compact_fia"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(fn)
    ns = dict(globals())
    exec(compile(module, str(path), "exec"), ns)
    return ns[fn.name]


def oracle_and_slots(q, k, v, md):
    # CPU FP32 reference, independent of NPU matmul/precision settings.
    q, k, v = (
        q.cpu().float(),
        k.cpu().float().view(-1, 2, 64),
        v.cpu().float().view(-1, 2, 64),
    )
    prefix, node, ancestor, pl, al = [x.cpu() for x in vars(md).values()]
    outs, rows, lengths = [], [], []
    for b in range(q.shape[0]):
        for r in range(q.shape[1]):
            ids = torch.cat((prefix[b, : pl[b]], node[b, ancestor[b, r, : al[b, r]]]))
            keys, vals = k[ids].repeat_interleave(2, 1), v[ids].repeat_interleave(2, 1)
            p = (torch.einsum("hd,shd->hs", q[b, r], keys) / 8).softmax(-1)
            outs.append(torch.einsum("hs,shd->hd", p, vals))
            rows.append(ids)
            lengths.append(len(ids))
    width = max(lengths)
    slots = torch.zeros(len(rows), width, dtype=torch.int64)
    for i, ids in enumerate(rows):
        slots[i, : len(ids)] = ids
    return torch.stack(outs).reshape(-1, 256), slots.npu(), lengths


@unittest.skipUnless(torch_npu is not None, "requires torch_npu and NPU hardware")
class TestNpuTreeSharedPrefix(unittest.TestCase):
    def test_draft_int32_metadata_eager_and_graph(self):
        """Exercise the real request-pool dtype before capture and every replay."""
        torch.manual_seed(73)
        for dtype in (torch.float16, torch.bfloat16):
            for page in (1, 128):
                for cap in (256, 512):
                    with self.subTest(dtype=dtype, page=page, cap=cap):
                        table_cpu = torch.randperm(4096).view(2, 2048).int()
                        table = table_cpu.npu()
                        pool = torch.tensor([0, 1], device="npu")
                        q = torch.randn(2, 3, 4, 64, device="npu", dtype=dtype)
                        k = torch.randn(32, 128, 2, 64, device="npu", dtype=dtype)
                        v = torch.randn_like(k)
                        md = SharedPrefixMetadata.allocate(2, 3, cap, 15, 5, "npu")
                        pointers = [x.data_ptr() for x in vars(md).values()]

                        def fill(lengths, step, request_order):
                            pool.copy_(torch.tensor(request_order, device="npu"))
                            fill_shared_draft_(
                                md,
                                table,
                                pool,
                                lengths,
                                page_size=page,
                                topk=3,
                                steps=5,
                                step=step,
                            )

                        def run():
                            return shared_prefix_attention(
                                q, k, v, md, scale=1 / 8, kv_heads=2
                            )

                        # The production runner prepares metadata outside the
                        # graph; capture/replay only records attention math.
                        fill([127, 128], 0, [0, 1])
                        for _ in range(3):
                            run()
                        torch.npu.synchronize()
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph, auto_dispatch_capture=True):
                            graph_out = run()
                        for lengths, step, order in (
                            ([127, 128], 0, [0, 1]),
                            ([129], 3, [1, 0]),
                            ([7, cap - 1], 1, [1, 0]),
                            ([0], 0, [0, 1]),
                        ):
                            fill(lengths, step, order)
                            node_cpu = md.node_slots.cpu().view(2, 3, 5)
                            for b, p in enumerate(lengths):
                                stride = (
                                    5
                                    if page == 1
                                    else ((p % page + 5 + page - 1) // page) * page
                                )
                                for branch in range(3):
                                    start = p + branch * stride
                                    torch.testing.assert_close(
                                        node_cpu[b, branch, : step + 1],
                                        table_cpu[
                                            order[b], start : start + step + 1
                                        ].long(),
                                    )
                                self.assertEqual(
                                    node_cpu[b, :, step + 1 :].count_nonzero(), 0
                                )
                            for tensor in vars(md).values():
                                self.assertEqual(
                                    tensor[len(lengths) :].cpu().count_nonzero(), 0
                                )
                            expected, _, _ = oracle_and_slots(q, k, v, md)
                            eager = run()
                            graph.replay()
                            torch.npu.synchronize()
                            tolerance = (
                                dict(rtol=1e-2, atol=1e-2)
                                if dtype == torch.bfloat16
                                else dict(rtol=3e-3, atol=3e-3)
                            )
                            torch.testing.assert_close(
                                eager.cpu().float(), expected, **tolerance
                            )
                            torch.testing.assert_close(
                                graph_out.cpu().float(), expected, **tolerance
                            )
                            self.assertEqual(
                                pointers, [x.data_ptr() for x in vars(md).values()]
                            )

    def test_eager_graph_and_compact(self):
        torch.manual_seed(73)
        compact = compact_production()
        for dtype in (torch.float16, torch.bfloat16):
            for cap in (256, 512):
                q = torch.randn(2, 3, 4, 64, device="npu", dtype=dtype)
                k = torch.randn(6, 128, 2, 64, device="npu", dtype=dtype)
                v = torch.randn_like(k)
                table = (
                    (torch.stack((torch.randperm(512), torch.randperm(512))) + 1)
                    .int()
                    .npu()
                )
                pool = torch.tensor([0, 1], device="npu")
                nodes = torch.arange(600, 606, device="npu")
                md = SharedPrefixMetadata.allocate(2, 3, cap, 3, 3, "npu")
                run = lambda: shared_prefix_attention(
                    q, k, v, md, scale=1 / 8, kv_heads=2
                )

                def fill(lengths):
                    tree = torch.tensor(
                        [[1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=torch.bool
                    )
                    mask = torch.cat(
                        [
                            torch.cat(
                                (torch.ones(3, p, dtype=torch.bool), tree), 1
                            ).flatten()
                            for p in lengths
                        ]
                    ).npu()
                    fill_shared_verify_(md, table, pool, lengths, nodes, mask, 3)

                fill([127, 3])
                for _ in range(3):
                    run()
                torch.npu.synchronize()
                baseline = torch.npu.memory_allocated()
                torch.npu.reset_peak_memory_stats()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, auto_dispatch_capture=True):
                    graph_out = run()
                graph_peak = torch.npu.max_memory_allocated() - baseline
                pointers = [x.data_ptr() for x in vars(md).values()]
                for lengths in ([127, 3], [cap - 1, 5], [1, 0]):
                    # Change both the mapping and lengths without recapturing.
                    pool.copy_(pool.flip(0))
                    fill(lengths)
                    expected, slots, lens = oracle_and_slots(q, k, v, md)
                    eager = run()
                    graph.replay()
                    torch.npu.synchronize()
                    tolerance = (
                        dict(rtol=1e-2, atol=1e-2)
                        if dtype == torch.bfloat16
                        else dict(rtol=3e-3, atol=3e-3)
                    )
                    torch.testing.assert_close(
                        eager.cpu().float(), expected, **tolerance
                    )
                    torch.testing.assert_close(
                        graph_out.cpu().float(), expected, **tolerance
                    )
                    self.assertEqual(
                        pointers, [x.data_ptr() for x in vars(md).values()]
                    )
                    scratch = (
                        torch.empty(
                            6, slots.shape[1], 2, 64, device="npu", dtype=dtype
                        ),
                        torch.empty(
                            6, slots.shape[1], 2, 64, device="npu", dtype=dtype
                        ),
                    )
                    backend = NS(
                        _ensure_tree_kv_scratch=lambda *a: scratch,
                        forward_metadata=NS(tree_fia_kv_lens_cpu=lens),
                        graph_mode=False,
                    )
                    lens_t = torch.tensor(lens, device="npu", dtype=torch.int32)

                    def run_compact():
                        return compact(
                            backend,
                            q,
                            k,
                            v,
                            kv_slots=slots,
                            kv_lens=lens_t,
                            scale=1 / 8,
                            n_q_heads=4,
                            n_kv_heads=2,
                            qk_head_dim=64,
                            v_head_dim=64,
                        )

                    torch.testing.assert_close(
                        run_compact().cpu().float(), expected, **tolerance
                    )
                compact_graph = torch.npu.NPUGraph()
                for _ in range(3):
                    run_compact()
                torch.npu.synchronize()
                compact_baseline = torch.npu.memory_allocated()
                torch.npu.reset_peak_memory_stats()
                with torch.npu.graph(compact_graph, auto_dispatch_capture=True):
                    compact_out = run_compact()
                compact_peak = (
                    torch.npu.max_memory_allocated()
                    - compact_baseline
                    + sum(x.numel() * x.element_size() for x in scratch)
                )
                for name, call in (
                    ("shared_eager", run),
                    ("shared_graph", graph.replay),
                    ("compact_eager", run_compact),
                    ("compact_graph", compact_graph.replay),
                ):
                    torch.npu.synchronize()
                    baseline = torch.npu.memory_allocated()
                    torch.npu.reset_peak_memory_stats()
                    start = time.perf_counter()
                    for _ in range(20):
                        call()
                    torch.npu.synchronize()
                    peak = torch.npu.max_memory_allocated() - baseline
                    print(
                        f"{dtype=} {cap=} {name} host_completed_ms={(time.perf_counter() - start) * 50:.3f} transient_bytes={peak}"
                    )
                print(
                    f"{dtype=} {cap=} shared_graph_capture_bytes={graph_peak} compact_graph_capture_plus_scratch_bytes={compact_peak}"
                )
                torch.testing.assert_close(
                    compact_out.cpu().float(), expected, **tolerance
                )


if __name__ == "__main__":
    unittest.main()
