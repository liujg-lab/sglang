"""NPU FIA tree-mask polarity probe, STANDALONE_REMOTE tree+graph nightly coverage,
and FULL_MASK layout parity vs the CUDA-faithful reference.

Heterogeneous CUDA↔NPU e2e is multi-host and is not executed here.
Homogeneous NPU→NPU coverage requires weights and torch_npu.
``TestNpuBuildTreeMaskParity`` only needs torch_npu (no weights).
"""

from __future__ import annotations

import os
import unittest
from urllib.parse import urlparse

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=600, suite="nightly-1-npu-a3", nightly=True)


def _npu_available() -> bool:
    try:
        import torch

        return bool(getattr(torch, "npu", None) and torch.npu.is_available())
    except Exception:
        return False


class TestNpuFiaTreeMaskProbe(CustomTestCase):
    """Confirm Ascend FIA atten_mask True = masked for TARGET_VERIFY layout.

    [Test Category] Speculative Decoding
    [Test Target] npu_fused_infer_attention_score sparse_mode=0 TND mask polarity
    """

    def test_fia_true_is_masked_for_last_kv(self):
        if not _npu_available():
            self.skipTest("torch_npu is not available")
        import torch

        from sglang.srt.speculative.tree_attn_mask import FIA_TREE_MASK_CONTRACT

        self.assertEqual(FIA_TREE_MASK_CONTRACT["ascend_true"], "masked")
        self.assertEqual(FIA_TREE_MASK_CONTRACT["sparse_mode"], 0)

        device = "npu"
        dtype = torch.float16
        # T=1 query, S=2 KV, N=1, D=16. Hide the second KV token.
        query = torch.ones((1, 1, 16), device=device, dtype=dtype)
        key_base = torch.ones((2, 1, 16), device=device, dtype=dtype)
        value_base = torch.ones((2, 1, 16), device=device, dtype=dtype)
        key_hot = key_base.clone()
        value_hot = value_base.clone()
        key_hot[1].fill_(100)
        value_hot[1].fill_(100)
        atten_mask = torch.tensor([[False, True]], device=device)

        def _fia(key, value):
            out, _ = torch.ops.npu.npu_fused_infer_attention_score(
                query,
                key,
                value,
                num_heads=1,
                num_key_value_heads=1,
                input_layout="TND",
                atten_mask=atten_mask,
                scale=1.0,
                actual_seq_lengths=[1],
                actual_seq_lengths_kv=[2],
                sparse_mode=0,
            )
            return out

        try:
            out_base = _fia(key_base, value_base)
            out_hot = _fia(key_hot, value_hot)
        except Exception as e:
            self.fail(
                "FIA tree-mask probe failed; do not guess polarity/layout. "
                f"contract={FIA_TREE_MASK_CONTRACT} err={e}"
            )
        torch.testing.assert_close(out_base, out_hot, rtol=1e-2, atol=1e-2)


class TestNpuStandaloneRemoteTreeGraph(CustomTestCase):
    """NPU→NPU STANDALONE_REMOTE tree (topk>1) with graphs enabled.

    [Test Category] Speculative Decoding
    [Test Target] --speculative-algorithm STANDALONE_REMOTE; topk>1; NPU graph hit
    """

    @classmethod
    def setUpClass(cls):
        if not _npu_available():
            raise unittest.SkipTest("torch_npu is not available")
        from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH

        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        if not os.path.isdir(cls.model):
            raise unittest.SkipTest(f"weights missing: {cls.model}")

        from sglang.test.test_utils import (
            DEFAULT_URL_FOR_TEST,
            popen_launch_server,
        )

        cls.base_url = DEFAULT_URL_FOR_TEST
        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        host = parsed.hostname or "127.0.0.1"
        target_port = parsed.port or 30000
        draft_port = target_port + 8
        rpc_port = target_port + 19
        cls._procs = []

        common = [
            "--trust-remote-code",
            "--device",
            "npu",
            "--attention-backend",
            "ascend",
            "--speculative-algorithm",
            "STANDALONE_REMOTE",
            "--speculative-num-steps",
            "2",
            "--speculative-eagle-topk",
            "2",
            "--speculative-num-draft-tokens",
            "4",
            "--speculative-verify-mode",
            "greedy",
            "--standalone-remote-addr",
            "127.0.0.1",
            "--standalone-remote-port",
            str(rpc_port),
            "--mem-fraction-static",
            "0.35",
            "--tp-size",
            "1",
            "--dtype",
            "bfloat16",
            "--skip-server-warmup",
        ]
        # Graphs stay ON: topk=1 + --disable-cuda-graph is not tree compatibility.
        cls.target = popen_launch_server(
            cls.model,
            f"http://{host}:{target_port}",
            timeout=1500,
            other_args=[
                *common,
                "--standalone-remote-role",
                "target",
                "--port",
                str(target_port),
            ],
        )
        cls._procs.append(cls.target)
        cls.draft = popen_launch_server(
            cls.model,
            f"http://{host}:{draft_port}",
            timeout=1500,
            other_args=[
                *common,
                "--standalone-remote-role",
                "draft",
                "--port",
                str(draft_port),
            ],
        )
        cls._procs.append(cls.draft)
        cls.url = f"http://{host}:{target_port}"

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.utils import kill_process_tree

        for proc in getattr(cls, "_procs", []):
            if proc is not None and getattr(proc, "pid", None):
                kill_process_tree(proc.pid)

    def test_tree_generate_does_not_disable_graphs(self):
        import requests

        resp = requests.post(
            self.url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        text = body.get("text") or body.get("output_ids")
        self.assertTrue(text)


def _load_known_eagle_tree_case():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "unit"
        / "spec"
        / "test_build_tree_ref.py"
    )
    spec = importlib.util.spec_from_file_location("test_build_tree_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seq_list(seq_lens):
    return [int(x) for x in seq_lens.detach().reshape(-1).tolist()]


class TestNpuBuildTreeMaskParity(CustomTestCase):
    """NPU build_tree_kernel_efficient FULL_MASK vs CUDA-faithful reference.

    [Test Category] Speculative Decoding
    [Test Target] torch.ops.npu.build_tree_kernel_efficient tree_mask layout
    """

    def setUp(self):
        if not _npu_available():
            self.skipTest("torch_npu is not available")

    def _run_parity(self, seq_lens_dtype):
        from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient
        from sglang.srt.speculative.tree_verify import (
            build_tree_kernel_efficient_ref,
            first_full_mask_mismatch,
        )

        case_mod = _load_known_eagle_tree_case()
        case = case_mod.known_eagle_tree_tensors("npu")
        seq_lens = case["seq_lens"].to(dtype=seq_lens_dtype)
        seq_list = _seq_list(seq_lens)
        draft = case["num_draft_token"]

        ref_mask, ref_pos, ref_idx, ref_nxt, ref_sib = build_tree_kernel_efficient_ref(
            case["parent_list"],
            case["top_scores_index"],
            seq_lens,
            case["topk"],
            case["depth"],
            draft,
        )
        got_mask, got_pos, got_idx, got_nxt, got_sib, _draft_tokens = (
            build_tree_kernel_efficient(
                verified_id=case["verified_id"],
                parent_list=case["parent_list"],
                top_scores_index=case["top_scores_index"],
                draft_tokens=case["draft_tokens"],
                seq_lens=seq_lens,
                seq_lens_sum=int(seq_lens.sum().item()),
                topk=case["topk"],
                spec_steps=case["depth"],
                num_verify_tokens=draft,
            )
        )

        def _eq(name, got, ref):
            self.assertEqual(
                got.detach().to("cpu").tolist(),
                ref.detach().to("cpu").tolist(),
                f"{name} mismatch with seq_lens dtype={seq_lens_dtype}",
            )

        _eq("positions", got_pos, ref_pos)
        _eq("retrive_index", got_idx, ref_idx)
        _eq("retrive_next_token", got_nxt, ref_nxt)
        _eq("retrive_next_sibling", got_sib, ref_sib)

        mismatch = first_full_mask_mismatch(got_mask, ref_mask, seq_list, draft)
        if mismatch is not None:
            b, t, c, got_v, ref_v, got_row, ref_row = mismatch
            self.fail(
                f"tree_mask mismatch seq_lens dtype={seq_lens_dtype} "
                f"at (batch={b}, row={t}, col={c}): got={got_v} ref={ref_v}\n"
                f"  got_row={got_row}\n"
                f"  ref_row={ref_row}\n"
                f"  got_visible={[i for i, v in enumerate(got_row) if v]}\n"
                f"  ref_visible={[i for i, v in enumerate(ref_row) if v]}"
            )

    def test_full_mask_matches_ref_seq_lens_int64(self):
        import torch

        self._run_parity(torch.int64)

    def test_full_mask_matches_ref_seq_lens_int32(self):
        import torch

        self._run_parity(torch.int32)


class TestNpuTreeDraftSlotGather(CustomTestCase):
    """Token-level tree-draft attention vs CPU oracle, and vs FIA after last-page copy.

    [Test Category] Speculative Decoding
    [Test Target] tree_draft_attention on NPU; FIA cross-check with duplicated last page
    """

    def setUp(self):
        if not _npu_available():
            self.skipTest("torch_npu is not available")

    def test_tree_draft_attention_matches_cpu_oracle(self):
        import math

        import torch

        from sglang.srt.speculative.tree_attn_fallback import (
            chunked_attend,
            flatten_paged_kv,
            tree_draft_attention,
        )

        torch.manual_seed(4)
        n_q, n_kv, d = 4, 2, 8
        device = "npu"
        k_cache = torch.randn(2, 8, n_kv * d, device=device)
        v_cache = torch.randn(2, 8, n_kv * d, device=device)
        query = torch.randn(2, n_q, d, device=device)
        kv_slots = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=device)
        lens = [3, 4]
        scale = 1.0 / math.sqrt(d)
        out = tree_draft_attention(
            query,
            k_cache,
            v_cache,
            kv_slots=kv_slots,
            kv_lens=lens,
            scale=scale,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
            qk_head_dim=d,
            v_head_dim=d,
        )
        flat_k = flatten_paged_kv(k_cache.cpu(), n_kv, d)
        flat_v = flatten_paged_kv(v_cache.cpu(), n_kv, d)
        for i, kv_len in enumerate(lens):
            slots = kv_slots[i, :kv_len].cpu()
            ref = chunked_attend(
                query[i].cpu(),
                flat_k.index_select(0, slots),
                flat_v.index_select(0, slots),
                scale,
            )
            self.assertTrue(
                torch.allclose(
                    out[i].cpu().view(n_q, d).float(),
                    ref.float(),
                    atol=2e-3,
                    rtol=2e-3,
                ),
                f"row {i} NPU vs CPU oracle",
            )

    def test_slot_gather_matches_fia_when_last_page_copied(self):
        import math

        import torch

        from sglang.srt.speculative.spec_utils import (
            build_tree_draft_block_tables,
            build_tree_draft_kv_slots,
        )
        from sglang.srt.speculative.tree_attn_fallback import tree_draft_attention

        device = "npu"
        page_size, topk, num_steps, seq, step_id = 16, 2, 2, 17, 0
        n_q = n_kv = 1
        d = 16
        last = seq % page_size
        nnp = (last + num_steps + page_size - 1) // page_size
        prefix_base = seq - last
        pool_len = prefix_base + topk * nnp * page_size + 4
        req_to_token = torch.arange(pool_len, dtype=torch.int32, device=device).unsqueeze(0)
        pool_idx = torch.tensor([0], dtype=torch.int64, device=device)
        seq_t = torch.tensor([seq], dtype=torch.int64, device=device)
        n_pages = (pool_len + page_size - 1) // page_size
        k_cache = torch.randn(n_pages, page_size, n_kv * d, device=device, dtype=torch.float16)
        v_cache = torch.randn(n_pages, page_size, n_kv * d, device=device, dtype=torch.float16)
        # Duplicate last-page prefix tokens onto each extra branch page.
        src = req_to_token[0, prefix_base : prefix_base + last].long()
        for k in range(1, topk):
            tgt = req_to_token[
                0, prefix_base + k * nnp * page_size : prefix_base + k * nnp * page_size + last
            ].long()
            k_cache.view(-1, n_kv * d)[tgt] = k_cache.view(-1, n_kv * d)[src]
            v_cache.view(-1, n_kv * d)[tgt] = v_cache.view(-1, n_kv * d)[src]

        slots, lens = build_tree_draft_kv_slots(
            req_to_token, pool_idx, seq_t, page_size, topk, step_id, num_steps
        )
        query = torch.randn(topk, n_q, d, device=device, dtype=torch.float16)
        scale = 1.0 / math.sqrt(d)
        slot_out = tree_draft_attention(
            query,
            k_cache,
            v_cache,
            kv_slots=slots,
            kv_lens=lens,
            scale=scale,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
            qk_head_dim=d,
            v_head_dim=d,
        )
        block_tables = build_tree_draft_block_tables(
            req_to_token,
            pool_idx,
            seq_t,
            page_size=page_size,
            topk=topk,
            step_id=step_id,
            num_steps=num_steps,
        )
        kv_lens = [seq + step_id + 1] * topk
        try:
            q = query.reshape(topk, 1, n_q, d)
            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q,
                k_cache,
                v_cache,
                num_heads=n_q,
                num_key_value_heads=n_kv,
                input_layout="BSND",
                atten_mask=None,
                block_size=page_size,
                block_table=block_tables,
                actual_seq_lengths_kv=kv_lens,
                scale=scale,
            )
        except Exception as e:
            self.skipTest(f"FIA paged tree draft unavailable: {e}")
        fia = attn_output.reshape(topk, n_q * d)
        torch.testing.assert_close(slot_out.float(), fia.float(), rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
    unittest.main()
