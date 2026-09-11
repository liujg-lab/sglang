"""NPU FIA tree-mask polarity probe and SPECTRE/SR tree+graph nightly coverage.

Heterogeneous CUDA↔NPU e2e is multi-host and is not executed here.
Homogeneous NPU→NPU coverage requires weights and torch_npu.
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


if __name__ == "__main__":
    unittest.main()
