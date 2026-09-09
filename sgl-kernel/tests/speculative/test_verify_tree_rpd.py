import math
import sys

import pytest
import torch
from sgl_kernel import verify_tree_rpd
from sglang.srt.speculative.rpd_verify import (
    rpd_gap_max,
    verify_tree_rpd as verify_tree_rpd_py,
)


def _branching_tree(device="cpu"):
    n = 7
    candidates = torch.tensor([[0, 1, 3, 4, 2, 5, 6]], dtype=torch.int64, device=device)
    retrive_index = torch.arange(n, dtype=torch.int64, device=device).unsqueeze(0)
    retrive_next_token = torch.tensor(
        [[1, 2, 3, -1, 5, 6, -1]], dtype=torch.int64, device=device
    )
    retrive_next_sibling = torch.tensor(
        [[-1, 4, -1, -1, -1, -1, -1]], dtype=torch.int64, device=device
    )
    logits = torch.zeros(n, 8, dtype=torch.float32, device=device)
    logits[0, 1] = 10.0
    logits[0, 2] = 9.85
    logits[1, 3] = 10.0
    logits[2, 7] = 10.0
    logits[4, 5] = 10.0
    logits[5, 6] = 10.0
    logits[6, 7] = 10.0
    return candidates, retrive_index, retrive_next_token, retrive_next_sibling, logits


def _chain_tree(device="cpu"):
    n = 4
    candidates = torch.tensor([[9, 10, 11, 12]], dtype=torch.int64, device=device)
    retrive_index = torch.arange(n, dtype=torch.int64, device=device).unsqueeze(0)
    retrive_next_token = torch.tensor([[1, 2, 3, -1]], dtype=torch.int64, device=device)
    retrive_next_sibling = torch.full((1, n), -1, dtype=torch.int64, device=device)
    logits = torch.zeros(n, 16, dtype=torch.float32, device=device)
    logits[0, 10] = 5.0
    logits[1, 11] = 5.0
    logits[2, 12] = 5.0
    logits[3, 1] = 5.0
    return candidates, retrive_index, retrive_next_token, retrive_next_sibling, logits


def _run_cpu(tensors, tau):
    candidates, retrive_index, retrive_next_token, retrive_next_sibling, logits = tensors
    tot = int(retrive_index.max().item()) + 1
    bs, n = candidates.shape
    predicts = torch.full((tot,), -1, dtype=torch.int32)
    accept_index = torch.full((bs, n), -1, dtype=torch.int32)
    accept_token_num = torch.zeros((bs,), dtype=torch.int32)
    return verify_tree_rpd_py(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        logits,
        tau,
    )


def _run_cuda_kernel(tensors, tau, dtype=torch.float32):
    candidates, retrive_index, retrive_next_token, retrive_next_sibling, logits = [
        t.cuda() for t in tensors
    ]
    logits = logits.to(dtype)
    tot = int(retrive_index.max().item()) + 1
    bs, n = candidates.shape
    predicts = torch.full((tot,), -1, dtype=torch.int32, device="cuda")
    accept_index = torch.full((bs, n), -1, dtype=torch.int32, device="cuda")
    accept_token_num = torch.zeros((bs,), dtype=torch.int32, device="cuda")
    z_star = logits.amax(dim=-1)
    target_predict = logits.argmax(dim=-1).to(torch.int64)
    verify_tree_rpd(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        logits,
        z_star,
        target_predict,
        float(rpd_gap_max(tau)),
        float(tau) == 0.0,
    )
    return predicts, accept_index, accept_token_num


@pytest.mark.parametrize("tau", [0.0, 0.2])
def test_verify_tree_rpd_matches_cpu_branching(tau):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cpu_out = _run_cpu(_branching_tree("cpu"), tau)
    cuda_out = _run_cuda_kernel(_branching_tree("cpu"), tau)
    for c, g in zip(cpu_out, cuda_out):
        torch.testing.assert_close(c.cpu(), g.cpu())


def test_verify_tree_rpd_tau_zero_chain():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cpu_out = _run_cpu(_chain_tree("cpu"), 0.0)
    cuda_out = _run_cuda_kernel(_chain_tree("cpu"), 0.0)
    for c, g in zip(cpu_out, cuda_out):
        torch.testing.assert_close(c.cpu(), g.cpu())
    assert cuda_out[2].tolist() == [3]


def test_verify_tree_rpd_bf16_longest_path():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cpu_out = _run_cpu(_branching_tree("cpu"), 0.2)
    cuda_out = _run_cuda_kernel(_branching_tree("cpu"), 0.2, dtype=torch.bfloat16)
    assert cuda_out[2].tolist() == cpu_out[2].tolist()
    assert cuda_out[1][0, :4].tolist() == cpu_out[1][0, :4].tolist()


def test_verify_tree_rpd_python_cuda_dispatch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    tensors = [t.cuda() for t in _branching_tree("cpu")]
    tot = int(tensors[1].max().item()) + 1
    bs, n = tensors[0].shape
    predicts = torch.full((tot,), -1, dtype=torch.int32, device="cuda")
    accept_index = torch.full((bs, n), -1, dtype=torch.int32, device="cuda")
    accept_token_num = torch.zeros((bs,), dtype=torch.int32, device="cuda")
    verify_tree_rpd_py(
        predicts,
        accept_index,
        accept_token_num,
        *tensors,
        0.2,
    )
    cpu_out = _run_cpu(_branching_tree("cpu"), 0.2)
    assert accept_token_num.tolist() == cpu_out[2].tolist()
    assert accept_index[0, :4].tolist() == cpu_out[1][0, :4].tolist()


def test_gap_max_formula():
    assert rpd_gap_max(0.0) == 0.0
    assert math.isclose(rpd_gap_max(0.2), -math.log(0.8), rel_tol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
