#!/usr/bin/env python3
"""Compare NPU ``build_tree_kernel_efficient`` FULL_MASK output to the CUDA-faithful ref.

No pytest. Run on an Ascend host:

    PYTHONPATH=python python scripts/playground/check_npu_tree_mask.py

Exit 0 if NPU matches the reference (or if torch_npu is missing, after printing
the CPU reference self-check). Exit 1 on a layout mismatch.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_case_mod():
    path = ROOT / "test" / "registered" / "unit" / "spec" / "test_build_tree_ref.py"
    spec = importlib.util.spec_from_file_location("test_build_tree_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _npu_available() -> bool:
    try:
        import torch

        return bool(getattr(torch, "npu", None) and torch.npu.is_available())
    except Exception:
        return False


def _seq_list(seq_lens):
    return [int(x) for x in seq_lens.detach().reshape(-1).tolist()]


def _report_mismatch(name, got, ref):
    got_l = got.detach().to("cpu").tolist()
    ref_l = ref.detach().to("cpu").tolist()
    if got_l == ref_l:
        print(f"  OK  {name}")
        return False
    print(f"  FAIL {name}")
    print(f"    got={got_l}")
    print(f"    ref={ref_l}")
    return True


def _compare_one(seq_lens_dtype, case_mod):
    import torch

    from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient
    from sglang.srt.speculative.tree_verify import (
        build_tree_kernel_efficient_ref,
        first_full_mask_mismatch,
    )

    case = case_mod.known_eagle_tree_tensors("npu")
    seq_lens = case["seq_lens"].to(dtype=seq_lens_dtype)
    seq_list = _seq_list(seq_lens)
    draft = case["num_draft_token"]
    print(f"seq_lens dtype={seq_lens_dtype} device={seq_lens.device} values={seq_list}")

    ref_mask, ref_pos, ref_idx, ref_nxt, ref_sib = build_tree_kernel_efficient_ref(
        case["parent_list"],
        case["top_scores_index"],
        seq_lens,
        case["topk"],
        case["depth"],
        draft,
    )
    got_mask, got_pos, got_idx, got_nxt, got_sib, _ = build_tree_kernel_efficient(
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

    failed = False
    failed |= _report_mismatch("positions", got_pos, ref_pos)
    failed |= _report_mismatch("retrive_index", got_idx, ref_idx)
    failed |= _report_mismatch("retrive_next_token", got_nxt, ref_nxt)
    failed |= _report_mismatch("retrive_next_sibling", got_sib, ref_sib)

    mismatch = first_full_mask_mismatch(got_mask, ref_mask, seq_list, draft)
    if mismatch is None:
        print("  OK  tree_mask")
    else:
        failed = True
        b, t, c, got_v, ref_v, got_row, ref_row = mismatch
        print(
            f"  FAIL tree_mask at (batch={b}, row={t}, col={c}): "
            f"got={got_v} ref={ref_v}"
        )
        print(f"    got_row={got_row}")
        print(f"    ref_row={ref_row}")
        print(f"    got_visible={[i for i, v in enumerate(got_row) if v]}")
        print(f"    ref_visible={[i for i, v in enumerate(ref_row) if v]}")
        prefix_ok = all(got_row[: seq_list[b]]) if b >= 0 else False
        if b >= 0 and not prefix_ok:
            print("    hint: prefix columns are not all True (prefill/layout)")
        if b >= 0 and len(got_row) != seq_list[b] + draft:
            print(
                f"    hint: row length {len(got_row)} != seq_len+draft "
                f"{seq_list[b] + draft} (stride used draft instead of seq_len+draft?)"
            )
        if b >= 0 and got_row and ref_row and got_row == [not x for x in ref_row]:
            print("    hint: polarity is inverted")
    return failed


def _cpu_self_check(case_mod):
    from sglang.srt.speculative.tree_verify import build_tree_kernel_efficient_ref

    case = case_mod.known_eagle_tree_tensors("cpu")
    _, positions, retrive_index, retrive_next_token, retrive_next_sibling = (
        build_tree_kernel_efficient_ref(
            case["parent_list"],
            case["top_scores_index"],
            case["seq_lens"],
            case["topk"],
            case["depth"],
            case["num_draft_token"],
        )
    )
    checks = [
        ("positions", positions.tolist(), case_mod.EXPECTED_POSITIONS),
        ("retrive_index", retrive_index.tolist(), case_mod.EXPECTED_RETRIVE_INDEX),
        (
            "retrive_next_token",
            retrive_next_token.tolist(),
            case_mod.EXPECTED_RETRIVE_NEXT_TOKEN,
        ),
        (
            "retrive_next_sibling",
            retrive_next_sibling.tolist(),
            case_mod.EXPECTED_RETRIVE_NEXT_SIBLING,
        ),
    ]
    failed = False
    for name, got, exp in checks:
        if got != exp:
            print(f"  FAIL CPU ref {name}: got={got} expected={exp}")
            failed = True
        else:
            print(f"  OK  CPU ref {name}")
    return failed


def main() -> int:
    case_mod = _load_case_mod()
    print("CPU reference self-check (known eagle tree golden outputs)")
    if _cpu_self_check(case_mod):
        return 1

    if not _npu_available():
        print(
            "torch_npu is not available; skip NPU parity. "
            "Re-run this script on an Ascend host."
        )
        return 0

    import torch

    print(f"NPU parity against reference (device={torch.npu.current_device()})")
    failed = False
    failed |= _compare_one(torch.int64, case_mod)
    failed |= _compare_one(torch.int32, case_mod)
    if failed:
        print(
            "NPU FULL_MASK output diverges from the CUDA-faithful reference. "
            "If only tree_mask differs, fix eagle_utils.py _is_npu (prefill/layout). "
            "If positions/retrive_* differ, file against sgl_kernel_npu."
        )
        return 1
    print("NPU FULL_MASK matches the CUDA-faithful reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
