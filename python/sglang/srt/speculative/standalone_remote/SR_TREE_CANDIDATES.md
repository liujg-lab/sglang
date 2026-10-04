# SR candidate workspace: preflight and capture ownership

The NPU/CUDA SR candidate workspace preserves the original softmax, top-k, dtype,
score multiplication and final ordering. This document describes its failure
and capture contracts; it does not certify complete model or performance results.

CUDA keeps `select_top_k_tokens()` compiled rather than replacing its fused
selection with the NPU `out=` arithmetic sequence. Its results are copied into
step-specific tables. Probability and final top-k/sort/gather `out=` operations
are checked with private data before use. A completed numerical mismatch may
decline the workspace; uncertain device errors never retry another implementation.
CUDA graph ownership binds the replay stream on first use and rejects a change.
Capture-time allocations are not counted as allocations on every replay.

## Preflight

Private numerical comparisons run before capture or live candidate execution.
Only comparisons whose device work has completed may select the original
operators. If both variants complete but differ from the reference tree,
`CandidatePreflightDeclined` selects the original assembly for that domain.

An operator exception is not evidence of an unsupported `out=` signature before
submission. Launch, comparison or synchronization errors on the accelerator mark
the workspace unresolved and propagate `SRTransferUnresolved` with the original
cause (an existing unresolved exception is preserved). They never select another
operator or retry preflight. Interrupts retain their original type.

The owner installs the workspace before preflight. Uncertain failures keep it,
its private input references and buffers alive; subsequent admission is refused.
No normal per-round synchronization is added. Preflight completion waits remain
startup/first-shape work.

## Capture decisions

`candidate_capture_scope(worker, workspace)` binds a prepared decision for the
whole capture forward, including `workspace=None` for original assembly. The
separate prepared flag distinguishes this from an ordinary eager forward.
`draft_candidate_workspace()` must not consult the eager factory while that flag
is set. Scope exit, including exceptional exit, restores the previous decision.

Graph preparation and replay keep their fixed workspace ownership and storage
checks. A safe graph preflight decline does not allocate an eager workspace or
run another preflight inside graph capture.

## Device identity and verification

Unindexed accelerator names resolve to the current device before allocation and
stream binding. Explicit device indices remain unchanged. Input checks still
reject tensors from another device. NPU tests use explicit device indices and
include a separate unindexed-name regression.

Run the existing entries with `PYTHONPATH=python`:

```bash
python test/registered/unit/spec/test_remote_spec_device.py
python test/registered/unit/spec/test_tree_replay_plan.py
python test/manual/test_sr_kv_copy_device.py
```

CPU fault injection covers launch/wait failures, original exception identity,
owner retention, refusal to reuse, completed numerical fallback, capture decline
and device metadata. It cannot certify Triton/Ascend compilation or graph replay.
NPU acceptance still requires FP16/BF16/FP32, eager/graph, B=1/2/3/4 (including
3-to-4 padding) and TP=1/2. Compare complete candidate trees and output tokens
against an independent baseline before measuring throughput.
