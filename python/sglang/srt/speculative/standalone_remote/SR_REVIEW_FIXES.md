# SR review fixes (2026-10-09)

This change repairs the current uncommitted SR work on top of `c223c433d`.
It does not certify the original 21-item optimization proposal, restore CUDA
candidate admission, or claim an acceptance-length/performance improvement.
Startup scripts, model settings, RPD math, and KV gather/scatter order are unchanged.

## Ownership and error boundaries

- `sr_transfer_staging.owned_ints` reuses host staging by concrete device,
  stream, field and dtype. Every returned device tensor still owns a new
  allocation; this is **not device-buffer reuse**. Event creation precedes the
  copy. References are installed before attempting the copy. Copy, record or
  reuse-wait failure retains the buffers and raises `SRTransferUnresolved` with
  the cause; the slot cannot be filled again. Unpinned storage selects blocking
  copy before submission. Normal reuse queries an event and waits only if needed.
- `sr_resource_quarantine.KVQuarantine` tracks each release unit separately.
  Successful units become `done`; an attempted release that throws becomes
  `unknown` and cannot be retried. No event, or an incomplete event, does not
  authorize release. A later successful tree expansion alone proves nothing
  about the abandoned consumer. `lease_hold` comparisons use identity.
- FINISH and session wipe check unresolved ownership before request filtering,
  lease release, MM release or allocator operations. Quarantine keeps the
  request/pool index, lease and related input references, even after removing
  protocol state. Current, cumulative and reclaimed KV slot counts are distinct.
  These quarantined slots are unavailable for allocation; they are not declared
  free merely to reduce a leak counter.
- A normal session wipe releases only its SR requests if HTTP requests remain.
  A full allocator reset is permitted only without live HTTP consumers and with
  completion confirmed. After reset, quarantine records are invalidated without
  freeing their old slot numbers again. A poisoned device is not repaired by
  session rotation: references remain worker-owned until process teardown.
  Abandoned records without a trustworthy completion event are intentionally
  retained, rather than reclaimed by guessing completion.
- The Triton SR eager metadata workspace is not in this working tree.
  A–D acceptance-length ablation has not been run, so B=1 graph acceptance
  length is unknown. Do not attach FlashInfer or NPU to that path until the
  ablation is done in a separate change.

## Candidate, allocation and protocol corrections

- Candidate preflight preserves `SRTransferUnresolved` and device-context errors.
  Ordinary capability/semantic declines can fall back only after completion is
  confirmed. The worker retains a failed workspace. Graph capture carries an
  explicit choice, including `None` for the original algorithm; that choice must
  never trigger an eager-workspace lookup. CUDA continues using the original
  production candidate path pending independent graph/eager A/B evidence.
- Commit slot outputs are exact-length, independently owned tensors. The unused
  capacity buffers and clone-based pseudo-reuse are removed. The nonempty output
  allocation count is `2*(N>0)+(F>0)`, and bytes remain `(2*N+F)*8`. In particular,
  N=6/F=1 means three allocations and 104 bytes, not zero allocations. Independent
  storage protects the downstream NPU sort contract. Partial-finish compaction
  still uses the existing selection/cat implementation; no fused compaction or
  allocation-free output is claimed.
- NPU `alloc_extend` retains CPU page counting, restores the existing small-page
  device kernel and large-page branch, and returns `None` on insufficient pages.
  The remaining free-page list stays a view; it is not cloned on every allocation.
  Exact-storage treatment remains at sort consumption boundaries. CPU tests use
  the actual host dispatch methods with a launch oracle, not the Ascend kernel.
- Stage 5 binary STEP/REPLY control (`SRB2`, `SGLANG_SR_BINARY_CONTROL`,
  `sr-control-v2`) was reverted. STEP and REPLY are pickled again. PREFILL and
  snapshots stay pickled. Cross-round pipelining was not implemented.

## Local validation

Use the existing unittest entry points from the repository root. On this Windows
host the interpreter is
`C:\Users\liujianguo\AppData\Local\Temp\codex-sr-metadata-tests\Scripts\python.exe`
(CPU PyTorch 2.8.0), with `PYTHONUTF8=1` and `PYTHONPATH` containing `python` plus
`C:\Users\liujianguo\AppData\Local\Temp\codex-sr-metadata-test-deps`.

```sh
python test/registered/unit/spec/test_sr_resource_guards.py
python test/registered/unit/spec/test_sr_npu_alloc_extend.py
python test/registered/unit/spec/test_sr_rpd.py
python test/registered/unit/spec/test_sr_fixed_accept.py
python test/registered/unit/spec/test_sr_tree_paged_layout.py
python test/registered/unit/spec/test_remote_spec_device.py
python test/registered/unit/spec/test_sr_tree_kv_lease.py
python test/registered/unit/spec/test_sr_kv_copy.py
python test/registered/unit/spec/test_tree_shared_prefix.py
python test/registered/unit/spec/test_sr_tail_extend.py
python test/registered/unit/spec/test_sr_incremental_commit.py
python test/registered/unit/spec/test_standalone_remote.py
python -c "import torch, runpy; runpy.run_path('test/registered/unit/spec/test_sr_comm_metrics.py', run_name='__main__')"
git diff --check
```

The metrics entry point preloads torch before its mock/module-isolation tests;
the first direct invocation on this host failed from a partially re-imported
torch (`_has_torch_function` already has a docstring). Preloading does not mock
the implementation under test or change the unittest runner.

New fault injection covers copy/record/wait failures, device/stream isolation,
partial allocator release, poisoned FINISH, candidate fatal propagation and
explicit graph decline, kernel writes followed by an exception, failed cross-
stream waits, negotiated compatibility, malformed frames and integer boundaries.
Mocked device operations and AST-extracted host methods are orchestration tests,
not hardware correctness evidence. The obsolete exact exception-handler text
checks are replaced by the existing actual-handler identity/no-retry tests.

Local dependency limitations remain explicit: the full fixed-accept allocator
import needs Triton; tail/incremental tests import Linux `resource`; two transport
tests use IPC, unsupported by this Windows libzmq. Re-running the IPC tests outside
the sandbox confirmed `Protocol not supported`, after the sandbox initially
reported permission denial. No dependency versions were changed.

Final local results (counts exclude earlier failed/repeated invocations):

| Test file | Passed | Skipped | Environment-blocked errors |
| --- | ---: | ---: | ---: |
| `test_sr_resource_guards.py` | 21 | 0 | 0 |
| `test_sr_npu_alloc_extend.py` (CPU host dispatch/oracle) | 5 | 0 | 0 |
| `test_sr_rpd.py` | 27 | 0 | 0 |
| `test_sr_tree_paged_layout.py` | 66 | 0 | 0 |
| `test_remote_spec_device.py` | 52 | 10 | 0 |
| `test_sr_tree_kv_lease.py` | 42 | 0 | 0 |
| `test_sr_kv_copy.py` | 38 | 0 | 0 |
| `test_tree_shared_prefix.py` | 31 | 0 | 0 |
| `test_sr_comm_metrics.py` (torch preloaded) | 44 | 0 | 0 |
| `test_sr_fixed_accept.py` | 35 | 0 | 1 (Triton) |
| `test_sr_tail_extend.py` | 91 | 0 | 1 (`resource`) |
| `test_sr_incremental_commit.py` | 14 | 1 | 4 (`resource`) |
| `test_standalone_remote.py` | 100 | 70 | 2 (Windows IPC) |

Total: 566 passed, 81 skipped, 8 environment-blocked errors in the run above,
after removing `test_sr_draft_metadata.py` with the workspace. A skipped or blocked
case is not a passed hardware check. The full
`stage-a-test-cpu` suite was not run; these are the affected-file entry points.

## Hardware acceptance still required

Start with the same Target/Draft versions and short deterministic requests. Check
full token IDs, accepted paths, KV contents, stop truncation and next seeds before
performance. Cover CUDA/NPU, eager/graph, B=1/2/3/4 (including 3-to-4 padding),
greedy/RPD and TP=1/2. Validate real event faults and allocator kernels separately.
Then alternate an independent baseline and each isolated optimization for at
least three warmed groups of 500 rounds. Include host waits, device time, whole
round latency, acceptance length and resident memory. No real-device compilation,
model inference or end-to-end speedup was measured in this local repair.
