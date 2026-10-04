# Shared KV movement with reusable scratch

Implementation and local validation: 2026-09-30. The baseline repository
revision inspected for this change was `c3b69acdc`. No commit or branch was
created. Remote execution and file transfer were not performed.

## Behavior and migration

All callers of the former SR shared KV copy helpers now use the new
`sr_kv_copy.py` workspace API. The old helper definitions and forwarding
wrapper have been removed; there is no environment toggle or runtime legacy
implementation. CUDA's unrelated native memory-pool copy kernels are outside
this change.

Migrated consumers:

* SR Draft historical parent remap, prefix-tail fork and prewarming.
* SR Draft lease reuse, including its dedicated copy stream.
* SR Target fixed accept for both greedy and RPD host plans.
* Target private-tensor warmup.
* `NPUMHATokenToKVPool.move_kv_cache(tgt_loc, src_loc)`. The public signature
  remains unchanged, so NPU EAGLE and N-gram callers enter the new mover too.

`sr_verify_layout.py` no longer owns KV movement. Imports and assertions in
neighboring tests were migrated to the explicit workspace API.

## Layouts and two-stage execution

CUDA extension: ordinary three-dimensional token-major layer buffers now select
the grouped `cuda_layered` backend. Pointer/stride tables and all gather groups
share the same workspace lifecycle. Other CUDA layouts retain `torch_out`.
CUDA page-size-one Target acceptance retains physical slots without KV compaction;
its independent token commit policy does not call this mover. See
[SR_CUDA_OPTIMIZATION.md](SR_CUDA_OPTIMIZATION.md) for scope and device validation.

`KVMoveLayout.from_pool()` validates the entire pool before allocating scratch
or modifying KV. It describes:

| Layout | Token axis after view | Execution |
| --- | --- | --- |
| `[2,L,pages,page_size,H,D]` | 2 in `[2,L,slots,H*D]` | NPU chunked gather-then-scatter kernel for ND FP16/BF16/FP32/byte storage; fixed-out backend on CPU/CUDA |
| NPU five-dimensional MLA K/V/index | 1 after flattening pages | Fixed-out backend |
| Per-layer K/V/index collections | 0 per token-major tensor | Fixed-out backend |
| Token-major K/V/index tensors | 0 | Fixed-out backend |
| `kv_buffer` tensor lists | 0 per token-major tensor | Fixed-out backend |

The six-dimensional buffer takes precedence over K/V alias views. Exact
aliases are deduplicated. Non-overlapping layer views in one storage are
supported; partially overlapping buffer descriptions are rejected.
FP8 storage is viewed as bytes, without numeric conversion. Incomplete K/V or
index collections, inconsistent token capacities/devices, and unsupported
layouts are errors before any KV destination write. An NPU six-dimensional
pool whose storage dtype is outside FP16, BF16, FP32, and byte raises
`UnsupportedKVMoveLayout`; it does not fall through to the portable backend.
CPU and CUDA six-dimensional pools still use that backend.

The NPU implementation is lazily imported from `sr_kv_copy_kernels.py`. It
gathers every source slot into scratch, then scatters. Each launch's program
count stays below 65536. A copy that fits is one gather and one scatter; a
longer copy is split into slot and column chunks, and every gather chunk
runs before any scatter chunk. Slot base, column base, valid count, pool slot
count and scratch capacity are ordinary runtime scalar parameters marked
`do_not_specialize`. No device parameter tensor or parameter `fill_()` remains.
Groups, contiguous width, mapping strides and mode flags retain structural
specialization. The contiguous tile has 256 elements and a tail
mask. Pointer offsets use 64-bit arithmetic. Unused pointer arguments are
replaced with a tensor from the same launch after `Tree` and `HasActive` are
taken from the original arguments. Chunk launches run on the workspace
stream; the gather/scatter boundary ensures the complete source snapshot
precedes the first destination write. No parallel in-place permutation is
used.

The portable backend performs `torch.index_select(..., out=...)` into prepared
scratch for **every** buffer, then scatters every buffer with `index_copy_()`.
Integer conversion, when required by `index_copy_`, also uses reusable
int64 staging. No gather result is allocated per move. Backend selection is
based on admitted layout/device/dtype, not a retry after kernel failure.

Repeated sources and overlapping source/target slots are legal. Live
destinations must be unique, as guaranteed by existing tree allocation and
commit plans. This interface is not an arbitrary conflicting-write resolver.
CPU calls check index bounds. Device calls consume the existing validated
mapping producers; Triton device assertions also provide diagnostic checks
when device assertion support/debugging is enabled. No device-index readback
was added to discover lengths or validate maps in the hot path.

## Parent remap and graphs

SR Draft reads historical source and target slots directly from device tensors:

```text
source = out_cache_loc[step, parent_rows[row]]
target = out_cache_loc[step, row]
```

The NPU kernel uses real slot/parent/mask strides and reloads their values on
each replay. It masks inactive rows in both stages, so padded branches do not
read/write the dummy KV slot during remap. The existing paged metadata's
`active_rows` supplies this mask; no extra host round trip is introduced.
CPU tests can execute masked remaps. Portable accelerator tree remaps use
the ordinary unmasked layout; masked paged remaps require the NPU kernel.

The final candidate selection does not launch another Draft forward. Thus
historical remap depths are `1..S-2`, and graph capacity is
`Bcap * K * max(S-2,0)`. `K<=1` and `S<=2` need no tree scratch.
Node-identity remapping is a separate two-stage gather-then-scatter on
Draft-owned scratch. It does not read `active_rows`, and the KV remap still
runs after it.

The existing Draft CUDA graph runner prepares SR-specific scratch before
capture; its NPU subclass uses that hook too. Tree-layout warmup still
enumerates page tables. Slot warmup uses the parsed production Groups,
Width, KV dtype, source/destination index dtypes and index strides with a
small private pool. Both index dtypes are part of the signature: lease copies
can have int64 sources and int32 destinations. Draft startup also warms that
lease signature; Target warms its real pool geometry. Slot
warmup no longer performs the former swap check against allocator-owned KV;
Target's allocation warmup also calls the private mover rather than writing
the live pool. Successful startup
warmup synchronizes once and releases private tensor references. Failed
warmup retains its tensors and blocks retry. Each pool records complete
signatures, separately from workspace growth, with no global warmup cache.
Tree remap uses its real mapping layout before capture. Capture scopes bind
stable workspace addresses, and replay validates KV storage identity and
stream ownership. Non-SR EAGLE graph users do not bind an SR tree workspace.

Graph counters are recorded at replay submission, not by captured Python
calls. They count the live historical slot work across all depths. Invalid
graph preparation retains the existing graph-to-eager behavior, using the
new mover in eager as well. A failure after graph replay submission poisons
the workspace and raises `KVMoveSubmittedError`, so Draft does not roll the
allocation back or retry the batch per request.

## Ownership, capacity and failures

Workspaces and the cached layout belong to the KV pool. Complete normalization,
alias validation and backend selection occur once. Steady checks inspect the
original tensor collection, identity, pointers, shape, stride, dtype and device
without building normalized views or repeating pairwise alias checks. Stacked
MLA K/V/index must agree on layers, pages and page size; differing feature widths
remain legal. Storage replacement fails explicitly, including during replay.
There is no global pointer cache. All eager
consumers on the same stream share one capacity; separate streams and named
graph domains have separate allocations. Graph allocations never grow once
bound. Eager capacity grows geometrically.

The paged scratch payload is
`2 * L * capacity * Hkv * D * element_size` bytes. The portable implementation
also owns small integer buffers. Scratch is persistent: this reduces repeated
allocation, but does not imply lower resident or peak device memory.

Growth allocates all replacement buffers transactionally. If an allocation
fails, the prior capacity remains usable. After accelerator use, growth
records a completion event and retains old scratch/index/input references
until its query confirms completion. Query/record failure preserves those
references and poisons the pool. Steady moves do not synchronize the host;
allocator stream recording protects tensor lifetimes.

Target preflights capacity/layout once per greedy or RPD round before CPU token
append and stop checking, passing that workspace through finalization/commit.
Lease copies preserve their ready/wait/done events and keep the workspace in
the transaction hold alongside source/target indices. Failure to record the
copy's completion event, a missing completion event or a failed consumer stream
wait is a submitted movement error, retaining lease pages, indices and scratch.
An unresolved lease has an explicit non-releasable flag; cleanup cannot override
it with another event. Event query failure retains pending pages and raises the
existing unresolved-transfer error before any page in that poll is freed.
TP=1 and TP=2 do not retry or roll back these cases. Safe ordinary transaction
errors retain their existing synchronize-before-rollback behavior.

Timing is separate from completion protection. `phase(..., stream=...)` records
events on the actual copy stream. Ordinary timing event creation, recording,
query and elapsed-time failures discard only the sample; business executes
once, and its original exception is preserved. Recognized device context errors
still propagate. Lease copies now pass round metrics on every stream branch.

`KVMoveSubmittedError` derives from the existing `SRTransferUnresolved`.
Submitted gather/scatter errors poison the pool and propagate through Draft
batch expansion and tail ingest without per-request retry or automatic
rollback. Future preparation rejects a poisoned pool before another token
append. Existing Target commit and Draft lease/page algorithms remain in use.

## Local checks executed

### ND allocation startup fix (2026-09-30)

The supplied TP=2 Target log fails in `warm_private_slot_move()` before the
KV mover kernel starts: its newly allocated four-dimensional private tensor
does not pass the ND check. The earlier model graph capture succeeding does
not validate this later warmup. The log does not print the actual format ID
and does not demonstrate corruption of the live pool.

NPU paged private warmup and production scratch now share `_empty_nd()`:
allocate a one-dimensional `torch_npu.empty_with_format(..., acl_format=2)`
backing, then view it as `[2,L,slots,H*D]`. Both descriptors must be ND;
shape/stride, dtype, device and pointer identity are checked before use.
Warmup initializes the one-dimensional backing only. No live KV format cast,
legacy fallback or new switch is introduced. Generic backends keep their
previous allocation behavior. This fixes the allocation contract locally;
the exact torch_npu version's real view behavior still needs hardware testing.

Initial admission checks both live KV and its normalized view. Cached source
identity now includes NPU storage format, without device synchronization or
renormalizing views. Format diagnostics identify `live_pool`,
`private_warmup` or `scratch`, actual/expected format, shape, stride, dtype
and device. All growth validation precedes replacement of the current
workspace. Scratch backings are explicitly retained with active/submitted
and retired buffers, without double-counting their aliased memory.

Checks executed for this fix, with the isolated local CPU environment below:

| Entrypoint under `test/registered/unit/spec/` | Result |
| --- | --- |
| `test_sr_kv_copy.py` | 31 passed (4 new metadata/failure tests) |
| `test_sr_fixed_accept.py` | 29 passed |
| `test_sr_rpd.py` | 19 passed |
| `test_tree_draft_kv_slots.py` | 19 passed |
| `test_sr_tree_kv_lease.py` | 30 passed |
| `test_sr_tree_paged_layout.py` | 64 passed |
| `test/manual/test_sr_kv_copy_device.py` (separate path) | 7 skipped: no NPU/CUDA |

Total: 192 CPU tests passed. Black `--check` on the three edited Python files
and `git diff --check` also passed (Git reported an existing LF/CRLF warning
for `test_sr_rpd.py`, which this fix did not edit).

Each was invoked as `python <entrypoint>` with `PYTHONPATH=python`,
`PYTHONUTF8=1`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`. Initial runs of the tree slots and paged layout tests
failed while reading source with Windows' default GBK encoding; rerunning
with UTF-8 passed. No dependency changes were needed. CPU metadata tests use
CPU-backed tensor subclasses and a mocked torch_npu factory: they exercise
admission/allocation/failure contracts, not NPU descriptor or kernel behavior.

The device suite now additionally checks real backing/view ND formats for
FP16/BF16/FP32/byte storage, private warmup isolation/cache reuse, scratch
growth after submission, fixed-capacity reuse and exact overlap results.
The existing explicit graph test rejects ND allocation during capture;
existing tree replay tests change parents, physical slots and active masks,
including B=3 padded to 4. Existing chunk tests cover cross-chunk cycles.

Hardware acceptance commands (not executed locally):

```bash
PYTHONPATH=/mnt/user/liujg/sglang/python python test/manual/test_sr_kv_copy_device.py
```

Then reproduce Target startup with the logged TP=2, BF16 KV, page size 128,
topk=3, steps=5, width=15 and RPD tau=0 configuration. Require both ranks to
finish fixed-accept warmup and serve requests. Extend to greedy/RPD,
eager/graph, B=1/2/3/4 and TP=1/2, plus CUDA layout tests, before declaring
hardware correctness. Compare an independent baseline in three alternating
groups of at least 500 warmed rounds; report startup, movement, round latency,
TPOT, throughput and resident memory separately. No hardware speedup is claimed.

The same log's roughly 181-second distributed initialization and hostname/Gloo
warnings are a separate investigation. Inspect hostname resolution, actual
interfaces and phase timings on the host; do not hardcode a network interface
or attribute the whole delay to DNS from this log alone.

### Earlier workspace implementation checks

Environment: isolated temporary Python 3.12.14 venv with PyTorch 2.8.0+cpu on
Windows. Global dependencies were not changed. Each unittest command used
`PYTHONPATH=python`, UTF-8, disabled bytecode output and one OpenMP/MKL thread.

| Existing unittest entrypoint | Result |
| --- | --- |
| `test/registered/unit/spec/test_sr_kv_copy.py` | 27 passed |
| `test/registered/unit/spec/test_sr_fixed_accept.py` | 29 passed |
| `test/registered/unit/spec/test_sr_rpd.py` | 19 passed |
| `test/registered/unit/spec/test_remote_spec_device.py` | 43 passed, 9 skipped for unavailable runtime/device dependencies |
| `test/registered/unit/spec/test_sr_tree_paged_layout.py` | 64 passed |
| `test/registered/unit/spec/test_sr_tree_kv_lease.py` | 30 passed |
| `test/registered/unit/spec/test_sr_comm_metrics.py` | 41 passed |
| `test/registered/unit/spec/test_tree_draft_kv_slots.py` | 19 passed |
| `test/registered/unit/spec/test_paged_tree_draft_cache.py` | 4 passed |
| `test_sr_tail_extend.py TestBatchedKvCopy TestTailTransaction TestTailGraphBuckets TestTreeIngestLifecycle` | 67 passed; these overlap the full file |
| Full `test/registered/unit/spec/test_sr_tail_extend.py` | 84 passed; one error importing the Linux `resource` module through an unrelated penalty test; full command exits nonzero |
| `test_standalone_remote.py TestStandaloneRemoteTree` | 12 passed, 16 dependency-related skips |
| `test/manual/test_sr_kv_copy_device.py` | All six device tests skipped: no torch_npu/NPU or CUDA |
| `test/manual/test_npu_rpd_verify.py` | All four NPU tests skipped |

Static validation includes AST parsing of all 25 affected Python files,
Ruff undefined-name checks, unused-import checks on new files, isort checks,
Black checks on the four new implementation/device-test files and formatting of changed tracked ranges, plus
`git diff --check`. Existing unused imports outside the change were preserved.

New CPU checks include swaps/cycles/fan-out, three floating dtypes and byte
storage, FP8 bits, MLA index storage, gather-before-scatter ordering, stream
isolation, graph capacity ownership, live remap/mask/stride changes, failed
growth, unresolved events, and submitted-error protection. Existing RPD tests
also check that preflight fails before append and a submitted KV failure
blocks a subsequent valid host plan without another append/page release.
The CPU chunk test executes the production launch scheduler with a snapshot
oracle, rather than compiling Triton. It verifies no scalar fills, exactly two
launches for one chunk, and all gathers preceding all scatters across chunks.
Additional tests cover cached parsing, changed storage, mixed index warmup,
per-round workspace preparation, timing faults, TP=1/2 submitted completion
errors, and scratch/retired/peak accounting including integer staging.

All table entries use the existing unittest entrypoints. The selected passing
CPU regressions contain 355 tests (full-tail results overlap that selection).
The complete tail file's missing `resource` dependency was not patched or
treated as a pass. No global dependency changes were made.

Commands executed from the repository root (the temporary interpreter already
existed; it is not a required public path):

```powershell
$py = Join-Path $env:TEMP 'codex-sr-rpd-venv/Scripts/python.exe'
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& $py test/registered/unit/spec/test_sr_kv_copy.py
& $py test/registered/unit/spec/test_sr_fixed_accept.py
& $py test/registered/unit/spec/test_sr_rpd.py
& $py test/registered/unit/spec/test_sr_comm_metrics.py
& $py test/registered/unit/spec/test_sr_tree_kv_lease.py
& $py test/registered/unit/spec/test_sr_tree_paged_layout.py
& $py test/registered/unit/spec/test_tree_draft_kv_slots.py
& $py test/registered/unit/spec/test_paged_tree_draft_cache.py
& $py test/registered/unit/spec/test_sr_tail_extend.py TestBatchedKvCopy TestTailTransaction TestTailGraphBuckets TestTreeIngestLifecycle
& $py test/registered/unit/spec/test_sr_tail_extend.py
& $py test/registered/unit/spec/test_remote_spec_device.py
& $py test/registered/unit/spec/test_standalone_remote.py TestStandaloneRemoteTree
& $py test/manual/test_sr_kv_copy_device.py
& $py test/manual/test_npu_rpd_verify.py
git diff --check
```

AST parsing and `ruff check --select F821` covered all 25 modified/new Python
files. The four new mover/device-test files additionally passed `F401`,
`black --check` and `isort --profile black --check-only`. Formatting tracked
files was limited to changed ranges. Two pre-existing unused imports in Draft
files (`lookup_candidate_slots`, `dataclasses.field`) were preserved.

## Required hardware validation

These commands are supplied for the real device environment; they were not
executed remotely:

```bash
PYTHONPATH=python python test/manual/test_sr_kv_copy_device.py -v
PYTHONPATH=python python test/manual/test_npu_rpd_verify.py -v
```

The new manual tests execute the real NPU chunked mover and pool API,
mutable graph remap with B=1/2/3/4/2/1 and B=3 padding, runtime scalar changes,
381-slot cross-chunk cycles, changing capacity and strided graph slot mappings,
cross-stream MLA/index movement, and CUDA list-layout graph replay. They require matching device
runtimes and the normal repository dependencies. They do not substitute for
full model/TP integration tests.

For performance, use an independent copy of the pre-fix scratch implementation
(the version producing the earlier device logs) to isolate these follow-up
changes. The inspected `c3b69acdc` revision is the baseline for the original
shared-mover migration, not attribution of the scalar/cache fixes alone.
No baseline branch or commit was created locally. The production implementation
has no old-path toggle. Fix model,
inputs, topk/steps, KV dtype, attention backend and other optimization settings.
Cover eager/graph, B=1/2/3/4, TP=1/2, both Target acceptance modes, and NPU
EAGLE/N-gram callers. After warmup, alternate baseline/new runs for at least
three groups of 500 steady rounds per configuration.

Record parent remap device latency, Draft expansion, complete-round latency,
TPOT/throughput, scratch allocation/growth and resident/peak memory, Draft
tree outputs, token IDs and acceptance lengths. Use device events or a
compatible trace; do not depend on `prof.key_averages()`. Service graph timing
uses existing whole-forward metrics; isolated mover timing requires the
device test or trace. The `kv_move_eager` phase samples eager moves on their
actual stream, including lease copies. It does not time all remaps captured
inside a graph.

`kv_move_calls`, `kv_move_slots`, `kv_move_bytes`, `kv_move_grow`, and
`kv_move_backend_*` join existing SR round counters. `kv_move_eager_calls` and
`kv_move_graph_calls` separate eager submissions from actual graph replay work.
Captured Python calls do not count as replays. `kv_move_scratch_bytes`,
`kv_move_retired_bytes` and `kv_move_peak_bytes` are pool workspace gauges,
including integer staging; peak is the maximum tracked scratch residency over
the pool lifetime, not total accelerator allocator peak memory. Compilation
warmup has a separate signature log/count and is not reported as workspace
growth. Allocation logs report backend, graph ownership, capacity, current,
retired and peak scratch bytes. Slot/byte
counters describe logical copied data, not all gather/scratch/scatter memory
traffic. No NPU/CUDA kernel compilation, end-to-end correctness or speedup
has been certified by the local CPU results.

Earlier NPU service performance logs preceded these scalar/cache/lifecycle fixes. They are
not hardware acceptance evidence for this revision. Real scalar argument
specialization, warmup compile coverage, graph replay and performance remain
device acceptance items; no speedup is inferred from the local checks.
