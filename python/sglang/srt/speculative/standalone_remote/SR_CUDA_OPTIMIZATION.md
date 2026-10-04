# SR CUDA workspaces and acceptance

This implementation adds CUDA acceptance, layered KV copying, candidate storage
and eager attention metadata reuse. Protocol, CLI, experiment scripts, model
quantization and attention arithmetic are unchanged. No new switch is added.

## Target acceptance

CUDA admission covers ordinary `MHATokenToKVPool` and `TokenToKVPoolAllocator`,
`page_size=1`, top-k greater than one, and existing capacity limits. Contiguous
token-major FP16/BF16/FP32 K/V must share a device and token capacity. Grammar,
logprob, hidden/hybrid consumers, custom processors and simulated acceptance
choose V1 before verification. NPU's existing switch and paged policy are separate.

RPD reuses normalized CPU topology, packet generation/order identity and edge
indices uploaded with the input packet. `torch.max` and original-dtype edge
gather are retained. Integer argmax and edge logits have separate pinned buffers:
two D2H copies, one completion wait. No-edge input copies argmax only; empty input
does neither. Pure host selection retains Python float conversion/accumulation,
tau rules, sibling order and repeated prediction positions. Its host plan enters
stop processing directly, without intermediate accept result upload/readback.

Greedy uses a single CUDA `[B,2L+2]` packing kernel and existing accept readback.
Both modes validate whole-batch paths and request mapping before appending tokens.
After CPU stopping/truncation, kept indices, their complements, tokens, counts
and unfinished rows share one final control H2D. One slot kernel produces two
independently owned physical-slot arrays. Original mapping assignment and
`allocator.free()` follow. **Accepted physical KV slots are retained without
compacting KV.** Publishing preserves paths, next-round seeds, reasoning state,
pre-truncation statistics and the `A`/`A-1` convention. Page-size-one updates all
sequence lengths even when all requests finish, as V1 does.

Unknown submission completion retains packet, batch and output references and
poisons reuse. Admission/finalization refuse another attempt; exceptions do not
select V1 after submission or cause a second append. Normal execution adds no
device-global synchronization.

Greedy buffer binding rejects unsupported capacity, shape, dtype, device or
contiguity with `FixedAcceptBufferUnsupported` before initialization. Only this
metadata rejection may select V1. Once initialization attempts its first device
write, a failure retains the views, marks acceptance unresolved and propagates
`SRTransferUnresolved` with the original cause (existing unresolved errors and
interrupts retain their identity). The caller must not classify arbitrary
`RuntimeError` as noncontiguous input. This protection also applies to NPU;
successful binding adds no event or synchronization.

## Layered CUDA KV copying

The unified workspace selects `cuda_layered` for three-dimensional axis-zero
token-major tensors. Other supported layouts use the existing `torch_out`
workspace. Unrelated CUDA memory-pool native kernels remain unchanged.
Complete layout/alias validation precedes scratch allocation. CUDA caches GPU
pointer/stride tables and groups compatible raw storage widths (1/2/4/8 bytes).
Copying raw bits preserves dtype, NaN payloads and signed zero.

Groups batch layers, K/V, slots and feature tiles. **Every group and grid chunk
gathers before any scatter.** Runtime tree parents, physical slots and active
masks are read on replay; no historical source-index temporary is constructed.
Identity slots mask both phases and scratch accesses. Repeated sources are legal;
active destinations must be unique according to the caller's allocation rules.

Scratch and pointer tables belong to device/stream/graph domains. Graph addresses
freeze before capture. Eager growth retains old tables/buffers until consumer
events complete. Failed table upload or kernel submission retains partial
resources and raises `KVMoveSubmittedError`. Resident metrics include pointer
tables. Logical-byte counts do not subtract identity masking, which also does
not shrink resident scratch. Graph counters reflect actual replay, not capture.

## Draft candidates and Triton metadata

Candidate workspaces use concrete CUDA devices and explicit graph/eager choices.
Each step owns independent tables; final concatenation uses preallocated regions.
CUDA retains compiled `select_top_k_tokens()` rather than splitting it into the
NPU out-operator sequence. Softmax/top-k dtype, score multiplication order and
final selection/sorting are unchanged. Completed numerical preflight may retain
allocating probability operators or decline the workspace. Launch/sync errors
retain resources and never select a retry. See `SR_TREE_CANDIDATES.md`.

Only SR CUDA Triton attention with page size one uses eager metadata reuse.
Each stream owns geometric capacity and disjoint step segments retained by child
backends without clones. Growth protects previous consumers. Graph metadata
has fixed buffers, retained input references and unresolved-state refusal.
SR generates `S-1` rows with **explicit branch stride `S`**; reducing the grid
without that separate parameter would read the wrong branch slots. Other
consumers keep the generator's old default and existing layout.

## Local checks

Use `PYTHONPATH=python` and the existing entrypoints:

```bash
python test/registered/unit/spec/test_sr_fixed_accept.py
python test/registered/unit/spec/test_sr_rpd.py
python test/registered/unit/spec/test_sr_kv_copy.py
python test/registered/unit/spec/test_remote_spec_device.py
python test/registered/unit/spec/test_sr_comm_metrics.py
python test/registered/unit/spec/test_tree_replay_plan.py
python test/registered/unit/spec/test_sr_tree_kv_lease.py
python test/registered/unit/spec/test_sr_target_tree_fia.py
python test/registered/unit/spec/test_sr_paged_metadata.py
python test/registered/unit/spec/test_sr_tree_paged_layout.py
python test/manual/test_sr_kv_copy_device.py
```

Windows source/AST tests need `PYTHONUTF8=1`. Local execution uses a private CPU
environment without changing global dependencies. Fault injection covers batch
validation, no second append/pack, retirement query failures, graph metadata
exceptions, retained inputs and candidate capture ownership. Hardware cases
skip locally. Ruff error checks, syntax checks and `git diff --check` also run.

The final local run passed 416 tests in the ten registered files, with 10
hardware-only skips. The manual device file skipped all 21 tests on CPU.
An earlier invocation used the nonexistent name `test_sr_tree_replay_plan.py`;
it was corrected to the existing `test_tree_replay_plan.py` (54 passed).
Initial source-reading failures without `PYTHONUTF8=1` were also corrected;
neither initial environment/command error is counted as a passing check.

## RTX 3090 component validation

The subsequent greedy binding lifecycle fix reran `test_sr_fixed_accept.py`
(46 passed), `test_sr_rpd.py` (27 passed), `test_remote_spec_device.py`
(62 passed, 10 skipped), `test_tree_replay_plan.py` (54 passed), and
`test_sr_comm_metrics.py` (44 passed), using the same local entrypoints above:
233 passed and 10 hardware-only skips. Added CPU fault injection exercises the
actual Eagle caller branch, first/second initialization failures, retained
references, refusal to retry, original exception/cause, metadata-only fallback,
and empty/successful bindings. Syntax, Ruff error checks and `git diff --check`
passed. This fix was not rerun on CUDA/NPU hardware; the component results below
belong to the earlier implementation validation.

The manual file's `TestCUDAKVMove` and `TestCUDASRPaths` were loaded with current
local sources into a private process in the existing container: **10 tests passed**
in 21.079 seconds, PyTorch 2.9.1+cu128, Triton 3.5.1, RTX 3090. No server project,
dependencies or startup configuration were modified; compiler caches used a
temporary directory and were cleaned after synchronization.

Coverage includes FP16/BF16/FP32 full-vocabulary candidates, ties and NaN/Inf;
greedy/host token-slot commits and stop/finish states; RPD tau=0/0.2/0.5;
strided KV, scratch growth and a 65,537-slot cross-chunk cycle; dynamic graph
parents/slots/active masks with padded rows; and actual backend eager/graph
metadata binding/reuse across B=1/2/3/4. Dtypes reset Dynamo guards in the private
test only, preventing test-only accumulated guards from causing eager fallback.
Production compiler settings are unchanged. This is component validation, not
a model-service or TP numerical/performance result.

## Outstanding end-to-end acceptance

Align both ends' K/S/W before comparing 2/4/8 and 3/5/15 as distinct configurations.
Use identical models, tokenizers, cache state and request order against an
independent baseline. Cover eager/graph, B=1/2/3/4 (especially 3-to-4 padding),
greedy/RPD and TP=1/2. Compare complete output tokens, accepted paths, committed
KV and next-round seeds. After warmup alternate three groups of at least 500
rounds between versions; record host/device stages, round latency, TPOT,
throughput, acceptance length, allocation counts and resident memory.

Candidate copies may offset allocation savings, and CUDA graphs already reuse
capture allocations. Report eager allocations and graph execution separately.
RPC/D2H waiting includes preceding compute, not just transport. No throughput
or memory-peak improvement is claimed without this service benchmark.
