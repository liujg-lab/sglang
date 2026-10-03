# SR fixed-capacity integer preparation

The NPU path lazily imports `sr_small_kernels_npu.py`. CPU/CUDA use the
CPU-importable PyTorch implementation. There is no new option or retry after a
failed NPU submission. This change does not fuse KV data movement or attention.

## Data and lifetime contracts

* `pack_accept` writes the existing int64 `[B, 2L+2]` packet with one launch.
  It preserves invalid-index masking, pre-truncation lengths, and the legacy
  empty-predict error flag. CPU validation still rejects the entire invalid
  batch before appending tokens. RPD host plans never invoke this pack.
* `gather_commit_slots` now takes cache, source/destination/page logical indices,
  and page size. The caller checks logical index ranges while building the CPU
  commit packet. One kernel writes three independent int64 outputs of sizes
  `[N]`, `[N]`, `[F]`, each with zero storage offset. They never alias one another
  or the reusable commit packet. This intentionally retains up to three output
  allocations per round because destinations can outlive the round as
  `batch.out_cache_loc`. Total logical output bytes remain `(2N+F)*8`; no clones
  or intermediate gather tensors are required on NPU. Page IDs preserve order;
  the page allocator and two-phase KV mover are unchanged.
* Prefix-tail preparation stays **outside** the model graph. CPU computes
  `[base, remainder, output_offset]` for each real request and uploads one
  int64 `[B,3]` packet. One kernel reads current physical request/page mappings
  with their real strides. Ordinary allocation skips branch 0; leases include
  it. Zero remainders produce no output. CPU/CUDA retain the generic helper.
* A prefix workspace belongs to one device and stream. Its output views are
  borrowed until the next fill; the caller queues KV consumption on that stream
  first. CPU metadata reuse waits for its H2D event. Growth records a retirement
  event after existing consumers and holds old buffers until completion.
  Mapping inputs are recorded on the consumer stream. Submission/retirement
  uncertainty retains references and poisons reuse via `KVMoveSubmittedError`,
  which the Draft's existing non-retry paths already handle. Pre-allocation
  errors leave previous allocations intact.

Pack and commit kernels have no data-dependent Python launch loop. Prefix
launch count is independent of batch size; its grid handles requests, branches,
and page-tail tiles. Empty batches launch nothing. Integer metadata is not
converted through floating point.

## Warmup and observations

Target startup warms packing and commit gather with private mappings, including
int32/int64 cache slots. The actual gather outputs also pass through
`apply_free_unique_pages` (native sort and list concatenation), using private
free/release lists, F=0/1/3, N=0/1/6 and both `need_sort` modes. Successful
synchronization and result checks precede the completion log. Failures propagate
to the existing fatal startup path; live allocator lists are never borrowed.
Existing Draft layout warmup runs both ordinary and
lease allocation using the production mapping dtypes and strides, with
`commit_kv=False`; the existing warmup completion guard synchronizes before
marking coverage complete. No warmup writes live KV.

Metrics separate `fixed_accept_pack`, `fixed_accept_slot_gather`, and
`prefix_tail_plan_cpu`, `prefix_tail_metadata_h2d`,
`prefix_tail_metadata_reuse_wait`, `prefix_tail_slot_kernel`. The prefix device
timer explicitly uses its execution stream. Counts report pack/slot launches,
commit output allocation bytes, prefix workspace growth and metadata H2D bytes.
`fixed_accept_slot_output_alloc` counts nonempty allocations:
`2 * (N > 0) + (F > 0)`, not one shared allocation.
These are graph-external preparation costs, not graph-internal KV timings.
The extra prefix H2D and any reuse wait must be included in performance results.

## Validation

From the repository root, with its `python` directory on `PYTHONPATH`:

```sh
python test/registered/unit/spec/test_sr_fixed_accept.py
python test/registered/unit/spec/test_sr_tree_paged_layout.py
python test/registered/unit/spec/test_sr_rpd.py
python test/registered/unit/spec/test_sr_kv_copy.py
python test/registered/unit/spec/test_tree_draft_kv_slots.py
python test/registered/unit/spec/test_sr_tree_kv_lease.py
python test/manual/test_sr_small_kernels_device.py
```

The manual file uses standard unittest. NPU tests compare packet values against
the CPU oracle, count actual kernel launches without `prof.key_averages()`,
exercise zero/odd/strided inputs, owned output retention, mixed prefix remainders
and growth, and replay a prefix kernel while changing device metadata, request
order and physical pages (three live requests in four-row storage). CUDA tests
exercise the generic accept/commit implementation. Without hardware these tests
skip; CPU staging mocks only validate lifetime orchestration, not kernels.

Hardware acceptance additionally requires full-service greedy and RPD, eager
and graph, B=1/2/3/4, TP=1/2, early stops and changing batch order, comparing
tokens, accepted lengths, KV contents and next-round seeds against an independent
pre-change checkout. Investigate the first divergent token before attributing
differences to numerical effects.

Performance: fully warm both versions, alternate at least three groups of 500
steady rounds per configuration, and report host submission and device time,
the three preparation phases, extra H2D, allocations, total verify/postprocess,
round latency, TPOT and end-to-end throughput. Keep cold compilation separate.
The historical 0.274 ms pack microbenchmark is not a measured saving for this
change, and cannot predict RPD speedup because RPD bypasses that pack.

Implementation and checks are local only. Real NPU compilation, graph behavior,
CUDA execution and performance remain hardware acceptance items.

Local results (2026-10-01, isolated Python 3.12 / PyTorch 2.8 CPU environment):
fixed accept 32, paged layout 66, RPD 19, KV copy 31, Draft slots 19, lease 30:
**197 CPU tests passed**. The five manual hardware tests were skipped for missing
devices. UTF-8 and `PYTHONPATH=python` were set; no server connection or global
dependency changes were performed.

## NPU sort storage regression (2026-10-02)

The previous shared `[2N+F]` backing exposed a one-element page view to native
NPU sort. For N=6,F=1 its logical size was 8 bytes, storage size 104 bytes and
offset 12. This matches the supplied `aclnnSort` 8/104-byte error, followed by
graph-update failure, but the exact backend failure still needs real-device
confirmation. `contiguous()` is not a fix: this view is already contiguous.
Independent outputs remove this storage dependency while retaining one kernel.

The manual device test now includes N=6,F=1 and passes actual returned pages to
sort/list release, testing both allocator modes. A separate unittest method
runs `view`, `contiguous`, `clone`, and `independent` probes in fresh subprocesses
so a failing legacy operation cannot contaminate the next probe. Legacy failures
are printed as diagnostic evidence, not assumed to occur on every runtime;
clone is also diagnostic, and independent outputs must succeed. Each probe
synchronizes before and after sort, solely for diagnosis. Use the standard
unittest entrypoint:

```sh
python test/manual/test_sr_small_kernels_device.py TestNPUSmallKernels.test_sort_storage_variants_in_isolated_processes
python test/manual/test_sr_small_kernels_device.py TestNPUSmallKernels.test_commit_single_launch_and_owned_results
python test/registered/unit/spec/test_sr_target_tree_fia.py
```

First validate the isolated chain, then TP=2, BF16, page_size=128, topk=3,
steps=5, RPD tau=0, including cross-page release followed by the next graph
replay. Extend to the full service matrix above. No graph overlap policy,
submitted-error handling, KV mover, or retry policy is changed by this fix.

Local checks for this repair (2026-10-02):

| Executed entrypoint | Result |
| --- | --- |
| `python test/registered/unit/spec/test_sr_fixed_accept.py` | 35 passed |
| `python test/registered/unit/spec/test_sr_rpd.py` | 20 passed |
| `python test/registered/unit/spec/test_sr_target_tree_fia.py` | 27 passed |
| `python test/registered/unit/spec/test_sr_kv_copy.py` | 31 passed |
| `python test/manual/test_sr_small_kernels_device.py` | 6 skipped (no NPU/CUDA) |

The CPU storage probe for N=6,F=1 now reports separate 48/48/8-byte storages,
all with offset zero, and the expected sorted page ID. The 113 CPU tests include
private warmup sort/synchronization error propagation and a consumed RPD plan
that cannot append twice after page-sort failure. Syntax compilation, Black
checks and `git diff --check` also passed. No real-device reproduction, kernel
compilation, service graph replay, or performance result is claimed.
