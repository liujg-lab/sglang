# ClearTree bonus-batch end-to-end result

本轮最终状态为 **PASS**；两次正式试验均观测到 bonus 批量前向实际走 graph 路径，验证了已捕获图的实际复用，而非仅有启动配置。合并吞吐相对历史 c4 A1/A2 为 +8.45%，仅为跨会话描述性比较，不是因果加速结论；历史 AA1/AA2 另列。

- Final state: **pass**.
- Analyzer runtime/configuration/batch status: **pass**.
- Completed client trials: c4-A1, c4-A2, smoke-A.
- Cleanup evidence: **pass**; services stopped=True, tunnels preserved=True.

## Scope and gates

Only bonus on, ledger off, concurrency 4, graph capture batch sizes 1/2/4. Smoke is 4 requests with maximum 128 output tokens; formal c4-A1 and c4-A2 are 100 requests each with maximum 1024 output tokens. These are generation budgets, not requirements to emit that many tokens.

Each formal trial requires batch forwards > 0, requests >= 2, graph > 0, failed = 0, and skipped_graph + skipped_kv + skipped_allocator + skipped_map <= forwards. No batch warning, device poison, OOM, traceback, protocol/fallback/timeout or other unsuppressed runtime failure is allowed.

Periodic counters are independent windows, summed rather than treated as cumulative snapshots. They include warmup and only flushed windows; the unflushed tail and benchmark-only activity are unknown. Sparse absent failure counters may be zero only with explicit parsed counter windows; no metrics is missing, not zero proof.

## Observed trials

- **c4-A1**: pass; successful requests 100/100; output tokens 65872.0; wall seconds 473.0215671341866; output tok/s 139.25792094235197; weighted acceptance 0.3186017427787639.
  Batch windows=158, forwards=4696, guess-install request occurrences=17760 (not distinct client requests), graph=4696, failed=0; skips graph/kv/allocator/map=0/0/0/0.
  TTFT: mean=200.511 ms, p99=313.128 ms, n=100.
  TPOT: mean=30.850 ms, p99=41.862 ms, n=100.
  client E2E: mean=18595.589 ms, p99=37606.966 ms, n=100.
  server E2E: mean=18593.648 ms, p99=37605.179 ms, n=100.
  Batch graph fraction graph/(graph+eager+unknown)=1.000 (100.00%).
  Mechanism counters: opt_bonus_rounds=18065, opt_bonus_hits=9106, opt_ingest_zero_fwd=9106, opt_layer_forwards_deferred=14086. opt_bonus_rounds counts eligible request attempts, not model forwards; hits and zero-ingest-forward events are not additional completed client requests.
  Round-weighted host means: opt_ingest=5.666 ms.
  opt_post_reply host_mean_ms field unavailable; derived opt_post_reply_us / metered_rounds / 1000=30.773 ms per metered round (not an observed host_mean_ms field).
  Mechanism/graph/host observations sum or weight rank-0 flushed periodic windows including warmup; unflushed tail is unknown, not isolated to client timing. Guess installs can recur for one client request; counters do not measure distinct client request count or causal savings.
- **c4-A2**: pass; successful requests 100/100; output tokens 65872.0; wall seconds 475.47815952822566; output tok/s 138.53843479447906; weighted acceptance 0.3186017427787639.
  Batch windows=158, forwards=4696, guess-install request occurrences=17760 (not distinct client requests), graph=4696, failed=0; skips graph/kv/allocator/map=0/0/0/0.
  TTFT: mean=205.184 ms, p99=340.961 ms, n=100.
  TPOT: mean=31.071 ms, p99=43.019 ms, n=100.
  client E2E: mean=18692.386 ms, p99=37928.739 ms, n=100.
  server E2E: mean=18690.459 ms, p99=37926.959 ms, n=100.
  Batch graph fraction graph/(graph+eager+unknown)=1.000 (100.00%).
  Mechanism counters: opt_bonus_rounds=18065, opt_bonus_hits=9106, opt_ingest_zero_fwd=9106, opt_layer_forwards_deferred=14086. opt_bonus_rounds counts eligible request attempts, not model forwards; hits and zero-ingest-forward events are not additional completed client requests.
  Round-weighted host means: opt_ingest=5.652 ms.
  opt_post_reply host_mean_ms field unavailable; derived opt_post_reply_us / metered_rounds / 1000=30.742 ms per metered round (not an observed host_mean_ms field).
  Mechanism/graph/host observations sum or weight rank-0 flushed periodic windows including warmup; unflushed tail is unknown, not isolated to client timing. Guess installs can recur for one client request; counters do not measure distinct client request count or causal savings.
- **smoke-A**: pass; successful requests 4/4; output tokens 386.0; wall seconds 4.093914685770869; output tok/s 94.28628333209089; weighted acceptance 0.355.
  Batch windows=4, forwards=35, guess-install request occurrences=99 (not distinct client requests), graph=35, failed=0; skips graph/kv/allocator/map=0/0/0/0.
- Prior **smoke-A**: instrumentation_abort_before_client; immutable original abort reason: RuntimeError("target: runtime failure:  error when loading sglang.srt.models.gemma4_causal: cannot import name 'Gemma4TextConfig' from 'transformers' (/usr/local/python3.11.14/lib/python3.11/site-packages/transformers/__init__.py)"). Original files remain hashed/inventoried in `summary.json`; no client result was silently replaced.
- Explicit smoke candidate: smoke-A-retry1; accepted=True. Original and retry remain separate artifact directories.

## Failed or missing evidence

- All analyzer hard gates passed on the supplied artifacts.

## Descriptive performance and outputs

- Current formal aggregate: tokens=131744.0, elapsed seconds=948.4997266624123, output tok/s=138.89724614215942. Method: sum tokens / sum elapsed; never mean of trial rates.
- A1/A2 repeat output observation: matched_saved_text. Descriptive only; no A/B or output-equivalence hard gate.
- Two-formal-trial throughput relative difference max/min - 1 = 0.5193%; descriptive repeat spread, not a confidence interval or causal noise bound.
- Token-ID completeness: available for 100 paired requests. The label matched_saved_text describes the saved-text comparison; token first-difference inspection is separately available. Saved-output agreement does not prove full request/sampling equivalence or a general correctness theorem.
- Full request payload fingerprints are unavailable; derived saved input identities are not proof of identical full prompts/sampling.
- Historical 2026-11-17-c A1_A2, trials=c4-A1,c4-A2: tokens=132930.0, elapsed=1037.8718579569831, aggregate output tok/s=128.0793953327421; relative current-minus-history=0.08446206965072892.
- Historical 2026-11-17-c AA1_AA2, trials=c4-AA1,c4-AA2: tokens=132930.0, elapsed=1088.4596118424088, aggregate output tok/s=122.12671793581083; relative current-minus-history=0.1373207148264075.
- Historical individual c4-A1: bonus=True, tokens=66465.0, elapsed=521.6079785237089, output tok/s=127.42328096305938.
- Historical individual c4-A2: bonus=True, tokens=66465.0, elapsed=516.2638794332743, output tok/s=128.7423014621158.
- Historical individual c4-AA1: bonus=True, tokens=66465.0, elapsed=540.1715207071975, output tok/s=123.04425067242238.
- Historical individual c4-AA2: bonus=True, tokens=66465.0, elapsed=548.2880911352113, output tok/s=121.22276787443357.

Historical 2026-11-17-c bonus-on c4 A1/A2 and later AA1/AA2 are separate groups, not a contemporaneous control. Cross-session load, drift, code/environment and outputs may differ; no causal speedup or equivalence claim, no hard gate.

## Cleanup and provenance

`raw/cleanup-evidence.json` inventories the archived cleanup declaration and probe files. Missing evidence is not converted to services-stopped or tunnels-preserved success. `artifacts-sha256.json` hashes the final artifacts (excluding itself and Python bytecode). No live device, process, network or deployment operation is performed by this finalizer.

Client completion, hard-gate success, and final pass are distinct: final pass also requires complete cleanup evidence and no unresolved blocked records. A fail remains fail; an incomplete or blocked run is not stamped successful.
