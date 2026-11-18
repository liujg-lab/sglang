# ClearTree dedicated bonus graph validation

Final state: **fail**.

Only c1-A1 and c4-A1: one formal trial each, 100 requests, maximum 1024 new tokens, bonus1/ledger0/KV1/lookahead0; graph capture bs1/2/3/4. No smoke, repeated trials or bonus-off control.

## Observed formal requests

- No formal client trial completed. Missing timing/replay metrics are unavailable, not zero-performance measurements. See raw/startup-diagnosis.json and preserved blocked records.

- **c1-A1**: fail; successful=0/0; actual tokens=None; client wall=Nones; output tok/s=None; accepted/draft weighted=None.
  Draft rank0 windows=0, forwards=None, installed request-groups=None, graph/eager/unknown=None/None/None, failed=None, lease_publish_failed=None.
  Skips KV/map/allocator=None/None/None; empty=None, unready=None (separate); Target verify_graph_batches=None.
  Actual graph/(graph+eager+unknown)=None; not a per-bs/G breakdown.
  ttft_ms: n=0, mean=None, p50=None, p90=None, p95=None, p99=None.
  tpot_ms: n=0, mean=None, p50=None, p90=None, p95=None, p99=None.
  client_e2e_ms: n=0, mean=None, p50=None, p90=None, p95=None, p99=None.
  server_e2e_ms: n=0, mean=None, p50=None, p90=None, p95=None, p99=None.
  post_first_delivery_ms: n=0, mean=None, p50=None, p90=None, p95=None, p99=None.

## Missing or failed gates

- global: `expected_trials` **missing**; details in checks.json.
- c1-A1: `client_log` **missing**; details in checks.json.
- c1-A1: `no_protocol_log_errors` **missing**; details in checks.json.
- c1-A1: `no_device_log_errors` **missing**; details in checks.json.
- c1-A1: `no_timeout_log_errors` **missing**; details in checks.json.
- c1-A1: `no_fallback_log_errors` **missing**; details in checks.json.
- c1-A1: `no_other_log_errors` **fail**; details in checks.json.
- c1-A1: `no_CLEAR_ledger` **missing**; details in checks.json.
- c1-A1: `summary_consistency` **missing**; details in checks.json.
- c1-A1: `wall_clock` **missing**; details in checks.json.
- c1-A1: `request_errors` **missing**; details in checks.json.
- c1-A1: `token_coverage` **missing**; details in checks.json.
- c1-A1: `request_order` **missing**; details in checks.json.
- c1-A1: `fallback_zero` **missing**; details in checks.json.
- c1-A1: `ttft_ms_coverage` **missing**; details in checks.json.
- c1-A1: `client_e2e_ms_coverage` **missing**; details in checks.json.
- c1-A1: `acceptance_coverage` **missing**; details in checks.json.
- c1-A1: `client_artifacts` **missing**; details in checks.json.
- c1-A1: `request_count` **missing**; details in checks.json.
- c1-A1: `concurrency` **missing**; details in checks.json.
- c1-A1: `completion_budget` **missing**; details in checks.json.
- c1-A1: `flush_http_success` **missing**; details in checks.json.
- c1-A1: `completed_client` **missing**; details in checks.json.
- c1-A1: `target_round_metrics` **missing**; details in checks.json.
- c1-A1: `target_no_bad_counters` **missing**; details in checks.json.
- c1-A1: `draft_round_metrics` **missing**; details in checks.json.
- c1-A1: `draft_no_bad_counters` **missing**; details in checks.json.
- c1-A1: `unified_bonus_metrics` **missing**; details in checks.json.
- c1-A1: `bonus_forwards_positive` **missing**; details in checks.json.
- c1-A1: `bonus_graph_positive` **missing**; details in checks.json.
- c1-A1: `bonus_eager_zero` **missing**; details in checks.json.
- c1-A1: `bonus_unknown_zero` **missing**; details in checks.json.
- c1-A1: `bonus_failed_zero` **missing**; details in checks.json.
- c1-A1: `bonus_skipped_map_zero` **missing**; details in checks.json.
- c1-A1: `bonus_skipped_allocator_zero` **missing**; details in checks.json.
- c1-A1: `lease_publish_failed_zero` **missing**; details in checks.json.
- c1-A1: `bonus_kv_skip_limit` **missing**; details in checks.json.
- c1-A1: `target_verify_graph_positive` **missing**; details in checks.json.
- c1-A1: `actual_configuration_parse` **missing**; details in checks.json.
- c1-A1: `target_actual_max_running_requests` **missing**; details in checks.json.
- c1-A1: `target_actual_disable_cuda_graph` **missing**; details in checks.json.
- c1-A1: `target_actual_clear_tree_role` **missing**; details in checks.json.
- c1-A1: `target_actual_page_size` **missing**; details in checks.json.
- c1-A1: `target_actual_argv_graph` **missing**; details in checks.json.
- c1-A1: `draft_actual_max_running_requests` **missing**; details in checks.json.
- c1-A1: `draft_actual_disable_cuda_graph` **missing**; details in checks.json.
- c1-A1: `draft_actual_clear_tree_role` **missing**; details in checks.json.
- c1-A1: `draft_actual_page_size` **missing**; details in checks.json.
- c1-A1: `draft_actual_argv_graph` **missing**; details in checks.json.
- c1-A1: `manifest_graph` **missing**; details in checks.json.
- c1-A1: `target_actual_graph` **missing**; details in checks.json.
- c1-A1: `target_actual_ledger` **missing**; details in checks.json.
- c1-A1: `target_actual_kv_reuse` **missing**; details in checks.json.
- c1-A1: `target_actual_lookahead` **missing**; details in checks.json.
- c1-A1: `draft_actual_graph` **missing**; details in checks.json.
- c1-A1: `draft_actual_ledger` **missing**; details in checks.json.
- c1-A1: `draft_actual_kv_reuse` **missing**; details in checks.json.
- c1-A1: `draft_actual_lookahead` **missing**; details in checks.json.
- c1-A1: `draft_actual_bonus` **missing**; details in checks.json.
- c1-A1: `target_capture_batches` **missing**; details in checks.json.
- c1-A1: `target_server_graph_bs` **missing**; details in checks.json.
- c1-A1: `target_max_running_requests` **missing**; details in checks.json.
- Preflight evidence: **pass**; formal cleanup: **pass**; trial cleanup: **missing**; formal-complete record: **missing**.
- Unresolved raw blocked records: 3. Completed client trials: none.

## Interpretation and provenance

- rank 0; sum independent flushed periodic windows over complete supplied logs including warmup; unflushed tail unknown; NOT formal-only
- No smoke, repetitions or bonus-off; c1 and c4 are different workloads, NOT repeat trials or causal A/B.
- Dedicated bonus is_bonus=True r{G}_{bs}_s0, not a normal layer graph. Unified graph counter reports actual forward result.
- Capture/width counters cannot establish per-bs/G replay; joint hit breakdown unavailable unless separately instrumented.
- Default candidates, batch1/2/3/4, one stream, token capacity256 imply 51 Draft graphs (28 ordinary +23 dedicated). This is conditional inference, not observed hits; bs4*65=260 cannot fit and is NOT required.
- requests counts installed request-group occurrences; c4 requires requests>forwards to prove at least one multi-request group. This does not prove split groups per request.
- Empty/unready skips are reported separately; only KV skips <= forwards, map/allocator skips and lease_publish_failed must be zero.
- Throughput=sum(actual output tokens)/formal elapsed client wall. Acceptance=sum accepted/sum draft, never average per-request rates.
- Full archived logs are scanned; .recent tails are never health proof. Missing evidence remains missing.
- Planned shutdown suppression needs completed-client evidence and matching actual owned SIGTERM identity. Real independent faults remain failures even after stop; optional-model imports are excluded only by exact full-line patterns.
- Finalization reads archived evidence only, never performs SSH, service cleanup or experiments. Original completed/blocked records are not changed; finalization status is separate.
- SHA256 inventory is produced after output files are written, excluding itself and Python bytecode, then every listed digest and inventory membership is rechecked. It describes the files present at finalization, not future changes.
