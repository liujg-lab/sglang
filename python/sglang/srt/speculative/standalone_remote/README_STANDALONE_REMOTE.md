# STANDALONE_REMOTE

跨机 **同步** 投机解码。Draft 与 Target 跑在两个 SGLang 进程里。评测 HTTP 应打 Target。
Draft 自己的 HTTP 口也可以做普通 generate，与 Target RPC **分时**（有 RPC 先处理 RPC；
ZMQ 空闲时才跑 HTTP）。每个 decode 步是一次阻塞批量 RPC：Target 等 Draft token 再核验；
Draft 等下一段已提交前缀再生成。

首 token 的 PREFILL 与 Target GPU prefill 重叠（先 send，再 GPU，再 recv）。
收包后，若 ``topk=1``，Target 会丢掉 extend 阶段已经采样过的前缀 draft token
（通常 ``D0 == T0``），verify 从 ``D1`` 开始。若 ``topk>1``，PREFILL 只填 Draft KV，
第一棵树在 ``T0`` 之后的 STEP 才到。Decode STEP 始终是阻塞 send+recv。

本模式 **不是** SPECTRE：无流水线、无乱序 mm 旁路、无 C++ ZMQ 扩展。
传输用 pyzmq DEALER/ROUTER。同步 RPC 上有 Draft 状态 TTL、连续超时熔断
（跳过 STEP，退回 1-token AR），以及 Draft 忙时快速 REJECT。

## 启动

两个进程。投机超参（`--speculative-num-steps`、`--speculative-eagle-topk`、
`--speculative-num-draft-tokens`）两侧必须 **完全一致**。先起 Target，再起 Draft。
词表 / tokenizer 必须对齐（协议传的是 token id）。

评测 HTTP 仍应打 **Target**。Draft 也会在自己的 HTTP 口提供普通 `/generate`
（与 Target RPC 分时：有 RPC 优先 RPC，ZMQ 空闲才跑 HTTP）。
若在意投机延迟，不要把评测流量打到 Draft。

| 进程 | `--standalone-remote-role` | 作用 |
|---|---|---|
| Target | `target` | 对外 `/generate`、核验、连接 Draft RPC |
| Draft | `draft` | 绑定 ZMQ RPC 端口，生成链或树草稿。可选 HTTP generate 与 RPC 分时共享 GPU |

### 链（`topk=1`，默认）

```bash
# Target — HTTP :30000，RPC 客户端 → :30019
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /path/to/Qwen3-VL-8B \
  --speculative-algorithm STANDALONE_REMOTE \
  --standalone-remote-role target \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --standalone-remote-addr 127.0.0.1 \
  --standalone-remote-port 30019 \
  --page-size 1 --skip-server-warmup \
  --port 30000

# Draft — RPC 服务 :30019（不要 --skip-tokenizer-init）
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path /path/to/Qwen3-VL-2B \
  --speculative-algorithm STANDALONE_REMOTE \
  --standalone-remote-role draft \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --context-length 32768 \
  --standalone-remote-addr 127.0.0.1 \
  --standalone-remote-port 30019 \
  --page-size 1
```

Draft 的 HTTP warmup 可以跑完（空闲循环会跑普通 generate）。
Draft 上 `--skip-server-warmup` 可选。

`topk=1` 会强制 `num-draft-tokens = steps + 1`。同模型 greedy 时
`accept len` 应接近 5，`accept rate` 接近 1.0。

### 树（`topk>1`）

启动命令相同，但 **两侧** 的树超参和 `page-size 1` 必须一致：

```bash
  --speculative-num-steps 4 \
  --speculative-eagle-topk 2 \
  --speculative-num-draft-tokens 8 \
  --page-size 1
```

Draft 日志：`[SR] Draft scheduler ready (tree=True topk=2)`。
`page_size>1` 且 `topk>1` 会在启动时失败。

### 网络

`127.0.0.1` / `0.0.0.0` 走 IPC，其它地址走 TCP。Draft **bind** RPC 端口，
Target **connect**。跨机时，Target 的 `--standalone-remote-addr` 填 Draft 主机 IP，
两侧端口相同。

评测：`POST http://<target-host>:30000/generate`。Draft HTTP 是可选双通道，
GPU 时间与投机 RPC 共享。

## 容错（同步 RPC）

这些都是 SR 自己的同步路径，不是 SPECTRE 的异步缓冲。

- **Draft TTL**（`--standalone-remote-draft-ttl-s`，默认 60s，`<=0` 关闭）：
  Target abort / 崩溃且没发 FINISH 时，Draft 会丢掉空闲 RPC 的 KV。
  当前 RPC 里的 rid 不会被清掉。
- **连续超时熔断**（`--standalone-remote-breaker-failures` 默认 3，
  `--standalone-remote-breaker-cooldown` 默认 32）：Target 连续 recv 超时后
  跳过 STEP RPC，decode 走 1-token AR；冷却结束后试一次 STEP。
  PREFILL 在 OPEN 时仍会发出（探活）。Draft **REJECT**（忙）不算超时。
  `/flush_cache` 会 reset 熔断器。
- **Draft 忙 REJECT**：running batch（或其中的 HTTP 请求数）超过
  `--standalone-remote-max-batch-size` 时，Draft 立刻回 REJECT、不跑 GPU。
  有任意 HTTP 存活就 REJECT 会把 warmup 期间的每一拍 RPC 都打掉，所以不用那种启发式。
  Target 把非 OK 回复当成空窗，该步退回 AR。

## 约束

- ``topk=1``（默认）是链：Draft AR 窗口，Target EAGLE 核验。
- ``topk>1`` 是 Draft 上的 STANDALONE 树（`select_top_k_tokens` /
  `organize_draft_results`），以 `parent_list` / `top_scores_index` 发出。
  树 drafter 初始化失败会直接中止 Draft 启动（不会回退到 AR / vine）。
  不使用 SPECTRE 那种把链在 top-k 槽上重复的假灌木。
  示例：`--speculative-eagle-topk 2 --speculative-num-draft-tokens 8`。
  每次 STEP 先把新接受的 committed token 链 decode 进线性 KV
  （Target 一次接受超过 1 个 token 时走 ``append_n``），再从该前缀展开树。
  整段 re-prefill 只用于序列中段分叉，不用于更长的已接受后缀。
- overlap scheduler 与 mixed chunked prefill 关闭。
- Target verify 复用 EAGLE 树核验：structured output 走 `generate_token_bitmask`；
  `return_logprob` 只写接受路径（含 bonus）。Hybrid Mamba/GDN/Lightning 走与 EAGLE 相同的
  MTP scatter（`mamba_track_interval >= speculative_num_draft_tokens`，建议 extra_buffer）。
  Draft 在 PREFILL 用同一份 json/regex/ebnf schema 编译 grammar，每窗生成前按 committed
  前缀 replay；编译失败则 Draft 不约束，Target mask 仍保证正确性。
- Draft GPU 按 RPC 融合，类似同进程 STANDALONE 的 `draft(batch)`：align 仍按 rid，
  然后链窗口 / 树 ingest / 树 expand 对本 RPC 里所有活 rid 各跑一次。
  `_sr_isolate_need` 只 pause **不在** 本 RPC 里的请求（调度器残留），不 pause 同 RPC 的兄弟 rid。
  Draft 上的 HTTP generate 在 RPC 拍内同样被 pause，拍后再 resume；
  HTTP 与 RPC 不会进入同一个 `ScheduleBatch`。
- `page_size>1` 且 `topk>1` 不支持（树 KV 只支持 `page_size=1`）。
- CUDA graph 默认开启：Target 捕获 `TARGET_VERIFY`（`ntpb = speculative_num_draft_tokens`）
  以及 SPECTRE 同款的 `ntpb=1` DECODE graph；`can_run` / replay 按 forward mode 分流。
  Draft 链（`topk=1`）走普通 DECODE graph；Draft 树复用 v1 `EAGLEDraftCudaGraphRunner`
  （不捕获 draft-extend graph，也不走 spec v2 / plan stream）。
  树仍要求 `page_size=1`。Target 1-token AR fallback **强制 eager**，不 replay verify graph。
- 视觉仅 Qwen3-VL：`Qwen3VLForConditionalGeneration` /
  `Qwen3VLMoeForConditionalGeneration`。
- Draft **禁止** `--skip-tokenizer-init`（3D M-RoPE）。
- Draft `--context-length` 必须 ≥ Target **已 pad** 的最大 prompt。
- Draft 复用 Target 已经 pad 好的 `input_ids` 和 `pad_value`，自己不调用
  `pad_input_ids`。不一致则该请求跳过投机。
- Re-prefill 会重算 M-RoPE，但保留 `precomputed_embeddings`（不再跑一遍 ViT）。
  Target 不发送 `mm_input_embeds`。
- 纯文本请求不带 mm frame。
- Draft TP>1：rank 0 收 RPC，再 `broadcast_pyobj` 给其它 rank。

## 协议身份（过期回复）

超时是正常路径（Target 退回 1-token AR）。上一拍的回复仍可能落到 socket 上。
只有下列字段 **全部** 匹配才接受回复：`session_id`、`rpc_seq`、rid 在 pending 集合里、
`step_id`、`base_committed_len`。Recv 循环直到匹配或超时；每次 send 前会 drain socket。

`base_committed_len` 是必需的：AR fallback 后 Target 前缀会长 1 个 token，Draft 必须整段对齐。
若只检查 `step_id`，会发出错位草稿，表现为莫名其妙的低 accept rate。

指标：`sglang:sr_stale_replies_dropped_total{reason=session|rpc_seq|step|base_len}`。
非零表示发生过超时或乱序。

## 正确性检查

`temperature=0` 时，输出必须与 **仅 Target 的 greedy** 一致，而不是 Draft。
accept rate 低可以接受。缺 token、重复 token、过早 EOS 才是 bug。

## 排查

| 现象 | 可能原因 |
|---|---|
| accept rate 接近 0，无报错 | `padded_input_ids` / 视觉几何不一致；或过期回复错位（看上面的 counter） |
| Draft 卡住 | Target 没发 FINISH；abort 掉 Target 上的请求。超过 `--standalone-remote-draft-ttl-s` 后 Draft 会自己丢掉空闲 KV |
| Draft 上 `q_len` flashinfer 报错 | `prepare_for_decode` 没有收成 last token（remote draft 路径） |
| `/flush_cache` 之后 VL 崩 `128 vs 64` embeddings | Draft wipe 必须在 **没有** HTTP 请求存活时 reset radix / waiting_queue / last_batch / VLM embedding cache（HTTP flush 到不了 Draft） |
| 并发 VL `gather kernel index out of bounds` | Draft `_sr_run_until_ready` 混进了调度器残留请求；isolate 必须 pause **本 RPC 之外** 的 rid |
| 第一窗之后 `accept len` 卡在 1.00 | Draft `max_new_tokens` 撞上 FINISH_LENGTH（约 9）；STEP 必须把预算抬上去 |
| 5 个 draft token 时 `accept rate` 卡在 0.20 | STEP 跳过了 align，或 rollback 裁掉后没有套上 Target committed ids |
| `topk>1` 的 accept rate 不比链好 | Draft 树展开失败（看 `[SR] tree expand failed`）；Target 必须保留 `parent_list` |
| `topk>1` 时 `accept len` 卡在约 2，rate 约 2/num-draft-tokens | 树 ingest 跳过了链上的 `prepare_for_decode`（committed token 没进 Draft KV）；或 `_draft_forward` 冻住了 Qwen3-VL 的 `mrope_positions` |
| Draft 每个 STEP 都是 Prefill 且 `#cached-token: 0` | Align 走了 reprefill 而不是 `append_n`；Target 接受 >1 个 token 时只应 ingest 新尾巴 |
| M-RoPE / VL 质量崩掉 | Draft 用了 `--skip-tokenizer-init` |
| IPC bind 失败 | 被杀掉的进程留下了 `/tmp/sr_*` socket |
