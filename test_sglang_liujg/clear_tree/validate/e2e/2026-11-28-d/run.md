# ClearTree 复用拟合：双模式、双数据集端到端验证

## 事先

- 运行号 `2026-11-28-d`，提交 `1a15ad1ca5331e15712fca337bf5fb7232ba3922`。服务配置沿用 [2026-11-28-c](../2026-11-28-c/run.md)，矩阵参考 [2026-11-27-b](../2026-11-27-b/run.md)。两种验证方式分块，每格一轮，不是 ABAB 或同条件重复。
- Draft：RTX 3090 卡 0，Qwen3-VL-2B W8A8 INT8、TP1、page 1、`triton`/`triton_attn`、静态内存 0.6、容量 64。Target：910c 卡 2/3，Qwen3-VL-32B、TP2、page 128、`ascend`/`ascend_attn`、静态内存 0.8、容量 65。账本、bonus、KV reuse 开，初始 W=8、深度/steps=8、ClearTree τ=0.5，默认 W/K 候选，最大请求数 4，捕图 batch 1/2/4。仅 Draft 开 lookahead，接受比半衰期 1000。RPD τ=0.4 只给 Target。
- 复用 Draft 现行两份拟合，与上场封存哈希核对；不移动、不覆盖，不做自动或手动预拟合。备份远端包并窄同步源码。每块先启动 Draft，再启动 Target；双端 HTTP 就绪、图形状和已有拟合覆盖符合预期、Target 双 rank 同一会话 `ready/ready` 后放行。Greedy 完成并清理后只重启一次切到 RPD。逐块复核两份拟合哈希和无新写盘。逐轮统计正式样本的在线层尖峰限幅，不主动注入尖峰或加跑。
- RSTeller 沿用上场同序 100 条，WaterCaption 沿用 `2026-11-27-b` 同序 100 条 AR 清单。温度 0、生成预算 1024、请求速率 `inf`、超时 600 s、`--skip-metrics`。Target 容器内每轮预热首条一次并 flush，二者不计正式 wall。同模式四轮共用双端进程；双端日志按块留存，客户产物及双端本地日志边界按轮留存。

| 轮次 | 负载 | 验证方式 | 并发 | 第几次 | 阶段 |
|---|---|---|---:|---:|---|
| 01 | RSTeller | Greedy | 1 | 1 | 正式 |
| 02 | RSTeller | Greedy | 4 | 1 | 正式 |
| 03 | WaterCaption | Greedy | 1 | 1 | 正式 |
| 04 | WaterCaption | Greedy | 4 | 1 | 正式 |
| 05 | RSTeller | RPD τ=0.4 | 1 | 1 | 正式 |
| 06 | RSTeller | RPD τ=0.4 | 4 | 1 | 正式 |
| 07 | WaterCaption | RPD τ=0.4 | 1 | 1 | 正式 |
| 08 | WaterCaption | RPD τ=0.4 | 4 | 1 | 正式 |

## 结果

**通过。** 八轮共 800/800 请求成功，800 个响应 ID 唯一，无请求 fallback。每轮都在正式计时前得到 Target 双 rank 的 flush 确认和 HTTP 200。两套数据及图片哈希与指定参考场一致。

| 轮次 | 负载 | 验证 | 并发 | 成功 | 输出 token | 正式 wall (s) | 吞吐 (tok/s) | 接受 token / Draft token | 在线限幅样本 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | RSTeller | Greedy | 1 | 100/100 | 45,311 | 767.046 | 59.072 | 34,778 / 85,528 = 40.66% | 436 |
| 02 | RSTeller | Greedy | 4 | 100/100 | 46,681 | 291.377 | 160.208 | 33,843 / 103,592 = 32.67% | 154 |
| 03 | WaterCaption | Greedy | 1 | 100/100 | 29,725 | 584.334 | 50.870 | 23,378 / 51,896 = 45.05% | 275 |
| 04 | WaterCaption | Greedy | 4 | 100/100 | 30,139 | 269.703 | 111.749 | 22,135 / 65,152 = 33.97% | 112 |
| 05 | RSTeller | RPD τ=0.4 | 1 | 100/100 | 43,893 | 688.838 | 63.720 | 34,576 / 75,600 = 45.74% | 275 |
| 06 | RSTeller | RPD τ=0.4 | 4 | 100/100 | 44,735 | 263.440 | 169.811 | 33,499 / 90,872 = 36.86% | 168 |
| 07 | WaterCaption | RPD τ=0.4 | 1 | 100/100 | 28,973 | 531.624 | 54.499 | 23,364 / 46,048 = 50.74% | 200 |
| 08 | WaterCaption | RPD τ=0.4 | 4 | 100/100 | 28,934 | 244.895 | 118.149 | 21,991 / 56,680 = 38.80% | 100 |

- **复用拟合：** 两块均加载原 schema 5 文件（指纹 `babb423d`，90 个 H 格、18 个层格），Draft 各记录一次 `cap=64 cells=4` 和安装完成，Target TP0/TP1 同会话 `ready/ready`、`run=null`。没有预拟合启动或写盘，两份现行拟合的 SHA-256 在预检、同步后、每块启动前、放行时及结束后均与上场封存值一致。证据见 [Greedy](raw/greedy/fit-gate.txt)、[RPD](raw/rpd/fit-gate.txt) 和 [在线验收](checks.json)。
- **1.5× 在线限幅：** Draft `d_round` 覆盖全部 800 个正式响应 ID；按 `(session_id, rpc_seq)` 关联 `d_layer` 后，124,199 个可比较拟合层样本中有 1,720 个记录 `fit=true, clipped=true`（Greedy 977、RPD 743）。每个都满足原始耗时大于 `1.5 × pred_ms`，没有超阈值但未记录限幅的可比较样本。第 1 层未参与拟合，正式轮未观测到无效 hold。
- **图与 RPD：** 双模式各捕获 Draft 38 / Target 21 个形状，覆盖 batch 1/2/4；Draft `(4,65)` 超容量，未捕获。Target 的正式段用本机 flush 边界界定，所记录 42,559 条 `t_verify` 均走 graph；RPD 路径为 `npu_sr_host_v1`，四轮均有关联的有效晚到掩码。Draft rank 0 已刷窗口中 bonus forwards/requests/graph/eager 为 Greedy `22905/38395/22905/0`、RPD `20254/33742/20254/0`。这些是整块已刷窗口增量（含预热），不是逐轮命中率；Greedy 末窗 `full`，尾数未确认刷出，RPD 末窗 `idle`。
- **清理与重试：** Greedy 完成后只重启一次切到 RPD。两块结束时本场双端 session 无存活成员，HTTP 端口释放，选定设备空闲，既有隧道身份不变。首次同步命令因预检证据写入路径问题在触碰远端包前停止；修正后完成窄同步，误写的旧场基线已恢复并通过原 `SHA256SUMS` 校验。没有正式轮重试。详见 [修正记录](raw/sync/execution-script-hash-amendments.txt)。
- **未验证：** 没有重新预拟合，未覆盖预拟合提交时的在线样本 replay。每格仅一次且无关闭限幅对照，不能由本场吞吐推断限幅带来的性能收益；未验证质量等价、BLEU/ROUGE 或重复性。

宿主机依序执行 `python3 -B test_sglang_liujg/clear_tree/validate/e2e/2026-11-28-d/tools/execute.py {setup,sync,greedy,rpd}`；双端启动参数见 [`server.sh`](tools/server.sh)，实机客户端参数见 [`run-client.sh`](tools/run-client.sh)。离线检查与封存命令：

```bash
run=test_sglang_liujg/clear_tree/validate/e2e/2026-11-28-d
python3 -B "$run/tools/review.py"
python3 -B "$run/tools/accept.py"
sha256sum --check --quiet "$run/raw/sync/execution-script-hashes-final.txt"
python3 -B "$run/tools/seal.py"
(cd "$run/raw" && sha256sum --check --quiet SHA256SUMS)
```

离线验收 8/8 通过；共享日志验收、最终脚本哈希及 [134 个原始产物](raw/SHA256SUMS) 校验均通过。详见 [汇总](summary.json) 和 [门禁](checks.json)。
