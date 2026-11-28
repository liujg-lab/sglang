# ClearTree schema 5 双模式端到端验证（重试）

## 事先

- 运行号 `2026-11-28-b`，提交 `87725b0d7baac4fcadc6a1a67f69e052bfeb9795`。两种模式各自预拟合、各跑两轮，不复用旧拟合或统计，不是 ABAB 或同条件重复。164 宿主机负责编排；依赖 SGLang 的覆盖检查和离线验收在 Draft 容器执行。开跑前用已有的归档拟合验证同一个检查入口，仅作环境预检。
- Draft：RTX 3090 物理卡 0，Qwen3-VL-2B W8A8 INT8、TP1、page 1、`triton`/`triton_attn`、静态内存 0.6、容量 64。Target：910c 物理卡 2/3，Qwen3-VL-32B、TP2、page 128、`ascend`/`ascend_attn`、静态内存 0.8、容量 65。双方账本、bonus、KV reuse 开，初始 W=8、深度/steps=8、ClearTree τ=0.5，默认 W/K 候选 2/4/8/16/32/64，最大请求数 4，捕图 batch 1/2/4。只有 Draft 开 lookahead，接受比半衰期 1000。只有 Target 接收验证方式和 RPD τ。
- 空闲预检后，把两端上场拟合移入本场 `raw/sync/` 并校验哈希，核对双端包。每个模式先启动 Draft 再启动 Target，等 schema 5 拟合写盘、覆盖过关且两个 TP 均 `ready/prefit` 才发流量。同一模式复用服务，切换模式重启两端；新 Greedy 拟合保留并前后验哈希。
- RSTeller 用参考场同序 100 条，温度 0、生成预算 1024、请求速率 `inf`、单请求超时 600 s、跳过 BLEU/ROUGE。每轮客户端预热首条一次后 flush，预热和 flush 不计正式 wall。按轮归档客户端和按模式保留服务原始日志，不额外发请求。

| 轮次 | 负载 | 验证方式 | 并发 | 第几次 | 阶段 |
|---|---|---|---:|---:|---|
| 01 | RSTeller | Greedy | 1 | 1 | 正式 |
| 02 | RSTeller | Greedy | 4 | 1 | 正式 |
| 03 | RSTeller | RPD τ=0.4 | 1 | 1 | 正式 |
| 04 | RSTeller | RPD τ=0.4 | 4 | 1 | 正式 |

## 结果

**通过。** `2026-11-28-b` 完成两次独立预拟合和四轮正式流量。源码清单、RSTeller 数据及四轮的 100 条顺序均一致。四轮各成功 100/100，响应 ID 共 400 个且不重复；fallback 均为 0。每轮 Target TP0、TP1 各记录一次 cache flush，HTTP 均为 200。

| 轮次 | 模式 | 并发 | 成功 | 输出 token | 正式 wall (s) | 吞吐 (tok/s) | 接受 token / Draft token |
|---|---|---:|---:|---:|---:|---:|---:|
| 01 | Greedy | 1 | 100/100 | 45,400 | 773.278 | 58.711 | 34,790 / 86,088 = 40.41% |
| 02 | Greedy | 4 | 100/100 | 45,615 | 285.909 | 159.544 | 32,810 / 103,368 = 31.74% |
| 03 | RPD τ=0.4 | 1 | 100/100 | 43,770 | 682.834 | 64.100 | 34,521 / 75,200 = 45.91% |
| 04 | RPD τ=0.4 | 4 | 100/100 | 45,058 | 262.385 | 171.725 | 33,628 / 92,496 = 36.36% |

- **预拟合：** 每块开始时各自的拟合缺失；Draft 各写一次，Target 双 rank 均记录同代 `ready/prefit`。两份 schema 5 的指纹均为 `babb423d`，各覆盖 90 个 H 单元（最少 4 样本）和 18 个 layer 单元（去尖峰后最少 Greedy 58 / RPD 56）。RPD 结束时 Greedy 拟合哈希不变。见 [Greedy](raw/greedy/fit-coverage.json) 和 [RPD](raw/rpd/fit-coverage.json) 覆盖。
- **图与账本：** 两块各捕获 Draft 38 个、Target 21 个形状，均覆盖 batch 1/2/4；Draft 的 `(4,65)` 超容量，不要求捕获。正式轮次无无效 hold，layer 1 未进入拟合，RPD 两轮均观察到有效接受掩码。按 Draft rank 0 已刷窗口累计（包含预拟合等非正式流量），Greedy `full=497`、`idle=2`，`opt_bonus_forwards/requests/graph/eager=15941/27383/15941/0`；RPD `full=440`、`idle=2`，对应 `14123/24462/14123/0`。两块末窗均为 `idle`，不把这些计数当成逐试验的命中率。
- **清理与归档：** 两块均确认 owned session 无存活成员、服务端口释放、所选设备空闲、既有 SSH 隧道身份未变。旧拟合及远端备份在 `raw/sync/`；本场两个新拟合均保留在 Draft 并留有快照。`raw/SHA256SUMS` 的 91 个条目全部校验通过。未运行 BLEU/ROUGE；每格仅一轮，不据此作质量等价或可重复性结论。

实际执行：宿主机从仓库根通过 `python3 -B test_sglang_liujg/clear_tree/validate/e2e/2026-11-28-b/tools/execute.py {setup,sync,greedy,rpd}` 依序编排，拟合检查由该脚本调用 Draft 容器内的 `check-fit`，正式客户端通过 `tools/run-client.sh` 在 NPU 容器执行。离线验收和校验命令：

```bash
docker exec -e PYTHONPATH=python -w /home/liujg/workspace/sglang liujg_sglang \
  python3 -B test_sglang_liujg/clear_tree/validate/e2e/2026-11-28-b/tools/review.py
(cd test_sglang_liujg/clear_tree/validate/e2e/2026-11-28-b/raw && sha256sum --check --quiet SHA256SUMS)
```

`review.py`（两块均调用 `accept_logs.py`）及 checksum 校验退出码均为 0。见 [验收](checks.json)、[汇总](summary.json)、[Greedy 日志摘录](raw/greedy/accept-logs.txt)、[RPD 日志摘录](raw/rpd/accept-logs.txt) 和 [校验清单](raw/SHA256SUMS)。
