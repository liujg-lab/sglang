# ClearTree 层拟合权重封顶：双模式端到端验证

## 事先

- 运行号 `2026-11-28-c`，提交 `e051b5ddac842540b32fdd196454fd256f21b4a9`。沿用 [2026-11-28-b](../2026-11-28-b/run.md) 的运行条件，以两个验证模式分块、每格一轮，不是 ABAB 或同条件重复。参考场快照只用于检查覆盖入口，不装入服务。宿主机编排，依赖 SGLang 的拟合和验收检查均在 Draft 容器，正式客户端在 Target 容器。
- Draft：RTX 3090 卡 0，Qwen3-VL-2B W8A8 INT8、TP1、page 1、`triton`/`triton_attn`、静态内存 0.6、容量 64。Target：910c 卡 2/3，Qwen3-VL-32B、TP2、page 128、`ascend`/`ascend_attn`、静态内存 0.8、容量 65。账本、bonus、KV reuse 开，初始 W=8、深度/steps=8、ClearTree τ=0.5，W/K 候选 2/4/8/16/32/64，最大请求数 4，捕图 batch 1/2/4。只在 Draft 开 lookahead，接受比半衰期 1000；只在 Target 设置验证方式及 RPD τ。
- 备份远端包、双端旧拟合移入 `raw/sync/` 并校验哈希，单独同步 ClearTree 包。两块都用新 schema 5 预拟合；先启动 Draft 再启动 Target，每块须确认写盘、覆盖、双 rank 同代 `ready/prefit`，并与快照核对默认权重封顶 `cap=64` 的实际格数，方可发流量。同模式复用服务；Greedy 清理后只重启一次切到 RPD，核验 Greedy 新拟合哈希不变。不做封顶开关 A/B、质量等价或额外轮次。
- RSTeller 使用参考场同序 100 条，温度 0、生成预算 1024、请求速率 `inf`、单请求超时 600 s、`--skip-metrics`。每轮仅预热首条一次并 flush，预热与 flush 不计正式 wall。按轮保存客户端产物、按模式保存双端服务日志。

| 轮次 | 负载 | 验证方式 | 并发 | 第几次 | 阶段 |
|---|---|---|---:|---:|---|
| 01 | RSTeller | Greedy | 1 | 1 | 正式 |
| 02 | RSTeller | Greedy | 4 | 1 | 正式 |
| 03 | RSTeller | RPD τ=0.4 | 1 | 1 | 正式 |
| 04 | RSTeller | RPD τ=0.4 | 4 | 1 | 正式 |

## 结果

**通过。** `e051b5dda` 完成两次独立预拟合和四轮正式请求。三方源码清单与 RSTeller 数据身份一致，四轮均为参考场同序 100 条，400 个响应 ID 均唯一。每轮无 fallback，Target TP0/TP1 各确认一次 flush，HTTP 均为 200。

| 轮次 | 模式 | 并发 | 成功 | 输出 token | 正式 wall (s) | 吞吐 (tok/s) | 接受 token / Draft token |
|---|---|---:|---:|---:|---:|---:|---:|
| 01 | Greedy | 1 | 100/100 | 45,337 | 766.858 | 59.120 | 34,801 / 85,576 = 40.67% |
| 02 | Greedy | 4 | 100/100 | 45,860 | 287.606 | 159.454 | 33,043 / 103,416 = 31.95% |
| 03 | RPD τ=0.4 | 1 | 100/100 | 44,085 | 691.543 | 63.749 | 34,708 / 76,216 = 45.54% |
| 04 | RPD τ=0.4 | 4 | 100/100 | 44,424 | 257.593 | 172.458 | 33,322 / 89,856 = 37.08% |

- **预拟合和权重封顶：** 两个模式均由缺失拟合开始，Draft 各写一次；Target 双 rank 均报告同代 `ready/prefit`，预拟合用时 Greedy 175.368 s / RPD 166.444 s。两份均为 schema 5、指纹 `babb423d`，覆盖 90 个 H 格（至少 4 样本）和 18 个层格（去尖峰后最少 56 / 54 样本）。两块 Draft 均记录 `cap=64 cells=4`，与快照中带斜率先验且 `xtx[0][0] > 64` 的 4 格一致：`(W,g)=(2,1),(4,2),(4,4),(8,1)`，均是 `graph` 路径。见 [Greedy](raw/greedy/fit-cap.json) 和 [RPD](raw/rpd/fit-cap.json) 的封顶证据。RPD 预拟合及请求结束后，Greedy 新拟合的哈希均未变化。
- **图与账本：** 两块均捕获 Draft 38 个、Target 21 个形状，覆盖 batch 1/2/4；Draft 的 `(4,65)` 超容量，不要求捕获。四轮未见无效 hold，layer 1 未入拟合，RPD 两轮均看到有效接受掩码。Draft rank 0 已刷窗口的 `opt_bonus_forwards/requests/graph/eager` 为 Greedy `15913/27336/15913/0`，RPD `14159/24254/14159/0`，两块末窗均为 `idle`。这些计数包括预拟合等流量，不等于逐轮命中率。
- **清理：** 每个模式结束后，本场双端 session 无存活成员、HTTP 端口已释放、所选设备空闲，既有 SSH 隧道身份未变。旧 Draft 两份拟合和远端包备份位于 `raw/sync/`；Target 未发现旧拟合。两份新拟合留在 Draft，并各留一份快照。没有重试。
- **未验证：** 每格仅一轮，未做 `cap=0` 对照、质量等价或可重复性验证。吞吐不能证明在线适应变快。

宿主机以本场环境依次执行 `python3 -B test_sglang_liujg/clear_tree/validate/e2e/2026-11-28-c/tools/execute.py {setup,sync,greedy,rpd}`；实际启动配置在 `tools/server.sh`，正式客户端在 NPU 容器用 `tools/run-client.sh` 执行。离线验收和归档命令：

```bash
run=test_sglang_liujg/clear_tree/validate/e2e/2026-11-28-c
docker exec -e PYTHONPATH=python -w /home/liujg/workspace/sglang liujg_sglang python3 -B "$run/tools/review.py"
python3 -B "$run/tools/seal.py"
(cd "$run/raw" && sha256sum --check --quiet SHA256SUMS)
sha256sum --check --quiet "$run/raw/sync/execution-script-hashes.txt"
```

离线 `review.py`、两模式的 `accept_logs.py` 和归档校验均返回 0；[原始产物清单](raw/SHA256SUMS) 共 97 项。详见 [验收](checks.json) 与 [汇总](summary.json)。
