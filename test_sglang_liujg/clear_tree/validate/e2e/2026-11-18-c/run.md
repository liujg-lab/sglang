# 运行

## 事先

- 运行号：2026-11-18-c
- 要回答的问题：当前 ClearTree bonus 专用图在 NPU Target 与 CUDA Draft 上，捕图 batch 1 2 3 4、bonus 开启时，并发 1 和并发 4 能否完成请求并实际回放专用图。
- 对照的上一次保留运行：2026-11-18-b（该场没有正式请求，只作编排参考）
- 相对那次运行改了什么：省略 `--clear-tree-widths` 和 `--clear-tree-ks`，使用包内默认候选 2 4 8 16 32 64。其余负载与 2026-11-18-b 相同：100 条、最多 1024 token、温度 0、request-rate inf、ledger 关、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4、捕图 batch 1 2 3 4。每轮冷启动，先 Draft 后 Target。客户端在 NPU 上打 Target，内部预热一条并 flush，不计吞吐。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 放该次日志和客户端输出。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 3 4 | 1 | 正式 |
| 02 | 100 条，最多 1024 token，温度 0 | 并发 4，bonus 开，捕图 1 2 3 4 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - `01`：100/100 成功，输出 token 66187，正式 wall 1134.918 s，吞吐 58.319 tok/s。接受率 50430/127320 = 0.396。fallback 0。Draft rank 0 已刷窗口 506 个：`opt_bonus_forwards` 16149，`opt_bonus_requests` 16149，`opt_bonus_graph` 16149，`opt_bonus_eager` 0，`opt_bonus_unknown` 0，`opt_bonus_failed` 0。Draft 捕获 51 张图，batch 1/2/3/4 都有；batch 4 的 ntpb 65 没有捕获。Target 捕获 124 张图，batch 1/2/3/4 都有，含 ntpb 65。预热后 flush 返回 200，日志有 `Cache flushed successfully`。
  - `02`：100/100 成功，输出 token 66299，正式 wall 476.950 s，吞吐 139.006 tok/s，客户端统计并发 3.925。接受率 47721/149400 = 0.319。fallback 0。Draft rank 0 已刷窗口 159 个：`opt_bonus_forwards` 5041，`opt_bonus_requests` 18813，`opt_bonus_graph` 5041，`opt_bonus_eager` 0，`opt_bonus_unknown` 0，`opt_bonus_failed` 0。`requests` 大于 `forwards`。捕图与 `01` 相同：Draft 51 张、缺 batch 4 的 ntpb 65；Target 124 张，batch 1/2/3/4 都有。flush 同样成功。
- 总结：两轮正式请求都跑完。已刷窗口里的 bonus 前向全部记在 `opt_bonus_graph`，eager、unknown、failed 都是 0。并发 4 出现了多请求 bonus 组。计数含服务预热和客户端丢掉的那一条预热，最后一条已刷窗口的原因是 idle。捕图完成不表示每个 batch 在正式请求里都回放过。没有 bonus 关闭对照。
