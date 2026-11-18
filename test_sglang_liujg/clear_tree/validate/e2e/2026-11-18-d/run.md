# 运行

## 事先

- 运行号：2026-11-18-d
- 要回答的问题：层前向与 bonus 前向合并为无后缀 `r{G}_{bs}` 之后，捕图 batch 1 2 4、账本关闭、bonus 开启时，并发 1 和并发 4 能否完成请求，并且 bonus 回放命中共用图而不是 eager。
- 对照的上一次保留运行：2026-11-18-c（该场捕图含 batch 3，且每轮冷启动）
- 相对那次运行改了什么：捕图改为 `--cuda-graph-bs 1 2 4`；Draft 与 Target 只启动一次，两轮正式请求之间不重启。其余负载与 2026-11-18-c 相同：100 条、最多 1024 token、温度 0、request-rate inf、ledger 关、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4。省略 `--clear-tree-widths` 和 `--clear-tree-ks`，使用包内默认候选 2 4 8 16 32 64。先 Draft 后 Target。客户端在 NPU 上打 Target，内部预热一条并 flush，不计吞吐。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 放该次日志和客户端输出。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本关 | 1 | 正式 |
| 02 | 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本关 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - `01`：100/100 成功，输出 token 66187，正式 wall 1129.573 s，吞吐 58.595 tok/s。接受率 50430/127320 = 0.396。fallback 0。Draft rank 0 已刷窗口 506 个，末窗原因 `idle`：`opt_bonus_forwards` 16149，`opt_bonus_requests` 16149，`opt_bonus_graph` 16149，`opt_bonus_eager` 0，`opt_bonus_unknown` 0，`opt_bonus_failed` 0。预热后 flush 在 14:15:27 UTC 返回 200，日志有 `Cache flushed successfully`。
  - `02`：100/100 成功，输出 token 66507，正式 wall 474.514 s，吞吐 140.158 tok/s，客户端统计并发 3.932。接受率 48032/148832 = 0.323。fallback 0。Draft rank 0 已刷窗口 155 个，末窗 15:18:09 UTC 原因 `idle`：`opt_bonus_forwards` 4932，`opt_bonus_requests` 18665，`opt_bonus_graph` 4932，`opt_bonus_eager` 0，`opt_bonus_unknown` 0，`opt_bonus_failed` 0。`requests` 大于 `forwards`。flush 在 15:08:43 UTC 返回 200。
  - 捕图只发生在这次共同启动里。Draft 捕获 38 张无后缀图，`bs` 为 1、2、4，没有 `bs=3`，也没有 `_s0`。batch 4 的 ntpb 65 没有捕获。Target 的 `Capturing: bs=` 覆盖 1、2、4，完成行是 93 张图；key 里的 `_s256` 一类是 KV 桶，不是 bonus 后缀。日志里没有 `[CLEAR ledger`。
- 总结：两轮正式请求都在同一对 Draft 与 Target 进程上跑完，中间没有重启。已刷窗口里的 bonus 前向全部记在 `opt_bonus_graph`，eager、unknown、failed 都是 0。并发 4 的 requests 大于 forwards，说明多请求组走了共用图。计数含服务预热和客户端丢掉的那一条预热；`01` 收到 14:35:48 UTC 的 idle 窗，`02` 收到自己的 idle 窗。捕图完成不表示每个 batch 在正式请求里都回放过。清理后 Draft 会话和 GPU 0 计算进程已空，30001 已释放；Target 会话只剩僵尸，30000 已释放。清理时 NPU 2 和 3 上已有另一项 vLLM 作业，启动时间是本场结束后，没有动它。两端 `ssh -N` 的 PID 和启动时间未变。
