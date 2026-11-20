# 运行

## 事先

- 运行号：2026-11-20-a
- 要回答的问题：分段配对成为账本开启时的在线 W 之后，RSTeller 并发 1 和并发 4 是否出现 `d_wpair` 和 `superset`，服务宽度有没有换，吞吐和接受率是多少。
- 对照的上一次保留运行：2026-11-19-d
- 相对那次运行改了什么：只换 Draft 与 Target 的 `clear_tree_remote`，在线 W 改为分段配对。负载和服务开关不变。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4、捕图 batch 1 2 4。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。先 Draft 后 Target，两轮之间不重启。客户端在 NPU 容器里打 Target，计时用 `eval_spectre_rsteller_serving.py`，每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 和 `raw/02/` 放该次日志和客户端输出。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 同一对进程，两轮之间没有重启。后一轮继承前面已经写过的账本。Draft 与 NPU 上的 `clear_tree_remote` 逐文件 SHA256 一致，共 29 个文件，含 `clear_tree_wpair.py`。数据 json 与 100 张图的 hash 和 2026-11-19-d 的 `raw/sync/data-sha256.txt` 一致。NPU 上 json 和 100 张图在容器 `/tmp/clear-shared-20261120a/data`。
  - 预检时 Phy-ID 2 和 3 没有进程。NPU 序号 0 上原有的 vLLM，以及 Phy-ID 4 到 7 上的 vLLM worker，都没有动。两端 `ssh -N` 的 PID 和启动时间与上一场记录相同。
  - Draft `Capturing: bs=` 共 38 行，bs 为 1、2、4。没有出现的形状只有 batch 4 的 65 宽。Target 共 21 行，bs 为 1、2、4，各 7 行。
  - `01` RSTeller 并发 1：100/100，输出 token 46373，正式 wall 867.057 s，吞吐 53.483 tok/s。接受率 34534/95936 = 0.360。fallback 0。Draft rank 0 已刷窗口 346 个（345 个 `full`，1 个 `idle`，末窗是 `full`）：`opt_bonus_forwards` 11041，`opt_bonus_requests` 11041，`opt_bonus_graph` 11041，`opt_bonus_eager` 0。末窗不是 `idle`，这轮窗口之后的尾数记为 missing。Target 两个 TP 都有 `Cache flushed successfully`，HTTP 200，客户端没有打印 flush 失败。
  - `02` RSTeller 并发 4：100/100，输出 token 45692，正式 wall 321.306 s，吞吐 142.207 tok/s，客户端统计并发 3.921。接受率 31774/112128 = 0.283。fallback 0。已刷窗口 149 个（147 个 `full`，2 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 4758，`opt_bonus_requests` 15072，`opt_bonus_graph` 4758，eager 0。`requests` 大于 `forwards`。flush 同样成功。
  - 对照 2026-11-19-d：并发 1 是 53.778 tok/s、接受率 0.356；并发 4 是 141.650 tok/s、接受率 0.254。这场并发 1 的吞吐低约 0.3 tok/s，并发 4 高约 0.6 tok/s，并发 4 的接受率更高。
  - Draft 日志有 41 条 `d_wpair`，没有 `d_replay`。`01` 里 18 条停在 w=8，然后 1 条 `up` 到 16，之后 5 条停在 w=16。`02` 里 1 条 `up` 从 4 到 8，其余是停留。日志行里没有 `superset` 和 `probe_w` 这两个字。`d_round` 的 `w` 能对上这两种轮：b=1 开头是 4，接着是 8，切宽前已经出现 16，切到 16 之后出现 32；b≥2 开头是 2，接着是 4。
- 总结：两轮正式请求都在同一对进程上跑完。分段配对会写出 `d_wpair`，并且这两轮都发生了一次加宽。并发 1 的吞吐和接受率与上一场接近，并发 4 的吞吐接近、接受率更高。批量 bonus 已刷窗口的 eager 都是 0。`01` 的 bonus 尾数没有刷出。Target 日志在 `raw/01/target-server.log`，覆盖两轮；Draft 日志同样在 `raw/01/draft-server.log`。清理后本场 Draft 与 Target 的 `launch_server` 都不在，30001 和 30000 已释放，GPU 0 没有计算进程，Phy-ID 2 和 3 没有进程。两端 `ssh -N` 的 PID 和启动时间未变。
