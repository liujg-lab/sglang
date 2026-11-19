# 运行

## 事先

- 运行号：2026-11-19-d
- 要回答的问题：Draft 层前向和 bonus 改成无状态 cascade、直接读 `req_to_token` 之后，RSTeller 在并发 1 和并发 4 上的吞吐和接受率相对 2026-11-19-c 如何。
- 对照的上一次保留运行：2026-11-19-c
- 相对那次运行改了什么：只换 Draft 的 cascade。负载和服务开关不变。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4、捕图 batch 1 2 4。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。先 Draft 后 Target，两轮之间不重启。客户端在 NPU 上打 Target，每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 和 `raw/02/` 放该次日志和客户端输出。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 同一对进程，两轮之间没有重启。后一轮继承前面已经写过的账本。Draft 与 NPU 上的 `clear_tree_remote` 逐文件 SHA256 一致，共 28 个文件。数据文件 hash 与 2026-11-19-c 相同，在 `raw/sync/data-sha256.txt`。NPU 宿主机没有这份数据，json 和 100 张图拷到了容器里的 `/tmp/clear-shared-20261119d/data`，内容没有改。
  - NPU 宿主机的 Python 3.9 没有 `aiohttp`，那次客户端在发出请求前退出，记录在 `raw/01/client-host-import-error.out`。正式客户端改在 NPU 容器里跑，没有重跑已经计时的轮次。
  - 预检时 npu-smi 的 NPU 1 没有进程，Target 用逻辑设备 2 和 3。NPU 0 上原有的 vLLM 没有动。
  - Draft `Capturing: bs=` 共 38 行，bs 为 1、2、4。没有出现的形状只有 batch 4 的 65 宽。Target 共 21 行，bs 为 1、2、4，各 7 行，捕获完成时共 93 张带 KV 桶的图。
  - `01` RSTeller 并发 1：100/100，输出 token 46018，正式 wall 855.698 s，吞吐 53.778 tok/s。接受率 34182/95912 = 0.356。fallback 0。Draft rank 0 已刷窗口 378 个（377 个 `full`，末窗 `idle`）：`opt_bonus_forwards` 12089，`opt_bonus_requests` 12089，`opt_bonus_graph` 12089，`opt_bonus_eager` 0。Target 两个 TP 都有 `Cache flushed successfully`，客户端没有打印 flush 失败。
  - `02` RSTeller 并发 4：100/100，输出 token 45519，正式 wall 321.349 s，吞吐 141.650 tok/s，客户端统计并发 3.921。接受率 30553/120384 = 0.254。fallback 0。已刷窗口 125 个，末窗 `idle`：`opt_bonus_forwards` 3974，`opt_bonus_requests` 15054，`opt_bonus_graph` 3974，eager 0。`requests` 大于 `forwards`。flush 同样成功。
  - 对照 2026-11-19-c：并发 1 是 53.923 tok/s、接受率 0.355；并发 4 是 139.959 tok/s、接受率 0.254。这场并发 1 几乎一样，并发 4 的吞吐高约 1.7 tok/s，接受率相同。
  - 在线 K 离开了平表。`01` 共 12089 轮：平表 W=8、K=16 有 5899 轮，W=8、K=32 有 4432 轮。`02` 最多的是 b=4、W=4、K=4 的 1836 轮。两轮 `verify_residual` 全部是 null。
  - 在线 W 没有切服务宽度。两轮 `d_replay` 的 `applied` 都是 0。
- 总结：两轮正式请求都在同一对进程上跑完。无状态 cascade 下，并发 1 的吞吐和接受率与上一场持平，并发 4 的吞吐略高、接受率不变，批量 bonus 全部命中图。在线 K 会离开平表，在线 W 没有切宽。Target 日志在 `raw/01/target-server.log`，覆盖两轮；Draft 日志同样在 `raw/01/draft-server.log`。清理后本场 Draft 与 Target 的 `launch_server` 都不在，30001 和 30000 已释放，GPU 0 没有计算进程，NPU 1 没有进程。NPU 0 上原有的 vLLM 还在。Target 会话里还剩 136 个僵尸，父进程是容器自己的 `sleep infinity`，没有向它发信号。两端 `ssh -N` 的 PID 和启动时间未变。
