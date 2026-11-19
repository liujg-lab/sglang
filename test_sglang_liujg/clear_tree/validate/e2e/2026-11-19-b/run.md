# 运行

## 事先

- 运行号：2026-11-19-b
- 要回答的问题：账本打开时，RSTeller 和 WaterCaption 在并发 1 与并发 4 上的吞吐、接受率、在线 W、在线 K 和层耗时拟合如何。
- 对照的上一次保留运行：无。这场只测开账本，不另跑关账本。
- 相对那次运行改了什么：服务参数与 2026-11-19-a 相同，负载换成图文。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4、捕图 batch 1 2 4。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。先 Draft 后 Target，四轮之间不重启。客户端在 NPU 上打 Target，每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 到 `raw/04/` 放该次日志和客户端输出。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 03 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 1，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 04 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 同一对进程，四轮之间没有重启。后一轮继承前面已经写过的账本。Draft 与 NPU 上的 `clear_tree_remote` 逐文件 SHA256 一致，共 27 个文件。数据文件 hash 在 `raw/sync/data-sha256.txt`。
  - `01` RSTeller 并发 1：100/100，输出 token 46374，正式 wall 871.470 s，吞吐 53.214 tok/s。接受率 34465/96528 = 0.357。fallback 0。Draft rank 0 已刷窗口 381 个（380 个 `full`，末窗 `idle`）：`opt_bonus_forwards` 12164，`opt_bonus_requests` 12164，`opt_bonus_graph` 12164，`opt_bonus_eager` 0。Target 两个 TP 都有 `Cache flushed successfully`，客户端没有打印 flush 失败。
  - `02` RSTeller 并发 4：100/100，输出 token 45696，正式 wall 330.015 s，吞吐 138.466 tok/s，客户端统计并发 3.928。接受率 30619/121064 = 0.253。fallback 0。已刷窗口 125 个，末窗 `idle`：`opt_bonus_forwards` 3995，`opt_bonus_requests` 15143，`opt_bonus_graph` 3995，eager 0。`requests` 大于 `forwards`。flush 同样成功。
  - `03` WaterCaption 并发 1：100/100，输出 token 29608，正式 wall 669.582 s，吞吐 44.219 tok/s。接受率 22451/58384 = 0.385。fallback 0。已刷窗口 230 个，末窗 `idle`：`opt_bonus_forwards` 7356，`opt_bonus_requests` 7356，`opt_bonus_graph` 7356，eager 0。flush 成功。
  - `04` WaterCaption 并发 4：100/100，输出 token 30329，正式 wall 336.289 s，吞吐 90.187 tok/s，客户端统计并发 3.952。接受率 19793/84624 = 0.234。fallback 0。已刷窗口 87 个，末窗 `idle`：`opt_bonus_forwards` 2701，`opt_bonus_requests` 10242，`opt_bonus_graph` 2701，eager 0。flush 成功。
  - 在线 K 离开了平表。`01` 共 12164 轮：平表 W=8、K=16 有 6011 轮，W=8、K=32 有 4535 轮。`02` 最多的是 b=4、W=4、K=8 的 3156 轮，离开平表 531 轮。`03` 的 7356 轮没有一轮停在平表 K=16：W=8、K=8 有 3788 轮，W=8、K=32 有 3142 轮。`04` 里 b=4、W=4、K=4 有 1196 轮，多于平表 K=8 的 1081 轮。四轮 `verify_residual` 全部是 null；Target 是 NPU，影子回归不选 K。
  - 在线 W 没有切服务宽度。四轮 `d_replay` 的 `applied` 都是 0。`would_w` 在 b=1 保持 8，在 2–4 档保持 4。探索轮分别是 595、179、360、122。
  - 层拟合用墙钟。`01` 主组 W=8、graph、图档 1：33523 条，残差中位数 -0.029，绝对残差中位数 0.075。`02` 主组 W=4、graph、图档 4：6525 条，中位数 -0.006，绝对中位数 0.063。`03` 主组 W=8、图档 1：21562 条，中位数 -0.040，绝对中位数 0.086。`04` 主组 W=4、图档 4：3548 条，中位数 -0.014，绝对中位数 0.046；同宽图档 2 只有 379 条，中位数 0.181。
  - 捕图在共同启动里完成。Draft `Capturing: bs=` 共 38 行，bs 为 1、2、4。Target 共 21 行，bs 为 1、2、4，各 7 行。
- 总结：四轮正式请求都在同一对进程上跑完。在线 K 会离开平表，WaterCaption 并发 1 已经不再使用平表 K=16，并发 4 上 K=4 多于平表 K=8。在线 W 只探索，没有切宽。主要图档的层耗时中位偏差大约 0% 到 4%，绝对残差中位数大约 5% 到 9%。小图档样本少时偏差到 18%。清理后本场 Draft 与 Target 进程都不在，30001 和 30000 已释放，GPU 0 没有计算进程，本场用的 NPU 1 没有进程。进程表 NPU 2 和 3 上出现了另一项 vLLM，不是本场进程，没有动它。NPU 0 上原有的 vLLM 也没有动。两端 `ssh -N` 的 PID 和启动时间未变。
