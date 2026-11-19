# 运行

## 事先

- 运行号：2026-11-19-a
- 要回答的问题：账本打开后，同一对进程上并发 1 和并发 4 的在线 W、在线 K 是否离开平表，层耗时拟合残差如何，以及吞吐和接受率相对账本关闭的 2026-11-18-d 差多少。
- 对照的上一次保留运行：2026-11-18-d（账本关，捕图 batch 1/2/4，并发 1 与 4 各一轮且不重启）
- 相对那次运行改了什么：Draft 与 Target 的 `SGLANG_CLEAR_LEDGER` 从 0 改为 1。其余与 2026-11-18-d 相同：100 条、最多 1024 token、温度 0、request-rate inf、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4、捕图 batch 1 2 4。省略 `--clear-tree-widths` 和 `--clear-tree-ks`，使用包内默认候选 2 4 8 16 32 64。先 Draft 后 Target，两轮正式请求之间不重启。客户端在 NPU 上打 Target，内部预热一条并 flush，不计吞吐。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 放该次日志和客户端输出。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 02 | 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 同一对进程，中间没有重启。`02` 继承 `01` 已经写过的账本。Draft 与 NPU 上的 `clear_tree_remote` 逐文件 SHA256 一致，共 27 个文件。2026-11-18-d 没有留下包 hash，吞吐差不能单独归到账本。
  - `01`：100/100 成功，输出 token 66869，正式 wall 1135.208 s，吞吐 58.905 tok/s。接受率 51548/123920 = 0.416。fallback 0。对照 d 的并发 1：吞吐 58.595 tok/s，接受率 0.396。Draft rank 0 已刷窗口 492 个，其中 491 个 `full`、末窗 `idle`：`opt_bonus_forwards` 15721，`opt_bonus_requests` 15721，`opt_bonus_graph` 15721，`opt_bonus_eager` 0。flush 后 Target TP0 与 TP1 都有 `Cache flushed successfully`，客户端没有打印 flush 失败。
  - `02`：100/100 成功，输出 token 66200，正式 wall 454.654 s，吞吐 145.605 tok/s，客户端统计并发 3.913。接受率 45647/164888 = 0.277。fallback 0。对照 d 的并发 4：吞吐 140.158 tok/s，接受率 0.323。Draft rank 0 已刷窗口 173 个，其中 172 个 `full`、末窗 `idle`：`opt_bonus_forwards` 5507，`opt_bonus_requests` 20727，`opt_bonus_graph` 5507，`opt_bonus_eager` 0。`requests` 大于 `forwards`。flush 同样在两个 TP 上都有成功日志。
  - 在线 K 离开了平表。`01` 的 `d_round` 共 15721 轮，全部 b=1：K=16 且 W=8 的平表轮 6713，K=32 且 W=8 的 5899，K=8 且 W=8 的 1991。`02` 共 5507 轮，最多的是 b=4、W=4、K=8 的 3682 轮；同宽 K=4 有 552 轮，K=16 有 286 轮。两场 `fallback` 轮都是 0。`verify_residual` 全部是 null，`verify_fit` 没有为真；Target 是 NPU，这条影子回归不更新，选 K 用的是 Draft 的 `T_K`。
  - 在线 W 没有切服务宽度。`d_replay` 的 `applied` 两场都是 0。`01` 的 `would_w` 保持 8；ordinary 14692，explore 770。`02` 在 b 档 2–4 上 `would_w` 保持 4，ordinary 4854，explore 252。没有 `probe_w` 行。`01` 另有约 106 轮是 b=1、W=4，没有进回放。
  - 层拟合用的是 `wall_us`。非 `post_reply` 层都有墙钟。残差是样本写入前的 measured/pred - 1。`01` 主组 W=8、graph、图档 1：49808 条，残差中位数 -0.029，绝对残差中位数 0.076。`02` 主组 W=4、graph、图档 4：9424 条，残差中位数 -0.004，绝对残差中位数 0.060。`02` 里 W=4、图档 1 只有 380 条，残差中位数 -0.222。
  - 捕图在共同启动里完成。Draft 的 `Capturing: bs=` 共 38 行，bs 为 1、2、4。Target 共 21 行，bs 为 1、2、4，每个 bs 都有 ntpb 65、33、17、9、5、3、1。
- 总结：两轮正式请求都在同一对进程上跑完。在线 K 会离开平表，并发 1 上 K=32 接近 K=16 的轮数。在线 W 只做了探索，没有一次切宽。层耗时在主要图档上中位偏差大约 0% 到 3%，绝对残差中位数大约 6% 到 8%；小图档样本少，偏差到 22%。相对账本关闭的 d，并发 1 吞吐几乎一样、接受率略高，并发 4 吞吐更高、接受率更低。这次包和 11 月 18 日那场没有可对的 hash。清理后本场 Draft 与 Target 进程都不在，30001 和 30000 已释放，GPU 0 没有计算进程，NPU 2 和 3 没有进程。NPU 0 上原有的 vLLM 没有动。两端 `ssh -N` 的 PID 和启动时间未变。
