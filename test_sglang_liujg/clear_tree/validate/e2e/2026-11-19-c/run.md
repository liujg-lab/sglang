# 运行

## 事先

- 运行号：2026-11-19-c
- 要回答的问题：层回归改成按 (W, path, 图档) 分格、特征为截距加前缀之和之后，RSTeller 在并发 1 和并发 4 上的吞吐、接受率、在线 W、在线 K 和层拟合如何。
- 对照的上一次保留运行：无。这场只测开账本，不另跑关账本，也不跑 WaterCaption。
- 相对那次运行改了什么：服务参数与 2026-11-19-b 相同。负载只留 RSTeller。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4、捕图 batch 1 2 4。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。先 Draft 后 Target，两轮之间不重启。客户端在 NPU 上打 Target，每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 和 `raw/02/` 放该次日志和客户端输出。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 同一对进程，两轮之间没有重启。后一轮继承前面已经写过的账本。Draft 与 NPU 上的 `clear_tree_remote` 逐文件 SHA256 一致，共 27 个文件。数据文件 hash 在 `raw/sync/data-sha256.txt`。
  - 第一次 Target 用 `ASCEND_RT_VISIBLE_DEVICES=2,3` 启动失败。这两枚逻辑设备对应 npu-smi 的 NPU 1，上面已有别人的 `evaluate_selection_opd_dolly.py`。日志是 `raw/target-failed/target-server.log`，报错是内存不平衡，`pre_model_load_memory=44.88` GB。那个进程已退出，别人的进程没有动。正式服务改用空闲的 NPU 2，逻辑设备 4 和 5。
  - `01` RSTeller 并发 1：100/100，输出 token 46206，正式 wall 856.893 s，吞吐 53.923 tok/s。接受率 34283/96496 = 0.355。fallback 0。Draft rank 0 已刷窗口 381 个（380 个 `full`，末窗 `idle`）：`opt_bonus_forwards` 12161，`opt_bonus_requests` 12161，`opt_bonus_graph` 12161，`opt_bonus_eager` 0。Target 两个 TP 都有 `Cache flushed successfully`，客户端没有打印 flush 失败。
  - `02` RSTeller 并发 4：100/100，输出 token 46104，正式 wall 329.411 s，吞吐 139.959 tok/s，客户端统计并发 3.931。接受率 30917/121952 = 0.254。fallback 0。已刷窗口 126 个，末窗 `idle`：`opt_bonus_forwards` 4021，`opt_bonus_requests` 15261，`opt_bonus_graph` 4021，eager 0。`requests` 大于 `forwards`。flush 同样成功。
  - 在线 K 离开了平表。`01` 共 12060 轮：平表 W=8、K=16 有 6180 轮，W=8、K=32 有 4076 轮。`02` 最多的是 b=4、W=4、K=8 的 3006 轮，离开平表 704 轮。两轮 `verify_residual` 全部是 null。
  - 在线 W 没有切服务宽度。两轮 `d_replay` 的 `applied` 都是 0。`would_w` 在 b=1 保持 8，在 b=4 保持 4。探索轮分别是 594 和 173。
  - 层拟合按 (W, path, 图档) 分开，用墙钟。`01` 主组 W=8、graph、图档 1：32872 条，残差中位数 -0.029，绝对残差中位数 0.081。`02` 主组 W=4、graph、图档 4：6627 条，中位数 -0.012，绝对中位数 0.067；同宽图档 2 有 370 条，中位数 0.010；同宽图档 1 有 268 条，中位数 -0.098。
  - 捕图在正式启动里完成。Draft `Capturing: bs=` 共 38 行，bs 为 1、2、4。Target 共 21 行，bs 为 1、2、4，各 7 行。
- 总结：两轮正式请求都在同一对进程上跑完。在线 K 会离开平表，并发 1 上 K=32 是第二多的档。在线 W 只探索，没有切宽。主要图档的层耗时中位偏差大约 1% 到 3%，绝对残差中位数大约 7% 到 8%；并发 4 的小图档 W=4、图档 1 中位偏差大约 10%。逻辑设备 2 和 3 被占用，正式 Target 用的是逻辑设备 4 和 5。清理后本场 Draft 与 Target 都不在，30001 和 30000 已释放，GPU 0 没有计算进程，NPU 2 没有进程。NPU 0 上原有的 vLLM 还在。启动时占着 NPU 1 的两个评测进程结束时已经不在，本场没有向它们发信号。两端 `ssh -N` 的 PID 和启动时间未变。

