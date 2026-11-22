# 运行

## 事先

- 运行号：2026-11-22-b
- 要回答的问题：移开现有 `clear_tree_fit.json` 后，当前实现是否仍由 Target rank 0 的 FIT 握手驱动扫描，并在必需 H 格和层格各满 4 条后只写一次 schema 2；随后 RSTeller 与 WaterCaption 的并发 1 和并发 4 是否在同一对进程上跑完，正式轮的 `d_round` 是否带上 `hold_ms` 和 `q_ms`。
- 对照的上一次保留运行：2026-11-21-c
- 相对那次运行改了什么：Draft 与 Target 换成当前 `clear_tree_remote`。在线 Q 在写入有效样本时更新该 batch 的斜率和残差中位数，选 K 时不再重新拟合；拟合文件仍是 schema 2，只在预拟合提交时写一次，不含 Q，也不含在线流量。启动前把包内已有拟合文件移到本场 `raw/sync/`，不删除，迫使这次重扫。不设置 `SGLANG_CLEAR_AUTO_PREFIT=0`，不加 `--api-key`。负载和服务开关不变。Target 用物理卡 2 和 3。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4、捕图 batch 1 2 4。容量是 Target 65、Draft 64。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。显式 `--clear-tree-max-depth 8`。先 Draft 后 Target，四轮之间不重启 Target，也不重启 Draft。Draft 打出 `no fit file`、Target 的 `[CLEAR fit_state]` 为 `ready` 且原因是 `prefit`、拟合文件落盘且 `prefit driver finished state=ready` 之前不发 RSTeller。出现 `stashed fit file`、`installed` 旧格子，或扫描没有写盘时不发客户端。客户端在 NPU 容器里打 Target。`01` 和 `02` 用 `eval_spectre_rsteller_serving.py`，`03` 和 `04` 用 `eval_spectre_watercaption_serving.py`。每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。`--img-dir` 用数据集根目录。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 到 `raw/04/` 放该次客户端输出。双端日志是同一对进程的全程，放在 `raw/`。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开，重扫拟合 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 03 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 1，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 04 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 同一对进程，四轮之间没有重启。Draft host pid 2594236，Target host pid 3594680。Target 用物理卡 2 和 3。
  - 启动前包内拟合文件已移到 `raw/sync/clear_tree_fit.before.json`，sha256 是 `12906fe1fbd48fb91496241917ad2feba57311e024822418e8349d3805ce0967`。Draft 先打了一次 `no fit file`。没有 `stashed fit file`，也没有 `installed ... hold cells`。扫描期间打了 117 次 `fit file does not cover this Target; prefit required`。
  - Target 只有 TP0 打了一次 `[CLEAR] prefit started`。`[CLEAR fit_state]` 到 `ready`，原因 `prefit`。`prefit driver finished state=ready elapsed_s=160.927` 只有一次。第一次 `prefit_commit` 因缺 H 格 `[1, 8, 256]` 返回 `missing`，补了一组 `["v", 1, 128, 8]`；下一次提交 `ready`，`refill_groups` 为空。
  - Draft 写了一次拟合文件，`cells=90`。文件 schema 2，有 `h` 90 行和 `layers` 18 行，没有 Q。H 行样本数最少 4。层行 `n` 最少 15，path 都是 `graph`。写盘后 sha256 是 `2689232b99aea25eeb7032d786dec72f44f7f6d395f613ba071b1b2e0d2f7d1f`，四轮结束后没变，没有第二次 `wrote fit file`。全程 `[CLEAR] fit write` 24796 条，path 全是 `graph`。
  - `accept_logs.py` 把预拟合和四轮正式流量分开。正式 `d_round` 按 Draft 时钟对齐到四次客户端区间，并把区间之间各 1 秒的尾轮归到前一轮。四轮 `prefit` 都为空，`hold_invalid` 都为 0。
  - `01` RSTeller 并发 1：100/100，输出 token 45892，正式 wall 851.193 s，吞吐 53.915 tok/s。接受率 34169/94896 = 0.360068。fallback 0。客户端并发 1.000。这一段 `d_round` 11979 条，11878 条同时有 `hold_ms` 和 `q_ms`，101 条排除且 `hold_ms` 为空，排除原因都是 `prefill/finish/new_rid`。Draft rank 0 已刷窗口 374 个（373 个 `full`，1 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 11964，`opt_bonus_requests` 11964，`opt_bonus_graph` 11964，eager 0。flush 是 TP0 与 TP1 的 `Cache flushed successfully`，以及 HTTP 200。
  - `02` RSTeller 并发 4：100/100，输出 token 45957，正式 wall 312.799 s，吞吐 146.922 tok/s。接受率 31387/117344 = 0.267479。fallback 0。客户端并发 3.952。`d_round` 3849 条，3657 条同时有 `hold_ms` 和 `q_ms`，192 条排除且 `hold_ms` 为空，排除原因是 `prefill` 96、`finish` 99、`new_rid` 96。已刷窗口 121 个（120 个 `full`，1 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 3850，`opt_bonus_requests` 14679，`opt_bonus_graph` 3850，eager 0。`requests` 大于 `forwards`。flush 同样是两个 TP 和 HTTP 200。
  - `03` WaterCaption 并发 1：100/100，输出 token 29585，正式 wall 655.562 s，吞吐 45.129 tok/s。接受率 22838/55160 = 0.414032。fallback 0。客户端并发 1.000。`d_round` 6946 条，6845 条同时有 `hold_ms` 和 `q_ms`，101 条排除且 `hold_ms` 为空，排除原因都是 `prefill/finish/new_rid`。已刷窗口 218 个（217 个 `full`，1 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 6913，`opt_bonus_requests` 6913，`opt_bonus_graph` 6913，eager 0。flush 同样是两个 TP 和 HTTP 200。
  - `04` WaterCaption 并发 4：100/100，输出 token 29820，正式 wall 306.441 s，吞吐 97.311 tok/s。接受率 21002/71352 = 0.294344。fallback 0。客户端并发 3.984。`d_round` 2323 条，2141 条同时有 `hold_ms` 和 `q_ms`，182 条排除且 `hold_ms` 为空，排除原因是 `prefill` 93、`finish` 95、`new_rid` 93。已刷窗口 75 个（72 个 `full`，3 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 2233，`opt_bonus_requests` 8487，`opt_bonus_graph` 2233，eager 0。`requests` 大于 `forwards`。flush 同样是两个 TP 和 HTTP 200。
  - Target 捕图 batch 1、2、4 都包含 ntpb 1、3、5、9、17、33、65。Draft 捕到 38 张图，batch 1 和 2 含 ntpb 65，batch 4 没有 ntpb 65。
- 总结：移开旧文件后，扫描由 Target rank 0 启动，必需格补齐后只写了一次 schema 2，文件不含 Q。四轮正式请求都在同一对进程上跑完，可写轮都有 `hold_ms` 和 `q_ms`，`hold_invalid` 为 0。接受率按总接受 token 除以总草稿 token。四轮 flush 都是 HTTP 200。清理时身份核对后，只向本场容器 pid 发了 TERM。Draft 进程已退出，本机 GPU 0 没有计算进程，30001 已释放。Target 主进程已退出，该 session 没有存活成员；30000 已释放，卡 2 和 3 没有计算进程。两条既有隧道的 PID 和启动时间未变。
