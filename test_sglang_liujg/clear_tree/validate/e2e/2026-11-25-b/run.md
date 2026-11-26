# 运行

## 事先

- 运行号：2026-11-25-b
- 要回答的问题：当前代码在同一对进程上先跑 RSTeller、再跑 WaterCaption 时，正式请求是否跑完，半衰期 1000 的 `e_pred` 是否仍贴近 `accepted`，以及正式 `d_round` 是否带上 `e_rid`（`path_ok`、`ranks`、`next_rank`）。
- 对照的上一次保留运行：2026-11-25-a
- 相对那次运行改了什么：只保留 25-a 半衰期 1000 那一块，不重启做半衰期 0。服务开关、负载、设备和客户端沿用那一块。Draft 在清掉 `SGLANG_CLEAR_*` 之后仍导出账本、KV reuse、bonus、`SGLANG_CLEAR_TARGET_BACKEND=npu`、`SGLANG_CLEAR_DRAFT_LOOKAHEAD=1`，并显式导出 `SGLANG_CLEAR_ACCEPT_HALF_LIFE=1000`。Target 仍导出 lookahead `0`，不导出半衰期。不设置 `SGLANG_CLEAR_ACCEPT_PER_REQUEST`、`SGLANG_CLEAR_WPAIR_METRIC`、`SGLANG_CLEAR_WPAIR_RULE`、`SGLANG_CLEAR_DEFER_CLOSE`、`SGLANG_CLEAR_LAYER_DEPTH`、`SGLANG_CLEAR_H_CURVE`、`SGLANG_CLEAR_LAYER_SLOPE_PRIOR` 或 `SGLANG_CLEAR_AUTO_PREFIT=0`，不加 `--api-key`。不移走包内 `clear_tree_fit.json`。出现 `no fit file`、`wrote fit file` 或 `prefit started` 时不发客户端。要看到安装 hold cells，Target 的 `[CLEAR fit_state]` 为 `ready` 且原因不是 `prefit`。Target 用物理卡 2 和 3。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、bonus 开、max-running-requests 4、捕图 batch 1 2 4。容量是 Target 65、Draft 64。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。显式 `--clear-tree-max-depth 8`。先 Draft 后 Target。`01` 到 `04` 同一对进程。客户端在 NPU 容器里打 Target。`01`、`02` 用 `eval_spectre_rsteller_serving.py`，`03`、`04` 用 `eval_spectre_watercaption_serving.py`。每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。`--img-dir` 用数据集根目录。数据仍是 25-a 在 NPU 上用过的那一份。

试验从 `01` 编号。`tools/` 放这场的启动脚本。`raw/01/` 到 `raw/04/` 放客户端输出，双端日志放在 `raw/`。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，安装现有拟合，半衰期 1000 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 1000 | 1 | 正式 |
| 03 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 1，同一进程，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 1000 | 1 | 正式 |
| 04 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 1000 | 1 | 正式 |

## 结果

- 11 月 25 日第一次预检时物理卡 2 和 3 被其他容器占用，证据在 `raw/sync/precheck-cards.txt`，当时没有同步、没有启动。11 月 26 日两张卡空闲后继续本场。
- 关键数字（指向 `raw/<试验名>/`）：
  - 本机和 NPU 的包清单不一致。备份后只同步了 ClearTree 包，同步后清单一致。本机拟合 sha256 开跑前和结束后都是 `34e56789f083d9f6b75010e4f3f14948f4b2b5cac2db6aab327fdf0ce29f3dcb`。NPU 包里本来没有拟合文件。Draft 是 `stashed fit file` 一次、`installed 90 hold cells` 一次、`hold anchors installed cells=90 curves=18 curve=on` 一次。没有 `no fit file`、`wrote fit file` 或 `prefit started`。`accept_logs.py` 的 `prefit_status` 是 `reused`。Target 两个 TP 的 `[CLEAR fit_state]` 都是 `ready`，原因 `ready`。
  - Draft host pid 1620684，环境里 `SGLANG_CLEAR_ACCEPT_HALF_LIFE=1000`，没有 `SGLANG_CLEAR_ACCEPT_PER_REQUEST`。Target host pid 192768，物理卡 2 和 3。四轮之间没有重启。
  - 容器临时目录里还有更早的四份客户端输出，时间在本场 `01` 之前，已挪到 `raw/sync/earlier-client/`，没有并进下面四轮。它们留下 23231 条对不上本场客户端的正式轮。
  - 接受率是总 `spec_accept_token_num / spec_draft_token_num`。`e_pred / accepted` 来自正式 `d_round` 里同时有 `accepted`、`e_pred`、`e_mass` 的轮。`path_ok` 按请求行计。
  - `01` RSTeller 并发 1：100/100，输出 token 46059，正式 wall 779.629 s，吞吐 59.078 tok/s。接受率 35342/86952 = 0.406569。fallback 0。客户端并发 1.000。`d_round` 10869 条，10769 条同时有 `hold_ms` 和 `q_ms`，100 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 10769，`accepted` 34853，`e_pred` 34860.035630，`e_mass` 37209.564173，`e_pred/accepted` 1.000202。前半 0.934263/0.930677，后半 0.939484/0.942735。`e_rid` 10769 轮，全部 `path_ok`，10769 行都有 `ranks`，其中 1203 行有 `next_rank`。
  - `02` RSTeller 并发 4：100/100，输出 token 45515，正式 wall 284.051 s，吞吐 160.235 tok/s。接受率 32855/102208 = 0.321452。fallback 0。客户端并发 3.921。`d_round` 3294 条，3101 条同时有 `hold_ms` 和 `q_ms`，193 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 3293，`accepted` 32413，`e_pred` 32355.534214，`e_mass` 34385.650071，`e_pred/accepted` 0.998227。前半 0.936660/0.936381，后半 0.945526/0.949268。`e_rid` 3293 轮，全部 `path_ok`，12580 行都有 `ranks`，其中 1770 行有 `next_rank`。
  - `03` WaterCaption 并发 1：100/100，输出 token 29418，正式 wall 580.557 s，吞吐 50.672 tok/s。接受率 23131/51632 = 0.447998。fallback 0。客户端并发 1.000。`d_round` 6454 条，6354 条同时有 `hold_ms` 和 `q_ms`，100 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 6354，`accepted` 22626，`e_pred` 22658.767781，`e_mass` 23181.889432，`e_pred/accepted` 1.001448。前半 0.970616/0.963826，后半 0.984419/0.988515。`e_rid` 6354 轮，全部 `path_ok`，6354 行都有 `ranks`，其中 1130 行有 `next_rank`。
  - `04` WaterCaption 并发 4：100/100，输出 token 29932，正式 wall 267.365 s，吞吐 111.952 tok/s。接受率 22074/64016 = 0.344819。fallback 0。客户端并发 3.969。`d_round` 2049 条，1865 条同时有 `hold_ms` 和 `q_ms`，184 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 2049，`accepted` 21603，`e_pred` 21636.715167，`e_mass` 22136.159994，`e_pred/accepted` 1.001561。前半 0.978713/0.972326，后半 0.976150/0.979539。`e_rid` 2049 轮，全部 `path_ok`，7806 行都有 `ranks`，其中 1479 行有 `next_rank`。
  - Target 捕到 batch 1、2、4，且各含 ntpb 1、3、5、9、17、33、65。Draft 捕到 38 张图，batch 1 和 2 含 ntpb 65，batch 4 没有 ntpb 65。rank 0 已刷窗口 1437 个，1434 个 `full`、3 个 `idle`，末窗 `idle`，尾数已刷出。这些窗口覆盖整段 Draft 日志，含更早那一轮。增量合计 `opt_bonus_forwards` 45831、`opt_bonus_graph` 45831、`opt_bonus_eager` 0、`opt_lookahead_layers` 226857、`opt_lookahead_fallbacks` 0、`opt_lookahead_group_templates` 12。
  - 日志里 `POST /flush_cache` 是 8 次 HTTP 200，两个 TP 各有 8 次 `Cache flushed successfully`，没有其他状态。本场四份 stdout 都在 flush 之后进入正式计时，没有 flush 失败字样。
- 总结：四轮在同一对进程上跑完。先跑 RSTeller 再跑 WaterCaption 时，半衰期 1000 的 WaterCaption 正式轮 `e_pred/accepted` 是 1.001448 和 1.001561。四轮打分轮的 `e_rid` 都有 `path_ok` 和 `ranks`。`01` 开始前池子已经见过更早那一轮的 RSTeller 和 WaterCaption，不是从先验重新开始。清理时身份核对后，只向本场容器 pid 发了 TERM。Draft session 没有存活成员，本机 GPU 0 没有计算进程，30001 已释放。Target 主进程已退出，30000 已释放，物理卡 2 和 3 没有计算进程。Target session 只剩僵尸，不是在跑的服务。两条既有隧道的 PID、UID、启动 ticks 和 boot ID 未变。
