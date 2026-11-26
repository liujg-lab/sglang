# 运行

## 事先

- 运行号：2026-11-26-c
- 要回答的问题：当前代码在 RPD 下是否离开平表握手，只把预拟合写入 `clear_tree_fit.rpd.json`，不改 greedy 的 `clear_tree_fit.json`；同一对进程随后是否跑完 RSTeller 并发 1 和并发 4。
- 对照的上一次保留运行：2026-11-26-b
- 相对那次运行改了什么：服务开关、设备、容量、负载和客户端沿用 26-b。只换当前代码，并重新预拟合。Draft 在清掉 `SGLANG_CLEAR_*` 之后仍导出账本、KV reuse、bonus、`SGLANG_CLEAR_TARGET_BACKEND=npu`、`SGLANG_CLEAR_DRAFT_LOOKAHEAD=1`，并显式导出 `SGLANG_CLEAR_ACCEPT_HALF_LIFE=1000`。Draft 不传 `--speculative-verify-mode`。Target 仍导出 lookahead `0`，不导出半衰期，并带 `--speculative-verify-mode rpd` 和 `--speculative-rpd-tau 0.2`。不设置 `SGLANG_CLEAR_ACCEPT_PER_REQUEST`、`SGLANG_CLEAR_WPAIR_METRIC`、`SGLANG_CLEAR_WPAIR_RULE`、`SGLANG_CLEAR_DEFER_CLOSE`、`SGLANG_CLEAR_LAYER_DEPTH`、`SGLANG_CLEAR_H_CURVE`、`SGLANG_CLEAR_LAYER_SLOPE_PRIOR`、`SGLANG_CLEAR_FIT_PATH` 或 `SGLANG_CLEAR_AUTO_PREFIT=0`，不加 `--api-key`。greedy 的 `clear_tree_fit.json` 留在原地。若开跑时已有 `clear_tree_fit.rpd.json`，移到本场 `raw/sync/`，不删除。RPD 缺这份文件时，FIT 握手应走 `need_prefit`。发客户端之前要看到：Draft 对 `.rpd.json` 打 `no fit file`；打一次 `[CLEAR] target verifies with RPD; W/K run online on fit file`；Target 打 `[CLEAR] prefit started`，随后 `[CLEAR fit_state]` 为 `ready` 且原因 `prefit`，Draft 的 `wrote fit file` 指向 `clear_tree_fit.rpd.json`。出现 `rpd_flat`、`W/K use the flat table`、`fit_state` 为 `disabled`、把 greedy 格子装进这次 RPD，或写盘改了 `clear_tree_fit.json` 时不发客户端。Target 用物理卡 2 和 3。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、bonus 开、max-running-requests 4、捕图 batch 1 2 4。容量是 Target 65、Draft 64。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。显式 `--clear-tree-max-depth 8`。先 Draft 后 Target。`01` 和 `02` 同一对进程。客户端在 NPU 容器里打 Target。两轮都用 `eval_spectre_rsteller_serving.py`。数据是 26-b 在 NPU 上用过的那一份。每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。`--img-dir` 用数据集根目录。

试验从 `01` 编号。`tools/` 放这场的启动脚本。`raw/01/` 和 `raw/02/` 放客户端输出，双端日志放在 `raw/`。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 1000，Target RPD τ=0.2，重新预拟合 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 1000，Target RPD τ=0.2，重新预拟合 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 本机和 NPU 的包清单在同步后一致。两端开跑前都没有 `clear_tree_fit.rpd.json`。greedy 的 `clear_tree_fit.json` sha256 开跑前、预拟合后和两轮结束后都是 `34e56789f083d9f6b75010e4f3f14948f4b2b5cac2db6aab327fdf0ce29f3dcb`。Draft 打了一次 `no fit file`，路径是 `clear_tree_fit.rpd.json`，并打了一次 `[CLEAR] target verifies with RPD; W/K run online on fit file`。没有 `rpd_flat`，也没有 `W/K use the flat table`。Target 两个 TP 的 `[CLEAR fit_state]` 都是 `ready`，原因 `prefit`。`prefit driver finished state=ready elapsed_s=205.484`。Draft 只打了一次 `wrote fit file`，路径是 `clear_tree_fit.rpd.json`，`cells=90`。这份文件是 schema 3，头里 `verify_mode` 是 `rpd`，不含 Q，sha256 是 `17a46968d2d157e3fe44e16506904647afbe38fbfa27379b2e05085fc590bd19`。随后 `hold anchors installed cells=90 curves=18 curve=on`。Target 两个 TP 都打了一次 `Speculative RPD verify path: npu_sr_host_v1`。
  - Draft host pid 2797985，环境里 `SGLANG_CLEAR_ACCEPT_HALF_LIFE=1000`。Target host pid 3367759，物理卡 2 和 3。预拟合落盘之后才发客户端，两轮之间没有重启。数据 sha256 是 `f9a74112455c63c67e566e68d7b7c520769c67bcb27a7f960054ca6e1beabbbc`，100 条。
  - 接受率是总 `spec_accept_token_num / spec_draft_token_num`。`e_pred / accepted` 来自正式打分轮。平表按当前实现：batch 1 是 W=8、K=16，更大的 batch 是 W=4、K=8。`d_round.rpd_masks` 在打分轮全是空，`d_round` 里没有 `accept_k`。晚到掩码在 `d_rpd`。
  - `01` RSTeller 并发 1：100/100，输出 token 45155，正式 wall 755.747 s，吞吐 59.749 tok/s。接受率 34970/82792 = 0.422384。fallback 0。客户端并发 1.000。`d_round` 10349 条，10249 条同时有 `hold_ms` 和 `q_ms`，100 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 10249，`k_mode` 全是 `value`。其中 7822 轮离开平表，最多的是 W=8、K=32，共 6814 轮；留在 W=8、K=16 的有 2427 轮。`accepted` 34444，`e_pred` 34430.224796，`e_mass` 35472.868663，`e_pred/accepted` 0.999600。`e_rid` 10249 行 `path_ok` 为真，其中 9533 行 `ranks` 非空，716 行 `ranks` 是空列表。打分轮里 10149 条有 `d_rpd`，`rpd_masks` 合计 10149，`accept_k` 行数 10149，`rpd_ok` 为假的行数 0。另外 100 条打分轮没有 `d_rpd`，位置散在整段里，不单是进程结束时的 8 轮尾数；最后一条打分轮也在其中。
  - `02` RSTeller 并发 4：100/100，输出 token 45615，正式 wall 286.434 s，吞吐 159.251 tok/s。接受率 33209/100104 = 0.331745。fallback 0。客户端并发 3.908。`d_round` 3243 条，3053 条同时有 `hold_ms` 和 `q_ms`，190 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 3053，`k_mode` 是 `value` 3039、`probe` 11、`tau` 3。其中 1863 轮离开平表。batch 4 最多的是 W=4、K=16，共 1506 轮；W=4、K=8 有 1157 轮。`accepted` 31495，`e_pred` 31430.304095，`e_mass` 32206.617608，`e_pred/accepted` 0.997946。`e_rid` 11759 行 `path_ok` 为真，其中 10399 行 `ranks` 非空，1360 行 `ranks` 是空列表。打分轮里 3052 条有 `d_rpd`，`rpd_masks` 合计 11758，`accept_k` 行数 11758，`rpd_ok` 为假的行数 0。剩下 1 条是该试验最后一条打分轮，记为 tail=missing。
  - 另有 176 条 `d_round` 的请求不在两份正式 JSONL 里，没有并进上面两轮。
  - Target 捕到 batch 1、2、4，且各含 ntpb 1、3、5、9、17、33、65。Draft 捕到 38 张图，batch 1 和 2 含 ntpb 65，batch 4 没有 ntpb 65。rank 0 已刷窗口 478 个，473 个 `full`、5 个 `idle`，末窗 `idle`，尾数已刷出。增量合计 `opt_bonus_forwards` 15191、`opt_bonus_graph` 15191、`opt_bonus_eager` 0、`opt_lookahead_layers` 72418、`opt_lookahead_fallbacks` 0、`opt_lookahead_group_templates` 24。
  - 日志里 `POST /flush_cache` 是 2 次 HTTP 200，两个 TP 各有 2 次 `Cache flushed successfully`，没有其他状态。两份 stdout 都在 flush 之后进入正式计时。
  - `clear_rpd_accept_curve.py` 读完整份 Draft 日志，含预拟合轮：`rounds=15190 late=14542 excluded=1226 no_masks=14343 rids=23907 rpd_not_ok=0`。匹配集的档差是 b=1 的 1.8%（K′=2–32）和 b=2–4 的 1.4%（K′=2–16）。26-b 的 RSTeller 对照是并发 1 吞吐 58.972 tok/s、接受率 0.357794，并发 4 吞吐 158.659 tok/s、接受率 0.287793，并且 W/K 留在平表。
- 总结：两轮在同一对进程上跑完。RPD 没有停在 `rpd_flat`。预拟合只写了 `clear_tree_fit.rpd.json`，greedy 的 `clear_tree_fit.json` 哈希没变。正式打分轮的 `k_mode` 以 `value` 为主，W 和 K 都有离开平表的轮。`d_round` 仍然没有掩码；对上的晚到掩码都有 `accept_k`，没有 `rpd_ok` 为假的行。并发 1 有 100 条打分轮没有 `d_rpd`。并发 4 只有最后一条还没对上。清理时身份核对后，只向本场 session 里的容器 pid 发了 TERM。Draft session 没有存活成员，本机 GPU 0 没有计算进程，30001 已释放。Target 主进程已退出，30000 已释放，物理卡 2 和 3 没有计算进程。Target session 只剩僵尸，不是在跑的服务。两条既有隧道的 PID、UID、启动 ticks 和 boot ID 未变。

