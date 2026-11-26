# 运行

## 事先

- 运行号：2026-11-26-e
- 要回答的问题：装入 26-d 留下的 RPD 拟合、不重扫时，当前代码的 `build_clear_rpd_input` 能否让同一对进程跑完 RSTeller 并发 1 和并发 4，以及正式轮的 `rpd_build_us` 落在哪里。
- 对照的上一次保留运行：2026-11-26-d
- 相对那次运行改了什么：服务开关、设备、容量、负载和客户端沿用 26-d。只换当前代码（RPD 主机拓扑不再对每条边写 tensor 标量），并且不移走、不重扫拟合。Draft 在清掉 `SGLANG_CLEAR_*` 之后仍导出账本、KV reuse、bonus、`SGLANG_CLEAR_TARGET_BACKEND=npu`、`SGLANG_CLEAR_DRAFT_LOOKAHEAD=1`，并显式导出 `SGLANG_CLEAR_ACCEPT_HALF_LIFE=1000`。Draft 不传 `--speculative-verify-mode`。Target 仍导出 lookahead `0`，不导出半衰期，并带 `--speculative-verify-mode rpd` 和 `--speculative-rpd-tau 0.2`。不设置 `SGLANG_CLEAR_ACCEPT_PER_REQUEST`、`SGLANG_CLEAR_WPAIR_METRIC`、`SGLANG_CLEAR_WPAIR_RULE`、`SGLANG_CLEAR_DEFER_CLOSE`、`SGLANG_CLEAR_LAYER_DEPTH`、`SGLANG_CLEAR_H_CURVE`、`SGLANG_CLEAR_LAYER_SLOPE_PRIOR`、`SGLANG_CLEAR_FIT_PATH` 或 `SGLANG_CLEAR_AUTO_PREFIT=0`，不加 `--api-key`。`clear_tree_fit.json`、`clear_tree_fit.rpd.json`、`clear_tree_fit.bak.json`、`clear_tree_fit.bak2.json` 留在包目录。缺文件只记 `ABSENT`。发客户端之前要看到：Draft 对 `clear_tree_fit.rpd.json` 打 `stashed fit file`，随后 `installed` hold cells；Target 的 `[CLEAR fit_state]` 为 `ready`，原因 `ready`。出现该路径的 `no fit file`、`does not match`、`does not cover` 或 `prefit started` 时不发客户端。Target 用物理卡 2 和 3。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、bonus 开、max-running-requests 4、捕图 batch 1 2 4。容量是 Target 65、Draft 64。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。显式 `--clear-tree-max-depth 8`。先 Draft 后 Target。`01` 和 `02` 同一对进程。客户端在 NPU 容器里打 Target。两轮都用 `eval_spectre_rsteller_serving.py`。数据是 26-d 在 NPU 上用过的那一份。每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。`--img-dir` 用数据集根目录。计时字段只记日志，不进拟合。

试验从 `01` 编号。`tools/` 放这场的启动脚本。`raw/01/` 和 `raw/02/` 放客户端输出，双端日志放在 `raw/`。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 1000，Target RPD τ=0.2，装入既有拟合 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 1000，Target RPD τ=0.2，装入既有拟合 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 本机和 NPU 的包清单在同步后一致。开跑前本机 `clear_tree_fit.json`、`clear_tree_fit.bak.json`、`clear_tree_fit.bak2.json` 都是 `ABSENT`。`clear_tree_fit.rpd.json` 的 sha256 开跑前和两轮结束后都是 `0d872ad58e3e08860966c3247bed6b2188be7da5c40a8608826711d9f44e0feb`。NPU 包目录开跑前四份都是 `ABSENT`；同步把同一份 `.rpd.json` 拷了过去，没有删本机这份。没有 `prefit started`、`wrote fit file`、`does not match`、`does not cover`、`rpd_flat` 或 `W/K use the flat table`。Draft 对 greedy 的 `clear_tree_fit.json` 打了一次 `no fit file`，对 `clear_tree_fit.rpd.json` 打了一次 `stashed fit file`，随后 `installed 90 hold cells from the fit file`，并打了一次 `[CLEAR] target verifies with RPD; W/K run online on fit file`。Target 两个 TP 的 `[CLEAR fit_state]` 都是 `ready`，原因 `ready`。两个 TP 都打了一次 `Speculative RPD verify path: npu_sr_host_v1`。
  - Draft host pid 3438067。Target host pid 857261，物理卡 2 和 3。拟合装入并且 `/health` 为 200 之后才发客户端，两轮之间没有重启。数据 sha256 是 `f9a74112455c63c67e566e68d7b7c520769c67bcb27a7f960054ca6e1beabbbc`，100 条。
  - 接受率是总 `spec_accept_token_num / spec_draft_token_num`。`e_pred / accepted` 来自正式打分轮。平表按当前实现：batch 1 是 W=8、K=16，更大的 batch 是 W=4、K=8。`d_round.rpd_masks` 在打分轮全是空，`d_round` 里没有 `accept_k`。晚到掩码在 `d_rpd`。
  - 拟合 `ready` 之后的 Target `t_verify` 有 13874 条，全部是 `path=graph`，没有 `fallback`。`verify_wall_us`、`accept_ready_us`、`rpd_build_us`、`rpd_stats_us`、`rpd_select_us`、`rpd_mask_us` 都是非负整数，没有 null，也没有 `verify_wall_us` 小于 `accept_ready_us` 的行。中位数分别是 43391、41228、173、33281、64、67 微秒。26-d 同一口径的中位数是 43254、41110、875、33003、66、67。`rpd_build_us` 的中位数从 875 降到 173。按 (b, K) 看，下降随验证长度变大：b=1、K=8 是 138 对 441，K=16 是 149 对 585，K=32 是 173 对 875，K=64 是 217 对 1374；b=4、K=8 是 197 对 898，K=16 是 241 对 1471。这些字段只在日志里。
  - `01` RSTeller 并发 1：100/100，输出 token 46022，正式 wall 764.234 s，吞吐 60.220 tok/s。接受率 35600/84496 = 0.421322。fallback 0。客户端并发 1.000。`d_round` 10562 条，10462 条同时有 `hold_ms` 和 `q_ms`，100 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 10462，`k_mode` 全是 `value`。其中 7988 轮离开平表，最多的是 W=8、K=32，共 6978 轮；留在 W=8、K=16 的有 2474 轮。`accepted` 35083，`e_pred` 35029.462217，`e_mass` 36021.906336，`e_pred/accepted` 0.998474。`e_rid` 10462 行 `path_ok` 为真，其中 9729 行 `ranks` 非空，733 行 `ranks` 是空列表。打分轮里 10362 条有 `d_rpd`，`rpd_masks` 合计 10362，`accept_k` 行数 10362，`rpd_ok` 为假的行数 0。另外 100 条打分轮没有 `d_rpd`。
  - `02` RSTeller 并发 4：100/100，输出 token 45152，正式 wall 278.918 s，吞吐 161.883 tok/s。接受率 33124/97392 = 0.340110。fallback 0。客户端并发 3.924。`d_round` 3134 条，2940 条同时有 `hold_ms` 和 `q_ms`，194 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 2940，`k_mode` 是 `value` 2921、`probe` 16、`tau` 3。其中 2109 轮离开平表。batch 4 最多的是 W=4、K=16，共 1771 轮；W=4、K=8 有 814 轮。`accepted` 31332，`e_pred` 31303.982891，`e_mass` 32159.740986，`e_pred/accepted` 0.999106。`e_rid` 11404 行 `path_ok` 为真，其中 10150 行 `ranks` 非空，1254 行 `ranks` 是空列表。打分轮里 2939 条有 `d_rpd`，`rpd_masks` 合计 11403，`accept_k` 行数 11403，`rpd_ok` 为假的行数 0。剩下 1 条没有对上。
  - 另有 177 条 `d_round` 的请求不在两份正式 JSONL 里，没有并进上面两轮。
  - Target 捕到 batch 1、2、4，且各含 ntpb 1、3、5、9、17、33、65。Draft 捕到 38 张图，batch 1 和 2 含 ntpb 65，batch 4 没有 ntpb 65。rank 0 已刷窗口 435 个，432 个 `full`、3 个 `idle`，末窗 `idle`，尾数已刷出。增量合计 `opt_bonus_forwards` 13874、`opt_bonus_graph` 13874、`opt_bonus_eager` 0、`opt_lookahead_layers` 68293、`opt_lookahead_group_templates` 12。窗口计数里没有 `opt_lookahead_fallbacks`。
  - 日志里 `POST /flush_cache` 是 2 次 HTTP 200，两个 TP 各有 2 次 `Cache flushed successfully`，没有其他状态。两份 stdout 都在 flush 之后进入正式计时。
  - 26-d 的 RSTeller 对照是并发 1 吞吐 59.553 tok/s、接受率 0.419254，并发 4 吞吐 159.523 tok/s、接受率 0.334098。
- 总结：两轮在同一对进程上跑完。现有 RPD 拟合被装入，没有重新预拟合，正式打分轮离开平表。`rpd_build_us` 的中位数从 26-d 的 875 微秒降到 173 微秒，下降幅度随 b 和 K 变大。`d_round` 仍然没有掩码；对上的晚到掩码都有 `accept_k`，没有 `rpd_ok` 为假的行。并发 1 有 100 条打分轮没有 `d_rpd`。并发 4 有 1 条没有对上。清理时身份核对后，只向本场 session 里的容器 pid 发了 TERM。Draft session 没有存活成员，本机 GPU 0 没有计算进程，30001 已释放。Target 主进程已退出，30000 已释放，物理卡 2 和 3 没有计算进程。Target session 只剩僵尸，不是在跑的服务。两条既有隧道的 PID、UID、启动 ticks 和 boot ID 未变。

