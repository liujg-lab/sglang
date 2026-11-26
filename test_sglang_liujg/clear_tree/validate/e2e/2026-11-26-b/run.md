# 运行

## 事先

- 运行号：2026-11-26-b
- 要回答的问题：当前代码在与 2026-11-26-a 相同的条件下，正式轮的晚到 RPD 掩码是否写入 `d_rpd` 并带上 `accept_k`。`d_round.rpd_masks` 仍为 null 符合晚到路径，不把它当成修正失败。
- 对照的上一次保留运行：2026-11-26-a
- 相对那次运行改了什么：服务开关、设备、容量、负载和客户端沿用 26-a。只换当前代码：关账没对上报告时留下最近 8 轮，晚到的 `rpd` 另打 `[CLEAR ledger d_rpd]`。Draft 在清掉 `SGLANG_CLEAR_*` 之后仍导出账本、KV reuse、bonus、`SGLANG_CLEAR_TARGET_BACKEND=npu`、`SGLANG_CLEAR_DRAFT_LOOKAHEAD=1`，并显式导出 `SGLANG_CLEAR_ACCEPT_HALF_LIFE=1000`。Draft 不传 `--speculative-verify-mode`。Target 仍导出 lookahead `0`，不导出半衰期，并增加 `--speculative-verify-mode rpd` 和 `--speculative-rpd-tau 0.2`。不设置 `SGLANG_CLEAR_ACCEPT_PER_REQUEST`、`SGLANG_CLEAR_WPAIR_METRIC`、`SGLANG_CLEAR_WPAIR_RULE`、`SGLANG_CLEAR_DEFER_CLOSE`、`SGLANG_CLEAR_LAYER_DEPTH`、`SGLANG_CLEAR_H_CURVE`、`SGLANG_CLEAR_LAYER_SLOPE_PRIOR` 或 `SGLANG_CLEAR_AUTO_PREFIT=0`，不加 `--api-key`。不移走包内 `clear_tree_fit.json`。RPD 下 FIT 握手应回 `rpd_flat`，Target 的 `[CLEAR fit_state]` 为 `disabled`、原因 `rpd_flat`，不安装 hold cells。出现 `no fit file`、`wrote fit file` 或 `prefit started` 时不发客户端。Draft 应打一次 `[CLEAR] target verifies with RPD; W/K use the flat table`。W 或 K 离开平表记为不符合当前 RPD 规则。Target 用物理卡 2 和 3。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、bonus 开、max-running-requests 4、捕图 batch 1 2 4。容量是 Target 65、Draft 64。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。显式 `--clear-tree-max-depth 8`。先 Draft 后 Target。`01` 和 `02` 同一对进程。客户端在 NPU 容器里打 Target。两轮都用 `eval_spectre_rsteller_serving.py`。数据是 26-a 在 NPU 上用过的那一份。每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。`--img-dir` 用数据集根目录。

试验从 `01` 编号。`tools/` 放这场的启动脚本。`raw/01/` 和 `raw/02/` 放客户端输出，双端日志放在 `raw/`。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 1000，Target RPD τ=0.2，不重拟合 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 1000，Target RPD τ=0.2，不重拟合 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 本机和 NPU 的包清单在同步后一致。NPU 同步前没有 `clear_tree_fit.json`，同步后和结束后仍然没有。本机拟合 sha256 开跑前和结束后都是 `34e56789f083d9f6b75010e4f3f14948f4b2b5cac2db6aab327fdf0ce29f3dcb`。没有 `no fit file`、`wrote fit file` 或 `prefit started`。Target 两个 TP 的 `[CLEAR fit_state]` 都是 `disabled`，原因 `rpd_flat`。Draft 打了一次 `[CLEAR] target verifies with RPD; W/K use the flat table`。Target 两个 TP 都打了一次 `Speculative RPD verify path: npu_sr_host_v1`。客户端记录里 `speculative_verify_mode` 是 `rpd`，`speculative_rpd_tau` 是 0.2。
  - Draft host pid 2357652，环境里 `SGLANG_CLEAR_ACCEPT_HALF_LIFE=1000`。Target host pid 2216227，物理卡 2 和 3。两轮之间没有重启。数据 sha256 是 `f9a74112455c63c67e566e68d7b7c520769c67bcb27a7f960054ca6e1beabbbc`，100 条。
  - 接受率是总 `spec_accept_token_num / spec_draft_token_num`。`e_pred / accepted` 来自正式打分轮。平表按当前实现：batch 1 是 W=8、K=16，更大的 batch 是 W=4、K=8。`d_round.rpd_masks` 全是 null，`d_round` 里没有 `accept_k`。晚到掩码在 `d_rpd`。
  - `01` RSTeller 并发 1：100/100，输出 token 45399，正式 wall 769.845 s，吞吐 58.972 tok/s。接受率 33729/94264 = 0.357794。fallback 0。客户端并发 1.000。`d_round` 11783 条，11683 条同时有 `hold_ms` 和 `q_ms`，100 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 11683，`accepted` 33298，`e_pred` 33261.309412，`e_mass` 34268.176395，`e_pred/accepted` 0.998898。`e_rid` 11683 行 `path_ok` 为真，其中 10729 行 `ranks` 非空，954 行 `ranks` 是空列表。K 全是 16，W 全是 8，`k_mode` 在打分轮是 `tau`。打分轮里 11583 条有 `d_rpd`，`rpd_masks` 合计 11583，`accept_k` 行数 11583，`rpd_ok` 为假的行数 0。另有 100 条打分轮没有 `d_rpd`，不是进程结束时的 8 轮尾数。
  - `02` RSTeller 并发 4：100/100，输出 token 45502，正式 wall 286.791 s，吞吐 158.659 tok/s。接受率 31800/110496 = 0.287793。fallback 0。客户端并发 3.923。`d_round` 3558 条，3368 条同时有 `hold_ms` 和 `q_ms`，190 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 3368，`accepted` 30120，`e_pred` 30083.445373，`e_mass` 31097.769045，`e_pred/accepted` 0.998786。`e_rid` 13056 行 `path_ok` 为真，其中 11431 行 `ranks` 非空，1625 行 `ranks` 是空列表。batch 1 的轮是 W=8、K=16，更大 batch 的轮是 W=4、K=8，没有离开平表。`k_mode` 全是 `tau`。打分轮里 3367 条有 `d_rpd`，`rpd_masks` 合计 13055，`accept_k` 行数 13055，`rpd_ok` 为假的行数 0。剩下 1 条打分轮是该请求的最后一轮，日志收回时还没有对应的 `d_rpd`，记为 tail=missing。
  - 另有 206 条 `d_round` 的请求不在两份正式 JSONL 里，没有并进上面两轮。
  - Target 捕到 batch 1、2、4，且各含 ntpb 1、3、5、9、17、33、65。Draft 捕到 38 张图，batch 1 和 2 含 ntpb 65，batch 4 没有 ntpb 65。rank 0 已刷窗口 488 个，485 个 `full`、3 个 `idle`，末窗 `idle`，尾数已刷出。增量合计 `opt_bonus_forwards` 15548、`opt_bonus_graph` 15548、`opt_bonus_eager` 0、`opt_lookahead_layers` 65509、`opt_lookahead_fallbacks` 0、`opt_lookahead_group_templates` 4。
  - 日志里 `POST /flush_cache` 是 2 次 HTTP 200，两个 TP 各有 2 次 `Cache flushed successfully`，没有其他状态。两份 stdout 都在 flush 之后进入正式计时。
  - `clear_rpd_accept_curve.py` 读完整份 Draft 日志：`rounds=15547 late=15151 excluded=293 no_masks=15254 rids=24839 rpd_not_ok=0`。匹配集的档差是并发 1 的 1.5% 和并发 4 的 1.3%。26-a 的 RSTeller 对照是并发 1 吞吐 59.038 tok/s、接受率 0.357794，并发 4 吞吐 154.581 tok/s、接受率 0.287397。
- 总结：两轮在同一对进程上跑完。Target 用 RPD 主机路径 `npu_sr_host_v1`，拟合停在 `rpd_flat`，没有重扫。正式轮的 W 和 K 都落在平表上。`d_round` 仍然没有掩码；晚到的掩码写进了 `d_rpd`，这些行都有 `accept_k`，没有 `rpd_ok` 为假的行。并发 1 另有 100 条打分轮没有 `d_rpd`。并发 4 只有最后一轮还没对上，记为 missing。清理时身份核对后，只向本场 session 里的容器 pid 发了 TERM。Draft session 没有存活成员，本机 GPU 0 没有计算进程，30001 已释放。Target 主进程已退出，30000 已释放，物理卡 2 和 3 没有计算进程。Target session 只剩僵尸，不是在跑的服务。两条既有隧道的 PID、UID、启动 ticks 和 boot ID 未变。
