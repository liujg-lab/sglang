# 运行

## 事先

- 运行号：2026-11-22-d
- 要回答的问题：移开现有 `clear_tree_fit.json` 后，当前实现是否仍由 Target rank 0 的 FIT 握手驱动扫描，并在必需 H 格和层格各满 4 条后只写一次 schema 3；写出的每条 H 是否带有与 `samples_ms` 等长的 `ells`，且不含 Q。随后 RSTeller 与 WaterCaption 的并发 1 和并发 4 是否在同一对进程上跑完。正式轮的 `d_round` 是否带上 `hold_ms`、`q_ms`、`close_ms`、`close_deferred`，以及 `policy_recv_ms`、`policy_plan_ms`、`policy_grow_ms`、`policy_final_k_ms`，四项之和是否等于 `policy_ms`。正式 `d_layer` 是否带上 `last`、`depth_ratio`、`depth_src`。提交时 H 曲线是否为开启。
- 对照的上一次保留运行：2026-11-22-c
- 相对那次运行改了什么：Draft 与 Target 换成当前 `clear_tree_remote`。拟合文件是 schema 3。上一轮关账默认推迟到回包之后、`run_post_reply` 之前。层耗时另有按深度的乘法校正。H 按已提交前缀 ell 连成曲线。启动前把包内已有拟合文件移到本场 `raw/sync/`，不删除，迫使这次重扫。不设置 `SGLANG_CLEAR_DEFER_CLOSE`、`SGLANG_CLEAR_LAYER_DEPTH`、`SGLANG_CLEAR_H_CURVE` 或 `SGLANG_CLEAR_AUTO_PREFIT=0`，不加 `--api-key`。负载和服务开关不变。Target 用物理卡 2 和 3。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4、捕图 batch 1 2 4。容量是 Target 65、Draft 64。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。显式 `--clear-tree-max-depth 8`。先 Draft 后 Target，四轮之间不重启 Target，也不重启 Draft。Draft 打出 `no fit file`、Target 的 `[CLEAR fit_state]` 为 `ready` 且原因是 `prefit`、拟合文件落盘且 `prefit driver finished state=ready`、`hold anchors installed` 的 `curve=on` 之前不发 RSTeller。出现 `stashed fit file`、`installed` 旧格子，或扫描没有写盘时不发客户端。客户端在 NPU 容器里打 Target。`01` 和 `02` 用 `eval_spectre_rsteller_serving.py`，`03` 和 `04` 用 `eval_spectre_watercaption_serving.py`。每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。`--img-dir` 用数据集根目录。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 到 `raw/04/` 放该次客户端输出。双端日志是同一对进程的全程，放在 `raw/`。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开，重扫拟合 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 03 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 1，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 04 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 同一对进程，四轮之间没有重启。Draft host pid 3974610，Target host pid 3186408。Target 用物理卡 2 和 3。
  - 启动前包内拟合文件已移到 `raw/sync/clear_tree_fit.before.json`，sha256 是 `2398893d50033213e9824561dfcefaeed343842d8fc125ac2243c7857d61a765`，schema 2。Draft 先打了一次 `no fit file`。没有 `stashed fit file`，也没有 `installed ... hold cells from the fit file`。
  - Target 只有 TP0 打了一次 `[CLEAR] prefit started`。`[CLEAR fit_state]` 先是 `querying`，再是 `prefitting`，然后 TP0 和 TP1 各到一次 `ready`，原因 `prefit`。`prefit driver finished state=ready elapsed_s=159.623` 只有一次。`accept_logs.py` 记到的 refill group 是 0。
  - Draft 写了一次拟合文件，`cells=90`。文件 schema 3，有 `h` 90 行和 `layers` 18 行，没有 Q。H 行 `samples_ms` 和 `ells` 都等长，最少 4 条。层行 `n` 最少 15，path 都是 `graph`。写盘后 sha256 是 `21d4bb9c3e17c466afd0f19b6e2d6bfbb55f0094ad6c66d95005b951199b698c`，四轮结束后没变，没有第二次 `wrote fit file`。提交日志是 `hold anchors installed cells=90 curves=18 curve=on`，只有一次。全程 `[CLEAR] fit write` 23602 条，path 全是 `graph`。
  - `accept_logs.py` 把预拟合和四轮正式流量分开。正式 `d_round` 按请求 id 归到四次客户端。四轮 `prefit` 都为空，`hold_invalid` 都为 0。另有 313 条正式轮对不上这四次客户端的请求 id，没有算进下面四轮。
  - 归入各轮的 `d_round` 都带有 `policy_recv_ms`、`policy_plan_ms`、`policy_grow_ms`、`policy_final_k_ms`、`close_ms`、`close_deferred`，以及 `accepted`、`e_pred`、`e_mass`、`e_n`、`tree_depth`。四个 policy 分项之和与 `policy_ms` 的最大绝对差是 0。四轮的 `close_deferred` 都为真，没有同步关账的假值。下面的均值是该轮全部正式 `d_round` 的算术平均，单位毫秒。
  - 归入各轮的正式 `d_layer` 里，回包前样本的 `depth_src` 只有 `cell`、`pooled` 和 `none`。回包后的层没有比值，单独记为 post_reply_unset。`01` 是 cell 33774、pooled 483、none 1119、post_reply_unset 11143。`02` 是 cell 7431、pooled 761、none 818、post_reply_unset 4142。`03` 是 cell 23116、pooled 104、none 889、post_reply_unset 6744。`04` 是 cell 5582、pooled 184、none 349、post_reply_unset 2872。
  - `01` RSTeller 并发 1：100/100，输出 token 45717，正式 wall 843.289 s，吞吐 54.213 tok/s。接受率 34721/89144 = 0.389492。fallback 0。客户端并发 1.000。这一段 `d_round` 11143 条，11043 条同时有 `hold_ms` 和 `q_ms`，100 条排除且 `hold_ms` 为空，排除原因都是 `prefill/finish/new_rid`。均值 `policy_recv_ms` 0.0742，`policy_plan_ms` 0.3216，`policy_grow_ms` 0.7300，`policy_final_k_ms` 0.0246，`policy_ms` 1.1505。按正式轮时间切出的 Draft rank 0 窗口 349 个（348 个 `full`，1 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 11143，`opt_bonus_requests` 11143，`opt_bonus_graph` 11143，eager 0。flush 是 TP0 与 TP1 的 `Cache flushed successfully`，以及 HTTP 200。
  - `02` RSTeller 并发 4：100/100，输出 token 46352，正式 wall 316.176 s，吞吐 146.602 tok/s。接受率 32359/112888 = 0.286646。fallback 0。客户端并发 3.945。`d_round` 3609 条，3419 条同时有 `hold_ms` 和 `q_ms`，190 条排除且 `hold_ms` 为空，排除原因是 `prefill` 95、`finish` 98、`new_rid` 95。均值 `policy_recv_ms` 0.0814，`policy_plan_ms` 0.3506，`policy_grow_ms` 1.5300，`policy_final_k_ms` 0.0278，`policy_ms` 1.9899。窗口 114 个（113 个 `full`，1 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 3638，`opt_bonus_requests` 14044，`opt_bonus_graph` 3638，eager 0。`requests` 大于 `forwards`。flush 同样是两个 TP 和 HTTP 200。
  - `03` WaterCaption 并发 1：100/100，输出 token 29448，正式 wall 658.251 s，吞吐 44.737 tok/s。接受率 22865/53952 = 0.423802。fallback 0。客户端并发 1.000。`d_round` 6744 条，6644 条同时有 `hold_ms` 和 `q_ms`，100 条排除且 `hold_ms` 为空，排除原因都是 `prefill/finish/new_rid`。均值 `policy_recv_ms` 0.0854，`policy_plan_ms` 0.4164，`policy_grow_ms` 0.7608，`policy_final_k_ms` 0.0237，`policy_ms` 1.2862。窗口 212 个（211 个 `full`，1 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 6746，`opt_bonus_requests` 6746，`opt_bonus_graph` 6746，eager 0。flush 同样是两个 TP 和 HTTP 200。
  - `04` WaterCaption 并发 4：100/100，输出 token 29616，正式 wall 310.544 s，吞吐 95.368 tok/s。接受率 21453/66096 = 0.324573。fallback 0。客户端并发 3.985。`d_round` 2101 条，1915 条同时有 `hold_ms` 和 `q_ms`，186 条排除且 `hold_ms` 为空，排除原因是 `prefill` 93、`finish` 96、`new_rid` 93。均值 `policy_recv_ms` 0.1026，`policy_plan_ms` 0.4872，`policy_grow_ms` 1.7555，`policy_final_k_ms` 0.0271，`policy_ms` 2.3723。窗口 66 个（64 个 `full`，2 个 `idle`，末窗 `full`）：`opt_bonus_forwards` 1971，`opt_bonus_requests` 7684，`opt_bonus_graph` 1971，eager 0。`requests` 大于 `forwards`。这一段末窗不是 `idle`，尾数记 missing。flush 同样是两个 TP 和 HTTP 200。
  - Draft 全程已刷窗口 796 个（788 个 `full`，8 个 `idle`，末窗 `idle`）。Target 捕图 batch 1、2、4 都包含 ntpb 1、3、5、9、17、33、65。Draft 捕到 38 张图，batch 1 和 2 含 ntpb 65，batch 4 没有 ntpb 65。
- 总结：移开旧文件后，扫描由 Target rank 0 启动，必需格写满后只写了一次 schema 3，每条 H 的 `ells` 与 `samples_ms` 等长，文件不含 Q。H 曲线在提交时是开启。四轮正式请求都在同一对进程上跑完。可写轮都有 `hold_ms` 和 `q_ms`，`hold_invalid` 为 0。`policy_ms` 的四个分项相加等于 `policy_ms`。归入四轮的 `close_deferred` 都为真。正式层样本有 `cell`、`pooled` 和 `none` 三种深度来源，回包后的层没有比值。接受率按总接受 token 除以总草稿 token。四轮 flush 都是 HTTP 200。清理时身份核对后，只向本场容器 pid 发了 TERM。Draft 进程已退出，本机 GPU 0 没有计算进程，30001 已释放。Target 主进程已退出，30000 已释放，卡 2 和 3 没有计算进程；该 session 还留下 136 个僵尸，不是在跑的服务。两条既有隧道的 PID 和启动时间未变。

