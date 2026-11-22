# 运行

## 事先

- 运行号：2026-11-22-c
- 要回答的问题：移开现有 `clear_tree_fit.json` 后，当前实现是否仍由 Target rank 0 的 FIT 握手驱动扫描，并在必需 H 格和层格各满 4 条后只写一次 schema 2；随后 RSTeller 与 WaterCaption 的并发 1 和并发 4 是否在同一对进程上跑完。正式轮的 `d_round` 是否带上 `hold_ms`、`q_ms`，以及 `policy_recv_ms`、`policy_plan_ms`、`policy_grow_ms`、`policy_final_k_ms`，四项之和是否等于 `policy_ms`。
- 对照的上一次保留运行：2026-11-22-b
- 相对那次运行改了什么：Draft 换成当前 `clear_tree_remote`。同一轮候选 K 的 H+Q 查价在停层和最终选 K 之间复用；最终选出的 K 直接传给记账；`policy_ms` 仍是回包前总账，并按最外层样本拆成收包、计划、停层和最终选 K。拟合文件仍是 schema 2，只在预拟合提交时写一次，不含 Q，也不含在线流量。启动前把包内已有拟合文件移到本场 `raw/sync/`，不删除，迫使这次重扫。不设置 `SGLANG_CLEAR_AUTO_PREFIT=0`，不加 `--api-key`。负载和服务开关不变。Target 用物理卡 2 和 3。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4、捕图 batch 1 2 4。容量是 Target 65、Draft 64。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。显式 `--clear-tree-max-depth 8`。先 Draft 后 Target，四轮之间不重启 Target，也不重启 Draft。Draft 打出 `no fit file`、Target 的 `[CLEAR fit_state]` 为 `ready` 且原因是 `prefit`、拟合文件落盘且 `prefit driver finished state=ready` 之前不发 RSTeller。出现 `stashed fit file`、`installed` 旧格子，或扫描没有写盘时不发客户端。客户端在 NPU 容器里打 Target。`01` 和 `02` 用 `eval_spectre_rsteller_serving.py`，`03` 和 `04` 用 `eval_spectre_watercaption_serving.py`。每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。`--img-dir` 用数据集根目录。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 到 `raw/04/` 放该次客户端输出。双端日志是同一对进程的全程，放在 `raw/`。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开，重扫拟合 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 03 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 1，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 04 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 同一对进程，四轮之间没有重启。Draft host pid 3048910，Target host pid 635726。Target 用物理卡 2 和 3。
  - 启动前包内拟合文件已移到 `raw/sync/clear_tree_fit.before.json`，sha256 是 `2689232b99aea25eeb7032d786dec72f44f7f6d395f613ba071b1b2e0d2f7d1f`。Draft 先打了一次 `no fit file`。没有 `stashed fit file`，也没有 `installed ... hold cells`。这次 Draft 日志里没有 `fit file does not cover`。
  - Target 只有 TP0 打了一次 `[CLEAR] prefit started`。`[CLEAR fit_state]` 到 `ready`，原因 `prefit`。`prefit driver finished state=ready elapsed_s=160.155` 只有一次。`accept_logs.py` 记到的 refill group 是 0。
  - Draft 写了一次拟合文件，`cells=90`。文件 schema 2，有 `h` 90 行和 `layers` 18 行，没有 Q。H 行 `samples_ms` 最少 4。层行 `n` 最少 15，path 都是 `graph`。写盘后 sha256 是 `2398893d50033213e9824561dfcefaeed343842d8fc125ac2243c7857d61a765`，四轮结束后没变，没有第二次 `wrote fit file`。全程 `[CLEAR] fit write` 24841 条，path 全是 `graph`。
  - `accept_logs.py` 把预拟合和四轮正式流量分开。正式 `d_round` 按请求 id 归到四次客户端。四轮 `prefit` 都为空，`hold_invalid` 都为 0。另有 325 条正式轮对不上这四次客户端的请求 id，没有算进下面四轮。
  - 归入各轮的 `d_round` 都带有 `policy_recv_ms`、`policy_plan_ms`、`policy_grow_ms`、`policy_final_k_ms`。四项之和与 `policy_ms` 的最大绝对差是 0。下面的均值是该轮全部正式 `d_round` 的算术平均，单位毫秒。
  - `01` RSTeller 并发 1：100/100，输出 token 46240，正式 wall 857.932 s，吞吐 53.897 tok/s。接受率 34437/95688 = 0.359888。fallback 0。客户端并发 1.000。这一段 `d_round` 11961 条，11861 条同时有 `hold_ms` 和 `q_ms`，100 条排除且 `hold_ms` 为空，排除原因都是 `prefill/finish/new_rid`。均值 `policy_recv_ms` 0.612，`policy_plan_ms` 0.351，`policy_grow_ms` 0.552，`policy_final_k_ms` 0.024，`policy_ms` 1.538。按正式轮时间切出的 Draft rank 0 窗口 377 个，都是 `full`，末窗也是 `full`。bonus：`opt_bonus_forwards` 12064，`opt_bonus_requests` 12064，`opt_bonus_graph` 12064，eager 0。flush 是 TP0 与 TP1 的 `Cache flushed successfully`，以及 HTTP 200。
  - `02` RSTeller 并发 4：100/100，输出 token 45566，正式 wall 310.735 s，吞吐 146.639 tok/s。接受率 31147/115912 = 0.268712。fallback 0。客户端并发 3.920。`d_round` 3735 条，3544 条同时有 `hold_ms` 和 `q_ms`，191 条排除且 `hold_ms` 为空，排除原因是 `prefill` 96、`finish` 99、`new_rid` 96。均值 `policy_recv_ms` 0.487，`policy_plan_ms` 0.358，`policy_grow_ms` 1.188，`policy_final_k_ms` 0.026，`policy_ms` 2.059。窗口 119 个（118 个 `full`，1 个 `idle`，末窗 `full`）：`opt_bonus_forwards` 3779，`opt_bonus_requests` 14437，`opt_bonus_graph` 3779，eager 0。`requests` 大于 `forwards`。flush 同样是两个 TP 和 HTTP 200。
  - `03` WaterCaption 并发 1：100/100，输出 token 29424，正式 wall 655.565 s，吞吐 44.883 tok/s。接受率 22713/54968 = 0.413204。fallback 0。客户端并发 1.000。`d_round` 6871 条，6771 条同时有 `hold_ms` 和 `q_ms`，100 条排除且 `hold_ms` 为空，排除原因都是 `prefill/finish/new_rid`。均值 `policy_recv_ms` 0.720，`policy_plan_ms` 0.432，`policy_grow_ms` 0.618，`policy_final_k_ms` 0.023，`policy_ms` 1.792。窗口 218 个（216 个 `full`，2 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 6911，`opt_bonus_requests` 6911，`opt_bonus_graph` 6911，eager 0。flush 同样是两个 TP 和 HTTP 200。
  - `04` WaterCaption 并发 4：100/100，输出 token 29717，正式 wall 306.156 s，吞吐 97.065 tok/s。接受率 20944/70904 = 0.295385。fallback 0。客户端并发 3.985。`d_round` 2253 条，2072 条同时有 `hold_ms` 和 `q_ms`，181 条排除且 `hold_ms` 为空，排除原因是 `prefill` 91、`finish` 94、`new_rid` 91。均值 `policy_recv_ms` 0.566，`policy_plan_ms` 0.436，`policy_grow_ms` 1.229，`policy_final_k_ms` 0.025，`policy_ms` 2.255。窗口 72 个（70 个 `full`，2 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 2161，`opt_bonus_requests` 8399，`opt_bonus_graph` 2161，eager 0。`requests` 大于 `forwards`。flush 同样是两个 TP 和 HTTP 200。
  - Draft 全程已刷窗口 835 个（829 个 `full`，6 个 `idle`，末窗 `idle`）。Target 捕图 batch 1、2、4 都包含 ntpb 1、3、5、9、17、33、65。Draft 捕到 38 张图，batch 1 和 2 含 ntpb 65，batch 4 没有 ntpb 65。
- 总结：移开旧文件后，扫描由 Target rank 0 启动，必需格写满后只写了一次 schema 2，文件不含 Q。四轮正式请求都在同一对进程上跑完。可写轮都有 `hold_ms` 和 `q_ms`，`hold_invalid` 为 0。`policy_ms` 是总账，四个分项在每一条归入试验的正式轮上都在，并且相加等于 `policy_ms`。接受率按总接受 token 除以总草稿 token。四轮 flush 都是 HTTP 200。清理时身份核对后，只向本场容器 pid 发了 TERM。Draft 进程已退出，本机 GPU 0 没有计算进程，30001 已释放。Target 主进程已退出，30000 已释放，卡 2 和 3 没有计算进程；该 session 还留下 136 个僵尸 python，不是在跑的服务。两条既有隧道的 PID 和启动时间未变。
