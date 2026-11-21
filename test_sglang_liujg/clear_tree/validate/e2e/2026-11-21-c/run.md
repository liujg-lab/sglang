# 运行

## 事先

- 运行号：2026-11-21-c
- 要回答的问题：schema 2 拟合是否在必需 H 格和层格都有 4 条样本后写一次文件，Target 空闲时是否用 FIT 握手驱动扫描；随后 RSTeller 与 WaterCaption 的并发 1 和并发 4 是否在同一对进程上跑完，正式轮的 `d_round` 是否带上 `hold_ms` 和 `q_ms`。
- 对照的上一次保留运行：2026-11-21-b
- 相对那次运行改了什么：Draft 与 Target 换成当前未提交的 `clear_tree_remote`。拟合文件改为 schema 2，必需格各有 4 条样本才写，内容是这次预拟合的 H 和层回归，不含 Q，也不含在线流量。Target 空闲时用 FIT 询问 Draft；提交、取消和状态走 `ClearAction.CONTROL`；`tp>1` 时只有 rank 0 启动扫描。Q 按 `(b, K)` 保存最近 32 个原始毫秒。旧拟合文件因此不能复用。不设置 `SGLANG_CLEAR_AUTO_PREFIT=0`，不加 `--api-key`。负载和服务开关不变。Target 用物理卡 2 和 3。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4、捕图 batch 1 2 4。容量是 Target 65、Draft 64。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。显式 `--clear-tree-max-depth 8`。先 Draft 后 Target，四轮之间不重启。`[CLEAR fit_state]` 为 `ready`、拟合文件落盘且 `prefit driver finished state=ready` 之前不发 RSTeller。客户端在 NPU 容器里打 Target。`01` 和 `02` 用 `eval_spectre_rsteller_serving.py`，`03` 和 `04` 用 `eval_spectre_watercaption_serving.py`。每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。`--img-dir` 用数据集根目录。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 到 `raw/04/` 放该次客户端输出。双端日志是同一对进程的全程，放在 `raw/`。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 03 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 1，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 04 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 同一对进程，四轮之间没有重启。Draft host pid 291654，Target host pid 1397008。Target 用物理卡 2 和 3。
  - 启动时包内没有 `clear_tree_fit.json`。Draft 先打 `no fit file`。扫描期间打了 118 次 `fit file does not cover this Target; prefit required`。没有 `已有拟合文件，跳过预拟合`。
  - Target 只有 TP0 打了一次 `[CLEAR] prefit started`。TP1 没有启动扫描。`[CLEAR fit_state]` 到 `ready`，原因 `prefit`。`prefit driver finished state=ready elapsed_s=162.939` 只有一次。
  - Draft 写了一次拟合文件，`cells=90`。文件 schema 2，有 `h` 90 行和 `layers` 18 行，没有 Q。H 行样本数最少 4。层行 `n` 最少 15，path 都是 `graph`。`[CLEAR] fit write` 24367 条，path 全是 `graph`。没有 `refill_groups`。四轮结束后文件 sha256 仍是 `12906fe1fbd48fb91496241917ad2feba57311e024822418e8349d3805ce0967`，没有第二次 `wrote fit file`。
  - `accept_logs.py` 打印 `prefit_finished missing`。Draft 日志按 20 秒间隔分成五段。第一段是扫描，1456 条 `d_round`，其中 1455 条带 `prefit`。后四段对应四轮正式流量。
  - `01` RSTeller 并发 1：100/100，输出 token 45545，正式 wall 908.402 s，吞吐 50.137 tok/s。接受率 34108/92536 = 0.368592。fallback 0。客户端并发 1.000。这一段 `d_round` 11663 条，其中 1 条仍带 `prefit`；11562 条同时有 `hold_ms` 和 `q_ms`，101 条排除且 `hold_ms` 为空，排除原因都是 `prefill/finish/new_rid`，`hold_invalid` 为 0。Draft rank 0 已刷窗口 365 个（364 个 `full`，1 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 11663，`opt_bonus_requests` 11663，`opt_bonus_graph` 11663，eager 0。flush 在仍有 1 个 running request 时返回 HTTP 400，日志是 `Cache not flushed because there are pending requests`。客户端打印了 flush 失败，随后仍进入正式计时。
  - `02` RSTeller 并发 4：100/100，输出 token 45861，正式 wall 319.389 s，吞吐 143.590 tok/s。接受率 31344/116680 = 0.268634。fallback 0。客户端并发 3.940。`d_round` 3839 条，3647 条同时有 `hold_ms` 和 `q_ms`，192 条排除且 `hold_ms` 为空，`hold_invalid` 为 0，`prefit` 为空。已刷窗口 120 个（119 个 `full`，1 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 3839，`opt_bonus_requests` 14597，`opt_bonus_graph` 3839，eager 0。`requests` 大于 `forwards`。flush 是 TP0 与 TP1 的 `Cache flushed successfully`，以及 HTTP 200。
  - `03` WaterCaption 并发 1：100/100，输出 token 29589，正式 wall 699.004 s，吞吐 42.330 tok/s。接受率 22978/54120 = 0.424575。fallback 0。客户端并发 1.000。`d_round` 6827 条，6726 条同时有 `hold_ms` 和 `q_ms`，101 条排除且 `hold_ms` 为空，`hold_invalid` 为 0，`prefit` 为空。已刷窗口 214 个（213 个 `full`，1 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 6787，`opt_bonus_requests` 6787，`opt_bonus_graph` 6787，eager 0。flush 同样是两个 TP 和 HTTP 200。
  - `04` WaterCaption 并发 4：100/100，输出 token 29937，正式 wall 314.527 s，吞吐 95.181 tok/s。接受率 21066/71664 = 0.293955。fallback 0。客户端并发 3.975。`d_round` 2346 条，2156 条同时有 `hold_ms` 和 `q_ms`，190 条排除且 `hold_ms` 为空，`hold_invalid` 为 0，`prefit` 为空。已刷窗口 74 个（72 个 `full`，2 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 2241，`opt_bonus_requests` 8509，`opt_bonus_graph` 2241，eager 0。`requests` 大于 `forwards`。flush 同样是两个 TP 和 HTTP 200。
  - Target 捕图 batch 1、2、4 都包含 ntpb 1、3、5、9、17、33、65。Draft 捕到 38 张图，batch 1 和 2 含 ntpb 65，batch 4 没有 ntpb 65。
- 总结：schema 2 拟合在必需格有样本后写了一次，只含这次预拟合的 H 和层回归。扫描只由 Target rank 0 启动。四轮正式请求都在同一对进程上跑完，可写轮都有 `hold_ms` 和 `q_ms`，`hold_invalid` 为 0。接受率按总接受 token 除以总草稿 token。`01` 的 flush 因当时还有 1 个请求而返回 400，正式计时仍然完成。写盘失败、`api_key`、已写盘 run 的取消、`graph_unsupported`、`SGLANG_CLEAR_AUTO_PREFIT=0` 和省略深度时的默认 10 没有注入。清理时身份核对后，只向本场容器 pid 发了 TERM。Draft 进程已退出，本机 GPU 0 没有计算进程，30001 已释放。Target 主进程已退出，同 session 留下僵尸；30000 已释放。停完之后卡 2 和 3 上出现了别人的 vLLM，端口 37751，启动时间是 23:49，没有对它发信号。NPU 0 上原有的 vLLM 没有动。两条既有隧道的 PID 和启动时间未变。
