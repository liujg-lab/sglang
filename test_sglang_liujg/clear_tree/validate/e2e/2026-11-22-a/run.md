# 运行

## 事先

- 运行号：2026-11-22-a
- 要回答的问题：现有拟合文件能安装、且没有新的预拟合扫描时，RSTeller 并发 1 的一轮正式流量里，验证报告是否按冻结的 rpc、b、K 和验证前桶写入；写表日志是否带上这些字段和 path。
- 对照的上一次保留运行：2026-11-21-c
- 相对那次运行改了什么：Draft 与 Target 换成当前未提交的 `clear_tree_remote`。不删除、不移动已有 `clear_tree_fit.json`，不重扫。只跑 RSTeller 并发 1 一轮，不加并发 4、WaterCaption 或重复轮。不设置 `SGLANG_CLEAR_AUTO_PREFIT=0`，不加 `--api-key`。Target 用物理卡 2 和 3。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4、捕图 batch 1 2 4。容量是 Target 65、Draft 64。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。显式 `--clear-tree-max-depth 8`。先 Draft 后 Target。Draft 安装拟合文件、Target 的 `[CLEAR fit_state]` 为 `ready`、且没有 `[CLEAR] prefit started` 之前不发 RSTeller。出现拟合文件缺失、不匹配、不覆盖或开始扫描时不发客户端。客户端在 NPU 容器里打 Target，用 `eval_spectre_rsteller_serving.py`。这一轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。`--img-dir` 用数据集根目录。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 放该次客户端输出。双端日志是同一对进程的全程，放在 `raw/`。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，同一进程，bonus 开，捕图 1 2 4，账本开，沿用已有拟合文件 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/01/`）：
  - 同一对进程。Draft host pid 662445，Target host pid 2428004。Target 用物理卡 2 和 3。
  - Draft 启动时打了 `stashed fit file for the Target header check`。Target 空闲后 `[CLEAR fit_state]` 到 `ready`。Draft 随后打了一次 `installed 90 hold cells from the fit file`。没有 `no fit file`、`does not match`、`does not cover` 或 `prefit started`。没有 `wrote fit file`。拟合文件 sha256 启动前和结束后都是 `12906fe1fbd48fb91496241917ad2feba57311e024822418e8349d3805ce0967`。
  - `accept_logs.py` 打印 `prefit_finished missing`。正式段之前有 1 个 rank 0 窗口，原因是 `idle`，这是启动预热刷出的窗，不是扫描。
  - `01` RSTeller 并发 1：100/100，输出 token 46365，正式 wall 869.703 s，吞吐 53.311 tok/s。接受率 34505/96056 = 0.359218。fallback 0。客户端并发 1.000。这一段 `d_round` 12097 条，11996 条同时有 `hold_ms` 和 `q_ms`，101 条排除且 `hold_ms` 为空，排除原因都是 `prefill/finish/new_rid`，`hold_invalid` 为 0，`prefit` 为空。`[CLEAR] fit write` 11895 条，path 全是 `graph`。Draft rank 0 已刷窗口 379 个（378 个 `full`，1 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 12097，`opt_bonus_requests` 12097，`opt_bonus_graph` 12097，eager 0。flush 是 TP0 与 TP1 的 `Cache flushed successfully`，以及 HTTP 200。
  - Target 捕图 batch 1、2、4 都包含 ntpb 1、3、5、9、17、33、65。Draft 捕到 38 张图，batch 1 和 2 含 ntpb 65，batch 4 没有 ntpb 65。
  - 日志里没有 `refill_groups`，也没有 `graph_unsupported`。
- 总结：已有拟合文件安装成功，没有新的预拟合扫描，文件没有被重写。这一轮正式请求跑完，可写轮有 `hold_ms` 和 `q_ms`，`hold_invalid` 为 0，写表日志带 rpc、b、K、桶，并且 path 都是 `graph`。接受率按总接受 token 除以总草稿 token。补测组 ID 和关闭 graph 的握手这场没有出现。清理时身份核对后，只向本场容器 pid 发了 TERM。Draft 进程已退出，本机 GPU 0 没有计算进程，30001 已释放。Target 主进程已退出，同 session 留下僵尸；30000 已释放，卡 2 和 3 没有计算进程。NPU 0 上原有的 vLLM 没有动。两条既有隧道的 PID 和启动时间未变。
