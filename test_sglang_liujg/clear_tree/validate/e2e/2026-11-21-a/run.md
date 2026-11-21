# 运行

## 事先

- 运行号：2026-11-21-a
- 要回答的问题：`T_K = H + Q` 和自动预拟合接入之后，RSTeller 并发 1 和并发 4 是否跑完；预拟合是否先结束并只写一次拟合文件；正式轮的 `d_round` 是否带上 `hold_ms` 和 `q_ms`。
- 对照的上一次保留运行：2026-11-20-c
- 相对那次运行改了什么：Draft 与 Target 换成当前含预拟合的 `clear_tree_remote`。负载和服务开关不变。不设置 `SGLANG_CLEAR_AUTO_PREFIT=0`。没有匹配拟合文件时，Target 先自动预拟合，完成日志出现之前不发 RSTeller。Target 用物理卡 2 和 3。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4、捕图 batch 1 2 4。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。先 Draft 后 Target，两轮之间不重启。客户端在 NPU 容器里打 Target，计时用 `eval_spectre_rsteller_serving.py`，每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。`--img-dir` 用数据集根目录。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 和 `raw/02/` 放该次日志和客户端输出。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 同一对进程，两轮之间没有重启。Draft host pid 3038199，Target host pid 1668082。Target 用物理卡 2 和 3。
  - 没有匹配的拟合文件。Target 日志有 `[CLEAR] prefit started` 和 `[CLEAR] prefit driver finished`。Draft 在 06:39:24 UTC 写了一次拟合文件，`prefit finished reason=cells missing=[]`，90 个 H 格、7 条层回归。文件只有 `schema`、`header`、`h`、`layers`，没有 Q。正式客户端在这之后才发出。
  - 01：100/100，45926 token，正式 wall 854.246 秒，53.762 tok/s。接受 34076/95864 = 0.355462。fallback 0。客户端并发 1.000。输出在 `raw/01/`。
  - 02：100/100，46032 token，正式 wall 317.137 秒，145.149 tok/s。接受 31457/117400 = 0.267947。fallback 0。客户端并发 3.931。输出在 `raw/02/`。
  - 两轮 flush 都是 TP0 与 TP1 的 `Cache flushed successfully`，以及 `POST /flush_cache` HTTP 200。
  - Target 捕图 batch 1、2、4 各 7 个形状，含 ntpb=65。Draft 捕到 batch 1、2、4，只缺 bs=4 ntpb=65。
  - 正式阶段去掉 `clearfit-` 的 `d_round` 共 15951 条。其中 15658 条同时有 `hold_ms` 和 `q_ms`，293 条被排除且 `hold_ms` 为空，`hold_invalid` 为 0，`prefit` 为空。
  - Draft rank 0 的 bonus：01 是 377 个满窗加 1 个 idle，末窗 idle，forwards=requests=graph=12081，eager 0。02 是 120 个满窗加 1 个 idle，末窗 idle，forwards=graph=3871，requests=14680，eager 0。预拟合和服务预热的窗口在这两轮之前，没有并进这两轮。
- 总结：两轮都跑完。预拟合先结束并只写一次不含 Q 的拟合文件。正式轮写出了 `hold_ms` 和 `q_ms`。接受率按总接受 token 除以总草稿 token。开跑前 NPU 上的包有 9 个文件和本机不一致，备份后只替换了 `clear_tree_remote`，31 个文件的 SHA256 一致。清理后本场进程已退出，只剩僵尸；物理卡 2 和 3 没有进程，GPU 0 没有计算进程，两条既有隧道的 PID 和启动时间未变。
