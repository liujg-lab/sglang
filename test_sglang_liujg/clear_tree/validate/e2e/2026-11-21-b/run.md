# 运行

## 事先

- 运行号：2026-11-21-b
- 要回答的问题：预拟合收尾修正之后，扫描是否只在 `clearfit-done` 时写一次拟合文件，而不是在 90 个 H 格攒满时结束；随后 RSTeller 并发 1 和并发 4 是否跑完。
- 对照的上一次保留运行：2026-11-21-a
- 相对那次运行改了什么：Draft 与 Target 换成当前收尾修正后的 `clear_tree_remote`。H 格攒满只记日志，拟合文件只在 `clearfit-done` 时写。Target 头增加 `disable_cuda_graph`、`attention_backend`、`page_size`、`cuda_graph_bs`，Draft 头增加 `attention_backend`、`disable_cuda_graph`。旧拟合文件因此不能复用。不设置 `SGLANG_CLEAR_AUTO_PREFIT=0`，不加 `--api-key`。负载和服务开关不变。Target 用物理卡 2 和 3。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4、捕图 batch 1 2 4。容量是 Target 65、Draft 64。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。先 Draft 后 Target，两轮之间不重启。`clearfit-done` 和拟合文件落盘之前不发 RSTeller。客户端在 NPU 容器里打 Target，计时用 `eval_spectre_rsteller_serving.py`，每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。`--img-dir` 用数据集根目录。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 和 `raw/02/` 放该次日志和客户端输出。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 03 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 1，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 04 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |

`03` 和 `04` 是 RSTeller 已经开跑之后加上的。客户端用 `eval_spectre_watercaption_serving.py`，数据在 NPU 的 `/mnt/user/liujg/Downloads/test_WaterCaption`。四轮仍是同一对进程，不重启 Target。

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 同一对进程，四轮之间没有重启。Draft host pid 3387013，Target host pid 2636967。Target 用物理卡 2 和 3。
  - 旧拟合文件和当前 Draft 头不一致，日志是 `fit file does not match this Draft; prefit required`，随后 `fit file does not match this Target; prefit open` 和 `prefit started`。没有在扫描前出现 `已有拟合文件，跳过预拟合`。
  - `[CLEAR] cells complete` 只有一次。没有 `prefit finished reason=cells`。Target 在这之后仍跑完宽度组，`prefit driver finished groups=109 elapsed_s=164.612`。Draft 在 `08:29:44` 写一次文件，`prefit finished reason=clearfit-done missing=[]`。写完之后 Target 才打 `已有拟合文件，跳过预拟合`。
  - 新文件有 90 行 H、7 条层回归，没有 Q。Target 头含 `disable_cuda_graph`、`attention_backend`、`page_size`、`cuda_graph_bs`，Draft 头含 `attention_backend`、`disable_cuda_graph`。预拟合里的 `clearfit-w` 层样本中，宽度 16、32、64 全部是 `post_reply` 且 `wall_us` 为空，所以没有进拟合；宽度 2、4、8 有 `fit=true` 的样本。层记录代码没有改。
  - `01` RSTeller 并发 1：100/100，输出 token 45598，正式 wall 850.301 s，吞吐 53.626 tok/s。接受率 33805/95368 = 0.354469。fallback 0。客户端并发 1.000。Draft rank 0 已刷窗口 376 个（375 个 `full`，1 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 12025，`opt_bonus_requests` 12025，`opt_bonus_graph` 12025，`opt_bonus_eager` 0。正式 `d_round` 12025 条，其中 11924 条同时有 `hold_ms` 和 `q_ms`，101 条被排除且 `hold_ms` 为空，`hold_invalid` 为 0。
  - `02` RSTeller 并发 4：100/100，输出 token 45534，正式 wall 315.067 s，吞吐 144.522 tok/s。接受率 31195/115320 = 0.270508。fallback 0。客户端并发 3.917。已刷窗口 120 个（119 个 `full`，1 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 3825，`opt_bonus_requests` 14422，`opt_bonus_graph` 3825，eager 0。`requests` 大于 `forwards`。正式 `d_round` 3825 条，3635 条同时有 `hold_ms` 和 `q_ms`，190 条被排除且 `hold_ms` 为空，`hold_invalid` 为 0，`prefit` 为空。
  - `03` WaterCaption 并发 1：100/100，输出 token 29704，正式 wall 666.059 s，吞吐 44.597 tok/s。接受率 23125/53952 = 0.428622。fallback 0。客户端并发 1.000。已刷窗口 213 个（212 个 `full`，1 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 6782，`opt_bonus_requests` 6782，`opt_bonus_graph` 6782，eager 0。正式 `d_round` 6796 条，6695 条同时有 `hold_ms` 和 `q_ms`，101 条被排除且 `hold_ms` 为空，`hold_invalid` 为 0，`prefit` 为空。
  - `04` WaterCaption 并发 4：100/100，输出 token 30321，正式 wall 310.439 s，吞吐 97.671 tok/s。接受率 21283/72776 = 0.292445。fallback 0。客户端并发 3.990。已刷窗口 75 个（72 个 `full`，3 个 `idle`，末窗 `idle`）：`opt_bonus_forwards` 2271，`opt_bonus_requests` 8682，`opt_bonus_graph` 2271，eager 0。`requests` 大于 `forwards`。正式 `d_round` 2366 条，2176 条同时有 `hold_ms` 和 `q_ms`，190 条被排除且 `hold_ms` 为空，`hold_invalid` 为 0，`prefit` 为空。
  - 四轮 flush 都是 TP0 与 TP1 的 `Cache flushed successfully`，以及 `POST /flush_cache` HTTP 200。客户端没有打印 flush 失败。
  - Target 捕图 batch 1、2、4 都包含 ntpb 1、3、5、9、17、33、65。Draft 捕到 38 张图，batch 1 和 2 含 ntpb 65，batch 4 没有 ntpb 65。
- 总结：预拟合没有在 90 格时结束，只在 `clearfit-done` 写了一次带新头部的拟合文件。RSTeller 和 WaterCaption 的并发 1、并发 4 都在同一对进程上跑完。接受率按总接受 token 除以总草稿 token。`api_key` 鉴权、写盘失败后的关闭、重试组重新预热没有在本场注入。清理后 Draft 进程已退出，GPU 0 没有计算进程，本机 30001 和 30019 已释放。Target 主进程是僵尸，30000 已释放，物理卡 2 和 3 没有进程。NPU 0 上原有的 vLLM 没有动。两条既有隧道的 PID 和启动时间未变。
