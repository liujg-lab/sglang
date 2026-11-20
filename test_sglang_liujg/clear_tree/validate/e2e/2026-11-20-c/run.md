# 运行

## 事先

- 运行号：2026-11-20-c
- 要回答的问题：超集轮才记加宽和缩窄、满 15 两边都成立继续探测、满 30 两边都成立则清空样本但超集比例不变之后，RSTeller 并发 1 和并发 4 是否跑完，`d_wpair` 怎么判定，吞吐和接受率是多少。
- 对照的上一次保留运行：2026-11-20-a
- 相对那次运行改了什么：只换 Draft 与 Target 的 `clear_tree_remote`。负载和服务开关不变。Target 用物理卡 2 和 3。2026-11-20-b 没有正式请求，这场不并入那次输出。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、lookahead 关、bonus 开、max-running-requests 4、捕图 batch 1 2 4。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。先 Draft 后 Target，两轮之间不重启。客户端在 NPU 容器里打 Target，计时用 `eval_spectre_rsteller_serving.py`，每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。

试验从 `01` 编号。`tools/` 放这场的启动脚本，`raw/01/` 和 `raw/02/` 放该次日志和客户端输出。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<试验名>/`）：
  - 同一对进程，两轮之间没有重启。Target 用物理卡 2 和 3。正式客户端第一次把图片目录指到了图片文件夹，json 里的相对路径又带了同一段，预热在发出请求前失败。改成数据集根目录后重跑 01，服务没有重启。
  - 01：100/100，45610 token，正式 wall 846.355 秒，53.890 tok/s。接受 34398/90800 = 0.378833。fallback 0。客户端并发 1.000。输出在 `raw/01/`。
  - 02：100/100，45942 token，正式 wall 320.511 秒，143.340 tok/s。接受 31842/113456 = 0.280655。fallback 0。客户端并发 3.929。输出在 `raw/02/`。
  - 两轮 flush 都是 TP0 与 TP1 的 `Cache flushed successfully`，以及 HTTP 200。
  - Target 捕图 21 行，batch 1、2、4 各 7 个形状，含 ntpb=65。Draft 捕到 batch 1、2、4，只缺 bs=4 ntpb=65。
  - 日志里没有字面量 `superset` 或 `probe_w`。`d_wpair` 共 28 条，没有缩窄。并发 1 时 b=1 在 15 个样本、只有加宽成立时从 8 切到 16。并发 4 时 b=4 在 30 个样本、只有加宽成立时从 4 切到 8。其后高于 5% 的停留都把 p 减半，所以这些停留不是「两边都成立且比例不变」。落到 5% 之后，两边都成立和两边都不成立都会留在地板上，日志分不出这两种。
  - Draft rank 0 的 bonus：01 是 357 个满窗加 1 个 idle，末窗 idle，forwards=requests=graph=11444，eager 0。02 是 116 个满窗加 1 个 idle，末窗 idle，forwards=graph=3734，requests=14183，eager 0。服务自带预热另有 1 个 idle 窗，没有并进这两轮。
- 总结：两轮都跑完。接受率按总接受 token 除以总草稿 token。在线宽度有两次加宽，没有缩窄。高于地板的停留都在减半探测比例。清理后本场进程已退出，物理卡 2 和 3 没有进程，两条既有隧道的 PID 和启动时间未变。
