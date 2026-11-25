# 运行

## 事先

- 运行号：2026-11-25-a
- 要回答的问题：同一对进程先跑 RSTeller、再跑 WaterCaption 时，Draft 池化接受比按路径质量遗忘（半衰期 1000）和不遗忘（半衰期 0）的正式吞吐、总接受率，以及 `e_pred` 相对 `accepted` 差多少。每块内并发 1 和并发 4 都要跑完。两块都安装现有拟合，不重扫。
- 对照的上一次保留运行：2026-11-24-b
- 相对那次运行改了什么：服务开关、负载、设备和客户端沿用 24-b。Draft 在清掉 `SGLANG_CLEAR_*` 之后仍导出账本、KV reuse、bonus、`SGLANG_CLEAR_TARGET_BACKEND=npu`、`SGLANG_CLEAR_DRAFT_LOOKAHEAD=1`，并显式导出 `SGLANG_CLEAR_ACCEPT_HALF_LIFE`。第一块是 `1000`，第二块是 `0`。Target 仍导出 lookahead `0`，不导出半衰期。两块都不设置 `SGLANG_CLEAR_ACCEPT_PER_REQUEST`、`SGLANG_CLEAR_WPAIR_METRIC`、`SGLANG_CLEAR_WPAIR_RULE`、`SGLANG_CLEAR_DEFER_CLOSE`、`SGLANG_CLEAR_LAYER_DEPTH`、`SGLANG_CLEAR_H_CURVE`、`SGLANG_CLEAR_LAYER_SLOPE_PRIOR` 或 `SGLANG_CLEAR_AUTO_PREFIT=0`，不加 `--api-key`。不移走包内 `clear_tree_fit.json`。出现 `no fit file`、`wrote fit file` 或 `prefit started` 时不发客户端。两块都要看到安装 hold cells，Target 的 `[CLEAR fit_state]` 为 `ready` 且原因不是 `prefit`。第二块启动前再核拟合 sha256，文件被改写就停。Target 用物理卡 2 和 3。100 条、最多 1024 token、温度 0、request-rate inf、ledger 开、KV reuse 开、bonus 开、max-running-requests 4、捕图 batch 1 2 4。容量是 Target 65、Draft 64。省略 `--clear-tree-widths` 和 `--clear-tree-ks`。显式 `--clear-tree-max-depth 8`。先 Draft 后 Target。`01` 到 `04` 同一对进程，然后重启，`05` 到 `08` 是另一对进程。客户端在 NPU 容器里打 Target。`01`、`02`、`05`、`06` 用 `eval_spectre_rsteller_serving.py`，其余用 `eval_spectre_watercaption_serving.py`。每轮内部预热一条并 flush，不计吞吐。不计算 BLEU/ROUGE。`--img-dir` 用数据集根目录。数据仍是 24-b 在 NPU 上用过的那一份。

试验从 `01` 编号。`tools/` 放这场的启动脚本。`raw/on/01/` 到 `raw/on/04/` 放半衰期 1000 的客户端输出，该块双端日志放在 `raw/on/`。`raw/off/05/` 到 `raw/off/08/` 放半衰期 0，日志放在 `raw/off/`。开跑前填完，看到数字后不改。

| 试验名 | 负载 | 条件 | 第几次 | 阶段 |
|---|---|---|---|---|
| 01 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，安装现有拟合，半衰期 1000 | 1 | 正式 |
| 02 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 1000 | 1 | 正式 |
| 03 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 1，同一进程，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 1000 | 1 | 正式 |
| 04 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 1000 | 1 | 正式 |
| 05 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 1，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，安装同一拟合，半衰期 0，新进程 | 1 | 正式 |
| 06 | RSTeller 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 0 | 1 | 正式 |
| 07 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 1，同一进程，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 0 | 1 | 正式 |
| 08 | WaterCaption 100 条，最多 1024 token，温度 0 | 并发 4，同一进程，bonus 开，捕图 1 2 4，账本开，Draft lookahead 开，半衰期 0 | 1 | 正式 |

## 结果

- 关键数字（指向 `raw/<块>/<试验名>/`）：
  - 第一次预检时 NPU 的 30000 和 phy-id 2、3 被 host pid 236662 占用，证据留在 `raw/sync/precheck.txt`。该进程退出后继续。本机和 NPU 的包清单不一致，备份后只同步了 ClearTree 包，同步后清单一致。拟合 sha256 开跑前、两块之间和结束后都是 `34e56789f083d9f6b75010e4f3f14948f4b2b5cac2db6aab327fdf0ce29f3dcb`。两块 Draft 都是 `stashed fit file` 一次、`installed 90 hold cells` 一次、`hold anchors installed cells=90 curves=18 curve=on` 一次。没有 `no fit file`、`wrote fit file` 或 `prefit started`。`accept_logs.py` 的 `prefit_status` 都是 `reused`。Target 两个 TP 的 `[CLEAR fit_state]` 都是 `ready`，原因 `ready`。
  - 半衰期 1000 的 Draft host pid 2862830，环境里 `SGLANG_CLEAR_ACCEPT_HALF_LIFE=1000`，没有 `SGLANG_CLEAR_ACCEPT_PER_REQUEST`。Target host pid 246322，物理卡 2 和 3。四轮之间没有重启。
  - 半衰期 0 的 Draft host pid 3009304，环境里 `SGLANG_CLEAR_ACCEPT_HALF_LIFE=0`，没有 `SGLANG_CLEAR_ACCEPT_PER_REQUEST`。Target host pid 657993，物理卡 2 和 3。四轮之间没有重启。两块之间重启了 Draft 和 Target。
  - 接受率是总 `spec_accept_token_num / spec_draft_token_num`。`e_pred / accepted` 来自正式 `d_round` 里同时有 `accepted`、`e_pred`、`e_mass` 的轮，按客户端请求 id 归到该轮。预拟合没有写入。
  - `01` RSTeller 并发 1，半衰期 1000：100/100，输出 token 45727，正式 wall 774.135 s，吞吐 59.069 tok/s。接受率 35082/86400 = 0.406042。fallback 0。客户端并发 1.000。`d_round` 10800 条，10700 条同时有 `hold_ms` 和 `q_ms`，100 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 10700，`accepted` 34568，`e_pred` 34560.461366，`e_mass` 36907.507471，`e_pred/accepted` 0.999782。前半 0.933197/0.933635，后半 0.939639/0.939608，斜线两边是 `e_pred/e_mass` 和 `accepted/e_mass`。flush 是两个 TP 和 HTTP 200。
  - `02` RSTeller 并发 4，半衰期 1000：100/100，输出 token 45776，正式 wall 285.511 s，吞吐 160.330 tok/s。接受率 33149/102264 = 0.324151。fallback 0。客户端并发 3.932。`d_round` 3282 条，3093 条同时有 `hold_ms` 和 `q_ms`，189 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 3281，`accepted` 32692，`e_pred` 32614.569152，`e_mass` 34682.563537，`e_pred/accepted` 0.997632。前半 0.940528/0.941325，后半 0.940210/0.943965。flush 是两个 TP 和 HTTP 200。
  - `03` WaterCaption 并发 1，半衰期 1000：100/100，输出 token 29721，正式 wall 583.853 s，吞吐 50.905 tok/s。接受率 23389/51840 = 0.451177。fallback 0。客户端并发 1.000。`d_round` 6480 条，6380 条同时有 `hold_ms` 和 `q_ms`，100 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 6380，`accepted` 22872，`e_pred` 22922.093813，`e_mass` 23347.541977，`e_pred/accepted` 1.002190。前半 0.979317/0.974596，后半 0.984282/0.984758。flush 是两个 TP 和 HTTP 200。
  - `04` WaterCaption 并发 4，半衰期 1000：100/100，输出 token 29849，正式 wall 264.030 s，吞吐 113.052 tok/s。接受率 21924/64400 = 0.340435。fallback 0。客户端并发 3.971。`d_round` 2059 条，1870 条同时有 `hold_ms` 和 `q_ms`，189 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 2059，`accepted` 21479，`e_pred` 21477.860271，`e_mass` 22114.142283，`e_pred/accepted` 0.999947。前半 0.973484/0.969530，后半 0.968885/0.973095。flush 是两个 TP 和 HTTP 200。
  - `05` RSTeller 并发 1，半衰期 0：100/100，输出 token 45849，正式 wall 770.238 s，吞吐 59.526 tok/s。接受率 35210/86304 = 0.407976。fallback 0。客户端并发 1.000。`d_round` 10788 条，10688 条同时有 `hold_ms` 和 `q_ms`，100 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 10688，`accepted` 34686，`e_pred` 34590.456107，`e_mass` 36986.542267，`e_pred/accepted` 0.997245。前半 0.933777/0.941847，后半 0.936665/0.933733。flush 是两个 TP 和 HTTP 200。
  - `06` RSTeller 并发 4，半衰期 0：100/100，输出 token 46274，正式 wall 287.595 s，吞吐 160.900 tok/s。接受率 33205/105488 = 0.314775。fallback 0。客户端并发 3.953。`d_round` 3364 条，3171 条同时有 `hold_ms` 和 `q_ms`，193 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 3363，`accepted` 32762，`e_pred` 32679.171276，`e_mass` 34909.794908，`e_pred/accepted` 0.997472。前半 0.936436/0.930806，后半 0.935751/0.946584。flush 是两个 TP 和 HTTP 200。
  - `07` WaterCaption 并发 1，半衰期 0：100/100，输出 token 29635，正式 wall 586.208 s，吞吐 50.554 tok/s。接受率 23165/53032 = 0.436812。fallback 0。客户端并发 1.000。`d_round` 6629 条，6529 条同时有 `hold_ms` 和 `q_ms`，100 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 6529，`accepted` 22657，`e_pred` 21825.727087，`e_mass` 23135.702538，`e_pred/accepted` 0.963311。前半 0.940785/0.968357，后半 0.946065/0.990656。flush 是两个 TP 和 HTTP 200。
  - `08` WaterCaption 并发 4，半衰期 0：100/100，输出 token 29756，正式 wall 265.917 s，吞吐 111.900 tok/s。接受率 21701/65416 = 0.331738。fallback 0。客户端并发 3.974。`d_round` 2089 条，1903 条同时有 `hold_ms` 和 `q_ms`，186 条排除且 `hold_ms` 为空，`hold_invalid` 0。打分轮 2089，`accepted` 21250，`e_pred` 20900.929468，`e_mass` 21990.572823，`e_pred/accepted` 0.983573。前半 0.949871/0.962605，后半 0.951062/0.970258。flush 是 HTTP 400。Target 日志是 `Cache not flushed because there are pending requests`，`#running-req: 1`，两个 TP 都没有 `Cache flushed successfully`。客户端打了 `flush_cache failed: HTTP 400`，随后仍进入正式计时。
  - 半衰期 1000 另有 322 条正式轮对不上这四次客户端，半衰期 0 另有 312 条，没有算进上面八轮。八轮 `hold_invalid` 都是 0。
  - 两块 Target 都捕到 batch 1、2、4，且各含 ntpb 1、3、5、9、17、33、65，`Capturing` 行在 TP0。两块 Draft 都捕到 38 张图，batch 1 和 2 含 ntpb 65，batch 4 没有 ntpb 65。半衰期 1000 的 rank 0 已刷窗口 720 个，715 个 `full`、5 个 `idle`，末窗 `idle`。半衰期 0 是 728 个，723 个 `full`、5 个 `idle`，末窗 `idle`。两块尾数都已刷出。
- 总结：八轮都在各自的一对进程上跑完。同一进程先跑 RSTeller 再跑 WaterCaption 时，半衰期 1000 的 WaterCaption 正式轮 `e_pred/accepted` 是 1.002190 和 0.999947。半衰期 0 的对应两轮是 0.963311 和 0.983573，预测低于结算接受数。RSTeller 上两臂的这个比值都在 0.997 到 1.000 之间。吞吐和总接受率两臂接近：WaterCaption 并发 1 的吞吐是 50.905 对 50.554 tok/s，接受率是 0.451177 对 0.436812。`08` 的 flush 没有成功。清理时身份核对后，只向本场容器 pid 发了 TERM。Draft 两块的 session 都没有存活成员，本机 GPU 0 没有计算进程，30001 已释放。Target 两块的主进程已退出，30000 已释放，phy-id 2 和 3 没有计算进程。半衰期 0 的 Target session 还剩 136 个僵尸，不是在跑的服务。两条既有隧道的 PID、UID、启动 ticks 和 boot ID 未变。
