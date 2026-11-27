# 运行

## 事先

- 运行号：`2026-11-27-b`。验证 `786b437c3` 的 Greedy / RPD τ=0.4 两块，不是重复或 ABAB。启动配置沿用 [2026-11-27-a](../2026-11-27-a/run.md)，另加 WaterCaption。只跑下列正式矩阵。
- 两端 `CLEAR_TREE_REMOTE`，初始 W=8，最大深度/steps=8，ClearTree τ=0.5，默认 W/K 候选，最大请求数 4，捕图 batch 1/2/4。CUDA Draft：物理 GPU 0、Qwen3-VL-2B W8A8、TP1、page 1、Triton/Triton MM、静态内存 0.6、容量 64。NPU Target：物理卡 2/3、Qwen3-VL-32B、TP2、page 128、Ascend/Ascend MM、静态内存 0.8、容量 65。账本、bonus、KV reuse 开，lookahead 只在 Draft 开；Draft 的接受比半衰期 1000。RPD 模式和 τ 只传给 Target。
- 双端备份旧拟合到 `raw/sync/`，只同步一次包。每块先启动 Draft 再启动 Target，各只写一次本模式的拟合，Target 确认预拟合 ready 后才发正式请求。每块复用同一对进程，块间仅安全停止并重启本场双端；RPD 块保留新生成的 Greedy 拟合。
- 每数据集取固定顺序的同一批 100 条；温度 0，预算 1024 token，request-rate inf，timeout 600 秒。NPU 容器内用 `tools/run-client.sh` 调对应 serving 客户端。每轮内置预热一条并 flush，预热、flush 与离线 BLEU/ROUGE 不计正式 wall。两端脚本为 `tools/server.sh <role> <mode> <parse|serve>`，设备、仓库、模型与端口由调用时注入。每轮独立目录与客户端产物，双端日志按块留存。

| 试验 | 负载 | 验证 | 并发 | 第几次 | 阶段 |
|---|---|---|---:|---:|---|
| 01 | RSTeller | Greedy | 1 | 1 | 正式 |
| 02 | RSTeller | Greedy | 4 | 1 | 正式 |
| 03 | WaterCaption | Greedy | 1 | 1 | 正式 |
| 04 | WaterCaption | Greedy | 4 | 1 | 正式 |
| 05 | RSTeller | RPD τ=0.4 | 1 | 1 | 正式 |
| 06 | RSTeller | RPD τ=0.4 | 4 | 1 | 正式 |
| 07 | WaterCaption | RPD τ=0.4 | 1 | 1 | 正式 |
| 08 | WaterCaption | RPD τ=0.4 | 4 | 1 | 正式 |

## 结果

**通过：8 轮共 800/800 请求成功，800 条均完成 BLEU/ROUGE，未观测到 fallback；没有额外测试或重跑。**

- `01` Greedy / RSTeller / 并发 1：100/100，45,952 token，780.920 s，58.843 tok/s；接受 35,287 / 86,528（40.78%）；BLEU 0.5613，ROUGE-1/2/L 0.7970 / 0.5950 / 0.6398。
- `02` Greedy / RSTeller / 并发 4：100/100，46,428 token，289.041 s，160.628 tok/s；接受 33,659 / 103,208（32.61%）；BLEU 0.5312，ROUGE-1/2/L 0.7832 / 0.5685 / 0.6180。
- `03` Greedy / WaterCaption / 并发 1：100/100，29,261 token，578.198 s，50.607 tok/s；接受 23,009 / 51,192（44.95%）；BLEU 0.6287，ROUGE-1/2/L 0.8282 / 0.6706 / 0.7171。
- `04` Greedy / WaterCaption / 并发 4：100/100，30,107 token，260.773 s，115.453 tok/s；接受 21,988 / 66,000（33.32%）；BLEU 0.6651，ROUGE-1/2/L 0.8457 / 0.7025 / 0.7460。
- `05` RPD / RSTeller / 并发 1：100/100，43,746 token，680.495 s，64.286 tok/s；接受 34,550 / 74,752（46.22%）；BLEU 0.2892，ROUGE-1/2/L 0.6870 / 0.3602 / 0.4154。
- `06` RPD / RSTeller / 并发 4：100/100，44,034 token，254.843 s，172.788 tok/s；接受 33,103 / 88,512（37.40%）；BLEU 0.3032，ROUGE-1/2/L 0.6929 / 0.3704 / 0.4229。
- `07` RPD / WaterCaption / 并发 1：100/100，28,937 token，530.102 s，54.588 tok/s；接受 23,377 / 45,752（51.10%）；BLEU 0.3582，ROUGE-1/2/L 0.7218 / 0.4337 / 0.4894。
- `08` RPD / WaterCaption / 并发 4：100/100，28,796 token，243.071 s，118.467 tok/s；接受 21,939 / 55,888（39.26%）；BLEU 0.3863，ROUGE-1/2/L 0.7366 / 0.4589 / 0.5124。

吞吐取客户端正式 wall，预热、flush 与评分不计入；接受率用总接受 token 除以总草稿 token。与配对的 Greedy 轮相比，RPD 的四档吞吐分别高 9.25%、7.57%、7.87%、2.61%，接受率也都更高，BLEU 和三项 ROUGE 都更低。这是该矩阵每格一次的观察，未检验重复性或输出等价性。

### 运行证据

- **代码与数据：** Draft 与 NPU 包清单均匹配 `786b437c3`，NPU 只同步一次 ClearTree 包。Draft 的旧 Greedy/RPD 拟合保存在 `raw/sync/draft-old-fit/` 及 SHA256 清单；NPU 的旧拟合全部为 `ABSENT`，同步前整包备份另存。RSTeller 的 JSON 与 `2026-11-27-a` 同哈希；两套数据的前 100 条顺序和图片 SHA256 在 RPD 开跑前复核一致。
- **重新拟合：** Greedy 153.586 s，RPD 149.011 s。两块的 Target 双 TP 都是 `ready`、原因 `prefit`；Draft 每块只写一次 schema 3 文件，均为 90 个 H 格、`lookahead: true`。RPD 握手确认 `tau=0.4`，没有 `rpd_flat`，没有改写 Greedy 文件。拟合文件及 SHA256 都已保留。
- **捕图与实际回放：** 两块的 Target 都捕获了 21 个形状，batch 1/2/4；Draft 各捕 38 个形状，`bs=4, ntpb=65` 超容量未捕获。Target 从各自预拟合完成行后计，Greedy 有 22,950 个 `graph` 验证轮，RPD 有 19,993 个；无 `eager` 或 `fallback`。RPD 的 `rpd_build_us` 中位数 174 µs。Draft rank 0 已刷窗口内的 bonus 前向分别为 24,372 和 21,302，全部命中图、eager 为 0；两块末窗均为 `idle`。这些窗口是整块统计，不拆到各轮。
- **flush 与账本：** 两块分别 4 次 HTTP 200，各 TP 均 4 次 `Cache flushed successfully`。按客户端请求 ID 分轮，41,195 个打分轮的 68,062 条 `e_rid` 可检查；五个新字段全部出现，`gain_last` 只在深度 1 为 `null`，深度/质量、窗口质量加总和 `e_pred` 一致性检查均无异常，`hold_invalid=0`。按深度统计的实际接受、预测质量、样本数见 `summary.json`；`depth == depth_cap` 不按主动停层解释。这是离线诊断，不是在线深度倍率的验证。
- **RPD 掩码：** 用 `(session_id, rpc_seq)` 联结打分轮与晚到 `d_rpd`；`05`–`08` 分别有 100、1、100、1 轮未匹配，匹配记录未见 `rpd_ok=false`。不把未匹配轮当成 0 掩码或视作已验证。Greedy 未出现 `d_rpd`。
- **清理与偏差：** Greedy 捕图时重接了一次本场本地监督器，以纠正预拟合状态 JSON 的解析；确认两端服务 PID 和身份不变，没有重启服务、发送额外请求或丢失日志。两块各只对本场 owned session 成员发 TERM；确认没有存活成员，HTTP 与 Draft RPC 监听释放、所选设备空闲，两条既有隧道身份未变。本场未验证同一 Draft 进程内切换 τ，也未验证在线深度倍率/策略变化。

### 复验入口

- 实际驱动：`python3 -u -B tools/drive.py greedy`（捕图期间以 `greedy --adopt` 接管同一会话）；`python3 -u -B tools/drive.py rpd`。两块均经 `tools/server.sh <role> <mode> parse` 后以 `serve` 启动；每轮由 `tools/run-client.sh` 执行上方固定参数，运行环境依据实验室约定注入。启动解析与拟合放行证据位于 `raw/greedy/`、`raw/rpd/`。
- 离线从仓库根目录运行 `python3 -B .cursor/skills/clear-tree-e2e-validation/scripts/accept_logs.py --raw <场次>/raw/greedy` 和对应 `--raw <场次>/raw/rpd`，再运行 `python3 -B <场次>/tools/review.py --run <场次> --accept .cursor/skills/clear-tree-e2e-validation/scripts/accept_logs.py`。详细数据在 [summary.json](summary.json)，门禁结果在 [checks.json](checks.json)，每轮原始产物在 `raw/01/`–`raw/08/`。
