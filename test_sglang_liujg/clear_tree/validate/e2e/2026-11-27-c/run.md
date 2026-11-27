# ClearTree RSTeller 双模式验证

## 事先

- 运行号 `2026-11-27-c`；被测提交 `4b5de83245a134624fdd88acd610c7175ad06f64`。沿用 [2026-11-27-b](../2026-11-27-b/run.md) 的 RSTeller 条件，两模式分块对照，不是重复或 ABAB。只跑以下四轮。
- 两端 `CLEAR_TREE_REMOTE`，初始 W=8，最大深度/steps=8，ClearTree τ=0.5，默认 W/K 候选 `2 4 8 16 32 64`，最大请求数 4，捕图 batch 1/2/4。CUDA Draft：RTX 3090 物理 GPU 0，Qwen3-VL-2B W8A8 INT8，TP1、page 1、Triton/Triton MM、静态内存 0.6、容量 64。NPU Target：910c 物理卡 2/3，Qwen3-VL-32B，TP2、page 128、Ascend/Ascend MM、静态内存 0.8、容量 65。两端启用账本、bonus、KV reuse；lookahead 仅 Draft 开，接受比半衰期 1000。验证方式与 RPD τ 只传 Target。
- 两端旧拟合按角色移入本场 `raw/sync/` 并保存哈希。两块独立启动、重新拟合 schema 4；每块先等 Draft HTTP 就绪再启动 Target，双端健康且两 TP 报 `ready/prefit` 后才发正式请求。两轮共用一对进程，块间安全停止并重启双端。RPD 块不改写本场 Greedy 拟合。
- RSTeller 固定同序 100 条，温度 0，生成预算 1024 token，`request-rate=inf`，请求超时 600 秒，保留 BLEU 与 ROUGE-1/2/L。每轮客户端内置预热 1 条，随后 flush；预热、flush 和离线评分不计正式 wall。每轮独立保存客户端日志和产物，双端服务日志按模式保存。

| 轮次 | 负载 | 验证方式 | 并发 | 第几次 | 阶段 |
|---|---|---|---:|---:|---|
| 01 | RSTeller | Greedy | 1 | 1 | 正式 |
| 02 | RSTeller | Greedy | 4 | 1 | 正式 |
| 03 | RSTeller | RPD τ=0.4 | 1 | 1 | 正式 |
| 04 | RSTeller | RPD τ=0.4 | 4 | 1 | 正式 |

## 结果

本场本地端到端验证到此结束。Greedy 的 `01`、`02` 完成；RPD 的 `03`、`04` 没有正式请求，不补跑。

- `01` Greedy / RSTeller / 并发 1：100/100，45,692 token，778.545 s，58.689 tok/s；接受 35,047 / 86,384（40.57%）；BLEU 0.5600，ROUGE-1/2/L 0.7996 / 0.5962 / 0.6419。
- `02` Greedy / RSTeller / 并发 4：100/100，45,887 token，285.381 s，160.792 tok/s；接受 33,199 / 102,440（32.41%）；BLEU 0.5371，ROUGE-1/2/L 0.7874 / 0.5735 / 0.6238。

Greedy 重新拟合为 schema 4，指纹 `7a093a1e`，90 个 H 格和 18 个必需层格均达到样本数要求。Target 捕获 21 个形状，Draft 捕获 38 个；Draft 的 `(bs=4, ntpb=65)` 因容量未捕获。两轮均无 fallback，两次 flush 都是 HTTP 200，两个 TP rank 均确认。正式轮 `hold_invalid=0`。离线复核确认第 1 层有日志且未进入拟合，第 2 层及以后有拟合样本。Draft rank 0 的 bonus 窗口末尾为 `idle`，图命中 16,076，eager 为 0。

RPD 在 Target 初始化期间失去通往跳板的连接，没有写入本模式拟合，也没有 `03`、`04` 的客户端产物。结束时本机 Draft 已停止，30001 无监听，GPU 0 无计算进程。跳板机的 22 端口仍不可达，NPU Target 进程 `1299141` 未能复核或停止。
