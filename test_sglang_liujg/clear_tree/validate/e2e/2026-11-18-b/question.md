# ClearTree bonus 专用图：并发1与4端到端验证

## 事前问题和范围

运行号：2026-11-18-b。验证当前工作区 ClearTree bonus 专用图重构、验证 padding 本地剥离、统一事务入口及默认开启修改，在真实 NPU Target + CUDA Draft、bonus-on 下能否实际图回放并完成请求。唯一受测实现是当前 ClearTree 包；客户端并发为预先指定的两个负载档，不做 bonus-off、Target-only、旧实现A/B、独立冒烟或重复轮。旧场结果仅作来源导航，不构成同期性能对照。

## 固定矩阵

按顺序 c1-A1、c4-A1，客户端并发分别1、4，各仅一轮前100条有效文本、max-new-tokens1024、temperature0、request-rate inf、不忽略EOS。两档分别冷启动双端；现有客户端各用首条样本预热一次，预热后严格确认flush成功，才开始100条正式计时。预热不是独立冒烟，不计正式吞吐。1024是预算，不强制输出长度。

两端 max-running-requests4、cuda-graph-bs 1 2 3 4；W/K候选均2 4 8 16 32 64，容量Target65/Draft64，握手width8、tau0.5、depth8。ledger0、tree KV reuse1、lookahead关闭、移除lock-K/probe/dump；Draft bonus显式1、target backend=npu。默认开关本身另由CPU测试覆盖。平台、容器、设备、模型以AGENTS.md真实实验环境为准：Target NPU910c物理2/3、32B TP2 page128 ascend/ascend_attn static0.8；Draft本地3090、2B INT8 TP1 page1 triton/triton_attn static0.6。仅沿用现有RPC隧道。

数据事前预期SHA256：8cf47bd9104c062c0ae2729f582044c97b407847a092fc2f3fa343280b4eec22。启动前核对包、脚本、客户端、数据身份。每端启动超时2400秒、单请求600秒、每档客户端阶段5400秒。

## 事前通过线（结果出现后不修改）

- 两档分别100/100正式请求成功、fallback0，无运行期设备、协议、形状、超时、Traceback、OOM或捕图错误。配置与包数据身份一致；flush有明确成功证据。
- 启动命令、实际参数、manifest、完整捕图日志互证batch1/2/3/4。Draft使用bonus专用r{G}_{bs}_s0图，不再以普通层图作为bonus证明。默认单stream、256 token容量预计51张Draft图，容量不足的r65_4_s0不要求捕获；以实际容量核实。
- 每档Draft rank0有有效窗口：opt_bonus_forwards>0、graph>0、failed0、unknown0。报告graph/eager/unknown及graph/(graph+eager+unknown)；有eager则全图执行验收不通过且保留原因。
- c4在无失败时要求opt_bonus_requests>opt_bonus_forwards，证明至少有多请求组；否则批量覆盖不足。现有指标没有逐batch×G矩阵，捕图不等于实际回放，不能宣称2/3/4各自命中。
- opt_lease_publish_failed、opt_bonus_skipped_map、opt_bonus_skipped_allocator均0；opt_bonus_skipped_kv<=opt_bonus_forwards。这是实验阈值，不是算法不变量。empty/unready分别报告。hits和ingest_zero_fwd为0则不宣称命中复用已验证。
- Target verify_graph_batches>0，RPC/fallback/circuit-breaker无隐性降级。
- 周期counter按独立窗口求和，只统计正确角色rank；含预热，不与正式请求分母混算。缺整个窗口为missing，不是零失败。尾窗未确认则保留未知。
- 正式token、wall、成功数与JSONL/CSV/manifest互证，吞吐=tokens/wall。报告TTFT/TPOT/E2E均值和分位数；TPOT不等于ITL，接受率总accepted/总draft。

## 停止、清理和边界

同步前及每次启动前确认双端空闲；冲突或观测不完整即阻塞，不停止他人任务。仅备份并窄同步ClearTree包，经/tmp中转及容器docker cp/chown，实验助手客户端放独立/tmp。仅停止本场PID/session/starttime/UID/boot匹配的owned进程，不动现有ssh -N，不执行tc。每档保留双端无存活owned成员、端口释放、设备观测与隧道身份不变的非空证明。

真实运行错误即停止场次并保存原始blocked/fail，不擅自重试或增补负载。CPU/mock通过不等于E2E通过。无关闭对照、无重复、无Target-only，不能得出因果加速、重复稳定或token/logits等价；K32/K64回放、拆组、lookahead、异常回收和逐batch回放不视为完成覆盖。

归档question/meta/summary/checks/verdict/hashes和完整raw产物；raw通常不入库，需说明本机可用。原始失败、missing和清理缺项不可被finalizer改写为通过。不创建commit/branch、不改已有服务实现或启动脚本，不修改批准计划。
