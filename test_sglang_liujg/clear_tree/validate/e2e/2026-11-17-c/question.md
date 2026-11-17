# ClearTree 平表 bonus 开关端到端 A/B

## 问题

两端关闭账本，按实际 batch 使用平表，比较 Draft bonus 预计算开启（A）与关闭（B）的端到端效果。关闭 bonus 后仍应补算最后层 KV，并保持树 KV lease 复用。

## 事前配置与判据

- A：`SGLANG_CLEAR_BONUS_PREDICTION=1`；B：`0`。同一修改后代码，仅切换这一项。
- 两端 `SGLANG_CLEAR_LEDGER=0`，移除锁 K，KV reuse 开启，不启用 lookahead。
- 平表 batch1=W8/K16，其余=W4/K8，τ=0.5，depth8；不做固定 K 扫描或在线策略验证。
- 真实 NPU 32B TP2（物理2/3、page128）+ 3090 CUDA 2B INT8 TP1（page1）。两端 max-running-requests4，图只捕获 batch1/2/4，候选全集，容量 Target65/Draft64。
- 先 A/B 冒烟（4条、max-new-tokens128），再并发1/4各ABAB，每轮文本前100条、temperature0、max-new-tokens1024、inf到达。并发4额外A/A校准。
- 每轮独立重启两端，保留客户端一次预热，预热后显式 flush cache，再进入正式计时；捕图与预热不计吞吐。
- 主要比较成功率、总输出token、墙钟输出吞吐、TTFT、TPOT及延迟分布。噪声门限=max(同期A/A相对差,1%)；不预设关闭必然更快。
- 现有周期日志提供 bonus、零ingest、deferred补层和lease复用旁证；不修改生产代码补埋点，不将旁证视为严格逐轮成功证明。
- 输出按问题顺序配对，差异记录首次可观测分叉；无token ID时不声称token级首分叉。
- 本轮不生成 T_K、draft_floor 或在线K16选择结论。
- 设备/协议/形状错误、持续超时、捕图失败或请求失败时停止该阶段并留证，不更改配置继续冒充同一对照。

## 归档与清理

本目录保存全部记录。容器原始日志与客户端产物同步到 raw/，进度镜像到 progress.log；保存版本、最小diff、包hash、数据hash及产物hash。只终止本次启动进程，保留既有SSH隧道与他人任务，不修改仓库已有启动脚本。
