# ClearTree 批量 bonus：并发 4 端到端验证

## 事前问题与范围

- 运行号：2026-11-18-a。
- 问题：修改后的批量 bonus 是否在真实 NPU Target + CUDA Draft、并发 4、捕图 batch 1/2/4 下实际命中层 CUDA graph，稳定完成请求？
- 唯一实现变化：ClearTree 多请求 bonus 从逐请求 forward 改为合并 forward。不新增 A/B 开关，不修改其他算法或 cuda_graph_runner.py。
- 历史描述性对照：2026-11-17-c 的 c4-A1/A2（原串行 bonus 开启）；c4-AA1/AA2 单列，不能合并掩盖该场时间漂移。
- 不测试 bonus 关闭，不测试并发 1，不据跨场对照声称因果性能提升或输出等价。

## 固定配置

两端 ledger=0、tree KV reuse=1、lookahead 关闭、移除 lock K / probe / dump。平表实际 batch1=W8/K16，其他=W4/K8；tau=0.5，depth8；候选全集，容量 Target65/Draft64。两端 max-running-requests=4，cuda-graph-bs=1 2 4。

Target：实验室 910c，物理卡 2/3，32B TP2，ascend/ascend_attn，page128，静态内存0.8。
Draft：3090，2B INT8 TP1，triton/triton_attn，page1，静态内存0.6，SGLANG_CLEAR_BONUS_PREDICTION=1。沿用既有 localhost:30019 链路，不新建或关闭 SSH 隧道。

## 矩阵

1. smoke-A：并发4，前4条，max-new-tokens128，bonus开。
2. c4-A1：并发4，前100条，max-new-tokens1024，bonus开。
3. c4-A2：同上，独立冷启动两端。

客户端 text_only_ledger.py，temperature0、request-rate inf；每轮首条预热后 flush cache，预热/捕图不计入正式吞吐。文本数据预期 SHA256=8cf47bd9104c062c0ae2729f582044c97b407847a092fc2f3fa343280b4eec22，由 preflight 实测核对。

## 事前通过线

- 运行：冒烟4/4、正式200/200成功、fallback=0，无运行期设备/协议/超时/Traceback/Capture failed/OOM。
- 配置：启动脚本、config、manifest 与捕图日志互证 bonus开启、ledger关闭、并发4、max-running-requests4、图batch仅1/2/4。
- 每个正式轮批量路径：opt_bonus_batch_forwards>0，opt_bonus_batch_requests>=2，opt_bonus_batch_graph>0，opt_bonus_batch_failed=0；无 bonus batch 异常。graph/KV/allocator/map skips 总和 <= forwards，作为“不主导”的固定界限。
- 未出现指标窗口不得当作零失败证明；周期窗口求和，包括预热且不保证覆盖未刷新尾部；不与正式请求分母混算。
- 冒烟先满足运行完整和配置一致，再开始正式；若发生设备、协议、形状错误、持续超时或捕图失败，停止该阶段、归档失败，不调整配置继续充当同一试验。
- 吞吐为描述性：sum(output tokens)/sum(formal wall time)，报告 TTFT/TPOT/E2E 分布、接受/verify旁证、两轮相对差；历史早期A与晚期AA分别对照，跨场漂移不作噪声已控制的保证。
- 重复输出差异仅记录，不设 A/B 等价硬门；不声称与 Target-only 或改前 token 等价。

## 归档与清理

本目录保存 meta/question/verdict/summary/checks/artifact hashes/progress；raw 保存脚本、日志、客户端记录、代码diff、包/数据hash、同步前备份及清理证据。只同步 ClearTree 包到 NPU（/tmp 中转、docker cp、chown1008），不覆盖整仓。只停止本场拥有且 PID/starttime 身份匹配的进程，保留所有既有隧道。不创建提交、不改现有启动脚本、不改计划文件。
