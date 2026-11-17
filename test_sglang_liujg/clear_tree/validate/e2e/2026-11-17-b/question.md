# 本次问题与事前判据

验证 ClearTree C1–C3 合并改动在 NPU Target / CUDA Draft 的真实端到端行为。只运行当前开账本代码；并发 1、4 各独立冷启动、各运行 text_only 前 100 条。温度 0，max_new_tokens=1024，保留客户端一次预热，两端捕获 batch 1、2、4，max-running-requests=4。K/W 候选均为 2、4、8、16、32、64，容量自动为 Target 65 / Draft 64。

## 验收

- 各完成 100 条；统计请求错误、fallback、协议/形状错误、超时。gain_ratio 完整且 <= 1。
- 未排除且字段完整轮次的时间恒等式绝对误差 <= 0.01 ms。
- C1：policy_ms、post_ms、wait_ms 完整且非负；按实际 b、K/W 报告控制器/后处理/层时间及干净 T_K；检查回放、日志顺序和空闲 flush。只有真实切宽发生才评价运行时切宽边界；否则该部分仅由单测覆盖。日志不能单独证明所有工作均在 send 后。
- C2：报告 draft_floor 比例与 wait_ms 分布；精确 queued 映射结合传输单测判断。wait_ms 不是纯 socket 等待，queued 不要求 wait_ms=0。
- C3：服务 W=4、depth=8 时主动 K 探测 <= 32；服务 W=8 仍允许 K=64。按 KV 桶、平表就绪和切宽分段，区分建树宽度与服务宽度。现有日志没有主动探测标签时，明确直接观测的边界，不改生产代码补埋点。
- 性能只与 2026-11-17-a 同并发作描述性历史对照，报告墙钟 tok/s、请求 decode、接受长度、K/W 分布、分 K T_K 和 replay 实建档 ordinary/explore token/ms 总和。
- Target device_us 按实际 b/K/KV 桶分组检查中位数漂移和离群；遵循 protocol/DECISION.md，但本次没有同期 A/A 或 ABAB，不将历史差异判作可归因的性能提升。
- 图捕获失败、协议/形状拒绝、设备异常或连续超时停止该组、留存证据，不改配置续跑。

运行后归档完整两端日志、客户端 JSONL/CSV/manifest、包 hash、相关 diff 和机器可读汇总；停止仅本次启动的服务，不动现存 SSH 隧道与其他任务。
