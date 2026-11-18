---
name: clear-tree-e2e-validation
description: Organizes and validates real ClearTree remote end-to-end experiments across CUDA Draft and Ascend NPU Target, including guarded deployment, owned-process cleanup, graph replay evidence, metrics accounting, retry provenance, and archived verdicts. Use when explicitly requested to run or prepare ClearTree E2E, bonus prediction, concurrency, or graph-capture validation.
disable-model-invocation: true
---

# ClearTree 真实端到端验证

将实验组织为可追溯的「事前问题 → 预检 → 窄同步 → 冒烟 → 正式重复 → 离线验收 → 清理证明」。本 skill 不授权额外实验，不替代当前实现、专项 README 或用户指定范围。

## 先读与定界

以下路径相对仓库根目录；首先确认当前工作区就是预期仓库。

1. 读 `AGENTS.md` 的真实实验环境与同步限制；地址、容器、设备、模型路径和认证从这里核实，不在 skill 中复制口令。
2. 读 `python/sglang/srt/speculative/clear_tree_remote/README.md`、待验证行为的直接控制点及邻近测试。
3. 读 `test_sglang_liujg/clear_tree/validate/README.md`、`validate/e2e/README.md` 及 `test/README.md`。
4. 根据用户要求固定并发、bonus/ledger/lookahead/KV reuse、图 batch、模型、容量、候选档位、负载、轮数及对照。只询问确实不能推断的关键决策。
5. 不将旧脚本的活动配置、注释或历史结果当成当前协议；不顺带扩大为 A/B、并发扫描或在线策略验证。

## 1. 先建档，再开跑

- 新建 `test_sglang_liujg/clear_tree/validate/e2e/<YYYY-MM-DD-后缀>/`，避免覆盖既有场次；参考 `validate/_template/`。
- 开跑前写 `question.md`、`meta.json`：问题、唯一变化因素、精确矩阵、预热/flush 口径、通过线、停止条件、历史对照与结论边界。看到数字后不得回改判据。
- 容器临时工作目录和每个 trial 名均唯一；重试使用新目录，禁止拼接多个 client run 冒充一次。
- `raw/` 放实验脚本、启动命令、双端原始日志、客户端 JSONL/CSV/manifest、最小 diff、包与数据 hash、同步前备份、清理证据。它通常被 git 忽略，交付时说明本地可用性。
- 不创建 commit/branch，不改已有启动脚本，不修改已批准的计划文件。

## 2. 选择编排参考，不盲跑历史脚本

优先阅读已完成场次的脚本，再在新场次中改编。历史 `raw/` 未入库时应明确缺失，不假设其他机器能读到，也不要为取脚本触发旧实验。

可参考本次归档（链接相对本 skill）：
- [事前问题](../../../test_sglang_liujg/clear_tree/validate/e2e/2026-11-18-a/question.md)
- [完成报告](../../../test_sglang_liujg/clear_tree/validate/e2e/2026-11-18-a/verdict.md)
- [编排入口](../../../test_sglang_liujg/clear_tree/validate/e2e/2026-11-18-a/raw/run_experiment.py)
- [预检与同步](../../../test_sglang_liujg/clear_tree/validate/e2e/2026-11-18-a/raw/preflight.py)
- [进程与观测助手](../../../test_sglang_liujg/clear_tree/validate/e2e/2026-11-18-a/raw/runtime.py)
- [离线分析](../../../test_sglang_liujg/clear_tree/validate/e2e/2026-11-18-a/raw/analyze.py)
- [离线收尾](../../../test_sglang_liujg/clear_tree/validate/e2e/2026-11-18-a/raw/finalize.py)

这些是特定场次参考，不是通用 runner。改编时检查路径、容器、矩阵、容量、环境变量、parser、清理契约和 hash；不要照抄该场误报重试逻辑到正常新场次。分析器先做离线自检，真实实验沿用现有客户端和 unittest/pytest 入口。

## 3. 预检与窄同步

- 先执行受影响 CPU 检查并归档结果；CPU/mock 通过不等于端到端通过。
- 确认本机 Docker 与远端 SSH 能访问。沙箱缺 Docker socket 不等于宿主机没 Docker；工具明确因权限受限时申请对应权限重试，不升级全局依赖。
- **同步前、启动前都做空闲保护**：检查双端宿主机/容器的非僵尸服务、HTTP 端口、设备占用和既有 `ssh -N` 身份。观测不完整或发现冲突任务就阻塞，不能擅自停止他人任务。
- 记录隧道 PID、进程启动时间、UID、命令和 boot ID；不新建/不杀已有 `ssh -N`，不运行 `sudo tc`。
- 本地与 NPU 代码是不同副本；先备份远端 ClearTree 包，再只同步本次授权包。按当前 `AGENTS.md` 走 NPU `/tmp` 中转 → `docker cp` → 容器内解包 → 正确属主；禁止直接 rsync 无写权限的目标包目录或整仓覆盖。
- 对两端包文件清单逐文件 SHA256 比对，排除 pycache/pyc；记录双端 HEAD、dirty diff、脚本 hash、模型与数据 hash，并核对事前数据身份。
- 启动前复查包、数据与脚本未变化。实验凭据只通过授权环境或受控来源读取，不写入脚本、日志、快照或归档。

## 4. 运行与可观测进度

- Target/Draft 启动参数同时核实角色、设备、TP、page、注意力后端、静态内存、最大请求数、候选与容量；不要只相信 `config.json`。
- NPU 容器启动前设置正确 `PYTHONPATH`。客户端在能访问 Target HTTP 的 NPU 侧运行，不假设本机存在 HTTP 转发。
- 先冒烟，核对请求完整、配置和运行健康后再正式。设备/协议/形状错误、持续超时、捕图失败或请求失败，停止阶段并留证。
- 每轮按事前约定冷启动、预热、flush，然后正式计时；捕图和预热不计入客户端吞吐。`max-new-tokens` 是预算，不是必须输出的长度。
- 捕图耗时长不能仅凭静默判挂；检查新日志、捕获档位推进和存活状态，在明确的启动/客户端超时内等待。
- 持续镜像容器日志到本场 `progress.log`，清理 ANSI/回车；原始日志仍完整保存。跨主机绝对时间需考虑 UTC/时区与时钟偏差，性能用客户端自身计时。
- **截取日志尾部必须保留行边界**：非零字节偏移后丢弃第一条残行；不能把残缺的 `Ignore import error ...` 后半句当真实错误。
- 错误分类基于完整行及完整日志。仅对明确可选模型导入提示作窄排除，不一概忽略 ImportError/ERROR；计划停机噪声也必须由完成与 owned-stop 证据支持。

## 5. 批量 bonus 与图复用验收

以下针对批量 bonus 实现，其他验证目标须从当前实现制定自己的通过线：

- Draft 显式 `SGLANG_CLEAR_BONUS_PREDICTION=1`，关闭 lookahead 并核对 page1、KV reuse 等准入条件。客户端并发不等于实际每轮 batch；只有同次 flush 中至少两个合格 job 才能命中批量路径。
- 用户指定捕图 `1 2 4` 时，两端设置 `--cuda-graph-bs 1 2 4`，同时从启动参数、manifest 和完整捕图日志互证；不要将 ntpb/K/V 宽度当 batch。实际 batch3 可 padding 至4，不代表额外捕获 batch3。
- 单看「捕获过图」不足以证明实际 replay。检查 Draft rank0 的 `[SR Draft round]`：`opt_bonus_batch_forwards>0`、`opt_bonus_batch_requests>=2`、`opt_bonus_batch_graph>0`、`opt_bonus_batch_failed=0`。
- 事前定义 skip 限额。本次用 `skipped_graph + skipped_kv + skipped_allocator + skipped_map <= forwards`；它是实验通过线，不是算法不变量。
- 同时报 graph/eager/unknown 与失败、各类 skips，图命中分母为 `graph+eager+unknown`。若要求图复用而全是 eager，不得判通过。
- 当前批量 bonus 通过 `is_bonus=False` 复用普通层图；单合格请求和 lookahead 可仍走单请求 bonus 路径。需检查真实 padded batch 的层图 key，而非只看配置宽度。
- `opt_bonus_batch_requests` 是重复发生的 guesses 安装请求级次数，不是独立客户端请求数；`opt_bonus_rounds` 是合格请求尝试，不是 forward 数。
- 周期 counter 是独立刷新窗口，应求和而非当累计快照取最大值；只统计正确角色/rank。host mean 按 metered rounds 加权。
- 明示计数是否含预热、未刷尾部未知、是否与正式计时分离。有显式 metrics 窗口时稀疏缺省 key 才可按已知语义补零；完全没窗口是 missing，不是零失败证据。

## 6. 失败与重试不可洗白

- 保留原 trial、原始 blocked 记录、完整日志和原脚本 hash。不得覆盖失败目录、删除错误或默认选择最新文件。
- 确认为实验工具误报时，只修采样/parser，不改服务/负载；记录原/新脚本 digest、原因、新 retry 身份和不可变原 blocked hash。
- 只有新 retry 的验收与清理都通过后，才另写 resolution，并验证原 blocked、retry completed、阶段清理和两端快照的 hash 链接。
- 原始完整日志含真实运行错误时，不能套用「工具误报」豁免。服务实现、配置或负载变化应新建实验场次。

## 7. 结果口径与交付

- 客户端请求数、成功数、fallback、总输出 token、正式 wall time 必须相互核对。吞吐合并为 `sum(tokens)/sum(wall_s)`，不平均 tok/s。
- 报 TTFT、TPOT、E2E 的均值/分位数与样本口径；TPOT 不等于 ITL，多 token 首 chunk 的语义必须保留。
- 接受率按总 accepted/总 draft 加权，不平均请求接受率。派生值标明公式，不伪称日志原生字段。
- 重复差 `max(rate)/min(rate)-1` 仅是描述性离散，不是置信区间或全场漂移界限。历史早期 A 与晚期 AA 有漂移时分组报告。
- 不同场次、不同输出量/轨迹下，只能描述性能差异；不声称因果加速或无语义变化。保存文本相同、完整 token ID 相同、完整请求与采样相同是不同强度证据。
- 生成 `summary.json`、`checks.json`、`verdict.md`、`meta.json` 和产物 hashes，保留实际 blocked/fail/missing，不把「客户端结束」等同总体验收通过。

## 8. owned-process 清理与最终自检

- 启动时记录 PID、session、starttime、UID/boot 身份；发信号前复核身份，只停本场 owned session。禁止 `pkill -f clear-tree-role`、宽泛杀 Python 或杀 SSH 隧道。
- 清理后分别保存 Target/Draft 非空证据：owned session 无存活成员、HTTP 端口释放、设备/进程观测完整、隧道身份前后一致。僵尸不等于运行服务。
- 冒烟和正式的清理证据分别保留，避免 generic 文件被下一阶段覆盖后破坏重试验收链。
- 离线 finalizer 只验证已有证明，不应自行宣称执行过实时清理；cleanup missing/false 不得变成 true。
- hash 应在结果文件落盘后生成，排除 hash 清单自身和 pycache，并再次逐文件校验。不要把正在增长的 finalizer stdout 重定向进自己正哈希的目录。
- 最终交付：结果目录链接、实际执行命令、每轮请求/吞吐/图命中、失败与重试说明、CPU 检查、清理状态、未验证项。任何必需条件缺失时明确报告阻塞，不伪造完成。

## 调用示例

请求：`/clear-tree-e2e-validation 验证批量 bonus，只测并发4，捕图1 2 4，结果归档到 e2e 新场次。`

按用户范围建立 bonus-on-only 冒烟与正式重复；先读本 skill，再阅读当前仓库行为与环境。示例场次的 4条×128 冒烟、两轮100条×1024 正式负载只在用户要求或上下文明确沿用时使用，不作为所有任务默认矩阵。
