# 2026-09-30 RPD 主机计划日志核查与本地验证

结论：所贴 NPU 日志证明主机计划在 B=1、Target TP=2 的 graph 轮次实际执行，
其传输计数与设计一致。CPU 对照没有发现选路、停止截断或提交结果差异。
故障注入复现了最终提交完成状态不明时的缓冲复用缺口，已在本地修复。
尚不能据这些日志断言完整 NPU 数值等价、重复生成稳定性或性能加速。

## 日志依据与范围

依据用户提供的 `run_edge.sh` / `run_cloude.sh` 两端日志，以及两次
`eval_spectre_rsteller.py --port 30000 --batch-size 1 --max-items 2` 输出。
日志中的实际参数优先于本地启动脚本：Qwen3-VL-2B-Instruct，STANDALONE_REMOTE，
Target TP=2、Draft TP=1，topk=3、steps=5、W=15，page_size=128，rpd tau=0。
两端都从实验项目的 `python/sglang` 路径导入；日志不含文件 hash，不能逐行核实远端
与本地最后修订完全相同。本次修复未传输或部署到远端。

两个 Target rank 都报告 fixed accept requested/effective=True 和
`Speculative RPD verify path: npu_sr_host_plan`。以下汇总分别针对每个 rank，
不能将两个 TP rank 相加成两倍逻辑请求数：

| 观测项 | 每个 rank 的累计值 | 每轮 |
|---|---:|---:|
| 已刷出的 round 窗口 | 8 × 32 = 256 | — |
| `rpd_host_plan_hit` / `rpd_host_finalize` | 256 / 256 | 1 / 1 |
| 输入上传 `verify_packet_upload` | 256 | 1 |
| 边索引 `rpd_input_edge_bytes` | 57,344 B | 224 B |
| 统计 D2H 次数 | 512 | 2 个不同 dtype 拷贝 |
| 统计 D2H 字节 | 59,392 B | 232 B |
| 统计等待 `rpd_host_stats_waits` | 256 | 1 |
| 最终提交上传 `fixed_accept_h2d_count` | 256 | 1 |
| 统计工作区扩容 | 1 | 首次分配；随后窗口无扩容 |
| `fixed_accept_multimodal_hit` | 256 | 1 |
| graph 验证批次 | 256 | 1 |

224 B = 2 × 14 × 8，与 E=14 的边索引相符。
232 B = 15 × 8 + 2 × 14 × 4，按代码公式推断该统计的 logit 元素为 4 B。
两个 rank 的接受计数和路径计数一致，没有记录动态 RPD 回退、核验/提交异常或
graph eager fallback。启动时缺少其他可选模型依赖的 import 提示不能当作本轮 Qwen
核验失败。日志计数配合代码和调用禁用单测证明传输结构；它不是完整硬件 memcpy trace。

## 性能和重复性

去掉第一个 32 轮窗口后的七个窗口平均如下；这只是剔除首次窗口的描述统计，
并非满足正式性能验收的完整预热实验。

| 主机阶段 | TP0 | TP1 |
|---|---:|---:|
| 整轮 | 71.228 ms | 71.251 ms |
| RPC 等待 | 59.143 ms | 59.240 ms |
| `run_batch_including_verify` | 11.872 ms | 11.851 ms |
| 树构造（含 CPU 拓扑准备和输入上传） | 1.023 ms | 1.025 ms |
| `verify_forward` | 5.418 ms | 4.989 ms |
| `accept_commit_including_wait` | 4.605 ms | 5.010 ms |

RPC 等待约占整轮 83%。通信日志的稳态 Draft residence 约 58.3 ms，non-Draft
elapsed 约 0.28 ms，当前主要时间在 Draft 处理与等待。Draft 的 `tree_d2h_wait`
约 31 ms 包含等待前面排队的设备工作；不能把它解读成小 payload 的纯传输成本。
Target 后处理墙钟也包含前向尚未完成的等待，不能全部归因于 RPD CPU 选路。

两次评估报告 76.588 / 77.944 token/s，但没有固定接受开关 0 的同条件对照。
本地脚本 `summarize()` 的 `decode_tok_s` 实际为 completion_tokens / 整体客户端
wall，包含 prefill 和其他开销，不是独立测得的稳态 TPOT。不能据两次值差异宣称加速。

第二条样本的输出由 234 token / 42 verify 变为 254 token / 45 verify；默认
temperature=0 且 rpd tau=0，重复性仍需排查。日志中 deterministic inference=False，
但这不足以将差异直接归因为 NPU 数值波动。没有关闭优化的 token-id 对照、完整两次
JSONL 或首个分歧处 logits，无法判断是原路径也存在的差异还是本次路径回归。

`alen` 为 completion_tokens / verify，包含 bonus；accept_rate 使用截断前
`spec_accepted_tokens / (verify × 14)`。两者不必在最后一次停止截断后严格满足
`rate=(alen-1)/14`。BLEU≈0.0024–0.0025、ROUGE-L≈0.09 是这两条描述的评分，
缺少相同输入的旧路径/普通 Target baseline，不能定位为核验代码错误或证明质量等价。

## 本次代码检查和修复

`sr_rpd.py` 沿用原 `_rpd_compact_select()` / `_longest_path()` 的浮点计算和 tie
顺序。归一化拓扑快照独立持有数据，边下标随输入包上传；主机计划在 penalty 和
logit bias 后生成。Eagle 明确区分旧路径、greedy 固定接受和 RPD 主机计划；RPD
不绑定 greedy 结果缓冲。批次行序、generation、容量、索引、token 和长度先校验，
再进入共享 CPU 停止处理与 KV 提交。统计 event 失败会锁住统计工作区和输入包。

新增故障测试在修复前失败：注入 `submit_copy()` 的 SRTransferUnresolved，首次
计划已经追加 token、没有释放页面，但另一份新计划仍能再次追加并复用提交源。
原因是 event 缺失被当作已完成，仅检查 plan.consumed 不足以保护整个状态。

修复在 `SRFixedAcceptState` 增加 `_commit_unresolved`：最终 H2D 提交或等待无法
确认完成时设置该状态；后续 `_wait_previous_h2d()` 在任何 token 追加前拒绝继续。
保留提交缓冲，不重跑核验，不提前释放页面。同计划和不同新计划都不能重复追加。
这是异常路径修复，所贴 NPU 日志没有触发该故障；修复尚未做 NPU 故障注入。

同时补充了日志参数 K=3/S=5/W=15、B=1/2/3/4 的 CPU 拓扑/旧 compact 对照，
batch 重排与容量变化，以及直接执行生产 Req 停止方法的 EOS、stop token、stop
string、regex、max_new_tokens、ignore_eos、reasoning 和截断前统计对照。
其他主要覆盖包括开关关闭/缺上下文/过期上下文及消费者回退、整批校验、typed
D2H 一次等待、旧树回读/中间结果写回/greedy 打包禁用和提交失败。

## 已执行检查

环境为独立临时 venv：Windows、Python 3.12.14、PyTorch 2.8.0+cpu；未修改全局
环境。与实机 Python 3.11/NPU kernel 不同，CPU 测试不能替代实机数值验收。

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$srPython="$env:TEMP/codex-sr-rpd-venv/Scripts/python.exe"
& $srPython test/registered/unit/spec/test_sr_rpd.py
& $srPython test/registered/unit/spec/test_sr_fixed_accept.py
& $srPython test/registered/unit/spec/test_rpd_verify.py
& $srPython test/registered/unit/spec/test_standalone_remote.py
& $srPython test/registered/unit/spec/test_standalone_remote.py TestStandaloneRemoteTree -v
& $srPython test/registered/unit/spec/test_sr_incremental_commit.py
& $srPython test/registered/unit/spec/test_sr_incremental_commit.py TestSRIncrementalCommit -v
& $srPython test/manual/test_npu_rpd_verify.py
```

| 检查 | 结果 |
|---|---|
| SR RPD | 18 / 18 通过 |
| 固定接受（含 greedy） | 29 / 29 通过 |
| 共享 RPD | 34 通过，5 依赖缺失跳过 |
| standalone remote 全文件 | 95 通过，63 跳过，2 错误：Windows ZMQ 不支持 IPC |
| 其中输入包/树相关 TestStandaloneRemoteTree | 12 通过，16 环境/硬件条件跳过 |
| incremental commit 全文件 | 14 通过，4 错误：Draft 导入依赖 Linux resource |
| 其中协议/增量提交 TestSRIncrementalCommit | 13 / 13 通过 |
| NPU 手工入口 | 4 全部跳过，未配置 Torch NPU |

核心三文件合计 81 通过、5 跳过。较窄重跑与全文件结果有重叠，不能相加计算唯一
通过项。没有用 resource stub 或伪造 NPU 依赖让不可运行的测试显示通过。
语法、Ruff F401/F821、isort 和 diff whitespace 检查通过；Black 检查新模块/新测试，
已有大文件仅格式化本次修改区域。

## 剩余验收

需要固定输入、tokenizer、服务参数与 seed，保存完整输出 token IDs，比较开关 0/1
并与普通 Target greedy 对照。出现首个不同 token 时保存对应 logits、树、接受路径
和 KV/下一轮 Draft seed。覆盖 eager/graph、B=1/2/3/4、TP=1/2，独立预热后交替
至少三组、每组至少 500 稳态轮次，再报告核验+后处理、整轮、TPOT、吞吐和接受长度。
这一步未在本次本地检查中执行。
