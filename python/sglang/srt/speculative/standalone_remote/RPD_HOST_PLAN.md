# SR RPD 主机计划

本实现将归一化输入、RPD 核验和固定容量接受提交贯通。它不改变 Draft、远程协议、
KV 搬移算法或共享 `verify_tree_rpd()` 的签名及返回值。

## 数据与所有权

1. `VerifyInputPacket.load()` 在 `write_draft_regions()` 后构造 `SRRPDHostInput`。
   CPU 只建立 `[4,B,W]` 候选/retrieve/孩子/sibling 和边关系，不生成 attention mask。
   逆序插入和父索引查找遵循设备树构建；None、空数组、截断、padding 已由原输入
   归一化流程决定。主机树独立持有数据，以 packet generation 和实际请求行序绑定。
2. `int64[2,E]` 边下标追加到本轮原输入包尾部，与输入共用一次 H2D。
   未请求 RPD 时包布局不变。NPU mask、position 和树构建仍走原实现。
3. `EagleVerifyInput.verify()` 在 penalty/logit bias 之后调用 `verify_sr_rpd_host()`。
   `torch.max` 的 argmax 使用 int64，边 logit 使用原 FP16/BF16/FP32 dtype。
   NPU 上父节点最大 logit 与候选 logit 由一次 edge-gather 直接写入复用缓冲，
   不在设备上计算 gap；CPU 契约测试仍用原来的 index_select 与高级下标。
   复用按容量增长的设备/主机 pinned 缓冲；两次 D2H 同流提交，末尾一个 event、
   一次等待。dtype、D2H 次数和等待次数不变。无边只读 argmax，空批次没有传输和等待。
4. 输入准备时缓存独立 Python 拓扑；统计一次转列表后由共享 `_rpd_select_host()` /
   `_longest_path()` 返回选路结果。SR 直接生成主机计划，不再生成中间 CPU Tensor；
   旧 `_rpd_compact_select()` 作为适配层保持原 Tensor 返回契约。
   原 logit 转 Python float 后减法、比较与累加顺序不变；边界为
   `gap <= -ln(1-tau)`，tau=0 比较 argmax token；优先最长路径，然后累计 gap
   较小，最后保持原 sibling 次序。token 按原折叠结果构造，重复写位置最后一次生效。
5. `SRFixedAcceptState.finalize_from_host()` 整批校验主机计划，再进入与 greedy
   共用的 CPU 停止判断、路径导出、提交包、KV 搬移和页面释放。
   不调用树 D2H、`_rpd_compact_apply()` 或 greedy 接受结果打包/D2H。
   最终提交包一次 H2D。输出/KV 增量为含 bonus 的 A，对外草稿数为 A-1；接受
   histogram 保留停止截断前口径。结果按原固定接受流程独立持有，不借用下一轮缓冲。

## 分派与失败

沿用 `SGLANG_NPU_SR_FIXED_ACCEPT`，未设置或真值请求启用；设为 0 并重启 Target，
恢复整条旧 RPD 路径，包括旧输入包布局。仅原固定接受支持的 NPU、六维分页
MHA/GQA、topk>1、容量匹配批次可用。RPD 上下文缺失/过期、grammar、logprob、
hidden 消费、自定义处理器、模拟接受长度、非支持 logits 布局或 pinned/event
能力不足时，在优化核验前走旧路径。共享 conservative mode 判定未放宽；CUDA 不变。

输入 generation 或请求行序变化拒绝旧计划；同一上下文、同一计划只消费一次。
缺失上下文的回退原因是 `rpd_context`，请求行序变化是 `rpd_batch_key`，二者都不是共享的 `mode`。
形状、dtype、容量和 kernel 导入能力在设备提交前检查；缺少边 kernel 的 admission
回退原因为 `rpd_edge_kernel`。提交前校验失败不锁工作区、不提交计算或改变请求。
NPU 首次归约提交前保留 logits、上下文、边索引及设备/主机缓冲引用。归约、边取值、
任一次统计 D2H、event 记录或等待失败且完成状态不明时，锁住工作区与输入包；普通
异常包装为 `SRTransferUnresolved` 并保留 cause，已有该类异常和中断原样传播。
不能因为尚未提交 D2H 就允许覆盖仍可能被计算读取的缓冲。下一轮填包、准备或直接
统计均拒绝复用；只有等待成功后释放本轮引用。正常非空路径仍只等待一次。
CPU 上的同步统计失败不锁包。禁止将完成状态不明当作普通 fallback。CPU 校验失败不修改任何请求；停止判断或
提交开始后的异常不重试，不重复追加或提前释放页面。最终提交 H2D 的提交或等待
无法确认完成时，接受状态也锁住提交包；即使换一个新主机计划，也不能覆盖源缓冲
或再次追加。输入包复用等待与上一轮提交包
保护等待仍保留，不能计入“统计一次等待”而隐去。

## 观测

命中日志为 `Speculative RPD verify path: npu_sr_host_plan`。
回退日志为 `[SR] RPD host plan fallback reason=...`；静态禁用沿用原固定接受初始化日志。
启用原 SR round metrics 后可读取：

| 字段 | 含义 |
|---|---|
| `rpd_host_plan_hit` | 核验产出主机计划次数 |
| `rpd_host_finalize`（paths） | 主机计划进入接受后处理次数 |
| `rpd_host_fallback_<reason>`（paths） | 动态回退原因。缺上下文为 `rpd_host_fallback_rpd_context`，行序变化为 `rpd_host_fallback_rpd_batch_key` |
| `rpd_input_edge_bytes` | 输入包附带的边索引字节数 |
| `rpd_host_stats_d2h_bytes` | `8*B*W + 2*E*logit_element_size` |
| `rpd_host_stats_d2h_count` | 非空批次 1 或 2 次 typed D2H |
| `rpd_host_stats_waits` | 非空批次一次统计等待尝试 |
| `rpd_host_edge_gather` | 真实 NPU 边 gather 启动次数；无边为 0，不计入 D2H |
| `rpd_host_workspace_grow` | 容量或 dtype/device 改变引起的工作区分配 |
| `fixed_accept_h2d_count` | 最终提交包上传次数 |

`verify_packet_wait` 和 `fixed_accept_staging_wait` 仍单独记录原复用等待。
主机路径仍可能需要 CPU 停止判断与设备算子调度；字节数减少不等于 TPOT 收益。

## 检查入口

从仓库根目录、已有依赖的 Python 环境执行，沿用仓库 unittest 入口：

```bash
PYTHONPATH=python python3 test/registered/unit/spec/test_sr_rpd.py
PYTHONPATH=python python3 test/registered/unit/spec/test_rpd_verify.py
PYTHONPATH=python python3 test/registered/unit/spec/test_sr_fixed_accept.py
PYTHONPATH=python python3 test/registered/unit/spec/test_standalone_remote.py
PYTHONPATH=python python3 test/registered/unit/spec/test_sr_incremental_commit.py
```

`test_sr_rpd.py` 覆盖 CPU 拓扑/compact 对照、dtype/门限、generation、主机计划
接受对照、整批校验、真实 Eagle verify 方法分派以及模拟 event 故障。
模拟异步仅证明调度与生命周期契约，不证明 NPU 的 pinned/event 实际行为。

NPU 手工入口包含实际树输出逐项对照、三种 dtype、B=1/2/3/4、tau=0/0.2/0.5、
禁止旧往返调用、统计一次等待和主机计划 KV 提交对照；传输断言不使用
已知在部分环境不兼容的 `prof.key_averages()`：

```bash
PYTHONPATH=/mnt/user/liujg/sglang/python python3 test/manual/test_npu_rpd_verify.py
```

## 尚待实机验收

按本次用户要求仅完成本地修改和可运行检查，智能体未运行 NPU/CUDA 服务或远程
测试。用户提供的日志已显示 B=1、Target TP=2、graph、tau=0 主机计划命中，
详细结论和检查结果见 [日志核查记录](RPD_HOST_PLAN_VALIDATION.md)。上述 NPU 手工
测试在本机全部跳过，没有开关对照的性能收益数据。

后续必须显式导入待测源码，核实 `sglang.__file__`。在相同模型、输入、seed 和服务
配置下比较开关 0/1，覆盖 eager/graph、B=1/2/3/4、TP=1/2。每种配置独立预热，
交替至少三组，每组每个开关至少 500 稳态轮次，保留双端日志和请求输出。
先逐项确认输出、接受路径/长度、已提交 KV、释放页面及下一轮 Draft seed，再报告：

| 配置 | 开关 | 组/轮次 | 核验+后处理 | 整轮延迟 | TPOT | 吞吐 | 接受长度 |
|---|---|---|---|---|---|---|---|
| eager/graph, B, TP | 0/1 | >=3 / >=500 | 待测 | 待测 | 待测 | 待测 | 待测 |

同时记录均值及 p50/p95、统计传输字节和等待次数，区分稳态与首次分配/JIT。
现有 `--bench` 只测旧 compact 分段，不能替代整条主机计划的服务对照或据此推算收益。

## 2026-10-04 提交保护本地验收

使用 Windows 临时 Python 3.12 / PyTorch 2.8 CPU 环境，设置 `PYTHONPATH=python`、
`PYTHONUTF8=1`、`PYTHONDONTWRITEBYTECODE=1` 和 `OMP_NUM_THREADS=MKL_NUM_THREADS=1`。
以下每项均执行 `python test/registered/unit/spec/<文件名>`：

| 文件 | 通过 | 跳过 |
|---|---:|---:|
| `test_sr_rpd.py` | 25 | 0 |
| `test_sr_fixed_accept.py` | 35 | 0 |
| `test_sr_kv_copy.py` | 31 | 0 |
| `test_sr_tree_paged_layout.py` | 66 | 0 |
| `test_sr_comm_metrics.py` | 44 | 0 |
| `test_sr_tree_kv_lease.py` | 42 | 0 |
| `test_sr_paged_metadata.py` | 24 | 0 |
| `test_remote_spec_device.py` | 44 | 10 |

另执行 `python test/manual/test_sr_paged_metadata_device.py -v` 和
`python test/manual/test_npu_rpd_verify.py -v`，各 5 项因无 torch_npu/NPU 跳过。
合计 311 通过、20 跳过；remote-spec 跳过项因本地缺少运行时依赖/加速设备。
统计测试中的超时、event 错误日志来自预期故障注入。

六个修改的生产/测试 Python 文件通过内存 `compile()` 语法检查，以及
`python -m ruff check --select E9,F63,F7,F82 <六个文件>`；`git diff --check` 通过。
新增 CPU 测试覆盖 max、edge gather、两次 D2H、record、wait 失败、提交前拒绝、
graph 致命异常穿透和预热双重异常。graph 初始化测试提取实际方法执行，不能作为
完整 SGLang 模块导入或服务启动成功的证据。未连接服务器、未做硬件或性能验收。
