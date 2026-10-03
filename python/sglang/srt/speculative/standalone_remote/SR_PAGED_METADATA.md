# SR 分页元数据融合

## 路径与边界

NPU Target 的 `tree_paged_fia` 通过 `sr_paged_metadata.py` 准备固定容量
`int64[Bcap,3]` 参数，随后调用 `sr_paged_metadata_kernels_npu.py` 的页表和
掩码 kernel。Draft 的 `paged_fia` / `paged_atb` 共用 `int64[Bcap,5]`
参数和一个 kernel，直接写最终 attention 页表、reserved branch pages 和 active rows。
CPU/CUDA 保留原通用实现；不新增开关、CLI 或协议字段。

| 正常非空准备 | 参数 H2D | kernel 启动 |
|---|---:|---:|
| Target | 1 | 2 |
| Draft | 1 | 1 |

此表仅统计本模块，不包括已有验证输入包、prefix-tail 参数包、KV 搬运和前向。
允许的 graph 提交前回退会额外准备一次 eager 元数据，单独计数；不会再复制一次 prefix-tail KV。
kernel、上传或完成 event 失败时不回退、不重跑。

Target 的 FULL_MASK 仍由原树构建生成。融合 kernel 读取完整 FULL_MASK（包括 prefix），
输出 `True=masked`；padding 行只保留第 0 列可见，CPU KV 长度为 1。
所有有效/无效页和 mask 列均被覆盖，稳态不先清零或填充。

Draft 继续使用既有 CPU 页数规划函数。attention 只暴露 query branch pages，
prefix-tail 仍可读取 reserved branch pages。例：page=128、prefix=124、steps=5
时 query branch pages=1、reserved branch pages=2；两者不可混淆。
本次不修改 KV 分配、部分页复制、接受算法、CPU attention 长度更新或远程协议。

## 所有权与 graph

- Graph 捕获前分配精确形状的输出和工作区，私有预热完成后再捕获原前向。
  Draft 提前选择 graph/eager，kernel 直接写选定缓冲；replay 绑定不再清零或复制 eager 页表。
- eager 按执行流保留当前形状工作区；形状变化先分配新工作区，旧工作区进入退休列表，
  仅在消费者 event 确认完成后释放。不同流不共用 staging。Graph 工作区跨流使用直接拒绝。
- NPU 新输出通过一维 ND backing 加 view 分配并验证格式。未转换活跃请求映射或 KV。
- 参数上传 event 只保护 pinned staging 的改写。最后一次前向/graph replay 之后记录的
  consumer event 保护输出退休；两者不能互相替代。同流执行顺序保证下一轮覆盖在消费者之后。
- 工作区保留 kernel 输入，未确认读取完成的历史输入通过独立 event 保留。
  generation 随本轮参数上传更新，并绑定 Draft ForwardBatch；已消费或过期的视图不能 replay。
- `SRPagedMetadataSubmittedError` 属于现有 `KVMoveSubmittedError` / `SRTransferUnresolved`
  致命提交后异常类别。发生未确认完成时保留工作区、输入、输出和相关租约，禁止普通回退与回收。
  Draft 原有可同步确认完成的普通事务错误仍遵循原回滚规则。

预热只写私有映射、slots 和输出，保留生产 dtype、stride、Q/K/S 与页容量。
预热成功标记必须在完成同步之后设置。运行时出现尚未预热的 eager 几何仍可能首次编译，
不能把新几何的冷启动时间混入已预热稳态收益。

## 指标

计数前缀为 `paged_metadata_target_` / `paged_metadata_draft_`：

| 后缀 | 口径 |
|---|---|
| `h2d_count`, `h2d_bytes` | 本模块参数包上传次数、字节数 |
| `kernel_calls` | 实际元数据 kernel 提交数；不是模型 graph replay 次数 |
| `staging_waits` | pinned staging 复用时，event 查询未完成而发生的等待 |
| `workspace_grow` | 工作区首次进入受统计的准备路径；同形状稳态不重复计数 |
| `eager_prepares`, `graph_prepares` | 所选执行域；提交失败不能被解读为成功 replay |
| `admission_fallback` | Draft 在写元数据前发现捕获资源不满足条件 |
| `graph_preparation_fallback` | Draft 已准备 graph 元数据，前向提交前转 eager |
| `current_bytes`, `retired_bytes`, `peak_bytes` | 设备参数与输出常驻/待退休字节及其和的峰值 |

主机阶段 `paged_metadata_params_cpu` 记录 pinned 参数填写；
`paged_metadata_staging_wait` 记录复用等待。既有 `tree_paged_view`、`tree_paged_copy`、
`tree_paged_bind` 与 graph 前向阶段独立保留。字节指标不含 pinned 主机内存和输入引用，
减少短算子不等于降低峰值显存。

## 本地检查与硬件验收

在仓库根目录、`PYTHONPATH=python` 下使用既有 unittest 入口：

```bash
python test/registered/unit/spec/test_sr_paged_metadata.py
python test/registered/unit/spec/test_sr_target_tree_fia.py
python test/registered/unit/spec/test_sr_tree_paged_layout.py
python test/registered/unit/spec/test_sr_fixed_accept.py
python test/registered/unit/spec/test_sr_rpd.py
python test/registered/unit/spec/test_sr_comm_metrics.py
python test/manual/test_sr_paged_metadata_device.py
```

CPU 新测试用真实工作区/后端接入方法及 CPU kernel 替身检查包内容、数值契约、
graph/eager 选择、回退、复用和故障注入；**不验证 Triton 编译或 NPU 执行**。
手工测试在无 torch_npu 时跳过；有 NPU 时对真实 kernel 检查 int32/int64、strided 输入、
混合长度、reserved/query 边界、完整输出覆盖、启动数、ND、私有预热隔离及动态 graph。
不使用 `prof.key_averages()`。

整服务验收由实机执行，使用显式 `/mnt/user/liujg/sglang/python` 导入路径，先确认导入文件：

```bash
PYTHONPATH=/mnt/user/liujg/sglang/python python test/manual/test_sr_paged_metadata_device.py -v
```

随后覆盖 B=1/2/3/4（包括 3→4 padding）、TP=1/2、eager/graph、greedy/RPD、
paged FIA/ATB、停止截断、跨页释放及下一轮 seed；逐项比较页表、mask、KV 与 token，
不能只看接受率汇总。补做 CUDA 相邻路径回归。

性能使用独立基线、同模型/请求顺序/缓存策略，预热后新旧版本交替至少三组，每组 500 轮。
分别报告冷启动、元数据准备、整轮延迟、TPOT、端到端吞吐、分配和常驻内存。
此前 0.85–0.96 ms 的旧路径探针只用于定位，不能作为本实现收益。
本地 CPU 通过和代码启动计数均不能代替 NPU 编译、graph、TP 或吞吐验收。

### 本次本地记录（2026-10-03）

Windows 临时验证环境：Python 3.12、PyTorch 2.8 CPU；设置 `PYTHONPATH=python`、
`PYTHONUTF8=1`、`PYTHONDONTWRITEBYTECODE=1`，CPU 线程数限制为 1。
未修改系统依赖或连接服务器。

| 上述单文件命令 | 结果 |
|---|---:|
| `test_sr_paged_metadata.py` | 23 通过 |
| `test_sr_target_tree_fia.py` | 27 通过 |
| `test_sr_tree_paged_layout.py` | 66 通过 |
| `test_sr_fixed_accept.py` | 35 通过 |
| `test_sr_rpd.py` | 20 通过 |
| `test_sr_comm_metrics.py` | 44 通过 |
| `test_sr_paged_metadata_device.py -v` | 5 跳过，缺少 torch_npu/NPU |

另外执行：

```bash
python test/registered/unit/spec/test_standalone_remote.py TestStandaloneRemoteTree TestSRTargetHiddenSkip TestSRTreeSeedSourceGuard TestSRDiscardUnusedVerifyLogits -v
```

此组 34 通过、16 跳过；原因分别为 Windows 无 `resource`、缺少 Triton/Transformers
以及无加速设备。通信统计测试中的超时/异常日志来自故障注入，测试本身通过。
9 个新增/修改 Python 文件完成内存 `compile()` 语法检查；同组文件的
`python -m ruff check --select E9,F63,F7,F82` 和 `git diff --check` 通过。
尚未执行 NPU/CUDA 服务、实际 graph attention、TP=1/2 或性能对照。
