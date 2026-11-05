
## Project Overview

这个项目是在SGlang项目的基础上增加`SPECTRE`、`STANDALONE_REMOTE`、`OPT_TREE_REMOTE`和空壳`TOLL_TREE_REMOTE`类型投机采样模式。主要特点是草稿模型和目标模型可以运行在不同的服务器上面。同时增加 RPD 柔性核验模式。同时为这些模式支持 CUDA 和 NPU 架构。

 # SGLang 实验分支智能体指南

 ## 项目定位

 本仓库基于 SGLang，主要维护以下跨进程投机采样模式：

 - `SPECTRE`：Target 负责验证，Draft 负责生成草稿；通过 ZMQ 传递控制信息和 token。
 - `STANDALONE_REMOTE`：Target 与 Draft 可运行在不同服务器上，使用独立的远程投机协议。
 - `OPT_TREE_REMOTE`：原版 OPT-Tree。宽度 N 和阈值 τ 固定，不含 AdaCost。默认 greedy；RPD 仅 Target 可选，Draft 拒绝。Target 验证，Draft 按层调用 `update`。两端可在不同机器上。算法和启动命令以 [python/sglang/srt/speculative/opt_tree_remote/README.md](python/sglang/srt/speculative/opt_tree_remote/README.md) 为准。
 - `TOLL_TREE_REMOTE`：已注册为空壳，服务循环拒绝启动。算法以 [python/sglang/srt/speculative/toll_tree_remote/README.md](python/sglang/srt/speculative/toll_tree_remote/README.md) 为准。
 - `RPD`：Target 侧的柔性核验模式，通过 `--speculative-verify-mode rpd` 和
	 `--speculative-rpd-tau` 配置。`OPT_TREE_REMOTE` 默认仍是 greedy。该模式只允许 Target，Draft 启动时拒绝。

 这些功能同时维护 CUDA 和 Ascend NPU 路径。文档或代码中写的“支持”表示当前分支的设计目标或已有实现；实际可用性仍需在对应硬件、依赖、模型权重和网络条件下验证。

 ## 代码导航

 优先从行为的直接控制点开始阅读，不要只在启动脚本或转发层反复搜索：

 - 核心包：[python/sglang](python/sglang)
 - 共享服务参数：[python/sglang/srt/server_args.py](python/sglang/srt/server_args.py)
 - 共享调度接入：[python/sglang/srt/managers/scheduler.py](python/sglang/srt/managers/scheduler.py)
 - 投机算法和共享类型：[python/sglang/srt/speculative/spec_info.py](python/sglang/srt/speculative/spec_info.py)
 - SPECTRE：`python/sglang/srt/speculative/spectre/`
	 - 协议和通信：`spectre_protocol.py`、`spectre_communication.py`、`cpp_zmq/`
	 - Target 验证：`verifier/`
	 - Draft 调度和生成：`drafter/`
 - STANDALONE_REMOTE：`python/sglang/srt/speculative/standalone_remote/`
	 - 远程协议和传输：`sr_protocol.py`、`sr_transport.py`
	 - Target 验证：`verifier/`
	 - Draft 生成：`drafter/`
	 - 对齐、回滚和 RPD 辅助逻辑：同目录下的 `sr_align.py`、`sr_kv_rollbacker.py`、`sr_verify_layout.py` 等
 - OPT_TREE_REMOTE：`python/sglang/srt/speculative/opt_tree_remote/`
	 - `is_opt_tree_remote()` 与 `is_standalone_remote()` 分开。OPT 不把后者设为真。
	 - 复用 STANDALONE_REMOTE 的 ZMQ 传输，不改 SR 消息格式。线上 `spec_shape` 的第三项是 `opt_tree_width`，不是验证前向的 N+1。
	 - `--opt-tree-width` 是 N。Target 的 `--speculative-num-draft-tokens` 是 N+1（验证 token 数，含根）；Draft 的是 N（每层前向宽度）。手写值必须等于本角色期望，否则启动报错。`--speculative-num-steps` 等于 `--opt-tree-max-depth`。`draft_token` 是一维，长度为 `bs * (N+1)`，这样 `prepare_for_verify` 的 `len(input_ids)` 会分配同样多的 KV 槽。父指针 `-1` 表示已提交尾 token。
	 - Target 使用 `OptTreeRemoteWorker`，Draft 仍是 `TpModelWorker`。事件循环是独立的 `is_opt_tree_remote()` 分支。
	 - CUDA `--page-size` 为 1，NPU `--page-size` 为 128。NPU 紧内存时走 KV 预算回退并跳过 graph；空闲内存不足 252 MiB 时，树核验不走 compact FIA，改用分块注意力。级联注意力只在 CUDA page size 1 的 Triton Draft 上启用。

 修改协议、命令行参数或调度流程时，必须同时检查 Target/Draft 两端、CUDA/NPU 分支、fallback/circuit-breaker 路径以及邻近测试。先定位直接计算或状态变更的代码，再决定是否需要修改公共抽象。算法相关改动留在对应目录，见下方「修改纪律」。

 ## 环境边界

 ### 编写环境

 - Python：3.11
 - 当前编写环境不能完整启动 SGLang 服务，也不保证存在 GPU、NPU、模型权重、跨机网络或全部运行时依赖。
 - 不要把本文件中的版本、路径、IP 或端口当作通用默认值。

 ### 当前实验环境参考值

 以下版本用于记录已知实验环境，不是所有机器的硬性要求：

 - CUDA：PyTorch 2.9.1、Transformers 5.3.0、Triton 3.5.1
 - NPU：`sgl_kernel_npu` 2026.5.1、SGLang 0.5.6、`sglang-router` 0.3.2、PyTorch 2.8.0+cpu、`torch_npu` 2.8.0.post2、Transformers 5.3.0、`triton-ascend` 3.2.0

 依赖定义以 [python/pyproject.toml](python/pyproject.toml) 及对应平台配置为准。不要为了验证而擅自升级/降级全局依赖，或覆盖用户已有环境。

 ## 安装与构建

 完整的 SPECTRE 安装、C++17/ZMQ 扩展构建和启动说明以 [README_SPECTRE.md](README_SPECTRE.md) 为准。

 - 仓库根目录不是 Python 包；editable 安装从 `python` 目录进行：

	 ```bash
	 python -m pip install -e "python"
	 ```

 - SPECTRE 不只是 Python 代码。两端都需要构建 `python/sglang/srt/speculative/spectre/cpp_zmq/` 下的扩展，并确认可以导入 `DealerEndpoint`、`RouterEndpoint`。
 - 构建脚本可能执行 `sudo`、`apt-get` 或修改系统环境。智能体不得未经确认直接执行这类操作；应先报告缺失依赖和需要用户执行的命令。
 - 构建失败时优先检查扩展文件、`zmq.hpp`、动态库依赖和当前 Python 环境，不要用静默失败掩盖问题。

 ## 运行与调试

 [run_base.sh](run_base.sh)、[run_cloude.sh](run_cloude.sh)、[run_edge.sh](run_edge.sh) 是当前实验参考脚本：

 - `run_base.sh`：普通 SGLang 服务。
 - `run_cloude.sh`：当前启用的 `STANDALONE_REMOTE` Target，以及注释中的历史 SPECTRE Target 配置。
 - `run_edge.sh`：当前启用的 `STANDALONE_REMOTE` Draft，以及注释中的历史 SPECTRE Draft 配置。
 - 在npu后端，参数--page-size默认为128；在cuda后端，参数--page-size默认为1。

 这些脚本包含硬编码的模型路径、IP、端口、设备号和多组历史配置，只能作为参数模板，不能直接当作通用命令，也不要在修复代码时随意覆盖用户的实验配置。

 远程投机的最小调试顺序：

 1. 先确认两端使用相同 tokenizer 和 vocabulary，并确认通信扩展与依赖已就绪。
 2. 先单机、`tp=1`、短请求和 greedy/低复杂度参数验证正确性。
 3. 先启动 Target，等待服务就绪后再启动 Draft；HTTP 请求只发送到 Target 端口。
 4. 正确性通过后，再逐步尝试跨机 TCP、混合流量、`TP>1` 和 CUDA Graph。
 5. 出现通信、接受率或回滚异常时，优先收集 Target/Draft 两端日志、模式参数、端口映射和最小复现请求。

 `SPECTRE` 的详细约束包括 `num_steps`、`topk`、`num_draft_tokens` 的窗口关系、视觉数据旁路和 ZMQ 通信内容，均以 [README_SPECTRE.md](README_SPECTRE.md) 的当前说明为准。不要仅凭注释掉的启动命令推断协议行为。

 `OPT_TREE_REMOTE` 的启动命令以 [python/sglang/srt/speculative/opt_tree_remote/README.md](python/sglang/srt/speculative/opt_tree_remote/README.md) 为准。上述 `run_cloude.sh` 与 `run_edge.sh` 的已启用配置仍是 `STANDALONE_REMOTE`，不要把脚本里的地址或模型路径写成 OPT 的默认值。

 ## 测试与验证

 测试目录、CI 注册规则和 suite 选择以 [test/README.md](test/README.md) 为准。按风险从小到大选择验证：

 1. 先做语法、导入、静态检查和受影响模块的 CPU 测试。
 2. 单文件测试示例：

		```bash
		python3 test/registered/core/test_srt_endpoint.py
		python3 python/sglang/jit_kernel/tests/test_add_constant.py
		```

 3. 需要 CI suite 时使用项目已有 runner，例如：

		```bash
		python3 test/run_suite.py --hw cpu --suite stage-a-test-cpu
		python3 test/run_suite.py --hw cuda --suite stage-b-test-1-gpu-small
		```

 4. SPECTRE/STANDALONE_REMOTE/OPT_TREE_REMOTE 的完整验证还需要对应 CUDA/NPU 硬件、模型权重、编译扩展和网络条件。当前环境无法完成时，应明确说明未验证项，不得把静态检查结果写成端到端通过。OPT 的 CPU 检查：

		```bash
		PYTHONPATH=python python3.10 test/registered/unit/spec/test_opt_tree_remote.py
		```

 修改测试文件时，遵循现有注册方式和 `test/README.md` 的 unittest/pytest 入口约定；不要自行引入新的测试 runner 或参数解析方式。

 ## 修改纪律

 - 先阅读邻近实现、调用点和测试，再提出一个可证伪的本地假设，并用最小编辑验证。
 - 保持改动聚焦，不顺手重构无关代码，不回滚用户已有改动，不创建 commit 或 branch。
 - `SPECTRE`、`STANDALONE_REMOTE`、`OPT_TREE_REMOTE` 的算法、协议字段和调度状态尽量留在各自目录。共享改动只放在参数注册、`spec_info.py` 的算法判断、`scheduler.py` 的事件循环分发，以及 `triton_backend.py`。`triton_backend.py` 只通过 `target_verify_attention` 的 `prepare` / `ready` 接入，不写 OPT 名字。不要为了复用去改另一条算法的消息格式、角色字段或 `is_*()` 判断，也不要把一种算法的验证长度、page size 或采样规则写进另一种算法的默认路径。
 - 共享协议、采样参数、状态机或通信字段发生变化时，同时检查序列化/反序列化、Target/Draft、fallback、超时和 circuit-breaker。
 - 保持 CUDA 与 NPU 行为对称；如果平台能力不同，显式记录差异和验证边界。
 - 不把实验脚本中的本地路径、私有地址、端口或模型名称写入公共默认值。
 - 修改后至少执行一个与行为相关的窄验证；无法执行时说明阻塞原因和建议的实机验证命令。
 - 交付时说明修改文件、验证命令、验证结果、未验证项及其运行前提。

 ## 事实来源优先级

 当本文件、启动脚本和实现出现不一致时，按以下顺序确认事实：

 1. 当前实现及其邻近测试。
 2. [README_SPECTRE.md](README_SPECTRE.md)、[python/sglang/srt/speculative/opt_tree_remote/README.md](python/sglang/srt/speculative/opt_tree_remote/README.md) 和 [test/README.md](test/README.md) 等专项文档。
 3. 启动脚本中的已启用配置。
 4. 启动脚本中的注释配置和历史实验记录。

 若硬件版本、参数或目录发生变化，先同步对应实现/专项文档，再更新本文件，避免形成第二套事实来源。
