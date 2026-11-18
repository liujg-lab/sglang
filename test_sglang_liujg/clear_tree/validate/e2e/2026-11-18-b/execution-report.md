# 执行结论：Target 启动失败，端到端未完成

## 实际发生的阶段

- 事前建档、CPU回归、分析器及清理工具离线自检完成。
- 双端空闲保护通过；NPU物理卡2/3空闲，物理卡0既有任务保持不动；已有SSH隧道身份未变。
- 远端ClearTree原包已备份；仅ClearTree包同步，两端27个文件逐文件SHA256一致；数据hash符合事前约定。
- c1-A1本地CUDA Draft成功就绪，实际捕获batch1/2/3/4、共51张图，日志含r33_4_s0、r17_4_s0、r9_4_s0等bonus专用key。
- NPU Target参数解析失败：不识别`--clear-tree-widths`与`--clear-tree-ks`。只读确认远端`server_args.py`缺这两项CLI注册及对应dataclass字段。
- 尚未启动客户端：预热0条，正式0条；c4未启动。吞吐、TTFT、TPOT、接受率、bonus实际replay均缺失，不能报告为0或通过。
- 按计划停止条件终止，无自动重试、无配置降级、无共享文件同步。服务实现与用户现有启动脚本前后diff一致。

## 清理与证据

本场owned进程已清理；c1阶段与全场两端清理证明均保留，HTTP端口释放、所选设备空闲、隧道身份前后一致。`raw/formal-cleanup.json`及`raw/c1-A1/cleanup.json`均记录services_stopped=true、tunnels_preserved=true。c4未启动，不制造c4清理或完成记录；最终trial_cleanup汇总因此保留missing。

详见`raw/startup-diagnosis.json`、`raw/target-shared-integration-readonly.json`、`formal-blocked.json`和双端完整日志。Draft捕图成功仅是capture证明，不是运行时replay证明。

## 验证命令与结果

- `docker exec -w /home/liujg/workspace/sglang liujg_sglang bash -lc 'PYTHONPATH=python python test/registered/unit/spec/test_clear_tree_remote_shell.py -q && PYTHONPATH=python python test/registered/unit/spec/test_clear_tree_ledger.py -q'`：71+112项通过；完整输出在`raw/cpu-checks.log`。
- `/usr/bin/python3 -B raw/preflight.py --execute`（从场次目录解释相对路径）：成功，包和数据身份一致。
- `/usr/bin/python3 -B raw/run_experiment.py formal --execute`：退出1，Target真实启动错误；未继续c4。
- `/usr/bin/python3 -B raw/analyze.py raw --strict`：退出2，预期保留fail/missing。
- `/usr/bin/python3 -B raw/finalize.py raw --strict`：非零，验已有清理和产物hash，不执行实时清理，不洗白失败。

实际从仓库根执行的完整命令见`raw/commands.txt`。离线分析器修正了argparse usage中的`WATCHDOG_TIMEOUT`等选项名被误归为真实超时的问题；未过滤真正的`error: unrecognized arguments`，没有重跑实验。

## 后续前提和边界

当前修改的E2E效果尚未验证。若继续，需要另行批准并建新场次，先核对NPU共享参数/算法接入与当前ClearTree包的兼容性，做最小对齐后再测；不能直接整文件覆盖远端`server_args.py`或整仓同步，因为NPU有独立修改。不能仅删除候选参数绕过失败并声称原计划完成。

模型配置已按内容hash，权重仅记录文件名与大小清单，未hash大权重内容。`raw/`完整原始产物本机可用、通常不入库。未创建commit或branch，批准计划未改。
