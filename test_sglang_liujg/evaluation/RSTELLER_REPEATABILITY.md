# RSTeller 运行记录与重复性对照

`eval_spectre_rsteller.py` 保留原串行/闭式并发请求方式。每轮生成独立的
`<stem>.<UTC时间-唯一编号>.jsonl`、CSV 和 `.run.json`，包括显式指定输出路径时。
CLI 输出路径是文件名模板；终端打印实际路径。旧结果不清空，精度脚本只更新本轮文件。
CSV 的汇总吞吐列改名为 `End-to-end tok/s`，终端使用 `e2e_tok/s`。
这是生成 token 数除以 chunk 墙钟时间，包含 prefill、HTTP 和客户端准备开销，
不包含启动、预热、缓存清理及精度计算。没有 TTFT 的并发请求不再伪造 decode 耗时。

运行示例（服务、图片和数据集已就绪）：

```bash
python test_sglang_liujg/evaluation/eval_spectre_rsteller.py --port 30000 --batch-size 1 --max-items 2 --flush-cache
python test_sglang_liujg/evaluation/eval_spectre_rsteller.py --port 30000 --batch-size 1 --max-items 2 --flush-cache --compare-jsonl <上一轮实际JSONL路径>
```

## 保存的证据

- `.run.json`：CLI 参数、采样参数、可获取的 Target 服务配置、数据集索引顺序、
  预热结果、每次缓存清理的 HTTP 结果、实际文件路径和汇总结果。
  服务配置获取失败明确记录，不阻止生成。未记录的 Draft 配置、源码版本及模型权重
  应随实验另行归档；本地客户端信息不能证明远端使用相同源码或权重。
- JSONL：原始样本、输出文本、服务端 `output_ids`、`finish_reason`、响应元数据、
  采样参数、请求内容指纹、客户端开始/结束时间、chunk/global index 和缓存策略。
  指纹包含 prompt、实际图片内容和采样参数，不包含 HTTP 地址。
- SSE 累计/增量 token 序列依据每条响应的 `completion_tokens` 重建。
  最终长度与计数一致才标记 `token_ids_complete=true`。缺失或断裂时不重新分词填充。
  没有完整 token IDs 的历史文件不能作为逐 token 对照基线。
- 并发时间戳描述客户端提交与完成，数据集顺序不等于服务端实际调度顺序。
  不清缓存时初始缓存状态未知；清理请求失败仍记录并继续运行。
  `http_success` 仅说明 HTTP 成功，需检查响应内容确认清理被服务端接受。

## 对照结果的解释

`--compare-jsonl` 将每个 global index 的对照写入本轮 manifest：请求指纹不同、
请求失败、缺失 token 与 token 不同分开报告。完整序列报告从 0 开始的首个差异位置，
以及两端 token 和序列长度；一端先结束时该端 token 为 null。
同时报告结束原因、文本是否一致及并发度、缓存策略等上下文差异。
`equal_tokens` 只表示这两条输出相等，不意味着参数、缓存状态或 KV 已严格对齐。

先核对两份 manifest 的服务配置、缓存清理结果、预热和请求顺序。
在这些条件一致时，使用独立基线与当前版本保存的接受路径、logits、KV 和下一轮 seed
追踪首个分歧。不能把旧结果中的 234/254 token 差异自动归因于某次 kernel 修改。

## SR 统计口径

`host_mean_ms` 是阶段窗口累计时间除以 32；缺席轮次计零。
`host_max_ms` 是该阶段单轮累计时间的最大值。正常和失败轮次均计入，
同轮多次 KV 搬运先求和。各阶段可能嵌套，不应相加推导总时间。
设备事件采样口径不变，eager 样本不代表 graph 内全部搬运。

本地验证入口：

```bash
PYTHONPATH=python python test/registered/unit/spec/test_sr_comm_metrics.py
PYTHONPATH=python python test/registered/unit/spec/test_sr_rsteller_eval.py
PYTHONPATH=python python test/registered/unit/spec/test_sr_fixed_accept.py
PYTHONPATH=python python test/registered/unit/spec/test_sr_rpd.py
PYTHONPATH=python python test/registered/unit/spec/test_sr_target_tree_fia.py
PYTHONPATH=python python test/registered/unit/spec/test_sr_tail_extend.py
```

2026-10-02 本地检查：使用已有临时 Python 3.12 / PyTorch 2.8 CPU 环境，
设置 `PYTHONPATH=python`、`PYTHONUTF8=1`、`PYTHONDONTWRITEBYTECODE=1`，
未安装或升级依赖。

| 入口 | 结果 |
|---|---|
| `test_sr_comm_metrics.py` | 44 项通过，含同轮重复调用和失败窗口结算 |
| `test_sr_rsteller_eval.py` | 8 项通过，HTTP 使用模拟响应，无远端连接 |
| `test_sr_fixed_accept.py` | 35 项通过 |
| `test_sr_rpd.py` | 20 项通过 |
| `test_sr_target_tree_fia.py` | 27 项通过 |
| `test_sr_tail_extend.py` | 85 项中 84 项通过；1 项因 Windows 缺少 `resource` 模块导入失败 |

最后一项为 `TestIncrementalTailFill.test_mixed_penalty_leaves_incremental_row_neutral`，
失败发生在 penalty 模块依赖导入，不能将该测试记为通过。
本次六个 Python 文件语法编译、四个格式化文件的 Black 检查和 `git diff --check` 通过。

仍需硬件验收：B=1/2/3/4（含 B=3 补齐到 4）、TP=1/2、greedy/RPD、eager/graph、
跨页释放与连续请求。性能对照用独立基线，预热后交替至少三组、每组不少于 500 轮。
本地模拟 HTTP 与 CPU 测试不能代替上述验证，也不构成性能收益证据。
