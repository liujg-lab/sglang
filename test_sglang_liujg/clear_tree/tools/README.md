# ClearTree 验证工具

`python/sglang/srt/speculative/clear_tree/progress/plan.md` 里前置测量和各阶段实验用到的客户端和分析脚本。只用 Python 标准库和 `requests`，不需要 root。

**状态：** 只在本地的模拟 HTTP 服务器上自测过，没有在真实的 SGLang 服务上跑过。第一次使用前，先用一条请求核对 `/generate` 的返回字段（`output_ids`、`meta_info.output_top_logprobs`、`completion_tokens` 等）和 Target 日志里 `Decode batch` 行的格式，不符时改对应的解析函数。

| 脚本 | 用途 | 用在哪个实验 |
|---|---|---|
| `delay_proxy.py` | 用户态 TCP 延迟代理，放在 Target 和 Draft 的 RPC 连接之间：每个方向加 RTT/2，可选抖动和限带宽；上游未就绪时断开，让 ZMQ 自动重连。本机回环：设定 0 / 10 / 40 ms，实测 0.34 / 11.04 / 41.15 ms | V0，以及所有 RTT 和带宽扫描 |
| `closed_loop_client.py` | 闭环客户端：始终保持 C 个请求在跑；可选 top-k logprobs、流式、按时长循环发送、`max_new_tokens` 抖动 | 所有在线实验 |
| `flip_rate.py` | 按 Target top-1/top-2 差距分档统计翻转率。teacher 模式：预填充打分对比一次运行的文本；pair 模式：两次运行对比，只计到第一次分叉 | V1，统计无损护栏 |
| `window_cv.py` | 按 token 数切窗口，算吞吐的变异系数、自相关，以及分辨 3% 或 5% 的差距需要多少个窗口 | V6 |
| `decode_table.py` | 从 Target 日志的 `Decode batch` 行算 Target-only 的步时表，只用 `#running-req` 等于 b 的行 | V5 |
| `mixed_throughput.py` | 几个请求池同时跑时，统计所有池都满的那段时间的吞吐；可以按时间点切成前后两段 | V3 |
| `echo_test.py` | `delay_proxy.py` 的自测：在本机起一个回显服务，经代理测 0 / 10 / 40 ms 设定下的实际往返时间。在本目录下运行，会占用 29100、29101 端口 | 代理自测 |
| `selftest/fake_server.py` | 模拟 `/generate` 的本地 HTTP 服务（固定的 token 和 top-2 logprobs，支持流式），监听本机 29201 端口，用来自测客户端和分析脚本。自测产生的数据不放进仓库 | 工具自测 |

用法见各脚本开头的说明。代理和客户端的地址、端口都由命令行给出；只有两个自测脚本固定使用本机回环地址和上表里的端口。
