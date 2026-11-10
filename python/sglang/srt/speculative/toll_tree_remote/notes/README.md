# OPT 实测报告索引

这里的四份底稿整理的是已经落盘的 `OPT_TREE_REMOTE` 测量，用来分开写实验报告。`TOLL_TREE_REMOTE` 没有单独的实机扫描。配置扫描阶段的书面结论是停掉 TollTree，见 [01-serving-config.md](01-serving-config.md)。

原始表在本机 `notes/raw/`，不进仓库。本目录只保留能对上这些文件的均值和结论，不复制日志。`ext.csv` 里状态不是 `ok` 的行没有进入主表。`gate/item1.json` 里的 `bug` 字段不采用，以 `gate/item1-verdict.json` 为准。

| 报告 | 内容 |
|---|---|
| [01-serving-config.md](01-serving-config.md) | 预检、τ/N 扫描、lookahead、KV 复用、Target-only、A/B/C 组 |
| [02-cost-model-gate.md](02-cost-model-gate.md) | 成本拟合和门控第 1–7 项，含 P1–P4 |
| [03-draft-kv-window.md](03-draft-kv-window.md) | 影子行 E1b、E1′、E2–E5，窗口关闭后的输出分叉，以及单前缀 t=72 的逐层比对、惩罚和残差 |

## 共同配置

这些格子用的是同一对模型和服务形态。主机地址、口令和启动脚本里的私有路径留在本机 `notes/raw/`，不进仓库。

| 项 | 取值 |
|---|---|
| 算法 | `OPT_TREE_REMOTE` |
| Target | Qwen3-VL-32B，NPU 物理卡 2 和 3，tp=2，page size 128 |
| Draft | Qwen3-VL-2B INT8，Triton，page size 1，CUDA graph batch 1、2、4、8 |
| 采样 | 温度 0 |
| 输出长度 | `max_new_tokens=256`。合成对照是 192，N=32 的预检格子是 320。E5 是 1024 |
| 树 | `max_depth=10`。宽度 N 和阈值 τ 按格子变化 |
| 请求数 | 预检记录：b=1 为 16，b=2 为 16，b=4 为 32，b=8 为 64 |
| 文本 | 200 条里留下 64 条自然文本，输出长度不少于 256。过滤经过当时的 greedy OPT Target，被接受的 token 就是 Target token |

投机吞吐用 parser 的 `throughput_tok_s`。Target-only 用 client 的 tok/s。接受长度含 bonus。b=1 时，吞吐等于 `accept_len_mean / T_ms * 1000`，其中 `T_ms` 是轮时。

b=8 且 KV 复用打开时，记录里的 R 会被推迟的前向抬高，这个数不是网络时延。同一局域网经隧道测到的 R 约 3–5 ms。D 组的 RTT 扫描没有跑。

重复噪声取自然短提示、N=8、τ=0.1、b=1 的两次：49.27 和 48.99 tok/s，变异系数 0.41%，两倍是 0.82%。后面 4096、b=4 的对照自己就超过这个带宽，窗口实验因此改了门槛，见第三份报告。

## 没有跑的项

- D 组：`sudo tc` 改变隧道时延。当时实验室 RTT 已是 3–5 ms。
- 门控第 8 项：同样是 RTT 扫描，`item7-bound.json` 写明未跑。
- 影子行 E3。E2 过了绝对预算后直接用 w=256 做 E4、E5，没有再插一档宽度。
- TollTree 的实机正确性对照。README 里要求的 `tp=1`、短请求、与同 N 和 τ 且关闭 lookahead 的 OPT 对照，不在这批日志里。
