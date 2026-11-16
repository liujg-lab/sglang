# 在线实验

一次运行回答一个事先写好的问题。运行号是 `validate/<实验>/<YYYY-MM-DD-a>/`。从 `../_template/` 复制四份骨架，再开测。

循环：

1. 对照上一次判为保留的运行。没有保留运行时，对照写「无」。
2. 先写 `question.md`，再改代码、再测。判据不能等看到数字再写。
3. 一次运行只改一个因素。修 bug 和改算法分成两次运行。
4. 跑完写 `verdict.md`。结论是继续改时，停在这个目录，不追加决定。
5. 结论改变规则，或者放弃一条路时，才追加服务包里的 `design/progress/decisions.md` 和 `design/progress/results.md`。决定里引用运行号。

两端日志、客户端记录、`code.patch` 和抽出的 `ledger.jsonl` 放在 `raw/`。这个目录不进仓库。`question.md`、`verdict.md`、`meta.json` 可以提交。

实验格对应文稿 `paper/draft.md` 的小节：

| 目录 | 回答 |
|---|---|
| `cost-layer/` | 第 4 节，层耗时拟合能否被决策使用 |
| `cost-verify/` | 第 4 节，验证耗时随 K 和 KV 桶的变化 |
| `e2e/` | 第 7 节，四条对照的接受长度和吞吐 |
| `ablate/` | 第 7 节，冻住 W、冻住 K，或停层只用 τ |
| `replay-w/` | 第 5 节，宽度是否按配对轮数和 3% 切换 |
| `correctness/` | 第 7 节，温度 0 下与 Target-only 对到第一次分叉 |
| `rtt/` | 第 7 节，往返时间改变后结论是否换方向 |

判据在 `../protocol/DECISION.md`。
