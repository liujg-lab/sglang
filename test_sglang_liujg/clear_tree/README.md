# ClearTree 实验材料

ClearTree 的设计文档在 `python/sglang/srt/speculative/clear_tree_remote/design/README.md`，进展、结果和计划在同目录的 `progress/` 下。这里放实验用的脚本：

| 目录 | 内容 |
|---|---|
| [theory/](theory/README.md) | 引理、命题和设计选择的数值检验（合成实验），作为论文附录的可复现材料 |
| [sim/](sim/README.md) | 端到端的合成模拟器：收益来源、模块消融、规划器耗时；结果在 `sim/results/` |
| [tools/](tools/README.md) | 在线验证实验用的代理、客户端和分析脚本 |

这些脚本不属于 SGLang 的 Python 包，也不进 CI。
