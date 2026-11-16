# 原始记录

把这一次运行的大文件放在本目录：

- `target.log`：Target 服务日志
- `draft.log`：Draft 服务日志
- `client.jsonl`：客户端记录
- `code.patch`：开跑时工作区相对 `HEAD` 的 diff。两端各一份时命名为 `target.patch` 和 `draft.patch`
- `ledger.jsonl`：从服务日志抽出的 `[CLEAR ledger ...]` 行

本目录不进仓库。`question.md`、`verdict.md` 和 `meta.json` 留在上一级。
