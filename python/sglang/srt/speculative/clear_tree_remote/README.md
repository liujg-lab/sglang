# CLEAR_TREE_REMOTE

`CLEAR_TREE_REMOTE` 是从 `OPT_TREE_REMOTE` 复制出来的独立算法。这一步的行为仍是固定宽度 N 和阈值 τ。第十一版的回放控制器、在线选 K 和成本账本还没有接入。

启动用 `--speculative-algorithm CLEAR_TREE_REMOTE` 和 `--clear-tree-role`。这些参数与 `--opt-tree-*` 分开。默认 RPC 地址是 `127.0.0.1`，端口是 `30039`。Target 的 `--speculative-num-draft-tokens` 是 N+1，Draft 的是 N。

当前建树、验证和调度规则见 [../opt_tree_remote/README.md](../opt_tree_remote/README.md)。要接入的控制器见 [../clear_tree/README.md](../clear_tree/README.md)。
