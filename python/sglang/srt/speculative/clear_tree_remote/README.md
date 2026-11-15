# CLEAR_TREE_REMOTE

`CLEAR_TREE_REMOTE` 是从 `OPT_TREE_REMOTE` 复制出来的独立算法。这一步的行为仍是固定宽度 N 和阈值 τ。第十一版的回放控制器和在线选 K 还没有接入。旁路账本默认关闭，见下一节。

启动用 `--speculative-algorithm CLEAR_TREE_REMOTE` 和 `--clear-tree-role`。这些参数与 `--opt-tree-*` 分开。默认 RPC 地址是 `127.0.0.1`，端口是 `30039`。Target 的 `--speculative-num-draft-tokens` 是 N+1，Draft 的是 N。

当前建树、验证和调度规则见 [../opt_tree_remote/README.md](../opt_tree_remote/README.md)。要接入的控制器见 [design/README.md](design/README.md)。

## 旁路账本

`SGLANG_CLEAR_LEDGER=1` 时，Draft 记录时间账本和每次层前向，并拟合层耗时。默认关闭。打开后宽度、τ 和回包内容不变，预测只写日志，不参与建树。只在 tp rank 0 输出。

- `SGLANG_CLEAR_TARGET_BACKEND=cuda`（默认）时，Draft 用 `T_K` 拟合验证耗时。`npu` 时 Draft 仍记录 `T_K`，但不拿它更新验证模型；Target 用 `device_us` 和 NPU 特征拟合。
- `SGLANG_CLEAR_NPU_KV_BUCKETS`：Draft 计算 KV 桶用的列表，逗号分隔。默认 `256,512,1024,2048,4096`。
- Target 的 `t_verify` 用 `SGLANG_NPU_TREE_GRAPH_KV_BUCKETS`（未设置时用同一组默认桶）和 `select_tree_kv_bucket`。超出最大桶记为 eager。事件未完成时，下一次 poll 补记 `[CLEAR ledger t_verify_device]` 并更新同一条回归。

日志是一行一个 JSON：`[CLEAR ledger d_round]`、`[CLEAR ledger d_layer]`、`[CLEAR ledger t_verify]`、`[CLEAR ledger t_verify_device]`。这一阶段验证长度仍等于启动宽度，所以回归里只有一个 K，看不出各 K 的台阶。
