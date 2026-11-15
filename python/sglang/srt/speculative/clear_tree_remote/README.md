# CLEAR_TREE_REMOTE

`CLEAR_TREE_REMOTE` 是从 `OPT_TREE_REMOTE` 复制出来的独立算法。回放控制器没有接入。层宽 W 用平表：一批一个请求时为 8，其余为 4。验证数 K 在 8、16、32 里选；前 10 轮、探测轮和样本不够的轮仍用 τ。旁路账本默认关闭，见下一节。

启动用 `--speculative-algorithm CLEAR_TREE_REMOTE` 和 `--clear-tree-role`。这些参数与 `--opt-tree-*` 分开。默认 RPC 地址是 `127.0.0.1`，端口是 `30039`。`--speculative-num-draft-tokens` 是图和缓冲容量：Target 为 33，Draft 为 8。握手仍用 `--clear-tree-width`。Target 按 9、17、33 各捕获一张验证图，回放按本轮 K+1 选图。Draft 层图是 4 和 8，bonus 是 5 和 9。

当前建树、验证和调度规则见 [../opt_tree_remote/README.md](../opt_tree_remote/README.md)。要接入的控制器见 [design/README.md](design/README.md)。

## 旁路账本

`SGLANG_CLEAR_LEDGER=1` 时，Draft 记录时间账本和每次层前向，并拟合层耗时。默认关闭。只在 tp rank 0 输出。层拟合够样本后，停层比较“再长一层的 J”和“现在停下的 J”；验证样本够了以后，K 取 `J_K` 最大的已捕获档。样本不够时仍用 τ。Target 把已经完成的 `device_us` 放进下一轮 ClearTree 请求的 `verify_samples`。这条字段不属于 STANDALONE_REMOTE。`SGLANG_CLEAR_LOCK_K=32` 把验证数锁在 32，停层仍用 τ，并且不消耗探测。

- `SGLANG_CLEAR_TARGET_BACKEND=cuda`（默认）时，Draft 用 `T_K` 拟合验证耗时。`npu` 时 Draft 仍记录 `T_K`，但不拿它更新验证模型；Target 用 `device_us` 和 NPU 特征拟合。
- `SGLANG_CLEAR_NPU_KV_BUCKETS`：Draft 计算 KV 桶用的列表，逗号分隔。默认 `256,512,1024,2048,4096`。
- Target 的 `t_verify` 用 `SGLANG_NPU_TREE_GRAPH_KV_BUCKETS`（未设置时用同一组默认桶）和 `select_tree_kv_bucket`。超出最大桶记为 eager。事件未完成时，下一次 poll 补记 `[CLEAR ledger t_verify_device]` 并更新同一条回归。

日志是一行一个 JSON：`[CLEAR ledger d_round]`、`[CLEAR ledger d_layer]`、`[CLEAR ledger t_verify]`、`[CLEAR ledger t_verify_device]`。验证回归按本轮 K 分开。完成的 `device_us` 随下一轮请求回到 Draft。
