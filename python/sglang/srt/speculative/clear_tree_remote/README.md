# CLEAR_TREE_REMOTE

`CLEAR_TREE_REMOTE` 是从 `OPT_TREE_REMOTE` 复制出来的独立算法。服务 W 从平表起步：一批一个请求时为 8，其余为 4。在线 K 和在线 W 都要 Target 与 Draft 设置 `SGLANG_CLEAR_LEDGER=1`。验证数 K 在 8、16、32 里选。前 10 轮、探测轮、样本不够的轮、非贪心轮，以及批建树失败后的回退轮，都用 τ，并且 K 锁在装得下的最大档。回退轮不加轮次、不取探测、不进回放统计。τ 轮的停层按本轮锁定的 K 算质量，不是按层宽。某一对宽度的实测偏差进了 3% 才换服务 W。账本默认关闭；关着时不取探测，每一轮都是 τ 加最大档。

启动用 `--speculative-algorithm CLEAR_TREE_REMOTE` 和 `--clear-tree-role`。这些参数与 `--opt-tree-*` 分开。默认 RPC 地址是 `127.0.0.1`，端口是 `30019`。`--speculative-num-draft-tokens` 是图和缓冲容量：Target 为 33，Draft 为 8。握手仍用 `--clear-tree-width`。未设置 `--max-running-requests` 时改为 48。CUDA 和 NPU 图按该进程的 batch 列表捕获，这个列表不是算法常量。Target 验证图是 9、17、33，另加 decode 的 1，回放按本轮 K+1 选图。Draft 层图是 4 和 8，bonus 是 5 和 9，另加 decode 的 1。

在线 K、在线 W、账本和回放以本文件和本目录代码为准。[design/README.md](design/README.md) 是设计，不判断正在跑的服务。从 OPT 复制来的建树和验证细节见 [../opt_tree_remote/README.md](../opt_tree_remote/README.md)。

## 账本

在线 K 和在线 W 需要 Target、Draft 都设 `SGLANG_CLEAR_LEDGER=1`。只开一端时，Θ、校准比或 `verify_samples` 缺一块，决策留在 τ 加最大档。默认关闭。只在 tp rank 0 输出。Draft `tp_size > 1` 直接拒绝启动：账本和样本只在 rank 0，其他 rank 会另走一套 τ。

账本开着时，Draft 记录时间账本和每次层前向，并拟合层耗时。层拟合够样本后，停层用同一个 `J_K`：人数从 1 枚举到还能再长的请求数，只保留继续价值更大的那一组。验证样本够了以后，K 取 `J_K` 最大的已捕获档。样本不够、正在探测、或者非贪心时仍用 τ，K 锁最大档。Target 把已经完成的 `device_us` 放进下一轮 ClearTree 请求的 `verify_samples`。这条字段不属于 STANDALONE_REMOTE。`SGLANG_CLEAR_LOCK_K=32` 把验证数锁在 32，停层仍用 τ，并且不消耗探测。批建树失败后的回退轮同样是 τ 加最大档，不进回放。

- `SGLANG_CLEAR_TARGET_BACKEND=cuda`（默认）时，Draft 用 `T_K` 拟合验证耗时。`npu` 时 Draft 仍记录 `T_K`，但不拿它更新验证模型；Target 用 `device_us` 和 NPU 特征拟合。
- `SGLANG_CLEAR_NPU_KV_BUCKETS`：Draft 计算 KV 桶用的列表，逗号分隔。默认 `256,512,1024,2048,4096`。
- Target 的 `t_verify` 用 `SGLANG_NPU_TREE_GRAPH_KV_BUCKETS`（未设置时用同一组默认桶）和 `select_tree_kv_bucket`。超出最大桶记为 eager。事件当时完成时，`device_us` 只在 `[CLEAR ledger t_verify]` 里，不再另打一行。事件未完成时，下一次 poll 补记 `[CLEAR ledger t_verify_device]` 并更新同一条回归。没有 `t_verify_device` 不表示没记到验证耗时。

日志是一行一个 JSON：`[CLEAR ledger d_round]`、`[CLEAR ledger d_layer]`、`[CLEAR ledger d_replay]`、`[CLEAR ledger t_verify]`、`[CLEAR ledger t_verify_device]`。Target 上的验证回归只打日志，选 K 用最近的 `device_us` 样本。完成的 `device_us` 随下一轮请求回到 Draft。账本开着时，预热、探测、锁 K 和非贪心之外，当前宽度是 4 就每 20 个合格轮建一次 8，当前宽度是 8 就每 20 个合格轮建一次 4。判定也按这个 b 档进统计的轮数，每 20 轮一次。账本关着时不离开平表，也不取 K 探测。W=2 只出现在回放的 `would_w` 里，不能成为服务宽度。
