# CLEAR_TREE_REMOTE

`CLEAR_TREE_REMOTE` 是从 `OPT_TREE_REMOTE` 复制出来的独立算法。服务 W 从平表起步：一批一个请求时为 8，其余为 4。在线 K 和在线 W 都要 Target 与 Draft 设置 `SGLANG_CLEAR_LEDGER=1`。验证数 K 默认在 2、4、8、16、32、64 里选，可用 `--clear-tree-ks` 改。前 10 轮、探测轮、样本不够的轮、非贪心轮，以及批建树失败后的回退轮，都用 τ，并且 K 锁在装得下的最大档。回退轮不加轮次、不取探测、不进回放统计。τ 轮的停层按本轮锁定的 K 算质量，不是按层宽。某一对宽度的实测偏差进了 3% 才换服务 W。账本默认关闭；关着时不取探测，每一轮都是 τ 加最大档。

启动用 `--speculative-algorithm CLEAR_TREE_REMOTE` 和 `--clear-tree-role`。这些参数与 `--opt-tree-*` 分开。默认 RPC 地址是 `127.0.0.1`，端口是 `30019`。`--clear-tree-widths` 未指定时是 2 4 8 16 32 64，`--clear-tree-ks` 未指定时是同一组数。`--speculative-num-draft-tokens` 是图和缓冲容量：Target 为最大 K 加 1，默认 65；Draft 为最大 W，默认 64。Draft 的 bonus 缓冲是最大 W 加 1，默认 65。省略该参数时按这个容量填上；显式传入的值必须等于它。握手仍用 `--clear-tree-width`，它不是候选列表。未设置 `--max-running-requests` 时改为 48。CUDA 和 NPU 图按该进程的 batch 列表捕获，这个列表不是算法常量。启动捕获每一个候选。Target 验证图默认是 65、33、17、9、5、3，另加 decode 的 1，回放按本轮 K+1 选图。Draft 层图默认是 64、32、16、8、4、2，bonus 默认是 65、33、17、9、5、3，另加 decode 的 1。bonus 仍只在 batch 1 捕获。

在线 K、在线 W、账本和回放以本文件和本目录代码为准。[design/README.md](design/README.md) 是设计，不判断正在跑的服务。从 OPT 复制来的建树和验证细节见 [../opt_tree_remote/README.md](../opt_tree_remote/README.md)。

## 账本

在线 K 和在线 W 需要 Target、Draft 都设 `SGLANG_CLEAR_LEDGER=1`。只开一端时，Θ、校准比或 `verify_samples` 缺一块，决策留在 τ 加最大档。默认关闭。只在 tp rank 0 输出。Draft `tp_size > 1` 直接拒绝启动：账本和样本只在 rank 0，其他 rank 会另走一套 τ。

账本开着时，Draft 记录时间账本和每次层前向，并拟合层耗时。层拟合够样本后，停层用同一个 `J_K`：人数从 1 枚举到还能再长的请求数，只保留继续价值更大的那一组。验证样本够了以后，K 取 `J_K` 最大的已捕获档。样本不够、正在探测、或者非贪心时仍用 τ，K 锁最大档。Target 把已经完成的 `device_us` 放进下一轮 ClearTree 请求的 `verify_samples`。这条字段不属于 STANDALONE_REMOTE。`SGLANG_CLEAR_LOCK_K=32` 把验证数锁在 32，停层仍用 τ，并且不消耗探测。批建树失败后的回退轮同样是 τ 加最大档，不进回放。

- `SGLANG_CLEAR_TARGET_BACKEND=cuda`（默认）时，Draft 用 `T_K` 拟合验证耗时。`npu` 时 Draft 仍记录 `T_K`，但不拿它更新验证模型；Target 用 `device_us` 和 NPU 特征拟合。
- `SGLANG_CLEAR_NPU_KV_BUCKETS`：Draft 计算 KV 桶用的列表，逗号分隔。默认 `256,512,1024,2048,4096`。
- Target 的 `t_verify` 用 `SGLANG_NPU_TREE_GRAPH_KV_BUCKETS`（未设置时用同一组默认桶）和 `select_tree_kv_bucket`。超出最大桶记为 eager。事件当时完成时，`device_us` 只在 `[CLEAR ledger t_verify]` 里，不再另打一行。事件未完成时，下一次 poll 补记 `[CLEAR ledger t_verify_device]` 并更新同一条回归。没有 `t_verify_device` 不表示没记到验证耗时。

日志是一行一个 JSON：`[CLEAR ledger d_round]`、`[CLEAR ledger d_layer]`、`[CLEAR ledger d_replay]`、`[CLEAR ledger t_verify]`、`[CLEAR ledger t_verify_device]`。Target 上的验证回归只打日志，选 K 用最近的 `device_us` 样本。完成的 `device_us` 随下一轮请求回到 Draft。账本开着时，预热、探测、锁 K 和非贪心之外，每 20 个合格轮只真建比当前服务宽度大的相邻一档，种类记为 explore。已经是最大档时这一轮留在当前宽度。判定按这个 b 档进统计的轮数，每 20 轮一次。比当前窄至少 30 轮、比当前宽至少 10 轮、增益超过 3%，并且该对宽度的偏差门开着，才换成该档。往更窄只换成相邻的更小一档。`would_w` 可以是门还没开的候选。偏差按 (宽, 窄) 记账：真建成一档时累计这一档的真实 token 和耗时；建成宽档且回放里有更窄一档并有耗时时，累计这次截树，并用这一轮真实宽树做分母。某一对至少各有 30 轮，并且截树吞吐相对真实宽树的偏差与真实窄树相对同一分母的偏差之差不超过 3%，这道门才开。上移看 (下一档, 当前)。往更窄退回不靠探索，窄档的真实前向只在它做过服务宽度时留下。一批从 8 起步且从未服务过 4 时，(8, 4) 开不了，服务宽度不能退回 4。日志的 `bias` 和 `gate` 报当前服务宽度对 `would_w` 那一对。账本关着时不离开平表，也不取 K 探测。
