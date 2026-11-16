# CLEAR_TREE_REMOTE

`CLEAR_TREE_REMOTE` 是从 `OPT_TREE_REMOTE` 复制出来的独立算法。服务 W 和 τ 上的 K 都从平表起步：一批一个请求时 W 为 8、K 为 16，其余 W 为 4、K 为 8。在线 K 和在线 W 都要 Target 与 Draft 设置 `SGLANG_CLEAR_LEDGER=1`。验证数 K 默认在 2、4、8、16、32、64 里选，可用 `--clear-tree-ks` 改。前 10 轮、探测轮、样本不够的轮、非贪心轮，以及批建树失败后的回退轮，都用 τ，并且 K 锁在平表。这一档的 KV 桶装不下时退到装得下的最大档。回退轮不加轮次、不取探测、不进回放统计，b 取整个包的请求数。τ 轮的停层按本轮锁定的 K 算质量，不是按层宽。相邻窄档配对满 30 轮、宽档配对满 10 轮，并且增益超过 3%，才换服务 W。偏差只记在日志里。账本默认关闭；关着时不取探测，每一轮都是平表的 W 和 K，停层用 τ，质量按这个 K 算。

启动用 `--speculative-algorithm CLEAR_TREE_REMOTE` 和 `--clear-tree-role`。这些参数与 `--opt-tree-*` 分开。默认 RPC 地址是 `127.0.0.1`，端口是 `30019`。`--clear-tree-widths` 未指定时是 2 4 8 16 32 64，`--clear-tree-ks` 未指定时是同一组数。平表要求宽度里有 4 和 8、验证数里有 8 和 16，缺了拒绝启动。`--speculative-num-draft-tokens` 是图和缓冲容量：Target 为最大 K 加 1，默认 65；Draft 为最大 W，默认 64。Draft 的 bonus 缓冲是最大 W 加 1，默认 65。省略该参数时按这个容量填上；显式传入的值必须等于它。握手仍用 `--clear-tree-width`，它不是候选列表。未设置 `--max-running-requests` 时改为 48。CUDA 和 NPU 图按该进程的 batch 列表捕获，这个列表不是算法常量。启动捕获每一个候选。Target 验证图默认是 65、33、17、9、5、3，另加 decode 的 1，回放按本轮 K+1 选图。Draft 层图默认是 64、32、16、8、4、2，bonus 默认是 65、33、17、9、5、3，另加 decode 的 1。bonus 仍只在 batch 1 捕获。

在线 K、在线 W、账本和回放以本文件和本目录代码为准。[design/README.md](design/README.md) 是设计，不判断正在跑的服务。从 OPT 复制来的建树和验证细节见 [../opt_tree_remote/README.md](../opt_tree_remote/README.md)。

## 账本

在线 K 和在线 W 需要 Target、Draft 都设 `SGLANG_CLEAR_LEDGER=1`。只开一端时，Θ、校准比或 `verify_samples` 缺一块，决策留在 τ 加平表 K。默认关闭。只在 tp rank 0 输出。Draft `tp_size > 1` 直接拒绝启动：账本和样本只在 rank 0，其他 rank 会另走一套 τ。

账本开着时，Draft 记录时间账本和每次层前向，并拟合层耗时。层回归的岭按特征给：截距 0、图档 1e-6、前缀之和 128²。查询的前缀之和比见过的最小或最大再超出 max(128, 这段跨度)，或者预测不大于 0 时，这一档没有层耗时。服务宽度因此走平表，回放里这个候选不配对。层拟合够样本后，停层用同一个 `J_K`：人数从 1 枚举到还能再长的请求数，只保留继续价值更大的那一组。验证样本够了以后，K 在不超过覆盖上界的已捕获档里取 `J_K` 最大的一档。覆盖上界是装得下本轮建树宽度乘以 max_depth 减 1 个节点的最小档，选 K 时再收到各请求的实际节点数。普通轮的建树宽度就是服务宽度，所以 ready 和选 K 只看服务宽度的覆盖档。样本不够、正在做 K 探测、或者非贪心时仍用 τ，K 锁平表。探测上界是宽一档的覆盖档；没有更宽档时是服务宽度的覆盖档。探测轮数达到 max(54, 已过轮数的 5%) 后不再取探测：这一格还有档没有样本时留在平表，样本已经齐则继续在线选 K。Target 把已经完成的 `device_us` 放进下一轮 ClearTree 请求的 `verify_samples`。这条字段不属于 STANDALONE_REMOTE。`SGLANG_CLEAR_LOCK_K=32` 把验证数锁在 32，停层仍用 τ，并且不消耗探测。批建树失败后的回退轮同样是 τ 加平表 K，不进回放。

- `SGLANG_CLEAR_TARGET_BACKEND=cuda`（默认）时，Draft 用 `T_K` 拟合验证耗时。`npu` 时 Draft 仍记录 `T_K`，但不拿它更新验证模型；Target 用 `device_us` 和 NPU 特征拟合。
- `SGLANG_CLEAR_NPU_KV_BUCKETS`：Draft 计算 KV 桶用的列表，逗号分隔。默认 `256,512,1024,2048,4096`。
- Target 的 `t_verify` 用 `SGLANG_NPU_TREE_GRAPH_KV_BUCKETS`（未设置时用同一组默认桶）和 `select_tree_kv_bucket`。超出最大桶记为 eager。事件当时完成时，`device_us` 只在 `[CLEAR ledger t_verify]` 里，不再另打一行。事件未完成时，下一次 poll 补记 `[CLEAR ledger t_verify_device]` 并更新同一条回归。没有 `t_verify_device` 不表示没记到验证耗时。

日志是一行一个 JSON：`[CLEAR ledger d_round]`、`[CLEAR ledger d_layer]`、`[CLEAR ledger d_replay]`、`[CLEAR ledger t_verify]`、`[CLEAR ledger t_verify_device]`。Target 上的验证回归只打日志，选 K 用最近的 `device_us` 样本。完成的 `device_us` 随下一轮请求回到 Draft。账本开着时，预热、K 探测、锁 K 和非贪心之外的合格轮里，当前服务宽度的相邻窄一档如果连续 300 个合格轮没真跑过，就真跑 `probe_w`。这一档的层样本还不到 4 个时，这次最多连续 4 轮，样本满 4 就停；已经满了就仍是 2 轮。计数从 300 起步，所以预热后的第一个合格轮就是第一次。`probe_w` 优先于同一轮的探索，不进回放统计；K 照常选，层回归还没有样本或前缀超出已见范围时走平表。其余每 20 个合格轮真建比当前服务宽度大的相邻一档，种类记为 explore。已经是最大档时这一轮留在当前宽度。判定按这个 b 档进统计的轮数，每 20 轮一次。比当前窄的相邻一档至少 30 个配对轮、比当前宽至少 10 个配对轮，并且增益超过 3%，才换成该档。不设偏差门。往更窄只换成相邻的更小一档。`would_w` 可以是增益更大、但不是这次切换目标的档。偏差仍按 (宽, 窄) 记账，只写在 `bias` 里作诊断：真建成一档时累计这一档的真实 token 和耗时；建成宽档且回放里有更窄一档并有耗时时，累计这次截树，并用这一轮真实宽树做分母。某一对至少各有 30 轮才算出偏差。不想让服务宽度走到 16 以上时，启动用 `--clear-tree-widths 2 4 8`。变宽不再有门；模拟里出现过一次切错的变宽，损失约 0.7% 到 1.9%，这是已知风险。探索轮如果还在 τ，不进回放统计。探测轮数达到上限后不再取 K 探测，已经探完的格继续在线选 K。账本关着时不离开平表，也不取 K 探测。Target 会话号变大时清验证表、探测计数、层回归、只写日志的验证回归、回放累计和还没结算的 c_i，并丢掉还没关闭的那一轮。服务宽度、Θ 和已经结算的 c_i 保留。下一合格轮重新做窄档探测。某个请求的前缀增长超过本轮树深加 1 时，这一轮记为 `target_gap`，不进 Θ、c_i、层回归和回放。
