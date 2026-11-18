# CLEAR_TREE_REMOTE

`CLEAR_TREE_REMOTE` 是从 `OPT_TREE_REMOTE` 复制出来的独立算法。服务 W 和 τ 上的 K 都从平表起步：一批一个请求时 W 为 8、K 为 16，其余 W 为 4、K 为 8。在线 K 和在线 W 由 Draft 的 `SGLANG_CLEAR_LEDGER=1` 打开。验证数 K 默认在 2、4、8、16、32、64 里选，可用 `--clear-tree-ks` 改。探测轮、平表档还没有样本的轮、层样本不够的轮、非贪心轮，以及批建树失败后的回退轮，都用 τ，并且 K 锁在平表。这一档的 KV 桶装不下时退到装得下的最大档；平表档超过本轮覆盖上界时，改用上界以内最大的一档。回退轮不加轮次、不取探测、不进回放统计，b 取整个包的请求数。τ 轮的停层按本轮锁定的 K 算质量，不是按层宽。相邻窄档和宽档配对都满 30 轮，并且增益超过 3%，才换服务 W。偏差只记在日志里。账本默认关闭；关着时不取探测，每一轮都是平表的 W 和 K，停层用 τ，质量按这个 K 算。

启动用 `--speculative-algorithm CLEAR_TREE_REMOTE` 和 `--clear-tree-role`。这些参数与 `--opt-tree-*` 分开。默认 RPC 地址是 `127.0.0.1`，端口是 `30019`。`--clear-tree-widths` 未指定时是 2 4 8 16 32 64，`--clear-tree-ks` 未指定时是同一组数。平表要求宽度里有 4 和 8、验证数里有 8 和 16，缺了拒绝启动。`--speculative-num-draft-tokens` 是图和缓冲容量：Target 为最大 K 加 1，默认 65；Draft 为最大 W，默认 64。Draft 的 bonus 缓冲是最大 W 加 1，默认 65。省略该参数时按这个容量填上；显式传入的值必须等于它。握手仍用 `--clear-tree-width`，它不是候选列表。未设置 `--max-running-requests` 时改为 48。CUDA 和 NPU 图按该进程的 batch 列表捕获，这个列表不是算法常量。启动捕获每一个候选。Target 验证图默认是 65、33、17、9、5、3，另加 decode 的 1，回放按本轮 K+1 选图。Draft 层图默认是 64、32、16、8、4、2，另加 decode 的 1。Draft bonus 预测默认关闭，不捕获 bonus 专用图；显式开启时，bonus 图按候选宽度加 1 捕获，默认是 65、33、17、9、5、3，仍只在 batch 1 捕获。

在线 K、在线 W、账本和回放以本文件和本目录代码为准。[design/README.md](design/README.md) 是设计，不判断正在跑的服务。从 OPT 复制来的建树和验证细节见 [../opt_tree_remote/README.md](../opt_tree_remote/README.md)。

## Draft 回包后的 KV 补算与 bonus 预测

Draft 的 `SGLANG_CLEAR_BONUS_PREDICTION` 默认关闭（未设置或设为 `0`）。page size 为 1、`SGLANG_CLEAR_TREE_KV_REUSE` 开启且请求可复用时，回包后仍按各请求的停止深度分组补算最后一层 KV，再发布树 KV lease；下一轮照常复用已接受路径的 KV。关闭时不保存 bonus 候选、不做最后层 bonus 专用 top-k/拷贝，也不做猜测 token 的额外前向。Target 验证后实际提交的 bonus token 不受影响，在下一轮由正常 ingest 路径处理。

在 Draft 启动前设置 `SGLANG_CLEAR_BONUS_PREDICTION=1` 可开启预计算：在根和每个进入验证的节点处各猜一个未包含在验证子节点中的 token，提前写入其 KV 和后续 logits；命中且成为新提交尾部时可跳过该请求下一轮的 ingest 前向。此开关只控制 ClearTree Draft，独立于树 KV 复用，不修改 Target 的 K+1 验证图、协议、在线 K 或账本时间定义。按启动配置使用，修改开关后需重启 Draft，不支持运行中热切换。page size 非 1 的现有限制不变。

同一次回包后补层中，至少两个非 lookahead 请求完成最后层且有 bonus 候选时，直接使用一次批量 bonus 前向，不再逐请求计算，也没有额外 batch 开关。各请求补齐到统一查询宽度 G：优先取能覆盖最大候选数的最小现有层图档。例如最多 9 个候选用 G=16，最多 5 个用 G=8。候选超过最大可用档时，按残余路径质量 `end_rank` 保留前 G 个，保留原有相对顺序；这会降低预计算覆盖率，不改变 Target 提交 bonus token 的规则。padding 行不保留为猜测，前向结束后释放其槽位。

CUDA Draft 通过 `is_bonus=False` 复用普通多请求层图，不新增图。只捕获 batch 1/2/4 且允许 graph padding 时，实际 batch 3 使用 batch 4 图；准入时检查实际 replay key，避免只根据配置宽度推定图存在。禁用 padding 且缺少精确 batch 图时走 eager；图显式关闭以及没有 Draft verify 图的 NPU 路径也走批量 eager。CUDA 图启用但找不到安全可复用的层图时跳过本批 bonus。只有一个合格请求或使用 lookahead 时，仍走原来的 W+1 单请求 bonus 图与 cascade 路径。

批量 bonus 仅使用 allocator 当前空闲的 b×G 个槽位，不把可驱逐前缀 KV 算入预算；空间不足、映射越界、重复请求行，或使用混合 SWA allocator 时整批跳过，不驱逐、不拆批、不回退逐请求串行 bonus。SWA 的 `alloc()` 可能先从一个子池取走槽位，再在另一侧失败，调用方拿不到返回值；这项优化因此直接跳过。临时映射在完成前向后清除，`kv_allocated_len` 恢复到树末尾，真实猜测槽位脱离映射后交给请求持有；下一轮命中安装和未用猜测释放不变。普通失败确认设备完成后回收全部未移交槽位，仍可发布有效树；设备完成或 allocator 释放结果不明时隔离槽位并停止正常复用，不冒险重复释放。

批量路径沿用 Draft 周期计数器，`opt_bonus_rounds` 仍是请求尝试数，不是模型 forward 数。新增计数包括：`opt_bonus_batch_forwards`（提交调用次数，失败也可能计入）、`opt_bonus_batch_requests` / `opt_bonus_batch_query_rows` / `opt_bonus_batch_padding_rows`（成功安装的请求/查询/padding 行）、`opt_bonus_batch_truncated`（丢弃候选数）、`opt_bonus_batch_graph` / `opt_bonus_batch_eager` / `opt_bonus_batch_unknown`（前向返回的执行路径），以及 `opt_bonus_batch_skipped_kv` / `opt_bonus_batch_skipped_graph` / `opt_bonus_batch_skipped_map` / `opt_bonus_batch_skipped_allocator` / `opt_bonus_batch_failed`。这些不是普通建树层样本，不写入层耗时拟合。

CPU/mock 测试覆盖 batch 2/3/4 的拼装、图 key 选择、预算、padding、logits 保活与异常回收，不代表 CUDA/NPU 实机验证通过。实机应先以固定前缀/树核对 position、可见 KV、槽位及参考 logits，再检查 batch 2/3/4 的一次前向和实际图命中；batch 1 与 lookahead 单独回归。没有受控测量不能把既有回包后耗时推算当成批量版本收益；既有 bonus 开关实验的输出分叉也不因本次修改而视为解决。

## 账本

在线 K 和在线 W 由 Draft 的 `SGLANG_CLEAR_LEDGER=1` 打开。Target 的账本只记自己的日志，不参与选 K。默认关闭。只在 tp rank 0 输出。Draft `tp_size > 1` 直接拒绝启动：账本和样本只在 rank 0，其他 rank 会另走一套 τ。

账本开着时，Draft 记录时间账本和每次层前向，并拟合层耗时。层回归的岭按特征给：截距 0、图档 1e-6、前缀之和 128²。查询的前缀之和比见过的最小或最大再超出 max(128, 这段跨度)，或者预测不大于 0 时，这一档没有层耗时。服务宽度因此走平表，回放里这个候选不配对。层拟合够样本后，停层用同一个 `J_K`：人数从 1 枚举到还能再长的请求数，只保留继续价值更大的那一组。平表档有样本以后，K 在不超过覆盖上界、并且这一格自己有样本的档里取 `J_K` 最大的一档。覆盖上界是装得下本轮建树宽度乘以 max_depth 减 1 个节点的最小档，选 K 时再收到各请求的实际节点数。普通轮的建树宽度就是服务宽度，所以 ready 和选 K 只看服务宽度的覆盖档。平表档超过这个上界时，改用上界以内最大的一档。平表档还没有样本、正在做 K 探测、或者非贪心时仍用 τ，K 锁平表；只有非平表档有样本时不取探测。平表档有样本之后才探别的档，跳过平表档，离它最近的先探，每档 2 个样本。探测上界是服务宽度的覆盖档，不为下一档更宽的探索树预先补齐所有 K。探测轮数达到 max(2 × K 的档数, 已过轮数 / 20) 后不再取探测，这一格停在已有样本的档上继续在线选 K。Target 把已经完成的 `device_us` 放进下一轮 ClearTree 请求的 `verify_samples`。这条字段不属于 STANDALONE_REMOTE。Draft 解析它并打日志，不再写入选 K 用的验证表。验证表在轮次关闭时写入 Draft 的 `T_K`，托底轮也写。`SGLANG_CLEAR_LOCK_K=32` 把验证数锁在 32，停层仍用 τ，并且不消耗探测。批建树失败后的回退轮同样是 τ 加平表 K，排除原因含 `fallback`，不进 Θ、c_i、层回归、验证表和回放。

- `SGLANG_CLEAR_TARGET_BACKEND` 不决定选 K。CUDA 和 NPU Target 都用 Draft 的 `T_K`。`cuda`（默认）时 Draft 仍用 `T_K` 拟合只写日志的影子回归；`npu` 时这条影子回归不更新。Target 自己的 `device_us` 回归只留在 Target 进程。
- `SGLANG_CLEAR_NPU_KV_BUCKETS`：Draft 计算 KV 桶用的列表，逗号分隔。默认 `256,512,1024,2048,4096`。
- Target 的 `t_verify` 用 `SGLANG_NPU_TREE_GRAPH_KV_BUCKETS`（未设置时用同一组默认桶）和 `select_tree_kv_bucket`。超出最大桶记为 eager。事件当时完成时，`device_us` 只在 `[CLEAR ledger t_verify]` 里，不再另打一行。事件未完成时，下一次 poll 补记 `[CLEAR ledger t_verify_device]` 并更新同一条回归。没有 `t_verify_device` 不表示没记到验证耗时。

Draft 在下一条 STEP 到达时只检查上一轮回放的有效性并切出新提交的 token；回包后先完成上一轮回放和宽度判定，再冻结本轮树，最后输出暂存的 `d_round`、`d_layer`、`d_replay`。空闲轮也输出已生成的日志。切宽因此晚一棵树生效：跨切宽边界的那棵旧服务宽度树不进回放统计，正常建宽一档的探索树仍进统计。消息格式不变，Target 日志仍即时输出。

`d_round` 的 `policy_ms` 是该轮回包前的账本收包、计划、停层、选 K 和回放准备耗时；`post_ms` 是该轮 `run_post_reply`（回放、冻结树和日志输出）的耗时，不含此前的 deferred forward。`wait_ms` 是 post 打点至下一条 STEP 收包打点的非负墙钟间隔（无 post 时为 0），包含循环和解包开销，不是纯 socket 等待。`draft_floor` 表示下一条 STEP 在接收入口的非阻塞探测中已经排队；阻塞等到的包不记托底，是否发生完整 50 ms 超时不再参与判断。`T_K` 和原有时间恒等式不变。

日志是一行一个 JSON：`[CLEAR ledger d_round]`、`[CLEAR ledger d_layer]`、`[CLEAR ledger d_replay]`、`[CLEAR ledger t_verify]`、`[CLEAR ledger t_verify_device]`。Target 上的验证回归只打日志。选 K 用 Draft 账本里最近的 `T_K`。`device_us` 仍随下一轮请求的 `verify_samples` 回到 Draft，只进日志。账本开着时，K 探测、锁 K 和非贪心之外的合格轮里，当前服务宽度的相邻窄一档如果连续 300 个合格轮没真跑过，就真跑 `probe_w`。这一档的层样本还不到 4 个时，这次最多连续 4 轮，样本满 4 就停；已经满了就仍是 2 轮。计数从 300 起步，所以第一个合格轮就是第一次。窄档已经有 4 个层样本、本轮前缀之和给不出层耗时、并且距上次 `probe_w` 已过 20 个合格轮时，提前再做一轮。`probe_w` 优先于同一轮的探索，不进回放统计；K 照常选，层回归还没有样本或前缀超出已见范围时走平表。其余每 20 个合格轮真建比当前服务宽度大的相邻一档，种类记为 explore。已经是最大档时这一轮留在当前宽度。判定按这个 b 档进统计的轮数，每 20 轮一次。比当前窄或宽的相邻一档都至少 30 个配对轮，并且增益超过 3%，才换成该档。不设偏差门。往更窄只换成相邻的更小一档。`would_w` 可以是增益更大、但不是这次切换目标的档。偏差仍按 (宽, 窄) 记账，只写在 `bias` 里作诊断：真建成一档时累计这一档的真实 token 和耗时；建成宽档且回放里有更窄一档并有耗时时，累计这次截树，并用这一轮真实宽树做分母。某一对至少各有 30 轮才算出偏差。不想让服务宽度走到 16 以上时，启动用 `--clear-tree-widths 2 4 8`。变宽不再有门；模拟里出现过一次切错的变宽，损失约 0.7% 到 1.9%，这是已知风险。探索轮如果还在 τ，不进回放统计。探测轮数达到上限后不再取 K 探测，已经有平表档样本的格继续在已有档上在线选 K。账本关着时不离开平表，也不取 K 探测。Target 会话号变大时清验证表、探测计数、层回归、只写日志的验证回归、回放累计和还没结算的 c_i，并丢掉还没关闭的那一轮。服务宽度、Θ、已经结算的 c_i，以及下一层增益的累计保留。下一合格轮重新做窄档探测。某个请求的前缀增长超过本轮树深加 1 时，这一轮记为 `target_gap`，不进 Θ、c_i、层回归和回放。
