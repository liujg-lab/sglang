# ClearTree 平表 bonus A/B 结果

## 判定

真实 NPU 32B Target（TP2、物理卡2/3）+ CUDA 3090 2B INT8 Draft 已完成本次矩阵：2轮冒烟、8轮正式ABAB、2轮A/A校准。正式及校准共1000/1000请求成功，冒烟8/8成功，客户端fallback为0；离线分类未发现运行期协议、设备、超时或fallback异常。

**运行验证通过，但输出等价性未通过，因此不能交付为“无语义变化的纯性能优化已验证”。** 两端关闭账本、走平表时，观察到并发1关闭bonus后吞吐下降约5.33%，并发4提升约7.03%；这些是此次工作负载的描述性系统结果，不是严格同输出、同轨迹下的纯bonus成本差。

未更改算法行为、共享协议、已有启动脚本或计划文件；本次只新增实验记录与实验辅助脚本。

## 配置与负载

- A：Draft `SGLANG_CLEAR_BONUS_PREDICTION=1`；B：`0`。同一代码版本，双端27个ClearTree包文件hash一致。
- 双端 `SGLANG_CLEAR_LEDGER=0`，移除锁K，树KV reuse开启，lookahead关闭。
- 平表：实际batch1=W8/K16，其余=W4/K8；τ=0.5、depth8。并发4不等于全程batch4。
- 候选全集、Target65/Draft64容量，max-running-requests4；实际两端捕图batch均只有1/2/4。
- 每轮独立冷启动两端；客户端首条预热丢弃，之后flush cache。正式100条、temperature0、max-new-tokens1024、inf到达。捕图与预热不计入吞吐。
- 文本数据SHA256：`8cf47bd9104c062c0ae2729f582044c97b407847a092fc2f3fa343280b4eec22`。请求顺序核对一致；保存的是派生输入hash，不是完整HTTP payload fingerprint。

## 吞吐与延迟

以下吞吐为客户端正式墙钟输出token/s。合并值使用两轮token总和除以两轮墙钟总和，不是速率算术平均。

### 并发1

- A1：66,617 token，58.5045 tok/s；A2：66,617 token，58.7633 tok/s。
- B1：66,304 token，55.5875 tok/s；B2：66,304 token，55.4268 tok/s。
- 合并A **58.6336 tok/s**，B **55.5071 tok/s**，B相对A **−5.3323%**。
- 每轮平均请求E2E：A 11.386/11.336秒；B 11.928/11.962秒。
- 每轮平均TTFT：A 75.25/77.25毫秒；B 74.60/78.02毫秒。
- 每轮平均TPOT：A 18.018/17.920毫秒；B 18.957/18.953毫秒。

### 并发4

- A1：66,465 token，127.4233 tok/s；A2：66,465 token，128.7423 tok/s。
- B1：66,147 token，136.9685 tok/s；B2：66,147 token，137.1937 tok/s。
- 合并A **128.0794 tok/s**，B **137.0810 tok/s**，B相对A **+7.0282%**。
- 每轮平均请求E2E：A 20.508/20.297秒；B 18.950/18.918秒。
- 每轮平均TTFT：A 216.83/217.00毫秒；B 207.03/207.30毫秒。
- 每轮平均TPOT：A 33.533/33.394毫秒；B 31.399/31.235毫秒。

分位数、接受率、verify次数和所有原始客户端指标见 `summary.json` 与 `raw/results.csv`。上述TPOT沿用现有客户端口径，不与ITL混用。

### A/A校准与时间漂移

- c4-AA1：123.0443 tok/s；c4-AA2：121.2228 tok/s；完整输出100/100一致。
- 两次A/A相对差1.4803%，因此事前规则的吞吐门槛为1.4803%。两种并发的A/B变化幅度均超过该简单门槛。
- **但晚期A/A比早期c4-A1/A2更慢，存在约3%–6%的跨阶段漂移。** 相邻A/A差不能完整约束整个多小时场次漂移；并发1没有独立同负载A/A。两轮重复不是统计显著性检验。
- 分析器仍将性能结论标为 `inconclusive_checks_not_passed`，因为A/B输出等价性失败；不把“超过门槛”升级为严格因果收益。

## bonus、补层与ingest机制证据

周期指标按metered rounds加权，覆盖所提供完整日志的已刷新窗口，**包含服务和客户端预热**，不与正式客户端分母混用。计数单位不一定是请求或模型forward，不能把 `opt_bonus_rounds` 解释为bonus forward次数。

### 并发1

- 每个A重复：16,169次bonus尝试，8,045次命中及零ingest；16,361次deferred最终层forward。
- 每个B重复：bonus尝试/命中/零ingest均未观察到；**16,250次deferred最终层forward仍保留**。
- Draft平均ingest约 A 2.85–2.88ms/轮，B 5.99–6.02ms/轮。
- 由 `opt_post_reply_us / metered_rounds` 计算回包后阶段约 A 15.86–16.00ms/轮，B 5.56–5.58ms/轮。
- Target周期 `rpc_wait` 约 A 27.89–28.14ms/轮，B 32.11–32.15ms/轮。

观察与以下解释一致：单请求时原bonus后台成本较容易被Target工作隐藏，省下下一轮ingest有价值；关闭后回包后阶段变短，但ingest回到关键路径。这里只是机制一致性，不是对每轮重叠关系的证明。

### 并发4

- 每个A重复：18,140次bonus尝试，9,111次命中及零ingest；14,107次deferred最终层forward。
- 每个B重复：bonus尝试/命中/零ingest均未观察到；**14,084次deferred最终层forward仍保留**。
- Draft平均ingest约 A 5.38–5.45ms/轮，B 6.92–7.00ms/轮。
- 回包后阶段约 A 53.33–53.77ms/轮，B 17.17–17.37ms/轮。
- Target周期 `rpc_wait` 约 A 57.85–58.80ms/轮，B 51.62–51.66ms/轮。

观察与以下解释一致：并发4时bonus额外工作更重，关闭后回包后工作明显减少，超过失去零ingest的代价。但本轮无账本，不能推导draft_floor比例或T_K。

### 树KV与图隔离

- B仍有补层计数、树深度与接受路径深度计数；两组misaligned/path lease丢弃计数均为0。
- 这些是复用路径的旁证，**不是严格逐轮lease发布/安装成功率**；CPU单测补充覆盖发布与消费语义。
- Draft A捕获27张图，B捕获21张图；差异为batch1的6个bonus图档。Target保持93张图。所有实际batch捕图档位为1/2/4。
- 全部正式日志无 `[CLEAR ledger ...]`，未为取诊断指标开启账本。

## 输出等价性：未通过，根因未定

- 并发1每次A/B均 **85/100** 条输出不同；并发4每次均 **86/100** 条不同。
- 各并发内A1/A2、B1/B2的100条完整token序列分别全部一致；A/A校准也全部一致。差异是可复现的条件相关差异，不是简单的重复随机波动。
- JSONL保存完整 `output_ids`，100/100配对支持token级首分叉。
- 并发1首分叉token下标（0起）：最小19、中位115、最大844；并发4：最小31、中位106.5、最大631。
- 例：并发1第0条在token163分叉，A token1931、B token12528；第1条在token54分叉，A token1365、B token25。完整逐请求证据见summary中的comparisons及 `raw/repeat-consistency.json`。
- 没有首分叉处的Target/Draft logits、top2 margin和逐轮路径，**不能断言是纯浮点差异，也不能断言已发现bonus KV语义错误**。不同Draft ingest/bonus图形状可能改变草稿与验证树，Target树验证的数值路径也可能改变，但仍是待证伪假设。
- 未额外启动正确性复现实验或改变生产实现；本次测试完成不意味着此问题已解决。后续应以短前缀首分叉重放、Target-only参考及同形状logits/KV核对单独定位。

### 后续只读核查

- 核对bonus候选的逻辑position、祖先mask、接受路径parent/token匹配、KV安装位置和后继seed logits的clone/消费条件，未找到明确off-by-one、候选互相可见或将父节点logits当后继seed的静态证据。未发现足以认定本次存在确定KV/协议错误的证据；这不证明隐藏错误不存在。
- A命中走多候选 `TARGET_VERIFY` 预计算KV/logits，B走正常DECODE补token，计算形状与attention路径确实不同。但仅有Draft树变化，在精确等价的Target greedy验证下本应只改变接受效率，不能直接解释最终输出变化；仍须定位Target logits、上下文/KV或接受映射的哪一步不等价。
- 最早分叉来自并发1的 `global_index=13`（id=`lWLBUhQ`）：前19个输出token相同，第20个A为305、B为5312，文字分别走向 `MK sat hunched ...` 与 `MK sat cross-legged ...`。可用该短前缀优先诊断。
- 核验四次并发1 Target启动的 `random_seed` 分别为173889902、684523933、1025089854、822221773；`enable_deterministic_inference` 均为False。本次未固定服务器种子或启用确定性推理。temperature0且组内输出完全重复，使随机抽样不是首选解释，但不能宣称控制了全部随机/确定性变量。
- 后续聚焦诊断先比较同前缀bonus与普通ingest的token/position/可见KV集合/缓存上下文，再核对首分叉逻辑节点的Target logits及argmax；若argmax一致而提交token不同，应查接受索引与提交映射。以上均为待执行诊断，不是本次已完成验证。

## 验证命令与运行边界

- `PYTHONPATH=python /usr/bin/python3 test/registered/unit/spec/test_clear_tree_remote_shell.py -q`：34项通过。
- `PYTHONPATH=python /usr/bin/python3 test/registered/unit/spec/test_clear_tree_ledger.py -q`：112项通过。
- `python3 raw/analyze.py --self-test`：5组临时回归通过，覆盖真实错误保留与已知日志误报分类。
- `/usr/bin/python3 raw/run_experiment.py smoke` 与 `formal`：在实际执行环境通过；脚本默认拒绝覆盖既有trial目录，不应直接覆盖本归档重跑。
- `/usr/bin/python3 raw/analyze.py`：完整10组收齐，`runtime_status=pass`、`output_equivalence_status=fail`、整体status=fail。默认退出码0仅表示分析完成；`--strict`会因等价性失败返回2。
- 运行期诊断与正常停止、可选模型导入和NPU保存格式提示分开；ignored_diagnostics保留行号、摘要、哈希及分类原因，原始日志未改写。
- 未验证在线K/W、T_K、draft_floor；未验证lookahead实机路径、其他模型/设备/并发或图关闭场景。

## 归档与清理

- `raw/`保存全部两端日志、客户端JSONL/CSV/manifest、逐轮配置和启动脚本、实际包hash、最小diff、同步前NPU包备份、分析与运行辅助代码；`artifacts-sha256.json`记录产物校验。
- `progress.log`是自动每15秒左右追加的清洗日志，启动与捕图阶段也持续刷新；正式任务结束后镜像随驱动退出，无独立遗留镜像进程。
- 所有本次服务已停止：CUDA显存回到138MiB，NPU物理2/3无模型进程，30000/30001已关闭。只向本次进程session发送停止信号，没有宽泛pkill。
- 既有SSH隧道保留，Target侧30019仍由原pid6741监听，他人设备任务未操作。未安装或调整全局依赖。

最终建议：保留本次开关实现与默认配置，不据本次观测自动改回默认值；对性能应按负载陈述“并发1回退、并发4改善”，并在任何等价优化结论前先定位可复现输出分叉。
