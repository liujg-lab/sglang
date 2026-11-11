# ClearTree 合成模拟器

端到端的合成模拟，用来回答"ClearTree 的控制逻辑在原理上是否可行"：收益从哪里来、哪些模块有用、规划器要多快。**不是实测**，接受模型、快照分布和成本大多按方案自己的假设生成，结论只作为实测之前的先验。结果的汇总见 `python/sglang/srt/speculative/clear_tree/progress/results.md` 第 7 节。

## 模拟的世界

- **文本固定（greedy 语义）：** 每个请求有一条固定的"真实 token 序列"。在位置 p，Draft 给出 C=8 个候选的概率 Q[p]；Target 的 token 是第 H[p] 个候选，或者不在候选里。命中概率 = sigmoid(a + g·logit(q1) + δ_i)，命中后按 q^β 选候选。所有策略在同一批请求上配对比较。
- **模型设错（`misspec`）：** 每个位置多一个规划器不知道的 AR(1) 难度（ρ=0.9，标准差 1.0），而且 Target 在候选间更集中（真实 β=2，模型 β=1）。
- **成本（ms）：** Target 验证 t0 + t1·b·(K+1)；Draft 每层 cD0 + cD1·bucket(n)·W；每轮加 RTT；第 1 层免费（回包后的补算和等待重叠）。`World2` 还可以加 5% 的成本噪声、RTT 指数抖动、按请求不同的输出长度、按位置相关的 Draft 难度（`draft_ar`）。
- **校准（`calibrated.CAL`）：** a=1.5，请求置信度 s∈(0.5, 0.9)，t1=0.2，使 b=1 时 OPT(8) 每轮约 3.5 个 token（实测约 3.6），RTT 4 ms 时投机和 Target-only 在 b≈7–8 打平（实测短前缀 b≥7 时 Target-only 更快）。
- **第 1–6 轮的 OPT 停止规则和真实 OPT 不同（第 7 轮发现）：**
  - `OPT`、`ext2.OPT2` 在"前沿最好路径分数 ≥ τ"时继续长。真实 OPT 在"前 N 路径概率之和 E 这一层的增量 > τ"时继续长（`opt_tree_ops.py` 的 `depth_rule`）。
  - 旧规则做出的静态基线偏弱，平均约 4%。从第 7 轮起，基线用 `ext7.OPTReal`。
- **第 7 轮的世界（`run7.World7`）：**
  - 成本按 T0 实测拟合（`fit7.py`），单位是真实的 ms：每个请求每轮有一项和 K 无关的开销，验证 token 几乎免费，Draft 层宽和层数是贵的部分。
  - Draft 置信度分布用 `ext7.set_conc` 调成更两极。
  - 第 1 层在关键路径上，因为拟合用的实测数据关了 KV 复用。
  - 结果见 progress/results.md 7.8 节。

## 文件

| 文件 | 内容 |
|---|---|
| `feasibility.py` | 世界、Target-only、OPT(N, τ)、ClearTree 第一版原型（逐请求循环）；第一组（未校准）场景 |
| `calibrated.py` | 校准后的 11 个场景：静态网格、ClearTree 参数已知版和在线学习版、规划开销敏感性 |
| `ext.py` | 向量化规划器、一次性子集取最大的 DP、晚一层决策、随位置变化的 δ、自适应投机开关、静态开关、按时间切换 RTT |
| `run_rest.py`、`beta_rerun.py`、`plan_bench.py` | 规划开销（按实测扣、晚一层决策）、学 β、规划器计时 |
| `gen_cases.py`、`planner_bench.js`、`planner_bench_opt.js` | 规划器核心的 JS 移植（V8 即时编译）：先用 numpy 参考用例核对，再计时 |
| `ext2.py`、`run2.py` | 第二批：成本噪声和在线学习、强制 K、偏差修正和 C_dec、沿用 K_g、W 慢环（UCB）、开关退避、拿不到 r_i、快照库的选择偏差（kNN、IPW 加探测） |
| `ext3.py`、`run3a.py`、`run3b.py` | 第三批：保守系数、用模型做先验的慢环、第一版核心方案的完整评估 |
| `ext4.py`、`run4.py`、`run5.py` | 第四、五批：按上下文记状态、探测占比有上限的开关（`SpecSwitch2`），三种退出测试（滑动窗口、不重叠块、两档）；长前缀成本（`long_kw`） |
| `ext5.py`、`run6.py`、`run6b.py` | 第六批（审稿意见 13）：成本模型（EMA 表、水平 + 效应、锚定的水平 + 差值、前缀分格、已知成本）、K_g 重选触发、δ 衰减、开关计时口径（`WorldOH`、`SpecSwitch3`）、停止边界审计（`audit_summary`）、按路径对 δ 后验积分（`CT6`）、真实参数（`TrueCal`）；`simulate4` 支持随时间变化的 RTT 和前缀阶段 |
| `real_t0.py`、`ext7.py`、`fit7.py` | 第七批：T0 实测数据；真实 OPT 停止规则（`OPTReal`）、可调的 Draft 置信度分布（`set_conc`）、层宽和验证数解耦的静态 OPT（`OPTDecoupled`）；按实测拟合成本（非负最小二乘，"只按 token"和完整两种形式，b=4 留出检验） |
| `run7.py`、`run7b.py`、`run7c.py`、`analyze7.py` | 第七批评估：拟合出的世界（`World7`）和原来的世界里，ClearTree 核心对真实规则的耦合静态 OPT、解耦静态 OPT、生产规则；K=W 的 ClearTree 消融 |
| `run8.py`、`diag8.py`、`analyze8.py` | 第八批：低、中、高三种接受率负载（同一套拟合成本）下的静态网格、互换参数的损失、混合负载（按类型设 τ、ClearTree 核心）；高接受率负载下 ClearTree 少长层的诊断 |
| `ext8.py`、`run9.py`、`diag9.py`、`diag9b.py`、`analyze9.py` | 第九批：ClearTree 变体（K 固定、成本裕量 1.0、已知参数）和停止边界审计；按最大似然拟合的接受模型（`CT5MLE`）；两条简单规则："按 batch 算账"（`BatchAccount`）和"接受率 → τ"（`OPTRhoTau`），在纯、混合、分阶段负载下对比 |
| `run10.py`、`analyze10.py` | 第十批：按 batch 算账加 D42 开关，种子 8–10 样本外复核（纯、混合、分阶段负载）；高接受率 b=8 时宽度和收尾阶段的诊断（满批和收尾分开统计） |
| `ext9.py`、`run11.py`、`run11b.py`、`analyze11.py` | 第十一批：第十版规则 `BatchAccount2`（Θ 按 b 分开并跨 b 共享先验、层成本按 (图档, W) 加线性模型、(W, K) 按 b 档和接受率类别查表）和 b=1 不启用的开关 `SwitchB1`；在种子 7 上定表，种子 8–10 评估；简化表 |
| `ext10.py`、`run12.py`、`run12b.py`、`diag12.py`、`analyze12.py` | 第十二批（审稿意见 14）：`BatchAccount3`（按机会计的 ρ̂、首位 top-1 分类、校准比遗忘、按轮冻结、全局 Θ + 重置、不滞后的预知版）、`SwitchTok`（b=1 的保守开关）、退出审计（复制树往下长 1–2 层，算一步后悔和两层结构差距）；估计量对树形的依赖 |
| `ext11.py`、`run13.py`、`analyze13.py` | 第十三批：在线选 K（多验证节点的边际价值）和在线选 W（首位排名分布的链式模型，带 K/W 深度上限）对比查表；没见过的负载和验证 ×10 的世界 |
| `ext12.py`、`run14.py`、`run14b.py`、`analyze14.py` | 第十四批：用已提交文本配对回放选 (W, K)（`BatchAccount5`，可选实测层耗时），对比查表和只在线选 K；没见过的负载、验证 ×10、Draft 快一倍和慢一倍；截窄近似的审计 |
| `diag15.py` | 第十五批探查：用配对回放比较逐层宽度表和固定宽度 |
| `ext13.py`、`run16.py`、`analyze16.py` | 第十六批（审稿意见 15）：选 K 的三种方式（逐档早停、全档枚举、生长和验证同一口径），验证耗时按图档跳变的世界；运行中验证成本突变，两版漂移检测（`BatchAccount6`、`BatchAccount6L`）。`BatchAccount6L` 原来用 `self.n0` 存学基线的轮数，覆盖了 Θ 跨 b 先验的权重；第十七批改名为 `n_off`，B2 用 `r16B2fix` 重跑 |
| `ext14.py`、`run17.py`、`analyze17.py` | 第十七批（审稿意见 16）：S1 跨 K 停止（两层乐观前瞻、真值一层和两层前瞻、跨 K 停止诊断）；S2 跨 K 的边际校准（同树回放的按档位段偏差、f_band）；S3 没在用的 K=32 变便宜（第二版漂移检测、随机交错短测量、低频探测、近似平局探测）（`BatchAccount7`、`BatchAccount7L`） |
| `diag4.py`、`diag4b.py` | 诊断：请求数 ×4 的长时间运行；起步方式和探测日志；不带开关逐轮记录，比较 20 轮和 200 轮窗口 |
| `analyze4.py`、`analyze5.py`、`analyze6.py` | 汇总第四、五、六批，和第三批配对比较 |

跨轮的 rollout 基线在 `../theory/cross_round_rollout.py`。

## 运行

需要带 numpy 的 Python（结果用 Python 3.10 + numpy 1.21 跑出）；JS 计时需要 Node.js。在本目录下运行，输出是一行一个 JSON 或一行文本：

```bash
python3 feasibility.py 0 > out0.txt; python3 feasibility.py 1 > out1.txt   # 两个进程分别跑一半场景
python3 calibrated.py 0 > cal0.jsonl; python3 calibrated.py 1 > cal1.jsonl
python3 run_rest.py A > restA.jsonl; python3 run_rest.py B > restB.jsonl
python3 run2.py A > r2A.jsonl;       python3 run2.py B > r2B.jsonl
python3 run3a.py cons > r3cons.jsonl; python3 run3a.py slow > r3slow.jsonl
python3 run3b.py 0 1.15 fixed > r3b0.jsonl; python3 run3b.py 1 1.15 fixed > r3b1.jsonl
python3 run4.py 0 > r4p0.jsonl;     python3 run4.py 1 > r4p1.jsonl        # 读 results/r3b*.jsonl
python3 run5.py 0 > r5p0.jsonl;     python3 run5.py 1 > r5p1.jsonl        # 读 results/r3b*、r4p*
python3 run5.py 0 b > r5b0.jsonl;   python3 run5.py 1 b > r5b1.jsonl      # 两档退出测试
python3 diag4.py A > d4A.jsonl;     python3 diag4.py B > d4B.jsonl
for a in "3 7" "3 8" "4 7" "4 8"; do python3 diag4b.py $a; done > d4b.jsonl
python3 run6.py 0 r6E.jsonl > r6p0.jsonl; python3 run6.py 1 r6E.jsonl > r6p1.jsonl   # 审计记录写到 r6E.jsonl.0/1
python3 run6b.py 0 > r6b0.jsonl;    python3 run6b.py 1 > r6b1.jsonl
python3 fit7.py 3.0 results/fit7r_C1.json 1 0.5 0.9 1 > results/fit7r_C1.txt   # 其余设定见 results.md 7.8 节
python3 run7.py 0 results/fit7r_C1.json full > r7_C1full.0.jsonl;  python3 run7.py 1 results/fit7r_C1.json full > r7_C1full.1.jsonl
python3 run7.py 0 none original > r7_orig.0.jsonl;                  python3 run7.py 1 none original > r7_orig.1.jsonl
python3 run7b.py 0 results/fit7r_C1.json full > r7b_C1full.0.jsonl; python3 run7b.py 1 results/fit7r_C1.json full > r7b_C1full.1.jsonl
python3 run7b.py 0 none original > r7b_orig.0.jsonl;                python3 run7b.py 1 none original > r7b_orig.1.jsonl
python3 run7c.py 0 results/fit7r_C1.json > r7c_C1.0.jsonl;          python3 run7c.py 1 results/fit7r_C1.json > r7c_C1.1.jsonl
for p in 0 1; do python3 run8.py G $p > results/r8G.$p.jsonl; done   # 然后 P，再 M（M 读 r8G）
python3 diag8.py 4 > results/d8.4.jsonl; python3 diag8.py 8 > results/d8.8.jsonl
for p in 0 1; do python3 run9.py D $p > results/r9D.$p.jsonl; python3 run9.py R $p > results/r9R.$p.jsonl; done   # R 读 r8G
for p in 0 1; do python3 diag9.py $p > results/d9.$p.jsonl; done; python3 diag9b.py > results/d9b.json
for p in 0 1; do python3 run10.py W $p > results/r10W.$p.jsonl; python3 run10.py S $p > results/r10S.$p.jsonl; done   # S 读 r8G、r8M
for p in 0 1; do python3 run11.py C $p > results/r11C.$p.jsonl; done                  # 然后 W、S（读 r11C、r8G、r8M）
for p in 0 1; do python3 run11.py W $p > results/r11W.$p.jsonl; python3 run11.py S $p > results/r11S.$p.jsonl; python3 run11b.py $p > results/r11b.$p.jsonl; done
python3 run12.py R 0; for p in 0 1; do python3 run12.py S $p > results/r12S.$p.jsonl; python3 run12b.py $p > results/r12b.$p.jsonl; done
python3 diag12.py > results/d12_top1.txt
for p in 0 1; do python3 run13.py $p > results/r13.$p.jsonl; done
for p in 0 1; do python3 run14.py $p > results/r14.$p.jsonl; python3 run14b.py $p > results/r14b.$p.jsonl; done
python3 diag15.py
for p in 0 1; do python3 run16.py A $p > results/r16A.$p.jsonl; python3 run16.py B $p > results/r16B.$p.jsonl; python3 run16.py B2 $p > results/r16B2.$p.jsonl; done
for p in 0 1; do python3 run17.py S2 $p > results/r17S2.$p.jsonl; python3 run16.py B2 $p > results/r16B2fix.$p.jsonl; python3 run17.py S1 $p > results/r17S1.$p.jsonl; python3 run17.py S3 $p > results/r17S3.$p.jsonl; done
python3 analyze4.py; python3 analyze5.py; python3 analyze6.py; python3 analyze7.py; python3 analyze8.py; python3 analyze9.py; python3 analyze10.py; python3 analyze11.py; python3 analyze12.py; python3 analyze13.py; python3 analyze14.py; python3 analyze16.py; python3 analyze17.py   # 读 results/
python3 gen_cases.py && node planner_bench.js && node planner_bench_opt.js
```

单个场景从几十秒到几分钟；整批在 2 核机器上 20–60 分钟。规划器计时是在 2 核虚拟机、两个进程同时跑时测的，只能看量级。

## 结果文件（`results/`）

| 文件 | 内容 |
|---|---|
| `out0.txt`、`out1.txt` | 第一组（未校准）14 个场景 |
| `cal0.jsonl`、`cal1.jsonl` | 校准后 11 个场景，含完整静态网格 |
| `beta.out` | 模型设错时学 β 的结果 |
| `restA.jsonl`、`restB.jsonl` | 规划开销（按实测扣、晚一层决策）、随位置变化的 δ、投机开关、两个额外随机种子 |
| `r2A.jsonl`、`r2B.jsonl` | 第二批模块消融 |
| `r3cons.jsonl`、`r3slow.jsonl` | 保守系数（3 个种子）、W 慢环 |
| `r3b0.jsonl`、`r3b1.jsonl` | 第一版核心方案的完整评估（13 个场景 × 2 个种子） |
| `r4p0.jsonl`、`r4p1.jsonl` | 新开关的 4 组设置（同样 26 组）；长前缀 b=1–4 × 2 个种子（含完整静态网格） |
| `r5p0.jsonl`、`r5p1.jsonl`、`r5b0.jsonl`、`r5b1.jsonl` | 退出测试：不重叠块（1 块、连续 2 块）和两档 |
| `r6p0.jsonl`、`r6p1.jsonl` | 第六批 A–E 的汇总（成本模型、K_g 触发、δ 衰减、开关计时口径、审计的条数） |
| `r6E.jsonl.0`、`r6E.jsonl.1` | 停止边界审计的逐条记录（类型、深度、b、μ̂、未来增量之和、余量、dG、dA） |
| `r6b0.jsonl`、`r6b1.jsonl` | 校准对照：学到的 plug-in、按路径对 δ 积分、真实参数，含审计汇总 |
| `fit7_a15.*`、`fit7_a20.*` | 第七批的第一次拟合：原接受模型、旧停止规则 |
| `fit7r_C0.*`–`fit7r_C3.*` | 真实停止规则下 4 种接受设定的拟合（`.txt` 含逐格预测对实测） |
| `r7_C1full.*.jsonl`、`r7_orig.*.jsonl` | C1 世界和原来的世界：Target-only、生产规则、旧规则和真实规则的静态网格、ClearTree 核心（W=4、8，带不带开关） |
| `r7b_C1full.*.jsonl`、`r7b_orig.*.jsonl` | 解耦静态 OPT 的网格（W × K × τ）和最好配置加开关 |
| `r7c_C1.*.jsonl` | K=W 的 ClearTree（只保留逐轮自适应） |
| `analyze7.txt` | `analyze7.py` 的输出 |
| `r8G.*.jsonl`、`r8P.*.jsonl`、`r8M.*.jsonl` | 第八批：三种负载 × b=1、4、8 的静态网格；不调参的 ClearTree；混合负载 |
| `r9D.*.jsonl`、`r9R.*.jsonl`、`d9.*.jsonl`、`d9b.json`、`analyze9.txt` | 第九批：ClearTree 变体和审计；两条规则；最大似然拟合；b=1 用 W=8 的按 batch 算账 |
| `r10S.*.jsonl`、`r10W.*.jsonl`、`analyze10.txt`、`analyze10_combined.txt` | 第十批：开关和多种子；宽度诊断；按 b 组合后的汇总 |
| `r11C.*.jsonl`、`r11W.*.jsonl`、`r11S.*.jsonl`、`r11b.*.jsonl`、`analyze11.txt` | 第十一批：定表；收尾阶段；第十版各变体的样本外评估；简化表 |
| `r12R.json`、`r12S.*.jsonl`、`r12b.*.jsonl`、`d12_top1.*`、`analyze12.txt` | 第十二批：估计量对树形的依赖；各项改动和退出审计；拆开看的跟进；首位 top-1 命中率 |
| `r13.*.jsonl`、`analyze13.txt` | 第十三批：查表、在线 K、在线 W、在线 W+K 和 6 种固定 (W, K) |
| `r14.*.jsonl`、`r14b.*.jsonl`、`analyze14.txt` | 第十四批：查表、在线 K、回放（在线学层耗时 / 实测层耗时）和 6 种固定 (W, K) |
| `d15_schedules.json` | 第十五批：各场景下每种宽度表和 K 的回放吞吐 |
| `r16A.*.jsonl`、`r16B.*.jsonl`、`r16B2.*.jsonl`、`analyze16.txt` | 第十六批：三种选 K 方式；漂移检测两版（`r16B2` 带 n0 覆盖的 bug，以 `r16B2fix` 为准） |
| `r17S1.*.jsonl`、`r17S2.*.jsonl`、`r17S3.*.jsonl`、`r16B2fix.*.jsonl`、`analyze17.txt` | 第十七批：跨 K 停止；跨 K 边际校准；没在用的 K 变便宜；第十六批 B2 的修正重跑 |
| `d8.4.jsonl`、`d8.8.jsonl`、`analyze8.txt` | 高接受率负载的诊断（每个请求每轮的提交数、轮时）；`analyze8.py` 的输出 |
| `d4A.jsonl`、`d4B.jsonl`、`d4b.jsonl` | 诊断。`d4A`、`d4B` 的探测日志里速率单位是 token/ms，只保留 1 位小数（之后改成了 tok/s） |

## 局限

- 只有一类世界生成器；不同模块的"没有收益"可能是因为这个世界里没有它们要防的情况（严重失配、强非平稳）。
- 静态网格的最优配置是事后挑的；吞吐差 2% 以内的比较可能是噪声（多数只有 1–3 个种子）。
- 没有模拟 KV 紧张、NPU 路径、非 greedy 请求和 stop 字符串。长前缀只是把 Target 验证和 Draft 每个 token 的成本乘 3（让 OPT(8, 0.4) 在 b=3 打平），不是实测系数。
- 多数场景只跑约 26 秒，开关的起步和结尾占比偏大。
- 第 1–6 轮的 OPT 基线用的不是真实的停止规则（见"模拟的世界"），那几轮里"相对最好的静态配置"的收益偏高。
- 模拟器的在线学习（`feasibility.ClearTree.learn`）把全局截距从 0 开始用 SGD 更新，接受率高时收敛很慢，会低估接受率（第 9 轮）。ClearTree 在高接受率负载下的数字受这个实现影响。
- D42 开关的块按轮数划分，b=1 时块吞吐噪声大，会误退出（第 10 轮）。
- 第 8 轮的三种负载只在接受率参数上不同，低接受率负载没有实测锚点；静态网格只跑了一个种子。
- 第 7 轮只用了已有的 39 个实测格（lookahead、KV 复用都关，多数只跑 1 次），实测接受长度只有两个点；评估只在 C1 一个设定下跑。
