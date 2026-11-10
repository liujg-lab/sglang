# TollTree 设计记录

2026-11-07 起停止开发。这里只保留当时的设计和实测，服务路径以 [README.md](README.md) 为准。下一份算法不在这个目录，见 [../opt_tree_remote/PRICE.md](../opt_tree_remote/PRICE.md)。

## 为什么停止 TollTree

数据都来自 `OPT_TREE_REMOTE`。部署配置：
- Target：Qwen3-VL-32B，NPU 910c，tp=2，page 128，`tree_paged_fia`。
- Draft：Qwen3-VL-2B INT8，RTX 3090，page 1，Triton。
- 采样：greedy，`max_new_tokens=256`。网络是单实验室内的 SSH 隧道，RTT 3–5 ms。

**T0：自适应能多赚多少（基线 RTT）。**

| 可调量 | 相对最优固定配置的上界收益 |
|---|---:|
| 只调 τ，即 V(τ) | 0.36% |
| 调 τ 和 N，即 V(τ, N) | 3.57% |

逐层、逐轮细调的空间只有几个百分点，不值得为它写一套新算法和新协议。b=1 的 lookahead 和 KV 复用已经把单请求做得很好：N=8、τ=0.8 时是 62.6 tok/s，Target-only 约 31 tok/s。

**gate 实验：真正的问题在 b>1 时成本的形状变了。**

| 条件 | 相对 Target-only 的加速比 |
|---|---|
| 短前缀，b=1 / 2 / 4 / 6 / 8 | 2.04 / 1.62 / 1.37 / 1.08 / 0.87 |
| 4096 前缀，b=1 / 2 / 4 | 1.71 / 1.09 / 0.82 |

- **长前缀很早就打平：** 4096 前缀的打平点在 b=2 和 b=4 之间，短前缀约在 b=7。
- **每多一个请求，每轮多出的时间：** 4096 前缀约 36 ms，短前缀约 15 ms。
- **b=1 的模型搬不到 b>1：**
  - cascade 注意力和 lookahead 都只在 b=1 生效，所以成本在 b=1→2 不连续；
  - 用 b=1 推出的模型预测 4096 b=2，误差 −29.7%（预测 75.3，实测 58.0 tok/s）；
  - "长前缀要更深"只在 b=1 成立，b=4 时 τ=0.4（86.3）反而好于 τ=0.2（84）。
- **b=4 和 b=8 的短前缀：** 吞吐在 τ=0.8（扫描上限）还在上升，最优 τ 还没测到。
- **NPU 输出与 Target-only 不同：** 当时看到的分叉都落在 top-2 内，最大差距 0.25 nat，Target-only 自己重复也只有 15/16 一致。后来 4096、t=72 的分叉被记成数值放大，见 [notes/gate-window/DECISION.md](notes/gate-window/DECISION.md)；V1 又把 ≥3 nat 的一档标成疑似 bug，没有改代码，见 [notes/validate/DECISION.md](notes/validate/DECISION.md)。
- **N=8 和 N=16 的图内存：** 分别多 0.20 GB 和 0.53 GB，都走 `tree_paged_fia`，没有回退。

## TollTree 实现状态（停在 T2）

| 部分 | 状态 |
|---|---|
| 决策函数、成本拟合、λ、校准、增益表 | 已实现，只有 CPU 测试 |
| `policy=fixed`、bs=1、无 graph 的服务路径 | CUDA 已验证正确性；NPU 正在验证 |
| `policy=toll`、可变 W 和 K | 未接入服务，启动时拒绝 |
| bs>1、CUDA graph、NPU graph | 未接入，启动时拒绝 |
| lookahead、级联注意力、KV 租约、bonus | 未实现 |
| 多模态、NPU Draft | 多模态绕过投机；NPU Draft 启动时拒绝 |

## 设计思想

1. **把时间换算成 token。** 长期吞吐是每轮 token 的期望除以每轮时间的期望。按分式规划（Dinkelbach），最优点满足 `max(token − λ·时间) = 0`，其中 λ 是吞吐，也就是 1 ms 的机会成本。知道 λ 后，一项投入值得做的条件就是 `增益 > λ × 成本`，W、D、K 不需要联合搜索。
2. **按成本由谁承担来分层。** 深度串行，每加一层都付一次完整前向；宽度和验证节点都在一次前向里并行，单价很低。名字来自这个结构：开一层像过一次收费站，整层只付一次，同层请求分摊。
3. **用单步前瞻停止。** 只判断下一层是否值得开，用闭式门槛替代 M 步采样前瞻。
4. **只用在线数据。** 成本、λ、校准和增益都从服务中的观测得到，不写硬件相关的固定系数，也不做离线 replay。

## 建模

一轮的时间：

```text
T   = R + H + T_t + Σ_d C_d
R   = RTT + (s0·b + s1·b·K) / BW
T_t = w_t + κ_t·Σℓ_i + β·b·K + β⁺·(b·K − K̂)⁺
C_d = w + κ·Σ_{i∈A_d} ℓ_i + ε·|A_d|·W_d + ε⁺·(|A_d|·W_d − M̂)⁺
```

| 符号 | 含义 |
|---|---|
| `b`、`ℓ_i` | batch size；请求 i 的前缀长度 |
| `R`、`H` | 网络时间；Draft 的准备时间 |
| `T_t`、`C_d` | Target 验证时间；Draft 第 d 层的时间 |
| `w`、`κ`、`ε`、`ε⁺`、`M̂` | 层固定成本；前缀斜率；行斜率；拐点后的额外斜率；行数拐点 |
| `β`、`β⁺`、`K̂` | 验证节点斜率；拐点后的额外斜率；节点数拐点 |
| `A_d`、`W_d` | 第 d 层的活跃请求；物理宽度桶 |

收益来自路径接受概率。节点 v 的 `π̂_v = π̂_parent × a_v`，其中 `a_v` 是把 Draft 概率校准后的条件接受率。

三种成本对应三层门槛：

| 成本 | 谁来付 | 值得付的条件 |
|---|---|---|
| 节点单价 `β̃ = β + ms_per_byte × bytes_per_token` | 每个验证节点 | `π̂ > λ·β̃` |
| `κ·ℓ_i + ε·|F_i|` | 每个请求 | `s_i > 0` |
| `w` 和行数拐点 | 同层活跃请求分摊 | `Σ s_i⁺ > 层费` |

## 求解

每轮不做全局搜索，而是在树增长时逐层判断（[policy/decide.py](policy/decide.py)）：

```text
cut_i = max(λ·β̃, 请求 i 已有节点中第 K_max 大的 π̂)
F_i   = 请求 i 当前层 π̂ > cut_i 的节点，最多 W_max 个
ĝ_i   = Σ_{p∈F_i} π̂_p · φ(cut_i / π̂_p)，φ(x) = E[Σ(a − x)⁺]
s_i   = ĝ_i − λ·(κ·ℓ_i + ε·|F_i|)

s_i ≤ 0：请求 i 退出，之后不再回来
W     = 不小于 max_i |F_i| 的最小桶
层费  = λ·w + λ·ε⁺·(活跃请求数·W − M̂)⁺
Σ s_i⁺ ≤ 层费、到 max_depth 或 KV 不足：整轮停止

树长完后：
K = 桶上 argmax Σ_i Σ_{r≤K} π̂_i(r) − λ·β̃·b·K − λ·β⁺·(b·K − K̂)⁺
```

- **宽度：** 没有单独的优化项，由超过 cut 的节点数决定，再上取到桶。
- **深度：** 没有单独的优化项，由逐层判断决定。bs=1 时可以写成闭式 `g_{D+1} > alen/(D+n)`，其中 `n = (R+H+T_t)/C_L`。
- **验证节点数：** 树长完后在桶上选净收益最大的值。`π̂` 从父到子单调不增，所以取前 K 个时祖先自然包含在内。

在线输入：

| 量 | 取法 | 代码 |
|---|---|---|
| λ | 提交 token 数的均值除以相邻 STEP 间隔的均值，按 b 分桶。单步超过慢均值 3 倍视为异常；快慢均值持续偏离 30% 时切到新水平。决策使用 tok/ms | [policy/price.py](policy/price.py) |
| 成本系数 | 带拐点的线性拟合，按路径分开，系数不为负。行数没有变化时，斜率不可辨识 | [policy/cost.py](policy/cost.py) |
| `a` | 按深度和概率分桶的 EMA。只有被接受节点的孩子才打标签，其余按删失处理 | [policy/calib.py](policy/calib.py) |
| `φ` | 按深度保存 `Σ(a−x)⁺` 的 EMA，在对数网格上插值 | [policy/gain.py](policy/gain.py) |

成本系数就绪前（行数至少有两种取值，样本数达到 `warmup_rounds`）使用 warmup，轮换 (W, K)。`policy=fixed` 的建树和验证集与 OPT 相同，`K=N`。

### 网络和 batch size

**网络：**
- RTT 算进 STEP 间隔。RTT 变大时 λ 下降，所有门槛一起下降，树变深。这是通过 λ 间接生效的，决策不直接读 RTT。
- 随 K 增长的字节由 `β̃` 计入，但 `--toll-tree-bytes-per-token` 默认是 0，也就是默认不生效。
- `should_speculate`（`λ < (1+δ)·b/T_AR(b)` 时不投机）已实现，但还没有调用方。

**Batch size：**
- 层费由同层请求分摊，b 变大时层更容易开。
- 请求可以单独退出，其他请求继续长树。
- 张量是矩形的，物理宽度取最大的 `F_i` 再上取到桶。
- 验证成本按 `b·K` 计算。总吞吐 λ 变大时节点单价也变高，K 会收缩。
- λ 按 b 分桶，空桶用 `b/T_AR(b)` 作为先验。

## 启动条件和服务流程

当前服务只接受同时满足这些条件的启动：`--disable-cuda-graph`、`--disable-piecewise-cuda-graph`、`--toll-tree-policy=fixed`、`--max-running-requests=1`，以及 `--toll-tree-role=target` 或 `draft`。`--toll-tree-fixed-width` 必须同时在宽度桶和验证桶上，还要给出 `--toll-tree-fixed-tau`。Draft 必须是 tp=1，并跳过启动 warmup。DP attention 和 NPU Draft 启动时拒绝。

Draft 按层生成 top-C，回包后回滚树区 KV，这一步没有 lease。Target 用 `K+1` 个槽做验证，槽 0 是已提交尾 token，padding 填 0。`commit_len` 是前缀加已输出 token 的长度，STEP 只带增量；对不上时 Draft 回 `NEED_SNAPSHOT`，Target 下一轮重发完整前缀。`K=0` 或空回包退回普通一步解码，超时和 `REJECT` 才记熔断。TP>1 时只有 rank 0 收发，回包广播给其他 rank。

带 `multimodal_inputs` 的请求不进入投机。Qwen3-VL 的启动 warmup 会带图片，而当前 PREFILL 只发送 `input_ids`，所以这类请求的 PREFILL 和 STEP 都走普通解码。

**CUDA 和 NPU 的差异：**
- CUDA `--page-size` 是 1，NPU 是 128。
- NPU Target 由 `npu_target_tree_fia` 进入树注意力选择，符合条件时走 `tree_paged_fia`，否则保留原来选定的实现。
- `speculative_eagle_topk` 保持为空，避免影响普通前向的注意力选择和 KV 预算。验证宽度读 `toll_tree_fixed_width`。

## 关键测试结论

### 建模依据：OPT 上的成本

这些数据来自 `OPT_TREE_REMOTE`，不是 TollTree。配置是 Target Qwen3-VL-32B（NPU）、Draft Qwen3-VL-2B INT8（RTX 3090）、bs=1。

| 量 | 能否并行 | 单位成本 |
|---|---|---|
| 深度 | 不能 | 每层约 5.8 ms，随前缀变长 |
| 验证节点 | 能 | 每节点约 0.22 ms（11→20 token 多 2 ms） |
| 宽度 | 能 | 每节点约 0.02 ms（N=10→19 每层多 0.2 ms） |

| N | τ | 接受长度 | 单步 | 吞吐 | 关键路径层数 |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.5 | ≈2.8 | ≈58 ms | ≈48 tok/s | 未记录 |
| 10 | 0.2 | ≈3.6 | ≈74 ms | ≈49 tok/s | 未记录 |
| 10 | 0.05 | 3.67 | ≈72.7 ms | 49.5 tok/s | 4.58 |
| 19 | 0.05 | 4.25–4.36 | ≈84 ms | 51.3 tok/s | 5.72 |

后两行都开了 lookahead。从 N=10 到 N=19，接受长度增加约 16%，单步时间也增加 15.5%，吞吐只增加 2.6%。多出的时间主要来自 τ 固定时自动变深（约 +6.6 ms），而不是宽度和验证节点。

Draft 单层的图回放 GPU 中位数说明，层成本由前缀决定，几乎与宽度无关：

| 前缀 | 宽度 10 | 宽度 4 |
|---:|---:|---:|
| 128 | 4.49 ms | 4.43 ms |
| 1024 | 7.16 ms | 7.10 ms |
| 4096 | 16.17 ms | 16.06 ms |
| 8192 | 26.87 ms | 28.07 ms |

按这些数推算，N=19 时加一层的盈亏点约为 `0.29`，加一个验证节点的盈亏点约为 `0.011`。这是推算值，不是单独测量的结果。

这些数据跨了多次提交，lookahead 和 KV 复用的状态也不完全相同，而且吞吐主要来自 2 条请求。它们能说明趋势，但不是同一版代码上的对照实验。

### CUDA 正确性

配置是 `policy=fixed`、N=8、τ=0.1、bs=1、无 graph。模型是 bf16 的 Qwen3-0.6B，单条文本请求，`temperature=0`、`ignore_eos=true`、`max_new_tokens=64`。这些结果在 `0c6da7e` 之前测得，只说明正确性，耗时没有参考价值。

假 Draft（[test/manual/toll_fake_draft.py](../../../../../test/manual/toll_fake_draft.py)）加真 Target，tp=1 和 tp=2 结果相同，输出都与 baseline 逐 token 一致：

| 模式 | verify 轮数 | 接受 | 回退 |
|---|---:|---|---:|
| gold | 16 | 前 15 轮各 3 个，最后一轮 2 个 | 0 |
| wrong | 63 | 全部 0 | 0 |
| pad | 32 | 前 31 轮各 1 个，最后一轮 0 个 | 0 |
| silent | 0 | 0 | 63，前 3 次各超时约 5 秒后熔断 |

其他结果：
- **层 logits：** Draft 层前向与普通 EXTEND 的最大绝对差，在 Triton 上约 0.13，短句到过 0.22；FlashInfer 上约 0.15，并且出现过 top-1 不同。这是 kernel 差异，容差定为 `max_abs <= 0.5`，`top1_match` 只记录。`τ=0.1` 时只检查到两层前向。
- **真 Draft + 真 Target：** 输出与 baseline 一致。tp=1 是 17 轮，接受 46 个，回退 0；tp=2 是 19 轮，接受 44 个，回退 0。
- **与 OPT 对照：** N=8、τ=0.1、深度 10，OPT 关闭 lookahead。18 轮接受长度全部相同，匹配率 100%，输出也相同。
- **资源（tp=1）：** 25 次 200-token 的 PREFILL 加 FINISH（共 5000，超过 4096 的 KV 池）都成功；Target 连续 20 次请求输出都正确，未见泄漏。

比较输出时，baseline 要用相同的注意力后端和相同的 tp。不同后端、不同 tp 的 Target-only 文本本身就可能不同。

### NPU 正确性（进行中）

配置是 tp=2、page 128、Target Qwen3-VL-2B、假 Draft，Target-only 和 TollTree 使用同一套开关。日志确认 `speculative_eagle_topk=None`、`npu_target_tree_fia=True`、实现是 `tree_paged_fia`、`fallback_reason` 为空，没有 paged metadata 报错和 Traceback。

| 模式 | 相对同开关 Target-only | 计数 |
|---|---|---|
| silent | 64 个 token 全部相同 | verify 0，回退 63 |
| gold | **第 12 个 token 不同**（gold `374`，baseline `594`） | 16 轮，直方图 `[0,0,0,16]`，回退 0 |

- **gold 的差异还没有解释，不能算通过。** 同开关 Target-only 和旧的 tp=1 输出也从第 12 个 token 分叉，所以这里可能是近似平局。还要确认假 Draft 读的是哪个文件，并测量这个位置的 top-2 差距。
- **accept 48 和日志 `accepted_draft_tokens=47` 差 1 不是错误。** 输出上限截断了最后一轮的 1 个 token，请求上的直方图按原始行计数，Worker 按实际追加的数量计数。
- **这一轮还不能证明接受逻辑和树 mask 正确。** gold 是单链，要靠 wrong 确认 Target 在比较 token，靠 branch 确认兄弟节点互不可见。

## 已知缺口和未验证

**NPU（T2 剩余）：**
- wrong、pad、branch
- 真 CUDA Draft 加 NPU Target，并与 OPT 逐轮对照
- page 128 的资源归还、tp=1、部署用的 32B Target
- `0c6da7e` 之后的 CUDA 回归

**接入 `policy=toll` 之前要处理：**
- `verify_tree_topk` 读 `toll_tree_fixed_width`，`toll` 下该值为空，NPU 会退回线性注意力。
- `_toll_root_topk`、`_toll_layer_forward`、回包的 `verify_bucket` 和 Target 的验证宽度都按固定宽度处理；padding 行没有切掉，`_admit_children` 会抛错。
- `continuous` 写死为 True，请求首轮和空闲间隙会进入 λ 和 Target 成本样本。
- Draft 层耗时在 `.cpu()` 之前停表，可能漏掉 GPU 异步时间，也包含自检耗时。
- `[TOLL round]` 只进内存，没有打日志；预测值和实际值错开一轮。
- `decide_layer` 读 `φ` 用的是“前沿深度 − 1”，`_observe_phi` 写入时用的是父节点深度，线上两边差一层，需要确认后修正。

**NPU graph：** `npu_graph_runner.py` 的 `_is_tree_verify_batch` 直接读 `speculative_eagle_topk`，接入 graph 时要一起处理。

**模型本身未验证：**
- b>1 的成本形状已由 OPT 的 gate 实验部分测出（见前文"为什么停止 TollTree"），但成本系数还没有按路径拟合；RTT 和带宽的影响都没有实测。
- 自适应相比最优固定配置的收益，T0 测得 V(τ)=0.36%、V(τ,N)=3.57%，这是停止开发的依据。最优 τ 是否跟随 `λ·C_L/b` 没有单独检验。
- λ 是当前策略测出来的，不动点迭代的收敛性没有证明；单步前瞻也不保证全局最优。
- `κ·ℓ` 会让长前缀请求更早退出，模型没有处理公平性。
