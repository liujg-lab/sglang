# OPT-Tree Remote

原版 OPT-Tree 的 CPU 实现，行为对照 `opt_classic_qwen3/tree.py` 里的 `Tree`。一轮里的宽度 N 和阈值 τ 固定。

`OPT_TREE_REMOTE` 已经注册为 `--speculative-algorithm` 的一个选项。Target 和 Draft 是两个进程：Target 用 `build_tree_from_parents` 做验证，Draft 的每一层调用 `update`，不走 EAGLE 的 top-k 树。这个包不包含 AdaCost、代价模型或在线宽度控制器。

## 算法

1. 用根分布的 top-N 作为第 1 层。
2. 每一层把父节点的路径概率乘以该父节点 top-N 个孩子的概率，再取全局 top-N。概率是温度 1 的 fp32 全词表 softmax，不做 top-N 内的再归一化。
3. E 是已经加入的每一层节点里，top-N 路径概率之和。
4. 加完第 k 层后立刻判断。当 `E - E_prev > tau` 且 `depth < max_depth - 1` 时，前向第 k 层并展开下一层。否则停止。验证集就是这 k 层里的 N 个节点，不再为了丢掉下一层而多做一次草稿前向。
5. 同分时按路径概率降序、深度升序、列号升序。浅层父节点排在前面，因此选出的 N 个节点已经包含各自的祖先。

`max_depth` 与参考实现的 `Tree.max_depth` 相同，容量里仍留着那个参考实现会填上再丢掉的槽。这里在造出该层之前就停止，所以一条草稿链最长仍是 `max_depth - 1`。

## 注意力和父指针

默认草稿 mask 只让每个节点看到自己和祖先。`ancestor_mask(state, leak=True)` 才会复现参考实现里 `kv_cache_mask` 从不清零的结果。后续草稿前向应使用 `ancestor_mask`。

验证集里的 `parents[i] == -1` 表示已经提交的 tail token。其余父下标必须小于 `i`。`build_tree_from_parents` 把这个列表变成 FULL_MASK、positions，以及 `retrive_index` / `retrive_next_token` / `retrive_next_sibling`。根 token 占验证序列的第 0 位，不在 `draft_tokens` 里。

`positions` 是 1-based 层深。调用方再加前缀偏移。未停止时，这一层的每个节点位置相同；停止后，每个验证节点使用自己的深度。

## CPU 用法

`update` 接收的是概率，不是 logits。第一调用形状可以是 `[vocab]` 或 `[1, 1, vocab]`，之后是 `[N, vocab]` 或 `[1, N, vocab]`。

```python
import torch

from sglang.srt.speculative.opt_tree_remote import (
    OptTreeConfig,
    OptTreeState,
    build_tree_from_parents,
    topk_probs_from_logits,
    update,
)

config = OptTreeConfig(width=4, tau=0.5, max_depth=10)
state = OptTreeState.create(config)

# 根分布。logits 形状 [vocab]。
root_probs, _ = topk_probs_from_logits(root_logits, k=root_logits.shape[-1])
step = update(state, root_probs)

while not step.is_final:
    # step.tokens: [N]
    # step.positions: [N]，值为当前层的 1-based 深度
    # step.ancestor_mask: [N, depth * N]，1 表示可见
    next_logits = draft_forward(step.tokens, step.positions, step.ancestor_mask)
    flat = next_logits.reshape(config.width, -1)
    probs = torch.softmax(flat.float(), dim=-1)
    step = update(state, probs)

verify = step.verify
seq_lens = torch.tensor([prefix_len], dtype=torch.long)
layout = build_tree_from_parents(verify.parents, seq_lens)
```

`topk_probs_from_logits` 用 `exp(logit - logsumexp)` 计算全词表概率，再取 top-k。它和先 softmax 再 top-k 的下标一致。

## 检查

```bash
PYTHONPATH=python python3.10 test/registered/unit/spec/test_opt_tree_remote.py -q
```

## 启动命令

先启动 Target，等服务就绪后再启动 Draft。HTTP 请求只发到 Target 端口。两端的 N、τ 和 `max_depth` 必须相同。`--opt-tree-width` 是树宽 N，`--opt-tree-max-depth` 含种子层。未写 `--speculative-num-draft-tokens` 时，Target 填 N + 1（验证 token 数，含已提交尾 token），Draft 填 N（每层前向宽度）。手写该参数必须等于本角色的期望值。未写 `--speculative-num-steps` 时它等于 `max_depth`，手写时也必须等于 `max_depth`。线上 `spec_shape` 是 `(N, max_depth, N)`，与角色无关。默认验证是 greedy。Target 可以额外加 `--speculative-verify-mode rpd` 和 `--speculative-rpd-tau`。`--opt-tree-tau` 决定树长多深，`--speculative-rpd-tau` 决定验证放宽多少。Draft 不接受 RPD，建树规则不变。CUDA 的 `--page-size` 是 1，NPU 是 128。

Target：

```bash
python -m sglang.launch_server \
  --model-path <target-model> \
  --speculative-algorithm OPT_TREE_REMOTE \
  --opt-tree-role target \
  --opt-tree-width 4 \
  --opt-tree-tau 0.5 \
  --opt-tree-max-depth 10 \
  --opt-tree-addr <draft-host> \
  --opt-tree-port <rpc-port> \
  --attention-backend triton \
  --page-size 1 \
  --port <target-http-port>
```

Target，开启 RPD 时再加验证参数。Draft 命令不加这两项：

```bash
python -m sglang.launch_server \
  --model-path <target-model> \
  --speculative-algorithm OPT_TREE_REMOTE \
  --opt-tree-role target \
  --opt-tree-width 4 \
  --opt-tree-tau 0.5 \
  --opt-tree-max-depth 10 \
  --speculative-verify-mode rpd \
  --speculative-rpd-tau 0.2 \
  --opt-tree-addr <draft-host> \
  --opt-tree-port <rpc-port> \
  --attention-backend triton \
  --page-size 1 \
  --port <target-http-port>
```

Draft：

```bash
python -m sglang.launch_server \
  --model-path <draft-model> \
  --speculative-algorithm OPT_TREE_REMOTE \
  --opt-tree-role draft \
  --opt-tree-width 4 \
  --opt-tree-tau 0.5 \
  --opt-tree-max-depth 10 \
  --opt-tree-addr <draft-host> \
  --opt-tree-port <rpc-port> \
  --attention-backend triton \
  --page-size 1 \
  --port <draft-http-port>
```

CUDA 使用 `--page-size 1` 和 `triton`。NPU 把 `--page-size` 改为 128，并改用该后端对应的 attention backend。

## 级联注意力

Draft 的 TARGET_VERIFY 可以走 `opt_tree_attention.py`。前缀按 flash-decoding 切分，树列单独一块。只在 CUDA、`--page-size 1`、attention backend 为 triton、batch size 为 1，并且 `model_supports` 通过时生效。`model_supports` 还要求非 deterministic、非 MLA，且每一层是普通 decoder attention。NPU 和 page size 128 不走级联，仍用原来的 mask kernel。已经捕获了级联 kernel 的图在重放时 prepare 填不下就抛错，不能退回普通 metadata。层前向不传 prefix rows，spec mask 不完整，prepare 失败时 eager 同样抛错。bonus 只在 prefix 大于 0 时把 mask 的前缀列交给 prefix rows，prepare 返回 False 才退回原来的 mask kernel；prefix 为 0 的 bonus 也没有完整 mask，失败同样抛错。

page=1 时，Draft 在把验证窗口回包之后补算最后一层的 KV，下一轮 ingest 只缺 bonus 那一个 token。同一段空闲里还会为每个验证节点和根预测一个不含验证子节点的 bonus；下一轮只有这个 token 确实是 bonus 时才装上它的 slot，并用暂存 logits 当种子，ingest 不再前向。`SGLANG_OPT_TREE_KV_REUSE=0` 关掉这两次补算。page=128 不补算，也不发布 tree lease，行为与补算引入之前相同。

Bonus 的 mask、positions、tokens 在 CPU 上拼好后一次拷上设备，真实行的前缀按整段赋值。槽长用缓存在设备上的 buffer `fill_`。前向走宽度 `N+1` 的模板和 `model_runner.forward`；没有模板时退回 `run_batch`，并计 `opt_bonus_template_fallbacks`。`model_runner.forward` 返回的 `can_run_graph` 为 False 时这一组已经走 eager，计入 `opt_bonus_eager_groups`，不再第二次前向。一个请求的全部分组共用一次模板准入，slot id 在最后一组返回后一次拷回主机。

Draft 的 `[SR Draft round]` 里，`host_mean_ms["total"]` 和 `busy_ms` 覆盖整个计量轮：从进入 `round()` 到回包之后的最后一层补算，以及延后的 `drain_opt_layer_gpu_us` 结束。`total` 含回包之后的工作。`counters["opt_pre_reply_us"]` 是本步 `round_start` 到回复路径结束（`send_batch` 返回，或本 rank 不发送）的墙钟，单位微秒，按窗口累加。和旧日志里回包前的 total 对齐时，用 `opt_pre_reply_us / rounds / 1000` 得到毫秒均值。`counters["opt_post_reply_us"]` 只覆盖 `forward_deferred_final_layers`。`total` 减去回包前时间，还可能包含延后的事件读取同步，不要把这两个差当成同一个数。`grow_batch` 和未开 lookahead 的单请求扩树不设置延后排水。补算层的事件在同一步、`forward_deferred_final_layers` 返回后读取，同步留在本步回包之后。`opt_layer_forwards_deferred` 计 `forward_deferred_final_layers` 里每一次完成的 `_forward_verify_batch`。同一深度一次装下仍计 1；KV 预算把该深度拆成多次前向时按次数累加。整组失败不加。`grow_batch` 的 `opt_layer_forwards` 同样按 `_forward_verify_batch` 次数累加。lookahead 的 `launch_layer` 仍各计一次，写在 `opt_layer_forwards`。单层 `opt_layer_host_us`、`opt_layer_launch_us`、`opt_layer_gpu_us` 的分母是 `opt_layer_forwards + opt_layer_forwards_deferred`。`opt_layer_select_gpu_us` 是 lookahead 每次非终点扩层上 `_vocab_topk`、`expand_on_device` 和 `copy_expand` 的设备时间，单位微秒，按窗口累加。种子 top-k、停层 bonus top-k、`clone_cpu` 和 `launch_layer` 不在其中。平均值是 `opt_layer_select_gpu_us / opt_layer_forwards_critical`；分母为 0 时这一窗没有成功的 lookahead 扩层。回退丢掉已排队的选层事件。lookahead 提前提交的最后一层仍只在 `opt_layer_forwards`。Target 的 `opt_step_rtt_us` 是 STEP 发送返回到 `_sr_recv` 返回的墙钟，只在 OPT Target、`send_ok` 且回复非空时累加；超时和错包清掉起点但不计样本。同一验证轮在还没有草稿时会有两次 STEP，所以均值是 `opt_step_rtt_us / opt_step_rtt_samples`。和 Draft 的 `opt_pre_reply_us / rounds` 相减。postprocess 先于回包结束时，差接近双向线路和序列化；postprocess 更长时，差里还含这段等待。发送侧组包在起点之前。`opt_seed_d2h_wait_us` 不除。`opt_tree_depth_K` 在 `consume_opt_tree_lease` 通过对齐和路径检查之后计一次，发生在 `_install_accepted` 之前。`K` 是租约上的层数 `len(lease.layer_slots)`，不是接受长度。安装失败仍会计上，随后释放租约。`opt_accept_depth_K` 同一次成功检查计一次，`K` 是接受路径上的槽数 `len(kept)`，路径在根上就停时可以是 0。`opt_lease_dropped_misaligned` 和 `opt_lease_dropped_path` 不加这两个计数。`opt_bonus_hit_rank_R` 在 bonus 猜测安装成功时和 `opt_bonus_hits` 一起计。`R` 是 `min(end_rank, 3)`。`end_rank` 按父节点剩余路径质量从大到小排，0 最大；3 以及更大的名次都进 `opt_bonus_hit_rank_3`。没装上不计数。`opt_end_rank_R` 是同一次停止节点上猜测的名次，不论 token 是否命中、槽是否装上；没有该 parent 的猜测则不计。

`SGLANG_OPT_DRAFT_LOOKAHEAD=1` 让 CUDA Draft 在设备上扩树，并在 host 看到这一层之前把下一层前向提交出去。默认关闭。只在 page size 为 1、batch size 为 1、attention backend 为 triton、并且该层能命中已捕获的 CUDA graph 时生效。图没有命中，或者空闲 slot 不够整棵树加一组 bonus，就退回原来的按层等待，并计 `opt_lookahead_fallbacks`。NPU 的 page size 是 128，没有回包后的补算；停层时提前提交的那一层只能回滚，所以不启用。batch size 大于 1 仍走原来的 `grow_active_layers`。CPU 树只按设备拷回的 parent、token 和 score 重建，不再自己做一次 top-k。

`SGLANG_OPT_PROBE_LAYER_WAIT=1` 在每一层 `_run` 之前插入一次 stream sync，把上一层等待拆进 `opt_lookahead_launch_wait_us`。默认关闭。打开后测到的不是线上路径。默认的 `opt_layer_launch_us` 是整段 `_run` 的墙钟，包含其中隐含的等待。`opt_layer_gpu_us` 含 `_run` 期间流上空闲，GPU 忙闲用 nsys 看。

lookahead 在 admission 时保存一份 `ForwardBatch`，之后每层浅拷贝并直接 `model_runner.forward`，不再每层调用 `init_new`，也不跑 sampler 的 argmax。层模板和它的 M-RoPE delta 跟 bonus 模板一样留在 `_opt_la_batches[rid]`，直到 KV pool 下标变化或请求结束。命中后这一步不再调用 `init_new`。模板不可用时这一层退回 `run_batch`，并计 `opt_lookahead_template_fallbacks`。`model_runner.forward` 返回的 `can_run_graph` 为 False 时，这一层已经走 eager，计入 `opt_lookahead_eager_layers`。缓存命中之后，每一层是否走图由这个计数兜底，admission 不再重复 `can_run`。

admission 在 ingest 入队之前完成。这时 GPU 是空的，`prefix_len` 已经是 committed 长度。每步仍检查 KV 预算：`available_size` 是 free list 的长度，不会同步。空闲 slot 不够整棵树加一组 bonus，就不开 lookahead，已有的层模板留在缓存里。第一次没有层模板，或者图没有命中，也退回原来的按层扩树，并计 `opt_lookahead_fallbacks`。这次检查的墙钟是 `opt_lookahead_admit_us`。这份缓存只覆盖 batch size 为 1 的 lookahead。batch size 大于 1 仍走 `grow_batch` / `grow_active_layers`，不建层模板。
