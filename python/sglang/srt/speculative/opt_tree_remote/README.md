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

先启动 Target，等服务就绪后再启动 Draft。HTTP 请求只发到 Target 端口。两端的 N、τ 和 `max_depth` 必须相同。`--opt-tree-width` 是树宽 N，`--opt-tree-max-depth` 含种子层。未写 `--speculative-num-draft-tokens` 时，Target 填 N + 1（验证 token 数，含已提交尾 token），Draft 填 N（每层前向宽度）。手写该参数必须等于本角色的期望值。未写 `--speculative-num-steps` 时它等于 `max_depth`，手写时也必须等于 `max_depth`。线上 `spec_shape` 是 `(N, max_depth, N)`，与角色无关。不要开启 RPD。CUDA 的 `--page-size` 是 1，NPU 是 128。

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

page=1 时，Draft 在把验证窗口回包之后补算最后一层的 KV，下一轮 ingest 只缺 bonus 那一个 token。同一段空闲里还会为每个验证节点和根预测一个不含验证子节点的 bonus；下一轮只有这个 token 确实是 bonus 时才装上它的 slot，并用暂存 logits 当种子，ingest 不再前向。`SGLANG_OPT_TREE_KV_REUSE=0` 关掉这两次补算。page=128 不补算，也不发布 tree lease，行为与补算引入之前相同。

Bonus 的 mask、positions、tokens 在 CPU 上拼好后一次拷上设备，真实行的前缀按整段赋值。槽长用缓存在设备上的 buffer `fill_`。前向走宽度 `N+1` 的模板和 `model_runner.forward`；没有模板时退回 `run_batch`，并计 `opt_bonus_template_fallbacks`。`model_runner.forward` 返回的 `can_run_graph` 为 False 时这一组已经走 eager，计入 `opt_bonus_eager_groups`，不再第二次前向。一个请求的全部分组共用一次模板准入，slot id 在最后一组返回后一次拷回主机。

Draft 的 `[SR Draft round]` 里，`host_mean_ms["total"]` 和 `busy_ms` 覆盖整个计量轮：从进入 `round()` 到回包之后的最后一层补算，以及延后的 `drain_opt_layer_gpu_us` 结束。`total` 含回包之后的工作。`counters["opt_pre_reply_us"]` 是本步 `round_start` 到回复路径结束（`send_batch` 返回，或本 rank 不发送）的墙钟，单位微秒，按窗口累加。和旧日志里回包前的 total 对齐时，用 `opt_pre_reply_us / rounds / 1000` 得到毫秒均值。`counters["opt_post_reply_us"]` 只覆盖 `forward_deferred_final_layers`。`total` 减去回包前时间，还可能包含延后的事件读取同步，不要把这两个差当成同一个数。

`SGLANG_OPT_DRAFT_LOOKAHEAD=1` 让 CUDA Draft 在设备上扩树，并在 host 看到这一层之前把下一层前向提交出去。默认关闭。只在 page size 为 1、batch size 为 1、attention backend 为 triton、并且该层能命中已捕获的 CUDA graph 时生效。图没有命中，或者空闲 slot 不够整棵树加一组 bonus，就退回原来的按层等待，并计 `opt_lookahead_fallbacks`。NPU 的 page size 是 128，没有回包后的补算；停层时提前提交的那一层只能回滚，所以不启用。batch size 大于 1 仍走原来的 `grow_active_layers`。CPU 树只按设备拷回的 parent、token 和 score 重建，不再自己做一次 top-k。

`SGLANG_OPT_PROBE_LAYER_WAIT=1` 在每一层 `_run` 之前插入一次 stream sync，把上一层等待拆进 `opt_lookahead_launch_wait_us`。默认关闭。打开后测到的不是线上路径。默认的 `opt_layer_launch_us` 是整段 `_run` 的墙钟，包含其中隐含的等待。`opt_layer_gpu_us` 含 `_run` 期间流上空闲，GPU 忙闲用 nsys 看。

lookahead 在 admission 时保存一份 `ForwardBatch`，之后每层浅拷贝并直接 `model_runner.forward`，不再每层调用 `init_new`，也不跑 sampler 的 argmax。模板不可用时这一层退回 `run_batch`，并计 `opt_lookahead_template_fallbacks`。`model_runner.forward` 返回的 `can_run_graph` 为 False 时，这一层已经走 eager，计入 `opt_lookahead_eager_layers`。

admission 在 ingest 入队之前完成。这时 GPU 是空的，`prefix_len` 已经是 committed 长度。失败则不开 lookahead，退回原来的按层扩树，并计 `opt_lookahead_fallbacks`。这次检查的墙钟是 `opt_lookahead_admit_us`。
