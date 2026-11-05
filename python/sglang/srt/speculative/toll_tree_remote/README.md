# TOLL_TREE_REMOTE

`TOLL_TREE_REMOTE` 是独立的 `--speculative-algorithm` 名称。它不是 `OPT_TREE_REMOTE`，也不是 `STANDALONE_REMOTE`。

当前服务循环只接受同时满足这些条件的启动：`--disable-cuda-graph`、`--disable-piecewise-cuda-graph`、`--toll-tree-policy=fixed`、`--max-running-requests=1`，以及 `--toll-tree-role=target` 或 `draft`。`policy=toll`、更大的 batch、开启 CUDA graph 仍然会拒绝。这一步没有 CUDA graph，也没有 lease。NPU Target 不走 `tree_paged_fia`。

Draft 按层生成 top-C，回包后回滚树区 KV。Target 用 `K+1` 个槽做验证，padding 填 0。`K=0` 或空回包退回普通一步解码。

## 策略

λ 按 batch size 分桶，是提交 token 数的均值除以这一步间隔的均值；全追踪器共用一个时钟，间隔记到这次 `on_step` 的 b 上。层费是 `λ·w + λ·ε⁺·(活跃请求数 · 宽度桶 − M̂)⁺`。决策使用 tok/ms。`policy=fixed` 的验证集与 OPT 一样，`K=N`。

## 检查

```bash
PYTHONPATH=python python3.10 test/registered/unit/spec/test_toll_tree_policy.py -q
PYTHONPATH=python python3.10 test/registered/unit/spec/test_toll_tree_round.py -q
PYTHONPATH=python python3.10 test/registered/unit/spec/test_toll_tree_remote.py -q
```

实机正确性还需要 `tp=1`、短请求、与同 `N` 和 `τ` 且关闭 lookahead 的 OPT 对照。耗时没有参考价值。
