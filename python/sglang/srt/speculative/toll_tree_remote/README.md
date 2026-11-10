# TOLL_TREE_REMOTE

`TOLL_TREE_REMOTE` 是独立的 `--speculative-algorithm` 名称。它不是 `OPT_TREE_REMOTE`，也不是 `STANDALONE_REMOTE`。Target 和 Draft 是两个进程，可以运行在不同机器上。

当前服务循环只接受同时满足这些条件的启动：`--disable-cuda-graph`、`--disable-piecewise-cuda-graph`、`--toll-tree-policy=fixed`、`--max-running-requests=1`，以及 `--toll-tree-role=target` 或 `draft`。`policy=toll`、更大的 batch、开启 CUDA graph 仍然会拒绝。这一步没有 CUDA graph，也没有 lease。

带 `multimodal_inputs` 的请求不进入投机。Qwen3-VL 的启动 warmup 会带图片，而当前 PREFILL 只发送 `input_ids`，所以这类请求的 PREFILL 和 STEP 都走普通解码。

Draft 按层生成 top-C，回包后回滚树区 KV。Target 用 `K+1` 个槽做验证，槽 0 是已提交尾 token，padding 填 0。`commit_len` 是前缀加已输出 token 的长度，STEP 只带增量；对不上时 Draft 回 `NEED_SNAPSHOT`，Target 下一轮重发完整前缀。`K=0` 或空回包退回普通一步解码，超时和 `REJECT` 才记熔断。

**2026-11-07 起停止开发。** 这个目录保留 `fixed` 路径和 CPU 策略代码，不再做 T3 以后的阶段。只调 τ 的上界收益是 0.36%，调 τ 和 N 是 3.57%。下一份算法是全局价格的树控制，计划作为默认关闭的开关放进 `OPT_TREE_REMOTE`，见 [../opt_tree_remote/PRICE.md](../opt_tree_remote/PRICE.md)。

停止前的设计和缺口记在 [DESIGN.md](DESIGN.md)。窗口、4096 分叉和 V0–V6 的结论在 `test_result/raw/gate-window/DECISION.md` 与 `test_result/validate/DECISION.md`，这两份记录留在本机，不进仓库。

## 策略

λ 按 batch size 分桶，是提交 token 数的均值除以这一步间隔的均值；全追踪器共用一个时钟，间隔记到这次 `on_step` 的 b 上。层费是 `λ·w + λ·ε⁺·(活跃请求数 · 宽度桶 − M̂)⁺`。决策使用 tok/ms。`policy=fixed` 的验证集与 OPT 一样，`K=N`。

## 检查

```bash
PYTHONPATH=python python3.10 test/registered/unit/spec/test_toll_tree_policy.py -q
PYTHONPATH=python python3.10 test/registered/unit/spec/test_toll_tree_round.py -q
PYTHONPATH=python python3.10 test/registered/unit/spec/test_toll_tree_remote.py -q
```

实机正确性还需要 `tp=1`、短请求、与同 `N` 和 `τ` 且关闭 lookahead 的 OPT 对照。耗时没有参考价值。
