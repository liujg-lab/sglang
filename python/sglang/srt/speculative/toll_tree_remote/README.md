# TOLL_TREE_REMOTE

`TOLL_TREE_REMOTE` 是独立的 `--speculative-algorithm` 名称。它不是 `OPT_TREE_REMOTE`，也不是 `STANDALONE_REMOTE`。

服务循环尚未实现。选择该算法启动服务时会拒绝。这一阶段只有 CPU 策略和树状态。

## 策略

λ 按 batch size 分桶，是提交 token 数的均值除以这一步间隔的均值；全追踪器共用一个时钟，间隔记到这次 `on_step` 的 b 上。层费是 `λ·w + λ·ε⁺·(活跃请求数 · 宽度桶 − M̂)⁺`。节点只付 `ε`。请求在 `s ≤ 0` 时退出，不再进入后面的层，已有节点仍参与选 K。整轮在正盈余盖不住层费、到达 `max_depth` 或 KV 不够时停止。验证数 `K` 必须落在验证桶上。`b` 是本轮请求数，含已退出的请求。

`policy/` 只依赖 numpy。`fixed` 用固定宽度和阈值复刻 OPT 的选层，不 import OPT。测试可以 import OPT 做逐位对照。

## 检查

```bash
PYTHONPATH=python python3.10 test/registered/unit/spec/test_toll_tree_policy.py -q
PYTHONPATH=python python3.10 test/registered/unit/spec/test_toll_tree_remote.py -q
```

CUDA 与 NPU 的服务差异留到接入调度之后。Draft 的 NPU 路径现在会在配置里拒绝。
