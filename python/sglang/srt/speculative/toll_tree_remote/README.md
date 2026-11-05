# TOLL_TREE_REMOTE

`TOLL_TREE_REMOTE` 是独立的 `--speculative-algorithm` 名称。它不是 `OPT_TREE_REMOTE`，也不是 `STANDALONE_REMOTE`。

服务循环尚未实现。选择该算法启动服务时会拒绝。这一阶段只有 CPU 策略和树状态。

## 策略

λ 按 batch size 分桶，是提交 token 数的均值除以相邻 STEP 时间的均值。层费 `w` 由同层请求分摊一次。节点只付 `ε`。请求在 `s ≤ 0` 时退出，并且不再进入后面的层。整轮在 `Σ s⁺ ≤ λ·w`、到达 `max_depth` 或 KV 不够时停止。验证数 `K` 必须落在验证桶上。

`policy/` 只依赖 numpy。`fixed` 用固定宽度和阈值复刻 OPT 的选层，不 import OPT。测试可以 import OPT 做逐位对照。

## 检查

```bash
PYTHONPATH=python python3.10 test/registered/unit/spec/test_toll_tree_policy.py -q
PYTHONPATH=python python3.10 test/registered/unit/spec/test_toll_tree_remote.py -q
```

CUDA 与 NPU 的服务差异留到接入调度之后。Draft 的 NPU 路径现在会在配置里拒绝。
