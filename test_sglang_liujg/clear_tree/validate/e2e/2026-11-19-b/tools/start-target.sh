#!/bin/bash
set -e
for key in ${!SGLANG_CLEAR_@} ${!SGLANG_NPU_TREE_GRAPH_@} ${!SGLANG_FORK_@}; do unset "$key"; done
export SGLANG_CLEAR_LEDGER=1 SGLANG_CLEAR_TREE_KV_REUSE=1 SGLANG_CLEAR_DRAFT_LOOKAHEAD=0
export ASCEND_RT_VISIBLE_DEVICES=2,3 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/mnt/user/liujg/sglang/python:${PYTHONPATH:-}
cd /mnt/user/liujg/sglang
exec python -m sglang.launch_server \
  --model-path /mnt/user/liujg/Downloads/Qwen3-VL-32B-Instruct \
  --clear-tree-role target \
  --speculative-algorithm CLEAR_TREE_REMOTE \
  --clear-tree-width 8 \
  --clear-tree-tau 0.5 \
  --clear-tree-max-depth 8 \
  --speculative-num-steps 8 \
  --clear-tree-addr localhost \
  --clear-tree-port 30019 \
  --max-running-requests 4 \
  --cuda-graph-bs 1 2 4 \
  --host 127.0.0.1 \
  --speculative-num-draft-tokens 65 \
  --tp 2 \
  --attention-backend ascend \
  --mm-attention-backend ascend_attn \
  --page-size 128 \
  --mem-fraction-static 0.8 \
  --port 30000
