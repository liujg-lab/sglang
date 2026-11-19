#!/bin/bash
set -e
for key in ${!SGLANG_CLEAR_@} ${!SGLANG_NPU_TREE_GRAPH_@} ${!SGLANG_FORK_@}; do unset "$key"; done
export SGLANG_CLEAR_LEDGER=1 SGLANG_CLEAR_TREE_KV_REUSE=1 SGLANG_CLEAR_DRAFT_LOOKAHEAD=0
export SGLANG_CLEAR_BONUS_PREDICTION=1 SGLANG_CLEAR_TARGET_BACKEND=npu
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/liujg/workspace/sglang/python:${PYTHONPATH:-}
cd /home/liujg/workspace/sglang
exec python -m sglang.launch_server \
  --model-path /home_18T/liujg/Hugging_Face/mllmTeam/Qwen3-VL-2B-Instruct-W8A8-INT8 \
  --clear-tree-role draft \
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
  --speculative-num-draft-tokens 64 \
  --tp 1 \
  --attention-backend triton \
  --mm-attention-backend triton_attn \
  --page-size 1 \
  --mem-fraction-static 0.6 \
  --port 30001
