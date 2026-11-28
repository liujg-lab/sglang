#!/usr/bin/env bash
set -euo pipefail
role=${1:?role}; mode=${2:?mode}; action=${3:?parse or serve}
: "${CT_ROOT:?}" "${CT_MODEL:?}" "${CT_DEVICE:?}" "${CT_RPC_ADDR:?}" "${CT_RPC_PORT:?}" "${CT_HTTP_ADDR:?}" "${CT_HTTP_PORT:?}" "${CT_PARSER:?}"
case "$mode" in greedy|rpd) ;; *) exit 2 ;; esac
for key in ${!SGLANG_CLEAR_@} ${!SGLANG_NPU_TREE_GRAPH_@} ${!SGLANG_FORK_@}; do unset "$key"; done
export SGLANG_CLEAR_LEDGER=1 SGLANG_CLEAR_TREE_KV_REUSE=1 SGLANG_CLEAR_BONUS_PREDICTION=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CT_ROOT/python:${PYTHONPATH:-}"
args=(--model-path "$CT_MODEL" --speculative-algorithm CLEAR_TREE_REMOTE --clear-tree-role "$role"
  --clear-tree-width 8 --clear-tree-tau 0.5 --clear-tree-max-depth 8 --speculative-num-steps 8
  --clear-tree-addr "$CT_RPC_ADDR" --clear-tree-port "$CT_RPC_PORT" --host "$CT_HTTP_ADDR" --port "$CT_HTTP_PORT"
  --max-running-requests 4 --cuda-graph-bs 1 2 4)
case "$role" in
  draft)
    export CUDA_VISIBLE_DEVICES="$CT_DEVICE" SGLANG_CLEAR_DRAFT_LOOKAHEAD=1 SGLANG_CLEAR_TARGET_BACKEND=npu SGLANG_CLEAR_ACCEPT_HALF_LIFE=1000
    args+=(--speculative-num-draft-tokens 64 --tp 1 --attention-backend triton --mm-attention-backend triton_attn --page-size 1 --mem-fraction-static 0.6)
    ;;
  target)
    export ASCEND_RT_VISIBLE_DEVICES="$CT_DEVICE" SGLANG_CLEAR_DRAFT_LOOKAHEAD=0 SGLANG_CLEAR_AUTO_PREFIT=0
    args+=(--speculative-num-draft-tokens 65 --tp 2 --attention-backend ascend --mm-attention-backend ascend_attn --page-size 128 --mem-fraction-static 0.8 --speculative-verify-mode "$mode")
    if [[ $mode == rpd ]]; then args+=(--speculative-rpd-tau 0.4); fi
    ;;
  *) exit 2 ;;
esac
cd "$CT_ROOT"
case "$action" in
  parse) exec python3 "$CT_PARSER" --role "$role" -- "${args[@]}" ;;
  serve) exec python3 -m sglang.launch_server "${args[@]}" ;;
  *) exit 2 ;;
esac
