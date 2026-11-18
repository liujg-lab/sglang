#!/bin/bash
set -euo pipefail
trial=${1:?trial}
conc=${2:?concurrency}
root=/tmp/clear-bonus-20261118c
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${root}/client:${PYTHONPATH:-}"
mkdir -p "${root}/${trial}"
cd "${root}/client"
exec python3 -B eval_spectre_text_serving.py \
  --host 127.0.0.1 \
  --port 30000 \
  --data-file /mnt/user/liujg/Downloads/text_only.jsonl \
  --max-items 100 \
  --max-new-tokens 1024 \
  --max-concurrency "${conc}" \
  --temperature 0 \
  --request-rate inf \
  --timeout 600 \
  --flush-cache \
  --skip-metrics \
  --output-jsonl "${root}/${trial}/client.jsonl" \
  --output-csv "${root}/${trial}/client.csv"
