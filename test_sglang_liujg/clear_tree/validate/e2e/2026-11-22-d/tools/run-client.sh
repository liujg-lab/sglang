#!/bin/bash
set -euo pipefail
trial=${1:?trial}
script=${2:?script}
conc=${3:?concurrency}
data=${4:?data-file}
img=${5:?img-dir}
root=/tmp/clear-20261122d
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${root}/client:${PYTHONPATH:-}"
mkdir -p "${root}/${trial}"
cd "${root}/client"
exec python3 -B "${script}" \
  --host 127.0.0.1 \
  --port 30000 \
  --data-file "${data}" \
  --img-dir "${img}" \
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
