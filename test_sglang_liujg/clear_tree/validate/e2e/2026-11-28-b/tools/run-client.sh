#!/usr/bin/env bash
set -euo pipefail
trial=${1:?trial}; dataset=${2:?dataset}; conc=${3:?concurrency}; data=${4:?data-file}; img=${5:?img-dir}; out=${6:?output directory}
: "${CT_HTTP_ADDR:?}" "${CT_HTTP_PORT:?}" "${CT_CLIENT_DIR:?}"
case "$trial:$dataset:$conc" in
  01:rsteller:1|02:rsteller:4|03:rsteller:1|04:rsteller:4) ;;
  *) exit 2 ;;
esac
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$CT_CLIENT_DIR"
exec python3 -B eval_spectre_rsteller_serving.py --host "$CT_HTTP_ADDR" --port "$CT_HTTP_PORT" \
  --data-file "$data" --img-dir "$img" --max-items 100 --max-new-tokens 1024 \
  --max-concurrency "$conc" --temperature 0 --request-rate inf --timeout 600 --flush-cache --skip-metrics \
  --output-jsonl "$out/client.jsonl" --output-csv "$out/client.csv"
