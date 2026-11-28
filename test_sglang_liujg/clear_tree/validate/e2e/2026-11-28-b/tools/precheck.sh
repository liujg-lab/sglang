#!/usr/bin/env bash
set -euo pipefail
role=${1:?role}; tunnel=${2:?existing tunnel pid}; identity=${3:?identity script}
[[ $(basename "$(readlink "/proc/$tunnel/exe")") == ssh ]]
python3 -B "$identity" "$tunnel" | awk '$1 != "state" && $1 != "ppid"'
LC_ALL=C ps -p "$tunnel" -o lstart= | awk '{$1=$1; print "started",$0}'
sha256sum "/proc/$tunnel/cmdline" | awk '{print "cmdline_sha256",$1}'
case "$role" in
  draft)
    [[ -z $(ss -H -ltn 'sport = :30001 or sport = :30019') ]]
    nvidia-smi -i 0 --query-gpu=index,utilization.gpu --format=csv,noheader |
      awk -F, '$1==0 && $2+0==0 {print "gpu",$1,"utilization",$2; found=1} END {exit !found}'
    nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | awk 'NF {print; busy=1} END {exit busy}'
    ;;
  target)
    [[ -z $(ss -H -ltn 'sport = :30000') ]]
    ss -H -ltnp 'sport = :30019' | awk -v owner="pid=$tunnel," 'index($0,owner) {n++} END {exit n!=1 || NR!=1}'
    for chip in 0 1; do
      npu-smi info -t proc-mem -i 1 -c "$chip" | awk '/No process in device/ {found=1} END {exit !found}'
      echo "selected_npu_chip $chip no_compute_process"
    done
    ;;
  *) exit 2 ;;
esac
ps -eo pid=,sid=,stat=,comm=,args= | awk '$3 !~ /^Z/ && ($4 ~ /^sglang::/ || ($4 ~ /python/ && $0 ~ /[s]glang[.]launch_server/)) {print "live_service_pid",$1,"sid",$2; busy=1} END {exit busy}'
echo "${role}_ports_and_service_precheck OK"
