#!/usr/bin/env bash
set -euo pipefail
role=${1:?role}; tunnel=${2:?tunnel pid}; identity=${3:?identity script}
# Record a hash of the complete tunnel argv, not the connection details.
python3 "$identity" "$tunnel" | awk '$1 != "state" && $1 != "ppid"'
readlink "/proc/$tunnel/exe" | xargs basename | awk '$0 == "ssh" {found=1} END {exit !found}'
sha256sum "/proc/$tunnel/cmdline" | awk '{print "cmdline_sha256",$1}'
if [[ $role == draft ]]; then
  listeners=$(ss -H -ltn 'sport = :30001 or sport = :30019')
  [[ -z $listeners ]] || exit 4
  nvidia-smi -i 0 --query-gpu=index,utilization.gpu --format=csv,noheader | awk -F, '$1==0 && $2+0==0 {print "gpu",$1,"utilization",$2; found=1} END {exit !found}'
  nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | awk 'NF {print; busy=1} END {exit busy}'
elif [[ $role == target ]]; then
  listeners=$(ss -H -ltn 'sport = :30000')
  [[ -z $listeners ]] || exit 4
  ss -H -ltnp 'sport = :30019' | awk -v owner="pid=$tunnel," 'index($0,owner) {n++} END {exit n!=1 || NR!=1}'
  for chip in 0 1; do
    npu-smi info -t proc-mem -i 1 -c "$chip" | awk '/No process in device/ {found=1} END {exit !found}'
    echo "selected_npu_chip $chip no_compute_process"
  done
fi
ps -eo pid=,sid=,stat=,comm=,args= | awk '$3 !~ /^Z/ && $4 !~ /^(awk|bash|sh|sshd|ssh|timeout|docker)$/ && ($4 ~ /^sglang::/ || $0 ~ /[s]glang\.launch_server/) {print "live_service_pid",$1,"sid",$2; busy=1} END {exit busy}'
echo "${role}_ports_and_service_precheck OK"
