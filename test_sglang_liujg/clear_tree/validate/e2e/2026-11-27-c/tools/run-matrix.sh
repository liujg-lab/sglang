#!/usr/bin/env bash
set -euo pipefail
: "${CT_WORK:?}" "${CT_REMOTE:?}" "${CT_JUMP:?}" "${CT_TARGET_CONTAINER:?}"
: "${CT_RSTELLER_DATA:?}" "${CT_RSTELLER_IMG:?}"
mode=${1:?greedy or rpd}
case "$mode" in greedy|rpd) ;; *) exit 2 ;; esac
tool=$(cd -- "$(dirname -- "$0")" && pwd)
sync="$tool/../raw/sync"
if [[ $mode == rpd ]]; then
  # Recheck the ordered inputs before the second start without sending requests.
  opt=(-o StrictHostKeyChecking=yes -o ConnectTimeout=10 -J "$CT_JUMP")
  printf -v cmd '%q ' docker exec "$CT_TARGET_CONTAINER" python3 -B "$CT_WORK/data_identity.py" "$CT_RSTELLER_DATA" "$CT_RSTELLER_IMG"
  encoded=$(printf '%s' "$cmd" | base64 -w0)
  ssh -n "${opt[@]}" "$CT_REMOTE" "printf %s $encoded | base64 -d | bash" | tr -d '\r' > "$sync/data-identity-rpd.json"
  diff -u "$sync/data-identity.json" "$sync/data-identity-rpd.json"
fi
exec python3 -u -B "$tool/drive.py" "$mode"
