#!/usr/bin/env bash
set -euo pipefail
: "${CT_WORK:?}" "${CT_REMOTE:?}" "${CT_JUMP:?}" "${CT_DRAFT_CONTAINER:?}" "${CT_TARGET_CONTAINER:?}"
tool=$(cd -- "$(dirname -- "$0")" && pwd)
opt=(-o StrictHostKeyChecking=yes -o ConnectTimeout=10 -J "$CT_JUMP")
bash -n "$tool/server.sh" "$tool/run-client.sh"
scp "${opt[@]}" "$tool/server.sh" "$tool/run-client.sh" "$CT_REMOTE:$CT_WORK/"
for name in server.sh run-client.sh; do
  ssh -n "${opt[@]}" "$CT_REMOTE" "docker cp '$CT_WORK/$name' '$CT_TARGET_CONTAINER:$CT_WORK/'"
done
docker cp "$tool/server.sh" "$CT_DRAFT_CONTAINER:$CT_WORK/"
