#!/usr/bin/env bash
set -euo pipefail
: "${CT_WORK:?}" "${CT_REMOTE:?}" "${CT_JUMP:?}"
: "${CT_DRAFT_CONTAINER:?}" "${CT_DRAFT_ROOT:?}" "${CT_DRAFT_TUNNEL:?}"
: "${CT_TARGET_CONTAINER:?}" "${CT_TARGET_ROOT:?}" "${CT_TARGET_TUNNEL:?}"
tool=$(cd -- "$(dirname -- "$0")" && pwd)
raw="$tool/../raw/sync"
skill="$CT_DRAFT_ROOT/.cursor/skills/clear-tree-e2e-validation/scripts"
pkg=python/sglang/srt/speculative/clear_tree_remote
local_pkg="$CT_DRAFT_ROOT/$pkg"
remote_pkg="$CT_TARGET_ROOT/$pkg"
opt=(-o StrictHostKeyChecking=yes -o ConnectTimeout=10 -J "$CT_JUMP")

remote() {
  local encoded shellcmd
  printf -v shellcmd '%q ' "$@"
  encoded=$(printf '%s' "$shellcmd" | base64 -w0)
  ssh -n "${opt[@]}" "$CT_REMOTE" "printf %s $encoded | base64 -d | bash"
}

remote_check() {
  remote bash "$CT_WORK/precheck.sh" target "$CT_TARGET_TUNNEL" "$CT_WORK/process_identity.py" |
    tr -d '\r' | diff - "$raw/precheck-target.txt"
}

draft_check() {
  bash "$tool/precheck.sh" draft "$CT_DRAFT_TUNNEL" "$skill/process_identity.py" | diff - "$raw/precheck-draft.txt"
}

check_archive() {
  local hashlist
  hashlist=$(awk '$1 != "ABSENT" {print}' "$1")
  if [[ -n $hashlist ]]; then (cd "$2"; printf '%s\n' "$hashlist" | sha256sum -c -); fi
}

draft_check
remote_check
remote docker exec "$CT_TARGET_CONTAINER" tar -czf "$CT_WORK/npu-package-before.tar.gz" -C "$remote_pkg" .
remote docker cp "$CT_TARGET_CONTAINER:$CT_WORK/npu-package-before.tar.gz" "$CT_WORK/"
remote sha256sum "$CT_WORK/npu-package-before.tar.gz" | tr -d '\r' | tee "$raw/npu-package-before.remote-sha256" >/dev/null
scp "${opt[@]}" "$CT_REMOTE:$CT_WORK/npu-package-before.tar.gz" "$raw/"
read -r npu_hash _ < "$raw/npu-package-before.remote-sha256"
printf '%s  %s\n' "$npu_hash" "$raw/npu-package-before.tar.gz" | sha256sum -c -

scp "${opt[@]}" "$raw/clear_tree_remote.tar.gz" "$CT_REMOTE:$CT_WORK/"
remote docker cp "$CT_WORK/clear_tree_remote.tar.gz" "$CT_TARGET_CONTAINER:$CT_WORK/"
remote_check
remote docker exec "$CT_TARGET_CONTAINER" bash "$CT_WORK/archive-fits.sh" "$remote_pkg" "$CT_WORK/npu-old-fit" |
  tr -d '\r' | tee "$raw/npu-old-fit.sha256"
remote docker cp "$CT_TARGET_CONTAINER:$CT_WORK/npu-old-fit" "$CT_WORK/"
scp -r "${opt[@]}" "$CT_REMOTE:$CT_WORK/npu-old-fit" "$raw/"
check_archive "$raw/npu-old-fit.sha256" "$raw/npu-old-fit"

draft_check
docker exec "$CT_DRAFT_CONTAINER" bash "$tool/archive-fits.sh" "$local_pkg" "$raw/draft-old-fit" |
  tee "$raw/draft-old-fit.sha256"
check_archive "$raw/draft-old-fit.sha256" "$raw/draft-old-fit"
docker exec "$CT_DRAFT_CONTAINER" python3 -B "$skill/package_manifest.py" --root "$local_pkg" |
  tee "$raw/draft.manifest" >/dev/null
diff -u "$raw/head.manifest" "$raw/draft.manifest"

remote_check
read -r pkg_hash _ < <(sha256sum "$raw/clear_tree_remote.tar.gz")
remote docker exec "$CT_TARGET_CONTAINER" bash -c '
  set -euo pipefail
  pkg=$1; work=$2; hash=$3
  printf "%s  %s\n" "$hash" "$work/clear_tree_remote.tar.gz" | sha256sum -c --status
  stage="$pkg.${work##*/}-new"
  old="$work/npu-package-before-unpacked"
  test ! -e "$old"
  mkdir "$stage"
  tar -xzf "$work/clear_tree_remote.tar.gz" -C "$stage"
  chown -R 1008:1008 "$stage"
  mv -T "$pkg" "$old"
  if ! mv -T "$stage" "$pkg"; then mv -T "$old" "$pkg"; exit 1; fi
' bash "$remote_pkg" "$CT_WORK" "$pkg_hash"
remote docker exec "$CT_TARGET_CONTAINER" python3 -B "$CT_WORK/package_manifest.py" --root "$remote_pkg" |
  tr -d '\r' | tee "$raw/npu.manifest" >/dev/null
diff -u "$raw/head.manifest" "$raw/npu.manifest"
echo 'Both live ClearTree package manifests match fixed HEAD'
