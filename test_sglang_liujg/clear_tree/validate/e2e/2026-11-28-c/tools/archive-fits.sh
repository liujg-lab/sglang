#!/usr/bin/env bash
set -euo pipefail
pkg=${1:?live package}; dest=${2:?new archive directory}
cd "$pkg"
mkdir -m 755 "$dest"
shopt -s nullglob
files=(clear_tree_fit*.json clear_tree_fit*.json.tmp)
for n in clear_tree_fit.json clear_tree_fit.rpd.json clear_tree_fit.bak.json clear_tree_fit.bak2.json; do
  if [[ ! -e $n && ! -L $n ]]; then printf 'ABSENT %s\n' "$n"; fi
done
for n in "${files[@]}"; do [[ -f $n && ! -L $n ]]; done
for n in "${files[@]}"; do
  read -r expected _ < <(sha256sum -- "$n")
  mv -n -- "$n" "$dest/$n"
  [[ ! -e $n ]]
  printf '%s  %s\n' "$expected" "$dest/$n" | sha256sum -c --status
  printf '%s  %s\n' "$expected" "$n"
done
