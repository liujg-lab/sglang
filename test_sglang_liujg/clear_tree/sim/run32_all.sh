#!/bin/bash
# Round 32 parts in sequence on two processes. usage (conda sglang env): bash run32_all.sh P K S
for part in "$@"; do
  tag=$(echo $part | tr "A-Z" "a-z")
  python run32.py 0 $part > results/r32$tag.0.jsonl 2> results/r32$tag.0.err &
  python run32.py 1 $part > results/r32$tag.1.jsonl 2> results/r32$tag.1.err &
  wait
done
