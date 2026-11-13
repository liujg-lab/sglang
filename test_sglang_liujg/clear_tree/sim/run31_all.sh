#!/bin/bash
# Round 31, all parts in sequence on two processes. usage (conda sglang env): bash run31_all.sh
for part in A C D Z T B; do
  tag=$(echo $part | tr 'A-Z' 'a-z')
  python run31.py 0 $part > results/r31$tag.0.jsonl 2> results/r31$tag.0.err &
  python run31.py 1 $part > results/r31$tag.1.jsonl 2> results/r31$tag.1.err &
  wait
done
