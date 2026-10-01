#!/bin/bash
# The scale point, spread over two cards because neither has room for a 32B in
# bf16 beside the other jobs on this machine. Same scaffold, same items, same
# clean control as the 7B and 14B, so the three are a sweep and not three
# unrelated runs.
set -u
cd "$(dirname "$(readlink -f "$0")")"
P=/opt/conda/bin/python
mkdir -p build/dumps
for arm in withheld_aware question; do
  $P agentic_probe.py --items build/bixbench_numeric_q.jsonl \
     --clean build/bixbench_numeric_clean.jsonl --model Qwen/Qwen2.5-32B-Instruct \
     --shard 0,2 --shard-cap 34GiB --condition "$arm" --draws 3 --batch-size 4 \
     --dump build/dumps/qwen32b_${arm}.jsonl \
     --out agentic_bixbench_qwen32b.json >> logs/a6_qwen32b.log 2>&1
done
echo "done qwen32b"
