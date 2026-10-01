#!/bin/bash
# The conditions the paper quotes, re-run so every margin carries a difference
# interval: the headline no-data arm on both files, then the positive control.
set -u
cd "$(dirname "$0")"
MODEL="$1"; DEV="$2"; TAG="$3"
P=/opt/conda/bin/python
$P agentic_probe.py --items build/bixbench_numeric_q.jsonl \
   --clean build/bixbench_numeric_clean.jsonl --model "$MODEL" --device "$DEV" \
   --condition withheld_aware --draws 3 --batch-size 10 \
   --out agentic_bixbench_$TAG.json >> logs/a2_${TAG}.log 2>&1
$P agentic_probe.py --items build/mmlu_pro_matched_released.jsonl \
   --clean build/mmlu_pro_matched_clean.jsonl --model "$MODEL" --device "$DEV" \
   --condition withheld_aware --draws 2 --batch-size 10 \
   --out agentic_mmlupro_$TAG.json >> logs/a2_${TAG}.log 2>&1
$P agentic_probe.py --items build/bixbench_numeric_q.jsonl \
   --clean build/bixbench_numeric_clean.jsonl --model "$MODEL" --device "$DEV" \
   --condition question --draws 3 --batch-size 10 \
   --out agentic_bixbench_$TAG.json >> logs/a2_${TAG}.log 2>&1
$P agentic_probe.py --items build/bixbench_numeric_q.jsonl \
   --clean build/bixbench_numeric_clean.jsonl --model "$MODEL" --device "$DEV" \
   --condition withheld --draws 3 --batch-size 10 \
   --out agentic_bixbench_$TAG.json >> logs/a2_${TAG}.log 2>&1
echo "done $TAG"
