#!/bin/bash
# The 14B's MMLU-Pro arm again, this time dumping the rollouts.  The arm itself
# is already reported; what is missing is the per-rollout record, without which
# its interval cannot be recomputed at the level the cluster bootstrap actually
# covers at, and its rank term cannot be fitted the way the 32B's was.
# Greedy decoding and fixed batching make this a replication, not a second
# sample: the accuracy it prints must match results/agentic_mmlupro_qwen14b.json.
set -u
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs build/dumps
/opt/conda/bin/python agentic_probe.py \
  --items build/mmlu_pro_matched_released.jsonl \
  --clean build/mmlu_pro_matched_clean.jsonl \
  --model Qwen/Qwen2.5-14B-Instruct --device cuda:0 \
  --condition withheld_aware --draws 2 --batch-size 12 \
  --dump build/dumps/qwen14b_mmlupro.jsonl \
  --out agentic_mmlupro_qwen14b_dump.json >> logs/a8_qwen14b.log 2>&1
echo "done 14b mmlupro dump rc=$?"
