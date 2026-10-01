#!/bin/bash
# The arms the 14B agent needs, on the first GPU that frees: the positive
# control, then the placebo/repair contrast. Placebo redraws the distractors
# with the key's rank held at the released file's; repair uniformises the rank
# with the same generator. A margin that survives the placebo and dies on the
# repair is the key's rank and not the rewriting.
set -u
cd "$(dirname "$0")"
MODEL="$1"; DEV="$2"; TAG="$3"; WAITPID="$4"
P=/opt/conda/bin/python
while kill -0 "$WAITPID" 2>/dev/null; do sleep 30; done
$P agentic_probe.py --items build/bixbench_numeric_q.jsonl \
   --clean build/bixbench_numeric_clean.jsonl --model "$MODEL" --device "$DEV" \
   --condition question --draws 3 --batch-size 10 \
   --out agentic_bixbench_$TAG.json >> logs/a4_${TAG}.log 2>&1
for arm in placebo repaired; do
  $P agentic_probe.py --items build/bixbench_numeric_$arm.jsonl \
     --clean build/bixbench_numeric_clean.jsonl --model "$MODEL" --device "$DEV" \
     --condition withheld_aware --draws 3 --batch-size 10 \
     --dump build/dumps/${TAG}_${arm}.jsonl \
     --out agentic_bixarm_${arm}_$TAG.json >> logs/a4_${TAG}.log 2>&1
done
echo "done $TAG"
