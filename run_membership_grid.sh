#!/bin/bash
# The membership arm on more than one model. The decisive test -- does the
# membership score predict which rollouts the model gets right -- needs only
# that model's own neutral-cell dump, not a placebo, so it runs wherever a
# neutral dump exists.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=${PY:-/opt/conda/bin/python}
DEV="$1"; shift
# A fourth field spreads one model's weights over several cards, for the two
# 70B-class rows. Each card is given only a share of what is FREE on it, so a
# neighbour's job is not evicted.
# ARM names the dump suffix, so the same sweep runs the venue's own file:
# ``bixall_bixprompt_argmax`` is BixBench's own template on all 205 released
# items, which is the cell every model answers.
ARM=${ARM:-mmlupro_neutral_notools}
OUT=${OUT:-results/option_membership_grid.json}
for entry in "$@"; do
  IFS='|' read -r MODEL TAG BATCH SHARD <<< "$entry"
  dump="build/dumps/${TAG}_${ARM}.jsonl"
  [ -s "$dump" ] || { echo "  no dump for $TAG"; continue; }
  echo "=== $MODEL ==="
  $PY option_membership.py --model "$MODEL" --device "$DEV" \
      ${SHARD:+--shard "$SHARD"} \
      --released "$dump" --placebo "" --batch-size "$BATCH" \
      --output "$OUT" 2>&1 | tail -16
  echo "  rc=$?"
done
echo "done membership $DEV"
