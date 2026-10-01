#!/bin/bash
# Re-analyse every model from its score cache, with no GPU.
#
# The scoring is the expensive half and its cache is the artifact; the analysis
# is cheap and changed twice while the sweep was running -- once to add the
# clean control's own correct-minus-wrong gap, and once to add the paired
# released-minus-control difference. Models analysed before those were added
# carry a different set of keys, so every model is re-read here in one pass
# with one version of the code, rather than a grid whose rows were computed by
# three.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=${PY:-/opt/conda/bin/python}
JOBS=${JOBS:-5}
# ARM/OUT/SLUG mirror run_membership_grid.sh, so the same re-analysis runs over
# either file's caches.
ARM=${ARM:-mmlupro_neutral_notools}
OUT=${OUT:-results/option_membership_grid.json}
SLUG=${SLUG:-}
running=0
for entry in "$@"; do
  IFS='|' read -r MODEL TAG <<< "$entry"
  tag_file=$(echo "$MODEL" | sed 's|.*/||; s|-Instruct$||')
  cache="build/option_membership_scores_${tag_file}${SLUG}.json"
  shipped="results/agentic_dumps/option_membership_scores_${tag_file}${SLUG}.json.gz"
  plain="results/agentic_dumps/option_membership_scores${SLUG}.json.gz"
  [ -s "$cache" ] || [ -s "$shipped" ] || \
    { [ "$tag_file" = "Qwen2.5-14B" ] && [ -s "$plain" ]; } || \
    { echo "  no cache for ${tag_file}${SLUG}"; continue; }
  echo "=== $MODEL ==="
  $PY option_membership.py --model "$MODEL" \
      --released "build/dumps/${TAG}_${ARM}.jsonl" \
      --placebo "" --output "$OUT" \
      > "build/reanalyse_${tag_file}${SLUG}.log" 2>&1 &
  running=$((running + 1))
  if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi
done
wait

# Ship every cache the sweep produced. package_submission.py refuses a grid
# with a row whose scores are not in the bundle, and the gzip step is the kind
# of thing that gets done for the first model and forgotten for the next
# twelve.
for cache in build/option_membership_scores_*.json; do
  [ -s "$cache" ] || continue
  tag=$(basename "$cache" .json); tag=${tag#option_membership_scores_}
  if [ "$tag" = "Qwen2.5-14B" ]; then
    out="results/agentic_dumps/option_membership_scores.json.gz"
  elif [ "$tag" = "Qwen2.5-14B_bixall" ]; then
    out="results/agentic_dumps/option_membership_scores_bixall.json.gz"
  else
    out="results/agentic_dumps/option_membership_scores_${tag}.json.gz"
  fi
  gzip -9 -c "$cache" > "$out"
  echo "  shipped $out"
done
echo "done reanalyse"
