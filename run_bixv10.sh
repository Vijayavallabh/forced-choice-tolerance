#!/bin/bash
# The grid's template cell -- BixBench's own template, one letter at a
# time -- on BixBench v1.0, the release the published zero-shot baselines ran on.
# Same invocation as run_bixgrid_remote.sh, so the rows compare with tab:gridbix;
# the control is v1.0's own option-string pool (build_set_control.py).
#   ./run_bixv10.sh "--device cuda:0" "Qwen/Qwen2.5-7B-Instruct|qwen7b|24" ...
#   ./run_bixv10.sh "--shard 2,3 --shard-auto 0.9" "meta-llama/Llama-3.3-70B-Instruct|llama70b|8"
set -u
cd "$(dirname "$(readlink -f "$0")")"
PLACE="$1"; shift
PY=${PY:-/opt/conda/bin/python}
TAGDEV=$(echo "$PLACE" | tr -c 'a-z0-9' '_')
mkdir -p logs build/dumps
for entry in "$@"; do
  IFS='|' read -r MODEL TAG BATCH <<< "$entry"
  dump="build/dumps/${TAG}_bixv10_bixprompt_argmax.jsonl"
  [ -s "$dump" ] && { echo "  have $TAG"; continue; }
  $PY agentic_probe.py \
    --items build/bixbench_v10.jsonl --clean build/bixbench_v10_clean.jsonl \
    --model "$MODEL" $PLACE --condition withheld --prompt bixbench --argmax \
    --draws 3 --batch-size "$BATCH" --dump "$dump" \
    --out "agentic_bixv10_${TAG}.json" >> "logs/bixv10_${TAG}.log" 2>&1
  echo "  $TAG rc=$?  $(date +%H:%M)"
done
echo "done bixv10 $PLACE"
