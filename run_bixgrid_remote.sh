#!/bin/bash
# Complete tab:gridbix: the two grid cells on BixBench's whole released file for
# the six models tab:grid has and tab:gridbix does not. Same invocation as
# run_agentic21.sh so the rows are comparable -- 205 items, 3 draws, the file's
# own option-string control.
#
# Gemma-3 and OLMo-2 ship a tokenizer.json that tokenizers 0.19 rejects; this
# host has transformers 5.3, so they run on its system python like everything
# else.
set -u
cd "$(dirname "$(readlink -f "$0")")"
DEV="$1"; shift
PY=${PY:-python3}
mkdir -p logs build/dumps
for entry in "$@"; do
  IFS='|' read -r MODEL TAG BATCH <<< "$entry"
  for cell in "bixprompt_argmax|--prompt bixbench --argmax" "neutral_notools|--no-tools"; do
    IFS='|' read -r NAME FLAGS <<< "$cell"
    dump="build/dumps/${TAG}_bixall_${NAME}.jsonl"
    [ -s "$dump" ] && { echo "  have $TAG $NAME"; continue; }
    $PY agentic_probe.py \
      --items build/bixbench_v15.jsonl --clean build/bixbench_all_clean.jsonl \
      --model "$MODEL" --device "$DEV" --condition withheld $FLAGS \
      --draws 3 --batch-size "$BATCH" --dump "$dump" \
      --out agentic_bixgrid_remote.json >> logs/bixgrid_$DEV.log 2>&1
    echo "  $TAG $NAME rc=$?  $(date +%H:%M)"
  done
done
echo "done bixgrid $DEV"
