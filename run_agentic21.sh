#!/bin/bash
# The grid's two cells on the venue's own benchmark, not only on MMLU-Pro.
#
# tab:grid is eleven models on MMLU-Pro, and the one result this workshop cares
# about most -- that BixBench's own template already reads +8.5 on BixBench's
# own file -- rests on a single model there. This runs the same two cells on all
# 205 released items against the file's own option-string control, for models
# that span the ladder and two families.
#
# phi-4 is in the list on purpose. It refuses the neutral cell on MMLU-Pro on
# 64.7% of rollouts; whether it refuses the same cell on a four-option file with
# different option text says whether that refusal is the framing or the file.
#
# 205 items x 3 draws is 615 rollouts a cell, so the whole sweep is minutes.
set -u
cd "$(dirname "$(readlink -f "$0")")"
DEV="$1"
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
mkdir -p logs build/dumps
while pgrep -f "^([^ ]*/)?python[0-9.]* agentic_probe\.py .*--device $DEV" > /dev/null; do sleep 20; done
GRID=(
  "Qwen/Qwen2.5-1.5B-Instruct|qwen1_5b|24"
  "meta-llama/Llama-3.2-3B-Instruct|llama3b|24"
  "Qwen/Qwen2.5-7B-Instruct|qwen7b|16"
  "microsoft/phi-4|phi4|12"
)
for entry in "${GRID[@]}"; do
  IFS='|' read -r MODEL TAG BATCH <<< "$entry"
  for cell in "bixprompt_argmax|--prompt bixbench --argmax" "neutral_notools|--no-tools"; do
    IFS='|' read -r NAME FLAGS <<< "$cell"
    /opt/conda/bin/python agentic_probe.py \
      --items build/bixbench_v15.jsonl --clean build/bixbench_all_clean.jsonl \
      --model "$MODEL" --device "$DEV" --condition withheld $FLAGS \
      --draws 3 --batch-size "$BATCH" \
      --dump build/dumps/${TAG}_bixall_${NAME}.jsonl \
      --out agentic_bixgrid.json >> logs/a21.log 2>&1
    echo "  $TAG $NAME rc=$?  $(date +%H:%M)"
  done
done
echo "done bixgrid"

# The one cell tab:grid is missing: Qwen2.5-32B under BixBench's template. Its
# neutral arm came from the 2x2 sweep, which never ran the template cell, so the
# table has a gap exactly where the ladder is steepest.
/opt/conda/bin/python agentic_probe.py \
  --items build/mmlu_pro_matched_released.jsonl \
  --clean build/mmlu_pro_matched_clean.jsonl \
  --model Qwen/Qwen2.5-32B-Instruct --device "$DEV" --condition withheld \
  --prompt bixbench --argmax --draws 2 --batch-size 8 \
  --dump build/dumps/qwen32b_mmlupro_bixprompt_argmax.jsonl \
  --out agentic_arms_qwen32b.json >> logs/a21.log 2>&1
echo "  qwen32b bixprompt_argmax rc=$?  $(date +%H:%M)"
echo "done bixgrid+32b"
