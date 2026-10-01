#!/bin/bash
# The two families the shared interpreter cannot load.
#
# Gemma-3 and OLMo-2 ship tokenizer.json in a format tokenizers 0.19 rejects
# ("data did not match any variant of untagged enum ModelWrapper"), so the grid
# skipped them with rc=1. They run under a venv that inherits torch from the
# same prefix and carries a newer transformers, and every arm records the
# library version it ran under, because an arm whose stack is not on the record
# cannot be compared with one whose is.
#
# Adding them takes the grid from three families to five, which is what "across
# model scale families" asks for.
set -u
cd "$(dirname "$(readlink -f "$0")")"
DEV="$1"
# Gemma-3 and OLMo-2 need the newer transformers, which lives in a venv.
# The path has moved once already, so take the first that exists rather than
# failing on a hardcoded one, and say which was used.
for _py in "${PY:-}" ${AGENTICLS_VENV:+"$AGENTICLS_VENV/bin/python"}; do
  [ -n "$_py" ] && [ -x "$_py" ] && { PY="$_py"; break; }
done
[ -n "${PY:-}" ] && [ -x "$PY" ] || {
  echo "no usable interpreter: set PY to one with transformers>=4.50" >&2
  exit 2
}
echo "using $PY"
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
mkdir -p logs build/dumps
# Anchor on the interpreter. `pgrep -f agentic_probe.py` matches any process
# whose command line merely CONTAINS that string, including the `bash -c`
# wrapper of an interactive session that launched one -- which once held a
# waiter for five hours with no probe running (see run_agentic17.sh).
while pgrep -f "^([^ ]*/)?python[0-9.]* agentic_probe\.py .*--device $DEV" > /dev/null; do sleep 20; done
GRID=(
  "google/gemma-3-4b-it|gemma4b|24"
  "allenai/OLMo-2-1124-7B-Instruct|olmo7b|24"
)
for entry in "${GRID[@]}"; do
  IFS='|' read -r MODEL TAG BATCH <<< "$entry"
  for cell in "bixprompt_argmax|--prompt bixbench --argmax" "neutral_notools|--no-tools"; do
    IFS='|' read -r NAME FLAGS <<< "$cell"
    $PY agentic_probe.py \
      --items build/mmlu_pro_matched_released.jsonl \
      --clean build/mmlu_pro_matched_clean.jsonl \
      --model "$MODEL" --device "$DEV" --condition withheld $FLAGS \
      --draws 2 --batch-size "$BATCH" \
      --dump build/dumps/${TAG}_mmlupro_${NAME}.jsonl \
      --out agentic_grid.json >> logs/a20.log 2>&1
    echo "  $TAG $NAME rc=$?  $(date +%H:%M)"
  done
done
echo "done venv grid"
