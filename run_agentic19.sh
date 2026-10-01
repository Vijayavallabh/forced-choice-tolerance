#!/bin/bash
# The family x scale grid, which is what "a threshold with an estimable
# boundary" would need.
#
# The paper's scale claim rests on four models, two of which are Qwen2.5 and
# both of which clear.  That is an existence proof, not a law.  This runs the
# two cells that carry the finding -- BixBench's own template read one letter
# at a time, and the neutral framing generated -- across five families and the
# sizes each one ships, on MMLU-Pro's 913 ten-option items against the matched
# clean control.  Same items, same two letter orderings, same control, same
# bootstrap; only the weights change.
#
# Two cells and not five: the arg-max cell is what the field runs and the
# neutral cell is what reads highest, and the distance between them is the
# quantity the grid is being built to trace.  Everything else is already
# measured on the models where it mattered.
#
# Compliance is recorded per arm and per cell, because Llama-3.1-8B and phi-4
# both answer one of these cells and refuse the other, and a refusal rate read
# as an option preference is the failure this whole section is about.
set -u
cd "$(dirname "$(readlink -f "$0")")"
DEV="$1"
# The weights cache has moved once; take the first that exists so a re-run
# reads the cache instead of silently re-downloading sixty gigabytes.
# Where it lives on a given host goes in AGENTICLS_HF_CACHES (colon-separated), not
# here: a path in a shipped script names the machine it ran on.
for _hf in "${HF_HUB_CACHE:-}" $(printf '%s' "${AGENTICLS_HF_CACHES:-}" | tr ':' ' ') \
           "$HOME/.cache/huggingface/hub"; do
  [ -n "$_hf" ] && [ -d "$_hf" ] && { export HF_HUB_CACHE="$_hf"; break; }
done
mkdir -p logs build/dumps
# Anchor on the interpreter. `pgrep -f agentic_probe.py` matches any process
# whose command line merely CONTAINS that string, including the `bash -c`
# wrapper of an interactive session that launched one -- which once held a
# waiter for five hours with no probe running (see run_agentic17.sh).
while pgrep -f "^([^ ]*/)?python[0-9.]* agentic_probe\.py .*--device $DEV" > /dev/null; do sleep 20; done

# model, tag, batch size.  Batch is set by size, not tuned: the read-out is
# batch-sensitive in bf16 and a per-model batch keeps each model's two cells
# comparable to each other, which is the comparison the grid makes.
GRID=(
  "meta-llama/Llama-3.2-1B-Instruct|llama1b|48"
  "Qwen/Qwen2.5-1.5B-Instruct|qwen1_5b|48"
  "meta-llama/Llama-3.2-3B-Instruct|llama3b|32"
  "microsoft/Phi-3.5-mini-instruct|phi35mini|32"
  "google/gemma-3-4b-it|gemma4b|32"
  "allenai/OLMo-2-1124-7B-Instruct|olmo7b|24"
  "Qwen/Qwen2.5-7B-Instruct|qwen7b|24"
  "meta-llama/Llama-3.1-8B-Instruct|llama8b|24"
)
for entry in "${GRID[@]}"; do
  IFS='|' read -r MODEL TAG BATCH <<< "$entry"
  for cell in "bixprompt_argmax|--prompt bixbench --argmax" "neutral_notools|--no-tools"; do
    IFS='|' read -r NAME FLAGS <<< "$cell"
    /opt/conda/bin/python agentic_probe.py \
      --items build/mmlu_pro_matched_released.jsonl \
      --clean build/mmlu_pro_matched_clean.jsonl \
      --model "$MODEL" --device "$DEV" --condition withheld $FLAGS \
      --draws 2 --batch-size "$BATCH" \
      --dump build/dumps/${TAG}_mmlupro_${NAME}.jsonl \
      --out agentic_grid.json >> logs/a19.log 2>&1
    echo "  $TAG $NAME rc=$?  $(date +%H:%M)"
  done
done
echo "done grid"
