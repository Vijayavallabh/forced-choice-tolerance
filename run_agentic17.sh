#!/bin/bash
# The 70B-class row.
#
# The paper's scale ladder stops at 32B and its two collectors are both
# Qwen2.5, so "the two largest of four" is also "the two Qwen of four".  phi-4
# broke half of that -- the null under BixBench's template replicates outside
# Qwen2.5 -- and could not break the other half, because it refuses the neutral
# framing on 64.7% of rollouts.  A 70B-class model settles what is left: whether
# the crossover is scale, and whether the refusal is the family or the size.
#
# Cells, in value order.  The first two are the ones that carry the finding, run
# on both files; the last extends the four-model sweep the paper already has.
#   1  MMLU-Pro, BixBench's template, one letter   (what the field runs)
#   2  MMLU-Pro, question absent, generated        (the cell that reads highest)
#   3  BixBench 205, BixBench's template, one letter
#   4  BixBench 205, question absent, generated
#   5  MMLU-Pro, steered, holding the interpreter  (the ladder's own condition)
#
# Weights live on the array, not the root disk: 261GB free there against ~140GB
# a model.  HF_HUB_CACHE is passed through to the probe for the same reason.
set -u
cd "$(dirname "$(readlink -f "$0")")"
MODEL="$1"; TAG="$2"; PLACE="$3"; BATCH="$4"; WAITFOR="$5"
# Llama-3.3-70B ships a tokenizer.json that tokenizers 0.19 rejects, the same
# way Gemma-3 and OLMo-2 do, so it needs the venv rather than the shared prefix.
# Set PY to pick; every arm records the version that loaded it either way.
PY=${PY:-/opt/conda/bin/python}
# The weights cache has moved once; take the first that exists so a re-run
# reads the cache instead of silently re-downloading sixty gigabytes.
# Where it lives on a given host goes in AGENTICLS_HF_CACHES (colon-separated), not
# here: a path in a shipped script names the machine it ran on.
for _hf in "${HF_HUB_CACHE:-}" $(printf '%s' "${AGENTICLS_HF_CACHES:-}" | tr ':' ' ') \
           "$HOME/.cache/huggingface/hub"; do
  [ -n "$_hf" ] && [ -d "$_hf" ] && { export HF_HUB_CACHE="$_hf"; break; }
done
mkdir -p logs build/dumps
# Wait for the sweep that holds the cards, not for them to look idle: between a
# sweep's arms there is a gap, and two jobs on one device here collide silently.
# A waiter that can never succeed should say so. `grep -q ... 2>/dev/null`
# hides "no such file" because the marker is expected to appear later -- but it
# hides a wrong path and a dead producer just as well, and then this spins
# forever. Check the path is plausible up front, and bound the wait.
wait_for () {
  local marker="$1" waited=0 limit=${WAIT_TIMEOUT:-21600}
  [ -n "$marker" ] || { echo "wait_for: empty marker path" >&2; exit 2; }
  [ -d "$(dirname "$marker")" ] || {
    echo "wait_for: $(dirname "$marker") does not exist, so $marker never will" >&2
    exit 2; }
  while ! grep -q "^done" "$marker" 2>/dev/null; do
    sleep 45
    waited=$((waited + 45))
    [ "$waited" -lt "$limit" ] || {
      echo "wait_for: $marker has no ^done after ${waited}s; the sweep that" \
           "writes it is probably gone" >&2
      exit 3; }
  done
}

for waiter in $WAITFOR; do wait_for "$waiter"; done
# And then for every probe of ours still holding a card, whichever card it is:
# a 70B spread over four of them collides with any one.
#
# Anchor the pattern on the interpreter. `pgrep -f agentic_probe.py` matches any
# process whose command line merely CONTAINS that string, which includes the
# shell that launched one, a grep for one, and -- the way this was found -- the
# `bash -c` wrappers of an interactive session that ran one hours earlier. Two
# of those sat in the process table all night and held this loop for five hours
# with no probe running at all. The directory part is optional because a sweep
# run with PY=python3 writes a bare interpreter, and a waiter that MISSES is
# worse than one that self-matches: it starts while a job still holds the card.
probes_running () {
  pgrep -f "^([^ ]*/)?python[0-9.]* agentic_probe\.py" > /dev/null
}
while probes_running; do sleep 30; done

# Skip a cell already in the out file. A 70B cell costs half an hour, and the
# reason to restart this sweep is usually the placement rather than the arm --
# resharding to free a card should not re-run what is already measured.
have () {   # dump name
  [ -s "build/dumps/$1.jsonl" ]
}

run () {   # items, clean, extra flags, draws, name
  if have "${TAG}_$6"; then echo "  $TAG $6 already run, skipping"; return; fi
  $PY agentic_probe.py \
    --items "$1" --clean "$2" \
    --model "$MODEL" $PLACE --condition "$3" $4 \
    --draws "$5" --batch-size "$BATCH" \
    --dump build/dumps/${TAG}_$6.jsonl \
    --out agentic_arms_$TAG.json >> logs/a17_${TAG}.log 2>&1
  echo "  $TAG $6 rc=$?  $(date +%H:%M)"
}
PRO_R=build/mmlu_pro_matched_released.jsonl
PRO_C=build/mmlu_pro_matched_clean.jsonl
BIX_R=build/bixbench_v15.jsonl
BIX_C=build/bixbench_all_clean.jsonl

run "$PRO_R" "$PRO_C" withheld "--prompt bixbench --argmax" 2 mmlupro_bixprompt_argmax
run "$PRO_R" "$PRO_C" withheld "--no-tools"                 2 mmlupro_neutral_notools
run "$BIX_R" "$BIX_C" withheld "--prompt bixbench --argmax" 3 bixall_bixprompt_argmax
run "$BIX_R" "$BIX_C" withheld "--no-tools"                 3 bixall_neutral_notools
run "$PRO_R" "$PRO_C" withheld_aware ""                     2 mmlupro_steered_tools
echo "done $TAG"
