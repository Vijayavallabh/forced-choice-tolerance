#!/bin/bash
# The family confound, and the finding rather than one arm.
#
# Both models that collect the channel are Qwen2.5, so "the two largest of four"
# is also "the two Qwen of four", and the sweep cannot separate them. phi-4 is
# a 14B instruction-tuned model from another family and
# another pretraining corpus, at the size where Qwen2.5 first collects.
#
# It runs the two cells that carry the result, not the whole grid: BixBench's
# own prompt read one letter at a time, which is what the field runs and where
# Qwen2.5-14B reads +1.3, and the neutral agent framing answered directly, where
# the same weights read +10.4. If the gap replicates it is about the prompt and
# not about Qwen2.5; if it does not, the finding must be named for the family.
set -u
cd "$(dirname "$(readlink -f "$0")")"
DEV="$1"; WAITFOR="$2"
mkdir -p logs build/dumps
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

wait_for "$WAITFOR"
# Anchor on the interpreter. `pgrep -f agentic_probe.py` matches any process
# whose command line merely CONTAINS that string, including the `bash -c`
# wrapper of an interactive session that launched one -- which once held a
# waiter for five hours with no probe running (see run_agentic17.sh).
while pgrep -f "^([^ ]*/)?python[0-9.]* agentic_probe\.py .*--device $DEV" > /dev/null; do sleep 20; done
run () {   # extra flags, name
  /opt/conda/bin/python agentic_probe.py \
    --items build/mmlu_pro_matched_released.jsonl \
    --clean build/mmlu_pro_matched_clean.jsonl \
    --model microsoft/phi-4 --device "$DEV" --condition withheld $1 \
    --draws 2 --batch-size 16 \
    --dump build/dumps/phi4_mmlupro_$2.jsonl \
    --out agentic_arms_phi4.json >> logs/a11_phi4.log 2>&1
  echo "  phi4 $2 rc=$?  $(date +%H:%M)"
}
run "--prompt bixbench --argmax" bixprompt_argmax
run "--no-tools"                 neutral_notools
echo "done phi4"
