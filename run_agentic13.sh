#!/bin/bash
# The other half of the decomposition: BixBench's own verbatim MCQ prompt, with
# each read-out.  Crossed with run_agentic12's two cells this is
#
#              |  read one letter  |  generate the answer
#   ------------------------------------------------------
#   BixBench's |   what the field  |
#   own prompt |   actually runs   |
#   our agent  |                   |  what section 4 reports
#
# on one model, one file, one set of items and one set of letter orderings. Any
# difference between the four cells is the prompt, the read-out, or their
# interaction, and nothing else -- which is the question the title asks.
set -u
cd "$(dirname "$(readlink -f "$0")")"
MODEL="$1"; TAG="$2"; DEV="$3"; WAITFOR="$4"
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
for readout in "--argmax" "--no-tools"; do
  name=$([ "$readout" = "--argmax" ] && echo argmax || echo generated)
  /opt/conda/bin/python agentic_probe.py \
    --items build/mmlu_pro_matched_released.jsonl \
    --clean build/mmlu_pro_matched_clean.jsonl \
    --model "$MODEL" --device "$DEV" --condition withheld --prompt bixbench $readout \
    --draws 2 --batch-size 24 \
    --dump build/dumps/${TAG}_mmlupro_bixprompt_${name}.jsonl \
    --out agentic_arms_$TAG.json >> logs/a13_$TAG.log 2>&1
  echo "  $TAG bixprompt $name rc=$?  $(date +%H:%M)"
done
echo "done bixprompt $TAG"
