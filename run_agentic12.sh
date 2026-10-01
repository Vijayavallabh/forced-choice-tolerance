#!/bin/bash
# The instrument comparison with everything held fixed that can be.
#
# \S4 contrasts "single-token probes read nothing" with "the agent reads +6.3",
# and those two numbers come from different prompts, different templates and
# different read-outs, so the contrast is not identified. This runs the *same*
# prompt -- same system message, same user message, same chat template, same
# items and orderings -- and changes only how the answer is taken: ``FINAL:`` is
# pre-filled and the letter is the arg-max over the k single-token
# continuations, which is how every no-data baseline in this literature reads.
#
# Crossed with the instruction, that gives {steered, neutral} x {generated,
# arg-max} on one model and one file, and says which of the two is doing the
# work. One forward pass per rollout, so it is minutes rather than hours.
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
for cond in withheld_aware withheld; do
  /opt/conda/bin/python agentic_probe.py \
    --items build/mmlu_pro_matched_released.jsonl \
    --clean build/mmlu_pro_matched_clean.jsonl \
    --model "$MODEL" --device "$DEV" --condition "$cond" --argmax \
    --draws 2 --batch-size 24 \
    --dump build/dumps/${TAG}_mmlupro_argmax_${cond}.jsonl \
    --out agentic_arms_$TAG.json >> logs/a12_$TAG.log 2>&1
  echo "  $TAG argmax $cond rc=$?  $(date +%H:%M)"
done
echo "done argmax $TAG"
