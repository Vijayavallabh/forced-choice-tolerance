#!/bin/bash
# The instrument decomposition, on the venue's own benchmark.
#
# Table 2 crosses {how the question is put} x {how the answer is taken} on
# MMLU-Pro, because 913 items in 60 clusters can resolve a margin and
# BixBench's 105 numeric items cannot.  The cost is that the claim that
# matters here is then made on a file that is not this workshop's.
# BixBench's *whole* released file is 205 items in 59 capsules,
# which is enough, and the steered cell on it already reads +10.6 [+3.8,+17.7].
#
# So run the other three corners on exactly that file and control.  If
# BixBench's own template reads nothing where the neutral framing reads the
# channel, the decomposition is not a fact about MMLU-Pro.
#
# Cheap: 205 items x 3 draws is 615 rollouts a cell, and two of the four are
# arg-max, which generates nothing.
set -u
cd "$(dirname "$(readlink -f "$0")")"
DEV="$1"; WAITFOR="$2"
mkdir -p logs build/dumps
# Wait for the sweep, not for the card to look idle: between a sweep's arms
# there is a gap, and two jobs on one CUDA device here kill each other silently.
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
    --items build/bixbench_v15.jsonl --clean build/bixbench_all_clean.jsonl \
    --model Qwen/Qwen2.5-14B-Instruct --device "$DEV" --condition withheld $1 \
    --draws 3 --batch-size 12 \
    --dump build/dumps/qwen14b_bixall_$2.jsonl \
    --out agentic_bixarms_qwen14b.json >> logs/a15.log 2>&1
  echo "  bixall $2 rc=$?  $(date +%H:%M)"
}
run "--prompt bixbench --argmax" bixprompt_argmax
run "--prompt bixbench"          bixprompt_generated
run "--argmax --no-tools"        neutral_argmax
run "--no-tools"                 neutral_notools
echo "done bixarms"
