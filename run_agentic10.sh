#!/bin/bash
# BixBench whole, not BixBench numeric.  \S4's agent arm runs on the 105 items
# whose options are distinct numbers, because the clean control it is read
# against redraws numeric values.  Gamma is defined over an option set whatever
# the options are, so that restriction belongs to the control.  This runs the
# same agent on all 205 released items against a control drawn i.i.d. from the
# file's own pool of 709 option strings -- same vocabulary, same register, and
# no relation between a key and the set it sits in.
#
# It also asks something \S3 could not: a character n-gram reader finds nothing
# in BixBench's option *text*, and this says whether an agent does.
set -u
cd "$(dirname "$(readlink -f "$0")")"
MODEL="$1"; TAG="$2"; DEV="$3"
# Wait for the whole 2x2 sweep to finish, not merely for the card to look idle:
# between its arms there is a gap, and two jobs on one CUDA device on this
# machine kill each other with no traceback.
WAITFOR="$4"
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
while pgrep -f "^([^ ]*/)?python[0-9.]* agentic_probe\.py .*--device $DEV" > /dev/null; do sleep 30; done
/opt/conda/bin/python agentic_probe.py \
  --items build/bixbench_v15.jsonl --clean build/bixbench_all_clean.jsonl \
  --model "$MODEL" --device "$DEV" \
  --condition withheld_aware --draws 3 --batch-size 12 \
  --dump build/dumps/${TAG}_bixall.jsonl \
  --out agentic_bixall_$TAG.json >> logs/a10_$TAG.log 2>&1
echo "done bixall $TAG rc=$?  $(date +%H:%M)"
