#!/bin/bash
# Memorisation, tested where the surplus actually is.
#
# Fitting (1) to each cell of the 2x2 says the rank term is the same everywhere
# (+2.6 to +3.5 points at ten options) and what varies is the *surplus* over it,
# which is largest -- +7.6 -- in the cell where the model answers immediately
# and writes nothing. Qwen2.5 postdates MMLU-Pro, so a solver recognising option
# sets it has seen would produce exactly that.
#
# The placebo separates them. Every distractor's value is redrawn and the key's
# rank is held at the released file's, so the rank channel is intact to within a
# tenth of a point while no option set is one that was published. A surplus that
# survives is a property of the options; one that dies was recognition.
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
/opt/conda/bin/python agentic_probe.py \
  --items build/mmlu_pro_matched_placebo.jsonl \
  --clean build/mmlu_pro_matched_clean.jsonl \
  --model "$MODEL" --device "$DEV" --condition withheld --no-tools \
  --draws 2 --batch-size 24 \
  --dump build/dumps/${TAG}_mmlupro_placebo_neutral_notools.jsonl \
  --out agentic_placebo_mmlupro.json >> logs/a14_$TAG.log 2>&1
echo "  $TAG placebo neutral_notools rc=$?  $(date +%H:%M)"
echo "done placebo $TAG"
