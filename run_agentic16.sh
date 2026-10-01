#!/bin/bash
# Restore the rollouts behind the paper's MMLU-Pro headline.
#
# run_agentic8 re-ran the 14B's steered arm at batch 12 and wrote its dump over
# the batch-10 one.  That is not a lost file, it is a lost *reproduction*: the
# paper quotes +6.25 [+4.04,+8.69] from the batch-10 arm, and arm_intervals.py
# promises to rebuild every published interval from the rollouts it ships.  With
# the dump gone it rebuilt +5.60 instead and said so, which is the check working.
#
# Greedy decoding is deterministic given the batching, so re-running at batch 10
# returns the same rollouts.  If it does not, that is worth knowing more than
# the interval is.
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
/opt/conda/bin/python agentic_probe.py \
  --items build/mmlu_pro_matched_released.jsonl \
  --clean build/mmlu_pro_matched_clean.jsonl \
  --model Qwen/Qwen2.5-14B-Instruct --device "$DEV" \
  --condition withheld_aware --draws 2 --batch-size 10 \
  --dump build/dumps/qwen14b_mmlupro_steered_tools.jsonl \
  --out agentic_mmlupro_qwen14b_rerun.json >> logs/a16.log 2>&1
echo "done rerun rc=$?  $(date +%H:%M)"
/opt/conda/bin/python - <<'PY'
import json, pathlib
a = json.loads(pathlib.Path("results/agentic_mmlupro_qwen14b.json").read_text())
b = json.loads(pathlib.Path("results/agentic_mmlupro_qwen14b_rerun.json").read_text())
for name, doc in (("published", a), ("rerun", b)):
    arm = next(iter(doc.values()))
    print(name, round(arm["file"]["accuracy"], 4), round(arm["clean"]["accuracy"], 4),
          [round(v, 4) for v in arm["margin_over_clean_ci95"]])
PY
