#!/bin/bash
# The 2x2 the agentic claim needs: {instruction names the options, instruction
# says nothing} x {holding an interpreter, holding nothing}.  The cell that is
# already run is (named, interpreter); these are the other three.
#
# Why both axes.  A margin that appears only when the system prompt tells the
# solver to read the options is steerability and should be named that.  A margin
# that survives with the interpreter taken away is not the calculator, so "agent"
# is the wrong word for the collector and "generation pass" is the right one.
# Splitting one arm's rollouts on whether they happened to run code cannot settle
# the second question, because that split is the model's choice and not ours.
set -u
cd "$(dirname "$(readlink -f "$0")")"
P=/opt/conda/bin/python
MODEL="$1"; TAG="$2"; PLACE="$3"; BATCH="$4"   # PLACE: "--device cuda:1" or "--shard 2,3 ..."
mkdir -p logs build/dumps
run () {   # condition, notools-flag, name
  $P agentic_probe.py --items build/mmlu_pro_matched_released.jsonl \
     --clean build/mmlu_pro_matched_clean.jsonl --model "$MODEL" $PLACE \
     --condition "$1" $2 --draws 2 --batch-size "$BATCH" \
     --dump build/dumps/${TAG}_mmlupro_$3.jsonl \
     --out agentic_arms_$TAG.json >> logs/a7_${TAG}.log 2>&1
  echo "  $TAG $3 rc=$?  $(date +%H:%M)"
}
run withheld_aware --no-tools steered_notools
run withheld       ""         neutral_tools
run withheld       --no-tools neutral_notools
echo "done $TAG"
