#!/bin/bash
# The memorisation control the resolving file was missing.  BixBench has a
# rank-holding placebo in \S4; MMLU-Pro, which is the arm the agentic claim
# actually rests on, did not.  Every option set here is new and the key's rank
# law is the released file's to within a tenth of a point, so a margin that
# survives is not recognition of a published option set.
# Waits for the card to free rather than sharing it: two jobs on one device on
# this machine kill each other silently.
set -u
cd "$(dirname "$(readlink -f "$0")")"
# Anchor on the interpreter. `pgrep -f agentic_probe.py` matches any process
# whose command line merely CONTAINS that string, including the `bash -c`
# wrapper of an interactive session that launched one -- which once held a
# waiter for five hours with no probe running (see run_agentic17.sh).
while pgrep -f "^([^ ]*/)?python[0-9.]* agentic_probe\.py .*--device cuda:0" > /dev/null; do sleep 60; done
/opt/conda/bin/python agentic_probe.py \
  --items build/mmlu_pro_matched_placebo.jsonl \
  --clean build/mmlu_pro_matched_clean.jsonl \
  --model Qwen/Qwen2.5-14B-Instruct --device cuda:0 \
  --condition withheld_aware --draws 2 --batch-size 12 \
  --dump build/dumps/qwen14b_mmlupro_placebo.jsonl \
  --out agentic_placebo_mmlupro.json >> logs/a9.log 2>&1
echo "done placebo rc=$?  $(date +%H:%M)"
