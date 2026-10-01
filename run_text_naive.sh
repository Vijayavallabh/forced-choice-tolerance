#!/bin/bash
# The same reader on SeqQA and DbQA with the benchmark's own subtask structure
# ignored, which is what an audit that does not look for one would do.
set -u
cd "$(dirname "$(readlink -f "$0")")"
# Anchor on the interpreter. `pgrep -f agentic_probe.py` matches any process
# whose command line merely CONTAINS that string, including the `bash -c`
# wrapper of an interactive session that launched one -- which once held a
# waiter for five hours with no probe running (see run_agentic17.sh).
while pgrep -f "^([^ ]*/)?python[0-9.]* text_options\.py --device cuda:2 --context set --replicates 8 --out" > /dev/null; do sleep 30; done
/opt/conda/bin/python text_options.py --device cuda:2 --context set --replicates 8 \
    --ignore-groups --only "LAB-Bench SeqQA" --only "LAB-Bench DbQA" \
    --out text_options_naive.json >> logs/text_naive.log 2>&1
echo done
