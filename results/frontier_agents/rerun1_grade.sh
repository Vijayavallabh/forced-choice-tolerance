#!/bin/bash
# Grade DeepSeek-V4-Pro's single-driver rerun (build/openai/agent_runs_rerun1) as the current agents were graded on
# 27 Sep (VERIFICATION.md, "Two current agents"): bixbench_withdata.py with gpt-4o (2024-11-20, the default Azure
# resource) as the MCQ reader -- forced and with BixBench's refusal option, through the released, placebo and
# repaired option sets, with the notebook -- and as the open-ended judge; the reads go into the same cache,
# build/openai/reader_cache_frontier.jsonl, after the first grading's. Run on 1 Oct 2026 (04:22-04:24 IST) with the
# camera-ready working tree's bixbench_withdata.py, whose prompts are those of the first grading (its cached reads
# of the same answers were reused); the number a row stores as within 5% is re-read downstream from the answer.
cd "$(dirname "$0")/../.."   # the repository root
BASE_A="$(/opt/conda/bin/python openai_api.py --print-base)"
export AGENTICLS_OPENAI_BUDGET_USD="${AGENTICLS_OPENAI_BUDGET_USD:-1100}"
exec /opt/conda/bin/python bixbench_withdata.py --runs build/openai/agent_runs_rerun1 --models DeepSeek-V4-Pro-react \
  --reader "gpt-4o=$BASE_A" --judge "gpt-4o=$BASE_A" --cache build/openai/reader_cache_frontier.jsonl \
  --rows build/openai/withdata_rows_frontier_rerun1.json --out build/openai/withdata_frontier_rerun1.json "$@"
