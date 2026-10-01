#!/bin/bash
# DeepSeek-V4-Pro run again with ONE driver (1 Oct 2026). Its first run (27 Sep, logs/launch_frontier.sh) had a
# second driver on the same --out that overwrote finished episodes, and three content-filter refusals crashed and
# were run again. This is launch_frontier.sh's command for DeepSeek-V4-Pro with the settings its kept episodes
# record -- the second Azure resource, prices 2,0.2,8, --max-model-len 128000, --view-budget 200000, no reasoning
# effort, concurrency 20, and every other flag at the harness default (temperature 1.0, 40 steps, one hour,
# 600 s cells, seed 20260923) -- written to new directories, so nothing of the first run is touched. The budget
# is the ledger's spend before this run ($909.55) plus $300.
cd "$(dirname "$0")/../.."   # the repository root
export AZURE_OPENAI_API_KEY="$(cat ~/.config/agenticls/openai_b.key)"
BASE_B="$(AZURE_OPENAI_ENDPOINT="$(cat ~/.config/agenticls/azure_openai_b.endpoint)" /opt/conda/bin/python openai_api.py --print-base)"
export AGENTICLS_OPENAI_BUDGET_USD=1210
AGENTICLS_OPENAI_PRICES="2,0.2,8" exec /opt/conda/bin/python bixbench_agent.py --model DeepSeek-V4-Pro --base "$BASE_B" \
  --protocol react --condition data --only-file build/openai/v15_numeric_question_ids.txt --rollouts 1 \
  --max-tokens 32768 --max-model-len 128000 --view-budget 200000 --concurrency 20 \
  --out build/openai/agent_runs_rerun1 --work build/openai/bixbench-work-rerun1 "$@"
