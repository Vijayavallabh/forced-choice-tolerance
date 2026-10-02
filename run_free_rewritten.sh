#!/bin/bash
# The deleted-option arm's forced cell through the two rewritten option sets the
# with-data runs are read through: the placebo (every distractor redrawn, the
# key's rank kept) and the repair (the rank drawn uniformly), on the six open
# models of results/free_response.json. rank_attribution.py reads the dumps
# (one sentence of Appendix C.3); the dumps ship gzipped under results/agentic_dumps/.
set -u
cd "$(dirname "$(readlink -f "$0")")"
DEV=${DEV:-cuda:0}
MODELS=(Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2.5-7B-Instruct
        meta-llama/Meta-Llama-3.1-8B-Instruct microsoft/phi-4 Qwen/Qwen2.5-14B-Instruct)
for arm in placebo repaired; do
  DEV="$DEV" ITEMS="build/bixbench_numeric_${arm}.jsonl" TAG="bixnum_${arm}" \
    OUT="results/free_response_${arm}.json" ./run_free_response.sh "${MODELS[@]}"
done
