#!/bin/bash
# Proposition 1's repair, run by us across models, to remove "one benchmark and
# two models" from the limitation. Two arms per model on the same items: the
# question with its options and a forced letter, and the question alone with the
# answer as a number.
set -u
cd "$(dirname "$0")"
# The two hosts this ran on have different interpreters -- one has the conda
# prefix, the other only a system python3 plus a --target install on
# PYTHONPATH -- so the copies diverged and had to be edited apart. Take the
# first that exists instead, and carry the extra path if one is set.
for _py in "${PY:-}" /opt/conda/bin/python "$(command -v python3)"; do
  [ -n "$_py" ] && [ -x "$_py" ] && { PY="$_py"; break; }
done
[ -n "${PY:-}" ] && [ -x "$PY" ] || { echo "no usable interpreter" >&2; exit 2; }
[ -d "${PYTHONPATH:-}" ] && export PYTHONPATH
[ -z "${PYTHONPATH:-}" ] && [ -d "$HOME/agenticls/pylib" ] \
  && export PYTHONPATH="$HOME/agenticls/pylib"
DEV=${DEV:-cuda:2}
ITEMS=${ITEMS:-build/bixbench_numeric_q.jsonl}
TAG=${TAG:-bixnum}
OUT=${OUT:-results/free_response.json}
shift_models=("$@")
for m in "${shift_models[@]}"; do
  short=$(echo "$m" | sed 's#.*/##;s/-Instruct//;s/\./_/g')
  dump="build/dumps/free_${short}_${TAG}.jsonl"
  if grep -q "\"$m|" "$OUT" 2>/dev/null; then echo "have $m"; continue; fi
  echo "=== $m on $DEV ==="
  rm -f "$dump"
  $PY free_response.py --model "$m" --device "$DEV" --items "$ITEMS" \
      --out "$OUT" --dump "$dump" 2>&1 | tail -8
  echo "  rc=$?"
done
echo "done free_response $TAG $DEV"
