#!/bin/bash
# The temporal test on every LAB-Bench file whose dates were recovered, not just
# LitQA2: TableQA has more dated items (243 of 244) over a wider range
# (1991-2024), and FigQA and SuppQA date completely.
set -u
cd "$(dirname "$0")"
PY=${PY:-/opt/conda/bin/python}
M=meta-llama/Meta-Llama-3.1-8B-Instruct
D=${D:-cuda:3}
for spec in "TableQA:LAB-Bench TableQA" "FigQA:LAB-Bench FigQA" "SuppQA:LAB-Bench SuppQA"; do
  file=${spec%%:*}; label=${spec#*:}
  out=results/temporal_test_$(echo "$file" | tr 'A-Z' 'a-z').json
  if [ -s "$out" ]; then echo "have $out"; continue; fi
  echo "=== $label ==="
  $PY temporal_test.py --jsonl "data/external/labbench/$file.jsonl" \
      --benchmark "$label" --model "$M" --device "$D" \
      --output "$out" 2>&1 | tail -14
  echo "  rc=$?"
done
echo "done temporal"
