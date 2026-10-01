#!/bin/bash
set -u
cd "$(dirname "$(readlink -f "$0")")"
for ctx in set option; do
  /opt/conda/bin/python text_options.py --device "$1" --context $ctx --replicates 8 \
      --out text_options.json >> logs/text_$ctx.log 2>&1
done
echo done
