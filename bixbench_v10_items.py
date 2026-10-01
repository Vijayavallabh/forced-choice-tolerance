#!/usr/bin/env python3
r"""BixBench v1.0 as an items file: the release the published no-data baselines ran on.

Every other BixBench number in this paper is on v1.5, the release shipped now.
The two published zero-shot runs are on v1.0, whose key-rank law is a fifth as
wide (``release_drift.py``: a +5.8 ceiling against v1.5's +26.0). The obvious
question is whether the field's own instrument collects anything on the release
the published baselines actually ran on. This writes that release
in the shape ``agentic_probe.py`` reads, so the grid's template cell can be run
on it unchanged.

The option sets are taken from the published gpt-4o run file, which records each
item's options as presented and the keyed letter; the key is the option under
that letter. The Claude run file holds the same 296 items. The question field is
left empty, as ``build/bixbench_v15.jsonl`` leaves it, because the template cell
withholds the question.

    python3 bixbench_v10_items.py
    python3 build_set_control.py --items build/bixbench_v10.jsonl \
        --out build/bixbench_v10_clean.jsonl
"""
import argparse
import ast
import csv
import json
import re
from pathlib import Path

LABEL = re.compile(r"^\(([A-Z])\)\s*")
RUN = "data/external/zero_shot_v10/bixbench_llm_baseline_refusal_False_mcq_gpt-4o_1.0.csv"


def items_from_run(path):
    out = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            choices = ast.literal_eval(row["choices"])
            letters = [LABEL.match(c).group(1) for c in choices]
            texts = [LABEL.sub("", c, count=1) for c in choices]
            key = letters.index(row["target"].strip())
            out.append({"ideal": texts[key],
                        "distractors": [t for i, t in enumerate(texts) if i != key],
                        "question": "", "cluster": row["uuid"], "short_qid": row["short_qid"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--out", default="build/bixbench_v10.jsonl")
    args = ap.parse_args()
    rows = items_from_run(args.run)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} items over {len({r['cluster'] for r in rows})} capsules -> {args.out}")


if __name__ == "__main__":
    main()
