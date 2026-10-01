#!/usr/bin/env python3
"""Put the two science-agent option sets into the row shape the probes read.

``survey_to_jsonl.py`` does this for the vendored HuggingFace files. BixBench and
LAB-Bench are read straight out of ``data/``, by the same parsers
``channel_survey.py`` uses, so the items here are the items the survey figure
counts and no third reading of either release enters the paper.

    python3 bio_to_jsonl.py --source bixbench_v15 --out build/bixbench_v15.jsonl
    python3 bio_to_jsonl.py --source labbench_SeqQA --out build/labbench_seqqa.jsonl
"""
import argparse
import json
from pathlib import Path

from channel_survey import bixbench_v15, bixbench_v10, labbench


def source(name):
    if name == "bixbench_v15":
        return bixbench_v15()
    if name == "bixbench_v10":
        return bixbench_v10()
    if name.startswith("labbench_"):
        return labbench(Path("data/external/labbench") / f"{name[len('labbench_'):]}.jsonl")
    raise SystemExit(f"unknown source: {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, counts = [], {}
    for item in source(args.source):
        options = [str(o) for o in item["options"]]
        if item["key"] != 0:
            raise SystemExit("the parsers put the key at slot 0")
        counts[len(options)] = counts.get(len(options), 0) + 1
        rows.append({"ideal": options[0], "distractors": options[1:],
                     "question": "", "cluster": item.get("cluster") or ""})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows),
                              encoding="utf-8")
    modal = max(counts, key=counts.get) if counts else 0
    print(f"wrote {args.out}: {len(rows)} rows, modal k={modal} "
          f"({counts.get(modal, 0)} rows)")


if __name__ == "__main__":
    main()
