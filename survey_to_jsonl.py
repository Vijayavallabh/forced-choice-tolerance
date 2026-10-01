#!/usr/bin/env python3
"""Turn a vendored survey option set into the row shape ``mcq_audit.py`` reads.

``channel_survey.py`` measures eighteen benchmarks from ``data/external/survey``,
where each row is ``{options, key, n_options, cluster}``. The audit and the
repair want ``{ideal, distractors, question, cluster}``. This converts one file
so the ten-option MMLU-Pro results in the paper can be rebuilt from vendored
data rather than from a scratch file:

    python3 survey_to_jsonl.py --survey mmlu_pro --out build/mmlu_pro.jsonl
    python3 mcq_audit.py --jsonl build/mmlu_pro.jsonl --cluster-field cluster \\
        --n-options 10 --two-channel --search construct \\
        --repair build/mmlu_pro_repaired.jsonl \\
        --output results/mcq_audit_mmlu_pro_multi.json

The survey vendors option sets only, not question text, so ``question`` is
empty. That is a real restriction and the audit reports under it: the surface
rule family includes rules that score an option by its overlap with the stem,
and with no stem they contribute nothing. Every claim this repository makes
about MMLU-Pro is therefore about the option set alone -- which is the claim it
wants to make, since the option set is what a repair rewrites.
"""
import argparse
import gzip
import json
from pathlib import Path

SURVEY_DIR = Path("data/external/survey")


def convert(path):
    """One row per survey row, key first and distractors in their released order."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            options, key = row["options"], int(row["key"])
            yield {"ideal": options[key],
                   "distractors": [o for i, o in enumerate(options) if i != key],
                   "question": "",
                   "cluster": row["cluster"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--survey", default="mmlu_pro",
                        help="basename under data/external/survey (default: mmlu_pro)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    path = SURVEY_DIR / f"{args.survey}.jsonl.gz"
    if not path.exists():
        path = SURVEY_DIR / f"{args.survey}.jsonl"
    if not path.exists():
        raise SystemExit(f"no vendored survey file for {args.survey!r} in {SURVEY_DIR}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with args.out.open("w", encoding="utf-8") as handle:
        for row in convert(path):
            handle.write(json.dumps(row) + "\n")
            k = 1 + len(row["distractors"])
            counts[k] = counts.get(k, 0) + 1
    total = sum(counts.values())
    shape = ", ".join(f"{n} with {k} options" for k, n in sorted(counts.items(),
                                                                key=lambda kv: -kv[1]))
    print(f"wrote {args.out}: {total} rows ({shape})")


if __name__ == "__main__":
    main()
