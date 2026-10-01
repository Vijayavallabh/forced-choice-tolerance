#!/usr/bin/env python3
r"""A rank repair guarantees a marginal, and a marginal does not survive subsetting.

Two numbers in this paper describe the same thing and disagree. On the whole
repaired MMLU file the audited-rank learner scores $2.9$ points *below* its
clean control; on the 594 items every row of Table 1 shares it scores $1.9$
*above*. Both are right, and the gap is the finding rather than the error: the
repair draws each item's target rank uniformly, but the rank an item *reaches*
is the rank drawn filtered by what its own options can render, so uniformity
holds over the file and not over an arbitrary subset of it.

The subset in question is the one that admits an exchangeable rewriting, which
needs $k-1$ distinct renderable values on the key's other side. That is a
condition on the key's magnitude, so the filter is correlated with exactly the
thing the repair randomises.

This measures both halves: the key-rank histogram of each file whole and cut,
and what the cut costs the guarantee.

    python3 subset_guarantee.py
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from mcq_audit import build_items, parse_number, read_rows

FILES = [
    ("MMLU", 4, "cluster",
     [("released, whole", "build/mmlu.jsonl"),
      ("released, feasible", "build/mmlu_matched_released.jsonl"),
      ("rank uniform, whole", "results/repaired/mmlu_repaired_multi.jsonl.gz"),
      ("rank uniform, feasible", "build/mmlu_matched_repaired.jsonl")]),
    ("MedMCQA", 4, "cluster",
     [("released, whole", "build/medmcqa.jsonl"),
      ("released, feasible", "build/medmcqa_matched_released.jsonl"),
      ("rank uniform, whole", "build/medmcqa_repaired.jsonl"),
      ("rank uniform, feasible", "build/medmcqa_matched_repaired.jsonl")]),
    ("MMLU-Pro", 10, "cluster",
     [("released, whole", "build/mmlu_pro_numeric.jsonl"),
      ("released, feasible", "build/mmlu_pro_matched_released.jsonl"),
      ("rank uniform, whole", "build/mmlu_pro_repaired.jsonl"),
      ("rank uniform, feasible", "build/mmlu_pro_matched_repaired.jsonl")]),
]


def key_rank_histogram(path, k, cluster_field):
    """Share of items at each rank of the keyed value, over the distinct-numeric ones."""
    items, _ = build_items(read_rows(Path(path)), "ideal", "distractors",
                           cluster_field, "question", k)
    counts, total = Counter(), 0
    for item in items:
        values = [parse_number(o) for o in item["options"]]
        if any(v is None for v in values) or len({float(v) for v in values}) != k:
            continue
        order = sorted(range(k), key=lambda i: float(values[i]))
        counts[order.index(0)] += 1
        total += 1
    if not total:
        return 0, []
    return total, [100 * counts[rank] / total for rank in range(k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="results/subset_guarantee.json")
    args = ap.parse_args()

    report = {}
    for benchmark, k, cluster_field, paths in FILES:
        block = {"k": k, "files": {}}
        for label, path in paths:
            if not Path(path).exists():
                print(f"  (missing {path})")
                continue
            n, histogram = key_rank_histogram(path, k, cluster_field)
            if not n:
                continue
            block["files"][label] = {
                "path": path, "n_items": n, "histogram": histogram,
                "plug_in_credit": max(histogram) - 100 / k,
            }
            print(f"{benchmark:<9} {label:<24} n={n:5d}  "
                  + " ".join(f"{x:4.1f}" for x in histogram)
                  + f"   max - 1/k = {max(histogram) - 100 / k:+.1f}")
        report[benchmark] = block
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
