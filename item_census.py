#!/usr/bin/env python3
r"""Every item and cluster count the paper quotes, from one filter, in one table.

Three different subsets of the same file appear in the paper and each is the
right one for its instrument, but the counts drift across sections unless they
are laid out together:

    released    rows in the file as shipped
    modal k     rows carrying the file's modal option count
    numeric     of those, the ones whose k options are distinct numbers, which
                is where a rank is defined and so where the survey and the
                decomposition apply
    feasible    of those, the ones admitting an exchangeable rewriting in their
                own written style, which is where Table 1's three-file blocks
                are cut so every row holds the same items

Clusters are counted at the numeric and feasible stages, because the bootstrap
resamples clusters and its width follows that count and not the item count.

    python3 item_census.py --output results/item_census.json
"""
import argparse
import json
from pathlib import Path

from mcq_audit import build_items, read_rows

FILES = [
    ("MMLU", "build/mmlu.jsonl", "build/mmlu_exchangeable.jsonl", 4),
    ("MedMCQA", "build/medmcqa.jsonl", "build/medmcqa_exchangeable.jsonl", 4),
    ("MMLU-Pro", "build/mmlu_pro.jsonl", "build/mmlu_pro_exchangeable.jsonl", 10),
    ("BixBench v1.5", "build/bixbench_v15.jsonl", None, 4),
    ("AQuA-RAT", "build/aqua_rat.jsonl", None, 5),
]


def stage(path, k):
    """(rows, modal-k items, numeric items, clusters over the numeric items)."""
    rows = read_rows(Path(path))
    items, found = build_items(rows, "ideal", "distractors", "cluster", "question", k)
    numeric = [item for item in items if item["rank"] is not None]
    clusters = {item["cluster"] for item in numeric}
    return len(rows), len(items), len(numeric), len(clusters), found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="results/item_census.json")
    args = ap.parse_args()

    census = []
    for name, released, feasible, k in FILES:
        if not Path(released).exists():
            print(f"skipping {name}: {released} is absent")
            continue
        n_rows, n_modal, n_numeric, n_clusters, found = stage(released, k)
        entry = {"benchmark": name, "path": released, "k": found,
                 "n_released_rows": n_rows, "n_modal_k": n_modal,
                 "n_numeric": n_numeric, "n_numeric_clusters": n_clusters}
        if feasible and Path(feasible).exists():
            _, _, n_feasible, n_feasible_clusters, _ = stage(feasible, k)
            entry.update({"n_feasible": n_feasible,
                          "n_feasible_clusters": n_feasible_clusters})
        census.append(entry)
        line = (f"{name:<15} k={found:<3} released {n_rows:>6}  modal-k {n_modal:>6}  "
                f"numeric {n_numeric:>5} in {n_clusters:>3} clusters")
        if "n_feasible" in entry:
            line += (f"  feasible {entry['n_feasible']:>5} in "
                     f"{entry['n_feasible_clusters']:>3}")
        print(line)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"files": census}, indent=2),
                                 encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
