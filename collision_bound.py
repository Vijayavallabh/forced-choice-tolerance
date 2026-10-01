#!/usr/bin/env python3
r"""How far Proposition~\ref{prop:writers}(iii)'s i.i.d.\ expression can sit from the distinct draw.

Part (iii) prices a writer that draws the key and its $k-1$ distractors i.i.d.\
from the key marginal $\kappa$; part (i)'s writer conditions that same draw on
the $k$ values being distinct. For any event $B$ -- here, that a solver ranking
options by a score picks the key -- and the event $A$ that the draw is distinct,
$\Pr(B)=\Pr(B\mid A)\Pr(A)+\Pr(B\mid A^c)\Pr(A^c)$, so
$|\Pr(B\mid A)-\Pr(B)|=\Pr(A^c)\,|\Pr(B\mid A^c)-\Pr(B\mid A)|\le\Pr(A^c)$.
The bound is the collision probability of $k$ i.i.d.\ draws from the item's
subject pool, which this computes exactly for every MMLU numeric item, with
the narrower probability that some distractor repeats the key beside it.

    python3 collision_bound.py
"""
import argparse
import json
import pathlib
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from repair_frontier import numeric_rows


def all_distinct(probs, k):
    """Probability that k i.i.d. draws from ``probs`` are pairwise distinct: k! e_k(p)."""
    e = np.zeros(k + 1)
    e[0] = 1.0
    for p in probs:                               # elementary symmetric polynomials, one value at a time
        e[1:] = e[1:] + p * e[:-1]
    return float(np.prod(np.arange(1, k + 1)) * e[k])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", default="build/mmlu_questions.jsonl")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--output", default="results/collision_bound.json")
    args = ap.parse_args()
    rows = numeric_rows(args.jsonl)
    by_subject = defaultdict(list)
    for row in rows:
        by_subject[row.get("cluster") or ""].append(str(row["ideal"]).strip())
    collide, repeat_key = [], []
    subjects = {}
    for subject, keys in sorted(by_subject.items()):
        counts = Counter(keys)
        probs = np.array(list(counts.values()), float) / len(keys)
        p_collide = 1.0 - all_distinct(probs, args.k)
        mass = {v: c / len(keys) for v, c in counts.items()}
        # for an item whose key is y: some distractor equals y
        per_item = [1.0 - (1.0 - mass[y]) ** (args.k - 1) for y in keys]
        collide += [p_collide] * len(keys)
        repeat_key += per_item
        subjects[subject] = {"n_items": len(keys), "distinct_keys": len(counts),
                             "p_collision": p_collide, "p_distractor_repeats_key_mean": float(np.mean(per_item))}
    report = {"n_items": len(rows), "k": args.k, "subjects": subjects,
              "item_weighted": {"p_collision_mean": float(np.mean(collide)),
                                "p_collision_median": float(np.median(collide)),
                                "p_distractor_repeats_key_mean": float(np.mean(repeat_key)),
                                "p_distractor_repeats_key_median": float(np.median(repeat_key))}}
    w = report["item_weighted"]
    print(f"{len(rows)} items: P(collision) mean {100 * w['p_collision_mean']:.1f}%, median "
          f"{100 * w['p_collision_median']:.1f}%; P(a distractor repeats the key) mean "
          f"{100 * w['p_distractor_repeats_key_mean']:.1f}%, median {100 * w['p_distractor_repeats_key_median']:.1f}%")
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
