#!/usr/bin/env python3
r"""Do BixBench's two releases share a key-rank law? They do not.

\S3 measures the rank channel on \textbf{v1.5}, the current release, and \S4
fits \eqref{eq:split} to the two published no-data runs, which exist only for
\textbf{v1.0} because only v1.0's run files record the order the model was
shown. The obvious question is whether those are the same file.

They are not, and the direction matters. This computes the key-rank law on both
-- v1.5 from the released items, v1.0 from the ``choices`` column of the
published runs -- and intersects them on the option sets they share.

    python3 release_drift.py
"""
import argparse
import collections
import json
import pathlib

import choice_model as cm
from mcq_audit import parse_number

RUNS = "data/external/zero_shot_v10/bixbench_llm_baseline_refusal_False_mcq_gpt-4o_1.0.csv"


def law(ranks, k=4):
    counts = collections.Counter(ranks)
    n = sum(counts.values())
    return n, [100.0 * counts[i] / n for i in range(k)]


def v15_items(path):
    r"""[(sorted option values, key rank)] over items with k distinct numbers.

    A list and not a dict keyed by the sorted values: two of v1.5's items
    present the same four numbers, and collapsing them put the law over $104$
    items, $50.96\%$ second-smallest, where \S3 reads all $105$ at $51.4\%$ ---
    which is how one ceiling came to be printed as both $+26.0$ and $+26.4$.
    """
    out = []
    for line in open(path):
        row = json.loads(line)
        values = [parse_number(str(x)) for x in [row["ideal"]] + list(row["distractors"])]
        if any(v is None for v in values) or len(set(values)) != len(values):
            continue
        out.append((tuple(sorted(values)),
                    sorted(range(len(values)), key=lambda i: values[i]).index(0)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v15", default="build/bixbench_v15.jsonl")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--output", default="results/release_drift.json")
    args = ap.parse_args()

    pairs, skipped = cm.pairs_from_v10(pathlib.Path(args.runs))
    n10, p10 = law([x["key_rank"] for x in pairs])
    listed15 = v15_items(args.v15)
    n15, p15 = law([rank for _, rank in listed15])
    items15 = dict(listed15)

    # the option sets both releases carry, which is the only place a disagreement
    # would be about labelling rather than about which items were kept
    shared, agree = 0, 0
    for key, rank in _v10_sets(args.runs).items():
        if key in items15:
            shared += 1
            agree += int(rank == items15[key])

    report = {
        "v1_0": {"source": args.runs, "n_items": n10, "key_by_rank": p10,
                 "max_rank_share": max(p10), "rank_ceiling_points": max(p10) - 25.0,
                 "skipped": skipped},
        "v1_5": {"source": args.v15, "n_items": n15, "key_by_rank": p15,
                 "max_rank_share": max(p15), "rank_ceiling_points": max(p15) - 25.0},
        "shared_option_sets": shared, "shared_agreeing_on_key_rank": agree,
        "note": ("the releases carry largely different items; where an option set "
                 "appears in both the key is almost always the same one, so the "
                 "laws differ because the populations differ and not because a "
                 "key moved")}
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"v1.0 published runs: n={n10}  key rank {[round(x,1) for x in p10]}  "
          f"ceiling {max(p10)-25:+.1f} points")
    print(f"v1.5 released items: n={n15}  key rank {[round(x,1) for x in p15]}  "
          f"ceiling {max(p15)-25:+.1f} points")
    print(f"option sets in both: {shared}; key rank agrees on {agree}")
    print(f"wrote {args.output}")


def _v10_sets(path):
    """(sorted values) -> key rank, from the published run's presented order."""
    import ast
    import csv
    out = {}
    for row in csv.DictReader(open(path)):
        options = [o.split(") ", 1)[1] if ") " in o else o
                   for o in ast.literal_eval(row["choices"])]
        values = [parse_number(o.strip()) for o in options]
        if any(v is None for v in values) or len(set(values)) != len(values):
            continue
        letters = "ABCDEFGH"[:len(options)]
        if row["target"] not in letters:
            continue
        key = letters.index(row["target"])
        out[tuple(sorted(values))] = sorted(
            range(len(values)), key=lambda i: values[i]).index(key)
    return out


if __name__ == "__main__":
    main()
