#!/usr/bin/env python3
r"""Is v1.0's non-numeric channel a surface cue?

Table~\ref{tab:gridv10} finds BixBench's own template collecting on v1.0's
$137$ items whose options are not all numbers, where no rank rule applies. The
first suspects are the cues that need no reading: the longest option, the
shortest, and the one sharing the most words with the others (the ``centroid''
a writer drifts toward when every distractor is an edit of the key). Each rule
is scored with ties broken uniformly at random -- averaged exactly over the
tie, since breaking ties to the first option would hand the rule whatever
position the file's key happens to take -- and read over the $25\%$ chance with
a cluster bootstrap over capsules.

    python3 surface_cues_v10.py
"""
import argparse
import json
import pathlib
import re

import numpy as np

from mcq_audit import rank_of

WORD = re.compile(r"[A-Za-z0-9]+")


def rule_scores(options):
    """Per rule, the score each option gets; the rule picks the arg-max."""
    lengths = [len(o) for o in options]
    words = [set(w.lower() for w in WORD.findall(o)) for o in options]
    overlap = [sum(len(words[i] & words[j]) for j in range(len(options)) if j != i)
               for i in range(len(options))]
    return {"longest": lengths, "shortest": [-x for x in lengths], "most shared words": overlap}


def expected_hit(scores, key=0):
    """Probability the rule picks the key when ties are broken uniformly."""
    top = max(scores)
    tied = [i for i, s in enumerate(scores) if s == top]
    return (1.0 / len(tied)) if key in tied else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="build/bixbench_v10.jsonl")
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--level", type=float, default=0.96)
    ap.add_argument("--output", default="results/surface_cues_v10.json")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.items)]
    other = [r for r in rows if rank_of([r["ideal"], *r["distractors"]]) is None]
    clusters = sorted({r["cluster"] for r in other})
    report = {"items": args.items, "n_other": len(other), "n_clusters": len(clusters), "level": args.level,
              "rules": {}}
    rng = np.random.default_rng(0)
    for rule in ("longest", "shortest", "most shared words"):
        hits = np.array([expected_hit(rule_scores([str(r["ideal"]), *map(str, r["distractors"])])[rule])
                         for r in other])
        by_c = {c: hits[[i for i, r in enumerate(other) if r["cluster"] == c]] for c in clusters}
        draws = []
        for _ in range(args.bootstrap):
            take = rng.choice(clusters, size=len(clusters))
            draws.append(100.0 * np.concatenate([by_c[c] for c in take]).mean() - 25.0)
        lo, hi = np.quantile(draws, [(1 - args.level) / 2, (1 + args.level) / 2])
        report["rules"][rule] = {"accuracy": 100.0 * float(hits.mean()),
                                 "over_chance": 100.0 * float(hits.mean()) - 25.0,
                                 "interval": [float(lo), float(hi)]}
        e = report["rules"][rule]
        print(f"{rule:18s} {e['accuracy']:5.1f}%  over chance {e['over_chance']:+5.1f} "
              f"[{e['interval'][0]:+5.1f},{e['interval'][1]:+5.1f}]  ({len(other)} items, {len(clusters)} capsules)")
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
