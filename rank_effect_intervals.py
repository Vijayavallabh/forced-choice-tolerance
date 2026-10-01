#!/usr/bin/env python3
r"""Each single-token solver's rank effect, with an interval on it.

\S4 reports the controlled probe as a range -- every solver between $-7.8$ and
$+3.0$ points -- and a range's endpoint is not a finding until someone says
which solver it is and whether it is distinguishable from zero. This says both.

The quantity is the one Figure~\ref{fig:solvers} plots: stem-withheld accuracy
on the \textsc{repair} arm minus the \textsc{placebo} arm, over each file's
numeric items. Placebo and repair are drawn by the same generator from the same
item and differ only in where the key's rank is put, so a negative value is a
solver scoring \emph{worse} once the key stops being where the released file
puts it -- the footprint of collecting the rank channel.

Paired at the item, bootstrapped over clusters, which is how every other
interval in the paper is formed.

    python3 rank_effect_intervals.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
from collections import defaultdict

import numpy as np

PANELS = [("BixBench v1.5", ["results/probe_bix_placebo"], 4),
          ("MMLU-Pro", ["results/probe_mmlupro_placebo"], 10),
          ("MMLU", ["results/probe_mmlu_placebo"], 4)]


def paired(path):
    """Per-item (repaired - placebo) differences and their cluster, stem withheld."""
    per_item = defaultdict(lambda: {"placebo_stemless": [], "repaired_stemless": []})
    cluster, model = {}, None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            model = model or row["model"]
            if not row["is_numeric"] or row["arm"] not in (
                    "placebo_stemless", "repaired_stemless"):
                continue
            per_item[row["question_id"]][row["arm"]].append(row["correct"])
            cluster[row["question_id"]] = row["cluster"]
    by_cluster = defaultdict(list)
    for item, arms in per_item.items():
        if not (arms["placebo_stemless"] and arms["repaired_stemless"]):
            continue
        by_cluster[cluster[item]].append(
            float(np.mean(arms["repaired_stemless"])) - float(np.mean(arms["placebo_stemless"])))
    return model, by_cluster


def bootstrap(by_cluster, reps, seed):
    keys = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    point = 100.0 * float(np.mean([v for k in keys for v in by_cluster[k]]))
    draws = []
    for _ in range(reps):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        flat = [v for i in picked for v in by_cluster[keys[i]]]
        draws.append(100.0 * float(np.mean(flat)))
    return point, [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], len(keys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--output", default="results/rank_effect_intervals.json")
    args = ap.parse_args()

    report = {"reps": args.reps, "panels": {}}
    for label, dirs, k in PANELS:
        paths = sorted(p for d in dirs if pathlib.Path(d).exists()
                       for p in pathlib.Path(d).glob("*.jsonl.gz"))
        if not paths:
            print(f"skipping {label}: no dumps"); continue
        rows, seen = [], set()
        for path in paths:
            model, by_cluster = paired(path)
            if model in seen or not by_cluster:
                continue
            seen.add(model)
            point, ci, n_clusters = bootstrap(by_cluster, args.reps, args.seed)
            rows.append({"model": model, "rank_effect": point, "ci95": ci,
                         "n_clusters": n_clusters,
                         "clears": ci[1] < 0 or ci[0] > 0})
        rows.sort(key=lambda r: r["rank_effect"])
        report["panels"][label] = {"n_options": k, "models": rows,
                                   "n_clearing": sum(r["clears"] for r in rows)}
        print(f"=== {label} (k={k}), {len(rows)} solvers")
        for r in rows:
            print(f"   {r['model']:<42} {r['rank_effect']:+6.2f} "
                  f"[{r['ci95'][0]:+6.2f},{r['ci95'][1]:+6.2f}]"
                  + ("  clears" if r["clears"] else ""))
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
