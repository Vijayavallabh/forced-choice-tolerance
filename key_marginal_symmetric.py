#!/usr/bin/env python3
r"""\textsc{key marginal} rebuilt so that it is the writer Proposition~\ref{prop:writers}(i) prices.

The arm in Table~\ref{tab:frontier} removed from each item's pool every value
equal to that item's key *or to one of its human distractors*. Reading the
human distractors makes the draw depend on the item, and its exact population
$\Gamma$ (``key_marginal_gamma.py``) is $+7.1$ points, not zero. This rebuilds it
on the same $464$ aligned items with the only dependence any assembler of
distinct options must have: the key's own value is removed, and three values are
drawn from the rest of the subject's keys, weighted by multiplicity, without
repeats. Its $\Gamma$ is computed exactly for the file it writes (the same
Monte Carlo over presented sets with exact posteriors), so what its difficulty
column prices is stated rather than assumed.

    python3 key_marginal_symmetric.py
"""
import argparse
import json
import pathlib
import random
from collections import Counter, defaultdict

import numpy as np

from key_marginal_gamma import estimate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aligned", default="build/mmlu_frontier_released_matched_aligned.jsonl")
    ap.add_argument("--jsonl", default="build/mmlu_questions.jsonl")
    ap.add_argument("--out", default="build/mmlu_frontier_key_marginal_symmetric_aligned.jsonl")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--report", default="results/key_marginal_symmetric.json")
    ap.add_argument("--cost", action="store_true",
                    help="read the rebuilt arm's difficulty (frontier_validity.py's output) into the report")
    args = ap.parse_args()
    if args.cost:
        report = json.loads(pathlib.Path(args.report).read_text())
        report["cost"] = cost()
        pathlib.Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        lv = report["cost"]["level"]
        print(f"covering level {100 * lv['level_for_95']:.2f}% on {lv['n_clusters']} subjects "
              f"({lv['n_effective']:.1f} effective)")
        for model, arms in report["cost"]["models"].items():
            print(model.split("/")[-1] + ": " + "   ".join(
                f"{a} {v['points']:+.1f} [{v['covering'][0]:+.1f},{v['covering'][1]:+.1f}]" for a, v in arms.items()))
        return

    items = [json.loads(l) for l in open(args.aligned)]
    # The pool is the file's OWN keys, subject by subject. repair_frontier drew from
    # the keys of all 635 numeric items, of which the file keeps 464; a drawn value
    # that is no item's key in the file then marks itself as a distractor, which
    # alone is worth +6.2 points of Gamma on these items. Proposition 2(i) is about
    # the file's own key marginal, and so is this draw.
    pools = defaultdict(list)
    for row in items:
        pools[row["cluster"]].append(str(row["ideal"]).strip())
    rng = random.Random(args.seed)
    out, dropped = [], []
    for row in items:
        key = str(row["ideal"]).strip()
        pool = [v for v in pools[row["cluster"]] if v != key]
        chosen, seen = [], {key}
        for _ in range(4000 if len(set(pool)) >= 3 else 0):
            v = rng.choice(pool)
            if v not in seen:
                seen.add(v)
                chosen.append(v)
            if len(chosen) == 3:
                break
        if len(chosen) < 3:
            dropped.append(row["cluster"])      # a subject with fewer than four distinct keys
            continue
        out.append({**row, "distractors": chosen})
    pathlib.Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out))

    # the exact Gamma of the file just written, subject by subject
    by_subject = defaultdict(list)
    for row in out:
        by_subject[row["cluster"]].append(str(row["ideal"]).strip())
    grng = np.random.default_rng(args.seed)
    total, n = 0.0, 0
    per = {}
    for subject, keys in sorted(by_subject.items()):
        counts = Counter(pools[subject])
        item_pools = [{v: c for v, c in counts.items() if v != k} for k in keys]
        got = estimate(keys, item_pools, grng, args.samples)
        per[subject] = {"n_items": len(keys), "gamma": got[0], "mc_se": got[1]}
        total += len(keys) * got[0]
        n += len(keys)
    report = {"out": args.out, "n_items": len(out), "dropped_items": len(dropped),
              "dropped_subjects": sorted(set(dropped)), "gamma_item_weighted": total / n, "subjects": per}
    pathlib.Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out} ({len(out)} items, {len(dropped)} dropped in {sorted(set(dropped))}); "
          f"exact Gamma, item-weighted {total / n:+.2f} points")
    for s, e in per.items():
        print(f"   {s[:36]:36s} n={e['n_items']:3d}  {e['gamma']:+6.2f}")



def cost(path="results/frontier_validity_symmetric.json", reps=3000, bootstrap=2000, seed=20260924,
         validity_reps=4000, validity_seed=20260919):
    """The rebuilt arm's price, beside the arm as built, read at the level that covers this shape."""
    import frontier_calibrated as fc

    doc = json.loads(pathlib.Path(path).read_text())
    first = next(iter(doc["models"]))
    clusters = doc["clusters"]
    table = np.array([[sum(np.mean(h) for h, c in zip(doc["models"][first]["released"]["hits"], clusters) if c == cl),
                       sum(1 for c in clusters if c == cl)] for cl in sorted(set(clusters))])
    level = fc.covering_level(table, reps, bootstrap, seed)
    out = {"level": level, "models": {}}
    for model in doc["models"]:
        out["models"][model] = {}
        for arm in ("key-marginal", "key-marginal-near", "key-marginal-symmetric"):
            point, drawn = fc.difficulty_draws(doc, model, arm, validity_reps, validity_seed)
            out["models"][model][arm] = {"points": point,
                                         "nominal_95": [float(np.quantile(drawn, 0.025)), float(np.quantile(drawn, 0.975))],
                                         "covering": fc.quantiles(drawn, level["level_for_95"])}
    return out


if __name__ == "__main__":
    main()
