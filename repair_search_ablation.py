#!/usr/bin/env python3
"""How a repair searches decides whether it repairs anything.

``mcq_audit``'s four-coordinate repair draws a target rank for the key under
every ordering it measures and searches for an option set that meets them all.
Two choices inside that sentence turn out to matter more than the specification
does:

*How it searches.* ``sample`` redraws the whole option set and keeps the closest
of forty-eight candidates. ``nearest`` walks one distractor's written style and
roundness at a time and keeps the closest set it reaches. ``guided`` walks the
same way but, when it cannot reach the targets, hands back a plain redraw
instead of its near-miss. ``construct`` does not search at all: written length
and significant digits are counts over the distractors, so it enumerates each
distractor's possible renderings and *assigns* them, which meets both targets
exactly wherever the item admits them and reports honestly when it does not.

*Where the targets come from.* ``uniform`` draws each target uniformly, which is
the obvious reading of "uniformise the key's rank". ``calibrated`` runs the
repair on a sample first to see where this file's items can actually land, then
draws against that.

The cell worth the script is nearest/uniform. It meets its targets two to three
times as often as the sampler and leaves a *worse* file. A target can only be
met where the arithmetic allows one, and every item that cannot reach its target
fails toward the same few ranks, so a search good enough to insist piles the
keys up there. The bound of Section 2 is about the realised marginal, not about
the target distribution, which is why a higher hit rate can mean a worse repair.
Two fixes within the searching frame follow: let the misses be unguided
(``guided``), and draw the targets against what the file can actually reach
(``calibrated``). Neither fully works, because reachability is a property of
each item. ``construct`` leaves the frame: with each item's reachable set in
hand the target is drawn from it, so there are no misses to be biased.

    python3 repair_search_ablation.py --seeds 5

Writes ``results/repair_search_ablation.json``: per cell, the audit verdict, the
simultaneous lower bound on each surface coordinate, the share of first-drawn
targets met, and the widest realised rank share.
"""
import argparse
import json
import statistics
from pathlib import Path

from mcq_audit import SURFACE_ORDERINGS, apply_repair, audit, build_items, read_rows

CELLS = ([(search, draw) for search in ("sample", "nearest", "guided")
          for draw in ("uniform", "calibrated")]
         + [("construct", "frontier")])


def one_cell(rows, items, k, args, search, draw, seed):
    repaired_rows, changed, _, _, hit = apply_repair(
        rows, items, args.distractor_field, seed, k=k, two_channel=True,
        search=search, calibrate=draw == "calibrated")
    repaired_items, _ = build_items(repaired_rows, args.key_field, args.distractor_field,
                                    args.cluster_field, args.question_field, k)
    report = audit(repaired_items, args.reps, k)
    orderings = report["surface_channels"]["orderings"]
    return {
        "seed": seed, "n_repaired": changed, "coordinates_firing": report["coordinates_firing"],
        "fired": {"numeric rule family": bool(report["numeric_family"]
                                              and report["numeric_family"]["leaks"]),
                  "surface rule family": bool(report["surface_family"]
                                              and report["surface_family"]["leaks"]),
                  **{name: bool(orderings[name]["numeric items"]["leaks"])
                     for name in SURFACE_ORDERINGS}},
        "value_credit": report["numeric_family"]["max_geometry_credit"],
        "bound": {name: orderings[name]["numeric items"]["credit_bound"]["credit_lower_bound"]
                  for name in SURFACE_ORDERINGS},
        "first_draw_met": hit["first_draw"],
        "assigned_met": hit["assigned"],
        "hit_rate_by_target_rank": hit["hit_rate_by_target_rank"],
        "hit_rate_spread": {name: (max(v for v in row if v is not None)
                                   - min(v for v in row if v is not None))
                            if any(v is not None for v in row) else None
                            for name, row in hit["hit_rate_by_target_rank"].items()},
        "widest_realised_rank": {name: max(hit["realised_rank_share"][name])
                                 for name in SURFACE_ORDERINGS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", type=Path, default=Path("data/bixbench.jsonl"))
    parser.add_argument("--key-field", default="ideal")
    parser.add_argument("--distractor-field", default="distractors")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--cluster-field", default="capsule_uuid")
    parser.add_argument("--n-options", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=20260919)
    parser.add_argument("--reps", type=int, default=4000)
    parser.add_argument("--output", type=Path,
                        default=Path("results/repair_search_ablation.json"))
    args = parser.parse_args()

    rows = read_rows(args.jsonl)
    items, k = build_items(rows, args.key_field, args.distractor_field,
                           args.cluster_field, args.question_field, args.n_options)
    seeds = [args.first_seed + i for i in range(args.seeds)]
    print(f"{len(items)} items, {k} options, {len(seeds)} seeds x {len(CELLS)} cells")

    cells = []
    for search, draw in CELLS:
        runs = [one_cell(rows, items, k, args, search, draw, seed) for seed in seeds]
        summary = {
            "search": search, "draw": draw,
            "leaks": sum(bool(run["coordinates_firing"]) for run in runs),
            "n_seeds": len(runs),
            "bound_mean": {name: statistics.fmean(run["bound"][name] for run in runs)
                           for name in SURFACE_ORDERINGS},
            "bound_worst": {name: max(run["bound"][name] for run in runs)
                            for name in SURFACE_ORDERINGS},
            "first_draw_met_mean": {name: statistics.fmean(run["first_draw_met"][name]
                                                           for run in runs)
                                    for name in SURFACE_ORDERINGS},
            # For the constructive repair the first-draw rate is low by design --
            # it does not ask for a uniformly drawn rank, it asks for one the item
            # can reach -- so the rate that means anything there is this one.
            "assigned_met_mean": {name: statistics.fmean(run["assigned_met"][name]
                                                         for run in runs)
                                  for name in SURFACE_ORDERINGS},
            "widest_realised_mean": {name: statistics.fmean(run["widest_realised_rank"][name]
                                                            for run in runs)
                                     for name in SURFACE_ORDERINGS},
            "hit_rate_spread_mean": {name: statistics.fmean(run["hit_rate_spread"][name]
                                                            for run in runs
                                                            if run["hit_rate_spread"][name]
                                                            is not None)
                                     for name in SURFACE_ORDERINGS},
            "runs": runs,
        }
        cells.append(summary)
        met = " ".join(f"{summary['first_draw_met_mean'][n]:4.0%}" for n in SURFACE_ORDERINGS)
        bound = " ".join(f"{summary['bound_worst'][n]:+6.1%}" for n in SURFACE_ORDERINGS)
        wide = " ".join(f"{summary['widest_realised_mean'][n]:5.1%}" for n in SURFACE_ORDERINGS)
        spread = " ".join(f"{summary['hit_rate_spread_mean'][n]:4.0%}"
                          for n in SURFACE_ORDERINGS)
        kept = " ".join(f"{summary['assigned_met_mean'][n]:4.0%}"
                        for n in SURFACE_ORDERINGS)
        print(f"  {search:9s} {draw:10s} leaks {summary['leaks']}/{len(runs)}   "
              f"drawn {met}   assigned {kept}   spread {spread}   "
              f"worst bound {bound}   widest {wide}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "file": str(args.jsonl), "n_options": k, "n_items": len(items),
        "chance": 1.0 / k, "reps": args.reps, "seeds": seeds,
        "orderings": list(SURFACE_ORDERINGS), "cells": cells}, indent=2) + "\n",
        encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
