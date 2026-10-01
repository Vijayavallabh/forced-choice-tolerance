#!/usr/bin/env python3
r"""Table~\ref{tab:frontier} at the level that covers, not the level we asked for.

The paper corrects its intervals to a measured covering level where the arms sit
on $46$ and $60$ clusters, and leaves the frontier's at nominal $95\%$ --- and
the frontier is the table the repair conclusion rests on. Its twenty MMLU
subjects are worth fewer than five by Kish, which is the regime where a
percentile cluster bootstrap is least trustworthy, so correcting everything
except this table gets the risk allocation exactly backwards.

This measures the covering level for the frontier's own shape with
``bootstrap_calibration``'s machinery --- a beta-binomial fitted by moments to
the released arm's per-subject accuracies, files simulated at the frontier's
exact subject sizes, coverage read off at every nominal level --- and reprints
each row's leak there. It also reports the two-dominant-subject cut, since two
of the twenty subjects hold $282$ of the $464$ items.

It reprints the table's other two columns at the same level. The first version
reprinted only the set column, which left the difficulty column --- the one
that says what closing the channel costs --- and the one-option column at
nominal $95\%$ on the same twenty subjects: the asymmetry this script was
written to remove, kept on two of the table's three columns. Both are read off
the draws their own scripts print the nominal interval from, and each script's
nominal interval is reproduced before the covering one is trusted.

    python3 frontier_calibrated.py

Rows whose interval stops clearing zero are named, because the conclusion in
\S\ref{sec:repair} is a statement about which rows clear.
"""
import argparse
import json
import pathlib

import numpy as np

import bootstrap_calibration as bc
import frontier_validity as fv
import key_identity as ki
import learned_probe as lp

ARMS = ("released", "key-marginal", "key-marginal-near", "wrong-step",
        "rank-uniform", "exchangeable", "imitation")
FAMILY = "every feature"


def picks_for(path, k, steps, learning_rate, l2, seed):
    items, found = lp.build_items(lp.read_rows(pathlib.Path(path)), "ideal",
                                  "distractors", "cluster", "question", k)
    if found != k:
        raise SystemExit(f"{path}: found {found} options, expected {k}")
    x, clusters = lp.build_matrix(items)
    picks, tied = lp.cross_validate(x, clusters, steps, learning_rate, l2, seed)
    return picks, tied


def cluster_table(picks):
    hits, seen = {}, {}
    for cluster, hit in picks:
        seen[cluster] = seen.get(cluster, 0) + 1
        hits[cluster] = hits.get(cluster, 0) + int(hit)
    return np.array([[hits[c], seen[c]] for c in sorted(seen)], dtype=float)


def draws_at(picks, reps, seed):
    """``learned_probe``'s own cluster bootstrap, not a second one.

    Rolling our own here gave a nominal 95% interval 0.2 points from the one
    the shipped result file carries -- same picks, different RNG -- which would
    have put two intervals for one arm in the paper. Both levels are read off
    the same draws that produced ``cv_ci95``.
    """
    return np.array(lp.bootstrap(picks, reps, seed))


def quantiles(drawn, level):
    """``learned_probe``'s interval convention, not numpy's.

    It sorts the draws and indexes, ``draws[int(0.025 * n)]``; numpy's
    ``percentile`` interpolates between neighbours. On the same draws the two
    differ by about $0.06$ points here, which is enough to put two intervals
    for one arm in the paper -- the released row's upper end was $-0.57$ one
    way and $-0.64$ the other. Indexing is what the shipped result file used,
    so indexing is what both levels use.
    """
    order = np.sort(np.asarray(drawn))
    tail = (1.0 - level) / 2.0
    n = len(order)
    return [float(order[int(tail * n)]), float(order[int((1.0 - tail) * n)])]


def one_option_draws(path, k, family, steps, learning_rate, l2, reps, seed):
    """``key_identity.py``'s picks and bootstrap for one per-option family."""
    items, found = ki.build_items(ki.read_rows(pathlib.Path(path)), "ideal",
                                  "distractors", "cluster", "question", k)
    if found != k:
        raise SystemExit(f"{path}: found {found} options, expected {k}")
    x, clusters = ki.build_matrix(items)
    columns = (np.arange(len(ki.FEATURES)) if family == "written form" else
               np.asarray([ki.FEATURES.index(n) for n in ki.ROUNDNESS]))
    picks, _ = lp.cross_validate(x[:, :, columns], clusters, steps, learning_rate, l2)
    return lp.accuracy(picks), np.array(lp.bootstrap(picks, reps, seed))


def difficulty_draws(items_doc, model, arm, reps, seed):
    """``frontier_validity.py``'s paired cluster bootstrap, same data, same order."""
    block, clusters = items_doc["models"][model], items_doc["clusters"]
    per_cluster = {}
    for i, cluster in enumerate(clusters):
        per_cluster.setdefault(cluster, []).append(
            float(np.mean(block[arm]["hits"][i])) - float(np.mean(block["released"]["hits"][i])))
    point = 100.0 * float(np.mean([v for vs in per_cluster.values() for v in vs]))
    return point, np.array(fv.paired_bootstrap(per_cluster, reps, seed))


def covering_level(table, reps, bootstrap, seed):
    """Nominal level whose measured coverage reaches 95% at this shape."""
    rate, spread, sizes = bc.fit_beta_binomial(table)
    rng = np.random.default_rng(seed)
    below = []
    for _ in range(reps):
        hits, trials = bc.simulate(rate, spread, sizes, None, rng)
        drawn = bc.bootstrap_draws(hits, trials, bootstrap, rng)
        # Against the rate the file was *drawn with*, not the rate it came out
        # at. The bootstrap is centred on the latter by construction, so scoring
        # coverage against it reports 100% at every level -- which is what the
        # first version of this did, and the only reason it was caught is that
        # a percentile bootstrap on 4.8 effective clusters over-covering
        # perfectly is not a believable result.
        below.append(float(np.mean(drawn < rate)))
    curve = bc.coverage_curve(below)
    nominal = bc.LEVELS[np.argmin(np.abs(bc.LEVELS - 0.95))]
    return {"rate": rate, "beta_concentration": spread,
            "n_clusters": int(len(sizes)),
            "n_effective": bc.effective_clusters(sizes),
            "coverage_at_95": float(curve[np.argmin(np.abs(bc.LEVELS - 0.95))]),
            "level_for_95": bc.level_for(curve, 0.95),
            "nominal": float(nominal)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default="results/learned_probe_frontier.json")
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--identity", default="results/key_identity_frontier.json")
    ap.add_argument("--identity-reps", type=int, default=2000,
                    help="key_identity.py's own --reps, so its interval reproduces")
    ap.add_argument("--difficulty", default="results/frontier_validity_items.json")
    ap.add_argument("--validity", default="results/frontier_validity.json")
    ap.add_argument("--validity-reps", type=int, default=4000,
                    help="frontier_validity.py's own --reps")
    ap.add_argument("--validity-seed", type=int, default=20260919,
                    help="frontier_validity.py's own --seed")
    ap.add_argument("--large", action="append",
                    default=["results/frontier_validity_items_qwen72b.json",
                             "results/frontier_validity_items_llama70b.json"],
                    help="per-item files for further solvers (the 70B class), read at the "
                         "same level beside the table's three and kept out of its union")
    ap.add_argument("--output", default="results/frontier_calibrated.json")
    args = ap.parse_args()

    probe = json.load(open(args.probe, encoding="utf-8"))
    clean = probe["clean"][FAMILY]["max"]
    print(f"clean {FAMILY} worst case: {100 * clean:.2f}%  "
          f"(mean {100 * probe['clean'][FAMILY]['mean']:.2f}%, "
          f"{probe['clean'][FAMILY]['replicates']} replicates)\n")

    picks, tables = {}, {}
    for arm in ARMS:
        path = probe["files"][arm]["path"]
        got, tied = picks_for(path, probe["n_options"], args.steps,
                              args.learning_rate, args.l2, args.seed)
        picks[arm], tables[arm] = got, cluster_table(got)
        shipped = probe["files"][arm][FAMILY]["cv_accuracy"]
        got_accuracy = lp.accuracy(got)
        flag = "" if abs(got_accuracy - shipped) < 1e-9 else \
            f"  <-- differs from the shipped {100 * shipped:.2f}%"
        print(f"{arm:20s} cv {100 * got_accuracy:6.2f}%  ties {tied}{flag}")

    level = covering_level(tables["released"], args.reps, args.bootstrap,
                           args.seed)
    print(f"\nfrontier shape: {level['n_clusters']} subjects, "
          f"{level['n_effective']:.1f} effective; a nominal 95% interval covers "
          f"{100 * level['coverage_at_95']:.1f}%; "
          f"95% needs a nominal {100 * level['level_for_95']:.1f}%\n")

    report = {"clean_worst_case": clean, "calibration": level, "rows": {}}
    header = (f"{'operator':20s} {'leak':>7s} {'nominal 95%':>20s} "
              f"{'covering':>20s}  verdict")
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        drawn = 100.0 * (draws_at(picks[arm], args.bootstrap, args.seed) - clean)
        leak = 100.0 * (lp.accuracy(picks[arm]) - clean)
        nominal = quantiles(drawn, 0.95)
        covering = quantiles(drawn, level["level_for_95"])
        verdict = ("clears at both" if nominal[0] > 0 and covering[0] > 0 else
                   "clears at nominal only" if nominal[0] > 0 else
                   "clears at neither")
        report["rows"][arm] = {"leak": leak, "nominal_95": nominal,
                               "covering": covering, "verdict": verdict}
        print(f"{arm:20s} {leak:+7.2f} [{nominal[0]:+7.2f},{nominal[1]:+7.2f}] "
              f"  [{covering[0]:+7.2f},{covering[1]:+7.2f}]  {verdict}")

    # --- the one-option column, at the same level ---------------------------
    # key_identity's own picks and draws, so the nominal interval it shipped is
    # reproduced first; the table prints the better of its two families.
    ident = json.load(open(args.identity, encoding="utf-8"))
    print(f"\none option, at the same {100 * level['level_for_95']:.2f}%:")
    for arm in ARMS:
        best = None
        for family in ("written form", "roundness"):
            shipped = ident["files"][arm][family]
            worst = ident["clean"][family]["max"]
            got, drawn = one_option_draws(ident["files"][arm]["path"], ident["n_options"],
                                          family, ident["steps"], args.learning_rate,
                                          ident["l2"], args.identity_reps, args.seed)
            nominal = [float(drawn[int(0.025 * len(drawn))]),
                       float(drawn[int(0.975 * len(drawn))])]
            if abs(got - shipped["cv_accuracy"]) > 1e-9 or any(
                    abs(a - b) > 1e-9 for a, b in zip(nominal, shipped["cv_ci95"])):
                raise SystemExit(f"{arm} {family}: key_identity shipped "
                                 f"{shipped['cv_accuracy']:.4f} {shipped['cv_ci95']}, "
                                 f"recomputed {got:.4f} {nominal}")
            cell = {"family": family, "margin": 100.0 * (got - worst),
                    "nominal_95": [100.0 * (v - worst) for v in nominal],
                    "covering": [100.0 * (v - worst) for v in
                                 quantiles(drawn, level["level_for_95"])]}
            if best is None or cell["margin"] > best["margin"]:
                best = cell
        report["rows"][arm]["one_option"] = best
        print(f"{arm:20s} {best['margin']:+7.2f} [{best['nominal_95'][0]:+7.2f},"
              f"{best['nominal_95'][1]:+7.2f}]   [{best['covering'][0]:+7.2f},"
              f"{best['covering'][1]:+7.2f}]  ({best['family']})")

    # --- the difficulty column, at the same level ---------------------------
    items_doc = json.load(open(args.difficulty, encoding="utf-8"))
    shipped_val = json.load(open(args.validity, encoding="utf-8"))
    print(f"\ndifficulty given the question, at the same {100 * level['level_for_95']:.2f}%:")
    for arm in ARMS[1:]:
        per_model = {}
        for model in items_doc["models"]:
            point, drawn = difficulty_draws(items_doc, model, arm, args.validity_reps,
                                            args.validity_seed)
            nominal = [float(drawn[int(0.025 * len(drawn))]),
                       float(drawn[int(0.975 * len(drawn))])]
            shipped = shipped_val["models"][model][arm]["vs_reference"]
            if abs(point - shipped["points"]) > 1e-9 or any(
                    abs(a - b) > 1e-9 for a, b in zip(nominal, shipped["ci95"])):
                raise SystemExit(f"{arm} {model}: frontier_validity shipped "
                                 f"{shipped['points']:.4f} {shipped['ci95']}, "
                                 f"recomputed {point:.4f} {nominal}")
            per_model[model] = {"points": point, "nominal_95": nominal,
                                "covering": quantiles(drawn, level["level_for_95"])}
        cells = list(per_model.values())
        union = {"range": [min(c["points"] for c in cells), max(c["points"] for c in cells)],
                 "nominal_95": [min(c["nominal_95"][0] for c in cells),
                                max(c["nominal_95"][1] for c in cells)],
                 "covering": [min(c["covering"][0] for c in cells),
                              max(c["covering"][1] for c in cells)],
                 "contains_zero_at_every_solver": all(
                     c["covering"][0] <= 0 <= c["covering"][1] for c in cells),
                 "clears_at_every_solver": all(c["covering"][0] > 0 for c in cells)}
        report["rows"][arm]["difficulty"] = {"models": per_model, "union": union}
        print(f"{arm:20s} {union['range'][0]:+6.1f} to {union['range'][1]:+6.1f}  "
              f"[{union['nominal_95'][0]:+6.1f},{union['nominal_95'][1]:+6.1f}]   "
              f"[{union['covering'][0]:+6.1f},{union['covering'][1]:+6.1f}]"
              + ("  clears at every solver" if union["clears_at_every_solver"] else ""))

    # --- the same column for the 70B-class solvers -----------------------------
    # Is the cost a small-solver artefact? These solvers
    # are read exactly as the table's three are, each file's own nominal interval
    # reproduced first; they are reported beside the table's union, not folded
    # into it, because the union was fixed over the table's three first.
    large_docs = [json.load(open(p, encoding="utf-8")) for p in args.large
                  if pathlib.Path(p).exists()]
    if large_docs:
        print(f"\ndifficulty for the 70B-class solvers, at the same {100 * level['level_for_95']:.2f}%:")
    for arm in ARMS[1:]:
        per_model = {}
        for doc in large_docs:
            for model in doc["models"]:
                point, drawn = difficulty_draws(doc, model, arm, args.validity_reps,
                                                args.validity_seed)
                nominal = [float(drawn[int(0.025 * len(drawn))]),
                           float(drawn[int(0.975 * len(drawn))])]
                shipped = doc["models"][model][arm]["vs_reference"]
                if abs(point - shipped["points"]) > 1e-9 or any(
                        abs(a - b) > 1e-9 for a, b in zip(nominal, shipped["ci95"])):
                    raise SystemExit(f"{arm} {model}: its own file says "
                                     f"{shipped['points']:.4f} {shipped['ci95']}, "
                                     f"recomputed {point:.4f} {nominal}")
                per_model[model] = {"points": point, "nominal_95": nominal,
                                    "covering": quantiles(drawn, level["level_for_95"])}
        if per_model:
            report["rows"][arm]["difficulty_large"] = per_model
            print(f"{arm:20s} " + "   ".join(
                f"{m.split('/')[-1]} {c['points']:+5.1f} [{c['covering'][0]:+5.1f},{c['covering'][1]:+5.1f}]"
                for m, c in per_model.items()))

    clears_nominal = [a for a in ARMS if report["rows"][a]["nominal_95"][0] > 0]
    clears_covering = [a for a in ARMS if report["rows"][a]["covering"][0] > 0]
    report["clears_nominal"] = clears_nominal
    report["clears_covering"] = clears_covering
    lost = [a for a in clears_nominal if a not in clears_covering]
    print(f"\nclears at nominal 95%: {', '.join(clears_nominal) or 'none'}")
    print(f"clears at the covering level: {', '.join(clears_covering) or 'none'}")
    print(f"lost by calibrating: {', '.join(lost) or 'none'}")

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
