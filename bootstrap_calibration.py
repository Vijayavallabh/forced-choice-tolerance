#!/usr/bin/env python3
r"""Does the cluster bootstrap cover at the cluster counts this paper runs at?

Nearly every interval in this paper is a percentile cluster bootstrap over
between $37$ and $60$ groups: $46$ BixBench capsules, $60$ MMLU-Pro clusters,
$37$ at the frontier's feasible cut. That is few enough to ask whether the
interval means what it says, and asserting it does is not an answer.

So this measures it. The simulation is not invented: cluster sizes and the
spread of per-cluster accuracy are taken from the paper's own arms, so the
intra-cluster correlation is the one the real files have. A beta-binomial is
fitted to each arm by moments -- per-cluster success probability $q_c \sim
\mathrm{Beta}$, $y_c \sim \mathrm{Binom}(n_c, q_c)$ -- and files are then drawn
from it at a sweep of cluster counts, holding the true margin fixed at the
fitted mean. The bootstrap that runs on each simulated file is the same code
path the paper's arms use.

Two estimands, because the paper reports both:

  margin      one arm against a fixed chance baseline, as \S3's bounds are
  difference  one arm minus a second independent arm, as every margin over a
              clean control in \S4 is, formed by differencing paired draws

    python3 bootstrap_calibration.py --reps 2000

Reports, per estimand and cluster count, the share of simulated files whose
95% interval contains the value the file was drawn with.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
from collections import defaultdict

import numpy as np

RESULTS = pathlib.Path("results")
SHIPPED = RESULTS / "agentic_dumps"

# The arms whose geometry the simulation borrows: a dump of one rollout per line,
# the cluster field, and the k that fixes chance.
ARMS = [
    ("BixBench, 14B agent", "build/dumps/qwen14b_withheld_aware.jsonl", 4),
    ("MMLU-Pro, 32B agent", "build/dumps/qwen32b_mmlupro.jsonl", 10),
]
COUNTS = (20, 30, 37, 46, 60, 120)


def resolve(path):
    """The dump where the GPU run wrote it, else the gzipped copy that ships.

    Arms are dumped into ``build/dumps/``, which a clone does not have; the
    repository carries the same rollouts as ``results/agentic_dumps/*.jsonl.gz``.
    None when neither exists.
    """
    here = pathlib.Path(path)
    if here.exists():
        return here
    shipped = SHIPPED / (here.name + ".gz")
    return shipped if shipped.exists() else None


def cluster_table(path, arm="file"):
    """(successes, trials) per cluster for one arm of a dump."""
    opener = gzip.open if str(path).endswith(".gz") else open
    hits, seen = defaultdict(int), defaultdict(int)
    with opener(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("arm", "file") != arm:
                continue
            seen[r["cluster"]] += 1
            hits[r["cluster"]] += int(r["answer"] == r["gold"])
    return np.array([[hits[c], seen[c]] for c in sorted(seen)], dtype=float)


def fit_beta_binomial(table):
    """Method-of-moments beta for per-cluster success, from observed counts.

    The between-cluster variance an arm shows is part sampling and part real, so
    the sampling part is subtracted before the beta is fitted; otherwise the
    simulation would be handed more heterogeneity than the file has and the
    bootstrap would be flattered.
    """
    hits, trials = table[:, 0], table[:, 1]
    rate = hits.sum() / trials.sum()
    observed = float(np.var(hits / trials, ddof=1))
    within = float(np.mean(rate * (1 - rate) / trials))
    between = max(observed - within, 1e-9)
    spread = rate * (1 - rate) / between - 1.0
    if spread <= 0:                                   # no excess over binomial
        spread = 1e6
    return rate, max(spread, 1e-6), trials


def simulate(rate, spread, sizes, n_clusters, rng):
    """One file: cluster sizes resampled, per-cluster rate drawn from the beta.

    ``n_clusters=None`` keeps the arm's own sizes exactly, which is the cell that
    speaks to the interval the paper actually printed.
    """
    n = sizes.astype(int) if n_clusters is None else rng.choice(
        sizes, size=n_clusters, replace=True).astype(int)
    q = rng.beta(rate * spread, (1 - rate) * spread, size=len(n))
    return rng.binomial(n, q).astype(float), n.astype(float)


def bootstrap_draws(hits, trials, reps, rng):
    """The paper's percentile cluster bootstrap: resample clusters, not items."""
    idx = rng.integers(0, len(trials), size=(reps, len(trials)))
    return hits[idx].sum(axis=1) / trials[idx].sum(axis=1)


def effective_clusters(sizes):
    """Kish's effective count: what unequal cluster sizes leave you resampling.

    Sixty clusters of which one holds a quarter of the rollouts is not sixty
    draws' worth of independence, and the nominal count is what the paper had
    been reporting.
    """
    return float(sizes.sum() ** 2 / (sizes ** 2).sum())


LEVELS = np.arange(0.80, 0.999, 0.0025)


def coverage_curve(below):
    """Coverage at every nominal level at once.

    ``below`` is, per simulated file, the share of bootstrap draws falling under
    the value the file was drawn with. The truth sits inside the central
    ``level`` percentile interval exactly when that share is inside the matching
    central band, so one pass over the simulations calibrates every level.
    """
    below = np.asarray(below)
    lo = (1.0 - LEVELS) / 2.0
    return np.array([float(np.mean((below > a) & (below < 1.0 - a))) for a in lo])


def level_for(curve, target=0.95):
    """The nominal level whose actual coverage first reaches ``target``."""
    reached = np.nonzero(curve >= target)[0]
    return float(LEVELS[reached[0]]) if len(reached) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=2000, help="simulated files per cell")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--output", default="results/bootstrap_calibration.json")
    args = ap.parse_args()

    # Both arms or nothing. Skipping a missing dump used to write {"arms": []}
    # over the committed calibration, and arm_intervals.py then read every arm
    # at the nominal level without saying so.
    found = {path: resolve(path) for _, path, _ in ARMS}
    missing = [f"{path} (or {SHIPPED / (pathlib.Path(path).name + '.gz')})"
               for path, where in found.items() if where is None]
    if missing:
        raise SystemExit(f"refusing to write {args.output}: no dump for "
                         + "; ".join(missing))

    report = {"reps": args.reps, "bootstrap": args.bootstrap, "arms": []}
    for label, path, k in ARMS:
        # The report names the dump as the arms do. The shipped copy is the same
        # rollouts gzipped, so a rerun writes the same bytes from either.
        source = found[path]
        table = cluster_table(source)
        rate, spread, sizes = fit_beta_binomial(table)
        clean_rate, clean_spread, clean_sizes = fit_beta_binomial(
            cluster_table(source, arm="clean"))
        entry = {"arm": label, "dump": path, "k": k,
                 "observed_clusters": int(len(table)),
                 "observed_rate": rate, "beta_concentration": spread,
                 "clean_rate": clean_rate, "cells": []}
        print(f"\n{label}: {len(table)} clusters, rate {100*rate:.1f}%, "
              f"beta concentration {spread:.1f}")
        entry["effective_clusters_file"] = effective_clusters(sizes)
        entry["effective_clusters_clean"] = effective_clusters(clean_sizes)
        print(f"   effective (Kish) clusters: {entry['effective_clusters_file']:.1f} "
              f"released, {entry['effective_clusters_clean']:.1f} clean")
        for n_clusters in (None,) + COUNTS:
            rng = np.random.default_rng(args.seed + (n_clusters or 0))
            below, below_diff, wide, wide_diff = [], [], 0.0, 0.0
            for _ in range(args.reps):
                hits, trials = simulate(rate, spread, sizes, n_clusters, rng)
                draws = bootstrap_draws(hits, trials, args.bootstrap, rng)
                below.append(float(np.mean(draws < rate)))
                lo, hi = np.percentile(draws, [2.5, 97.5])
                wide += float(hi - lo)
                # the paired-difference interval, exactly as the arms form it
                c_hits, c_trials = simulate(clean_rate, clean_spread, clean_sizes,
                                            n_clusters, rng)
                c_draws = bootstrap_draws(c_hits, c_trials, args.bootstrap, rng)
                below_diff.append(float(np.mean(draws - c_draws < rate - clean_rate)))
                d = np.percentile(draws - c_draws, [2.5, 97.5])
                wide_diff += float(d[1] - d[0])
            curve, curve_diff = coverage_curve(below), coverage_curve(below_diff)
            at95 = int(np.argmin(np.abs(LEVELS - 0.95)))
            cell = {"n_clusters": n_clusters or entry["observed_clusters"],
                    "as_observed": n_clusters is None,
                    "coverage_margin": float(curve[at95]),
                    "coverage_difference": float(curve_diff[at95]),
                    "level_for_95_margin": level_for(curve),
                    "level_for_95_difference": level_for(curve_diff),
                    "mean_width_margin": 100.0 * wide / args.reps,
                    "mean_width_difference": 100.0 * wide_diff / args.reps}
            entry["cells"].append(cell)
            tag = "as observed" if n_clusters is None else f"{n_clusters:4d} clusters"
            print(f"   {tag:>14s}   margin {cell['coverage_margin']:.3f} "
                  f"(95% needs {cell['level_for_95_margin']:.3f})   "
                  f"difference {cell['coverage_difference']:.3f} "
                  f"(needs {cell['level_for_95_difference']:.3f})")
        report["arms"].append(entry)

    RESULTS.mkdir(exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
