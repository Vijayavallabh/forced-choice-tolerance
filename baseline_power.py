#!/usr/bin/env python3
r"""How many items a no-data baseline needs before it can detect anything.

A benchmark publishes a no-data baseline so that a reader can conclude
something from it. Of BixBench's two published zero-shot multiple-choice runs,
one scores $27.7\%$ and one $30.2\%$ against $25\%$ chance, and neither's
recall term is distinguishable from zero once \eqref{eq:split} is fitted
($\widehat\lambda = 0.028$ and $0.069$, boundary-corrected $p = 0.27$ and
$0.07$ at $n = 159$). That is not a null result about contamination; it is a
statement about the instrument, and the instrument's resolution is a function
of how many items carry a rank.

This measures that resolution: the power of the boundary-corrected test of
$\lambda = 0$ over item counts and true $\lambda$, at both option counts, and
the smallest item count reaching $80\%$ power.

    python3 baseline_power.py --reps 600
"""
import argparse
import json
import random
import statistics
from bisect import bisect_left

from choice_model import fit_table, log_likelihood, lrt_no_knowledge
from estimator_simulation import draw, table_from

SHAPES = [
    # name, k, key-rank distribution (None is uniform)
    ("BixBench v1.5 key ranks", 4, [0.124, 0.514, 0.295, 0.067]),
    ("uniform key ranks", 4, None),
    ("uniform key ranks", 10, None),
]
COUNTS = (105, 159, 250, 500, 1000, 2000, 4000)
LAMBDAS = (0.05, 0.10, 0.20)


def power_at(n, k, p, b, lam, reps, alpha, rng):
    rejects, fits = 0, 0
    for _ in range(reps):
        pairs = draw(n, k, p, b, lam, rng)
        fitted = fit_table(table_from(pairs, k))
        if fitted is None:
            continue
        fits += 1
        statistic = {"loglik": log_likelihood(pairs, fitted["lam"], fitted["b"], k)}
        rejects += lrt_no_knowledge(pairs, statistic, k)["p_value"] < alpha
    return rejects / max(1, fits)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=600)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--output", default="results/baseline_power.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    report = {"reps": args.reps, "alpha": args.alpha, "target_power": args.target,
              "counts": list(COUNTS), "lambdas": list(LAMBDAS), "curves": []}

    for name, k, p in SHAPES:
        p = p or [1 / k] * k
        b = [(1 + 0.4 * ((i % 3) - 1)) / k for i in range(k)]
        b = [value / sum(b) for value in b]
        for lam in LAMBDAS:
            powers = []
            for n in COUNTS:
                powers.append(power_at(n, k, p, b, lam, args.reps, args.alpha, rng))
                print(f"{name:<24} k={k:<3} lambda={lam:.2f}  n={n:<5} "
                      f"power {powers[-1]:.3f}", flush=True)
            # The smallest listed count that reaches the target, by the
            # monotone envelope so a noisy cell cannot move the answer down.
            envelope, running = [], 0.0
            for value in powers:
                running = max(running, value)
                envelope.append(running)
            index = bisect_left(envelope, args.target)
            report["curves"].append({
                "shape": name, "k": k, "key_ranks": p, "lambda": lam,
                "power": powers,
                "n_for_target": COUNTS[index] if index < len(COUNTS) else None,
            })

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
