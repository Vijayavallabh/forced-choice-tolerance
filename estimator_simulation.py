#!/usr/bin/env python3
r"""The two estimator claims in \S2, measured instead of asserted.

Section 2 says that at the published sample sizes the EM fit of $\lambda$ is
unbiased and that the boundary-corrected test of $\lambda = 0$ is correctly
sized. Both are checkable by simulation from the model itself, which is what
this does: draw key ranks from a chosen $p$, draw the solver's pick from
$\lambda\mathbf{1}[r=j] + (1-\lambda)b_r$, fit, and compare.

Sizes are the ones the paper reports on: BixBench's 105 four-option items,
MMLU's 661, MMLU-Pro's 1,263 at ten options. Rank distributions are the
benchmarks' own where they are known, and uniform as the null case.

    python3 estimator_simulation.py --reps 2000
"""
import argparse
import json
import random
import statistics
from collections import Counter

from choice_model import fit_table, lrt_no_knowledge, log_likelihood

# Key-rank distributions actually observed, so the simulation is run where the
# paper's fits live rather than at a convenient point.
SETTINGS = [
    ("BixBench v1.5", 105, [0.124, 0.514, 0.295, 0.067]),
    ("MMLU", 661, None),
    ("MMLU-Pro", 1263, None),
]
LAMBDAS = (0.0, 0.05, 0.15, 0.40)


def draw(n, k, p, b, lam, rng):
    """One simulated (key rank, chosen rank) sample of size n."""
    pairs = []
    for _ in range(n):
        j = rng.choices(range(k), weights=p)[0]
        if rng.random() < lam:
            r = j
        else:
            r = rng.choices(range(k), weights=b)[0]
        pairs.append({"key_rank": j, "chosen_rank": r})
    return pairs


def table_from(pairs, k):
    counts = [[0] * k for _ in range(k)]
    for item in pairs:
        counts[item["key_rank"]][item["chosen_rank"]] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output", default="results/estimator_simulation.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    report = {"reps": args.reps, "alpha": args.alpha, "settings": []}

    for name, n, p in SETTINGS:
        k = len(p) if p else (10 if "Pro" in name else 4)
        p = p or [1 / k] * k
        # A mildly non-uniform preference, so the fit has something to separate.
        b = [(1 + 0.4 * ((i % 3) - 1)) / k for i in range(k)]
        b = [value / sum(b) for value in b]
        for lam in LAMBDAS:
            estimates, rejects, failures = [], 0, 0
            for _ in range(args.reps):
                pairs = draw(n, k, p, b, lam, rng)
                counts = table_from(pairs, k)
                fitted = fit_table(counts)
                if fitted is None:
                    failures += 1
                    continue
                estimates.append(fitted["lam"])
                fitted_for_lrt = {"loglik": log_likelihood(pairs, fitted["lam"],
                                                           fitted["b"], k)}
                rejects += lrt_no_knowledge(pairs, fitted_for_lrt,
                                            k)["p_value"] < args.alpha
            mean = statistics.fmean(estimates)
            entry = {"benchmark": name, "n": n, "k": k, "lambda": lam,
                     "mean_lambda_hat": mean, "bias": mean - lam,
                     "sd_lambda_hat": statistics.pstdev(estimates),
                     "reject_rate": rejects / max(1, len(estimates)),
                     "n_failed_fits": failures}
            report["settings"].append(entry)
            tag = "  <- size of the test" if lam == 0 else ""
            print(f"{name:<15} n={n:<5} k={k:<3} lambda={lam:.2f}  "
                  f"mean lambda-hat {mean:.4f}  bias {mean - lam:+.4f}  "
                  f"sd {entry['sd_lambda_hat']:.4f}  "
                  f"reject {entry['reject_rate']:.3f}{tag}")

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
