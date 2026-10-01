#!/usr/bin/env python3
"""Sample-size planning for a matched temporal contrast.

Two things are separated here, because conflating them is how underpowered
benchmark comparisons get published:

  * the marginal spread of per-study scores, sigma, which we now estimate from
    published per-question runs rather than assume; and
  * the within-pair correlation rho, which no released artifact lets us
    estimate, and which therefore stays an explicit sensitivity axis.

The estimated sigma comes from the BixBench authors' published zero-shot
baselines. Those are no-data runs, not agent trajectories, so the value anchors
the calculation without being a substitute for a pilot on the real design.
"""
import json
import math
import statistics
from pathlib import Path

from scipy.optimize import brentq
from scipy.stats import nct, t


def power(n, sd, gap=0.10, alpha=0.05):
    """Exact noncentral-t power for a two-sided paired test."""
    critical = t.ppf(1 - alpha / 2, n - 1)
    noncentrality = gap * math.sqrt(n) / sd
    return float(nct.sf(critical, n - 1, noncentrality) + nct.cdf(-critical, n - 1, noncentrality))


def required_pairs(sd, gap=0.10, target=0.80):
    return next(n for n in range(2, 10001) if power(n, sd, gap) >= target)


def minimum_detectable(n, sd, target=0.80):
    """The gap at which the test reaches ``target`` power. The bracket grows from a small gap only until it
    holds the root, so the noncentral t is never evaluated far past it, where SciPy 1.16's evaluates to NaN."""
    hi = 1e-3
    while power(n, sd, hi) < target:
        hi *= 2
    return brentq(lambda gap: power(n, sd, gap) - target, 1e-5, hi)


def sd_of_difference(sigma, rho):
    """SD of a paired difference given equal marginal SDs and correlation rho."""
    return math.sqrt(2 * sigma ** 2 * (1 - rho))


def measured_sigma(path=Path("results/no_data_baseline.json")):
    """Spread of per-study mean scores, measured from published per-question runs.

    A source study, not a question, is the unit a temporal contrast compares,
    so the relevant sigma is the SD across study-level mean scores.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for run in data["runs"]:
        clustering = run.get("clustering") or {}
        if not clustering:
            continue
        out[run["run"]] = {
            "accuracy": run["accuracy"],
            "icc_questions_within_study": clustering["icc"],
            "design_effect": clustering["design_effect"],
            "effective_n_questions": clustering["effective_n"],
            "nominal_n_questions": clustering["n_questions"],
        }
    return out


def study_level_sigma(baseline=Path("results/no_data_baseline.json"),
                      jsonl=Path("data/bixbench.jsonl"),
                      csv_path=Path("data/external/zero_shot_v15/gpt-4o-grader-mcq-refusal-False.csv")):
    """SD across capsules of the per-capsule mean score, for each published run."""
    import csv as _csv
    if not jsonl.exists() or not csv_path.parent.exists():
        return None
    capsule_of = {}
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                capsule_of[row["question_id"]] = row["capsule_uuid"]
    sigmas = {}
    for path in sorted(csv_path.parent.glob("*.csv")):
        groups = {}
        with path.open(encoding="utf-8", newline="") as handle:
            for row in _csv.DictReader(handle):
                cap = capsule_of.get((row.get("uuid") or "").strip())
                if cap:
                    groups.setdefault(cap, []).append(
                        1.0 if str(row.get("correct", "")).strip().lower() == "true" else 0.0)
        means = [statistics.fmean(v) for v in groups.values() if v]
        if len(means) > 2:
            sigmas[path.stem] = {"n_studies": len(means), "mean": statistics.fmean(means),
                                 "sd_of_study_means": statistics.stdev(means)}
    return sigmas


def main():
    Path("results").mkdir(exist_ok=True)

    # ---- sensitivity to the assumed SD of the paired difference
    assumed = []
    for sd in (0.20, 0.32, 0.53):
        assumed.append({
            "sd_difference": sd,
            "implied_rho_at_sigma_0_45": 1 - sd ** 2 / (2 * 0.45 ** 2),
            "pairs_for_10pp": required_pairs(sd),
            "power_n3": power(3, sd), "power_n8": power(8, sd),
            "mde": {str(n): minimum_detectable(n, sd)
                    for n in (8, 10, 30, 75)},
        })

    # ---- the same calculation anchored on a measured marginal SD
    sigmas = study_level_sigma() or {}
    grounded = []
    if sigmas:
        values = [v["sd_of_study_means"] for v in sigmas.values()]
        sigma_hat = statistics.fmean(values)
        for rho in (0.0, 0.3, 0.5, 0.7, 0.9):
            sd_d = sd_of_difference(sigma_hat, rho)
            grounded.append({
                "rho": rho, "sigma_hat": sigma_hat, "sd_difference": sd_d,
                "pairs_for_10pp": required_pairs(sd_d),
                "mde": {str(n): minimum_detectable(n, sd_d)
                        for n in (10, 30, 75)},
            })

    result = {
        "alpha": 0.05, "target_power": 0.80,
        "test": "two-sided paired t; independent normally distributed differences",
        "assumed_sd_rows": assumed,
        "measured_inputs": {
            "per_run_study_level_sd": sigmas,
            "sigma_hat_note": ("SD across capsule-level mean scores, averaged over the six published "
                               "zero-shot baseline runs. These are no-data runs, not agent "
                               "trajectories; the value anchors planning and does not replace a pilot."),
            "clustering": measured_sigma() or {},
        },
        "grounded_rows": grounded,
    }
    Path("results/power.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("Assumed-SD sensitivity:")
    for row in assumed:
        print(f"  SD(d)={row['sd_difference']:.2f}  n for 10 pts = {row['pairs_for_10pp']:3d}"
              f"  MDE at n=30: {row['mde']['30'] * 100:.1f} pts")
    if grounded:
        print(f"\nMeasured marginal SD across studies: sigma = {grounded[0]['sigma_hat']:.3f}")
        print("Required independent source-study pairs for a 10-point gap at 80% power:")
        for row in grounded:
            print(f"  rho={row['rho']:.1f}  SD(d)={row['sd_difference']:.3f}"
                  f"  n = {row['pairs_for_10pp']:3d}"
                  f"  MDE at n=30: {row['mde']['30'] * 100:.1f} pts")
    print("\nwrote results/power.json")


if __name__ == "__main__":
    main()
