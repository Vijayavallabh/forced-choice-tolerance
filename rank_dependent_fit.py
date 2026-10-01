"""Does the goodness-of-fit failure move the geometry term?

\\eqref{eq:split}'s model gives the solver one recall probability and one rank
preference.  A per-draw test rejects it on 12 of 40 MMLU draws, concentrated on
the two largest lambda-hat, which is what a solver that mostly knows the answer
should look like: recall is not the same at every key rank.  So refit with
lambda free to depend on the key's rank and see what happens to the other term.

  model A   Pr(r|j) = lambda*1[r=j] + (1-lambda)*b_r          (k parameters)
  model B   Pr(r|j) = lambda_j*1[r=j] + (1-lambda_j)*b_r      (2k-1 parameters)

Under B the accuracy still splits as sum_j p_j lambda_j (1-b_j) + (G - 1/k) with
the same G = <p,b>, so the two fits are directly comparable on the quantity this
paper reads.  Both are fitted by expectation-maximisation on the k x k table and
both are tested against the saturated multinomial.

    /opt/conda/bin/python rank_dependent_fit.py
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

from scipy.stats import chi2

from choice_model import pairs_from_probe, table_of
from no_data_probe import outcome_files

RESULTS = pathlib.Path("results")


def fit_shared(counts, iterations=4000, tol=1e-13):
    """Model A: one lambda, one preference b."""
    k = len(counts)
    n = sum(map(sum, counts))
    lam, b = 0.3, [1.0 / k] * k
    previous = -math.inf
    for _ in range(iterations):
        known = 0.0
        soft = [0.0] * k
        for j in range(k):
            share = lam / (lam + (1 - lam) * b[j]) if lam + (1 - lam) * b[j] > 0 else 0.0
            known += counts[j][j] * share
            for r in range(k):
                soft[r] += counts[j][r] - (counts[j][j] * share if r == j else 0.0)
        lam = known / n
        total = sum(soft)
        b = [x / total for x in soft] if total else b
        value = loglik(counts, [lam] * k, b)
        if abs(value - previous) < tol:
            break
        previous = value
    return [lam] * k, b


def fit_per_rank(counts, iterations=4000, tol=1e-13):
    """Model B: one lambda per key rank, one shared preference b."""
    k = len(counts)
    rows = [sum(row) for row in counts]
    lam = [0.3] * k
    b = [1.0 / k] * k
    previous = -math.inf
    for _ in range(iterations):
        soft = [0.0] * k
        new_lam = list(lam)
        for j in range(k):
            denominator = lam[j] + (1 - lam[j]) * b[j]
            share = lam[j] / denominator if denominator > 0 else 0.0
            known = counts[j][j] * share
            new_lam[j] = known / rows[j] if rows[j] else 0.0
            for r in range(k):
                soft[r] += counts[j][r] - (known if r == j else 0.0)
        lam = new_lam
        total = sum(soft)
        b = [x / total for x in soft] if total else b
        value = loglik(counts, lam, b)
        if abs(value - previous) < tol:
            break
        previous = value
    return lam, b


def loglik(counts, lam, b):
    total = 0.0
    for j, row in enumerate(counts):
        for r, c in enumerate(row):
            if not c:
                continue
            p = (lam[j] if r == j else 0.0) + (1 - lam[j]) * b[r]
            total += c * math.log(p) if p > 0 else -math.inf
    return total


def saturated(counts):
    total = 0.0
    for row in counts:
        n = sum(row)
        for c in row:
            if c:
                total += c * math.log(c / n)
    return total


def pearson(counts, lam, b, free):
    """The same chi-square ``choice_model.goodness_of_fit`` uses, with the
    parameter count passed in so both models are tested the same way."""
    k = len(counts)
    statistic, cells = 0.0, 0
    for j in range(k):
        nj = sum(counts[j])
        if not nj:
            continue
        for r in range(k):
            expected = nj * ((lam[j] if r == j else 0.0) + (1 - lam[j]) * b[r])
            if expected <= 0:
                continue
            statistic += (counts[j][r] - expected) ** 2 / expected
            cells += 1
    df = max(cells - k - free, 1)
    return statistic, df


def report(counts, lam, b, free):
    k = len(counts)
    n = sum(map(sum, counts))
    p = [sum(row) / n for row in counts]
    geometry = sum(p[j] * b[j] for j in range(k))
    recall = sum(p[j] * lam[j] * (1 - b[j]) for j in range(k))
    statistic, df = pearson(counts, lam, b, free)
    return {
        "lambda_mean": sum(p[j] * lam[j] for j in range(k)),
        "lambda_by_rank": lam,
        "geometry_G": geometry,
        "geometry_component": geometry - 1 / k,
        "recall_component": recall,
        "chi2": statistic, "df": df,
        "gof_p": float(chi2.sf(statistic, df)),
        "deviance_vs_saturated": 2 * (saturated(counts) - loglik(counts, lam, b))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-dir", default="results/probe_mmlu")
    ap.add_argument("--arm", default="original")
    ap.add_argument("--out", default="rank_dependent_fit.json")
    args = ap.parse_args()

    rows = []
    for path in outcome_files(pathlib.Path(args.probe_dir)):
        model, pairs, k = pairs_from_probe(path, args.arm)
        if not pairs:
            continue
        for draw in sorted({item["draw"] for item in pairs}):
            counts = table_of([i for i in pairs if i["draw"] == draw], k)
            a = report(counts, *fit_shared(counts), free=k)
            bfit = report(counts, *fit_per_rank(counts), free=2 * k - 1)
            rows.append({"model": model, "draw": draw, "k": k,
                         "n": sum(map(sum, counts)), "shared": a, "per_rank": bfit})

    for r in rows:
        k = r["k"]
        lrt = r["shared"]["deviance_vs_saturated"] - r["per_rank"]["deviance_vs_saturated"]
        r["lrt_shared_vs_per_rank"] = {
            "statistic": lrt, "df": k - 1,
            "p_value": float(chi2.sf(max(lrt, 0.0), k - 1))}
    rejecting = [r for r in rows if r["shared"]["gof_p"] < 0.05]
    moved = [abs(r["per_rank"]["geometry_component"] - r["shared"]["geometry_component"])
             for r in rejecting]
    payload = {
        "probe_dir": args.probe_dir, "arm": args.arm,
        "n_draws": len(rows),
        "n_rejecting_shared": len(rejecting),
        "n_rejecting_per_rank": sum(1 for r in rows if r["per_rank"]["gof_p"] < 0.05),
        "max_geometry_shift_points": 100 * max(moved) if moved else 0.0,
        "mean_geometry_shift_points": 100 * (sum(moved) / len(moved)) if moved else 0.0,
        "sign_changes": sum(
            1 for r in rejecting
            if (r["shared"]["geometry_component"] > 0) != (r["per_rank"]["geometry_component"] > 0)),
        "geometry_range_shared_points": [
            100 * min(r["shared"]["geometry_component"] for r in rows),
            100 * max(r["shared"]["geometry_component"] for r in rows)],
        "n_lrt_favouring_per_rank": sum(
            1 for r in rows if r["lrt_shared_vs_per_rank"]["p_value"] < 0.05),
        "geometry_range_per_rank_points": [
            100 * min(r["per_rank"]["geometry_component"] for r in rows),
            100 * max(r["per_rank"]["geometry_component"] for r in rows)],
        "draws": rows}
    target = RESULTS / args.out
    target.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "draws"}, indent=1))
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
