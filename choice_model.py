#!/usr/bin/env python3
"""Split a no-data multiple-choice score into knowledge and option geometry.

A solver given a question but not the data it needs can still beat chance two
ways. It may know the answer, from the literature or from pretraining exposure.
Or it may have a preference over *where a value sits among the options* -- a
liking for middling numbers, say -- which pays off whenever the benchmark puts
its key there more often than chance. The field reads an above-chance no-data
score as the first explanation and does not measure the second.

Both are identifiable from the published predictions, with no new inference
runs, because the release records which options were shown, which was keyed,
and which was chosen. Write ``k`` for the number of options, ``j`` for the
sorted rank of the keyed value among them, and ``r`` for the rank of the value
the solver picked. Model the solver as knowing the answer with probability
``lambda`` and otherwise choosing by a rank preference ``b``:

    P(r | j) = lambda * 1[r = j] + (1 - lambda) * b_r

The off-diagonal cells of the k x k table of (j, r) pin down ``b``; the excess
on the diagonal pins down ``lambda``. That is ``k`` free parameters against
``k(k-1)`` free cells -- the row totals are fixed by the benchmark, not by the
solver -- so the fit is over-identified and can be tested rather than assumed.

The decomposition of the margin over chance is then exact. With
``G = sum_j p_j b_j`` the score a purely rank-driven solver would get,

    accuracy - 1/k  =  lambda * (1 - G)   +   (G - 1/k)
                       \\-- knowledge --/      \\- geometry -/

Nothing in the account is specific to four options: ``k`` is read off the data,
which is what lets the same model be fitted to a ten-option benchmark.

This says what a no-data margin is evidence of. It is fitted here to the
benchmark authors' own published runs, and separately to open-weight models for
which the rank preference can also be measured directly -- with the question
withheld -- so the estimator can be checked against a quantity it did not see.
"""
import argparse
import ast
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from option_artifacts import parse_number

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

N_OPTIONS = 4               # the option count of the BixBench releases
CHANCE = 1.0 / N_OPTIONS    # ... and its chance rate; both are defaults only
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABEL_RE = re.compile(r"^\s*\(?([A-Z])\)?[\.\:\)]?\s*(.*)$", re.DOTALL)
SEED = 20260920


# --------------------------------------------------------------------------
def strip_label(text):
    """Turn ``"(B) 0.0031"`` into ``"0.0031"``; leave unlabelled text alone."""
    match = LABEL_RE.match(str(text))
    return match.group(2).strip() if match else str(text).strip()


def ranks_of(values):
    """Sorted-rank of each position, or None unless the values are distinct numbers."""
    if not values or any(v is None for v in values):
        return None
    if len(set(values)) < len(values):
        return None
    order = sorted(range(len(values)), key=lambda i: values[i])
    position = {src: rank for rank, src in enumerate(order)}
    return [position[i] for i in range(len(values))]


def option_count(pairs):
    """The option count implied by the ranks present, for callers that lack it.

    Ranks are 0-based positions among the options, so the count is one more than
    the largest rank seen. A rank that never occurs in a small sample would
    under-count, which is why every caller that knows ``k`` passes it instead.
    """
    return 1 + max(max(item["key_rank"], item["chosen_rank"]) for item in pairs)


def pairs_from_v10(path, n_options=N_OPTIONS):
    """(key rank, chosen rank) for every numeric item in a published v1.0 run."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out, skipped = [], Counter()
    for row in rows:
        try:
            choices = ast.literal_eval(row["choices"])
        except (ValueError, SyntaxError):
            skipped["unparsable choices"] += 1
            continue
        if len(choices) != n_options:
            skipped[f"not {n_options} options"] += 1
            continue
        values = [parse_number(strip_label(c)) for c in choices]
        ranks = ranks_of(values)
        if ranks is None:
            skipped["not distinct numbers"] += 1
            continue
        letters = ALPHABET[:n_options]
        target, predicted = row.get("target", "").strip(), row.get("predicted", "").strip()
        if target not in letters or predicted not in letters:
            skipped["unparsable letter"] += 1
            continue
        out.append({"cluster": row.get("uuid", ""),
                    "key_rank": ranks[letters.index(target)],
                    "chosen_rank": ranks[letters.index(predicted)],
                    "correct": int(target == predicted)})
    return out, dict(skipped)


def pairs_from_probe(path, arm="original"):
    from no_data_probe import open_outcomes
    with open_outcomes(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    model = rows[0]["model"]
    # The option count is recorded per row from the k-generalised probe; older
    # files carry one probability per answer letter, which says the same thing.
    k = rows[0].get("n_options") or len(rows[0]["letter_probs"])
    out = [{"cluster": r["cluster"], "key_rank": r["key_rank"],
            "chosen_rank": r["chosen_rank"], "correct": r["correct"], "draw": r["draw"]}
           for r in rows
           if r["arm"] == arm and r["key_rank"] is not None and r["chosen_rank"] is not None]
    return model, out, k


# --------------------------------------------------------------------------
def table_of(pairs, k=None):
    """Collapse items to the k x k count table of (key rank j, chosen rank r).

    The likelihood depends on the items only through these ``k^2`` counts, so
    every fit and every bootstrap replicate runs on the table. This is an exact
    reformulation, not an approximation, and it is what makes a cluster
    bootstrap that refits the model on every replicate affordable.
    """
    k = k or option_count(pairs)
    counts = [[0] * k for _ in range(k)]
    for item in pairs:
        counts[item["key_rank"]][item["chosen_rank"]] += 1
    return counts


def log_likelihood_table(counts, lam, b):
    total = 0.0
    for j, row in enumerate(counts):
        for r, n in enumerate(row):
            if n:
                q = (lam if r == j else 0.0) + (1 - lam) * b[r]
                total += n * math.log(max(q, 1e-300))
    return total


def fit_table(counts, iterations=5000, tolerance=1e-13, starts=(0.05, 0.3, 0.6)):
    """Maximum-likelihood ``lambda`` and rank preference ``b`` by EM on the table.

    Several starts are used because the likelihood flattens when the key's rank
    distribution is close to uniform: with nothing to align to, knowledge and
    preference stop being separately identified, and the fit should then be
    reported as uninformative rather than as a point.
    """
    k = len(counts)
    n = sum(sum(row) for row in counts)
    if n == 0:
        return None
    best = None
    for lam0 in starts:
        lam, b, previous = lam0, [1 / k] * k, -math.inf
        for _ in range(iterations):
            # E step: the "knew it" component can only explain diagonal cells.
            responsibility, known = [0.0] * k, 0.0
            for j in range(k):
                denom = lam + (1 - lam) * b[j]
                responsibility[j] = 0.0 if denom <= 0 else lam / denom
                known += counts[j][j] * responsibility[j]
            lam = min(max(known / n, 0.0), 1.0)
            guess = [0.0] * k
            for j in range(k):
                for r in range(k):
                    guess[r] += counts[j][r] * ((1 - responsibility[j]) if r == j else 1.0)
            total = sum(guess)
            b = [g / total for g in guess] if total > 0 else [1 / k] * k
            loglik = log_likelihood_table(counts, lam, b)
            if loglik - previous < tolerance:
                break
            previous = loglik
        loglik = log_likelihood_table(counts, lam, b)
        if best is None or loglik > best["loglik"]:
            best = {"lam": lam, "b": b, "loglik": loglik}
    return best


def fit(pairs, k=None, **kwargs):
    """Fit from a list of items, for callers that hold items rather than counts."""
    for dropped in ("starts", "seed"):
        kwargs.pop(dropped, None)
    return fit_table(table_of(pairs, k), **kwargs) if pairs else None


def log_likelihood(pairs, lam, b, k=None):
    return log_likelihood_table(table_of(pairs, k), lam, b)


def key_distribution(pairs, k=None):
    k = k or option_count(pairs)
    counts = Counter(item["key_rank"] for item in pairs)
    return [counts.get(j, 0) / len(pairs) for j in range(k)]


def decompose(pairs, lam, b, k=None):
    """Exact split of the margin over chance into knowledge and geometry."""
    return _decompose(key_distribution(pairs, k), lam, b,
                      statistics.fmean([item["correct"] for item in pairs]))


def decompose_table(counts, lam, b, accuracy):
    n = sum(sum(row) for row in counts)
    return _decompose([sum(row) / n for row in counts], lam, b, accuracy)


def _decompose(p, lam, b, accuracy):
    chance = 1.0 / len(p)
    geometry_score = sum(pj * bj for pj, bj in zip(p, b))
    return {
        "accuracy": accuracy, "n_options": len(p),
        "chance": chance, "margin": accuracy - chance,
        "lambda": lam, "rank_preference_b": {str(r): b[r] for r in range(len(b))},
        "key_by_rank_p": {str(j): p[j] for j in range(len(p))},
        "geometry_only_score_G": geometry_score,
        "knowledge_component": lam * (1 - geometry_score),
        "geometry_component": geometry_score - chance,
        "fitted_accuracy": lam + (1 - lam) * geometry_score,
    }


def geometry_bounds(p):
    """What option geometry can hand a solver, over every possible rank preference.

    A no-knowledge solver that picks sorted rank r with probability b_r scores
    ``G = <p, b>``. Since ``b`` is a distribution, ``G`` is a convex combination
    of the entries of ``p``, so

        min_j p_j  <=  G  <=  max_j p_j,

    both attained by a point mass. Three things follow, and they are the reason
    the repair takes the form it does.

    First, a uniform key-rank distribution gives ``G = 1/k`` for *every* ``b``:
    geometry is then worth exactly nothing to any solver, present or future, and
    no assumption about solver behaviour is needed. Uniformity is not merely one
    fix among several; it is the condition that removes the channel.

    Second, any departure from uniformity is exploitable by construction, and
    the worst case ``max_j p_j`` is reached by a solver as simple as "always take
    the second-smallest value".

    Third, shuffling the options changes which *letter* carries the key, not its
    rank among the values, so ``p`` -- and hence this whole range -- is invariant
    to presentation shuffling. A shuffle neutralises letter preference, which is
    a real bias and worth neutralising, but it cannot touch this one.
    """
    worst, best = max(p), min(p)
    chance = 1.0 / len(p)
    return {"key_by_rank_p": {str(j): p[j] for j in range(len(p))},
            "n_options": len(p), "chance": chance,
            "max_geometry_credit": worst - chance,
            "min_geometry_credit": best - chance,
            "worst_case_score": worst, "best_case_score": best,
            "uniform_key_ranks_give_chance_for_every_solver": True,
            "exploitable": worst > chance + 1e-12}


def realised_geometry(p, b):
    """The geometry credit a specific solver preference actually collects."""
    G = sum(pj * bj for pj, bj in zip(p, b))
    return {"G": G, "geometry_credit": G - 1.0 / len(p)}


def goodness_of_fit(pairs, lam, b, k=None):
    """Chi-square over the k x k (key rank, chosen rank) table against the fit."""
    table = table_of(pairs, k)
    k = len(table)
    chi2 = 0.0
    cells = 0
    for j in range(k):
        nj = sum(table[j])
        if nj == 0:
            continue
        for r in range(k):
            expected = nj * ((lam if r == j else 0.0) + (1 - lam) * b[r])
            if expected <= 0:
                continue
            chi2 += (table[j][r] - expected) ** 2 / expected
            cells += 1
    # cells, minus one per row total, minus the k fitted parameters
    # (lambda and the k-1 free entries of b)
    df = max(cells - k - k, 1)
    return {"chi2": chi2, "df": df, "p_value": _chi2_sf(chi2, df),
            "note": "large p means the two-source account is adequate for this table"}


def _chi2_sf(x, df):
    """Upper tail of a chi-square, by series; adequate for the df used here."""
    if x <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    if df == 2:
        return math.exp(-x / 2)
    # regularised upper incomplete gamma Q(df/2, x/2) by continued fraction
    a, z = df / 2.0, x / 2.0
    if z < a + 1:
        term, total, k = 1.0 / a, 1.0 / a, 0
        while abs(term) > 1e-16 * abs(total) and k < 10000:
            k += 1
            term *= z / (a + k)
            total += term
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    b0, c, d, h = z + 1 - a, 1e300, 1.0 / (z + 1 - a), 1.0 / (z + 1 - a)
    for i in range(1, 10000):
        an = -i * (i - a)
        b0 += 2
        d = an * d + b0
        if abs(d) < 1e-300:
            d = 1e-300
        c = b0 + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return max(0.0, min(1.0, h * math.exp(-z + a * math.log(z) - math.lgamma(a))))


def lrt_no_knowledge(pairs, fitted, k=None):
    """Likelihood-ratio test of ``lambda = 0``: is any knowledge needed at all?"""
    k = k or option_count(pairs)
    counts = Counter(item["chosen_rank"] for item in pairs)
    b0 = [counts.get(r, 0) / len(pairs) for r in range(k)]
    null = log_likelihood(pairs, 0.0, b0, k)
    statistic = max(0.0, 2 * (fitted["loglik"] - null))
    # lambda is on the boundary under the null, so the reference distribution is
    # a half-half mixture of a point mass at zero and a chi-square with 1 df.
    return {"statistic": statistic, "p_value": 0.5 * _chi2_sf(statistic, 1),
            "loglik_null": null, "loglik_fitted": fitted["loglik"],
            "note": "boundary-corrected: 0.5 * P(chi2_1 > statistic)"}


def cluster_bootstrap(pairs, reps, alpha=0.05, seed=SEED, k=None):
    """Resample whole capsules and refit, for intervals on every fitted quantity.

    Per-capsule count tables add, so a replicate is a sum of ``k^2``-cell tables
    rather than a rebuilt item list, and the model is genuinely refitted on each
    one instead of being held fixed while only the data move.
    """
    rng = random.Random(seed)
    k = k or option_count(pairs)
    grouped = defaultdict(list)
    for item in pairs:
        grouped[item["cluster"]].append(item)
    keys = list(grouped)
    tables = {c: table_of(v, k) for c, v in grouped.items()}
    hits = {c: sum(item["correct"] for item in v) for c, v in grouped.items()}
    sizes = {c: len(v) for c, v in grouped.items()}

    draws = defaultdict(list)
    for _ in range(reps):
        counts = [[0] * k for _ in range(k)]
        correct = total = 0
        for key in (rng.choice(keys) for _ in keys):
            source = tables[key]
            for j in range(k):
                row, out = source[j], counts[j]
                for r in range(k):
                    out[r] += row[r]
            correct += hits[key]
            total += sizes[key]
        fitted = fit_table(counts, starts=(0.2,))
        if not fitted or total == 0:
            continue
        rec = decompose_table(counts, fitted["lam"], fitted["b"], correct / total)
        for field in ("lambda", "knowledge_component", "geometry_component",
                      "geometry_only_score_G", "accuracy"):
            draws[field].append(rec[field])

    out = {}
    for field, values in draws.items():
        values.sort()
        lo = values[max(0, int(math.floor((alpha / 2) * len(values))))]
        hi = values[min(len(values) - 1, int(math.ceil((1 - alpha / 2) * len(values))) - 1)]
        out[field] = [lo, hi]
    return {"reps": reps, "n_clusters": len(keys), "ci95": out}


# --------------------------------------------------------------------------
def per_draw_diagnostics(pairs, k=None):
    """Goodness of fit and the knowledge test, evaluated one redraw at a time.

    The probe answers every item under many independent redraws of the
    distractor values. Pooling them gives a precise point estimate, but they are
    not twenty independent observations of the benchmark, and a chi-square or a
    likelihood-ratio test run on the pooled table would be reading a sample
    twenty times larger than the one that exists. Both tests are therefore
    computed within each redraw, where the sample is the 205 questions, and
    summarised across redraws. Point estimates still come from the pooled fit,
    for which the capsule bootstrap supplies the interval.
    """
    k = k or option_count(pairs)
    draws = defaultdict(list)
    for item in pairs:
        if "draw" in item:
            draws[item["draw"]].append(item)
    if len(draws) < 2:
        return None
    gof, lrt, lams = [], [], []
    for group in draws.values():
        # k is passed down so a single redraw's table keeps its full dimension
        # even when a rank happens not to be chosen within that redraw.
        fitted = fit(group, k)
        if not fitted:
            continue
        lams.append(fitted["lam"])
        gof.append(goodness_of_fit(group, fitted["lam"], fitted["b"], k)["p_value"])
        lrt.append(lrt_no_knowledge(group, fitted, k)["p_value"])
    if not gof:
        return None
    return {"n_draws": len(gof), "n_items_per_draw": len(next(iter(draws.values()))),
            "goodness_of_fit_p_median": statistics.median(gof),
            "goodness_of_fit_rejected_at_0.05": sum(q < 0.05 for q in gof),
            "lrt_no_knowledge_p_median": statistics.median(lrt),
            "lrt_rejected_at_0.05": sum(q < 0.05 for q in lrt),
            "lambda_median_over_draws": statistics.median(lams),
            "lambda_spread_over_draws": (max(lams) - min(lams))}


def analyse(label, pairs, reps, extra=None, k=None):
    k = k or option_count(pairs)
    fitted = fit(pairs, k)
    if fitted is None:
        return None
    report = {"run": label, "n_items": len(pairs), "n_options": k,
              "n_clusters": len({item["cluster"] for item in pairs})}
    report.update(decompose(pairs, fitted["lam"], fitted["b"], k))
    report["bootstrap"] = cluster_bootstrap(pairs, reps, k=k)
    diagnostics = per_draw_diagnostics(pairs, k)
    if diagnostics:
        # Many redraws of the same questions: test within a redraw, not across.
        report["diagnostics_per_draw"] = diagnostics
    else:
        report["goodness_of_fit"] = goodness_of_fit(pairs, fitted["lam"], fitted["b"], k)
        report["lrt_no_knowledge"] = lrt_no_knowledge(pairs, fitted, k)
    if extra:
        report.update(extra)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v10-dir", type=Path, default=Path("data/external/zero_shot_v10"))
    parser.add_argument("--probe-dir", type=Path, default=Path("results/probe_raw"))
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/choice_model.json"))
    parser.add_argument("--question-file", type=Path, default=Path("data/bixbench.jsonl"),
                        help="question file whose key-rank distribution sets the bound")
    parser.add_argument("--benchmark-label", default="BixBench v1.5")
    args = parser.parse_args()

    # The key-rank distribution of the release under study, which is what makes
    # the channel exploitable and is measurable from the question file alone.
    v15_p = None
    question_file = args.question_file
    if question_file.exists():
        from mcq_audit import build_items, read_rows
        items, k_file = build_items(read_rows(question_file), "ideal", "distractors",
                                    "capsule_uuid", "question")
        ranks = Counter(it["rank"] for it in items if it["rank"] is not None)
        total = sum(ranks.values())
        v15_p = [ranks.get(j, 0) / total for j in range(k_file)]

    published = []
    for path in sorted(args.v10_dir.glob("*mcq*.csv")):
        pairs, skipped = pairs_from_v10(path)
        if not pairs:
            continue
        record = analyse(path.stem, pairs, args.reps, {"source": "published run",
                                                       "items_skipped": skipped})
        if record and v15_p and len(v15_p) == record["n_options"]:
            b = [record["rank_preference_b"][str(r)] for r in range(record["n_options"])]
            record["transferred_to_v15_key_ranks"] = realised_geometry(v15_p, b)
        published.append(record)
        print(f"{path.stem}", flush=True)

    probe = []
    from no_data_probe import outcome_files
    for path in (outcome_files(args.probe_dir) if args.probe_dir.exists() else []):
        model, pairs, k = pairs_from_probe(path, "original")
        if not pairs:
            continue
        # The same preference, measured directly with the question withheld.
        _, stemless, _ = pairs_from_probe(path, "original_stemless")
        counts = Counter(item["chosen_rank"] for item in stemless)
        measured = ({str(r): counts.get(r, 0) / len(stemless) for r in range(k)}
                    if stemless else None)
        record = analyse(model, pairs, args.reps,
                         {"source": "open-weight probe, question shown",
                          "rank_preference_measured_with_question_withheld": measured}, k)
        if record and measured:
            fitted_b = record["rank_preference_b"]
            record["b_vs_measured_total_variation"] = 0.5 * sum(
                abs(fitted_b[str(r)] - measured[str(r)]) for r in range(k))
            # The credit this preference collects, using the preference measured
            # with the question withheld rather than the one inferred with it shown.
            record["realised_geometry_from_measured_b"] = realised_geometry(
                key_distribution(pairs, k), [measured[str(r)] for r in range(k)])
        probe.append(record)
        print(f"{model}", flush=True)

    bounds = {}
    if v15_p:
        bounds[args.benchmark_label] = geometry_bounds(v15_p)
    if published:
        first = published[0]
        bounds["BixBench v1.0"] = geometry_bounds(
            [first["key_by_rank_p"][str(j)] for j in range(first["n_options"])])

    realised = [r["geometry_component"] for r in published + probe if r]
    doc = {"model": ("P(chosen rank r | key rank j) = lambda*1[r=j] + (1-lambda)*b_r; "
                     "margin over chance = lambda*(1-G) + (G-1/k), G = sum_j p_j b_j"),
           "n_options": len(v15_p) if v15_p else None, "bootstrap_reps": args.reps,
           "geometry_bounds": bounds,
           "realised_vs_worst_case": {
               "n_solvers_measured": len(realised),
               "min_realised_geometry_credit": min(realised) if realised else None,
               "max_realised_geometry_credit": max(realised) if realised else None,
               "worst_case_available_on_benchmark": (bounds.get(args.benchmark_label) or {}
                                                     ).get("max_geometry_credit"),
               "note": ("the channel is open but no measured solver walks through it: "
                        "exploitability is not exploitation")},
           "published_runs": published, "open_weight_runs": probe}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}\n")
    for name, rec in bounds.items():
        print(f"{name}: key ranks " + " ".join(
            f"j{j}={rec['key_by_rank_p'][str(j)]:.1%}" for j in range(rec["n_options"]))
            + f"  -> geometry credit available in "
              f"[{rec['min_geometry_credit']:+.1%}, {rec['max_geometry_credit']:+.1%}]")
    print()

    header = f"{'run':52s} {'acc':>6s} {'lambda':>8s} {'know':>7s} {'geom':>7s} {'GoF p':>7s}"
    for group, rows in (("published", published), ("open-weight", probe)):
        if not rows:
            continue
        print(f"--- {group}")
        print(header)
        for r in rows:
            gof = (r["goodness_of_fit"]["p_value"] if "goodness_of_fit" in r
                   else r["diagnostics_per_draw"]["goodness_of_fit_p_median"])
            print(f"{r['run'][:52]:52s} {r['accuracy']:6.1%} {r['lambda']:8.3f} "
                  f"{r['knowledge_component']:+7.1%} {r['geometry_component']:+7.1%} "
                  f"{gof:7.3f}")
        print()


if __name__ == "__main__":
    main()
