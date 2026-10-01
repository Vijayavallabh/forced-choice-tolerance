#!/usr/bin/env python3
"""How open is the option-geometry channel across released MCQ benchmarks?

A benchmark whose keyed answer sits at a predictable position among the sorted
numeric options hands free credit to any solver that happens to prefer that
position. The size of the gift is ``max_k p_k - 1/k`` where ``p`` is the key's
rank distribution over ``k`` options: the score of the best rank-only solver,
which needs no assumption about who is solving. This measures that quantity in
every released multiple-choice benchmark we could obtain, so the defect can be
reported as a measured prevalence rather than one benchmark's mistake.

Two estimates are given for each file, because the first is optimistic.

  *plug-in*  ``max_k p_k - 1/k`` with the maximum taken over the estimated
             distribution. Taking a maximum over noisy estimates is biased
             upward, badly so when there are few items.
  *bound*    a simultaneous one-sided lower bound on ``max_k p_k``, from a
             cluster bootstrap of each share at level ``alpha/k``. Conservative
             by construction, which is the right direction for a claim that a
             benchmark hands out at least this much.

The bound decides whether a channel is called open. A held-out selection
estimate is also reported, but it is biased *downward* here and is not used for
that call; see ``credit_lower_bound`` for why.

Reads the vendored option sets under ``data/external/survey/`` and the two
BixBench releases; standard library only.
"""
import argparse
import csv
import gzip
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from option_artifacts import parse_number

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
SURVEY = Path("data/external/survey")
SEED = 20260920


# --------------------------------------------------------------------------
def numeric_ranks(options):
    """Rank of each option among the sorted values, or None if not all numeric."""
    values = [parse_number(o) for o in options]
    if any(v is None for v in values) or len(set(values)) < len(values):
        return None
    order = sorted(range(len(values)), key=lambda i: values[i])
    return [order.index(i) for i in range(len(values))]


def read_records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def bixbench_v15(path=Path("data/bixbench.jsonl")):
    for row in read_records(path):
        options = [row["ideal"]] + list(row["distractors"])
        yield {"options": [str(o) for o in options], "key": 0,
               "n_options": len(options), "cluster": row.get("capsule_uuid")}


def bixbench_v10(path=Path("data/external/zero_shot_v10/"
                           "bixbench_llm_baseline_refusal_False_mcq_gpt-4o_1.0.csv")):
    """v1.0 publishes the options as presented, so the key is found by letter."""
    import ast
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                choices = ast.literal_eval(row["choices"])
            except (ValueError, SyntaxError):
                continue
            labels, values = [], []
            for choice in choices:
                text = str(choice).strip()
                if text.startswith("(") and ")" in text:
                    labels.append(text[1:text.index(")")].strip().upper())
                    values.append(text[text.index(")") + 1:].strip())
                else:
                    labels.append(None)
                    values.append(text)
            target = (row.get("target") or "").strip().upper()
            if target not in labels:
                continue
            index = labels.index(target)
            yield {"options": [values[index]] + values[:index] + values[index + 1:],
                   "key": 0, "n_options": len(values), "cluster": row.get("uuid")}


def labbench(path):
    for row in read_records(path):
        key, distractors = row.get("ideal"), row.get("distractors")
        if key is None or not distractors:
            continue
        options = [str(key)] + [str(d) for d in distractors]
        if len(set(options)) != len(options):
            continue
        # Same cluster fields as option_artifacts.py, so the two analyses put
        # SeqQA's items into the same four groups rather than disagreeing.
        source = None
        for field in ("source", "sources", "paper-title", "subtask"):
            value = row.get(field)
            if isinstance(value, list):
                value = value[0] if value else None
            if value not in (None, ""):
                source = value
                break
        yield {"options": options, "key": 0, "n_options": len(options),
               "cluster": None if source in (None, "") else str(source)}


# --------------------------------------------------------------------------
def credit_lower_bound(items, k, reps, alpha=0.05, seed=SEED):
    """A valid lower bound on ``max_k p_k - 1/k``: what some solver can collect.

    The quantity of interest is the best rank-only solver's score, ``max_k p_k``.
    The plug-in maximum over estimated shares is biased upward, and selecting a
    rank on training folds and scoring it on held-out ones -- the obvious
    remedy, and the one this repository uses elsewhere -- is biased *downward*
    here, because a rank over-represented in the training portion is by
    construction under-represented in the held-out portion. Under exact
    uniformity that estimator returns about 17% where chance is 25%.

    So neither is used for the headline. Instead each share gets a one-sided
    cluster-bootstrap lower bound at level ``alpha/k``; since
    ``max_k p_k >= p_r >= L_r`` for every ``r``, the largest of those bounds is a
    simultaneous ``1-alpha`` lower bound on the maximum, by Bonferroni. It is
    conservative by construction, which is the right direction for a claim that
    a benchmark hands out *at least* this much free credit.
    """
    by_cluster = defaultdict(list)
    for index, item in enumerate(items):
        by_cluster[item["cluster"] or f"item-{index}"].append(item["rank"])
    keys = list(by_cluster)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    draws = [[] for _ in range(k)]
    for _ in range(reps):
        counts = Counter()
        total = 0
        for key in (rng.choice(keys) for _ in keys):
            for rank in by_cluster[key]:
                counts[rank] += 1
                total += 1
        if not total:
            continue
        for r in range(k):
            draws[r].append(counts.get(r, 0) / total)
    if not draws[0]:
        return None
    level = alpha / k                      # Bonferroni over the k shares
    bounds = []
    for r in range(k):
        draws[r].sort()
        bounds.append(draws[r][max(0, int(math.floor(level * len(draws[r]))))])
    best = max(range(k), key=lambda r: bounds[r])
    return {"per_rank_lower_bound": {str(r): bounds[r] for r in range(k)},
            "argmax_rank": best, "lower_bound_on_worst_case": bounds[best],
            "credit_lower_bound": bounds[best] - 1 / k,
            "alpha": alpha, "bonferroni_level": level, "n_groups": len(keys)}


def grouped_cv_credit(items, k, reps, folds=10, seed=SEED):
    """K-fold over clusters: select the rank on the training folds, score held out.

    Leave-one-out is the wrong estimator here and fails loudly. When two ranks
    are nearly tied, removing a single held-out item drops its own rank's count
    below the other's, so the selector picks the other rank -- for every item in
    turn. On MedQA that produced an accuracy of exactly zero, which is a
    property of the estimator and not of the benchmark. Folds are therefore
    groups of clusters, never single items, so the rank depleted in training is
    not by construction the rank about to be scored.

    Clusters are the grouping unit where a benchmark provides one; otherwise
    items are grouped at random, which still avoids the singleton pathology.
    """
    by_cluster = defaultdict(list)
    for index, item in enumerate(items):
        by_cluster[item["cluster"] or f"item-{index}"].append(item["rank"])
    keys = sorted(by_cluster)
    if len(keys) < 2:
        return None
    folds = max(2, min(folds, len(keys)))
    random.Random(seed).shuffle(keys)
    assignment = {key: i % folds for i, key in enumerate(keys)}

    scored, selected = {}, Counter()
    for fold in range(folds):
        held = [key for key in keys if assignment[key] == fold]
        train = Counter(r for key in keys if assignment[key] != fold
                        for r in by_cluster[key])
        if not train or not held:
            continue
        best = min(range(k), key=lambda r: (-train.get(r, 0), r))
        selected[best] += 1
        for key in held:
            scored[key] = [1.0 if r == best else 0.0 for r in by_cluster[key]]
    if not scored:
        return None
    flat = [x for v in scored.values() for x in v]
    lo, hi = cluster_bootstrap(scored, reps, seed)
    return {"accuracy": statistics.fmean(flat), "ci95": [lo, hi],
            "credit": statistics.fmean(flat) - 1 / k,
            "credit_ci95": [lo - 1 / k, hi - 1 / k],
            "n_folds": folds, "n_groups": len(keys),
            "rank_selected_per_fold": dict(selected)}


def cluster_bootstrap(scored, reps, seed=SEED, alpha=0.05):
    rng = random.Random(seed)
    keys = list(scored)
    draws = []
    for _ in range(reps):
        pooled = [x for _ in keys for x in scored[rng.choice(keys)]]
        if pooled:
            draws.append(statistics.fmean(pooled))
    if not draws:
        return None, None
    draws.sort()
    return (draws[max(0, int(math.floor((alpha / 2) * len(draws))))],
            draws[min(len(draws) - 1, int(math.ceil((1 - alpha / 2) * len(draws))) - 1)])


def chi_square_uniform(counts, k, n):
    expected = n / k
    chi2 = sum((counts.get(r, 0) - expected) ** 2 / expected for r in range(k))
    return {"chi2": chi2, "df": k - 1, "p_value": _chi2_sf(chi2, k - 1)}


def _solve(matrix, vector):
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(vector)
    a = [list(row) + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-15:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r != col:
                f = a[r][col] / a[col][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
    return [a[i][n] / a[i][i] for i in range(n)]


def _betacf(a, b, x):
    """Continued fraction for the regularised incomplete beta (Numerical Recipes)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _f_sf(f, d1, d2):
    """P(F(d1, d2) > f)."""
    if f <= 0:
        return 1.0
    return _betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def cluster_robust_uniformity(items, k):
    r"""A Wald test of a uniform key rank that resamples nothing and trusts no item's independence.

    The $\chi^2$ above treats every item as an independent draw, which a file built in
    capsules or subjects is not: items that share a source can share a rank pattern. This
    replaces the multinomial covariance of the rank shares with the cluster-robust one,
    $\hat V=\tfrac{C}{C-1}\sum_c r_cr_c^\top/N^2$ with $r_c$ cluster $c$'s rank counts minus
    their expectation under the estimated shares, and refers
    $W=(\hat p-u)^\top\hat V^{-1}(\hat p-u)$ over the first $k-1$ shares to an $F$ with
    $(k-1,\,C-k+1)$ degrees of freedom after the usual small-sample scaling. A file without
    recorded clusters has each item as its own.
    """
    by_cluster = defaultdict(lambda: [0] * k)
    for index, item in enumerate(items):
        by_cluster[item["cluster"] or f"item-{index}"][item["rank"]] += 1
    counts = list(by_cluster.values())
    n_clusters, total = len(counts), sum(sum(c) for c in counts)
    if n_clusters <= k - 1:
        return {"n_clusters": n_clusters, "p_value": None, "note": "too few clusters for the test"}
    share = [sum(c[j] for c in counts) / total for j in range(k)]
    m = k - 1
    cov = [[0.0] * m for _ in range(m)]
    for c in counts:
        size = sum(c)
        resid = [c[j] - size * share[j] for j in range(m)]
        for i in range(m):
            for j in range(m):
                cov[i][j] += resid[i] * resid[j]
    scale = n_clusters / (n_clusters - 1) / total ** 2
    cov = [[x * scale for x in row] for row in cov]
    diff = [share[j] - 1.0 / k for j in range(m)]
    solved = _solve(cov, diff)
    if solved is None:
        return {"n_clusters": n_clusters, "p_value": None, "note": "singular covariance"}
    wald = sum(d * s for d, s in zip(diff, solved))
    d2 = n_clusters - m
    f = wald * d2 / (m * (n_clusters - 1))
    return {"wald": wald, "f": f, "df": [m, d2], "n_clusters": n_clusters, "p_value": _f_sf(f, m, d2)}


def null_in_sample_credit(n, k, reps=20000, seed=SEED):
    """What the in-sample best rank rule scores over chance when the key rank is uniform:
    the upward bias of taking a maximum over $k$ estimated shares, at this file's size."""
    rng = random.Random(seed)
    credit = []
    for _ in range(reps):
        counts = Counter(rng.randrange(k) for _ in range(n))
        credit.append(max(counts.values()) / n - 1.0 / k)
    credit.sort()
    return {"mean": statistics.fmean(credit), "q95": credit[int(0.95 * reps)]}


def _chi2_sf(x, df):
    from choice_model import _chi2_sf as exact
    return exact(x, df)


# --------------------------------------------------------------------------
def length_ranks(options):
    """Rank of each option by how long it is written out; ties break on the text.

    A second coordinate of the same channel, and the cheap one: the bound in
    ``choice_model.geometry_bounds`` never used the options being numbers, so
    the key's rank under *any* ordering has a distribution and a worst case.
    This one needs no parsing, so it is measurable on every item of every file,
    numeric or not.
    """
    scores = [float(len(str(o))) for o in options]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], str(options[i])))
    return [order.index(i) for i in range(len(options))]


def survey_one(name, records, reps):
    records = list(records)
    if not records:
        return None
    counts_by_k = Counter(r["n_options"] for r in records)
    k, n_at_k = counts_by_k.most_common(1)[0]

    items = []
    for record in records:
        if record["n_options"] != k:
            continue
        ranks = numeric_ranks(record["options"])
        if ranks is None:
            continue
        items.append({"rank": ranks[record["key"]], "cluster": record.get("cluster")})

    chance = 1 / k
    report = {"benchmark": name, "n_items": len(records), "modal_n_options": k,
              "n_items_at_modal_k": n_at_k, "n_numeric_items": len(items),
              "share_numeric": len(items) / n_at_k if n_at_k else 0.0,
              "chance": chance,
              "n_clusters": len({i["cluster"] for i in items if i["cluster"]}) or None}
    if len(items) < 20:
        report["verdict"] = "too few numeric items to assess"
        return report

    ranks = Counter(i["rank"] for i in items)
    shares = [ranks.get(r, 0) / len(items) for r in range(k)]
    interior = sum(shares[1:k - 1])
    report.update({
        "key_by_rank": {str(r): shares[r] for r in range(k)},
        "interior_rate": interior, "interior_expected": (k - 2) / k,
        "chi_square_vs_uniform": chi_square_uniform(ranks, k, len(items)),
        "cluster_robust_vs_uniform": cluster_robust_uniformity(items, k),
        "null_in_sample_credit": null_in_sample_credit(len(items), k),
        "plug_in_credit": max(shares) - chance,
        "plug_in_worst_case_score": max(shares),
        "least_favoured_rank_share": min(shares),
        "credit_bound": credit_lower_bound(items, k, reps),
        "cross_validated": grouped_cv_credit(items, k, reps),
    })
    # The same measurement under written length, over every item at the modal
    # option count rather than the numeric subset: a benchmark whose options are
    # words has no value channel but can still have this one.
    length_items = [{"rank": length_ranks(r["options"])[r["key"]],
                     "cluster": r.get("cluster")}
                    for r in records if r["n_options"] == k]
    length_shares = Counter(i["rank"] for i in length_items)
    length_bound = credit_lower_bound(length_items, k, reps)
    report["length_channel"] = {
        "n_items": len(length_items),
        "key_by_rank": {str(r): length_shares.get(r, 0) / len(length_items)
                        for r in range(k)},
        "plug_in_credit": max(length_shares.get(r, 0) for r in range(k)) / len(length_items)
                          - chance,
        "credit_bound": length_bound,
        "exploitable": bool(length_bound and length_bound["credit_lower_bound"] > 0),
    }

    bound = report["credit_bound"]
    report["exploitable"] = bool(bound and bound["credit_lower_bound"] > 0)
    # Resampling a handful of clusters gives a bootstrap distribution with very
    # few distinct values, so the bound is not trustworthy there. Flagged rather
    # than dropped, and counted separately in the manuscript.
    report["bootstrap_reliable"] = bool(bound and bound["n_groups"] >= 10)
    report["verdict"] = (("channel open" if report["bootstrap_reliable"]
                          else "channel open (too few groups to trust)")
                         if report["exploitable"] else "not demonstrated")
    return report


def targets():
    yield "BixBench v1.5", bixbench_v15()
    yield "BixBench v1.0", bixbench_v10()
    labbench_dir = Path("data/external/labbench")
    for path in sorted(labbench_dir.glob("*.jsonl")):
        yield f"LAB-Bench {path.stem}", labbench(path)
    pretty = {"sciq": "SciQ", "mmlu": "MMLU", "mmlu_pro": "MMLU-Pro",
              "aqua_rat": "AQuA-RAT", "medmcqa": "MedMCQA",
              "medqa_usmle": "MedQA-USMLE", "openbookqa": "OpenBookQA",
              "arc_challenge": "ARC-Challenge"}
    for path in sorted(SURVEY.glob("*.jsonl*")):
        stem = path.name.split(".")[0]
        yield pretty.get(stem, stem), read_records(path)


ORDER = ("BixBench v1.5", "BixBench v1.0", "LAB-Bench SeqQA", "SciQ", "MMLU", "MMLU-Pro",
         "MedMCQA", "MedQA-USMLE", "AQuA-RAT")


def _p(value):
    """A p-value as printed: two significant figures, powers of ten below 0.001."""
    if value is None:
        return "--"
    if value < 1e-3:
        exponent = math.floor(math.log10(value))
        mantissa = round(value / 10 ** exponent)
        if mantissa == 10:
            mantissa, exponent = 1, exponent + 1
        return f"${mantissa}\\times10^{{{exponent}}}$"
    return f"${value:.2g}$"


def table_rows(doc):
    """tab:survey: every assessed file, the floor, the rule in-sample and held out, both tests."""
    by_name = {b["benchmark"]: b for b in doc["benchmarks"]}
    rows = []
    for name in ORDER:
        b = by_name[name]
        k = b["modal_n_options"]
        clusters = b["cluster_robust_vs_uniform"]["n_clusters"]
        own = b["n_clusters"] is not None
        rows.append(
            f"{name.replace('LAB-Bench ', '').replace('-USMLE', '')} & ${k}$ & ${b['n_numeric_items']}$ & "
            f"{('$' + str(clusters) + '$') if own else '--'} & ${100 * b['interior_rate']:.0f}$ & "
            f"${100 * (b['interior_rate'] / (k - 2) - 1 / k):+.1f}$ & ${100 * b['plug_in_credit']:+.1f}$ & "
            f"${100 * b['null_in_sample_credit']['mean']:+.1f}$ & ${100 * b['cross_validated']['credit']:+.1f}$ & "
            f"{_p(b['chi_square_vs_uniform']['p_value'])} & {_p(b['cluster_robust_vs_uniform']['p_value'])}\\\\")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/channel_survey.json"))
    parser.add_argument("--latex", action="store_true", help="print tab:survey's rows from the report")
    args = parser.parse_args()
    if args.latex:
        for line in table_rows(json.loads(args.output.read_text())):
            print(line)
        return

    rows = []
    for name, records in targets():
        report = survey_one(name, records, args.reps)
        if report:
            rows.append(report)
            print(f"  {name}", flush=True)

    assessed = [r for r in rows if "exploitable" in r]
    open_channel = [r for r in assessed if r["exploitable"]]
    doc = {
        "method": ("key-rank distribution over items whose k options are all distinct numbers; "
                   "credit is the best rank-only solver's score minus 1/k, estimated both as a "
                   "plug-in maximum and by leave-one-cluster-out selection"),
        "bootstrap_reps": args.reps,
        "summary": {
            "n_benchmark_files": len(rows),
            "n_assessed": len(assessed),
            "n_channel_open": len(open_channel),
            "n_channel_open_with_enough_groups": sum(
                1 for r in open_channel if r.get("bootstrap_reliable")),
            "n_rejecting_uniformity": sum(
                1 for r in assessed
                if r["chi_square_vs_uniform"]["p_value"] < 0.05),
            "n_rejecting_uniformity_cluster_robust": sum(
                1 for r in assessed
                if (r["cluster_robust_vs_uniform"].get("p_value") or 1.0) < 0.05),
            "median_plug_in_credit": (statistics.median([r["plug_in_credit"] for r in assessed])
                                      if assessed else None),
            "median_credit_lower_bound": (
                statistics.median([r["credit_bound"]["credit_lower_bound"] for r in assessed
                                   if r.get("credit_bound")]) if assessed else None),
            "largest_credit_lower_bound": max(
                ((r["credit_bound"]["credit_lower_bound"], r["benchmark"])
                 for r in assessed if r.get("credit_bound")), default=None),
        },
        "benchmarks": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'benchmark':22s} {'k':>2s} {'numeric':>8s} {'grp':>5s} "
          f"{'interior':>9s} {'chi2':>8s} {'plug-in':>8s} {'>= bound':>9s}  verdict")
    for r in rows:
        if "exploitable" not in r:
            print(f"{r['benchmark'][:22]:22s} {r['modal_n_options']:2d} "
                  f"{r['n_numeric_items']:8d} {'':>5s} {'':>9s} {'':>8s} {'':>8s} "
                  f"{'':>9s}  {r['verdict']}")
            continue
        bound = r["credit_bound"]
        print(f"{r['benchmark'][:22]:22s} {r['modal_n_options']:2d} "
              f"{r['n_numeric_items']:8d} {bound['n_groups']:5d} "
              f"{r['interior_rate']:9.1%} {r['chi_square_vs_uniform']['chi2']:8.1f} "
              f"{r['plug_in_credit'] * 100:+8.1f} "
              f"{bound['credit_lower_bound'] * 100:+9.1f}  {r['verdict']}")
    print(f"\nchannel open in {len(open_channel)} of {len(assessed)} assessed files; "
          f"median lower bound {doc['summary']['median_credit_lower_bound'] * 100:+.1f} points")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
