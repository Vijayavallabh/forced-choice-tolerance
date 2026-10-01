#!/usr/bin/env python3
"""Turn the raw no-data probe outcomes into the reported statistics.

``no_data_probe.py`` needs a GPU; this does not. It reads the per-item outcome
files in ``results/probe_raw/`` and produces ``results/no_data_probe.json``
using the standard library alone, so every number in the manuscript can be
re-derived from the committed raw outcomes without accelerators.

The design is paired: one item appears in every arm, so the contrast between
arms is a within-item difference and inference is over capsules, not questions.
Two procedures are used, both clustering on the capsule:

  * a paired cluster bootstrap for intervals, resampling whole capsules; and
  * a paired sign-flip randomisation test, which reverses the arm labels of all
    items in a capsule at once. That is the exchangeability the design actually
    licenses -- the redraw is assigned to items, and items are nested in
    capsules -- so it does not borrow independence the data do not have.

The decomposition at the end is the point of the whole exercise. If a solver
picks sorted rank r with probability q_r while the benchmark places its key at
rank r with probability p_r, a solver that knows nothing still scores
sum_r q_r p_r. Comparing that with the observed score separates the part of a
no-data margin that is option geometry from the part that needs an explanation.
"""
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

N_OPTIONS = 4               # BixBench's option count, a default only
CHANCE = 1.0 / N_OPTIONS


def option_count(rows):
    """The option count of a probe file, from the per-letter probabilities."""
    return rows[0].get("n_options") or len(rows[0]["letter_probs"])
ARMS = ("original", "placebo", "repaired", "original_stemless", "repaired_stemless")
RAW = Path("results/probe_raw")
OUT = Path("results/no_data_probe.json")
SEED = 20260920


# --------------------------------------------------------------------------
def mean(values):
    return statistics.fmean(values) if values else None


def cluster_bootstrap_ci(clusters, reps=10000, alpha=0.05, seed=SEED):
    """Percentile interval for a mean, resampling whole capsules."""
    rng = random.Random(seed)
    keys = list(clusters)
    if not keys:
        return None, None
    draws = []
    for _ in range(reps):
        pooled = [x for _ in keys for x in clusters[rng.choice(keys)]]
        if pooled:
            draws.append(statistics.fmean(pooled))
    if not draws:
        return None, None
    draws.sort()
    lo = draws[max(0, int(math.floor((alpha / 2) * len(draws))))]
    hi = draws[min(len(draws) - 1, int(math.ceil((1 - alpha / 2) * len(draws))) - 1)]
    return lo, hi


def sign_flip_test(clusters, reps=10000, seed=SEED):
    """Randomisation p-value for a mean paired difference, flipping per capsule."""
    rng = random.Random(seed)
    keys = list(clusters)
    flat = [x for k in keys for x in clusters[k]]
    if not flat:
        return None
    observed = statistics.fmean(flat)
    extreme = 0
    for _ in range(reps):
        total = count = 0.0
        for k in keys:
            s = -1.0 if rng.random() < 0.5 else 1.0
            for x in clusters[k]:
                total += s * x
                count += 1
        if abs(total / count) >= abs(observed) - 1e-12:
            extreme += 1
    return {"difference": observed, "p_value": (extreme + 1) / (reps + 1), "reps": reps}


def icc(clusters):
    groups = [v for v in clusters.values() if v]
    k, n = len(groups), sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return None
    grand = sum(sum(g) for g in groups) / n
    ms_b = sum(len(g) * (mean(g) - grand) ** 2 for g in groups) / (k - 1)
    ms_w = sum(sum((x - mean(g)) ** 2 for x in g) for g in groups) / (n - k)
    m0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    denom = ms_b + (m0 - 1) * ms_w
    value = 0.0 if denom == 0 else max(0.0, (ms_b - ms_w) / denom)
    m_bar = n / k
    return {"icc": value, "design_effect": 1 + (m_bar - 1) * value,
            "effective_n": n / (1 + (m_bar - 1) * value),
            "n_items": n, "n_clusters": k}


# --------------------------------------------------------------------------
def load(path):
    from no_data_probe import open_outcomes
    with open_outcomes(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def per_item(rows, arm, predicate=None):
    """Average each item over its draws, keyed by question id."""
    acc, cluster = defaultdict(list), {}
    for row in rows:
        if row["arm"] != arm or (predicate and not predicate(row)):
            continue
        acc[row["question_id"]].append(row["correct"])
        cluster[row["question_id"]] = row["cluster"]
    return {qid: mean(v) for qid, v in acc.items()}, cluster


def by_cluster(scores, clusters):
    out = defaultdict(list)
    for qid, value in scores.items():
        out[clusters[qid]].append(value)
    return dict(out)


def arm_summary(rows, arm, predicate, reps):
    scores, clusters = per_item(rows, arm, predicate)
    if not scores:
        return None
    grouped = by_cluster(scores, clusters)
    lo, hi = cluster_bootstrap_ci(grouped, reps=reps)
    return {"arm": arm, "n_items": len(scores), "n_clusters": len(grouped),
            "accuracy": mean(list(scores.values())), "ci95": [lo, hi],
            "clustering": icc(grouped)}


def paired_contrast(rows, arm_a, arm_b, predicate, reps):
    """Within-item difference arm_a - arm_b, with capsule-level inference."""
    a, clusters = per_item(rows, arm_a, predicate)
    b, _ = per_item(rows, arm_b, predicate)
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    diffs = {qid: a[qid] - b[qid] for qid in shared}
    grouped = by_cluster(diffs, clusters)
    lo, hi = cluster_bootstrap_ci(grouped, reps=reps)
    test = sign_flip_test(grouped, reps=reps)
    return {"contrast": f"{arm_a} - {arm_b}", "n_items": len(shared),
            "n_clusters": len(grouped), "difference": mean(list(diffs.values())),
            "ci95": [lo, hi], "p_value": test["p_value"] if test else None,
            "accuracy_a": mean([a[q] for q in shared]),
            "accuracy_b": mean([b[q] for q in shared])}


# --------------------------------------------------------------------------
def rank_choice_distribution(rows, arm, predicate=None, k=None):
    """How often the solver picks each sorted rank, and how often the key is there."""
    k = k or option_count(rows)
    chosen, keys = Counter(), Counter()
    for row in rows:
        if row["arm"] != arm or row["chosen_rank"] is None or row["key_rank"] is None:
            continue
        if predicate and not predicate(row):
            continue
        chosen[row["chosen_rank"]] += 1
        keys[row["key_rank"]] += 1
    total = sum(chosen.values())
    if not total:
        return None
    return {"n": total, "n_options": k,
            "choice_by_rank": {str(r): chosen.get(r, 0) / total for r in range(k)},
            "key_by_rank": {str(j): keys.get(j, 0) / total for j in range(k)}}


def decomposition(rows, arm, bias_arm, predicate=None, k=None):
    """Split a no-data score into option geometry and everything else.

    ``q_r`` is the solver's preference over sorted ranks, taken from
    ``bias_arm``; ``p_r`` is where the benchmark puts its key in ``arm``. Their
    inner product is what a solver scores from geometry alone. The residual is
    the part of the observed score that geometry does not explain.
    """
    k = k or option_count(rows)
    target = rank_choice_distribution(rows, arm, predicate, k)
    bias = rank_choice_distribution(rows, bias_arm, predicate, k)
    if not (target and bias):
        return None
    chance = 1.0 / k
    q = [bias["choice_by_rank"][str(r)] for r in range(k)]
    p = [target["key_by_rank"][str(j)] for j in range(k)]
    predicted = sum(qi * pi for qi, pi in zip(q, p))
    observed = mean([row["correct"] for row in rows
                     if row["arm"] == arm and row["key_rank"] is not None
                     and (not predicate or predicate(row))])
    return {"arm": arm, "bias_estimated_from": bias_arm, "n_options": k,
            "choice_by_rank_q": {str(r): q[r] for r in range(k)},
            "key_by_rank_p": {str(j): p[j] for j in range(k)},
            "predicted_from_geometry": predicted, "observed": observed,
            "residual": observed - predicted, "chance": chance,
            "geometry_share_of_margin": (None if abs(observed - chance) < 1e-9
                                         else (predicted - chance) / (observed - chance))}


# --------------------------------------------------------------------------
# Parameter counts in billions, from each model's own config (hidden size,
# layers and vocabulary), used only to order the panel for the trend test.
PARAMETERS_B = {
    "meta-llama/Llama-3.2-1B-Instruct": 1.24,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.54,
    "meta-llama/Llama-3.2-3B-Instruct": 3.21,
    "microsoft/Phi-3.5-mini-instruct": 3.82,
    "google/gemma-3-4b-it": 4.30,
    "allenai/OLMo-2-1124-7B-Instruct": 7.30,
    "Qwen/Qwen2.5-7B-Instruct": 7.62,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 8.03,
    "Groq/Llama-3-Groq-8B-Tool-Use": 8.03,
    "Qwen/Qwen2.5-14B-Instruct": 14.77,
    "Qwen/Qwen2.5-32B-Instruct": 32.76,
    "meta-llama/Llama-3.3-70B-Instruct": 70.6,
    "Qwen/Qwen2.5-72B-Instruct": 72.7,
}


def spearman(xs, ys):
    def ranked(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    rx, ry = ranked(xs), ranked(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def scale_trend(models, contrast, label, reps=100000, seed=SEED, quantity="contrast"):
    """One pre-specified test, instead of eleven.

    Reading eleven per-model contrasts at the 5% level will turn up significant
    results in both directions by chance, and it does here. Capability order is
    fixed in advance and independent of the outcome, so the panel supports a
    single rank-correlation test between parameter count and the size of the
    effect. The p-value is exact by permutation of the model labels.
    """
    rows = []
    for m in models:
        size = PARAMETERS_B.get(m["model"])
        if size is None:
            continue
        if quantity == "arm":
            found = m["subsets"][label]["arms"].get(contrast)
            value = found["accuracy"] if found else None
        else:
            found = next((c for c in m["subsets"][label]["contrasts"]
                          if c["contrast"] == contrast), None)
            value = found["difference"] if found else None
        if value is not None:
            rows.append((size, value, m["model"]))
    if len(rows) < 5:
        return None
    sizes = [r[0] for r in rows]
    effects = [r[1] for r in rows]
    observed = spearman(sizes, effects)
    rng = random.Random(seed)
    shuffled = list(effects)
    extreme = 0
    for _ in range(reps):
        rng.shuffle(shuffled)
        if abs(spearman(sizes, shuffled)) >= abs(observed) - 1e-12:
            extreme += 1
    return {"contrast": contrast, "subset": label, "n_models": len(rows),
            "spearman_rho": observed, "p_value": (extreme + 1) / (reps + 1),
            "reps": reps,
            "per_model": [{"model": m, "parameters_b": s, "effect": e}
                          for s, e, m in sorted(rows)]}


NUMERIC = lambda row: row["is_numeric"]
NUMERIC_REDRAWN = lambda row: row["is_numeric"] and row["redraw_applied"]
NON_NUMERIC = lambda row: not row["is_numeric"]
SUBSETS = (("all items", None), ("numeric options", NUMERIC),
           ("numeric, redraw feasible", NUMERIC_REDRAWN),
           ("non-numeric options", NON_NUMERIC))
CONTRASTS = (("original", "repaired"), ("original", "placebo"),
             ("placebo", "repaired"), ("original_stemless", "repaired_stemless"),
             ("original", "original_stemless"))


def analyse_model(path, reps):
    rows = load(path)
    name = rows[0]["model"]
    report = {"model": name, "n_rows": len(rows), "n_options": option_count(rows),
              "n_draws": 1 + max(r["draw"] for r in rows), "subsets": {}}
    for label, predicate in SUBSETS:
        block = {"arms": {}, "contrasts": []}
        for arm in ARMS:
            summary = arm_summary(rows, arm, predicate, reps)
            if summary:
                block["arms"][arm] = summary
        for arm_a, arm_b in CONTRASTS:
            contrast = paired_contrast(rows, arm_a, arm_b, predicate, reps)
            if contrast:
                block["contrasts"].append(contrast)
        report["subsets"][label] = block
    report["rank_bias"] = {arm: rank_choice_distribution(rows, arm, NUMERIC) for arm in ARMS}
    report["decomposition"] = {
        "original_bias_from_stemless": decomposition(rows, "original",
                                                     "original_stemless", NUMERIC),
        "original_bias_from_self": decomposition(rows, "original", "original", NUMERIC),
        "repaired_bias_from_stemless": decomposition(rows, "repaired",
                                                     "repaired_stemless", NUMERIC),
    }
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    from no_data_probe import outcome_files
    paths = outcome_files(args.raw)
    if not paths:
        raise SystemExit(f"no raw outcome files in {args.raw}")

    models = []
    for path in paths:
        print(f"analysing {path.name}", flush=True)
        models.append(analyse_model(path, args.reps))

    # Pooled across the panel: the mean per-model effect, so a model with more
    # items cannot dominate. Each model contributes one number per quantity.
    pooled = {}
    for label, _ in SUBSETS:
        block = {}
        for arm in ARMS:
            values = [m["subsets"][label]["arms"][arm]["accuracy"] for m in models
                      if arm in m["subsets"][label]["arms"]]
            if values:
                block[arm] = {"mean_over_models": mean(values), "n_models": len(values),
                              "min": min(values), "max": max(values)}
        contrasts = {}
        for arm_a, arm_b in CONTRASTS:
            key = f"{arm_a} - {arm_b}"
            values = [c["difference"] for m in models
                      for c in m["subsets"][label]["contrasts"] if c["contrast"] == key]
            ps = [c["p_value"] for m in models
                  for c in m["subsets"][label]["contrasts"] if c["contrast"] == key]
            if values:
                contrasts[key] = {"mean_over_models": mean(values), "n_models": len(values),
                                  "min": min(values), "max": max(values),
                                  "n_models_p_below_0.05": sum(p < 0.05 for p in ps)}
        pooled[label] = {"arms": block, "contrasts": contrasts}

    trends = {}
    for contrast in ("placebo - repaired", "original - repaired"):
        rec = scale_trend(models, contrast, "numeric options")
        if rec:
            trends[contrast] = rec
    for arm in ("original", "original_stemless"):
        rec = scale_trend(models, arm, "numeric options", quantity="arm")
        if rec:
            trends[f"accuracy in arm: {arm}"] = rec

    doc = {
        "design": {
            "arms": list(ARMS),
            "chance": (1.0 / models[0]["n_options"]) if models else CHANCE,
            "unit_of_analysis": "question, averaged over redraws; clustered on capsule",
            "inference": ("paired cluster bootstrap for intervals; paired sign-flip "
                          "randomisation over capsules for p-values"),
            "bootstrap_reps": args.reps,
        },
        "models": models,
        "pooled_over_models": pooled,
        "scale_trend": trends,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")

    for contrast, rec in trends.items():
        print(f"\nscale trend for '{contrast}' on numeric items: "
              f"Spearman rho = {rec['spearman_rho']:+.3f} over {rec['n_models']} models, "
              f"permutation p = {rec['p_value']:.4f}")

    print(f"\n{'model':44s} {'orig':>7s} {'placebo':>8s} {'repaired':>9s} "
          f"{'o-r':>7s} {'p':>7s}")
    for m in models:
        block = m["subsets"]["numeric options"]
        arms = block["arms"]
        contrast = next((c for c in block["contrasts"]
                         if c["contrast"] == "original - repaired"), None)
        print(f"{m['model']:44s} {arms['original']['accuracy']:7.1%} "
              f"{arms['placebo']['accuracy']:8.1%} {arms['repaired']['accuracy']:9.1%} "
              f"{contrast['difference']:+7.1%} {contrast['p_value']:7.4f}")


if __name__ == "__main__":
    main()
