#!/usr/bin/env python3
"""Does in-context exposure teach a solver to exploit option geometry?

Reads the arms written by ``icl_probe.py`` and answers three questions, each
with capsule-level inference:

  1. **Is the channel learnable at all?** Compare the stem-withheld arm with
     ``n`` original demonstrations against the same arm with none. With no
     question text there is nothing to know, so any gain is geometry.
  2. **Is it the ranks that are learned?** Compare original demonstrations
     against repaired ones -- same examples, same letters, same count, key
     ranks made uniform. This is the contrast that licenses a causal claim;
     the zero-shot comparison alone would also move if examples merely taught
     formatting or nudged the solver away from a letter habit.
  3. **How much of the channel does it capture?** The chosen-rank distribution
     gives ``b`` directly, so ``G = <p, b>`` is the credit collected, to be
     read against the ``max_j p_j - 1/k`` the benchmark leaves open.

The shot ladder is pre-specified and analysed with a Spearman trend test over
the shot counts, so a monotone increase is one test rather than several.
"""
import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path

from choice_model import geometry_bounds, realised_geometry
from probe_analysis import (PARAMETERS_B, by_cluster, cluster_bootstrap_ci, mean,
                            paired_contrast, per_item, spearman)

SEED = 20260920


def load(path):
    from no_data_probe import open_outcomes
    with open_outcomes(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def arm_name(stem, shots, pool):
    return f"{stem}_s{shots}_{'none' if shots == 0 else pool}"


def rank_preference(rows, arm):
    """The rank the solver picks, measured rather than inferred."""
    counts = Counter(r["chosen_rank"] for r in rows
                     if r["arm"] == arm and r["chosen_rank"] is not None)
    total = sum(counts.values())
    if not total:
        return None
    k = rows[0]["n_options"]
    return [counts.get(r, 0) / total for r in range(k)]


def key_p(rows):
    counts = Counter(r["key_rank"] for r in rows if r["key_rank"] is not None)
    total = sum(counts.values())
    k = rows[0]["n_options"]
    return [counts.get(j, 0) / total for j in range(k)]


def arm_record(rows, arm, reps, p):
    scores, clusters = per_item(rows, arm)
    if not scores:
        return None
    grouped = by_cluster(scores, clusters)
    lo, hi = cluster_bootstrap_ci(grouped, reps=reps)
    b = rank_preference(rows, arm)
    rec = {"arm": arm, "n_items": len(scores), "n_clusters": len(grouped),
           "accuracy": mean(list(scores.values())), "ci95": [lo, hi]}
    if b:
        rec["rank_preference_b"] = {str(r): b[r] for r in range(len(b))}
        rec["realised_geometry"] = realised_geometry(p, b)
    return rec


def trend(records, stem, shots, pool):
    """Spearman trend of accuracy over the pre-specified shot ladder.

    The ladder has only a handful of rungs, so the null distribution is
    enumerated exactly over all orderings rather than sampled: with four rungs
    the smallest attainable one-sided p is 1/24, and saying so is better than
    implying more resolution than four points can carry.
    """
    xs, ys = [], []
    for n in shots:
        rec = records.get(arm_name(stem, n, pool))
        if rec:
            xs.append(float(n))
            ys.append(rec["accuracy"])
    if len(xs) < 3:
        return None
    rho = spearman(xs, ys)
    orderings = list(itertools.permutations(ys))
    atleast = sum(1 for perm in orderings if spearman(xs, list(perm)) >= rho - 1e-12)
    return {"rho": rho, "p_value": atleast / len(orderings),
            "n_orderings": len(orderings), "alternative": "increasing in shots",
            "shots": xs, "accuracies": ys}


def analyse_model(path, shots, reps):
    rows = load(path)
    model = rows[0]["model"]
    k = rows[0]["n_options"]
    p = key_p(rows)
    bounds = geometry_bounds(p)

    records = {}
    for arm in sorted({r["arm"] for r in rows}):
        rec = arm_record(rows, arm, reps, p)
        if rec:
            records[arm] = rec

    contrasts = []
    for stem in sorted({r["stem"] for r in rows}):
        for n in [s for s in shots if s > 0]:
            for a, b_arm, label in (
                    (arm_name(stem, n, "original"), arm_name(stem, 0, "none"),
                     "learnable at all"),
                    (arm_name(stem, n, "original"), arm_name(stem, n, "repaired"),
                     "attributable to key ranks")):
                if a not in records or b_arm not in records:
                    continue
                rec = paired_contrast(rows, a, b_arm, None, reps)
                if rec:
                    rec.update({"stem": stem, "n_shots": n, "question": label})
                    contrasts.append(rec)

    trends = {}
    for stem in sorted({r["stem"] for r in rows}):
        for pool in ("original", "repaired"):
            t = trend(records, stem, shots, pool)
            if t:
                trends[f"{stem}_{pool}"] = t

    return {"model": model, "n_options": k, "n_items": len({r["question_id"] for r in rows}),
            "n_clusters": len({r["cluster"] for r in rows}),
            "geometry_bounds": bounds, "arms": records,
            "contrasts": contrasts, "shot_trend": trends}


def scale_trend(models, question, stem, n_shots, reps=100000, seed=SEED):
    """One pre-specified test over the panel, instead of one per model.

    A larger model is the one most likely to pick a preference up in context,
    so capability order is the natural alternative and is fixed in advance.
    Reading six per-model contrasts at the 5% level would turn up a significant
    one either way; this asks the question once, with an exact permutation
    p-value over the model labels.
    """
    rows = []
    for record in models:
        size = PARAMETERS_B.get(record["model"])
        if size is None:
            continue
        found = next((c for c in record["contrasts"]
                      if c["question"] == question and c["stem"] == stem
                      and c["n_shots"] == n_shots), None)
        if found:
            rows.append((size, found["difference"], record["model"]))
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
    return {"question": question, "stem": stem, "n_shots": n_shots,
            "n_models": len(rows), "spearman_rho": observed,
            "p_value": (extreme + 1) / (reps + 1),
            "per_model": [{"model": m, "parameters_b": s, "effect": e}
                          for s, e, m in sorted(rows)]}


SHORT = {"Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B", "microsoft/Phi-3.5-mini-instruct": "Phi-3.5-mini",
         "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B", "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
         "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B", "Qwen/Qwen2.5-32B-Instruct": "Qwen2.5-32B",
         "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-70B", "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B"}


def table_rows(doc, shots=64):
    """tab:icl (no longer in the paper; Figure~\ref{fig:norank}b instead): per model, in order of size, the no-example and the ``shots``-example accuracies with the
    question withheld, and the rank-isolating contrast with the question withheld and shown."""
    rows = []
    for rec in sorted(doc["models"], key=lambda m: PARAMETERS_B[m["model"]]):
        acc = lambda arm: f"${100 * rec['arms'][arm]['accuracy']:.1f}$"
        rank = {c["stem"]: c for c in rec["contrasts"]
                if c["question"] == "attributable to key ranks" and c["n_shots"] == shots}
        cell = lambda c: (f"${100 * c['difference']:+.1f}$ {{\\scriptsize$[{100 * c['ci95'][0]:+.1f},"
                          f"{100 * c['ci95'][1]:+.1f}]$}}")
        rows.append(f"{SHORT[rec['model']]} & ${PARAMETERS_B[rec['model']]:.1f}$ & {acc('withheld_s0_none')} & "
                    f"{acc(f'withheld_s{shots}_original')} & {acc(f'withheld_s{shots}_repaired')} & "
                    f"{cell(rank['withheld'])} & {cell(rank['shown'])}\\\\")
    return rows


def main():
    import sys
    if "--latex" in sys.argv:
        for row in table_rows(json.loads(Path("results/icl_probe.json").read_text(encoding="utf-8"))):
            print(row)
        return
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=Path("results/probe_icl"))
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 8, 32, 64])
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("results/icl_probe.json"))
    parser.add_argument("--raw", action="store_true", help="print and do not write")
    args = parser.parse_args()

    from no_data_probe import outcome_files
    models = []
    for path in outcome_files(args.dir):
        models.append(analyse_model(path, args.shots, args.reps))
        print(f"  analysed {models[-1]['model']}", flush=True)

    available = models[0]["geometry_bounds"]["max_geometry_credit"] if models else None
    trends = {}
    for question in ("learnable at all", "attributable to key ranks"):
        for stem in ("withheld", "shown"):
            rec = scale_trend(models, question, stem, max(args.shots))
            if rec:
                trends[f"{question} ({stem}, {max(args.shots)} shots)"] = rec

    doc = {"design": ("target item identical across arms; only the demonstrations before it "
                      "change. Stem withheld removes knowledge as an explanation; repaired "
                      "demonstrations remove the rank signal while holding count, format and "
                      "letter distribution fixed."),
           "geometry_credit_available": available,
           "bootstrap_reps": args.reps, "scale_trend": trends, "models": models}
    if not args.raw:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")

    print(f"\ngeometry credit available on this benchmark: {available:+.1%}\n")
    for rec in models:
        print(f"--- {rec['model']}  ({rec['n_items']} items, {rec['n_clusters']} capsules)")
        for arm, a in sorted(rec["arms"].items()):
            credit = a.get("realised_geometry", {}).get("geometry_credit")
            extra = f"  geometry {credit:+.1%}" if credit is not None else ""
            print(f"  {arm:28s} {a['accuracy']:6.1%} "
                  f"[{a['ci95'][0]:.1%}, {a['ci95'][1]:.1%}]{extra}")
        for c in rec["contrasts"]:
            print(f"  {c['contrast']:52s} {c['difference']:+6.1%} "
                  f"[{c['ci95'][0]:+.1%}, {c['ci95'][1]:+.1%}]  p={c['p_value']:.4f}")
        for name, t in sorted(rec["shot_trend"].items()):
            print(f"  trend {name:22s} rho={t['rho']:+.2f}  p={t['p_value']:.4f}  "
                  + " ".join(f"{y:.1%}" for y in t["accuracies"]))
        print()
    for name, rec in sorted(trends.items()):
        print(f"scale trend, {name}: rho={rec['spearman_rho']:+.2f}  "
              f"p={rec['p_value']:.4f}  over {rec['n_models']} models")


if __name__ == "__main__":
    main()
