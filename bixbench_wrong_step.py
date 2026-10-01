#!/usr/bin/env python3
r"""\textsc{wrong step} on BixBench: the frontier's closest row, on this workshop's file.

Table~\ref{tab:frontier} is on MMLU, and a claim about repairing a
science-agent benchmark has to be tested on one. This gathers the
same three readings for BixBench's $105$ numeric items that the frontier reports
for its $464$:

* the key-rank law each arm realises (``mcq_audit.rank_of``);
* the set leak -- ``learned_probe.py``'s ``every feature`` reader, held out by
  capsule, over the worst of twenty clean files of the same size -- read from
  ``results/learned_probe_bixbench_wrong_step.json``;
* the difficulty column -- three solvers given the question and not the data
  (``frontier_validity.py``) -- each file's own nominal interval reproduced from
  its per-item answers and then read at the level that covers at this shape,
  measured with the frontier's own ``covering_level`` on the released arm.

The writer is the frontier's: Qwen2.5-14B shown the question and nothing else,
and Llama-3.1-8B as the second generator (``keyblind_operators.py --jsonl
build/bixbench_numeric_q.jsonl``). What the frontier's prediction says should
change is the writer's ability to make the mistake: on MMLU it can work the
item, so its wrong values sit around the key; here the answer is a property of
data it is not shown.

    python3 bixbench_wrong_step.py
"""
import argparse
import json
import pathlib
from collections import Counter

import numpy as np

import frontier_calibrated as fc
from mcq_audit import rank_of

ARMS = {"released": "build/bixbench_numeric_q.jsonl",
        "wrong step (Qwen2.5-14B)": "build/bixbench_numeric_wrong_step.jsonl",
        "wrong step (Llama-3.1-8B)": "build/bixbench_numeric_wrong_step_llama.jsonl",
        "placebo": "build/bixbench_numeric_placebo.jsonl",
        "repaired": "build/bixbench_numeric_repaired.jsonl"}
# validity file, arm name inside it, label here
DIFFICULTY = [("results/bixbench_numeric_validity.json", "wrong-step", "wrong step (Qwen2.5-14B)"),
              ("results/bixbench_numeric_validity.json", "placebo", "placebo"),
              ("results/bixbench_numeric_validity.json", "repaired", "repaired"),
              ("results/bixbench_numeric_validity_llamaws.json", "wrong-step-llama",
               "wrong step (Llama-3.1-8B)")]


def rank_law(path):
    rows = [json.loads(l) for l in open(path)]
    ranks = Counter(rank_of([r["ideal"], *r["distractors"]]) for r in rows)
    n = sum(v for k, v in ranks.items() if k is not None)
    return {"n_items": len(rows), "n_ranked": n,
            "key_rank_pct": [100.0 * ranks.get(i, 0) / n for i in range(4)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default="results/learned_probe_bixbench_wrong_step.json")
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--validity-reps", type=int, default=4000)
    ap.add_argument("--validity-seed", type=int, default=20260919)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--output", default="results/bixbench_wrong_step.json")
    args = ap.parse_args()

    report = {"arms": {}}
    for label, path in ARMS.items():
        report["arms"][label] = {"path": path, **rank_law(path)}
        law = report["arms"][label]["key_rank_pct"]
        print(f"{label:28s} key rank {' '.join(f'{p:5.1f}' for p in law)}")

    probe = json.load(open(args.probe))
    worst = probe["clean"]["every feature"]["max"]
    report["clean_worst_case"] = worst
    for label, entry in probe["files"].items():
        ef = entry["every feature"]
        report["arms"][label]["set_leak"] = {
            "accuracy": 100 * ef["cv_accuracy"], "ci95": [100 * x for x in ef["cv_ci95"]],
            "over_worst_case": 100 * (ef["cv_accuracy"] - worst),
            "over_worst_case_ci95": [100 * (x - worst) for x in ef["cv_ci95"]]}
    print(f"\nset leak, every feature, over the clean worst case {100 * worst:.1f}%:")
    for label, arm in report["arms"].items():
        if "set_leak" not in arm:          # the placebo holds the key's rank; not re-read here
            continue
        s = arm["set_leak"]
        print(f"  {label:28s} {s['over_worst_case']:+6.1f} [{s['over_worst_case_ci95'][0]:+6.1f},"
              f"{s['over_worst_case_ci95'][1]:+6.1f}]")

    released = json.load(open(DIFFICULTY[0][0]))
    table = np.array([[sum(np.mean(h) for h, c in zip(released["models"][m]["released"]["hits"],
                                                      released["clusters"]) if c == cluster),
                       sum(1 for c in released["clusters"] if c == cluster)]
                      for m in [next(iter(released["models"]))]
                      for cluster in sorted(set(released["clusters"]))])
    level = fc.covering_level(table, args.reps, args.bootstrap, args.seed)
    report["level"] = level
    print(f"\ndifficulty given the question, read at {100 * level['level_for_95']:.2f}% "
          f"(covers 95% on {level['n_clusters']} capsules, {level['n_effective']:.1f} effective):")
    for path, arm, label in DIFFICULTY:
        doc = json.load(open(path))
        per_model = {}
        for model in doc["models"]:
            point, drawn = fc.difficulty_draws(doc, model, arm, args.validity_reps, args.validity_seed)
            nominal = [float(drawn[int(0.025 * len(drawn))]), float(drawn[int(0.975 * len(drawn))])]
            shipped = doc["models"][model][arm]["vs_reference"]
            if abs(point - shipped["points"]) > 1e-9 or any(
                    abs(a - b) > 1e-9 for a, b in zip(nominal, shipped["ci95"])):
                raise SystemExit(f"{label} {model}: file says {shipped}, recomputed {point} {nominal}")
            per_model[model] = {"points": point, "nominal_95": nominal,
                                "covering": fc.quantiles(drawn, level["level_for_95"]),
                                "n_items": doc["n_items"]}
        report["arms"][label]["difficulty"] = per_model
        print(f"  {label:28s} " + "   ".join(
            f"{m.split('/')[-1].replace('-Instruct', '')} {c['points']:+5.1f} "
            f"[{c['covering'][0]:+5.1f},{c['covering'][1]:+5.1f}]" for m, c in per_model.items()))
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
