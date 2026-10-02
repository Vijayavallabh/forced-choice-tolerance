#!/usr/bin/env python3
r"""The grid's template cell on BixBench v1.0, the release the published baselines ran on.

tab:gridbix (no longer in the paper) reads BixBench's own template, one letter at a time, on
v1.5 -- the release shipped now -- and five of thirteen models clear. The two
published zero-shot runs are on v1.0, whose rank law is a fifth as wide. What
the published baseline measured is decided by running the same cell on v1.0.
If it does not clear, "the field's instrument already
collects" is false of the published baseline; if it does, v1.0's narrow rank
ceiling has to be reconciled with it.

This reads ``run_bixv10.sh``'s dumps with the paper's own machinery:

* the margin and its nominal 95% interval are ``arm_intervals.draws``, the
  bootstrap every other arm's interval is, draw for draw;
* the level that covers 95% is measured for *this* file's shape -- its 53
  capsules at their own sizes -- with ``bootstrap_calibration``'s beta-binomial
  simulation, per model, and the widest (the most conservative) is the level
  every row is read at;
* the same cell is split into v1.0's 159 numeric items, where the rank law
  lives, and its 137 others, because the two halves can carry different
  channels and one number would hide which.

    python3 bixbench_v10_grid.py
"""
import argparse
import json
import pathlib

import numpy as np

import arm_intervals as ai
import bootstrap_calibration as bc
from mcq_audit import rank_of

MODELS = [  # tag, name, billions of parameters (Figure~\ref{fig:norank}a orders the models by it)
    ("llama1b", "Llama-3.2-1B", 1.2), ("qwen1.5b", "Qwen2.5-1.5B", 1.5),
    ("llama3b", "Llama-3.2-3B", 3.2), ("phi3.5mini", "Phi-3.5-mini", 3.8),
    ("gemma3_4b", "gemma-3-4b", 4.3), ("olmo2_7b", "OLMo-2-1124-7B", 7.3),
    ("qwen7b", "Qwen2.5-7B", 7.6), ("llama8b", "Meta-Llama-3.1-8B", 8.0),
    ("phi4", "phi-4", 14.7), ("qwen14b", "Qwen2.5-14B", 14.8),
    ("qwen32b", "Qwen2.5-32B", 32.8), ("llama70b", "Llama-3.3-70B", 70.6),
    ("qwen72b", "Qwen2.5-72B", 72.7)]


def covering_level(dump, reps, bootstrap, seed):
    """The nominal level whose coverage reaches 95% for a file-minus-control
    difference on this dump's own capsules, by ``bootstrap_calibration``'s simulation."""
    rate, spread, sizes = bc.fit_beta_binomial(bc.cluster_table(dump))
    c_rate, c_spread, c_sizes = bc.fit_beta_binomial(bc.cluster_table(dump, arm="clean"))
    rng = np.random.default_rng(seed)
    below = []
    for _ in range(reps):
        hits, trials = bc.simulate(rate, spread, sizes, None, rng)
        draws = bc.bootstrap_draws(hits, trials, bootstrap, rng)
        c_hits, c_trials = bc.simulate(c_rate, c_spread, c_sizes, None, rng)
        c_draws = bc.bootstrap_draws(c_hits, c_trials, bootstrap, rng)
        below.append(float(np.mean(draws - c_draws < rate - c_rate)))
    return bc.level_for(bc.coverage_curve(below)), bc.effective_clusters(sizes)


def margin(rollouts_file, rollouts_clean, bootstrap, seed, level):
    f_draws, f_margin, n_clusters, _ = ai.draws(rollouts_file, bootstrap, seed)
    c_draws, c_margin, _, _ = ai.draws(rollouts_clean, bootstrap, seed + 1)
    diff = 100.0 * (f_draws - c_draws)
    return {"margin": f_margin - c_margin, "file_over_chance": f_margin,
            "clean_over_chance": c_margin, "n_clusters": n_clusters,
            "ci95": ai.interval(diff, 0.95), "ci_level": ai.interval(diff, level),
            # the same file read against chance, because the control sits below it
            "chance_ci_level": ai.interval(100.0 * f_draws, level)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="build/bixbench_v10.jsonl")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration-reps", type=int, default=1000)
    ap.add_argument("--output", default="results/bixbench_v10_grid.json")
    ap.add_argument("--latex", action="store_true",
                    help="print tab:gridv10's rows, each part over chance, from the report")
    args = ap.parse_args()
    if args.latex:
        rep = json.loads(pathlib.Path(args.output).read_text())
        for r in rep["rows"]:
            cells = [f"${r[p]['file_over_chance']:+.1f}$ $[{r[p]['chance_ci_level'][0]:+.1f},"
                     f"{r[p]['chance_ci_level'][1]:+.1f}]$" for p in ("all", "numeric", "other")]
            print(f"{r['model']} & ${r['billions']:.1f}$ & " + " & ".join(cells) + "\\\\")
        return

    rows = [json.loads(l) for l in open(args.items)]
    numeric = [rank_of([r["ideal"], *r["distractors"]]) is not None for r in rows]
    dumps = {tag: ai.resolve(f"build/dumps/{tag}_bixv10_bixprompt_argmax.jsonl") for tag, _, _ in MODELS}
    present = [(t, n, b) for t, n, b in MODELS if dumps[t].exists()]
    missing = [n for t, n, _ in MODELS if not dumps[t].exists()]
    if missing:
        print("no dump yet for:", ", ".join(missing))

    levels = {}
    for tag, name, _ in present:
        levels[tag], n_eff = covering_level(dumps[tag], args.calibration_reps, args.bootstrap, 20260923)
        print(f"  {name:20s} covers 95% at {100 * levels[tag]:.2f}%  ({n_eff:.1f} effective capsules)")
    level = max(levels.values())

    report = {"items": args.items, "n_items": len(rows), "n_numeric": sum(numeric),
              "n_clusters": len({r["cluster"] for r in rows}), "level": level,
              "levels_by_model": levels, "rows": []}
    print(f"\nevery row read at {100 * level:.2f}%, the widest any model's dump needs\n")
    print(f"{'model':20s} {'B':>5s}  {'all ' + str(len(rows)):>28s}  {'numeric ' + str(sum(numeric)):>28s}  "
          f"{'other ' + str(len(rows) - sum(numeric)):>28s}")
    for tag, name, billions in present:
        f_all, c_all = ai.load(dumps[tag], "file"), ai.load(dumps[tag], "clean")
        unparsed = [r for r in f_all + c_all if r["answer"] not in "ABCD" or not r["answer"]]
        entry = {"model": name, "billions": billions, "dump": str(dumps[tag]),
                 "unparsed_rollouts": len(unparsed)}
        for part, keep in (("all", lambda i: True), ("numeric", lambda i: numeric[i]),
                           ("other", lambda i: not numeric[i])):
            entry[part] = margin([r for r in f_all if keep(r["item"])],
                                 [r for r in c_all if keep(r["item"])],
                                 args.bootstrap, args.seed, level)
        report["rows"].append(entry)

        def cell(m):
            lo, hi = m["ci_level"]
            flag = "*" if lo > 0 else " "
            return f"{m['margin']:+6.2f} [{lo:+6.2f},{hi:+6.2f}]{flag}"
        print(f"{name:20s} {billions:5.1f}  {cell(entry['all']):>28s}  {cell(entry['numeric']):>28s}  "
              f"{cell(entry['other']):>28s}")
    cleared = [r["model"] for r in report["rows"] if r["all"]["ci_level"][0] > 0]
    report["clear_on_all_items"] = cleared
    report["clear_over_chance"] = [r["model"] for r in report["rows"]
                                   if r["all"]["chance_ci_level"][0] > 0]
    for part in ("numeric", "other"):
        report[f"clear_on_{part}"] = [r["model"] for r in report["rows"] if r[part]["ci_level"][0] > 0]
    print(f"over chance instead: {len(report['clear_over_chance'])} of {len(report['rows'])} clear; "
          f"numeric half {len(report['clear_on_numeric'])}, the rest {len(report['clear_on_other'])}")
    print(f"\n{len(cleared)} of {len(report['rows'])} clear on all {len(rows)} items: {', '.join(cleared)}")
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
