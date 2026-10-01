#!/usr/bin/env python3
"""What a repair takes away, separated from what rewriting the options adds.

A maintainer who repairs a benchmark re-runs its no-data baseline and reads the
new number. That number is not comparable to the old one, because regenerating
distractors changes how dismissible they are as well as what rank the key sits
at. The five-arm design already separates the two with the question shown; this
does it with the question withheld, which is the condition the contamination
reading is actually applied to.

Three arms per solver, all stem-withheld: the released options, a **placebo**
that redraws the distractor values through the identical generator while
holding the key's rank, and the **repair**. Then

    placebo - released   is what rewriting the options is worth,
    repair  - placebo    is the rank effect: what the repair removes.

The second is the quantity of interest and the first is why it cannot be read
off the repaired arm alone.

One more thing falls out. If the rank effect is the channel closing, it should
be largest for the solvers that were collecting most, and the released arm is a
measure of that. So the trend across solvers is reported too, as Spearman's rho
with a bootstrap interval over models -- a prediction the design could have
falsified and did not.

    python3 placebo_contrast.py --probe-dir results/probe_bix_placebo --n-options 4
"""
import argparse
import gzip
import json
import random
import statistics
from pathlib import Path


def arm_rates(path):
    """Stem-withheld accuracy per arm over the numeric items, and the model."""
    hits, total, model = {}, {}, None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            model = model or row["model"]
            if not row["is_numeric"]:
                continue
            hits[row["arm"]] = hits.get(row["arm"], 0) + row["correct"]
            total[row["arm"]] = total.get(row["arm"], 0) + 1
    return model, {arm: hits[arm] / total[arm] for arm in hits}


def spearman(xs, ys):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2
            for index in order[i:j + 1]:
                out[index] = shared
            i = j + 1
        return out

    a, b = ranks(xs), ranks(ys)
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den = (sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe-dir", action="append", required=True,
                        help="one directory, or several joined by commas when a "
                             "panel was run in more than one batch")
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--n-options", type=int, action="append", default=None)
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--output", type=Path,
                        default=Path("results/placebo_contrast.json"))
    args = parser.parse_args()

    groups = [[Path(part) for part in spec.split(",")] for spec in args.probe_dir]
    labels = args.label or [group[0].name.replace("probe_", "") for group in groups]
    counts = args.n_options or [4] * len(args.probe_dir)
    if not (len(labels) == len(counts) == len(groups)):
        raise SystemExit("--label and --n-options must be given once per --probe-dir")

    report = {"reps": args.reps, "panels": {}}
    for label, group, k in zip(labels, groups, counts):
        rows, seen = [], set()
        paths = sorted(path for directory in group
                       for path in directory.glob("*.jsonl.gz"))
        for path in paths:
            model, rates = arm_rates(path)
            if model in seen:
                print(f"  skipping a second copy of {model}")
                continue
            seen.add(model)
            needed = {"original_stemless", "placebo_stemless", "repaired_stemless"}
            if not needed <= set(rates):
                print(f"  skipping {model}: has {sorted(rates)}")
                continue
            released = rates["original_stemless"]
            rows.append({
                "model": model, "chance": 1 / k, "released": released,
                "placebo_lift": rates["placebo_stemless"] - released,
                "repaired_lift": rates["repaired_stemless"] - released,
                "rank_effect": rates["repaired_stemless"] - rates["placebo_stemless"],
            })
        if not rows:
            continue
        rows.sort(key=lambda r: r["rank_effect"])
        rho = spearman([r["released"] for r in rows],
                       [r["rank_effect"] for r in rows])
        rng = random.Random(args.seed)
        draws = []
        for _ in range(args.reps):
            sample = [rows[rng.randrange(len(rows))] for _ in rows]
            if len({r["model"] for r in sample}) < 3:
                continue
            draws.append(spearman([r["released"] for r in sample],
                                  [r["rank_effect"] for r in sample]))
        draws.sort()
        report["panels"][label] = {
            "n_options": k, "n_models": len(rows), "models": rows,
            "placebo_lift_median": statistics.median(r["placebo_lift"] for r in rows),
            "rank_effect_median": statistics.median(r["rank_effect"] for r in rows),
            "rank_effect_positive": sum(1 for r in rows if r["rank_effect"] > 0),
            "released_vs_rank_effect_rho": rho,
            "rho_ci95": [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]],
        }
        print(f"=== {label}: {len(rows)} solvers, {k} options, chance {1 / k:.1%}")
        print(f"  {'model':<40}{'released':>9}{'placebo':>9}{'repaired':>10}"
              f"{'rank effect':>13}")
        for row in rows:
            print(f"  {row['model']:<40}{row['released']:8.1%}"
                  f"{100 * row['placebo_lift']:+9.1f}{100 * row['repaired_lift']:+10.1f}"
                  f"{100 * row['rank_effect']:+13.1f}")
        block = report["panels"][label]
        print(f"  placebo lift median {100 * block['placebo_lift_median']:+.1f}; "
              f"rank effect median {100 * block['rank_effect_median']:+.1f}, "
              f"positive for {block['rank_effect_positive']} of {len(rows)}")
        print(f"  released accuracy vs rank effect: rho = {rho:+.2f} "
              f"[{block['rho_ci95'][0]:+.2f}, {block['rho_ci95'][1]:+.2f}]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
