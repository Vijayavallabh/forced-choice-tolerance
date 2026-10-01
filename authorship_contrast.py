#!/usr/bin/env python3
"""The paired contrast between the two arms ``authorship_causal.py`` builds.

Both files hold the same 518 items in the same subjects, with the same key and
the same three human-written distractors. Six more options were added to each:
written by an open-weight model in one arm, lifted from other items' human-written
options in the other. So the two arms differ in one thing, and the right analysis
is paired rather than each arm against a clean file.

Both comparisons are reported, because they answer different questions and give
different answers. Against its own clean control, which is what the audit in
section 3 would do, neither arm clears the bar. Against each other, which is what the
design was built for, the roundness family separates them.

    python3 authorship_contrast.py --output results/authorship_contrast.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

import key_identity as ki
import learned_probe as lp


def rows(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append({"options": [str(row["ideal"]).strip()]
                               + [str(d).strip() for d in row["distractors"]],
                    "cluster": row.get("cluster", "")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generated", default="build/mmlu_aug_generated.jsonl")
    ap.add_argument("--borrowed", default="build/mmlu_aug_borrowed.jsonl")
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--learning-rate", type=float, default=0.5)
    ap.add_argument("--output", default="results/authorship_contrast.json")
    args = ap.parse_args()

    arms = {"generated": rows(args.generated), "borrowed": rows(args.borrowed)}
    if len(arms["generated"]) != len(arms["borrowed"]):
        raise SystemExit("the two arms must hold the same items")

    families = {"written form": list(ki.FEATURES), "roundness": list(ki.ROUNDNESS)}
    report = {"n_items": len(arms["generated"]), "reps": args.reps,
              "seed": args.seed, "families": {},
              "design": ("same items, same key, same three human distractors; the "
                         "six added options are model-written in one arm and "
                         "human-written in the other")}

    for family, feature_names in families.items():
        columns = [ki.FEATURES.index(f) for f in feature_names]
        picks = {}
        for arm, data in arms.items():
            matrix, clusters = ki.build_matrix(data)
            chosen, _ = lp.cross_validate(matrix[:, :, columns], clusters,
                                          args.steps, args.learning_rate,
                                          args.l2)
            picks[arm] = chosen
        ids = [c for c, _ in picks["generated"]]
        if [c for c, _ in picks["borrowed"]] != ids:
            raise SystemExit("the two arms did not line up item for item")
        hits = {arm: [h for _, h in picks[arm]] for arm in arms}

        by_cluster = {}
        for cluster, a, b in zip(ids, hits["generated"], hits["borrowed"]):
            by_cluster.setdefault(cluster, []).append((a, b))
        keys = list(by_cluster)
        rng = random.Random(args.seed)
        draws = []
        for _ in range(args.reps):
            flat = [pair for _ in keys for pair in by_cluster[rng.choice(keys)]]
            draws.append(np.mean([a for a, _ in flat]) - np.mean([b for _, b in flat]))
        draws.sort()
        report["families"][family] = {
            "generated": float(np.mean(hits["generated"])),
            "borrowed": float(np.mean(hits["borrowed"])),
            "difference": float(np.mean(hits["generated"]) - np.mean(hits["borrowed"])),
            "ci95": [float(draws[int(0.025 * len(draws))]),
                     float(draws[int(0.975 * len(draws))])],
            "p_one_sided": sum(1 for d in draws if d <= 0) / len(draws),
            "p_two_sided": min(1.0, 2 * min(
                sum(1 for d in draws if d <= 0) / len(draws),
                sum(1 for d in draws if d >= 0) / len(draws))),
            "n_clusters": len(keys),
        }
        block = report["families"][family]
        print(f"{family:14s} generated {100 * block['generated']:5.1f}%  "
              f"borrowed {100 * block['borrowed']:5.1f}%  "
              f"difference {100 * block['difference']:+5.1f} "
              f"[{100 * block['ci95'][0]:+.1f},{100 * block['ci95'][1]:+.1f}]  "
              f"one-sided p={block['p_one_sided']:.3f} "
              f"two-sided p={block['p_two_sided']:.3f}")

    # Two families are tested, so a family-wise statement needs a correction.
    # Bonferroni over the two, on the two-sided p, is the conservative one.
    n_families = len(report["families"])
    for family, block in report["families"].items():
        block["p_two_sided_bonferroni"] = min(1.0, n_families * block["p_two_sided"])
        print(f"{family:14s} family-wise two-sided p "
              f"(Bonferroni over {n_families}) = "
              f"{block['p_two_sided_bonferroni']:.3f}")
    report["n_families"] = n_families
    report["multiplicity"] = (
        f"Bonferroni over the {n_families} feature families tested; the reported "
        "family-wise value is the two-sided bootstrap p times the family count")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
