#!/usr/bin/env python3
"""Flat margins are not a flat joint, and a file need not admit both.

\\S2's bound is stated one ordering at a time: if the key's rank under an
ordering is uniform, that ordering is worth nothing to a solver whose
preference is over its ranks. The audit measures exactly that, one coordinate at
a time, and the constructive repair makes each margin flat by fitting one weight
per rank of each coordinate.

Neither covers a solver that reads the *tuple*. Marginal uniformity does not
imply joint uniformity -- an item's frontier of reachable (length, roundness)
pairs is not a product set, so drawing from it with marginal weights leaves the
pair dependent -- and learned_probe.py finds the residue: on the repaired
ten-option file every margin is within 1.6 points of flat and a learner reading
all eight orderings still scores well above what it scores on a file that
cannot leak.

So this measures the joint directly, with the estimator the rest of the audit
uses: the share of items whose key lands on the most common rank tuple, chosen
on the training clusters and scored on a held-out one. ``mcq_audit.py
--joint-draw`` fits the joint instead of the margins, and the comparison is the
point: at four options flattening the joint costs the margins, which means the
file does not admit both.

    python3 joint_uniformity.py --jsonl data/bixbench.jsonl --label released \\
        --jsonl results/repaired/bixbench_v15_repaired_multi.jsonl --label repaired \\
        --n-options 4
"""
import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from mcq_audit import build_items, ordering_rank, read_rows

PAIR = ("written length", "significant digits")


def measure(items, k, coordinates=PAIR):
    """Margins, the joint plug-in maximum, and the cross-validated joint pick."""
    numeric = [item for item in items if item["rank"] is not None]
    rows, margins = [], {name: Counter() for name in coordinates}
    for item in numeric:
        ranks = tuple(ordering_rank(item["options"], name) for name in coordinates)
        if any(rank is None for rank in ranks):
            continue
        rows.append((ranks, item["cluster"]))
        for name, rank in zip(coordinates, ranks):
            margins[name][rank] += 1
    if not rows:
        return None
    n = len(rows)
    counts = Counter(tuple_ for tuple_, _ in rows)
    clusters = sorted({cluster for _, cluster in rows})
    hits = 0
    for held in clusters:
        train = Counter(t for t, c in rows if c != held)
        if not train:
            continue
        best = train.most_common(1)[0][0]
        hits += sum(1 for t, c in rows if c == held and t == best)
    return {
        "n_items": n, "n_clusters": len(clusters),
        "widest_margin": {name: max(counter.values()) / n
                          for name, counter in margins.items()},
        "flat_margin": 1 / k,
        "joint_plug_in": max(counts.values()) / n,
        "joint_cv": hits / n,
        "flat_joint": 1 / k ** len(coordinates),
        "distinct_tuples": len(counts),
        "picks": [(cluster, tuple_ == counts.most_common(1)[0][0])
                  for tuple_, cluster in rows],
    }


def interval(picks, reps, seed):
    by_cluster = {}
    for cluster, hit in picks:
        by_cluster.setdefault(cluster, []).append(hit)
    keys = list(by_cluster)
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        pooled = [h for k in (rng.choice(keys) for _ in keys) for h in by_cluster[k]]
        draws.append(sum(pooled) / len(pooled))
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def clean_control(k, n_items, n_clusters, replicates, seed):
    from audit_calibration import synthetic_rows

    out = []
    for replicate in range(replicates):
        rng = random.Random(f"{seed}|clean|{k}|{replicate}")
        rows = synthetic_rows(n_items, k, n_clusters, rng)
        items, found = build_items(rows, "ideal", "distractors", "capsule_uuid",
                                   "question", k)
        if found != k:
            continue
        got = measure(items, k)
        if got:
            out.append(got["joint_cv"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--n-options", type=int, default=4)
    parser.add_argument("--cluster-field", default="capsule_uuid")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--clean-items", type=int, default=None)
    parser.add_argument("--clean-clusters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--output", type=Path,
                        default=Path("results/joint_uniformity.json"))
    args = parser.parse_args()

    k = args.n_options
    labels = args.label or [path.stem for path in args.jsonl]
    report = {"n_options": k, "coordinates": list(PAIR), "flat_margin": 1 / k,
              "flat_joint": 1 / k ** len(PAIR), "files": {}}
    shape = None
    for label, path in zip(labels, args.jsonl):
        items, found = build_items(read_rows(path), "ideal", "distractors",
                                   args.cluster_field, "question", k)
        if found != k:
            raise SystemExit(f"{path}: {found} options, expected {k}")
        got = measure(items, k)
        if got is None:
            raise SystemExit(f"{path}: no numeric items")
        low, high = interval(got.pop("picks"), args.reps, args.seed)
        got["joint_cv_ci95"] = [low, high]
        report["files"][label] = got
        shape = shape or (got["n_items"], got["n_clusters"])
        margins = ", ".join(f"{name} {share:.1%}"
                            for name, share in got["widest_margin"].items())
        print(f"{label}: {got['n_items']} items")
        print(f"    widest margin: {margins} (flat {1 / k:.1%})")
        print(f"    joint: plug-in {got['joint_plug_in']:.1%}, cross-validated "
              f"{got['joint_cv']:.1%} [{low:.1%}, {high:.1%}] "
              f"(flat {report['flat_joint']:.1%}, {got['distinct_tuples']} tuples seen)")

    clean = clean_control(k, args.clean_items or shape[0],
                          args.clean_clusters or shape[1], args.replicates, args.seed)
    report["clean"] = {"scores": clean, "mean": statistics.fmean(clean),
                       "max": max(clean), "replicates": len(clean)}
    print(f"clean control ({len(clean)} files): joint cross-validated mean "
          f"{statistics.fmean(clean):.1%}, worst case {max(clean):.1%}")
    for label, block in report["files"].items():
        block["over_clean_worst_case"] = block["joint_cv"] - max(clean)
        print(f"    {label}: {100 * block['over_clean_worst_case']:+.1f} points over it")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
