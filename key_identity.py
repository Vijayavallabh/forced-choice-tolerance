"""What survives every rank repair is not geometry: it is the key's own handwriting.

``learned_probe.py`` ends on a finding the rank guarantee does not cover. Given
features that are not ranks, a learner scores *higher* on the repaired file than
on the released one -- 41.2% against 28.4% on MMLU, 13.4 points over its clean
control -- and its heaviest weight is ``relative magnitude``. The obvious reading
is geometric: the repair makes distractors by perturbing the key, so the key
becomes the centre of the option set, and a solver that reads how large an option
is relative to the largest finds it.

``exchangeable_repair.py`` tests that reading by removing the geometry. It draws
the offsets so the key is a uniform draw from its own option set, which makes the
key's relative magnitude distributed exactly like a distractor's. The leak did
not go with it: 11.8 points over the clean control became 10.9, and the heaviest
weight moved from ``relative magnitude`` to ``has separator`` and
``trailing zeros``. So the geometric reading is wrong, and the weights say what
the right one is.

This script tests it directly. It fits the same solver, over features that read
**one option at a time**: whether it is an integer, how many significant digits
it has, whether it carries a thousands separator, how many trailing zeros. No
feature here refers to any other option, so nothing this solver reads is a
property of the option *set* -- not a rank, not a spread, not a distance. If it
still finds the key on a repaired file, the residue is not option geometry at
all. It is that a hand-chosen answer is written differently from a generated
distractor, and no rewriting of the distractors can hide that, because the thing
that gives it away is the answer.

Two nested families, each against its own clean-file control:

    written form    every per-option feature below
    roundness       the four that say how round the number is: integer,
                    significant digits, trailing zeros, decimal places

The second exists because it is the sharpest form of the claim. "The key is the
round one" is a statement about a single option read alone, and a maintainer can
act on it: match the distractors' roundness to the key's, or accept that the
answer is legible.

    python3 key_identity.py --jsonl build/mmlu.jsonl --label released \
        --jsonl build/mmlu_repaired.jsonl --label repaired \
        --n-options 4 --cluster-field cluster --output results/key_identity_mmlu.json
"""
import argparse
import json
import random
import statistics
from pathlib import Path

import numpy as np

from learned_probe import accuracy, bootstrap, cross_validate
from mcq_audit import build_items, parse_number, read_rows, significant_digits

# Every column reads one option's text. The order is the reported order.
FEATURES = ("is integer", "has percent", "has exponent", "has separator",
            "has minus", "has point", "distinct chars", "trailing zeros",
            "significant digits", "decimal places", "digit sum", "digit count")
# The sharpest subset: how round the number reads, and nothing else. Four
# integer-valued features tie at the top on two thirds of items, so this family
# is only meaningful once ties are broken at random -- resolved to slot 0 it
# reads 63% on MMLU and 34% on a file with nothing in it (``cross_validate``).
ROUNDNESS = ("is integer", "significant digits", "trailing zeros", "decimal places")


def _digits(text):
    body = str(text).strip().rstrip("%").replace(",", "").replace("−", "-")
    body = body.split("e")[0].split("E")[0]
    return "".join(c for c in body if c.isdigit())


def option_features(text, value):
    """One option's written form, read without reference to any other option."""
    text = str(text).strip()
    body = text.rstrip("%").replace(",", "").replace("−", "-")
    body = body.split("e")[0].split("E")[0]
    digits = _digits(text)
    decimals = len(body.split(".")[1]) if "." in body else 0
    return [
        float(value == int(value)) if abs(value) < 1e15 else 0.0,
        float("%" in text),
        float("e" in text.lower()),
        float("," in text),
        float(text.lstrip().startswith(("-", "−"))),
        float("." in body),
        len(set(text)) / max(1, len(text)),
        float(len(digits) - len(digits.rstrip("0"))),
        significant_digits(text),
        float(decimals),
        float(sum(int(c) for c in digits)),
        float(len(digits)),
    ]


def build_matrix(items):
    """(X, clusters) with the key at slot 0, as ``learned_probe`` does it."""
    stack, clusters = [], []
    for item in items:
        options = item["options"]
        values = [parse_number(o) for o in options]
        if any(v is None for v in values):
            continue
        stack.append([option_features(o, v) for o, v in zip(options, values)])
        clusters.append(item["cluster"])
    if not stack:
        return np.zeros((0, 0, 0)), []
    return np.asarray(stack, dtype=float), clusters


def clean_control(k, n_items, n_clusters, replicates, steps, learning_rate, l2,
                  seed, columns=None):
    """The same solver on files with nothing in them, shaped like the measured one.

    Read against zero this probe would fire on any file: the features are
    correlated with each other and the fold count is small, so a little
    overfitting shows as accuracy. The control is what makes a margin mean
    something, and it is regenerated per feature family because a narrower
    family overfits less.
    """
    from audit_calibration import synthetic_rows

    scores = []
    for replicate in range(replicates):
        rng = random.Random(f"{seed}|clean|{k}|{replicate}")
        rows = synthetic_rows(n_items, k, n_clusters, rng)
        items, found = build_items(rows, "ideal", "distractors", "capsule_uuid",
                                   "question", k)
        if found != k:
            continue
        x, clusters = build_matrix(items)
        if columns is not None:
            x = x[:, :, columns]
        picks, _ = cross_validate(x, clusters, steps, learning_rate, l2)
        if picks:
            scores.append(accuracy(picks))
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=Path, action="append", required=True)
    ap.add_argument("--label", action="append", default=None)
    ap.add_argument("--n-options", type=int, default=4)
    ap.add_argument("--cluster-field", default="cluster")
    ap.add_argument("--key-field", default="ideal")
    ap.add_argument("--distractor-field", default="distractors")
    ap.add_argument("--question-field", default="question")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--replicates", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--output", type=Path,
                    default=Path("results/key_identity.json"))
    args = ap.parse_args()

    k = args.n_options
    labels = args.label or [p.stem for p in args.jsonl]
    if len(labels) != len(args.jsonl):
        raise SystemExit("--label must be given once per --jsonl")
    families = {"written form": np.arange(len(FEATURES)),
                "roundness": np.asarray([FEATURES.index(n) for n in ROUNDNESS])}

    report = {"n_options": k, "chance": 1 / k, "features": list(FEATURES),
              "roundness_features": list(ROUNDNESS), "steps": args.steps,
              "l2": args.l2, "files": {}, "clean": {}}
    shape = None
    for label, path in zip(labels, args.jsonl):
        items, found = build_items(read_rows(path), args.key_field,
                                   args.distractor_field, args.cluster_field,
                                   args.question_field, k)
        if found != k:
            raise SystemExit(f"{path}: found {found} options, expected {k}")
        x, clusters = build_matrix(items)
        shape = shape or (int(x.shape[0]), len(set(clusters)))
        block = {"path": str(path), "n_items": int(x.shape[0]),
                 "n_clusters": len(set(clusters))}
        print(f"{label}: {x.shape[0]} items, {len(set(clusters))} clusters, "
              f"chance {1 / k:.1%}")
        for family, columns in families.items():
            subset = x[:, :, columns]
            picks, tied = cross_validate(subset, clusters, args.steps,
                                         args.learning_rate, args.l2)
            draws = bootstrap(picks, args.reps, args.seed)
            from learned_probe import fit
            weights = fit(subset, args.steps, args.learning_rate, args.l2)
            named = {FEATURES[c]: float(w) for c, w in zip(columns, weights)}
            heaviest = sorted(named.items(), key=lambda kv: -abs(kv[1]))[:4]
            block[family] = {
                "cv_accuracy": accuracy(picks),
                "cv_ci95": [draws[int(0.025 * len(draws))],
                            draws[int(0.975 * len(draws))]],
                "argmax_ties": tied, "n_scored": len(picks),
                "weights": named}
            print(f"    {family:<14} {accuracy(picks):6.1%}  "
                  f"(tied {tied}/{len(picks)})  heaviest: "
                  + ", ".join(f"{n} {w:+.2f}" for n, w in heaviest))
        report["files"][label] = block

    n_items, n_clusters = shape
    for family, columns in families.items():
        scores = clean_control(k, n_items, n_clusters, args.replicates,
                               args.steps, args.learning_rate, args.l2,
                               args.seed, columns)
        worst = max(scores) if scores else float("nan")
        report["clean"][family] = {"scores": scores,
                                   "mean": statistics.fmean(scores) if scores else None,
                                   "max": worst, "replicates": len(scores),
                                   "n_items": n_items, "n_clusters": n_clusters}
        print(f"\nclean control, {family} ({len(scores)} files of {n_items} items): "
              f"mean {statistics.fmean(scores):.1%}, worst case {worst:.1%}")
        for label in labels:
            got = report["files"][label][family]["cv_accuracy"]
            print(f"    {label}: {100 * (got - worst):+.1f} points over that worst case")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
