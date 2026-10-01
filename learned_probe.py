#!/usr/bin/env python3
"""Can a *learned* rank-only solver find a coordinate we did not name?

Every channel in this repository is a hand-designed statistic: sorted value,
lexical isolation, written length, significant digits, and the four orderings
held_out_orderings.py holds out. The obvious objection to a null built on
hand-designed statistics is that the wrong ones were designed. So this fits a
solver instead of writing one.

The model is the weakest thing that could work, on purpose. Each option gets a
feature vector read off the option set alone -- no question, no data -- and a
linear score; the solver picks the arg-max. That makes it a *rank-only solver*
in exactly the sense of \\S2, so its cross-validated accuracy is an estimate of
what some solver can collect, and it subsumes every rule in rule_solvers.py
because the eight measured ranks are among its features.

Three things make the number readable rather than impressive:

  * Leave-one-cluster-out cross-validation, as everywhere else here. A
    learner selected on the items it is scored on will beat chance on noise.
  * A clean-file control. Synthetic files built the way audit_calibration.py
    builds them cannot leak, so whatever the learner scores there is its own
    false-positive rate, and the released and repaired files are read against
    it rather than against 1/k.
  * A cluster bootstrap interval on the difference from the clean file.

    python3 learned_probe.py --jsonl data/bixbench.jsonl --n-options 4

Requires numpy (matplotlib and scipy already pull it in).
"""
import argparse
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np

import mcq_audit
from mcq_audit import (build_items, feature_rank, isolation_scores, length_scores,
                       parse_number, read_rows, significant_digits)

# The nine orderings enter as *indicators*, one per rank, not as a number.
# That is the whole point of \S2: a solver's preference ``b`` is a distribution
# over ranks, so "the second-smallest" is a preference a linear score has to be
# able to express. Encoding rank as a single continuous feature cannot -- it can
# only prefer small ranks or large ones -- and a learner built that way scores
# 31% on a file where the hand-designed rule scores 51%.
ORDERINGS = ("value", "length", "roundness", "isolation",
             "leading digit", "decimals", "digit sum", "text",
             # Written length again with the other tie-break. Ties are dense on
             # these files, so this is a different ordering from "length" on
             # most items rather than a duplicate of it, and it is the one the
             # hand-written rule family actually reads.
             "length, later ties")
PLAIN = ("is integer", "has percent", "has exponent", "has separator",
         "distinct chars", "relative magnitude", "distance from the rest",
         "is the median", "trailing zeros", "relative length")


def feature_names(k):
    """One indicator per (ordering, rank), then the per-option features."""
    return tuple(f"{name} rank {rank}" for name in ORDERINGS for rank in range(k)) + PLAIN


AUDITED = ("value", "length", "roundness", "isolation")


def audited_columns(k):
    """The rank indicators for the four orderings the audit measures and the
    repair targets. A learner restricted to these is the strongest solver the
    repair's guarantee actually speaks about; what it scores on a repaired file
    is the guarantee's own empirical check, and what the wider learners score
    beyond it is the cost of the coordinates nobody audited."""
    keep = []
    for index, name in enumerate(ORDERINGS):
        if name in AUDITED:
            keep.extend(range(index * k, (index + 1) * k))
    return np.asarray(keep)


def rank_only_columns(k):
    r"""The columns that are functions of the key's rank under some ordering.

    \S2's bound is about a solver whose preference is a distribution over
    *ranks*, and these are exactly the features such a solver may read. The
    remaining ones -- how large the key is relative to the largest option, how
    long relative to the longest, whether it is an integer -- are not functions
    of any rank, so a learner given them is outside the class the bound covers.
    Reporting the two separately is the difference between "what the repair
    guarantees" and "what is left identifiable".
    """
    return np.arange(len(ORDERINGS) * k)


def _digits(text):
    body = str(text).strip().rstrip("%").replace(",", "")
    body = body.split("e")[0].split("E")[0]
    return "".join(c for c in body if c.isdigit())


def _ranks(options, scores, ties="earlier"):
    return [feature_rank(options, scores, i, ties=ties) for i in range(len(options))]


def item_features(options):
    """One row per option: everything a rank-only solver could read off the set."""
    k = len(options)
    values = [parse_number(o) for o in options]
    if any(v is None for v in values):
        return None
    texts = [str(o) for o in options]
    magnitude = max(abs(v) for v in values) or 1.0
    mean = statistics.fmean(values)
    spread = statistics.pstdev(values) or 1.0
    median = statistics.median(values)
    longest = max(len(t) for t in texts) or 1

    columns = {
        "value": _ranks(options, values),
        "length": _ranks(options, length_scores(options)),
        "roundness": _ranks(options, [significant_digits(o) for o in options]),
        "isolation": _ranks(options, isolation_scores(options)),
        "leading digit": _ranks(options, [
            float(_digits(o).lstrip("0")[0]) if _digits(o).lstrip("0") else 0.0
            for o in options]),
        "decimals": _ranks(options, [
            float(len(t.split("e")[0].split("E")[0].split(".")[1]))
            if "." in t.split("e")[0].split("E")[0] else 0.0 for t in texts]),
        "digit sum": _ranks(options, [
            float(sum(int(c) for c in _digits(o))) for o in options]),
        "text": _ranks(options, [0.0] * k),
        "length, later ties": _ranks(options, length_scores(options), ties="later"),
    }
    rows = []
    for i, (text, value) in enumerate(zip(texts, values)):
        digits = _digits(text)
        indicators = []
        for name in ORDERINGS:
            one_hot = [0.0] * k
            one_hot[columns[name][i]] = 1.0
            indicators.extend(one_hot)
        rows.append(indicators + [
            float(value == int(value)) if abs(value) < 1e15 else 0.0,
            float("%" in text),
            float("e" in text.lower()),
            float("," in text),
            len(set(text)) / max(1, len(text)),
            abs(value) / magnitude,
            (value - mean) / spread,
            float(value == median),
            float(len(digits) - len(digits.rstrip("0"))),
            len(text) / longest,
        ])
    return np.asarray(rows, dtype=float)


def build_matrix(items):
    """(X, clusters): X is (items, options, features); the key is always slot 0."""
    stack, clusters = [], []
    for item in items:
        features = item_features(item["options"])
        if features is None:
            continue
        stack.append(features)
        clusters.append(item["cluster"])
    if not stack:
        return np.zeros((0, 0, 0)), []
    return np.stack(stack), clusters


def fit(x, steps=400, learning_rate=0.5, l2=1e-3):
    """Softmax over the options of each item, by gradient ascent on the likelihood.

    Vectorised over items: the per-item Python loop this replaces was fine at
    105 items and would have taken hours at 1,263 x 67 folds, which is the kind
    of thing that quietly turns into "we only ran it on the small file".
    """
    if x.shape[0] == 0:
        return None
    weights = np.zeros(x.shape[2])
    for _ in range(steps):
        scores = x @ weights                       # (items, options)
        scores -= scores.max(axis=1, keepdims=True)
        probability = np.exp(scores)
        probability /= probability.sum(axis=1, keepdims=True)
        expected = np.einsum("nk,nkd->nd", probability, x)
        gradient = (x[:, 0, :] - expected).mean(axis=0) - l2 * weights
        weights += learning_rate * gradient
    return weights


def cross_validate(x, clusters, steps, learning_rate, l2, seed=20260920):
    """Leave-one-cluster-out accuracy, the picks for the bootstrap, and the ties.

    Ties are broken uniformly at random rather than by ``np.argmax``, which
    resolves them to the first index -- and the key is slot 0 of every row, so
    that would score every tie correct. It matters as soon as a family is small
    enough to tie: four roundness features read off one option at a time tie on
    two thirds of MMLU's items and score 63% against 25% chance that way, and
    on a synthetic file that cannot leak they score 34%. The three families
    reported here tie on none of the items, which is why the count travels with
    the result instead of being asserted in a comment.
    """
    clusters = np.asarray(clusters, dtype=object)
    rng = np.random.default_rng(seed)
    picks, tied = [], 0
    for held in sorted(set(clusters.tolist())):
        test = clusters == held
        weights = fit(x[~test], steps, learning_rate, l2)
        if weights is None:
            continue
        scores = x[test] @ weights
        winners = np.abs(scores - scores.max(axis=1, keepdims=True)) < 1e-12
        tied += int((winners.sum(axis=1) > 1).sum())
        chosen = np.argmax(rng.random(scores.shape) * winners, axis=1)
        picks.extend((held, bool(pick == 0)) for pick in chosen)
    return picks, tied


def accuracy(picks):
    return sum(hit for _, hit in picks) / len(picks) if picks else float("nan")


def bootstrap(picks, reps, seed):
    """Cluster bootstrap of the accuracy. Sorted, ascending."""
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
    return draws


def clean_control(k, n_items, n_clusters, replicates, steps, learning_rate, l2,
                  seed, columns=None):
    """What the learner scores on files that cannot leak."""
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
        scores.append(accuracy(picks))
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", type=Path, action="append", required=True,
                        help="a question file; repeat to compare released and repaired")
    parser.add_argument("--label", action="append", default=None,
                        help="a name per --jsonl, in the same order")
    parser.add_argument("--n-options", type=int, default=4)
    parser.add_argument("--cluster-field", default="capsule_uuid")
    parser.add_argument("--key-field", default="ideal")
    parser.add_argument("--distractor-field", default="distractors")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--replicates", type=int, default=10,
                        help="clean synthetic files for the control")
    parser.add_argument("--clean-items", type=int, default=None)
    parser.add_argument("--clean-clusters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--output", type=Path, default=Path("results/learned_probe.json"))
    args = parser.parse_args()

    k = args.n_options
    labels = args.label or [path.stem for path in args.jsonl]
    if len(labels) != len(args.jsonl):
        raise SystemExit("--label must be given once per --jsonl")

    names = feature_names(k)
    report = {"n_options": k, "chance": 1 / k, "features": list(names),
              "steps": args.steps, "l2": args.l2, "files": {}}
    first_rows = None
    for label, path in zip(labels, args.jsonl):
        items, found = build_items(read_rows(path), args.key_field,
                                   args.distractor_field, args.cluster_field,
                                   args.question_field, k)
        if found != k:
            raise SystemExit(f"{path}: found {found} options, expected {k}")
        x, clusters = build_matrix(items)
        if first_rows is None:
            first_rows = (x, clusters)
        block = {"path": str(path), "n_items": int(x.shape[0]),
                 "n_clusters": len(set(clusters))}
        print(f"{label}: {x.shape[0]} items, {len(set(clusters))} clusters, "
              f"chance {1 / k:.1%}")
        for family, columns in (("audited ranks", audited_columns(k)),
                                ("rank-only", rank_only_columns(k)),
                                ("every feature", np.arange(x.shape[2]))):
            subset = x[:, :, columns]
            picks, tied = cross_validate(subset, clusters, args.steps,
                                         args.learning_rate, args.l2)
            draws = bootstrap(picks, args.reps, args.seed)
            weights = fit(subset, args.steps, args.learning_rate, args.l2)
            got = accuracy(picks)
            order = np.argsort(-np.abs(weights))[:4]
            block[family] = {
                "cv_accuracy": got,
                "cv_ci95": [draws[int(0.025 * len(draws))],
                            draws[int(0.975 * len(draws))]],
                # How many held-out items the fitted score left tied at the top.
                # Zero on every file reported here; a family where it is not
                # zero is being read partly off the tie-break (cross_validate).
                "argmax_ties": tied,
                "weights": {names[columns[i]]: float(weights[i])
                            for i in range(len(columns))},
            }
            print(f"    {family:<14} {got:6.1%} "
                  f"[{block[family]['cv_ci95'][0]:.1%}, "
                  f"{block[family]['cv_ci95'][1]:.1%}]   heaviest: "
                  + ", ".join(f"{names[columns[i]]} {weights[i]:+.2f}" for i in order))
        report["files"][label] = block

    n_items = args.clean_items or int(first_rows[0].shape[0])
    n_clusters = args.clean_clusters or len(set(first_rows[1]))
    report["clean"] = {}
    for family, columns in (("audited ranks", audited_columns(k)),
                            ("rank-only", rank_only_columns(k)),
                            ("every feature", None)):
        clean = clean_control(k, n_items, n_clusters, args.replicates, args.steps,
                              args.learning_rate, args.l2, args.seed, columns)
        report["clean"][family] = {"scores": clean, "mean": statistics.fmean(clean),
                                   "max": max(clean), "replicates": len(clean),
                                   "n_items": n_items, "n_clusters": n_clusters}
        print(f"clean control, {family} ({len(clean)} files of {n_items} items): "
              f"mean {statistics.fmean(clean):.1%}, worst case {max(clean):.1%}")
        for label, block in report["files"].items():
            margin = block[family]["cv_accuracy"] - max(clean)
            block[family]["over_clean_worst_case"] = margin
            print(f"    {label}: {100 * margin:+.1f} points over that worst case")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
