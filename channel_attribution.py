#!/usr/bin/env python3
"""Attribute a no-data margin to each coordinate of the option-geometry channel.

\\S2's bound never used the options being numbers: for *any* statistic that
orders them, the key's rank under that statistic has a distribution ``p^f``, a
solver's preference over those ranks is a distribution ``b^f``, and the credit
such a solver collects is ``<p^f, b^f> - 1/k``. So a margin can be attributed
per coordinate, and a repair can be checked per coordinate.

That check is the reason this script exists. Uniformising the *numeric* rank
regenerates the distractors from the key, which makes them resemble one another
and leaves the human-written key as the option least like the others. At four
options nothing much happens; at ten the numeric repair closes one coordinate
and opens another, and the second is the one solvers actually use.

Both ``p^f`` and ``b^f`` are measured, not fitted: ``p^f`` from the presented
options, ``b^f`` from what the solver picked with the question withheld. The
options themselves are not in the outcome files -- those hold no question or
option text, so they can be released -- but the conditions are a deterministic
function of the question file, the seed and the redraw count, so they are
rebuilt here and joined on (question, arm, redraw).
"""
import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from mcq_audit import ORDERINGS
from no_data_probe import build_conditions, letters_for, load_items, open_outcomes, outcome_files
from option_artifacts import parse_number

SEED = 20260920


# --------------------------------------------------------------------------
# What attribution measures. A coordinate is measured here first; one seen
# leaking is then promoted into ``mcq_audit.ORDERINGS``, which is what the
# repair uniformises. "Significant digits" was diagnosed that way and promoted.
# Measuring more than you repair is fine; repairing more than you measure is not.
DIAGNOSTIC_ORDERINGS = {}
CHANNELS = {**ORDERINGS, **DIAGNOSTIC_ORDERINGS}


def add_held_out_channels():
    """Measure the orderings held_out_orderings.py holds out, as diagnostics.

    They are not repaired and not audited; this reports what a solver collects
    on them anyway. It is the question the ten-option probe raises: the audit
    says every coordinate it measures is closed, and two solvers still gain
    three to four points with the question withheld, so either the gain is on a
    coordinate nobody measured or it is not geometry at all.
    """
    from held_out_orderings import HELD_OUT

    DIAGNOSTIC_ORDERINGS.update(HELD_OUT)
    CHANNELS.update(HELD_OUT)
    return sorted(HELD_OUT)


def ranks_from_scores(options, scores):
    """Rank of each presented slot under one statistic; ties break on the text."""
    order = sorted(range(len(scores)), key=lambda i: (scores[i], str(options[i])))
    return [order.index(i) for i in range(len(options))]


def channel_ranks(options):
    """For each channel, the rank of every presented slot, or None if undefined."""
    out = {}
    for name, score in CHANNELS.items():
        scores = score(options)
        out[name] = None if scores is None else ranks_from_scores(options, scores)
    return out


# --------------------------------------------------------------------------
def credit(p, b):
    k = len(p)
    return sum(pi * bi for pi, bi in zip(p, b)) - 1.0 / k


def cluster_ci(per_cluster, k, reps, alpha=0.05, seed=SEED):
    """Interval on the collected credit, resampling whole source studies.

    ``per_cluster`` maps a cluster to its (key-rank counter, chosen-rank counter);
    both add across clusters, so a replicate is a sum of two length-``k`` vectors
    and the credit is recomputed on each one rather than held fixed.
    """
    keys = list(per_cluster)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        key_counts, choice_counts = [0] * k, [0] * k
        for name in (rng.choice(keys) for _ in keys):
            a, b = per_cluster[name]
            for r in range(k):
                key_counts[r] += a[r]
                choice_counts[r] += b[r]
        total_a, total_b = sum(key_counts), sum(choice_counts)
        if total_a and total_b:
            draws.append(credit([x / total_a for x in key_counts],
                                [x / total_b for x in choice_counts]))
    if not draws:
        return None
    draws.sort()
    lo = draws[max(0, int(math.floor((alpha / 2) * len(draws))))]
    hi = draws[min(len(draws) - 1, int(math.ceil((1 - alpha / 2) * len(draws))) - 1)]
    return [lo, hi]


def attribute(rows, conditions, k, reps):
    """Per arm and per channel: where the key sits, what the solver picks, credit."""
    letters = letters_for(k)
    cache = {}
    per_arm = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: ([0] * k, [0] * k))))
    accuracy = defaultdict(list)

    matched = mismatched = 0
    for row in rows:
        key = (row["question_id"], row["arm"], row["draw"])
        condition = conditions.get(key)
        if condition is None:
            continue
        # The outcome files record no option text, so the options a channel is
        # measured on are *rebuilt* here from the question file and the seed. If
        # the repair has changed since the probe ran, the rebuild is a different
        # file and every channel below is measured on options the solver never
        # saw -- silently. The recorded rank and letter are the two things that
        # can be checked, so they are.
        if (condition["key_rank"] == row["key_rank"]
                and condition["target_letter"] == row["target_letter"]):
            matched += 1
        else:
            mismatched += 1
        accuracy[row["arm"]].append(row["correct"])
        options = tuple(condition["options_shown"])
        if options not in cache:
            cache[options] = channel_ranks(list(options))
        ranks = cache[options]
        target = letters.index(row["target_letter"])
        picked = letters.index(row["predicted_letter"])
        for channel, slot_ranks in ranks.items():
            if slot_ranks is None:
                continue
            key_counts, choice_counts = per_arm[row["arm"]][channel][row["cluster"]]
            key_counts[slot_ranks[target]] += 1
            choice_counts[slot_ranks[picked]] += 1

    out = {"_rebuild": {"matched": matched, "mismatched": mismatched}}
    for arm, channels in per_arm.items():
        block = {"accuracy": statistics.fmean(accuracy[arm]),
                 "n_decisions": len(accuracy[arm]), "channels": {}}
        for channel, clusters in channels.items():
            key_total, choice_total = [0] * k, [0] * k
            for a, b in clusters.values():
                for r in range(k):
                    key_total[r] += a[r]
                    choice_total[r] += b[r]
            n_key, n_choice = sum(key_total), sum(choice_total)
            if not (n_key and n_choice):
                continue
            p = [x / n_key for x in key_total]
            b = [x / n_choice for x in choice_total]
            block["channels"][channel] = {
                "n_items": n_key, "n_clusters": len(clusters),
                "key_by_rank_p": {str(j): p[j] for j in range(k)},
                "choice_by_rank_b": {str(r): b[r] for r in range(k)},
                "credit_collected": credit(p, b),
                "credit_collected_ci95": cluster_ci(clusters, k, reps),
                "credit_available": max(p) - 1.0 / k,
            }
        out[arm] = block
    return out


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe-dir", type=Path, default=Path("results/probe_raw"))
    parser.add_argument("--jsonl", required=True,
                        help="the question file the probe was run on")
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--n-options", type=int, default=None)
    parser.add_argument("--repair-mode", default=None,
                        help="override the repair recorded in the outcome files")
    parser.add_argument("--search", default=None,
                        help="override the search recorded in the outcome files")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--held-out", action="store_true",
                        help="also attribute the orderings no repair targets "
                             "(held_out_orderings.py), as diagnostics")
    parser.add_argument("--output", type=Path, default=Path("results/channel_attribution.json"))
    args = parser.parse_args()
    if args.held_out:
        print(f"diagnostic channels: {', '.join(add_held_out_channels())}")

    paths = outcome_files(args.probe_dir)
    if not paths:
        raise SystemExit(f"no outcome files in {args.probe_dir}")

    items, k = load_items(args.jsonl, args.question_field, args.n_options)
    models, rebuilt = [], {}
    for path in paths:
        with open_outcomes(path) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        arms = tuple(sorted({r["arm"] for r in rows}))
        repair = args.repair_mode or rows[0].get("repair_mode", "rank")
        search = args.search or rows[0].get("search", "sample")
        calibrated = bool(rows[0].get("calibrated", False))
        # Rebuilt once per (arms, repair, search) combination, not once per model:
        # a multi-channel repair over a thousand items is not cheap to redraw.
        key = (arms, repair, search, calibrated)
        if key not in rebuilt:
            rebuilt[key] = {(c["question_id"], c["arm"], c["draw"]): c
                            for c in build_conditions(items, args.draws, args.seed, k,
                                                      arms, repair, calibrate=calibrated,
                                                      search=search)}
        conditions = rebuilt[key]
        arms_block = attribute(rows, conditions, k, args.reps)
        rebuild = arms_block.pop("_rebuild")
        seen = rebuild["matched"] + rebuild["mismatched"]
        if seen and rebuild["mismatched"] / seen > 0.001:
            raise SystemExit(
                f"{rows[0]['model']}: {rebuild['mismatched']} of {seen} rebuilt "
                "conditions disagree with the recorded rank or letter, so the "
                "repair has changed since this probe ran and every channel below "
                "would be measured on options the solver never saw. Re-run the "
                "probe, or attribute with the code that produced it.")
        record = {"model": rows[0]["model"], "n_options": k, "repair_mode": repair,
                  "search": search, "calibrated": calibrated,
                  "rebuild_agreement": (rebuild["matched"] / seen) if seen else None,
                  "arms": arms_block}
        models.append(record)
        print(f"  attributed {record['model']} "
              f"({rebuild['matched']}/{seen} conditions rebuilt exactly)", flush=True)

    doc = {"channels": list(CHANNELS),
           "note": ("p^f measured from the presented options, b^f from what the solver "
                    "picked; credit is <p^f,b^f> - 1/k, the score a solver reading only "
                    "that statistic would collect. Intervals resample source studies."),
           "question_file": args.jsonl, "draws": args.draws, "seed": args.seed,
           "bootstrap_reps": args.reps, "models": models}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}\n")

    for record in models:
        print(f"--- {record['model']}  (k={record['n_options']}, "
              f"{record['repair_mode']} repair)")
        print(f"  {'arm':22s} {'accuracy':>9s}  " + "  ".join(
            f"{name[:18]:>18s}" for name in CHANNELS))
        for arm in sorted(record["arms"]):
            block = record["arms"][arm]
            cells = []
            for name in CHANNELS:
                channel = block["channels"].get(name)
                cells.append("---".rjust(18) if not channel
                             else f"{channel['credit_collected']:+7.1%} of "
                                  f"{channel['credit_available']:+6.1%}".rjust(18))
            print(f"  {arm:22s} {block['accuracy']:9.1%}  " + "  ".join(cells))
        print()


if __name__ == "__main__":
    main()
