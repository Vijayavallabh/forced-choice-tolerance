#!/usr/bin/env python3
"""Does closing the orderings you audit close the ones you do not?

``mcq_audit`` audits four orderings of the options and the constructive repair
uniformises the key's rank under all four. The obvious objection is that this
is circular: a repair evaluated on the coordinates it targets will of course
look closed, and a benchmark maintainer cares about the orderings nobody
thought to write down.

So this holds five *more* orderings out. None of them is a target of the shipped
repair and none is reported by the audit; each is a different way of reading a
written number off the page:

    leading digit    the first significant digit, 1-9
    decimal places   digits after the point -- not the same as roundness,
                     since 0.0004 has one significant digit and four decimals
    digit sum        the digits added up
    string order     the rendered text in lexicographic order, which is what a
                     sorter that never parses the number sees
    length, later    written length again, with the opposite tie-break: among
      ties           options of equal length the audit's rank counts the ones
                     whose text sorts earlier and this counts the ones whose
                     text sorts later, which is what the surface rule family's
                     `take shortest_option` actually reads. A score with ties
                     is a family of orderings, and only one member was repaired

The same estimator the audit uses (``channel_survey.credit_lower_bound``: a
Bonferroni-simultaneous one-sided cluster-bootstrap lower bound on
``max_j p_j - 1/k``) is applied to each, on the released file and on the
constructed repair of it, at four options and at ten.

Two comparisons make the answer readable. Transfer is the released bound minus
the repaired one: how much of a channel the repair closed without aiming at it.
The clean-file reference is what these same orderings return on synthetic
files that have no channel at all (audit_calibration's construction), which is
the noise floor for a bound that is meant to be near zero.

The answer is not the same at both option counts, and the third thing measured
here is why. At four options the repair closes the held-out orderings the
released file leaks; at ten it closes none of them. The difference is *coupling*: lexicographic order of a written number is
monotone in its value only while the options share a digit width ("9.5"
precedes "12.7" as text and follows it as a number), and BixBench's
distractors, being perturbations of the key, mostly do share one -- so
uniformising the value rank uniformises the text rank with it. MMLU-Pro's
options span magnitudes, and the repair lowers the agreement further by
rendering distractors at varied precision. ``coupling`` reports both the
whole-permutation agreement and the per-ordering one, so the question "will
this repair transfer?" can be answered before the repair is run.

Two cautions about these orderings, both measured rather than assumed.
Decimal places is degenerate on released files -- every option carries the same
number of decimals on 87% of BixBench's items and 80% of MMLU-Pro's -- so there
it is mostly the lexicographic ordering under another name. And ``string
order`` is tied on every item by construction, which is its definition rather
than a flaw: it scores every option equally and lets feature_rank's text
tie-break do the ordering.

    python3 held_out_orderings.py --replicates 20 \
        --repaired-4 results/repaired/bixbench_v15_repaired_multi.jsonl \
        --released-10 build/mmlu_pro.jsonl --repaired-10 build/mmlu_pro_repaired.jsonl
"""
import argparse
import json
import random
import statistics
from pathlib import Path

import mcq_audit
from audit_calibration import synthetic_rows
from mcq_audit import build_items, read_rows, surface_channels


def _numbers(options):
    values = [mcq_audit.parse_number(o) for o in options]
    return None if any(v is None for v in values) else values


def _digits(text):
    """The digit characters of a written number, exponent and separators dropped."""
    body = str(text).strip().rstrip("%").replace(",", "")
    body = body.split("e")[0].split("E")[0]
    return "".join(c for c in body if c.isdigit())


def leading_digit_scores(options):
    """First significant digit. Benford's cue, and no function of length."""
    if _numbers(options) is None:
        return None
    out = []
    for option in options:
        stripped = _digits(option).lstrip("0")
        out.append(float(stripped[0]) if stripped else 0.0)
    return out


def decimal_place_scores(options):
    """Digits after the point. Distinct from roundness: 0.0004 is 1 sig digit, 4 places."""
    if _numbers(options) is None:
        return None
    out = []
    for option in options:
        body = str(option).strip().rstrip("%").replace(",", "")
        body = body.split("e")[0].split("E")[0]
        out.append(float(len(body.split(".")[1])) if "." in body else 0.0)
    return out


def digit_sum_scores(options):
    """The digits added up: a surface statistic with no ordering of its own."""
    if _numbers(options) is None:
        return None
    return [float(sum(int(c) for c in _digits(o) if c.isdigit())) for o in options]


def string_order_scores(options):
    """Lexicographic order of the rendered text.

    ``mcq_audit.feature_rank`` sorts on ``(score, str(option))``, so a constant
    score leaves the text as the whole sort key -- which is precisely this
    ordering. Restricted to numeric items, like the other three, because those
    are the items any repair here rewrites.
    """
    if _numbers(options) is None:
        return None
    return [0.0] * len(options)


def length_later_ties(options):
    """Written length again, with the tie broken the other way.

    A score with ties does not define one ordering but a family of them. The
    audit ranks by ``(length, text)`` ascending and the construction uniformises
    the key's rank under exactly that; ``text_artifacts.pick``, which is what the
    surface rule family uses, takes the arg-max of ``(-length, text)``, so among
    options tied on length it prefers the *later* text. That is a different
    statistic and the repair never saw it, which is why it is held out here.
    """
    if _numbers(options) is None:
        return None
    return [float(len(str(o))) for o in options]


HELD_OUT = {
    "leading digit": leading_digit_scores,
    "decimal places": decimal_place_scores,
    "digit sum": digit_sum_scores,
    "string order": string_order_scores,
    "length, later ties": length_later_ties,
}

# Which of them take the opposite tie-break when ranked.
HELD_OUT_TIES = {"length, later ties": "later"}


def coupling(items, k):
    """How close each held-out ordering is to the value ordering, on this file.

    This is the diagnostic that explains the result rather than restating it. A
    repair uniformises the key's rank under the orderings it targets; a held-out
    ordering inherits that only where it is close to a monotone function of one
    of them. Two summaries: how often the key's held-out rank equals its value
    rank, and how often the two orderings agree as whole permutations. The
    second is the stronger condition and the one that predicts transfer.
    """
    numeric = [it for it in items if it["rank"] is not None]
    out = {"n_items": len(numeric)}
    same_permutation = 0
    for item in numeric:
        options = list(item["options"])
        by_value = sorted(range(k), key=lambda i: mcq_audit.parse_number(options[i]))
        by_text = sorted(range(k), key=lambda i: str(options[i]))
        same_permutation += (by_value == by_text)
    out["lexicographic_is_value_order"] = same_permutation / max(1, len(numeric))
    for name, score in HELD_OUT.items():
        same = total = 0
        distinct = 0
        for item in numeric:
            # Scored directly rather than through mcq_audit.ORDERINGS: this
            # function must not depend on the held-out orderings having been
            # registered there, which only main() does.
            value_scores = mcq_audit.ORDERINGS["sorted value"](item["options"])
            held_scores = score(item["options"])
            if value_scores is None or held_scores is None:
                continue
            value_rank = mcq_audit.feature_rank(item["options"], value_scores)
            held_rank = mcq_audit.feature_rank(
                item["options"], held_scores,
                ties=HELD_OUT_TIES.get(name, "earlier"))
            same += (value_rank == held_rank)
            distinct += len(set(held_scores))
            total += 1
        out[name] = {"key_rank_agrees_with_value": same / max(1, total),
                     "mean_distinct_scores": distinct / max(1, total)}
    return out


def measure(rows, k, reps, cluster_field, key_field="ideal",
            distractor_field="distractors", question_field="question"):
    items, found_k = build_items(rows, key_field, distractor_field, cluster_field,
                                 question_field, k)
    if found_k != k:
        raise SystemExit(f"expected {k} options, found {found_k}")
    channels = surface_channels(items, reps, k, orderings=tuple(HELD_OUT))
    out = {}
    for name, block in channels["orderings"].items():
        stats = block.get("numeric items") or next(iter(block.values()), None)
        if stats is None:
            continue
        out[name] = {
            "n_items": stats["n_items"],
            "top_rank_share": stats["top_rank_share"],
            "plug_in": stats["max_credit_plug_in"],
            "bound": (stats["credit_bound"] or {}).get("credit_lower_bound"),
            "leaks": stats["leaks"],
        }
    out["_coupling"] = coupling(items, k)
    return out


def clean_reference(k, n_items, n_clusters, replicates, reps, seed):
    """What the held-out orderings return on files with no channel at all."""
    per_ordering = {name: [] for name in HELD_OUT}
    for replicate in range(replicates):
        rng = random.Random(f"{seed}|clean|{k}|{replicate}")
        rows = synthetic_rows(n_items, k, n_clusters, rng)
        got = measure(rows, k, reps, "capsule_uuid")
        for name, stats in got.items():
            if name in per_ordering:
                per_ordering[name].append(stats["bound"] or 0.0)
    return {name: {"widest": max(values),
                   "median": statistics.median(values),
                   "fire_rate": sum(1 for v in values if v > 0) / len(values)}
            for name, values in per_ordering.items()}


SURVEY_DIR = Path("data/external/survey")


def survey_sweep(reps, seed, min_items=60):
    """The four held-out orderings across every vendored survey file.

    Two files are not a general claim. This runs the same measurement over the
    eighteen-file survey's numeric subsets, which is the widest evidence in this
    repository that the channel is plural -- and puts the question to the file
    the paper holds up as the counterexample. AQuA-RAT's *value* channel is
    closed (its distractors are answers a wrong step produces, not perturbations
    of the right one); whether an ordering nobody measured is closed there too
    is a different question, and one the audit never asked.

    Files are grouped by option count, since a rank bound is only comparable
    within one.
    """
    from survey_to_jsonl import convert

    out = {}
    for path in sorted(SURVEY_DIR.glob("*.jsonl.gz")):
        rows = list(convert(path))
        counts = {}
        for row in rows:
            counts.setdefault(1 + len(row["distractors"]), []).append(row)
        for k, subset in sorted(counts.items()):
            if k < 3 or len(subset) < min_items:
                continue
            try:
                got = measure(subset, k, reps, "cluster")
            except SystemExit:
                continue
            numeric = got.get("_coupling", {}).get("n_items", 0)
            if numeric < min_items:
                continue
            out[f"{path.name.split('.')[0]}:{k}"] = got
            leaking = sorted(name for name, stats in got.items()
                             if name != "_coupling" and stats["leaks"])
            print(f"  {path.name.split('.')[0]:<14} k={k:<3} {numeric:>5} numeric items  "
                  f"leaks on {len(leaking)}/4: {', '.join(leaking) or 'none'}")
            for name, stats in sorted(got.items()):
                if name == "_coupling":
                    continue
                print(f"      {name:<16} bound {100 * (stats['bound'] or 0.0):+5.1f}"
                      f"{'  LEAKS' if stats['leaks'] else ''}")
            print(f"      lexicographic order is the value order on "
                  f"{got['_coupling']['lexicographic_is_value_order']:.1%} of items")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--released-4", type=Path, default=Path("data/bixbench.jsonl"))
    parser.add_argument("--repaired-4", type=Path)
    parser.add_argument("--released-10", type=Path)
    parser.add_argument("--repaired-10", type=Path)
    parser.add_argument("--survey-sweep", action="store_true",
                        help="instead of the two audited files, measure the four "
                             "held-out orderings across every vendored survey "
                             "file, grouped by option count")
    parser.add_argument("--lexicographic-10", type=Path,
                        help="a ten-option repair built with mcq_audit.py's "
                             "--with-lexicographic, which targets one of the four "
                             "held-out orderings directly. Measured as a third "
                             "file, to separate 'the repair missed it' from 'the "
                             "file cannot be repaired on it'")
    parser.add_argument("--option-counts", nargs="+", type=int, default=[4, 10],
                        choices=[4, 10],
                        help="run one option count at a time when the other "
                             "file is still being built")
    parser.add_argument("--cluster-field-4", default="capsule_uuid")
    parser.add_argument("--cluster-field-10", default="cluster")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--replicates", type=int, default=20,
                        help="clean synthetic files per option count")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--output", type=Path,
                        default=Path("results/held_out_orderings.json"))
    args = parser.parse_args()

    mcq_audit.ORDERINGS.update(HELD_OUT)
    mcq_audit.TIE_ORDERINGS.update(HELD_OUT_TIES)

    if args.survey_sweep:
        print("=== the held-out orderings across the vendored survey ===")
        sweep = survey_sweep(args.reps, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"orderings": sorted(HELD_OUT), "reps": args.reps, "files": sweep},
            indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")
        return

    # Only the files the requested option counts actually read: --option-counts
    # exists so one benchmark can be measured on its own, and demanding the
    # other one's files made that impossible.
    needed = [name for k, names in ((4, ("released_4", "repaired_4")),
                                    (10, ("released_10", "repaired_10")))
              if k in args.option_counts for name in names]
    for name in needed:
        if getattr(args, name) is None:
            raise SystemExit(f"--{name.replace('_', '-')} is required without "
                             "--survey-sweep")

    report = {"orderings": sorted(HELD_OUT), "reps": args.reps,
              "replicates": args.replicates, "files": {}, "clean": {}}
    for k, released, repaired, cluster_field in (
            (4, args.released_4, args.repaired_4, args.cluster_field_4),
            (10, args.released_10, args.repaired_10, args.cluster_field_10)):
        if k not in args.option_counts:
            continue
        print(f"=== {k} options ===")
        block = {}
        # Shape the clean reference like the file it is a reference for. These
        # were constants tuned to v1.5, and pointing --released-4 at a file with
        # 661 items in 39 subjects then compared its bound against a noise floor
        # computed for 105 items in 46.
        shaped = [item for item in build_items(read_rows(released), "ideal",
                                               "distractors", cluster_field,
                                               "question", k)[0]
                  if item["rank"] is not None]
        items_per_clean = len(shaped)
        clusters = len({item["cluster"] for item in shaped})
        for label, path in (("released", released), ("repaired", repaired)):
            block[label] = measure(read_rows(path), k, args.reps, cluster_field)
            print(f"  {path} as {label}")
            for name, stats in sorted(block[label].items()):
                if name == "_coupling":
                    continue
                flag = "  LEAKS" if stats["leaks"] else ""
                print(f"    {name:<16} top rank {stats['top_rank_share']:6.1%}  "
                      f"plug-in {100 * stats['plug_in']:+5.1f}  "
                      f"bound {100 * (stats['bound'] or 0.0):+5.1f} points{flag}")
            couple = block[label]["_coupling"]
            print(f"      coupling: lexicographic order is the value order on "
                  f"{couple['lexicographic_is_value_order']:.1%} of items; "
                  + ", ".join(
                      f"{name} agrees with the value rank on "
                      f"{couple[name]['key_rank_agrees_with_value']:.0%} "
                      f"({couple[name]['mean_distinct_scores']:.1f}/{k} distinct scores)"
                      for name in sorted(HELD_OUT)))
        if k == 10 and args.lexicographic_10 is not None:
            label = "repaired_lexicographic"
            block[label] = measure(read_rows(args.lexicographic_10), k, args.reps,
                                   cluster_field)
            print(f"  {args.lexicographic_10} as {label}")
            for name, stats in sorted(block[label].items()):
                if name == "_coupling":
                    continue
                flag = "  LEAKS" if stats["leaks"] else ""
                print(f"    {name:<16} top rank {stats['top_rank_share']:6.1%}  "
                      f"plug-in {100 * stats['plug_in']:+5.1f}  "
                      f"bound {100 * (stats['bound'] or 0.0):+5.1f} points{flag}")
        block["transfer"] = {
            name: (block["released"][name]["bound"] or 0.0)
                  - (block["repaired"][name]["bound"] or 0.0)
            for name in block["released"]
            if name in block["repaired"] and name != "_coupling"}
        report["files"][str(k)] = block
        clean = clean_reference(k, items_per_clean, clusters, args.replicates,
                                args.reps, args.seed)
        report["clean"][str(k)] = clean
        print(f"  clean reference ({args.replicates} synthetic files)")
        for name, stats in sorted(clean.items()):
            print(f"    {name:<16} widest {100 * stats['widest']:+5.1f} points, "
                  f"fires on {stats['fire_rate']:.0%} of clean files")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and set(args.option_counts) != {4, 10}:
        # Written one option count at a time: keep the other one's block.
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        for section in ("files", "clean"):
            merged = dict(previous.get(section, {}))
            merged.update(report[section])
            report[section] = merged
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
