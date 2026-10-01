#!/usr/bin/env python3
r"""Is BixBench's forced no-data score what the models know, short of stating it? (not registered)

BixBench reads a model's forced choice among four options, question shown and data absent, as "the pure
recall performance of both models". Its own runs with the options deleted show the same models stating the
answer on a few per cent of questions. A forced choice can still exceed what a model states: a model that
cannot give a value to within 5% may know roughly where it lies, and knowing that much is enough to pick
among four values. That is testable on the published runs themselves, with no model of ours:

* read each options-deleted reply that gives a number (``answer_extraction.last_number``, the tolerance's
  reading) by the option nearest it, through the released options: the key is nearest on some share of
  those replies, 25% if the number carries nothing about the answer;
* a model that picked, forced, the option nearest the number it would state, and guessed where it states
  none, would score ``stated * nearest + (1 - stated) * 25%`` on the numeric questions. The published
  forced run's accuracy on the same questions minus that is the margin the stated numbers do not explain;
* on v1.0, whose forced runs record the order shown, how often the forced pick is the option nearest the
  model's own stated number.

A reply that says the value cannot be determined without the data (``published_repair.DECLINE_TEXT``)
counts as stating no number even when it goes on to quote one. Plain 95% percentile cluster bootstraps
over capsules (``bixbench_withdata.cluster_interval``).

    python3 partial_knowledge.py     # results/partial_knowledge.json
"""
import ast
import glob
import json
import pathlib

import numpy as np

import answer_extraction as ax
import bixbench_withdata as bw
import published_repair as pub
import replication as rp
from option_artifacts import parse_number
from answer_numbers import nearest_ranks

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "partial_knowledge.json"
MODELS = {"gpt-4o": "gpt-4o", "claude-3-5-sonnet-latest": "Claude 3.5 Sonnet"}
CHANCE = 0.25
LEVEL = 0.95


def stated(reply, key):
    """The number a no-data reply states, or None: a reply saying the value cannot be determined states
    none, whatever it goes on to quote."""
    if not reply or pub.DECLINE_TEXT.search(str(reply)):
        return None
    return ax.last_number(reply, key)


def summary(rows):
    """rows: one per numeric question, with capsule, forced correctness, the stated number's reading."""
    cap = [r["capsule"] for r in rows]
    said = [r for r in rows if r["number"] is not None]
    out = {"n_items": len(rows), "n_clusters": len(set(cap)),
           "stated_share": 100.0 * len(said) / len(rows),
           "nearest_is_key_when_stated": bw.cluster_interval([r["nearest"] for r in said],
                                                             [r["capsule"] for r in said], level=LEVEL),
           "within_5pct_when_stated": 100.0 * float(np.mean([r["within"] for r in said])) if said else None,
           "forced": bw.cluster_interval([r["forced"] for r in rows], cap, level=LEVEL),
           "forced_when_stated": bw.cluster_interval([r["forced"] for r in said], [r["capsule"] for r in said],
                                                     level=LEVEL),
           "forced_when_not": bw.cluster_interval([r["forced"] for r in rows if r["number"] is None],
                                                  [r["capsule"] for r in rows if r["number"] is None], level=LEVEL)}
    predicted = [r["nearest"] if r["number"] is not None else CHANCE for r in rows]
    out["predicted_forced"] = 100.0 * float(np.mean(predicted))
    out["forced_minus_predicted"] = bw.cluster_interval([r["forced"] - p for r, p in zip(rows, predicted)], cap,
                                                        level=LEVEL)
    agree = [r for r in said if r.get("pick_is_nearest") is not None]
    if agree:
        out["pick_is_nearest_when_stated"] = bw.cluster_interval([r["pick_is_nearest"] for r in agree],
                                                                 [r["capsule"] for r in agree], level=LEVEL)
    return out


def v15(model, sets, capsule):
    base = "data/external/zero_shot_v15"
    # the published files write some ids in another case than the benchmark's
    free = {r["uuid"].strip().lower(): r for r in pub.read(f"{base}/{model}-grader-openended.csv")}
    forced = {r["uuid"].strip().lower(): r for r in pub.read(f"{base}/{model}-grader-mcq-refusal-False.csv")}
    rows = []
    for q, s in sets.items():
        options = s["released"]
        reply = free[q.lower()]["predicted"]
        number = stated(reply, options[0])
        rows.append({"q": q, "capsule": capsule[q], "forced": 1.0 if pub.truthy(forced[q.lower()]["correct"]) else 0.0,
                     "number": number,
                     "nearest": ax.nearest_is_key_last(reply, options) if number is not None else None,
                     "within": bool(pub_graded(reply, options[0])) if number is not None else None})
    return rows


def pub_graded(reply, key):
    from answer_numbers import graded
    return graded(reply, key, 0.05)


def v10(model, sets, capsule):
    base = "data/external/zero_shot_v10"
    free = {(r["uuid"], r["short_qid"]): r
            for r in pub.read(glob.glob(f"{base}/*refusal_True_openended_{model}_1.0.csv")[0])}
    forced = {(r["uuid"], r["short_qid"]): r
              for r in pub.read(glob.glob(f"{base}/*refusal_False_mcq_{model}_1.0.csv")[0])}
    rows = []
    for (uuid, short), f in forced.items():
        q = f"CapsuleFolder-{uuid}_q{short.rsplit('_q', 1)[1]}"
        if q not in sets:
            continue
        options = sets[q]["released"]
        choices = ast.literal_eval(f["choices"])
        shown = [(c.split(") ", 1)[1] if ") " in c else c).strip() for c in choices]
        values = [parse_number(v) for v in shown]
        # the forced run's options are the released set, in the order shown
        if sorted(values) != sorted(parse_number(o) for o in options):
            continue
        reply = free[(uuid, short)]["predicted"]
        number = stated(reply, options[0])
        row = {"q": q, "capsule": capsule[q], "forced": 1.0 if pub.truthy(f["correct"]) else 0.0,
               "number": number,
               "nearest": ax.nearest_is_key_last(reply, options) if number is not None else None,
               "within": bool(pub_graded(reply, options[0])) if number is not None else None,
               "pick_is_nearest": None}
        if number is not None and f["predicted"] in "ABCD" and f["predicted"]:
            picked = values["ABCD".index(f["predicted"])]
            nearest = [values[i] for i in nearest_ranks(number, values)]
            row["pick_is_nearest"] = (1.0 / len(nearest)) if picked in nearest else 0.0
        rows.append(row)
    return rows


def main():
    report = {"level": LEVEL, "chance": CHANCE, "not_registered": True}
    sets15 = bw.option_sets()
    cap15 = {json.loads(l)["question_id"]: json.loads(l)["capsule_uuid"] for l in open(ROOT / "data" / "bixbench.jsonl")}
    doc = rp.load_extract()
    sets10 = rp.v10_sets()
    cap10 = {q: e["capsule"] for q, e in doc["items"].items()}
    for model, label in MODELS.items():
        report[f"v1.5|{label}"] = summary(v15(model, sets15, cap15))
        report[f"v1.0|{label}"] = summary(v10(model, sets10, cap10))
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    fmt = lambda v: "--" if v is None else f"{v['mean']:5.1f} [{v['lo']:5.1f},{v['hi']:5.1f}] (n {v['n_items']})"
    for k, v in report.items():
        if not isinstance(v, dict):
            continue
        print(f"{k:24s} items {v['n_items']:3d}; states a number on {v['stated_share']:4.1f}%; nearest is key "
              f"{fmt(v['nearest_is_key_when_stated'])}; within 5% {v['within_5pct_when_stated']}")
        print(f"{'':24s} forced {fmt(v['forced'])}; when stated {fmt(v['forced_when_stated'])}; when not "
              f"{fmt(v['forced_when_not'])}; predicted {v['predicted_forced']:.1f}; forced - predicted "
              f"{fmt(v['forced_minus_predicted'])}"
              + (f"; pick nearest own number {fmt(v['pick_is_nearest_when_stated'])}"
                 if "pick_is_nearest_when_stated" in v else ""))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
