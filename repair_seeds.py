#!/usr/bin/env python3
r"""Is the trade a property of one rewrite? The repair, drawn again many times.

Every with-data result reads the released options against one repair, drawn once at
``mcq_audit.py``'s default seed; another draw moves other keys to an edge. This redraws
the repair at many seeds on each release -- ``mcq_audit.apply_repair`` with the
command's defaults, the call that built both releases' repairs -- and re-reads the same
frozen runs through every draw by the option nearest the submitted number: v1.5's seven
run sets with the data and without it, and v1.0's published runs of gpt-4o and Claude
3.5 Sonnet. It is exploratory, and not part of PREREGISTRATION.md.

    python3 repair_seeds.py --seeds 100      # results/repair_seeds.json
"""
import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

import bixbench_withdata as bw
import bracketing
import mcq_audit
import replication
from bracketing import miss_bin, open_side
from channel_survey import numeric_ranks
from option_artifacts import parse_number

ROOT = pathlib.Path(__file__).resolve().parent
FIRST_SEED = 20260918                      # mcq_audit.py's default, the draw the paper reads


def repairs(jsonl, seed):
    """{question_id: repaired options, key first} for every numeric item of the file."""
    rows = mcq_audit.read_rows(pathlib.Path(jsonl))
    items, k = mcq_audit.build_items(rows, "ideal", "distractors", "capsule_uuid", "question", None)
    repaired, *_ = mcq_audit.apply_repair(rows, items, "distractors", seed, k=k)
    out = {}
    for row in repaired:
        options = [row["ideal"], *row["distractors"]]
        if numeric_ranks(options) is not None:
            out[row["question_id"]] = options
    return out


def read(runs, released, repaired, capsule):
    """The moved keys' pooled gain, and what the draw newly credits there, for one repair."""
    moved = {q for q in released if q in repaired
             and replication.key_class(released[q]) == "bracketed" and replication.key_class(repaired[q]) == "edge"}
    per_item = defaultdict(list)
    gained, far, side = 0, 0, 0
    for trajectories in runs.values():
        by = defaultdict(list)
        for q, answer in trajectories:
            if q not in moved:
                continue
            before, after = bw.nearest_is_key(answer, released[q]), bw.nearest_is_key(answer, repaired[q])
            by[q].append(after - before)
            if after > before and answer and parse_number(answer) is not None:
                gained += 1
                far += miss_bin(answer, released[q][0]) in ("25 to 100%", "over 100%")
                side += int(open_side(answer, repaired[q]))
        for q, gs in by.items():
            per_item[q].append(float(np.mean(gs)))
    items = sorted(per_item)
    ci = bw.cluster_interval([float(np.mean(per_item[q])) for q in items], [capsule[q] for q in items], level=0.95)
    return {"n_moved": len(moved), "gain": ci, "n_gained": gained,
            "share_far": far / gained if gained else None, "share_open_side": side / gained if gained else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=100)
    args = ap.parse_args()
    seeds = [FIRST_SEED + i for i in range(args.seeds)]

    sets15 = bw.option_sets()
    released15 = {q: s["released"] for q, s in sets15.items()}
    items15 = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    capsule15 = {q: it["capsule_uuid"] for q, it in items15.items()}
    runs15 = {cond: {run: [(r["question_id"], r["answer"]) for r in bracketing.rows_of(run)
                           if r["numeric"] and r["condition"] == cond] for run in bracketing.RUNS}
              for cond in ("data", "nodata")}
    doc = replication.load_extract()
    released10 = {r["question_id"]: [r["ideal"], *r["distractors"]] for r in map(json.loads, open(replication.NUMERIC))}
    capsule10 = {q: e["capsule"] for q, e in doc["items"].items()}
    runs10 = {run: [(q, a) for q, a, _ in doc["open_runs"][run]] for run in replication.OPEN_RUNS}

    report = {"seeds": seeds, "v1.5|data": [], "v1.5|nodata": [], "v1.0|published": []}
    for seed in seeds:
        rep15 = repairs(ROOT / "data" / "bixbench.jsonl", seed)
        rep10 = repairs(replication.ITEMS, seed)
        for key, runs, rel, rep, cap in (("v1.5|data", runs15["data"], released15, rep15, capsule15),
                                         ("v1.5|nodata", runs15["nodata"], released15, rep15, capsule15),
                                         ("v1.0|published", runs10, released10, rep10, capsule10)):
            res = read(runs, rel, rep, cap)
            res["seed"] = seed
            report[key].append(res)
        a, b, c = (report[k][-1] for k in ("v1.5|data", "v1.5|nodata", "v1.0|published"))
        print(f"seed {seed}: v1.5 data {a['gain']['mean']:+5.1f} ({a['n_moved']} moved), nodata "
              f"{b['gain']['mean']:+5.1f}; v1.0 {c['gain']['mean']:+5.1f} ({c['n_moved']} moved)", flush=True)
    summary = {}
    for key in ("v1.5|data", "v1.5|nodata", "v1.0|published"):
        means = np.array([r["gain"]["mean"] for r in report[key]])
        summary[key] = {"n_seeds": len(means), "min": float(means.min()), "max": float(means.max()),
                        "median": float(np.median(means)),
                        "n_lower_end_above_zero": int(sum(r["gain"]["lo"] > 0 for r in report[key])),
                        "n_moved_range": [min(r["n_moved"] for r in report[key]), max(r["n_moved"] for r in report[key])],
                        "share_far_median": float(np.median([r["share_far"] for r in report[key] if r["share_far"] is not None])),
                        "share_open_side_median": float(np.median([r["share_open_side"] for r in report[key]
                                                                   if r["share_open_side"] is not None]))}
        print(key, summary[key])
    report["summary"] = summary
    out = ROOT / "results" / "repair_seeds.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
