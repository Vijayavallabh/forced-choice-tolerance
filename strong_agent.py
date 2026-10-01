#!/usr/bin/env python3
r"""A strong closed agent with the data: gpt-5.1 as BixBench's published agent (not part of the confirmatory test).

gpt-5.1 (served through Azure OpenAI) ran ``bixbench_agent.py``'s published protocol with the data on v1.5's
105 numeric questions, two runs each; ``bixbench_withdata.py`` scored every run as it scores the paper's other
configurations, with gpt-4o as the MCQ grader through the three option sets (forced and with BixBench's refusal
option) and as the open-ended judge. This reads, per item and then over items with capsule-bootstrap intervals
(plain 95%):

* how often the submitted number is within 5% of the key, and the open-ended grade;
* how many answers give no number, and how many of those say the value cannot be determined;
* gpt-4o's forced-choice and refusal-option grades of the released options, and each minus the tolerance
  (the paper prints these from ``score_decomposition.py``, with its double-bootstrap intervals);
* $U-P$ on the moved keys and over all numeric items, under the nearest-option rule
  (``run_set_scaling.one``, the definition of the paper's who-pays table) and under gpt-4o;
* where gpt-5.1 falls against the trend of the other with-data configurations: the rule's gain over all items
  predicted by a least-squares line in the share within 5%, fitted on them.

The trajectories (anonymised as the paper's other runs are) and gpt-4o's cached replies ship in
``results/strong_agent/``; the scored rows the numbers come from are ``results/strong_agent_rows.json.gz``.

    python3 strong_agent.py               # results/strong_agent.json (from the shipped scored rows)
    python3 strong_agent.py --latex       # the paper's table rows
"""
import collections
import gzip
import json
import pathlib
import re
import sys

import numpy as np

import bixbench_withdata as bw
import replication as rp
import run_set_scaling as rss
from answer_numbers import graded

ROOT = pathlib.Path(__file__).resolve().parent
ROWS = ROOT / "results" / "strong_agent_rows.json.gz"
BUILD_ROWS = ROOT / "build" / "openai" / "withdata_rows_gpt51.json"
BUILD_RUNS = ROOT / "build" / "openai" / "agent_runs" / "gpt-5.1-react" / "data"
BUILD_CACHE = ROOT / "build" / "openai" / "reader_cache_gpt51.jsonl"
SHIPPED = ROOT / "results" / "strong_agent"
OUT = ROOT / "results" / "strong_agent.json"
MODEL = "gpt-5.1-react"
READER = "gpt-4o"
LEVEL = 0.95


def load_rows():
    if BUILD_ROWS.exists():
        rows = json.loads(BUILD_ROWS.read_text())
        with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=ROWS.open("wb")) as fh:
            fh.write(json.dumps(sorted(rows, key=lambda r: (r["question_id"], r["rollout"])), sort_keys=True).encode())
    with gzip.open(ROWS, "rt") as fh:
        return [r for r in json.load(fh) if r["model"] == MODEL and r["condition"] == "data" and r["numeric"]]


def pack():
    """Write the shipped copies where the runs were made: every trajectory, one gzipped JSON line each,
    and the grader's replies."""
    if not BUILD_RUNS.exists():
        return
    SHIPPED.mkdir(parents=True, exist_ok=True)
    ts = sorted((json.loads(p.read_text()) for p in BUILD_RUNS.glob("*.json")),
                key=lambda t: (t["question_id"], t["rollout"]))
    with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=(SHIPPED / f"trajectories_{MODEL}.jsonl.gz").open("wb")) as fh:
        fh.write("".join(json.dumps(bw.anonymous(t)) + "\n" for t in ts).encode())
    if BUILD_CACHE.exists():
        with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=(SHIPPED / f"reader_cache_{MODEL}.jsonl.gz").open("wb")) as fh:
            fh.write(BUILD_CACHE.read_bytes())


def interval(per_run, qs, capsule):
    """Mean over items of each item's mean over runs, with a capsule-bootstrap interval, in points."""
    by_q = collections.defaultdict(list)
    for q, v in per_run:
        if v is not None:
            by_q[q].append(float(v))
    items = [q for q in qs if q in by_q]
    return bw.cluster_interval([np.mean(by_q[q]) for q in items], [capsule[q] for q in items], level=LEVEL)


def served():
    """The versions the API reported serving the agent, from the local usage ledger when it is present,
    else as the shipped report recorded them."""
    ledger = ROOT / "build" / "openai_usage.jsonl"
    if ledger.exists():
        got = {json.loads(l).get("model") for l in ledger.open() if json.loads(l).get("tag") == "agent:gpt-5.1"}
        return sorted(m for m in got if m)
    return json.loads(OUT.read_text())["served"] if OUT.exists() else []


# an answer that says the value cannot be had, in the words gpt-5.1 used
DECLINE = re.compile(r"cannot|can't|could not|unable|not be (?:determined|computed)|undefined|no \w+ (?:values|output)",
                     re.I)


def read_share(r, key):
    reads = r["reads"].get(key)
    return None if not reads else float(np.mean([x["correct"] for x in reads]))


def main():
    if "--latex" in sys.argv:
        print("\n".join(table_rows(json.loads(OUT.read_text()))))
        return
    pack()
    rows = load_rows()
    sets = bw.option_sets()
    capsule = {r["question_id"]: r["capsule"] for r in rows}
    qs = sorted(capsule)
    moved = [q for q in qs if rp.group_of(sets[q]) == "moved to an edge"]
    # read from the answer by the current reader (``answer_numbers``), not the field stored at grading
    within = [(r["question_id"], float(bool(r["answer"]) and bool(graded(r["answer"], sets[r["question_id"]]["released"][0],
                                                                          0.05)))) for r in rows]
    # a run with no answer is graded wrong, as upstream scores it
    forced = [(r["question_id"], read_share(r, f"{READER}|released|forced") or 0.0) for r in rows]
    refusal = [(r["question_id"], read_share(r, f"{READER}|released|refusal") or 0.0) for r in rows]
    diff = lambda a, b: [(q, x - y) for (q, x), (_, y) in zip(a, b)]
    g = {arm: [(r["question_id"], read_share(r, f"{READER}|{arm}|forced") or 0.0) for r in rows]
         for arm in ("released", "placebo", "repaired")}
    rule = {arm: [(r["question_id"], float(bw.nearest_is_key(r["answer"], sets[r["question_id"]][arm])))
                  for r in rows] for arm in ("released", "placebo", "repaired")}
    report = {"not_part_of_the_confirmatory_test": True, "model": MODEL, "reader": READER,
              "served": served(), "n_runs": len(rows), "n_items": len(qs),
              "n_moved": len(moved), "level": LEVEL,
              "submitted": 100.0 * float(np.mean([r["termination"] == "submitted" for r in rows])),
              "no_number": 100.0 * float(np.mean([not re.search(r"\d", str(r["answer"] or "")) for r in rows])),
              "no_number_runs": sum(not re.search(r"\d", str(r["answer"] or "")) for r in rows),
              "no_number_declines": sum(not re.search(r"\d", str(r["answer"] or "")) and bool(DECLINE.search(str(r["answer"])))
                                        for r in rows),
              "within_5pct": interval(within, qs, capsule),
              "open": interval([(r["question_id"], float(bool(r["open"]))) for r in rows], qs, capsule),
              "forced": interval(forced, qs, capsule), "refusal": interval(refusal, qs, capsule),
              "forced_minus_tolerance": interval(diff(forced, within), qs, capsule),
              "refusal_minus_tolerance": interval(diff(refusal, within), qs, capsule)}
    for name, src in (("rule", rule), (READER, g)):
        up = diff(src["repaired"], src["placebo"])
        report[f"{name}|repaired-placebo|moved"] = interval(up, moved, capsule)
        report[f"{name}|repaired-placebo|all"] = interval(up, qs, capsule)
    # the who-pays table's own quantities, from its own function
    report["who_pays"] = rss.one([(r["question_id"], r["answer"]) for r in rows], sets)
    scaling = json.loads((ROOT / "results" / "run_set_scaling.json").read_text())["run_sets"]
    xs = np.array([v["within_5pct"] for v in scaling.values()])
    ys = np.array([v["all"] for v in scaling.values()])
    slope, intercept = np.polyfit(xs, ys, 1)
    x = report["who_pays"]["within_5pct"]
    report["trend"] = {"n_run_sets": len(xs), "slope": float(slope), "intercept": float(intercept),
                       "predicted_all": float(intercept + slope * x),
                       "max_within_5pct_fitted": float(xs.max())}
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    for k, v in report.items():
        print(f"{k:34s} {v}")


def table_rows(report):
    ci = lambda e: f"${e['mean']:+.1f}$ $[{e['lo']:+.1f},{e['hi']:+.1f}]$"
    pc = lambda e: f"${e['mean']:.1f}$ $[{e['lo']:.1f},{e['hi']:.1f}]$"
    return [f"within $5\\%$ of the key & {pc(report['within_5pct'])}\\\\",
            f"graded correct open-ended by gpt-4o & {pc(report['open'])}\\\\",
            f"$U-P$, nearest-option rule, moved keys & {ci(report['rule|repaired-placebo|moved'])}\\\\",
            f"\\quad all numeric items & {ci(report['rule|repaired-placebo|all'])}\\\\",
            f"$U-P$, gpt-4o forced, moved keys & {ci(report['gpt-4o|repaired-placebo|moved'])}\\\\",
            f"\\quad all numeric items & {ci(report['gpt-4o|repaired-placebo|all'])}\\\\"]


if __name__ == "__main__":
    main()
