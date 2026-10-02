#!/usr/bin/env python3
r"""Three current agents with the data: gpt-6-luna, DeepSeek-V4-Pro and Kimi-K3 as BixBench's published agent
(not part of the pre-specified test).

Each ran ``bixbench_agent.py``'s published protocol with the data, once on each of v1.5's 105 numeric questions
(Kimi-K3's runs still going when its runs were stopped are left out, and each agent's count is reported), through
Azure AI Foundry. gpt-6-luna ran with gpt-5.1's settings: medium reasoning effort (through the Responses API, which
is where that deployment takes function tools with reasoning), a context of 272,000 tokens and 300,000 characters of
notebook in view. DeepSeek-V4-Pro and Kimi-K3 ran through chat completions with no reasoning effort set, a context of
128,000 tokens and 200,000 characters in view. All three: 32,768-token replies, temperature 1.0, 40 steps, an hour a
run and 10 minutes a cell. DeepSeek-V4-Pro's runs are a second run with one driver (1 Oct 2026): its first had two
drivers on one output directory, which overwrote each other's finished episodes (``RERUN`` below;
results/frontier_agents/reruns_DeepSeek-V4-Pro-react.json records both). ``bixbench_withdata.py`` scored every run
as it scored gpt-5.1's: gpt-4o 2024-11-20 as the MCQ grader
through the three option sets, forced and with BixBench's refusal option, and as the open-ended judge. For each
agent this reports what ``strong_agent.py`` reports for gpt-5.1 (the share within 5% of the key, the open-ended
grade, the forced-choice and refusal-option scores minus the tolerance, and U - P on the moved keys and over all
numeric items under the nearest-option rule and under gpt-4o), Table 2's quantities from
``score_decomposition.summarise`` (calibrated intervals; the misses accepted by the option nearest the number;
the correct answers accepted), and the forced score corrected for guessing from ``formula_scoring.summarise``.

The trajectories and gpt-4o's cached replies ship in ``results/frontier_agents/``; the scored rows are
``results/frontier_agents_rows.json.gz``.

    python3 frontier_agents.py            # results/frontier_agents.json
    python3 frontier_agents.py --latex    # the appendix table's rows
"""
import gzip
import json
import pathlib
import re
import sys
import zlib

import numpy as np

import bixbench_withdata as bw
import formula_scoring as fs
import replication as rp
import run_set_scaling as rss
import score_decomposition as sd
import strong_agent as sa
from answer_numbers import graded

ROOT = pathlib.Path(__file__).resolve().parent
ROWS = ROOT / "results" / "frontier_agents_rows.json.gz"
BUILD_ROWS = ROOT / "build" / "openai" / "withdata_rows_frontier.json"
BUILD_RUNS = ROOT / "build" / "openai" / "agent_runs"
BUILD_CACHE = ROOT / "build" / "openai" / "reader_cache_frontier.jsonl"
# DeepSeek-V4-Pro was run again with one driver: its first runs came from two drivers at once, which overwrote each
# other's finished episodes. Its runs and scored rows are the rerun's (gpt-4o's reads of them were added to
# BUILD_CACHE); the first runs ship in results/frontier_agents/superseded/.
RERUN = {"DeepSeek-V4-Pro-react": {"runs": ROOT / "build" / "openai" / "agent_runs_rerun1",
                                   "rows": ROOT / "build" / "openai" / "withdata_rows_frontier_rerun1.json"}}
SHIPPED = ROOT / "results" / "frontier_agents"
OUT = ROOT / "results" / "frontier_agents.json"
MODELS = (("gpt-6-luna-react", "gpt-6-luna"), ("DeepSeek-V4-Pro-react", "DeepSeek-V4-Pro"),
          ("FW-Kimi-K3-react", "Kimi-K3"))
READER = sa.READER
# the agents the paper reports: those whose answers the tolerance reads, the last number in the answer being the
# number they submit; Kimi-K3 states its number first in long answers, so the last number is often another
REPORTED = ("gpt-6-luna", "DeepSeek-V4-Pro")


def load_rows():
    if BUILD_ROWS.exists() and all(where["rows"].exists() for where in RERUN.values()):
        rows = [r for r in json.loads(BUILD_ROWS.read_text()) if r["model"] not in RERUN]
        for model, where in RERUN.items():
            rows += [r for r in json.loads(where["rows"].read_text()) if r["model"] == model]
        with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=ROWS.open("wb")) as fh:
            fh.write(json.dumps(sorted(rows, key=lambda r: (r["model"], r["question_id"], r["rollout"])),
                                sort_keys=True).encode())
    with gzip.open(ROWS, "rt") as fh:
        return [r for r in json.load(fh) if r["condition"] == "data" and r["numeric"]]


def pack():
    """The shipped copies, where the runs were made: every trajectory, one gzipped JSON line each, and the
    grader's replies."""
    if not BUILD_RUNS.exists():
        return
    SHIPPED.mkdir(parents=True, exist_ok=True)
    for model, _ in MODELS:
        if model in RERUN and not RERUN[model]["rows"].exists():
            continue                      # a rerun ships once it has been graded
        paths = sorted((RERUN.get(model, {}).get("runs", BUILD_RUNS) / model / "data").glob("*.json"))
        if not paths:
            continue
        ts = sorted((json.loads(p.read_text()) for p in paths), key=lambda t: (t["question_id"], t["rollout"]))
        with gzip.GzipFile(filename="", mode="wb", mtime=0,
                           fileobj=(SHIPPED / f"trajectories_{model}.jsonl.gz").open("wb")) as fh:
            fh.write("".join(json.dumps(bw.anonymous(t)) + "\n" for t in ts).encode())
    if BUILD_CACHE.exists():
        with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=(SHIPPED / "reader_cache.jsonl.gz").open("wb")) as fh:
            fh.write(BUILD_CACHE.read_bytes())


def as_runs(rows, label, sets):
    """The rows as ``score_decomposition``'s runs, read by gpt-4o through the released options."""
    out = []
    for r in rows:
        reads = {}
        for name, mode in (("gpt-4o forced", "forced"), ("gpt-4o may-decline", "refusal")):
            got = r["reads"].get(f"{READER}|released|{mode}")
            if got is not None:
                reads[name] = [{"correct": x["correct"], "rank": x.get("rank"), "refused": x.get("refused", False)}
                               for x in got]
        out.append({"key": f"D4|{r['model']}|data|{r['question_id']}|{r['rollout']}", "group": label,
                    "run": r["model"], "q": r["question_id"], "capsule": r["capsule"], "answer": r["answer"],
                    "options": sets[r["question_id"]]["released"], "grader": bool(r["open"]), "reads": reads})
    return out


def one(rows, label, sets, written):
    capsule = {r["question_id"]: r["capsule"] for r in rows}
    qs = sorted(capsule)
    moved = [q for q in qs if rp.group_of(sets[q]) == "moved to an edge"]
    # read from the answer by the current reader (answer_numbers), not the field stored at grading; the first
    # number too, since Kimi-K3 states its number first
    key = lambda r: sets[r["question_id"]]["released"][0]
    within = [(r["question_id"], float(bool(r["answer"]) and bool(graded(r["answer"], key(r), 0.05)))) for r in rows]
    first = [(r["question_id"], float(bool(r["answer"]) and bool(graded(r["answer"], key(r), 0.05, pick="first"))))
             for r in rows]
    forced = [(r["question_id"], sa.read_share(r, f"{READER}|released|forced") or 0.0) for r in rows]
    refusal = [(r["question_id"], sa.read_share(r, f"{READER}|released|refusal") or 0.0) for r in rows]
    diff = lambda a, b: [(q, x - y) for (q, x), (_, y) in zip(a, b)]
    g = {arm: [(r["question_id"], sa.read_share(r, f"{READER}|{arm}|forced") or 0.0) for r in rows]
         for arm in ("released", "placebo", "repaired")}
    rule = {arm: [(r["question_id"], float(bw.nearest_is_key(r["answer"], sets[r["question_id"]][arm])))
                  for r in rows] for arm in ("released", "placebo", "repaired")}
    no_number = [not re.search(r"\d", str(r["answer"] or "")) for r in rows]
    lengths = sorted(len(str(r["answer"] or "")) for r in rows)
    out = {"label": label, "n_runs": len(rows), "n_items": len(qs), "n_moved": len(moved),
           "median_answer_chars": lengths[len(lengths) // 2] if lengths else None,
           "submitted": 100.0 * float(np.mean([r["termination"] == "submitted" for r in rows])),
           "no_answer": sum(not str(r["answer"] or "").strip() for r in rows),
           "no_number_runs": int(sum(no_number)),
           "within_5pct": sa.interval(within, qs, capsule),
           "within_5pct_first": sa.interval(first, qs, capsule),
           "open": sa.interval([(r["question_id"], float(bool(r["open"]))) for r in rows], qs, capsule),
           "forced": sa.interval(forced, qs, capsule), "refusal": sa.interval(refusal, qs, capsule),
           "forced_minus_tolerance": sa.interval(diff(forced, within), qs, capsule),
           "refusal_minus_tolerance": sa.interval(diff(refusal, within), qs, capsule)}
    for name, src in (("rule", rule), (READER, g)):
        up = diff(src["repaired"], src["placebo"])
        out[f"{name}|repaired-placebo|moved"] = sa.interval(up, moved, capsule)
        out[f"{name}|repaired-placebo|all"] = sa.interval(up, qs, capsule)
    out["who_pays"] = rss.one([(r["question_id"], r["answer"]) for r in rows], sets)
    runs = as_runs(rows, label, sets)
    caps = {r["q"]: r["capsule"] for r in runs}
    tag = f"v1.5 frontier|{label}"
    rng = np.random.default_rng(zlib.crc32(tag.encode()))
    out["table1"] = {reading: sd.summarise(runs, reading, caps, rng, written, f"{tag}|{reading}")
                     for reading in ("gpt-4o forced", "gpt-4o may-decline")}
    out["corrected"] = fs.summarise(runs, "gpt-4o forced", caps, f"{tag}|corrected")
    return out


def main():
    if "--latex" in sys.argv:
        print("\n".join(table_rows(json.loads(OUT.read_text()))))
        return
    pack()
    rows = load_rows()
    sets = bw.option_sets()
    written = sd.load_notebook_values()
    report = {"not_part_of_the_confirmatory_test": True, "reader": READER, "agents": {}}
    for model, label in MODELS:
        mine = [r for r in rows if r["model"] == model]
        if mine:
            report["agents"][label] = one(mine, label, sets, written)
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    for label, a in report["agents"].items():
        t = a["table1"]["gpt-4o forced"]
        print(f"{label:16s} n={a['n_runs']} within {a['within_5pct']['mean']:.1f} open {a['open']['mean']:.1f} | "
              f"forced-tol {t['score_minus_tolerance']['mean']:+.1f} [{t['score_minus_tolerance']['lo']:+.1f},"
              f"{t['score_minus_tolerance']['hi']:+.1f}] refusal-tol "
              f"{a['table1']['gpt-4o may-decline']['score_minus_tolerance']['mean']:+.1f} | corrected-tol "
              f"{a['corrected']['corrected_minus_tolerance']['mean']:+.1f} | rule U-P moved "
              f"{a['rule|repaired-placebo|moved']['mean']:+.1f} gpt-4o U-P moved {a['gpt-4o|repaired-placebo|moved']['mean']:+.1f}")
    print(f"wrote {OUT.relative_to(ROOT)}")


def table_rows(report):
    """One row per agent: the questions it finished, within 5%, open-ended, forced and refusal-option score minus the tolerance, the forced
    score corrected for guessing minus the tolerance, the misses accepted (all; nearest the key; nearest another
    option), and U - P on the moved keys under the rule and under gpt-4o."""
    ci = lambda e: f"${e['mean']:+.1f}$ {{\\scriptsize$[{e['lo']:+.1f},{e['hi']:+.1f}]$}}"
    rows = []
    for label, a in report["agents"].items():
        if label not in REPORTED:
            continue
        f, d = a["table1"]["gpt-4o forced"], a["table1"]["gpt-4o may-decline"]
        m = f["miss_rates"]
        rate = lambda k: "--" if m[k]["accepted"] is None else f"${m[k]['accepted']:.1f}$"
        rows.append(f"{label} & ${a['n_items']}$ & ${a['within_5pct']['mean']:.1f}$ & ${a['open']['mean']:.1f}$ & "
                    f"{ci(f['score_minus_tolerance'])} & ${d['score_minus_tolerance']['mean']:+.1f}$ & "
                    f"{ci(a['corrected']['corrected_minus_tolerance'])} & {rate('all')} & {rate('nearest')} & "
                    f"{rate('other')} & {ci(a['rule|repaired-placebo|moved'])} & "
                    f"${a['gpt-4o|repaired-placebo|moved']['mean']:+.1f}$\\\\")
    return rows


if __name__ == "__main__":
    main()
