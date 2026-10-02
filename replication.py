#!/usr/bin/env python3
r"""The preregistered replication (PREREGISTRATION.md): the trade on runs the paper had not read.

D1 is BixBench v1.0's own with-data runs of gpt-4o and Claude 3.5 Sonnet, which the
benchmark's authors published as ``eval_df.csv``: one row per trajectory, with the
agent's submitted answer, the question's released options (key first) and the
benchmark's grader verdict. D2 is new v1.5 run sets made after the registration
(``build/agent_runs_replication/``). Both are read the one way the plan fixes:
``bixbench_withdata.nearest_is_key`` through the released options, a placebo that
redraws every distractor and keeps the key's rank, and a repair that draws the rank
uniformly -- built by the same commands that built v1.5's.

    python3 replication.py build      # v1.0 question file + compact run extract
    python3 mcq_audit.py --jsonl build/bixbench_v10_items.jsonl \
        --repair build/bixbench_v10_repaired.jsonl --output results/mcq_audit_bixbench_v10.json
    python3 build_placebo_arm.py --items build/bixbench_v10_numeric_released.jsonl \
        --out build/bixbench_v10_numeric_placebo.jsonl
    python3 replication.py analyse    # results/replication.json

The published CSV is 1.1 GB and is not shipped; ``build`` checks its md5 and writes
the part the analysis reads -- every open-answer trajectory's answer and verdict,
and each question's options -- to ``results/bixbench_v10_published_runs.json.gz``,
from which ``analyse`` re-derives everything.
"""
import argparse
import ast
import gzip
import hashlib
import json
import pathlib
import random
import re
from collections import Counter, defaultdict

import numpy as np

import bixbench_withdata as bw
import claim_budget
from bracketing import MISS_BINS, miss_bin, open_side
from channel_survey import numeric_ranks
from option_artifacts import parse_number

ROOT = pathlib.Path(__file__).resolve().parent
SOURCE = {"url": "https://storage.googleapis.com/bixbench-results/eval_df.csv",
          "bytes": 1075070319, "md5": "3df10be01a348d2729de61da8e739b93"}
EXTRACT = ROOT / "results" / "bixbench_v10_published_runs.json.gz"
ITEMS = ROOT / "build" / "bixbench_v10_items.jsonl"
NUMERIC = ROOT / "build" / "bixbench_v10_numeric_released.jsonl"
REPAIRED = ROOT / "build" / "bixbench_v10_repaired.jsonl"
PLACEBO = ROOT / "build" / "bixbench_v10_numeric_placebo.jsonl"
OPEN_RUNS = ("4o_open_image", "4o_open_no_image", "claude_open_image", "claude_open_no_image")
MCQ_RUNS = {"4o_open_image": "4o_mcq_image_without_refusal",
            "4o_open_no_image": "4o_mcq_no_image_without_refusal",
            "claude_open_image": "claude_mcq_image_without_refusal",
            "claude_open_no_image": "claude_mcq_no_image_without_refusal"}
SEED = 20260924


# ---------------------------------------------------------------- build

def md5_of(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def picked_option(formatted, letter):
    """The option text a published MCQ reading picked, from the question as it was shown."""
    for line in (formatted or "").splitlines():
        m = re.match(r"\s*([A-Z])\.\s+(.*)$", line)
        if m and m.group(1) == (letter or "").strip().upper()[:1]:
            return m.group(2).strip()
    return None


def build(args):
    import csv
    import sys
    csv.field_size_limit(sys.maxsize)
    path = pathlib.Path(args.eval_df)
    if path.stat().st_size != SOURCE["bytes"] or md5_of(path) != SOURCE["md5"]:
        raise SystemExit(f"{path} is not the published eval_df.csv ({SOURCE['md5']})")
    items, runs, reads = {}, defaultdict(list), defaultdict(list)
    wanted = set(OPEN_RUNS) | set(MCQ_RUNS.values())
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            run = row["run_name"]
            if run not in wanted:
                continue
            options = ast.literal_eval(row["mcq_options"])
            qid = row["uuid"]
            entry = {"capsule": row["problem_id"], "question": row["question"],
                     "ideal": row["ideal_answer"], "options": options}
            if qid in items:
                assert items[qid] == entry, qid          # one question, one option set
            items[qid] = entry
            assert options[0] == row["ideal_answer"], qid
            if run in OPEN_RUNS:
                runs[run].append([qid, row["agent_answer"], row["correct"] == "True"])
            else:
                reads[run].append([qid, picked_option(row["formatted_question"], row["llm_answer"]),
                                   row["correct"] == "True"])
    doc = {"source": SOURCE, "items": items, "open_runs": dict(runs), "mcq_reads": dict(reads)}
    EXTRACT.parent.mkdir(exist_ok=True)
    with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=EXTRACT.open("wb")) as fh:
        fh.write(json.dumps(doc, sort_keys=True).encode())
    write_items(doc)
    print(f"{len(items)} questions, {sum(len(v) for v in runs.values())} open-answer trajectories "
          f"in {len(runs)} run sets; wrote {EXTRACT.relative_to(ROOT)}")


def load_extract():
    with gzip.open(EXTRACT, "rt") as fh:
        return json.load(fh)


def write_items(doc):
    """The v1.0 question file the repair reads, and its numeric part the placebo reads."""
    rows = [{"question_id": q, "question": e["question"], "ideal": e["ideal"],
             "distractors": e["options"][1:], "capsule_uuid": e["capsule"]}
            for q, e in sorted(doc["items"].items())]
    ITEMS.parent.mkdir(exist_ok=True)
    ITEMS.write_text("".join(json.dumps(r) + "\n" for r in rows))
    numeric = [r for r in rows if numeric_ranks([r["ideal"], *r["distractors"]]) is not None]
    NUMERIC.write_text("".join(json.dumps(r) + "\n" for r in numeric))
    print(f"{len(rows)} questions to {ITEMS.relative_to(ROOT)}, {len(numeric)} numeric "
          f"to {NUMERIC.relative_to(ROOT)}")


# ---------------------------------------------------------------- analysis

def _rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def key_class(options):
    ranks = numeric_ranks(options)
    return None if ranks is None else ("bracketed" if 0 < ranks[0] < len(options) - 1 else "edge")


def group_of(sets):
    """Where the repair took the key, from the option sets alone."""
    if "repaired" not in sets:
        return None
    before, after = key_class(sets["released"]), key_class(sets["repaired"])
    if before == "bracketed" and after == "edge":
        return "moved to an edge"
    if before == "edge" and after == "bracketed":
        return "moved inward"
    return "kept"


GROUPS = ("moved to an edge", "moved inward", "kept")


def v10_sets():
    """Each numeric v1.0 item's released, placebo and repaired values, key first."""
    tables = {arm: {r["question_id"]: [r["ideal"], *r["distractors"]] for r in _rows(path)}
              for arm, path in (("released", NUMERIC), ("repaired", REPAIRED), ("placebo", PLACEBO))}
    sets = {}
    for q, released in tables["released"].items():
        entry = {"released": released}
        for arm in ("repaired", "placebo"):
            options = tables[arm].get(q)
            if options and options[0] == released[0] and numeric_ranks(options) is not None:
                entry[arm] = options
        sets[q] = entry
    return sets


def hit_rates(trajectories, sets, arm):
    """Per item: the mean over a run set's trajectories of the nearest-option reading."""
    by_item = defaultdict(list)
    for q, answer in trajectories:
        if q in sets and arm in sets[q]:
            by_item[q].append(bw.nearest_is_key(answer, sets[q][arm]))
    return {q: float(np.mean(v)) for q, v in by_item.items()}


def gains(runs, sets, arm="repaired", base="released"):
    """Per run set and item, arm minus base; and pooled, each item averaged over run sets."""
    per_run = {}
    for name, trajectories in runs.items():
        a, b = hit_rates(trajectories, sets, arm), hit_rates(trajectories, sets, base)
        per_run[name] = {q: a[q] - b[q] for q in a if q in b}
    pooled = defaultdict(list)
    for per_item in per_run.values():
        for q, g in per_item.items():
            pooled[q].append(g)
    return per_run, {q: float(np.mean(v)) for q, v in pooled.items()}


def zero_ends(sums, level, lo, hi):
    """The interval's ends, with the point mass at 0 counted exactly. A resample that draws only
    capsules whose values sum to 0 has a mean of exactly 0, with probability (n_zero / n)^n. When
    every other capsule lies on one side of 0, 0 is the end of the bootstrap distribution on that
    side, and it is the percentile whenever that probability exceeds the tail; the Monte Carlo
    percentile misses it whenever the draws happen to hold fewer such resamples than the tail
    (Qwen3-235B-A22B's moved keys: 16 of 21 capsules sum to 0, a mass of 0.33% against a 0.26% tail)."""
    zero = np.isclose(sums, 0.0, rtol=0.0, atol=1e-12)
    if (float(np.mean(zero)) ** len(sums)) > (1.0 - level) / 2.0:
        if np.all(sums > -1e-12):
            lo = 0.0
        if np.all(sums < 1e-12):
            hi = 0.0
    return lo, hi


def calibrated(per_item, capsule, rng, outer, inner, final=10000):
    """Mean over items, and the percentile cluster bootstrap over capsules at the level the
    double bootstrap finds covers 95% on these capsules (claim_budget.covering_levels)."""
    values = defaultdict(list)
    for q, v in per_item.items():
        values[capsule[q]].append(v)
    if len(values) < 2:
        return None
    groups = [claim_budget.group(values)]
    levels, cov95 = claim_budget.covering_levels(groups, [0.95], outer, inner, rng)
    level = levels[0.95]
    capped = level is None
    level = float(claim_budget.LEVELS[-1]) if capped else level
    lo, hi = claim_budget.interval(claim_budget.draws(groups, final, rng), level)
    lo, hi = zero_ends(groups[0][0], level, lo, hi)
    return {"mean": 100.0 * claim_budget.estimate(groups), "lo": 100.0 * lo, "hi": 100.0 * hi,
            "level": level, "level_capped": capped, "coverage_of_nominal_95": cov95,
            "n_items": len(per_item), "n_clusters": len(values)}


def newly_credited(runs, sets, moved_items):
    """On moved keys, every trajectory the repair reads as right and the released set as wrong
    -- ``after > before``, as bracketing.mechanism counts it -- by distance and side."""
    out = {"gained": {name: 0 for name, _ in MISS_BINS}, "gained_open_side": 0,
           "gained_strict": 0, "n_moved": 0, "n_with_a_number": 0, "per_run": {}}
    for name, trajectories in runs.items():
        entry = {"gained": {b: 0 for b, _ in MISS_BINS}, "gained_open_side": 0}
        for q, answer in trajectories:
            if q not in moved_items:
                continue
            out["n_moved"] += 1
            if not answer or parse_number(answer) is None:
                continue
            out["n_with_a_number"] += 1
            o = sets[q]
            before, after = bw.nearest_is_key(answer, o["released"]), bw.nearest_is_key(answer, o["repaired"])
            if after > before:
                b = miss_bin(answer, o["released"][0])
                side = int(open_side(answer, o["repaired"]))
                for target in (out, entry):
                    target["gained"][b] += 1
                    target["gained_open_side"] += side
                out["gained_strict"] += int(before == 0 and after == 1)
        out["per_run"][name] = entry
    total = sum(out["gained"].values())
    out["n_gained"] = total
    out["share_over_25pct"] = (out["gained"]["25 to 100%"] + out["gained"]["over 100%"]) / total if total else None
    out["share_open_side"] = out["gained_open_side"] / total if total else None
    return out


def single_run_spread(runs, sets, items, capsule, rng, draws=2000):
    """The pooled moved-key gain a single run would have shown: one trajectory drawn per
    (run set, item), repeated."""
    by = {name: defaultdict(list) for name in runs}
    for name, trajectories in runs.items():
        for q, answer in trajectories:
            if q in items:
                o = sets[q]
                by[name][q].append(bw.nearest_is_key(answer, o["repaired"]) - bw.nearest_is_key(answer, o["released"]))
    out = []
    for _ in range(draws):
        pooled = defaultdict(list)
        for name in runs:
            for q, gs in by[name].items():
                pooled[q].append(gs[rng.integers(len(gs))])
        out.append(100.0 * float(np.mean([np.mean(v) for v in pooled.values()])))
    return {"central_95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "median": float(np.median(out)), "draws": draws}


def test_set(runs, sets, capsule, rng, heavy=True):
    """H1-H5 on one collection of run sets, pooled, and H1 per run set."""
    outer, inner = (4000, 20000) if heavy else (2000, 5000)
    groups = {g: {q for q, s in sets.items() if group_of(s) == g} for g in GROUPS}
    per_run, pooled = gains(runs, sets)
    _, pooled_placebo = gains(runs, {q: s for q, s in sets.items() if "placebo" in s}, "repaired", "placebo")
    res = {"n_items": {g: len(v) for g, v in groups.items()},
           "n_trajectories": {name: len(t) for name, t in runs.items()},
           # the runs the tests read: those on the numeric questions
           "n_numeric_runs": {name: sum(1 for q, _ in t if q in sets) for name, t in runs.items()}}
    for g in GROUPS:
        res[f"gain|{g}"] = calibrated({q: v for q, v in pooled.items() if q in groups[g]}, capsule, rng, outer, inner)
    res["repaired-placebo|moved to an edge"] = calibrated(
        {q: v for q, v in pooled_placebo.items() if q in groups["moved to an edge"]}, capsule, rng, outer, inner)
    res["per_run"] = {name: calibrated({q: v for q, v in g.items() if q in groups["moved to an edge"]},
                                       capsule, rng, 2000, 5000)
                      for name, g in per_run.items()}
    res["newly_credited"] = newly_credited(runs, sets, groups["moved to an edge"])
    h1, h2 = res["gain|moved to an edge"], res["gain|kept"]
    h4, h5 = res["repaired-placebo|moved to an edge"], res["gain|moved inward"]
    nc = res["newly_credited"]
    res["verdicts"] = {
        "H1": bool(h1 and h1["lo"] > 0),
        "H2": bool(h2 and (h2["lo"] <= 0 <= h2["hi"] or abs(h2["mean"]) < 5)),
        "H3": bool(nc["n_gained"] and nc["share_over_25pct"] > 0.5 and nc["share_open_side"] > 0.5),
        "H4": bool(h4 and h4["lo"] > 0),
        "H5": bool(h5 is not None and h5["mean"] <= 0),
    }
    return res


def published_reader_check(doc, sets):
    """How often the published MCQ reading of the same runs picked the key on the numeric items,
    beside the nearest-option reading of their submitted answers through the same options."""
    out = {}
    for run, mcq in MCQ_RUNS.items():
        reads = [r for r in doc["mcq_reads"].get(mcq, []) if r[0] in sets]
        mine = [bw.nearest_is_key(a, sets[q]["released"]) for q, a, _ in doc["open_runs"][run] if q in sets]
        out[run] = {"published_reader_right": 100.0 * float(np.mean([r[2] for r in reads])) if reads else None,
                    "n_published_reads": len(reads),
                    "nearest_option_right": 100.0 * float(np.mean(mine)) if mine else None,
                    "n_trajectories": len(mine)}
    return out


PACKED = ROOT / "results" / "agent_runs_replication"
PACK_FIELDS = ("model", "protocol", "served_as", "condition", "question_id", "rollout", "answer",
               "termination", "steps")


def d2_trajectories(root=ROOT / "build" / "agent_runs_replication", packed="answers"):
    """Every trajectory of the new v1.5 run sets: from where they were made, else from the
    compact copies that ship (``pack``), which keep what the analysis reads and no settings."""
    paths = sorted(root.glob("*/*/*.json"))
    if paths or packed is None:
        return [json.loads(p.read_text()) for p in paths]
    out = []
    for path in sorted(PACKED.glob(f"{packed}_*.jsonl.gz")):
        with gzip.open(path, "rt") as fh:
            out.extend(json.loads(line) for line in fh)
    return out


def turn_counts(t):
    """An episode's replies and how many of them ran to the reply-token limit, counted from its log
    when it is packed, so that the degenerate-run census (``degenerate_runs.flags_of``) sees a
    truncated reply in the copies that ship as in the full trajectories."""
    if t.get("log") is None:
        return {}
    finishes = [rec["finish"] for rec in t["log"] if rec.get("finish")]
    finishes = [(f[0] if f else None) if isinstance(f, (list, tuple)) else f for f in finishes]
    return {"turns": len(finishes), "truncated_turns": sum(f == "length" for f in finishes)}


def pack():
    by_run = defaultdict(list)
    for t in d2_trajectories():
        # only the keys the trajectory has: a text-protocol run records no protocol, and a
        # null one would read back as a run of its own (``bixbench_withdata.run_key``)
        by_run[bw.run_key(t)].append({**{k: t[k] for k in PACK_FIELDS if k in t}, **turn_counts(t)})
    PACKED.mkdir(exist_ok=True)
    for run, rows in sorted(by_run.items()):
        rows.sort(key=lambda r: (r["condition"], r["question_id"], r["rollout"]))
        path = PACKED / f"answers_{run}.jsonl.gz"
        with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=path.open("wb")) as fh:
            fh.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode())
        print(f"{run}: {len(rows)} trajectories to {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    # the episodes the driver's own wall clock cut, which were rerun from the start: their record,
    # and the cut runs themselves, compact as the others, so that no run ships cut and none is lost
    for record in sorted((ROOT / "build" / "agent_runs_replication").glob("reruns_*.json")):
        (PACKED / record.name).write_text(record.read_text())
        print(f"{record.name} to {PACKED.relative_to(ROOT) if PACKED.is_relative_to(ROOT) else PACKED}")
    cut = defaultdict(list)
    for t in d2_trajectories(ROOT / "build" / "agent_runs_superseded_replication", packed=None):
        cut[bw.run_key(t)].append({**{k: t[k] for k in PACK_FIELDS if k in t}, **turn_counts(t)})
    for run, rows in sorted(cut.items()):
        rows.sort(key=lambda r: (r["condition"], r["question_id"], r["rollout"]))
        path = PACKED / f"superseded_{run}.jsonl.gz"
        with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=path.open("wb")) as fh:
            fh.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode())
        print(f"{run}: {len(rows)} cut trajectories to {path.name}")


def d2_runs():
    """The v1.5 run sets made after the registration: {run: {condition: [(qid, answer)]}}."""
    out = defaultdict(lambda: defaultdict(list))
    for t in d2_trajectories():
        out[f"{bw.run_key(t)}|r{t['rollout']}"][t["condition"]].append((t["question_id"], t.get("answer")))
    return out


def not_registered(runs2, families, sets, capsule):
    """Beside the registered tests and not among them, to read a failed one by: how often each
    new run set's number is within 5% of the key on the numeric items (the with-data section's
    grader), and the placebo's own change on each key group, placebo - released, per family.
    Plain 95% cluster intervals over capsules (``bixbench_withdata.cluster_interval``, its own
    seed), so the registered draws above are untouched."""
    from answer_numbers import graded
    level = 0.95
    out = {"level": level, "within_5pct": {}, "placebo-released": {}}
    for name, conds in sorted(runs2.items()):
        for cond, trajectories in sorted(conds.items()):
            num = [(q, a) for q, a in trajectories if q in sets]
            hits = [float(bool(a) and bool(graded(a, sets[q]["released"][0], 0.05))) for q, a in num]
            out["within_5pct"][f"{name}|{cond}"] = bw.cluster_interval(hits, [capsule[q] for q, _ in num], level=level)
    with_placebo = {q: s for q, s in sets.items() if "placebo" in s}
    for fam, names in families.items():
        for cond in ("data", "nodata"):
            sub = {n: runs2[n][cond] for n in names if runs2[n].get(cond)}
            if not sub:
                continue
            _, pooled = gains(sub, with_placebo, "placebo", "released")
            for g in GROUPS:
                items = sorted(q for q in pooled if group_of(sets[q]) == g)
                out["placebo-released"][f"{fam}|{cond}|{g}"] = bw.cluster_interval(
                    [pooled[q] for q in items], [capsule[q] for q in items], level=level)
    return out


def analyse(args):
    rng = np.random.default_rng(SEED)
    report = {"preregistration": "PREREGISTRATION.md", "seed": SEED}
    doc = load_extract()
    sets = v10_sets()
    capsule = {q: e["capsule"] for q, e in doc["items"].items()}
    runs = {name: [(q, a) for q, a, _ in doc["open_runs"][name]] for name in OPEN_RUNS}
    report["D1"] = test_set(runs, sets, capsule, rng)
    report["D1"]["n_numeric_items"] = len(sets)
    report["D1"]["n_without_repair"] = sum("repaired" not in s for s in sets.values())
    report["D1"]["n_without_placebo"] = sum("placebo" not in s for s in sets.values())
    moved = {q for q, s in sets.items() if group_of(s) == "moved to an edge"}
    report["D1"]["single_run_spread"] = single_run_spread(runs, sets, moved, capsule, rng)
    report["D1"]["published_reader_check"] = published_reader_check(doc, sets)
    runs2 = d2_runs()
    if runs2:
        sets15 = bw.option_sets()
        items15 = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
        cap15 = {q: it["capsule_uuid"] for q, it in items15.items()}
        report["D2"] = {}
        families = {"qwen3-235b": [r for r in runs2 if r.startswith("qwen3-235b|")],
                    "reruns": [r for r in runs2 if not r.startswith("qwen3-235b|")]}
        for fam, names in families.items():
            if not names:
                continue
            for cond in ("data", "nodata"):
                sub = {n: runs2[n][cond] for n in names if runs2[n].get(cond)}
                if sub:
                    res = test_set(sub, sets15, cap15, rng, heavy=True)
                    if cond == "nodata":          # H6: the gain needs no data
                        h = res["gain|moved to an edge"]
                        res["verdicts"] = {"H6": bool(h and h["lo"] > 0)}
                    report["D2"][f"{fam}|{cond}"] = res
        # run to run: each new seed's moved-key gain beside the paper's r0 of the same run set
        import bracketing
        brk = json.loads((ROOT / "results" / "bracketing.json").read_text())
        moved15 = {q for q, st in sets15.items() if group_of(st) == "moved to an edge"}
        seeds = defaultdict(dict)
        for name, conds in runs2.items():
            model, rollout = name.split("|")
            for cond, trajectories in conds.items():
                per_run, _ = gains({name: trajectories}, sets15)
                g = [v for q, v in per_run[name].items() if q in moved15]
                seeds[f"{model}|{cond}"][rollout] = 100.0 * float(np.mean(g)) if g else None
                r0 = brk["runs"].get(model, {}).get(cond, {}).get("nearest", {}).get("repaired-released|moved to an edge")
                if r0:
                    seeds[f"{model}|{cond}"]["r0"] = r0["mean"]
        report["D2"]["run_to_run"] = seeds
        report["not_registered"] = not_registered(runs2, families, sets15, cap15)
    out = ROOT / "results" / "replication.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    for key in ["D1"] + [f"D2 {k}" for k in report.get("D2", {}) if k != "run_to_run"]:
        res = report["D1"] if key == "D1" else report["D2"][key[3:]]
        h1 = res["gain|moved to an edge"]
        print(f"{key}: moved {h1['mean']:+.1f} [{h1['lo']:+.1f},{h1['hi']:+.1f}] at {100 * h1['level']:.2f}% "
              f"({h1['n_items']} items, {h1['n_clusters']} capsules); verdicts {res['verdicts']}")
    print(f"wrote {out.relative_to(ROOT)}")


def _est(v):
    return f"${v['mean']:+.1f}$" if v else "--"


def _ci(v):
    # an interval whose level reached the calibration grid's top without covering 95% is not printed
    if v and v.get("level_capped"):
        return "{\\scriptsize$\\ddagger$}"
    return f"{{\\scriptsize$[{v['lo']:+.1f},{v['hi']:+.1f}]$}}" if v else ""


# each row's label, on its two lines: the estimates' and the intervals'
ROW_LABELS = {"D1": ("v1.0: gpt-4o,", "Claude 3.5$^{*}$"),
              "qwen3-235b|data": ("v1.5: Qwen3-235B-A22B", ""),
              "reruns|data": ("v1.5: new seeds,", "three agents")}
COLUMNS = ("gain|moved to an edge", "gain|kept", "gain|moved inward", "repaired-placebo|moved to an edge")


def table_rows(report):
    """tab:replication's rows (no longer in the paper), two lines per data set: estimates, then intervals."""
    rows = []
    entries = [("D1", report["D1"])] + [(k, v) for k, v in report.get("D2", {}).items() if k.endswith("|data")]
    for key, res in entries:
        nc = res["newly_credited"]
        count = f"{sum(res['n_numeric_runs'].values()):,}".replace(",", "{,}")
        first, second = ROW_LABELS[key]
        rows.append(f"{first} & ${count}$ & " + " & ".join(_est(res[c]) for c in COLUMNS)
                    + f" & ${nc['n_gained']}$ & ${100 * nc['share_over_25pct']:.0f}$ & ${100 * nc['share_open_side']:.0f}$\\\\")
        rows.append(f"{second} & & " + " & ".join(_ci(res[c]) for c in COLUMNS) + " & & & \\\\")
    return rows


# tab:predictions: each registered prediction as PREREGISTRATION.md words its pass rule, and every
# data set's result under it. H2's registered rule passes on a mean within 5 points of zero; its last
# row reads the same change as an equivalence claim, the whole interval inside +-5 points.
PREDICTIONS = (
    ("H1", "moved keys gain", "the interval's lower end is above 0", "gain|moved to an edge"),
    ("H2", "unchanged keys do not", "an interval that contains 0, or a mean within 5 points of 0", "gain|kept"),
    ("H3", "newly accepted answers are far and on the open side",
     "more than half are more than 25\\% from the key, and more than half lie beyond the key on the side "
     "without a distractor", None),
    ("H4", "not the redrawing", "$U-P$ on moved keys: the interval's lower end is above 0",
     "repaired-placebo|moved to an edge"),
    ("H5", "inward keys do not gain", "a mean of at most 0", "gain|moved inward"),
    ("H6", "the gain needs no data", "without the data, the moved keys' interval's lower end is above 0", None),
)


def prediction_rows(report):
    """tab:predictions' rows: estimate [interval] and verdict per data set, in the order D1, the
    new seeds, Qwen3-235B-A22B."""
    sets = [("D1", report["D1"], None), ("reruns", report["D2"]["reruns|data"], report["D2"]["reruns|nodata"]),
            ("qwen3-235b", report["D2"]["qwen3-235b|data"], report["D2"]["qwen3-235b|nodata"])]

    def verdict(ok, margin_only=False):
        # H2 passes on "a mean within 5 points of 0" when its interval does not contain 0: the paper counts
        # that pass as a failure of the prediction as a test, and marks it
        return ("\\textbf{fail}$^{\\S}$" if margin_only else "pass") if ok else "\\textbf{fail}"

    def cell(v, ok, margin_only=False):
        # an interval whose level reached the calibration grid's top without covering 95% is not printed
        if v.get("level_capped"):
            return f"${v['mean']:+.1f}$$^{{\\ddagger}}$ {verdict(ok)}"
        return f"${v['mean']:+.1f}$ {{\\scriptsize$[{v['lo']:+.1f},{v['hi']:+.1f}]$}} {verdict(ok, margin_only)}"

    rows = []
    for h, what, rule, key in PREDICTIONS:
        cells = []
        for name, res, nodata in sets:
            if h == "H3":
                nc = res["newly_credited"]
                cells.append(f"${100 * nc['share_over_25pct']:.0f}$, ${100 * nc['share_open_side']:.0f}\\%$ "
                             f"of ${nc['n_gained']}$ {verdict(res['verdicts']['H3'])}")
            elif h == "H6":
                cells.append("--" if nodata is None else cell(nodata["gain|moved to an edge"], nodata["verdicts"]["H6"]))
            else:
                v = res[key]
                margin_only = h == "H2" and not v.get("level_capped") and not (v["lo"] <= 0 <= v["hi"])
                cells.append(cell(v, res["verdicts"][h], margin_only))
        rows.append(f"{h}: {what} & {rule} & " + " & ".join(cells) + "\\\\")
    eq = []
    for name, res, _ in sets:
        v = res["gain|kept"]
        eq.append("shown" if -5 < v["lo"] and v["hi"] < 5 else f"not shown (upper end ${v['hi']:+.1f}$)")
    rows.append("H2 as equivalence & the whole interval within $\\pm5$ points & " + " & ".join(eq) + "\\\\")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--eval-df", default=str(ROOT / "build" / "external" / "bixbench_v10_trajectories" / "eval_df.csv"))
    sub.add_parser("items", help="rewrite the v1.0 question files from the shipped extract")
    sub.add_parser("analyse")
    sub.add_parser("latex", help="print tab:replication's rows (no longer in the paper) from results/replication.json")
    sub.add_parser("pack", help="compact copies of the new v1.5 run sets' answers, to ship")
    args = ap.parse_args()
    if args.cmd == "pack":
        pack()
    elif args.cmd == "latex":
        for line in table_rows(json.loads((ROOT / "results" / "replication.json").read_text())):
            print(line)
    elif args.cmd == "build":
        build(args)
    elif args.cmd == "items":
        write_items(load_extract())
    else:
        analyse(args)


if __name__ == "__main__":
    main()
