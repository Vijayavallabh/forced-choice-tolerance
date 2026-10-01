#!/usr/bin/env python3
r"""Why the released options read lowest with the data: the key is bracketed.

A distractor catches a near miss only if it sits on the side the miss fell. A
key with distractors on both sides -- one not at an extreme of its four values
-- is *bracketed*, and BixBench v1.5's numeric keys are bracketed on $80.9\%$ of
items (second-smallest $51.4\%$, third $29.5\%$). The same fact is the rank
channel: at $k$ options a key bracketed on a share $\beta$ of items puts at least
$\beta/(k-2)$ of them on one middle rank, so a rule taking that rank gains at
least $\beta/(k-2)-1/k$. Making the rank uniform caps $\beta$ at $(k-2)/k$ -- a
half, at four options -- so on the other half the key sits at an edge and a miss
on its open side is nearer the key than any distractor.

This reads that prediction off the frozen with-data runs, where only the option
set changes: for each run set and option set, the nearest-option rule and the
primary reader's forced read, split by whether the key is bracketed in that set,
and the released-to-repaired gain split by whether the repair moved the key to
an edge. Two consequences are also counted: runs BixBench's own tolerance credits
that the released options read as wrong, and whether reading the same runs
through the released options reorders the agents.

The two groups of the split span fewer capsules than the file (21 and 39 of 46),
so each is read at the level that covers 95% on its own shape, measured the way
claim_budget.py measures it: a beta-binomial fitted to the released options'
nearest-option hits per capsule, averaged over the run sets, and files simulated
at the group's own capsule sizes. The whole-file intervals keep the file's level.

    python3 bracketing.py
"""
import argparse
import gzip
import json
import pathlib
from itertools import combinations

import numpy as np

import bixbench_withdata as bw
from option_artifacts import parse_number

RUNS = ["qwen72b", "llama70b", "gemma27b", "qwen72b-react", "llama70b-react", "glm45air-react", "qwen3a3b-react"]
LABEL = {"qwen72b": "Qwen2.5-72B, text", "llama70b": "Llama-3.3-70B, text", "gemma27b": "gemma-3-27b, text",
         "qwen72b-react": "Qwen2.5-72B, published", "llama70b-react": "Llama-3.3-70B, published",
         "glm45air-react": "GLM-4.5-Air, published", "qwen3a3b-react": "Qwen3-30B-A3B, published"}
ARMS = ("released", "placebo", "repaired")


def primary(run):
    return bw.PRIMARY_READER.get(run, run.replace("-react", ""))


def rows_of(run):
    return json.load(gzip.open(f"results/agent_runs/withdata_rows_{run}.json.gz", "rt"))


def read_score(row, reader, arm):
    reads = row["reads"].get(f"{reader}|{arm}|forced")
    if reads is None:
        return 0.0 if not row["answer"] else None
    return float(np.mean([x["correct"] for x in reads]))


def bracketed(rank, k=4):
    return rank is not None and 0 < rank < k - 1


def ci(values, clusters, level=None):
    return bw.cluster_interval(values, clusters, level=level)


def moved(row):
    """The repair moved this item's key from a bracketed rank to an edge."""
    return bracketed(row["key_rank"]["released"]) and not bracketed(row["key_rank"]["repaired"])


def moved_inward(row):
    """The repair moved this item's key from an edge to a bracketed rank."""
    return not bracketed(row["key_rank"]["released"]) and bracketed(row["key_rank"]["repaired"])


def kept(row):
    return bracketed(row["key_rank"]["released"]) == bracketed(row["key_rank"]["repaired"])


# The three ways the repair can move a key, each a group with its own capsules.
GROUPS = {"moved to an edge": moved, "not moved": lambda r: not moved(r),
          "moved inward": moved_inward, "rank class kept": kept}


def split_levels(sets, reps=4000, inner=2000, seed=20260923, groups=("moved to an edge", "not moved")):
    """The nominal level that covers 95% on each group's own capsules, with the data."""
    from collections import defaultdict

    import claim_budget
    rng = np.random.default_rng(seed)
    out = {}
    for group in groups:
        want = GROUPS[group]
        hits, capsule = defaultdict(list), {}
        for run in RUNS:
            for r in rows_of(run):
                if r["numeric"] and r["condition"] == "data" and want(r):
                    capsule[r["question_id"]] = r["capsule"]
                    hits[r["question_id"]].append(bw.nearest_is_key(r["answer"], sets[r["question_id"]]["released"]))
        shape = defaultdict(list)
        for q in sorted(hits):
            shape[capsule[q]].append(float(np.mean(hits[q])))
        levels, coverage = claim_budget.parametric_levels(claim_budget.fit(shape), None, [0.95], reps, inner, rng)
        out[group] = {"level": levels[0.95], "coverage_of_nominal_95": coverage,
                      "n_items": len(hits), "n_clusters": len(shape)}
    return out


# How far a newly credited answer is from the key, relative to the key.
MISS_BINS = (("within 5%", 0.05), ("5 to 25%", 0.25), ("25 to 100%", 1.0), ("over 100%", float("inf")))


def miss_bin(answer, key):
    a, y = parse_number(answer), parse_number(key)
    miss = (0.0 if a == y else float("inf")) if y == 0 else abs(a - y) / abs(y)
    return next(name for name, top in MISS_BINS if miss <= top)


def open_side(answer, options):
    """The answer lies beyond an edge key, on the side with no distractor."""
    a, values = parse_number(answer), [parse_number(o) for o in options]
    return (values[0] == max(values) and a > values[0]) or (values[0] == min(values) and a < values[0])


def mechanism(sets, report):
    """What the repair credits, what the placebo already moves, and how near the distractors sit.

    * On the keys the repair moved to an edge, each answer the nearest-option rule reads as
      right through the repaired options and wrong through the released ones, binned by how far
      it is from the key and by whether it lies on the key's open side -- with the data and
      without it.
    * Repaired minus placebo, which share the redraw and differ only in the key's rank, split
      by how the repair moved the key: to an edge, inward from an edge, or not across the
      bracketed/edge line.
    * The nearest distractor's distance from the key in each option set, in the rule's own
      metric, and a format rule that reads no value: take the option written with the fewest
      significant digits, ties split evenly.
    """
    report["miss_distance"], totals = {}, {}
    for cond in ("data", "nodata"):
        totals[cond] = {"gained": {name: 0 for name, _ in MISS_BINS}, "lost": {name: 0 for name, _ in MISS_BINS},
                        "gained_open_side": 0}
    for run in RUNS:
        report["miss_distance"][run] = {}
        for cond in ("data", "nodata"):
            entry = {"gained": {name: 0 for name, _ in MISS_BINS}, "lost": {name: 0 for name, _ in MISS_BINS},
                     "gained_open_side": 0, "n_moved": 0, "n_with_a_number": 0}
            for r in rows_of(run):
                if not (r["numeric"] and r["condition"] == cond and moved(r)):
                    continue
                entry["n_moved"] += 1
                o = sets[r["question_id"]]
                if not r["answer"] or parse_number(r["answer"]) is None:
                    continue
                entry["n_with_a_number"] += 1
                before, after = bw.nearest_is_key(r["answer"], o["released"]), bw.nearest_is_key(r["answer"], o["repaired"])
                if after > before:
                    entry["gained"][miss_bin(r["answer"], o["released"][0])] += 1
                    entry["gained_open_side"] += int(open_side(r["answer"], o["repaired"]))
                elif after < before:
                    entry["lost"][miss_bin(r["answer"], o["released"][0])] += 1
            report["miss_distance"][run][cond] = entry
            for side in ("gained", "lost"):
                for name in entry[side]:
                    totals[cond][side][name] += entry[side][name]
            totals[cond]["gained_open_side"] += entry["gained_open_side"]
    report["miss_distance_total"] = totals
    for cond, t in totals.items():
        print(f"newly credited on moved keys, {cond}: {t['gained']} ({sum(t['gained'].values())}, "
              f"{t['gained_open_side']} on the open side); newly refused {t['lost']}")

    levels = report["levels"]
    report["placebo_contrast"] = {}
    for run in RUNS:
        report["placebo_contrast"][run] = {}
        for cond in ("data", "nodata"):
            rs = [r for r in rows_of(run) if r["numeric"] and r["condition"] == cond]
            block = {}
            for group in ("moved to an edge", "moved inward", "rank class kept"):
                part = [r for r in rs if GROUPS[group](r)]
                diff = [bw.nearest_is_key(r["answer"], sets[r["question_id"]]["repaired"])
                        - bw.nearest_is_key(r["answer"], sets[r["question_id"]]["placebo"]) for r in part]
                block[group] = ci(diff, [r["capsule"] for r in part], levels[group]["level"])
            report["placebo_contrast"][run][cond] = block
        b = report["placebo_contrast"][run]["data"]
        print(f"{LABEL[run]:26s} repaired - placebo, with the data: to an edge {b['moved to an edge']['mean']:+5.1f}, "
              f"inward {b['moved inward']['mean']:+5.1f}, kept {b['rank class kept']['mean']:+5.1f}")

    def sig(text):
        t = str(text).strip().rstrip("%").replace(",", "").lower()
        mantissa = t.split("e")[0].lstrip("+-").replace(".", "").lstrip("0").rstrip("0")
        return max(len(mantissa), 1)

    # the format rule's margin over the quarter it scores where the key's format tells nothing,
    # over capsules at the whole file's level
    capsule = {r["question_id"]: r["capsule"] for run in RUNS for r in rows_of(run)}
    report["proximity"], report["format_rule"], report["format_rule_over_chance"] = {}, {}, {}
    for arm in ARMS:
        near = []
        credit = []
        for o in sets.values():
            values = [parse_number(x) for x in o[arm]]
            near.append(min(abs(v - values[0]) / (abs(v) + abs(values[0])) if abs(v) + abs(values[0]) else 0.0
                            for v in values[1:]))
            digits = [sig(x) for x in o[arm]]
            tied = [i for i, d in enumerate(digits) if d == min(digits)]
            credit.append(1.0 / len(tied) if 0 in tied else 0.0)
        report["proximity"][arm] = {"median": float(np.median(near)),
                                    "quartiles": [float(x) for x in np.quantile(near, [0.25, 0.75])]}
        report["format_rule"][arm] = 100.0 * float(np.mean(credit))
        report["format_rule_over_chance"][arm] = ci([c - 0.25 for c in credit], [capsule[q] for q in sets],
                                                    levels["whole file"])
    report["non_positive_keys"] = sum(1 for o in sets.values() if parse_number(o["released"][0]) <= 0)
    print("nearest distractor, median distance: " + ", ".join(
        f"{a} {report['proximity'][a]['median']:.3f}" for a in ARMS)
          + "; fewest-digits rule: " + ", ".join(f"{a} {report['format_rule'][a]:.1f}%" for a in ARMS))


PROTOCOL = {"qwen72b": "text", "llama70b": "text", "gemma27b": "text", "qwen72b-react": "published",
            "llama70b-react": "published", "glm45air-react": "published", "qwen3a3b-react": "published"}
NAME = {"qwen72b": "Qwen2.5-72B", "llama70b": "Llama-3.3-70B", "gemma27b": "gemma-3-27b",
        "qwen72b-react": "Qwen2.5-72B", "llama70b-react": "Llama-3.3-70B", "glm45air-react": "GLM-4.5-Air",
        "qwen3a3b-react": "Qwen3-30B-A3B$^\\dagger$"}


def _ci(x):
    return f"${x['mean']:+.1f}$ $[{x['lo']:+.1f},{x['hi']:+.1f}]$"


def table_rows(report, summary="results/bixbench_withdata.json"):
    """tab:bracket: the run set's own reading through each option set, and the nearest-option
    rule's repaired-minus-released gain overall and split by whether the repair moved the key,
    under a header row per protocol."""
    wd = json.loads(pathlib.Path(summary).read_text())
    header = {"text": "\\multicolumn{7}{l}{\\emph{tools called in text}}\\\\",
              "published": "\\multicolumn{7}{l}{\\emph{as published: BixBench's ReAct agent}}\\\\"}
    rows, seen = [], set()
    for run in RUNS:
        if PROTOCOL[run] not in seen:
            if seen:
                rows.append("\\addlinespace")
            rows.append(header[PROTOCOL[run]])
            seen.add(PROTOCOL[run])
        read = wd[f"{run}|data"][f"reader:{primary(run)}"]["forced"]["numeric"]
        near = wd[f"{run}|data"]["nearest"]["released-repaired"]
        gain = {"mean": -near["mean"], "lo": -near["hi"], "hi": -near["lo"]}
        split = report["runs"][run]["data"]["nearest"]
        rows.append(f"{NAME[run]} & ${read['released']['mean']:.1f}$ & "
                    f"${read['placebo']['mean']:.1f}$ & ${read['repaired']['mean']:.1f}$ & {_ci(gain)} & "
                    f"{_ci(split['repaired-released|moved to an edge'])} & "
                    f"{_ci(split['repaired-released|not moved'])}\\\\")
    return rows


def mechanism_rows(report):
    """tab:mechanism: what the repair newly credits on the moved keys, and what it does against
    the placebo, per run set."""
    header = {"text": "\\multicolumn{10}{l}{\\emph{tools called in text}}\\\\",
              "published": "\\multicolumn{10}{l}{\\emph{as published: BixBench's ReAct agent}}\\\\"}
    rows, seen = [], set()
    for run in RUNS:
        if PROTOCOL[run] not in seen:
            if seen:
                rows.append("\\addlinespace")
            rows.append(header[PROTOCOL[run]])
            seen.add(PROTOCOL[run])
        md = report["miss_distance"][run]
        gained = md["data"]["gained"]
        pc = report["placebo_contrast"][run]["data"]
        rows.append(f"{NAME[run]} & " + " & ".join(f"${gained[name]}$" for name, _ in MISS_BINS)
                    + f" & ${md['data']['gained_open_side']}$ & ${sum(md['nodata']['gained'].values())}$ & "
                    + f"{_ci(pc['moved to an edge'])} & ${pc['moved inward']['mean']:+.1f}$ & "
                    + f"${pc['rank class kept']['mean']:+.1f}$\\\\")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="results/bracketing.json")
    ap.add_argument("--latex", action="store_true", help="print tab:bracket's rows from the two result files")
    args = ap.parse_args()
    if args.latex:
        report = json.loads(pathlib.Path(args.output).read_text())
        for line in table_rows(report) + [""] + mechanism_rows(report):
            print(line)
        return
    sets = bw.option_sets()
    report = {"share_bracketed": {}, "runs": {}, "credited_but_read_wrong": {}, "agent_pairs": {}}
    report["levels"] = {"whole file": bw.LEVEL, **split_levels(sets, groups=tuple(GROUPS))}
    print("levels: " + ", ".join(f"{g} {v if isinstance(v, float) else v['level']:.4f}"
                                  for g, v in report["levels"].items()))

    # how often each option set brackets the key, over the 105 numeric items
    for arm in ARMS:
        ranks = [bw.pick_rank(o[arm][0], o[arm]) for o in sets.values()]
        report["share_bracketed"][arm] = 100.0 * float(np.mean([bracketed(r) for r in ranks]))
    beta = report["share_bracketed"]["released"] / 100.0
    report["rank_channel_floor_from_bracketing"] = 100.0 * (beta / 2 - 0.25)
    print("key bracketed: " + ", ".join(f"{a} {v:.1f}%" for a, v in report["share_bracketed"].items())
          + f"; floor on the rank rule from bracketing alone {report['rank_channel_floor_from_bracketing']:+.1f}")

    per_run = {}
    for run in RUNS:
        rows = [r for r in rows_of(run) if r["numeric"]]
        per_run[run] = rows
        reader = primary(run)
        report["runs"][run] = {}
        for cond in ("data", "nodata"):
            rs = [r for r in rows if r["condition"] == cond]
            clusters = [r["capsule"] for r in rs]
            out = {}
            for scorer in ("nearest", "reader"):
                def s(r, arm):
                    if scorer == "nearest":
                        return bw.nearest_is_key(r["answer"], sets[r["question_id"]][arm])
                    return read_score(r, reader, arm)
                block = {}
                for arm in ARMS:
                    for want in (True, False):
                        part = [(s(r, arm), r["capsule"]) for r in rs
                                if bracketed(r["key_rank"][arm]) == want]
                        block[f"{arm}|{'bracketed' if want else 'edge'}"] = {
                            "n": len(part), "mean": 100.0 * float(np.mean([v for v, _ in part if v is not None]))
                            if part else None}
                # the released-to-repaired gain, on items the repair moved to an edge and on the rest
                for group, want in (("moved to an edge", True), ("not moved", False)):
                    part = [r for r in rs if moved(r) == want]
                    diff = [None if (a := s(r, "repaired")) is None or (b := s(r, "released")) is None else a - b
                            for r in part]
                    block[f"repaired-released|{group}"] = ci(
                        diff, [r["capsule"] for r in part], report["levels"][group]["level"])
                out[scorer] = block
            report["runs"][run][cond] = out
            n, rd = out["nearest"], out["reader"]
            print(f"{LABEL[run]:26s} {cond:6s} nearest: gain on moved "
                  f"{n['repaired-released|moved to an edge']['mean']:+5.1f} (n={n['repaired-released|moved to an edge']['n_items']}), "
                  f"on the rest {n['repaired-released|not moved']['mean']:+5.1f} | reader: moved "
                  f"{rd['repaired-released|moved to an edge']['mean']:+5.1f}, rest {rd['repaired-released|not moved']['mean']:+5.1f}")

        # runs BixBench's own tolerance credits, read through each option set
        rs = [r for r in rows if r["condition"] == "data" and r["within_5pct"]]
        entry = {"n_within_5pct": len(rs)}
        for arm in ARMS:
            entry[f"nearest_wrong|{arm}"] = sum(1 for r in rs if not bw.nearest_is_key(r["answer"], sets[r["question_id"]][arm]))
            scores = [read_score(r, reader, arm) for r in rs]
            entry[f"reader_mean|{arm}"] = 100.0 * float(np.mean([v for v in scores if v is not None])) if rs else None
        report["credited_but_read_wrong"][run] = entry
        print(f"   within 5% with the data: {len(rs)} runs; nearest released option not the key on "
              f"{entry['nearest_wrong|released']}; {reader} reads them right {entry['reader_mean|released']:.1f}% "
              f"released, {entry['reader_mean|repaired']:.1f}% repaired" if rs else "   no run within 5%")

    # does reading through the released options reorder agents? paired over the 105 items
    for group in (("qwen72b", "llama70b", "gemma27b"), ("qwen72b-react", "llama70b-react", "glm45air-react", "qwen3a3b-react")):
        for a, b in combinations(group, 2):
            ra = {r["question_id"]: r for r in per_run[a] if r["condition"] == "data"}
            rb = {r["question_id"]: r for r in per_run[b] if r["condition"] == "data"}
            qids = sorted(set(ra) & set(rb))
            clusters = [ra[q]["capsule"] for q in qids]
            res = {}
            res["within_5pct"] = ci([float(bool(ra[q]["within_5pct"])) - float(bool(rb[q]["within_5pct"])) for q in qids], clusters)
            for arm in ("released", "repaired"):
                res[f"nearest|{arm}"] = ci([bw.nearest_is_key(ra[q]["answer"], sets[q][arm])
                                            - bw.nearest_is_key(rb[q]["answer"], sets[q][arm]) for q in qids], clusters)
                sa = [read_score(ra[q], "gemma27b", arm) for q in qids]
                sb = [read_score(rb[q], "gemma27b", arm) for q in qids]
                res[f"gemma27b|{arm}"] = ci([None if x is None or y is None else x - y for x, y in zip(sa, sb)], clusters)
                sa = [read_score(ra[q], primary(a), arm) for q in qids]
                sb = [read_score(rb[q], primary(b), arm) for q in qids]
                res[f"own|{arm}"] = ci([None if x is None or y is None else x - y for x, y in zip(sa, sb)], clusters)
            report["agent_pairs"][f"{a} - {b}"] = res
            f = lambda x: f"{x['mean']:+5.1f} [{x['lo']:+5.1f},{x['hi']:+5.1f}]"
            print(f"{LABEL[a]:26s} - {LABEL[b]:26s} within5 {f(res['within_5pct'])}  own-released {f(res['own|released'])}  "
                  f"gemma-released {f(res['gemma27b|released'])}  nearest-released {f(res['nearest|released'])}  "
                  f"gemma-repaired {f(res['gemma27b|repaired'])}")
    mechanism(sets, report)
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
