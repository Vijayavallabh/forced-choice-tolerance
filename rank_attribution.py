#!/usr/bin/env python3
r"""Does a no-data forced-choice score follow the key's rank?

BixBench v1.5's options hand a rule that never reads the question $+26.4$
points: the key is the second-smallest of four values on $51.4\%$ of its $105$
numeric items. That is what the channel is *worth*. Whether a given score
*collected* it is a separate question, and the answer is in the runs:

* **BixBench's published v1.5 runs** (gpt-4o, Claude 3.5 Sonnet) record the
  letter picked and the letter keyed, not the order shown, so the pick's rank is
  lost and the key's is not. A reader that drew its margin from the rank would be
  more accurate where the key is second-smallest than where it is not, and would
  have no margin on questions whose options are not numbers. Both are read off.
* **v1.0's published runs** record the order shown (``choices``), so the rank of
  every pick is read directly.
* **Our six open models, question shown and data absent** (``free_response.py``'s
  forced arm) on the released options, and on the same items with the options
  rewritten two ways: a placebo that redraws every distractor and keeps the key's
  rank, and a repair that draws the rank uniformly. Letter orderings are the same
  across the three, so released $-$ repaired is paired rollout by rollout.
* **Thirteen open models reading the options with the question withheld**, under
  BixBench's own template (``scale_grid.py``'s dumps): which rank they pick.

Every interval is a cluster bootstrap over capsules at the level that covers
$95\%$ on its own shape, measured as ``claim_budget.py`` measures it; paired
differences on the $105$ numeric items keep the file's $96.0\%$.

    python3 rank_attribution.py
    python3 rank_attribution.py --latex     # tab:attribution's rows
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import gzip
import json
import pathlib
import re
from collections import Counter, defaultdict

import numpy as np

import arm_intervals
import bixbench_withdata as bw
import claim_budget
import published_repair as pr
from option_artifacts import parse_number

K = 4
CHANCE = 1.0 / K
OPTION_LINE = re.compile(r"^\(([A-J])\)\s*(.*)$")
MODELS = {"gpt-4o": "gpt-4o", "claude-3-5-sonnet-latest": "Claude 3.5 Sonnet"}
SHORT = {"gpt-4o": "gpt-4o", "claude-3-5-sonnet-latest": "Claude 3.5"}      # tab:attribution's width
OPEN = {"Qwen2_5-1_5B": "Qwen2.5-1.5B", "Llama-3_2-3B": "Llama-3.2-3B", "Qwen2_5-7B": "Qwen2.5-7B",
        "Meta-Llama-3_1-8B": "Llama-3.1-8B", "phi-4": "phi-4", "Qwen2_5-14B": "Qwen2.5-14B"}
WITHHELD = {"qwen1_5b": "Qwen2.5-1.5B", "llama1b": "Llama-3.2-1B", "llama3b": "Llama-3.2-3B",
            "phi35mini": "Phi-3.5-mini", "gemma4b": "gemma-3-4b", "olmo7b": "OLMo-2-1124-7B",
            "qwen7b": "Qwen2.5-7B", "llama8b": "Llama-3.1-8B", "phi4": "phi-4", "qwen14b": "Qwen2.5-14B",
            "qwen32b": "Qwen2.5-32B", "llama70b": "Llama-3.3-70B", "qwen72b": "Qwen2.5-72B"}
# free_response.py's rule: a forced cell is withheld when more than this share does not parse
UNPARSED_LIMIT = 5.0


def ranks_of(values):
    """Sorted rank of each position, or None unless all are distinct numbers."""
    if len(values) != K or any(v is None for v in values) or len(set(values)) < K:
        return None
    order = sorted(range(K), key=lambda i: values[i])
    return {src: rank for rank, src in enumerate(order)}


def covering_level(first, second=None, reps=2000, inner=2000, seed=20260924):
    """The nominal level that covers 95% for a statistic on these clusters' shape.

    ``first`` (and ``second``, for a difference of two disjoint groups) map a
    capsule to its items' 0/1 outcomes; the shape is fitted and simulated as
    ``claim_budget.parametric_levels`` does for the abstract's claims.
    """
    rng = np.random.default_rng(seed)
    shape = claim_budget.fit(first)
    other = claim_budget.fit(second) if second is not None else None
    levels, coverage = claim_budget.parametric_levels(shape, other, [0.95], reps, inner, rng)
    return {"level": levels[0.95], "coverage_of_nominal_95": coverage,
            "n_clusters": len(first) if second is None else [len(first), len(second)]}


def interval(values, clusters, level):
    return bw.cluster_interval(values, clusters, level=level)


def difference(a, b, level, reps=10000, seed=20260923):
    """Mean of group a minus mean of group b, capsules resampled jointly, in points."""
    groups = defaultdict(lambda: [[], []])
    for v, c in a:
        groups[c][0].append(v)
    for v, c in b:
        groups[c][1].append(v)
    keys = sorted(groups)
    sa = np.array([sum(groups[k][0]) for k in keys])
    na = np.array([len(groups[k][0]) for k in keys], dtype=float)
    sb = np.array([sum(groups[k][1]) for k in keys])
    nb = np.array([len(groups[k][1]) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(reps, len(keys)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = sa[idx].sum(1) / na[idx].sum(1) - sb[idx].sum(1) / nb[idx].sum(1)
    d = d[np.isfinite(d)]
    lo, hi = np.quantile(d, [(1 - level) / 2, (1 + level) / 2])
    point = sa.sum() / na.sum() - sb.sum() / nb.sum()
    return {"mean": round(100 * float(point), 2), "lo": round(100 * float(lo), 2),
            "hi": round(100 * float(hi), 2), "n_items": [int(na.sum()), int(nb.sum())],
            "n_clusters": len(keys)}


def by_cluster(pairs):
    out = defaultdict(list)
    for v, c in pairs:
        out[c].append(v)
    return out


def rank_reader_prediction(key_shares, accuracy):
    """What a reader drawing its whole numeric accuracy from a second-smallest preference
    would score on second-smallest keys and on the rest (the other three ranks alike)."""
    p2 = key_shares[1]
    # accuracy = p2*b2 + (1-p2)*(1-b2)/3  ->  b2
    b2 = (accuracy - (1 - p2) / 3) / (p2 - (1 - p2) / 3)
    b2 = min(max(b2, 0.0), 1.0)
    return {"b_second_smallest": b2, "accuracy_on_second_smallest": 100 * b2,
            "accuracy_elsewhere": 100 * (1 - b2) / 3, "difference": 100 * (b2 - (1 - b2) / 3)}


# ---------------------------------------------------------------- published runs

def published_v15(model):
    rows = pr.read(f"data/external/zero_shot_v15/{model}-grader-mcq-refusal-False.csv")
    items = {}
    for line in open("data/bixbench.jsonl", encoding="utf-8"):
        row = json.loads(line)
        items[row["question_id"].lower()] = row
    out = []
    for r in rows:
        it = items[r["uuid"].strip().lower()]
        ranks = ranks_of([parse_number(o) for o in [it["ideal"], *it["distractors"]]])
        out.append({"capsule": it["capsule_uuid"], "correct": 1.0 if pr.truthy(r["correct"]) else 0.0,
                    "key_rank": None if ranks is None else ranks[0]})
    return out


def published_v10(model):
    path = glob.glob(f"data/external/zero_shot_v10/*refusal_False_mcq_{model}_1.0.csv")[0]
    out = []
    for r in pr.read(path):
        choices = ast.literal_eval(r["choices"])
        values = [parse_number((c.split(") ", 1)[1] if ") " in c else c).strip()) for c in choices]
        ranks = ranks_of(values)
        letters = "ABCD"
        row = {"capsule": r["uuid"], "correct": 1.0 if pr.truthy(r["correct"]) else 0.0,
               "key_rank": None, "pick_rank": None}
        if ranks is not None and r["target"] in letters and r["predicted"] in letters:
            row["key_rank"] = ranks[letters.index(r["target"])]
            row["pick_rank"] = ranks[letters.index(r["predicted"])]
        out.append(row)
    return out


def summarise_published(rows, calibrate):
    numeric = [r for r in rows if r["key_rank"] is not None]
    other = [r for r in rows if r["key_rank"] is None]
    report = {"n_numeric": len(numeric), "n_other": len(other)}
    for name, part in (("numeric", numeric), ("other", other)):
        hits = by_cluster((r["correct"], r["capsule"]) for r in part)
        level = calibrate(hits) if part else None
        report[f"margin_{name}"] = interval([r["correct"] - CHANCE for r in part], [r["capsule"] for r in part],
                                            level["level"]) if part else None
        if part:
            report[f"margin_{name}"]["level"] = level
    counts = Counter(r["key_rank"] for r in numeric)
    report["key_rank_shares"] = [100 * counts[j] / len(numeric) for j in range(K)]
    report["accuracy_by_key_rank"] = {str(j): {"n": counts[j], "accuracy": 100 * float(np.mean(
        [r["correct"] for r in numeric if r["key_rank"] == j])) if counts[j] else None} for j in range(K)}
    second = [(r["correct"], r["capsule"]) for r in numeric if r["key_rank"] == 1]
    rest = [(r["correct"], r["capsule"]) for r in numeric if r["key_rank"] != 1]
    level = calibrate(by_cluster(second), by_cluster(rest))
    report["second_smallest_minus_rest"] = difference(second, rest, level["level"])
    report["second_smallest_minus_rest"]["level"] = level
    # the test is only about the rank this file leaks: where the second-smallest is the modal rank
    if max(range(K), key=lambda j: report["key_rank_shares"][j]) == 1:
        report["rank_reader_would_score"] = rank_reader_prediction(
            [s / 100 for s in report["key_rank_shares"]], float(np.mean([r["correct"] for r in numeric])))
    picks = [r["pick_rank"] for r in numeric if r.get("pick_rank") is not None]
    if picks:
        pc = Counter(picks)
        report["pick_rank_shares"] = [100 * pc[j] / len(picks) for j in range(K)]
    return report


# ---------------------------------------------------------------- our runs

def parse_options(user):
    shown = {}
    for line in str(user).split("\n"):
        m = OPTION_LINE.match(line.strip())
        if m:
            shown[m.group(1)] = m.group(2).strip()
    return shown


def dump_rows(path, arm):
    """(item, draw, capsule, key rank, pick rank, correct, parsed) per forced rollout."""
    out = []
    for line in (gzip.open if str(path).endswith(".gz") else open)(path, "rt", encoding="utf-8"):
        r = json.loads(line)
        if r.get("arm") != arm:
            continue
        shown = parse_options(r["user"])
        letters = sorted(shown)
        ranks = ranks_of([parse_number(shown[x]) for x in letters])
        if ranks is None:
            continue
        answer = r.get("answer")
        out.append({"item": int(r["item"]), "draw": int(r.get("draw", 0)), "capsule": r["cluster"],
                    "key_rank": ranks[letters.index(r["gold"])],
                    "pick_rank": ranks[letters.index(answer)] if answer in letters else None,
                    "correct": 1.0 if answer == r["gold"] else 0.0, "parsed": answer in letters})
    return out


def summarise_dump(rows):
    picks = Counter(r["pick_rank"] for r in rows if r["pick_rank"] is not None)
    n_pick = sum(picks.values())
    keys = Counter(r["key_rank"] for r in rows)
    return {"n": len(rows), "unparsed": 100 * (1 - float(np.mean([r["parsed"] for r in rows]))),
            "margin": interval([r["correct"] - CHANCE for r in rows], [r["capsule"] for r in rows], None),
            "pick_rank_shares": [100 * picks[j] / n_pick for j in range(K)] if n_pick else None,
            "accuracy_by_key_rank": [100 * float(np.mean([r["correct"] for r in rows if r["key_rank"] == j]))
                                     if keys[j] else None for j in range(K)]}


def paired(a_rows, b_rows):
    """a - b on the same (item, draw), paired over capsules at the file's 96.0%."""
    b = {(r["item"], r["draw"]): r for r in b_rows}
    both = [(r, b[(r["item"], r["draw"])]) for r in a_rows if (r["item"], r["draw"]) in b]
    return interval([x["correct"] - y["correct"] for x, y in both], [x["capsule"] for x, _ in both], bw.LEVEL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="results/rank_attribution.json")
    ap.add_argument("--reps", type=int, default=2000, help="simulated files per covering level")
    ap.add_argument("--latex", action="store_true", help="print tab:attribution's rows from the report")
    args = ap.parse_args()
    if args.latex:
        for line in table_rows(json.loads(pathlib.Path(args.output).read_text())):
            print(line)
        return

    def calibrate(first, second=None):
        return covering_level(first, second, reps=args.reps)

    report = {"published": {}, "open_question_shown": {}, "open_question_withheld": {}}
    for release, loader in (("v1.5", published_v15), ("v1.0", published_v10)):
        for model in MODELS:
            s = summarise_published(loader(model), calibrate)
            report["published"][f"{release}|{model}"] = s
            d = s["second_smallest_minus_rest"]
            print(f"{release} {MODELS[model]:18s} numeric {s['margin_numeric']['mean']:+5.1f} "
                  f"[{s['margin_numeric']['lo']:+.1f},{s['margin_numeric']['hi']:+.1f}] (n={s['n_numeric']})  "
                  f"other {s['margin_other']['mean']:+5.1f} [{s['margin_other']['lo']:+.1f},{s['margin_other']['hi']:+.1f}] "
                  f"(n={s['n_other']})  2nd-smallest minus rest {d['mean']:+5.1f} [{d['lo']:+.1f},{d['hi']:+.1f}]  "
                  + (f"a rank reader: {s['rank_reader_would_score']['difference']:+.1f}"
                     if "rank_reader_would_score" in s else "")
                  + (f"  picks {[round(x, 1) for x in s['pick_rank_shares']]}" if "pick_rank_shares" in s else ""))

    for short, name in OPEN.items():
        entry = {}
        for arm, suffix in (("released", ""), ("placebo", "_placebo"), ("repaired", "_repaired")):
            path = arm_intervals.resolve(f"build/dumps/free_{short}_bixnum{suffix}.jsonl")
            if path.exists():
                entry[arm] = dump_rows(path, "mcq")
        if "released" not in entry:
            continue
        out = {arm: summarise_dump(rows) for arm, rows in entry.items()}
        for arm in ("placebo", "repaired"):
            if arm in entry:
                out[f"released-{arm}"] = paired(entry["released"], entry[arm])
        report["open_question_shown"][name] = out
        line = "  ".join(f"{arm} {out[arm]['margin']['mean']:+5.1f} (unparsed {out[arm]['unparsed']:.0f}%)"
                         for arm in ("released", "placebo", "repaired") if arm in out)
        extra = "  ".join(f"released-{arm} {out[f'released-{arm}']['mean']:+5.1f} "
                          f"[{out[f'released-{arm}']['lo']:+.1f},{out[f'released-{arm}']['hi']:+.1f}]"
                          for arm in ("placebo", "repaired") if f"released-{arm}" in out)
        print(f"question shown  {name:14s} {line}  {extra}")

    for tag, name in WITHHELD.items():
        path = arm_intervals.resolve(f"build/dumps/{tag}_bixall_bixprompt_argmax.jsonl")
        if not path.exists():
            continue
        s = summarise_dump(dump_rows(path, "file"))
        report["open_question_withheld"][name] = s
        print(f"question withheld {name:14s} margin {s['margin']['mean']:+5.1f}  picks "
              f"{[round(x, 1) for x in s['pick_rank_shares']]}")
    shares = [s["pick_rank_shares"] for s in report["open_question_withheld"].values()]
    report["withheld_summary"] = {
        "n_models": len(shares),
        "max_second_smallest_pick_share": max(x[1] for x in shares),
        "n_favouring_smallest": sum(1 for x in shares if max(range(K), key=lambda j: x[j]) == 0),
        "n_favouring_second_smallest": sum(1 for x in shares if max(range(K), key=lambda j: x[j]) == 1)}
    print(report["withheld_summary"])
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


def _ci(x):
    return f"${x['mean']:+.1f}$ $[{x['lo']:+.1f},{x['hi']:+.1f}]$"


def _ci_small(x):
    return f"${x['mean']:+.1f}$ {{\\scriptsize$[{x['lo']:+.1f},{x['hi']:+.1f}]$}}"


def table_rows(report):
    """tab:attribution: the published runs, split by where the key sits, then tab:rewritten:
    our open models' forced arm through the three option sets."""
    rows = []
    for release in ("v1.5", "v1.0"):
        for model in MODELS:
            s = report["published"][f"{release}|{model}"]
            acc = [s["accuracy_by_key_rank"][str(j)]["accuracy"] for j in range(K)]
            rows.append(f"{release} & {SHORT[model]} & {_ci_small(s['margin_numeric'])} & "
                        f"{_ci_small(s['margin_other'])} & " + " & ".join(f"${a:.1f}$" for a in acc)
                        + f" & {_ci_small(s['second_smallest_minus_rest'])}\\\\")
    rows.append("")
    for name, out in report["open_question_shown"].items():
        cells = []
        for arm in ("released", "placebo", "repaired"):
            m = out[arm]
            cells.append(f"unparsed ${m['unparsed']:.0f}\\%$" if m["unparsed"] > UNPARSED_LIMIT
                         else f"${m['margin']['mean']:+.1f}$")
        withheld = any(out[arm]["unparsed"] > UNPARSED_LIMIT for arm in ("released", "placebo", "repaired"))
        diffs = ["--" if withheld else _ci(out[f"released-{arm}"]) for arm in ("placebo", "repaired")]
        rows.append(f"{name} & " + " & ".join(cells) + " & " + " & ".join(diffs) + "\\\\")
    return rows


if __name__ == "__main__":
    main()
