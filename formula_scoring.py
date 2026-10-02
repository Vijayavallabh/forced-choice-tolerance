#!/usr/bin/env python3
r"""Forced-choice scores corrected for guessing, and what the correction leaves (not registered).

Formula scoring, the classical correction for guessing, maps a forced-choice score $S$ over $k$ options to
$(S-1/k)/(1-1/k)$: it removes what a random choice among the options would add on every answer it cannot
place, and it is a monotone map, so it keeps every ranking. Applied to a 2026 system's $64.4\%$ forced-choice
score it gives $52.5\%$, against its $48.8\%$ open-ended. This asks what it leaves on the paper's runs.

For each grading of Table 2 (and the code-free gradings of the same runs) the corrected score minus the share
within 5% of the key is split exactly into what each kind of answer contributes beyond a random choice:

    (S - 1/k)/(1 - 1/k) - t = k/(k-1) * [ h (a_h - 1) + sum_j m_j (a_j - 1/k) - e/k ]

with $t=h$ the share within 5%, $a_h$ the share of those the grading accepts, $m_j$ and $a_j$ the share of runs
that are misses of kind $j$ and the share of them it accepts (the key the single nearest option to the number,
another option, or neither: no number, or a number equidistant from several options), and $e$ the share of
empty answers, scored wrong without grading. A grading that accepted every correct answer and chose at random
on every miss would leave zero; what remains is the part of the score that depends on where each miss lies
among the options. Each question's runs are averaged, then the questions, as in Table 2. Beside it: the share of
correct answers each grading accepts, the share of misses it accepts, and the share of misses it accepts among
those on which it selects an option.

    python3 formula_scoring.py            # results/formula_scoring.json
    python3 formula_scoring.py --latex    # the table's rows
"""
import argparse
import collections
import gzip
import json
import pathlib
import zlib

import numpy as np

import answer_extraction as ax
import replication as rp
import score_decomposition as sd
from answer_numbers import graded

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "formula_scoring.json"
K = 4
TOL = 0.05
REPORTED = {"forced": 64.4, "open": 48.8}          # the 2026 system's v1.5 scores (bioagents)
KINDS = ("correct", "miss, key nearest", "miss, other nearest", "miss, neither", "empty")


def kind_of(r):
    if not (r["answer"] and str(r["answer"]).strip()):
        return "empty"
    if graded(str(r["answer"]), r["options"][0], TOL):
        return "correct"
    near = sd.single_nearest(r)
    return {"key": "miss, key nearest", "other": "miss, other nearest", None: "miss, neither"}[near]


def selected(reads):
    """The share of a run's reads that select one of the k options (not none, not the refusal option)."""
    return float(np.mean([(x.get("rank") is not None) and not x.get("refused", False) for x in reads]))


def per_item(runs, value):
    acc = collections.defaultdict(list)
    for r in runs:
        v = value(r)
        if v is not None:
            acc[r["q"]].append(v)
    return {q: float(np.mean(v)) for q, v in acc.items()}


def mean(d):
    return 100.0 * float(np.mean(list(d.values()))) if d else None


def summarise(runs, reading, capsule, tag):
    read = [r for r in runs if sd.named_share(r, reading) is not None]
    acc = lambda r: sd.named_share(r, reading)
    tol = lambda r: float(kind_of(r) == "correct")
    out = {"n_runs": len(read), "n_items": len({r["q"] for r in read})}
    out["score"], out["tolerance"] = mean(per_item(read, acc)), mean(per_item(read, tol))
    out["corrected"] = (out["score"] - 100.0 / K) / (1 - 1 / K)
    # the corrected score minus the tolerance, per question, with the level the double bootstrap finds covers 95%
    diff = per_item(read, lambda r: (acc(r) - 1 / K) / (1 - 1 / K) - tol(r))
    rng = np.random.default_rng(zlib.crc32(tag.encode()))
    out["corrected_minus_tolerance"] = rp.calibrated(diff, capsule, rng, 2000, 5000)
    # its exact split by kind of answer: each run contributes k/(k-1) * (accepted - expected at random)
    expected = {"correct": 1.0, "empty": 1 / K}
    out["parts"] = {}
    for kind in KINDS:
        share = lambda r, kind=kind: float(kind_of(r) == kind)
        part = per_item(read, lambda r, kind=kind: (float(kind_of(r) == kind) *
                                                     K / (K - 1) * (acc(r) - expected.get(kind, 1 / K))))
        out["parts"][kind] = {"points": mean(part), "share_of_runs": mean(per_item(read, share))}
    # acceptance by kind, each question's runs averaged and the questions summed, as score_decomposition does
    for kind in KINDS[:-1]:
        num = per_item(read, lambda r, kind=kind: float(kind_of(r) == kind) * acc(r))
        den = per_item(read, lambda r, kind=kind: float(kind_of(r) == kind))
        out["parts"][kind]["accepted"] = 100.0 * sum(num.values()) / sum(den.values()) if sum(den.values()) else None
    # shares weighted as Table 2 weighs them: each question's runs averaged, the questions summed
    is_miss = lambda r: float(kind_of(r).startswith("miss"))
    num = per_item(read, lambda r: is_miss(r) * acc(r))
    den = per_item(read, is_miss)
    sel = per_item(read, lambda r: is_miss(r) * (selected(r["reads"][reading]) if r["reads"].get(reading) else 0.0))
    ratio = lambda a, b: 100.0 * sum(a.values()) / sum(b.values()) if sum(b.values()) else None
    out["misses_accepted"] = ratio(num, den)
    out["misses_selected"] = ratio(sel, den)
    out["misses_accepted_of_selected"] = ratio(num, sel)
    out["correct_accepted"] = out["parts"]["correct"]["accepted"]
    # classical formula scoring with omissions: a run the grading selects no option for, or an empty answer,
    # scores zero, a wrong selection -1/(k-1); for a grading that always selects, the correction above plus
    # the empty answers' share over k-1
    # an empty answer is scored wrong without grading (acc 0); it is an omission, not a wrong selection, even
    # where a published grade happened to name a letter for it
    wrong = lambda r: 0.0 if kind_of(r) == "empty" else (
        (selected(r["reads"][reading]) if r["reads"].get(reading) else 0.0) - acc(r))
    out["omission_aware_minus_tolerance"] = mean(per_item(read, lambda r: acc(r) - wrong(r) / (K - 1) - tol(r)))
    # the same with its interval, at the level the double bootstrap finds covers 95%, from its own generator
    out["omission_aware_minus_tolerance_ci"] = rp.calibrated(
        per_item(read, lambda r: acc(r) - wrong(r) / (K - 1) - tol(r)), capsule,
        np.random.default_rng(zlib.crc32(("omission|" + tag).encode())), 2000, 5000)
    check = sum(p["points"] for p in out["parts"].values())
    assert abs(check - out["corrected_minus_tolerance"]["mean"]) < 1e-6, (check, out["corrected_minus_tolerance"])
    return out


def codefree_runs(reader):
    """The code-free forced gradings of the same with-data runs through the released options, as
    ``score_decomposition``'s runs, so the same summary applies."""
    path = ROOT / "results" / f"grading_variants_{reader}_codefree_forced_rows.jsonl.gz"
    doc = rp.load_extract()
    sets10 = rp.v10_sets()
    import bixbench_withdata as bw
    sets15 = bw.option_sets()
    capsule15 = {json.loads(l)["question_id"]: json.loads(l)["capsule_uuid"] for l in open(ROOT / "data" / "bixbench.jsonl")}
    out = []
    with gzip.open(path, "rt") as fh:
        for x in map(json.loads, fh):
            if x["condition"] != "data":
                continue
            if x["set"] == "D1":
                if x["q"] not in sets10:
                    continue
                options, cap = sets10[x["q"]]["released"], doc["items"][x["q"]]["capsule"]
                group = "gpt-4o" if x["run"].startswith("4o_") else "Claude 3.5 Sonnet"
            else:
                if x["q"] not in sets15 or "|" in x["run"] or x["run"].startswith("gpt-5.1"):
                    continue                  # the seven configurations' first runs only
                options, cap = sets15[x["q"]]["released"], capsule15[x["q"]]
                group = "seven configurations"
            reads = x["reads"].get("released")
            out.append({"key": f"{x['set']}|{x['run']}|{x['q']}|{x['i']}", "group": group, "run": x["run"],
                        "q": x["q"], "capsule": cap, "answer": x["answer"], "options": options,
                        "reads": {} if reads is None else {"code-free": [dict(r, refused=False) for r in reads]}})
    return out


ROWS = (("v1.0", "gpt-4o", "published forced", "v1.0, gpt-4o", "forced"),
        ("v1.0", "Claude 3.5 Sonnet", "published forced", "v1.0, Claude 3.5", "forced"),
        ("v1.5", "pooled", "family forced", "v1.5, seven configurations", "forced"),
        ("v1.5", "pooled", "gemma forced", "", "gemma-3-27b, forced"),
        ("v1.5 new", "new seeds", "gemma forced", "v1.5, new seeds", "forced"),
        ("v1.5 new", "Qwen3-235B-A22B", "gemma forced", "v1.5, Qwen3-235B-A22B", "forced"),
        ("v1.5 closed", "gpt-5.1", "gpt-4o forced", "v1.5, gpt-5.1", "forced"),
        ("current", "gpt-6-luna", "gpt-4o forced", "v1.5, gpt-6-luna", "forced"),
        ("current", "DeepSeek-V4-Pro", "gpt-4o forced", "v1.5, DeepSeek-V4-Pro", "forced"),
        ("code-free", "gpt-4o", "code-free", "v1.0, gpt-4o", "code-free"),
        ("code-free", "Claude 3.5 Sonnet", "code-free", "v1.0, Claude 3.5", "code-free"),
        ("code-free", "seven configurations", "code-free", "v1.5, seven configurations", "code-free"))


def current_runs():
    """The two current agents' runs as ``score_decomposition`` reads them (``frontier_agents.as_runs``), graded by
    gpt-4o; imported here, since frontier_agents imports this module."""
    import bixbench_withdata as bw
    import frontier_agents as fa
    sets, rows = bw.option_sets(), fa.load_rows()
    return [r for model, label in fa.MODELS if label in fa.REPORTED
            for r in fa.as_runs([x for x in rows if x["model"] == model], label, sets)]


def analyse(_args):
    blocks = {"v1.0": sd.d1_runs(), "v1.5": sd.d0_runs(), "v1.5 new": sd.d2_runs(), "v1.5 closed": sd.d3_runs(),
              "current": current_runs(), "code-free": codefree_runs("gpt-4o")}
    report = {"k": K, "tolerance": TOL, "not_registered": True, "rows": []}
    rep = REPORTED
    report["reported"] = dict(rep, corrected=(rep["forced"] - 100 / K) / (1 - 1 / K),
                              corrected_minus_open=(rep["forced"] - 100 / K) / (1 - 1 / K) - rep["open"],
                              guessing_share_of_gap=(100 - rep["open"]) / K)
    for block, group, reading, label, how in ROWS:
        runs = blocks[block]
        runs = runs if group == "pooled" else [r for r in runs if r["group"] == group]
        capsule = {r["q"]: r["capsule"] for r in runs}
        # a current agent's interval from the generator frontier_agents.py draws it from, so the two agree
        tag = f"v1.5 frontier|{group}|corrected" if block == "current" else f"{block}|{group}|{reading}"
        s = summarise(runs, reading, capsule, tag)
        report["rows"].append({"block": block, "group": group, "reading": reading, "label": label, "how": how, **s})
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    f = lambda v: "  --" if v is None else f"{v:5.1f}"
    r0 = report["reported"]
    print(f"2026 system: forced {r0['forced']} -> corrected {r0['corrected']:.1f}, open {r0['open']}: "
          f"{r0['corrected_minus_open']:+.1f}; a quarter of its misses {r0['guessing_share_of_gap']:.1f}")
    for r in report["rows"]:
        d = r["corrected_minus_tolerance"]
        parts = " ".join(f"{k.split(', ')[-1][:7]} {v['points']:+5.1f}" for k, v in r["parts"].items())
        print(f"{r['block']:11s} {r['group']:22s} {r['how']:12s} tol {f(r['tolerance'])} score {f(r['score'])} "
              f"corr {f(r['corrected'])} | corr-tol {d['mean']:+5.1f} [{d['lo']:+5.1f},{d['hi']:+5.1f}] | {parts} | "
              f"correct acc {f(r['correct_accepted'])} misses acc {f(r['misses_accepted'])} "
              f"of selected {f(r['misses_accepted_of_selected'])} selected {f(r['misses_selected'])} "
              f"omission-aware {r['omission_aware_minus_tolerance']:+.1f}")
    print(f"wrote {OUT.relative_to(ROOT)}")


def table_rows(report):
    """The rows of the paper's table of what the correction for guessing leaves."""
    rows = []
    for r in report["rows"]:
        d, p = r["corrected_minus_tolerance"], r["parts"]
        tol = f"${r['tolerance']:.1f}$" if r["label"] else ""
        rows.append(f"{r['label']} & {r['how']} & {tol} & ${r['correct_accepted']:.1f}$ & ${r['misses_accepted']:.1f}$ & "
                    f"${r['misses_accepted_of_selected']:.1f}$ & ${d['mean']:+.1f}$ {{\\scriptsize$[{d['lo']:+.1f},{d['hi']:+.1f}]$}} & "
                    f"${p['miss, key nearest']['points']:+.1f}$ & ${p['miss, other nearest']['points']:+.1f}$ & "
                    f"${p['miss, neither']['points']:+.1f}$ & ${p['correct']['points'] + p['empty']['points']:+.1f}$\\\\")
    return rows


def latex(_args):
    for row in table_rows(json.loads(OUT.read_text())):
        print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    (latex if args.latex else analyse)(args)


if __name__ == "__main__":
    main()
