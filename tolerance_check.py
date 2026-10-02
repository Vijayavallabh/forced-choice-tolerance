#!/usr/bin/env python3
r"""How wide a numeric key's tolerance is, measured against the benchmark's own judgment
(PREREGISTRATION.md, T1 and T2).

T1. BixBench's authors wrote some keys as ranges -- 61 of v1.5's 205 -- and a range is
    the tolerance its author accepts for that quantity. Each one's relative half-width
    ``(hi-lo)/(hi+lo)`` is read off the file.
T2. BixBench's graders judged every published with-data answer: the v1.0 grader on
    gpt-4o's and Claude 3.5 Sonnet's (``results/bixbench_v10_published_runs.json.gz``)
    and the v1.5 graders on this repository's seven run sets. Against each verdict, a
    number within t of the key (``free_response.graded``, the paper's rule) for t of 1,
    2, 5, 10 and 20%, and the key-precision rule (``bixbench_withdata.strict_numeric``):
    Cohen's kappa, and how often the grader accepts an answer at each distance from the
    key -- the tolerance the grader applies, whatever it was told.

    python3 tolerance_check.py      # results/tolerance_check.json
"""
import gzip
import json
import math
import pathlib
import re
from collections import defaultdict

import numpy as np

import bixbench_withdata as bw
from channel_survey import numeric_ranks
from answer_numbers import NUMBER, as_number, graded, normalise

ROOT = pathlib.Path(__file__).resolve().parent
TOLERANCES = (0.01, 0.02, 0.05, 0.10, 0.20)
BINS = ((0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, math.inf))
RANGE = re.compile(r"^\s*\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)\s*$")


def range_widths():
    """T1: every key written as (lo, hi), with its relative half-width."""
    out = []
    for line in open(ROOT / "data" / "bixbench.jsonl"):
        row = json.loads(line)
        m = RANGE.match(row["ideal"])
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if lo + hi:
                out.append({"release": "v1.5", "question_id": row["question_id"], "ideal": row["ideal"],
                            "half_width": abs(hi - lo) / abs(hi + lo)})
    with gzip.open(ROOT / "results" / "bixbench_v10_published_runs.json.gz", "rt") as fh:
        for q, e in sorted(json.load(fh)["items"].items()):
            m = RANGE.match(e["ideal"])
            if m:
                lo, hi = float(m.group(1)), float(m.group(2))
                out.append({"release": "v1.0", "question_id": q, "ideal": e["ideal"],
                            "half_width": abs(hi - lo) / abs(hi + lo)})
    return out


def relative_error(answer, ideal):
    """|a-y|/|y| for the last number in the answer, read as ``free_response.graded`` reads it."""
    key = as_number(ideal)
    found = NUMBER.findall(normalise(answer or ""))
    if key is None or not found:
        return None
    got = as_number(found[-1])
    if got is None:
        return None
    (y, y_pct), (a, a_pct) = key, got
    if y_pct and not a_pct and abs(a) <= 1.0:
        a *= 100.0
    if a_pct and not y_pct and abs(y) <= 1.0:
        y *= 100.0
    return abs(a - y) / abs(y) if y else (0.0 if a == 0 else math.inf)


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(rows):
    """rows: (answer, ideal, verdict). Each rule against the verdict, and acceptance by distance."""
    out = {"n": len(rows), "grader_accepts": int(sum(v for _, _, v in rows))}
    rules = {f"within {round(100 * t)}%": [bool(graded(a, y, t)) for a, y, _ in rows] for t in TOLERANCES}
    rules["key precision"] = [bool(bw.strict_numeric(a, y)) for a, y, _ in rows]
    verdict = [bool(v) for _, _, v in rows]
    out["rules"] = {}
    for name, accept in rules.items():
        out["rules"][name] = {
            "kappa": kappa(accept, verdict),
            "rule_accepts": int(sum(accept)),
            "grader_yes_rule_no": int(sum(v and not r for v, r in zip(verdict, accept))),
            "grader_no_rule_yes": int(sum(r and not v for v, r in zip(verdict, accept)))}
    by_bin = defaultdict(lambda: [0, 0])
    for (a, y, v) in rows:
        e = relative_error(a, y)
        if e is None:
            continue
        for lo, hi in BINS:
            if lo <= e < hi or (e == 0 and lo == 0):
                by_bin[f"{lo:g}-{hi:g}"][0] += int(bool(v))
                by_bin[f"{lo:g}-{hi:g}"][1] += 1
                break
    out["acceptance_by_distance"] = {k: {"accepted": a, "n": n, "rate": a / n if n else None}
                                     for k, (a, n) in by_bin.items()}
    return out


T3_TOLERANCES = (0.01, 0.02, 0.05, 0.10)


def restated():
    """T3: the paper's tolerance-dependent numbers at 1%, 2%, 5% and 10%."""
    import bracketing
    import published_repair as pr
    import release_arms as ra
    out = {"table_1_regrade": {}, "withdata": {}, "table_8_cap": {}, "released_reading": {}, "writers": {}}
    # the forced and may-decline readings' excess over the tolerance (score_decomposition.py computes it at
    # every tolerance here) and the writers' share of misses credited (writer_panel.py)
    sd = json.loads((ROOT / "results" / "score_decomposition.json").read_text())
    for block, group, reading, label in (("v1.0", "gpt-4o", "published forced", "v1.0|gpt-4o|forced"),
                                         ("v1.0", "Claude 3.5 Sonnet", "published forced", "v1.0|claude|forced"),
                                         ("v1.0", "gpt-4o", "published may-decline", "v1.0|gpt-4o|may decline"),
                                         ("v1.0", "Claude 3.5 Sonnet", "published may-decline", "v1.0|claude|may decline"),
                                         ("v1.5", "pooled", "family forced", "v1.5|seven|forced"),
                                         ("v1.5", "pooled", "family may-decline", "v1.5|seven|may decline"),
                                         ("v1.5", "pooled", "gemma forced", "v1.5|seven|gemma forced"),
                                         ("v1.5 new", "new seeds", "gemma forced", "v1.5|new seeds|forced"),
                                         ("v1.5 new", "Qwen3-235B-A22B", "gemma forced", "v1.5|qwen3-235b|forced"),
                                         ("v1.5 closed", "gpt-5.1", "gpt-4o forced", "v1.5|gpt-5.1|forced"),
                                         ("v1.5 closed", "gpt-5.1", "gpt-4o may-decline", "v1.5|gpt-5.1|may decline")):
        got = sd[block][group][reading]["restated"]
        out["released_reading"][label] = {f"{round(100 * t)}%": got[f"{t:g}"] for t in T3_TOLERANCES}
    # the current agents' rows of Table 2 (frontier_agents.py, which summarises them as score_decomposition does)
    fa = json.loads((ROOT / "results" / "frontier_agents.json").read_text())["agents"]
    for label in ("gpt-6-luna", "DeepSeek-V4-Pro"):
        for reading, mode in (("gpt-4o forced", "forced"), ("gpt-4o may-decline", "may decline")):
            got = fa[label]["table1"][reading]["restated"]
            out["released_reading"][f"v1.5|{label}|{mode}"] = {f"{round(100 * t)}%": got[f"{t:g}"]
                                                               for t in T3_TOLERANCES}
    import writer_panel as wp
    import replication
    qid_of, capsule_of = wp.to_qid()
    doc = replication.load_extract()
    answers10 = [(q, a) for run in replication.OPEN_RUNS for q, a, _ in doc["open_runs"][run]]
    capsule10 = {q: e["capsule"] for q, e in doc["items"].items()}
    w15, w10 = wp.writers(), wp.writers_v10()
    for name in ("released", "rank uniform"):
        for release, rows, cap, answers in (("v1.5", w15[name], capsule_of, None),
                                            ("v1.0", w10[name], capsule10, answers10)):
            out["writers"][f"{release}|{name}"] = {
                f"{round(100 * t)}%": wp.catch(rows, qid_of, cap, answers, tol=t)["credited_misses"]
                for t in T3_TOLERANCES}
    for release, loader in (("v1.0", pr.v10), ("v1.5", pr.v15)):
        for model in ra.MODELS:
            _, raw = loader(model)
            numeric = [(r, ra.key_of(r["target"])) for r in raw["open_ended"]]
            numeric = [(r, k) for r, k in numeric if k is not None]
            out["table_1_regrade"][f"{release}|{model}"] = {
                f"{round(100 * t)}%": 100.0 * float(np.mean([ra.hits(r["predicted"], k, t, "any") for r, k in numeric]))
                for t in T3_TOLERANCES}
    items = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    per = {}
    for run in ("qwen72b", "llama70b", "gemma27b"):
        per[run] = {(r["question_id"], r["condition"]): r for r in bracketing.rows_of(run) if r["numeric"]}
    for t in T3_TOLERANCES:
        within = lambda r: float(bool(graded(r["answer"], items[r["question_id"]]["ideal"], t))) if r["answer"] else 0.0
        entry = {}
        qids = sorted(q for (q, c) in per["qwen72b"] if c == "data")
        caps = [items[q]["capsule_uuid"] for q in qids]
        for cond in ("data", "nodata"):     # through the interval routine, as the paper's rates are
            entry[f"qwen72b|{cond}"] = bw.cluster_interval([within(per["qwen72b"][(q, cond)]) for q in qids],
                                                           caps)["mean"]
        entry["qwen72b data - nodata"] = bw.cluster_interval(
            [within(per["qwen72b"][(q, "data")]) - within(per["qwen72b"][(q, "nodata")]) for q in qids], caps)
        for other in ("llama70b", "gemma27b"):
            entry[f"qwen72b - {other}"] = bw.cluster_interval(
                [within(per["qwen72b"][(q, "data")]) - within(per[other][(q, "data")]) for q in qids], caps)
        out["withdata"][f"{round(100 * t)}%"] = entry
    # Table 8's free arm: the last number in the reply (its column), and any number in it
    # (release_arms.our_free_runs), each model's best
    doc = json.load(open(ROOT / "results" / "free_response.json"))
    import arm_intervals
    keys = [ra.key_of(json.loads(l)["ideal"]) for l in open(ROOT / "build" / "bixbench_numeric_q.jsonl")]
    dumps = {}
    for tag in ("Qwen2_5-1_5B", "Llama-3_2-3B", "Qwen2_5-7B", "Meta-Llama-3_1-8B", "phi-4", "Qwen2_5-14B"):
        path = arm_intervals.resolve(f"build/dumps/free_{tag}_bixnum.jsonl")
        opener = gzip.open if str(path).endswith(".gz") else open
        dumps[tag] = [r for r in map(json.loads, opener(path, "rt")) if r.get("arm") == "free"]
    for t in T3_TOLERANCES:
        # free_response.py graded its arm at 1, 5 and 10% only
        last = [v["arms"]["free"]["by_tolerance"][str(t)]["accuracy"] for k, v in doc.items()
                if k.endswith("|bixbench_numeric_q") and str(t) in v["arms"]["free"]["by_tolerance"]]
        anyn = [100.0 * float(np.mean([ra.hits(r.get("reply", ""), keys[r["item"]], t, "any")
                                        for r in rows if keys[r["item"]]])) for rows in dumps.values()]
        out["table_8_cap"][f"{round(100 * t)}%"] = {"last_number": max(last) if last else None, "any_number": max(anyn)}
    return out


def pvalue_log(factors=(2.0, 10.0)):
    """T3's with-data results with the keys that are p-values (``reference_check.kinds``) graded on a log
    scale, the last number within a factor of 2 or of 10 of the key, and every other key within 5%."""
    import bracketing
    import reference_check as rc
    kind = rc.kinds()
    items = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    per = {run: {(r["question_id"], r["condition"]): r for r in bracketing.rows_of(run) if r["numeric"]}
           for run in ("qwen72b", "llama70b", "gemma27b")}
    qids = sorted(q for (q, c) in per["qwen72b"] if c == "data")
    caps = [items[q]["capsule_uuid"] for q in qids]
    out = {"n_items": len(qids), "n_p_value_items": sum(kind[q] == "p-value" for q in qids)}
    for factor in factors:
        def within(r, factor=factor):
            if not r["answer"]:
                return 0.0
            ideal = items[r["question_id"]]["ideal"]
            if kind[r["question_id"]] == "p-value":
                return float(rc.graded_log(r["answer"], ideal, factor))
            return float(bool(graded(r["answer"], ideal, 0.05)))
        entry = {}
        for cond in ("data", "nodata"):
            entry[f"qwen72b|{cond}"] = bw.cluster_interval([within(per["qwen72b"][(q, cond)]) for q in qids], caps)["mean"]
        entry["qwen72b data - nodata"] = bw.cluster_interval(
            [within(per["qwen72b"][(q, "data")]) - within(per["qwen72b"][(q, "nodata")]) for q in qids], caps)
        for other in ("llama70b", "gemma27b"):
            entry[f"qwen72b - {other}"] = bw.cluster_interval(
                [within(per["qwen72b"][(q, "data")]) - within(per[other][(q, "data")]) for q in qids], caps)
        out[f"x{factor:g}"] = entry
    return out


def table_rows(report):
    """tab:tolerance (each rule against each grader) and tab:restated (the paper at 1, 2, 5, 10%)."""
    g10, g15 = report["T2_v10_published_grader"]["rules"], report["T2_v15_graders"]["rules"]
    # a rule's name carries a per cent sign, which LaTeX would take for a comment and drop the row's cells
    rows = [f"{name.replace('%', chr(92) + '%')} & ${g10[name]['kappa']:.2f}$ & ${g10[name]['grader_yes_rule_no']}$ & "
            f"${g10[name]['grader_no_rule_yes']}$ & ${g15[name]['kappa']:.2f}$\\\\" for name in g10]
    rows.append("")
    t3 = report["T3"]
    f = lambda x: f"${x['mean']:+.1f}$ {{\\scriptsize$[{x['lo']:+.1f},{x['hi']:+.1f}]$}}"
    tols = ("1%", "2%", "5%", "10%")
    reg = t3["table_1_regrade"]
    for key, label in (("v1.5|gpt-4o", "Table~\\ref{tab:arms}, v1.5, gpt-4o"),
                       ("v1.5|claude-3-5-sonnet-latest", "Table~\\ref{tab:arms}, v1.5, Claude 3.5"),
                       ("v1.0|gpt-4o", "Table~\\ref{tab:arms}, v1.0, gpt-4o"),
                       ("v1.0|claude-3-5-sonnet-latest", "Table~\\ref{tab:arms}, v1.0, Claude 3.5")):
        rows.append(f"{label} & " + " & ".join(f"${reg[key][t]:.1f}$" for t in tols) + "\\\\")
    w = t3["withdata"]
    rows.append("Qwen2.5-72B within, with the data & " + " & ".join(f"${w[t]['qwen72b|data']:.1f}$" for t in tols) + "\\\\")
    rows.append("\\quad without it & " + " & ".join(f"${w[t]['qwen72b|nodata']:.1f}$" for t in tols) + "\\\\")
    rows.append("\\quad with $-$ without & " + " & ".join(f(w[t]["qwen72b data - nodata"]) for t in tols) + "\\\\")
    rows.append("\\quad lead over Llama-3.3-70B & " + " & ".join(f(w[t]["qwen72b - llama70b"]) for t in tols) + "\\\\")
    rows.append("\\quad lead over gemma-3-27b & " + " & ".join(f(w[t]["qwen72b - gemma27b"]) for t in tols) + "\\\\")
    rows.append("Table~\\ref{tab:free}, any number, best model & "
                + " & ".join(f"${t3['table_8_cap'][t]['any_number']:.1f}$" for t in tols) + "\\\\")
    rr = t3["released_reading"]
    for key, label in (("v1.0|gpt-4o|forced", "Table~\\ref{tab:released}, gpt-4o, forced"),
                       ("v1.0|claude|forced", "\\quad Claude 3.5 Sonnet, forced"),
                       ("v1.0|gpt-4o|may decline", "\\quad gpt-4o, with refusal"),
                       ("v1.0|claude|may decline", "\\quad Claude 3.5 Sonnet, with refusal"),
                       ("v1.5|seven|forced", "\\quad v1.5, seven configurations, forced"),
                       ("v1.5|seven|may decline", "\\quad v1.5, seven configurations, with refusal"),
                       ("v1.5|seven|gemma forced", "\\quad v1.5, seven configurations, gemma-3-27b, forced"),
                       ("v1.5|new seeds|forced", "\\quad v1.5, new seeds, forced"),
                       ("v1.5|qwen3-235b|forced", "\\quad v1.5, Qwen3-235B-A22B, forced"),
                       ("v1.5|gpt-5.1|forced", "\\quad v1.5, gpt-5.1, forced"),
                       ("v1.5|gpt-5.1|may decline", "\\quad v1.5, gpt-5.1, with refusal"),
                       ("v1.5|gpt-6-luna|forced", "\\quad v1.5, gpt-6-luna, forced"),
                       ("v1.5|gpt-6-luna|may decline", "\\quad v1.5, gpt-6-luna, with refusal"),
                       ("v1.5|DeepSeek-V4-Pro|forced", "\\quad v1.5, DeepSeek-V4-Pro, forced"),
                       ("v1.5|DeepSeek-V4-Pro|may decline", "\\quad v1.5, DeepSeek-V4-Pro, with refusal")):
        rows.append(f"{label} & " + " & ".join(f(rr[key][t]) for t in tols) + "\\\\")
    wr = t3["writers"]
    for key, label in (("v1.5|released", "Misses the rule accepts, v1.5, $R$"),
                       ("v1.5|rank uniform", "\\quad v1.5, $U$"),
                       ("v1.0|released", "\\quad v1.0, $R$"),
                       ("v1.0|rank uniform", "\\quad v1.0, $U$")):
        rows.append(f"{label} & " + " & ".join(f"${wr[key][t]['mean']:.1f}$" for t in tols) + "\\\\")
    return rows


def main():
    import sys
    if "--latex" in sys.argv:
        for line in table_rows(json.loads((ROOT / "results" / "tolerance_check.json").read_text())):
            print(line)
        return
    report = {}
    widths = range_widths()
    hw = np.array([w["half_width"] for w in widths if w["release"] == "v1.5"])
    report["T1"] = {"n_v15": int(len(hw)), "n_v10": sum(w["release"] == "v1.0" for w in widths),
                    "median": float(np.median(hw)), "quartiles": [float(np.percentile(hw, 25)),
                                                                  float(np.percentile(hw, 75))],
                    "share_within_5pct": float(np.mean(hw <= 0.05)),
                    "share_within_2pct": float(np.mean(hw <= 0.02)),
                    "share_within_10pct": float(np.mean(hw <= 0.10)),
                    "ranges": widths}
    print(f"T1: {len(hw)} v1.5 range keys, half-width median {100 * np.median(hw):.1f}% "
          f"(IQR {100 * np.percentile(hw, 25):.1f}-{100 * np.percentile(hw, 75):.1f}%), "
          f"{100 * np.mean(hw <= 0.05):.0f}% at or under 5%")

    with gzip.open(ROOT / "results" / "bixbench_v10_published_runs.json.gz", "rt") as fh:
        doc = json.load(fh)
    numeric = {q for q, e in doc["items"].items() if numeric_ranks(e["options"]) is not None}
    rows = [(a, doc["items"][q]["ideal"], v) for run in sorted(doc["open_runs"])
            for q, a, v in doc["open_runs"][run] if q in numeric]
    report["T2_v10_published_grader"] = agreement(rows)

    import bracketing
    rows15 = []
    items15 = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    for run in bracketing.RUNS:
        for r in bracketing.rows_of(run):
            if r["numeric"] and r["condition"] == "data" and r["open"] is not None:
                rows15.append((r["answer"] or "", items15[r["question_id"]]["ideal"], bool(r["open"])))
    report["T2_v15_graders"] = agreement(rows15)
    for name in ("T2_v10_published_grader", "T2_v15_graders"):
        res = report[name]
        best = max(res["rules"].items(), key=lambda kv: kv[1]["kappa"])
        print(f"{name}: {res['n']} answers, grader accepts {res['grader_accepts']}; "
              + ", ".join(f"{k} kappa {v['kappa']:.2f}" for k, v in res["rules"].items())
              + f"; best {best[0]}")
        print("   acceptance by distance: " + ", ".join(
            f"{k}: {v['accepted']}/{v['n']}" for k, v in sorted(res["acceptance_by_distance"].items(),
                                                                 key=lambda kv: float(kv[0].split('-')[0]))))
    report["T3"] = restated()
    t3 = report["T3"]
    for t, e in t3["withdata"].items():
        f = lambda x: f"{x['mean']:+.1f} [{x['lo']:+.1f},{x['hi']:+.1f}]"
        print(f"T3 {t:>3s}: Qwen2.5-72B within with data {e['qwen72b|data']:.1f}%, without {e['qwen72b|nodata']:.1f}%, "
              f"data-nodata {f(e['qwen72b data - nodata'])}; leads Llama {f(e['qwen72b - llama70b'])}, "
              f"gemma {f(e['qwen72b - gemma27b'])}; Table 8 cap last "
              + ("--" if t3['table_8_cap'][t]['last_number'] is None else f"{t3['table_8_cap'][t]['last_number']:.1f}")
              + f", any {t3['table_8_cap'][t]['any_number']:.1f}")
    print("T3 Table 2 regrade: " + "; ".join(f"{k} " + "/".join(f"{v[t]:.1f}" for t in ('1%', '2%', '5%', '10%'))
                                              for k, v in t3["table_1_regrade"].items()))
    for name in ("released_reading", "writers"):
        for k, v in t3[name].items():
            print(f"T3 {name} {k}: " + " / ".join(f"{t} {x['mean']:+.1f}" for t, x in v.items()))
    report["T3_pvalue_log"] = pvalue_log()
    for f, e in report["T3_pvalue_log"].items():
        if f.startswith("x"):
            g = lambda x: f"{x['mean']:+.1f} [{x['lo']:+.1f},{x['hi']:+.1f}]"
            print(f"T3 p-values within {f}: Qwen2.5-72B with data {e['qwen72b|data']:.1f}%, without "
                  f"{e['qwen72b|nodata']:.1f}%, data-nodata {g(e['qwen72b data - nodata'])}; leads Llama "
                  f"{g(e['qwen72b - llama70b'])}, gemma {g(e['qwen72b - gemma27b'])}")
    out = ROOT / "results" / "tolerance_check.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
