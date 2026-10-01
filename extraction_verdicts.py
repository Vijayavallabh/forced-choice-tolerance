#!/usr/bin/env python3
r"""The pre-specified test's hypotheses with the number read as the tolerance reads it (not registered).

The pre-specified test grades a run by the option nearest its submitted number and takes a number only when
the whole answer is one (``bixbench_withdata.nearest_is_key``); the tolerance, and the paper's other gradings,
take the last number in the answer (``answer_extraction.last_number``, which also reads an answer written as a
fraction against a key written as a per cent as that fraction times 100). This reruns H1-H6
(``PREREGISTRATION.md``; ``replication.test_set``) on every data set of the test twice:

* ``whole answer``: the answers as submitted, which must reproduce ``results/replication.json`` exactly;
* ``last number``: each answer replaced by the number the tolerance reads from it, so that the rule, the
  newly accepted answers' distance bins and their side (H3) all read that number; an answer equidistant from
  several options splits its credit, as the rule does.

Both passes use the test's statistics, calibration levels, pass criteria and random stream (the same seed,
consumed in the same order as ``replication.analyse`` consumes it). Beside each moved-key contrast against $P$
(H4): the capsule sign-flip $p$ (``randomization.signflip``), from the same pooled gains.

    python3 extraction_verdicts.py     # results/extraction_verdicts.json
"""
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import answer_extraction as ax
import bixbench_withdata as bw
import randomization
import replication as rp

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "extraction_verdicts.json"
READINGS = ("whole answer", "last number")
MV = "moved to an edge"
# what each hypothesis reads, as tab:predictions shows it
KEYS = {"H1": "gain|moved to an edge", "H2": "gain|kept", "H4": "repaired-placebo|moved to an edge",
        "H5": "gain|moved inward"}


def as_read(reading, sets):
    """The answer the rule is given: as submitted, or the number the tolerance reads from it, written out."""
    if reading == "whole answer":
        return lambda q, a: a
    def last(q, a):
        if q not in sets or not a:
            return a
        v = ax.last_number(a, sets[q]["released"][0])
        return None if v is None else repr(v)
    return last


def signflip_h4(runs, sets, capsule):
    """The capsule sign-flip test of $U-P$ on the moved keys, from the test's own pooled gains."""
    usable = {q: s for q, s in sets.items() if "repaired" in s and "placebo" in s}
    _, pooled = rp.gains(runs, usable, "repaired", "placebo")
    moved = {q: v for q, v in pooled.items() if rp.group_of(sets[q]) == MV}
    return randomization.signflip(moved, capsule)


def one_pass(reading):
    """H1-H6 on every data set of the test, as ``replication.analyse`` computes them, under one reading."""
    rng = np.random.default_rng(rp.SEED)
    doc = rp.load_extract()
    sets = rp.v10_sets()
    capsule = {q: e["capsule"] for q, e in doc["items"].items()}
    read = as_read(reading, sets)
    runs = {name: [(q, read(q, a)) for q, a, _ in doc["open_runs"][name]] for name in rp.OPEN_RUNS}
    out = {"D1": rp.test_set(runs, sets, capsule, rng)}
    moved = {q for q, s in sets.items() if rp.group_of(s) == MV}
    # the registered analysis draws the single-run spread from the stream here: draw it too, so that the
    # new runs' intervals come from the same place in the stream
    out["D1"]["single_run_spread"] = rp.single_run_spread(runs, sets, moved, capsule, rng)
    out["D1"]["signflip_H4"] = signflip_h4(runs, sets, capsule)
    runs2 = rp.d2_runs()
    sets15 = bw.option_sets()
    items15 = [json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")]
    cap15 = {it["question_id"]: it["capsule_uuid"] for it in items15}
    read15 = as_read(reading, sets15)
    families = {"qwen3-235b": [r for r in runs2 if r.startswith("qwen3-235b|")],
                "reruns": [r for r in runs2 if not r.startswith("qwen3-235b|")]}
    for fam, names in families.items():
        for cond in ("data", "nodata"):
            sub = {n: [(q, read15(q, a)) for q, a in runs2[n][cond]] for n in names if runs2[n].get(cond)}
            if not sub:
                continue
            res = rp.test_set(sub, sets15, cap15, rng, heavy=True)
            if cond == "nodata":
                h = res["gain|moved to an edge"]
                res["verdicts"] = {"H6": bool(h and h["lo"] > 0)}
            res["signflip_H4"] = signflip_h4(sub, sets15, cap15)
            out[f"{fam}|{cond}"] = res
    return out


def same(a, b, tol=1e-9):
    """Two report blocks agree number for number."""
    if isinstance(a, dict) and isinstance(b, dict):
        return all(k in b and same(a[k], b[k], tol) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol
    return a == b


def check_registered(whole):
    """Every hypothesis's estimate, interval, level and verdict under the whole-answer reading against the
    registered report."""
    reg = json.loads((ROOT / "results" / "replication.json").read_text())
    pairs = [("D1", reg["D1"])] + [(k, reg["D2"][k]) for k in ("qwen3-235b|data", "qwen3-235b|nodata",
                                                                "reruns|data", "reruns|nodata")]
    fields = list(KEYS.values()) + ["newly_credited", "verdicts"]
    bad = [f"{name}: {f}" for name, block in pairs for f in fields
           if f in block and not same(block[f], whole[name][f])]
    return {"reproduced": not bad, "mismatches": bad}


def summary(report):
    """{hypothesis: {data set: {reading: estimate, interval and verdict}}}."""
    sets = (("v1.0: gpt-4o, Claude 3.5", "D1", None), ("v1.5: new seeds", "reruns|data", "reruns|nodata"),
            ("v1.5: Qwen3-235B-A22B", "qwen3-235b|data", "qwen3-235b|nodata"))
    out = {}
    for h in ("H1", "H2", "H3", "H4", "H5", "H6"):
        out[h] = {}
        for label, data, nodata in sets:
            cells = {}
            for reading in READINGS:
                res = report[reading]
                if h == "H6":
                    if nodata is None:
                        continue
                    v = res[nodata]["gain|moved to an edge"]
                    cells[reading] = {"mean": v["mean"], "lo": v["lo"], "hi": v["hi"], "level": v["level"],
                                      "level_capped": v["level_capped"], "pass": res[nodata]["verdicts"]["H6"]}
                elif h == "H3":
                    nc = res[data]["newly_credited"]
                    cells[reading] = {"n_gained": nc["n_gained"], "share_over_25pct": nc["share_over_25pct"],
                                      "share_open_side": nc["share_open_side"], "pass": res[data]["verdicts"]["H3"]}
                else:
                    v = res[data][KEYS[h]]
                    cell = {"mean": v["mean"], "lo": v["lo"], "hi": v["hi"], "level": v["level"],
                            "level_capped": v["level_capped"], "pass": res[data]["verdicts"][h]}
                    if h == "H2":
                        # the paper counts a pass by the 5-point clause alone, with an interval that excludes 0, as
                        # a failure (tab:predictions)
                        cell["pass_by_interval"] = bool(v["lo"] <= 0 <= v["hi"])
                        cell["equivalence_shown"] = bool(-5 < v["lo"] and v["hi"] < 5)
                    if h == "H4":
                        sf = res[data]["signflip_H4"]
                        cell["signflip_p"] = sf["p"] if sf else None
                    cells[reading] = cell
            if cells:
                out[h][label] = cells
    return out


NAMES = {"H1": "H1: moved keys gain", "H2": "H2: unchanged keys do not", "H3": "H3: far, on the open side",
         "H4": "H4: not the redrawing", "H5": "H5: inward keys do not gain", "H6": "H6: the gain needs no data"}
SETS = ("v1.0: gpt-4o, Claude 3.5", "v1.5: new seeds", "v1.5: Qwen3-235B-A22B")


def table_rows(report, reading="last number"):
    """tab:lastnumber: each hypothesis on each data set, the number read as the tolerance reads it, with the verdict
    under the stated criterion (H2 counted as failed wherever its interval excludes zero, as tab:predictions does)."""
    rows = []
    for h, name in NAMES.items():
        cells = []
        for label in SETS:
            c = report["summary"][h].get(label, {}).get(reading)
            if c is None:
                cells.append("--")
                continue
            if h == "H3":
                cell = f"${100 * c['share_over_25pct']:.0f}$, ${100 * c['share_open_side']:.0f}\\%$ of ${c['n_gained']}$"
                ok = c["pass"]
            else:
                ci = ("$^{\\ddagger}$" if c["level_capped"] else
                      f" {{\\scriptsize$[{c['lo']:+.1f},{c['hi']:+.1f}]$}}")
                cell = f"${c['mean']:+.1f}${ci}"
                ok = c["pass"] and (h != "H2" or c.get("pass_by_interval", True))
            cells.append(f"{cell} {'pass' if ok else '\\textbf{fail}'}")
        rows.append(f"{name} & " + " & ".join(cells) + "\\\\")
    return rows


def main():
    if "--latex" in sys.argv:
        for row in table_rows(json.loads(OUT.read_text())):
            print(row)
        return
    with ProcessPoolExecutor(2) as ex:
        passes = dict(zip(READINGS, ex.map(one_pass, READINGS)))
    report = {"seed": rp.SEED, "not_registered": True, "readings": list(READINGS),
              "check": check_registered(passes["whole answer"]), **passes}
    report["summary"] = summary(report)
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"whole answer reproduces results/replication.json: {report['check']['reproduced']}"
          + ("" if report["check"]["reproduced"] else f" ({report['check']['mismatches']})"))
    for h, by_set in report["summary"].items():
        for label, cells in by_set.items():
            line = []
            for reading, c in cells.items():
                if h == "H3":
                    line.append(f"{reading}: {100 * c['share_over_25pct']:.0f}, {100 * c['share_open_side']:.0f}% "
                                f"of {c['n_gained']} {'pass' if c['pass'] else 'FAIL'}")
                else:
                    extra = f" p={c['signflip_p']:.4f}" if c.get("signflip_p") is not None else ""
                    cap = " (no level covers 95%)" if c["level_capped"] else ""
                    line.append(f"{reading}: {c['mean']:+.1f} [{c['lo']:+.1f},{c['hi']:+.1f}]{cap} "
                                f"{'pass' if c['pass'] else 'FAIL'}{extra}")
            print(f"{h} {label:26s} " + " | ".join(line))
    print(f"wrote {OUT.relative_to(ROOT)}")
    if not report["check"]["reproduced"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
