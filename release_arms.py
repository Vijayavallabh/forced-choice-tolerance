#!/usr/bin/env python3
r"""Which runs the quoted sentence describes, release by release, and how each was graded.

BixBench's paper says ``we measure the pure recall performance of both models by
asking the BixBench questions without any notebook or other context'' in its
first version (28 February 2025, $296$ questions, release v1.0) and again in its
third (8 October 2025, $205$ questions, release v1.5), and the repository
publishes both releases' no-data runs, each in three arms. The two releases are
easy to conflate, so this lays them side by side:

    release x arm x model -> accuracy, chance, margin, declines, grader

and adds the two checks the comparison needs.

* **The graders differ between releases.** v1.0's open-ended arm was graded by
  one language-model judge on every question; v1.5 grades each question with
  one of three verifiers (exact match, a numeric range, a judge). So each
  release's open answers are also regraded here by one rule that reads no model:
  on the questions whose key is a number or a numeric range, a reply counts if
  *any* number in it is within $5\%$ of the key (or inside the range). That rule
  is deliberately generous -- it credits a reply that merely mentions the value
  -- so it bounds from above how much a strict grader could be hiding.
* **The same rule on the runs where options were deleted by us**
  (``free_response.py``'s dumps, Table~\ref{tab:free}): the last number is what
  that table grades, and the any-number rule says how many right values it
  missed.

    python3 release_arms.py
"""
import argparse
import csv
import glob
import gzip
import json
import pathlib
import re

import numpy as np

import published_repair as pr
from answer_numbers import NUMBER, normalise

PAPER = {"v1.0": "arXiv:2503.00096v1 (28 Feb 2025)", "v1.5": "arXiv:2503.00096v3 (8 Oct 2025)"}
SUMMARY = {"v1.0": "data/external/zero_shot_summary_v10.json",
           "v1.5": "data/external/zero_shot_summary_v15.json"}
RANGE = re.compile(r"^\s*[\(\[]\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*[\)\]]\s*$")
MODELS = ("gpt-4o", "claude-3-5-sonnet-latest")


def value(text):
    """A number written as ``text``, or None; per cent read as written."""
    t = str(text).strip().replace(",", "")
    pct = t.endswith("%")
    t = t.rstrip("%").strip()
    try:
        return float(t), pct
    except ValueError:
        return None


def key_of(target):
    """('point', v, pct) | ('range', lo, hi) | None for a text key."""
    m = RANGE.match(str(target))
    if m:
        try:
            lo, hi = float(m.group(1)), float(m.group(2))
            return ("range", min(lo, hi), max(lo, hi))
        except ValueError:
            return None
    v = value(target)
    return ("point", v[0], v[1]) if v else None


def hits(reply, key, tolerance=0.05, which="any"):
    """Does any (or the last) number in ``reply`` satisfy ``key``?"""
    found = NUMBER.findall(normalise(str(reply or "")))
    if which == "last":
        found = found[-1:]
    for token in found:
        got = value(token)
        if got is None:
            continue
        g = got[0]
        if key[0] == "range":
            if key[1] <= g <= key[2]:
                return True
            continue
        k, kpct = key[1], key[2]
        if kpct and not got[1] and abs(g) <= 1.0:
            g *= 100.0
        if got[1] and not kpct and abs(k) <= 1.0:
            k *= 100.0
        if (abs(g) <= tolerance) if k == 0 else (abs(g - k) <= tolerance * abs(k)):
            return True
    return False


def regrade(rows, reply_field, target_field):
    """Grader's own verdict against the model-free rule, on questions with a numeric key."""
    numeric = [(r, key_of(r[target_field])) for r in rows]
    numeric = [(r, k) for r, k in numeric if k is not None]
    own = [pr.truthy(r["correct"]) for r, _ in numeric]
    anyhit = [hits(r[reply_field], k, 0.05, "any") for r, k in numeric]
    last = [hits(r[reply_field], k, 0.05, "last") for r, k in numeric]
    wide = [hits(r[reply_field], k, 0.10, "any") for r, k in numeric]
    return {"n_numeric_key": len(numeric),
            "grader": 100.0 * float(np.mean(own)) if own else None,
            "last_number_5pct": 100.0 * float(np.mean(last)) if last else None,
            "any_number_5pct": 100.0 * float(np.mean(anyhit)) if anyhit else None,
            "any_number_10pct": 100.0 * float(np.mean(wide)) if wide else None,
            "grader_wrong_but_any_number_5pct": sum(1 for o, a in zip(own, anyhit) if a and not o),
            "grader_right_but_no_number_5pct": sum(1 for o, a in zip(own, anyhit) if o and not a)}


def v10_decline(model):
    """v1.0's decline arm survives only as the summary the repository publishes."""
    doc = json.load(open(SUMMARY["v1.0"]))
    s = doc[f"{model}-grader-mcq-refusal-True"]
    return {"n": s["n_total"], "accuracy": 100.0 * s["accuracy"], "chance": 20.0,
            "declined": 100.0 * (1 - s["coverage"]), "n_answered": s["n_sure"],
            "accuracy_when_answered": 100.0 * s["precision"], "source": "summary only"}


def our_free_runs(pattern="build/dumps/free_*_bixnum.jsonl"):
    """Table~\ref{tab:free}'s free arm on BixBench, last number vs any number."""
    out = {}
    items = [json.loads(l) for l in open("build/bixbench_numeric_q.jsonl")]
    for path in sorted(glob.glob(pattern)):
        rows = [json.loads(l) for l in (gzip.open if path.endswith(".gz") else open)(path, "rt")]
        free = [r for r in rows if r.get("arm") == "free"]
        if not free:
            continue
        keys = [key_of(items[r["item"]]["ideal"]) for r in free]
        last = [hits(r.get("reply", ""), k, 0.05, "last") for r, k in zip(free, keys) if k]
        anyn = [hits(r.get("reply", ""), k, 0.05, "any") for r, k in zip(free, keys) if k]
        out[pathlib.Path(path).stem] = {"n": len(last),
                                        "last_number_5pct": 100.0 * float(np.mean(last)),
                                        "any_number_5pct": 100.0 * float(np.mean(anyn))}
    return out


def attempted(rows):
    """The open arm's accuracy over the replies that do not say the data is missing: how low
    the unaided rate is once the declinations published_repair.DECLINE_TEXT finds are set aside."""
    tried = [r for r in rows if not pr.DECLINE_TEXT.search(str(r["predicted"]))]
    right = sum(1 for r in tried if pr.truthy(r["correct"]))
    return {"n_attempted": len(tried), "n_right": right,
            "n_right_all": sum(1 for r in rows if pr.truthy(r["correct"])),
            "accuracy": 100.0 * right / len(tried) if tried else None}


NAMES = {"gpt-4o": "gpt-4o", "claude-3-5-sonnet-latest": "Claude 3.5 Sonnet"}


def table_rows(report, drift="results/release_drift.json"):
    """tab:arms, one row per release and model, from the report and the rank-law file."""
    law = json.loads(pathlib.Path(drift).read_text())
    ceiling = {"v1.0": law["v1_0"]["rank_ceiling_points"], "v1.5": law["v1_5"]["rank_ceiling_points"]}
    rows = []
    for release in ("v1.0", "v1.5"):
        for model in MODELS:
            e = report["releases"][release][model]
            a = e["arms"]
            n = a["mcq_forced"]["n"]
            rows.append(f"{release} ({n}) & {NAMES[model]} & ${a['mcq_forced']['accuracy']:.1f}$ & "
                        f"${a['mcq_decline']['accuracy']:.1f}$ & ${a['mcq_decline']['declined']:.1f}$ & "
                        f"${a['open_ended']['accuracy']:.1f}$ & ${e['open_regrade']['any_number_5pct']:.1f}$ & "
                        f"${ceiling[release]:+.1f}$\\\\")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--output", default="results/release_arms.json")
    ap.add_argument("--latex", action="store_true", help="print tab:arms' rows from the report")
    args = ap.parse_args()
    if args.latex:
        for line in table_rows(json.loads(pathlib.Path(args.output).read_text())):
            print(line)
        return

    report = {"paper_versions": PAPER, "releases": {}}
    for release, loader in (("v1.0", pr.v10), ("v1.5", pr.v15)):
        report["releases"][release] = {}
        for model in MODELS:
            arms, raw = loader(model)
            entry = {"arms": {n: pr.summarise(r, args.bootstrap) for n, r in arms.items()},
                     "forced_minus_open": pr.paired(arms["mcq_forced"], arms["open_ended"],
                                                    args.bootstrap, 0),
                     "said_no_data": pr.said_no_data(raw["open_ended"]),
                     "open_when_attempted": attempted(raw["open_ended"])}
            if release == "v1.0":
                entry["arms"]["mcq_decline"] = v10_decline(model)
                entry["open_grader"] = "one language-model judge on every question"
                entry["open_regrade"] = regrade(raw["open_ended"], "predicted", "target")
            else:
                d = pr.declined(raw["mcq_decline"])
                entry["arms"]["mcq_decline"].update({"declined": d["rate"], "n_answered": d["n_answered"],
                                                     "accuracy_when_answered": d["accuracy_when_answered"]})
                entry["open_grader"] = "exact match, numeric range or a judge, per question"
                entry["open_regrade"] = regrade(raw["open_ended"], "predicted", "target")
            report["releases"][release][model] = entry
            a = entry["arms"]
            f = entry["forced_minus_open"]
            g = entry["open_regrade"]
            print(f"{release} {model:26s} forced {a['mcq_forced']['accuracy']:5.1f}  decline "
                  f"{a['mcq_decline']['accuracy']:5.1f} ({a['mcq_decline']['declined']:.1f}% declined)  "
                  f"open {a['open_ended']['accuracy']:5.1f}  options worth {f['difference']:+5.1f} "
                  f"[{f['interval'][0]:+.1f},{f['interval'][1]:+.1f}]")
            print(f"      open answers with a numeric key ({g['n_numeric_key']}): grader {g['grader']:.1f}%  "
                  f"last number 5% {g['last_number_5pct']:.1f}%  any number 5% {g['any_number_5pct']:.1f}%  "
                  f"10% {g['any_number_10pct']:.1f}%  (grader wrong, a number right: "
                  f"{g['grader_wrong_but_any_number_5pct']}; grader right, no number: "
                  f"{g['grader_right_but_no_number_5pct']})")
    report["our_free_runs_bixbench"] = our_free_runs()
    for name, s in report["our_free_runs_bixbench"].items():
        print(f"ours {name:32s} last {s['last_number_5pct']:4.1f}%  any {s['any_number_5pct']:4.1f}%  (n={s['n']})")
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
