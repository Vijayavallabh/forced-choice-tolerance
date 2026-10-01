#!/usr/bin/env python3
r"""The forced-choice excess of Table~\ref{tab:released}, measured against other references.

Table~\ref{tab:released} sets BixBench's grading of its released options against a 5% tolerance
on the last number in each answer. A reference that reads an answer differently could move that
excess, so the same gradings of the same with-data runs are set against:

* BixBench's own open-answer grading of each run (the v1.0 grader on the published runs, the v1.5
  graders on the seven configurations), which reads the whole answer and uses no extraction of ours;
* the tolerance applied to any number in the answer, the lenient bound for answers that give several;
* the tolerance without the items whose key is a p-value (by the question's wording,
  ``leak_origin.kind``), and with those keys graded on a log scale, within a factor of 2 or of 10.

For the misses whose answer gives no number at all, it asks how often the key's value is written in
the run's notebook, and how often the grading accepts such a miss when it is and when it is not.

Over the models behind Table~\ref{tab:scaling}'s nineteen configurations it asks whether the cost per
miss still falls with accuracy when each model counts once (Spearman over models, exact permutation
p-value).

And it restates, as arithmetic and not as a regrading, what the published forced-choice grading's own
rates would add to a score like the one a 2026 system reports (``sources/cited/bioagents_*``): with
open-ended accuracy ``o``, a grader that accepts a share ``a`` of wrong answers and rejects a share ``r``
of correct ones scores ``o (1 - r) + (1 - o) a``.

Intervals are percentile cluster bootstraps over capsules at the level the double bootstrap finds
covers 95% on the capsules of each cell (``replication.calibrated``), as Table~\ref{tab:released}'s are;
each cell draws from its own generator, seeded by its name, and the tolerance's is Table~\ref{tab:released}'s
own interval (``score_decomposition.py``), so a quantity carries one interval in both tables. Nothing here is
part of the confirmatory test.

    python3 reference_check.py            # results/reference_check.json
    python3 reference_check.py --latex    # Table~\ref{tab:reference}'s rows
"""
import itertools
import json
import math
import pathlib
import re
import sys
import zlib

import numpy as np

import answer_extraction as ax
import bixbench_withdata as bw
import leak_origin as lo
import replication as rp
import score_decomposition as sd
from answer_numbers import NUMBER, as_number, graded, normalise

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "reference_check.json"
TABLE_1 = ROOT / "results" / "score_decomposition.json"
# where score_decomposition.py records each group's gradings: its block and group
TABLE_1_GROUP = {"v1.0|gpt-4o": ("v1.0", "gpt-4o"), "v1.0|Claude 3.5 Sonnet": ("v1.0", "Claude 3.5 Sonnet"),
                 "v1.5|seven configurations": ("v1.5", "pooled")}
TOL = 0.05
LOG_FACTORS = (2.0, 10.0)
READINGS = {"v1.0": ("published forced", "published may-decline"),
            "v1.5": ("family forced", "family may-decline")}
BIOAGENTS = ROOT / "sources" / "cited" / "bioagents_arxiv_2601.12542.txt"
# each of Table~\ref{tab:scaling}'s configurations belongs to one agent model
MODEL_OF = (("4o_", "gpt-4o"), ("claude_", "Claude 3.5 Sonnet"), ("qwen3-235b", "Qwen3-235B-A22B"),
            ("qwen3a3b", "Qwen3-30B-A3B"), ("glm45air", "GLM-4.5-Air"), ("qwen72b", "Qwen2.5-72B"),
            ("llama70b", "Llama-3.3-70B"), ("gemma27b", "gemma-3-27b"))


def graded_any(reply, ideal, tolerance):
    """Is any number in ``reply`` within ``tolerance`` of ``ideal``? The same per-cent and zero-key
    conventions as ``free_response.graded``, which reads the last number only."""
    return any(graded(tok, ideal, tolerance) for tok in NUMBER.findall(normalise(reply or "")))


def graded_log(reply, ideal, factor):
    """The last number in ``reply`` within a factor ``factor`` of ``ideal``, both positive."""
    found = NUMBER.findall(normalise(reply or ""))
    got, key = (as_number(found[-1]) if found else None), as_number(ideal)
    if got is None or key is None or got[0] <= 0 or key[0] <= 0:
        return False
    return max(got[0] / key[0], key[0] / got[0]) <= factor


def kinds():
    """{question id: the kind of quantity its key is}, both releases."""
    return {it["q"]: lo.kind(it["question"]) for it in lo.v10_items() + lo.v15_items()}


def interval(runs, value, capsule, name):
    """The mean over questions of ``value`` and its interval at the level the double bootstrap finds covers
    95% on these capsules, drawn from a generator seeded by the cell's ``name``."""
    d = sd.per_item(runs, value)
    out = rp.calibrated(d, capsule, np.random.default_rng(zlib.crc32(name.encode())), 2000, 5000)
    out["n_items"] = len(d)
    return out


def references(runs, reading, capsule, kind, label):
    """The grading's score minus each reference, and each reference's own score, for one reading. The excess
    over the tolerance is Table~\\ref{tab:released}'s, interval and all."""
    read = [r for r in runs if sd.named_share(r, reading) is not None]
    named = lambda r: sd.named_share(r, reading)
    has = lambda r: bool(r["answer"] and str(r["answer"]).strip())
    last = lambda r, t=TOL: float(bool(graded(r["answer"], r["options"][0], t))) if has(r) else 0.0
    anyn = lambda r: float(graded_any(r["answer"], r["options"][0], TOL)) if has(r) else 0.0
    pval = lambda r: kind[r["q"]] == "p-value"

    def logscale(factor):
        return lambda r: (float(graded_log(r["answer"], r["options"][0], factor)) if has(r) else 0.0) \
            if pval(r) else last(r)

    refs = {"tolerance": last, "any number": anyn}
    if all(r["grader"] is not None for r in read):
        refs["graders"] = lambda r: float(r["grader"])
    for f in LOG_FACTORS:
        refs[f"p-values within x{f:g}"] = logscale(f)
    out = {"n_runs": len(read), "n_items": len({r["q"] for r in read}),
           "n_p_value_items": len({r["q"] for r in read if pval(r)}),
           "score": sd.mean_over_items(sd.per_item(read, named))}
    block, group = TABLE_1_GROUP[label]
    table_1 = json.loads(TABLE_1.read_text())[block][group][reading]["score_minus_tolerance"]
    for name, ref in refs.items():
        value = lambda r, ref=ref: named(r) - ref(r)
        if name == "tolerance":
            excess = dict(table_1, n_items=out["n_items"])
            mean = sd.mean_over_items(sd.per_item(read, value))
            assert abs(excess["mean"] - mean) < 1e-9, (label, reading, excess["mean"], mean)
        else:
            excess = interval(read, value, capsule, f"{label}|{reading}|{name}")
        out[name] = {"reference": sd.mean_over_items(sd.per_item(read, ref)), "excess": excess}
    rest = [r for r in read if not pval(r)]
    out["without p-values"] = {"reference": sd.mean_over_items(sd.per_item(rest, last)),
                               "excess": interval(rest, lambda r: named(r) - last(r), capsule,
                                                  f"{label}|{reading}|without p-values")}
    return out


def no_number_misses(runs, reading, written):
    """Answers that give no number: how often the key's value is in the notebook, and how often the
    grading accepts the answer when it is and when it is not (pooled over runs)."""
    rows = [r for r in runs if ax.category(r["answer"]) == "no number" and sd.named_share(r, reading) is not None
            and written.get(r["key"]) is not None]
    key = [(float(written[r["key"]][0]), sd.named_share(r, reading)) for r in rows]
    base = [float(written[r["key"]][i]) for r in rows for i in (1, 2, 3)]
    w = sum(k for k, _ in key)
    out = {"n_runs": len(rows), "key_written": 100.0 * w / len(rows) if rows else None,
           "distractor_written": 100.0 * float(np.mean(base)) if base else None,
           "accepted": 100.0 * float(np.mean([v for _, v in key])) if key else None,
           "accepted_if_written": 100.0 * sum(k * v for k, v in key) / w if w else None,
           "accepted_if_not": 100.0 * sum((1 - k) * v for k, v in key) / (len(key) - w) if len(key) > w else None,
           "n_key_written": int(w)}
    graders = [r["grader"] for r in rows if r["grader"] is not None]
    out["graders_accept"] = 100.0 * float(np.mean(graders)) if graders else None
    return out


def spearman(x, y):
    """Spearman's rho with tied values given their average rank."""
    from scipy.stats import rankdata
    rx, ry = (rankdata(v, method="average") for v in (x, y))
    return float(np.corrcoef(rx, ry)[0, 1])


def per_model():
    """Table~\\ref{tab:scaling} with each agent model counted once: its configurations averaged."""
    rs = json.loads((ROOT / "results" / "run_set_scaling.json").read_text())["run_sets"]
    groups = {}
    for name, row in rs.items():
        model = next(m for tag, m in MODEL_OF if tag in name)
        groups.setdefault(model, []).append(row)
    fields = ("within_5pct", "moved", "all", "moved_per_miss")
    models = {m: {f: float(np.mean([r[f] for r in rows])) for f in fields} | {"n_configurations": len(rows)}
              for m, rows in sorted(groups.items())}
    x = [v["within_5pct"] for v in models.values()]
    out = {"models": models}
    for f in ("moved", "all", "moved_per_miss"):
        y = [v[f] for v in models.values()]
        rho = spearman(x, y)
        perms = [spearman(x, [y[i] for i in p]) for p in itertools.permutations(range(len(y)))]
        out[f] = {"spearman_rho": rho, "p_value": float(np.mean([abs(r) >= abs(rho) - 1e-12 for r in perms])),
                  "n_models": len(y)}
    return out


def published_rates(report):
    """The published forced grading's share of misses accepted and of correct answers rejected, per model
    (Table~\\ref{tab:released}), and what those rates add to a reported open-ended accuracy."""
    dec = json.loads((ROOT / "results" / "score_decomposition.json").read_text())["v1.0"]
    text = " ".join(BIOAGENTS.read_text(encoding="utf-8").split())
    o = float(re.search(r"([\d.]+)% accuracy on Open Response", text).group(1))
    reported = float(re.search(r"([\d.]+)% on MCQ without Refusal", text).group(1))
    out = {"open_ended": o, "forced_reported": reported, "models": {}}
    for model, readings in dec.items():
        s = readings["published forced"]
        accept = (s["miss_credited_nearest"] + s["miss_credited_no_number"] + s["miss_credited_other"]) \
            / (100.0 - s["tolerance"])
        reject = s["hit_refused"] / s["tolerance"]
        forced = o * (1.0 - reject) + (100.0 - o) * accept
        out["models"][model] = {"accept_miss": 100.0 * accept, "reject_correct": 100.0 * reject,
                                "forced_implied": forced, "gap_implied": forced - o}
    out["gap_reported"] = reported - o
    return out


def analyse():
    kind = kinds()
    written = sd.load_notebook_values()
    report = {"tolerance": TOL, "log_factors": list(LOG_FACTORS), "not_part_of_the_confirmatory_test": True}
    d1 = sd.d1_runs()
    cap1 = {r["q"]: r["capsule"] for r in d1}
    d0 = sd.d0_runs()
    cap0 = {r["q"]: r["capsule"] for r in d0}
    report["p_value_items"] = {"v1.0": len({q for q in {r["q"] for r in d1} if kind[q] == "p-value"}),
                               "v1.5": len({q for q in {r["q"] for r in d0} if kind[q] == "p-value"}),
                               "v1.0 numeric": len({r["q"] for r in d1}), "v1.5 numeric": len({r["q"] for r in d0})}
    report["groups"] = {}
    report["no_number"] = {}
    for label, runs, capsule, block in (("v1.0|gpt-4o", [r for r in d1 if r["group"] == "gpt-4o"], cap1, "v1.0"),
                                        ("v1.0|Claude 3.5 Sonnet", [r for r in d1 if r["group"] == "Claude 3.5 Sonnet"],
                                         cap1, "v1.0"),
                                        ("v1.5|seven configurations", d0, cap0, "v1.5")):
        report["groups"][label] = {reading: references(runs, reading, capsule, kind, label)
                                   for reading in READINGS[block]}
        report["no_number"][label] = no_number_misses(runs, READINGS[block][0], written)
    report["per_model"] = per_model()
    report["published_rates"] = published_rates(report)
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    for label, readings in report["groups"].items():
        for reading, s in readings.items():
            parts = [f"{name} {s[name]['excess']['mean']:+.1f} [{s[name]['excess']['lo']:+.1f},"
                     f"{s[name]['excess']['hi']:+.1f}] (ref {s[name]['reference']:.1f})"
                     for name in ("tolerance", "graders", "any number", "without p-values", "p-values within x2",
                                  "p-values within x10") if name in s]
            print(f"{label:28s} {reading:22s} score {s['score']:.1f} | " + " | ".join(parts))
    for label, s in report["no_number"].items():
        print(f"no number {label}: {s}")
    print("per model:", json.dumps({k: v for k, v in report["per_model"].items() if k != "models"}))
    print("published rates:", json.dumps(report["published_rates"]))
    print(f"p-value items: {report['p_value_items']}")
    print(f"wrote {OUT.relative_to(ROOT)}")


LATEX = (("v1.0|gpt-4o", "published forced", "v1.0, gpt-4o", "forced"),
         ("v1.0|gpt-4o", "published may-decline", "", "with refusal"),
         ("v1.0|Claude 3.5 Sonnet", "published forced", "v1.0, Claude 3.5 Sonnet", "forced"),
         ("v1.0|Claude 3.5 Sonnet", "published may-decline", "", "with refusal"),
         ("v1.5|seven configurations", "family forced", "v1.5, seven configurations", "forced"),
         ("v1.5|seven configurations", "family may-decline", "", "with refusal"))
COLUMNS = ("tolerance", "graders", "any number", "without p-values", "p-values within x10")


def table_rows(report):
    rows = []
    for group, reading, label, how in LATEX:
        s = report["groups"][group][reading]
        cells = []
        for name in COLUMNS:
            e = s[name]["excess"]
            cells.append(f"${e['mean']:+.1f}$ {{\\scriptsize$[{e['lo']:+.1f},{e['hi']:+.1f}]$}}")
        rows.append(f"{label} & {how} & " + " & ".join(cells) + "\\\\")
    return rows


def main():
    if "--latex" in sys.argv:
        for row in table_rows(json.loads(OUT.read_text())):
            print(row)
        return
    analyse()


if __name__ == "__main__":
    main()
