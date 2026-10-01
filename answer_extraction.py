#!/usr/bin/env python3
r"""How a submitted answer becomes a number, and whether the paper's contrasts depend on it (not registered).

Two readings of an agent's submitted answer take a number from it, and they take it differently:

* the nearest-option rule (``bixbench_withdata.nearest_is_key``, the reading the registered tests fix)
  reads an answer only when the whole answer is a number, once commas and a trailing per cent sign are
  stripped; an answer that is a sentence with a number in it scores zero through every option set,
  so it adds nothing to any contrast between option sets;
* the tolerance (``free_response.graded``) takes the last number in the answer, and reads a key
  written as a per cent against an answer written as a fraction (at most 1 in size) as that fraction
  times 100.

This script counts, per run set, the with-data answers to numeric questions that are empty, hold no
number, are a number, hold one number inside text, or hold several, and re-reads the rule's
headline contrasts four ways:

* ``whole answer``: as registered;
* ``last number``: the rule reading the number the tolerance reads, normalised the same way;
* ``one number``: the last-number rule on the answers that hold exactly one number;
* ``positive keys``: as registered, without the items whose key is zero or negative, where the
  distance of a miss of the other sign is 1 from every option.

Run sets: the published gpt-4o and Claude 3.5 Sonnet runs on v1.0 (``D1``), the seven v1.5 run sets
(``D0``) and the registered new v1.5 runs (``D2``). Contrasts: repaired - placebo and repaired - released
on the keys the repair moved to an edge, and repaired - placebo over every numeric item; plain 95%
percentile cluster bootstraps over capsules (``bixbench_withdata.cluster_interval``).

    python3 answer_extraction.py            # results/answer_extraction.json
    python3 answer_extraction.py --latex    # tab:extraction's rows
"""
import collections
import contextlib
import json
import pathlib

import bixbench_withdata as bw
import bracketing as br
import replication as rp
import answer_numbers as an
from answer_numbers import NUMBER, as_number, normalise
from option_artifacts import parse_number

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "answer_extraction.json"
LEVEL = 0.95
CATEGORIES = ("empty", "no number", "a number", "one number in text", "several numbers")
READINGS = ("whole answer", "last number", "one number", "positive keys")
CONTRASTS = (("repaired", "placebo", "moved to an edge"), ("repaired", "released", "moved to an edge"),
             ("repaired", "placebo", "all"))


def category(answer):
    """Which kind of answer this is, for the purpose of taking a number from it."""
    if not (answer and str(answer).strip()):
        return "empty"
    if parse_number(answer) is not None:
        return "a number"
    found = NUMBER.findall(normalise(str(answer)))
    if not found:
        return "no number"
    return "one number in text" if len(found) == 1 else "several numbers"


def last_number(answer, key):
    """The number the tolerance reads from ``answer`` against ``key``, normalised as it normalises
    a per cent written on one side only; None when there is none (``answer_numbers.last_number``)."""
    return an.last_number(answer, key)


def nearest_is_key_last(answer, options):
    """``bixbench_withdata.nearest_is_key`` on the number the tolerance reads (key first)."""
    a = last_number(answer, options[0]) if answer else None
    values = [parse_number(o) for o in options]
    if a is None or any(v is None for v in values):
        return 0.0
    tied = an.nearest_ranks(a, values)
    return 1.0 / len(tied) if 0 in tied else 0.0


@contextlib.contextmanager
def reading(fn):
    """Every contrast in ``replication`` reads through ``bixbench_withdata.nearest_is_key``: swap it."""
    saved = bw.nearest_is_key
    bw.nearest_is_key = fn
    try:
        yield
    finally:
        bw.nearest_is_key = saved


# ---------------------------------------------------------------- the runs

def published_runs(doc, sets):
    """{run set: [(question, answer)]} of the published open-answer runs on numeric questions."""
    return {name: [(q, a) for q, a, _ in doc["open_runs"][name] if q in sets] for name in rp.OPEN_RUNS}


def seven_runs():
    return {run: [(r["question_id"], r["answer"]) for r in br.rows_of(run)
                  if r["numeric"] and r["condition"] == "data"] for run in br.RUNS}


def new_runs(sets):
    runs2 = rp.d2_runs()
    out = {"new seeds": {}, "Qwen3-235B-A22B": {}}
    for name, conds in runs2.items():
        fam = "Qwen3-235B-A22B" if name.startswith("qwen3-235b|") else "new seeds"
        out[fam][name] = [(q, a) for q, a in conds.get("data", []) if q in sets]
    return out


# ---------------------------------------------------------------- the contrasts

def contrast(runs, sets, capsule, arm, base, group, keep=lambda q, a: True, keep_item=lambda q: True):
    sub = {name: [(q, a) for q, a in trajectories if keep(q, a)] for name, trajectories in runs.items()}
    usable = {q: s for q, s in sets.items() if arm in s and base in s and keep_item(q)}
    _, pooled = rp.gains(sub, usable, arm, base)
    items = sorted(q for q in pooled if group == "all" or rp.group_of(sets[q]) == group)
    return bw.cluster_interval([pooled[q] for q in items], [capsule[q] for q in items], level=LEVEL)


def contrasts(runs, sets, capsule):
    positive = lambda q: parse_number(sets[q]["released"][0]) > 0
    one = lambda q, a: category(a) in ("a number", "one number in text")
    out = {}
    for arm, base, group in CONTRASTS:
        tag = f"{arm}-{base}|{group}"
        out[f"whole answer|{tag}"] = contrast(runs, sets, capsule, arm, base, group)
        out[f"positive keys|{tag}"] = contrast(runs, sets, capsule, arm, base, group, keep_item=positive)
        with reading(nearest_is_key_last):
            out[f"last number|{tag}"] = contrast(runs, sets, capsule, arm, base, group)
            out[f"one number|{tag}"] = contrast(runs, sets, capsule, arm, base, group, keep=one)
    return out


def counts(runs):
    c = collections.Counter(category(a) for trajectories in runs.values() for _, a in trajectories)
    n = sum(c.values())
    return {"n": n, **{k: 100.0 * c[k] / n for k in CATEGORIES}}


# tab:extraction: the answers by kind, then the rule's contrasts read each way
KIND_ROWS = (("v1.0|gpt-4o", "v1.0, gpt-4o"), ("v1.0|Claude 3.5 Sonnet", "v1.0, Claude 3.5 Sonnet"),
             ("v1.5|seven run sets", "v1.5, seven configurations"), ("v1.5|new seeds", "v1.5, new seeds"),
             ("v1.5|Qwen3-235B-A22B", "v1.5, Qwen3-235B-A22B"))
CONTRAST_ROWS = (("v1.0|published", "v1.0, published"), ("v1.5|seven run sets", "v1.5, seven configurations"),
                 ("v1.5|new seeds", "v1.5, new seeds"), ("v1.5|Qwen3-235B-A22B", "v1.5, Qwen3-235B-A22B"))


def table_rows(report):
    count = lambda n: f"{n:,}".replace(",", "{,}")      # the number alone: the label has commas of its own
    kinds = [f"{label} & ${count(report['categories'][key]['n'])}$ & "
             + " & ".join(f"${report['categories'][key][c]:.1f}$" for c in CATEGORIES) + "\\\\"
             for key, label in KIND_ROWS]
    cell = lambda v: f"${v['mean']:+.1f}$"
    contrasts = []
    for key, label in CONTRAST_ROWS:
        got = report["contrasts"][key]
        moved = [cell(got[f"{r}|repaired-placebo|moved to an edge"]) for r in READINGS]
        whole = [cell(got[f"{r}|repaired-placebo|all"]) for r in ("whole answer", "last number")]
        contrasts.append(f"{label} & " + " & ".join(moved + whole) + "\\\\")
    return kinds, contrasts


def main():
    import sys
    if "--latex" in sys.argv:
        kinds, rows = table_rows(json.loads(OUT.read_text()))
        print("\n".join(kinds) + "\n\n" + "\n".join(rows))
        return
    report = {"level": LEVEL, "not_registered": True, "categories": {}, "contrasts": {}}
    doc = rp.load_extract()
    v10 = rp.v10_sets()
    cap10 = {q: e["capsule"] for q, e in doc["items"].items()}
    d1 = published_runs(doc, v10)
    for model, names in (("gpt-4o", rp.OPEN_RUNS[:2]), ("Claude 3.5 Sonnet", rp.OPEN_RUNS[2:])):
        report["categories"][f"v1.0|{model}"] = counts({n: d1[n] for n in names})
    report["contrasts"]["v1.0|published"] = contrasts(d1, v10, cap10)

    v15 = bw.option_sets()
    cap15 = {json.loads(l)["question_id"]: json.loads(l)["capsule_uuid"] for l in open(ROOT / "data" / "bixbench.jsonl")}
    d0 = seven_runs()
    for run in br.RUNS:
        report["categories"][f"v1.5|{run}"] = counts({run: d0[run]})
    report["categories"]["v1.5|seven run sets"] = counts(d0)
    report["contrasts"]["v1.5|seven run sets"] = contrasts(d0, v15, cap15)
    for fam, runs in new_runs(v15).items():
        report["categories"][f"v1.5|{fam}"] = counts(runs)
        report["contrasts"][f"v1.5|{fam}"] = contrasts(runs, v15, cap15)
    report["non_positive_keys"] = {"v1.0": sum(parse_number(s["released"][0]) <= 0 for s in v10.values()),
                                   "v1.5": sum(parse_number(s["released"][0]) <= 0 for s in v15.values())}
    OUT.write_text(json.dumps(report, indent=1) + "\n")

    for k, v in report["categories"].items():
        print(f"{k:32s} n={v['n']:5d} " + " ".join(f"{c} {v[c]:4.1f}" for c in CATEGORIES))
    print(f"keys zero or negative: {report['non_positive_keys']}")
    for block, got in report["contrasts"].items():
        for k, v in got.items():
            if v:
                print(f"{block:26s} {k:48s} {v['mean']:+6.1f} [{v['lo']:+6.1f},{v['hi']:+6.1f}] "
                      f"({v['n_items']} items, {v['n_clusters']} capsules)")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
