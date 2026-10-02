#!/usr/bin/env python3
r"""Does the reading change who wins? Every agent comparison the paper's runs support, read every way.

Two sets of with-data runs, each read several ways:

* **the published v1.0 runs** of gpt-4o and Claude 3.5 Sonnet on the 159 questions with four
  numeric options (four run sets: each model with and without images). Readings: the
  benchmark's own grader of the open answer; the benchmark's own multiple-choice reading
  (the model reading its own run through the released options); the option nearest the
  submitted number through the released, placebo and repaired options; the number within
  1, 5 or 10% of the key; and, where ``published_reads.py`` has run, gemma-3-27b's
  ``MCQ_EVAL_PROMPT`` reading through each option set. The comparison is gpt-4o minus
  Claude, each question's runs averaged within a model and the two run sets of a model
  pooled, paired over the questions, with a percentile cluster bootstrap over capsules.
* **the paper's seven v1.5 run sets** on the 105 numeric items (and, for the benchmark's
  open-answer graders, all 205). Readings: the v1.5 graders, the number within 1, 5 or 10%,
  the nearest option and both language-model readers through each option set. For each
  reading, the ordering of the seven run sets, and Kendall's tau between each reading's
  ordering and the benchmark's two own ones -- its open-answer graders on the same numeric
  items, and its multiple-choice reading (the family reader through the released options)
  -- with a cluster bootstrap interval over capsules.

Every interval is the percentile cluster bootstrap over capsules at the level a double bootstrap finds
covers 95% on these capsules, as ``replication.calibrated`` sets it for a mean. Kendall's tau of seven
run sets takes few values, so a draw often equals the point estimate; coverage is therefore counted
by whether each pseudo-sample's percentile interval holds the estimate, not by the share of draws
below it, and a level the double bootstrap finds below 95% -- where an interval ending on the estimate
counts as covering it -- is raised to 95%, so no interval is narrower than the nominal one.

Not registered.

    python3 ranking_check.py            # results/ranking_check.json
    python3 ranking_check.py --latex    # tab:ranking's rows (no longer in the paper)
"""
import collections
import gzip
import json
import pathlib
import zlib
from itertools import combinations

import numpy as np

import bixbench_withdata as bw
import bracketing as br
import claim_budget as cb
import replication as rp
from answer_numbers import graded

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "ranking_check.json"
TOLS = (0.01, 0.05, 0.10)
REPS = 10000
SEED = 20260925
OUTER, INNER, FINAL = 2000, 2000, 10000     # the tau intervals' double bootstrap


def kendall(a, b):
    """Kendall's tau-b between two score vectors over the same run sets."""
    n = len(a)
    conc = disc = ta = tb = 0
    for i, j in combinations(range(n), 2):
        da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
        if da == 0 and db == 0:
            continue
        if da == 0:
            ta += 1
        elif db == 0:
            tb += 1
        elif da == db:
            conc += 1
        else:
            disc += 1
    denom = np.sqrt((conc + disc + ta) * (conc + disc + tb))
    return float((conc - disc) / denom) if denom else 0.0


def boot_capsules(per_item, capsule, stat, reps=REPS, seed=SEED):
    """Percentile cluster bootstrap of ``stat`` (a function of {item: value}) over capsules."""
    caps = sorted({capsule[q] for q in per_item})
    by = collections.defaultdict(list)
    for q in per_item:
        by[capsule[q]].append(q)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        pick = rng.integers(0, len(caps), len(caps))
        sample = [q for c in pick for q in by[caps[c]]]
        draws.append(stat(sample))
    return draws


def taus(a, b):
    """``kendall`` for each row of two (draws x run sets) arrays of scores."""
    i, j = np.triu_indices(a.shape[1], 1)
    da, db = np.sign(a[:, i] - a[:, j]), np.sign(b[:, i] - b[:, j])
    conc, disc = (da * db > 0).sum(1), (da * db < 0).sum(1)
    ta, tb = ((da == 0) & (db != 0)).sum(1), ((db == 0) & (da != 0)).sum(1)
    denom = np.sqrt((conc + disc + ta) * (conc + disc + tb))
    return np.divide(conc - disc, denom, out=np.zeros(len(denom)), where=denom > 0)


def capsule_sums(tables, items, capsule):
    """For each table ({run: {item: score}}), the (run sets x capsules) sums of the items' scores; and
    the capsules' item counts."""
    caps = sorted({capsule[q] for q in items})
    at = {c: i for i, c in enumerate(caps)}
    counts = np.zeros(len(caps))
    for q in items:
        counts[at[capsule[q]]] += 1
    sums = []
    for table in tables:
        s = np.zeros((len(br.RUNS), len(caps)))
        for q in items:
            for k, run in enumerate(br.RUNS):
                s[k, at[capsule[q]]] += table[run][q]
        sums.append(s)
    return sums, counts


def resampled(sums, counts, idx, stat):
    """``stat`` of each resample (a row of capsule indices): each run set's mean over the items drawn,
    rounded so that run sets with equal scores tie as ``kendall`` sees them tie."""
    total = counts[idx].sum(-1)
    return stat(*[np.round((s[:, idx].sum(-1) / total).T, 12) for s in sums])


def calibrated_tau(tables, items, capsule, stat, label):
    """``stat`` (of the tables' per-run-set means) with its calibrated percentile cluster bootstrap
    interval over capsules."""
    rng = np.random.default_rng([SEED, zlib.crc32(label.encode())])
    sums, counts = capsule_sums(tables, items, capsule)
    n = len(counts)
    theta = float(resampled(sums, counts, np.arange(n)[None, :], stat)[0])
    tails = (1.0 - cb.LEVELS) / 2.0
    covered = np.zeros(len(cb.LEVELS))
    for _ in range(OUTER):
        pick = rng.integers(0, n, n)
        inner = resampled([s[:, pick] for s in sums], counts[pick], rng.integers(0, n, (INNER, n)), stat)
        lo, hi = np.percentile(inner, 100.0 * tails), np.percentile(inner, 100.0 * (1.0 - tails))
        covered += (lo <= theta + 1e-12) & (theta - 1e-12 <= hi)
    curve = covered / OUTER
    reached = np.nonzero(curve >= 0.95)[0]
    capped = not len(reached)
    level = float(cb.LEVELS[-1] if capped else cb.LEVELS[reached[0]])
    # never narrower than the nominal 95% interval: a draw often equals the estimate, and an interval that ends on
    # it counts as covering it, so the double bootstrap can find a level far below 95% sufficient
    floored = level < 0.95
    level = max(level, 0.95)
    final = resampled(sums, counts, rng.integers(0, n, (FINAL, n)), stat)
    lo, hi = cb.interval(final, level)
    return {"point": theta, "lo": lo, "hi": hi, "level": level, "level_capped": capped, "level_floored": floored,
            "coverage_of_nominal_95": float(curve[int(np.argmin(np.abs(cb.LEVELS - 0.95)))]),
            "ci95_nominal": cb.interval(final, 0.95), "n_items": len(items), "n_clusters": n}


# ---------------------------------------------------------------- v1.0, published

def v10(reads=None):
    doc = rp.load_extract()
    sets = rp.v10_sets()
    capsule = {q: e["capsule"] for q, e in doc["items"].items()}
    model = {"4o_open_image": "gpt-4o", "4o_open_no_image": "gpt-4o",
             "claude_open_image": "claude", "claude_open_no_image": "claude"}
    # every run: (model, run set, question, {reading: score})
    runs = []
    mcq = {run: collections.defaultdict(list) for run in rp.OPEN_RUNS}
    for run, m in rp.MCQ_RUNS.items():
        for q, picked, ok in doc["mcq_reads"][m]:
            mcq[run][q].append(float(ok))
    seen = collections.Counter()
    for run in rp.OPEN_RUNS:
        for q, answer, ok in doc["open_runs"][run]:
            if q not in sets:
                continue
            i = seen[(run, q)]
            seen[(run, q)] += 1
            s = {"grader": float(ok), "published MCQ reading": mcq[run][q][i]}
            for arm in ("released", "placebo", "repaired"):
                if arm in sets[q]:
                    s[f"nearest|{arm}"] = bw.nearest_is_key(answer, sets[q][arm])
            for t in TOLS:
                s[f"within {round(100 * t)}%"] = float(bool(answer) and bool(graded(answer, sets[q]["released"][0], t)))
            if reads is not None:
                row = reads.get((run, q, i))
                if row is not None:
                    for arm in ("released", "placebo", "repaired"):
                        r = row["reads"].get(arm)
                        s[f"gemma|{arm}"] = (float(np.mean([x["correct"] for x in r])) if r
                                             else 0.0 if not row["reads"] else None)
            runs.append((model[run], run, q, s))
    readings = sorted({k for *_, s in runs for k in s})
    out = {"readings": readings, "n_runs": len(runs), "per_run_set": {}, "gpt4o_minus_claude": {}}
    for reading in readings:
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for m, run, q, s in runs:
            if s.get(reading) is not None:
                per[run][q].append(s[reading])
        out["per_run_set"][reading] = {run: 100.0 * float(np.mean([np.mean(v) for v in per[run].values()]))
                                       for run in rp.OPEN_RUNS}
        # gpt-4o minus Claude per question: each model's runs on it averaged over both its run sets
        by_model = collections.defaultdict(lambda: collections.defaultdict(list))
        for m, run, q, s in runs:
            if s.get(reading) is not None:
                by_model[m][q].append(s[reading])
        qs = sorted(set(by_model["gpt-4o"]) & set(by_model["claude"]))
        diff = {q: float(np.mean(by_model["gpt-4o"][q]) - np.mean(by_model["claude"][q])) for q in qs}
        got = rp.calibrated(diff, capsule, np.random.default_rng([SEED, zlib.crc32(reading.encode())]), 2000, 5000)
        out["gpt4o_minus_claude"][reading] = {"mean": 100.0 * float(np.mean(list(diff.values()))),
                                              "lo": got["lo"], "hi": got["hi"], "level": got["level"],
                                              "level_capped": got["level_capped"], "n_items": len(qs)}
    return out


# ---------------------------------------------------------------- v1.5, the seven run sets

def v15():
    sets = bw.option_sets()
    capsule, scores = {}, collections.defaultdict(dict)       # reading -> run -> {item: score}
    for run in br.RUNS:
        own = br.primary(run)
        for r in br.rows_of(run):
            if r["condition"] != "data":
                continue
            q = r["question_id"]
            capsule[q] = r["capsule"]
            if r["open"] is not None:
                scores["graders, all 205"].setdefault(run, {})[q] = float(r["open"])
            if not r["numeric"]:
                continue
            if r["open"] is not None:
                scores["graders"].setdefault(run, {})[q] = float(r["open"])
            ideal = sets[q]["released"][0]
            for t in TOLS:
                scores[f"within {round(100 * t)}%"].setdefault(run, {})[q] = float(
                    bool(r["answer"]) and bool(graded(r["answer"], ideal, t)))
            for arm in ("released", "placebo", "repaired"):
                scores[f"nearest|{arm}"].setdefault(run, {})[q] = bw.nearest_is_key(r["answer"], sets[q][arm])
                for name, reader in (("own", own), ("gemma", "gemma27b")):
                    v = br.read_score(r, reader, arm)
                    if v is not None:
                        scores[f"{name}|{arm}"].setdefault(run, {})[q] = v
    out = {"references": ["graders", "own|released"], "readings": sorted(scores), "mean": {}, "order": {}, "tau": {}}
    for reading, per in scores.items():
        out["mean"][reading] = {run: 100.0 * float(np.mean(list(per[run].values()))) for run in br.RUNS}
        out["order"][reading] = sorted(br.RUNS, key=lambda run: -out["mean"][reading][run])
    out["against_tolerance"] = against_tolerance(scores, capsule)
    for reference in out["references"]:
        ref = scores[reference]
        out["tau"][reference] = {}
        for reading, per in scores.items():
            items = sorted(set.intersection(*(set(per[run]) for run in br.RUNS), *(set(ref[run]) for run in br.RUNS)))
            point = kendall([np.mean([per[run][q] for q in items]) for run in br.RUNS],
                            [np.mean([ref[run][q] for q in items]) for run in br.RUNS])
            got = calibrated_tau((per, ref), items, capsule, taus, f"tau|{reference}|{reading}")
            assert abs(got["point"] - point) < 1e-9, (reading, got["point"], point)
            out["tau"][reference][reading] = {"tau": point, "lo": got["lo"], "hi": got["hi"], "level": got["level"],
                                              "level_capped": got["level_capped"], "ci95_nominal": got["ci95_nominal"],
                                              "n_items": len(items)}
    return out


NOISE_DRAWS = 1000


def against_tolerance(scores, capsule, reference="graders", tolerance="within 5%"):
    """Does a reading through options lose the graders' order more than noise would? For each
    reading, the paired bootstrap of tau(tolerance) - tau(reading) against the graders, on the
    same capsule draws; and tau for the tolerance with the reading's own noise added -- every
    miss credited at the reading's pooled rate of crediting misses, every hit refused at its
    pooled rate of refusing hits, at random -- so that a reading which credits misses as often as
    it does, but blindly, is the benchmark it is held to."""
    ref, tol = scores[reference], scores[tolerance]
    rng = np.random.default_rng(SEED)
    out = {}
    for reading, per in scores.items():
        if not (reading.startswith("nearest|") or reading.startswith("own|") or reading.startswith("gemma|")):
            continue
        items = sorted(set.intersection(*(set(per[run]) for run in br.RUNS), *(set(ref[run]) for run in br.RUNS),
                                        *(set(tol[run]) for run in br.RUNS)))

        def tau_of(table, sample):
            return kendall([np.mean([table[run][q] for q in sample]) for run in br.RUNS],
                           [np.mean([ref[run][q] for q in sample]) for run in br.RUNS])

        got = calibrated_tau((tol, per, ref), items, capsule, lambda t, p, r: taus(t, r) - taus(p, r),
                             f"difference|{reading}")
        assert abs(got["point"] - (tau_of(tol, items) - tau_of(per, items))) < 1e-9, reading
        misses = [per[run][q] for run in br.RUNS for q in items if tol[run][q] == 0.0]
        hits = [per[run][q] for run in br.RUNS for q in items if tol[run][q] == 1.0]
        credit, refuse = float(np.mean(misses)), float(1.0 - np.mean(hits)) if hits else 0.0
        noisy = []
        for _ in range(NOISE_DRAWS):
            table = {run: {q: float(rng.random() < (1.0 - refuse if tol[run][q] == 1.0 else credit)) for q in items}
                     for run in br.RUNS}
            noisy.append(tau_of(table, items))
        point = tau_of(per, items)
        out[reading] = {"tau": point, "tau_tolerance": tau_of(tol, items),
                        "difference": {"mean": tau_of(tol, items) - point, "lo": got["lo"], "hi": got["hi"],
                                       "level": got["level"], "level_capped": got["level_capped"],
                                       "ci95_nominal": got["ci95_nominal"]},
                        "miss_credit_rate": credit, "hit_refusal_rate": refuse,
                        "noise_matched": {"median": float(np.median(noisy)),
                                          "lo": float(np.percentile(noisy, 2.5)), "hi": float(np.percentile(noisy, 97.5)),
                                          "share_at_or_below": float(np.mean([t <= point for t in noisy]))},
                        "n_items": len(items)}
    return out


# tab:ranking (no longer in the paper): one row per reading; the v1.5 column is Kendall's tau of the seven run sets' order under
# that reading against the benchmark's open-answer graders on the same numeric items, the v1.0 column
# gpt-4o minus Claude 3.5 Sonnet on the published runs
ROWS = (("BixBench's graders", "graders", "grader"),
        ("its multiple-choice grades, $R$", None, "published MCQ reading"),
        ("within $1\\%$", "within 1%", "within 1%"), ("within $5\\%$", "within 5%", "within 5%"),
        ("within $10\\%$", "within 10%", "within 10%"),
        ("nearest option, $R$", "nearest|released", "nearest|released"),
        ("nearest option, $P$", "nearest|placebo", "nearest|placebo"),
        ("nearest option, $U$", "nearest|repaired", "nearest|repaired"),
        ("own model, $R$", "own|released", None), ("own model, $P$", "own|placebo", None),
        ("own model, $U$", "own|repaired", None),
        ("gemma-3-27b, $R$", "gemma|released", "gemma|released"),
        ("gemma-3-27b, $P$", "gemma|placebo", "gemma|placebo"),
        ("gemma-3-27b, $U$", "gemma|repaired", "gemma|repaired"))


def table_rows(report):
    tau = report["v1.5"]["tau"]["graders"]
    vs = report["v1.5"]["against_tolerance"]
    g = report["v1.0"]["gpt4o_minus_claude"]
    rows = []
    for label, k15, k10 in ROWS:
        a = ("the reference" if k15 == "graders" else
             f"${tau[k15]['tau']:+.2f}$ {{\\scriptsize$[{tau[k15]['lo']:+.2f},{tau[k15]['hi']:+.2f}]$}}"
             if k15 else "--")
        if k15 in vs:
            d, n = vs[k15]["difference"], vs[k15]["noise_matched"]
            c = f"${d['mean']:+.2f}$ {{\\scriptsize$[{d['lo']:+.2f},{d['hi']:+.2f}]$}}"
            e = f"${n['median']:+.2f}$ ({n['share_at_or_below']:.2f})"
        else:
            c = e = "--"
        b = (f"${g[k10]['mean']:+.1f}$ {{\\scriptsize$[{g[k10]['lo']:+.1f},{g[k10]['hi']:+.1f}]$}}"
             if k10 and k10 in g else "--")
        rows.append(f"{label} & {a} & {c} & {e} & {b}\\\\")
    return rows


def main():
    import sys
    if "--latex" in sys.argv:
        for row in table_rows(json.loads(OUT.read_text())):
            print(row)
        return
    reads = None
    path = ROOT / "results" / "published_reads_rows.jsonl.gz"
    if path.exists():
        reads = {}
        for line in gzip.open(path, "rt"):
            r = json.loads(line)
            if r["set"] == "D1":
                reads[(r["run"], r["q"], r["i"])] = r
    report = {"seed": SEED, "not_registered": True, "v1.0": v10(reads), "v1.5": v15()}
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    g = report["v1.0"]["gpt4o_minus_claude"]
    print("v1.0 published runs, gpt-4o minus Claude 3.5 Sonnet, 159 numeric questions:")
    for reading, v in g.items():
        print(f"   {reading:24s} {v['mean']:+6.1f} [{v['lo']:+6.1f},{v['hi']:+6.1f}]  per run set "
              + ", ".join(f"{k} {x:.1f}" for k, x in report['v1.0']['per_run_set'][reading].items()))
    t = report["v1.5"]
    print("v1.5 seven run sets, tau(within 5%) - tau(reading) against the graders, and the noise-matched tolerance:")
    for reading, v in t["against_tolerance"].items():
        d, n = v["difference"], v["noise_matched"]
        print(f"   {reading:18s} tau {v['tau']:+.2f}  difference {d['mean']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}]  "
              f"noise-matched {n['median']:+.2f} [{n['lo']:+.2f},{n['hi']:+.2f}] (reading at or below: "
              f"{n['share_at_or_below']:.2f}; misses credited {100 * v['miss_credit_rate']:.1f}%, "
              f"hits refused {100 * v['hit_refusal_rate']:.1f}%)")
    for reference in t["references"]:
        print(f"v1.5 seven run sets, Kendall tau against {reference}:")
        for reading, v in t["tau"][reference].items():
            print(f"   {reading:18s} tau {v['tau']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]  order "
                  + " > ".join(t["order"][reading]))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
