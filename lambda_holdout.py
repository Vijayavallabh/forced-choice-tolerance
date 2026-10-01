#!/usr/bin/env python3
r"""The proximity weight estimated without the keys it predicts (not registered).

Figure 1c sets each grading's gain on the moved keys, $U-P$, beside its proximity weight
$\lambda=(1-\delta)(qk-1)/(k-1)$ (``proximity_weight.py``), where $q$ and $\delta$ are measured on the misses of
every key and every option set, the moved keys' own gradings through $P$ and $U$ among them. A weight
measured on the same selections as the gain it predicts shares their noise. Here the weight is estimated
three ways that leave the predicted gradings out, and each prediction, $\lambda$ times the gain of the
nearest-option rule's single-nearest indicator on the same answers, is set beside the observed gain:

* ``non-moved keys``: from the grading's selections on the misses of the unchanged and inward keys only,
  every option set pooled (none of the moved keys' gradings is used);
* ``released only``: from its selections through the released options only, every key (none of the
  gradings through $P$ or $U$ is used);
* ``both``: through the released options, on the unchanged and inward keys only.

The gradings are Figure 1c's 46 (``grading_variants_analysis.proximity_cost``), rebuilt as it builds them;
the in-sample weight, observed gain and rule's gain reproduce its points. Two baselines use no weight: every
grading's gain predicted as the rule's gain times the mean share of the rule's gain over the other gradings,
and the same within the grading's kind (code-free, with the notebook, or with the within-5% option).

Intervals resample capsules, one draw per release applied to every grading of its runs (the v1.5 groups
share their questions), and recompute each weight, each gain and every summary: $2{,}000$ draws.

    python3 lambda_holdout.py      # results/lambda_holdout.json
"""
import collections
import json
import pathlib

import numpy as np

import bixbench_withdata as bw
import bracketing as br
import grading_variants as gv
import grading_variants_analysis as gva
import proximity_weight as pw
import replication as rp
import score_decomposition as sd
from answer_numbers import graded

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "lambda_holdout.json"
SEED = 20260927
DRAWS = 2000
K = pw.K
MV = "moved to an edge"
FITS = ("in sample", "non-moved keys", "released only", "both")
BASELINES = ("constant share", "kind share")


# ---------------------------------------------------------------- the gradings

def gradings():
    """(runs, grader, kind, entries) for each of Figure 1c's gradings, built as ``proximity_weight.main`` and
    ``grading_variants_analysis.rank_costs`` build them (the original rewrites)."""
    out = []
    cases = [("v1.0, published", "gpt-4o", lambda: pw.published("gpt-4o")),
             ("v1.0, published", "Qwen2.5-72B", lambda: pw.published("qwen72b")),
             ("v1.0, published", "gemma-3-27b", lambda: pw.published("gemma27b")),
             ("v1.0, published", "Llama-3.3-70B", lambda: pw.published("llama70b")),
             ("v1.0, published", "gemma-3-27b, refusal", lambda: pw.published("gemma27b", "decline")),
             ("v1.5, seven configurations", "own model", lambda: pw.seven("own")),
             ("v1.5, seven configurations", "gemma-3-27b", lambda: pw.seven("gemma27b")),
             ("v1.5, seven configurations", "Qwen2.5-72B", lambda: pw.seven("qwen72b")),
             ("v1.5, seven configurations", "Llama-3.3-70B", lambda: pw.seven("llama70b")),
             ("v1.5, new seeds", "gemma-3-27b", lambda: pw.new_runs("gemma27b", "new seeds")),
             ("v1.5, new seeds", "Qwen2.5-72B", lambda: pw.new_runs("qwen72b", "new seeds")),
             ("v1.5, new seeds", "Llama-3.3-70B", lambda: pw.new_runs("llama70b", "new seeds")),
             ("v1.5, new seeds", "gemma-3-27b, refusal", lambda: pw.new_runs("gemma27b", "new seeds", "decline")),
             ("v1.5, Qwen3-235B-A22B", "gemma-3-27b", lambda: pw.new_runs("gemma27b", "Qwen3-235B-A22B")),
             ("v1.5, Qwen3-235B-A22B", "Qwen2.5-72B", lambda: pw.new_runs("qwen72b", "Qwen3-235B-A22B")),
             ("v1.5, Qwen3-235B-A22B", "Llama-3.3-70B", lambda: pw.new_runs("llama70b", "Qwen3-235B-A22B")),
             ("v1.5, gpt-5.1", "gpt-4o", pw.strong)]
    for runs, grader, load in cases:
        entries = load()
        if entries:
            out.append((runs, grader, "notebook", entries))
    v10, v15 = gv.option_sets()
    numeric10 = {q: s for q, s in v10.items() if "repaired" in s or "repaired_digits" in s}
    numeric15 = {q: s for q, s in v15.items() if "repaired" in s or "repaired_digits" in s}
    families = (("v1.0, published", lambda r: r["set"] == "D1", numeric10),
                ("v1.5, seven configurations", lambda r: r["set"] == "V15" and r["run"] in br.RUNS, numeric15),
                ("v1.5, new seeds", lambda r: r["set"] == "V15" and r["run"].endswith(("|r1", "|r2"))
                 and not r["run"].startswith(("qwen3-235b", "gpt-5.1")), numeric15),
                ("v1.5, Qwen3-235B-A22B", lambda r: r["set"] == "V15" and r["run"].startswith("qwen3-235b"), numeric15),
                ("v1.5, gpt-5.1", lambda r: r["set"] == "V15" and r["run"].startswith("gpt-5.1"), numeric15))
    readings = [(reader, label, "codefree", "forced") for reader, label in gva.GRADERS] + \
               [("gpt-4o", "gpt-4o", "letter", "forced")] + \
               [(reader, label, "codefree", "withintol") for reader, label in gva.GRADERS]
    for reader, label, config, mode in readings:
        rows = gva.load(reader, config, mode)
        if not rows:
            continue
        for fam, want, sets in families:
            sub = [r for r in rows if want(r) and r["condition"] == "data"]
            if not sub or not any("repaired" in r["reads"] for r in sub):
                continue
            s = gva.rank_sets(sets, "original")
            entries = [pw.entry(r["q"], r["run"], r["answer"], s[r["q"]],
                                {a: r["reads"].get(a) for a in ("released", "placebo", "repaired") if a in s[r["q"]]})
                       for r in sub if r["q"] in s]
            entries = [e for e in entries if e["reads"]]
            grading = f"{label}, {config}" + ("" if mode == "forced" else f", {mode}")
            kind = "none within 5%" if mode == "withintol" else "code-free" if config == "codefree" else "notebook"
            out.append((fam, grading, kind, entries))
    return out


# ---------------------------------------------------------------- per-item statistics

def fit_counts(entries, keep_entry, keep_arm):
    """Per item, the counts ``proximity_weight.fit`` pools: misses' reads with a single nearest option, those
    selecting none, those selecting an option, and those selecting the nearest."""
    out = collections.defaultdict(lambda: np.zeros(4))
    for e in entries:
        if not keep_entry(e):
            continue
        for arm, reads in e["reads"].items():
            if not keep_arm(arm):
                continue
            options = e["sets"][arm]
            if graded(e["answer"], options[0], sd.TOL):
                continue
            r = pw.single_nearest_rank(e["answer"], options)
            if r is None:
                continue
            c = out[e["q"]]
            for x in reads:
                c[0] += 1
                if x.get("rank") is None:
                    c[1] += 1
                else:
                    c[2] += 1
                    c[3] += x["rank"] == r
    return out


def weight(n, none, sel, near):
    if not sel or not n:
        return None
    q, delta = near / sel, none / n
    return (1 - delta) * (q * K - 1) / (K - 1)


def moved_items(entries, arm="repaired", base="placebo"):
    """Per moved item: the observed gain, the gain of the single-nearest indicator that the weight scales into a
    prediction (``proximity_weight.predicted``: a correct answer, an empty one, or a miss without a single
    nearest option under either set adds nothing), and the rule's gain, each run set's runs averaged, then
    the run sets, as ``proximity_weight.contrast`` averages them."""
    obs, ind, rul = (collections.defaultdict(lambda: collections.defaultdict(list)) for _ in range(3))
    for e in entries:
        if e["group"] != MV or arm not in e["reads"] or base not in e["reads"]:
            continue
        a_opts, b_opts = e["sets"][arm], e["sets"][base]
        obs[e["q"]][e["run"]].append(float(np.mean([x["correct"] for x in e["reads"][arm]]))
                                     - float(np.mean([x["correct"] for x in e["reads"][base]])))
        d = 0.0
        if e["answer"] and str(e["answer"]).strip() and not graded(e["answer"], a_opts[0], sd.TOL):
            ra, rb = pw.single_nearest_rank(e["answer"], a_opts), pw.single_nearest_rank(e["answer"], b_opts)
            if ra is not None and rb is not None:
                d = float(ra == bw.pick_rank(a_opts[0], a_opts)) - float(rb == bw.pick_rank(b_opts[0], b_opts))
        ind[e["q"]][e["run"]].append(d)
        rul[e["q"]][e["run"]].append(pw.rule(e["answer"], a_opts) - pw.rule(e["answer"], b_opts))
    item = lambda d: {q: float(np.mean([np.mean(v) for v in by_run.values()])) for q, by_run in d.items()}
    return item(obs), item(ind), item(rul)


# ---------------------------------------------------------------- summaries

def ordinal_spearman(x, y):
    """Spearman's rho on ordinal ranks, as Figure 1c's is computed (``proximity_cost``)."""
    rank = lambda v: np.argsort(np.argsort(v))
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def errors(obs, pred):
    e = np.asarray(obs) - np.asarray(pred)
    return {"bias": float(np.mean(e)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2)))}


def predictions(points, subset):
    """{predictor: [prediction per grading in ``subset``]}: each fit's weight times the indicator's gain, and the
    two baselines, which scale the rule's gain by the mean share of it over the other gradings (of the same
    kind, for the second)."""
    out = {f: [np.nan if points[i]["lambda"][f] is None else points[i]["lambda"][f] * points[i]["indicator"]
               for i in subset] for f in FITS}
    share = {i: points[i]["observed"] / points[i]["rule"] for i in subset}
    out["constant share"] = [points[i]["rule"] * float(np.mean([share[j] for j in subset if j != i])) for i in subset]
    out["kind share"] = []
    for i in subset:
        same = [share[j] for j in subset if j != i and points[j]["kind"] == points[i]["kind"]]
        out["kind share"].append(points[i]["rule"] * float(np.mean(same)) if same else np.nan)
    return out


def summarise(points, subset):
    obs = [points[i]["observed"] for i in subset]
    share = [points[i]["observed"] / points[i]["rule"] for i in subset]
    out = {}
    for name, pred in predictions(points, subset).items():
        ok = [k for k, p in enumerate(pred) if np.isfinite(p)]
        o, p = [obs[k] for k in ok], [pred[k] for k in ok]
        out[name] = {"n": len(ok), **errors(o, p), "spearman_pred_obs": ordinal_spearman(p, o)}
        if name in FITS:
            lam = [points[i]["lambda"][name] for i in subset]
            ok = [k for k, v in enumerate(lam) if v is not None]
            out[name]["spearman_lambda_share"] = ordinal_spearman([lam[k] for k in ok], [share[k] for k in ok])
    return out


# ---------------------------------------------------------------- main

def main():
    doc = rp.load_extract()
    cap = {"v1.0": {q: e["capsule"] for q, e in doc["items"].items()},
           "v1.5": {json.loads(l)["question_id"]: json.loads(l)["capsule_uuid"] for l in open(ROOT / "data" / "bixbench.jsonl")}}
    stored = {(p["runs"], p["grader"]): p for p in json.loads((ROOT / "results" / "proximity_cost.json").read_text())["gradings"]}
    keep_entry = {"in sample": lambda e: True, "non-moved keys": lambda e: e["group"] != MV,
                  "released only": lambda e: True, "both": lambda e: e["group"] != MV}
    keep_arm = {"in sample": lambda a: True, "non-moved keys": lambda a: True,
                "released only": lambda a: a == "released", "both": lambda a: a == "released"}
    points, raw = [], []
    for runs, grader, kind, entries in gradings():
        release = runs.split(",")[0]
        counts = {f: fit_counts(entries, keep_entry[f], keep_arm[f]) for f in FITS}
        obs, ind, rul = moved_items(entries)
        lam = {f: weight(*sum(counts[f].values(), np.zeros(4))) for f in FITS}
        mean = lambda d: 100.0 * float(np.mean(list(d.values())))
        f_in = pw.fit(entries)
        c_in = pw.contrast(entries, f_in)
        point = {"runs": runs, "grader": grader, "kind": kind, "n_runs": len(entries), "n_moved_items": len(obs),
                 "lambda": lam, "observed": mean(obs), "rule": mean(rul), "indicator": mean(ind),
                 "n_miss_reads": {f: int(sum(c[0] for c in counts[f].values())) for f in FITS}}
        point["predicted"] = {f: (None if lam[f] is None else lam[f] * point["indicator"]) for f in FITS}
        # the in-sample weight, the observed gain and the rule's gain are Figure 1c's; the in-sample prediction is
        # proximity_weight.contrast's
        ref = stored[(runs, grader)]
        point["reproduces_figure"] = bool(abs(lam["in sample"] - ref["lambda"]) < 1e-9
                                          and abs(point["observed"] - ref["observed"]) < 1e-9
                                          and abs(point["rule"] - ref["rule"]) < 1e-9
                                          and abs(point["predicted"]["in sample"] - c_in["predicted"]) < 1e-9)
        points.append(point)
        raw.append((release, counts, obs, ind, rul))
        p = point["predicted"]
        print(f"{runs:28s} {grader:34s} lambda {lam['in sample']:.3f} / "
              + " / ".join("--" if lam[f] is None else f"{lam[f]:.3f}" for f in FITS[1:])
              + f"  observed {point['observed']:+6.1f}  predicted {p['in sample']:+6.1f} / "
              + " / ".join("--" if p[f] is None else f"{p[f]:+6.1f}" for f in FITS[1:])
              + ("" if point["reproduces_figure"] else "  (DIFFERS FROM FIGURE)"), flush=True)
    everyone = list(range(len(points)))
    common = [i for i in everyone if all(points[i]["lambda"][f] is not None for f in FITS)]
    report = {"seed": SEED, "draws": DRAWS, "not_registered": True, "n_gradings": len(points),
              "all_reproduce_figure": all(p["reproduces_figure"] for p in points),
              "common": [f"{points[i]['runs']}|{points[i]['grader']}" for i in common], "gradings": points}
    report["summary"] = {"common": summarise(points, common)}
    for f in FITS[1:]:
        own = [i for i in everyone if points[i]["lambda"][f] is not None]
        report["summary"][f"defined|{f}"] = summarise(points, own)

    # ---- capsule bootstrap: one draw of capsules per release, applied to every grading of that release's runs
    rng = np.random.default_rng(SEED)
    used = {rel: set() for rel in cap}
    for release, counts, obs, _, _ in raw:
        used[release] |= set(obs) | {q for c in counts.values() for q in c}
    caps = {rel: sorted({cap[rel][q] for q in qs}) for rel, qs in used.items()}
    index = {rel: {c: j for j, c in enumerate(cs)} for rel, cs in caps.items()}
    arrays = []
    for release, counts, obs, ind, rul in raw:
        at = lambda qs: np.array([index[release][cap[release][q]] for q in qs], dtype=int)
        fits = {f: (at(list(counts[f])), np.array([counts[f][q] for q in counts[f]]).reshape(-1, 4)) for f in FITS}
        qs = list(obs)
        arrays.append((release, fits, at(qs), np.array([obs[q] for q in qs]), np.array([ind[q] for q in qs]),
                       np.array([rul[q] for q in qs])))
    draws = {name: collections.defaultdict(list) for name in ("pred", "err")}
    summaries = []
    for _ in range(DRAWS):
        mult = {rel: np.bincount(rng.integers(0, len(cs), len(cs)), minlength=len(cs)).astype(float)
                for rel, cs in caps.items()}
        rep_points = []
        for i, (release, fits, mi, o, d, r) in enumerate(arrays):
            lam = {f: (weight(*(mult[release][idx] @ mat)) if len(idx) else None) for f, (idx, mat) in fits.items()}
            wt = mult[release][mi]
            if wt.sum() == 0:
                rep_points.append(None)
                continue
            m = lambda v: 100.0 * float(wt @ v / wt.sum())
            rep_points.append({"kind": points[i]["kind"], "lambda": lam, "observed": m(o), "rule": m(r),
                               "indicator": m(d)})
        usable = [i for i in common if rep_points[i] is not None
                  and all(rep_points[i]["lambda"][f] is not None for f in FITS) and rep_points[i]["rule"] != 0]
        summaries.append(summarise(rep_points, usable))
        for i, p in enumerate(rep_points):
            if p is None:
                continue
            for f in FITS:
                if p["lambda"][f] is not None:
                    draws["pred"][(i, f)].append(p["lambda"][f] * p["indicator"])
                    draws["err"][(i, f)].append(p["observed"] - p["lambda"][f] * p["indicator"])
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    report["summary_intervals"] = {"common": {
        name: {stat: q([s[name][stat] for s in summaries]) for stat in ("bias", "mae", "rmse", "spearman_pred_obs")}
        for name in FITS + BASELINES}}
    report["summary_intervals"]["common"]["mae difference, constant share minus non-moved keys"] = q(
        [s["constant share"]["mae"] - s["non-moved keys"]["mae"] for s in summaries])
    report["summary_intervals"]["common"]["mae difference, kind share minus non-moved keys"] = q(
        [s["kind share"]["mae"] - s["non-moved keys"]["mae"] for s in summaries])
    report["summary_intervals"]["common"]["mae difference, released only minus non-moved keys"] = q(
        [s["released only"]["mae"] - s["non-moved keys"]["mae"] for s in summaries])
    within = {}
    for f in FITS:
        inside = covered = n = 0
        for i, p in enumerate(points):
            if p["predicted"][f] is None or len(draws["pred"][(i, f)]) < 100:
                continue
            lo, hi = q(draws["pred"][(i, f)])
            elo, ehi = q(draws["err"][(i, f)])
            p.setdefault("prediction_interval", {})[f] = [lo, hi]
            p.setdefault("error_interval", {})[f] = [elo, ehi]
            n += 1
            inside += lo <= p["observed"] <= hi
            covered += elo <= 0.0 <= ehi
        within[f] = {"n": n, "observed_inside_prediction_interval": inside, "error_interval_contains_zero": covered}
    report["within"] = within
    OUT.write_text(json.dumps(report, indent=1) + "\n")

    print(f"\n{len(points)} gradings; all reproduce Figure 1c: {report['all_reproduce_figure']}; "
          f"{len(common)} with every weight defined")
    for block, s in report["summary"].items():
        for name, v in s.items():
            iv = report["summary_intervals"]["common"].get(name) if block == "common" else None
            fmt = lambda stat: (f"{v[stat]:+.1f}" + (f" [{iv[stat][0]:+.1f},{iv[stat][1]:+.1f}]" if iv else ""))
            print(f"{block:24s} {name:16s} n={v['n']:2d} bias {fmt('bias')} MAE {fmt('mae')} RMSE {fmt('rmse')} "
                  f"rho(pred,obs) {v['spearman_pred_obs']:.2f}"
                  + (f" rho(lambda,share) {v['spearman_lambda_share']:.2f}" if "spearman_lambda_share" in v else ""))
    for k, v in report["summary_intervals"]["common"].items():
        if k.startswith("mae difference"):
            print(f"{k}: [{v[0]:+.1f},{v[1]:+.1f}]")
    for f, v in within.items():
        print(f"{f:16s} observed inside the prediction's 95% interval: {v['observed_inside_prediction_interval']} of "
              f"{v['n']}; error interval contains 0: {v['error_interval_contains_zero']} of {v['n']}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
