#!/usr/bin/env python3
r"""How much of the nearest-option rule's cost of hiding the rank an MCQ grader bears, from how closely it
follows the nearest option.

Proposition 1's trade-off binds a grader that always selects the option nearest the submitted number.
Take a forced grader that, on a miss whose number has one nearest option, selects no option with
probability $\delta$, and otherwise the nearest option with probability $q$ and each of the other $k-1$
with probability $(1-q)/(k-1)$, whatever the options (so it cannot tell which is the key). It accepts
such a miss with probability $(1-\delta)q$ when the key is the nearest option and $(1-\delta)(1-q)/(k-1)$
when it is not, so a change of option set that makes the key the nearest option for a share $\Delta$ of
the runs raises its acceptance by $\lambda\Delta$ with

    lambda = (1 - delta) (q k - 1) / (k - 1),

where the nearest-option rule has $\delta=0$, $q=1$ and $\lambda=1$. This measures $q$ and $\delta$ for
every grading of the same with-data runs on their misses (every option set pooled), predicts each run's
acceptance through $P$ and $U$ from them, and sets the predicted $U-P$ on the moved keys beside the
observed one. An answer with no number, or with a number equidistant from several options, is predicted
to be accepted equally through both sets, and a correct answer at the grading's own rate on correct
answers. BixBench's own published grades, which exist only through $R$, have their weight and prediction
alone, pooled and for each model's runs.

    python3 proximity_weight.py      # results/proximity_weight.json
"""
import collections
import gzip
import json
import pathlib

import numpy as np

import answer_extraction as ax
import bixbench_withdata as bw
import bracketing as br
import published_reads as pr
import replication as rp
import score_decomposition as sd
import answer_numbers as an
from answer_numbers import graded

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "proximity_weight.json"
K = 4
ARMS = ("released", "placebo", "repaired")


def nearest_ranks(answer, options):
    """The ranks of the options nearest the number the tolerance reads: least relative distance, ties broken
    by the absolute distance, so that the option nearest an answer of zero, which every positive option is at
    relative distance 1 from, is the smallest."""
    a = ax.last_number(answer, options[0]) if answer else None
    values = [bw.parse_number(o) for o in options]
    if a is None or any(v is None for v in values):
        return None
    tied = an.nearest_ranks(a, values)
    closest = min(abs(a - values[i]) for i in tied)
    return {bw.pick_rank(options[i], options) for i in tied if abs(a - values[i]) == closest}


def single_nearest_rank(answer, options):
    """The rank of the one option nearest the number the tolerance reads, or None."""
    if not (answer and str(answer).strip()) or ax.category(answer) == "no number":
        return None
    ranks = nearest_ranks(answer, options)
    return next(iter(ranks)) if ranks is not None and len(ranks) == 1 else None


def rule(answer, options):
    """The nearest-option rule on the number the tolerance reads, ties split."""
    ranks = nearest_ranks(answer, options) if answer and ax.category(answer) != "no number" else None
    if not ranks:
        return 0.0
    return (1.0 / len(ranks)) if bw.pick_rank(options[0], options) in ranks else 0.0


def fit(entries):
    """q, delta and the rate on correct answers, from every read of the misses (single nearest option) and
    of the correct answers."""
    near = sel = none = n = 0
    hits = []
    for e in entries:
        for arm, reads in e["reads"].items():
            options = e["sets"][arm]
            if graded(e["answer"], options[0], sd.TOL):
                hits.extend(float(x["correct"]) for x in reads)
                continue
            r = single_nearest_rank(e["answer"], options)
            if r is None:
                continue
            for x in reads:
                n += 1
                if x.get("rank") is None:
                    none += 1
                else:
                    sel += 1
                    near += x["rank"] == r
    q, delta = (near / sel if sel else None), (none / n if n else None)
    lam = (1 - delta) * (q * K - 1) / (K - 1) if q is not None else None
    return {"q": q, "delta": delta, "lambda": lam, "n_miss_reads": n,
            "hit_rate": float(np.mean(hits)) if hits else None}


def predicted(e, arm, f):
    options = e["sets"][arm]
    if not (e["answer"] and str(e["answer"]).strip()):
        return 0.0
    if graded(e["answer"], options[0], sd.TOL):
        return f["hit_rate"]
    r = single_nearest_rank(e["answer"], options)
    if r is None:
        return None                               # predicted equal through both sets: left out of the contrast
    key = bw.pick_rank(options[0], options)
    return (1 - f["delta"]) * (f["q"] if r == key else (1 - f["q"]) / (K - 1))


def contrast(entries, f, arm="repaired", base="placebo", observed=True):
    """On the moved keys: observed, predicted and rule U-P, each run set's runs of an item averaged, then the
    run sets, then the items. A grading read only through $R$ (``observed=False``) has its prediction alone,
    over every run on a moved key."""
    obs, pre, rul = (collections.defaultdict(lambda: collections.defaultdict(list)) for _ in range(3))
    for e in entries:
        if e["group"] != "moved to an edge":
            continue
        if observed and (arm not in e["reads"] or base not in e["reads"]):
            continue
        if not observed and (arm not in e["sets"] or base not in e["sets"]):
            continue
        if observed:
            obs[e["q"]][e["run"]].append(float(np.mean([x["correct"] for x in e["reads"][arm]]))
                                         - float(np.mean([x["correct"] for x in e["reads"][base]])))
        pa, pb = predicted(e, arm, f), predicted(e, base, f)
        pre[e["q"]][e["run"]].append(0.0 if pa is None or pb is None else pa - pb)
        rul[e["q"]][e["run"]].append(rule(e["answer"], e["sets"][arm]) - rule(e["answer"], e["sets"][base]))
    mean = lambda d: 100.0 * float(np.mean([np.mean([np.mean(v) for v in by_run.values()]) for by_run in d.values()]))
    if not pre:
        return None
    return {"n_items": len(pre), "observed": mean(obs) if obs else None, "predicted": mean(pre), "rule": mean(rul)}


def entry(q, run, answer, sets, reads):
    return {"q": q, "run": run, "answer": answer, "sets": sets, "reads": {a: v for a, v in reads.items() if v},
            "group": rp.group_of(sets)}


def published(reader, mode="forced"):
    sets = rp.v10_sets()
    rows = pr.load_rows() if (reader, mode) == ("gemma27b", "forced") else pr.load_rows(pr.rows_path(reader, mode))
    return [entry(r["q"], r["run"], r["answer"], sets[r["q"]], {a: r["reads"].get(a) for a in ARMS if a in sets[r["q"]]})
            for r in rows if r["set"] == "D1" and r["reads"]]


def published_grades(model=None):
    """BixBench's own forced grades of its runs, which exist only through $R$: the option each selected, or none;
    both models' runs, or one model's (``"4o"`` or ``"claude"``, as the run sets are named)."""
    sets = rp.v10_sets()
    out = []
    for r in pr.load_rows():
        if r["set"] != "D1" or (model and not r["run"].startswith(model + "_")):
            continue
        options = sets[r["q"]]["released"]
        picked = r["published"]["picked"]
        rank = bw.pick_rank(picked, options) if picked in options else None
        out.append(entry(r["q"], r["run"], r["answer"], sets[r["q"]],
                         {"released": [{"correct": bool(r["published"]["correct"]), "rank": rank}]}))
    return out


def seven(reader):
    sets = bw.option_sets()
    out = []
    for run in br.RUNS:
        name = br.primary(run) if reader == "own" else reader
        for r in br.rows_of(run):
            if r["condition"] == "data" and r["numeric"]:
                reads = {a: r["reads"].get(f"{name}|{a}|forced") for a in ARMS}
                if any(reads.values()):
                    out.append(entry(r["question_id"], run, r["answer"], sets[r["question_id"]], reads))
    if reader in ("qwen72b", "llama70b"):                   # their reads of the moved keys, by published_reads.py --d0
        for r in pr.load_rows(pr.rows_path(reader, "forced")):
            if r["set"] == "D0" and r["condition"] == "data" and r["reads"]:
                out.append(entry(r["q"], r["run"], r["answer"], sets[r["q"]], {a: r["reads"].get(a) for a in ARMS}))
    return out


def new_runs(reader, family, mode="forced"):
    sets = bw.option_sets()
    rows = pr.load_rows() if (reader, mode) == ("gemma27b", "forced") else pr.load_rows(pr.rows_path(reader, mode))
    want = (lambda n: n.startswith("qwen3-235b|")) if family == "Qwen3-235B-A22B" else (lambda n: not n.startswith("qwen3-235b|"))
    return [entry(r["q"], r["run"], r["answer"], sets[r["q"]], {a: r["reads"].get(a) for a in ARMS})
            for r in rows if r["set"] == "D2" and r["condition"] == "data" and want(r["run"]) and r["reads"]]


def strong():
    import strong_agent as sa
    sets = bw.option_sets()
    return [entry(r["question_id"], f"r{r['rollout']}", r["answer"], sets[r["question_id"]],
                  {a: r["reads"].get(f"{sa.READER}|{a}|forced") for a in ARMS}) for r in sa.load_rows()]


def main():
    cases = [("v1.0, published", "gpt-4o", lambda: published("gpt-4o")),
             ("v1.0, published", "Qwen2.5-72B", lambda: published("qwen72b")),
             ("v1.0, published", "gemma-3-27b", lambda: published("gemma27b")),
             ("v1.0, published", "Llama-3.3-70B", lambda: published("llama70b")),
             ("v1.0, published", "gemma-3-27b, refusal", lambda: published("gemma27b", "decline")),
             ("v1.0, published", "published grades", published_grades),
             ("v1.0, published", "published grades, gpt-4o", lambda: published_grades("4o")),
             ("v1.0, published", "published grades, Claude 3.5 Sonnet", lambda: published_grades("claude")),
             ("v1.5, seven configurations", "own model", lambda: seven("own")),
             ("v1.5, seven configurations", "gemma-3-27b", lambda: seven("gemma27b")),
             ("v1.5, seven configurations", "Qwen2.5-72B", lambda: seven("qwen72b")),
             ("v1.5, seven configurations", "Llama-3.3-70B", lambda: seven("llama70b")),
             ("v1.5, new seeds", "gemma-3-27b", lambda: new_runs("gemma27b", "new seeds")),
             ("v1.5, new seeds", "Qwen2.5-72B", lambda: new_runs("qwen72b", "new seeds")),
             ("v1.5, new seeds", "Llama-3.3-70B", lambda: new_runs("llama70b", "new seeds")),
             ("v1.5, new seeds", "gemma-3-27b, refusal", lambda: new_runs("gemma27b", "new seeds", "decline")),
             ("v1.5, Qwen3-235B-A22B", "gemma-3-27b", lambda: new_runs("gemma27b", "Qwen3-235B-A22B")),
             ("v1.5, Qwen3-235B-A22B", "Qwen2.5-72B", lambda: new_runs("qwen72b", "Qwen3-235B-A22B")),
             ("v1.5, Qwen3-235B-A22B", "Llama-3.3-70B", lambda: new_runs("llama70b", "Qwen3-235B-A22B")),
             ("v1.5, gpt-5.1", "gpt-4o", strong)]
    report = {"k": K, "not_registered": True, "cases": [], "predicted_only": []}
    for runs, grader, load in cases:
        entries = load()
        if not entries:
            continue
        f = fit(entries)
        only_r = grader.startswith("published grades")
        c = contrast(entries, f, observed=not only_r)
        report["cases" if not only_r else "predicted_only"].append(
            {"runs": runs, "grader": grader, "n_runs": len(entries), **f, "moved": c})
        print(f"{runs:28s} {grader:14s} runs {len(entries):5d}  q {f['q']:.3f} delta {f['delta']:.3f} "
              f"lambda {f['lambda']:.3f}  rule {c['rule']:+5.1f}  predicted {c['predicted']:+5.1f}  "
              + (f"observed {c['observed']:+5.1f}  " if c["observed"] is not None else "") + f"({c['n_items']} items)")
    obs = np.array([c["moved"]["observed"] for c in report["cases"]])
    pre = np.array([c["moved"]["predicted"] for c in report["cases"]])
    report["fit"] = {"n": len(obs), "pearson": float(np.corrcoef(obs, pre)[0, 1]),
                     "mean_abs_error": float(np.mean(np.abs(obs - pre)))}
    print(f"observed against predicted over {len(obs)} gradings: r {report['fit']['pearson']:.2f}, "
          f"mean absolute error {report['fit']['mean_abs_error']:.1f} points")
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
