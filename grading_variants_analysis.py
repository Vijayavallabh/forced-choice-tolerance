#!/usr/bin/env python3
r"""What the gradings of ``grading_variants.py`` show (``python3 grading_variants.py analyse``).

* **What grading through options adds to a score, over every question.** Each grading's score minus
  BixBench's own open-ended grade of the same runs, over all questions, the numeric ones and the rest,
  beside what a grader that accepts every correct answer and chooses at random among the four options on
  every miss would add, a quarter of the misses. The published runs (every v1.0 question) under
  BixBench's published forced grades, which see the notebook, and under the code-free grader, which does
  not; our seven v1.5 configurations and gpt-5.1 (every v1.5 question) under their graders and the
  code-free one.
* **The cost of hiding the rank under each new grading.** $U-P$ and $U'-P'$ on the keys each rank-uniform
  rewrite moves to an extreme, on the unchanged keys and over all numeric items, pooled over run sets as the
  pre-specified test pools them, with the double-bootstrap level and a capsule sign-flip test; the
  nearest-option rule on the same runs beside them; and the proximity weight
  (``proximity_weight.py``) of each grading.
* **The letter-constrained gpt-4o**, beside the unconstrained one: how often it selects no option, which
  misses it accepts, and its agreement with the published grades.
* **Options with a "none within 5%" option**, read code-free: the misses and correct answers accepted.
* **What the data are worth to each v1.5 configuration**, with the data minus without it on all 205
  questions, under the open-ended graders, the configuration's own forced grader and the code-free one.

Intervals are plain 95% percentile cluster bootstraps over capsules (``bixbench_withdata.cluster_interval``)
except the rank contrasts, which use the pre-specified test's calibrated levels.
"""
import collections
import gzip
import json
import pathlib
import zlib

import numpy as np

import answer_extraction as ax
import bixbench_withdata as bw
import bracketing as br
import digit_matched as dm
import grading_variants as gv
import proximity_weight as pw
import published_reads as pr
import randomization
import replication as rp
import score_decomposition as sd
from answer_numbers import graded

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "grading_variants.json"
SEED = gv.SEED
GRADERS = (("gpt-4o", "gpt-4o"), ("gemma27b", "gemma-3-27b"), ("qwen72b", "Qwen2.5-72B"), ("llama70b", "Llama-3.3-70B"))
MODEL = {"4o_open_image": "gpt-4o", "4o_open_no_image": "gpt-4o",
         "claude_open_image": "Claude 3.5 Sonnet", "claude_open_no_image": "Claude 3.5 Sonnet"}


# The readings taken (``grading_variants.py read``); no other reader, configuration and mode was run.
READ = {("gpt-4o", "codefree", "forced"), ("gemma27b", "codefree", "forced"), ("qwen72b", "codefree", "forced"),
        ("llama70b", "codefree", "forced"), ("gpt-4o", "codefree", "decline"), ("gemma27b", "codefree", "decline"),
        ("gpt-4o", "letter", "forced"), ("gemma27b", "notebook", "forced"), ("qwen72b", "notebook", "forced"),
        ("llama70b", "notebook", "forced"), ("gpt-4o", "codefree", "withintol"),
        ("gemma27b", "codefree", "withintol"), ("qwen72b", "codefree", "withintol")}


def load(reader, config, mode="forced"):
    """The rows of one reader, configuration and mode: the merged file if there is one, else the work
    directory's passes, one row per run with every option set it was read through. A reading never
    taken has no rows; one taken whose rows are missing stops the analysis, rather than dropping that
    grader from the tables."""
    if (reader, config, mode) not in READ:
        return []
    path = gv.rows_path(reader, config, mode)
    if path.exists():
        with gzip.open(path, "rt") as fh:
            return [json.loads(l) for l in fh]
    rows = {}
    for p in sorted(gv.work_dir(reader, config, mode).glob("rows_*.jsonl")):
        for line in p.open():
            r = json.loads(line)
            prior = rows.get(pr.key_of(r))
            if prior is None:
                rows[pr.key_of(r)] = r
            else:
                for arm, reads in r["reads"].items():
                    prior["reads"].setdefault(arm, reads)
    if not rows:
        raise SystemExit(f"no rows for {reader}, {config}, {mode}: neither {path.relative_to(ROOT)} nor "
                         f"{gv.work_dir(reader, config, mode).relative_to(ROOT)}/rows_*.jsonl")
    return list(rows.values())


def score(row, arm):
    if not (row["answer"] and str(row["answer"]).strip()):
        return 0.0
    reads = row["reads"].get(arm)
    return None if not reads else float(np.mean([x["correct"] for x in reads]))


def interval(per_item, capsule):
    qs = sorted(per_item)
    return bw.cluster_interval([per_item[q] for q in qs], [capsule[q] for q in qs], level=0.95)


def by_item(rows, value, group=lambda r: r["run"]):
    """Each question's value: its runs averaged within a run set, then the run sets averaged."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        v = value(r)
        if v is not None:
            acc[r["q"]][group(r)].append(v)
    return {q: float(np.mean([np.mean(v) for v in g.values()])) for q, g in acc.items()}


# ---------------------------------------------------------------- scores over every question

def inflation_block(rows, forced, capsule, numeric_q, by_model=False):
    """Score minus the open-ended grade by partition, with the random-choice reference. The published runs'
    questions pool a model's runs, as Table~\\ref{tab:released} does (``by_model``)."""
    out = {}
    usable = [r for r in rows if r.get("open") is not None and forced(r) is not None]
    group = (lambda r: MODEL[r["run"]]) if by_model else (lambda r: r["run"])
    by_item = lambda rows, value, _b=globals()["by_item"]: _b(rows, value, group)
    for part, keep in (("all", lambda q: True), ("numeric", lambda q: q in numeric_q),
                       ("other", lambda q: q not in numeric_q)):
        sub = [r for r in usable if keep(r["q"])]
        if not sub:
            continue
        excess = by_item(sub, lambda r: forced(r) - float(r["open"]))
        opened = by_item(sub, lambda r: float(r["open"]))
        scored = by_item(sub, forced)
        chance = {q: (1.0 - v) / 4.0 for q, v in opened.items()}
        out[part] = {"n_runs": len(sub), "n_items": len(excess),
                     "score": 100.0 * float(np.mean(list(scored.values()))),
                     "open": 100.0 * float(np.mean(list(opened.values()))),
                     "excess": interval(excess, capsule),
                     "at_chance": 100.0 * float(np.mean(list(chance.values())))}
    return out


def inflation():
    doc = rp.load_extract()
    cap10 = {q: e["capsule"] for q, e in doc["items"].items()}
    numeric10 = set(rp.v10_sets())
    items15 = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    cap15 = {q: it["capsule_uuid"] for q, it in items15.items()}
    numeric15 = set(bw.option_sets())
    out = {"v1.0": {}, "v1.5": {}}
    # the published forced grades of the same runs, an empty answer scored wrong as everywhere else
    with gzip.open(gv.PUBLISHED_ALL, "rt") as fh:
        published = [json.loads(l) for l in fh]
    for model in ("gpt-4o", "Claude 3.5 Sonnet"):
        sub = [dict(r, answer="x" if r["answered"] else "") for r in published if MODEL[r["run"]] == model]
        out["v1.0"][f"{model}|published, with the notebook"] = inflation_block(
            sub, lambda r: float(r["forced"]) if r["answered"] else 0.0, cap10, numeric10, by_model=True)
    for reader, label in GRADERS:
        for mode in ("forced", "decline"):
            rows = [r for r in load(reader, "codefree", mode) if r["set"] == "D1"]
            if not rows:
                continue
            for model in ("gpt-4o", "Claude 3.5 Sonnet"):
                sub = [r for r in rows if MODEL[r["run"]] == model]
                out["v1.0"][f"{model}|code-free {label}, {mode}"] = inflation_block(
                    sub, lambda r: score(r, "released"), cap10, numeric10, by_model=True)
                if mode == "decline":
                    refused = [x["refused"] for r in sub for x in r["reads"].get("released", [])]
                    out["v1.0"][f"{model}|code-free {label}, {mode}"]["refused"] = 100.0 * float(np.mean(refused))
    # v1.5: the seven configurations under their own forced grader (with the notebook) and code-free
    own = []
    for run in br.RUNS:
        name = br.primary(run)
        for r in br.rows_of(run):
            if r["condition"] == "data" and r["rollout"] == 0:
                reads = r["reads"].get(f"{name}|released|forced")
                own.append({"q": r["question_id"], "run": run, "open": r["open"], "answer": r["answer"],
                            "reads": {"released": reads} if reads else {}})
    out["v1.5"]["seven configurations|own model, with the notebook"] = inflation_block(
        own, lambda r: score(r, "released"), cap15, numeric15)
    # gpt-5.1 under gpt-4o with the notebook, as BixBench grades a run (strong_agent.py), once it ran every question
    with gzip.open(ROOT / "results" / "strong_agent_rows.json.gz", "rt") as fh:
        strong = [r for r in json.load(fh) if r["condition"] == "data"]
    if len({r["question_id"] for r in strong}) == len(items15):
        rows = [{"q": r["question_id"], "run": f"gpt-5.1|r{r['rollout']}", "open": r["open"], "answer": r["answer"],
                 "reads": {"released": r["reads"].get("gpt-4o|released|forced")}} for r in strong]
        out["v1.5"]["gpt-5.1|gpt-4o, with the notebook"] = inflation_block(
            rows, lambda r: score(r, "released"), cap15, numeric15)
    for reader, label in GRADERS:
        for mode in ("forced", "decline"):
            rows = [r for r in load(reader, "codefree", mode) if r["set"] == "V15" and r["condition"] == "data"]
            seven = [r for r in rows if r["run"] in br.RUNS]
            strong = [r for r in rows if r["run"].startswith("gpt-5.1")]
            for runs, sub in (("seven configurations", seven), ("gpt-5.1", strong)):
                if not sub:
                    continue
                key = f"{runs}|code-free {label}, {mode}"
                out["v1.5"][key] = inflation_block(sub, lambda r: score(r, "released"), cap15, numeric15)
                if mode == "decline":
                    refused = [x["refused"] for r in sub for x in r["reads"].get("released", [])]
                    out["v1.5"][key]["refused"] = 100.0 * float(np.mean(refused))
    return out


# ---------------------------------------------------------------- the rank's cost

def rank_sets(sets, design):
    """{q: {released, placebo, repaired}} for the original rewrites or the digit-matched ones."""
    if design == "original":
        return {q: {a: s[a] for a in ("released", "placebo", "repaired") if a in s} for q, s in sets.items()}
    return {q: {"released": s["released"], **({"placebo": s["placebo_digits"]} if "placebo_digits" in s else {}),
                **({"repaired": s["repaired_digits"]} if "repaired_digits" in s else {})} for q, s in sets.items()}


def contrasts(rows, sets, capsule, rng, arm_map):
    """U-P by key group and over all items, pooled over run sets, calibrated as the pre-specified test is,
    with a capsule sign-flip test on the moved keys; the rule's the same on the same runs."""
    groups = {g: {q for q, s in sets.items() if rp.group_of(s) == g} for g in rp.GROUPS}
    groups["all"] = set(sets)
    a, b = arm_map["repaired"], arm_map["placebo"]
    per_run = collections.defaultdict(lambda: collections.defaultdict(list))
    rule_run = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        s = sets.get(r["q"])
        if not s or "repaired" not in s or "placebo" not in s:
            continue
        x, y = score(r, a), score(r, b)
        if x is None or y is None:
            continue
        per_run[r["q"]][r["run"]].append(x - y)
        rule_run[r["q"]][r["run"]].append(pw.rule(r["answer"], s["repaired"]) - pw.rule(r["answer"], s["placebo"]))
    pooled = {q: float(np.mean([np.mean(v) for v in g.values()])) for q, g in per_run.items()}
    pooled_rule = {q: float(np.mean([np.mean(v) for v in g.values()])) for q, g in rule_run.items()}
    out = {}
    for g, items in groups.items():
        sub = {q: v for q, v in pooled.items() if q in items}
        out[g] = rp.calibrated(sub, capsule, rng, 2000, 5000) if len({capsule[q] for q in sub}) > 1 else None
        out[f"rule|{g}"] = 100.0 * float(np.mean([pooled_rule[q] for q in sub])) if sub else None
    moved = {q: v for q, v in pooled.items() if q in groups["moved to an edge"]}
    out["signflip"] = randomization.signflip(moved, capsule) if len(moved) > 1 else None
    return out


def rank_costs():
    """Each contrast draws from its own stream, seeded by its key, so that its interval does not depend on which
    other gradings exist."""
    v10, v15 = gv.option_sets()
    doc = rp.load_extract()
    cap10 = {q: e["capsule"] for q, e in doc["items"].items()}
    items15 = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    cap15 = {q: it["capsule_uuid"] for q, it in items15.items()}
    numeric10 = {q: s for q, s in v10.items() if "repaired" in s or "repaired_digits" in s}
    numeric15 = {q: s for q, s in v15.items() if "repaired" in s or "repaired_digits" in s}
    families = (("v1.0, published", lambda r: r["set"] == "D1", numeric10, cap10),
                ("v1.5, seven configurations", lambda r: r["set"] == "V15" and r["run"] in br.RUNS, numeric15, cap15),
                ("v1.5, new seeds", lambda r: r["set"] == "V15" and r["run"].endswith(("|r1", "|r2"))
                 and not r["run"].startswith(("qwen3-235b", "gpt-5.1")), numeric15, cap15),
                ("v1.5, Qwen3-235B-A22B", lambda r: r["set"] == "V15" and r["run"].startswith("qwen3-235b"), numeric15, cap15),
                ("v1.5, gpt-5.1", lambda r: r["set"] == "V15" and r["run"].startswith("gpt-5.1"), numeric15, cap15))
    out = {}
    readings = [(reader, label, "codefree", "forced") for reader, label in GRADERS] + \
               [("gpt-4o", "gpt-4o", "letter", "forced")] + \
               [(reader, label, "notebook", "forced") for reader, label in GRADERS[1:]] + \
               [(reader, label, "codefree", "withintol") for reader, label in GRADERS]
    for reader, label, config, mode in readings:
        rows = load(reader, config, mode)
        if not rows:
            continue
        for fam, want, sets, capsule in families:
            sub = [r for r in rows if want(r) and r["condition"] == "data"]
            if not sub:
                continue
            for design in ("original", "digit-matched"):
                if config == "notebook" and design == "original":
                    continue
                s = rank_sets(sets, design)
                arm_map = ({"repaired": "repaired", "placebo": "placebo"} if design == "original"
                           else {"repaired": "repaired_digits", "placebo": "placebo_digits"})
                if not any(arm_map["repaired"] in r["reads"] for r in sub):
                    continue
                key = f"{fam}|{label}, {config}" + ("" if mode == "forced" else f", {mode}") + f"|{design}"
                out[key] = contrasts(sub, s, capsule, np.random.default_rng([SEED, zlib.crc32(key.encode())]), arm_map)
                out[key]["n_runs"] = len(sub)
                # the proximity weight of this grading on these runs, and its prediction
                entries = [pw.entry(r["q"], r["run"], r["answer"], s[r["q"]],
                                    {a: r["reads"].get(arm_map.get(a, a)) for a in ("released", "placebo", "repaired")
                                     if a in s[r["q"]]}) for r in sub if r["q"] in s]
                entries = [e for e in entries if e["reads"]]
                f = pw.fit(entries)
                c = pw.contrast(entries, f) if f["q"] is not None else None
                out[key]["proximity"] = {**f, "moved": c}
    return out


# ---------------------------------------------------------------- the letter-constrained grader

def letter_summary():
    rows = [r for r in load("gpt-4o", "letter") if r["set"] == "D1"]
    if not rows:
        return None
    v10 = rp.v10_sets()
    quarter = [r for r in rows if pr.chunk_of(r, pr.SAMPLE_OF) in pr.SAMPLE_CHUNKS]
    published = {pr.key_of(r): r for r in pr.load_rows() if r["set"] == "D1"}
    unconstrained = {pr.key_of(r): r for r in pr.load_rows(pr.rows_path("gpt-4o", "forced")) if r["set"] == "D1"}
    reads = [x for r in rows for arm in r["reads"] for x in r["reads"][arm]]
    out = {"n_runs": len(rows), "n_quarter": len(quarter), "no_pick": 100.0 * float(np.mean([x["no_pick"] for x in reads])),
           "server_errors": sum(x["server_error"] for x in reads)}
    # which misses it accepts through R, on the quarter, as Table 2 counts them
    kinds = collections.defaultdict(list)
    kinds_u = collections.defaultdict(list)
    kinds_p = collections.defaultdict(list)          # the published grades of the same runs
    agree = []
    for r in quarter:
        options = v10[r["q"]]["released"]
        if not (r["answer"] and str(r["answer"]).strip()):
            continue
        s = score(r, "released")
        if s is None:
            continue
        base = unconstrained.get(pr.key_of(r))
        su = score(base, "released") if base else None
        pub = published.get(pr.key_of(r))
        sp = float(pub["published"]["correct"]) if pub is not None else None
        if graded(r["answer"], options[0], sd.TOL):
            kinds["correct"].append(s)
            if su is not None:
                kinds_u["correct"].append(su)
            if sp is not None:
                kinds_p["correct"].append(sp)
        else:
            k = sd.single_nearest({"answer": r["answer"], "options": options}) or "neither"
            for kk in ("all misses", k):
                kinds[kk].append(s)
                if su is not None:
                    kinds_u[kk].append(su)
                if sp is not None:
                    kinds_p[kk].append(sp)
        if pub is not None:
            agree.append((float(s == 1.0), float(pub["published"]["correct"])))
    out["accepted"] = {k: 100.0 * float(np.mean(v)) for k, v in kinds.items()}
    out["accepted_unconstrained"] = {k: 100.0 * float(np.mean(v)) for k, v in kinds_u.items()}
    out["accepted_published"] = {k: 100.0 * float(np.mean(v)) for k, v in kinds_p.items()}
    out["n"] = {k: len(v) for k, v in kinds.items()}
    if agree:
        g = np.array([a for a, _ in agree])
        p = np.array([b for _, b in agree])
        po = float(np.mean(g == p))
        pe = float(np.mean(g) * np.mean(p) + (1 - np.mean(g)) * (1 - np.mean(p)))
        out["kappa_published"] = (po - pe) / (1 - pe)
        out["agree_published"] = 100.0 * po
    return out


# ---------------------------------------------------------------- the "none within 5%" option

def withintol_summary():
    out = {}
    v10, v15 = gv.option_sets()
    for reader, label in GRADERS:
        rows = load(reader, "codefree", "withintol")
        if not rows:
            continue
        for fam, want, sets in (("v1.0, published", lambda r: r["set"] == "D1", v10),
                                ("v1.5, with the data", lambda r: r["set"] == "V15" and r["condition"] == "data", v15)):
            sub = [r for r in rows if want(r)]
            if not sub:
                continue
            entry = {}
            for arm in ("released", "repaired"):
                acc = collections.defaultdict(list)
                for r in sub:
                    s = sets[r["q"]].get(arm)
                    reads = r["reads"].get(arm)
                    if not s or not reads or not (r["answer"] and str(r["answer"]).strip()):
                        continue
                    hit = graded(r["answer"], sets[r["q"]]["released"][0], sd.TOL)
                    tag = "correct" if hit else "miss"
                    acc[f"{tag}|accepted"].append(float(np.mean([x["correct"] for x in reads])))
                    acc[f"{tag}|none within"].append(float(np.mean([x.get("refused", False) for x in reads])))
                entry[arm] = {k: 100.0 * float(np.mean(v)) for k, v in acc.items()}
                entry[arm]["n_correct"] = len(acc["correct|accepted"])
                entry[arm]["n_miss"] = len(acc["miss|accepted"])
            out[f"{fam}|{label}"] = entry
    return out


# ---------------------------------------------------------------- the data's worth

def data_worth():
    items15 = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    cap15 = {q: it["capsule_uuid"] for q, it in items15.items()}
    out = {}
    cf = {reader: {pr.key_of(r): r for r in load(reader, "codefree", "forced") if r["set"] == "V15"}
          for reader, _ in GRADERS}
    for run in br.RUNS:
        name = br.primary(run)
        rows = {(r["question_id"], r["condition"]): r for r in br.rows_of(run) if r["rollout"] == 0}
        qs = sorted(q for (q, c) in rows if c == "data" and (q, "nodata") in rows)
        entry = {}
        gradings = {"open-ended": lambda r: None if r["open"] is None else float(r["open"]),
                    "own model, forced": lambda r: score({"answer": r["answer"],
                                                          "reads": {"released": r["reads"].get(f"{name}|released|forced")}},
                                                         "released")}
        for reader, label in GRADERS:
            got = cf[reader]
            gradings[f"code-free {label}"] = (lambda r, got=got: (lambda row: None if row is None else score(row, "released"))(
                got.get(f"V15|{run}|{r['condition']}|{r['question_id']}|0")))
        for g, value in gradings.items():
            diffs = {}
            for q in qs:
                a, b = value(rows[(q, "data")]), value(rows[(q, "nodata")])
                if a is not None and b is not None:
                    diffs[q] = a - b
            if len(diffs) >= 150:
                entry[g] = interval(diffs, cap15)
        out[run] = entry
    return out


def proximity_cost(report):
    """Figure 1c's points: each grading's gain on the moved keys as a share of the nearest-option rule's gain on
    the same answers, against its proximity weight; BixBench's graders from ``proximity_weight.py``, the code-free,
    letter-constrained and "none within 5%" gradings from the rank contrasts above (original rewrites)."""
    points = []
    pwr = json.loads((ROOT / "results" / "proximity_weight.json").read_text())
    for c in pwr["cases"]:
        m = c["moved"]
        points.append({"runs": c["runs"], "grader": c["grader"], "kind": "notebook", "lambda": c["lambda"],
                       "observed": m["observed"], "rule": m["rule"], "share": m["observed"] / m["rule"]})
    for key, c in report["rank"].items():
        fam, grading, design = key.split("|")
        if design != "original" or not c["proximity"].get("moved"):
            continue
        m = c["proximity"]["moved"]
        kind = ("none within 5%" if "withintol" in grading else "code-free" if "codefree" in grading else "notebook")
        points.append({"runs": fam, "grader": grading, "kind": kind, "lambda": c["proximity"]["lambda"],
                       "observed": m["observed"], "rule": m["rule"], "share": m["observed"] / m["rule"]})
    lam = np.array([p["lambda"] for p in points])
    share = np.array([p["share"] for p in points])
    rank = lambda x: np.argsort(np.argsort(x))
    out = {"gradings": points, "n": len(points),
           "spearman": float(np.corrcoef(rank(lam), rank(share))[0, 1]),
           "pearson": float(np.corrcoef(lam, share)[0, 1])}
    (ROOT / "results" / "proximity_cost.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


# ---------------------------------------------------------------- tab:readers2

GRADING_LABELS = {"codefree": "code-free", "letter": "letter only", "withintol": "within $5\\%$"}
# the result files keep "code-free"; the paper calls that grader answer-only
paper_label = lambda label: label.replace("code-free", "answer-only")


def grading_label(grading):
    """'gemma-3-27b, codefree, withintol' -> 'gemma-3-27b, within 5%'; 'gpt-4o, letter' -> 'gpt-4o, letter only'."""
    model, *rest = [x.strip() for x in grading.split(",")]
    return f"{model}, {GRADING_LABELS[rest[-1]]}"


def table2(report):
    """tab:readers2's numbers: on the published runs and the new v1.5 seeds, the nearest-option rule of the
    pre-specified test, BixBench's own grades (their proximity weight and its prediction alone), and every other
    grading, each with its proximity weight, $U-P$ on the moved keys with its interval and a capsule sign-flip $p$,
    the gain its weight predicts, and $U-P$ over all numeric items."""
    mv, al = "repaired-placebo|moved to an edge", "repaired-placebo|all"
    res = lambda name: json.loads((ROOT / "results" / name).read_text())
    pwr = res("proximity_weight.json")
    weight = {(c["runs"], c["grader"]): c for c in pwr["cases"] + pwr["predicted_only"]}
    readers, gem, rep = res("published_reads_readers.json"), res("published_reads.json"), res("replication.json")
    other = lambda n: res(f"published_reads_{n}.json")
    rows = []

    def add(runs, grader, lam, moved, sf, pred, allv):
        rows.append({"runs": runs, "grader": grader, "lambda": lam, "moved": moved, "p": sf["p"] if sf else None,
                     "predicted": pred, "all": allv["mean"] if allv else None})

    def bixbench(runs, block, label, grader):
        w = weight[(runs, grader)]
        add(runs, grader, w["lambda"], block[mv], block["signflip"][f"{label}|{mv}"], w["moved"]["predicted"], block.get(al))

    # the gradings the table shows beside BixBench's: code-free by gpt-4o and gemma-3-27b, gpt-4o constrained to a
    # letter, and gemma-3-27b with the within-5% option; every other one is in tab:variants
    shown = {"gpt-4o, codefree", "gemma-3-27b, codefree", "gpt-4o, letter", "gemma-3-27b, codefree, withintol"}

    def variants(runs):
        out = []
        for key, c in report["rank"].items():
            fam, grading, design = key.split("|")
            if fam == runs and design == "original" and grading in shown and c.get("moved to an edge"):
                prox = c["proximity"]
                out.append((grading_label(grading), prox["lambda"], c["moved to an edge"], c["signflip"],
                            prox["moved"]["predicted"] if prox.get("moved") else None, c["all"]))
        return out

    fam = "v1.0, published"
    add(fam, "nearest option", 1.0, readers["rule"][mv], readers["rule"]["signflip"], None, readers["rule"][al])
    w = weight[(fam, "published grades")]
    add(fam, "published", w["lambda"], None, None, w["moved"]["predicted"], None)
    start = len(rows)
    bixbench(fam, other("gpt-4o_forced")["D1"], "gpt-4o", "gpt-4o")
    bixbench(fam, other("qwen72b_forced")["D1"], "qwen72b", "Qwen2.5-72B")
    bixbench(fam, gem["D1"], "gemma", "gemma-3-27b")
    bixbench(fam, other("llama70b_forced")["D1"], "llama70b", "Llama-3.3-70B")
    bixbench(fam, other("gemma27b_decline")["D1"], "gemma27b", "gemma-3-27b, refusal")
    for v in variants(fam):
        add(fam, *v)
    rows[start:] = sorted(rows[start:], key=lambda r: -r["lambda"])
    # the new seeds: the rule as the pre-specified test pools it, and over all numeric items
    fam = "v1.5, new seeds"
    sets15 = bw.option_sets()
    items15 = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    cap15 = {q: it["capsule_uuid"] for q, it in items15.items()}
    seeds = [r for r in pr.load_rows() if r["set"] == "D2" and r["condition"] == "data" and not r["run"].startswith("qwen3-235b|")]
    rule_all = rp.calibrated(pr.rule_gains(seeds, sets15, "repaired", "placebo"), cap15, np.random.default_rng(SEED), 4000, 20000)
    d2 = gem["D2"]["reruns|data"]
    add(fam, "nearest option", 1.0, rep["D2"]["reruns|data"][mv], d2["signflip"][f"rule|{mv}"], None, rule_all)
    start = len(rows)
    bixbench(fam, other("qwen72b_forced")["D2|reruns|data"], "qwen72b", "Qwen2.5-72B")
    bixbench(fam, d2, "gemma", "gemma-3-27b")
    bixbench(fam, other("llama70b_forced")["D2|reruns|data"], "llama70b", "Llama-3.3-70B")
    bixbench(fam, other("gemma27b_decline")["D2|reruns|data"], "gemma27b", "gemma-3-27b, refusal")
    for v in variants(fam):
        add(fam, *v)
    rows[start:] = sorted(rows[start:], key=lambda r: -r["lambda"])
    return rows


def _pv(p):
    return "--" if p is None else ("$<0.001$" if p < 0.001 else f"${p:.3f}$")


def _ci(v):
    """A mean and its interval; where no nominal level attains 95% coverage on the capsules, the mean alone,
    marked, as the pre-specified test's tables print it."""
    if not v:
        return "--"
    if v.get("level_capped"):
        return f"${v['mean']:+.1f}$$^{{\\ddagger}}$"
    return f"${v['mean']:+.1f}$ {{\\scriptsize$[{v['lo']:+.1f},{v['hi']:+.1f}]$}}"


def table2_rows(report):
    """tab:readers2's rows, one line a grading."""
    out, last = [], None
    for r in report["table2"]:
        runs = r["runs"] if r["runs"] != last else ""
        last = r["runs"]
        lam = "$1$" if r["grader"] == "nearest option" else f"${r['lambda']:.2f}$"
        allv = "--" if r["all"] is None else f"${r['all']:+.1f}$"
        out.append(f"{runs} & {paper_label(r['grader'])} & {lam} & {_ci(r['moved'])} & {_pv(r['p'])} & {allv}\\\\")
    return out


def _excess(b, part, with_ci=False):
    if part not in b:
        return "--"
    e = b[part]["excess"]
    iv = f" {{\\scriptsize$[{e['lo']:+.1f},{e['hi']:+.1f}]$}}" if with_ci else ""
    return f"${e['mean']:+.1f}${iv} {{\\scriptsize$({b[part]['at_chance']:+.1f})$}}"


ALLQ_ORDER = ("published, with the notebook", "own model, with the notebook", "gpt-4o, with the notebook")


def allq_rows(report):
    """tab:allq's rows: every grading of the same runs minus BixBench's open-ended grade, by partition, with what
    a random choice on every miss would add."""
    out = []
    for rel, blocks in (("v1.0", report["inflation"]["v1.0"]), ("v1.5", report["inflation"]["v1.5"])):
        groups = collections.defaultdict(list)
        for name, b in blocks.items():
            runs, grading = name.split("|")
            groups[runs].append((grading, b))
        for runs, gradings in groups.items():
            rank = {label: i for i, (_, label) in enumerate(GRADERS)}
            gradings.sort(key=lambda gb: (gb[0] not in ALLQ_ORDER, "decline" in gb[0],
                                          next((i for m, i in rank.items() if m in gb[0]), 9)))
            for i, (grading, b) in enumerate(gradings):
                label = (grading.replace(", forced", "").replace(", decline", ", refusal")
                         .replace("code-free ", "").replace("own model, with the notebook", "own model")
                         .replace("published, with the notebook", "published"))
                if grading.startswith("code-free"):
                    label += ", code-free" if "refusal" not in label else ""
                    label = label.replace(", refusal", ", code-free, refusal")
                elif grading == "gpt-4o, with the notebook":
                    label = "gpt-4o"
                refused = f"${b['refused']:.0f}$" if "refused" in b else "--"
                out.append(f"{(rel + ', ' + runs) if i == 0 else ''} & {paper_label(label)} & {_excess(b, 'numeric')} & "
                           f"{_excess(b, 'other')} & {_excess(b, 'all', with_ci=True)} & {refused}\\\\")
    return out


def variants_rows(report):
    """tab:variants' rows: for every grading that ``grading_variants.py`` made, $U-P$ and $U'-P'$ on the moved
    keys with a sign-flip $p$, $U-P$ on the unchanged keys and over all numeric items, the proximity weight and
    the gain it predicts."""
    out, last = [], None
    by = collections.defaultdict(dict)
    for key, c in report["rank"].items():
        fam, grading, design = key.split("|")
        by[(fam, grading)][design] = c
    fams = (("v1.0, published", "BixBench's published runs (v1.0)"), ("v1.5, seven configurations", "our seven configurations (v1.5)"),
            ("v1.5, new seeds", "the new seeds (v1.5)"), ("v1.5, Qwen3-235B-A22B", "Qwen3-235B-A22B (v1.5)"),
            ("v1.5, gpt-5.1", "gpt-5.1 (v1.5)"))
    for fam, title in fams:
        keys = sorted((k for k in by if k[0] == fam),
                      key=lambda k: -(by[k].get("original") or by[k]["digit-matched"])["proximity"]["lambda"])
        if keys:
            out.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{title}}}}}\\\\")
        for k in keys:
            o, d = by[k].get("original"), by[k].get("digit-matched")
            base = o or d
            prox = base["proximity"]
            label = grading_label(k[1]) if "notebook" not in k[1] else k[1].split(",")[0]
            pred = f"${prox['moved']['predicted']:+.1f}$" if prox.get("moved") else "--"
            mv = lambda c: "--" if not c else _ci(c["moved to an edge"])
            pv = lambda c: "--" if not c or not c.get("signflip") else _pv(c["signflip"]["p"])
            est = lambda c, g: "--" if not c or not c.get(g) else f"${c[g]['mean']:+.1f}$"
            out.append(f"{paper_label(label)} & ${prox['lambda']:.2f}$ & {mv(o)} & {pv(o)} & {mv(d)} & {pv(d)} & "
                       f"{est(o, 'kept')} & {est(o, 'all')} & {pred}\\\\")
    return out


def main():
    report = {"not_registered": True, "inflation": inflation(), "rank": rank_costs(), "letter": letter_summary(),
              "withintol": withintol_summary(), "data_worth": data_worth()}
    report["proximity_cost"] = {k: v for k, v in proximity_cost(report).items() if k != "gradings"}
    report["table2"] = table2(report)
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    f = lambda h: f"{h['mean']:+.1f} [{h['lo']:+.1f},{h['hi']:+.1f}]" if h else "--"
    for rel, blocks in report["inflation"].items():
        for name, b in blocks.items():
            print(f"{rel} {name:52s} " + "  ".join(f"{p}: {b[p]['score']:.1f}-{b[p]['open']:.1f} = {f(b[p]['excess'])} "
                                                   f"(chance {b[p]['at_chance']:+.1f})" for p in ("all", "numeric", "other") if p in b)
                  + (f"  refused {b['refused']:.1f}" if "refused" in b else ""))
    for key, c in report["rank"].items():
        sf = c.get("signflip")
        prox = c["proximity"]
        print(f"{key:70s} moved {f(c['moved to an edge'])} (rule {c['rule|moved to an edge']:+.1f})"
              + (f" p={sf['p']:.3f}" if sf else "") + f"  unchanged {f(c['kept'])}  all {f(c['all'])}  "
              f"q {prox['q']:.2f} delta {prox['delta']:.2f} lambda {prox['lambda']:.2f}"
              + (f" predicted {prox['moved']['predicted']:+.1f}" if prox.get("moved") else ""))
    if report["letter"]:
        print("letter:", json.dumps({k: v for k, v in report["letter"].items()}, default=str)[:600])
    for k, v in report["withintol"].items():
        print("withintol", k, json.dumps(v)[:400])
    for run, e in report["data_worth"].items():
        print(f"data worth {run:16s} " + "  ".join(f"{g}: {f(h)}" for g, h in e.items()))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    import sys
    if "--latex" in sys.argv:
        which = sys.argv[sys.argv.index("--latex") + 1] if len(sys.argv) > sys.argv.index("--latex") + 1 else "readers2"
        for row in {"readers2": table2_rows, "allq": allq_rows, "variants": variants_rows}[which](json.loads(OUT.read_text())):
            print(row)
    else:
        main()
