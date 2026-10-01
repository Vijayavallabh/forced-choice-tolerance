#!/usr/bin/env python3
"""Measure how much of a biology-agent benchmark is answerable without its data.

The BixBench authors publish per-question zero-shot baselines: the same
questions answered by a language model that never receives the analysis
capsule. Those runs are a no-data solvability probe that the benchmark already
ships. This script joins them to the pinned question file and to the
provenance audit, then separates two competing explanations for above-chance
no-data accuracy:

  (a) answer-option structure that a model can exploit without any biology, and
  (b) familiarity with the source study.

Inference is clustered on capsule_uuid throughout, because questions drawn
from one source study are not independent observations.
"""
import argparse
import ast
import csv
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path

CHANCE_MCQ = 0.25  # ideal answer + three distractors, no refusal option
NUM_RE = re.compile(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$")


# --------------------------------------------------------------------------
# loading and joining
# --------------------------------------------------------------------------
def load_questions(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["question_id"].lower()] = row
    return rows


def load_zero_shot(path):
    """Read one published baseline CSV into {question_id: correct}.

    Question ids are lowercased before joining: two v1.5 rows are published as
    ``Bix-33-q6`` and ``Bix-47-q3`` while the question file spells every id in
    lower case, and a case-sensitive join silently drops them.
    """
    out = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            qid = (row.get("uuid") or "").strip()
            if not qid.lower().startswith("bix"):
                # v1.0 files key on capsule uuid and carry short_qid instead.
                qid = (row.get("short_qid") or "").strip().replace("_", "-")
            if not qid:
                continue
            out[qid.lower()] = str(row.get("correct", "")).strip().lower() == "true"
    return out


def capsule_of(questions, qid):
    row = questions.get(qid)
    return row["capsule_uuid"] if row else None


# --------------------------------------------------------------------------
# cluster-robust inference
# --------------------------------------------------------------------------
def cluster_bootstrap_ci(clusters, statistic, reps=10000, alpha=0.05, seed=20260918):
    """Percentile CI from resampling whole clusters with replacement."""
    rng = random.Random(seed)
    keys = list(clusters)
    draws = []
    for _ in range(reps):
        sample = [clusters[rng.choice(keys)] for _ in keys]
        value = statistic([v for group in sample for v in group])
        if value is not None:
            draws.append(value)
    draws.sort()
    lo = draws[max(0, int(math.floor((alpha / 2) * len(draws))))]
    hi = draws[min(len(draws) - 1, int(math.ceil((1 - alpha / 2) * len(draws))) - 1)]
    return lo, hi


def mean(values):
    return statistics.fmean(values) if values else None


def group_by_capsule(per_question, questions, subset=None):
    clusters = {}
    for qid, correct in per_question.items():
        cap = capsule_of(questions, qid)
        if cap is None or (subset is not None and cap not in subset):
            continue
        clusters.setdefault(cap, []).append(1.0 if correct else 0.0)
    return clusters


def icc_and_design_effect(clusters):
    """One-way random-effects ICC on binary outcomes, with the design effect.

    deff = 1 + (m0 - 1) * ICC is the factor by which clustering inflates the
    variance of a pooled mean relative to independent sampling.
    """
    groups = [v for v in clusters.values() if v]
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    if k < 2 or n_total <= k:
        return None
    grand = sum(sum(g) for g in groups) / n_total
    ms_between = sum(len(g) * (mean(g) - grand) ** 2 for g in groups) / (k - 1)
    ms_within = sum(sum((x - mean(g)) ** 2 for x in g) for g in groups) / (n_total - k)
    m0 = (n_total - sum(len(g) ** 2 for g in groups) / n_total) / (k - 1)
    denom = ms_between + (m0 - 1) * ms_within
    icc = 0.0 if denom == 0 else max(0.0, (ms_between - ms_within) / denom)
    m_bar = n_total / k
    return {"icc": icc, "mean_cluster_size": m_bar, "m0": m0,
            "design_effect": 1 + (m_bar - 1) * icc,
            "effective_n": n_total / (1 + (m_bar - 1) * icc),
            "n_questions": n_total, "n_clusters": k}


def cluster_permutation_diff(clusters_a, clusters_b, reps=10000, seed=20260918):
    """Two-sample difference in means, permuting whole clusters between arms."""
    rng = random.Random(seed)
    pool = [("a", v) for v in clusters_a.values()] + [("b", v) for v in clusters_b.values()]
    n_a = len(clusters_a)

    def diff(assignment):
        arm_a = [x for lab, g in assignment if lab == "a" for x in g]
        arm_b = [x for lab, g in assignment if lab == "b" for x in g]
        if not arm_a or not arm_b:
            return None
        return mean(arm_a) - mean(arm_b)

    observed = diff(pool)
    groups = [g for _, g in pool]
    extreme = 0
    for _ in range(reps):
        rng.shuffle(groups)
        relabelled = [("a" if i < n_a else "b", g) for i, g in enumerate(groups)]
        value = diff(relabelled)
        if value is not None and abs(value) >= abs(observed) - 1e-12:
            extreme += 1
    return {"difference": observed, "p_value": (extreme + 1) / (reps + 1), "reps": reps}


# --------------------------------------------------------------------------
# model-free distractor heuristics
# --------------------------------------------------------------------------
def parse_number(text):
    cleaned = str(text).strip().replace(",", "").rstrip("%")
    return float(cleaned) if NUM_RE.match(cleaned) else None


def heuristic_accuracy(questions):
    """Can a rule with no biology and no model pick the keyed answer?

    Each item is the unordered set {ideal} u distractors. A rule that beats
    1/4 shows the option set itself carries the answer, independently of any
    exposure to the source study.
    """
    rules = {
        "longest_option": lambda opts: max(opts, key=lambda o: (len(str(o)), str(o))),
        "shortest_option": lambda opts: min(opts, key=lambda o: (len(str(o)), str(o))),
        "most_decimal_places": lambda opts: max(
            opts, key=lambda o: (len(str(o).split(".")[1]) if "." in str(o) else 0, str(o))),
        "numeric_median": _numeric_median,
        "numeric_extreme": _numeric_extreme,
    }
    scores = {name: {"hits": 0, "n": 0, "per_capsule": {}} for name in rules}
    for qid, row in questions.items():
        ideal = row.get("ideal")
        distractors = row.get("distractors") or []
        if isinstance(distractors, str):
            try:
                distractors = json.loads(distractors.replace("'", '"'))
            except json.JSONDecodeError:
                distractors = [d.strip(" '\"") for d in distractors.strip("[]").split(",")]
        options = [ideal] + list(distractors)
        if len(options) != 4 or any(o is None for o in options):
            continue
        for name, rule in rules.items():
            pick = rule(options)
            if pick is None:
                continue
            hit = str(pick).strip() == str(ideal).strip()
            scores[name]["hits"] += hit
            scores[name]["n"] += 1
            scores[name]["per_capsule"].setdefault(row["capsule_uuid"], []).append(1.0 if hit else 0.0)
    out = {}
    for name, rec in scores.items():
        if not rec["n"]:
            continue
        acc = rec["hits"] / rec["n"]
        lo, hi = cluster_bootstrap_ci(rec["per_capsule"], mean)
        out[name] = {"accuracy": acc, "n_scored": rec["n"], "ci95": [lo, hi],
                     "above_chance": lo > CHANCE_MCQ}
    return out


def _numeric_median(options):
    values = [(parse_number(o), o) for o in options]
    if any(v is None for v, _ in values):
        return None
    values.sort(key=lambda t: t[0])
    return values[1][1]  # lower of the two middle values, deterministic


def _numeric_extreme(options):
    values = [(parse_number(o), o) for o in options]
    if any(v is None for v, _ in values):
        return None
    return max(values, key=lambda t: abs(t[0]))[1]


# --------------------------------------------------------------------------
# main analysis
# --------------------------------------------------------------------------
def provenance_subsets(audit):
    """Capsules that name a source, and capsules that name none."""
    capsules = audit["capsules"]
    identified = {u for u, c in capsules.items() if c["class"] in {"doi", "repository_url"}}
    missing = {u for u, c in capsules.items() if c["class"] == "missing"}
    return identified, missing


def analyse_run(label, path, questions, audit, reps):
    per_question = load_zero_shot(path)
    joined = {q: c for q, c in per_question.items() if q in questions}
    clusters = group_by_capsule(joined, questions)
    if not clusters:
        return None
    all_scores = [x for g in clusters.values() for x in g]
    lo, hi = cluster_bootstrap_ci(clusters, mean, reps=reps)
    identified, missing = provenance_subsets(audit)
    ca = group_by_capsule(joined, questions, identified)
    cb = group_by_capsule(joined, questions, missing)
    record = {
        "run": label,
        "source_file": str(path),
        "n_questions_joined": len(joined),
        "n_questions_in_file": len(per_question),
        "n_capsules": len(clusters),
        "accuracy": mean(all_scores),
        "ci95_cluster_bootstrap": [lo, hi],
        "clustering": icc_and_design_effect(clusters),
        "by_provenance": {
            "source_identified": {"n_questions": sum(len(g) for g in ca.values()),
                                  "n_capsules": len(ca), "accuracy": mean([x for g in ca.values() for x in g])},
            "no_source_identifier": {"n_questions": sum(len(g) for g in cb.values()),
                                     "n_capsules": len(cb), "accuracy": mean([x for g in cb.values() for x in g])},
        },
    }
    if ca and cb:
        record["by_provenance"]["cluster_permutation"] = cluster_permutation_diff(ca, cb, reps=reps)
        record["by_provenance"]["difference_ci95"] = _diff_ci(ca, cb, reps)
    if "mcq" in label:
        record["chance"] = CHANCE_MCQ
        record["above_chance_margin"] = record["accuracy"] - CHANCE_MCQ
        record["above_chance_ci_excludes"] = lo > CHANCE_MCQ
    return record


def _diff_ci(ca, cb, reps, seed=20260918):
    rng = random.Random(seed)
    ka, kb = list(ca), list(cb)
    draws = []
    for _ in range(reps):
        a = [x for k in (rng.choice(ka) for _ in ka) for x in ca[k]]
        b = [x for k in (rng.choice(kb) for _ in kb) for x in cb[k]]
        if a and b:
            draws.append(mean(a) - mean(b))
    draws.sort()
    return [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]]


# --------------------------------------------------------------------------
# is the provenance contrast confounded with how the items were built?
# --------------------------------------------------------------------------
def item_structure(questions):
    """Per-question construction features that compete with exposure as an
    explanation for above-chance no-data accuracy."""
    from option_artifacts import numeric_rank

    out = {}
    for qid, row in questions.items():
        distractors = row.get("distractors") or []
        if isinstance(distractors, str):
            try:
                distractors = ast.literal_eval(distractors)
            except (ValueError, SyntaxError):
                distractors = []
        options = [row.get("ideal")] + list(distractors)
        rank = numeric_rank(options, 0) if len(options) == 4 else None
        out[qid] = {"numeric": rank is not None, "rank": rank,
                    "interior": rank in (1, 2) if rank is not None else None,
                    "eval_mode": row.get("eval_mode"),
                    "capsule": row["capsule_uuid"]}
    return out


def confounding_analysis(per_question, questions, audit, structure, reps):
    """Compare the two provenance strata on composition, then on score.

    If the strata differ in item construction and grading mode, a raw score
    difference between them is not evidence about temporal exposure.
    """
    identified, missing = provenance_subsets(audit)
    strata = {"source_identified": identified, "no_source_identifier": missing}

    composition = {}
    for name, caps in strata.items():
        items = [s for q, s in structure.items() if s["capsule"] in caps]
        numeric = [s for s in items if s["numeric"]]
        composition[name] = {
            "n_questions": len(items),
            "n_capsules": len(caps),
            "numeric_share": len(numeric) / len(items) if items else None,
            "interior_rate_among_numeric": (sum(1 for s in numeric if s["interior"]) / len(numeric)
                                            if numeric else None),
            "eval_mode_mix": dict(Counter(s["eval_mode"] for s in items)),
            "llm_graded_share": (sum(1 for s in items if s["eval_mode"] == "llm_verifier") / len(items)
                                 if items else None),
        }

    def contrast(predicate, label):
        ca, cb = {}, {}
        for qid, correct in per_question.items():
            s = structure.get(qid)
            if s is None or not predicate(s):
                continue
            target = ca if s["capsule"] in identified else (cb if s["capsule"] in missing else None)
            if target is not None:
                target.setdefault(s["capsule"], []).append(1.0 if correct else 0.0)
        if not ca or not cb:
            return None
        flat_a = [x for g in ca.values() for x in g]
        flat_b = [x for g in cb.values() for x in g]
        rec = {"restriction": label,
               "source_identified": {"accuracy": mean(flat_a), "n": len(flat_a), "n_capsules": len(ca)},
               "no_source_identifier": {"accuracy": mean(flat_b), "n": len(flat_b), "n_capsules": len(cb)},
               "difference": mean(flat_a) - mean(flat_b)}
        rec.update(cluster_permutation_diff(ca, cb, reps=reps))
        rec["difference_ci95"] = _diff_ci(ca, cb, reps)
        return rec

    contrasts = [c for c in (
        contrast(lambda s: True, "all questions"),
        contrast(lambda s: s["numeric"], "numeric-option questions only"),
        contrast(lambda s: not s["numeric"], "non-numeric questions only"),
        contrast(lambda s: s["eval_mode"] == "llm_verifier", "LLM-graded questions only"),
        contrast(lambda s: s["eval_mode"] != "llm_verifier", "deterministically graded questions only"),
    ) if c]

    return {"composition": composition, "contrasts": contrasts,
            "interpretation": ("The strata differ in numeric-option share, distractor geometry, and "
                               "grading mode. A score difference between them is therefore not "
                               "attributable to temporal exposure without adjusting for construction.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", type=Path, default=Path("data/bixbench.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("results/audit.json"))
    parser.add_argument("--zero-shot-dir", type=Path, default=Path("data/external/zero_shot_v15"))
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("results/no_data_baseline.json"))
    args = parser.parse_args()

    questions = load_questions(args.jsonl)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))

    runs = []
    for path in sorted(args.zero_shot_dir.glob("*.csv")):
        label = path.stem
        record = analyse_run(label, path, questions, audit, args.reps)
        if record:
            runs.append(record)
            print(f"{label:52s} acc={record['accuracy']:.3f} "
                  f"[{record['ci95_cluster_bootstrap'][0]:.3f}, {record['ci95_cluster_bootstrap'][1]:.3f}] "
                  f"n={record['n_questions_joined']}", flush=True)

    structure = item_structure(questions)
    primary = args.zero_shot_dir / "gpt-4o-grader-mcq-refusal-False.csv"
    confounding = None
    if primary.exists():
        pq = {q: c for q, c in load_zero_shot(primary).items() if q in questions}
        confounding = confounding_analysis(pq, questions, audit, structure, args.reps)
        print("\nComposition of the two provenance strata (BixBench v1.5):")
        for name, rec in confounding["composition"].items():
            print(f"  {name:22s} n={rec['n_questions']:3d}  numeric={rec['numeric_share']:.1%}"
                  f"  interior={rec['interior_rate_among_numeric']:.1%}"
                  f"  LLM-graded={rec['llm_graded_share']:.1%}")
        print("\nNo-data accuracy gap (source-identified minus unidentified), gpt-4o MCQ:")
        for c in confounding["contrasts"]:
            print(f"  {c['restriction']:38s} {c['difference']:+.3f}"
                  f"  [{c['difference_ci95'][0]:+.3f}, {c['difference_ci95'][1]:+.3f}]"
                  f"  p={c['p_value']:.3f}  (n={c['source_identified']['n']}"
                  f" vs {c['no_source_identifier']['n']})")

    heuristics = heuristic_accuracy(questions)
    print("\nModel-free option-set heuristics (chance = 0.25):")
    for name, rec in sorted(heuristics.items(), key=lambda kv: -kv[1]["accuracy"]):
        flag = "*" if rec["above_chance"] else " "
        print(f"  {flag} {name:22s} {rec['accuracy']:.3f} "
              f"[{rec['ci95'][0]:.3f}, {rec['ci95'][1]:.3f}]  n={rec['n_scored']}")

    result = {
        "chance_mcq": CHANCE_MCQ,
        "bootstrap_reps": args.reps,
        "inference": "cluster bootstrap and cluster permutation over capsule_uuid",
        "runs": runs,
        "option_set_heuristics": heuristics,
        "confounding_analysis": confounding,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
