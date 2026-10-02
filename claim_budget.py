#!/usr/bin/env python3
r"""The paper-wide error budget: the findings the abstract states, corrected together.

Every result file here is corrected within its own family, which leaves the
argument that runs across them uncorrected. This reads, as one family, every
interval the abstract and the introduction's contributions state as a finding, and
the two no-data findings Appendix C rests on, ten of them (the camera-ready's; the
submitted version's eight included the configurations' own graders' moved-key gain,
which the paper no longer states as a finding, and not the three new headline claims):

  rank rule     BixBench v1.5's 105 numeric items, the second-smallest rule vs chance
  published x2  BixBench's released no-data runs, forced choice among the options,
                each against its own chance rate, one per model
  not the rank  each released run's accuracy where the key is second-smallest
                minus elsewhere, whose upper end has to fall below what a reader
                drawing its margin from the rank would show
  forced over   BixBench's published forced grades of its released v1.0 options
  tolerance     on the published gpt-4o and Claude 3.5 Sonnet runs, minus the
                share of the same runs within 5% of the key, both models
  key over      the same grades: of the misses whose number has one nearest option,
  other         the share accepted where it is the key minus where it is another,
                both models, the two shares resampled over the same capsules
  corrected     the seven v1.5 configurations, each graded by its own model forced:
  below         the tolerance minus the score corrected for guessing
  tolerance
  current       gpt-5.1, gpt-6-luna and DeepSeek-V4-Pro graded by gpt-4o forced,
  agents        minus the tolerance, every agent
  bracketing    with the data, the nearest-option rule's gain from the repaired
                options over the placebo (which shares the redraw and keeps the
                rank) on the 37 items whose key the repair moved to an edge,
                averaged over the seven run sets
  readers of    the registered published v1.0 runs read forced through
  the published MCQ_EVAL_PROMPT by each of three open models, the repaired options
  runs          against the placebo on the keys the repair moved to an edge, every
                reader (published_reads.py)

**The family is post hoc.** It was chosen after the results were known, as the
set the abstract leads with; the correction controls the error across these
ten, not the choice of which ten. Nulls are not in it: widening an interval
that already contains zero changes nothing a reader is told. A conjunction --
the rank test in both models -- is an intersection-union test: every part has to
clear at the family level, and no further correction is owed.

**Each claim is read at the level that covers 1 - 0.05/k on its own file.** A
nominal level is not a covering one when clusters are few and unequal, and the
paper measures the covering level one way: a beta-binomial fitted by moments to
an arm's per-cluster accuracy, files simulated at that arm's exact cluster
sizes, and coverage read off every nominal level at once
(bootstrap_calibration.py). This uses the same machinery with the target moved
from 95% to the family's, on a finer grid near one where a Bonferroni level on a
small file lands. The level is a property of the file's shape, fitted to one arm
and applied to the statistic read on it, paired or not.

**The paired with-data claims are calibrated by the double bootstrap instead.** Every
claim but the rank rule is a paired difference of gradings of the same runs; the
moved-key gains are on 37 items in 21 capsules, where the arm the shape would be
fitted to is right on few of those items: the moment fit finds no spread between
capsules beyond the binomial, and the simulated percentile bootstrap never reached
the target on the same contrast without the data at any level on the grid.
Resampling the capsules' own paired values and re-bootstrapping each resample needs
no model of them.

A nonparametric double bootstrap would need no model of the clusters, and it was
tried first. It agrees on BixBench's shape and not on MMLU-Pro's: it has the
nominal 95% interval covering 0.948 there, where the parametric simulation the
paper reports has 0.903 -- optimistic in exactly the regime the calibration
exists for, because resampling sixty labels cannot manufacture the file where
the subject holding a quarter of the items is unusual. ``--check-method`` prints
that comparison.

**One interval per finding.** Beside the corrected interval, tab:claims prints each finding's interval at
the level that attains 95% coverage exactly as the paper states it elsewhere (``quoted_95``): this script's own
for the findings the text quotes from it (the rank rule, the two no-data margins), and for the others the
interval of the table or section that states the finding, read from that analysis's report, whose estimate must
agree with this one. The nominal 95% percentile interval stays in the report, not in the table.

    python3 claim_budget.py
    python3 claim_budget.py --attach-quoted   # attach the quoted intervals to results/claim_budget.json only
    python3 claim_budget.py --latex     # tab:claims' rows
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

import arm_intervals
import bootstrap_calibration as bc
import option_artifacts
import published_repair

FAMILY_ALPHA = 0.05
# Fine near one, where a Bonferroni level on a small file lands.
LEVELS = np.concatenate([np.arange(0.80, 0.99, 0.0025),
                         np.arange(0.99, 0.99995, 0.00005)])


# --- a statistic is a signed sum of cluster means, each resampled on its own --

def group(values_by_cluster, sign=1.0):
    """(per-cluster sums, per-cluster counts, sign) for one resampled arm."""
    keys = sorted(values_by_cluster)
    sums = np.array([float(np.sum(values_by_cluster[k])) for k in keys])
    counts = np.array([float(len(values_by_cluster[k])) for k in keys])
    return sums, counts, sign


def joint(values_by_cluster, signs):
    """Several ratios over the same clusters, resampled together, their signed sum the statistic:
    ((m x n) per-cluster sums, (m x n) per-cluster weights, m signs), one row a ratio. ``values_by_cluster``
    is one {cluster: (sum, weight)} a ratio; a cluster with none of a ratio's runs has weight 0 there."""
    keys = sorted(set().union(*values_by_cluster))
    sums = np.array([[float(v.get(k, (0.0, 0.0))[0]) for k in keys] for v in values_by_cluster])
    weights = np.array([[float(v.get(k, (0.0, 0.0))[1]) for k in keys] for v in values_by_cluster])
    return sums, weights, np.asarray(signs, dtype=float)


def estimate(groups):
    total = 0.0
    for a, n, s in groups:
        total += float(np.sum(s * a.sum(1) / n.sum(1))) if np.ndim(a) == 2 else s * a.sum() / n.sum()
    return float(total)


def draws(groups, reps, rng):
    """The paper's percentile cluster bootstrap, vectorised. A joint group's ratios share each draw of clusters."""
    out = np.zeros(reps)
    for sums, counts, sign in groups:
        if np.ndim(sums) == 2:
            idx = rng.integers(0, sums.shape[1], size=(reps, sums.shape[1]))
            with np.errstate(invalid="ignore", divide="ignore"):
                out += (sign[:, None] * sums[:, idx].sum(-1) / counts[:, idx].sum(-1)).sum(0)
            continue
        idx = rng.integers(0, len(sums), size=(reps, len(sums)))
        out += sign * sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return out


def effective(groups):
    """Kish's effective count of the first (released) arm's clusters."""
    counts = groups[0][1]
    counts = counts.sum(0) if np.ndim(counts) == 2 else counts
    return float(counts.sum() ** 2 / (counts ** 2).sum())


def covering_levels(groups, targets, outer, inner, rng):
    """Nominal level whose actual coverage first reaches each target."""
    theta = estimate(groups)
    below = np.empty(outer)
    for r in range(outer):
        pseudo = []
        for sums, counts, sign in groups:
            if np.ndim(sums) == 2:
                pick = rng.integers(0, sums.shape[1], size=sums.shape[1])
                pseudo.append((sums[:, pick], counts[:, pick], sign))
                continue
            pick = rng.integers(0, len(sums), size=len(sums))
            pseudo.append((sums[pick], counts[pick], sign))
        below[r] = float(np.mean(draws(pseudo, inner, rng) < theta))
    tails = (1.0 - LEVELS) / 2.0
    curve = np.array([float(np.mean((below > a) & (below < 1.0 - a))) for a in tails])
    out = {}
    for target in targets:
        reached = np.nonzero(curve >= target)[0]
        out[target] = float(LEVELS[reached[0]]) if len(reached) else None
    at95 = int(np.argmin(np.abs(LEVELS - 0.95)))
    return out, float(curve[at95])


def curve_of(below):
    tails = (1.0 - LEVELS) / 2.0
    below = np.asarray(below)
    return np.array([float(np.mean((below > a) & (below < 1.0 - a))) for a in tails])


def level_for(curve, target):
    reached = np.nonzero(curve >= target)[0]
    return float(LEVELS[reached[0]]) if len(reached) else None


def parametric_levels(shape, clean_shape, targets, reps, inner, rng):
    """bootstrap_calibration.py's simulation, read at any target.

    ``shape`` is (rate, beta concentration, cluster sizes) for the released arm,
    as ``bc.fit_beta_binomial`` returns it; ``clean_shape`` the control's, when
    the statistic is a difference from it. Cluster sizes are the arm's own,
    exactly, as in that script's as-observed cell, which is the one that speaks
    to an interval actually printed.
    """
    rate, spread, sizes = shape
    if clean_shape is not None:
        c_rate, c_spread, c_sizes = clean_shape
    below = []
    for _ in range(reps):
        hits, trials = bc.simulate(rate, spread, sizes, None, rng)
        d = bc.bootstrap_draws(hits, trials, inner, rng)
        if clean_shape is None:
            below.append(float(np.mean(d < rate)))
        else:
            c_hits, c_trials = bc.simulate(c_rate, c_spread, c_sizes, None, rng)
            c = bc.bootstrap_draws(c_hits, c_trials, inner, rng)
            below.append(float(np.mean(d - c < rate - c_rate)))
    curve = curve_of(below)
    at95 = int(np.argmin(np.abs(LEVELS - 0.95)))
    return {t: level_for(curve, t) for t in targets}, float(curve[at95])


def table(rows_by_cluster):
    """(successes, trials) per cluster from 0/1 outcomes."""
    keys = sorted(rows_by_cluster)
    return np.array([[float(np.sum(rows_by_cluster[k])), float(len(rows_by_cluster[k]))]
                     for k in keys])


def interval(values, level):
    tail = 100.0 * (1.0 - level) / 2.0
    # a draw of clusters with none of a ratio's runs has no value (only a joint group's can)
    return [float(np.nanpercentile(values, tail)), float(np.nanpercentile(values, 100.0 - tail))]


# --- the five claims, each from the file the paper quotes it from -----------
#
# Each returns the groups the statistic is read from and the shape it is
# calibrated on: the released arm's, fitted as bootstrap_calibration.py fits it.

def per_cluster(rows, value):
    out = defaultdict(list)
    for r in rows:
        out[r["cluster"]].append(value(r))
    return out


def margin(r):
    return (1.0 if r["answer"] == r["gold"] else 0.0) - 1.0 / r["k"]


def hit(r):
    return 1.0 if r["answer"] == r["gold"] else 0.0


def fit(values_by_cluster):
    return bc.fit_beta_binomial(table(values_by_cluster))


def rank_rule():
    items, _ = option_artifacts.items_from_bixbench_jsonl(
        pathlib.Path("data/bixbench.jsonl"))
    counts = defaultdict(int)
    for it in items:
        counts[it["rank"]] += 1
    top = min(counts, key=lambda r: (-counts[r], r))
    hits = defaultdict(list)
    for it in items:
        hits[it["cluster"]].append(1.0 if it["rank"] == top else 0.0)
    values = {c: [h - 0.25 for h in v] for c, v in hits.items()}
    return [group(values)], fit(hits), None


def published_forced(model):
    """The released forced-choice arm on v1.5, each question against its own chance rate."""
    arms, _ = published_repair.v15(model)
    values, forced = defaultdict(list), defaultdict(list)
    for r in arms["mcq_forced"]:
        values[r["cluster"]].append(r["correct"] - r["chance"])
        forced[r["cluster"]].append(r["correct"])
    return [group(values)], fit(forced), None


def bracketing_gain(condition="data", base="placebo", reader="nearest"):
    """The repaired options against ``base`` on the items whose key the repair moved from a
    bracketed rank to an edge, averaged over the seven run sets -- with the data, or
    (``condition="nodata"``) under the same agents' runs without it -- read by the nearest-option
    rule or (``reader="own"``) by each run set's own reader through MCQ_EVAL_PROMPT."""
    import bixbench_withdata as bw
    import bracketing
    sets = bw.option_sets()
    per_item, based = defaultdict(list), defaultdict(list)
    capsule = {}
    for run in bracketing.RUNS:
        for r in bracketing.rows_of(run):
            if not r["numeric"] or r["condition"] != condition:
                continue
            if not (bracketing.bracketed(r["key_rank"]["released"])
                    and not bracketing.bracketed(r["key_rank"]["repaired"])):
                continue
            q = r["question_id"]
            capsule[q] = r["capsule"]
            if reader == "nearest":
                b = bw.nearest_is_key(r["answer"], sets[q][base])
                rep = bw.nearest_is_key(r["answer"], sets[q]["repaired"])
            else:
                b = bracketing.read_score(r, bracketing.primary(run), base)
                rep = bracketing.read_score(r, bracketing.primary(run), "repaired")
                if b is None or rep is None:
                    continue
            per_item[q].append(rep - b)
            based[q].append(b)
    values, shape = defaultdict(list), defaultdict(list)
    for q in sorted(per_item):
        if reader == "nearest" and len(per_item[q]) != len(bracketing.RUNS):
            raise SystemExit(f"{q}: {len(per_item[q])} run sets, expected {len(bracketing.RUNS)}")
        values[capsule[q]].append(float(np.mean(per_item[q])))
        shape[capsule[q]].append(float(np.mean(based[q])))
    return [group(values)], fit(shape), None


def frontier_shape(source="results/frontier_validity_items.json",
                   stored="results/frontier_calibrated.json"):
    """frontier_calibrated.py's fitted shape, at the frontier's own subject sizes.

    Not a claim in the family any more; kept because it is the check that this
    script's calibration reproduces the level the frontier table is printed at
    (tests/test_published_repair.py).
    """
    cal = json.loads(pathlib.Path(stored).read_text())["calibration"]
    clusters = json.loads(pathlib.Path(source).read_text())["clusters"]
    sizes = np.array(sorted(np.unique(clusters, return_counts=True)[1]), dtype=float)
    if len(sizes) != cal["n_clusters"]:
        raise SystemExit(f"{source} has {len(sizes)} subjects; {stored} was fitted "
                         f"on {cal['n_clusters']}")
    return cal["rate"], cal["beta_concentration"], sizes


# --- the check on the method itself ------------------------------------------

def parametric_check(outer, inner, rng):
    """The double bootstrap against bootstrap_calibration.py, where both exist."""
    stored = {entry["arm"]: entry for entry in json.loads(
        pathlib.Path("results/bootstrap_calibration.json").read_text())["arms"]}
    rows = []
    for label, entry in stored.items():
        observed = next(c for c in entry["cells"] if c["as_observed"])
        path = arm_intervals.resolve(entry["dump"])
        if not path.exists():
            continue
        groups = [group(per_cluster(arm_intervals.load(path, "file"), margin)),
                  group(per_cluster(arm_intervals.load(path, "clean"), margin), -1.0)]
        levels, cov95 = covering_levels(groups, [0.95], outer, inner, rng)
        rows.append({"arm": label,
                     "parametric_level_for_95": observed["level_for_95_difference"],
                     "parametric_coverage_at_95": observed["coverage_difference"],
                     "double_bootstrap_level_for_95": levels[0.95],
                     "double_bootstrap_coverage_at_95": cov95})
    return rows


def rank_reader(model):
    """BixBench's released v1.5 forced run on its 105 numeric items: accuracy where the key is
    the second-smallest value minus accuracy where it is not. A reader that drew its margin
    from the rank would show ``rank_attribution.rank_reader_prediction``'s difference; the
    claim is that the interval's upper end falls below it."""
    import rank_attribution as ra
    rows = [r for r in ra.published_v15(model) if r["key_rank"] is not None]
    second, rest = defaultdict(list), defaultdict(list)
    for r in rows:
        (second if r["key_rank"] == 1 else rest)[r["capsule"]].append(r["correct"])
    counts = defaultdict(int)
    for r in rows:
        counts[r["key_rank"]] += 1
    shares = [counts[j] / len(rows) for j in range(4)]
    prediction = ra.rank_reader_prediction(shares, float(np.mean([r["correct"] for r in rows])))
    return ([group(second), group(rest, -1.0)], fit(second), fit(rest)), prediction["difference"]


def forced_over_tolerance(model):
    """BixBench's published forced reading of its released options on one model's published v1.0
    runs (both run sets) on the 159 numeric questions, minus the share of the same runs whose
    submitted number is within 5% of the key: per question, its runs averaged
    (``score_decomposition.py``)."""
    import score_decomposition as sd
    runs = [r for r in sd.d1_runs() if r["group"] == model]
    per_item, capsule = defaultdict(list), {}
    for r in runs:
        hit = float(bool(sd.graded(r["answer"], r["options"][0], sd.TOL))) if r["answer"] else 0.0
        per_item[r["q"]].append(sd.named_share(r, "published forced") - hit)
        capsule[r["q"]] = r["capsule"]
    values, shape = defaultdict(list), defaultdict(list)
    for q in sorted(per_item):
        values[capsule[q]].append(float(np.mean(per_item[q])))
        shape[capsule[q]].append(float(np.mean([v > 0 for v in per_item[q]])))
    return [group(values)], fit(shape), None


def excess_over_tolerance(runs, reading):
    """A grading's score minus the tolerance's, per question (its runs averaged), grouped by capsule: Table 2's
    excess, as ``score_decomposition.summarise`` computes it."""
    import score_decomposition as sd
    per_item, capsule = defaultdict(list), {}
    for r in runs:
        named = sd.named_share(r, reading)
        if named is None:
            continue
        hit = float(bool(sd.graded(r["answer"], r["options"][0], sd.TOL))) if r["answer"] else 0.0
        per_item[r["q"]].append(named - hit)
        capsule[r["q"]] = r["capsule"]
    values, shape = defaultdict(list), defaultdict(list)
    for q in sorted(per_item):
        values[capsule[q]].append(float(np.mean(per_item[q])))
        shape[capsule[q]].append(float(np.mean([v > 0 for v in per_item[q]])))
    return [group(values)], fit(shape), None


def current_agent_runs(label):
    """A current agent's runs with the data as ``score_decomposition`` reads them, graded by gpt-4o."""
    import bixbench_withdata as bw
    import frontier_agents as fa
    import score_decomposition as sd
    if label == "gpt-5.1":
        return sd.d3_runs()
    model = dict((b, a) for a, b in fa.MODELS)[label]
    return fa.as_runs([r for r in fa.load_rows() if r["model"] == model], label, bw.option_sets())


def key_over_other(runs, reading):
    """Of the graded misses whose number has a single nearest option, the share a grading accepts where that
    option is the key minus the share where it is another: each question's runs averaged and the questions
    summed, as ``score_decomposition.miss_rates`` weighs them, the two shares resampled over the same capsules."""
    import answer_extraction as ax
    import score_decomposition as sd
    read = [r for r in runs if sd.named_share(r, reading) is not None]
    capsule = {r["q"]: r["capsule"] for r in read}
    hit = lambda r: float(bool(sd.graded(r["answer"], r["options"][0], sd.TOL))) if r["answer"] else 0.0
    miss = lambda r: (1.0 - hit(r)) * float(ax.category(r["answer"]) != "empty")
    ratios = []
    for kind in ("key", "other"):
        weight = lambda r, kind=kind: miss(r) * float(sd.single_nearest(r) == kind)
        num = sd.per_item(read, lambda r, w=weight: w(r) * sd.named_share(r, reading))
        den = sd.per_item(read, weight)
        by = defaultdict(lambda: [0.0, 0.0])
        for q in den:
            by[capsule[q]][0] += num[q]
            by[capsule[q]][1] += den[q]
        ratios.append({c: tuple(v) for c, v in by.items()})
    return [joint(ratios, [1.0, -1.0])], ("key over other", id(runs), reading), None


def corrected_below_tolerance(runs, reading, k=4):
    """The tolerance minus the forced-choice score corrected for guessing, (S - 1/k)/(1 - 1/k), per question (its
    runs averaged), grouped by capsule: ``formula_scoring.summarise``'s residual with its sign turned, so that a
    correction that falls short of the tolerance is positive."""
    import score_decomposition as sd
    per_item, capsule = defaultdict(list), {}
    for r in runs:
        named = sd.named_share(r, reading)
        if named is None:
            continue
        hit = float(bool(sd.graded(r["answer"], r["options"][0], sd.TOL))) if r["answer"] else 0.0
        per_item[r["q"]].append(hit - (named - 1 / k) / (1 - 1 / k))
        capsule[r["q"]] = r["capsule"]
    values, shape = defaultdict(list), defaultdict(list)
    for q in sorted(per_item):
        values[capsule[q]].append(float(np.mean(per_item[q])))
        shape[capsule[q]].append(float(np.mean([v > 0 for v in per_item[q]])))
    return [group(values)], fit(shape), None


READERS_OF_PUBLISHED = (("qwen72b", "Qwen2.5-72B"), ("gemma27b", "gemma-3-27b"), ("llama70b", "Llama-3.3-70B"))


def published_reader(reader):
    """The registered published v1.0 runs read forced by ``reader`` through MCQ_EVAL_PROMPT, as
    BixBench reads a run (``published_reads.py``): the repaired options against the placebo on the
    keys the repair moved to an edge, each item's per-run-set gain averaged over the run sets, as
    ``published_reads.contrasts`` pools it."""
    import published_reads as pr
    import replication as rp
    v10 = rp.v10_sets()
    capsule = {q: e["capsule"] for q, e in rp.load_extract()["items"].items()}
    rows = [r for r in pr.load_rows(pr.rows_path(reader, "forced")) if r["set"] == "D1"]
    moved = {q for q, s in v10.items() if rp.group_of(s) == "moved to an edge"}
    gain = pr.gains(rows, "repaired", "placebo")
    base = defaultdict(list)
    for r in rows:
        b = pr.score(r, "placebo")
        if r["q"] in moved and b is not None:
            base[r["q"]].append(b)
    values, shape = defaultdict(list), defaultdict(list)
    for q in sorted(gain):
        if q in moved:
            values[capsule[q]].append(gain[q])
            shape[capsule[q]].append(float(np.mean(base[q])))
    return [group(values)], fit(shape), None


# tab:claims: one row per part, in the family's order; a part whose corrected interval does not clear
# is marked, and the caption says what the mark means
ROWS = {"rank rule": ("A rule that ignores the question selects the second-smallest option",
                      "v1.5, $105$ numeric; chance"),
        "published, gpt-4o": ("BixBench's released forced-choice run without the data, gpt-4o",
                              "v1.5, $205$; chance"),
        "published, claude": ("The same, Claude 3.5 Sonnet", "v1.5, $205$; chance"),
        "not the rank|gpt-4o": ("Accuracy where the key is second-smallest minus elsewhere, gpt-4o",
                                "v1.5, $105$ numeric; below"),
        "not the rank|claude-3-5-sonnet-latest": ("The same, Claude 3.5 Sonnet", "v1.5, $105$ numeric; below"),
        "forced over tolerance|gpt-4o": ("BixBench's forced-choice grading of $R$, gpt-4o's published runs",
                                         "v1.0, $159$ numeric; within $5\\%$"),
        "forced over tolerance|Claude 3.5 Sonnet": ("The same, Claude 3.5 Sonnet's",
                                                    "v1.0, $159$ numeric; within $5\\%$"),
        "key over other|gpt-4o": ("Its acceptance of misses nearest the key, gpt-4o's published runs",
                                  "v1.0; misses nearest another"),
        "key over other|Claude 3.5 Sonnet": ("The same, Claude 3.5 Sonnet's", "v1.0; misses nearest another"),
        "corrected below tolerance": ("The tolerance above the forced-choice score corrected for guessing, seven "
                                      "configurations", "v1.5, $105$ numeric; the correction"),
        "current agents|gpt-5.1": ("Forced-choice grading by gpt-4o, gpt-5.1's runs", "v1.5, $105$ numeric; within $5\\%$"),
        "current agents|gpt-6-luna": ("The same, gpt-6-luna's", "v1.5, $105$ numeric; within $5\\%$"),
        "current agents|DeepSeek-V4-Pro": ("The same, DeepSeek-V4-Pro's", "v1.5, $105$ numeric; within $5\\%$"),
        "bracketing": ("With the data, $U-P$ on the moved keys, nearest-option rule",
                       "$37$ items, $7$ configurations"),
        "readers of the published runs|Qwen2.5-72B": (
            "With the data, $U-P$ on the moved keys, Qwen2.5-72B grading BixBench's published runs",
            "v1.0, $40$ items"),
        "readers of the published runs|gemma-3-27b": ("The same, gemma-3-27b grading them", "v1.0, $40$ items"),
        "readers of the published runs|Llama-3.3-70B": ("The same, Llama-3.3-70B grading them",
                                                        "v1.0, $40$ items")}


def quoted_95(claim, part):
    """(the finding's interval at the level that attains 95% coverage, as the paper states it elsewhere, and the
    report it is read from), for one part of a claim; its estimate must be this part's."""
    def load(path):
        return json.loads(pathlib.Path(path).read_text())
    name, label = claim["claim"], part["part"]
    if name in ("rank rule", "published, gpt-4o", "published, claude", "key over other"):
        src, where = {"mean": part["estimate"], "lo": part["ci_covering_95"][0], "hi": part["ci_covering_95"][1]}, \
            "results/claim_budget.json"
    elif name == "corrected below tolerance":
        where = "results/formula_scoring.json"
        row = next(r for r in load(where)["rows"] if (r["block"], r["group"]) == ("v1.5", "pooled"))
        c = row["corrected_minus_tolerance"]
        src = {"mean": -c["mean"], "lo": -c["hi"], "hi": -c["lo"]}
    elif name == "current agents":
        if label == "gpt-5.1":
            where = "results/score_decomposition.json"
            src = load(where)["v1.5 closed"]["gpt-5.1"]["gpt-4o forced"]["score_minus_tolerance"]
        else:
            where = "results/frontier_agents.json"
            src = load(where)["agents"][label]["table1"]["gpt-4o forced"]["score_minus_tolerance"]
    elif name in ("bracketing", "reader"):
        where = "results/reader_split.json"
        grader = "nearest" if name == "bracketing" else "own"
        src = load(where)[f"data|{grader}|repaired-placebo|moved to an edge"]["pooled"]
    elif name == "not the rank":
        where = "results/rank_attribution.json"
        src = load(where)["published"][f"v1.5|{label}"]["second_smallest_minus_rest"]
    elif name == "forced over tolerance":
        where = "results/score_decomposition.json"
        src = load(where)["v1.0"][label]["published forced"]["score_minus_tolerance"]
    elif name == "readers of the published runs":
        where = {"Qwen2.5-72B": "results/published_reads_qwen72b_forced.json",
                 "gemma-3-27b": "results/published_reads.json",
                 "Llama-3.3-70B": "results/published_reads_llama70b_forced.json"}[label]
        src = load(where)["D1"]["repaired-placebo|moved to an edge"]
    else:
        raise KeyError(name)
    if abs(src["mean"] - part["estimate"]) >= 0.01:
        raise SystemExit(f"{name}|{label}: {where} states {src['mean']:.2f}, this report {part['estimate']:.2f}")
    return [float(src["lo"]), float(src["hi"])], where


def attach_quoted(report):
    for claim in report["claims"]:
        for part in claim["parts"]:
            part["ci_quoted_95"], part["ci_quoted_95_source"] = quoted_95(claim, part)
    return report


def table_rows(report):
    rows = []
    for claim in report["claims"]:
        for part in claim["parts"]:
            key = claim["claim"] if len(claim["parts"]) == 1 else f"{claim['claim']}|{part['part']}"
            finding, ref = ROWS[key]
            if "rank_reader_difference" in part:
                ref = f"{ref} ${part['rank_reader_difference']:+.1f}$"
            lo, hi = part["ci_family"]
            n_lo, n_hi = part["ci_quoted_95"]
            mark = "" if part["clears_family"] else "$^{\\times}$"
            rows.append(f"{finding} & {ref} & ${part['estimate']:+.1f}$ & $[{n_lo:+.1f},{n_hi:+.1f}]$ & "
                        f"$[{lo:+.1f},{hi:+.1f}]${mark}\\\\")
    return rows


def main() -> None:
    import sys
    if "--latex" in sys.argv:
        for row in table_rows(json.loads(pathlib.Path("results/claim_budget.json").read_text())):
            print(row)
        return
    if "--attach-quoted" in sys.argv:
        path = pathlib.Path("results/claim_budget.json")
        path.write_text(json.dumps(attach_quoted(json.loads(path.read_text())), indent=2) + "\n")
        print(f"attached the quoted intervals to {path}")
        return
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=4000, help="simulated files per claim")
    ap.add_argument("--inner", type=int, default=20000,
                    help="bootstrap draws per simulated file; the family level sits "
                         "far enough into the tail that 2000 cannot resolve it")
    ap.add_argument("--final", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--check-method", action="store_true",
                    help="also run the nonparametric double bootstrap this replaced")
    ap.add_argument("--output", default="results/claim_budget.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    import score_decomposition as sd
    published = sd.d1_runs()
    claims = [
        ("rank rule", "BixBench v1.5, 105 numeric items, second-smallest rule vs chance",
         [("", rank_rule())]),
        ("published, gpt-4o", "BixBench v1.5's released forced-choice run vs its own chance",
         [("", published_forced("gpt-4o"))]),
        ("published, claude", "BixBench v1.5's released forced-choice run vs its own chance",
         [("", published_forced("claude-3-5-sonnet-latest"))]),
    ]
    # not the rank: each published run's (second-smallest minus rest) must fall short of a
    # rank reader's, an intersection-union claim over the two models
    below = {}
    parts = []
    for model in ("gpt-4o", "claude-3-5-sonnet-latest"):
        spec, prediction = rank_reader(model)
        parts.append((model, spec))
        below[model] = prediction
    claims.append(("not the rank", "BixBench v1.5's released forced runs, 105 numeric items: accuracy "
                   "where the key is second-smallest minus elsewhere, below a rank reader's", parts))
    claims += [
        # the forced grades over the tolerance on the published runs, in both models (intersection-union)
        ("forced over tolerance", "BixBench v1.0's published runs, 159 numeric questions: the published forced "
         "grades through the released options minus the share within 5%",
         [(model, forced_over_tolerance(model)) for model in ("gpt-4o", "Claude 3.5 Sonnet")]),
        # the published forced grades accept the misses nearest the key more often than those nearest another
        ("key over other", "BixBench v1.0's published runs: of the misses whose number has one nearest option, "
         "the share the published forced grades accept where it is the key minus where it is another",
         [(model, key_over_other([r for r in published if r["group"] == model], "published forced"))
          for model in ("gpt-4o", "Claude 3.5 Sonnet")]),
        # the correction for guessing, applied to the score, falls below the tolerance on v1.5
        ("corrected below tolerance", "the seven v1.5 configurations with the data, each graded by its own model "
         "forced: the tolerance minus the score corrected for guessing",
         [("", corrected_below_tolerance(sd.d0_runs(), "family forced"))]),
        # the current agents' forced-choice excess, every agent (intersection-union)
        ("current agents", "gpt-5.1, gpt-6-luna and DeepSeek-V4-Pro with the data, graded by gpt-4o forced: the "
         "score minus the share within 5%",
         [(label, excess_over_tolerance(current_agent_runs(label), "gpt-4o forced"))
          for label in ("gpt-5.1", "gpt-6-luna", "DeepSeek-V4-Pro")]),
        ("bracketing", "with the data, nearest-option rule, repaired minus placebo, "
                       "items moved to an edge, mean over seven run sets",
         [("", bracketing_gain("data"))]),
        # the published runs graded with the notebook, as BixBench grades a run, every grader (intersection-union)
        ("readers of the published runs", "BixBench v1.0's published runs, with the data: repaired minus "
         "placebo on the keys moved to an edge, graded forced through MCQ_EVAL_PROMPT by three open models",
         [(label, published_reader(name)) for name, label in READERS_OF_PUBLISHED]),
    ]
    double = {"forced over tolerance", "key over other", "corrected below tolerance", "current agents", "bracketing",
              "readers of the published runs"}
    k = len(claims)
    target = 1.0 - FAMILY_ALPHA / k
    print(f"family of {k}; each claim read at the level covering {100 * target:.2f}%\n")

    report = {"family_alpha": FAMILY_ALPHA, "k": k, "target_coverage": target,
              "reps": args.reps, "inner": args.inner, "final": args.final,
              "seed": args.seed, "claims": []}
    levels_by_shape = {}
    for name, what, parts in claims:
        entry = {"claim": name, "what": what, "parts": []}
        for label, (groups, file_shape, clean_shape) in parts:
            # one simulation per shape: the repair-cost parts share theirs
            key = (id(file_shape), id(clean_shape))
            if key not in levels_by_shape:
                levels_by_shape[key] = (covering_levels(groups, [0.95, target], args.reps, args.inner, rng)
                                        if name in double else parametric_levels(
                                            file_shape, clean_shape, [0.95, target], args.reps, args.inner, rng))
            levels, cov95 = levels_by_shape[key]
            final = draws(groups, args.final, rng)
            level = levels[target]
            part = {"part": label or name,
                    "calibration": "double bootstrap" if name in double else "parametric",
                    "estimate": 100.0 * estimate(groups),
                    "n_clusters": int(np.shape(groups[0][0])[-1]),   # a joint group's sums are ratios x clusters
                    "effective_clusters": effective(groups),
                    "coverage_of_nominal_95": cov95,
                    "level_covering_95": levels[0.95],
                    "level_covering_target": level,
                    "ci95_nominal": [100.0 * x for x in interval(final, 0.95)],
                    "ci_covering_95": ([100.0 * x for x in interval(final, levels[0.95])]
                                       if levels[0.95] else None),
                    "ci_family": ([100.0 * x for x in interval(final, level)]
                                  if level else None)}
            if name == "not the rank" and label in below:
                hi = part["ci_family"][1] if part["ci_family"] else None
                part["rank_reader_difference"] = below[label]
                part["clears_family"] = bool(hi is not None and hi < below[label])
            else:
                lo = part["ci_family"][0] if part["ci_family"] else None
                part["clears_family"] = bool(lo is not None and lo > 0)
            entry["parts"].append(part)
            fam, c95 = part["ci_family"], part["ci_covering_95"]
            print(f"{part['part']:34s} {part['estimate']:+6.2f}  "
                  f"95% [{part['ci95_nominal'][0]:+6.2f},{part['ci95_nominal'][1]:+6.2f}] "
                  f"covers {cov95:.3f}; covering-95 "
                  + (f"[{c95[0]:+6.2f},{c95[1]:+6.2f}]" if c95 else "[   n/a   ]")
                  + f"; family {('%.5f' % level) if level else 'n/a'} "
                  + (f"[{fam[0]:+6.2f},{fam[1]:+6.2f}]" if fam else "[   n/a   ]")
                  + ("  clears" if part["clears_family"] else "  DOES NOT CLEAR"))
        entry["survives"] = all(p["clears_family"] for p in entry["parts"])
        report["claims"].append(entry)
        print(f"  -> {name}: {'survives' if entry['survives'] else 'does not survive'}\n")

    if args.check_method:
        report["method_check"] = parametric_check(2000, 4000, rng)
        for row in report["method_check"]:
            print(f"method check, {row['arm']}: parametric covers "
                  f"{row['parametric_coverage_at_95']:.3f}, double bootstrap covers "
                  f"{row['double_bootstrap_coverage_at_95']:.3f}")

    survivors = [c["claim"] for c in report["claims"] if c["survives"]]
    report["survivors"] = survivors
    attach_quoted(report)
    print(f"{len(survivors)} of {k} survive: {', '.join(survivors)}")
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
