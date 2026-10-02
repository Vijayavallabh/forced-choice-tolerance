#!/usr/bin/env python3
r"""Degenerate runs, and whether the with-data results rest on them (not registered).

Our open-weight agents run BixBench's agent at temperature 1.0, and some episodes end badly: with no
answer, at the step limit, cut by a wall clock, or with a reply that ran to the reply-token limit (the
published protocol's reasoning step sends both Qwen families into a repetition loop). This counts them
and asks whether the with-data results hold without them.

* **Census.** For every configuration with the data on v1.5 -- the seven configurations' first runs, the
  pre-specified test's new runs (two new seeds of each text-protocol agent and two runs of
  Qwen3-235B-A22B), the four later runs of Qwen3-235B-A22B, and gpt-5.1's two runs -- the runs that give no
  answer, reach the step limit, have any reply truncated at the reply-token limit, or are cut by a wall
  clock, and the share of turns truncated; a run is *degenerate* if any of the four holds. The census
  reads the trajectories where the runs were made (``build/``), else the copies that ship: the seven
  configurations and gpt-5.1 ship every turn, the new runs how each episode ended and how many of its
  replies ran to the limit (``replication.turn_counts``), so the census is the same from either.
* **Without them.** Table~\ref{tab:released}'s v1.5 rows (forced and refusal-option grading through the
  released options, minus the tolerance, with the interval ``score_decomposition`` gives it; the misses
  and the correct answers accepted) and the nearest-option rule's $U-P$ on the moved keys and over all
  numeric items (pooled as the pre-specified test pools run sets, at the level the double bootstrap finds
  covers 95%, with a capsule sign-flip test), on all runs and without the degenerate ones.
* **The configurations that use the data.** The same restricted to the three whose accuracy measurably
  rises with the data (Qwen2.5-72B in text, GLM-4.5-Air and Qwen3-30B-A3B as published), and with
  Qwen3-235B-A22B and gpt-5.1 added, each model's runs averaged first so that each model weighs once.
* **Per miss.** For each configuration of Table~\ref{tab:scaling}, gpt-5.1 and the two current agents
  (gpt-6-luna and DeepSeek-V4-Pro, graded by gpt-4o forced as gpt-5.1 is), the share of its graded
  misses that forced-choice grading through the released options accepts, against how often it is within
  5% of the key: as each is graded in Table~\ref{tab:released}, and under one grader for all, gemma-3-27b
  and gpt-4o code-free. Spearman's rho over configurations and with each model counted once, with
  permutation p-values. What forced choice adds is the misses times the share accepted; this asks whether
  the share shrinks as agents improve, or only the misses do.

    python3 degenerate_runs.py        # results/degenerate_runs.json
"""
import collections
import gzip
import itertools
import json
import pathlib
import zlib

import numpy as np

import answer_extraction as ax
import bixbench_withdata as bw
import bracketing as br
import grading_variants_analysis as gva
import randomization
import replication as rp
import score_decomposition as sd
from answer_numbers import graded

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "degenerate_runs.json"
SEED = 20260929
TOL = 0.05
USES_DATA = ("qwen72b", "glm45air-react", "qwen3a3b-react")
LABEL = {"qwen72b": "Qwen2.5-72B, text", "llama70b": "Llama-3.3-70B, text", "gemma27b": "gemma-3-27b, text",
         "qwen72b-react": "Qwen2.5-72B, published", "llama70b-react": "Llama-3.3-70B, published",
         "glm45air-react": "GLM-4.5-Air, published", "qwen3a3b-react": "Qwen3-30B-A3B, published",
         "4o_open_image": "v1.0, gpt-4o, images", "4o_open_no_image": "v1.0, gpt-4o, no images",
         "claude_open_image": "v1.0, Claude 3.5 Sonnet, images",
         "claude_open_no_image": "v1.0, Claude 3.5 Sonnet, no images"}
MODEL = {"qwen72b": "Qwen2.5-72B", "qwen72b-react": "Qwen2.5-72B", "llama70b": "Llama-3.3-70B",
         "llama70b-react": "Llama-3.3-70B", "gemma27b": "gemma-3-27b", "glm45air-react": "GLM-4.5-Air",
         "qwen3a3b-react": "Qwen3-30B-A3B", "qwen3-235b": "Qwen3-235B-A22B", "gpt-5.1-react": "gpt-5.1",
         "gpt-6-luna-react": "gpt-6-luna", "DeepSeek-V4-Pro-react": "DeepSeek-V4-Pro",
         "4o_open_image": "gpt-4o", "4o_open_no_image": "gpt-4o", "claude_open_image": "Claude 3.5 Sonnet",
         "claude_open_no_image": "Claude 3.5 Sonnet"}


def label_of(run):
    """A run set's name as Table~\\ref{tab:scaling} gives it: seeds 2 and 3 are rollouts 1 and 2."""
    base, _, r = run.partition("|")
    if base == "gpt-5.1-react":
        return "v1.5, gpt-5.1" + (f", run {int(r[1:]) + 1}" if r else "")
    if base == "qwen3-235b":
        return f"v1.5, Qwen3-235B-A22B, run {int(r[1:]) + 1}"
    if base in ("gpt-6-luna-react", "DeepSeek-V4-Pro-react"):
        return "v1.5, " + MODEL[base]
    if base.startswith(("4o_", "claude_")):
        return LABEL[base]
    return "v1.5, " + LABEL[base] + (f", seed {int(r[1:]) + 1}" if r else "")


def model_of(run):
    return MODEL[run.partition("|")[0]]


# ---------------------------------------------------------------- the trajectories

def _finish(rec):
    f = rec.get("finish")
    return (f[0] if f else None) if isinstance(f, (list, tuple)) else f


def flags_of(t):
    """One episode: no answer, step limit, a truncated reply (None when the turns did not ship), wall clock."""
    answer = t.get("answer")
    if t.get("log") is not None:
        finished = [rec for rec in t["log"] if rec.get("finish")]
        n_turns, truncated = len(finished), sum(_finish(rec) == "length" for rec in finished)
    else:                         # the compact copies count them when packed (``replication.turn_counts``)
        n_turns, truncated = t.get("turns"), t.get("truncated_turns")
    out = {"no_answer": not (answer and str(answer).strip()), "step_limit": t.get("termination") == "max_steps",
           "truncated_reply": None if truncated is None else truncated > 0,
           "wall_clock": t.get("termination") == "wallclock"}
    out["degenerate"] = any(bool(v) for v in out.values())
    return out, n_turns, truncated, t.get("termination")


def _jsonl_gz(path):
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def trajectories():
    """{run set: {question: trajectory}} of every with-data run set on v1.5 counted here."""
    out = collections.defaultdict(dict)
    for t in bw.load_trajectories(ROOT / "build" / "agent_runs"):
        if bw.run_key(t) in br.RUNS and t["rollout"] == 0 and t["condition"] == "data":
            out[bw.run_key(t)][t["question_id"]] = t
    for t in rp.d2_trajectories():
        if t["condition"] == "data":
            out[f"{bw.run_key(t)}|r{t['rollout']}"][t["question_id"]] = t
    extra = sorted((ROOT / "build" / "agent_runs_q235_extra").glob("*/*/*.json"))
    rows = ([json.loads(p.read_text()) for p in extra] if extra else
            [t for p in sorted((ROOT / "results" / "agent_runs_q235_extra").glob("answers_*.jsonl.gz"))
             for t in _jsonl_gz(p)])
    for t in rows:
        if t["condition"] == "data":
            out[f"{bw.run_key(t)}|r{t['rollout']}"][t["question_id"]] = t
    built = sorted((ROOT / "build" / "openai" / "agent_runs" / "gpt-5.1-react" / "data").glob("*.json"))
    rows = ([json.loads(p.read_text()) for p in built] if built else
            _jsonl_gz(ROOT / "results" / "strong_agent" / "trajectories_gpt-5.1-react.jsonl.gz"))
    for t in rows:
        if t["condition"] == "data" and bw.run_key(t) == "gpt-5.1-react":
            out[f"gpt-5.1-react|r{t['rollout']}"][t["question_id"]] = t
    return out


def census(runs, numeric):
    """Per run set, over all its questions and over the numeric ones: counts of each flag, the turns and the
    turns truncated, and how the episodes ended."""
    out, flags = {}, {}
    for run, by_q in sorted(runs.items()):
        entry = {"label": label_of(run), "model": model_of(run)}
        for part, qs in (("all", sorted(by_q)), ("numeric", sorted(q for q in by_q if q in numeric))):
            c = collections.Counter()
            ends = collections.Counter()
            turns = truncated_turns = 0
            seen_turns = True
            for q in qs:
                f, n_turns, n_trunc, end = flags_of(by_q[q])
                flags[(run, q)] = f
                for k, v in f.items():
                    c[k] += int(bool(v))
                ends[str(end)] += 1
                if n_turns is None:
                    seen_turns = False
                else:
                    turns, truncated_turns = turns + n_turns, truncated_turns + n_trunc
            entry[part] = {"runs": len(qs), **{k: c[k] for k in ("no_answer", "step_limit", "truncated_reply",
                                                                  "wall_clock", "degenerate")},
                           "turns": turns if seen_turns else None,
                           "truncated_turns": truncated_turns if seen_turns else None,
                           "truncated_turn_share": (100.0 * truncated_turns / turns) if seen_turns and turns else None,
                           "truncation_visible": seen_turns, "endings": dict(sorted(ends.items()))}
        out[run] = entry
    return out, flags


# ---------------------------------------------------------------- Table 2's rows

def by_item(rows, value, group=None):
    """Each question's value: the mean over its runs or, with ``group``, over each group's runs and then over
    the groups, so that each group weighs once."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        v = value(r)
        if v is not None:
            acc[r["q"]][group(r) if group else 0].append(v)
    return {q: float(np.mean([np.mean(v) for v in g.values()])) for q, g in acc.items()}


def table1_row(runs, reading, capsule, seed, group=None):
    """Score minus the tolerance (``replication.calibrated`` at 2000 x 5000, as ``score_decomposition``), the
    tolerance, the misses accepted (``score_decomposition.miss_rates`` when the runs are pooled as Table 2
    pools them; the same ratio of sums over questions when ``group`` weighs each group once) and the correct
    answers accepted."""
    hit = lambda r: float(bool(graded(r["answer"], r["options"][0], TOL))) if r["answer"] else 0.0
    read = [r for r in runs if sd.named_share(r, reading) is not None]
    if not read:
        return None
    named = lambda r: sd.named_share(r, reading)
    diff = by_item(read, lambda r: named(r) - hit(r), group)
    hits = by_item(read, lambda r: hit(r) * named(r), group)
    tol = by_item(read, hit, group)
    miss = lambda r: (1.0 - hit(r)) * float(ax.category(r["answer"]) != "empty")
    if group is None:
        m = sd.miss_rates(read, reading, hit, capsule, reps=4000)
        accepted, at_chance = m["all"]["accepted"], m["at_chance"]
    else:
        num, den = by_item(read, lambda r: miss(r) * named(r), group), by_item(read, miss, group)
        accepted = 100.0 * sum(num.values()) / sum(den.values()) if sum(den.values()) else None
        at_chance = (20.0 if reading == "refusal" else 25.0) * sum(den.values()) / len(den)
    return {"n_runs": len(read), "n_items": len(tol), "tolerance": sd.mean_over_items(tol),
            "score": sd.mean_over_items(by_item(read, named, group)),
            "score_minus_tolerance": rp.calibrated(diff, capsule, np.random.default_rng(seed), 2000, 5000),
            "misses_accepted": accepted, "at_chance": at_chance,
            "correct_accepted": (100.0 * sum(hits.values()) / sum(tol.values())) if sum(tol.values()) else None,
            "weighting": "each group once" if group else "each question's runs pooled"}


def generic(runs, readings):
    """Runs whose Table-1 readings have different names (a configuration's own model, gemma-3-27b, gpt-4o),
    under the two names ``forced`` and ``refusal``."""
    out = []
    for r in runs:
        reads = {}
        for new, old in readings.items():
            if old in r["reads"]:
                reads[new] = r["reads"][old]
        out.append({**r, "reads": reads})
    return out


# ---------------------------------------------------------------- the rule's U - P

def rule_gain(answers, sets, capsule, seed, heavy=True):
    """{run set: [(question, answer)]} -> the nearest-option rule's U - P per item, each item's run sets
    averaged (the pre-specified test's pooling, ``replication.gains``), on the moved keys and over all numeric
    items: the calibrated interval and the capsule sign-flip test."""
    with_p = {q: s for q, s in sets.items() if "placebo" in s and "repaired" in s}
    _, pooled = rp.gains(answers, with_p, "repaired", "placebo")
    moved = {q for q, s in with_p.items() if rp.group_of(s) == "moved to an edge"}
    outer, inner = (4000, 20000) if heavy else (2000, 5000)
    out = {}
    for part, items in (("moved", {q: v for q, v in pooled.items() if q in moved}), ("all", pooled)):
        rng = np.random.default_rng(seed + zlib.crc32(part.encode()))
        # fewer than two capsules have no interval: the mean alone
        out[part] = rp.calibrated(items, capsule, rng, outer, inner) or {
            "mean": 100.0 * float(np.mean(list(items.values()))) if items else None, "lo": None, "hi": None,
            "level_capped": True, "n_items": len(items), "n_clusters": len({capsule[q] for q in items})}
        sf = randomization.signflip(items, capsule)
        out[part]["signflip_p"] = None if sf is None else float(sf["p"])
    return out


def by_model(answers):
    """Each model's run sets averaged into one, per question: {model: [(question, answer)]} is not enough,
    since an item's value is a mean over runs, so return the per-item U - P directly."""
    return {m: {n: a for n, a in answers.items() if model_of(n) == m} for m in {model_of(n) for n in answers}}


def rule_gain_models(answers, sets, capsule, seed, heavy=True):
    """As ``rule_gain``, each model's run sets averaged first, then the models."""
    with_p = {q: s for q, s in sets.items() if "placebo" in s and "repaired" in s}
    per_model = {}
    for m, sub in by_model(answers).items():
        _, pooled = rp.gains(sub, with_p, "repaired", "placebo")
        per_model[m] = pooled
    acc = collections.defaultdict(list)
    for pooled in per_model.values():
        for q, v in pooled.items():
            acc[q].append(v)
    pooled = {q: float(np.mean(v)) for q, v in acc.items()}
    moved = {q for q, s in with_p.items() if rp.group_of(s) == "moved to an edge"}
    outer, inner = (4000, 20000) if heavy else (2000, 5000)
    out = {"models": sorted(per_model)}
    for part, items in (("moved", {q: v for q, v in pooled.items() if q in moved}), ("all", pooled)):
        rng = np.random.default_rng(seed + zlib.crc32(part.encode()))
        # fewer than two capsules have no interval: the mean alone
        out[part] = rp.calibrated(items, capsule, rng, outer, inner) or {
            "mean": 100.0 * float(np.mean(list(items.values()))) if items else None, "lo": None, "hi": None,
            "level_capped": True, "n_items": len(items), "n_clusters": len({capsule[q] for q in items})}
        sf = randomization.signflip(items, capsule)
        out[part]["signflip_p"] = None if sf is None else float(sf["p"])
    return out


# ---------------------------------------------------------------- per miss against accuracy

def ranks(x):
    x = np.asarray(x, float)
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):                    # ties share their mean rank
        idx = np.where(x == v)[0]
        r[idx] = r[idx].mean()
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_p(x, y, draws=200000, seed=SEED):
    """Spearman's rho against permutations of one variable, two-sided: exact over every permutation up to nine
    points, else ``draws`` random ones."""
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    rho = float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))
    if len(x) <= 9:
        perms = np.array(list(itertools.permutations(range(len(x)))))
    else:
        rng = np.random.default_rng(seed)
        perms = rng.permuted(np.tile(np.arange(len(x)), (draws, 1)), axis=1)
    stats = (ry[perms] @ rx) / np.sqrt((rx @ rx) * (ry @ ry))
    return float(np.mean(np.abs(stats) >= abs(rho) - 1e-12))


def per_miss_rows():
    """Every configuration of Table~\\ref{tab:scaling}, gpt-5.1 and the two current agents, through the released
    options: its share
    within 5%, and the share of its graded misses accepted forced -- as Table~\\ref{tab:released} grades it and
    by gemma-3-27b and gpt-4o code-free -- with the share of its misses whose single nearest option is the key."""
    hit = lambda r: float(bool(graded(r["answer"], r["options"][0], TOL))) if r["answer"] else 0.0
    groups = collections.defaultdict(list)
    d1 = sd.d1_runs()
    cap = {r["q"]: r["capsule"] for r in d1}
    for r in d1:
        groups[r["run"]].append(generic([r], {"as graded": "published forced"})[0])
    d0 = sd.d0_runs()
    for r in d0:
        groups[r["run"]].append(generic([r], {"as graded": "family forced"})[0])
        cap[r["q"]] = r["capsule"]
    for r in sd.d2_runs():
        groups[r["run"]].append(generic([r], {"as graded": "gemma forced"})[0])
        cap[r["q"]] = r["capsule"]
    for r in sd.d3_runs():
        groups["gpt-5.1-react"].append(generic([r], {"as graded": "gpt-4o forced"})[0])
        cap[r["q"]] = r["capsule"]
    # the two current agents, graded by gpt-4o forced as gpt-5.1 is (frontier_agents.py)
    import frontier_agents as fa
    for run, label in fa.MODELS:
        if label in fa.REPORTED:
            for r in fa.as_runs([x for x in fa.load_rows() if x["model"] == run], label, bw.option_sets()):
                groups[run].append(generic([r], {"as graded": "gpt-4o forced"})[0])
                cap[r["q"]] = r["capsule"]
    # one grader for all: the code-free gradings of the same runs, through the released options, matched run by
    # run: a published run by its configuration, question and index; a v1.5 run by its run set and question
    v10, v15 = rp.v10_sets(), bw.option_sets()

    def ident(r):
        part = r["key"].split("|")
        if part[0] == "D1":
            return ("D1", r["run"], r["q"], int(part[-1]))
        if part[0] == "D3":
            return ("V15", f"gpt-5.1-react|r{part[-1]}", r["q"])
        return ("V15", r["run"], r["q"])

    for reader, name in (("gemma27b", "gemma-3-27b code-free"), ("gpt-4o", "gpt-4o code-free")):
        reads = {}
        for g in gva.load(reader, "codefree", "forced"):
            if g["condition"] == "data" and "released" in g["reads"]:
                k = ("D1", g["run"], g["q"], g["i"]) if g["set"] == "D1" else ("V15", g["run"], g["q"])
                reads[k] = (g["answer"], g["reads"]["released"])
        for run, rows in groups.items():
            for r in rows:
                got = reads.get(ident(r))
                if got is not None:
                    assert (got[0] or "") == (r["answer"] or ""), (ident(r), got[0], r["answer"])
                    r["reads"][name] = got[1]
    out = []
    for run, rows in groups.items():
        sets = v10 if run in LABEL and run.startswith(("4o_", "claude_")) else v15
        rows = [r for r in rows if r["q"] in sets]
        entry = {"run": run, "label": label_of(run),
                 "model": model_of(run), "n_runs": len(rows),
                 "within_5pct": sd.mean_over_items(sd.per_item(rows, hit))}
        for reading in ("as graded", "gemma-3-27b code-free", "gpt-4o code-free"):
            read = [r for r in rows if sd.named_share(r, reading) is not None]
            # an empty answer is scored wrong under every reading; a configuration a reading never graded has none
            if not any(reading in r["reads"] for r in read):
                continue
            m = sd.miss_rates(read, reading, hit, cap, reps=2000)
            entry[reading] = {"misses_accepted": m["all"]["accepted"], "n_runs": len(read),
                              "key_nearest_share_of_misses": (100.0 * m["nearest"]["share_of_runs"]
                                                              / m["all"]["share_of_runs"]),
                              "accepted_if_key_nearest": m["nearest"]["accepted"],
                              "accepted_if_other_nearest": m["other"]["accepted"],
                              "accepted_if_neither": m["neither"]["accepted"],
                              "score_minus_tolerance": sd.mean_over_items(sd.per_item(
                                  read, lambda r: sd.named_share(r, reading) - hit(r)))}
        out.append(entry)
    return sorted(out, key=lambda e: e["within_5pct"])


def correlations(rows):
    out = {}
    for reading in ("as graded", "gemma-3-27b code-free", "gpt-4o code-free"):
        have = [r for r in rows if reading in r]
        for what in ("misses_accepted", "key_nearest_share_of_misses", "accepted_if_key_nearest",
                     "accepted_if_other_nearest", "score_minus_tolerance"):
            pts = [(r["within_5pct"], r[reading][what]) for r in have if r[reading][what] is not None]
            x, y = zip(*pts)
            models = collections.defaultdict(list)
            for r in have:
                if r[reading][what] is not None:
                    models[r["model"]].append((r["within_5pct"], r[reading][what]))
            mx = [float(np.mean([a for a, _ in v])) for v in models.values()]
            my = [float(np.mean([b for _, b in v])) for v in models.values()]
            out[f"{reading}|{what}"] = {
                "configurations": {"n": len(x), "rho": spearman(x, y), "p": permutation_p(x, y)},
                "models": {"n": len(mx), "rho": spearman(mx, my), "p": permutation_p(mx, my)}}
    return out


# ---------------------------------------------------------------- the report

def main():
    sets = bw.option_sets()
    items = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    capsule = {q: it["capsule_uuid"] for q, it in items.items()}
    runs = trajectories()
    report = {"seed": SEED, "tolerance": TOL, "not_registered": True,
              "definition": "degenerate: no answer, the step limit reached, any reply truncated at the reply-token "
                            "limit, or cut by a wall clock"}
    report["census"], flags = census(runs, set(sets))
    ok = lambda run, q: not flags[(run, q)]["degenerate"]
    for run, e in report["census"].items():
        n = e["numeric"]
        print(f"{e['label']:42s} numeric {n['runs']:4d} | no answer {n['no_answer']:3d} step limit {n['step_limit']:3d} "
              f"truncated {n['truncated_reply'] if n['truncation_visible'] else '--':>3} wall clock {n['wall_clock']:2d} "
              f"| degenerate {n['degenerate']:3d} | turns truncated {n['truncated_turn_share']}", flush=True)

    # Table 2's rows, all runs and without the degenerate ones
    d0, d2, d3 = sd.d0_runs(), sd.d2_runs(), sd.d3_runs()
    run_q = lambda r: (r["run"] if not r["key"].startswith("D3") else f"gpt-5.1-react|r{r['key'].split('|')[-1]}", r["q"])
    keep = lambda rows: [r for r in rows if ok(*run_q(r))]
    blocks = {"seven configurations": (d0, {"forced": "family forced", "refusal": "family may-decline"}),
              "new seeds": ([r for r in d2 if r["group"] == "new seeds"],
                            {"forced": "gemma forced", "refusal": "gemma may-decline"}),
              "Qwen3-235B-A22B": ([r for r in d2 if r["group"] == "Qwen3-235B-A22B"],
                                  {"forced": "gemma forced", "refusal": "gemma may-decline"}),
              "gpt-5.1": (d3, {"forced": "gpt-4o forced", "refusal": "gpt-4o may-decline"}),
              "use the data: three": ([r for r in d0 if r["run"] in USES_DATA],
                                      {"forced": "family forced", "refusal": "family may-decline"})}
    report["table1"] = {}
    for name, (rows, readings) in blocks.items():
        rows = generic(rows, readings)
        report["table1"][name] = {}
        for which, subset in (("all runs", rows), ("without degenerate", keep(rows))):
            report["table1"][name][which] = {
                how: table1_row(subset, how, capsule, SEED + zlib.crc32(f"{name}|{which}|{how}".encode()))
                for how in ("forced", "refusal")}
            f = report["table1"][name][which]["forced"]
            d = report["table1"][name][which]["refusal"]
            print(f"T1 {name:24s} {which:20s} n={f['n_runs']:4d} tol {f['tolerance']:5.1f} forced "
                  f"{f['score_minus_tolerance']['mean']:+5.1f} [{f['score_minus_tolerance']['lo']:+5.1f},"
                  f"{f['score_minus_tolerance']['hi']:+5.1f}] miss acc {f['misses_accepted']:5.1f} correct acc "
                  f"{f['correct_accepted']} | refusal {d['score_minus_tolerance']['mean']:+5.1f} "
                  f"[{d['score_minus_tolerance']['lo']:+5.1f},{d['score_minus_tolerance']['hi']:+5.1f}] "
                  f"miss acc {d['misses_accepted']:5.1f}", flush=True)
    # the three that use the data, with Qwen3-235B-A22B's two test runs and gpt-5.1 added, each model once
    mixed = (generic([r for r in d0 if r["run"] in USES_DATA], {"forced": "family forced",
                                                                "refusal": "family may-decline"})
             + generic([r for r in d2 if r["group"] == "Qwen3-235B-A22B"], {"forced": "gemma forced",
                                                                             "refusal": "gemma may-decline"})
             + generic(d3, {"forced": "gpt-4o forced", "refusal": "gpt-4o may-decline"}))
    report["table1"]["use the data: five models"] = {}
    for which, subset in (("all runs", mixed), ("without degenerate", keep(mixed))):
        report["table1"]["use the data: five models"][which] = {
            how: table1_row(subset, how, capsule, SEED + zlib.crc32(f"five|{which}|{how}".encode()),
                            group=lambda r: model_of(r["run"]))
            for how in ("forced", "refusal")}
        f = report["table1"]["use the data: five models"][which]["forced"]
        print(f"T1 five models {which:20s} forced {f['score_minus_tolerance']['mean']:+5.1f} "
              f"[{f['score_minus_tolerance']['lo']:+5.1f},{f['score_minus_tolerance']['hi']:+5.1f}] "
              f"miss acc {f['misses_accepted']:5.1f}", flush=True)

    # the nearest-option rule's U - P, all runs and without the degenerate ones
    answers = {run: [(q, t.get("answer")) for q, t in sorted(by_q.items()) if q in sets]
               for run, by_q in runs.items()}
    pools = {"seven configurations": [r for r in answers if r in br.RUNS],
             "new seeds": [r for r in answers if "|" in r and r.split("|")[0] in ("qwen72b", "llama70b", "gemma27b")],
             "Qwen3-235B-A22B, test runs": ["qwen3-235b|r0", "qwen3-235b|r1"],
             "Qwen3-235B-A22B, six runs": [f"qwen3-235b|r{i}" for i in range(6)],
             "gpt-5.1": ["gpt-5.1-react|r0", "gpt-5.1-react|r1"],
             "use the data: three": list(USES_DATA)}
    report["rule_u_minus_p"] = {}
    for name, names in pools.items():
        report["rule_u_minus_p"][name] = {"run_sets": names}
        for which in ("all runs", "without degenerate"):
            sub = {n: [(q, a) for q, a in answers[n] if which == "all runs" or ok(n, q)] for n in names}
            g = rule_gain(sub, sets, capsule, SEED + zlib.crc32(f"{name}|{which}".encode()))
            report["rule_u_minus_p"][name][which] = g
            print(f"U-P {name:30s} {which:20s} moved {g['moved']['mean']:+5.1f} [{g['moved']['lo']:+5.1f},"
                  f"{g['moved']['hi']:+5.1f}] p={g['moved']['signflip_p']:.4f} | all {g['all']['mean']:+5.1f} "
                  f"[{g['all']['lo']:+5.1f},{g['all']['hi']:+5.1f}]", flush=True)
    five = list(USES_DATA) + [f"qwen3-235b|r{i}" for i in range(6)] + ["gpt-5.1-react|r0", "gpt-5.1-react|r1"]
    report["rule_u_minus_p"]["use the data: five models"] = {"run_sets": five, "pooling": "each model's run sets "
                                                             "averaged first, then the models"}
    for which in ("all runs", "without degenerate"):
        sub = {n: [(q, a) for q, a in answers[n] if which == "all runs" or ok(n, q)] for n in five}
        g = rule_gain_models(sub, sets, capsule, SEED + zlib.crc32(f"five|{which}".encode()))
        report["rule_u_minus_p"]["use the data: five models"][which] = g
        print(f"U-P five models {which:20s} moved {g['moved']['mean']:+5.1f} [{g['moved']['lo']:+5.1f},"
              f"{g['moved']['hi']:+5.1f}] p={g['moved']['signflip_p']:.4f} | all {g['all']['mean']:+5.1f}", flush=True)

    # per miss against accuracy
    rows = per_miss_rows()
    report["per_miss"] = {"configurations": rows, "correlations": correlations(rows)}
    for r in rows:
        cells = " | ".join(f"{k[:6]} {r[k]['misses_accepted']:5.1f} (key-nearest {r[k]['key_nearest_share_of_misses']:4.1f})"
                           for k in ("as graded", "gemma-3-27b code-free", "gpt-4o code-free") if k in r)
        print(f"{r['label']:42s} within {r['within_5pct']:5.1f} | {cells}")
    for k, v in report["per_miss"]["correlations"].items():
        print(f"rho {k:58s} configs {v['configurations']['rho']:+.2f} (p={v['configurations']['p']:.3f}, "
              f"n={v['configurations']['n']}) | models {v['models']['rho']:+.2f} (p={v['models']['p']:.3f}, "
              f"n={v['models']['n']})")
    OUT.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
