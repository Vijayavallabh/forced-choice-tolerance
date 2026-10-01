#!/usr/bin/env python3
r"""How the design of numeric options sets what forced-choice grading accepts (not registered).

Proposition 1 states the trade-off at four options: options that reject misses on both sides reveal the key's
rank, and a uniform rank leaves $2/k$ of the keys at an extreme, where every miss on the open side is nearest
the key. This measures the trade-off as a curve over the designs an author could choose, on the runs the paper
already has. Each design keeps every item's key and redraws its distractors; for each design it reports

* the leak: what a rule that ignores the question scores by always selecting one rank, the rank chosen on
  held-out capsules (``channel_survey.grouped_cv_credit``), in points over $1/k$;
* the cost: the share of misses, and of answers within 5% of the key, that the nearest-option rule accepts on
  the same with-data runs, the number read as the tolerance reads it (``answer_extraction.nearest_is_key_last``);
* for the designs graded by models (``read``), what code-free graders accept (``analyse``).

Designs:

* ``k{4,6,8,10}-uniform`` and ``-middle``: $k$ options, the key's rank drawn uniformly from all $k$ ranks, or
  from the $k-2$ middle ones, so that no key is extreme; distractors spaced as $P$ and $U$ are, multiplicatively
  at the released distractors' mean relative distance $s$ (``mcq_audit.redraw_row``'s scheme), each written with
  the significant digits of a released distractor, cycled.
* ``k4-released``: the same generator at the key's released rank (the rank-preserving rewrite, redrawn).
* ``s{0.1,0.2,0.5,1}``: the same, at a fixed relative spacing $s$ instead of the released one.
* ``errors-released``, ``errors-uniform``, ``errors-middle``: four options whose distractors are the wrong numbers
  other agents submitted to the same question, the most frequent first, with the key at its released rank, a
  uniform rank or a middle rank. A graded run's own model is left out of the pool its distractors come from, so
  an agent is never graded against its own errors; a side the pool cannot fill is filled by the generator.

    python3 option_design.py build          # build/option_design/designs.json.gz
    python3 option_design.py rule           # results/option_design.json (leak and the rule's cost)
    python3 option_design.py read --reader gemma27b=http://127.0.0.1:8103 --designs k4-released,k4-uniform
    python3 option_design.py analyse        # adds the graders' acceptance to results/option_design.json
"""
import argparse
import asyncio
import collections
import gzip
import hashlib
import json
import math
import pathlib
import random
import zlib

import numpy as np

import answer_extraction as ax
import bixbench_withdata as bw
import channel_survey as cs
import digit_matched as dm
import mcq_audit as ma
import replication as rp
from answer_numbers import graded
from option_artifacts import parse_number

ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT / "build" / "option_design"
DESIGNS_PATH = WORK / "designs.json.gz"
OUT = ROOT / "results" / "option_design.json"
# the shipped copies: the designs' option sets and the code-free graders' reads, gzipped
SHIPPED = ROOT / "results" / "option_design"
TOL = 0.05
SEED = 20260930
KS = (4, 6, 8, 10)
SPACINGS = (0.1, 0.2, 0.5, 1.0)
LETTERS = "ABCDEFGHIJ"
FAMILY = (("qwen3-235b", "Qwen3-235B-A22B"), ("qwen3a3b", "Qwen3-30B-A3B"), ("qwen72b", "Qwen2.5-72B"),
          ("llama70b", "Llama-3.3-70B"), ("gemma27b", "gemma-3-27b"), ("glm45air", "GLM-4.5-Air"),
          ("gpt-5.1", "gpt-5.1"), ("4o_", "gpt-4o"), ("claude_", "Claude 3.5 Sonnet"))


def family(run):
    return next(name for prefix, name in FAMILY if run.startswith(prefix))


def item_rng(*parts):
    return random.Random(zlib.crc32("|".join(map(str, parts)).encode()) ^ SEED)


# ---------------------------------------------------------------- items and runs

def load_items():
    """{release: {question: {options (key first), capsule, question}}} for both releases' numeric items."""
    doc = rp.load_extract()
    v10 = {q: {"options": s["released"], "capsule": doc["items"][q]["capsule"], "question": doc["items"][q]["question"]}
           for q, s in rp.v10_sets().items()}
    rows = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    v15 = {q: {"options": s["released"], "capsule": rows[q]["capsule_uuid"], "question": rows[q]["question"]}
           for q, s in dm.v15_sets().items()}
    return {"v1.0": v10, "v1.5": v15}


def load_runs():
    """Every run on a numeric item: release, run set, model, condition, question, answer. The with-data runs are
    graded; runs with and without the data both feed the pools of wrong numbers."""
    import bracketing as br
    import q235_extra as qx
    import strong_agent as sa
    items = load_items()
    out = []
    doc = rp.load_extract()
    for name in rp.OPEN_RUNS:
        seen = collections.Counter()
        for q, answer, _ in doc["open_runs"][name]:
            if q in items["v1.0"]:
                out.append({"release": "v1.0", "group": family(name), "run": name, "q": q, "i": seen[q],
                            "condition": "data", "answer": answer})
                seen[q] += 1
    v15 = items["v1.5"]
    for run in br.RUNS:
        for r in br.rows_of(run):
            if r["question_id"] in v15:
                out.append({"release": "v1.5", "group": "seven configurations", "run": run, "q": r["question_id"],
                            "i": 0, "condition": r["condition"], "answer": r["answer"]})
    for name, conds in rp.d2_runs().items():
        group = "Qwen3-235B-A22B" if name.startswith("qwen3-235b|") else "new seeds"
        for cond, pairs in conds.items():
            for q, answer in pairs:
                if q in v15:
                    out.append({"release": "v1.5", "group": group, "run": name, "q": q, "i": 0,
                                "condition": cond, "answer": answer})
    for t in qx.trajectories():
        if t["question_id"] in v15:
            out.append({"release": "v1.5", "group": "Qwen3-235B-A22B, four more runs",
                        "run": f"qwen3-235b|r{t['rollout']}", "q": t["question_id"], "i": 0,
                        "condition": t["condition"], "answer": t.get("answer")})
    for r in sa.load_rows():
        out.append({"release": "v1.5", "group": "gpt-5.1", "run": f"gpt-5.1|r{r['rollout']}", "q": r["question_id"],
                    "i": 0, "condition": "data", "answer": r["answer"]})
    for r in out:
        r["model"] = family(r["run"])
        r["key"] = f"{r['release']}|{r['run']}|{r['condition']}|{r['q']}|{r['i']}"
    return items, out


# ---------------------------------------------------------------- the generator

def values_of(options):
    return [parse_number(o) for o in options]


def rank_of(options):
    """The key's rank among the values (0 = smallest)."""
    v = values_of(options)
    return sorted(range(len(v)), key=lambda i: v[i]).index(0)


def spaced(options, rng, k, target, step=None):
    """``k - 1`` distractors around the key ``options[0]``, ``target`` of them below it, spaced as
    ``mcq_audit.redraw_row`` spaces P's and U's: multiplicatively when the released values share the key's sign,
    at ``|y|(1 + s j u)`` and ``|y|/(1 + s j u)`` counting outward, u uniform on [0.6, 1.4], with s the released
    mean relative distance (at least 0.05) or ``step``; additively otherwise. Each is written with the significant
    digits of a released distractor, cycled; a draw that collides, changes sign or misses the rank is redrawn with
    the spread widened by half, up to 24 times, the last twelve without matching digits."""
    values = values_of(options)
    key = values[0]
    gaps = [abs(v - key) for v in values[1:] if abs(v - key) > 0]
    templates = [options[1 + i % (len(options) - 1)] for i in range(k - 1)]
    same_sign = key != 0 and all((v > 0) == (key > 0) for v in values)
    relative = (step if step is not None else
                max(float(np.mean([g / abs(key) for g in gaps])) if (gaps and key) else 0.25, 0.05))
    scale = float(np.mean(gaps)) if gaps else (abs(key) * 0.1 or 0.1)
    for attempt in range(24):
        widen = 1.0 + 0.5 * (attempt % 12)
        if same_sign:
            s, sign, magnitude = relative * widen, (1.0 if key > 0 else -1.0), abs(key)
            n_small = target if key > 0 else k - 1 - target
            smaller = [sign * magnitude / (1 + s * (i + 1) * rng.uniform(0.6, 1.4)) for i in range(n_small)]
            larger = [sign * magnitude * (1 + s * (i + 1) * rng.uniform(0.6, 1.4)) for i in range(k - 1 - n_small)]
            drawn = smaller + larger
        else:
            spread = (step * (abs(key) or scale) if step is not None else scale) * widen
            drawn = ([key - spread * (i + 1) * rng.uniform(0.6, 1.4) for i in range(target)]
                     + [key + spread * (i + 1) * rng.uniform(0.6, 1.4) for i in range(k - 1 - target)])
        match = attempt < 12
        styles = templates
        if match:
            size = lambda text: abs(parse_number(text) or 0.0)
            styles = [templates[i] for i in sorted(range(len(templates)), key=lambda i: size(templates[i]))]
            drawn = sorted(drawn, key=abs)
        rendered = [ma.format_like(t, v, match) for t, v in zip(styles, drawn)]
        parsed = [parse_number(x) for x in rendered]
        if any(v is None for v in parsed):
            continue
        if match and any(ma.significant_digits(x) != ma.significant_digits(t) for x, t in zip(rendered, styles)):
            continue
        if same_sign and any((v > 0) != (key > 0) for v in parsed):
            continue
        if len(set(parsed) | {key}) < k:
            continue
        out = [options[0]] + rendered
        if rank_of(out) == target:
            return out
    return None


def cluster_numbers(numbers, key):
    """Wrong numbers merged when within 5% of each other: [(value, count)], most frequent first."""
    out = []
    for v in sorted(numbers):
        if out and abs(v - out[-1][0]) <= TOL * max(abs(v), abs(out[-1][0])):
            out[-1][1].append(v)
        else:
            out.append((v, [v]))
    clusters = [(float(np.median(vs)), len(vs)) for _, vs in out]
    return sorted(clusters, key=lambda c: (-c[1], abs(math.log(abs(c[0]) / abs(key))) if c[0] and key else 0.0))


def from_errors(options, pool, rng, target, k=4):
    """Four options whose distractors are the most frequent wrong numbers in ``pool`` on each side of the key,
    ``target`` below it, written with the key's significant digits; a side the pool cannot fill is filled by
    ``spaced``'s scheme. Returns (options, number of distractors taken from the pool) or None."""
    key_text, key = options[0], values_of(options)[0]
    if not key:
        return None
    chosen_below, chosen_above = [], []
    for value, _ in cluster_numbers([v for v in pool if (v > 0) == (key > 0) and v != 0], key):
        text = ma.format_like(key_text, value, True)
        v = parse_number(text)
        if v is None or (v > 0) != (key > 0) or abs(v - key) <= TOL * abs(key):
            continue
        if any(abs(v - parse_number(x)) <= TOL * max(abs(v), abs(parse_number(x)))
               for x in chosen_below + chosen_above):
            continue
        side = chosen_below if v < key else chosen_above
        if len(side) < (target if side is chosen_below else k - 1 - target):
            side.append(text)
        if len(chosen_below) == target and len(chosen_above) == k - 1 - target:
            break
    from_pool = len(chosen_below) + len(chosen_above)
    if from_pool < k - 1:
        # fill the short side outward from the farthest chosen value, by the generator's multiplicative scheme
        filler = spaced(options, rng, k, target)
        if filler is None:
            return None
        fill = sorted(filler[1:], key=lambda x: parse_number(x))
        below_fill = [x for x in fill if parse_number(x) < key][::-1]
        above_fill = [x for x in fill if parse_number(x) > key]
        taken = [parse_number(x) for x in chosen_below + chosen_above]
        for need, side, extra in ((target, chosen_below, below_fill), (k - 1 - target, chosen_above, above_fill)):
            for x in extra:
                if len(side) >= need:
                    break
                v = parse_number(x)
                if all(abs(v - t) > TOL * max(abs(v), abs(t)) for t in taken):
                    side.append(x)
                    taken.append(v)
        if len(chosen_below) != target or len(chosen_above) != k - 1 - target:
            return None
    out = [key_text] + chosen_below + chosen_above
    if len({parse_number(x) for x in out}) < k or rank_of(out) != target:
        return None
    return out, from_pool


def allowed(policy, k, released_rank):
    return [released_rank] if policy == "released" else list(range(k)) if policy == "uniform" else list(range(1, k - 1))


def draw_target(policy, k, rng, released_rank):
    """The rank drawn for an item, then the policy's other ranks in random order: an item no draw can place at
    the drawn rank takes another the policy allows, as $U$ does, rather than leaving the file."""
    ranks = allowed(policy, k, released_rank)
    first = ranks[rng.randrange(len(ranks))]
    rest = [r for r in ranks if r != first]
    rng.shuffle(rest)
    return [first] + rest


def first_placed(make, targets):
    for t in targets:
        got = make(t)
        if got is not None:
            return got, t
    return None, None


def wrong_numbers(runs, items):
    """{(release, question): [(model, value)]}: every number a run submitted that is more than 5% from the key."""
    out = collections.defaultdict(list)
    for r in runs:
        key = items[r["release"]][r["q"]]["options"][0]
        a = ax.last_number(r["answer"], key) if r["answer"] else None
        if a is None or graded(str(r["answer"]), key, TOL):
            continue
        out[(r["release"], r["q"])].append((r["model"], a))
    return out


def build(_args):
    items, runs = load_runs()
    pools = wrong_numbers(runs, items)
    models = {rel: sorted({r["model"] for r in runs if r["release"] == rel and r["condition"] == "data"})
              for rel in items}
    designs = {rel: {} for rel in items}
    report = {rel: {} for rel in items}
    for rel, its in items.items():
        for q, it in sorted(its.items()):
            released_rank = rank_of(it["options"])
            for k in KS:
                for policy in ("uniform", "middle") + (("released",) if k == 4 else ()):
                    name = f"k{k}-{policy}"
                    rng = item_rng(rel, q, name)
                    targets = draw_target(policy, k, rng, released_rank)
                    got, _ = first_placed(lambda t: spaced(it["options"], rng, k, t), targets)
                    designs[rel].setdefault(name, {})[q] = got
            for s in SPACINGS:
                for k in (4, 8):
                    for policy in ("uniform", "middle"):
                        name = f"k{k}-{policy}-s{s:g}"
                        rng = item_rng(rel, q, name)
                        targets = draw_target(policy, k, rng, released_rank)
                        got, _ = first_placed(lambda t: spaced(it["options"], rng, k, t, step=s), targets)
                        designs[rel].setdefault(name, {})[q] = got
            # error-modelled: one option set per graded model, from the other models' wrong numbers; and the
            # author's file, from every model's. The rank is drawn once per item, for every pool alike
            for policy in ("released", "uniform", "middle"):
                name = f"errors-{policy}"
                targets = draw_target(policy, 4, item_rng(rel, q, name), released_rank)
                entry = {}
                for left_out in models[rel] + [None]:
                    pool = [v for m, v in pools.get((rel, q), []) if m != left_out]
                    rng = item_rng(rel, q, name, left_out)
                    got, _ = first_placed(lambda t: from_errors(it["options"], pool, rng, t), targets)
                    entry["all" if left_out is None else left_out] = (
                        None if got is None else {"options": got[0], "from_pool": got[1]})
                designs[rel].setdefault(name, {})[q] = entry
        for name, sets in designs[rel].items():
            if name.startswith("errors-"):
                built = [e for e in sets.values() if e.get("all")]
                report[rel][name] = {"built": len(built), "left_out": len(sets) - len(built),
                                     "from_pool_share": float(np.mean([e["all"]["from_pool"] / 3 for e in built]))}
            else:
                report[rel][name] = {"built": sum(v is not None for v in sets.values()),
                                     "left_out": sum(v is None for v in sets.values())}
    WORK.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=DESIGNS_PATH.open("wb")) as fh:
        fh.write(json.dumps({"seed": SEED, "designs": designs, "models": models}, sort_keys=True).encode())
    for rel in report:
        for name, r in report[rel].items():
            print(rel, name, r)
    print(f"wrote {DESIGNS_PATH.relative_to(ROOT)}")


def load_designs():
    with gzip.open(DESIGNS_PATH if DESIGNS_PATH.exists() else SHIPPED / "designs.json.gz", "rt") as fh:
        return json.load(fh)


def read_files():
    """{reader: lines} of the code-free graders' reads: the working copies, else the shipped gzipped ones."""
    work = sorted(WORK.glob("reads_*.jsonl"))
    if work:
        return {p.stem[len("reads_"):]: p.open() for p in work}
    return {p.name[len("reads_"):-len(".jsonl.gz")]: gzip.open(p, "rt") for p in sorted(SHIPPED.glob("reads_*.jsonl.gz"))}


def options_for(designs, rel, name, q, model):
    """The option set a run of ``model`` is graded through under design ``name``, or None."""
    entry = designs["designs"][rel][name].get(q)
    if entry is None:
        return None
    if name.startswith("errors-"):
        got = entry.get(model)
        return None if got is None else got["options"]
    return entry


def author_file(designs, rel, name):
    """{question: options} as an author would publish the design: for the error-modelled designs, from every
    model's wrong numbers."""
    out = {}
    for q, entry in designs["designs"][rel][name].items():
        if name.startswith("errors-"):
            entry = entry.get("all") and entry["all"]["options"]
        if entry:
            out[q] = entry
    return out


# ---------------------------------------------------------------- the rule

def per_item(rows, value):
    acc = collections.defaultdict(list)
    for r in rows:
        v = value(r)
        if v is not None:
            acc[r["q"]].append(v)
    return {q: float(np.mean(v)) for q, v in acc.items()}


def interval(per_q, capsule, rng=None, level=0.95, reps=4000):
    qs = sorted(per_q)
    if not qs:
        return None
    return bw.cluster_interval([per_q[q] for q in qs], [capsule[q] for q in qs], reps=reps, level=level)


def nearest_distance(options):
    v = values_of(options)
    return min(abs(x - v[0]) / (abs(x) + abs(v[0])) if abs(x) + abs(v[0]) else 0.0 for x in v[1:])


def leak(sets, capsule, k):
    """The best single-rank rule on held-out capsules, and in-sample, in points over 1/k; the rank shares."""
    ranks = [{"rank": rank_of(o), "cluster": capsule[q]} for q, o in sorted(sets.items())]
    held = cs.grouped_cv_credit(ranks, k, 2000)
    shares = collections.Counter(r["rank"] for r in ranks)
    n = len(ranks)
    return {"held_out": 100.0 * held["credit"], "held_out_ci95": [100.0 * x for x in held["credit_ci95"]],
            "in_sample": 100.0 * (max(shares.values()) / n - 1 / k),
            "rank_shares": [shares.get(j, 0) / n for j in range(k)],
            "extreme": (shares.get(0, 0) + shares.get(k - 1, 0)) / n, "n_items": n,
            "median_nearest_d": float(np.median([nearest_distance(o) for o in sets.values()]))}


def rule_cost(rows, sets_of, capsule):
    """The nearest-option rule on the number the tolerance reads: the share of misses it accepts (all misses,
    and those giving a number) and of answers within 5% of the key, each question's runs averaged. Runs whose
    option set the design could not build are left out of every design's summary alike (``common``)."""
    hit = lambda r: float(bool(graded(str(r["answer"]), r["key_text"], TOL))) if r["answer"] else 0.0
    rule = lambda r: ax.nearest_is_key_last(r["answer"], sets_of(r))
    answered = [r for r in rows if r["answer"] and str(r["answer"]).strip()]
    misses = [r for r in answered if not hit(r)]
    numbered = [r for r in misses if ax.last_number(r["answer"], r["key_text"]) is not None]
    hits = [r for r in answered if hit(r)]
    out = {}
    for name, sub in (("misses", misses), ("misses with a number", numbered), ("correct", hits)):
        num = per_item(sub, rule)
        out[name] = {"accepted": 100.0 * float(np.mean(list(num.values()))) if num else None, "n_runs": len(sub),
                     "per_item": num}
    return out


PAIRS = (("repaired", "placebo"), ("k4-uniform", "k4-released"), ("errors-uniform", "errors-released"),
         ("k4-uniform", "k4-middle"), ("k6-uniform", "k6-middle"), ("k8-uniform", "k8-middle"),
         ("k10-uniform", "k10-middle"), ("errors-uniform", "errors-middle"),
         ("k8-uniform-s0.2", "k8-middle-s0.2"), ("k8-uniform-s0.1", "k8-middle-s0.1"))


def key_class(options):
    r = rank_of(options)
    return "bracketed" if 0 < r < len(options) - 1 else "extreme"


def theory_leak(name, k):
    """What a rule selecting one middle rank gains when the key's rank is uniform over the k-2 middle ones."""
    return 100.0 * (1 / (k - 2) - 1 / k) if "middle" in name else 0.0 if "uniform" in name or name == "repaired" else None


def rule(_args):
    items, runs = load_runs()
    designs = load_designs()
    report = {"seed": SEED, "tolerance": TOL, "not_registered": True, "releases": {}}
    for rel, its in items.items():
        capsule = {q: it["capsule"] for q, it in its.items()}
        graded_runs = [dict(r, key_text=its[r["q"]]["options"][0]) for r in runs
                       if r["release"] == rel and r["condition"] == "data"]
        base = {"released": {q: it["options"] for q, it in its.items()}}
        base.update({arm: {q: s[arm] for q, s in (rp.v10_sets() if rel == "v1.0" else dm.v15_sets()).items()
                           if arm in s} for arm in ("placebo", "repaired")})
        names = ["released", "placebo", "repaired"] + sorted(designs["designs"][rel])

        def sets_for(name, r):
            if name in base:
                return base[name].get(r["q"])
            return options_for(designs, rel, name, r["q"], r["model"])

        files = {n: (base[n] if n in base else author_file(designs, rel, n)) for n in names}
        # designs are compared on the runs every design can grade
        common = [r for r in graded_runs if all(sets_for(n, r) is not None for n in names)]
        out = {"n_runs": len(graded_runs), "n_common_runs": len(common),
               "n_items": len({r["q"] for r in common}), "designs": {}, "pairs": {}}
        for name in names:
            k = len(files[name][next(iter(files[name]))])
            entry = {"k": k, "leak": leak({q: o for q, o in files[name].items() if q in capsule}, capsule, k),
                     "leak_theory": theory_leak(name, k)}
            groups = {"all": common}
            if rel == "v1.0":
                groups.update({m: [r for r in common if r["model"] == m] for m in ("gpt-4o", "Claude 3.5 Sonnet")})
            else:
                for g in sorted({r["group"] for r in common}):
                    groups[g] = [r for r in common if r["group"] == g]
            entry["rule"] = {g: {kind: {"accepted": v["accepted"], "n_runs": v["n_runs"]}
                                 for kind, v in rule_cost(rows, lambda r, n=name: sets_for(n, r), capsule).items()}
                             for g, rows in groups.items()}
            out["designs"][name] = entry
        # each pair: the rank drawn uniformly against a policy that keeps keys bracketed or at their released rank,
        # on the keys the uniform design moves from bracketed to extreme (as U - P on the moved keys), over all
        # items, and in the share of misses accepted; paired on the same runs, intervals at the level the double
        # bootstrap finds covers 95% on these capsules (replication.calibrated)
        hit = lambda r: bool(graded(str(r["answer"]), r["key_text"], TOL)) if r["answer"] else False
        for a, b in PAIRS:
            if a not in files or b not in files:
                continue
            moved = {q for q in files[a] if q in files[b]
                     and key_class(files[b][q]) == "bracketed" and key_class(files[a][q]) == "extreme"}
            both = [r for r in graded_runs if sets_for(a, r) is not None and sets_for(b, r) is not None]
            diff = lambda r: ax.nearest_is_key_last(r["answer"], sets_for(a, r)) - \
                ax.nearest_is_key_last(r["answer"], sets_for(b, r))
            entry = {"n_moved": len(moved)}
            for label, rows in (("moved", [r for r in both if r["q"] in moved]), ("all", both),
                                ("misses", [r for r in both if r["answer"] and str(r["answer"]).strip() and not hit(r)])):
                per_q = per_item(rows, diff)
                rng = np.random.default_rng(zlib.crc32(f"{rel}|{a}|{b}|{label}".encode()))
                entry[label] = rp.calibrated(per_q, capsule, rng, 2000, 5000) if len(per_q) > 1 else None
            out["pairs"][f"{a} - {b}"] = entry
        report["releases"][rel] = out
    OUT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    show(report)


def show(report):
    f = lambda d: "--" if not d else f"{d['mean']:+5.1f} [{d['lo']:+.1f},{d['hi']:+.1f}]"
    for rel, out in report["releases"].items():
        print(f"== {rel}: {out['n_common_runs']} of {out['n_runs']} runs on {out['n_items']} items")
        for name, e in out["designs"].items():
            L, R = e["leak"], e["rule"]["all"]
            t = "" if e["leak_theory"] is None else f"(theory {e['leak_theory']:+.1f})"
            print(f"  {name:22s} k={e['k']:2d} leak held-out {L['held_out']:+6.1f} {t:15s} extreme {100 * L['extreme']:5.1f}% "
                  f"nearest d {L['median_nearest_d']:.3f} | misses accepted {R['misses']['accepted']:5.1f} "
                  f"(with a number {R['misses with a number']['accepted']:5.1f}) correct {R['correct']['accepted']:5.1f}")
        for pair, e in out["pairs"].items():
            print(f"  {pair:36s} moved {e['n_moved']:3d}: {f(e['moved'])} | all {f(e['all'])} | misses {f(e['misses'])}")


# ---------------------------------------------------------------- graded by models

GRADED = ("k4-released", "k4-uniform", "k4-middle", "k8-middle", "k8-uniform", "k8-middle-s0.2",
          "errors-released", "errors-uniform")
SAMPLE_OF, SAMPLE_CHUNKS = 4, {0}           # the published runs a closed grader reads: a random quarter


def chunk_of(r, of):
    return int(hashlib.sha256(r["key"].encode()).hexdigest()[:8], 16) % of


def rows_path(reader):
    return WORK / f"reads_{reader}.jsonl"


async def read_one(r, question, options, shuffle, client, reader):
    """One code-free reading, as ``grading_variants.read_one`` takes it: BixBench's shuffle for the question and
    shuffle, the question with its options and the answer, and the reply constrained to one shown letter."""
    import grading_variants as gv
    name, base = reader
    seed = int(hashlib.sha256(f"{r['q']}|{shuffle}|False".encode()).hexdigest()[:8], 16)
    formatted, correct, _, shown = bw.questions_to_mcq(question, options, False, random.Random(seed))
    prompt = gv.CODEFREE_PROMPT.replace("{{question}}", formatted).replace("{{proposed_answer}}", str(r["answer"]))
    reply = await client.ask(name, base, prompt, 64)
    letter = gv.parse_letter(reply, True)
    picked = shown[ord(letter) - 65] if letter != "Z" and ord(letter) - 65 < len(shown) else None
    rank = None
    if picked is not None:
        v = values_of(options)
        rank = sorted(range(len(v)), key=lambda i: v[i]).index(options.index(picked))
    return {"correct": letter == correct, "no_pick": picked is None, "rank": rank,
            "server_error": reply.startswith("[error")}


async def read_designs(todo, items, designs, reader, wanted, shuffles, concurrency):
    path = rows_path(reader[0])
    done = collections.defaultdict(set)
    if path.exists():
        for line in path.open():
            x = json.loads(line)
            done[x["key"]].update(x["reads"])
    client = bw.Client(WORK / f"cache_{reader[0]}.jsonl", concurrency)
    handle = path.open("a")
    count = 0

    async def one(r):
        nonlocal count
        question = items[r["release"]][r["q"]]["question"]
        jobs = []
        for d in wanted:
            options = options_for(designs, r["release"], d, r["q"], r["model"])
            if options is not None and d not in done[r["key"]]:
                jobs += [(d, options, s) for s in range(shuffles)]
        if not jobs:
            return
        got = await asyncio.gather(*(read_one(r, question, o, s, client, reader) for _, o, s in jobs))
        row = {"key": r["key"], "reads": {}}
        for (d, _, _), x in zip(jobs, got):
            row["reads"].setdefault(d, []).append(x)
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        count += 1
        if count % 500 == 0:
            print(f"  {count} runs read; calls {dict(client.calls)}", flush=True)

    await asyncio.gather(*(one(r) for r in todo))
    await client.http.aclose()
    handle.close()
    print(f"done: {count} runs; calls {dict(client.calls)}", flush=True)


def read(args):
    import grading_variants as gv
    name, _, base = args.reader.partition("=")
    items, runs = load_runs()
    designs = load_designs()
    todo = [r for r in runs if r["condition"] == "data" and r["answer"] and str(r["answer"]).strip()]
    if args.published_quarter:
        todo = [r for r in todo if r["release"] == "v1.5" or chunk_of(r, SAMPLE_OF) in SAMPLE_CHUNKS]
    wanted = args.designs.split(",") if args.designs else list(GRADED)
    WORK.mkdir(parents=True, exist_ok=True)
    by_k = collections.defaultdict(list)
    for d in wanted:
        by_k[len(next(iter(author_file(designs, "v1.5", d).values())))].append(d)
    for k, ds in sorted(by_k.items()):
        # the reply is constrained to one of the k letters shown
        bw.EXTRA_BODY[name] = {"response_format": gv.schema(LETTERS[:k], analysis=False)}
        n = sum(1 for r in todo for d in ds if options_for(designs, r["release"], d, r["q"], r["model"]))
        print(f"k={k}: {ds}: {len(todo)} runs, {n * args.shuffles} reads at most", flush=True)
        if args.dry_run:
            continue
        asyncio.run(read_designs(todo, items, designs, (name, base), ds, args.shuffles, args.concurrency))


def analyse(_args):
    """Each model's code-free grading of the designs: the share of misses and of correct answers it accepts, and
    the pairs' contrasts on the moved keys, beside the rule's on the same runs."""
    items, runs = load_runs()
    designs = load_designs()
    report = json.loads(OUT.read_text())
    report["graders"] = {}
    for reader, lines in read_files().items():
        reads = collections.defaultdict(dict)
        for line in lines:
            x = json.loads(line)
            for d, rs in x["reads"].items():
                reads[x["key"]].setdefault(d, rs)
        out = {}
        for rel, its in items.items():
            capsule = {q: it["capsule"] for q, it in its.items()}
            rows = [dict(r, key_text=its[r["q"]]["options"][0]) for r in runs
                    if r["release"] == rel and r["condition"] == "data" and r["key"] in reads]
            if not rows:
                continue
            hit = lambda r: bool(graded(str(r["answer"]), r["key_text"], TOL))
            share = lambda r, d: (float(np.mean([x["correct"] for x in reads[r["key"]][d]]))
                                  if d in reads[r["key"]] else None)
            present = sorted({d for r in rows for d in reads[r["key"]]})
            # compare designs on the runs every one of them was read through
            common = [r for r in rows if all(d in reads[r["key"]] for d in present)]
            entry = {"n_runs": len(common), "designs": {}, "pairs": {}}
            for d in present:
                m = per_item([r for r in common if not hit(r)], lambda r, d=d: share(r, d))
                c = per_item([r for r in common if hit(r)], lambda r, d=d: share(r, d))
                nopick = per_item(common, lambda r, d=d: float(np.mean([x["no_pick"] for x in reads[r["key"]][d]])))
                entry["designs"][d] = {"misses_accepted": 100.0 * float(np.mean(list(m.values()))) if m else None,
                                       "correct_accepted": 100.0 * float(np.mean(list(c.values()))) if c else None,
                                       "no_pick": 100.0 * float(np.mean(list(nopick.values()))) if nopick else None,
                                       "rule_misses_accepted": 100.0 * float(np.mean(list(per_item(
                                           [r for r in common if not hit(r)],
                                           lambda r, d=d: ax.nearest_is_key_last(
                                               r["answer"], options_for(designs, rel, d, r["q"], r["model"]))
                                       ).values())))}
            files = {d: author_file(designs, rel, d) for d in present}
            for a, b in PAIRS:
                if a not in present or b not in present:
                    continue
                moved = {q for q in files[a] if q in files[b]
                         and key_class(files[b][q]) == "bracketed" and key_class(files[a][q]) == "extreme"}
                pe = {"n_moved": len(moved)}
                for label, sub in (("moved", [r for r in common if r["q"] in moved]), ("all", common),
                                   ("misses", [r for r in common if not hit(r)])):
                    per_q = per_item(sub, lambda r: share(r, a) - share(r, b))
                    rng = np.random.default_rng(zlib.crc32(f"{reader}|{rel}|{a}|{b}|{label}".encode()))
                    pe[label] = rp.calibrated(per_q, capsule, rng, 2000, 5000) if len(per_q) > 1 else None
                entry["pairs"][f"{a} - {b}"] = pe
            out[rel] = entry
        report["graders"][reader] = out
    OUT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    f = lambda d: "--" if not d else f"{d['mean']:+5.1f} [{d['lo']:+.1f},{d['hi']:+.1f}]"
    for reader, out in report["graders"].items():
        for rel, e in out.items():
            print(f"== {reader} {rel}: {e['n_runs']} runs")
            for d, v in e["designs"].items():
                print(f"  {d:18s} misses {v['misses_accepted']:5.1f} (rule {v['rule_misses_accepted']:5.1f}) "
                      f"correct {v['correct_accepted']:5.1f} no pick {v['no_pick']:4.1f}")
            for pair, v in e["pairs"].items():
                print(f"  {pair:34s} moved {v['n_moved']:3d}: {f(v['moved'])} | all {f(v['all'])} | misses {f(v['misses'])}")


# ---------------------------------------------------------------- the tables

TABLE = (("released", "$R$, released", "released"), ("placebo", "$P$", "released"), ("repaired", "$U$", "uniform"),
         ("k4-released", "redrawn", "released"), ("k4-uniform", "redrawn", "uniform"), ("k4-middle", "redrawn", "middle"),
         ("k6-uniform", "redrawn", "uniform"), ("k6-middle", "redrawn", "middle"),
         ("k8-uniform", "redrawn", "uniform"), ("k8-middle", "redrawn", "middle"),
         ("k10-uniform", "redrawn", "uniform"), ("k10-middle", "redrawn", "middle"),
         ("k8-uniform-s0.2", "a fifth apart", "uniform"), ("k8-middle-s0.2", "a fifth apart", "middle"),
         ("k8-middle-s0.1", "a tenth apart", "middle"),
         ("errors-released", "agents' errors", "released"), ("errors-uniform", "agents' errors", "uniform"),
         ("errors-middle", "agents' errors", "middle"))
PAIR_ROWS = (("repaired - placebo", "$U-P$"), ("k4-uniform - k4-released", "$k=4$, uniform $-$ released rank"),
             ("errors-uniform - errors-released", "agents' errors, uniform $-$ released rank"),
             ("k4-uniform - k4-middle", "$k=4$, uniform $-$ middle"), ("k6-uniform - k6-middle", "$k=6$, uniform $-$ middle"),
             ("k8-uniform - k8-middle", "$k=8$, uniform $-$ middle"), ("k10-uniform - k10-middle", "$k=10$, uniform $-$ middle"),
             ("k8-uniform-s0.2 - k8-middle-s0.2", "$k=8$, a fifth apart, uniform $-$ middle"))
READERS = ("gpt-4o", "gemma27b", "qwen72b", "llama70b")


def span(values, fmt="{:.1f}"):
    values = [v for v in values if v is not None]
    if not values:
        return "--"
    lo, hi = min(values), max(values)
    return f"${fmt.format(lo)}$" if fmt.format(lo) == fmt.format(hi) else f"${fmt.format(lo)}$--${fmt.format(hi)}$"


def design_rows(report):
    """tab:design: each design's k and rank policy, the rank rule's gain in theory and on held-out capsules, the
    share of misses the rule and the code-free graders accept, and the share of correct answers the graders accept."""
    rows = []
    rel = report["releases"]
    for name, spacing, rank in TABLE:
        e10, e15 = rel["v1.0"]["designs"][name], rel["v1.5"]["designs"][name]
        theory = "--" if e10["leak_theory"] is None else f"${e10['leak_theory']:.1f}$"
        g = lambda relname, key: [report["graders"][r][relname]["designs"][name][key]
                                  for r in READERS if r in report.get("graders", {})
                                  and name in report["graders"][r].get(relname, {}).get("designs", {})]
        rows.append(f"{spacing} & ${e10['k']}$ & {rank} & {theory} & ${e10['leak']['held_out']:+.1f}$ & "
                    f"${e15['leak']['held_out']:+.1f}$ & ${e10['rule']['all']['misses']['accepted']:.1f}$ & "
                    f"${e15['rule']['all']['misses']['accepted']:.1f}$ & {span(g('v1.0', 'misses_accepted'))} & "
                    f"{span(g('v1.5', 'misses_accepted'))} & {span(g('v1.0', 'correct_accepted') + g('v1.5', 'correct_accepted'))}\\\\")
    return rows


def pair_rows(report):
    """tab:designpairs: a rank drawn uniformly against one kept at the released rank or off the extremes, on the
    keys it moves to an extreme and over all items, under the rule, and the code-free graders' range on the moved
    keys."""
    rows = []
    rel = report["releases"]
    ci = lambda d: "--" if not d else f"${d['mean']:+.1f}$ {{\\scriptsize$[{d['lo']:+.1f},{d['hi']:+.1f}]$}}"
    for pair, label in PAIR_ROWS:
        p10, p15 = rel["v1.0"]["pairs"][pair], rel["v1.5"]["pairs"][pair]
        g = lambda relname: [report["graders"][r][relname]["pairs"][pair]["moved"]["mean"]
                             for r in READERS if r in report.get("graders", {})
                             and pair in report["graders"][r].get(relname, {}).get("pairs", {})
                             and report["graders"][r][relname]["pairs"][pair]["moved"]]
        rows.append(f"{label} & ${p10['n_moved']}$, ${p15['n_moved']}$ & {ci(p10['moved'])} & {ci(p15['moved'])} & "
                    f"${p10['all']['mean']:+.1f}$ & ${p15['all']['mean']:+.1f}$ & {span(g('v1.0'), '{:+.1f}')} & "
                    f"{span(g('v1.5'), '{:+.1f}')}\\\\")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("rule")
    sub.add_parser("show")
    sub.add_parser("analyse")
    t = sub.add_parser("latex")
    t.add_argument("table", choices=("design", "pairs"))
    r = sub.add_parser("read")
    r.add_argument("--reader", required=True, help="NAME=BASE[,BASE...]")
    r.add_argument("--designs", default="")
    r.add_argument("--shuffles", type=int, default=2)
    r.add_argument("--published-quarter", action="store_true", help="of the published runs, a random quarter only")
    r.add_argument("--concurrency", type=int, default=96)
    r.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "show":
        show(json.loads(OUT.read_text()))
        return
    if args.cmd == "latex":
        report = json.loads(OUT.read_text())
        for row in (design_rows(report) if args.table == "design" else pair_rows(report)):
            print(row)
        return
    {"build": build, "rule": rule, "read": read, "analyse": analyse}[args.cmd](args)


if __name__ == "__main__":
    main()
