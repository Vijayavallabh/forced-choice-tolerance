#!/usr/bin/env python3
r"""Why forced-choice grading exceeds the open-ended grades on questions whose options are not all numbers
(not registered).

On BixBench's published v1.0 runs, forced-choice grading exceeds the open-ended grades of the same runs on the
questions whose four options are not all numbers by more than a random choice among four on every miss would
add (``grading_variants_analysis.inflation``, Table~\ref{tab:allq}'s "other" column). The nearest-option rule
accounts for the numeric questions; this asks what accounts for the rest. Four views of the same runs:

* **What the options are.** Each question is classed by its four option texts, by deterministic rules
  (``classify``): numeric intervals or bounds written as text; numbers written with words (a unit, a label, a
  ratio); a direction or a change; a list of names; a name (a gene, pathway, cell type, sample); a longer
  statement. Per class: questions, runs, the open-ended accuracy, the forced score, their difference, what a
  random choice among four on every miss would add, and the share of the misses the forced grade accepts.
* **The option nearest the answer.** For each miss --- an answer BixBench's open-ended grader judged wrong ---
  the option nearest it (``nearest_kind``): in value on the questions whose options each state one number or one
  interval (``number_nearest``, the last number in the answer as the tolerance reads it), and in wording
  otherwise, the option whose distinguishing words (those not in all four options) the answer contains in the
  largest share (``wording_nearest``); the key, another option, or neither (a tie, no number, no shared word).
  This is Table~\ref{tab:released}'s key, other and neither for text. Beside it, the answers that contain the
  key's text or a distractor's verbatim.
* **What the grader knows without the data.** BixBench released the same two models' forced choices on the same
  questions without the data (``data/external/zero_shot_v10``). The misses are split by whether that model chose
  the key there, and, on a miss nearest no option, whether the published grade selects the option the model chose
  without the data.
* **A cue in the options alone.** Whether the key is the longest option, a test-wiseness cue, with ties split.

Every split reports its share of the graded misses, the share of it accepted, and the points it adds to the score
beyond what a random choice among four on it would; each question's runs are averaged within a model and the
questions summed, as in Table~\ref{tab:released}, so the parts add up. An empty answer is scored wrong without
being graded and is not among the misses graded. Gradings: BixBench's published forced grades, which read the
notebook (the option each selected is recovered from the published grading rows by ``build``, which needs
``eval_df.csv``; the result ships), and the code-free graders of ``grading_variants.py``, which see only the
question, the options and the answer. On v1.5 (unless ``--no-v15``), the seven configurations graded by their
own models through the released options, on the $100$ questions whose options are not all numbers, most of them
numeric intervals. Intervals are plain 95% percentile cluster bootstraps over capsules.

    python3 nonnumeric_excess.py build        # results/nonnumeric_published_picks.jsonl.gz (needs eval_df.csv)
    python3 nonnumeric_excess.py analyse      # results/nonnumeric_excess.json
"""
import argparse
import ast
import collections
import csv
import gzip
import hashlib
import json
import math
import pathlib
import re
import sys
import zlib

import numpy as np

import bixbench_withdata as bw
import bracketing as br
import grading_variants as gv
import grading_variants_analysis as gva
import replication as rp
from answer_numbers import NUMBER, as_number, normalise

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "nonnumeric_excess.json"
PICKS = ROOT / "results" / "nonnumeric_published_picks.jsonl.gz"
EVAL_DF = ROOT / "build" / "external" / "bixbench_v10_trajectories" / "eval_df.csv"
SEED = 20260927
MODEL = gva.MODEL
CLASSES = ("yes or no", "interval or bound", "number with words", "list of names", "direction or change", "name",
           "statement")
CODE_FREE = (("gpt-4o", "gpt-4o"), ("gemma27b", "gemma-3-27b"), ("qwen72b", "Qwen2.5-72B"),
             ("llama70b", "Llama-3.3-70B"))


# ---------------------------------------------------------------- the published grades' selections

def build(args):
    """The option each published forced grade selected, for every published open-answer run, matched to its run
    as ``grading_variants.build_published`` matches the grade (a grading row shares the run's notebook and
    answer). The grade's letter is read from the question as it was shown (``replication.picked_option``)."""
    csv.field_size_limit(sys.maxsize)
    mcq_of = {m: o for o, m in rp.MCQ_RUNS.items()}
    runs, pool = collections.defaultdict(list), collections.defaultdict(list)
    with open(args.eval_df, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["run_name"]
            if name not in rp.OPEN_RUNS and name not in mcq_of:
                continue
            body = hashlib.md5((row["md_notebook"] + "\x00" + row["agent_answer"]).encode()).hexdigest()
            if name in rp.OPEN_RUNS:
                runs[name].append((row["uuid"], body))
            else:
                pool[(mcq_of[name], row["uuid"], body)].append(
                    (row["correct"] == "True", rp.picked_option(row["formatted_question"], row["llm_answer"])))
    with gzip.open(gv.PUBLISHED_ALL, "rt") as fh:
        forced = {(r["run"], r["q"], r["i"]): r["forced"] for r in map(json.loads, fh)}
    out = []
    for name in rp.OPEN_RUNS:
        seen = collections.Counter()
        for q, body in runs[name]:
            ok, picked = pool[(name, q, body)].pop(0)
            assert ok == forced[(name, q, seen[q])], (name, q, seen[q])
            out.append({"run": name, "q": q, "i": seen[q], "picked": picked})
            seen[q] += 1
    assert not any(pool.values())
    with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=PICKS.open("wb")) as fh:
        fh.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in out).encode())
    print(f"{len(out)} published forced selections to {PICKS.relative_to(ROOT)}")


def load_picks():
    with gzip.open(PICKS, "rt") as fh:
        return {(r["run"], r["q"], r["i"]): r["picked"] for r in map(json.loads, fh)}


# ---------------------------------------------------------------- what the options are

STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "and", "or", "is", "are", "was", "were",
        "be", "as", "that", "this", "these", "those", "from", "it", "its", "which", "into", "per", "vs", "versus"}
TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
# a name that carries digits (CD4, TP53, P184, SRR5566591, 1620056at2759) is not a quantity
IDENTIFIER = re.compile(r"\b[A-Za-z]+\d+[A-Za-z\d]*\b|\b\d+at\d+\b")
NUM = r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*(?:[eE]|x\s*10\^)\s*[-+]?\d+)?\s*[kK%]?"
INTERVAL = re.compile(rf"^\s*[\(\[]\s*{NUM}\s*,\s*{NUM}\s*[\)\]]\s*$"            # (0.25, 0.35)
                      rf"|^\s*[\(\[]?\s*{NUM}\s*(?:-|–|to)\s*{NUM}\s*[\)\]]?\s*[A-Za-z% ]*$"   # 0.25-0.35, 40-60 variants
                      rf"|^\s*[A-Za-z- ]*[<>≤≥]=?\s*{NUM}"                       # <1000, ARI > 0.80
                      rf"|\b(?:between|more than|less than|greater than|fewer than|at least|at most|close to|"
                      rf"above|below|under|over)\s+{NUM}", re.IGNORECASE)
DIRECTION = ("increas", "decreas", "higher", "lower", "upregulat", "downregulat", "up-regulat", "down-regulat",
             "more", "less", "larger", "smaller", "longer", "shorter", "faster", "slower", "positive", "negative",
             "skew", "reduc", "elevat", "enrich", "deplet", "gain", "loss", "synergi", "antagon", "stronger",
             "weaker", "unchanged", "equal", "no change", "no significant", "not significant", "double", "halv")
YES_NO = {"yes", "no", "true", "false", "maybe", "cannot be determined"}
# a verb that makes an option a claim rather than a name
VERB = re.compile(r"\b(?:is|are|was|were|has|have|show|shows|remain|remains|occur|occurs|evolve|evolves|affect|"
                  r"affects|alter|alters|appear|appears|become|becomes|exist|exists|depend|depends|indicat\w*|"
                  r"suggest\w*|pass|passes|accumulat\w*|preferentially|cannot)\b", re.IGNORECASE)


def quantity_numbers(text):
    """The numbers a text states as quantities, names that carry digits removed."""
    return NUMBER.findall(IDENTIFIER.sub(" ", normalise(str(text))))


def is_interval(text):
    return bool(INTERVAL.search(str(text)))


def has_direction(text):
    t = str(text).lower()
    return any(re.search(r"\b" + re.escape(stem), t) for stem in DIRECTION)


def words(text):
    return len(re.findall(r"\S+", re.sub(r"\([^)]*\)", " ", str(text))))


def classify(options):
    """What a question's four options are, by the first rule that holds."""
    texts = [str(o).strip() for o in options]
    if all(t.lower().rstrip(".") in YES_NO for t in texts):
        return "yes or no"
    if sum(is_interval(t) for t in texts) >= 3:
        return "interval or bound"
    if all(quantity_numbers(t) for t in texts):
        return "number with words"
    if sum(bool(re.search(r",\s", t)) and all(len(p.split()) <= 5 for p in t.split(",")) for t in texts) >= 2:
        return "list of names"
    if sum(has_direction(t) for t in texts) >= 2:
        return "direction or change"
    if all(words(t) <= 6 and not VERB.search(t) for t in texts):
        return "name"
    return "statement"


# ---------------------------------------------------------------- the option nearest the answer

def tokens(text):
    """Lower-case word and number tokens, commas in numbers removed and numbers written one way."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", str(text or "").lower())
    out = set()
    for t in TOKEN.findall(text):
        if t in STOP:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            t = f"{float(t):g}"
        out.add(t)
    return out


def wording_nearest(answer, options):
    """'key', 'other', 'tied' or 'none': the option whose distinguishing tokens (those not in all four options)
    the answer contains in the largest share, when one option has the largest share and it is above zero."""
    sets = [tokens(o) for o in options]
    common = set.intersection(*sets)
    have = tokens(answer)
    shares = []
    for s in sets:
        own = s - common
        shares.append(len(own & have) / len(own) if own else 0.0)
    best = max(shares)
    if best == 0.0:
        return "none"
    top = [i for i, v in enumerate(shares) if v == best]
    if len(top) > 1:
        return "tied"
    return "key" if top == [0] else "other"


def _value(text):
    got = as_number(text.replace(" ", ""))
    return None if got is None else got[0]


LOW_BOUND = re.compile(r"<|≤|\bless than\b|\bfewer than\b|\bbelow\b|\bunder\b|\bat most\b", re.IGNORECASE)
HIGH_BOUND = re.compile(r">|≥|\bmore than\b|\bgreater than\b|\babove\b|\bover\b|\bat least\b", re.IGNORECASE)
PAIR = re.compile(rf"({NUM})\s*(?:,|–|to|and|-)\s*({NUM})")


def as_quantities(text):
    """A text with names that carry digits removed, powers of ten and thousands written as numbers."""
    text = IDENTIFIER.sub(" ", str(text or ""))
    text = re.sub(r"(\d(?:\.\d+)?)\s*[x×]\s*10\^\s*([-+]?\d+)", r"\1e\2", text)
    return re.sub(r"(\d)\s*[kK]\b", r"\g<1>000", text)


def option_span(option):
    """(low, high) for an option stating one interval or bound, (v, v) for one stating one number; else None."""
    text = as_quantities(option)
    if is_interval(option):
        pair = PAIR.search(text)
        if pair:
            a, b = _value(pair.group(1)), _value(pair.group(2))
            if a is not None and b is not None:
                return (min(a, b), max(a, b))
    found = [v for v in (_value(x) for x in NUMBER.findall(normalise(text))) if v is not None]
    if len(found) != 1:
        return None
    v = found[0]
    if LOW_BOUND.search(text):
        return (-math.inf, v)
    if HIGH_BOUND.search(text):
        return (v, math.inf)
    return (v, v)


def _distance(a, span):
    lo, hi = span
    if lo <= a <= hi:
        return 0.0
    edge = lo if a < lo else hi
    return abs(a - edge) / (abs(a) + abs(edge)) if abs(a) + abs(edge) else 0.0


def number_nearest(answer, options):
    """'key', 'other' or 'neither' (no number, or a tie): the option nearest the answer's last number, for a
    question whose four options each state one number or one interval; None for any other question."""
    spans = [option_span(o) for o in options]
    if any(s is None for s in spans):
        return None
    found = NUMBER.findall(normalise(as_quantities(answer)))
    got = as_number(found[-1]) if found else None
    if got is None:
        return "neither"
    d = [_distance(got[0], s) for s in spans]
    top = [i for i, v in enumerate(d) if v == min(d)]
    if len(top) > 1:
        return "neither"
    return "key" if top == [0] else "other"


def normalised(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip().rstrip(".")


def contains(answer, option):
    o = normalised(option)
    return bool(o) and o in normalised(answer)


# ---------------------------------------------------------------- the runs

NODATA = {"gpt-4o": "gpt-4o", "Claude 3.5 Sonnet": "claude-3-5-sonnet-latest"}


def nodata_forced():
    """{(model, question): (correct, the option chosen)} for BixBench's released v1.0 runs of the same two models
    without the data, forced to choose among the four options with the question (``published_repair.v10``'s
    forced arm); the chosen letter is mapped to its option through the choices as that run showed them."""
    out = {}
    for model, name in NODATA.items():
        path = ROOT / "data" / "external" / "zero_shot_v10" / f"bixbench_llm_baseline_refusal_False_mcq_{name}_1.0.csv"
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                q = f"CapsuleFolder-{r['uuid']}_{r['short_qid'].split('_')[-1]}"
                shown = {c[1]: re.sub(r"^\(([A-Z])\)\s*", "", c) for c in ast.literal_eval(r["choices"])}
                out[(model, q)] = (r["correct"] == "True", shown.get(r["predicted"].strip()[:1]))
    return out


def published_runs():
    """BixBench's published open-answer runs on v1.0's questions whose options are not all numbers: the answer,
    the open-ended grade, the published forced grade and the option it selected."""
    doc = rp.load_extract()
    numeric = set(rp.v10_sets())
    with gzip.open(gv.PUBLISHED_ALL, "rt") as fh:
        published = {(r["run"], r["q"], r["i"]): r for r in map(json.loads, fh)}
    picks = load_picks()
    nodata = nodata_forced()
    out = []
    for run in rp.OPEN_RUNS:
        seen = collections.Counter()
        for q, answer, ok in doc["open_runs"][run]:
            i = seen[q]
            seen[q] += 1
            if q in numeric:
                continue
            p = published[(run, q, i)]
            assert p["open"] == bool(ok)
            e = doc["items"][q]
            out.append({"model": MODEL[run], "run": run, "q": q, "i": i, "capsule": e["capsule"],
                        "answer": answer if p["answered"] else "", "open": bool(ok), "options": e["options"],
                        "nodata_correct": nodata[(MODEL[run], q)][0], "nodata_choice": nodata[(MODEL[run], q)][1],
                        "grades": {"published": {"accepted": float(p["forced"]) if p["answered"] else 0.0,
                                                 "picked": picks[(run, q, i)] if p["answered"] else None}}})
    for reader, label in CODE_FREE:
        rows = {(r["run"], r["q"], r["i"]): r for r in gva.load(reader, "codefree", "forced") if r["set"] == "D1"}
        for r in out:
            got = rows.get((r["run"], r["q"], r["i"]))
            if got is None:
                continue
            s = gva.score(got, "released")
            if s is not None:
                r["grades"][f"code-free {label}"] = {"accepted": s}
    return out


def v15_runs():
    """The seven v1.5 configurations with the data, on the questions whose options are not all numbers, graded by
    their own models through the released options, forced."""
    items = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    numeric = set(bw.option_sets())
    out = []
    for run in br.RUNS:
        name = br.primary(run)
        for r in br.rows_of(run):
            q = r["question_id"]
            if r["condition"] != "data" or r["rollout"] != 0 or q in numeric:
                continue
            answered = bool(r["answer"] and str(r["answer"]).strip())
            reads = r["reads"].get(f"{name}|released|forced") if r["reads"] else None
            if answered and not reads:
                continue
            it = items[q]
            out.append({"model": run, "run": run, "q": q, "i": 0, "capsule": it["capsule_uuid"],
                        "answer": r["answer"] if answered else "", "open": bool(r["open"]),
                        "options": [it["ideal"], *it["distractors"]],
                        "grades": {"own model": {"accepted": float(np.mean([x["correct"] for x in reads]))
                                                 if answered else 0.0, "picked": None}}})
    return out


# ---------------------------------------------------------------- summaries

def per_item(runs, value):
    acc = collections.defaultdict(list)
    for r in runs:
        v = value(r)
        if v is not None:
            acc[r["q"]].append(v)
    return {q: float(np.mean(v)) for q, v in acc.items()}


def ratio(runs, num, den, capsule, label, reps=10000, level=0.95):
    """sum over questions of the mean of num over sum of the mean of den, each question's runs averaged, with a
    plain cluster-bootstrap interval over capsules (its own seed, from the label)."""
    n, d = per_item(runs, num), per_item(runs, den)
    total = sum(d.values())
    if not total:
        return {"share": None, "weight": 0.0}
    sums = collections.defaultdict(lambda: [0.0, 0.0])
    for q in d:
        sums[capsule[q]][0] += n.get(q, 0.0)
        sums[capsule[q]][1] += d[q]
    keys = sorted(sums)
    a, b = np.array([sums[k][0] for k in keys]), np.array([sums[k][1] for k in keys])
    idx = np.random.default_rng(zlib.crc32(label.encode())).integers(0, len(keys), size=(reps, len(keys)))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = 100.0 * a[idx].sum(1) / b[idx].sum(1)
    lo, hi = np.nanquantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return {"share": 100.0 * sum(n.values()) / total, "lo": float(lo), "hi": float(hi),
            "weight": total, "n_questions": sum(1 for v in d.values() if v > 0)}


def interval(per_q, capsule):
    qs = sorted(per_q)
    return bw.cluster_interval([per_q[q] for q in qs], [capsule[q] for q in qs], level=0.95)


def excess_block(runs, grading, capsule, label):
    """Open-ended accuracy, forced score, their difference with its interval, what a random choice among four on
    every miss would add, and the share of the graded misses accepted."""
    graded = [r for r in runs if grading in r["grades"]]
    if not graded:
        return None
    acc = lambda r: r["grades"][grading]["accepted"]
    opened = per_item(graded, lambda r: float(r["open"]))
    scored = per_item(graded, acc)
    diff = per_item(graded, lambda r: acc(r) - float(r["open"]))
    miss = lambda r: float(not r["open"] and bool(str(r["answer"]).strip()))
    return {"questions": len(opened), "runs": len(graded),
            "open": 100.0 * float(np.mean(list(opened.values()))),
            "score": 100.0 * float(np.mean(list(scored.values()))),
            "excess": interval(diff, capsule),
            "at_chance": 100.0 * float(np.mean([(1.0 - v) / 4.0 for v in opened.values()])),
            "misses_accepted": ratio(graded, lambda r: miss(r) * acc(r), miss, capsule, f"{label}|misses")}


def nearest_kind(r):
    """'key', 'other' or 'neither': the option nearest the answer in value where every option states one number or
    one interval, in wording otherwise; a tie, no number or no shared wording is neither."""
    n = number_nearest(r["answer"], r["options"])
    if n is not None:
        return n
    w = wording_nearest(r["answer"], r["options"])
    return w if w in ("key", "other") else "neither"


def longest_is_key(options):
    """1 if the key is the longest option, a share of 1 when it ties with others, else 0."""
    size = [len(str(o).strip()) for o in options]
    tied = [i for i, v in enumerate(size) if v == max(size)]
    return 1.0 / len(tied) if 0 in tied else 0.0


def nearest_block(runs, grading, capsule, label):
    """The graded misses split by the option nearest the answer (``nearest_kind``; and in wording alone, and in
    value on the questions whose options each state one number or interval), by whether the answer contains the
    key's or a distractor's text, and by whether the same model chose the key without the data: per part, its
    share of the graded misses, the share of it accepted, and the points it adds to the score beyond what a random
    choice among four on it would. Every ratio weighs a question's runs as Table 2 does -- each question's runs
    averaged, the questions summed -- so that the parts' points add up to those of all the misses."""
    graded = [r for r in runs if grading in r["grades"]]
    if not graded:
        return None
    acc = lambda r: r["grades"][grading]["accepted"]
    miss = lambda r: float(not r["open"] and bool(str(r["answer"]).strip()))
    kinds = {id(r): (nearest_kind(r), wording_nearest(r["answer"], r["options"]),
                     number_nearest(r["answer"], r["options"])) for r in graded if miss(r)}
    applicable = {r["q"] for r in graded if all(option_span(o) is not None for o in r["options"])}
    n_q = len({r["q"] for r in graded})

    def part(name, test, subset=None):
        subset = graded if subset is None else subset
        inside = {id(r) for r in subset}
        m = lambda r: miss(r) * test(r)
        points = per_item(graded, lambda r: m(r) * (acc(r) - 0.25) if id(r) in inside else 0.0)
        return {"share_of_misses": ratio(subset, m, miss, capsule, f"{label}|{name}|share")["share"],
                "accepted": ratio(subset, lambda r: m(r) * acc(r), m, capsule, f"{label}|{name}"),
                "points_above_chance": 100.0 * sum(points.values()) / n_q}

    kind = lambda r, j: kinds[id(r)][j] if id(r) in kinds else None
    out = {"n_misses": int(sum(miss(r) for r in graded)),
           "misses_accepted": ratio(graded, lambda r: miss(r) * acc(r), miss, capsule, f"{label}|misses"),
           "all": part("all", lambda r: 1.0)}
    out["nearest"] = {k: part(f"nearest|{k}", lambda r, k=k: float(kind(r, 0) == k)) for k in ("key", "other", "neither")}
    out["wording"] = {k: part(f"wording|{k}", lambda r, k=k: float(kind(r, 1) == k))
                      for k in ("key", "other", "tied", "none")}
    in_value = [r for r in graded if r["q"] in applicable]
    out["value"] = {k: part(f"value|{k}", lambda r, k=k: float(kind(r, 2) == k), in_value)
                    for k in ("key", "other", "neither")}
    out["value_questions"] = len({r["q"] for r in in_value})
    has_key = lambda r: float(contains(r["answer"], r["options"][0]))
    has_other = lambda r: float(any(contains(r["answer"], o) for o in r["options"][1:]))
    out["contains the key"] = part("contains the key", has_key)
    out["contains a distractor"] = part("contains a distractor", has_other)
    neither = lambda r: float(kind(r, 0) == "neither")
    out["key longest"] = {"questions": 100.0 * float(np.mean([longest_is_key(o) for o in
                                                                {r["q"]: r["options"] for r in graded}.values()])),
                          "neither, key longest": part("neither|longest", lambda r: neither(r) * longest_is_key(r["options"])),
                          "neither, key not longest": part("neither|not longest",
                                                           lambda r: neither(r) * (1 - longest_is_key(r["options"])))}
    if all("nodata_correct" in r for r in graded):
        # does a grade accept a miss more often where the same model, without the data, chose the key?
        nd = lambda r: float(r["nodata_correct"])
        out["no-data answer"] = {
            "questions answered without the data": 100.0 * float(np.mean(
                [float(v) for v in {r["q"]: r["nodata_correct"] for r in graded}.values()])),
            "all misses, answered without the data": part("nodata|all|yes", nd),
            "all misses, not": part("nodata|all|no", lambda r: 1 - nd(r)),
            "neither, answered without the data": part("nodata|neither|yes", lambda r: neither(r) * nd(r)),
            "neither, not": part("nodata|neither|no", lambda r: neither(r) * (1 - nd(r)))}
    if grading == "published":
        # on a miss nearest no option, does the grade select the option the same model chose without the data?
        own = lambda r: float(r["grades"][grading]["picked"] == r["nodata_choice"])
        wrong = lambda r: neither(r) * (1 - float(r["nodata_correct"]))
        out["neither, selects its own no-data choice"] = {
            "all": ratio(graded, lambda r: neither(r) * own(r), neither, capsule, f"{label}|own|all"),
            "where that choice is a distractor": ratio(graded, lambda r: wrong(r) * own(r), wrong, capsule,
                                                       f"{label}|own|distractor")}
        none = lambda r: float(r["grades"][grading]["picked"] is None)
        out["selects none"] = ratio(graded, lambda r: miss(r) * none(r), miss, capsule, f"{label}|none")["share"]
        out["accepted, of those selecting"] = ratio(graded, lambda r: miss(r) * (1 - none(r)) * acc(r),
                                                    lambda r: miss(r) * (1 - none(r)), capsule, f"{label}|selecting")
    return out


def summarise(runs, gradings, prefix):
    capsule = {r["q"]: r["capsule"] for r in runs}
    classes = {q: classify(r["options"]) for r in runs for q in [r["q"]]}
    out = {"classes": collections.Counter(classes.values()), "by_class": {}, "all": {}, "nearest": {}}
    for model in sorted({r["model"] for r in runs}):
        sub = [r for r in runs if r["model"] == model]
        for grading in gradings:
            key = f"{model}|{grading}"
            block = excess_block(sub, grading, capsule, f"{prefix}|{key}|all")
            if block is None:
                continue
            out["all"][key] = block
            out["by_class"][key] = {c: excess_block([r for r in sub if classes[r["q"]] == c], grading, capsule,
                                                     f"{prefix}|{key}|{c}")
                                    for c in CLASSES if any(classes[r["q"]] == c for r in sub)}
            out["nearest"][key] = nearest_block(sub, grading, capsule, f"{prefix}|{key}")
    return out, classes


def pooled_models(runs, gradings, prefix):
    """The v1.5 configurations pooled, each question's runs averaged over configurations."""
    pooled = [dict(r, model="seven configurations") for r in runs]
    return summarise(pooled, gradings, prefix)


def _f(v, width=5):
    return "--" if v is None else f"{v:{width}.1f}"


def show(report, classes, options_of):
    print("classes:", dict(report["classes"]))
    for c in CLASSES:
        qs = sorted(q for q, k in classes.items() if k == c)
        for q in qs[:4]:
            print(f"  [{c}] {options_of[q]}"[:200])
    for key, block in report["all"].items():
        e = block["excess"]
        m = block["misses_accepted"]
        print(f"{key:45s} all: n={block['questions']:3d} open {_f(block['open'])} forced {_f(block['score'])} "
              f"excess {e['mean']:+.1f} [{e['lo']:+.1f},{e['hi']:+.1f}] chance {block['at_chance']:+.1f} "
              f"misses accepted {_f(m['share'])} [{_f(m.get('lo'))},{_f(m.get('hi'))}]")
        for c, b in report["by_class"][key].items():
            if b is None:
                continue
            e, m = b["excess"], b["misses_accepted"]
            print(f"   {c:22s} q={b['questions']:3d} runs={b['runs']:5d} open {_f(b['open'])} forced {_f(b['score'])} "
                  f"excess {e['mean']:+6.1f} chance {b['at_chance']:+5.1f} misses accepted {_f(m['share'])}")
        n = report["nearest"][key]
        pt = lambda v: (f"{_f(v['share_of_misses'])}% of misses, accepted {_f(v['accepted']['share'])} "
                        f"[{_f(v['accepted'].get('lo'))},{_f(v['accepted'].get('hi'))}], {v['points_above_chance']:+.1f} pts")
        print(f"   all graded misses (n={n['n_misses']}): {pt(n['all'])}")
        for name in ("nearest", "wording", "value"):
            for k, v in n[name].items():
                print(f"   {name:8s} {k:8s} {pt(v)}")
        print(f"   value questions: {n['value_questions']}")
        for name in ("contains the key", "contains a distractor"):
            print(f"   {name:22s} {pt(n[name])}")
        kl = n["key longest"]
        print(f"   key longest on {_f(kl['questions'])}% of questions; neither-kind misses: key longest "
              f"{pt(kl['neither, key longest'])}; not {pt(kl['neither, key not longest'])}")
        if "no-data answer" in n:
            b = n["no-data answer"]
            print(f"   model without the data chose the key on {_f(b['questions answered without the data'])}% of questions")
            for name in ("all misses, answered without the data", "all misses, not", "neither, answered without the data",
                         "neither, not"):
                print(f"     {name:40s} {pt(b[name])}")
        if "neither, selects its own no-data choice" in n:
            o = n["neither, selects its own no-data choice"]
            print(f"   neither-kind misses: the grade selects the model's own no-data choice on {_f(o['all']['share'])} "
                  f"[{_f(o['all'].get('lo'))},{_f(o['all'].get('hi'))}]%; where that choice is a distractor, on "
                  f"{_f(o['where that choice is a distractor']['share'])} [{_f(o['where that choice is a distractor'].get('lo'))},"
                  f"{_f(o['where that choice is a distractor'].get('hi'))}]%")
        if "selects none" in n:
            print(f"   selects none {_f(n['selects none'])}%, accepted of those selecting "
                  f"{_f(n['accepted, of those selecting']['share'])}")


def analyse(args):
    report = {"seed": SEED, "not_registered": True, "interval": "plain 95% percentile cluster bootstrap over capsules"}
    runs = published_runs()
    gradings = ["published"] + [f"code-free {label}" for _, label in CODE_FREE]
    v10, classes10 = summarise(runs, gradings, "v1.0")
    report["v1.0"] = v10
    report["v1.0"]["question_classes"] = classes10
    options10 = {r["q"]: r["options"] for r in runs}
    show(v10, classes10, options10)
    if args.v15:
        runs15 = v15_runs()
        v15, classes15 = pooled_models(runs15, ["own model"], "v1.5")
        report["v1.5"] = v15
        report["v1.5"]["question_classes"] = classes15
        show(v15, classes15, {r["q"]: r["options"] for r in runs15})
    OUT.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--eval-df", default=str(EVAL_DF))
    a = sub.add_parser("analyse")
    a.add_argument("--no-v15", dest="v15", action="store_false")
    args = ap.parse_args()
    {"build": build, "analyse": analyse}[args.cmd](args)


if __name__ == "__main__":
    main()
