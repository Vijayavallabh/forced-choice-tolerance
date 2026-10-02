#!/usr/bin/env python3
r"""What BixBench's reading through its released options does to a numeric score, run by run
(not registered).

The paper's counterfactual rewrites ask what options that hide the rank *would* credit. This asks
what the options as released *do* to the scores the benchmark reports, under the benchmark's own
reading. Every with-data run on a numeric question is set beside the tolerance (its submitted
number within 5% of the key, ``free_response.graded``) and falls in one of four cells of a reading:

* a hit named: within 5%, and the reading names the key;
* a hit refused: within 5%, and the reading names a distractor, or declines;
* a miss credited: more than 5% off (or no number), and the reading names the key -- split by
  whether the key is the option nearest the submitted number (the rule's channel, which an edge
  key opens) or not (the reader named it from something else);
* a miss refused.

The reading's score minus the tolerance's is credited misses minus refused hits. Readings:

* **v1.0, published**: gpt-4o's and Claude 3.5 Sonnet's 5,161 published runs on the 159 numeric
  questions, each run set pooled per model; the published forced reading (each model reading its
  own runs), the published may-decline reading (``published_reads.py build-decline``), and
  gemma-3-27b's forced reading (``published_reads.py``); the nearest-option rule beside them.
* **v1.5, seven run sets**: the paper's runs with the data (``bracketing.py``'s rows); each run
  set's own reader forced and may-decline, and gemma-3-27b, through the released options.
* **v1.5, new runs**: the registered new seeds and Qwen3-235B-A22B, read by gemma-3-27b.
* **v1.5, gpt-5.1**: the closed agent's runs through the published protocol (``strong_agent.py``), read by
  gpt-4o forced and may-decline; not part of the confirmatory test.

A score is each question's runs averaged, then the mean over questions, as the paper's other
v1.0 analyses weigh them; differences carry the percentile cluster bootstrap over capsules at the
level the double bootstrap finds covers 95% (``replication.calibrated``), at the 5% tolerance and at
the others it is restated at alike.

Beside the cells: how often a reading's pick is the option nearest the submitted number, and,
where it is not, whether the value it picked is written in the run's notebook -- against the
same check for the options it did not pick, the chance of a value turning up there anyway.

    python3 score_decomposition.py            # results/score_decomposition.json
    python3 score_decomposition.py --latex    # the rows of the released-options table
    python3 score_decomposition.py --latex-picks   # the rows of the table of what the published grades select on a miss
"""
import argparse
import collections
import gzip
import json
import math
import pathlib
import re
import zlib

import numpy as np

import answer_extraction as ax
import bixbench_withdata as bw
import bracketing as br
import published_reads as pr
import replication as rp
import answer_numbers as an
from answer_numbers import graded

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "score_decomposition.json"
NOTEBOOK_VALUES = ROOT / "results" / "score_decomposition_notebook_values.jsonl.gz"
TOL = 0.05
SEED = 20260926
MODELS = {"gpt-4o": ("4o_open_image", "4o_open_no_image"),
          "Claude 3.5 Sonnet": ("claude_open_image", "claude_open_no_image")}
CELLS = ("hit_named", "hit_refused", "miss_credited_nearest", "miss_credited_no_number", "miss_credited_other",
         "miss_refused")
RESTATED = (0.01, 0.02, 0.05, 0.10)
# a submitted number read on another scale than the key's: what a slip of units, a log or a sign gives
RESCALES = (("x100", lambda a: a * 100.0), ("/100", lambda a: a / 100.0), ("x1000", lambda a: a * 1000.0),
            ("/1000", lambda a: a / 1000.0), ("negated", lambda a: -a), ("reciprocal", lambda a: 1.0 / a if a else None),
            ("2^a", lambda a: 2.0 ** a if abs(a) < 60 else None), ("log2", lambda a: math.log2(a) if a > 0 else None),
            ("10^a", lambda a: 10.0 ** a if abs(a) < 30 else None), ("log10", lambda a: math.log10(a) if a > 0 else None),
            ("e^a", lambda a: math.exp(a) if abs(a) < 60 else None), ("ln", lambda a: math.log(a) if a > 0 else None))


# ---------------------------------------------------------------- one run

def nearest_ranks(answer, options):
    """The ranks among the four values of the options nearest the submitted number (ties kept),
    by the paper's rule's distance, or None when there is no number to place."""
    a = bw.parse_number(answer) if answer else None
    values = [bw.parse_number(o) for o in options]
    if a is None or any(v is None for v in values):
        return None
    return {bw.pick_rank(options[i], options) for i in an.nearest_ranks(a, values)}


def nearest_ranks_last(answer, options):
    """``nearest_ranks`` on the number the tolerance reads (``answer_extraction.last_number``)."""
    a = ax.last_number(answer, options[0]) if answer else None
    values = [bw.parse_number(o) for o in options]
    if a is None or any(v is None for v in values):
        return None
    return {bw.pick_rank(options[i], options) for i in an.nearest_ranks(a, values)}


def cells(hit, named, near, nonum):
    """A run's share in each cell: ``named`` is the share of the reading's reads naming the key,
    ``near`` the nearest-option rule's credit for the key (ties split) on the number the tolerance
    reads, and ``nonum`` 1 when the answer holds no number at all, so that nothing is nearest."""
    miss = 1.0 - hit
    return {"hit_named": hit * named, "hit_refused": hit * (1.0 - named),
            "miss_credited_nearest": miss * named * near, "miss_credited_no_number": miss * named * nonum,
            "miss_credited_other": miss * named * (1.0 - near) * (1.0 - nonum),
            "miss_refused": miss * (1.0 - named)}


def rescaled_near(a, value, tol=TOL):
    """Is some rescaling of the submitted number ``a`` (RESCALES) within ``tol`` of ``value``?"""
    if a is None or value is None or value == 0:
        return False
    for _, f in RESCALES:
        try:
            t = f(a)
        except (OverflowError, ValueError, ZeroDivisionError):
            t = None
        if t is not None and math.isfinite(t) and abs(t - value) <= tol * abs(value):
            return True
    return False


# ---------------------------------------------------------------- values in a notebook

NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _match(option):
    """A test for a number in the notebook that writes the option's value: equal at the precision
    the option is written to (its decimals; its significant digits in scientific notation)."""
    text = option.strip().rstrip("%").strip().replace(",", "")
    try:
        value = float(text)
    except ValueError:
        return None
    if "e" in text.lower():
        digits = len(re.sub(r"[^0-9]", "", text.lower().split("e")[0]).lstrip("0")) or 1
        return lambda x: x != 0 and value != 0 and float(f"{x:.{digits - 1}e}") == float(f"{value:.{digits - 1}e}")
    decimals = len(text.split(".")[1]) if "." in text else 0
    return lambda x: round(x, decimals) == round(value, decimals)


def values_written(text, options):
    """For each option, is its value written in ``text``?"""
    found = []
    for tok in NUMBER.findall(text or ""):
        try:
            x = float(tok)
        except ValueError:
            continue
        if math.isfinite(x):
            found.append(x)
    out = []
    for o in options:
        test = _match(o)
        out.append(bool(test) and any(test(x) for x in found))
    return out


def notebook_text(r):
    return r["notebook"] if "notebook" in r else "\n".join(
        f"{c.get('source', '')}\n{c.get('output', '')}" for c in r["cells"])


def build_notebook_values():
    """Which of each run's released options are written in its notebook, for every run the
    decomposition reads, from the notebooks (not shipped) into a file that ships."""
    v10, v15 = rp.v10_sets(), bw.option_sets()
    out = []
    for r in pr.d1_inputs():
        out.append({"key": pr.key_of(r), "written": values_written(r["notebook"], v10[r["q"]]["released"])})
    for r in pr.d0_inputs():
        if r["condition"] == "data":
            out.append({"key": pr.key_of(r), "written": values_written(notebook_text(r), v15[r["q"]]["released"])})
    with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=NOTEBOOK_VALUES.open("wb")) as fh:
        fh.write("".join(json.dumps(x, sort_keys=True) + "\n" for x in out).encode())
    print(f"{len(out)} runs' notebook values to {NOTEBOOK_VALUES.relative_to(ROOT)}")


def load_notebook_values():
    if not NOTEBOOK_VALUES.exists():
        return {}
    with gzip.open(NOTEBOOK_VALUES, "rt") as fh:
        return {x["key"]: x["written"] for x in map(json.loads, fh)}


# ---------------------------------------------------------------- runs, each with its readings

def d1_runs():
    """The published runs: answer, key, the grader's verdict and every reading's picks."""
    doc = rp.load_extract()
    sets = rp.v10_sets()
    grader = {}
    for run in rp.OPEN_RUNS:
        seen = collections.Counter()
        for q, answer, ok in doc["open_runs"][run]:
            if q in sets:
                grader[(run, q, seen[q])] = (answer, ok)
                seen[q] += 1
    declined = decline_reads()
    decline = {}
    with gzip.open(pr.PUBLISHED_DECLINE, "rt") as fh:
        for x in map(json.loads, fh):
            decline[x["key"]] = x
    out = []
    for r in pr.load_rows():
        if r["set"] != "D1":
            continue
        options = sets[r["q"]]["released"]
        answer, ok = grader[(r["run"], r["q"], r["i"])]
        assert answer == r["answer"], (r["run"], r["q"], r["i"])
        rank_of = lambda picked: bw.pick_rank(picked, options) if picked in options else None
        d = decline[pr.key_of(r)]
        reads = {"published forced": [{"correct": r["published"]["correct"], "rank": rank_of(r["published"]["picked"]),
                                       "refused": False}],
                 "published may-decline": [{"correct": d["correct"], "rank": rank_of(d["picked"]),
                                            "refused": d["refused"]}]}
        if r["reads"]:
            reads["gemma forced"] = [dict(x, refused=False) for x in r["reads"]["released"]]
        # read allowed to decline on a random quarter of the runs and on the moved keys: only the quarter,
        # a sample of every key group, is summarised
        if pr.key_of(r) in declined and pr.chunk_of(r, pr.SAMPLE_OF) in pr.SAMPLE_CHUNKS:
            reads["gemma may-decline"] = declined[pr.key_of(r)]
        out.append({"key": pr.key_of(r), "group": next(m for m, runs in MODELS.items() if r["run"] in runs),
                    "run": r["run"], "q": r["q"], "capsule": doc["items"][r["q"]]["capsule"], "answer": r["answer"],
                    "options": options, "grader": bool(ok), "reads": reads})
    return out


def d0_runs():
    """The paper's seven v1.5 run sets with the data, through the released options."""
    sets = bw.option_sets()
    out = []
    for run in br.RUNS:
        own = br.primary(run)
        for r in br.rows_of(run):
            if not r["numeric"] or r["condition"] != "data":
                continue
            reads = {}
            for label, name, mode in (("family forced", own, "forced"), ("family may-decline", own, "refusal"),
                                      ("gemma forced", "gemma27b", "forced"),
                                      ("gemma may-decline", "gemma27b", "refusal")):
                got = r["reads"].get(f"{name}|released|{mode}")
                if got is not None:
                    reads[label] = [{"correct": x["correct"], "rank": x.get("rank"), "refused": x.get("refused", False)}
                                    for x in got]
            out.append({"key": f"D0|{run}|data|{r['question_id']}|0", "group": run, "run": run,
                        "q": r["question_id"], "capsule": r["capsule"], "answer": r["answer"],
                        "options": sets[r["question_id"]]["released"], "grader": bool(r["open"]), "reads": reads})
    return out


def decline_reads(reader="gemma27b"):
    """{run key: released reads} of a reader's may-decline pass (``published_reads.py --mode decline``)."""
    path = pr.rows_path(reader, "decline")
    if not path.exists():
        return {}
    return {pr.key_of(r): [dict(x) for x in r["reads"]["released"]] for r in pr.load_rows(path)
            if r["reads"] and "released" in r["reads"]}


def d2_runs():
    """The new v1.5 runs with the data, read by gemma-3-27b through the released options."""
    declined = decline_reads()
    sets = bw.option_sets()
    capsule = {json.loads(l)["question_id"]: json.loads(l)["capsule_uuid"] for l in open(ROOT / "data" / "bixbench.jsonl")}
    out = []
    for r in pr.load_rows():
        if r["set"] != "D2" or r["condition"] != "data":
            continue
        group = "Qwen3-235B-A22B" if r["run"].startswith("qwen3-235b|") else "new seeds"
        reads = {"gemma forced": [dict(x, refused=False) for x in r["reads"]["released"]]} if r["reads"] else {}
        if pr.key_of(r) in declined:
            reads["gemma may-decline"] = declined[pr.key_of(r)]
        out.append({"key": pr.key_of(r), "group": group, "run": r["run"], "q": r["q"], "capsule": capsule[r["q"]],
                    "answer": r["answer"], "options": sets[r["q"]]["released"], "grader": None, "reads": reads})
    return out


def d3_runs():
    """gpt-5.1's runs with the data (``strong_agent.py``), read by gpt-4o through the released options."""
    import strong_agent as sa
    if not (sa.ROWS.exists() or sa.BUILD_ROWS.exists()):
        raise SystemExit(f"{sa.ROWS.relative_to(ROOT)} is missing, and Table 2's gpt-5.1 rows are read from it")
    sets = bw.option_sets()
    out = []
    for r in sa.load_rows():
        reads = {}
        for label, mode in (("gpt-4o forced", "forced"), ("gpt-4o may-decline", "refusal")):
            got = r["reads"].get(f"{sa.READER}|released|{mode}")
            if got is not None:
                reads[label] = [{"correct": x["correct"], "rank": x.get("rank"), "refused": x.get("refused", False)}
                                for x in got]
        out.append({"key": f"D3|{sa.MODEL}|data|{r['question_id']}|{r['rollout']}", "group": "gpt-5.1",
                    "run": sa.MODEL, "q": r["question_id"], "capsule": r["capsule"], "answer": r["answer"],
                    "options": sets[r["question_id"]]["released"], "grader": bool(r["open"]), "reads": reads})
    return out


# ---------------------------------------------------------------- summaries

def named_share(run, reading):
    """The share of the reading's reads that name the key; 0 for a run with no answer, which
    upstream scores wrong unread; None if the reading did not read this run."""
    if not (run["answer"] and str(run["answer"]).strip()):
        return 0.0
    reads = run["reads"].get(reading)
    return None if reads is None else float(np.mean([x["correct"] for x in reads]))


def per_item(runs, value):
    """{item: mean over its runs of value(run)}, runs where value is None left out."""
    acc = collections.defaultdict(list)
    for r in runs:
        v = value(r)
        if v is not None:
            acc[r["q"]].append(v)
    return {q: float(np.mean(v)) for q, v in acc.items()}


def mean_over_items(d):
    return 100.0 * float(np.mean(list(d.values()))) if d else None


def single_nearest(r):
    """'key' or 'other' when the number the tolerance reads has one nearest option, the key or another;
    None when the answer gives no number or one equidistant from several options (in practice zero, or of
    the other sign than every option, at distance 1 from each)."""
    ranks = nearest_ranks_last(r["answer"], r["options"]) if ax.category(r["answer"]) != "no number" else None
    if ranks is None or len(ranks) != 1:
        return None
    return "key" if ranks == {bw.pick_rank(r["options"][0], r["options"])} else "other"


def miss_rates(read, reading, hit, capsule, reps=10000, level=0.95):
    """Of the misses a grading grades (an empty answer is scored wrong without being graded), the share it
    accepts: over all of them, and by what the answer gives it, a number whose single nearest option is the
    key, a number whose single nearest option is another, or neither (no number, or a number equidistant from
    several options); and, within the last, the answers with no number. Each question's runs are averaged
    and the questions summed, so that every question weighs as in the table's other columns and the misses
    accepted, in points, are the share of all misses accepted times the misses' share of the runs. Beside
    it, what a grader choosing at random among the options it is shown would accept: a quarter of the
    misses forced, a fifth with the refusal option. The share of all misses carries a plain cluster-bootstrap
    interval over capsules, from its own generator."""
    miss = lambda r: (1.0 - hit(r)) * float(ax.category(r["answer"]) != "empty")
    kinds = {"all": lambda r: 1.0, "nearest": lambda r: float(single_nearest(r) == "key"),
             "other": lambda r: float(single_nearest(r) == "other"),
             "neither": lambda r: float(single_nearest(r) is None),
             "no number": lambda r: float(ax.category(r["answer"]) == "no number")}
    n_items = len({r["q"] for r in read})
    chance = 20.0 if "decline" in reading else 25.0
    out = {"chance": chance}
    for kind, weight in kinds.items():
        num = per_item(read, lambda r, w=weight: miss(r) * w(r) * named_share(r, reading))
        den = per_item(read, lambda r, w=weight: miss(r) * w(r))
        total = sum(den.values())
        out[kind] = {"accepted": 100.0 * sum(num.values()) / total if total else None,
                     "share_of_runs": 100.0 * total / n_items}
        if kind == "all":
            sums = collections.defaultdict(lambda: [0.0, 0.0])
            for q in den:
                sums[capsule[q]][0] += num[q]
                sums[capsule[q]][1] += den[q]
            keys = sorted(sums)
            a = np.array([sums[k][0] for k in keys])
            b = np.array([sums[k][1] for k in keys])
            idx = np.random.default_rng(SEED).integers(0, len(keys), size=(reps, len(keys)))
            with np.errstate(invalid="ignore", divide="ignore"):
                draws = 100.0 * a[idx].sum(1) / b[idx].sum(1)   # a draw of capsules with no miss has no share
            lo, hi = np.nanquantile(draws, [(1 - level) / 2, (1 + level) / 2])
            out[kind].update(lo=float(lo), hi=float(hi), level=level, n_clusters=len(keys))
    out["at_chance"] = chance * out["all"]["share_of_runs"] / 100.0
    return out


PICKS = ("nearest", "key", "third", "none")


def picks_on_misses(read, reading, hit):
    """What the grading selects on the misses whose number (the last in the answer, as the tolerance reads
    it) has a single nearest option, by whether that option is the key: the nearest option, the key when it
    is not the nearest, a third option, or none (no letter the parser can read, scored wrong). Each
    question's runs weigh as one question, as in ``miss_rates``, whose shares of these two kinds of miss
    accepted are the nearest column of the key's row and the key column of the other row; ``n_runs``
    counts the runs, and ``nearest_of_selecting`` is the nearest option's share of the grades that select
    an option."""
    per_q = collections.Counter(r["q"] for r in read)
    out = {row: {**{c: 0.0 for c in PICKS}, "n_runs": 0} for row in ("key", "other")}
    for r in read:
        reads = r["reads"].get(reading)
        if hit(r) or not reads or ax.category(r["answer"]) == "empty" or single_nearest(r) is None:
            continue
        (nearest,) = nearest_ranks_last(r["answer"], r["options"])
        key = bw.pick_rank(r["options"][0], r["options"])
        row = out[single_nearest(r)]
        row["n_runs"] += 1
        for x in reads:
            col = ("none" if x["rank"] is None else "nearest" if x["rank"] == nearest
                   else "key" if x["rank"] == key else "third")
            row[col] += 1.0 / per_q[r["q"]] / len(reads)
    for row in out.values():
        total = sum(row[c] for c in PICKS)
        selecting = total - row["none"]
        row["nearest_of_selecting"] = 100.0 * row["nearest"] / selecting if selecting else None
        for col in PICKS:
            row[col] = 100.0 * row[col] / total if total else None
        row["weight"] = total
    return out


def picks_rows(report):
    """tab:picks: what the published forced grades select on a miss with a single nearest option."""
    rows = []
    for model, name in (("gpt-4o", "gpt-4o"), ("Claude 3.5 Sonnet", "Claude 3.5 Sonnet")):
        p = report["v1.0"][model]["published forced"]["picks_on_misses"]
        for kind, label in (("key", "the key"), ("other", "another")):
            s = p[kind]
            key = "--" if kind == "key" else f"${s['key']:.1f}$"
            rows.append(f"{name if kind == 'key' else ''} & {label} & ${s['n_runs']:,}$ & ${s['nearest']:.1f}$ & {key} & "
                        f"${s['third']:.1f}$ & ${s['none']:.1f}$ & ${s['nearest_of_selecting']:.1f}$\\\\"
                        .replace(",", "{,}"))
    return rows


def summarise(runs, reading, capsule, rng, written, label=""):
    """Score, tolerance, the cells and the score minus the tolerance, for one reading. The number an
    answer gives is the one the tolerance reads (``answer_extraction.last_number``), for the
    nearest option as for the tolerance. ``label`` names the block and group, and seeds the generators
    of the intervals at the tolerances other than 5%."""
    hit_at = lambda r, t: float(bool(graded(r["answer"], r["options"][0], t))) if r["answer"] else 0.0
    hit = lambda r: hit_at(r, TOL)
    near = lambda r: ax.nearest_is_key_last(r["answer"], r["options"])
    nonum = lambda r: float(ax.category(r["answer"]) == "no number")
    read = [r for r in runs if named_share(r, reading) is not None]
    out = {"n_runs": len(read), "n_items": len({r["q"] for r in read})}
    out["tolerance"] = mean_over_items(per_item(read, hit))
    out["rule"] = mean_over_items(per_item(read, near))
    out["score"] = mean_over_items(per_item(read, lambda r: named_share(r, reading)))
    if all(r["grader"] is not None for r in read):
        out["graders"] = mean_over_items(per_item(read, lambda r: float(r["grader"])))
    for c in CELLS:
        out[c] = mean_over_items(per_item(read, lambda r, c=c: cells(hit(r), named_share(r, reading), near(r),
                                                                       nonum(r))[c]))
    diff = per_item(read, lambda r: named_share(r, reading) - hit(r))
    out["score_minus_tolerance"] = rp.calibrated(diff, capsule, rng, 2000, 5000)
    # the same at other tolerances, each interval at the level the double bootstrap finds covers 95% on these
    # capsules, as the one at 5% is: at 5% that interval itself, at every other tolerance one drawn from its own
    # generator, seeded by the reading's name and the tolerance, so that ``rng``'s stream -- and every interval
    # above -- is what it was without them. The plain 95% cluster intervals are kept beside them.
    out["restated"], out["restated_plain"] = {}, {}
    for t in RESTATED:
        d = per_item(read, lambda r, t=t: named_share(r, reading) - hit_at(r, t))
        qs = sorted(d)
        tolerance = mean_over_items(per_item(read, lambda r, t=t: hit_at(r, t)))
        out["restated_plain"][f"{t:g}"] = dict(bw.cluster_interval([d[q] for q in qs], [capsule[q] for q in qs],
                                                                   level=0.95), tolerance=tolerance)
        if t == TOL:
            cal = out["score_minus_tolerance"]
        else:
            seed = zlib.crc32(f"{label}|{reading}|{t:g}".encode())
            cal = rp.calibrated(d, capsule, np.random.default_rng(seed), 2000, 5000)
        out["restated"][f"{t:g}"] = dict(cal or {"mean": 100.0 * float(np.mean(list(d.values())))},
                                         tolerance=tolerance)
    # how often the reading names the key on a miss, by what the answer gives it to read: the key nearest
    # the number, no number, or a number nearest another option; pooled over runs
    kinds = {"nearest": near, "no number": nonum,
             "other": lambda r: (1.0 - near(r)) * (1.0 - nonum(r)) * float(ax.category(r["answer"]) != "empty")}
    out["named_on_a_miss"] = {}
    for kind, weight in kinds.items():
        pairs = [(weight(r), named_share(r, reading)) for r in read if not hit(r)]
        den = sum(w for w, _ in pairs)
        out["named_on_a_miss"][kind] = {"named": 100.0 * sum(w * v for w, v in pairs) / den if den else None,
                                        "n_runs": den}
    # the misses credited some other way: was the key's value written in the run's notebook, or is a
    # rescaled number within 5% of it -- against the same for the options the reading passed over
    other = [r for r in read if not hit(r) and ax.category(r["answer"]) not in ("empty", "no number")
             and near(r) == 0.0]
    checks = {"written": lambda r, i: (written.get(r["key"]) or [None] * 4)[i],
              "rescaled": lambda r, i: rescaled_near(ax.last_number(r["answer"], r["options"][0]),
                                                     bw.parse_number(r["options"][i]))}
    out["other_misses"] = {"n_runs": len(other)}
    for name, test in checks.items():
        rows = [r for r in other if test(r, 0) is not None]
        key = [(float(test(r, 0)), named_share(r, reading)) for r in rows]
        base = [float(test(r, i)) for r in rows for i in (1, 2, 3)]
        out["other_misses"][name] = {
            "n_runs": len(rows),
            "key": 100.0 * float(np.mean([k for k, _ in key])) if key else None,
            "distractor": 100.0 * float(np.mean(base)) if base else None,
            "named_if_key": (100.0 * sum(k * v for k, v in key) / sum(k for k, _ in key)
                             if key and sum(k for k, _ in key) else None),
            "named_if_not": (100.0 * sum((1 - k) * v for k, v in key) / sum(1 - k for k, _ in key)
                             if key and sum(1 - k for k, _ in key) else None),
            "credited_share": (100.0 * sum(k * v for k, v in key) / sum(v for _, v in key)
                               if key and sum(v for _, v in key) else None)}
    out["declined"] = mean_over_items(per_item(read, lambda r: float(np.mean([x["refused"] for x in r["reads"][reading]]))
                                               if r["reads"].get(reading) else None))
    # the reading's pick against the option nearest the submitted number
    agree, other_written, other_base = [], [], []
    for r in read:
        ranks = nearest_ranks(r["answer"], r["options"])
        if ranks is None or not r["reads"].get(reading):
            continue
        for x in r["reads"][reading]:
            if x["rank"] is None:
                continue
            agree.append(float(x["rank"] in ranks))
            w = written.get(r["key"])
            if w is None or x["rank"] in ranks:
                continue
            by_rank = {bw.pick_rank(o, r["options"]): flag for o, flag in zip(r["options"], w)}
            other_written.append(float(by_rank[x["rank"]]))
            rest = [by_rank[k] for k in range(4) if k != x["rank"] and k not in ranks]
            other_base.extend(float(f) for f in rest)
    out["pick_is_nearest"] = 100.0 * float(np.mean(agree)) if agree else None
    out["n_picks_placed"] = len(agree)
    out["not_nearest_pick_written"] = 100.0 * float(np.mean(other_written)) if other_written else None
    out["not_nearest_other_written"] = 100.0 * float(np.mean(other_base)) if other_base else None
    out["n_not_nearest_picks"] = len(other_written)
    # the share of misses accepted against a choice at random, and what is selected on a miss; neither
    # draws from ``rng``, so the intervals above are those they were without them
    out["miss_rates"] = miss_rates(read, reading, hit, capsule)
    out["picks_on_misses"] = picks_on_misses(read, reading, hit)
    return out


def analyse(args):
    rng = np.random.default_rng(SEED)
    written = load_notebook_values()
    report = {"seed": SEED, "tolerance": TOL, "not_registered": True}
    d1 = d1_runs()
    cap1 = {r["q"]: r["capsule"] for r in d1}
    report["v1.0"] = {}
    for model in MODELS:
        runs = [r for r in d1 if r["group"] == model]
        report["v1.0"][model] = {reading: summarise(runs, reading, cap1, rng, written, f"v1.0|{model}")
                                 for reading in ("published forced", "published may-decline", "gemma forced",
                                                 "gemma may-decline")
                                 if any(reading in r["reads"] for r in runs)}
    d0 = d0_runs()
    cap0 = {r["q"]: r["capsule"] for r in d0}
    report["v1.5"] = {}
    for group in br.RUNS + ["pooled"]:
        runs = d0 if group == "pooled" else [r for r in d0 if r["group"] == group]
        report["v1.5"][group] = {reading: summarise(runs, reading, cap0, rng, written, f"v1.5|{group}")
                                 for reading in ("family forced", "family may-decline", "gemma forced",
                                                 "gemma may-decline")}
    d2 = d2_runs()
    cap2 = {r["q"]: r["capsule"] for r in d2}
    report["v1.5 new"] = {g: {reading: summarise([r for r in d2 if r["group"] == g], reading, cap2, rng, written,
                                                       f"v1.5 new|{g}")
                              for reading in ("gemma forced", "gemma may-decline")
                              if any(reading in r["reads"] for r in d2)}
                          for g in ("new seeds", "Qwen3-235B-A22B")}
    # last, so that the rng draws of the blocks above are those they had without it
    d3 = d3_runs()
    cap3 = {r["q"]: r["capsule"] for r in d3}
    report["v1.5 closed"] = {"gpt-5.1": {reading: summarise(d3, reading, cap3, rng, written, "v1.5 closed|gpt-5.1")
                                         for reading in ("gpt-4o forced", "gpt-4o may-decline")}} if d3 else {}
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    f = lambda v: "--" if v is None else f"{v:5.1f}"
    for block, groups in (("v1.0", report["v1.0"]), ("v1.5", report["v1.5"]), ("v1.5 new", report["v1.5 new"]),
                          ("v1.5 closed", report["v1.5 closed"])):
        for g, readings in groups.items():
            for reading, s in readings.items():
                d = s["score_minus_tolerance"]
                print(f"{block:8s} {g:18s} {reading:22s} n={s['n_runs']:5d} tol {f(s['tolerance'])} "
                      f"score {f(s['score'])} rule {f(s['rule'])} | hit+ {f(s['hit_named'])} hit- {f(s['hit_refused'])} "
                      f"miss+near {f(s['miss_credited_nearest'])} miss+nonum {f(s['miss_credited_no_number'])} "
                      f"miss+other {f(s['miss_credited_other'])} | "
                      f"score-tol {d['mean']:+.1f} [{d['lo']:+.1f},{d['hi']:+.1f}] | pick=nearest "
                      f"{f(s['pick_is_nearest'])} | written {f(s['not_nearest_pick_written'])} vs "
                      f"{f(s['not_nearest_other_written'])} | declined {f(s['declined'])}")
    print(f"wrote {OUT.relative_to(ROOT)}")


# tab:released: one row per reading; v1.0 by the published readings, v1.5 by each run set's own family
# (the seven run sets) or by gemma-3-27b (the new runs)
# Table 2's rows: (block, group, reading, runs, grading). The current agents' blocks are frontier_agents.py's, which
# summarises their runs as this report does (``summarise``).
LATEX = (("v1.0", "gpt-4o", "published forced", "v1.0, gpt-4o", "forced"),
         ("v1.0", "gpt-4o", "published may-decline", "", "with refusal"),
         ("v1.0", "Claude 3.5 Sonnet", "published forced", "v1.0, Claude 3.5", "forced"),
         ("v1.0", "Claude 3.5 Sonnet", "published may-decline", "", "with refusal"),
         ("v1.5", "pooled", "family forced", "v1.5, seven configurations", "forced"),
         ("v1.5", "pooled", "family may-decline", "", "with refusal"),
         ("v1.5", "pooled", "gemma forced", "", "gemma-3-27b, forced"),
         ("v1.5 new", "new seeds", "gemma forced", "v1.5, new seeds", "forced"),
         ("v1.5 new", "Qwen3-235B-A22B", "gemma forced", "v1.5, Qwen3-235B-A22B", "forced"),
         ("v1.5 closed", "gpt-5.1", "gpt-4o forced", "v1.5, gpt-5.1", "forced"),
         ("v1.5 closed", "gpt-5.1", "gpt-4o may-decline", "", "with refusal"),
         ("current", "gpt-6-luna", "gpt-4o forced", "v1.5, gpt-6-luna", "forced"),
         ("current", "DeepSeek-V4-Pro", "gpt-4o forced", "v1.5, DeepSeek-V4-Pro", "forced"))


FORMULA = ROOT / "results" / "formula_scoring.json"
FRONTIER = ROOT / "results" / "frontier_agents.json"


def contribution(s, kind):
    """What one kind of miss adds to the score, in points: the share of the runs that are graded misses of that
    kind times the share of them accepted (``miss_rates``); the parts and the correct answers rejected sum to the
    score minus the tolerance."""
    m = s["miss_rates"][kind]
    return 0.0 if m["accepted"] is None else m["accepted"] * m["share_of_runs"] / 100.0


def table_rows(report, formula=None, frontier=None):
    """Table 2, one row per grading: the tolerance; the score minus the tolerance with its interval, split into what
    the misses whose single nearest option is the key, another option or neither add, less the correct answers the
    grading rejects; for a forced grading, the score corrected for guessing (``formula_scoring.py``) minus the
    tolerance; and the share of graded misses accepted, over all of them and by their single nearest option."""
    if formula is None and FORMULA.exists():
        formula = json.loads(FORMULA.read_text())
    if frontier is None and FRONTIER.exists():
        frontier = json.loads(FRONTIER.read_text())
    corrected = {(r["block"], r["group"], r["reading"]): r["corrected_minus_tolerance"]
                 for r in (formula or {}).get("rows", [])}
    blocks = dict(report)
    blocks["current"] = {label: a["table1"] for label, a in (frontier or {}).get("agents", {}).items()}
    for label, a in (frontier or {}).get("agents", {}).items():
        corrected.setdefault(("current", label, "gpt-4o forced"), a["corrected"]["corrected_minus_tolerance"])
    ci = lambda v: f"${v['mean']:+.1f}$ {{\\scriptsize$[{v['lo']:+.1f},{v['hi']:+.1f}]$}}"
    rows = []
    for block, group, reading, label, how in LATEX:
        s = blocks.get(block, {}).get(group, {}).get(reading)
        if s is None:
            continue
        m = s["miss_rates"]
        tol = f"${s['tolerance']:.1f}$" if label else ""
        c = corrected.get((block, group, reading))
        if "decline" in reading:
            corr = "--"
        elif c is None:           # a grading the correction's table does not split: the point value alone
            corr = f"${(s['score'] - 25.0) / 0.75 - s['tolerance']:+.1f}$"
        else:
            corr = ci(c)
        rate = lambda k: "--" if m[k]["accepted"] is None else f"${m[k]['accepted']:.1f}$"
        parts = " & ".join(f"${contribution(s, k):+.1f}$" for k in ("nearest", "other", "neither"))
        rows.append(f"{label} & {how} & {tol} & {ci(s['score_minus_tolerance'])} & {parts} & "
                    f"${-s['hit_refused']:+.1f}$ & {corr} & {rate('all')} & {rate('nearest')} & {rate('other')}\\\\")
    return rows


def main():
    import sys
    if "--latex" in sys.argv:
        for row in table_rows(json.loads(OUT.read_text())):
            print(row)
        return
    if "--latex-picks" in sys.argv:
        for row in picks_rows(json.loads(OUT.read_text())):
            print(row)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook-values", action="store_true",
                    help="rebuild the shipped record of which option values each notebook writes (needs the notebooks)")
    args = ap.parse_args()
    if args.notebook_values:
        build_notebook_values()
        return
    analyse(args)


if __name__ == "__main__":
    main()
