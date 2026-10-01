#!/usr/bin/env python3
r"""The same runs graded through the options in three more ways.

* **Code-free** (``--config codefree``). A 2026 system that compares its multiple-choice scores with
  BixBench's published ones grades them differently: "the second-stage model received only the
  agent's open-response answer and the answer options (with or without an explicit 'insufficient
  information' option, depending on the regime) and returned the selected option in a constrained
  output format" \citep{bioagents}. This is BixBench's MCQ prompt without the notebook -- the question
  with its options shuffled as BixBench shuffles them, and the submitted answer -- with the reply
  constrained to one option's letter, forced or with the refusal option. Its selection depends on the
  answer and the options alone, the class of grader Proposition 1 is about. Every question of both
  releases, through the released options; on the numeric questions also through $P$, $U$ and the
  digit-matched $P'$ and $U'$ (``digit_matched.py``).
* **Letter-constrained, with the notebook** (``--config letter``). BixBench's MCQ prompt exactly as
  ``published_reads.py`` sends it, the notebook included, with the reply constrained to an analysis and
  one option's letter, so that a grader told to pick an option cannot select none. gpt-4o 2024-11-20,
  unconstrained, selects none on more than half of the misses it grades (``grader_declines.py``).
* **With the notebook, through the digit-matched rewrites** (``--config notebook``): the reads of
  ``published_reads.py``, unchanged, through $P'$ and $U'$.

Rows record, per run and option set, each shuffle's selection: whether it is the key, whether it is no
option, the refusal option, and the selected option's rank among the four values.

    python3 grading_variants.py read --reader gpt-4o=BASE --config codefree --mode forced --runs d1
    python3 grading_variants.py read --reader gemma27b=http://127.0.0.1:8103 --config notebook --runs v15 --moved-only
    python3 grading_variants.py merge --reader-name gpt-4o --config codefree --mode forced
    python3 grading_variants.py analyse          # results/grading_variants.json
"""
import argparse
import asyncio
import collections
import gzip
import hashlib
import json
import os
import pathlib
import random
import re

import numpy as np

import bixbench_withdata as bw
import digit_matched as dm
import published_reads as pr
import replication as rp

ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT / "build" / "grading_variants"
CONFIGS = ("codefree", "letter", "notebook")
MODES = ("forced", "decline", "withintol")
# the option that turns a reading through options into a tolerance: chosen when no option is within 5%
NONE_WITHIN = "None of the other options is within 5% of the proposed answer"
RANK_ARMS = ("placebo", "repaired", "placebo_digits", "repaired_digits")
SEED = 20260927

CODEFREE_PROMPT = """
Consider the following multiple-choice question:

<question>
{{question}}
</question>

This was an open response answer submitted to the question:

<proposed_answer>
{{proposed_answer}}
</proposed_answer>

Map the proposed answer to the most appropriate answer option, and give the letter of that option.
"""

WITHINTOL_PROMPT = """
Consider the following multiple-choice question:

<question>
{{question}}
</question>

This was an open response answer submitted to the question:

<proposed_answer>
{{proposed_answer}}
</proposed_answer>

Map the proposed answer to the answer option whose value is within 5% of it. If no option's value is within
5% of the proposed answer, choose the option that says so. Give the letter of the option you choose.
"""


def with_none_within(question, options, rng):
    """BixBench's ``questions_to_mcq`` with the option that no value is within 5% in place of its refusal."""
    options = list(options) + [NONE_WITHIN]
    correct_answer = options[0]
    rng.shuffle(options)
    formatted = f"{question}\n" + "".join(f"{chr(65 + j)}. {opt}\n" for j, opt in enumerate(options))
    return formatted, chr(65 + options.index(correct_answer)), chr(65 + options.index(NONE_WITHIN)), options


def schema(letters, analysis):
    """The constrained reply: one of the shown options' letters, after an analysis if ``analysis``."""
    props = {"answer": {"type": "string", "enum": list(letters)}}
    if analysis:
        props = {"analysis": {"type": "string"}, **props}
    return {"type": "json_schema",
            "json_schema": {"name": "mcq_choice", "strict": True,
                            "schema": {"type": "object", "properties": props, "required": list(props),
                                       "additionalProperties": False}}}


# ---------------------------------------------------------------- option sets and runs

def option_sets():
    """Every question's released options, key first; the numeric ones also through P, U, P' and U'."""
    doc = rp.load_extract()
    v10 = {q: {"released": e["options"]} for q, e in doc["items"].items()}
    for q, s in rp.v10_sets().items():
        v10[q].update({a: s[a] for a in ("placebo", "repaired") if a in s})
    for q, s in dm.v10_sets(dm.V10_PD, dm.V10_UD).items():
        v10[q].update({f"{a}_digits": s[a] for a in ("placebo", "repaired") if a in s})
    items = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    v15 = {q: {"released": [it["ideal"], *it["distractors"]]} for q, it in items.items()}
    for q, s in dm.v15_sets().items():
        v15[q].update({a: s[a] for a in ("placebo", "repaired") if a in s})
    for q, s in dm.v15_sets(dm.V15_PD, dm.V15_UD).items():
        v15[q].update({f"{a}_digits": s[a] for a in ("placebo", "repaired") if a in s})
    return v10, v15


def numeric(sets):
    return "placebo" in sets or "repaired" in sets or rp.numeric_ranks(sets["released"]) is not None


def moved(sets):
    """The key is moved to an extreme by U or by U'."""
    for arm in ("repaired", "repaired_digits"):
        if arm in sets and rp.group_of({"released": sets["released"], "repaired": sets[arm]}) == "moved to an edge":
            return True
    return False


def d1_runs(with_notebook):
    """BixBench's published open-answer runs with the data, every question: the answer, the published
    open-ended grade, and (numeric questions, when asked) the notebook as published."""
    doc = rp.load_extract()
    notebooks = {pr.key_of(r): r["notebook"] for r in pr.d1_inputs()} if with_notebook else {}
    out = []
    for name in rp.OPEN_RUNS:
        seen = collections.Counter()
        for q, answer, correct in doc["open_runs"][name]:
            r = {"set": "D1", "run": name, "q": q, "i": seen[q], "condition": "data", "answer": answer,
                 "open": bool(correct), "question": doc["items"][q]["question"]}
            seen[q] += 1
            if with_notebook:
                if pr.key_of(r) not in notebooks:
                    continue                 # a non-numeric question: the notebooks shipped are the numeric ones'
                r["notebook"] = notebooks[pr.key_of(r)]
            out.append(r)
    return out


def v15_runs(which, conditions=("data",)):
    """v1.5 runs with the data, every question: the seven configurations' first runs (``d0``), the
    pre-specified test's new runs (``d2``) or gpt-5.1's (``gpt51``), with their trajectories' cells and,
    where BixBench's v1.5 graders graded the run open-ended, that grade."""
    import bracketing as br
    items = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    if which == "d0":
        trajectories = [t for t in bw.load_trajectories(ROOT / "build" / "agent_runs")
                        if bw.run_key(t) in br.RUNS and t["rollout"] == 0]
        run_of = lambda t: bw.run_key(t)
        graded = {(run, r["condition"], r["question_id"], r["rollout"]): r["open"]
                  for run in br.RUNS for r in br.rows_of(run)}
        grade_of = lambda t: graded.get((bw.run_key(t), t["condition"], t["question_id"], t["rollout"]))
    elif which == "d2":
        trajectories = rp.d2_trajectories(packed=None)
        run_of = lambda t: f"{bw.run_key(t)}|r{t['rollout']}"
        grade_of = lambda t: None
    else:
        # load_trajectories also returns the shipped open-weight runs: keep gpt-5.1's own
        trajectories = [t for t in bw.load_trajectories(ROOT / "build" / "openai" / "agent_runs")
                        if bw.run_key(t).startswith("gpt-5.1")]
        run_of = lambda t: f"{bw.run_key(t)}|r{t['rollout']}"
        path = ROOT / "build" / "openai" / "withdata_rows_gpt51.json"
        rows = json.loads(path.read_text()) if path.exists() else []
        graded = {(r["question_id"], r["rollout"]): r["open"] for r in rows if r["condition"] == "data"}
        grade_of = lambda t: graded.get((t["question_id"], t["rollout"]))
    out = []
    for t in trajectories:
        if t["condition"] not in conditions:
            continue
        out.append({"set": "V15", "run": run_of(t), "q": t["question_id"], "i": t["rollout"], "condition": t["condition"],
                    "answer": t.get("answer"), "open": grade_of(t), "question": items[t["question_id"]]["question"],
                    "cells": t["cells"]})
    return out


# ---------------------------------------------------------------- reading

# A constrained reply is {"answer": "<letter>"}. Replies cut off at the token cap (a complete answer field
# followed by whitespace padding, the closing brace never written) are read from the field itself.
ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"\s*([A-Za-z])\s*"')


def parse_letter(reply, constrained):
    if constrained:
        try:
            letter = str(json.loads(reply)["answer"]).strip().upper()
            if len(letter) == 1 and "A" <= letter <= "Z":
                return letter
        except (ValueError, KeyError, TypeError):
            found = ANSWER_FIELD.search(reply or "")
            if found:
                return found.group(1).upper()
    return bw.xml_extract(reply)


async def read_one(r, options, shuffle, client, reader, config, refusal):
    """One read: BixBench's shuffle for this question, shuffle and refusal setting; the prompt of the
    configuration; the letter mapped back to an option."""
    name, base = reader
    if refusal == "withintol":
        seed = int(hashlib.sha256(f"{r['q']}|{shuffle}|withintol".encode()).hexdigest()[:8], 16)
        formatted, correct, refusal_letter, shown = with_none_within(r["question"], options, random.Random(seed))
        prompt = (WITHINTOL_PROMPT.replace("{{question}}", formatted)
                  .replace("{{proposed_answer}}", str(r["answer"])))
        reply = await client.ask(name, base, prompt, 64)
    else:
        seed = int(hashlib.sha256(f"{r['q']}|{shuffle}|{refusal}".encode()).hexdigest()[:8], 16)
        formatted, correct, refusal_letter, shown = bw.questions_to_mcq(r["question"], options, refusal,
                                                                        random.Random(seed))
    if refusal == "withintol":
        pass
    elif config == "codefree":
        prompt = (CODEFREE_PROMPT.replace("{{question}}", formatted)
                  .replace("{{proposed_answer}}", str(r["answer"])))
        reply = await client.ask(name, base, prompt, 64)
    else:
        budget = pr.BUDGET
        for _ in range(3):
            notebook = (bw.notebook_markdown(r["cells"], budget) if "cells" in r
                        else r["notebook"] if len(r["notebook"]) <= budget else r["notebook"][-budget:])
            prompt = (bw.PROMPTS["MCQ_EVAL_PROMPT"].replace("{{notebook}}", notebook)
                      .replace("{{question}}", formatted).replace("{{proposed_answer}}", str(r["answer"])))
            reply = await client.ask(name, base, prompt, bw.READ_TOKENS.get(name, 1536))
            if not reply.startswith("[error 400"):
                break
            budget //= 2
    letter = parse_letter(reply, config in ("codefree", "letter"))
    picked = shown[ord(letter) - 65] if letter != "Z" and ord(letter) - 65 < len(shown) else None
    out = {"correct": letter == correct, "no_pick": picked is None,
           "rank": bw.pick_rank(picked, options) if rp.numeric_ranks(options) is not None else None,
           "server_error": reply.startswith("[error")}
    if refusal:
        out["refused"] = letter == refusal_letter
    return out


def arms_for(config, mode, sets):
    if config == "notebook":
        return [a for a in ("placebo_digits", "repaired_digits") if a in sets]
    if mode == "withintol":
        return [a for a in ("released",) + RANK_ARMS if a in sets] if numeric(sets) else []
    if config == "letter":
        return [a for a in ("released",) + RANK_ARMS if a in sets]
    if mode == "decline":
        return ["released"]
    return [a for a in ("released",) + RANK_ARMS if a in sets]


async def read_all(todo, sets_of, reader, config, mode, cache, rows_path, concurrency):
    client = bw.Client(cache, concurrency)
    done = {pr.key_of(json.loads(l)) for path in sorted(rows_path.parent.glob("rows_*.jsonl")) for l in path.open()}
    todo = [r for r in todo if pr.key_of(r) not in done]
    print(f"{len(todo)} runs to read ({len(done)} already read)", flush=True)
    refusal = "withintol" if mode == "withintol" else mode == "decline"
    native = bw.oa.is_openai(reader[1].split(",")[0])
    runs_in_flight = asyncio.Semaphore(max(1, concurrency // 4)) if native and config != "codefree" else None
    handle = rows_path.open("a")
    count = 0

    async def one(r):
        nonlocal count
        bases = reader[1].split(",")
        base = bases[int(hashlib.sha256(pr.key_of(r).encode()).hexdigest()[:8], 16) % len(bases)]
        row = {k: r[k] for k in ("set", "run", "q", "i", "condition", "answer", "open") if k in r}
        row["reads"] = {}
        if r["answer"] and str(r["answer"]).strip():
            jobs = [(arm, s) for arm in arms_for(config, mode, sets_of(r)) for s in range(bw.SHUFFLES)]
            go = lambda arm, s: read_one(r, sets_of(r)[arm], s, client, (reader[0], base), config, refusal)
            if runs_in_flight is not None and jobs:
                # on OpenAI's API a run's first read fills the prompt cache with its notebook, then the rest
                async with runs_in_flight:
                    first = await go(*jobs[0])
                    rest = await asyncio.gather(*(go(a, s) for a, s in jobs[1:]))
                reads = [first, *rest]
            else:
                reads = await asyncio.gather(*(go(a, s) for a, s in jobs))
            for (arm, _), x in zip(jobs, reads):
                row["reads"].setdefault(arm, []).append(x)
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        count += 1
        if count % 200 == 0:
            print(f"  {count}/{len(todo)} read; calls {dict(client.calls)}", flush=True)

    await asyncio.gather(*(one(r) for r in todo))
    await client.http.aclose()
    handle.close()
    print(f"done; calls {dict(client.calls)}", flush=True)


def work_dir(reader, config, mode):
    return WORK / f"{reader}_{config}_{mode}"


def read(args):
    name, _, base = args.reader.partition("=")
    if args.config != "codefree" and args.mode != "forced":
        raise SystemExit("with the notebook, only the forced reading is taken here")
    if args.mode == "withintol":
        args.numeric_only = True
    v10, v15 = option_sets()
    sets_of = lambda r: v10[r["q"]] if r["set"] == "D1" else v15[r["q"]]
    if args.config in ("codefree", "letter"):
        letters = "ABCDE"[:4 if args.mode == "forced" else 5]
        bw.EXTRA_BODY[name] = {"response_format": schema(letters, analysis=args.config == "letter")}
    with_notebook = args.config != "codefree"
    if args.runs == "d1":
        todo = d1_runs(with_notebook)
    else:
        todo = v15_runs(args.runs, ("data", "nodata") if args.condition == "both" else (args.condition,))
    chunks = pr.parse_chunks(args.chunks)
    todo = [r for r in todo if pr.chunk_of(r, args.of) in chunks or (args.moved_too and moved(sets_of(r)))]
    if with_notebook or args.numeric_only:
        todo = [r for r in todo if numeric(sets_of(r))]
    if args.moved_only:
        todo = [r for r in todo if moved(sets_of(r))]
    tag = f"{args.runs}_{args.chunks.replace(',', '_')}_of{args.of}" + ("_moved" if args.moved_only else "") \
        + ("" if args.condition == "data" else f"_{args.condition}") \
        + ("_movedtoo" if args.moved_too else "") + ("_numeric" if args.numeric_only else "")
    work = work_dir(name, args.config, args.mode)
    work.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        n = sum(len(arms_for(args.config, args.mode, sets_of(r))) * bw.SHUFFLES for r in todo
                if r["answer"] and str(r["answer"]).strip())
        print(f"{len(todo)} runs, {n} reads")
        return
    asyncio.run(read_all(todo, sets_of, (name, base), args.config, args.mode, work / f"cache_{tag}.jsonl",
                         work / f"rows_{tag}.jsonl", args.concurrency))


PUBLISHED_ALL = ROOT / "results" / "published_forced_all_questions.jsonl.gz"


def build_published(args):
    """Every published open-answer run on every v1.0 question, with its published open-ended grade and the
    published forced grade of the same run (its MCQ row shares the run's notebook and answer), matched as
    ``published_reads.build`` matches them on the numeric questions. It ships, so the analysis needs no copy of
    eval_df.csv."""
    import ast
    import csv
    import sys
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
                runs[name].append((row["uuid"], row["agent_answer"], row["correct"] == "True", body))
            else:
                pool[(mcq_of[name], row["uuid"], body)].append(row["correct"] == "True")
    doc = rp.load_extract()
    out = []
    for name in rp.OPEN_RUNS:
        assert [(q, a) for q, a, _, _ in runs[name]] == [(q, a) for q, a, _ in doc["open_runs"][name]], name
        seen = collections.Counter()
        for q, answer, ok, body in runs[name]:
            out.append({"run": name, "q": q, "i": seen[q], "answered": bool(answer and answer.strip()), "open": ok,
                        "forced": pool[(name, q, body)].pop(0)})
            seen[q] += 1
    assert not any(pool.values())
    with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=PUBLISHED_ALL.open("wb")) as fh:
        fh.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in out).encode())
    print(f"{len(out)} published runs with both grades to {PUBLISHED_ALL.relative_to(ROOT)}")


def rows_path(reader, config, mode):
    return ROOT / "results" / f"grading_variants_{reader}_{config}_{mode}_rows.jsonl.gz"


def merge(args):
    rows = {}
    for path in sorted(work_dir(args.reader_name, args.config, args.mode).glob("rows_*.jsonl")):
        for line in path.open():
            r = json.loads(line)
            prior = rows.get(pr.key_of(r))
            if prior is None:
                rows[pr.key_of(r)] = r
            else:                                 # the same run read through other option sets elsewhere
                for arm, reads in r["reads"].items():
                    prior["reads"].setdefault(arm, reads)
    ordered = sorted(rows.values(), key=lambda r: (r["set"], r["run"], r["q"], r["i"]))
    dest = rows_path(args.reader_name, args.config, args.mode)
    with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=dest.open("wb")) as fh:
        fh.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in ordered).encode())
    print(f"{len(ordered)} runs to {dest.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--reader", required=True, help="NAME=BASE[,BASE...]")
    r.add_argument("--config", choices=CONFIGS, required=True)
    r.add_argument("--mode", choices=MODES, default="forced")
    r.add_argument("--runs", choices=("d1", "d0", "d2", "gpt51"), required=True)
    r.add_argument("--chunks", default="0-19")
    r.add_argument("--of", type=int, default=20)
    r.add_argument("--moved-only", action="store_true", help="only runs on keys U or U' moves to an extreme")
    r.add_argument("--moved-too", action="store_true", help="the chunks, and every run on a moved key besides")
    r.add_argument("--numeric-only", action="store_true")
    r.add_argument("--condition", choices=("data", "nodata", "both"), default="data",
                   help="the v1.5 runs with the data, without it, or both")
    r.add_argument("--concurrency", type=int, default=96)
    r.add_argument("--dry-run", action="store_true")
    b = sub.add_parser("build-published")
    b.add_argument("--eval-df", default=str(ROOT / "build" / "external" / "bixbench_v10_trajectories" / "eval_df.csv"))
    m = sub.add_parser("merge")
    m.add_argument("--reader-name", required=True)
    m.add_argument("--config", choices=CONFIGS, required=True)
    m.add_argument("--mode", choices=MODES, default="forced")
    sub.add_parser("analyse")
    args = ap.parse_args()
    if args.cmd == "analyse":
        import grading_variants_analysis as gva
        gva.main()
        return
    {"read": read, "merge": merge, "build-published": build_published}[args.cmd](args)


if __name__ == "__main__":
    main()
