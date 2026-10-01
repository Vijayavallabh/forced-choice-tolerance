#!/usr/bin/env python3
r"""The registered contrasts under BixBench's own reader (not registered).

The registered replication (PREREGISTRATION.md, ``replication.py``) reads every run by the
option nearest its submitted number, a rule that reads no notebook. BixBench reads a run
through its options otherwise: ``MCQ_EVAL_PROMPT`` shows a second model the notebook, the
question with its options shuffled and the submitted answer, and asks for a letter. This
reads the registered data that way, through the same three option sets -- released, the
placebo (rank held) and the repair (rank uniform) -- so that each registered contrast can be
restated under the benchmark's reader:

* **D1**: the published v1.0 open-answer runs of gpt-4o and Claude 3.5 Sonnet on the 159
  questions with four numeric options (5,161 runs), each shown with its notebook as
  published (``eval_df.csv``'s ``md_notebook``; the image run sets' images are not shown);
* **D2**: the new v1.5 run sets on the 105 numeric questions, with and without the data,
  their notebooks rendered as ``bixbench_withdata.read_mcq`` renders them.

The reader is gemma-3-27b, the paper's common reader (``bixbench_withdata.py``'s second
reader), served as it was there -- FP8, a 65,536-token context, greedy, 1,536 reply tokens,
a 150,000-character notebook budget -- with two shuffles per option set under
``read_mcq``'s seeds, so that the letters fall where they fell for the paper's reads. Forced
reads only. A run with no submitted answer is scored wrong without being read, as upstream.
The published runs were read once by their own model through the released options, and
that reading ships in ``eval_df.csv`` beside each run (its MCQ rows share the run's notebook
and answer); gemma's released reading is set beside it run by run.

    python3 published_reads.py build                  # build/published_reads/input.jsonl.gz
    python3 published_reads.py read --chunks 0-11 --of 20 --reader gemma27b=http://127.0.0.1:8103
    python3 published_reads.py read --d2 --reader gemma27b=...
    python3 published_reads.py merge                  # results/published_reads_rows.jsonl.gz
    python3 published_reads.py analyse                # results/published_reads.json

The same reads can be taken by another reader, or with BixBench's "insufficient information"
option added (``--mode decline``, the benchmark's may-decline reading), and of the paper's
seven v1.5 run sets as well (``--d0``); each reader and mode keeps its own work directory,
rows and report, so the gemma-3-27b forced reads above are untouched:

    python3 published_reads.py read --reader qwen72b=http://127.0.0.1:8131 --mode forced
    python3 published_reads.py read --d0 --reader gemma27b=... --mode decline
    python3 published_reads.py merge --reader-name qwen72b --mode forced
    python3 published_reads.py analyse --reader-name qwen72b --mode forced
    python3 published_reads.py compare                # results/published_reads_readers.json

``compare`` sets every reader of the published runs beside the published readings on the runs all
of them read.
"""
import argparse
import asyncio
import ast
import collections
import gzip
import hashlib
import json
import os
import pathlib
import random

import numpy as np

import bixbench_withdata as bw
import randomization
import replication as rp

ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT / "build" / "published_reads"
INPUT = WORK / "input.jsonl.gz"
ROWS = ROOT / "results" / "published_reads_rows.jsonl.gz"
OUT = ROOT / "results" / "published_reads.json"
ARMS = ("released", "placebo", "repaired")
READER = "gemma27b"
BUDGET = 150000
SEED = 20260925
MODES = ("forced", "decline")


def work_dir(reader=READER, mode="forced"):
    """The reads of one reader in one mode; gemma-3-27b's forced reads keep the first directory."""
    return WORK if (reader, mode) == (READER, "forced") else WORK / f"{reader}_{mode}"


def rows_path(reader=READER, mode="forced"):
    return ROWS if (reader, mode) == (READER, "forced") else \
        ROOT / "results" / f"published_reads_{reader}_{mode}_rows.jsonl.gz"


def out_path(reader=READER, mode="forced"):
    return OUT if (reader, mode) == (READER, "forced") else ROOT / "results" / f"published_reads_{reader}_{mode}.json"


# ---------------------------------------------------------------- build

def build(args):
    """Every published open-answer run on a numeric v1.0 question, with its notebook and the
    published MCQ reading of the same run (the MCQ rows carry the run's notebook and answer)."""
    import csv
    import sys
    csv.field_size_limit(sys.maxsize)
    sets = rp.v10_sets()
    doc = rp.load_extract()
    runs, mcq = collections.defaultdict(list), collections.defaultdict(list)
    mcq_of = {m: o for o, m in rp.MCQ_RUNS.items()}
    with open(args.eval_df, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            name, q = row["run_name"], row["uuid"]
            if q not in sets or (name not in rp.OPEN_RUNS and name not in mcq_of):
                continue
            body = hashlib.md5((row["md_notebook"] + "\x00" + row["agent_answer"]).encode()).hexdigest()
            if name in rp.OPEN_RUNS:
                runs[name].append({"q": q, "answer": row["agent_answer"], "notebook": row["md_notebook"],
                                   "question": row["question"], "body": body})
            else:
                mcq[mcq_of[name]].append({"q": q, "body": body,
                                          "picked": rp.picked_option(row["formatted_question"], row["llm_answer"]),
                                          "correct": row["correct"] == "True",
                                          "options": ast.literal_eval(row["mcq_options"])})
    out = []
    for name in rp.OPEN_RUNS:
        # the extract's order is the CSV's, so the i-th run of a question here is its i-th there
        extract = [(q, a) for q, a, _ in doc["open_runs"][name] if q in sets]
        assert [(r["q"], r["answer"]) for r in runs[name]] == extract, name
        pool = collections.defaultdict(list)
        for r in mcq[name]:
            pool[(r["q"], r["body"])].append(r)
        seen = collections.Counter()
        for r in runs[name]:
            i = seen[r["q"]]
            seen[r["q"]] += 1
            pub = pool[(r["q"], r["body"])].pop(0)
            assert pub["options"][0] == sets[r["q"]]["released"][0], r["q"]
            out.append({"set": "D1", "run": name, "q": r["q"], "i": i, "condition": "data",
                        "answer": r["answer"], "question": r["question"], "notebook": r["notebook"],
                        "published": {"picked": pub["picked"], "correct": pub["correct"]}})
        assert not any(pool.values()), name
    WORK.mkdir(parents=True, exist_ok=True)
    with gzip.open(INPUT, "wt") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(out)} published runs on {len(sets)} numeric questions to {INPUT.relative_to(ROOT)}")


PUBLISHED_DECLINE = ROOT / "results" / "published_reads_published_decline.jsonl.gz"
DECLINE_RUNS = {"4o_open_image": "4o_mcq_image_with_refusal", "4o_open_no_image": "4o_mcq_no_image_with_refusal",
                "claude_open_image": "claude_mcq_image_with_refusal",
                "claude_open_no_image": "claude_mcq_no_image_with_refusal"}


def build_decline(args):
    """The published may-decline reading of every run ``build`` keeps: BixBench's own run of
    MCQ_EVAL_PROMPT with the "insufficient information" option, matched to its open-answer run by
    notebook and answer as ``build`` matches the forced reading. It ships, so the analysis needs
    no copy of eval_df.csv."""
    import csv
    import sys
    csv.field_size_limit(sys.maxsize)
    sets = rp.v10_sets()
    of = {m: o for o, m in DECLINE_RUNS.items()}
    pool = collections.defaultdict(list)
    with open(args.eval_df, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            name, q = row["run_name"], row["uuid"]
            if q not in sets or name not in of:
                continue
            body = hashlib.md5((row["md_notebook"] + "\x00" + row["agent_answer"]).encode()).hexdigest()
            letter = (row["llm_answer"] or "").strip().upper()[:1]
            pool[(of[name], q, body)].append({
                "picked": rp.picked_option(row["formatted_question"], row["llm_answer"]),
                "correct": row["correct"] == "True",
                "refused": bool(letter) and letter == (row["insufficient_letter"] or "").strip().upper()[:1]})
    out = []
    for r in d1_inputs():
        body = hashlib.md5((r["notebook"] + "\x00" + r["answer"]).encode()).hexdigest()
        out.append({"key": key_of(r), **pool[(r["run"], r["q"], body)].pop(0)})
    assert not any(pool.values())
    with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=PUBLISHED_DECLINE.open("wb")) as fh:
        fh.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in out).encode())
    print(f"{len(out)} published may-decline readings to {PUBLISHED_DECLINE.relative_to(ROOT)}")


def d1_inputs():
    with gzip.open(INPUT, "rt") as fh:
        return [json.loads(line) for line in fh]


def d2_inputs():
    """The new v1.5 runs on the numeric questions, as the registered analysis reads them."""
    sets = bw.option_sets()
    items = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    out = []
    for t in rp.d2_trajectories(packed=None):
        if t["question_id"] not in sets:
            continue
        out.append({"set": "D2", "run": f"{bw.run_key(t)}|r{t['rollout']}", "q": t["question_id"],
                    "i": t["rollout"], "condition": t["condition"], "answer": t.get("answer"),
                    "question": items[t["question_id"]]["question"], "cells": t["cells"]})
    return out


def d0_inputs():
    """The paper's seven v1.5 run sets on the numeric questions, with and without the data: the
    first rollout of each, the runs ``bracketing.py`` and ``reader_split.py`` read."""
    import bracketing as br
    sets = bw.option_sets()
    items = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    want = set(br.RUNS)
    out = []
    for t in bw.load_trajectories(ROOT / "build" / "agent_runs"):
        if bw.run_key(t) not in want or t["rollout"] != 0 or t["question_id"] not in sets:
            continue
        out.append({"set": "D0", "run": bw.run_key(t), "q": t["question_id"], "i": 0, "condition": t["condition"],
                    "answer": t.get("answer"), "question": items[t["question_id"]]["question"], "cells": t["cells"]})
    return out


def key_of(r):
    return f"{r['set']}|{r['run']}|{r['condition']}|{r['q']}|{r['i']}"


# ---------------------------------------------------------------- read

async def read_one(r, options, shuffle, client, reader, refusal=False):
    """One MCQ_EVAL_PROMPT read, as bixbench_withdata.read_mcq takes it: the same seed, shuffle,
    prompt and parse; the published notebook is shown as published. With ``refusal`` the options
    carry BixBench's "insufficient information" option, as its may-decline reading does."""
    name, base = reader
    seed = int(hashlib.sha256(f"{r['q']}|{shuffle}|{refusal}".encode()).hexdigest()[:8], 16)
    formatted, correct, refusal_letter, shown = bw.questions_to_mcq(r["question"], options, refusal,
                                                                    random.Random(seed))
    budget = BUDGET
    for _ in range(3):
        notebook = (bw.notebook_markdown(r["cells"], budget) if "cells" in r
                    else r["notebook"] if len(r["notebook"]) <= budget else r["notebook"][-budget:])
        prompt = (bw.PROMPTS["MCQ_EVAL_PROMPT"].replace("{{notebook}}", notebook)
                  .replace("{{question}}", formatted).replace("{{proposed_answer}}", str(r["answer"])))
        reply = await client.ask(name, base, prompt, bw.READ_TOKENS.get(name, 1536))
        if not reply.startswith("[error 400"):
            break
        budget //= 2
    letter = bw.xml_extract(reply)
    picked = shown[ord(letter) - 65] if letter != "Z" and ord(letter) - 65 < len(shown) else None
    out = {"correct": letter == correct, "no_pick": letter == "Z", "rank": bw.pick_rank(picked, options),
           "server_error": reply.startswith("[error")}
    if refusal:
        out["refused"] = letter == refusal_letter
    return out


async def read_all(todo, sets_of, reader, cache, rows_path, concurrency, refusal=False):
    client = bw.Client(cache, concurrency)
    # a run read by any pass, on this machine or copied from another, is not read again
    done = {key_of(json.loads(l)) for path in sorted(rows_path.parent.glob("rows_*.jsonl")) for l in path.open()}
    todo = [r for r in todo if key_of(r) not in done]
    print(f"{len(todo)} runs to read ({len(done)} already read)", flush=True)
    handle = rows_path.open("a")
    count = 0
    # OpenAI's prompt cache fills only once a request has been served, and a run's later reads
    # queued behind every other run's first read would meet it expired: so on OpenAI's API a
    # run sends one read, then the rest, and only as many runs are in flight as keep the
    # connections busy.
    native = bw.oa.is_openai(reader[1].split(",")[0])
    runs_in_flight = asyncio.Semaphore(max(1, concurrency // 4)) if native else None

    async def one(r):
        nonlocal count
        # all of a run's reads go to one replica, so they share its notebook in the prefix cache
        bases = reader[1].split(",")
        base = bases[int(hashlib.sha256(key_of(r).encode()).hexdigest()[:8], 16) % len(bases)]
        row = {k: r[k] for k in ("set", "run", "q", "i", "condition", "answer") if k in r}
        if "published" in r:
            row["published"] = r["published"]
        row["reads"] = {}
        if r["answer"] and str(r["answer"]).strip():
            jobs = [(arm, s) for arm in ARMS if sets_of(r).get(arm) is not None for s in range(bw.SHUFFLES)]
            if native and jobs:
                async with runs_in_flight:
                    first = await read_one(r, sets_of(r)[jobs[0][0]], jobs[0][1], client, (reader[0], base), refusal)
                    rest = await asyncio.gather(*(read_one(r, sets_of(r)[arm], s, client, (reader[0], base), refusal)
                                                  for arm, s in jobs[1:]))
                reads = [first, *rest]
            else:
                # every read of the run at once, so that they meet its notebook still in the cache
                reads = await asyncio.gather(*(read_one(r, sets_of(r)[arm], s, client, (reader[0], base), refusal)
                                               for arm, s in jobs))
            for (arm, _), x in zip(jobs, reads):
                row["reads"].setdefault(arm, []).append(x)
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        count += 1
        if count % 100 == 0:
            print(f"  {count}/{len(todo)} read; calls {dict(client.calls)}", flush=True)

    await asyncio.gather(*(one(r) for r in todo))
    await client.http.aclose()
    handle.close()
    print(f"done; calls {dict(client.calls)}", flush=True)
    if native:
        print(f"content filtered: {client.calls['content_filter']} reads", flush=True)


def chunk_of(r, of):
    return int(hashlib.sha256(key_of(r).encode()).hexdigest()[:8], 16) % of


def parse_chunks(spec):
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


CHARS_PER_TOKEN = 3.5         # this corpus's notebooks: code, tables and prose


def reply_tokens_estimate(default=500):
    """Tokens a read's reply runs to: the median of the other readers' cached replies, at 4 characters
    a token (they are prose), or ``default`` where none is on this host."""
    lengths = []
    for path in sorted(WORK.glob("*/cache_*.jsonl"))[:6]:
        with path.open() as fh:
            for _, line in zip(range(2000), fh):
                try:
                    lengths.append(len(json.loads(line)["reply"]))
                except (ValueError, KeyError):
                    continue
    return int(np.median(lengths) / 4) if lengths else default


def dry_run(todo, sets_of, name, base, work, refusal):
    """The reads a ``read`` would take, their prompt tokens (and on OpenAI's API the share its prompt
    cache would hold: a run's later reads share its first one's notebook), and the cost; nothing is called."""
    done = {key_of(json.loads(l)) for path in sorted(work.glob("rows_*.jsonl")) for l in path.open()}
    todo = [r for r in todo if key_of(r) not in done]
    native = bw.oa.is_openai(base.split(",")[0])
    runs = reads = prompt = cached = 0
    for r in todo:
        if not (r["answer"] and str(r["answer"]).strip()):
            continue
        prompts = []
        for arm in ARMS:
            if sets_of(r).get(arm) is None:
                continue
            for shuffle in range(bw.SHUFFLES):
                seed = int(hashlib.sha256(f"{r['q']}|{shuffle}|{refusal}".encode()).hexdigest()[:8], 16)
                formatted = bw.questions_to_mcq(r["question"], sets_of(r)[arm], refusal, random.Random(seed))[0]
                notebook = (bw.notebook_markdown(r["cells"], BUDGET) if "cells" in r
                            else r["notebook"] if len(r["notebook"]) <= BUDGET else r["notebook"][-BUDGET:])
                prompts.append(bw.PROMPTS["MCQ_EVAL_PROMPT"].replace("{{notebook}}", notebook)
                               .replace("{{question}}", formatted).replace("{{proposed_answer}}", str(r["answer"])))
        runs, reads = runs + 1, reads + len(prompts)
        prompt += sum(len(p) for p in prompts) / CHARS_PER_TOKEN
        if native:
            for later in prompts[1:]:
                shared = len(os.path.commonprefix([prompts[0], later])) / CHARS_PER_TOKEN
                # OpenAI caches a prefix of at least 1,024 tokens, in steps of 128
                cached += 1024 + 128 * ((shared - 1024) // 128) if shared >= 1024 else 0
    per_reply = reply_tokens_estimate()
    out = reads * per_reply
    print(f"dry run, {name} at {base}: {runs} runs to read ({len(done)} already read), {reads} reads; "
          f"{prompt / 1e6:.2f} M prompt tokens" + (f", ~{cached / 1e6:.2f} M of them cached" if native else "")
          + f", ~{out / 1e6:.2f} M reply tokens (~{per_reply} a read)")
    if native:
        inp, cin, cout = bw.oa.prices(name)
        cost = ((prompt - cached) * inp + cached * cin + out * cout) / 1e6
        print(f"  estimated cost ${cost:,.0f} at ${inp}/{cin}/{cout} per 1M input/cached/output tokens")


def read(args):
    name, _, base = args.reader.partition("=")
    v10, v15 = rp.v10_sets(), bw.option_sets()
    work = work_dir(name, args.mode)
    if not args.dry_run:
        work.mkdir(parents=True, exist_ok=True)
    sets_of = lambda r: v10[r["q"]] if r["set"] == "D1" else v15[r["q"]]
    refusal = args.mode == "decline"
    if args.read_tokens:
        bw.READ_TOKENS[name] = args.read_tokens
    if args.d0:
        # one pass per run set, each on a cache named as that run set's shipped reader cache, so a
        # read the paper already took (the same prompt, reader and reply budget) is not taken again
        todo = d0_inputs()
        if args.moved_only:
            todo = [r for r in todo if rp.group_of(sets_of(r)) == "moved to an edge"]
        if args.dry_run:
            dry_run(todo, sets_of, name, base, work, refusal)
            return
        for run in sorted({r["run"] for r in todo}):
            asyncio.run(read_all([r for r in todo if r["run"] == run], sets_of, (name, base),
                                 work / f"reader_cache_{run}.jsonl", work / f"rows_d0_{run}.jsonl",
                                 args.concurrency, refusal))
        return
    chunks = parse_chunks(args.chunks)
    if args.d2:
        todo = [r for r in d2_inputs() if chunk_of(r, args.of) in chunks]
        tag = "d2" if (args.chunks, args.of) == ("0-19", 20) else f"d2_{args.chunks.replace(',', '_')}_of{args.of}"
    else:
        todo = [r for r in d1_inputs() if chunk_of(r, args.of) in chunks]
        tag = f"d1_{args.chunks.replace(',', '_')}_of{args.of}"
    if args.moved_only:
        # every run on a question whose key the repair moved to an edge: the rank's contrast
        todo = [r for r in todo if rp.group_of(sets_of(r)) == "moved to an edge"]
        tag += "_moved"
    if args.data_only:
        todo = [r for r in todo if r["condition"] == "data"]
        tag += "_data"
    if args.dry_run:
        dry_run(todo, sets_of, name, base, work, refusal)
        return
    asyncio.run(read_all(todo, sets_of, (name, base), work / f"cache_{tag}.jsonl", work / f"rows_{tag}.jsonl",
                         args.concurrency, refusal))


def merge(args):
    """Every read row, one per run, into the file that ships."""
    rows = {}
    for path in sorted(work_dir(args.reader_name, args.mode).glob("rows_*.jsonl")):
        for line in path.open():
            r = json.loads(line)
            rows.setdefault(key_of(r), r)
    ordered = sorted(rows.values(), key=lambda r: (r["set"], r["run"], r["condition"], r["q"], r["i"]))
    dest = rows_path(args.reader_name, args.mode)
    with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=dest.open("wb")) as fh:
        fh.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in ordered).encode())
    print(f"{len(ordered)} runs to {dest.relative_to(ROOT)}")


# ---------------------------------------------------------------- analyse

def load_rows(path=ROWS):
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def score(row, arm):
    """The run's forced reading through one option set: the share of its two shuffles that name
    the key; 0 for a run with no submitted answer, as upstream scores it; None if not read."""
    if not row["reads"]:
        return 0.0 if not (row["answer"] and str(row["answer"]).strip()) else None
    reads = row["reads"].get(arm)
    return None if reads is None else float(np.mean([x["correct"] for x in reads]))


def gains(rows, arm, base):
    """Per run set and item, arm minus base, each item's runs averaged; pooled, each item
    averaged over run sets -- replication.gains, with the reader's score in place of the rule's."""
    per_run = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        a, b = score(r, arm), score(r, base)
        if a is not None and b is not None:
            per_run[r["run"]][r["q"]].append(a - b)
    pooled = collections.defaultdict(list)
    for per_item in per_run.values():
        for q, v in per_item.items():
            pooled[q].append(float(np.mean(v)))
    return {q: float(np.mean(v)) for q, v in pooled.items()}


def rule_gains(rows, sets, arm, base):
    """The registered analysis's own pooled gains (the nearest-option rule), for the sign-flip check."""
    runs = collections.defaultdict(list)
    for r in rows:
        runs[r["run"]].append((r["q"], r["answer"]))
    _, pooled = rp.gains(dict(runs), {q: s for q, s in sets.items() if arm in s and base in s}, arm, base)
    return pooled


def contrasts(rows, sets, capsule, rng, heavy, label="gemma"):
    outer, inner = (4000, 20000) if heavy else (2000, 5000)
    groups = {g: {q for q, s in sets.items() if rp.group_of(s) == g} for g in rp.GROUPS}
    groups["all"] = set(sets)
    out = {"n_runs": len(rows), "n_items": {g: len(v) for g, v in groups.items()}, "signflip": {}}
    for arm, base in (("repaired", "released"), ("repaired", "placebo"), ("placebo", "released")):
        pooled = gains(rows, arm, base)
        for g, items in groups.items():
            per_item = {q: v for q, v in pooled.items() if q in items}
            out[f"{arm}-{base}|{g}"] = rp.calibrated(per_item, capsule, rng, outer, inner) if per_item else None
        # a capsule sign-flip test of the moved keys' contrast, under this reader and under the rule
        moved = groups["moved to an edge"]
        out["signflip"][f"{label}|{arm}-{base}|moved to an edge"] = randomization.signflip(
            {q: v for q, v in pooled.items() if q in moved}, capsule)
        rule = rule_gains(rows, sets, arm, base)
        out["signflip"][f"rule|{arm}-{base}|moved to an edge"] = randomization.signflip(
            {q: v for q, v in rule.items() if q in moved}, capsule)
    for arm in ARMS:
        vals = [score(r, arm) for r in rows]
        out[f"read|{arm}"] = 100.0 * float(np.mean([v for v in vals if v is not None]))
        declined = [x["refused"] for r in rows for x in r["reads"].get(arm, []) if "refused" in x]
        if declined:
            out[f"declined|{arm}"] = 100.0 * float(np.mean(declined))
    return out


def kappa_against(rows, published):
    """A reader's released reading against the published reading of the same runs, run by run
    (a run counts as named when both of the reader's shuffles name the key), per run set and over
    all of them ("pooled")."""
    out = {}
    for name in rp.OPEN_RUNS + ("pooled",):
        pairs = [((float(np.mean([x["correct"] for x in r["reads"]["released"]])) if r["reads"] else 0.0),
                  float(published(r))) for r in rows if name in (r["run"], "pooled")
                 and (not r["reads"] or "released" in r["reads"])]
        if not pairs:
            continue
        g = np.array([a for a, _ in pairs])
        p = np.array([b for _, b in pairs])
        gb = (g == 1.0).astype(float)
        po = float(np.mean(gb == p))
        pe = float(np.mean(gb) * np.mean(p) + (1 - np.mean(gb)) * (1 - np.mean(p)))
        out[name] = {"n": len(pairs), "published_right": 100.0 * float(np.mean(p)),
                     "reader_right": 100.0 * float(np.mean(g)), "agree": 100.0 * po,
                     "kappa": (po - pe) / (1 - pe) if pe < 1 else None}
    return out


def agreement3(rows, published):
    """A may-decline reading against the published may-decline reading over the three outcomes a read can
    have -- the key, another option, a decline -- each of the reader's reads against the published reading
    of its run; runs the reader did not read (no answer) are left out."""
    kind = lambda correct, refused: "declined" if refused else ("key" if correct else "other")
    pairs = [(kind(x["correct"], x.get("refused", False)), kind(published(r)["correct"], published(r)["refused"]))
             for r in rows if r["reads"] for x in r["reads"].get("released", [])]
    cats = ("key", "other", "declined")
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in cats)
    confusion = {a: {b: sum(1 for x, y in pairs if (x, y) == (a, b)) for b in cats} for a in cats}
    return {"n_reads": n, "agree": 100.0 * po, "kappa": (po - pe) / (1 - pe), "reader_by_published": confusion}


SAMPLE_CHUNKS, SAMPLE_OF = range(0, 5), 20


def released_picks(r):
    """The ranks a reader's released reads name (None where a read names no option)."""
    return [x.get("rank") for x in r["reads"].get("released", [])] if r["reads"] else []


def pick_is_nearest(rows, sets, picks=released_picks):
    """How often a reading names the option nearest the submitted number -- the option the rule
    names -- over the reads that name an option on runs whose answer is a number with one nearest
    option, and over those whose number is a miss, beyond 5% of the key, where following the nearest
    option is what credits a miss when the key is nearest. An answer at the same distance from two
    options (one of the other sign, or zero, is at the same distance from all four) has no nearest
    option to follow, and any pick would count as following it, so those runs are left out."""
    import score_decomposition as sd
    tally = {"all": [0, 0], "miss": [0, 0]}
    for r in rows:
        options = sets[r["q"]]["released"]
        ranks = sd.nearest_ranks(r["answer"], options) if r["answer"] else None
        if ranks is None or len(ranks) != 1:
            continue
        miss = not sd.graded(r["answer"], options[0], sd.TOL)
        for rank in picks(r):
            if rank is None:
                continue
            for tag in ("all", "miss") if miss else ("all",):
                tally[tag][0] += rank in ranks
                tally[tag][1] += 1
    return {tag: {"share": 100.0 * a / n if n else None, "n_reads": n} for tag, (a, n) in tally.items()}


COMPARED = (("gemma27b", "gemma-3-27b"), ("qwen72b", "Qwen2.5-72B"), ("llama70b", "Llama-3.3-70B"),
            ("gpt-4o", "gpt-4o"))


def compare(args):
    """Every reader of the published runs beside BixBench's published readings of the same runs, on
    the random sample every reader read in full (chunks 0-4 of 20): how often each names the option
    nearest the submitted number, over all numeric answers and over the misses; its agreement with
    the published reading in its mode; and the moved keys' gain against the placebo from its own
    report. The rule is set beside them: it names the nearest option by definition."""
    v10 = rp.v10_sets()
    gem = [r for r in load_rows() if r["set"] == "D1"]
    base = [r for r in gem if chunk_of(r, SAMPLE_OF) in SAMPLE_CHUNKS]
    keys = {key_of(r) for r in base}
    with gzip.open(PUBLISHED_DECLINE, "rt") as fh:
        pub_dec = {x["key"]: x for x in map(json.loads, fh)}

    def rank_of(r, picked):
        options = v10[r["q"]]["released"]
        return bw.pick_rank(picked, options) if picked in options else None

    mv = "repaired-placebo|moved to an edge"
    report = {"sample": f"chunks {min(SAMPLE_CHUNKS)}-{max(SAMPLE_CHUNKS)} of {SAMPLE_OF}", "n_runs": len(base),
              "not_registered": True,
              "published forced": {"follows_nearest": pick_is_nearest(base, v10,
                                                                      lambda r: [rank_of(r, r["published"]["picked"])])},
              "published may-decline": {
                  "follows_nearest": pick_is_nearest(base, v10, lambda r: [rank_of(r, pub_dec[key_of(r)]["picked"])]),
                  "declined": 100.0 * float(np.mean([pub_dec[key_of(r)]["refused"] for r in base]))},
              "rule": {"agreement": kappa_against(
                  [dict(r, reads=({"released": [{"correct": bw.nearest_is_key(r["answer"], v10[r["q"]]["released"])}]}
                                  if r["answer"] and str(r["answer"]).strip() else {})) for r in base],
                  lambda r: r["published"]["correct"]),
                  mv: json.loads((ROOT / "results" / "replication.json").read_text())["D1"][mv],
                  "signflip": json.loads(OUT.read_text())["D1"]["signflip"][f"rule|{mv}"]}}
    # the rule's contrast on the other keys, over every published run, pooled as the registered analysis pools
    rng = np.random.default_rng(SEED)
    cap10 = {q: e["capsule"] for q, e in rp.load_extract()["items"].items()}
    rule = rule_gains(gem, v10, "repaired", "placebo")
    for g in ("kept", "all"):
        items = {q: v for q, v in rule.items() if g == "all" or rp.group_of(v10[q]) == g}
        report["rule"][f"repaired-placebo|{g}"] = rp.calibrated(items, cap10, rng, 4000, 20000)
    for name, label in COMPARED + (("gemma27b|decline", "gemma-3-27b, may decline"),):
        reader, mode = name.split("|")[0], ("decline" if name.endswith("|decline") else "forced")
        if not out_path(reader, mode).exists():
            continue
        rows = gem if (reader, mode) == (READER, "forced") else \
            [r for r in load_rows(rows_path(reader, mode)) if r["set"] == "D1"]
        sample = [r for r in rows if key_of(r) in keys]
        assert len(sample) == len(base), (name, len(sample), len(base))
        rep = json.loads(out_path(reader, mode).read_text())["D1"]
        published = ((lambda r: r["published"]["correct"]) if mode == "forced"
                     else (lambda r: pub_dec[key_of(r)]["correct"]))
        entry = {"label": label, "follows_nearest": pick_is_nearest(sample, v10),
                 "agreement": kappa_against(sample, published),
                 mv: rep[mv], "signflip": rep["signflip"][f"{'gemma' if name == READER else reader}|{mv}"]}
        if mode == "decline":
            entry["declined"] = 100.0 * float(np.mean([x["refused"] for r in sample
                                                       for x in r["reads"].get("released", [])]))
            entry["agreement3"] = agreement3(sample, lambda r: pub_dec[key_of(r)])
        report[name] = entry
    dest = ROOT / "results" / "published_reads_readers.json"
    dest.write_text(json.dumps(report, indent=1) + "\n")
    for name, e in report.items():
        if isinstance(e, dict) and "follows_nearest" in e:
            fn = e["follows_nearest"]
            kap = " ".join(f"{a['kappa']:.2f}" for a in e.get("agreement", {}).values())
            print(f"{name:26s} nearest {fn['all']['share']:.1f} (miss {fn['miss']['share']:.1f}, "
                  f"n {fn['miss']['n_reads']})  kappa {kap or '--'}"
                  + (f"  moved {e[mv]['mean']:+.1f} [{e[mv]['lo']:+.1f},{e[mv]['hi']:+.1f}]"
                     if mv in e else "") + (f"  declined {e['declined']:.1f}" if "declined" in e else ""))
    print(f"wrote {dest.relative_to(ROOT)}")


def analyse_other(args):
    """Any reader and mode but gemma-3-27b forced (``analyse``): the same contrasts on whatever of
    D0, D1 and D2 it read; for D1, agreement with the published reading in the same mode."""
    rng = np.random.default_rng(SEED)
    rows = load_rows(rows_path(args.reader_name, args.mode))
    report = {"reader": args.reader_name, "mode": args.mode, "seed": SEED, "not_registered": True}
    doc = rp.load_extract()
    v10, v15 = rp.v10_sets(), bw.option_sets()
    cap10 = {q: e["capsule"] for q, e in doc["items"].items()}
    items15 = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    cap15 = {q: it["capsule_uuid"] for q, it in items15.items()}
    label = args.reader_name
    d1 = [r for r in rows if r["set"] == "D1"]
    if d1:
        report["D1"] = contrasts(d1, v10, cap10, rng, heavy=True, label=label)
        if args.mode == "forced":
            published = lambda r: r["published"]["correct"]
        else:
            with gzip.open(PUBLISHED_DECLINE, "rt") as fh:
                pub = {x["key"]: x for x in map(json.loads, fh)}
            published = lambda r: pub[key_of(r)]["correct"]
        # agreement is read on a random sample of the runs: all of them, or the chunks every run of which
        # was read when the rest were read on the moved keys alone (--moved-only)
        full = len(d1) == len(d1_inputs())
        sample = d1 if full else [r for r in d1 if chunk_of(r, SAMPLE_OF) in SAMPLE_CHUNKS]
        report["D1"]["agreement_sample"] = "all runs" if full else f"chunks {min(SAMPLE_CHUNKS)}-{max(SAMPLE_CHUNKS)} of {SAMPLE_OF}"
        report["D1"]["agreement_with_published_reader"] = kappa_against(sample, published)
        report["D1"]["runs_by_group"] = dict(collections.Counter(rp.group_of(v10[r["q"]]) for r in d1))
        report["D1"]["pick_is_nearest"] = pick_is_nearest(sample, v10)
    d0 = [r for r in rows if r["set"] == "D0"]
    for cond in ("data", "nodata"):
        sub = [r for r in d0 if r["condition"] == cond]
        if sub:
            report[f"D0|{cond}"] = contrasts(sub, v15, cap15, rng, heavy=True, label=label)
    d2 = [r for r in rows if r["set"] == "D2"]
    for fam, want in (("qwen3-235b", lambda n: n.startswith("qwen3-235b|")),
                      ("reruns", lambda n: not n.startswith("qwen3-235b|"))):
        for cond in ("data", "nodata"):
            sub = [r for r in d2 if want(r["run"]) and r["condition"] == cond]
            if sub:
                report[f"D2|{fam}|{cond}"] = contrasts(sub, v15, cap15, rng, heavy=True, label=label)
    # a set read on the moved keys alone (--moved-only) has nothing to say about the other groups
    for key, res in list(report.items()):
        if not isinstance(res, dict) or "read|released" not in res:
            continue
        sets = v10 if key == "D1" else v15
        present = {rp.group_of(sets[r["q"]]) for r in (d1 if key == "D1" else d0 + d2)
                   if (key == "D1") or key.startswith("D0") == (r["set"] == "D0")}
        res["groups_read"] = sorted(g for g in present if g)
        for arm, base in (("repaired", "released"), ("repaired", "placebo"), ("placebo", "released")):
            for g in rp.GROUPS:
                if g not in present:
                    res[f"{arm}-{base}|{g}"] = None
            if len(present - {None}) < len(rp.GROUPS):
                res[f"{arm}-{base}|all"] = None
    reads = [x for r in rows for a in r["reads"].values() for x in a]
    report["no_pick"] = 100.0 * float(np.mean([x["no_pick"] for x in reads])) if reads else None
    out = out_path(args.reader_name, args.mode)
    out.write_text(json.dumps(report, indent=1) + "\n")
    f = lambda h: f"{h['mean']:+.1f} [{h['lo']:+.1f},{h['hi']:+.1f}]" if h else "--"
    for key, res in report.items():
        if not isinstance(res, dict) or "read|released" not in res:
            continue
        print(f"{key}: runs {res['n_runs']}; reads released {res['read|released']:.1f} placebo {res['read|placebo']:.1f} "
              f"repaired {res['read|repaired']:.1f}" + "".join(f"; declined {a} {res[f'declined|{a}']:.1f}"
                                                            for a in ARMS if f"declined|{a}" in res))
        for g in ("moved to an edge", "moved inward", "kept", "all"):
            sf = res["signflip"][f"{label}|repaired-placebo|moved to an edge"]
            print(f"   {g:17s} rep-rel {f(res[f'repaired-released|{g}'])}  rep-pla {f(res[f'repaired-placebo|{g}'])}  "
                  f"pla-rel {f(res[f'placebo-released|{g}'])}" + (f"  p={sf['p']:.3f}" if g == "moved to an edge" and sf
                                                                  else ""))
    if "D1" in report:
        print(f"   agreement on {report['D1']['agreement_sample']}; pick is nearest {report['D1']['pick_is_nearest']}")
        for name, a in report["D1"]["agreement_with_published_reader"].items():
            print(f"   {name}: n {a['n']} published {a['published_right']:.1f} reader {a['reader_right']:.1f} "
                  f"agree {a['agree']:.1f} kappa {a['kappa']:.2f}")
    print(f"no pick {report['no_pick']:.1f}%; wrote {out.relative_to(ROOT)}")


def agreement(rows):
    """gemma's released reading against the published reading of the same run, run by run:
    a run counts as named by gemma when both shuffles name the key."""
    out = {}
    for name in rp.OPEN_RUNS:
        pairs = [(float(np.mean([x["correct"] for x in r["reads"]["released"]])) if r["reads"] else 0.0,
                  float(r["published"]["correct"])) for r in rows if r["run"] == name]
        g = np.array([p[0] for p in pairs])
        p = np.array([p[1] for p in pairs])
        gb = (g == 1.0).astype(float)
        po = float(np.mean(gb == p))
        pe = float(np.mean(gb) * np.mean(p) + (1 - np.mean(gb)) * (1 - np.mean(p)))
        out[name] = {"n": len(pairs), "published_right": 100.0 * float(np.mean(p)),
                     "gemma_right": 100.0 * float(np.mean(g)), "agree": 100.0 * po,
                     "kappa": (po - pe) / (1 - pe) if pe < 1 else None}
    return out


def analyse(args):
    if (args.reader_name, args.mode) != (READER, "forced"):
        return analyse_other(args)
    rng = np.random.default_rng(SEED)
    rows = load_rows()
    report = {"reader": READER, "seed": SEED, "not_registered": True}
    doc = rp.load_extract()
    v10 = rp.v10_sets()
    cap10 = {q: e["capsule"] for q, e in doc["items"].items()}
    d1 = [r for r in rows if r["set"] == "D1"]
    report["D1"] = contrasts(d1, v10, cap10, rng, heavy=True)
    report["D1"]["agreement_with_published_reader"] = agreement(d1)
    report["D1"]["no_pick"] = 100.0 * float(np.mean([x["no_pick"] for r in d1 for a in r["reads"].values() for x in a]))
    v15 = bw.option_sets()
    items15 = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    cap15 = {q: it["capsule_uuid"] for q, it in items15.items()}
    d2 = [r for r in rows if r["set"] == "D2"]
    report["D2"] = {}
    for fam, want in (("qwen3-235b", lambda n: n.startswith("qwen3-235b|")),
                      ("reruns", lambda n: not n.startswith("qwen3-235b|"))):
        for cond in ("data", "nodata"):
            sub = [r for r in d2 if want(r["run"]) and r["condition"] == cond]
            if sub:
                report["D2"][f"{fam}|{cond}"] = contrasts(sub, v15, cap15, rng, heavy=True)
    report["D2"]["no_pick"] = 100.0 * float(np.mean([x["no_pick"] for r in d2 for a in r["reads"].values() for x in a])) if d2 else None
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    f = lambda h: f"{h['mean']:+.1f} [{h['lo']:+.1f},{h['hi']:+.1f}]" if h else "--"
    for key, res in [("D1", report["D1"])] + [(f"D2 {k}", v) for k, v in report["D2"].items() if k != "no_pick"]:
        print(f"{key}: reads released {res['read|released']:.1f} placebo {res['read|placebo']:.1f} "
              f"repaired {res['read|repaired']:.1f}")
        for g in ("moved to an edge", "moved inward", "kept", "all"):
            print(f"   {g:17s} rep-rel {f(res[f'repaired-released|{g}'])}  rep-pla {f(res[f'repaired-placebo|{g}'])}  "
                  f"pla-rel {f(res[f'placebo-released|{g}'])}")
    print(f"wrote {OUT.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "build-decline"):
        b = sub.add_parser(name)
        b.add_argument("--eval-df", default=str(ROOT / "build" / "external" / "bixbench_v10_trajectories" / "eval_df.csv"))
    r = sub.add_parser("read")
    r.add_argument("--reader", required=True, help="gemma27b=BASE[,BASE...]")
    r.add_argument("--chunks", default="0-19")
    r.add_argument("--of", type=int, default=20)
    r.add_argument("--d2", action="store_true", help="read the new v1.5 runs instead of the published ones")
    r.add_argument("--d0", action="store_true", help="read the paper's seven v1.5 run sets instead")
    r.add_argument("--mode", choices=MODES, default="forced",
                   help="forced, or with BixBench's 'insufficient information' option (decline)")
    r.add_argument("--concurrency", type=int, default=192)
    r.add_argument("--moved-only", action="store_true",
                   help="only the runs on questions whose key the repair moved to an edge")
    r.add_argument("--data-only", action="store_true", help="only the runs made with the data")
    r.add_argument("--read-tokens", type=int, default=None,
                   help="reply tokens the reader may spend on one read (default 1536, the paper's readers')")
    r.add_argument("--dry-run", action="store_true",
                   help="count the reads this would take and estimate their tokens and cost; call nothing")
    for name in ("merge", "analyse"):
        p = sub.add_parser(name)
        p.add_argument("--reader-name", default=READER)
        p.add_argument("--mode", choices=MODES, default="forced")
    sub.add_parser("compare")
    args = ap.parse_args()
    {"build": build, "build-decline": build_decline, "read": read, "merge": merge,
     "analyse": analyse, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
