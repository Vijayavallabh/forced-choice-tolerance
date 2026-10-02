#!/usr/bin/env python3
"""Does the option geometry reach the scores agents get *with* the data?

``bixbench_agent.py`` ran BixBench's published agent in open-answer mode, with
the capsule's data and without it. The agent never saw the options. This script
applies the two readings BixBench publishes for such a run, both verbatim from
the vendored sources:

* **open-ended** -- the v1.5 graders (``graders.py``): each item's own
  ``eval_mode``; a string verifier that tries an exact then a partial match
  before an LLM, a range verifier and an LLM verifier, with the upstream
  prompts. Beside it, two LLM-free grades for the numeric items: the answer's
  number, rounded to the precision the key is written at, equals the key; and
  the tolerance's rule, the number within 5% of the key.
* **through the options** -- ``MCQ_EVAL_PROMPT``: a reader model is shown the
  agent's notebook, the question with the options shuffled, and the agent's
  answer, and picks a letter, with and without BixBench's refusal option.
  Upstream formatting (``questions_to_mcq``), upstream parse
  (``<answer>X</answer>``, anything else counts as no pick), and, as upstream,
  a run with no submitted answer is scored wrong without being read.

Because the agent never saw the options, the options can be swapped *under a
fixed trajectory*. On BixBench's 105 numeric items the same notebook and answer
are read through three option sets that share the key: the released one; a
placebo that redraws every distractor and holds the key's rank; and the repair,
which draws the key's rank uniformly (``build/bixbench_numeric_*.jsonl``). The
agent's work is identical across the three, so any difference in the MCQ score
between them was put there by the option set and collected by the reader.

Each family grades and reads its own runs, as the published runs were; a
second ``--reader`` reads the same runs again, to see whether an effect belongs
to the option set or to one reader, and a reading with no reader at all -- the
option nearest the agent's own number (``nearest_is_key``) -- sets each option
set against the answers directly. Every reader and judge call is greedy and
cached (``--cache``), keyed by model and prompt, so a rerun reproduces the
scores without a server, and ``--summarise-only`` over the shipped rows
re-derives ``results/bixbench_withdata.json``.

    python3 bixbench_withdata.py --models qwen72b \\
        --reader qwen72b=http://127.0.0.1:18101 --judge qwen72b=http://127.0.0.1:18101 \\
        --cache build/agent_runs/reader_cache_qwen72b.jsonl \\
        --rows build/agent_runs/withdata_rows_qwen72b.json
    python3 bixbench_withdata.py --summarise-only --rows build/agent_runs/withdata_rows_qwen72b.json ...
    python3 bixbench_withdata.py --latex         # the two with-data tables' rows
"""
import argparse
import asyncio
import gzip
import hashlib
import json
import random
import re
import runpy
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import numpy as np

import openai_api as oa
from answer_numbers import nearest_ranks
from option_artifacts import parse_number

ROOT = Path(__file__).resolve().parent
PROMPTS = runpy.run_path(str(ROOT / "sources" / "bixbench_49311180" / "prompts.py"))
REFUSE = "Insufficient information to answer the question"
OPTION_SETS = {"released": ROOT / "build" / "bixbench_numeric_q.jsonl",
               "placebo": ROOT / "build" / "bixbench_numeric_placebo.jsonl",
               "repaired": ROOT / "build" / "bixbench_numeric_repaired.jsonl"}
SHUFFLES = 2
# The refusal-option reading is taken on the released set and the repair, which
# are the two the paired contrast needs; the placebo is read forced only.
REFUSAL_ARMS = ("released", "repaired")


# ---------------------------------------------------------------- upstream pieces

def questions_to_mcq(question, options, refusal_option, rng):
    """``postprocessing_utils.questions_to_mcq`` with the shuffle's generator passed in."""
    options = list(options)
    correct_answer = options[0]
    if refusal_option:
        options.append(REFUSE)
    rng.shuffle(options)
    correct_letter = chr(65 + options.index(correct_answer))
    refusal_letter = chr(65 + options.index(REFUSE)) if refusal_option else None
    formatted = f"{question}\n"
    for j, opt in enumerate(options):
        formatted += f"{chr(65 + j)}. {opt}\n"
    return formatted, correct_letter, refusal_letter, options


def xml_extract(text):
    """``postprocessing_utils.xml_extract``: the letter, or 'Z' for no pick."""
    match = re.search(r"<answer>([A-Z])</answer>", text or "")
    return match[1] if match else "Z"


def parse_grade(response):
    """``GradingFunction._parse_grade_response``: anything but 'correct' is incorrect."""
    match = re.search(r"<grade>\s*(.*?)\s*</grade>", response or "", re.DOTALL)
    return bool(match) and match[1].strip().lower() == "correct"


def notebook_markdown(cells, budget):
    """The notebook the reader is shown: every cell and its output, oldest outputs
    shortened first when the whole would not fit the reader's context."""
    limits = [len(c["output"]) for c in cells]

    def render():
        parts = []
        for i, (cell, lim) in enumerate(zip(cells, limits)):
            out = cell["output"]
            if len(out) > lim:
                keep = max(lim // 2, 100)
                out = out[:keep] + f"\n... [{len(out) - 2 * keep} characters not shown] ...\n" + out[-keep:]
            parts.append(f"### Cell {i}:\n```python\n{cell['source']}\n```\n### Output {i}:\n```\n{out}\n```")
        return "\n".join(parts)

    text = render()
    for short in (2000, 800, 300):
        if len(text) <= budget:
            break
        for i in range(len(cells)):
            limits[i] = min(limits[i], short)
            text = render()
            if len(text) <= budget:
                break
    if len(text) > budget:                    # still too long: keep the latest cells
        text = text[-budget:]
    return text


# ---------------------------------------------------------------- LLM calls, cached

# Per-model request settings (``--extra-body name=JSON``): GLM-4.5-Air reads and
# grades with its thinking off, as it ran as an agent. A model given none keeps the
# cache keys its calls were made under, so every shipped call is still found.
EXTRA_BODY = {}


def cache_key(model, max_tokens, prompt):
    extra = EXTRA_BODY.get(model)
    tail = f"\x00{json.dumps(extra, sort_keys=True)}" if extra else ""
    return hashlib.sha256(f"{model}\x00{max_tokens}\x00{prompt}{tail}".encode()).hexdigest()


class Client:
    def __init__(self, cache_path: Path, concurrency=48):
        self.cache_path = cache_path
        self.cache = {}
        shipped = SHIPPED / (cache_path.name + ".gz")
        for source in (shipped, cache_path):          # the shipped copy, then any newer calls
            if source.exists():
                opener = gzip.open if source.suffix == ".gz" else open
                with opener(source, "rt") as fh:
                    for line in fh:
                        rec = json.loads(line)
                        self.cache[rec["key"]] = rec["reply"]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = cache_path.open("a")
        self.sem = asyncio.Semaphore(concurrency)
        # kept under vLLM's 5 s keep-alive, so a pooled connection is never reused after
        # the server has dropped it (bixbench_agent.py hit that as "server disconnected")
        self.http = httpx.AsyncClient(timeout=1800, limits=httpx.Limits(max_connections=concurrency * 2,
                                                                        keepalive_expiry=2.0))
        self.calls = Counter()
        # Identical prompts in flight at once share one request. Two trajectories
        # can give the reader the same prompt (an empty notebook and the same
        # answer), and greedy decoding in different batches returned different
        # replies to 37 of Llama's 57 such pairs, six of them different letters;
        # a rerun from the cache then gave both the one reply kept, and the
        # numbers moved by one read.
        self.pending = {}

    async def ask(self, model, base, prompt, max_tokens):
        key = cache_key(model, max_tokens, prompt)
        if key in self.cache:
            self.calls["cached"] += 1
            return self.cache[key]
        if key in self.pending:
            self.calls["shared"] += 1
            return await asyncio.shield(self.pending[key])
        self.pending[key] = asyncio.get_running_loop().create_future()
        try:
            reply = await self._call(key, model, base, prompt, max_tokens)
        except BaseException as err:
            self.pending.pop(key).set_exception(err)
            raise
        self.pending.pop(key).set_result(reply)
        return reply

    async def _call(self, key, model, base, prompt, max_tokens):
        bases = base.split(",")                   # replicas of one server config
        base = bases[int(key[:8], 16) % len(bases)]
        async with self.sem:
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0, **EXTRA_BODY.get(model, {})}
            if oa.is_openai(base):
                # OpenAI's API or an Azure resource (openai_api.py): its key, retries and ledger. A
                # prompt past the context comes back as the same "[error 400" reply a vLLM server's
                # does, so the notebook is shown shorter; a read Azure's content filter refuses, or
                # cuts to nothing, is kept as "[content filter]", which reads as no pick; any other
                # refusal stops the run instead of being cached.
                try:
                    data = await oa.post_chat(self.http, base, payload, self.cache_path.stem)
                    choice = data["choices"][0]
                    reply = choice["message"].get("content") or ""
                    if choice.get("finish_reason") == "content_filter" and not reply:
                        reply = oa.FILTERED_REPLY
                except oa.ContentFiltered:
                    reply = oa.FILTERED_REPLY
                except oa.ContextLength as err:
                    reply = f"[error 400: {err.text[:200]}]"
                if reply == oa.FILTERED_REPLY:
                    self.calls["content_filter"] += 1
                self.cache[key] = reply
                self.handle.write(json.dumps({"key": key, "model": model, "reply": reply}) + "\n")
                self.handle.flush()
                self.calls["new"] += 1
                return reply
            for attempt in range(8):
                try:
                    r = await self.http.post(f"{base}/v1/chat/completions", json=payload)
                    if r.status_code == 400:
                        reply = f"[error 400: {r.text[:200]}]"
                        break
                    r.raise_for_status()
                    reply = r.json()["choices"][0]["message"].get("content") or ""
                    break
                except (httpx.TransportError, httpx.HTTPStatusError, json.JSONDecodeError) as err:
                    if attempt == 7:
                        raise
                    await asyncio.sleep(5 * (attempt + 1))
        self.cache[key] = reply
        self.handle.write(json.dumps({"key": key, "model": model, "reply": reply}) + "\n")
        self.handle.flush()
        self.calls["new"] += 1
        return reply


# ---------------------------------------------------------------- grading

def strict_numeric(answer, ideal):
    """The answer's first number, rounded to the precision the key is written at, equals the key."""
    key = parse_number(ideal)
    if key is None or not answer:
        return None
    found = re.findall(r"[-−]?\d[\d,]*\.?\d*(?:[eE][-+−]?\d+)?%?", str(answer))
    if not found:
        return False
    value = parse_number(found[0])
    if value is None:
        return False
    text = str(ideal).strip().rstrip("%").replace(",", "")
    if re.search(r"[eE]", text):
        mantissa = text.lower().split("e")[0]
        digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
        return f"{value:.{digits}e}" == f"{key:.{digits}e}"
    digits = len(text.split(".")[1]) if "." in text else 0
    return round(value, digits) == round(key, digits)


async def grade_open(traj, item, client, judge):
    """BixBench v1.5's ``OpenEndedGrader.grade`` with partial and LLM matching on."""
    answer = traj["answer"]
    if not answer:
        return False, "no answer"
    mode = item["eval_mode"]
    target, question = item["ideal"], item["question"]
    name, base = judge
    if mode == "str_verifier":
        clean_t = re.sub(r"[^a-zA-Z0-9]", "", target).lower()
        clean_p = re.sub(r"[^a-zA-Z0-9]", "", answer).lower()
        if clean_p == clean_t:
            return True, "exact"
        if clean_p and clean_p in clean_t:
            return True, "partial"
        template = PROMPTS["OPEN_ENDED_GRADING_PROMPT"]
    elif mode == "range_verifier":
        template = PROMPTS["OPEN_ENDED_RANGE_GRADING_PROMPT"]
    else:
        template = PROMPTS["OPEN_ENDED_GRADING_PROMPT"]
    prompt = template.format(question=question, target=target, predicted=answer)
    reply = await client.ask(name, base, prompt, 512)
    return parse_grade(reply), "llm"


# Tokens a reader may spend on one MCQ_EVAL_PROMPT reply. BixBench's call sets none
# (``litellm.acompletion`` with the model's default); 1,536 cut off fewer than one reply in
# nine for the three families read first, but 44% of GLM-4.5-Air's, which re-derives the
# notebook's analysis before it answers -- so a reader may be given more (``--read-tokens``;
# at 4,096 GLM-4.5-Air is cut off on 1.4% of its own runs' reads). Qwen3-30B-A3B does the
# same and is not helped: cut off on 45% of its reads at 1,536 and still on 36% at 4,096
# (PRIMARY_READER).
READ_TOKENS = {}


async def read_mcq(traj, item, options, refusal, shuffle, client, reader, budget):
    """One ``MCQ_EVAL_PROMPT`` read: the reader's letter mapped back to an option."""
    name, base = reader
    seed = int(hashlib.sha256(f"{item['question_id']}|{shuffle}|{refusal}".encode()).hexdigest()[:8], 16)
    formatted, correct, refusal_letter, shown = questions_to_mcq(
        item["question"], options, refusal, random.Random(seed))
    for attempt in range(3):
        prompt = (PROMPTS["MCQ_EVAL_PROMPT"].replace("{{notebook}}", notebook_markdown(traj["cells"], budget))
                  .replace("{{question}}", formatted)
                  .replace("{{proposed_answer}}", str(traj["answer"])))
        reply = await client.ask(name, base, prompt, READ_TOKENS.get(name, 1536))
        # A notebook too long for the reader's context is shown shorter, not scored as no pick.
        if not reply.startswith("[error 400"):
            break
        budget //= 2
    letter = xml_extract(reply)
    picked = shown[ord(letter) - 65] if letter != "Z" and ord(letter) - 65 < len(shown) else None
    return {"correct": letter == correct, "refused": refusal_letter is not None and letter == refusal_letter,
            "no_pick": letter == "Z", "picked": picked, "server_error": reply.startswith("[error")}


def nearest_is_key(answer, options):
    """1.0 if the option nearest the submitted number is the key (options[0]), else 0.0;
    0.0 for no answer or no number. Nearness is |a - v| / (|a| + |v|), which treats a
    factor of two alike at every scale BixBench's keys span: for values of one sign it is
    tanh(|log a - log v| / 2), so nearest means nearest on a log scale, and options are compared
    on that scale (``answer_numbers.nearest_ranks``): the ratio itself rounds to exactly 1.0 in
    floating point once an answer is about 10^16 times an option, which tied a miss far beyond an
    extreme key four ways when every one of those misses goes to the key.

    Ties are split evenly, the expected score of a tie broken at random. An answer of 0,
    or one of the other sign from every option, is at distance 1 from all four; breaking
    that tie by position would credit the key, which sits first.
    """
    a = parse_number(answer) if answer else None
    values = [parse_number(o) for o in options]
    if a is None or any(v is None for v in values):
        return 0.0
    tied = nearest_ranks(a, values)
    return 1.0 / len(tied) if 0 in tied else 0.0


_OPTION_SETS = {}


def option_sets():
    """Each numeric item's four values in each option set, key first (load_items)."""
    if not _OPTION_SETS:
        _OPTION_SETS.update(load_items()[1])
    return _OPTION_SETS


def pick_rank(picked, options):
    """Rank of the picked option among the four numeric values, or None."""
    values = [parse_number(o) for o in options]
    if picked is None or picked not in options or any(v is None for v in values):
        return None
    order = sorted(range(len(options)), key=lambda i: values[i])
    return order.index(options.index(picked))


# ---------------------------------------------------------------- inference

# BixBench's capsules need a nominal 96.0% to cover 95% (Appendix~\ref{app:stats});
# every interval here is read there, as the paper's other BixBench intervals are.
LEVEL = 0.96


def cluster_interval(per_item, clusters, reps=10000, seed=20260923, level=None):
    level = LEVEL if level is None else level
    """Mean over items, and a cluster bootstrap interval over capsules, in points."""
    groups = defaultdict(list)
    for value, c in zip(per_item, clusters):
        if value is not None:
            groups[c].append(value)
    keys = sorted(groups)
    if not keys:
        return None
    sums = np.array([sum(groups[k]) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(reps, len(keys)))
    draws = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    point = sums.sum() / counts.sum()
    # full precision: a value stored at two decimals and printed at one is rounded twice (4.054 -> 4.05 -> 4.0)
    return {"mean": round(float(100 * point), 6), "lo": round(float(100 * lo), 6), "hi": round(float(100 * hi), 6),
            "n_items": int(counts.sum()), "n_clusters": len(keys)}


# ---------------------------------------------------------------- main

def load_items():
    items = {}
    for line in open(ROOT / "data" / "bixbench.jsonl"):
        row = json.loads(line)
        items[row["question_id"]] = row
    by_text = {(r["question"], r["ideal"]): q for q, r in items.items()}
    numeric = {}
    released = [json.loads(l) for l in open(OPTION_SETS["released"])]
    arms = {arm: [json.loads(l) for l in open(path)] for arm, path in OPTION_SETS.items()}
    for i, row in enumerate(released):
        qid = by_text[(row["question"], row["ideal"])]
        numeric[qid] = {arm: [arms[arm][i]["ideal"], *arms[arm][i]["distractors"]] for arm in arms}
        assert all(numeric[qid][arm][0] == row["ideal"] for arm in arms), qid
    return items, numeric


async def score(args):
    # the deleted-option arm's grader, so a with-data answer is held to the tolerance a no-data one was
    from answer_numbers import graded as within_tolerance
    items, numeric = load_items()
    readers = dict(r.split("=", 1) for r in args.reader)
    judge = tuple(args.judge.split("=", 1))
    budgets = dict(b.split("=", 1) for b in args.budget) if args.budget else {}
    client = Client(args.cache, args.concurrency)
    trajs = [t for t in load_trajectories(args.runs) if not args.models or run_key(t) in args.models]
    print(f"{len(trajs)} trajectories", flush=True)

    rows = []

    async def one(t):
        item = items[t["question_id"]]
        ok, how = await grade_open(t, item, client, judge)
        row = {"model": run_key(t), "condition": t["condition"], "question_id": t["question_id"],
               "rollout": t["rollout"], "capsule": item["capsule_uuid"], "answer": t["answer"],
               "termination": t["termination"], "steps": t["steps"], "open": ok, "open_how": how,
               "strict": strict_numeric(t["answer"], item["ideal"]) if t["question_id"] in numeric else None,
               "within_5pct": (bool(within_tolerance(t["answer"], item["ideal"], 0.05)) if t["answer"] else False)
               if t["question_id"] in numeric else None,
               "numeric": t["question_id"] in numeric, "reads": {},
               # where the key sits among the four values in each option set, for the rank model
               "key_rank": {arm: pick_rank(options[0], options) for arm, options in numeric[t["question_id"]].items()}
               if t["question_id"] in numeric else {}}
        if t["answer"]:                           # upstream drops unanswered runs before reading
            option_sets = {"released": [item["ideal"], *item["distractors"]]}
            if t["question_id"] in numeric:
                option_sets.update({a: o for a, o in numeric[t["question_id"]].items() if a != "released"})
            for rname, base in readers.items():
                budget = int(budgets.get(rname, args.default_budget))
                for arm, options in option_sets.items():
                    for refusal in ((False, True) if arm in REFUSAL_ARMS else (False,)):
                        reads = await asyncio.gather(*(
                            read_mcq(t, item, options, refusal, s, client, (rname, base), budget)
                            for s in range(SHUFFLES)))
                        for r in reads:
                            r["rank"] = pick_rank(r["picked"], options) if t["question_id"] in numeric else None
                        row["reads"][f"{rname}|{arm}|{'refusal' if refusal else 'forced'}"] = reads
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f"  {len(rows)}/{len(trajs)} scored; calls {dict(client.calls)}", flush=True)

    await asyncio.gather(*(one(t) for t in trajs))
    await client.http.aclose()
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    args.rows.write_text(json.dumps(sorted(rows, key=lambda r: (r["model"], r["condition"], r["question_id"],
                                                                r["rollout"])), indent=1))
    print(f"wrote {args.rows}; calls {dict(client.calls)}")
    if client.calls["content_filter"]:
        print(f"content filtered: {client.calls['content_filter']} reads")
    return rows


SHIPPED = ROOT / "results" / "agent_runs"


def run_key(t):
    """Which run a trajectory belongs to: its model, suffixed by any protocol but the
    first runs' -- ``bixbench_agent.run_name``'s rule -- so one model's runs under
    two protocols are never pooled or taken for duplicates of each other."""
    protocol = t.get("protocol", "text")
    return t["model"] if protocol == "text" else f"{t['model']}-{protocol}"


def load_trajectories(runs):
    """Every trajectory: from ``build/agent_runs`` where the runs were made, else
    from the gzipped copies the artifact ships, so the scores re-derive with no
    GPU and -- every call being cached -- no server either."""
    out, seen = [], set()
    for path in sorted(Path(runs).glob("*/*/*.json")):
        t = json.loads(path.read_text())
        out.append(t)
        seen.add((run_key(t), t["condition"], t["question_id"], t["rollout"]))
    for path in sorted(SHIPPED.glob("trajectories_*.jsonl.gz")):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                t = json.loads(line)
                if (run_key(t), t["condition"], t["question_id"], t["rollout"]) not in seen:
                    out.append(t)
    return out


def anonymous(t):
    """A trajectory as shipped. Its settings recorded the absolute paths it ran
    with, and a path names the machine's owner, which a double-blind bundle must
    not: paths inside the repository become relative, and any other (the scratch
    work directory) becomes its role."""
    settings = dict(t.get("settings") or {})
    for key, value in settings.items():
        if isinstance(value, str) and value.startswith("/"):
            try:
                settings[key] = str(Path(value).relative_to(ROOT))
            except ValueError:
                settings[key] = f"<{key}>"
        elif isinstance(value, str):
            settings[key] = oa.public_base(value)
    out = {**t, "settings": settings}
    if isinstance(out.get("served_from"), str):
        out["served_from"] = oa.public_base(out["served_from"])
    return out


def pack(runs):
    """Write the shipped copies: one gzipped JSON line per trajectory, per model,
    each model's reader cache beside it, and the scored rows the summary is made
    from (``tests/test_withdata.py`` re-derives ``results/bixbench_withdata.json``
    from those)."""
    SHIPPED.mkdir(parents=True, exist_ok=True)
    by_model = defaultdict(list)
    for t in load_trajectories(runs):
        by_model[run_key(t)].append(t)
    for model, ts in by_model.items():
        ts.sort(key=lambda t: (t["condition"], t["question_id"], t["rollout"]))
        with gzip.open(SHIPPED / f"trajectories_{model}.jsonl.gz", "wt", compresslevel=9) as fh:
            for t in ts:
                fh.write(json.dumps(anonymous(t)) + "\n")
        for name in (f"reader_cache_{model}.jsonl", f"withdata_rows_{model}.json"):
            if (Path(runs) / name).exists():
                with gzip.open(SHIPPED / f"{name}.gz", "wt", compresslevel=9) as fh:
                    fh.write((Path(runs) / name).read_text())
        print(f"packed {len(ts)} trajectories for {model}")
    # runs that were replaced by a complete re-run (reruns.json says which and why) ship too
    superseded = sorted(Path(runs).parent.glob(f"{Path(runs).name}_superseded/*/*/*.json"))
    by_model = defaultdict(list)
    for path in superseded:
        t = json.loads(path.read_text())
        by_model[run_key(t)].append(t)
    for model, ts in by_model.items():
        with gzip.open(SHIPPED / f"superseded_{model}.jsonl.gz", "wt", compresslevel=9) as fh:
            for t in sorted(ts, key=lambda t: (t["condition"], t["question_id"])):
                fh.write(json.dumps(anonymous(t)) + "\n")
        print(f"packed {len(ts)} superseded trajectories for {model}")


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def reread(rows):
    """The rows with the fields read from the answer alone -- the key-precision rule and the tolerance --
    recomputed by the current readers, rather than taken as stored when the run was graded."""
    from answer_numbers import graded
    ideal = {q: it["ideal"] for q, it in load_items()[0].items()}
    out = []
    for r in rows:
        if r.get("numeric") and r["question_id"] in ideal:
            q, answer = r["question_id"], r["answer"]
            r = dict(r, strict=strict_numeric(answer, ideal[q]),
                     within_5pct=bool(graded(answer, ideal[q], 0.05)) if answer else False)
        out.append(r)
    return out


def summarise(rows, args):
    out = {}
    rows = reread(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["condition"])].append(r)
    for (model, cond), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: (r["question_id"], r["rollout"]))
        # the readers that read *these* trajectories: a reader that read another
        # model's runs would otherwise score only the unanswered ones here, at 0%
        readers = sorted({k.split("|")[0] for r in rs for k in r["reads"]})
        clusters = [r["capsule"] for r in rs]
        block = {"n": len(rs), "submitted": sum(r["termination"] == "submitted" for r in rs),
                 "open": cluster_interval([float(r["open"]) for r in rs], clusters)}
        num = [r for r in rs if r["numeric"]]
        nclu = [r["capsule"] for r in num]
        block["numeric"] = {"n": len(num),
                            "open": cluster_interval([float(r["open"]) for r in num], nclu),
                            "strict": cluster_interval([float(bool(r["strict"])) for r in num], nclu),
                            "within_5pct": cluster_interval([float(bool(r["within_5pct"])) for r in num], nclu)}
        # A reader that reads no notebook: the option nearest the agent's own number. It
        # sets each option set against the answers the agent gave, with no model's
        # preferences in it; a run with no number scores 0 in every set, as an unread run does.
        sets = option_sets()
        near = {arm: [nearest_is_key(r["answer"], sets[r["question_id"]][arm]) if r["question_id"] in sets
                      else None for r in num]
                for arm in ("released", "placebo", "repaired")}
        block["nearest"] = {arm: cluster_interval(v, nclu) for arm, v in near.items()}
        block["nearest"]["released-repaired"] = cluster_interval(
            [None if a is None or b is None else a - b for a, b in zip(near["released"], near["repaired"])], nclu)
        for reader in readers:
            rb = {}
            for mode in ("forced", "refusal"):
                def score_of(r, arm):
                    reads = r["reads"].get(f"{reader}|{arm}|{mode}")
                    if reads is None:
                        return 0.0 if not r["answer"] else None      # unanswered: wrong, unread
                    return mean_or_none([float(x["correct"]) for x in reads])
                rb[mode] = {"all": cluster_interval([score_of(r, "released") for r in rs], clusters)}
                arm_names = ("released", "placebo", "repaired") if mode == "forced" else REFUSAL_ARMS
                arms = {arm: [score_of(r, arm) for r in num] for arm in arm_names}
                rb[mode]["numeric"] = {arm: cluster_interval(v, nclu) for arm, v in arms.items()}
                pairs = (("released", "repaired"), ("released", "placebo"), ("placebo", "repaired"))
                for a, b in (p for p in pairs if p[0] in arms and p[1] in arms):
                    diff = [None if x is None or y is None else x - y for x, y in zip(arms[a], arms[b])]
                    rb[mode]["numeric"][f"{a}-{b}"] = cluster_interval(diff, nclu)
                # where the agent's own answer is wrong: does the option set rescue it?
                wrong = [r for r in num if r["answer"] and not r["strict"]]
                wclu = [r["capsule"] for r in wrong]
                rb[mode]["rescue"] = {arm: cluster_interval([score_of(r, arm) for r in wrong], wclu)
                                      for arm in ("released", "placebo", "repaired")
                                      if mode == "forced" or arm in REFUSAL_ARMS}
                diff = [None if (x := score_of(r, "released")) is None or (y := score_of(r, "repaired")) is None
                        else x - y for r in wrong]
                rb[mode]["rescue"]["released-repaired"] = cluster_interval(diff, wclu)
                ranks = {}
                for arm in ("released", "repaired"):
                    c = Counter()
                    for r in wrong:
                        for x in r["reads"].get(f"{reader}|{arm}|{mode}", []):
                            c[x["rank"]] += 1
                    total = sum(v for k, v in c.items() if k is not None)
                    ranks[arm] = {str(k): round(100 * v / total, 1) for k, v in sorted(c.items(), key=str)
                                  if k is not None} if total else {}
                rb[mode]["rescue"]["pick_rank"] = ranks
                # What the rank channel alone would do: a reader that picks rank r with the
                # probability this reader picks it on the released file, whatever the file,
                # scores the mean of that probability at each item's key rank.
                if mode == "forced" and ranks["released"] and all(r.get("key_rank") for r in wrong):
                    b = ranks["released"]
                    predicted = {arm: mean_or_none([b.get(str(r["key_rank"][arm]), 0.0) for r in wrong])
                                 for arm in ("released", "placebo", "repaired")}
                    predicted["released-repaired"] = predicted["released"] - predicted["repaired"]
                    rb[mode]["rescue"]["rank_model"] = {k: round(v, 2) for k, v in predicted.items()}
                if mode == "refusal":
                    rb[mode]["refused"] = mean_or_none(
                        [mean_or_none([float(x["refused"]) for x in r["reads"].get(f"{reader}|released|refusal", [])])
                         for r in rs if r["answer"]])
                rb[mode]["no_pick"] = mean_or_none(
                    [mean_or_none([float(x["no_pick"]) for x in r["reads"].get(f"{reader}|released|{mode}", [])])
                     for r in rs if r["answer"]])
            block[f"reader:{reader}"] = rb
        out[f"{model}|{cond}"] = block

    # What the data itself is worth, paired by item over capsules: the scale the
    # option set's effect is read against.
    for model in sorted({m for m, _ in groups}):
        if not all((model, c) in groups for c in ("data", "nodata")):
            continue
        by = {c: {(r["question_id"], r["rollout"]): r for r in groups[(model, c)]}
              for c in ("data", "nodata")}
        keys = sorted(set(by["data"]) & set(by["nodata"]))
        clusters = [by["data"][k]["capsule"] for k in keys]
        num = [k for k in keys if by["data"][k]["numeric"]]
        nclu = [by["data"][k]["capsule"] for k in num]
        block = {"n": len(keys),
                 "open": cluster_interval([float(by["data"][k]["open"]) - float(by["nodata"][k]["open"])
                                           for k in keys], clusters),
                 "numeric_strict": cluster_interval([float(bool(by["data"][k]["strict"]))
                                                     - float(bool(by["nodata"][k]["strict"])) for k in num],
                                                    nclu),
                 "numeric_within_5pct": cluster_interval([float(bool(by["data"][k]["within_5pct"]))
                                                          - float(bool(by["nodata"][k]["within_5pct"]))
                                                          for k in num], nclu)}
        # only a reader that read both conditions: an unread condition would count only its unanswered runs
        both = set.intersection(*({x.split("|")[0] for r in by[c].values() for x in r["reads"]} for c in by))
        for reader in sorted(both):
            def released(r):
                reads = r["reads"].get(f"{reader}|released|forced")
                if reads is None:
                    return 0.0 if not r["answer"] else None
                return mean_or_none([float(x["correct"]) for x in reads])

            def paired(ks):
                return [None if (a := released(by["data"][k])) is None or (b := released(by["nodata"][k])) is None
                        else a - b for k in ks]
            block[f"reader:{reader}"] = {"forced_all": cluster_interval(paired(keys), clusters),
                                         "forced_numeric": cluster_interval(paired(num), nclu)}
        out[f"{model}|data-nodata"] = block

    # What the protocol is worth: one model's runs under the published protocol
    # against its runs with tools called in text, paired by item over capsules.
    # The runs are separate draws, so this is the protocol's effect plus draw noise,
    # which the interval carries.
    for model in sorted({m for m, _ in groups}):
        base_model, _, protocol = model.partition("-")
        if not protocol or base_model not in {m for m, _ in groups}:
            continue
        for cond in ("data", "nodata"):
            if (model, cond) not in groups or (base_model, cond) not in groups:
                continue
            new = {(r["question_id"], r["rollout"]): r for r in groups[(model, cond)]}
            old = {(r["question_id"], r["rollout"]): r for r in groups[(base_model, cond)]}
            keys = sorted(set(new) & set(old))
            clusters = [new[k]["capsule"] for k in keys]
            num = [k for k in keys if new[k]["numeric"]]
            nclu = [new[k]["capsule"] for k in num]
            block = {"n": len(keys),
                     "open": cluster_interval([float(new[k]["open"]) - float(old[k]["open"]) for k in keys],
                                              clusters),
                     "numeric_within_5pct": cluster_interval(
                         [float(bool(new[k]["within_5pct"])) - float(bool(old[k]["within_5pct"])) for k in num],
                         nclu)}
            readers = set.intersection(*({x.split("|")[0] for r in side.values() for x in r["reads"]}
                                         for side in (new, old)))
            for reader in sorted(readers):
                def forced(r, arm):
                    reads = r["reads"].get(f"{reader}|{arm}|forced")
                    if reads is None:
                        return 0.0 if not r["answer"] else None
                    return mean_or_none([float(x["correct"]) for x in reads])

                def effect(r):
                    a, b = forced(r, "released"), forced(r, "repaired")
                    return None if a is None or b is None else a - b

                def diff(f, ks):
                    return [None if (a := f(new[k])) is None or (b := f(old[k])) is None else a - b
                            for k in ks]
                block[f"reader:{reader}"] = {
                    "forced_all": cluster_interval(diff(lambda r: forced(r, "released"), keys), clusters),
                    "released-repaired": cluster_interval(diff(effect, num), nclu)}
            out[f"{model}|{cond}|vs-{base_model}"] = block
    return out


AGENTS = [("qwen72b", "Qwen2.5-72B"), ("llama70b", "Llama-3.3-70B"), ("gemma27b", "gemma-3-27b")]
# The runs under BixBench's published protocol: (run key, name, the family that reads them).
AGENTS_REACT = [("qwen72b-react", "Qwen2.5-72B", "qwen72b"), ("llama70b-react", "Llama-3.3-70B", "llama70b"),
                ("glm45air-react", "GLM-4.5-Air", "glm45air"), ("qwen3a3b-react", "Qwen3-30B-A3B", "qwen3a3b")]
# The reader a run set's row in Table~\ref{tab:withdata} is read by, where it is not the
# run set's own family: Qwen3-30B-A3B, reading, deliberates past any reply limit (READ_TOKENS)
# and gives no pick on half its reads, so gemma-3-27b, every run set's second reader,
# stands in; tab:readers, no longer in the paper, printed Qwen3-30B-A3B's own reading beside it.
PRIMARY_READER = {"qwen3a3b-react": "gemma27b"}
# ldp 0.26.0 puts each reasoning turn back as "Thought: ... Based on this reasoning, let's
# select the appropriate tool!\nAction: "; a model that copies it into its next reasoning
# turn and repeats it never writes the "Action:" that would stop the turn.
WRAPPER_TAIL = ". Based on this reasoning, let's select the appropriate tool!"


def reply_limit_census(trajectories):
    """Published-protocol runs: how many reasoning turns ran to the reply limit, how many
    of those repeat the wrapper's sentence (five times or more), and, for each way an
    episode ended, how many episodes and how many of them had such a turn."""
    turns = capped = loops = 0
    endings = defaultdict(lambda: [0, 0])
    for t in trajectories:
        steps = [rec for rec in t["log"] if rec.get("finish")]
        cut = [rec for rec in steps if rec["finish"][0] == "length"]
        turns, capped = turns + len(steps), capped + len(cut)
        loops += sum((rec.get("reasoning") or "").count(WRAPPER_TAIL) >= 5 for rec in cut)
        endings[t["termination"]][0] += 1
        endings[t["termination"]][1] += bool(cut)
    return {"turns": turns, "capped": capped, "loops": loops,
            "endings": {k: {"episodes": v[0], "with_capped_turn": v[1]} for k, v in sorted(endings.items())}}


def reader_rows(summary, second="gemma27b"):
    """tab:readers' rows (no longer in the paper): each run set's released - repaired contrast with the
    data, read by its own family, by the second reader, and by the option nearest the
    agent's number. ``validate_artifact.py`` checks the table against these."""
    def cell(c):
        return f"${c['mean']:+.1f}$ $[{c['lo']:+.1f},{c['hi']:+.1f}]$" if c else "---"
    rows = []
    for tag, name, reader, proto in ([(t, n, t, "text") for t, n in AGENTS]
                                     + [(t, n, r, "published") for t, n, r in AGENTS_REACT]):
        block = summary.get(f"{tag}|data")
        if block is None or f"reader:{reader}" not in block:
            continue
        own = block[f"reader:{reader}"]["forced"]["numeric"]["released-repaired"]
        other = block.get(f"reader:{second}") if reader != second else None
        other = other["forced"]["numeric"]["released-repaired"] if other else None
        name = name + "$^\\dagger$" if tag in PRIMARY_READER else name
        rows.append(f"{proto} & {name} & {cell(own)} & {cell(other)} & "
                    f"{cell(block['nearest']['released-repaired'])}\\\\")
    return rows


def table_rows(summary):
    """Table~\\ref{tab:withdata}'s rows as the paper prints them: each agent read by
    its own family (or its PRIMARY_READER, marked), forced, and the same runs through
    three option sets -- the text-protocol runs first, then any run under the published
    protocol. ``validate_artifact.py`` checks the table against these."""
    rows = []
    for tag, name, reader in [(t, n, t) for t, n in AGENTS] + AGENTS_REACT:
        if tag in PRIMARY_READER:
            reader, name = PRIMARY_READER[tag], name + "$^\\dagger$"
        for cond, label in (("data", "with"), ("nodata", "without")):
            block = summary.get(f"{tag}|{cond}")
            if block is None or f"reader:{reader}" not in block:
                continue
            forced = block[f"reader:{reader}"]["forced"]
            refusal = block[f"reader:{reader}"]["refusal"]
            arms, diff = forced["numeric"], forced["numeric"]["released-repaired"]
            rows.append(f"{name if cond == 'data' else ''} & {label} & ${block['open']['mean']:.1f}$ & "
                        f"${block['numeric']['within_5pct']['mean']:.1f}$ & ${forced['all']['mean']:.1f}$ & "
                        f"${refusal['all']['mean']:.1f}$ $({100 * refusal['refused']:.0f})$ & "
                        f"${arms['released']['mean']:.1f}$ & ${arms['placebo']['mean']:.1f}$ & "
                        f"${arms['repaired']['mean']:.1f}$ & "
                        f"${diff['mean']:+.1f}$ $[{diff['lo']:+.1f},{diff['hi']:+.1f}]$\\\\")
    return rows


def main():
    global LEVEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "build" / "agent_runs")
    ap.add_argument("--models", action="append")
    ap.add_argument("--reader", action="append", default=[], help="name=base URL (repeatable)")
    ap.add_argument("--judge", help="name=base URL for the open-ended grader (needed to score)")
    ap.add_argument("--budget", action="append", help="name=chars of notebook a reader is shown")
    ap.add_argument("--extra-body", action="append", default=[],
                    help="name=JSON merged into that model's requests (and into its cache keys)")
    ap.add_argument("--read-tokens", action="append", default=[],
                    help="name=N: tokens that reader may spend on one MCQ reply (default 1536)")
    ap.add_argument("--default-budget", type=int, default=60000)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--cache", type=Path, default=ROOT / "build" / "agent_runs" / "reader_cache.jsonl")
    ap.add_argument("--rows", type=Path, action="append",
                    help="where scored rows go (default build/agent_runs/withdata_rows.json); "
                         "with --summarise-only, every file given is read and merged")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "bixbench_withdata.json")
    ap.add_argument("--level", type=float, default=LEVEL,
                    help="nominal level every interval is read at (the covering one on BixBench)")
    ap.add_argument("--summarise-only", action="store_true")
    ap.add_argument("--pack", action="store_true",
                    help="write results/agent_runs/*.jsonl.gz from build/agent_runs and stop")
    ap.add_argument("--latex", action="store_true",
                    help="print the rows of Table~\\ref{tab:withdata}, then tab:readers' (no longer in the paper), from --out and stop")
    args = ap.parse_args()
    if args.pack:
        pack(args.runs)
        return
    if args.latex:
        summary = json.loads(args.out.read_text())
        print("\n".join(table_rows(summary)), "", "\n".join(reader_rows(summary)), sep="\n")
        return
    LEVEL = args.level
    for spec in args.extra_body:
        name, _, body = spec.partition("=")
        EXTRA_BODY[name] = json.loads(body)
    for spec in args.read_tokens:
        name, _, n = spec.partition("=")
        READ_TOKENS[name] = int(n)
    args.rows = args.rows or [ROOT / "build" / "agent_runs" / "withdata_rows.json"]
    if args.summarise_only:                   # the shipped rows are gzipped; either kind reads
        rows = [r for path in args.rows for r in json.loads(
            gzip.open(path, "rt").read() if path.suffix == ".gz" else path.read_text())]
    else:
        if not args.judge:
            ap.error("--judge is needed to score")
        args.rows = args.rows[0]
        rows = asyncio.run(score(args))
    summary = summarise(rows, args)
    args.out.write_text(json.dumps(summary, indent=1))
    print(f"wrote {args.out}")
    for key, block in summary.items():
        if "|vs-" in key:
            print(f"\n{key}: n={block['n']} open={block['open']}")
            for k, v in block.items():
                if k.startswith("reader:"):
                    print(f"   {k} forced all={v['forced_all']} released-repaired={v['released-repaired']}")
            continue
        if key.endswith("|data-nodata"):
            print(f"\n{key}: open={block['open']} numeric strict={block['numeric_strict']}")
            for k, v in block.items():
                if k.startswith("reader:"):
                    print(f"   {k} forced all={v['forced_all']} numeric={v['forced_numeric']}")
            continue
        print(f"\n{key}: n={block['n']} submitted={block['submitted']} open={block['open']}")
        print(f"   numeric strict={block['numeric']['strict']}")
        for k, rb in block.items():
            if k.startswith("reader:"):
                f = rb["forced"]
                print(f"   {k} forced all={f['all']}")
                for arm in ("released", "placebo", "repaired", "released-repaired"):
                    print(f"      numeric {arm:18s} {f['numeric'][arm]}")
                print(f"      rescue released={f['rescue']['released']} repaired={f['rescue']['repaired']} "
                      f"diff={f['rescue']['released-repaired']}")


if __name__ == "__main__":
    main()
