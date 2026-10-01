#!/usr/bin/env python3
r"""An OpenAI model on BixBench's numeric items without the data, forced to choose.

The open models' no-data forced choice is run here on a model served only through OpenAI's
API, with the open models' own prompts, imported rather than restated:

* **question shown**: ``free_response.py``'s forced arm -- its system prompt, its
  ``Question: ... Options: ...`` message, two letter orderings per item drawn as it draws them,
  greedy, 160 reply tokens, the letter read by ``agentic_probe.final_letter`` -- whose dumps
  ``rank_attribution.py`` reads for the rewritten-options table;
* **template**: BixBench's own no-data template with the question shown, the prompt behind its
  published forced-choice runs, three orderings per item, read as the withheld condition is;
* **question withheld**: ``agentic_probe.py``'s BixBench template with ``[withheld]`` in place
  of the question, three orderings per item, 320 reply tokens. The open models' letter was the
  arg-max after a pre-filled ``<answer>``; an API cannot pre-fill, so the letter is read from
  the generated reply as ``agentic_probe.py``'s generated read-out reads it, and the top
  log-probabilities at the letter are kept, from which an arg-max read-out follows.

Each condition runs through the three option sets with the same key -- released (R), the
rank-preserving rewrite (P) and the rank-uniform rewrite (U) -- on v1.5's 105 numeric items and
on the 158 of v1.0's 159 that have all three, with the same letter orderings across the three
sets, so R - P and R - U pair ordering by ordering. Dumps and a summary go to
``build/openai/nodata/``, in the formats ``rank_attribution.py`` reads; every reply is cached,
so a rerun calls nothing it has called before. Calls go through ``openai_api.py`` (its key,
retries, ledger and budget).

    python3 openai_nodata.py --model gpt-4o --dry-run
    AGENTICLS_OPENAI_BUDGET_USD=20 python3 openai_nodata.py --model gpt-4o
    python3 openai_nodata.py --model gpt-4o --summarise
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import pathlib

import agentic_probe as ap           # sets USE_TF=0 before free_response imports transformers
import free_response as fr
import openai_api as oa
import rank_attribution as ra
import replication as rp

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "build" / "openai" / "nodata"
SHIPPED = ROOT / "results" / "openai_nodata"
ARMS = (("released", ""), ("placebo", "_placebo"), ("repaired", "_repaired"))
V15 = {"released": ROOT / "build" / "bixbench_numeric_q.jsonl",
       "placebo": ROOT / "build" / "bixbench_numeric_placebo.jsonl",
       "repaired": ROOT / "build" / "bixbench_numeric_repaired.jsonl"}
RELEASES = {"v15": "bixnum", "v10": "bixv10num"}       # the dumps' file tags
SHOWN_DRAWS, WITHHELD_DRAWS = 2, 3                      # free_response.py's and the withheld runs' --draws
SHOWN_TOKENS, WITHHELD_TOKENS = 160, 320                # their --max-new-tokens
SEED = 0                                                # both scripts' --seed
TOP_LOGPROBS = 20


def release_rows(release):
    """{arm: rows} for one release, the three arms aligned item by item and keyed alike."""
    if release == "v15":
        return {arm: [json.loads(l) for l in open(path)] for arm, path in V15.items()}
    sets = rp.v10_sets()
    released = [json.loads(l) for l in open(rp.NUMERIC)]
    keep = [r for r in released if all(arm in sets[r["question_id"]] for arm, _ in ARMS)]
    return {arm: [{"question_id": r["question_id"], "question": r["question"], "cluster": r["capsule_uuid"],
                   "ideal": sets[r["question_id"]][arm][0], "distractors": sets[r["question_id"]][arm][1:]}
                  for r in keep] for arm, _ in ARMS}


def rollouts(rows, condition):
    if condition == "shown":
        # free_response.main keeps the items whose key reads as a number before it builds
        rows = [r for r in rows if fr.as_number(r["ideal"]) is not None]
        out = fr.build(rows, "mcq", SHOWN_DRAWS, SEED)
        for r in out:
            r["messages"] = [{"role": "system", "content": fr.MCQ_SYSTEM}, {"role": "user", "content": r["user"]}]
        return out
    mode = "question" if condition == "template" else "withheld"
    out = ap.build_rollouts(rows, mode, WITHHELD_DRAWS, SEED, prompt="bixbench")
    for r in out:
        r["messages"] = [{"role": "user", "content": r["user"]}]      # the BixBench template has no system turn
    return out


def dump_path(model, release, suffix, condition):
    safe = model.replace("/", "_")
    tag = RELEASES[release]
    if condition == "shown":
        return OUT / f"free_{safe}_{tag}{suffix}.jsonl"
    if condition == "template":
        return OUT / f"{safe}_{tag}{suffix}_bixprompt_question_generated.jsonl"
    return OUT / f"{safe}_{tag}{suffix}_bixprompt_generated.jsonl"


def request(model, r, condition):
    payload = {"model": model, "messages": r["messages"], "temperature": 0.0,
               "max_tokens": SHOWN_TOKENS if condition == "shown" else WITHHELD_TOKENS}
    if condition in ("withheld", "template") and not oa.is_reasoning(model):
        payload.update(logprobs=True, top_logprobs=TOP_LOGPROBS)
    return payload


def cache_key(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def letter_logprobs(logprobs, k):
    """The log-probability of each option letter where the reply writes its letter after
    ``<answer>``: {letter: logprob} over the letters in the top alternatives there."""
    text = ""
    for position in (logprobs or {}).get("content") or []:
        token = position.get("token", "")
        if "<answer>" in text.lower() and token.strip():
            found = {}
            for alt in position.get("top_logprobs") or []:
                letter = alt.get("token", "").strip().upper()
                if len(letter) == 1 and letter in ap.ALPHABET[:k] and letter not in found:
                    found[letter] = alt.get("logprob")
            return found
        text += token
    return {}


def read_shown(r, reply):
    answer = ap.final_letter(reply, r["k"])
    return {"item": r["item"], "draw": r["draw"], "k": r["k"], "cluster": r["cluster"], "gold": r["gold"],
            "arm": "mcq", "user": r["user"], "reply": reply, "answer": answer,
            "correct": 1.0 if answer == r["gold"] else 0.0, "chance": 1.0 / r["k"]}


def read_withheld(r, reply, letters):
    # agentic_probe.play's read-out of the BixBench template: the tagged letter, else a FINAL line
    tag = ap.ANSWER_TAG.search(reply)
    letter = tag.group(1).upper() if tag else None
    answer = letter if letter and letter in ap.ALPHABET[:r["k"]] else (ap.final_letter(reply, r["k"]) or "")
    return {"arm": "file", "item": r["item"], "draw": r["draw"], "cluster": r["cluster"], "k": r["k"],
            "gold": r["gold"], "answer": answer, "turns": 1, "used_tool": False, "user": r["user"],
            "reply": [reply], "letter_logprobs": letters,
            "argmax_answer": max(letters, key=letters.get) if letters else None}


def plan(model, releases, conditions, limit):
    """[(release, arm, suffix, condition, rollouts)]"""
    out = []
    for release in releases:
        rows = release_rows(release)
        for arm, suffix in ARMS:
            for condition in conditions:
                built = rollouts(rows[arm][:limit] if limit else rows[arm], condition)
                out.append((release, arm, suffix, condition, built))
    return out


def load_cache(path):
    cache = {}
    if path.exists():
        for line in path.open():
            rec = json.loads(line)
            cache[rec["key"]] = rec
    return cache


async def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / f"cache_{args.model.replace('/', '_')}.jsonl"
    cache = load_cache(cache_path)
    handle = cache_path.open("a")
    sem = asyncio.Semaphore(args.concurrency)
    filtered = []
    import httpx
    async with httpx.AsyncClient(timeout=600) as http:
        async def post(payload):
            try:
                return await oa.post_chat(http, args.base, payload, f"nodata:{args.model}")
            except oa.ContentFiltered:
                return None
            except oa.Rejected as err:
                # Azure may cap top_logprobs below OpenAI's 20: ask again, once, at 5
                if err.status == 400 and "top_logprobs" in err.text and payload.get("top_logprobs", 0) > 5:
                    return await post({**payload, "top_logprobs": 5})
                raise

        async def call(payload):
            key = cache_key(payload)
            if key in cache:
                rec = cache[key]
            else:
                async with sem:
                    data = await post(payload)
                if data is None:            # Azure's content filter: kept as an unparsed reply
                    rec = {"key": key, "reply": "", "logprobs": None, "model": None, "content_filter": True}
                else:
                    choice = data["choices"][0]
                    rec = {"key": key, "reply": choice["message"].get("content") or "",
                           "logprobs": choice.get("logprobs"), "model": data.get("model"),
                           **({"content_filter": True} if choice.get("finish_reason") == "content_filter" else {})}
                cache[key] = rec
                handle.write(json.dumps(rec) + "\n")
                handle.flush()
            if rec.get("content_filter"):
                filtered.append(key)
            return rec

        for release, arm, suffix, condition, built in plan(args.model, args.releases, args.conditions, args.limit):
            recs = await asyncio.gather(*(call(request(args.model, r, condition)) for r in built))
            rows = [read_shown(r, rec["reply"]) if condition == "shown"
                    else read_withheld(r, rec["reply"], letter_logprobs(rec.get("logprobs"), r["k"]))
                    for r, rec in zip(built, recs)]
            path = dump_path(args.model, release, suffix, condition)
            with path.open("w") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            print(f"  {release} {arm:8s} {condition:8s} {len(rows)} rollouts -> {path.relative_to(ROOT)}", flush=True)
    handle.close()
    print(f"content filtered: {len(filtered)} calls, read as unparsed", flush=True)


def summarise(args):
    """Margins per release, condition and option set, and R - P, R - U paired ordering by ordering."""
    report = {"model": args.model, "releases": {}}
    # the versions the API reported serving, from every cached reply
    cache = load_cache(OUT / f"cache_{args.model.replace('/', '_')}.jsonl")
    report["served"] = sorted({rec["model"] for rec in cache.values() if rec.get("model")})
    for release in args.releases:
        entry = {}
        for condition in args.conditions:
            arm_name = "mcq" if condition == "shown" else "file"
            got = {}
            for arm, suffix in ARMS:
                path = dump_path(args.model, release, suffix, condition)
                if path.exists():
                    got[arm] = ra.dump_rows(path, arm_name)
            if "released" not in got:
                continue
            cell = {arm: ra.summarise_dump(rows) for arm, rows in got.items()}
            for arm in ("placebo", "repaired"):
                if arm in got:
                    cell[f"released-{arm}"] = ra.paired(got["released"], got[arm])
            if condition in ("withheld", "template"):
                # the arg-max read-out, from the log-probabilities at the letter
                arg = {}
                for arm, suffix in ARMS:
                    path = dump_path(args.model, release, suffix, condition)
                    if path.exists():
                        tmp = path.with_name(path.name.replace("_generated", "_logprob"))
                        with tmp.open("w") as fh:
                            for line in path.open():
                                row = json.loads(line)
                                row["answer"] = row.get("argmax_answer")
                                fh.write(json.dumps(row) + "\n")
                        arg[arm] = ra.dump_rows(tmp, arm_name)
                cell["argmax"] = {arm: ra.summarise_dump(rows) for arm, rows in arg.items()}
            entry[condition] = cell
        report["releases"][release] = entry
    path = OUT / f"summary_{args.model.replace('/', '_')}.json"
    path.write_text(json.dumps(report, indent=1) + "\n")
    # the shipped copies: the summary, and every dump it was read from
    safe = args.model.replace("/", "_")
    (ROOT / "results" / f"openai_nodata_{safe}.json").write_text(json.dumps(report, indent=1) + "\n")
    SHIPPED.mkdir(parents=True, exist_ok=True)
    for release in args.releases:
        for condition in args.conditions:
            for arm, suffix in ARMS:
                src = dump_path(args.model, release, suffix, condition)
                if src.exists():
                    with gzip.GzipFile(filename="", mode="wb", mtime=0,
                                       fileobj=(SHIPPED / (src.name + ".gz")).open("wb")) as fh:
                        fh.write(src.read_bytes())
    for release, entry in report["releases"].items():
        for condition, cell in entry.items():
            line = "  ".join(f"{arm} {cell[arm]['margin']['mean']:+5.1f} (unparsed {cell[arm]['unparsed']:.0f}%)"
                             for arm, _ in ARMS if arm in cell)
            diffs = "  ".join(f"{k} {cell[k]['mean']:+5.1f} [{cell[k]['lo']:+.1f},{cell[k]['hi']:+.1f}]"
                              for k in ("released-placebo", "released-repaired") if k in cell)
            print(f"{release} {condition:8s} {line}  {diffs}")
    print(f"wrote {path.relative_to(ROOT)}")


CONDITION_LABELS = (("template", "BixBench template"), ("shown", "chat prompt"),
                    ("withheld", "question withheld"))
RELEASE_LABELS = (("v15", "v1.5"), ("v10", "v1.0"))


def table_rows(report):
    """The rows of the paper's table of gpt-4o without the data: per release and condition, the margin
    over chance through R, P and U, the paired differences, and how often R's second-smallest option
    is picked."""
    est = lambda e: f"${e['mean']:+.1f}$"
    ci = lambda e: f"${e['mean']:+.1f}$ $[{e['lo']:+.1f},{e['hi']:+.1f}]$"
    rows = []
    for release, rlabel in RELEASE_LABELS:
        entry = report["releases"].get(release, {})
        first = True
        for condition, clabel in CONDITION_LABELS:
            if condition not in entry:
                continue
            c = entry[condition]
            n_items = c["released"]["margin"]["n_items"] // (SHOWN_DRAWS if condition == "shown" else WITHHELD_DRAWS)
            head = f"{rlabel} (${n_items}$)" if first else ""
            first = False
            rows.append(f"{head} & {clabel} & " + " & ".join(est(c[a]["margin"]) for a, _ in ARMS)
                        + f" & {ci(c['released-placebo'])} & {ci(c['released-repaired'])} & "
                        + f"${c['released']['pick_rank_shares'][1]:.1f}$\\\\")
    return rows


def dry_run(args):
    cache = load_cache(OUT / f"cache_{args.model.replace('/', '_')}.jsonl")
    calls = prompt = 0
    lines = []
    for release, arm, suffix, condition, built in plan(args.model, args.releases, args.conditions, args.limit):
        todo = [r for r in built if cache_key(request(args.model, r, condition)) not in cache]
        chars = sum(sum(len(m["content"]) for m in r["messages"]) for r in todo)
        calls, prompt = calls + len(todo), prompt + chars / 4 + 8 * len(todo)
        lines.append(f"  {release} {arm:8s} {condition:8s} {len(todo):5d} calls")
    # a forced letter is short: most replies are a line or two; the caps bound the rest
    out_typical = calls * 40
    out_cap = sum(len(b) * (SHOWN_TOKENS if c == "shown" else WITHHELD_TOKENS)
                  for _, _, _, c, b in plan(args.model, args.releases, args.conditions, args.limit))
    print("\n".join(lines))
    inp, _, cout = oa.prices(args.model)
    print(f"dry run, {args.model}: {calls} calls, ~{prompt / 1e6:.2f} M prompt tokens, ~{out_typical / 1e6:.2f} M "
          f"reply tokens typical (at most {out_cap / 1e6:.2f} M); ~${(prompt * inp + out_typical * cout) / 1e6:.2f} "
          f"typical, at most ${(prompt * inp + out_cap * cout) / 1e6:.2f}")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--model", default="gpt-4o")
    a.add_argument("--base", default=None, help="default: the Azure endpoint if one is configured "
                                                "(openai_api.default_base), else OpenAI's API; a mock "
                                                "server's base in tests")
    a.add_argument("--releases", default="v15,v10")
    a.add_argument("--conditions", default="template,shown,withheld")
    a.add_argument("--limit", type=int, default=0, help="the first N items of each release only")
    a.add_argument("--concurrency", type=int, default=16)
    a.add_argument("--dry-run", action="store_true", help="count the calls and estimate their cost; call nothing")
    a.add_argument("--summarise", action="store_true", help="summarise the dumps already written; call nothing")
    a.add_argument("--latex", action="store_true", help="print the paper's table rows from the shipped summary")
    args = a.parse_args()
    args.releases = [r for r in args.releases.split(",") if r]
    args.conditions = [c for c in args.conditions.split(",") if c]
    if args.latex:
        report = json.loads((ROOT / "results" / f"openai_nodata_{args.model.replace('/', '_')}.json").read_text())
        print("\n".join(table_rows(report)))
        return
    if args.dry_run:
        dry_run(args)
        return
    if not args.summarise:
        args.base = args.base or oa.default_base()
        if not oa.is_openai(args.base):
            raise SystemExit(f"{args.base} is neither OpenAI's API nor an Azure OpenAI resource")
        asyncio.run(run(args))
    summarise(args)


if __name__ == "__main__":
    main()
