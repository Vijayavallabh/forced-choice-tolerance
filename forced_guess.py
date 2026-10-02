#!/usr/bin/env python3
r"""gpt-4o without the data and without options, asked for a number (not registered).

Section~\ref{sec:conclusion} recommends measuring the no-data baseline without options and asking for a number,
since BixBench's condition without options lets a model decline and so counts what it will state, not what it
knows. This runs that recommendation on the model whose forced no-data margin Section~\ref{sec:open} examines,
gpt-4o (version 2024-11-20; 2024-08-06 is no longer offered to new Azure OpenAI customers), on each release's numeric questions -- the
items and order of ``openai_nodata.py`` -- under two prompts:

* ``stated``: ``free_response.py``'s free arm, the question alone and the answer as a number, the condition the
  open-weight models ran (tab:free, no longer in the paper); a model may still decline;
* ``forced``: the same, adding that the data are not available and that a best estimate is required even when
  unsure.

Greedy, 320 reply tokens, one reply per question. A reply is read as the tolerance reads an answer (the last
number, ``answer_extraction.last_number``); one saying the value cannot be determined states no number
(``partial_knowledge.stated``). Per release and prompt: the share of questions with a number; the share within
5% of the key; among the numbers, the share whose nearest released option is the key
(``answer_extraction.nearest_is_key_last``; 25% if the number carries nothing about the answer); and gpt-4o's
forced choice among the released options on the same questions under BixBench's template with the question
(``openai_nodata.py``, three orderings a question), beside what a model choosing the option nearest its own
number, and guessing where it gives none, would score. Plain 95% percentile cluster bootstraps over capsules
(``bixbench_withdata.cluster_interval``). Every reply is cached, so a rerun calls nothing it has called before;
calls go through ``openai_api.py`` (its key, retries, ledger and budget).

    AGENTICLS_OPENAI_BUDGET_USD=800 python3 forced_guess.py run
    python3 forced_guess.py analyse     # results/forced_guess.json, from the shipped replies
"""
import argparse
import asyncio
import gzip
import hashlib
import json
import pathlib

import numpy as np

import answer_extraction as ax
import bixbench_withdata as bw
import free_response as fr
import openai_nodata as on
import partial_knowledge as pk

ROOT = pathlib.Path(__file__).resolve().parent
CACHE = ROOT / "build" / "openai" / "forced_guess"
SHIPPED = ROOT / "results" / "forced_guess_replies.jsonl.gz"
OUT = ROOT / "results" / "forced_guess.json"
MODEL = "gpt-4o"
TOKENS = 320
FORCED_SYSTEM = (fr.FREE_SYSTEM + "\nThe data the question refers to are not available to you. Give your best "
                 "estimate of the value as a number even if you are unsure; do not decline to answer.")
PROMPTS = {"stated": fr.FREE_SYSTEM, "forced": FORCED_SYSTEM}
RELEASES = ("v15", "v10")
CHANCE = 0.25


def items(release):
    """openai_nodata.py's released rows for one release, in its order (the template dumps' item index)."""
    return on.release_rows(release)["released"]


def payload(system, row):
    return {"model": MODEL, "temperature": 0.0, "max_tokens": TOKENS,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": f"Question: {row['question']}"}]}


def key_of(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()


async def run(args):
    import httpx
    import openai_api as oa
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"cache_{MODEL}.jsonl"
    cache = {}
    if path.exists():
        for line in path.open():
            rec = json.loads(line)
            cache[rec["key"]] = rec
    handle = path.open("a")
    sem = asyncio.Semaphore(args.concurrency)
    base = args.base or oa.default_base()
    out = []
    async with httpx.AsyncClient(timeout=600) as http:
        async def call(p):
            k = key_of(p)
            if k not in cache:
                async with sem:
                    try:
                        data = await oa.post_chat(http, base, p, f"forced_guess:{MODEL}")
                        choice = data["choices"][0]
                        rec = {"key": k, "reply": choice["message"].get("content") or "", "model": data.get("model")}
                    except oa.ContentFiltered:
                        rec = {"key": k, "reply": "", "model": None, "content_filter": True}
                cache[k] = rec
                handle.write(json.dumps(rec) + "\n")
                handle.flush()
            return cache[k]

        for release in RELEASES:
            rows = items(release)
            for prompt, system in PROMPTS.items():
                recs = await asyncio.gather(*(call(payload(system, r)) for r in rows))
                for index, (r, rec) in enumerate(zip(rows, recs)):
                    out.append({"release": release, "prompt": prompt, "item": index, "cluster": r["cluster"],
                                "question_id": r.get("question_id"), "reply": rec["reply"], "served": rec["model"],
                                **({"content_filter": True} if rec.get("content_filter") else {})})
                print(f"  {release} {prompt:7s} {len(rows)} replies", flush=True)
    handle.close()
    with gzip.GzipFile(filename="", mode="wb", mtime=0, fileobj=SHIPPED.open("wb")) as fh:
        fh.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in out).encode())
    print(f"{len(out)} replies -> {SHIPPED.relative_to(ROOT)}")


def forced_choice(release):
    """{item: gpt-4o's forced-choice accuracy through R under BixBench's template with the question}, from the
    dump ``openai_nodata.py`` ships."""
    name = f"{MODEL}_{on.RELEASES[release]}_bixprompt_question_generated.jsonl.gz"
    got = {}
    with gzip.open(ROOT / "results" / "openai_nodata" / name, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            got.setdefault(row["item"], []).append(1.0 if row["answer"] == row["gold"] else 0.0)
    return {i: float(np.mean(v)) for i, v in got.items()}


def summary(rows, forced):
    cap = [r["cluster"] for r in rows]
    said = [r for r in rows if r["number"] is not None]
    iv = lambda xs, cs: bw.cluster_interval(xs, cs, level=0.95) if xs else None
    predicted = [r["nearest"] if r["number"] is not None else CHANCE for r in rows]
    choice = [forced[r["item"]] for r in rows]
    return {"n_items": len(rows), "n_clusters": len(set(cap)),
            "stated_share": 100.0 * len(said) / len(rows),
            "within_5pct": iv([float(r["within"]) for r in rows], cap),
            "nearest_is_key_when_stated": iv([r["nearest"] for r in said], [r["cluster"] for r in said]),
            "forced_choice": iv(choice, cap),
            "predicted_forced": 100.0 * float(np.mean(predicted)),
            "forced_minus_predicted": iv([c - p for c, p in zip(choice, predicted)], cap)}


def read(reply, options):
    """(the number the reply states or None, nearest-is-key credit, within 5% of the key)."""
    number = pk.stated(reply, options[0])
    return (number, ax.nearest_is_key_last(reply, options) if number is not None else None,
            bool(fr.graded(reply, options[0], 0.05)) if number is not None else False)


def analyse(_args):
    with gzip.open(SHIPPED, "rt") as fh:
        replies = [json.loads(l) for l in fh]
    report = {"model": MODEL, "not_registered": True, "chance": CHANCE,
              "served": sorted({r["served"] for r in replies if r.get("served")}),
              "content_filtered": sum(1 for r in replies if r.get("content_filter"))}
    for release in RELEASES:
        rows = items(release)
        forced = forced_choice(release)
        for prompt in PROMPTS:
            got = []
            for r in (x for x in replies if x["release"] == release and x["prompt"] == prompt):
                row = rows[r["item"]]
                number, nearest, within = read(r["reply"], [row["ideal"], *row["distractors"]])
                got.append({"item": r["item"], "cluster": r["cluster"], "number": number, "nearest": nearest,
                            "within": within})
            report[f"{release}|{prompt}"] = summary(got, forced)
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    f = lambda v: "--" if v is None else f"{v['mean']:5.1f} [{v['lo']:5.1f},{v['hi']:5.1f}]"
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"{k:10s} n {v['n_items']:3d}: states a number {v['stated_share']:5.1f}%; within 5% {f(v['within_5pct'])}; "
                  f"nearest is key {f(v['nearest_is_key_when_stated'])}; forced choice {f(v['forced_choice'])}, "
                  f"predicted {v['predicted_forced']:.1f}, forced - predicted {f(v['forced_minus_predicted'])}")
    print(f"served {report['served']}; wrote {OUT.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base", default=None)
    r.add_argument("--concurrency", type=int, default=16)
    r.set_defaults(fn=lambda a: asyncio.run(run(a)))
    sub.add_parser("analyse").set_defaults(fn=analyse)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
