#!/usr/bin/env python3
r"""Delete the option set and ask for the number instead, on models we can run.

\S\ref{sec:open} reads Proposition~\ref{prop:exch}'s repair off BixBench's own
published arms: with the options gone, the two models the benchmark ran score
$2.9\%$ where the quoted run scores $34$ to $36$. That is one file and two
models, which is the limitation \S\ref{sec:limits} states. This removes it, by
running the same contrast ourselves across open-weight models:

    mcq    the question and the $k$ options, one letter, no way to decline
    free   the question alone, no options, answer as a number

Same items, same models, same decoding. The difference is what the option set
adds to a score that is supposed to measure recall.

Grading the free arm needs a rule, and the rule is the argument's weak point, so
it is stated and varied rather than chosen: the last number in the reply counts
as correct when it is within a relative tolerance of the key, reported at
$1\%$, $5\%$ and $10\%$ together. Exact string equality after normalisation
counts too, which is what catches answers written as percentages or in
scientific notation. And because a model that says ``I cannot determine this
without the data'' is not failing the tolerance but declining, the declination
rate is reported beside the accuracy, with the same conservative rule
``published_repair.py`` applies to BixBench's own transcripts.

    python3 free_response.py --model Qwen/Qwen2.5-14B-Instruct --device cuda:2 \
        --items build/bixbench_numeric_q.jsonl --out results/free_response.json
    python3 free_response.py regrade    # every free arm again from its dumped replies, no GPU

Chance for the free arm is zero: there is nothing to guess from.
"""
import argparse
import gzip
import json
import pathlib
import random
import re
import sys

import numpy as np

import agentic_probe as ap

# The tolerance's reading of a number (``NUMBER``, ``as_number``, ``graded``) lives in answer_numbers,
# which every analysis imports; it is re-exported here for the scripts that import it from this module.
from answer_numbers import NUMBER, as_number, graded  # noqa: E402,F401

# trailing per cent sign. Deliberately greedy about format and not about
# position: the last match in the reply is the answer.
MCQ_SYSTEM = ("You are answering a multiple-choice question. Reply with a single "
              "final line in exactly this form:\nFINAL: <letter>")
FREE_SYSTEM = ("You are answering a question with a numeric answer. Reply with a "
               "single final line in exactly this form:\nFINAL: <number>")

DECLINE = re.compile(
    r"cannot (be )?(determine|provide|answer|calculate|give|specify)"
    r"|can't (determine|provide|answer|calculate)"
    r"|(do|does) not have access"
    r"|(is|are) not provided|not available in|without (the )?(actual|specific"
    r"|access to|further|additional) (data|information|dataset)"
    r"|unable to (determine|provide|answer)"
    r"|no specific data|requires? (the )?(actual|specific) data"
    r"|insufficient (data|information)", re.IGNORECASE)


def build(rows, arm, draws, seed):
    """One rollout per (item, ordering) for mcq; one per item for free."""
    rng = random.Random(seed)
    out = []
    for index, row in enumerate(rows):
        options = [row["ideal"]] + list(row["distractors"])
        k = len(options)
        cluster = str(row.get("cluster") or row.get("capsule_uuid") or index)
        if arm == "free":
            out.append({"item": index, "draw": 0, "k": k, "cluster": cluster,
                        "gold": row["ideal"], "arm": "free",
                        "user": f"Question: {row['question']}"})
            continue
        for draw in range(draws):
            permutation = list(range(k))
            rng.shuffle(permutation)
            block = ap.option_block(options, permutation)
            out.append({"item": index, "draw": draw, "k": k, "cluster": cluster,
                        "gold": ap.ALPHABET[permutation.index(0)], "arm": "mcq",
                        "user": f"Question: {row['question']}\n\nOptions:\n{block}"})
    return out


def play(model, tok, rollouts, system, args):
    import torch
    with torch.no_grad():
        return _play(model, tok, rollouts, system, args)


def _play(model, tok, rollouts, system, args):
    for start in range(0, len(rollouts), args.batch_size):
        chunk = rollouts[start:start + args.batch_size]
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": r["user"]}],
            tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                             do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for row, sequence in zip(chunk, out):
            row["reply"] = tok.decode(sequence[enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
        print(f"  {min(start + args.batch_size, len(rollouts))}/{len(rollouts)}",
              flush=True)
    return rollouts


def cluster_boot(rows, statistic, bootstrap, seed, level=0.95):
    clusters = sorted({r["cluster"] for r in rows})
    index = {c: [i for i, r in enumerate(rows) if r["cluster"] == c]
             for c in clusters}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        take = [i for c in rng.choice(clusters, size=len(clusters), replace=True)
                for i in index[c]]
        draws.append(statistic([rows[i] for i in take]))
    tail = 100.0 * (1.0 - level) / 2.0
    sizes = np.array([len(index[c]) for c in clusters], dtype=float)
    return {"n_clusters": len(clusters),
            "n_effective": float(sizes.sum() ** 2 / (sizes ** 2).sum()),
            "interval": [float(np.percentile(draws, tail)),
                         float(np.percentile(draws, 100.0 - tail))]}


def free_cell(rollouts, tolerances, bootstrap, seed, grade=None):
    """The free arm's report from its rollouts: the declines, and at each tolerance the accuracy and its
    cluster interval, each reply graded by ``grade`` (``graded``, the last number within the tolerance)."""
    grade = grade or graded
    cell = {"chance": 0.0,
            "declined": 100.0 * float(np.mean([1.0 if DECLINE.search(r["reply"]) else 0.0 for r in rollouts])),
            "by_tolerance": {}}
    for tolerance in tolerances:
        for r in rollouts:
            r["correct"] = 1.0 if grade(r["reply"], r["gold"], tolerance) else 0.0
        got = {"accuracy": 100.0 * float(np.mean([r["correct"] for r in rollouts]))}
        got.update(cluster_boot(rollouts, lambda part: 100.0 * float(np.mean([r["correct"] for r in part])),
                                bootstrap, seed))
        cell["by_tolerance"][f"{tolerance:g}"] = got
    return cell


# each report's items, and the name its models' dumps carry
REGRADED = {"results/free_response.json": ("bixbench_numeric_q", "bixnum"),
            "results/free_response_placebo.json": ("bixbench_numeric_placebo", "bixnum_placebo"),
            "results/free_response_repaired.json": ("bixbench_numeric_repaired", "bixnum_repaired"),
            "results/free_response_mmlupro.json": ("mmlu_pro_matched_released", "mmlupro")}


def dump_of(model, items="bixnum"):
    """The rollouts a model's run dumped, where it wrote them or the gzipped copy that ships."""
    import arm_intervals
    tag = model.split("/")[-1].replace("-Instruct", "").replace(".", "_")
    path = arm_intervals.resolve(f"build/dumps/free_{tag}_{items}.jsonl")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def regrade(out="results/free_response.json", bootstrap=2000, seed=0, grade=None):
    """Grade every model's free arm again from its dumped replies, with no model loaded: the numbers
    it reports follow ``answer_numbers``, the reader every other table uses. The
    multiple-choice arm is untouched."""
    stem, items = REGRADED[out]
    path = pathlib.Path(out)
    report = json.loads(path.read_text())
    for key, entry in report.items():
        if not key.endswith(f"|{stem}"):
            continue
        rollouts = [r for r in dump_of(entry["model"], items) if r.get("arm") == "free"]
        assert rollouts, f"no free-arm rollouts dumped for {key}"
        entry["arms"]["free"] = free_cell(rollouts, entry["tolerances"], bootstrap, seed, grade)
        mcq = entry["arms"]["mcq"]
        entry["option_set_worth"] = {t: mcq["margin"] - cell["accuracy"]
                                     for t, cell in entry["arms"]["free"]["by_tolerance"].items()}
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"regraded the free arm of {sum(k.endswith(f'|{stem}') for k in report)} models in {out}")


def main():
    if sys.argv[1:2] == ["regrade"]:
        for out in REGRADED:
            regrade(out)
        return
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    a.add_argument("--device", default="cuda:2")
    a.add_argument("--items", default="build/bixbench_numeric_q.jsonl")
    a.add_argument("--draws", type=int, default=2)
    a.add_argument("--batch-size", type=int, default=16)
    a.add_argument("--max-new-tokens", type=int, default=160)
    a.add_argument("--bootstrap", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--tolerances", default="0.01,0.05,0.10")
    a.add_argument("--out", default="results/free_response.json")
    a.add_argument("--dump", default=None)
    args = a.parse_args()

    rows = [json.loads(l) for l in open(args.items, encoding="utf-8")
            if l.strip()]
    # only items whose key is a number can be graded free-response at all
    numeric = [r for r in rows if as_number(r["ideal"]) is not None]
    print(f"{args.items}: {len(rows)} items, {len(numeric)} with a numeric key")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": args.device})
    model.eval()

    tolerances = [float(t) for t in args.tolerances.split(",")]
    report = {"model": args.model, "items": args.items,
              "n_items": len(numeric), "tolerances": tolerances, "arms": {}}

    for arm, system in (("mcq", MCQ_SYSTEM), ("free", FREE_SYSTEM)):
        rollouts = build(numeric, arm, args.draws, args.seed)
        print(f"\n{arm}: {len(rollouts)} rollouts", flush=True)
        play(model, tok, rollouts, system, args)
        if arm == "mcq":
            for r in rollouts:
                r["answer"] = ap.final_letter(r["reply"], r["k"])
                r["correct"] = 1.0 if r["answer"] == r["gold"] else 0.0
                r["chance"] = 1.0 / r["k"]
            cell = {"accuracy": 100.0 * float(np.mean([r["correct"] for r in rollouts])),
                    "chance": 100.0 * float(np.mean([r["chance"] for r in rollouts])),
                    "unparsed": 100.0 * float(np.mean(
                        [1.0 if r["answer"] is None else 0.0 for r in rollouts])),
                    "declined": 100.0 * float(np.mean(
                        [1.0 if DECLINE.search(r["reply"]) else 0.0
                         for r in rollouts]))}
            cell["margin"] = cell["accuracy"] - cell["chance"]
            cell.update(cluster_boot(
                rollouts,
                lambda part: 100.0 * float(np.mean(
                    [r["correct"] - r["chance"] for r in part])),
                args.bootstrap, args.seed))
            report["arms"]["mcq"] = cell
        else:
            report["arms"]["free"] = free_cell(rollouts, tolerances, args.bootstrap, args.seed)
        if args.dump:
            path = pathlib.Path(args.dump)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                for r in rollouts:
                    fh.write(json.dumps(r) + "\n")

    # what the option set is worth, at each tolerance
    mcq = report["arms"]["mcq"]
    report["option_set_worth"] = {
        t: mcq["margin"] - cell["accuracy"]
        for t, cell in report["arms"]["free"]["by_tolerance"].items()}

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(out.read_text()) if out.exists() else {}
    key = f"{args.model}|{pathlib.Path(args.items).stem}"
    if key in merged:
        print(f"note: replacing an existing entry for {key}")
    merged[key] = report
    out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== {args.model} on {pathlib.Path(args.items).stem} ===")
    print(f"  mcq   acc {mcq['accuracy']:6.2f}%  chance {mcq['chance']:5.2f}%  "
          f"margin {mcq['margin']:+6.2f} "
          f"[{mcq['interval'][0]:+.2f},{mcq['interval'][1]:+.2f}]  "
          f"unparsed {mcq['unparsed']:.1f}%  declined {mcq['declined']:.1f}%")
    for t, cell in report["arms"]["free"]["by_tolerance"].items():
        print(f"  free  acc {cell['accuracy']:6.2f}% at {float(t):.0%} tolerance "
              f"[{cell['interval'][0]:+.2f},{cell['interval'][1]:+.2f}]  "
              f"option set worth {report['option_set_worth'][t]:+6.2f}")
    print(f"  free  declined {report['arms']['free']['declined']:.1f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
