#!/usr/bin/env python3
"""Can a solver learn the option-geometry channel from in-context examples?

The probe in ``no_data_probe.py`` answers a narrow question: do published and
open-weight solvers, as they are, collect the geometry credit a benchmark
leaves lying around? The answer there is no. That is a null about today's
solvers, and on its own it invites the reading "the channel does not matter".

This script tests the obvious next threat. Nobody hand-codes "take the
second-smallest value", but anything that fits to a benchmark -- few-shot
prompting, prompt search, fine-tuning on its train split -- could pick the
preference up implicitly. In-context learning is the cheapest instance of that
and the easiest to run as a controlled experiment.

The design isolates rank geometry from everything else:

  * The **target item is identical in every arm**: the released options, one
    shuffled presentation per draw, the benchmark's own prompt, the same
    letter-logit read-out. Only the demonstrations before it change.
  * **Withholding the stem** ("[withheld]") removes the question from both the
    demonstrations and the target, so there is no content to know. Above-chance
    accuracy in that condition cannot be knowledge; option geometry is the only
    signal left.
  * The control arm draws its demonstrations from the **repaired** benchmark,
    whose key ranks are uniform. Repaired demonstrations carry the same number
    of examples, the same formatting, and the same (uniform) distribution over
    answer *letters*; they differ only in carrying no rank signal. So a gain
    over zero-shot that appears with original demonstrations and not with
    repaired ones is attributable to the key-rank distribution and not to
    format, letters, or the mere presence of examples.

Demonstrations are drawn from other clusters than the target's, so no item and
no source study teaches its own answer.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from mcq_audit import repair_row
from no_data_probe import (ANSWER_PREFIX, MCQ_PROMPT_TEMPLATE, WITHHELD, letters_for,
                           letter_token_ids, load_items, open_outcomes, option_block,
                           rank_of_each_slot, render, score)

SEED = 20260920
DEMO_HEADER = ("Here are {n} solved examples from this benchmark, in the same format. "
               "Use them to answer the question that follows.")


def demo_block(examples, k):
    """Solved examples: the options as presented, then the keyed letter."""
    letters = letters_for(k)
    out = []
    for options, permutation, key in examples:
        out.append("Question: " + WITHHELD + "\n"
                   + option_block(options, permutation, letters) + "\n"
                   + f"<answer>{letters[permutation.index(key)]}</answer>")
    return "\n\n".join(out)


def presentation(item, options, seed, tag):
    """A shuffled presentation of ``options``, seeded so arms share it."""
    rng = random.Random(f"{seed}|{tag}|{item['question_id']}")
    permutation = list(range(len(options)))
    rng.shuffle(permutation)
    return permutation


def build_conditions(items, arms, draws, seed, k):
    """One record per (item, arm, draw).

    The repaired pool is drawn once per draw, not once per target item, so the
    control arm sees a single coherent "repaired benchmark" rather than a fresh
    redraw per example -- the same thing a maintainer who shipped the repair
    would have published.
    """
    numeric = [it for it in items if it["rank"] is not None]
    # Demonstration pool for each cluster: every numeric item outside it.
    clusters = sorted({it["cluster"] for it in numeric})
    pool_of = {c: [it for it in numeric if it["cluster"] != c] for c in clusters}

    records = []
    for draw in range(draws):
        rep_rng = random.Random(f"{seed}|{draw}|repair-pool")
        repaired = {}
        for item in numeric:
            redrawn = repair_row(item["options"], rep_rng)
            repaired[item["question_id"]] = redrawn or item["options"]

        for item in items:
            # The target's own presentation, shared by every arm and demo type.
            target_perm = presentation(item, item["options"], f"{seed}|{draw}", "target")
            target_ranks = rank_of_each_slot(item["options"], target_perm)
            target_options = option_block(item["options"], target_perm, letters_for(k))

            for n_shots, pool, stem in arms:
                if n_shots == 0:
                    demos, demo_ids = "", []
                else:
                    # The pool is not seeded on ``pool``, so the original and
                    # repaired arms show the same example items in the same
                    # presentation order and differ only in the values shown.
                    rng = random.Random(f"{seed}|{draw}|{item['question_id']}|{n_shots}")
                    others = pool_of[item["cluster"]]
                    picked = rng.sample(others, min(n_shots, len(others)))
                    examples, demo_ids = [], [o["question_id"] for o in picked]
                    for other in picked:
                        options = (other["options"] if pool == "original"
                                   else repaired[other["question_id"]])
                        perm = list(range(k))
                        rng.shuffle(perm)
                        examples.append((options, perm, other["key"]))
                    demos = (DEMO_HEADER.format(n=len(examples)) + "\n\n"
                             + demo_block(examples, k) + "\n\n")

                # The target is rendered exactly as the main probe renders it.
                prompt = demos + MCQ_PROMPT_TEMPLATE.format(
                    question=WITHHELD if stem == "withheld" else item["question"],
                    options=target_options)
                records.append({
                    "question_id": item["question_id"], "cluster": item["cluster"],
                    "arm": f"{stem}_s{n_shots}_{pool}", "draw": draw,
                    "n_shots": n_shots, "demo_pool": pool, "stem": stem,
                    # Which items were shown as examples: not written to the
                    # outcome file, but rebuildable from the seed, so the
                    # "no demonstration from the target's own study" control
                    # is checkable rather than asserted.
                    "demo_ids": demo_ids,
                    "prompt": prompt, "is_numeric": item["rank"] is not None,
                    "target_letter": letters_for(k)[target_perm.index(item["key"])],
                    "key_rank": item["rank"], "slot_ranks": target_ranks,
                })
    return records


def run_model(name, records, args, k):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n=== {name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids, variant = letter_token_ids(tokenizer, k)
    print(f"  scoring letters as {variant.format('A')!r}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map=args.device,
        attn_implementation=args.attn)
    model.eval()

    unique, index = {}, []
    for record in records:
        text = render(tokenizer, record["prompt"])
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest not in unique:
            unique[digest] = (len(unique), text)
        index.append(unique[digest][0])
    prompts = [text for _, text in sorted(unique.values())]
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    print(f"  {len(records)} conditions -> {len(prompts)} distinct prompts, "
          f"max {max(lengths)} tokens", flush=True)

    scored = score(model, tokenizer, prompts, ids, args.batch_size, args.device)
    rows = []
    for record, slot in zip(records, index):
        pick, probs = scored[slot]
        chosen = letters_for(k)[pick]
        rows.append({
            "model": name, "question_id": record["question_id"],
            "cluster": record["cluster"], "arm": record["arm"], "draw": record["draw"],
            "n_shots": record["n_shots"], "demo_pool": record["demo_pool"],
            "stem": record["stem"], "n_options": k,
            "is_numeric": record["is_numeric"], "target_letter": record["target_letter"],
            "predicted_letter": chosen, "correct": int(chosen == record["target_letter"]),
            "key_rank": record["key_rank"],
            "chosen_rank": record["slot_ranks"][pick] if record["slot_ranks"] else None,
            "letter_probs": probs,
        })
    del model
    torch.cuda.empty_cache()
    return rows


def run_model_vllm(name, records, args, k):
    """The same read-out through vLLM, for the 70B-class models: one greedy token restricted to
    the $k$ letter tokens, which is the arg-max the HF path takes over the same tokens. Only the
    kernels differ (and the weights' precision where ``--quantization`` is set), so the letter
    probabilities are not recorded."""
    from vllm import LLM, SamplingParams

    print(f"\n=== {name} (vllm)", flush=True)
    llm = LLM(model=name, tensor_parallel_size=args.tp, dtype="bfloat16",
              quantization=args.quantization, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_util, enable_prefix_caching=True, seed=0)
    tokenizer = llm.get_tokenizer()
    ids, variant = letter_token_ids(tokenizer, k)
    print(f"  scoring letters as {variant.format('A')!r}", flush=True)
    unique, index = {}, []
    for record in records:
        text = render(tokenizer, record["prompt"])
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest not in unique:
            unique[digest] = (len(unique), text)
        index.append(unique[digest][0])
    prompts = [text for _, text in sorted(unique.values())]
    # token ids, not text: vLLM adds special tokens to a text prompt, and the chat template
    # already opens with the model's BOS, as the HF path assumes
    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    print(f"  {len(records)} conditions -> {len(prompts)} distinct prompts, "
          f"max {max(map(len, encoded))} tokens", flush=True)
    params = SamplingParams(max_tokens=1, temperature=0.0, allowed_token_ids=list(ids))
    outs = llm.generate([{"prompt_token_ids": e} for e in encoded], params, use_tqdm=False)
    picks = [ids.index(o.outputs[0].token_ids[0]) for o in outs]
    rows = []
    for record, slot in zip(records, index):
        pick = picks[slot]
        chosen = letters_for(k)[pick]
        rows.append({
            "model": name, "question_id": record["question_id"],
            "cluster": record["cluster"], "arm": record["arm"], "draw": record["draw"],
            "n_shots": record["n_shots"], "demo_pool": record["demo_pool"],
            "stem": record["stem"], "n_options": k,
            "is_numeric": record["is_numeric"], "target_letter": record["target_letter"],
            "predicted_letter": chosen, "correct": int(chosen == record["target_letter"]),
            "key_rank": record["key_rank"],
            "chosen_rank": record["slot_ranks"][pick] if record["slot_ranks"] else None,
            "letter_probs": None, "engine": "vllm", "quantization": args.quantization,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", default="data/bixbench.jsonl")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 8, 32])
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="results/probe_icl")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--attn", default="eager", choices=["eager", "sdpa"],
                        help="attention kernel; eager is exact and the default, sdpa is "
                             "needed for the long prompts of a 64-shot run")
    parser.add_argument("--n-options", type=int, default=None)
    parser.add_argument("--stems", nargs="+", default=["withheld", "shown"])
    parser.add_argument("--engine", default="hf", choices=("hf", "vllm"),
                        help="vllm for the 70B-class models, tensor-parallel over --tp GPUs")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--quantization", default=None, help="vllm only, e.g. fp8")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-util", type=float, default=0.9)
    args = parser.parse_args()

    items, k = load_items(args.jsonl, args.question_field, args.n_options)
    if args.limit:
        items = items[:args.limit]
    items = [it for it in items if it["rank"] is not None]

    arms = []
    for stem in args.stems:
        for n in sorted(set(args.shots)):
            for pool in (["none"] if n == 0 else ["original", "repaired"]):
                arms.append((n, pool, stem))
    print(f"{len(items)} numeric items with {k} options, "
          f"{len({it['cluster'] for it in items})} clusters, {len(arms)} arms")

    records = build_conditions(items, arms, args.draws, args.seed, k)
    print(f"{len(records)} conditions over {args.draws} draws")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        rows = (run_model_vllm if args.engine == "vllm" else run_model)(name, records, args, k)
        path = out / (name.replace("/", "__") + ".jsonl.gz")
        with open_outcomes(path, "wt") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"  wrote {path}", flush=True)
        for arm in sorted({r["arm"] for r in rows}):
            sel = [r["correct"] for r in rows if r["arm"] == arm]
            print(f"    {arm:28s} {sum(sel) / len(sel):6.1%}  (n={len(sel)})", flush=True)


if __name__ == "__main__":
    main()
