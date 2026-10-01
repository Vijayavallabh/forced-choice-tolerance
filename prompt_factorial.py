#!/usr/bin/env python3
r"""Which part of the prompt moves a no-data reading: the five differences, crossed.

Table~\ref{tab:instrument}'s two framings differ in five ways at once, and
attributing the difference to ``the framing'' names none of them. BixBench's template (``prompts.py`` at the pinned commit) and the chat
frame differ in

  instruction  "Extract the single letter answer to the following question from
               the given options."  vs  "You are answering a multiple-choice
               question."
  force        "You must pick one answer even if you are unsure."  vs  nothing
  reason       nothing  vs  "Think it through, then ..."
  format       an XML answer tag, read after ``<answer>``  vs  a ``FINAL:`` line
  placement    everything in one user message  vs  the instructions in a system
               message and the options alone in the user message

Both carry ``Question: [withheld]``; neither names a rank, a value or an order.
This crosses all five (32 cells) with everything else held: the same weights, the
same items, the same letter orderings (``agentic_probe.build_rollouts``' seeds),
the same clean control and the same read-out -- the arg-max over the $k$
single-token letter continuations after the format's own answer marker. The
all-template corner is BixBench's prompt byte for byte and the all-chat corner is
the chat frame byte for byte, so the two reproduce the paper's cells; a 33rd cell
adds the interpreter's rules to the chat corner, which is the system text the
paper's letter read-out of the chat frame carried.

Every rollout's pick is dumped, so each factor's effect is a paired contrast on
identical items and orderings (``prompt_factorial_summary.py``).

    python3 prompt_factorial.py --model Qwen/Qwen2.5-14B-Instruct --device cuda:0 --tag qwen14b
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import pathlib
import random

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np

from agentic_probe import ALPHABET, TOOL_RULES, WITHHELD, letter_ids, option_block

FACTORS = ("instruction", "force", "reason", "format", "placement")
INSTRUCTION = ("Extract the single letter answer to the following question from the given options.",
               "You are answering a multiple-choice question.")
FORCE = " You must pick one answer even if you are unsure."
XML_TAIL = ("IMPORTANT: You must only output a single letter answer in XML format.",
            "IMPORTANT: Think it through, then output a single letter answer in XML format.")
XML_EXAMPLE = "\n\n Example Output: <answer> X </answer>"
FINAL_TAIL = ("Reply with a single final line in exactly this form:\nFINAL: <letter>",
              "Think it through, then reply with a single final line in exactly this form:\nFINAL: <letter>")
FILES = {
    "mmlupro": ("build/mmlu_pro_matched_released.jsonl", "build/mmlu_pro_matched_clean.jsonl", 2),
    "bixbench": ("build/bixbench_v15.jsonl", "build/bixbench_all_clean.jsonl", 3),
}


def cell_name(levels):
    """``T`` for the template's level of a factor, ``C`` for the chat frame's."""
    return "".join("TC"[x] for x in levels)


def messages_for(levels, block, tools=False):
    """The prompt at one corner of the 2^5 design, and the marker the letter follows."""
    instruction, force, reason, fmt, placement = levels
    head = INSTRUCTION[instruction] + (FORCE if force == 0 else "")
    if fmt == 0:
        tail, prefix = XML_TAIL[reason] + XML_EXAMPLE, "<answer>"
    else:
        tail, prefix = FINAL_TAIL[reason], "FINAL:"
    if tools:
        # the chat frame's system text with the interpreter's rules instead, which
        # is what the paper's letter read-out of that frame was given
        if levels != (1, 1, 1, 1, 1):
            raise ValueError("the tool rules are only crossed with the chat corner")
        tail = TOOL_RULES.format(turns=3)
    question = f"Question: {WITHHELD}\n\nOptions:\n"
    if placement == 0:
        user = head + "\n\n" + question + block + "\n" + tail
        return [{"role": "user", "content": user}], prefix, user
    user = question + block
    return [{"role": "system", "content": head + " " + tail},
            {"role": "user", "content": user}], prefix, user


def orderings(rows, draws, seed):
    """The letter orderings ``agentic_probe.build_rollouts`` draws, one per (item, draw)."""
    rng = random.Random(seed)
    out = []
    for index, row in enumerate(rows):
        options = [row["ideal"]] + list(row["distractors"])
        for draw in range(draws):
            permutation = list(range(len(options)))
            rng.shuffle(permutation)
            out.append({"item": index, "draw": draw, "k": len(options),
                        "cluster": str(row.get("cluster") or row.get("capsule_uuid") or index),
                        "gold": ALPHABET[permutation.index(0)],
                        "block": option_block(options, permutation)})
    return out


def read_cell(model, tok, rollouts, levels, tools, batch_size, device):
    """Arg-max letter for every rollout under one cell, batched shortest-first."""
    import torch

    built = []
    for r in rollouts:
        messages, prefix, user = messages_for(levels, r["block"], tools)
        built.append((r, messages, prefix, user))
    ids, _ = letter_ids(tok, max(r["k"] for r in rollouts))
    # shortest-first on the user message, as the probe orders its batches, so the
    # two corners see the padding the paper's cells saw
    built.sort(key=lambda t: len(t[3]))
    picks = {}
    for start in range(0, len(built), batch_size):
        chunk = built[start:start + batch_size]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) + p
                 for _, m, p, _ in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        for (r, _, _, _), row in zip(chunk, logits):
            picks[(r["item"], r["draw"])] = ALPHABET[int(torch.argmax(row[ids[:r["k"]]]).item())]
    return [picks[(r["item"], r["draw"])] for r in rollouts]


def read_cell_vllm(llm, tok, rollouts, levels, tools):
    """The same read-out through vLLM: one greedy token restricted to the letters.

    Greedy decoding over ``allowed_token_ids`` returns the arg-max over exactly
    the $k$ letter tokens the HF path compares, so the read-out is the same
    function; only the kernels differ, which is why the corners are checked
    against the paper's cells rather than assumed to match them.
    """
    from vllm import SamplingParams

    k = max(r["k"] for r in rollouts)
    ids, _ = letter_ids(tok, k)
    params = SamplingParams(max_tokens=1, temperature=0.0, allowed_token_ids=list(ids[:k]))
    prompts = []
    for r in rollouts:
        messages, prefix, _ = messages_for(levels, r["block"], tools)
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix
        # token ids, not text: vLLM adds special tokens to a text prompt, and the
        # chat template already opens with the model's BOS, as the HF path assumes
        prompts.append({"prompt_token_ids": tok.encode(text, add_special_tokens=False)})
    outs = llm.generate(prompts, params, use_tqdm=False)
    back = {tid: ALPHABET[i] for i, tid in enumerate(ids[:k])}
    return [back.get(o.outputs[0].token_ids[0], "") for o in outs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default=None, help="comma-separated GPU indices for a large model")
    ap.add_argument("--shard-auto", type=float, default=0.85)
    ap.add_argument("--files", default="mmlupro,bixbench")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--engine", default="hf", choices=("hf", "vllm"),
                    help="vllm for the 70B-class models: tensor-parallel over --shard")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = model = device = None
    if args.engine == "vllm":
        from vllm import LLM
        llm = LLM(model=args.model, tensor_parallel_size=len(args.shard.split(",")) if args.shard else 1,
                  dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=args.shard_auto,
                  enable_prefix_caching=True, seed=0)
        tok = llm.get_tokenizer()
    elif args.shard:
        cards = [int(c) for c in args.shard.split(",")]
        caps = {c: f"{int(args.shard_auto * torch.cuda.mem_get_info(c)[0] / 2 ** 30)}GiB" for c in cards}
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                     device_map="auto", max_memory=caps).eval()
        device = f"cuda:{cards[0]}"
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
        device = args.device

    cells = [(levels, False) for levels in itertools.product((0, 1), repeat=len(FACTORS))]
    cells.append(((1, 1, 1, 1, 1), True))
    out = pathlib.Path(args.out or f"results/prompt_factorial_{args.tag}.json")
    report = json.loads(out.read_text()) if out.exists() else {}
    report.update({"model": args.model, "factors": FACTORS, "transformers": transformers.__version__,
                   "torch": torch.__version__, "engine": args.engine,
                   "batch_size": args.batch_size if args.engine == "hf" else None})
    if args.engine == "vllm":
        import vllm
        report["vllm"] = vllm.__version__
    dump_dir = pathlib.Path("build/dumps")
    dump_dir.mkdir(parents=True, exist_ok=True)
    for name in args.files.split(","):
        items, clean, draws = FILES[name]
        arms = {"file": orderings([json.loads(l) for l in open(items)], draws, args.seed),
                "clean": orderings([json.loads(l) for l in open(clean)], draws, args.seed + 1)}
        dump = dump_dir / f"prompt_factorial_{args.tag}_{name}.jsonl.gz"
        entry = report.setdefault("files", {}).setdefault(name, {"items": items, "clean": clean,
                                                                  "draws": draws, "cells": {}})
        with gzip.open(dump, "wt") as fh:
            for levels, tools in cells:
                label = cell_name(levels) + ("+tools" if tools else "")
                summary = {}
                for arm, rollouts in arms.items():
                    picks = (read_cell_vllm(llm, tok, rollouts, levels, tools) if llm is not None
                             else read_cell(model, tok, rollouts, levels, tools, args.batch_size, device))
                    hits = np.array([p == r["gold"] for p, r in zip(picks, rollouts)], float)
                    chance = np.array([1.0 / r["k"] for r in rollouts])
                    summary[arm] = {"accuracy": 100 * hits.mean(),
                                    "margin_over_chance": 100 * (hits - chance).mean()}
                    for p, r in zip(picks, rollouts):
                        fh.write(json.dumps({"cell": label, "arm": arm, "item": r["item"], "draw": r["draw"],
                                             "cluster": r["cluster"], "k": r["k"], "gold": r["gold"],
                                             "pick": p}) + "\n")
                summary["margin_over_clean"] = (summary["file"]["margin_over_chance"]
                                                - summary["clean"]["margin_over_chance"])
                entry["cells"][label] = summary
                print(f"{args.tag} {name} {label:9s} file {summary['file']['accuracy']:5.1f}% "
                      f"clean {summary['clean']['accuracy']:5.1f}%  over clean "
                      f"{summary['margin_over_clean']:+5.1f}", flush=True)
        entry["dump"] = str(dump)
        out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
