"""The no-data baseline run as an agent rather than as a single token.

Every probe elsewhere in this paper reads one letter token's logit.  A benchmark
for agents is not scored that way: the system gets turns, writes code, reads the
output and then answers.  That difference matters here because the exploit this
paper prices -- "take the second-smallest option" -- is an arithmetic rule over
the option values, and arithmetic is the one thing a scaffold adds.  So the
question is not whether a language model collects the option geometry.  It is
whether an *agent* does.

Three conditions, all with the data withheld exactly as a no-data baseline
withholds it:

  question        the question is shown (the ordinary evaluation)
  withheld        the question is replaced by ``[withheld]``
  withheld_aware  the question is withheld and the system prompt says so, and
                  tells the agent to reason from the options themselves --
                  which is what a no-data baseline asks of it, stated out loud.
                  It is not told any rule about ranks.

Each condition is also run against a matched clean file whose key is uniform
among i.i.d. options, so a margin is read against what the same scaffold scores
where there is provably nothing to find.

    /opt/conda/bin/python agentic_probe.py --items build/bixbench_v15.jsonl \
        --model Qwen/Qwen2.5-14B-Instruct --device cuda:3 --out agentic_bixbench.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os

# transformers probes for TensorFlow when it inspects a tensor's type, and this
# machine's TensorFlow is broken against its protobuf. We never want it.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import pathlib
import random
import re
import subprocess
import sys
import tempfile

import numpy as np

RESULTS = pathlib.Path("results")
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
WITHHELD = "[withheld]"

TOOL_RULES = (
    "You can run Python. To run code, write a fenced block:\n"
    "```python\nprint(2 + 2)\n```\n"
    "I will run it and send you its output. You may do this up to {turns} times. "
    "When you are ready, reply with a single final line in exactly this form:\n"
    "FINAL: <letter>")

# The same loop with the interpreter taken away: the model still writes free text
# before it answers, it just has nothing to run.  This is the arm that separates
# "an agent collects it" from "a generation pass collects it", and it is the one
# the tool-use split inside a single arm cannot settle, because which rollouts
# run code is the model's choice and not ours.
PLAIN_RULES = (
    "Think it through, then reply with a single final line in exactly this "
    "form:\nFINAL: <letter>")

# Two preambles, one leading and one not.  ``withheld_aware`` names the thing
# being measured -- it tells the solver to read the options -- which is what a
# no-data baseline asks of a solver out loud.  ``withheld`` says nothing at all
# beyond the ordinary framing and simply does not supply a question, which is
# what a no-data baseline does to a solver in silence.  A margin present under
# the first and absent under the second is steerability, not a channel the
# instrument collects on its own; the paper needs both to tell them apart.
PREAMBLE = {
    "question": "You are answering a multiple-choice question. ",
    "withheld": "You are answering a multiple-choice question. ",
    "withheld_aware": (
        "You are answering a multiple-choice question, but the question text and "
        "the data it refers to have been withheld from you: all you are given is "
        "the list of options. You must still pick one. Study the options "
        "themselves and choose the one that is most likely to be the intended "
        "answer. "),
}

SYSTEM = {name: text + TOOL_RULES for name, text in PREAMBLE.items()}

CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
FINAL_LINE = re.compile(r"FINAL\s*:(.*)$", re.IGNORECASE | re.MULTILINE)
# A letter counts only where it stands alone. Reading the first capital after the
# colon takes the T of "FINAL: The answer is B", which costs a model that answers
# in sentences sixteen points and reports the loss as an option preference.
BRACKETED = re.compile(r"\(([A-Z])\)")
NAMED = re.compile(r"(?:option|answer|choice)\s+\(?([A-Z])\)?", re.IGNORECASE)
# A standalone letter must end a clause, or "I am confident" reads as option I.
STANDALONE = re.compile(r"(?:^|[^A-Za-z])([A-Z])\**\s*(?:[.,;:)\]]|$)")


def final_letter(text, k):
    """The option letter named on the last FINAL line, or None if none was.

    Read in priority order, because at ten options a bare standalone letter is
    ambiguous: "I think it is D" contains two of them. A parenthesised letter
    wins, then one introduced by "option"/"answer"/"choice", then the *last*
    standalone letter on the line, which is where a sentence puts its verdict.
    """
    allowed = set(ALPHABET[:k])
    for line in reversed(FINAL_LINE.findall(text)):
        for pattern, pick in ((BRACKETED, 0), (NAMED, 0), (STANDALONE, -1)):
            found = [m.upper() for m in pattern.findall(line) if m.upper() in allowed]
            if found:
                return found[pick]
    return None
BANNED = re.compile(
    r"\b(?:import\s+(?:os|sys|subprocess|socket|shutil|urllib|requests|pathlib|ctypes)"
    r"|__import__|open\s*\(|eval\s*\(|exec\s*\()")


def run_code(source: str, timeout: float = 5.0) -> str:
    """Run one block in a throwaway interpreter and return what it printed."""
    if BANNED.search(source):
        return "error: this sandbox allows arithmetic only (no imports, no file access)"
    with tempfile.TemporaryDirectory() as cwd:
        try:
            done = subprocess.run(
                [sys.executable, "-I", "-S", "-c", source], cwd=cwd, timeout=timeout,
                capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return f"error: timed out after {timeout:g}s"
    out = (done.stdout or "") + (("\n" + done.stderr) if done.returncode else "")
    out = out.strip()
    if not out:
        return "(no output)"
    return out[:800] + ("\n... (truncated)" if len(out) > 800 else "")


def option_block(options, permutation) -> str:
    return "\n".join(f"({ALPHABET[slot]}) {options[src]}"
                     for slot, src in enumerate(permutation))


def build_rollouts(rows, condition, draws, seed, prompt="agent"):
    """One rollout per (item, letter ordering)."""
    rng = random.Random(seed)
    out = []
    for index, row in enumerate(rows):
        options = [row["ideal"]] + list(row["distractors"])
        k = len(options)
        for draw in range(draws):
            permutation = list(range(k))
            rng.shuffle(permutation)
            question = row.get("question", "") if condition == "question" else WITHHELD
            block = option_block(options, permutation)
            user = (BIXBENCH_PROMPT.format(question=question, options=block + "\n")
                    if prompt == "bixbench"
                    else f"Question: {question}\n\nOptions:\n{block}")
            out.append({
                "item": index, "draw": draw, "k": k,
                "cluster": str(row.get("cluster") or row.get("capsule_uuid") or index),
                "gold": ALPHABET[permutation.index(0)], "user": user})
    return out


def generate(model, tok, chunk, args):
    """Greedy continuations for one batch, halving it on a CUDA failure.

    A conversation grows with the turn, so a batch that fits at turn one can
    fail at turn three. Splitting and retrying keeps the arm complete rather
    than dropping whichever rollouts happened to run long.
    """
    import torch

    texts = [tok.apply_chat_template(r["messages"], tokenize=False,
                                     add_generation_prompt=True) for r in chunk]
    try:
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(args.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.pad_token_id)
        return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)
    except RuntimeError:
        torch.cuda.empty_cache()
        if len(chunk) == 1:
            raise
        half = len(chunk) // 2
        return (generate(model, tok, chunk[:half], args)
                + generate(model, tok, chunk[half:], args))


# BixBench's own MCQ prompt, verbatim from bixbench/prompts.py at commit 4931118
# (MCQ_PROMPT_TEMPLATE_WITHOUT_REFUSAL); the missing newline before "IMPORTANT"
# is theirs. ``no_data_probe.py`` scores with this and reads the letter after a
# pre-filled ``<answer>``, which is what the field's baseline is. Carrying it
# here lets the prompt and the read-out be crossed on one model and one file,
# which is the only way to say which of them the margin belongs to.
BIXBENCH_PROMPT = (
    "Extract the single letter answer to the following question from the given options."
    " You must pick one answer even if you are unsure."
    "\n\nQuestion: {question}"
    "\n\nOptions:\n{options}"
    "IMPORTANT: You must only output a single letter answer in XML format."
    "\n\n Example Output: <answer> X </answer>")
ANSWER_PREFIX = "<answer>"
ANSWER_TAG = re.compile(r"<answer>\s*([A-Z])", re.IGNORECASE)

LETTER_VARIANTS = (" {}", "{}")


def letter_ids(tok, k):
    """Single distinct token id per answer letter, as it follows ``FINAL:``."""
    for variant in LETTER_VARIANTS:
        ids = []
        for letter in ALPHABET[:k]:
            encoded = tok.encode(variant.format(letter), add_special_tokens=False)
            if len(encoded) != 1:
                ids = None
                break
            ids.append(encoded[0])
        if ids and len(set(ids)) == len(ids):
            return ids, variant
    raise RuntimeError("no single-token, distinct encoding of the answer letters")


def play_argmax(model, tok, rollouts, args):
    """The same prompt, read one letter at a time.

    This is the instrument comparison the paper's title promises, with
    everything held fixed that can be held fixed: identical system prompt,
    identical user message, identical chat template, identical items and
    orderings. The only difference from ``play`` is that the answer is not
    generated -- ``FINAL:`` is pre-filled and the letter is the arg-max over the
    ``k`` single-token continuations, which is how every no-data baseline in
    this literature reads. Any difference between the two is the read-out and
    nothing else.
    """
    import torch

    bix = args.prompt == "bixbench"
    system = (PREAMBLE[args.condition]
              + (PLAIN_RULES if args.no_tools else TOOL_RULES)).format(turns=args.max_turns)
    prefix = ANSWER_PREFIX if bix else "FINAL:"
    ids, variant = letter_ids(tok, max(r["k"] for r in rollouts))
    print(f"  reading one letter at a time after {prefix!r}, variant {variant!r}")
    for r in rollouts:
        r["messages"] = ([] if bix else [{"role": "system", "content": system}]) + [
            {"role": "user", "content": r["user"]}]
        r["turns"], r["used_tool"] = 1, False
    order = sorted(rollouts, key=lambda r: len(r["user"]))
    for start in range(0, len(order), args.batch_size):
        chunk = order[start:start + args.batch_size]
        texts = [tok.apply_chat_template(r["messages"], tokenize=False,
                                         add_generation_prompt=True) + prefix
                 for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(args.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        for r, row in zip(chunk, logits):
            pick = int(torch.argmax(row[ids[:r["k"]]]).item())
            r["answer"] = ALPHABET[pick]
            r["messages"].append({"role": "assistant",
                                  "content": prefix + variant.format(r["answer"])})
    return rollouts


def play(model, tok, rollouts, args):
    """Run every rollout to a final letter, generating the live ones in batches."""
    import torch

    bix = args.prompt == "bixbench"
    rules = PLAIN_RULES if args.no_tools else TOOL_RULES
    system = (PREAMBLE[args.condition] + rules).format(turns=args.max_turns)
    for r in rollouts:
        r["messages"] = ([] if bix else [{"role": "system", "content": system}]) + [
            {"role": "user", "content": r["user"]}]
        r["answer"], r["turns"], r["used_tool"] = None, 0, False

    for _ in range(args.max_turns + 2):
        live = [r for r in rollouts if r["answer"] is None]
        if not live:
            break
        live.sort(key=lambda r: len(r["user"]))
        for start in range(0, len(live), args.batch_size):
            chunk = live[start:start + args.batch_size]
            replies = generate(model, tok, chunk, args)
            for r, reply in zip(chunk, replies):
                r["messages"].append({"role": "assistant", "content": reply})
                r["turns"] += 1
                if bix:
                    # the bixbench prompt asks for <answer> X </answer> and nothing
                    # else, so there is no tool loop and no final line to chase
                    tag = ANSWER_TAG.search(reply)
                    letter = tag.group(1).upper() if tag else None
                    r["answer"] = letter if letter and letter in ALPHABET[:r["k"]] else (
                        final_letter(reply, r["k"]) or "")
                    continue
                marker = FINAL_LINE.search(reply)
                hit = final_letter(reply, r["k"])
                block = CODE.search(reply)
                if hit and not (block and reply.index(block.group(0)) > marker.start()):
                    r["answer"] = hit
                elif block:
                    r["used_tool"] = True
                    r["messages"].append(
                        {"role": "user", "content": f"Output:\n{run_code(block.group(1))}"})
                elif hit:
                    r["answer"] = hit
                elif r["turns"] > args.max_turns:
                    r["answer"] = ""            # out of turns without a final line
                else:
                    # ran past the token budget mid-thought; ask for the line only.
                    r["messages"].append({"role": "user", "content":
                                          "Reply now with only the final line, "
                                          "in exactly this form: FINAL: <letter>"})
    for r in rollouts:
        if r["answer"] is None:                       # out of turns: take the last letter
            tail = r["messages"][-1]["content"] if r["messages"][-1]["role"] == "assistant" else ""
            r["answer"] = final_letter(tail, r["k"]) or ""
    return rollouts


def summarise(rollouts, bootstrap, seed) -> dict:
    correct = np.array([1.0 if r["answer"] == r["gold"] else 0.0 for r in rollouts])
    chance = np.array([1.0 / r["k"] for r in rollouts])
    clusters = sorted({r["cluster"] for r in rollouts})
    by_cluster = {c: [i for i, r in enumerate(rollouts) if r["cluster"] == c]
                  for c in clusters}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        take = [i for c in rng.choice(clusters, size=len(clusters), replace=True)
                for i in by_cluster[c]]
        draws.append(float(np.mean(correct[take] - chance[take])))
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (float("nan"),) * 2
    return {
        "bootstrap_draws": draws,
        "n_rollouts": len(rollouts), "n_clusters": len(clusters),
        "accuracy": 100.0 * float(correct.mean()),
        "chance": 100.0 * float(chance.mean()),
        "margin_over_chance": 100.0 * float(np.mean(correct - chance)),
        "margin_ci95": [100.0 * float(lo), 100.0 * float(hi)],
        "used_tool": 100.0 * float(np.mean([r["used_tool"] for r in rollouts])),
        "unparsed": 100.0 * float(np.mean([r["answer"] == "" for r in rollouts])),
        "mean_turns": float(np.mean([r["turns"] for r in rollouts])),
        "second_smallest_rate": _rule_rate(rollouts),
        "by_tool": _by_tool(rollouts)}


def _by_tool(rollouts) -> dict:
    """Accuracy split by whether the rollout ever ran code.

    A margin carried by the rollouts that computed something is the exploit
    being executed; one spread evenly across both is not.
    """
    out = {}
    for name, want in (("used", True), ("not_used", False)):
        part = [r for r in rollouts if r["used_tool"] is want]
        out[name] = {"n": len(part), "accuracy": (
            100.0 * float(np.mean([r["answer"] == r["gold"] for r in part]))
            if part else float("nan"))}
    return out


def _rule_rate(rollouts) -> float:
    """How often the agent's pick is the second-smallest option, where that is defined.

    Not a claim about intent: it is the footprint the rule would leave.
    """
    from mcq_audit import parse_number
    hits, seen = 0, 0
    for r in rollouts:
        block = re.findall(r"\(([A-Z])\)\s*(.+)", r["user"])
        values = [(letter, parse_number(text.strip())) for letter, text in block]
        if any(v is None for _, v in values) or len(values) < 2:
            continue
        ranked = sorted(values, key=lambda t: t[1])
        seen += 1
        hits += int(r["answer"] == ranked[1][0])
    return 100.0 * hits / seen if seen else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--clean", default=None, help="matched clean file (built if absent)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default=None,
                    help="comma-separated GPU indices to spread the weights over")
    ap.add_argument("--shard-cap", default="38GiB",
                    help="per-GPU ceiling when --shard is used; one value for "
                         "every card, or one per card, comma-separated in the "
                         "same order as --shard")
    ap.add_argument("--shard-auto", type=float, default=0.0,
                    help="instead of a fixed cap, give each sharded card this "
                         "fraction of what is FREE on it right now. The cards "
                         "here carry other people's jobs, so a fixed ceiling is "
                         "either too small to hold a 70B or large enough to "
                         "evict them.")
    ap.add_argument("--condition", default="withheld_aware", choices=sorted(SYSTEM))
    ap.add_argument("--no-tools", action="store_true",
                    help="same loop and same instruction with no Python interpreter")
    ap.add_argument("--argmax", action="store_true",
                    help="same prompt, read one letter at a time instead of generated")
    ap.add_argument("--prompt", default="agent", choices=("agent", "bixbench"),
                    help="the scaffold's own framing, or BixBench's verbatim MCQ prompt")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default=None,
                    help="write one JSON line per rollout, for post-hoc analysis")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in open(args.items)]
    if args.limit:
        rows = rows[:args.limit]
    clean_rows = None
    if args.clean:
        clean_rows = [json.loads(l) for l in open(args.clean)][:len(rows)]

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.shard:
        # A 32B in bf16 does not fit beside another job on one card, so spread it
        # over the cards named here with a cap that leaves the others room.
        cards = [int(c) for c in args.shard.split(",")]
        caps = [c.strip() for c in args.shard_cap.split(",")]
        if args.shard_auto:
            # Read each card's free memory now and take a fraction of it. The
            # cards are shared, so a ceiling written down yesterday is either too
            # small to hold the model or large enough to push a neighbour out.
            caps = []
            for card in cards:
                free, _ = torch.cuda.mem_get_info(card)
                caps.append(f"{int(args.shard_auto * free / 2 ** 30)}GiB")
        elif len(caps) == 1:
            caps = caps * len(cards)
        if len(caps) != len(cards):
            raise SystemExit(f"--shard names {len(cards)} cards and --shard-cap "
                             f"gives {len(caps)} ceilings")
        print("sharding over " + ", ".join(f"cuda:{c} <= {m}"
                                           for c, m in zip(cards, caps)),
              flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            max_memory=dict(zip(cards, caps))).eval()
        args.device = f"cuda:{cards[0]}"       # where the batch's tensors go
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16).to(args.device).eval()

    runner = play_argmax if args.argmax else play
    # Which stack loaded the weights. Gemma-3 and OLMo-2 need a transformers
    # newer than the one every other arm here ran under, and an arm whose
    # library version is not on the record cannot be compared with one whose is.
    import transformers as _tf
    payload = {"model": args.model, "items": args.items, "condition": args.condition,
               "transformers": _tf.__version__, "torch": torch.__version__,
               "dump": args.dump,
               "readout": "argmax" if args.argmax else "generated",
               "prompt": args.prompt,
               "tools": (not args.no_tools) and not args.argmax, "system": (
                   PREAMBLE[args.condition]
                   + (PLAIN_RULES if args.no_tools else TOOL_RULES)).format(
                       turns=args.max_turns),
               "draws": args.draws, "max_turns": args.max_turns, "n_items": len(rows)}
    played = {"file": runner(model, tok,
                             build_rollouts(rows, args.condition, args.draws, args.seed, args.prompt), args)}
    payload["file"] = summarise(played["file"], args.bootstrap, args.seed)
    if clean_rows:
        played["clean"] = runner(
            model, tok,
            build_rollouts(clean_rows, args.condition, args.draws, args.seed + 1, args.prompt), args)
        payload["clean"] = summarise(played["clean"], args.bootstrap, args.seed + 1)
        payload["margin_over_clean"] = (payload["file"]["margin_over_chance"]
                                        - payload["clean"]["margin_over_chance"])
        # The two arms hold different items, so their cluster bootstraps are
        # independent and the difference of paired draws is the interval for
        # the difference. Both draw sequences use the same seed and length.
        difference = [a - b for a, b in zip(payload["file"].pop("bootstrap_draws"),
                                            payload["clean"].pop("bootstrap_draws"))]
        payload["margin_over_clean_ci95"] = [
            100.0 * float(np.percentile(difference, 2.5)),
            100.0 * float(np.percentile(difference, 97.5))]
    else:
        payload["file"].pop("bootstrap_draws", None)

    if args.dump:
        opener = gzip.open if args.dump.endswith(".gz") else open
        with opener(args.dump, "wt") as fh:
            for arm, rollouts in played.items():
                for r in rollouts:
                    fh.write(json.dumps({
                        "arm": arm, "item": r["item"], "cluster": r["cluster"],
                        "k": r["k"], "gold": r["gold"], "answer": r["answer"],
                        "turns": r["turns"], "used_tool": r["used_tool"],
                        "user": r["user"],
                        "reply": [m["content"] for m in r["messages"]
                                  if m["role"] == "assistant"]}) + "\n")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / args.out
    merged = json.loads(out.read_text()) if out.exists() else {}
    # The key names the file as well as the arm. Without it, the same cell run
    # on two benchmarks into one --out is one key, and the second run silently
    # deletes the first: Qwen2.5-72B's two MMLU-Pro cells were overwritten by
    # its two BixBench cells before this line said `items`. Nothing was lost --
    # the dumps are per-file -- but the result file quietly held four arms'
    # worth of work as two.
    key = (f"{args.model}|{pathlib.Path(args.items).stem}|{args.condition}"
           + ("|argmax" if args.argmax else "|notools" if args.no_tools else "")
           + ("|bixprompt" if args.prompt == "bixbench" else ""))
    if key in merged:
        print(f"note: replacing an existing entry for {key}", flush=True)
    merged[key] = payload
    out.write_text(json.dumps(merged, indent=1) + "\n")
    print(f"wrote {out}\n" + json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
