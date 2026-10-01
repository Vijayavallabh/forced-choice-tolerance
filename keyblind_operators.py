#!/usr/bin/env python3
r"""Distractor writers that never see the key, and writers that are not Qwen.

``repair_frontier.py`` spans one axis: whether the distractor-writing process
may condition on the keyed answer. Its key-blind arms borrow *other items'
keys*, which closes the channel and destroys the item, and its key-conditioned
arms all leak. That leaves the quadrant the survey says exists in the wild
unoccupied: AQuA-RAT's options come from exam-practice websites and crowdworkers
(its paper does not say how the distractors were chosen), none of them a
perturbation of the key, and AQuA-RAT's key rank is indistinguishable from
uniform. The operator here is the hypothesis that a writer conditioning on the
*question* and not on the key reproduces that.

This builds that operator.

    wrong step      the writer is shown the question and nothing else, and is
                    asked for three values a plausible mistake would produce.
                    It is key-blind by construction: the only thing it can know
                    about the key is what the question says, which is what
                    makes the item hard in the first place.

and, because "imitation is one generator at one prompt" is a fair objection to
the conclusion drawn from it, re-runs the key-conditioned arm under other
generators so the class claim is a claim about the class:

    imitation       the ``repair_frontier.py`` prompt, other models.

Collisions are the one place the assembler must look at the key: a wrong-step
value that equals the keyed answer is not a distractor, and a writer good
enough to be useful produces it often -- on a quarter of items here. The
*writer* never sees the key; the assembler asks for five values, drops any that
equal it and keeps the first three, which is what any builder does, and both
counts are reported. That filter is the operator's one dependence on the key
and it is a dependence of the *assembler*, not of the distractor's content.

    python3 keyblind_operators.py --operator wrong_step \
        --model Qwen/Qwen2.5-14B-Instruct --device cuda:0
"""
import argparse
import json
import re
from pathlib import Path

from option_artifacts import parse_number
from repair_frontier import ASK, NUMBER, numeric_rows, parse_values

# Key-blind: the question, and no option of any kind. Asking for the mistakes a
# student makes rather than for "wrong answers" is what keeps the values near
# the true one; a writer that cannot do this writes a file a solver can reject
# without reading the question, which is the cost this operator is measured on.
WRONG_STEP = (
    "Here is a question from a multiple-choice exam. You have not been told the "
    "correct answer, and you are not being asked for it.\n\n"
    "Question: {question}\n\n"
    "Work the problem out, then give five values that a student could reach by "
    "making a plausible mistake on it: using a wrong but tempting formula, "
    "dropping or inverting a factor, confusing two units, or stopping one step "
    "early. Write each value the way the answer to this question would be "
    "written. Keep your working to at most three short sentences, and end your "
    "reply with one line in exactly this form:\n\n"
    "ANSWERS: <value>; <value>; <value>; <value>; <value>")

# The writer is allowed to reason, so its reply is full of working, and the
# three values it means are the ones after the marker. Reading the first number
# on each line instead would collect the working.
MARKER = re.compile(r"^\s*ANSWERS\s*:(.*)$", re.IGNORECASE | re.MULTILINE)


def parse_marked(text, want=5):
    """The values on the last ANSWERS line, or None if the writer wrote none."""
    found = MARKER.findall(text)
    if not found:
        return None
    pieces = [piece.strip() for piece in found[-1].split(";")]
    out, seen = [], set()
    for piece in pieces:
        for token in NUMBER.findall(piece):
            token = token.strip()
            if token and token not in seen and parse_number(token) is not None:
                seen.add(token)
                out.append(token)
                break
        if len(out) == want:
            break
    return out

OPERATORS = {"wrong_step": WRONG_STEP, "imitation": ASK}


def build_prompt(operator, tok, row):
    if operator == "wrong_step":
        content = WRONG_STEP.format(question=str(row["question"]).strip())
    else:
        options = [str(row["ideal"]).strip()] + [str(d).strip() for d in row["distractors"]]
        listing = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
        content = ASK.format(question=str(row["question"]).strip(), options=listing)
    return tok.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True)


def generate(args, rows):
    cache = Path(args.generations)
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if (stored.get("n_source_items") != len(rows) or stored.get("model") != args.model
                or stored.get("operator") != args.operator):
            raise SystemExit(f"{cache} was written for a different run; delete it")
        print(f"reusing {len(stored['replies'])} cached replies from {cache}")
        return stored["replies"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    prompts = [build_prompt(args.operator, tok, row) for row in rows]
    written = []
    for start in range(0, len(prompts), args.batch_size):
        chunk = prompts[start:start + args.batch_size]
        batch = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        for i in range(len(chunk)):
            written.append(tok.decode(out[i][batch["input_ids"].shape[1]:],
                                      skip_special_tokens=True))
        print(f"  generated {min(start + args.batch_size, len(prompts))}/{len(prompts)}",
              flush=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"model": args.model, "operator": args.operator,
                                 "n_source_items": len(rows), "replies": written},
                                indent=1), encoding="utf-8")
    print(f"wrote {cache}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--operator", choices=sorted(OPERATORS), default="wrong_step")
    ap.add_argument("--jsonl", default="build/mmlu_questions.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=768,
                    help="wrong step asks the writer to work the problem, so it "
                         "needs room to do that before the three values")
    ap.add_argument("--tag", default=None,
                    help="suffix for the arm's file name; defaults to the "
                         "operator, or operator_model for a second generator")
    ap.add_argument("--outdir", default="build")
    ap.add_argument("--out", default=None,
                    help="the arm's path; defaults to <outdir>/mmlu_frontier_<tag>.jsonl. "
                         "Needed for any items file that is not MMLU's (BixBench's "
                         "numeric slice, for one).")
    ap.add_argument("--generations", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N items")
    ap.add_argument("--on-collision", choices=("drop-value", "drop-item"),
                    default="drop-value",
                    help="drop-value keeps the item and takes the writer's next "
                         "value, which is what a builder does; drop-item removes "
                         "the item instead, so no retained item was filtered at "
                         "all and the arm's leak cannot be the filter's doing")
    args = ap.parse_args()

    short = args.model.split("/")[-1].replace(".", "").replace("-", "_").lower()
    tag = args.tag or args.operator
    args.generations = args.generations or f"build/mmlu_{args.operator}_{short}_generations.json"
    args.report = args.report or f"results/keyblind_{tag}.json"

    rows = numeric_rows(args.jsonl)
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} four-option numeric items carrying question text")

    written = generate(args, rows)

    records, kept, drop_short, collisions, drop_unmarked = [], 0, 0, 0, 0
    drop_collided = 0
    for row, reply in zip(rows, written):
        key = str(row["ideal"]).strip()
        human = [str(d).strip() for d in row["distractors"]]
        # The writer's own reply, parsed without reference to the key; then the
        # assembler removes any value that IS the key, by string and by value.
        if args.operator == "wrong_step":
            candidates = parse_marked(reply, 5)
            if candidates is None:
                drop_unmarked += 1
                continue
        else:
            candidates = parse_values(reply, [key] + human, 6)
        key_value = parse_number(key)
        usable, hit = [], False
        for value in candidates:
            parsed = parse_number(value)
            same = value == key or (parsed is not None and key_value is not None
                                    and float(parsed) == float(key_value))
            if same:
                collisions += 1
                hit = True
                continue
            usable.append(value)
        if hit and args.on_collision == "drop-item":
            drop_collided += 1
            continue
        if len(usable) < 3:
            drop_short += 1
            continue
        kept += 1
        records.append({"ideal": key, "distractors": usable[:3],
                        "question": str(row.get("question", "")),
                        "cluster": row.get("cluster") or ""})

    path = Path(args.out) if args.out else Path(args.outdir) / f"mmlu_frontier_{tag}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    print(f"wrote {path}: {len(records)} rows")

    report = {"operator": args.operator, "model": args.model, "tag": tag,
              "prompt": OPERATORS[args.operator],
              "n_source_items": len(rows), "n_kept": kept,
              "n_dropped_short_generation": drop_short,
              "n_dropped_no_answers_line": drop_unmarked,
              "on_collision": args.on_collision,
              "n_dropped_for_collision": drop_collided,
              "n_key_collisions_removed": collisions,
              "arm": str(path),
              "conditions_on": ("the question alone" if args.operator == "wrong_step"
                                else "the question, the key and the human options")}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.report}: kept {kept}, dropped {drop_short} short and "
          f"{drop_unmarked} unmarked, removed {collisions} collisions with the key")


if __name__ == "__main__":
    main()
