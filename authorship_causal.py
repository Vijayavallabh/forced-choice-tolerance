#!/usr/bin/env python3
"""Who wrote the distractor, held as the only difference between two files.

``key_identity.py`` finds the key in released MMLU-Pro and not in released MMLU,
MedMCQA or AQuA-RAT, and MMLU-Pro is the one whose extra options a model wrote.
That is one released file, so the positive half of the prediction rests on a
single observation we did not control.

This builds the controlled version. Both arms start from the same MMLU item: the
same key, the same three human-written distractors, augmented to ten options in
the same shape, which is how MMLU-Pro was made from MMLU. They differ in one
thing.

    generated   the six added options are written by an open-weight model, shown
                the key and the three human distractors and asked to match them
    borrowed    the six added options are human-written options lifted from
                other items in the same subject

If authorship is what a one-option-at-a-time solver reads, ``generated`` clears
its clean control and ``borrowed`` does not. Nothing else about the two files
differs, so a difference cannot be the option count, the key, the subject mix or
the augmentation itself.

    python3 authorship_causal.py --jsonl build/mmlu.jsonl \
        --model Qwen/Qwen2.5-14B-Instruct --device cuda:2
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from option_artifacts import parse_number
from repair_frontier import parse_values

ASK = ("Here are the four answer options of a multiple-choice question. "
       "Option 1 is correct; the rest are distractors written by the question's "
       "author.\n\n{options}\n\nWrite six more distractors for the same question, "
       "in the same style and the same format as the ones above, each a plausible "
       "wrong answer. Reply with the six values only, one per line, and nothing "
       "else.")


def numeric_rows(path, k=4):
    """The items with k distinct numeric options, in file order."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        options = [str(row["ideal"]).strip()] + [str(d).strip() for d in row["distractors"]]
        if len(options) != k:
            continue
        values = [parse_number(o) for o in options]
        if any(v is None for v in values) or len({float(v) for v in values}) != k:
            continue
        rows.append(row)
    return rows


def parse_six(text, existing):
    """Six distinct numeric strings the model wrote, none already present.

    Shares ``repair_frontier.parse_values`` rather than repeating it: the two
    had the same list-marker bug, which stripped a value's own leading digits
    and turned "1,2,3" into ",2,3".
    """
    return parse_values(text, existing, 6)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--generated-out", default="build/mmlu_aug_generated.jsonl")
    ap.add_argument("--borrowed-out", default="build/mmlu_aug_borrowed.jsonl")
    ap.add_argument("--report", default="results/authorship_causal.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = numeric_rows(args.jsonl)
    print(f"{len(rows)} four-option numeric items")

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    prompts = []
    for row in rows:
        options = [str(row["ideal"]).strip()] + [str(d).strip() for d in row["distractors"]]
        listing = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": ASK.format(options=listing)}],
            tokenize=False, add_generation_prompt=True))

    written = []
    for start in range(0, len(prompts), args.batch_size):
        chunk = prompts[start:start + args.batch_size]
        batch = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for i in range(len(chunk)):
            reply = tok.decode(out[i][batch["input_ids"].shape[1]:],
                               skip_special_tokens=True)
            written.append(reply)
        print(f"  generated {min(start + args.batch_size, len(prompts))}"
              f"/{len(prompts)}", flush=True)

    # Options a person wrote, pooled by subject, to draw the borrowed arm from.
    human_pool = defaultdict(list)
    for row in rows:
        cluster = row.get("cluster") or ""
        for option in [str(row["ideal"]).strip()] + [str(d).strip() for d in row["distractors"]]:
            human_pool[cluster].append(option)

    rng = random.Random(args.seed)
    generated, borrowed, kept, short = [], [], 0, 0
    for row, reply in zip(rows, written):
        options = [str(row["ideal"]).strip()] + [str(d).strip() for d in row["distractors"]]
        six = parse_six(reply, options)
        if len(six) < 6:
            short += 1
            continue
        pool = [o for o in human_pool.get(row.get("cluster") or "", [])
                if o not in options]
        if len(pool) < 6:
            short += 1
            continue
        loan, seen = [], set(options)
        for option in rng.sample(pool, min(len(pool), 60)):
            if option not in seen:
                seen.add(option)
                loan.append(option)
            if len(loan) == 6:
                break
        if len(loan) < 6:
            short += 1
            continue
        kept += 1
        generated.append({"ideal": row["ideal"],
                          "distractors": row["distractors"] + six,
                          "question": "", "cluster": row.get("cluster", "")})
        borrowed.append({"ideal": row["ideal"],
                         "distractors": row["distractors"] + loan,
                         "question": "", "cluster": row.get("cluster", "")})

    for path, arm in ((args.generated_out, generated), (args.borrowed_out, borrowed)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("".join(json.dumps(r) + "\n" for r in arm),
                              encoding="utf-8")
        print(f"wrote {path}: {len(arm)} rows")

    report = {"model": args.model, "n_source_items": len(rows), "n_kept": kept,
              "n_dropped": short, "seed": args.seed,
              "arms": {"generated": args.generated_out,
                       "borrowed": args.borrowed_out},
              "design": ("same key, same three human distractors, augmented to ten "
                         "options; the six added options are model-written in one "
                         "arm and human-written in the other")}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.report}: kept {kept}, dropped {short}")


if __name__ == "__main__":
    main()
