#!/usr/bin/env python3
r"""Repairs that condition on the key, and repairs that do not.

The paper's earlier claim was that a repair cannot close the channel, and it
was too strong. Both repairs it tested condition on the key: ``mcq_audit.py``
redraws each distractor around the keyed value and ``exchangeable_repair.py``
multiplies the key by i.i.d. offsets. A repair that never looks at the key
cannot make the option set a function of it, so the conserved quantity has to
be stated over an operator class rather than over repairs in general.

This builds the operators that span the class, on the same items, with the same
key, so the only thing that varies is what the distractor-writing process is
allowed to see.

    released         what the benchmark shipped
    rank uniform     key-conditioned, style-preserving, rank-targeted
    exchangeable     key-conditioned multiplicative offsets
    imitation        key- and question-conditioned: a model is shown the
                     question, the keyed answer and the human distractors and
                     asked for replacements in the same style
    key marginal     not conditioned on the key at all: the distractors are
                     *keys of other items in the same subject*, drawn from the
                     empirical multiset so a value keyed ten times is ten times
                     as likely, so they are human-written by construction and
                     carry the subject's own distribution of written forms
    key marginal, near
                     the same draw restricted to keys within a factor of ten of
                     this key, which buys plausibility back by conditioning on
                     the key's magnitude and nothing else

Each operator is scored two ways, because the interesting quantity is a
trade-off and not a number: what a no-data solver reads off the options
(``key_identity.py``), and what a solver *given the question* scores, which
rises when a repair has made the distractors too easy to reject.

    python3 repair_frontier.py --jsonl build/mmlu_questions.jsonl \
        --model Qwen/Qwen2.5-14B-Instruct --device cuda:1
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from option_artifacts import parse_number

ASK = ("Here is a multiple-choice question and its four answer options.\n\n"
       "Question: {question}\n\n{options}\n\nOption 1 is the correct answer. "
       "Write three new distractors to replace options 2, 3 and 4: plausible "
       "wrong answers to this question, written in the same style and the same "
       "format as the options above. Reply with the three values only, one per "
       "line, and nothing else.")


def numeric_rows(path, k=4, need_question=True):
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
        if need_question and not str(row.get("question", "")).strip():
            continue
        rows.append(row)
    return rows


# A leading list marker, and nothing more. Stripping a *character class* here
# instead (``lstrip("-*0123456789.) ")``) also eats the value's own leading
# digits, so "1,2,3" came back as ",2,3": a leading comma that appears only on
# distractors and so makes the key easier to find, which is the opposite of
# harmless. It mangled 21% of replies before it was caught.
# The bullet must be followed by space, or "-14" loses its sign.
LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]\s+)?(?:\d{1,2}\s*[.)]\s+)?")
# A number must start with a digit, a sign or a point, never with a separator.
NUMBER = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s*%?")


def parse_values(text, existing, want):
    """`want` distinct numeric strings the model wrote, none already present."""
    out, seen = [], {str(e).strip() for e in existing}
    for line in text.splitlines():
        stripped = LIST_MARKER.sub("", line.strip(), count=1) or line.strip()
        for token in NUMBER.findall(stripped):
            token = token.strip()
            if not token or token in seen or parse_number(token) is None:
                continue
            seen.add(token)
            out.append(token)
            break
        if len(out) == want:
            break
    return out


def draw_distinct(pool, want, forbidden, rng):
    """`want` values drawn from the multiset `pool`, distinct from each other.

    Rejection sampling rather than ``random.sample`` over the distinct values,
    because the whole point of this arm is that the distractors carry the
    subject's *empirical* distribution of keys.
    """
    chosen, seen = [], set(forbidden)
    for _ in range(4000):
        candidate = rng.choice(pool)
        if candidate in seen:
            continue
        seen.add(candidate)
        chosen.append(candidate)
        if len(chosen) == want:
            return chosen
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", default="build/mmlu_questions.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--near-factor", type=float, default=10.0)
    ap.add_argument("--outdir", default="build")
    ap.add_argument("--generations", default="build/mmlu_frontier_generations.json",
                    help="cache of the model's replies; reused when it exists, "
                         "so the borrowed arms can be rebuilt without a GPU")
    ap.add_argument("--report", default="results/repair_frontier.json")
    args = ap.parse_args()

    rows = numeric_rows(args.jsonl)
    print(f"{len(rows)} four-option numeric items carrying question text")

    cache = Path(args.generations)
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("n_source_items") != len(rows) or stored.get("model") != args.model:
            raise SystemExit(f"{cache} was written for a different run; delete it to regenerate")
        written = stored["replies"]
        print(f"reusing {len(written)} cached replies from {cache}")
        return finish(args, rows, written)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
            [{"role": "user", "content": ASK.format(
                question=str(row["question"]).strip(), options=listing)}],
            tokenize=False, add_generation_prompt=True))

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
    cache.write_text(json.dumps({"model": args.model,
                                 "n_source_items": len(rows),
                                 "replies": written}, indent=1), encoding="utf-8")
    print(f"wrote {cache}")
    return finish(args, rows, written)


def finish(args, rows, written):
    """Build the arms from the model's replies. No GPU past this point."""
    # Keys of other items, pooled by subject: the draw that never sees this key.
    keys_by_cluster = defaultdict(list)
    for row in rows:
        keys_by_cluster[row.get("cluster") or ""].append(str(row["ideal"]).strip())

    rng = random.Random(args.seed)
    arms = {name: [] for name in
            ("imitation", "key_marginal", "key_marginal_near", "released_matched")}
    kept, drop_short, drop_pool, drop_near = 0, 0, 0, 0

    for row, reply in zip(rows, written):
        key = str(row["ideal"]).strip()
        human = [str(d).strip() for d in row["distractors"]]
        options = [key] + human
        cluster = row.get("cluster") or ""
        key_value = float(parse_number(key))

        three = parse_values(reply, options, 3)
        if len(three) < 3:
            drop_short += 1
            continue

        # Draw from the multiset, not from its distinct values: deduplicating
        # would reweight the subject's keys towards its rare written forms (on
        # MMLU it lifts the share carrying a percent sign from 15.4% to 18.8%),
        # and that reweighting is itself readable one option at a time.
        pool = [k for k in keys_by_cluster[cluster] if k not in options]
        if len(set(pool)) < 3:
            drop_pool += 1
            continue
        loan = draw_distinct(pool, 3, options, rng)

        def near(candidate):
            value = parse_number(candidate)
            if value is None:
                return False
            value = float(value)
            if key_value == 0 or value == 0:
                return value == key_value
            ratio = abs(value) / abs(key_value)
            return 1.0 / args.near_factor <= ratio <= args.near_factor

        near_pool = [k for k in pool if near(k)]
        if len(set(near_pool)) < 3:
            drop_near += 1
            continue
        loan_near = draw_distinct(near_pool, 3, options, rng)

        kept += 1
        base = {"question": str(row.get("question", "")), "cluster": cluster}
        arms["released_matched"].append({"ideal": key, "distractors": human, **base})
        arms["imitation"].append({"ideal": key, "distractors": three, **base})
        arms["key_marginal"].append({"ideal": key, "distractors": loan, **base})
        arms["key_marginal_near"].append({"ideal": key, "distractors": loan_near, **base})

    paths = {}
    for name, records in arms.items():
        path = str(Path(args.outdir) / f"mmlu_frontier_{name}.jsonl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("".join(json.dumps(r) + "\n" for r in records),
                              encoding="utf-8")
        paths[name] = path
        print(f"wrote {path}: {len(records)} rows")

    report = {
        "model": args.model, "seed": args.seed, "near_factor": args.near_factor,
        "n_source_items": len(rows), "n_kept": kept,
        "n_dropped_short_generation": drop_short,
        "n_dropped_small_subject_pool": drop_pool,
        "n_dropped_no_near_keys": drop_near,
        "arms": paths,
        "design": ("same items and same key in every arm; the arms differ only in "
                   "what the distractor-writing process was allowed to condition on"),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.report}: kept {kept}, dropped "
          f"{drop_short}/{drop_pool}/{drop_near}")


if __name__ == "__main__":
    main()
