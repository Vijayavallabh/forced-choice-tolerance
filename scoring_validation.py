#!/usr/bin/env python3
"""Check that the probe's forced-choice read-out predicts what the model writes.

``no_data_probe.py`` does not sample text. It pre-fills the ``<answer>`` tag the
benchmark's own template asks for and takes the arg-max over the four letter
logits at that position. This is deterministic and cheap, but it is only a fair
stand-in if it agrees with the letter the model would actually emit.

Two quantities are reported per model. *Compliance* is how often greedy
generation produces a parsable letter at all; a model that answers a
multiple-choice prompt with the bare number instead is not refusing so much as
ignoring the format, and BixBench's own no-data runs contain such responses.
*Agreement* is how often, among compliant cases, the emitted letter matches the
arg-max. Forcing a choice is faithful to the condition being replicated --
BixBench's prompt says "You must pick one answer even if you are unsure" -- but
the reader should know how often the force was needed.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from no_data_probe import (ARMS, LETTERS, build_conditions, letter_token_ids,
                           load_items, render)

PANEL = [
    "meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct", "microsoft/Phi-3.5-mini-instruct",
    "google/gemma-3-4b-it", "Qwen/Qwen2.5-7B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Groq/Llama-3-Groq-8B-Tool-Use", "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]


def first_letter(text):
    """The choice a lenient parser would extract, or None.

    The whole string must not be upper-cased first: ``</answer>`` upper-cases to
    ``</ANSWER>``, whose leading 'A' would be read as a choice. That mistake
    scored this check at 50% agreement before it was caught.
    """
    stripped = text.strip()
    for candidate in (stripped, stripped.lstrip("(<[ \t\n*")):
        if candidate and candidate[0].upper() in LETTERS:
            return candidate[0].upper()
    return None


def validate(name, records, batch_size, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids, variant = letter_token_ids(tokenizer)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                     device_map=device)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16,
                                                            device_map=device)
    model.eval()

    compliant = agree = 0
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            batch = tokenizer([render(tokenizer, r["prompt"]) for r in chunk],
                              return_tensors="pt", padding=True,
                              add_special_tokens=False).to(device)
            picks = [LETTERS[int(i)] for i in
                     model(**batch).logits[:, -1, :].float()[:, ids].argmax(dim=-1)]
            generated = model.generate(**batch, max_new_tokens=8, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(generated[:, batch["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
            for pick, text in zip(picks, texts):
                letter = first_letter(text)
                if letter is not None:
                    compliant += 1
                    agree += (letter == pick)
    del model
    torch.cuda.empty_cache()
    return {"model": name, "letter_variant": variant.format("A"),
            "n_sampled": len(records), "n_compliant": compliant,
            "compliance_rate": compliant / len(records),
            "agreement_with_forced_choice": agree / max(compliant, 1), "n_agree": agree}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=PANEL)
    parser.add_argument("--jsonl", default="data/bixbench.jsonl")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("results/scoring_validation.json"))
    args = parser.parse_args()

    items, k = load_items(args.jsonl)
    records = build_conditions(items, 2, args.seed, k)
    random.Random(args.seed).shuffle(records)
    records = records[:args.sample]

    rows = []
    for name in args.models:
        try:
            rec = validate(name, records, args.batch_size, args.device)
        except Exception as exc:                       # noqa: BLE001 - reported, not hidden
            rec = {"model": name, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(rec)
        print(json.dumps(rec), flush=True)

    ok = [r for r in rows if "agreement_with_forced_choice" in r]
    doc = {"note": __doc__.strip().split("\n")[0],
           "sample_conditions": args.sample, "arms_sampled": list(ARMS),
           "models": rows,
           "summary": {"n_models": len(ok),
                       "min_agreement": min((r["agreement_with_forced_choice"] for r in ok),
                                            default=None),
                       "min_compliance": min((r["compliance_rate"] for r in ok),
                                             default=None)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
