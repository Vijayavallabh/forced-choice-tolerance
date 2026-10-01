#!/usr/bin/env python3
r"""A rank-holding placebo for any released file, built the way \S4's BixBench one was.

The placebo redraws every distractor's value while holding the key's rank among
them, so the option \emph{text} is new and the rank channel is exactly the
released file's. A margin that survives it is not recognition of a published
option set; one that dies on it was. \S4 has this arm on BixBench, and the file
which resolves the agentic claim -- MMLU-Pro, nine times the items -- needs it
too.

At ten options the redraw is much harder than at four: placing nine same-style
values around a key at an extreme rank often has no solution, so items where it
fails are reported rather than quietly dropped.

    python3 build_placebo_arm.py --items build/mmlu_pro_matched_released.jsonl \
        --out build/mmlu_pro_matched_placebo.jsonl
"""
import argparse
import json
import pathlib
import random

from mcq_audit import placebo_row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260920)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.items)]
    kept, missed = [], 0
    for row in rows:
        options = [row["ideal"]] + list(row["distractors"])
        drawn = placebo_row(options, rng)
        if drawn is None:
            missed += 1
            continue
        out = dict(row)
        out["ideal"], out["distractors"] = drawn[0], list(drawn[1:])
        kept.append(out)
    pathlib.Path(args.out).write_text(
        "".join(json.dumps(r) + "\n" for r in kept))
    print(f"{len(kept)} of {len(rows)} items redrawn with the key's rank held "
          f"({missed} had no reachable redraw)\nwrote {args.out}")


if __name__ == "__main__":
    main()
