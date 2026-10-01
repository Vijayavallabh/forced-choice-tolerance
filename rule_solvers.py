#!/usr/bin/env python3
"""Score model-free solvers on exactly the conditions the models saw.

The panel of language models turns out to be unaffected by the repair. That is
what the decomposition predicts for a solver whose rank preference is flat --
``G = <p, b>`` cannot move when ``b`` is uniform, however far ``p`` moves -- but
a null is only informative if the design could have shown something else.

So the same arms, items, draws and inference are applied to solvers whose rank
preference is known by construction:

  rank-0 .. rank-3   point mass on one sorted rank; the exploit an audit finds
  interior           mass split over the two middle ranks
  uniform-random     flat preference, the negative control
  longest-option     a surface rule, blind to the numbers entirely

The point-mass solvers must show a large drop under the repair and the flat one
must show none. Those two outcomes calibrate the panel's null: if the positive
control moves by twenty-six points through this pipeline and the models do not
move at all, the models really are not using the channel.

Output is written in the same schema as ``no_data_probe.py``, so every
downstream step treats these as just more solvers.
"""
import argparse
import json
import random
from pathlib import Path

from no_data_probe import (ARMS, build_conditions, letters_for, load_items,
                           open_outcomes)
from option_artifacts import N_OPTIONS
from text_artifacts import RULES as SURFACE_RULES
from text_artifacts import pick as surface_pick


def rank_rule(rank):
    def choose(record, rng):
        slots = record["slot_ranks"]
        if slots and rank in slots:
            return slots.index(rank)
        return rng.randrange(len(record["options_shown"]))
    return choose


def interior_rule(record, rng):
    slots = record["slot_ranks"]
    n = len(record["options_shown"])
    if not slots:
        return rng.randrange(n)
    middle = [i for i, r in enumerate(slots) if 0 < r < n - 1]
    return rng.choice(middle) if middle else rng.randrange(n)


def uniform_rule(record, rng):
    return rng.randrange(len(record["options_shown"]))


def surface_rule(name):
    def choose(record, rng):
        return surface_pick(SURFACE_RULES[name], record["options_shown"],
                            record["question_text"])
    return choose


def avoid_rule(name):
    """Pick uniformly among the options a surface rule does *not* choose.

    A rule that is reliably *wrong* is as much of a leak as one that is
    reliably right: excluding one option lifts a guess from 1/4 to 1/3. Rules
    are therefore scored in both directions.
    """
    def choose(record, rng):
        avoided = surface_pick(SURFACE_RULES[name], record["options_shown"],
                               record["question_text"])
        n = len(record["options_shown"])
        return rng.choice([i for i in range(n) if i != avoided])
    return choose


# The rank family is what the paper uses: point masses bracket the achievable
# range, "interior" is the audit's own exploit, and the uniform guesser is the
# negative control. These are the files kept under version control.
RANK_FAMILY = {
    "rule:rank-0": rank_rule(0), "rule:rank-1": rank_rule(1),
    "rule:rank-2": rank_rule(2), "rule:rank-3": rank_rule(3),
    "rule:interior": interior_rule, "rule:uniform-random": uniform_rule,
}
# The surface family, in both directions, is exploratory: it established that
# these cues track value rank rather than forming a second channel (see
# text_artifacts.py). Regenerate with --family all; the outcomes are not
# committed because no reported number depends on them.
SURFACE_FAMILY = {f"surface:{name.replace('_', '-')}": surface_rule(name)
                  for name in SURFACE_RULES}
SURFACE_FAMILY.update({f"avoid:{name.replace('_', '-')}": avoid_rule(name)
                       for name in SURFACE_RULES})
SOLVERS = dict(RANK_FAMILY)
FAMILIES = {"rank": RANK_FAMILY, "surface": SURFACE_FAMILY,
            "all": {**RANK_FAMILY, **SURFACE_FAMILY}}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", default="data/bixbench.jsonl")
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260920)
    # A separate directory from the language models: the two are analysed
    # separately, and mixing them would put rules into the model panel.
    parser.add_argument("--out", default="results/probe_rules")
    parser.add_argument("--family", choices=sorted(FAMILIES), default="rank",
                        help="which model-free solvers to score (default: rank)")
    args = parser.parse_args()

    items, k = load_items(args.jsonl)
    records = build_conditions(items, args.draws, args.seed, k)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, solver in FAMILIES[args.family].items():
        rows = []
        for record in records:
            # Seeded per condition so a tie-breaking draw is reproducible and
            # independent of the order solvers are run in.
            rng = random.Random(f"{args.seed}|{name}|{record['question_id']}|"
                                f"{record['arm']}|{record['draw']}")
            slot = solver(record, rng)
            chosen = letters_for(k)[slot]
            rows.append({
                "model": name, "question_id": record["question_id"],
                "cluster": record["cluster"], "arm": record["arm"],
                "draw": record["draw"], "is_numeric": record["is_numeric"],
                "target_letter": record["target_letter"], "predicted_letter": chosen,
                "correct": int(chosen == record["target_letter"]),
                "key_rank": record["key_rank"], "released_rank": record["released_rank"],
                "redraw_applied": record["redraw_applied"],
                "chosen_rank": record["slot_ranks"][slot] if record["slot_ranks"] else None,
                "letter_probs": None,
            })
        path = out / (name.replace("/", "__").replace(":", "_") + ".jsonl.gz")
        with open_outcomes(path, "wt") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        hits = sum(r["correct"] for r in rows)
        print(f"{name:24s} -> {path.name}  ({hits}/{len(rows)} = {hits / len(rows):.1%} pooled)")


if __name__ == "__main__":
    main()
