#!/usr/bin/env python3
"""What the agent says it is doing, and whether saying it pays.

`agentic_probe.py --dump` writes one record per rollout, including the
assistant's own text. This script splits those records on whether the agent's
reasoning reaches for the middle of the option set --- the verbal form of the
rank rule, since BixBench's key is the second-smallest of four --- and scores
each side against its own cluster bootstrap, on the released file and on the
matched clean file.

It is a post-hoc split on text the model wrote, not a controlled arm: it says
which rollouts carry the margin, not that the wording caused it. The control
is the clean file, where the same wording is available and buys nothing.
"""
import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np

from mcq_audit import parse_number

RESULTS = Path("results")

# The rank rule as a sentence rather than as an index. BixBench's distractors
# are perturbations that bracket the true value, so "the middle one" and "the
# second smallest" name the same option; the agent never writes the second.
MIDDLE = re.compile(r"\bmedian\b|\bmiddle\b|\bcentral\b")
OPTION = re.compile(r"\(([A-Z])\) (.+)")


def value_ranks(rollout):
    """(rank of the key, rank of the pick) among the option values, or None.

    Rank is over the sorted values, which is what eq (1) is written in: the
    letter the option carries is not part of it.
    """
    pairs = [(letter, parse_number(text.strip()))
             for letter, text in OPTION.findall(rollout["user"])]
    if len(pairs) != rollout["k"] or any(v is None for _, v in pairs):
        return None
    order = [letter for letter, _ in sorted(pairs, key=lambda t: t[1])]
    if rollout["answer"] not in order:
        return None
    return order.index(rollout["gold"]), order.index(rollout["answer"])


def rank_fit(rollouts, k, draws, seed):
    """Fit eq (1)'s geometry term to the agent: p from the file, b from its picks.

    With no question and no data the recall term is zero, so <p,b> - 1/k is the
    whole predicted margin. Reporting it beside the observed accuracy says how
    much of what the agent takes is the rank channel and how much is not.
    """
    scored = [(r, value_ranks(r)) for r in rollouts]
    scored = [(r, jr) for r, jr in scored if jr is not None]
    if not scored:
        return None
    clusters = sorted({r["cluster"] for r, _ in scored})
    index = {c: [i for i, (r, _) in enumerate(scored) if r["cluster"] == c] for c in clusters}

    def terms(rows):
        p = np.bincount([j for _, (j, _) in rows], minlength=k) / len(rows)
        b = np.bincount([r for _, (_, r) in rows], minlength=k) / len(rows)
        return p, b, float(p @ b) - 1.0 / k

    p, b, geometry = terms(scored)
    rng = np.random.default_rng(seed)
    spread = [terms([scored[i] for c in rng.choice(clusters, len(clusters), replace=True)
                     for i in index[c]])[2] for _ in range(draws)]
    return {"n": len(scored),
            "key_rank": [100.0 * float(v) for v in p],
            "pick_rank": [100.0 * float(v) for v in b],
            "geometry": 100.0 * geometry,
            "geometry_ci95": [100.0 * float(np.percentile(spread, 2.5)),
                              100.0 * float(np.percentile(spread, 97.5))],
            "observed_margin_over_chance":
                100.0 * float(np.mean([j == r for _, (j, r) in scored])) - 100.0 / k}


def bootstrap(part, draws, seed):
    """Accuracy and a cluster-bootstrap interval over the capsules present."""
    if not part:
        return {"n": 0, "accuracy": float("nan"), "ci95": [float("nan")] * 2}
    correct = np.array([1.0 if r["answer"] == r["gold"] else 0.0 for r in part])
    clusters = sorted({r["cluster"] for r in part})
    index = {c: [i for i, r in enumerate(part) if r["cluster"] == c] for c in clusters}
    rng = np.random.default_rng(seed)
    means = [float(np.mean(correct[[i for c in rng.choice(clusters, len(clusters), replace=True)
                                    for i in index[c]]])) for _ in range(draws)]
    return {"n": len(part), "accuracy": 100.0 * float(correct.mean()),
            "ci95": [100.0 * float(np.percentile(means, 2.5)),
                     100.0 * float(np.percentile(means, 97.5))]}


def split(rollouts, draws, seed):
    """Score the rollouts that reach for the middle against the ones that do not."""
    hit = [bool(MIDDLE.search(" ".join(r["reply"]).lower())) for r in rollouts]
    says = [r for r, h in zip(rollouts, hit) if h]
    rest = [r for r, h in zip(rollouts, hit) if not h]
    return {"all": bootstrap(rollouts, draws, seed),
            "says_middle": bootstrap(says, draws, seed),
            "does_not": bootstrap(rest, draws, seed),
            "reaches_for_it": 100.0 * len(says) / len(rollouts) if rollouts else float("nan")}


# A reply short enough to be nothing but the answer line. "FINAL: C" is eight
# characters; twenty leaves room for a stray word and none for an argument.
BARE = 20


def jsonable(value):
    """The payload with every NaN turned into null.

    An arg-max arm generates no text, so the ``wrote`` split is empty and its
    accuracy, its interval and the difference against the clean file are all
    NaN. ``NaN`` is not JSON -- Python's own parser reads it back, and nothing
    else does -- and this file ships in the evidence bundle, so an empty split
    is recorded as null rather than as a token only one language accepts.
    """
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def wrote_first(rollouts, draws, seed):
    """Score the rollouts that wrote something before answering against the rest.

    Taking the interpreter away does not turn the loop into a reasoning pass --
    on MMLU-Pro most rollouts answer with the final line and nothing else -- so
    whether the margin needs the writing is a question this settles rather than
    assumes.
    """
    wrote = [r for r in rollouts if len(" ".join(r["reply"]).strip()) >= BARE]
    bare = [r for r in rollouts if len(" ".join(r["reply"]).strip()) < BARE]
    return {"wrote": bootstrap(wrote, draws, seed),
            "answered_bare": bootstrap(bare, draws, seed),
            "bare_share": 100.0 * len(bare) / len(rollouts) if rollouts else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="the JSONL agentic_probe.py --dump wrote")
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="agentic_mechanism.json")
    args = ap.parse_args()

    opener = gzip.open if args.dump.endswith(".gz") else open
    rows = [json.loads(line) for line in opener(args.dump, "rt")]
    payload = {"dump": args.dump, "n_rollouts": len(rows)}
    for arm in ("file", "clean"):
        part = [r for r in rows if r["arm"] == arm]
        payload[arm] = split(part, args.draws, args.seed)
        payload[arm]["rank_fit"] = rank_fit(part, part[0]["k"], args.draws, args.seed)
        payload[arm]["by_writing"] = wrote_first(part, args.draws, args.seed)
    payload["margin_when_said"] = (payload["file"]["says_middle"]["accuracy"]
                                   - payload["clean"]["says_middle"]["accuracy"])
    payload["margin_when_not"] = (payload["file"]["does_not"]["accuracy"]
                                  - payload["clean"]["does_not"]["accuracy"])
    payload["margin_when_wrote"] = (payload["file"]["by_writing"]["wrote"]["accuracy"]
                                    - payload["clean"]["by_writing"]["wrote"]["accuracy"])
    payload["margin_when_bare"] = (payload["file"]["by_writing"]["answered_bare"]["accuracy"]
                                   - payload["clean"]["by_writing"]["answered_bare"]["accuracy"])
    payload = jsonable(payload)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / args.out
    merged = jsonable(json.loads(out.read_text())) if out.exists() else {}
    merged[args.dump] = payload
    out.write_text(json.dumps(merged, indent=1, allow_nan=False) + "\n")
    print(f"wrote {out}\n" + json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
