#!/usr/bin/env python3
r"""The $\Gamma$ of the \textsc{key marginal} operator as built, not as idealised.

Proposition~\ref{prop:writers}(i) certifies an item-blind writer only when its
distractors are $\kappa$ conditioned on the whole $k$-tuple being distinct. The
operator ``repair_frontier.py`` builds does something else: it removes from the subject's key pool every value equal to *this*
item's key or to one of its human distractors, then draws three values from what
is left, weighted by multiplicity and rejecting repeats. Removing the key's own
mass is the redraw-against-the-key rule Table~\ref{tab:distinctness} prices;
removing the human distractors' values makes the draw depend on the item too.

$\Gamma=\mathbb{E}_V[\max_v\Pr(Y=v\mid V)]-1/k$ is estimated by Monte Carlo over
presented sets drawn from the file's own law, with the posterior of each drawn
set computed exactly: $\Pr(Y=v\mid V)\propto\sum_{i:\,y_i=v}\Pr_i(V\setminus\{v\})$,
where $\Pr_i$ is item $i$'s three-value draw summed over the six orders. Given
exact posteriors the estimator is unbiased for $\Gamma$. The same draw removing
only the key's value (the rule the distinctness trap names) runs beside it;
Proposition~\ref{prop:writers}(i)'s jointly conditioned draw is zero by symmetry.

    python3 key_marginal_gamma.py
"""
import argparse
import json
import pathlib
from collections import Counter, defaultdict
from itertools import permutations

import numpy as np

from repair_frontier import numeric_rows


def p_draw(triple, weights, total):
    """Probability that three multiplicity-weighted draws without repeats return ``triple``."""
    p = 0.0
    for a, b, c in permutations(triple):
        wa, wb, wc = weights.get(a, 0), weights.get(b, 0), weights.get(c, 0)
        if not (wa and wb and wc):
            return 0.0
        p += (wa / total) * (wb / (total - wa)) * (wc / (total - wa - wb))
    return p


def draw(weights, rng):
    values = list(weights)
    w = np.array([weights[v] for v in values], float)
    out = []
    for _ in range(3):
        live = [i for i in range(len(values)) if values[i] not in out]
        p = w[live] / w[live].sum()
        out.append(values[live[rng.choice(len(live), p=p)]])
    return out


def estimate(keys, pools, rng, samples):
    """Monte Carlo Gamma with exact posteriors; ``pools[i]`` is item i's draw weights."""
    by_key = defaultdict(list)
    for i, k in enumerate(keys):
        by_key[k].append(i)
    totals = [float(sum(p.values())) for p in pools]
    live = [i for i in range(len(keys)) if len(pools[i]) >= 3]
    if not live:
        return None
    acc = []
    for _ in range(samples):
        i = live[rng.integers(len(live))]
        triple = draw(pools[i], rng)
        v_set = [keys[i]] + triple
        post = []
        for v in v_set:
            rest = tuple(u for u in v_set if u != v)
            post.append(sum(p_draw(rest, pools[j], totals[j]) for j in by_key.get(v, [])
                            if len(pools[j]) >= 3))
        post = np.array(post) / sum(post)
        acc.append(post.max())
    return 100.0 * (float(np.mean(acc)) - 0.25), 100.0 * float(np.std(acc) / np.sqrt(samples))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", default="build/mmlu_questions.jsonl")
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--output", default="results/key_marginal_gamma.json")
    args = ap.parse_args()

    rows = numeric_rows(args.jsonl)
    by_subject = defaultdict(list)
    for row in rows:
        by_subject[row.get("cluster") or ""].append(row)
    rng = np.random.default_rng(args.seed)
    report = {"n_items": len(rows), "samples_per_subject": args.samples, "subjects": {}}
    sums = defaultdict(float)
    n_used = 0
    for subject, members in sorted(by_subject.items()):
        keys = [str(r["ideal"]).strip() for r in members]
        counts = Counter(keys)
        human = [{str(d).strip() for d in r["distractors"]} for r in members]
        variants = {
            "as_built": [{v: c for v, c in counts.items() if v != k and v not in h} for k, h in zip(keys, human)],
            "against_key": [{v: c for v, c in counts.items() if v != k} for k in keys],
        }
        entry = {"n_items": len(members), "distinct_keys": len(counts)}
        for name, pools in variants.items():
            got = estimate(keys, pools, rng, args.samples)
            entry[name] = None if got is None else {"gamma": got[0], "mc_se": got[1]}
        report["subjects"][subject] = entry
        if entry["as_built"] is not None and entry["against_key"] is not None:
            for name in variants:
                sums[name] += len(members) * entry[name]["gamma"]
            n_used += len(members)
        print(f"{subject[:36]:36s} n={len(members):3d} distinct {len(counts):3d}  as built "
              + (f"{entry['as_built']['gamma']:+6.2f} (se {entry['as_built']['mc_se']:.2f})"
                 if entry["as_built"] else "  --  ")
              + "  against key "
              + (f"{entry['against_key']['gamma']:+6.2f}" if entry["against_key"] else "--"), flush=True)
    report["item_weighted"] = {name: v / n_used for name, v in sums.items()}
    report["n_items_with_a_draw"] = n_used
    print("item-weighted Gamma, points: " + ", ".join(f"{k} {v:+.2f}" for k, v in report["item_weighted"].items()))
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
