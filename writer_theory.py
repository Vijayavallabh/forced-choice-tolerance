"""What a distractor writer must do to close the channel, and what it costs.

Two halves, run with one command each.

``--exact`` enumerates finite generative laws and computes Gamma exactly, which
checks four statements the paper makes about writers:

  * the presented set is unordered, so full k-tuple exchangeability is
    *sufficient* and not necessary -- sorting the distractors breaks the tuple's
    exchangeability and leaves Gamma at zero;
  * a writer that reads nothing about the item closes the channel exactly when it
    draws from the key marginal itself;
  * at k=2 an item-conditional writer closes it exactly when its kernel is
    reversible with respect to the key marginal;
  * redrawing a colliding distractor *against the key* -- the loop every
    assembler writes -- is itself a dependence on the key, and it opens the
    channel on a skewed marginal where the same draw applied symmetrically
    does not.

``--empirical`` tests the price.  For an item-independent writer the accuracy of
any solver that scores options separately given the question is
E_Q[F_Q(s(key,Q))^(k-1)], where F_Q is the c.d.f. of the score of a value drawn
from the key marginal.  Nothing in that expression mentions the writer, so the
KEY MARGINAL arm's difficulty column can be predicted before the arm is built.
We estimate it with a language model scoring each value against its question and
compare with the arm's measured accuracy.

    /opt/conda/bin/python writer_theory.py --exact
    /opt/conda/bin/python writer_theory.py --empirical --device cuda:0
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import pathlib

import numpy as np

RESULTS = pathlib.Path("results")
BUILD = pathlib.Path("build")


# --------------------------------------------------------------------------- #
# Exact Gamma over finite laws                                                 #
# --------------------------------------------------------------------------- #

def gamma_exact(joint: dict, k: int) -> float:
    """Gamma for a law given as {(key, sorted-others): probability}.

    The solver sees the *set*, so the posterior is pooled over every way the set
    could have arisen.  Gamma is the mean of its maximum, minus chance.
    """
    posterior: dict = collections.defaultdict(dict)
    total = 0.0
    for (key, others), p in joint.items():
        options = frozenset({key} | set(others))
        posterior[options][key] = posterior[options].get(key, 0.0) + p
        total += p
    return sum(max(d.values()) for d in posterior.values()) / total - 1.0 / k


def _normalise(joint: dict) -> dict:
    total = sum(joint.values())
    return {a: b / total for a, b in joint.items()}


def joint_sorted_distractors(n_values: int, k: int) -> tuple[dict, bool]:
    """The counterexample to necessity: key uniform on the set, distractors sorted.

    Returns the law and whether the *ordered* k-tuple is exchangeable.
    """
    ordered: dict = collections.Counter()
    for options in itertools.combinations(range(n_values), k):
        for key in options:
            ordered[(key,) + tuple(sorted(v for v in options if v != key))] += 1
    exchangeable = all(
        ordered[t] == ordered[(t[0],) + tuple(perm)]
        for t in ordered
        for perm in itertools.permutations(t[1:]))
    joint: dict = collections.Counter()
    for tup, c in ordered.items():
        joint[(tup[0], tuple(sorted(tup[1:])))] += c
    return _normalise(joint), exchangeable


def joint_symmetric(kappa: np.ndarray, k: int) -> dict:
    """k values i.i.d. from ``kappa`` conditioned on all k distinct; slot 1 is the key.

    The distinctness rule touches every slot alike, so it cannot mark one.
    """
    joint: dict = collections.Counter()
    for tup in itertools.permutations(range(len(kappa)), k):
        joint[(tup[0], tuple(sorted(tup[1:])))] += float(np.prod([kappa[x] for x in tup]))
    return _normalise(joint)


def joint_reject_against_key(kappa: np.ndarray, k: int) -> dict:
    """The assembler's loop: draw the key, then redraw each distractor while it
    collides with the key or an earlier distractor."""
    joint: dict = collections.Counter()
    values = range(len(kappa))
    for key in values:
        for others in itertools.permutations([v for v in values if v != key], k - 1):
            p, used = float(kappa[key]), {key}
            for u in others:
                p *= float(kappa[u]) / (1.0 - sum(float(kappa[x]) for x in used))
                used.add(u)
            joint[(key, tuple(sorted(others)))] += p
    return _normalise(joint)


def joint_item_independent(kappa: np.ndarray, nu: np.ndarray, k: int) -> dict:
    """Key from ``kappa``, distractors i.i.d. from ``nu``, jointly conditioned on distinctness."""
    joint: dict = collections.Counter()
    values = range(len(kappa))
    for key in values:
        for others in itertools.permutations([v for v in values if v != key], k - 1):
            joint[(key, tuple(sorted(others)))] += float(
                kappa[key] * np.prod([nu[u] for u in others]))
    return _normalise(joint)


def joint_item_conditional(kappa: np.ndarray, nu_matrix: np.ndarray, k: int) -> dict:
    """Key from ``kappa``, distractors i.i.d. from ``nu_matrix[key]``."""
    joint: dict = collections.Counter()
    values = range(len(kappa))
    for key in values:
        for others in itertools.permutations([v for v in values if v != key], k - 1):
            joint[(key, tuple(sorted(others)))] += float(
                kappa[key] * np.prod([nu_matrix[key][u] for u in others]))
    return _normalise(joint)


def reversible_kernel(kappa: np.ndarray) -> np.ndarray:
    """A kernel reversible with respect to ``kappa`` on three values.

    Detailed balance asks for a symmetric edge measure whose row sums are
    ``kappa``; on three values that pins it.
    """
    assert len(kappa) == 3
    edge = np.zeros((3, 3))
    for i, j, m in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        edge[i, j] = edge[j, i] = (kappa[i] + kappa[j] - kappa[m]) / 2.0
    return edge / kappa[:, None]


def detailed_balance_residual(kappa: np.ndarray, nu: np.ndarray) -> float:
    return float(max(abs(kappa[v] * nu[v][u] - kappa[u] * nu[u][v])
                     for v in range(len(kappa)) for u in range(len(kappa))))


def joint_law_solutions(kappa: np.ndarray, k: int, tol: float = 1e-9) -> dict:
    r"""Every item-blind writer with Gamma = 0, not just the i.i.d. ones.

    Proposition 2(i) as stated assumes the ``k-1`` distractors are drawn
    independently.  A real generator does not: it emits a tuple, and the tuple's
    entries constrain each other.  So drop independence.  Let the writer produce
    the distractor set from an arbitrary joint law ``mu`` over ``(k-1)``-subsets
    of the value pool, let the key be drawn from ``kappa`` independently of it,
    and condition on the ``k`` options coming out distinct, which any assembler
    must do.  Then

        Pr(Y = a | V = S)  proportional to  kappa(a) * mu(S \ {a}),

    so Gamma = 0 asks that ``kappa(a) * mu(S \ {a})`` be constant over the
    members of every presented set ``S`` -- and that is *linear* in ``mu``.  The
    whole family of item-blind writers that close the channel is therefore a
    null space, which this computes.

    It comes out one-dimensional, spanned by ``mu(T)`` proportional to the
    product of ``kappa`` over ``T``: the only item-blind writer that closes the
    channel is still the key marginal, now with no independence assumed, and the
    distinctness conditioning is part of the statement rather than an
    implementation detail.  That is also why ``joint_reject_against_key`` is a
    trap: rejecting a collision *against the key* is a different conditioning,
    and it leaves the family.
    """
    subsets = list(itertools.combinations(range(len(kappa)), k - 1))
    index = {T: i for i, T in enumerate(subsets)}
    rows = []
    for presented in itertools.combinations(range(len(kappa)), k):
        for a, b in itertools.combinations(presented, 2):
            row = np.zeros(len(subsets))
            row[index[tuple(sorted(set(presented) - {a}))]] += kappa[a]
            row[index[tuple(sorted(set(presented) - {b}))]] -= kappa[b]
            rows.append(row)
    matrix = np.array(rows)
    singular = np.linalg.svd(matrix, compute_uv=False)
    dimension = len(subsets) - int((singular > tol * max(1.0, singular[0])).sum())
    product = np.array([float(np.prod([kappa[x] for x in T])) for T in subsets])
    product /= product.sum()
    return {"n_values": len(kappa), "k": k, "n_free_entries": len(subsets),
            "solution_space_dimension": dimension,
            "product_law_residual": float(np.abs(matrix @ product).max())}


def run_exact() -> dict:
    out: dict = {}

    law, exchangeable = joint_sorted_distractors(n_values=10, k=3)
    out["sorted_distractors"] = {
        "gamma": gamma_exact(law, 3), "tuple_exchangeable": exchangeable,
        "note": "key uniform on the set, distractors listed in increasing order"}

    kappa4 = np.array([0.40, 0.30, 0.20, 0.10])
    rows = []
    for name, nu in (("nu = kappa", kappa4),
                     ("nu uniform", np.full(4, 0.25)),
                     ("nu reversed", kappa4[::-1].copy())):
        for k in (2, 3, 4):
            rows.append({"nu": name, "k": k,
                         "gamma": gamma_exact(joint_item_independent(kappa4, nu, k), k)})
    out["item_independent"] = rows

    kappa3 = np.array([0.50, 0.30, 0.20])
    nu_rev = reversible_kernel(kappa3)
    nu_not = np.array([[0.0, 0.9, 0.1], [0.1, 0.0, 0.9], [0.9, 0.1, 0.0]])
    out["reversibility_k2"] = [
        {"kernel": "kappa-reversible",
         "detailed_balance_residual": detailed_balance_residual(kappa3, nu_rev),
         "gamma": gamma_exact(joint_item_conditional(kappa3, nu_rev, 2), 2)},
        {"kernel": "not reversible",
         "detailed_balance_residual": detailed_balance_residual(kappa3, nu_not),
         "gamma": gamma_exact(joint_item_conditional(kappa3, nu_not, 2), 2)}]

    traps = []
    for name, kappa in (("uniform", np.full(6, 1 / 6)),
                        ("mild skew", np.array([.25, .22, .20, .15, .10, .08])),
                        ("heavy skew", np.array([.45, .25, .15, .08, .05, .02]))):
        for k in (4,):
            traps.append({
                "key_marginal": name, "k": k,
                "gamma_symmetric": gamma_exact(joint_symmetric(kappa, k), k),
                "gamma_reject_against_key": gamma_exact(joint_reject_against_key(kappa, k), k)})
    out["distinctness_trap"] = traps
    # how the trap grows with the key marginal's concentration: kappa_j proportional to exp(-s j) over six
    # values, from uniform (s = 0) past the heavy skew above (whose largest share is 0.45)
    sweep = []
    for s in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        kappa = np.exp(-s * np.arange(6))
        kappa = kappa / kappa.sum()
        sweep.append({"s": s, "kappa": [float(x) for x in kappa], "largest_share": float(kappa.max()),
                      "gamma_symmetric": gamma_exact(joint_symmetric(kappa, 4), 4),
                      "gamma_reject_against_key": gamma_exact(joint_reject_against_key(kappa, 4), 4)})
    out["distinctness_sweep"] = sweep

    # Proposition 2(i) without the independence assumption: the family of
    # item-blind writers closing the channel is a null space, and it is a line.
    solutions = []
    rng = np.random.default_rng(20260920)
    for n_values, k in ((4, 3), (5, 3), (6, 3), (5, 4), (6, 4), (7, 4), (6, 5), (8, 3)):
        skewed = rng.dirichlet(np.ones(n_values) * 0.7)
        solutions.append(joint_law_solutions(skewed, k))
    out["item_blind_joint_laws"] = solutions
    return out


# --------------------------------------------------------------------------- #
# The price: predicting the difficulty column without building the arm         #
# --------------------------------------------------------------------------- #

def _prompt(question: str, value: str) -> tuple[str, str]:
    return (f"Question: {question}\nAnswer: ", value)


def score_values(model, tok, pairs, batch_size: int, device) -> np.ndarray:
    """Mean log-probability of each value's tokens given its prefix."""
    import torch

    scores = np.zeros(len(pairs), dtype=np.float64)
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        texts = [p + v for p, v in (pairs[i] for i in idx)]
        prefix_lens = [len(tok(pairs[i][0], add_special_tokens=True)["input_ids"]) for i in idx]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits.float().log_softmax(-1)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        for row, i in enumerate(idx):
            n = int(mask[row].sum())
            offset = int(ids.shape[1] - n) if tok.padding_side == "left" else 0
            lo, hi = offset + prefix_lens[row], offset + n
            if hi - lo < 1:
                scores[i] = float("nan")
                continue
            token_lp = logits[row, lo - 1:hi - 1].gather(
                1, ids[row, lo:hi].unsqueeze(1)).squeeze(1)
            scores[i] = float(token_lp.mean())
    return scores


def run_empirical(args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in
            (BUILD / "mmlu_frontier_released_matched_aligned.jsonl").open()]
    arm = [json.loads(l) for l in
           (BUILD / "mmlu_frontier_key_marginal_aligned.jsonl").open()]
    assert len(rows) == len(arm)
    k = 1 + len(rows[0]["distractors"])

    # The pool a subject-restricted item-independent writer draws from is that
    # subject's own keys; the draw is the empirical multiset, not its support.
    pool: dict = collections.defaultdict(list)
    for r in rows:
        pool[r["cluster"]].append(r["ideal"])

    rng = np.random.default_rng(args.seed)
    jobs, index = [], []
    for i, r in enumerate(rows):
        jobs.append(_prompt(r["question"], r["ideal"]))
        index.append((i, "key", r["ideal"]))
        draws = pool[r["cluster"]]
        take = min(args.pool, len(draws))
        for v in rng.choice(draws, size=take, replace=len(draws) < take):
            jobs.append(_prompt(r["question"], str(v)))
            index.append((i, "pool", str(v)))
        for v in arm[i]["distractors"]:           # the arm the theory predicts
            jobs.append(_prompt(r["question"], str(v)))
            index.append((i, "arm", str(v)))

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    scores = score_values(model, tok, jobs, args.batch_size, args.device)

    per_item: dict = collections.defaultdict(lambda: {"pool": [], "arm": []})
    for s, (i, kind, _) in zip(scores, index):
        if kind == "key":
            per_item[i]["key"] = s
        else:
            per_item[i][kind].append(s)

    beats, predicted, realised, item_cluster = [], [], [], []
    for i, rec in sorted(per_item.items()):
        key, draws, armed = rec["key"], np.array(rec["pool"]), np.array(rec["arm"])
        if not len(draws) or np.isnan(key):
            continue
        beats.append(float((draws >= key).mean()))          # the collision rate, per item
        predicted.append(float(((draws < key).mean()) ** (k - 1)))
        realised.append(float(bool(np.all(armed < key))))   # the separable solver on the real arm
        item_cluster.append(rows[i]["cluster"])

    rho = float(np.mean(beats))
    clusters = sorted(set(item_cluster))
    boot = []
    for _ in range(args.bootstrap):
        pick = rng.choice(len(clusters), size=len(clusters), replace=True)
        take = [j for c in (clusters[p] for p in pick)
                for j in range(len(predicted)) if item_cluster[j] == c]
        if take:
            boot.append(float(np.mean([predicted[j] for j in take])))
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"),) * 2)

    return {
        "model": args.model, "n_items": len(predicted), "k": k,
        "pool_draws_per_item": args.pool,
        "collision_rate_rho": rho,
        "union_bound_accuracy": 1.0 - (k - 1) * rho,
        "predicted_accuracy": 100.0 * float(np.mean(predicted)),
        "predicted_ci95": [100.0 * float(lo), 100.0 * float(hi)],
        "separable_solver_on_arm": 100.0 * float(np.mean(realised)),
        "n_clusters": len(clusters)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--empirical", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pool", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    if args.exact:
        out = RESULTS / (args.out or "writer_theory.json")
        out.write_text(json.dumps(run_exact(), indent=1) + "\n")
        print(f"wrote {out}")
    if args.empirical:
        out = RESULTS / (args.out or "writer_price.json")
        payload = run_empirical(args)
        merged = json.loads(out.read_text()) if out.exists() else {}
        merged[args.model] = payload
        out.write_text(json.dumps(merged, indent=1) + "\n")
        print(f"wrote {out}: " + json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
