#!/usr/bin/env python3
r"""Distractor writers on BixBench itself, placed on the trade's two axes.

The writer frontier of Appendix F is measured on MMLU, where a solver can be given
the question. On BixBench a question is answered from its data, so a writer is
judged here by the two things Proposition 2 trades against each other:

    leak    what a reader that never sees the question takes from the option set:
            the best single-rank rule, its rank chosen on held-out capsules, over
            the 25% chance (and in-sample, beside the bracketing floor).
    catch   what the option set does to an agent that did the analysis: over the
            seven run sets' with-data answers (frozen, read by the option nearest
            the submitted number), how often an answer more than 5% from the key
            is read as the key (a credited miss), and how often one within 5% is
            (a kept hit).

Writers, all on the same 105 numeric items and keys:

    released          what BixBench ships
    placebo           every distractor redrawn, the key's rank held
    rank uniform      the repair: redrawn with the rank drawn uniformly
    slot-symmetric    the key a uniform draw from its own set (exchangeable_repair.py;
                      98 items admit it)
    key marginal      other items' numeric keys, drawn from the file's multiset
    key marginal, near  the same within a factor of ten of the key
    wrong step        a model shown the question and never the key writes the
                      values a mistaken analysis would give (Qwen2.5-14B; Llama-3.1-8B)

The same writers (those that need no model) are built on v1.0's 159 numeric items and
read through the published with-data answers of gpt-4o and Claude 3.5 Sonnet
(``replication.py``'s extract), so the catch axis is also measured on closed models.

    python3 writer_panel.py         # results/writer_panel.json
"""
import json
import pathlib
import random
from collections import defaultdict

import numpy as np

import bixbench_withdata as bw
import bracketing
from bracketing import open_side
from channel_survey import cluster_robust_uniformity, grouped_cv_credit, numeric_ranks
from answer_numbers import graded
from option_artifacts import parse_number
from repair_frontier import draw_distinct

ROOT = pathlib.Path(__file__).resolve().parent
SEED = 20260925
K = 4
FILES = {"released": "build/bixbench_numeric_q.jsonl",
         "placebo": "build/bixbench_numeric_placebo.jsonl",
         "rank uniform": "build/bixbench_numeric_repaired.jsonl",
         "slot-symmetric": "build/bixbench_numeric_slot_symmetric.jsonl",
         "wrong step, Qwen2.5-14B": "build/bixbench_numeric_wrong_step.jsonl",
         "wrong step, Llama-3.1-8B": "build/bixbench_numeric_wrong_step_llama.jsonl"}
NEAR_FACTOR = 10.0


def rows_of(path):
    return [json.loads(line) for line in open(ROOT / path) if line.strip()]


def key_marginal(released, seed=SEED, cluster="cluster"):
    """Other items' keys as distractors: the file's multiset of numeric keys, and the same
    restricted to within a factor of ten of this key."""
    rng = random.Random(seed)
    pool = [str(r["ideal"]).strip() for r in released]
    wide, near = [], []
    for r in released:
        key = str(r["ideal"]).strip()
        options = [key] + [str(d).strip() for d in r["distractors"]]
        others = [p for p in pool if p != key]
        loan = draw_distinct(others, 3, options, rng)
        k = float(parse_number(key))
        close = [p for p in others
                 if parse_number(p) is not None and k != 0 and parse_number(p) != 0
                 and 1 / NEAR_FACTOR <= abs(parse_number(p)) / abs(k) <= NEAR_FACTOR]
        base = {k: r[k] for k in ("question", cluster, "question_id") if k in r}
        if len(loan) == 3:
            wide.append({"ideal": key, "distractors": loan, **base})
        if len(set(close)) >= 3:
            near.append({"ideal": key, "distractors": draw_distinct(close, 3, options, rng), **base})
    return wide, near


def writers():
    released = rows_of(FILES["released"])
    out = {name: rows_of(path) for name, path in FILES.items()}
    out["key marginal"], out["key marginal, near"] = key_marginal(released)
    return out


def to_qid():
    """(question, key) -> BixBench question_id, as bixbench_withdata.load_items maps them."""
    items = [json.loads(line) for line in open(ROOT / "data" / "bixbench.jsonl")]
    return {(r["question"], r["ideal"]): r["question_id"] for r in items}, \
        {r["question_id"]: r["capsule_uuid"] for r in items}


def leak(rows, cluster="cluster"):
    """The rank channel of one writer's option set, read without the question."""
    items = []
    for r in rows:
        ranks = numeric_ranks([r["ideal"], *r["distractors"]])
        if ranks is not None:
            items.append({"rank": ranks[0], "cluster": r[cluster]})
    n = len(items)
    share = [sum(i["rank"] == j for i in items) / n for j in range(K)]
    beta = share[1] + share[2]
    cv = grouped_cv_credit(items, K, 2000)
    robust = cluster_robust_uniformity(items, K)
    return {"n_items": n, "key_rank_share": share, "bracketed": beta,
            "floor": beta / (K - 2) - 1 / K, "in_sample": max(share) - 1 / K,
            "held_out": cv["credit"] if cv else None,
            "held_out_ci95": cv["credit_ci95"] if cv else None,
            "uniform_rank_p": robust.get("p_value")}


def v15_answers():
    """The seven run sets' with-data answers on the numeric items: (question_id, answer)."""
    return [(row["question_id"], row["answer"]) for run in bracketing.RUNS for row in bracketing.rows_of(run)
            if row["numeric"] and row["condition"] == "data"]


def catch(rows, qid_of, capsule_of, answers=None, tol=0.05):
    """Over with-data answers, how the option set reads hits and misses (a hit: within ``tol``)."""
    sets = {}
    for r in rows:
        q = r.get("question_id") or qid_of.get((r["question"], r["ideal"]))
        options = [r["ideal"], *r["distractors"]]
        if q is not None and numeric_ranks(options) is not None:   # the items the leak axis reads
            sets[q] = options
    miss, hit, side = defaultdict(list), defaultdict(list), []
    for q, answer in (answers if answers is not None else v15_answers()):
        if q not in sets or not answer or parse_number(answer) is None:
            continue
        read = bw.nearest_is_key(answer, sets[q])
        if graded(answer, sets[q][0], tol):
            hit[capsule_of[q]].append(read)
        else:
            miss[capsule_of[q]].append(read)
            if read > 0:
                side.append(open_side(answer, sets[q]))
    flat = lambda d: [x for v in d.values() for x in v]
    ci = lambda d: bw.cluster_interval(flat(d), [c for c, v in d.items() for _ in v])
    return {"n_items": len(sets), "n_misses": len(flat(miss)), "n_hits": len(flat(hit)),
            "credited_misses": ci(miss), "kept_hits": ci(hit),
            "credited_misses_open_side": float(np.mean(side)) if side else None}


V10 = {"released": "build/bixbench_v10_numeric_released.jsonl",
       "placebo": "build/bixbench_v10_numeric_placebo.jsonl",
       "rank uniform": "build/bixbench_v10_repaired.jsonl",
       "slot-symmetric": "build/bixbench_v10_numeric_slot_symmetric.jsonl"}


def writers_v10():
    released = rows_of(V10["released"])
    numeric = {r["question_id"] for r in released}
    out = {name: [r for r in rows_of(path) if r["question_id"] in numeric] for name, path in V10.items()}
    out["key marginal"], out["key marginal, near"] = key_marginal(released, cluster="capsule_uuid")
    return out


ORDER = ("released", "placebo", "rank uniform", "slot-symmetric", "key marginal", "key marginal, near",
         "wrong step, Qwen2.5-14B", "wrong step, Llama-3.1-8B")


def _half(entry):
    if entry is None:
        return "-- & -- & -- & --"
    lk, ct = entry["leak"], entry["catch"]
    cm = ct["credited_misses"]
    return (f"${100 * lk['bracketed']:.0f}$ & ${100 * lk['held_out']:+.1f}$ & "
            f"${cm['mean']:.1f}$ {{\\scriptsize$[{cm['lo']:.1f},{cm['hi']:.1f}]$}} & ${ct['kept_hits']['mean']:.0f}$")


def table_rows(report):
    """tab:panel's rows: each writer on v1.5 (open agents) and v1.0 (the published closed models)."""
    short = {"wrong step, Qwen2.5-14B": "wrong step (Qwen)", "wrong step, Llama-3.1-8B": "wrong step (Llama)",
             "released": "released ($R$)", "placebo": "rank-preserving ($P$)", "rank uniform": "rank-uniform ($U$)"}
    return [f"{short.get(name, name)} & {_half(report['writers'].get(name))} & "
            f"{_half(report['writers_v10'].get(name))}\\\\" for name in ORDER]


def main():
    import sys
    if "--latex" in sys.argv:
        for line in table_rows(json.loads((ROOT / "results" / "writer_panel.json").read_text())):
            print(line)
        return
    import replication
    qid_of, capsule_of = to_qid()
    report = {"seed": SEED, "near_factor": NEAR_FACTOR, "writers": {}, "writers_v10": {}}
    doc = replication.load_extract()
    answers10 = [(q, a) for run in replication.OPEN_RUNS for q, a, _ in doc["open_runs"][run]]
    capsule10 = {q: e["capsule"] for q, e in doc["items"].items()}
    runs = [("writers", writers(), "cluster", capsule_of, None),
            ("writers_v10", writers_v10(), "capsule_uuid", capsule10, answers10)]
    for key, panel, cluster, capsules, answers in runs:
        print(f"--- {key}")
        for name, rows in panel.items():
            entry = {"leak": leak(rows, cluster), "catch": catch(rows, qid_of, capsules, answers)}
            report[key][name] = entry
            lk, ct = entry["leak"], entry["catch"]
            cm, kh = ct["credited_misses"], ct["kept_hits"]
            print(f"{name:26s} n={lk['n_items']:3d} bracketed {100 * lk['bracketed']:5.1f}% floor "
                  f"{100 * lk['floor']:+5.1f} in-sample {100 * lk['in_sample']:+5.1f} held-out "
                  f"{100 * lk['held_out']:+5.1f} | credited misses {cm['mean']:5.1f}% [{cm['lo']:.1f},{cm['hi']:.1f}] "
                  f"(n={ct['n_misses']}), kept hits {kh['mean']:5.1f}% (n={ct['n_hits']}), open side "
                  f"{100 * (ct['credited_misses_open_side'] or 0):.0f}%")
    out = ROOT / "results" / "writer_panel.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
