#!/usr/bin/env python3
"""Test whether non-numeric multiple-choice items leak through surface features.

The numeric leak reported in ``option_artifacts.py`` only applies to items
whose four options parse as numbers. That leaves half of BixBench v1.5
unexamined, and a reader is entitled to ask whether the other half leaks too.

Testing one hand-picked rule on the whole set would invite the obvious
objection, so this module defines a *family* of surface rules and selects
among them by leave-one-cluster-out cross-validation: the rule is chosen on
training folds and scored only on the held-out fold. A family that contains no
generalising rule therefore reports chance-level accuracy no matter how many
members it has, which is the behaviour a multiple-comparison guard should have.

A null result here is a real result: it says the numeric leak is specific to
numeric distractor generation rather than a general property of the benchmark.
"""
import argparse
import ast
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path

from option_artifacts import (N_OPTIONS, cluster_bootstrap, numeric_rank,
                              parse_number, rate)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
DIGIT_RE = re.compile(r"\d")


def tokens(text):
    return set(TOKEN_RE.findall(str(text).lower()))


def similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


# --------------------------------------------------------------------------
# the rule family: each scores an option, and the rule picks the argmax
# --------------------------------------------------------------------------
def _stem_token_overlap(option, others, question):
    q = tokens(question)
    o = tokens(option)
    return len(q & o) / len(o) if o else 0.0


def _stem_token_overlap_absolute(option, others, question):
    return len(tokens(question) & tokens(option))


def _length(option, others, question):
    return len(str(option))


def _neg_length(option, others, question):
    return -len(str(option))


def _token_count(option, others, question):
    return len(TOKEN_RE.findall(str(option)))


def _odd_one_out(option, others, question):
    """Least similar to the other three: a keyed answer written by a human
    among three generated distractors may stand out lexically.

    ``math.fsum`` rather than ``sum``: these scores are dense with near-ties and
    the rule takes an arg-max over them, so the last bit decides which option
    the rule picks. CPython 3.12 changed ``sum`` over floats to compensated
    summation, which made this rule -- and mcq_audit.isolation_scores, the same
    statistic on the audit's side -- answer differently on different
    interpreters. ``fsum`` is exactly rounded on all of them.
    """
    return -math.fsum(similarity(option, other) for other in others)


def _consensus(option, others, question):
    """Most similar to the other three: the opposite prediction."""
    return math.fsum(similarity(option, other) for other in others)


def _has_digit(option, others, question):
    return 1.0 if DIGIT_RE.search(str(option)) else 0.0


def _specificity(option, others, question):
    """Qualified answers ('approximately', units, parentheses) often read as
    the careful, correct one."""
    text = str(option).lower()
    marks = sum(text.count(ch) for ch in "(),;%")
    hedges = sum(text.count(word) for word in ("approx", "about", "~", "per ", "/"))
    return marks + hedges


RULES = {
    "longest_option": _length,
    "shortest_option": _neg_length,
    "most_tokens": _token_count,
    "stem_overlap_fraction": _stem_token_overlap,
    "stem_overlap_count": _stem_token_overlap_absolute,
    "least_like_other_options": _odd_one_out,
    "most_like_other_options": _consensus,
    "contains_a_digit": _has_digit,
    "most_qualified": _specificity,
}


def pick(rule, options, question):
    """Deterministic argmax; ties break on the option's own text."""
    best, best_key = None, None
    for i, option in enumerate(options):
        others = [o for j, o in enumerate(options) if j != i]
        key = (rule(option, others, question), str(option))
        if best_key is None or key > best_key:
            best, best_key = i, key
    return best


# --------------------------------------------------------------------------
def load_items(jsonl, include):
    """include: 'non_numeric', 'numeric', or 'all'."""
    items = []
    with open(jsonl, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            distractors = row.get("distractors") or []
            if isinstance(distractors, str):
                try:
                    distractors = ast.literal_eval(distractors)
                except (ValueError, SyntaxError):
                    continue
            options = [row.get("ideal")] + list(distractors)
            if len(options) != N_OPTIONS or any(o is None for o in options):
                continue
            is_numeric = numeric_rank(options, 0) is not None
            if include == "non_numeric" and is_numeric:
                continue
            if include == "numeric" and not is_numeric:
                continue
            items.append({"options": [str(o) for o in options], "key": 0,
                          "question": row.get("question", ""),
                          "cluster": row["capsule_uuid"], "id": row["question_id"]})
    return items


def in_sample_accuracy(items):
    out = {}
    for name, rule in RULES.items():
        hits = sum(pick(rule, it["options"], it["question"]) == it["key"] for it in items)
        out[name] = hits / len(items) if items else None
    return out


def cross_validated(items, reps=10000):
    """Select the rule on training folds only, score on the held-out fold."""
    by_cluster = {}
    for it in items:
        by_cluster.setdefault(it["cluster"], []).append(it)
    if len(by_cluster) < 2:
        return None

    # precompute each rule's hit/miss per item so folds are cheap
    hits = {name: {id(it): pick(rule, it["options"], it["question"]) == it["key"]
                   for it in items}
            for name, rule in RULES.items()}

    scored, selected = {}, {}
    for held_out, held_items in by_cluster.items():
        train = [it for c, group in by_cluster.items() if c != held_out for it in group]
        best = max(RULES, key=lambda n: (sum(hits[n][id(it)] for it in train), n))
        selected[best] = selected.get(best, 0) + 1
        scored.setdefault(held_out, []).extend(
            1.0 if hits[best][id(it)] else 0.0 for it in held_items)

    flat = [x for v in scored.values() for x in v]
    lo, hi = cluster_bootstrap(scored, rate, reps=reps)
    return {"accuracy": rate(flat), "n_scored": len(flat), "folds": len(by_cluster),
            "ci95": [lo, hi], "rules_in_family": len(RULES),
            "rule_selected_per_fold": selected,
            "above_chance": bool(lo is not None and lo > 1 / N_OPTIONS)}


def rank_proxy_analysis(jsonl):
    """Are the surface rules a separate channel, or a shadow of the value rank?

    A surface rule that scores below chance looks like an independent leak --
    avoid what it picks and a guess improves from 1/4 to 1/3. On this benchmark
    it is not independent: every rule lands overwhelmingly on the largest-valued
    option, which the key almost never is. Reporting the two as separate leaks
    would double-count one defect, so the overlap is measured rather than
    assumed.
    """
    from collections import Counter

    from option_artifacts import parse_number

    items = [it for it in load_items(jsonl, "numeric")]
    rows = {}
    for name, rule in RULES.items():
        counts = Counter()
        scored = 0
        for item in items:
            values = [parse_number(o) for o in item["options"]]
            if any(v is None for v in values) or len(set(values)) < N_OPTIONS:
                continue
            order = sorted(range(N_OPTIONS), key=lambda i: values[i])
            counts[order.index(pick(rule, item["options"], item["question"]))] += 1
            scored += 1
        if scored:
            rows[name] = {"n_scored": scored,
                          "selects_value_rank": {str(r): counts.get(r, 0) / scored
                                                 for r in range(N_OPTIONS)},
                          "selects_largest_value": counts.get(N_OPTIONS - 1, 0) / scored}
    shares = [v["selects_largest_value"] for v in rows.values()]
    return {"note": ("share of items on which each surface rule selects the largest-valued "
                     "option; the key is largest on only a small fraction of items, so a rule "
                     "concentrated here is a rank proxy rather than a separate channel"),
            "per_rule": rows,
            "min_share_selecting_largest": min(shares) if shares else None,
            "max_share_selecting_largest": max(shares) if shares else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", type=Path, default=Path("data/bixbench.jsonl"))
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("results/text_artifacts.json"))
    args = parser.parse_args()

    report = {"chance": 1 / N_OPTIONS, "rules_in_family": sorted(RULES),
              "bootstrap_reps": args.reps, "subsets": {}}

    for include in ("non_numeric", "numeric", "all"):
        items = load_items(args.jsonl, include)
        if not items:
            continue
        cv = cross_validated(items, reps=args.reps)
        report["subsets"][include] = {
            "n_items": len(items),
            "n_clusters": len({it["cluster"] for it in items}),
            "in_sample_best": max(in_sample_accuracy(items).items(), key=lambda kv: kv[1]),
            "in_sample_all": in_sample_accuracy(items),
            "cross_validated": cv,
        }
        best_name, best_acc = report["subsets"][include]["in_sample_best"]
        print(f"\n{include}: {len(items)} items in {report['subsets'][include]['n_clusters']} clusters")
        print(f"  best of {len(RULES)} rules, in sample: {best_name} at {best_acc:.1%}")
        if cv:
            flag = "  <-- above chance" if cv["above_chance"] else ""
            print(f"  cross-validated (honest):      {cv['accuracy']:.1%} "
                  f"[{cv['ci95'][0]:.1%}, {cv['ci95'][1]:.1%}] vs 25% chance{flag}")

    report["rank_proxy_check"] = rank_proxy_analysis(args.jsonl)
    proxy = report["rank_proxy_check"]
    print(f"\nsurface rules select the largest-valued option on "
          f"{proxy['min_share_selecting_largest']:.0%}-{proxy['max_share_selecting_largest']:.0%} "
          f"of numeric items, so they track value rank rather than a separate cue")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
