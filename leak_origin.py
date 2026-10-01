#!/usr/bin/env python3
r"""Where v1.5's second-smallest key came from: v1.0's items, which v1.5 kept, and how it rewrote them.

BixBench v1.5 brackets 81% of its 105 numeric keys and puts 51.4% at the second-smallest of
the four values; v1.0 brackets 61% of its 159. This traces the difference to the release
history. A v1.5 numeric item is *carried over* when v1.0 has a numeric item in the same
capsule with the same key value; carried-over items either keep v1.0's three distractors
(the same values) or had them rewritten. Every other v1.5 numeric item is *new*. For each
part: the key-rank shares, the share bracketed, and -- for the rewritten ones -- where the
key sat before and after. Beside it, every distractor's ratio to its key, and the pattern
of how many distractors sit above the key, per release; and the rank shares by the kind of
quantity the key is, read from the question's wording (a p-value, a count, a percentage or
proportion, a fold change or ratio, anything else).

    python3 leak_origin.py            # results/leak_origin.json
    python3 leak_origin.py --latex    # tab:origin's rows
"""
import json
import pathlib
import re
from collections import Counter

import numpy as np

from channel_survey import numeric_ranks
from option_artifacts import parse_number

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "leak_origin.json"
K = 4


def v15_items():
    items = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    by_text = {(it["question"], it["ideal"]): q for q, it in items.items()}
    out = []
    for line in open(ROOT / "build" / "bixbench_numeric_q.jsonl"):
        r = json.loads(line)
        q = by_text[(r["question"], r["ideal"])]
        out.append({"q": q, "capsule": items[q]["capsule_uuid"], "question": r["question"],
                    "options": [r["ideal"], *r["distractors"]]})
    return out


def v10_items():
    return [{"q": r["question_id"], "capsule": r["capsule_uuid"].replace("CapsuleFolder-", ""),
             "question": r["question"], "options": [r["ideal"], *r["distractors"]]}
            for r in map(json.loads, open(ROOT / "build" / "bixbench_v10_numeric_released.jsonl"))]


def rank(options):
    return numeric_ranks(options)[0]


def shares(ranks):
    c = Counter(ranks)
    n = len(ranks)
    return {"n": n, "rank_shares": [100.0 * c[j] / n for j in range(K)],
            "bracketed": 100.0 * sum(c[j] for j in range(1, K - 1)) / n}


# A question asks for a p-value only if it asks what the p-value is: most questions that mention one use it
# as a filter ("How many genes ... (padj < 0.05)"), and their keys are counts or percentages.
ASKS_P = re.compile(
    r"\b(?:adjusted\s+)?p-?val(?:ue)?s?\s+(?:for|of|from|associated|obtained|threshold|cutoff used)\b"
    r"|\bwhat\s+(?:is|was)\s+the\s+(?:\w+\s+){0,3}(?:adjusted\s+)?(?:p-?val(?:ue)?|padj|fdr|q-?value)\b"
    r"|\bsignificance level\s*\(p-?value\)")
ASKS_COUNT = re.compile(r"\bhow many\b|\bwhat is the (?:total )?number of\b")
ASKS_SHARE = re.compile(r"\bwhat (?:is the )?(?:total )?(?:percentage|proportion|fraction|percent)\b")


def kind(question):
    """The quantity a numeric question asks for. A question that asks how many, or what percentage,
    is a count or a percentage even when it filters on a p-value; the remaining questions keep the
    keyword order the table was first built with."""
    t = question.lower()
    asks_count, asks_share = ASKS_COUNT.search(t), ASKS_SHARE.search(t)
    if ASKS_P.search(t) and not asks_count and not asks_share:
        return "p-value"
    if asks_count:
        return "count"
    if asks_share:
        return "percentage or proportion"
    if re.search(r"fold|ratio|log2|odds", t):
        return "fold change or ratio"
    if re.search(r"percent|percentage|proportion|fraction|%", t):
        return "percentage or proportion"
    if re.search(r"how many|number of|count", t):
        return "count"
    return "other"


def above(options):
    """How many distractors exceed the key, for keys of positive value."""
    y = parse_number(options[0])
    return None if y is None or y <= 0 else sum(1 for d in options[1:] if parse_number(d) > y)


ROWS = (("v1.0, every numeric item", None, "v1.0"),
        ("\\quad dropped by v1.5", "v1.0 items v1.5 dropped", None),
        ("\\quad kept by v1.5, as v1.0 wrote them", "v1.0 items v1.5 carried over, as v1.0 wrote them", None),
        ("v1.5, kept from v1.0", "carried over, as v1.5 writes them", None),
        ("\\quad with v1.0's distractors", "carried over with v1.0's distractors", None),
        ("\\quad rewritten: before", "carried over with rewritten distractors, before", None),
        ("\\quad rewritten: after", "carried over with rewritten distractors, after", None),
        ("v1.5, new", "new in v1.5", None),
        ("v1.5, every numeric item", None, "v1.5"))


def table_rows(report):
    """tab:origin's rows: items, the key's rank shares and the share bracketed, per part."""
    rows = []
    for label, part, whole in ROWS:
        sh = report["parts"][part] if part else report[whole]
        rows.append(f"{label} & ${sh['n']}$ & " + " & ".join(f"${x:.0f}$" for x in sh["rank_shares"])
                    + f" & ${sh['bracketed']:.0f}$\\\\")
    return rows


def main():
    import sys
    if "--latex" in sys.argv:
        for row in table_rows(json.loads(OUT.read_text())):
            print(row)
        return
    v15, v10 = v15_items(), v10_items()
    index = {}
    for r in v10:
        index.setdefault((r["capsule"], parse_number(r["options"][0])), r)
    report = {"v1.0": shares([rank(r["options"]) for r in v10]),
              "v1.5": shares([rank(r["options"]) for r in v15])}
    carried, new, kept_same, rewritten, moves = [], [], [], [], Counter()
    matched10 = set()
    for r in v15:
        old = index.get((r["capsule"], parse_number(r["options"][0])))
        if old is None:
            new.append(r)
            continue
        matched10.add(old["q"])
        carried.append((r, old))
        same = sorted(map(parse_number, r["options"][1:])) == sorted(map(parse_number, old["options"][1:]))
        (kept_same if same else rewritten).append((r, old))
        if not same:
            moves[f"{rank(old['options'])}->{rank(r['options'])}"] += 1
    dropped = [r for r in v10 if r["q"] not in matched10]
    report["parts"] = {
        "v1.0 items v1.5 dropped": shares([rank(r["options"]) for r in dropped]),
        "v1.0 items v1.5 carried over, as v1.0 wrote them": shares([rank(o["options"]) for _, o in carried]),
        "carried over, as v1.5 writes them": shares([rank(r["options"]) for r, _ in carried]),
        "carried over with v1.0's distractors": shares([rank(r["options"]) for r, _ in kept_same]),
        "carried over with rewritten distractors, before": shares([rank(o["options"]) for _, o in rewritten]),
        "carried over with rewritten distractors, after": shares([rank(r["options"]) for r, _ in rewritten]),
        "new in v1.5": shares([rank(r["options"]) for r in new])}
    report["rewritten_rank_moves"] = dict(sorted(moves.items()))
    # the rewrite's own pattern: each rewritten item's three distractors over its key, sorted, for the
    # positive keys it put at the second-smallest rank
    trip = [sorted(parse_number(d) / parse_number(r["options"][0]) for d in r["options"][1:])
            for r, _ in rewritten if (parse_number(r["options"][0]) or 0) > 0 and rank(r["options"]) == 1]
    report["rewritten_second_smallest_ratios"] = {"n": len(trip),
                                                  "median": [float(np.median([t[i] for t in trip])) for i in range(3)]}
    report["counts"] = {"v1.5 numeric": len(v15), "carried over": len(carried), "kept distractors": len(kept_same),
                        "rewritten": len(rewritten), "new": len(new), "v1.0 numeric": len(v10),
                        "v1.0 dropped": len(dropped)}
    for name, rows in (("v1.0", v10), ("v1.5", v15)):
        pattern = Counter(above(r["options"]) for r in rows)
        ratios = [parse_number(d) / parse_number(r["options"][0]) for r in rows
                  if (parse_number(r["options"][0]) or 0) > 0 for d in r["options"][1:]]
        ratios = np.array([x for x in ratios if x > 0])
        report[f"{name}|distractors_above_key"] = {str(k): v for k, v in sorted(pattern.items(), key=lambda kv: str(kv[0]))}
        report[f"{name}|log2_ratio_quantiles"] = [float(x) for x in np.percentile(np.log2(ratios), [5, 25, 50, 75, 95])]
    report["by_kind"] = {}
    for name, rows in (("v1.0", v10), ("v1.5", v15)):
        groups = {}
        for r in rows:
            groups.setdefault(kind(r["question"]), []).append(rank(r["options"]))
        report["by_kind"][name] = {k: shares(v) for k, v in sorted(groups.items())}
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    f = lambda s: f"n={s['n']:3d} ranks " + "/".join(f"{x:.0f}" for x in s["rank_shares"]) + f", bracketed {s['bracketed']:.0f}%"
    print("v1.0:", f(report["v1.0"]))
    print("v1.5:", f(report["v1.5"]))
    for k, s in report["parts"].items():
        print(f"  {k:52s} {f(s)}")
    print("rewritten, key rank before->after:", report["rewritten_rank_moves"])
    print("rewritten to the second rank, positive keys: distractor/key medians",
          report["rewritten_second_smallest_ratios"])
    print("distractors above a positive key:", {n: report[f'{n}|distractors_above_key'] for n in ('v1.0', 'v1.5')})
    for name, block in report["by_kind"].items():
        for k, s in block.items():
            print(f"  {name} {k:26s} {f(s)}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
