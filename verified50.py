#!/usr/bin/env python3
r"""The with-data results on the questions of BixBench-Verified-50 (not registered).

BixBench-Verified-50 \citep{verified50} is a subset of 50 BixBench questions, in 33 capsules, whose
answer keys and wording were reviewed with domain experts; 17 were revised. Its files are gated on
Hugging Face, so the question list here comes from two public sources that pin Phylo's revision
``c77fc8ea4611b546947b8a4710ed36ab72ee233c``: the per-question precision policy of an Apache-2.0 port of
the subset (``SOURCES["ids"]``), and the public per-question grade records of an agent evaluated on it
(``SOURCES["grades"]``), whose grader notes show the reference value for most numeric questions. The two
list the same 50 identifiers, every one a v1.5 question id.

* **Overlap.** How many of the 50 are among v1.5's 105 numeric items, in which key group of the
  pre-specified test (moved, unchanged, inward), and how many are also v1.0 numeric items (same capsule and
  key, as ``leak_origin.py`` matches releases).
* **Keys.** For each overlapping numeric item, the reference value the public grade record shows (or, where
  the agent's answer matched it exactly, that answer) beside v1.5's key: ``VISIBLE_KEYS``. One differs: for
  bix-43-q2 the record accepts 5.812406 as containing a two-decimal reference, which v1.5's 6.02 is not, so
  the subset's key is 5.81. The question wording cannot be compared without the gated file.
* **On the overlap.** Table~\ref{tab:released}'s v1.5 rows (forced and refusal-option grading through the
  released options minus the tolerance, the misses and the correct answers accepted) and the nearest-option
  rule's $U-P$, on the overlapping numeric items and on those whose key the subset confirms, beside all 105;
  and the published v1.0 runs' rows on the v1.0 items in the subset. Intervals at the level the double
  bootstrap finds covers 95% on the items' own capsules (``replication.calibrated``, 2000 x 5000; a level
  that reaches the grid's top without covering is flagged ``level_capped``), and a capsule sign-flip test
  for $U-P$.

    python3 verified50.py        # results/verified50.json
"""
import collections
import json
import pathlib
import zlib

import numpy as np

import bixbench_withdata as bw
import bracketing as br
import degenerate_runs as dg
import leak_origin as lo
import randomization
import replication as rp
import score_decomposition as sd
from option_artifacts import parse_number

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "verified50.json"
SEED = 20260930
SOURCES = {
    "dataset": {"name": "phylobio/BixBench-Verified-50", "revision": "c77fc8ea4611b546947b8a4710ed36ab72ee233c",
                "url": "https://huggingface.co/datasets/phylobio/BixBench-Verified-50", "license": "Apache-2.0",
                "access": "gated: the dataset card is public, the files require an accepted access request",
                "card": "50 questions, 33 capsules, 17 revised; evaluation modes llm_verifier 20, str_verifier 17, "
                        "range_verifier 13"},
    "ids": {"url": "https://github.com/marin-community/harbor/blob/0dcd65486c9956e7cbb561226ca78d2a8a1725cb/"
                   "adapters/bixbench-verified-50/src/bixbench_verified50/precision_policy.json",
            "license": "Apache-2.0", "what": "one entry per question of the subset, pinned to the revision above"},
    "grades": {"url": "https://github.com/omicverse/OmicOS-BixBench/tree/19e5acfdce1605591346b72919463c88fbe188d2/results",
               "license": "PolyForm Noncommercial 1.0.0",
               "what": "per-question grade records of one agent on the subset; only the question identifiers and the "
                       "reference values their grader notes show are used, none of the files is copied"}}
VERIFIED50 = ("bix-11-q1", "bix-11-q2", "bix-12-q2", "bix-12-q4", "bix-12-q5", "bix-12-q6", "bix-14-q1", "bix-16-q1",
              "bix-16-q3", "bix-16-q4", "bix-17-q2", "bix-18-q1", "bix-18-q3", "bix-20-q3", "bix-22-q1", "bix-22-q4",
              "bix-24-q2", "bix-26-q3", "bix-26-q5", "bix-27-q5", "bix-28-q3", "bix-30-q3", "bix-31-q2", "bix-32-q2",
              "bix-34-q2", "bix-34-q5", "bix-35-q1", "bix-35-q2", "bix-37-q1", "bix-37-q4", "bix-38-q1", "bix-41-q5",
              "bix-43-q2", "bix-43-q4", "bix-45-q1", "bix-46-q4", "bix-47-q3", "bix-49-q4", "bix-51-q2", "bix-51-q8",
              "bix-52-q2", "bix-52-q6", "bix-52-q7", "bix-53-q2", "bix-53-q5", "bix-54-q7", "bix-55-q1", "bix-6-q4",
              "bix-61-q2", "bix-61-q5")
# the subset's reference value for each numeric item of v1.5 it holds, as the public grade records show it:
# "note" where the grader's note states it, "exact" where the agent's answer matched it exactly (the answer is
# then the reference), "inferred" where it follows from how the note matched
VISIBLE_KEYS = {
    "bix-11-q1": ("0.05", "note"), "bix-11-q2": ("35%", "note"), "bix-12-q2": ("3.5%", "note"),
    "bix-12-q4": ("6948.0", "note"), "bix-12-q5": ("29", "note"), "bix-12-q6": ("6748.0", "exact"),
    "bix-16-q3": ("3", "exact"), "bix-17-q2": ("2", "exact"), "bix-26-q3": ("11", "exact"),
    "bix-26-q5": ("3", "note"), "bix-28-q3": ("-30.4551", "exact"), "bix-32-q2": ("2", "note"),
    "bix-34-q2": ("2.63", "note"), "bix-34-q5": ("1.95", "note"), "bix-35-q1": ("0.0471", "note"),
    "bix-35-q2": ("3661", "note"), "bix-37-q4": ("2.27", "exact"),
    "bix-43-q2": ("5.81", "inferred: 5.812406 accepted as containing a two-decimal reference"),
    "bix-45-q1": ("7.70e-54", "note"), "bix-46-q4": ("-4.10", "exact"), "bix-49-q4": ("2118", "note"),
    "bix-52-q7": ("19159", "note"), "bix-53-q5": ("0.1", "note: the answer 10.0 accepted as the reference x 100"),
    "bix-55-q1": ("101", "exact"), "bix-61-q2": ("12.1283", "exact"), "bix-61-q5": ("2.68", "note")}
# range keys whose bounds the public records show changed (not among the numeric items, whose options are numbers)
CHANGED_RANGES = {"bix-14-q1": ("(0.6 , 0.7)", "[0.7, 0.8]"), "bix-27-q5": ("(88,89)", "[55, 56]"),
                  "bix-31-q2": ("(-0.45, -0.35)", "[-0.07, -0.05]")}


def same_key(released, shown):
    a, b = parse_number(released), parse_number(shown)
    if a is None or b is None:
        return None
    return abs(a - b) <= 0.005 * max(abs(a), abs(b)) or a == b


def rows_on(rows, qs):
    return [r for r in rows if r["q"] in qs]


def iv(x):
    """A contrast for the log: its mean and interval, or the mean alone."""
    if x is None or x.get("mean") is None:
        return "--"
    ci = "" if x.get("lo") is None else f" [{x['lo']:+5.1f},{x['hi']:+5.1f}]"
    p = x.get("signflip_p")
    return f"{x['mean']:+5.1f}{ci}" + ("" if p is None else f" p={p:.4f}") + (" capped" if x.get("level_capped") else "")


def main():
    items = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    capsule = {q: it["capsule_uuid"] for q, it in items.items()}
    sets = bw.option_sets()
    group = {q: rp.group_of(s) for q, s in sets.items()}
    assert all(q in items for q in VERIFIED50) and len(set(VERIFIED50)) == 50
    numeric = [q for q in VERIFIED50 if q in sets]
    report = {"seed": SEED, "not_registered": True, "sources": SOURCES, "question_ids": list(VERIFIED50)}
    report["overlap"] = {
        "in_v15": sum(q in items for q in VERIFIED50), "capsules": len({capsule[q] for q in VERIFIED50}),
        "numeric": len(numeric), "numeric_capsules": len({capsule[q] for q in numeric}),
        "by_group": dict(collections.Counter(group[q] for q in numeric)),
        "group_sizes_in_v15": dict(collections.Counter(group.values())),
        "eval_modes_in_v15": dict(collections.Counter(items[q].get("eval_mode") for q in VERIFIED50))}
    keys = {}
    for q in numeric:
        shown, how = VISIBLE_KEYS[q]
        keys[q] = {"v1.5": items[q]["ideal"], "verified50": shown, "how_shown": how,
                   "same": same_key(items[q]["ideal"], shown), "group": group[q]}
    report["keys"] = keys
    report["changed_ranges"] = {q: {"v1.5": a, "verified50": b} for q, (a, b) in CHANGED_RANGES.items()}
    confirmed = [q for q in numeric if keys[q]["same"]]
    changed = [q for q in numeric if not keys[q]["same"]]
    report["overlap"]["numeric_key_confirmed"] = len(confirmed)
    report["overlap"]["numeric_key_changed"] = changed
    print(f"{report['overlap']['in_v15']} of 50 in v1.5; {len(numeric)} numeric ({report['overlap']['by_group']}); "
          f"{len(confirmed)} keys confirmed, changed: {changed}")

    # v1.0: the same capsule and key, as leak_origin matches releases
    index = {}
    for r in lo.v10_items():
        index.setdefault((r["capsule"], parse_number(r["options"][0])), r["q"])
    v10_of = {q: index.get((capsule[q], parse_number(items[q]["ideal"]))) for q in VERIFIED50}
    v10_of = {q: m for q, m in v10_of.items() if m}
    v10sets = rp.v10_sets()
    v10_group = {m: rp.group_of(v10sets[m]) for m in v10_of.values() if m in v10sets}
    report["overlap"]["v10"] = {"items": len(v10_of), "of_numeric_overlap": sum(q in v10_of for q in numeric),
                                "by_group": dict(collections.Counter(v10_group.values())),
                                "matched": v10_of}
    print(f"v1.0: {len(v10_of)} items, groups {report['overlap']['v10']['by_group']}")

    # Table 2's v1.5 rows on the overlap, the items with a confirmed key, and all 105
    d0, d2, d3 = sd.d0_runs(), sd.d2_runs(), sd.d3_runs()
    blocks = {"seven configurations": dg.generic(d0, {"forced": "family forced", "refusal": "family may-decline"}),
              "new seeds": dg.generic([r for r in d2 if r["group"] == "new seeds"],
                                      {"forced": "gemma forced", "refusal": "gemma may-decline"}),
              "Qwen3-235B-A22B": dg.generic([r for r in d2 if r["group"] == "Qwen3-235B-A22B"],
                                            {"forced": "gemma forced", "refusal": "gemma may-decline"}),
              "gpt-5.1": dg.generic(d3, {"forced": "gpt-4o forced", "refusal": "gpt-4o may-decline"})}
    parts = {"verified50": set(numeric), "verified50, key confirmed": set(confirmed), "all 105": set(sets)}
    report["table1"] = {}
    for name, rows in blocks.items():
        report["table1"][name] = {}
        for part, qs in parts.items():
            report["table1"][name][part] = {
                how: dg.table1_row(rows_on(rows, qs), how, capsule,
                                   SEED + zlib.crc32(f"{name}|{part}|{how}".encode()))
                for how in ("forced", "refusal")}
            f, d = (report["table1"][name][part][h] for h in ("forced", "refusal"))
            e = lambda x: f"{x['score_minus_tolerance']['mean']:+5.1f} [{x['score_minus_tolerance']['lo']:+5.1f},{x['score_minus_tolerance']['hi']:+5.1f}]"
            print(f"T1 {name:22s} {part:26s} items {f['n_items']:3d} tol {f['tolerance']:5.1f} forced {e(f)} "
                  f"miss acc {f['misses_accepted']:5.1f} correct acc {f['correct_accepted']} | refusal {e(d)} "
                  f"miss acc {d['misses_accepted']:5.1f}", flush=True)

    # the nearest-option rule's U - P on the overlap
    runs = dg.trajectories()
    answers = {run: [(q, t.get("answer")) for q, t in sorted(by_q.items()) if q in sets] for run, by_q in runs.items()}
    pools = {"seven configurations": [r for r in answers if r in br.RUNS],
             "new seeds": [r for r in answers if "|" in r and r.split("|")[0] in ("qwen72b", "llama70b", "gemma27b")],
             "Qwen3-235B-A22B, six runs": [f"qwen3-235b|r{i}" for i in range(6)],
             "gpt-5.1": ["gpt-5.1-react|r0", "gpt-5.1-react|r1"]}
    report["rule_u_minus_p"] = {}
    for name, names in pools.items():
        report["rule_u_minus_p"][name] = {}
        for part, qs in parts.items():
            sub = {n: [(q, a) for q, a in answers[n] if q in qs] for n in names}
            g = dg.rule_gain(sub, {q: s for q, s in sets.items() if q in qs}, capsule,
                             SEED + zlib.crc32(f"{name}|{part}".encode()), heavy=False)
            report["rule_u_minus_p"][name][part] = g
            print(f"U-P {name:26s} {part:26s} moved {iv(g['moved'])} ({g['moved']['n_items']} items, "
                  f"{g['moved']['n_clusters']} capsules) | all {iv(g['all'])}", flush=True)

    # v1.0: the published runs on the subset's v1.0 items
    d1 = sd.d1_runs()
    cap1 = {r["q"]: r["capsule"] for r in d1}
    v10_items = set(v10_of.values())
    report["table1_v10"] = {}
    for model in sd.MODELS:
        rows = dg.generic([r for r in d1 if r["group"] == model],
                          {"forced": "published forced", "refusal": "published may-decline"})
        report["table1_v10"][model] = {}
        for part, qs in (("verified50", v10_items), ("all 159", set(v10sets))):
            report["table1_v10"][model][part] = {
                how: dg.table1_row(rows_on(rows, qs), how, cap1, SEED + zlib.crc32(f"v10|{model}|{part}|{how}".encode()))
                for how in ("forced", "refusal")}
            f = report["table1_v10"][model][part]["forced"]
            d = report["table1_v10"][model][part]["refusal"]
            print(f"T1 v1.0 {model:18s} {part:12s} items {f['n_items']:3d} runs {f['n_runs']:5d} tol {f['tolerance']:5.1f} "
                  f"forced {f['score_minus_tolerance']['mean']:+5.1f} [{f['score_minus_tolerance']['lo']:+5.1f},"
                  f"{f['score_minus_tolerance']['hi']:+5.1f}] miss acc {f['misses_accepted']:5.1f} | refusal "
                  f"{d['score_minus_tolerance']['mean']:+5.1f} miss acc {d['misses_accepted']:5.1f}", flush=True)
    doc = rp.load_extract()
    published = {name: [(q, a) for q, a, _ in doc["open_runs"][name]] for name in rp.OPEN_RUNS}
    report["rule_u_minus_p_v10"] = {}
    for part, qs in (("verified50", v10_items), ("all 159", set(v10sets))):
        sub = {n: [(q, a) for q, a in rows if q in qs] for n, rows in published.items()}
        g = dg.rule_gain(sub, {q: s for q, s in v10sets.items() if q in qs}, cap1,
                         SEED + zlib.crc32(f"v10|{part}".encode()), heavy=False)
        report["rule_u_minus_p_v10"][part] = g
        print(f"U-P v1.0 published {part:12s} moved {iv(g['moved'])} ({g['moved']['n_items']} items) | all "
              f"{iv(g['all'])}", flush=True)
    OUT.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
