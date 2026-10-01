#!/usr/bin/env python3
r"""The two rewrites written as round as the released distractors.

The rank-preserving and rank-uniform rewrites ($P$, $U$; ``mcq_audit.redraw_row``) write each new
distractor to the precision of the released distractor it replaces, but not as round: a released
``0.40`` becomes, say, ``0.42``. A key written as round as its released distractors is then the
roundest option more often than a random one, and a rule that picks the option written with the
fewest significant digits is correct on 31.7 and 31.9% of v1.5's $P$ and $U$, against 25.0% on the
released options. This builds the same two rewrites from the same generator at the same seeds, with
each new distractor rounded to exactly the significant digits of the one it replaces
(``match_digits``): $P'$ and $U'$. Their digit counts are then the released ones, option for option,
so the fewest-digits rule scores on them what it scores on the released set. It reports:

* the fewest-digits rule on every option set, and the items on which $P'$ and $U'$ give it the
  released set's credit;
* the key's rank under each set, and $U'$'s key groups (moved to an extreme, moved inward, unchanged);
* the nearest distractor's distance from the key;
* the nearest-option rule's contrasts on BixBench's published v1.0 runs and on the v1.5 runs with
  the data, computed as the pre-specified test computes them (``replication.gains`` and
  ``replication.calibrated``), for the original rewrites and the digit-matched ones side by side.

    python3 digit_matched.py build      # build/bixbench_numeric_{placebo,repaired}_digits.jsonl, v1.0's likewise
    python3 digit_matched.py analyse    # results/digit_matched.json
"""
import argparse
import collections
import gzip
import json
import pathlib
import random

import numpy as np

import bixbench_withdata as bw
import mcq_audit as ma
import replication as rp
from option_artifacts import parse_number

ROOT = pathlib.Path(__file__).resolve().parent
BUILD = ROOT / "build"
U_SEED, P_SEED = 20260918, 20260920        # the seeds of U and P
V15 = BUILD / "bixbench_numeric_q.jsonl"
V15_P, V15_U = BUILD / "bixbench_numeric_placebo.jsonl", BUILD / "bixbench_numeric_repaired.jsonl"
V15_PD, V15_UD = BUILD / "bixbench_numeric_placebo_digits.jsonl", BUILD / "bixbench_numeric_repaired_digits.jsonl"
V10_ITEMS, V10_NUMERIC = rp.ITEMS, rp.NUMERIC
V10_PD, V10_UD = BUILD / "bixbench_v10_numeric_placebo_digits.jsonl", BUILD / "bixbench_v10_repaired_digits.jsonl"
OUT = ROOT / "results" / "digit_matched.json"
SEED = 20260927


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def rewrite(path, seed, preserve_rank=False):
    rows = ma.read_rows(path)
    items, k = ma.build_items(rows, "ideal", "distractors", "capsule_uuid", "question", None)
    return ma.apply_repair(rows, items, "distractors", seed, k=k, preserve_rank=preserve_rank, match_digits=True)


def build(_args):
    report = {}
    for path, dest in ((V15, V15_UD), (V10_ITEMS, V10_UD)):
        rows, changed, infeasible, off_target, _ = rewrite(path, U_SEED)
        # an item no rank could be reached for keeps its released distractors: leave it out instead
        _write(dest, [r for r in rows if [r["ideal"], *r["distractors"]] not in infeasible])
        report[dest.name] = {"changed": changed, "infeasible": len(infeasible), "other_rank": off_target}
    # P' as build_placebo_arm.py draws v1.0's P: one stream over the numeric items, each redrawn at the
    # key's released rank, and an item without such a redraw left out rather than moved to another rank
    for path, dest in ((V15, V15_PD), (V10_NUMERIC, V10_PD)):
        rng, kept, missed = random.Random(P_SEED), [], 0
        for row in (json.loads(l) for l in path.open()):
            drawn = ma.placebo_row([row["ideal"], *row["distractors"]], rng, match_digits=True)
            if drawn is None:
                missed += 1
                continue
            kept.append({**row, "distractors": drawn[1:]})
        _write(dest, kept)
        report[dest.name] = {"changed": len(kept), "infeasible": missed, "other_rank": 0}
    for name, r in report.items():
        print(f"{name}: {r['changed']} redrawn, {r['infeasible']} left out, {r['other_rank']} at another rank")


# ---------------------------------------------------------------- the option sets

def v15_sets(placebo=V15_P, repaired=V15_U):
    """Each v1.5 numeric item's released, rank-preserving and rank-uniform values, key first, by question id."""
    items = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    by_text = {(r["question"], r["ideal"]): q for q, r in items.items()}
    tables = {arm: {(r["question"], r["ideal"]): [r["ideal"], *r["distractors"]] for r in map(json.loads, path.open())}
              for arm, path in (("released", V15), ("placebo", placebo), ("repaired", repaired))}
    sets = {}
    for text in tables["released"]:
        # an item a rewrite left out lacks that set
        sets[by_text[text]] = {arm: t[text] for arm, t in tables.items() if text in t}
    return sets


def v10_sets(placebo=rp.PLACEBO, repaired=rp.REPAIRED):
    tables = {arm: {r["question_id"]: [r["ideal"], *r["distractors"]] for r in rp._rows(path)}
              for arm, path in (("released", V10_NUMERIC), ("repaired", repaired), ("placebo", placebo))}
    sets = {}
    for q, released in tables["released"].items():
        entry = {"released": released}
        for arm in ("repaired", "placebo"):
            options = tables[arm].get(q)
            if options and options[0] == released[0] and rp.numeric_ranks(options) is not None:
                entry[arm] = options
        sets[q] = entry
    return sets


def sig(text):
    """``bracketing.py``'s count of significant digits, with which the fewest-digits rule was reported."""
    t = str(text).strip().rstrip("%").replace(",", "").lower()
    mantissa = t.split("e")[0].lstrip("+-").replace(".", "").lstrip("0").rstrip("0")
    return max(len(mantissa), 1)


def fewest_digits_credit(options):
    digits = [sig(x) for x in options]
    tied = [i for i, d in enumerate(digits) if d == min(digits)]
    return 1.0 / len(tied) if 0 in tied else 0.0


def nearest_distance(options):
    values = [parse_number(x) for x in options]
    return min(abs(v - values[0]) / (abs(v) + abs(values[0])) if abs(v) + abs(values[0]) else 0.0
               for v in values[1:])


def describe(sets, arms):
    """The fewest-digits rule, the key's rank shares and the nearest distractor's distance, per option set."""
    out = {}
    for arm in arms:
        present = [s for s in sets.values() if arm in s]
        credit = [fewest_digits_credit(s[arm]) for s in present]
        ranks = collections.Counter(rp.numeric_ranks(s[arm])[0] for s in present)
        near = [nearest_distance(s[arm]) for s in present]
        out[arm] = {"n_items": len(present), "fewest_digits": 100.0 * float(np.mean(credit)),
                    "same_credit_as_released": sum(fewest_digits_credit(s[arm]) == fewest_digits_credit(s["released"])
                                                   for s in present),
                    "same_digits_as_released": sum(sorted(sig(x) for x in s[arm][1:]) ==
                                                   sorted(sig(x) for x in s["released"][1:]) for s in present),
                    "rank_share": [100.0 * ranks[j] / len(present) for j in range(4)],
                    "bracketed": 100.0 * sum(ranks[j] for j in (1, 2)) / len(present),
                    "nearest_d_median": float(np.median(near))}
    return out


def groups(sets):
    return {g: sorted(q for q, s in sets.items() if rp.group_of(s) == g) for g in rp.GROUPS}


# ---------------------------------------------------------------- the runs

def v15_runs():
    """The v1.5 runs with the data, by the families the pre-specified test pools: the seven configurations
    (their first runs), the new seeds, Qwen3-235B-A22B, and, outside the test, gpt-5.1 and the two current agents
    (last, so that adding them left the other families' draws as they were)."""
    import bracketing as br
    fams = {"seven configurations": {}, "new seeds": {}, "Qwen3-235B-A22B": {}, "gpt-5.1": {}}
    for run in br.RUNS:
        fams["seven configurations"][run] = [(r["question_id"], r["answer"]) for r in br.rows_of(run)
                                             if r["condition"] == "data" and r["rollout"] == 0]
    for name, conds in rp.d2_runs().items():
        fam = "Qwen3-235B-A22B" if name.startswith("qwen3-235b|") else "new seeds"
        fams[fam][name] = conds["data"]
    strong = ROOT / "results" / "strong_agent_rows.json.gz"
    if strong.exists():
        with gzip.open(strong, "rt") as fh:
            rows = json.load(fh)
        for r in rows:
            fams["gpt-5.1"].setdefault(f"gpt-5.1|r{r['rollout']}", []).append((r["question_id"], r["answer"]))
    import frontier_agents as fa
    for run, label in fa.MODELS:
        if label in fa.REPORTED:
            fams[label] = {f"{label}|r0": [(r["question_id"], r["answer"]) for r in fa.load_rows()
                                           if r["model"] == run]}
    return {f: runs for f, runs in fams.items() if runs}


def contrasts(runs, sets, capsule, rng):
    """The rule's U-P, U-R and P-R by key group and over all items, pooled over run sets as the
    pre-specified test pools them, with the test's calibrated intervals; and the answers U newly accepts
    on the moved keys, by distance and side."""
    g = groups(sets)
    g["all"] = sorted(sets)
    out = {"n_items": {k: len(v) for k, v in g.items()}}
    for arm, base in (("repaired", "placebo"), ("repaired", "released"), ("placebo", "released")):
        usable = {q: s for q, s in sets.items() if arm in s and base in s}
        _, pooled = rp.gains(runs, usable, arm, base)
        for name, items in g.items():
            per_item = {q: v for q, v in pooled.items() if q in set(items)}
            out[f"{arm}-{base}|{name}"] = rp.calibrated(per_item, capsule, rng, 2000, 5000) if per_item else None
    nc = rp.newly_credited(runs, sets, set(g["moved to an edge"]))
    out["newly_accepted"] = {k: nc[k] for k in ("n_gained", "share_over_25pct", "share_open_side")}
    return out


def analyse(_args):
    rng = np.random.default_rng(SEED)
    report = {"seed": SEED, "not_registered": True}
    s15 = {"original": v15_sets(), "digit-matched": v15_sets(V15_PD, V15_UD)}
    s10 = {"original": rp.v10_sets(), "digit-matched": v10_sets(V10_PD, V10_UD)}
    for release, sets in (("v1.5", s15), ("v1.0", s10)):
        entry = {}
        for design, s in sets.items():
            entry[design] = {"option_sets": describe(s, ("released", "placebo", "repaired")),
                             "groups": {k: len(v) for k, v in groups(s).items()}}
        # which keys each design moves, and how many both move
        a, b = groups(sets["original"]), groups(sets["digit-matched"])
        entry["moved_by_both"] = len(set(a["moved to an edge"]) & set(b["moved to an edge"]))
        report[release] = entry
    items15 = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    cap15 = {q: it["capsule_uuid"] for q, it in items15.items()}
    doc = rp.load_extract()
    cap10 = {q: e["capsule"] for q, e in doc["items"].items()}
    published = {name: [(q, a) for q, a, _ in doc["open_runs"][name]] for name in rp.OPEN_RUNS}
    report["runs"] = {"v1.0, published": {d: contrasts(published, s, cap10, rng) for d, s in s10.items()}}
    for fam, runs in v15_runs().items():
        report["runs"][f"v1.5, {fam}"] = {d: contrasts(runs, s, cap15, rng) for d, s in s15.items()}
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    f = lambda h: f"{h['mean']:+.1f} [{h['lo']:+.1f},{h['hi']:+.1f}]" if h else "--"
    for release in ("v1.5", "v1.0"):
        for design in ("original", "digit-matched"):
            d = report[release][design]
            print(f"{release} {design:13s} fewest digits " + ", ".join(
                f"{a} {d['option_sets'][a]['fewest_digits']:.1f}" for a in ("released", "placebo", "repaired"))
                  + f"; U bracketed {d['option_sets']['repaired']['bracketed']:.0f}%; groups {d['groups']}"
                  + "; nearest d " + ", ".join(f"{d['option_sets'][a]['nearest_d_median']:.3f}"
                                               for a in ("released", "placebo", "repaired")))
    for key, by in report["runs"].items():
        for design, c in by.items():
            print(f"{key:28s} {design:13s} U-P moved {f(c['repaired-placebo|moved to an edge'])}  "
                  f"unchanged {f(c['repaired-placebo|kept'])}  all {f(c['repaired-placebo|all'])}  "
                  f"P-R moved {f(c['placebo-released|moved to an edge'])}")
    print(f"wrote {OUT.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "analyse"))
    args = ap.parse_args()
    {"build": build, "analyse": analyse}[args.cmd](args)


if __name__ == "__main__":
    main()
