#!/usr/bin/env python3
r"""Three questions about the misses that grading through options accepts, from the recorded gradings.

* **Does the key's rank steer what a grader selects when the answer gives nothing to place?** For
  answers that contain no number, the rank among the four values of the option each grading selects,
  through the released options ($R$), the rank-preserving rewrite ($P$) and the rank-uniform one ($U$),
  and how often it is the key. A grader that follows v1.5's rank cue would select the second-smallest
  option more often than the others and accept more such answers through $R$, whose key is
  second-smallest on half the items, than through $U$.
* **How wrong are the misses accepted through $R$?** Their distance from the key, the number the
  tolerance reads against the key, by whether the key is the option nearest that number.
* **What moves the keys the rank-uniform rewrite leaves in their class?** The pre-specified test's
  unchanged keys split by what $U$ does to them: the same rank, another middle rank, the same extreme,
  or the other extreme, where a miss on the new open side becomes nearest the key; the nearest-option
  rule's $U-R$ on each part, and its share of the whole.

    python3 miss_anatomy.py        # results/miss_anatomy.json
"""
import collections
import gzip
import json
import pathlib

import numpy as np

import answer_extraction as ax
import bixbench_withdata as bw
import bracketing as br
import published_reads as pr
import replication as rp
import score_decomposition as sd
from answer_numbers import graded

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "miss_anatomy.json"
ARMS = ("released", "placebo", "repaired")
BINS = (("5 to 10%", 0.05, 0.10), ("10 to 25%", 0.10, 0.25), ("25 to 100%", 0.25, 1.0), ("over 100%", 1.0, np.inf))
PUBLISHED_READERS = (("published", None), ("gpt-4o", "gpt-4o"), ("Qwen2.5-72B", "qwen72b"),
                     ("gemma-3-27b", "gemma27b"), ("Llama-3.3-70B", "llama70b"))


def no_number(answer):
    return bool(answer and str(answer).strip()) and ax.category(answer) == "no number"


def pick_table(entries):
    """entries: (arm, options, reads) for answers with no number. Per arm: the share of selections at each of
    the four ranks and of no selection, the share of reads that select the key, and the key's rank share."""
    out = {}
    for arm in ARMS:
        picks, correct, keys, n = collections.Counter(), [], collections.Counter(), 0
        for a, options, reads in entries:
            if a != arm:
                continue
            keys[bw.pick_rank(options[0], options)] += 1
            for x in reads:
                n += 1
                picks["none" if x.get("rank") is None else x["rank"]] += 1
                correct.append(bool(x["correct"]))
        if not n:
            continue
        total_keys = sum(keys.values())
        out[arm] = {"n_reads": n, "select_rank": [100.0 * picks[j] / n for j in range(4)],
                    "select_none": 100.0 * picks["none"] / n, "accepted": 100.0 * float(np.mean(correct)),
                    "key_rank": [100.0 * keys[j] / total_keys for j in range(4)]}
    return out


def published_entries(reader):
    """(arm, options, reads) for the published runs' answers with no number, graded by ``reader``: the
    published forced grades (released options only), or a regrading through R, P and U."""
    sets = rp.v10_sets()
    rows = pr.load_rows() if reader in (None, "gemma27b") else pr.load_rows(pr.rows_path(reader, "forced"))
    out = []
    for r in rows:
        if r["set"] != "D1" or not no_number(r["answer"]):
            continue
        s = sets[r["q"]]
        if reader is None:
            picked = r["published"]["picked"]
            options = s["released"]
            rank = bw.pick_rank(picked, options) if picked in options else None
            out.append(("released", options, [{"correct": r["published"]["correct"], "rank": rank}]))
            continue
        for arm in ARMS:
            if arm in s and r["reads"].get(arm):
                out.append((arm, s[arm], r["reads"][arm]))
    return out


def v15_entries(reader):
    """(arm, options, reads) for the v1.5 runs' answers with no number: the seven configurations graded by
    their own model (``own``) or gemma-3-27b, and the new runs graded by gemma-3-27b."""
    sets = bw.option_sets()
    out = []
    for run in br.RUNS:
        name = br.primary(run) if reader == "own" else reader
        for r in br.rows_of(run):
            if r["condition"] != "data" or not r["numeric"] or not no_number(r["answer"]):
                continue
            for arm in ARMS:
                reads = r["reads"].get(f"{name}|{arm}|forced")
                if reads:
                    out.append((arm, sets[r["question_id"]][arm], reads))
    if reader == "gemma27b":
        for r in pr.load_rows():
            if r["set"] == "D2" and r["condition"] == "data" and no_number(r["answer"]):
                for arm in ARMS:
                    if r["reads"].get(arm):
                        out.append((arm, sets[r["q"]][arm], r["reads"][arm]))
    return out


def accepted_misses():
    """The misses the published forced grades accept through R, by distance from the key and by whether the
    key is the option nearest the number, each question's runs weighted as one question."""
    sets = rp.v10_sets()
    by_q = collections.defaultdict(list)
    distances = []
    for r in pr.load_rows():
        if r["set"] != "D1" or not (r["answer"] and str(r["answer"]).strip()):
            continue
        options = sets[r["q"]]["released"]
        if graded(r["answer"], options[0], sd.TOL) or not r["published"]["correct"]:
            continue
        a, y = ax.last_number(r["answer"], options[0]), bw.parse_number(options[0])
        if a is None:
            kind, where = "no number", "no number"
        else:
            d = abs(a - y) / abs(y) if y else abs(a)
            distances.append(d)
            where = next(name for name, lo, hi in BINS if lo < d <= hi) if d > 0.05 else "5 to 10%"
            kind = sd.single_nearest({"answer": r["answer"], "options": options}) or "neither"
        by_q[r["q"]].append((kind, where))
    # each question weighted as one: its accepted misses shared out over their kinds and distances
    weights = collections.Counter()
    for q, xs in by_q.items():
        for kind, where in xs:
            weights[(kind, where)] += 1.0 / len(xs)
    total = sum(weights.values())
    out = {"n_runs": sum(len(v) for v in by_q.values()), "n_questions": len(by_q), "share": {}}
    for kind in ("key", "other", "neither", "no number"):
        for where in [b[0] for b in BINS] + ["no number"]:
            if weights[(kind, where)]:
                out["share"][f"{kind}|{where}"] = 100.0 * weights[(kind, where)] / total
    runs = [x for xs in by_q.values() for x in xs]
    out["runs"] = {where: 100.0 * sum(1 for _, w in runs if w == where) / len(runs)
                   for where in [b[0] for b in BINS] + ["no number"]}
    # against the widths of BixBench's own range keys (tolerance_check.py): median and upper quartile
    t1 = json.loads((ROOT / "results" / "tolerance_check.json").read_text())["T1"]
    out["range_widths"] = {"median": t1["median"], "upper_quartile": t1["quartiles"][1]}
    out["within_range_width"] = {name: 100.0 * sum(1 for d in distances if d <= w) / len(runs)
                                 for name, w in (("median", t1["median"]), ("upper_quartile", t1["quartiles"][1]))}
    return out


def kept_parts(sets):
    """The unchanged keys by what the rank-uniform rewrite does to their rank."""
    parts = {}
    for q, s in sets.items():
        if rp.group_of(s) != "kept":
            continue
        a, b = bw.pick_rank(s["released"][0], s["released"]), bw.pick_rank(s["repaired"][0], s["repaired"])
        if a == b:
            parts[q] = "same rank"
        elif 0 < a < 3:
            parts[q] = "another middle rank"
        else:
            parts[q] = "the other extreme"
    return parts


def unchanged_split(runs, sets, capsule):
    """The rule's U-R on the unchanged keys, pooled over run sets as the pre-specified test pools them, by
    part; each part's mean, its share of the unchanged keys, and its contribution to their mean."""
    parts = kept_parts(sets)
    _, pooled = rp.gains(runs, sets, "repaired", "released")
    _, pooled_p = rp.gains(runs, {q: s for q, s in sets.items() if "placebo" in s}, "repaired", "placebo")
    kept = [q for q in pooled if q in parts]
    out = {"n_items": len(kept), "mean": 100.0 * float(np.mean([pooled[q] for q in kept])), "parts": {}}
    for part in ("same rank", "another middle rank", "the other extreme"):
        items = [q for q in kept if parts[q] == part]
        if not items:
            continue
        vals = [pooled[q] for q in items]
        vp = [pooled_p[q] for q in items if q in pooled_p]
        out["parts"][part] = {"n_items": len(items), "n_capsules": len({capsule[q] for q in items}),
                              "U-R": 100.0 * float(np.mean(vals)),
                              "U-P": 100.0 * float(np.mean(vp)) if vp else None,
                              "contribution": 100.0 * float(np.sum(vals)) / len(kept)}
    return out


def main():
    report = {"no_number": {"v1.0, published runs": {}, "v1.5, with the data": {}}}
    for label, reader in PUBLISHED_READERS:
        report["no_number"]["v1.0, published runs"][label] = pick_table(published_entries(reader))
    for label, reader in (("own model", "own"), ("gemma-3-27b", "gemma27b")):
        report["no_number"]["v1.5, with the data"][label] = pick_table(v15_entries(reader))
    report["accepted_misses"] = accepted_misses()
    doc = rp.load_extract()
    cap10 = {q: e["capsule"] for q, e in doc["items"].items()}
    runs = {name: [(q, a) for q, a, _ in doc["open_runs"][name]] for name in rp.OPEN_RUNS}
    report["unchanged_keys"] = {"v1.0, published": unchanged_split(runs, rp.v10_sets(), cap10)}
    items15 = {json.loads(l)["question_id"]: json.loads(l) for l in open(ROOT / "data" / "bixbench.jsonl")}
    cap15 = {q: it["capsule_uuid"] for q, it in items15.items()}
    sets15 = bw.option_sets()
    for fam, want in (("v1.5, new seeds", lambda n: not n.startswith("qwen3-235b|")),
                      ("v1.5, Qwen3-235B-A22B", lambda n: n.startswith("qwen3-235b|"))):
        sub = {n: c["data"] for n, c in rp.d2_runs().items() if want(n)}
        report["unchanged_keys"][fam] = unchanged_split(sub, sets15, cap15)
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    for where, graders in report["no_number"].items():
        for label, t in graders.items():
            for arm, e in t.items():
                print(f"{where:22s} {label:14s} {arm:9s} n {e['n_reads']:5d} accepted {e['accepted']:5.1f}  "
                      f"select ranks " + "/".join(f"{v:.0f}" for v in e["select_rank"])
                      + f" none {e['select_none']:.0f}   key ranks " + "/".join(f"{v:.0f}" for v in e["key_rank"]))
    am = report["accepted_misses"]
    print(f"accepted misses (published forced, R): {am['n_runs']} runs on {am['n_questions']} questions; by distance "
          + ", ".join(f"{k} {v:.1f}" for k, v in am["runs"].items()))
    for fam, u in report["unchanged_keys"].items():
        print(f"{fam}: unchanged keys U-R {u['mean']:+.1f} over {u['n_items']}: " + "; ".join(
            f"{p} n={e['n_items']} U-R {e['U-R']:+.1f} (contributes {e['contribution']:+.1f})"
            for p, e in u["parts"].items()))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
