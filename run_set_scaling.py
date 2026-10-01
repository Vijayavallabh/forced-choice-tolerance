#!/usr/bin/env python3
r"""Does hiding the rank cost less as agents get better? (not registered)

The repair's credit falls on misses, so an agent that misses less has less to be credited; the question
is whether that is all there is to it. Per run set with the data -- the four published gpt-4o
and Claude 3.5 Sonnet run sets on v1.0, the seven v1.5 run sets and the eight registered new v1.5 runs --
this reads, under the nearest-option rule as registered:

* how often the submitted number is within 5% of the key, over every numeric question and over the keys
  the repair moved to an edge;
* repaired - placebo on those keys and over every numeric item;
* the moved keys' contrast per miss on them: the contrast divided by their miss rate.

Run sets are weighed alike; the trend is a Spearman correlation over run sets with a permutation p-value.

    python3 run_set_scaling.py            # results/run_set_scaling.json
    python3 run_set_scaling.py --latex    # tab:scaling's rows, by how often the run set is within 5%
"""
import collections
import json
import pathlib

import numpy as np

import bixbench_withdata as bw
import bracketing as br
import replication as rp
from answer_numbers import graded
from probe_analysis import spearman

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "run_set_scaling.json"
SEED = 20260926
PERMUTATIONS = 5000


def one(trajectories, sets):
    both = {q: s for q, s in sets.items() if "placebo" in s and "repaired" in s}
    moved = {q for q in both if rp.group_of(sets[q]) == "moved to an edge"}
    gain, hit = collections.defaultdict(list), collections.defaultdict(list)
    for q, a in trajectories:
        if q not in sets:
            continue
        # the share within 5% counts every numeric item; only the contrast needs both rewrites
        hit[q].append(float(bool(a) and bool(graded(a, sets[q]["released"][0], 0.05))))
        if q in both:
            gain[q].append(bw.nearest_is_key(a, sets[q]["repaired"]) - bw.nearest_is_key(a, sets[q]["placebo"]))
    item = lambda d, qs: 100.0 * float(np.mean([np.mean(d[q]) for q in qs if q in d]))
    out = {"within_5pct": item(hit, list(hit)), "within_5pct_moved": item(hit, moved),
           "moved": item(gain, moved), "all": item(gain, list(gain)), "n_runs": sum(map(len, hit.values()))}
    miss = 1.0 - out["within_5pct_moved"] / 100.0
    out["moved_per_miss"] = out["moved"] / miss if miss > 0 else None
    return out


NAMES = {"4o_open_image": "gpt-4o, images", "4o_open_no_image": "gpt-4o, no images",
         "claude_open_image": "Claude 3.5 Sonnet, images", "claude_open_no_image": "Claude 3.5 Sonnet, no images",
         "qwen72b": "Qwen2.5-72B, text", "llama70b": "Llama-3.3-70B, text", "gemma27b": "gemma-3-27b, text",
         "qwen72b-react": "Qwen2.5-72B, published", "llama70b-react": "Llama-3.3-70B, published",
         "glm45air-react": "GLM-4.5-Air, published", "qwen3a3b-react": "Qwen3-30B-A3B, published",
         "qwen3-235b": "Qwen3-235B-A22B"}


# the new runs' rollouts as the paper names them: a text-protocol agent's first run is seed 1, so its two new
# rollouts are seeds 2 and 3; Qwen3-235B-A22B has only its own two runs
SEEDS = {"r1": "seed 2", "r2": "seed 3"}
RUNS_Q3 = {"r0": "run 1", "r1": "run 2"}


def label(name):
    """``v1.5 new|gemma27b|r1`` -> ``v1.5, gemma-3-27b, text, seed 2``."""
    release, rest = name.split("|", 1)
    parts = rest.split("|")
    tags = [(RUNS_Q3 if parts[0] == "qwen3-235b" else SEEDS)[t] for t in parts[1:]]
    return ", ".join([release.replace(" new", ""), NAMES[parts[0]]] + tags)


def table_rows(report):
    rows = sorted(report["run_sets"].items(), key=lambda kv: (kv[1]["within_5pct"], kv[0]))
    return [f"{label(n)} & ${r['within_5pct']:.1f}$ & ${r['moved']:+.1f}$ & ${r['all']:+.1f}$ & "
            f"${r['moved_per_miss']:+.1f}$\\\\" for n, r in rows]


def main():
    import sys
    if "--latex" in sys.argv:
        for row in table_rows(json.loads(OUT.read_text())):
            print(row)
        return
    doc = rp.load_extract()
    v10, v15 = rp.v10_sets(), bw.option_sets()
    rows = {}
    for name in rp.OPEN_RUNS:
        rows[f"v1.0|{name}"] = one([(q, a) for q, a, _ in doc["open_runs"][name]], v10)
    for run in br.RUNS:
        rows[f"v1.5|{run}"] = one([(r["question_id"], r["answer"]) for r in br.rows_of(run)
                                   if r["numeric"] and r["condition"] == "data"], v15)
    for name, conds in sorted(rp.d2_runs().items()):
        rows[f"v1.5 new|{name}"] = one(conds.get("data", []), v15)
    names = sorted(rows)
    report = {"not_registered": True, "run_sets": rows, "trend": {}}
    rng = np.random.default_rng(SEED)
    for x in ("within_5pct", "within_5pct_moved"):
        for y in ("moved", "all", "moved_per_miss"):
            xs = [rows[n][x] for n in names]
            ys = [rows[n][y] for n in names]
            rho = spearman(xs, ys)
            null = [spearman(xs, list(rng.permutation(ys))) for _ in range(PERMUTATIONS)]
            p = float(np.mean([abs(r) >= abs(rho) - 1e-12 for r in null]))
            report["trend"][f"{y} on {x}"] = {"spearman_rho": rho, "p_value": p, "n_run_sets": len(names),
                                             "permutations": len(null)}
    per_miss = [rows[n]["moved_per_miss"] for n in names]
    report["moved_per_miss_range"] = [min(per_miss), max(per_miss)]
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    for n in names:
        r = rows[n]
        print(f"{n:34s} within {r['within_5pct']:5.1f} (moved {r['within_5pct_moved']:5.1f})  moved {r['moved']:+6.1f}  "
              f"all {r['all']:+5.1f}  per miss {r['moved_per_miss']:+6.1f}")
    for k, v in report["trend"].items():
        print(f"{k:36s} rho {v['spearman_rho']:+.2f}  p {v['p_value']:.4f}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
