#!/usr/bin/env python3
r"""What each of the five prompt differences is worth, from ``prompt_factorial.py``'s dumps.

Every cell of the $2^5$ design reads the same rollouts (items and letter
orderings), so a factor's effect is a paired contrast: the mean, over the
sixteen pairs of cells that differ in that factor alone, of the chat level's
accuracy minus the template level's, on the released file. Its interval is a
cluster bootstrap over the file's own groups at the level that covers there
(Table~\ref{tab:bootcal}). The two corners are reported beside it, and so is the
control: every cell's clean-file accuracy against chance, which
Proposition~\ref{prop:exch} fixes at $1/k$ for a reader that sees only options.

    python3 prompt_factorial_summary.py
"""
import argparse
import glob
import gzip
import itertools
import json
import pathlib
import random
import re
from collections import defaultdict

import numpy as np

from prompt_factorial import ALPHABET, FILES, orderings

FACTORS = ("instruction", "force", "reason", "format", "placement")
# covering levels (Table~\ref{tab:bootcal}): a paired difference, and a margin over chance
LEVEL = {"mmlupro": 0.9825, "bixbench": 0.96}
LEVEL_MARGIN = {"mmlupro": 0.985, "bixbench": 0.9675}
CORNERS = {"template": "TTTTT", "chat": "CCCCC", "chat, interpreter rules": "CCCCC+tools"}
MODEL = {"qwen14b": "Qwen2.5-14B", "llama70b": "Llama-3.3-70B", "qwen72b": "Qwen2.5-72B"}
FILE = {"mmlupro": "MMLU-Pro", "bixbench": "BixBench v1.5"}


def control_null(name, cells, seed=0, draws=20000):
    """Each cell's clean-control accuracy as a percentile of its exact null: every clean item's
    key re-drawn uniformly among its options, the held picks fixed (Proposition~\\ref{prop:exch})."""
    _, clean_path, n_draws = FILES[name]
    rows = [json.loads(line) for line in open(clean_path)]
    # the permutations prompt_factorial.orderings drew for the control (its --seed 0, plus one)
    rng = random.Random(0 + 1)
    perm = {}
    for index, row in enumerate(rows):
        options = [row["ideal"]] + list(row["distractors"])
        for draw in range(n_draws):
            permutation = list(range(len(options)))
            rng.shuffle(permutation)
            perm[(index, draw)] = permutation
    gold = {(r["item"], r["draw"]): r["gold"] for r in orderings(rows, n_draws, 0 + 1)}
    assert all(ALPHABET[perm[x].index(0)] == g for x, g in gold.items())
    out = {}
    gen = np.random.default_rng(seed)
    for c, arms in cells.items():
        picks = arms["pick_clean"]
        by_item = defaultdict(list)
        for (item, draw), letter in picks.items():
            if letter and letter in ALPHABET[:len(perm[(item, draw)])]:
                by_item[item].append(perm[(item, draw)][ALPHABET.index(letter)])
        totals = np.zeros(draws)
        for item, chosen in by_item.items():
            key = gen.integers(0, len(perm[(item, 0)]), size=draws)
            for option in chosen:
                totals += key == option
        acc = sum(arms["clean"].values()) / len(picks)
        out[c] = float(100.0 * np.mean(totals / len(picks) <= acc))
    return out


def load(path):
    cells = defaultdict(lambda: {"file": {}, "clean": {}, "pick_clean": {}})
    meta = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            cells[r["cell"]][r["arm"]][(r["item"], r["draw"])] = r["pick"] == r["gold"]
            if r["arm"] == "clean":
                cells[r["cell"]].setdefault("pick_clean", {})[(r["item"], r["draw"])] = r["pick"]
            if r["arm"] == "file":
                meta[(r["item"], r["draw"])] = (r["cluster"], r["k"])
    return cells, meta


def boot(values_by_cluster, reps, seed, level):
    keys = sorted(values_by_cluster)
    sums = np.array([np.sum(values_by_cluster[c]) for c in keys])
    counts = np.array([len(values_by_cluster[c]) for c in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(reps, len(keys)))
    draws = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return [float(100 * lo), float(100 * hi)]


def table_rows(report):
    """tab:factorial: per file and model, the two corners over chance and the five paired main
    effects, each estimate over its interval, under a header row per file."""
    rows = []
    header = {"mmlupro": "\\multicolumn{8}{l}{\\emph{MMLU-Pro, $913$ ten-option items, $60$ subjects}}\\\\",
              "bixbench": "\\multicolumn{8}{l}{\\emph{BixBench v1.5, $205$ four-option items, $59$ capsules}}\\\\"}
    for name in ("mmlupro", "bixbench"):
        tags = [t for t in MODEL if name in report.get(t, {})]
        if name == "bixbench":
            rows.append("\\addlinespace")
        rows.append(header[name])
        for i, tag in enumerate(tags):
            e = report[tag][name]
            vals = [e["corners"]["template"], e["corners"]["chat"]]
            vals = [(v["over_chance"], v["interval"]) for v in vals]
            vals += [(e["factors"][f]["effect"], e["factors"][f]["interval"]) for f in FACTORS]
            rows.append(f"{MODEL[tag]} & " + " & ".join(f"${v:+.1f}$" for v, _ in vals) + "\\\\")
            rows.append("& " + " & ".join(f"\\scriptsize$[{lo:+.1f},{hi:+.1f}]$" for _, (lo, hi) in vals)
                        + "\\\\" + ("[2pt]" if i < len(tags) - 1 else ""))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", default="build/dumps/prompt_factorial_*_*.jsonl.gz")
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--output", default="results/prompt_factorial_summary.json")
    ap.add_argument("--latex", action="store_true", help="print tab:factorial's rows from the summary")
    args = ap.parse_args()
    if args.latex:
        for line in table_rows(json.loads(pathlib.Path(args.output).read_text())):
            print(line)
        return

    report = {}
    paths = sorted(glob.glob(args.dumps)) or sorted(glob.glob("results/agentic_dumps/prompt_factorial_*_*.jsonl.gz"))
    for path in paths:
        m = re.search(r"prompt_factorial_(.+)_(mmlupro|bixbench)\.jsonl\.gz$", path)
        tag, name = m.group(1), m.group(2)
        try:
            cells, meta = load(path)
        except EOFError:                        # a run still writing its dump
            print(f"{tag} {name}: dump still being written; skipped")
            continue
        full = [c for c in cells if len(c) == 5 and c in {"".join(p) for p in itertools.product("TC", repeat=5)}]
        if len(full) < 32:
            print(f"{tag} {name}: only {len(full)} of 32 cells so far; skipped")
            continue
        keys = sorted(meta)
        k = meta[keys[0]][1]
        chance = 1.0 / k
        clusters = [meta[x][0] for x in keys]
        acc = {c: np.array([float(cells[c]["file"][x]) for x in keys]) for c in cells}
        entry = {"k": k, "n_rollouts": len(keys), "cells": {}, "factors": {}, "corners": {}}
        for c in sorted(cells):
            clean = list(cells[c]["clean"].values())
            entry["cells"][c] = {"over_chance": 100 * (acc[c].mean() - chance),
                                 "control_minus_chance": 100 * (np.mean(clean) - chance)}
        for i, factor in enumerate(FACTORS):
            diffs = []
            for other in itertools.product("TC", repeat=4):
                t = list(other)
                t.insert(i, "T")
                cc = list(other)
                cc.insert(i, "C")
                diffs.append(acc["".join(cc)] - acc["".join(t)])
            per_item = np.mean(diffs, axis=0)
            by_c = defaultdict(list)
            for v, cl in zip(per_item, clusters):
                by_c[cl].append(v)
            entry["factors"][factor] = {"effect": 100 * float(per_item.mean()),
                                        "interval": boot(by_c, args.reps, 0, LEVEL[name])}
        for label, c in CORNERS.items():
            if c not in acc:
                continue
            by_c = defaultdict(list)
            for v, cl in zip(acc[c] - chance, clusters):
                by_c[cl].append(v)
            entry["corners"][label] = {"over_chance": 100 * float(acc[c].mean() - chance),
                                       "interval": boot(by_c, args.reps, 0, LEVEL_MARGIN[name])}
        by_c = defaultdict(list)
        for v, cl in zip(acc["CCCCC"] - acc["TTTTT"], clusters):
            by_c[cl].append(v)
        entry["chat_minus_template"] = {"effect": 100 * float((acc["CCCCC"] - acc["TTTTT"]).mean()),
                                        "interval": boot(by_c, args.reps, 0, LEVEL[name])}
        ctrl = [e["control_minus_chance"] for e in entry["cells"].values()]
        entry["control_range"] = [min(ctrl), max(ctrl)]
        pct = control_null(name, cells)
        for c, v in pct.items():
            entry["cells"][c]["control_null_percentile"] = v
        entry["control_outside_central_95"] = sorted(c for c, v in pct.items() if not 2.5 <= v <= 97.5)
        report.setdefault(tag, {})[name] = entry
        f = lambda d: f"{d['effect']:+5.1f} [{d['interval'][0]:+5.1f},{d['interval'][1]:+5.1f}]"
        print(f"{tag} {name}: template {entry['corners']['template']['over_chance']:+5.1f}  chat "
              f"{entry['corners']['chat']['over_chance']:+5.1f}  chat-template {f(entry['chat_minus_template'])}  "
              f"control {entry['control_range'][0]:+.1f}..{entry['control_range'][1]:+.1f}")
        for factor in FACTORS:
            print(f"     {factor:12s} {f(entry['factors'][factor])}")
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
