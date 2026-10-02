#!/usr/bin/env python3
r"""Table~\ref{tab:instrument}'s cells, read over chance from the shipped rollouts.

A cell could be read as a margin over its clean control, but the control's
expected accuracy is exactly $1/k$ for a reader that sees only the options
(Proposition~\ref{prop:exch}), so subtracting a realised control adds its noise
and, where it happens to sit below chance, inflates the reading. Each cell is
read here against chance with the released arm's own cluster bootstrap, at the
level that covers each file's shape (bootstrap_calibration.py), and the control
is reported beside it as a check. Two things about the cells are recorded
rather than assumed: both prompts carry ``Question: [withheld]``, and the chat
framing's letter read-out on MMLU-Pro was given the interpreter's rules in its
system message where the generated read-out was not.

    python3 instrument_cells.py
"""
import argparse
import json
import pathlib

import numpy as np

from prompt_contrasts import exact_null, interval, load

LEVEL = {"mmlupro": 0.985, "bixall": 0.9675}
CELLS = [
    ("mmlupro", "BixBench's template", "letter", "qwen14b_mmlupro_bixprompt_argmax"),
    ("mmlupro", "BixBench's template", "generated", "qwen14b_mmlupro_bixprompt_generated"),
    ("mmlupro", "``withheld, read the options''", "letter", "qwen14b_mmlupro_argmax_withheld_aware"),
    ("mmlupro", "``withheld, read the options''", "generated", "qwen14b_mmlupro_steered_notools"),
    ("mmlupro", "chat framing", "letter", "qwen14b_mmlupro_argmax_withheld"),
    ("mmlupro", "chat framing", "generated", "qwen14b_mmlupro_neutral_notools"),
    ("bixall", "BixBench's template", "letter", "qwen14b_bixall_bixprompt_argmax"),
    ("bixall", "BixBench's template", "generated", "qwen14b_bixall_bixprompt_generated"),
    ("bixall", "chat framing", "letter", "qwen14b_bixall_neutral_argmax"),
    ("bixall", "chat framing", "generated", "qwen14b_bixall_neutral_notools"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--output", default="results/instrument_cells.json")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    if args.latex:
        rep = json.loads(pathlib.Path(args.output).read_text())
        rows = {}
        for c in rep["cells"]:
            rows.setdefault((c["file"], c["framing"]), {})[c["readout"]] = c
        for (f, framing), r in rows.items():
            cells = []
            for ro in ("letter", "generated"):
                x = r[ro]
                cells.append(f"${x['accuracy']:.1f}$ & ${x['over_chance']:+.1f}$ "
                             f"$[{x['interval'][0]:+.1f},{x['interval'][1]:+.1f}]$")
            print(f"{framing} & " + " & ".join(cells) + "\\\\")
        return
    report = {"levels": LEVEL, "cells": []}
    for name, framing, readout, stem in CELLS:
        d = load(f"build/dumps/{stem}.jsonl")
        rel, clean = d["file"], d["clean"]
        k = rel[0]["k"]
        chance = 100.0 / k
        hits = np.array([r["answer"] == r["gold"] for r in rel], float)
        clusters = sorted({r["cluster"] for r in rel})
        by_c = {c: [i for i, r in enumerate(rel) if r["cluster"] == c] for c in clusters}
        rng = np.random.default_rng(0)
        draws = np.array([100.0 * hits[[i for c in rng.choice(clusters, size=len(clusters)) for i in by_c[c]]].mean()
                          - chance for _ in range(args.bootstrap)])
        acc_clean = 100.0 * np.mean([r["answer"] == r["gold"] for r in clean])
        null = exact_null(clean)
        entry = {"file": name, "framing": framing, "readout": readout, "dump": stem,
                 "accuracy": 100.0 * hits.mean(), "over_chance": 100.0 * hits.mean() - chance,
                 "interval": interval(draws, LEVEL[name]),
                 "unparsed": 100.0 * float(np.mean([r["answer"] == "" for r in rel])),
                 "control_minus_chance": acc_clean - chance,
                 "control_null_percentile": float(100.0 * np.mean(null <= acc_clean))}
        report["cells"].append(entry)
        print(f"{name:7s} {framing:32s} {readout:9s} {entry['accuracy']:5.1f}%  {entry['over_chance']:+5.1f} "
              f"[{entry['interval'][0]:+5.1f},{entry['interval'][1]:+5.1f}]  unparsed {entry['unparsed']:.1f}%  "
              f"control {entry['control_minus_chance']:+.1f} (p{entry['control_null_percentile']:.0f})")
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
