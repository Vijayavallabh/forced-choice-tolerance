#!/usr/bin/env python3
r"""Every clean control this paper reads, against its exact null.

A clean control places each item's key uniformly among its options, so with the
reader's picks held fixed the only randomness left in its accuracy is that
placement. Re-drawing it gives the exact null (Proposition~\ref{prop:exch}), and
a control whose accuracy falls outside the central $95\%$ of that null is either
a rare placement or a pipeline that is not reading what it should. This counts
both kinds over every dumped arm. The counts are not independent checks: every
cell on one file reads the same realised placement.

    python3 control_null.py
"""
import argparse
import glob
import json
import os
import pathlib

import numpy as np

from prompt_contrasts import exact_null, load


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", default="build/dumps/*.jsonl")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--output", default="results/control_null.json")
    args = ap.parse_args()
    cells = {}
    paths = sorted(glob.glob(args.dumps)) or [
        p for p in sorted(glob.glob("results/agentic_dumps/*.jsonl.gz"))
        if not os.path.basename(p).startswith(("prompt_factorial_", "free_"))]
    for path in paths:
        stem = os.path.basename(path).split(".jsonl")[0]
        try:
            clean = load(path).get("clean") or []
        except KeyError:                        # a dump with arms of another design (free response)
            continue
        if not clean:
            continue
        k = clean[0]["k"]
        acc = 100.0 * float(np.mean([r["answer"] == r["gold"] for r in clean]))
        null = exact_null(clean, draws=args.draws)
        cells[stem] = {"control_minus_chance": acc - 100.0 / k,
                       "null_percentile": 100.0 * float(np.mean(null <= acc)),
                       "unparsed": 100.0 * float(np.mean([r["answer"] == "" for r in clean]))}
    outside = sorted(c for c, v in cells.items() if not 2.5 <= v["null_percentile"] <= 97.5)
    report = {"n_controls": len(cells), "outside_central_95": outside, "cells": cells}
    for c in outside:
        v = cells[c]
        print(f"{c:44s} control {v['control_minus_chance']:+6.2f}  percentile {v['null_percentile']:5.1f}  "
              f"unparsed {v['unparsed']:4.1f}%")
    print(f"{len(outside)} of {len(cells)} controls outside the central 95% of their exact null")
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
