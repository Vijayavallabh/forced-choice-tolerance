#!/usr/bin/env python3
r"""Every cell read three ways: against chance, against its control, and split
by whether the option values are numbers.

Two objections motivate this. The first: the margins the paper leads with are
over a resampled clean control, and on BixBench that control sits below chance,
so the control-relative reading is the larger of the two and the comparator is
doing work. Both are printed here, side by side, for every cell.

The second: the $205$-item BixBench claim is not localised to a mechanism,
because half the file's options are not numbers and the rank channel the paper
measures is a statement about values. The split is made on the rollout itself --
an item is numeric when every one of its option strings parses as a number --
so it needs no join against a question file and is exactly the condition under
which a rank is defined.

    python3 cell_split.py --dump build/dumps/qwen14b_bixall_bixprompt_argmax.jsonl

``--latex`` emits the rows the validator pins.
"""
import argparse
import json
import pathlib
import re

import numpy as np

import arm_intervals

# BixBench's template puts an "IMPORTANT:" instruction *after* the options, so
# a greedy match to end-of-string swallows it and the last option line fails to
# parse -- which silently classified all 205 BixBench items as non-numeric.
OPTIONS = re.compile(r"Options:\n(.*?)(?:\nIMPORTANT:|IMPORTANT:|\Z)", re.DOTALL)
LINE = re.compile(r"\(([A-Z])\)\s*(.*)")


def values(user):
    """The option strings a rollout read, in presented order, or None."""
    found = OPTIONS.search(user)
    if not found:
        return None
    out = []
    for line in found.group(1).strip().splitlines():
        head = LINE.match(line.strip())
        if not head:
            return None
        out.append(head.group(2).strip())
    return out


def numeric(user):
    """True when every option parses as a number -- when a rank is defined."""
    options = values(user)
    if not options:
        return False
    for option in options:
        try:
            float(option.replace(",", "").rstrip("%").replace("E", "e"))
        except ValueError:
            return False
    return True


def cell(path, predicate, bootstrap, level):
    """(vs chance, vs control) for one dump, optionally on a subset of items.

    The control is subset by *item index* and not by its own option text. Its
    option sets are resampled from the file's pooled strings, so a control row
    can be numeric where the item it stands in for is not; selecting it by text
    breaks the pairing and the first version of this printed a numeric interval
    twenty points wider than the whole file's as a result.
    """
    resolved = arm_intervals.resolve(path)
    file_rows = arm_intervals.load(resolved, "file")
    clean_rows = arm_intervals.load(resolved, "clean")
    if predicate is not None:
        keep = {r["item"] for r in file_rows if predicate(r["user"])}
        file_rows = [r for r in file_rows if r["item"] in keep]
        clean_rows = [r for r in clean_rows if r["item"] in keep]
    if not file_rows or not clean_rows:
        return None
    file_draws, file_margin, clusters, effective = arm_intervals.draws(
        file_rows, bootstrap, 0)
    clean_draws, clean_margin, _, _ = arm_intervals.draws(clean_rows, bootstrap, 1)
    unparsed = 100.0 * sum(1 for r in file_rows if r["answer"] is None) / len(file_rows)
    return {"n_rollouts": len(file_rows), "n_items": len({r["item"] for r in file_rows}),
            "n_clusters": clusters, "n_effective": effective,
            "accuracy": 100.0 * np.mean([1.0 if r["answer"] == r["gold"] else 0.0
                                         for r in file_rows]),
            "control_accuracy": 100.0 * np.mean([1.0 if r["answer"] == r["gold"] else 0.0
                                                 for r in clean_rows]),
            "unparsed": unparsed,
            "vs_chance": file_margin,
            "vs_chance_interval": arm_intervals.interval(100.0 * file_draws, level),
            "control_vs_chance": clean_margin,
            "vs_control": file_margin - clean_margin,
            "vs_control_interval": arm_intervals.interval(
                100.0 * (file_draws - clean_draws), level)}


CELLS = {
    "bix template argmax": "build/dumps/qwen14b_bixall_bixprompt_argmax.jsonl",
    "bix template generated": "build/dumps/qwen14b_bixall_bixprompt_generated.jsonl",
    "bix neutral argmax": "build/dumps/qwen14b_bixall_neutral_argmax.jsonl",
    "bix neutral generated": "build/dumps/qwen14b_bixall_neutral_notools.jsonl",
    "pro template argmax": "build/dumps/qwen14b_mmlupro_bixprompt_argmax.jsonl",
    "pro template generated": "build/dumps/qwen14b_mmlupro_bixprompt_generated.jsonl",
    "pro steered argmax": "build/dumps/qwen14b_mmlupro_argmax_withheld_aware.jsonl",
    "pro steered generated": "build/dumps/qwen14b_mmlupro_steered_notools.jsonl",
    "pro neutral argmax": "build/dumps/qwen14b_mmlupro_argmax_withheld.jsonl",
    "pro neutral generated": "build/dumps/qwen14b_mmlupro_neutral_notools.jsonl",
}

SPLITS = {"all": None,
          "numeric": numeric,
          "text": lambda user: not numeric(user)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default=None, help="one dump instead of every cell")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--output", default="results/cell_split.json")
    args = ap.parse_args()

    wanted = ({"given": args.dump} if args.dump else CELLS)
    report = {"level": args.level, "cells": {}}
    for name, path in wanted.items():
        if not arm_intervals.resolve(path).exists():
            print(f"  (missing {path})")
            continue
        report["cells"][name] = {}
        for split, predicate in SPLITS.items():
            got = cell(path, predicate, args.bootstrap, args.level)
            if got:
                report["cells"][name][split] = got

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n",
                                         encoding="utf-8")

    header = (f"{'cell':26s} {'split':8s} {'items':>5s} {'acc':>6s} {'ctrl':>6s} "
              f"{'vs chance':>22s} {'vs control':>22s}")
    print(header)
    print("-" * len(header))
    for name, splits in report["cells"].items():
        for split, s in splits.items():
            print(f"{name:26s} {split:8s} {s['n_items']:5d} "
                  f"{s['accuracy']:6.2f} {s['control_accuracy']:6.2f} "
                  f"{s['vs_chance']:+6.2f} [{s['vs_chance_interval'][0]:+6.2f},"
                  f"{s['vs_chance_interval'][1]:+6.2f}] "
                  f"{s['vs_control']:+6.2f} [{s['vs_control_interval'][0]:+6.2f},"
                  f"{s['vs_control_interval'][1]:+6.2f}]")
    print(f"\nwrote {args.output}")

    if args.latex:
        print()
        for name, splits in report["cells"].items():
            if "numeric" not in splits:
                continue
            for split in ("numeric", "text"):
                s = splits[split]
                print(f"{name} & {split} & ${s['n_items']}$ & "
                      f"${s['vs_chance']:+.1f}$ "
                      f"$[{s['vs_chance_interval'][0]:+.1f},"
                      f"{s['vs_chance_interval'][1]:+.1f}]$ & "
                      f"${s['vs_control']:+.1f}$ "
                      f"$[{s['vs_control_interval'][0]:+.1f},"
                      f"{s['vs_control_interval'][1]:+.1f}]$\\\\")


if __name__ == "__main__":
    main()
