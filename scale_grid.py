#!/usr/bin/env python3
r"""The two headline cells across families and sizes, with compliance first.

Four models, two of them Qwen2.5, cannot carry a scale claim. The question is
whether the geometry-collection crossover is a property of scale, with a
boundary one could estimate, or a property of one pretraining corpus.

This reads every arm run under ``--condition withheld`` in two cells --
BixBench's own template read one letter at a time, which is what the field
runs, and the neutral framing generated, which is what reads highest -- on
MMLU-Pro's 913 ten-option items against the same matched clean control, and
prints them by parameter count.

**Compliance is printed before the margin, and decides whether the margin is
printed at all.** phi-4 reads -2.11 [-3.92,-0.44] under the neutral framing and
that number means nothing: it produces no final line on 64.7% of released
rollouts and 38.8% of control rollouts, so the difference between the arms is a
difference in refusal propensity. Llama-3.1-8B fails the same cell the same way.
A grid that plots those points is a grid about scaffold compatibility wearing
the label of a grid about option geometry, so a cell is marked UNREADABLE when
either arm refuses more than ``--max-unparsed`` of the time or when the two arms'
rates differ by more than ``--max-unparsed-gap``.

Parameter counts come from the weights themselves -- the safetensors index's
total byte count over two, since everything here is bf16 -- and not from a table
in this file that could drift from what was loaded.

    python3 scale_grid.py
    python3 scale_grid.py --max-unparsed 2.0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re

import numpy as np

RESULTS = pathlib.Path("results")
# (prompt, read-out, holds an interpreter). The third entry matters: Qwen2.5-32B
# has a neutral arm that holds one and reads +7.5, and the rest of the grid does
# not, so keying on prompt and read-out alone would have put two different cells
# in one column and called the difference scale.
CELLS = {("bixbench", "argmax", False): "BixBench's template, one letter",
         ("agent", "generated", False): "question absent, generated, no tool"}


def params_billions(model_name):
    """Parameter count read off the weights, or None if they are not local."""
    repo = "models--" + model_name.replace("/", "--")
    # Where the weights are is a property of the host, not of the paper, so the
    # list is searched rather than assumed and AGENTICLS_HF_CACHES can extend
    # it. Hardcoding one path is why the two 70B-class rows printed their
    # parameter count as NaN: the big weights live on a different volume here
    # than the note that recorded the path said.
    roots = [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME"),
             *(os.environ.get("AGENTICLS_HF_CACHES", "").split(":")),
             os.path.expanduser("~/.cache/huggingface/hub")]
    for root in roots:
        if not root:
            continue
        for index in glob.glob(f"{root}/{repo}/snapshots/*/*.safetensors.index.json"):
            try:
                total = json.load(open(index))["metadata"]["total_size"]
            except (KeyError, ValueError, OSError):
                continue
            return total / 2 / 1e9          # bf16: two bytes a parameter
        # single-shard repos have no index; fall back to the file size
        for weights in glob.glob(f"{root}/{repo}/snapshots/*/model.safetensors"):
            return os.path.getsize(os.path.realpath(weights)) / 2 / 1e9
        # a sharded repo whose index was not fetched: sum the shards
        shards = glob.glob(f"{root}/{repo}/snapshots/*/model-*-of-*.safetensors")
        if shards:
            return sum(os.path.getsize(os.path.realpath(w))
                       for w in shards) / 2 / 1e9
    return None


def calibrated_level(calibration="results/bootstrap_calibration.json",
                     shape="MMLU-Pro"):
    """The nominal level a percentile bootstrap needs here to cover 95%.

    MMLU-Pro's sixty clusters are source subjects of wildly unequal size, so
    Kish's effective count is about eleven and a nominal 95% interval covers
    0.90. Every cell in this grid is on that file, so reading it at 95% would
    call a cell clear that the file cannot resolve -- which is exactly what
    Qwen2.5-72B's template cell does: +2.48 [+0.23,+4.80] nominal, and
    [-0.49,+5.25] at the level that covers.
    """
    path = pathlib.Path(calibration)
    if not path.exists():
        return 0.95
    for entry in json.loads(path.read_text())["arms"]:
        if entry["arm"].split(",")[0].split()[0] != shape:
            continue
        return next(c for c in entry["cells"]
                    if c["as_observed"])["level_for_95_difference"]
    return 0.95


def recompute(dump, level):
    """(margin, nominal 95%, calibrated interval) from the shipped rollouts.

    The bootstrap is ``arm_intervals``' own, imported rather than copied: two
    implementations of one interval are two intervals.
    """
    import arm_intervals

    path = arm_intervals.resolve(dump)
    if not path.exists():
        return None
    file_draws, file_margin, _, _ = arm_intervals.draws(
        arm_intervals.load(path, "file"), 2000, 0)
    clean_draws, clean_margin, _, _ = arm_intervals.draws(
        arm_intervals.load(path, "clean"), 2000, 1)
    difference = 100.0 * (file_draws - clean_draws)
    return (file_margin - clean_margin,
            arm_intervals.interval(difference, 0.95),
            arm_intervals.interval(difference, level))


def dump_index():
    """Every dumped arm, keyed by what uniquely identifies it in a result file.

    Arms written before ``agentic_probe.py`` recorded its own ``--dump`` path
    carry no pointer back to their rollouts, and the dump file names follow the
    sweep script's tags rather than anything derivable from the model id. The
    rollouts themselves are the identifier: how many there are and how many were
    right. A collision would mean two arms with identical counts, which is
    reported rather than resolved.
    """
    import arm_intervals

    index, collisions = {}, set()
    for path in (sorted(glob.glob("build/dumps/*.jsonl"))
                 + sorted(glob.glob("results/agentic_dumps/*.jsonl.gz"))):
        try:
            rows = arm_intervals.load(path, "file")
        except (OSError, ValueError):
            continue
        # The prompt factorial's dumps share the directory and are not arms of
        # this grid: their rows record a cell's ``pick``, not an ``answer``.
        if not rows or "answer" not in rows[0]:
            continue
        correct = sum(1 for r in rows if r["answer"] == r["gold"])
        key = (len(rows), correct)
        # The bundle ships a gzipped copy of every dump, so the same arm is on
        # disk twice under one name. Compare the names, not the paths, or every
        # shipped arm looks like a collision with itself and is dropped.
        name = pathlib.Path(path).name.removesuffix(".gz").removesuffix(".jsonl")
        if key in index and pathlib.Path(index[key]).name.removesuffix(
                ".gz").removesuffix(".jsonl") != name:
            collisions.add(key)
        index.setdefault(key, path)
    for key in collisions:
        index.pop(key, None)
    return index


FILES = {
    # selector: (substring of the arm's --items, calibration shape, label)
    "mmlupro": ("mmlu_pro", "MMLU-Pro", "MMLU-Pro's 913 ten-option items"),
    "bixbench": ("bixbench_v15", "BixBench", "BixBench's 205 released items"),
}


def collect(paths, wanted="mmlu_pro"):
    """Every ``withheld`` arm on one file, keyed by model and cell."""
    seen = {}
    for path in paths:
        if not pathlib.Path(path).exists():
            continue
        for key, arm in json.loads(pathlib.Path(path).read_text()).items():
            if arm["condition"] != "withheld":
                continue
            if wanted not in arm["items"] or "placebo" in arm["items"]:
                continue
            # arms run before --prompt existed carry neither key, and every one
            # of them is the agent framing with a generated answer
            cell = (arm.get("prompt", "agent"), arm.get("readout", "generated"),
                    bool(arm.get("tools", True)))
            if cell not in CELLS:
                continue
            seen[(arm["model"], cell)] = arm
    return seen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="mmlupro", choices=sorted(FILES),
                    help="which benchmark's grid to print; each carries its own "
                         "bootstrap calibration, and BixBench's balanced capsules "
                         "cover at a level MMLU-Pro's lopsided subjects do not")
    ap.add_argument("--max-unparsed", type=float, default=5.0,
                    help="a cell where either arm refuses more than this is unreadable")
    ap.add_argument("--max-unparsed-gap", type=float, default=3.0,
                    help="and so is one where the two arms' rates differ by more")
    ap.add_argument("--output", default=None,
                    help="defaults to results/scale_grid_<file>.json")
    ap.add_argument("--latex", action="store_true",
                    help="also print the appendix table's rows, so the numbers "
                         "in the paper are transcribed by a program and pinned "
                         "against this file rather than typed twice")
    args = ap.parse_args()

    wanted, shape, label = FILES[args.file]
    arms = collect(sorted(glob.glob("results/agentic_arms_*.json"))
                   + sorted(glob.glob("results/agentic_*grid*.json"))
                   + sorted(glob.glob("results/agentic_bixarms_*.json")), wanted)
    level = calibrated_level(shape=shape)
    print(f"{label}: intervals at {100 * level:.1f}% nominal, the level that "
          f"covers 95% on {shape}'s own clusters\n")
    dumps = dump_index()

    def dump_for(arm):
        """The arm's own dump: recorded if it was, fingerprinted if it was not."""
        if arm.get("dump"):
            return arm["dump"]
        n = arm["file"]["n_rollouts"]
        correct = round(arm["file"]["accuracy"] / 100.0 * n)
        return dumps.get((n, correct))

    # named from the shape, not hardcoded: with --file bixbench this line used
    # to credit the level to MMLU-Pro's clusters, which is a different file with
    # a different covering level.
    print(f"intervals at {100 * level:.1f}% nominal, which is what covers 95% on "
          f"{shape}'s own clusters\n")
    models = sorted({m for m, _ in arms},
                    key=lambda m: (params_billions(m) or 0.0, m))

    def cell_key(cell):
        prompt, readout, tools = cell
        return f"{prompt}/{readout}" + ("/tools" if tools else "")

    report = {"file": args.file, "calibrated_level": level,
              "cells": {cell_key(c): lab for c, lab in CELLS.items()},
              "max_unparsed": args.max_unparsed,
              "max_unparsed_gap": args.max_unparsed_gap, "models": []}
    header = f"{'model':34s} {'B':>6s} | " + " | ".join(
        f"{label + '  (margin [95%] unparsed)':60s}" for label in CELLS.values())
    print(header)
    print("-" * len(header))
    for model in models:
        size = params_billions(model)
        row = {"model": model, "parameters_billions": size, "cells": {}}
        line = f"{model.split('/')[-1][:34]:34s} {size or float('nan'):6.1f} | "
        for cell, label in CELLS.items():
            arm = arms.get((model, cell))
            if arm is None:
                line += f"{'not run':>60s} | "
                continue
            up_f = arm["file"]["unparsed"]
            up_c = arm["clean"]["unparsed"]
            readable = (max(up_f, up_c) <= args.max_unparsed
                        and abs(up_f - up_c) <= args.max_unparsed_gap)
            entry = {"accuracy": arm["file"]["accuracy"],
                     "clean_accuracy": arm["clean"]["accuracy"],
                     "margin_over_clean": arm["margin_over_clean"],
                     "ci95": arm["margin_over_clean_ci95"],
                     "unparsed_file": up_f, "unparsed_clean": up_c,
                     "readable": readable}
            # Read the cell at the level that covers, not at the nominal one.
            # Qwen2.5-72B's template cell clears at 95% and does not at 98.2%,
            # and it is the largest template reading in the grid, so the whole
            # "nothing clears" claim turns on getting this right.
            _dump = dump_for(arm)
            rebuilt = recompute(_dump, level) if _dump else None
            if rebuilt is not None:
                _, nominal, calibrated = rebuilt
                entry["ci95_rebuilt"] = nominal
                entry["ci_calibrated"] = calibrated
                entry["calibrated_level"] = level
            row["cells"][cell_key(cell)] = entry
            lo, hi = entry.get("ci_calibrated", entry["ci95"])
            if readable:
                cleared = "*" if lo > 0 else " "
                line += (f"{entry['margin_over_clean']:+6.1f} "
                         f"[{lo:+6.1f},{hi:+6.1f}]{cleared} {up_f:5.1f}%"
                         .ljust(60) + " | ")
            else:
                line += (f"UNREADABLE: refuses {up_f:.1f}% of released and "
                         f"{up_c:.1f}% of control".ljust(60) + " | ")
        print(line)
        report["models"].append(row)

    # the question the grid exists to answer, asked of the readable cells only
    for cell in CELLS:
        key = cell_key(cell)
        pts = [(r["parameters_billions"], r["cells"][key])
               for r in report["models"]
               if key in r["cells"] and r["cells"][key]["readable"]
               and r["parameters_billions"]]
        if len(pts) < 3:
            continue
        x = np.log10([p for p, _ in pts])
        y = np.array([c["margin_over_clean"] for _, c in pts])
        slope, intercept = np.polyfit(x, y, 1)
        clears = [(p, c) for p, c in pts
                  if c.get("ci_calibrated", c["ci95"])[0] > 0]
        report[key + "_trend"] = {
            "n_readable": len(pts), "slope_per_decade": float(slope),
            "intercept": float(intercept),
            "smallest_clearing_billions": min((p for p, _ in clears), default=None),
            "largest_not_clearing_billions": max(
                (p for p, c in pts
                 if c.get("ci_calibrated", c["ci95"])[0] <= 0), default=None),
            "level": level}
        _t = report[key + "_trend"]
        def _b(v):
            return "none" if v is None else f"{v:.1f}B"
        print(f"\n{CELLS[cell]}: {len(pts)} readable cells, "
              f"{slope:+.2f} points per decade of parameters; "
              f"smallest that clears {_b(_t['smallest_clearing_billions'])}, "
              f"largest that does not {_b(_t['largest_not_clearing_billions'])}")

    if args.latex:
        print("\n% --- rows for the appendix grid table ---")
        for row in report["models"]:
            name = re.sub(r"-(instruct|it)$", "", row["model"].split("/")[-1],
                          flags=re.IGNORECASE)
            cells = []
            for cell in CELLS:
                entry = row["cells"].get(cell_key(cell))
                if entry is None:
                    cells.append("--")
                elif not entry["readable"]:
                    cells.append(f"refuses ${entry['unparsed_file']:.0f}\\%$")
                else:
                    lo, hi = entry.get("ci_calibrated", entry["ci95"])
                    cells.append(f"${entry['margin_over_clean']:+.1f}$ "
                                 f"$[{lo:+.1f},{hi:+.1f}]$")
            size = f"{row['parameters_billions']:.1f}" if row["parameters_billions"] else "--"
            print(f"{name} & ${size}$ & " + " & ".join(cells) + r"\\")

    RESULTS.mkdir(exist_ok=True)
    out = args.output or (f"results/scale_grid.json" if args.file == "mmlupro"
                          else f"results/scale_grid_{args.file}.json")
    pathlib.Path(out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
