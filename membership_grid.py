#!/usr/bin/env python3
"""The membership arm across the scale grid, and its placebo.

``option_membership.py`` asks one question of one model: inside the released
file, do the rollouts the model gets right carry option blocks it scores as
more-seen?  A positive answer is what recognition of published option sets
would look like.  A null answer is what the paper reported, from one model.

One model is not an answer, because the statistic is a function of model size:
a bigger model's likelihoods separate typical text from atypical text more
sharply, and typical option blocks are also easier ones.  So the same test runs
on every model in the scale grid, beside the placebo that decides what a
positive reading means --- the same difference on the *clean control*, which we
generated, so there is no pretraining exposure in it to detect.  A gap that is
the same size on both arms is measuring difficulty; a gap that appears only on
the released file is measuring the file.

Usage:  python membership_grid.py [--grid results/option_membership_grid.json]
        python membership_grid.py --latex
"""
import argparse
import gzip
import json
import pathlib

# The grid's own order, so the rows read small to large.
SIZES = {
    "meta-llama/Llama-3.2-1B-Instruct": 1.2,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "meta-llama/Llama-3.2-3B-Instruct": 3.2,
    "microsoft/Phi-3.5-mini-instruct": 3.8,
    "google/gemma-3-4b-it": 4.3,
    "allenai/OLMo-2-1124-7B-Instruct": 7.3,
    "Qwen/Qwen2.5-7B-Instruct": 7.6,
    "meta-llama/Llama-3.1-8B-Instruct": 8.0,
    "microsoft/phi-4": 14.7,
    "Qwen/Qwen2.5-14B-Instruct": 14.8,
    "Qwen/Qwen2.5-32B-Instruct": 32.8,
    "meta-llama/Llama-3.3-70B-Instruct": 70.6,
    "Qwen/Qwen2.5-72B-Instruct": 72.7,
}
STATS = ("loglik", "mink", "minkpp")
LABELS = ("pooled", "within key rank")


def control_vocabulary(dump):
    """What share of the control's option strings are published strings.

    This decides what the released-minus-control difference isolates, and the
    two files answer differently.

    BixBench's control is built by ``build_set_control.py``, which draws each
    item's k options i.i.d. from the *file's own pool* of option strings. Every
    string in it was published; what was not published is the grouping. So the
    difference holds string-level familiarity fixed and isolates recognition of
    the published *set* -- which is exactly the confound the paper concedes.

    MMLU-Pro's control redraws numeric values instead, so most of its strings
    are new. There the difference conflates the two, and it does so in the
    conservative direction: string-level recognition would inflate the released
    arm, so a null difference on that file is the stronger null.
    """
    import re

    options = re.compile(r"Options:\n(.*?)(?:\nIMPORTANT:|IMPORTANT:|\Z)",
                         re.DOTALL)
    letter = re.compile(r"^\([A-Z]\)\s*")
    pools = {"file": set(), "clean": set()}
    # The dumps are written to build/dumps/ on the GPU machine and shipped
    # gzipped under results/agentic_dumps/. arm_intervals.resolve knows both
    # places; reading the raw path meant this function -- and so the 12.9%
    # figure in sec:limits -- could not be re-derived from the bundle at all.
    import arm_intervals

    dump = arm_intervals.resolve(dump)
    opener = gzip.open if str(dump).endswith(".gz") else open
    for line in opener(dump, "rt", encoding="utf-8"):
        row = json.loads(line)
        arm = row.get("arm", "file")
        if arm not in pools:
            continue
        found = options.search(row["user"])
        if not found:
            continue
        for option in found.group(1).strip().splitlines():
            pools[arm].add(letter.sub("", option).strip())
    released, control = pools["file"], pools["clean"]
    return {"released_pool": len(released), "control_pool": len(control),
            "published_share": (len(control & released) / len(control)
                                if control else None)}


def clears(cell):
    if cell["difference"] is None or cell["interval"] is None:
        return False
    lo, hi = cell["interval"]
    return lo > 0 or hi < 0


def widest(block):
    """The cell whose reading is furthest from zero, by point estimate."""
    named = [(k, v) for k, v in block.items()
             if v["difference"] is not None and v["interval"] is not None]
    if not named:
        return "---", {"difference": None, "interval": None}
    return max(named, key=lambda kv: abs(kv[1]["difference"]))


def short(model):
    """``meta-llama/Meta-Llama-3.1-8B-Instruct`` -> ``Llama-3.1-8B``."""
    name = model.split("/")[-1]
    for suffix in ("-Instruct", "-instruct", "-it"):
        name = name.removesuffix(suffix)
    return name.removeprefix("Meta-")


def compliance(path="results/scale_grid.json", cell="agent/generated",
               options=10):
    """How often each model refused the arm these rollouts come from.

    The membership test runs on the neutral cell's dump, and five of the
    thirteen models decline that cell on 13 to 65% of their rollouts. The gap
    is then measured on whatever they did answer, which is a different
    population -- so the rate is carried beside every reading rather than left
    for a reader to find in another table.
    """
    grid = pathlib.Path(path)
    if not grid.exists():
        # Loudly, because an empty map does not read as "unknown" downstream:
        # every model looks compliant and above chance, and the informative
        # subgroup silently grows from ten models to thirteen. Reproducing on
        # a machine without the scale grids gave a different headline count
        # with nothing to say it had.
        raise FileNotFoundError(
            f"{path} is needed for the refusal and accuracy columns; without "
            f"it every model reads as compliant and above chance, which "
            f"changes the counts. Run scale_grid.py first, or pass "
            f"--no-compliance to accept that.")
        
    # Keyed on the short name, because the same weights carry two ids across
    # our own files: the BixBench sweep recorded Llama-3.1-8B as
    # ``meta-llama/Meta-Llama-3.1-8B-Instruct`` and the MMLU-Pro one as
    # ``meta-llama/Llama-3.1-8B-Instruct``, which silently left that row with
    # no refusal rate and no accuracy at all.
    return {short(model["model"]): {
                "refused": max(model["cells"][cell]["unparsed_file"],
                               model["cells"][cell]["unparsed_clean"]),
                # A model answering at chance has a correct/wrong split that is
                # nearly a coin flip, so its gap has nothing to be a gap
                # between. The 1B reads 10.7% against 10% and is the one model
                # whose released-minus-control difference clears, which is the
                # reading that needs this column beside it.
                "accuracy": model["cells"][cell]["accuracy"],
                # k, not ten: BixBench's items have four options, so chance
                # there is 25% and not 10%. Hardcoding ten would have marked
                # every BixBench row as comfortably above chance.
                "chance": 100.0 / options}
            for model in json.loads(grid.read_text())["models"]
            if cell in model["cells"]}


# Which cell the paper quotes, and why it differs by file.
#
# On MMLU-Pro the stratified log-likelihood is the right one: the geometry
# channel is out of it and no Min-K% hyperparameter is in it, and both arms
# carry a rank on every rollout because every item's options are numbers.
#
# On BixBench they are not. Its control is resampled from the benchmark's own
# pool of 709 option strings, so most control items mix numbers with text and
# only 66 of 615 control rollouts have a readable rank, against 315 of 615
# released. A stratified difference between a 315-rollout arm and a 66-rollout
# one is mostly the second arm's noise, so BixBench is read pooled and the
# reason is stated rather than the cell quietly swapped.
CELL = {"mmlupro": "loglik:within key rank", "bixall": "loglik:pooled"}


def summary(grid, refused=None, cell="loglik:within key rank"):
    refused = compliance() if refused is None else refused
    rows = []
    for model, report in sorted(
            grid.items(), key=lambda kv: SIZES.get(kv[0], 1e3)):
        released = report["correct_minus_wrong"]
        control = report.get("control_correct_minus_wrong")
        key, extreme = widest(released)
        row = {
            "model": model,
            "billions": SIZES.get(model),
            "widest": key,
            "difference": extreme["difference"],
            "interval": extreme["interval"],
            "clearing": sum(1 for c in released.values() if clears(c)),
            "cells": len(released),
            # the stratified log-likelihood cell, which is the one the paper
            # quotes: the geometry channel is removed and the statistic is the
            # plain one, not a Min-K% variant with a hyperparameter in it
            "stratified": released[cell],
            "refused": (refused.get(short(model)) or {}).get("refused"),
            "accuracy": (refused.get(short(model)) or {}).get("accuracy"),
            "over_chance": (
                None if short(model) not in refused else
                refused[short(model)]["accuracy"]
                - refused[short(model)]["chance"]),
        }
        if control:
            ckey, cextreme = widest(control)
            row["control"] = {
                "widest": ckey, "difference": cextreme["difference"],
                "interval": cextreme["interval"],
                "clearing": sum(1 for c in control.values() if clears(c)),
                "stratified": control[cell]}
        paired = report.get("released_minus_control")
        if paired:
            row["paired"] = {
                "clearing": sum(1 for c in paired.values() if clears(c)),
                "stratified": paired[cell]}
        rows.append(row)
    return rows


def headline(grid, rows, cell="loglik:within key rank"):
    """The three counts and the bound the paper quotes, derived here.

    Recognition of published option sets predicts one thing about this grid:
    the correct-minus-wrong gap should be larger on the released file than on
    a control we generated. So the number that answers it is not how many
    released arms clear zero -- a nuisance correlate of difficulty clears zero
    too -- but how many *differences* do, and how large a difference the data
    could have ruled out.
    """
    counts = {"models": len(rows), "released": 0, "control": 0,
              "difference": 0, "positive": 0, "negative": 0, "refusing": 0,
              "which_difference": [], "informative": 0,
              "informative_difference": 0, "which_informative": []}
    # The exclusion is quoted from the decisive cell only -- the plain
    # log-likelihood, stratified on the key's rank, so the geometry channel is
    # out and no Min-K% hyperparameter is in -- and only from models whose arm
    # the paper's own filter calls readable. Taking the loosest interval over
    # all six cells and all thirteen models instead quotes the noisiest 1B
    # model's Min-K% cell, which excludes almost nothing and says so about
    # nothing.
    ceilings = []
    for row in rows:
        report = grid[row["model"]]
        released = report["correct_minus_wrong"]
        clearing = [c for c in released.values() if clears(c)]
        counts["released"] += bool(clearing)
        counts["positive"] += any(c["difference"] > 0 for c in clearing)
        counts["negative"] += any(c["difference"] < 0 for c in clearing)
        refusing = bool(row["refused"] and row["refused"] > 5)
        counts["refusing"] += refusing
        control = report.get("control_correct_minus_wrong", {})
        counts["control"] += any(clears(c) for c in control.values())
        difference = report.get("released_minus_control", {})
        cleared = [c for c in difference.values() if clears(c)]
        if cleared:
            counts["difference"] += 1
            counts["which_difference"].append(row["model"])
        quoted = difference.get(cell)
        at_chance = row["over_chance"] is not None and row["over_chance"] < 2
        # The test means something only where the model both answers the arm
        # and beats chance on it: a model at chance has a correct/wrong split
        # that is nearly a coin flip, so its gap has nothing to be a gap
        # between. This subgroup is the number the paper quotes.
        if not refusing and not at_chance and quoted and quoted["interval"]:
            counts["informative"] += 1
            counts["informative_difference"] += bool(cleared)
            if cleared:
                counts["which_informative"].append(row["model"])
            ceilings.append(quoted["interval"][1])
    counts["ceiling_lowest"] = min(ceilings) if ceilings else None
    counts["ceiling_highest"] = max(ceilings) if ceilings else None

    # Per-cell, because that is the unit a 95% interval is calibrated on.
    # Thirteen models by six statistics is 78 intervals per arm, so about four
    # of them exclude zero by construction whatever is true. Counting models
    # with at-least-one-clearing-cell -- the counts above -- is the wrong
    # denominator for that comparison and makes noise look like signal.
    for arm, name in (("correct_minus_wrong", "released"),
                      ("control_correct_minus_wrong", "control"),
                      ("released_minus_control", "difference")):
        cells = [c for r in grid.values() for c in r.get(arm, {}).values()
                 if c["interval"]]
        hit = [c for c in cells if clears(c)]
        counts[f"{name}_cells"] = len(cells)
        counts[f"{name}_cells_clearing"] = len(hit)
        counts[f"{name}_cells_positive"] = sum(1 for c in hit
                                               if c["difference"] > 0)
        counts[f"{name}_cells_negative"] = sum(1 for c in hit
                                               if c["difference"] < 0)
    counts["expected_by_chance"] = 0.05 * counts["difference_cells"]
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="results/option_membership_grid.json")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--vocabulary", default=None,
                    help="a rollout dump; report what share of its control's "
                         "option strings are published strings")
    ap.add_argument("--cell", default=None,
                    help="which of the six gaps to quote; the default follows "
                         "CELL from the grid's file")
    ap.add_argument("--no-compliance", action="store_true",
                    help="run without the scale grid, accepting that no model "
                         "can be marked as refusing or at chance")
    ap.add_argument("--arm", default=None,
                    help="the dump suffix these rollouts come from, for the "
                         "refusal and accuracy columns")
    args = ap.parse_args()
    if args.vocabulary:
        counts = control_vocabulary(args.vocabulary)
        print(f"{args.vocabulary}: released pool {counts['released_pool']}, "
              f"control pool {counts['control_pool']}, "
              f"{100 * counts['published_share']:.1f}% of the control's "
              f"strings were published")
        print(json.dumps(counts))
        return 0

    grid = json.loads(pathlib.Path(args.grid).read_text())
    which = "bixall" if "bix" in pathlib.Path(args.grid).stem else "mmlupro"
    cell = args.cell or CELL[which]
    args_for = (("results/scale_grid_bixbench.json", "bixbench/argmax", 4)
                if which == "bixall"
                else ("results/scale_grid.json", "agent/generated", 10))
    if args.no_compliance:
        print("# no scale grid: no model marked refusing or at chance, so "
              "the informative subgroup is every row")
        refused = {}
    else:
        refused = compliance(*args_for)
    rows = summary(grid, refused=refused, cell=cell)

    if args.summary:
        counts = headline(grid, rows, cell)
        print(f"{counts['models']} models, {counts['refusing']} of them on an "
              f"arm the >5% filter calls unreadable")
        print(f"  released arm clears zero somewhere: {counts['released']}"
              f"  (positive on {counts['positive']}, "
              f"negative on {counts['negative']})")
        print(f"  clean control clears zero somewhere: {counts['control']}")
        print(f"  released MINUS control clears zero: {counts['difference']}")
        if counts["which_difference"]:
            print(f"    on {', '.join(counts['which_difference'])}")
        print(f"  of the {counts['informative']} models that answer this arm "
              f"and beat chance on it, the difference clears on "
              f"{counts['informative_difference']}"
              + (f" ({', '.join(counts['which_informative'])})"
                 if counts["which_informative"] else ""))
        for name in ("released", "control", "difference"):
            print(f"  {name:10s} arm: "
                  f"{counts[f'{name}_cells_clearing']:2d} of "
                  f"{counts[f'{name}_cells']} cells clear "
                  f"({100 * counts[f'{name}_cells_clearing'] / counts[f'{name}_cells']:4.1f}%"
                  f", {counts['expected_by_chance']:.1f} expected by chance), "
                  f"{counts[f'{name}_cells_positive']} positive and "
                  f"{counts[f'{name}_cells_negative']} negative")
        print(f"  over those {counts['informative']}, "
              f"the stratified log-likelihood difference's interval tops out "
              f"between {counts['ceiling_lowest']:+.3f} and "
              f"{counts['ceiling_highest']:+.3f} nats")
        print(json.dumps(counts))
        return 0

    if args.latex:
        for row in rows:
            name = row["model"].split("/")[-1].replace("-Instruct", "")
            s, c = row["stratified"], row.get("control", {}).get("stratified")
            def span(cell):
                if cell is None or cell["difference"] is None or cell["interval"] is None:
                    return "---"
                lo, hi = cell["interval"]
                # No bold: the other tables in the paper let the interval say
                # whether a cell clears, and \textbf does not bold math anyway.
                # A thin space, not a word space: these are one quantity and
                # its interval, and the table is wide enough that the
                # difference shows. The emitted row and validate_artifact's
                # pin have to agree character for character, so the spacing
                # lives here rather than being applied to the tex afterwards.
                return f"${cell['difference']:+.3f}$\\,$[{lo:+.3f},{hi:+.3f}]$"
            rate = ("---" if row["refused"] is None else
                    (f"\\emph{{{row['refused']:.0f}}}"
                     if row["refused"] > 5 else f"{row['refused']:.0f}"))
            paired = row.get("paired", {}).get("stratified")
            over = ("---" if row["over_chance"] is None else
                    (f"\\emph{{{row['over_chance']:+.1f}}}"
                     if row["over_chance"] < 2 else f"{row['over_chance']:+.1f}"))
            print(f"{name} & ${row['billions']:.1f}$ & {rate} & {over} & "
                  f"{span(s)} & {span(c)} & {span(paired)}\\\\")
        return 0

    print(f"{'model':30s} {'B':>5s} {'refused':>8s} {'over ch':>7s}  "
          f"{cell:>34s}  "
          f"{'same, clean control':>34s}  "
          f"{'released minus control':>34s}  clearing")
    for row in rows:
        def show(cell):
            if cell is None or cell["difference"] is None or cell["interval"] is None:
                return f"{'---':>34s}"
            lo, hi = cell["interval"]
            mark = "*" if lo > 0 or hi < 0 else " "
            return f"{cell['difference']:+8.4f} [{lo:+.4f},{hi:+.4f}]{mark}"
        control = row.get("control", {})
        rate = ("      --" if row["refused"] is None
                else f"{row['refused']:6.1f}%" + ("!" if row["refused"] > 5 else " "))
        over = ("    --" if row["over_chance"] is None
                else f"{row['over_chance']:+5.1f}" + ("?" if row["over_chance"] < 2 else " "))
        print(f"{row['model'].split('/')[-1][:30]:30s} "
              f"{row['billions'] or 0:5.1f} {rate:>8s} {over:>7s}  "
              f"{show(row['stratified'])}  "
              f"{show(control.get('stratified'))}  "
              f"{show(row.get('paired', {}).get('stratified'))}  "
              f"{row['clearing']}/{row['cells']} vs "
              f"{control.get('clearing', '-')}/{row['cells']} vs "
              f"{row.get('paired', {}).get('clearing', '-')}/{row['cells']}")
    print("\n* = the interval excludes zero.  ! = the arm this is measured on "
          "is one the\n    paper's own filter calls unreadable, because the "
          "model declines more than 5%\n    of its rollouts.  ? = the model "
          "reads within two points of chance on this arm, so\n    its correct "
          "and wrong rollouts are nearly the same draw.  Widest cell per "
          "model:")
    for row in rows:
        if row["interval"] is None:
            continue
        lo, hi = row["interval"]
        print(f"  {row['model'].split('/')[-1][:30]:30s} {row['widest']:22s} "
              f"{row['difference']:+.4f} [{lo:+.4f},{hi:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
