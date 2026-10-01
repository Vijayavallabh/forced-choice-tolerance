#!/usr/bin/env python3
r"""Every agent arm's margin, at the nominal level and at the level it covers at.

``bootstrap_calibration.py`` measures what a percentile cluster bootstrap
actually covers on files shaped like these. On BixBench's $46$ capsules it
covers at its nominal level. On MMLU-Pro it does not, and the reason is
model-free: the file's $60$ clusters are source subjects of wildly unequal size
-- one holds $23\%$ of the items -- so Kish's effective count is about $11$, and
a resample of $60$ labels is not $60$ independent draws.

This recomputes each arm from its dumped rollouts, at the nominal $95\%$ and at
whatever level the calibration says covers $95\%$ on that shape, so both are on
the page and the wider one is the one the text quotes.

The $95\%$ column reproduces the published interval bit-for-bit: the bootstrap
here mirrors ``agentic_probe.summarise``'s draw sequence exactly, same seeds and
same order, so a mismatch is a bug and not a second sample.

    python3 arm_intervals.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import pathlib

import numpy as np

RESULTS = pathlib.Path("results")

# dump, label, the result file whose 95% interval it must reproduce, and the key
DUMPS = [
    ("build/dumps/qwen14b_withheld_aware.jsonl", "BixBench, Qwen2.5-14B",
     "results/agentic_bixbench_qwen14b.json", None),
    ("build/dumps/qwen32b_withheld_aware.jsonl", "BixBench, Qwen2.5-32B",
     "results/agentic_bixbench_qwen32b.json", None),
    ("build/dumps/qwen14b_mmlupro_steered_tools.jsonl", "MMLU-Pro, Qwen2.5-14B",
     "results/agentic_mmlupro_qwen14b.json", None),
    # The same arm re-run at batch 12 rather than 10. Greedy decoding is
    # deterministic given the batching and not otherwise, so this is a different
    # number (+5.60 against +6.25) from the same weights, prompts and items --
    # which is the point of keeping it, and the reason it has its own name.
    ("build/dumps/qwen14b_mmlupro_batch12.jsonl",
     "MMLU-Pro, Qwen2.5-14B at batch 12",
     "results/agentic_replication_mmlupro_qwen14b.json", None),
    ("build/dumps/qwen32b_mmlupro.jsonl", "MMLU-Pro, Qwen2.5-32B",
     "results/agentic_mmlupro_qwen32b.json", None),
    # the instrument cells: same model, same file, same items and orderings
    ("build/dumps/qwen14b_mmlupro_steered_notools.jsonl",
     "MMLU-Pro, 14B steered, no tool", None, None),
    ("build/dumps/qwen14b_mmlupro_neutral_tools.jsonl",
     "MMLU-Pro, 14B neutral, interpreter", None, None),
    ("build/dumps/qwen14b_mmlupro_neutral_notools.jsonl",
     "MMLU-Pro, 14B neutral, no tool", None, None),
    ("build/dumps/qwen14b_mmlupro_argmax_withheld.jsonl",
     "MMLU-Pro, 14B neutral, one letter", None, None),
    ("build/dumps/qwen14b_mmlupro_argmax_withheld_aware.jsonl",
     "MMLU-Pro, 14B steered, one letter", None, None),
    ("build/dumps/qwen14b_mmlupro_bixprompt_argmax.jsonl",
     "MMLU-Pro, 14B BixBench prompt, one letter", None, None),
    ("build/dumps/qwen14b_mmlupro_bixprompt_generated.jsonl",
     "MMLU-Pro, 14B BixBench prompt, generated", None, None),
    ("build/dumps/qwen32b_mmlupro_steered_notools.jsonl",
     "MMLU-Pro, 32B steered, no tool", None, None),
    ("build/dumps/qwen32b_mmlupro_neutral_tools.jsonl",
     "MMLU-Pro, 32B neutral, interpreter", None, None),
    ("build/dumps/qwen32b_mmlupro_neutral_notools.jsonl",
     "MMLU-Pro, 32B neutral, no tool", None, None),
    # the same two corners in a second model family: phi-4 is a 14B instruction
    # model from another pretraining corpus, so "the two largest of four" stops
    # also meaning "the two Qwen2.5 of four".
    ("build/dumps/phi4_mmlupro_bixprompt_argmax.jsonl",
     "MMLU-Pro, phi-4 BixBench prompt, one letter", None, None),
    ("build/dumps/phi4_mmlupro_neutral_notools.jsonl",
     "MMLU-Pro, phi-4 neutral, no tool", None, None),
    # and the 70B-class row, where a nominal interval that just clears is
    # exactly the case the calibration was built to catch
    ("build/dumps/qwen72b_mmlupro_bixprompt_argmax.jsonl",
     "MMLU-Pro, Qwen2.5-72B BixBench prompt, one letter", None, None),
    ("build/dumps/qwen72b_mmlupro_neutral_notools.jsonl",
     "MMLU-Pro, Qwen2.5-72B neutral, no tool", None, None),
    ("build/dumps/llama70b_mmlupro_bixprompt_argmax.jsonl",
     "MMLU-Pro, Llama-3.3-70B BixBench prompt, one letter", None, None),
    ("build/dumps/llama70b_mmlupro_neutral_notools.jsonl",
     "MMLU-Pro, Llama-3.3-70B neutral, no tool", None, None),
    # the rank-holding placebo under the cell that reads highest
    ("build/dumps/qwen14b_mmlupro_placebo_neutral_notools.jsonl",
     "MMLU-Pro placebo, 14B neutral, no tool", None, None),
    ("build/dumps/qwen14b_bixall.jsonl", "BixBench 205, Qwen2.5-14B", None, None),
    # The same instrument decomposition on the venue's own benchmark: all 205
    # released items in 59 capsules, against the control drawn from the file's
    # own pool of option strings.
    ("build/dumps/qwen14b_bixall_bixprompt_argmax.jsonl",
     "BixBench 205, 14B BixBench prompt, one letter", None, None),
    ("build/dumps/qwen14b_bixall_bixprompt_generated.jsonl",
     "BixBench 205, 14B BixBench prompt, generated", None, None),
    ("build/dumps/qwen14b_bixall_neutral_argmax.jsonl",
     "BixBench 205, 14B neutral, one letter", None, None),
    ("build/dumps/qwen14b_bixall_neutral_notools.jsonl",
     "BixBench 205, 14B neutral, no tool", None, None),
]


def resolve(path):
    """The dump where it was written, or the gzipped copy the artifact ships.

    Arms are run into ``build/dumps/`` on the machine with the GPUs; the
    evidence bundle carries them as ``results/agentic_dumps/*.jsonl.gz`` so
    every interval here re-derives from an unzipped artifact with no GPU.
    """
    here = pathlib.Path(path)
    if here.exists():
        return here
    shipped = RESULTS / "agentic_dumps" / (here.name + ".gz")
    return shipped if shipped.exists() else here


def load(path, arm):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        rows = [json.loads(l) for l in fh]
    return [r for r in rows if r.get("arm", "file") == arm]


def draws(rollouts, bootstrap, seed):
    """``agentic_probe.summarise``'s bootstrap, reproduced draw for draw."""
    correct = np.array([1.0 if r["answer"] == r["gold"] else 0.0 for r in rollouts])
    chance = np.array([1.0 / r["k"] for r in rollouts])
    clusters = sorted({r["cluster"] for r in rollouts})
    by_cluster = {c: [i for i, r in enumerate(rollouts) if r["cluster"] == c]
                  for c in clusters}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(bootstrap):
        take = [i for c in rng.choice(clusters, size=len(clusters), replace=True)
                for i in by_cluster[c]]
        out.append(float(np.mean(correct[take] - chance[take])))
    sizes = np.array([len(by_cluster[c]) for c in clusters], dtype=float)
    return (np.array(out), 100.0 * float(np.mean(correct - chance)),
            len(clusters), float(sizes.sum() ** 2 / (sizes ** 2).sum()))


def interval(values, level):
    tail = 100.0 * (1.0 - level) / 2.0
    return [float(np.percentile(values, tail)),
            float(np.percentile(values, 100.0 - tail))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration", default="results/bootstrap_calibration.json")
    ap.add_argument("--output", default="results/arm_intervals.json")
    args = ap.parse_args()

    # The level each shape needs, keyed by the benchmark the calibration ran on.
    # There is no default. This used to fall back to 95% when the calibration
    # was missing or empty, and then printed a nominal interval in the column
    # that says it covers. So a missing calibration, a shape it does not cover,
    # a level it could not find and a missing dump each stop the run here,
    # before anything is written.
    cal = pathlib.Path(args.calibration)
    if not cal.exists():
        raise SystemExit(f"{cal} is missing, so no arm has a covering level. Run "
                         f"bootstrap_calibration.py first; {args.output} was not written.")
    needed = {}
    for entry in json.loads(cal.read_text())["arms"]:
        observed = next(c for c in entry["cells"] if c["as_observed"])
        needed[entry["arm"].split(",")[0].split()[0]] = {
            "level": observed["level_for_95_difference"],
            "coverage_at_95": observed["coverage_difference"],
            "effective_clusters": entry["effective_clusters_file"]}
    problems = []
    for path, label, _, _ in DUMPS:
        shape = needed.get(label.split(",")[0].split()[0])
        if shape is None:
            problems.append(f"{label}: {cal} has no {label.split(',')[0].split()[0]} arm")
        elif not (isinstance(shape["level"], float) and math.isfinite(shape["level"])):
            # level_for() returns NaN when no level on its grid reaches 95%
            problems.append(f"{label}: the calibration found no covering level "
                            f"({shape['level']!r})")
        if not resolve(path).exists():
            problems.append(f"{label}: no dump at {path} or "
                            f"{RESULTS / 'agentic_dumps' / (pathlib.Path(path).name + '.gz')}")
    if problems:
        raise SystemExit("refusing to write " + args.output + ":\n  " + "\n  ".join(problems))

    report = {"bootstrap": args.bootstrap, "arms": []}
    for name, label, source, _ in DUMPS:
        # Read the shipped copy when the GPU run's own file is absent. The report
        # names the dump as the arms do, so a rerun writes the same bytes from
        # either copy: the shipped file is the same rollouts, gzipped.
        path = resolve(name)
        file_draws, file_margin, n_clusters, n_eff = draws(
            load(path, "file"), args.bootstrap, args.seed)
        clean_draws, clean_margin, _, _ = draws(
            load(path, "clean"), args.bootstrap, args.seed + 1)
        difference = 100.0 * (file_draws - clean_draws)
        shape = needed[label.split(",")[0].split()[0]]
        level = shape["level"]
        entry = {
            "arm": label, "dump": name, "n_clusters": n_clusters,
            "effective_clusters": n_eff,
            "margin_over_clean": file_margin - clean_margin,
            "ci95": interval(difference, 0.95),
            "calibrated_level": level,
            "ci_calibrated": interval(difference, level),
            "coverage_of_the_95_interval": shape.get("coverage_at_95")}
        report["arms"].append(entry)
        published = ""
        if source and pathlib.Path(source).exists():
            doc = json.loads(pathlib.Path(source).read_text())
            arm = next(iter(doc.values()))
            got = arm.get("margin_over_clean_ci95")
            if got:
                agree = (abs(got[0] - entry["ci95"][0]) < 0.02
                         and abs(got[1] - entry["ci95"][1]) < 0.02)
                published = "  (reproduces published)" if agree else (
                    f"  (PUBLISHED SAYS {got[0]:+.2f},{got[1]:+.2f})")
        print(f"{label:26s} {n_clusters:3d} clusters, {n_eff:5.1f} effective\n"
              f"   margin {entry['margin_over_clean']:+5.2f}  "
              f"95% [{entry['ci95'][0]:+5.2f},{entry['ci95'][1]:+5.2f}]{published}\n"
              f"   {100*level:5.1f}% [{entry['ci_calibrated'][0]:+5.2f},"
              f"{entry['ci_calibrated'][1]:+5.2f}]  <- covers 95%")

    RESULTS.mkdir(exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
