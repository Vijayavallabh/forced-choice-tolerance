#!/usr/bin/env python3
"""How often does the audit cry leak on a file that has none?

This is the measurement that took the headline out of ``mcq_audit``. The
obvious summary of a six-coordinate audit is an OR: the numeric rule family, the
surface rule family, and one bound per ordering each get a say. Their
false-positive rates then add up, so a maintainer reading only that line sees
"LEAKS" on clean files far more often than the nominal 5%, and a *repair*
evaluated by it looks unreliable when it is not. The tool now returns the list
of coordinates that fired and no summary; the disjunction is still formed here,
because its rate is the thing being measured.

This measures the rate directly. Each synthetic file is clean by construction:
the ``k`` options of an item are drawn i.i.d. and rendered in styles drawn
i.i.d., so they are exchangeable, and the key is then chosen uniformly among
them. Exchangeability makes the key's rank uniform under *every* ordering at
once -- value, isolation, length, roundness -- which is exactly the null the
audit is testing, and no repair can do better than this file.

    python3 audit_calibration.py --replicates 40 --items 105 --clusters 46

Reports, per component and for their disjunction, the share of replicates that
fired. Use it to read the audit: one coordinate clearing zero is roughly a coin
flip away from noise, while a bound several points clear of zero, or the same
coordinate firing across seeds, is not.
"""
import argparse
import json
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mcq_audit import audit, build_items

STYLES = ["1234", "1234.0", "1234.00", "1234.000", "1,234", "1234.0%", "1234.00%"]


def render(value, style):
    from mcq_audit import format_like
    return format_like(style, value)


def synthetic_rows(n_items, k, n_clusters, rng):
    """A file whose options are exchangeable and whose key is one of them at random.

    Not a model of any benchmark -- the point is the opposite. The values carry
    no relationship to a "true" answer, so nothing about the key is recoverable
    from the option set, and every ordering's rank distribution is uniform by
    construction.
    """
    rows = []
    for index in range(n_items):
        scale = 10 ** rng.uniform(-1, 4)
        for _ in range(40):
            values = [scale * rng.lognormvariate(0, 0.6) for _ in range(k)]
            styles = [rng.choice(STYLES) for _ in range(k)]
            options = [render(value, style) for value, style in zip(values, styles)]
            if len(set(options)) == k:
                break
        else:
            continue
        key = rng.randrange(k)
        rows.append({"ideal": options[key],
                     "distractors": [o for i, o in enumerate(options) if i != key],
                     "question": _stem(rng),
                     "capsule_uuid": f"c{index % n_clusters}"})
    return rows


WORDS = ("cells growth assay buffer control median fraction sample rate yield "
         "protein reads counts window signal bound ratio decay uptake slope").split()


def _stem(rng):
    """A question drawn independently of the options.

    The surface family includes rules that score an option by how much it
    overlaps the stem, and an empty stem would switch them off and understate
    the audit's multiplicity. Drawn independently so the option set stays
    exchangeable -- a stem that echoed one option would be a leak, and this
    file must have none.
    """
    body = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 20)))
    numbers = " ".join(f"{rng.uniform(0, 1000):.{rng.randint(0, 3)}f}"
                       for _ in range(rng.randint(0, 3)))
    return f"{body} {numbers}".strip() + "?"


def one_replicate(job):
    seed, n_items, k, n_clusters, reps = job
    rng = random.Random(seed)
    rows = synthetic_rows(n_items, k, n_clusters, rng)
    items, found_k = build_items(rows, "ideal", "distractors", "capsule_uuid",
                                 "question", k)
    report = audit(items, reps, found_k)
    fired = {"numeric rule family": bool(report["numeric_family"]
                                         and report["numeric_family"]["leaks"]),
             "surface rule family": bool(report["surface_family"]
                                         and report["surface_family"]["leaks"])}
    # Every block the disjunction covers, so the components add up to it.
    bounds = {}
    for name, blocks in report["surface_channels"]["orderings"].items():
        for subset, block in blocks.items():
            label = name if len(blocks) == 1 else f"{name} ({subset})"
            fired[label] = bool(block["leaks"])
            bounds[label] = block["credit_bound"]["credit_lower_bound"]
    fired["any coordinate"] = bool(report["coordinates_firing"])
    return {"seed": seed, "n_items": len(items), "fired": fired, "bounds": bounds}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicates", type=int, default=40)
    parser.add_argument("--items", type=int, default=105)
    parser.add_argument("--clusters", type=int, default=46)
    parser.add_argument("--n-options", type=int, default=4)
    parser.add_argument("--reps", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=8,
                        help="the machine is shared; keep this modest")
    parser.add_argument("--first-seed", type=int, default=20260919)
    parser.add_argument("--output", type=Path,
                        default=Path("results/audit_calibration.json"))
    args = parser.parse_args()

    jobs = [(args.first_seed + i, args.items, args.n_options, args.clusters, args.reps)
            for i in range(args.replicates)]
    print(f"{args.replicates} clean files of {args.items} items, "
          f"{args.n_options} options, {args.clusters} clusters")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(one_replicate, jobs))

    # A replicate whose options all parse reports one block per ordering; one
    # with an unparseable option reports two, so the labels are not the same
    # across replicates and a missing label means "did not fire".
    names = list(dict.fromkeys(name for run in runs for name in run["fired"]))
    rates = {name: statistics.fmean(float(run["fired"].get(name, False)) for run in runs)
             for name in names}
    for name in names:
        print(f"  {name:28s} fired on {rates[name]:5.1%} of clean files")
    bound_names = list(dict.fromkeys(name for run in runs for name in run["bounds"]))
    worst = {name: max(run["bounds"][name] for run in runs if name in run["bounds"])
             for name in bound_names}
    for name, value in worst.items():
        print(f"  widest bound ever reported on a clean file, {name}: {value:+.1%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "design": ("options drawn i.i.d. and rendered in i.i.d. styles, key chosen "
                   "uniformly among them: every ordering's rank distribution is "
                   "uniform by construction"),
        "n_items": args.items, "n_clusters": args.clusters,
        "n_options": args.n_options, "replicates": args.replicates,
        "reps": args.reps, "false_positive_rate": rates,
        "widest_bound_on_a_clean_file": worst, "runs": runs}, indent=2) + "\n",
        encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
