#!/usr/bin/env python3
r"""The free-response contrast across models, with the compliance rule applied.

``free_response.py`` runs two arms per model. This reads them back and reports
the grid, with one rule that decides what is printable: the multiple-choice arm
is UNREADABLE where more than $5\%$ of its rollouts do not parse, because a
margin on a model that will not answer is a refusal rate wearing a margin's
name -- the same rule ``scale_grid.py`` applies. The free-response arm has no
such problem: a reply with no number in it is simply wrong, and the declination
rate is printed beside it.

Read the two columns differently. The free-response accuracy is what a model
can state unaided and is comparable across every model. The difference between
the arms is what the option set adds, and is only interpretable where the
multiple-choice arm is readable.

    python3 free_grid.py --latex
"""
import argparse
import json
import pathlib

FILES = {
    "bixnum": ("results/free_response.json", "bixbench_numeric_q",
               "BixBench's 105 four-option numeric items"),
    "mmlupro": ("results/free_response_mmlupro.json", "mmlu_pro_matched_released",
                "MMLU-Pro's 901 ten-option items"),
}


def params_billions(model):
    """Reuse scale_grid's reader so the ladder is the same one tab:grid uses."""
    import scale_grid
    try:
        return scale_grid.params_billions(model)
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="bixnum", choices=sorted(FILES))
    ap.add_argument("--max-unparsed", type=float, default=5.0)
    ap.add_argument("--tolerance", default="0.05")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    path, stem, label = FILES[args.file]
    if not pathlib.Path(path).exists():
        raise SystemExit(f"no {path} yet")
    raw = json.loads(pathlib.Path(path).read_text())
    rows = []
    for key, entry in raw.items():
        model, items = key.split("|", 1)
        if items != stem:
            continue
        mcq = entry["arms"]["mcq"]
        free = entry["arms"]["free"]
        cell = free["by_tolerance"][args.tolerance]
        rows.append({
            "model": model,
            "params": params_billions(model),
            "mcq_accuracy": mcq["accuracy"],
            "mcq_margin": mcq["margin"],
            "mcq_interval": mcq["interval"],
            "mcq_unparsed": mcq["unparsed"],
            "mcq_readable": mcq["unparsed"] <= args.max_unparsed,
            "free_accuracy": cell["accuracy"],
            "free_interval": cell["interval"],
            "free_declined": free["declined"],
            "option_set_worth": mcq["margin"] - cell["accuracy"],
        })
    rows.sort(key=lambda r: (r["params"] if r["params"] == r["params"] else 1e9))

    print(f"{label}, free response graded at {float(args.tolerance):.0%} relative "
          f"tolerance\n")
    head = (f"{'model':30s} {'B':>5s} {'mcq':>22s} {'unp':>5s} "
            f"{'free response':>20s} {'decl':>5s} {'option set':>10s}")
    print(head)
    print("-" * len(head))
    for r in rows:
        mcq = (f"{r['mcq_margin']:+6.2f} [{r['mcq_interval'][0]:+.1f},"
               f"{r['mcq_interval'][1]:+.1f}]" if r["mcq_readable"]
               else f"unreadable {r['mcq_unparsed']:.0f}% unparsed")
        print(f"{r['model'].split('/')[-1]:30s} {r['params']:5.1f} {mcq:>22s} "
              f"{r['mcq_unparsed']:5.1f} "
              f"{r['free_accuracy']:6.2f}% [{r['free_interval'][0]:.1f},"
              f"{r['free_interval'][1]:.1f}]".ljust(0)
              + f" {r['free_declined']:5.1f} "
              + (f"{r['option_set_worth']:+10.2f}" if r["mcq_readable"] else
                 f"{'--':>10s}"))

    free = [r["free_accuracy"] for r in rows]
    readable = [r for r in rows if r["mcq_readable"]]
    print(f"\nfree-response accuracy spans {min(free):.2f}% to {max(free):.2f}% "
          f"over {len(rows)} models")
    print(f"the multiple-choice arm is readable on {len(readable)} of {len(rows)}")
    if readable:
        worth = [r["option_set_worth"] for r in readable]
        print(f"where it is, the option set is worth {min(worth):+.2f} to "
              f"{max(worth):+.2f} points")

    report = {"file": args.file, "label": label, "tolerance": float(args.tolerance),
              "max_unparsed": args.max_unparsed, "rows": rows,
              "free_accuracy_range": [min(free), max(free)],
              "n_models": len(rows), "n_readable": len(readable)}
    out = args.output or f"results/free_grid_{args.file}.json"
    pathlib.Path(out).write_text(json.dumps(report, indent=2) + "\n",
                                 encoding="utf-8")
    print(f"wrote {out}")

    if args.latex:
        print()
        for r in rows:
            mcq = (f"${r['mcq_margin']:+.1f}$ $[{r['mcq_interval'][0]:+.1f},"
                   f"{r['mcq_interval'][1]:+.1f}]$" if r["mcq_readable"]
                   else f"unparsed ${r['mcq_unparsed']:.0f}\\%$")
            print(f"{r['model'].split('/')[-1].replace('-Instruct','')} & "
                  f"${r['params']:.1f}$ & {mcq} & "
                  f"${r['free_accuracy']:.1f}\\%$ & "
                  f"${r['free_declined']:.0f}\\%$\\\\")


if __name__ == "__main__":
    main()
