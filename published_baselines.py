#!/usr/bin/env python3
r"""What BixBench's own published zero-shot baselines say once \eqref{eq:split} is fitted.

The benchmark publishes two no-data multiple-choice runs and reads their
above-chance scores as a floor on what a model already knows. Fitting the
decomposition to them says how much of that survives as recall, and the
boundary-corrected test of $\lambda=0$ says whether any of it is
distinguishable from none.

Only v1.0 can be used: v1.5 records the keyed letter and not the presented
order, so no rank table can be built from it.

    python3 published_baselines.py
"""
import argparse
import glob
import json
from pathlib import Path

import choice_model as cm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="data/external/zero_shot_v10/*mcq*.csv")
    ap.add_argument("--output", default="results/published_baselines.json")
    args = ap.parse_args()

    runs = []
    for path in sorted(glob.glob(args.glob)):
        pairs, skipped = cm.pairs_from_v10(Path(path))
        fitted = cm.fit(pairs)
        lrt = cm.lrt_no_knowledge(pairs, fitted)
        gof = cm.goodness_of_fit(pairs, fitted["lam"], fitted["b"])
        accuracy = sum(p["correct"] for p in pairs) / len(pairs)
        entry = {
            "run": Path(path).stem, "n_items": len(pairs),
            "n_options": 4, "chance": 0.25,
            "accuracy": accuracy, "margin": accuracy - 0.25,
            "lambda": fitted["lam"],
            "lrt_lambda_zero_p": lrt["p_value"],
            "goodness_of_fit_p": gof["p_value"],
            "skipped": skipped,
        }
        runs.append(entry)
        print(f"{entry['run']}\n   n={entry['n_items']} acc={100*accuracy:.1f}% "
              f"lambda={fitted['lam']:.3f}  LRT p={lrt['p_value']:.3f}  "
              f"GoF p={gof['p_value']:.3f}")

    report = {"runs": runs,
              "statement": ("neither published zero-shot baseline's recall term is "
                            "distinguishable from zero at alpha=0.05")
              if all(r["lrt_lambda_zero_p"] >= 0.05 for r in runs) else
              "at least one published baseline rejects lambda=0"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}: {report['statement']}")


if __name__ == "__main__":
    main()
