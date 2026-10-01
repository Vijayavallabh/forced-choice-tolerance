#!/usr/bin/env python3
r"""$\hat\lambda$ under the relaxed model, on the draws that reject the shared one.

``rank_dependent_fit.py`` answers half of the question: it reports what freeing
one $\lambda_j$ per key rank does to the \emph{geometry} term, and not what it
does to $\hat\lambda$ --- the
quantity \S\ref{sec:conclusion} tells the field to publish. The fits are
already on disk with $\lambda$ by rank recorded per draw, so this reads them
rather than refitting.

Reported separately for the draws where the shared-$\lambda$ model is rejected
against the saturated multinomial and for the draws where it is not, because
the question is whether the rejection is what carries the estimate.

    python3 lambda_relaxed.py
"""
import argparse
import json
import pathlib
import statistics


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", default="results/rank_dependent_fit.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output", default="results/lambda_relaxed.json")
    args = ap.parse_args()

    fit = json.load(open(args.fit, encoding="utf-8"))
    by_model = {}
    for draw in fit["draws"]:
        entry = by_model.setdefault(draw["model"], {"rejected": [], "kept": []})
        side = "rejected" if draw["shared"]["gof_p"] < args.alpha else "kept"
        entry[side].append({
            "shared": draw["shared"]["lambda_mean"],
            "relaxed": draw["per_rank"]["lambda_mean"],
            "relaxed_by_rank": draw["per_rank"]["lambda_by_rank"],
            "spread": (max(draw["per_rank"]["lambda_by_rank"])
                       - min(draw["per_rank"]["lambda_by_rank"])),
            "gof_p_shared": draw["shared"]["gof_p"],
            "gof_p_relaxed": draw["per_rank"]["gof_p"]})

    report = {"alpha": args.alpha, "models": {}}
    print(f"{'model':32s} {'draws':>5s} {'rej':>4s} "
          f"{'shared lam':>11s} {'relaxed lam':>12s} {'max shift':>10s} "
          f"{'lam spread':>11s}")
    for model, sides in sorted(by_model.items()):
        every = sides["rejected"] + sides["kept"]
        block = {"n_draws": len(every), "n_rejected": len(sides["rejected"])}
        for name, rows in (("rejected", sides["rejected"]),
                           ("kept", sides["kept"]), ("all", every)):
            if not rows:
                continue
            block[name] = {
                "n": len(rows),
                "shared_median": statistics.median(r["shared"] for r in rows),
                "relaxed_median": statistics.median(r["relaxed"] for r in rows),
                "max_abs_shift": max(abs(r["relaxed"] - r["shared"]) for r in rows),
                "max_spread_across_ranks": max(r["spread"] for r in rows)}
        report["models"][model] = block
        a = block["all"]
        print(f"{model.split('/')[-1]:32s} {block['n_draws']:5d} "
              f"{block['n_rejected']:4d} {a['shared_median']:11.3f} "
              f"{a['relaxed_median']:12.3f} {a['max_abs_shift']:10.3f} "
              f"{a['max_spread_across_ranks']:11.3f}")

    order_shared = sorted(report["models"],
                          key=lambda m: report["models"][m]["all"]["shared_median"])
    order_relaxed = sorted(report["models"],
                           key=lambda m: report["models"][m]["all"]["relaxed_median"])
    report["ordering_shared"] = [m.split("/")[-1] for m in order_shared]
    report["ordering_relaxed"] = [m.split("/")[-1] for m in order_relaxed]
    report["ordering_survives"] = order_shared == order_relaxed
    biggest = max(
        (b["all"]["max_abs_shift"] for b in report["models"].values()), default=0.0)
    report["max_abs_shift_any_draw"] = biggest
    print(f"\nordering by shared  lambda: {' < '.join(report['ordering_shared'])}")
    print(f"ordering by relaxed lambda: {' < '.join(report['ordering_relaxed'])}")
    print(f"ordering survives the relaxation: {report['ordering_survives']}")
    print(f"largest shift in lambda on any draw: {biggest:.4f}")

    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
