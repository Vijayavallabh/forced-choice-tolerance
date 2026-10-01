#!/usr/bin/env python3
r"""The test the no-data baseline stands in for, run on the items it stands in for.

The fallback exists because the principled test is unavailable: date each task's
source, split on the model's training cutoff, and compare. ``temporal_recovery.py``
shows the dates are recoverable for LAB-Bench's literature subtasks from the
DOIs the release already carries, so on those items the principled test is
available and the proxy can be checked against it.

Two arms, on the same items, with the same letter orders:

    question     the benchmark's own condition, closed book
    withheld     the question replaced by ``[withheld]``, which is the no-data
                 baseline: whatever is scored here comes from the options

and three readings of the result:

    trend        the correlation between an item's source date and whether the
                 solver gets it right, which needs no claim about any cutoff
    split        accuracy before and after a *stated* cutoff, reported only for
                 models whose vendor states one
    agreement    whether the items the no-data arm gets right are the old ones,
                 which is what the proxy claims about itself

    python3 temporal_test.py --jsonl build/labbench_litqa2.jsonl \
        --dates results/item_dates.json --benchmark "LAB-Bench LitQA2" \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct --device cuda:0
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from no_data_probe import (MCQ_PROMPT_TEMPLATE, letter_token_ids, letters_for,
                           option_block, render, score)

WITHHELD = "[withheld]"
# Cutoffs a vendor states in the model card. A model absent here is reported on
# the trend only, because guessing a cutoff would make the split unfalsifiable.
STATED_CUTOFF = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "2023-12-31",
    "meta-llama/Llama-3.1-8B-Instruct": "2023-12-31",
    "meta-llama/Llama-3.2-3B-Instruct": "2023-12-31",
    "meta-llama/Llama-3.2-1B-Instruct": "2023-12-31",
}


def spearman(xs, ys):
    """Rank correlation, with average ranks for ties."""
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for index in order[i:j + 1]:
                out[index] = average
            i = j + 1
        return out

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def cluster_bootstrap(per_cluster, statistic, reps, seed):
    """Resample clusters, recompute, return the sorted draws."""
    clusters = sorted(per_cluster)
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        picked = [per_cluster[rng.choice(clusters)] for _ in clusters]
        flat = [row for block in picked for row in block]
        value = statistic(flat)
        if value is not None:
            draws.append(value)
    draws.sort()
    return draws


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--dates", default="results/item_dates.json")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--max-options", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--output", default="results/temporal_test.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dated = json.loads(Path(args.dates).read_text(encoding="utf-8"))[args.benchmark]
    rows = [json.loads(line) for line in
            Path(args.jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    items = []
    for row in rows:
        record = dated.get(str(row.get("id", "")))
        if record is None:
            continue
        options = [str(row["ideal"]).strip()] + [str(d).strip() for d in row["distractors"]]
        if not 2 <= len(options) <= args.max_options:
            continue
        items.append({"options": options, "date": record["date"],
                      "cluster": record["cluster"], "id": row.get("id", ""),
                      "question": str(row.get("question", "")).strip()})
    print(f"{len(items)} dated items of {len(rows)} in {args.jsonl}")
    if not items:
        raise SystemExit("no dated items")

    rng = random.Random(args.seed)
    conditions = []
    for index, item in enumerate(items):
        k = len(item["options"])
        for draw in range(args.draws):
            permutation = rng.sample(range(k), k)
            for arm, question in (("question", None), ("withheld", WITHHELD)):
                conditions.append({
                    "arm": arm, "item": index, "k": k, "draw": draw,
                    "cluster": item["cluster"], "date": item["date"],
                    "prompt": MCQ_PROMPT_TEMPLATE.format(
                        question=question or item["question"],
                        options=option_block(item["options"], permutation,
                                             letters_for(k))),
                    "target": permutation.index(0)})
    print(f"{len(conditions)} conditions per model")

    report = {"benchmark": args.benchmark, "jsonl": args.jsonl,
              "n_items": len(items), "draws": args.draws,
              "date_range": [min(i["date"] for i in items),
                             max(i["date"] for i in items)],
              "models": {}}

    for name in args.model:
        print(f"\n=== {name}", flush=True)
        tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map=args.device)
        model.eval()

        outcomes = []
        for k in sorted({c["k"] for c in conditions}):
            group = [c for c in conditions if c["k"] == k]
            ids, _ = letter_token_ids(tok, k)
            texts = [render(tok, c["prompt"]) for c in group]
            picks = score(model, tok, texts, ids, args.batch_size, model.device)
            for condition, (pick, _) in zip(group, picks):
                outcomes.append({**{key: condition[key] for key in
                                    ("arm", "item", "k", "cluster", "date")},
                                 "correct": int(pick == condition["target"]),
                                 "chance": 1.0 / k})
        del model
        torch.cuda.empty_cache()

        block = {}
        for arm in ("question", "withheld"):
            arm_rows = [o for o in outcomes if o["arm"] == arm]
            per_item = defaultdict(list)
            for row in arm_rows:
                per_item[row["item"]].append(row)
            # One number per item: accuracy over its draws, and its margin over
            # that item's own chance, because k varies across items here.
            item_rows = []
            for index, block_rows in per_item.items():
                item_rows.append({
                    "item": index, "date": block_rows[0]["date"],
                    "cluster": block_rows[0]["cluster"],
                    "accuracy": statistics.fmean(r["correct"] for r in block_rows),
                    "margin": statistics.fmean(r["correct"] - r["chance"]
                                               for r in block_rows)})
            by_cluster = defaultdict(list)
            for row in item_rows:
                by_cluster[row["cluster"]].append(row)

            rho = spearman([r["date"] for r in item_rows],
                           [r["margin"] for r in item_rows])
            rho_draws = cluster_bootstrap(
                by_cluster,
                lambda rows: spearman([r["date"] for r in rows],
                                      [r["margin"] for r in rows]) if len(rows) > 2 else None,
                args.reps, args.seed)
            entry = {
                "n_items": len(item_rows),
                "accuracy": statistics.fmean(r["accuracy"] for r in item_rows),
                "margin_over_chance": 100 * statistics.fmean(r["margin"] for r in item_rows),
                "spearman_date_vs_margin": rho,
                "spearman_ci95": [rho_draws[int(0.025 * len(rho_draws))],
                                  rho_draws[int(0.975 * len(rho_draws))]],
                "spearman_p_two_sided": 2 * min(
                    sum(d <= 0 for d in rho_draws), sum(d >= 0 for d in rho_draws)
                ) / max(1, len(rho_draws)),
            }
            cutoff = STATED_CUTOFF.get(name)
            if cutoff:
                before = [r for r in item_rows if r["date"] <= cutoff]
                after = [r for r in item_rows if r["date"] > cutoff]
                if before and after:
                    def gap(rows):
                        old = [r["margin"] for r in rows if r["date"] <= cutoff]
                        new = [r["margin"] for r in rows if r["date"] > cutoff]
                        if not old or not new:
                            return None
                        return 100 * (statistics.fmean(old) - statistics.fmean(new))
                    draws = cluster_bootstrap(by_cluster, gap, args.reps, args.seed)
                    entry["stated_cutoff"] = cutoff
                    entry["n_before"] = len(before)
                    entry["n_after"] = len(after)
                    entry["margin_before"] = 100 * statistics.fmean(r["margin"] for r in before)
                    entry["margin_after"] = 100 * statistics.fmean(r["margin"] for r in after)
                    entry["before_minus_after"] = (entry["margin_before"]
                                                   - entry["margin_after"])
                    entry["before_minus_after_ci95"] = [
                        draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]
                    entry["before_minus_after_p_two_sided"] = 2 * min(
                        sum(d <= 0 for d in draws), sum(d >= 0 for d in draws)
                    ) / max(1, len(draws))
            block[arm] = entry
            print(f"  {arm:<9} n={entry['n_items']} acc={100*entry['accuracy']:.1f}% "
                  f"margin {entry['margin_over_chance']:+.1f}  "
                  f"rho(date, margin) {rho:+.3f} "
                  f"[{entry['spearman_ci95'][0]:+.3f}, {entry['spearman_ci95'][1]:+.3f}] "
                  f"p={entry['spearman_p_two_sided']:.3f}"
                  + (f"  before-after {entry['before_minus_after']:+.1f} "
                     f"p={entry['before_minus_after_p_two_sided']:.3f}"
                     if "before_minus_after" in entry else ""), flush=True)
        report["models"][name] = block

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
