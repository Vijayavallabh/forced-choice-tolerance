#!/usr/bin/env python3
r"""What a repair costs, measured by giving a solver the question.

``key_identity.py`` says what a repair leaks. It says nothing about what the
repair cost, and the two move together: distractors that share nothing with the
key leak nothing and are also trivial to reject, which is not a repaired
benchmark but a broken one.

This measures the second axis on the same files. Each arm is scored by a real
solver *given the question*, in the benchmark's own prompt, with the letter
order shared across arms for a given (item, draw) so the contrast is paired.
Accuracy rising above the released file is the repair making the item easier.

    python3 frontier_validity.py --arm released:build/mmlu_frontier_released_matched.jsonl \
        --arm imitation:build/mmlu_frontier_imitation.jsonl \
        --model Qwen/Qwen2.5-7B-Instruct --device cuda:2
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from no_data_probe import (MCQ_PROMPT_TEMPLATE, letter_token_ids, letters_for,
                           option_block, render, score)


def read_arm(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def paired_bootstrap(per_cluster, reps, seed):
    """Cluster bootstrap of a paired per-item difference, in accuracy points."""
    clusters = sorted(per_cluster)
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        picked = [per_cluster[rng.choice(clusters)] for _ in clusters]
        flat = [value for block in picked for value in block]
        if flat:
            draws.append(100 * statistics.fmean(flat))
    draws.sort()
    return draws


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="label:path, the first being the reference arm")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--output", default="results/frontier_validity.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    arms = []
    for spec in args.arm:
        label, _, path = spec.partition(":")
        arms.append((label, read_arm(path), path))
    sizes = {len(rows) for _, rows, _ in arms}
    if len(sizes) != 1:
        raise SystemExit(f"arms hold different item counts: {sizes}")
    n_items = sizes.pop()
    reference = arms[0][0]
    print(f"{len(arms)} arms of {n_items} items, reference {reference!r}")

    # One letter order per (item, draw), shared by every arm.
    rng = random.Random(args.seed)
    permutations = [[rng.sample(range(4), 4) for _ in range(args.draws)]
                    for _ in range(n_items)]

    conditions = []
    for label, rows, _ in arms:
        for i, row in enumerate(rows):
            options = [str(row["ideal"]).strip()] + [str(d).strip() for d in row["distractors"]]
            if len(options) != 4:
                raise SystemExit(f"{label} item {i} has {len(options)} options")
            for d, permutation in enumerate(permutations[i]):
                conditions.append({
                    "arm": label, "item": i, "draw": d,
                    "cluster": row.get("cluster") or f"item-{i}",
                    "prompt": MCQ_PROMPT_TEMPLATE.format(
                        question=str(row.get("question", "")).strip(),
                        options=option_block(options, permutation, letters_for(4))),
                    "target": permutation.index(0),
                })
    print(f"{len(conditions)} conditions per model")

    report = {"arms": {label: path for label, _, path in arms},
              "reference": reference, "n_items": n_items, "draws": args.draws,
              "models": {}}

    for name in args.model:
        print(f"\n=== {name}")
        tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map=args.device)
        model.eval()
        ids, _ = letter_token_ids(tok, 4)
        texts = [render(tok, c["prompt"]) for c in conditions]
        picks = score(model, tok, texts, ids, args.batch_size, model.device)

        correct = defaultdict(list)
        for condition, (pick, _) in zip(conditions, picks):
            correct[condition["arm"]].append(
                (condition["item"], condition["cluster"], int(pick == condition["target"])))

        block = {}
        by_item = {}
        for label in correct:
            per_item = defaultdict(list)
            cluster_of = {}
            for item, cluster, hit in correct[label]:
                per_item[item].append(hit)
                cluster_of[item] = cluster
            by_item[label] = (per_item, cluster_of)
            flat = [h for hits in per_item.values() for h in hits]
            block[label] = {"accuracy": 100 * statistics.fmean(flat),
                            "n_conditions": len(flat),
                            # Every (item, draw) hit, in item then draw order.
                            # Without these the difficulty column's intervals
                            # exist only at the level they were printed at, and
                            # re-reading them at the level that covers needed a
                            # GPU and three models to be run again.
                            "hits": [per_item[i] for i in range(n_items)]}
            print(f"  {label:<20} {block[label]['accuracy']:6.2f}%")

        base_items, cluster_of = by_item[reference]
        report.setdefault("clusters", [cluster_of[i] for i in range(n_items)])
        for label in correct:
            if label == reference:
                continue
            per_cluster = defaultdict(list)
            arm_items, _ = by_item[label]
            for item in base_items:
                difference = statistics.fmean(arm_items[item]) - statistics.fmean(base_items[item])
                per_cluster[cluster_of[item]].append(difference)
            draws = paired_bootstrap(per_cluster, args.reps, args.seed)
            point = 100 * statistics.fmean(
                [v for block_ in per_cluster.values() for v in block_])
            low = draws[int(0.025 * len(draws))]
            high = draws[int(0.975 * len(draws))]
            block[label]["vs_reference"] = {
                "points": point, "ci95": [low, high],
                "n_clusters": len(per_cluster)}
            print(f"  {label:<20} vs {reference}: {point:+.2f} points "
                  f"[{low:+.2f}, {high:+.2f}] over {len(per_cluster)} clusters")
        report["models"][name] = block
        del model
        torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
