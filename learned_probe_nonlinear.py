#!/usr/bin/env python3
r"""The same measurement, with the features and the scorer taken out of our hands.

``learned_probe.py`` answers "maybe the wrong statistics were designed" by
fitting a solver rather than writing one. It does not answer the next version
of the objection, which is fair: the solver is *linear* over features this
paper designed, so every null it reports could be a fact about that reader
rather than about the file.

This runs the identical protocol -- leave-one-cluster-out, a clean synthetic
control of the same shape, the margin read against the control's worst case --
over readers that are neither.

Features
    designed   the 46 columns of ``learned_probe.item_features``
    chars      hashed character 1--4-grams of the option's rendered text, and
               nothing else, so no statistic in the reader was chosen by us
    embed      mean-pooled last hidden states of a language model reading the
               option's text, which is the literal "representation-based
               feature extractor" version of the same idea

Context
    option     the reader sees one option at a time, so a margin cannot be
               geometry of any kind
    set        the reader also sees the mean of the *other* options' vectors
               and its difference from them, which is permutation-invariant in
               the others, so it is a set reader in the sense of \S2

Scorer
    linear     the softmax-over-options likelihood of ``learned_probe.fit``
    mlp        one hidden layer on the same likelihood, i.e. a DeepSets scorer
               when the context is ``set``

    python3 learned_probe_nonlinear.py --jsonl build/mmlu.jsonl --label released \
        --extractor chars --context set --scorer mlp
"""
import argparse
import json
import random
import statistics
import zlib
from pathlib import Path

import numpy as np

from learned_probe import item_features
from mcq_audit import build_items, read_rows

NGRAM_ORDERS = (1, 2, 3, 4)


def hashed_ngrams(text, dim):
    """L2-normalised hashed character n-gram counts. Deterministic across runs."""
    vector = np.zeros(dim, dtype=np.float32)
    padded = f"\x02{text}\x03"
    for n in NGRAM_ORDERS:
        for i in range(len(padded) - n + 1):
            gram = padded[i:i + n]
            index = zlib.crc32(f"{n}|{gram}".encode("utf-8")) % dim
            vector[index] += 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def char_features(options, dim):
    return np.stack([hashed_ngrams(str(o), dim) for o in options])


def add_set_context(rows):
    """[own, mean of the others, own - that mean]: permutation-invariant in the others."""
    k = rows.shape[0]
    if k < 2:
        return np.concatenate([rows, rows, np.zeros_like(rows)], axis=1)
    total = rows.sum(axis=0, keepdims=True)
    others = (total - rows) / (k - 1)
    return np.concatenate([rows, others, rows - others], axis=1)


def build_matrix(items, extractor, context, dim, embedder=None):
    stack, clusters = [], []
    for item in items:
        options = item["options"]
        if extractor == "designed":
            rows = item_features(options)
            if rows is None:
                continue
            rows = np.asarray(rows, dtype=np.float32)
        elif extractor == "chars":
            rows = char_features(options, dim).astype(np.float32)
        else:
            rows = embedder([str(o) for o in options]).astype(np.float32)
        if context == "set":
            rows = add_set_context(rows)
        stack.append(rows)
        clusters.append(item["cluster"])
    if not stack:
        return np.zeros((0, 0, 0), dtype=np.float32), []
    return np.stack(stack), clusters


def _torch():
    import torch
    return torch


def fit(x, scorer, steps, learning_rate, l2, hidden, seed, device):
    """Maximise the per-item softmax likelihood that slot 0 is chosen."""
    torch = _torch()
    if x.shape[0] == 0:
        return None
    generator = torch.Generator(device="cpu").manual_seed(seed)
    data = torch.as_tensor(x, dtype=torch.float32, device=device)
    d = data.shape[2]
    if scorer == "linear":
        params = [torch.zeros(d, device=device, requires_grad=True)]

        def score(batch):
            return batch @ params[0]
    else:
        w1 = (torch.randn(d, hidden, generator=generator) / max(1.0, d ** 0.5)).to(device)
        w1.requires_grad_(True)
        b1 = torch.zeros(hidden, device=device, requires_grad=True)
        w2 = torch.zeros(hidden, device=device, requires_grad=True)
        params = [w1, b1, w2]

        def score(batch):
            return torch.tanh(batch @ params[0] + params[1]) @ params[2]

    optimiser = torch.optim.Adam(params, lr=learning_rate)
    for _ in range(steps):
        optimiser.zero_grad()
        scores = score(data)
        loss = -(scores[:, 0] - torch.logsumexp(scores, dim=1)).mean()
        loss = loss + l2 * sum((p * p).sum() for p in params)
        loss.backward()
        optimiser.step()
    return [p.detach() for p in params]


def predict(x, params, scorer, device):
    torch = _torch()
    data = torch.as_tensor(x, dtype=torch.float32, device=device)
    if scorer == "linear":
        return (data @ params[0]).cpu().numpy()
    return (torch.tanh(data @ params[0] + params[1]) @ params[2]).cpu().numpy()


def cross_validate(x, clusters, args, device, seed):
    clusters = np.asarray(clusters, dtype=object)
    rng = np.random.default_rng(seed)
    picks, tied = [], 0
    for held in sorted(set(clusters.tolist())):
        test = clusters == held
        params = fit(x[~test], args.scorer, args.steps, args.learning_rate,
                     args.l2, args.hidden, seed, device)
        if params is None:
            continue
        scores = predict(x[test], params, args.scorer, device)
        winners = np.abs(scores - scores.max(axis=1, keepdims=True)) < 1e-9
        tied += int((winners.sum(axis=1) > 1).sum())
        chosen = np.argmax(rng.random(scores.shape) * winners, axis=1)
        picks.extend((held, bool(pick == 0)) for pick in chosen)
    return picks, tied


def accuracy(picks):
    return sum(hit for _, hit in picks) / len(picks) if picks else float("nan")


def bootstrap(picks, reps, seed):
    by_cluster = {}
    for cluster, hit in picks:
        by_cluster.setdefault(cluster, []).append(hit)
    keys = list(by_cluster)
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        pooled = [h for key in (rng.choice(keys) for _ in keys) for h in by_cluster[key]]
        draws.append(sum(pooled) / len(pooled))
    draws.sort()
    return draws


def clean_control(args, k, n_items, n_clusters, device, embedder):
    """What this reader scores on files built so that they cannot leak."""
    from audit_calibration import synthetic_rows

    scores = []
    for replicate in range(args.replicates):
        rng = random.Random(f"{args.seed}|clean|{k}|{replicate}")
        rows = synthetic_rows(n_items, k, n_clusters, rng)
        items, found = build_items(rows, "ideal", "distractors", "capsule_uuid",
                                   "question", k)
        if found != k:
            continue
        x, clusters = build_matrix(items, args.extractor, args.context, args.dim,
                                   embedder)
        picks, _ = cross_validate(x, clusters, args, device, args.seed + replicate)
        scores.append(accuracy(picks))
        print(f"  clean {replicate + 1}/{args.replicates}: {scores[-1]:.1%}", flush=True)
    return scores


def make_embedder(model_name, device):
    """Mean-pooled last hidden states, cached per distinct string."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    model.eval()
    cache = {}

    def embed(texts):
        missing = [t for t in texts if t not in cache]
        for start in range(0, len(missing), 128):
            chunk = missing[start:start + 128]
            batch = tok(chunk, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            for text, vector in zip(chunk, pooled.cpu().numpy()):
                cache[text] = vector
        return np.stack([cache[t] for t in texts])

    return embed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=Path, action="append", required=True)
    ap.add_argument("--label", action="append", default=None)
    ap.add_argument("--n-options", type=int, default=4)
    ap.add_argument("--cluster-field", default="capsule_uuid")
    ap.add_argument("--key-field", default="ideal")
    ap.add_argument("--distractor-field", default="distractors")
    ap.add_argument("--question-field", default="question")
    ap.add_argument("--extractor", choices=("designed", "chars", "embed"), default="chars")
    ap.add_argument("--context", choices=("option", "set"), default="set")
    ap.add_argument("--scorer", choices=("linear", "mlp"), default="mlp")
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--clean-items", type=int, default=None)
    ap.add_argument("--clean-clusters", type=int, default=None)
    ap.add_argument("--embed-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    k = args.n_options
    labels = args.label or [path.stem for path in args.jsonl]
    if len(labels) != len(args.jsonl):
        raise SystemExit("--label must be given once per --jsonl")

    embedder = make_embedder(args.embed_model, args.device) if args.extractor == "embed" else None
    report = {"n_options": k, "chance": 1 / k, "extractor": args.extractor,
              "context": args.context, "scorer": args.scorer, "dim": args.dim,
              "hidden": args.hidden, "steps": args.steps, "l2": args.l2,
              "embed_model": args.embed_model if args.extractor == "embed" else None,
              "files": {}}
    first = None
    for label, path in zip(labels, args.jsonl):
        items, found = build_items(read_rows(path), args.key_field,
                                   args.distractor_field, args.cluster_field,
                                   args.question_field, k)
        if found != k:
            raise SystemExit(f"{path}: found {found} options, expected {k}")
        x, clusters = build_matrix(items, args.extractor, args.context, args.dim,
                                   embedder)
        if first is None:
            first = (x, clusters)
        picks, tied = cross_validate(x, clusters, args, args.device, args.seed)
        draws = bootstrap(picks, args.reps, args.seed)
        got = accuracy(picks)
        report["files"][label] = {
            "path": str(path), "n_items": int(x.shape[0]),
            "n_clusters": len(set(clusters)), "n_features": int(x.shape[2]),
            "cv_accuracy": got, "argmax_ties": tied,
            "cv_ci95": [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]],
        }
        print(f"{label}: {x.shape[0]} items, {len(set(clusters))} clusters, "
              f"{x.shape[2]} features -> {got:.1%} "
              f"[{report['files'][label]['cv_ci95'][0]:.1%}, "
              f"{report['files'][label]['cv_ci95'][1]:.1%}]", flush=True)

    n_items = args.clean_items or int(first[0].shape[0])
    n_clusters = args.clean_clusters or len(set(first[1]))
    clean = clean_control(args, k, n_items, n_clusters, args.device, embedder)
    report["clean"] = {"scores": clean, "mean": statistics.fmean(clean),
                       "max": max(clean), "replicates": len(clean),
                       "n_items": n_items, "n_clusters": n_clusters}
    print(f"clean control ({len(clean)} files of {n_items} items): "
          f"mean {statistics.fmean(clean):.1%}, worst case {max(clean):.1%}")
    for label, block in report["files"].items():
        block["over_clean_worst_case"] = block["cv_accuracy"] - max(clean)
        block["margin_ci95"] = [block["cv_ci95"][0] - max(clean),
                                block["cv_ci95"][1] - max(clean)]
        print(f"    {label}: {100 * block['over_clean_worst_case']:+.1f} points "
              f"[{100 * block['margin_ci95'][0]:+.1f}, "
              f"{100 * block['margin_ci95'][1]:+.1f}]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
