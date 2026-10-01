#!/usr/bin/env python3
r"""Did the model read the option values, or recognise the option set?

\S\ref{sec:probe} reports that a framing which omits the question reads $+10.4$
on MMLU-Pro's released option sets and $+2.8$ on a placebo that holds the key's
rank and redraws every value. Two accounts fit that. Under the one the paper
argues, the reader is reading value structure, and the placebo drops because the
structure was disturbed. Under the other it is recognising option *sets* it saw
in pretraining and recalling which member was keyed, and the placebo drops
because the sets it recognises are gone.

Membership inference separates them, because the second account makes a
prediction about the option text itself and the first makes none. For every
option block a rollout actually read -- the exact string, taken from the dump --
this scores the block alone under the same weights, with the three statistics
that literature uses: mean token log-likelihood, Min-K\% Prob
\citep{minkprob} and Min-K\%++ \citep{minkpp}.

Three readings, in increasing order of how hard they are to explain away:

  1. released against the matched clean control, as an attack AUC. The control's
     option sets are synthesised here and were never published, so an attack
     that cannot separate them from MMLU-Pro's has no purchase on this file.
  2. released against the placebo, whose sets are equally unpublished. The
     recognition account needs released to score as more-seen.
  3. *within* the released file, per rollout: does the membership score predict
     whether the model got that rollout right, once the key's rank is in the
     regression? This one is immune to any difference between files, because it
     never compares two files. Recognition predicts a positive membership
     coefficient. Geometry predicts the rank coefficient carries it and the
     membership coefficient does not clear.

    python3 option_membership.py --model Qwen/Qwen2.5-14B-Instruct --device cuda:2

Every interval is the paper's cluster bootstrap over the file's own groups.
"""
import argparse
import gzip
import json
import os
import pathlib
import random
import re
import time

import numpy as np

# torch and transformers are imported inside the two functions that score option
# blocks under a model's weights. Re-running the analysis from the shipped score
# cache, and importing this module for its parsers, needs neither.

# BixBench's own template puts an "IMPORTANT:" instruction *after* the options,
# so a greedy match to end-of-string swallows it. That has two costs, and the
# second is worse than the first: no line parses as an option, so every rollout
# loses its key rank and the stratified statistic is undefined -- and the
# membership score is then taken over the options *plus* a paragraph of
# boilerplate identical on every row, which is not the option block. The same
# regex bug cost cell_split.py all 205 BixBench items once already. MMLU-Pro's
# prompts end at the options, so no reading on that file moves: the two parses
# agree on all 3,644 of its rollouts and differ on all 1,230 of BixBench's.
OPTIONS = re.compile(r"Options:\n(.*?)(?:\nIMPORTANT:|IMPORTANT:|\Z)",
                     re.DOTALL)


def file_slug(released):
    """Which file a dump is of, for the score cache's name.

    Dumps are named ``{tag}_{file}_{cell}.jsonl``, and the first version of
    this took the second underscore-separated field. Twelve of the thirteen
    tags have no underscore in them; ``qwen1_5b`` does, so its BixBench scores
    were filed under ``_5b`` and the sweep looked for a cache that was never
    going to be there. The file is named from a registry instead, and an
    unrecognised one keeps its whole stem rather than guessing.
    """
    stem = pathlib.Path(released).stem
    if "mmlupro" in stem:
        return ""
    for name in ("bixall", "bixnum", "bixbench", "labbench", "mmlu"):
        if name in stem:
            return f"_{name}"
    return f"_{stem}"


def option_block(user):
    """The option lines a rollout read, without the withheld-question preamble."""
    found = OPTIONS.search(user)
    if not found:
        raise ValueError(f"no option block in {user[:60]!r}")
    return found.group(1).strip()


def rank_of_key(block, gold):
    """Rank of the keyed value among the option values, or None if not numeric.

    Same convention as the rest of the paper: rank 1 is the smallest value.
    """
    values, letters = [], []
    for line in block.splitlines():
        head = re.match(r"\(([A-Z])\)\s*(.*)", line.strip())
        if not head:
            return None
        try:
            values.append(float(head.group(2).replace(",", "").rstrip("%")))
        except ValueError:
            return None
        letters.append(head.group(1))
    if gold not in letters:
        return None
    order = np.argsort(np.array(values), kind="stable")
    place = {letters[src]: slot + 1 for slot, src in enumerate(order)}
    return place[gold]


def score(model, tok, texts, batch_size, device):
    """Mean log-likelihood, Min-K% and Min-K%++ for each text, scored alone."""
    import torch

    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      add_special_tokens=True).to(device)
            logits = model(**enc).logits.float()
            logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
            target = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].bool()
            taken = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            # Min-K%++ standardises each position by the spread of that position's
            # own next-token log-probabilities, which is what makes it insensitive
            # to how surprising the *position* is rather than the token.
            probs = logprobs.exp()
            mu = (probs * logprobs).sum(-1)
            sigma = ((probs * (logprobs - mu.unsqueeze(-1)) ** 2).sum(-1)).clamp_min(1e-12).sqrt()
            standardised = (taken - mu) / sigma
            for row in range(len(chunk)):
                keep = mask[row]
                lp = taken[row][keep].cpu().numpy()
                st = standardised[row][keep].cpu().numpy()
                cut = max(1, int(0.2 * len(lp)))
                out.append({"n_tokens": int(len(lp)),
                            "loglik": float(lp.mean()),
                            "mink": float(np.sort(lp)[:cut].mean()),
                            "minkpp": float(np.sort(st)[:cut].mean())})
    return out


def auc(positive, negative):
    """Rank AUC of separating two score sets; 0.5 is no purchase.

    Ties get their average rank, which is what makes a constant score give
    exactly 0.5 rather than 1.0. ``scipy``'s ``rankdata`` does that in one
    sort; the first version of this looped over every distinct value and made
    the bootstrap below take ten minutes instead of ten seconds. The values are
    identical -- there is a test.
    """
    from scipy.stats import rankdata

    pos, neg = np.asarray(positive, dtype=float), np.asarray(negative, dtype=float)
    if not len(pos) or not len(neg):
        return None
    ranks = rankdata(np.concatenate([pos, neg]), method="average")
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def cluster_boot_idx(rows, statistic, bootstrap, seed, level=0.95, **kwargs):
    """Cluster bootstrap that hands the statistic row *indices*, not rows.

    Rebuilding a list of 1,800 dicts per draw was most of the runtime here.
    """
    clusters = sorted({r["cluster"] for r in rows})
    index = {c: np.array([i for i, r in enumerate(rows) if r["cluster"] == c])
             for c in clusters}
    rng = np.random.default_rng(seed)
    keys = np.array(clusters, dtype=object)
    draws = []
    for _ in range(bootstrap):
        take = np.concatenate([index[c] for c in
                               rng.choice(keys, size=len(keys), replace=True)])
        value = statistic(take, **kwargs)
        if value is not None:
            draws.append(value)
    if not draws:
        # Every draw was undefined: a cell where no stratum holds both a
        # correct and an incorrect rollout, which a refusing arm can produce.
        # numpy's percentile raises IndexError on the empty list, and an
        # interval that cannot be computed should be reported, not crashed on.
        return None
    tail = 100.0 * (1.0 - level) / 2.0
    return [float(np.percentile(draws, tail)),
            float(np.percentile(draws, 100.0 - tail))]


def cluster_boot(rows, statistic, bootstrap, seed, level=0.95):
    clusters = sorted({r["cluster"] for r in rows})
    index = {c: [i for i, r in enumerate(rows) if r["cluster"] == c] for c in clusters}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        take = [i for c in rng.choice(clusters, size=len(clusters), replace=True)
                for i in index[c]]
        value = statistic([rows[i] for i in take])
        if value is not None:
            draws.append(value)
    tail = 100.0 * (1.0 - level) / 2.0
    return [float(np.percentile(draws, tail)),
            float(np.percentile(draws, 100.0 - tail))]


def _gap(idx, score, hit, rank, strat):
    """mean(score | correct) - mean(score | wrong) over an index set."""
    h, s_ = hit[idx], score[idx]
    if not strat:
        if not h.any() or h.all():
            return None
        return float(s_[h].mean() - s_[~h].mean())
    r_ = rank[idx]
    total, weighted = 0, 0.0
    for value in np.unique(r_):
        # -1 marks a rollout whose options are not all numbers, so it has no
        # rank stratum. Folding those in as a stratum of their own shifted
        # every within-rank gap by about a thousandth -- caught only by
        # comparing against the shipped point estimates.
        if value < 0:
            continue
        cell = r_ == value
        hc, mc = cell & h, cell & ~h
        if hc.any() and mc.any():
            weighted += cell.sum() * (s_[hc].mean() - s_[mc].mean())
            total += cell.sum()
    return weighted / total if total else None


def gap_difference(released, control, bootstrap, seed):
    """Released gap minus control gap, as one interval.

    The recognition question is comparative: is the correct-minus-wrong gap
    *larger on the file the model may have seen* than on one we generated? Two
    separate intervals do not answer it, and at 32B they answer it
    misleadingly --- both arms clear zero there, by almost the same amount.

    The two arms are resampled independently, because they share nothing to
    pair on. The first version of this tried a paired bootstrap over shared
    clusters and crashed on an empty intersection, which is the evidence: the
    released file's clusters are MMLU's subject names and the control's are
    ``c0``--``c59``, it holds a different number of rollouts, and it is a
    synthetic file of the same *shape* rather than a rewriting of the same
    items. Independent resampling is what two independent files ask for, and
    the resulting interval is wider than a paired one would be, not narrower.
    """
    report = {}

    def arm(rows):
        return (np.array([r["correct"] for r in rows], dtype=bool),
                np.array([r["key_rank"] if r["key_rank"] else -1
                          for r in rows]))

    hit_r, rank_r = arm(released)
    hit_c, rank_c = arm(control)

    def index(rows):
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault(row["cluster"], []).append(i)
        keys = sorted(groups)
        return np.array(keys, dtype=object), {k: np.array(groups[k])
                                              for k in keys}

    keys_r, index_r = index(released)
    keys_c, index_c = index(control)

    for stat in ("loglik", "mink", "minkpp"):
        score_r = np.array([r[stat] for r in released], dtype=float)
        score_c = np.array([r[stat] for r in control], dtype=float)

        def both(take_r, take_c, strat):
            one = _gap(take_r, score_r, hit_r, rank_r, strat)
            two = _gap(take_c, score_c, hit_c, rank_c, strat)
            return None if one is None or two is None else one - two

        for label, strat in (("pooled", False), ("within key rank", True)):
            point = both(np.arange(len(released)), np.arange(len(control)),
                         strat)
            rng = np.random.default_rng(seed)
            draws = []
            for _ in range(bootstrap):
                take_r = np.concatenate(
                    [index_r[c] for c in
                     rng.choice(keys_r, size=len(keys_r), replace=True)])
                take_c = np.concatenate(
                    [index_c[c] for c in
                     rng.choice(keys_c, size=len(keys_c), replace=True)])
                value = both(take_r, take_c, strat)
                if value is not None:
                    draws.append(value)
            report[f"{stat}:{label}"] = {
                "difference": point,
                "interval": (None if not draws else
                             [float(np.percentile(draws, 2.5)),
                              float(np.percentile(draws, 97.5))])}
    return report


def show_gap(key, cell):
    """One gap, or why there is not one."""
    if cell["difference"] is None or cell["interval"] is None:
        return f"{key:26s}   no stratum holding both a right and a wrong rollout"
    lo, hi = cell["interval"]
    return (f"{key:26s} {cell['difference']:+7.4f} [{lo:+.4f},{hi:+.4f}]"
            + ("" if lo > 0 or hi < 0 else "   covers zero"))


def gaps(rows, bootstrap, seed):
    """Membership score, correct rollouts minus wrong, over one group."""
    report = {}
    hit = np.array([r["correct"] for r in rows], dtype=bool)
    rank = np.array([r["key_rank"] if r["key_rank"] else -1 for r in rows])
    for stat in ("loglik", "mink", "minkpp"):
        score = np.array([r[stat] for r in rows], dtype=float)

        def gap(idx, score=score, hit=hit, rank=rank, strat=False):
            return _gap(idx, score, hit, rank, strat)

        for label, strat in (("pooled", False), ("within key rank", True)):
            report[f"{stat}:{label}"] = {
                "difference": gap(np.arange(len(rows)), strat=strat),
                "interval": cluster_boot_idx(rows, gap, bootstrap, seed,
                                             strat=strat)}
    return report


def logistic(rows, features, steps=4000, lr=0.2, ridge=1e-3):
    """Plain logistic fit, returned as coefficients on standardised features."""
    if not rows:
        # No rollout had a readable rank, so there is nothing to regress. This
        # crashed on ``axis 1 is out of bounds`` rather than saying so.
        return None
    x = np.array([[r[f] for f in features] for r in rows], dtype=float)
    y = np.array([r["correct"] for r in rows], dtype=float)
    keep = np.isfinite(x).all(axis=1)
    x, y = x[keep], y[keep]
    if len(y) < 40 or y.std() == 0:
        return None
    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1.0
    x = (x - mean) / std
    x = np.hstack([np.ones((len(x), 1)), x])
    weight = np.zeros(x.shape[1])
    for _ in range(steps):
        prediction = 1.0 / (1.0 + np.exp(-x @ weight))
        gradient = x.T @ (prediction - y) / len(y) + ridge * np.r_[0.0, weight[1:]]
        weight -= lr * gradient
    return dict(zip(features, weight[1:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--shard", default=None,
                    help="comma-separated card indices to spread the weights "
                         "over, for a model that does not fit on one. Each card "
                         "is given the fraction of what is FREE on it that "
                         "--shard-auto names, read at launch, so a neighbour's "
                         "job is not evicted.")
    ap.add_argument("--shard-auto", type=float, default=0.85)
    ap.add_argument("--released", default="build/dumps/qwen14b_mmlupro_neutral_notools.jsonl")
    ap.add_argument("--placebo", default="build/dumps/qwen14b_mmlupro_placebo_neutral_notools.jsonl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--output", default="results/option_membership.json")
    ap.add_argument("--scores", default=None,
                    help="score cache; the shipped one under results/ is used "
                         "when this is not given, so the analysis re-runs "
                         "without a GPU")
    args = ap.parse_args()

    # The shipped cache is gzipped beside the rollout dumps, for the same
    # reason they are: the scores take a GPU to produce and nothing else here
    # needs one.
    tag = args.model.split("/")[-1].replace("-Instruct", "")
    # Which file the rollouts come from has to be in the cache name. It was
    # not, and the first BixBench run would have loaded the MMLU-Pro scores for
    # the same model and reported them as BixBench's -- silently, since the
    # cache carries no record of the dump it came from. MMLU-Pro keeps the
    # original names so the shipped 14B cache still resolves.
    slug = file_slug(args.released)
    shipped = pathlib.Path(
        f"results/agentic_dumps/option_membership_scores{slug}.json.gz" if
        tag == "Qwen2.5-14B" else
        f"results/agentic_dumps/option_membership_scores_{tag}{slug}.json.gz")
    cache = pathlib.Path(args.scores) if args.scores else (
        shipped if shipped.exists()
        else pathlib.Path(f"build/option_membership_scores_{tag}{slug}.json"))
    if cache.exists():
        opener = gzip.open if str(cache).endswith(".gz") else open
        with opener(cache, "rt") as fh:
            groups = json.load(fh)
        for rows in groups.values():
            for row in rows:
                if row["rank_is_modal"] is None:
                    row["rank_is_modal"] = np.nan
        print(f"reusing {cache}")
        return analyse(groups, args)

    # Only a fresh scoring run needs the model, so only it needs torch.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if args.shard:
        cards = [int(c) for c in args.shard.split(",")]
        caps = {}
        for card in cards:
            free, _ = torch.cuda.mem_get_info(card)
            caps[card] = f"{int(args.shard_auto * free / 2 ** 30)}GiB"
        print(f"sharding over {cards} with caps {caps}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            max_memory=caps)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            device_map={"": args.device})
    model.eval()

    # The placebo is only needed for the cross-file AUC, which is the reading
    # that cannot be interpreted anyway; the decisive within-file test needs
    # neither it nor the control. So a model with no placebo arm still gets the
    # test that matters, which is how this runs on more than one model.
    wanted = [("released", args.released, "file"),
              ("control", args.released, "clean")]
    if args.placebo and pathlib.Path(args.placebo).exists():
        wanted.append(("placebo", args.placebo, "file"))
    groups = {}
    for name, path, arm in wanted:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = [r for r in rows if r.get("arm", "file") == arm]
        groups[name] = [{"cluster": r["cluster"], "item": r["item"],
                         "block": option_block(r["user"]),
                         "correct": 1.0 if r["answer"] == r["gold"] else 0.0,
                         "gold": r["gold"], "k": r["k"]} for r in rows]
        print(f"{name}: {len(groups[name])} rollouts", flush=True)

    for name, rows in groups.items():
        scored = score(model, tok, [r["block"] for r in rows],
                       args.batch_size,
                       next(model.parameters()).device if args.shard
                       else args.device)
        for row, value in zip(rows, scored):
            row.update(value)
            rank = rank_of_key(row["block"], row["gold"])
            row["key_rank"] = rank
            row["rank_is_modal"] = (1.0 if rank == 5 else 0.0) if rank else np.nan
        print(f"scored {name}", flush=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        {n: [{k: (None if isinstance(v, float) and np.isnan(v) else v)
              for k, v in r.items()} for r in rows]
         for n, rows in groups.items()}), encoding="utf-8")
    return analyse(groups, args)


def analyse(groups, args):
    report = {"model": args.model, "statistics": {}, "attacks": {}, "within": {}}
    for stat in ("loglik", "mink", "minkpp"):
        report["statistics"][stat] = {
            name: {"mean": float(np.mean([r[stat] for r in rows])),
                   "interval": cluster_boot(
                       rows, lambda part, s=stat: float(np.mean([r[s] for r in part])),
                       args.bootstrap, 0)}
            for name, rows in groups.items()}
        for other in [g for g in ("control", "placebo") if g in groups]:
            value = auc([r[stat] for r in groups["released"]],
                        [r[stat] for r in groups[other]])
            report["attacks"][f"{stat}:released_vs_{other}"] = {
                "auc": value,
                "interval": cluster_boot(
                    [{"cluster": r["cluster"], stat: r[stat], "side": 1}
                     for r in groups["released"]]
                    + [{"cluster": r["cluster"], stat: r[stat], "side": 0}
                       for r in groups[other]],
                    lambda part, s=stat: (
                        auc([r[s] for r in part if r["side"] == 1],
                            [r[s] for r in part if r["side"] == 0])
                        if any(r["side"] == 1 for r in part)
                        and any(r["side"] == 0 for r in part) else None),
                    args.bootstrap, 1)}

    # The decisive test, stated as a difference rather than a regression
    # coefficient: if the reading were recognition of sets seen in pretraining,
    # the rollouts the model gets right would be the ones it scores as more
    # seen. Stratifying by the key's rank removes the geometry channel, so what
    # is left is recognition or nothing.
    report["correct_minus_wrong"] = gaps(groups["released"], args.bootstrap, 3)
    if "control" in groups:
        # The same difference on the clean control, which we generated, so
        # there is no membership in it to detect. A membership statistic is
        # also a typicality statistic, and a solver does better on typical
        # option blocks; if the gap is the same size here it is measuring that
        # and not pretraining exposure. Without this arm a positive gap reads
        # as recognition, which is the reading the 32B arm would have got.
        report["control_correct_minus_wrong"] = gaps(
            groups["control"], args.bootstrap, 4)
        report["released_minus_control"] = gap_difference(
            groups["released"], groups["control"], args.bootstrap, 5)

    numeric = [r for r in groups["released"] if r["key_rank"] is not None]
    print(f"\nwithin the released file: {len(numeric)} rollouts with a readable rank")
    for stat in ("loglik", "mink", "minkpp"):
        features = [stat, "rank_is_modal"]
        fit = logistic(numeric, features)
        if fit is None:
            continue
        report["within"][stat] = {
            "coefficients": fit,
            "intervals": {
                f: cluster_boot(numeric,
                                lambda part, fs=features, key=f: (
                                    logistic(part, fs) or {}).get(key),
                                400, 2)
                for f in features}}

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(args.output)
    # "_grid" anywhere in the stem, not "_grid.json" at the end: the BixBench
    # sweep writes ``option_membership_grid_bix.json``, which failed that test,
    # so thirteen runs each overwrote the file with their own single report and
    # the file ended up holding whichever finished last -- with no sign that
    # anything was lost, because a single report is a valid file.
    if "_grid" not in pathlib.Path(args.output).stem:
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        # Thirteen of these run at once over several cards, and the merge is a
        # read-modify-write: without the lock two models finishing together
        # lose one of themselves, which is the same silent-loss failure as the
        # arm whose entry a path bug ate while its dump sat complete on disk.
        lock = out.with_suffix(".lock")
        for attempt in range(600):
            try:
                handle = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if attempt == 599:
                    raise
                time.sleep(0.5 + 0.5 * random.random())
        try:
            merged = json.loads(out.read_text()) if (
                out.exists() and out.stat().st_size) else {}
            merged[args.model] = report
            out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        finally:
            os.close(handle)
            lock.unlink(missing_ok=True)
        print(f"  merged into {out} ({len(merged)} models)")

    print("\n=== membership score by file ===")
    for stat, cells in report["statistics"].items():
        for name, s in cells.items():
            print(f"  {stat:8s} {name:9s} {s['mean']:+8.4f} "
                  f"[{s['interval'][0]:+.4f},{s['interval'][1]:+.4f}]")
    print("\n=== attack AUC (0.5 = no purchase) ===")
    for key, s in report["attacks"].items():
        print(f"  {key:34s} {s['auc']:.3f} [{s['interval'][0]:.3f},{s['interval'][1]:.3f}]")
    print("\n=== released file: membership score, correct minus wrong ===")
    for key, s_ in report["correct_minus_wrong"].items():
        print("  " + show_gap(key, s_))
    if "control_correct_minus_wrong" in report:
        print("\n=== the same, on the clean control we generated ===")
        for key, s_ in report["control_correct_minus_wrong"].items():
            print("  " + show_gap(key, s_))
    if "released_minus_control" in report:
        print("\n=== released minus control, each arm resampled ===")
        for key, s_ in report["released_minus_control"].items():
            print("  " + show_gap(key, s_))
    print("\n=== within the released file: what predicts a correct rollout ===")
    for stat, s in report["within"].items():
        for feature, coefficient in s["coefficients"].items():
            lo, hi = s["intervals"][feature]
            print(f"  {stat:8s} {feature:14s} {coefficient:+7.3f} [{lo:+.3f},{hi:+.3f}]")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
