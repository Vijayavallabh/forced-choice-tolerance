#!/usr/bin/env python3
r"""The repair Proposition~\ref{prop:exch} prescribes, as BixBench published it.

$\Gamma$ is a property of the option set, so deleting the option set sets it to
zero identically --- there is no $V$ left to read. That repair is usually
hypothetical. It is not here: BixBench ran its no-data condition three ways and
released all three beside each other, and the field quotes one of them.

    mcq, forced      question + $k$ options, no way to decline   (chance $1/k$)
    mcq, may decline question + options + "insufficient information"
    open ended       question alone, answer in the model's words (chance $0$)

The first is the run called ``the pure recall performance of both models''. The
third is the same models, the same questions, the same graders, with the option
set removed. Nothing here is our instrument: the responses and the grades are
upstream bytes, pinned in ``data/external/PROVENANCE.json``.

Two alternative readings are checked rather than argued. That open-ended
grading is merely stricter is checked against the second arm, which keeps the
letter grading and the options and only adds a way to decline; and against
BixBench's own ``range_verifier``, a numeric tolerance, versus its
``llm_verifier``. That the questions differ between arms is checked by
requiring the identifier sets to be equal.

    python3 published_repair.py
"""
import argparse
import ast
import csv
import glob
import json
import pathlib
import re

import numpy as np

# The fifth option BixBench adds in the "may decline" arm, so that arm's chance
# rate is 1/5 and not 1/4. Its own CSV records the letter it was given per row.
DECLINE_K = 5


def truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


_CAPSULES = {}


def capsule_v15(ident):
    """The benchmark's own capsule for a question id, as ``data/bixbench.jsonl`` records it.

    Joining on question text is what we tried first and it silently matched
    nothing. Grouping by the id's ``bix-N`` prefix was next, and it is not the
    benchmark's grouping either: ``bix-61`` names six capsules, and the prefix
    put the $205$ questions in $56$ groups where every other BixBench interval
    in the paper resamples the dataset's $59$ capsules. The published files
    spell two ids with a capital (``Bix-33-q6``), so the join ignores case, and
    an id the dataset does not carry is an error, not a cluster of its own.
    """
    if not _CAPSULES:
        for line in open("data/bixbench.jsonl", encoding="utf-8"):
            row = json.loads(line)
            _CAPSULES[row["question_id"].lower()] = row["capsule_uuid"]
    key = str(ident).strip().lower()
    if key not in _CAPSULES:
        raise SystemExit(f"published id {ident!r} is not a question in data/bixbench.jsonl")
    return _CAPSULES[key]


def norm(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def declined(rows):
    """Share of rows where the model took the "insufficient information" option.

    The CSV records the letter that option was given per row (``unsure``) and
    what the model picked (``predicted``), so this is read off and not inferred.
    v1.0 writes the literal string ``empty`` in that column when there was no
    such option, which the first version of this read as a letter and reported
    a spurious zero for an arm that had no decline option at all.
    """
    if not rows:
        return None
    letter = str(rows[0].get("unsure", "")).strip()
    if not letter or letter.lower() == "empty":
        return None
    take = [r for r in rows
            if str(r["predicted"]).strip() == str(r["unsure"]).strip()]
    answered = [r for r in rows if r not in take]
    hit = sum(1 for r in answered if truthy(r["correct"]))
    return {"rate": 100.0 * len(take) / len(rows),
            "n_answered": len(answered),
            "accuracy_when_answered": (100.0 * hit / len(answered)
                                       if answered else None)}


# Phrases a model uses to say it has not been given the data. Deliberately
# conservative: a response is only counted as a declination if it contains one
# of these, so the rate is a lower bound on how often the model said so. The
# point of the rule is that its answer can be checked against a number nobody
# had to classify --- the share that picked the explicit "insufficient
# information" letter in the arm that offered one.
DECLINE_TEXT = re.compile(
    r"cannot (be )?(determine|provide|answer|calculate|give|specify)"
    r"|can't (determine|provide|answer|calculate)"
    r"|(do|does) not have access"
    r"|(is|are) not provided|not available in|without (the )?(actual|specific"
    r"|access to|further|additional) (data|information|dataset)"
    r"|unable to (determine|provide|answer)"
    r"|no specific data|requires? (the )?(actual|specific) data"
    r"|insufficient (data|information)", re.IGNORECASE)


def said_no_data(rows):
    """Share of open-ended responses that state the data was not given."""
    if not rows or str(rows[0].get("unsure", "")).strip().lower() not in ("", "empty"):
        return None
    hit = sum(1 for r in rows if DECLINE_TEXT.search(str(r["predicted"])))
    return 100.0 * hit / len(rows)


def arm(rows, key, chance, cluster_of):
    """Per-row correctness, chance rate and cluster, for one published arm."""
    out = []
    for row in rows:
        ident = key(row)
        out.append({"id": ident,
                    "correct": 1.0 if truthy(row["correct"]) else 0.0,
                    "chance": chance(row),
                    "mode": str(row.get("evaluation_mode", "")).strip(),
                    "cluster": cluster_of(row, ident)})
    return out


def paired(left, right, bootstrap=4000, seed=0, level=0.95):
    """Paired cluster bootstrap of (left margin - right margin) over capsules.

    Paired because both arms answered the same questions: resampling the
    capsules once and reading both arms on the same resample removes the
    between-item variance the two arms share.
    """
    by_id = {r["id"]: r for r in right}
    rows = [(l, by_id[l["id"]]) for l in left if l["id"] in by_id]
    clusters = sorted({l["cluster"] for l, _ in rows})
    index = {c: [i for i, (l, _) in enumerate(rows) if l["cluster"] == c]
             for c in clusters}
    gap = np.array([(l["correct"] - l["chance"]) - (r["correct"] - r["chance"])
                    for l, r in rows])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        take = [i for c in rng.choice(clusters, size=len(clusters), replace=True)
                for i in index[c]]
        draws.append(float(np.mean(gap[take])))
    tail = 100.0 * (1.0 - level) / 2.0
    sizes = np.array([len(index[c]) for c in clusters], dtype=float)
    return {"n_paired": len(rows), "n_clusters": len(clusters),
            "n_effective": float(sizes.sum() ** 2 / (sizes ** 2).sum()),
            "difference": 100.0 * float(np.mean(gap)),
            "interval": [100.0 * float(np.percentile(draws, tail)),
                         100.0 * float(np.percentile(draws, 100.0 - tail))]}


def summarise(rows, bootstrap=4000, seed=0, level=0.95):
    clusters = sorted({r["cluster"] for r in rows})
    index = {c: [i for i, r in enumerate(rows) if r["cluster"] == c]
             for c in clusters}
    gap = np.array([r["correct"] - r["chance"] for r in rows])
    hit = np.array([r["correct"] for r in rows])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        take = [i for c in rng.choice(clusters, size=len(clusters), replace=True)
                for i in index[c]]
        draws.append(float(np.mean(gap[take])))
    tail = 100.0 * (1.0 - level) / 2.0
    return {"n": len(rows), "n_clusters": len(clusters),
            "accuracy": 100.0 * float(np.mean(hit)),
            "chance": 100.0 * float(np.mean([r["chance"] for r in rows])),
            "margin": 100.0 * float(np.mean(gap)),
            "interval": [100.0 * float(np.percentile(draws, tail)),
                         100.0 * float(np.percentile(draws, 100.0 - tail))]}


def v15(model, subset=None):
    base = "data/external/zero_shot_v15"
    paths = {"mcq_forced": f"{base}/{model}-grader-mcq-refusal-False.csv",
             "mcq_decline": f"{base}/{model}-grader-mcq-refusal-True.csv",
             "open_ended": f"{base}/{model}-grader-openended.csv"}
    chance = {"mcq_forced": lambda r: 0.25,
              "mcq_decline": lambda r: 1.0 / DECLINE_K,
              "open_ended": lambda r: 0.0}
    arms, raw, ids = {}, {}, {}
    for name, path in paths.items():
        rows = read(path)
        if subset is not None:
            rows = [r for r in rows
                    if str(r.get("evaluation_mode", "")).strip() == subset]
        raw[name] = rows
        arms[name] = arm(rows, lambda r: r["uuid"], chance[name],
                         lambda r, _i: capsule_v15(r["uuid"]))
        ids[name] = {r["id"] for r in arms[name]}
    if len({frozenset(v) for v in ids.values()}) != 1:
        raise SystemExit(f"v1.5 {model}: the three arms are not the same questions")
    return arms, raw


def v10(model, subset=None):
    base = "data/external/zero_shot_v10"
    mcq = glob.glob(f"{base}/*refusal_False_mcq_{model}_1.0.csv")
    opn = glob.glob(f"{base}/*refusal_True_openended_{model}_1.0.csv")
    if not (mcq and opn):
        return {}, {}
    out, raw = {}, {}
    for name, path, chance in (("mcq_forced", mcq[0], lambda r: 0.25),
                               ("open_ended", opn[0], lambda r: 0.0)):
        rows = read(path)
        raw[name] = rows
        out[name] = arm(rows, lambda r: r["short_qid"], chance,
                        lambda r, _i: r["uuid"])
    left = {r["id"] for r in out["mcq_forced"]}
    right = {r["id"] for r in out["open_ended"]}
    if left != right:
        raise SystemExit(f"v1.0 {model}: arms differ on {len(left ^ right)} rows")
    return out, raw


def by_grader(rows):
    """Accuracy split by BixBench's own verifier, tolerance against judge."""
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        part = [r for r in rows if r["mode"] == mode]
        out[mode] = {"n": len(part),
                     "accuracy": 100.0 * float(np.mean([r["correct"] for r in part]))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--output", default="results/published_repair.json")
    ap.add_argument("--verifier", default=None,
                    help="restrict v1.5 to one of BixBench's own graders: "
                         "range_verifier is a numeric tolerance, str_verifier an "
                         "exact match, llm_verifier a judge")
    args = ap.parse_args()

    report = {"releases": {}, "bootstrap": args.bootstrap,
              "verifier": args.verifier}

    for release, loader, models in (
            ("v1.5", lambda m: v15(m, args.verifier),
             ("claude-3-5-sonnet-latest", "gpt-4o")),
            ("v1.0", lambda m: v10(m),
             ("claude-3-5-sonnet-latest", "gpt-4o"))):
        report["releases"][release] = {}
        for model in models:
            arms, raw = loader(model)
            if not arms:
                continue
            entry = {"arms": {n: summarise(r, args.bootstrap) for n, r in arms.items()},
                     "graders": {n: by_grader(r) for n, r in arms.items()},
                     "declined": {n: declined(rows) for n, rows in raw.items()},
                     "said_no_data": {n: said_no_data(rows)
                                      for n, rows in raw.items()}}
            entry["forced_minus_open"] = paired(
                arms["mcq_forced"], arms["open_ended"], args.bootstrap, 0)
            report["releases"][release][model] = entry
            print(f"\n=== {release} {model} ===")
            for name, s_ in entry["arms"].items():
                extra = ""
                if entry["declined"].get(name) is not None:
                    d_ = entry["declined"][name]
                    when = d_["accuracy_when_answered"]
                    extra = (f"  declined {d_['rate']:.1f}%, right on "
                             + (f"{when:.1f}%" if when is not None else "--")
                             + f" of the {d_['n_answered']} it answered")
                elif entry["said_no_data"].get(name) is not None:
                    extra = f"  said the data was not given {entry['said_no_data'][name]:.1f}%"
                print(f"  {name:12s} n={s_['n']:4d} caps={s_['n_clusters']:3d} "
                      f"acc {s_['accuracy']:6.2f}%  chance {s_['chance']:5.2f}%  "
                      f"margin {s_['margin']:+6.2f} "
                      f"[{s_['interval'][0]:+.2f},{s_['interval'][1]:+.2f}]{extra}")
            d = entry["forced_minus_open"]
            print(f"  options - no options: {d['difference']:+6.2f} "
                  f"[{d['interval'][0]:+.2f},{d['interval'][1]:+.2f}] "
                  f"on {d['n_paired']} paired rows, {d['n_clusters']} capsules "
                  f"({d['n_effective']:.1f} effective)")
            for name, g in entry["graders"].items():
                cells = "  ".join(f"{m or 'single'}={v['accuracy']:.1f}% (n={v['n']})"
                                  for m, v in sorted(g.items()))
                print(f"    {name:12s} by verifier: {cells}")

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
