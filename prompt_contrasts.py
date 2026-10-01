#!/usr/bin/env python3
r"""The prompt contrast as a paired difference, per model, and the control as a check.

Two readings the grids need, computed from the rollouts they ship.

**Paired, not two marginal intervals.** Tables~\ref{tab:grid} and
\ref{tab:gridbix} read BixBench's template (one letter) and the chat frame
(generated) on the same items and the same letter orderings, so the difference
between the two cells is a paired contrast: one bootstrap over the file's
clusters, both cells read on each resample. It is reported per model, and pooled
over the models that answer both cells by a DerSimonian--Laird random-effects
average, whose between-model spread is reported beside it.

**Chance, and the control as a check.** Both prompts carry ``Question:
[withheld]``, so the reader sees the options alone, and Proposition~\ref{prop:exch}
then fixes its expected accuracy on a clean file at exactly $1/k$. A margin is
therefore read against chance, with the released file's own cluster bootstrap;
the clean control is not subtracted, which would only add its sampling noise.
What the control is for is checking the pipeline. Its key was placed uniformly
among each item's options, so re-drawing that placement with every pick held
gives the exact null distribution of its accuracy. One cell in twenty outside
its central 95% is what the placement gives; many cells far outside it would be
a parsing or letter-mapping fault, not a finding (``control_null.py`` counts).

    python3 prompt_contrasts.py
"""
import argparse
import json
import pathlib
import re

import numpy as np

TAGS = ["llama1b", "qwen1_5b", "llama3b", "phi35mini", "gemma4b", "olmo7b", "qwen7b",
        "llama8b", "phi4", "qwen14b", "qwen32b", "llama70b", "qwen72b"]
NAMES = {"llama1b": "Llama-3.2-1B", "qwen1_5b": "Qwen2.5-1.5B", "llama3b": "Llama-3.2-3B",
         "phi35mini": "Phi-3.5-mini", "gemma4b": "gemma-3-4b", "olmo7b": "OLMo-2-1124-7B",
         "qwen7b": "Qwen2.5-7B", "llama8b": "Llama-3.1-8B", "phi4": "phi-4", "qwen14b": "Qwen2.5-14B",
         "qwen32b": "Qwen2.5-32B", "llama70b": "Llama-3.3-70B", "qwen72b": "Qwen2.5-72B"}
# the level that covers 95% on each file's own clusters (Table~\ref{tab:bootcal})
LEVELS = {"mmlupro": {"margin": 0.985, "difference": 0.9825},
          "bixall": {"margin": 0.9675, "difference": 0.96}}
OPTION = re.compile(r"^\(([A-Z])\) (.*)$", re.MULTILINE)


def load(path):
    import gzip
    from arm_intervals import resolve
    path = resolve(path)
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = [json.loads(l) for l in opener(path, "rt")]
    out = {"file": [], "clean": []}
    for r in rows:
        out[r.get("arm", "file")].append(r)
    return out


def options_of(r):
    block = r["user"].split("Options:\n", 1)[1]
    return {letter: text.strip() for letter, text in OPTION.findall(block)}


def exact_null(rollouts, draws=20000, seed=0):
    """Accuracy of the held picks when each item's key is re-drawn uniformly among its options."""
    items = {}
    for r in rollouts:
        opts = options_of(r)
        items.setdefault(r["item"], {"texts": sorted(set(opts.values())), "picks": []})
        items[r["item"]]["picks"].append(opts.get(r["answer"]))
    rng = np.random.default_rng(seed)
    n = len(rollouts)
    totals = np.zeros(draws)
    for it in items.values():
        k = len(it["texts"])
        key = rng.integers(0, k, size=draws)
        for pick in it["picks"]:
            if pick is None:
                continue
            totals += (key == it["texts"].index(pick))
    return 100.0 * totals / n


def boot(rows_by_cluster, clusters, stat, bootstrap, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(bootstrap):
        take = [i for c in rng.choice(clusters, size=len(clusters), replace=True) for i in rows_by_cluster[c]]
        out.append(stat(take))
    return np.array(out)


def interval(values, level):
    tail = 100.0 * (1.0 - level) / 2.0
    return [float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail))]


def random_effects(estimates, ses):
    """DerSimonian--Laird: pooled mean, its 95% interval, and tau."""
    y, v = np.array(estimates), np.array(ses) ** 2
    w = 1.0 / v
    fixed = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - fixed) ** 2))
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (q - (len(y) - 1)) / c) if c > 0 else 0.0
    ws = 1.0 / (v + tau2)
    mean = float(np.sum(ws * y) / np.sum(ws))
    se = float(np.sqrt(1.0 / np.sum(ws)))
    return {"mean": mean, "ci95": [mean - 1.96 * se, mean + 1.96 * se], "tau": float(np.sqrt(tau2)),
            "q": q, "n_models": len(y)}


PARAMS = {"llama1b": 1.2, "qwen1_5b": 1.5, "llama3b": 3.2, "phi35mini": 3.8, "gemma4b": 4.3, "olmo7b": 7.3,
          "qwen7b": 7.6, "llama8b": 8.0, "phi4": 14.7, "qwen14b": 14.8, "qwen32b": 32.8, "llama70b": 70.6,
          "qwen72b": 72.7}


def grid_rows(report, name):
    """One row per model: the template read one letter, the chat framing generated, and the paired
    difference, each over chance; a cell whose arms refuse is printed as its refusal rate."""
    out = []
    for e in report["files"][name]["rows"]:
        t, f, d = e["template"], e["framing"], e["framing_minus_template"]
        cell = lambda x: f"${x['over_chance']:+.1f}$ $[{x['interval'][0]:+.1f},{x['interval'][1]:+.1f}]$"
        if f["readable"]:
            chat = cell(f)
            diff = f"${d['difference']:+.1f}$ $[{d['interval'][0]:+.1f},{d['interval'][1]:+.1f}]$"
        else:
            chat, diff = f"refuses ${f['unparsed']['file']:.0f}\\%$", "--"
        out.append(f"{e['model']} & ${PARAMS[e['tag']]:.1f}$ & {cell(t)} & {chat} & {diff}\\\\")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--max-unparsed", type=float, default=5.0)
    ap.add_argument("--max-unparsed-gap", type=float, default=3.0)
    ap.add_argument("--output", default="results/prompt_contrasts.json")
    ap.add_argument("--latex", choices=("mmlupro", "bixall"), default=None,
                    help="print that file's grid table rows from the report")
    args = ap.parse_args()
    if args.latex:
        for line in grid_rows(json.loads(pathlib.Path(args.output).read_text()), args.latex):
            print(line)
        return

    report = {"levels": LEVELS, "files": {}}
    for name in ("mmlupro", "bixall"):
        lv = LEVELS[name]
        rows_out, pooled_in = [], ([], [])
        for tag in TAGS:
            from arm_intervals import resolve
            tp = resolve(f"build/dumps/{tag}_{name}_bixprompt_argmax.jsonl")
            fp = resolve(f"build/dumps/{tag}_{name}_neutral_notools.jsonl")
            if not (tp.exists() and fp.exists()):
                raise SystemExit(f"missing dump for {tag} {name}")
            t, f = load(tp), load(fp)
            entry = {"model": NAMES[tag], "tag": tag}
            for arm in ("file", "clean"):
                a, b = t[arm], f[arm]
                if len(a) != len(b) or any(x["item"] != y["item"] or x["gold"] != y["gold"] for x, y in zip(a, b)):
                    raise SystemExit(f"{tag} {name} {arm}: the two cells are not the same rollouts")
            released_t, released_f = t["file"], f["file"]
            k = released_t[0]["k"]
            chance = 100.0 / k
            clusters = sorted({r["cluster"] for r in released_t})
            by_c = {c: [i for i, r in enumerate(released_t) if r["cluster"] == c] for c in clusters}
            ht = np.array([r["answer"] == r["gold"] for r in released_t], float)
            hf = np.array([r["answer"] == r["gold"] for r in released_f], float)
            up_t = {arm: 100.0 * float(np.mean([r["answer"] == "" for r in t[arm]])) for arm in ("file", "clean")}
            up_f = {arm: 100.0 * float(np.mean([r["answer"] == "" for r in f[arm]])) for arm in ("file", "clean")}
            readable = (max(up_f.values()) <= args.max_unparsed
                        and abs(up_f["file"] - up_f["clean"]) <= args.max_unparsed_gap)
            dt = boot(by_c, clusters, lambda i: 100.0 * ht[i].mean() - chance, args.bootstrap, 0)
            df = boot(by_c, clusters, lambda i: 100.0 * hf[i].mean() - chance, args.bootstrap, 0)
            dd = boot(by_c, clusters, lambda i: 100.0 * (hf[i] - ht[i]).mean(), args.bootstrap, 0)
            entry["template"] = {"accuracy": 100.0 * ht.mean(), "over_chance": 100.0 * ht.mean() - chance,
                                 "interval": interval(dt, lv["margin"]), "unparsed": up_t}
            entry["framing"] = {"accuracy": 100.0 * hf.mean(), "over_chance": 100.0 * hf.mean() - chance,
                                "interval": interval(df, lv["margin"]), "unparsed": up_f, "readable": bool(readable)}
            entry["framing_minus_template"] = {"difference": 100.0 * (hf - ht).mean(),
                                               "interval": interval(dd, lv["difference"]),
                                               "se": float(np.std(dd, ddof=1))}
            # the control, as a check on the pipeline
            for cell, doc in (("template", t), ("framing", f)):
                clean = doc["clean"]
                acc = 100.0 * np.mean([r["answer"] == r["gold"] for r in clean])
                null = exact_null(clean)
                entry[cell]["control"] = {"accuracy": acc, "minus_chance": acc - chance,
                                          "null_percentile": float(100.0 * np.mean(null <= acc)),
                                          "null_2_5_97_5": interval(null, 0.95)}
            if readable:
                pooled_in[0].append(entry["framing_minus_template"]["difference"])
                pooled_in[1].append(entry["framing_minus_template"]["se"])
            rows_out.append(entry)
            c_t, c_f = entry["template"]["control"], entry["framing"]["control"]
            print(f"{name:7s} {NAMES[tag]:15s} template {entry['template']['over_chance']:+5.1f} "
                  f"[{entry['template']['interval'][0]:+5.1f},{entry['template']['interval'][1]:+5.1f}]  "
                  f"framing {entry['framing']['over_chance']:+5.1f} "
                  f"[{entry['framing']['interval'][0]:+5.1f},{entry['framing']['interval'][1]:+5.1f}]"
                  f"{'' if readable else ' (unreadable)'}  paired {entry['framing_minus_template']['difference']:+5.1f} "
                  f"[{entry['framing_minus_template']['interval'][0]:+5.1f},"
                  f"{entry['framing_minus_template']['interval'][1]:+5.1f}]  control "
                  f"{c_t['minus_chance']:+4.1f} (p{c_t['null_percentile']:.0f}) / "
                  f"{c_f['minus_chance']:+4.1f} (p{c_f['null_percentile']:.0f})")
        pooled = random_effects(*pooled_in)
        report["files"][name] = {"rows": rows_out, "pooled_framing_minus_template": pooled}
        print(f"{name}: pooled paired difference over {pooled['n_models']} readable models "
              f"{pooled['mean']:+.1f} [{pooled['ci95'][0]:+.1f},{pooled['ci95'][1]:+.1f}], tau {pooled['tau']:.1f}\n")
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
