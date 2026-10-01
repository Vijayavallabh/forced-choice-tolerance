#!/usr/bin/env python3
r"""The with-data result under BixBench's own reader, split by where the repair took the key.

The paper's seven v1.5 run sets were read three ways through the released, placebo and
repaired options (``bixbench_withdata.py``): by the option nearest the submitted number, a
rule that reads no notebook; by each run set's own family through ``MCQ_EVAL_PROMPT``, as
BixBench reads a run (Qwen3-30B-A3B's by gemma-3-27b, ``PRIMARY_READER``); and by
gemma-3-27b, one reader for all seven. This splits every reading's contrasts by the key
groups of the registered replication -- keys the repair moved to an edge, moved inward, or
kept in their bracketed-or-edge class -- and over all 105 numeric items:

* repaired - placebo: the two rewrites share the redraw and differ only in the key's rank,
  so this is the rank's contrast with the redraw held fixed;
* repaired - released: the rewrite as a whole, the contrast the paper first reported;
* placebo - released: the redraw alone, rank held.

Each contrast is pooled over the seven run sets as the registered analysis pools its run
sets (each item's per-run-set difference averaged over run sets, the mean over items), read
by the percentile cluster bootstrap at the level the double bootstrap finds covers 95% on
the group's own capsules (``replication.calibrated``), and checked by a capsule sign-flip
test and the interval it inverts to (``randomization.signflip``), which assumes only that
a capsule's deviation is symmetric under the null. None of this was registered.

    python3 reader_split.py              # results/reader_split.json
    python3 reader_split.py --latex      # tab:readers2's rows, one line a reading
    python3 reader_split.py --latex-full # tab:readersfull's rows, every contrast with its interval
"""
import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

import bixbench_withdata as bw
import bracketing as br
import randomization
import replication as rp

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "reader_split.json"
READERS = ("nearest", "own", "gemma27b")
CONTRASTS = (("repaired", "released"), ("repaired", "placebo"), ("placebo", "released"))
GROUPS = ("moved to an edge", "moved inward", "kept", "all")
SEED = 20260925


def scorer(reader, run, sets):
    if reader == "nearest":
        return lambda r, arm: bw.nearest_is_key(r["answer"], sets[r["question_id"]][arm])
    name = br.primary(run) if reader == "own" else reader
    return lambda r, arm: br.read_score(r, name, arm)


def per_item(cond, reader, arm, base, sets, runs=br.RUNS):
    """{run: {item: arm - base}} over the numeric items, one run per item per run set."""
    out = {}
    for run in runs:
        s = scorer(reader, run, sets)
        block = {}
        for r in br.rows_of(run):
            if not r["numeric"] or r["condition"] != cond:
                continue
            a, b = s(r, arm), s(r, base)
            if a is not None and b is not None:
                block[r["question_id"]] = a - b
        out[run] = block
    return out


def pooled(by_run):
    acc = defaultdict(list)
    for block in by_run.values():
        for q, v in block.items():
            acc[q].append(v)
    return {q: float(np.mean(v)) for q, v in acc.items()}


# tab:readers2: the three readers on the seven run sets, then gemma-3-27b on the registered runs
COLUMNS = (("repaired-placebo", "moved to an edge"), ("placebo-released", "moved to an edge"),
           ("repaired-released", "moved to an edge"), ("repaired-placebo", "kept"), ("repaired-placebo", "all"))
LABELS = {"nearest": "nearest option", "own": "own model", "gemma27b": "gemma-3-27b"}


def _est(v):
    return f"${v['mean']:+.1f}$" if v else "--"


def _iv(v):
    if not v:
        return ""
    if v.get("level_capped"):
        return "{\\scriptsize$\\ddagger$}"
    return f"{{\\scriptsize$[{v['lo']:+.1f},{v['hi']:+.1f}]$}}"


def _p(sf):
    if not sf:
        return "--"
    return "$<0.001$" if sf["p"] < 0.001 else f"${sf['p']:.3f}$"


def _other(name, mode):
    """Another reader's or mode's report from ``published_reads.py analyse``; a missing one stops the table
    rather than dropping its rows."""
    path = ROOT / "results" / f"published_reads_{name}_{mode}.json"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} is missing: run published_reads.py analyse for {name}, {mode}")
    return json.loads(path.read_text())


# tab:readersfull's rows after the seven run sets' three readers: (runs, reader label, report, sign-flip label);
# gemma-3-27b allowed to decline, and the two other open readers, on whatever of each run set they read
def _registered_rows(reads):
    dec, q72, l70 = _other("gemma27b", "decline"), _other("qwen72b", "forced"), _other("llama70b", "forced")
    others = ((q72, "qwen72b", "Qwen2.5-72B"), (l70, "llama70b", "Llama-3.3-70B"))
    out = []
    if dec and "D0|data" in dec:
        out.append(("", "gemma-3-27b, refusal", dec["D0|data"], "gemma27b"))
    for rep, name, lab in others:
        if rep and "D0|data" in rep:
            out.append(("", lab, rep["D0|data"], name))
    for label, key in (("v1.0, published", "D1"), ("v1.5, new seeds", "D2|reruns|data"),
                       ("v1.5, Qwen3-235B-A22B", "D2|qwen3-235b|data")):
        gem = reads["D1"] if key == "D1" else reads["D2"][key.split("|", 1)[1]]
        g4o = _other("gpt-4o", "forced")
        if key == "D1" and g4o and "D1" in g4o:
            out.append((label, "gpt-4o", g4o["D1"], "gpt-4o"))
            label = ""
        out.append((label, "gemma-3-27b", gem, "gemma"))
        if dec and key in dec:
            out.append(("", "gemma-3-27b, refusal", dec[key], "gemma27b"))
        for rep, name, lab in others:
            if rep and key in rep:
                out.append(("", lab, rep[key], name))
    return out


# tab:readers2, one line a reading: on the published runs, each reader beside the published reading --
# how often it names the option the rule names when the number is a miss, and its agreement with the
# published reading -- then the seven run sets; the rank's contrast on the moved keys with its interval,
# its sign-flip p, and the same contrast on the kept keys and over all numeric items
def _cell(v):
    return "--" if not v else f"{_est(v)} {_iv(v)}"


def compact_rows(split, reads, readers):
    mv = "repaired-placebo|moved to an edge"
    fol = lambda e: f"${e['follows_nearest']['miss']['share']:.0f}$"
    kap = lambda e: f"${e['agreement']['pooled']['kappa']:.2f}$"
    row = lambda runs, reader, f, k, res, sf: (
        f"{runs} & {reader} & {f} & {k} & {_cell(res.get(mv) if res else None)} & {_p(sf) if sf else '--'} & "
        f"{_est(res.get('repaired-placebo|kept')) if res else '--'} & "
        f"{_est(res.get('repaired-placebo|all')) if res else '--'}\\\\")
    rows = []
    rule, pub = readers["rule"], readers["published forced"]
    rows.append(row("v1.0, published", "nearest option", "$100$", kap(rule), rule, rule["signflip"]))
    rows.append(row("", "published", fol(pub), "--", None, None))
    d1 = {"gemma27b": reads["D1"]}
    for name, mode in (("gpt-4o", "forced"), ("qwen72b", "forced"), ("llama70b", "forced"), ("gemma27b", "decline")):
        rep = _other(name, mode)
        if rep and "D1" in rep:
            d1[name if mode == "forced" else f"{name}|decline"] = rep["D1"]
    forced = (["gpt-4o"] if "gpt-4o" in d1 and "gpt-4o" in readers else []) + sorted(
        (n for n in ("qwen72b", "gemma27b", "llama70b") if n in d1),
        key=lambda n: -readers[n]["follows_nearest"]["miss"]["share"])
    for name in forced + (["gemma27b|decline"] if "gemma27b|decline" in d1 else []):
        e = readers[name]
        rows.append(row("", READER_LABELS[name], fol(e), kap(e), d1[name], e["signflip"]))
    for i, reader in enumerate(READERS):
        res = {f"repaired-placebo|{g}": split[f"data|{reader}|repaired-placebo|{g}"]["pooled"]
               for g in ("moved to an edge", "kept", "all")}
        rows.append(row("v1.5, seven configurations" if i == 0 else "", READER_LABELS.get(reader, LABELS[reader]), "--", "--", res,
                        split[f"data|{reader}|{mv}"]["signflip"]))
    for name, mode in (("qwen72b", "forced"), ("llama70b", "forced"), ("gemma27b", "decline")):
        rep = _other(name, mode)
        if rep and "D0|data" in rep:
            label = READER_LABELS[name if mode == "forced" else f"{name}|decline"]
            rows.append(row("", label, "--", "--", rep["D0|data"],
                            rep["D0|data"]["signflip"][f"{name}|{mv}"]))
    return rows


READER_LABELS = {"gemma27b": "gemma-3-27b", "qwen72b": "Qwen2.5-72B", "llama70b": "Llama-3.3-70B", "gpt-4o": "gpt-4o",
                 "gemma27b|decline": "gemma-3-27b, refusal"}


def table_rows(split, reads=None):
    rows = []
    for i, reader in enumerate(READERS):
        cells = [split[f"data|{reader}|{c}|{g}"]["pooled"] for c, g in COLUMNS]
        sf = split[f"data|{reader}|repaired-placebo|moved to an edge"]["signflip"]
        rows.append(f"{'v1.5, seven configurations' if i == 0 else ''} & {LABELS[reader]} & "
                    + " & ".join(_est(v) for v in cells) + f" & {_p(sf)}\\\\")
        rows.append(" & & " + " & ".join(_iv(v) for v in cells) + " & \\\\")
    if reads:
        for label, reader, res, sfl in _registered_rows(reads):
            cells = [res.get(f"{c}|{g}") for c, g in COLUMNS]
            sf = res["signflip"][f"{sfl}|repaired-placebo|moved to an edge"]
            rows.append(f"{label} & {reader} & " + " & ".join(_est(v) for v in cells) + f" & {_p(sf)}\\\\")
            rows.append(" & & " + " & ".join(_iv(v) for v in cells) + " & \\\\")
    return rows


def main():
    import sys
    if "--latex" in sys.argv or "--latex-full" in sys.argv:
        reads = json.loads((ROOT / "results" / "published_reads.json").read_text())
        split = json.loads(OUT.read_text())
        if "--latex-full" in sys.argv:
            rows = table_rows(split, reads)
        else:
            rows = compact_rows(split, reads,
                                json.loads((ROOT / "results" / "published_reads_readers.json").read_text()))
        for row in rows:
            print(row)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--light", action="store_true", help="a faster double bootstrap (2000 x 5000) for a check")
    args = ap.parse_args()
    outer, inner = (2000, 5000) if args.light else (4000, 20000)
    rng = np.random.default_rng(SEED)
    sets = bw.option_sets()
    capsule = {}
    for run in br.RUNS:
        for r in br.rows_of(run):
            capsule[r["question_id"]] = r["capsule"]
    group_of = {q: rp.group_of(s) for q, s in sets.items()}
    report = {"seed": SEED, "not_registered": True, "n_items": {
        g: (len(sets) if g == "all" else sum(1 for v in group_of.values() if v == g)) for g in GROUPS}}
    # per run set, each group at the level bracketing.py found covers its own capsules, as the
    # with-data tables print their per-run-set intervals
    brk_levels = json.loads((ROOT / "results" / "bracketing.json").read_text())["levels"]
    level_of = {"moved to an edge": brk_levels["moved to an edge"]["level"],
                "moved inward": brk_levels["moved inward"]["level"],
                "kept": brk_levels["rank class kept"]["level"], "all": brk_levels["whole file"]}
    for cond in ("data", "nodata"):
        for reader in READERS:
            for arm, base in CONTRASTS:
                by_run = per_item(cond, reader, arm, base, sets)
                pool = pooled(by_run)
                for g in GROUPS:
                    items = {q: v for q, v in pool.items() if g == "all" or group_of[q] == g}
                    key = f"{cond}|{reader}|{arm}-{base}|{g}"
                    entry = {"pooled": rp.calibrated(items, capsule, rng, outer, inner),
                             "signflip": randomization.signflip(items, capsule),
                             "per_run": {}}
                    for run, block in by_run.items():
                        qs = [q for q in block if g == "all" or group_of[q] == g]
                        entry["per_run"][run] = bw.cluster_interval([block[q] for q in qs], [capsule[q] for q in qs],
                                                                    level=level_of[g])
                    report[key] = entry
                    p, sf = entry["pooled"], entry["signflip"]
                    print(f"{key:58s} {p['mean']:+6.1f} [{p['lo']:+6.1f},{p['hi']:+6.1f}] at {100 * p['level']:.2f}%"
                          f"{' CAPPED' if p['level_capped'] else ''} | sign-flip p={sf['p']:.4f} "
                          f"[{sf['lo']:+6.1f},{sf['hi']:+6.1f}]", flush=True)
    report.update(protocol_pools(sets, capsule, group_of, outer, inner))
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


# Two protocols read Qwen2.5-72B and Llama-3.3-70B, so the seven run sets hold five families; each pool below
# keeps one protocol per family, so no family is counted twice.
ONE_PROTOCOL = {"one protocol, published": ["qwen72b-react", "llama70b-react", "gemma27b", "glm45air-react",
                                            "qwen3a3b-react"],
                "one protocol, text": ["qwen72b", "llama70b", "gemma27b", "glm45air-react", "qwen3a3b-react"]}


def protocol_pools(sets, capsule, group_of, outer, inner):
    """The rank's contrast on the moved keys, with the data, pooled over one protocol per family, under
    each reader; on its own generator, so it re-derives alone (``--pools``) or in a full run alike."""
    rng = np.random.default_rng(SEED + 1)
    out = {}
    for label, runs in ONE_PROTOCOL.items():
        for reader in READERS:
            pool = pooled(per_item("data", reader, "repaired", "placebo", sets, runs=runs))
            items = {q: v for q, v in pool.items() if group_of[q] == "moved to an edge"}
            key = f"data|{reader}|repaired-placebo|moved to an edge|{label}"
            out[key] = {"pooled": rp.calibrated(items, capsule, rng, outer, inner), "runs": runs,
                        "signflip": randomization.signflip(items, capsule)}
            p = out[key]["pooled"]
            print(f"{key:78s} {p['mean']:+6.1f} [{p['lo']:+6.1f},{p['hi']:+6.1f}]", flush=True)
    return out


def pools_only():
    """Add the one-protocol pools to the shipped report without recomputing the rest."""
    sets = bw.option_sets()
    capsule = {r["question_id"]: r["capsule"] for run in br.RUNS for r in br.rows_of(run)}
    group_of = {q: rp.group_of(s) for q, s in sets.items()}
    report = json.loads(OUT.read_text())
    report.update(protocol_pools(sets, capsule, group_of, 4000, 20000))
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    import sys
    if "--pools" in sys.argv:
        pools_only()
    else:
        main()
