#!/usr/bin/env python3
r"""Table 2 (tab:cost): what hiding the key's rank costs each grader, from the files the appendix tables print.

$U-P$ on the keys $U$ moves to an extreme and over all numeric items, for every set of runs graded with the data.
A row is one kind of grader on one set of runs; a kind with several graders is given as the range of their means,
and the nearest-option rule carries the intervals the appendix gives it (over capsules, at the level the double
bootstrap finds attains 95% coverage). The kinds:

* code-free, and within 5%: ``grading_variants.py``'s gradings without the notebook (tab:variants), without and
  with the option stating that none of the others is within 5% of the answer;
* with the notebook: BixBench's MCQ prompt, forced (tab:readersfull, tab:readers2) -- gpt-4o, Qwen2.5-72B,
  gemma-3-27b and Llama-3.3-70B on the published runs; each configuration's own model and the three open models on
  the seven configurations; the three open models on the new seeds and on Qwen3-235B-A22B; gpt-4o on gpt-5.1's runs
  and the current agents' (tab:strong, tab:frontier). The grading constrained to a letter is left out;
* refusal: gemma-3-27b offered BixBench's refusal option.

Nothing here is computed afresh: every value is read from the result the appendix prints it from, so the table
changes only when they do. Not part of the pre-specified test.

    python3 cost_table.py            # tab:cost's rows
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
MV, AL = "repaired-placebo|moved to an edge", "repaired-placebo|all"
CURRENT = ("gpt-6-luna", "DeepSeek-V4-Pro")


def load(name):
    return json.loads((ROOT / "results" / name).read_text())


def ci(v):
    """A mean with its interval; where no nominal level attains 95% coverage, the mean alone, marked."""
    if v.get("level_capped"):
        return f"${v['mean']:+.1f}$$^{{\\ddagger}}$"
    return f"${v['mean']:+.1f}$ {{\\scriptsize$[{v['lo']:+.1f},{v['hi']:+.1f}]$}}"


def pts(x):
    """A value in points with its sign, a value that rounds to zero as +0.0 rather than -0.0."""
    return f"{0.0 if round(x, 1) == 0 else x:+.1f}"


def span(values):
    """The range of a kind's means, or the one mean; '--' when the kind has none."""
    values = [v for v in values if v is not None]
    if not values:
        return "--"
    lo, hi = pts(min(values)), pts(max(values))
    return f"${lo}$" if lo == hi else f"${lo}$ to ${hi}$"


def mean(block, key):
    v = block.get(key) if block else None
    return v["mean"] if v else None


def variants(rank, fam, within):
    """The code-free gradings of ``fam`` (tab:variants, original rewrites): with the within-5% option or without."""
    out = []
    for key, c in rank.items():
        f, grading, design = key.split("|")
        if f == fam and design == "original" and "codefree" in grading and ("withintol" in grading) == within:
            out.append(c)
    return out


def cells():
    """(runs, grader, moved keys, all items) for every row, in the table's order. A cell is ("interval", entry),
    ("range", [means]) or ("each", [means]), the last one value per current agent."""
    gv = load("grading_variants.json")
    rank = gv["rank"]
    t2 = {(r["runs"], r["grader"]): r for r in gv["table2"]}
    readers, split, rep = load("published_reads_readers.json"), load("reader_split.json"), load("replication.json")
    gem, g4o = load("published_reads.json"), load("published_reads_gpt-4o_forced.json")
    q72, l70 = load("published_reads_qwen72b_forced.json"), load("published_reads_llama70b_forced.json")
    dec = load("published_reads_gemma27b_decline.json")
    deg = load("degenerate_runs.json")["rule_u_minus_p"]
    strong, frontier = load("strong_agent.json"), load("frontier_agents.json")["agents"]
    pooled = lambda reader, g: split[f"data|{reader}|repaired-placebo|{g}"]["pooled"]
    one = lambda v: ("range", [v])
    rows = []

    def kinds(runs, notebook, refusal, fam):
        cf, wt = variants(rank, fam, False), variants(rank, fam, True)
        if cf:
            rows.append((runs, "code-free", ("range", [mean(c, "moved to an edge") for c in cf]),
                         ("range", [mean(c, "all") for c in cf])))
        rows.append((runs, "with the notebook", ("range", [mean(b, MV) for b in notebook]),
                     ("range", [mean(b, AL) for b in notebook])))
        if refusal:
            rows.append((runs, "refusal", one(mean(refusal, MV)), one(mean(refusal, AL))))
        if wt:
            rows.append((runs, "within $5\\%$", ("range", [mean(c, "moved to an edge") for c in wt]),
                         ("range", [mean(c, "all") for c in wt])))

    fam = "v1.0, published"
    rule = readers["rule"]
    assert abs(rule[MV]["mean"] - t2[(fam, "nearest option")]["moved"]["mean"]) < 1e-9
    rows.append((fam, "nearest-option rule", ("interval", rule[MV]), one(rule[AL]["mean"])))
    kinds(fam, [g4o["D1"], q72["D1"], gem["D1"], l70["D1"]], dec["D1"], fam)

    fam = "v1.5, seven configurations"
    rows.append((fam, "nearest-option rule", ("interval", pooled("nearest", "moved to an edge")),
                 ("interval", pooled("nearest", "all"))))
    own = {MV: pooled("own", "moved to an edge"), AL: pooled("own", "all")}
    gem7 = {MV: pooled("gemma27b", "moved to an edge"), AL: pooled("gemma27b", "all")}
    kinds(fam, [own, gem7, q72["D0|data"], l70["D0|data"]], dec["D0|data"], fam)

    fam = "v1.5, new seeds"
    rows.append((fam, "nearest-option rule", ("interval", rep["D2"]["reruns|data"][MV]),
                 one(t2[(fam, "nearest option")]["all"])))
    kinds(fam, [gem["D2"]["reruns|data"], q72["D2|reruns|data"], l70["D2|reruns|data"]], dec["D2|reruns|data"], fam)

    fam = "v1.5, Qwen3-235B-A22B"
    q235, q235_deg = rep["D2"]["qwen3-235b|data"][MV], deg["Qwen3-235B-A22B, test runs"]["all runs"]
    assert abs(q235["mean"] - q235_deg["moved"]["mean"]) < 1e-9, "the rule's moved keys disagree between files"
    rows.append((fam, "nearest-option rule", ("interval", q235), one(q235_deg["all"]["mean"])))
    kinds(fam, [gem["D2"]["qwen3-235b|data"], q72["D2|qwen3-235b|data"], l70["D2|qwen3-235b|data"]],
          dec.get("D2|qwen3-235b|data"), fam)

    fam = "v1.5, gpt-5.1"
    rows.append((fam, "nearest-option rule", one(strong["rule|repaired-placebo|moved"]["mean"]),
                 one(strong["rule|repaired-placebo|all"]["mean"])))
    cf = variants(rank, fam, False)
    rows.append((fam, "code-free", ("range", [mean(c, "moved to an edge") for c in cf]),
                 ("range", [mean(c, "all") for c in cf])))
    reader = strong["reader"]
    rows.append((fam, "with the notebook", one(strong[f"{reader}|repaired-placebo|moved"]["mean"]),
                 one(strong[f"{reader}|repaired-placebo|all"]["mean"])))

    fam = "v1.5, " + ", ".join(CURRENT)
    each = lambda key: ("each", [frontier[a][key]["mean"] for a in CURRENT])
    rows.append((fam, "nearest-option rule", each("rule|repaired-placebo|moved"), each("rule|repaired-placebo|all")))
    rows.append((fam, "with the notebook", each("gpt-4o|repaired-placebo|moved"), each("gpt-4o|repaired-placebo|all")))
    return rows


def means(cell):
    """The means a cell shows."""
    kind, v = cell
    return [v["mean"]] if kind == "interval" else [x for x in v if x is not None]


def fmt(cell):
    kind, v = cell
    if kind == "interval":
        return ci(v)
    if kind == "each":
        return ", ".join(f"${pts(x)}$" for x in v)
    return span(v)


def table_rows():
    """tab:cost's rows, the set of runs named on its first row."""
    out, last = [], None
    for runs, grader, moved, allv in cells():
        out.append(f"{runs if runs != last else ''} & {grader} & {fmt(moved)} & {fmt(allv)}\\\\")
        last = runs
    return out


if __name__ == "__main__":
    for row in table_rows():
        print(row)
