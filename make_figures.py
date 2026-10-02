#!/usr/bin/env python3
"""Generate the manuscript's figures from the shipped results.

Every number drawn here is read from a file in ``results/``; nothing is typed
into this script except the definitions Figure 2's schematic states.
``validate_artifact.py`` re-reads the same files and checks
them against the numbers typed in ``main.tex``, so a figure and the text cannot
disagree without one of the two failing.

One command, ``python3 make_figures.py`` from any directory, writes every figure:

    figures/overview.{pdf,png}       Figure 1 as the paper prints it: panels (a)-(b)
    figures/overview_full.{pdf,png}  Figure 1 with all four panels, for the poster
    figures/design.{pdf,png}         Figure 2: what is compared, one answer graded two ways
    figures/cost.{pdf,png}           Figure 3: who bears the cost of a hidden rank, and what each design trades
    figures/test.{pdf,png}           Figure 4: the pre-specified test, each hypothesis on each data set
    figures/norank.{pdf,png}         Figure 6: no model we tested follows v1.5's key rank
    figures/withdata.{pdf,png}       Figure 8: with the data, the options decide the reading
    figures/scaling.{pdf,png}        Figure 9: the cost of a hidden rank against agent accuracy

(Figures 5 and 7, the prompts, are typeset in main.tex.)

Canvases are the NeurIPS text width, 5.5 in, and the paper includes each at
``width=\\linewidth`` with no trim, so the absolute font sizes below are the
sizes printed. The paper's Figure 1 is its own canvas holding only the panels it
shows, not a clip of the four-panel one, so its PDF carries no text from panels
(c) and (d).
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tate-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                     "axes.titlesize": 7.8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     # read when a patch is made, so a bar's legend handle is hatched alike
                     "hatch.linewidth": 0.5})

# BLUE, RUST and MID pass the dataviz palette check on a white page (validate_palette.js, light mode):
# adjacent pairs differ by at least deltaE 12.0 under deuteranopia and 15.6 with full colour vision; MID is
# the neutral "released" class and carries no hue on purpose. Marker shapes encode the same classes again.
BLUE, GREEN, RUST, GREY = "#2f6aa8", "#327969", "#b5562b", "#6b6b6b"
PURPLE, PALE, MID = "#7a4d8c", "#d9d9d9", "#8c8c8c"
WIDTH = 5.5
SMALL = 6.5  # the smallest text in the paper's figures, in points as printed


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save(fig, stem):
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    # no creation date, so a figure whose results have not changed regenerates byte for byte
    fig.savefig(out / f"{stem}.pdf", metadata={"CreationDate": None})
    fig.savefig(out / f"{stem}.png", dpi=220)
    smallest = min(t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip())
    width, height = fig.get_size_inches()
    plt.close(fig)
    print(f"wrote figures/{stem}.pdf: {width:g} x {height:.3f} in, smallest text {smallest:g} pt")


def spines(ax):
    ax.spines[["top", "right"]].set_visible(False)


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=7.8, fontweight="bold")


def interval(ax, x, block, colour, fmt="o", ms=3.4, horizontal=False, **kw):
    """One point with its interval; ``block`` has mean/lo/hi or points/covering. Returns (lo, hi)."""
    mean = block["mean"] if "mean" in block else block["points"]
    lo, hi = (block["lo"], block["hi"]) if "lo" in block else block["covering"]
    err = [[mean - lo], [hi - mean]]
    if horizontal:
        ax.errorbar(mean, x, xerr=err, fmt=fmt, ms=ms, color=colour, lw=0.9, capsize=1.5, **kw)
    else:
        ax.errorbar(x, mean, yerr=err, fmt=fmt, ms=ms, color=colour, lw=0.9, capsize=1.5, **kw)
    return lo, hi


RUNS = [("qwen72b", "Qwen2.5-72B", "text"), ("llama70b", "Llama-3.3-70B", "text"),
        ("gemma27b", "gemma-3-27b", "text"), ("qwen72b-react", "Qwen2.5-72B", "publ."),
        ("llama70b-react", "Llama-3.3-70B", "publ."), ("glm45air-react", "GLM-4.5-Air", "publ."),
        ("qwen3a3b-react", "Qwen3-30B-A3B", "publ.")]

# ---------------------------------------------------------------------------
# Figure 1: the argument in four panels
# ---------------------------------------------------------------------------
drift = load("results/release_drift.json")
arms_law = load("results/bixbench_wrong_step.json")["arms"]
brk = load("results/bracketing.json")


def panel_a(ax):
    """(a) one option set, read two ways"""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(ax, "(a) One property, two effects")
    # Values on a line; the agent's answer (triangle) is read as the nearest option.
    ROWS = [(0.66, "Bracketed key (as released)", [0.05, 0.36, 0.50], 0.21,
             [(0.11, False), (0.31, False)],
             ["without data: the 2nd smallest", "is the key", "with data: a miss on either",
              "side is nearer a distractor"]),
            (0.16, "Extreme key (uniform rank)", [0.24, 0.36, 0.50], 0.09,
             [(0.02, True)],
             ["without data: no rank favoured", "(rank drawn uniformly)", "with data: any miss on the",
              "open side is nearest the key"])]
    for y, label, distractors, key, misses, notes in ROWS:
        ax.plot([0.0, 0.54], [y, y], color="black", lw=0.7)
        ax.annotate("", xy=(0.56, y), xytext=(0.52, y),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black"))
        ax.text(0.0, y + 0.13, label, fontsize=6.9, fontweight="bold", va="bottom")
        ax.plot(distractors, [y] * 3, "o", ms=5.0, color=GREY, mec="black", mew=0.4, zorder=3)
        ax.plot([key], [y], "*", ms=9.0, color=RUST, mec="black", mew=0.4, zorder=4)
        for x, right in misses:
            ax.plot([x], [y - 0.085], "^", ms=4.2, color=BLUE, zorder=4)
            ax.text(x, y - 0.215, "✓" if right else "✗", fontsize=7.2, ha="center",
                    va="bottom", color=GREEN if right else "#a33a3a", fontweight="bold")
        for i, line in enumerate(notes):
            ax.text(0.60, y + 0.14 - 0.10 * i - (0.035 if i >= 2 else 0), line, fontsize=SMALL,
                    va="center", color="#222222", fontweight="bold" if i in (0, 2) else "normal")
    handles = [Line2D([], [], marker="*", ls="", ms=7.5, color=RUST, mec="black", mew=0.4, label="key"),
               Line2D([], [], marker="o", ls="", ms=4.6, color=GREY, mec="black", mew=0.4,
                      label="distractor"),
               Line2D([], [], marker="^", ls="", ms=4.2, color=BLUE, label="agent's answer")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.24), ncol=3,
              frameon=False, handletextpad=0.25, columnspacing=0.9)


def panel_b(ax):
    """(b) BixBench: where the key sits among its four values; the rank-uniform
    rewrite's bars are hatched as well as coloured, so they survive grey print"""
    laws = [("v1.0", drift["v1_0"]["key_by_rank"], PALE, None),
            ("v1.5", drift["v1_5"]["key_by_rank"], RUST, None),
            ("v1.5, $U$", arms_law["repaired"]["key_rank_pct"], BLUE, "//////")]
    width = 0.26
    for i, (label, law, colour, hatch) in enumerate(laws):
        xs = [r + (i - 1) * width for r in range(4)]
        ax.bar(xs, law, width, color=colour, edgecolor="black", lw=0.4, hatch=hatch,
               label=f"{label}: {law[1] + law[2]:.0f}% bracketed")
    second = drift["v1_5"]["key_by_rank"][1]
    ax.text(1.0, second + 1.5, f"{second:.1f}%", ha="center", fontsize=SMALL, color=RUST)
    ax.axhline(25, color="black", lw=0.7, ls=":")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["smallest", "2nd", "3rd", "largest"], fontsize=6.6)
    ax.set_ylabel("key at that rank (%)")
    ax.set_ylim(0, 86)
    ax.legend(frameon=False, loc="upper right", handlelength=1.0, borderaxespad=0.0, fontsize=SMALL)
    title(ax, "(b) BixBench: the key's rank")
    spines(ax)


def panel_c(ax, heading="(c) The cost a grader bears", releases=False):
    """(c) how much of the nearest-option rule's cost of hiding the rank each grading bears: its gain
    on the moved keys as a share of the rule's on the same answers, against its proximity weight
    (Remark 1); code-free graders filled, BixBench's graders with the notebook open, the "none
    within 5%" option a cross; with ``releases``, a second legend names the two colours"""
    cost = load("results/proximity_cost.json")
    STYLE = {"code-free": dict(marker="o", mfc=None), "notebook": dict(marker="o", mfc="white"),
             "none within 5%": dict(marker="x", mfc=None)}
    for g in cost["gradings"]:
        colour = RUST if g["runs"].startswith("v1.0") else BLUE
        st = STYLE[g["kind"]]
        ax.plot(g["lambda"], g["share"], st["marker"], ms=3.8, color=colour, mfc=st["mfc"] or colour,
                mew=0.8, zorder=3)
    ax.plot([1.0], [1.0], "*", ms=7.5, color="black", zorder=4)
    ax.text(0.97, 1.08, "rule", fontsize=SMALL, ha="right")
    ax.plot([0, 1], [0, 1], color="black", lw=0.6, ls="--")
    ax.axhline(0, color=PALE, lw=0.6)
    ax.set_xlim(-0.02, 1.05)
    shares = [g["share"] for g in cost["gradings"]]
    ax.set_ylim(min(-0.15, min(shares) - 0.05), max(1.45, max(shares) + 0.1))  # every grading in view
    ax.set_xlabel(r"proximity weight $\lambda$")
    ax.set_ylabel("share of the rule's gain\non the moved keys")
    ax.plot([], [], "o", color="black", ms=3.4, label="code-free")
    ax.plot([], [], "o", color="black", mfc="white", mew=0.8, ms=3.4, label="with the notebook")
    ax.plot([], [], "x", color="black", ms=3.6, label="within-5% option" if releases else "none within 5%")
    # 5.9 pt: at 6.5 its last entry runs into the highest code-free grading (poster only)
    kinds = ax.legend(frameon=False, loc="upper left", handlelength=0.8, borderaxespad=0.0,
                      fontsize=SMALL if releases else 5.9)
    if releases:
        ax.add_artist(kinds)
        ax.text(0.80, 0.70, r"share $=\lambda$", fontsize=SMALL, rotation=33, ha="center", va="bottom")
        handles = [Line2D([], [], marker="s", ls="", ms=3.6, color=RUST, label="v1.0, published runs"),
                   Line2D([], [], marker="s", ls="", ms=3.6, color=BLUE, label="v1.5 runs")]
        ax.legend(handles=handles, frameon=False, loc="lower right", handlelength=0.8, borderaxespad=0.0,
                  fontsize=SMALL)
    title(ax, heading)
    spines(ax)


def panel_d(ax):
    """(d) what the rank alone credits: repaired - placebo on the keys the repair moved to an edge,
    with the data, read by the nearest option and by each run set's own reader (reader_split.py)"""
    split = load("results/reader_split.json")
    nearest = split["data|nearest|repaired-placebo|moved to an edge"]["per_run"]
    own = split["data|own|repaired-placebo|moved to an edge"]["per_run"]
    spans = []
    for j, (run, _, _) in enumerate(RUNS):
        spans.append(interval(ax, j - 0.14, nearest[run], BLUE))
        spans.append(interval(ax, j + 0.14, own[run], RUST, mfc="white", mew=0.8))
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(2.5, color=PALE, lw=0.8)
    ax.set_xticks(range(len(RUNS)))
    # 5.5 pt: seven two-line labels share 2.4 in, and above about 5.9 pt neighbours touch (poster only)
    ax.set_xticklabels([f"{name.split('-')[0]}\n{'text' if proto == 'text' else 'publ.'}"
                        for _, name, proto in RUNS], fontsize=5.5)
    # wide enough for every interval, with room above them for the legend
    ax.set_ylim(min(-22, min(lo for lo, _ in spans) - 3), max(62, max(hi for _, hi in spans) + 16))
    ax.set_ylabel("$U-P$ on the\nmoved keys (points)")
    moved = split["n_items"]["moved to an edge"]
    ax.plot([], [], "o", color=BLUE, ms=3.4, label=f"nearest option ({moved} keys)")
    ax.plot([], [], "o", color=RUST, mfc="white", mew=0.8, ms=3.4, label="own model as MCQ grader")
    ax.legend(frameon=False, loc="upper right", handlelength=0.8, borderaxespad=0.0, fontsize=SMALL)
    title(ax, "(d) What hiding the rank accepts")
    spines(ax)


# The four-panel canvas is 3.75 in tall, and its top row, titles to legend, lies in its top 128 pt.
# The paper prints that row alone on a canvas 128 pt tall. Rows are laid out as fractions of the
# four-panel canvas and placed by their distance from the top edge, so (a) and (b) print the same
# on either canvas.
FULL_HEIGHT, TOP_HEIGHT = 3.75, 128 / 72


def overview(height, panels):
    fig = plt.figure(figsize=(WIDTH, height))

    def row(fraction):  # a height on the four-panel canvas, as a fraction of this one
        return 1 - (1 - fraction) * FULL_HEIGHT / height

    upper = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], left=0.005, right=0.99, top=row(0.95),
                             bottom=row(0.60), wspace=0.46)
    panel_a(fig.add_subplot(upper[0]))
    panel_b(fig.add_subplot(upper[1]))
    if panels == "abcd":
        lower = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.3], left=0.095, right=0.99, top=row(0.47),
                                 bottom=row(0.115), wspace=0.40)
        panel_c(fig.add_subplot(lower[0]))
        panel_d(fig.add_subplot(lower[1]))
    return fig


save(overview(TOP_HEIGHT, "ab"), "overview")
save(overview(FULL_HEIGHT, "abcd"), "overview_full")

# ---------------------------------------------------------------------------
# Figure 2: what is compared -- one answer, graded by the tolerance and through three option sets
# ---------------------------------------------------------------------------
from matplotlib.patches import FancyBboxPatch  # noqa: E402

LINE = 2.35  # one 6.5-pt line, in the schematic's units (100 across the 5.5-in canvas)


def box(ax, x, y, w, h, heading, lines, fill="white"):
    """A rounded box with a bold heading and its lines, top-aligned."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fill,
                                ec=GREY, lw=0.7))
    top = y + h - 1.1
    ax.text(x + 1.2, top, heading, fontsize=7.0, fontweight="bold", va="top")
    for i, line in enumerate(lines):
        ax.text(x + 1.2, top - 2.7 - LINE * i, line, fontsize=SMALL, va="top", color="#222222")


def arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black", shrinkA=0, shrinkB=0,
                                mutation_scale=7))


fig = plt.figure(figsize=(WIDTH, 1.5))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100 * 1.5 / WIDTH)
ax.axis("off")
box(ax, 0.3, 3.0, 19.8, 20.0, "One run",
    ["BixBench's agent", "analyses the data,", "never sees the", "options and submits", "a free-text answer.",
     "$a$: its last number"])
box(ax, 24.0, 19.0, 40.0, 8.0, "Reference: a tolerance",
    [r"correct when $|a-y|\leq 0.05\,|y|$, $y$ the key"], fill="#f3f3f3")
box(ax, 24.0, 0.3, 40.0, 16.7, "Through options, after the run",
    ["$R$ released; $P$ redrawn, key's rank kept;", "$U$ redrawn, key's rank drawn uniformly",
     "graders: MCQ grader with the notebook,", "forced or with refusal; code-free;",
     "within-5% option; nearest-option rule"], fill="#f3f3f3")
box(ax, 67.5, 17.0, 32.2, 10.0, "Excess over the tolerance",
    ["graded correct through $R$", "minus within 5%"])
box(ax, 67.5, 0.3, 32.2, 12.0, "Cost of hiding the rank",
    ["$U-P$ on the keys that $U$", "moves from bracketed", "to extreme"])
arrow(ax, (20.1, 19.0), (24.0, 22.5))
arrow(ax, (20.1, 8.0), (24.0, 8.0))
arrow(ax, (64.0, 23.0), (67.5, 23.0))
arrow(ax, (64.0, 13.5), (67.5, 19.0))
arrow(ax, (64.0, 6.0), (67.5, 6.0))
save(fig, "design")

# ---------------------------------------------------------------------------
# Figure 3: who bears the cost of a hidden rank, and what each design of the options trades
# ---------------------------------------------------------------------------
from option_design import TABLE as DESIGNS  # tab:design's rows: (design, its distractors, its key rank)

design = load("results/option_design.json")
RANK_COLOUR = {"released": MID, "middle": RUST, "uniform": BLUE}
SPACING_MARKER = {"$R$, released": "s", "$P$": "s", "$U$": "s", "redrawn": "o", "a fifth apart": "D",
                  "a tenth apart": "D", "agents' errors": "^"}


def panel_designs(ax):
    """(b) every design of v1.5's numeric options (tab:design): the rank rule's gain on held-out capsules
    against the share of misses the nearest-option rule accepts. Colour is the key's rank policy and shape
    how the distractors are written; each rank policy is labelled on the plot, so no class rests on colour
    alone"""
    rel = design["releases"]["v1.5"]["designs"]
    points = []
    for name, distractors, rank in DESIGNS:
        e = rel[name]
        x, y = e["leak"]["held_out"], e["rule"]["all"]["misses"]["accepted"]
        points.append((x, y))
        ax.plot(x, y, SPACING_MARKER[distractors], ms=4.0, color=RANK_COLOUR[rank], mec="black", mew=0.4,
                zorder=3)
    xs, ys = zip(*points)
    lo_x, hi_y = min(xs) - 3, max(ys) + 4
    # the corner a design would need, little rank leak and few misses accepted: none lies in it
    ax.fill_between([lo_x, 5], 0, 10, color=PALE, alpha=0.6, lw=0, zorder=1)
    ax.text(lo_x + 0.8, 0.8, "no design\nreaches here", fontsize=SMALL, color="#444444", va="bottom")
    for x, y, text, colour, ha in ((lo_x + 0.8, 27.2, "rank uniform", BLUE, "left"),
                                   (21.0, 9.4, "rank off the extremes", RUST, "center"),
                                   (23.0, 18.2, "rank as released", MID, "center"),
                                   (16.2, 4.4, "a fifth or a\ntenth apart", RUST, "left")):
        ax.text(x, y, text, fontsize=SMALL, color=colour, ha=ha, va="center", fontweight="bold")
    ax.set_xlim(lo_x, max(xs) + 4.5)
    ax.set_ylim(0, hi_y)
    ax.set_xlabel("rank leak on held-out capsules (points)")
    ax.set_ylabel("misses the nearest-option\nrule accepts (%)")
    title(ax, "(b) What each design trades")
    spines(ax)


fig = plt.figure(figsize=(WIDTH, 2.3))
grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], left=0.135, right=0.99, top=0.90, bottom=0.17,
                        wspace=0.44)
panel_c(fig.add_subplot(grid[0]), heading="(a) The cost a grader bears", releases=True)
panel_designs(fig.add_subplot(grid[1]))
save(fig, "cost")

# ---------------------------------------------------------------------------
# Figure 4: the pre-specified test -- each hypothesis's contrast under the nearest-option rule on each data set
# ---------------------------------------------------------------------------
test = load("results/replication.json")
# (hypothesis, what it contrasts, its result key, read without the data); H3, a share of the newly accepted runs,
# is a count rather than a contrast and is left to the table
TESTED = [("H1", "moved keys, $U-R$", "gain|moved to an edge", False),
          ("H2", "unchanged keys, $U-R$", "gain|kept", False),
          ("H5", "inward keys, $U-R$", "gain|moved inward", False),
          ("H4", "moved keys, $U-P$", "repaired-placebo|moved to an edge", False),
          ("H6", "moved keys, $U-R$, no data", "gain|moved to an edge", True)]
# RUST, BLUE and MID pass the palette check together (MID carries no hue on purpose); markers repeat the identity
DATASETS = [("v1.0, published runs", test["D1"], None, RUST, "o"),
            ("v1.5, new seeds", test["D2"]["reruns|data"], test["D2"]["reruns|nodata"], BLUE, "s"),
            ("v1.5, Qwen3-235B-A22B", test["D2"]["qwen3-235b|data"], test["D2"]["qwen3-235b|nodata"], MID, "D")]

fig = plt.figure(figsize=(WIDTH, 1.5))
ax = fig.add_axes([0.283, 0.235, 0.43, 0.74])
for i, (h, label, key, nodata) in enumerate(TESTED):
    for j, (name, res, res_nodata, colour, marker) in enumerate(DATASETS):
        block = res_nodata if nodata else res
        if block is None:          # the published runs have no run without the data
            continue
        v = block[key]
        passed = block["verdicts"][h]
        # H2 passes its stated criterion on a mean within 5 points of 0 even where the interval excludes 0; the
        # paper counts that as a failure, and so does the figure
        if h == "H2" and not v.get("level_capped") and not v["lo"] <= 0 <= v["hi"]:
            passed = False
        y = i + (j - 1) * 0.24
        if not v.get("level_capped"):  # no interval where no nominal level attains 95% coverage
            ax.plot([v["lo"], v["hi"]], [y, y], color=colour, lw=0.9, solid_capstyle="butt")
        ax.plot(v["mean"], y, marker, ms=3.6, color=colour, mfc=colour if passed else "white", mew=0.8, zorder=3)
ax.axvline(0, color="black", lw=0.7)
ax.set_yticks(range(len(TESTED)))
ax.set_yticklabels([f"{h}: {label}" for h, label, _, _ in TESTED], fontsize=SMALL)
ax.set_ylim(len(TESTED) - 0.55, -0.55)
ax.set_xlabel("contrast under the nearest-option rule (points)")
spines(ax)
handles = [Line2D([], [], marker=m, ls="-", lw=0.9, ms=3.6, color=c, label=n) for n, _, _, c, m in DATASETS]
handles += [Line2D([], [], marker="o", ls="", ms=3.6, color="black", label="hypothesis passes"),
            Line2D([], [], marker="o", ls="", ms=3.6, color="black", mfc="white", mew=0.8, label="fails")]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.725, 0.56), frameon=False, handlelength=1.4,
           fontsize=SMALL)
save(fig, "test")

# ---------------------------------------------------------------------------
# Figure 6: no model we tested follows v1.5's key rank
# ---------------------------------------------------------------------------
import bixbench_v10_grid as v10grid
import icl_analysis as icla

withheld = load("results/rank_attribution.json")["open_question_withheld"]
gpt_nodata = load("results/openai_nodata_gpt-4o.json")
gpt_withheld = gpt_nodata["releases"]["v15"]["withheld"]["released"]
survey = {b["benchmark"]: b for b in load("results/channel_survey.json")["benchmarks"]}
key_second = 100 * survey["BixBench v1.5"]["key_by_rank"]["1"]
icl = load("results/icl_probe.json")

# the thirteen models in order of size; the grid's module writes Llama-3.1-8B with its "Meta-" prefix
size = {name.replace("Meta-", ""): b for _, name, b in v10grid.MODELS}
rows = [(m, withheld[m]["pick_rank_shares"][1], BLUE, "o") for m in sorted(withheld, key=lambda m: size[m])]
rows.append((gpt_nodata["served"][0].replace("gpt-4o-", "gpt-4o "), gpt_withheld["pick_rank_shares"][1], RUST, "D"))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 2.3), gridspec_kw={"width_ratios": [1.0, 0.85]})
for i, (name, share, colour, marker) in enumerate(rows):
    ax1.plot(share, i, marker, ms=3.4, color=colour)
ax1.axvline(100 / 4, color="black", lw=0.7, ls=":")
ax1.axvline(key_second, color=RUST, lw=0.9, ls="--")
ax1.text(100 / 4 - 1, -1.1, "uniform choice", ha="right", va="center", fontsize=SMALL)
ax1.text(key_second - 1, -1.1, "keys at this rank", ha="right", va="center", fontsize=SMALL, color=RUST)
ax1.set_yticks(range(len(rows)))
ax1.set_yticklabels([r[0] for r in rows], fontsize=SMALL)
ax1.set_ylim(len(rows) - 0.5, -1.8)
ax1.set_xlim(0, 60)
ax1.set_xlabel("selections on the second-smallest option (%)")
title(ax1, "(a) Without the question")
spines(ax1)

models = sorted(icl["models"], key=lambda r: icla.PARAMETERS_B[r["model"]])
STEMS = (("withheld", "question withheld", BLUE, "o"), ("shown", "shown", RUST, "s"))
for i, rec in enumerate(models):
    for j, (stem, _, colour, marker) in enumerate(STEMS):
        c = next(c for c in rec["contrasts"] if c["question"] == "attributable to key ranks"
                 and c["stem"] == stem and c["n_shots"] == 64)
        block = {"mean": 100 * c["difference"], "lo": 100 * c["ci95"][0], "hi": 100 * c["ci95"][1]}
        interval(ax2, i + (j - 0.5) * 0.3, block, colour, fmt=marker, ms=3.0, horizontal=True)
for _, label, colour, marker in STEMS:
    ax2.plot([], [], marker, color=colour, ms=3, label=label)
ax2.axvline(0, color="black", lw=0.7)
ax2.set_yticks(range(len(models)))
ax2.set_yticklabels([icla.SHORT[r["model"]] for r in models], fontsize=SMALL)
ax2.set_ylim(len(models) - 0.5, -1.6)
ax2.set_xlabel("released minus uniform (points)")
ax2.legend(frameon=False, loc="upper center", ncol=2, fontsize=SMALL, handlelength=0.8, borderaxespad=0.0,
           columnspacing=1.0)
title(ax2, "(b) Examples in context")
spines(ax2)
fig.tight_layout(pad=0.3, w_pad=1.2)
save(fig, "norank")

# ---------------------------------------------------------------------------
# Figure 8: with the data, the options decide the reading
# ---------------------------------------------------------------------------
withdata = load("results/bixbench_withdata.json")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 2.25), gridspec_kw={"width_ratios": [0.9, 1.1]})

OWN = {"qwen72b": "qwen72b", "llama70b": "llama70b", "gemma27b": "gemma27b",
       "qwen72b-react": "qwen72b", "llama70b-react": "llama70b", "glm45air-react": "glm45air",
       "qwen3a3b-react": "gemma27b"}
SETS = ("released", "placebo", "repaired")
# the two protocols differ in marker and line as well as colour, so they survive grey print
FAMILY = {"text": dict(color=BLUE, marker="o", ls="-"), "publ.": dict(color=RUST, marker="s", ls="--")}
for run, name, proto in RUNS:
    block = withdata[f"{run}|data"][f"reader:{OWN[run]}"]["forced"]["numeric"]
    values = [block[s]["mean"] for s in SETS]
    ax1.plot(range(3), values, ms=3.0, lw=0.9, alpha=0.9, **FAMILY[proto])
ax1.axhline(25, color="black", lw=0.7, ls=":")
ax1.set_xticks(range(3))
ax1.set_xticklabels(["$R$", "$P$", "$U$"])
ax1.set_xlim(-0.25, 2.25)
ax1.set_ylabel("graded correct, forced (%)")
ax1.plot([], [], ms=3, lw=0.9, label="tools in text", **FAMILY["text"])
ax1.plot([], [], ms=3, lw=0.9, label="as published (ReAct)", **FAMILY["publ."])
ax1.legend(frameon=False, loc="upper left", handlelength=2.0, borderaxespad=0.1)
title(ax1, "(a) Same runs, three option sets")
spines(ax1)

PAIRS = [("qwen72b - llama70b", "over Llama-3.3-70B", BLUE),
         ("qwen72b - gemma27b", "over gemma-3-27b", GREEN)]
READINGS = [("within_5pct", "open answer within 5%"),
            ("nearest|released", "$R$: nearest option"),
            ("own|released", "$R$: own model"),
            ("gemma27b|released", "$R$: gemma-3-27b"),
            ("nearest|repaired", "$U$: nearest option"),
            ("own|repaired", "$U$: own model"),
            ("gemma27b|repaired", "$U$: gemma-3-27b")]
for p, (pair, label, colour) in enumerate(PAIRS):
    for i, (key, _) in enumerate(READINGS):
        interval(ax2, i + (p - 0.5) * 0.3, brk["agent_pairs"][pair][key], colour,
                 horizontal=True, fmt="o" if i == 0 else "s", ms=3.0)
    ax2.plot([], [], "o", color=colour, ms=3, label=label)
ax2.axvline(0, color="black", lw=0.7)
ax2.axhline(0.5, color=PALE, lw=0.8)
ax2.set_yticks(range(len(READINGS)))
ax2.set_yticklabels([label for _, label in READINGS], fontsize=SMALL)
ax2.invert_yaxis()
ax2.set_xlabel("Qwen2.5-72B's lead (points)")
ax2.set_ylim(len(READINGS) - 0.4, -1.5)
# tighter rows than the default keep the 6.5-pt legend clear of the first interval
ax2.legend(frameon=False, loc="upper left", handlelength=0.8, borderaxespad=0.0, labelspacing=0.3,
           fontsize=SMALL)
title(ax2, "(b) Lead under each grading")
spines(ax2)
fig.tight_layout(pad=0.3, w_pad=1.0)
save(fig, "withdata")


# ---------------------------------------------------------------------------
# Figure 9: the cost of a hidden rank against agent accuracy
# ---------------------------------------------------------------------------
import run_set_scaling as scaling_mod

scal = load("results/run_set_scaling.json")
model_of = lambda name: scaling_mod.label(name).split(", ")[1]
sets = sorted(scal["run_sets"].items(), key=lambda kv: (kv[1]["within_5pct"], kv[0]))
models = list(dict.fromkeys(model_of(n) for n, _ in sets))
MARKERS = dict(zip(models, "osD^vPX*"))
RELEASE = {"v1.0": RUST, "v1.5": BLUE}
fig = plt.figure(figsize=(WIDTH, 2.0))
axes = [fig.add_axes([0.075, 0.2, 0.31, 0.68]), fig.add_axes([0.47, 0.2, 0.31, 0.68])]
# each panel's correlation is printed in a corner its points leave empty
PANELS = (("all", "all on within_5pct", "(a) Over all numeric items", "$U-P$ (points)", (0.97, 0.95, "right", "top")),
          ("moved_per_miss", "moved_per_miss on within_5pct", "(b) Moved keys, per miss", "$U-P$ per miss (points)",
           (0.03, 0.04, "left", "bottom")))
for ax, (field, trend, heading, ylabel, (tx, ty, ha, va)) in zip(axes, PANELS):
    for name, r in sets:
        ax.plot(r["within_5pct"], r[field], MARKERS[model_of(name)], ms=3.4, mew=0.8,
                color=RELEASE[name.split("|")[0].replace(" new", "")], mfc="none")
    t = scal["trend"][trend]
    p = "p<0.001" if t["p_value"] < 0.001 else f"p={t['p_value']:.3f}"
    ax.text(tx, ty, f"$\\rho={t['spearman_rho']:+.2f}$, {p}", transform=ax.transAxes, ha=ha, va=va, fontsize=SMALL)
    ax.axhline(0, color=PALE, lw=0.8, zorder=0)
    ax.set_xlabel("answers within 5% of the key (%)")
    ax.set_ylabel(ylabel)
    title(ax, heading)
    spines(ax)
handles = [Line2D([], [], marker=MARKERS[m], ls="", ms=3.4, mew=0.8, mfc="none", color=GREY, label=m)
           for m in models]
handles += [Line2D([], [], marker="s", ls="", ms=4, color=c, label=f"{r} runs") for r, c in RELEASE.items()]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.8, 0.53), frameon=False, handlelength=1.0,
           fontsize=SMALL)
save(fig, "scaling")
