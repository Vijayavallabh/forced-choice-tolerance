#!/usr/bin/env python3
"""Generate the manuscript's figures from the shipped results.

Every number drawn here is read from a file in ``results/``; nothing is typed
into this script. ``validate_artifact.py`` re-reads the same files and checks
them against the numbers typed in ``main.tex``, so a figure and the text cannot
disagree without one of the two failing.

One command, ``python3 make_figures.py`` from any directory, writes every figure:

    figures/overview.{pdf,png}       Figure 1 as the paper prints it: panels (a)-(b)
    figures/overview_full.{pdf,png}  Figure 1 with all four panels, for the poster
    figures/withdata.{pdf,png}       Figure 2

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

BLUE, GREEN, RUST, GREY = "#35618d", "#327969", "#925738", "#6b6b6b"
PURPLE, PALE = "#7a4d8c", "#d9d9d9"
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


def panel_c(ax):
    """(c) how much of the nearest-option rule's cost of hiding the rank each grading bears: its gain
    on the moved keys as a share of the rule's on the same answers, against its proximity weight
    (Remark 1); code-free graders filled, BixBench's graders with the notebook open, the "none
    within 5%" option a cross"""
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
    ax.plot([], [], "x", color="black", ms=3.6, label="none within 5%")
    # 5.9 pt: at 6.5 its last entry runs into the highest code-free grading (poster only)
    ax.legend(frameon=False, loc="upper left", handlelength=0.8, borderaxespad=0.0, fontsize=5.9)
    title(ax, "(c) The cost a grader bears")
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
# Figure 2: with the data, the options decide the reading
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
