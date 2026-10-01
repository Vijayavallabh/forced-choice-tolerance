#!/usr/bin/env python3
r"""When a forced MCQ grader selects no option: gpt-4o's gradings of BixBench's published v1.0 runs.

BixBench's forced grading tells the grader to pick one option, and its parser (``xml_extract``) scores a
reply without a single letter as no pick, which is graded wrong. gpt-4o 2024-11-20, regrading the published
runs through $R$, $P$ and $U$ (``published_reads.py --reader gpt-4o``), gives no letter on about half of its
reads. This asks what those reads are:

* how often it selects no option, by what it grades: a correct answer (the submitted number within 5% of the
  key), or a miss whose number is nearest the key, nearest another option, or has no number (the number
  the tolerance reads, the last in the answer); the published forced grades of the same runs, by the gpt-4o
  and Claude 3.5 Sonnet of 2024, beside it;
* what the replies without a letter say, from the recorded replies where they are present (they are not
  shipped): an empty answer or text with no digit, a number, or no answer tag at all.

Counts are pooled over reads, through the released options, on the random quarter of the published runs
that the grader read in full (``published_reads.SAMPLE_CHUNKS``).

    python3 grader_declines.py        # results/grader_declines.json
"""
import collections
import gzip
import json
import pathlib
import re

import answer_extraction as ax
import bixbench_withdata as bw
import published_reads as pr
import replication as rp
import score_decomposition as sd
from answer_numbers import graded

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "grader_declines.json"
READER = "gpt-4o"
CACHES = pr.work_dir(READER, "forced")
KINDS = ("correct", "miss", "miss, nearest the key", "miss, nearest another option", "miss, no number")


def kinds_of(answer, options):
    """The kinds of answer a run's answer is, among KINDS."""
    if graded(answer, options[0], sd.TOL):
        return ("correct",)
    if ax.category(answer) == "no number":
        return ("miss", "miss, no number")
    ranks = sd.nearest_ranks_last(answer, options)
    if ranks is None or len(ranks) != 1:
        return ("miss",)
    key = bw.pick_rank(options[0], options)
    return ("miss", "miss, nearest the key" if ranks == {key} else "miss, nearest another option")


def in_quarter(r):
    return pr.chunk_of(r, pr.SAMPLE_OF) in pr.SAMPLE_CHUNKS


def no_picks():
    """No-pick shares by kind of answer: gpt-4o 2024-11-20's reads through R, and the published forced
    grades of the same runs."""
    sets = rp.v10_sets()
    tally = {"gpt-4o 2024-11-20": collections.defaultdict(lambda: [0, 0]),
             "published, gpt-4o": collections.defaultdict(lambda: [0, 0]),
             "published, Claude 3.5 Sonnet": collections.defaultdict(lambda: [0, 0])}
    for r in pr.load_rows(pr.rows_path(READER, "forced")):
        if r["set"] != "D1" or not in_quarter(r) or not (r["answer"] and str(r["answer"]).strip()):
            continue
        options = sets[r["q"]]["released"]
        who = "published, gpt-4o" if r["run"].startswith("4o") else "published, Claude 3.5 Sonnet"
        for kind in kinds_of(r["answer"], options):
            for x in r["reads"].get("released", []):
                tally["gpt-4o 2024-11-20"][kind][0] += bool(x["no_pick"])
                tally["gpt-4o 2024-11-20"][kind][1] += 1
            tally[who][kind][0] += r["published"]["picked"] is None
            tally[who][kind][1] += 1
    return {who: {k: {"no_pick": 100.0 * t[k][0] / t[k][1] if t[k][1] else None, "n_reads": t[k][1]}
                  for k in KINDS} for who, t in tally.items()}


# a reply's reasoning saying that the answer matches none of the options
NO_MATCH = re.compile(r"none of the|no option|not among|does not match|do not match|none match|not present|no correct",
                      re.I)


def replies():
    """What the recorded replies without a letter hold, when the recorded replies are present: an empty answer or
    a word (None, N/A and the like), a number, or no answer tag; and how many say that no option matches."""
    paths = sorted(CACHES.glob("cache_*.jsonl"))
    if not paths:
        return None
    c = collections.Counter()
    for path in paths:
        for line in path.open():
            reply = json.loads(line)["reply"]
            if bw.xml_extract(reply) != "Z":
                c["letter"] += 1
                continue
            c["says no option matches"] += bool(NO_MATCH.search(reply))
            m = re.search(r"<answer>(.*?)</answer>", reply, re.S)
            if m is None:
                c["no answer tag"] += 1
            elif re.search(r"\d", m.group(1)):
                c["a number"] += 1
            else:
                c["empty or text"] += 1
    no_letter = sum(c[k] for k in ("empty or text", "a number", "no answer tag"))
    return {"n_replies": c["letter"] + no_letter, "no_letter": no_letter,
            **{k: 100.0 * c[k] / no_letter for k in ("empty or text", "a number", "no answer tag",
                                                       "says no option matches")}}


def main():
    report = {"reader": READER, "sample": f"chunks {min(pr.SAMPLE_CHUNKS)}-{max(pr.SAMPLE_CHUNKS)} of {pr.SAMPLE_OF}",
              "no_pick": no_picks()}
    got = replies()
    if got is None and OUT.exists():
        got = json.loads(OUT.read_text()).get("replies")
    report["replies"] = got
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    for who, kinds in report["no_pick"].items():
        print(who, {k: f"{v['no_pick']:.1f}% of {v['n_reads']}" for k, v in kinds.items() if v["no_pick"] is not None})
    print("replies without a letter:", report["replies"])
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
