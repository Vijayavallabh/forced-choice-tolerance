#!/usr/bin/env python3
r"""Re-attach MMLU's question text to the option sets the survey vendored.

``fetch_survey.py`` deliberately drops question text, which is right for a
survey that only asks where the key sits in the sorted order. The repair
frontier needs it: a repair that conditions on the question is the one
\S\ref{sec:frontier} tests, and a solver that is *given* the question is how
the cost of a repair is measured.

Alignment is by content, not by row order: each vendored item is matched to the
upstream item with the same cluster, the same keyed option and the same
multiset of distractors. An item whose option set occurs more than once in its
own subject is dropped rather than guessed at, and the count is reported.

    python3 fetch_mmlu_questions.py --out build/mmlu_questions.jsonl
"""
import argparse
import io
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

API = "https://huggingface.co/api/datasets/cais/mmlu/parquet/all/test"


def get(url, timeout=300):
    request = urllib.request.Request(url, headers={"User-Agent": "option-geometry-survey/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="build/mmlu.jsonl")
    ap.add_argument("--out", default="build/mmlu_questions.jsonl")
    ap.add_argument("--report", default="results/mmlu_questions.json")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    urls = json.loads(get(API).decode("utf-8"))
    upstream = []
    for url in urls:
        table = pq.read_table(io.BytesIO(get(url)))
        cols = table.to_pydict()
        for question, choices, answer, subject in zip(
                cols["question"], cols["choices"], cols["answer"], cols["subject"]):
            index = int(answer)
            options = [str(o) for o in choices]
            if not 0 <= index < len(options):
                continue
            upstream.append({
                "question": str(question),
                "key": options[index].strip(),
                "distractors": tuple(sorted(o.strip() for i, o in enumerate(options) if i != index)),
                "cluster": str(subject),
            })
    print(f"{len(upstream)} upstream MMLU test items")

    # Index by content. A signature occurring twice is ambiguous and is dropped.
    index = defaultdict(list)
    for row in upstream:
        index[(row["cluster"], row["key"], row["distractors"])].append(row["question"])

    out, matched, ambiguous, missing = [], 0, 0, 0
    for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        signature = (row.get("cluster", ""), str(row["ideal"]).strip(),
                     tuple(sorted(str(d).strip() for d in row["distractors"])))
        hits = index.get(signature, [])
        if len(hits) == 1:
            question, matched = hits[0], matched + 1
        elif len(hits) > 1:
            question, ambiguous = "", ambiguous + 1
        else:
            question, missing = "", missing + 1
        out.append({**row, "question": question})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    report = {"n_rows": len(out), "n_matched": matched,
              "n_ambiguous": ambiguous, "n_missing": missing,
              "source": API, "n_upstream": len(upstream)}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: matched {matched}, ambiguous {ambiguous}, missing {missing}")


if __name__ == "__main__":
    main()
