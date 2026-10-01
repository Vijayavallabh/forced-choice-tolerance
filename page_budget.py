#!/usr/bin/env python3
r"""Where the main text ends, measured rather than eyeballed.

AgenticLS Track II allows nine pages excluding references and appendices, and
this paper has run against that limit all the way through -- every structural
decision in it (which table is promoted, which strand goes to an appendix) was
a page-budget decision first. The limit was nonetheless checked by hand, with a
throwaway script, which is how a submission ends up a quarter page over on the
morning it is due.

The rule: ``References`` must be the first line of the page after the last page
of main text, and the main text may occupy at most ``--limit`` pages. Reading
the boundary is exact; estimating a fraction of a page from text coordinates is
not, and an earlier version of this measurement reported ``8.91 pages'' for a
paper whose main text ended at the foot of page 9, because it found the word
``References'' inside a bibliography entry and divided by a page height.

    python3 page_budget.py                  # build/main.pdf, 9 pages
    python3 page_budget.py --pdf x.pdf --limit 8
"""
from __future__ import annotations

import argparse
import sys


def boundary(pdf_path):
    """(last main-text page, the page References heads, lines left on it).

    ``References`` heads a page when it is the first line of that page's text
    in reading order. Anything else -- the word inside a title, a heading part
    way down -- is not the boundary and is not treated as one.
    """
    import pypdf

    reader = pypdf.PdfReader(pdf_path)
    for index, page in enumerate(reader.pages):
        text = page.extract_text()
        first = next((line for line in text.splitlines() if line.strip()), "")
        # the template prints a line number after the heading: "References436"
        if first.strip().rstrip("0123456789") == "References":
            return index, index + 1, len(reader.pages)
    return None, None, len(reader.pages)


def overshoot(pdf_path):
    """How far past a page top the References heading sits, in lines.

    When the heading is part way down a page the budget is over by exactly the
    main-text lines above it, which is the number to cut. Reported so that
    closing the budget is a measurement rather than a guess.
    """
    import pypdf

    reader = pypdf.PdfReader(pdf_path)
    for index, page in enumerate(reader.pages):
        lines = [l for l in (page.extract_text() or "").splitlines() if l.strip()]
        for offset, line in enumerate(lines):
            if line.strip().rstrip("0123456789") == "References":
                return index + 1, offset, lines[:offset]
    return None, None, []


def main_text_end(pdf_path):
    """(last page holding main text, the page References starts on).

    References either heads the page after the main text or follows it part
    way down the main text's last page, and both are within a budget when that
    page is. The validator once accepted only the first, so a paper with room
    to spare on page 9 failed its own page check.
    """
    last, refs, _ = boundary(pdf_path)
    if last is not None:
        return last, refs
    page, _, _ = overshoot(pdf_path)
    return page, page


def page_lines(pdf_path, before):
    """Text lines on the fullest main-text page before ``before``, for scale."""
    import pypdf

    reader = pypdf.PdfReader(pdf_path)
    counts = [len([l for l in (reader.pages[i].extract_text() or "").splitlines() if l.strip()])
              for i in range(max(0, before - 3), before)]
    return max(counts) if counts else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default="build/main.pdf")
    ap.add_argument("--limit", type=int, default=9,
                    help="pages of main text the venue allows")
    args = ap.parse_args()

    main_pages, refs_page, total = boundary(args.pdf)
    if main_pages is None:
        page, offset, spill = overshoot(args.pdf)
        if page is None:
            print(f"{args.pdf}: no References heading found at all; the "
                  "boundary cannot be read, so the budget is unchecked")
            return 2
        # A heading part way down a page is only an overshoot when that page is
        # past the limit. On the last allowed page it means the main text ends
        # there with room below it -- the first version reported that as "over
        # by 46 lines", which read literally asks for 46 lines to be cut from a
        # paper that was under the limit.
        if page > args.limit:
            print(f"{args.pdf}: {total} pages, References sits {offset} lines down "
                  f"page {page}, so the main text is over by {offset} lines")
            for line in spill:
                print(f"    | {line[:76]}")
            return 1
        full = page_lines(args.pdf, page - 1)
        print(f"{args.pdf}: {total} pages, main text ends {offset} lines down page "
              f"{page}, References follows on the same page, limit {args.limit}")
        if page < args.limit:
            print(f"under by {args.limit - page} page(s) and {full - offset} lines")
        else:
            print(f"within the limit, about {full - offset} lines to spare on page {page} "
                  f"(a full page holds about {full})")
        return 0
    print(f"{args.pdf}: {total} pages, main text ends on page {main_pages}, "
          f"References heads page {refs_page}, limit {args.limit}")
    if main_pages > args.limit:
        print(f"OVER by {main_pages - args.limit} page(s)")
        return 1
    if main_pages < args.limit:
        print(f"under by {args.limit - main_pages} page(s) -- room for more, "
              "which at this limit has always meant something was cut too hard")
    else:
        print("exactly at the limit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
