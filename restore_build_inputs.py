#!/usr/bin/env python3
r"""Put back the build/ inputs that a fresh clone does not have.

The analyses, the tests and the validator open item and option files under
``build/``, which git ignores. A clone therefore lacks them: the tests fail or
skip and the validator stops at the first one. ``data/derived/`` carries them,
with the SHA-256 of every file in ``data/derived/SHA256SUMS``. This script checks
every checksum and then writes each file back under ``build/`` at its own
relative path, so no analysis path changes:

    python restore_build_inputs.py            # verify, then copy into build/
    python restore_build_inputs.py --check    # verify only; write nothing
    python restore_build_inputs.py --symlink  # link the plain files instead

Files over 1 MiB ship gzipped (their name plus ``.gz``) and are decompressed on
the way back. ``cd data/derived && sha256sum -c SHA256SUMS`` checks the shipped
files without Python.

Two kinds of file are also rebuilt here, and each must come out byte-identical:

* ``build/bixbench_numeric_q.jsonl`` is the numeric subset of
  ``data/bixbench.jsonl`` that ``channel_survey.numeric_ranks`` defines: the rows
  whose key and distractors all parse as distinct numbers, in file order, as
  ``{"ideal", "distractors", "question", "cluster"}``. It also ships, and the
  rebuilt bytes must equal the shipped ones.
* ``build/{mmlu_pro,mmlu,medmcqa,aqua_rat}.jsonl`` are the survey's option sets
  in the audit's row shape, which ``survey_to_jsonl.convert`` writes from
  ``data/external/survey/``. They would add 14 MB to ``data/derived/``; they are
  rebuilt in about a second and checked against the SHA-256 recorded below.

``build/`` files that already hold the right bytes are left alone. A file that
holds different bytes is reported and left alone, and the script exits
non-zero. ``--force`` replaces it.

With ``--symlink``, a script that rebuilds one of these files writes through
the link into ``data/derived/``. The checksums catch that, but copying is the
default for this reason.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DERIVED = ROOT / "data" / "derived"
SUMS = DERIVED / "SHA256SUMS"
BUILD = ROOT / "build"

NUMERIC_Q = "bixbench_numeric_q.jsonl"

# The survey conversions, rebuilt rather than shipped, with the SHA-256 of the
# files the results were computed from.
SURVEY = {
    "mmlu_pro.jsonl": ("mmlu_pro", "7dedf93d267b87353abbbc60d86a0f78236520e55c622649467868ad40076733"),
    "mmlu.jsonl": ("mmlu", "c36c71d07f222a35906cd02c0260d070a2895ad0f8c3d5fbd58f293c8a08fbc1"),
    "medmcqa.jsonl": ("medmcqa", "fc182adae82703a2bf6cc8f8153ea404fa8812145a0b421df4f1ad79e734261d"),
    "aqua_rat.jsonl": ("aqua_rat", "f4791c33579de62517a78cf66bae5cddd3539c35ac49980de3d1a8426a858082"),
}


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def manifest(sums: Path = SUMS) -> dict[str, str]:
    """SHA256SUMS as {path relative to data/derived: sha256}."""
    entries = {}
    for number, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        digest, sep, name = line.partition("  ")
        if not sep or len(digest) != 64:
            raise ValueError(f"{sums}:{number}: not a sha256sum line: {line!r}")
        entries[name] = digest
    return entries


def restored_name(name: str) -> str:
    """Where a shipped file goes under build/: its own path, less any .gz."""
    return name[:-3] if name.endswith(".gz") else name


def verify(derived: Path = DERIVED) -> list[str]:
    """Every problem with data/derived: missing, altered or unlisted files."""
    listed = manifest(derived / "SHA256SUMS")
    problems = []
    for name, digest in sorted(listed.items()):
        path = derived / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif sha256(path.read_bytes()) != digest:
            problems.append(f"checksum differs: {name}")
    for path in sorted(derived.rglob("*")):
        name = path.relative_to(derived).as_posix()
        if path.is_file() and name != "SHA256SUMS" and name not in listed:
            problems.append(f"not in SHA256SUMS: {name}")
    targets = [restored_name(n) for n in listed]
    if len(set(targets)) != len(targets):
        problems.append("two shipped files restore to the same build/ path")
    return problems


def payload(name: str, derived: Path = DERIVED) -> bytes:
    """The bytes a shipped file restores to."""
    blob = (derived / name).read_bytes()
    return gzip.decompress(blob) if name.endswith(".gz") else blob


def numeric_q_bytes(bixbench: Path = ROOT / "data" / "bixbench.jsonl") -> bytes:
    """build/bixbench_numeric_q.jsonl, rebuilt from data/bixbench.jsonl.

    channel_survey's numeric subset: every row whose key and distractors all
    parse as distinct numbers, in file order, with the four fields the audit
    reads, one json.dumps line each.
    """
    sys.path.insert(0, str(ROOT))
    try:
        from channel_survey import numeric_ranks
    finally:
        sys.path.remove(str(ROOT))
    lines = []
    with open(bixbench, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if numeric_ranks([row["ideal"], *row["distractors"]]) is None:
                continue
            lines.append(json.dumps({"ideal": row["ideal"], "distractors": row["distractors"],
                                     "question": row["question"],
                                     "cluster": row["capsule_uuid"]}) + "\n")
    return "".join(lines).encode("utf-8")


def survey_bytes(survey: str) -> bytes:
    """One survey option set in the audit's row shape, as survey_to_jsonl.py writes it."""
    sys.path.insert(0, str(ROOT))
    try:
        import survey_to_jsonl
    finally:
        sys.path.remove(str(ROOT))
    path = ROOT / "data" / "external" / "survey" / f"{survey}.jsonl.gz"
    return "".join(json.dumps(row) + "\n" for row in survey_to_jsonl.convert(path)).encode("utf-8")


def plan(derived: Path = DERIVED) -> dict[str, tuple[bytes, str | None]]:
    """{path under build/: (bytes, the shipped file it comes from or None)}.

    Raises SystemExit if data/derived fails its checksums, if the numeric subset
    no longer regenerates, or if a survey conversion comes out different.
    """
    problems = verify(derived)
    if problems:
        raise SystemExit(f"{derived} does not match its SHA256SUMS:\n  " + "\n  ".join(problems))
    out = {restored_name(name): (payload(name, derived), name)
           for name in manifest(derived / "SHA256SUMS")}
    if NUMERIC_Q not in out:
        raise SystemExit(f"data/derived/{NUMERIC_Q} is not in SHA256SUMS")
    if numeric_q_bytes() != out[NUMERIC_Q][0]:
        raise SystemExit(f"{NUMERIC_Q} no longer regenerates from data/bixbench.jsonl: "
                         "the shipped copy and the rebuilt one differ")
    for target, (survey, digest) in SURVEY.items():
        blob = survey_bytes(survey)
        if sha256(blob) != digest:
            raise SystemExit(f"build/{target} rebuilt from data/external/survey/{survey}.jsonl.gz "
                             f"has SHA-256 {sha256(blob)}, not {digest}")
        out[target] = (blob, None)
    return out


def restore(build: Path = BUILD, derived: Path = DERIVED, symlink: bool = False,
            force: bool = False) -> dict[str, list[str]]:
    """Write every planned file under ``build``. Returns what happened to each."""
    report = {"written": [], "present": [], "conflict": []}
    for target, (blob, source) in sorted(plan(derived).items()):
        path = build / target
        if path.exists() or path.is_symlink():
            same = path.exists() and path.is_file() and sha256(path.read_bytes()) == sha256(blob)
            if same:
                report["present"].append(target)
                continue
            if not force:
                report["conflict"].append(target)
                continue
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        if symlink and source is not None and not source.endswith(".gz"):
            os.symlink(os.path.relpath(derived / source, path.parent), path)
        else:
            path.write_bytes(blob)
        report["written"].append(target)
    return report


BIG = 1 << 20  # files larger than this ship gzipped


def refresh(build: Path = BUILD, derived: Path = DERIVED) -> list[str]:
    """Copy the listed files from ``build`` into ``derived`` and rewrite SHA256SUMS.

    For the maintainer, after an input legitimately changes (a fixed generator
    rewrites a file under build/). The set of files is the one SHA256SUMS
    already lists; adding a file means adding its line first.
    """
    lines, changed = [], []
    for name in sorted(manifest(derived / "SHA256SUMS")):
        source = build / restored_name(name)
        if not source.is_file():
            raise SystemExit(f"cannot refresh {name}: {source} does not exist")
        blob = source.read_bytes()
        stored = restored_name(name) + (".gz" if len(blob) > BIG else "")
        data = gzip.compress(blob, compresslevel=9, mtime=0) if stored.endswith(".gz") else blob
        target = derived / stored
        if not target.is_file() or target.read_bytes() != data:
            if not (target.is_file() and stored.endswith(".gz")
                    and gzip.decompress(target.read_bytes()) == blob):
                changed.append(stored)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        if stored != name:
            (derived / name).unlink(missing_ok=True)
        lines.append((stored, sha256(target.read_bytes())))
    (derived / "SHA256SUMS").write_text("".join(f"{digest}  {stored}\n"
                                                for stored, digest in sorted(lines)))
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify data/derived and the rebuilt files; write nothing")
    ap.add_argument("--symlink", action="store_true",
                    help="link the plain files into build/ instead of copying them")
    ap.add_argument("--force", action="store_true",
                    help="replace build/ files that hold different bytes")
    ap.add_argument("--refresh", action="store_true",
                    help="maintainer: copy the listed files from build/ into data/derived "
                         "and rewrite SHA256SUMS")
    ap.add_argument("--build", type=Path, default=BUILD, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.refresh:
        changed = refresh(args.build)
        print(f"data/derived refreshed from {args.build}: {len(changed)} file(s) changed"
              + ("".join(f"\n  {name}" for name in changed)))
        return 0

    if args.check:
        planned = plan()
        print(f"data/derived: {len(manifest())} files match SHA256SUMS; "
              f"{NUMERIC_Q} regenerates byte-identically from data/bixbench.jsonl; "
              f"{len(SURVEY)} survey files rebuild to their recorded SHA-256 "
              f"({len(planned)} build/ files in all)")
        return 0
    report = restore(args.build, symlink=args.symlink, force=args.force)
    print(f"{len(report['written'])} written, {len(report['present'])} already in place, "
          f"under {args.build}")
    if report["conflict"]:
        print("left alone because they hold different bytes (rerun with --force to "
              "replace them):\n  " + "\n  ".join(report["conflict"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
