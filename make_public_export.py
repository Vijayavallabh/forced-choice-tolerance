#!/usr/bin/env python3
r"""Build the public repository's tree from an explicit allowlist, and scan it.

The development repository stays private: it is the provenance record, and
its history narrates drafts and holds internal documents. The public repository
starts from a clean tree that this script writes into an empty directory:

    python make_public_export.py /path/to/empty/dir

What goes in is listed, not left over: the analysis code (the top-level ``*.py``
and ``*.sh`` files, less the internal ones), ``tests/``, ``data/`` with
``data/derived/``, ``results/``, ``figures/``, ``sources/`` (less the NeurIPS
author kit and an internal source ledger), ``sandbox_image/``, ``provenance/``,
the paper's source, the documentation, ``PREREGISTRATION.md`` byte for byte, the
requirements files, the licenses and the notices. Any other top-level file,
the internal notes and audits among them, stays out because it is not listed.
Files are taken from the working tree, tracked or not, and anything git ignores
stays out; the run says how many of them differ from the last commit. With
``--committed`` they are taken from ``HEAD`` instead, so the public tree is
exactly a commit of this one.

The copy is then scanned, ``.gz`` members decompressed, and the script exits
non-zero if it finds any of these:

* secrets: API-key and token shapes, private keys, Azure resource hosts other
  than the test fixtures', and the keys and endpoints configured on this host
  under ``~/.config/agenticls/`` (read to compare, never printed);
* absolute local paths of the machines this work ran on: the checkout's own
  location, this host's home and the checkout's volume, the home and volume
  paths in ``HOST_PATH_HASHES`` (stored as SHA-256, so this file does not name
  them), and per-session scratch directories under ``/tmp``;
* narration of a pre-publication assessment round, in the text this repository
  writes (code, docs, LaTeX, and the JSON results the analyses write): the words
  and tags ``NARRATION`` below matches.

Other ``/home/<name>``, ``/mnt/<name>``, ``/Users/<name>`` and ``/root/`` paths
(agents in the notebook sandbox invent their own), mentions of the files the
export leaves out, e-mail addresses and the usually harmless words in
``WARN_WORDS`` are listed as warnings and do not fail the run. Nothing here talks
to GitHub.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- what goes in ------------------------------------------------------------

INCLUDE_FILES = {
    "main.tex", "refs.bib", "neurips_2026.sty",
    "README.md", "VERIFICATION.md", "PREREGISTRATION.md",
    "LICENSE", "LICENSE-DATA", "THIRD_PARTY_NOTICES.md", "CITATION.cff", "MODEL_OUTPUTS.md",
    "requirements.txt", "requirements-gpu.txt", "requirements-fetch.txt",
    # the manifest schema the validator loads
    "temporal_manifest.schema.json",
}
INCLUDE_DIRS = ("tests/", "data/", "results/", "figures/", "sources/", "sandbox_image/",
                "provenance/")
# Top-level scripts that exist only to maintain the internal submission notes.
INTERNAL_SCRIPTS = {"submission_text.py"}
# Top-level files are allowlisted above, so the internal notes and audits need no
# entry here. Inside the included directories, these stay out:
EXCLUDE_FILES = {"sources/verification.md", "sources/neurips2026.zip"}
EXCLUDE_PREFIXES = ("archive/", "submission/", "build/", "logs/", "claude-only/", ".claude/")
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}

# written into the export, which has no .gitignore of its own to copy
GITIGNORE = """\
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.venv/

# Inputs restored by restore_build_inputs.py, analysis scratch and LaTeX output
build/

# Per-host run logs
logs/

# Submission bundles, rebuilt by package_submission.py
submission/*.zip
"""


def candidates(root: Path = ROOT) -> list[str]:
    """Tracked files plus untracked ones git does not ignore, as posix paths."""
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                         cwd=root, capture_output=True, check=True).stdout
    names = sorted({n for n in out.decode("utf-8").split("\0") if n})
    return [n for n in names if (root / n).is_file()]


def selected(name: str) -> bool:
    parts = name.split("/")
    if EXCLUDE_PARTS & set(parts) or name in EXCLUDE_FILES or name.startswith(EXCLUDE_PREFIXES):
        return False
    if name.endswith(".zip") and name.startswith("submission/"):
        return False
    if len(parts) == 1:
        if name in INCLUDE_FILES:
            return True
        return name.endswith((".py", ".sh")) and name not in INTERNAL_SCRIPTS
    return name.startswith(INCLUDE_DIRS)


def committed(root: Path = ROOT) -> list[str]:
    """The files in HEAD, as posix paths."""
    out = subprocess.run(["git", "ls-tree", "-r", "-z", "--name-only", "HEAD"], cwd=root,
                         capture_output=True, check=True).stdout
    return sorted(n for n in out.decode("utf-8").split("\0") if n)


def uncommitted(root: Path = ROOT) -> set[str]:
    """Paths whose working copy differs from HEAD, or that HEAD does not have."""
    out = subprocess.run(["git", "status", "--porcelain", "-z", "--untracked-files=all"],
                         cwd=root, capture_output=True, check=True).stdout
    entries, names, i = out.decode("utf-8").split("\0"), set(), 0
    while i < len(entries):
        entry = entries[i]
        if entry:
            names.add(entry[3:])
            if entry[0] in "RC":  # a rename or copy is followed by its source path
                i += 1
        i += 1
    return names


def export(dest: Path, root: Path = ROOT, from_head: bool = False) -> list[str]:
    """Copy the allowlisted files into ``dest``, which must be empty or absent.

    From the working tree by default; from HEAD with ``from_head``.
    """
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} is not empty; the export writes only into an empty directory")
    names = [n for n in (committed(root) if from_head else candidates(root)) if selected(n)]
    missing = sorted(f for f in INCLUDE_FILES if f not in names)
    if missing:
        raise SystemExit(f"refusing to export: allowlisted files are missing: {missing}")
    if from_head:
        dest.mkdir(parents=True, exist_ok=True)
        with subprocess.Popen(["git", "archive", "--format=tar", "HEAD", "--", *names], cwd=root,
                              stdout=subprocess.PIPE) as proc:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                tar.extractall(dest, filter="data")
        if proc.returncode:
            raise SystemExit("git archive failed; the export is incomplete")
    else:
        for name in names:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target)
    (dest / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    # The registered plan's hash is pinned; the public copy must be the committed bytes.
    head = subprocess.run(["git", "show", "HEAD:PREREGISTRATION.md"], cwd=root,
                          capture_output=True, check=True).stdout
    if (dest / "PREREGISTRATION.md").read_bytes() != head:
        raise SystemExit("PREREGISTRATION.md differs from the committed copy; its hash is pinned")
    return names


# --- the scan ------------------------------------------------------------------

def _rx(*pieces: str, flags: int = 0) -> re.Pattern:
    # Patterns are assembled from pieces so this file does not match its own scan.
    return re.compile("".join(pieces), flags)


SECRET_PATTERNS = {
    "OpenAI-style key": _rx(r"\bs", r"k-(?:proj-|ant-|svcacct-)?[A-Za-z0-9_\-]{20,}"),
    "Hugging Face token": _rx(r"\bh", r"f_[A-Za-z0-9]{30,}"),
    "GitHub token": _rx(r"\b(?:gh", r"p|gho|ghs|ghu|ghr)_[A-Za-z0-9]{30,}|\bgithub", r"_pat_[A-Za-z0-9_]{20,}"),
    "AWS key id": _rx(r"\bAK", r"IA[0-9A-Z]{16}\b"),
    "Google API key": _rx(r"\bAI", r"za[0-9A-Za-z_\-]{35}\b"),
    "Slack token": _rx(r"\bxo", r"x[abprs]-[A-Za-z0-9\-]{10,}"),
    "private key": _rx(r"-----BEGIN [A-Z ]*PRI", r"VATE KEY-----"),
    "bearer token": _rx(r"\bBea", r"rer\s+([A-Za-z0-9._\-]{20,})"),
    "key assignment": _rx(r"(?i)(?:api[_-]?key|subscription[_-]?key|access[_-]?token)[\"']?\s*[:=]\s*",
                          r"[\"']([A-Za-z0-9_\-]{16,})[\"']"),
    "OpenAI org or project id": _rx(r"\b(?:or", r"g-|pro", r"j_)[A-Za-z0-9]{24}\b"),
    "Azure resource host": _rx(r"(?i)\b([a-z0-9][a-z0-9\-]*)\.(?:openai\.azure\.com|services\.ai\.azure\.com"
                               r"|cognitiveservices\.azure\.com|inference\.ai\.azure\.com"
                               r"|models\.ai\.azure\.com|azure-api\.net)"),
}
# Lower-case substrings at least one of which every match of each pattern contains.
TRIGGERS = {
    "OpenAI-style key": ("sk-",), "Hugging Face token": ("hf_",),
    "GitHub token": ("ghp_", "gho_", "ghs_", "ghu_", "ghr_", "github_pat_"),
    "AWS key id": ("akia",), "Google API key": ("aiza",), "Slack token": ("xox",),
    "private key": ("private key",), "bearer token": ("bearer",),
    "key assignment": ("key", "token"), "OpenAI org or project id": ("org-", "proj_"),
    "Azure resource host": (".azure.com", "azure-api.net"),
}
PATH_TRIGGERS = ("/home/", "/mnt/", "/Users/", "/root/", "/tmp/cla" + "ude-")
# Values that tests/test_openai_api.py and openai_api.py's docstring use as stand-ins.
FIXTURE_VALUES = {"azure-key-0123456789", "azure-from-env", "sk-from-file"}
FIXTURE_HOSTS = {"myres", "name", "evil", "fromenv", "fromfile", "proj", "x", "your-resource"}

# Paths that belong to other machines and are not ours: the notebook sandbox's
# home and data mount, and LAB-Bench's CI runner in its figure paths.
BENIGN_PATHS = {"/home/runner", "/home/user", "/mnt/data"}
LOCAL_PATH = _rx(r"(?<![\w.\-])(?:/(?:home|mnt|Users)/[A-Za-z0-9_.\-]+|/root/[A-Za-z0-9_.\-]+",
                 r"|/tmp/cla", r"ude-[A-Za-z0-9_.\-]*)")
# The home and volume paths of the machines this work ran on, as the SHA-256 of
# their first two components ("/<top>/<name>"), so that this file, which ships,
# does not name them. This host's home and the checkout's volume are added at run
# time (host_tops).
HOST_PATH_HASHES = {
    "b44f9fd7fd5f85fb3372fe46dfdad9a72694905185468850ddb3960ecda424af",
    "41847eee6e20d74cf50d7526935d051d69e338674dc44e1d2c399a7f54c03d13",
    "ed7b0526e091e64a679f5c9f77cade522f47e98b1d48266ec6a884324b4d46ef",
}


def host_tops(root: Path = ROOT) -> set[str]:
    """The first two components of this host's home and of the checkout's location."""
    tops = set()
    for path in (Path.home(), root):
        parts = path.resolve().parts
        if len(parts) >= 3:
            tops.add("/" + "/".join(parts[1:3]))
    return tops


HOST_TOPS = host_tops()


def host_path(path: str) -> bool:
    """Whether an absolute path found in a file is one of this work's machines'."""
    top = "/".join(path.split("/")[:3])
    return (path.startswith("/tmp/") or top in HOST_TOPS
            or hashlib.sha256(top.encode("utf-8")).hexdigest() in HOST_PATH_HASHES)

# Labels avoid the words they look for, so this file passes its own scan.
NARRATION = {
    "the assessors": _rx(r"(?i)\bre", r"viewers?\b"),
    "the assessments": _rx(r"(?i)\bre", r"views\b"),
    "a response letter": _rx(r"(?i)\breb", r"uttal\b"),
    "an assessor's tag": _rx(r"\bR[0-9]-?", r"Q[0-9]+\b|\b[WQ][0-9]+ in (?:one|an", r"other) review\b"),
    "a superseded text": _rx(r"(?i)\b(?:earlier|previous|prior) dr", r"afts?\b"),
    "an award": _rx(r"(?i)\bbest[- ]pa", r"per\b"),
    "an assessment round": _rx(r"(?i)\bmock re", r"view|\breview ro", r"und\b"),
}
WARN_WORDS = {
    "the final version's name": _rx(r"(?i)\bcamera[- ]re", r"ady\b"),
    "a due date": _rx(r"(?i)\bdead", r"lines?\b"),
}
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")

# Text this repository writes, where narration would be ours: code, docs, LaTeX
# and the JSON the analyses write. Model outputs (the .gz files) and third-party
# data are left out: models and benchmark authors use these words for their own
# reasons.
THIRD_PARTY = ("data/", "sources/", "sandbox_image/", "provenance/")


def own_text(name: str) -> bool:
    if name.startswith(THIRD_PARTY) or name == "refs.bib":
        return False
    if name.startswith("results/"):
        return name.endswith(".json") and name.count("/") == 1
    return name.endswith((".py", ".sh", ".md", ".tex", ".cff", ".txt", ".toml", ".yaml", ".yml"))


def internal_names(root: Path = ROOT) -> list[str]:
    """The files the export leaves out that shipped text might still point to.

    Derived from the checkout rather than listed, for the reason HOST_PATH_HASHES
    is hashed: the top-level documents that are not exported, the excluded files
    inside the exported directories, and the internal scripts and archive.
    """
    names = set(EXCLUDE_FILES) | INTERNAL_SCRIPTS | {"archive/"}
    for name in candidates(root):
        if "/" not in name and not selected(name) and name.endswith((".md", ".txt", ".tex")):
            names.add(name)
    return sorted(names)


def host_needles(root: Path = ROOT) -> dict[str, str]:
    """Strings that would identify this machine or its credentials.

    Derived, never listed: the checkout's location and the keys and endpoints
    configured for openai_api.py. Values are compared, not printed.
    """
    needles = {str(root.parent): "the checkout's location", str(root): "the checkout's location"}
    config = Path.home() / ".config" / "agenticls"
    if config.is_dir():
        for path in sorted(config.iterdir()):
            try:
                value = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                continue
            if path.suffix == ".endpoint":
                host = re.sub(r"^[a-z]+://", "", value).split("/")[0].split(".")[0]
                if len(host) >= 6:
                    needles[host] = f"the Azure resource in {path.name}"
            elif path.suffix == ".key" and len(value) >= 16:
                needles[value] = f"the API key in {path.name}"
    for extra in os.environ.get("AGENTICLS_LEAK_NEEDLES", "").split(","):
        if len(extra.strip()) >= 4:
            needles[extra.strip()] = "a needle from AGENTICLS_LEAK_NEEDLES"
    return needles


def text_of(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix == ".gz" or raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError):
            pass
    return raw.decode("utf-8", errors="ignore")


def scan_text(name: str, text: str, needles: dict[str, str] | None = None,
              narration: bool | None = None, internal: list[str] | tuple = ()
              ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(fatal, warnings) for one file's text, each a list of (category, detail).

    Details never repeat a secret: a match is described by its kind and length.
    ``internal`` names the files the export leaves out (internal_names()).
    """
    fatal, warn = [], []
    low = text.lower()
    for kind, rx in SECRET_PATTERNS.items():
        # a substring every match must contain, so a multi-gigabyte scan stays cheap
        if not any(t in low for t in TRIGGERS[kind]):
            continue
        for m in rx.finditer(text):
            value = m.group(1) if m.groups() and m.group(1) else m.group(0)
            if kind == "Azure resource host":
                if value.lower() in FIXTURE_HOSTS:
                    continue
                fatal.append(("secret", f"{kind} ({len(value)}-character resource name)"))
                continue
            if value in FIXTURE_VALUES or any(v in m.group(0) for v in FIXTURE_VALUES):
                continue
            fatal.append(("secret", f"{kind} ({len(m.group(0))} characters)"))
    for needle, what in (needles or {}).items():
        if needle in text:
            category = "secret" if "key" in what or "Azure" in what else "local path"
            fatal.append((category, what))
    # an absolute path, not a URL's path segment ("nih.gov/home/about")
    paths = LOCAL_PATH.finditer(text) if any(t in text for t in PATH_TRIGGERS) else ()
    for m in paths:
        path = m.group(0)
        if "/".join(path.split("/")[:3]) in BENIGN_PATHS:
            continue
        if host_path(path):
            fatal.append(("local path", path))
        else:
            # a path on some other machine, most often one an agent made up
            warn.append(("path", path))
    if narration is None:
        narration = own_text(name)
    if narration:
        for kind, rx in NARRATION.items():
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                fatal.append(("narration", f"line {line}: {kind}: {m.group(0)!r}"))
        for kind, rx in WARN_WORDS.items():
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                warn.append(("word", f"line {line}: {kind}"))
        for left_out in internal:
            if left_out in text:
                warn.append(("internal reference", left_out))
    for m in (EMAIL.finditer(text) if "@" in text else ()):
        address = m.group(0)
        if not address.lower().startswith("noreply@"):
            warn.append(("e-mail", address))
    return fatal, warn


def scan(dest: Path) -> tuple[dict, dict]:
    """Scan every file under ``dest``: {path: [(category, detail)]} for fatal and warnings."""
    needles, internal = host_needles(), internal_names()
    fatal, warn = defaultdict(list), defaultdict(list)
    for path in sorted(p for p in dest.rglob("*") if p.is_file()):
        name = path.relative_to(dest).as_posix()
        f, w = scan_text(name, text_of(path), needles, internal=internal)
        if f:
            fatal[name].extend(f)
        if w:
            warn[name].extend(w)
    return dict(fatal), dict(warn)


def summarise(findings: dict, limit: int = 8) -> str:
    lines = []
    for name, items in sorted(findings.items()):
        counts = defaultdict(int)
        for category, detail in items:
            counts[(category, detail)] += 1
        shown = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
        rest = len(counts) - len(shown)
        lines.append(f"  {name}")
        for (category, detail), n in shown:
            lines.append(f"      {category}: {detail}" + (f"  (x{n})" if n > 1 else ""))
        if rest > 0:
            lines.append(f"      ... and {rest} more")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", type=Path, help="an empty (or absent) directory to write the tree into")
    ap.add_argument("--report", type=Path, help="also write the findings as JSON here")
    ap.add_argument("--committed", action="store_true",
                    help="export the files as HEAD has them, not as the working tree does")
    args = ap.parse_args()

    names = export(args.dest, from_head=args.committed)
    size = sum((args.dest / n).stat().st_size for n in names)
    print(f"exported {len(names)} files, {size / 1e6:.1f} MB, to {args.dest}"
          + (" from HEAD" if args.committed else ""))
    if not args.committed:
        dirty = sorted(set(names) & uncommitted())
        if dirty:
            print(f"{len(dirty)} of them differ from HEAD or are not committed; commit them, "
                  f"or pass --committed to export HEAD")
    fatal, warn = scan(args.dest)
    if args.report:
        args.report.write_text(json.dumps({"files": len(names), "bytes": size,
                                           "fatal": fatal, "warnings": warn}, indent=1) + "\n")
    if warn:
        print(f"\n{len(warn)} file(s) with warnings (not fatal):\n{summarise(warn)}")
    if fatal:
        print(f"\nFAILED: {len(fatal)} file(s) with secrets, local paths or review narration:\n"
              f"{summarise(fatal)}", file=sys.stderr)
        return 1
    print("\nscan: no secrets, local paths or review narration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
