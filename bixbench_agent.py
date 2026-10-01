#!/usr/bin/env python3
"""BixBench's published agent, run with the capsule's data and without it.

Every no-data number in this paper reads an option set with nothing else in
view. The number BixBench's leaderboard reports is a different object: an agent
works through the capsule's data in a notebook, answers in its own words, and a
second model then reads that notebook and answer *through the four options*
(``MCQ_EVAL_PROMPT``). Whether the option geometry reaches that number is the
question every no-data result leaves open. This script produces the
trajectories it is answered on; ``bixbench_withdata.py`` reads them.

What is BixBench's, verbatim, from the vendored sources:

* the system prompt ``CAPSULE_SYSTEM_PROMPT_OPEN`` and task prompt
  ``OPEN_PROMPT_TEMPLATE`` + ``AVOID_IMAGES`` of data-analysis-crow v1.5.0, the
  version BixBench pins, formatted exactly as ``generate_trajectories.py`` does
  (``language="python"``, open-answer mode -- the agent never sees the options);
* the three tools, ``edit_cell``, ``list_workdir`` and ``submit_answer``, with
  their semantics: appending runs the new cell, editing an earlier cell reruns
  the notebook, and submitting ends the episode;
* the rollout: at most 40 steps, temperature 1.0, and ``hide_old_env_states`` --
  only the latest view of the notebook stays in the context; each reply may run to
  4,096 tokens, the limit fhlmi 0.25.2, through which ldp makes its calls, gives a
  config that sets none, as BixBench's does;
* the capsule layout: the zip's Data folder flattened into the working
  directory, its Notebook folder and any ``.ipynb`` removed, exactly as
  ``_extract_and_process_files`` does;
* the kernel's environment: fhda v1.5.0's pinned image (``sandbox_image/``).

Two protocols, chosen with ``--protocol``:

* ``react`` is the agent ``generate_trajectories.yaml`` names, ldp's ReActAgent
  at the release fhda v1.5.0 pins (0.26.0): two calls a step, fhda's tools
  passed natively, ldp's own system prompt, reasoning wrapper, stop strings and
  retries (see "the published protocol" below, and
  ``tests/test_published_protocol.py``, which checks each piece against the
  vendored upstream files).
* ``text``, the first runs': **tools called in text, not through a
  function-calling API.** The three open model families here have three
  incompatible native tool-call formats; a fenced code block (``edit_cell``),
  ``list_workdir()`` and ``submit_answer("<answer>...</answer>")`` read the same
  way for all of them. A reply that contains a code block runs that block -- one
  tool per step, as ``parallel_tool_calls: False`` has it -- and a submission in
  the same reply is not taken, so no agent answers without having seen its last
  cell's output.

What is ours, and why:

* **The notebook view has a budget.** Open models here have 32k--128k tokens of
  context where the published agents had 200k. Each cell's output is shown up to
  4,000 characters (under ``react``, fhda's own 3,000) and, when the whole view
  would pass ``--view-budget``, the oldest outputs are shortened first; the
  latest cell is always shown whole.
* **The kernel persists between cells** (``sandbox_repl.py``) instead of the
  whole notebook being re-executed after each append. For an append the state
  is the same; an edit of an earlier cell restarts the kernel and reruns every
  cell, which is what re-execution would do.
* **Every code cell runs in a container with no network**, the capsule mounted
  read-only; the agent's working directory is writable and holds links to it.
* **One driver per run, one run per episode.** A driver holds its run's
  directories under ``--out`` and ``--work`` while it lives, and one started on
  either while that driver is alive stops at once; each episode claims its output
  before it runs, and a finished result is never replaced. (Two drivers once ran
  one model at the same time, each writing every episode it finished over the
  other's.)

The *no-data* condition is the same agent, prompt and tools with an empty
working directory. Run through BixBench's own MCQ step, it is the published
pipeline's own no-data baseline.

    python3 bixbench_agent.py --model qwen72b --base http://127.0.0.1:18101 \\
        --condition data --condition nodata --concurrency 16
"""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import random
import re
import runpy
import shutil
import socket
import sys
import time
import zipfile
from pathlib import Path

import httpx

import openai_api as oa

ROOT = Path(__file__).resolve().parent
FHDA_PROMPTS = ROOT / "sources" / "bixbench_49311180" / "fhda_prompts_v1.5.0.py"
REPL = ROOT / "sandbox_repl.py"
IMAGE = os.environ.get("BIXBENCH_IMAGE", "agenticls/bixbench-env:fhda-1.5.0")
# The capsules are 5.9 GB and the work directories grow with every episode, so both
# are set per host (BIXBENCH_CAPSULES, --work) rather than named here.
CAPSULES = Path(os.environ.get("BIXBENCH_CAPSULES", str(ROOT / "build" / "bixbench-capsules")))
LANGUAGE = "python"
HIDDEN = "[Previous environment state - hidden]"
# Episodes in flight per replica. An episode stays on one replica, so its prefix
# cache holds, and takes whichever has the fewest when it gets its sandbox:
# pinned by a hash instead, the slower replica's episodes outlived the other's,
# filled the slots, and left one replica at 10% of its cache with fourteen
# requests queued on the other.
REPLICA_LOAD = {}
# Replicas whose last request failed. Least-loaded alone steered new episodes *to*
# a replica that had died (a GPU fault took one down mid-run): its episodes
# failed fast, so it always looked the least loaded, and 66 episodes failed there.
REPLICA_DOWN = set()

# The tools as data-analysis-crow documents them, restated for a text protocol.
TOOLS = """
You work in the notebook through three tools. Call exactly one tool per reply, at the end of the reply.

1. edit_cell -- write the cell's contents as one fenced code block:
```python
# your code
```
   The cell is appended to the notebook and run, and you are shown the notebook with its outputs.
   ONLY CODE CELLS ARE SUPPORTED; you may (and should) write comments in the code.
   To edit an existing cell instead of appending one, make the block's first line `# edit_cell idx=N`;
   the notebook is then rerun from the top.
   A cell whose first line is %%bash runs in bash; a cell whose first line is %%R runs in R.
2. list_workdir -- write `list_workdir()` on its own line. It recursively lists the contents of the working
   directory as a nested JSON dictionary.
3. submit_answer -- write `submit_answer("<answer>your answer</answer>")` on its own line. This submits
   your final answer and ends the episode; it may only be called once.
list_workdir() and submit_answer(...) may also be called from Python inside a code cell.
"""

CODE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\n(.*?)```", re.S)
SUBMIT_RE = re.compile(r"submit_answer\s*\(", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
EDIT_RE = re.compile(r"^\s*#\s*edit_cell\s*\(?\s*idx\s*=\s*(\d+)", re.I)
PRE_EDIT_RE = re.compile(r"^#?\s*edit_cell\s*\(?\s*idx\s*=\s*(\d+)", re.I)


def fhda_prompts():
    return runpy.run_path(str(FHDA_PROMPTS))


def task_prompt(question, prompts):
    """``generate_trajectories.py``'s ``base_prompt``, formatted as it formats it."""
    template = prompts["OPEN_PROMPT_TEMPLATE"] + "\n" + prompts["AVOID_IMAGES"]
    return template.format(question=question, language=LANGUAGE)


# ---------------------------------------------------------------- capsules

def extract_capsule(zip_path: Path, extract_dir: Path) -> Path:
    """BixBench's ``_extract_and_process_files``, minus deleting the zip."""
    done = extract_dir.with_name(extract_dir.name + ".extracted")
    if done.exists():
        return extract_dir
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    data_folder = next((p for p in extract_dir.rglob("*") if p.is_dir() and "Data" in p.name), None)
    if data_folder is None:
        raise FileNotFoundError(f"{zip_path.name}: no directory with 'Data' in its name")
    for item in data_folder.iterdir():
        shutil.move(str(item), str(extract_dir / item.name))
    shutil.rmtree(data_folder)
    notebook_folder = next((p for p in extract_dir.rglob("*") if p.is_dir() and "Notebook" in p.name),
                           None)
    if notebook_folder is not None:
        shutil.rmtree(notebook_folder)
    for ipynb in extract_dir.glob("*.ipynb"):
        ipynb.unlink()
    done.write_text(zip_path.name + "\n")
    return extract_dir


def prepare_workdir(workdir: Path, capsule: Path | None):
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    if capsule is not None:
        for item in sorted(capsule.iterdir()):
            (workdir / item.name).symlink_to(Path("/capsule") / item.name)


def list_workdir(workdir: Path, capsule: Path | None, limit=2000):
    """The working directory as a nested dictionary, links followed into the capsule."""
    count = [0]

    def host_path(p: Path) -> Path:
        if p.is_symlink():
            target = Path(os.readlink(p))
            if capsule is not None and target.parts[:2] == ("/", "capsule"):
                return capsule.joinpath(*target.parts[2:])
        return p

    def walk(p: Path, depth: int):
        out = {"directories": {}, "files": []}
        try:
            entries = sorted(p.iterdir(), key=lambda e: e.name)
        except OSError:
            return out
        for e in entries:
            count[0] += 1
            if count[0] > limit:
                out["files"].append("... (listing truncated)")
                break
            real = host_path(e)
            if real.is_dir():
                out["directories"][e.name] = walk(real, depth + 1) if depth < 8 else {}
            else:
                out["files"].append(e.name)
        return out

    return json.dumps(walk(workdir, 0), indent=2)


# ---------------------------------------------------------------- sandbox

class Sandbox:
    """One container running ``sandbox_repl.py``; restarted when it dies or hangs."""

    def __init__(self, name, workdir: Path, capsule: Path | None, cpus, memory, cell_timeout,
                 tools_in_kernel=True):
        self.name, self.workdir, self.capsule = name, workdir, capsule
        self.cpus, self.memory, self.cell_timeout = cpus, memory, cell_timeout
        self.tools_in_kernel = tools_in_kernel
        self.proc = None
        self.restarts = 0
        self.submitted = None

    async def start(self):
        # Rootless docker here delegates the memory and pids controllers but not cpu,
        # so a container's CPU share is set by the thread pools its libraries open.
        threads = [f"{var}={self.cpus}" for var in
                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "MC_CORES")]
        cmd = ["docker", "run", "-i", "--rm", "--name", self.name, "--network", "none",
               f"--memory={self.memory}", "--pids-limit=2048",
               "-v", f"{self.workdir}:/workspace", "-v", f"{REPL}:/opt/repl.py:ro",
               "-w", "/workspace", "-e", "HOME=/tmp", "-e", "REPL_OUTPUT_LIMIT=6000",
               "-e", f"REPL_TOOLS={int(self.tools_in_kernel)}"]
        for t in threads:
            cmd += ["-e", t]
        if self.capsule is not None:
            cmd += ["-v", f"{self.capsule}:/capsule:ro"]
        cmd += ["--entrypoint", "python", IMAGE, "-u", "/opt/repl.py"]
        self.proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=1 << 23)

    async def kill(self):
        if self.proc is None:
            return
        killer = await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", self.name, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL)
        await killer.wait()
        try:
            self.proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(self.proc.wait(), 30)
        except asyncio.TimeoutError:
            pass
        self.proc = None

    async def restart(self):
        await self.kill()
        self.restarts += 1
        await self.start()

    async def run(self, code):
        """Run one cell. Returns (output, error, kernel_restarted)."""
        if self.proc is None:
            await self.start()
        request = json.dumps({"code": code, "timeout": self.cell_timeout}) + "\n"
        try:
            self.proc.stdin.write(request.encode())
            await self.proc.stdin.drain()
            line = await asyncio.wait_for(self.proc.stdout.readline(), self.cell_timeout + 60)
        except asyncio.TimeoutError:
            await self.restart()
            return (f"error: the cell ran past {self.cell_timeout}s and did not stop, so the kernel "
                    "was restarted; every variable defined so far is gone."), True, True
        except (BrokenPipeError, ConnectionResetError):
            line = b""
        if not line:
            await self.restart()
            return ("error: the kernel died while running this cell (out of memory?) and was "
                    "restarted; every variable defined so far is gone."), True, True
        reply = json.loads(line)
        self.submitted = reply.get("submitted")
        return reply["output"], bool(reply["error"]), False


# ---------------------------------------------------------------- notebook view

def view_notebook(cells, budget, per_cell=4000, short=400):
    """The notebook as the agent sees it: every cell and its output, within a budget."""
    limits = [per_cell] * len(cells)

    def render():
        parts = []
        for i, (cell, lim) in enumerate(zip(cells, limits)):
            out = cell["output"]
            if len(out) > lim:
                keep = max(lim // 2, 100)
                out = (out[:keep] + f"\n... [{len(out) - 2 * keep} characters not shown] ...\n"
                       + out[-keep:])
            parts.append(f"### Cell {i}:\n```python\n{cell['source']}\n```\n"
                         f"### Output {i}:\n```\n{out}\n```")
        return "\n".join(parts) if parts else "(the notebook is empty)"

    text = render()
    for i in range(len(cells) - 1):          # never the latest cell
        if len(text) <= budget:
            break
        limits[i] = short
        text = render()
    return text


# ---------------------------------------------------------------- the agent's reply

def parse_action(reply):
    """One action from a reply: ('edit', code, idx) | ('submit', answer) | ('list',) | ('none',)."""
    blocks = []
    for match in CODE_RE.finditer(reply):
        lang, body = match.group(1), match.group(2)
        # an edit directive on the line just before the block counts as its first line
        before = reply[:match.start()].rstrip().rsplit("\n", 1)[-1].strip("`* ")
        directive = PRE_EDIT_RE.match(before)
        if directive and not EDIT_RE.match(body.split("\n", 1)[0]):
            body = f"# edit_cell idx={directive.group(1)}\n" + body
        stripped = body.strip()
        lang = lang.lower()
        if re.fullmatch(r"list_workdir\(\s*\)", stripped):
            continue
        if SUBMIT_RE.match(stripped) and "\n" not in stripped:
            continue
        if lang in ("bash", "sh", "shell") and not stripped.startswith("%%"):
            body = "%%bash\n" + body
        elif lang == "r" and not stripped.startswith("%%"):
            body = "%%R\n" + body
        blocks.append(body.rstrip("\n"))
    if blocks:
        code = blocks[0]
        first = code.split("\n", 1)[0]
        m = EDIT_RE.match(first)
        idx = int(m.group(1)) if m else None
        if m:
            code = code.split("\n", 1)[1] if "\n" in code else ""
        return ("edit", code, idx, len(blocks), bool(SUBMIT_RE.search(_outside_code(reply))))
    m = SUBMIT_RE.search(reply)
    if m:
        tail = reply[m.end():]
        tagged = ANSWER_RE.search(tail)
        if tagged:
            answer = tagged.group(1).strip()
        else:
            depth, j = 1, 0
            while j < len(tail) and depth:
                depth += {"(": 1, ")": -1}.get(tail[j], 0)
                j += 1
            answer = tail[:j - 1].strip().strip("\"'").strip()
        return ("submit", answer, tail[:2000])
    if re.search(r"list_workdir\s*\(\s*\)", reply):
        return ("list",)
    return ("none",)


def _outside_code(reply):
    return CODE_RE.sub("", reply)


# ---------------------------------------------------------------- the model

class ContextFull(Exception):
    pass


class MalformedDraw(Exception):
    """A forced tool call the server could not parse: cut off by a stop string or the
    token limit, or off the schema. ldp retries the whole step on it (up to 5 times)."""


async def chat(client, route, model, messages, max_tokens, temperature, seed, extra, failures=None):
    """One reply's text: (content, reasoning, finish_reason, usage)."""
    msg, finish, usage = await chat_message(client, route, model, messages, max_tokens,
                                            temperature, seed, extra, failures)
    return msg.get("content") or "", msg.get("reasoning_content") or "", finish, usage


async def chat_message(client, route, model, messages, max_tokens, temperature, seed, extra,
                       failures=None):
    """One reply, as the server's message. A failed draw is retried on a *new* seed:
    with the seed fixed, a sample that trips the server (gpt-oss under vLLM 0.10.2
    can emit a token its harmony parser rejects, an HTTP 500) would be drawn again
    on every retry.

    ``route`` is the episode's replica and the others of the same config; after
    two failures on one the episode moves to the next for good, so a replica that
    dies mid-run costs its episodes a few seconds rather than the episode."""
    if oa.is_openai(route["base"]):
        # OpenAI's API or an Azure resource (openai_api.py): its key, its parameters, its retries
        # and the ledger. A request or a reply Azure's content filter stops ends the episode
        # (ContentFiltered, caught by the episode), rather than being retried.
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "seed": seed, **extra}
        try:
            d = await oa.post_chat(client, route["base"], payload, f"agent:{model}")
        except oa.ContextLength as err:
            raise ContextFull(str(err)[:300])
        except oa.Rejected as err:
            if failures is not None:
                failures.append(str(err)[:300])
            raise
        if d["choices"][0].get("finish_reason") == "content_filter":
            err = oa.ContentFiltered(200, "the reply was cut by the content filter (finish_reason content_filter)")
            if failures is not None:
                failures.append(str(err)[:300])
            raise err
        return (d["choices"][0]["message"], d["choices"][0].get("finish_reason"),
                d.get("usage") or {})
    delay = 5
    for attempt in range(8):
        base = route["base"]
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "seed": seed + 7919 * attempt, **extra}
        try:
            r = await client.post(f"{base}/v1/chat/completions", json=payload, timeout=1800)
            # Checked before the context test: the error quotes the cut-off call, whose
            # code can say "maximum" or "too long" as well as anything else.
            if r.status_code == 400 and ("json_invalid" in r.text
                                         or "validation error for list" in r.text):
                raise MalformedDraw(r.text[:300])
            if r.status_code == 400 and ("context length" in r.text or "maximum" in r.text
                                         or "too long" in r.text):
                raise ContextFull(r.text[:300])
            r.raise_for_status()
            d = r.json()
            REPLICA_DOWN.discard(base)
            return (d["choices"][0]["message"], d["choices"][0].get("finish_reason"),
                    d.get("usage") or {})
        except (ContextFull, MalformedDraw):
            raise
        except (httpx.TransportError, httpx.HTTPStatusError, json.JSONDecodeError) as err:
            body = getattr(getattr(err, "response", None), "text", "")[:300]
            if failures is not None:
                failures.append(f"{type(err).__name__}: {body or err}"[:300])
            status = getattr(getattr(err, "response", None), "status_code", None)
            if status != 500:                   # a bad draw says nothing about the replica
                REPLICA_DOWN.add(base)
            if attempt % 2 == 1 and route["others"]:
                route["others"].append(route["base"])
                route["base"] = route["others"].pop(0)
            if attempt == 7:
                raise
            if status == 500:                   # a bad draw, not a busy server: redraw now
                continue
            print(f"    retry {attempt + 1} {model}: {type(err).__name__}: {err} {body}", flush=True)
            await asyncio.sleep(delay + random.random() * delay)
            delay = min(delay * 2, 120)


# How hard to shorten old turns, tried in order when the context would overflow:
# (turns kept whole, characters kept of an older reply, of an older observation).
CONDENSE = [None, (4, 600, 300), (3, 400, 200), (2, 250, 120), (1, 150, 80)]


def condense(messages, keep_last=4, assistant_chars=600, tool_chars=300):
    """Shorten the oldest turns; the system prompt, task and recent turns stay whole."""
    head, turns = messages[:2], messages[2:]
    cut = max(0, len(turns) - 2 * keep_last)
    out = list(head)
    for i, m in enumerate(turns):
        if i < cut:
            limit = assistant_chars if m["role"] == "assistant" else tool_chars
            if len(m["content"]) > limit:
                m = {**m, "content": m["content"][:limit] + "\n[... shortened to fit the context]"}
        out.append(m)
    return out


def size_of(messages):
    return sum(len(m["content"]) for m in messages)


# ---------------------------------------------------------------- one driver per run, one run per episode
#
# DeepSeek-V4-Pro's first with-data run had two drivers at once on one --out: a second was started on the
# questions the first had not finished, while the first still had them queued. Each driver looked for an
# episode's output only when its coroutine started, and wrote with a rename, so an episode both finished kept
# whichever finished last, and about 55 finished runs were lost that way. Now
#
# * a driver holds its run's directories, ``--out/<run>`` and ``--work/<run>``, for as long as it lives: a
#   lock file in each names its process and is flock()ed, and a driver started on either while that one is
#   alive stops before it runs anything (the lock is the run's, not all of ``--out``'s, so drivers for
#   different models can still share one ``--out``, as the current agents' runs did);
# * each episode claims its output before it runs: ``<output>.claim``, created only if absent and naming its
#   process; a claim whose process has died is taken over, one whose process lives leaves the episode to it;
# * a finished result is written whole beside its place and linked into it, which fails rather than replace
#   a result already there: that one is kept, and the new one set aside as ``<output>.duplicate-...``.
#
# A process is alive if its pid exists on this host and started when the record says it did, so a pid the
# system has since reused does not count; one on another host is taken to be alive.

LOCK_NAME = ".driver.lock"
HELD = []                                       # this process's driver locks, open until it exits


class Busy(RuntimeError):
    """A live driver holds the run's directory."""


class Claimed(RuntimeError):
    """A live process holds an episode's claim."""


def _process_start(pid):
    """(state, start time in clock ticks since boot) of a process from /proc, or None if there is none."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError):
        return None
    fields = stat.rsplit(")", 1)[1].split()   # the name, in parentheses, may hold spaces or parentheses
    return fields[0], int(fields[19])


def whoami():
    """This process as a lock or a claim records it."""
    found = _process_start(os.getpid())
    return {"pid": os.getpid(), "host": socket.gethostname(), "start": found[1] if found else None}


def _mine(owner):
    return isinstance(owner, dict) and {k: owner.get(k) for k in ("pid", "host", "start")} == whoami()


def alive(owner):
    """Whether the process a lock or a claim names is running."""
    if not isinstance(owner, dict) or not isinstance(owner.get("pid"), int):
        return False
    if owner.get("host") != socket.gethostname():
        return True
    if not Path("/proc/self/stat").exists():     # no /proc: the pid alone
        try:
            os.kill(owner["pid"], 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    found = _process_start(owner["pid"])
    if found is None or found[0] in ("Z", "X"):
        return False
    return owner.get("start") is None or owner["start"] == found[1]


def _read_owner(path):
    try:
        owner = json.loads(Path(path).read_text() or "null")
    except (OSError, ValueError):
        return None
    return owner if isinstance(owner, dict) else None


def _create_whole(path: Path, data: bytes) -> bool:
    """Create ``path`` holding ``data`` if nothing is there, as one step: written beside it, then linked
    into place, which fails if the name is taken. False if something was there already."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp")
    tmp.write_bytes(data)
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    except OSError:                              # a file system without hard links
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return True
    finally:
        tmp.unlink(missing_ok=True)


def claim_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.name + ".claim")


def claim_output(out_path: Path):
    """Claim an episode's output before it runs: the claim's path, or None if a live process holds it."""
    path = claim_path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = (json.dumps(whoami()) + "\n").encode()
    for _ in range(3):
        if _create_whole(path, record):
            return path
        owner = _read_owner(path)
        if owner is None and path.exists():      # unreadable: taken over only once it is old
            try:
                if time.time() - path.stat().st_mtime < 60:
                    return None
            except FileNotFoundError:
                continue
        elif owner is not None and (_mine(owner) or alive(owner)):
            return None
        path.unlink(missing_ok=True)             # its process is gone
    return None


def release_claim(path: Path):
    """Remove a claim this process holds; anyone else's is left."""
    if path is not None and _mine(_read_owner(path)):
        path.unlink(missing_ok=True)


def save_result(out_path: Path, result):
    """Write a finished episode without replacing one already there. Returns the result kept there."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(result, indent=1).encode()
    if _create_whole(out_path, data):
        return result
    aside = out_path.with_name(f"{out_path.name}.duplicate-{os.getpid()}-{time.strftime('%Y%m%dT%H%M%S')}")
    aside.write_bytes(data)
    print(f"  {out_path.name}: a finished result was already there and is kept; this run is set aside as "
          f"{aside.name}", flush=True)
    return json.loads(out_path.read_text())


def hold_run(args):
    """Hold this run's directories under --out and --work for the life of the process, or raise Busy if a
    live driver holds either."""
    fds = []
    try:
        for top in dict.fromkeys((Path(args.out), Path(args.work))):
            where = top / run_name(args)
            where.mkdir(parents=True, exist_ok=True)
            path = where / LOCK_NAME
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            fds.append(fd)
            owner = _read_owner(path)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise Busy(f"{where} is held by a live driver ({_describe(owner)}); wait for it or stop it "
                           "before starting another on the same --out or --work") from None
            except OSError:                      # no flock on this file system: the record alone decides
                pass
            if owner is not None and not _mine(owner) and alive(owner):
                raise Busy(f"{where}: its lock names a live process that does not hold it ({_describe(owner)})")
        # both held: only now does either record name this process
        record = {**whoami(), "since": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "argv": [oa.public_base(a) for a in sys.argv]}
        for fd in fds:
            os.ftruncate(fd, 0)
            os.pwrite(fd, (json.dumps(record) + "\n").encode(), 0)
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise
    HELD.extend(fds)
    return fds


def _describe(owner):
    if not owner:
        return "no record of which"
    return (f"pid {owner.get('pid')} on {owner.get('host')} since {owner.get('since', '?')}: "
            f"{' '.join(map(str, owner.get('argv') or []))[:300]}")


def output_path(args, qid, condition, rollout):
    return Path(args.out) / run_name(args) / condition / f"{qid}__r{rollout}.json"


# ---------------------------------------------------------------- one episode

def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def run_name(args):
    """Where a run's episodes are kept: the model's name, suffixed by any protocol but
    the one the first runs used, so the two protocols' runs of one model never mix."""
    return args.model if args.protocol == "text" else f"{args.model}-{args.protocol}"


async def episode(item, condition, rollout, args, prompts, client, container_slots):
    qid = item["question_id"]
    out_path = args.out / run_name(args) / condition / f"{qid}__r{rollout}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    capsule = None
    if condition == "data":
        stem = item["data_folder"].replace(".zip", "")
        capsule = CAPSULES / "extracted" / stem
    workdir = args.work / run_name(args) / condition / f"{qid}__r{rollout}"
    prepare_workdir(workdir, capsule)
    name = f"bixagent-{run_name(args)}-{condition}-{qid}-r{rollout}-{os.getpid()}"
    box = Sandbox(name, workdir, capsule, args.cpus, args.memory, args.cell_timeout)

    bases = args.base.split(",")               # replicas of one server config
    system = prompts["CAPSULE_SYSTEM_PROMPT_OPEN"]
    task = task_prompt(item["question"], prompts) + "\n" + TOOLS
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    cells, log, usage = [], [], {"prompt_tokens": 0, "completion_tokens": 0}
    failures = []
    answer, raw_submit, why = None, None, "max_steps"
    char_budget = int(args.max_model_len * args.chars_per_token) - args.max_tokens * 4
    extra = request_extra(args)

    async with container_slots:
        # The clock starts when the episode gets its sandbox, not when it is queued:
        # timed from the queue, every episode still waiting after an hour was cut
        # off at step 0 (303 of one model's 410 before this was caught).
        started = time.time()
        live = [b for b in bases if b not in REPLICA_DOWN] or bases
        base = min(live, key=lambda b: (REPLICA_LOAD.get(b, 0), bases.index(b)))
        REPLICA_LOAD[base] = REPLICA_LOAD.get(base, 0) + 1
        route = {"base": base, "others": [b for b in bases if b != base]}
        try:
            for step in range(args.max_steps):
                if time.time() - started > args.wallclock:
                    why = "wallclock"
                    break
                reply, level = None, 0
                full = strip_private(messages)
                while size_of(full if level == 0 else condense(full, *CONDENSE[level])) > char_budget \
                        and level < len(CONDENSE) - 1:
                    level += 1
                filtered = None
                while reply is None and level < len(CONDENSE):
                    sent = full if level == 0 else condense(full, *CONDENSE[level])
                    try:
                        reply, reasoning, finish, used = await chat(
                            client, route, args.model, sent, args.max_tokens, args.temperature,
                            seed_of(args.model, condition, qid, rollout, step), extra, failures)
                    except ContextFull as err:
                        level += 1
                        last_error = str(err)
                    except oa.ContentFiltered as err:  # Azure's filter: no answer, and the run goes on
                        filtered = str(err)[:300]
                        break
                if filtered is not None:
                    why = "content_filter"
                    log.append({"step": step, "error": filtered})
                    break
                if reply is None:
                    why = "context"
                    log.append({"step": step, "error": last_error})
                    break
                for key in usage:
                    usage[key] += int(used.get(key) or 0)
                action = parse_action(reply)
                record = {"step": step, "reply": reply, "finish": finish, "action": action[0]}
                if reasoning:
                    record["reasoning"] = reasoning
                if action[0] == "edit":
                    _, code, idx, n_blocks, also_submitted = action
                    notes = []
                    if idx is not None and 0 <= idx < len(cells):
                        cells[idx] = {"source": code, "output": "", "error": False}
                        await box.restart()
                        for j, cell in enumerate(cells):
                            t0 = time.time()
                            output, error, restarted = await box.run(cell["source"])
                            cell.update(output=output, error=error, seconds=round(time.time() - t0, 1))
                        head = f"Edited cell #{idx} and reran the notebook."
                        record["edited"] = idx
                    else:
                        t0 = time.time()
                        output, error, restarted = await box.run(code)
                        cells.append({"source": code, "output": output, "error": error,
                                      "seconds": round(time.time() - t0, 1)})
                        head = f"Appended cell #{len(cells) - 1} and ran it."
                        if restarted:
                            head += " The kernel was restarted."
                    if n_blocks > 1:
                        notes.append(f"Your reply had {n_blocks} code blocks; only the first was run "
                                     "(one tool per reply).")
                    if also_submitted:
                        notes.append("Your reply also called submit_answer, which was not taken because "
                                     "the reply ran code first (one tool per reply). Call it again once "
                                     "you have seen the output.")
                    if box.submitted is not None:        # submit_answer() called from the cell
                        tagged = ANSWER_RE.search(box.submitted)
                        answer = tagged.group(1).strip() if tagged else box.submitted.strip()
                        raw_submit, why = box.submitted[:2000], "submitted"
                        record["submitted_from_cell"] = True
                        messages.append({"role": "assistant", "content": reply})
                        log.append(record)
                        break
                    observation = head + ("\n" + "\n".join(notes) if notes else "")
                    view = view_notebook(cells, args.view_budget)
                    obs_full = observation + "\n\nMarkdown representation of the notebook:\n\n" + view
                    record["cell"] = len(cells) - 1 if idx is None else idx
                elif action[0] == "submit":
                    answer, raw_submit = action[1], action[2]
                    why = "submitted"
                    messages.append({"role": "assistant", "content": reply})
                    log.append(record)
                    break
                elif action[0] == "list":
                    obs_full = list_workdir(workdir, capsule)
                else:
                    obs_full = ("No tool call was found in your reply. Reply with exactly one of: a fenced "
                                "code block (edit_cell), list_workdir(), or "
                                "submit_answer(\"<answer>your answer</answer>\").")
                log.append(record)
                # hide_old_env_states: only the latest notebook view stays in context
                for m in messages:
                    if m.get("env_state"):
                        m["content"] = m["content"].split("\n\nMarkdown representation", 1)[0] \
                            + "\n\n" + HIDDEN
                        m["env_state"] = False
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": obs_full,
                                 "env_state": action[0] == "edit"})
        finally:
            REPLICA_LOAD[base] -= 1
            await box.kill()

    result = {
        "question_id": qid, "capsule": item["capsule_uuid"], "condition": condition,
        "rollout": rollout, "model": args.model, "served_as": args.served_as or args.model,
        "answer": answer, "raw_submit": raw_submit, "termination": why,
        "steps": len(log), "cells": cells, "log": log, "usage": usage,
        "messages": strip_private(messages),
        "kernel_restarts": box.restarts, "seconds": round(time.time() - started, 1),
        "server_failures": failures, "served_from": oa.public_base(route["base"]),
        "ideal": item["ideal"], "eval_mode": item.get("eval_mode"),
        "settings": {k: (str(v) if isinstance(v, Path) else oa.public_base(v) if k == "base" else v)
                     for k, v in vars(args).items() if k not in ("condition",)},
    }
    result = save_result(out_path, result)        # never over a result already there
    if not args.keep_workdirs:
        shutil.rmtree(workdir, ignore_errors=True)
    return result


def strip_private(messages):
    return [{"role": m["role"], "content": m["content"]} for m in messages]


# ---------------------------------------------------------------- the published protocol
#
# ``--protocol react`` runs the agent BixBench's ``generate_trajectories.yaml`` names:
# ldp's ReActAgent with its default ``single_prompt=False``, at the release BixBench
# runs it with -- ldp 0.26.0, which fhda v1.5.0 pins and BixBench's ``uv.lock``
# resolves. Each step is two calls with fhda's tools passed natively: a reasoning
# call that may not pick a tool (``tool_choice="none"``) and stops at "Action:";
# then, with that reasoning put back as ldp puts it ("Thought: ... Based on this
# reasoning, let's select the appropriate tool!\nAction: ") and a "Continue..." turn,
# a call that must pick one (``"required"``). The tool's reply comes back prefixed
# "Observation:". Where each piece comes from:
#
# * REACT_SYSTEM is ldp's REACT_DEFAULT_PROMPT_TEMPLATE; the two calls, the stop
#   strings on both, the reasoning's wrapper (``postprocess_and_concat_resoning_msg``)
#   and the five attempts at a step are ReActModule's and ReActAgent's. The release
#   matters: ldp 0.37.0 (19 Sep 2025) dropped the wrapper, and puts the reasoning
#   back bare.
# * FHDA_TOOLS is what aviary's ``Tool.from_function`` makes of fhda v1.5.0's three
#   tool functions, ``DataAnalysisEnv`` overriding ``submit_answer``'s type, built
#   with the versions BixBench's ``uv.lock`` pins (fhaviary 0.19.0, pydantic 2.10.1;
#   pydantic 2.12 adds ``additionalProperties`` to the answer's object type).
# * The first observation is ``DataAnalysisEnv.reset``'s -- the task, the notebook's
#   (empty) state, then the system prompt, in that order -- after ldp's own system
#   prompt. Tool replies are fhda's strings, each cell's output is cut at fhda's
#   3,000 characters, and every notebook state but the latest is ldp's
#   ``HiddenEnvStateMessage``. The kernel defines no tools of its own.
#
# What vLLM does differently from the APIs the published agents ran on, and what is
# done about it. The selection call is ``tool_choice="required"``: the API makes the
# model's reply a tool call, in the model's own tool-call format. vLLM's "required"
# instead forces a generic JSON array (``[{"name": ..., "parameters": ...}]``) token
# by token, a format none of these models was trained to call tools in -- in a smoke
# run GLM-4.5-Air, having written "let me start by exploring the working directory",
# was made to call ``submit_answer(null)`` at step 0 in two of three episodes -- and
# "auto" lets the model answer in ReAct's text format instead ("Action:
# list_workdir()" and an invented observation), which ended 3 of Qwen2.5-72B's first
# 10 episodes at step 0 after five draws. So the selection call opens the model's
# reply with the model's own tool-call tag (``continue_final_message``) and stops at
# the closing tag: the model must call a tool, chooses which and with what
# arguments in the format it was trained on, and makes exactly one call, as
# ``parallel_tool_calls: False`` has it. The call is read with that format's rules
# (TOOL_FORMATS); one that cannot be read is drawn again, one of the step's five
# attempts, as ldp's MalformedMessageError is. The ReAct stop strings stay on the
# reasoning call, where they cut an invented "Observation:"; inside a forced call
# they would only cut code that happens to contain them.
#
# vLLM's "none" does not keep a call out of the reasoning turn either: the tools are
# in the prompt, nothing stops the model writing its own call tag, and vLLM hands
# the call back as text. Before the ban GLM-4.5-Air did so in 1,402 of 1,432 reasoning
# turns and Qwen2.5-72B in 574 of 2,102 (the 226 episodes of
# build/agent_runs_discarded_nowrapper), and Qwen3-30B-A3B in four draws of four at
# step 0. The APIs the published agents ran on never return a call there, so the
# reasoning call bans the opening the selection call forces (vLLM's ``bad_words``,
# ``reasoning_ban``): the model has to reason in text, as ldp asks it to.

REACT_SYSTEM = (
    "Answer the following questions as best you can, using the provided tools.\n\n"
    "Use the following format:\n\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, should be one of the provided tools with necessary arguments\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Observation can repeat N times)\n\n"
    "Example:\n\n"
    "Thought: I need to use the get_weather tool\n"
    'Action: get_weather("New York", 7)\n'
    "Observation: The 7 day forecast for New York is [...]"
)
REACT_STOP = ["Observation:", "Action:"]
REACT_CONTINUE = "Continue..."
REACT_ATTEMPTS = 5


def react_thought(reasoning):
    """The reasoning as ldp 0.26.0's ``postprocess_and_concat_resoning_msg`` puts it
    back, for the selection call and for every later step."""
    return (f"Thought: {(reasoning or '').removeprefix('Thought: ')}."
            " Based on this reasoning, let's select the appropriate tool!"
            "\nAction: ")


NB_PATH = "/workspace/notebook.ipynb"
NB_OUTPUT_LIMIT = 3000          # fhda.config
FHDA_TOOLS = [{'type': 'function',
  'function': {'name': 'edit_cell',
               'description': 'Edit the notebook by modifying a specific code cell.\n'
                              '\n'
                              'ONLY CODE CELLS ARE SUPPORTED. Do no attempt to write Markdown '
                              'or raw text,\n'
                              'though you are permitted (and encouraged) to write comments in '
                              'the code cells.\n'
                              'The notebook will be automatically rerun if a successful edit '
                              'is made.',
               'parameters': {'type': 'object',
                              'properties': {'contents': {'description': 'Cell contents to '
                                                                         'insert. We assume '
                                                                         'the cell is a code '
                                                                         'block.',
                                                          'title': 'Contents',
                                                          'type': 'string'},
                                             'idx': {'anyOf': [{'type': 'integer'},
                                                               {'type': 'null'}],
                                                     'description': 'Index of the cell to '
                                                                    'edit. If not provided '
                                                                    '(None default),\n'
                                                                    'then appends a new cell.',
                                                     'title': 'Idx'}},
                              'required': ['contents']}}},
 {'type': 'function',
  'function': {'name': 'list_workdir',
               'description': 'Recursively lists the contents of the working directory.\n'
                              '\n'
                              'The contents is represented as a nested JSON dictionary.',
               'parameters': {'type': 'object', 'properties': {}, 'required': []}}},
 {'type': 'function',
  'function': {'name': 'submit_answer',
               'description': 'Submit an answer to the problem.\n'
                              '\n'
                              'Note that this tool may only be called once and ends the '
                              'episode.',
               'parameters': {'type': 'object',
                              'properties': {'answer': {'anyOf': [{'type': 'string'},
                                                                  {'type': 'number'},
                                                                  {'type': 'object'},
                                                                  {'type': 'null'}],
                                                        'description': 'The answer to the '
                                                                       'problem',
                                                        'title': 'Answer'}},
                              'required': ['answer']}}}]


# Each model family's own tool-call opening and closing, as its chat template writes
# them, and how the call between them is read (vLLM's own parsers, ``hermes`` and
# ``glm45``, read them the same way).
TOOL_FORMATS = {"hermes": ("<tool_call>\n", "</tool_call>"),    # Qwen2.5
                "glm45": ("<tool_call>", "</tool_call>"),         # GLM-4.5
                # Llama 3.x writes a call as a bare object right after its header and
                # ends the turn, so the call closes itself: no closing tag to stop at.
                "llama3_json": ('{"name": "', None)}


def reasoning_ban(fmt):
    """What the reasoning turn may not write: the opening the selection turn forces,
    less the newline hermes puts after its tag (the tag is one token in Qwen's and
    GLM's vocabularies; Llama's opening is four)."""
    return TOOL_FORMATS[fmt][0].strip()


_PARAM_TYPES = {t["function"]["name"]: {k: v.get("type") for k, v in
                                         t["function"]["parameters"]["properties"].items()}
                for t in FHDA_TOOLS}


def parse_forced_call(fmt, text):
    """The call a forced reply continues with: (name, arguments), or ValueError."""
    if fmt == "hermes":
        obj = json.loads(text.strip())
        if not isinstance(obj, dict):
            raise ValueError("the call is not an object")
        name = obj.get("name")
        arguments = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
    elif fmt == "glm45":
        head, _, rest = text.partition("\n")
        name, arguments = head.strip(), {}
        for key, value in re.findall(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", rest, re.S):
            key = key.strip()
            if _PARAM_TYPES.get(name, {}).get(key) == "string":
                arguments[key] = value
            else:
                try:
                    arguments[key] = json.loads(value)
                except json.JSONDecodeError:
                    arguments[key] = value
    elif fmt == "llama3_json":
        obj, _ = json.JSONDecoder().raw_decode((TOOL_FORMATS[fmt][0] + text).strip())
        if not isinstance(obj, dict):
            raise ValueError("the call is not an object")
        name = obj.get("name")
        arguments = obj.get("parameters", obj.get("arguments", {}))
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
    else:
        raise ValueError(f"no tool-call format {fmt!r}")
    if name not in _PARAM_TYPES:
        raise ValueError(f"no tool named {name!r}")
    if not isinstance(arguments, dict):
        raise ValueError("the arguments are not an object")
    return name, arguments


def parse_native_call(message):
    """The call an API with native tool calling returned (``tool_choice: required``): its first,
    as (name, arguments), or ValueError."""
    calls = message.get("tool_calls") or []
    if not calls:
        raise ValueError("no tool call")
    function = calls[0].get("function") or {}
    name, arguments = function.get("name"), function.get("arguments") or "{}"
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if name not in _PARAM_TYPES:
        raise ValueError(f"no tool named {name!r}")
    if not isinstance(arguments, dict):
        raise ValueError("the arguments are not an object")
    return name, arguments


def fhda_cut(output, limit=NB_OUTPUT_LIMIT):
    """fhda's ``limit_notebook_output``."""
    if len(output) < limit:
        return output
    half = int(limit / 2)
    return output[:half] + "\n<...output limited...>\n" + output[-half:]


def fhda_view(cells, budget=float("inf"), short=400):
    """fhda's ``view_notebook``: each cell, then its output if it printed any. When the
    whole view would pass ``budget`` (ours: the published agents had 200k tokens of
    context), the oldest outputs are cut shorter first; the latest cell never is."""
    limits = [NB_OUTPUT_LIMIT] * len(cells)

    def render():
        md = []
        for i, (cell, limit) in enumerate(zip(cells, limits)):
            md += [f"### Cell {i}:", f"```{LANGUAGE}", cell["source"], "```"]
            output = cell["output"]
            if output and output != "(no output)":
                md += [f"### Output {i}:", "```", fhda_cut(output, limit), "```"]
        return "\n".join(md)

    text = render()
    for i in range(len(cells) - 1):
        if len(text) <= budget:
            break
        limits[i] = short
        text = render()
    return text


def fhda_state(cells, budget):
    """``get_env_state_msg``'s text."""
    return f"Markdown representation of notebook contents ({NB_PATH}):\n\n" + fhda_view(cells, budget)


def fhda_list_dir(workdir: Path, capsule: Path | None, limit=2000):
    """fhda's ``_list_dir``: a directory's files under "files", each subdirectory under
    its own name (with the empty "directories" key fhda also writes), links followed
    into the capsule. Sorted, and capped at ``limit`` entries, where fhda is neither."""
    count = [0]

    def host_path(p: Path) -> Path:
        if p.is_symlink():
            target = Path(os.readlink(p))
            if capsule is not None and target.parts[:2] == ("/", "capsule"):
                return capsule.joinpath(*target.parts[2:])
        return p

    def walk(p: Path, depth: int):
        index = {}
        try:
            entries = sorted(p.iterdir(), key=lambda e: e.name)
        except OSError:
            return index
        for e in entries:
            count[0] += 1
            if count[0] > limit:
                index.setdefault("files", []).append("... (listing truncated)")
                break
            real = host_path(e)
            if real.is_dir():
                index.setdefault("directories", {})
                index[e.name] = walk(real, depth + 1) if depth < 8 else {}
            else:
                index.setdefault("files", []).append(e.name)
        return index

    return json.dumps(walk(workdir, 0), indent=2)


def save_notebook(workdir: Path, cells):
    """The notebook and its markdown, where fhda keeps them in the working directory."""
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": ([{"name": "stdout", "output_type": "stream", "text": c["output"]}]
                                 if c["output"] and c["output"] != "(no output)" else []),
                     "source": c["source"]} for c in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    (workdir / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
    (workdir / "notebook.md").write_text(fhda_view(cells))


def react_size(messages):
    return sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or []))
               for m in messages)


def _shorten_call(call, limit):
    try:
        arguments = json.loads(call["function"]["arguments"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return call
    for key, value in list(arguments.items()):
        if isinstance(value, str) and len(value) > limit:
            arguments[key] = value[:limit] + "\n# [... shortened to fit the context]"
    return {**call, "function": {**call["function"], "arguments": json.dumps(arguments)}}


def react_condense(messages, keep_last=4, assistant_chars=600, tool_chars=300):
    """Shorten the oldest turns (a turn is five messages here); the two system prompts,
    the task, the first notebook state and the recent turns stay whole."""
    head, turns = messages[:4], messages[4:]
    cut = max(0, len(turns) - 5 * keep_last)
    out = list(head)
    for i, m in enumerate(turns):
        if i < cut:
            m = dict(m)
            limit = assistant_chars if m["role"] == "assistant" else tool_chars
            if m.get("content") and len(m["content"]) > limit:
                m["content"] = m["content"][:limit] + "\n[... shortened to fit the context]"
            if m.get("tool_calls"):
                m["tool_calls"] = [_shorten_call(c, limit) for c in m["tool_calls"]]
        out.append(m)
    return out


async def react_send(client, route, args, messages, seed, extra, failures, char_budget):
    """One call with the context fitted, the oldest turns shortened first."""
    level = 0
    while level < len(CONDENSE) - 1 and react_size(
            messages if level == 0 else react_condense(messages, *CONDENSE[level])) > char_budget:
        level += 1
    last = "context"
    while level < len(CONDENSE):
        sent = messages if level == 0 else react_condense(messages, *CONDENSE[level])
        try:
            return await chat_message(client, route, args.model, sent, args.max_tokens,
                                      args.temperature, seed, extra, failures)
        except ContextFull as err:
            level, last = level + 1, str(err)
    raise ContextFull(last)


def request_extra(args):
    extra = dict(json.loads(args.extra_body)) if args.extra_body else {}
    if args.reasoning_effort:
        extra["reasoning_effort"] = args.reasoning_effort
    return extra


def public(messages):
    """The messages as sent: this script's own markers dropped."""
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


async def episode_react(item, condition, rollout, args, prompts, client, container_slots):
    qid = item["question_id"]
    run = run_name(args)
    out_path = args.out / run / condition / f"{qid}__r{rollout}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    capsule = None
    if condition == "data":
        capsule = CAPSULES / "extracted" / item["data_folder"].replace(".zip", "")
    workdir = args.work / run / condition / f"{qid}__r{rollout}"
    prepare_workdir(workdir, capsule)
    name = f"bixagent-{run}-{condition}-{qid}-r{rollout}-{os.getpid()}"
    box = Sandbox(name, workdir, capsule, args.cpus, args.memory, args.cell_timeout,
                  tools_in_kernel=False)
    bases = args.base.split(",")

    cells = []
    save_notebook(workdir, cells)
    system = {"role": "system", "content": REACT_SYSTEM}
    history = [                                # DataAnalysisEnv.reset's observation
        {"role": "user", "content": task_prompt(item["question"], prompts)},
        {"role": "user", "content": fhda_state(cells, args.view_budget), "_state": True},
        {"role": "system", "content": prompts["CAPSULE_SYSTEM_PROMPT_OPEN"]},
    ]
    log, failures = [], []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    answer, raw_submit, why = None, None, "max_steps"
    char_budget = int(args.max_model_len * args.chars_per_token) - args.max_tokens * 4
    extra = request_extra(args)
    # On OpenAI's API the selection call is ldp's own, a native call with tool_choice "required";
    # a vLLM server's forced call is opened in the model's own format and continued instead.
    native = oa.is_openai(bases[0])
    if native:
        reason_extra = {**extra, "tools": FHDA_TOOLS, "tool_choice": "none", "stop": REACT_STOP}
        select_extra = {**extra, "tools": FHDA_TOOLS, "tool_choice": "required"}
        opening = None
    else:
        reason_extra = {**extra, "tools": FHDA_TOOLS, "tool_choice": "none", "stop": REACT_STOP,
                        "bad_words": [reasoning_ban(args.tool_format)]}
        opening, closing = TOOL_FORMATS[args.tool_format]
        select_extra = {**extra, "tools": FHDA_TOOLS, "tool_choice": "none", "stop": [closing] if closing else [],
                        "continue_final_message": True, "add_generation_prompt": False}

    async with container_slots:
        started = time.time()                  # from the sandbox, not the queue
        live = [b for b in bases if b not in REPLICA_DOWN] or bases
        base = min(live, key=lambda b: (REPLICA_LOAD.get(b, 0), bases.index(b)))
        REPLICA_LOAD[base] = REPLICA_LOAD.get(base, 0) + 1
        route = {"base": base, "others": [b for b in bases if b != base]}
        try:
            for step in range(args.max_steps):
                if time.time() - started > args.wallclock:
                    why = "wallclock"
                    break
                record = {"step": step, "malformed": []}
                picked = None
                try:
                    for attempt in range(REACT_ATTEMPTS):
                        seed = seed_of(args.model, "react", condition, qid, rollout, step, attempt)
                        try:
                            thought, finish1, used1 = await react_send(
                                client, route, args, [system, *public(history)], seed, reason_extra,
                                failures, char_budget)
                            reasoning = thought.get("content") or ""
                            convo = [*history, {"role": "assistant", "content": react_thought(reasoning)},
                                     {"role": "user", "content": REACT_CONTINUE}]
                            chosen, finish2, used2 = await react_send(
                                client, route, args,
                                [system, *public(convo)] if native
                                else [system, *public(convo), {"role": "assistant", "content": opening}],
                                seed + 1, select_extra, failures, char_budget)
                        except MalformedDraw as err:
                            record["malformed"].append(str(err)[:300])
                            continue
                        for used in (used1, used2):
                            for key in usage:
                                usage[key] += int(used.get(key) or 0)
                        text = chosen.get("content") or ""
                        try:
                            tool, arguments = (parse_native_call(chosen) if native
                                               else parse_forced_call(args.tool_format, text))
                        except (KeyError, TypeError, ValueError) as err:
                            record["malformed"].append(f"{type(err).__name__}: {err} | {text[:200]!r}"[:400])
                            continue
                        call = {"id": f"call_{step}_{attempt}", "type": "function",
                                "function": {"name": tool, "arguments": json.dumps(arguments)}}
                        picked = (reasoning, call, arguments, finish1, finish2)
                        break
                except ContextFull as err:
                    why = "context"
                    record["error"] = str(err)[:300]
                    log.append(record)
                    break
                except oa.ContentFiltered as err:    # Azure's filter: no answer, and the run goes on
                    why = "content_filter"
                    record["error"] = str(err)[:300]
                    log.append(record)
                    break
                if picked is None:
                    why = "malformed"
                    log.append(record)
                    break
                reasoning, call, arguments, finish1, finish2 = picked
                tool = call["function"]["name"]
                record.update(reasoning=reasoning, tool=tool, arguments=arguments,
                              finish=[finish1, finish2])
                history += [{"role": "assistant", "content": react_thought(reasoning)},
                            {"role": "user", "content": REACT_CONTINUE},
                            {"role": "assistant", "content": "", "tool_calls": [call]}]

                done = False
                try:
                    if tool == "edit_cell":
                        contents = arguments.get("contents")
                        if not isinstance(contents, str):
                            contents = "" if contents is None else str(contents)
                        idx = arguments.get("idx")
                        if idx is not None:
                            try:
                                idx = int(idx)
                            except (ValueError, TypeError):
                                idx = None
                        if idx is None or idx >= len(cells):
                            t0 = time.time()
                            output, error, restarted = await box.run(contents)
                            cells.append({"source": contents, "output": output, "error": error,
                                          "seconds": round(time.time() - t0, 1)})
                            result = f"Appended new cell (#{len(cells) - 1})."
                            record["cell"] = len(cells) - 1
                        else:
                            cells[idx]["source"] = contents     # a negative idx counts from the end
                            await box.restart()
                            for cell in cells:
                                t0 = time.time()
                                output, error, restarted = await box.run(cell["source"])
                                cell.update(output=output, error=error, seconds=round(time.time() - t0, 1))
                            result = f"Edited cell #{idx}."
                            record["edited"] = idx
                        save_notebook(workdir, cells)
                    elif tool == "list_workdir":
                        result = fhda_list_dir(workdir, capsule)
                    elif tool == "submit_answer":
                        submitted = arguments.get("answer")
                        if submitted is None:             # no answer, as upstream reads it
                            raw_submit = answer = None
                        else:
                            raw_submit = submitted if isinstance(submitted, str) else json.dumps(submitted)
                            tagged = ANSWER_RE.search(raw_submit)
                            answer = tagged.group(1).strip() if tagged else raw_submit.strip()
                        result, why, done = f"Submitted answer: {submitted}", "submitted", True
                    else:
                        raise ValueError(f"no tool named {tool!r}")
                except IndexError as err:             # fhda's own edit_cell raises it too
                    result = f"Encountered exception during tool call for tool {tool}: {err!r}"
                record["result"] = result[:500]
                log.append(record)
                if done:
                    break
                for m in history:                     # hide_old_env_states
                    if m.get("_state"):
                        m["content"], m["_state"] = HIDDEN, False
                history += [{"role": "tool", "tool_call_id": call["id"], "name": tool,
                             "content": f"Observation: {result}"},
                            {"role": "user", "content": fhda_state(cells, args.view_budget),
                             "_state": True}]
        finally:
            REPLICA_LOAD[base] -= 1
            await box.kill()

    result = {
        "question_id": qid, "capsule": item["capsule_uuid"], "condition": condition,
        "rollout": rollout, "model": args.model, "protocol": args.protocol,
        "served_as": args.served_as or args.model,
        "answer": answer, "raw_submit": raw_submit, "termination": why,
        "steps": len(log), "cells": cells, "log": log, "usage": usage,
        "messages": [system, *public(history)],
        "kernel_restarts": box.restarts, "seconds": round(time.time() - started, 1),
        "server_failures": failures, "served_from": oa.public_base(route["base"]),
        "ideal": item["ideal"], "eval_mode": item.get("eval_mode"),
        "settings": {k: (str(v) if isinstance(v, Path) else oa.public_base(v) if k == "base" else v)
                     for k, v in vars(args).items() if k not in ("condition",)},
    }
    result = save_result(out_path, result)        # never over a result already there
    if not args.keep_workdirs:
        shutil.rmtree(workdir, ignore_errors=True)
    return result


# ---------------------------------------------------------------- main

def only_file_ids(path):
    """Question ids listed one per line (blank lines and ``#`` comments skipped); none without a file."""
    if not path:
        return set()
    return {line.split("#", 1)[0].strip() for line in open(path) if line.split("#", 1)[0].strip()}


def dry_run(args):
    """The episodes a run would make, and on OpenAI's API what they would cost, estimated from the
    token usage of the published protocol's episodes already on this host; nothing is called."""
    items = [json.loads(l) for l in open(args.items)]
    if args.only or args.only_file:
        wanted = set(args.only or []) | only_file_ids(args.only_file)
        items = [it for it in items if it["question_id"] in wanted]
    if args.limit:
        items = items[:args.limit]
    run = run_name(args)
    jobs = [(it, cond, r) for r in range(args.first_rollout, args.first_rollout + args.rollouts)
            for cond in args.condition for it in items]
    todo = [j for j in jobs if not (args.out / run / j[1] / f"{j[0]['question_id']}__r{j[2]}.json").exists()]
    missing = sorted({it["data_folder"] for it, cond, _ in todo if cond == "data"
                      and not (CAPSULES / it["data_folder"]).exists()})
    print(f"dry run, {args.model} at {args.base}: {len(todo)} episodes to run of {len(jobs)} "
          f"({len(items)} questions x {args.rollouts} rollouts x {len(args.condition)} conditions); "
          f"{len(missing)} capsule zips missing from {CAPSULES}" + (f": {missing[:3]} ..." if missing else ""))
    usage = {}
    for path in sorted((ROOT / "build" / "agent_runs").glob("*-react/data/*.json")):
        u = json.loads(path.read_text()).get("usage") or {}
        usage.setdefault(path.parent.parent.name, []).append((u.get("prompt_tokens", 0), u.get("completion_tokens", 0)))
    if not usage or not oa.is_openai(args.base.split(",")[0]):
        return
    inp, cin, cout = oa.prices(args.model)
    cached_share = 0.8          # a step resends the last one's context: only the newest turn is fresh
    print(f"  at ${inp}/{cin}/{cout} per 1M input/cached/output tokens, {cached_share:.0%} of prompt tokens "
          f"cached; per episode as the open models' published-protocol episodes ran (a reasoning model's "
          f"hidden reasoning comes on top of their completion tokens):")
    for name, rows in sorted(usage.items()):
        prompt = sum(r[0] for r in rows) / len(rows)
        completion = sum(r[1] for r in rows) / len(rows)
        per = (prompt * (1 - cached_share) * inp + prompt * cached_share * cin + completion * cout) / 1e6
        print(f"    like {name:16s} {prompt / 1e3:6.0f}k prompt, {completion / 1e3:5.1f}k completion: "
              f"${per:.2f} an episode, ${per * len(todo):,.0f} for the run")


async def main_async(args):
    prompts = fhda_prompts()
    items = [json.loads(l) for l in open(args.items)]
    if args.only or args.only_file:
        wanted = set(args.only or []) | only_file_ids(args.only_file)
        items = [it for it in items if it["question_id"] in wanted]
    if args.limit:
        items = items[:args.limit]
    if "data" in args.condition:
        (CAPSULES / "extracted").mkdir(exist_ok=True)
        with open(CAPSULES / "extracted" / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)          # runs for several models share one copy
            for folder in sorted({it["data_folder"] for it in items}):
                zip_path = CAPSULES / folder
                extract_capsule(zip_path, CAPSULES / "extracted" / folder.replace(".zip", ""))
    rollouts = range(args.first_rollout, args.first_rollout + args.rollouts)
    jobs = [(it, cond, r) for r in rollouts for cond in args.condition for it in items]
    random.Random(args.seed).shuffle(jobs)
    slots = asyncio.Semaphore(args.concurrency)
    # vLLM's server drops a connection idle for 5 s, and a cell often runs longer than
    # that: a pooled connection reused after it had been dropped came back "server
    # disconnected". Kept at most 2 s, one is closed here first.
    limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency,
                          keepalive_expiry=2.0)
    done = {"submitted": 0, "other": 0}
    filtered = left = 0
    t0 = time.time()
    async with httpx.AsyncClient(limits=limits, timeout=1800) as client:
        async def run(job):
            nonlocal filtered, left
            it, cond, r = job
            out_path, claim = output_path(args, it["question_id"], cond, r), None
            try:
                # claimed before it runs; one a live process holds is left to that process
                if not out_path.exists() and (claim := claim_output(out_path)) is None:
                    raise Claimed(f"{claim_path(out_path).name} is held by a live process; left to it")
                run_one = episode_react if args.protocol == "react" else episode
                res = await run_one(it, cond, r, args, prompts, client, slots)
                done["submitted" if res["termination"] == "submitted" else "other"] += 1
                filtered += res["termination"] == "content_filter"
            except Claimed as err:
                left += 1
                print(f"  {it['question_id']} {cond} r{r}: {err}", flush=True)
            except Exception as err:          # one broken episode must not stop the run
                done["other"] += 1
                print(f"  {it['question_id']} {cond} r{r}: {type(err).__name__}: {err}", flush=True)
            finally:
                release_claim(claim)
            n = sum(done.values()) + left
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} episodes, {done['submitted']} submitted, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
        await asyncio.gather(*(run(j) for j in jobs))
    print(f"done {args.model}: {sum(done.values())} episodes, {done['submitted']} submitted, "
          f"{filtered} ended by the content filter" + (f", {left} left to another live process" if left else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--served-as", default=None, help="the weights behind --model, for the record")
    ap.add_argument("--base", required=True,
                    help="OpenAI-compatible server, e.g. http://127.0.0.1:18101; several replicas "
                         "of one config may be given comma-separated")
    ap.add_argument("--items", default=str(ROOT / "data" / "bixbench.jsonl"))
    ap.add_argument("--condition", action="append", choices=["data", "nodata"], required=True)
    ap.add_argument("--only", action="append", help="question_id to run (repeatable)")
    ap.add_argument("--only-file", default=None, help="a file of question_ids to run, one per line")
    ap.add_argument("--dry-run", action="store_true",
                    help="count the episodes and, on OpenAI's API, estimate their cost; run nothing")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--first-rollout", type=int, default=0,
                    help="number rollouts from here, so more seeds of a finished run add to it")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--chars-per-token", type=float, default=3.0)
    ap.add_argument("--view-budget", type=int, default=36000)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--protocol", choices=["text", "react"], default="text",
                    help="text: tools called in the reply (the first runs); react: BixBench's "
                         "published ReActAgent, two calls a step, tools called natively")
    ap.add_argument("--tool-format", choices=sorted(TOOL_FORMATS), default=None,
                    help="react: the model family's own tool-call format (hermes: Qwen2.5; glm45: GLM-4.5; "
                         "llama3_json: Llama 3.x)")
    ap.add_argument("--extra-body", default=None,
                    help="JSON merged into every request, e.g. chat_template_kwargs")
    ap.add_argument("--cell-timeout", type=int, default=600)
    ap.add_argument("--wallclock", type=int, default=3600)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--cpus", default="4")
    ap.add_argument("--memory", default="24g")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "agent_runs")
    ap.add_argument("--work", type=Path,
                    default=Path(os.environ.get("BIXBENCH_WORK", str(ROOT / "build" / "bixbench-work"))))
    ap.add_argument("--keep-workdirs", action="store_true")
    args = ap.parse_args()
    # docker mounts need absolute paths: a relative one is read as a volume name and the container never starts
    args.out, args.work = Path(args.out).resolve(), Path(args.work).resolve()
    if args.protocol == "react" and args.tool_format is None and not oa.is_openai(args.base.split(",")[0]):
        ap.error("--protocol react needs --tool-format: the forced call opens in the model's own format")
    if args.dry_run:
        dry_run(args)
        return
    try:
        hold_run(args)                          # one live driver per run's --out and --work
    except Busy as err:
        raise SystemExit(f"bixbench_agent.py: {err}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
