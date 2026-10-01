#!/usr/bin/env python
"""A notebook kernel without Jupyter: one persistent namespace, cells in, outputs out.

Runs *inside* the sandbox container (``bixbench_agent.py`` starts one per rollout,
with no network and the capsule mounted read-only). BixBench's agents write cells
into a Jupyter notebook whose state persists between cells; this keeps that
property -- a dataframe loaded in cell 2 is still there in cell 9 -- with a
protocol small enough to audit:

    stdin   one JSON object per line: {"code": "...", "timeout": 300}
    stdout  one JSON object per line: {"output": "...", "error": false}

Output is captured at the file-descriptor level, not by swapping ``sys.stdout``,
because cells run subprocesses (``%%bash``, ``!cmd``, ``Rscript``) whose writes go
straight to fd 1 and would otherwise land in the protocol stream. The protocol
therefore runs on duplicates of the original descriptors taken at start-up, and
fd 0 is pointed at /dev/null so nothing a cell starts can read the next request.

As in a notebook the value of a cell's last expression is displayed, magics that
only configure the display (``%matplotlib inline``) are dropped, ``!cmd`` lines
run in a shell in place, and a ``%%bash`` cell runs its body with bash -- the cell
kind BixBench's prompt tells the agent it may use besides Python.

A ``%%R`` cell runs in one embedded R session that persists between cells, the way
rpy2's ``%%R`` magic does in the image BixBench's agents used (it ships rpy2 for
that). Running each R cell as a fresh ``Rscript`` would silently lose a count
matrix loaded two cells earlier, and a DESeq2 analysis is several R cells long.
Every visible top-level value is printed, as at the R prompt; ``-i name`` and
``-o name`` move a variable in and out as the magic's flags do. Rscript is the
fallback only where rpy2 will not import.
"""
import ast
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import traceback

LIMIT = int(os.environ.get("REPL_OUTPUT_LIMIT", "6000"))
os.environ.setdefault("MPLBACKEND", "Agg")



class CellTimeout(Exception):
    pass


# The agent's tools, callable from code as well: agents write ``submit_answer(x)``
# inside a cell as often as on its own line, and a NameError there would strand
# an agent that has finished its analysis.
SUBMITTED = []


def submit_answer(answer=None):
    """Submit an answer to the problem. This may only be called once and ends the episode."""
    SUBMITTED.append(str(answer))
    print("(answer submitted)")


def list_workdir(path="/workspace"):
    """Recursively list the working directory as a nested JSON dictionary."""
    def walk(p, depth):
        out = {"directories": {}, "files": []}
        try:
            entries = sorted(os.listdir(p))
        except OSError:
            return out
        for name in entries:
            full = os.path.join(p, name)
            if os.path.isdir(full):
                out["directories"][name] = walk(full, depth + 1) if depth < 8 else {}
            else:
                out["files"].append(name)
        return out
    print(json.dumps(walk(path, 0), indent=2))


def _alarm(signum, frame):
    raise CellTimeout()


class ShellFailed(Exception):
    pass


def _sh(command, timeout, raise_on_error=False):
    """Run a shell command in place. As in IPython, ``!cmd`` never raises and a
    ``%%bash`` cell whose script exits non-zero is an error."""
    done = subprocess.run(["bash", "-c", command], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=timeout)
    sys.stdout.write(done.stdout)
    sys.stdout.write(done.stderr)
    if raise_on_error and done.returncode != 0:
        raise ShellFailed(done.returncode)


# Line magics that only configure the front end; anything else is reported as ignored.
QUIET_MAGICS = ("%matplotlib", "%config", "%load_ext", "%reload_ext", "%autoreload")


def _translate(code):
    """Notebook-isms into Python: drop display magics, run ``!cmd`` in place."""
    out = []
    for line in code.split("\n"):
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        if stripped.startswith("!"):
            out.append(f"{indent}__sandbox_sh__({stripped[1:]!r})")
        elif stripped.startswith("%"):
            if stripped.startswith(QUIET_MAGICS):
                out.append(f"{indent}pass")
            else:
                note = f"(line magic not supported here, skipped: {stripped.split()[0]})"
                out.append(f"{indent}print({note!r})")
        else:
            out.append(line)
    return "\n".join(out)


# At the R prompt every visible top-level value prints; ``eval`` of the whole
# block would print only the last one.
R_RUNNER = """
function(code) {
  exprs <- parse(text = code)
  for (e in exprs) {
    res <- withVisible(eval(e, envir = globalenv()))
    if (res$visible) print(res$value)
  }
  invisible(NULL)
}
"""
_R = {}


def _r_session():
    if "run" not in _R:
        import rpy2.robjects as ro
        from rpy2.rinterface_lib import callbacks
        callbacks.consolewrite_print = lambda text: sys.stdout.write(text)
        callbacks.consolewrite_warnerror = lambda text: sys.stderr.write(text)
        _R["ro"] = ro
        _R["run"] = ro.r(R_RUNNER)
    return _R


def _run_r(head, body, namespace, timeout):
    flags = head.split()[1:]
    ins = [flags[i + 1] for i, f in enumerate(flags[:-1]) if f == "-i"]
    outs = [flags[i + 1] for i, f in enumerate(flags[:-1]) if f == "-o"]
    try:
        session = _r_session()
    except Exception:
        with tempfile.NamedTemporaryFile("w", suffix=".R", dir=".", delete=False) as f:
            f.write(body)
        try:
            _sh(f"Rscript {f.name}", timeout, raise_on_error=True)
        finally:
            os.unlink(f.name)
        return
    ro = session["ro"]
    from rpy2.robjects import default_converter, numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter
    convert = default_converter + pandas2ri.converter + numpy2ri.converter
    with localconverter(convert):
        for name in ins:
            ro.globalenv[name] = namespace[name]
    try:
        session["run"](body)
    except Exception:
        # R has already written its own error message to the console.
        raise _RFailed() from None
    finally:
        sys.stdout.flush()
    with localconverter(convert):
        for name in outs:
            namespace[name] = ro.globalenv[name]


class _RFailed(Exception):
    pass


def _run_python(code, namespace):
    tree = ast.parse(_translate(code), filename="<cell>", mode="exec")
    last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    if last is not None:
        tree.body = tree.body[:-1]
    exec(compile(tree, "<cell>", "exec"), namespace)
    if last is not None:
        value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), namespace)
        if value is not None:
            print(repr(value))


def run_cell(code, namespace, timeout):
    stripped = code.lstrip()
    head = stripped.split("\n", 1)[0].strip()
    body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    magic = head.split()[0].lower() if head.startswith("%%") else ""
    if magic in ("%%bash", "%%sh", "%%script"):
        _sh(body, timeout, raise_on_error=True)
    elif magic == "%%r":
        _run_r(head, body, namespace, timeout)
    elif magic in ("%%time", "%%capture", "%%timeit"):
        _run_python(body, namespace)
    else:
        _run_python(code, namespace)


def truncate(text):
    if len(text) <= LIMIT:
        return text
    keep = LIMIT // 2
    return (text[:keep] + f"\n... [{len(text) - 2 * keep} characters truncated] ...\n"
            + text[-keep:])


def main():
    # The protocol keeps the original descriptors; everything a cell does sees
    # fresh ones. Done here rather than at import, so the module can be tested.
    proto_in = os.fdopen(os.dup(0), "r")
    proto_out = os.fdopen(os.dup(1), "w")
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    namespace = {"__name__": "__main__",
                 "__sandbox_sh__": lambda command: _sh(command, 600)}
    # The text protocol's tools are callable from code too; fhda's kernel, which the
    # published protocol (REPL_TOOLS=0) reproduces, defines neither.
    if os.environ.get("REPL_TOOLS", "1") == "1":
        namespace.update(submit_answer=submit_answer, list_workdir=list_workdir)
    try:
        import pandas as pd
        pd.set_option("display.max_rows", 40)
        pd.set_option("display.max_columns", 30)
        pd.set_option("display.width", 200)
    except Exception:
        pass
    signal.signal(signal.SIGALRM, _alarm)
    for line in proto_in:
        if not line.strip():
            continue
        request = json.loads(line)
        timeout = int(request.get("timeout", 300))
        error = False
        with tempfile.TemporaryFile(mode="w+b") as capture:
            sys.stdout.flush()
            sys.stderr.flush()
            saved = os.dup(1), os.dup(2)
            os.dup2(capture.fileno(), 1)
            os.dup2(capture.fileno(), 2)
            signal.alarm(timeout)
            try:
                run_cell(request["code"], namespace, timeout)
            except CellTimeout:
                error = True
                print(f"error: the cell timed out after {timeout}s")
            except subprocess.TimeoutExpired:
                error = True
                print(f"error: the shell command timed out after {timeout}s")
            except _RFailed:
                error = True
            except ShellFailed as failed:
                error = True
                print(f"error: the cell's script exited with status {failed.args[0]}")
            except BaseException as err:
                error = True
                # Show the cell's frames only, as a notebook does, not this kernel's.
                tb = err.__traceback__
                while tb is not None and tb.tb_frame.f_code.co_filename == __file__:
                    tb = tb.tb_next
                if isinstance(err, SyntaxError):
                    tb = None
                traceback.print_exception(type(err), err, tb, limit=6)
            finally:
                signal.alarm(0)
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved[0], 1)
                os.dup2(saved[1], 2)
                os.close(saved[0])
                os.close(saved[1])
            capture.seek(0)
            output = capture.read().decode("utf-8", errors="replace")
        output = re.sub(r"\x1b\[[0-9;]*m", "", output).strip()
        submitted = SUBMITTED[0] if SUBMITTED else None
        SUBMITTED.clear()
        proto_out.write(json.dumps({"output": truncate(output) or "(no output)",
                                    "error": error, "submitted": submitted}) + "\n")
        proto_out.flush()


if __name__ == "__main__":
    main()
