"""What a fresh clone needs: the build/ inputs, the calibration and the imports.

A clone has no build/, which git ignores, and the tests, the validator and the
analyses open item files there. data/derived/ carries them with their SHA-256,
and restore_build_inputs.py puts them back. These tests check that the shipped
copies match their checksums, that a restore into an empty directory yields
every file the test suite reads, byte for byte, that the scripts which used to
fall back silently now stop instead, and that the public export takes only what
it lists and tells this work's machines' paths from paths an agent invented.
"""
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import restore_build_inputs as rb  # noqa: E402

# The build/ files the test suite opens (traced with an audit hook over a full
# run). A restore must produce every one of them.
TEST_INPUTS = [
    "bixbench_numeric_q.jsonl", "bixbench_numeric_placebo.jsonl",
    "bixbench_numeric_repaired.jsonl", "bixbench_numeric_placebo_digits.jsonl",
    "bixbench_numeric_repaired_digits.jsonl", "bixbench_v10.jsonl", "bixbench_v10_items.jsonl",
    "bixbench_v10_numeric_released.jsonl", "bixbench_v10_numeric_placebo.jsonl",
    "bixbench_v10_repaired.jsonl", "bixbench_v10_numeric_placebo_digits.jsonl",
    "bixbench_v10_repaired_digits.jsonl",
    "mmlu_frontier_exchangeable_aligned.jsonl", "mmlu_frontier_imitation_aligned.jsonl",
    "mmlu_frontier_key_marginal_aligned.jsonl", "mmlu_frontier_key_marginal_near_aligned.jsonl",
    "mmlu_frontier_rank_uniform_aligned.jsonl", "mmlu_frontier_released_matched_aligned.jsonl",
    "mmlu_questions.jsonl",
]


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def test_the_derived_inputs_match_their_checksums():
    assert rb.verify() == []
    listed = rb.manifest()
    assert len(listed) >= len(TEST_INPUTS)
    for name, digest in listed.items():
        assert sha256((rb.DERIVED / name).read_bytes()) == digest, name


def test_the_manifest_is_what_sha256sum_checks():
    for line in rb.SUMS.read_text().splitlines():
        digest, _, name = line.partition("  ")
        assert len(digest) == 64 and name and not name.startswith("/"), line
        assert ".." not in Path(name).parts, line


def test_only_large_files_ship_compressed():
    for name in rb.manifest():
        size = (rb.DERIVED / name).stat().st_size
        if name.endswith(".gz"):
            assert len(gzip.decompress((rb.DERIVED / name).read_bytes())) > 1 << 20, name
        else:
            assert size <= 1 << 20, f"{name} is {size} bytes and ships uncompressed"


def test_the_numeric_subset_regenerates_byte_for_byte():
    assert rb.numeric_q_bytes() == rb.payload(rb.NUMERIC_Q)
    rows = [json.loads(line) for line in rb.payload(rb.NUMERIC_Q).decode().splitlines()]
    assert len(rows) == 105
    assert all(list(r) == ["ideal", "distractors", "question", "cluster"] for r in rows)


def test_a_restore_gives_every_file_the_tests_read(tmp_path):
    report = rb.restore(build=tmp_path)
    assert report["conflict"] == []
    written = set(report["written"])
    missing = [name for name in TEST_INPUTS if name not in written]
    assert not missing, f"restore does not produce: {missing}"
    for name, digest in rb.manifest().items():
        assert sha256((tmp_path / rb.restored_name(name)).read_bytes()) == sha256(rb.payload(name))
    for target, (_, digest) in rb.SURVEY.items():
        assert sha256((tmp_path / target).read_bytes()) == digest
    # and a second restore finds everything in place
    again = rb.restore(build=tmp_path)
    assert again["written"] == [] and again["conflict"] == []
    assert len(again["present"]) == len(written)


def test_a_restore_leaves_a_different_file_alone_unless_forced(tmp_path):
    rb.restore(build=tmp_path)
    target = tmp_path / rb.NUMERIC_Q
    target.write_text("edited\n")
    report = rb.restore(build=tmp_path)
    assert report["conflict"] == [rb.NUMERIC_Q]
    assert target.read_text() == "edited\n"
    forced = rb.restore(build=tmp_path, force=True)
    assert forced["written"] == [rb.NUMERIC_Q]
    assert target.read_bytes() == rb.payload(rb.NUMERIC_Q)


def test_a_tampered_derived_file_stops_the_restore(tmp_path):
    derived = tmp_path / "derived"
    derived.mkdir()
    (derived / "SHA256SUMS").write_text(rb.SUMS.read_text())
    for name in rb.manifest():
        (derived / name).parent.mkdir(parents=True, exist_ok=True)
        (derived / name).write_bytes((rb.DERIVED / name).read_bytes())
    (derived / rb.NUMERIC_Q).write_bytes(b"{}\n")
    assert any(rb.NUMERIC_Q in p for p in rb.verify(derived))
    with pytest.raises(SystemExit):
        rb.restore(build=tmp_path / "build", derived=derived)
    assert not (tmp_path / "build").exists()


def run_script(script, *args, cwd):
    return subprocess.run([sys.executable, str(ROOT / script), *args], cwd=cwd,
                          capture_output=True, text=True, timeout=600)


def test_the_calibration_refuses_to_write_without_its_dumps(tmp_path):
    out = tmp_path / "bootstrap_calibration.json"
    run = run_script("bootstrap_calibration.py", "--reps", "1", "--bootstrap", "1",
                     "--output", str(out), cwd=tmp_path)
    assert run.returncode != 0
    assert "refusing to write" in run.stderr
    assert not out.exists()


def test_the_calibration_reads_the_shipped_dumps():
    import bootstrap_calibration as bc

    for _, path, _ in bc.ARMS:
        found = bc.resolve(path)
        assert found is not None, path
        if not (ROOT / path).exists():
            assert found == bc.SHIPPED / (Path(path).name + ".gz")


def test_the_intervals_refuse_to_run_without_a_calibration(tmp_path):
    out = tmp_path / "arm_intervals.json"
    run = run_script("arm_intervals.py", "--calibration", str(tmp_path / "none.json"),
                     "--output", str(out), cwd=ROOT)
    assert run.returncode != 0 and "bootstrap_calibration.py" in run.stderr
    assert not out.exists()
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"reps": 1, "bootstrap": 1, "arms": []}))
    run = run_script("arm_intervals.py", "--calibration", str(empty), "--output", str(out),
                     cwd=ROOT)
    assert run.returncode != 0 and "has no BixBench arm" in run.stderr
    assert not out.exists()


def test_option_membership_imports_without_torch():
    blocker = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('torch', 'transformers'):\n"
        "            raise ModuleNotFoundError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import option_membership\n"
        "print(option_membership.file_slug('build/dumps/qwen14b_bixall_x.jsonl'))\n")
    run = subprocess.run([sys.executable, "-c", blocker], capture_output=True, text=True,
                         timeout=120)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "_bixall"


def path(*parts):
    # built from parts so that this file carries no absolute path for the scan to find
    return "/" + "/".join(parts)


def test_the_export_takes_only_what_is_listed():
    import make_public_export as mpe

    for name in ("restore_build_inputs.py", "make_public_export.py", "package_submission.py",
                 "LICENSE", "LICENSE-DATA", "CITATION.cff", "MODEL_OUTPUTS.md",
                 "THIRD_PARTY_NOTICES.md", "PREREGISTRATION.md", "data/derived/SHA256SUMS",
                 "data/NOTICE", "tests/test_release.py", "provenance/plan_commit.txt"):
        assert mpe.selected(name), name
    for name in ("NOTES.md", "build/bixbench_v15.jsonl", "archive/original/main.tex",
                 "submission/latex_source.zip", "sources/neurips2026.zip",
                 "sources/verification.md", "logs/run.txt", "tests/__pycache__/x.pyc",
                 "submission_text.py", ".claude/settings.json"):
        assert not mpe.selected(name), name


def test_the_export_scan_tells_host_paths_from_invented_ones(monkeypatch):
    import make_public_export as mpe

    monkeypatch.setattr(mpe, "HOST_PATH_HASHES",
                        {hashlib.sha256(path("home", "somebody").encode()).hexdigest()})
    monkeypatch.setattr(mpe, "HOST_TOPS", set())
    fatal, warn = mpe.scan_text("x.py", f"open('{path('home', 'somebody', 'data.csv')}')", {},
                                narration=False)
    assert fatal == [("local path", path("home", "somebody"))] and warn == []
    # an agent's own guess at a home directory, and CI's runner, are not this work's machines
    text = f"cd {path('home', 'ubuntu', 'work')} && ls {path('home', 'runner')}/x"
    fatal, warn = mpe.scan_text("x.py", text, {}, narration=False)
    assert fatal == [] and warn == [("path", path("home", "ubuntu"))]
    # nor is a URL's path
    assert mpe.scan_text("x.py", "https://example.org" + path("home", "about"), {},
                         narration=False) == ([], [])
    scratch = path("tmp", "cla" + "ude-1", "x")
    fatal, _ = mpe.scan_text("x.py", scratch, {}, narration=False)
    assert fatal == [("local path", path("tmp", "cla" + "ude-1"))]


def test_the_export_scan_describes_a_secret_without_repeating_it():
    import make_public_export as mpe

    key = "s" + "k-" + "A1" * 16
    fatal, _ = mpe.scan_text("x.py", f"KEY = '{key}'", {}, narration=False)
    assert fatal and all(category == "secret" for category, _ in fatal)
    assert key not in repr(fatal)
    # the stand-ins the API tests use are not secrets
    assert mpe.scan_text("x.py", "key = 'azure-key-0123456789'", {}, narration=False)[0] == []


def test_the_release_scripts_pass_their_own_scan():
    import make_public_export as mpe

    for name in ("make_public_export.py", "package_submission.py", "restore_build_inputs.py",
                 "tests/test_release.py"):
        fatal, _ = mpe.scan_text(name, (ROOT / name).read_text(encoding="utf-8"), {})
        assert fatal == [], (name, fatal)
