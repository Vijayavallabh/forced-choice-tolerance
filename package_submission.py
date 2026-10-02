#!/usr/bin/env python3
"""Build the source and evidence bundles with explicit file allowlists.

Internal development notes and archive/ are deliberately excluded: they are
working documents, not evidence, and an allowlist keeps them out by
construction.

    python package_submission.py                  # anonymous bundles, for review
    python package_submission.py --camera-ready   # the de-anonymized paper
    python package_submission.py --dry-run        # list and scan; write nothing
    python package_submission.py --out DIR        # write the bundles into DIR

Every mode refuses to package a member that holds a secret or names this host.
The anonymous mode also refuses any member that names an author, and any code,
doc or LaTeX file that narrates an earlier assessment round. ``--camera-ready``
allows the author's name and e-mail in main.tex and the public documents
(README.md, VERIFICATION.md, the licenses, the notices, MODEL_OUTPUTS.md,
CITATION.cff and provenance/), and keeps every other check for the code, the
results and the trajectories.

Run it with the Python 3.12 environment of requirements.txt (on the machine
that built the paper, /opt/conda/bin/python). The bundle's self-test runs its
tests with this same interpreter (sys.executable).
"""
import argparse
import sys
import zipfile
from pathlib import Path

_args = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
_args.add_argument("--camera-ready", action="store_true",
                   help="allow the author's name and e-mail in main.tex and the public docs")
_args.add_argument("--dry-run", action="store_true",
                   help="build the file lists and run every scan; write no bundle")
_args.add_argument("--out", type=Path, default=None,
                   help="directory for the bundles (default: submission/)")
ARGS = _args.parse_args()

if sys.version_info < (3, 12):
    raise SystemExit(
        f"refusing to package: {sys.executable} is Python {sys.version.split()[0]}, and the "
        f"code needs 3.12 (extraction_verdicts.py does not parse before it). Run this with the "
        f"environment of requirements.txt, e.g. /opt/conda/bin/python package_submission.py.")

root = Path(__file__).resolve().parent
out = ARGS.out.resolve() if ARGS.out else root / "submission"
if not ARGS.dry_run:
    out.mkdir(parents=True, exist_ok=True)

# every figure main.tex includes, so the LaTeX source compiles on its own (a hand-kept list
# missed the figures added to the main text)
import re as _re_src  # noqa: E402
_figures = sorted({f if f.endswith(".pdf") else f"{f}.pdf" for f in _re_src.findall(
    r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", (root / "main.tex").read_text(encoding="utf-8"))})
_missing = [f for f in _figures if not (root / f).exists()]
if _missing:
    raise SystemExit(f"refusing to package: main.tex includes {_missing}, which do not exist")
source_files = ["main.tex", "refs.bib", "neurips_2026.sty"] + _figures

code_files = [
    "audit_provenance.py", "authorship_causal.py", "authorship_contrast.py", "bio_to_jsonl.py", "benchmark_census.py", "no_data_baseline.py",
    "repair_frontier.py", "frontier_align.py", "frontier_validity.py",
    "item_census.py", "estimator_simulation.py", "fetch_mmlu_questions.py",
    "option_artifacts.py", "text_artifacts.py", "mcq_audit.py", "fetch_external.py",
    "choice_model.py", "no_data_probe.py", "probe_analysis.py", "rule_solvers.py",
    "icl_probe.py", "icl_analysis.py", "channel_attribution.py",
    "scoring_validation.py", "channel_survey.py", "fetch_survey.py",
    "power_analysis.py", "make_figures.py", "make_manifest.py",
    "temporal_eligibility.py", "temporal_manifest.schema.json",
    "validate_artifact.py", "package_submission.py",
    "audit_calibration.py", "repair_search_ablation.py",
    "held_out_orderings.py", "survey_to_jsonl.py", "learned_probe.py",
    "placebo_contrast.py", "joint_uniformity.py",
    "exchangeable_repair.py", "key_identity.py",
    "keyblind_operators.py", "learned_probe_nonlinear.py", "subset_guarantee.py",
    "baseline_power.py", "published_baselines.py", "temporal_recovery.py",
    "temporal_test.py", "writer_theory.py", "frontier_robustness.py",
    "agentic_probe.py", "agentic_mechanism.py", "text_options.py",
    "rank_dependent_fit.py", "bootstrap_calibration.py", "arm_intervals.py",
    "rank_effect_intervals.py", "build_placebo_arm.py", "build_set_control.py", "release_drift.py",
    "page_budget.py", "scale_grid.py", "fetch_big_models.py", "rebuild_arms.py",
]

# The sweeps as they were launched, so the exact flags behind every arm are on
# the record rather than reconstructed from a results file's ``system`` string.
#
# Code and sweeps are globbed rather than listed. The allowlist above was
# written for the *documents*, where an allowlist is the point -- working notes
# stay out by construction -- and extending that habit to code meant every new
# analysis had to be remembered twice. It was not: at one point nine scripts
# behind the paper's lead result, including ``published_repair.py``, sat outside
# the bundle while the paper named them, so the artifact could not have
# reproduced the number in its own abstract. A glove that fits every finger.
# The same argument applies to the shell: this globbed ``run_*.sh`` and so
# shipped eighteen sweeps but not ``chain_free.sh``, which is the script that
# produced ``results/free_response_mmlupro.json`` -- a number the paper reports.
# The naming convention is not the criterion; being a script here is.
code_files += sorted(
    p.name for p in list(root.glob("*.py")) + list(root.glob("*.sh"))
    if p.name not in set(code_files))

doc_files = ["README.md", "VERIFICATION.md",
             "requirements.txt", "requirements-gpu.txt", "requirements-fetch.txt",
             # the registered plan, and its results below it: the replication is checkable
             # only if what was registered ships beside what it found
             "PREREGISTRATION.md",
             # what may be done with all of it, and whose it is
             "LICENSE", "LICENSE-DATA", "THIRD_PARTY_NOTICES.md", "MODEL_OUTPUTS.md",
             "CITATION.cff"]

data_files = [
    "data/bixbench.jsonl", "data/README.upstream.md", "data/LICENSE.txt", "data/NOTICE",
    "data/external/PROVENANCE.json", "data/external/labbench/LICENSE",
    "data/external/zero_shot_summary_v15.json", "data/external/zero_shot_summary_v10.json",
    "data/external/survey/PROVENANCE.json",
]

result_files = [
    "results/audit.json", "results/power.json", "results/temporal_manifest.json",
    "results/benchmark_census.json", "results/no_data_baseline.json",
    "results/option_artifacts.json", "results/text_artifacts.json",
    "results/mcq_audit.json", "results/repaired/bixbench_v15_repaired.jsonl",
    "results/choice_model.json", "results/choice_model_calibration.json",
    "results/no_data_probe.json", "results/no_data_probe_rules.json",
    "results/scoring_validation.json", "results/channel_survey.json",
    "results/choice_model_mmlu.json", "results/no_data_probe_mmlu.json",
    "results/mcq_audit_mmlu_pro.json", "results/choice_model_mmlupro.json",
    "results/no_data_probe_mmlupro.json", "results/icl_probe.json",
    "results/mcq_audit_multi.json", "results/mcq_audit_mmlu_pro_multi.json",
    "results/mcq_audit_mmlu_pro_isolation.json",
    "results/channel_attribution.json", "results/channel_attribution_mmlupro.json",
    "results/channel_attribution_bix_four.json",
    "results/repaired/bixbench_v15_repaired_multi.jsonl",
    "results/audit_calibration.json", "results/audit_calibration_k10.json",
    "results/repair_search_ablation.json",
    "results/channel_attribution_bix_nearest.json",
    "results/channel_attribution_bix_construct.json",
    "results/channel_attribution_mmlupro_construct.json",
    "results/held_out_orderings.json",
    "results/mcq_audit_mmlu_pro_lexicographic.json",
    "results/channel_attribution_mmlupro_construct_heldout.json",
    "results/held_out_survey.json", "results/placebo_contrast.json",
    "results/learned_probe.json", "results/learned_probe_mmlupro.json",
    "results/joint_uniformity.json", "results/joint_uniformity_mmlupro.json",
    "results/mcq_audit_mmlu.json", "results/mcq_audit_mmlu_tie.json",
    "results/held_out_mmlu.json", "results/held_out_mmlu_tie.json",
    "results/mmlu_tie_seeds.json",
    "results/learned_probe_mmlu.json",
    "results/mcq_audit_mmlu_pro_tie.json", "results/held_out_mmlu_pro_tie.json",
    "results/repaired/mmlu_pro_repaired_tie.jsonl.gz",
    "results/repaired/mmlu_repaired_multi.jsonl.gz",
    "results/repaired/mmlu_repaired_tie.jsonl.gz",
    "results/mcq_audit_medmcqa.json", "results/exchangeable_medmcqa.json",
    "results/key_identity_medmcqa.json", "results/learned_probe_medmcqa.json",
    "results/repaired/medmcqa_matched_released.jsonl.gz",
    "results/repaired/medmcqa_matched_repaired.jsonl.gz",
    "results/repaired/medmcqa_exchangeable.jsonl.gz",
    "results/key_identity_aqua_rat.json", "results/key_identity_medqa_usmle.json",
    "results/key_identity_sciq.json", "results/key_identity_arc_challenge.json",
    "results/key_identity_openbookqa.json",
    "results/authorship_causal.json", "results/authorship_contrast.json",
    "results/repair_frontier.json", "results/frontier_alignment.json",
    "results/key_identity_frontier.json", "results/learned_probe_frontier.json",
    "results/frontier_validity.json", "results/item_census.json",
    "results/estimator_simulation.json", "results/mmlu_questions.json",
    "results/key_identity_aug_generated.json",
    "results/key_identity_bixbench_v15.json",
    "results/key_identity_labbench_seqqa.json",
    "results/exchangeable_mmlu.json", "results/learned_probe_exchangeable.json",
    "results/key_identity_mmlu.json", "results/held_out_exchangeable.json",
    "results/key_identity_mmlu_pro.json", "results/exchangeable_mmlu_pro.json",
    "results/learned_probe_mmlu_pro_exchangeable.json",
    "results/audit_calibration_594.json",
    "results/subset_guarantee.json", "results/baseline_power.json",
    "results/writer_theory.json", "results/writer_price.json",
    "results/writer_price_qwen7b.json", "results/writer_price_llama8b.json",
    "results/writer_price_qwen14b.json",
    "results/frontier_robustness.json",
    "results/frontier_holdout_set.json", "results/frontier_holdout_one_option.json",
    "results/frontier_wrongstep_clean_set.json",
    "results/frontier_wrongstep_clean_one_option.json",
    "results/frontier_wrongstep_collided_set.json",
    "results/frontier_wrongstep_collided_one_option.json",
    "results/rank_dependent_fit.json", "results/rank_dependent_fit_bix.json",
    "results/text_options.json", "results/text_options_naive.json",
    "results/provenance_figure.json",
    "results/agentic_bixarm_placebo_qwen14b.json",
    "results/agentic_bixarm_repaired_qwen14b.json",
    "results/agentic_bixbench_qwen7b.json", "results/agentic_bixbench_qwen14b.json",
    "results/agentic_bixbench_llama8b.json",
    "results/agentic_mmlupro_qwen7b.json", "results/agentic_mmlupro_qwen14b.json",
    "results/agentic_mmlupro_llama8b.json",
    "results/agentic_bixbench_qwen14b_dump.json", "results/agentic_mechanism.json",
    "results/bootstrap_calibration.json", "results/arm_intervals.json",
    "results/rank_effect_intervals.json", "results/agentic_arms_qwen14b.json",
    "results/agentic_arms_qwen32b.json", "results/agentic_placebo_mmlupro.json",
    "results/agentic_bixall_qwen14b.json", "results/agentic_replication_mmlupro_qwen14b.json",
    "results/release_drift.json", "results/agentic_arms_phi4.json",
    "results/agentic_bixarms_qwen14b.json", "results/agentic_grid.json",
    "results/scale_grid.json", "results/scale_grid_bixbench.json",
    "results/agentic_bixgrid.json", "results/agentic_arms_qwen72b.json", "results/agentic_arms_llama70b.json", "results/agentic_mmlupro_qwen14b_rerun.json",
    "results/agentic_mechanism_arms.json",
    "results/agentic_bixbench_qwen32b.json", "results/agentic_mmlupro_qwen32b.json",
    "results/published_baselines.json", "results/temporal_recovery.json",
    "results/temporal_test_litqa2.json",
    "results/nonlinear_mmlu_chars_set.json", "results/nonlinear_mmlupro_chars_set.json",
    "results/nonlinear_mmlu_chars_option.json",
    "results/nonlinear_mmlupro_chars_option.json",
    "results/nonlinear_frontier_chars_set.json",
    "results/nonlinear_frontier_designed_mlp.json",
    "results/learned_probe_frontier_nofilter.json",
    "results/learned_probe_frontier_robustness.json",
    "results/key_identity_frontier_robustness.json",
    "results/frontier_alignment_robustness.json",
    "results/frontier_alignment_nofilter.json",
    "results/keyblind_wrong_step.json", "results/keyblind_wrong_step_strict.json",
    "results/keyblind_wrong_step_llama.json", "results/keyblind_imitation_llama.json",
    "results/keyblind_imitation_phi.json", "results/keyblind_imitation_qwen7b.json",
    "results/source_dates.json", "results/item_dates.json",
]

evidence_files = ["sources/t3_biorxiv.json", "sources/t3.xml", "sources/gse145926.txt"]


def relative(pattern):
    return [str(p.relative_to(root)) for p in sorted(root.glob(pattern)) if p.is_file()]


artifact_files = sorted(set(
    source_files + code_files + doc_files + data_files + result_files + evidence_files
    + relative("sources/api_cache/*.json")
    + relative("data/external/zero_shot_v15/*.csv")
    + relative("data/external/zero_shot_v10/*.csv")
    + relative("data/external/labbench/*.jsonl")
    + relative("data/external/other/*")
    + relative("tests/test_*.py")
    # The aligned frontier arms. These live under build/ with the LaTeX
    # intermediates, which is why they were never shipped -- but they are not
    # intermediates: validate_artifact.py, frontier_robustness.py and four
    # tests all read them, so the bundle could not run its own validator.
    + relative("build/mmlu_frontier_*_aligned.jsonl")
    # Per-decision probe outcomes: these are what make every reported statistic
    # re-derivable without a GPU, so they belong in the evidence bundle.
    + relative("results/probe_raw/*.jsonl.gz")
    + relative("results/probe_rules/*.jsonl.gz")
    + relative("data/external/survey/*.jsonl.gz")
    + relative("results/probe_mmlu/*.jsonl.gz")
    + relative("results/probe_mmlupro/*.jsonl.gz")
    + relative("results/probe_icl/*.jsonl.gz")
    + relative("results/probe_bix_four/*.jsonl.gz")
    + relative("results/probe_bix_nearest/*.jsonl.gz")
    + relative("results/probe_bix_construct/*.jsonl.gz")
    + relative("results/probe_mmlupro_construct/*.jsonl.gz")
    + relative("results/probe_mmlupro_placebo/*.jsonl.gz")
    + relative("results/probe_bix_placebo/*.jsonl.gz")
    + relative("results/probe_mmlupro_panel/*.jsonl.gz")
    + relative("results/probe_mmlu_construct/*.jsonl.gz")
    + relative("results/probe_mmlu_tie/*.jsonl.gz")
    # the agent's own rollouts, so the mechanism split re-runs without a GPU
    + relative("results/agentic_dumps/*.jsonl.gz")
    # and the membership scores, for the same reason: producing them needs a
    # GPU per model and nothing else in the arm does, so without these the
    # grid is a table a reader has to take on trust
    + relative("results/agentic_dumps/*.json.gz")
    # The with-data round: every agent trajectory and every reader/judge reply,
    # so app:withdata's numbers re-derive with no GPU and no server; the items
    # and arms its analyses read, which live under build/ like the frontier's;
    # the notebook image's recipe; and the upstream BixBench/fhda sources the
    # agent's prompts and graders are taken from verbatim.
    + relative("results/agent_runs/*.gz")
    + relative("results/agent_runs/*.json")
    + relative("build/bixbench_v10*.jsonl")
    + relative("build/bixbench_numeric_*.jsonl")
    # the question lists the closed agent was given
    + relative("build/openai/v15_numeric_question_ids.txt")
    + relative("build/openai/v15_nonnumeric_question_ids.txt")
    # The prompt factorial, the grids and the control census read these item
    # and control files; the key-marginal rebuild and the collision bound read
    # the MMLU question file.
    + relative("build/bixbench_v15.jsonl")
    + relative("build/bixbench_all_clean.jsonl")
    + relative("build/mmlu_pro_matched_*.jsonl")
    + relative("build/mmlu_questions.jsonl")
    + relative("build/mmlu_frontier_key_marginal_symmetric_aligned.jsonl")
    + relative("build/mmlu_frontier_*_sym456.jsonl")
    + relative("sandbox_image/*")
    + relative("sources/bixbench_49311180/*")
    # verbatim excerpts of the three sources the paper cites for BixBench's use in
    # 2026, which the validator checks the quoted numbers against
    + relative("sources/cited/*")
    # The build/ inputs above as a clone gets them: data/derived/ with its
    # SHA256SUMS, and restore_build_inputs.py (a root script, so globbed with the
    # code) to put them back. The self-test below runs it in the unpacked bundle.
    + relative("data/derived/**/*")
    # the archived record of the pushes that date the registered plan
    + relative("provenance/*")
))

# Every results file, for the same reason: sixteen were outside the bundle,
# among them the ones behind the published-arm repair, the numeric/text cell
# split, the membership grid and the calibrated frontier.
artifact_files = sorted(set(artifact_files) | {
    str(q.relative_to(root)) for q in root.glob("results/*.json")}
    # compressed results: the published v1.0 runs the replication reads, and the new
    # v1.5 run sets' answers, the runs the wall clock cut, and the record of their reruns
    | {str(q.relative_to(root)) for q in root.glob("results/*.json.gz")}
    # and the reads of the registered runs by BixBench's reader, one row per run
    | {str(q.relative_to(root)) for q in root.glob("results/*.jsonl.gz")}
    | {str(q.relative_to(root)) for q in root.glob("results/agent_runs_replication/*.gz")}
    | {str(q.relative_to(root)) for q in root.glob("results/agent_runs_replication/*.json")}
    # the closed models: gpt-4o's reads without the data, and gpt-5.1's runs with it and their grades
    | {str(q.relative_to(root)) for q in root.glob("results/openai_nodata/*.gz")}
    | {str(q.relative_to(root)) for q in root.glob("results/strong_agent/*.gz")}
    # the four more runs of Qwen3-235B-A22B, apart from the pre-specified test's
    | {str(q.relative_to(root)) for q in root.glob("results/agent_runs_q235_extra/*.gz")}
    # the other designs' option sets and the code-free graders' reads of them
    | {str(q.relative_to(root)) for q in root.glob("results/option_design/*.gz")}
    # the current agents' trajectories and gpt-4o's replies
    | {str(q.relative_to(root)) for q in root.glob("results/frontier_agents/*.gz")})

missing = [f for f in artifact_files if not (root / f).exists()]
if missing:
    raise SystemExit(f"refusing to package; missing files: {missing}")

# And the paper is the authority on what has to be in there. Anything it names
# in \texttt{} that exists here as a script must ship with it; a bare name with
# a path in front of it (``bixbench/prompts.py``) is somebody else's file.
import re

named = {m for m in re.findall(r"\\texttt\{([a-z0-9_\\]+\.py)\}",
                             (root / "main.tex").read_text())}
named = {n.replace("\\_", "_") for n in named}
unshipped = sorted(n for n in named
                   if (root / n).exists() and n not in set(artifact_files))
if unshipped:
    raise SystemExit(
        f"refusing to package; main.tex names these and they are not in the "
        f"bundle: {unshipped}")

# Every model in the membership grid needs its score cache in the bundle, or
# that row cannot be re-derived without the GPU that produced it.
import json

for grid_name, slug in (("results/option_membership_grid.json", ""),
                        ("results/option_membership_grid_bix.json", "_bixall")):
    grid = root / grid_name
    if not grid.exists():
        continue

    def cache_name(model, slug=slug):
        # option_membership.py's own rule, character for character: it strips
        # "-Instruct" and nothing else, so gemma-3-4b-it and
        # Phi-3.5-mini-instruct keep their suffixes. Normalising further here
        # made this guard fail on two models whose caches were present.
        tag = model.split("/")[-1].replace("-Instruct", "")
        return (f"results/agentic_dumps/option_membership_scores{slug}.json.gz"
                if tag == "Qwen2.5-14B" else
                f"results/agentic_dumps/"
                f"option_membership_scores_{tag}{slug}.json.gz")

    uncached = sorted(m for m in json.loads(grid.read_text())
                      if cache_name(m) not in set(artifact_files))
    if uncached:
        raise SystemExit(
            f"refusing to package; {grid_name} has rows whose scores are not "
            f"in the bundle: {uncached}")

# --- what may not ship --------------------------------------------------------
# Nine shipped scripts named the machine they ran on -- cache and checkout paths
# whose components are an institution and an account -- and every with-data
# trajectory recorded the absolute paths it was given. What would identify the
# host or an author is derived here from the host, never listed: a list of it
# would ship in this very file. Every member is scanned, gzip members
# decompressed.
#
# For review the bundle is anonymous, so an author's name or e-mail is refused
# everywhere. The camera-ready paper carries them, so --camera-ready allows them
# in main.tex and the public documents below, and nowhere else: code, results
# and trajectories that name an author, the checkout or this host are refused in
# both modes, and so is anything shaped like a secret (make_public_export.py's
# patterns, which include the keys and endpoints configured on this host).
import gzip as _gzip  # noqa: E402
import os as _os  # noqa: E402
import subprocess as _subprocess  # noqa: E402

import make_public_export as _mpe  # noqa: E402

PUBLIC_DOCS = {"main.tex", "README.md", "VERIFICATION.md", "LICENSE", "LICENSE-DATA",
               "THIRD_PARTY_NOTICES.md", "MODEL_OUTPUTS.md", "CITATION.cff",
               "data/external/labbench/LICENSE"}


def _public_doc(member):
    # provenance/ is the record of the pushes that date the registered plan, and
    # a push record names who pushed
    return member in PUBLIC_DOCS or member.startswith("provenance/")


def _git_config(key):
    return _subprocess.run(["git", "config", key], cwd=root, capture_output=True,
                           text=True).stdout.strip()


_GENERIC = {"home", "mnt", "backup", "users", "user", "data", "work", "projects", "src", "opt", "srv",
            "tmp", "var", "media", "scratch", "storage", "volumes"}
_host = {str(root.parent)}
_host |= {part for part in root.parts[1:-1] if len(part) >= 4 and part.lower() not in _GENERIC}
if _os.path.expanduser("~") not in ("/", "/root"):
    _host.add(_os.path.expanduser("~"))
# Runs made on another host carry that host's names, which nothing here can derive;
# they are passed in, comma-separated, rather than written into a file that ships.
_host |= set(_os.environ.get("AGENTICLS_LEAK_NEEDLES", "").split(","))
_host = {n.lower() for n in _host if n}
_author = {n.lower() for n in (_git_config("user.name"), _git_config("user.email")) if len(n) >= 4}
# A path component that is part of an author's name or e-mail (the home directory
# named after the author, the institution in the e-mail's domain) identifies the
# author, not the host, and is treated as the author's.
_identity = _author | {n for n in _host if any(n in a for a in _author)}
_host -= _identity
_configured = _mpe.host_needles(root)

_leaking, _secret = {}, {}
for _member in sorted(set(source_files) | set(artifact_files)):
    _raw = (root / _member).read_bytes()
    if _member.endswith(".gz"):
        _raw = _gzip.decompress(_raw)
    _text = _raw.decode("utf-8", errors="ignore")
    _low = _text.lower()
    _found = [n for n in _host if n in _low]
    if not (ARGS.camera_ready and _public_doc(_member)):
        _found += [n for n in _identity if n in _low]
    if _found:
        _leaking[_member] = len(_found)
    _fatal, _ = _mpe.scan_text(_member, _text, _configured, narration=False)
    if _fatal:
        _secret[_member] = sorted({category for category, _ in _fatal})
if _leaking:
    raise SystemExit(f"refusing to package; {len(_leaking)} member(s) name the host or an author "
                     f"(the strings themselves are not printed): {sorted(_leaking)}")
if _secret:
    raise SystemExit(f"refusing to package; {len(_secret)} member(s) hold a secret or a local "
                     f"path (not printed): {_secret}")

# For review, what ships must not narrate an earlier round of assessment: who
# asked for an analysis, or what the text used to say, tells a reader the paper
# has been read before. Scanned in the files this repository writes -- code,
# docs, LaTeX and the JSON the analyses write -- and not in data or vendored
# upstream files, where the word is question text or someone else's. The
# camera-ready paper may acknowledge its referees, so --camera-ready scans only
# the code and the results.
import re as _re  # noqa: E402

_HISTORY = _re.compile(r"\breview(?:er)?s\b|\breviewer\b|\brebuttal\b|\bR[0-9]-?Q[0-9]+\b"
                       r"|\b(?:earlier|previous|prior)\s+drafts?\b", _re.I)
_VENDORED = ("sources/bixbench_49311180/", "data/external/", "sandbox_image/")
_WRITTEN = (".py", ".sh") if ARGS.camera_ready else (".py", ".sh", ".md", ".tex")
_narrating = {}
for _member in sorted(set(source_files) | set(artifact_files)):
    _analysis_json = _member.startswith("results/") and _member.count("/") == 1 \
        and _member.endswith(".json")
    if _member.startswith(_VENDORED) or not (_member.endswith(_WRITTEN) or _analysis_json):
        continue
    _hits = _HISTORY.findall((root / _member).read_text(encoding="utf-8", errors="ignore"))
    if _hits:
        _narrating[_member] = len(_hits)
if _narrating:
    raise SystemExit(f"refusing to package; {len(_narrating)} member(s) narrate an earlier "
                     f"assessment or a superseded text: {_narrating}")

if ARGS.dry_run:
    print(f"dry run ({'camera-ready' if ARGS.camera_ready else 'anonymous'}): "
          f"latex_source.zip would hold {len(source_files)} files and "
          f"supplementary_artifact.zip {len(artifact_files)} files, "
          f"{sum((root / f).stat().st_size for f in artifact_files) // 1024} KB before "
          f"compression; every scan passed; nothing written")
    raise SystemExit(0)

for name, files in (("latex_source.zip", source_files),
                    ("supplementary_artifact.zip", artifact_files)):
    with zipfile.ZipFile(out / name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in files:
            archive.write(root / member, member)
    print(f"{name}: {len(files)} files, {(out / name).stat().st_size // 1024} KB")

# --- and the bundle has to pass its own tests -------------------------------
# Every guard above asks whether a named file is present. None of them can ask
# whether the bundle *works*, and it did not: unpacked, it failed two tests and
# skipped seven more, because ``chain_free.sh`` was outside the glob, the
# aligned frontier arms sat under build/, and three tests plus
# membership_grid.control_vocabulary read the uncompressed working copies of
# dumps that ship gzipped. Each of those reads green in this tree and is
# invisible from inside it. A skip is the dangerous case: it is not a failure,
# so anyone who runs the suite sees a pass and a number that was never
# checked. So the suite runs against the unpacked zip, and a skip is a defect
# unless it is named here with its reason.
import subprocess
import tempfile

ALLOWED_SKIPS = {
    # The orphan half of the refs.bib check reads archive/original/main.tex,
    # the superseded draft. That is a working document, deliberately excluded,
    # and whether refs.bib has an orphan is this tree's hygiene, not evidence.
    "archive/original/*.tex not present, so 'unused' is unreadable",
    # build/mmlu_questions.jsonl is 7 MB of upstream MMLU (1.9 MB compressed). It
    # is a re-fetch, not evidence: fetch_mmlu_questions.py ships and the skip
    # names it.
    "build/mmlu_questions.jsonl not built; run: python3 fetch_mmlu_questions.py",
}

with tempfile.TemporaryDirectory() as tmp:
    with zipfile.ZipFile(out / "supplementary_artifact.zip") as archive:
        archive.extractall(tmp)
    # A clone gets its build/ inputs from data/derived, so the bundle is tested
    # that way too: the restore checks every checksum, rebuilds what it rebuilds,
    # and refuses if a bundled build/ file differs from the derived copy.
    restore = subprocess.run([sys.executable, "restore_build_inputs.py"], cwd=tmp,
                             capture_output=True, text=True, timeout=600)
    if restore.returncode != 0:
        raise SystemExit(f"refusing to package; restore_build_inputs.py failed in the unpacked "
                         f"bundle:\n{restore.stdout}{restore.stderr}")
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-rs",
         "-p", "no:cacheprovider"],
        cwd=tmp, capture_output=True, text=True, timeout=3600)
    if "No module named pytest" in run.stderr:
        raise SystemExit(
            f"refusing to package; {sys.executable} has no pytest, so the "
            f"bundle's own tests did not run. Re-run with an interpreter that "
            f"has it rather than shipping an unexercised bundle.")

    skips = {}
    for line in run.stdout.splitlines():
        if line.startswith("SKIPPED") and ": " in line:
            where, reason = line.split(": ", 1)
            skips.setdefault(reason.strip(), []).append(where.strip())
    unexpected = {r: w for r, w in skips.items() if r not in ALLOWED_SKIPS}

    if run.returncode != 0 or unexpected:
        tail = "\n".join(run.stdout.splitlines()[-25:])
        detail = "".join(f"\n  {r}  <- {w}" for r, w in sorted(unexpected.items()))
        raise SystemExit(
            f"refusing to package; the unpacked bundle does not pass its own "
            f"tests.{' Skipped for inputs it does not carry:' if unexpected else ''}"
            f"{detail}\n\n{tail}")
    print(f"bundle self-test: {run.stdout.splitlines()[-1].strip()}")

if not (root / "build/main.pdf").exists():
    raise SystemExit("the bundles are written, but build/main.pdf is missing: compile the paper "
                     "(see README) and rerun to add the PDF")
# The camera-ready PDF is named for the paper's title.
pdf_name = "forced_choice_is_not_a_tolerance.pdf" if ARGS.camera_ready else "stale_science_agenticls.pdf"
(out / pdf_name).write_bytes((root / "build/main.pdf").read_bytes())
print(f"Wrote the PDF, the LaTeX source ZIP and the supplementary artifact ZIP to {out}.")
