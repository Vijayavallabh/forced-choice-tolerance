#!/usr/bin/env python3
"""Check that the released results support every number the manuscript states.

This is the guard against a manuscript drifting away from its artifact: each
assertion below pins a claim in ``main.tex`` to the JSON that produced it, so a
re-run that changes a result fails here rather than silently disagreeing with
the paper.
"""
import gzip
import hashlib
import json
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# AGENTICLS_VALIDATE_COLLECT=1 records every failed check and reports them all at
# the end instead of stopping at the first, which is how pins are updated after an edit.
import os
_COLLECT = os.environ.get("AGENTICLS_VALIDATE_COLLECT") == "1"
_FAILURES = []
if _COLLECT:
    import atexit

    @atexit.register
    def _report_collected():
        for line, message in _FAILURES:
            print(f"FAILED line {line}: {message}")
        print(f"{len(_FAILURES)} failed checks so far")


def check(condition, message):
    if not condition:
        if _COLLECT:
            import inspect
            _FAILURES.append((inspect.currentframe().f_back.f_lineno, message))
            return
        raise AssertionError(message)


# --- schema, manifest, and the pinned dataset identity ---------------------
schema = load("temporal_manifest.schema.json")
manifest = load("results/temporal_manifest.json")
audit = load("results/audit.json")
Draft202012Validator.check_schema(schema)
Draft202012Validator(schema, format_checker=FormatChecker()).validate(manifest)

digest = hashlib.sha256(Path("data/bixbench.jsonl").read_bytes()).hexdigest()
check(digest == audit["dataset_sha256"] == manifest["benchmark"]["sha256"],
      "pinned dataset digest disagrees with the audit or the manifest")
check(len(manifest["tasks"]) == audit["n_capsules"] == 59, "expected 59 capsule records")
check(len({t["task_id"] for t in manifest["tasks"]}) == 59, "duplicate task_id in manifest")
check(audit["n_questions"] == 205, "expected 205 questions")
check(audit["date_resolution"]["publication_by_end_2024_questions"] == 137,
      "expected 137 questions with pre-2025 publication evidence")
check(len(audit["explicit_source_groups"]) == 16, "expected 16 connected identifier groups")

# --- vendored external evidence still matches its recorded digests ---------
provenance = load("data/external/PROVENANCE.json")
for entry in provenance["files"]:
    blob = Path(entry["local_path"]).read_bytes()
    check(hashlib.sha256(blob).hexdigest() == entry["sha256"],
          f"vendored evidence changed: {entry['local_path']}")

# --- the census claim in the abstract and Table 1 --------------------------
census = load("results/benchmark_census.json")
check(census["summary"]["n_files_publishing_any_date"] == 0,
      "a benchmark file now publishes a date; the abstract's claim is stale")
check(census["summary"]["n_benchmark_files"] == 12, "census file count changed")
check(census["summary"]["n_items_total"] == 2558, "census task total changed")

# --- the no-data baselines reproduce the authors' published numbers --------
baseline = load("results/no_data_baseline.json")
published = load("data/external/zero_shot_summary_v15.json")
runs = {r["run"]: r for r in baseline["runs"]}
for name, rec in published.items():
    check(name in runs, f"missing re-analysis of published run {name}")
    check(abs(runs[name]["accuracy"] - rec["accuracy"]) < 1e-9,
          f"{name}: re-analysis does not reproduce the published accuracy")
    check(runs[name]["n_questions_joined"] == 205, f"{name}: join lost questions")

mcq = runs["gpt-4o-grader-mcq-refusal-False"]
check(mcq["ci95_cluster_bootstrap"][0] > 0.25, "no-data MCQ interval no longer excludes chance")
check(runs["gpt-4o-grader-openended"]["accuracy"] < 0.05, "open-ended no-data accuracy changed")

# --- the option artifact, its cross-validation, and its control ------------
artifacts = {b["benchmark"]: b for b in load("results/option_artifacts.json")["benchmarks"]}
v15, v10 = artifacts["BixBench v1.5"], artifacts["BixBench v1.0"]
check(v15["n_numeric_items"] == 105, "v1.5 numeric item count changed")
check(v15["interior_rate_ci95"][0] > 0.5, "v1.5 interior excess is no longer significant")
check(v15["best_single_rank_rule"]["ci95"][0] > 0.25, "the model-free rule no longer beats chance")
# sec:open types the second-smallest rule's score with its interval and the
# uniformity test beside it. Both come from option_artifacts.json, not from the
# survey's cluster bootstrap, which gives a different lower end for the same
# rule; typing one and sourcing the other is exactly the confusion to prevent.
_rule = v15["best_single_rank_rule"]
check(abs(100 * _rule["accuracy"] - 51.4) < 0.06,
      f"sec:open types 51.4% for the second-smallest rule, against "
      f"{100 * _rule['accuracy']:.2f}%")
check(abs(100 * _rule["ci95"][0] - 39.5) < 0.06
      and abs(100 * _rule["ci95"][1] - 63.4) < 0.06,
      f"sec:open types [39.5, 63.4] for that rule, against "
      f"[{100 * _rule['ci95'][0]:.1f}, {100 * _rule['ci95'][1]:.1f}]")
check(abs(v15["chi_square_vs_uniform"]["chi2"] - 51.0) < 0.05
      and v15["chi_square_vs_uniform"]["p_value_upper_bound"] < 1e-10,
      "sec:open types chi2 = 51.0 with p < 1e-10 for BixBench v1.5")

check(abs(v15["cross_validated_rule"]["accuracy"]
          - v15["best_single_rank_rule"]["accuracy"]) < 1e-9,
      "v1.5 cross-validated accuracy no longer equals in-sample accuracy")
check(v10["interior_rate_ci95"][0] > 0.5, "v1.0 interior excess is no longer significant")
check(v10["cross_validated_rule"]["accuracy"] < v10["best_single_rank_rule"]["accuracy"] - 0.1,
      "v1.0 rule now cross-validates; the revision narrative is stale")
control = v10["presentation_order_control"]["chi_square_vs_uniform"]
check(control["p_value_upper_bound"] > 0.05, "presentation-order control is no longer null")

# the rule and the models must be compared on the same items
matched = load("results/option_artifacts.json")["matched_subset_comparison"]
check(matched["n_matched_items"] == v15["n_numeric_items"],
      "matched comparison covers a different item set from the rule")

# --- surface rules are a shadow of the rank channel, not a second one -------
text = load("results/text_artifacts.json")
non_numeric = text["subsets"]["non_numeric"]
check(non_numeric["n_items"] == 100, "non-numeric item count changed")
check(len(text["rules_in_family"]) == 9, "surface rule family size changed")
proxy = text["rank_proxy_check"]
check(abs(proxy["min_share_selecting_largest"] - 0.429) < 0.01
      and abs(proxy["max_share_selecting_largest"] - 0.838) < 0.01,
      "the 43-84% rank-proxy range in \\S5 no longer holds: "
      f"{proxy['min_share_selecting_largest']:.3f}-{proxy['max_share_selecting_largest']:.3f}")

# --- the audit-and-repair tool ----------------------------------------------
mcq = load("results/mcq_audit.json")
check("numeric rule family" in mcq["before"]["coordinates_firing"],
      "the released file's value channel is no longer flagged")
check(mcq["after_repair"]["coordinates_firing"] == [],
      "the repair no longer closes every coordinate")
check(mcq["n_repaired"] == 105, "the repair no longer covers every numeric item")
before_cv = mcq["before"]["numeric_family"]["cv_accuracy"]
after_cv = mcq["after_repair"]["numeric_family"]["cv_accuracy"]
check(abs(before_cv - 0.514) < 0.001 and abs(after_cv - 0.248) < 0.002,
      f"repair effect changed: {before_cv:.3f} -> {after_cv:.3f}")
check(mcq["after_repair"]["numeric_family"]["cv_ci95"][0] <= 0.25,
      "the repaired file's interval no longer covers chance")
before_credit = mcq["before"]["numeric_family"]["max_geometry_credit"]
after_credit = mcq["after_repair"]["numeric_family"]["max_geometry_credit"]
check(abs(before_credit - 0.264) < 0.002 and abs(after_credit - 0.045) < 0.005,
      f"worst-case credit changed: {before_credit:+.3f} -> {after_credit:+.3f}")
check(mcq["before"]["surface_family"]["rules_in_family"] == 18,
      "the surface family is no longer searched in both directions")
check(not mcq["after_repair"]["surface_family"]["leaks"],
      "the repair now introduces a surface leak, which is what it must avoid")

# --- the cross-benchmark survey --------------------------------------------
survey = load("results/channel_survey.json")
by_benchmark = {r["benchmark"]: r for r in survey["benchmarks"]}
check(survey["summary"]["n_benchmark_files"] == 18, "survey file count changed")
check(survey["summary"]["n_assessed"] == 9, "assessable survey file count changed")
check(survey["summary"]["n_rejecting_uniformity"] == 7,
      "\\S3 says seven of nine reject a uniform key rank")
check(survey["summary"]["n_channel_open"] == 5,
      "\\S3 says five of nine have a positive credit bound")
check(survey["summary"]["n_channel_open_with_enough_groups"] == 4,
      "the count of trustworthy open channels changed")
mmlu_pro = by_benchmark["MMLU-Pro"]
check(mmlu_pro["n_numeric_items"] == 1263 and mmlu_pro["modal_n_options"] == 10,
      "the MMLU-Pro row the abstract cites changed")
check(abs(mmlu_pro["credit_bound"]["credit_lower_bound"] - 0.084) < 0.003,
      f"MMLU-Pro bound changed: {mmlu_pro['credit_bound']['credit_lower_bound']:+.3f}")
aqua = by_benchmark["AQuA-RAT"]
check(aqua["chi_square_vs_uniform"]["p_value"] > 0.5 and not aqua["exploitable"],
      "AQuA-RAT no longer supports the 'avoidable' claim")
# fig:survey's caption types both AQuA-RAT figures, so both are pinned.
check(abs(aqua["chi_square_vs_uniform"]["chi2"] - 1.2) < 0.05,
      f"fig:survey types chi2 = 1.2 for AQuA-RAT, against "
      f"{aqua['chi_square_vs_uniform']['chi2']:.2f}")
check(abs(aqua["chi_square_vs_uniform"]["p_value"] - 0.87) < 0.005,
      f"fig:survey types p = 0.87 for AQuA-RAT, against "
      f"{aqua['chi_square_vs_uniform']['p_value']:.3f}")
check(aqua["n_numeric_items"] > 100, "AQuA-RAT has too few items to carry the claim")
# the survey and the standalone analysis must agree about BixBench
survey_v15 = by_benchmark["BixBench v1.5"]
check(survey_v15["n_numeric_items"] == v15["n_numeric_items"]
      and abs(survey_v15["interior_rate"] - v15["interior_rate"]) < 1e-9,
      "the survey and option_artifacts disagree about BixBench v1.5")
for record in survey["benchmarks"]:
    if record.get("credit_bound"):
        check(record["credit_bound"]["credit_lower_bound"] <= record["plug_in_credit"] + 1e-9,
              f"{record['benchmark']}: the lower bound exceeds the plug-in estimate")
survey_provenance = load("data/external/survey/PROVENANCE.json")
check(len(survey_provenance["files"]) == 8, "survey provenance file count changed")
for entry in survey_provenance["files"]:
    blob = Path(entry["local_path"]).read_bytes()
    check(hashlib.sha256(blob).hexdigest() == entry["sha256"],
          f"vendored option set changed: {entry['local_path']}")

# --- the decomposition: bounds, additivity, and what solvers collect --------
choice = load("results/choice_model.json")
bounds = choice["geometry_bounds"]["BixBench v1.5"]
check(abs(bounds["max_geometry_credit"] - 0.264) < 0.002,
      "the available-credit upper bound in the abstract changed")
check(abs(bounds["min_geometry_credit"] + 0.183) < 0.002,
      "the available-credit lower bound changed")
check(bounds["exploitable"] and bounds["uniform_key_ranks_give_chance_for_every_solver"],
      "the geometry bound no longer reports the channel as open and closable")

solvers = choice["published_runs"] + choice["open_weight_runs"]
check(len(solvers) == 13, f"expected 13 measured solvers, found {len(solvers)}")
for run in solvers:
    check(abs(run["knowledge_component"] + run["geometry_component"]
              - (run["fitted_accuracy"] - 0.25)) < 1e-9,
          f"{run['run']}: the decomposition no longer adds up")
realised = choice["realised_vs_worst_case"]
check(abs(realised["max_realised_geometry_credit"] - 0.026) < 0.003
      and abs(realised["min_realised_geometry_credit"] + 0.013) < 0.003,
      f"realised-credit range changed: {realised['min_realised_geometry_credit']:+.3f} to "
      f"{realised['max_realised_geometry_credit']:+.3f}")
check(realised["max_realised_geometry_credit"] < bounds["max_geometry_credit"] / 3,
      "a solver now collects more than a third of the available credit; the thesis is stale")

published_fits = {r["run"]: r for r in choice["published_runs"]}
gpt = next(r for k, r in published_fits.items() if "gpt-4o" in k)
check(abs(gpt["lambda"] - 0.069) < 0.005, f"gpt-4o lambda changed: {gpt['lambda']:.3f}")
check(gpt["goodness_of_fit"]["p_value"] > 0.05,
      "the two-source account no longer fits the published gpt-4o table")
open_fits = {r["run"]: r for r in choice["open_weight_runs"]}
big = open_fits["Qwen/Qwen2.5-32B-Instruct"]
check(big["bootstrap"]["ci95"]["lambda"][0] > 0.05,
      "the largest model's knowledge component no longer excludes zero")
check(abs(big["lambda"] - 0.168) < 0.005, f"32B lambda changed: {big['lambda']:.3f}")
# EM converges to the boundary rather than landing exactly on it, so "reaches
# zero" is a tolerance, not an equality.
small = [r for r in choice["open_weight_runs"]
         if r["bootstrap"]["ci95"]["lambda"][0] < 0.01]
check(len(small) == 9, f"expected 9 models with lambda reaching zero, got {len(small)}")

# --- lambda validated where the answers really are known --------------------
mmlu_fit = {r["run"]: r for r in load("results/choice_model_mmlu.json")["open_weight_runs"]}
mmlu_probe = load("results/no_data_probe_mmlu.json")
check(len(mmlu_fit) == 4, "the MMLU validation panel changed size")
ladder = [(m["subsets"]["numeric options"]["arms"]["original"]["accuracy"],
           mmlu_fit[m["model"]]["lambda"], m["model"]) for m in mmlu_probe["models"]]
ladder.sort()
check([x[1] for x in ladder] == sorted(x[1] for x in ladder),
      "lambda no longer orders with MMLU accuracy; \\S4's validation is stale")
check(ladder[0][1] < 0.01 and ladder[-1][1] > 0.5,
      f"the MMLU lambda ladder collapsed: {[round(x[1], 3) for x in ladder]}")
for model, fit in mmlu_fit.items():
    withheld = next(m["subsets"]["numeric options"]["arms"]["original_stemless"]["accuracy"]
                    for m in mmlu_probe["models"] if m["model"] == model)
    check(0.20 < withheld < 0.27,
          f"{model}: withholding MMLU's question no longer lands at chance ({withheld:.3f})")
    check(abs(fit["geometry_component"]) < 0.02,
          f"{model}: MMLU geometry component exceeds its own bound")
# the same three models are an order of magnitude lower with the data withheld
for model in ("Qwen/Qwen2.5-7B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"):
    check(mmlu_fit[model]["lambda"] > 5 * open_fits[model]["lambda"],
          f"{model}: lambda no longer separates MMLU from the no-data condition")
# the value redraw dominates the rank effect, which is why the placebo exists
for m in mmlu_probe["models"]:
    contrasts = {c["contrast"]: c for c in m["subsets"]["numeric options"]["contrasts"]}
    values = abs(contrasts["original - placebo"]["difference"])
    rank = abs(contrasts["placebo - repaired"]["difference"])
    check(values > rank, f"{m['model']}: the value redraw no longer dominates the rank effect")

# --- the same decomposition where the channel is widest --------------------
# MMLU-Pro is the hardest case for the paper's claim: the widest open channel
# in the survey and solvers that are not guessing. If anything collects the
# credit, it should show here.
wide_fit = load("results/choice_model_mmlupro.json")
wide_bounds = wide_fit["geometry_bounds"]["MMLU-Pro (numeric, 10 options)"]
check(wide_bounds["n_options"] == 10 and len(wide_bounds["key_by_rank_p"]) == 10,
      "the ten-option fit is no longer ten-option")
check(abs(wide_bounds["max_geometry_credit"] - 0.115) < 0.005,
      f"MMLU-Pro available credit changed: {wide_bounds['max_geometry_credit']:+.3f}")
check(abs(wide_bounds["chance"] - 0.1) < 1e-12, "ten options no longer means 10% chance")
for run in wide_fit["open_weight_runs"]:
    check(abs(run["knowledge_component"] + run["geometry_component"]
              - (run["fitted_accuracy"] - 0.1)) < 1e-9,
          f"{run['run']}: the ten-option decomposition no longer adds up")
    check(run["geometry_component"] < 0.02,
          f"{run['run']}: now collects {run['geometry_component']:+.3f} of MMLU-Pro's "
          "channel; \\S4's claim is stale")
    check(run["lambda"] > 0.05,
          f"{run['run']}: MMLU-Pro margin is no longer recall-shaped")

# --- the repair at ten options, and why it is only partial -----------------
wide = load("results/mcq_audit_mmlu_pro.json")
check(wide["n_options"] == 10, "the MMLU-Pro audit is no longer at ten options")
check(wide["before"]["numeric_family"]["n_items"] == 1263, "MMLU-Pro item count changed")
before_wide = wide["before"]["numeric_family"]["max_geometry_credit"]
after_wide = wide["after_repair"]["numeric_family"]["max_geometry_credit"]
check(abs(before_wide - 0.115) < 0.005 and abs(after_wide - 0.064) < 0.008,
      f"MMLU-Pro repair effect changed: {before_wide:+.3f} -> {after_wide:+.3f}")
reach_wide = wide["reachability"]["reachable_share_by_rank"]
reach_narrow = mcq["reachability"]["reachable_share_by_rank"]
check(abs(wide["reachability"]["mean_reachable_share"] - 0.83) < 0.02
      and abs(mcq["reachability"]["mean_reachable_share"] - 0.97) < 0.02,
      "the 83%-vs-97% reachability contrast in \\S5 is stale")
check(reach_wide["9"] < reach_wide["0"] and reach_narrow["3"] < reach_narrow["0"],
      "reachability no longer falls towards the extreme ranks")

# --- the survey, now drawn rather than typed ------------------------------
# Figure 2 is generated from results/channel_survey.json by make_figures.py, so
# a transcription cannot drift; what is still typed is its caption's counts.
by_name = {record["benchmark"]: record for record in survey["benchmarks"]}
_assessable = [b for b in survey["benchmarks"]
               if (b.get("credit_bound") or {}).get("credit_lower_bound") is not None]
check(len(_assessable) == 9, f"{len(_assessable)} assessable files, expected 9")
check(sum(1 for b in _assessable
          if b["credit_bound"]["credit_lower_bound"] > 0) == 5,
      "no longer five files with a positive value bound")
check(sum(1 for b in _assessable
          if b["length_channel"]["credit_bound"]["credit_lower_bound"] > 0) == 6,
      "no longer six files with a positive length bound")
check(survey["summary"]["n_rejecting_uniformity"] == 7,
      "no longer seven files rejecting a uniform key rank")

# --- the submission notes quote the manuscript, so they must still match ----
# (neither file ships in the public repository, so this runs only where they exist)
if Path("SUBMISSION.md").exists() and Path("submission_text.py").exists():
    notes = subprocess.run([sys.executable, "submission_text.py"], capture_output=True, text=True)
    check(notes.returncode == 0,
          f"SUBMISSION.md has drifted from main.tex: {notes.stdout.strip()}{notes.stderr.strip()}")

# --- the channel is plural, and a repair can move solvers between coordinates
# The paper's second finding: uniformising one ordering closed that coordinate
# and left solvers collecting a different one. Every number in \S5's second
# paragraph is pinned here.
multi = load("results/mcq_audit_multi.json")
check(multi["two_channel"], "the released multi-channel audit is not two-channel")
SURFACES = {"lexical isolation", "written length", "significant digits"}
for phase in ("before", "after_repair"):
    channels = multi[phase]["surface_channels"]["orderings"]
    check(set(channels) == SURFACES, f"the audited orderings changed: {sorted(channels)}")
bounds_before = {name: block["numeric items"]["credit_bound"]["credit_lower_bound"]
                 for name, block in multi["before"]["surface_channels"]["orderings"].items()}
bounds_after = {name: block["numeric items"]["credit_bound"]["credit_lower_bound"]
                for name, block in multi["after_repair"]["surface_channels"]["orderings"].items()}
check(abs(bounds_before["written length"] - 0.089) < 0.006,
      f"BixBench's length channel changed: {bounds_before['written length']:+.3f}")
check(abs(bounds_before["significant digits"] - 0.101) < 0.008,
      f"BixBench's roundness channel changed: {bounds_before['significant digits']:+.3f}")
# Two of the three, not three: v1.5's lexical-isolation bound is negative once
# each pair of options is compared once rather than in both directions
# (mcq_audit._sim). \S5 claims the two that are open plus the value channel.
open_before = {name for name, value in bounds_before.items() if value > 0}
check(open_before == {"written length", "significant digits"},
      f"the surface coordinates open on released v1.5 changed: {bounds_before}")
check(bounds_before["lexical isolation"] < 0,
      "v1.5's isolation bound is no longer negative, so Appendix G's reason for "
      f"not claiming it is stale: {bounds_before['lexical isolation']:+.3f}")
check(bounds_before["significant digits"] > bounds_before["written length"],
      "roundness is no longer the widest surface coordinate on v1.5")
check(all(value < 0 for value in bounds_after.values()),
      f"the multi-channel repair no longer closes the surface channels: {bounds_after}")
check(multi["after_repair"]["coordinates_firing"] == [],
      "the multi-channel repair no longer closes every coordinate on v1.5: "
      f"{multi['after_repair']['coordinates_firing']}")
check(abs(multi["after_repair"]["numeric_family"]["max_geometry_credit"] - 0.045) < 0.010,
      "the multi-channel repair's value-channel figure changed")
hit = multi["surface_target_hit_rate"]
check(set(hit["first_draw"]) == SURFACES, f"the repaired orderings changed: {sorted(hit)}")
check(multi["search"] == "construct" and multi["calibrated"],
      "the shipped repair is no longer the construction drawing from each item's "
      "frontier, which is what \\S5 and results/repair_search_ablation.json describe")
# The construction assigns the two counted coordinates, so it meets them; the
# first-draw rate is low there by design and is not the number to read.
check(min(hit["assigned"][name] for name in ("written length", "significant digits")) > 0.99,
      f"the construction no longer meets the counts it assigns: {hit['assigned']}")
check(hit["assigned"]["lexical isolation"] > 0.85,
      f"the isolation survey no longer reaches its target: {hit['assigned']}")
attainable = hit["calibration_worst_rank_share"][-1]["widest_attainable_rank_share"]
check(set(attainable) == SURFACES | {"sorted value"},
      f"the fitted coordinates changed: {sorted(attainable)}")
check(max(attainable.values()) < 0.2501,
      "BixBench no longer admits a flat marginal on every coordinate, which is what "
      f"makes a residual there the repair's doing: {attainable}")

# At ten options the same repair is only partial, and we say which coordinate resists.
wide_multi = load("results/mcq_audit_mmlu_pro_multi.json")
wide_before = {name: block["numeric items"]["credit_bound"]["credit_lower_bound"] for name, block
               in wide_multi["before"]["surface_channels"]["orderings"].items()}
wide_after = {name: block["numeric items"]["credit_bound"]["credit_lower_bound"] for name, block
              in wide_multi["after_repair"]["surface_channels"]["orderings"].items()}
check(abs(wide_before["written length"] - 0.041) < 0.008,
      f"MMLU-Pro's length channel changed: {wide_before['written length']:+.3f}")
check(abs(wide_multi["after_repair"]["numeric_family"]["max_geometry_credit"] - 0.012) < 0.008,
      "the ten-option value channel no longer closes to +1.2 points: "
      f"{wide_multi['after_repair']['numeric_family']['max_geometry_credit']:+.3f}")
# It closes the file's *numeric* items and \S5 says only that. The other 8,717
# ten-option items are text, a numeric repair does not touch them, and the
# verdict keeps firing on them -- which Appendix G quotes as the scope.
all_items_after = {name: block["all items"]["credit_bound"]["credit_lower_bound"]
                   for name, block in
                   wide_multi["after_repair"]["surface_channels"]["orderings"].items()
                   if "all items" in block}
check(all_items_after.get("lexical isolation", 0) > 0.02,
      "the repaired ten-option file no longer leaks over all items, so Appendix G's "
      f"scope sentence is stale: {all_items_after}")
check(wide_multi["after_repair"]["coordinates_firing"] != [],
      "the repaired ten-option file now closes every coordinate; Appendix G says "
      "at least one still clears zero")
check(wide_multi["n_rank_reassigned"] == 6,
      f"the ten-option off-target count changed: {wide_multi['n_rank_reassigned']}")
check(load("results/mcq_audit_mmlu_pro.json")["n_rank_reassigned"] == 210
      and load("results/mcq_audit_mmlu_pro_isolation.json")["n_rank_reassigned"] == 123,
      "the searching repairs' off-target counts changed; Appendix G quotes both")
# \S5's headline: at ten options every coordinate lands inside what a file with
# no leak in it produces, which is the only sense in which a file is "closed".
clean_wide = max(load("results/audit_calibration_k10.json")
                 ["widest_bound_on_a_clean_file"].values())
for name, value in wide_after.items():
    check(value < clean_wide,
          f"the ten-option {name} bound {value:+.3f} is no longer inside the "
          f"{clean_wide:+.3f} a clean ten-option file produces; \\S5 is stale")
    check(value < wide_before[name] or wide_before[name] < 0,
          f"the ten-option repair no longer narrows {name}")
check(wide_multi["surface_target_hit_rate"]["first_draw"]["written length"] < 0.30,
      "the length target is no longer the hard one at ten options")
check(min(wide_multi["surface_target_hit_rate"]["assigned"][name]
          for name in ("written length", "significant digits")) > 0.99,
      "the ten-option construction no longer meets the counts it assigns")

# The rank-only repair at ten options is what opened the surface coordinates:
# it is the finding of \S5's second paragraph, so pin both sides of it.
wide_rank = load("results/mcq_audit_mmlu_pro.json")
rank_before = {name: block["numeric items"]["credit_bound"]["credit_lower_bound"] for name, block
               in wide_rank["before"]["surface_channels"]["orderings"].items()}
rank_after = {name: block["numeric items"]["credit_bound"]["credit_lower_bound"] for name, block
              in wide_rank["after_repair"]["surface_channels"]["orderings"].items()}
check(rank_before["lexical isolation"] < 0 < rank_after["lexical isolation"],
      f"uniformising the value rank no longer opens isolation at ten options: "
      f"{rank_before['lexical isolation']:+.3f} -> {rank_after['lexical isolation']:+.3f}")
for name in SURFACES:
    check(rank_after[name] > rank_before[name],
          f"the value-only repair no longer widens {name} at ten options: "
          f"{rank_before[name]:+.3f} -> {rank_after[name]:+.3f}")
check(abs(rank_after["significant digits"] - 0.091) < 0.010,
      f"the opened roundness channel changed: {rank_after['significant digits']:+.3f}")

# --- what each coordinate actually pays a solver ---------------------------
attribution = load("results/channel_attribution_mmlupro.json")
check(set(attribution["channels"]) == SURFACES | {"sorted value"},
      f"the attributed channels changed: {attribution['channels']}")
check(len(attribution["models"]) == 3,
      f"the ten-option panel has {len(attribution['models'])} models")
# One of the three does collect on the released file, and \S4 says so: the
# largest takes a fifth of the value channel with the question withheld. The
# check is therefore "exactly one, and it is the largest", not "none".
collectors = []
for record in attribution["models"]:
    chance = 1.0 / record["n_options"]
    released = record["arms"]["original_stemless"]
    repaired = record["arms"]["repaired_stemless"]
    value = released["channels"]["sorted value"]
    if value["credit_collected_ci95"][0] > 0:
        collectors.append((record["model"], value, released["accuracy"]))
    else:
        check(abs(released["accuracy"] - chance) < 0.02,
              f"{record['model']}: the released file no longer leaves it at chance "
              f"({released['accuracy']:.3f})")
        for name, channel in released["channels"].items():
            check(abs(channel["credit_collected"]) < 0.02,
                  f"{record['model']}: collects {channel['credit_collected']:+.3f} of "
                  f"{name} on the released file")
        check(repaired["accuracy"] > released["accuracy"] + 0.02,
              f"{record['model']}: the numeric repair no longer lifts it above chance")
    collected = {name: channel["credit_collected"]
                 for name, channel in repaired["channels"].items()}
    # Length and roundness track each other for numbers, so the claim is that
    # a written-form coordinate leads, not that one particular one does.
    check(max(collected, key=collected.get) in ("written length", "significant digits"),
          f"{record['model']}: a written-form coordinate no longer leads: {collected}")
    check(repaired["channels"]["significant digits"]["credit_available"]
          > released["channels"]["significant digits"]["credit_available"],
          f"{record['model']}: the repair no longer widens the roundness channel")

check(len(collectors) == 1,
      "\\S4 says exactly one ten-option solver collects on the released file; "
      f"{len(collectors)} do: {[m for m, _, _ in collectors]}")
_collector, _value, _accuracy = collectors[0]
check("32B" in _collector,
      f"the collector is no longer the largest model in the panel: {_collector}")
check(abs(_value["credit_collected"] - 0.023) < 0.004,
      f"{_collector}: the value credit it collects changed "
      f"({_value['credit_collected']:+.4f})")
check(abs(_accuracy - 0.134) < 0.006,
      f"{_collector}: its stem-withheld accuracy on the released file changed "
      f"({_accuracy:.3f})")
check(_value["credit_collected"] < 0.3 * _value["credit_available"],
      "the collector now takes more than a third of the value channel, so "
      "\\S4's \"a fifth\" is stale")

# --- does the repair leave a solver where it found one? --------------------
# The claim \S5 scopes: yes at four options, no at ten. Read from the probe
# outcomes directly, since this is an accuracy comparison and not a fit.
def stemless_pair(path):
    """(released, repaired) stem-withheld accuracy over the numeric items."""
    hits, total = {}, {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row["is_numeric"] or not row["arm"].endswith("stemless"):
                continue
            hits[row["arm"]] = hits.get(row["arm"], 0) + row["correct"]
            total[row["arm"]] = total.get(row["arm"], 0) + 1
    return (hits["original_stemless"] / total["original_stemless"],
            hits["repaired_stemless"] / total["repaired_stemless"],
            json.loads(gzip.open(path, "rt", encoding="utf-8").readline())["model"])


def stemless_gain_interval(path, reps=2000, seed=7):
    """Repaired minus released stem-withheld accuracy, numeric items, by capsule.

    The four-option claim is that the repair leaves a solver where it found it,
    which is a statement about an interval and not about a point: one model's
    point estimate is $+3.4$ and its interval spans zero, so the check is that
    no interval clears zero from above.
    """
    by_cluster = {}
    model = None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            model = model or row["model"]
            if not row["is_numeric"] or not row["arm"].endswith("stemless"):
                continue
            by_cluster.setdefault(row["cluster"], []).append((row["arm"], row["correct"]))

    def delta(groups):
        released = [c for g in groups for arm, c in g if arm == "original_stemless"]
        repaired = [c for g in groups for arm, c in g if arm == "repaired_stemless"]
        if not released or not repaired:
            return None
        return sum(repaired) / len(repaired) - sum(released) / len(released)

    keys = list(by_cluster)
    point = delta([by_cluster[k] for k in keys])
    rng = random.Random(seed)
    draws = sorted(d for d in (delta([by_cluster[rng.choice(keys)] for _ in keys])
                               for _ in range(reps)) if d is not None)
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return point, lo, hi, model


four_option = [stemless_pair(p) for p in sorted(Path("results/probe_bix_four").glob("*.jsonl.gz"))]
check(len(four_option) == 4, f"the four-option repair panel has {len(four_option)} models")
four_intervals = [stemless_gain_interval(p)
                  for p in sorted(Path("results/probe_bix_four").glob("*.jsonl.gz"))]
for point, lo, hi, model in four_intervals:
    check(lo <= 0,
          f"{model}: the four-ordering repair now lifts a solver at four options "
          f"({point:+.3f} [{lo:+.3f}, {hi:+.3f}]); \\S5's scoping is stale")
check(abs(min(p for p, _, _, _ in four_intervals) + 0.055) < 0.015
      and abs(max(p for p, _, _, _ in four_intervals) - 0.034) < 0.015,
      "\\S5's four-option range is stale: "
      f"{[round(p, 3) for p, _, _, _ in four_intervals]}")

# The value-only repair against the construction, on the same three solvers.
# \S4 says uniformising one coordinate moves solvers rather than stopping them,
# and that drawing every target from what the item can reach closes all four.
ten_option = {}
for path in sorted(Path("results", "probe_mmlupro").glob("*.jsonl.gz")):
    with gzip.open(path, "rt", encoding="utf-8") as rows:
        if '"repaired_stemless"' not in rows.read(200000):
            continue
    released, repaired, model = stemless_pair(path)
    ten_option.setdefault(model, {})["probe_mmlupro"] = repaired - released
check(len(ten_option) == 3, f"the ten-option panel has {len(ten_option)} models")
# The two solvers the released file leaves at chance are the ones whose gain
# means anything: the third already collects on the released file, so it has
# little room to be lifted.
lifted = {model: gains for model, gains in ten_option.items() if "32B" not in model}
check(len(lifted) == 2, f"the at-chance pair changed: {sorted(lifted)}")
for model, gains in sorted(lifted.items()):
    check(min(gains.values()) > 0.02,
          f"{model}: the value-only repair no longer lifts it "
          f"({ {k: round(v, 3) for k, v in gains.items()} })")

# And whether the construction does, measured the same way.
built_dir = Path("results/probe_mmlupro_construct")
if any(built_dir.glob("*.jsonl.gz")):
    for path in sorted(built_dir.glob("*.jsonl.gz")):
        released, repaired, model = stemless_pair(path)
        if "32B" in model:
            continue
        check(repaired - released < min(ten_option[model].values()),
              f"{model}: the constructed repair no longer leaves it lower than every "
              f"searching one ({repaired - released:+.3f} vs "
              f"{ {k: round(v, 3) for k, v in ten_option[model].items()} })")

# --- the audit's own false-positive rate, and the search that undoes a repair
# \S5 quotes both, and Appendix G is nothing but these two files.
for path, label, n_files, verdict_rate, widest in (
        ("results/audit_calibration.json", "four-option", 60, 0.167, 0.033),
        ("results/audit_calibration_k10.json", "ten-option", 30, 0.133, 0.010)):
    cal = load(path)
    check(cal["replicates"] == n_files,
          f"the {label} calibration ran {cal['replicates']} files, not {n_files}")
    rates = cal["false_positive_rate"]
    check(abs(rates["any coordinate"] - verdict_rate) < 0.001,
          "the rate at which some coordinate fires on a clean "
          f"{label} file changed: {rates['any coordinate']:.3f}")
    check(rates["numeric rule family"] <= 0.05 and rates["surface rule family"] <= 0.05,
          f"a rule-family test is over-firing on clean {label} files: {rates}")
    coordinates = {name: value for name, value in rates.items()
                   if name != "any coordinate" and "rule family" not in name}
    check(max(coordinates.values()) <= 0.10,
          f"a {label} coordinate test is over-firing on clean files: {coordinates}")
    check(abs(max(cal["widest_bound_on_a_clean_file"].values()) - widest) < 0.002,
          "the widest bound ever seen on a clean "
          f"{label} file changed: {max(cal['widest_bound_on_a_clean_file'].values()):+.3f}")
    check(rates["any coordinate"] > max(coordinates.values()),
          f"on clean {label} files the six tests together no longer fire more often "
          "than any single one, which is the measurement \\S5 cites for removing "
          "the one-line verdict from the tool")

ablation = load("results/repair_search_ablation.json")
check(len(ablation["seeds"]) == 8, f"the ablation ran {len(ablation['seeds'])} seeds")
cells = {(c["search"], c["draw"]): c for c in ablation["cells"]}
check(set(cells) == {(s, d) for s in ("sample", "nearest", "guided")
                     for d in ("uniform", "calibrated")} | {("construct", "frontier")},
      f"the ablation's cells changed: {sorted(cells)}")
shipped, descent = cells[("sample", "uniform")], cells[("nearest", "uniform")]
check(descent["first_draw_met_mean"]["written length"]
      > 2 * shipped["first_draw_met_mean"]["written length"],
      "the descent no longer meets more than twice as many length targets as the "
      f"sampler: {descent['first_draw_met_mean']} vs {shipped['first_draw_met_mean']}")
check(descent["bound_worst"]["written length"] > 0.05 > shipped["bound_worst"]["written length"],
      "the better search no longer leaves a worse file, which is \\S5's claim: "
      f"{descent['bound_worst']['written length']:+.3f} vs "
      f"{shipped['bound_worst']['written length']:+.3f}")
check(descent["leaks"] >= 6 and shipped["leaks"] <= 2,
      f"the seed counts changed: descent {descent['leaks']}/8, sampler {shipped['leaks']}/8")
# The mechanism, not just the outcome: the descent's hit rate is the one that
# varies with the target rank, which is what concentrates the realised ranks.
check(descent["hit_rate_spread_mean"]["written length"]
      > 2 * shipped["hit_rate_spread_mean"]["written length"],
      "the descent's hit rate no longer varies more across target ranks than the "
      f"sampler's: {descent['hit_rate_spread_mean']} vs {shipped['hit_rate_spread_mean']}")
check(descent["widest_realised_mean"]["written length"]
      > shipped["widest_realised_mean"]["written length"] + 0.05,
      "the descent no longer concentrates the realised length ranks")
check(cells[("nearest", "calibrated")]["leaks"] < descent["leaks"],
      "calibrating the target draw no longer helps the descent, which \\S5 says it "
      "does by about half")

# The constructive repair is what \S5 now ships, so it has to beat the sampler
# on the thing that matters -- the bound, not the hit rate.
built = cells[("construct", "frontier")]
check(built["bound_worst"]["written length"] < shipped["bound_worst"]["written length"],
      "the construction no longer beats the sampler's worst written-length bound: "
      f"{built['bound_worst']['written length']:+.3f} vs "
      f"{shipped['bound_worst']['written length']:+.3f}")
# Read against the clean-file maximum, which is the rule Appendix G gives a
# maintainer, not against zero. On one seed in eight the constructed file's
# isolation bound clears it, by 0.6 points, and \S5 reports that rather than
# rounding it away.
clean_four = load("results/audit_calibration.json")["widest_bound_on_a_clean_file"]
per_coordinate = {name: clean_four[f"{name} (numeric items)"]
                  for name in built["bound_worst"]}
over = {name: (value, per_coordinate[name])
        for name, value in built["bound_worst"].items()
        if value > per_coordinate[name]}
check(set(over) <= {"lexical isolation"},
      f"a constructed file now clears its clean-file maximum on more than "
      f"isolation: {over}")
check(built["bound_worst"]["lexical isolation"] < 0.025,
      "the constructed file's worst isolation bound moved: "
      f"{built['bound_worst']['lexical isolation']:+.3f}")
check(built["leaks"] <= shipped["leaks"],
      f"the construction is flagged more often than the sampler: "
      f"{built['leaks']}/8 vs {shipped['leaks']}/8")
check(min(built["assigned_met_mean"].values()) > 0.85,
      f"the construction no longer meets the ranks it assigns: "
      f"{built['assigned_met_mean']}")
check(built["first_draw_met_mean"]["written length"]
      < built["assigned_met_mean"]["written length"] / 2,
      "the construction's drawn and assigned rates no longer differ, so the table's "
      "two columns are telling the same story")

# The four numbers app:repair types about the search ablation. They used to be
# a table the validator parsed; the table is gone, the numbers are not.
_ablation = {(c["search"], c["draw"]): c for c in ablation["cells"]}
_name = "written length"
for _key, _drawn, _bound in ((("nearest", "uniform"), 71, 12.9),
                             (("sample", "uniform"), 25, 3.7)):
    _cell = _ablation[_key]
    check(abs(100 * _cell["first_draw_met_mean"][_name] - _drawn) < 0.6,
          f"app:repair types {_drawn}% of written-length targets met by "
          f"{_key[0]}, measured {100 * _cell['first_draw_met_mean'][_name]:.1f}")
    check(abs(100 * _cell["bound_worst"][_name] - _bound) < 0.06,
          f"app:repair types {_bound:+.1f} for {_key[0]}, measured "
          f"{100 * _cell['bound_worst'][_name]:+.2f}")

# Figure 1b's caption types six numbers. The bars themselves are drawn from the
# same two files by make_figures.py, so this pins the caption to the bars.
_fig1b = {}
for _tag, _path in (("released", "results/channel_attribution_mmlupro.json"),
                    ("construct", "results/channel_attribution_mmlupro_construct.json")):
    _doc = load(_path)
    for _arm in ("original_stemless", "repaired_stemless"):
        _ch = _doc["models"][0]["arms"][_arm]["channels"]
        _fig1b[(_tag, _arm)] = {c: _ch[c]["credit_available"] for c in _doc["channels"]}
for _name, _before, _after in (("sorted value", 11.5, 3.1),
                               ("significant digits", 4.6, 9.8),
                               ("lexical isolation", 1.4, 7.0)):
    check(abs(100 * _fig1b[("released", "original_stemless")][_name] - _before) < 0.06,
          f"Figure 1b types {_before:+.1f} for released {_name}, measured "
          f"{100 * _fig1b[('released', 'original_stemless')][_name]:+.2f}")
    check(abs(100 * _fig1b[("released", "repaired_stemless")][_name] - _after) < 0.06,
          f"Figure 1b types {_after:+.1f} for {_name} after the value repair, "
          f"measured {100 * _fig1b[('released', 'repaired_stemless')][_name]:+.2f}")
check(max(_fig1b[("construct", "repaired_stemless")].values()) < 0.017 + 5e-4,
      "Figure 1b types that the construction closes all four at or below +1.7; "
      f"widest is {100 * max(_fig1b[('construct', 'repaired_stemless')].values()):+.2f}")

# \S4 contrasts the guarantee on the whole repaired files with Table 1's
# exchangeable-feasible subset, and the two have opposite signs; both are typed.
for _path, _typed in (("results/learned_probe_mmlu.json", -2.9),
                      ("results/learned_probe_mmlupro.json", -1.5)):
    _doc = load(_path)
    _worst = _doc["clean"]["audited ranks"]["max"]
    _got = _doc["files"]["repaired"]["audited ranks"]["cv_accuracy"] - _worst
    check(abs(100 * _got - _typed) < 0.06,
          f"\\S4 types {_typed:+.1f} for the audited-rank learner on the whole "
          f"repaired file in {_path}, measured {100 * _got:+.2f}")
    check(_got < 0, f"the guarantee's own check no longer holds on {_path}")

# --- do the shipped results still come out of the shipped code? ------------
# They did not, once. `isolation_scores` was changed to compare each pair of
# options once and use the value for both directions, because SequenceMatcher is
# not symmetric, and the results files predated the fix -- which nothing here
# would have caught, because every check read the stored JSON. So recompute the
# cheap part of the audit and compare it to what is stored.
from mcq_audit import build_items as _build_items  # noqa: E402
from mcq_audit import read_rows as _read_rows  # noqa: E402
from mcq_audit import surface_channels as _surface_channels  # noqa: E402

for label, path, stored, cluster_field, k in (
        ("BixBench as released", Path("data/bixbench.jsonl"),
         load("results/mcq_audit_multi.json")["before"], "capsule_uuid", 4),
        ("the constructed repair of it",
         Path("results/repaired/bixbench_v15_repaired_multi.jsonl"),
         load("results/mcq_audit_multi.json")["after_repair"], "capsule_uuid", 4),
        ("the value-only repair of it",
         Path("results/repaired/bixbench_v15_repaired.jsonl"),
         load("results/mcq_audit.json")["after_repair"], "capsule_uuid", 4)):
    items, found_k = _build_items(_read_rows(path), "ideal", "distractors",
                                  cluster_field, "question", k)
    fresh = _surface_channels(items, 2000, found_k)["orderings"]
    for name, block in fresh.items():
        for subset, stats in block.items():
            got = (stats["credit_bound"] or {}).get("credit_lower_bound")
            want = ((stored["surface_channels"]["orderings"][name][subset]
                     ["credit_bound"]) or {}).get("credit_lower_bound")
            check(got is not None and want is not None and abs(got - want) < 1e-9,
                  f"{label}: the stored {name} bound on {subset} is {want}, but "
                  f"the code now computes {got} -- a results file has drifted "
                  "from the code that made it")

# Appendix E: the isolation statistic must not depend on the interpreter. An
# explicit accumulation loop reproduces what `sum` did before CPython 3.12 made
# it compensated, on 3.12 as well, so the count below is the same everywhere.
from mcq_audit import _sim as _similarity  # noqa: E402
from mcq_audit import feature_rank as _feature_rank  # noqa: E402
from mcq_audit import isolation_rank as _isolation_rank  # noqa: E402
from mcq_audit import isolation_scores as _isolation_scores  # noqa: E402


def _naive_total(values):
    total = 0.0
    for value in values:
        total += value
    return total


def _isolation_with(options, total):
    n = len(options)
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim[i][j] = sim[j][i] = _similarity(options[i], options[j])
    return [-total(row) for row in sim]


knife = ["0.1126", "0.0867", "0.0713", "0.1433"]
check(_isolation_scores(knife)[:2] == [-1.5, -1.5] and _isolation_rank(knife) == 2,
      "isolation_scores is not summing exactly: this option set is the one whose "
      "key rank moved between CPython 3.10 and 3.12")
check(_isolation_with(knife, _naive_total)[1] != -1.5,
      "the naive reference no longer reproduces the pre-3.12 sum, so the count "
      "of affected option sets below means nothing")

import math as _math  # noqa: E402

moved = pairs = 0
for _path in ("data/bixbench.jsonl",
              "results/repaired/bixbench_v15_repaired.jsonl",
              "results/repaired/bixbench_v15_repaired_multi.jsonl"):
    for _row in _read_rows(Path(_path)):
        _options = [_row.get("ideal")] + list(_row.get("distractors") or [])
        if not all(isinstance(o, str) for o in _options):
            continue
        pairs += 1
        moved += (_feature_rank(_options, _isolation_with(_options, _naive_total))
                  != _feature_rank(_options, _isolation_with(_options, _math.fsum)))
check((moved, pairs) == (16, 615),
      f"Appendix E says the summation changes the key's isolation rank on 16 of "
      f"v1.5's 615 option sets; recomputing gives {moved} of {pairs}")

# Appendix D's last paragraph: closing every audited coordinate at ten options
# takes the collector to chance and halves, rather than removes, what the repair
# hands the other two. Both halves are claims, so both are checked.
built_probe = Path("results/probe_mmlupro_construct")
if any(built_probe.glob("*.jsonl.gz")):
    built_gains = {}
    for path in sorted(built_probe.glob("*.jsonl.gz")):
        released, repaired, model = stemless_pair(path)
        built_gains[model] = (released, repaired)
    check(len(built_gains) == 3,
          f"the constructed ten-option panel has {len(built_gains)} models")
    collector = [m for m in built_gains if "32B" in m]
    check(len(collector) == 1, f"the collector changed: {sorted(built_gains)}")
    released, repaired = built_gains[collector[0]]
    check(released > 0.12 and abs(repaired - 0.10) < 0.015,
          f"{collector[0]}: the construction no longer takes the collector to "
          f"chance ({released:.3f} -> {repaired:.3f}); Appendix D says it does")
    residue = {m: rep - rel for m, (rel, rep) in built_gains.items()
               if "32B" not in m}
    check(all(0.02 < value < 0.06 for value in residue.values()),
          f"the residue on the other two moved outside 2--6 points: {residue}")
    for model, value in residue.items():
        check(value < min(ten_option[model].values()),
              f"{model}: the constructed repair no longer leaves less than every "
              f"searching one ({value:+.3f} vs {ten_option[model]})")

# --- the full-length build is a transformation, not a second manuscript ------
HELD_OUT = {"leading digit", "decimal places", "digit sum", "string order",
            "length, later ties"}

# --- the placebo panels, as Appendix D quotes them ---------------------------
contrast = load("results/placebo_contrast.json")
check(set(contrast["panels"]) == {"BixBench v1.5", "MMLU-Pro", "MMLU"},
      f"the placebo panels changed: {sorted(contrast['panels'])}")
for label, n_models, k in (("BixBench v1.5", 11, 4), ("MMLU-Pro", 10, 10)):
    panel = contrast["panels"][label]
    check(panel["n_models"] == n_models and panel["n_options"] == k,
          f"{label}: {panel['n_models']} solvers at {panel['n_options']} options")
    check(panel["rank_effect_median"] < 0,
          f"{label}: the median rank effect is no longer negative")
    check(panel["rank_effect_positive"] <= 2,
          f"{label}: {panel['rank_effect_positive']} solvers have a positive rank "
          "effect, and Appendix D says two")
    worst = min(row["rank_effect"] for row in panel["models"])
    collector = min(panel["models"], key=lambda row: row["rank_effect"])
    check("32B" in collector["model"],
          f"{label}: the largest rank effect is now {collector['model']}'s, not "
          "the solver that collected most")
    check(panel["released_vs_rank_effect_rho"] < -0.4,
          f"{label}: the released-accuracy trend is now "
          f"{panel['released_vs_rank_effect_rho']:+.2f}")
    check(panel["rho_ci95"][1] > 0,
          f"{label}: the trend interval no longer includes zero, so Appendix D "
          "understates it; it is reported as a suggestion")
    check(panel["placebo_lift_median"] > 0.005,
          f"{label}: the placebo's median lift is {panel['placebo_lift_median']:+.3f}")

# Appendix D's third panel: MMLU, where the released geometry was nearly closed,
# so the rank effect should not be negative and the placebo should carry it.
panel_block = load("results/placebo_contrast.json")["panels"]["MMLU"]
check(panel_block["n_models"] == 11,
      f"the MMLU panel has {panel_block['n_models']} models, expected 11")
check(panel_block["rank_effect_positive"] == 8,
      f"the MMLU rank effect is positive for {panel_block['rank_effect_positive']} "
      "models, and Appendix D says eight of eleven")
check(abs(panel_block["placebo_lift_median"] - 0.025) < 0.002
      and abs(panel_block["rank_effect_median"] - 0.008) < 0.002,
      f"the MMLU placebo lift median is {panel_block['placebo_lift_median']:+.3f} "
      f"and the rank effect median {panel_block['rank_effect_median']:+.3f}; "
      "Appendix D says +2.5 and +0.8 points")
check(max(record["released"] for record in panel_block["models"]) <= 0.25,
      "an MMLU solver is now above chance on the released file with the question "
      "withheld, which Appendix D says none is")
negatives = sorted(record["model"] for record in panel_block["models"]
                   if record["rank_effect"] < 0)
check(all("Qwen2.5-32B" in name or "Qwen2.5-14B" in name or "Qwen2.5-7B" in name
          for name in negatives),
      f"the MMLU rank effect is negative for {negatives}, and Appendix D says "
      "the three largest models")
check(panel_block["rho_ci95"][0] < 0 < panel_block["rho_ci95"][1],
      "the MMLU trend interval no longer includes zero, so Appendix D's "
      "'suggestion' wording understates it")

# --- the learned solver, and the table app:learned types out --------------------
learned = load("results/learned_probe_mmlupro.json")
FAMILIES = ("audited ranks", "rank-only", "every feature")
check(set(learned["clean"]) == set(FAMILIES),
      f"the learner's families changed: {sorted(learned['clean'])}")
clean_worst = {family: learned["clean"][family]["max"] for family in FAMILIES}
built = learned["files"]["repaired"]
released_learned = learned["files"]["released"]
# One: the guarantee holds against the strongest learned solver in its own class.
check(built["audited ranks"]["cv_accuracy"] < clean_worst["audited ranks"],
      "the constructed repair no longer holds against a learned solver restricted "
      f"to the audited orderings: {built['audited ranks']['cv_accuracy']:.4f} "
      f"against a clean worst case of {clean_worst['audited ranks']:.4f}")
# Two: allowed the held-out orderings, the same learner collects.
check(built["rank-only"]["cv_accuracy"] > clean_worst["rank-only"] + 0.05,
      "the held-out orderings no longer show up as learned accuracy: "
      f"{built['rank-only']['cv_accuracy']:.4f}")
# Three: given features that are not ranks it does better on the repaired file.
check(built["every feature"]["cv_accuracy"]
      > released_learned["every feature"]["cv_accuracy"],
      "the repaired ten-option file is no longer easier than the released one for "
      "a learner given non-rank features, which is app:learned's third column: "
      f"{built['every feature']['cv_accuracy']:.4f} against "
      f"{released_learned['every feature']['cv_accuracy']:.4f}")
heaviest = max(built["every feature"]["weights"].items(), key=lambda kv: abs(kv[1]))
check(heaviest[0] == "relative magnitude",
      f"the heaviest non-rank weight is now {heaviest}, not relative magnitude")
from option_artifacts import parse_number

_tex = Path("main.tex").read_text(encoding="utf-8")
# The same text with every run of whitespace collapsed, so a check for a phrase
# does not depend on where the line happened to wrap. Several checks below
# broke exactly once because a re-wrap moved a number onto the next line.
_flat = re.sub(r"\s+", " ", _tex)
# A per cent sign not escaped starts a LaTeX comment: after text on a line it drops the rest
# of that line from the PDF -- a table row's cells, silently -- while every check against the
# source still passes. The body has no comments after text, so any such sign is an error, unless it ends
# its line, where it drops nothing but the line break's space (``\author{%``).
_pct = [i + 1 for i, line in enumerate(_tex.splitlines())
        if line.strip() and not line.lstrip().startswith("%") and re.search(r"(?<!\\)%(?!\s*$)", line)]
check(not _pct, f"main.tex has an unescaped % after text on line(s) {_pct[:8]}; LaTeX drops the rest of each")


def _table_body(label):
    """The rows of one table environment, by its \\label."""
    if f"\\label{{{label}}}" not in _tex:
        check(False, f"main.tex has no table labelled {label}")
        return ""
    marker = _tex.index(f"\\label{{{label}}}")
    return _tex[marker:_tex.index("\\end{table}", marker)]


# app:learned's MMLU paragraph: the learner goes to the coordinate the shipped
# repair does not target, without being told the coordinate exists.
mmlu_learned = load("results/learned_probe_mmlu.json")
top_two = sorted(mmlu_learned["files"]["repaired"]["rank-only"]["weights"].items(),
                 key=lambda kv: -abs(kv[1]))[:2]
check({name for name, _ in top_two}
      == {"length, later ties rank 0", "length, later ties rank 3"},
      "the learner's two heaviest weights on the repaired MMLU file are no "
      f"longer the reversed tie-break's: {top_two}")
after_tie = sorted(mmlu_learned["files"]["repaired+tie"]["rank-only"]["weights"].items(),
                   key=lambda kv: -abs(kv[1]))[:4]
check(not any(name.startswith("length, later ties") for name, _ in after_tie),
      "repairing the reversed tie-break no longer takes it out of the learner's "
      f"heaviest four: {after_tie}")

# --- flat margins are not a flat joint ---------------------------------------
for path, label, n_files in (("results/joint_uniformity.json", "four-option", 3),
                             ("results/joint_uniformity_mmlupro.json", "ten-option", 2)):
    joint = load(path)
    check(len(joint["files"]) == n_files,
          f"the {label} joint measurement covers {len(joint['files'])} files")
    clean_joint = joint["clean"]["max"]
    for name, block_ in joint["files"].items():
        block_["_over"] = block_["joint_cv"] - clean_joint
    if label == "ten-option":
        # app:learned's negative result: neither file clears its own control.
        check(all(block_["joint_cv"] < clean_joint
                  for block_ in joint["files"].values()),
              "a ten-option file's joint now clears the clean control, so "
              f"app:learned's negative result is stale: {joint['files'].keys()}")
    else:
        released_joint = joint["files"]["released"]
        check(released_joint["joint_cv"] > clean_joint,
              "released v1.5's joint is no longer concentrated")
        drawn = joint["files"].get("repaired (joint draw)")
        check(drawn is not None and drawn["joint_cv"] < 0.05,
              "the joint draw no longer flattens the joint")
        marginal = joint["files"]["repaired (marginal draw)"]
        check(max(marginal["widest_margin"].values())
              < max(drawn["widest_margin"].values()),
              "flattening the joint no longer costs the margins, which is the "
              "trade app:learned reports")

# The same contrast at four options, where it settles a claim \S5 could
# otherwise only make as an interval: the rank effect is negative for every
# solver, and the placebo lifts every one of them.
four_placebo = Path("results/probe_bix_placebo")
if any(four_placebo.glob("*.jsonl.gz")):
    panel = {}
    for path in sorted(four_placebo.glob("*.jsonl.gz")):
        hits, total, model = {}, {}, None
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                model = model or row["model"]
                if not row["is_numeric"]:
                    continue
                hits[row["arm"]] = hits.get(row["arm"], 0) + row["correct"]
                total[row["arm"]] = total.get(row["arm"], 0) + 1
        rates = {arm: hits[arm] / total[arm] for arm in hits}
        released = rates["original_stemless"]
        panel[model] = (rates["placebo_stemless"] - released,
                        rates["repaired_stemless"] - rates["placebo_stemless"])
    # The claim that carries the argument is the rank effect: after the
    # construction, no solver is collecting rank-driven credit. The placebo's
    # own lift is a description of the rewriting and is not uniform in sign.
    # Appendix D's claim over the eleven-model panel, which is what it says:
    # negative for most, median negative, and no solver lifted by more than two
    # points. "Negative for every one" was true of the four-model early run and
    # is not true of the panel -- gemma-3-4b comes out at +1.4.
    positive = {model: effect for model, (_, effect) in panel.items() if effect > 0}
    check(len(positive) <= max(2, len(panel) // 4),
          f"the four-option rank effect is positive for {len(positive)} of "
          f"{len(panel)} solvers: { {k: round(v, 3) for k, v in positive.items()} }")
    check(max(positive.values(), default=0.0) < 0.02,
          f"a solver's four-option rank effect now exceeds two points: "
          f"{ {k: round(v, 3) for k, v in positive.items()} }")
    check(statistics.median(effect for _, effect in panel.values()) < 0,
          "the median four-option rank effect is no longer negative")
    lifted = [lift for lift, _ in panel.values()]
    check(statistics.median(lifted) > 0.005,
          f"the four-option placebo no longer lifts the median solver: "
          f"{statistics.median(lifted):+.3f}")
    check(len(panel) in (4, 11),
          f"the four-option placebo panel has {len(panel)} models, expected the "
          "four of an early run or the eleven of \\S4's panel")

# Appendix D's answer to its own residue: the stem-withheld placebo, which holds
# the key's rank and redraws the values, lifts the same solvers as much or more.
placebo_dir = Path("results/probe_mmlupro_placebo")
if any(placebo_dir.glob("*.jsonl.gz")):
    for path in sorted(placebo_dir.glob("*.jsonl.gz")):
        hits, total, model = {}, {}, None
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                model = model or row["model"]
                if not row["is_numeric"]:
                    continue
                hits[row["arm"]] = hits.get(row["arm"], 0) + row["correct"]
                total[row["arm"]] = total.get(row["arm"], 0) + 1
        rates = {arm: hits[arm] / total[arm] for arm in hits}
        check({"original_stemless", "placebo_stemless", "repaired_stemless"}
              <= set(rates),
              f"{model}: the placebo run is missing an arm: {sorted(rates)}")
        released = rates["original_stemless"]
        placebo = rates["placebo_stemless"] - released
        repaired = rates["repaired_stemless"] - released
        # Placebo minus repair is the rank effect, the contrast \S4 uses. After
        # the construction no solver collects, which is Appendix D's claim.
        rank_effect = repaired - placebo
        check(rank_effect < 0.01,
              f"{model}: the rank effect on the repaired ten-option file is "
              f"{rank_effect:+.3f}, so a solver is collecting again")
        check(placebo > 0.01,
              f"{model}: the stem-withheld placebo no longer lifts a solver "
              f"({placebo:+.3f}); Appendix D's account of the residue is stale")
        if "32B" in model:
            # The one that was collecting: the rank effect is where the channel
            # closing shows, and it is much larger than the other two.
            check(rank_effect < -0.03,
                  f"{model}: the collector's rank effect is only {rank_effect:+.3f}")

# And where the residue is not: attribution over eight coordinates -- the four
# audited and the four held out -- on the repaired ten-option file.
# Every attribution rebuilds the options the probe showed, from the question file
# and the seed, and records how much of that rebuild matched what the outcome
# file says. Anything below one means the repair has moved since the probe ran.
for _path in sorted(Path("results").glob("channel_attribution_*.json")):
    for _record in load(_path)["models"]:
        agreement = _record.get("rebuild_agreement")
        check(agreement == 1.0,
              f"{_path.name}: {_record['model']}'s conditions rebuild to "
              f"{agreement}, not 1.0 -- the repair has changed since the probe "
              "ran, so these channels are measured on options nobody saw")

held_attr = load("results/channel_attribution_mmlupro_construct_heldout.json")
check(len(held_attr["channels"]) == 9,
      f"the diagnostic attribution measures {len(held_attr['channels'])} channels, "
      "expected the four audited plus the five held out")
check(set(held_attr["channels"]) == SURFACES | {"sorted value"} | HELD_OUT,
      f"the diagnostic channels changed: {sorted(held_attr['channels'])}")
for record in held_attr["models"]:
    arm = record["arms"]["repaired_stemless"]
    collected = {name: block_["credit_collected"]
                 for name, block_ in arm["channels"].items()}
    available = {name: block_["credit_available"]
                 for name, block_ in arm["channels"].items()}
    check(max(abs(value) for value in collected.values()) < 0.01,
          f"{record['model']}: a coordinate now pays more than a point on the "
          f"repaired ten-option file: { {k: round(v, 4) for k, v in collected.items()} }")
    check(arm["accuracy"] - 1.0 / record["n_options"] > 0.005
          or "32B" in record["model"],
          f"{record['model']}: the residue this attribution explains away is gone "
          f"({arm['accuracy']:.4f}); Appendix D's paragraph is stale")
    check(max(available[name] for name in HELD_OUT) > 0.05,
          f"{record['model']}: the held-out coordinates are no longer open on the "
          f"repaired file, so there is nothing to rule out: {available}")

# --- five orderings no repair targeted -------------------------------------
# Appendix G's second half, and the one result in this paper that differs
# between the two benchmarks. Both directions are claims, so both are pinned.
held = load("results/held_out_orderings.json")
check(set(held["orderings"]) == HELD_OUT,
      f"the held-out orderings changed: {sorted(held['orderings'])}")
check(set(held["files"]) == {"4", "10"},
      f"the held-out experiment covers {sorted(held['files'])}, expected both option counts")
for option_count, block_ in held["files"].items():
    for label in ("released", "repaired"):
        check(set(block_[label]) == HELD_OUT | {"_coupling"},
              f"{option_count}/{label}: {sorted(block_[label])}")

four, ten = held["files"]["4"], held["files"]["10"]
clean_four = max(stats["widest"] for stats in held["clean"]["4"].values())
clean_ten = max(stats["widest"] for stats in held["clean"]["10"].values())
# Four options: the repair closes all five, four of which the released file leaks.
leaking_four = {name for name, stats in four["released"].items()
                if name != "_coupling" and stats["leaks"]}
check(len(leaking_four) == 4,
      f"BixBench now leaks on {len(leaking_four)} held-out orderings, not four: "
      f"{sorted(leaking_four)}")
for name, stats in four["repaired"].items():
    if name == "_coupling":
        continue
    check(not stats["leaks"] and (stats["bound"] or 0.0) < clean_four,
          f"the four-option repair no longer closes the held-out {name}: "
          f"{stats['bound']:+.3f} against a clean file's {clean_four:+.3f}")
check(max(stats["top_rank_share"] for name, stats in four["released"].items()
          if name != "_coupling") > 0,
      "the four-option held-out block lost its rank shares")

# Ten options: it closes none of the four that leak, which is the point.
leaking_ten = {name for name, stats in ten["released"].items()
               if name != "_coupling" and stats["leaks"]}
check(len(leaking_ten) == 4,
      f"MMLU-Pro now leaks on {len(leaking_ten)} held-out orderings: {sorted(leaking_ten)}")
still_open = {name for name, stats in ten["repaired"].items()
              if name != "_coupling" and stats["leaks"]}
check(still_open == leaking_ten,
      "the ten-option repair now moves a held-out ordering; Appendix G says it closes "
      f"none of them: released {sorted(leaking_ten)}, repaired {sorted(still_open)}")
for name in leaking_ten:
    check((ten["repaired"][name]["bound"] or 0.0) > clean_ten,
          f"the repaired ten-option {name} bound is now inside the clean-file range")

# And the diagnostic that explains the difference, which is what \S5 claims.
couplings = {(k, label): held["files"][k][label]["_coupling"]
             ["lexicographic_is_value_order"] for k in ("4", "10")
             for label in ("released", "repaired")}
check(abs(couplings[("4", "released")] - 0.676) < 0.01
      and abs(couplings[("4", "repaired")] - 0.686) < 0.01,
      f"BixBench's lexicographic/value agreement changed: {couplings}")
check(abs(couplings[("10", "released")] - 0.488) < 0.01
      and abs(couplings[("10", "repaired")] - 0.306) < 0.01,
      f"MMLU-Pro's lexicographic/value agreement changed: {couplings}")
check(couplings[("4", "repaired")] > 2 * couplings[("10", "repaired")] - 0.35,
      "the two files' coupling no longer separates, so \\S5's explanation is stale")
check(couplings[("10", "repaired")] < couplings[("10", "released")] - 0.1,
      "the ten-option repair no longer lowers the coupling, which Appendix G says "
      "it does by rendering distractors at varied precision")
# The degeneracy caution in Appendix G is measured, so pin the measurement.
check(four["released"]["_coupling"]["decimal places"]["mean_distinct_scores"] < 1.5,
      "decimal places is no longer degenerate on released v1.5; the caution is stale")
check(ten["released"]["_coupling"]["digit sum"]["mean_distinct_scores"] > 7.0,
      "digit sum is no longer the non-degenerate one of the four")

# Appendix G's last paragraph: targeting the held-out lexicographic coordinate
# narrows it and closes nothing, and the frontier says so in advance.
built_lex = ten.get("repaired_lexicographic")
check(built_lex is not None,
      "the ten-option block no longer carries the lexicographic-targeted repair")
check(built_lex["string order"]["leaks"]
      and built_lex["string order"]["bound"] < ten["repaired"]["string order"]["bound"],
      "targeting lexicographic order no longer narrows it without closing it: "
      f"{built_lex['string order']['bound']:+.3f} against "
      f"{ten['repaired']['string order']['bound']:+.3f}")
lex_audit = load("results/mcq_audit_mmlu_pro_lexicographic.json")
lex_after = {name: block_["numeric items"]["credit_bound"]["credit_lower_bound"]
             for name, block_ in
             lex_audit["after_repair"]["surface_channels"]["orderings"].items()}
# This used to assert the opposite: that the fifth coordinate reopened roundness.
# It did, until the isolation-tuning stage was taught to treat every assigned
# coordinate as part of a rendering's class; the trade was a defect in the
# repair, not a property of the file, and the check now pins the corrected
# reading -- all four audited coordinates stay shut.
check(all(value < 0 for value in lex_after.values()),
      "the five-coordinate repair now reopens an audited coordinate, which "
      f"Appendix G says it does not: { {k: round(v, 4) for k, v in lex_after.items()} }")
check(lex_audit["surface_target_hit_rate"]["assigned"]["lexicographic order"] == 1.0,
      "the five-coordinate repair no longer meets its lexicographic target on "
      "every item, which is the fix Appendix G describes: "
      f"{lex_audit['surface_target_hit_rate']['assigned']['lexicographic order']:.3f}")
lex_fit = lex_audit["surface_target_hit_rate"]["calibration_worst_rank_share"][-1]
attainable_lex = lex_fit["widest_attainable_rank_share"]
check(abs(attainable_lex["lexicographic order"] - 0.162) < 0.004,
      "the flattest attainable lexicographic marginal changed: "
      f"{attainable_lex['lexicographic order']:.4f}")
check(attainable_lex["lexicographic order"] > 0.10 + 0.05,
      "a uniform lexicographic rank is now attainable on the ten-option file, so "
      "Appendix G's limit claim is stale")
check(abs(lex_fit["mean_frontier_size"] - 36.1) < 0.5,
      f"the five-coordinate frontier changed: {lex_fit['mean_frontier_size']:.1f}")

# The same four orderings across the vendored survey: Appendix G's claim that
# the file whose value channel is closed is closed on these too.
sweep = load("results/held_out_survey.json")
check(set(sweep["orderings"]) == HELD_OUT,
      f"the swept orderings changed: {sorted(sweep['orderings'])}")
leaks_by_file = {name: sorted(o for o, stats in block_.items()
                              if o != "_coupling" and stats["leaks"])
                 for name, block_ in sweep["files"].items()}
check(len(leaks_by_file) == 5,
      f"the sweep covers {len(leaks_by_file)} files: {sorted(leaks_by_file)}")
check(leaks_by_file.get("aqua_rat:5") == [],
      "AQuA-RAT now leaks on a held-out ordering, which \\S3 and Appendix G say "
      f"it does not: {leaks_by_file.get('aqua_rat:5')}")
check(set(leaks_by_file.get("mmlu:4", []))
      == {"decimal places", "string order", "length, later ties"},
      f"MMLU's held-out leaks changed: {leaks_by_file.get('mmlu:4')}")
check(len(leaks_by_file.get("mmlu_pro:10", [])) == 4,
      f"MMLU-Pro's held-out leaks changed: {leaks_by_file.get('mmlu_pro:10')}")
# The reversed tie-break behaves like the rest of the family rather than like a
# coordinate of its own: open exactly on the two files the others are open on.
tie_open = {name for name, leaks in leaks_by_file.items() if "length, later ties" in leaks}
check(tie_open == {"mmlu:4", "mmlu_pro:10"},
      f"the reversed tie-break now leaks on {sorted(tie_open)}, not on the two "
      "files the other held-out orderings leak on")
for name in ("medmcqa:4", "medqa_usmle:4"):
    check(leaks_by_file.get(name) == [],
          f"{name} now leaks on a held-out ordering: {leaks_by_file.get(name)}")
# The coupling separates the two groups, which is the reading Appendix G gives.
closed = [name for name, leaks in leaks_by_file.items() if not leaks]
open_ = [name for name, leaks in leaks_by_file.items() if leaks]
couple = {name: sweep["files"][name]["_coupling"]["lexicographic_is_value_order"]
          for name in leaks_by_file}
check(min(couple[name] for name in closed) > max(couple[name] for name in open_),
      "the coupling no longer separates the files that leak on a held-out "
      f"ordering from those that do not: { {k: round(v, 3) for k, v in couple.items()} }")

# --- a tie-break is an ordering (Appendix G's last paragraph) --------------
# The audit ranks written length with ties to the earlier text; the rule family
# takes the later. Both directions of that paragraph are claims, so both are
# pinned: what the shipped repair leaves open, and what targeting it costs.
tie_three = load("results/held_out_mmlu.json")["files"]["4"]
tie_four = load("results/held_out_mmlu_tie.json")["files"]["4"]
tie_clean = load("results/held_out_mmlu.json")["clean"]["4"]["length, later ties"]["widest"]
TIE = "length, later ties"
for label, value, want in (
        ("released", tie_three["released"][TIE]["bound"], 0.018),
        ("after the three-coordinate repair", tie_three["repaired"][TIE]["bound"], 0.026),
        ("after the four-coordinate repair", tie_four["repaired"][TIE]["bound"], -0.029),
        ("on clean files", tie_clean, 0.006)):
    check(abs(value - want) < 0.0006,
          f"MMLU's reversed tie-break {label} is {100 * value:+.1f} points, and "
          f"Appendix G says {100 * want:+.1f}")
check(tie_three["released"][TIE]["leaks"] and tie_three["repaired"][TIE]["leaks"]
      and not tie_four["repaired"][TIE]["leaks"],
      "the paragraph's shape is gone: the shipped repair is supposed to leave "
      "this coordinate open and the four-coordinate one to close it")

# The same coordinate at ten options, where the construction has to be wider.
tie_ten = load("results/held_out_mmlu_pro_tie.json")["files"]["10"]["repaired"]
check(abs(tie_ten[TIE]["bound"] + 0.016) < 0.0006 and not tie_ten[TIE]["leaks"],
      f"the ten-option reversed tie-break is {100 * tie_ten[TIE]['bound']:+.1f} "
      "points after the four-coordinate repair, and Appendix G says -1.6 and closed")
check(sum(1 for name, stats in tie_ten.items()
          if not name.startswith("_") and stats["leaks"]) == 3,
      "the four-coordinate ten-option repair no longer leaves exactly three "
      "held-out orderings open, which is Appendix G's reading")
tie_ten_audit = load("results/mcq_audit_mmlu_pro_tie.json")
ten_after = {name: block_["numeric items"]["credit_bound"]["credit_lower_bound"]
             for name, block_ in
             tie_ten_audit["after_repair"]["surface_channels"]["orderings"].items()}
check(all(value < 0 for value in ten_after.values()),
      "the ten-option four-coordinate repair reopens an audited coordinate: "
      f"{ {k: round(v, 4) for k, v in ten_after.items()} }")
ten_fit = tie_ten_audit["surface_target_hit_rate"]["calibration_worst_rank_share"][-1]
check(abs(ten_fit["mean_frontier_size"] - 146.2) < 1.0,
      f"the four-coordinate ten-option frontier changed: "
      f"{ten_fit['mean_frontier_size']:.1f}, and Appendix G says 146.2")
check(abs(ten_fit["widest_attainable_rank_share"][TIE] - 0.144) < 0.004,
      "the flattest attainable reversed-tie marginal at ten options changed: "
      f"{ten_fit['widest_attainable_rank_share'][TIE]:.4f}")

tie_audit = load("results/mcq_audit_mmlu_tie.json")["surface_target_hit_rate"]
check(tie_audit["assigned"][TIE] == 1.0,
      f"the four-coordinate repair meets its new target on "
      f"{tie_audit['assigned'][TIE]:.1%} of items, and Appendix G says all of them")
check(abs(max(tie_audit["realised_rank_share"][TIE]) - 0.309) < 0.001,
      "the widest rank the four-coordinate repair leaves the key at under the "
      f"reversed tie-break is {max(tie_audit['realised_rank_share'][TIE]):.3f}, "
      "and Appendix G says 0.309")

# The concrete form of the paragraph, recomputed from the shipped repairs: the
# audited length rank is as flat as the repair can make it and the rule that
# reads the other tie-break still finds the key a third of the time.
from text_artifacts import RULES as _RULES  # noqa: E402
from text_artifacts import pick as _pick  # noqa: E402
from mcq_audit import ordering_rank as _ordering_rank  # noqa: E402

for label, path, rank0, hits in (
        ("released", Path("build/mmlu.jsonl"), 0.191, 0.238),
        ("three-coordinate", Path("results/repaired/mmlu_repaired_multi.jsonl.gz"),
         0.250, 0.366),
        ("four-coordinate", Path("results/repaired/mmlu_repaired_tie.jsonl.gz"),
         0.244, 0.309)):
    if not path.exists():
        check(label == "released",
              f"the {label} MMLU repair is missing from results/repaired")
        continue
    numeric = [item for item in _build_items(_read_rows(path), "ideal", "distractors",
                                             "cluster", "question", 4)[0]
               if item["rank"] is not None]
    check(len(numeric) == 661,
          f"MMLU {label}: {len(numeric)} numeric items, expected 661")
    flat = sum(_ordering_rank(item["options"], "written length") == 0
               for item in numeric) / len(numeric)
    taken = sum(_pick(_RULES["shortest_option"], item["options"], item["question"]) == 0
                for item in numeric) / len(numeric)
    check(abs(flat - rank0) < 0.002 and abs(taken - hits) < 0.002,
          f"MMLU {label}: the key is at length rank 0 on {flat:.1%} of items and "
          f"`take shortest_option` finds it {taken:.1%} of the time; Appendix G "
          f"says {rank0:.1%} and {hits:.1%}")

seeds = load("results/mmlu_tie_seeds.json")["repairs"]
for coordinate, three_, four_ in (("written length", 0.029, 0.002),
                                  ("significant digits", 0.016, 0.012),
                                  ("lexical isolation", -0.001, 0.011)):
    for name, want in (("three coordinates", three_), ("four coordinates", four_)):
        got = seeds[name]["worst_bound"][coordinate]
        check(abs(got - want) < 0.0006,
              f"over eight seeds the {name} repair's worst {coordinate} bound is "
              f"{100 * got:+.1f}, and Appendix G says {100 * want:+.1f}")
check(len(seeds["four coordinates"]["per_seed"]) == 8,
      "the seed sweep is no longer eight seeds")

# The models, recomputed from the two probe directories rather than from a
# summary: the point of the paragraph is that a 5.5-point move in the bound is
# half a point in what is taken.
tie_moves = []
for path in sorted(Path("results/probe_mmlu_tie").glob("*.jsonl.gz")):
    other = Path("results/probe_mmlu_construct") / path.name
    if not other.exists():
        continue
    scores = {}
    for label, source in (("four", path), ("three", other)):
        hits = {}
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not row["is_numeric"] or not row["arm"].startswith("repaired"):
                    continue
                got, seen = hits.get(row["arm"], (0, 0))
                hits[row["arm"]] = (got + row["correct"], seen + 1)
        scores[label] = {arm: got / seen for arm, (got, seen) in hits.items()}
    for arm in ("repaired", "repaired_stemless"):
        tie_moves.append(scores["four"][arm] - scores["three"][arm])
check(len(tie_moves) == 20,
      f"the tie-break probe covers {len(tie_moves) // 2} models, expected ten")
check(max(abs(move) for move in tie_moves) < 0.01,
      "a model now moves by more than a point between the three- and "
      f"four-coordinate repairs: {max(abs(m) for m in tie_moves):.4f}")

# --- and whether the channel can be learned in context ---------------------
icl = load("results/icl_probe.json")
check(abs(icl["geometry_credit_available"] - 0.264) < 0.002,
      "the ICL probe is no longer run where 26.4 points are available")
check(len(icl["models"]) == 8, f"the in-context panel has {len(icl['models'])} models, not the eight app:arms names")
for record in icl["models"]:
    chance = 1.0 / record["n_options"]
    base = record["arms"]["withheld_s0_none"]["accuracy"]
    check(abs(base - chance) < 0.08,
          f"{record['model']}: zero-shot stemless is {base:.3f}, not near chance")
    # Demonstrations do move solvers -- up to +6.1 points at 64 shots -- so the
    # claim is not that they do nothing. It is that they move a solver just as
    # far when their key ranks are uniform, which is the rank-isolating
    # contrast and the only one the repaired-demonstration arm exists to form.
    for contrast in record["contrasts"]:
        if contrast["question"] != "attributable to key ranks":
            continue
        check(abs(contrast["difference"]) < 0.03,
              f"{record['model']}: {contrast['contrast']} moved "
              f"{contrast['difference']:+.3f}; the in-context null is stale")
# Forty-eight contrasts at the 5% level turn up two or three by chance, so the check is
# on the count, and on the directions the appendix states: one each way.
rank_contrasts = [contrast for record in icl["models"] for contrast in record["contrasts"]
                  if contrast["question"] == "attributable to key ranks"]
check(len(rank_contrasts) == 48, f"{len(rank_contrasts)} rank contrasts, expected 48")
nominal = [contrast for contrast in rank_contrasts if contrast["p_value"] < 0.05]
check(len(nominal) == 2 and sorted(c["difference"] > 0 for c in nominal) == [False, True],
      f"{len(nominal)} of 48 rank contrasts clear at 5% "
      f"({[round(100 * c['difference'], 1) for c in nominal]}); app:arms says two, one in each direction")
check("Of the $48$ such contrasts, $2$ are significant at $5\\%$, one in each direction." in _flat,
      "app:arms no longer counts the in-context panel's nominal rank contrasts")
# the rank-isolating trends over model size are flat; the any-example ones are not, and the paper says so
trend = icl["scale_trend"]
for stem in ("withheld", "shown"):
    rec = trend[f"attributable to key ranks ({stem}, 64 shots)"]
    check(rec["p_value"] > 0.05, f"the in-context rank trend over size ({stem}) is now significant "
          f"(rho={rec['spearman_rho']:+.2f}, p={rec['p_value']:.3f})")
_rw, _rs = (trend[f"attributable to key ranks ({s_}, 64 shots)"]["spearman_rho"] for s_ in ("withheld", "shown"))
_lw, _ls = (trend[f"learnable at all ({s_}, 64 shots)"]["spearman_rho"] for s_ in ("withheld", "shown"))
check(f"with no trend in size ($\\rho={_rw:+.2f}$ and ${_rs:+.2f}$)" in _flat,
      "app:arms' in-context rank trends over size are stale")
check(f"(over models, $\\rho={_lw:+.2f}$ with size; with the question shown they lower it, $\\rho={_ls:+.2f}$)"
      in _flat, "app:arms' in-context example trends over size are stale")
learned = {record["model"]: contrast["difference"] for record in icl["models"]
           for contrast in record["contrasts"]
           if contrast["question"] == "learnable at all"
           and contrast["stem"] == "withheld" and contrast["n_shots"] == 64}
check(max(learned.values()) > 0.03,
      "no model gains from demonstrations at all, so the repaired-demonstration "
      "control is not doing any work; \\S4 says it is")
check(f"Qwen2.5-14B's by ${100 * learned['Qwen/Qwen2.5-14B-Instruct']:+.1f}$ and Qwen2.5-72B's by "
      f"${100 * learned['Qwen/Qwen2.5-72B-Instruct']:+.1f}$ points at $64$ examples" in _flat,
      "app:arms' in-context gains from any examples are stale")
_geo = [100 * r["arms"][f"withheld_s64_{pool}"]["realised_geometry"]["geometry_credit"]
        for r in icl["models"] if r["model"] in ("Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
                                                 "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct")
        for pool in ("original", "repaired")]
check(2.5 <= min(_geo) and max(_geo) <= 4.05 and "gain about $3$ of the $26.4$ points the rank offers" in _flat,
      f"the larger models' rank credit with examples is {min(_geo):.1f} to {max(_geo):.1f}, not about 3")
_r64 = {st: [100 * c["difference"] for c in rank_contrasts if c["n_shots"] == 64 and c["stem"] == st]
        for st in ("withheld", "shown")}
check(f"released minus uniform is ${min(_r64['withheld']):+.1f}$ to ${max(_r64['withheld']):+.1f}$ points at $64$ "
      f"examples with the question withheld and ${min(_r64['shown']):+.1f}$ to ${max(_r64['shown']):+.1f}$ with it "
      "shown" in _flat,
      "app:arms' in-context rank contrasts at 64 examples are stale")
check("solved v1.5 items in context teach eight nothing about the key's rank" in _flat
      and max(abs(v) for v in _r64["withheld"]) < 2 and len(icl["models"]) == 8,
      "sec:open's in-context sentence is stale")
import icl_analysis
_icl_rows = [" ".join(r.split()) for r in icl_analysis.table_rows(icl)]
_icl_body = " ".join(_table_body("tab:icl").split())
check(all(r in _icl_body for r in _icl_rows), "tab:icl's rows no longer match results/icl_probe.json")

# --- the controlled probe: arms, controls, and the pre-specified trend ------
probe = load("results/no_data_probe.json")
check(len(probe["models"]) == 11, "the model panel size changed")
for model in probe["models"]:
    block = model["subsets"]["numeric options"]
    check(set(block["arms"]) == {"original", "placebo", "repaired",
                                 "original_stemless", "repaired_stemless"},
          f"{model['model']}: arm set changed")
    check(model["n_draws"] == 20, f"{model['model']}: redraw count changed")
rank_effects = [c["difference"] for m in probe["models"]
                for c in m["subsets"]["numeric options"]["contrasts"]
                if c["contrast"] == "placebo - repaired"]
check(abs(min(rank_effects) + 0.037) < 0.004 and abs(max(rank_effects) - 0.058) < 0.004,
      f"model rank effects changed: {min(rank_effects):+.3f} to {max(rank_effects):+.3f}")
trend = probe["scale_trend"]["placebo - repaired"]
check(trend["n_models"] == 11 and trend["p_value"] > 0.05,
      "the pre-specified scale trend is now significant; \\S4 says it is not")

rules = load("results/no_data_probe_rules.json")
rule_arms = {m["model"]: m["subsets"]["numeric options"] for m in rules["models"]}
for name, expected_original in (("rule:rank-0", 0.124), ("rule:rank-1", 0.514),
                                ("rule:rank-2", 0.295), ("rule:rank-3", 0.067)):
    got = rule_arms[name]["arms"]["original"]["accuracy"]
    check(abs(got - expected_original) < 0.001,
          f"{name} no longer scores the key-rank share exactly: {got:.4f}")
    placebo = rule_arms[name]["arms"]["placebo"]["accuracy"]
    check(abs(placebo - got) < 1e-9,
          f"{name}: the placebo moved a rank-only solver, so it does not hold rank fixed")
positive = next(c for c in rule_arms["rule:rank-1"]["contrasts"]
                if c["contrast"] == "placebo - repaired")
check(positive["difference"] > 0.20 and positive["p_value"] < 0.01,
      "the positive control no longer moves; the panel's null is uninformative")
negative = next(c for c in rule_arms["rule:uniform-random"]["contrasts"]
                if c["contrast"] == "placebo - repaired")
check(abs(negative["difference"]) < 0.02 and negative["p_value"] > 0.05,
      "the flat-preference control now moves, which would indicate a pipeline artefact")

# --- the read-out predicts what the models actually write -------------------
scoring = load("results/scoring_validation.json")
check(scoring["summary"]["n_models"] == 11, "scoring validation does not cover the panel")
check(scoring["summary"]["min_agreement"] > 0.95,
      f"forced-choice agreement fell to {scoring['summary']['min_agreement']:.3f}")

# --- the estimator is unbiased and the test is correctly sized --------------
calibration = load("results/choice_model_calibration.json")
null = calibration["null_lambda0_interior_bias"]
check(null["lrt_rejection_rate_at_0.05"] < 0.08,
      f"the knowledge test is no longer correctly sized: {null['lrt_rejection_rate_at_0.05']:.3f}")
check(null["mean_lambda_hat"] < 0.03, "lambda is now biased upward under the null")
for key, truth in (("true_lambda_0.10", 0.10), ("true_lambda_0.20", 0.20)):
    got = calibration[key]["mean_lambda_hat"]
    check(abs(got - truth) < 0.01, f"{key}: estimator is biased, mean {got:.3f}")

# --- the confounding result -------------------------------------------------
contrasts = {c["restriction"]: c for c in baseline["confounding_analysis"]["contrasts"]}
check(contrasts["all questions"]["p_value"] < 0.05, "pooled provenance gap is no longer significant")
for restricted in ("numeric-option questions only", "LLM-graded questions only"):
    check(contrasts[restricted]["p_value"] > 0.05,
          f"gap now survives restriction to {restricted}; \\S5 is stale")

# --- power planning ---------------------------------------------------------
power = load("results/power.json")
sigma = power["grounded_rows"][0]["sigma_hat"]
check(abs(sigma - 0.230) < 0.001, f"measured sigma changed: {sigma:.4f}")
pairs = {r["rho"]: r["pairs_for_10pp"] for r in power["grounded_rows"]}
check((pairs[0.0], pairs[0.5], pairs[0.9]) == (86, 44, 11),
      f"required-pairs row changed: {pairs}")

# --- bibliography hygiene ---------------------------------------------------
tex = Path("main.tex").read_text(encoding="utf-8")
bib = Path("refs.bib").read_text(encoding="utf-8")
keys = re.findall(r"@\w+\{([^,]+),", bib)
check(len(keys) == len(set(keys)), "duplicate BibTeX key")
cited = {k.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex) for k in group.split(",")}
check(cited <= set(keys), f"citation without a bib entry: {cited - set(keys)}")
check(not re.search(r"\band\s+and\b", bib), "empty BibTeX author")

# --- app:exchangeable: removing the geometry does not remove the leak --------
# Every number in that appendix comes from two shipped files, and the two
# solvers in it have to be reading different things: one is allowed the option
# set, the other is allowed one option at a time.
exch = load("results/exchangeable_mmlu.json")
check(exch["n_feasible"] + exch["n_infeasible"] == exch["n_numeric"],
      "the exchangeable repair's feasible and infeasible counts do not add up")
check(exch["n_infeasible"] == 67 and exch["n_numeric"] == 661,
      f"the feasibility frontier moved: {exch['n_feasible']}/{exch['n_numeric']} "
      f"feasible, {exch['n_infeasible']} not")
# The slot is drawn uniformly among k sets already known to be legal, so this is
# uniform by construction; a drift here means the all-slots rule has been lost.
check(max(abs(share - 0.25) for share in exch["key_slot_share"]) < 0.04,
      f"the key slot is no longer drawn uniformly: {exch['key_slot_share']}")
naive = exch["naive_retry_rank_share"]
check(max(naive) - 0.25 > 0.05,
      "the one-slot retry no longer skews the key's rank, which is the whole "
      f"reason the all-slots rule is there: {naive}")
_value = exch["channels"]["orderings"]["sorted value"]
_value = (_value.get("numeric items") or next(iter(_value.values())))
check(_value["credit_bound"]["credit_lower_bound"] < 0.005,
      "the exchangeable repair no longer closes the value channel: "
      f"{_value['credit_bound']['credit_lower_bound']:+.4f}")

# The five columns of tab:exch, against the two files that produced them.
exch_learned = load("results/learned_probe_exchangeable.json")
identity = load("results/key_identity_mmlu.json")
check([block_["argmax_ties"] for _f in exch_learned["files"].values()
       for block_ in (_f[fam] for fam in FAMILIES)].count(0) == 9,
      "a learned-probe family now ties at the top, so its accuracy is partly "
      "the tie-break; the exchangeable table reads it as if it were not")
# The bar app:exchangeable reads every bound against: a clean file of the same
# shape as the measured one, because the widest bound a clean file produces
# depends on how many items and clusters it has.
_calib594 = load("results/audit_calibration_594.json")
check(_calib594["n_items"] == 594 and _calib594["n_clusters"] == 37
      and _calib594["n_options"] == 4,
      "the clean reference app:exchangeable reads against is no longer shaped "
      f"like the measured file: {_calib594['n_items']} items, "
      f"{_calib594['n_clusters']} clusters")
_worst594 = {name.split(" (")[0]: value
             for name, value in _calib594["widest_bound_on_a_clean_file"].items()
             if "numeric" in name}
_exch_audited = {}
for _name, _block in exch["channels"]["orderings"].items():
    _entry = _block.get("numeric items") or next(iter(_block.values()))
    _exch_audited[_name] = _entry["credit_bound"]["credit_lower_bound"]
_exch10 = load("results/exchangeable_mmlu_pro.json")
_exch10_audited = {}
for _name, _block in _exch10["channels"]["orderings"].items():
    _entry = _block.get("numeric items") or next(iter(_block.values()))
    _exch10_audited[_name] = _entry["credit_bound"]["credit_lower_bound"]
for _name, _typed in (("written length", 5.5), ("significant digits", 7.0)):
    check(abs(100 * _exch10_audited[_name] - _typed) < 0.06,
          f"the appendix types {_typed:+.1f} for {_name} on the ten-option "
          f"exchangeable file, measured {100 * _exch10_audited[_name]:+.2f}")
check(_exch10_audited["sorted value"] < 0.005,
      "the ten-option exchangeable repair no longer closes the value channel")
for _name, _typed, _clean_typed in (("significant digits", 5.8, 1.5),
                                    ("lexical isolation", 3.0, 0.9),
                                    ("written length", 1.4, 1.0)):
    check(abs(100 * _exch_audited[_name] - _typed) < 0.06,
          f"app:exchangeable types {_typed:+.1f} for {_name} on the exchangeable "
          f"file, measured {100 * _exch_audited[_name]:+.2f}")
    check(abs(100 * _worst594[_name] - _clean_typed) < 0.06,
          f"app:exchangeable types a clean {_clean_typed:+.1f} for {_name}, "
          f"measured {100 * _worst594[_name]:+.2f}")
    check(_exch_audited[_name] > _worst594[_name],
          f"{_name} no longer clears its clean control on the exchangeable file, "
          "so app:exchangeable's list of what stays open is stale")

# The five held-out orderings on the same file, each against a clean file of the
# same shape: three close and two do not, and app:exchangeable types all five.
_held_exch = load("results/held_out_exchangeable.json")
_clean_exch = {name: block_["widest"]
               for name, block_ in _held_exch["clean"]["4"].items()}
_repaired_exch = {name: block_["bound"]
                  for name, block_ in _held_exch["files"]["4"]["repaired"].items()
                  if not name.startswith("_")}
_closed = {name for name, got in _repaired_exch.items() if got <= _clean_exch[name]}
check(_closed == {"decimal places", "leading digit", "string order"},
      "app:exchangeable says three of the five held-out orderings close on the "
      f"exchangeable file; the set is now {sorted(_closed)}")
for _name, _typed, _clean_typed in (
        ("decimal places", 0.6, 0.9), ("leading digit", -2.5, 1.0),
        ("string order", 0.6, 1.2), ("digit sum", 2.4, 0.3),
        ("length, later ties", 2.7, 2.3)):
    check(abs(100 * _repaired_exch[_name] - _typed) < 0.06,
          f"app:exchangeable types {_typed:+.1f} for {_name}, measured "
          f"{100 * _repaired_exch[_name]:+.2f}")
    check(abs(100 * _clean_exch[_name] - _clean_typed) < 0.06,
          f"app:exchangeable types a clean {_clean_typed:+.1f} for {_name}, "
          f"measured {100 * _clean_exch[_name]:+.2f}")

# The weights app:exchangeable quotes, from the file they are in. They moved
# between two different solvers once already, which is what this guards.
_heaviest = {}
for _label in ("repaired (rank)", "exchangeable offsets"):
    _w = exch_learned["files"][_label]["every feature"]["weights"]
    _heaviest[_label] = sorted(_w.items(), key=lambda kv: -abs(kv[1]))[:2]
check(_heaviest["repaired (rank)"][0][0] == "relative magnitude"
      and abs(_heaviest["repaired (rank)"][0][1] + 2.53) < 0.005,
      "the rank-repaired file's heaviest non-rank weight is no longer relative "
      f"magnitude at -2.53: {_heaviest['repaired (rank)']}")
check([name for name, _ in _heaviest["exchangeable offsets"]]
      == ["has separator", "trailing zeros"],
      "the exchangeable file's two heaviest weights are no longer has separator "
      f"and trailing zeros: {_heaviest['exchangeable offsets']}")
for _name, _typed in _heaviest["exchangeable offsets"]:
    check(abs(abs(_typed) - (2.56 if _name == "has separator" else 1.15)) < 0.005,
          f"app:exchangeable types a different weight for {_name}: {_typed:+.3f}")

# The finding, not just the table: the leak barely moves when the geometry goes,
# and the per-option solver finds nothing on the file a human wrote.
_every = {label: block_["every feature"]["cv_accuracy"]
          for label, block_ in exch_learned["files"].items()}
check(_every["exchangeable offsets"] > _every["repaired (rank)"] - 0.02,
      "the exchangeable repair now closes most of the non-rank leak, which "
      f"would make app:exchangeable's conclusion wrong: {_every}")
_identity_released = identity["files"]["released"]["written form"]["cv_accuracy"]
check(_identity_released < identity["clean"]["written form"]["max"],
      "a per-option solver now beats its control on the released MMLU file, so "
      "the claim that the authorship signal is introduced by repair is stale")
for _name in ("relative", "median", "rank"):
    check(not any(_name in _f for _f in identity["features"]),
          f"key_identity.py grew a feature naming {_name!r}; its margins would "
          "no longer be free of option geometry")

# --- tab:fits: every fitted pair in the body, cell by cell -------------------
# The table is the paper's central estimator, so nothing in it is typed by hand
# without being read back out of the run files it came from.
FIT_SOURCES = (("results/choice_model.json", "BixBench v1.5"),
               ("results/choice_model_mmlu.json", "MMLU"),
               ("results/choice_model_mmlupro.json", "MMLU-Pro"))
fitted = {}
for path, bench in FIT_SOURCES:
    document = load(path)
    for key in ("published_runs", "open_weight_runs"):
        for run in document.get(key, []):
            fitted[(bench, run["run"])] = run

check(len(fitted) == 19,
      f"the run files now hold {len(fitted)} pairs, not 19")


rejects = {"BixBench v1.5": [0, 0], "MMLU": [0, 0], "MMLU-Pro": [0, 0]}
for (bench, name), run in fitted.items():
    diagnostics = run.get("diagnostics_per_draw") or {}
    if diagnostics:
        rejects[bench][0] += diagnostics["goodness_of_fit_rejected_at_0.05"]
        rejects[bench][1] += diagnostics["n_draws"]
    # The identity itself: the two terms reconstruct the margin the fit implies,
    # exactly, and the fit sits close to what was observed.
    check(abs(run["knowledge_component"] + run["geometry_component"]
              - (run["fitted_accuracy"] - run["chance"])) < 1e-9,
          f"the decomposition no longer adds up for {name} on {bench}")

# What the fits say, pinned to the run files the repository ships.
check(max(abs(100 * r["fitted_accuracy"] - 100 * r["accuracy"]) for r in fitted.values())
      < 1.3,
      "sec:probe claims fitted and observed accuracy agree to within 1.3 points")
check(sum(abs(100 * r["fitted_accuracy"] - 100 * r["accuracy"]) >= 0.5
          for r in fitted.values()) == 1,
      "sec:probe claims all but one pair agree to within 0.5 points")
check(rejects["BixBench v1.5"] == [19, 220],
      "sec:probe says the goodness-of-fit test rejects on 19 of 220 BixBench "
      f"draws; the run files say {rejects['BixBench v1.5']}")
check(rejects["MMLU"] == [12, 40],
      "sec:probe says 12 of 40 on MMLU; the run files say "
      f"{rejects['MMLU']}")
check(max(abs(100 * r["geometry_component"]) for r in fitted.values()) <= 2.61,
      "the introduction's -1.3 to +2.6 geometry range has drifted")

# sec:open quotes the same four models' BixBench lambdas beside their MMLU ones.
# A summary such as "<= 0.03" would drop the 32B model, so the four values are
# typed out and checked one by one.
for _model, _typed in (("Llama-3.2-1B-Instruct", 0.019),
                       ("Meta-Llama-3.1-8B-Instruct", 0.025),
                       ("Qwen2.5-7B-Instruct", 0.024),
                       ("Qwen2.5-32B-Instruct", 0.168)):
    _run = next(r for (b, n), r in fitted.items()
                if b == "BixBench v1.5" and n.split("/")[-1] == _model)
    check(abs(_run["lambda"] - _typed) < 0.0006,
          f"sec:open types {_typed} for {_model} on BixBench, against "
          f"{_run['lambda']:.4f}")
    if _model == "Qwen2.5-32B-Instruct":
        _mmlu = next(r for (b, n), r in fitted.items()
                     if b == "MMLU" and n.split("/")[-1] == _model)
        check(_run["lambda"] < _mmlu["lambda"] / 3,
              "sec:open says the 32B model's BixBench lambda is under a third of "
              f"its MMLU one: {_run['lambda']:.3f} against {_mmlu['lambda']:.3f}")

# --- sec:authorship: the prediction is about who wrote the distractors -------
# Two released files with written distractors, one with generated ones. The
# claim is that a learner reads only the third, so all four columns are tested.
WRITTEN = (("results/learned_probe_exchangeable.json",
            "results/key_identity_mmlu.json", "MMLU"),
           ("results/learned_probe_medmcqa.json",
            "results/key_identity_medmcqa.json", "MedMCQA"))
for set_file, own_file, bench in WRITTEN:
    for path, families in ((set_file, ("audited ranks", "every feature")),
                           (own_file, ("written form", "roundness"))):
        document = load(path)
        for family in families:
            stats = document["files"]["released"][family]
            margin = 100 * (stats["cv_accuracy"] - document["clean"][family]["max"])
            check(margin < 0.6,
                  f"released {bench} now hands a learner {margin:+.1f} points on "
                  f"{family}, so sec:authorship's written-distractor null is stale")
# sec:limits reports why BixBench carries no learned margin: its clean control
# sits so far above chance at 105 items that there is little range left.
_bix_learned = load("results/learned_probe.json")
_bix_clean = [100 * block["max"] for block in _bix_learned["clean"].values()]
check(abs(min(_bix_clean) - 31.4) < 0.06 and abs(max(_bix_clean) - 36.2) < 0.06,
      "sec:limits says BixBench's clean control runs 31.4% to 36.2%, against "
      f"{min(_bix_clean):.1f}% to {max(_bix_clean):.1f}%")
check(all(block["n_items"] == 105 for block in _bix_learned["files"].values()),
      "sec:limits says BixBench has 105 four-option numeric items")

# --- fig:provenance: the prediction across every released file we have ------
# One bar above zero, and it must be the one benchmark whose distractors a model
# wrote. If a second file ever clears its control, the figure's claim is false
# and this check is how we find out.
PROVENANCE = (("MMLU-Pro", "key_identity_mmlu_pro.json", "generated"),
              ("MMLU", "key_identity_mmlu.json", "written"),
              ("MedMCQA", "key_identity_medmcqa.json", "written"),
              ("LAB-Bench SeqQA", "key_identity_labbench_seqqa.json", "computed"),
              ("AQuA-RAT", "key_identity_aqua_rat.json", "computed"),
              ("BixBench v1.5", "key_identity_bixbench_v15.json", "computed"),
              ("MedQA-USMLE", "key_identity_medqa_usmle.json", "written"),
              ("SciQ", "key_identity_sciq.json", "unclear"),
              ("ARC-Challenge", "key_identity_arc_challenge.json", "written"),
              ("OpenBookQA", "key_identity_openbookqa.json", "written"))
# Each file's own control is read beside its margin, because a bar means
# nothing without it: a control already far above chance leaves a margin no room.
_above_zero, _headroom = [], {}
for _name, _path, _provenance in PROVENANCE:
    _doc = load("results/" + _path)
    _rel = _doc["files"]["released"]
    _family = max(("written form", "roundness"),
                  key=lambda f: _rel[f]["cv_accuracy"] - _doc["clean"][f]["max"])
    _best = 100 * (_rel[_family]["cv_accuracy"] - _doc["clean"][_family]["max"])
    _headroom[_name] = 100 * _doc["clean"][_family]["max"] - 100 * _doc["chance"]
    if _best > 0:
        _above_zero.append((_name, _provenance, round(_best, 1)))
check(len(PROVENANCE) == 10,
      "sec:authorship says ten released benchmarks")
check(sorted(n for n, _, _ in _above_zero) == ["BixBench v1.5", "MMLU-Pro"],
      "fig:provenance says two released files clear their clean control, "
      f"MMLU-Pro and BixBench; got {_above_zero}")
check(round(_headroom["MMLU-Pro"]) == 2 and round(_headroom["BixBench v1.5"]) == 13,
      "fig:provenance types a control 2 points over chance for MMLU-Pro and 13 "
      f"for BixBench; got {_headroom['MMLU-Pro']:.1f} and "
      f"{_headroom['BixBench v1.5']:.1f}")
# The paragraph's whole point: only MMLU-Pro clears by more than its own control
# clears chance. That is the sentence, so it is the check.
check(dict(((n, m) for n, _, m in _above_zero))["MMLU-Pro"] > _headroom["MMLU-Pro"]
      and dict(((n, m) for n, _, m in _above_zero))["BixBench v1.5"]
      < _headroom["BixBench v1.5"],
      "sec:authorship says MMLU-Pro's margin exceeds its control's own distance "
      "above chance and BixBench's does not")

# sec:limits reports the two controls that show why the lower five bars are weak.
_usmle = load("results/key_identity_medqa_usmle.json")
check(abs(100 * _usmle["clean"]["written form"]["max"] - 51.9) < 0.06,
      "sec:limits types a 51.9% control for MedQA-USMLE at n=77")
_obqa = load("results/key_identity_openbookqa.json")
check(abs(100 * _obqa["clean"]["written form"]["max"] - 85.7) < 0.06,
      "sec:limits types an 85.7% control for OpenBookQA at n=7")
for _name, _path in (("MedQA-USMLE", "key_identity_medqa_usmle.json"),
                     ("SciQ", "key_identity_sciq.json"),
                     ("ARC-Challenge", "key_identity_arc_challenge.json"),
                     ("OpenBookQA", "key_identity_openbookqa.json"),
                     ("AQuA-RAT", "key_identity_aqua_rat.json")):
    _released = load("results/" + _path)["files"]["released"]
    check(_released["n_clusters"] == _released["n_items"],
          f"sec:limits says {_name} carries no cluster field, so it is scored "
          "leave-one-item-out")

# --- sec:authorship's controlled arm: the causal claim and its bound --------
# Two ten-option files over the same items, same key, same three human
# distractors; only the six added options differ in who wrote them. The section
# reports both comparisons because they disagree, and both are pinned.
contrast = load("results/authorship_contrast.json")
check(contrast["n_items"] == 633,
      f"sec:authorship says 633 paired items, not {contrast['n_items']}")
check(contrast["families"]["roundness"]["n_clusters"] == 28,
      f"sec:authorship says 28 subjects, not "
      f"{contrast['families']['roundness']['n_clusters']}")
for _family, _points, _low, _high in (("written form", 2.5, -0.2, 6.0),
                                      ("roundness", 1.6, -0.3, 2.8)):
    _block = contrast["families"][_family]
    check(abs(100 * _block["difference"] - _points) < 0.06,
          f"sec:authorship types {_points:+.1f} for {_family}, against "
          f"{100 * _block['difference']:+.2f}")
    check(abs(100 * _block["ci95"][0] - _low) < 0.4
          and abs(100 * _block["ci95"][1] - _high) < 0.4,
          f"sec:authorship types [{_low:+.1f}, {_high:+.1f}] for {_family}, against "
          f"[{100 * _block['ci95'][0]:+.1f}, {100 * _block['ci95'][1]:+.1f}]")
    # The claim is that both move the predicted way and neither is
    # significant once the two families are corrected for. Both halves are
    # pinned, because on data a parser bug had mangled the table said the
    # opposite.
    check(_block["difference"] > 0,
          f"sec:authorship says both families move the predicted way; {_family} "
          f"moves {100 * _block['difference']:+.2f}")
    check(_block["p_two_sided_bonferroni"] > 0.05,
          f"sec:authorship says neither family reaches significance once "
          f"corrected; {_family} is at {_block['p_two_sided_bonferroni']:.3f}")

# The bound is the part that keeps the claim honest: neither arm may clear its
# own clean control, or the paragraph is wrong.
_aug = load("results/key_identity_aug_generated.json")
for _label in ("generated", "borrowed"):
    for _family in ("written form", "roundness"):
        _margin = 100 * (_aug["files"][_label][_family]["cv_accuracy"]
                         - _aug["clean"][_family]["max"])
        check(_margin < 0,
              f"sec:authorship says neither controlled arm clears its clean "
              f"control; {_label}/{_family} is now {_margin:+.1f}")
check(abs(100 * _aug["files"]["generated"]["written form"]["cv_accuracy"] - 10.7) < 0.06
      and abs(100 * _aug["files"]["generated"]["roundness"]["cv_accuracy"] - 7.7) < 0.06
      and abs(100 * _aug["clean"]["written form"]["max"] - 12.6) < 0.06
      and abs(100 * _aug["clean"]["roundness"]["max"] - 11.9) < 0.06,
      "sec:authorship types 10.7% and 7.7% against 12.6% and 11.9% controls")

# --- sec:authorship's mechanism: regeneration alone is not the trigger -------
# The rank repair preserves each replaced option's written style and lands at the
# control; the exchangeable rewriting does not and clears it. Both halves are
# checked, because the section's claim is the contrast and not either number.
for _label, _path, _rank_max, _exch_min in (
        ("MMLU", "results/key_identity_mmlu.json", 0.6, 7.0),
        ("MedMCQA", "results/key_identity_medmcqa.json", 1.0, 20.0),
        ("MMLU-Pro", "results/key_identity_mmlu_pro.json", 0.6, 9.0)):
    _doc = load(_path)
    _worst = {f: 100 * b["max"] for f, b in _doc["clean"].items()}
    _rank = max(100 * _doc["files"]["repaired (rank)"][f]["cv_accuracy"] - _worst[f]
                for f in ("written form", "roundness"))
    _exch = (100 * _doc["files"]["exchangeable offsets"]["roundness"]["cv_accuracy"]
             - _worst["roundness"])
    check(_rank <= _rank_max,
          f"sec:authorship says the rank repair lands at its control on {_label}; "
          f"it is now {_rank:+.1f} points above")
    check(_exch >= _exch_min,
          f"sec:authorship says the exchangeable rewriting clears its control on "
          f"{_label} roundness; it is now {_exch:+.1f}")

generated = load("results/key_identity_mmlu_pro.json")
roundness = generated["files"]["released"]["roundness"]
check(100 * (roundness["cv_accuracy"] - generated["clean"]["roundness"]["max"]) > 5.0,
      "released MMLU-Pro no longer shows the authorship signature sec:authorship "
      "rests on")

# --- app:stats: the MedMCQA file the three new rows are cut from -------------
medmcqa_audit = load("results/mcq_audit_medmcqa.json")
check(medmcqa_audit["before"]["n_items"] == 29878,
      f"MedMCQA is {medmcqa_audit['before']['n_items']} items, not the 29,878 "
      "app:stats states")
check(medmcqa_audit["before"]["numeric_family"]["n_items"] == 454,
      "app:stats states 454 four-option numeric MedMCQA items")
check(medmcqa_audit["before"]["numeric_family"]["n_clusters"] == 21,
      "app:stats states 21 MedMCQA subjects")
check(load("results/key_identity_medmcqa.json")["files"]["released"]["n_items"] == 368,
      "app:stats states 368 MedMCQA items admit an exchangeable rewriting")

# sec:authorship explains the size of the MedMCQA signature by what its keys are,
# so those three statistics are recomputed from the file the table is cut to.
_medmcqa_file = Path("results/repaired/medmcqa_matched_released.jsonl.gz")
_medmcqa_plain = Path("build/medmcqa_matched_released.jsonl")
if _medmcqa_file.exists():
    with gzip.open(_medmcqa_file, "rt", encoding="utf-8") as handle:
        _medmcqa_keys = [json.loads(line)["ideal"] for line in handle if line.strip()]
elif _medmcqa_plain.exists():
    _medmcqa_keys = [json.loads(line)["ideal"] for line
                     in _medmcqa_plain.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
else:
    _medmcqa_keys = None
if _medmcqa_keys is None:
    print("  (skipped the MedMCQA key statistics: the matched file is not here)")
else:
  check(len(_medmcqa_keys) == 368,
      f"the matched MedMCQA file holds {len(_medmcqa_keys)} items, not 368")
  _medmcqa_values = [parse_number(str(key).strip()) for key in _medmcqa_keys]
  check(all(value is not None for value in _medmcqa_values),
        "a MedMCQA key no longer parses as a number")


  def _significant_digits(value):
      digits = re.sub(r"[^0-9]", "", "%g" % abs(float(value))).strip("0")
      return max(len(digits), 1)


  _whole = sum(float(v) == int(float(v)) for v in _medmcqa_values)
  _single = sum(_significant_digits(v) == 1 for v in _medmcqa_values)
  _median = sorted(abs(float(v)) for v in _medmcqa_values)[len(_medmcqa_values) // 2]
  check(round(100 * _whole / 368) == 86,
        f"sec:authorship says 86% of MedMCQA keys are whole numbers, not "
        f"{100 * _whole / 368:.0f}%")
  check(round(100 * _single / 368) == 48,
        f"sec:authorship says 48% carry a single significant digit, not "
        f"{100 * _single / 368:.0f}%")
  check(abs(_median - 25) < 1e-9,
        f"sec:authorship says the median MedMCQA key is 25, not {_median}")


# --- how many rollouts an item, per file --------------------------------------
# Table 1's caption said "two greedy rollouts an item" for both blocks, and the
# appendix said the same; every BixBench arm runs three. Count them in the dumps.
import collections as _collections
import arm_intervals as _ai
_counts = {}
for _path, _label, _, _ in _ai.DUMPS:
    _p = _ai.resolve(_path)
    if _p.exists():
        _per = _collections.Counter(r["item"] for r in _ai.load(_p, "file"))
        _counts.setdefault(_label.split(",")[0].split()[0], set()).update(_per.values())
if _counts:
    check(_counts.get("MMLU-Pro") == {2} and _counts.get("BixBench") == {3},
          f"the paper says two rollouts an item on MMLU-Pro and three on BixBench; the "
          f"dumps hold {_counts}")
    check("numeric items, three letter orderings each" in _flat,
          "app:reader no longer states three letter orderings per BixBench item")

# --- SUBMISSION.md's page statement is the measured one ----------------------
# It said "8 main-text pages, references starting on page 8" for a paper whose
# main text ended on page 9 -- in the file read while filling in the form.
_sub = Path("SUBMISSION.md").read_text(encoding="utf-8") if Path("SUBMISSION.md").exists() else None
_said = _sub and re.search(r"(\d+) main-text pages,\s+references starting on page (\d+)", _sub)
check(_sub is None or _said is not None, "SUBMISSION.md no longer states the main-text page count")
if _said and Path("build/main.pdf").exists():
    import page_budget as _pb
    _last, _refs = _pb.main_text_end("build/main.pdf")
    if _last is not None:
        check((int(_said.group(1)), int(_said.group(2))) == (_last, _refs),
              f"SUBMISSION.md says {_said.group(1)} main-text pages with references on page "
              f"{_said.group(2)}; the built PDF has {_last} and {_refs}")

import bracketing as bk  # noqa: E402
import bixbench_withdata as bw  # noqa: E402

# --- tab:claims and the paper-wide budget ------------------------------------
# The ten findings the abstract and the introduction state, and the two no-data margins Appendix C rests on,
# corrected together (claim_budget.py). Each estimate is re-derived there from the same data its own section reads,
# so each is pinned to that section's source (claim_budget.quoted_95 refuses an estimate its source does not
# state), and every row of the table to the budget. Eight survive; the two that do not are said not to where the
# text states them.
import claim_budget as _cbm  # noqa: E402
_cb_path = Path("results/claim_budget.json")
_cb = json.loads(_cb_path.read_text()) if _cb_path.exists() else {"claims": [], "k": 0, "survivors": [],
                                                                   "target_coverage": 0.0}
_CB_FAMILY = ["rank rule", "published, gpt-4o", "published, claude", "not the rank", "forced over tolerance",
              "key over other", "corrected below tolerance", "current agents", "bracketing",
              "readers of the published runs"]
check(sorted(c["claim"] for c in _cb["claims"]) == sorted(_CB_FAMILY) and _cb["k"] == 10
      and abs(_cb["target_coverage"] - (1 - 0.05 / 10)) < 1e-12,
      f"results/claim_budget.json holds {sorted(c['claim'] for c in _cb['claims'])}; run claim_budget.py")
_cbc = {c["claim"]: c for c in _cb["claims"]}
if _cbc:
    _cl_lo = _cbc["published, claude"]["parts"][0]["ci_family"][0]
    _cur_parts = {p["part"]: p for p in _cbc["current agents"]["parts"]}
    _luna_lo = _cur_parts["gpt-6-luna"]["ci_family"][0]
    _fails = sorted(c["claim"] for c in _cb["claims"] if not c["survives"])
    check(_fails == ["current agents", "published, claude"]
          and sorted(_cb["survivors"]) == sorted(set(_CB_FAMILY) - set(_fails))
          and all(c["survives"] == all(p["clears_family"] for p in c["parts"]) for c in _cb["claims"])
          and [p for p, x in _cur_parts.items() if not x["clears_family"]] == ["gpt-6-luna"]
          and -0.5 < _cl_lo <= 0 and _luna_lo <= 0
          and "Ten headline findings are treated as one family (Table~\\ref{tab:claims})" in _flat
          and "Each is read at the level that attains $1-0.05/10$ coverage on its own file" in _flat
          and "Eight of the ten survive; at the nominal $95\\%$ all ten hold, so calibration only widens the intervals"
              in _flat
          and f"Claude 3.5 Sonnet's forced-choice margin without the data reaches ${_cl_lo:.1f}$ at the family level, "
              f"and gpt-6-luna's forced-choice excess over the tolerance ${_luna_lo:.1f}$; the paper says so where it "
              f"states them" in _flat
          and "and every part does except gpt-6-luna's" in _flat
          and f"although gpt-6-luna's interval reaches ${_luna_lo:.1f}$ when the paper's ten headline findings are "
              f"corrected jointly" in _flat,
          f"app:stats or the introduction misstate which of the ten claims survive; the budget says {_cb['survivors']}")
    # every claim clears at the nominal 95%, which app:stats and the caption say
    _nominal_ok = all((p["ci95_nominal"][1] < p["rank_reader_difference"]) if "rank_reader_difference" in p
                      else p["ci95_nominal"][0] > 0 for c in _cb["claims"] for p in c["parts"])
    check(_nominal_ok and "At the nominal $95\\%$ every finding holds." in _flat,
          "a headline claim no longer clears at the nominal 95%, which app:stats and tab:claims' caption say they all do")
    # how each claim is calibrated, as app:stats says
    _parametric = {"rank rule", "published, gpt-4o", "published, claude", "not the rank"}
    check(all(p["calibration"] == ("parametric" if c["claim"] in _parametric else "double bootstrap")
              for c in _cb["claims"] for p in c["parts"])
          and "The rank rule, the two margins and the rank claim are calibrated by the simulation above with the "
              "target changed" in _flat,
          "app:stats misstates which claims the simulation calibrates and which the double bootstrap")
    # the 95% column: each part's interval as the paper states it elsewhere, from the report that states it
    for _c in _cb["claims"]:
        for _p in _c["parts"]:
            try:
                _q, _qsrc = _cbm.quoted_95(_c, _p)
            except SystemExit as _e:
                check(False, f"claim_budget's estimate is not its section's: {_e}")
                continue
            check(_p["ci_quoted_95_source"] == _qsrc and all(abs(a - b) < 1e-9 for a, b in zip(_q, _p["ci_quoted_95"])),
                  f"tab:claims' 95% interval for {_c['claim']}|{_p['part']} is not {_qsrc}'s; run claim_budget.py "
                  f"--attach-quoted")
    # the table
    _claims_tab = " ".join(_table_body("tab:claims").split())
    _cb_rows = _cbm.table_rows(_cb)
    check(len(_cb_rows) == 17 and _claims_tab.count("\\\\") == len(_cb_rows) + 1,
          f"tab:claims should hold the budget's {len(_cb_rows)} rows and no others")
    for _row in _cb_rows:
        check(" ".join(_row.split()) in _claims_tab, f"tab:claims should carry the row {_row}")
    check("\\caption{The paper's ten headline findings, corrected jointly" in _tex
          and "the current agents' excess for all three agents, and the graders' gain on the published runs for all "
              "three graders" in _flat,
          "tab:claims' caption misnames the family or its conjunctions")
    # sec:open's bound on the rank's share of the published margins: the upper ends of the corrected intervals over
    # what a reader drawing its whole margin from the rank would show
    _nrp = {p["part"]: p for p in _cbc["not the rank"]["parts"]}
    _shr_rank = lambda p: f"{100 * p['ci_family'][1] / p['rank_reader_difference']:.0f}"
    check(f"a preference for the second-smallest option could explain at most "
          f"${_shr_rank(_nrp['gpt-4o'])}\\%$ of gpt-4o's numeric margin and "
          f"${_shr_rank(_nrp['claude-3-5-sonnet-latest'])}\\%$ of Claude's." in _flat,
          "app:arms' bound on the rank's share of the published margins is stale")
# shared with later sections
_brk = load("results/bracketing.json")
_prv = load("results/published_repair.json")["releases"]["v1.5"]
_ra = load("results/rank_attribution.json")["published"]
_f1 = lambda v: f"{v:+.1f}"

# --- the estimator simulation: no longer quoted in the paper, still checked --
_sim_path = Path("results/estimator_simulation.json")
if _sim_path.exists():
    _sim = json.loads(_sim_path.read_text())["settings"]
    _by = {(row["benchmark"], row["lambda"]): row for row in _sim}
    for _name, _n in (("BixBench v1.5", 105), ("MMLU", 661), ("MMLU-Pro", 1263)):
        _null = _by[(_name, 0.0)]
        check(_null["n"] == _n, f"{_name}'s simulation ran at n={_null['n']}")
        check(_null["bias"] > 0, f"lambda-hat is no longer biased upward at zero for {_name}")
        check(0.02 < _null["reject_rate"] < 0.08,
              f"the lambda test is not correctly sized: {_name} rejects on "
              f"{_null['reject_rate']:.3f} of null draws")
    _away = [abs(row["bias"]) for row in _sim if row["lambda"] > 0]
    check(max(_away) < 0.007,
          f"lambda-hat is no longer within 0.007 of truth away from the boundary; the "
          f"worst simulated bias there is {max(_away):.4f}")

# --- the causal contrast's multiplicity correction --------------------------
_contrast = json.loads(Path("results/authorship_contrast.json").read_text())
for _family, _stated_fw in (("written form", 0.18), ("roundness", 0.19)):
    _block = _contrast["families"][_family]
    check(abs(_block["p_two_sided_bonferroni"] - _stated_fw) < 0.01,
          f"the {_family} contrast's family-wise p was {_stated_fw}, now "
          f"{_block['p_two_sided_bonferroni']:.3f}")
    check(_block["p_two_sided_bonferroni"] > 0.05,
          f"the {_family} contrast now reaches significance once corrected: "
          f"{_block['p_two_sided_bonferroni']:.3f}")
    check(_block["difference"] > 0,
          f"the {_family} contrast no longer moves the predicted way: "
          f"{100 * _block['difference']:+.1f}")
check(_contrast["n_items"] == 633 and
      _contrast["families"]["roundness"]["n_clusters"] == 28,
      f"the contrast held 633 items in 28 subjects; it now holds "
      f"{_contrast['n_items']} in "
      f"{_contrast['families']['roundness']['n_clusters']}")

# Neither causal arm may clear its own clean control.
_aug = Path("results/key_identity_aug_generated.json")
if _aug.exists():
    _augd = json.loads(_aug.read_text())
    for _arm in ("generated", "borrowed"):
        for _family in ("written form", "roundness"):
            _margin_ = 100 * (_augd["files"][_arm][_family]["cv_accuracy"]
                              - _augd["clean"][_family]["max"])
            check(_margin_ < 0,
                  f"a causal arm now clears its own clean control: "
                  f"{_arm}/{_family} is {_margin_:+.1f}")

# --- AQuA-RAT is the flattest key rank, not the only flat one ---------------
_survey_path = Path("results/channel_survey.json")
if _survey_path.exists():
    _sv = json.loads(_survey_path.read_text())
    _assessed = [b for b in _sv["benchmarks"]
                 if b.get("chi_square_vs_uniform") and b.get("n_numeric_items", 0) >= 20]
    _uniformish = [b for b in _assessed if b["chi_square_vs_uniform"]["p_value"] >= 0.05]
    check(len(_uniformish) == 2,
          f"sec:theory says two of the assessed files do not reject a uniform key "
          f"rank; {len(_uniformish)} do not: {[b['benchmark'] for b in _uniformish]}")
    _best = min(_uniformish, key=lambda b: b["chi_square_vs_uniform"]["chi2"])
    check(_best["benchmark"] == "AQuA-RAT",
          f"sec:theory says AQuA-RAT is the flattest; it is {_best['benchmark']}")
    _other = next(b for b in _uniformish if b is not _best)
    check(f"is flat ($p={_best['cluster_robust_vs_uniform']['p_value']:.2f}$, "
          f"$n={_best['n_numeric_items']}$)" in _flat,
          f"sec:theory's AQuA-RAT statistics are p={_best['cluster_robust_vs_uniform']['p_value']:.2f} over "
          f"{_best['n_numeric_items']} items")
    # the survey's verdicts: clustered and held out, as Figure 1c fills them and sec:theory counts them
    _robust = [b for b in _assessed if (b["cluster_robust_vs_uniform"].get("p_value") or 1.0) < 0.05]
    _iid = [b for b in _assessed if b["chi_square_vs_uniform"]["p_value"] < 0.05]
    _held = sorted((b for b in _robust if b["cross_validated"]["credit"] > 0),
                   key=lambda b: -b["cross_validated"]["credit"])
    _num = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    _short = lambda name: name.replace("LAB-Bench ", "")
    # a file with no recorded clusters takes each item as its own, and the sentence says so
    _hname = lambda b: (f"{_short(b['benchmark'])} (${100 * b['cross_validated']['credit']:+.1f}$"
                        + (f"; ${b['n_numeric_items']}$ items, unclustered)" if b.get("n_clusters") is None else ")"))
    _held_txt = ", ".join(_hname(b) for b in _held[:-1]) + f" and {_hname(_held[-1])}"
    check(len(_assessed) == 9 and _sv["summary"]["n_rejecting_uniformity_cluster_robust"] == len(_robust)
          and f"Of nine released files with enough numeric items to assess "
              f"\\citep{{labbench,aquarat,medmcqa,medqa,mmlu,mmlupro,sciq}} (Table~\\ref{{tab:survey}}), "
              f"{_num[len(_robust)]} reject a uniform key rank with items clustered by source "
              f"({_num[len(_iid)]} if items are treated as independent), and a rank chosen on held-out folds of each "
              f"file's clusters stays above chance on {_num[len(_held)]}: {_held_txt}." in _flat
          and len(_held) == 4,
          f"sec:theory's survey counts are stale: {len(_robust)} clustered, {len(_iid)} independent, held out "
          f"{[b['benchmark'] for b in _held]}")
    _sq = next(b for b in _assessed if b["benchmark"] == "LAB-Bench SeqQA")
    check(f"SeqQA's ${100 * _sq['plug_in_credit']:+.1f}$ in-sample falls to "
          f"${100 * _sq['cross_validated']['credit']:+.1f}$ with its four subtasks held out" in _flat
          and _sq["cluster_robust_vs_uniform"]["n_clusters"] == 4,
          "sec:theory's SeqQA held-out sentence is stale")
    # tab:survey is channel_survey.py's rows, and Figure 1c's fill is the same verdict
    import channel_survey as _cs
    _svbody = _table_body("tab:survey")
    for _row in _cs.table_rows(_sv):
        check(_row in _svbody, f"tab:survey should carry the row {_row}")
    _mf = Path("make_figures.py").read_text()
    check('cost = load("results/proximity_cost.json")' in _mf and 'ax.plot(g["lambda"], g["share"]' in _mf,
          "Figure 1c no longer plots each grading's share of the rule's cost against its proximity weight")
    # the key-rank test on v1.5 with items clustered by capsule
    _b15s = next(b for b in _assessed if b["benchmark"] == "BixBench v1.5")
    _p = _b15s["cluster_robust_vs_uniform"]["p_value"]
    _e = int(f"{_p:.0e}".split("e")[1])
    _m = round(_p / 10 ** _e)
    check(f"far from uniform with items clustered by capsule\n($p={_m}\\times10^{{{_e}}}$" in _tex
          or f"far from uniform with items clustered by capsule ($p={_m}\\times10^{{{_e}}}$" in _flat,
          f"sec:theory's clustered test on v1.5 should read p={_m}x10^{_e}")

# --- the published runs: no longer fitted in the paper, still checked -------
_published = Path("results/published_baselines.json")
if _published.exists():
    _runs = {("gpt-4o" if "gpt-4o" in r["run"] else "claude"): r
             for r in json.loads(_published.read_text())["runs"]}
    check(all(r["n_items"] == 159 for r in _runs.values()),
          "the published runs no longer cover v1.0's 159 numeric items")
    check(min(r["lrt_lambda_zero_p"] for r in _runs.values()) >= 0.05,
          "a published baseline now rejects lambda=0")

# --- the temporal test: no longer in the paper, its result still checked ----
# Four files, and the result does not replicate in sign: the no-data arm's date
# correlation clears zero on two of them, in opposite directions.
_LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
_temporal_files = {
    "litqa2": "LitQA2", "tableqa": "TableQA", "figqa": "FigQA", "suppqa": "SuppQA",
}
_temporal = {}
for _stem, _short in _temporal_files.items():
    _p = Path(f"results/temporal_test_{_stem}.json")
    if _p.exists():
        _temporal[_short] = json.loads(_p.read_text())

if "LitQA2" in _temporal:
    check(len(_temporal) == 4,
          f"the temporal test covers four files; {sorted(_temporal)} are on disk")
    _clearing = {}
    for _short, _t in _temporal.items():
        _w = _t["models"][_LLAMA]["withheld"]
        _lo, _hi = _w["spearman_ci95"]
        if _lo > 0 or _hi < 0:
            _clearing[_short] = _w["spearman_date_vs_margin"]
    check(len(_clearing) == 2,
          f"the no-data date correlation cleared zero on two of the four files; it "
          f"now clears on {sorted(_clearing)}")
    check(min(_clearing.values()) < 0 < max(_clearing.values()),
          f"the two clearing date correlations no longer have opposite signs: {_clearing}")

_option_reader = Path("results/nonlinear_mmlu_chars_option.json")
if _option_reader.exists():
    _nl = json.loads(_option_reader.read_text())
    _worst = _nl["clean"]["max"]
    _block = _nl["files"]["rank uniform"]
    _m = 100 * (_block["cv_accuracy"] - _worst)
    _lo, _hi = (100 * v for v in _block["margin_ci95"])
    # no longer quoted in the paper; the result file is still checked for its sign
    check(_lo > 0, f"the per-option character reader no longer clears on the rank repair: "
                   f"{_m:+.1f} [{_lo:+.1f},{_hi:+.1f}]")

# --- the authorship contrast with no designed features: data only ---------
_pro_option = Path("results/nonlinear_mmlupro_chars_option.json")
_mmlu_option = Path("results/nonlinear_mmlu_chars_option.json")
if _pro_option.exists() and _mmlu_option.exists():
    def _released(path):
        _nl = json.loads(Path(path).read_text())
        _block = _nl["files"]["released"]
        return (100 * (_block["cv_accuracy"] - _nl["clean"]["max"]),
                100 * _block["margin_ci95"][0], 100 * _block["margin_ci95"][1])

    _gen = _released(_pro_option)
    _hand = _released(_mmlu_option)
    check(_gen[1] > 0 >= _hand[1],
          f"the character reader no longer clears on the generated file and not on the "
          f"hand-written one; their interval floors are {_gen[1]:+.1f} and {_hand[1]:+.1f}")

# --- a rank guarantee does not survive subsetting ---------------------------
_subset = Path("results/subset_guarantee.json")
if _subset.exists():
    _sub = json.loads(_subset.read_text())
    _whole = [_sub[b]["files"]["rank uniform, whole"]["plug_in_credit"]
              for b in ("MMLU", "MedMCQA", "MMLU-Pro")]
    _cut = [_sub[b]["files"]["rank uniform, feasible"]["plug_in_credit"]
            for b in ("MMLU", "MedMCQA", "MMLU-Pro")]
    # the subsetting result is no longer quoted in the paper; its direction is still checked
    check(all(c > w for c, w in zip(_cut, _whole)),
          f"the feasible subsets no longer carry more plug-in credit than the whole files: "
          f"{_cut} against {_whole}")

# --- Lemma 1's counterexample and the distinctness trap, as enumerated -----
_theory = Path("results/writer_theory.json")
if _theory.exists():
    _th = json.loads(_theory.read_text())
    check(abs(_th["sorted_distractors"]["gamma"]) < 1e-9
          and not _th["sorted_distractors"]["tuple_exchangeable"],
          "app:theory's counterexample no longer separates Gamma=0 from "
          "exchangeability of the tuple")
    _traps = {t["key_marginal"]: t for t in _th["distinctness_trap"]}
    for _name, _t in _traps.items():
        check(abs(_t["gamma_symmetric"]) < 1e-9,
              f"tab:distinctness: the symmetric draw leaks "
              f"{100 * _t['gamma_symmetric']:+.1f} on the {_name} marginal")
        _line = next((ln for ln in _tex.splitlines()
                      if ln.startswith(_name + " &")), None)
        check(_line is not None, f"tab:distinctness has no row for {_name!r}")
        check(f"${100 * _t['gamma_reject_against_key']:+.1f}$" in _line,
              f"tab:distinctness {_name!r}: redrawing against the key gives "
              f"{100 * _t['gamma_reject_against_key']:+.1f}")

    _worst = max(t["gamma_reject_against_key"] for t in _th["distinctness_trap"])
    check(f"heavy skew & $+0.0$ & ${100 * _worst:+.1f}$" in _table_body("tab:distinctness"),
          f"tab:distinctness should give the trap at {100 * _worst:+.1f} points")

# --- freeing recall per key rank: no longer in the paper, still checked -----
_refit = Path("results/rank_dependent_fit.json")
if _refit.exists():
    _rf = json.loads(_refit.read_text())
    check(_rf["n_rejecting_shared"] == 12 and _rf["n_draws"] == 40,
          f"12 of 40 MMLU draws rejected the shared fit; the refit sees "
          f"{_rf['n_rejecting_shared']} of {_rf['n_draws']}")
    check(_rf["sign_changes"] == 0,
          f"freeing recall per rank now changes the geometry term's sign "
          f"{_rf['sign_changes']} times")
    _bix = Path("results/rank_dependent_fit_bix.json")
    if _bix.exists():
        _bx = json.loads(_bix.read_text())
        check(_bx["sign_changes"] == 1,
              f"the BixBench refit changed the term's sign once; it now changes it "
              f"{_bx['sign_changes']} times")

# --- the readers on the science-agent files' option text: data only ---------
_text = Path("results/text_options.json")
if _text.exists():
    _tx = json.loads(_text.read_text())["set"]
    _scored = {k: v for k, v in _tx.items() if "skipped" not in v}
    check(len(_scored) == 7, f"seven files were scored on option text; {len(_scored)} are")
    check(all(v["margin_ci95"][0] <= 0 for v in _scored.values()),
          "a text-option file now clears its control")
    check(sorted(_scored, key=lambda k: -_scored[k]["margin_over_clean"])[:2]
          == ["LAB-Bench ProtocolQA", "LAB-Bench LitQA2"],
          "ProtocolQA and LitQA2 are no longer the two largest text-option margins")
    _naive = Path("results/text_options_naive.json")
    if _naive.exists():
        _nv = json.loads(_naive.read_text())["set|groups ignored"]
        for _full in ("LAB-Bench SeqQA", "LAB-Bench DbQA"):
            check(_full in _nv and _nv[_full]["margin_ci95"][0] > 0,
                  f"random folds no longer clear on {_full}, so the fold trap does not bite")

# --- the no-data baseline run as an agent: data only ------------------------
_agentic = sorted(Path("results").glob("agentic_bixbench_*.json"))
if _agentic:
    _ag = {}
    for _f in _agentic:
        _ag.update(json.loads(_f.read_text()))
    _aware = {k.split("|")[0]: v for k, v in _ag.items() if k.endswith("|withheld_aware")}
    if len(_aware) >= 3:
        check(max(v["file"]["unparsed"] for v in _aware.values()) < 5.0,
              "the agentic arm is readable only while compliance is high; worst unparsed is "
              f"{max(v['file']['unparsed'] for v in _aware.values()):.1f}%")
        _clears = [m for m, v in _aware.items() if v["file"]["margin_ci95"][0] > 0]
        check(len(_clears) <= 1,
              f"at most one agent cleared its control, and {len(_clears)} do: {_clears}")

_mech = Path("results/agentic_mechanism.json")
if _mech.exists():
    _all_mech = json.loads(_mech.read_text())
    # the released arm, not the placebo one the same file also holds
    _m = next(v for k, v in _all_mech.items() if "withheld_aware" in k)
    _f, _c = _m["file"], _m["clean"]
    check(_f["rank_fit"]["geometry_ci95"][0] > 0 >= _c["rank_fit"]["geometry_ci95"][0],
          "the agent's rank term no longer clears on the released file alone; "
          f"they are {_f['rank_fit']['geometry_ci95']} and {_c['rank_fit']['geometry_ci95']}")
    check(_f["rank_fit"]["observed_margin_over_chance"] - _f["rank_fit"]["geometry"] > 2.0,
          "the 14B agent no longer takes a surplus over its own rank term: "
          f"{_f['rank_fit']['observed_margin_over_chance']:+.1f} against "
          f"{_f['rank_fit']['geometry']:+.1f}")
    _big = next((v for k, v in _all_mech.items() if "qwen32b" in k), None)
    if _big:
        _bf = _big["file"]["rank_fit"]
        check(abs(_bf["observed_margin_over_chance"] - _bf["geometry"]) < 2.0,
              "the 32B agent now takes a surplus over its rank term: "
              f"{_bf['observed_margin_over_chance']:+.1f} against {_bf['geometry']:+.1f}")
    check(_f["says_middle"]["accuracy"] - _c["says_middle"]["accuracy"]
          > _f["does_not"]["accuracy"] - _c["does_not"]["accuracy"],
          "the middle-reaching wording is no longer where the agent's margin lives")

_arms = {}
for _name in ("placebo", "repaired"):
    _f = Path(f"results/agentic_bixarm_{_name}_qwen14b.json")
    if _f.exists():
        _arms[_name] = next(iter(json.loads(_f.read_text()).values()))
if len(_arms) == 2:
    _rel = json.loads(Path("results/agentic_bixbench_qwen14b.json").read_text())
    _rel = _rel["Qwen/Qwen2.5-14B-Instruct|withheld_aware"]["margin_over_clean"]
    check(_rel > _arms["placebo"]["margin_over_clean"]
          > _arms["repaired"]["margin_over_clean"],
          "the agentic arms no longer order released > placebo > repaired: "
          f"{_rel:.1f}, {_arms['placebo']['margin_over_clean']:.1f}, "
          f"{_arms['repaired']['margin_over_clean']:.1f}")
    check(all(_v["file"]["margin_ci95"][0] <= 0 for _v in _arms.values()),
          "one of the two rewritten agentic arms now clears")

# the replication is only a replication if it agrees with the run it repeats
_dump_result = Path("results/agentic_bixbench_qwen14b_dump.json")
_main = Path("results/agentic_bixbench_qwen14b.json")
if _dump_result.exists() and _main.exists():
    _d = next(iter(json.loads(_dump_result.read_text()).values()))
    _o = json.loads(_main.read_text())["Qwen/Qwen2.5-14B-Instruct|withheld_aware"]
    check(abs(_o["margin_over_clean"] - _d["margin_over_clean"]) < 1e-9,
          f"the two 14B runs disagree: {_o['margin_over_clean']:.3f} "
          f"against {_d['margin_over_clean']:.3f}")

_pro = sorted(Path("results").glob("agentic_mmlupro_*.json"))
if _pro:
    _pg = {}
    for _f in _pro:
        _pg.update(json.loads(_f.read_text()))
    _pa = {k.split("|")[0]: v for k, v in _pg.items() if k.endswith("|withheld_aware")}
    _pclears = {k: v for k, v in _pa.items() if v["file"]["margin_ci95"][0] > 0}
    check(sorted(_pclears) == ["Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct"],
          f"the two largest agents cleared on MMLU-Pro; the ones that do now are "
          f"{sorted(_pclears)}")


# --- figures referenced by the manuscript exist -----------------------------
for figure in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex):
    check(Path(figure).exists(), f"missing figure: {figure}")

# --- greedy is deterministic given the batching, and not otherwise -----------
_rep = Path("results/agentic_replication_mmlupro_qwen14b.json")
_orig = Path("results/agentic_mmlupro_qwen14b.json")
if _rep.exists() and _orig.exists():
    _r = next(iter(json.loads(_rep.read_text()).values()))
    _o = next(iter(json.loads(_orig.read_text()).values()))
    check(_r["model"] == _o["model"] and _r["items"] == _o["items"]
          and _r["condition"] == _o["condition"],
          "the replication no longer re-runs the same arm")
    _gap = abs(_r["margin_over_clean"] - _o["margin_over_clean"])
    check(0.0 < _gap < 1.5,
          "a re-run at another batch size should move the margin by under a point and a "
          f"half without reproducing it; the gap is {_gap:.2f}")

# --- whose run the template cells are ----------------------------------------
# Every template cell is our run of BixBench's template on models the benchmark
# never used. Its published baselines are v1.0 runs, and the paper must not
# describe them as collecting (README and VERIFICATION.md say this check exists).
check("already collecting" not in _flat,
      "the paper says BixBench's published baseline is 'already collecting'; the "
      "template cells are our runs of its template, not its published baselines")

# --- the instrument cells: no longer tabulated, still checked ---------------
_icp = Path("results/instrument_cells.json")
check(_icp.exists(), "results/instrument_cells.json is missing")
_ic = json.loads(_icp.read_text())
_irows = {}
for _cell in _ic["cells"]:
    _irows.setdefault((_cell["file"], _cell["framing"]), {})[_cell["readout"]] = _cell
check(len(_irows) == 5 and all(len(r) == 2 for r in _irows.values()),
      f"the instrument cells should hold five framings by two read-outs; there are {len(_irows)}")
_it, _ich = _irows[("mmlupro", "BixBench's template")], _irows[("mmlupro", "chat framing")]
check(all(_it[ro]["interval"][0] < 0 < _ich[ro]["interval"][0] for ro in ("letter", "generated")),
      "on MMLU-Pro the template should not clear and the chat framing should")
_read_gap = max(abs(r["letter"]["over_chance"] - r["generated"]["over_chance"]) for r in _irows.values())
_prompt_gap = max(_ich[ro]["over_chance"] - _it[ro]["over_chance"] for ro in ("letter", "generated"))
check(_read_gap < 2.5 and _prompt_gap > 8,
      f"the read-out should move a cell far less than the prompt does; they move it "
      f"{_read_gap:.1f} and {_prompt_gap:.1f}")

# --- the rank term is the reader's preference, and the prompt sets it --------
_mech = Path("results/agentic_mechanism_arms.json")
if _mech.exists():
    _ma = json.loads(_mech.read_text())
    # Key on the model as well as the cell: two families run the same cells.
    _by = {k.split("/")[-1].replace("_mmlupro", "").replace(".jsonl", ""): v
           for k, v in _ma.items()}
    _by = {k.split("_", 1)[1]: v for k, v in _by.items() if k.startswith("qwen14b_")}
    if "bixprompt_argmax" in _by and "neutral_notools" in _by:
        _bx = _by["bixprompt_argmax"]["file"]["rank_fit"]["geometry_ci95"]
        _ag = _by["neutral_notools"]["file"]["rank_fit"]["geometry_ci95"]
        check(_bx[0] < 0 < _ag[0],
              "the rank term should clear under the agent framing and not under "
              f"BixBench's template: {_bx} and {_ag}")
    if "placebo_neutral_notools" in _by:
        _pl = _by["placebo_neutral_notools"]["file"]["rank_fit"]
        check(_pl["geometry_ci95"][0] < 0 < _pl["geometry_ci95"][1],
              "the placebo no longer flattens the rank term: "
              f"{_pl['geometry']:+.2f} {_pl['geometry_ci95']}")
        _plk = _pl["key_rank"]
        _rel = _by["neutral_notools"]["file"]["rank_fit"]["key_rank"]
        check(max(abs(a - b) for a, b in zip(_plk, _rel)) < 0.5,
              "the placebo is supposed to hold the key-rank law and does not: "
              f"{[round(a - b, 2) for a, b in zip(_plk, _rel)]}")

_bixall = Path("results/agentic_bixall_qwen14b.json")
if _bixall.exists():
    _ba = next(iter(json.loads(_bixall.read_text()).values()))
    _flo = _ba["file"]["margin_ci95"][0]
    check(_ba["file"]["n_clusters"] == 59 and _flo > 0,
          f"the steered arm should clear on the whole released file's 59 capsules: {_flo:+.2f}")

# --- the two releases do not share a key-rank law ---------------------------
_drift = Path("results/release_drift.json")
if _drift.exists():
    _dr = json.loads(_drift.read_text())
    check(f"On v1.0's ${_dr['v1_0']['n_items']}$ numeric items the best rule's in-sample" in _flat
          and f"BixBench v1.5's ${_dr['v1_5']['n_items']}$ four-option numeric items" in _flat,
          f"sec:theory should give the two releases' numeric counts, {_dr['v1_0']['n_items']} and "
          f"{_dr['v1_5']['n_items']}")
    # the direction is the whole point: the shipped release is the wider one
    check(_dr["v1_5"]["rank_ceiling_points"] > 4 * _dr["v1_0"]["rank_ceiling_points"],
          "v1.5's channel should be several times v1.0's, and it is not: "
          f"{_dr['v1_5']['rank_ceiling_points']:+.1f} against "
          f"{_dr['v1_0']['rank_ceiling_points']:+.1f}")

# --- the cluster bootstrap's own calibration --------------------------------
_cal = Path("results/bootstrap_calibration.json")
if _cal.exists():
    _cb = json.loads(_cal.read_text())
    _shape = {e["arm"].split(",")[0]: e for e in _cb["arms"]}
    check(set(_shape) == {"BixBench", "MMLU-Pro"},
          f"app:stats' calibration should cover both agent shapes, not {sorted(_shape)}")
    for _name, _entry in _shape.items():
        _obs = next(c for c in _entry["cells"] if c["as_observed"])
        # the whole row, not the numbers loose: a bare "$10.7$" collides with a
        # fitted accuracy three tables away, and a pin that cannot bite is not one
        _row = (f"${_entry['observed_clusters']}$ & ${_entry['effective_clusters_file']:.1f}$ & "
                f"${_obs['coverage_margin']:.3f}$ & ${_obs['coverage_difference']:.3f}$ & "
                f"${_obs['level_for_95_margin']:.3f}$ & ${_obs['level_for_95_difference']:.3f}$")
        check(_row in _flat,
              f"tab:bootcal's {_name} row should read {_row}")
        # and the prose that names the same count, which the row pin does not reach
        check(f"worth ${_entry['effective_clusters_file']:.1f}$" in _flat,
              f"tab:bootcal's caption should call {_name}'s clusters worth "
              f"{_entry['effective_clusters_file']:.1f}")
        for _n in (20, 120):
            _cell = next(c for c in _entry["cells"] if c["n_clusters"] == _n
                         and not c["as_observed"])
            check(f"${_n}$ & -- & ${_cell['coverage_margin']:.3f}$ & "
                  f"${_cell['coverage_difference']:.3f}$" in _flat,
                  f"tab:bootcal's {_name} row at {_n} clusters should read "
                  f"{_cell['coverage_margin']:.3f} and {_cell['coverage_difference']:.3f}")
    # the finding the appendix rests on: MMLU-Pro undercovers and BixBench does not
    _bix = next(c for c in _shape["BixBench"]["cells"] if c["as_observed"])
    _pro = next(c for c in _shape["MMLU-Pro"]["cells"] if c["as_observed"])
    check(_pro["coverage_difference"] < 0.92 < _bix["coverage_difference"],
          "app:stats says BixBench's bootstrap covers and MMLU-Pro's does not: "
          f"{_bix['coverage_difference']:.3f} and {_pro['coverage_difference']:.3f}")
    check(_shape["MMLU-Pro"]["effective_clusters_file"] < 0.25 * 60,
          "tab:bootcal's claim that MMLU-Pro's clusters are grossly unequal no longer "
          f"holds: {_shape['MMLU-Pro']['effective_clusters_file']:.1f} of 60")

_ai = Path("results/arm_intervals.json")
if _ai.exists():
    _arms_cal = {a["arm"]: a for a in json.loads(_ai.read_text())["arms"]}
    # every arm must reproduce its nominal interval, which arm_intervals.py asserts itself
    check(all(a["calibrated_level"] >= 0.95 for a in _arms_cal.values()),
          "arm_intervals.py reports a covering level below nominal")

# --- the controlled probe's extreme reading: no longer in the paper ----------
_re = Path("results/rank_effect_intervals.json")
if _re.exists():
    _bixp = json.loads(_re.read_text())["panels"].get("BixBench v1.5")
    if _bixp:
        check("Qwen2.5-32B" in _bixp["models"][0]["model"],
              f"the extreme rank effect is now {_bixp['models'][0]['model']}'s")
        check(_bixp["n_clearing"] == 3,
              f"three of eleven rank effects cleared on BixBench; {_bixp['n_clearing']} do")
        check(all(r["rank_effect"] < 0 for r in _bixp["models"] if r["clears"]),
              "a clearing rank effect is no longer negative")


# --- the grids: thirteen models, five families, read over chance and paired ---
# tab:grid and tab:gridbix are printed by `prompt_contrasts.py --latex`, so a row
# can only disagree with its result file if someone edited one of them by hand.
# The counts the prose states are asserted as relations on the same files.
import prompt_contrasts as _pcm
_pcp = Path("results/prompt_contrasts.json")
check(_pcp.exists(), "results/prompt_contrasts.json is missing; tab:grid has nothing behind it")
_pc = json.loads(_pcp.read_text())
for _name, _table in (("bixall", "tab:gridbix"),):
    _body = re.sub(r"\s+", " ", _table_body(_table))
    for _line in _pcm.grid_rows(_pc, _name):
        check(_line in _body, f"{_table}'s row should read: {_line}")


def _grid_counts(name):
    rows = _pc["files"][name]["rows"]
    t_clear = [r for r in rows if r["template"]["interval"][0] > 0]
    t_below = [r for r in rows if r["template"]["interval"][1] < 0]
    readable = [r for r in rows if r["framing"]["readable"]]
    f_clear = [r for r in readable if r["framing"]["interval"][0] > 0]
    d_clear = [r for r in readable if r["framing_minus_template"]["interval"][0] > 0]
    return rows, t_clear, t_below, readable, f_clear, d_clear


_pool = _pc["files"]["bixall"]["pooled_framing_minus_template"]
_ptxt = f"${_pool['mean']:+.1f}$ $[{_pool['ci95'][0]:+.1f},{_pool['ci95'][1]:+.1f}]$"
_capt = _flat[_flat.index("\\label{tab:gridbix}") - 1500:_flat.index("\\label{tab:gridbix}")]
check(_ptxt in _capt and f"($\\tau={_pool['tau']:.1f}$)" in _capt,
      f"tab:gridbix's caption should pool BixBench at {_ptxt}, tau {_pool['tau']:.1f}")
_rows_b, _tcb, _tbb, _rdb, _fcb, _dcb = _grid_counts("bixall")
check(len(_rows_b) == 13 and len(_tcb) == 6
      and "the template exceeds chance for six of the thirteen without the question" in _flat,
      f"the template clears on {len(_tcb)} of 13 on BixBench")
check(len(_rdb) == 9 and "over the nine models that answer under both prompts" in _flat,
      f"tab:gridbix's caption pools the nine models that answer both; {len(_rdb)} do")
check(all(r["template"]["unparsed"]["file"] == 0 and r["template"]["unparsed"]["clean"] == 0 for r in _rows_b)
      and "Every model answers under the template" in _flat,
      "tab:gridbix's caption says every model answers BixBench's template cell")

# the v1.0 grid: no longer tabulated, its counts still checked
_g10 = json.loads(Path("results/bixbench_v10_grid.json").read_text())
_c10 = {p: [r for r in _g10["rows"] if r[p]["chance_ci_level"][0] > 0] for p in ("all", "numeric", "other")}
check((len(_c10["all"]), len(_c10["numeric"]), len(_c10["other"])) == (11, 2, 10),
      f"the v1.0 grid counts are {[len(v) for v in _c10.values()]}, not 11, 2 and 10")
check(abs(_g10["level"] - 0.96) < 1e-9, "the v1.0 grid's level is not 96.0%")
_sc = json.loads(Path("results/surface_cues_v10.json").read_text())["rules"]
_lowest = min(r["other"]["file_over_chance"] for r in _c10["other"])
check(all(v["interval"][0] <= 0 and v["over_chance"] < _lowest for v in _sc.values()),
      "a surface-cue rule on v1.0 now clears, or reads above the lowest clearing model")

# --- every control behind the grids, against its exact null (app:stats) -----
_cn = json.loads(Path("results/control_null.json").read_text())
_grid_arms = ("_bixall_bixprompt_argmax", "_bixall_neutral_notools")
_grid_ctl = [c for c in _cn["cells"] if c.endswith(_grid_arms)]
_grid_out = [c for c in _grid_ctl if c in _cn["outside_central_95"]]
check(len(_grid_ctl) == 26
      and f"Of the ${len(_grid_ctl)}$ control files behind Table~\\ref{{tab:gridbix}}, "
          f"${len(_grid_out)}$ fall outside" in _flat,
      f"app:stats' grid control census is {len(_grid_out)} of {len(_grid_ctl)}")


def _prints_refusal(cell):
    tag, rest = next((t, cell[len(t) + 1:]) for t in _pcm.TAGS if cell.startswith(t + "_"))
    name, arm = rest.split("_", 1)
    row = next(r for r in _pc["files"][name]["rows"] if r["tag"] == tag)
    return arm == "neutral_notools" and not row["framing"]["readable"]


check(len(_grid_out) == 3 and all(_prints_refusal(c) for c in _grid_out)
      and "all three in cells that report a refusal rate" in _flat,
      f"app:stats says every grid control outside its null is in a refusal cell: {_grid_out}")

# --- Proposition 2's floors and the collision bound ----------------------------
_dr2 = json.loads(Path("results/release_drift.json").read_text())
_beta15 = _brk["share_bracketed"]["released"] / 100.0
check(abs(_beta15 - 0.810) < 0.0005
      and f"$\\beta=0.810$, a floor of ${_beta15 / 2:.3f}-0.25=+{100 * (_beta15 / 2 - 0.25):.1f}$" in _flat,
      "app:theory's v1.5 bracketing floor is stale")
_p10 = _dr2["v1_0"]["key_by_rank"]
_beta10 = (_p10[1] + _p10[2]) / 100.0
check(f"$\\beta={_beta10:.3f}$, and the floor and the ceiling are ${100 * (_beta10 / 2 - 0.25):+.1f}$ and "
      f"${_dr2['v1_0']['rank_ceiling_points']:+.1f}$" in _flat,
      f"app:theory's v1.0 floor and ceiling are {100 * (_beta10 / 2 - 0.25):+.1f} and {_dr2['v1_0']['rank_ceiling_points']:+.1f}")
_sv10 = {b["benchmark"]: b for b in json.loads(Path("results/channel_survey.json").read_text())["benchmarks"]}["BixBench v1.0"]
check(f"On v1.0's $159$ numeric items the best rule's in-sample ${_dr2['v1_0']['rank_ceiling_points']:+.1f}$ "
      f"falls to ${100 * _sv10['cross_validated']['credit']:+.1f}$ when its rank is chosen on held-out folds" in _flat
      and _sv10["cross_validated"]["credit"] <= 0,
      "sec:theory's v1.0 rank rule, in sample and held out, is stale")
# --- the hardest constraint on the submission, and the least checked --------
# Nine pages excluding references and appendices. Every structural decision in
# this paper was a page-budget decision first, and until now the budget was
# checked by hand with a throwaway script -- which once reported "8.91 pages"
# for a paper whose main text ended at the foot of page 9, by finding the word
# "References" inside a bibliography entry.
_pdf = Path("build/main.pdf")
if _pdf.exists():
    import page_budget
    _main_pages, _refs_page = page_budget.main_text_end(str(_pdf))
    check(_main_pages is not None,
          "build/main.pdf has no References heading, so the nine-page budget "
          "cannot be read at all")
    check(_main_pages is not None and _main_pages <= 9,
          f"the main text runs to page {_main_pages}, against a nine-page limit")

# --- the two documents a reader of the artifact opens first -----------------
# README.md's opening paragraph and VERIFICATION.md's instrument section carry
# the same 2x3 the paper does, and neither had ever been checked against a
# result file. Both were stale for a day: they were still describing the finding
# as "an agent collects what a letter-reader does not", which is a comparison
# across three uncontrolled axes. Numbers in prose drift; pin them.
_intervals = Path("results/arm_intervals.json")
if _intervals.exists():
    _arms = {a["arm"]: a for a in json.loads(_intervals.read_text())["arms"]}
    _docs = {name: re.sub(r"\s+", " ", Path(name).read_text())
             for name in ("README.md", "VERIFICATION.md")
             if Path(name).exists()}
    # en dashes and minus signs: the docs use U+2212, the JSON uses ASCII
    def _cells(arm):
        """The arm as the docs may legitimately round it: one place or two."""
        _a = _arms[arm]
        _lo, _hi = _a["ci95"]
        return [(f"{_a['margin_over_clean']:+.{_d}f} "
                 f"[{_lo:+.{_d}f},{_hi:+.{_d}f}]").replace("-", "\u2212")
                for _d in (1, 2)]

    for _arm in ("MMLU-Pro, 14B BixBench prompt, one letter",
                 "MMLU-Pro, 14B BixBench prompt, generated",
                 "MMLU-Pro, 14B steered, one letter",
                 "MMLU-Pro, 14B steered, no tool",
                 "MMLU-Pro, 14B neutral, one letter",
                 "MMLU-Pro, 14B neutral, no tool",
                 "MMLU-Pro placebo, 14B neutral, no tool",
                 "MMLU-Pro, phi-4 BixBench prompt, one letter",
                 "BixBench 205, 14B BixBench prompt, one letter",
                 "BixBench 205, 14B BixBench prompt, generated",
                 "BixBench 205, 14B neutral, one letter",
                 "BixBench 205, 14B neutral, no tool"):
        if _arm not in _arms:
            continue
        _want = _cells(_arm)
        for _name, _text in _docs.items():
            _norm = _text.replace("-", "\u2212")
            check(any(_w in _norm for _w in _want),
                  f"{_name} does not carry {_arm} as {_want[0]} or {_want[1]}")

# --- the cell that reads highest is a cell every rollout answered ------------
# The condition does not "have to" name the withheld question: the neutral arm,
# which reads highest, would contradict that.  What is true is narrower: Llama
# refuses under it and Qwen does not, so the claim is about compliance and is
# checked as one.
_arms14 = Path("results/agentic_arms_qwen14b.json")
if _arms14.exists():
    _neutral = json.loads(_arms14.read_text())[
        "Qwen/Qwen2.5-14B-Instruct|withheld|notools"]
    check(_neutral["file"]["unparsed"] == 0.0 and _neutral["clean"]["unparsed"] == 0.0,
          "Qwen2.5-14B should produce a final line on every rollout of the neutral no-tool "
          f"arm; the unparsed rates are {_neutral['file']['unparsed']} and "
          f"{_neutral['clean']['unparsed']}")

# --- the prompt the paper quotes is the prompt the code runs -----------------
# fig:prompts reproduces BixBench's own MCQ template and names the revision it
# was taken from.  Both are checkable against the things they claim to copy, and a
# paper that misquotes the prompt it is decomposing has no decomposition.
_probe_src = Path("agentic_probe.py").read_text()
_bix_literal = re.search(r"BIXBENCH_PROMPT = \((.*?)\)\n", _probe_src, re.DOTALL)
check(_bix_literal is not None,
      "agentic_probe.py no longer defines BIXBENCH_PROMPT, so fig:prompts' quoted "
      "template cannot be checked against the code that ran")
if _bix_literal:
    _template = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', _bix_literal.group(1)))
    _template = _template.encode().decode("unicode_escape")
    for _fragment in ("Extract the single letter answer to the following question "
                      "from the given options.",
                      "You must pick one answer even if you are unsure.",
                      "You must only output a single letter answer in XML format"):
        check(_fragment in _template,
              f"agentic_probe.py's BixBench template no longer contains "
              f"{_fragment!r}, which fig:prompts quotes")
        check(_fragment in _flat,
              f"fig:prompts no longer quotes {_fragment!r} from BixBench's template")

_prov_commit = json.loads(
    Path("data/external/PROVENANCE.json").read_text())["upstream_commit"]
check(len(_prov_commit) >= 7 and "\\emph{Template.} BixBench's own, verbatim" in _flat,
      "fig:prompts no longer says its template is BixBench's own, verbatim, or the vendored revision is unrecorded")


# --- BixBench's own three published no-data arms -----------------------------
# The paper's lead result, and it is not our instrument: the responses and the
# grades are upstream bytes. What is pinned is every number sec:open quotes,
# the direction of the comparison, and the claim that the arms are the same
# questions -- because the whole argument is that one file, one pair of models
# and three conditions disagree by thirty points.
_repair = Path("results/published_repair.json")
_repair_num = Path("results/published_repair_numeric.json")
if _repair.exists():
    _rp = json.loads(_repair.read_text())["releases"]
    _v15 = _rp["v1.5"]
    _CLAUDE, _GPT = "claude-3-5-sonnet-latest", "gpt-4o"
    check(set(_v15) == {_CLAUDE, _GPT},
          f"published_repair covers {sorted(_v15)}, not the two models BixBench ran")

    # every arm is the same 205 questions in the same 59 capsules
    for _model, _entry in _v15.items():
        _ns = {a["n"] for a in _entry["arms"].values()}
        _cs = {a["n_clusters"] for a in _entry["arms"].values()}
        check(_ns == {205} and _cs == {59},
              f"v1.5 {_model}: the arms hold {sorted(_ns)} questions in "
              f"{sorted(_cs)} capsules, not 205 in 59")
        check(len(_entry["arms"]) == 3,
              f"v1.5 {_model} has {len(_entry['arms'])} arms, not three")
    _fc = _v15[_CLAUDE]["arms"]["mcq_forced"]["accuracy"]
    _fg = _v15[_GPT]["arms"]["mcq_forced"]["accuracy"]
    _open = [_v15[_CLAUDE]["arms"]["open_ended"]["accuracy"], _v15[_GPT]["arms"]["open_ended"]["accuracy"]]
    _dc, _dg = _v15[_CLAUDE]["declined"]["mcq_decline"], _v15[_GPT]["declined"]["mcq_decline"]
    # The abstract gives one deleted-option rate for both models; that is only
    # honest while they share it, to the digit it is typed at.
    check(f"{_open[0]:.1f}" == f"{_open[1]:.1f}",
          f"the abstract gives one deleted-option rate for both models, and they now "
          f"differ: {_open[0]:.1f}% and {_open[1]:.1f}%")
    _sc_, _sg_ = _v15[_CLAUDE]["said_no_data"]["open_ended"], _v15[_GPT]["said_no_data"]["open_ended"]
    _att = {m: json.loads(Path("results/release_arms.json").read_text())["releases"]["v1.5"][m]["open_when_attempted"]
            for m in (_CLAUDE, _GPT)}
    _right = {m: _att[m]["n_right_all"] for m in (_CLAUDE, _GPT)}
    _cnt = {m: round(_v15[m]["arms"]["mcq_forced"]["accuracy"] * 205 / 100) for m in (_CLAUDE, _GPT)}
    check(_right[_CLAUDE] == _right[_GPT] and all(abs(100 * _right[m] / 205 - _o) < 1e-6 for m, _o in
                                                  zip((_CLAUDE, _GPT), _open))
          and all(abs(100 * _cnt[m] / 205 - _v15[m]["arms"]["mcq_forced"]["accuracy"]) < 1e-6 for m in _cnt),
          "the counts behind the published arms' rates do not reproduce them")
    for _site, _span in (
            ("sec:open's recall bound", f"What the models can state does not account for this margin. A model that knew "
                                        f"the answers it states unaided and guessed among four on the "
                                        f"rest would score ${75 * _right[_GPT] / 205:.1f}$ points above chance; one that also knew "
                                        f"the questions it declines as often as those it attempts, "
                                        f"${0.75 * _att[_GPT]['accuracy']:.1f}$ and ${0.75 * _att[_CLAUDE]['accuracy']:.1f}$. "
                                        f"Where the two models state a number"),
            ("sec:open", f"On v1.5, gpt-4o and Claude 3.5 Sonnet each state the answer unaided on ${_right[_GPT]}$ of "
                         f"$205$ questions (${_open[0]:.1f}\\%$); on ${_sg_:.1f}$ and ${_sc_:.1f}\\%$ they state that the "
                         f"value cannot be determined without the data, and among the replies that attempt an answer "
                         f"they are correct on ${_att[_GPT]['accuracy']:.1f}$ and ${_att[_CLAUDE]['accuracy']:.1f}\\%$. "
                         f"Given a refusal option, they take it on ${_dg['rate']:.1f}$ and ${_dc['rate']:.1f}\\%$. Forced "
                         f"to choose among four options, they are correct on ${_cnt[_GPT]}$ and ${_cnt[_CLAUDE]}$ "
                         f"(${_fg:.1f}$ and ${_fc:.1f}\\%$)")):
        check(_span in _flat, f"{_site} should say {_span!r}")
    check(all(_o < _f for _o, _f in zip(_open, [_fc, _fg])),
          "the deleted-option arm no longer scores below the forced arm, so "
          "sec:open's whole argument is stale")
    # knowledge short of a value: the stated numbers read by their nearest option (partial_knowledge.py)
    _pk = json.loads(Path("results/partial_knowledge.json").read_text())
    _pkg, _pkc = _pk["v1.5|gpt-4o"], _pk["v1.5|Claude 3.5 Sonnet"]
    _pkg0, _pkc0 = _pk["v1.0|gpt-4o"], _pk["v1.0|Claude 3.5 Sonnet"]
    check(all(_pk[k]["nearest_is_key_when_stated"]["mean"] < 25.0 for k in _pk if "|" in k),
          "a published model's stated numbers are now nearer the key than a random option, which the abstract, "
          "the introduction and sec:open deny")
    _civ = lambda v: f"$[{v['lo']:.1f},{v['hi']:.1f}]$"
    _civs = lambda v: f"$[{v['lo']:+.1f},{v['hi']:+.1f}]$"
    for _span in (
            f"Where the two models state a number, the option nearest it is the key on "
            f"${_pkg['nearest_is_key_when_stated']['mean']:.1f}$ and ${_pkc['nearest_is_key_when_stated']['mean']:.1f}\\%$ "
            f"of v1.5's numeric questions, where a random option would be on $25\\%$, while their forced choice on those "
            f"questions is correct on ${_pkg['forced_when_stated']['mean']:.1f}$ and "
            f"${_pkc['forced_when_stated']['mean']:.1f}\\%$. These bound what the models can state, not what they know",
            f"On v1.5's $105$ numeric questions gpt-4o states a number on ${_pkg['stated_share']:.1f}\\%$ and Claude 3.5 Sonnet "
            f"on ${_pkc['stated_share']:.1f}\\%$, and the option nearest it is the key on "
            f"${_pkg['nearest_is_key_when_stated']['mean']:.1f}$ {_civ(_pkg['nearest_is_key_when_stated'])} and "
            f"${_pkc['nearest_is_key_when_stated']['mean']:.1f}\\%$ {_civ(_pkc['nearest_is_key_when_stated'])} of those",
            f"would score ${_pkg['predicted_forced']:.1f}$ and ${_pkc['predicted_forced']:.1f}\\%$; the forced-choice runs score "
            f"${_pkg['forced_minus_predicted']['mean']:.1f}$ {_civs(_pkg['forced_minus_predicted'])} and "
            f"${_pkc['forced_minus_predicted']['mean']:.1f}$ {_civs(_pkc['forced_minus_predicted'])} points more",
            f"On v1.0's $159$ the two state a number on ${_pkg0['stated_share']:.1f}$ and ${_pkc0['stated_share']:.1f}\\%$, "
            f"nearest the key on ${_pkg0['nearest_is_key_when_stated']['mean']:.1f}$ and "
            f"${_pkc0['nearest_is_key_when_stated']['mean']:.1f}\\%$ of those; the forced-choice runs score "
            f"${_pkg0['forced_minus_predicted']['mean']:.1f}$ {_civs(_pkg0['forced_minus_predicted'])} and "
            f"${_pkc0['forced_minus_predicted']['mean']:.1f}$ {_civs(_pkc0['forced_minus_predicted'])} points above that "
            f"prediction",
            f"is the option nearest the model's own stated number on "
            f"${_pkg0['pick_is_nearest_when_stated']['mean']:.1f}$ {_civ(_pkg0['pick_is_nearest_when_stated'])} and "
            f"${_pkc0['pick_is_nearest_when_stated']['mean']:.1f}\\%$ {_civ(_pkc0['pick_is_nearest_when_stated'])} of "
            f"those questions"):
        check(_span in _flat, f"the partial-knowledge test is stated otherwise than partial_knowledge.json: {_span!r}")
    # tab:arms is printed by release_arms.py --latex
    import release_arms as _ra
    _arms_body = re.sub(r"\s+", " ", _table_body("tab:arms"))
    for _line in _ra.table_rows(json.loads(Path("results/release_arms.json").read_text())):
        check(_line in _arms_body, f"tab:arms' row should read {_line}")
    for _model in (_CLAUDE, _GPT):
        _p = _v15[_model]["forced_minus_open"]
        check(_p["n_paired"] == 205 and _p["n_clusters"] == 59,
              f"{_model}'s paired comparison is {_p['n_paired']} rows in "
              f"{_p['n_clusters']} capsules")
    # app:arms: each of BixBench's three verifiers, forced and with the options deleted
    _g = {m: _v15[m]["graders"] for m in (_CLAUDE, _GPT)}
    _pc3 = lambda m, arm: [f"${_g[m][arm][v]['accuracy']:.1f}\\%$"
                           for v in ("llm_verifier", "str_verifier", "range_verifier")]
    _n3 = [_g[_CLAUDE]["mcq_forced"][v]["n"] for v in ("llm_verifier", "str_verifier", "range_verifier")]
    _c3, _g3 = _pc3(_CLAUDE, "mcq_forced"), _pc3(_GPT, "mcq_forced")
    _co, _go = _pc3(_CLAUDE, "open_ended"), _pc3(_GPT, "open_ended")
    check(f"Claude 3.5 Sonnet scores {_c3[0]} on the ${_n3[0]}$ judged by a model, {_c3[1]} on the ${_n3[1]}$ "
          f"matched exactly and {_c3[2]} on the ${_n3[2]}$ given a numeric range, and gpt-4o {_g3[0]}, {_g3[1]} "
          f"and {_g3[2]}. Without options the same three cells score {_co[0]}, {_co[1]} and {_co[2]}, "
          f"and {_go[0]}, {_go[1]} and {_go[2]}" in _flat,
          "app:arms' per-verifier sentence no longer matches published_repair.json")
    check(max(float(x.strip("$\\%")) for x in _co + _go) < 6.5,
          "app:arms says the deleted-option collapse holds under every grader")

    # sec:open: the generous regrade, the six open models under it, and the drift between releases
    _ra_d = json.loads(Path("results/release_arms.json").read_text())
    _any = {m: _ra_d["releases"]["v1.5"][m]["open_regrade"]["any_number_5pct"] for m in (_CLAUDE, _GPT)}
    _ours = [r["any_number_5pct"] for r in _ra_d["our_free_runs_bixbench"].values()]
    _nk = {r: {_ra_d["releases"][r][m]["open_regrade"]["n_numeric_key"] for m in (_CLAUDE, _GPT)}
           for r in ("v1.0", "v1.5")}
    check(len(_ours) == 6 and all(r["n"] == 105 for r in _ra_d["our_free_runs_bixbench"].values())
          and all(len(v) == 1 for v in _nk.values())
          and f"(${_nk['v1.0'].pop()}$ on v1.0, ${_nk['v1.5'].pop()}$ on v1.5)" in _flat
          and f"raises no model above ${max(_ours):.1f}\\%$" in _flat,
          "tab:arms' numeric-key counts or tab:free's any-number ceiling no longer match release_arms.json")
    _v10a = _ra_d["releases"]["v1.0"]
    _o10 = {m: _v10a[m]["open_regrade"]["any_number_5pct"] for m in (_CLAUDE, _GPT)}
    check(f"the numeric keys give gpt-4o ${_o10[_GPT]:.1f}\\%$ and Claude ${_o10[_CLAUDE]:.1f}\\%$ on v1.0, "
          f"and ${_any[_GPT]:.1f}\\%$ and ${_any[_CLAUDE]:.1f}\\%$ on v1.5" in _flat,
          "app:arms' regrade sentence no longer matches release_arms.json")
    _drift_f = max(abs(_ra_d["releases"]["v1.5"][m]["arms"]["mcq_forced"]["accuracy"]
                       - _v10a[m]["arms"]["mcq_forced"]["accuracy"]) for m in (_CLAUDE, _GPT))
    check(_drift_f < 5.0, f"the forced score moved {_drift_f:.2f} points between releases")

    # v1.0 is the release the published baselines were actually run on, and
    # there deleting the options makes no measurable difference.
    if "v1.0" in _rp and _rp["v1.0"]:
        for _model, _entry in _rp["v1.0"].items():
            _p = _entry["forced_minus_open"]
            check(_p["interval"][0] < 0 < _p["interval"][1],
                  f"v1.0 {_model}: deleting the options now makes a measurable "
                  f"difference ({_p['difference']:+.2f} "
                  f"{[round(x, 1) for x in _p['interval']]})")

if _repair_num.exists():
    _rn = json.loads(_repair_num.read_text())["releases"]["v1.5"]
    # the numeric-tolerance subset: the benchmark's own range verifier, where
    # the deleted-option arm is zero of sixty-one for both models
    for _model, _entry in _rn.items():
        check(_entry["arms"]["open_ended"]["accuracy"] == 0.0,
              f"{_model}'s open-ended accuracy on the range-verified items is "
              f"{_entry['arms']['open_ended']['accuracy']:.2f}%, not the zero app:arms gives")
        check(_entry["arms"]["mcq_forced"]["n"] == 61,
              f"the range-verified subset holds "
              f"{_entry['arms']['mcq_forced']['n']} items, not 61")

# --- the numeric/text split of the BixBench cells: data only ---------------
_split = Path("results/cell_split.json")
if _split.exists():
    _cs = json.loads(_split.read_text())["cells"]
    if "bix template argmax" in _cs:
        _num = _cs["bix template argmax"]["numeric"]
        _txt = _cs["bix template argmax"]["text"]
        check(_num["n_items"] == 105 and _txt["n_items"] == 100,
              f"the BixBench split is {_num['n_items']}/{_txt['n_items']}, not 105/100")
        check(_num["vs_chance_interval"][0] > 0,
              "the template's numeric half no longer clears chance")
        check(_txt["vs_chance_interval"][0] < 0 < _txt["vs_chance_interval"][1],
              "the template's text half now clears chance")
    if "bix neutral generated" in _cs:
        for _which in ("numeric", "text"):
            check(_cs["bix neutral generated"][_which]["vs_chance_interval"][0] > 0,
                  f"the neutral framing's {_which} half no longer clears chance")
    for _name, _icell in (("bix template argmax", ("BixBench's template", "letter")),
                          ("bix template generated", ("BixBench's template", "generated")),
                          ("bix neutral argmax", ("chat framing", "letter")),
                          ("bix neutral generated", ("chat framing", "generated"))):
        if _name not in _cs:
            continue
        _a = _cs[_name]["all"]
        check(abs(_irows[("bixall", _icell[0])][_icell[1]]["over_chance"] - _a["vs_chance"]) < 0.05,
              f"cell_split.py and instrument_cells.py disagree on {_name}")
        check(_a["control_vs_chance"] < 0, f"{_name}'s control no longer sits below chance")

# --- membership inference on the option sets: no longer in the paper --------
# Recognition predicts a gap larger on the released file than on its control;
# every count is derived from the grids rather than typed.
_mem = Path("results/option_membership.json")
if _mem.exists():
    _mm = json.loads(_mem.read_text())
    _gaps = _mm["correct_minus_wrong"]
    check(len(_gaps) == 6, f"the membership arm reports {len(_gaps)} gaps, not six")
    for _key, _g in _gaps.items():
        check(_g["interval"][0] <= 0 <= _g["interval"][1],
              f"the membership gap {_key} now clears zero ({_g['difference']:+.3f} "
              f"{[round(x, 3) for x in _g['interval']]})")
    _aucs = [v["auc"] for k, v in _mm["attacks"].items() if "_vs_control" in k]
    check(min(_aucs) > 0.9,
          f"the released-against-control attack AUCs are now {_aucs}, not near-perfect")

_mgrid = Path("results/option_membership_grid.json")
_bgrid = Path("results/option_membership_grid_bix.json")
if _mgrid.exists():
    import membership_grid as _mgmod

    _mg = json.loads(_mgrid.read_text())
    _mrows = _mgmod.summary(_mg, cell=_mgmod.CELL["mmlupro"])
    _mc = _mgmod.headline(_mg, _mrows, _mgmod.CELL["mmlupro"])
    check(_mc["models"] == 13, f"the membership grid holds {_mc['models']} models, not thirteen")
    check(all("released_minus_control" in r and "control_correct_minus_wrong" in r
              for r in _mg.values()),
          "some membership rows have no control arm")
    check(_mc["informative_difference"] == 0 and _mc["difference_cells"] == 78,
          f"the membership difference now clears on {_mc['informative_difference']} "
          f"informative models")
    _q = _mg["Qwen/Qwen2.5-32B-Instruct"]
    _rel = _q["correct_minus_wrong"][_mgmod.CELL["mmlupro"]]
    _dif = _q["released_minus_control"][_mgmod.CELL["mmlupro"]]
    check(_rel["interval"][0] > 0 and _dif["interval"][0] <= 0,
          "the 32B's released membership arm should clear and its difference should not; now "
          f"{_rel['interval']} and {_dif['interval']}")
    if _bgrid.exists():
        _bg = json.loads(_bgrid.read_text())
        _brows = _mgmod.summary(_bg, refused=_mgmod.compliance(
            "results/scale_grid_bixbench.json", "bixbench/argmax", 4),
            cell=_mgmod.CELL["bixall"])
        _bc = _mgmod.headline(_bg, _brows, _mgmod.CELL["bixall"])
        check(_bc["models"] == 13 and _bc["refusing"] == 0,
              f"the BixBench membership grid holds {_bc['models']} models with "
              f"{_bc['refusing']} declining, not thirteen and none")
        check(_bc["difference_cells_clearing"] <= _bc["expected_by_chance"],
              f"BixBench's membership difference now clears on "
              f"{_bc['difference_cells_clearing']} of {_bc['difference_cells']}, above the "
              f"{_bc['expected_by_chance']:.1f} expected by chance")

# --- lambda under the per-rank relaxation: no longer in the paper ------------
_lam = Path("results/lambda_relaxed.json")
if _lam.exists():
    _lr = json.loads(_lam.read_text())
    check(_lr["ordering_survives"],
          f"freeing one lambda per rank now reorders the models: shared "
          f"{_lr['ordering_shared']} against relaxed {_lr['ordering_relaxed']}")
    check(sum(b["n_rejected"] for b in _lr["models"].values()) > 0,
          "no draw rejects the shared-lambda model any more")

# --- our own run of the repair, across models --------------------------------
# sec:open's lead result is BixBench's two models; this is the arm that removes
# "two models" from the limitation. The finding is that free-response accuracy
# is near-constant where the multiple-choice margin is not, so both halves of
# that contrast are pinned, and so is the compliance rule that decides which
# multiple-choice cells are printable at all.
_FREE = {"bixnum": Path("results/free_grid_bixnum.json"),
         "mmlupro": Path("results/free_grid_mmlupro.json")}
if _FREE["bixnum"].exists():
    _fgs = {k: json.loads(p.read_text()) for k, p in _FREE.items() if p.exists()}
    _fg = _fgs["bixnum"]
    _ftab = _table_body("tab:free")
    _every = [r for g in _fgs.values() for r in g["rows"]]
    for _key, _g in _fgs.items():
        _lo, _hi = _g["free_accuracy_range"]
        check(_g["n_models"] >= 6,
              f"the {_key} free-response grid holds {_g['n_models']} models, not six")
        # sec:open quotes BixBench's any-number reading, the generous one; the
        # last-number accuracies are tab:free's rows, pinned below
        if _key == "bixnum":
            _anyn = [v["any_number_5pct"] for v in
                     json.loads(Path("results/release_arms.json").read_text())["our_free_runs_bixbench"].values()]
            check(f"raises no model above ${max(_anyn):.1f}\\%$" in _flat and max(_anyn) < 5.0,
                  f"tab:free's caption should cap the any-number reading at {max(_anyn):.1f}%")
        check(_hi < 5.0,
              f"{_key}'s widest free-response reading is now {_hi:.1f}%, so "
              f"'none above four per cent' is stale")
        _margins = [r["mcq_margin"] for r in _g["rows"] if r["mcq_readable"]]
        check(len(_margins) == 5, f"{_key}'s multiple-choice arm should parse for five of six models")
        check(max(_margins) - min(_margins) > (_hi - _lo),
              f"on {_key} the multiple-choice arm no longer varies more across "
              f"models than the free-response arm does, which is the point")
        check(_g["n_readable"] < _g["n_models"],
              f"no {_key} multiple-choice cell is withheld any more, so "
              f"tab:free's caption describes a filter that does nothing")
        check(_g["max_unparsed"] == 5.0,
              f"the {_key} grid filters at {_g['max_unparsed']}% unparsed, not "
              f"the 5% tab:grid uses")
    check(f"within ${100 * _fg['tolerance']:.0f}\\%$ of the key" in _flat,
          f"tab:free's caption should state the {100 * _fg['tolerance']:.0f}% "
          f"grading tolerance")
    # every row of tab:free, whole, so a bare number cannot drift past the pin
    for _row in _every:
        _name = _row["model"].split("/")[-1].replace("-Instruct", "").replace(
            "Meta-", "")
        _mcq = (f"${_row['mcq_margin']:+.1f}$ "
                f"$[{_row['mcq_interval'][0]:+.1f},{_row['mcq_interval'][1]:+.1f}]$"
                if _row["mcq_readable"]
                else f"unparsed ${_row['mcq_unparsed']:.0f}\\%$")
        _want = (f"{_name} & ${_row['params']:.1f}$ & {_mcq} & "
                 f"${_row['free_accuracy']:.1f}\\%$ & "
                 f"${_row['free_declined']:.0f}\\%$")
        check(_want in re.sub(r"\s+", " ", _ftab).replace(" \\\\", "\\\\")
              or _want in _ftab,
              f"tab:free's row for {_name} should read {_want!r}")

# ---------------------------------------------------------------------------
# v1.0's template grid, WRONG STEP on BixBench, and the frontier's difficulty
# column for two 70B-class solvers. The first two are no longer in the paper and
# keep their data checks; the third's printed cells are recomputed here.
def _pm(x, digits=1):
    s = f"{x:+.{digits}f}"
    return "-0.0" if s in ("-0.0", "+0.0") and x < 0 else s


_v10 = Path("results/bixbench_v10_grid.json")
if _v10.exists():
    _vg = json.loads(_v10.read_text())
    for _r in _vg["rows"]:
        check(_r["unparsed_rollouts"] == 0,
              f"every model answered every v1.0 rollout; {_r['model']} left "
              f"{_r['unparsed_rollouts']} unparsed")

_bws = Path("results/bixbench_wrong_step.json")
if _bws.exists():
    _ws = json.loads(_bws.read_text())
    _A = _ws["arms"]
    _q, _l = _A["wrong step (Qwen2.5-14B)"], _A["wrong step (Llama-3.1-8B)"]

    check(all(c["covering"][0] > 0 for arm in (_q, _l) for c in arm["difficulty"].values()),
          "every solver's WRONG STEP difficulty on BixBench should clear at the covering level")
    check(not any(c["covering"][0] > 0 or c["covering"][1] < 0
                  for c in _A["repaired"]["difficulty"].values()),
          "the BixBench repair should move no solver given the question")

# --- BixBench's own agent, with the data and without it (app:withdata) ------
# The one experiment in the paper where an agent has the data. tab:withdata's
# rows are what bixbench_withdata.table_rows prints from the summary, which
# tests/test_withdata.py re-derives from the shipped rows; each number the prose
# quotes is rebuilt here from the same file, with the helper's formatting.
_wd_path = Path("results/bixbench_withdata.json")
check(_wd_path.exists(), "results/bixbench_withdata.json is missing, so tab:withdata cannot be checked")
if _wd_path.exists():
    import bixbench_withdata as _bw
    _wd = json.loads(_wd_path.read_text())
    _wrows = _bw.table_rows(_wd)
    _wbody = " ".join(_table_body("tab:withdata").split())
    # the text-protocol runs' three agents, and every published-protocol run set the
    # summary holds, each by two conditions
    _react = [t for t, _, _ in _bw.AGENTS_REACT if f"{t}|data" in _wd]
    check(len(_wrows) == 2 * (3 + len(_react)),
          f"tab:withdata should hold {3 + len(_react)} run sets by two conditions; the summary "
          f"gives {len(_wrows)} rows")
    for _row in _wrows:
        check(" ".join(_row.split()) in _wbody, f"tab:withdata should carry the row {_row!r}")
    _tags = [t for t, _ in _bw.AGENTS]

    def _own(tag, cond):
        return _wd[f"{tag}|{cond}"][f"reader:{tag}"]
    for _t in _tags:
        for _c in ("data", "nodata"):
            _n = _own(_t, _c)["forced"]["numeric"]
            check(_n["released"]["mean"] <= min(_n["placebo"]["mean"], _n["repaired"]["mean"]),
                  f"the paper says the released options give every family's runs their lowest score; "
                  f"{_t} {_c} reads {_n['released']['mean']} released")
    def _ci(c):
        return f"${_pm(c['mean'])}$ $[{_pm(c['lo'])},{_pm(c['hi'])}]$"

    def _clears(c):
        return c["lo"] > 0 or c["hi"] < 0
    _nd = {t: _wd[f"{t}|data-nodata"] for t in _tags}
    _num = {(t, c): _own(t, c)["forced"]["numeric"] for t in _tags for c in ("data", "nodata")}
    _eff = {t: -_num[(t, "data")]["released-repaired"]["mean"] for t in _tags}
    _plc = {t: -_num[(t, "data")]["released-placebo"]["mean"] for t in _tags}
    _qfn = _nd["qwen72b"]["reader:qwen72b"]["forced_numeric"]
    _qw5 = _nd["qwen72b"]["numeric_within_5pct"]
    # every run set, both protocols, read by the reader tab:withdata prints (its own
    # family, or its PRIMARY_READER), with the data
    _prim = {t: t for t in _tags} | {t: _bw.PRIMARY_READER.get(t, r) for t, _, r in _bw.AGENTS_REACT if t in _react}
    _fam = {t: t for t in _tags} | {t: r for t, _, r in _bw.AGENTS_REACT if t in _react}

    def _pr(tag, cond):
        return _wd[f"{tag}|{cond}"][f"reader:{_prim[tag]}"]
    _pnum = {(t, c): _pr(t, c)["forced"]["numeric"] for t in _prim for c in ("data", "nodata")}
    _peff = {t: -_pnum[(t, "data")]["released-repaired"]["mean"] for t in _prim}
    _pmcq = {t: _wd[f"{t}|data-nodata"][f"reader:{_prim[t]}"]["forced_numeric"] for t in _prim}
    # sec:withdata and the introduction: the two steps of the swap, and the bracketing split
    _plift = {t: -_pnum[(t, "data")]["released-placebo"]["mean"] for t in _prim}
    _rstep = {t: -_pnum[(t, "data")]["placebo-repaired"]["mean"] for t in _prim}
    _rs2 = load("results/reader_split.json")
    _rsv = lambda k: _rs2[k]["pooled"]
    _civ = lambda v: f"${v['mean']:+.1f}$ $[{v['lo']:+.1f},{v['hi']:+.1f}]$"
    check(min(_plift.values()) > 0
          and f"$P$ alone raises the configurations' own graders by {_civ(_rsv('data|own|placebo-released|moved to an edge'))} "
              f"on the moved keys and ${_rsv('data|own|placebo-released|all')['mean']:+.1f}$ over all $105$ items, where the "
              f"nearest-option rule changes by ${_rsv('data|nearest|placebo-released|moved to an edge')['mean']:+.1f}$ and "
              f"${_rsv('data|nearest|placebo-released|all')['mean']:+.1f}$" in _flat,
          "sec:withdata's placebo step under the readers is stale")
    _rr_all = json.loads(Path("results/published_reads_readers.json").read_text())
    _pub_reps = [json.loads(Path(f).read_text())["D1"] for f in ("results/published_reads.json",
                                                                  "results/published_reads_qwen72b_forced.json",
                                                                  "results/published_reads_llama70b_forced.json")]
    _pub_all = [x["repaired-placebo|all"]["mean"] for x in _pub_reps]
    _pub_inward = [x["repaired-placebo|moved inward"]["mean"] for x in _pub_reps]
    # the gradings with the notebook in tab:readers2 on the published runs: their change over all numeric items
    _t2 = load("results/grading_variants.json")["table2"]
    _t2nb = [r["all"] for r in _t2 if r["runs"] == "v1.0, published" and r["all"] is not None
             and r["grader"] not in ("nearest option", "published") and "code-free" not in r["grader"]
             and "within" not in r["grader"]]
    # sec:cost: U also moves keys inward, where the rule loses, and so do most gradings (not those offered an option
    # stating that no value is within 5%, which grade the tolerance)
    _inw = [c["moved inward"]["mean"] for k, c in load("results/grading_variants.json")["rank"].items()
            if k.endswith("|original") and c.get("moved inward")] + _pub_inward
    check("also moves keys inward, where the rule and most gradings lose." in _flat
          and _rsv('data|nearest|repaired-placebo|moved inward')['mean'] < 0
          and sum(x < 0 for x in _inw) > len(_inw) / 2 and all(x < 0 for x in _pub_inward)
          and {round(x, 6) for x in _pub_all} <= {round(x, 6) for x in _t2nb},
          "sec:cost's account of the inward keys is stale")
    import reader_split as _rsm
    import grading_variants_analysis as _gva
    # tab:readers2, one line a grading, printed by grading_variants_analysis.py --latex readers2; app:replication's
    # tab:readersfull, every contrast with its interval, by reader_split.py --latex-full; tab:variants and tab:allq by
    # grading_variants_analysis.py --latex variants|allq. Each holds its header rows and the rows printed, no more
    _gvr = load("results/grading_variants.json")
    for _tab, _rows, _head in (("tab:readers2", _gva.table2_rows(_gvr), 2),
                               ("tab:readersfull", _rsm.table_rows(_rs2, load("results/published_reads.json")), 2),
                               ("tab:variants", _gva.variants_rows(_gvr), 2),
                               ("tab:allq", _gva.allq_rows(_gvr), 1)):
        _r2body = " ".join(_table_body(_tab).split())
        for _row in _rows:
            check(" ".join(_row.split()) in _r2body, f"{_tab} should carry the row {_row}")
        check(_r2body.count("\\\\") == len(_rows) + _head,
              f"{_tab} should hold exactly the {len(_rows)} rows its script gives")
    # tab:cost (Table 2), every cell read from the file an appendix table prints it from, by cost_table.py
    import cost_table as _ctm
    _ctbody = " ".join(_table_body("tab:cost").split())
    _ctrows = _ctm.table_rows()
    for _row in _ctrows:
        check(" ".join(_row.split()) in _ctbody, f"tab:cost should carry the row {_row}")
    check(_ctbody.count("\\\\") == len(_ctrows) + 1, f"tab:cost should hold exactly the {len(_ctrows)} rows cost_table.py gives")
    # sec:cost's paragraph on who bears the cost quotes tab:cost's cells
    _ctc = {(r, g): (m, a) for r, g, m, a in _ctm.cells()}
    _ctv = lambda runs, grader, col=0: _ctm.means(_ctc[(runs, grader)][col])
    _ct_cf = [x for r in ("v1.0, published", "v1.5, seven configurations", "v1.5, new seeds")
              for x in _ctv(r, "code-free")]
    _ct_nbp, _ct_nbpa = _ctv("v1.0, published", "with the notebook"), _ctv("v1.0, published", "with the notebook", 1)
    _ct_nbs = _ctv("v1.5, new seeds", "with the notebook")
    _ct_cur = _ctv("v1.5, gpt-5.1", "with the notebook") + _ctv("v1.5, gpt-6-luna, DeepSeek-V4-Pro", "with the notebook")
    _ct_wt = [x for (r, g), (m, a) in _ctc.items() if g == "within $5\\%$" for x in _ctm.means(m)]
    _ct_ps = [r["p"] for r in _gvr["table2"] if r["runs"] == "v1.5, new seeds" and r["grader"] in
              ("Qwen2.5-72B", "gemma-3-27b", "Llama-3.3-70B")]
    _ct_ra = _ctv("v1.5, gpt-5.1", "nearest-option rule", 1) + _ctv("v1.5, gpt-6-luna, DeepSeek-V4-Pro", "nearest-option rule", 1)
    check(f"and bear most of the cost, ${min(_ct_cf):+.1f}$ to ${max(_ct_cf):+.1f}$ points on the moved keys of the "
          f"published runs, the seven configurations and the new seeds" in _flat
          and f"Graders given BixBench's prompt and the notebook bear less: ${min(_ct_nbp):+.1f}$ to "
              f"${max(_ct_nbp):+.1f}$ on the published runs' moved keys and ${min(_ct_nbpa):+.1f}$ to "
              f"${max(_ct_nbpa):+.1f}$ over all their items, against the rule's "
              f"${_ctv('v1.0, published', 'nearest-option rule', 1)[0]:+.1f}$. On the new seeds they bear "
              f"${min(_ct_nbs):+.1f}$ to ${max(_ct_nbs):+.1f}$, none significant by a capsule sign-flip test "
              f"($p={min(_ct_ps):.3f}$ to ${max(_ct_ps):.3f}$), and on the current agents gpt-4o's change is "
              f"${min(_ct_cur):+.1f}$ to ${max(_ct_cur):+.1f}$." in _flat
          and len(_ct_ps) == 3 and min(_ct_ps) > 0.05
          and f"code-free graders change by ${min(_ct_wt):+.1f}$ to ${max(_ct_wt):+.1f}$ points" in _flat
          and "to ${:+.1f}$, ${:+.1f}$ and ${:+.1f}$ for gpt-5.1, gpt-6-luna and DeepSeek-V4-Pro".format(*_ct_ra) in _flat,
          "sec:cost's paragraph on who bears the cost no longer quotes tab:cost")
    # the introduction's summary of the same cells
    _ct_rule = [_ctv(r, "nearest-option rule")[0] for r in ("v1.0, published", "v1.5, seven configurations",
                                                             "v1.5, new seeds")]
    check(f"Code-free graders gain about as much or more. Graders given BixBench's prompt and the notebook "
          f"gain ${min(_ct_nbp):.1f}$ to ${max(_ct_nbp):.1f}$ points on the published runs' moved keys and at "
          f"most ${max(_ct_nbpa):.1f}$ over all items, ${min(_ct_nbs):.1f}$ to ${max(_ct_nbs):.1f}$ on new "
          f"runs' moved keys (none significant by a capsule sign-flip test) and ${min(_ct_cur):+.1f}$ to "
          f"${max(_ct_cur):+.1f}$ on the current agents'." in _flat
          and min(_ct_cf) > min(_ct_rule) - 2 and max(_ct_cf) > max(_ct_rule),
          "the introduction's summary of what hiding the rank costs each grader no longer quotes tab:cost")
    _kac = load("results/key_audit_check.json")
    check(f"Without the ${_kac['n_dropped']['v1.5']}$ items, the moved keys' contrasts on v1.5 under the "
          f"nearest-option rule, the configurations' own models and gemma-3-27b are "
          f"${_kac['seven|nearest|repaired-placebo|moved|without audited']['mean']:+.1f}$, "
          f"${_kac['seven|own|repaired-placebo|moved|without audited']['mean']:+.1f}$ and "
          f"${_kac['seven|gemma27b|repaired-placebo|moved|without audited']['mean']:+.1f}$." in _flat,
          "sec:withdata's sentence on the audited capsules is stale")
    # tab:bracket, printed by bracketing.py --latex
    import bracketing as _brkm
    _bb_body = re.sub(r"\s+", " ", _table_body("tab:bracket"))
    for _line in _brkm.table_rows(_brk):
        check(" ".join(_line.split()) in _bb_body, f"tab:bracket's row should read {_line}")
    _mv = [_brk["runs"][r]["data"]["nearest"]["repaired-released|moved to an edge"] for r in _brk["runs"]]
    _nm = [_brk["runs"][r]["data"]["nearest"]["repaired-released|not moved"] for r in _brk["runs"]]
    _mvn = [_brk["runs"][r]["nodata"]["nearest"]["repaired-released|moved to an edge"] for r in _brk["runs"]]
    _pcr = [_brk["placebo_contrast"][r]["data"]["moved to an edge"]["mean"] for r in _brk["placebo_contrast"]]
    # the rule's contrast pooled, per run set, and with one protocol per family (reader_split.py's pools)
    _opf = [_rs2[f"data|nearest|repaired-placebo|moved to an edge|{k}"]["pooled"]["mean"]
            for k in ("one protocol, published", "one protocol, text")]
    # sec:cost: the rule's gain pooled over the seven configurations, positive in each and with one protocol per family
    check(f"Under the nearest-option rule, hiding the rank raises the share of the seven configurations' answers graded "
          f"correct by {_civ(_rsv('data|nearest|repaired-placebo|moved to an edge'))} points on the moved keys and by "
          f"{_civ(_rsv('data|nearest|repaired-placebo|all'))} over all numeric items" in _flat
          and min(_pcr) > 0 and min(_opf) > 0
          and not any(o["lo"] > 0 or o["hi"] < 0 for o in _nm),
          "sec:cost's paragraph on the rule is stale")
    check(abs(_brk["share_bracketed"]["released"] - 81.0) < 0.1 and abs(_brk["share_bracketed"]["repaired"] - 57.1) < 0.1,
          f"the bracketed shares are {_brk['share_bracketed']}")
    # what the swap does to a reading of agents (sec:withdata, and the introduction's sentence)
    _pair = _brk["agent_pairs"]
    _ql, _qg = _pair["qwen72b - llama70b"], _pair["qwen72b - gemma27b"]
    check(f"and it leads Llama-3.3-70B by ${_ql['within_5pct']['mean']:+.1f}$ "
          f"$[{_ql['within_5pct']['lo']:+.1f},{_ql['within_5pct']['hi']:+.1f}]$ and gemma-3-27b by "
          f"${_qg['within_5pct']['mean']:+.1f}$ $[{_qg['within_5pct']['lo']:+.1f},{_qg['within_5pct']['hi']:+.1f}]$"
          in _flat and _ql["within_5pct"]["lo"] > 0 and _qg["within_5pct"]["lo"] > 0,
          "sec:withdata's open-answer agent differences are stale")
    check(all(not _clears(_p[f"{k}|{a}"]) for _p in (_ql, _qg) for k in ("own", "gemma27b", "nearest")
              for a in ("released", "repaired"))
          and "none of the six gradings through options (three graders, two option sets) resolves either lead"
              in _flat,
          "sec:withdata says no reading through options resolves either lead")
    _qd, _qn = _wd["qwen72b|data"], _wd["qwen72b|nodata"]
    check(f"within $5\\%$ of the key on ${_qd['numeric']['within_5pct']['mean']:.1f}\\%$ of numeric items with the "
          f"data and ${_qn['numeric']['within_5pct']['mean']:.1f}\\%$ without, {_ci(_qw5)}, and it leads" in _flat
          and f"under its own MCQ grader, through $R$, the data are worth {_ci(_qfn)}, and none of the six "
              f"gradings through options" in _flat,
          "sec:withdata's Qwen2.5-72B sentence is stale")
    check(_clears(_qw5) and not _clears(_qfn),
          "the paper says the data lifts Qwen2.5-72B's open answers and not its MCQ reading measurably")
    for _t in ("llama70b", "gemma27b"):
        check(not _clears(_nd[_t]["open"]) and not _clears(_nd[_t][f"reader:{_t}"]["forced_numeric"]),
              f"the paper says the data is worth nothing measurable to {_t}")
    _gaps, _rgaps, _rms = {"data": [], "nodata": []}, [], []
    for _t in _tags:
        for _c in ("data", "nodata"):
            for _k, _rb in _wd[f"{_t}|{_c}"].items():
                if _k.startswith("reader:") and "rank_model" in _rb["forced"]["rescue"]:
                    _rs = _rb["forced"]["rescue"]
                    _gaps[_c].append(_rs["released"]["mean"] - _rs["rank_model"]["released"])
                    _rgaps.append(_rs["repaired"]["mean"] - _rs["rank_model"]["repaired"])
    check(len(_gaps["data"]) == len(_gaps["nodata"]) == 5,
          "the rank model should cover five run-reader cells of the text runs in each condition")
    check(max(_gaps["data"] + _gaps["nodata"]) < 0,
          "every text-protocol reader should score below its rank model on the released file")
    # app:withdata
    check(f"within $5\\%$ of the key on ${_qd['numeric']['within_5pct']['mean']:.1f}\\%$ of the numeric items, "
          f"against ${_qn['numeric']['within_5pct']['mean']:.1f}\\%$ without the data, and BixBench's graders score its runs "
          f"{_ci(_nd['qwen72b']['open'])} points higher with the data over all $205$" in _flat,
          "app:withdata's sentence on what the data is worth to Qwen2.5-72B is stale")
    _tm = [_pmcq[t]["mean"] for t in _tags]
    _rm_ = [_pmcq[t]["mean"] for t in _react]
    check(f"the data change the score by ${_pm(min(_tm))}$ to ${_pm(max(_tm))}$ points on the numeric items under the "
          f"text protocol and by ${_pm(min(_rm_))}$ to ${_pm(max(_rm_))}$ under the published protocol" in _flat,
          "app:withdata's MCQ data-effect ranges are stale")
    _mcl = [t for t in _prim if _clears(_pmcq[t])]
    check(_mcl == ["llama70b-react", "qwen3a3b-react"],
          f"the MCQ reading should credit the data measurably for Llama and Qwen3 as published; {_mcl}")
    _cl = [t for t in _tags if _clears(_num[(t, "data")]["released-repaired"])]
    check(_cl == ["qwen72b", "llama70b"], f"the repair should clear for Qwen and Llama; it clears for {_cl}")
    _pcl = [t for t in _tags if _clears(_num[(t, "data")]["released-placebo"])]
    check(_pcl == ["qwen72b", "gemma27b"], f"the placebo should clear for Qwen and gemma; it clears for {_pcl}")
    # the rank model, over every reader but a failed own reading
    _ok_readers = [(t, c, k) for t in _prim for c in ("data", "nodata") for k in _wd[f"{t}|{c}"]
                   if k.startswith("reader:") and not (t in _bw.PRIMARY_READER and k == f"reader:{_fam[t]}")]
    _rm_all = [_wd[f"{t}|{c}"][k]["forced"]["rescue"]["rank_model"]["released-repaired"] for t, c, k in _ok_readers
               if "rank_model" in _wd[f"{t}|{c}"][k]["forced"]["rescue"]]
    for _t, _c, _k in _ok_readers:
        _rs = _wd[f"{_t}|{_c}"][_k]["forced"]["rescue"]
        if _c == "data" and "rank_model" in _rs:
            check(_rs["released"]["mean"] - _rs["rank_model"]["released"]
                  < _rs["repaired"]["mean"] - _rs["rank_model"]["repaired"],
                  f"every reader should read the released file lower against its rank prediction than "
                  f"the repair, with the data; {_k} on {_t} does not")
    # declining, over every run set's table reader
    _dec = [100 * _pr(t, c)["refusal"]["refused"] for t in _prim for c in ("data", "nodata")]
    check(f"select it on ${min(_dec):.0f}$ to ${max(_dec):.0f}\\%$ of the runs they grade" in _flat,
          "app:withdata's decline rates are stale")
    _refc = {t: _pr(t, "data")["refusal"]["numeric"]["released-repaired"] for t in _prim}
    check(all(abs(v["mean"]) < 1 for t, v in _refc.items() if t != "glm45air-react")
          and [t for t, v in _refc.items() if _clears(v)] == (["glm45air-react"] if "glm45air-react" in _refc else []),
          "the swap should move nothing once the reader may decline, but for GLM-4.5-Air's runs")
    _rr = json.loads(Path("results/agent_runs/reruns.json").read_text())
    # an episode that failed twice is one episode run again
    _rr_eps = {(e["question_id"], e["condition"]) for e in _rr["replica_failed"] + _rr["wallclock"]}
    check(f"${len(_rr_eps)}$ of Qwen2.5-72B's text-protocol episodes by a GPU fault and the load it left" in _flat,
          "app:withdata's text-protocol rerun count does not match results/agent_runs/reruns.json")
    # every run the table reads is complete under one protocol
    _runs = [json.loads(l) for _p in sorted(Path("results/agent_runs").glob("trajectories_*.jsonl.gz"))
             for l in gzip.open(_p, "rt")] if Path("results/agent_runs").exists() else []
    check(len(_runs) == 410 * (3 + len(_react)),
          f"{3 + len(_react)} run sets by two conditions by 205 items is {410 * (3 + len(_react))} "
          f"runs; results/agent_runs ships {len(_runs)}")
    # one run per run set, condition and item: a model's runs under two protocols are
    # two run sets, never duplicates of each other
    _final = {(_bw.run_key(r), r["condition"], r["question_id"]): r for r in _runs}
    check(len(_final) == len(_runs), "results/agent_runs ships a run twice")
    for _e in _rr["replica_failed"] + _rr["wallclock"]:
        _r = _final.get(("qwen72b", _e["condition"], _e["question_id"]))
        check(_r is not None and bool(_r["settings"].get("only")),
              f"reruns.json lists qwen72b {_e['condition']} {_e['question_id']} as run again from the "
              f"start, and the shipped run is not the re-run")
    # the published-protocol run sets: an episode our hour limit cut was set aside and run
    # again from the start, so none ships cut, and each cut one ships beside its re-run
    _ncut = 0
    for _t in _react:
        _rp = Path(f"results/agent_runs/reruns_{_t}.json")
        _cut = {(e["condition"], e["question_id"]) for e in json.loads(_rp.read_text())["wallclock"]} \
            if _rp.exists() else set()
        _sp = Path(f"results/agent_runs/superseded_{_t}.jsonl.gz")
        _sup = [json.loads(l) for l in gzip.open(_sp, "rt")] if _sp.exists() else []
        check({(r["condition"], r["question_id"]) for r in _sup} == _cut
              and all(r["termination"] == "wallclock" for r in _sup),
              f"{_t}: the runs shipped as superseded should be exactly the ones reruns_{_t}.json lists, "
              f"each cut by the hour limit")
        check(not any(r["termination"] == "wallclock" for r in _runs if _bw.run_key(r) == _t),
              f"{_t} ships a run the hour limit cut")
        _ncut += len(_cut)
    # --- the published protocol, and the readings every run set gets -----------
    # tab:readers is what bixbench_withdata.reader_rows prints; every number the prose
    # quotes about the published runs, the second readers and the no-reader rule is
    # rebuilt here from the summary, and the reply-limit loop from the shipped runs.
    if _react:
        check(f"${_ncut}$ of the published protocol's $1{{,}}640$ by our one-hour limit" in _flat,
              "app:withdata's count of published-protocol episodes run again is stale")
        _rbody = " ".join(_table_body("tab:readers").split())
        _rrows = _bw.reader_rows(_wd)
        check(len(_rrows) == 3 + len(_react), f"tab:readers should hold {3 + len(_react)} run sets")
        for _row in _rrows:
            check(" ".join(_row.split()) in _rbody, f"tab:readers should carry the row {_row!r}")

        def _lowest(n):
            rel, others = round(n["released"]["mean"], 6), [round(n[a]["mean"], 6) for a in ("placebo", "repaired")]
            return "strict" if rel < min(others) else "tie" if rel == min(others) else "no"
        for _t in _prim:
            check(_lowest(_pnum[(_t, "data")]) == "strict",
                  f"sec:withdata says the released options read lowest with the data for every run set; {_t} does not")
        _nod = [_lowest(_pnum[(t, "nodata")]) for t in _prim]
        check("no" not in _nod and _nod.count("tie") == 1,
              f"every run set but one should read its released options lowest without the data, the "
              f"one a three-way tie; {_nod}")
        _rcl = [t for t in _react if _clears(_pnum[(t, "data")]["released-repaired"])]
        check(_rcl == ["qwen72b-react"], f"the published repair should clear for Qwen2.5 alone; {_rcl}")
        for _t, _txt in (("qwen72b", "Qwen2.5-72B"), ("llama70b", "Llama-3.3-70B")):
            _vs = _wd[f"{_t}-react|data|vs-{_t}"][f"reader:{_t}"]["released-repaired"]
            check(not _clears(_vs), f"the two protocols' repair contrasts now differ measurably for {_txt}")
        # the second readers: gemma-3-27b on every run set, Qwen2.5-72B on GLM-4.5-Air's
        _second = [(t, c, "reader:gemma27b") for t in _prim for c in ("data", "nodata")]
        _second += [("glm45air-react", c, "reader:qwen72b") for c in ("data", "nodata")]
        for _t, _c, _k in _second:
            check(_wd[f"{_t}|{_c}"][_k]["forced"]["numeric"]["released-repaired"]["mean"] < 0,
                  f"every second reader should read the released options below the repair; "
                  f"{_k} on {_t} {_c} does not")
        # the option nearest the agent's number, no reader
        _near = {(t, c): _wd[f"{t}|{c}"]["nearest"]["released-repaired"] for t in _prim for c in ("data", "nodata")}
        check(all(v["mean"] < 0 for v in _near.values()),
              "the nearest-option rule should read the released options below the repair everywhere")
        # what the data is worth, as published
        _rnd = {t: _wd[f"{t}|data-nodata"] for t in _react}
        check(f"the data raise the graders' score by {_ci(_rnd['glm45air-react']['open'])} points for\nGLM-4.5-Air and "
              f"{_ci(_rnd['qwen3a3b-react']['open'])} for Qwen3-30B-A3B".replace("\n", " ") in _flat,
              "app:withdata's published open-ended data effects are stale")
        # where the published agent makes more of the data than the swap
        check(all(_pmcq[t]["mean"] > _peff[t] for t in ("glm45air-react", "llama70b-react", "qwen3a3b-react"))
              and _pmcq["qwen72b-react"]["mean"] < _peff["qwen72b-react"],
              "the published agents' data effects no longer compare with the swap as they did")
        _sub = {c: _wd[f"qwen3a3b-react|{c}"]["submitted"] for c in ("data", "nodata")}
        check(_sub["data"] == _sub["nodata"], f"Qwen3-30B-A3B's submission counts now differ: {_sub}")
        # Qwen3-30B-A3B's own reading, which the tables replace with gemma-3-27b's
        _q3np = [100 * _wd[f"qwen3a3b-react|{c}"]["reader:qwen3a3b"]["forced"]["no_pick"] for c in ("data", "nodata")]
        check(min(_q3np) > 40
              and f"selects no\noption on ${_q3np[0]:.1f}$ and ${_q3np[1]:.1f}\\%$ of its forced gradings".replace("\n", " ") in _flat
              and f"selects no option on ${_q3np[0]:.1f}\\%$ of its forced gradings" in _flat
              and f"replies, cut at $1{{,}}536$ tokens, select no option on ${_q3np[0]:.1f}\\%$ of its own gradings through $R$" in _flat,
              "tab:withdata, tab:readers or tab:bracket misstate Qwen3-30B-A3B's own reading")
        # the reply-limit loop, counted again from the shipped trajectories
        _cen = {t: _bw.reply_limit_census([r for r in _runs if _bw.run_key(r) == t]) for t in _react}
        _q2, _q3 = _cen["qwen72b-react"], _cen["qwen3a3b-react"]
        check(_q2["capped"] == _q2["loops"]
              and f"all ${_q2['capped']}$ of Qwen2.5-72B's reasoning turns that reach the limit" in _flat,
              "app:withdata's Qwen2.5 loop count is stale")
        _thou = lambda n: f"{n:,}".replace(",", "{,}")
        check(f"${_thou(_q3['loops'])}$ of Qwen3-30B-A3B's ${_thou(_q3['capped'])}$, a third of its turns" in _flat
              and 0.30 < _q3["capped"] / _q3["turns"] < 0.37, "app:withdata's Qwen3 loop count is stale")

# --- every other number the abstract, sections and captions state -------------
# Rebuilt from the file each comes from. tests/ and the sections above pin the
# tables their scripts print; these are the numbers typed into prose.
import collections  # noqa: E402

import arm_intervals  # noqa: E402

_sv = {b["benchmark"]: b for b in load("results/channel_survey.json")["benchmarks"]}
_b15, _b10 = _sv["BixBench v1.5"], _sv["BixBench v1.0"]
_p15 = [100 * _b15["key_by_rank"][str(j)] for j in range(4)]
_p10 = [100 * _b10["key_by_rank"][str(j)] for j in range(4)]
_beta15, _beta10 = 100 * _b15["interior_rate"], 100 * _b10["interior_rate"]
_floor15, _rule15 = _beta15 / 2 - 25, _p15[1] - 25
check(abs(_beta15 - _brk["share_bracketed"]["released"]) < 0.1,
      "channel_survey.json and bracketing.json disagree on v1.5's bracketed share")
_n2 = round(_p15[1] * _b15["n_numeric_items"] / 100)
check(abs(100 * _n2 / _b15["n_numeric_items"] - _p15[1]) < 1e-9
      and f"The options also leak the key: v1.5 places it second-smallest on ${_p15[1]:.0f}\\%$ of numeric items, a "
          f"rank that a rule ignoring the question learns on held-out capsules" in _flat
      and abs(100 * _b15["cross_validated"]["credit"] - _rule15) < 1e-9,
      "the abstract's bracketing sentence no longer matches channel_survey.json")
check(f"BixBench v1.5's ${_b15['n_numeric_items']}$ numeric items: the key is the second-smallest of four "
      f"values on ${_p15[1]:.1f}\\%$ of them; ${_beta15:.0f}\\%$ are bracketed, against ${_beta10:.0f}\\%$ of "
      f"v1.0's ${_b10['n_numeric_items']}$ and ${_brk['share_bracketed']['repaired']:.0f}\\%$ under the rank-uniform "
      f"rewrite"
      in _flat, "fig:overview's panel (b) caption is stale")
check(f"(${_b15['n_clusters']}$ capsules) the key's rank distribution is "
      f"$(p_1,\\dots,p_4)=({','.join(f'{x:.1f}' for x in _p15)})\\%$, far from uniform with items clustered by "
      f"capsule" in _flat,
      "sec:theory's v1.5 key-rank law is stale")
_oa15 = {b["benchmark"]: b for b in load("results/option_artifacts.json")["benchmarks"]}["BixBench v1.5"]
_rrule, _cvr = _oa15["best_single_rank_rule"], _oa15["cross_validated_rule"]
_rr95 = load("results/claim_budget.json")
_rr95 = next(c for c in _rr95["claims"] if c["claim"] == "rank rule")["parts"][0]
check(_cvr["rule_selected_per_fold"] == {"1": _cvr["folds"]} and _cvr["folds"] == _b15["n_clusters"]
      and abs(_rr95["estimate"] - 100 * (_rrule["accuracy"] - _rrule["chance"])) < 0.01
      and f"The second-smallest rule scores ${_rr95['estimate']:+.1f}$ "
          f"$[{_rr95['ci_covering_95'][0]:+.1f},{_rr95['ci_covering_95'][1]:+.1f}]$ points above chance, "
          f"correct on ${_n2}$ of the ${_b15['n_numeric_items']}$, and choosing the rank out of sample" in _flat
      and f"in each of ${_cvr['folds']}$ leave-one-capsule-out folds the rank chosen on the other "
          f"${_cvr['folds'] - 1}$ capsules is the second-smallest" in _flat,
      "sec:theory's rank-rule sentence is stale")
check(f"The key is second-smallest on ${_p15[1]:.1f}\\%$ of v1.5's ${_b15['n_numeric_items']}$ numeric items, worth "
      f"${_rule15:+.1f}$ points to a rule whose rank is chosen on held-out capsules; the skew is sharpest in the items "
      f"whose distractors v1.5 rewrote" in _flat,
      "the introduction's v1.5 rank rule is stale")
check(f"on v1.5's ${_b15['n_numeric_items']}$ numeric items $\\beta={_beta15 / 100:.3f}$" in _flat.replace("On v1.5", "on v1.5")
      and f"the second rank holds ${_p15[1]:.1f}\\%$, ${_rule15:+.1f}$; on v1.0's ${_b10['n_numeric_items']}$" in _flat,
      "app:theory's bracketing paragraph is stale")

# with the data: the moved keys, in the abstract, fig:overview, sec:withdata and tab:bracket
_mvd = [_brk["runs"][r]["data"]["nearest"]["repaired-released|moved to an edge"] for r in _brk["runs"]]
_oth = [_brk["runs"][r]["data"]["nearest"]["repaired-released|not moved"] for r in _brk["runs"]]
_blv = _brk["levels"]
check(_blv["moved to an edge"]["n_items"] == _mvd[0]["n_items"] and _blv["not moved"]["n_items"] == _oth[0]["n_items"]
      and all(m["n_clusters"] == _blv["moved to an edge"]["n_clusters"] for m in _mvd)
      and all(o["n_clusters"] == _blv["not moved"]["n_clusters"] for o in _oth),
      "bracketing.json's split levels were not measured on the groups its intervals read")
_mdt = _brk["miss_distance_total"]["data"]
_far = _mdt["gained"]["25 to 100%"] + _mdt["gained"]["over 100%"]
_all = sum(_mdt["gained"].values())
_rs3 = load("results/reader_split.json")
_rule_rp = _rs3["data|nearest|repaired-placebo|moved to an edge"]["pooled"]["mean"]
_own_rp = _rs3["data|own|repaired-placebo|moved to an edge"]["pooled"]["mean"]
# the registered test on the published runs: four of five predictions hold, the kept keys' passing on its margin
# alone and counted as failed
_rpd1 = load("results/replication.json")["D1"]
_kept1 = _rpd1["gain|kept"]
_hold1 = sum(bool(v) for h, v in _rpd1["verdicts"].items() if h in ("H1", "H2", "H3", "H4", "H5")
             and not (h == "H2" and not (_kept1["lo"] <= 0 <= _kept1["hi"])))
_rule1 = _rpd1["repaired-placebo|moved to an edge"]["mean"]
_rpd2q = load("results/replication.json")["D2"]["qwen3-235b|data"]["verdicts"]
_rule1_all = load("results/published_reads_readers.json")["rule"]["repaired-placebo|all"]["mean"]
_rule_all = _rs3["data|nearest|repaired-placebo|all"]["pooled"]["mean"]
_scl = load("results/run_set_scaling.json")["trend"]
_rule7 = _rs3["data|nearest|repaired-placebo|moved to an edge"]["pooled"]["mean"]
check(not any(o["lo"] > 0 or o["hi"] < 0 for o in _oth) and _hold1 == 4
      and sorted(h for h, v in _rpd2q.items() if not v) == ["H1", "H2", "H4"]
      and _scl["all on within_5pct"]["spearman_rho"] < 0 and _scl["all on within_5pct"]["p_value"] < 0.05
      and f"The nearest-option rule accepts ${_rule7:.1f}$ points more on the v1.5 keys that $U$ moves to an "
          f"extreme, and ${_rule_all:.1f}$ more over all numeric items." in _flat
      # H2 fails on the published runs by its interval and H4 holds there; Qwen3-235B-A22B fails H1, H2 and H4
      and _kept1["lo"] > 0 and _rpd1["verdicts"]["H4"]
      and "The gain from a hidden rank falls on misses, so an agent that misses less has less to gain" in _flat,
      "the abstract's or the introduction's with-data sentence is stale")
_lvs = lambda x: f"{round(100 * x, 2):g}"
# sec:withdata and tab:mechanism: how far the newly credited answers are from the key, and on which side
_g = _mdt["gained"]
_kac2 = load("results/key_audit_check.json")["newly_credited_with_data"]
check(f"Of the ${_all}$ answers that $U$ newly accepts with the data, ${_mdt['gained_open_side']}$ lie on the "
      f"key's open side and ${_g['over 100%']}$ are off by more than $100\\%$ (Table~\\ref{{tab:mechanism}})." in _flat
      and abs(_kac2["flagged"] + _kac2["not flagged"] - _all) < 1e-9,
      "sec:withdata's miss-distance sentence is stale, or the audit's flagged keys hold another count")
_mbody = " ".join(_table_body("tab:mechanism").split())
for _row in bk.mechanism_rows(_brk):
    check(" ".join(_row.split()) in _mbody, f"tab:mechanism should carry the row {_row}")
_pc = _brk["placebo_contrast"]
_to_edge = [_pc[r]["data"]["moved to an edge"]["mean"] for r in _pc]
_inward = [_pc[r]["data"]["moved inward"]["mean"] for r in _pc]
check(min(_to_edge) > 0 and max(_inward) <= 0
      and f"and ${_pc['qwen72b']['data']['moved inward']['n_items']}$ from extreme to bracketed" in _flat,
      "sec:withdata's placebo contrast is stale")
_px = _brk["proximity"]
check(f"$P$'s distractors lie farther from the key (median $d$ of the nearest ${_px['placebo']['median']:.3f}$, "
      f"against ${_px['released']['median']:.3f}$ for $R$)" in _flat
      and f"median $d$ of ${_px['released']['median']:.3f}$ from the key in $R$, "
          f"${_px['placebo']['median']:.3f}$ in $P$ and ${_px['repaired']['median']:.3f}$ in $U$"
          in _flat,
      "the proximity of the rewritten distractors is misquoted")
_fr = _brk["format_rule"]
check(f"is correct on ${_fr['released']:.1f}$, ${_fr['placebo']:.1f}$ and ${_fr['repaired']:.1f}\\%$" in _flat,
      "tab:mechanism's format-rule sentence is stale")
check(f"${_brk['non_positive_keys']}$ of the $105$ keys are zero or negative" in _flat,
      "app:theory's count of non-positive keys is stale")
check(f"${100 * _blv['whole file']:.1f}\\%$ over all $46$, ${_lvs(_blv['moved to an edge']['level'])}\\%$ over the "
      f"moved keys' ${_blv['moved to an edge']['n_clusters']}$ and ${_lvs(_blv['not moved']['level'])}\\%$ over the "
      f"other ${_blv['not moved']['n_clusters']}$" in _flat,
      "tab:bracket's caption no longer states the levels its columns are read at")
check(f"$U$ moves ${_mvd[0]['n_items']}$ of v1.5's $105$ numeric keys from bracketed to extreme" in _flat
      and f"$U$ moves ${_mvd[0]['n_items']}$ of the $105$ keys from bracketed to extreme" in _flat,
      "sec:withdata or tab:bracket miscounts the moved keys")
check(f"The cue is available, worth ${_rule15:+.1f}$ points to a rule," in _flat,
      "sec:open's rank-rule figure is stale")
check(_ql["within_5pct"]["n_clusters"] == 46 and abs(_bw.LEVEL - 0.96) < 1e-12
      and "Intervals paired over $46$ capsules at $96\\%$.} \\label{fig:withdata}" in _flat
      and "paired over $46$ capsules at $96.0\\%$, the level that attains $95\\%$ coverage" in _flat
      and abs(_shape["BixBench"]["cells"][[c["as_observed"] for c in _shape["BixBench"]["cells"]].index(True)]
              ["level_for_95_difference"] - 0.96) < 1e-9,
      "fig:withdata or tab:withdata misstates the with-data intervals' capsules or level")

# app:theory's enumerated counterexample, and the distinctness trap below
_thy = load("results/writer_theory.json")
check(abs(_thy["sorted_distractors"]["gamma"]) < 2e-15 and "$\\Gamma=1\\times10^{-15}$" in _flat,
      "app:theory's sorted-distractor enumeration is misquoted")

# app:stats and the captions that state levels
_obs = {n: next(c for c in e["cells"] if c["as_observed"]) for n, e in _shape.items()}
_pro_o, _bix_o = _obs["MMLU-Pro"], _obs["BixBench"]
_levels_txt = (_lvs(_pro_o["level_for_95_margin"]), _lvs(_pro_o["level_for_95_difference"]),
               _lvs(_bix_o["level_for_95_margin"]), f"{100 * _bix_o['level_for_95_difference']:.1f}")
check(round(50 * (_pro_o["coverage_margin"] + _pro_o["coverage_difference"])) == 90
      and f"a nominal $95\\%$ interval covers about $90\\%$ on MMLU-Pro, so its margins are read at "
          f"${_levels_txt[0]}\\%$ and its differences at ${_levels_txt[1]}\\%$, and BixBench's at ${_levels_txt[2]}$ "
          f"and ${_levels_txt[3]}\\%$" in _flat
      and f"margins at ${_levels_txt[2]}\\%$ and differences at ${_levels_txt[3]}\\%$" in _flat,
      f"app:stats or a table caption misstates the covering levels {_levels_txt}")
_cal_all = load("results/bootstrap_calibration.json")
_pro_rows = arm_intervals.load(arm_intervals.resolve(next(e["dump"] for e in _cal_all["arms"]
                                                          if e["arm"].startswith("MMLU-Pro"))), "file")
_pro_sizes = collections.Counter(r["cluster"] for r in _pro_rows)
check(round(100 * max(_pro_sizes.values()) / sum(_pro_sizes.values())) == 23
      and "one of them holding $23\\%$ of the items" in _flat
      and f"${_cal_all['reps']:,}$ simulated files per cell".replace(",", "{,}") in _flat,
      "tab:bootcal's caption misstates MMLU-Pro's largest subject or the simulation count")
check(f"and the interval at the level that attains $1-0.05/{load('results/claim_budget.json')['k']}$ coverage on each file's "
      f"own clusters" in _flat,
      "tab:claims' caption misstates the family level")

# tab:free, fig:reader and app:withdata's settings
_fg_b = load("results/free_grid_bixnum.json")
_anyn = [v["any_number_5pct"] for v in load("results/release_arms.json")["our_free_runs_bixbench"].values()]
_tols = sorted(load("results/free_response.json")[next(iter(load("results/free_response.json")))]
               ["arms"]["free"]["by_tolerance"])
check(_tols == ["0.01", "0.05", "0.1"]
      and f"raises no model above ${max(_anyn):.1f}\\%$" in _flat
      and f"omitted where more than ${_fg_b['max_unparsed']:.0f}\\%$ of the runs cannot be parsed" in _flat,
      "tab:free's caption is stale")
_pcm_params = [_pcm.PARAMS[t] for t in _pcm.TAGS]
check(f"Thirteen open-weight models from five families, from ${min(_pcm_params):.1f}$ to ${max(_pcm_params):.1f}$B"
      in _flat and len(_pcm.TAGS) == 13, "app:reader's model range is stale")
_agent_src = Path("bixbench_agent.py").read_text()
_def = lambda flag: re.search(rf'"--{flag}", type=\w+, default=([\d.]+)\)', _agent_src).group(1)
check(f"at most ${_def('max-steps')}$ steps at temperature ${float(_def('temperature')):.1f}$" in _flat
      and f"${int(_def('max-tokens')):,}$-token reply limit".replace(",", "{,}") in _flat,
      "app:withdata misstates the agent's step limit, temperature or reply limit")
check('"ldp==0.26.0"' in Path("sources/bixbench_49311180/fhda_pyproject_v1.5.0.toml").read_text()
      and "BixBench's ReAct agent at the versions the\nbenchmark specifies".replace("\n", " ") in _flat,
      "app:withdata no longer says the published protocol runs the agent at the versions BixBench pins")

# the files' item and cluster counts, as the captions give them
_n_bix = _prv["gpt-4o"]["arms"]["mcq_forced"]["n"]
_c_pro = _shape["MMLU-Pro"]["observed_clusters"]
_c_bix = _prv["gpt-4o"]["arms"]["mcq_forced"]["n_clusters"]
check(_n_bix == 205
      and f"${_n_bix}$ four-option items in ${_c_bix}$ capsules" in _flat
      and f"gpt-5.1's runs over all ${_n_bix}$ v1.5 questions" in _flat,
      "a caption or sec:withdata misstates MMLU-Pro's or BixBench's item or cluster count")
check(f"BixBench's ${_shape['BixBench']['observed_clusters']}$ capsules, of unequal size, are worth" in _flat
      and f"MMLU-Pro's ${_c_pro}$ source subjects are not" in _flat
      and all(0.86 < c["coverage_margin"] < 0.93 and 0.86 < c["coverage_difference"] < 0.93
              for c in _shape["MMLU-Pro"]["cells"]) and "coverage stays near $0.90$ however many" in _flat,
      "tab:bootcal's caption misstates the shapes or MMLU-Pro's coverage")
check("Paired over $46$ capsules at $96.0\\%$. $^\\dagger$Qwen3-30B-A3B's own" in _flat
      and all(_brk["agent_pairs"][p]["within_5pct"]["n_clusters"] == 46 for p in _brk["agent_pairs"]),
      "tab:readers' caption misstates its capsules or level")

# pins a duplicate elsewhere in the paper would otherwise satisfy
_dist_body = " ".join(_table_body("tab:distinctness").split())
for _t in _thy["distinctness_trap"]:
    _want = (f"{_t['key_marginal']} & ${100 * _t['gamma_symmetric']:+.1f}$ & "
             f"${100 * _t['gamma_reject_against_key']:+.1f}$\\\\")
    check(_want in _dist_body, f"tab:distinctness should read {_want}")
_gr = _prv["gpt-4o"]["graders"]["mcq_forced"]
check(sum(v["n"] for v in _gr.values()) == _prv["gpt-4o"]["arms"]["mcq_forced"]["n"] and len(_gr) == 3
      and f"BixBench grades its ${_prv['gpt-4o']['arms']['mcq_forced']['n']}$ v1.5 questions with three graders" in _flat,
      "app:arms misstates how many questions or verifiers BixBench grades with")
check(f"\\emph{{Open}}: BixBench v1.5's graders on all ${_n_bix}$ items" in _flat
      and f"\\emph{{MCQ}}: BixBench's MCQ grader on all ${_n_bix}$ with the released\noptions".replace("\n", " ") in _flat,
      "tab:withdata's caption miscounts the items")

# --- gpt-4o without the data, through the three option sets (tab:gptnodata) ---------------
import openai_nodata as _ond
_gnd = load("results/openai_nodata_gpt-4o.json")
_gnd_body = " ".join(_table_body("tab:gptnodata").split())
for _row in _ond.table_rows(_gnd):
    check(" ".join(_row.split()) in _gnd_body, f"tab:gptnodata should carry the row {_row}")
check(_gnd_body.count("\\\\") == len(_ond.table_rows(_gnd)) + 2,
      "tab:gptnodata should hold exactly the rows openai_nodata.table_rows gives")
check(_gnd["served"] == ["gpt-4o-2024-11-20"] and "The version is 2024-11-20" in _flat
      and f"(the ${_gnd['releases']['v10']['template']['released']['margin']['n_items'] // 3}$ of v1.0's $159$ that "
          f"have all three)" in _flat,
      "tab:gptnodata's caption misstates the version served or v1.0's item count")
check(f"differences paired over the same items and orderings at ${100 * bw.LEVEL:.1f}\\%$ over capsules. 2nd:" in _flat
      and all(c[k]["n_clusters"] >= 40 for r in _gnd["releases"].values() for c in r.values()
              for k in ("released-placebo", "released-repaired")),
      "tab:gptnodata's caption misstates the level of its paired differences")

_g15, _g10 = _gnd["releases"]["v15"], _gnd["releases"]["v10"]
_gm = lambda c, arm: f"${c[arm]['margin']['mean']:+.1f}$"
_gd = lambda e: f"${e['mean']:.1f}$ $[{e['lo']:+.1f},{e['hi']:+.1f}]$"
check(f"A later gpt-4o (version 2024-11-20; Table~\\ref{{tab:gptnodata}}), run again without the data, is at chance "
      f"with the question withheld ({_gm(_g15['withheld'], 'released')} and {_gm(_g10['withheld'], 'released')} points "
      f"on the two releases) and selects the second-smallest option on "
      f"${_g15['withheld']['released']['pick_rank_shares'][1]:.1f}\\%$ of v1.5's items, where the key lies on "
      f"${_p15[1]:.1f}\\%$" in _flat
      and all(c["withheld"]["released"]["margin"]["lo"] <= 0 <= c["withheld"]["released"]["margin"]["hi"]
              for c in (_g15, _g10))
      and "under BixBench's template, redrawing every distractor changes its margin by no significant amount on "
          "either release, so there the margin rests on the question and the key's value" in _flat
      and all(c["template"][k]["lo"] <= 0 <= c["template"][k]["hi"]
              for c in (_g15, _g10) for k in ("released-placebo", "released-repaired"))
      and f"under our chat prompt redrawing removes {_gd(_g10['shown']['released-placebo'])} of v1.0's "
          f"{_gm(_g10['shown'], 'released')}" in _flat
      and _g10["shown"]["released-placebo"]["lo"] > 0,
      "sec:open's account of gpt-4o without the data is stale")
check(f"published v1.5 run scored ${load('results/rank_attribution.json')['published']['v1.5|gpt-4o']['margin_numeric']['mean']:+.1f}$ on the same "
      f"${_b15['n_numeric_items']}$ items" in _flat,
      "tab:gptnodata's caption misstates the published run's numeric margin")

# --- what the no-data forced score reads (sec:open, tab:attribution, tab:rewritten) --------
# rank_attribution.py splits BixBench's published forced runs by where the key sits, runs the
# open models' forced arm through the rewritten option sets, and reads which rank the
# question-blind readers pick; every number sec:open types is pinned to it.
import rank_attribution as _ram  # noqa: E402
_rat = load("results/rank_attribution.json")
_rows_ra = _ram.table_rows(_rat)
_att_body = " ".join(_table_body("tab:attribution").split())
_rw_body = " ".join(_table_body("tab:rewritten").split())
_split = _rows_ra.index("")
for _row in _rows_ra[:_split]:
    check(" ".join(_row.split()) in _att_body, f"tab:attribution should carry the row {_row}")
for _row in _rows_ra[_split + 1:]:
    check(" ".join(_row.split()) in _rw_body, f"tab:rewritten should carry the row {_row}")
_pg, _pc15 = _rat["published"]["v1.5|gpt-4o"], _rat["published"]["v1.5|claude-3-5-sonnet-latest"]
_pv0 = [_rat["published"]["v1.0|gpt-4o"], _rat["published"]["v1.0|claude-3-5-sonnet-latest"]]
_kr = [_pg["accuracy_by_key_rank"][str(j)]["n"] for j in range(4)]
_rr = lambda e: e["rank_reader_would_score"]["difference"]
check(_pg["n_numeric"] == 105 and _pg["n_other"] == 100 and _pv0[0]["n_numeric"] == 159 and _pv0[0]["n_other"] == 137
      and f"($105$ on v1.5, $159$ on v1.0), and on the others ($100$, $137$)" in _flat
      and f"(v1.5: ${_kr[0]}$, ${_kr[1]}$, ${_kr[2]}$ and ${_kr[3]}$ items)" in _flat
      and f"would show ${_rr(_pg):+.1f}$ (gpt-4o) and ${_rr(_pc15):+.1f}$ (Claude 3.5 Sonnet)" in _flat
      and all("rank_reader_would_score" not in e for e in _pv0),
      "tab:attribution's caption miscounts items or misquotes the rank reader")
_pk = lambda e: "/".join(f"{x:.0f}" for x in e["pick_rank_shares"])
check(f"the selections fall on the four ranks at ${_pk(_pv0[0])}\\%$ (gpt-4o) and ${_pk(_pv0[1])}\\%$ (Claude)" in _flat,
      "tab:attribution's caption misquotes v1.0's pick ranks")
_d = lambda e: e["second_smallest_minus_rest"]
check(_d(_pg)["mean"] < 0 and _d(_pc15)["mean"] < 0
      and f"the rank would be correct far more often where the key is second-smallest (${_rr(_pg):.1f}$ and "
          f"${_rr(_pc15):.1f}$ points more often," in _flat
      and f"The two models are ${-_d(_pg)['mean']:.1f}$ and ${-_d(_pc15)['mean']:.1f}$ points \\emph{{less}} accurate"
          in _flat
      and f"as far above chance on the ${_pg['n_other']}$ non-numeric questions (${_pg['margin_other']['mean']:+.1f}$ "
          f"and ${_pc15['margin_other']['mean']:+.1f}$) as on the numeric ones (${_pg['margin_numeric']['mean']:+.1f}$ "
          f"and ${_pc15['margin_numeric']['mean']:+.1f}$)" in _flat,
      "sec:open's rank test on the published runs is stale")
_osh = _rat["open_question_shown"]
_parse = [m for m, o in _osh.items() if all(o[a]["unparsed"] <= _ram.UNPARSED_LIMIT for a in ("released", "placebo", "repaired"))]
_rr_d = [_osh[m]["released-repaired"] for m in _parse]
_rw_caps = {o[d]["n_clusters"] for o in _osh.values() for d in ("released-placebo", "released-repaired")}
check(len(_rw_caps) == 1 and f"over the same items and letter orderings, at ${100 * bw.LEVEL:.1f}\\%$ over "
      f"${_rw_caps.pop()}$ capsules. A cell with more than ${_ram.UNPARSED_LIMIT:g}\\%$ unparsed runs reports" in _flat,
      "tab:rewritten's caption misstates its level, capsules or unparsed cap")
_q14 = _osh["Qwen2.5-14B"]
# tab:rewritten: the five models that parse keep their margins through the rank-uniform rewrite
check(len(_osh) == 6 and len(_parse) == 5 and not any(x["lo"] > 0 or x["hi"] < 0 for x in _rr_d),
      "tab:rewritten no longer shows the five parsed models keeping their margins under the rank-uniform rewrite")
_ws = _rat["withheld_summary"]
_num13 = {9: "nine", 10: "ten", 8: "eight", 7: "seven"}
check(_ws["n_models"] == 13 and f"none selects the second-smallest option on more than "
      f"${_ws['max_second_smallest_pick_share']:.1f}\\%$ of items without the question" in _flat,
      "sec:open's question-blind pick-rank sentence is stale")
_w13 = {1: "one", 2: "two", 3: "three", 4: "four", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
check(f"were shown BixBench v1.5's ${_n_bix}$ items with the question withheld" in _flat
      and f"On the ${_b15['n_numeric_items']}$ numeric items, three letter orderings each, the rank a model selects most "
          f"often is the smallest for {_w13[_ws['n_favouring_smallest']]} of the thirteen and the second-smallest for "
          f"{_w13[_ws['n_favouring_second_smallest']]}, and none selects the second-smallest on more than "
          f"${_ws['max_second_smallest_pick_share']:.1f}\\%$ of its runs, where the key lies on ${_p15[1]:.1f}\\%$ of "
          f"the items" in _flat
      and all(v["n"] == 3 * _b15["n_numeric_items"] for v in _rat["open_question_withheld"].values()),
      "app:reader's account of the thirteen models without the question is stale")
# the Terms paragraph: the range of covering levels printed on BixBench
_lv = [e[k]["level"]["level"] for e in _rat["published"].values() for k in ("margin_numeric", "margin_other")]
_lv += [e["second_smallest_minus_rest"]["level"]["level"] for e in _rat["published"].values()]
_lv += [_brk["levels"]["whole file"], _brk["levels"]["moved to an edge"]["level"], _brk["levels"]["not moved"]["level"]]
_lv += [p["level_covering_95"] for c in load("results/claim_budget.json")["claims"] for p in c["parts"] if c["claim"] in
        ("rank rule", "published, gpt-4o", "published, claude")]
_lv += [_brk["levels"][g]["level"] for g in ("moved inward", "rank class kept")]
# the registered replication's intervals, every group and run set, on either release
_rpl_all = load("results/replication.json")
for _res in [_rpl_all["D1"]] + [v for k, v in _rpl_all.get("D2", {}).items() if k != "run_to_run"]:
    _lv += [v["level"] for k, v in _res.items() if k.startswith(("gain|", "repaired-placebo|")) and v]
    _lv += [v["level"] for v in _res["per_run"].values() if v]
check("coverage on the group's own capsules, and none is given where no level does (Appendix~\\ref{app:stats})" in _flat
      and 0.95 <= min(_lv) and max(_lv) < 1.0,
      f"the Terms paragraph no longer says a capped interval is not printed, or a level is off: "
      f"{100 * min(_lv):.2f} to {100 * max(_lv):.2f}")
# the survey's folds
check(all(b["cross_validated"]["n_folds"] == (4 if b["benchmark"] == "LAB-Bench SeqQA" else 10)
          for b in _sv.values() if b.get("cross_validated"))
      and "SeqQA's four subtasks make four folds" in _flat,
      "app:stats misstates the survey's held-out folds")
# sec:conclusion: the redraw-against-the-key loop's worth, from tab:distinctness
_heavy = max(100 * _t["gamma_reject_against_key"] for _t in _thy["distinctness_trap"])
# numbers stated twice, pinned where each one sits
_lvm, _lvi, _lvk = (_brk["levels"][g] for g in ("moved to an edge", "moved inward", "rank class kept"))
check(f"on the moved keys (interval over ${_lvm['n_clusters']}$ capsules at ${_lvs(_lvm['level'])}\\%$), on the "
      f"${_lvi['n_items']}$ inward keys (${_lvi['n_clusters']}$ capsules, "
      f"${_lvs(_lvi['level'])}\\%$) and on the ${_lvk['n_items']}$ unchanged keys "
      f"(${_lvk['n_clusters']}$ capsules, ${_lvs(_lvk['level'])}\\%$)" in _flat
      and f"What $U$ accepts on the ${_lvm['n_items']}$ moved keys, per configuration" in _flat
      and _lvm["n_items"] + _lvi["n_items"] + _lvk["n_items"] == 105,
      "tab:mechanism's caption misstates its groups, capsules or levels")
# app:stats's Intervals paragraph: the double bootstrap's grid, the levels no grid point reaches, and the exact zero
_lv_all = []
for _lvf in Path("results").glob("*.json"):
    def _lvw(x):
        if isinstance(x, dict):
            if isinstance(x.get("level"), float) and "level_capped" in x:
                _lv_all.append((x["level"], bool(x["level_capped"]), bool(x.get("level_floored"))))
            for _v in x.values():
                _lvw(_v)
        elif isinstance(x, list):
            for _v in x:
                _lvw(_v)
    _lvw(json.loads(_lvf.read_text()))
check(abs(float(_cbm.LEVELS[0]) - 0.80) < 1e-12 and abs(float(_cbm.LEVELS[-1]) - 0.99995) < 1e-9
      and "the level is the lowest, on a grid from $80$ to $99.995\\%$, at which the resamples' own intervals hold the "
          "full sample's estimate $95\\%$ of the time" in _flat
      and all(float(_cbm.LEVELS[0]) - 1e-9 <= l <= float(_cbm.LEVELS[-1]) + 1e-9 for l, c, f in _lv_all)
      and any(c for l, c, f in _lv_all)
      and "Kendall's $\\tau$ over seven configurations takes few values, so its level is never set below $95\\%$" in _flat
      and "that end of the interval is set to zero whenever the chance of such a resample exceeds the tail" in _flat,
      "app:stats's Intervals paragraph misstates the double bootstrap's grid, the lowest level, or the zero rule")

# --- the registered replication, the writer panel and the tolerance ---------
# PREREGISTRATION.md's plan is the text pushed before any of its analyses ran
# (6,631 bytes); results and deviations are appended below its line. The plan
# must still be a byte-identical prefix of the file.
_prereg = Path("PREREGISTRATION.md").read_bytes()
check(len(_prereg) > 6631 and hashlib.sha256(_prereg[:6631]).hexdigest()
      == "cedf2e706f55232d66ab513af1ad974c20391a51019c2e4e8dbd15878dc4a62d",
      "PREREGISTRATION.md's registered plan has changed; results belong below its line")
import replication as _rep  # noqa: E402
import writer_panel as _wpn  # noqa: E402
import tolerance_check as _tck  # noqa: E402
_rpl = load("results/replication.json")
_d1 = _rpl["D1"]
# the six hypotheses the plan fixes (five with the data, H6 without it, for the new runs), and its date: the day the
# first of its analyses was run, which its results section records
_d1_day = re.search(r"### D1, analysed (\d+) Sep 2026", _prereg.decode()).group(1)
# the archived push record (provenance/github_push_activity.json, GitHub's own record of pushes to the private
# repository): the push that added the plan, and the next, which added the first results
import datetime as _dt
_pushes = {e["after"]: _dt.datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
           for e in load("provenance/github_push_activity.json") if e.get("activity_type") == "push"}
_plan_sha = Path("provenance/plan_commit.txt").read_text().split()[0]
_push_plan = _pushes[_plan_sha]
_first_results = next(e for e in load("provenance/github_push_activity.json") if e.get("before") == _plan_sha)
_push_gap = round((_dt.datetime.fromisoformat(_first_results["timestamp"].replace("Z", "+00:00")) - _push_plan)
                  .total_seconds() / 60)
check(_push_gap == 15 and "Replication D1" in Path("provenance/plan_commit.txt").read_text(),
      "the archived push record no longer puts the plan's push 15 minutes before the first results'")
check(len(_rep.PREDICTIONS) == 6 and "**H6**" in _prereg[:6631].decode()
      and "we fixed six hypotheses about the nearest-option rule's contrasts, each with a pass criterion, in a dated "
          "plan that was not deposited publicly (Appendix~\\ref{app:replication}, Table~\\ref{tab:predictions})" in _flat
      and f"The plan is dated {_d1_day} September 2026, before the published runs were analysed and before the new runs "
          f"were made. It was committed to our version history and pushed to GitHub, whose push record, archived with "
          f"the code, dates the push at {_push_plan:%H:%M} UTC, ${_push_gap}$ minutes before that of the first results"
          in _flat
      and "it was not deposited in a public registry, so a reader of this paper cannot check its precedence independently"
          in _flat
      and "the plan was not deposited in a public registry, so the test's precedence rests on GitHub's push record, "
          "archived with the code (Appendix~\\ref{app:replication})" in _flat,
      "the paper misstates the hypotheses the plan fixes, its date, or the push record")
check(all(_d1["verdicts"].values()) and sorted(_d1["verdicts"]) == ["H1", "H2", "H3", "H4", "H5"],
      f"sec:withdata says all five registered predictions hold on v1.0; they are {_d1['verdicts']}")
_repb = " ".join(_table_body("tab:replication").split())
for _row in _rep.table_rows(_rpl):
    check(" ".join(_row.split()) in _repb, f"tab:replication should carry the row {_row}")
_h1, _kpt, _inw, _plc = (_d1[k] for k in ("gain|moved to an edge", "gain|kept", "gain|moved inward",
                                          "repaired-placebo|moved to an edge"))
_ncr = _d1["newly_credited"]
_ivl = lambda v: f"${v['mean']:+.1f}$ $[{v['lo']:+.1f},{v['hi']:+.1f}]$"
_prun = [_d1["per_run"][r] for r in _rep.OPEN_RUNS]
_ntr = f"{sum(_d1['n_numeric_runs'].values()):,}".replace(",", "{,}")
_x1 = _rep.load_extract()
_nq = [n for n in collections.Counter((r, q) for r in _rep.OPEN_RUNS for q, _, _ in _x1["open_runs"][r]
                                      if q in _rep.v10_sets()).values()]
check(f"released with the benchmark, are ${_ntr}$ runs of gpt-4o and Claude 3.5 Sonnet on v1.0's "
      f"${_d1['n_numeric_items']}$ numeric questions, each graded by the model that made it" in _flat
      and sum(_d1["n_numeric_runs"].values()) == 5161 and sum(_d1["n_trajectories"].values()) == 9284
      and min(_nq) == 1 and max(_nq) == 10 and len(_rep.OPEN_RUNS) == 4,
      "sec:withdata misstates the published runs it registered")
_r1all = load("results/published_reads_readers.json")["rule"]["repaired-placebo|all"]["mean"]
_unk = load("results/miss_anatomy.json")["unchanged_keys"]["v1.0, published"]
_flip = _unk["parts"]["the other extreme"]
_q235 = _rpl["D2"]["qwen3-235b|data"]
check(f"The moved keys gain {_ivl(_h1)} points under $U$ against $R$ on the published runs and "
      f"{_ivl(_rpl['D2']['reruns|data']['gain|moved to an edge'])} on the new seeds; for Qwen3-235B-A22B the "
      f"moved keys' {_ivl(_q235['gain|moved to an edge'])} reaches zero and H1 fails." in _flat
      and _q235["gain|moved to an edge"]["lo"] == 0.0 and not _q235["verdicts"]["H1"]
      and f"On the published runs the unchanged keys' {_ivl(_kpt)} excludes zero, and ${_flip['contribution']:.1f}$ points "
          f"of it come from ${_flip['n_items']}$ keys that $U$ moves from one extreme to the other; we count H2 as failed "
          f"there, although its stated criterion, which also accepts a mean within $5$ points of zero, passes it" in _flat
      and f"its unchanged keys gain {_ivl(_q235['gain|kept'])} and its moved keys {_ivl(_q235['repaired-placebo|moved to an edge'])} "
          f"against $P$" in _flat
      and abs(_unk["mean"] - _kpt["mean"]) < 0.05 and _flip["contribution"] > _unk["mean"] / 2
      and all(v["lo"] > 0 for v in _prun) and _inw["hi"] < 0 and _kpt["lo"] > 0 and abs(_kpt["mean"]) < 5
      and _kpt["hi"] > 5 and _rpl["D2"]["reruns|data"]["gain|kept"]["hi"] > 5,
      "sec:cost's account of the pre-specified test is stale")
# tab:predictions: each prediction's registered rule and every data set's verdict under it, from the results
_pdb = " ".join(_table_body("tab:predictions").split())
for _row in _rep.prediction_rows(_rpl):
    check(" ".join(_row.split()) in _pdb, f"tab:predictions should carry the row {_row}")
_plan = " ".join(_prereg[:6631].decode().split())
check("an interval that contains 0, or a mean within 5 points of 0" in _plan
      and "the interval's lower end is above 0" in _plan and "has a mean of at most 0" in _plan
      and "more than half are more than 25% from the key" in _plan,
      "tab:predictions quotes a pass rule the registered plan does not contain")
_srs = _d1["single_run_spread"]
_ndraw = f"{_srs['draws']:,}".replace(",", "{,}")
check(f"A single run per question reproduces the published runs' gain (${_srs['central_95'][0]:.1f}$ to "
      f"${_srs['central_95'][1]:.1f}$ points in $95\\%$ of ${_ndraw}$ draws)" in _flat and _srs["central_95"][0] > 0,
      "sec:withdata's single-run spread is stale")
# the new v1.5 runs, registered before they were made (PREREGISTRATION.md's D2): the verdicts the
# abstract, the introduction and sec:withdata state, every number of sec:withdata's paragraph on
# them, tab:replication's caption and App G's account of how they ran
_d2 = _rpl["D2"]
_nx = _rpl["not_registered"]
_q3, _rr, _q3n, _rrn = (_d2[k] for k in ("qwen3-235b|data", "reruns|data", "qwen3-235b|nodata", "reruns|nodata"))
_new_names = {"qwen72b": "Qwen2.5-72B", "llama70b": "Llama-3.3-70B", "gemma27b": "gemma-3-27b"}
check(sorted(_q3["n_trajectories"]) == ["qwen3-235b|r0", "qwen3-235b|r1"]
      and sorted(_rr["n_trajectories"]) == sorted(f"{m}|r{r}" for m in _new_names for r in (1, 2))
      and set(_q3["n_trajectories"].values()) | set(_rr["n_trajectories"].values()) == {205}
      and "The v1.5 runs were made after the hypotheses were fixed: two runs per question of Qwen3-235B-A22B, and "
          "two new seeds of each text-protocol agent." in _flat,
      "tab:replication's caption misstates the new runs, or a new run set is short of its 205 questions")
check(all(_rr["verdicts"].values()) and len(_rr["verdicts"]) == 5
      and sorted(k for k, v in _q3["verdicts"].items() if v) == ["H3", "H5"]
      and _q3n["verdicts"] == {"H6": True} and _rrn["verdicts"] == {"H6": True}
      and _q3["gain|moved inward"].get("level_capped")
      and "so these test only magnitude; they pass on the published runs and the new seeds" in _flat
      and "H4 holds on the published runs and the new seeds" in _flat
      and "reaches zero and H1 fails" in _flat and "fails both: its unchanged keys gain" in _flat
      and "No nominal level attains $95\\%$ coverage on these" in _flat,
      f"the new runs' registered verdicts are not the ones the paper states: {_q3['verdicts']} {_rr['verdicts']} "
      f"{_q3n['verdicts']} {_rrn['verdicts']}")
_rtr = _d2["run_to_run"]
_sds = {m: [_rtr[f"{m}|data"][k] for k in ("r0", "r1", "r2")] for m in _new_names}
_wide = max(_sds, key=lambda m: max(_sds[m]) - min(_sds[m]))
_w5 = {k: v["mean"] for k, v in _nx["within_5pct"].items()}
_w5new = [_w5[f"{m}|r{r}|data"] for m in _new_names for r in (1, 2)]
_plk = _nx["placebo-released"]["qwen3-235b|data|kept"]
check(all(_rr["verdicts"].values()) and _rr["gain|kept"]["lo"] <= 0 <= _rr["gain|kept"]["hi"]
      and all(v > 0 for m in _sds for v in _sds[m]),
      "the new seeds no longer pass every hypothesis, as tab:predictions and sec:cost say")
check(_q3n["gain|moved to an edge"]["lo"] > 0 and _rrn["gain|moved to an edge"]["lo"] > 0
      and _w5["qwen3-235b|r0|data"] == _w5["qwen3-235b|r1|data"] and _plk["lo"] > 0
      and _q3["repaired-placebo|moved to an edge"]["lo"] < 0 < _q3["gain|kept"]["lo"]
      and f"whose number is within $5\\%$ of the key on ${_w5['qwen3-235b|r0|data']:.1f}\\%$ of numeric questions, fails both"
          in _flat,
      "Qwen3-235B-A22B's verdicts, or H6 without the data, no longer hold as tab:predictions and sec:cost give them")
# PREREGISTRATION.md's D2 results, family by family, say what results/replication.json says: the 25 Sep entry, or the
# 1 Oct correction's line where it restates one (the entries above a correction are left as they were written)
_prd = _prereg.decode().split("### D2, analysed", 1)
_segs = _prd[1].split("**The reruns, six run sets pooled") if len(_prd) == 2 else []
_corr = _prereg.decode().split("### Corrections, analysed 1 Oct 2026", 1)
_corr = _corr[1] if len(_corr) == 2 else ""
_pv = lambda v: f"{v['mean']:+.1f} [{v['lo']:+.1f},{v['hi']:+.1f}]"
_fname = {"qwen3-235b": "Qwen3-235B-A22B", "reruns": "the reruns"}


def _stated(fam, seg, h):
    """The plan's last word on hypothesis ``h`` for a D2 family, without its leading '- '."""
    tag = f"- D2, {_fname[fam]}, "
    for _line in _corr.splitlines():
        if _line.startswith(f"{tag}{h}, "):
            return _line[len(tag):]
    _m = re.search(rf"^- {h}, .*$", seg, re.M)
    return _m.group(0)[2:] if _m else ""


check(len(_segs) == 2 and all(
      _stated(fam, seg, h).startswith(f"{h}, {'pass' if _d2[f'{fam}|data']['verdicts'][h] else 'FAIL'}")
      for fam, seg in (("qwen3-235b", _segs[0]), ("reruns", _segs[1])) for h in ("H1", "H2", "H3", "H4", "H5"))
      and all(f": {_pv(_d2[f'{fam}|data']['gain|moved to an edge'])}" in _stated(fam, seg, "H1")
              and f"kept keys {_pv(_d2[f'{fam}|data']['gain|kept'])}" in _stated(fam, seg, "H2")
              and f"moved keys {_pv(_d2[f'{fam}|data']['repaired-placebo|moved to an edge'])}" in _stated(fam, seg, "H4")
              and f"- H6, pass: without the data, moved keys {_pv(_d2[f'{fam}|nodata']['gain|moved to an edge'])}" in seg
              for fam, seg in (("qwen3-235b", _segs[0]), ("reruns", _segs[1])))
      and f"- D1, H2, pass on the 5-point margin: kept keys {_pv(_d1['gain|kept'])}" in _corr
      and f"- D1, H5, pass: keys moved inward {_pv(_d1['gain|moved inward'])}" in _corr,
      "PREREGISTRATION.md's D2 results do not match results/replication.json")
# the only interval in tab:replication whose level reached the grid's top, said where it sits
_capped = [(k, g) for k, res in [("D1", _d1)] + [(k, v) for k, v in _d2.items() if k.endswith("|data")]
           for g, v in res.items() if g.startswith(("gain|", "repaired-placebo|")) and v and v["level_capped"]]
_cpv = _q3["gain|moved inward"]
check(_capped == [("qwen3-235b|data", "gain|moved inward")] and (_cpv["n_items"], _cpv["n_clusters"]) == (12, 10)
      and "$^{\\ddagger}$No nominal level attains $95\\%$ coverage on these $12$ keys in $10$ capsules, so no interval "
          "is given." in _flat,
      f"the capped intervals in tab:replication are not the one the Terms paragraph and the caption name: {_capped}")
# App G: how the new runs ran, from the answers and the rerun records that ship
_pk = Path("results/agent_runs_replication")
_nq15 = sum(1 for l in open("data/bixbench.jsonl") if l.strip())
_npk = sum(1 for f in sorted(_pk.glob("answers_*.jsonl.gz")) for _ in gzip.open(f, "rt"))
_cut = {f.stem.removeprefix("reruns_"): len(json.loads(f.read_text())["episodes"]) for f in sorted(_pk.glob("reruns_*.json"))}
_sd = lambda m, c: "{}, {} and {}".format(*(f"${_rtr[f'{m}|{c}'][k]:+.1f}$" for k in ("r0", "r1", "r2")))
_q3r = lambda c: f"${_rtr[f'qwen3-235b|{c}']['r0']:+.1f}$ and ${_rtr[f'qwen3-235b|{c}']['r1']:+.1f}$"
_plm = _nx["placebo-released"]
check(f"gemma-3-27b: ${_npk:,}$ episodes, served in FP8 with vLLM".replace(",", "{,}", 1) in _flat
      and _npk == 2 * 2 * 205 * 4
      and f"the clock cut ${_cut['qwen3-235b']}$ and ${_cut['qwen72b']}$ episodes, which were run again from the start." in _flat and sorted(_cut) == ["qwen3-235b", "qwen72b"]
      and all(sum(1 for _ in gzip.open(_pk / f"superseded_{m}.jsonl.gz", "rt")) == n for m, n in _cut.items())
      and "the gain with the data is " + "; ".join(f"{_sd(m, 'data')} for {n}" for m, n in _new_names.items())
          + f"; and {_q3r('data')} for Qwen3-235B-A22B's two runs. Without the data it is "
          + "; ".join(_sd(m, "nodata") for m in _new_names) + f"; and {_q3r('nodata')}." in _flat
      and f"on the ${_nq15}$ questions with and without the data" in _flat
      and set(_q3["n_trajectories"].values()) | set(_rr["n_trajectories"].values()) == {_nq15}
      and f"Qwen3-235B-A22B's number is within $5\\%$ of the key on ${_w5['qwen3-235b|r0|data']:.1f}\\%$ of numeric "
          f"questions in each run with the data" in _flat
      and f"(${100 * _nx['level']:.0f}\\%$ cluster intervals)." in _flat and _nx["level"] == 0.95
      and f"and ${_w5['qwen3-235b|r0|nodata']:.1f}$ and ${_w5['qwen3-235b|r1|nodata']:.1f}\\%$ without; with the data "
          f"$P$ raises its unchanged keys by {_ivl(_plk)} and its moved keys by "
          f"{_ivl(_plm['qwen3-235b|data|moved to an edge'])}, and without the data by "
          f"${_plm['qwen3-235b|nodata|kept']['mean']:+.1f}$ and ${_plm['qwen3-235b|nodata|moved to an edge']['mean']:+.1f}$; "
          f"for the new seeds $P$ changes neither by more than "
          f"${max(abs(_plm[f'reruns|{c}|{g}']['mean']) for c in ('data', 'nodata') for g in ('kept', 'moved to an edge')):.1f}$ "
          "points" in _flat,
      "app:replication's account of the new runs is stale")
# App G: the published runs and the repair built on them
_a10 = load("results/mcq_audit_bixbench_v10.json")
_sizes = [f"${_d1['n_trajectories'][r]:,}$".replace(",", "{,}") for r in _rep.OPEN_RUNS]
_rdr = _d1["published_reader_check"]
check(f"hold {_sizes[0]}, {_sizes[1]}, {_sizes[2]} and {_sizes[3]} runs of the ${len(_x1['items'])}$ questions, "
      f"of which ${_d1['n_numeric_items']}$ have four numeric options" in _flat
      and f"$U$ redrew all ${_a10['n_repaired']}$ option sets "
          f"(${_a10['n_rank_reassigned']}$ could not take the rank drawn for them and took another); $P$ "
          f"redrew ${_d1['n_numeric_items'] - _d1['n_without_placebo']}$." in _flat
      and _a10["n_repaired"] == _d1["n_numeric_items"] and _a10["seed"] == 20260918
      and "On the moved keys the four configurations gain " + ", ".join(f"${v['mean']:+.1f}$" for v in _prun[:3])
          + f" and ${_prun[3]['mean']:+.1f}$, each interval above zero." in _flat
      and f"select the key on ${min(v['published_reader_right'] for v in _rdr.values()):.1f}$ to "
          f"${max(v['published_reader_right'] for v in _rdr.values()):.1f}\\%$ of numeric runs through $R$, "
          f"against ${min(v['nearest_option_right'] for v in _rdr.values()):.1f}$ to "
          f"${max(v['nearest_option_right'] for v in _rdr.values()):.1f}\\%$ for the nearest option" in _flat,
      "app:replication's account of the published runs or the repair is stale")
# the tolerance: sec:conclusion, app:replication, and their two tables
_tol = load("results/tolerance_check.json")
_t1, _g10, _g15, _t3 = _tol["T1"], _tol["T2_v10_published_grader"], _tol["T2_v15_graders"], _tol["T3"]
_n61 = sum(json.loads(l).get("eval_mode") == "range_verifier" for l in open("data/bixbench.jsonl"))
_k5 = _g10["rules"]["within 5%"]["kappa"]
_ng10 = f"{_g10['n']:,}".replace(",", "{,}")
_r5 = _g10["rules"]["within 5%"]
_dis = _r5["grader_yes_rule_no"] + _r5["grader_no_rule_yes"]
_k15 = {t: _g15["rules"][f"within {t}%"]["kappa"] for t in (1, 2, 5)}
check(f"the ${_t1['n_v15']}$ keys its authors wrote as ranges have a median half-width of ${100 * _t1['median']:.1f}\\%$, "
      f"and over ${_ng10}$ of gpt-4o's and Claude 3.5 Sonnet's published answers the v1.0 grader and a $5\\%$ rule "
      f"disagree on ${_dis}$ (${100 * _dis / _g10['n']:.1f}\\%$; $\\kappa={_k5:.2f}$, the highest of the rules we tried)." in _flat
      and f"own open-ended grader on ${100 * _dis / _g10['n']:.1f}\\%$ of verdicts (\\S\\ref{{sec:conclusion}})" in _flat
      and f"The v1.0 grader and the $5\\%$ rule disagree on ${_dis}$ of the ${_ng10}$ answers, "
          f"${100 * _dis / _g10['n']:.1f}\\%$: the grader rejects ${_r5['grader_no_rule_yes']}$ within $5\\%$ of the key "
          f"and accepts ${_r5['grader_yes_rule_no']}$ beyond it" in _flat
      and f"against the v1.5 graders', $1$ and $2\\%$ agree better ($\\kappa={_k15[1]:.2f}$) than $5\\%$ "
          f"(${_k15[5]:.2f}$)" in _flat
      and _k5 == max(r["kappa"] for r in _g10["rules"].values())
      and round(_k15[1], 2) == round(_k15[2], 2) == round(max(r["kappa"] for r in _g15["rules"].values()), 2) > round(_k15[5], 2),
      "sec:conclusion's tolerance evidence is stale")
_wrong = _g10["n"] - _g10["grader_accepts"]
check(_k5 == max(r["kappa"] for r in _g10["rules"].values())
      and f"Against the v1.0 grader's verdicts, a rule that accepts a number within $5\\%$ agrees better than any other we "
      f"tried; against the v1.5 graders', $1$ and $2\\%$ agree better ($\\kappa={_k15[1]:.2f}$) than $5\\%$ "
      f"(${_k15[5]:.2f}$)" in _flat
      and f"The v1.0 grader and the $5\\%$ rule disagree on ${_dis}$ of the ${_ng10}$ answers, ${100 * _dis / _g10['n']:.1f}\\%$: "
          f"the grader rejects ${_r5['grader_no_rule_yes']}$ within $5\\%$ of the key and accepts ${_r5['grader_yes_rule_no']}$ beyond "
          f"it, and of the ${f'{_wrong:,}'.replace(',', '{,}')}$ answers it judges wrong, "
          f"${100 * _r5['grader_no_rule_yes'] / _wrong:.1f}\\%$ are within $5\\%$" in _flat,
      "app:replication's account of the tolerance's disagreements with the graders is stale")
_acc = {k: v["rate"] for k, v in _g10["acceptance_by_distance"].items()}
check(f"The ${_n61}$ keys that BixBench v1.5 writes as ranges are its authors' own tolerance for those quantities; the "
      f"${_t1['n_v15']}$ with a relative width (one straddles zero) have half-widths with median "
      f"${100 * _t1['median']:.1f}\\%$ and quartiles ${100 * _t1['quartiles'][0]:.1f}$ and "
      f"${100 * _t1['quartiles'][1]:.1f}\\%$, and ${100 * _t1['share_within_5pct']:.0f}\\%$ are $5\\%$ or narrower." in _flat
      and _n61 == _t1["n_v15"] + 1
      and f"The v1.0 grader accepts ${100 * _acc['0-0.01']:.0f}\\%$ of answers within $1\\%$ of the key, "
          f"${100 * _acc['0.01-0.02']:.0f}\\%$ of those $1$ to $2\\%$ off, ${100 * _acc['0.02-0.05']:.0f}\\%$ at $2$ to "
          f"$5\\%$, ${100 * _acc['0.05-0.1']:.0f}\\%$ at $5$ to $10\\%$ and ${100 * _acc['0.1-0.2']:.0f}\\%$ at $10$ to "
          f"$20\\%$" in _flat and _acc["0.01-0.02"] > 0.5 and _acc["0.1-0.2"] < 0.05,
      "app:replication's tolerance paragraph is stale")
_w1, _w10 = _t3["withdata"]["1%"], _t3["withdata"]["10%"]
_w2 = _t3["withdata"]["2%"]
check("The lead over gemma-3-27b holds at $1$ and $10\\%$; the lead over Llama-3.3-70B reaches zero at $1$ and $2\\%$ and "
      "is not significant at $10\\%$" in _flat
      and "Qwen2.5-72B's lead over Llama-3.3-70B, significant at $5\\%$, reaches zero at $1$ and $2\\%$ and includes it at "
          "$10\\%$." in _flat
      and _w1["qwen72b - gemma27b"]["lo"] > 0 and _w10["qwen72b - gemma27b"]["lo"] > 0
      and _w10["qwen72b - llama70b"]["lo"] < 0 and _w1["qwen72b - llama70b"]["lo"] == 0 == _w2["qwen72b - llama70b"]["lo"]
      and _t3["withdata"]["5%"]["qwen72b - llama70b"]["lo"] > 0,
      "sec:withdata's tolerance sentence no longer matches tolerance_check.json")
_rr3a = _t3["released_reading"]
_trows = _tck.table_rows(_tol)
_split = _trows.index("")
for _label, _rows in (("tab:tolerance", _trows[:_split]), ("tab:restated", _trows[_split + 1:])):
    _tb = " ".join(_table_body(_label).split())
    for _row in _rows:
        check(" ".join(_row.split()) in _tb, f"{_label} should carry the row {_row}")
check(f"$5{{,}}161$ with-data answers to numeric questions" in _flat and _g10["n"] == 5161
      and f"configurations' ${_g15['n']}$ with-data answers to numeric questions" in _flat,
      "tab:tolerance's caption misstates its answer counts")

# the three captions' levels and counts that only a duplicate elsewhere would satisfy
_v10_sv = {b["benchmark"]: b for b in load("results/channel_survey.json")["benchmarks"]}["BixBench v1.0"]
check(f"of which ${_d1['n_numeric_items']}$ have four numeric options, the same ${_v10_sv['n_numeric_items']}$ option "
      f"sets that the survey uses" in _flat and _v10_sv["n_numeric_items"] == _d1["n_numeric_items"],
      "app:replication's count of v1.0's numeric option sets is stale")
_rq = _t3["withdata"]["5%"]["qwen72b - llama70b"]
check(f"; intervals paired over ${_rq['n_clusters']}$ capsules at ${100 * bw.LEVEL:.0f}\\%$), the best of" in _flat,
      "tab:restated's caption misstates its capsules or level")

# the repair redrawn: sec:withdata's range over 100 seeds, and the first seed is the paper's draw
_rsd = load("results/repair_seeds.json")
_rsm = _rsd["summary"]
_nmv = [r["n_moved"] for k in ("v1.5|data", "v1.0|published") for r in _rsd[k]]
check(f"Redrawn with ${_rsm['v1.5|data']['n_seeds']}$ seeds by the same procedure, $U$ moves ${min(_nmv)}$ to "
      f"${max(_nmv)}$ keys to an extreme, and the gain on them ranges from ${_rsm['v1.5|data']['min']:+.1f}$ to "
      f"${_rsm['v1.5|data']['max']:+.1f}$ on v1.5 with the data, ${_rsm['v1.5|nodata']['min']:+.1f}$ to "
      f"${_rsm['v1.5|nodata']['max']:+.1f}$ without it and ${_rsm['v1.0|published']['min']:+.1f}$ to "
      f"${_rsm['v1.0|published']['max']:+.1f}$ on the published v1.0 runs, with its $95\\%$ interval excluding zero "
      f"at every seed" in _flat
      and all(v["n_lower_end_above_zero"] == v["n_seeds"] for v in _rsm.values()),
      "sec:withdata's repair-seed range is stale")
_rs4 = load("results/reader_split.json")
check(_rsd["seeds"][0] == 20260918
      and abs(_rsd["v1.5|data"][0]["gain"]["mean"] - _rs4["data|nearest|repaired-released|moved to an edge"]["pooled"]["mean"]) < 0.01
      and abs(_rsd["v1.5|nodata"][0]["gain"]["mean"] - _rs4["nodata|nearest|repaired-released|moved to an edge"]["pooled"]["mean"]) < 0.01
      and abs(_rsd["v1.0|published"][0]["gain"]["mean"] - _d1["gain|moved to an edge"]["mean"]) < 0.01,
      "repair_seeds.py's first seed no longer reproduces the paper's and the replication's draws")

# --- where v1.5's rank came from (sec:theory, app:stats' tab:origin) ------------------
import leak_origin as _lom  # noqa: E402
_lor = load("results/leak_origin.json")
_lob = " ".join(_table_body("tab:origin").split())
for _row in _lom.table_rows(_lor):
    check(" ".join(_row.split()) in _lob, f"tab:origin should carry the row {_row}")
_lp, _lc = _lor["parts"], _lor["counts"]
_rw_after = _lp["carried over with rewritten distractors, after"]
_rw_before = _lp["carried over with rewritten distractors, before"]
_n_rw2 = round(_rw_after["rank_shares"][1] * _rw_after["n"] / 100)
_rat3 = _lor["rewritten_second_smallest_ratios"]
check(f"v1.5 keeps ${_lc['carried over']}$ of v1.0's ${_lc['v1.0 numeric']}$ numeric items (same capsule and "
      f"key), mostly bracketed ones "
      f"(${_lp['v1.0 items v1.5 carried over, as v1.0 wrote them']['bracketed']:.0f}\\%$, against "
      f"${_lp['v1.0 items v1.5 dropped']['bracketed']:.0f}\\%$ of those dropped). In the ${_lc['rewritten']}$ "
      f"whose distractors it rewrote, the key is second-smallest on ${_n_rw2}$ "
      f"(${_rw_after['rank_shares'][1]:.0f}\\%$, up from ${_rw_before['rank_shares'][1]:.0f}\\%$ in v1.0), "
      f"with new distractors typically ${_rat3['median'][0]:.2f}$, ${_rat3['median'][1]:.2f}$ and "
      f"${_rat3['median'][2]:.2f}$ times the key (medians over the ${_rat3['n']}$ positive keys); its "
      f"${_lc['new']}$ new items place the key there on ${_lp['new in v1.5']['rank_shares'][1]:.0f}\\%$." in _flat
      and f"The skew arose in v1.5 (Table~\\ref{{tab:origin}}): in the ${_lc['rewritten']}$ of v1.0's items "
          f"whose distractors it rewrote, the key is second-smallest on ${_n_rw2}$ "
          f"(${_rw_after['rank_shares'][1]:.0f}\\%$, up from ${_rw_before['rank_shares'][1]:.0f}\\%$ in v1.0), "
          f"with new distractors typically ${_rat3['median'][0]:.2f}$, ${_rat3['median'][1]:.2f}$ and "
          f"${_rat3['median'][2]:.2f}$ times the key; its ${_lc['new']}$ new items place the key there on "
          f"${_lp['new in v1.5']['rank_shares'][1]:.0f}\\%$." in _flat
      and _lp['new in v1.5']['rank_shares'][1] > 25
      and _lc["carried over"] + _lc["new"] == _lc["v1.5 numeric"] == 105
      and _lc["carried over"] + _lc["v1.0 dropped"] == _lc["v1.0 numeric"] == 159,
      "sec:theory's account of where v1.5's rank came from is stale")
_bk15 = _lor["by_kind"]["v1.5"]
_rest = [_bk15[k]["bracketed"] for k in ("count", "percentage or proportion", "other")]
check(f"its p-values are the exception, bracketed on ${_bk15['p-value']['bracketed']:.0f}\\%$ and at the smallest rank on "
      f"${_bk15['p-value']['rank_shares'][0]:.0f}\\%$, where its counts, percentages and other quantities are bracketed on "
      f"${min(_rest):.0f}$ to ${max(_rest):.0f}\\%$" in _flat,
      "app:stats' reading of v1.5's rank by kind of quantity is stale")

# --- who comes first (sec:withdata, app:replication's tab:ranking) -------------------
import ranking_check as _rkm  # noqa: E402
_rk = load("results/ranking_check.json")
_rkb = " ".join(_table_body("tab:ranking").split())
for _row in _rkm.table_rows(_rk):
    check(" ".join(_row.split()) in _rkb, f"tab:ranking should carry the row {_row}")
_tg = _rk["v1.5"]["tau"]["graders"]
_ivt = lambda v: f"${v['tau']:+.2f}$ $[{v['lo']:+.2f},{v['hi']:+.2f}]$"
_gc = _rk["v1.0"]["gpt4o_minus_claude"]
_vt = _rk["v1.5"]["against_tolerance"]
_ow, _nw, _nrp = _vt["own|released"], _vt["nearest|released"], _vt["nearest|repaired"]
_dv = lambda v: f"${v['difference']['mean']:+.2f}$ $[{v['difference']['lo']:+.2f},{v['difference']['hi']:+.2f}]$"
_below = lambda v: f"{100 * (1 - v['noise_matched']['share_at_or_below']):.0f}"
_gw = _vt["gemma|released"]
check("On BixBench's published runs Claude 3.5 Sonnet leads gpt-4o under every grading." in _flat
      and f"the tolerance preserves their open-answer ranking better than each configuration's own MCQ grader, "
          f"by ${_ow['difference']['mean']:+.2f}$ in Kendall's $\\tau$, and better than gemma-3-27b as one "
          f"grader for all, by ${_gw['difference']['mean']:+.2f}$, but both intervals reach zero" in _flat
      and _ow["difference"]["lo"] < 0 < _ow["difference"]["hi"] and _gw["difference"]["lo"] < 0 < _gw["difference"]["hi"]
      and _ow["noise_matched"]["share_at_or_below"] < 0.05
      and _nw["difference"]["lo"] < 0 < _nw["difference"]["hi"] and abs(_ow["tau"] - _tg["own|released"]["tau"]) < 1e-9,
      "sec:cost's ranking sentence is stale")
check(f"The tolerance's $\\tau$ exceeds that of each configuration's own MCQ grader through $R$ by {_dv(_ow)}, "
      f"an interval that reaches zero, although that grader preserves less of the ranking than the tolerance "
      f"with random errors at the grader's own rates in ${_below(_ow)}\\%$ of simulations. The nearest-option "
      f"rule's ${_nw['tau']:+.2f}$ through $R$ can be distinguished from neither, and through $U$ the rule's "
      f"$\\tau$, ${_nrp['tau']:+.2f}$, is the lowest, though its difference from the tolerance's, {_dv(_nrp)}, "
      f"also reaches zero."
      in _flat
      and _nw["noise_matched"]["share_at_or_below"] > 0.05 and _nrp["difference"]["lo"] < 0 < _nrp["difference"]["hi"]
      and _nrp["tau"] <= min(x["tau"] for x in _tg.values()) and _nrp["noise_matched"]["share_at_or_below"] < 0.05,
      "app:replication's account of who comes first is stale")
check(f"Claude 3.5 Sonnet leads gpt-4o under the benchmark's grader, by ${-_gc['grader']['mean']:.1f}$ "
      f"$[{-_gc['grader']['hi']:.1f},{-_gc['grader']['lo']:.1f}]$ points, under its multiple-choice grades, under "
      f"gemma-3-27b's and under every tolerance, and the nearest-option rule more than doubles the gap, to "
      f"${-_gc['nearest|released']['mean']:.1f}$" in _flat
      and all(f"gemma|{a}" in _gc for a in ("released", "placebo", "repaired"))
      and all(v["hi"] < 0 for v in _gc.values())
      and -_gc["nearest|released"]["mean"] > 2 * -_gc["grader"]["mean"],
      "sec:withdata's account of the published runs' order is stale, or an order flips")

# --- app:replication's checks beside the tests --------------------------------------
_kac2 = load("results/key_audit_check.json")
_moves = [abs(_kac2[k.replace("|all", "|without audited")]["mean"] - _kac2[k]["mean"])
          for k in _kac2 if isinstance(_kac2[k], dict) and k.endswith("|moved|all")]
# every reading the two readers tables print is restated without the audited capsules
_kset = {"D1": "D1", "D0|data": "D0", "D2|reruns|data": "D2 reruns", "D2|qwen3-235b|data": "D2 qwen3-235b"}
_kneed = []
for _rd, _md in (("gemma27b", "forced"), ("qwen72b", "forced"), ("llama70b", "forced"), ("gemma27b", "decline")):
    _krep = load("results/published_reads.json" if (_rd, _md) == ("gemma27b", "forced")
                 else f"results/published_reads_{_rd}_{_md}.json")
    _kflat = {**{k: v for k, v in _krep.items() if k != "D2"}, **{f"D2|{k}": v for k, v in _krep.get("D2", {}).items()}}
    _kneed += [f"reads|{_rd}|{_md}|{_kset[k]}|{c}|moved|{w}" for k in _kflat if k in _kset
               for c in ("repaired-released", "repaired-placebo") for w in ("all", "without audited")]
_kneed += [f"seven|{r}|{c}|moved|{w}" for r in ("nearest", "own", "gemma27b")
           for c in ("repaired-released", "repaired-placebo") for w in ("all", "without audited")]
check(all(k in _kac2 for k in _kneed) and len(_kneed) >= 40
      and f"pilot (${_kac2['n_dropped']['v1.5']}$ of v1.5's $105$ and ${_kac2['n_dropped']['v1.0']}$ of v1.0's "
          f"$159$, the keys it flags and their neighbours alike), changes no moved-key contrast of "
          f"Tables~\\ref{{tab:readers2}} and~\\ref{{tab:readersfull}} or of the pre-specified test by more "
          f"than ${max(_moves):.1f}$ points." in _flat,
      "app:replication's key-audit sentence is stale, or key_audit_check.py misses a reading the tables print: "
      f"{[k for k in _kneed if k not in _kac2][:4]}")
check(f"Dropping the six capsules a concurrent audit examined changes no moved-key contrast of the rule or of "
      f"a grader given the notebook by more than ${max(_moves):.1f}$ points." in _flat
      and not any(k.startswith("reads|") and "codefree" in k for k in _kac2),
      "the limitations misstate how far dropping the audited capsules moves a moved-key contrast")
# the learned option-only reader on the three option sets: held-out accuracy of the every-feature learner,
# and the worst of the ten clean files it is read against (the same clean files for every set)
_lpr = {a: load(f"results/learned_probe_bix_optionsets_{a}.json") for a in ("q", "placebo", "repaired")}
_lpa = {a: 100 * next(iter(d["files"].values()))["every feature"]["cv_accuracy"] for a, d in _lpr.items()}
_lpw = {a: 100 * d["clean"]["every feature"]["max"] for a, d in _lpr.items()}
check(len({f"{v:.1f}" for v in _lpw.values()}) == 1
      and f"selects the key on ${_lpa['q']:.1f}\\%$ of the released items and ${_lpa['placebo']:.1f}\\%$ of $P$'s, "
          f"above the worst of ten control files (${_lpw['q']:.1f}\\%$), and on ${_lpa['repaired']:.1f}\\%$ of "
          f"$U$'s, no higher" in _flat
      and _lpa["q"] > _lpw["q"] and _lpa["placebo"] > _lpw["placebo"] and _lpa["repaired"] <= _lpw["repaired"] + 1e-9,
      "app:replication's learned option-only reader is misquoted")
# the fewest-digits rule on the two rewrites, its margin over chance, and what it would add to the rank's contrast
_fr2, _fo = _brk["format_rule"], _brk["format_rule_over_chance"]
check(f"is correct on ${_fr2['repaired']:.1f}\\%$ of $U$'s items and ${_fr2['placebo']:.1f}\\%$ of $P$'s "
      f"(Table~\\ref{{tab:mechanism}}), ${_fo['repaired']['mean']:+.1f}$ $[{_fo['repaired']['lo']:+.1f},"
      f"{_fo['repaired']['hi']:+.1f}]$ and ${_fo['placebo']['mean']:+.1f}$ $[{_fo['placebo']['lo']:+.1f},"
      f"{_fo['placebo']['hi']:+.1f}]$ points over chance (${100 * _brk['levels']['whole file']:.0f}\\%$)" in _flat
      and f"under this rule $U$ gains ${_fr2['repaired'] - _fr2['placebo']:+.1f}$ over $P$" in _flat,
      "app:replication's fewest-digits sentence is stale")
# tab:arms' caption: the counts behind the two deleted-option columns on v1.5
_ra15 = load("results/release_arms.json")["releases"]["v1.5"]
_nk = [round(_ra15[m]["open_regrade"]["any_number_5pct"] * _ra15[m]["open_regrade"]["n_numeric_key"] / 100)
       for m in ("gpt-4o", "claude-3-5-sonnet-latest")]
_ng = [round(_ra15[m]["arms"]["open_ended"]["accuracy"] * _ra15[m]["arms"]["open_ended"]["n"] / 100)
       for m in ("gpt-4o", "claude-3-5-sonnet-latest")]
check(_ng[0] == _ng[1] and f"On v1.5 these two columns count ${_ng[0]}$ of $205$ answers for each model, and ${_nk[0]}$ "
      f"and ${_nk[1]}$ of ${_ra15['gpt-4o']['open_regrade']['n_numeric_key']}$." in _flat,
      "tab:arms' caption misstates the counts behind its deleted-option columns")

# --- app:stats's sign-flip paragraph: the test as randomization.py runs it, and its self-check ------
import randomization as _rz  # noqa: E402
_rsc = load("results/randomization_selfcheck.json")
_rsf = load("results/reader_split.json")["data|nearest|repaired-placebo|moved to an edge"]["signflip"]
check(f"over every pattern up to ${_rz.EXACT_UP_TO}$ capsules and ${_rz.DRAWS:,}$ random patterns beyond".replace(",", "{,}", 1)
      in _flat, "app:stats misstates how many sign patterns the test enumerates or draws")
check(f"on simulated files shaped like the moved keys (${_rsc['n_items']}$ items in ${_rsc['n_clusters']}$ capsules, a "
      f"true contrast of ${_rsc['true_contrast']:.0f}$ points, the items drawn independently of their capsule) covers the "
      f"truth in ${_rsc['covered']}$ of ${_rsc['reps']}$ files"
      in _flat and (_rsc["n_items"], _rsc["n_clusters"]) == (_rsf["n_items"], _rsf["n_clusters"]),
      "app:stats misquotes the sign-flip self-check, or its files are not shaped like the moved keys")

# --- the registered runs read by gemma-3-27b as BixBench reads a run (published_reads.py; not registered) ---
_prd2 = load("results/published_reads.json")
_mv2 = "repaired-placebo|moved to an edge"
_g1, _gs, _gq = _prd2["D1"], _prd2["D2"]["reruns|data"], _prd2["D2"]["qwen3-235b|data"]
_gsn, _gqn = _prd2["D2"]["reruns|nodata"], _prd2["D2"]["qwen3-235b|nodata"]
_iv2 = lambda v: f"${v['mean']:+.1f}$ $[{v['lo']:+.1f},{v['hi']:+.1f}]$"
_rep2 = load("results/replication.json")
_rule_d1 = _rep2["D1"][_mv2]["mean"]
_half = 0.4 < _g1[_mv2]["mean"] / _rule_d1 < 0.6 and _g1[_mv2]["lo"] > 0
_none = all(g[_mv2]["lo"] < 0 < g[_mv2]["hi"] for g in (_gs, _gq))
check(_half and _none, "the registered runs under gemma-3-27b no longer read half the rule's on the published runs and "
      "nothing clear of zero on the new ones with the data, as the abstract, sec:withdata and the limitations say")
# --- the readers of the same runs: three open models forced and gemma-3-27b allowed to decline, beside the
# published readings (published_reads.py analyse and compare; not registered) --------------------------------
_rdc = load("results/published_reads_readers.json")
_rq = load("results/published_reads_qwen72b_forced.json")
_rl = load("results/published_reads_llama70b_forced.json")
_rdd = load("results/published_reads_gemma27b_decline.json")
_fm = lambda e: e["follows_nearest"]["miss"]["share"]
# when a forced grader selects no option: gpt-4o 2024-11-20 and the published grades (grader_declines.py)
_gdn = load("results/grader_declines.json")["no_pick"]
_fa = lambda e: e["follows_nearest"]["all"]["share"]
_kp = lambda e: e["agreement"]["pooled"]["kappa"]
_agr = lambda e: e["agreement"]["pooled"]["agree"]
_qg, _lg, _dg = _rq["D1"][_mv2], _rl["D1"][_mv2], _rdd["D1"][_mv2]
_d0d = _rdd["D0|data"][_mv2]
_eq, _eg, _el = _rdc["qwen72b"], _rdc["gemma27b"], _rdc["llama70b"]
_epub, _edec, _erule = _rdc["published forced"], _rdc["gemma27b|decline"], _rdc["rule"]
_kmax = max(_kp(e) for e in (_eq, _eg, _el))
check(_rdc["n_runs"] == _eq["agreement"]["pooled"]["n"] and _fm(_eq) > _fm(_eg) > _fm(_el) and _qg["mean"] > max(_g1[_mv2]["mean"], _lg["mean"])
      and abs(_fm(_eq) - _fm(_epub)) < 2 and _eg[_mv2]["mean"] == _g1[_mv2]["mean"],
      "the readers' gains no longer follow how often they name the nearest option on a miss, or Qwen2.5-72B no longer "
      "names it about as often as the published reading")
# the introduction: the three forced readers' range on the published runs' moved keys
_rdr = [_qg["mean"], _g1[_mv2]["mean"], _lg["mean"]]
_rg4 = _rdc["gpt-4o"]
_rdr4 = _rdr + [_rg4[_mv2]["mean"]]
# the forced graders on the new runs' moved keys: gemma-3-27b, Qwen2.5-72B and Llama-3.3-70B on each group
_d2g = [(_gs[_mv2]["mean"], _gs["signflip"][f"gemma|{_mv2}"]["p"]), (_gq[_mv2]["mean"], _gq["signflip"][f"gemma|{_mv2}"]["p"])]
_d2g += [(_x[f"D2|{_grp}|data"][_mv2]["mean"], _x[f"D2|{_grp}|data"]["signflip"][f"{_nm2}|{_mv2}"]["p"])
         for _x, _nm2 in ((_rq, "qwen72b"), (_rl, "llama70b")) for _grp in ("reruns", "qwen3-235b")]
_d2m, _d2p = [m for m, _ in _d2g], [p for _, p in _d2g]
# the gradings with the notebook under BixBench's prompt in tab:readers2 (published runs and new seeds): their range
# on the moved keys, as the abstract and the introduction give it, and the order of their proximity weights
_t2g = load("results/grading_variants.json")
_nbm = [r["moved"]["mean"] for r in _t2g["table2"] if r["moved"] and r["grader"] != "nearest option"
        and "code-free" not in r["grader"] and "within" not in r["grader"]]
# the forced graders with the notebook under BixBench's prompt: their range on the published runs' moved keys, and
# none significant on the new seeds, as the abstract and the introduction say
_nbg = ("Qwen2.5-72B", "gemma-3-27b", "Llama-3.3-70B", "gpt-4o")
_nbp = [r["moved"]["mean"] for r in _t2g["table2"] if r["runs"] == "v1.0, published" and r["grader"] in _nbg]
_nbs = [r["p"] for r in _t2g["table2"] if r["runs"] == "v1.5, new seeds" and r["grader"] in _nbg]
check(len(_nbp) == 4 and len(_nbs) == 3 and min(_nbs) > 0.05
      and _t2g["proximity_cost"]["spearman"] > 0.8 and max(_nbm) < _rule_d1,
      "the graders with the notebook no longer bear less than the rule, or one is significant on the new seeds")
# sec:withdata: the graders with the notebook on the published runs, gpt-4o's selections, and refusal
_owg = [_qg["mean"], _g1[_mv2]["mean"], _lg["mean"]]
check(all(g["lo"] > 0 for g in (_qg, _g1[_mv2], _lg)) and _cbc["readers of the published runs"]["survives"]
      and _gdn["gpt-4o 2024-11-20"]["correct"]["no_pick"] < 5 < 50 < _gdn["gpt-4o 2024-11-20"]["miss"]["no_pick"],
      "the three open graders' gains on the published runs no longer clear zero and the claim family, or gpt-4o "
      "2024-11-20 no longer declines misses and not correct answers")
# the same graders on the new seeds' moved keys: gemma-3-27b, Qwen2.5-72B and Llama-3.3-70B
_d2s = [(_gs[_mv2]["mean"], _gs["signflip"][f"gemma|{_mv2}"]["p"])]
_d2s += [(_x["D2|reruns|data"][_mv2]["mean"], _x["D2|reruns|data"]["signflip"][f"{_nm2}|{_mv2}"]["p"])
         for _x, _nm2 in ((_rq, "qwen72b"), (_rl, "llama70b"))]
check(min(q for _, q in _d2s) > 0.05, "a grader with the notebook is now significant on the new seeds")
_kdr = [_edec["agreement"][r]["kappa"] for r in ("4o_open_image", "4o_open_no_image", "claude_open_image",
                                                   "claude_open_no_image")]
_sdn = load("results/score_decomposition.json")["v1.5 new"]
_pmd = _rdc["published may-decline"]
_a3 = _edec["agreement3"]
_c3 = _a3["reader_by_published"]
check(f"On the quarter it refuses ${_edec['declined']:.1f}\\%$ of its gradings through $R$, where the "
      f"published grades refuse ${_pmd['declined']:.1f}\\%$ of runs, and the two agree on whether a run is "
      f"graded correct for ${_agr(_edec):.1f}\\%$ of runs ($\\kappa={_kdr[0]:.2f}$, ${_kdr[1]:.2f}$, "
      f"${_kdr[2]:.2f}$ and ${_kdr[3]:.2f}$ by configuration). Over the three outcomes a grading can have (the "
      f"key, another option or a refusal), they agree on ${_a3['agree']:.1f}\\%$ of gradings "
      f"($\\kappa={_a3['kappa']:.2f}$): where gemma-3-27b selects the key, the published grader refuses on "
      f"${_c3['key']['declined']}$ of ${sum(_c3['key'].values())}$ gradings, and where it selects another "
      f"option, on ${_c3['other']['declined']}$ of ${sum(_c3['other'].values())}$. On a miss, the option it "
      f"selects is the nearest on ${_fm(_edec):.1f}\\%$ of its choices, the published grader's on "
      f"${_fm(_pmd):.1f}\\%$. Against $P$ its moved keys gain {_iv2(_dg)} on the published runs "
      f"($p={_rdd['D1']['signflip']['gemma27b|' + _mv2]['p']:.3f}$), {_iv2(_d0d)} on the seven configurations "
      f"and ${_rdd['D2|reruns|data'][_mv2]['mean']:+.1f}$ and ${_rdd['D2|qwen3-235b|data'][_mv2]['mean']:+.1f}"
      f"$ on the new runs (Table~\\ref{{tab:readersfull}}); through $R$ it scores the new seeds' runs "
      f"${_sdn['new seeds']['gemma may-decline']['score_minus_tolerance']['mean']:+.1f}$ and Qwen3-235B-A22B's "
      f"${_sdn['Qwen3-235B-A22B']['gemma may-decline']['score_minus_tolerance']['mean']:+.1f}$ points from the "
      f"tolerance."
      in _flat, "app:replication's paragraph on the may-decline reads is stale")
# sec:conclusion's recommendation: the may-decline reader's share of the rule's gain, and the largest forced share
check(0.14 < _dg["mean"] / _rule_d1 < 0.2 and 0.7 < max(_qg["mean"], _g1[_mv2]["mean"], _lg["mean"]) / _rule_d1 < 0.8,
      "the graders' shares of the rule's gain on the published runs have moved")
# the limitations
_g4agr = load("results/published_reads_gpt-4o_forced.json")["D1"]["agreement_with_published_reader"]
_ltr = load("results/grading_variants.json")["letter"]
check(f"Where the original graders cannot be run, we substitute. gpt-4o 2024-11-20 replaces 2024-08-06, which "
      f"Azure OpenAI, where our runs are served, no longer offers to new customers; 2024-08-06 selected no "
      f"option on ${_gdn['published, gpt-4o']['miss']['no_pick']:.1f}\\%$ of the published runs' misses, "
      f"2024-11-20 on ${_gdn['gpt-4o 2024-11-20']['miss']['no_pick']:.1f}\\%$. For Claude 3.5 Sonnet, which "
      f"has been retired, we substitute nothing. Our code-free grader is BixBench's framing without the "
      f"notebook, asked to map the answer; it approximates the 2026 system's grader but is not that grader." in _flat
      and abs(_ltr["accepted"]["all misses"] - _ltr["accepted_published"]["all misses"]) < 3 and _ltr["no_pick"] == 0
      and _gnd["served"] == ["gpt-4o-2024-11-20"] and any(k.startswith("claude") for k in _g4agr),
      "the limitations misstate which closed models were run again")
# app:replication's paragraph on the other readers
_agr = lambda e: e["agreement"]["pooled"]["agree"]
check(f"On that quarter (${_rdc['n_runs']:,}$ runs)".replace(",", "{,}", 1) + f" the published grades select the option nearest the submitted "
      f"number on ${_fa(_epub):.1f}\\%$ of their choices and on ${_fm(_epub):.1f}\\%$ of those on a miss; Qwen2.5-72B on "
      f"${_fa(_eq):.1f}$ and ${_fm(_eq):.1f}\\%$, gemma-3-27b on ${_fa(_eg):.1f}$ and ${_fm(_eg):.1f}\\%$ and Llama-3.3-70B "
      f"on ${_fa(_el):.1f}$ and ${_fm(_el):.1f}\\%$. Counting a run as correct when both orders select the key, they agree "
      f"with the published grades on ${_agr(_eq):.1f}$, ${_agr(_eg):.1f}$ and ${_agr(_el):.1f}\\%$ of runs "
      f"($\\kappa={_kp(_eq):.2f}$, ${_kp(_eg):.2f}$ and ${_kp(_el):.2f}$), and the nearest-option rule on ${_agr(_erule):.1f}\\%$ "
      f"($\\kappa={_kp(_erule):.2f}$)" in _flat,
      "app:replication's paragraph on the other readers is stale")
# tab:readers2's and tab:readersfull's captions: the moved keys on each release
_sf1 = _g1["signflip"][f"gemma|{_mv2}"]
check(f"($37$ on v1.5; ${_g1['n_items']['moved to an edge']}$ on v1.0, ${_sf1['n_items']}$ of them also redrawn by $P$)"
      in _flat and load("results/reader_split.json")["n_items"]["moved to an edge"] == 37
      and f"(${_g1['n_items']['moved to an edge']}$ on v1.0, ${_sf1['n_items']}$ of them also redrawn by $P$; $37$ on "
          f"v1.5)" in _flat,
      "tab:readers2's or tab:readersfull's caption miscounts the moved keys")
# app:replication's paragraph on the reads
_nrd, _nbl = collections.Counter(), collections.Counter()
with gzip.open("results/published_reads_rows.jsonl.gz", "rt") as _fh:
    for _line in _fh:
        _r = json.loads(_line)
        _nrd[_r["set"]] += sum(len(x) for x in _r["reads"].values())
        _nbl[_r["set"]] += not _r["reads"]
_c = lambda n: f"{n:,}".replace(",", "{,}")
_ag = _g1["agreement_with_published_reader"].values()
_rng = lambda key, f=".1f": f"${min(a[key] for a in _ag):{f}}$ to ${max(a[key] for a in _ag):{f}}"
_pv2 = lambda sf: "<0.001" if sf["p"] < 0.001 else f"={sf['p']:.3f}"
check(f"is scored wrong without grading (${_nbl['D1']}$ published runs, ${_nbl['D2']}$ new). That is "
      f"${_c(_nrd['D1'])}$ gradings of the published runs and ${_c(_nrd['D2'])}$ of the new ones through $R$, $P$ and $U$, "
      f"and it selects no option in ${_g1['no_pick']:.1f}$ and ${_prd2['D2']['no_pick']:.1f}\\%$ of them" in _flat
      and f"it selects the key in {_rng('gemma_right')}\\%$ of its gradings of each published configuration, where the "
          f"published grades select it for {_rng('published_right')}\\%$ of runs" in _flat
      and f"the two agree on {_rng('agree')}\\%$ of runs ($\\kappa={_rng('kappa', '.2f')[1:]}$)" in _flat
      and f"redrew gain {_iv2(_g1[_mv2])} under it ($p{_pv2(_sf1)}$), and their inward keys change by "
          f"{_iv2(_g1['repaired-placebo|moved inward'])}" in _flat
      and f"the new runs' moved keys gain ${_gs[_mv2]['mean']:+.1f}$ and ${_gq[_mv2]['mean']:+.1f}$ with the data "
          f"($p{_pv2(_gs['signflip']['gemma|' + _mv2])}$ and ${_gq['signflip']['gemma|' + _mv2]['p']:.3f}$) and "
          f"{_iv2(_gsn[_mv2])} and {_iv2(_gqn[_mv2])} without it" in _flat
      and f"${_sf1['n_items']}$ moved keys that $P$ also redrew" in _flat,
      "app:replication's paragraph on the registered runs read by a model is stale")
# app:stats: the rule's sign-flip test on the registered runs
_rsf1 = _g1["signflip"][f"rule|{_mv2}"]
_rsfs = _gs["signflip"][f"rule|{_mv2}"]
_rsfq = _gq["signflip"][f"rule|{_mv2}"]
check(_rsf1["p"] < 0.001 and _rsfs["p"] < 0.001 and _rep2["D2"]["qwen3-235b|data"][_mv2]["lo"] <= 0
      and f"under the nearest-option rule, on the pre-specified test's data, it gives $p<0.001$ on the published runs "
          f"and on the new seeds, and $p={_rsfq['p']:.3f}$ for Qwen3-235B-A22B with the data, whose interval also "
          f"reaches zero (Table~\\ref{{tab:replication}})" in _flat,
      "app:stats's sign-flip p-values on the registered runs are stale")

# --- numbers in this round's text that a one-digit mutation left unchecked, each against its source ---
import published_reads as _prm  # noqa: E402
import tolerance_check as _tcm  # noqa: E402
_rsn = load("results/reader_split.json")["n_items"]
check(f"extreme and ${_rsn['moved inward']}$ from extreme to bracketed (the \\emph{{inward keys}}), and leaves "
      f"${_rsn['kept']}$ in their class" in _flat,
      "sec:withdata's opening miscounts the keys the repair keeps")
check(f"as does $U$ redrawn with each of ${len(load('results/repair_seeds.json')['seeds'])}$ seeds" in _flat,
      "sec:withdata miscounts the repair's seeds")
_rk = [0, 0, 0, 0]
for _o in bw.option_sets().values():
    _rk[bw.pick_rank(_o["released"][0], _o["released"])] += 1
check(f"The cells are small (${_rk[0]}$, ${_rk[1]}$, ${_rk[2]}$ and ${_rk[3]}$ items)," in _flat,
      "sec:open miscounts the numeric items at each key rank")
_mp = next(b for b in load("results/channel_survey.json")["benchmarks"] if b["benchmark"] == "MMLU-Pro")
check(f"MMLU-Pro (${100 * _mp['cross_validated']['credit']:+.1f}$) and MMLU" in _flat,
      "app:stats misquotes MMLU-Pro's held-out rank rule")
check(f"is read at the level that attains $1-0.05/{load('results/claim_budget.json')['k']}$ coverage on its own file" in _flat,
      "app:stats misstates the claim family's size")
check(len(_prereg) > 6631
      and "a mean within 5 points of 0" in " ".join(_prereg[:6631].decode().split())
      and "The pass criterion for the unchanged keys also accepts a mean within $5$ points of zero" in _flat,
      "app:replication misquotes the plan's length or H2's registered margin")
_q3i = _rep2["D2"]["qwen3-235b|data"]["gain|moved inward"]
_capnote = f"No nominal level attains $95\\%$ coverage on these ${_q3i['n_items']}$ keys in ${_q3i['n_clusters']}$ capsules"
check(_q3i["level_capped"] and _capnote + "; H5's criterion" in _flat and _capnote + ", so no interval is given" in _flat,
      "tab:predictions' or tab:replication's capped-interval note miscounts its keys or capsules")
_nnr = _rep2["D1"]["n_numeric_runs"]
check(f"and {', '.join(f'${_c(_nnr[r])}$' for r in _rep.OPEN_RUNS[:-1])} and ${_c(_nnr[_rep.OPEN_RUNS[-1]])}$ of the runs, "
      f"${_c(sum(_nnr.values()))}$ in all, answer those" in _flat,
      "app:replication miscounts the published runs on the numeric questions")
check(f"truncated at ${_c(_prm.BUDGET)}$ characters" in _flat, "app:replication misstates the notebook budget of the reads")
_n15, _n10 = len(bw.option_sets()), _rep2["D1"]["n_numeric_items"]
check(f"graders on the same ${_n15}$ numeric items by Kendall's" in _flat
      and f"open-answer graders on the same ${_n15}$ numeric items, with the data" in _flat
      and f"on the published runs of the ${_n10}$ numeric questions, in points" in _flat,
      "tab:ranking or its paragraph miscounts the items")

# --- what the released options do to a score (score_decomposition.py, tab:released) ----------------
# Every published run on a numeric question beside the tolerance, read by BixBench's own forced and
# may-decline readings of its released options; the abstract, the introduction and sec:withdata
# quote them, and tab:released prints them.
import score_decomposition as _sdm  # noqa: E402
_sdr = load("results/score_decomposition.json")
_relb = " ".join(_table_body("tab:released").split())
_relrows = _sdm.table_rows(_sdr)
for _row in _relrows:
    check(" ".join(_row.split()) in _relb, f"tab:released should carry the row {_row}")
check(_relb.count("\\\\") == len(_relrows) + 2,
      f"tab:released should hold exactly the {len(_relrows)} rows score_decomposition.table_rows gives")
_pf = {m: _sdr["v1.0"][m]["published forced"] for m in ("gpt-4o", "Claude 3.5 Sonnet")}
_pd = {m: _sdr["v1.0"][m]["published may-decline"] for m in ("gpt-4o", "Claude 3.5 Sonnet")}
_s7f, _s7d = _sdr["v1.5"]["pooled"]["family forced"], _sdr["v1.5"]["pooled"]["family may-decline"]
_sg = lambda v: f"${v['mean']:+.1f}$ $[{v['lo']:+.1f},{v['hi']:+.1f}]$"
_smt = lambda s: s["score_minus_tolerance"]
_brk10 = 100 - _b10["interior_rate"] * 100
_g4, _cl = _pf["gpt-4o"], _pf["Claude 3.5 Sonnet"]
_dg4, _dcl = _pd["gpt-4o"], _pd["Claude 3.5 Sonnet"]
_crd = lambda s: 100 * (s["miss_credited_nearest"] + s["miss_credited_no_number"] + s["miss_credited_other"]) \
    / (100 - s["tolerance"])
_nm = {m: {k: v["named"] for k, v in _pf[m]["named_on_a_miss"].items()} for m in _pf}
_om = {m: _pf[m]["other_misses"] for m in _pf}
# the same gradings against other references than the tolerance on the last number (reference_check.py)
import reference_check as _rfm  # noqa: E402
_rfc = load("results/reference_check.json")
_rg = {m: _rfc["groups"][f"v1.0|{m}"]["published forced"] for m in ("gpt-4o", "Claude 3.5 Sonnet")}
_rg7 = _rfc["groups"]["v1.5|seven configurations"]["family forced"]
_nn = _rfc["no_number"]
_prr = _rfc["published_rates"]
_gaps = sorted(v["gap_implied"] for v in _prr["models"].values())
_bio_year = re.search(r"@\w+\{bioagents,.*?year\s*=\s*\{(\d{4})\}", bib, re.S).group(1)
_refb = " ".join(_table_body("tab:reference").split())
_refrows = _rfm.table_rows(_rfc)
for _row in _refrows:
    check(" ".join(_row.split()) in _refb, f"tab:reference should carry the row {_row}")
check(_refb.count("\\\\") == len(_refrows) + 1,
      f"tab:reference should hold exactly the {len(_refrows)} rows reference_check.table_rows gives")
# tab:restated's forced-choice excess at 1 to 10%, by set of runs
_rt = {k: [x["mean"] for x in _tol["T3"]["released_reading"][k].values()] for k in _tol["T3"]["released_reading"]}
_rt_forced = [v for k, xs in _rt.items() if k.endswith("forced") for v in xs]
# the published runs and the seven configurations (their own graders and gemma-3-27b's), and every other set of runs
_RT_MAIN = ("v1.0|gpt-4o|forced", "v1.0|claude|forced", "v1.5|seven|forced", "v1.5|seven|gemma forced")
_rt_main = [v for k in _RT_MAIN for v in _rt[k]]
_rt_other = [v for k, xs in _rt.items() if k.endswith("forced") and k not in _RT_MAIN for v in xs]
_rt_decline = [v for k in ("v1.0|gpt-4o|may decline", "v1.0|claude|may decline", "v1.5|seven|may decline")
               for v in _rt[k]]
check(all(abs(g["tolerance"]["excess"]["mean"] - _smt(x)["mean"]) < 0.05
          for g, x in ((_rg["gpt-4o"], _g4), (_rg["Claude 3.5 Sonnet"], _cl), (_rg7, _s7f)))
      and f"The excess does not come from how the tolerance reads an answer. Against BixBench's own open-ended "
          f"grading of the same runs it is ${_rg['gpt-4o']['graders']['excess']['mean']:+.1f}$ and "
          f"${_rg['Claude 3.5 Sonnet']['graders']['excess']['mean']:+.1f}$. At every tolerance from $1$ to "
          f"$10\\%$ it is ${min(_rt_main):.0f}$ to ${max(_rt_main):.0f}$ points on the published runs and the "
          f"seven configurations, and ${min(_rt_other):.0f}$ to ${max(_rt_other):.0f}$ on the other runs "
          f"graded with the data (Table~\\ref{{tab:restated}}). Grading the ${_rfc['p_value_items']['v1.0']}$ "
          f"keys that are p-values within a factor of ten leaves "
          f"${_rg['gpt-4o']['p-values within x10']['excess']['mean']:+.1f}$ and "
          f"${_rg['Claude 3.5 Sonnet']['p-values within x10']['excess']['mean']:+.1f}$ "
          f"(Table~\\ref{{tab:reference}})."
          in _flat
      and all(abs(g[k]["excess"]["mean"] - g["tolerance"]["excess"]["mean"]) < 1
              for g in _rg.values() for k in ("any number", "without p-values")),
      "sec:withdata's account of the excess against other references is stale")
_dev = max(abs(g[k]["excess"]["mean"] - g["tolerance"]["excess"]["mean"])
           for g in (*_rg.values(), _rg7) for k in ("any number", "without p-values"))
_x10 = sorted(g["tolerance"]["excess"]["mean"] - g["p-values within x10"]["excess"]["mean"]
              for g in (*_rg.values(), _rg7))
_ng4, _ncl = _nn["v1.0|gpt-4o"], _nn["v1.0|Claude 3.5 Sonnet"]
check(f"the forced-choice excess is ${_rg['gpt-4o']['graders']['excess']['mean']:+.1f}$, "
      f"${_rg['Claude 3.5 Sonnet']['graders']['excess']['mean']:+.1f}$ and ${_rg7['graders']['excess']['mean']:+.1f}$. "
      f"Counting any number in the answer, or leaving out the keys that are p-values by the question's wording "
      f"(${_rfc['p_value_items']['v1.0']}$ of v1.0's ${_rfc['p_value_items']['v1.0 numeric']}$, "
      f"${_rfc['p_value_items']['v1.5']}$ of v1.5's ${_rfc['p_value_items']['v1.5 numeric']}$), moves it by at most "
      f"${_dev:.1f}$ points; grading those keys within a factor of ten lowers it by ${_x10[0]:.1f}$ to ${_x10[-1]:.1f}$ "
      f"points" in _flat
      and f"the open-ended graders accept ${_ng4['graders_accept']:.1f}$ and ${_ncl['graders_accept']:.0f}\\%$ of the "
          f"answers that contain no number; pooled over runs, the key's value is in the notebook for "
          f"${_ng4['key_written']:.1f}$ and "
          f"${_ncl['key_written']:.1f}\\%$ of these runs, a distractor's for ${_ng4['distractor_written']:.1f}$ and "
          f"${_ncl['distractor_written']:.1f}\\%$, and the forced-choice grades accept ${_ng4['accepted_if_written']:.1f}$ and "
          f"${_ncl['accepted_if_written']:.1f}\\%$ of the answers where the key's value appears and "
          f"${_ng4['accepted_if_not']:.1f}$ and ${_ncl['accepted_if_not']:.1f}\\%$ where it does not" in _flat
      and abs(_ng4["accepted"] - _nm["gpt-4o"]["no number"]) < 0.05
      and abs(_ncl["accepted"] - _nm["Claude 3.5 Sonnet"]["no number"]) < 0.05,
      "app:replication's account of the other references and of the answers with no number is stale")
# the paper no longer applies the published grades' rates to the 2026 system's reported accuracy: a grader that does
# not see the notebook, on another release and a stronger agent, need not share them
check("points to a system with the" not in _flat and "from forced-choice grading, most of the" not in _flat
      and "o(1-r)+(1-o)a" not in _flat
      and abs(_prr["gap_reported"] - (_prr["forced_reported"] - _prr["open_ended"])) < 1e-9,
      "the paper again projects the published grades' rates onto the 2026 system")
_fsr = load("results/formula_scoring.json")["reported"]
check(f"a 2026 system reports ${_prr['forced_reported']:.1f}\\%$ forced-choice against ${_prr['open_ended']:.1f}\\%$ "
      f"open-ended accuracy on v1.5" in _flat
      and abs(_fsr["forced"] - _prr["forced_reported"]) < 1e-9 and abs(_fsr["open"] - _prr["open_ended"]) < 1e-9
      and abs(_fsr["corrected"] - (_fsr["forced"] - 25) / 0.75) < 1e-9
      and f"The correction maps the 2026 system's ${_fsr['forced']:.1f}\\%$ to ${_fsr['corrected']:.1f}\\%$, "
          f"${_fsr['corrected_minus_open']:.1f}$ points above its open-ended ${_fsr['open']:.1f}\\%$" in _flat,
      "sec:withdata's account of the 2026 system's corrected score no longer matches reference_check.json")
check(f"the tolerance without the items whose key is a p-value (${_rfc['p_value_items']['v1.0']}$ of v1.0's "
      f"${_rfc['p_value_items']['v1.0 numeric']}$, ${_rfc['p_value_items']['v1.5']}$ of v1.5's "
      f"${_rfc['p_value_items']['v1.5 numeric']}$); and the tolerance with those keys graded within a factor of ten"
      in _flat, "tab:reference's caption should count the p-value keys as reference_check.py does")
_wr = lambda m, k: round(_om[m]["written"][k])
# the share of the misses a grading grades that it accepts, over all of them and by what the answer gives, beside
# a choice at random (score_decomposition.miss_rates); and what the published grades select on a miss (picks_on_misses)
_mr = lambda x, k: x["miss_rates"][k]["accepted"]
_mra = lambda x: x["miss_rates"]["all"]
_pk2 = {m: _pf[m]["picks_on_misses"] for m in _pf}
# the share of the answers within 5% of the key that a grading accepts, and formula_scoring.py's rows
_cacc = lambda x: 100 * x["hit_named"] / (x["hit_named"] + x["hit_refused"])
_fsrows = {(r["block"], r["group"], r["reading"]): r for r in load("results/formula_scoring.json")["rows"]}
_fsg = _fsrows[("v1.0", "gpt-4o", "published forced")]
# Table 1's current agents: gpt-5.1 (score_decomposition.py) and the two of frontier_agents.py
_t1cur = [_sdr["v1.5 closed"]["gpt-5.1"]["gpt-4o forced"]] + [
    load("results/frontier_agents.json")["agents"][k]["table1"]["gpt-4o forced"] for k in ("gpt-6-luna", "DeepSeek-V4-Pro")]
_curx = [_smt(x)["mean"] for x in _t1cur]
_curk = [_sdm.contribution(x, "nearest") for x in _t1cur]
_ct1 = lambda x, k: _sdm.contribution(x, k)
_s7g = _sdr["v1.5"]["pooled"]["gemma forced"]
check(f"Forced to choose, BixBench's grader accepts ${_mr(_g4, 'all'):.1f}$ and ${_mr(_cl, 'all'):.1f}\\%$ of "
      f"the misses it grades on the published runs, where a random choice among four would accept $25\\%$ "
      f"(Table~\\ref{{tab:released}}). Which misses it accepts is not random. It accepts "
      f"${_mr(_g4, 'nearest'):.1f}$ and ${_mr(_cl, 'nearest'):.1f}\\%$ of the misses whose nearest option is "
      f"the key (v1.0 leaves ${_brk10:.0f}\\%$ of its keys extreme) and ${_mr(_g4, 'other'):.1f}$ and "
      f"${_mr(_cl, 'other'):.1f}\\%$ of those nearest another option. gpt-4o's grades select no option on "
      f"${100 - _fsg['misses_selected']:.1f}\\%$ of its misses, which BixBench's parser scores wrong, and "
      f"accept ${_fsg['misses_accepted_of_selected']:.1f}\\%$ of the others, more than a random choice would. "
      f"The grader rejects ${100 - _cacc(_g4):.1f}$ and ${100 - _cacc(_cl):.1f}\\%$ of the correct answers, "
      f"and so scores the published runs {_sg(_smt(_g4))} and {_sg(_smt(_cl))} points above the tolerance. The "
      f"excess splits exactly by what the answer gives: the misses nearest the key contribute "
      f"${_ct1(_g4, 'nearest'):.1f}$ and ${_ct1(_cl, 'nearest'):.1f}$ points, those nearest another option "
      f"${_ct1(_g4, 'other'):.1f}$ and ${_ct1(_cl, 'other'):.1f}$, and the answers with no number or no single "
      f"nearest option ${_ct1(_g4, 'neither'):.1f}$ and ${_ct1(_cl, 'neither'):.1f}$. The grader accepts "
      f"${_mr(_g4, 'no number'):.0f}$ and ${_mr(_cl, 'no number'):.0f}\\%$ of the answers with no number, "
      f"although BixBench's open-ended graders accept almost none of these and the key's value is in the "
      f"notebook for only ${_ng4['key_written']:.0f}$ and ${_ncl['key_written']:.0f}\\%$ of them." in _flat
      and _fsg["misses_accepted_of_selected"] > 25 and abs(_fsg["misses_accepted"] - _mr(_g4, "all")) < 1e-6
      and all(_mra(x)["lo"] <= 25 <= _mra(x)["hi"] and _mra(x)["level"] == 0.95 for x in (_g4, _cl))
      and all(x["miss_rates"]["chance"] == 25 and abs(x["miss_rates"]["at_chance"] - 0.25 * _mra(x)["share_of_runs"]) < 1e-9
              and abs(_smt(x)["mean"] - (_mr(x, "all") * _mra(x)["share_of_runs"] / 100 - x["hit_refused"])) < 0.05
              for x in (_g4, _cl))
      and max(_ng4["graders_accept"], _ncl["graders_accept"]) < 1
      and all(_om[m]["rescaled"]["key"] <= _om[m]["rescaled"]["distractor"] for m in _om)
      and all(abs(_pk2[m]["key"]["nearest"] - _mr(_pf[m], "nearest")) < 1e-6
              and abs(_pk2[m]["other"]["key"] - _mr(_pf[m], "other")) < 1e-6 for m in _pf)
      and f"The seven v1.5 configurations are scored {_sg(_smt(_s7f))} points above the tolerance by their own "
          f"models and {_sg(_smt(_s7g))} by gemma-3-27b for all seven, so the excess does not come from "
          f"self-grading; the current agents are scored ${min(_curx):+.1f}$ to ${max(_curx):+.1f}$, of which "
          f"the misses nearest the key contribute ${min(_curk):.1f}$ to ${max(_curk):.1f}$. With a refusal "
          f"option, BixBench's grading accepts ${_mr(_dg4, 'all'):.1f}$ and ${_mr(_dcl, 'all'):.1f}\\%$ of the "
          f"published runs' misses but rejects ${100 - _cacc(_dg4):.1f}$ and ${100 - _cacc(_dcl):.1f}\\%$ of "
          f"the correct answers, refusing ${_dg4['declined']:.0f}$ and ${_dcl['declined']:.0f}\\%$ of runs, "
          f"and lands ${_smt(_dg4)['mean']:+.1f}$ and ${_smt(_dcl)['mean']:+.1f}$ points from the tolerance, "
          f"the net of the misses it accepts and the correct answers it rejects." in _flat
      and max(_g4["hit_refused"], _cl["hit_refused"]) < 0.5 and min(_rt_decline) > -1.0 and min(_curx) > 0,
      "sec:withdata's account of what the released options do is stale")
# the abstract and the introduction: the forced reading's excess, how often it names the key on a miss and where,
# and the may-decline reading beside the tolerance
_follow = all(_mr(_pf[m], "nearest") > _mr(_pf[m], "no number") > _mr(_pf[m], "other") for m in _pf)
_rescue = all(_om[m]["written"]["named_if_key"] - _om[m]["written"]["named_if_not"] < 5 for m in _om)
check("Forced to choose, BixBench's grader accepts a fifth to a quarter of the wrong answers, about as many as a random "
      "choice among four would, but it favours the misses whose nearest option is the key" in _flat
      and all(20 <= _mr(x, "all") <= 26 for x in (_g4, _cl))
      and f"misses whose nearest option is the key: on the published runs it accepts them $"
          f"{_mr(_g4, 'nearest') / _mr(_g4, 'other'):.1f}$ and ${_mr(_cl, 'nearest') / _mr(_cl, 'other'):.1f}$ "
          f"times as often as those nearest a distractor." in _flat
      and f"On current agents forced choice still adds ${min(_curx):.0f}$ to ${max(_curx):.0f}$ points." in _flat
      and min(_mr(_g4, "no number"), _mr(_cl, "no number")) > 25
      and max(_smt(_dg4)["mean"], _smt(_dcl)["mean"]) < 4
      and f"BixBench's grader accepts ${_mr(_g4, 'all'):.1f}$ and ${_mr(_cl, 'all'):.1f}\\%$ of the misses on "
          f"the published gpt-4o and Claude 3.5 Sonnet runs, near a random choice's $25\\%$, but $"
          f"{_mr(_g4, 'nearest'):.1f}$ and ${_mr(_cl, 'nearest'):.1f}\\%$ of those whose nearest option is the "
          f"key and ${_mr(_g4, 'other'):.1f}$ and ${_mr(_cl, 'other'):.1f}\\%$ of those nearest another. "
          f"gpt-4o's rate is lowered by the ${100 - _fsg['misses_selected']:.1f}\\%$ of misses on which its "
          f"reply selects no option, which BixBench's parser scores wrong. Of the ${_smt(_g4)['mean']:.1f}$ "
          f"and ${_smt(_cl)['mean']:.1f}$ points forced choice adds over the tolerance, the misses nearest the "
          f"key contribute ${_ct1(_g4, 'nearest'):.1f}$ and ${_ct1(_cl, 'nearest'):.1f}$; the rest comes from "
          f"misses the options place elsewhere and from answers that give no number. Current agents are still "
          f"scored ${min(_curx):.1f}$ to ${max(_curx):.1f}$ points above the tolerance, although" in _flat
      and min(_g4["pick_is_nearest"], _cl["pick_is_nearest"]) > 50
      and _follow and _rescue
      and all(_smt(x)["lo"] > 0 for x in (_g4, _cl)),
      "the abstract's or the introduction's account of what the released options do is stale")
# app:replication's account of the decomposition: gemma-3-27b's forced reading of the same published runs, how
# often each reading names the option nearest the number, and whether a non-nearest pick is a value the
# notebook prints (score_decomposition_notebook_values.jsonl.gz, built from the notebooks)
_gf = {m: _sdr["v1.0"][m]["gemma forced"] for m in ("gpt-4o", "Claude 3.5 Sonnet")}
check(f"Graded by gemma-3-27b instead of by the published grades, the published runs score "
      f"${_smt(_gf['gpt-4o'])['mean']:+.1f}$ and "
      f"${_smt(_gf['Claude 3.5 Sonnet'])['mean']:+.1f}$ points above the tolerance. The published forced-choice grades "
      f"select the option nearest the submitted number on ${_g4['pick_is_nearest']:.1f}$ and "
      f"${_cl['pick_is_nearest']:.1f}\\%$ of their choices for answers that are numbers, gemma-3-27b on "
      f"${_gf['gpt-4o']['pick_is_nearest']:.1f}$ and ${_gf['Claude 3.5 Sonnet']['pick_is_nearest']:.1f}\\%$. Where the "
      f"published grades select another option, that option's value appears in the run's notebook for "
      f"${_g4['not_nearest_pick_written']:.1f}$ and ${_cl['not_nearest_pick_written']:.1f}\\%$ of those choices, against "
      f"${_g4['not_nearest_other_written']:.1f}$ and ${_cl['not_nearest_other_written']:.1f}\\%$ for the options not "
      f"selected" in _flat
      and all(s["not_nearest_pick_written"] - s["not_nearest_other_written"] < 5 for s in (_g4, _cl))
      and Path("results/score_decomposition_notebook_values.jsonl.gz").exists(),
      "app:replication's account of what the released options do is stale")
# the recommendations: which misses a forced grader accepts most often, and the answers with no number
check(_follow and "Which misses forced choice accepts depends on how the options were written" in _flat
      and f"more than a quarter of gpt-4o's published answers to numeric questions contain none, and forced-choice "
          f"grading accepts "
          f"${_mr(_g4, 'no number'):.0f}\\%$ of those" in _flat and _ng4["graders_accept"] < 1
      and f"and BixBench's refusal option ${100 - _cacc(_dg4):.1f}$ and ${100 - _cacc(_dcl):.1f}\\%$" in _flat,
      "sec:conclusion's account of what forced-choice grading accepts is stale")

# --- sec:theory's AQuA-RAT sentence and app:replication's set-aside reader ---------------------------
# AQuA-RAT brackets the share of its keys a uniform rank gives at its k, (k-2)/k
_aq = {b["benchmark"]: b for b in load("results/channel_survey.json")["benchmarks"]}["AQuA-RAT"]
check(abs(_aq["interior_expected"] - (_aq["modal_n_options"] - 2) / _aq["modal_n_options"]) < 1e-9
      and abs(_aq["interior_rate"] - _aq["interior_expected"]) < 0.01
      and f"brackets ${100 * _aq['interior_rate']:.0f}\\%$ of its keys, as a uniform rank does at "
          f"$k={_aq['modal_n_options']}$" in _flat,
      "sec:theory's AQuA-RAT bracketing share is stale, or no longer what a uniform rank gives")
# Qwen3-235B-A22B's attempted reads, shipped as replies: how many it finished and how many name no option
with gzip.open("results/published_reads_qwen3-235b_attempt.jsonl.gz", "rt") as _fh:
    _q3a = [json.loads(_line) for _line in _fh]
_q3z = sum(bw.xml_extract(_r["reply"]) == "Z" for _r in _q3a)
check({_r["model"] for _r in _q3a} == {"qwen3-235b"}
      and f"${_q3z}$ of the first ${len(_q3a)}$ replies it finished select no option" in _flat,
      "app:replication miscounts Qwen3-235B-A22B's attempted reads")


# --- this round's survivors of a one-digit mutation, each pinned in context against its source ----------
# the limitations: the agents' share of numeric items within 5% of the key, with the data, over the seven run
# sets and the registered new runs
_w5 = [s["family forced"]["tolerance"] for g, s in load("results/score_decomposition.json")["v1.5"].items()
       if g != "pooled"]
_w5 += [v["mean"] for k, v in load("results/replication.json")["not_registered"]["within_5pct"].items()
        if k.endswith("|data")]
check(f"with the data their number is within $5\\%$ of the key on ${min(_w5):.0f}$ to ${max(_w5):.0f}\\%$ of numeric "
      f"items" in _flat, "the limitations misstate the agents' share of answers within 5% of the key")
_agsrc = Path("bixbench_agent.py").read_text()
_temp = re.search(r'add_argument\("--temperature", type=float, default=([\d.]+)\)', _agsrc)
check(_temp is not None and f"Our open-weight agents, run at temperature ${_temp.group(1)}$, are weak" in _flat,
      "the limitations misstate the agents' sampling temperature")
# tab:ranking's caption: the noise-matched tolerance's draws
import ranking_check as _rcm  # noqa: E402
check("at its own pooled rates, $" + f"{_rcm.NOISE_DRAWS:,}".replace(",", "{,}") + "$ draws" in _flat,
      "tab:ranking's caption misstates the noise-matched tolerance's draws")
# app:replication: what was fixed before the analysis, the reply budget Qwen3-235B-A22B graded at, and
# tab:predictions' note on H2 against the plan's pass rule
check("Before the runs of the pre-specified test were analysed, we fixed the data, the option sets, the key groups, "
      "the grading rule, the statistics and a pass criterion for each hypothesis" in _flat,
      "app:replication misstates what was fixed before the pre-specified analysis")
check("Qwen3-235B-A22B, given an answer that no option is near, continued reasoning past a $4{,}096$-token reply rather "
      "than select one" in _flat, "app:replication misstates the reply budget the set-aside grader graded at")
check("$^{\\S}$The stated criterion passes it only by its $5$-point clause; the interval excludes $0$, so we count it "
      "as failed." in _flat
      and "a mean within 5 points of 0" in " ".join(_prereg[:6631].decode().split()),
      "tab:predictions' note on H2 no longer matches the plan's pass rule")
# sec:withdata: the range of tolerances the forced-choice excess is restated over is tab:restated's columns
_rsh = re.search(r"\\toprule\s*& \$(\d+)\\%\$ & \$(\d+)\\%\$ & \$5\\%\$ & \$(\d+)\\%\$", _table_body("tab:restated"))
check(_rsh is not None and (_rsh.group(1), _rsh.group(3)) == ("1", "10")
      and len(_rt_forced) == 4 * len([k for k in _rt if k.endswith("forced")]) and min(_rt_forced) > 0,
      "sec:withdata names tolerances tab:restated does not restate at")
# tab:readers2's and tab:readersfull's columns (sec:withdata's worked counts): the numeric items the with-data
# tables read
check("gemma-3-27b) on the $105$ numeric items through three option sets with the same key" in _flat
      and len(bw.option_sets()) == 105, "app:withdata miscounts the numeric items its readings cover")

# --- how a number is read (answer_extraction.py): the answers by kind and the rule's contrasts read four ways --------
import answer_extraction as _axm  # noqa: E402
_ax = load("results/answer_extraction.json")
_axc, _axk = _ax["contrasts"], _ax["categories"]
_kinds, _crows = _axm.table_rows(_ax)
_axb = " ".join(_table_body("tab:extraction").split())
check(all(" ".join(r.split()) in _axb for r in _kinds + _crows), "tab:extraction's rows no longer match answer_extraction.json")
_npk = _ax["non_positive_keys"]
check(f"($9$ of v1.5's $105$, $10$ of v1.0's $159$)" in _flat and (_npk["v1.5"], _npk["v1.0"]) == (9, 10),
      "app:replication miscounts the keys that are zero or negative")
_mv_ = "repaired-placebo|moved to an edge"
# "no other extraction gives a smaller contrast": nearly every Qwen3-235B-A22B answer that holds a number is that number
# (tab:extraction), so its last-number and whole-answer contrasts coincide
_larger_moved = all(_axc[b][f"{r}|{_mv_}"]["mean"] >= _axc[b][f"whole answer|{_mv_}"]["mean"] - 1e-6
                    for b in _axc for r in ("last number", "one number", "positive keys"))
_larger_all = all(_axc[b]["last number|repaired-placebo|all"]["mean"] >= _axc[b]["whole answer|repaired-placebo|all"]["mean"] - 1e-6
                  for b in _axc)
_clear = all(_axc[b][f"{r}|{_mv_}"]["lo"] > 0 for b in _axc if b != "v1.5|Qwen3-235B-A22B" for r in _axm.READINGS)
_q3zero = sum(_axc["v1.5|Qwen3-235B-A22B"][f"{r}|{_mv_}"]["lo"] <= 0.005 for r in _axm.READINGS)
check(_larger_moved and _larger_all and _clear and _q3zero == 3
      and "On the moved keys no other extraction gives a smaller contrast than the test's, nor over all items does "
          "the last number" in _flat
      and "whose lower end is at zero under three of the four extractions" in _flat,
      "app:replication's account of the four readings no longer holds")
_none26 = _axk["v1.0|gpt-4o"]["empty"] + _axk["v1.0|gpt-4o"]["no number"]
check(25 < _none26 < 100 / 3 and "more than a quarter of gpt-4o's published answers to numeric questions contain none"
      in _flat, f"sec:conclusion says more than a quarter of gpt-4o's published answers give no number; {_none26:.1f}% do")

# --- what the forced reading credits, in full (score_decomposition.py): each kind of miss, and the checks on the rest ---
_sdf = load("results/score_decomposition.json")
_pfg, _pfc = _sdf["v1.0"]["gpt-4o"]["published forced"], _sdf["v1.0"]["Claude 3.5 Sonnet"]["published forced"]
_fam = _sdf["v1.5"]["pooled"]["family forced"]
_nk = lambda s_, k: s_["named_on_a_miss"][k]["named"]
_wo = lambda s_, t, k: s_["other_misses"][t][k]
# tab:released's caption: the chance column is a random choice among four options forced and five with the refusal
# option, over the misses graded, as score_decomposition.miss_rates computes it for every row
# tab:released's corrected column is formula_scoring.py's correction for guessing, computed on the same runs as
# score_decomposition.py's rows: the same tolerance and the same share of correct answers accepted
_fsl = {(r["block"], r["group"], r["reading"]): r for r in load("results/formula_scoring.json")["rows"]}
_fsl_t1 = [(b, g, r) for b, g, r, _, _ in _sdm.LATEX if (b, g, r) in _fsl]
# the current agents' rows are frontier_agents.py's, as score_decomposition.table_rows reads them
_sdx = dict(_sdf, current={_a: _x["table1"] for _a, _x in load("results/frontier_agents.json")["agents"].items()})
check(len(_fsl_t1) == 9 and all("forced" in r for _, _, r in _fsl_t1)
      and all(abs(x["corrected"] - (x["score"] - 25) / 0.75) < 1e-9
              and abs(sum(q["points"] for q in x["parts"].values()) - x["corrected_minus_tolerance"]["mean"]) < 1e-6
              for x in _fsl.values())
      and all(abs(_fsl[k]["tolerance"] - _sdx[k[0]][k[1]][k[2]]["tolerance"]) < 1e-6
              and abs(_fsl[k]["score"] - _sdx[k[0]][k[1]][k[2]]["score"]) < 1e-6
              and abs(_fsl[k]["correct_accepted"] - _cacc(_sdx[k[0]][k[1]][k[2]])) < 1e-6
              and abs(_fsl[k]["misses_accepted"] - _sdx[k[0]][k[1]][k[2]]["miss_rates"]["all"]["accepted"]) < 1e-6
              for k in _fsl_t1)
      and "\\emph{Corrected}: the forced score corrected for guessing, $(S-1/4)/(3/4)$, minus the within-$5\\%$ share "
          "(Table~\\ref{tab:formula} splits it)" in _flat
      and "less the correct answers the grading rejects (\\emph{rej.}). The kinds are misses whose single nearest "
          "option is the key (\\emph{key}), another option (\\emph{other}), or neither (no number, or no single nearest "
          "option); an empty answer is scored wrong ungraded and adds nothing" in _flat
      and "which leaves a grader that accepts every correct answer and chooses at random on every miss at the tolerance"
          in _flat,
      "tab:released's corrected column is not the correction for guessing its caption describes, or its runs differ "
      "from score_decomposition.py's")
# tab:picks: what the published grades select on a miss with a single nearest option, and the paragraph around it
_pkb = " ".join(_table_body("tab:picks").split())
_pkrows = _sdm.picks_rows(_sdf)
_pubw = next(c for c in load("results/proximity_weight.json")["predicted_only"] if c["grader"] == "published grades")
# the published grades of each model alone (proximity_weight.py), quoted in the abstract, the introduction and sec:cost
_pubm = {c["grader"].split(", ", 1)[1]: c for c in load("results/proximity_weight.json")["predicted_only"]
         if c["grader"].startswith("published grades, ")}
_cfl = [c["proximity"]["lambda"] for k, c in load("results/grading_variants.json")["rank"].items()
        if k.startswith("v1.0, published|") and k.endswith(", codefree|original")]
check(sorted(_pubm) == ["Claude 3.5 Sonnet", "gpt-4o"]
      and min(_cfl) <= _pubm["Claude 3.5 Sonnet"]["lambda"] <= max(_cfl) and _pubm["gpt-4o"]["lambda"] < min(_cfl)
      and f"BixBench's own published grades exist only through $R$, but how often they select the nearest option "
          f"predicts what they would bear on the published runs' moved keys: ${_pubm['gpt-4o']['moved']['predicted']:+.1f}$ "
          f"for gpt-4o's grades, whose proximity weight is ${_pubm['gpt-4o']['lambda']:.2f}$, and "
          f"${_pubm['Claude 3.5 Sonnet']['moved']['predicted']:+.1f}$ for Claude 3.5 Sonnet's, whose "
          f"${_pubm['Claude 3.5 Sonnet']['lambda']:.2f}$ lies within the code-free graders' ${min(_cfl):.2f}$ to "
          f"${max(_cfl):.2f}$" in _flat
      and f"BixBench's own published grades cannot be regraded, but how often they select the nearest option predicts "
          f"${_pubm['gpt-4o']['moved']['predicted']:+.1f}$ on the moved keys for gpt-4o's and "
          f"${_pubm['Claude 3.5 Sonnet']['moved']['predicted']:+.1f}$ for Claude 3.5 Sonnet's." in _flat
      and "although BixBench's published Claude 3.5 Sonnet grades follow the nearest option as closely as the former"
          in _flat,
      "the published grades' per-model proximity weights or predicted costs are misquoted")
for _row in _pkrows:
    check(" ".join(_row.split()) in _pkb, f"tab:picks should carry the row {_row}")
check(_pkb.count("\\\\") == len(_pkrows) + 2, "tab:picks should hold exactly the rows score_decomposition.picks_rows gives")
_pkg2, _pkc2 = _pfg["picks_on_misses"], _pfc["picks_on_misses"]
_nos = lambda d, k: d[k]["nearest_of_selecting"]
check(all(27.5 <= _pkg2[k]["none"] <= 32.5 and _pkc2[k]["none"] < 1 for k in ("key", "other"))
      and "gpt-4o's grades select no option on about $30\\%$ of these misses, which BixBench's parser scores as wrong, "
          "and Claude's on fewer than $1\\%$." in _flat
      and f"Of the grades that select an option, gpt-4o's select the nearest on ${_nos(_pkg2, 'key'):.1f}$ and "
          f"${_nos(_pkg2, 'other'):.1f}\\%$, as it is or is not the key, and Claude's on ${_nos(_pkc2, 'key'):.1f}$ and "
          f"${_nos(_pkc2, 'other'):.1f}\\%$." in _flat
      and f"the published grades select the nearest option on ${100 * _pubw['q']:.1f}\\%$ of the misses on which they "
          f"select one and no option on ${100 * _pubw['delta']:.1f}\\%$, a proximity weight of "
          f"${_pubw['lambda']:.2f}$ (Table~\\ref{{tab:readers2}})" in _flat
      and abs(_pubw["lambda"] - (1 - _pubw["delta"]) * (4 * _pubw["q"] - 1) / 3) < 1e-9
      and f"; gpt-4o's grades alone have ${_pubm['gpt-4o']['lambda']:.2f}$ and Claude's "
          f"${_pubm['Claude 3.5 Sonnet']['lambda']:.2f}$, which predict ${_pubm['gpt-4o']['moved']['predicted']:+.1f}$ and "
          f"${_pubm['Claude 3.5 Sonnet']['moved']['predicted']:+.1f}$ points on the published runs' moved keys under $U$ "
          f"against $P$." in _flat
      and f"The seven configurations' own graders accept ${_mr(_fam, 'nearest'):.1f}\\%$ of their misses nearest the "
          f"key and ${_mr(_fam, 'other'):.1f}\\%$ of those nearest another option." in _flat
      and "its first column on the key's rows and its second on the other rows are Table~\\ref{tab:released}'s shares "
          "of misses accepted" in _flat,
      "app:replication's account of what the published grades select on a miss is stale")
check(f"the key's value appears in the notebook for ${_wo(_pfg, 'written', 'key'):.1f}$ and "
      f"${_wo(_pfc, 'written', 'key'):.1f}\\%$ of runs and a distractor's for ${_wo(_pfg, 'written', 'distractor'):.1f}$ and "
      f"${_wo(_pfc, 'written', 'distractor'):.1f}\\%$, and the grades select the key on "
      f"${_wo(_pfg, 'written', 'named_if_key'):.1f}$ and ${_wo(_pfc, 'written', 'named_if_key'):.1f}\\%$ of the runs where it "
      f"appears against ${_wo(_pfg, 'written', 'named_if_not'):.1f}$ and ${_wo(_pfc, 'written', 'named_if_not'):.1f}\\%$ "
      f"where it does not" in _flat
      and f"lands within $5\\%$ of the key for ${_wo(_pfg, 'rescaled', 'key'):.1f}$ and ${_wo(_pfc, 'rescaled', 'key'):.1f}\\%$ "
          f"of them and of a distractor for ${_wo(_pfg, 'rescaled', 'distractor'):.1f}$ and "
          f"${_wo(_pfc, 'rescaled', 'distractor'):.1f}\\%$" in _flat
      and f"where its value appears in the notebook (${_wo(_fam, 'written', 'named_if_key'):.1f}$ against "
          f"${_wo(_fam, 'written', 'named_if_not'):.1f}\\%$), on ${round(_wo(_fam, 'written', 'key') * _fam['other_misses']['n_runs'] / 100)}$ "
          f"of ${_fam['other_misses']['n_runs']}$ such runs" in _flat,
      "app:replication's checks on the credits of misses nearer another option are stale")
import score_decomposition as _sdm2  # noqa: E402
check([name for name, _ in _sdm2.RESCALES] == ["x100", "/100", "x1000", "/1000", "negated", "reciprocal", "2^a", "log2",
                                                "10^a", "log10", "e^a", "ln"]
      and "multiplied or divided by $100$ or $1{,}000$, negated, inverted, or exponentiated or logged in base $2$, $e$ "
          "or $10$" in _flat,
      "app:replication lists other rescalings than score_decomposition.RESCALES tries")

# --- who pays for a hidden rank (run_set_scaling.py) -------------------------------------------------------------
import run_set_scaling as _rssm  # noqa: E402
_rss = load("results/run_set_scaling.json")
_tr = _rss["trend"]
_sa = load("results/strong_agent.json")
_sci = lambda e: f"${e['mean']:+.1f}$ $[{e['lo']:+.1f},{e['hi']:+.1f}]$"
_scb = " ".join(_table_body("tab:scaling").split())
check(all(" ".join(r.split()) in _scb for r in _rssm.table_rows(_rss)) and len(_rss["run_sets"]) == 19,
      "tab:scaling's rows no longer match run_set_scaling.json, or it no longer holds nineteen run sets")
_rsets = sorted(_rss["run_sets"].values(), key=lambda r: r["within_5pct"])
_top_all = max(_rss["run_sets"].values(), key=lambda r: r["all"])
_q3s = [v for k, v in _rss["run_sets"].items() if "qwen3-235b" in k]
_pmd = load("results/reference_check.json")["per_model"]
check(f"Over the nineteen configurations with the data, the rule's gain over all items decreases as agents "
      f"land within $5\\%$ of the key more often (Spearman "
      f"$\\rho={_tr['all on within_5pct']['spearman_rho']:.2f}$; ${_pmd['all']['spearman_rho']:.2f}$ with each "
      f"of the eight models counted once; Table~\\ref{{tab:scaling}}), to "
      f"${_sa['rule|repaired-placebo|all']['mean']:+.1f}$, $+1.0$" in _flat
      and len(_rss["run_sets"]) == 19 and not any("gpt-5.1" in k for k in _rss["run_sets"])
      and "by exact permutation: only the decrease over all items, which fewer misses imply, survives." in _flat
      and _pmd["moved"]["p_value"] >= 0.05 and _pmd["moved_per_miss"]["p_value"] >= 0.05
      and _pmd["all"]["n_models"] == 8 and _pmd["all"]["p_value"] < 0.05 and _pmd["moved_per_miss"]["p_value"] >= 0.05
      and f"the correlations are ${_pmd['moved']['spearman_rho']:.2f}$ on the moved keys "
          f"($p={_pmd['moved']['p_value']:.2f}$), ${_pmd['all']['spearman_rho']:.2f}$ over all items "
          f"($p={_pmd['all']['p_value']:.3f}$) and ${_pmd['moved_per_miss']['spearman_rho']:.2f}$ per miss "
          f"($p={_pmd['moved_per_miss']['p_value']:.2f}$), by exact permutation" in _flat
      and all(abs(v["within_5pct"] - _q3s[0]["within_5pct"]) < 0.05 for v in _q3s)
      and _tr["moved on within_5pct"]["p_value"] < 0.05 and _tr["all on within_5pct"]["p_value"] < 0.05,
      "sec:withdata's account of who pays is stale")
_pm = _rss["moved_per_miss_range"]
check(f"(Spearman $\\rho={_tr['moved on within_5pct']['spearman_rho']:.2f}$, permutation "
      f"$p={_tr['moved on within_5pct']['p_value']:.3f}$), over all items it decreases faster "
      f"($\\rho={_tr['all on within_5pct']['spearman_rho']:.2f}$, $p<0.001$), and per miss on a moved key it also decreases "
      f"($\\rho={_tr['moved_per_miss on within_5pct']['spearman_rho']:.2f}$, "
      f"$p={_tr['moved_per_miss on within_5pct']['p_value']:.3f}$)" in _flat
      and _tr["all on within_5pct"]["p_value"] < 0.001
      and f"Per miss it still ranges from ${_pm[0]:+.1f}$ to ${_pm[1]:+.1f}$" in _flat
      and max(_rss["run_sets"].items(), key=lambda kv: kv[1]["moved_per_miss"])[0].startswith("v1.0|claude")
      and min(v["moved_per_miss"] for k, v in _rss["run_sets"].items() if "claude" in k)
          > max(v["moved_per_miss"] for k, v in _rss["run_sets"].items() if "4o_" in k),
      "app:replication's account of who pays is stale")
_dgr = load("results/degenerate_runs.json")
_dpm = _dgr["per_miss"]["correlations"]
_dpa, _dpk = _dpm["as graded|misses_accepted"]["configurations"], _dpm["as graded|key_nearest_share_of_misses"]["configurations"]
check(max(r["within_5pct"] for r in _rss["run_sets"].values() if not r is None) < 25
      and len(_dgr["per_miss"]["configurations"]) == _dpa["n"] == _dpk["n"] == 22
      and all(sum(c["model"] == m for c in _dgr["per_miss"]["configurations"]) == 1
              for m in ("gpt-5.1", "gpt-6-luna", "DeepSeek-V4-Pro"))
      and f"What forced choice adds scales with the misses, so it shrinks as agents improve, but not per miss. "
          f"Over the ${_dpa['n']}$ configurations with the data, the share of misses their forced graders "
          f"accept does not fall as agents land within $5\\%$ of the key more often (Spearman "
          f"$\\rho={_dpa['rho']:+.2f}$, $p={_dpa['p']:.2f}$), and the share of misses nearest the key rises "
          f"($\\rho={_dpk['rho']:+.2f}$, $p={_dpk['p']:.3f}$)." in _flat
      and _dpa["p"] >= 0.05 and _dpk["p"] < 0.05 and _dpk["rho"] > 0,
      "the limitations' account of what forced choice adds per miss as agents improve no longer matches "
      "degenerate_runs.json, or one of our agents now lands within 5% on a quarter of questions")
# --- a closed agent with the data (strong_agent.py, tab:strong; score_decomposition.py's gpt-5.1 rows) ----------
# gpt-5.1 as BixBench's published agent, graded by gpt-4o; outside the pre-specified test. tab:released prints its
# forced and may-decline rows (checked above with the rest of that table); tab:strong the rest.
import numpy as _np  # noqa: E402
import replication as _repm  # noqa: E402
import strong_agent as _sam  # noqa: E402
_sab = " ".join(_table_body("tab:strong").split())
for _row in _sam.table_rows(_sa):
    check(" ".join(_row.split()) in _sab, f"tab:strong should carry the row {_row}")
check(_sab.count("\\\\") == len(_sam.table_rows(_sa)) + 1,
      "tab:strong should hold exactly the rows strong_agent.table_rows gives")
_s51 = load("results/score_decomposition.json")["v1.5 closed"]["gpt-5.1"]
_s51f, _s51d = _s51["gpt-4o forced"], _s51["gpt-4o may-decline"]
_s51x = _s51f["score_minus_tolerance"]
_ssets = bw.option_sets()
_sxs = _np.array([v["within_5pct"] for v in _rss["run_sets"].values()])
_sys5 = _np.array([v["all"] for v in _rss["run_sets"].values()])
_ssl, _sic = _np.polyfit(_sxs, _sys5, 1)
check(_sa["served"] == ["gpt-5.1-2025-11-13"] and _sa["n_runs"] == 210 and _sa["n_items"] == 105
      and _sa["submitted"] == 100.0 and _s51f["n_runs"] == 210 and _s51d["n_runs"] == 210
      and abs(_sa["within_5pct"]["mean"] - _s51f["tolerance"]) < 0.05
      and abs(_sa["who_pays"]["within_5pct"] - _s51f["tolerance"]) < 0.05
      and _s51f["tolerance"] > max(r["within_5pct"] for r in _rss["run_sets"].values())
      and _s51f["tolerance"] > max(_w5),
      "gpt-5.1's runs are not the 210 of version 2025-11-13, each submitted, or their share within 5% disagrees "
      "between reports, or no longer exceeds every open-weight configuration's")
check("trend predicts" not in _flat, "sec:withdata extrapolates the trend to gpt-5.1")
check("only Qwen2.5-72B among the text-protocol models measurably uses the data" in _flat,
      "the limitations misstate gpt-5.1's share within 5% of the key, or which text-protocol model uses the data")
# sec:conclusion's recommendation restates Proposition 1(ii): a uniform rank leaves 2/k of the keys extreme
check("a rank drawn uniformly leaves $2/k$ of the keys at an extreme in expectation" in _flat
      and "among $k$ options, a key rank drawn uniformly leaves $2/k$ of the keys extreme in expectation" in _flat
      and "makes every $p_j=1/k$ and leaves $2/k$ of the keys at the extremes" in _flat,
      "sec:conclusion's share of extreme keys under a uniform rank no longer matches Proposition 1")
check("\\paragraph{A closed agent.} Outside the pre-specified test, gpt-5.1 (version 2025-11-13, medium reasoning effort) "
      "ran BixBench's published agent with the data, twice on each of v1.5's $105$ numeric questions, and gpt-4o "
      "(version 2024-11-20), one of BixBench's two graders, graded every run through $R$, $P$ and $U$, "
      "forced and with a refusal option, and open-ended" in _flat
      and all(_t.get("settings", {}).get("reasoning_effort") == "medium" and _t["settings"]["rollouts"] == 2
              for _t in map(json.loads, gzip.open("results/strong_agent/trajectories_gpt-5.1-react.jsonl.gz", "rt")))
      and re.search(r'models = \{\s*"4o": "gpt-4o",\s*"claude": "claude-3-5-sonnet-20241022",\s*\}',
                    Path("sources/bixbench_49311180/postprocessing_utils.py").read_text()) is not None,
      "app:replication's account of how gpt-5.1 ran and was graded is stale")
check(f"Every run submitted an answer. Its number is within $5\\%$ of the key on ${_s51f['tolerance']:.1f}\\%$ of items, "
      f"more often than any configuration of Table~\\ref{{tab:scaling}}; ${_sa['no_number_runs']}$ of its "
      f"${_sa['n_runs']}$ answers give no number, ${_sa['no_number_declines']}$ of them stating that the value cannot be "
      f"determined, and forced-choice grading accepts ${_s51f['miss_rates']['no number']['accepted']:.1f}\\%$ of those, "
      f"${_s51f['miss_credited_no_number']:.1f}$ points of its score. The rule's $U-P$ is "
      f"${_sa['rule|repaired-placebo|moved']['mean']:+.1f}$ on its moved keys and "
      f"${_sa['rule|repaired-placebo|all']['mean']:+.1f}$ over all items, and gpt-4o's, forced, "
      f"${_sa['gpt-4o|repaired-placebo|moved']['mean']:+.1f}$ on the moved keys." in _flat
      and _sa["no_number_declines"] * 2 > _sa["no_number_runs"],
      "app:replication's paragraph on gpt-5.1 is stale")
check("v1.0: the published runs, each model grading its own; v1.5: the seven configurations, graded by their own models "
      "(Qwen3-30B-A3B's by gemma-3-27b) and by gemma-3-27b for all, the later runs by gemma-3-27b, and "
      f"{_sam.MODEL.removesuffix('-react')} and the current agents by {_sam.READER}.}}" in _flat
      and _repm is not None and "qwen3a3b-react" in _sdm.br.RUNS and _sdm.br.primary("qwen3a3b-react") == "gemma27b",
      "tab:released's caption misnames who graded which runs")
_s5name = _sam.MODEL.removesuffix("-react")
check(f"\\caption{{{_s5name} as BixBench's published agent with the data, two runs on each of v1.5's "
      f"${_sa['n_items']}$ numeric questions, graded by {_sam.READER}: in per cent, and $U-P$ in points;" in _flat
      and f"\\toprule & {_s5name}\\\\ \\midrule" in _flat and _sa["n_runs"] == 2 * _sa["n_items"],
      "tab:strong's caption or header misnames the agent or its grader, or miscounts its questions")
check(f"with plain $95\\%$ cluster-bootstrap intervals over its ${_sa['within_5pct']['n_clusters']}$ capsules" in _flat
      and _sa["level"] == 0.95 and _sa["n_moved"] == sum(1 for q in {r for r in _ssets}
                                                          if _repm.group_of(_ssets[q]) == "moved to an edge"),
      "tab:strong's caption misstates its intervals, or gpt-5.1's moved keys are not v1.5's")
# sec:open's fitted reader
_lp = load("results/learned_probe.json")
check("selects the key on $43.8\\%$ of the released items" in _flat
      and "selects the key on $43.8\\%$ of the items (Appendix" not in _flat,
      "app:replication's fitted reader is misquoted")

# --- the in-context panel's design, as the text states it ---------------------------------------------------------
from probe_analysis import PARAMETERS_B as _pb  # noqa: E402
_icl_shots = max(int(re.search(r"_s(\d+)_", arm).group(1)) for r in icl["models"] for arm in r["arms"])
_icl_sizes = sorted(_pb[r["model"]] for r in icl["models"])
_icl_items = {(a["n_items"], a["n_clusters"]) for r in icl["models"] for a in r["arms"].values()}
check(len(_icl_items) == 1, f"the in-context arms read different items: {_icl_items}")
(_icl_n, _icl_c), = _icl_items
check(f"Eight open-weight models, ${_icl_sizes[0]:.1f}$ to ${int(_icl_sizes[-1])}$B, were shown up to ${_icl_shots}$ solved "
      f"v1.5 items" in _flat
      and len(_icl_sizes) == 8
      and f"on v1.5's ${_icl_n}$ released numeric items (${_icl_c}$ capsules)" in _flat
      and f"with no examples and with ${_icl_shots}$ solved items from other capsules" in _flat
      and f"released minus uniform at ${_icl_shots}$ examples, which isolates the rank" in _flat
      and f"released $-$ uniform, ${_icl_shots}$" in _flat,
      "the in-context panel's sizes, examples or items no longer match results/icl_probe.json")

# --- what a reading does with a number, as sec:withdata and app:replication say it -------------------------------
from answer_numbers import graded as _graded  # noqa: E402
import answer_extraction as _axm  # noqa: E402
_rsn = [name for name, _ in _sdm2.RESCALES]
check({"x100", "/100", "x1000", "/1000", "negated", "reciprocal"} <= set(_rsn)
      and any(n.startswith("log") or n == "ln" for n in _rsn)
      and "multiplied or divided by $100$ or $1{,}000$, negated, inverted, or exponentiated or logged in base $2$, $e$ "
          "or $10$" in _flat,
      "sec:withdata's slips of scale are not the ones score_decomposition.RESCALES tries")
check(_graded("0.35", "35%", 0.05) and not _graded("0.35", "35", 0.05) and _axm.last_number("0.35", "35%") == 35.0
      and "treats an answer written as a fraction against a key written as a per cent as that fraction times $100$"
          in _flat,
      "app:replication's account of the tolerance's per-cent reading no longer matches free_response.graded")

# --- the three sources cited for BixBench's use in 2026, against verbatim excerpts in sources/cited/ -------------
_ct = {n: " ".join(Path("sources/cited", n).read_text(encoding="utf-8").split()) for n in (
    "bioagents_arxiv_2601.12542.txt", "zhang2026_biorxiv_2026.04.28.721523.txt", "verified50_dataset_card.txt")}
_ba, _zh, _v5 = _ct.values()
_bibtext = Path("refs.bib").read_text(encoding="utf-8")
_year = lambda key: re.search(r"@\w+\{" + key + r",.*?year\s*=\s*\{(\d{4})\}", _bibtext, re.S).group(1)
_open = re.search(r"([\d.]+)% accuracy on Open Response", _ba).group(1)
_forced = re.search(r"([\d.]+)% on MCQ without Refusal", _ba).group(1)
check(f"The forced-choice score is still reported in {_year('bioagents')}." in _flat
      and f"a {_year('bioagents')} system reports ${_forced}\\%$ forced-choice against ${_open}\\%$ open-ended accuracy on "
          "v1.5, graded by a model that sees the answer and the options but not the notebook, and sets it beside "
          "published forced-choice scores graded with the notebook \\citep{bioagents}" in _flat
      and "BixBench version v1.5" in _ba and "excluding the analysis notebook" in _ba
      and "outperforming GPT-4o (" in _ba and "Claude (" in _ba
      and 40 <= float(_open) <= 60
      and f"The correction maps the {_year('bioagents')} system's ${_forced}\\%$ to ${(float(_forced) - 25) / 0.75:.1f}\\%$, "
          f"${(float(_forced) - 25) / 0.75 - float(_open):.1f}$ points above its open-ended ${_open}\\%$" in _flat,
      "the intro's account of BixBench's 2026 use no longer matches sources/cited/bioagents_arxiv_2601.12542.txt")
_zs = [float(x) for x in re.findall(r"([\d.]+)% \(\d+/50\)", _zh)]
check(len(_zs) == 3 and f"frontier agents with added skills now answer ${min(_zs):.0f}$ to ${max(_zs):.0f}\\%$ of them "
      "\\citep{zhang2026}" in _flat and "on the 50 verified questions" in _zh,
      "sec:related's Verified-50 scores no longer match sources/cited/zhang2026_biorxiv_2026.04.28.721523.txt")
_words = {50: "fifty"}
_vq = int(re.search(r"Total questions (\d+)", _v5).group(1))
_vr = int(re.search(r"range_verifier \((\d+)\)", _v5).group(1))
check(f"BixBench-Verified-50 re-checks {_words[_vq]} of its questions with domain experts and grades the open answer, "
      f"${_vr}$ by a numeric range, keeping the distractors \\citep{{verified50}}" in _flat
      and "several domain experts" in _v5 and "distractors list Incorrect answer choices" in _v5,
      "sec:related's account of BixBench-Verified-50 no longer matches sources/cited/verified50_dataset_card.txt")

# --- survivors of a one-digit mutation of the reworded text, each pinned in context against its source ----------
# the concurrent audit's replicated runs, against the vendored Zenodo record
_au = " ".join(Path("sources/cited/bixbenchaudit_zenodo_22151932.txt").read_text(encoding="utf-8").split())
_aun = re.search(r"(\d+) replicated agent runs", _au)
_auq = re.search(r"on three capsules \((\d+) questions", _au)
check(_aun is not None and _auq is not None
      and f"re-derives the answer keys of three capsules (${_auq.group(1)}$ questions) from the raw data and grades "
          f"${_aun.group(1)}$ replicated agent runs on them" in _flat,
      "sec:related's count of the audit's replicated runs no longer matches "
      "sources/cited/bixbenchaudit_zenodo_22151932.txt")
# every caption that counts v1.5's numeric items, each by its own continuation, and tab:bracket's header
_n15 = len(bw.option_sets())
_nmv15 = load("results/reader_split.json")["n_items"]["moved to an edge"]
for _cap in (f"The forced-choice condition of Table~\\ref{{tab:free}} on BixBench's ${_n15}$ numeric items through",
             f"the submitted number within $5\\%$ of the key on the ${_n15}$ numeric items, as in Table~\\ref{{tab:free}}",
             f"The last four columns grade the \\emph{{same}} runs on the ${_n15}$ numeric items through",
             f"The same runs graded three ways, with the data: $R-U$ on the ${_n15}$ numeric items",
             f"The same with-data runs, graded through the three option sets on the ${_n15}$ numeric items.",
             f"all ${_n15}$ & moved (${_nmv15}$) & other (${_n15 - _nmv15}$)"):
    check(_cap in _flat, f"a caption or header miscounts v1.5's numeric items: {_cap[:70]}")
# the agent's version, against the prompts bixbench_agent.py vendors
_agv = re.search(r"data-analysis-crow v(\d+\.\d+\.\d+)", _agsrc)
check(_agv is not None and f"at the version the benchmark specifies ({_agv.group(1)})" in _flat,
      "app:withdata misstates the agent's version")
# tab:scaling's caption names the new seeds as its rows do
_sdl = sorted(v.split()[-1] for v in _rssm.SEEDS.values())
check(f"seeds ${_sdl[0]}$ and ${_sdl[1]}$ are the new seeds of the pre-specified test" in _flat,
      "tab:scaling's caption names other seeds than its rows")
# app:replication's reading of the v1.0 grader's acceptance curve, and of tab:restated
check("the tolerance it applies, whatever it was instructed, lies between $2$ and $10\\%$" in _flat
      and _acc["0.01-0.02"] > 0.5 > _acc["0.02-0.05"] and _acc["0.1-0.2"] < 0.05
      and _rsh is not None
      and f"restates every tolerance-dependent result of the paper at ${_rsh.group(1)}$, ${_rsh.group(2)}$ and "
          f"${_rsh.group(3)}\\%$. The forced-choice excess is positive at every tolerance on every set of runs, but at "
          f"$10\\%$ the intervals of gpt-5.1 and gpt-6-luna reach zero; the refusal option's excess excludes zero at $1$ and "
          f"$2\\%$ on the published runs and the seven configurations, and at $1\\%$ on gpt-6-luna's runs" in _flat
      and "Every tolerance-dependent result is restated at $1$, $2$ and $10\\%$ (Table~\\ref{tab:restated}): the "
          "forced-choice excess stays positive at each, though at $10\\%$ the intervals of gpt-5.1 and gpt-6-luna reach "
          "zero; the refusal option's small excess excludes zero at $1$ and $2\\%$ on the published runs and the seven "
          "configurations; and Qwen2.5-72B's lead over Llama-3.3-70B is significant only at $5\\%$." in _flat
      # which verdicts change, read off tab:restated's source: a verdict is whether the interval excludes zero
      and all(_rr3a[k][t]["mean"] > 0 for k in _rr3a if k.endswith("forced") for t in ("1%", "2%", "5%", "10%"))
      and sorted(k for k in _rr3a if k.endswith("forced") and _rr3a[k]["10%"]["lo"] <= 0)
          == ["v1.5|gpt-5.1|forced", "v1.5|gpt-6-luna|forced"]
      and all(_rr3a[k][t]["lo"] > 0 for k in _rr3a if k.endswith("forced") for t in ("1%", "2%", "5%"))
      and {t: sorted(k for k in _rr3a if k.endswith("may decline") and _rr3a[k][t]["lo"] > 0) for t in ("1%", "2%", "5%", "10%")}
          == {"1%": ["v1.0|claude|may decline", "v1.0|gpt-4o|may decline", "v1.5|gpt-6-luna|may decline",
                     "v1.5|seven|may decline"],
              "2%": ["v1.0|claude|may decline", "v1.0|gpt-4o|may decline", "v1.5|seven|may decline"],
              "5%": [], "10%": []}
      and all(_t3["withdata"][t][k]["lo"] > 0 for t in ("1%", "2%", "5%", "10%")
              for k in ("qwen72b data - nodata", "qwen72b - gemma27b"))
      and _w10["qwen72b - llama70b"]["lo"] <= 0,
      "app:replication's reading of the v1.0 grader's tolerance or of tab:restated is stale")

# --- app:withdata: how P and U are built, against the generator (mcq_audit.redraw_row) and a regeneration --------
import contextlib as _ctl  # noqa: E402
import inspect as _insp  # noqa: E402
import io as _io  # noqa: E402
import mcq_audit as _mqa  # noqa: E402
_rdsrc = _insp.getsource(_mqa.redraw_row)
_arsrc = _insp.getsource(_mqa.apply_repair)
_jl = lambda f: [json.loads(_l) for _l in open(f)]


def _rewrite(src, seed, preserve_rank=False):
    with _ctl.redirect_stdout(_io.StringIO()), _ctl.redirect_stderr(_io.StringIO()):
        rows = _mqa.read_rows(Path(src))
        items, k = _mqa.build_items(rows, "ideal", "distractors", "capsule_uuid", "question", None)
        return _mqa.apply_repair(rows, items, "distractors", seed, k=k, preserve_rank=preserve_rank)


_u15 = _rewrite("build/bixbench_numeric_q.jsonl", 20260918)
_p15 = _rewrite("build/bixbench_numeric_q.jsonl", 20260920, preserve_rank=True)
_u10 = _rewrite("build/bixbench_v10_items.jsonl", 20260918)
_prng, _p10, _p10miss = random.Random(20260920), [], 0
for _row in _jl("build/bixbench_v10_numeric_released.jsonl"):
    _dr = _mqa.placebo_row([_row["ideal"], *_row["distractors"]], _prng)
    if _dr is None:
        _p10miss += 1
    else:
        _p10.append(_dr)
_same = lambda got, f: [r["distractors"] for r in got[0]] == [r["distractors"] for r in _jl(f)]
check(_same(_u15, "build/bixbench_numeric_repaired.jsonl") and _same(_p15, "build/bixbench_numeric_placebo.jsonl")
      and _same(_u10, "build/bixbench_v10_repaired.jsonl")
      and _p10 == [[r["ideal"], *r["distractors"]] for r in _jl("build/bixbench_v10_numeric_placebo.jsonl")]
      and "Each rewrite is one draw at a fixed seed, and regenerating the four option files reproduces them exactly."
          in _flat,
      "the rewritten option sets no longer regenerate from mcq_audit's generator at their seeds")
check(f"$U$ tries the other ranks in random order (${_u15[3]}$ of v1.5's ${_u15[1]}$ items and ${_u10[3]}$ of v1.0's "
      f"${_u10[1]}$ took a rank other than the one drawn), and $P$ leaves the item out (${_p10miss}$ of v1.0's)" in _flat
      and _p15[3] == 0 and not _u15[2] and not _u10[2] and len(_p10) + _p10miss == _u10[1]
      and f"$U$ redrew all ${_u10[1]}$ option sets (${_u10[3]}$ could not take the rank drawn for them and took "
          f"another); $P$ redrew ${len(_p10)}$" in _flat,
      "app:withdata's counts of items that took another rank, or were left out, are stale")
check("rng.uniform(0.6, 1.4)" in _rdsrc and "max(relative, 0.05)" in _rdsrc and "range(24)" in _rdsrc
      and "attempt < 12" in _rdsrc and "(1.0 + 0.5 * attempt)" in _rdsrc and "(1.0 + 0.5 * (attempt % 12))" in _rdsrc
      and "len(set(parsed) | {key}) < k" in _rdsrc and "format_like(t, v, match_digits)" in _rdsrc
      # the digit-matched rewrites: styles and drawn values paired by size, and a draw whose digits differ discarded
      and "drawn = sorted(drawn, key=abs)" in _rdsrc
      and "significant_digits(r) != significant_digits(t)" in _rdsrc
      and "and the drawn values are assigned to the released distractors in order of size" in _flat
      and "target = rng.randrange(k)" in _rdsrc and "rank_of(options, 0)" in _insp.getsource(_mqa.placebo_row)
      and "[target] + [r for r in rng.sample(range(k), k) if r != target]" in _arsrc
      and "where $u$ is drawn uniformly from $[0.6,1.4]$ for each distractor" in _flat
      and "and at least $0.05$; otherwise it places them at $y\\pm Sju$" in _flat
      and "the generator makes up to $24$ draws, the first twelve multiplicative when the values share a sign" in _flat
      and "and the whole set drawn again with the spread widened by half its initial value" in _flat
      and f"a median $d$ of ${_px['placebo']['median']:.3f}$ in $P$ and ${_px['repaired']['median']:.3f}$ in $U$, against "
          f"${_px['released']['median']:.3f}$, over v1.5's $105$" in _flat,
      "app:withdata's account of the generator no longer matches mcq_audit.redraw_row")
# names and formulas the mutation check found unpinned, each in its own context against the files that define it
_rpl5 = load("results/replication.json")
check("Before analysing runs that no earlier analysis had used (the published runs, and new v1.5 runs made "
      "afterwards), we fixed six hypotheses about the nearest-option rule's contrasts" in _flat
      and sorted(_rpl5["D1"]["n_numeric_runs"]) == sorted(("4o_open_image", "4o_open_no_image", "claude_open_image",
                                                           "claude_open_no_image"))
      and "of Qwen3-235B-A22B, and two new seeds of each text-protocol agent" in _flat
      and sorted(_rpl5["D2"]["qwen3-235b|data"]["n_trajectories"]) == ["qwen3-235b|r0", "qwen3-235b|r1"],
      "sec:withdata names other runs for the pre-specified test than replication.json holds")
check("Qwen2.5-72B and Llama-3.3-70B graded only the new seeds' moved keys." in _flat
      and all(load(f)["D2|reruns|data"].get("groups_read") == ["moved to an edge"]
              for f in ("results/published_reads_qwen72b_forced.json", "results/published_reads_llama70b_forced.json")),
      "tab:readers2's caption misstates which keys Qwen2.5-72B and Llama-3.3-70B graded on the new seeds")
check("The split into moved and unchanged keys and the paper's claim family were chosen after the v1.5 results" in _flat
      and "v1.0: the published runs, each model grading its own; v1.5: the seven configurations" in _flat,
      "the limitations or tab:released's caption name another release")
check("Outside the pre-specified test, gemma-3-27b graded every run of the test" in _flat
      and load("results/published_reads.json")["reader"] == "gemma27b"
      and "gemma-3-27b also graded with BixBench's ``insufficient information'' option added" in _flat
      and load("results/published_reads_gemma27b_decline.json")["reader"] == "gemma27b",
      "app:replication names another grader for the test's runs or for grading with refusal")
check("at a magnitude of $|y|(1+sju)$ or $|y|/(1+sju)$ with the key's sign" in _flat
      and "magnitude / (1 + step * (i + 1) * rng.uniform(0.6, 1.4))" in _rdsrc
      and "magnitude * (1 + step * (i + 1) * rng.uniform(0.6, 1.4))" in _rdsrc,
      "app:withdata's formula for the new distractors no longer matches mcq_audit.redraw_row")
# app:replication: what gpt-4o 2024-11-20's replies without a letter hold (grader_declines.py, from the recorded replies)
_gdr = load("results/grader_declines.json")["replies"]
check(_gdr is not None and _gdr["no_letter"] > 0.4 * _gdr["n_replies"]
      and f"Of gpt-4o 2024-11-20's replies without a letter, ${_gdr['empty or text']:.1f}\\%$ give an empty answer or a "
          f"word such as None and ${_gdr['a number']:.1f}\\%$ a number instead of a letter; "
          f"${_gdr['says no option matches']:.1f}\\%$ say that no option matches the answer." in _flat
      and _gdr["says no option matches"] > 75,
      "app:replication's account of gpt-4o's replies without a letter is stale")

# --- the gradings of grading_variants.py and proximity_weight.py: every number the text states --------------------
_gv = load("results/grading_variants.json")
_pwj = load("results/proximity_weight.json")
_inf10, _rk = _gv["inflation"]["v1.0"], _gv["rank"]
_G4, _CL = "gpt-4o", "Claude 3.5 Sonnet"
_pubi = lambda m: _inf10[f"{m}|published, with the notebook"]
_cfi = lambda m, g, mode="forced": _inf10[f"{m}|code-free {g}, {mode}"]
_ex = lambda b, part: b[part]["excess"]
_f1 = lambda x: f"{x:.1f}"
_ci2 = lambda v: f"${v['mean']:+.1f}$ $[{v['lo']:+.1f},{v['hi']:+.1f}]$"
_rkm = lambda fam, g: _rk[f"{fam}|{g}|original"]["moved to an edge"]
# sec:withdata: the questions whose options are not numbers, and the code-free graders over every question
check(f"The excess is not confined to numbers: on the questions whose options are not numbers, the published "
      f"forced grades exceed the open-ended ones by $+{_f1(_ex(_pubi(_G4), 'other')['mean'])}$ and "
      f"$+{_f1(_ex(_pubi(_CL), 'other')['mean'])}$ points, against $+{_f1(_pubi(_G4)['other']['at_chance'])}$ "
      f"and $+{_f1(_pubi(_CL)['other']['at_chance'])}$ at chance (Table~\\ref{{tab:allq}})." in _flat
      and all(_ex(_pubi(m), "other")["lo"] > 0 for m in (_G4, _CL))
      and [(_pubi(m)[part]["n_items"]) for m in (_G4, _CL) for part in ("all", "numeric", "other")]
      == [289, 158, 131, 296, 159, 137]
      and "v1.0: $296$ questions, $159$ of them numeric ($289$ and $158$ with gpt-4o's runs); v1.5: $205$, $105$ "
          "numeric." in _flat,
      "the account of the questions whose options are not numbers no longer matches grading_variants.json")
# ... and in full (nonnumeric_excess.py): the kinds of option, what the grader accepts by nearest option, and what
# it supplies on an answer nearest none
_nne = load("results/nonnumeric_excess.json")
_nn10, _nn15 = _nne["v1.0"], _nne["v1.5"]
_nnp = {m: _nn10["nearest"][f"{m}|published"] for m in (_G4, _CL)}
_nnacc = lambda m, part, k: _nnp[m][part][k]["accepted"]
_nnd = lambda m, k: _nnp[m]["no-data answer"][k]["accepted"]
_nnci = lambda a: f"${a['share']:.1f}$ $[{a['lo']:.1f},{a['hi']:.1f}]"
_cls = _nn10["classes"]
_below = {m: sorted(c for c, x in _nn10["by_class"][f"{m}|published"].items()
                    if c != "list of names" and x["excess"]["mean"] <= x["at_chance"]) for m in (_G4, _CL)}
_cfnd = [(x["no-data answer"]["neither, answered without the data"]["accepted"]["share"],
          x["no-data answer"]["neither, not"]["accepted"]["share"])
         for k, x in _nn10["nearest"].items() if "code-free" in k]
_n15 = _nn15["nearest"]["seven configurations|own model"]
check(all(abs(_nn10["all"][f"{m}|published"]["excess"]["mean"] - _ex(_pubi(m), "other")["mean"]) < 0.05
          and _nn10["all"][f"{m}|published"]["questions"] == _pubi(m)["other"]["n_items"] for m in (_G4, _CL))
      and f"By their options they are numbers written with words (${_cls['number with words']}$), directions or changes "
          f"(${_cls['direction or change']}$), names (${_cls['name']}$), intervals or bounds (${_cls['interval or bound']}$), "
          f"statements (${_cls['statement']}$) and lists of names (${_cls['list of names']}$)" in _flat
      and sum(_cls.values()) == _pubi(_CL)["other"]["n_items"] == 137
      and _below[_G4] == [] and len(_below[_CL]) == 2 and _cls["list of names"] == 2
      and "outside the two questions that list names, the published forced grades lie above what a random choice on "
          "every miss would add in every class but two of Claude 3.5 Sonnet's" in _flat
      and f"the published grades accept {_nnci(_nnacc(_G4, 'nearest', 'key'))}$ and "
          f"{_nnci(_nnacc(_CL, 'nearest', 'key'))}\\%$ of the misses whose nearest option is the key, "
          f"${_nnacc(_G4, 'nearest', 'other')['share']:.1f}$ and ${_nnacc(_CL, 'nearest', 'other')['share']:.1f}\\%$ of "
          f"those nearest another, and ${_nnacc(_G4, 'nearest', 'neither')['share']:.1f}$ and "
          f"${_nnacc(_CL, 'nearest', 'neither')['share']:.1f}\\%$ of those nearest neither" in _flat
      and f"it accepts {_nnci(_nnd(_G4, 'neither, answered without the data'))}$ and "
          f"${_nnd(_CL, 'neither, answered without the data')['share']:.1f}\\%$ "
          f"$[{_nnd(_CL, 'neither, answered without the data')['lo']:.1f},"
          f"{_nnd(_CL, 'neither, answered without the data')['hi']:.1f}]$ of such misses on the questions that the same "
          f"model answers correctly in BixBench's forced-choice runs without the data, and "
          f"${_nnd(_G4, 'neither, not')['share']:.1f}$ and ${_nnd(_CL, 'neither, not')['share']:.1f}\\%$ on the others"
          in _flat
      and len(_cfnd) == 8 and all(a > b for a, b in _cfnd) and "a grader shown no notebook does the same" in _flat
      and f"gpt-4o's grades select no option on only ${_nnp[_G4]['selects none']:.1f}\\%$ of these misses" in _flat
      and _nnp[_G4]["selects none"] < 100 - _fsg["misses_selected"]
      and f"On v1.5, where ${_nn15['classes']['interval or bound']}$ of the "
          f"${sum(_nn15['classes'].values())}$ other questions are numeric intervals, the configurations' own graders "
          f"accept ${_n15['all']['accepted']['share']:.1f}\\%$ $[{_n15['all']['accepted']['lo']:.1f},"
          f"{_n15['all']['accepted']['hi']:.1f}]$ of the misses: ${_n15['nearest']['key']['accepted']['share']:.1f}\\%$ "
          f"of those nearest the key and ${_n15['nearest']['other']['accepted']['share']:.1f}\\%$ of those nearest "
          f"another interval, half of them" in _flat
      and 45 <= _n15["nearest"]["other"]["share_of_misses"] <= 55,
      "the appendix's account of the questions whose options are not numbers no longer matches nonnumeric_excess.json")
_dec = [b for k, b in _inf10.items() if "code-free" in k and k.endswith(", decline")]
check(f"Nor does it depend on the notebook. Graded code-free, as the 2026 system grades, the published runs "
      f"score $+{_f1(_ex(_cfi(_G4, 'gpt-4o'), 'all')['mean'])}$ and "
      f"$+{_f1(_ex(_cfi(_CL, 'gpt-4o'), 'all')['mean'])}$ points above BixBench's open-ended grades over all "
      f"questions, where its forced grades with the notebook score $+{_f1(_ex(_pubi(_G4), 'all')['mean'])}$ "
      f"and $+{_f1(_ex(_pubi(_CL), 'all')['mean'])}$;" in _flat
      and _pubi(_CL)["all"]["n_items"] == 296 and len(_dec) >= 4
      and all(_pubi(m)["all"]["at_chance"] - 4 < _ex(_cfi(m, g), "all")["mean"] < _pubi(m)["all"]["at_chance"] + 6
              for m in (_G4, _CL) for g in ("gpt-4o", "gemma-3-27b", "Llama-3.3-70B")),
      "sec:withdata's account of the code-free graders over every question no longer matches grading_variants.json")
_g51 = {k.split("|")[1]: b for k, b in _gv["inflation"]["v1.5"].items() if k.startswith("gpt-5.1|")}
_g51nb, _g51cf = _g51["gpt-4o, with the notebook"], _g51["code-free gpt-4o, forced"]
check(f"gpt-5.1's runs over all $205$ v1.5 questions score ${_g51cf['all']['excess']['mean']:+.1f}$ code-free "
      f"against ${_g51nb['all']['excess']['mean']:+.1f}$ with the notebook. Forced-choice scores taken by "
      f"different graders are therefore not comparable." in _flat
      and _g51nb["all"]["n_items"] == _g51cf["all"]["n_items"] == 205
      and _g51nb["all"]["excess"]["lo"] > 0,
      "sec:withdata's account of gpt-5.1's runs over every v1.5 question no longer matches grading_variants.json")
# sec:withdata: the rank's cost under the code-free graders, BixBench's own grades' weight, the letter-only grader and
# the within-5% option
_pubw2 = next(c for c in _pwj["predicted_only"] if c["grader"] == "published grades")
_qw = next(c for c in _pwj["cases"] if c["runs"] == "v1.0, published" and c["grader"] == "Qwen2.5-72B")
_let = _gv["letter"]
_lrk = _rk["v1.0, published|gpt-4o, letter|original"]
_wt = {k.split("|")[1]: v["released"] for k, v in _gv["withintol"].items()
       if k.startswith("v1.0, published|") and v["released"]["n_miss"]}
_wtm = [c["moved to an edge"] for k, c in _rk.items() if k.startswith("v1.0, published|") and "withintol" in k
        and k.endswith("|original")]
_rng1 = lambda xs: f"${min(xs):.1f}$ to ${max(xs):.1f}"
check(min(_rkm("v1.0, published", g)["mean"] for g in ("gpt-4o, codefree", "gemma-3-27b, codefree")) > _rule_d1
      and "and more than the rule of the pre-specified test, which reads a number only when the whole answer is one"
          in _flat
      and f"a proximity weight of ${_pubw2['lambda']:.2f}$ (Table~\\ref{{tab:readers2}}); gpt-4o's grades alone have"
          in _flat
      and all(w["lo"] <= 0 <= w["hi"] for w in _wtm) and len(_wt) == 3,
      "sec:withdata's account of the other gradings no longer matches grading_variants.json")
# the recommendations: graded without the notebook the same runs score more, by sec:withdata's numbers
_nbd = [_ex(_cfi(m, "gpt-4o"), "all")["mean"] - _ex(_pubi(m), "all")["mean"] for m in (_G4, _CL)] \
       + [_g51cf["all"]["excess"]["mean"] - _g51nb["all"]["excess"]["mean"]]
check(f"graded without the notebook, the same runs score ${min(_nbd):.1f}$ to ${max(_nbd):.1f}$ points more over all "
      f"questions." in _flat and min(_nbd) > 0,
      "the recommendations misstate how much more the same runs score without the notebook")
# the introduction and the recommendations
_cf4 = [_rkm(f, g)["mean"] for f in ("v1.0, published", "v1.5, new seeds") for g in ("gpt-4o, codefree", "gemma-3-27b, codefree")]
_wtc = [100 - v["correct|accepted"] for v in _wt.values()]
# the within-5% option's rejection of correct answers on both releases (sec:design, the recommendations)
_wtc_all = [100 - v["released"]["correct|accepted"] for v in _gv["withintol"].values() if v["released"]["n_correct"]]
check(all(_rkm(f, g)["lo"] > 0 for f in ("v1.0, published", "v1.5, new seeds")
          for g in ("gpt-4o, codefree", "gemma-3-27b, codefree"))
      and f"and report the correct answers it rejects: code-free graders reject ${min(_wtc_all):.1f}$ to "
          f"${max(_wtc_all):.1f}\\%$ with it, and BixBench's refusal option ${100 - _cacc(_dg4):.1f}$ and "
          f"${100 - _cacc(_dcl):.1f}\\%$." in _flat
      and "An option stating that no value is within the tolerance removes the conflict, as the tolerance itself does."
          in _flat
      and all(w["lo"] <= 0 <= w["hi"] for w in _wtm),
      "the introduction's or the recommendations' account of the other gradings is stale")
# app:replication: the other gradings in full
_cfp = [c for k, c in _rk.items() if k.startswith("v1.0, published|") and ", codefree|original" in k]
_shr = [100 * c["proximity"]["moved"]["observed"] / c["proximity"]["moved"]["rule"] for c in _cfp]
_dmcf = [c["moved to an edge"]["mean"] for k, c in _rk.items()
         if k.startswith("v1.0, published|") and ", codefree|digit-matched" in k]
_dmnb = [c["moved to an edge"]["mean"] for k, c in _rk.items()
         if k.startswith("v1.0, published|") and ", notebook|digit-matched" in k]
check(f"Code-free graders follow the nearest option most closely ($\\lambda={min(c['proximity']['lambda'] for c in _cfp):.2f}$ "
      f"to ${max(c['proximity']['lambda'] for c in _cfp):.2f}$ on the published runs) and bear most of the cost: on the "
      f"moved keys they gain ${min(_shr):.0f}$ to ${max(_shr):.0f}\\%$ of what the nearest-option rule gains" in _flat
      and 50 < min(_shr) and max(_shr) < 100
      and f"Over the ${_gv['proximity_cost']['n']}$ gradings of Tables~\\ref{{tab:readers2}} and~\\ref{{tab:variants}} a "
          f"grading's share of the "
          f"rule's gain rises with its $\\lambda$ (Spearman ${_gv['proximity_cost']['spearman']:.2f}$)" in _flat
      and f"Through the digit-matched rewrites the gains on the published runs are much the same, "
          f"${min(_dmcf):+.1f}$ to ${max(_dmcf):+.1f}$ for the code-free graders and ${min(_dmnb):+.1f}$ to "
          f"${max(_dmnb):+.1f}$ for those with the notebook" in _flat
      and f"(${_let['accepted']['key']:.1f}$ and ${_let['accepted_published']['key']:.1f}\\%$ of those nearest the key), "
          f"and agrees with them at $\\kappa={_let['kappa_published']:.2f}$" in _flat
      and f"the code-free graders select that option on ${min(v['miss|none within'] for v in _wt.values()):.1f}$ to "
          f"${max(v['miss|none within'] for v in _wt.values()):.1f}\\%$ of the published runs' misses and on "
          f"${min(v['correct|none within'] for v in _wt.values()):.1f}$ to "
          f"${max(v['correct|none within'] for v in _wt.values()):.1f}\\%$ of their correct answers" in _flat,
      "app:replication's account of the other gradings no longer matches grading_variants.json")
# Figure 1c plots the same points
_pc = load("results/proximity_cost.json")
check(_pc["n"] == _gv["proximity_cost"]["n"] and abs(_pc["spearman"] - _gv["proximity_cost"]["spearman"]) < 1e-12
      and len(_pc["gradings"]) == _pc["n"],
      "results/proximity_cost.json and grading_variants.json disagree on Figure 1c's points")

# --- four more runs of Qwen3-235B-A22B, outside the pre-specified test (q235_extra.py) ----------------------------
_q3x = load("results/q235_extra.json")
_q3new, _q3all = _q3x["new"], _q3x["all six"]
_q3w = [v for k, v in _q3x["within_5pct"].items() if k in _q3new["n_trajectories"]]
_qci = lambda v: f"${v['mean']:+.1f}$ $[{v['lo']:+.1f},{v['hi']:+.1f}]$"
_q3cut = _q3x["cut_by_wallclock"]
_q3clock = ("A three-hour wall clock cut none of their $420$ episodes." if not _q3cut else
            "A three-hour wall clock cut one of their $420$ episodes, on a question whose R cells kept "
            "outrunning the ten-minute cell limit; it counts as unanswered." if len(_q3cut) == 1 else
            f"A three-hour wall clock cut ${len(_q3cut)}$ of their $420$ episodes, both on one question "
            f"whose R cells kept outrunning the ten-minute cell limit; they count as unanswered.")
check(sorted(_q3new["n_trajectories"]) == ["qwen3-235b|r2", "qwen3-235b|r3", "qwen3-235b|r4", "qwen3-235b|r5"]
      and all(n == 105 for n in _q3new["n_numeric_runs"].values())
      and f"of the key on ${min(_q3w):.1f}$ to ${max(_q3w):.1f}\\%$ of the numeric questions" in _flat
      and f"the moved keys gain {_qci(_q3new['repaired-placebo|moved to an edge'])} against $P$, on "
          f"${_q3new['newly_credited']['n_gained']}$ newly accepted runs, and the unchanged keys "
          f"{_qci(_q3new['gain|kept'])} against $R$; pooled with the test's two runs, "
          f"{_qci(_q3all['repaired-placebo|moved to an edge'])} and {_qci(_q3all['gain|kept'])}." in _flat
      and "gain against $P$ excludes zero, as a lack of power in the test would leave it, but the test's verdict "
          "stands" in _flat and "was one of power" not in _flat
      and _q3clock in _flat and len({c["question_id"] for c in _q3cut}) <= 1
      and sum(t.get("wallclock", 0) for t in _q3x["terminations"].values()) == len(_q3cut)
      and sum(sum(t.values()) for t in _q3x["terminations"].values()) == 420
      and all(c["answer"] is None and c["kernel_restarts"] >= 10 and c["seconds"] >= 10800 for c in _q3cut)
      and 'ap.add_argument("--cell-timeout", type=int, default=600)' in Path("bixbench_agent.py").read_text()
      and sorted(h for h, v in _rpl["D2"]["qwen3-235b|data"]["verdicts"].items() if not v) == ["H1", "H2", "H4"]
      and f"Its two runs in the pre-specified test fail H1, H2 and H4, and its moved keys' $U-P$ rests on "
          f"${_rpl['D2']['qwen3-235b|data']['newly_credited']['n_gained']}$ newly accepted runs." in _flat
      and f"(rollouts ${min(int(k[-1]) for k in _q3new['n_trajectories'])}$ to "
          f"${max(int(k[-1]) for k in _q3new['n_trajectories'])}$, with the data, on v1.5's numeric questions" in _flat
      and f"its unchanged keys still gain against $R$, but the "
          f"${_q3all['unchanged_split']['parts']['same rank']['n_items']}$ that keep their rank change by "
          f"${_q3all['unchanged_split']['parts']['same rank']['U-P']:+.1f}$ against $P$: the redrawing, not the rank, "
          f"moves them." in _flat
      and _q3all["repaired-placebo|moved to an edge"]["lo"] > 0 and _q3all["gain|kept"]["lo"] > 0
      and abs(_q3all["unchanged_split"]["parts"]["same rank"]["U-P"]) < 2,
      "the account of the four more Qwen3-235B-A22B runs no longer matches q235_extra.json")

# --- the digit-matched rewrites (digit_matched.py): sec:withdata's format cue and app:withdata's account ---------
_dmj = load("results/digit_matched.json")
_dmo, _dms = _dmj["v1.5"]["original"]["option_sets"], _dmj["v1.5"]["digit-matched"]["option_sets"]
_dmo10, _dms10 = _dmj["v1.0"]["original"]["option_sets"], _dmj["v1.0"]["digit-matched"]["option_sets"]
_dmg, _dmg10 = _dmj["v1.5"]["digit-matched"]["groups"], _dmj["v1.0"]["digit-matched"]["groups"]
_dmr = _dmj["runs"]
# how far the digit-matched rewrites move the rule's U - P, by key group: the runs of the paper's tables, and the two
# current agents apart
_dmcur = ("v1.5, gpt-6-luna", "v1.5, DeepSeek-V4-Pro")
_dmd = lambda r, g: r["digit-matched"][f"repaired-placebo|{g}"]["mean"] - r["original"][f"repaired-placebo|{g}"]["mean"]
_dmdiff = max(abs(_dmd(r, g)) for k, r in _dmr.items() if k not in _dmcur for g in ("moved to an edge", "kept", "all"))
_dmin = max(abs(_dmd(r, "moved inward")) for k, r in _dmr.items() if k not in _dmcur)
_dmlu = max(abs(_dmd(_dmr["v1.5, gpt-6-luna"], g)) for g in ("moved to an edge", "kept", "all"))
_dmds = _dmr["v1.5, DeepSeek-V4-Pro"]
_dmiv = lambda v: f"${v['mean']:+.1f}$ $[{v['lo']:+.1f},{v['hi']:+.1f}]$"
_fd = lambda e, a: f"{e[a]['fewest_digits']:.1f}"
check(f"The rule that selects the option with the fewest significant digits is correct on ${_fd(_dms, 'placebo')}$ and "
      f"${_fd(_dms, 'repaired')}\\%$ of v1.5's $P'$ and $U'$, against ${_fd(_dmo, 'released')}\\%$ on $R$ and "
      f"${_fd(_dmo, 'placebo')}$ and ${_fd(_dmo, 'repaired')}\\%$ on $P$ and $U$" in _flat
      and f"Rewrites that keep the released distractors' significant digits remove a roundness cue the "
          f"generator adds; they move the rule's contrasts on the moved and unchanged keys and over all items "
          f"by at most ${_dmdiff:.1f}$ points, except DeepSeek-V4-Pro's, which rest on few accepted answers "
          f"(Appendix~\\ref{{app:withdata}})." in _flat
      and f"Through $P'$ and $U'$ the rule's $U-P$ on the moved keys, the unchanged keys and all items moves by at most "
          f"${_dmdiff:.1f}$ points on the published runs, the seven configurations, the new seeds, Qwen3-235B-A22B and "
          f"gpt-5.1, and by up to ${_dmin:.1f}$ on their inward keys; gpt-6-luna's moves by at most ${_dmlu:.1f}$, and "
          f"DeepSeek-V4-Pro's, which rests on two to four answers, falls from "
          f"${_dmds['original']['repaired-placebo|moved to an edge']['mean']:+.1f}$, "
          f"${_dmds['original']['repaired-placebo|kept']['mean']:+.1f}$ and "
          f"${_dmds['original']['repaired-placebo|all']['mean']:+.1f}$ to zero." in _flat
      and all(abs(_dmds["digit-matched"][f"repaired-placebo|{g}"]["mean"]) < 1e-9 for g in ("moved to an edge", "kept", "all"))
      and 2 <= round(_dmds["original"]["repaired-placebo|moved to an edge"]["mean"] * 37 / 100)
      and round(_dmds["original"]["repaired-placebo|all"]["mean"] * 105 / 100) <= 4
      and sorted(_dmr) == sorted(["v1.0, published", "v1.5, seven configurations", "v1.5, new seeds",
                                  "v1.5, Qwen3-235B-A22B", "v1.5, gpt-5.1", *_dmcur]),
      "sec:withdata's format-cue paragraph no longer matches digit_matched.json")
check(f"On v1.5, $U'$ leaves out ${105 - _dms['repaired']['n_items']}$ of the $105$ numeric items and $P'$ "
      f"${105 - _dms['placebo']['n_items']}$, where no draw succeeds; on v1.0, $P'$ leaves out "
      f"${159 - _dms10['placebo']['n_items']}$ of $159$ and $U'$ none" in _flat
      and _dms10["repaired"]["n_items"] == 159
      and f"is correct on ${_fd(_dms, 'placebo')}$ and ${_fd(_dms, 'repaired')}\\%$ of v1.5's $P'$ and $U'$, against "
          f"${_fd(_dms, 'released')}\\%$ on $R$ and ${_fd(_dmo, 'placebo')}$ and ${_fd(_dmo, 'repaired')}\\%$ on $P$ and "
          f"$U$ (v1.0: ${_fd(_dms10, 'placebo')}$ and ${_fd(_dms10, 'repaired')}\\%$, against ${_fd(_dms10, 'released')}\\%$ "
          f"on $R$ and ${_fd(_dmo10, 'placebo')}$ and ${_fd(_dmo10, 'repaired')}\\%$ on $P$ and $U$), and the nearest "
          f"distractor lies at a median $d$ of ${_dms['placebo']['nearest_d_median']:.3f}$ and "
          f"${_dms['repaired']['nearest_d_median']:.3f}$" in _flat
      and f"$U'$ moves ${_dmg['moved to an edge']}$ of v1.5's keys from bracketed to extreme and ${_dmg['moved inward']}$ "
          f"inward, and leaves ${_dmg['kept']}$; on v1.0, ${_dmg10['moved to an edge']}$, ${_dmg10['moved inward']}$ and "
          f"${_dmg10['kept']}$" in _flat,
      "app:withdata's account of the digit-matched rewrites no longer matches digit_matched.json")


# --- gpt-4o asked for a number without the data or the options (forced_guess.py): sec:open and app:arms ------------
_fg = load("results/forced_guess.json")
_fgc = lambda rel, p: _fg[f"{rel}|{p}"]
_fgv = lambda v: f"${v['mean']:.1f}$ $[{v['lo']:.1f},{v['hi']:.1f}]$"
_fgd = lambda v: f"${v['mean']:.1f}$ $[{v['lo']:+.1f},{v['hi']:+.1f}]$"
_n15 = round(_fgc("v15", "stated")["stated_share"] * _fgc("v15", "stated")["n_items"] / 100)
check(_fg["served"] == ["gpt-4o-2024-11-20"] and _fg["content_filtered"] == 0
      and _fgc("v15", "stated")["n_items"] == 105 and _fgc("v10", "stated")["n_items"] == 158
      and _fgc("v10", "stated")["stated_share"] == 100.0
      and _fgc("v15", "forced")["stated_share"] == _fgc("v10", "forced")["stated_share"] == 100.0
      and f"asked for its answer as a number, gpt-4o {_fg['served'][0][len('gpt-4o-'):]} gives one on ${_n15}$ of v1.5's "
          f"$105$ and on all of v1.0's $158$; told as well that the data are "
          f"unavailable and that a best estimate is required, on all of both." in _flat
      and _fgc("v15", "stated")["within_5pct"]["mean"] == _fgc("v15", "forced")["within_5pct"]["mean"]
      and f"Its number is within $5\\%$ of the key on ${_fgc('v15', 'forced')['within_5pct']['mean']:.1f}\\%$ of v1.5's "
          f"questions under either prompt (${_fgc('v10', 'stated')['within_5pct']['mean']:.1f}$ and "
          f"${_fgc('v10', 'forced')['within_5pct']['mean']:.1f}\\%$ on v1.0), and the released option nearest it is the key "
          f"on {_fgv(_fgc('v15', 'stated')['nearest_is_key_when_stated'])} and "
          f"${_fgc('v15', 'forced')['nearest_is_key_when_stated']['mean']:.1f}\\%$ "
          f"$[{_fgc('v15', 'forced')['nearest_is_key_when_stated']['lo']:.1f},"
          f"{_fgc('v15', 'forced')['nearest_is_key_when_stated']['hi']:.1f}]$ "
          f"(${_fgc('v10', 'stated')['nearest_is_key_when_stated']['mean']:.1f}$ and "
          f"${_fgc('v10', 'forced')['nearest_is_key_when_stated']['mean']:.1f}\\%$ on v1.0), where a random option is "
          f"correct on $25\\%$." in _flat
      and f"it selects the key on ${_fgc('v15', 'stated')['forced_choice']['mean']:.1f}\\%$ "
          f"$[{_fgc('v15', 'stated')['forced_choice']['lo']:.1f},{_fgc('v15', 'stated')['forced_choice']['hi']:.1f}]$ of "
          f"v1.5's questions and ${_fgc('v10', 'stated')['forced_choice']['mean']:.1f}\\%$ of v1.0's "
          f"(Table~\\ref{{tab:gptnodata}}): {_fgd(_fgc('v15', 'stated')['forced_minus_predicted'])} and "
          f"{_fgd(_fgc('v15', 'forced')['forced_minus_predicted'])} points more on v1.5 than choosing the option nearest "
          f"its own number would score, and ${_fgc('v10', 'stated')['forced_minus_predicted']['mean']:.1f}$ and "
          f"${_fgc('v10', 'forced')['forced_minus_predicted']['mean']:.1f}$ more on v1.0." in _flat
      and abs(_fgc("v15", "stated")["forced_choice"]["mean"] - 25 - 5.8) < 0.05
      and abs(_fgc("v10", "stated")["forced_choice"]["mean"] - 25 - 10.2) < 0.05
      and all(_fgc(r, p)["forced_minus_predicted"]["lo"] > 0 for r in ("v15", "v10") for p in ("stated", "forced"))
      and "What the models can state does not account for this margin." in _flat,
      "the account of gpt-4o asked for a number without the data or the options no longer matches forced_guess.json")


# --- numbers a one-digit mutation of the text left standing: each now pinned to its source ---------------------------
_man = load("results/miss_anatomy.json")
_am, _nn = _man["accepted_misses"], _man["no_number"]
_amr = _am["runs"]
# sec:withdata and app:replication: how far the published grades' accepted misses are
check(f"Of the ${_am['n_runs']:,}$ published runs whose answer misses the key and whose forced grade accepts it, on "
          f"${_am['n_questions']}$ questions, ${_amr['5 to 10%']:.1f}\\%$ are within $10\\%$ of the key, "
          f"${_amr['10 to 25%']:.1f}\\%$ are $10$ to $25\\%$ off, ${_amr['25 to 100%']:.1f}\\%$ are $25$ to $100\\%$ off and "
          f"${_amr['over 100%']:.1f}\\%$ are further, and ${_amr['no number']:.1f}\\%$ give no number. "
          f"${_am['within_range_width']['median']:.1f}\\%$ lie within the median half-width of BixBench's own range keys "
          f"(${100 * _am['range_widths']['median']:.1f}\\%$), and ${_am['within_range_width']['upper_quartile']:.1f}\\%$ "
          f"within ${100 * _am['range_widths']['upper_quartile']:.1f}\\%$, a half-width that only a quarter of those keys "
          f"exceed".replace(",", "{,}", 1) in _flat
      and abs(sum(_amr.values()) - 100) < 1e-6,
      "the account of how far the accepted misses are no longer matches miss_anatomy.json")
# app:replication: what a grader selects on an answer with no number
_pubnn = _nn["v1.0, published runs"]["published"]["released"]
_own15, _gem15 = (_nn["v1.5, with the data"][g] for g in ("own model", "gemma-3-27b"))
_sr = _pubnn["select_rank"]
check(f"the published forced grades select the smallest of the four values on ${_sr[0]:.0f}\\%$ of runs, the "
      f"second-smallest on ${_sr[1]:.0f}\\%$, the third on ${_sr[2]:.0f}\\%$ and the largest on ${_sr[3]:.0f}\\%$, and no "
      f"option on ${_pubnn['select_none']:.0f}\\%$." in _flat
      and f"the configurations' own graders select the second-smallest on ${_own15['released']['select_rank'][1]:.0f}\\%$ "
          f"of their gradings and gemma-3-27b on ${_gem15['released']['select_rank'][1]:.0f}\\%$, and both accept more of "
          f"these answers through $U$ (${_own15['repaired']['accepted']:.1f}$ and ${_gem15['repaired']['accepted']:.1f}\\%$) "
          f"than through $R$ (${_own15['released']['accepted']:.1f}$ and ${_gem15['released']['accepted']:.1f}\\%$)" in _flat
      and min(_own15["released"]["key_rank"][1], _gem15["released"]["key_rank"][1]) > 50
      and _sr[0] == max(_sr) and _own15["repaired"]["key_rank"][0] > _own15["released"]["key_rank"][0],
      "app:replication's account of the selections on answers with no number no longer matches miss_anatomy.json")
# app:replication: what moves the unchanged keys
_u10 = _man["unchanged_keys"]["v1.0, published"]
_u10p = _u10["parts"]
_uq3 = _man["unchanged_keys"]["v1.5, Qwen3-235B-A22B"]["parts"]["same rank"]
check(f"On the published runs the ${_u10p['the other extreme']['n_items']}$ keys that $U$ moves from one extreme to the "
      f"other gain ${_u10p['the other extreme']['U-R']:+.1f}$ against $R$ and ${_u10p['the other extreme']['U-P']:+.1f}$ "
      f"against $P$, ${_u10p['the other extreme']['contribution']:.1f}$ of the unchanged keys' ${_u10['mean']:+.1f}$; the "
      f"${_u10p['same rank']['n_items']}$ that keep their rank change by ${_u10p['same rank']['U-R']:+.1f}$ against $R$ "
      f"and ${_u10p['same rank']['U-P']:+.1f}$ against $P$, and the ${_u10p['another middle rank']['n_items']}$ moved to "
      f"another middle rank by ${_u10p['another middle rank']['U-R']:+.1f}$ and "
      f"${_u10p['another middle rank']['U-P']:+.1f}$." in _flat
      and f"For Qwen3-235B-A22B the keys that keep their rank gain ${_uq3['U-R']:+.1f}$ against $R$ and "
          f"${_uq3['U-P']:+.1f}$ against $P$" in _flat,
      "app:replication's account of what moves the unchanged keys no longer matches miss_anatomy.json")
# numbers app:arms states, each once (its key-rank cells also in a caption)
_mg, _mc = _cbc["published, gpt-4o"]["parts"][0], _cbc["published, claude"]["parts"][0]
_m95 = lambda part: f"$[{part['ci_covering_95'][0]:+.1f},{part['ci_covering_95'][1]:+.1f}]$"
for _s, _n in ((f"${_mg['estimate']:+.1f}$ {_m95(_mg)} and ${_mc['estimate']:+.1f}$ {_m95(_mc)} points above chance; "
                f"corrected across the paper's ten headline claims, gpt-4o's margin survives and Claude's interval "
                f"reaches ${_mc['ci_family'][0]:.1f}$", 1),
               (f"${-_d(_pg)['mean']:.1f}$ and ${-_d(_pc15)['mean']:.1f}$ points \\emph{{less}} accurate where the key is "
                f"second-smallest", 1),
               (f"non-numeric questions (${_pg['margin_other']['mean']:+.1f}$ and ${_pc15['margin_other']['mean']:+.1f}$) as "
                f"on the numeric ones (${_pg['margin_numeric']['mean']:+.1f}$ and ${_pc15['margin_numeric']['mean']:+.1f}$)", 1),
               (f"${_kr[0]}$, ${_kr[1]}$, ${_kr[2]}$ and ${_kr[3]}$ items", 2)):
    check(_flat.count(_s) == _n, f"app:arms should state {_s!r} ({_flat.count(_s)} of {_n})")
# app:replication's gradings paragraph restates the letter-only grader and the factor-of-ten excess
check(f"Constrained to a letter, gpt-4o accepts ${_let['accepted']['all misses']:.1f}\\%$ of the misses of the published "
      f"runs' random quarter, where the published grades of the same runs accept "
      f"${_let['accepted_published']['all misses']:.1f}\\%$ (" in _flat,
      "app:replication's account of the letter-only grader no longer matches grading_variants.json")
_plog = load("results/tolerance_check.json")["T3_pvalue_log"]
_x10, _x2 = _plog["x10"], _plog["x2"]
_ivs = lambda v: f"${v['mean']:+.1f}$ $[{v['lo']:+.1f},{v['hi']:+.1f}]$"
check(f"the forced-choice excess on the published runs is still "
      f"${_rg['gpt-4o']['p-values within x10']['excess']['mean']:+.1f}$ and "
      f"${_rg['Claude 3.5 Sonnet']['p-values within x10']['excess']['mean']:+.1f}$ at a factor of ten "
      f"(Table~\\ref{{tab:reference}}) and the data are still worth "
      f"${_x10['qwen72b data - nodata']['mean']:+.1f}$ to Qwen2.5-72B. Its leads over Llama-3.3-70B and "
      f"gemma-3-27b are as at $5\\%$ at a factor of two and fall to {_ivs(_x10['qwen72b - llama70b'])} and "
      f"{_ivs(_x10['qwen72b - gemma27b'])} at a factor of ten, where the first is no longer significant." in _flat
      and all(abs(_x2[k][f] - _t3["withdata"]["5%"][k][f]) < 1e-9 for k in ("qwen72b - llama70b", "qwen72b - gemma27b")
              for f in ("mean", "lo", "hi"))
      and _x10["qwen72b - llama70b"]["lo"] < 0 < _x10["qwen72b - gemma27b"]["lo"]
      and _x10["qwen72b data - nodata"]["lo"] > 0 and _plog["n_p_value_items"] == _rfc["p_value_items"]["v1.5"],
      "app:replication's log-scale tolerance for p-value keys no longer matches tolerance_check.json")
# app:theory: the distinctness sweep and Table tab:distinctness's two skewed marginals
_swp = {round(e["s"], 2): e for e in load("results/writer_theory.json")["distinctness_sweep"]}
_sw3 = [_swp[s] for s in (0.25, 0.5, 0.75)]
check(f"it is ${100 * _sw3[0]['gamma_reject_against_key']:+.1f}$, ${100 * _sw3[1]['gamma_reject_against_key']:+.1f}$ and "
      f"${100 * _sw3[2]['gamma_reject_against_key']:+.1f}$ points when the largest share is "
      f"${_sw3[0]['largest_share']:.2f}$, ${_sw3[1]['largest_share']:.2f}$ and ${_sw3[2]['largest_share']:.2f}$, while "
      f"the jointly conditioned draw stays at zero" in _flat
      and all(abs(e["gamma_symmetric"]) < 1e-9 for e in _swp.values())
      and "$\\kappa=(.25,.22,.20,.15,.10,.08)$ (mild skew) and $\\kappa=(.45,.25,.15,.08,.05,.02)$ (heavy skew)" in _flat
      and '("mild skew", np.array([.25, .22, .20, .15, .10, .08]))' in Path("writer_theory.py").read_text()
      and '("heavy skew", np.array([.45, .25, .15, .08, .05, .02]))' in Path("writer_theory.py").read_text(),
      "app:theory's distinctness sweep or tab:distinctness's marginals no longer match writer_theory")
check(_flat.count(f"the ${_pg['n_other']}$ non-numeric questions (${_pg['margin_other']['mean']:+.1f}$") == 1,
      "app:arms should count the non-numeric questions")
# app:replication: H2 read as an equivalence claim, which no data set satisfies
_eqsets = [_rpl["D1"], _rpl["D2"]["reruns|data"], _rpl["D2"]["qwen3-235b|data"]]
check("tests the same change as an equivalence claim, with the whole interval within $\\pm5$ points: no data set "
      "satisfies it." in _flat
      and not any(-5 < r["gain|kept"]["lo"] and r["gain|kept"]["hi"] < 5 for r in _eqsets),
      "app:replication's equivalence reading of H2 no longer matches replication.json")
# app:withdata: the gradings' decoding, as bixbench_withdata.py sends it
check("Every grading is at temperature $0$, with two shuffles of each option set per run." in _flat
      and bw.SHUFFLES == 2 and '"temperature": 0.0' in Path("bixbench_withdata.py").read_text(),
      "app:withdata misstates the gradings' temperature or shuffles")
_ftol = "the last number within a factor of $2$ or of $10$ of the key, every other key within $5\\%$"
check(_ftol in _flat and {"x2", "x10"} <= set(_plog),
      "app:replication's log-scale tolerance names factors the result file does not have")
# tab:variants: the keys the digit-matched rewrites move
check(f"(which move ${_dmg10['moved to an edge']}$ keys on v1.0 and ${_dmg['moved to an edge']}$ on v1.5)" in _flat,
      "tab:variants's caption miscounts the keys the digit-matched rewrites move")

# --- the correction for guessing (formula_scoring.py, tab:formula) ----------------------------------------------
import formula_scoring as _fsm  # noqa: E402
_fsj = load("results/formula_scoring.json")
_fsb = " ".join(_table_body("tab:formula").split())
_fsrs = _fsm.table_rows(_fsj)
check(all(" ".join(r.split()) in _fsb for r in _fsrs) and _fsb.count("\\\\") == len(_fsrs) + 2,
      "tab:formula's rows no longer match formula_scoring.table_rows")
_fx = lambda b, g, r: _fsrows[(b, g, r)]
_fg4, _fcl = _fx("v1.0", "gpt-4o", "published forced"), _fx("v1.0", "Claude 3.5 Sonnet", "published forced")
_f15 = [_fx("v1.5", "pooled", "family forced"), _fx("v1.5 new", "new seeds", "gemma forced"),
        _fx("v1.5 new", "Qwen3-235B-A22B", "gemma forced"), _fx("v1.5 closed", "gpt-5.1", "gpt-4o forced")]
_fcm = lambda x: x["corrected_minus_tolerance"]
_fpt = lambda x, k: x["parts"][k]["points"]
_fbelow = sorted(-_fcm(x)["mean"] for x in _f15)
_fcur = [_f15[3], _fx("current", "gpt-6-luna", "gpt-4o forced"), _fx("current", "DeepSeek-V4-Pro", "gpt-4o forced")]
_fv15 = _f15 + [_fx("v1.5", "pooled", "gemma forced")] + _fcur[1:]
_fvbelow = sorted(-_fcm(x)["mean"] for x in _fv15)
_foa = lambda x: x["omission_aware_minus_tolerance"]
_fsel = lambda x: 100 - x["misses_selected"]
_foacur = [_foa(x) for x in _fcur]
_frng = lambda k, xs: sorted(_fpt(x, k) for x in xs)
_fci = lambda x: f"${_fcm(x)['mean']:+.1f}$ $[{_fcm(x)['lo']:+.1f},{_fcm(x)['hi']:+.1f}]$"
_empty = lambda x: x["parts"]["empty"]["share_of_runs"]
check(f"It brings the published runs to {_fci(_fg4)} and {_fci(_fcl)} points from the tolerance, but as a sum "
      f"of opposite parts (Table~\\ref{{tab:formula}}): beyond a random choice, the misses nearest the key add "
      f"${_fpt(_fg4, 'miss, key nearest'):.1f}$ and ${_fpt(_fcl, 'miss, key nearest'):.1f}$ points and those "
      f"nearest another option subtract ${-_fpt(_fg4, 'miss, other nearest'):.1f}$ and "
      f"${-_fpt(_fcl, 'miss, other nearest'):.1f}$. On v1.5, whose options bracket ${_beta15:.0f}\\%$ of the "
      f"keys and whose graders reject more correct answers, the same correction lands ${_fvbelow[0]:.1f}$ to "
      f"${_fvbelow[-1]:.1f}$ points below the tolerance, the current agents included. Much of that gap comes "
      f"from how it counts the parser's failures. Scoring a grade that selects no option as an omission, as "
      f"formula scoring treats an examinee who omits, moves gpt-4o's published residual from "
      f"${_fcm(_fg4)['mean']:+.1f}$ to ${_foa(_fg4):+.1f}$, the seven configurations' from "
      f"${_fcm(_f15[0])['mean']:+.1f}$ to ${_foa(_f15[0]):+.1f}$ and the current agents' to "
      f"${min(_foacur):+.1f}$ to ${max(_foacur):+.1f}$ (Appendix~\\ref{{app:replication}})." in _flat
      and all(_fpt(x, "miss, key nearest") > 0 > _fpt(x, "miss, other nearest") for x in (_fg4, _fcl))
      and all(_fcm(x)["mean"] < 0 for x in _fv15) and all(_fcm(x)["hi"] < 0 for x in _f15)
      and _foa(_f15[0]) < 0 < min(_foacur)
      and max(x["correct_accepted"] for x in _f15) < min(_fg4["correct_accepted"], _fcl["correct_accepted"])
      and _beta15 > 100 * (1 - 2 / 4) and _beta15 > _beta10,
      "sec:withdata's account of the correction for guessing no longer matches formula_scoring.json")
# "largely": scoring non-selections as omissions closes most of the v1.5 shortfall, over the gradings of tab:formula
_fclosed = sum(_foa(x) - _fcm(x)["mean"] for x in _fv15) / sum(-_fcm(x)["mean"] for x in _fv15)
check(_fcm(_fg4)["mean"] < 0 and _fcm(_fcl)["mean"] < 0 and max(abs(_fcm(_fg4)["mean"]), abs(_fcm(_fcl)["mean"])) < 4
      and f"It lands ${-_fcm(_fg4)['mean']:.1f}$ and ${-_fcm(_fcl)['mean']:.1f}$ points below the tolerance on "
          f"the published runs as a sum of opposite parts, and ${_fvbelow[0]:.1f}$ to ${_fvbelow[-1]:.1f}$ "
          f"points below it on v1.5, largely because it scores a grader's non-selections as wrong guesses. "
          f"Scored as omissions, the seven configurations' own gradings sit ${-_foa(_f15[0]):.1f}$ points "
          f"below the tolerance and the current agents' ${min(_foacur):.1f}$ to ${max(_foacur):.1f}$ above." in _flat
      and _fclosed > 0.5
      and "on the current release it falls below the tolerance, largely because it scores a grader's non-selections as "
          "wrong guesses" in _flat
      and f"it recovers the tolerance on BixBench's published runs only because two parts that depend on the options "
          f"cancel, lands ${_fvbelow[0]:.0f}$ to ${_fvbelow[-1]:.0f}$ points below it on v1.5, and cannot change a ranking"
          in _flat,
      "the abstract's, the introduction's or the recommendations' account of the correction for guessing is stale")
_fcf = [_fx("code-free", g, "code-free") for g in ("gpt-4o", "Claude 3.5 Sonnet", "seven configurations")]
# tab:formula's code-free rows (pinned with its other rows): near the tolerance on the published runs, below it on v1.5
check(max(abs(_fcm(_fcf[0])["mean"]), abs(_fcm(_fcf[1])["mean"])) < 4 and _fcm(_fcf[2])["hi"] < 0,
      "sec:withdata's code-free grades corrected for guessing are stale")
_fce = [_fpt(x, "correct") + _fpt(x, "empty") for x in _fv15]
check(f"On the published runs the misses nearest the key and those nearest another option contribute "
      f"${_fpt(_fg4, 'miss, key nearest'):+.1f}$ and ${_fpt(_fg4, 'miss, other nearest'):+.1f}$ points to gpt-4o's "
      f"residual and ${_fpt(_fcl, 'miss, key nearest'):+.1f}$ and ${_fpt(_fcl, 'miss, other nearest'):+.1f}$ to Claude 3.5 "
      f"Sonnet's" in _flat
      and f"on v1.5 the misses nearest another option (${_frng('miss, other nearest', _fv15)[-1]:+.1f}$ to "
          f"${_frng('miss, other nearest', _fv15)[0]:+.1f}$) and the correct answers rejected with the empty ones "
          f"(${max(_fce):+.1f}$ to ${min(_fce):+.1f}$) outweigh those nearest the key "
          f"(${_frng('miss, key nearest', _fv15)[0]:+.1f}$ to ${_frng('miss, key nearest', _fv15)[-1]:+.1f}$)" in _flat
      and all(_fpt(x, "miss, other nearest") + _fpt(x, "correct") + _fpt(x, "empty") < -_fpt(x, "miss, key nearest")
              for x in _fv15)
      and f"moves the published gpt-4o residual from ${_fcm(_fg4)['mean']:+.1f}$ to ${_foa(_fg4):+.1f}$, "
          f"gpt-5.1's from ${_fcm(_fcur[0])['mean']:+.1f}$ to ${_foa(_fcur[0]):+.1f}$ and the two current "
          f"agents' from ${_fcm(_fcur[1])['mean']:+.1f}$ and ${_fcm(_fcur[2])['mean']:+.1f}$ to "
          f"${_foa(_fcur[1]):+.1f}$ and ${_foa(_fcur[2]):+.1f}$. gpt-4o selects no option on ${_fsel(_fg4):.1f}"
          f"$, ${_fsel(_fcur[0]):.1f}$, ${_fsel(_fcur[1]):.1f}$ and ${_fsel(_fcur[2]):.1f}\\%$ of their "
          f"misses, the seven configurations' graders on ${_fsel(_f15[0]):.1f}\\%$ (their residual moves to "
          f"${_foa(_f15[0]):+.1f}$), and the others on fewer than $3\\%$."
          in _flat
      and max(_fsel(x) for x in (_fcl, _f15[1], _f15[2], _fx("v1.5", "pooled", "gemma forced"))) < 3,
      "app:replication's account of what the correction for guessing leaves is stale")

# --- other designs (option_design.py, tab:design, tab:designpairs) ------------------------------------------------
import option_design as _odm  # noqa: E402
_odj = load("results/option_design.json")
_odb, _odpb = " ".join(_table_body("tab:design").split()), " ".join(_table_body("tab:designpairs").split())
_odr, _odp = _odm.design_rows(_odj), _odm.pair_rows(_odj)
check(all(" ".join(r.split()) in _odb for r in _odr) and all(" ".join(r.split()) in _odpb for r in _odp),
      "tab:design's or tab:designpairs's rows no longer match option_design.py")
_R10, _R15 = _odj["releases"]["v1.0"], _odj["releases"]["v1.5"]
_pr = lambda R, pair, part="moved": R["pairs"][pair][part]
_ds = lambda R, name: R["designs"][name]
_rmis = lambda R, name: _ds(R, name)["rule"]["all"]["misses"]["accepted"]
_odg = _odj["graders"]
_gv_ = lambda rel, kind, name, key: [_odg[r][rel][kind][name][key] if kind == "designs"
                                     else _odg[r][rel][kind][name]["moved"]["mean"] for r in _odg]
_gsp = lambda name, key: _gv_("v1.0", "designs", name, key) + _gv_("v1.5", "designs", name, key)
_gerr = _gv_("v1.0", "pairs", "errors-uniform - errors-released", None) + \
        _gv_("v1.5", "pairs", "errors-uniform - errors-released", None)
_ks = (4, 6, 8, 10)
_kmv = [_pr(R, f"k{k}-uniform - k{k}-middle")["mean"] for R in (_R10, _R15) for k in _ks]
_kall = lambda R: [_pr(R, f"k{k}-uniform - k{k}-middle", "all")["mean"] for k in _ks]
_lth = [_ds(_R10, f"k{k}-middle")["leak_theory"] for k in _ks]
_lho = [_ds(_R10, f"k{k}-middle")["leak"]["held_out"] for k in _ks]
_f4 = lambda xs, fmt: ", ".join(fmt.format(x) for x in xs[:-1]) + " and " + fmt.format(xs[-1])
check(sorted(_odg) == ["gemma27b", "gpt-4o", "llama70b", "qwen72b"]
      and all(abs(t - 100 * 2 / (k * (k - 2))) < 1e-9 for t, k in zip(_lth, _ks))
      and f"Distractors taken from agents' errors leave the cost in place: drawing their key's rank uniformly "
          f"rather than at its released rank raises the rule's acceptance on the keys it moves to an extreme "
          f"by {_sci(_pr(_R10, 'errors-uniform - errors-released'))} on the published runs and "
          f"{_sci(_pr(_R15, 'errors-uniform - errors-released'))} on v1.5, as the generator's distractors do "
          f"(${_pr(_R10, 'k4-uniform - k4-released')['mean']:+.1f}$ and "
          f"${_pr(_R15, 'k4-uniform - k4-released')['mean']:+.1f}$)."
          in _flat
      and all(_pr(R, "errors-uniform - errors-released")["lo"] > 0 for R in (_R10, _R15))
      and f"More options shrink both sides of the conflict. Each moved key gains ${min(_kmv):.0f}$ to "
          f"${max(_kmv):.0f}$ points at every $k$, but only $2/k$ of the keys move, so a uniform rank costs "
          f"the rule ${_kall(_R10)[0]:.1f}$ and ${_kall(_R15)[0]:.1f}$ points over all items against one kept "
          f"off the extremes at $k=4$ on the published runs and v1.5, and ${_kall(_R10)[2]:.1f}$ and "
          f"${_kall(_R15)[2]:.1f}$ at $k=8$ (Table~\\ref{{tab:designpairs}}). The rank kept off the extremes "
          f"leaks at least ${_lth[0]:.0f}$ and ${_lth[2]:.1f}$ points in theory, and ${_lho[0]:+.1f}$ and "
          f"${_lho[2]:+.1f}$ on held-out capsules of v1.0." in _flat
      and _kall(_R10)[2] < _kall(_R10)[0] and _kall(_R15)[2] < _kall(_R15)[0]
      and all(_pr(R, f"k{k}-uniform - k{k}-middle", "all")["lo"] > 0 for R in (_R10, _R15) for k in (4, 8))
      and f"Closer spacing narrows what a bracketed key accepts: with eight options a fifth of the key apart "
          f"and the key never extreme, code-free graders accept "
          f"${min(_gsp('k8-middle-s0.2', 'misses_accepted')):.1f}$ to "
          f"${max(_gsp('k8-middle-s0.2', 'misses_accepted')):.1f}\\%$ of misses (against "
          f"${min(_gsp('k4-released', 'misses_accepted')):.1f}$ to "
          f"${max(_gsp('k4-released', 'misses_accepted')):.1f}\\%$ through four options redrawn at the "
          f"released ranks) and ${min(_gsp('k8-middle-s0.2', 'correct_accepted')):.1f}$ to "
          f"${max(_gsp('k8-middle-s0.2', 'correct_accepted')):.1f}\\%$ of correct answers." in _flat
      and f"The within-$5\\%$ option does, by grading the tolerance itself, at the price of rejecting "
          f"${min(_wtc_all):.1f}$ to ${max(_wtc_all):.1f}\\%$ of correct answers." in _flat,
      "sec:design's account of the other designs no longer matches option_design.json")
check(f"A key kept off the extremes then leaks its rank "
      f"(${_ds(_R10, 'k8-middle-s0.2')['leak']['held_out']:+.1f}$ and "
      f"${_ds(_R15, 'k8-middle-s0.2')['leak']['held_out']:+.1f}$ points on held-out capsules), so such options "
      f"suit grading, not a no-data baseline." in _flat
      and all(_ds(R, "k8-middle-s0.2")["leak"]["held_out"] > 5 for R in (_R10, _R15))
      and f"eight a fifth of the key apart leave code-free graders accepting "
          f"${min(_gsp('k8-middle-s0.2', 'misses_accepted')):.1f}$ to ${max(_gsp('k8-middle-s0.2', 'misses_accepted')):.1f}\\%$ "
          f"of misses" in _flat
      and "rank kept off the extremes leaks at least $25$ points at $k=4$ and $4.2$ at $k=8$, a uniform rank opens one "
          "side of $2/k$ of the keys" in _flat and abs(_lth[0] - 25) < 1e-9 and f"{_lth[2]:.1f}" == "4.2",
      "the introduction's, section 2's or the recommendations' account of the other designs is stale")
# app:design: how far the held-out leak lies from Proposition 1(iii)'s, and why
_lk = [(_ds(R, f"k{k}-middle")["leak"], _ds(R, f"k{k}-middle")["leak_theory"], k) for R in (_R10, _R15) for k in _ks]
_ldev = [e["held_out"] - t for e, t, _ in _lk]
_lk6 = _ds(_R10, "k6-middle")["leak"]
check(f"on held-out capsules, from ${-min(_ldev):.1f}$ points less to ${max(_ldev):.1f}$ more." in _flat
      and min(_ldev) < 0 < max(_ldev)
      and all(max(e["rank_shares"]) > 1 / (k - 2) for e, _, k in _lk)
      and f"the most frequent middle rank holds more than its uniform share (${100 * max(_lk6['rank_shares']):.1f}$ "
          f"against $25\\%$ of v1.0's keys at $k=6$)" in _flat
      and all(e["rank_shares"][0] == e["rank_shares"][-1] == 0 for e, _, _ in _lk),
      "app:design's account of the held-out leak against Proposition 1(iii) is stale")
# app:design: how the designs are built and compared
import gzip as _gzip  # noqa: E402
with _gzip.open(_odm.SHIPPED / "designs.json.gz", "rt") as _fh:
    _odd = json.load(_fh)
_pool = {rel: [e["all"]["from_pool"] for e in _odd["designs"][rel]["errors-released"].values() if e.get("all")]
         for rel in ("v1.0", "v1.5")}
_nitems = lambda R, pred: [R["designs"][n]["leak"]["n_items"] for n in R["designs"] if pred(n)]
_left = lambda R, n_all, pred: n_all - min(_nitems(R, pred))
_gen = lambda n: n.startswith("k")
_err = lambda n: n.startswith("errors-")
_mid = [_rmis(R, f"k{k}-middle") for R in (_R10, _R15) for k in _ks]
_th = lambda n: f"{n:,}".replace(",", "{,}")
check(f"At the released rank ${100 * sum(_pool['v1.0']) / (3 * len(_pool['v1.0'])):.0f}$ and "
      f"${100 * sum(_pool['v1.5']) / (3 * len(_pool['v1.5'])):.0f}\\%$ of the distractors come from the pool "
      f"on v1.0 and v1.5, and at most ${max(_left(_R10, 159, _err), _left(_R15, 105, _err))}$ items of a "
      f"release are left out." in _flat
      and f"one that no rank can be reached for is left out (at most "
          f"${max(_left(_R10, 159, _gen), _left(_R15, 105, _gen))}$ items of a release)" in _flat
      and f"(${_th(_R10['n_common_runs'])}$ of the ${_th(_R10['n_runs'])}$ published runs, ${_th(_R15['n_common_runs'])}$ "
          f"of the ${_th(_R15['n_runs'])}$ v1.5 runs with the data)" in _flat
      and _odm.SAMPLE_OF == 4
      and f"Where the key never sits at an extreme, the rule accepts ${min(_mid):.0f}$ to ${max(_mid):.0f}\\%$ of the "
          f"misses at every $k$" in _flat,
      "app:design's account of how the designs are built and compared no longer matches option_design.py")

# --- the proximity weight out of sample (lambda_holdout.py) -------------------------------------------------------
_lhj = load("results/lambda_holdout.json")
_lhc = set(_lhj["common"])
_lhg = [g for g in _lhj["gradings"] if f"{g['runs']}|{g['grader']}" in _lhc]
_lhs, _lhi = _lhj["summary"]["common"], _lhj["summary_intervals"]["common"]
_lhshift = sorted(abs(g["lambda"]["non-moved keys"] - g["lambda"]["in sample"]) for g in _lhg)
_lhbias = {k: float(_np.mean([g["predicted"]["non-moved keys"] - g["observed"] for g in _lhg if g["kind"] == k]))
           for k in {g["kind"] for g in _lhg}}
_lhin = sum(g["prediction_interval"]["non-moved keys"][0] <= g["observed"] <= g["prediction_interval"]["non-moved keys"][1]
            for g in _lhg)
_lmae = lambda k: f"${_lhs[k]['mae']:.1f}$ $[{_lhi[k]['mae'][0]:.1f},{_lhi[k]['mae'][1]:.1f}]$"
check(_lhj["all_reproduce_figure"] and len(_lhg) == _lhs["non-moved keys"]["n"] == 42
      and f"Estimated instead from a grading's selections on the unchanged and inward keys alone, $\\lambda$ "
          f"moves by a median of ${_np.median(_lhshift):.3f}$ and at most ${max(_lhshift):.3f}$. Its "
          f"predictions of the moved keys' gain err by {_lmae('non-moved keys')} points on average over the "
          f"$42$ gradings that also graded those keys, against ${_lhs['in sample']['mae']:.1f}$ in sample, "
          f"{_lmae('constant share')} for a constant share of the rule's gain and {_lmae('kind share')} for "
          f"the mean share of the grading's kind" in _flat
      and f"It under-predicts the code-free gradings by ${-_lhbias['code-free']:.1f}$ points and over-predicts "
          f"those with the within-$5\\%$ option by ${_lhbias['none within 5%']:.1f}$, and the observed gain "
          f"lies within the interval of its prediction for ${_lhin}$ of the $42$." in _flat
      and _lhbias["code-free"] < 0 < _lhbias["none within 5%"]
      and _lhs["kind share"]["mae"] < _lhs["non-moved keys"]["mae"] < _lhs["constant share"]["mae"]
      and _lhi["mae difference, constant share minus non-moved keys"][0] > 0,
      "the account of the proximity weight out of sample no longer matches lambda_holdout.json")

# --- the pre-specified test with the number read as the tolerance reads it (extraction_verdicts.py) --------------
import extraction_verdicts as _evm  # noqa: E402
_evj = load("results/extraction_verdicts.json")
_evb = " ".join(_table_body("tab:lastnumber").split())
_evr = _evm.table_rows(_evj)
check(all(" ".join(r.split()) in _evb for r in _evr) and _evj["check"]["reproduced"] and not _evj["check"]["mismatches"],
      "tab:lastnumber's rows no longer match extraction_verdicts.table_rows, or the whole-answer reading no longer "
      "reproduces the test")
# the paper's verdict: H2 fails wherever its interval excludes zero
_verdict = lambda h, x: (x["lo"] <= 0 <= x["hi"]) if h == "H2" else x["pass"]
_evs = _evj["summary"]
_h4 = _evs["H4"]
check(all(_verdict(h, v["whole answer"]) == _verdict(h, v["last number"]) for h, d in _evs.items() for v in d.values())
      and f"every hypothesis keeps its verdict on every data set, and the moved keys gain more against $P$: "
          f"${_h4['v1.0: gpt-4o, Claude 3.5']['last number']['mean']:+.1f}$ and "
          f"${_h4['v1.5: new seeds']['last number']['mean']:+.1f}$ on the published runs and the new seeds "
          f"(Table~\\ref{{tab:lastnumber}})." in _flat
      and all(_h4[s]["last number"]["mean"] > _h4[s]["whole answer"]["mean"] for s in ("v1.0: gpt-4o, Claude 3.5", "v1.5: new seeds")),
      "sec:withdata's last-number reading of the pre-specified test no longer matches extraction_verdicts.json")

# --- degenerate runs and the per-miss trend (degenerate_runs.py) --------------------------------------------------
_dgc = _dgr["census"]
_dsum = lambda names, k="degenerate": sum(_dgc[n]["numeric"][k] for n in names)
_d7 = _dgr["rule_u_minus_p"]["seven configurations"]["run_sets"]
_dns = _dgr["rule_u_minus_p"]["new seeds"]["run_sets"]
_dq0, _dq1 = ["qwen3-235b|r0", "qwen3-235b|r1"], ["qwen3-235b|r2", "qwen3-235b|r3", "qwen3-235b|r4", "qwen3-235b|r5"]
_dt1, _dru = _dgr["table1"], _dgr["rule_u_minus_p"]
_dx = lambda g, w: _dt1[g][w]["forced"]["score_minus_tolerance"]
_dm = lambda g, w: _dru[g][w]["moved"]
_dsc = lambda v: f"${v['mean']:+.1f}$ $[{v['lo']:+.1f},{v['hi']:+.1f}]$"
check(len(_d7) == 7 and len(_dns) == 6
      and f"${_dsum(_d7)}$ of the seven configurations' ${_dsum(_d7, 'runs')}$ with-data runs on numeric questions "
          f"(${_dsum(_d7, 'no_answer')}$ without an answer), ${_dgc['qwen3a3b-react']['numeric']['degenerate']}$ of them "
          f"Qwen3-30B-A3B's and ${_dsum(['qwen72b', 'qwen72b-react'])}$ Qwen2.5-72B's; ${_dsum(_dns)}$ of the new seeds' "
          f"${_dsum(_dns, 'runs')}$; ${_dsum(_dq0)}$ and ${_dsum(_dq1)}$ of Qwen3-235B-A22B's first two and four later "
          f"runs; and none of gpt-5.1's." in _flat
      and _dsum(["gpt-5.1-react|r0", "gpt-5.1-react|r1"]) == 0
      and f"Without them the seven configurations' forced-choice excess rises from "
          f"${_dx('seven configurations', 'all runs')['mean']:+.1f}$ to {_dsc(_dx('seven configurations', 'without degenerate'))} "
          f"and the rule's moved-key $U-P$ from ${_dm('seven configurations', 'all runs')['mean']:+.1f}$ to "
          f"{_dsc(_dm('seven configurations', 'without degenerate'))}, the new seeds' from "
          f"${_dx('new seeds', 'all runs')['mean']:+.1f}$ and ${_dm('new seeds', 'all runs')['mean']:+.1f}$ to "
          f"${_dx('new seeds', 'without degenerate')['mean']:+.1f}$ and ${_dm('new seeds', 'without degenerate')['mean']:+.1f}$, "
          f"and Qwen3-235B-A22B's six runs' moved-key gain from "
          f"${_dm('Qwen3-235B-A22B, six runs', 'all runs')['mean']:+.1f}$ to "
          f"${_dm('Qwen3-235B-A22B, six runs', 'without degenerate')['mean']:+.1f}$." in _flat
      and f"the excess is {_dsc(_dx('use the data: three', 'all runs'))} and the moved-key gain "
          f"{_dsc(_dm('use the data: three', 'all runs'))} (${_dx('use the data: three', 'without degenerate')['mean']:+.1f}$ "
          f"and ${_dm('use the data: three', 'without degenerate')['mean']:+.1f}$ without degenerate runs)" in _flat
      and f"Leaving out the ${_dsum(_d7)}$ of the seven configurations' ${_dsum(_d7, 'runs')}$ runs that gave "
          f"no answer, reached the step limit or had a reply cut at the token limit raises their forced-choice "
          f"excess from ${_dx('seven configurations', 'all runs')['mean']:+.1f}$ to "
          f"${_dx('seven configurations', 'without degenerate')['mean']:+.1f}$ and the rule's moved-key gain "
          f"from ${_dm('seven configurations', 'all runs')['mean']:+.1f}$ to "
          f"{_dsc(_dm('seven configurations', 'without degenerate'))}." in _flat
      and abs(_dx("seven configurations", "all runs")["mean"] - _smt(_s7f)["mean"]) < 0.05,
      "the account of degenerate runs no longer matches degenerate_runs.json")
_dpc = lambda key, who="configurations": _dpm[key][who]
check(f"Over these nineteen configurations, gpt-5.1 and the two current agents, the share of misses accepted "
      f"as each is graded is unrelated to the share within $5\\%$ (Spearman "
      f"${_dpc('as graded|misses_accepted')['rho']:+.2f}$, $p={_dpc('as graded|misses_accepted')['p']:.2f}$; "
      f"${_dpc('as graded|misses_accepted', 'models')['rho']:+.2f}$ with each of eleven models once). Under "
      f"one code-free grader for all, over the twenty configurations it graded, it rises "
      f"(${_dpc('gemma-3-27b code-free|misses_accepted')['rho']:+.2f}$ for gemma-3-27b, "
      f"${_dpc('gpt-4o code-free|misses_accepted')['rho']:+.2f}$ for gpt-4o), because a more accurate agent's "
      f"misses more often lie nearest the key (${_dpk['rho']:+.2f}$, $p={_dpk['p']:.3f}$)." in _flat
      and _dpc("as graded|misses_accepted", "models")["n"] == 11 and _dpc("as graded|misses_accepted")["n"] == 22
      and all(_dpc(f"{g} code-free|misses_accepted")["n"] == 20 for g in ("gemma-3-27b", "gpt-4o"))
      and all(_dpc(f"{g} code-free|misses_accepted")["p"] < 0.05 for g in ("gemma-3-27b", "gpt-4o")),
      "the appendix's per-miss trend no longer matches degenerate_runs.json")

# --- the questions BixBench-Verified-50 re-checked (verified50.py) -------------------------------------------------
_v5j = load("results/verified50.json")
_v5o = _v5j["overlap"]
_v5x = lambda g, s="verified50": _v5j["table1"][g][s]["forced"]
_v5m = lambda g, s="verified50": _v5j["rule_u_minus_p"][g][s]["moved"]
check(len(_v5j["question_ids"]) == 50 and _v5o["in_v15"] == 50
      and f"${_v5o['numeric']}$ of the fifty are among v1.5's $105$ numeric items (${_v5o['numeric_capsules']}$ capsules): "
          f"${_v5o['by_group']['moved to an edge']}$ moved keys and ${_v5o['by_group']['kept']}$ unchanged" in _flat
      and len(_v5o["numeric_key_changed"]) == 1 and _v5o["numeric_key_confirmed"] == _v5o["numeric"] - 1
      and f"On these ${_v5o['numeric']}$ the forced-choice excess is {_dsc(_v5x('seven configurations')['score_minus_tolerance'])} "
          f"for the seven configurations and {_dsc(_v5x('new seeds')['score_minus_tolerance'])} for the new seeds, "
          f"against ${_smt(_s7f)['mean']:+.1f}$ and ${_dx('new seeds', 'all runs')['mean']:+.1f}$ on all $105$; the rule's "
          f"$U-P$ on the ${_v5o['by_group']['moved to an edge']}$ moved keys is {_dsc(_v5m('seven configurations'))} and "
          f"{_dsc(_v5m('new seeds'))}" in _flat
      and f"the seven configurations' forced-choice excess, {_dsc(_v5x('seven configurations')['score_minus_tolerance'])}, "
          f"and the rule's gain on the ${_v5o['by_group']['moved to an edge']}$ moved keys among them, "
          f"{_dsc(_v5m('seven configurations'))}, are as on all items, though gpt-5.1's excess there is "
          f"{_dsc(_v5x('gpt-5.1')['score_minus_tolerance'])} (Appendix~\\ref{{app:replication}})" in _flat
      and f"On the ${_v5o['numeric']}$ numeric items whose keys BixBench-Verified-50's experts re-checked" in _flat
      and _v5x("seven configurations")["score_minus_tolerance"]["lo"] > 0 and _v5m("seven configurations")["lo"] > 0
      and f"gpt-5.1, within $5\\%$ of the key on ${_v5x('gpt-5.1')['tolerance']:.1f}\\%$ of them, is scored "
          f"{_dsc(_v5x('gpt-5.1')['score_minus_tolerance'])} above the tolerance" in _flat
      and f"${_v5o['v10']['items']}$ of the ${_v5o['numeric']}$ are also v1.0 numeric items, on which the published forced "
          f"grades lie ${_v5j['table1_v10']['gpt-4o']['verified50']['forced']['score_minus_tolerance']['mean']:+.1f}$ and "
          f"${_v5j['table1_v10']['Claude 3.5 Sonnet']['verified50']['forced']['score_minus_tolerance']['mean']:+.1f}$ points "
          f"above the tolerance" in _flat,
      "the account of the questions BixBench-Verified-50 re-checked no longer matches verified50.json")

# numbers a one-digit mutation of this round's text left standing, each pinned to its source
_v5k = _v5j["keys"][_v5o["numeric_key_changed"][0]]
check(f"their keys are v1.5's but one, which a port's grading records give as ${_v5k['verified50']}$ where v1.5 has "
      f"${_v5k['v1.5']}$" in _flat and not _v5k["same"],
      "the appendix misquotes the one key BixBench-Verified-50's port gives differently")
check(f"The forced-choice excess on v1.0's ${sum(_cls.values())}$ other questions has the same source" in _flat
      and f"and ${_nnd(_G4, 'neither, not')['share']:.1f}$ and ${_nnd(_CL, 'neither, not')['share']:.1f}\\%$ on the "
          f"others; a grader shown no notebook does the same" in _flat,
      "the appendix's account of the questions whose options are not numbers is stale")
check(f"whose acceptance of correct answers it leaves between ${min(_gsp('k8-middle-s0.2', 'correct_accepted')):.1f}$ "
      f"and ${max(_gsp('k8-middle-s0.2', 'correct_accepted')):.0f}\\%$" in _flat
      and tuple(_odm.KS) == (4, 6, 8, 10)
      and "drawn uniformly from all $k$ ranks (\\emph{uniform}), from the $k-2$ middle ones (\\emph{middle}), or the "
          "key's released rank ($k=4$)" in _flat
      and _odm.TOL == 0.05 and "that are more than $5\\%$ from the key and of its sign. Numbers within $5\\%$ of each "
          "other are merged" in _flat,
      "app:design's or sec:design's account of the designs' range, spacing or correct answers is stale")
check("with H2 counted as failed wherever its interval excludes zero" in _flat
      and all(_evs["H2"][s]["last number"]["lo"] > 0 for s in ("v1.0: gpt-4o, Claude 3.5", "v1.5: Qwen3-235B-A22B"))
      and _evs["H2"]["v1.5: new seeds"]["last number"]["lo"] <= 0,
      "tab:lastnumber's caption no longer names the hypothesis it counts as failed on an interval excluding zero")

# --- two current agents (frontier_agents.py, tab:frontier) -------------------------------------------------------
import frontier_agents as _fam  # noqa: E402
_frj = load("results/frontier_agents.json")
_frA = {k: v for k, v in _frj["agents"].items() if k in _fam.REPORTED}
_frK = _frj["agents"]["Kimi-K3"]
_frb = " ".join(_table_body("tab:frontier").split())
_frr = _fam.table_rows(_frj)
_ford = sorted(_frA, key=lambda k: _frA[k]["within_5pct"]["mean"])
_fg = lambda f: [f(_frA[k]) for k in _ford]
_fw = _fg(lambda a: a["within_5pct"]["mean"])
_fo = _fg(lambda a: a["open"]["mean"])
_ffx = _fg(lambda a: a["table1"]["gpt-4o forced"]["score_minus_tolerance"]["mean"])
_frx = _fg(lambda a: a["table1"]["gpt-4o may-decline"]["score_minus_tolerance"]["mean"])
_fcx = _fg(lambda a: a["corrected"]["corrected_minus_tolerance"]["mean"])
_fku = _fg(lambda a: a["table1"]["gpt-4o forced"]["miss_rates"]["nearest"]["accepted"])
_fot = _fg(lambda a: a["table1"]["gpt-4o forced"]["miss_rates"]["other"]["accepted"])
_fum = _fg(lambda a: a["rule|repaired-placebo|moved"]["mean"])
_fgm = _fg(lambda a: a["gpt-4o|repaired-placebo|moved"]["mean"])
_s1 = lambda xs: f"${min(xs):+.1f}$ to ${max(xs):+.1f}$"
_p1 = lambda xs: f"${min(xs):.1f}$ to ${max(xs):.1f}\\%$"
_fch = [_frA[k]["median_answer_chars"] for k in _fam.REPORTED]
check(sorted(_frj["agents"]) == sorted(label for _, label in _fam.MODELS) and sorted(_frA) == sorted(_fam.REPORTED)
      and all(a["n_runs"] == a["n_items"] >= 90 for a in _frj["agents"].values())
      and all(" ".join(r.split()) in _frb for r in _frr) and len(_frr) == 2 and _frb.count("\\\\") == len(_frr) + 2,
      "tab:frontier's rows no longer match frontier_agents.table_rows")
_fset = {m: json.loads(gzip.open(f"results/frontier_agents/trajectories_{m}.jsonl.gz", "rt").readline())["settings"]
         for m in ("gpt-6-luna-react", "DeepSeek-V4-Pro-react", "FW-Kimi-K3-react")}
_s51set = json.loads(gzip.open("results/strong_agent/trajectories_gpt-5.1-react.jsonl.gz", "rt").readline())["settings"]
_lset, _dset, _kset = _fset["gpt-6-luna-react"], _fset["DeepSeek-V4-Pro-react"], _fset["FW-Kimi-K3-react"]
# DeepSeek-V4-Pro's runs are its single-driver rerun; the record says what the first runs were and why they were replaced
_frec = load("results/frontier_agents/reruns_DeepSeek-V4-Pro-react.json")
_fterm = collections.Counter(json.loads(_l)["termination"] for _l in
                             gzip.open("results/frontier_agents/trajectories_DeepSeek-V4-Pro-react.jsonl.gz", "rt"))
_fnrun = sum(_fterm.values())
_fsup = "results/frontier_agents/" + _frec["superseded"]["file"]
check(Path(_fsup).exists() and hashlib.md5(Path(_fsup).read_bytes()).hexdigest() == _frec["superseded"]["md5"]
      and _frec["superseded"]["episodes_by_driver"] and len(_frec["superseded"]["episodes_by_driver"]) == 2
      and _frec["rerun"]["settings_as_the_first_driver"] and _frec["rerun"]["started"].startswith("2026-10-01")
      and "27 Sep 2026" in _frec["note"] and _frec["rerun"]["episodes"] == _fnrun == 105
      and dict(_fterm) == _frec["rerun"]["terminations"] and _frA["DeepSeek-V4-Pro"]["n_runs"] == _fnrun
      and all(Path("results/frontier_agents", _x.split(" ")[0]).exists()
              for _x in (_frec["rerun"]["command"], _frec["rerun"]["grading"])),
      "DeepSeek-V4-Pro's rerun record, its superseded runs or its scripts no longer match what ships")
_fsr0 = {(r["block"], r["group"], r["reading"]): r for r in load("results/formula_scoring.json")["rows"]}
_fnp = [100 - _fsr0[("current", k, "gpt-4o forced")]["misses_selected"] for k in _ford]
_fpara = (
    f"Outside the pre-specified test, gpt-6-luna and DeepSeek-V4-Pro ran BixBench's published agent with the data "
    f"through Azure AI Foundry, once on each of v1.5's ${len(bw.option_sets())}$ numeric questions, and gpt-4o graded "
    f"every run as it graded gpt-5.1's (Table~\\ref{{tab:frontier}}). gpt-6-luna ran with gpt-5.1's settings: "
    f"{_lset['reasoning_effort']} reasoning effort, through the Responses API, where that deployment takes function "
    f"tools with reasoning, a context of ${_th(_lset['max_model_len'])}$ tokens and ${_th(_lset['view_budget'])}$ "
    f"characters of the notebook in view. DeepSeek-V4-Pro ran through chat completions with no reasoning effort set, a "
    f"context of ${_th(_dset['max_model_len'])}$ tokens and ${_th(_dset['view_budget'])}$ characters in view. Its first "
    f"runs, on 27 September, had two drivers writing to one directory, each overwriting episodes the other had "
    f"finished, and ${len(_frec['superseded']['content_filter_refusals_not_recorded'])}$ episodes that the content "
    f"filter refused ended without a record and were run again. We ran all ${_fnrun}$ again on 1 October with one "
    f"driver and those faults fixed, graded them as before, and report these; both sets ship with the code. Of the "
    f"${_fnrun}$, ${_fterm['submitted']}$ submitted an answer, ${_fterm['max_steps']}$ reached the step limit, "
    f"${_fterm['wallclock']}$ the hour and ${_fterm['content_filter']}$ was refused by the content filter; a run "
    f"without an answer is graded wrong, as BixBench grades it. Their numbers are within $5\\%$ of the key on "
    f"{_p1(_fw)} of items, more often than gpt-5.1 and every configuration of Table~\\ref{{tab:scaling}}, and gpt-4o "
    f"grades {_p1(_fo)} of their answers correct open-ended. Forced-choice grading still scores them {_s1(_ffx)} "
    f"points above the tolerance and, with a refusal option, {_s1(_frx)}; corrected for guessing, {_s1(_fcx)}. It "
    f"accepts {_p1(_fku)} of their misses whose nearest option is the key and {_p1(_fot)} of those nearest another. "
    f"gpt-4o selects no option on {_p1(_fnp)} of their misses and on "
    f"${100 - _fsr0[('v1.5 closed', 'gpt-5.1', 'gpt-4o forced')]['misses_selected']:.1f}\\%$ of gpt-5.1's, which "
    f"BixBench's parser scores wrong, where BixBench's own gpt-4o grades did on "
    f"${100 - _fsr0[('v1.0', 'gpt-4o', 'published forced')]['misses_selected']:.1f}\\%$ of the published runs'. "
    f"The rule's $U-P$ on the moved keys is {_s1(_fum)}, and gpt-4o's {_s1(_fgm)}; over all numeric items the rule's "
    f"is {_s1(_fg(lambda a: a['rule|repaired-placebo|all']['mean']))} and gpt-4o's "
    f"{_s1(_fg(lambda a: a['gpt-4o|repaired-placebo|all']['mean']))}. Kimi-K3, run with DeepSeek-V4-Pro's settings "
    f"on the ${_frK['n_items']}$ questions it finished before its runs were stopped, by two drivers as DeepSeek-V4-Pro's "
    f"first runs were, writes long answers that state "
    f"its number first (median ${_frK['median_answer_chars']}$ characters, against ${_fch[0]}$ and ${_fch[1]}$): "
    f"gpt-4o grades ${_frK['open']['mean']:.1f}\\%$ of them correct open-ended, but the last number, which the "
    f"tolerance reads, is within $5\\%$ of the key in ${_frK['within_5pct']['mean']:.1f}\\%$, and the first number in "
    f"${_frK['within_5pct_first']['mean']:.1f}\\%$, so we leave it out.")
check(_fpara in _flat
      and min(_fw) > _s51f["tolerance"] and min(_fw) > max(r["within_5pct"] for r in _rss["run_sets"].values())
      and _frK["median_answer_chars"] > 5 * max(_fch) and _frK["open"]["mean"] > _frK["within_5pct"]["mean"] + 15
      and _frK["within_5pct_first"]["mean"] > _frK["within_5pct"]["mean"] + 15
      and min(_ffx) > 0 and all(_frA[k]["table1"]["gpt-4o forced"]["score_minus_tolerance"]["lo"] > 0 for k in _frA),
      "app:replication's account of the two current agents no longer matches frontier_agents.json")
# Section 7 repeats Kimi-K3's two shares; it said 30.5% after frontier_agents.json said 32.6%.
check(f"A fourth current agent, Kimi-K3, is left out because its long answers state the number first: gpt-4o grades "
      f"${_frK['open']['mean']:.1f}\\%$ of them correct open-ended, while the last number in the answer is within "
      f"$5\\%$ of the key in ${_frK['within_5pct']['mean']:.1f}\\%$" in _flat,
      "sec:limitations' Kimi-K3 sentence no longer matches frontier_agents.json")
check(f"one run on each of v1.5's ${len(bw.option_sets())}$ numeric questions, graded by gpt-4o" in _flat
      and _lset["reasoning_effort"] == _s51set["reasoning_effort"] == "medium"
      and all(_lset[k] == _s51set[k] for k in ("max_tokens", "max_model_len", "view_budget", "protocol"))
      and _dset["reasoning_effort"] is None and _kset["reasoning_effort"] is None
      and all(_dset[k] == _kset[k] for k in ("max_tokens", "max_model_len", "view_budget", "protocol"))
      and all(_dset[k] == _s51set[k] for k in ("max_tokens", "protocol")) and _dset["max_model_len"] != _s51set["max_model_len"]
      and _lset["rollouts"] == _dset["rollouts"] == _kset["rollouts"] == 1
      and len({json.loads(_l)["settings"]["work"] for _l in
               gzip.open("results/frontier_agents/trajectories_FW-Kimi-K3-react.jsonl.gz", "rt")}) == 2,
      "app:replication misstates how the two current agents were run")
_cur3w = [_s51f["tolerance"]] + list(_fw)
_cur3x = [_s51x["mean"]] + list(_ffx)
check(f"The three current agents reach ${min(_cur3w):.0f}$ to ${max(_cur3w):.0f}\\%$ and are still scored "
      f"${min(_cur3x):+.1f}$ to ${max(_cur3x):+.1f}$ points above the tolerance forced "
      f"(Appendix~\\ref{{app:replication}})." in _flat,
      "the limitations' account of the two current agents is stale")

# --- section 2: where the rule's extremes differ from the value rank --------------------------------------------
def _mag_vs_rank(sets):
    below, zero = 0, 0
    for opts in sets:
        v = _odm.values_of(opts)
        key, s = v[0], sorted(v)
        same = [x for x in v if x != 0 and (x > 0) == (key > 0)]
        by_value = key in (s[0], s[-1])
        by_magnitude = key != 0 and abs(key) in (min(map(abs, same)), max(map(abs, same)))
        below += (not by_value) and by_magnitude
        zero += key == 0 and by_value
        assert by_value == by_magnitude or key == 0 or (not by_value and by_magnitude)
    return below, zero
_mb15, _mz15 = _mag_vs_rank([s["released"] for s in bw.option_sets().values()])
_mb10, _mz10 = _mag_vs_rank([s["released"] for s in _repm.v10_sets().values()])
check(f"which differs from its rank among the values where zero or a negative option lies below a positive key "
      f"(${_mb15}$ of v1.5's $105$ released keys, ${_mb10}$ of v1.0's $159$) and where the key is zero (${_mz15}$ and "
      f"${_mz10}$), at distance $1$ from every nonzero answer" in _flat,
      "section 2's count of keys whose extremes by magnitude differ from their value rank is stale")

print("Validated: schema and manifest, pinned digests, vendored evidence; every table, caption")
print("and quoted number in main.tex against its result file -- Lemma 1's enumeration and the")
print("distinctness rule, the bracketing floors and the nine-file survey, BixBench's published")
print("no-data arms, the thirteen models without the question, the deleted-option arm on six")
print("open models, BixBench's own agent with and without the data through three option sets,")
print("what the released options' forced grading accepts, how a number is read from an answer,")
print("who pays for a hidden rank, the in-context panel, the other references, the claim family")
print("and the bootstrap's calibration; the data checks on analyses the paper no longer reports;")
print("citation keys and figures.")


if _COLLECT and _FAILURES:
    sys.exit(1)
