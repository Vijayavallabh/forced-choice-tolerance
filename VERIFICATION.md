# Verification notes

Every claim in `main.tex`, with the file and command that produced it.

## The current paper, claim by claim

The sections after this one are the long record and still hold their
commands; where one disagrees with this list, this list is current.

| Claim in the paper | Where | Command | Result file |
|---|---|---|---|
| Gamma and Lemma 1; exchangeability sufficient and strictly stronger (an exact enumeration at k = 3 over ten values); the distinctness trap and its growth with the key marginal's concentration (the enumeration and the trap are not in the paper) | App. B | `python3 writer_theory.py --exact` | `results/writer_theory.json` |
| v1.5's key-rank law (the second-smallest option is the key on 54 of 105), 81.0% bracketed, the cluster-robust test of a uniform rank, the second-smallest rule and its 46 leave-one-capsule-out folds | abstract, §4.3, Fig. 1b | `python3 channel_survey.py`; `python3 option_artifacts.py` | `results/channel_survey.json`, `results/option_artifacts.json` |
| Nine released files: bracketing floor against the best rank rule, the uniform-rank test with items clustered and as independent, the rule chosen on held-out groups, the in-sample maximum under a uniform rank | §4.3, §5, App. B, App. C.3, Table 13 | `python3 channel_survey.py` (`--latex` prints Table 13) | `results/channel_survey.json` |
| BixBench's agent with and without the data, read through three option sets | §3.1, §4.4, Table 1, Fig. 8, Table 18, App. A.1, App. C.4 | `bixbench_agent.py` (GPU), then `python3 bixbench_withdata.py` | `results/bixbench_withdata.json`, `results/agent_runs/` |
| The moved-key split, each group at the level covering 95% on its own capsules (97.75% on 21, 98% on 10, 96.75% on 37; the whole file's 97% on 39 is not in the paper); the miss distances and open side of the newly credited answers; the placebo contrast; the rewritten distractors' distance from the key; the fewest-digits rule on each option set and its margin over chance | abstract, §3.3, §4.4, Table 19, App. A.3, App. C.4 | `python3 bracketing.py` (`--latex` prints Table 19, after the rows of the moved-key table the paper no longer prints) | `results/bracketing.json` |
| Both releases' no-data runs, three arms each; the model-free regrade (the three verifiers, and what recall at the rate a model states unaided would add to a forced score, are not in the paper) | App. C.3, Table 15 | `python3 release_arms.py` (`--latex` prints Table 15); `python3 published_repair.py` | `results/release_arms.json`, `results/published_repair.json` |
| The published runs split by the key's rank, against what a rank reader would show; open models through the rewritten options; thirteen question-withheld readers' picks by rank | App. C.3, Table 16, Fig. 6a | `python3 rank_attribution.py` (`--latex` prints Table 16, then the rewritten options' table the paper no longer prints) | `results/rank_attribution.json` |
| The deleted-option arm on six open models (its forced cell through the placebo and the repair is not in the paper) | App. C.3, Table 11 | `run_free_response.sh` and `run_free_rewritten.sh` (GPU), then `python3 free_grid.py` (`--latex` prints a table the paper no longer prints) | `results/free_response*.json`, `results/agentic_dumps/free_*` |
| The 2^5 prompt factorial, three models, two files | not in the paper | `prompt_factorial.py` (GPU), then `python3 prompt_factorial_summary.py --latex` | `results/prompt_factorial_summary.json` |
| Thirteen models without the question on BixBench, over chance and paired, pooled by random effects, and the MMLU-Pro grid | not in the paper (Fig. 6a draws their picks by rank from `results/rank_attribution.json`) | `python3 prompt_contrasts.py --latex mmlupro` / `bixall` | `results/prompt_contrasts.json` |
| Controls against their exact key-randomisation null (3 of the BixBench grid's 26) | not in the paper | `python3 control_null.py` | `results/control_null.json` |
| Six writers on MMLU at the level covering the frontier's 20 subjects | not in the paper | `python3 frontier_calibrated.py` | `results/frontier_calibrated.json` |
| KEY MARGINAL as built (Gamma +7.1) and rebuilt (+0.65), and its cost | not in the paper | `python3 key_marginal_gamma.py`; `python3 key_marginal_symmetric.py --cost` | `results/key_marginal_gamma.json`, `results/key_marginal_symmetric.json` |
| Proposition 2(iii) against the built arm; the collision bound | not in the paper | `writer_theory.py --empirical --model M` (GPU, one solver each); `python3 collision_bound.py` | `results/writer_price.json`, `results/collision_bound.json` |
| Coverage of the cluster bootstrap and the levels that cover | App. A.4 | `python3 bootstrap_calibration.py` | `results/bootstrap_calibration.json` |
| The ten headline findings (the abstract's and the introduction's, and the two no-data margins App. C.3 rests on), corrected together (eight survive), and each at the nominal 95% (all ten clear) | §3.4, §4.1, App. A.4, Table 8 | `python3 claim_budget.py` (about 25 minutes on CPU; `--latex` prints Table 8) | `results/claim_budget.json` |
| The registered replication on BixBench v1.0's published gpt-4o and Claude 3.5 Sonnet runs: H1-H5, per run set, single-run spread | §3.4, §4.4, Fig. 4, Tables 20 and 27, App. C.4 | `python3 replication.py analyse` (the plan: `PREREGISTRATION.md`) | `results/replication.json`, `results/bixbench_v10_published_runs.json.gz` |
| The registered replication on new v1.5 runs (Qwen3-235B-A22B; two new seeds of three agents): H1-H6, run to run, and the placebo's own change beside them (not registered) | §3.4, §4.4, Fig. 4, Tables 20 and 21, App. C.4 | `python3 replication.py pack` then `analyse` (runs: `bixbench_agent.py`, see below) | `results/replication.json`, `results/agent_runs_replication/` |
| The repair redrawn at 100 seeds | App. C.4, Table 27 | `python3 repair_seeds.py --seeds 100` | `results/repair_seeds.json` |
| The share of misses the nearest-option rule accepts through R and U at 1, 2, 5 and 10% (the writer panel is not in the paper) | App. C.1, Table 11 | `python3 writer_panel.py` | `results/writer_panel.json` |
| The tolerance against BixBench's range keys and graders, and where the 5% rule and the v1.0 grader disagree; the paper's tolerance verdicts at 1, 2, 5 and 10%, the readings of Table 2 and the writers' credited misses among them | §3.2, §4.1, Tables 6 and 11, App. A.2, App. C.1 | `python3 score_decomposition.py` and `python3 writer_panel.py` first, then `python3 tolerance_check.py` (`--latex` prints Tables 6 and 11) | `results/tolerance_check.json` |
| Where v1.5's key rank came from: the v1.0 items it kept, dropped and rewrote | §4.3, App. C.3, Table 14 | `python3 leak_origin.py` (`--latex` prints Table 14) | `results/leak_origin.json` |
| The with-data contrasts under the rule, each run set's own reader and gemma-3-27b, split by key group, with capsule sign-flip tests (the pools over one protocol per family are not in the paper) | §4.4, Table 23, App. C.4 | `python3 reader_split.py` (`--latex-full` prints Table 23's rows; `--pools` adds the one-protocol pools alone) | `results/reader_split.json` |
| The registered runs read by gemma-3-27b through `MCQ_EVAL_PROMPT` (not registered), beside the published reading | §4.4, Tables 22 and 23, App. C.4 | `python3 published_reads.py build`, `read` (GPU), `merge`, `analyse` | `results/published_reads_rows.jsonl.gz`, `results/published_reads.json` |
| The capsule sign-flip test and the interval it inverts to; its coverage on simulated moved-key files | App. A.4, Table 23 | `python3 randomization.py` (its self-check; the tests are run by `reader_split.py` and `published_reads.py`) | `results/randomization_selfcheck.json` |
| Who comes first under each reading: Kendall tau on v1.5 with the tolerance's paired difference and a noise-matched tolerance, gpt-4o minus Claude on v1.0 (in the paper only as the statement that no published ranking is shown reversed) | §5 | `python3 ranking_check.py` (`--latex` prints a table the paper no longer prints) | `results/ranking_check.json` |
| The contrasts without the capsules a concurrent audit re-derived; the newly credited answers on the keys it flags (not in the paper) | App. C.4, Table 27 | `python3 key_audit_check.py` | `results/key_audit_check.json` |
| What BixBench's reading of its released options does to a score: every with-data run on a numeric question beside the 5% tolerance, under the published forced and may-decline readings (v1.0) and the paper's readers (v1.5); the credited misses split by what the answer gives the reader (the key nearest its last number, no number, a number nearer another option) and how often each kind is read as the key; the share of the misses read (an empty answer is not read) that each reading credits, over all and by kind (the key the single nearest option, another option, neither), beside what a reading choosing at random would credit (`miss_rates`); what the published forced readings pick on a miss with a single nearest option (`picks_on_misses`, not in the paper); on the last kind, whether the key's value is written in the notebook (and whether a rescaled number lands near it, not in the paper); the reading's pick against the nearest option, and whether a non-nearest pick is written in the notebook; the excess at 1, 2, 5 and 10% | abstract, §1, §4.1, Table 2, App. C.1 | `python3 published_reads.py build-decline` (needs `eval_df.csv`), then `python3 score_decomposition.py` (`--latex` prints Table 2, `--latex-picks` the picks' table the paper no longer prints; `--notebook-values` rebuilds the notebook record from the trajectories) | `results/score_decomposition.json`, `results/published_reads_published_decline.jsonl.gz`, `results/score_decomposition_notebook_values.jsonl.gz` |
| The forced-choice excess against other references: BixBench's own open-ended grading of the same runs, any number within 5%, without the p-value keys, and those keys graded within a factor of ten; the answers with no number against the notebook and the open-ended graders; the cost against accuracy with each agent model counted once (the published grades' rates applied to the open-ended accuracy a 2026 system reports are computed but not in the paper) | §4.1, App. C.1, Table 9 | `python3 score_decomposition.py` and `python3 run_set_scaling.py` first, then `python3 reference_check.py` (`--latex` prints Table 9) | `results/reference_check.json` |
| The registered runs and the seven run sets read by Qwen2.5-72B and Llama-3.3-70B, and by gemma-3-27b with BixBench's decline option (not registered); the Qwen3-235B-A22B reads set aside | §4.4, Tables 22 and 23, App. C.4 | `python3 published_reads.py read --reader NAME=URL --mode forced\|decline [--d0\|--d2] [--chunks 0-4 --of 20] [--moved-only] [--data-only]` (GPU), then `merge` and `analyse` with `--reader-name NAME --mode MODE` | `results/published_reads_<reader>_<mode>.json`, `results/published_reads_<reader>_<mode>_rows.jsonl.gz`, `results/published_reads_qwen3-235b_attempt.jsonl.gz` |
| Each reader of the published runs beside the published readings on the quarter all of them read: how often it names the option nearest the number, over all numeric answers and on misses; its agreement with the published reading (kappa); the rule's own agreement; the may-decline reader's agreement over the three outcomes a read can have (key, other option, decline) (the agreements are not in the paper) | §4.4, Tables 22 and 23, App. C.4 | `python3 published_reads.py compare` | `results/published_reads_readers.json` |
| How a number is taken from an answer: each run set's answers by kind (empty, no number, a number, one number in text, several); the rule's headline contrasts read as registered, from the last number as the tolerance reads it, on single-number answers, and without the keys that are zero or negative (not registered) | §3.2, §3.3, §4.1, §4.4, §5, App. A.2, Tables 7 and 21 | `python3 answer_extraction.py` (`--latex` prints Table 7) | `results/answer_extraction.json` |
| Knowledge short of a value: BixBench's own options-deleted replies that state a number, read by their nearest released option; the forced score that reading predicts; on v1.0, the forced pick against the option nearest the model's own number (not registered) | not in the paper | `python3 partial_knowledge.py` | `results/partial_knowledge.json` |
| Who pays for a hidden rank: per run set with the data, the rule's moved-key and whole-file contrasts and the moved-key contrast per miss, against how often the run set is within 5% (Spearman over nineteen run sets; not registered) | §4.4, §5, App. C.4, Fig. 9 | `python3 run_set_scaling.py` (`make_figures.py` draws Fig. 9 from its result; `--latex` prints the table Fig. 9 replaced) | `results/run_set_scaling.json` |
| Solved examples in context: eight open models shown up to 64 solved v1.5 items from other capsules, the examples' keys at their released ranks or redrawn to a uniform rank (not registered) | App. C.3, Fig. 6b | `icl_probe.py --models M --shots 0 8 32 64` (GPU; `--engine vllm --tp 2 --quantization fp8` for the 70B-class models), then `python3 icl_analysis.py` (`make_figures.py` draws Fig. 6b from `results/icl_probe.json`; `--latex` prints the table Fig. 6b replaced) | `results/icl_probe.json`, `results/probe_icl/` |
| A learned option-only reader on the released, placebo and repaired options | App. C.4, Table 27 | `python3 learned_probe.py --jsonl build/bixbench_numeric_{q,placebo,repaired}.jsonl --cluster-field cluster --output results/learned_probe_bix_optionsets_<set>.json` | `results/learned_probe_bix_optionsets_*.json` |
| gpt-4o (version 2024-11-20, through Azure OpenAI) as the forced multiple-choice grader of the published runs: a random quarter read in full and every moved-key run, through R, P and U; how often it selects the option nearest a miss, its agreement with the published grades, and how often it selects no option, by the kind of answer it grades, beside the published grades of the same runs, with what its replies without a letter say (`grader_declines.py`, which reads the recorded replies where they are present) | §4.4, §5, Tables 3, 22 and 23, App. C.4 | `AGENTICLS_OPENAI_BUDGET_USD=... python3 published_reads.py read --reader gpt-4o=$(python3 openai_api.py --print-base) --mode forced --chunks 0-4 --of 20`, again with `--chunks 5-19 --moved-only`, then `merge` and `analyse`; `python3 reader_split.py` | `results/published_reads_gpt-4o_forced.json`, `results/published_reads_gpt-4o_forced_rows.jsonl.gz`, `results/published_reads_readers.json`, `results/grader_declines.json` |
| gpt-4o without the data, forced, through R, P and U on v1.5's 105 and v1.0's 158 numeric items: BixBench's template with the question, our chat prompt, and the question withheld (not registered) | App. C.3, Table 17, Fig. 6a, Fig. 7 | `AGENTICLS_OPENAI_BUDGET_USD=... python3 openai_nodata.py --model gpt-4o`, then `--summarise` (`--latex` prints Table 17) | `results/openai_nodata_gpt-4o.json`, `results/openai_nodata/` |
| gpt-5.1 (version 2025-11-13) running BixBench's ReAct agent with the data, two runs on each of v1.5's 105 numeric questions, graded by gpt-4o forced, with a refusal option and open-ended: its share within 5%, the forced and refusal-option excess and what it credits (Table 2's gpt-5.1 rows), the rule's and gpt-4o's U - P (not registered; the report's extrapolation of the nineteen run sets' trend to its share is not in the paper) | abstract, §3.1, §4.1, §5, Tables 1, 2, 3 and 25, App. C.4 | `bixbench_agent.py --model gpt-5.1 --base $(python3 openai_api.py --print-base) --protocol react --condition data --only-file build/openai/v15_numeric_question_ids.txt --rollouts 2 --max-tokens 32768 --max-model-len 272000 --view-budget 300000 --reasoning-effort medium --out build/openai/agent_runs --work build/openai/bixbench-work` (Docker), then `python3 bixbench_withdata.py --runs build/openai/agent_runs --models gpt-5.1-react --reader gpt-4o=BASE --judge gpt-4o=BASE --cache build/openai/reader_cache_gpt51.jsonl --rows build/openai/withdata_rows_gpt51.json --out build/openai/withdata_gpt51.json`, then `python3 strong_agent.py` (`--latex` prints Table 25) and `python3 score_decomposition.py` | `results/strong_agent.json`, `results/strong_agent_rows.json.gz`, `results/strong_agent/` |
| How $P$ and $U$ are built: one generator (`mcq_audit.redraw_row`) asked for the key's released rank or a uniformly drawn one; its constants; the items that took another rank or were left out; the four option files regenerated exactly from their seeds | §3.3, App. A.3 | `python3 mcq_audit.py --jsonl build/bixbench_numeric_q.jsonl --repair build/bixbench_numeric_repaired.jsonl` (seed 20260918) and `mcq_audit.apply_repair(..., preserve_rank=True)` at seed 20260920 for v1.5; for v1.0 the commands in `PREREGISTRATION.md`; checked by `tests/test_option_sets.py` and `validate_artifact.py` | `build/bixbench_numeric_{q,placebo,repaired}.jsonl`, `build/bixbench_v10_*.jsonl` |
| BixBench's use in 2026: a system's forced and open-ended v1.5 scores and how its options are read; BixBench-Verified-50's curation, verifiers and distractors, and its scores | abstract, §1, §2, App. C.2, Table 27 | read by `validate_artifact.py` against the verbatim excerpts | `sources/cited/*.txt` |
| Remark 1: the proximity weight of every grading of the same with-data runs (q, the share of its selections on misses that are the nearest option, ties in d broken by the absolute distance; delta, its share of misses graded with no option), BixBench's own published grades (weight and prediction only, from R), and the gain the weight predicts beside the observed one | §3.3, §4.4, Fig. 3a, App. B, App. C.4, Tables 22 and 24 | `python3 proximity_weight.py` | `results/proximity_weight.json` |
| The digit-matched rewrites P' and U': the same generator at the same seeds with every new distractor written to the significant digits of the one it replaces; the fewest-digits rule on every option set (v1.5: 25.0, 25.3, 25.5%), U''s key groups (36, 12, 55), the nearest distractor's distance, and the nearest-option rule's contrasts through P' and U' on the published runs and every v1.5 group (not registered) | App. A.3, Table 24 | `python3 digit_matched.py build` then `analyse` | `build/bixbench_*_digits.jsonl`, `results/digit_matched.json` |
| The same with-data runs graded answer-only (question, options and answer, no notebook, the reply restricted to a letter; forced, with a refusal option, and with an option stating that none of the others is within 5%), by gpt-4o 2024-11-20 constrained to a letter with the notebook, and with the notebook through P' and U' (not registered): U - P and U' - P' on the moved keys with a capsule sign-flip test, each grading's proximity weight, the misses and correct answers accepted with the within-5% option, the letter-only grader beside the published grades of the same quarter, and every grading's score minus BixBench's open-ended grade over all questions | §3.1, §4.1, §4.4, App. C.1, App. C.4, Tables 10, 22 and 24 | `python3 grading_variants.py build-published`; `python3 grading_variants.py read --reader NAME=BASE --config codefree\|letter\|notebook --mode forced\|decline\|withintol --runs d1\|d0\|d2\|gpt51` (GPU servers or Azure OpenAI), then `merge`; `python3 grading_variants.py analyse`; `python3 grading_variants_analysis.py --latex readers2\|variants\|allq` | `results/grading_variants.json`, `results/proximity_cost.json`, `results/grading_variants_*_rows.jsonl.gz`, `results/published_forced_all_questions.jsonl.gz` |
| What graders select on answers with no number; how far the accepted misses are (and the share within BixBench's own range half-widths); the unchanged keys split by what U does to their rank | §4.1, §4.4, App. C.4 | `python3 miss_anatomy.py` | `results/miss_anatomy.json` |
| Four more runs of Qwen3-235B-A22B with the data (rollouts 2-5; not registered, apart from the pre-specified test's runs): the test's statistics on them and on all six pooled | App. C.4, Table 27 | `bixbench_agent.py --model qwen3-235b --protocol text --condition data --only-file build/openai/v15_numeric_question_ids.txt --first-rollout 2 --rollouts 4 --wallclock 10800 --out build/agent_runs_q235_extra` (GPU), then `python3 q235_extra.py pack` and `analyse` | `results/q235_extra.json`, `results/agent_runs_q235_extra/` |
| gpt-4o 2024-11-20 without the data or the options, asked for a number (and told a best estimate is required) on each release's numeric questions: the share giving a number, within 5% of the key, and whose nearest released option is the key; its forced choice under BixBench's template on the same questions beside choosing the option nearest its own number (not registered) | not in the paper | `AGENTICLS_OPENAI_BUDGET_USD=... python3 forced_guess.py run` (Azure OpenAI), then `python3 forced_guess.py analyse` | `results/forced_guess.json`, `results/forced_guess_replies.jsonl.gz` |
| Formula scoring (the correction for guessing) of every forced grading of Table 2 and gpt-4o's answer-only gradings of the same runs: the corrected score minus the tolerance with its calibrated interval, split exactly by the kind of answer; the share of correct answers accepted; the misses accepted over all and among those on which the grading selects an option; the omission-aware variant; the 2026 system's corrected score (not registered) | abstract, §1, §4.2, §5, Tables 2 and 12, App. C.2 | `python3 formula_scoring.py` (`--latex` prints Table 12) | `results/formula_scoring.json` |
| Other designs: every numeric item's distractors rebuilt for k = 4, 6, 8, 10 with the key's rank uniform over all or the middle ranks, at fixed spacings, and from other agents' wrong numbers (leave one model out); the rank rule's leak in theory and on held-out capsules; the rule's and four answer-only graders' acceptance of misses and correct answers; the uniform-minus-kept contrasts on the moved keys (not registered) | §4.5, Fig. 3b, App. C.5, Tables 28 and 29 | `python3 option_design.py build`, `rule`, `read --reader NAME=BASE` (GPU servers, or Azure OpenAI for gpt-4o with `--published-quarter`), `analyse`; `latex design\|pairs` | `results/option_design.json`, `results/option_design/` |
| The proximity weight fitted on each grading's selections on the unchanged and inward keys alone, its prediction of the moved keys' gain, against a constant share and the grading kind's mean share (not registered) | §4.4, App. C.4 | `python3 lambda_holdout.py` | `results/lambda_holdout.json` |
| The pre-specified test re-run with the number read as the tolerance reads it (the last number), beside the whole-answer reading it reproduces (not registered) | §3.3, §4.4, App. C.4, Table 21 | `python3 extraction_verdicts.py` (`--latex` prints Table 21) | `results/extraction_verdicts.json` |
| Degenerate runs (no answer, the step limit, a reply cut at the token limit, a wall clock) by run set, and Table 2's and the test's contrasts without them; the per-miss trend with accuracy over 22 run sets (not registered) | §4.1, App. C.4, Table 27| `python3 degenerate_runs.py` | `results/degenerate_runs.json` |
| The 26 numeric items BixBench-Verified-50 re-checked: Table 2's excess and the rule's moved-key contrast on them (not registered) | §5, App. C.4, Table 27 | `python3 verified50.py` | `results/verified50.json` |
| Two current agents (gpt-6-luna, DeepSeek-V4-Pro; Kimi-K3 run and left out) running BixBench's ReAct agent with the data, one run on each of v1.5's 105 numeric questions, graded by gpt-4o as gpt-5.1's runs are: within 5%, open-ended, the forced and refusal-option excess, the forced score corrected for guessing, the misses accepted, and U - P on the moved keys (not registered); DeepSeek-V4-Pro's runs are its single-driver rerun of 1 October, its first runs of 27 September having had two drivers | abstract, §3.1, §4.1, §5, Tables 1, 2, 3 and 26, App. C.4 | Runs: `bixbench_agent.py --model M --base B --protocol react --condition data --only-file build/openai/v15_numeric_question_ids.txt --rollouts 1 --max-tokens 32768` (gpt-6-luna with `AGENTICLS_OPENAI_RESPONSES=gpt-6-luna --reasoning-effort medium --max-model-len 272000 --view-budget 300000`; DeepSeek-V4-Pro with `--max-model-len 128000 --view-budget 200000`, rerun by `results/frontier_agents/rerun1_launch.sh` and graded by `rerun1_grade.sh`; Docker, Azure AI Foundry), then `python3 bixbench_withdata.py --runs build/openai/agent_runs --models M-react ... --reader gpt-4o=BASE --judge gpt-4o=BASE --cache build/openai/reader_cache_frontier.jsonl --rows build/openai/withdata_rows_frontier.json --out build/openai/withdata_frontier.json`, then `python3 frontier_agents.py` (`--latex` prints Table 26) | `results/frontier_agents.json`, `results/frontier_agents_rows.json.gz`, `results/frontier_agents/` (the rerun's record `reruns_DeepSeek-V4-Pro-react.json`; the two-driver runs in `superseded/`) |
| The questions whose options are not numbers: the published grades' excess over the open-ended ones against a random choice on every miss (their classes, the excess by class, and what the grades accept by the option nearest in value or wording, and by whether the model answers correctly without the data, are not in the paper; not registered) | App. C.1 | `python3 nonnumeric_excess.py build` (needs `eval_df.csv`), then `analyse` | `results/nonnumeric_excess.json`, `results/nonnumeric_published_picks.jsonl.gz` |
| What hiding the key's rank costs each grader: U - P on the moved keys and over all numeric items for every set of runs graded with the data, the nearest-option rule with its intervals and each other kind of grader (answer-only, with the notebook, refusal, within 5%) as the range of its graders, every cell read from the result an appendix table prints it from (not registered) | abstract, §1, §4.4, Table 3, Fig. 3a | `python3 cost_table.py` (prints Table 3's rows), after the analyses it reads | none of its own: it reads `results/grading_variants.json`, `results/reader_split.json`, `results/replication.json`, `results/published_reads*.json`, `results/degenerate_runs.json`, `results/strong_agent.json`, `results/frontier_agents.json` |
| How a number is read from an answer: the tolerance's reading (the last number in the answer; a fraction against a key written as a per cent, times 100; relative, and absolute for a key of zero) and the nearest-option distance d, compared on a log scale (an option of the other sign, or zero, at distance 1) | §1, §3.1, §3.2, §3.3, §5, App. A.2, App. B | `answer_numbers.py`, a module the analyses import (no command of its own); checked by `tests/test_answer_numbers.py` | none of its own |

`AGENTICLS_VALIDATE_COLLECT=1 python3 validate_artifact.py` lists every
failing pin in one pass instead of stopping at the first. The pins were
mutation-tested: every number in `main.tex`'s prose, and one cell in every table
row, was changed by one unit in its last digit, one at a time, and the validator
run again (table rows are also regenerated whole from their result files). It failed on every change to
a result; the changes it let through are to definitions -- chance rates, the 5%
tolerance, the 95% coverage target, and the constants in 1/k and 1-0.05/10.
The appendix's cut of 2 October 2026 was checked by coverage instead of a
new mutation run: every number in its rewritten prose and in Table 27 lies
inside a pinned string, apart from definitions and the settings of Table 5,
which were not pinned before the cut either.
The figures are drawn from the same result files by `make_figures.py`.

## Evidence identity

- BixBench revision `f8cc3bd…422dc95a` and its SHA-256 recorded in the
  manuscript, the manifest and the audit; all three agree.
- Every vendored external file (published zero-shot runs, LAB-Bench subsets,
  ScienceAgentBench, CORE-Bench, DiscoveryBench) carries its source URL and
  SHA-256 in `data/external/PROVENANCE.json`, pinned to upstream commit
  `49311180bdacb324c596f2e07596c126f2004008`.
- Offline API-cache replay reproduces `results/audit.json` byte for byte.
- Probe conditions are reproducible across processes: seeds derive from strings
  through `random.Random`, never from `hash()` of a tuple containing one, so
  `PYTHONHASHSEED` cannot change them. Checked by test.

## Census and provenance

- BixBench: 205 questions, 59 capsule IDs, 23 capsules / 68 questions with no
  source identifier; 36 capsules / 137 questions with external date evidence;
  16 connected identifier groups, largest 7 capsules.
- Cross-benchmark census: 12 released files, 2,558 tasks, **0** carrying any
  date field.
- All six published BixBench zero-shot accuracies reproduce to within 1e-9
  after a case-insensitive question-id join. A case-sensitive join drops
  `Bix-33-q6` and `Bix-47-q3`; the fix is covered by a regression test.
- No-data MCQ 36.1% [29.0, 43.1] and 34.1% [27.8, 41.1]; open-ended 2.9% for
  both models. ICC up to 0.32, design effect up to 1.79, effective sample
  115–200 of a nominal 205.

## The geometry channel

- BixBench v1.5, 105 four-option numeric items in 46 capsules: key-rank
  distribution (12.4, 51.4, 29.5, 6.7)%, interior rate 81.0% [72.4, 89.0],
  χ² = 51.0 with df = 3.
- Available geometry credit **[−18.3, +26.4] points**; the upper bound is
  `max p` and is attained by a point-mass preference.
- The "take the second-smallest" rule scores 51.4% [39.5, 63.4], and
  leave-one-capsule-out selection returns the same rank in all 46 folds.
- BixBench v1.0: (22.6, 30.2, 30.8, 16.4)%, available range [−8.6, +5.8], and
  the best in-sample rule (30.8%) collapses to 15.7% under cross-validation.
- Presentation-order control: key letter position uniform, χ² = 4.24, df = 3,
  p ≈ 0.24 — consistent with the bound being invariant to shuffling.
- Independent replication: LAB-Bench SeqQA interior rate 75.6% [62.5, 91.2]
  over 160 numeric items.
- Surface rules are not a second channel: each of the nine selects the
  largest-valued option on 43–84% of numeric items, so they track value rank.

## How open the channel is elsewhere

- 18 released multiple-choice benchmark files surveyed; 9 carry at least 20
  items whose *k* options are all distinct numbers. Nine further files
  (LAB-Bench DbQA, ProtocolQA, Cloning, FigQA, LitQA2, SuppQA, TableQA;
  ARC-Challenge; OpenBookQA) fall below that and are not assessed. GPQA is
  gated and returns HTTP 401 without credentials.
- **7 of 9 reject a uniform key rank** at the 5% level; p < 1e-10 for BixBench
  v1.5, LAB-Bench SeqQA and MMLU-Pro.
- **5 of 9 carry a positive lower bound** on the credit available to a
  rank-only solver, 4 of them on 10 or more source groups: BixBench v1.5
  ≥ +13.6 points, MMLU-Pro ≥ +8.4, SciQ ≥ +8.3, MMLU ≥ +1.8, and LAB-Bench
  SeqQA ≥ +1.9 — the last flagged, since 160 items in only 4 source groups make
  a group-resampling bootstrap untrustworthy.
- MMLU-Pro is the largest sample: 1,263 numeric items over 10 options, interior
  rate 94.2% against 80% expected, χ² = 575.1 with df = 9.
- **AQuA-RAT is a clean counterexample**: 141 numeric items, χ² = 1.2, p = 0.87,
  no positive bound. Its distractors are answers a wrong step produces rather
  than perturbations of the right one.
- Estimator choice is recorded because both obvious options fail. The plug-in
  maximum over estimated shares is biased upward. Selecting a rank on training
  folds and scoring it on a held-out fold is biased *downward* — a rank
  over-represented in training is by construction under-represented in the
  held-out fold — returning 17.5% on uniform synthetic data where chance is
  25%, and exactly 0% on MedQA in its leave-one-item-out form. Both failures
  are covered by tests. The reported bound is a simultaneous one-sided 95%
  cluster-bootstrap lower bound on max_k p_k at Bonferroni level α/k, which is
  conservative by construction.
- The survey and the standalone BixBench analysis agree on the shared rows to
  within 1e-9, checked by test and by the validator.
- Every vendored option set carries its dataset, config, split, parquet URLs and
  SHA-256 in `data/external/survey/PROVENANCE.json`; only option values, the
  keyed index and a cluster label are retained.

## The decomposition

- Additivity holds exactly for all 13 fitted solvers: recall + geometry equals
  the fitted margin to within 1e-9, checked by the validator.
- Estimator calibration at the published sample size (n = 296, 300 replicates):
  at λ = 0, mean λ̂ = 0.015 with median 0 and likelihood-ratio test size 5.3%
  against a nominal 5%; at λ = 0.10, λ̂ = 0.099 with power 83%; at λ = 0.20,
  λ̂ = 0.201 with power 100%.
- The preference `b` inferred where the question is shown agrees with the
  preference measured with the question withheld: median total-variation
  distance 0.06, maximum 0.20.
- Published runs (v1.0, 159 numeric items, 48 capsules): gpt-4o accuracy 30.2%,
  λ̂ = 0.069 [0, 0.158], geometry −0.3 points [−1.4, +1.1]; claude-3-5-sonnet
  27.7%, λ̂ = 0.028 [0, 0.119], geometry −0.7 [−2.4, +0.9]. Goodness-of-fit
  p = 0.41 and 0.54, so the two-source account is adequate for both tables.
- Open-weight panel: λ̂ reaches zero for nine of eleven models; Qwen2.5-14B
  0.130 [0.052, 0.203] and Qwen2.5-32B 0.168 [0.082, 0.251].
- Realised geometry credit over all 13 solvers spans −1.3 to +2.6 points,
  against +26.4 available.

## The decomposition validated where answers are known

- MMLU's 661 four-option numeric items, 39 subject groups, 10 redraws, four
  models, same five arms.
- **λ̂ orders exactly with accuracy**: 0.001 [0.000, 0.036] at 24.6%
  (Llama-3.2-1B), 0.167 [0.121, 0.232] at 37.7% (Llama-3.1-8B), 0.257
  [0.200, 0.331] at 44.4% (Qwen2.5-7B), 0.565 [0.447, 0.695] at 67.7%
  (Qwen2.5-32B).
- The same models on BixBench with the data withheld: λ̂ = 0.019, 0.025, 0.024
  and 0.168 — an order of magnitude lower for the three below 32B.
- Withholding MMLU's own question text drops all four to 22.2–25.3%, i.e. to
  chance, confirming the arm does what it is for.
- Geometry component +0.2 to +1.0 points, inside MMLU's own +1.8 bound.
- The value redraw dominates the rank effect on every model (−3.7 to −9.0
  points against −0.3 to −2.3), which is the measured case for the placebo
  arm: a naive original-vs-repaired contrast would attribute all of it to rank.
- The two-source account fits three of four (median per-redraw p 0.10–0.53) and
  is marginally rejected for Qwen2.5-32B (p = 0.024) — the most knowledgeable
  solver, and where unmodelled partial elimination should first appear.

## The repair at ten options

- `mcq_audit.py` audits any option count; BixBench results are unchanged by
  that generalisation (checked byte-for-byte against the pre-refactor output).
- MMLU-Pro, 1,263 ten-option numeric items in 67 source groups: key rank peaks
  at rank 4 (21.5%), interior rate 94.2% against 80% expected, available credit
  +11.5 points, verdict LEAKS.
- After repair: all 1,263 rewritten, credit +6.4 points — reduced, not closed.
- **Reachability** explains why: 82.9% of (item, rank) pairs are placeable,
  falling monotonically from 100% at rank 0 to 64% at rank 9, against 96.9%
  mean (93% worst) for BixBench's four options. Four generator variants
  (linear and geometric spacing, 24 and 60 attempts) give the same rate, so the
  binding constraint is arithmetic: an extreme rank needs k−1 same-style values
  on one side of the key, and rounding to the replaced options' written
  precision collapses them.

## The channel's other coordinates

- `mcq_audit.ORDERINGS` defines four orderings of the options — sorted value,
  lexical isolation, written length, significant digits — and is read by both
  the audit and the repair, so a coordinate cannot be audited without being
  repairable or repaired without being reported.
- **BixBench v1.5's 105 numeric items leak on three of the four** as released:
  value +26.4 (plug-in worst case), roundness +10.1, length +8.9 (simultaneous
  lower bounds). Lexical isolation reads −1.5, below zero and far inside the
  +1.5 that clean four-option files reach on that coordinate, so it is not
  established; the manuscript claims three. MMLU-Pro's 1,263: value +11.5,
  length +4.1, roundness +2.2, isolation −0.4, and the clean ten-option maximum
  is +1.0, so all three positive ones are established there.
- Across the 18 surveyed files the length channel's bound is positive in six of
  the nine assessable, against five for value; the six are those five plus
  MedMCQA, whose value channel is closed (−4.3) and whose length channel is not
  (+3.1). Length needs no numeric parsing, so it covers every item rather than
  the numeric subset — BixBench v1.5 reads +0.4 over all 205 items and +8.9
  over the 105 numeric ones.
- `channel_attribution.py` measures, per arm, what each coordinate *offers*
  (max p − 1/k) and what a solver *takes* (⟨p,b⟩ − 1/k), with p from the
  presented options and b from what was picked. Neither is fitted. The options
  are rebuilt from the question file, the seed and the redraw count and joined
  on (question, arm, redraw); the rebuild reproduces all 225,500 recorded key
  ranks of the four-option runs exactly.
- **With the question withheld on the released MMLU-Pro file**, Qwen2.5-7B and
  Llama-3.1-8B score 10.4% and 9.3% against 10% chance and take under ±1.7
  points on every coordinate. **Qwen2.5-32B does not**: it scores 13.4% and
  collects +2.3 points [+1.6, +3.1] of the value channel, a fifth of the +11.5
  available and the largest realised credit anywhere in this artifact. It is
  reported as the exception it is; the paper's claim is that the channel is
  *largely* unexploited, and the one solver that walks through the door is the
  most capable in the panel, which is the same direction the four-option rank
  effects point (+5.8 and +3.7 for the two largest).
- **After the value-rank repair** the two at chance score 14.1% and 18.4%
  (+3.7 and +9.1, both p = 0.0002, paired sign-flip over source groups), taking
  +1.7/+3.3 of length and +1.8/+3.2 of roundness; the 32B moves +1.1 and takes
  +1.6/+1.8. The repair leaves length exactly as wide and widens roundness
  (+4.6 → +9.8 plug-in).
- **Falsification test.** Attribution says isolation is *not* the coordinate
  being collected. A probe of a repair that uniformises value and isolation
  only should therefore change nothing, and does not: 14.1% → 14.5% and
  18.4% → 18.5%. Released-arm accuracies replicate the earlier run exactly
  (20.9%/10.4%, 17.9%/9.3%), which is the pipeline's own control.
- **Read-out precision, measured not assumed.** In bfloat16 the letter read-out
  depends on batch padding: the same prompt scored alone and in a batch padded
  two tokens wider moves a letter probability by up to 0.20, and one padded to
  fifteen times its length can change the arg-max (3 of 8 prompts). In float32
  the same comparison drifts by 3e-5, so this is precision and not a masking
  bug — passing mask-derived `position_ids` does not change it either. Prompts
  are therefore scored shortest-first, which keeps padding to a few tokens. The
  practical consequence: a run is deterministic given its settings, but changing
  the set of arms changes the batching, and about 3% of decisions then differ —
  0.12 points of pooled accuracy (20.932% vs 20.814% on the released arm across
  three runs), against reported effects of 3.7 and 9.1 points.
- **The four-ordering repair** returns BixBench v1.5 to *no leak detected*:
  value +26.4 → +4.5, isolation −1.5 → −4.5, length +8.9 → −8.5, roundness
  +10.1 → −6.2, no keyed answer altered, 0 of 105 items placed off their drawn
  value rank. These are the shipped repair's figures — the construction of
  finding 16, not the sampler that first carried the `--two-channel` flag.

- **Does the four-ordering repair leave solvers at chance?** At four options,
  yes. Probing BixBench's repaired file with the question withheld gives
  28.2% → 28.3%, 29.1% → 24.7%, 28.0% → 28.1% and 25.1% → 25.1% for
  Qwen2.5-14B/32B/7B and Llama-3.1-8B against 25% chance: nobody gains, and the
  largest model moves 4.4 points toward chance.
  Attribution on that file shows why: as released the four coordinates offer
  +26.4, +0.9, +8.2 and +20.7 points (plug-in worst cases) and the four solvers
  take at most +1.7; after the repair the same coordinates offer +1.0, +1.3,
  +3.3 and +4.4 and no solver takes more than +0.1.
- **At ten options, no.** The same repair on MMLU-Pro leaves Qwen2.5-7B and
  Llama-3.1-8B at +3.6 and +8.4 points above where the released file leaves
  them — the same as under the value-only repair (+3.7, +9.1), the
  value+isolation repair (+4.1, +9.2) and the value+isolation+length repair
  (+3.4, +8.4). Attribution on the four-ordering file shows every coordinate
  narrowed (value 11.5 → 2.6, length 7.0 → 5.7, roundness 4.6 → 3.0, plug-in)
  and the solver still taking +2.2 of length and +1.4 of roundness. The four
  orderings do not span what a regenerated ten-option set gives away, and the
  paper says so rather than quoting only the four-option case.

## Can the channel be learned in context?

- `icl_probe.py`: the target item is byte-identical across arms (the zero-shot
  prompt is an exact suffix of each few-shot one), demonstrations come only
  from other source groups, and both the demonstrations and the target have
  their stems withheld, so a demonstration can teach position and nothing else.
- The control arm draws demonstrations from the **repaired** file: same example
  items, same presentation order, same keyed letters, uniform ranks.
- Shot ladder 0/8/32/64 pre-specified; the trend is tested by exact enumeration
  of all 4! orderings, so the smallest attainable one-sided p is 1/24.
- Result: null, across six models (1.5–33B). Demonstrations do move solvers —
  −3.0 to +5.2 points at 64 shots — and move them as much when the key ranks
  are uniform, so the rank-isolating contrast is −1.3 to +1.3 at 64 shots and
  −2.7 to +1.9 over all 36 (model × shots × stem) cells. One cell is nominally
  significant at 5%, fewer than the 1.8 chance predicts, and it is negative:
  Qwen2.5-1.5B does 2.7 points *better* from uniform-rank demonstrations. The
  pre-specified scale trend on the rank contrast is ρ = +0.49, p = 0.36.

## The controlled probe

- 11 open-weight instruction-tuned models, 1.24–32.76B, five families
  (Llama, Qwen, Phi, Gemma, OLMo); bf16, one A100, deterministic read-out.
- 205 questions × 5 arms × 20 redraws = 20,500 conditions per model
  (11,740 distinct prompts after de-duplication), **225,500 decisions**.
- Prompt is BixBench's own `MCQ_PROMPT_TEMPLATE`, no-refusal variant,
  reproduced verbatim including its missing newline before `IMPORTANT`.
- Read-out agreement with greedy generation: 95.4–98.6% among compliant
  responses. Compliance itself ranges 32.5% (Llama-3.2-1B, which mostly writes
  the number rather than a letter) to 100%.
- Items the redraw cannot touch produce byte-identical prompts in every arm, so
  the pipeline contributes no arm difference of its own. Checked by test.
- **Positive controls.** Rank rules score exactly the key-rank share in the
  released arm (12.4 / 51.4 / 29.5 / 6.7%) — an end-to-end consistency check —
  and are *exactly* unmoved by the placebo. Under the repair: take-second
  51.4% → 26.8% (+24.6, p = 0.0007), take-interior 40.4% → 25.0% (+15.3,
  p = 0.0001), take-largest 6.7% → 23.1% (−16.4, p = 0.0001).
- **Negative control.** A flat-preference guesser: 25.7% → 26.0% (−0.3,
  p = 0.85).
- **Models.** Rank effects (placebo − repair) run −3.7 to +5.8 points.
  Qwen2.5-32B +5.8 [+1.3, +9.9] p = 0.020 and Qwen2.5-14B +3.7 [+0.4, +6.8]
  p = 0.041 are nominally positive; three other models are nominally negative.
  The pre-specified trend test over the panel gives Spearman ρ = +0.15 with
  permutation p = 0.67, so scale dependence is reported as a suggestion only.
- Tests run *within* each redraw (n = 205) rather than on the pooled 2,100
  rows. Under that correction the two-source model fits every run, median
  goodness-of-fit p 0.21–0.61; pooling had made it look rejected.

## Audit and repair

- Released file verdict **LEAKS**: numeric family cross-validated 51.4%
  [39.5, 63.4]; best achievable by any rank-only solver 51.4% (+26.4 points).
- After repair: all **105 of 105** numeric items rewritten (3 took a rank other
  than the one drawn, because a key of 1 among integer-styled options has no
  room below it), verdict **no leak detected**, cross-validated 24.8%
  [17.9, 32.2], best achievable 29.5% (**+4.5 points**), surface family 22.0%
  [16.1, 28.1].
- Every keyed answer in the repaired file is byte-identical to the original,
  checked by test.
- The surface family is searched in both directions, 18 members. An earlier
  one-directional version reported no leak where avoiding a reliably-wrong
  option is worth about eight points.

## What the audit says about a file with no leak in it

- `audit_calibration.py` generates files that cannot leak: an item's `k` options
  are drawn i.i.d. and rendered in i.i.d. styles, so the option set is
  exchangeable, and the key is then one of them chosen uniformly. Checked
  directly: over 20,000 synthetic four-option items the key's rank is within
  0.6 points of 25% under every one of the four orderings.
- **Four options** (60 files, 105 items, 46 clusters): the numeric and surface
  rule families fire on 0%; the coordinate bounds on 1.7–8.3%; the single
  `verdict` line on **16.7%**. Widest credit bound ever reported on a clean
  file: **+3.3 points** (roundness).
- **Ten options** (30 files, 1,263 items, 67 clusters): rule families 0%,
  coordinates 0–10%, `verdict` **13.3%**. Widest bound **+1.0 points**.
- Consequences for reading the tool, both stated in the paper: the headline is
  an OR over six tests, so read the coordinates; and at n = 1,263 a flag fires
  on departures worth under a point, so read the bound, not the flag.
- The rule families' one-sided level (`mcq_audit.FAMILY_ALPHA`) is set by this
  measurement, not by a correction formula — leave-one-cluster-out selection
  over 23 or 36 members is not a clean multiple comparison. At the naive 2.5%
  level they fired on 10% and 20% of clean ten-option files; at the shipped
  level, on none at either option count.

## Three searches for one repair

- `repair_search_ablation.py` crosses three searches with two target
  distributions over 8 seeds on BixBench v1.5, auditing each repaired file.
- `sample` (shipped) redraws whole option sets and keeps the closest of 48.
  `nearest` walks one distractor's rendering and roundness at a time and keeps
  the closest set it reaches. `guided` walks the same way but falls back to a
  plain redraw when it cannot reach the targets.
- Written-length targets met: **26%** (sample), **70%** (nearest), 66%
  (guided). Worst written-length credit bound across the 8 repaired files:
  **+2.9**, **+9.7**, **+12.2** points. Files flagged: **2/8**, **6/8**, **8/8**.
- The mechanism is recorded per cell as `hit_rate_by_target_rank`: the spread of
  the hit rate across the four target ranks is **19 points** for the sampler and
  **55** for `nearest`, and the widest realised rank share tracks it — 32.3%
  against 39.8% where flat is 25%. The rank achieved is the rank drawn filtered
  by what the item can reach; reachability is 97% of (item, rank) pairs at four
  options and **83%** at ten, the hardest rank reaching 64%.
- Calibrating the target draw against that reachability
  (`mcq_audit.fit_target_weights`, 400 items, three damped passes) takes
  `nearest` from 6/8 to 3/8 flagged and its worst bound from +9.7 to +8.3. It
  removes about half and no more, because reachability differs by item and not
  only by rank.
- The descent's cost is also visible without any channel theory: with the
  question *shown*, all four solvers score higher on the descent-repaired file
  than on the sampler's. Its distractors are more dismissible.
- Four solvers probed on the descent-repaired file with the question withheld
  collect at most +0.7 points of the +7.5 it leaves open on written length —
  the paper's own null holding for a channel we opened ourselves. What a bad
  search costs is the guarantee, not the current scores.

## Constructing the repair instead of searching for it

- `mcq_audit.construct_row` assigns renderings rather than searching over them.
  The invariant is checked by test at four and at ten options: every move the
  isolation tuner makes leaves the written-length and significant-digit ranks
  exactly where the assignment put them.
- **Frontier size.** Reachable (length, roundness) rank pairs per item,
  counting only those whose assignment survives every check (`legal_pairs`):
  **8.4 of 16** at four options, **29.5 of 100** at ten. Reachable rank
  *triples* including isolation: 23.2 at four options.
- **Attainable uniformity**, in closed form from the frontiers: BixBench allows
  exactly 25.0% on all three surface coordinates — a perfectly flat marginal is
  reachable there, so any residual is the repair's doing and not the file's.
- **BixBench, constructive repair**: verdict *no leak detected*; isolation −3.2,
  length −4.2, roundness −2.8 points; value +26.4 → +4.5; every keyed answer
  byte-identical. Ranks met: 100% length, 100% roundness, 92% isolation of the
  ranks it *assigned* (24/29/23% of the ranks a uniform draw would have asked
  for, which is the frontier's doing and not the repair's). Over 8 seeds:
  flagged on 1, worst-case bounds −1.3 / −3.0 / +0.9.
- **MMLU-Pro, constructive repair**: value cross-validated 9.1% [7.5, 11.3]
  against 10% chance, best rank-only solver +1.2 points (released: +11.5);
  isolation +(-0.4) → **−1.5**, written length +4.1 → **−0.9**, roundness +2.2 →
  **−1.1** — each inside the +1.0 a clean ten-option file produces. 6 of 1,263
  items end on a value rank other than the one drawn, against 210 for the
  value-only repair and 123 for the two-channel one, both of which search.
- The one-line verdict on that file still reads LEAKS, and correctly. It is the
  *numeric* items that close. 8,717 of the file's 9,980 ten-option items have
  options that are not numbers; a numeric repair does not touch them; measured
  over all of them lexical isolation is +2.9 points and written length +0.5,
  unmoved from the released file's +2.8 and +0.4. Closing a file's numeric items
  closes its numeric items.
- The repaired option sets keep their magnitude, notation and sign and vary
  their written precision, which is the lever: `0.05 → 0.077, 0.129, 0.132`;
  `35% → 27%, 24%, 18%`; `1.9E-05 → 1.90E-06, 1.840E-04, 4E-04`.

## Five orderings no repair targeted

- `held_out_orderings.py` measures five statistics that are not targets of the
  shipped repair and are not reported by the audit: the key's **leading
  significant digit**, its **decimal places**, its **digits added up**, plain
  **lexicographic order** of the rendered text, and **written length under the
  opposite tie-break**. They are measured only after the files are built, with
  the same estimator the audit uses.
- **Four options.** Released BixBench leaks on four of the five — lexicographic
  **+7.7**, length-with-later-ties **+7.1**, decimal places +5.3, leading digit
  +0.5 points; digit sum −2.9 does not. The constructed repair, which never saw
  any of them, **closes all five**: −3.0, −3.8, −6.1, −4.2, −5.4 points, against
  **+1.6** as the widest any of 20 clean four-option files reaches on these
  coordinates.
- **Ten options.** Released MMLU-Pro leaks on four — lexicographic +3.7,
  length-with-later-ties +2.5, leading digit +2.3, decimal places +1.4; digit
  sum −0.5 does not. The constructed repair **closes none of them**: +3.7, +1.7,
  +2.9, +1.7, against +0.4 on a clean ten-option file. Two are marginally wider
  after the repair than before.
- **Why they differ, measured.** Lexicographic order of a written number is
  monotone in its value only while the options share a digit width (`9.5`
  precedes `12.7` as text and follows it as a number). The two orders agree as
  whole permutations on **67.6%** of BixBench's numeric items and **68.6%** after
  its repair; on **48.8%** of MMLU-Pro's and **30.6%** after its repair, the
  repair lowering the agreement by rendering distractors at varied precision. A
  repair transfers to an unaudited ordering as far as that ordering is close to a
  monotone function of an audited one, and the coupling can be measured before
  the repair is run.
- **Two cautions about these, both measured.** Decimal places is degenerate
  on released files: every option carries the same decimal count on 87% of
  BixBench's items and 80% of MMLU-Pro's, so on those files it is largely the
  lexicographic ordering under another name, and it stops being degenerate on the
  repaired files — which makes its before/after comparison the weakest of the
  four. Lexicographic order is tied on every item by construction: it scores all
  options equally and lets `mcq_audit.feature_rank`'s text tie-break do the
  ordering, which is its definition rather than a defect. Digit sum — the one of
  the four that is neither degenerate nor magnitude-driven, 7.9 of 10 distinct
  scores per item — is closed on both files before any repair.

## A third benchmark for the placebo contrast

- MMLU's 661 four-option numeric items are the case where the repair should take
  nothing away: the released bounds are −2.9, −1.2 and +1.3 points on isolation,
  roundness and written length.
- All **11** models, six arms, 20 redraws, 13,220 decisions per model
  (`results/probe_mmlu_construct/`, the `MMLU` panel of
  `results/placebo_contrast.json`).
- Released no-data accuracy **21.6–24.7%**, at or below the 25% chance line for
  every solver. Placebo lift median **+2.5** points. Rank effect median **+0.8**,
  **positive for 8 of 11** — uniformising a rank that was already near-uniform
  makes models slightly better, and the movement is the rewriting.
- The three negative rank effects are the three largest models: Qwen2.5-32B
  −2.2, Qwen2.5-14B −0.7, Qwen2.5-7B −0.5. Reported as a suggestion, not a
  finding: ρ = −0.14 [−0.76, +0.49] over eleven models.
- Beside BixBench v1.5 (released file leaks +26.4 points, rank effect negative
  for 9 of 11) the pair is the contrast behaving as designed.

## A tie-break is an ordering

- Written length ties. The key is exactly as long as some distractor on
  **93.3%** of BixBench v1.5's numeric items, **87.3%** of MMLU's and **96.2%**
  of MMLU-Pro's, so a length score does not name one ordering but a family of
  them, one per tie-break.
- The two halves of this repository had picked different members.
  `mcq_audit.feature_rank` sorts by `(length, text)` ascending, so a tied
  distractor counts as *before* the key when its text sorts earlier;
  `text_artifacts.pick` takes the arg-max of `(-length, text)`, so among tied
  options it takes the *later* text. The key's rank under the two differs on
  **87.6%**, **79.3%** and **89.2%** of those files' numeric items.
- Held out and measured with the audit's estimator (`--option-counts 4`,
  20 clean replicates shaped like the file itself): on v1.5 the reversed
  tie-break is **+7.1** points released and **−3.8** after the shipped
  three-coordinate repair; on MMLU **+1.8** released and **+2.6** after it,
  against a clean floor of +0.6 — the repair does not close what it does not
  target; on ten-option MMLU-Pro **+2.5** released and **+1.7** repaired.
- The sharpest form of it: on the repaired MMLU file the key's written-length
  rank is flat at **25.0%** — as flat as the repair can make it — and
  `take shortest_option` still finds the key **36.6%** of the time, against
  23.8% on the released file. On v1.5 the same pair reads 24.8% and 25.7%.
- It is assignable, the key's rank under it still being a count over the
  distractors, and `mcq_audit.py --with-tie-break` adds it as a fourth count to
  the same dynamic program. On MMLU that takes the channel to **−2.9** and the
  rule's hit rate to 30.9%; on v1.5 the channel stays closed (−4.0).
- **At ten options too.** On MMLU-Pro the reversed tie-break runs **+2.5**
  released, **+1.7** after the shipped repair and **−1.6** after the
  four-coordinate one, while the four audited coordinates stay shut (−0.6, −0.8,
  −0.1) and the other four held-out orderings stay open (+3.3, +2.2, +1.8,
  −0.5). Adding the coordinate widens each item's reachable set from 29.6 rank
  pairs to **146.2 tuples**, and the fit reports the flattest reversed-tie
  marginal that file admits as **14.4%** against 10% flat. The channel closes
  anyway: a 13.8% top share over 1,263 items in 67 clusters is not separable
  from noise, which is the difference between the bound and the marginal.
  Shipped as `results/mcq_audit_mmlu_pro_tie.json`,
  `results/held_out_mmlu_pro_tie.json` and
  `results/repaired/mmlu_pro_repaired_tie.jsonl.gz`.
- **One thing had to change besides the program.** The isolation-tuning stage
  moves the key's isolation rank by re-rendering a distractor inside its own
  class and by letting two distractors exchange classes, both of which leave the
  assigned counts untouched — but a class was `(shorter, rounder)`, and a
  re-render inside such a class changes the *text*, which is what a tie-break
  reads. With the class widened to every assigned coordinate the repair meets
  the new target on **100%** of items rather than **85.6%**, and the widest rank
  it leaves the key at under that coordinate falls from 33.1% to 30.9% (25% is
  flat). Re-measured on `build/mmlu.jsonl` with the old grouping restored, not
  recalled from the run that found it.
- **What it costs.** Over eight seeds on MMLU the worst-case written-length
  bound falls +2.9 → **+0.2** and roundness +1.6 → +1.2, while isolation rises
  −0.1 → +1.1; on v1.5 the four-coordinate repair is flagged on 2 seeds of 8
  against the three-coordinate repair's 3.
- **What the models do.** Ten of the eleven panel models answered MMLU again
  under the four-coordinate repair, 13,220 decisions each
  (`results/probe_mmlu_tie/`; Qwen2.5-32B could not be loaded on the free GPU,
  see below). Every change is inside ±0.6 points with the question (median
  +0.1, largest −0.6 for gemma-3-4b) and inside ±0.9 with it withheld (median
  −0.4, largest −0.9 for Llama-3-Groq-8B), against a 5.5-point move in the
  bound.
- Qwen2.5-32B is missing from `results/probe_mmlu_tie/` because loading it a
  second time hit `NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_()` inside
  PyTorch's caching-allocator warm-up on this host, whose NVML is out of step
  with its driver. It is present in the eleven-model panel.
- Across the 18-file survey sweep the reversed tie-break leaks on exactly the
  two files the other held-out orderings leak on (MMLU +1.8, MMLU-Pro +2.5) and
  is closed on AQuA-RAT (−0.1), MedMCQA (−1.3) and MedQA-USMLE (−4.2).

## Two defects found by re-running the pipeline

- **The shipped results did not regenerate from the shipped code.** Re-running
  `mcq_audit.py --two-channel --search construct` with the documented arguments
  produced a BixBench repair differing from the released one on **91 of 205
  rows**, and the released audit JSON differed only in the lexical-isolation
  fields — in the *released* file's block as well as the repaired one, which
  located the cause immediately.
- **`isolation_scores` had stopped being a property of the pair.**
  `difflib.SequenceMatcher` is not symmetric: `1.90E-06` against `0.0004` scores
  **0.43** one way and **0.29** the other. An earlier version scored both
  directions independently, so two options had different similarities to each
  other and each option's "unlikeness to the rest" depended on where the loop
  reached it. The current version compares each pair once, in index order, and
  uses the value for both cells. The fix is right and the results files predated
  it; every number here that depends on lexical isolation was recomputed after
  it.
- **The guard.** `validate_artifact.py` now re-runs the audit on the released
  repaired file and compares its four coordinate bounds to the shipped JSON, so
  a results file that has drifted from the code fails the check rather than
  being quoted.

## What closing every audited coordinate does to a solver

- Ten options, constructive repair, question withheld, numeric items only
  (`results/probe_mmlupro_construct/`, 3 models x 8 redraws x 5 arms):
  - Qwen2.5-32B **13.4% -> 10.6%** (chance 10%) — the one solver that collected
    on the released file is returned to chance.
  - Qwen2.5-7B 10.4% -> 13.9% (**+3.4**), Llama-3.1-8B 9.3% -> 13.7% (**+4.3**).
- The same two under the other ten-option repairs: value-only +3.7 / +9.1,
  value-and-isolation +4.6 / +9.2, four-ordering sampler +3.8 / +8.1. The
  construction is the smallest residue of the four and is not zero.
- Reported rather than the audit alone, because the audit is the weaker
  instrument here: it calls every coordinate it measures closed while a solver
  still gains three to four points.
- **Where the residue is not.** `channel_attribution.py --held-out` attributes
  all nine coordinates on the repaired file — the four audited and the five held
  out. Audited: at most **+1.7 points available**, and every solver collects
  within **0.1 points of zero** on each. Held out: **+7.4 / +5.4 / +5.1 / +1.7 /
  +0.9 available**, and the same solvers collect between **−0.9 and +0.6**. A residue
  of three to four points is therefore not option geometry under any ordering
  measured here. The candidate that remains is the one already demonstrated
  stem-shown: a regenerated distractor is easier to dismiss than a curated one,
  which lifts a no-data score with no rank being informative.
- **The stem-withheld placebo measures exactly that**
  (`results/probe_mmlupro_placebo/`, `no_data_probe.py --arms placebo_stemless`).
  Values redrawn by the same generator with the key's rank *held*, gains over the
  released arm: Qwen2.5-32B **+1.4**, Qwen2.5-7B **+2.9**, Llama-3.1-8B **+5.9**,
  against the repair's −2.8, +3.5, +4.4.
- **Placebo minus repair isolates rank**, the contrast the four-option experiment
  uses, and after the construction it is **−4.2, +0.6, −1.5 points**: no solver
  collects on the repaired ten-option file. The apparent residue was the
  rewriting, not the ranks. It also explains the panel's shape — the solver that
  had been collecting loses 4.2 points of rank effect and the two that had not
  lose nothing, because they had nothing.
- **Both panels in full** (`placebo_contrast.py`; `results/probe_bix_placebo/`,
  `results/probe_mmlupro_panel/`): 11 solvers at four options, 10 at ten. Rank
  effect negative for **9 of 11** (median −1.1, largest −7.8) and **8 of 10**
  (median −0.7, largest −4.2). In both the largest belongs to the solver that
  collected most on the released file; the positives are at most +1.4 and sit on
  solvers the released file already left at chance.
- Released accuracy against rank effect: ρ = **−0.66** [−1.00, +0.03] over the
  eleven, **−0.54** [−0.96, +0.22] over the ten. The predicted direction in both
  and zero inside both intervals — reported as a suggestion, like the scale
  pattern.
- The placebo's own lift ranges **−6.0 to +5.9** points (medians +1.4 and +2.0)
  and reverses the sign of the rank effect for two of eleven solvers, which is
  why the repaired arm cannot be read alone.

## The same five orderings across the survey

- `held_out_orderings.py --survey-sweep` runs the same measurement over every
  vendored survey file's numeric subset, grouped by option count, skipping
  subsets under 60 items. Five qualify.
- Leaks on the five held-out orderings: **MMLU (k=4, 661 items) 3/5** — decimal
  places +2.7, lexicographic +2.5, reversed length tie-break +1.8; **MMLU-Pro
  (k=10, 1,263) 4/5**; **BixBench (k=4, 105) 4/5**; **MedMCQA (k=4, 454) 0/5**;
  **MedQA-USMLE (k=4, 77) 0/5**; **AQuA-RAT (k=5, 141) 0/5** (−0.9, −1.6, −1.6,
  −1.6, −0.1).
- AQuA-RAT is the one to read twice: it is the paper's counterexample because its
  *value* channel is uniform, and that closes every ordering measured here too. A
  distractor process that never looks at the key closes coordinates nobody
  thought to audit.
- MMLU leaking on three is worth recording because MMLU is the file the λ̂
  validation runs on.
- The coupling diagnostic separates the groups: lexicographic order is the value
  order on 65–70% of the items of each file that leaks on none, and on 49–53% of
  MMLU's and MMLU-Pro's. A file whose text order tracks its value order transmits
  whatever the value channel is, closure included.

## What a fifth coordinate costs, and what no repair can buy

- Lexicographic order is a count over the distractors, so `construct_row` can
  assign it one dimension wider (`mcq_audit.py --with-lexicographic`; **not** the
  shipped repair, and refused unless `--search construct`).
- On MMLU-Pro it narrows that channel **+3.7 → +2.0** points and leading digit
  +2.3 → +0.7 and closes neither, while the four audited coordinates stay shut
  (−1.7, −1.0, −0.8 against the four-coordinate repair's −1.5, −0.9, −1.1).
- **A correction.** This section previously reported the fifth coordinate
  reopening roundness at +1.5, read as the file over-determining the
  renderings. It was a defect in the repair, not a property of the file: the
  isolation-tuning stage re-renders a distractor inside its own
  `(shorter, rounder)` class, which the two counts cannot see and a text
  ordering can, so it undid the fifth assignment on 3.3% of items. With a class
  widened to carry every assigned coordinate the lexicographic target is met on
  **100%** of items rather than 96.7% and the trade goes away. The same defect,
  found again on the reversed tie-break, is the section below.
- The frontier explains it without running a repair. Adding the coordinate widens
  each item's reachable set from **29.5 rank pairs to 36.1 triples**, and IPF over
  those sets reports the flattest marginal they admit: exactly 10.0% on value,
  isolation and roundness, 15.7% on written length, **16.2% on lexicographic
  order**. A uniform lexicographic rank is not attainable on this file at any
  budget.
- BixBench admits 25.0% on every coordinate, and its five-coordinate repair leaves
  every held-out bound negative (−0.8, −3.1, −6.0, −2.5, −0.5) and the verdict at
  *no leak detected* — so the ten-option limit is a property of that file, not of the
  method.
- Shipped as `results/mcq_audit_mmlu_pro_lexicographic.json` and as the
  `repaired_lexicographic` block of `results/held_out_orderings.json`.

## A learned solver, and what it finds

- `learned_probe.py` fits a linear solver over per-option features read off the
  option set alone, leave-one-cluster-out, with rank encoded as one indicator per
  (ordering, rank). Three nested families, each read against its own clean-file
  control: the four **audited** orderings' ranks, **all eight** orderings' ranks,
  and **every feature** including ten that are not ranks.
- Sanity check on the encoding: with rank as a single continuous feature the
  learner scores 31% on released BixBench where the hand-designed rule scores
  51%, because a linear score cannot then express "the second-smallest". With
  indicators it scores 46.7%.
- MMLU-Pro, chance 10%, clean worst cases 11.2% / 11.6% / 11.4%:
  - released **25.7% / 24.9% / 24.8%**
  - constructed repair **10.6% / 20.2% / 28.3%**
  - also targeting lexicographic order **13.2% / 18.7% / 27.8%**
- BixBench, chance 25%, clean worst cases 36.2% / 31.4% / 31.4%: released
  46.7% / 40.0% / 43.8%; repaired **17.1% / 29.5% / 25.7%** — every family below
  its clean control.
- Reading: the repair's guarantee holds against the strongest learned solver in
  the class it is stated over (column one, −0.6 points at ten options and −19.0
  at four). Column two is the held-out orderings showing up as accuracy. Column
  three is outside the bound entirely — its heaviest weight is *relative
  magnitude* and it scores **higher on the repaired file than the released one**,
  because perturbing the key to make distractors makes the set's spread a
  function of the key. No probed model uses it.

## The learned solver on a third file

- `results/learned_probe_mmlu.json`: the same three nested families on MMLU's
  661 four-option numeric items, margins over each family's clean-file worst
  case in brackets (20 clean replicates of 663 items in 39 clusters).

  | file | audited ranks | all nine orderings | every feature |
  |---|---|---|---|
  | released | 28.4% [−0.6] | 28.8% [+0.5] | 27.9% [+0.2] |
  | constructed repair | 26.1% [−2.9] | 32.7% [+4.4] | 41.2% [+13.4] |
  | + the reversed tie-break | 25.0% [−3.9] | 31.7% [+3.3] | 41.8% [+14.0] |

- The guarantee's column holds on all three files and tightens as coordinates
  are added.
- **The learner names the defect.** Its two heaviest weights on the
  three-coordinate repair are `length, later ties rank 0` **+0.71** and
  `length, later ties rank 3` **−0.69** — it goes to the coordinate the shipped
  repair does not target without being told the coordinate exists. After
  `--with-tie-break` those weights leave the top four and plain text order takes
  their place.
- The third column is the scope statement, and it is larger here than at ten
  options: 41% on the repaired file against 28% on the released one, heaviest
  weight *relative magnitude* −2.68. Regenerating distractors by perturbing the
  key makes the set's spread a function of the key, and no repair of the key's
  rank touches that.
- The reversed tie-break is now the ninth ordering in `learned_probe.ORDERINGS`,
  so `results/learned_probe.json` and `results/learned_probe_mmlupro.json` were
  regenerated with it. On BixBench the four-option repair still closes every
  family (−19.0, −1.9, −5.7 against each clean worst case).

## Removing the geometry does not remove the leak

- `exchangeable_repair.py` tests the mechanism the third column came with. It
  draws `k` i.i.d. offsets and a slot `m` and sets
  `v_i = key · exp(σ(z_i − z_m))`, so `v_m = key` exactly and the key is a
  uniform draw from its own option set. The key's rank is then `#{i : z_i < z_m}`
  under **every** ordering, and every permutation-equivariant statistic —
  relative magnitude, distance from the rest, being the median — is exchangeable
  between key and distractors. No targets, no frontier, no calibration pass.
- **The rejection rule is the construction.** Renderings collapse, so draws are
  retried; whether a draw survives depends on which slot the key took, so
  retrying until *one* slot works gives a value-rank histogram of
  **32.4% / 21.6% / 23.4% / 22.5%** on MMLU — seven points of plug-in credit on
  the coordinate the construction closes by proof. A draw is accepted only when
  **all k slots** are legal, which makes acceptance a function of the offsets
  alone. The two differ in one line; `exchangeable_repair.py` runs both on every
  file and records the wrong one as `naive_retry_rank_share`, and
  `tests/test_exchangeable.py` holds the contrast as a regression.
- **Feasibility.** `['3', '1', '2', '4']` admits no exchangeable rewriting in its
  own written style: three distinct positive integers below 3 do not exist. 67 of
  MMLU's 661 numeric items (10.1%); the remaining **594** are the matched subset
  every number below is measured on, in all three files.
- **The bar is shape-dependent, so it was remeasured.** `audit_calibration.py
  --items 594 --clusters 37` (`results/audit_calibration_594.json`): the widest
  bound a clean file of that shape ever reports is +0.9 on lexical isolation,
  +1.0 on written length and +1.5 on significant digits, where the 105-item
  shape reaches +3.3. Every bound below is read against those.
- **Audit of the result** (`results/exchangeable_mmlu.json`): value −0.8 points
  (key rank 27.9 / 23.7 / 22.2 / 26.1%), written length +1.4, lexical isolation
  +3.0, significant digits +5.8 — the last three all clear of their clean
  reference. Three of the five held-out orderings close
  against a clean file of the same shape (`results/held_out_exchangeable.json`):
  decimal places +0.6 against +0.9, leading digit −2.5 against +1.0, string order
  +0.6 against +1.2. Two do not: digit sum +2.4 against +0.3 and the reversed
  length tie-break +2.7 against +2.3. Everything still open reads the key's own
  digits, which no redrawing of the distractors touches.
- **The test** (`results/learned_probe_exchangeable.json`, 594 items, 37
  clusters, chance 25%): every-feature margin over the clean worst case is
  **+11.8** on the rank-repaired file and **+10.9** on the exchangeable one.
  Removing the geometric explanation removed almost none of the leak. The
  heaviest weight moved from `relative magnitude` −2.53 to `has separator` +2.56
  and `trailing zeros` +1.15.

## What is conserved is authorship

- `key_identity.py` fits the same solver over twelve features that never refer to
  another option — integrality, a percent sign, an exponent, a thousands
  separator, a minus, a decimal point, distinct-character ratio, trailing zeros,
  significant digits, decimal places, digit sum, digit count. A margin it reports
  cannot be option geometry of any kind; `tests/test_key_identity.py` holds that
  as an identity (permuting the other options leaves a row unchanged).
- Margins over each family's clean worst case, same 594 items:

  | file | written form | roundness |
  |---|---|---|
  | released | 19.4% (−11.3) | 28.5% (−2.5) |
  | constructed repair | 30.8% (+0.2) | 31.3% (+0.3) |
  | exchangeable offsets | 33.2% (+2.5) | 38.4% (+7.4) |

- Reading: on the **released** file there is nothing to find, because MMLU's
  distractors are written by the same hand as its keys. On the rank-repaired file
  the per-option solver is level with its control, so that file's +11.8 really is
  a property of the set, just not of any rank. On the exchangeable file the same
  information has moved into the key's handwriting. **The two leaks are
  substitutes**; what is conserved is that a generated distractor is identifiable
  as generated.
- **The prediction, run on a file we picked for it.** MedMCQA's distractors are
  the wrong options of Indian medical entrance examinations, so they are written
  by people, and its domain is not MMLU's. It was added after the MMLU and
  MMLU-Pro numbers were fixed. Margins over each family's clean worst case, 368
  items in 20 subjects:

  | file | audited ranks | every feature | written form | roundness |
  |---|---|---|---|---|
  | released | 31.2% (+0.5) | 29.9% (+0.0) | 24.7% (−7.1) | 26.6% (−4.9) |
  | constructed repair | 33.4% (+2.7) | 41.3% (+11.4) | 27.4% (−4.3) | 32.3% (+0.8) |
  | exchangeable offsets | 44.3% (+13.6) | 48.6% (+18.8) | 48.4% (+16.6) | 52.7% (+21.2) |

  The released file leaks nothing in **any** of the four columns, the rank repair
  opens the non-rank channel by +11.4 where MMLU gives +11.8, and the
  exchangeable rewriting leaves +18.8 and moves it onto per-option features. Two
  released files whose distractors people wrote (MMLU, MedMCQA) give a no-data
  learner nothing anywhere; the one whose distractors a model wrote (MMLU-Pro)
  hands it +16.9 and +5.5 before we touch it. Reproduce with
  `python3 survey_to_jsonl.py --survey medmcqa --out build/medmcqa.jsonl` and the
  chain in `results/mcq_audit_medmcqa.json`,
  `results/key_identity_medmcqa.json`, `results/learned_probe_medmcqa.json`.
  The size of the exchangeable signature follows from what MedMCQA's keys are:
  86% whole numbers, median 25, 48% with a single significant digit.
- **A tie-break nearly hid it.** The roundness family's four features are
  integer-valued and tie at the top on two thirds of items. `numpy.argmax`
  resolves a tie to the first index and the key is slot 0 of every row, so the
  first version scored every tie correct: 63.3% on the released file, against
  33.6% on a synthetic file with nothing in it — which is what gave it away.
  `learned_probe.cross_validate` now breaks ties uniformly at random and reports
  `argmax_ties` with every result. It is **0** for all three families of the
  learned probe on every file, so no published number moved; the BixBench probe
  reproduces byte-identically under the fix.

## The prediction across every released file we have

- `key_identity.py` run on all ten released benchmarks on disk, each against
  a clean control matched to its own shape. Margin = released minus that
  family's clean worst case, best of the two per-option families.

  | benchmark | distractors | numeric n | clusters | control over chance | written form | roundness |
  |---|---|---|---|---|---|---|
  | MMLU-Pro | model-generated | 913 | 60 | +2 | −0.9 | **+5.5** |
  | MMLU | written by people | 594 | 37 | +6 | −11.3 | −2.5 |
  | MedMCQA | written by people | 368 | 20 | +7 | −7.1 | −4.9 |
  | LAB-Bench SeqQA | computed from the question | 160 | 4 | +9 | −13.1 | −7.5 |
  | AQuA-RAT | computed from the question | 141 | none | +9 | −15.6 | −2.8 |
  | BixBench v1.5 | computed from the question | 105 | 46 | +13 | **+1.9** | −14.3 |
  | MedQA-USMLE | written by people | 77 | none | +7 | −33.8 | −13.0 |
  | SciQ | unclear | 51 | none | +12 | −25.5 | −13.7 |
  | ARC-Challenge | written by people | 17 | none | +16 | −47.1 | −11.8 |
  | OpenBookQA | written by people | 7 | none | +46 | −71.4 | −57.1 |

- **Two files are above their control, not one.** BixBench v1.5 reads +1.9 on
  written form. It is reported, not hidden, and then set aside: its control
  already sits 13 points above chance at n=105, so +1.9 is inside the range the
  control itself wanders over, where MMLU-Pro's +5.5 sits on a file whose
  control is 2 points over chance. BixBench's distractors are script-written
  perturbations of the true value, so a null was never predicted there either.
  `validate_artifact.py` asserts the pair as a set equality and separately
  asserts that only MMLU-Pro clears its control by more than its control clears
  chance, so a third positive, or a change in which one has headroom, fails the
  build rather than passing quietly.
- **The two science-agent benchmarks are now in the test.** BixBench v1.5 and
  LAB-Bench SeqQA are read by `bio_to_jsonl.py` through the same parsers
  `channel_survey.py` uses, so the items are the ones the survey figure counts.
- **What this does not establish.** The five files below MedMCQA carry no
  cluster field, so `cross_validate` degrades to leave-one-item-out and the
  control balloons: 51.9% at n=77 and 85.7% at n=7, against 25% chance. Those
  rows are consistent with the prediction and are not evidence for it. The test
  is powered on the top three, which are the three in the paper's table.
- **The positive half of the prediction still rests on one released file with
  power behind it.** No other benchmark on disk was built with model-generated
  distractors, so there is nothing further to test it with observationally;
  the controlled arm below is what carries it instead. The replication is our own three
  exchangeable rewritings, below.
- **Regeneration alone is not the trigger.** The rank repair preserves each
  replaced option's written style and lands at its control on all three files
  (+0.2/+0.3 MMLU, −4.3/+0.8 MedMCQA, −1.1/+0.5 MMLU-Pro). The exchangeable
  construction renders distractors at generic magnitudes in the key's style and
  clears its control on roundness on all three (+7.4, +21.2, +9.2). So what is
  detectable is a distractor generated *without imitating the key*, which is
  also why AQuA-RAT's computed distractors leave nothing.
- Reproduce: `python3 survey_to_jsonl.py --survey <name> --out build/<name>.jsonl`
  then `python3 key_identity.py --jsonl build/<name>.jsonl --label released
  --n-options <k> --cluster-field cluster --output results/key_identity_<name>.json`.

## The controlled arm: who wrote the distractor, held as the only difference

- `authorship_causal.py` builds two ten-option files from the same MMLU items.
  Both keep the key and its three human-written distractors and add six more,
  which is how MMLU-Pro was made from MMLU. **generated**: the six are written
  by Qwen2.5-14B-Instruct, shown the human distractors and asked to match them.
  **borrowed**: the six are human-written options lifted from other items in the
  same subject. 633 items in 28 subjects survive in both arms.
- The model is *shown the human style and told to match it*, which is the hardest
  version of the test rather than the easiest.
- `authorship_contrast.py` does the paired analysis, which is the one the design
  was built for. Cluster bootstrap over subjects, 4000 reps:

  | family | generated | borrowed | difference | 95% CI | one-sided p | family-wise p |
  |---|---|---|---|---|---|---|
  | written form | 10.7% | 8.2% | +2.5 | [−0.2, +6.0] | 0.045 | 0.18 |
  | roundness | 7.7% | 6.2% | +1.6 | [−0.3, +2.8] | 0.047 | 0.19 |

- **The causal claim does not hold.** Both families move the way the
  observational half predicts and neither reaches significance once the two are
  corrected for. The experiment bounds the effect at a few points; it does not
  establish model authorship as a cause.
- **An earlier version of this table said otherwise, and was wrong.** It read
  +3.9 points at one-sided p = 0.011 on roundness. The generator's reply parser
  stripped a character class (`lstrip("-*0123456789.) ")`) that also ate a
  value's own leading digits, so "1,2,3" was recorded as ",2,3": a leading
  separator that appears only on distractors and so makes the key easier to
  find. It mangled 21% of replies and pushed the result in the direction of the
  paper's own claim. Fixed, shared between both generators, and held as a
  regression test in `tests/test_frontier.py`.
- **The bound stands either way.** Neither arm clears its own clean control
  (10.7% and 7.7% against 12.6% and 11.9%), so a benchmark built this way would
  pass the audit in the paper. A generator told to imitate the human options
  leaves no signature this instrument can resolve, which is the one actionable
  build instruction the paper offers, and MMLU-Pro's +5.5 over control is larger
  than imitation-instructed generation produces here.
- **What this cost the paper.** The strong claim "a generated distractor is
  detectable" is false as stated, and the paper says so. The surviving claim is
  narrower: generation that does not imitate the key is detectable. The repair
  frontier tests that claim directly and confirms it.
- Reproduce: `python3 authorship_causal.py --jsonl build/mmlu.jsonl --model
  Qwen/Qwen2.5-14B-Instruct --device cuda:2` then `python3
  authorship_contrast.py`.

## The repair frontier: what each operator closes, and what it costs

- "The channel cannot be repaired away" rests on two repairs, and **both
  condition on the key**, so it says nothing about operators that do not. An
  operator that never reads the key cannot make the option set a function of it,
  so the claim has to be stated over a class and tested at both edges. One
  operator the class must include is AQuA-RAT's construction, where the
  distractors are the values a *wrong solution step* produces. That row is in the
  table.
- `repair_frontier.py` and `keyblind_operators.py` build five processes on the
  same MMLU items with the same key, differing only in what the distractor
  writer may see, and `frontier_align.py` cuts every arm (including the two
  existing repairs) to the **464 items in 20 subjects** all seven could produce.
- Two axes, because the interesting quantity is a trade-off. `key_identity.py`
  and `learned_probe.py` give what a no-data solver reads; `frontier_validity.py`
  gives what three solvers score **when given the question**, paired by item, so
  a repair that made the distractors trivial to reject is visible as such.

  | operator | what the writer sees | set | one option | difficulty |
  |---|---|---|---|---|
  | released | — | −4.5 [−11.7, −0.6] | −5.8 [−10.1, −1.3] | — |
  | key marginal | other keys in the subject | −2.6 [−6.5, +4.7] | −6.7 [−13.8, −3.4] | +16.1 to +20.4 |
  | key marginal, near | other keys, within 10× | +1.7 [−5.4, +5.5] | −5.6 [−8.2, −1.4] | +11.1 to +16.2 |
  | wrong step | the question | **+8.6 [+3.1, +11.4]** | −2.2 [−6.6, +2.7] | **−1.5 to +2.3** |
  | rank uniform | the key | +9.5 [+6.9, +12.4] | −0.4 [−7.1, +4.5] | +6.4 to +9.9 |
  | exchangeable | the key | +9.5 [+4.9, +17.7] | **+6.9 [+1.7, +11.0]** | +3.6 to +9.9 |
  | imitation | question, key, human options | +4.3 [−1.3, +8.5] | −4.3 [−9.4, −1.3] | −6.9 to −1.1 |

- **The key/no-key framing was wrong, and the new row is what showed it.**
  `wrong step` never sees the key — the writer gets the question and nothing
  else — and it clears its control by +8.6, most of it on the rank coordinate
  (+8.4). A writer computing wrong answers to a question computes them *around*
  the right one, so conditioning on the question is conditioning on the key.
  What a repair cannot escape is knowing the item, which is the same thing as
  keeping the item hard.
- **`wrong step` is the only row whose difficulty interval contains zero at all
  three solvers** (−1.5 to +2.3). It leaves the benchmark exactly as hard as it
  found it and opens the widest rank channel in the table. That pair of facts is
  the trade-off with nothing rhetorical in it.
- **The collision filter is not where its leak comes from.** The assembler drops
  any value equal to the key, which happened on 114 of 635 items. On the 420
  items where nothing was filtered the arm still scores +7.1 against a released
  −4.8 (`results/learned_probe_frontier_nofilter.json`). An earlier version of
  this control was vacuous — the strict arm and the filtered arm coincide on
  their intersection by construction — and it was rebuilt on the strict arm's
  own item set.
- **The operators that close the set channel destroy the benchmark.** Borrowing
  other items' keys hands a solver given the question +11.1 to +20.4 points.
- **Only the style-replacing operator opens the per-option channel** on this
  table, which is the authorship mechanism confirmed on a held item set after
  the controlled arm failed to confirm it. That claim is reader-dependent and
  the paper says so: a per-option reader over raw character n-grams puts the
  rank repair at +7.1 [+3.4, +11.8] on MMLU's 594 items, so style preservation
  buys a channel the twelve designed features cannot see, not a closed one.
- **One generator is not an operator class.** On the 323 items all of them
  produced, `imitation` reads +5.9 (Llama-3.1-8B) and +2.5 (Phi-3.5-mini)
  against −0.9 for the Qwen-14B arm, and `wrong step` repeats at +9.0 under
  Llama-3.1-8B. A fourth generator, Qwen2.5-7B, reproduced at least two of the
  human distractors verbatim on half the items, which is imitation collapsing
  into copying and is why that arm is reported and not used.
- A bug worth recording: the borrowed arms first drew from the *distinct* values
  of a subject's keys rather than from the empirical multiset. Deduplicating
  reweights a subject towards its rare written forms (on MMLU it lifts the share
  carrying a percent sign from 15.4% to 18.8%), and a per-option solver reads
  exactly that reweighting, which showed up as several points of apparent leak.
  `draw_distinct` now samples the multiset, and `tests/test_frontier.py` holds
  the property.
- Reproduce: `python3 repair_frontier.py --jsonl build/mmlu_questions.jsonl`,
  `python3 keyblind_operators.py --operator wrong_step ...`, then the two
  repairs on `build/mmlu_frontier_released_matched.jsonl`, then
  `python3 frontier_align.py`, then `key_identity.py`, `learned_probe.py` and
  `frontier_validity.py` over the seven `*_aligned.jsonl` files.

## A reader with no feature this repository designed

- The objection one level up from "you designed the wrong statistics" is "you
  designed the wrong *features*". `learned_probe_nonlinear.py` answers it by
  running the identical protocol — leave-one-cluster-out, the same clean
  synthetic controls, the margin read against the control's worst case — over
  readers that use none of our features:
  hashed character 1–4-grams of the option's own text, optionally concatenated
  with the mean of the *other* options' vectors and its difference from them
  (permutation-invariant in the others, so it is a set reader), scored by a
  one-hidden-layer network.
- **Set reader, MMLU (594 items):** released **−3.5 [−6.2, +2.4]**, rank uniform
  **+6.7 [+2.5, +13.9]**, exchangeable **+8.9 [+5.6, +17.3]**. At ten options
  (MMLU-Pro, 913 items): released **+6.8 [+3.4, +9.5]**, rank uniform **+14.2**,
  exchangeable **+9.1**. The pattern is the designed reader's and the intervals
  exclude zero, so "the repaired file still leaks" is a fact about the file.
- **Per-option reader**, which cannot be geometry of any kind: on MMLU-Pro,
  whose extra options a model wrote, released reads **+4.9 [+2.1, +7.4]**; on
  MMLU, whose options people wrote, **+2.2 [−1.3, +6.7]**, which does not clear.
  That replicates the paper's observational authorship contrast with no designed
  feature anywhere in the reader.
- **Where it says nothing:** on the 464-item frontier set the same reader is at
  its control on all seven arms, including the released file. Twenty clusters
  and 464 items are not enough for a 6,144-feature reader, and that is reported
  as a limit of the instrument rather than as a null about the files — Γ is a
  maximum over readers, so a weaker reader finding less is not evidence of less.

## A rank guarantee does not survive being subsetted

- Two numbers for the same quantity disagree: on whole repaired files the
  audited-rank learner scores 2.9 points *below* its clean control at four
  options, and on the 594 items every row of Table 2 shares it scores 1.9
  *above*. It is not an error, and the explanation is worth more than the
  reconciliation.
- The repair draws each item's target rank independently, but the rank an item
  *reaches* is the rank drawn filtered by what its own options can render.
  Exchangeable rewritability is a condition on the same thing — putting the key
  at the top needs k−1 distinct renderable values below it — so cutting to the
  feasible items removes exactly the items whose reachable ranks were
  compensating.
- `subset_guarantee.py` measures it directly, as plug-in credit max_j p_j − 1/k:

  | file | whole | feasible subset |
  |---|---|---|
  | MMLU, rank uniform | +3.0 | +5.8 |
  | MedMCQA, rank uniform | +1.9 | +6.2 |
  | MMLU-Pro, rank uniform | +1.2 | +3.7 |

  MedMCQA's subset is back to the +6.2 its *released* file gives, so on that cut
  the repair bought nothing.
- The practical statement: a repaired benchmark stops being repaired the moment
  anyone filters it, which is what a maintainer does every time they drop the
  items they could not validate.

## How many items a no-data baseline needs

- `baseline_power.py` simulates from the model at other sizes and reports the
  power of the boundary-corrected test of λ = 0, 500 replicates per cell.
- At BixBench's own key-rank distribution and its own 105 four-option items, the
  test has power **0.18** against a true λ = 0.05 and **0.51** against λ = 0.10.
  Reaching 80% needs **500** items at λ = 0.10 and **2,000** at λ = 0.05.
- The key's rank distribution costs power as well as handing out credit: under a
  uniform p the count needed at λ = 0.10 is 250, under BixBench's skewed p it is
  500, because the geometry term absorbs variance the recall term needs.
  Uniformising the key's rank halves the benchmark a maintainer has to build.
- `published_baselines.py` applies that to BixBench's own two published
  zero-shot runs (v1.0, the only release recording the presented order): 27.7%
  and 30.2% against 25% chance, λ̂ = **0.028** and **0.069**, boundary-corrected
  p = **0.27** and **0.07**, goodness-of-fit p = 0.54 and 0.41. Neither
  published baseline's recall term is distinguishable from zero.

## The dates the releases do not publish, and the test they make possible

- "Not one of twelve released files carries a date" is true, and weaker than it
  sounds: five of the twelve publish a resolvable DOI for every item or for two
  thirds of them, and a DOI resolves to a date in one request. So the dates can
  be recovered, and this section recovers them.
- `temporal_recovery.py` resolves every identifier in the census against
  Crossref and the bioRxiv/medRxiv APIs, caching to `results/source_dates.json`.
  It recovers a date for **740 of 2,558 tasks**: 198/199 LitQA2, 243/244
  TableQA, 181/181 FigQA, 82/82 SuppQA, 36/205 BixBench — whose identifiers
  mostly name a dataset rather than a paper. No credential and no identifying
  header is sent.
- `temporal_test.py` then runs the test itself on LitQA2, whose questions are
  answerable only from one paper each. Four open-weight solvers, two arms
  (question given, question withheld), six letter orders, each item scored
  against its own chance because the option count varies.

  | solver | arm | accuracy | margin | ρ(date, margin) | pre − post cutoff |
  |---|---|---|---|---|---|
  | Llama-3.1-8B | question | 36.0% | +11.6 | −0.077 (p = 0.26) | **+6.8 [−2.5, +16.5]** |
  | Llama-3.1-8B | withheld | 23.8% | −0.6 | −0.148 (p = 0.05) | +4.6 [−1.7, +11.0] |
  | Llama-3.2-3B | question | 31.2% | +6.8 | −0.009 | +3.5 [−4.4, +11.5] |
  | Llama-3.2-3B | withheld | 25.2% | +0.7 | −0.009 | −0.9 [−7.6, +5.8] |
  | Qwen2.5-7B | question | 32.7% | +8.2 | +0.019 | — |
  | Qwen2.5-14B | question | 38.2% | +13.8 | −0.057 | — |

- The split is reported only for the models whose vendor states a cutoff
  (Llama-3.1 and 3.2, December 2023); everything else is the cutoff-free trend,
  which needs no claim about any model's corpus. The one nominal rejection
  (ρ = −0.148, p = 0.050) is one of eight tests and is in a *withheld* arm
  sitting at chance overall.
- **The null is the useful result.** Where the principled test can be run it is
  run, and at 198 items it cannot resolve an effect smaller than about fifteen
  points; the proxy agrees on the same items, and the agreement is worth nothing
  because neither instrument could have said otherwise.

## Proposition 1 needs uniform membership, not an exchangeable tuple

- "Γ = 0 iff the k-tuple (key, distractor₁…) is exchangeable" is false in the
  necessity direction. The counterexample: draw a k-subset from any law, let
  the key be uniform on it, and list the distractors in increasing order.
  Γ = 0 and the tuple is not exchangeable, because the non-exchangeability lives
  entirely in the ordering of the distractors among themselves, which Γ never
  observes.
- `writer_theory.py --exact` enumerates that law at k = 3 over ten values and
  returns Γ = 1.1e-15 with the tuple not exchangeable. Held as a test.
- The statement that holds is Γ = 0 **iff the key is uniform inside its own option
  set**, and both directions are one line from max_v Pr(Y=v|V) ≥ 1/k pointwise.
  Full exchangeability is sufficient, not necessary. The practical consequence is
  not a weakening: a maintainer does not have to randomise the presentation
  order, which BixBench already does and which delivers nothing here. Membership
  is what has to be uniform.

## Two writer classes characterised, and the price of the one that closes

- Model the key marginal as κ and the writer as a kernel ν(·|item) emitting the
  k−1 distractors. Then Pr(Y = v | V) ∝ κ(v)·∏ ν(u|v) over u ∈ V∖{v}, and Γ = 0
  asks that product to be flat on V.
- **(i) A writer that reads nothing about the item** has ν(·|v) ≡ ν, so the
  v-dependence is κ(v)/ν(v) and Γ = 0 forces ν = κ. That is the KEY MARGINAL
  operator, and it is the *only* member of its class that closes the channel —
  not one operator among seven. Enumerated at k = 2, 3, 4: ν = κ gives Γ within
  1e−16 of zero, a uniform ν gives +0.15 to +0.17, a reversed one +0.31 to +0.37.
- **(ii) At k = 2** the condition is κ(v)ν(u|v) = κ(u)ν(v|u), detailed balance.
  A κ-reversible kernel on three values gives Γ = 0 exactly; a non-reversible one
  gives +0.40.
- **The price.** Under (i) the accuracy of any solver that scores options
  separately given the question is E_Q[F_Q(s(key,Q))^(k−1)] ≥ 1 − (k−1)ρ, where ρ
  is the chance a key borrowed from another item of the pool outscores this item's
  key. Neither expression mentions the writer, so the difficulty column can be
  quoted before the arm is built. `writer_theory.py --empirical` measures ρ on
  MMLU's own key pool with 48 pool draws per item and a cluster bootstrap over the
  20 subjects:

  | solver | ρ | predicted accuracy | measured on the built arm |
  |---|---|---|---|
  | Qwen2.5-7B | 0.215 | 60.8 [46.3, 69.6] | 65.9 |
  | Llama-3.1-8B | 0.213 | 61.1 [43.5, 72.1] | 56.8 |
  | Qwen2.5-14B | 0.176 | 66.5 [51.4, 75.0] | 75.2 |

  Two of the three measurements are inside the prediction's interval; the third
  is 0.2 points past its upper end. The intervals are ten to fifteen points wide,
  so this is a calibration check and not a tight prediction — but it is computed
  without building the arm and it puts the column in the right place.

## Proposition 2(i) holds without the independence assumption

Modelling the writer as a kernel emitting the `k-1` distractors **i.i.d.** gives
the product form `Pr(Y=v|V) ∝ κ(v)·∏ν(u|v)`. No real generator satisfies it: a
model asked for three distractors writes a tuple whose entries constrain each
other, so under that model the "only" in the claim would not be earned at the
generality it is stated in.

It is earned without the assumption. Let an item-blind writer emit the
distractor set from an **arbitrary** joint law `μ` on `(k-1)`-subsets of the
value pool, let the key be drawn from `κ` independently, and condition on the
`k` options coming out distinct — which every assembler must do. Then

    Pr(Y = a | V = S)  ∝  κ(a) · μ(S \ {a}),

so Γ = 0 asks that `κ(a)·μ(S\{a})` be constant over the members of every
presented set `S`. **That is linear in μ**, so the item-blind writers closing the
channel form a subspace, and `writer_theory.py` computes it as a null space.

- **It is a line, at every shape tried.** Eight `(|X|, k)` shapes on skewed κ —
  (4,3), (5,3), (6,3), (5,4), (6,4), (7,4), (6,5), (8,3), with up to 35 free
  entries in μ — all return `solution_space_dimension = 1`.
- **The line is the key marginal.** `μ(T) ∝ ∏_{x∈T} κ(x)` sits in the null space
  with residual below 1e-17 in every case.

The proof is three lines. Put `f(T) = μ(T)/∏_{x∈T}κ(x)`. For a `(k-2)`-set `R`
and `a ≠ b` outside it, the condition on `S = R∪{a,b}` reads
`κ(a)μ(R∪{b}) = κ(b)μ(R∪{a})`, i.e. `f(R∪{a}) = f(R∪{b})`. Any two `(k-1)`-subsets
are joined by such single swaps once the pool has `k` or more values, so `f` is
constant.

**Two things follow.** The abstract's "only" is now a claim about all item-blind
writers, not only independent ones. And the distinctness conditioning is part of
the theorem rather than an implementation detail — which is exactly why the next
section's trap is a trap: rejecting a collision *against the key* is a different
conditioning and leaves the family.

## The assembler's distinctness loop is itself a dependence on the key

- An assembler must return k distinct options. Conditioning the whole k-tuple on
  distinctness treats every slot alike. Drawing the key and then *redrawing each
  distractor while it collides with the key* — the loop most assemblers write,
  ours included — conditions each draw on the key and removes the key's own mass
  from the pool.
- Γ in points, for an item-independent writer drawing from the key marginal at
  k = 4, by how distinctness is enforced:

  | key marginal | conditioned jointly | redrawn against the key |
  |---|---|---|
  | uniform | -0.0 | -0.0 |
  | mild skew | -0.0 | +4.8 |
  | heavy skew | +0.0 | +20.6 |

- On a uniform marginal it costs nothing, which is why it is easy to miss. On a
  skewed one it is worth up to +20.6 points. Held as a test.

## What the goodness-of-fit failure costs the geometry term

- A per-draw test rejects the shared-recall model on 12 of 40 MMLU draws,
  concentrated on the two largest λ̂. The natural suspect is that recall depends
  on the key's rank, so `rank_dependent_fit.py` frees it: λ_j per key rank, the
  same b, the same Pearson test with the parameter count adjusted.
- That does not rescue the fit. The likelihood ratio prefers the larger model on
  4 of the 40 draws and the larger model's own test rejects on
  16.
- It does bound the consequence. Across the twelve rejecting draws the geometry
  term moves by at most 0.19 points, never changes sign, and its range over all
  forty goes from [-0.84, +1.21] to [-0.88, +1.19]. The misfit is real and it is not
  in the term this paper reads.
- On BixBench's 105-item draws the same refit moves it by up to 2.3 points with
  1 sign change, which is the sample size speaking.
- A test holds the sensitivity: under a synthetic recall that swings from 0.05 to
  0.60 with the key's rank, the refit moves the geometry term by more than two
  points, so the 0.19 on the real data is a measurement and not a floor.

## The frontier without its two dominant subjects, and without the confound

- Two of the twenty subjects hold 282 of the 464 items. Dropping both leaves 182
  items in 18 subjects, and the set column orders the same way and further apart:
  released −7.7, IMITATION −6.6, KEY MARGINAL −2.7, KEY MARGINAL NEAR +3.3,
  RANK UNIFORM +4.9, WRONG STEP +10.4, EXCHANGEABLE +15.4.
- WRONG STEP's writer never sees the key, but on 81 of the 464 items it produced
  the key anyway and the assembler dropped that value. If the leak were carried by
  those items, "the question determines the key" would be confounded with "this
  writer knew the answer". It is not: on the 383 items where the writer never
  produced the key the arm reads +12.3 against the released file's −4.7 on the
  same items, and on the 81 where it did, +4.9 against +2.5. The leak is carried
  by the items where the writer did *not* know the answer.

## The no-data baseline run as an agent, and the one arm that collects the channel

- Every no-data baseline in this literature, and every probe elsewhere in this
  repository, reads a single letter token. A benchmark for agents is not scored
  that way. `agentic_probe.py` runs the same baseline as an agent: a system
  prompt saying the question is unavailable and that it should reason from the
  options, a sandboxed Python interpreter it may call for up to three turns, a
  `FINAL:` line, and a matched clean file of the same shape scored identically.
- **The scaffold works.** Given the question, Qwen2.5-7B scores 32.1% on
  BixBench's 105 four-option numeric items — where BixBench's own published
  zero-shot runs sit (27.7% and 30.2%).
- **Two of three agents read nothing**, as the single-token probes do:
  Qwen2.5-7B −1.3 [−8.2,+5.7] against the clean file and Llama-3.1-8B +1.9
  [−5.8,+9.9] on BixBench, −0.7 [−3.7,+2.7] and +1.5 [−1.0,+4.4] on MMLU-Pro.
- **Qwen2.5-14B collects the channel on both files, and MMLU-Pro is the arm
  that carries the claim.** 913 items in 60 clusters, 1,826 rollouts, 0.3%
  unparsed, 20.5% tool use: **16.2% against 10% chance**, +6.21 [+4.46,+8.23]
  over chance, against a clean control at 9.96% (−0.04 [−1.45,+1.31], which is
  what a clean control should read). Margin over clean **+6.25 [+4.04,+8.69]**.
  The second-smallest rate is 2.9% against the clean file's 5.9%, so at ten
  options it is not the BixBench rule the agent is running; a *fitted*
  single-token solver takes 29.1% on this same file, so the agent collects
  roughly a fifth of what the geometry makes available — and the letter-reading
  probes of the same family collect none of it.
- **Qwen2.5-14B reads +11.1 [+3.0,+19.6] on BixBench.** It scores 33.3% against 25% chance
  (+8.3 [+2.2,+14.9] over chance alone, which excludes zero without using the
  control at all) where the matched clean file gives 22.2%. 315 rollouts over
  105 items in 46 capsules, 0% unparsed, a Python call on 19.7% of rollouts,
  1.40 mean turns. The same run repeated on a different GPU reproduces it to the
  digit: 33.33/22.22, +11.11 [+3.0,+19.6]. Greedy decoding with fixed batching
  makes the arm deterministic, which is worth stating because it means the
  interval is sampling error over capsules and nothing else.
- **The margin is not the calculator.** Splitting the rollouts on whether the
  agent ran code, the margin over the clean file is +12.3 points among the 253
  that did not and +5.5 among the 62 that did.
- **It is eq (1) fitted to an agent.** `agentic_mechanism.py` reads the rank of
  the key and the rank of the agent's pick among each item's sorted values. The
  agent's pick distribution *b* is (8.9, 29.8, 39.4, 21.9)% on the released file
  and (13.0, 26.0, 38.1, 22.9)% on the clean one — the same taste for the middle
  whichever file it reads, which is what makes it a *preference* rather than a
  response to the file. Against the released file's key ranks *p* = (12.4, 51.4,
  29.5, 6.7)% that preference gives ⟨p,b⟩ − 1/4 = **+4.5 [+1.7, +7.4]** points;
  against the clean file's near-uniform ranks it gives **+0.1 [−1.8, +1.8]**.
  With no question and no data the recall term of eq (1) is zero, so that is the
  whole predicted margin: about half of the +8.3 the agent takes over chance is
  the rank channel and the rest is not — consistent with the per-option channel
  the rest of the paper measures, and reported as a residual rather than
  explained away.
- **It is the wording, and the wording is the rank rule in words.**
  `agentic_mechanism.py` splits the rollouts on whether the agent's own
  reasoning reaches for the *middle* of the option set (median, central, the
  middle value). On the released file it does so on 23.2% of rollouts and those
  score **45.2% [34.4,58.1]** against 29.8% for the rest. On the clean file the
  same wording is available, is reached for on only 6.7% of rollouts, and buys
  nothing: 23.8% against 22.1%. BixBench's distractors are perturbations that
  bracket the true value, so "the plausible middle" and "the second smallest"
  name the same option — and the agent never writes the second: across all three
  models the second-smallest option is taken on 20.6 to 29.8% of items where
  always taking it would score 51.4%.
- This is a post-hoc split on text the model wrote, not a pre-specified arm. It
  says which rollouts carry the margin, not that the wording caused it. What
  makes it readable at all is that the clean file admits the same split and
  gives nothing.
- **One model size up, the mechanism replicates and the surplus does not.**
  Qwen2.5-32B on BixBench, sharded over two cards because 62 GiB of bf16 weights
  do not fit beside another tenant's job: 29.84% against 24.13% on its matched
  clean file, **+5.71 [−2.05, +13.34]** — a positive point estimate whose
  interval covers zero, against the 14B's +11.1 [+3.0, +19.6]. Its positive
  control is the strongest of the four (37.8% given the question), so this is
  not a weaker model.

  What replicates exactly is everything below the end-to-end margin. Its pick
  distribution is (16.5, 32.1, 35.9, 15.6)% — the same concentration on the
  middle two ranks as the 14B's (8.9, 29.8, 39.4, 21.9)% — for a rank term of
  **+5.2 [+2.7, +7.6]** against **−0.2 [−1.6, +1.2]** on its clean file. Both
  models therefore collect the rank channel, and both intervals say so more
  tightly than either end-to-end margin does.

  What does not replicate is the surplus. The 14B scores +8.3 over chance where
  its rank term predicts +4.5; the 32B scores +4.8 where its term predicts +5.2.
  The 14B takes something beyond the rank channel and the 32B takes nothing
  beyond it, which is why only the 14B's margin clears. The wording tracks that
  split too: the 32B reaches for the middle half again as often (34.9% against
  23.2% of rollouts) and is paid a third as much (35.5% against 26.8%, where
  the 14B got 45.2% against 29.8%).

  **On BixBench this is a null for the scaling reading and a positive for the
  mechanism one.** Four agents, one margin clears, and the two largest cannot be
  told apart at 105 items. We report that ordering as non-monotone rather than
  implying a trend.
- **On the file that can resolve it, both large agents clear.** Qwen2.5-32B on
  MMLU-Pro's 913 ten-option items, 1,826 rollouts over 60 clusters, 0.27%
  unparsed, 15.5% tool use: **17.47% against 10% chance**, +7.47 [+5.30,+10.65]
  over chance, **+5.48 [+2.69, +9.23]** over its matched clean control. Beside
  the 14B's +6.25 [+4.04, +8.69] that makes it two of four agents, the two
  largest, with overlapping intervals and the two smallest at nothing.
- **And at ten options the rank channel is nearly the whole margin.** MMLU-Pro's
  key rank *p* is (2.9, 4.7, 10.0, 19.0, **24.5**, 18.5, 11.1, 5.2, 2.9, 1.3)%
  — peaked in the middle — and the 32B's pick *b* is (1.8, 2.1, 4.4, 9.7,
  **26.6, 31.0**, 10.0, 6.4, 4.1, 4.0)%, so ⟨p,b⟩ − 1/10 = **+6.30 [+5.00,
  +8.22]** against the +7.52 it scores over chance. At k = 4 the rank term was
  about half the margin; at k = 10 it is 84% of it.
- **The clean control makes the paper's own caveat about itself.** That file's
  key rank is uniform to within a point, so its rank term is −0.04 [−0.46,
  +0.38] — exactly zero, as a Γ = 0 construction should be *for a rank reader*.
  And the agent still scores +1.99 [+0.13, +3.77] over chance on it. A file with
  no rank channel is not a file with no channel; the control is clean for the
  reader it was built against and not for this one. That is why every margin in
  this section is taken over the control rather than over chance, and it is the
  same lower-bound caveat §7 states, observed on the control instead of on a
  released file.
- **The placebo and repair, applied to the agent.** The same two rewrites the
  rest of the paper uses: redraw every distractor with the key's rank *held* at
  the released file's, and again with it uniform, both through one generator
  differing only in the target rank.

  | arm | key rank *p* | agent's pick *b* | ⟨p,b⟩ − 1/4 | over chance | over clean |
  |---|---|---|---|---|---|
  | released | 12.4, 51.4, 29.5, 6.7 | 8.9, 29.8, 39.4, 21.9 | +4.5 [+1.7,+7.4] | +8.3 | **+11.1 [+3.0,+19.6]** |
  | placebo | 12.4, 51.4, 29.5, 6.7 | 9.5, 30.8, 40.0, 19.7 | +5.1 [+2.3,+8.3] | +3.3 | +6.0 [−1.7,+13.4] |
  | repaired | 23.8, 29.5, 27.6, 19.0 | 9.8, 31.4, 38.4, 20.3 | +1.1 [−1.1,+3.2] | +2.0 | +4.8 [−2.8,+12.3] |

  Three readings, in decreasing order of how much the data supports them.
  (i) **The agent's rank preference does not move.** *b* is within 1.6 points
  per cell across three files with different option text and, in the repaired
  arm, a different key-rank distribution. It is a property of the agent, which
  is what eq (1) assumes and what nothing else in this paper had measured
  directly. (ii) **The predicted rank term follows the file**, staying at +5.1
  where the rank is held and falling to +1.1 with an interval covering zero once
  it is uniform — the repair closes the channel it was built to close.
  (iii) **The observed margins fall monotonically**, +11.1, +6.0, +4.8, but only
  the released arm clears its control; at 105 items the two rewritten arms
  cannot be told apart, and we do not claim they can.
- **What the rewrite destroys is not the rank.** The wording split collapses on
  both rewritten files: rollouts reaching for the middle score 29.1% against
  28.0% on the placebo and 28.0% against 26.7% on the repair, where the released
  file gave 45.2% against 29.8% — even though the placebo holds the key's rank
  exactly. Holding the rank is not enough to reproduce what the agent reads; the
  key also has to be the plausible value in its own set, which is the per-option
  channel the rest of the paper measures. This also answers the recognition
  worry: the placebo's option sets are newly generated, so they are not the
  released ones a model could have memorised, and its margin is +6.0 rather
  than zero.
- **What the condition has to say, and what the control is worth.** Told only
  that the question is `[withheld]`, Llama-3.1-8B emits no answer on 69.2% of
  rollouts — that arm measures refusal, not geometry, which is why
  `withheld_aware` exists and why compliance is reported beside every margin.
  The positive control has the same problem in weaker form: given the question,
  Llama scores 20.6% while leaving 23.5% of its rollouts without a final line
  and calling the interpreter on 66% of them, where Qwen2.5-7B scores 32.1% at
  1.3% unparsed and the 14B 32.4% at 1.6%. A positive control is only a control
  while the model answers, so the paper names the rate rather than quoting the
  three accuracies side by side as if they measured the same thing.
- **Bugs this arm produced before it produced a number.** Reading the first
  capital after `FINAL:` takes the T of "The answer is B" and cost Llama 16
  points at 0% unparsed. A bootstrap difference taken in fractions where the
  margins are in points made the interval 100× too narrow and put the point
  estimate outside its own interval. Seeding the clean arm with the same seed as
  the file arm paired capsule *k* of one file with capsule *k* of an unrelated
  one and collapsed the difference interval toward zero; the clean arm is now
  seeded `seed + 1`.
- **What 105 public items cannot exclude** is that a 14B model recognises the
  released option set rather than reading its geometry. The placebo arm is the
  test that separates them: it redraws the distractors, so the presented set is
  new, while holding the key's rank at the released file's.

## The option text of seven science-agent files, and the folds that decide it

- Every other measurement in this paper reads option *values*, which confines it
  to each file's numeric slice. `text_options.py` runs the character-n-gram set
  reader on option **text**: BixBench plus the eight LAB-Bench files, each at its
  modal option count, 1,662 items in the seven files large enough to fit
  (SuppQA has 34 items at its modal k and CloningScenarios 24; both are skipped
  rather than reported at n≈30).
- The control is the part that took the most care. A synthetic numeric file is no
  null at all for a file of English sentences, so the control resamples each
  item's k options i.i.d. from the pool of every option string that file
  contains and marks one uniformly as the key: it carries the file's own
  vocabulary, length distribution and register, and has Γ = 0 by construction.
  `clean_worst` is the largest of eight such draws, the same conservative
  statistic `tab:frontier` uses.
- **No file clears its control.** Set reader: BixBench −3.4 [−6.8,+0.4],
  SeqQA −7.2 [−15.8,+1.5], DbQA −8.7 [−12.8,−2.3], TableQA −7.5 [−14.2,−1.6],
  LitQA2 +4.8 [−5.2,+16.6], FigQA −5.6 [−14.1,+6.7], ProtocolQA +9.1
  [−3.3,+20.1]. The option reader agrees file by file (BixBench −6.8, LitQA2
  −1.0, TableQA −2.8, FigQA −6.7, DbQA −5.2, SeqQA −6.3, ProtocolQA +9.1). The
  two positive point estimates are the two smallest files and both intervals
  cover zero by a wide margin.
- **The folds decide the answer, and random folds give the wrong one.** These
  files run from one source for every item (DbQA: 10 distinct subtasks over 520
  items) to one source per item (LitQA2: 103 distinct over 106). The grouping
  field is chosen per file as the first with at least ten distinct values and its
  groups are binned into at most twenty folds so that no group is split. Ignore
  that and the same reader on the same files reads SeqQA **+21.5 [+17.7,+25.1]**
  and DbQA **+5.6 [+2.8,+8.4]** — a channel manufactured entirely by splitting
  templated subtasks across the fold boundary. `--ignore-groups` reproduces it.
  This is the fold trap, and it is the reason a survey of this kind can report a
  positive result on a file with nothing in it.

## Flat margins are not a flat joint

- `joint_uniformity.py`: the share of keys landing on the most common
  (length, roundness) tuple, chosen on training clusters, scored on a held-out
  one — the audit's own estimator, one coordinate wider.
- Ten options: released joint 2.2%, repaired **4.4% [2.7, 5.9]** against 1% flat
  — but the clean control is 6.7% mean and **8.0% worst**, so neither file is
  established as concentrated. The hypothesis that the learner's column-two
  residue was joint dependence is **not supported**.
- Four options: released 41.9% (+14.5 over the clean worst case), repaired 16.2%
  (−11.3), joint-drawn 2.9% (−24.6).
- `mcq_audit.py --joint-draw` fits the joint instead of the margins and the two
  objectives trade: at four options the joint goes 16.2% → 2.9% and the written
  length margin 25.7% → 37.1%; at ten the fit reports 3.9% as the flattest
  attainable joint cell and written length reopens at +1.3 points against the
  marginal fit's −0.9. Neither file admits both, and the fit says so without
  running a repair.

## A bug the held-out measurement found

- `mcq_audit.significant_digits` stripped `+`/`-` from a numeral but not U+2212
  MINUS SIGN, which `parse_number` accepts. Every option written with it read one
  significant digit less round than it is: 65 options over 24 of MMLU-Pro's 1,263
  numeric items, enough to move those items' roundness rank. Fixed, regression
  test added, and every MMLU-Pro number in this file and in the paper recomputed
  after the fix. BixBench contains no such option and is unchanged.

## Four more defects the pipeline found in itself

- **The isolation statistic depended on the Python version.** `isolation_scores`
  summed each option's similarities with `sum`; CPython 3.12 sums floats with
  Neumaier compensation and 3.10 does not, so `['0.1126','0.0867','0.0713',
  '0.1433']` scores −1.5 on one and −1.4999999999999998 on the other, and the
  key's isolation rank comes out 2 or 1. It decides the rank on 841 of the
  38,710 option sets this artifact ships or rebuilds (2.2%): 16 of v1.5's 615,
  32 of MMLU's 14,033, and 793 of MMLU-Pro's 24,062, ten options making
  near-ties denser. Both the
  audit (`mcq_audit.isolation_scores`) and the rule family
  (`text_artifacts._odd_one_out`) now use `math.fsum`, which is exactly rounded
  on every interpreter and agrees with 3.12 on all 38,710 sets: no published
  number moves, `results/repaired/bixbench_v15_repaired_multi.jsonl` and
  `results/mcq_audit_multi.json` regenerate byte-for-byte under 3.10.12 and
  3.12.11 alike, and `tests/test_channels.py` pins the knife-edge set.
- **A missing cluster field silently became 661 clusters.** `build_items` gives
  an item its own cluster when the requested field is absent, so
  `--cluster-field cluster` on a file whose field is `capsule_uuid` bootstraps
  661 independent units where there are 39 subjects, and every interval comes out
  too narrow. It now prints a warning when no row carries the field. The MMLU
  measurements in this file were re-run with the default (`capsule_uuid`).
  `held_out_orderings.py` had the matching hazard in its clean reference, whose
  size was hard-coded to v1.5's 105 items in 46 capsules; it is now taken from
  the file being measured, which reproduces 105/46 on v1.5 and 1,263/67 on
  MMLU-Pro exactly.
- **`README.md` documented two commands that did not build the shipped file.**
  The BixBench multi-channel line omitted `--search construct`, so following it
  produced the sampler's repair rather than the constructive one; the
  `placebo_contrast.py` line named only `results/probe_mmlupro_placebo`, three
  of the ten models the MMLU-Pro panel is computed over, the other seven being
  in `results/probe_mmlupro_panel`. Both fixed, and
  `results/mcq_audit_multi.json` and `results/placebo_contrast.json` regenerated
  from the corrected commands — the second reproducing all ten MMLU-Pro solvers
  and every number the paper quotes from it.
- **Nine results recorded a session scratch path as their input.** The
  ten-option and MMLU results were produced from files built outside the
  repository, so a reader could not tell what `--jsonl` had been. The MMLU-Pro
  audits now name `build/mmlu_pro.jsonl`, which
  `survey_to_jsonl.py --survey mmlu_pro` rebuilds from vendored data:
  re-running them against it reproduces `results/mcq_audit_mmlu_pro.json` and
  `results/mcq_audit_mmlu_pro_isolation.json` **bit for bit apart from the
  recorded path**, which is also the check that the rebuild is the same file.
  The MMLU results are now built from `build/mmlu.jsonl`
  (`survey_to_jsonl.py --survey mmlu`), 661 numeric items in 39 subjects. The
  probe inputs carry the upstream question text and item ids and are not
  redistributed; their option sets are identical, in the same order, to the
  vendored rebuild (12,031 rows released and repaired, verified row by row).

## The cluster bootstrap, calibrated at the counts this paper runs at

Almost every interval here is a percentile cluster bootstrap over 20 to 60
groups. `bootstrap_calibration.py` simulates files from each agent arm's own
cluster-size distribution and its own between-cluster spread (a beta-binomial
matched by moments, so the intra-cluster correlation is the arm's) and runs the
same bootstrap on them, 4,000 files per cell.

- **BixBench's 46 capsules cover.** 0.934 for a margin against chance and 0.941
  for a difference against a second arm, against a nominal 0.95. A nominal
  0.960 covers 0.95.
- **MMLU-Pro's 60 clusters do not.** 0.894 and 0.903. A nominal 0.982 is needed.
  Coverage is flat in the cluster count — 0.886 at 20 clusters and 0.917 at 120
  — so the defect is not how many clusters there are.
- **It is how unequal they are, and that is model-free.** MMLU-Pro's clusters
  are source subjects: `theoremQA-Math` alone holds 214 of 913 items. Kish's
  effective count, `(Σn)²/Σn²`, is **10.7, not 60**. BixBench's capsules are
  near-equal and give 29.6 of 46.
- **Every MMLU-family file in the repository shows it.** Effective over nominal
  runs 0.17 to 0.26: the frontier's 20 subjects are 4.8, MMLU's 39 are 7.0, the
  feasible cut's 37 are 6.5. The synthesised clean controls, whose clusters are
  balanced by construction, sit at 1.00 — so a margin and its control do not
  have the same resolving power, and the difference inherits the worse of them.
- **The correction, applied.** `arm_intervals.py` recomputes each agent arm from
  its dumped rollouts. It reproduces every published 95% interval bit-for-bit
  (it mirrors `agentic_probe.summarise`'s draw sequence), then reprints at the
  level that covers. Qwen2.5-32B on MMLU-Pro goes from +5.48 [+2.69, +9.23] to
  +5.48 [+2.10, +10.20] and still clears; BixBench's move by under half a point.

**Lesson.** A nominal cluster count says nothing about an interval. The grouping
a benchmark ships is chosen for provenance, not balance, and one dominant group
can cost five-sixths of the independence the count advertises. "Size the
benchmark for the inference" is a statement about effective clusters.

## Taking the interpreter away, and what that exposed

This is the arm separating "an agent collects it" from "a tool collects it":
`agentic_probe.py --no-tools` runs the same loop and the same instruction with
no Python.

- **Qwen2.5-14B, MMLU-Pro, steered instruction:** +4.06 [+1.86, +6.34] with no
  interpreter, against +6.25 [+4.04, +8.69] holding one. The intervals overlap.
  The interpreter is worth about two points and is not the mechanism.
- **Equation (1) still fits it.** ⟨p,b⟩ − 1/10 is +2.57 [+1.86, +3.65] on the
  released file against +4.79 observed over chance, and −0.05 [−0.33, +0.24] on
  the clean control.

Then the dump said something the arm was not built to find. Told to think it
through with nothing to run, **the model does not think it through**: 89.8% of
its rollouts are the final line and nothing else, a median reply of 8
characters. Splitting on that (`agentic_mechanism.py`, `by_writing`):

| rollouts | share | released | clean | margin |
|---|---|---|---|---|
| answered bare | 89.8% | 15.06% | 9.99% | **+5.07** |
| wrote something first | 10.2% | 12.37% | 13.87% | −1.51 |

**The margin is entirely in the rollouts that wrote nothing.** So the channel is
not collected by the tool and not by the reasoning. It is a direct preference
over the option set, and what the scaffold changes is whether that preference
gets to express itself.

**Which exposed a confound in our own headline.** "Single-token probes read
nothing, the agent reads +6.3" compares two numbers produced by different
prompts, different chat templates and different read-outs. That contrast is not
identified, and reading this dump is what showed it. `--argmax` fixes
it: same system message, same user message, same template, same items and
orderings, with `FINAL:` pre-filled and the letter taken as the arg-max over the
k single-token continuations. Crossed with the instruction it gives
{steered, neutral} × {generated, arg-max} on one model and one file.

**Lesson.** An arm built to rule one thing out is worth reading for what it
shows about the arms you already trusted. The tool-use split inside a single arm
(+12.3 no code against +5.5 code) pointed the same way as this arm and could not
have replaced it: which rollouts run code is the model's choice, so that split
compares two self-selected populations, not two conditions.

## A reproduction that writes to the original's path is not a reproduction

`run_agentic8.sh` re-ran the 14B's MMLU-Pro steered arm on a second card to
confirm the determinism claim, at batch 12 rather than the published batch 10,
and pointed `--dump` at `build/dumps/qwen14b_mmlupro.jsonl` — the published
arm's own dump.

- Greedy decoding is deterministic **given the batching** and not otherwise, so
  the re-run is a different set of rollouts: **+5.60 against the published
  +6.25**, a tenth of the interval, reported in `app:agent` as its own finding.
- The published arm's rollouts were overwritten. No number in the paper moved —
  `results/agentic_mmlupro_qwen14b.json` was never touched — but
  `arm_intervals.py` rebuilds every published interval from the shipped
  rollouts, and it rebuilt `[+3.11,+8.58]` where the paper says
  `[+4.04,+8.69]`.
- **It printed `(PUBLISHED SAYS +4.04,+8.69)` rather than the number**, which is
  why this was caught at all. A reproduction script that silently agrees with
  whatever it is handed checks nothing.

Fixed by naming: every dump path now carries its cell and, where it differs, its
batch size (`qwen14b_mmlupro_steered_tools.jsonl`,
`qwen14b_mmlupro_batch12.jsonl`), and the batch-10 arm is re-run
(`run_agentic16.sh`) rather than explained away.

## It is the prompt. Not the tool, not the instruction, not the read-out

The confound above is closed by crossing the two axes on one model, one file,
one set of items and one set of letter orderings (`agentic_probe.py --argmax`,
`--prompt bixbench`, `--no-tools`). Qwen2.5-14B, MMLU-Pro's 913 ten-option
numeric items, 60 source-subject clusters, margins over the matched clean
control, intervals from `arm_intervals.py` at the nominal 95%:

| how the question is put | highest-logit letter | generated |
|---|---|---|
| BixBench's own template, verbatim | 11.0%, **+1.3 [−1.6,+3.8]** | 10.1%, **+0.5 [−2.4,+2.8]** |
| "withheld, read the options" | 17.0%, +5.2 [+2.3,+8.5] | 14.8%, +4.1 [+1.9,+6.3] |
| question simply absent | 19.4%, +8.3 [+5.0,+12.4] | 20.8%, **+10.4 [+7.1,+14.3]** |

**Read down a column, not across a row.** The read-out is worth about a point;
the prompt is worth nine. The top row is the 19-pair null reproduced as one
cell of a controlled design rather than as a separate experiment, which is what
it had been.

Three things it is not, each of which we expected it to be:

- **Not the interpreter.** +10.4 with none against +4.6 holding one, and at 32B
  +5.3 [+2.5,+8.9] against +5.5 [+2.7,+9.2]. Taking the tool away *raises* the
  margin, because the cell that reads highest is the one that answers
  immediately.
- **Not an instruction to read the options.** Naming the thing being measured —
  "study the options themselves and choose the one most likely to be the
  intended answer" — *lowers* every read-out against simply omitting the
  question (+5.2 against +8.3; +4.1 against +10.4). This is the direction
  opposite to steerability, and it is why the paper does not call the result one.
- **Not reasoning.** Told to think it through with nothing to run, the model
  answers with the final line and nothing else on 99.9% of rollouts, and that is
  the cell that reads highest.

**Equation (1) locates the difference.** Fitting the rank model per cell gives
the same key-rank law `p` throughout — it is one file — and a different
pick-rank preference `b`:

| cell | accuracy | ⟨p,b⟩ − 1/10 | answered bare |
|---|---|---|---|
| BixBench template, one letter | 11.0% | **+0.35 [−0.31,+0.81]** | 100% |
| BixBench template, generated | 10.1% | +0.31 [−0.36,+0.79] | 0% |
| steered, no tool | 14.8% | +2.57 [+1.86,+3.65] | 89.8% |
| steered, one letter | 17.0% | +2.87 [+2.03,+4.17] | 100% |
| neutral, interpreter | 15.9% | +3.53 [+2.69,+4.71] | 0.3% |
| neutral, one letter | 19.4% | +3.47 [+2.68,+4.60] | 100% |
| neutral, no tool | 20.8% | **+3.15 [+2.35,+4.25]** | 99.9% |

Under BixBench's template the preference is nearly flat and tilted to the
*smallest* option. Under the neutral framing it peaks where this file's key
sits, rank five of ten. **The prompt does not give the reader more of the file;
it changes what the reader prefers.**

**A second family, and the arm it cannot run.** Both models that collect the
channel are Qwen2.5, so "the two largest of four" was also "the two Qwen of
four", and the sweep cannot separate them. phi-4 (14B,
Microsoft, a different pretraining corpus) was run on the two cells that carry
the result.

- **BixBench's template, one letter:** **+2.59 [−0.35,+5.10]** over 1,826
  rollouts, **0.0% unparsed**. Does not clear. The null replicates outside
  Qwen2.5.
- **The neutral framing, generated:** it does not answer. **64.7% of released
  rollouts and 38.8% of control rollouts produce no final line**, and a regex
  over the refusal wordings matches **100%** of them — *"Without the specific
  question, it's impossible to determine which option is correct"*, then four
  turns of general advice on how to approach a multiple-choice question.

**So the prompt effect does not replicate in a second family, and we do not
claim it does.** The cell that reads highest is not a cell every model can be
put in; Llama-3.1-8B fails it the same way. Worse than noise: the refusal rate
*differs between the file and its control* (64.7% against 38.8%), so the −2.11
[−3.92,−0.44] that arm reports is a difference in refusal propensity between
two option sets, not a geometry reading. It is reported and nothing is read off
it.

**Lesson.** A no-data baseline has to publish compliance **per arm**. Pooled, or
omitted, a 65% refusal rate reads as an option preference — which is exactly the
failure the `FINAL:`-parsing trap produced earlier in this repository, arriving
by a different route.

## On the venue's own file, our run of the field's template collects ---
## and the numeric half is all of it

The same crossing, on BixBench's whole released file: all 205 items in 59
capsules, against the control resampled from the file's own pool of 709 option
strings (`run_agentic15.sh`).

| how the question is put | highest-logit letter | generated |
|---|---|---|
| BixBench's own template, verbatim | 30.1%, **+8.5 [+3.2,+13.6]** | 28.0%, **+8.1 [+3.0,+13.1]** |
| question simply absent | 37.9%, **+14.1 [+8.1,+19.9]** | 38.4%, **+15.0 [+8.8,+20.7]** |

**Every cell clears, including the one the field runs.** At ten options
BixBench's template reads +1.3 [−1.6,+3.8] and does not clear; on BixBench's own
file the same template reads +8.5 [+3.2,+13.6]. So **what an instrument reads is
a property of the file as well as the prompt.**

**Whose run this is, exactly.** That +8.5 is not BixBench's *published*
baseline. It is **our** run of their template, verbatim at commit `4931118`, on
open-weight models the benchmark never used. Their published runs are on v1.0
and fit λ̂ = 0.028 and 0.069, neither distinguishable from zero. So the paper
does not say "the published baseline is already collecting", and
`validate_artifact.py` fails if the phrase appears.

**And the reading is the numeric half.** Is the 205-item margin localised to the
mechanism the paper measures, when half the file's options are not numbers and a
rank is only defined over values? It is
(`cell_split.py`, which splits on the rollout itself — an item is numeric when
every one of its option strings parses as a number):

| BixBench cell | numeric, 105 items | text, 100 items |
|---|---|---|
| template, one letter | **+10.6 [+5.9,+15.8]** vs chance | **−0.7 [−5.2,+4.0]** vs chance |
| question absent, generated | **+15.0 [+9.4,+22.1]** vs chance | **+11.7 [+6.4,+17.4]** vs chance |

So **the union's +8.5 is the numeric half**: under the field's own template the
non-numeric hundred read nothing at all. Under the framing that omits the
question both halves clear, so the prompt reaches text options where the
template does not. Two bugs this surfaced, both of which had produced a
believable number:

- **BixBench's template puts an instruction *after* the options**, so a greedy
  match to end-of-string swallowed it, the last option line failed to parse, and
  all 205 items classified as non-numeric. The split looked like a null.
- **The control was subset by its own option text**, not by item. Its option sets
  are resampled from the file's pooled strings, so a control row can be numeric
  where the item it stands in for is not; selecting it by text broke the pairing
  and printed a numeric interval twenty points *wider* than the whole file's.

The steered framing holding the interpreter completes the row the table has no
column for: **+10.6 [+3.8,+17.7]**, between the template's +8.5 and the neutral
framing's +15.0, the same ordering the ten-option file gives with the whole band
shifted up — a four-option file with the key second-smallest half the time has
more to give.

**Two things to read carefully.**

- **The clean control sits below chance here**, 1.3 to 5.2 points, so margins
  over the control are the larger of the two readings. Both are reported: over
  chance alone the released file is +12.9 [+8.8,+17.3] under the neutral framing
  and +5.1 [+1.4,+8.9] under the template, so neither headline depends on the
  control reading low.
- **One cell does not clear over chance on its own**: BixBench's template
  generated, +3.0 [−0.7,+6.7]. It is also the only cell with an unparsed rate
  worth naming, 8.8%.

## And not a property of scale: eleven models, five families

Four models, two of them Qwen2.5, are an n=2 convenience sample for a scale
claim, and there is a worse problem: **those four arms were steered *and* held
an interpreter**, so they varied the model and the instruction together.

`run_agentic19.sh` and `run_agentic20.sh` separate them — the two extreme cells
of the instrument table, same 913 items, same clean control, only the weights
change (`scale_grid.py`):

| model | B | BixBench's template, one letter | question absent, generated |
|---|---|---|---|
| Llama-3.2-1B | 1.2 | −0.6 [−2.8,+1.6] | −0.1 [−2.1,+2.0] |
| Qwen2.5-1.5B | 1.5 | +0.2 [−1.8,+2.3] | **+4.1 [+2.3,+6.1]** |
| Llama-3.2-3B | 3.2 | −3.1 [−5.5,−1.0] | +1.7 [−0.6,+3.8] |
| Phi-3.5-mini | 3.8 | +0.8 [−1.4,+3.0] | *refuses 13%* |
| gemma-3-4b | 4.3 | +0.3 [−2.2,+2.6] | *refuses 54%* |
| OLMo-2-1124-7B | 7.3 | +1.2 [−0.8,+3.0] | *refuses 37%* |
| Qwen2.5-7B | 7.6 | +0.2 [−1.7,+2.2] | **+4.6 [+1.6,+7.9]** |
| Llama-3.1-8B | 8.0 | −1.5 [−4.1,+0.9] | *refuses 14%* |
| phi-4 | 14.7 | +2.6 [−0.4,+5.1] | *refuses 65%* |
| Qwen2.5-14B | 14.8 | +1.3 [−2.4,+4.5] | **+10.4 [+6.6,+15.4]** |
| Qwen2.5-32B | 32.8 | **+4.5 [+1.6,+7.5]** | **+5.6 [+2.4,+8.8]** |
| Llama-3.3-70B | 70.6 | +1.5 [−1.2,+3.8] | **+5.4 [+2.3,+9.2]** |
| Qwen2.5-72B | 72.7 | +2.5 [−0.5,+5.2] | **+8.6 [+5.1,+12.6]** |

*(Intervals at 98.2%, the level that covers 95% on this file. Earlier rows in
this file quoted the nominal 95%; the 32B and 72B are why that was changed.)*

**The prompt effect is not a Qwen2.5 fact.** phi-4 could not settle that — it
refuses the neutral cell. **Llama-3.3-70B answers every rollout of it** and goes
**+1.5 [−1.2,+3.8] → +5.4 [+2.3,+9.2]** across the same two cells. That closes the
family confound.

**Under the instrument the field runs the reading stays small at every size**:
thirteen readable cells spanning −3.1 to +4.5 across two decades of parameter
count, of which **one clears** — Qwen2.5-32B — with both 70B-class models above
it not clearing, so what ordering there is is not monotone. **Changing the prompt moves
one model further than changing the model by a factor of sixty does**: the 14B
goes +1.3 → +10.4, a gap of 9.1 against a 7.6-point spread over the whole
column. Under the neutral framing every Qwen2.5 clears, from **1.5B (+4.1)** to
72.7B (+8.6). The 7B that reads −0.7 steered-with-an-interpreter reads +4.6
neutral. There is no parameter threshold to estimate.

**An earlier version of this section said "nothing clears at any size."** That
was true of the ten cells measured at the time and false once Qwen2.5-32B was
added. The claim it was replaced with is the one the data supports and is
narrower.

**What does bind is compliance, and it is half the grid.** Five of eleven models
decline the neutral cell on 13% to 65% of rollouts, at rates that differ between
the released file and its control, so a margin there would be a difference in
refusal propensity. Those cells print a refusal rate where a number would go.
**The framing that reads the channel is one half these models will not answer** —
a fact about the instrument, not a gap in the table.

**Lesson.** Two arms that differ in two ways measure neither. The sweep varied
scale and instruction together and reported scale, which is the same error as
"single-token reads nothing, the agent reads +6.3" in a second costume, found
the same way: by building the cell that holds one of them fixed.

## The same grid on the venue's own file, where the field's instrument does collect

`run_agentic21.sh`, all 205 released BixBench items in 59 capsules against the
control resampled from the file's own pool of 709 option strings, intervals at
96.0% — the level that covers 95% on this file's balanced capsules:

| model | B | BixBench's template, one letter | question absent, generated |
|---|---|---|---|
| Llama-3.2-1B | 1.2 | −1.5 [−7.0,+3.8] | +1.6 [−4.4,+7.6] |
| Qwen2.5-1.5B | 1.5 | **+6.2 [+0.4,+11.7]** | *refuses 3%* |
| Llama-3.2-3B | 3.2 | +1.8 [−5.6,+8.8] | **+7.0 [+1.0,+13.3]** |
| Phi-3.5-mini | 3.8 | **+7.5 [+1.3,+13.8]** | *refuses 5%* |
| gemma-3-4b-it | 4.3 | +4.2 [−2.5,+11.1] | *refuses 10%* |
| OLMo-2-1124-7B | 7.3 | +2.4 [−3.1,+8.0] | +3.4 [−1.9,+9.3] |
| Qwen2.5-7B | 7.6 | +3.1 [−1.8,+7.9] | **+7.2 [+1.2,+13.1]** |
| Meta-Llama-3.1-8B | 8.0 | +3.9 [−2.4,+10.4] | **+7.0 [+0.3,+13.8]** |
| phi-4 | 14.7 | **+12.2 [+5.8,+18.9]** | *refuses 35%* |
| Qwen2.5-14B | 14.8 | **+8.5 [+3.2,+13.6]** | **+15.0 [+8.4,+20.9]** |
| Qwen2.5-32B | 32.8 | +5.4 [−1.4,+12.6] | **+6.7 [+0.2,+12.8]** |
| Llama-3.3-70B | 70.6 | **+6.8 [+0.7,+13.4]** | **+8.0 [+1.1,+14.7]** |
| Qwen2.5-72B | 72.7 | +6.2 [−0.7,+13.1] | **+6.7 [+0.1,+13.1]** |

**This is close to the opposite of the ten-option result, and that is the
finding.** At ten options the field's own instrument clears on **one** of the
same thirteen (Qwen2.5-32B, +4.5 [+1.6,+7.5]). On BixBench's own file **five of
thirteen clear under that same instrument** — phi-4 at +12.2, the 14B at +8.5,
Phi-3.5-mini at +7.5, Llama-3.3-70B at +6.8 and a **1.5B model** at +6.2, across
five families and sixty times the parameter count, with no change of prompt.

**Whose run that is.** These are *our* runs of BixBench's template, verbatim at
commit `4931118`, on open-weight models the benchmark never used — not the
benchmark's published baseline. An earlier version of this file said the
published baseline "is collecting option geometry as published"; that sentence
described this table, which is not a published baseline, and it was withdrawn.
What the published arms do say is measured separately and by counting only:
34.1% and 36.1% with four forced options against 2.9% and 2.9% with the options
deleted (`published_repair.py`).

**Neither column is monotone in scale.** The fitted trend is +2.9 points per
decade here and +2.2 at ten options, and the highest reading of the thirteen is
phi-4's at 14.7B — above both 70B-class models. Sixty times the parameters buys
about four points; changing the prompt on one model is worth nine.

**phi-4's refusal is the framing, not the file.** It declines the neutral cell
here too, on 35% of released against 16% of control — which is why it was in
this list.

**One convention, stated once.** The grids are read at the level that covers and
the instrument table at 95%. Across every arm in the paper **exactly one changes
conclusion between the two**: Qwen2.5-72B's ten-option template cell, +2.48
[+0.23,+4.80] nominal and [−0.49,+5.25] calibrated. That single case is why the
grids use the wider interval, and `validate_artifact.py` fails if a second one
ever appears.

## The placebo at ten options: the framing is necessary, and not sufficient

BixBench had a rank-holding placebo; MMLU-Pro, the arm the agentic claim rests
on, did not (`build_placebo_arm.py`, `run_agentic14.sh`). Every option set is
newly generated and the key's rank law matches the released file's to a tenth of
a point, so a margin that survives is not recognition of a published option set.
Run under the cell that reads highest — neutral framing, no tool, generated:

| file | accuracy | margin over clean | ⟨p,b⟩ − 1/10 |
|---|---|---|---|
| released | 20.8% | +10.4 [+7.1,+14.3] | +3.15 [+2.35,+4.25] |
| placebo (rank held, text redrawn) | 13.3% | **+2.8 [−0.3,+5.3]** | **+0.11 [−0.18,+0.57]** |

905 of 913 items survive the rewrite. **The placebo no longer clears**, and its
rank term goes to zero although its key-rank law is the released file's by
construction. So the reading needs *both* the framing and the published option
text.

**What this does and does not settle.** It rules out that any of the +10.4 is
carried by the rank law alone — hold the rank exactly, redraw the values, and
the margin is gone. It does *not* choose between recognition of option sets the
model has seen and a channel in the released text that the redraw destroys,
because the redraw destroys both at once. The paper says so and claims neither.
At four options on BixBench the same rewrite leaves `b` stable (§4), so this is
a statement about a ten-option file and not a general one.

## The published baselines were run against a different file

§3 measures the rank channel on **v1.5**, the current release. §4 fits equation
(1) to the two published no-data runs, which exist for **v1.0** alone, because
only v1.0's run files record the order the model was shown. Are those the same
file? `release_drift.py` says they are not.

- **v1.0**, over the 159 rows the published runs cover: key-rank law
  (22.6, 30.2, 30.8, 16.4)%. Ceiling on what a rank rule can take: **+5.8
  points**.
- **v1.5**, over its 104 items with four distinct numeric options: (12.5, 51.0,
  29.8, 6.7)%. Ceiling: **+26.0 points**.
- **The keys did not move; the items did.** On the 39 option sets both releases
  carry, the key is the same one on 37.

Two consequences, both stated in §4. The published λ̂ of 0.028 and 0.069 — neither
distinguishable from zero — were measured against a channel a fifth the width of
the one the current release ships. And the channel the current release ships has
never had a published baseline run against it at all.

The v1.0 law is computed through `choice_model.pairs_from_v10`, the same loader
`published_baselines.py` uses, and reproduces its item count exactly (159 rows,
137 skipped for options that are not distinct numbers).

## The probe's extreme reading, named

§4 reported the five-arm probe as a range, −7.8 to +3.0, and called it a null.
`rank_effect_intervals.py` puts a paired cluster bootstrap on each solver's
stem-withheld repair-minus-placebo and three of eleven clear zero on BixBench,
all negative: Qwen2.5-32B −7.8 [−11.1, −4.1], Qwen2.5-14B −3.1 [−5.1, −0.7],
Qwen2.5-1.5B −2.6 [−4.6, −0.3]; on MMLU-Pro only the 32B, at −4.2 [−6.1, −1.7].

The sign reads as rank collection and it is not rank:

- The 32B's own rollouts give `⟨p,b⟩ − 1/4` of **+1.2** on the placebo arm and
  **−0.0** on the repair, so the model predicts a gap of 1.3 against 7.8.
- Its pick distribution is nearly flat on both, (24.8, 27.4, 25.7, 22.1) and
  (19.7, 26.2, 25.8, 28.2).
- Neither arm clears on its own: +5.4 over chance on the placebo, −2.3 on the
  repair, both intervals straddling zero, and **no** solver of the eleven is
  distinguishably below chance on the repaired file.

What the paired contrast sees is about four points of non-rank channel in the
placebo arm and two in the other direction in the repair. Both arms are
rewritten, and a rank target constrains which values can be drawn and so how
they are written. **The design built to isolate rank does not fully isolate it**,
which is §5's conclusion arriving from inside the control meant to rule it out.
The fitted geometry terms are read off the released file and are untouched; the
caveat attaches to every placebo–repair difference, the agent's included.

## The benchmark published the repair, and it is the paper's lead result

Proposition 1 says Γ is a functional of V, so **deleting V sets it to zero
identically** — there is no option set left to read. That repair is usually
hypothetical. It is not here: **BixBench ran its no-data condition three ways and
released all three**, and the sentence the field quotes is from one of them.
Nothing below is our instrument. The responses and the grades are upstream bytes
at commit `4931118`, with SHA-256 in `data/external/PROVENANCE.json`;
`published_repair.py` only counts them.

All 205 v1.5 questions, 56 capsules (43.9 effective), BixBench's own graders:

| arm | claude-3-5-sonnet | gpt-4o |
|---|---|---|
| four options, no way to decline — **the quoted run** | **34.1%** (+9.2 [+2.6,+16.2]) | **36.1%** (+11.1 [+4.2,+18.6]) |
| four options plus "insufficient information" | 8.3%, declines on **78.5%** | 3.9%, declines on **90.2%** |
| **no options at all** | **2.9%** | **2.9%** |

Paired over capsules, each arm against its own chance rate, **the option set is
worth +6.2 [−0.4,+13.5] and +8.2 [+1.1,+15.9]**.

On the **61 questions BixBench grades with its own numeric tolerance** — a
range, so an answer close enough counts — the deleted-option arm reads **0.0%
for both models**, against 23.0% and 29.5% with the options in place at a 25%
chance rate.

**Three alternative readings, checked rather than argued.**

- *"Open-ended grading is just stricter."* The third arm keeps the four options
  and the letter and only adds a way to decline: 8.3% and 3.9%, because the
  models take that option on 78.5% and 90.2% of the questions. Same grader, same
  response format, same answer.
- *"They know it but the verifier fails them."* Where they do commit, they are
  right 38.6% and 40.0% of the time — no better than the forced arm manages on
  *everything*. Their own confidence signal carries almost nothing about which
  items they can answer.
- *"The open-ended responses are unparseable prose."* They are declinations. On
  54.6% and 41.0% of them (77.0% and 52.5% on the numeric subset) the model
  states that the value cannot be determined without the data — which is what
  the benchmark says it designed for. Measured by a deliberately conservative
  regex, and corroborated by a number nobody had to classify: the read-off
  decline rate from the arm that offered the option.

**Three arms, two graders, two response formats and three chance rates put these
models' unaided no-data score between nought and eight per cent. The published
thirty-four is what four options and a forced choice add.**

**And it is a release story.** On v1.0 the two arms *agree*: +8.8 against +11.5
and +7.1 against +9.1 over 296 questions in 53 capsules, differences of
−2.7 [−9.3,+3.6] and −2.0 [−8.3,+4.5]. v1.0 is the release the published
baselines were actually run on, and it is the release whose rank channel caps at
+5.8 rather than +26.0.

**One bug worth recording.** v1.0 writes the literal string `empty` in the
column that names the decline letter. Reading it as a letter reported a decline
rate of **0.0%** for an arm that had no decline option at all — a plausible
number for a condition that does not exist.

## And it is not two models: the repair run across six more

The published result above is one file and two closed models, which is the
limitation §6 states. `free_response.py` removes it by running **the same two
arms ourselves** — the question with its four options and a forced letter, and
the question alone with the answer as a number — across open-weight models on
BixBench's same 105 numeric items.

Grading a free-response number needs a rule, and the rule is the weak point of
the argument, so it is stated and varied rather than chosen: the **last** number
in the reply counts as correct when it is within a **relative tolerance** of the
key (absolute against a key of zero), reported at 1%, 5% and 10% together, with
exact string equality after normalisation also counting. A percentage written on
one side only is normalised, which is *generous* to the model — so a low score
here is a lower bound on strictness, not an artefact of it.

| model | B | MCQ, margin over 1/4 | free response (5%) | declines |
|---|---|---|---|---|
| Qwen2.5-1.5B | 1.5 | +9.8 [+3.5,+16.2] | **1.0%** | 0% |
| Llama-3.2-3B | 3.2 | +4.5 [-2.5,+11.5] | **3.8%** | 4% |
| Qwen2.5-7B | 7.6 | +6.4 [-0.8,+14.5] | **3.8%** | 0% |
| Meta-Llama-3.1-8B | 8.0 | -1.7 [-8.6,+4.8] | **1.0%** | 9% |
| phi-4 | 14.7 | **unreadable**, 40% unparsed | **1.0%** | 58% |
| Qwen2.5-14B | 14.8 | +15.5 [+6.8,+25.0] | **1.9%** | 27% |

**Free-response accuracy spans 1.0% to 3.8% across six models** — and
BixBench's two published closed models score 2.9%. **Eight models, four
families, every one between one and four per cent.** Meanwhile the
multiple-choice arm on the *same items* spans −1.7 to +15.5 over chance.

**What a model can state unaided is near-constant. What the instrument reads off
it is not.** That is the whole claim, and it now rests on eight models rather
than two.

**Read the two columns differently**, and the compliance rule says why. phi-4
leaves **40.5% of its multiple-choice rollouts unparsed**, so that cell is
withheld by the same >5% rule `scale_grid.py` applies — a margin on a model that
will not answer is a refusal rate wearing a margin's name. The free-response
column has no such problem: a reply with no number in it is simply wrong, and
the declination rate is printed beside it (phi-4 declines on 58%, Qwen-14B on
27%, Llama-3.1-8B on 9%).

## Membership inference on the option sets, and the control it needs

The paper's own conceded confound was that a large model might **recognise**
option sets it saw in pretraining rather than read their geometry. The placebo
bounds that reading and does not close it, so `option_membership.py` tests it
directly. Recognition makes a prediction the geometry account does not: **the
rollouts the model gets right should be the ones whose option sets it scores as
more-seen.**

Scoring each option block a rollout actually read — the exact string, from the
dump — alone under the same weights, with mean token log-likelihood,
Min-K% and Min-K%++:

| statistic | correct − wrong, pooled | correct − wrong, within key rank |
|---|---|---|
| mean log-likelihood | +0.019 [−0.007,+0.035] | +0.015 [−0.005,+0.030] |
| Min-K% | +0.013 [−0.041,+0.054] | +0.005 [−0.038,+0.047] |
| Min-K%++ | +0.028 [−0.029,+0.067] | +0.018 [−0.034,+0.062] |

**All six cover zero**, pooled and stratified on the key's rank — the stratified
version removing the geometry channel, so what is left is recognition or
nothing. 1,826 rollouts, cluster bootstrap over MMLU-Pro's 60 groups.

**One model is not an answer, and the null needed a control.** A membership
score is also a **typicality** score — a likelier option block is a more
ordinary one — and a solver does better on ordinary items. So a positive gap on
the released file is not yet recognition, and the comparison recognition
actually predicts is a gap **larger on the released file than on one nobody
published**. `option_membership.py` now reads the same difference on the matched
clean control and reports released minus control, each arm resampled over its
own clusters. Across all thirteen models of the scale grid
(`membership_grid.py`):

```
model                              B  refused over ch              loglik:within key rank                 same, clean control              released minus control  clearing
Llama-3.2-1B-Instruct            1.2    0.0%    +0.7?   +0.0068 [-0.0576,+0.0615]    -0.0955 [-0.1542,-0.0338]*   +0.1023 [+0.0165,+0.1814]*  0/6 vs 4/6 vs 2/6
Qwen2.5-1.5B-Instruct            1.5    0.0%    +4.5    +0.0177 [-0.0223,+0.0493]    -0.0209 [-0.0515,+0.0141]    +0.0386 [-0.0128,+0.0852]   0/6 vs 0/6 vs 0/6
Llama-3.2-3B-Instruct            3.2    0.9%    +1.8?   +0.0474 [+0.0009,+0.0772]*   -0.0185 [-0.0907,+0.0478]    +0.0659 [-0.0166,+0.1435]   2/6 vs 0/6 vs 2/6
Phi-3.5-mini-instruct            3.8   13.1%!   +4.2    +0.0407 [-0.0053,+0.0706]    +0.0387 [-0.0020,+0.0806]    +0.0020 [-0.0629,+0.0508]   0/6 vs 0/6 vs 0/6
gemma-3-4b-it                    4.3   54.4%!   -5.2?   +0.0045 [-0.0453,+0.0465]    +0.0254 [-0.0221,+0.0723]    -0.0209 [-0.0919,+0.0453]   0/6 vs 0/6 vs 2/6
OLMo-2-1124-7B-Instruct          7.3   37.1%!   -3.9?   +0.0184 [-0.0522,+0.0734]    -0.0188 [-0.1039,+0.0662]    +0.0373 [-0.0691,+0.1334]   0/6 vs 0/6 vs 0/6
Qwen2.5-7B-Instruct              7.6    0.0%    +3.9    -0.0241 [-0.0572,+0.0268]    -0.0078 [-0.0418,+0.0341]    -0.0163 [-0.0700,+0.0444]   1/6 vs 0/6 vs 0/6
Llama-3.1-8B-Instruct            8.0   45.8%!   +2.0?   +0.0137 [-0.0341,+0.0593]    +0.0656 [-0.0212,+0.1539]    -0.0520 [-0.1535,+0.0475]   0/6 vs 2/6 vs 0/6
phi-4                           14.7   64.7%!   -4.9?   -0.0543 [-0.0854,-0.0045]*   -0.0260 [-0.1300,+0.0706]    -0.0282 [-0.1317,+0.0889]   6/6 vs 0/6 vs 2/6
Qwen2.5-14B-Instruct            14.8    0.0%   +10.8    +0.0154 [-0.0053,+0.0304]    -0.0059 [-0.0450,+0.0330]    +0.0213 [-0.0245,+0.0631]   0/6 vs 0/6 vs 0/6
Qwen2.5-32B-Instruct            32.8    0.0%    +5.8    +0.0337 [+0.0184,+0.0480]*   -0.0033 [-0.0411,+0.0341]    +0.0369 [-0.0021,+0.0789]   4/6 vs 0/6 vs 0/6
Llama-3.3-70B-Instruct          70.6    0.0%    +5.2    +0.0410 [+0.0180,+0.0688]*   +0.0203 [-0.0588,+0.1027]    +0.0207 [-0.0635,+0.1025]   4/6 vs 0/6 vs 0/6
Qwen2.5-72B-Instruct            72.7    0.0%    +8.2    +0.0220 [-0.0257,+0.0551]    -0.0096 [-0.0456,+0.0231]    +0.0316 [-0.0292,+0.0792]   0/6 vs 2/6 vs 0/6

* = the interval excludes zero.  ! = the arm this is measured on is one the
    paper's own filter calls unreadable, because the model declines more than 5%
    of its rollouts.  ? = the model reads within two points of chance on this arm, so
    its correct and wrong rollouts are nearly the same draw.  Widest cell per model:
  Llama-3.2-1B-Instruct          mink:pooled            +0.0349 [-0.1846,+0.2023]
  Qwen2.5-1.5B-Instruct          minkpp:pooled          -0.0310 [-0.1208,+0.0405]
  Llama-3.2-3B-Instruct          mink:pooled            +0.1253 [-0.0118,+0.2133]
  Phi-3.5-mini-instruct          mink:pooled            +0.0527 [-0.0333,+0.1200]
  gemma-3-4b-it                  minkpp:within key rank -0.1558 [-0.3964,+0.0003]
  OLMo-2-1124-7B-Instruct        mink:pooled            +0.1279 [-0.0828,+0.2918]
  Qwen2.5-7B-Instruct            minkpp:within key rank -0.1544 [-0.2659,-0.0029]
  Llama-3.1-8B-Instruct          mink:pooled            +0.0361 [-0.1281,+0.1782]
  phi-4                          mink:pooled            -0.3399 [-0.4615,-0.1669]
  Qwen2.5-14B-Instruct           minkpp:pooled          +0.0279 [-0.0290,+0.0668]
  Qwen2.5-32B-Instruct           mink:pooled            +0.0671 [+0.0183,+0.1002]
  Llama-3.3-70B-Instruct         minkpp:pooled          +0.1092 [+0.0247,+0.1544]
  Qwen2.5-72B-Instruct           mink:pooled            +0.0482 [-0.0620,+0.1169]
```

```
13 models, 5 of them on an arm the >5% filter calls unreadable
  released arm clears zero somewhere: 5  (positive on 3, negative on 2)
  clean control clears zero somewhere: 3
  released MINUS control clears zero: 4
    on meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B-Instruct, google/gemma-3-4b-it, microsoft/phi-4
  of the 6 models that answer this arm and beat chance on it, the difference clears on 0
  over those 6, the stratified log-likelihood difference's interval tops out between +0.044 and +0.102 nats
{"models": 13, "released": 5, "control": 3, "difference": 4, "positive": 3, "negative": 2, "refusing": 5, "which_difference": ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-3-4b-it", "microsoft/phi-4"], "informative": 6, "informative_difference": 0, "which_informative": [], "ceiling_lowest": 0.04441984489003431, "ceiling_highest": 0.10247628739680445}
```

**Read it this way.** Five models have a released reading that clears zero and
**two of those clear negative** — correct rollouts scoring as *less* seen, which
recognition does not predict. The difference clears on four, and every one of
those four either declines the arm (gemma-3-4b on 54% of its rollouts, phi-4 on
65%) or reads within two points of chance on it, so its correct and its wrong
rollouts are nearly the same draw. Llama-3.2-1B is the clearest case: its
released arm is a null, its **control** arm is what clears, and it scores 10.7%
against 10% chance.

**Of the six models that both answer the arm and beat chance on it, the
difference clears on none**, with intervals topping out between +0.044 and
+0.102 nats. That is what these rollouts exclude — not zero, and the paper says
so in its limitations rather than claiming the confound is closed.

**Qwen2.5-32B is why the control is there.** Its released gap reads +0.034
[+0.018,+0.048] and clears comfortably; on its own that is what recognition
would look like. Its control gap is −0.003, and the difference is +0.037
[−0.002,+0.079], which does not clear. Reading the released arm alone would
have reported a memorisation finding.

**Counted per interval, which is the unit a 95% interval is calibrated on.**
Thirteen models by six statistics is 78 intervals an arm, so about four exclude
zero whatever is true. MMLU-Pro's **released** arm clears on **17 of 78** —
four times that, so there is real structure in which rollouts carry which
option blocks. Its control clears on 8, and the difference on 8, **four
positive and four negative**. That is what noise looks like; recognition has a
sign.

**What each control withholds, and why BixBench's is the cleaner test.** The
two controls are not the same kind of object.

| control | released pool | control pool | control strings that were published |
|---|---|---|---|
| BixBench, template cell | 709 | 499 | **100.0%** |
| MMLU-Pro, neutral cell | 4,009 | 5,965 | 12.9% |

MMLU-Pro's control redraws numeric **values**, so most of its strings are new.
A model recognising *strings* rather than sets would lift the released arm and
not the control, which biases that difference **towards** finding recognition —
so its null is the conservative reading. BixBench's control is built by
`build_set_control.py` from the file's own pool of 709 published option
strings: **every** string in it was published and the only thing withheld is
the grouping. There the difference is a clean test of set recognition, and it
is the venue's own file.

**On that file, run on the template cell all thirteen models answer:**

```
model                              B  refused over ch                       loglik:pooled                 same, clean control              released minus control  clearing
Llama-3.2-1B-Instruct            1.2    0.0%    -1.4?   +0.0370 [-0.0859,+0.1538]    -0.0169 [-0.1006,+0.0716]    +0.0539 [-0.0971,+0.1970]   0/6 vs 0/6 vs 0/6
Qwen2.5-1.5B-Instruct            1.5    0.0%    +3.3    -0.0995 [-0.2168,+0.0093]    -0.0943 [-0.2310,+0.0289]    -0.0051 [-0.1829,+0.1709]   0/6 vs 0/6 vs 0/6
Llama-3.2-3B-Instruct            3.2    0.0%    +1.3?   -0.0063 [-0.1490,+0.1310]    -0.0079 [-0.1191,+0.1187]    +0.0016 [-0.1840,+0.1812]   0/6 vs 0/6 vs 1/6
Phi-3.5-mini-instruct            3.8    0.0%    +5.2    -0.0157 [-0.0990,+0.0670]    -0.0355 [-0.1102,+0.0410]    +0.0198 [-0.0924,+0.1349]   0/6 vs 0/6 vs 0/6
gemma-3-4b-it                    4.3    0.0%    +5.1    -0.0790 [-0.2418,+0.0642]    -0.0211 [-0.1521,+0.1007]    -0.0579 [-0.2614,+0.1349]   1/6 vs 0/6 vs 0/6
OLMo-2-1124-7B-Instruct          7.3    0.0%    +3.0    +0.0497 [-0.0507,+0.1566]    -0.1013 [-0.2159,+0.0144]    +0.1510 [-0.0045,+0.3149]   0/6 vs 0/6 vs 0/6
Qwen2.5-7B-Instruct              7.6    0.0%    +1.0?   +0.0431 [-0.0654,+0.1392]    -0.0332 [-0.1277,+0.0618]    +0.0763 [-0.0610,+0.2142]   0/6 vs 0/6 vs 0/6
Llama-3.1-8B-Instruct            8.0    0.0%    +4.6    +0.0374 [-0.0583,+0.1382]    -0.0476 [-0.1619,+0.0827]    +0.0849 [-0.0710,+0.2438]   0/6 vs 0/6 vs 0/6
phi-4                           14.7    0.0%    +8.3    +0.0052 [-0.0856,+0.0958]    -0.0772 [-0.1962,+0.0465]    +0.0824 [-0.0745,+0.2320]   0/6 vs 0/6 vs 0/6
Qwen2.5-14B-Instruct            14.8    0.0%    +5.1    +0.0489 [-0.0462,+0.1417]    -0.0237 [-0.1215,+0.0798]    +0.0726 [-0.0602,+0.2122]   0/6 vs 0/6 vs 0/6
Qwen2.5-32B-Instruct            32.8    0.0%    +8.2    -0.0418 [-0.1443,+0.0637]    -0.1464 [-0.2564,-0.0254]*   +0.1046 [-0.0629,+0.2591]   0/6 vs 2/6 vs 0/6
Llama-3.3-70B-Instruct          70.6    0.0%    +3.0    -0.0667 [-0.1789,+0.0383]    -0.1099 [-0.2261,+0.0074]    +0.0432 [-0.1195,+0.2011]   0/6 vs 1/6 vs 0/6
Qwen2.5-72B-Instruct            72.7    0.0%    +4.1    -0.0500 [-0.1956,+0.0906]    -0.1389 [-0.2819,-0.0017]*   +0.0889 [-0.1153,+0.2915]   0/6 vs 1/6 vs 2/6

* = the interval excludes zero.  ! = the arm this is measured on is one the
    paper's own filter calls unreadable, because the model declines more than 5%
    of its rollouts.  ? = the model reads within two points of chance on this arm, so
    its correct and wrong rollouts are nearly the same draw.  Widest cell per model:
  Llama-3.2-1B-Instruct          mink:pooled            +0.2312 [-0.1097,+0.5663]
  Qwen2.5-1.5B-Instruct          mink:pooled            -0.3485 [-0.7503,+0.0124]
  Llama-3.2-3B-Instruct          mink:within key rank   -0.3984 [-0.7750,+0.0840]
  Phi-3.5-mini-instruct          mink:pooled            -0.1031 [-0.3925,+0.1830]
  gemma-3-4b-it                  minkpp:pooled          -0.6347 [-1.2927,-0.0574]
  OLMo-2-1124-7B-Instruct        mink:within key rank   -0.1673 [-0.5774,+0.3053]
  Qwen2.5-7B-Instruct            mink:pooled            +0.1387 [-0.2413,+0.4715]
  Llama-3.1-8B-Instruct          mink:within key rank   +0.1153 [-0.2643,+0.5150]
  phi-4                          minkpp:within key rank +0.1543 [-0.0398,+0.3867]
  Qwen2.5-14B-Instruct           mink:pooled            +0.1306 [-0.1709,+0.4204]
  Qwen2.5-32B-Instruct           minkpp:pooled          -0.0993 [-0.2964,+0.0821]
  Llama-3.3-70B-Instruct         minkpp:within key rank +0.1396 [-0.2511,+0.4852]
  Qwen2.5-72B-Instruct           mink:pooled            -0.2582 [-0.8272,+0.2494]
```

```
13 models, 0 of them on an arm the >5% filter calls unreadable
  released arm clears zero somewhere: 1  (positive on 0, negative on 1)
  clean control clears zero somewhere: 3
  released MINUS control clears zero: 2
    on meta-llama/Llama-3.2-3B-Instruct, Qwen/Qwen2.5-72B-Instruct
  of the 10 models that answer this arm and beat chance on it, the difference clears on 1 (Qwen/Qwen2.5-72B-Instruct)
  released   arm:  1 of 78 cells clear ( 1.3%, 3.9 expected by chance), 0 positive and 1 negative
  control    arm:  4 of 78 cells clear ( 5.1%, 3.9 expected by chance), 0 positive and 4 negative
  difference arm:  3 of 78 cells clear ( 3.8%, 3.9 expected by chance), 2 positive and 1 negative
  over those 10, the stratified log-likelihood difference's interval tops out between +0.135 and +0.315 nats
{"models": 13, "released": 1, "control": 3, "difference": 2, "positive": 0, "negative": 1, "refusing": 0, "which_difference": ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-72B-Instruct"], "informative": 10, "informative_difference": 1, "which_informative": ["Qwen/Qwen2.5-72B-Instruct"], "ceiling_lowest": 0.13486096208610499, "ceiling_highest": 0.3149156422640812, "released_cells": 78, "released_cells_clearing": 1, "released_cells_positive": 0, "released_cells_negative": 1, "control_cells": 78, "control_cells_clearing": 4, "control_cells_positive": 0, "control_cells_negative": 4, "difference_cells": 78, "difference_cells_clearing": 3, "difference_cells_positive": 2, "difference_cells_negative": 1, "expected_by_chance": 3.9000000000000004}
```

The released arm clears on **1 of 78**, the control on 4 and the difference on
**3** — every one at or below the 3.9 the procedure gives by construction. The
two blocks read a different cell and the caption says why: MMLU-Pro's control
carries a key rank on every rollout, so it is read stratified with the geometry
channel removed, while BixBench's is resampled from mixed-type option strings
and carries a rank on 66 of 615, so differencing a 315-rollout arm against a
66-rollout one would be mostly the second arm's noise.

**And the trap, reported because it is the comparison such an attack invites.**
The same statistics separate the released file from our synthetic control at
**AUC 0.968 to 0.992**, and from the placebo at 0.671 to 0.918. Neither is
evidence of memorisation: the non-member sets are *drawn here* rather than
published, so what the attack separates is real option values from synthesised
ones. **Any membership attack on option sets against a synthetic control will
succeed for that reason.** It is why the conclusion rests on the within-file
gap, where no file is compared to any other.

## The frontier at the level that covers, and the claim it costs

The paper corrects its intervals to a measured covering level where the arms
sit on 46 and 60 clusters, and the **frontier** — the table the repair
conclusion rests on — needs it most: its twenty MMLU subjects are worth
**4.8 by Kish**, which is the regime where a percentile cluster bootstrap is
least trustworthy. Correcting everything *except* that table would get the risk
allocation backwards.

`frontier_calibrated.py` measures it the way `bootstrap_calibration.py` does:
beta-binomial fitted by moments to the released arm's per-subject accuracies,
files simulated at the frontier's own subject sizes, coverage read off at every
nominal level. **A nominal 95% interval covers 88.1%; 95% needs a nominal 98.7%.**

| operator | leak | nominal 95% | covering |
|---|---|---|---|
| released | −4.5 | [−11.7,−0.6] | [−14.5,+0.3] |
| key marginal | −2.6 | [−6.5,+4.7] | [−7.5,+6.2] |
| key marginal, near | +1.7 | [−5.4,+5.5] | [−8.3,+6.3] |
| **wrong step** | **+8.6** | **[+3.1,+11.4]** clears | **[−0.9,+12.5]** does not |
| rank uniform | +9.5 | [+6.9,+12.4] | [+5.4,+13.8] |
| exchangeable | +9.5 | [+4.9,+17.7] | [+3.9,+19.6] |
| imitation | +4.3 | [−1.3,+8.5] | [−3.5,+9.5] |

**The claim this costs.** "The one operator that leaves difficulty where it
found it never sees the key and leaks +8.6 anyway" was the paper's strongest
repair claim. At the level that covers, **wrong step's interval covers zero**,
and it is the only row the calibration moves. The claim is withdrawn. What
survives is what the table was built to show: **seven operators and not one in
the corner**, the two that demonstrably close the channel paying +11.1 to +20.4
points of difficulty for it, and the one that leaves difficulty alone not shown
to have closed anything.

**Two bugs, both of which produced a believable number.**

- **Coverage was scored against the simulated file's own mean** rather than the
  rate it was drawn with. The bootstrap is centred on the former by
  construction, so it reported **100% coverage at every level** and a covering
  level of 0.80 — i.e. it claimed the intervals could be made *narrower*. The
  only reason it was caught is that a percentile bootstrap over-covering
  perfectly at 4.8 effective clusters is not a believable result.
- **A second bootstrap is a second interval.** Rolling our own gave a nominal
  interval 0.06 points from the shipped one — same picks, different RNG — and
  then numpy's interpolating `percentile` differed again from
  `learned_probe`'s index convention. Both levels now come off
  `learned_probe.bootstrap`'s own draws with its own indexing, and a test
  asserts the nominal column reproduces the shipped result file exactly.

## λ̂ under the per-rank relaxation, on the draws that reject

`rank_dependent_fit.py` reports what freeing one λ_j per key rank does to the
**geometry** term and not what it does to **λ̂** — the quantity the conclusion
tells the field to publish. And specifically: is the 32B's λ̂ = 0.565 among the
rejected fits? (`lambda_relaxed.py`.)

| model | draws | rejecting | shared λ̂ | relaxed λ̂ | max shift | λ spread across ranks |
|---|---|---|---|---|---|---|
| Qwen2.5-32B | 10 | **6** | 0.567 | 0.560 | 0.012 | 0.265 |
| Qwen2.5-7B | 10 | 4 | 0.258 | 0.254 | 0.009 | 0.206 |
| Llama-3.1-8B | 10 | 2 | 0.165 | 0.165 | 0.003 | 0.107 |
| Llama-3.2-1B | 10 | 0 | 0.002 | 0.011 | 0.011 | 0.070 |

**Yes** — six of the 32B's ten draws reject. **And the relaxation moves λ̂ by at
most 0.012 on any draw**, leaving the ordering across all four models exactly
intact. λ genuinely varies by rank (spread up to 0.265 on the 32B), which is
*why* the shared model rejects; averaging it back gives the same number. The
misspecification is real, diagnosable, and costs the reported quantities 0.012
on λ̂ and ≤0.19 points on the geometry term.

## The temporal test on every file whose dates were recovered

The temporal test is not confined to LitQA2: TableQA has more dated items (243
of 244) over a wider range (1991–2024), so the test runs on all four LAB-Bench
files that date: **704 dated items**, Llama-3.1-8B,
the one model whose vendor states a cutoff (`run_temporal_rest.sh`).

**It does not replicate in sign.** Before December 2023 against after, the
benchmark's own condition reads:

| file | dated items | before − after | date vs margin, no-data arm |
|---|---|---|---|
| LitQA2 | 198 | +6.8 [−2.5,+16.5] | −0.148 [−0.288,−0.001] **clears** |
| TableQA | 243 | −2.9 [−17.1,+9.2] | +0.157 [+0.028,+0.284] **clears** |
| SuppQA | 82 | −11.3 [−24.2,+1.0] | −0.069 [−0.287,+0.160] |
| FigQA | 181 | no split: every item predates the cutoff | +0.000 [−0.134,+0.182] |

**The no-data arm's date correlation clears zero on two of the four, in opposite
directions.** Part of the reason is that these files barely straddle any cutoff:
207 of TableQA's 243 dated items predate it, and all 181 of FigQA's do. That is
what the principled test looks like when it is finally run — runnable, and at
these sizes unable to resolve its own sign.

**And a result the rest of the paper needs.** The no-data arm reads between
**−1.0 and +3.6 points over chance on all four files** and across four solvers.
So LAB-Bench's literature subtasks do *not* hand a no-data solver what
BixBench's numeric items do: **the open channel is a property of distractors
that are perturbations of a numeric value, not of multiple choice.**

## The template cell on v1.0, the release the published baselines ran on

Both zero-shot runs BixBench published are on v1.0, whose key-rank law caps a
rank rule at **+5.8** points where v1.5's caps it at +26.0. What those baselines
measured is decided by running the grid's template cell there.
`bixbench_v10_items.py` writes all **296** v1.0 items in **53** capsules from
the published gpt-4o run file (options and key as published, question withheld);
the control is drawn from v1.0's own pool of 961 option strings; `run_bixv10.sh`
runs the thirteen models; `bixbench_v10_grid.py` reads every row at **96.0%**,
the widest level any of the thirteen dumps needs to cover 95% on these capsules
(`bootstrap_calibration`'s beta-binomial simulation, per model). Margins over
the control, at that level:

| model | B | all 296 | numeric 159 | other 137 |
|---|---|---|---|---|
| Llama-3.2-1B | 1.2 | +1.0 [−3.9,+5.8] | −0.4 [−7.2,+5.8] | +2.7 [−4.3,+9.9] |
| Qwen2.5-1.5B | 1.5 | **+8.0 [+2.5,+13.3]** | +10.3 [+3.1,+17.1] | +5.4 [−2.4,+13.1] |
| Llama-3.2-3B | 3.2 | **+9.5 [+2.9,+15.5]** | +2.1 [−5.1,+9.8] | +18.0 [+9.4,+26.9] |
| Phi-3.5-mini | 3.8 | **+10.5 [+5.0,+16.2]** | +8.0 [+0.8,+14.9] | +13.4 [+4.8,+22.0] |
| gemma-3-4b | 4.3 | **+6.6 [+1.3,+11.8]** | +4.6 [−2.1,+11.1] | +9.0 [+0.9,+17.5] |
| OLMo-2-1124-7B | 7.3 | **+5.3 [+0.6,+10.0]** | +2.7 [−3.5,+9.1] | +8.3 [+1.9,+14.5] |
| Qwen2.5-7B | 7.6 | **+6.1 [+1.6,+10.9]** | +4.4 [−1.8,+10.9] | +8.0 [+0.4,+15.9] |
| Meta-Llama-3.1-8B | 8.0 | **+6.4 [+1.4,+11.4]** | +1.0 [−6.2,+9.0] | +12.7 [+5.6,+19.4] |
| phi-4 | 14.7 | **+10.6 [+4.4,+17.3]** | +8.0 [−0.0,+15.4] | +13.6 [+3.6,+23.7] |
| Qwen2.5-14B | 14.8 | **+6.1 [+1.2,+11.2]** | +5.0 [−1.7,+11.5] | +7.3 [−0.2,+13.9] |
| Qwen2.5-32B | 32.8 | **+7.8 [+2.0,+14.0]** | +6.7 [+0.3,+13.4] | +9.0 [−0.3,+18.4] |
| Llama-3.3-70B | 70.6 | **+11.7 [+6.7,+16.9]** | +7.1 [+0.1,+13.8] | +17.0 [+9.7,+24.0] |
| Qwen2.5-72B | 72.7 | **+10.8 [+5.3,+16.3]** | +5.9 [−0.5,+12.0] | +16.5 [+7.6,+25.3] |

**Twelve of thirteen clear** (five do on v1.5), eleven against chance, and no
rollout is unparsed. The half that carries it is not the one the rank ceiling
describes: the numeric items clear on four, the items whose options are not all
numbers on nine. The ceiling bounds one channel of one half, so "v1.0's channel
is a fifth as wide" is true of the rank rule and not of what the template reads.
The dumps ship as `results/agentic_dumps/*_bixv10_bixprompt_argmax.jsonl.gz`,
and `tests/test_withdata.py` re-derives every cell from them.

## `wrong step` on BixBench: the prescription needs a writer who can make the mistake

`wrong step` — a writer shown the question alone and asked for five values a
plausible mistake would reach — is the frontier row that keeps MMLU's items
hard, so it belongs on the workshop's own file too. The same writer
(Qwen2.5-14B, and Llama-3.1-8B as a second generator) run on BixBench's 105
numeric items (`keyblind_operators.py --jsonl build/bixbench_numeric_q.jsonl`),
gathered by `bixbench_wrong_step.py`:

| arm | key rank (smallest→largest) | set leak over clean worst case | difficulty given the question, at 95.75% |
|---|---|---|---|
| released | 12.4 / 51.4 / 29.5 / 6.7 | +12.4 | — |
| wrong step (Qwen2.5-14B) | 33.3 / 19.0 / 21.0 / 26.7 | **+17.1 [+8.4,+26.3]** | **+17.4 to +18.3**, each clearing |
| wrong step (Llama-3.1-8B) | 36.0 / 14.6 / 15.7 / 33.7 | +33.4 | +23.9 to +29.1 |
| repaired (rank uniform) | 23.8 / 29.5 / 27.6 / 19.0 | −1.0 [−9.7,+8.4] | −6.7 to +1.4 |

On MMLU the writer can work the item, so its mistakes sit around the key; here
the answer is a property of data it is not shown. It produced the key on **6
of 105** items against MMLU's 114 of 635, and its values *miss* the key: the
key is at an extreme of its set **60.0%** of the time (69.7% under Llama)
against the released file's 19.0%. Both columns move the wrong way at once. The
difficulty level, 95.75%, is `frontier_calibrated.covering_level` on the
released arm's 46 capsules (29.6 effective); every nominal interval is
reproduced from the per-item answers before the covering one is read.

## Two 70B-class solvers on the frontier's difficulty axis

The difficulty column is three solvers of 7 to 14B, which leaves open whether
the repair cost is a small-solver artefact. `frontier_validity.py` run with
Qwen2.5-72B and Llama-3.3-70B on the same 464 items, and
`frontier_calibrated.py --large` reads them at the frontier's covering level,
98.7%:

| arm | Qwen2.5-72B | Llama-3.3-70B |
|---|---|---|
| key marginal | **+13.0 [+6.0,+18.9]** | **+20.1 [+6.1,+26.2]** |
| key marginal, near | **+10.5 [+3.1,+17.1]** | **+15.4 [+3.5,+21.2]** |
| wrong step | −1.4 [−7.1,+10.4] | +1.4 [−6.1,+6.1] |

The cost of closing the channel holds at 70B, and `wrong step` stays at the
released file's difficulty. They are reported beside the table's three rather
than folded into its union, which was fixed over those three before any of
this was run.

## BixBench's own agent, with the data and without it

Every other experiment in the paper withholds the data, which leaves a question
open: does what the option set does reach the scores an
agent gets *with* the data, under the scaffold BixBench published? So the
published agent was run, twice per item, on all 205 v1.5 items:

- **BixBench's, verbatim** (`sources/bixbench_49311180/`, see `SOURCES.md`
  there): data-analysis-crow v1.5.0's system prompt and open-answer task
  template, formatted as `generate_trajectories.py` formats them; its three
  tools (`edit_cell`, `list_workdir`, `submit_answer`); at most 40 steps at
  temperature 1.0 with old notebook states hidden; the capsule laid out as the
  benchmark's own extraction lays it out (Data folder flattened, reference
  notebook removed). Every cell runs in fhda's pinned notebook image, rebuilt
  for amd64 from its own recipe (`sandbox_image/`), with no network and the
  capsule mounted read-only (`sandbox_repl.py`).
- **Ours, and why:** three open families as agents (Qwen2.5-72B,
  Llama-3.3-70B, gemma-3-27b, FP8 under vLLM 0.10.2), each grading and reading
  its own runs as the published runs were; tools called in text, because the
  three families' function-calling formats differ; the notebook view budgeted
  to each model's context; the *no-data* condition is the same agent with an
  empty working directory (`bixbench_agent.py`).
- **Read the two ways BixBench publishes** (`bixbench_withdata.py`): the v1.5
  graders, plus two model-free grades on the 105 numeric items (exact at the
  key's precision, and the free-response table's 5% rule); and `MCQ_EVAL_PROMPT`, forced and
  with the "insufficient information" option, upstream formatting and upstream
  parse (`<answer> B </answer>` is no pick), runs with no answer scored wrong
  unread, two letter orderings per option set.

**The swap that makes it causal.** The agent never sees the options, so on the
105 numeric items each finished run is read through three option sets sharing
the key: the released one, a placebo that redraws every distractor holding the
key's rank, and the rank-uniform repair. Same notebook, same answer; a
difference is the option set's. Intervals are paired over the 46 capsules at
96.0%, the level that covers them.

Own-family readers; per cent; the last column paired over 46 capsules at
96.0%, bold where it clears.

| agent | data | open | within 5% | MCQ forced | may decline (declined) | released | placebo | repaired | released − repaired |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-72B | with | 18.1 | 14.3 | 27.6 | 20.0 (47%) | 19.5 | 22.9 | 30.0 | **−10.5 [−16.4,−5.2]** |
|  | without | 6.8 | 0.9 | 17.8 | 8.1 (67%) | 16.7 | 18.1 | 25.7 | **−9.1 [−17.6,−1.0]** |
| Llama-3.3-70B | with | 6.3 | 4.8 | 25.9 | 8.1 (70%) | 26.2 | 29.5 | 35.7 | **−9.5 [−15.7,−3.1]** |
|  | without | 9.8 | 4.8 | 27.1 | 11.0 (72%) | 24.8 | 30.0 | 31.9 | **−7.1 [−15.0,−0.4]** |
| gemma-3-27b | with | 4.4 | 3.8 | 21.7 | 9.0 (62%) | 21.9 | 25.2 | 25.2 | −3.3 [−7.1,+0.0] |
|  | without | 5.8 | 3.8 | 23.4 | 8.3 (60%) | 22.4 | 24.3 | 25.2 | −2.9 [−8.5,+2.5] |

**What the data is worth.** One family of the three uses it. Given the
capsule, Qwen2.5-72B's submitted number is within 5% of the key on **14.3%** of
the numeric items against 0.9% without it, **+13.3 [+5.6,+23.5]** paired over
capsules, and BixBench's graders credit it +11.2 [+1.5,+22.0] over all 205.
Llama-3.3-70B and gemma-3-27b score the same with the data as without (−3.4
[−8.7,+1.3] and −1.5 [−4.8,+1.9] open-ended), although both load the capsule's
files on most runs. And the published multiple-choice reading credits even
Qwen's data with only **+2.9 [−5.8,+12.5]** on the numeric items.

**What the options are worth, with the runs held fixed.** For every family, in
both conditions, **the released options give the same runs their lowest score**.
With the data the repair reads **+3.3 to +10.5** points higher, clearing for
Qwen2.5-72B (−10.5 [−16.4,−5.2]) and Llama-3.3-70B (−9.5 [−15.7,−3.1]); the
placebo, which holds the key's rank, reads +3.3 higher for all three and clears
for Qwen and gemma. **A second reader agrees**: gemma-3-27b reading Llama's runs
gives −9.1 [−14.3,−4.0], and reading Qwen's runs −7.6 [−12.2,−3.5]. So under
this protocol, in the published reading, which options are printed moves the same
runs more than giving the agent the data does; under the published agent (next
section) that holds for Qwen2.5-72B and not for the agents that use the data.

**Mostly not the rank.** A reader that picked each rank with the probability it
picks it on the released file, whatever the file, would put the repair −5.8 to
+1.0 points off the released file. On the released file every reader scores
*below* that prediction — by 3.3 to 8.7 points with the data, 0.3 to 4.7
without — and through the repair within 1.4 points of it or above. BixBench's
distractors are perturbations of the true value, and the data moves an agent's
misses toward the value: Qwen's wrong answers are read as the key on **11.0%**
of reads through the released options with the data against 18.1% without, and
on 27.2% against 28.6% through the repair. A near miss is read as the
distractor beside it — which is what a distractor is for.

**Declining.** Offered the "insufficient information" option the readers take
it on 47 to 72% of runs and the swap moves nothing (with the data −0.5
[−1.7,+0.0], −0.9 [−4.5,+3.2], +0.0 [−3.2,+3.1]). The readings of one run
disagree with the data as BixBench's three published arms do without it: the
forced choice reads 21.7 to 27.6%, declining or answering in words 4.4 to 20.0%.
None of these intervals is one of the paper's six, which were fixed first.

**What went wrong on the way, and what it cost.**

- **A wall clock started at queue time killed 303 of Qwen's 410 episodes at step
  0** once the run passed an hour: every job is gathered at once behind a
  semaphore, and the timer started before the semaphore. It now starts when the
  episode gets its sandbox (`tests/test_withdata.py` checks the order), and the
  303 were deleted and run again.
- **Pinning episodes to replicas by hash starved one replica**: the slower one's
  episodes lived longer and filled the slots. Assignment is now least-loaded.
- **A GPU fault killed one of Qwen's two replicas mid-run** (`CUDA error:
  unknown error`; the card then reported "requires reset"), and least-loaded
  assignment steered new episodes *to* it, because its episodes failed fast and
  it always looked idle: 66 episodes failed. They were run again from
  the start on the other replica, as were five that the resulting load
  pushed past the one-hour limit; a replica whose last call failed now takes no
  new episodes and an episode fails over after two failures
  (`tests/test_withdata.py`). Every run shipped is complete under one protocol:
  `results/agent_runs/reruns.json` lists each episode run again and why, the
  replaced runs ship as `superseded_qwen72b.jsonl.gz`, and `validate_artifact.py`
  checks there are 1,230 runs and that every listed episode ships as its re-run.
- **Identical prompts sent at once got different greedy replies.** Two runs can
  hand the reader the same prompt (an empty notebook and the same answer); vLLM's
  batch nondeterminism answered 37 of Llama's 57 such pairs differently, six with
  different letters, and a rerun from the cache then gave both the one reply
  kept, moving a number by one read. Identical prompts in flight now share one
  request, and every number printed is the cache rerun's.
- **Scoring beside a KV-full agent run thrashes it**: Qwen fell from five
  episodes a minute to under two. Readers ran after the agents, or beside them only
  where the server had cache to spare.

**Reproduce.** With no GPU, from the shipped rows:

    python3 bixbench_withdata.py --summarise-only \
        --rows results/agent_runs/withdata_rows_qwen72b.json.gz \
        --rows results/agent_runs/withdata_rows_llama70b.json.gz \
        --rows results/agent_runs/withdata_rows_gemma27b.json.gz

or `pytest tests/test_withdata.py`, which re-derives
`results/bixbench_withdata.json` from `results/agent_runs/withdata_rows_*.json.gz`.
With a server, `bixbench_withdata.py --models M --reader M=URL --judge M=URL`
re-scores from `results/agent_runs/trajectories_M.jsonl.gz` and hits the shipped
reader cache for every call.

## BixBench's own agent under its published protocol

The first with-data runs called fhda's tools in text, one call a step. The agent
BixBench's `generate_trajectories.yaml` names is ldp's `ReActAgent` with its
default `single_prompt=False`, and `bixbench_agent.py --protocol react` now runs
exactly that, at the release BixBench runs it with:

- **the release is pinned, not assumed.** BixBench's `pyproject.toml` lists `ldp`
  without a version, but fhda v1.5.0's pins `ldp==0.26.0` and
  `fhaviary[server]==0.19.0`, and BixBench's `uv.lock` at the pinned commit
  resolves ldp 0.26.0, fhaviary 0.19.0 and pydantic 2.10.1. Those files are
  vendored beside the code they pin, and the tool schemas are built with those
  versions.
- **each step is two calls.** A reasoning call with fhda's tools in the prompt but
  `tool_choice="none"`, stopped at `Observation:`/`Action:`; the reasoning put
  back as ldp 0.26.0 puts it, `Thought: {reasoning}. Based on this reasoning,
  let's select the appropriate tool!\nAction: `; a `"Continue..."` user turn;
  then a call that must pick a tool. Tool replies come back prefixed
  `Observation: `; every notebook state but the latest is ldp's
  `HiddenEnvStateMessage`; a step that cannot be read is tried again, five times at
  most, as ldp's retry on `MalformedMessageError` does.
- **the first observation is fhda's** `DataAnalysisEnv.reset`: the task, the (empty)
  notebook, then the system prompt, after ldp's own ReAct prompt.
- **the tools are fhda's,** schemas as aviary 0.19.0's `Tool.from_function` builds
  them from fhda v1.5.0's signatures and docstrings; replies, notebook view and its
  3,000-character output limit are fhda's; the kernel defines no tools of its own.

`tests/test_published_protocol.py` checks each of these against the vendored
upstream files (`sources/bixbench_49311180/`: ldp v0.26.0, data-analysis-crow
v1.5.0 and BixBench's own pins), parsed as text and never run.

**What vLLM does differently, and what is done about it.** The published agents'
APIs make a `tool_choice="required"` reply a tool call in the model's own format,
and never return a call from a `tool_choice="none"` one. vLLM 0.10.2 does
neither:

- its `"required"` forces a generic JSON array (`[{"name":..., "parameters":...}]`)
  token by token, a format these models were not trained to call tools in: in a
  smoke run GLM-4.5-Air, having written "let me start by exploring the working
  directory", was made to call `submit_answer(null)` at step 0 in two of three
  episodes; its `"auto"` lets the model answer in ReAct's text format instead
  ("Action: list_workdir()" and an invented observation);
- its `"none"` leaves the tools in the prompt and lets the model write its own call
  tag, which comes back as text: in the 226 episodes run before the fix,
  GLM-4.5-Air did in 1,402 of 1,432 reasoning turns, Qwen2.5-72B in 574 of 2,102.

So the selection call opens the model's reply with its own tool-call tag
(`continue_final_message`) and stops at the closing tag, and the reasoning call
bans that same opening (`bad_words`): the model reasons in text, then must call a
tool, chooses which and with what in the format it was trained on, and makes
exactly one call.

**The reply limit is the published one, and what it cuts.** ldp 0.26.0 makes
every call through fhlmi's `LiteLLMModel`, and fhlmi 0.25.2 (BixBench's lock)
gives a config that sets no reply limit `max_tokens=4096`; BixBench's config sets
none (`fhlmi_llms_v0.25.2.py`, vendored, its wheel's sha256 checked against the
lock; `test_a_reply_may_run_to_the_published_agents_limit`). The driver's limit is
the same 4,096. A reasoning turn that reaches it ends there, is put back in the
wrapper as it stands, and the step goes on. How often, over every reasoning turn
of the 410 episodes (`bixbench_withdata.reply_limit_census`, recounted by the
validator from the shipped trajectories):

| agent | reasoning turns | at the limit | of those, the wrapper loop | step-limit runs with one |
|---|---|---|---|---|
| GLM-4.5-Air | 5,169 | 126 (2.4%) | 13 | 1 of 1 |
| Qwen2.5-72B | 6,576 | 183 (2.8%) | **183** | **63 of 83** |
| Llama-3.3-70B | 3,267 | 0 | 0 | 0 of 1 |
| Qwen3-30B-A3B | 6,884 | 2,276 (33.1%) | **2,090** | **98 of 108** |

The loop is one thing: the reasoning repeats the wrapper's closing sentence,
"Based on this reasoning, let's select the appropriate tool!", line after line
and never writes the "Action:" that would stop it. The wrapper, put back in the
history as the model's own turn, is copied into the next reasoning turn, and once
a turn has looped most of the episode's later turns loop too (Qwen3: 85% of them
at 285 episodes). Qwen3-30B-A3B submits on 300 of its 410 runs, Qwen2.5-72B on
316. This is the published agent doing what it does with these models, so it is
left as it is; ldp 0.37.0 stopped putting the reasoning back in a wrapper.

**Three sets of runs were discarded, and are kept.** `build/agent_runs_discarded_*`
holds the runs made with vLLM's `"required"` and with `"auto"`, and 226 episodes
(GLM-4.5-Air 76, Qwen2.5-72B 150) made before the release pin was checked: they
put the reasoning back bare, as ldp 0.37.0 and later do, instead of in 0.26.0's
wrapper, and did not ban the call tag in the reasoning turn. None is used; every
published-protocol run in the paper was made after both fixes, from the start.

**Where it ran.** Agents: GLM-4.5-Air-FP8 on two H100s and Qwen2.5-72B (FP8, the
same server settings as its text-protocol runs) on two more; Llama-3.3-70B and
Qwen3-30B-A3B-Instruct-2507, FP8 weights, on two A100s each (Qwen3 on two
replicas; Llama's last episode on two H100s, which `served_as` records); vLLM
0.10.2 throughout. Sandboxes: a rootless Docker daemon on the GPU host, the same
image; `served_from` records the server each episode used. Episodes our hour limit
cut on shared GPUs (26 of 1,640: Qwen3 15, Qwen2.5 10, GLM 1) were set aside,
listed in `results/agent_runs/reruns_<run>.json`, run again from the start with the
limit raised, and ship as `superseded_<run>.jsonl.gz`; no run ships cut.

**The runs registered before they were made** (`PREREGISTRATION.md`, D2). The
text protocol and the first runs' settings, `--out build/agent_runs_replication`:
Qwen3-235B-A22B-Instruct-2507-FP8 on four H100s (`--rollouts 2`, one tensor-parallel
replica, 32,768 context), and `--first-rollout 1 --rollouts 2` of Qwen2.5-72B,
Llama-3.3-70B and gemma-3-27b (FP8 weights; Llama on two A100s; gemma on four A100s,
one replica each, as two drivers split by rollout; Qwen2.5-72B on two A100s for
237 episodes and four H100s for the other 583). vLLM 0.10.2 throughout; `served_as` and
`served_from` record each episode's server. Qwen3-235B-A22B ran under the first runs'
3,600 s wall clock and the new seeds under 10,800 s; the 12 episodes the clock cut
(Qwen3 7, with the data, on a server running 28 episodes at once; Qwen2.5-72B 5, four
of them on its A100 replica) were set aside, listed in
`results/agent_runs_replication/reruns_<run>.json`, run again from the start, and ship
as `superseded_<run>.jsonl.gz`. Episodes in flight when a driver was stopped to move
it to other GPUs wrote nothing and were run from the start by the next driver. 54
requests dropped by the tunnel to the A100 servers ("Server disconnected") were
retried on a new seed, as the driver retries any failed draw; each episode lists its
own under `server_failures`. `python3 replication.py pack` writes the answers the
analysis reads; `analyse` re-derives Table 21's v1.5 rows from them.

**What it found.** Each run set read by its own family (†: by gemma-3-27b, see
below); per cent; the last column paired over 46 capsules at 96.0%. The
text-protocol rows are the table above.

| agent (published protocol) | data | open | within 5% | MCQ forced | may decline (declined) | released | placebo | repaired | released − repaired |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-72B | with | 10.7 | 9.5 | 19.3 | 12.4 (56) | 16.2 | 19.5 | 24.8 | -8.6 [-15.1,-1.8] |
|  | without | 5.4 | 2.9 | 19.5 | 6.8 (73) | 17.1 | 17.1 | 17.1 | +0.0 [-6.5,+6.2] |
| Llama-3.3-70B | with | 9.8 | 5.7 | 30.5 | 6.6 (87) | 29.5 | 34.8 | 33.8 | -4.3 [-10.9,+1.8] |
|  | without | 4.9 | 1.9 | 21.9 | 0.5 (94) | 21.4 | 22.4 | 26.7 | -5.2 [-10.3,-0.4] |
| GLM-4.5-Air | with | 27.3 | 16.2 | 42.9 | 28.3 (55) | 33.8 | 38.6 | 40.5 | -6.7 [-13.9,+0.6] |
|  | without | 6.3 | 4.8 | 24.1 | 11.2 (63) | 24.8 | 26.7 | 30.0 | -5.2 [-10.9,+0.9] |
| Qwen3-30B-A3B† | with | 18.1 | 14.3 | 32.7 | 21.7 (50) | 23.8 | 28.1 | 27.6 | -3.8 [-8.2,+0.0] |
|  | without | 3.4 | 1.9 | 14.9 | 7.8 (61) | 13.3 | 17.1 | 17.6 | -4.3 [-9.6,+0.5] |

The same runs, with the data, read three ways (released − repaired, 105 numeric
items): the own family, gemma-3-27b, and a rule that reads no notebook and takes
the option nearest the agent's number (`bixbench_withdata.nearest_is_key`, added
after the readers' results were in).

| protocol | agent | own family | gemma-3-27b | nearest option |
|---|---|---|---|---|
| text | Qwen2.5-72B | -10.5 [-16.4,-5.2] | -7.6 [-12.2,-3.5] | -2.9 [-8.6,+3.0] |
| text | Llama-3.3-70B | -9.5 [-15.7,-3.1] | -9.1 [-14.3,-4.0] | -13.3 [-21.4,-5.5] |
| text | gemma-3-27b | -3.3 [-7.1,+0.0] | --- | -10.5 [-17.6,-3.6] |
| published | Qwen2.5-72B | -8.6 [-15.1,-1.8] | -4.3 [-7.7,-1.0] | -4.8 [-9.4,-0.9] |
| published | Llama-3.3-70B | -4.3 [-10.9,+1.8] | -6.2 [-12.3,-0.4] | -9.5 [-16.7,-3.0] |
| published | GLM-4.5-Air | -6.7 [-13.9,+0.6] | -4.3 [-9.6,+0.8] | -4.8 [-10.2,+0.9] |
| published | Qwen3-30B-A3B† | +0.5 [-4.2,+4.8] | -3.8 [-8.2,+0.0] | -3.8 [-9.6,+1.6] |

- **The option set's effect travels.** With the data, read by the table's
  reader, the released options give the same runs their lowest score for all
  seven run sets under both protocols; without it for six, the seventh
  (Qwen2.5-72B as published) a three-way tie. As published the repair reads +3.8
  to +8.6 higher, clearing for Qwen2.5-72B. One model's contrast does not differ
  measurably between the protocols: published minus text +1.9 [−5.3,+9.5] for
  Qwen2.5-72B, +5.2 [−4.4,+13.8] for Llama-3.3-70B.
- **Every other reading agrees on the direction.** gemma-3-27b puts the released
  options below the repair for every run set in both conditions (clearing with
  the data for Qwen2.5-72B and Llama-3.3-70B under both protocols), Qwen2.5-72B
  reading GLM-4.5-Air's runs does too, and so does the nearest-option rule, in all
  14 run-set × condition cells: +2.9 to +13.3 with the data, +3.8 to +12.4
  without. That rule needs no model: BixBench's perturbation distractors sit
  nearer the key than generated ones, so an answer near the key is often nearer
  one of them.
- **The data's worth does not travel.** GLM-4.5-Air and Qwen3-30B-A3B, run only
  as published, use the data most: the graders credit it +21.0 [+10.9,+31.5] and
  +14.6 [+6.5,+23.3]. In the MCQ reading the data is worth −0.5 to +2.9 under the
  text protocol and −0.9 to +10.5 as published, clearing for Llama-3.3-70B (+8.1
  [+0.5,+16.5]) and Qwen3-30B-A3B (+10.5 [+2.3,+20.0]). For GLM-4.5-Air,
  Llama-3.3-70B and Qwen3-30B-A3B the data moves the published reading more than
  the swap; for Qwen2.5-72B less. So "the swap moves the score more than the
  data does" holds for the text protocol and not for every agent: the paper says
  the swap moves the reading as much as the data does,
  and that with the data the released options give the same runs their lowest
  score in every case.
- **The published agent costs Qwen2.5-72B.** Its open answers move −7.3
  [−12.8,−2.3] against the text protocol, with the data, and it submits on 140 of
  205 runs against 163, most of the rest ending at the step limit inside the
  wrapper's loop. Llama-3.3-70B submits on every run under both.
- **Not the rank, with any working reader.** With the data every reader but a
  failed one reads the released file lower against its own rank-preference
  prediction than it reads the repair (13 of 13); without it, 12 of 13, the
  exception the tie above.
- **Declining.** Readers decline on 47–94% of runs; the refusal-mode contrast is
  within a point of zero for every run set with the data but GLM-4.5-Air's, −3.3
  [−6.8,−0.4].

**Three readers that did not work, and what was done.** BixBench's reader call
sets no reply length; ours allows 1,536 tokens.

- *GLM-4.5-Air* re-derives the notebook's analysis before it answers: at 1,536 it
  was cut off on 44% of its replies (a first pass, set aside as `*.cap1536.*` and
  not shipped), reading "no option effect" that was the cut. It reads with 4,096,
  cut off on 46 of 3,257 calls on its own runs.
- *GLM-4.5-Air as a common reader of the other run sets* was tried and dropped: at
  greedy decoding it loops when an agent's answer matches no option ("I will
  select the closest option, which is B, but this is not correct...") to the
  4,096 limit, on 707 of 2,518 calls on Qwen2.5-72B's published runs, and gave no
  pick on 48–58% of its numeric reads there in every option set (5% on its own
  runs). Its passes are not used. gemma-3-27b, the text runs' second reader (no
  pick on 0–3%), reads every run set instead, at the notebook budget it read the
  text runs with.
- *Qwen3-30B-A3B reading its own runs* is cut off on 1,043 of its 2,327 calls at
  1,536 and gives no pick on 51.7% (with the data) and 63.3% (without) of its
  forced reads; a re-read at 4,096 was stopped when 57 of its first 157 replies
  still ran to the limit. The paper's tables read its runs by gemma-3-27b
  (`PRIMARY_READER`) and print its own reading beside it, marked.
- *Qwen2.5-72B*, at 1,536, is cut off on 211 of 2,725 calls on its text runs, 127
  of 2,518 on its published runs and 433 of 3,257 on GLM-4.5-Air's, evenly across
  the option sets (no pick 18.6/17.1/17.1% on GLM's with the data), so a cut-off
  there shrinks a contrast and cannot make one.

**Reproduce.** With no GPU, from the shipped rows (`--summarise-only` over all
seven `withdata_rows_*.json.gz`), or `pytest tests/test_withdata.py`, which
re-derives `results/bixbench_withdata.json`; `validate_artifact.py` checks both
tables and every number the prose quotes, and recounts the reply-limit loop from
the shipped trajectories.

## Other results

- Confounding: pooled gap −18.6 points (p = 0.011); −9.8 (p = 0.302) restricted
  to numeric items; −2.6 (p = 0.821) restricted to LLM-graded items. Second
  model shows +0.5 points (p = 0.96).
- Power: measured σ = 0.230 across capsule-level means; 86 / 44 / 11 pairs
  required at ρ = 0 / 0.5 / 0.9 for a ten-point gap at 80% power.
- **651 tests** pass (`python -m pytest -q tests/`): audit, date semantics,
  eligibility, power, leakage, census, surface rules, repair, the decomposition
  and its geometry lemma, the probe and its controls, the survey and its
  estimator failures, the MMLU validation, repair reachability, the tolerance's
  number reader, the published protocol's single driver, and the release's
  checksums, restore and export scan. JSON Schema and all 59 manifest records
  validate with jsonschema 4.26.0. Under `-W error::ResourceWarning` they still
  pass, with 34 input files left for the garbage collector to close.

## PDF

- Official NeurIPS 2026 style, workshop mode with the `final` option, unmodified geometry.
- **8 main-text pages** against a nine-page limit, in six sections
  (Introduction, Related work, Methods, Results, Discussion, Conclusion) with
  four figures and three tables: the main text and the acknowledgments end on
  page 8, the references run from page 9 to page 12, and three appendices (A,
  methods in detail; B, proofs; C, results in detail) run from page 13 to page 33,
  33 pages in total, with five figures and twenty-six tables. Every figure and
  table carries a caption and is cited in the running text outside its own float,
  and each of the seven in the main text is cited in the main text's prose.
- No undefined references or citations, no missing glyphs, no overfull boxes;
  all fonts embedded and subset, none of them Type 3.
- PDF metadata names the title and the author (`pdftitle`, `pdfauthor`), as the camera-ready should.
- Non-blocking toolchain warnings in `build/main.log` (tectonic's XeTeX engine):
  `cmap` exits ("pdftex not detected"), `inputenc` is ignored, and 28 underfull
  boxes, 18 of them in Table 20's narrow columns and four at page breaks. No
  overfull boxes. Full logs retained in `build/`.
- Each appendix's floats print before the next appendix's heading, and those of
  Sections C.2 to C.4 before the next section's: a `\FloatBarrier` (`placeins`)
  precedes Appendices B and C and Sections C.3 to C.5. Appendix A's four short
  sections share pages 13 to 17 with their floats. The proximity weight is a plain
  λ, with no accent to extract as a stray glyph.

## Scope limits

The one agent given data access is BixBench's own, run with and without each
capsule's data on five open-weight families, in text and as published ("BixBench's
own agent, with the data and without it" and "... under its published protocol"
above), and as published, with the data only, by three current agents on v1.5's
105 numeric questions: gpt-5.1 twice on each, gpt-6-luna and DeepSeek-V4-Pro once
(Kimi-K3 was run and left out). Their within-5% shares describe these runs on
these questions; no capability claim about any deployed system is made or
implied. No candidate gold answer was validated, no human difficulty
audit conducted, and no authoring time measured.
In the long record below the current paper, cutoff eligibility is claimed only
where a vendor states a cutoff in its own model card (Llama-3.1 and 3.2), and every other temporal reading is the
cutoff-free trend. The published-run analysis
re-uses runs from two 2024-era models and adds inference; it is not a new agent
experiment. The long record's 11 open-weight probe models were run only in the
no-data condition except in its temporal test, where four of them are also run
with the question given.
The paper was submitted to AgenticLS @ NeurIPS 2026 and accepted as a poster
(OpenReview #207, decision 30 September 2026).

The guarantee's own scope, stated rather than implied: uniformity of the key's
rank closes a channel against any solver whose preference is a distribution over
*ranks*. It says nothing about a solver reading something that is not a rank —
relative magnitude, relative length, whether an option is an integer — and on
the repaired files a learner given those does **better** than on the released
ones (+13.4 points over its clean control at four options, +17.0 at ten). That
is measured here, not conceded: `learned_probe.py`'s third column exists to
report it, and `exchangeable_repair.py` and `key_identity.py` then establish that
the mechanism is not the one that column's heaviest weight suggests. Removing the
option geometry entirely leaves the margin where it was (+11.8 → +10.9) and moves
it into the key's handwriting. The stronger reading of that observation — that a
*generated* distractor is identifiable as generated — did not survive its own
controlled test (corrected p = 0.18 and 0.19, neither arm clearing its clean
control), and the long record states it as a null. What the operator sweep establishes
instead needs no detection: hold the items and the key fixed, vary the writer,
and the margin moves with the writer.
