# Forced Choice Is Not a Tolerance

*What multiple-choice grading accepts on BixBench's numeric questions.* Vijayavallabh Jayamanikandan, Indian
Institute of Technology Madras. Poster at **AgenticLS @ NeurIPS 2026** (non-archival); to cite it, see
`CITATION.cff`.

## Names in the paper and in the code

The paper uses standard terms; the scripts, result files and the rest of this
README keep the names they were written with.

| Paper | Code, result files and this README |
|---|---|
| MCQ grader; grading, graded correct, accepted | reader; reading, read as (names) the key, credited |
| forced / with a refusal option | forced / may decline (`decline`) |
| nearest-option rule | the rule (`nearest`) |
| released options (R) | released (`q`) |
| rank-preserving rewrite (P) | placebo |
| rank-uniform rewrite (U) | repair (`repaired`) |
| U − P, P − R, U − R | repaired − placebo, placebo − released, repaired − released |
| moved, unchanged and inward keys | moved to an edge, kept, moved inward |
| extreme key | key at an edge |
| configuration | run set |
| each configuration's own model as MCQ grader | family reader (`own`) |
| grader without the question | question-blind reader |
| control file | clean control |
| distractor generator | writer, arm |
| pre-specified test, hypotheses fixed before analysis | registered replication (`PREREGISTRATION.md`, `replication.py`) |
| seed 2, seed 3; Qwen3-235B-A22B run 1, run 2 | `r1`, `r2`; `r0`, `r1` |
| digit-matched rewrites (P′, U′) | `placebo_digits`, `repaired_digits` (`digit_matched.py`) |
| code-free grader (answer and options, no notebook) | `codefree` (`grading_variants.py`) |
| letter-only grader (notebook, reply constrained to a letter) | `letter` |
| the "within 5%" option | `withintol` (`NONE_WITHIN`) |
| proximity weight λ (Remark 1) | `lambda` (`proximity_weight.py`, `lambda_holdout.py`) |
| correction for guessing (formula scoring) | `corrected` (`formula_scoring.py`) |
| other designs; distractors from agents' errors | `k8-middle-s0.2` etc.; `errors-*` (`option_design.py`) |

## The paper as it stands

Every number below is printed by the script named beside it; those the paper
states are pinned to `main.tex` by `validate_artifact.py` (run it with
`AGENTICLS_VALIDATE_COLLECT=1` to list every failing pin at once).

- **What a forced reading credits** (`score_decomposition.py`, Tables 1 and
  25). As released, BixBench's forced reading credits 22.6 and 25.3% of the
  misses it reads on gpt-4o's and Claude 3.5 Sonnet's published runs (an empty
  answer is scored wrong unread), where a choice at random among four would
  credit 25%. It scores the runs 19.2 and 21.0 points above the share within
  5% of the key; a reader choosing at random on every miss would score them
  21.6 and 21.0 points above it (`miss_rates`). Which misses it credits
  follows the options: 44.0 and 68.3% of those whose single nearest option is
  the key, 12.7 and 11.4% of those nearest another option, 29 and 34% of those
  with no number. gpt-4o's published reading names no option on about 30% of
  misses (the parser's `Z`), Claude's on under 1%; of the reads that name one,
  the nearest option is named on 63.8 and 48.3% (gpt-4o, key nearest or not)
  and 68.7 and 68.0% (Claude) (`picks_on_misses`, `--latex-picks`). The misses
  nearest another option are not explained by the key's value in the notebook
  (written on 16% of those runs, a distractor's on 16 and 14%) or by a slip of
  scale. The may-decline reading credits 2.0 and 5.6% of misses but rejects 8.8 and
  10.2% of the correct answers, and is within 4 points of the tolerance; at
  every tolerance from 1 to 10% the forced readings sit 13 to 24 points above
  it (Table 36). Our seven run sets: +14.6
  forced, +1.3 may decline; gpt-5.1 as the published agent, read by gpt-4o,
  +10.0 forced and -0.2 may decline (the closed models, below).
- **Against other references** (`reference_check.py`, Table 23). Against
  BixBench's own open-ended grading of the same runs, which reads the whole
  answer, the published forced-choice excess is +20.4 and +21.7; counting any
  number in the answer, or dropping the 19 p-value keys, moves it by under a
  point, and grading those keys within a factor of ten leaves +17.5 and +19.1.
  The open-ended graders accept 0.2 and 0% of the answers with no number, which
  the forced grades accept at 29 and 34%. (`published_rates` also applies the
  published grades' rates to the 48.8% open-ended accuracy a 2026 system
  reports; the paper does not, since that system's reader sees no notebook.)
  With each of the eight agent models counted once, the cost over all
  items still falls with accuracy (rho -0.90, p = 0.005); per miss it does not
  (rho -0.43, p = 0.30).
- **How a number is read** (`answer_extraction.py`, Table 30;
  `extraction_verdicts.py`, Table 22). The registered
  rule reads only answers that are a number: 60% of our v1.5 answers, 58 and
  91% of gpt-4o's and Claude 3.5 Sonnet's published ones. Read from the last
  number in the answer, as the tolerance reads it, no moved-key contrast
  shrinks (+16.7 to +24.8 on v1.5, +18.6 to +21.9 on the published runs), so the
  registered reading is the conservative one. Re-run with the last number,
  every hypothesis of the pre-specified test keeps its verdict on every data
  set (`extraction_verdicts.py`; the whole-answer pass reproduces
  `replication.json` exactly).
- **The trade** (`bracketing.py`, `channel_survey.py`, Observation 1,
  Figure 1a-b). For the rule, a key is at an edge when it is the smallest or
  largest in magnitude among the options of its sign (which differs from the
  value rank on 2 + 4 zero keys of v1.5 and 2 + 2 of v1.0). Observation 1:
  a key whose nearest options of its sign sit at factors rho- and rho+ accepts
  the misses within sqrt(rho-) and sqrt(rho+); a rank uniform over the middle
  ranks leaks 2/(k(k-2)). When a numeric answer is read as the option nearest to it,
  four options cannot both hide the key and catch a miss. Bracketing a share
  beta of keys forces a question-blind rule at least beta/(k-2) - 1/k over
  chance; a file whose key rank is uniform leaves 2/k of its keys at an edge,
  where every miss on the open side is nearest the key. A reader that may
  decline is not bound by the trade, and a tolerance is outside it. BixBench
  v1.5 brackets 81% of its 105 numeric keys and the second-smallest option is
  the key on 54 (51.4%); the second-smallest rule takes +26.4 [+14.0, +38.6],
  chosen afresh in every leave-one-capsule-out fold. v1.0's best rule, +5.8
  in-sample, falls to -2.4 held out. Table 7 gives each of nine files' rule
  chosen on held-out clusters against its floor (App. B).
- **Where v1.5's rank came from** (`leak_origin.py`, Table 8). v1.5 kept 62 of
  v1.0's 159 numeric items (same capsule, same key), mostly the bracketed ones
  (77% against 51% of those dropped), and rewrote the distractors of 24 of
  them, putting the key at the second-smallest rank on 19 (from 29% to 79%);
  its 43 new items put the key there on 40%.
- **What hiding the rank would credit** (`bixbench_withdata.py`,
  `reader_split.py`, Table 27). Seven run sets frozen and re-read
  through the released options, a placebo (rank held) and a repair (rank
  uniform). On the 37 keys the repair moves to an edge, against the placebo:
  the option nearest the agent's number is the key +16.7 points more often
  (+15.3 and +16.5 keeping one protocol per family; +15.1 without the data),
  +5.3 over the whole file.
- **Who pays** (`run_set_scaling.py`, Table 32). Over nineteen run sets with
  the data the rule's moved-key gain falls as agents land within 5% more often
  (Spearman -0.56), and its whole-file gain falls faster (-0.89), from +10.2 to
  +0.7 for Qwen3-235B-A22B at 23.8%. gpt-5.1, at 27.6%, gives the rule +3.1
  [+0.7, +5.7] over the whole file (`strong_agent.py`, Table 33; the report
  also extrapolates the nineteen run sets' line to gpt-5.1's share, which the
  paper does not use).
- **Under BixBench's readers** (`published_reads.py`, `reader_split.py`,
  Tables 26 and 27, App. F). Three open models read BixBench's published gpt-4o and
  Claude 3.5 Sonnet runs through `MCQ_EVAL_PROMPT`, forced, where the rule
  gains +18.6 on the moved keys: Qwen2.5-72B, which names the option nearest
  a miss on 63.1% of misses as the published reading does on 63.3%, gains
  +14.4 [+7.0, +22.2]; gemma-3-27b (49.4%) +8.7 and Llama-3.3-70B (42.2%)
  +8.8; all three survive the paper-wide correction. Over the whole file they
  gain +0.5 to +3.0, against the rule's +3.4. None agrees with the published
  forced reading run by run beyond kappa 0.53. On the seven v1.5 run sets
  the readers inherit less: each run set's own reader +5.8 [+1.2, +10.4],
  which is not among the ten corrected findings; Qwen2.5-72B +10.2, gemma-3-27b +4.6,
  Llama-3.3-70B +0.4. On the registered new runs, where the rule gains +17.9
  on the new seeds, the three forced readers gain +0.7 to +6.8, none
  significant by the capsule sign-flip test (p 0.059 to 0.861).
  Allowed to decline, gemma-3-27b agrees with BixBench's published
  may-decline reading on which runs are read as the key (kappa 0.79), less on
  which of the rest it declines (kappa 0.54 over key, other option, decline),
  and gains +3.2 [+0.4, +7.3] on the published runs. gpt-4o itself, run again
  as the reader (version 2024-11-20), gains +5.9 [+1.0, +11.2] (p = 0.014) and
  names the option nearest a miss on 59.1% of the misses where it names one.
  Told to pick an option it names none on 2.1% of correct answers and 56.0% of
  misses, and 92.5% of those replies say no option matches; BixBench's parser
  scores them wrong, as it scores a decline. The published gpt-4o reading of
  the same runs names none on 24.4% of misses, Claude's on 0.2%
  (`grader_declines.py`).
- **The proximity weight** (`proximity_weight.py`, `lambda_holdout.py`, Remark
  1, Tables 26 and 28). A forced grader that on a miss selects no option with
  probability delta and the nearest with probability q bears lambda = (1 -
  delta)(4q - 1)/3 of the rule's gain from a change of options. Measured from
  each grading's own selections on the misses of the same runs (ties in d, as
  for an answer of 0, broken by the absolute distance), lambda orders the 46
  gradings of Tables 26 and 28 at Spearman 0.88: code-free graders 0.48 to 0.69,
  BixBench's graders with the notebook about 0.1 to 0.5, the within-5% option
  under 0.1. BixBench's own published grades, which exist only through R, have
  lambda 0.44, near Qwen2.5-72B's 0.43. Fitted on the unchanged and inward keys
  alone (`lambda_holdout.py`), lambda moves by a median of 0.006 and predicts
  the moved keys' gain with a mean absolute error of 3.9 [3.3, 5.8] points over
  42 gradings, against 9.7 [7.8, 12.5] for a constant share of the rule's gain
  and 3.4 for the mean share of the grading's kind: it orders graders and does
  not price them.
- **Without the notebook, constrained, or with a tolerance option**
  (`grading_variants.py`, `grading_variants_analysis.py`, Tables 26, 28 and 29;
  not registered). Graded as a 2026 system grades, from the answer and the
  options only (code-free), the published runs score +27.2 and +21.6 over
  BixBench's open-ended grades over all questions, where the published grades
  with the notebook score +24.3 and +21.0 and a random choice on every miss
  would add +22.2 and +20.4; on the questions whose options are not numbers the
  published grades exceed the open-ended ones by +29.0 and +20.2. Over all 205
  v1.5 questions gpt-5.1's runs gain +15.2 [+11.5,+19.2] graded with the
  notebook and +21.7 code-free, against +18.5 at chance. Code-free graders bear
  most of the cost of a hidden rank: +27.2 [+14.7,+38.5] (gpt-4o) and +28.2
  [+17.4,+39.4] (gemma-3-27b) on the published runs' moved keys, +21.2
  [+11.9,+30.6] and +21.8 [+15.1,+28.8] on the new seeds. gpt-4o 2024-11-20
  constrained to a letter never selects no option, accepts 26.3% of the
  quarter's misses (the published grades of the same runs 24.2%) and gains +6.6
  [+2.0,+12.1]. An option stating that none of the others is within 5% makes
  grading through options a tolerance: code-free graders accept 0.6 to 1.5% of
  misses and 86.8 to 93.4% of correct answers, and hiding the rank costs nothing
  (-1.1 to -0.1).
- **The digit-matched rewrites** (`digit_matched.py`, App. E,
  `tests/test_digit_matched.py`). The same generator and seeds with every new
  distractor written to the significant digits of the one it replaces: the
  fewest-digits rule falls from 31.7 and 31.9% on P and U to 25.3 and 25.5% on
  P' and U' (25.0% on R), and no moved-key, unchanged-key or all-item contrast
  of the rule moves by more than 2.9 points, DeepSeek-V4-Pro's few accepted
  answers apart (U' - P' on the published runs' moved keys +18.6 [+8.4, +31.4]).
- **The accepted misses** (`miss_anatomy.py`, App. F). On answers with no number
  the graders favour the smallest option, not the second-smallest; 62% of the
  published grades' accepted misses are more than 25% off and 7% within 11.5%;
  H2's +3.7 on the published runs comes mostly (2.4 points) from 11 keys U moves
  from one extreme to the other.
- **Four more runs of Qwen3-235B-A22B** (`q235_extra.py`, App. F; not
  registered). The same agent, protocol and settings as its two runs in the
  pre-specified test (rollouts 2-5, with the data, on v1.5's 105 numeric
  questions, under a three-hour wall clock that cut 2 of the 420 episodes, both
  on one question whose R cells kept outrunning the cell limit; they count as
  unanswered): within 5% of the key on 22.9 to 27.6% of questions; moved keys
  +9.5 [+0.9,+17.4] against P on 16 newly accepted runs, unchanged keys +4.8
  [+0.3,+10.3] against R; all six pooled, +7.9 [+0.7,+14.7] and +6.2 [+1.6,+11.6]:
  the moved keys' gain now clears zero, as a lack of power in the test would
  leave it, but the test's verdict stands; the unchanged keys' gain comes from
  the redrawing (+0.3 against P on the 30 that keep their rank).
- **Asked for a number** (`forced_guess.py`, App. C; not registered). gpt-4o
  2024-11-20 on each release's numeric questions without the data or the
  options, asked for its answer as a number (free_response.py's free arm) and,
  in a second prompt, told a best estimate is required: it gives a number on 104
  of v1.5's 105 and every other question, within 5% of the key on 1.9% of v1.5's
  (8.2 and 3.2% of v1.0's), and the option nearest it is the key on 17.3 and
  13.1% (v1.0: 22.8 and 18.8%), against 25% for a random option. Forced among
  the released options under BixBench's template it selects the key on 30.8%
  (v1.0: 35.2%), 13.4 to 17.7 points more on v1.5 than choosing the option
  nearest its own number would score. Replies:
  `results/forced_guess_replies.jsonl.gz`; about $0.14.
- **Registered in advance** (`PREREGISTRATION.md`, `replication.py`, Tables 20
  and 21). The plan was pushed before its analyses ran. On BixBench v1.0's own
  published with-data runs of gpt-4o and Claude 3.5 Sonnet (`eval_df.csv`,
  5,161 runs on its 159 numeric questions) four predictions hold: +19.8
  [+10.4, +31.2] on the 41 moved keys, +18.6 against the placebo, +3.4 over the
  whole file; the fifth, that kept keys do not move, passes only by its
  registered 5-point margin (+3.7 [+0.9, +6.9] excludes zero) and the paper
  counts it as failed. H1, H3 and H5 follow in direction from Observation 1;
  H2 and H4, whether the repair acts through the rank alone, are the
  informative ones. New v1.5 seeds of three agents pass all five;
  Qwen3-235B-A22B fails H1, H2 and H4 (H1 because its moved keys' interval,
  +9.5 [+0.0, +22.1], reaches zero).
- **Who comes first** (`ranking_check.py`, Table 31). Not settled on seven run
  sets: the tolerance's Kendall tau with the graders' order exceeds the family
  reader's by +0.85 [-0.10, +1.25], an interval that reaches zero, though a
  noise-matched control's exceeds it in 98% of draws; the rule's is not
  distinguishable from the tolerance's. On the published runs Claude 3.5
  Sonnet leads gpt-4o under every reading.
- **What the no-data baseline measures** (`release_arms.py`,
  `rank_attribution.py`, `partial_knowledge.py`, Tables 9 and 10). BixBench's
  published models state the answer unaided on 6 of 205 questions, which would
  lift a forced score 2.2 points over chance (3.7 and 4.8 crediting declined
  questions at the attempted rate); forced, they score 11.1 and 9.1 over it.
  Where they state a number, its nearest option is the key on 19.3 and 11.0% of
  v1.5's numeric questions, below the 25% of a random option; forced, they pick
  the option nearest their own number on 35.2 and 27.0% of v1.0's. A
  preference for the second-smallest rank could carry at most 45 and 59% of
  their margins (28 and 32% at 95%). gpt-4o run again without the data
  (`openai_nodata.py`, Table 14) is at chance with the question withheld (+1.0
  and +1.4) and, under BixBench's template, keeps its margin when every
  distractor is redrawn: the margin rests on the question and the key's value,
  not on the options' rank.
- **Solved examples in context** (`icl_probe.py`, `icl_analysis.py`, Table 11).
  Eight open models, 1.5 to 72B, shown up to 64 solved v1.5 items from other
  capsules gain no more from examples at the released ranks than from the
  same examples redrawn to a uniform rank (-1.3 to +1.3 points at 64 examples,
  question withheld); examples lift the larger models by up to +6.1 either way.
- **What survives a correction** (`claim_budget.py`, Table 6): the ten
  headline findings (the abstract's and the introduction's, and the two no-data
  margins App. C rests on), as one family at 1 - 0.05/10.
  Eight survive, the three open readers' gains on the published runs among them;
  Claude's no-data forced margin reaches -0.2, and the current agents' excess
  over the tolerance does not clear because gpt-6-luna's reaches -1.1. At the
  nominal 95% all ten clear. The family is post hoc and the paper says so.
- **Writers** (`writer_panel.py`, `frontier_calibrated.py`,
  `key_marginal_symmetric.py`; not in the paper). On
  BixBench only the bracketing writers credit few misses (12.9%, 14.3%) and
  they leak +26.4; every other writer we built credits 24 to 40%. A writer that
  reads nothing about the item closes the channel only by drawing distractors
  from other items' keys, which on MMLU makes items +14.9 to +20.6 points
  easier.
- **The tolerance** (`tolerance_check.py`, Tables 35-36). BixBench's own 60
  range keys have median half-width 6.1%; the v1.0 grader and a 5% rule
  disagree on 120 of 5,161 closed-model answers (2.3%; kappa 0.90, the highest
  of the rules tried);
  the v1.5 graders agree best with 1 or 2% (kappa 0.81 against 0.77 at 5%).
  The paper's tolerance verdicts are restated at 1, 2, 5 and 10%.
- **The reader** (`prompt_factorial.py`, `prompt_factorial_summary.py`; not in
  the paper, whose Appendix D keeps the thirteen models on BixBench). One instruction sentence of BixBench's template moves a
  question-blind reading on MMLU-Pro by 3.4 to 7.3 points on each of three
  models.
- **The closed models** (`openai_api.py`, `openai_nodata.py`,
  `published_reads.py`, `bixbench_agent.py`, `strong_agent.py`; Tables 1, 14,
  26 and 33). Served through an Azure OpenAI resource: gpt-4o as version
  2024-11-20 (2024-08-06, the published runs' version, is no longer offered to
  new Azure OpenAI customers) and gpt-5.1 as 2025-11-13; Claude 3.5 Sonnet has been retired, so
  gpt-4o also re-read Claude's published runs. gpt-5.1 ran BixBench's published
  agent with the data (medium reasoning effort), two runs on each of v1.5's 105
  numeric questions, read by gpt-4o: within 5% of the key on 27.6%, above every
  open-weight run set of Table 32, and 22.4% open-ended; forced it is read
  +10.0 [+3.5, +16.6] over the tolerance, with a refusal option -0.2. 35 of its
  210 answers give no number, 31 of them saying the value cannot be determined,
  and the forced reading credits 21.8% of those, 5.7 points of its score. Two more
  runs on each of v1.5's 100 other questions give its scores over all 205
  (Table 29). None of this is part of the registered test. Every call's tokens
  and estimated cost went to a local ledger (`build/openai_usage.jsonl`, not
  shipped): about $1,026 at list prices in all, the current agents' runs
  (below), the code-free and letter-only gradings of `grading_variants.py` and
  gpt-4o's code-free reads of the other designs (`option_design.py`) included.
  `results/strong_agent/` ships gpt-5.1's trajectories and gpt-4o's replies;
  `results/strong_agent_rows.json.gz` the scored rows every number above is
  computed from.
- **The correction for guessing** (`formula_scoring.py`, Tables 1 and 24; not
  registered). Formula scoring maps a forced score S over k options to
  (S - 1/k)/(1 - 1/k). On the published runs it lands -3.9 [-8.6, +1.3] and
  -0.1 [-4.7, +4.7] points from the tolerance, as a sum of opposite parts: the
  misses nearest the key add 2.6 and 7.8 points beyond a random choice, those
  nearest another option subtract 6.5 and 9.3. On v1.5, the current agents
  included, it lands 6.0 to 10.8 points below the tolerance (its graders reject
  2.5 to 8.6% of correct answers); the 2026 system's 64.4% maps to 52.5%, 3.7
  points above its 48.8% open-ended. gpt-4o's published grades select no option
  on 24.9% of misses and accept 30.1% of those on which they select one. The
  split is exact (the script asserts it); the omission-aware variant scores an
  unselected grade and an empty answer as omissions.
- **Other designs** (`option_design.py`, Section 4.5, Tables 37 and 38; not
  registered). Every numeric item's distractors rebuilt by the generator of P
  and U for k = 4, 6, 8, 10 options with the key's rank uniform over all or over
  the middle ranks, at fixed spacings, and from other agents' wrong numbers
  (leave-one-model-out), then regraded by the rule and four code-free graders
  (gemma-3-27b, Qwen2.5-72B, Llama-3.3-70B on the H100s; gpt-4o on a random
  quarter of the published runs and every v1.5 run). Error-modelled distractors
  leave the cost of a uniform rank in place (+24.1 [+16.3, +33.8] and +18.8
  [+11.3, +28.1] on the moved keys); hiding the rank costs the rule 11.1, 9.0,
  6.1 and 5.6 points over all items at k = 4, 6, 8, 10; eight options a fifth
  of the key apart, key never at an edge, leave the code-free graders accepting
  10.5 to 13.0% of misses and 92.0 to 100% of correct answers. The designs'
  option sets and the graders' reads ship gzipped in `results/option_design/`,
  from which `option_design.py analyse` reproduces `results/option_design.json`
  byte for byte.
- **Degenerate runs** (`degenerate_runs.py`, App. F; not registered). 123 of the
  seven run sets' 735 with-data numeric runs give no answer, hit the step limit
  or have a reply cut at the token limit (49 Qwen3-30B-A3B's, 58 Qwen2.5-72B's);
  without them the forced excess rises from +14.6 to +17.3 and the rule's
  moved-key U - P from +16.7 to +18.8 [+11.7, +25.9]; on the three run sets
  whose accuracy rises with the data the gain is +12.4 [+4.8, +21.9]. Per miss,
  what forced graders accept does not fall with accuracy over 22 run sets
  (Spearman +0.07, p = 0.75; -0.09 with each of eleven models once), and the
  share of misses nearest the key rises (+0.66, p = 0.001).
- **The questions experts re-checked** (`verified50.py`, App. F; not
  registered). BixBench-Verified-50's ids, from two public ports that pin the
  same revision (the dataset's files are gated): 26 are v1.5 numeric items (14
  moved keys). On them the seven run sets' forced excess is +13.5 [+6.1, +20.1]
  and the rule's moved-key gain +19.6 [+5.0, +31.1].
- **Options that are not numbers** (`nonnumeric_excess.py`, App. F; not
  registered). v1.0's 137 other questions by their options: numbers with words
  58, directions or changes 37, names 24, intervals 10, statements 6, lists of
  names 2. The published grades accept 71.0 and 69.3% of misses whose nearest
  option (in value or wording) is the key, 27.8 and 19.0% of those nearest
  another; on answers nearest no option they accept 52.1 and 34.5% where the
  same model answers correctly without the data, 24.4 and 22.4% elsewhere. The
  per-run picks ship as `results/nonnumeric_published_picks.jsonl.gz` (built
  from the 1.1 GB `eval_df.csv` by `nonnumeric_excess.py build`).
- **Two current agents** (`frontier_agents.py`, App. F, Table 34; not registered).
  gpt-6-luna (medium reasoning effort; through the Responses API, which
  `openai_api.py` uses for models named in `AGENTICLS_OPENAI_RESPONSES`, since that
  deployment takes function tools with reasoning only there) and DeepSeek-V4-Pro ran
  BixBench's published agent with the data through Azure AI Foundry, once per v1.5
  numeric question (all 105 finished), graded by gpt-4o as gpt-5.1's runs are:
  within 5% on 38.1 and 28.6%, open-ended 38.1 and 23.8%; forced choice +11.0 and
  +11.9 points above the tolerance, with a refusal option +3.8 and +0.5, corrected
  for guessing -6.0 and -7.9; the rule's U - P on the moved keys +2.7 and +4.7.
  Kimi-K3 (95 finished) is left out: its answers are long and state the number
  first, so the last number, which the tolerance reads, is often another (open-ended
  53.7% against 32.6% within 5%).
- **How the placebo and the repair are built** (`mcq_audit.py`,
  `build_placebo_arm.py`, App. E; `tests/test_option_sets.py`). One generator,
  `mcq_audit.redraw_row`, redraws the three distractors around the key: at the
  key's released rank for the placebo, at a rank drawn uniformly for the
  repair. It spaces them by the released distractors' mean relative distance
  (at least 0.05), each at a random 0.6 to 1.4 of its step, in the written style
  of the distractor it replaces, and redraws the whole set when rounding
  collides two options, flips a sign or moves the key off its rank (up to 24
  draws; the repair then tries the other ranks, 3 of v1.5's 105 items and 10
  of v1.0's 159; v1.0's placebo leaves one item out). Seeds: the repair
  20260918 (`mcq_audit.py --repair`), the placebo 20260920 (v1.5's by
  `mcq_audit.apply_repair(..., preserve_rank=True)`, v1.0's by
  `build_placebo_arm.py`). Regenerating the four files reproduces them option
  for option.
- **Still in use** (not ours). A 2026 system reports 64.4% forced against 48.8%
  open-ended on v1.5, reading the forced score without the notebook
  (arXiv:2601.12542); BixBench-Verified-50 grades fifty re-checked questions
  without options.

The sections below are the repository's longer working notes. Where one
disagrees with the paper, the paper and its validator are current.

## Working notes: the result in one paragraph

A no-data baseline is the field's fallback contamination probe: withhold the
data a task needs and read an above-chance score as recall. BixBench runs
exactly that and calls the result "the pure recall performance of both models".

**BixBench also published the control, and without options the answer all but goes.**
Beside the quoted run it released the same two models on the same 205 questions
with the **option set deleted**, and again with an "insufficient information"
option added. Its own graders, its own bytes, no instrument of ours:

| arm | claude-3-5-sonnet | gpt-4o |
|---|---|---|
| four options, no way to decline — **the quoted run** | **34.1%** | **36.1%** |
| four options plus "insufficient information" | 8.3%, declines on **78.5%** | 3.9%, declines on **90.2%** |
| **no options at all** | **2.9%** | **2.9%** |

On the 61 questions BixBench grades with its own **numeric tolerance** the
deleted-option arm reads **0.0% for both models**. Paired over capsules, each arm
against its own chance rate, the option set is worth **+6.2 [−0.4,+13.5]** and
**+8.2 [+1.1,+15.9]** — but that difference changes the grader and the response
format as well as the options, so read these arms as rates and not as a settled
difference. What survives a correction across the paper's ten headline findings
(`claim_budget.py`, below) is gpt-4o's forced arm's margin over its own chance rate,
+11.1; Claude's +9.1 reaches −0.2. Three arms, two graders, two response formats and three
chance rates put these models' unaided no-data score between nought and eight
per cent — **thirty-four is what four options and a forced choice add.** The rest
of this repository is about where those thirty points come from and what it
would cost to remove them.

The margin splits into a recall term and an option-geometry term — exactly, for
a solver that reads a rank, and as an upper bound on the geometry otherwise.
Two registers have to be kept apart, and this repository reports both. **The
channel is wide**: BixBench's numeric items hand a one-line rule +26.4 points,
and on MMLU-Pro's 913 ten-option numeric items a fitted solver given no question
and no data scores 29.1% against 10% chance, 16.9 points above a clean file of
the same shape. **And what a baseline reads off it is decided by how it asks.**
Hold one model, one file, 913 items, one set of letter orderings and one clean
control fixed, and vary only the prompt and the read-out:

| how the question is put | highest-logit letter | generated |
|---|---|---|
| BixBench's own template, verbatim | 11.0%, **+1.3 [−1.6,+3.8]** | 10.1%, **+0.5 [−2.4,+2.8]** |
| "withheld, read the options" | 17.0%, +5.2 [+2.3,+8.5] | 14.8%, +4.1 [+1.9,+6.3] |
| question simply absent | 19.4%, +8.3 [+5.0,+12.4] | 20.8%, **+10.4 [+7.1,+14.3]** |

**The read-out is worth a point; the prompt is worth nine.** The top row is the
19-pair null the field reports, reproduced as one cell of a controlled design.
**On BixBench's own 205 released items the template suppresses nothing.** Same
crossing, all 205 released items in 59 capsules, control resampled from the
file's own pool of 709 option strings:

| how the question is put | highest-logit letter | generated |
|---|---|---|
| BixBench's own template, verbatim | 30.1%, **+8.5 [+3.2,+13.6]** | 28.0%, **+8.1 [+3.0,+13.1]** |
| question simply absent | 37.9%, **+14.1 [+8.1,+19.9]** | 38.4%, **+15.0 [+8.8,+20.7]** |

Every cell clears, **including the one the field runs** — and not only for this
model: across thirteen models in five families on that file, **five clear under
BixBench's own template** — phi-4 at +12.2 [+5.8,+18.9], the 14B at
+8.5 [+3.2,+13.6], Phi-3.5-mini at +7.5 [+1.3,+13.8], Llama-3.3-70B at
+6.8 [+0.7,+13.4] and a **1.5B** model at +6.2 [+0.4,+11.7] — where the
ten-option file gives that same instrument **one** clearing cell in thirteen
(Qwen2.5-32B, +4.5 [+1.6,+7.5]). So what a baseline reads is a property of the
file as well as the prompt. The fitted trend is +2.9 points per decade of
parameters on BixBench's file and +2.2 on the ten-option one, and neither
ordering is monotone: phi-4 at 14.7B reads highest of the thirteen, above both
70B-class models.

**Whose run that is.** The +8.5 is *our* run of BixBench's template, verbatim at
commit `4931118`, on open-weight models the benchmark never used — not a
published baseline, and `validate_artifact.py` fails if the paper calls the
published no-data baseline "already collecting". **And it is
the numeric half**: under that template the 105 items whose options are all
numbers read +10.6 [+5.9,+15.8] against chance where the 100 that are not read
−0.7 [−5.2,+4.0]. Under the framing that omits the question both halves clear.
It is not the interpreter (+10.4 with none against +4.6 holding one), not an
instruction to read the options (saying so *lowers* every read-out), and not
reasoning (the cell that reads highest answers immediately on 99.9% of
rollouts). Equation (1) locates the difference in `b`, the reader's pick-rank
preference: under the template it is flat and tilted to the smallest option
(⟨p,b⟩ − 1/10 = +0.35 [−0.31,+0.81]); under the framing it peaks where this
file's key sits (+3.15 [+2.35,+4.25]). The same null replicates outside Qwen2.5
— phi-4 under BixBench's template reads +2.59 [−0.35,+5.10] on every rollout —
but the **prompt effect does not**: under the neutral framing phi-4 refuses on
64.7% of released and 38.8% of control rollouts, so that cell measures refusal
and is reported without being read. The reading also
needs the published option text as well as the framing: a placebo that holds the
key's rank exactly and redraws every value reads +2.8 [−0.3,+5.3], which no
longer clears.

**And it is not scale, nor one pretraining corpus.** Llama-3.3-70B answers
every rollout of the neutral cell and goes **+1.5 [−1.2,+3.8] → +5.4
[+2.3,+9.2]** across the same two cells — the family confound, closed, where
phi-4's refusal could not close it. Across thirteen models and five families, run head to
head on the same items and control, the template reading **spans −3.1 to +4.5
over two decades of parameter count and clears once** (Qwen2.5-32B; both
70B-class models above it do not). **Changing the prompt moves one model further than changing
the model by a factor of sixty does** — the 14B goes +1.3 → +10.4. Under the
neutral framing every Qwen2.5 clears, from **1.5B (+4.1 [+1.9,+6.5])** to 72.7B
(+8.6 [+5.1,+12.6]). The paper's earlier "only the two largest of four" varied
the model and the instruction together. What does bind is compliance: five of
the thirteen decline the neutral cell on 13% to 65% of rollouts, so the framing
that reads the channel is one half these models will not answer.

Maximised over readers rather than evaluated at one, the geometry term is a
single quantity Γ, and Γ = 0 **if and only if** the key is uniform inside its own
option set. Exchangeability of the (key, distractor) tuple is sufficient and
strictly stronger — sorted distractors satisfy Γ = 0 without it — and
`writer_theory.py` enumerates the separating law. Two writer classes can then be
characterised rather than surveyed: a writer that reads nothing about the item
closes the channel **only** by drawing distractors from the key marginal itself
— for an arbitrary joint law over the k−1 distractors, not only independent
draws — and a solver given the question then wins at a rate fixed by the
benchmark's own key pool rather than by the writer. Seven distractor-writing
operators on 464 fixed MMLU items, ordered by what the writer may see, leave the
corresponding corner of the trade-off empty. The two whose distractors have
nothing to do with the item close the channel and hand a solver given the
question +11.1 to +20.4 points; a writer shown the **question** and never the
key leaves the items exactly as hard as it found
them (−1.5 to +2.3) and still leaks +8.6 [+3.1, +11.4], carried by the items
where the writer never found the answer. A reader built from hashed character
n-grams, with no feature this repository designed, reads the repaired files the
same way, and reads one of them 7 points higher than the designed features do:
every Γ here is a lower bound from one reader class, and it is reported as one.
Finally the premise: no release publishes a date, but 740 of 2,558 tasks are
datable from identifiers the files already carry, and running the temporal test
on LAB-Bench LitQA2 returns a null with a nineteen-point interval — next to a
proxy that at BixBench's 105 items could not have detected recall below λ = 0.2
either.

## What this reports

The list below is the full set of findings in the order they were established.
**The paper's argument is findings 2, 3, 5, 12, 16, 19 and 25--27**, in that
order: the decomposition and its sufficient condition; how open the channel is
and who walks through it; that uniformising one ordering opens the others; that
the construction closes all four; that a learner still extracts as much from
statistics that are not ranks; and that removing the geometry entirely does not
remove it, because what is being detected is authorship. The rest is the
apparatus that makes those checkable, and the record of what we got wrong.


1. **A provenance census.** Twelve released files from five science-agent
   benchmarks (BixBench, LAB-Bench, ScienceAgentBench, CORE-Bench,
   DiscoveryBench), covering 2,558 tasks. **None** carries a date for the
   source analysis, the underlying data, or the benchmark artifact — which is
   why the temporal audit is unavailable and the fallback probe gets used.

2. **An exact decomposition.** With `p` the distribution of the keyed answer's
   rank among the sorted numeric options and `b` a solver's preference over
   ranks, `G = <p,b>` is the score a purely rank-driven solver gets, and

   ```
   accuracy - 1/4  =  λ(1 - G)   +   (G - 1/4)
                      ^recall^       ^geometry^
   ```

   `λ` and `b` are fitted by EM on the 4×4 table of (key rank, chosen rank),
   which is over-identified, so the fit is tested rather than assumed. At the
   published sample size the estimator is unbiased (λ̂ = 0.099 at λ = 0.10;
   0.201 at 0.20) and the knowledge test has size 5.3% against a nominal 5%.

3. **A bound, and the fix it implies.** `G` is a convex combination of the
   entries of `p`, so `min p ≤ G ≤ max p`. A **uniform** `p` therefore gives
   `G = 1/4` for *every* solver, present or future — no assumption about solver
   behaviour required. Shuffling the options cannot help: a permutation changes
   which *letter* carries the key, not its rank among the *values*.

   Nothing in that argument uses the options being numbers. **Every statistic
   that orders them opens its own channel** — sorted value, written length, how
   round the number reads, how unlike the others it is — each with its own `p`,
   its own `G` and the same bound. Uniformity in one coordinate says nothing
   about the others, which is finding 12. On BixBench's numeric items three of
   the four are demonstrably open as released: value +26.4, roundness +10.1,
   length +8.9 points (simultaneous lower bounds for the last two). The fourth,
   lexical isolation, reads −1.5 — below zero, and far inside the +1.5 that
   files with no leak in them reach on that coordinate (finding 14) — so it is
   *not* established, and the paper claims three.

4. **The defect, and how common it is.** On BixBench v1.5's 105 four-option
   numeric items,
   `p = (12.4, 51.4, 29.5, 6.7)%` — interior 81.0% [72.4, 89.0] against 50%
   expected, χ² = 51.0. Available geometry credit is **[−18.3, +26.4] points**,
   and the upper end is reached by "always take the second-smallest value",
   which scores **51.4% [39.5, 63.4]** reading no question and no data. v1.0
   was milder (`[−8.6, +5.8]`), so the revision tripled the payout and no
   released field records it.

   This is not one benchmark's mistake. The same measurement over **18 released
   multiple-choice benchmark files** (9 with enough numeric items to assess)
   rejects a uniform key rank in **seven of nine** and puts a positive lower
   bound on the available credit in **five** — four of them on enough source
   groups to trust the bootstrap. The largest sample is **MMLU-Pro**: 1,263
   numeric items over 10 options, interior rate 94.2% against 80% expected,
   χ² = 575 on 9 df, and **at least +8.4 points** available. BixBench v1.5
   gives ≥ +13.6, MMLU ≥ +1.8, SciQ ≥ +8.3, LAB-Bench SeqQA ≥ +1.9 (flagged:
   only four groups).

   **AQuA-RAT is the counterexample that matters.** Over 141 numeric items its
   key rank is statistically indistinguishable from uniform (χ² = 1.2,
   p = 0.87). Its options come from exam-practice websites and crowdworkers, and its
   paper does not say how the distractors were chosen; they are not perturbations
   of the right one. Numeric options do not have to leak.

5. **Who collects it: almost nobody.** Realised credit across thirteen solvers
   spans **−1.3 to +2.6 points**, at most a tenth of what is available.
   - *The models behind the published margin*, fitted from BixBench's own v1.0
     predictions with no new inference: gpt-4o λ̂ = 0.069 [0, 0.158], geometry
     −0.3 points; claude-3-5-sonnet λ̂ = 0.028 [0, 0.119], geometry −0.7. Both
     prefer *large* values, so the interior-heavy key distribution works mildly
     against them.
   - *A controlled experiment*: 11 open-weight instruction-tuned models
     (1.2–33B, five families), five arms, 20 redraws, **225,500 forced-choice
     decisions**. Rank effects run −3.7 to +5.8 points; the two largest models
     are nominally positive, but eleven tests at the 5% level also return three
     nominally *negative* effects and the pre-specified trend test does not
     reach significance (Spearman ρ = +0.15, permutation p = 0.67). Reported as
     a suggestion, not a finding.

6. **Controls that make the null informative.** The **placebo** arm redraws the
   distractor values through the identical generator while holding the key's
   rank, so *placebo minus repair* isolates rank. Rank-only solvers are
   *exactly* unmoved by it (51.4% → 51.4%) and collapse under the repair
   (51.4% → 26.8%, +24.6 points, p = 0.0007); a flat-preference guesser does
   not move (−0.3, p = 0.85). The models behave like the guesser, not like the
   rule.

7. **What the margin is instead.** Recall-shaped and estimable. Nine of eleven
   models have λ̂ indistinguishable from zero; Qwen2.5-14B gives 0.130
   [0.052, 0.203] and Qwen2.5-32B 0.168 [0.082, 0.251] — the latter answering
   39.3% of no-data numeric items correctly, above the published gpt-4o figure.

8. **λ is validated where the answers really are known.** A decomposition term
   is only useful if it tracks the thing it is named after, and BixBench cannot
   test that. MMLU can: its 661 four-option numeric items are answerable from
   what a model already knows. Running the same five arms there, **λ̂ orders
   exactly with accuracy** — 0.001 [0, 0.04] at 24.6% (Llama-3.2-1B), 0.167
   [0.12, 0.23] (Llama-3.1-8B), 0.257 [0.20, 0.33] (Qwen2.5-7B), 0.565
   [0.45, 0.70] at 67.7% (Qwen2.5-32B) — while the *same models* sit at
   λ̂ ≤ 0.03 on BixBench with the data withheld, and withholding MMLU's own
   question text drops all four to 22–25%. Geometry stays at +0.2 to +1.0
   points, as MMLU's own +1.8 bound requires.

   This run is also the strongest argument for the **placebo**: redrawing the
   option values makes MMLU items *easier* for a solver that knows the answer,
   by 3.7–9.0 points — three to four times the rank effect. Anyone running the
   naive original-vs-repaired contrast would attribute all of that to rank.

9. **A repair with a guarantee.** `mcq_audit.py` searches a rank-preference
   family **in both directions** (a rule that is reliably *wrong* leaks too:
   excluding one option lifts a guess from 1/4 to 1/3, and an earlier version
   of this audit missed a 31% exploit by only looking for rules that pick the
   key), selects by leave-one-cluster-out, and can rewrite the file. On v1.5 it
   covers all 105 numeric items, **alters no keyed answer**, moves the exploit
   from 51.4% to 24.8% [17.9, 32.2], and cuts the credit available to *any*
   rank-only solver from +26.4 to **+4.5 points**. The tool handles any option
   count, and the fix gets harder as options multiply: on **MMLU-Pro's 1,263
   ten-option** numeric items it cuts the credit only from +11.5 to +6.4,
   because an extreme rank needs nine same-style values on one side and
   rounding to the replaced options' written precision collapses them. It
   reports this as *reachability*: 83% of (item, rank) pairs are placeable
   against 97% at four options, falling to 64% at the top rank. Three
   different generators give the same rate, so that is the arithmetic and not
   the search — and for many-option numeric items, closing the channel needs
   wider value ranges, more written precision, or fewer options.

10. **One column would settle it.** v1.5 — the release carrying the artifact —
   publishes the keyed letter but not the options as presented, so whether the
   quoted models exploited *its* geometry is not answerable from the release.
   v1.0 publishes it, and can be decomposed.

11. **TATE.** Time-Anchored Twin Evaluation: a three-condition matched design,
    a temporal-manifest JSON Schema, a date-only eligibility checker, and a
    sample-size calculation anchored on a **measured** marginal SD (σ = 0.230).
    Specified, not validated.

BixBench's own agent was run with the data and without it on five open-weight
families, in text and as published (finding 32); no capability claim about any
deployed system is made.

12. **Closing one coordinate moves solvers onto another.** The guarantee in
    finding 3 is exactly as narrow as it sounds, and this is what that costs.
    MMLU-Pro's released numeric items leave **+4.1 points** open on *written
    length* (simultaneous lower bound) beside the +11.5 on value, and solvers
    collect neither: with the question withheld they sit at chance (9.3%,
    10.4% against 10%). Uniformising the value rank leaves the length channel
    exactly as wide — and puts them on it. The same two solvers, question still
    withheld, score **14.1% and 18.4%** (+3.7 and +9.1 over the released file,
    both *p* = 0.0002), and `channel_attribution.py` says where it comes from:
    +1.7 and +3.3 points of the length channel, under +1.6 of anything else.
    Regenerated distractors did not open a channel; they *aligned the solver
    with one already open*. The part left over named a fourth coordinate: the
    repair widens the *roundness* channel (significant digits) from +4.6 to
    +9.8 and solvers take +1.8 and +3.2 of that too. It was found the way the
    method says to find one — attribution showed a gain the audited orderings
    did not account for, so it was measured, seen to leak, and promoted into
    the set the repair fixes. `--two-channel` draws a uniform target rank for
    every ordering at once: on v1.5 that returns the file to **no leak
    detected** (value +26.4 → +4.5; isolation, length and roundness from −1.5,
    +8.9, +10.1 to −4.5, −8.5, −6.2), still altering no keyed answer. At ten
    options the two-channel sampler only narrows them (value +11.5 → +2.4,
    length +4.1 → +2.7, roundness +2.2 → +1.5, isolation from closed to +1.2),
    because a number's written form is nearly fixed by its value and the style
    it must keep — which is what the construction of finding 16 fixes. **Verify a repair on every ordering you can
    measure, not the one it targets** — and then probe the output, because at
    ten options that is still not enough. Repairing all four coordinates leaves
    the two MMLU-Pro solvers +3.6 and +8.4 points up with the question withheld,
    exactly where the value-only repair left them, while the audit reports every
    coordinate narrowed. At four options the two agree: BixBench's repaired file
    moves four solvers −4.4 to +0.2 points, nobody gaining. So the recommendation
    is scoped — at ten options prefer fewer options or curated distractors over a
    regeneration, because the coordinates we can name do not span what a
    regenerated set gives away.

    A practical trap comes with it: the arm maintainers actually run is the
    no-data baseline, question shown and data withheld, and regenerating
    distractors *raises* it — Llama-3.1-8B goes 17.9% → 27.2% on MMLU-Pro. Only
    0.5 of those 9.3 points is geometry; the placebo arm, which redraws values
    but holds the key's rank, already reaches 26.7%. The rest is the redrawn
    distractors being less plausible than curated ones. Repair a benchmark,
    re-run its no-data baseline, see the number go up, and it would be easy to
    call that contamination. It is the repair.

    On the *released* ten-option file the panel is not uniformly at chance:
    Qwen2.5-7B and Llama-3.1-8B sit at 10.4% and 9.3% against 10% and take
    under ±1.7 points on every coordinate, but **Qwen2.5-32B scores 13.4% and
    collects +2.3 points [+1.6, +3.1] of the +11.5 value channel** — a fifth of
    it, and the largest realised credit in this artifact. Reported as the
    exception: the claim is that the channel is *largely* unexploited, and the
    one solver collecting is the most capable in the panel.

13. **The channel is not learned in context either.** "Unexploited" might
    mean only "not yet learned", so `icl_probe.py` shows a solver up to 64
    solved items from the benchmark itself, stems withheld so a demonstration
    can teach only where the key sits, with the target item byte-identical
    across arms. Eight models, 1.5 to 72B (the two 70B-class ones through vLLM
    in FP8, `--engine vllm`). Demonstrations **do** help: with the stem
    withheld they move the models −3.0 to +6.1 points at 64 shots, the larger
    ones most (Spearman ρ = +0.86 over size). They help exactly as much when
    drawn from the *repaired* file — same examples, same letters, uniform ranks
    — so the rank-isolating contrast is −1.3 to +1.3 points at 64 shots with the
    stem withheld, −1.4 to +1.7 with it shown, and −2.7 to +2.4 over the full
    grid of eight models × three shot counts × two stem conditions. Two of those
    48 are nominally significant, about the 2.4 chance predicts, one in each
    direction. The rank contrast's trend over size is ρ = +0.21 (withheld) and
    +0.47 (shown), neither significant. Examples teach the format, not the
    channel.

14. **The audit's headline fires on files that cannot leak.** `audit_calibration.py`
    builds question files with no leak in them by construction: an item's
    options are drawn i.i.d. and rendered in i.i.d. styles, so the set is
    exchangeable, and the key is then one of them chosen uniformly — which makes
    the key's rank uniform under *every* ordering at once. Over 60 four-option
    files each individual test fires on at most 8.3% of them, and the widest
    credit bound the audit ever reports is **+3.3 points**; over 30 ten-option
    files the tests reach 10% and the widest bound +1.0. The single `verdict`
    line is another matter: it is an OR over six tests, so it reads `LEAKS` on
    **16.7%** and **13.3%** of clean files. Read the coordinates and their
    bounds, not the headline — and read the bound rather than the flag, since at
    n = 1,263 the flag fires on departures worth under a point while every
    number the paper rests on is +2.6 or larger. The two rule-family tests have
    their level set by this measurement rather than by a correction formula,
    because selecting a member on the training folds is not a clean multiple
    comparison; at the shipped level they fire on no clean file at either option
    count.

15. **A search that meets more of the repair's targets leaves a worse file.**
    The repair's specification — draw a uniform target rank for the key under
    every ordering and meet it — says nothing about *how*, and how matters more.
    `repair_search_ablation.py` runs three searches × two target distributions ×
    8 seeds. A descent that walks one distractor's rendering and roundness at a
    time meets **70%** of written-length targets where the shipped sampler meets
    **26%**, and leaves BixBench leaking up to **+9.7 points** on that
    coordinate — three times the widest bound ever seen on a clean file — on 6
    of 8 seeds against the sampler's 2. The mechanism is measurable: the rank a
    repair achieves is the rank it drew *filtered* by what the item can reach
    (at ten options only 83% of (item, rank) pairs are placeable at all), and
    the descent's hit rate varies by **55 points** across target ranks where the
    sampler's varies by 19, so the descent's output is a tightly filtered draw
    where the sampler's is two-thirds unguided redraw. Drawing targets against
    the file's own measured reachability (`mcq_audit.fit_target_weights`) removes
    about half the damage and no more, because reachability is a property of each
    item and not only of each rank. The descent's files are also *easier*: shown
    the question, all four solvers score higher on them than on the sampler's,
    because options bent to hit three surface targets at once carry distractors
    a solver can dismiss on sight. We ship the search that is bad at its own
    specification, and keep the better one behind `--search`.

16. **Search was the wrong frame; two of the three coordinates are counts.**
    Written length and significant digits are not objectives to optimise — they
    are *counts over the distractors*: how many are written shorter than the
    key, how many read rounder. A count can be **assigned**. `construct_row`
    enumerates each distractor's possible renderings (template precision,
    grouping, percent, notation, and how round its value is), tags each with
    what it does to the two counts, and picks one per distractor by dynamic
    program. Three things become exact:
    - **The frontier.** `legal_pairs` enumerates the (length, roundness)
      rank pairs an item's key can be given, counting only the pairs whose
      assignment survives every check: **8.4 of 16** at four options,
      **29.5 of 100** at ten. An independently drawn pair lands outside it for
      most ten-option items — which is why every search missed, and why the
      misses piled onto the same ranks. This is the item's repairability
      frontier, and it is a property of the file, not of a search budget.
    - **The target distribution.** With the frontiers in hand nothing is left to
      estimate by running the repair: `fit_pair_weights` does iterative
      proportional fitting over three per-rank vectors and reports the attainable
      uniformity in closed form. On BixBench it is **exactly 25.0% on all three**
      surface coordinates — a perfectly flat marginal is reachable there. Item
      draws stay independent, which is what the leave-one-cluster-out audit
      needs.
    - **The third coordinate.** Isolation is pairwise and does not decompose, so
      it is surveyed over the moves the assignment leaves free: re-render a
      distractor inside its own (shorter, rounder) class, or let two distractors
      *exchange* classes, which leaves the multiset of classes — and therefore
      both counts — identical.

    - **The fourth coordinate too.** The value rank had the same defect: the
      repair drew it uniformly and then walked a random permutation of the other
      ranks until one worked, which gives the ranks the item can reach weighted
      by nothing. At ten options only 83% of (item, rank) pairs are placeable and
      the realised histogram sloped from 13.7% down to 6.7%.
      `placeable_value_ranks` and a fourth fitted vector fix it; **6 of 1,263**
      ten-option items now end on a value rank other than the one drawn, against
      210 for the value-only repair and 123 for the two-channel one.

    Over 8 seeds on BixBench the constructive repair is the best of seven cells:
    flagged on **1 of 8** against the sampler's 2, meeting **100%** of the length
    and roundness ranks it assigns (sampler: 53% and 63%) and 95% of the isolation
    ranks, with worst-case bounds **−1.3 / −3.0 / +0.9** points against the
    sampler's −1.3 / +2.9 / −0.3. And **the ten-option file's numeric items
    close**: value +11.5 → **+1.2**, isolation −0.4 → **−1.5**, written length
    +4.1 → **−0.9**, roundness +2.2 → **−1.1** — every one inside the +1.0 that a
    ten-option file with no leak in it produces (finding 14), so the repair's
    numeric result is not scoped to four options. The file's *verdict* still reads
    `LEAKS`, and should: 8,717 of its 9,980 ten-option items have options that are
    not numbers, a numeric repair does not touch them, and over all of them lexical
    isolation is +2.9 points — unmoved from the released file's +2.8. Closing a
    file's numeric items closes its numeric items.
    `--search construct` in both `mcq_audit.py` and `no_data_probe.py`.

17. **A repair transfers to an ordering it never saw, or not, and the coupling
    says which.** Every number above evaluates the repair on the coordinates it
    targets, which settles nothing about an ordering nobody wrote down.
    `held_out_orderings.py` holds five more out — the key's leading significant
    digit, its decimal places, its digits added up, plain lexicographic order of
    the rendered text, and written length under the opposite tie-break (finding
    21) — none of them a target of the shipped repair, none reported by the
    audit, and all measured only after the files are built.
    - At **four options** the released file leaks on four of the five (up to
      **+7.7 points** on lexicographic order and +7.1 on the reversed tie-break,
      beside the +10.1 of the widest ordering the audit does measure) and the
      constructed repair **closes all five**: −6.1 to −3.0 points, against the
      +1.6 a clean four-option file reaches on these same coordinates.
    - At **ten options** it closes **none** of the four MMLU-Pro leaks
      (+1.4/+2.3/+3.7/+2.5 released → +1.7/+2.9/+3.7/+1.7 repaired, against +0.4
      on a clean file), and leaves two marginally wider than it found them.
    - The difference is **coupling, not effort**. Lexicographic order of a written
      number is monotone in its value only while the options share a digit width —
      `9.5` precedes `12.7` as text and follows it as a number. BixBench's
      distractors are perturbations of the key, so **67.6%** of its items have the
      two orders agreeing (68.6% after its repair) and uniformising one
      uniformises the other; MMLU-Pro's
      options span magnitudes, only 48.8% agree to begin with, and the repair
      *lowers* that to **30.6%** by rendering distractors at varied precision.
      A repair transfers to an unaudited ordering exactly as far as that ordering
      is close to a monotone function of an audited one — and the coupling is
      measurable before any repair is run.
    - **Repairing the held-out coordinate does not close it, and the frontier
      says so in advance.** Lexicographic order is a count over the distractors
      too — how many render before the key as text — so the same dynamic program
      can assign it (`mcq_audit.py --with-lexicographic`, not the shipped
      repair). On MMLU-Pro that narrows the channel +3.7 → **+2.0** and leading
      digit +2.3 → +0.7 and closes neither, while the four audited coordinates
      stay shut (−1.7, −1.0, −0.8). An earlier version of this line reported the
      fifth coordinate reopening roundness at +1.5; that was our bug, not the
      file's — the isolation-tuning stage re-rendered distractors inside their
      `(shorter, rounder)` class, which no count can see and a text ordering
      can, undoing the fifth assignment on 3.3% of items. Fitting over the
      widened frontier (29.5 rank pairs per item → **36.1 triples**) reports the
      flattest marginal the file admits — 10.0% on value, isolation and
      roundness, 15.7% on written length, and **16.2% on lexicographic order**.
      A uniform lexicographic rank is *not attainable* on this file at any
      budget, because a numeral's text order is largely fixed by its magnitude
      and the style it must keep. That is the useful half: a maintainer can
      compute what a file admits before promising anything. BixBench's fit
      returns 25.0% on every coordinate, and there the five-coordinate repair
      leaves all five held-out bounds negative (−0.8, −3.1, −6.0, −2.5, −0.5).
    - **The same five across the survey** (`--survey-sweep`,
      `results/held_out_survey.json`). MMLU's 661 four-option numeric items leak
      on three of the five — decimal places **+2.7**, lexicographic order
      **+2.5**, length-with-later-ties **+1.8** — which matters because MMLU is
      the file this repository uses to validate λ̂. MMLU-Pro leaks on four,
      BixBench on four. MedMCQA (454 items), MedQA-USMLE (77) and **AQuA-RAT
      (141) leak on none**: the benchmark finding 4 holds up because its *value*
      channel is uniform turns out to be closed on every ordering measured here
      as well (−0.9, −1.6, −1.6, −1.6, −0.1).
      A distractor process that never looks at the key closes coordinates nobody
      thought to audit; a repair has to be told about each one, and cannot always
      comply. The coupling is consistent throughout — lexicographic order is the
      value order on 65–70% of the items of every file that leaks on none of the
      four, and on 49–53% of MMLU's and MMLU-Pro's.
    - Two cautions about these four, both measured rather than assumed. *Decimal
      places* is degenerate on released files (all options share a decimal count on
      87% of BixBench's items and 80% of MMLU-Pro's), so there it is largely the
      lexicographic ordering under another name. *Lexicographic order* is tied on
      every item by construction — it scores all options equally and lets
      `feature_rank`'s text tie-break do the ordering, which is its definition.
      *Digit sum*, the one of the four that is neither degenerate nor
      magnitude-driven (7.9 of 10 distinct scores per item), is closed on both
      files before any repair: a held-out ordering can be open, closed, or closed
      by coincidence, and only measuring it says which.

18. **Closing every coordinate the audit measures halves the residue, and does
    not remove it.** At ten options the constructive repair closes all four
    audited coordinates (finding 16), and the five-arm probe agrees only in
    part. The solver that had been collecting falls to chance with the question
    withheld — Qwen2.5-32B **13.4% → 10.6%** against 10% chance — which is the
    repair doing its job on the only solver that was exploiting the released
    file. But the two that had been *at* chance still end above it, **+3.4** and
    **+4.3** points, against +3.7/+9.1 under the value-only repair and
    +3.8/+8.1 under the searching four-ordering one. So the audit is the weaker
    instrument here: it reports every coordinate it measures as closed while a
    solver still gains three to four points.

    **Attribution says where the residue is not.** `channel_attribution.py
    --held-out` measures all nine coordinates — the four audited and the five
    held out — on the repaired file. The audited four leave at most +1.7 points
    available and all three solvers collect within a tenth of a point of zero on
    each. The held-out five are genuinely open there (+7.4, +5.4, +5.1, +1.7,
    +0.9 available) and pay the same solvers between −0.9 and +0.6 points. So a
    residue of three to four points is not option geometry under any ordering
    this repository knows how to measure.

    **The placebo settles it, and no solver is collecting.** Run stem-withheld
    — distractor values redrawn by the same generator with the key's rank *held*
    — the placebo lifts the three solvers **+1.4, +2.9, +5.9** points where the
    repair moves them **−2.8, +3.5, +4.4**. *Placebo minus repair* isolates
    rank, the same contrast finding 8 uses at four options, and after the
    construction it is **−4.2, +0.6, −1.5** points: nothing is being collected
    on the repaired ten-option file. The apparent residue was the *rewriting*. A
    regenerated distractor is easier to dismiss than a curated one, which raises
    a no-data score with no rank being informative — a property of *any* repair
    that rewrites options, and the reason to run the placebo beside the repair
    rather than read the repaired number alone. It also explains the panel's
    shape: the solver that had been collecting loses 4.2 points of rank effect,
    the two that had not lose nothing because they had nothing.

    **Both panels, in full.** `placebo_contrast.py` runs the contrast over 11
    solvers at four options and 10 at ten. The rank effect is negative for 9 of
    11 (median −1.1, largest −7.8) and 8 of 10 (median −0.7, largest −4.2); in
    both the largest belongs to the solver that collected most on the released
    file, and the positives are ≤ +1.4 on solvers the released file left at
    chance. Released accuracy against rank effect gives ρ = **−0.66**
    [−1.00, +0.03] and **−0.54** [−0.96, +0.22] — the predicted direction in
    both, intervals including zero in both, so a suggestion and not a finding.
    The placebo's own lift is what a maintainer cannot skip *or* predict:
    median +1.4 and +2.0 points, but from −6.0 to +5.9 across solvers, and it
    reverses the sign of the rank effect for two of eleven.
    `results/probe_bix_placebo/`, `results/probe_mmlupro_panel/`.
    `no_data_probe.py --arms placebo_stemless`;
    `results/probe_mmlupro_construct/`, `results/probe_mmlupro_placebo/`,
    `results/channel_attribution_mmlupro_construct_heldout.json`.

19. **A learned solver, and the two things the guarantee does not cover.** Every
    channel above is a hand-designed statistic, so `learned_probe.py` fits one
    instead: each option gets a feature vector read off the option set alone and
    a linear score, and the solver takes the arg-max. Rank enters as one
    indicator per (ordering, rank) — never as a number, because "prefer the
    second-smallest" has to be expressible; with rank as one continuous feature
    the learner scores 31% where the hand rule scores 51%. Three nested families
    are fitted, each read against **its own** clean-file control, leave-one-cluster-out.
    On MMLU-Pro (chance 10%, margins over each family's clean worst case):

    | file | audited ranks | all eight orderings | every feature |
    |---|---|---|---|
    | released | 25.7% (+14.5) | 24.9% (+13.3) | 24.8% (+13.5) |
    | constructed repair | **10.6% (−0.6)** | 20.2% (+8.6) | **28.3% (+17.0)** |
    | + targeting lexicographic | 13.2% (+2.0) | 18.7% (+7.1) | 27.8% (+16.5) |

    - **The guarantee holds in its own class.** Restricted to the four audited
      coordinates, the strongest learned solver scores 10.6% on the repaired
      file — 0.6 points *below* what it manages on a file that cannot leak. At
      four options the same cell is 17.1% against a clean worst case of 36.2%.
    - **Column two is finding 17 as accuracy**: allowed all eight orderings the
      learner gets +8.6, with its heaviest weights on decimal places, which no
      repair here targets. Same conclusion as the bounds, reached by a solver
      rather than a test, and cheaper for a maintainer to run.
    - **Column three was not expected.** Given features that are *not* ranks the
      learner does **better on the repaired file than the released one** (28.3%
      vs 24.8%), and its heaviest weight by a factor of two is *relative
      magnitude* — how large an option is against the largest in its set.
      Uniformising rank says nothing about that: the repair perturbs the key to
      make distractors, so the set's spread becomes a function of the key. This
      is the placebo effect of finding 18 seen as a usable feature. No model
      probed uses it (rank effects −4.2/+0.6/−1.4), which is the paper's thesis
      recurring one level up — but the honest scope is that the bound is about
      *rank* preferences, and a repair that rewrites options can open something
      that is not one.

20. **Flat margins are not a flat joint, and a file need not admit both.** The
    first candidate for column three was the rank *tuple*: each margin flat
    while (length, roundness) is not, which the construction could easily cause
    since it draws from a frontier that is not a product set.
    `joint_uniformity.py` measures it with the audit's own estimator and the
    answer is **no** — on the repaired ten-option file the joint reaches 4.4%
    [2.7, 5.9] against 1% flat, but a *clean* ten-option file reaches 6.7% on
    average and 8.0% at worst, because a hundred cells over 1,263 items is a
    noisy thing to maximise over. Read against its control the joint is not
    established as concentrated on either file. A good hypothesis, and not the
    answer. It does name a real choice, though: `mcq_audit.py --joint-draw`
    fits one weight per reachable tuple instead of one per rank, and the two
    trade — at four options the joint goes 16.2% → **2.9%** while the length
    margin goes 25.7% → 37.1%; at ten the joint closes to the 3.9% the fit says
    is attainable and written length reopens at +1.3. Neither file admits both,
    and the frontier says so before either repair runs.

21. **A tie-break is an ordering, and the audit and the rule family had picked
    different ones.** Written length ties: the key is exactly as long as some
    distractor on 93.3% of v1.5's numeric items, 87.3% of MMLU's and 96.2% of
    MMLU-Pro's. A score with ties names not one ordering but a family, one per
    tie-break, and the two halves of this repository had silently chosen
    different members — `mcq_audit.feature_rank` sorts by `(length, text)`
    ascending, so a tied distractor counts as before the key when its text
    sorts earlier; `text_artifacts.pick` takes the arg-max of `(-length, text)`
    and so takes the *later* text. The key's rank under the two differs on
    87.6%, 79.3% and 89.2% of those files' numeric items. Measured as a fifth
    held-out ordering with the audit's own estimator, the shipped
    three-coordinate repair does not close the tie-break it does not target on
    MMLU — released +1.8, repaired +2.6 against a clean floor of +0.6. On the
    repaired MMLU file the key's length rank is flat at **25.0%** and
    `take shortest_option` still finds it **36.6%** of the time (23.8% on the
    released file). It is assignable like the others, the key's rank under it
    still being a count over the distractors, and `--with-tie-break` takes the
    channel to −2.9 and the rule's hit rate to 30.9%. It closes at ten options
    too: on MMLU-Pro the coordinate runs +2.5 released, +1.7 after the shipped
    repair and **−1.6** after the four-coordinate one, with the four audited
    coordinates still shut (−0.6, −0.8, −0.1) and the other four held-out
    orderings still open. Adding it widens each item's reachable set from 29.6
    rank pairs to 146.2 tuples, and the fit says the flattest reversed-tie
    marginal that file admits is 14.4%, not 10% — the channel closes anyway,
    because a 13.8% top share over 1,263 items in 67 clusters is not separable
    from noise. Two things
    had to change: the dynamic program takes a fourth count, and the
    isolation-tuning stage — which re-renders a distractor inside its
    `(shorter, rounder)` class, invisible to the counts but not to a tie-break
    that reads the text — had to treat every assigned coordinate as part of a
    class. Without that it met the new target on 85.6% of items instead of 100%,
    and left the key's widest rank under the coordinate at 33.1% instead of
    30.9% (25% is flat).
    Over eight seeds MMLU's worst-case written-length bound falls +2.9 → +0.2
    and roundness +1.6 → +1.2 while isolation rises −0.1 → +1.1, and on v1.5
    the four-coordinate repair is flagged on 2 seeds against 3. Ten of the
    eleven panel models answered MMLU again under it, 13,220 decisions each, and
    every change is inside ±0.6 points with the question and ±0.9 without it. A
    5.5-point move in what is available is half a point in what is taken.
22. **The isolation statistic changed when the Python version did.** Lexical
    isolation scores an option by minus the sum of its similarities to the
    others. CPython 3.12 changed `sum` over floats to compensated summation, so
    the same four options score −1.5 on 3.12 and −1.4999999999999998 on 3.10;
    the scores are dense with near-ties, and that last bit decides the key's
    isolation rank on 841 of the 38,710 option sets this artifact ships or
    rebuilds (2.2%, and 3.7% of MMLU-Pro's, where ten options make near-ties
    denser). Both sides now use `math.fsum`, which is exactly rounded everywhere
    and agrees with 3.12 on all 38,710 — no published number moves, and the
    repaired files regenerate byte-for-byte under 3.10 and 3.12 alike. A
    benchmark-integrity statistic that moves with the language's minor version
    is a failure mode nobody is looking for; ours would have shipped silently.

23. **The learned solver found the tie-break by itself.** The same three
    nested families on MMLU's 661 four-option numeric items
    (`results/learned_probe_mmlu.json`), margins over each family's clean-file
    worst case in brackets:

    | file | audited ranks | all nine orderings | every feature |
    |---|---|---|---|
    | released | 28.4% [−0.6] | 28.8% [+0.5] | 27.9% [+0.2] |
    | constructed repair | 26.1% [−2.9] | 32.7% [+4.4] | 41.2% [+13.4] |
    | + the reversed tie-break | 25.0% [−3.9] | 31.7% [+3.3] | 41.8% [+14.0] |

    The guarantee's column holds and tightens as coordinates are added. The
    held-out column opens under the repair — and the learner's two heaviest
    weights on the repaired file are **the reversed tie-break's rank 0 and rank
    3** (+0.71, −0.69): given nine orderings and told nothing, it goes straight
    to the coordinate the shipped repair does not target. Repairing that
    coordinate takes those weights out of the top four. The third column is the
    scope statement: a solver reading *relative magnitude* scores 41% on the
    repaired file against 28% on the released one, because regenerating
    distractors by perturbing the key makes the set's spread a function of the
    key, and no repair of the key's *rank* touches that.

24. **A third benchmark, where the released geometry was nearly closed
    already.** MMLU's 661 four-option numeric items carry almost no value
    channel (released bounds −2.9, −1.2, +1.3 on isolation, roundness, length),
    so they are the case where the repair should take nothing away. All eleven
    models, six arms, 13,220 decisions each
    (`results/probe_mmlu_construct/`, the `MMLU` panel of
    `results/placebo_contrast.json`):
    released no-data accuracy runs **21.6–24.7%**, at or below the 25% chance
    line for every solver; the placebo lifts it by a median **+2.5** points; and
    the rank effect is **positive for 8 of 11** (median +0.8). Uniformising a
    rank that was already near-uniform makes models slightly *better*, and the
    whole movement is the rewriting. The three negatives are the three largest
    models (−2.2, −0.7, −0.5 for 32B, 14B, 7B) — the size pattern of finding 9
    again, and again only a suggestion (ρ = −0.14 [−0.76, +0.49]). Beside v1.5,
    where the released file leaks +26.4 points and the rank effect is negative
    for 9 of 11, the pair is the contrast behaving: it takes credit away where
    there was credit and not where there was none.

25. **The residue is not geometry, and removing the geometry does not remove
    it.** Finding 19's third column had a mechanism attached: the repair makes
    distractors by perturbing the key, so the key is the centre of the set and a
    learner reading *relative magnitude* finds it. `exchangeable_repair.py`
    tests that by removing the mechanism — draw `k` i.i.d. offsets and a slot
    `m`, set `v_i = key · exp(σ(z_i − z_m))`, so the key is a uniform draw from
    its own option set and every equivariant statistic, not just every rank, is
    exchangeable between key and distractors. On the 594 MMLU items that admit
    such a rewriting the leak barely moves: **+11.8 → +10.9** points over the
    clean control. What moves is *where* it lives — the heaviest weight goes
    from `relative magnitude` −2.53 to `has separator` +2.56 and
    `trailing zeros` +1.15 (`results/learned_probe_exchangeable.json`).
26. **What is being detected is authorship, and the repair introduces it.**
    `key_identity.py` fits the same solver over features that never refer to
    another option, so a margin it reports cannot be option geometry of any
    kind. On the **released** file it is 11.3 points *below* its clean control —
    MMLU's distractors are written by the same hand as its keys, and there is
    nothing to find. On the rank-repaired file it is level with the control
    (+0.2, +0.3), so that file's +11.8 really is a property of the set, just not
    of any rank. On the exchangeable file it is **+2.5 and +7.4**: the same
    information has moved into the key's handwriting. The two leaks are
    substitutes, and what is conserved is that a *generated* distractor is
    identifiable as generated. The only repair that would not introduce it is
    one that makes the distractors as authored as the key, which is a demand on
    how a benchmark is built rather than on how it is rewritten
    (`results/key_identity_mmlu.json`; the paper's appendix "Who wrote the distractor,
    observed and controlled").
27. **Exchangeability is strictly harder than rank uniformity, and 10% of items
    admit none.** `['3', '1', '2', '4']` cannot be made exchangeable in its own
    written style: putting the key at the top needs three distinct positive
    integers below 3. That is 67 of MMLU's 661 numeric items. On the 594 that
    remain the value channel closes (−0.8 points) but written length, lexical
    isolation and significant digits do not (+1.4, +3.0, +5.8 against the +1.0,
    +0.9 and +1.5 a clean file of that shape reaches —
    `results/audit_calibration_594.json`, remeasured because the bar moves with
    the item count). The key's own value is a round number written in one
    person's style, and no redrawing of the *distractors* changes that.

28. **The instrument decides, and the prompt is nine tenths of the
    instrument.** The no-data baseline is run one way by the field — one letter
    token under the benchmark's own template — and that is one reader among
    many. Holding Qwen2.5-14B, MMLU-Pro's 913 ten-option items, the letter
    orderings and the clean control fixed, and varying only how the question is
    put and how the answer is taken, gives the 2×3 at the top of this file:
    **+1.3 [−1.6,+3.8] under BixBench's verbatim template, +10.4 [+7.1,+14.3]
    in a chat frame that simply omits the question.** The read-out is worth a
    point, the prompt nine. It is not the interpreter (taking it away *raises*
    the margin), not an instruction to read the options (saying so *lowers* it,
    which is the opposite of steerability), and not reasoning (the cell that
    reads highest answers immediately on 99.9% of rollouts). Equation (1) says
    where it lives: `p` is the same throughout — it is one file — and `b`, the
    reader's pick-rank preference, is flat under the template (+0.35
    [−0.31,+0.81]) and peaked at the key's own rank under the framing (+3.15
    [+2.35,+4.25]). **A no-data score without its prompt is not a measurement.**

29. **And the reading needs the published option text, not only the framing.**
    The rank-holding placebo for MMLU-Pro (`build_placebo_arm.py`) regenerates
    every option set and matches the released key-rank law to a tenth of a
    point. Under the cell that reads highest it gives **+2.8 [−0.3,+5.3]**,
    which no longer clears, and its rank term falls to +0.11 [−0.18,+0.57]
    although `p` is held by construction. That rules out the rank law alone
    carrying any of the +10.4. It does *not* separate recognition of option
    sets the model has seen from a channel in the released text, because the
    redraw destroys both — so `option_membership.py` tests recognition
    directly, on all thirteen models and against the control that test needs.
    A membership score is also a *typicality* score, so what recognition
    predicts is not a gap but a gap **larger on the released file than on one
    nobody published**. Of the six models that both answer this arm and beat
    chance on it, that difference clears on **none**. Qwen2.5-32B is why the
    control is there: **+0.034 [+0.018,+0.048]** released, −0.003 control,
    difference **+0.037 [−0.002,+0.079]** — reading the released arm alone
    would have reported a memorisation finding. Counted per interval, which is
    the unit a 95% interval is calibrated on, the difference clears on 8 of 78
    on MMLU-Pro, four positive and four negative. And the two files' controls
    are not the same object: **100%** of BixBench's control strings were
    published and only the grouping was not, against 12.9% of MMLU-Pro's, so
    the venue's own file is the clean test of set recognition — and there the
    released arm clears on **1 of 78**, the control on 4 and the difference on
    **3**, at or below the 3.9 the procedure gives by construction
    (VERIFICATION.md).

30. **On the release the published baselines ran on, the template collects at
    every size but the smallest.** Both zero-shot runs BixBench published are on
    v1.0, whose key-rank law caps a rank rule at +5.8 points. BixBench's own
    template, question withheld, run unchanged on all **296** v1.0 items in 53
    capsules (`bixbench_v10_items.py`, `run_bixv10.sh`, `bixbench_v10_grid.py`)
    and read at 96.0% — the widest level any of the thirteen dumps needs to cover
    95% on those capsules — clears over the control on **12 of 13** models, from
    **+5.3 to +11.7**, and over chance on 11. The half that carries it is not the
    one the rank ceiling describes: the 159 numeric items clear on four, the 137
    whose options are not all numbers on nine, up to +18.0.

31. **The item-writing prescription fails where the writer cannot see the
    data.** `wrong step` — a writer shown the question alone and asked for the
    values plausible mistakes reach — keeps MMLU's items hard. On BixBench's 105
    numeric items (`keyblind_operators.py --jsonl build/bixbench_numeric_q.jsonl`,
    `bixbench_wrong_step.py`) it cannot make the mistake: it produces the key on
    6 items against MMLU's 114 of 635, its values miss the key instead of
    bracketing it (the key at an extreme of its set **60.0%** of the time against
    the released 19.0%), and the arm both leaks more than the released file
    (**+17.1 [+8.4,+26.3]** over the clean worst case, against +12.4) and hands a
    question-given solver **+17.4 to +18.3** points at the level that covers the
    46 capsules, 95.75%. The rank-uniform repair sits at its control there and
    moves no question-given solver.

32. **In BixBench's own pipeline, with the data, the released options give the
    same runs their lowest score — under our protocol and under the published
    agent.** BixBench's agent — data-analysis-crow v1.5.0's prompts and three
    tools, 40 steps, fhda's notebook image — run on all 205 items with each
    capsule's data and without it: for three open families calling its tools in
    text, and for four running it **as published**, ldp's ReAct agent at the
    release BixBench's lock resolves (0.26.0: the reasoning wrapper, and the
    4,096-token reply limit its model client gives), each graded and read by its
    own family (`bixbench_agent.py`, `sandbox_repl.py`, `bixbench_withdata.py`).
    The agent never sees the options, so each finished run is read through the
    released options, a rank-holding placebo and the rank-uniform repair. With
    the data **the released options give the same runs their lowest score for all
    seven run sets**: the repair reads **+3.3 to +10.5** points higher in text
    (Qwen2.5-72B **−10.5 [−16.4,−5.2]**, Llama-3.3-70B −9.5 [−15.7,−3.1]) and
    +3.8 to +8.6 as published (Qwen2.5-72B −8.6 [−15.1,−1.8]), and one model's
    contrast does not differ measurably between the protocols. gemma-3-27b, as
    second reader, and **a rule that reads no notebook** — the option nearest the
    agent's own number — put the released options below the repair for every
    run set in both conditions: BixBench's perturbation distractors sit nearer
    the key than generated ones. What the data is worth does not travel:
    GLM-4.5-Air and Qwen3-30B-A3B, run as published, gain **+21.0 [+10.9,+31.5]**
    and +14.6 [+6.5,+23.3] open-ended from it, and in the published MCQ reading
    the data and the swap move the score by comparable amounts (−0.9 to +10.5
    against +3.8 to +8.6). The published agent's wrapper is copied into the next
    reasoning turn by both Qwen families, which then loop to the reply limit:
    2,090 of Qwen3-30B-A3B's 6,884 reasoning turns, and 98 of its 108 runs that
    end at the step limit contain one. Two readers failed and are reported, not
    used: GLM-4.5-Air on other families' runs, and Qwen3-30B-A3B on its own (no
    pick on half their reads); gemma-3-27b reads Qwen3's runs in the table.
    Every number re-derives from `results/agent_runs/` with no GPU
    (`tests/test_withdata.py`); VERIFICATION.md has the tables.

## Failures worth recording

Most are in the paper, because they are the argument for running the check; the
rest are in the tooling that does the checking.
Twenty of them were found only by later analyses, and each one had already
produced a number worth believing before it was caught.

- **A published agent's release was assumed, not read.** The first
  published-protocol runs vendored ldp 0.25.0 and put the agent's reasoning back
  bare; BixBench's own lock resolves ldp 0.26.0, which wraps it ("Thought: ...
  Based on this reasoning, let's select the appropriate tool!\nAction: "), and
  vLLM's `tool_choice="none"` let the models write their tool calls into the
  reasoning turn. 226 episodes were discarded and every run restarted after both
  fixes. The pins are now read from the lock file and tested
  (`tests/test_published_protocol.py`), the reply limit included.
- **A reader that cannot finish is not a reader.** GLM-4.5-Air, given 1,536
  tokens, was cut off on 44% of its replies and read "no option effect" on
  Qwen2.5-72B's runs — the cut, not the options. Given 4,096 it reads its own
  runs cleanly; on other families' runs it loops at greedy decoding whenever an
  answer matches no option, and gave no pick on half its numeric reads.
  Qwen3-30B-A3B does the same on its own runs at every limit tried. Both are
  reported and not used; gemma-3-27b reads instead. What caught all three is
  now run first on any new reader: its no-pick rate, by option set, on one pass.
- **The anonymous bundle named its authors' machine.** Nine shipped scripts
  carried cache, virtual-environment and checkout paths whose components are an
  institution and an account, and every with-data trajectory recorded the
  absolute paths it was given. Every file had passed the allowlist, which asks
  whether a file belongs and not what it says. Paths now come from environment
  variables (`AGENTICLS_HF_CACHES`, `AGENTICLS_VENV`, `BIXBENCH_CAPSULES`,
  `BIXBENCH_WORK`) or the script's own directory, `bixbench_withdata.py --pack`
  ships settings relative to the repository, and `package_submission.py` refuses
  to build a bundle any member of which names the host — the strings derived from
  the host at build time, since a list of them would ship in the checker.
- **The calibration reached one column of three.** tab:frontier's set column
  was reprinted at the level that covers; the difficulty column — which carries
  the claim that closing the channel costs 11 to 20 points — and the one-option
  column stayed at nominal 95% on the same 4.8 effective clusters. The one-option
  column's single clearing row, `exchangeable`, was kept by a normal widening to
  [+1.0,+12.8]; read off the same draws at the same level it is **[−0.8,+12.8]**
  and no longer clears. The cost claim survives: both closing operators clear at
  every solver. The per-item answers behind the difficulty column had not been
  kept, so `frontier_validity.py` was re-run to keep them — reproducing all 21
  shipped cells to the digit before any of them was re-read.
- **Nothing was corrected across the paper.** Every family was corrected within
  itself and none across the paper. `claim_budget.py` closes that: the ten
  headline findings (the abstract's and the introduction's, and the two no-data
  margins Appendix C rests on) form one family, each read at the
  level covering 1−0.05/10 on its own file, and **eight survive** — the rank rule,
  gpt-4o's forced arm over chance, the repair's gain on the keys it moves to an
  edge with the data, the published runs' accuracy where the key is
  second-smallest falling below a rank reader's, BixBench's forced reading of
  its published runs above the tolerance, its acceptance of the misses nearest
  the key over those nearest another option, the tolerance above the forced
  score corrected for guessing, and three open readers' gains on the
  published runs' moved keys; Claude's forced margin reaches −0.2, and the
  current agents' excess over the tolerance does not clear because gpt-6-luna's
  reaches −1.1. The family is chosen from what the
  paper claims, so it is post hoc, and the paper says so.
- **The calibration that needs no model was the optimistic one.** A
  nonparametric double bootstrap was tried for the budget first. It agrees with
  the parametric simulation on BixBench and has a nominal 95% interval covering
  0.948 on MMLU-Pro where the simulation has 0.903 — resampling sixty labels
  cannot manufacture the file where the subject holding a quarter of the items is
  unusual. The simulation is used throughout.
- **The page checker said "over by 46 lines" for a paper under the limit.**
  `page_budget.py` treated any References heading part way down a page as an
  overshoot; on the last allowed page it means the main text ends there with room
  to spare. It had never happened, because the paper had always been over or at
  the limit.
- **"Two greedy rollouts an item" was true of one file.** Every MMLU-Pro arm
  runs two letter orderings an item and every BixBench arm three; the caption and
  the appendix said two for both. Counted from the dumps and pinned.
- **Coverage scored against the wrong truth reported perfect coverage.**
  `frontier_calibrated.py` measured whether a nominal 95% interval covers, at
  the frontier's own shape — and compared each simulated interval against the
  simulated file's *own mean* rather than the rate it was drawn with. The
  bootstrap is centred on the former by construction, so it reported **100%
  coverage at every level** and concluded the intervals could be made
  *narrower*. Scored correctly, a nominal 95% interval covers **88.1%** and the
  frontier's headline row stops clearing zero. The only reason the bug was
  caught is that a percentile bootstrap over-covering perfectly on 4.8
  effective clusters is not a believable result.
- **A second bootstrap is a second interval.** Recomputing the frontier's
  nominal column with our own RNG put it 0.06 points from the shipped one on
  identical picks; numpy's interpolating `percentile` then differed again from
  `learned_probe`'s index convention. Two intervals for one arm is worse than
  either. Both levels now come off `learned_probe.bootstrap`'s own draws with
  its own indexing, and a test asserts the nominal column reproduces the
  shipped file exactly.
- **A bare-number pin passed on the wrong copy — for the third time.** The
  published-repair accuracies appear in the abstract, the introduction and §3.
  Corrupting §3's copy alone left the pin green, because the abstract's copy
  matched the same substring. Every such pin now keys on a span long enough to
  name its site, and the fix was verified by corrupting each site in turn.
- **An instruction after the options broke option parsing.** BixBench's
  template puts `IMPORTANT: ...` *after* the option block, so a greedy match to
  end-of-string swallowed it, the last option line failed to parse, and all 205
  BixBench items classified as non-numeric. The numeric/text split looked like
  a null.
- **A resampled control cannot be subset by its own text.** Its option sets are
  drawn from the file's pooled strings, so a control row can be numeric where
  the item it stands in for is not. Subsetting by text broke the pairing and
  printed a numeric interval twenty points *wider* than the whole file's;
  subsetting by item index fixed it.
- **A literal `empty` read as a letter.** BixBench v1.0 writes the string
  `empty` in the column naming the decline letter. Reading it as a letter
  reported a decline rate of **0.0%** for an arm that had no decline option —
  a plausible number for a condition that does not exist.

- **A re-run overwrote the evidence for the number it was checking.** The 14B's
  MMLU-Pro arm was re-run on a second card to confirm determinism, at batch 12
  instead of 10, and its `--dump` went to the same path. Greedy decoding is
  deterministic *given* the batching and not otherwise, so the re-run is a
  different set of rollouts (+5.60 against the published +6.25) — and the
  published arm's rollouts were gone. Nothing in the paper changed, because the
  fitted result file was untouched; what broke is the promise
  `arm_intervals.py` makes, that every published interval rebuilds from the
  rollouts shipped beside it. It rebuilt +3.11,+8.58 where the paper says
  +4.04,+8.69 and **printed the disagreement**, which is the check doing its
  job. Every dump path now names its cell and its batch size, and the batch-10
  arm is re-run rather than explained away. **A reproduction that writes to the
  original's path is not a reproduction.**

- **A nominal cluster count is not an effective one.** Every headline interval
  here is a cluster bootstrap, and "913 items in 60 clusters" sounds
  well-powered. MMLU-Pro's clusters are source subjects and one holds 23% of the
  items, so Kish's effective count is **10.7**, and the nominal 95% interval
  covers 90%. Every MMLU-family file in this repository runs 0.17 to 0.26 of its
  nominal count; the synthesised clean controls, balanced by construction, sit
  at 1.00 — so a margin and its control do not have the same resolving power.
  `bootstrap_calibration.py` measures it, `arm_intervals.py` reprints the arms at
  the level that covers, and the conclusions survive.
- **A range reported as a null.** The five-arm probe was summarised as "every
  model between −7.8 and +3.0", which reads as nothing happening. Three of
  eleven solvers clear zero. None of the three is rank: the decomposition
  predicts a 1.3-point placebo–repair gap for the 32B against the 7.8 observed,
  because assigning the key a rank constrains which values can be drawn and so
  how they are written. An endpoint is not a finding until someone says which
  solver it is and whether it clears.
- **Comparing two instruments that differ in three things.** "Single-token
  probes read nothing, the agent reads +6.3" contrasts numbers from different
  prompts, different chat templates and different read-outs. Reading the
  no-interpreter dump caught it: 90% of rollouts turn out to be the final line
  and nothing else, so "the agent reasons" was never what separated them.
  `--argmax` holds the prompt fixed and changes only the read-out.
- **Two releases of one benchmark need not share a key-rank law.** BixBench's
  published no-data runs exist for v1.0 alone. Its key-rank ceiling is +5.8
  points where the current release's is +26.0, and on the 39 option sets both
  carry the key is the same one on 37 — the items changed, not the keys. The
  published baselines were run against a channel a fifth as wide as the one
  shipped now.
- **Reading the first capital letter after `FINAL:`** takes the T of "FINAL: The
  answer is B". Models that answer in a sentence then score near zero, and the
  arm reports a *compliant* model as one with a violent option preference:
  Llama-3.1-8B read −16.1 points before the parser was fixed and +0.1 after.
  `final_letter` now reads a parenthesised letter first, then one introduced by
  "option"/"answer"/"choice", then the last standalone letter in the clause, and
  only ever a letter the item actually has.
- **A difference of two bootstraps in the wrong units.** The per-arm margins are
  in percentage points and the raw bootstrap draws are fractions, so the first
  difference interval came out a hundred times too narrow — and, being centred
  on zero, looked exactly like the null it was supposed to test. The point
  estimate falling *outside* its own interval is what gave it away.
- **Common random numbers across two unrelated files.** Seeding the file arm's
  cluster bootstrap and the clean arm's with the same value pairs cluster *k* of
  one with cluster *k* of the other, which are unrelated, and the paired
  difference cancels variation that is not shared. The two arms now use
  different seeds.
- **Random folds on a benchmark built in templated subtasks.** LAB-Bench SeqQA
  is fourteen subtasks of forty items and DbQA ten of fifty-two, and an item's
  neighbours inside a subtask share a template. Binning items at random for
  cross-validation lets the reader learn the template on one half and apply it
  to the other: SeqQA reads **+21.5 [+17.7, +25.1]** that way and **−7.2** when
  the subtasks are held out, DbQA **+5.6** and **−8.7**. Both naive readings
  clear their control. Neither is a channel.
- **A margin with no mechanism is not yet a finding.** The agent's +11.1 was
  reproducible, interval-bounded and controlled for a good while before anything
  here could say what produced it, and in that state it was indistinguishable
  from a 14B model recognising a public option set. What settled it was dumping
  the rollouts: splitting them on whether the agent ran code (+12.3 against
  +5.5), on where its pick lands among the sorted values (b unchanged across
  three different files), and on its own wording (45.2% against 29.8% on the
  released file, nothing on the clean one). The rule is now that an arm which
  clears its control ships its rollouts too.
- **Two jobs on one CUDA device kill each other, quietly.** Starting the 14B
  agent on `cuda:2` killed a `text_options.py` run already there; the log simply
  stopped after its last banner and a half-written result file stayed on disk.
  NVML is broken on this machine, so `nvidia-smi` cannot be used to check first
  — `torch.cuda.mem_get_info(i)` per device can.
- **Redrawing a colliding distractor against the key** is a dependence on the
  key. Drawing distractors from the key marginal and re-drawing whenever one
  equals the key — the loop most assemblers write, ours included — removes the
  key's own mass from the pool and tilts the posterior: worth up to +20.6 points
  on a skewed marginal where the same draw conditioned on the whole tuple is
  worth exactly zero.

- The **first repair** reformatted the distractors, which made the keyed answer
  lexically distinctive and drove a surface family to 60.0% — the same trade of
  one leak for another that the v1.5 revision made, caught in minutes because
  the check was run. Fixed by copying each replaced option's notation,
  precision, sign and separators, and verifying the *achieved* rank after
  rendering.
- The **second repair** assigned ranks from a balanced multiset. That flattens
  the histogram, but a rank over-represented in a held-out capsule is by
  construction depleted in the training capsules, so leave-one-cluster-out
  selection picks the rank about to become scarce and the audit read fifteen
  points *below* chance. The shipped repair draws independently and states its
  guarantee on the distribution, not on one realisation.
- The **third repair is the one we shipped.** Copying each replaced option's
  style — the fix for the first failure — is not enough at ten options: the
  regenerated values still leave the key at a distinctive *written length*, a
  coordinate the numeric repair never looks at. Solvers ignored that channel on
  the released file and collected it on the repaired one (9.3% → 18.4% with the
  question withheld). Caught not by inspection but by running the probe on our
  own output, which is the only reason we know about it. The response is
  finding 12: draw a uniform target rank for every ordering the audit measures.
- The **fourth failure was trying to make that repair better at its own job.** A
  descent that meets two to three times as many targets leaves a worse file
  (finding 15), and the first fix — tilting the target draw once against the
  realised histogram — over-corrected, because changing the targets changes
  which of them the search can meet. The shipped tilt is three damped passes,
  and it was the audit, not the hit rate, that said so.
- The **fifth was the first constructive repair.** It closed written length and
  roundness exactly and pushed the leak onto lexical isolation (+3.6 points):
  pinning two coordinates over-determines the renderings and leaves the third no
  freedom. Fixed by the class-exchange move, which both counts cannot see. A
  smaller sibling: the construction fell back to a value-only redraw on the
  first value rank whose frontier was empty instead of trying the others, which
  sent a fifth of ten-option items through a repair that closes nothing but
  value and brought written length back at +4.8 points.
- The **sixth was in a statistic, not a repair.** `significant_digits` stripped
  `+` and `-` from a numeral but not U+2212 MINUS SIGN, which `parse_number`
  accepts — so every option written with it read one digit less round than it
  is: 65 options over 24 of MMLU-Pro's 1,263 numeric items, enough to move those
  items' roundness rank. Found while writing finding 17, because a held-out
  ordering crashed on the character the audited one had been silently
  miscounting. Fixed, regression-tested, and every MMLU-Pro number here and in
  the paper recomputed; BixBench has no such option and is unchanged. A
  coordinate you never look at from a second angle is a coordinate you are
  trusting.
- The **seventh was that the shipped results did not regenerate from the shipped
  code.** Re-running `mcq_audit.py` with the documented arguments produced a
  BixBench repair differing from the released one on 91 of 205 rows. The cause
  was not the repair: `isolation_scores` had been changed to compare each pair
  of options once and use the value for both directions, because
  `SequenceMatcher` is not symmetric — `1.90E-06` against `0.0004` scores 0.43
  one way and 0.29 the other — and the earlier version gave two options
  different similarities *to each other*. The fix is right; the results files
  predated it. Every number that depends on lexical isolation has been
  recomputed, and `validate_artifact.py` now re-runs the audit on the released
  repaired file and compares it to the shipped JSON, so a results file that has
  drifted from the code fails the check instead of being read.
- The **eighth was that the audit and the rule family read different orderings
  and neither knew it.** Written length ties on most items, so a length score
  names a family of orderings, one per tie-break; `mcq_audit.feature_rank`
  breaks ties on the earlier text and `text_artifacts.pick` on the later, and
  the two ranks differ on 79–89% of the numeric items in these files. The
  shipped repair flattens one member of the family and leaves the other open
  (MMLU +1.8 released, +2.6 repaired). Found by measuring the repaired file
  under a rule family we had written ourselves. Finding 21 is the fix and what
  it costs. Fixing it also fixed something we had already written down as a
  finding: the isolation-tuning stage re-rendered distractors inside their
  `(shorter, rounder)` class, invisible to the two counts and visible to any
  ordering that reads the text, so it had been quietly undoing the
  *lexicographic* assignment on 3.3% of items too. What we had reported as a
  file over-determined by five counts was a stage of our own repair working
  against the assignment.
- The **ninth was arithmetic.** `isolation_scores` summed similarities with
  `sum`, and CPython 3.12 changed `sum` over floats to compensated summation, so
  the audit's own channel measurement depended on the minor version of the
  interpreter — 841 of 38,710 option sets change the key's isolation rank
  between 3.10 and 3.12. `math.fsum` on both sides fixes it without moving a
  published number. Found by running the documented command on a second
  interpreter, which nothing in the test suite had ever done.

## Layout

| Path | Contents |
|---|---|
| `main.tex`, `refs.bib`, `neurips_2026.sty` | manuscript and official template |
| `submission/` | not in the public repository; `package_submission.py --camera-ready` writes the PDF and the two ZIPs there |
| `data/derived/`, `restore_build_inputs.py` | the item and option files the analyses read under `build/`, with their SHA-256; the script checks them and puts them back |
| `LICENSE`, `LICENSE-DATA` | Apache-2.0 for the code; CC BY 4.0 for results, derived files, figures and `provenance/` |
| `THIRD_PARTY_NOTICES.md`, `data/NOTICE`, `MODEL_OUTPUTS.md` | each third-party file's source, revision, license and changes; the models whose outputs ship and their terms |
| `CITATION.cff` | how to cite the paper |
| `make_public_export.py` | writes this repository's tree from the development one, by allowlist, and scans it |
| `tests/test_release.py` | what a fresh clone needs: `data/derived/` against its checksums, a restore that yields every file the tests read, scripts that stop where they used to fall back silently, and the public export's allowlist and scan |
| `choice_model.py` | **the decomposition**: EM fit, bounds, bootstrap, tests |
| `channel_survey.py` | **how open the channel is across 18 benchmark files** |
| `fetch_survey.py` | vendors the option sets the survey reads |
| `no_data_probe.py` | the five-arm probe (the only step needing a GPU) |
| `probe_analysis.py` | paired, cluster-robust analysis of the probe (stdlib) |
| `rule_solvers.py` | model-free solvers as positive and negative controls |
| `channel_attribution.py` | **what each ordering offers, and what a solver takes** |
| `icl_probe.py`, `icl_analysis.py` | can the channel be learned from examples? |
| `scoring_validation.py` | forced-choice read-out vs. what models actually write |
| `mcq_audit.py` | **portable leakage linter + repair** for any MCQ benchmark |
| `audit_calibration.py` | **what the audit says about files that cannot leak** |
| `repair_search_ablation.py` | **three searches for one repair**, and why the worst wins |
| `held_out_orderings.py` | **five orderings no repair targeted**, the coupling that predicts transfer, and the same five across the survey |
| `learned_probe.py` | a **learned** rank-only solver, and what it finds that hand-designed statistics miss |
| `placebo_contrast.py` | **rank effect vs rewriting effect**, stem-withheld, over both panels |
| `exchangeable_repair.py` | a repair that makes the key a **uniform draw from its own option set**, and the items that admit none |
| `key_identity.py` | a solver that reads **one option at a time**: what is left when no feature is a property of the set |
| `writer_theory.py` | **the three propositions**: enumerated laws for Γ = 0, the two characterised writer classes shown to be the whole solution space (no independence assumed), the distinctness trap, and Proposition 3's price measured against the arm it predicts |
| `repair_frontier.py`, `keyblind_operators.py` | the seven distractor-writing operators, ordered by what the writer may see |
| `frontier_align.py`, `frontier_validity.py` | cut every arm to the common items, and score each one's cost with the question given |
| `frontier_robustness.py` | the frontier **without its two dominant subjects**, and WRONG STEP split on whether the writer ever produced the key |
| `agentic_probe.py` | **the no-data baseline run as an agent**: multi-turn, holding a Python interpreter, against a matched clean file |
| `agentic_mechanism.py` | the agent's own reasoning split on whether it reaches for the middle of the option set, scored on both files |
| `bootstrap_calibration.py` | **does the cluster bootstrap cover?** simulated from each arm's own cluster sizes and spread; reports Kish's effective cluster count |
| `arm_intervals.py` | every agent arm reprinted at the level that actually covers 95%, reproducing each published interval on the way |
| `rank_effect_intervals.py` | a paired interval on each single-token solver's repair-minus-placebo, which is what named the probe's extreme reading |
| `build_placebo_arm.py` | a rank-holding placebo for any released file: new option text, the released key-rank law |
| `build_set_control.py` | a clean control for a whole file, options and all, drawn from the file's own pool of option strings |
| `release_drift.py` | **the two BixBench releases do not share a key-rank law**: the published baselines were run against a channel a fifth as wide as the one shipped now |
| `text_options.py` | the same readers on the **text** options of BixBench and LAB-Bench, against a control resampled from each file's own option pool |
| `rank_dependent_fit.py` | what the goodness-of-fit failure costs: recall freed per key rank, and the geometry term refitted |
| `learned_probe_nonlinear.py` | the same measurement with **no feature and no scorer this repository designed**: hashed character n-grams, a set-invariant context, a small network |
| `subset_guarantee.py` | **a rank guarantee measured whole and subsetted**, which is where two numbers in the paper disagreed |
| `baseline_power.py` | **how many items a no-data baseline needs** before it can detect anything |
| `published_baselines.py` | BixBench's own two zero-shot runs, refitted with a boundary-corrected test of λ = 0 |
| `temporal_recovery.py` | **the dates the releases do not publish**, resolved from the identifiers they do |
| `temporal_test.py` | the temporal test itself, on the file the recovery makes it possible for |
| `survey_to_jsonl.py` | a vendored survey option set in the shape the audit reads |
| `benchmark_census.py` | provenance census across benchmarks |
| `no_data_baseline.py` | published no-data runs, clustering, confounding analysis |
| `option_artifacts.py` | numeric option-set geometry, cross-validation, controls |
| `text_artifacts.py` | surface-rule family, and its overlap with value rank |
| `audit_provenance.py` | BixBench provenance audit (standard library only) |
| `power_analysis.py` | noncentral-*t* planning on measured variance |
| `temporal_manifest.schema.json`, `temporal_eligibility.py` | TATE schema and checker |
| `fetch_external.py` | re-performs retrieval of every vendored file |
| `data/bixbench.jsonl` | pinned question file, upstream Apache-2.0 |
| `data/external/` | vendored published runs and benchmark metadata |
| `data/external/PROVENANCE.json` | URL + SHA-256 for every vendored file |
| `data/external/survey/` | option sets of 8 public MCQ benchmarks, gzipped |
| `results/` | all generated results, one JSON per analysis |
| `results/probe_raw/`, `results/probe_rules/` | per-decision probe outcomes |
| `results/probe_mmlu/` | the same arms on MMLU, for the λ validation |
| `results/probe_mmlu_construct/` | MMLU as a third benchmark: six arms, the whole panel |
| `results/probe_mmlu_tie/` | the same models on the four-coordinate repair (finding 21) |
| `results/probe_mmlupro*/` | ten options: released, and three repairs of it |
| `results/probe_bix_four/`, `results/probe_bix_nearest/`, `results/probe_bix_construct/` | four orderings repaired, by the sampler, the descent and the construction |
| `results/probe_icl/` | in-context exposure, 0 to 64 solved examples |
| `published_repair.py` | BixBench's own three published no-data arms: options forced, options plus a decline, and options deleted |
| `cell_split.py` | every cell read against chance beside its control, and split by whether the options are numbers |
| `option_membership.py` | Min-K%/Min-K%++ on the option sets a rollout read, against its own correctness |
| `frontier_calibrated.py` | the writer frontier's table (not the paper's tab:frontier, which `frontier_agents.py` prints) at the level that covers at 4.8 effective clusters — all three columns, each reproducing its own script's nominal interval first — and the difficulty column for two 70B-class solvers beside it (`--large`) |
| `frontier_validity.py` | the difficulty column: three solvers given the question, per-item answers kept (`results/frontier_validity_items.json`) |
| `claim_budget.py` | **the paper-wide error budget**: the ten headline findings (the abstract's and the introduction's, and the two no-data margins Appendix C rests on), as one family, each read at the level covering 1−0.05/10 on its own file (the paired with-data claims, the forced reading's excess over the tolerance and the readers' gains on the published runs by a double bootstrap over their capsules' paired gains) |
| `bracketing.py` | **the trade under BixBench's agent**: the nearest-option reading split by what the repair did to each key's rank (moved to an edge, moved inward, kept), each group at the level covering its own capsules; the newly credited answers' distance from the key and side; the placebo contrast; the rewritten distractors' distance from the key (`--latex` prints tab:bracket and tab:mechanism) |
| `rank_attribution.py` | **whether a no-data reader reads the rank**: BixBench's published forced runs split by the key's rank against what a reader drawing its margin from the rank would show; the open models' forced cell through the placebo and repaired options; thirteen question-withheld readers' picks by rank (`--latex` prints tab:attribution and tab:rewritten) |
| `run_free_rewritten.sh` | the deleted-option arm's forced cell through the placebo and repaired option sets, six open models (GPU) |
| `PREREGISTRATION.md` | **the registered replication**: data, option sets, groups, reading, statistics and pass criteria, pushed before its analyses ran; results and deviations appended below the plan, which `validate_artifact.py` checks is byte-identical to the registered one |
| `replication.py` | the registered analysis: `build` extracts BixBench v1.0's published with-data runs of gpt-4o and Claude 3.5 Sonnet from `eval_df.csv` (md5-checked) into `results/bixbench_v10_published_runs.json.gz`; `analyse` tests H1-H6 on them and on the new v1.5 run sets; `latex` prints tab:replication |
| `repair_seeds.py` | the repair redrawn at many seeds, the same frozen runs re-read through each draw (exploratory) |
| `writer_panel.py` | eight writers on BixBench's numeric items (six on v1.0), each on the trade's two axes: the rank rule's held-out leak, and the share of with-data misses its options credit (`--latex` prints tab:panel) |
| `tolerance_check.py` | the 5% tolerance against BixBench's own range keys and graders, and every tolerance-dependent number restated at 1% and 10% (`--latex` prints tab:tolerance and tab:restated) |
| `answer_numbers.py` | **how a number is read from an answer**, for every analysis: the last number for the tolerance (an inclusive bound, absolute against a key of zero), scientific notation as one number, no digits from identifiers or list markers, a per cent sign read on the answer or the key; and the nearest-option distance, compared on a log scale (standard library only) |
| `tests/test_answer_numbers.py` | pins `answer_numbers.py`'s readings and the camera-ready fixes that rest on them |
| `cost_table.py` | tab:cost: what hiding the key's rank costs each kind of grader on each set of runs, read from the results the appendix tables print, nothing computed afresh |
| `bixbench_agent.py`, `sandbox_repl.py`, `sandbox_image/` | **BixBench's published agent, with the data and without it**: fhda v1.5.0's prompts and three tools, 40 steps at temperature 1.0, every cell run in fhda's pinned notebook image with no network and the capsule read-only. `--protocol react` is the agent `generate_trajectories.yaml` names, ldp's ReActAgent at the release fhda v1.5.0 pins (0.26.0): two calls a step (reasoning with no tool and its own call tag banned, put back in ldp's "Thought: ... Action: " wrapper, a "Continue..." turn, then one forced tool call in the model's own tool-call format), fhda's tools passed natively, ldp's prompt, stop strings and five attempts a step; `--protocol text` is the first runs' one-call text protocol |
| `tests/test_published_protocol.py` | pins `--protocol react` to the vendored upstream files, parsed as text and never run: the release BixBench runs (fhda's `ldp==0.26.0` pin and BixBench's `uv.lock`) and the 4,096-token reply limit its fhlmi gives every call, ldp's ReAct prompt character for character, its wrapper around the reasoning, fhda's tool schemas from their docstrings and signatures, `DataAnalysisEnv.reset`'s message order, fhda's tool replies, notebook view and output limit, ldp's hidden-state placeholder; and a fake server and sandbox check that a step is two calls, the reasoning call banning the tool-call tag vLLM's `"none"` does not, with exactly one tool call reaching the context |
| `tests/test_one_driver.py` | `bixbench_agent.py` holds one driver per run and one claim per episode, never replaces a finished result, and records a content-filtered episode once: the two failures of DeepSeek-V4-Pro's first runs |
| `bixbench_withdata.py` | the two readings BixBench publishes of each trajectory — open-ended (the v1.5 graders, plus a strict numeric grade and the free-response table's 5% tolerance) and `MCQ_EVAL_PROMPT`, forced and with the refusal option — the latter through the released, placebo and repaired option sets **under the same trajectory**, so a difference is the option set's. Each family reads its own runs (Qwen3-30B-A3B's are read by gemma-3-27b, `PRIMARY_READER`); `--reader` repeated adds a second reader; `nearest_is_key` is the reading with no reader (the option nearest the agent's number), and `reply_limit_census` counts the published agent's wrapper loop. Every call is cached, so `--summarise-only` over the shipped rows re-derives `results/bixbench_withdata.json` with no server, and `--latex` prints both with-data tables' rows |
| `frontier_agents.py` | **two current agents** with the data: gpt-6-luna and DeepSeek-V4-Pro as BixBench's published agent through Azure AI Foundry, graded by gpt-4o as gpt-5.1's runs are (Kimi-K3, run too, is reported and left out); DeepSeek-V4-Pro's runs are its one-driver rerun of 1 October, its first runs kept in `results/frontier_agents/superseded/`; reads the shipped rows, and `--latex` prints tab:frontier's rows |
| `bixbench_v10_items.py`, `run_bixv10.sh`, `bixbench_v10_grid.py` | the grid's template cell on **BixBench v1.0**, the release both published zero-shot runs used, split numeric/non-numeric and read at the level that covers |
| `bixbench_wrong_step.py` | WRONG STEP on **BixBench's** numeric items: rank law, set leak and question-given difficulty, the frontier's three readings |
| `lambda_relaxed.py` | λ̂ under the per-rank relaxation, split by whether the draw rejects |
| `run_temporal_rest.sh` | the temporal test on TableQA, FigQA and SuppQA as well as LitQA2 |
| `sources/cited/` | verbatim excerpts, with their URLs and retrieval date, of the three sources the paper cites for BixBench's use in 2026 (Weidener et al.'s system, BixBench-Verified-50's dataset card, Zhang's preprint); `validate_artifact.py` checks the quoted numbers against them |

## Reproduce

Python 3.12 or later (`requirements.txt` pins the environment the results were computed in;
`extraction_verdicts.py` does not parse on 3.11). **Every statistic reproduces without a GPU**: `no_data_probe.py`
writes per-decision outcomes to `results/probe_raw/`, and those are committed,
so `probe_analysis.py` and `choice_model.py` re-derive the numbers from the
standard library alone. The census, audit, leakage and eligibility analyses are
also stdlib-only. SciPy supports the power calculation, Matplotlib the figures,
`jsonschema` the manifest validation. The manuscript was compiled with Tectonic
0.15.0.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python restore_build_inputs.py   # the derived item and option files, back under build/

python3 audit_provenance.py --resolve-dates --offline   # BixBench provenance
python3 benchmark_census.py                             # cross-benchmark census
python3 option_artifacts.py                             # numeric option geometry
python3 text_artifacts.py                               # surface rules + rank overlap
python3 no_data_baseline.py                             # published runs + confounding
python3 rule_solvers.py                                 # model-free controls
python3 probe_analysis.py                               # the five-arm experiment
python3 probe_analysis.py --raw results/probe_rules \
        --output results/no_data_probe_rules.json       # the controls
python3 choice_model.py                                 # the decomposition
python3 channel_survey.py                               # 18-file channel survey
python3 mcq_audit.py --repair results/repaired/bixbench_v15_repaired.jsonl
python3 mcq_audit.py --two-channel --search construct \
    --repair results/repaired/bixbench_v15_repaired_multi.jsonl \
    --output results/mcq_audit_multi.json       # every ordering, not just value
python3 audit_calibration.py --replicates 60            # the audit on clean files
python3 audit_calibration.py --replicates 30 --items 1263 --clusters 67 \
    --n-options 10 --output results/audit_calibration_k10.json
python3 repair_search_ablation.py --seeds 8              # three searches, one repair
python3 survey_to_jsonl.py --survey mmlu_pro --out build/mmlu_pro.jsonl
python3 mcq_audit.py --jsonl build/mmlu_pro.jsonl --cluster-field cluster \
    --n-options 10 --two-channel --search construct \
    --repair build/mmlu_pro_repaired.jsonl \
    --output results/mcq_audit_mmlu_pro_multi.json      # the ten-option file
python3 mcq_audit.py --jsonl build/mmlu_pro.jsonl --cluster-field cluster \
    --n-options 10 --two-channel --search construct --with-lexicographic \
    --repair build/mmlu_pro_lex.jsonl \
    --output results/mcq_audit_mmlu_pro_lexicographic.json  # a fifth coordinate
python3 held_out_orderings.py --replicates 20 \
    --repaired-4 results/repaired/bixbench_v15_repaired_multi.jsonl \
    --released-10 build/mmlu_pro.jsonl \
    --repaired-10 build/mmlu_pro_repaired.jsonl \
    --lexicographic-10 build/mmlu_pro_lex.jsonl         # orderings nobody targeted
python3 held_out_orderings.py --survey-sweep \
    --output results/held_out_survey.json               # the same five, 5 files
python3 survey_to_jsonl.py --survey mmlu --out build/mmlu.jsonl
python3 mcq_audit.py --jsonl build/mmlu.jsonl --cluster-field cluster \
    --two-channel --search construct --calibrate --seed 20260920 \
    --repair results/repaired/mmlu_repaired_multi.jsonl.gz \
    --output results/mcq_audit_mmlu.json                # 661 numeric, 39 subjects
python3 mcq_audit.py --jsonl build/mmlu.jsonl --cluster-field cluster \
    --two-channel --search construct --calibrate --with-tie-break --seed 20260920 \
    --repair results/repaired/mmlu_repaired_tie.jsonl.gz \
    --output results/mcq_audit_mmlu_tie.json            # the reversed tie-break
python3 held_out_orderings.py --option-counts 4 --replicates 20 \
    --released-4 build/mmlu.jsonl --cluster-field-4 cluster \
    --repaired-4 results/repaired/mmlu_repaired_multi.jsonl.gz \
    --output results/held_out_mmlu.json                 # five orderings held out
python3 held_out_orderings.py --option-counts 4 --replicates 20 \
    --released-4 build/mmlu.jsonl --cluster-field-4 cluster \
    --repaired-4 results/repaired/mmlu_repaired_tie.jsonl.gz \
    --output results/held_out_mmlu_tie.json             # what that coordinate cost
python3 mcq_audit.py --jsonl build/mmlu_pro.jsonl --cluster-field cluster \
    --n-options 10 --two-channel --search construct --calibrate --with-tie-break \
    --repair results/repaired/mmlu_pro_repaired_tie.jsonl.gz \
    --output results/mcq_audit_mmlu_pro_tie.json        # the same at ten options
python3 held_out_orderings.py --option-counts 10 --replicates 20 \
    --released-10 build/mmlu_pro.jsonl \
    --repaired-10 results/repaired/mmlu_pro_repaired_tie.jsonl.gz \
    --output results/held_out_mmlu_pro_tie.json
python3 placebo_contrast.py \
    --probe-dir results/probe_bix_placebo --label "BixBench v1.5" --n-options 4 \
    --probe-dir results/probe_mmlupro_placebo,results/probe_mmlupro_panel \
        --label MMLU-Pro --n-options 10 \
    --probe-dir results/probe_mmlu_construct --label MMLU --n-options 4
python3 learned_probe.py --jsonl data/bixbench.jsonl --label released \
    --jsonl results/repaired/bixbench_v15_repaired_multi.jsonl --label repaired \
    --n-options 4 --replicates 20                       # a learned rank solver
python3 learned_probe.py --jsonl build/mmlu_pro.jsonl --label released \
    --jsonl build/mmlu_pro_repaired.jsonl --label repaired \
    --jsonl build/mmlu_pro_lex.jsonl --label repaired+lexicographic \
    --n-options 10 --cluster-field cluster --replicates 20 \
    --output results/learned_probe_mmlupro.json         # the same at ten options
python3 learned_probe.py --jsonl build/mmlu.jsonl --label released \
    --jsonl results/repaired/mmlu_repaired_multi.jsonl.gz --label repaired \
    --jsonl results/repaired/mmlu_repaired_tie.jsonl.gz --label repaired+tie \
    --n-options 4 --cluster-field cluster --replicates 20 \
    --output results/learned_probe_mmlu.json            # and on the third file
python3 exchangeable_repair.py --jsonl build/mmlu.jsonl --n-options 4 \
    --cluster-field cluster --only-feasible \
    --restrict build/mmlu.jsonl build/mmlu_matched_released.jsonl \
    --restrict results/repaired/mmlu_repaired_multi.jsonl.gz \
        build/mmlu_matched_repaired.jsonl \
    --output build/mmlu_exchangeable.jsonl \
    --report results/exchangeable_mmlu.json     # a repair with no geometry in it
python3 learned_probe.py \
    --jsonl build/mmlu_matched_released.jsonl --label released \
    --jsonl build/mmlu_matched_repaired.jsonl --label "repaired (rank)" \
    --jsonl build/mmlu_exchangeable.jsonl --label "exchangeable offsets" \
    --n-options 4 --cluster-field cluster --replicates 20 \
    --output results/learned_probe_exchangeable.json    # does the leak follow?
python3 audit_calibration.py --replicates 20 --items 594 --clusters 37 \
    --n-options 4 --output results/audit_calibration_594.json   # the matched bar
python3 held_out_orderings.py --option-counts 4 \
    --released-4 build/mmlu_matched_released.jsonl \
    --repaired-4 build/mmlu_exchangeable.jsonl --cluster-field-4 cluster \
    --output results/held_out_exchangeable.json   # and the five held out
python3 key_identity.py \
    --jsonl build/mmlu_matched_released.jsonl --label released \
    --jsonl build/mmlu_matched_repaired.jsonl --label "repaired (rank)" \
    --jsonl build/mmlu_exchangeable.jsonl --label "exchangeable offsets" \
    --n-options 4 --cluster-field cluster --replicates 20 \
    --output results/key_identity_mmlu.json     # one option at a time
python3 channel_attribution.py --probe-dir results/probe_bix_four \
    --jsonl data/bixbench.jsonl --draws 20 --n-options 4 \
    --output results/channel_attribution_bix_four.json
python3 channel_attribution.py --probe-dir results/probe_mmlupro_construct \
    --jsonl build/mmlu_pro_numeric.jsonl --draws 8 --n-options 10 --held-out \
    --output results/channel_attribution_mmlupro_construct_heldout.json
python3 icl_analysis.py                                 # in-context exposure
python3 power_analysis.py                               # planning on measured variance
python3 subset_guarantee.py          # a rank guarantee, whole file and subsetted
python3 baseline_power.py --reps 500    # how many items the baseline needs
python3 published_baselines.py       # BixBench's own two runs, refitted and tested
python3 temporal_recovery.py         # the dates the releases did not publish
python3 make_manifest.py
python3 make_figures.py

python -m pytest tests
python3 validate_artifact.py                            # paper numbers vs results

mkdir -p build
tectonic --reruns 2 --keep-logs --keep-intermediates --outdir build main.tex
python package_submission.py --camera-ready   # the PDF and both ZIPs, into submission/
```

Re-vendoring the survey's option sets needs `pyarrow` (`pip install -r requirements-fetch.txt`;
only to read Hugging Face parquet); the gzipped JSONL under `data/external/survey/` is committed, so
`channel_survey.py` runs without it:

```bash
python3 fetch_survey.py                # optional: re-download the option sets
```

Regenerating the raw probe outcomes needs one GPU and `pip install -r requirements-gpu.txt` (its comments give the other
environments some runs used):

```bash
python3 no_data_probe.py --models meta-llama/Llama-3.2-1B-Instruct ... --draws 20
python3 no_data_probe.py --repair-mode multi --arms original_stemless \
    repaired_stemless ...                # does the repair leave a solver at chance?
python3 icl_probe.py --models ... --shots 0 8 32 64     # can the channel be learned?
python3 scoring_validation.py          # read-out vs. greedy generation

# the seventh operator: a writer shown the question and never the key
python3 keyblind_operators.py --operator wrong_step \
    --model Qwen/Qwen2.5-14B-Instruct --device cuda:0 --tag wrong_step
python3 keyblind_operators.py --operator wrong_step --tag wrong_step_strict \
    --on-collision drop-item             # the same arm with no value filtered
python3 keyblind_operators.py --operator imitation --tag imitation_llama \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct --max-new-tokens 48
python3 frontier_align.py                # cut all seven to their common items
python3 frontier_validity.py --arm ... --model ...   # the cost axis

# a reader with no feature this repository designed
python3 learned_probe_nonlinear.py --jsonl build/mmlu_matched_released.jsonl \
    --label released ... --extractor chars --context set --scorer mlp \
    --replicates 20 --device cuda:0 --output results/nonlinear_mmlu_chars_set.json

# the temporal test, on the file whose dates were recovered
python3 temporal_test.py --jsonl data/external/labbench/LitQA2.jsonl \
    --benchmark "LAB-Bench LitQA2" --model ... --device cuda:0

# the three propositions: enumerated exactly, then priced against the arm
python3 writer_theory.py --exact
python3 writer_theory.py --empirical --model Qwen/Qwen2.5-7B-Instruct \
    --device cuda:0 --out writer_price_qwen7b.json

# the frontier without its two dominant subjects, and the collision split
python3 frontier_robustness.py --holdout --collision

# the no-data baseline run as an agent rather than as one letter token
python3 agentic_probe.py --items build/bixbench_numeric_q.jsonl \
    --clean build/bixbench_numeric_clean.jsonl --condition withheld_aware \
    --model Qwen/Qwen2.5-14B-Instruct --device cuda:3 --batch-size 10 \
    --dump results/agentic_dumps/qwen14b_withheld_aware.jsonl.gz \
    --out agentic_bixbench_qwen14b.json

# the instrument decomposition: same model, same file, same items and orderings,
# varying only how the question is put and how the answer is taken
for prompt in "--prompt bixbench" ""; do
  for readout in "--argmax" ""; do
    python3 agentic_probe.py --items build/mmlu_pro_matched_released.jsonl \
        --clean build/mmlu_pro_matched_clean.jsonl --condition withheld \
        --model Qwen/Qwen2.5-14B-Instruct --device cuda:0 --no-tools \
        $prompt $readout --out agentic_arms_qwen14b.json
  done
done
python3 arm_intervals.py   # each cell at 95% and at the level that covers 95%
python3 frontier_calibrated.py   # the writer frontier, every column at the covering level
python3 bracketing.py     # the moved-key split and the mechanism (tab:bracket, tab:mechanism)
python3 rank_attribution.py   # the published runs by key rank, the rewritten options (tab:attribution, tab:rewritten)
python3 claim_budget.py   # the ten headline findings corrected together (tab:claims)
python3 replication.py build     # needs eval_df.csv (1.1 GB) from BixBench's bucket; a clone already has its item files from restore_build_inputs.py
python3 mcq_audit.py --jsonl build/bixbench_v10_items.jsonl \
    --repair build/bixbench_v10_repaired.jsonl --output results/mcq_audit_bixbench_v10.json
python3 build_placebo_arm.py --items build/bixbench_v10_numeric_released.jsonl \
    --out build/bixbench_v10_numeric_placebo.jsonl
python3 replication.py pack      # the new v1.5 runs' answers, compact, from build/agent_runs_replication
python3 replication.py analyse   # the registered tests (tab:replication)
python3 repair_seeds.py --seeds 100
python3 writer_panel.py          # tab:panel
python3 tolerance_check.py       # tab:tolerance, tab:restated

# the rank-holding placebo for the ten-option file the claim rests on
python3 build_placebo_arm.py --items build/mmlu_pro_matched_released.jsonl \
    --out build/mmlu_pro_matched_placebo.jsonl

# what the agent says it is doing, and whether saying it pays
python3 agentic_mechanism.py \
    --dump results/agentic_dumps/qwen14b_withheld_aware.jsonl.gz

# the same readers on the text options of BixBench and LAB-Bench
python3 text_options.py --device cuda:2 --context set

# what the goodness-of-fit failure costs the geometry term
python3 rank_dependent_fit.py --probe-dir results/probe_mmlu
```

Bootstrap and permutation replicate counts default to 10,000; pass `--reps` to
trade precision for speed. Random seeds are fixed and never derived from
`hash()` of a string, so repeated runs reproduce byte-identical results across
processes.

Omit `--offline` on the provenance audit only to resolve uncached records from
public APIs. Cached records are retained, including HTTP failures.

Pinned BixBench revision `f8cc3bdcc6357c88b8c3648306522b9c422dc95a`,
SHA-256 `0d1204dcdae7193a9132ced5a3502008f6b3b163debc1b65b2aa2d86cb132dc9`.
Published baselines come from the BixBench results repository at the commit
recorded in `data/external/PROVENANCE.json`; `validate_artifact.py` re-checks
every digest.

## Auditing your own benchmark

`mcq_audit.py` is the part of this repository most likely to be useful to
someone else. Point it at any JSONL, JSON or CSV question file:

```bash
python3 mcq_audit.py --jsonl your_questions.jsonl \
    --key-field ideal --distractor-field distractors \
    --cluster-field source_paper            # omit if items are independent
```

It reports the keyed answer's rank distribution, the interior rate, the
cross-validated accuracy of two rule families, the same bound under three
non-numeric orderings of the options, and — the number to watch — the **best
score any rank-only solver could reach**, which is `max p` and needs no
assumption about who is solving. Then it prints `LEAKS` or `no leak detected`.
Add `--repair fixed.jsonl` to write a corrected file, and `--two-channel` to
uniformise the key's rank under every ordering rather than value alone.

Four cautions from our own use. **Repair every ordering you audit, then check
the output**: uniformising value alone left our repaired MMLU-Pro file more
exploitable than the released one (finding 12), and we only know because we ran
the probe on it. Regenerating distractors also raises the file's own no-data
baseline by about nine points through plausibility alone, so do not read that
rise as contamination. Rules are selected by leave-one-cluster-out,
so a family with no generalising member reports chance however many rules it
holds. A low score is a finding too: check both directions, because avoiding a
reliably-wrong option is worth +8 points on its own. And **do not** estimate
`max p` by holding out a fold and selecting on the rest — a rank
over-represented in training is by construction under-represented in the
held-out fold, so that estimator reads *below* chance on uniform data and
returned exactly zero in its leave-one-item-out form. `channel_survey.py` uses
a simultaneous confidence lower bound instead, which is conservative in the
direction the claim needs.

If you want to know *which* coordinate a solver is using rather than which ones
are open, `channel_attribution.py` takes a probe directory and reports, per arm
and per ordering, what the coordinate offers against what the solver takes.
That is how the fourth ordering was found; a coordinate it shows leaking
belongs in `mcq_audit.ORDERINGS`, which both the audit and the repair read.

## Statistical conventions

Questions drawn from one source study are not independent, so every interval
and test clusters on the study. Intervals come from a cluster bootstrap
resampling whole capsules; where a model is fitted, every replicate refits it.
Arm contrasts are paired within item and tested by a sign-flip randomisation
that reverses the arm labels of a whole capsule at once; two-sample contrasts
instead reassign whole capsules between arms.

Two specific guards. Because the probe answers each item under twenty redraws,
goodness-of-fit and likelihood-ratio tests are computed *within* a redraw,
where the sample is the 205 questions — pooling would claim a sample twenty
times larger than the one that exists, and it is what made the two-source model
look rejected before the correction. And because eleven per-model contrasts at
the 5% level will produce significant results in both directions by chance,
capability order was fixed in advance and the panel is summarised by a single
rank-correlation test with an exact permutation p-value.

## Eligibility semantics

`public_by` is an evidenced upper bound on appearance: it can establish
pre-cutoff availability but cannot establish a post-cutoff first appearance.
`first_public` admits a post-cutoff twin only when `first_appearance_verified`
is true and evidence is supplied; that field is a human adjudication, not a
fact the schema proves.

The checker reads a JSON object with `original`, `twin`, `cutoff`, and optional
`margin_months` (default 3) and `novelty` (`data_novel` by default, or
`analysis_novel`). Missing or insufficient evidence yields `unknown`. A
recorded pre-boundary data release makes a data-novel twin `ineligible`.
Cutoffs must be documented exact dates.

An `eligible` output means the declared date criteria pass. It does not certify
task difficulty, tool exposure, artifact secrecy, undocumented post-training
exposure, executable correctness, or absence of contamination. The
audit-derived manifest intentionally leaves every first-appearance and
data-version certification unknown.

## Build notes

The official style comes from the NeurIPS-hosted 2026 author kit; geometry,
fonts and review line numbering are unchanged. The manuscript overrides only
the venue notice. It builds with the style's `final` option, which prints the author block.

Tectonic warns about an encoding byte in the bundled `lineno.sty`, one
underfull bibliography line, and one underfull box from float placement. The
final build has no overfull boxes, no undefined references or citations, and
all fonts embedded. Logs stay in `build/`.

## Credit and licensing

The code is licensed under Apache-2.0 (`LICENSE`), and the results, trajectories, grades, derived item files,
figures and `provenance/` under CC BY 4.0 (`LICENSE-DATA`). Third-party files keep their own licenses, listed file
by file in `THIRD_PARTY_NOTICES.md`: BixBench is Apache-2.0 (`data/NOTICE`), LAB-Bench and ARC are CC BY-SA 4.0
(ShareAlike), and SciQ is CC BY-NC 3.0 (non-commercial use only). `MODEL_OUTPUTS.md` lists the models whose outputs
are in `results/` and their terms. The BixBench and LAB-Bench files carry their canary strings; keep them, and do
not train on this repository's benchmark content. To cite the paper, see `CITATION.cff`. `sources/bixbench_49311180/` vendors, unmodified, BixBench's
evaluation code, data-analysis-crow v1.5.0's prompts, environment and notebook
image recipe, the ldp v0.26.0 agent files the published protocol reproduces, the
fhlmi 0.25.2 model file that sets its reply limit, and the pyproject and lock files
that pin them, all Apache-2.0, with each project's licence text; `SOURCES.md` there gives the
commit and tag of each. `sources/cited/` quotes a few sentences each from an arXiv abstract and paper, a
bioRxiv abstract and a Hugging Face dataset card, for verification only. This analysis was only possible because the BixBench authors
released per-question zero-shot baselines *with the options as presented* —
something almost no benchmark does, and the single change we ask for. The
findings here are about benchmark construction, not about any research group,
and our own first two repair attempts reproduced the same class of mistake.
