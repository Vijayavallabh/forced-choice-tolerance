# Preregistration: the trade under runs the paper has not read

Registered by committing and pushing this file before any analysis below is run on
the data it names. Nothing here is edited after that commit. Results, and any
deviation from this plan with its reason, are appended below the line at the end,
in a section of their own.

## What is tested

Proposition 2(ii) and the with-data result of the paper at commit `abd3813`
(Section 3): a rewrite that draws the key's rank uniformly (the *repair*) moves
some bracketed keys to an edge; on those keys the option nearest an agent's
submitted number is the key more often than through the released options; the
answers the repair newly credits are mostly far from the key and beyond it on the
side with no distractor; and on keys whose bracketed-or-edge class the repair keeps,
nothing changes. At `abd3813` this was measured on one run per question for seven
run sets of open models on BixBench v1.5, with the split into moved and other keys
chosen after the readings were in. Here it is tested on data that analysis never
saw.

## Data

**D1. BixBench v1.0, the published with-data runs of gpt-4o and Claude 3.5 Sonnet.**
`eval_df.csv`, published with the original BixBench paper at
`https://storage.googleapis.com/bixbench-results/eval_df.csv` (1,075,070,319 bytes,
md5 `3df10be01a348d2729de61da8e739b93`). Run sets: the four open-answer run names,
`4o_open_image`, `4o_open_no_image`, `claude_open_image`, `claude_open_no_image`.
One row is one trajectory: its submitted answer (`agent_answer`) to one question
(`uuid`), that question's released options (`mcq_options`, key first) and its
capsule (`problem_id`). Items: questions whose four released options all parse as
numbers (`channel_survey.numeric_ranks(options) is not None`). A trajectory with no
number in its answer scores 0 under every option set, as at `abd3813`.

**D2. BixBench v1.5, run sets made after this registration.**
- `qwen3-235b`: Qwen3-235B-A22B-Instruct-2507 (FP8 checkpoint), text protocol,
  rollouts r0 and r1, with and without the data.
- Rerun seeds r1 and r2 of the three text-protocol run sets of the paper
  (`qwen72b`, `llama70b`, `gemma27b`), with and without the data.

All are written to `build/agent_runs_replication/` and never pooled with the paper's
seven run sets.

## Option sets

- v1.5: the paper's three sets, unchanged (`build/bixbench_numeric_q.jsonl`,
  `..._placebo.jsonl`, `..._repaired.jsonl`).
- v1.0: *released* is the published options. The v1.0 question file
  `build/bixbench_v10_items.jsonl` has one row per question of `eval_df.csv`: its
  text, its published key and three distractors, and its capsule. The *repair* is
  built by the command that built v1.5's, at its default seed, on that file:
  `mcq_audit.py --jsonl build/bixbench_v10_items.jsonl --repair build/bixbench_v10_repaired.jsonl`.
  The *placebo* is `build_placebo_arm.py` at its default seed on the numeric items.
  An item either rewrite cannot redraw is reported and left out of the contrasts
  that need that set.

## Groups, fixed by the option sets before any answer is read

A key is *bracketed* when it is neither the smallest nor the largest of its four
values. *Moved to an edge*: bracketed in the released set, at an edge in the repair.
*Moved inward*: the reverse. *Kept*: the same class in both.

## Reading and statistics

Each trajectory is read through each option set by
`bixbench_withdata.nearest_is_key` (the option nearest the submitted number,
distance `|a-v|/(|a|+|v|)`, ties split evenly). No language model reads anything.

Per run set and item, the hit rate is the mean over that run set's trajectories of
the item (v1.0 has one to ten; D2 has one per rollout). An item's gain is its repair
hit rate minus its released one. Pooled over run sets, each item's gain is first
averaged over the run sets that answered it. A group's estimate is the mean over its
items. Intervals are percentile cluster bootstraps over capsules (10,000 draws) at
the nominal level that covers 95% on that group's capsules by the double bootstrap
(`claim_budget.covering_levels`), as the paper's two with-data bracketing claims are
calibrated.

## Hypotheses and what counts as a pass

- **H1, primary** (the moved keys gain). The pooled gain on keys moved to an edge is
  positive: the interval's lower end is above 0.
- **H2** (kept keys do not). The pooled change on kept keys has an interval that
  contains 0, or a mean within 5 points of 0.
- **H3** (what the repair credits is far and on the open side). Of all trajectories
  on moved keys that the released set reads wrong and the repair reads right, more
  than half are more than 25% from the key (`|a-y|/|y| > 0.25`), and more than half
  lie beyond the key on the side of the repaired set with no distractor.
- **H4** (not the redraw). The pooled repair-minus-placebo gain on moved keys is
  positive: the interval's lower end is above 0.
- **H5** (inward keys do not gain). The pooled change on keys moved inward has a
  mean of at most 0.

H1-H5 are tested on D1 (four run sets pooled, and reported per run set) and on D2,
for `qwen3-235b` (r0 and r1 pooled) and for the reruns (six run sets pooled), with
the data. **H6** (the gain needs no data): on D2 without the data, the pooled
moved-key gain is positive with its interval's lower end above 0.

## Run-to-run variation (descriptive, no test)

- D1: draw one trajectory per (run set, item) at random and compute the pooled
  moved-key gain; repeat 2,000 times; report the central 95% of this single-run
  gain beside the all-replica estimate.
- D2: each rerun seed's moved-key gain beside the paper's r0 for the same run set.

## Tolerance (descriptive, no test)

- **T1, the benchmark authors' own tolerance.** Every key BixBench v1.0 or v1.5
  writes as a numeric range: its relative half-width `(hi-lo)/(hi+lo)`. Report
  the distribution and where 5% falls in it.
- **T2, agreement with BixBench's grader.** On D1's numeric items, the published
  open-answer verdict (`correct`) against a number within t of the key for t in
  1, 2, 5, 10 and 20%, and against a rule that accepts an answer equal to the key at
  the key's written precision; Cohen's kappa and the count of each kind of
  disagreement.
- **T3.** Every tolerance-dependent number in the paper restated at 1% and 10%.

## What would count against the paper

H1 failing on D1 would mean the trade's with-data consequence does not carry to
the published closed models, and the paper will say so in its abstract.

---

## Results and deviations

(Appended after the analyses ran; nothing above this line changes.)

### D1, analysed 24 Sep 2026 (`python3 replication.py analyse`, `results/replication.json`)

159 numeric items (97 bracketed in the released set, as the paper's survey counts them),
9,284 open-answer trajectories in the four run sets, 5,161 of them on the numeric items. The repair redrew every item; the
placebo redrew 158 (one had no reachable redraw and is out of H4). The repair moved 41
keys to an edge, 32 inward and kept 86.

- H1, pass: +19.8 [+10.4,+31.2] at 98.50% (41 items, 30 capsules). Per run set: 4o_open_image +15.9 [+5.1,+32.6], 4o_open_no_image +13.7 [+7.2,+20.8], claude_open_image +25.2 [+11.7,+39.5], claude_open_no_image +24.6 [+12.6,+37.1].
- H2, pass on the 5-point margin: kept keys +3.7 [+0.8,+6.8] at 97.50% (86 items, 42 capsules). The interval does not
  contain 0; the mean is within 5 points of it.
- H3, pass: 283 trajectories newly credited on moved keys, 0 within 5% of the key,
  25 within 25%, 224 within 100% and 34 further (91.2% more than 25% off);
  272 (96.1%) on the open side. H3 counts a trajectory as newly credited
  when the repair reads it higher than the released set (`after > before`), as
  `bracketing.mechanism` does; 259 of the 283 go from wrong to right outright.
- H4, pass: repair minus placebo on moved keys +18.6 [+8.7,+30.7] at 99.00% (40 items, 29 capsules).
- H5, pass: keys moved inward -14.1 [-20.5,-8.2] at 96.25% (32 items, 25 capsules).
- Run to run: one trajectory drawn per (run set, item), 2,000 times, gives a pooled
  moved-key gain of 15.1 to 24.6 points (central 95%).

One deviation in D1: the repair command also writes an audit report, by default over
the shipped v1.5 one (`results/mcq_audit.json`). It was run again with
`--output results/mcq_audit_bixbench_v10.json`; the repaired file is byte-identical.

### T1 and T2, analysed 24 Sep 2026 (`python3 tolerance_check.py`, `results/tolerance_check.json`)

- T1: 60 of v1.5's keys are written as ranges that parse (one more is a range around 0
  and has no relative width), and 1 of v1.0's. Their relative half-widths have median
  6.1% (quartiles 1.3% and 11.5%); 47% are at or under 5%.
- T2: the published v1.0 grader on 5,161 numeric-item answers (it accepts 701):
  kappa within 1% 0.87, within 2% 0.89, within 5% 0.89, within 10% 0.85, within 20% 0.74, key precision 0.74. It accepts, by distance from the key: 0-0.01: 576/603; 0.01-0.02: 47/67; 0.02-0.05: 21/53; 0.05-0.1: 29/106; 0.1-0.2: 5/185; 0.2-0.5: 10/514; 0.5+: 9/2747.
  The same comparison on the v1.5 graders' verdicts on this repository's seven run sets
  (735 answers with the data): kappa within 1% 0.81, within 2% 0.81, within 5% 0.77, within 10% 0.71, within 20% 0.65, key precision 0.78.

### D2, analysed 25 Sep 2026 (`python3 replication.py analyse`, `results/replication.json`)

Four v1.5 run sets made after the registration, with the text protocol and the first runs' settings: `qwen3-235b` (Qwen3-235B-A22B-Instruct-2507, FP8), r0 and r1, and the rerun seeds r1 and r2 of `qwen72b`, `llama70b` and `gemma27b`, each with and without the data; 3,280 episodes on the 205 questions, 105 of them numeric (the repair moves 37 keys to an edge, 12 inward and keeps 56).

**Qwen3-235B-A22B, r0 and r1 pooled, with the data** (410 trajectories; 37 keys moved to an edge, 12 inward, 56 kept).

- H1, pass: +9.5 [+1.2,+22.1] at 99.49% (37 items, 21 capsules). Per run set: qwen3-235b|r0 +5.4 [+0.0,+24.9], qwen3-235b|r1 +13.5 [+3.0,+23.8].
- H2, FAIL (neither): kept keys +7.4 [+1.4,+14.9] at 98.50% (56 items, 37 capsules).
- H3, pass: 7 trajectories newly credited on moved keys, 0 within 5% of the key, 1 within 25%, 6 within 100% and 0 further (85.7% more than 25% off); 7 (100.0%) on the open side; 7 go from wrong to right outright.
- H4, FAIL: repair minus placebo on moved keys +4.7 [-0.8,+9.7] at 97.75% (37 items, 21 capsules).
- H5, pass: keys moved inward -12.5 [-54.9,+26.7] at 99.99% (12 items, 10 capsules).
- H6, pass: without the data, moved keys +21.6 [+10.0,+33.8] at 96.50% (37 items, 21 capsules).

**The reruns, six run sets pooled (Qwen2.5-72B, Llama-3.3-70B, gemma-3-27b; r1 and r2), with the data** (1,230 trajectories; 37 keys moved to an edge, 12 inward, 56 kept).

- H1, pass: +17.6 [+9.0,+25.8] at 96.50% (37 items, 21 capsules). Per run set: gemma27b|r1 +21.6 [+0.0,+43.9], gemma27b|r2 +27.0 [+15.6,+40.0], llama70b|r1 +13.5 [-5.7,+33.3], llama70b|r2 +24.3 [+11.8,+40.0], qwen72b|r1 +10.8 [+0.0,+21.4], qwen72b|r2 +8.1 [+0.0,+17.5].
- H2, pass (the interval contains 0): kept keys +2.2 [-0.8,+6.3] at 99.10% (56 items, 37 capsules).
- H3, pass: 42 trajectories newly credited on moved keys, 0 within 5% of the key, 3 within 25%, 28 within 100% and 11 further (92.9% more than 25% off); 41 (97.6%) on the open side; 42 go from wrong to right outright.
- H4, pass: repair minus placebo on moved keys +17.9 [+9.9,+25.5] at 96.75% (37 items, 21 capsules).
- H5, pass: keys moved inward -8.3 [-22.2,+0.0] at 99.97% (12 items, 10 capsules).
- H6, pass: without the data, moved keys +22.2 [+14.2,+30.9] at 96.25% (37 items, 21 capsules).

**Run to run** (each seed's moved-key gain, nearest option, repaired minus released):

- gemma27b|data: r0 +21.6, r1 +21.6, r2 +27.0
- gemma27b|nodata: r0 +24.3, r1 +21.6, r2 +29.7
- llama70b|data: r0 +32.4, r1 +13.5, r2 +24.3
- llama70b|nodata: r0 +32.4, r1 +33.1, r2 +27.0
- qwen3-235b|data: r0 +5.4, r1 +13.5
- qwen3-235b|nodata: r0 +18.9, r1 +24.3
- qwen72b|data: r0 +10.8, r1 +10.8, r2 +8.1
- qwen72b|nodata: r0 +13.5, r1 +8.1, r2 +13.5

**Beside the tests, not registered** (`not_registered` in the results; plain 95% cluster intervals):

- Share of numeric questions whose number is within 5% of the key, with the data: gemma27b|r1 5.7, gemma27b|r2 2.9, llama70b|r1 5.7, llama70b|r2 1.9, qwen3-235b|r0 23.8, qwen3-235b|r1 23.8, qwen72b|r1 10.5, qwen72b|r2 11.4; without it: gemma27b|r1 2.9, gemma27b|r2 2.9, llama70b|r1 4.8, llama70b|r2 2.9, qwen3-235b|r0 1.9, qwen3-235b|r1 3.8, qwen72b|r1 3.8, qwen72b|r2 4.8.
- The placebo's own change, placebo minus released: qwen3-235b|data|moved to an edge +4.7 [+0.0,+12.1]; qwen3-235b|data|moved inward +4.2 [+0.0,+13.6]; qwen3-235b|data|kept +5.6 [+0.9,+11.8]; qwen3-235b|nodata|moved to an edge -2.4 [-6.5,+0.7]; qwen3-235b|nodata|moved inward +0.0 [+0.0,+0.0]; qwen3-235b|nodata|kept -2.5 [-6.9,+1.8]; reruns|data|moved to an edge -0.3 [-2.9,+2.1]; reruns|data|moved inward +0.0 [+0.0,+0.0]; reruns|data|kept +0.5 [-1.2,+2.9]; reruns|nodata|moved to an edge +0.9 [-0.9,+3.4]; reruns|nodata|moved inward +0.0 [+0.0,+0.0]; reruns|nodata|kept +2.1 [-1.1,+6.0].
- Read together: Qwen3-235B-A22B's number falls near the key with the data, and any redraw of the distractors, which sit farther from the key than the released ones, credits some of those near misses; so H2 (kept keys) and H4 (repair against placebo) fail for it with the data. Without the data, where only H6 is registered, the placebo moves its kept and moved keys by -2.5 and -2.4; for the other run sets it changes little either way.

Deviations in D2 (none changes the plan's data, option sets, groups, reading or statistics):

- The driver's wall clock is a limit of ours, not of the protocol (which ends an episode at 40 steps or a full context). `qwen3-235b` ran with the first runs' 3,600 s and the reruns with 10,800 s. The 12 episodes it cut were set aside and rerun from the start, as the paper's published-protocol runs' were: `qwen3-235b` 7 (bix-1-q2 data r0, bix-31-q2 data r0, bix-35-q2 data r0, bix-43-q4 data r0, bix-45-q1 data r0, bix-49-q1 data r0, bix-9-q4 data r1); `qwen72b` 5 (bix-24-q1 data r2, bix-1-q1 nodata r1, bix-3-q4 nodata r1, bix-35-q1 nodata r1, bix-53-q5 nodata r2). The cut runs ship as `results/agent_runs_replication/superseded_<run>.jsonl.gz`, listed in `reruns_<run>.json`.
- Where they ran: `qwen3-235b` on four H100s; `llama70b` and `gemma27b` on A100s; `qwen72b` on two A100s for 237 episodes and four H100s for the other 583. Episodes in flight when a driver was stopped to move it wrote nothing and were run from the start. Each trajectory's `served_as` and `served_from` say where it ran.
- `results/replication.json` also carries `not_registered`, the two descriptive readings above; they are computed after the registered tests with their own seed and change none of them.

### Not registered: the registered runs read by a model, analysed 25 Sep 2026 (`python3 published_reads.py analyse`, `results/published_reads.json`)

None of this was registered; the plan reads the runs by the nearest option only. gemma-3-27b read every D1 and D2 run through `MCQ_EVAL_PROMPT`, as BixBench reads a run and as the paper reads its seven v1.5 run sets: the run's notebook (for D1 the `md_notebook` of `eval_df.csv`, without images, cut at 150,000 characters), the question with its options in two orders, and the answer, forced, through the released options, the placebo and the repair. Repaired minus placebo on the keys moved to an edge, with each group's interval at the level the double bootstrap finds covers 95% on its capsules, and a capsule sign-flip test:

- D1, the published runs: +8.7 [+4.3,+14.0] at 96.75%; sign-flip p = 0.0001; kept keys +1.0 [-0.8,+3.1] at 97.50%.
- D2, the reruns, with the data: +0.7 [-4.8,+5.7] at 97.25%; sign-flip p = 0.8610; kept keys +1.5 [-2.9,+5.0] at 97.50%.
- D2, Qwen3-235B-A22B, with the data: +1.4 [-4.6,+6.7] at 97.50%; sign-flip p = 0.7974; kept keys -2.7 [-7.0,+1.0] at 97.25%.
- D2, the reruns, without the data: +6.3 [+0.2,+12.9] at 97.50%; sign-flip p = 0.0559; kept keys -0.7 [-4.5,+3.2] at 96.75%.
- D2, Qwen3-235B-A22B, without the data: +6.1 [-1.5,+12.9] at 98.25%; sign-flip p = 0.1410; kept keys +1.8 [-4.2,+7.7] at 97.25%.

Against the published reading of the same D1 runs, through the released options: 4o_open_image agreement 78.0%, kappa 0.46; 4o_open_no_image agreement 79.5%, kappa 0.49; claude_open_image agreement 80.4%, kappa 0.58; claude_open_no_image agreement 79.7%, kappa 0.53.

### Not registered: more readers and the may-decline reading, analysed 25 Sep 2026 (`python3 published_reads.py analyse --reader-name NAME --mode MODE`, `python3 published_reads.py compare`, `results/published_reads_*.json`)

None of this was registered. Qwen2.5-72B and Llama-3.3-70B, each served as it read its own run sets in the paper (FP8, two-way tensor parallel), read the D1 runs forced through `MCQ_EVAL_PROMPT` as gemma-3-27b read them: every run in a random quarter (hash chunks 0-4 of 20) and every other run on a key the repair moves to an edge; and the moved keys of the paper's seven v1.5 run sets and of the D2 runs. gemma-3-27b read the D1 runs (the same quarter and moved keys), the seven run sets (all three option sets) and the D2 runs made with the data with BixBench's "insufficient information" option added, as BixBench's may-decline reading is taken. Qwen3-235B-A22B was tried as a reader and set aside: given an answer no option is near it reasoned past a 4,096-token reply (`results/published_reads_qwen3-235b_attempt.jsonl.gz`).

Repaired minus placebo on the keys moved to an edge, each at the level the double bootstrap finds covers 95% on its capsules, with a capsule sign-flip test:

- Qwen2.5-72B, D1, the published runs: +14.4 [+7.0,+22.2] at 97.25%; sign-flip p = 0.0001; kept keys +0.8 [-3.7,+5.9] at 97.00%
- Qwen2.5-72B, the seven v1.5 run sets, with the data: +10.2 [+4.5,+15.9] at 97.00%; sign-flip p = 0.0015 (moved keys only)
- Qwen2.5-72B, D2, the reruns, with the data: +6.8 [+0.2,+13.6] at 97.00%; sign-flip p = 0.0594 (moved keys only)
- Qwen2.5-72B, D2, Qwen3-235B-A22B, with the data: +2.7 [-5.3,+10.8] at 98.25%; sign-flip p = 0.5909 (moved keys only)
- Llama-3.3-70B, D1, the published runs: +8.8 [+3.7,+14.1] at 97.25%; sign-flip p = 0.0005; kept keys +3.7 [+0.3,+7.3] at 96.50%
- Llama-3.3-70B, the seven v1.5 run sets, with the data: +0.4 [-4.4,+5.0] at 98.00%; sign-flip p = 0.9277 (moved keys only)
- Llama-3.3-70B, D2, the reruns, with the data: +4.5 [-2.5,+11.1] at 98.00%; sign-flip p = 0.1783 (moved keys only)
- Llama-3.3-70B, D2, Qwen3-235B-A22B, with the data: +4.7 [-2.2,+12.5] at 97.75%; sign-flip p = 0.2430 (moved keys only)
- gemma-3-27b, may decline, D1, the published runs: +3.2 [+0.4,+7.3] at 99.46%; sign-flip p = 0.0029; kept keys +0.2 [-1.9,+2.2] at 96.75%
- gemma-3-27b, may decline, the seven v1.5 run sets, with the data: +1.4 [-0.9,+3.8] at 97.00%; sign-flip p = 0.3118; kept keys +1.0 [-0.4,+2.6] at 97.25%
- gemma-3-27b, may decline, D2, the reruns, with the data: +2.5 [-0.2,+5.8] at 99.63%; sign-flip p = 0.0548; kept keys +0.6 [-1.0,+2.5] at 97.75%
- gemma-3-27b, may decline, D2, Qwen3-235B-A22B, with the data: +0.7 [-3.5,+5.3] at 99.99%; sign-flip p = 1.0000; kept keys +0.4 [-3.0,+3.7] at 98.50%

On the quarter every reader read in full (1,270 D1 runs), the share of a reading's picks naming the option nearest the submitted number, over numeric answers and over misses (more than 5% from the key), and agreement with the published reading in the same mode (a run named when both of a reader's orders name the key; Cohen's kappa over all four run sets):

- published forced: nearest 74.9% of picks, 70.0% on misses
- published may-decline: nearest 90.9% of picks, 84.2% on misses; declines 74.9%
- Qwen2.5-72B: nearest 73.9% of picks, 69.2% on misses; agreement 80.5%, kappa 0.53
- gemma-3-27b: nearest 64.2% of picks, 58.2% on misses; agreement 79.1%, kappa 0.50
- Llama-3.3-70B: nearest 59.7% of picks, 52.8% on misses; agreement 78.0%, kappa 0.46
- gemma-3-27b, may decline: nearest 78.5% of picks, 69.8% on misses; agreement 94.4%, kappa 0.79; declines 59.1%
- rule: agreement 78.7%, kappa 0.48

The three forced readers' gains on the D1 moved keys enter the paper's claim family (`claim_budget.py`, now eight claims, each read at the level covering 1-0.05/8 on its own file); all three clear.

### Not registered: the registered runs re-read four ways and per run set, and the may-decline reading over three outcomes, analysed 25 Sep 2026 (`python3 answer_extraction.py`, `python3 run_set_scaling.py`, `python3 published_reads.py compare`, `results/answer_extraction.json`, `results/run_set_scaling.json`, `results/published_reads_readers.json`)

None of this was registered. The registered tests read an answer through the nearest-option rule only when the whole answer is a number (`bixbench_withdata.nearest_is_key`); the tolerance reads the last number in an answer. Of the D1 answers to numeric questions, 57.6% of gpt-4o's and 91.0% of Claude 3.5 Sonnet's are a number and 24.7 and 5.1% hold none; of the D2 answers made with the data, 68.1% of the new seeds' and 73.8% of Qwen3-235B-A22B's are a number. Repaired minus placebo on the keys moved to an edge, re-read four ways, each with a plain 95% cluster bootstrap over capsules:

- D1, the published runs: as registered +18.6 [+10.5,+27.7]; from the last number +22.4 [+13.8,+32.1]; the last number on answers holding one number +26.8 [+17.3,+37.3]; as registered without the keys zero or negative +19.6 [+11.2,+28.8]. Over every numeric item: +3.3 [+0.2,+6.9] as registered, +4.7 [+1.3,+8.6] from the last number
- D2, the reruns: as registered +17.9 [+10.5,+24.9]; from the last number +25.1 [+15.0,+34.3]; over every numeric item +6.3 [+3.0,+9.4]
- D2, Qwen3-235B-A22B: as registered +4.7 [+0.0,+9.0]; from the last number +6.1 [+0.0,+12.5]; over every numeric item +0.7 [-2.4,+3.5]

Per run set with the data (D1's four, the paper's seven v1.5 run sets and D2's eight, weighed alike), the rule's gain on the moved keys falls as the share of numeric questions whose submitted number is within 5% of the key rises (Spearman -0.61, permutation p = 0.008), and its gain over every numeric item faster (-0.92, none of 5,000 permutations as extreme); divided by the moved keys' miss rate it still falls (-0.52, p = 0.024).

gemma-3-27b allowed to decline and BixBench's published may-decline reading, over the three outcomes a read can have (the key, another option, a decline) on the quarter read in full: agreement 76.7% of 2,492 reads, kappa 0.54, against kappa 0.79 on whether a run is read as the key. Where gemma-3-27b names the key the published reading declines on 154 of 516 reads; where it names another option, on 309 of 504.

### Corrections, analysed 1 Oct 2026 (`python3 replication.py analyse`, `results/replication.json`)

Two faults in the analysis code were found after the entries above and fixed; the plan's data, option sets, groups,
reading and statistics are unchanged, and the entries above stand as they were written.

- The nearest-option rule compared an answer with each option by |a - v| / (|a| + |v|), which rounds to exactly 1 in
  floating point once the answer is about 10^16 times an option: such an answer tied all four options and was credited
  a quarter, where the option nearest it on the log scale should take it. The rule now compares the options on the log
  scale (`answer_numbers.nearest_ranks`, called by `bixbench_withdata.nearest_is_key`).
- An interval's end can be exactly 0: when every capsule with a nonzero sum lies on one side of 0, a resample drawing
  only capsules that sum to 0 has a mean of exactly 0, and when that has more probability than the interval's tail, 0
  is the percentile. The Monte Carlo percentile missed it when the draws held fewer such resamples than the tail. It is
  now counted exactly (`replication.zero_ends`). For Qwen3-235B-A22B's moved keys with the data, 16 of the 21 capsules
  sum to 0, a mass of 0.33% against the 0.26% tail of the 99.49% interval, so the interval's lower end is 0 and H1 fails.

Rerun, these results change; every other line above holds as written:

- D1, H2, pass on the 5-point margin: kept keys +3.7 [+0.9,+6.9] at 97.50% (86 items, 42 capsules); on 24 Sep +3.7 [+0.8,+6.8]. The interval still does not contain 0, and the paper counts H2 as failed there.
- D1, H5, pass: keys moved inward -14.2 [-20.6,-8.2] at 96.25% (32 items, 25 capsules); on 24 Sep -14.1 [-20.5,-8.2].
- D2, Qwen3-235B-A22B, H1, FAIL: +9.5 [+0.0,+22.1] at 99.49% (37 items, 21 capsules); on 25 Sep pass, +9.5 [+1.2,+22.1].
- D2, Qwen3-235B-A22B, H2, FAIL (neither): kept keys +8.9 [+2.7,+16.3] at 97.00% (56 items, 37 capsules); on 25 Sep +7.4 [+1.4,+14.9] at 98.50%.
- D2, the reruns, H2, pass (the interval contains 0): kept keys +2.9 [-0.2,+7.0] at 97.75% (56 items, 37 capsules); on 25 Sep +2.2 [-0.8,+6.3] at 99.10%.
- Not registered, the placebo's own change on the kept keys: qwen3-235b|data +6.2 [+1.5,+12.7] (on 25 Sep +5.6 [+0.9,+11.8]); reruns|nodata +2.0 [-1.2,+5.9] (on 25 Sep +2.1 [-1.1,+6.0]).
- Not registered: the four readings and the per-run-set trends of 25 Sep were rerun with the corrected rule and with the reader the tolerance now uses (`answer_numbers.py`, which no longer takes digits inside identifiers, list markers or exponents for the submitted number); their values are now those of `results/answer_extraction.json` and `results/run_set_scaling.json`, as the paper prints them.

Qwen3-235B-A22B thus fails H1, H2 and H4 with the data, and the reruns pass all six. The paper's claim family
(`claim_budget.py`) is now ten claims, each read at the level covering 1-0.05/10 on its own file; the three forced
readers' gains on the D1 moved keys still clear.
