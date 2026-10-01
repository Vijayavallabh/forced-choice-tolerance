# Model outputs in this repository

Much of `results/` is text that language models wrote: agent trajectories,
answers, option picks, grader and reader replies, per-decision probe outcomes
and membership scores. They ship so that every number in the paper can be
recomputed without a GPU or an API key. This file lists which models wrote
them, where each model ran, and the terms that come with each model.

The author's own contribution to these files is licensed under CC BY 4.0
(`LICENSE-DATA`). That covers the prompts, the file structure, the grades and
the analyses, and whatever rights the author holds in the outputs themselves.
The model terms below still apply to the outputs, and anyone who reuses them
takes them on. This summary is not legal advice. The licenses were read on
30 September and 1 October 2026 and may have changed since; read the current
ones before relying on them.

## Open-weight models, run on our own GPUs

Weights from Hugging Face, run with `transformers`, or with vLLM for the 70B-class
prompt factorial. Model licenses as each model card states them:

| Model | License | Where its outputs are |
|---|---|---|
| Qwen2.5-1.5B-, 7B-, 14B- and 32B-Instruct | Apache-2.0 | `probe_*/`, `agentic_dumps/` |
| Qwen2.5-72B-Instruct | Qwen License Agreement (19 Sep 2024) | `probe_*/`, `agentic_dumps/qwen72b_*`, `agent_runs/*qwen72b*`, `agent_runs_replication/*qwen72b*`, `grading_variants_qwen72b_*`, `published_reads_qwen72b_*`, `option_design/reads_qwen72b*` |
| Qwen3-30B-A3B, Qwen3-235B-A22B | Apache-2.0 | `agent_runs/*qwen3a3b*`, `agent_runs_replication/*qwen3-235b*`, `agent_runs_q235_extra/`, `published_reads_qwen3-235b_*` |
| Llama-3.2-1B- and 3B-Instruct | Llama 3.2 Community License | `probe_*/`, `agentic_dumps/` |
| Llama-3.1-8B-Instruct | Llama 3.1 Community License | `probe_*/`, `agentic_dumps/` |
| Llama-3.3-70B-Instruct | Llama 3.3 Community License | `probe_*/`, `agentic_dumps/llama70b_*`, `agent_runs/*llama70b*`, `agent_runs_replication/*llama70b*`, `grading_variants_llama70b_*`, `published_reads_llama70b_*`, `option_design/reads_llama70b*` |
| Llama-3-Groq-8B-Tool-Use (Groq) | Meta Llama 3 Community License | `probe_*/` |
| gemma-3-4b-it, gemma-3-27b-it | Gemma Terms of Use | `probe_*/`, `agentic_dumps/gemma*`, `agent_runs/*gemma27b*`, `agent_runs_replication/*gemma27b*`, `grading_variants_gemma27b_*`, `published_reads_gemma27b_*`, `option_design/reads_gemma27b*` |
| phi-4, Phi-3.5-mini-instruct | MIT | `probe_*/`, `agentic_dumps/` |
| OLMo-2-1124-7B-Instruct | Apache-2.0 | `probe_*/`, `agentic_dumps/` |
| GLM-4.5-Air | MIT | `agent_runs/*glm45air*` |

Paths are under `results/`. The per-model files name the model in their file
name or in a `model` field.

Some rewritten option sets in `data/derived/` also hold text a model wrote: the
wrong-step sets for BixBench (Qwen2.5-14B-Instruct and Llama-3.1-8B-Instruct)
and the MMLU frontier arms whose distractors an instruction model generated. The
terms of the model that wrote them apply to those values too.

## Models called through an API

These were called through the author's Azure OpenAI resource (Microsoft
Foundry). Microsoft's Product Terms for Azure govern those calls. Microsoft
states that prompts and completions sent to models sold by Azure are not made
available to OpenAI or other model providers and are not used to train models.

| Model | Terms of the model | Where its outputs are |
|---|---|---|
| gpt-4o (version 2024-11-20) | OpenAI model, served by Azure | `openai_nodata/`, `forced_guess_replies.jsonl.gz`, `grading_variants_gpt-4o_*`, `published_reads_gpt-4o_*`, `option_design/reads_gpt-4o*`, and the reader caches in `strong_agent/` and `frontier_agents/` |
| gpt-5.1 (version 2025-11-13) | OpenAI model, served by Azure | `strong_agent/trajectories_gpt-5.1-react.jsonl.gz`, `strong_agent_rows.json.gz` |
| gpt-6-luna | OpenAI model, served by Azure | `frontier_agents/trajectories_gpt-6-luna-react.jsonl.gz` |
| DeepSeek-V4-Pro | MIT (weights), served by Azure AI Foundry | `frontier_agents/trajectories_DeepSeek-V4-Pro-react.jsonl.gz` |
| Kimi-K3 | Kimi K3 License (a modified MIT license), served by Azure AI Foundry | `frontier_agents/trajectories_FW-Kimi-K3-react.jsonl.gz` |

`frontier_agents_rows.json.gz` holds the graded rows of the last three.

## Outputs we did not generate

BixBench's published runs, by gpt-4o (2024-08-06) and Claude 3.5 Sonnet, were
made by FutureHouse. `results/bixbench_v10_published_runs.json.gz` is an
extract of their `eval_df.csv`, and `data/external/zero_shot_v10/` and
`zero_shot_v15/` are their zero-shot baseline files. The files derived from the
extract are `results/nonnumeric_published_picks.jsonl.gz`,
`published_forced_all_questions.jsonl.gz` and `published_reads_*` (our
readers' replies to those runs), and in part
`score_decomposition_notebook_values.jsonl.gz`. `THIRD_PARTY_NOTICES.md`
gives their source and terms.

## Clauses passed on to you

None of these terms forbids publishing the outputs. Four of them attach
conditions to training a model on outputs, and one names a restriction on the
customer. They are quoted verbatim.

**Meta Llama 3 Community License** (Llama-3-Groq-8B-Tool-Use), Section 1.b.v:

> You will not use the Llama Materials or any output or results of the Llama
> Materials to improve any other large language model (excluding Meta Llama 3
> or derivative works thereof).

**Llama 3.1, 3.2 and 3.3 Community Licenses** (Llama-3.1-8B, Llama-3.2-1B and
3B, Llama-3.3-70B), Section 1.b.i, in the same words in all three:

> If you use the Llama Materials or any outputs or results of the Llama
> Materials to create, train, fine tune, or otherwise improve an AI model,
> which is distributed or made available, you shall also include "Llama" at
> the beginning of any such AI model name.

**Qwen License Agreement** (Qwen2.5-72B-Instruct), Section 5.b:

> If you use the Materials or any outputs or results therefrom to create,
> train, fine-tune, or improve an AI model that is distributed or made
> available, you shall prominently display "Built with Qwen" or "Improved
> using Qwen" in the related product documentation.

**Gemma Terms of Use** (gemma-3-4b-it, gemma-3-27b-it). Section 3.3 says
"Google claims no rights in Outputs you generate using Gemma. You and your
users are solely responsible for Outputs and their subsequent uses." Its
definition of Model Derivatives includes "any other machine learning model
which is created by transfer of patterns of the weights, parameters,
operations, or Output of Gemma, to that model in order to cause that model to
perform similarly to Gemma, including distillation methods that use
intermediate data representations or methods based on the generation of
synthetic data Outputs by Gemma for training that model." A model trained that
way is a Model Derivative, and the Gemma Terms of Use then apply to it.

**OpenAI models.** The calls went through Azure, under Microsoft's terms rather
than OpenAI's. For reference, OpenAI's Services Agreement (online
version 010126) says the customer "owns all Output", and in Section 3.3(e) it
bars the customer from using Output "to develop artificial intelligence models
that compete with OpenAI's products and services", with narrow exceptions.

The Apache-2.0 and MIT models and the Kimi K3 License place no condition on
their outputs beyond the usual disclaimer of warranty.

## Canary strings

The BixBench and LAB-Bench files in `data/` carry the benchmarks' canary
strings (see `THIRD_PARTY_NOTICES.md`). The trajectories and replies quote the
benchmark questions without them. Please do not train on this repository's
benchmark content.
