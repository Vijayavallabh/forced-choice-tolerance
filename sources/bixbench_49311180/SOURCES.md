# Vendored upstream files, unmodified

Kept so that `bixbench_agent.py` and `bixbench_withdata.py` run BixBench's
published pipeline from the exact bytes it was published with, rather than from
a paraphrase of it. None of these files is edited. The agent reads
`fhda_prompts_v1.5.0.py` and the scorer `prompts.py` at run time; the rest are
what the scorer's graders, option formatting and answer parse reproduce, kept
beside them so the reproduction can be checked line by line.

| file | from | licence |
|---|---|---|
| `generate_trajectories.py`, `generate_trajectories.yaml`, `graders.py`, `models.py`, `postprocessing.py`, `postprocessing_utils.py`, `postprocessing.yaml`, `prompts.py`, `utils.py`, `pyproject.toml`, `uv.lock` | [Future-House/BixBench](https://github.com/Future-House/BixBench) at commit `49311180bdacb324c596f2e07596c126f2004008` | Apache-2.0 |
| `fhda_prompts_v1.5.0.py` (`src/fhda/prompts.py`), `fhda_Dockerfile.pinned`, `fhda_kernel_requirements.txt` | [Future-House/data-analysis-crow](https://github.com/Future-House/data-analysis-crow) at tag `v1.5.0`, the version BixBench pins (the repository now redirects to Future-House/finch) | Apache-2.0 |
| `fhda_notebook_env_v1.5.0.py` (`src/fhda/notebook_env.py`), `fhda_data_analysis_env_v1.5.0.py` (`src/fhda/data_analysis_env.py`), `fhda_utils_v1.5.0.py` (`src/fhda/utils.py`), `fhda_config_v1.5.0.py` (`src/fhda/config.py`), `fhda_pyproject_v1.5.0.toml` (`pyproject.toml`) | Future-House/data-analysis-crow at tag `v1.5.0` | Apache-2.0 |
| `ldp_react_v0.26.0.py` (`src/ldp/graph/modules/react.py`), `ldp_react_agent_v0.26.0.py` (`src/ldp/agent/react_agent.py`), `ldp_simple_agent_v0.26.0.py` (`src/ldp/agent/simple_agent.py`), `ldp_common_ops_v0.26.0.py` (`src/ldp/graph/common_ops.py`) | [Future-House/ldp](https://github.com/Future-House/ldp) at tag `v0.26.0` | Apache-2.0 |
| `fhlmi_llms_v0.25.2.py` (`lmi/llms.py`) | the `fhlmi` 0.25.2 wheel on PyPI, the version BixBench's `uv.lock` resolves (its sha256 checked against the lock); the package is built from the ldp repository's `packages/lmi` | Apache-2.0 |

The licence texts are `LICENSE.BixBench`, `LICENSE.data-analysis-crow`,
`LICENSE.ldp` and `LICENSE.fhlmi`, each as its repository or wheel ships it.

The environment and ldp files are what `bixbench_agent.py --protocol react`
reproduces: BixBench's `generate_trajectories.yaml` names ldp's `ReActAgent`.
BixBench's `pyproject.toml` lists `ldp` without a version, but fhda v1.5.0's
(`fhda_pyproject_v1.5.0.toml`) pins `ldp==0.26.0` and `fhaviary[server]==0.19.0`,
and BixBench's `uv.lock` at the same commit resolves ldp 0.26.0, fhaviary 0.19.0
and pydantic 2.10.1. Those are the versions vendored here and the ones the tool
schemas are built with, and the release matters: ldp 0.37.0 (19 September 2025)
stopped putting the reasoning back as "Thought: ... Based on this reasoning, let's
select the appropriate tool!\nAction: ", and pydantic 2.12 adds
`additionalProperties` to the answer's object type. ldp makes every call through
fhlmi's `LiteLLMModel`, which gives a config that sets no reply limit 4,096 tokens
(`fhlmi_llms_v0.25.2.py`); BixBench's sets none. None of these is imported or
run: `tests/test_published_protocol.py` parses them as text to check the agent's
prompt, tool schemas, notebook view and pins against them.

`sandbox_image/Dockerfile` is `fhda_Dockerfile.pinned` with the changes its
header lists, each needed to build it on amd64 today.
