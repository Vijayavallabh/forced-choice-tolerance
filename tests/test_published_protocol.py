"""``bixbench_agent.py --protocol react``: BixBench's published agent, checked
against the upstream files it reproduces.

The first with-data runs called their tools in text, one call a step. The agent
BixBench's ``generate_trajectories.yaml`` names is ldp's ReActAgent, which makes
two calls a step with fhda's tools passed natively. Every piece of that protocol
the paper relies on is pinned here to the vendored upstream source -- parsed as
text, never imported or run -- so a paraphrase cannot pass for the original:

  * the system prompt is ldp's REACT_DEFAULT_PROMPT_TEMPLATE, character for character;
  * the tool schemas carry fhda v1.5.0's docstrings and signatures;
  * the first observation is ``DataAnalysisEnv.reset``'s, in its order;
  * tool replies, the notebook view, its output limit and the hidden-state
    placeholder are fhda's and ldp's strings;
  * the release is the one BixBench runs: fhda v1.5.0 pins ldp 0.26.0 and
    BixBench's lock resolves it, with the aviary and pydantic the schemas are built
    with, and the fhlmi whose 4,096-token reply limit every call gets;
  * a step is a reasoning call that may not pick a tool, put back in ldp 0.26.0's
    wrapper, a "Continue..." turn and a call that must: the reply is opened with
    the model's own tool-call tag and stopped at its closing tag, so the model calls
    exactly one tool, in the format it was trained on (vLLM's "required" forces a
    JSON format no model here was trained on, and "auto" lets it answer in ReAct's
    text format instead); and the reasoning call may not write that tag, since
    vLLM's "none", unlike the published agents' APIs, lets a model call a tool in
    text there.
"""
import ast
import asyncio
import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bixbench_agent as ba  # noqa: E402

SRC = ROOT / "sources" / "bixbench_49311180"


def _assignments(path):
    return {node.targets[0].id: node for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)}


def _methods(path, cls):
    tree = ast.parse(path.read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return {f.name: f for f in klass.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_the_system_prompt_is_ldps_react_template():
    found = _assignments(SRC / "ldp_react_v0.26.0.py")
    template = ast.literal_eval(found["_DEFAULT_PROMPT_TEMPLATE"].value.args[0])
    fields = {kw.arg: ast.literal_eval(kw.value)
              for kw in found["REACT_DEFAULT_PROMPT_TEMPLATE"].value.keywords}
    assert ba.REACT_SYSTEM == textwrap.dedent(template).format(**fields)


def test_the_step_is_ldps_two_calls():
    react = (SRC / "ldp_react_v0.26.0.py").read_text()
    module = react[react.index("class ReActModule(ReActModuleSinglePrompt)"):]
    assert 'llm_model["stop"] = ["Observation:", "Action:"]' in module
    assert module.count("self.llm_config") >= 2           # the stop strings ride on both calls
    assert 'tool_choice="none",  # Reasoning shouldn\'t pick a tool' in module
    assert ba.REACT_STOP == ["Observation:", "Action:"]
    assert 'Message(content="Continue...")' in react and ba.REACT_CONTINUE == "Continue..."
    ops = (SRC / "ldp_common_ops_v0.26.0.py").read_text()
    assert "tool_choice: Tool | str | None = LLMModel.TOOL_CHOICE_REQUIRED" in ops
    agent = (SRC / "ldp_react_agent_v0.26.0.py").read_text()
    assert "stop=stop_after_attempt(5)" in agent and ba.REACT_ATTEMPTS == 5
    assert 'update={"content": f"Observation: {m.content}"}' in agent
    simple = (SRC / "ldp_simple_agent_v0.26.0.py").read_text()
    assert f'content: str = "{ba.HIDDEN}"' in simple
    yaml = (SRC / "generate_trajectories.yaml").read_text()
    assert 'agent_type: "ReActAgent"' in yaml and "parallel_tool_calls: False" in yaml
    assert "single_prompt" not in yaml                    # so ReActAgent's default, False


def test_the_release_is_the_one_bixbench_runs():
    """fhda v1.5.0 pins ldp and aviary, and BixBench's lock resolves them (and pydantic)."""
    fhda = (SRC / "fhda_pyproject_v1.5.0.toml").read_text()
    assert '"ldp==0.26.0"' in fhda and '"fhaviary[server]==0.19.0"' in fhda
    assert "data-analysis-crow@v1.5.0" in (SRC / "pyproject.toml").read_text()
    lock = (SRC / "uv.lock").read_text()
    for name, version in (("ldp", "0.26.0"), ("fhaviary", "0.19.0"), ("pydantic", "2.10.1")):
        assert f'[[package]]\nname = "{name}"\nversion = "{version}"\n' in lock, name


def test_a_reply_may_run_to_the_published_agents_limit():
    """ldp 0.26.0 makes each call through fhlmi's LiteLLMModel, and fhlmi 0.25.2 (BixBench's
    lock) gives a config that sets no reply limit 4,096 tokens; BixBench's sets none. The
    driver's defaults are that limit and the config's temperature and step limit."""
    ops = (SRC / "ldp_common_ops_v0.26.0.py").read_text()
    assert "from lmi import LiteLLMModel as LLMModel" in ops and "model = LLMModel(config=config)" in ops
    assert '[[package]]\nname = "fhlmi"\nversion = "0.25.2"\n' in (SRC / "uv.lock").read_text()
    llms = (SRC / "fhlmi_llms_v0.25.2.py").read_text()
    assert 'if "model_list" not in data["config"]:' in llms
    assert '"max_tokens": data["config"].get("max_tokens", 4096),' in llms
    yaml = (SRC / "generate_trajectories.yaml").read_text()
    assert "max_tokens" not in yaml and "model_list" not in yaml
    assert "temperature: 1.0" in yaml and "max_steps: 40" in yaml
    defaults = {}
    for node in ast.walk(ast.parse((ROOT / "bixbench_agent.py").read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    defaults[node.args[0].value] = kw.value.value
    assert defaults["--max-tokens"] == 4096
    assert defaults["--temperature"] == 1.0 and defaults["--max-steps"] == 40


def test_the_reasoning_goes_back_in_ldps_wrapper():
    """ldp 0.26.0's postprocess_and_concat_resoning_msg, read from the source; 0.37.0
    dropped it, with the doubled period it leaves after a reasoning that ends in one."""
    tree = ast.parse((SRC / "ldp_react_v0.26.0.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "postprocess_and_concat_resoning_msg")
    assert "(react_message.content or '').removeprefix('Thought: ')" in ast.unparse(fn)
    content = next(k.value for n in ast.walk(fn) if isinstance(n, ast.Call)
                   for k in n.keywords if k.arg == "content" and isinstance(k.value, ast.JoinedStr))

    def render(reasoning):
        return "".join(v.value if isinstance(v, ast.Constant) else reasoning for v in content.values)
    for raw in ("Thought: count them.", "count them.", "", "\nThought: x", None):
        assert ba.react_thought(raw) == render((raw or "").removeprefix("Thought: "))
    assert ba.react_thought("Thought: count them.") == (
        "Thought: count them.. Based on this reasoning, let's select the appropriate tool!\nAction: ")


def test_the_reasoning_turn_may_not_open_a_call():
    """vLLM's "none" lets a model write its call in text; the published agents' APIs
    do not, so the reasoning turn bans the opening the selection turn forces."""
    assert ba.reasoning_ban("hermes") == "<tool_call>"
    assert ba.reasoning_ban("glm45") == "<tool_call>"
    assert ba.reasoning_ban("llama3_json") == '{"name": "'


def test_the_tools_are_fhdas_as_aviary_builds_them():
    nb = _methods(SRC / "fhda_notebook_env_v1.5.0.py", "NBEnvironment")
    da = _methods(SRC / "fhda_data_analysis_env_v1.5.0.py", "DataAnalysisEnv")
    functions = {"edit_cell": nb["edit_cell"], "list_workdir": nb["list_workdir"],
                 "submit_answer": da["submit_answer"]}      # DataAnalysisEnv overrides it
    schemas = {t["function"]["name"]: t["function"] for t in ba.FHDA_TOOLS}
    assert list(schemas) == ["edit_cell", "list_workdir", "submit_answer"]
    for name, fn in functions.items():
        doc = ast.get_docstring(fn)
        summary = doc.split("\n\nArgs:")[0].strip()
        assert schemas[name]["description"] == summary, name
        args = [a for a in fn.args.args if a.arg != "self"]
        required = [a.arg for a in args[:len(args) - len(fn.args.defaults)]]
        assert sorted(schemas[name]["parameters"]["properties"]) == sorted(a.arg for a in args)
        assert schemas[name]["parameters"]["required"] == required, name
        flat = " ".join(doc.split())
        for prop in schemas[name]["parameters"]["properties"].values():
            assert " ".join(prop["description"].split()) in flat, (name, prop)
    idx = schemas["edit_cell"]["parameters"]["properties"]["idx"]
    assert idx["anyOf"] == [{"type": "integer"}, {"type": "null"}]
    answer = schemas["submit_answer"]["parameters"]["properties"]["answer"]
    assert [a["type"] for a in answer["anyOf"]] == ["string", "number", "object", "null"]
    assert answer["anyOf"][2] == {"type": "object"}       # as pydantic 2.10.1 writes it


def test_the_first_observation_and_the_replies_are_fhdas():
    da = (SRC / "fhda_data_analysis_env_v1.5.0.py").read_text()
    reset = da[da.index("async def reset"):da.index("async def submit_answer")]
    order = [reset.index("Message(content=self.problem)"), reset.index("self.get_env_state_msg()"),
             reset.index('Message(role="system", content=self.system_prompt)')]
    assert order == sorted(order)
    assert 'return f"Submitted answer: {answer}"' in da
    nb = (SRC / "fhda_notebook_env_v1.5.0.py").read_text()
    assert 'return f"Appended new cell (#{new_idx})."' in nb
    assert 'return f"Edited cell #{idx}."' in nb
    assert ('text=f"Markdown representation of notebook contents ({nb_path}):\\n\\n{md_notebook}"'
            in nb)
    config = (SRC / "fhda_config_v1.5.0.py").read_text()
    assert f"NB_OUTPUT_LIMIT = {ba.NB_OUTPUT_LIMIT}  # chars" in config
    utils = (SRC / "fhda_utils_v1.5.0.py").read_text()
    assert 'return output[:cutoff] + "\\n<...output limited...>\\n" + output[-cutoff:]' in utils


def test_the_notebook_view_and_listing_are_fhdas(tmp_path):
    cells = [{"source": "x = 1", "output": "(no output)"},
             {"source": "print('a' * 5000)", "output": "a" * 5000}]
    view = ba.fhda_view(cells)
    # a cell that printed nothing has no output block, as in fhda's view_notebook
    assert view.startswith("### Cell 0:\n```python\nx = 1\n```\n### Cell 1:\n```python\n")
    shown = view.split("### Output 1:\n```\n", 1)[1].rsplit("\n```", 1)[0]
    assert shown == "a" * 1500 + "\n<...output limited...>\n" + "a" * 1500
    assert ba.fhda_view([]) == ""
    assert ba.fhda_state([], 1000) == ("Markdown representation of notebook contents "
                                       "(/workspace/notebook.ipynb):\n\n")
    (tmp_path / "a.csv").write_text("1")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("2")
    listing = json.loads(ba.fhda_list_dir(tmp_path, None))
    # fhda's own shape: files in a list, each subdirectory under its name, and the
    # empty "directories" key fhda writes beside them
    assert listing == {"files": ["a.csv"], "directories": {}, "sub": {"files": ["b.txt"]}}


def test_the_published_kernel_has_no_tools_of_its_own():
    def ask(tools):
        env = {**os.environ, "REPL_TOOLS": tools}
        proc = subprocess.run([sys.executable, str(ROOT / "sandbox_repl.py")], env=env,
                              input=json.dumps({"code": "print('submit_answer' in globals(), "
                                                        "'list_workdir' in globals())",
                                                "timeout": 60}) + "\n",
                              capture_output=True, text=True, timeout=120)
        return json.loads(proc.stdout.splitlines()[0])["output"]
    assert ask("1") == "True True"
    assert ask("0") == "False False"


class _Reply:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.text = json.dumps(body)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("an error reply reached raise_for_status")

    def json(self):
        return self._body


def _message(content="", calls=()):
    tool_calls = [{"id": f"call-{i}", "type": "function",
                   "function": {"name": name, "arguments": json.dumps(arguments)}}
                  for i, (name, arguments) in enumerate(calls)]
    return _Reply(200, {"choices": [{"message": {"content": content, "tool_calls": tool_calls},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


class _Client:
    def __init__(self, replies):
        self.replies, self.payloads = list(replies), []

    async def post(self, url, json=None, timeout=None):
        self.payloads.append(json)
        return self.replies.pop(0)


class _Box:
    made = []

    def __init__(self, *args, tools_in_kernel=True):
        self.tools_in_kernel, self.restarts, self.submitted, self.ran = tools_in_kernel, 0, None, []
        _Box.made.append(self)

    async def run(self, code):
        self.ran.append(code)
        return "1", False, False

    async def restart(self):
        self.restarts += 1

    async def kill(self):
        pass


def _args(tmp_path, **over):
    base = dict(model="m", protocol="react", out=tmp_path / "out", work=tmp_path / "work",
                base="http://server", max_steps=6, wallclock=3600, max_model_len=32768,
                chars_per_token=3.0, max_tokens=256, temperature=1.0, view_budget=36000, cpus="1",
                memory="1g", cell_timeout=60, extra_body=None, reasoning_effort=None,
                served_as=None, keep_workdirs=False, condition=["nodata"], tool_format="hermes")
    base.update(over)
    return types.SimpleNamespace(**base)


ITEM = {"question_id": "q1", "question": "How many genes?", "capsule_uuid": "c1", "ideal": "1",
        "data_folder": "d.zip"}


def _hermes(name, arguments):
    """A forced reply's continuation after ``<tool_call>\\n``, as Qwen2.5 writes it."""
    return _message(json.dumps({"name": name, "arguments": arguments}) + "\n")


def test_a_published_step_is_two_calls_and_one_forced_call(tmp_path, monkeypatch):
    monkeypatch.setattr(ba, "Sandbox", _Box)
    monkeypatch.setattr(ba, "REPLICA_DOWN", set())
    client = _Client([
        _message("Thought: count them in code."),                                   # step 0
        _hermes("edit_cell", {"contents": "print(1)", "idx": None}),
        _message("Thought: it printed 1."), _message("Action: submit_answer(1)"),   # step 1, try 1
        _message("Thought: it printed 1."),                                         # step 1, try 2
        _hermes("submit_answer", {"answer": "<answer>1</answer>"}),
    ])
    res = asyncio.run(ba.episode_react(ITEM, "nodata", 0, _args(tmp_path), ba.fhda_prompts(),
                                       client, asyncio.Semaphore(1)))
    assert res["termination"] == "submitted" and res["answer"] == "1"
    assert res["protocol"] == "react" and res["steps"] == 2
    assert _Box.made[-1].tools_in_kernel is False and _Box.made[-1].ran == ["print(1)"]
    first, second = client.payloads[0], client.payloads[1]
    assert first["tool_choice"] == "none" and first["stop"] == ["Observation:", "Action:"]
    assert first["bad_words"] == ["<tool_call>"] and "bad_words" not in second
    assert first["tools"] == ba.FHDA_TOOLS and second["tools"] == ba.FHDA_TOOLS
    # the selection call: the model's reply opened with its own tool-call tag, one call long
    assert second["continue_final_message"] is True and second["add_generation_prompt"] is False
    assert second["messages"][-1] == {"role": "assistant", "content": "<tool_call>\n"}
    assert second["stop"] == ["</tool_call>"] and second["tool_choice"] == "none"
    sent = first["messages"]
    prompts = ba.fhda_prompts()
    assert sent[0] == {"role": "system", "content": ba.REACT_SYSTEM}
    assert sent[1] == {"role": "user", "content": ba.task_prompt(ITEM["question"], prompts)}
    assert sent[2]["role"] == "user" and sent[2]["content"].startswith(
        "Markdown representation of notebook contents (/workspace/notebook.ipynb):")
    assert sent[3] == {"role": "system", "content": prompts["CAPSULE_SYSTEM_PROMPT_OPEN"]}
    thought = {"role": "assistant", "content": ba.react_thought("Thought: count them in code.")}
    assert second["messages"][-3:-1] == [thought, {"role": "user", "content": "Continue..."}]
    later = client.payloads[2]["messages"]
    assert later[2]["content"] == ba.HIDDEN                 # the old notebook state, hidden
    assert later[-5:-3] == [thought, {"role": "user", "content": "Continue..."}]
    call_turn = later[-3]
    assert call_turn["role"] == "assistant" and len(call_turn["tool_calls"]) == 1
    assert call_turn["tool_calls"][0]["function"]["name"] == "edit_cell"
    assert later[-2] == {"role": "tool", "tool_call_id": call_turn["tool_calls"][0]["id"],
                         "name": "edit_cell", "content": "Observation: Appended new cell (#0)."}
    assert later[-1]["content"].endswith("### Cell 0:\n```python\nprint(1)\n```\n"
                                         "### Output 0:\n```\n1\n```")
    assert not res["log"][0]["malformed"] and len(res["log"][1]["malformed"]) == 1
    assert not any("_state" in m for m in res["messages"])
    assert (tmp_path / "out" / "m-react" / "nodata" / "q1__r0.json").exists()
    assert not (tmp_path / "work" / "m-react" / "nodata" / "q1__r0").exists()


def test_five_unreadable_calls_end_the_episode(tmp_path, monkeypatch):
    monkeypatch.setattr(ba, "Sandbox", _Box)
    monkeypatch.setattr(ba, "REPLICA_DOWN", set())
    replies = []
    for _ in range(ba.REACT_ATTEMPTS):
        replies += [_message("Thought: hm."), _message("no call at all")]
    client = _Client(replies)
    res = asyncio.run(ba.episode_react(ITEM, "nodata", 0, _args(tmp_path), ba.fhda_prompts(),
                                       client, asyncio.Semaphore(1)))
    assert res["termination"] == "malformed" and res["answer"] is None
    assert len(res["log"][0]["malformed"]) == ba.REACT_ATTEMPTS
    assert ba.REPLICA_DOWN == set()                         # a bad draw says nothing of the server


def test_each_familys_calls_are_read_in_its_own_format():
    assert ba.parse_forced_call("hermes", '{"name": "list_workdir", "arguments": {}}\n') == \
        ("list_workdir", {})
    code = "import pandas as pd\nprint(pd.__version__)"
    glm = (f"edit_cell\n<arg_key>contents</arg_key>\n<arg_value>{code}</arg_value>\n"
           "<arg_key>idx</arg_key>\n<arg_value>2</arg_value>\n")
    assert ba.parse_forced_call("glm45", glm) == ("edit_cell", {"contents": code, "idx": 2})
    # a string parameter is taken as written, even when it would parse as JSON
    assert ba.parse_forced_call("glm45", "edit_cell\n<arg_key>contents</arg_key>\n<arg_value>1</arg_value>") \
        == ("edit_cell", {"contents": "1"})
    assert ba.parse_forced_call("glm45", "submit_answer\n<arg_key>answer</arg_key>\n"
                                         "<arg_value><answer>0.5</answer></arg_value>\n") == \
        ("submit_answer", {"answer": "<answer>0.5</answer>"})
    # Llama 3.x: the continuation after the forced '{"name": "', one object, whatever follows it
    assert ba.parse_forced_call("llama3_json", 'edit_cell", "parameters": {"contents": "x = 1", "idx": null}}') \
        == ("edit_cell", {"contents": "x = 1", "idx": None})
    assert ba.parse_forced_call("llama3_json", 'list_workdir", "parameters": {}}\n{"name": "submit_answer"}') \
        == ("list_workdir", {})
    for fmt, bad in (("hermes", "Action: list_workdir()"), ("glm45", "run_code\n"),
                     ("llama3_json", 'run_code", "parameters": {}}')):
        try:
            ba.parse_forced_call(fmt, bad)
        except ValueError:
            continue
        raise AssertionError(f"{fmt} read {bad!r} as a call")


def test_the_openings_are_the_families_own_tool_call_tags():
    """Each family's chat template, where the weights are on this machine."""
    import glob
    import pytest
    caches = {os.environ.get("HF_HUB_CACHE") or "", os.path.expanduser("~/.cache/huggingface/hub")}
    found = sorted(path for cache in caches if cache for path in glob.glob(
        os.path.join(cache, "models--Qwen--Qwen2.5-*-Instruct", "snapshots", "*", "tokenizer_config.json")))
    if not found:
        pytest.skip("no Qwen2.5 tokenizer here")
    template = json.load(open(found[0]))["chat_template"]
    assert "<tool_call>\\n" in template or "<tool_call>\n" in template
    assert ba.TOOL_FORMATS["hermes"] == ("<tool_call>\n", "</tool_call>")


def test_the_two_protocols_keep_their_runs_apart(tmp_path):
    assert ba.run_name(_args(tmp_path, protocol="text")) == "m"
    assert ba.run_name(_args(tmp_path)) == "m-react"


def test_the_scorer_keeps_the_protocols_apart_and_pairs_them():
    import bixbench_withdata as bw
    assert bw.run_key({"model": "m"}) == "m"
    assert bw.run_key({"model": "m", "protocol": "text"}) == "m"
    assert bw.run_key({"model": "m", "protocol": "react"}) == "m-react"

    def row(model, q, correct_released, correct_repaired, opened):
        def read(correct):
            return [{"correct": correct, "rank": 1, "refused": False, "no_pick": False}] * 2
        reads = {"r|released|forced": read(correct_released), "r|repaired|forced": read(correct_repaired),
                 "r|placebo|forced": read(correct_released)}
        return {"model": model, "condition": "data", "question_id": q, "rollout": 0,
                "capsule": f"c{q[-1]}", "answer": "x", "termination": "submitted", "open": opened,
                "strict": False, "within_5pct": False, "numeric": True, "reads": reads, "key_rank": {}}
    rows = []
    for i in range(6):
        rows.append(row("m", f"q{i}", False, True, False))       # text: released 0, repaired 1
        rows.append(row("m-react", f"q{i}", True, True, True))   # react: released 1, repaired 1
    summary = bw.summarise(rows, None)
    assert {"m|data", "m-react|data"} <= set(summary)
    block = summary["m-react|data|vs-m"]
    assert block["n"] == 6 and block["open"]["mean"] == 100.0
    assert block["reader:r"]["forced_all"]["mean"] == 100.0
    # the option set's effect, released minus repaired: -100 under text, 0 under react
    assert block["reader:r"]["released-repaired"]["mean"] == 100.0


def test_a_readers_settings_leave_every_other_readers_cache_keys_alone(monkeypatch):
    import hashlib
    import bixbench_withdata as bw
    monkeypatch.setattr(bw, "EXTRA_BODY", {"glm45air": {"chat_template_kwargs": {"enable_thinking": False}}})
    old = hashlib.sha256("qwen72b\x001536\x00the prompt".encode()).hexdigest()
    assert bw.cache_key("qwen72b", 1536, "the prompt") == old       # shipped keys still found
    assert bw.cache_key("glm45air", 1536, "the prompt") != \
        hashlib.sha256("glm45air\x001536\x00the prompt".encode()).hexdigest()
