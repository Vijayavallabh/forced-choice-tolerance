"""The OpenAI plumbing (openai_api.py) without a network: which bases get the key, what a request
sends to a reasoning model and to gpt-4o, the ledger's budget, and that the local readers' cache
keys are the ones the shipped caches were written under."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openai_api as oa  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in ("OPENAI_API_KEY", "AGENTICLS_OPENAI_TEST_BASE", "AGENTICLS_OPENAI_BUDGET_USD",
                 "AGENTICLS_OPENAI_PRICES", "AGENTICLS_GPT5_PRICES", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                 "AZURE_OPENAI_API_VERSION", "AGENTICLS_OPENAI_TEST_AUTH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(oa, "KEY_FILE", tmp_path / "no-key-here")
    monkeypatch.setattr(oa, "ENDPOINT_FILE", tmp_path / "no-endpoint-here")
    monkeypatch.setenv("AGENTICLS_OPENAI_USAGE_LOG", str(tmp_path / "usage.jsonl"))


def test_only_openai_itself_is_sent_the_key(monkeypatch):
    assert oa.is_openai("https://api.openai.com") and oa.is_openai("https://api.openai.com/")
    for base in ("http://api.openai.com", "https://api.openai.com.example.org", "https://example.org/api.openai.com",
                 "http://127.0.0.1:8103", "", None):
        assert not oa.is_openai(base), base

    def no_key():
        raise AssertionError("the key was read for a local server")
    monkeypatch.setattr(oa, "api_key", no_key)
    assert oa.auth_headers("http://127.0.0.1:8103") == {}
    monkeypatch.setenv("AGENTICLS_OPENAI_TEST_BASE", "http://127.0.0.1:9")
    assert oa.is_openai("http://127.0.0.1:9") and not oa.is_openai("http://127.0.0.1:10")


def test_the_key_comes_from_the_environment_then_the_file_and_is_never_quoted(monkeypatch, tmp_path):
    with pytest.raises(oa.KeyMissing) as missing:
        oa.api_key()
    assert "sk-" not in str(missing.value)
    key_file = tmp_path / "openai.key"
    key_file.write_text("sk-from-file\n")
    monkeypatch.setattr(oa, "KEY_FILE", key_file)
    assert oa.api_key() == "sk-from-file"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert oa.auth_headers("https://api.openai.com") == {"Authorization": "Bearer sk-from-env"}


def test_a_reasoning_model_gets_its_own_parameters_and_the_stops_are_kept_for_the_reply():
    payload = {"model": "gpt-5", "max_tokens": 4096, "temperature": 1.0, "seed": 3, "stop": ["Observation:", "Action:"],
               "reasoning_effort": "medium", "bad_words": ["<tool_call>"], "tools": [], "tool_choice": "none",
               "messages": [{"role": "user", "content": "x", "_state": True},
                            {"role": "tool", "tool_call_id": "c1", "name": "edit_cell", "content": "y"}]}
    body, stops = oa.adapt(payload)
    assert body["max_completion_tokens"] == 4096 and body["reasoning_effort"] == "medium"
    assert not {"max_tokens", "temperature", "seed", "stop", "bad_words"} & set(body)
    assert stops == ["Observation:", "Action:"]
    assert body["messages"] == [{"role": "user", "content": "x"}, {"role": "tool", "tool_call_id": "c1", "content": "y"}]
    assert "_state" in payload["messages"][0], "the caller's messages are not changed"
    data = {"choices": [{"message": {"content": "Thought: look.\nAction: edit_cell\nObservation: z"},
                         "finish_reason": "length"}]}
    assert oa.apply_stops(data, stops)["choices"][0]["message"]["content"] == "Thought: look.\n"
    assert data["choices"][0]["finish_reason"] == "stop"


def test_gpt_4o_keeps_its_sampling_and_its_server_side_stops():
    body, stops = oa.adapt({"model": "gpt-4o", "max_tokens": 1536, "temperature": 0.0, "stop": ["Action:"],
                            "reasoning_effort": "low", "continue_final_message": True, "messages": []})
    assert body == {"model": "gpt-4o", "max_tokens": 1536, "temperature": 0.0, "stop": ["Action:"], "messages": []}
    assert stops == []


def test_prices_cost_and_the_ledgers_budget(monkeypatch, tmp_path):
    usage = {"prompt_tokens": 10_000, "completion_tokens": 1_000, "prompt_tokens_details": {"cached_tokens": 4_000},
             "completion_tokens_details": {"reasoning_tokens": 600}}
    assert oa.cost_usd("gpt-4o", usage) == pytest.approx((6_000 * 2.5 + 4_000 * 1.25 + 1_000 * 10) / 1e6)
    assert oa.prices("gpt-5.1") == (1.25, 0.125, 10.0)
    monkeypatch.setenv("AGENTICLS_GPT5_PRICES", "2,0.2,20")
    assert oa.prices("gpt-5") == (2.0, 0.2, 20.0)
    with pytest.raises(KeyError):
        oa.prices("some-other-model")
    ledger = oa.Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(oa.BudgetExceeded):          # no budget set: no call
        ledger.check()
    monkeypatch.setenv("AGENTICLS_OPENAI_BUDGET_USD", "1.0")
    ledger.record({"cost_usd": 0.6})
    ledger.check()
    other = oa.Ledger(tmp_path / "ledger.jsonl")    # a second process sharing the file
    other.record({"cost_usd": 0.5})
    with pytest.raises(oa.BudgetExceeded):
        ledger.check()
    assert ledger.spent == pytest.approx(1.1)


def test_retry_after_is_read_in_seconds_or_milliseconds():
    assert oa.retry_after({"retry-after": "2"}) == 2.0
    assert oa.retry_after({"retry-after-ms": "1500"}) == 1.5
    assert oa.retry_after({}) is None and oa.retry_after({"retry-after": "soon"}) is None


def test_the_local_readers_cache_keys_are_unchanged():
    import bixbench_withdata as bw
    expected = hashlib.sha256("gemma27b\x001536\x00a prompt".encode()).hexdigest()
    assert bw.cache_key("gemma27b", 1536, "a prompt") == expected
    bw.EXTRA_BODY["glm45air"] = {"chat_template_kwargs": {"enable_thinking": False}}
    try:
        tail = "\x00" + json.dumps(bw.EXTRA_BODY["glm45air"], sort_keys=True)
        assert bw.cache_key("glm45air", 1536, "p") == hashlib.sha256(f"glm45air\x001536\x00p{tail}".encode()).hexdigest()
    finally:
        del bw.EXTRA_BODY["glm45air"]


def test_a_native_tool_call_is_read_as_the_forced_one_is():
    import bixbench_agent as ag
    message = {"tool_calls": [{"id": "c", "type": "function",
                               "function": {"name": "edit_cell", "arguments": json.dumps({"contents": "print(1)"})}}]}
    assert ag.parse_native_call(message) == ("edit_cell", {"contents": "print(1)"})
    for bad in ({}, {"tool_calls": [{"function": {"name": "rm", "arguments": "{}"}}]},
                {"tool_calls": [{"function": {"name": "edit_cell", "arguments": "[1]"}}]}):
        with pytest.raises((ValueError, KeyError, TypeError)):
            ag.parse_native_call(bad)


# ---------------------------------------------------------------- Azure OpenAI

AZURE = "https://myres.openai.azure.com/openai"


def test_azure_hosts_are_served_and_their_lookalikes_are_not():
    for base in (AZURE, AZURE + "/", "https://myres.openai.azure.com:443/openai", "https://MyRes.OpenAI.Azure.com/openai",
                 "https://myres.services.ai.azure.com/openai", "https://myres.cognitiveservices.azure.com/openai"):
        assert oa.auth_style(base) == "azure" and oa.is_openai(base) and oa.is_azure(base), base
    for base in ("http://myres.openai.azure.com/openai", "https://myres.openai.azure.com:8443/openai",
                 "https://myres.openai.azure.com.evil.com/openai", "https://openai.azure.com/openai",
                 "https://evilopenai.azure.com/openai", "https://myres.openai.azure.com@evil.com/openai",
                 "https://evil.com/myres.openai.azure.com", "https://myres.azure.com/openai",
                 "https://myres.openai.azure.com:notaport/openai"):
        assert oa.auth_style(base) is None and not oa.is_openai(base), base
    assert oa.auth_style("https://api.openai.com") == "openai" and not oa.is_azure("https://api.openai.com")


def test_each_host_gets_its_own_header_and_only_its_own_key(monkeypatch, tmp_path):
    key_file = tmp_path / "openai.key"
    key_file.write_text("azure-key-0123456789\n")
    monkeypatch.setattr(oa, "KEY_FILE", key_file)
    assert oa.auth_headers(AZURE) == {"api-key": "azure-key-0123456789"}
    assert oa.auth_headers("https://api.openai.com") == {"Authorization": "Bearer azure-key-0123456789"}
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-only")          # never sent to Azure
    assert oa.auth_headers(AZURE) == {"api-key": "azure-key-0123456789"}
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-from-env")    # never sent to OpenAI
    assert oa.auth_headers(AZURE) == {"api-key": "azure-from-env"}
    assert oa.auth_headers("https://api.openai.com") == {"Authorization": "Bearer sk-openai-only"}

    def no_key(*_):
        raise AssertionError("the key was read for a local server")
    monkeypatch.setattr(oa, "api_key", no_key)
    assert oa.auth_headers("http://127.0.0.1:8103") == {}
    monkeypatch.setenv("AGENTICLS_OPENAI_TEST_BASE", "http://127.0.0.1:9/openai")
    monkeypatch.setenv("AGENTICLS_OPENAI_TEST_AUTH", "azure")
    assert oa.auth_style("http://127.0.0.1:9/openai") == "azure"


def test_the_endpoint_becomes_the_resources_openai_base_or_is_refused(monkeypatch, tmp_path):
    for endpoint in ("https://myres.openai.azure.com/", "https://myres.openai.azure.com", "myres.openai.azure.com",
                     " https://myres.openai.azure.com/openai/v1/ ", "https://myres.openai.azure.com/openai"):
        assert oa.azure_base(endpoint) == AZURE, endpoint
    assert oa.azure_base("https://proj.services.ai.azure.com/api/projects/p1") == "https://proj.services.ai.azure.com/openai"
    for bad in ("http://myres.openai.azure.com/", "https://myres.openai.azure.com.evil.com/", "https://api.openai.com",
                "https://myres.openai.azure.com:8443/", ""):
        with pytest.raises(ValueError):
            oa.azure_base(bad)
    assert oa.default_base() == "https://api.openai.com"
    endpoint_file = tmp_path / "azure_openai.endpoint"
    endpoint_file.write_text("https://fromfile.openai.azure.com/\n")
    monkeypatch.setattr(oa, "ENDPOINT_FILE", endpoint_file)
    assert oa.default_base() == "https://fromfile.openai.azure.com/openai"
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fromenv.cognitiveservices.azure.com/")
    assert oa.default_base() == "https://fromenv.cognitiveservices.azure.com/openai"
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://not-azure.example.com/")
    with pytest.raises(ValueError):                  # refused, never replaced by OpenAI's API
        oa.default_base()


def test_the_chat_url_and_an_optional_api_version(monkeypatch):
    assert oa.chat_url(AZURE + "/") == AZURE + "/v1/chat/completions"
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "preview")
    assert oa.chat_url(AZURE) == AZURE + "/v1/chat/completions?api-version=preview"
    assert oa.chat_url("https://api.openai.com") == "https://api.openai.com/v1/chat/completions"


def test_azures_content_filter_is_told_from_other_refusals():
    body = {"error": {"message": "The response was filtered due to the prompt triggering Azure OpenAI's content "
                                 "management policy.", "type": None, "param": "prompt", "code": "content_filter",
                      "status": 400, "innererror": {"code": "ResponsibleAIPolicyViolation",
                                                    "content_filter_result": {"hate": {"filtered": True}}}}}
    assert oa.content_filtered(400, json.dumps(body))
    assert oa.content_filtered(400, json.dumps({"error": {"code": "x", "innererror": {"code": "ResponsibleAIPolicyViolation"}}}))
    assert oa.content_filtered(400, "blocked by the content management policy")
    assert not oa.content_filtered(400, json.dumps({"error": {"code": "context_length_exceeded"}}))
    assert not oa.content_filtered(429, json.dumps(body)) and not oa.content_filtered(400, "[1, 2]")


# Azure AI Foundry's refusal of DeepSeek-V4-Pro's bix-1-q2 request (logs/agent_DeepSeek-V4-Pro.log), as the
# driver logged it: a 400 whose body is a chat completion cut by the filter. The log keeps the first 400
# characters, so the body ends at "prompt_filter_results"; only the completion's id is redacted.
FOUNDRY_LOGGED = ('{"id":"chatcmpl-[redacted]","model":"","choices":[{"index":0,"message":{"role":"assistant",'
                  '"content":""},"finish_reason":"content_filter","content_filter_results":{"error":{"code":'
                  '"content_filter","message":"Response content blocked by label \'Jailbreak\'."}}}],"usage":'
                  '{"prompt_tokens":4684,"total_tokens":4684},"created":1790501211,"object":"chat.completion",'
                  '"prompt_filter_results')
# The same body whole, its last field closed empty (its contents are past what the log kept).
FOUNDRY_BODY = FOUNDRY_LOGGED + '":[]}'


def test_azure_ai_foundrys_filter_refusal_is_told_from_other_refusals():
    body = json.loads(FOUNDRY_BODY)
    assert body["choices"][0]["finish_reason"] == "content_filter" and "error" not in body
    assert oa.content_filtered(400, FOUNDRY_BODY)
    assert oa.content_filtered(400, FOUNDRY_LOGGED), "the logged, cut-off body"
    # either mark alone is the filter's
    only_reason = {**body, "choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                                        "finish_reason": "content_filter"}]}
    only_code = {**body, "choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                                      "finish_reason": None, "content_filter_results": {
                                          "error": {"code": "content_filter", "message": "blocked"}}}]}
    assert oa.content_filtered(400, json.dumps(only_reason)) and oa.content_filtered(400, json.dumps(only_code))
    # and neither is anything else's
    other = {**body, "choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                                  "finish_reason": "length", "content_filter_results": {}}]}
    assert not oa.content_filtered(400, json.dumps(other))
    assert not oa.content_filtered(400, json.dumps({"error": {"code": "invalid_request_error",
                                                              "message": "Unsupported parameter: 'seed'"}}))
    assert not oa.content_filtered(400, json.dumps({"choices": "content_filter"}))
    assert not oa.content_filtered(500, FOUNDRY_BODY) and not oa.content_filtered(200, FOUNDRY_BODY)


def test_a_foundry_refusal_is_logged_at_no_cost_and_not_retried(monkeypatch, tmp_path):
    import asyncio
    import httpx
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key-0123456789")
    monkeypatch.setenv("AGENTICLS_OPENAI_BUDGET_USD", "5")
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(400, text=FOUNDRY_BODY, headers={"content-type": "application/json"})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            await oa.post_chat(http, "https://myres.services.ai.azure.com/openai",
                               {"model": "DeepSeek-V4-Pro", "messages": []}, "agent:DeepSeek-V4-Pro")
    with pytest.raises(oa.ContentFiltered) as err:
        asyncio.run(go())
    assert err.value.status == 400 and "content_filter" in err.value.text
    assert len(seen) == 1, "the filtered request was retried"
    rows = [json.loads(l) for l in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["content_filter"] and rows[0]["status"] == 400
    assert rows[0]["cost_usd"] == 0.0 and rows[0]["prompt_tokens"] == 4684 and rows[0]["model_sent"] == "DeepSeek-V4-Pro"


def _transport(responses, seen):
    import httpx

    def handle(request):
        seen.append(request)
        status, body = responses.pop(0)
        return httpx.Response(status, json=body)
    return httpx.MockTransport(handle)


def test_a_filtered_request_is_logged_at_no_cost_and_not_retried(monkeypatch, tmp_path):
    import asyncio
    import httpx
    key_file = tmp_path / "openai.key"
    key_file.write_text("azure-key-0123456789")
    monkeypatch.setattr(oa, "KEY_FILE", key_file)
    monkeypatch.setenv("AGENTICLS_OPENAI_BUDGET_USD", "5")
    filtered = {"error": {"code": "content_filter", "message": "content management policy",
                          "innererror": {"code": "ResponsibleAIPolicyViolation"}}}
    cut = {"model": "gpt-4o-2024-08-06", "usage": {"prompt_tokens": 1000, "completion_tokens": 3},
           "choices": [{"finish_reason": "content_filter", "message": {"role": "assistant", "content": None}}]}
    seen = []
    responses = [(400, filtered), (200, cut)]

    async def go():
        async with httpx.AsyncClient(transport=_transport(responses, seen)) as http:
            with pytest.raises(oa.ContentFiltered) as err:
                await oa.post_chat(http, AZURE, {"model": "gpt-4o", "messages": []}, "t")
            assert "azure-key" not in str(err.value)
            return await oa.post_chat(http, AZURE, {"model": "gpt-4o", "messages": []}, "t")
    data = asyncio.run(go())
    assert len(seen) == 2, "the filtered request was retried"
    assert all(r.headers.get("api-key") == "azure-key-0123456789" and "authorization" not in r.headers for r in seen)
    assert str(seen[0].url) == AZURE + "/v1/chat/completions"
    assert data["choices"][0]["message"]["content"] == ""
    rows = [json.loads(l) for l in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert rows[0]["content_filter"] and rows[0]["cost_usd"] == 0.0 and rows[0]["status"] == 400
    assert rows[1]["content_filter"] and rows[1]["model"] == "gpt-4o-2024-08-06" and rows[1]["cost_usd"] > 0


def test_the_grader_keeps_a_filtered_read_as_no_pick(monkeypatch, tmp_path):
    import asyncio
    import bixbench_withdata as bw

    async def refuse(*_a, **_k):
        raise oa.ContentFiltered(400, "content management policy")
    monkeypatch.setattr(oa, "post_chat", refuse)

    async def go():
        client = bw.Client(tmp_path / "cache_filter_test.jsonl", concurrency=2)
        try:
            reply = await client.ask("gpt-4o", AZURE, "a prompt", 16)
            return reply, client.calls["content_filter"]
        finally:
            await client.http.aclose()
            client.handle.close()
    reply, count = asyncio.run(go())
    assert reply == oa.FILTERED_REPLY == "[content filter]" and count == 1
    assert bw.xml_extract(reply) == "Z"
    cached = [json.loads(l) for l in (tmp_path / "cache_filter_test.jsonl").read_text().splitlines()]
    assert cached[0]["reply"] == "[content filter]"


def test_errors_are_scrubbed_of_the_key(monkeypatch, tmp_path):
    key_file = tmp_path / "openai.key"
    key_file.write_text("azure-key-0123456789")
    monkeypatch.setattr(oa, "KEY_FILE", key_file)
    assert oa.scrub("denied for azure-key-0123456789 at x") == "denied for [key] at x"
    assert oa.scrub("nothing here") == "nothing here"


def test_the_command_line_prints_the_base_and_never_the_key(monkeypatch, tmp_path, capsys):
    key_file = tmp_path / "openai.key"
    key_file.write_text("azure-key-0123456789")
    monkeypatch.setattr(oa, "KEY_FILE", key_file)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myres.openai.azure.com/")
    monkeypatch.setattr(sys, "argv", ["openai_api.py", "--print-base"])
    oa.main()
    out = capsys.readouterr().out
    assert out.strip() == AZURE and "azure-key" not in out
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://evil.example.com/")
    with pytest.raises(SystemExit):
        oa.main()


def test_responses_translation_round_trip(monkeypatch):
    """A model named in AGENTICLS_OPENAI_RESPONSES goes to /v1/responses: the chat request's messages, tool
    calls, tool replies, tools and reasoning effort become Responses items, and the reply's text, function calls
    and usage come back as a chat completion."""
    monkeypatch.setenv("AGENTICLS_OPENAI_RESPONSES", "gpt-6-luna, other")
    assert oa.uses_responses("gpt-6-luna") and not oa.uses_responses("gpt-4o")
    assert oa.responses_url("https://x.openai.azure.com/openai").endswith("/openai/v1/responses")
    tools = [{"type": "function", "function": {"name": "edit_cell", "description": "run",
              "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}}]
    body = {"model": "gpt-6-luna", "max_completion_tokens": 100, "reasoning_effort": "medium", "tools": tools,
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                         {"role": "assistant", "content": None,
                          "tool_calls": [{"id": "c1", "type": "function",
                                          "function": {"name": "edit_cell", "arguments": "{\"code\": \"1\"}"}}]},
                         {"role": "tool", "tool_call_id": "c1", "content": "1"}]}
    out = oa.to_responses(body)
    assert out["reasoning"] == {"effort": "medium"} and out["max_output_tokens"] == 100 and out["store"] is False
    assert out["tools"][0]["name"] == "edit_cell" and out["tools"][0]["type"] == "function"
    assert [i.get("role") or i.get("type") for i in out["input"]] == ["system", "user", "function_call",
                                                                       "function_call_output"]
    assert out["input"][2]["call_id"] == "c1" and out["input"][3]["output"] == "1"
    reply = {"id": "r", "model": "gpt-6-luna", "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
        {"type": "function_call", "call_id": "c2", "name": "submit_answer", "arguments": "{\"answer\": \"4\"}"}],
             "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                       "input_tokens_details": {"cached_tokens": 2}, "output_tokens_details": {"reasoning_tokens": 3}}}
    chat = oa.from_responses(reply)
    msg = chat["choices"][0]["message"]
    assert msg["content"] == "done" and chat["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"][0]["id"] == "c2" and msg["tool_calls"][0]["function"]["name"] == "submit_answer"
    assert oa.usage_counts(chat["usage"]) == {"prompt_tokens": 10, "cached_tokens": 2, "completion_tokens": 5,
                                              "reasoning_tokens": 3}
