#!/usr/bin/env python3
"""Requests to the OpenAI API for the closed-model runs: the key, the request, the ledger.

The graders, the no-data runs and the agent speak the OpenAI chat format to local vLLM
servers; the same requests go to OpenAI's own API when the base URL is
``https://api.openai.com``, or to an Azure OpenAI resource when it is ``https://HOST/openai``
with HOST under ``openai.azure.com``, ``services.ai.azure.com`` or
``cognitiveservices.azure.com`` (Azure's v1 API: ``{base}/v1/chat/completions``, the
``model`` field naming the deployment). Only then is anything here used:

* **The key** is read from ``OPENAI_API_KEY`` (``AZURE_OPENAI_API_KEY`` for Azure), else from
  ``~/.config/agenticls/openai.key``, and is sent to those hosts and to no other: in the
  ``Authorization`` header to OpenAI, in the ``api-key`` header to Azure. It is never printed,
  logged, cached or written anywhere; an error names the key's source, not its value, and any
  error text from a server is scrubbed of it.
* **The endpoint.** ``default_base()`` gives the Azure resource's base when
  ``AZURE_OPENAI_ENDPOINT`` or ``~/.config/agenticls/azure_openai.endpoint`` names one, and
  OpenAI's API otherwise; an endpoint that is not https at an Azure OpenAI host is refused,
  never replaced by another base. ``AZURE_OPENAI_API_VERSION``, if set, is sent as the
  ``api-version`` query parameter, for a resource whose v1 path asks for one.
* **Content filtering.** A request Azure's filter refuses (a 400 with code ``content_filter``, or,
  from an Azure AI Foundry model, a 400 whose choice has ``finish_reason`` ``content_filter``)
  raises ``ContentFiltered``, is logged at no cost and is not retried; a reply the filter cut
  (``finish_reason`` ``content_filter``) is returned with its content, empty if it has none.
* **The request** drops what only vLLM accepts (``bad_words``, ``continue_final_message``,
  ...). A reasoning model (``gpt-5*``, ``o1*``, ``o3*``, ``o4*``) takes ``max_completion_tokens``
  for ``max_tokens``, no sampling parameters and no ``stop``: its stop strings are applied to the
  reply here, which is what a server-side stop would have returned.
* **The ledger.** Every call's usage -- prompt, cached, completion and reasoning tokens, the
  snapshot the API reports in ``model``, the estimated cost -- is appended to
  ``build/openai_usage.jsonl`` (``AGENTICLS_OPENAI_USAGE_LOG`` moves it), without any prompt
  or reply text. ``AGENTICLS_OPENAI_BUDGET_USD`` must be set, and once the estimated spend
  of every process sharing the ledger reaches it, no new call is made.
* **The Responses API.** A model named in ``AGENTICLS_OPENAI_RESPONSES`` (comma-separated) is sent to
  ``{base}/v1/responses`` instead, for a deployment that takes function tools with reasoning only there:
  the chat request is translated (messages to input items, tool calls and tool replies to
  ``function_call`` and ``function_call_output`` items, ``reasoning_effort`` to ``reasoning.effort``,
  nothing stored) and the reply translated back into a chat completion, usage included.
* **Retries.** A 429 or 5xx is retried with exponential backoff, waiting at least as long as
  the ``Retry-After`` header asks; an exhausted quota, a rejected request or a failed
  authentication stops at once.

Prices are USD per million tokens (input, cached input, output); a model the table does not
name needs ``AGENTICLS_OPENAI_PRICES="input,cached,output"``, and the GPT-5 family's prices
can be set with ``AGENTICLS_GPT5_PRICES``.

``AGENTICLS_OPENAI_TEST_BASE`` names one more base to treat as OpenAI, for tests against a
local mock server (as Azure, with the ``api-key`` header, when ``AGENTICLS_OPENAI_TEST_AUTH`` is
``azure``); unset, only the hosts above are.

    python3 openai_api.py --print-base         # the base the runs will call (never the key)
    python3 openai_api.py --list-deployments   # GET {base}/v1/models once: what the resource can serve
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
OPENAI_HOST = "api.openai.com"
OPENAI_BASE = "https://api.openai.com"
AZURE_SUFFIXES = (".openai.azure.com", ".services.ai.azure.com", ".cognitiveservices.azure.com")
KEY_FILE = Path.home() / ".config" / "agenticls" / "openai.key"
ENDPOINT_FILE = Path.home() / ".config" / "agenticls" / "azure_openai.endpoint"
FILTERED_REPLY = "[content filter]"     # what a grader caches for a read the filter refused
DEFAULT_LEDGER = ROOT / "build" / "openai_usage.jsonl"

# USD per 1M tokens: input, cached input, output. Keyed by the string sent as ``model``.
PRICES = {"gpt-4o": (2.50, 1.25, 10.00),
          "gpt-4o-2024-08-06": (2.50, 1.25, 10.00),
          "gpt-4o-2024-11-20": (2.50, 1.25, 10.00)}
GPT5_DEFAULT = "1.25,0.125,10"
REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")
# Accepted by vLLM's server and not by OpenAI's.
VLLM_ONLY = ("bad_words", "continue_final_message", "add_generation_prompt", "chat_template_kwargs",
             "top_k", "min_p", "repetition_penalty", "skip_special_tokens",
             "spaces_between_special_tokens", "guided_json", "guided_regex", "guided_choice",
             "guided_grammar", "guided_decoding_backend", "structured_outputs", "echo",
             "include_stop_str_in_output", "ignore_eos", "min_tokens", "truncate_prompt_tokens")
# Not accepted by the reasoning models.
SAMPLING = ("temperature", "top_p", "seed", "logprobs", "top_logprobs", "presence_penalty",
            "frequency_penalty", "logit_bias", "n")
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 520, 522, 524, 529}
MAX_ATTEMPTS = 16


class KeyMissing(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class Rejected(RuntimeError):
    """A request the API refused for a reason retrying will not change."""

    def __init__(self, status, text):
        super().__init__(f"the API answered {status}: {text}")
        self.status, self.text = status, text


class ContextLength(Rejected):
    """The prompt does not fit the model's context."""


class ContentFiltered(Rejected):
    """A request Azure's content filter refused; retrying it will not change that."""


# ---------------------------------------------------------------- which base, which key

def _https_host(url):
    """The lower-cased host of an https URL on port 443, or None."""
    try:
        parts = urlsplit(str(url).strip())
        port = parts.port
    except ValueError:
        return None
    if parts.scheme != "https" or port not in (None, 443) or not parts.hostname:
        return None
    return parts.hostname.lower()


def auth_style(base):
    """``"openai"`` for OpenAI's own API, ``"azure"`` for an Azure OpenAI resource, None for anything
    else (a local vLLM server): https on port 443 at exactly OpenAI's host or a host under an Azure
    OpenAI domain, or the test base if one is set."""
    if not base:
        return None
    base = str(base).strip().rstrip("/")
    test = os.environ.get("AGENTICLS_OPENAI_TEST_BASE", "").strip().rstrip("/")
    if test and base == test:
        return "azure" if os.environ.get("AGENTICLS_OPENAI_TEST_AUTH", "").strip().lower() == "azure" else "openai"
    host = _https_host(base)
    if host == OPENAI_HOST:
        return "openai"
    if host and host.endswith(AZURE_SUFFIXES):
        return "azure"
    return None


def is_openai(base) -> bool:
    """True for OpenAI's API and for an Azure OpenAI resource: every base this module serves."""
    return auth_style(base) is not None


def is_azure(base) -> bool:
    return auth_style(base) == "azure"


def public_base(base):
    """What a result file records about where a model was served: an Azure resource's host names
    its owner, so it is recorded as the service alone; OpenAI's API and local servers as they are."""
    if isinstance(base, str) and "," not in base and is_azure(base):
        return "azure-openai"
    if isinstance(base, str) and any(is_azure(b) for b in base.split(",")):
        return "azure-openai"
    return base


def api_key(style="openai") -> str:
    env = "AZURE_OPENAI_API_KEY" if style == "azure" else "OPENAI_API_KEY"
    key = os.environ.get(env, "").strip()
    if key:
        return key
    try:
        key = KEY_FILE.read_text().strip()
    except OSError:
        key = ""
    if not key:
        raise KeyMissing(f"no key: set {env} or put it in {KEY_FILE} (mode 600)")
    return key


def auth_headers(base) -> dict:
    """The key's header for OpenAI's API (Authorization) or an Azure resource (api-key); nothing,
    and no key read, for any other base."""
    style = auth_style(base)
    if style == "azure":
        return {"api-key": api_key("azure")}
    if style == "openai":
        return {"Authorization": f"Bearer {api_key('openai')}"}
    return {}


def scrub(text):
    """``text`` with any key this module could send replaced, so that no error quotes one."""
    text = str(text)
    keys = [os.environ.get(n, "").strip() for n in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")]
    try:
        keys.append(KEY_FILE.read_text().strip())
    except OSError:
        pass
    for key in keys:
        if len(key) >= 8 and key in text:
            text = text.replace(key, "[key]")
    return text


def azure_base(endpoint) -> str:
    """The base for an Azure OpenAI endpoint, ``https://HOST/openai``, whatever path the endpoint was
    copied with; anything but https on port 443 at an Azure OpenAI host is refused."""
    text = str(endpoint).strip()
    host = _https_host(text if "://" in text else "https://" + text)
    if not host or not host.endswith(AZURE_SUFFIXES):
        raise ValueError(f"{text!r} is not an Azure OpenAI endpoint: expected https://NAME.openai.azure.com/ "
                         f"(or a host under {', '.join(s.lstrip('.') for s in AZURE_SUFFIXES[1:])})")
    return f"https://{host}/openai"


def default_base() -> str:
    """The Azure resource named by ``AZURE_OPENAI_ENDPOINT`` or the endpoint file, else OpenAI's API."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if not endpoint:
        try:
            endpoint = ENDPOINT_FILE.read_text().strip()
        except OSError:
            endpoint = ""
    return azure_base(endpoint) if endpoint else OPENAI_BASE


def uses_responses(model) -> bool:
    names = {n.strip() for n in os.environ.get("AGENTICLS_OPENAI_RESPONSES", "").split(",") if n.strip()}
    return str(model) in names


def responses_url(base) -> str:
    url = f"{str(base).strip().rstrip('/')}/v1/responses"
    version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
    return f"{url}?api-version={version}" if version and is_azure(base) else url


def _text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")


def _input_parts(content):
    if isinstance(content, str) or content is None:
        return content or ""
    parts = []
    for part in content:
        if part.get("type") == "text":
            parts.append({"type": "input_text", "text": part.get("text", "")})
        elif part.get("type") == "image_url":
            url = part.get("image_url")
            parts.append({"type": "input_image", "image_url": url.get("url") if isinstance(url, dict) else url})
    return parts


def to_responses(body):
    """A chat-completions request as a Responses request."""
    out = {"model": body["model"], "store": False}
    limit = body.get("max_completion_tokens", body.get("max_tokens"))
    if limit is not None:
        out["max_output_tokens"] = limit
    if body.get("reasoning_effort"):
        out["reasoning"] = {"effort": body["reasoning_effort"]}
    if body.get("tools"):
        out["tools"] = [{"type": "function", "name": t["function"]["name"],
                         "description": t["function"].get("description", ""),
                         "parameters": t["function"].get("parameters", {"type": "object", "properties": {}})}
                        for t in body["tools"] if t.get("type") == "function"]
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "function":
        out["tool_choice"] = {"type": "function", "name": choice["function"]["name"]}
    elif choice is not None:
        out["tool_choice"] = choice
    if "parallel_tool_calls" in body:
        out["parallel_tool_calls"] = body["parallel_tool_calls"]
    items = []
    for m in body.get("messages", []):
        role = m.get("role")
        if role in ("system", "developer", "user"):
            items.append({"role": role, "content": _input_parts(m.get("content"))})
        elif role == "assistant":
            text = _text_of(m.get("content"))
            if text:
                items.append({"role": "assistant", "content": text})
            for call in m.get("tool_calls") or []:
                items.append({"type": "function_call", "call_id": call["id"], "name": call["function"]["name"],
                              "arguments": call["function"].get("arguments") or "{}"})
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id"),
                          "output": _text_of(m.get("content"))})
    out["input"] = items
    return out


def from_responses(data):
    """A Responses reply as a chat completion: its text, its function calls and its usage."""
    texts, calls = [], []
    for item in data.get("output") or []:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") in ("output_text", "text"):
                    texts.append(part.get("text", ""))
                elif part.get("type") == "refusal":
                    texts.append(part.get("refusal", ""))
        elif item.get("type") == "function_call":
            calls.append({"id": item.get("call_id"), "type": "function",
                          "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"}})
    message = {"role": "assistant", "content": "".join(texts) if texts else None}
    if calls:
        message["tool_calls"] = calls
    reason = (data.get("incomplete_details") or {}).get("reason")
    finish = ("tool_calls" if calls else "length" if reason == "max_output_tokens"
              else "content_filter" if reason == "content_filter" else "stop")
    u = data.get("usage") or {}
    usage = {"prompt_tokens": u.get("input_tokens", 0), "completion_tokens": u.get("output_tokens", 0),
             "total_tokens": u.get("total_tokens", 0),
             "prompt_tokens_details": {"cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens", 0)},
             "completion_tokens_details": {"reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)}}
    return {"id": data.get("id"), "object": "chat.completion", "model": data.get("model"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}


def chat_url(base) -> str:
    url = f"{str(base).strip().rstrip('/')}/v1/chat/completions"
    version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
    return f"{url}?api-version={version}" if version and is_azure(base) else url


# A filter's code or finish reason in a body that is cut off or is not JSON.
FILTER_MARK = re.compile(r'"(?:code|finish_reason)"\s*:\s*"content_filter"')


def content_filtered(status, text) -> bool:
    """Whether an error reply is Azure's content filter refusing the request. Azure OpenAI answers
    with an error (``error.code`` ``content_filter``, or ``innererror.code``
    ``ResponsibleAIPolicyViolation``); Azure AI Foundry's other models answer with a chat completion
    under the same 400, a choice with ``finish_reason`` ``content_filter`` and an empty message, its
    ``content_filter_results.error.code`` ``content_filter`` (DeepSeek-V4-Pro's refusals, "blocked by
    label 'Jailbreak'", took that form and were read as ordinary rejections)."""
    if status != 400:
        return False
    text = str(text or "")
    try:
        body = json.loads(text)
    except ValueError:
        body = None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            inner = err.get("innererror")
            if err.get("code") == "content_filter" or (
                    isinstance(inner, dict) and inner.get("code") == "ResponsibleAIPolicyViolation"):
                return True
        choices = body.get("choices")
        for choice in choices if isinstance(choices, list) else []:
            if not isinstance(choice, dict):
                continue
            results = choice.get("content_filter_results")
            error = results.get("error") if isinstance(results, dict) else None
            if choice.get("finish_reason") == "content_filter" or (
                    isinstance(error, dict) and error.get("code") == "content_filter"):
                return True
    return ("content management policy" in text or "ResponsibleAIPolicyViolation" in text
            or bool(FILTER_MARK.search(text)))


def is_reasoning(model) -> bool:
    return str(model).startswith(REASONING_PREFIXES)


# ---------------------------------------------------------------- the request

def clean_message(message):
    """A message as OpenAI takes it: this repo's private ``_`` keys dropped, no ``name`` on a tool reply."""
    out = {k: v for k, v in message.items() if not k.startswith("_")}
    if out.get("role") == "tool":
        out.pop("name", None)
    return out


def adapt(payload):
    """(the payload as OpenAI takes it, stop strings to apply to the reply here)."""
    body = {k: v for k, v in payload.items() if k not in VLLM_ONLY}
    body["messages"] = [clean_message(m) for m in body.get("messages", [])]
    stops = []
    if is_reasoning(body.get("model", "")):
        if "max_tokens" in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
        for key in SAMPLING:
            body.pop(key, None)
        stop = body.pop("stop", None)
        stops = [stop] if isinstance(stop, str) else [s for s in (stop or []) if s]
    else:
        body.pop("reasoning_effort", None)
    return body, stops


def apply_stops(data, stops):
    """Cut each choice's text at the first stop string, as a server-side stop would have."""
    if not stops:
        return data
    for choice in data.get("choices") or []:
        message = choice.get("message") or {}
        text = message.get("content")
        if not text:
            continue
        cuts = [i for i in (text.find(s) for s in stops) if i >= 0]
        if cuts:
            message["content"] = text[:min(cuts)]
            choice["finish_reason"] = "stop"
    return data


def retry_after(headers):
    """Seconds the server asks us to wait, or None."""
    for name, scale in (("retry-after-ms", 1e-3), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            return max(0.0, float(value) * scale)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------- prices and the ledger

def prices(model):
    env = os.environ.get("AGENTICLS_OPENAI_PRICES")
    if env:
        return tuple(float(x) for x in env.split(","))
    model = str(model)
    if model in PRICES:
        return PRICES[model]
    if model.startswith("gpt-5"):
        return tuple(float(x) for x in os.environ.get("AGENTICLS_GPT5_PRICES", GPT5_DEFAULT).split(","))
    if model.startswith("gpt-4o") and not model.startswith("gpt-4o-mini"):
        return PRICES["gpt-4o"]
    raise KeyError(f"no price for {model!r}: set AGENTICLS_OPENAI_PRICES='input,cached,output' (USD per 1M)")


def usage_counts(usage):
    usage = usage or {}
    return {"prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "reasoning_tokens": int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)}


def reported_usage(text):
    """The token counts an error body reports (Azure AI Foundry's filter refusals carry the prompt's), for
    the ledger; nothing when it reports none."""
    try:
        body = json.loads(text)
    except ValueError:
        return {}
    usage = body.get("usage") if isinstance(body, dict) else None
    return usage_counts(usage) if isinstance(usage, dict) else {}


def cost_usd(model, usage):
    """Estimated cost of one call; reasoning tokens are billed inside completion tokens."""
    inp, cached, out = prices(model)
    u = usage_counts(usage)
    fresh = max(0, u["prompt_tokens"] - u["cached_tokens"])
    return (fresh * inp + u["cached_tokens"] * cached + u["completion_tokens"] * out) / 1e6


def budget_usd():
    value = os.environ.get("AGENTICLS_OPENAI_BUDGET_USD", "").strip()
    if not value:
        raise BudgetExceeded("set AGENTICLS_OPENAI_BUDGET_USD to call the OpenAI API")
    return float(value)


class Ledger:
    """The usage log, shared by every process that writes it; its total is the spend so far."""

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.spent = 0.0

    def refresh(self):
        if not self.path.exists():
            return self.spent
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            data = fh.read()
        end = data.rfind(b"\n") + 1            # whole lines only; a line being written waits
        for line in data[:end].splitlines():
            try:
                self.spent += float(json.loads(line).get("cost_usd") or 0.0)
            except (ValueError, AttributeError):
                continue
        self.offset += end
        return self.spent

    def check(self):
        budget = budget_usd()
        if self.refresh() >= budget:
            raise BudgetExceeded(f"estimated OpenAI spend ${self.spent:.2f} has reached the "
                                 f"${budget:.2f} budget ({self.path})")

    def record(self, entry):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")


_LEDGERS = {}


def ledger():
    path = os.environ.get("AGENTICLS_OPENAI_USAGE_LOG") or str(DEFAULT_LEDGER)
    if path not in _LEDGERS:
        _LEDGERS[path] = Ledger(path)
    return _LEDGERS[path]


# ---------------------------------------------------------------- the call

async def post_chat(http, base, payload, tag, attempts=MAX_ATTEMPTS):
    """One chat completion from OpenAI's API, as its JSON. ``http`` is an ``httpx.AsyncClient``."""
    import httpx

    book = ledger()
    book.check()
    body, stops = adapt(payload)
    responses = uses_responses(body.get("model"))
    url = responses_url(base) if responses else chat_url(base)
    sent = to_responses(body) if responses else body
    delay = 2.0
    for attempt in range(attempts):
        last = attempt == attempts - 1
        started = time.time()
        try:
            r = await http.post(url, json=sent, headers=auth_headers(base))
        except httpx.TransportError as err:
            if last:
                raise
            await asyncio.sleep(delay + random.random() * delay)
            delay = min(delay * 2, 60.0)
            continue
        if r.status_code == 200:
            data = from_responses(r.json()) if responses else r.json()
            counts = usage_counts(data.get("usage"))
            filtered = False
            for choice in data.get("choices") or []:
                if choice.get("finish_reason") == "content_filter":
                    filtered = True
                    if not isinstance(choice.get("message"), dict):
                        choice["message"] = {}
                    if choice["message"].get("content") is None:
                        choice["message"]["content"] = ""
            book.record({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tag": tag,
                         "model_sent": body.get("model"), "model": data.get("model"),
                         "system_fingerprint": data.get("system_fingerprint"), **counts,
                         "cost_usd": round(cost_usd(body.get("model"), data.get("usage")), 6),
                         "seconds": round(time.time() - started, 2),
                         **({"content_filter": True} if filtered else {})})
            return apply_stops(data, stops)
        full = r.text
        text = scrub(full[:400])
        if content_filtered(r.status_code, full):
            book.record({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tag": tag,
                         "model_sent": body.get("model"), "status": r.status_code, "content_filter": True,
                         **reported_usage(full), "cost_usd": 0.0, "seconds": round(time.time() - started, 2)})
            raise ContentFiltered(r.status_code, text)
        if r.status_code == 429 and "insufficient_quota" in text:
            raise Rejected(r.status_code, text)
        if r.status_code in RETRY_STATUS and not last:
            wait = retry_after(r.headers)
            await asyncio.sleep(max(wait or 0.0, delay) + random.random())
            delay = min(delay * 2, 60.0)
            book.check()
            continue
        if r.status_code == 400 and ("context_length_exceeded" in text or "maximum context length" in text
                                     or "too long" in text):
            raise ContextLength(r.status_code, text)
        raise Rejected(r.status_code, text)
    raise Rejected(0, "no attempt was made")


# ---------------------------------------------------------------- the command line

def list_models(base):
    """GET {base}/v1/models once: the model ids the key can reach there, printed without the key."""
    import httpx

    r = httpx.get(f"{str(base).rstrip('/')}/v1/models", headers=auth_headers(base), timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"{base}/v1/models answered {r.status_code}: {scrub(r.text[:300])}")
    data = r.json().get("data") or []
    ids = sorted({str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id")})
    print(f"{len(ids)} models at {base}:")
    for model in ids:
        print(f"  {model}")
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=None, help="the base to use; default: the Azure endpoint if one is "
                                                 "configured, else OpenAI's API")
    ap.add_argument("--print-base", action="store_true", help="print the base the runs will call and exit")
    ap.add_argument("--list-deployments", action="store_true",
                    help="list the model ids the key can reach at the base (one GET, no chat call)")
    args = ap.parse_args()
    try:
        base = args.base or default_base()
    except ValueError as err:
        raise SystemExit(str(err))
    if not is_openai(base):
        raise SystemExit(f"{base} is neither OpenAI's API nor an Azure OpenAI resource")
    if args.print_base:
        print(base)
    if args.list_deployments:
        list_models(base)
    if not (args.print_base or args.list_deployments):
        ap.print_help()


if __name__ == "__main__":
    main()
