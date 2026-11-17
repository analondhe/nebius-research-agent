"""Fakes must return what the real APIs return, and the factory must pick real clients by default."""
from __future__ import annotations

import pytest
import requests
from helpers import page
from tavily.errors import InvalidAPIKeyError

from research_agent.config import USE_FAKES_ENV, ConfigError, build_clients
from research_agent.llm import FakeLLMClient, LLMError, NebiusLLMClient, completion_text, make_chat_completion
from research_agent.search import FakeSearchClient, SearchError, TavilySearchClient

# Required fields from Tavily's POST /search 200 schema and its default per-result fields.
TAVILY_REQUIRED = {"query", "results", "images", "response_time", "answer"}
TAVILY_RESULT_FIELDS = {"title": str, "url": str, "content": str, "score": float}
# Required fields from Token Factory's ChatCompletionResponse / ChatCompletionChoice / Usage schemas.
COMPLETION_REQUIRED = {"id", "object", "created", "model", "choices", "usage", "service_tier"}
CHOICE_REQUIRED = {"index", "message", "finish_reason", "logprobs"}
USAGE_REQUIRED = {"prompt_tokens", "completion_tokens", "total_tokens"}


def assert_tavily_shape(body: dict) -> None:
    assert TAVILY_REQUIRED <= body.keys()
    assert isinstance(body["response_time"], float)
    for result in body["results"]:
        for name, kind in TAVILY_RESULT_FIELDS.items():
            assert isinstance(result[name], kind), name
        assert result["raw_content"] is None  # only filled when include_raw_content is requested


def assert_completion_shape(body: dict) -> None:
    assert COMPLETION_REQUIRED <= body.keys()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert CHOICE_REQUIRED <= choice.keys()
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] in {"stop", "length", "tool_calls", "content_filter"}
    assert USAGE_REQUIRED <= body["usage"].keys()


@pytest.mark.parametrize("client", [FakeSearchClient(), FakeSearchClient({"q": [page("https://a.io")]})])
def test_fake_search_returns_tavily_search_body(client):
    body = client.search("q", max_results=5)
    assert_tavily_shape(body)
    assert body["query"] == "q"
    assert body["results"]


def test_fake_search_respects_max_results_and_returns_empty_for_unknown_queries():
    client = FakeSearchClient({"q": [page(f"https://a.io/{i}") for i in range(5)]})
    assert len(client.search("q", max_results=2)["results"]) == 2
    empty = client.search("other", max_results=5)
    assert_tavily_shape(empty)
    assert empty["results"] == []


def test_fake_llm_returns_chat_completion_body():
    body = FakeLLMClient(["hello"]).complete([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert_completion_shape(body)
    assert completion_text(body) == "hello"
    assert_completion_shape(make_chat_completion("x", model="m"))


def test_completion_text_rejects_empty_and_malformed_bodies():
    empty = make_chat_completion("", model="m")
    empty["choices"][0]["finish_reason"] = "length"
    with pytest.raises(LLMError, match="finish_reason=length"):
        completion_text(empty)
    with pytest.raises(LLMError, match="malformed"):
        completion_text({"choices": []})


def test_real_clients_are_the_default():
    llm, search = build_clients({"NEBIUS_API_KEY": "nb-key", "TAVILY_API_KEY": "tvly-key", "NEBIUS_MODEL": "Qwen/Qwen3-32B"})
    assert isinstance(llm, NebiusLLMClient)
    assert isinstance(search, TavilySearchClient)
    assert llm.model == "Qwen/Qwen3-32B"


def test_missing_keys_fail_instead_of_falling_back_to_fakes():
    with pytest.raises(ConfigError, match="missing NEBIUS_API_KEY, TAVILY_API_KEY"):
        build_clients({})
    with pytest.raises(ConfigError, match="missing TAVILY_API_KEY"):
        build_clients({"NEBIUS_API_KEY": "nb-key", "TAVILY_API_KEY": "  ", USE_FAKES_ENV: "0"})


def test_test_flag_selects_fakes_without_keys():
    llm, search = build_clients({USE_FAKES_ENV: "1"})
    assert isinstance(llm, FakeLLMClient)
    assert isinstance(search, FakeSearchClient)


def test_tavily_sdk_errors_surface_as_search_errors(monkeypatch):
    client = TavilySearchClient("tvly-bogus")

    def raise_auth(*args, **kwargs):
        raise InvalidAPIKeyError("Unauthorized: missing or invalid API key.")

    monkeypatch.setattr(client._client, "search", raise_auth)
    with pytest.raises(SearchError, match="InvalidAPIKeyError"):
        client.search("q", max_results=5)

    def raise_network(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(client._client, "search", raise_network)
    with pytest.raises(SearchError, match="ConnectionError"):
        client.search("q", max_results=5)


def test_nebius_transport_errors_surface_as_llm_errors():
    client = NebiusLLMClient("nb-bogus", base_url="http://127.0.0.1:9/v1/", max_retries=0, timeout=2)
    with pytest.raises(LLMError, match="APIConnectionError"):
        client.complete([{"role": "user", "content": "hi"}])
