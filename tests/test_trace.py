from __future__ import annotations

import json

from helpers import THREE_DOMAINS, all_supported, page, plan

import agent
from research_agent.config import USE_FAKES_ENV
from research_agent.llm import FakeLLMClient, LLMError
from research_agent.pipeline import AgentConfig, Status, research
from research_agent.search import FakeSearchClient, SearchError
from research_agent.trace import REDACTED, Trace

Q = "Should we move from Airflow to Dagster?"
NEBIUS_KEY = "nb-live-Zx81QpLm42"
TAVILY_KEY = "tvly-dev-Hq73NwPz19Kd"


def test_trace_follows_a_run_from_plan_to_stop_reason():
    llm = FakeLLMClient([
        plan("dagster docs"),
        plan("dagster docs", "airflow changelog"),
        "Both tools schedule data pipelines today [1][2].",
        all_supported,
    ])
    search = FakeSearchClient({
        "dagster docs": [THREE_DOMAINS[0], page("https://spam.example.com", score=0.05)],
        "airflow changelog": [THREE_DOMAINS[1], THREE_DOMAINS[2], {**THREE_DOMAINS[0], "url": THREE_DOMAINS[0]["url"] + "/"}],
    })

    result = research(Q, llm, search)
    trace = result.trace

    assert [e["event"] for e in trace.events] == [
        "run_started",
        "plan", "search", "source", "source", "evidence_check",
        "refine", "search", "source", "source", "source", "evidence_check",
        "stop", "writer", "claim_check", "finished",
    ]
    assert trace.of("plan")[0]["queries"] == ["dagster docs"]
    assert trace.of("refine")[0]["queries"] == ["airflow changelog"]
    assert trace.of("refine")[0]["discarded"] == [{"query": "dagster docs", "reason": "already searched"}]
    assert trace.of("search")[0]["results"] == [
        {"title": THREE_DOMAINS[0]["title"], "url": THREE_DOMAINS[0]["url"], "score": 0.9},
        {"title": "Page https://spam.example.com", "url": "https://spam.example.com", "score": 0.05},
    ]
    assert [(e["url"], e["kept"], e["reason"]) for e in trace.of("source")] == [
        (THREE_DOMAINS[0]["url"], True, "new source"),
        ("https://spam.example.com", False, "weak: score 0.05 below 0.30"),
        (THREE_DOMAINS[1]["url"], True, "new source"),
        (THREE_DOMAINS[2]["url"], True, "new source"),
        (THREE_DOMAINS[0]["url"] + "/", False, f"duplicate (same page) of {THREE_DOMAINS[0]['url']}"),
    ]
    assert [e["sufficient"] for e in trace.of("evidence_check")] == [False, True]
    assert trace.of("stop")[0]["reason"] == "enough evidence after round 2"
    assert trace.of("stop")[0]["rounds"] == 2
    finished = trace.of("finished")[0]
    assert finished["status"] == "answered"
    assert finished["sources"] == [THREE_DOMAINS[0]["url"], THREE_DOMAINS[1]["url"]]
    assert finished["not_cited"] == [THREE_DOMAINS[2]["url"]]
    assert trace.warnings() == []


def test_trace_records_round_limit_as_stop_reason():
    llm = FakeLLMClient([plan("a"), plan("b")])
    result = research(Q, llm, FakeSearchClient({"a": [], "b": []}), AgentConfig(max_rounds=2))

    assert result.trace.of("stop")[0]["reason"] == "reached the limit of 2 search rounds"
    assert result.trace.of("finished")[0]["status"] == "insufficient_evidence"


def test_trace_records_queries_cut_by_the_per_round_limit():
    llm = FakeLLMClient([plan("a", "b", "c"), "Answer sentence with enough words [1].", all_supported])
    result = research(Q, llm, FakeSearchClient({"a": THREE_DOMAINS}), AgentConfig(max_queries=2))

    assert result.trace.of("plan")[0]["queries"] == ["a", "b"]
    assert result.trace.of("plan")[0]["discarded"] == [{"query": "c", "reason": "over the 2-query limit per round"}]


def test_trace_flags_failures_and_fallbacks_as_warnings():
    llm = FakeLLMClient([LLMError("Nebius Token Factory call failed: RateLimitError"), plan()])
    search = FakeSearchClient({Q: SearchError("Tavily search failed: TimeoutError: 30")})

    result = research(Q, llm, search)

    assert result.status is Status.FAILED
    plan_event = result.trace.of("plan")[0]
    assert plan_event["level"] == "warning"
    assert plan_event["queries"] == [Q]
    assert result.trace.of("search")[0]["error"] == "Tavily search failed: TimeoutError: 30"
    assert result.trace.of("stop")[0]["reason"] == "no new queries to try"


def test_api_keys_never_reach_the_trace():
    trace = Trace(secrets=[NEBIUS_KEY, TAVILY_KEY])
    leaky_page = page("https://a.io/x", title=f"debug dump api_key={NEBIUS_KEY}")
    llm = FakeLLMClient([LLMError(f"401 for key {NEBIUS_KEY}"), "Answer sentence with enough words [1].", all_supported])
    search = FakeSearchClient({
        f"{Q} {TAVILY_KEY}": [leaky_page, *THREE_DOMAINS],
    })

    research(f"{Q} {TAVILY_KEY}", llm, search, trace=trace)
    trace.record("probe", "unknown keys too: tvly-prod-Aa11Bb22Cc33 and Authorization: Bearer sk_9f8e7d6c5b4a")

    dumped = trace.to_jsonl() + trace.render()
    assert NEBIUS_KEY not in dumped
    assert TAVILY_KEY not in dumped
    assert "tvly-prod-Aa11Bb22Cc33" not in dumped
    assert "sk_9f8e7d6c5b4a" not in dumped
    assert REDACTED in trace.of("plan")[0]["message"]
    assert trace.of("search")[0]["results"][0]["title"] == f"debug dump api_key={REDACTED}"


def test_cli_writes_trace_file_without_keys(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(USE_FAKES_ENV, "1")
    monkeypatch.setenv("NEBIUS_API_KEY", NEBIUS_KEY)
    monkeypatch.setenv("TAVILY_API_KEY", TAVILY_KEY)
    trace_file = tmp_path / "trace.jsonl"

    code = agent.main([Q, f"(key {NEBIUS_KEY})", "--trace", "--trace-file", str(trace_file)])

    assert code == 0
    raw = trace_file.read_text()
    events = [json.loads(line) for line in raw.splitlines()]
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "finished"
    assert NEBIUS_KEY not in raw and TAVILY_KEY not in raw
    err = capsys.readouterr().err
    assert "stopped searching: enough evidence after round 1" in err
    assert NEBIUS_KEY not in err
