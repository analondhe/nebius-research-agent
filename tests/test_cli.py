from __future__ import annotations

import agent
from research_agent.config import USE_FAKES_ENV


def test_cli_runs_end_to_end_on_fakes(monkeypatch, capsys):
    monkeypatch.setenv(USE_FAKES_ENV, "1")

    code = agent.main(["Should we move from Airflow to Dagster?"])

    out = capsys.readouterr().out
    assert code == 0
    answer, sources = out.split("\n\nSources:\n")
    assert "[1]" in answer
    assert sources.startswith("[1] ")


def test_cli_without_keys_exits_with_a_config_error(monkeypatch, capsys):
    monkeypatch.setattr(agent, "load_dotenv", lambda: None)
    for name in (USE_FAKES_ENV, "NEBIUS_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    assert agent.main(["q"]) == 2
    assert "missing NEBIUS_API_KEY, TAVILY_API_KEY" in capsys.readouterr().err
