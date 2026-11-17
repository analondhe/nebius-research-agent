"""Choose clients: real Nebius + Tavily by default, offline fakes only when the test flag is set."""
from __future__ import annotations

from typing import Mapping

from .llm import DEFAULT_BASE_URL, DEFAULT_MODEL, FakeLLMClient, LLMClient, NebiusLLMClient
from .search import FakeSearchClient, SearchClient, TavilySearchClient

USE_FAKES_ENV = "RESEARCH_AGENT_USE_FAKES"
REQUIRED_KEYS = ("NEBIUS_API_KEY", "TAVILY_API_KEY")


class ConfigError(Exception):
    pass


def use_fakes(env: Mapping[str, str]) -> bool:
    return env.get(USE_FAKES_ENV, "").strip().lower() in {"1", "true", "yes"}


def secret_values(env: Mapping[str, str]) -> list[str]:
    """The configured API keys, for redaction."""
    return [env[k].strip() for k in REQUIRED_KEYS if env.get(k, "").strip()]


def build_clients(env: Mapping[str, str]) -> tuple[LLMClient, SearchClient]:
    if use_fakes(env):
        return FakeLLMClient(), FakeSearchClient()
    missing = [k for k in REQUIRED_KEYS if not env.get(k, "").strip()]
    if missing:
        raise ConfigError(f"missing {', '.join(missing)}; copy .env.example to .env and add your keys")
    llm = NebiusLLMClient(
        env["NEBIUS_API_KEY"].strip(),
        model=env.get("NEBIUS_MODEL", "").strip() or DEFAULT_MODEL,
        base_url=env.get("NEBIUS_BASE_URL", "").strip() or DEFAULT_BASE_URL,
    )
    return llm, TavilySearchClient(env["TAVILY_API_KEY"].strip())
