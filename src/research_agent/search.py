"""Web search behind a small interface: the real Tavily client and an offline fake.

Both implementations return the body of Tavily's `POST /search` response
(https://docs.tavily.com/documentation/api-reference/endpoint/search), so the
pipeline parses the same shape in tests and in production.
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Callable, Mapping, Protocol, Sequence, Union

import requests
from tavily import TavilyClient
from tavily import errors as tavily_errors


class SearchError(Exception):
    """A search call failed: network, auth, quota, bad request or malformed body."""


class SearchClient(Protocol):
    def search(self, query: str, *, max_results: int) -> dict[str, Any]:
        """Run one web search and return a Tavily /search response body."""
        ...


_TAVILY_FAILURES = (
    tavily_errors.BadRequestError,
    tavily_errors.ForbiddenError,
    tavily_errors.InvalidAPIKeyError,
    tavily_errors.MissingAPIKeyError,
    tavily_errors.UsageLimitExceededError,
    tavily_errors.TimeoutError,
    requests.RequestException,
)


class TavilySearchClient:
    """Live search through the Tavily Python SDK."""

    def __init__(self, api_key: str, *, search_depth: str = "basic", timeout: float = 30.0) -> None:
        self._client = TavilyClient(api_key=api_key)
        self._search_depth = search_depth
        self._timeout = timeout

    def search(self, query: str, *, max_results: int) -> dict[str, Any]:
        try:
            return self._client.search(
                query,
                search_depth=self._search_depth,
                max_results=max_results,
                timeout=self._timeout,
            )
        except _TAVILY_FAILURES as exc:
            raise SearchError(f"Tavily search failed: {type(exc).__name__}: {exc}") from exc


def make_result(*, title: str, url: str, content: str, score: float) -> dict[str, Any]:
    """One entry of `results[]` as Tavily returns it with default request options."""
    return {"title": title, "url": url, "content": content, "score": score, "raw_content": None}


def make_search_response(query: str, results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """A full Tavily /search body; `answer` is null and `images` empty unless requested."""
    return {
        "query": query,
        "answer": None,
        "images": [],
        "results": list(results),
        "response_time": 0.42,
        "request_id": str(uuid.uuid4()),
    }


FakeOutcome = Union[Sequence[dict[str, Any]], Exception]


class FakeSearchClient:
    """Offline stand-in for TavilySearchClient.

    `responses` maps a query to its results (built with `make_result`) or to an
    exception to raise; unknown queries return no results, as Tavily does for
    queries with no matches. A callable receives the query instead. With no
    `responses`, every query gets three synthetic results on three domains.
    """

    def __init__(
        self,
        responses: Mapping[str, FakeOutcome] | Callable[[str], FakeOutcome] | None = None,
    ) -> None:
        self._responses = responses if responses is not None else synthetic_results
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, *, max_results: int) -> dict[str, Any]:
        self.calls.append({"query": query, "max_results": max_results})
        if callable(self._responses):
            outcome = self._responses(query)
        else:
            outcome = self._responses.get(query, [])
        if isinstance(outcome, Exception):
            raise outcome
        return make_search_response(query, list(outcome)[:max_results])


def synthetic_results(query: str) -> list[dict[str, Any]]:
    topic = query.rstrip("?!. ")
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:60] or "query"
    return [
        make_result(
            title=f"{topic} | Official documentation",
            url=f"https://docs.example.com/{slug}",
            content=f"The official documentation for {topic} describes setup, configuration and operational limits.",
            score=0.86,
        ),
        make_result(
            title=f"Changelog: {topic}",
            url=f"https://github.example.org/releases/{slug}",
            content=f"Release notes covering {topic} list recent features, deprecations and breaking changes.",
            score=0.74,
        ),
        make_result(
            title=f"Pricing for {topic}",
            url=f"https://pricing.example.net/{slug}",
            content=f"Published pricing relevant to {topic} is billed per seat with a free tier for small teams.",
            score=0.63,
        ),
    ]
