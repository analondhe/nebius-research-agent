"""Evidence handling: parse Tavily results, drop weak ones, dedupe, judge sufficiency."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

from .search import SearchError

_TRACKING_PARAMS = re.compile(r"^(utm_.*|fbclid|gclid|mc_cid|mc_eid|ref|ref_src)$", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
NEAR_DUPLICATE_JACCARD = 0.85
_MIN_WORDS_FOR_CONTENT_MATCH = 12


@dataclass(frozen=True)
class Source:
    url: str
    title: str
    content: str
    score: float
    query: str

    @property
    def domain(self) -> str:
        return domain_of(self.url)


@dataclass(frozen=True)
class Decision:
    """What happened to one search result and why."""

    url: str
    title: str
    score: float | None
    kept: bool
    reason: str


@dataclass(frozen=True)
class EvidenceCheck:
    sufficient: bool
    sources: int
    domains: int
    reason: str


def domain_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host.removeprefix("www.")


def normalize_url(url: str) -> str:
    """Key for "same page": ignores scheme, www., fragment, trailing slash, tracking params."""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/") or "/"
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if not _TRACKING_PARAMS.match(k)))
    return f"{domain_of(url)}{path}" + (f"?{query}" if query else "")


def _words(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def parse_results(response: Any) -> list[dict[str, Any]]:
    """`results[]` of a Tavily /search body; raises SearchError if the body has none."""
    results = response.get("results") if isinstance(response, dict) else None
    if not isinstance(results, list):
        raise SearchError("malformed Tavily response: no results list")
    return results


class EvidencePool:
    """Sources kept so far, one per page, highest score wins."""

    def __init__(self, min_score: float) -> None:
        self.min_score = min_score
        self._by_key: dict[str, Source] = {}

    def add(self, query: str, results: list[dict[str, Any]]) -> list[Decision]:
        return [self._add_one(query, raw) for raw in results]

    def _add_one(self, query: str, raw: Any) -> Decision:
        raw = raw if isinstance(raw, dict) else {}
        url, content = raw.get("url"), raw.get("content")
        title = raw.get("title") if isinstance(raw.get("title"), str) and raw.get("title").strip() else url
        score = raw.get("score")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return Decision(str(url), str(title), None, False, "no usable URL")
        if not isinstance(content, str) or not content.strip():
            return Decision(url, title, None, False, "empty content")
        if not isinstance(score, (int, float)):
            return Decision(url, title, None, False, "no relevance score")
        if score < self.min_score:
            return Decision(url, title, score, False, f"weak: score {score:.2f} below {self.min_score:.2f}")

        candidate = Source(url=url, title=title.strip(), content=content.strip(), score=float(score), query=query)
        key = normalize_url(url)
        existing_key, existing, how = self._find_duplicate(key, candidate)
        if existing is None:
            self._by_key[key] = candidate
            return Decision(url, candidate.title, candidate.score, True, "new source")
        if candidate.score <= existing.score:
            return Decision(url, candidate.title, candidate.score, False, f"duplicate ({how}) of {existing.url}")
        del self._by_key[existing_key]
        self._by_key[key] = candidate
        return Decision(url, candidate.title, candidate.score, True, f"replaces lower-scored duplicate ({how}) {existing.url}")

    def _find_duplicate(self, key: str, candidate: Source) -> tuple[str, Source | None, str]:
        if key in self._by_key:
            return key, self._by_key[key], "same page"
        words = _words(candidate.content)
        if len(words) < _MIN_WORDS_FOR_CONTENT_MATCH:
            return key, None, ""
        for other_key, other in self._by_key.items():
            if _jaccard(words, _words(other.content)) >= NEAR_DUPLICATE_JACCARD:
                return other_key, other, "same content"
        return key, None, ""

    def sources(self) -> list[Source]:
        return sorted(self._by_key.values(), key=lambda s: s.score, reverse=True)

    def check(self, *, min_sources: int, min_domains: int) -> EvidenceCheck:
        sources = len(self._by_key)
        domains = len({s.domain for s in self._by_key.values()})
        sufficient = sources >= min_sources and domains >= min_domains
        have = f"{sources} source{'s' if sources != 1 else ''} from {domains} domain{'s' if domains != 1 else ''}"
        reason = f"{have}; need at least {min_sources} sources from {min_domains} domains"
        return EvidenceCheck(sufficient, sources, domains, reason)
