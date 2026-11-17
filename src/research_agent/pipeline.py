"""planner -> search -> evidence check (dedupe, drop weak, re-search) -> writer."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

from . import citations, prompts
from .evidence import EvidenceCheck, EvidencePool, Source, parse_results
from .llm import LLMClient, LLMError, Message, completion_text
from .search import SearchClient, SearchError

MAX_QUERY_CHARS = 400  # Tavily rejects longer queries


@dataclass(frozen=True)
class AgentConfig:
    max_queries: int = 3  # per round
    results_per_query: int = 5
    max_rounds: int = 2  # first search round plus re-search rounds
    min_score: float = 0.3
    min_sources: int = 3
    min_domains: int = 2


class Status(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


@dataclass(frozen=True)
class Citation:
    id: int
    title: str
    url: str


@dataclass
class ResearchResult:
    question: str
    status: Status
    answer: str
    reason: str
    sources: list[Citation]
    queries: list[str]
    notes: list[str] = field(default_factory=list)


def research(
    question: str,
    llm: LLMClient,
    search: SearchClient,
    config: AgentConfig = AgentConfig(),
) -> ResearchResult:
    question = question.strip()
    notes: list[str] = []
    pool = EvidencePool(config.min_score)
    tried: list[str] = []
    failed_searches = 0

    queries = _plan(question, llm, config, notes)
    check = pool.check(min_sources=config.min_sources, min_domains=config.min_domains)
    for round_no in range(1, config.max_rounds + 1):
        fresh = _unseen(queries, tried)[: config.max_queries]
        if not fresh:
            notes.append("planner proposed no new queries")
            break
        for query in fresh:
            tried.append(query)
            try:
                results = parse_results(search.search(query, max_results=config.results_per_query))
            except SearchError as exc:
                failed_searches += 1
                notes.append(f"search failed for {query!r}: {exc}")
                continue
            pool.add(query, results)
        check = pool.check(min_sources=config.min_sources, min_domains=config.min_domains)
        if check.sufficient or round_no == config.max_rounds:
            break
        queries = _refine(question, tried, pool, check, llm, config, notes)

    def result(status: Status, answer: str, reason: str, sources: list[Citation]) -> ResearchResult:
        return ResearchResult(question, status, answer, reason, sources, tried, notes)

    found = [Citation(i, s.title, s.url) for i, s in enumerate(pool.sources(), start=1)]
    if not found:
        if tried and failed_searches == len(tried):
            return result(Status.FAILED, "", "every search call failed", [])
        return result(Status.INSUFFICIENT_EVIDENCE, "", "search returned no usable sources", [])
    if not check.sufficient:
        return result(Status.INSUFFICIENT_EVIDENCE, "", check.reason, found)
    return _write(question, pool.sources(), llm, notes, result)


def _plan(question: str, llm: LLMClient, config: AgentConfig, notes: list[str]) -> list[str]:
    messages = [
        {"role": "system", "content": prompts.PLANNER_SYSTEM},
        {"role": "user", "content": prompts.planner_user(question, config.max_queries)},
    ]
    try:
        queries = parse_queries(completion_text(llm.complete(messages, max_tokens=300, json_mode=True)))
    except (LLMError, ValueError) as exc:
        notes.append(f"planner failed ({exc}); searching the question as written")
        return [question[:MAX_QUERY_CHARS]]
    if not queries:
        notes.append("planner returned no queries; searching the question as written")
        return [question[:MAX_QUERY_CHARS]]
    return queries


def _refine(
    question: str,
    tried: list[str],
    pool: EvidencePool,
    check: EvidenceCheck,
    llm: LLMClient,
    config: AgentConfig,
    notes: list[str],
) -> list[str]:
    found = [(s.title, s.domain) for s in pool.sources()]
    messages = [
        {"role": "system", "content": prompts.REFINE_SYSTEM},
        {"role": "user", "content": prompts.refine_user(question, tried, found, check.reason, config.max_queries)},
    ]
    try:
        return parse_queries(completion_text(llm.complete(messages, max_tokens=300, json_mode=True)))
    except (LLMError, ValueError) as exc:
        notes.append(f"follow-up planning failed ({exc})")
        return []


def parse_queries(text: str) -> list[str]:
    """Queries from {"queries": [...]} or a bare JSON list, tolerating prose or code fences around it."""
    data = None
    for candidate in (text, *re.findall(r"\{.*\}|\[.*\]", text, flags=re.DOTALL)):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if isinstance(data, dict):
        data = data.get("queries")
    if not isinstance(data, list):
        raise ValueError("planner output is not a JSON list of queries")
    queries = [q.strip()[:MAX_QUERY_CHARS] for q in data if isinstance(q, str) and q.strip()]
    return _unseen(queries, [])


def _unseen(queries: list[str], tried: list[str]) -> list[str]:
    seen = {q.casefold() for q in tried}
    out = []
    for q in queries:
        if q.casefold() not in seen:
            seen.add(q.casefold())
            out.append(q)
    return out


def _write(question, sources: list[Source], llm: LLMClient, notes: list[str], result) -> ResearchResult:
    n = len(sources)
    messages: list[Message] = [
        {"role": "system", "content": prompts.WRITER_SYSTEM},
        {
            "role": "user",
            "content": prompts.writer_user(question, [(i, s.title, s.url, s.content) for i, s in enumerate(sources, 1)]),
        },
    ]
    try:
        draft = completion_text(llm.complete(messages, max_tokens=1500))
    except LLMError as exc:
        return result(Status.FAILED, "", f"writer failed: {exc}", [])

    report = citations.check(draft, n)
    if not _is_insufficient(draft) and not report.ok:
        messages += [{"role": "assistant", "content": draft}, {"role": "user", "content": prompts.writer_fix(report.problems())}]
        try:
            draft = completion_text(llm.complete(messages, max_tokens=1500))
        except LLMError as exc:
            notes.append(f"citation repair call failed ({exc}); using first draft")

    if _is_insufficient(draft):
        return result(Status.INSUFFICIENT_EVIDENCE, "", "writer found the sources insufficient", _all(sources))

    if not citations.check(draft, n).ok:
        draft, dropped = citations.drop_uncited(draft, n)
        notes.extend(f"removed uncited claim: {s!r}" for s in dropped)
    if not citations.key_claims(draft):
        return result(Status.INSUFFICIENT_EVIDENCE, "", "no claim in the answer had a valid citation", _all(sources))

    answer, order = citations.renumber(draft, n)
    cited = [Citation(new, sources[old - 1].title, sources[old - 1].url) for new, old in enumerate(order, 1)]
    return result(Status.ANSWERED, answer, "", cited)


def _is_insufficient(text: str) -> bool:
    return text.strip().startswith(prompts.INSUFFICIENT_EVIDENCE)


def _all(sources: list[Source]) -> list[Citation]:
    return [Citation(i, s.title, s.url) for i, s in enumerate(sources, 1)]


def render(result: ResearchResult) -> str:
    if result.status is Status.ANSWERED:
        refs = "\n".join(f"[{c.id}] {c.title} — {c.url}" for c in result.sources)
        return f"{result.answer}\n\nSources:\n{refs}"
    if result.status is Status.INSUFFICIENT_EVIDENCE:
        out = f"Not enough evidence to answer this question.\nReason: {result.reason}."
        if result.sources:
            out += "\n\nSources found (not enough to support an answer):\n"
            out += "\n".join(f"- {c.title} — {c.url}" for c in result.sources)
        return out
    return f"Research failed: {result.reason}."
