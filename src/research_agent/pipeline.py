"""planner -> search -> evidence check (dedupe, drop weak, re-search) -> writer.

Every step is recorded in a Trace: planned queries, search results, why each
source was kept or dropped, evidence checks, why searching stopped, and what
the writer produced.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import Enum

from . import citations, prompts
from .evidence import EvidenceCheck, EvidencePool, Source, parse_results
from .llm import LLMClient, LLMError, Message, completion_text
from .search import SearchClient, SearchError
from .trace import Trace

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
    trace: Trace


def research(
    question: str,
    llm: LLMClient,
    search: SearchClient,
    config: AgentConfig = AgentConfig(),
    trace: Trace | None = None,
) -> ResearchResult:
    return _Run(question.strip(), llm, search, config, trace if trace is not None else Trace()).execute()


class _Run:
    def __init__(self, question: str, llm: LLMClient, search: SearchClient, config: AgentConfig, trace: Trace) -> None:
        self.question = question
        self.llm = llm
        self.search = search
        self.config = config
        self.trace = trace
        self.pool = EvidencePool(config.min_score)
        self.tried: list[str] = []
        self.failed_searches = 0

    def execute(self) -> ResearchResult:
        self.trace.record(
            "run_started", f"question: {self.question}", question=self.question, model=self.llm.model, config=asdict(self.config)
        )
        check = self._search_until_enough()
        found = _numbered(self.pool.sources())
        if not found:
            if self.tried and self.failed_searches == len(self.tried):
                return self._finish(Status.FAILED, "", "every search call failed", [])
            return self._finish(Status.INSUFFICIENT_EVIDENCE, "", "search returned no usable sources", [])
        if not check.sufficient:
            return self._finish(Status.INSUFFICIENT_EVIDENCE, "", check.reason, found)
        return self._write(self.pool.sources())

    # -- planning and searching ------------------------------------------------

    def _search_until_enough(self) -> EvidenceCheck:
        queries = self._plan()
        check = self.pool.check(min_sources=self.config.min_sources, min_domains=self.config.min_domains)
        rounds_run, stop = 0, ""
        for round_no in range(1, self.config.max_rounds + 1):
            if not queries:
                stop = "no new queries to try"
                break
            rounds_run = round_no
            for query in queries:
                self._search_one(round_no, query)
            check = self.pool.check(min_sources=self.config.min_sources, min_domains=self.config.min_domains)
            self.trace.record(
                "evidence_check",
                f"round {round_no}: {'sufficient' if check.sufficient else 'insufficient'} ({check.reason})",
                round=round_no,
                sufficient=check.sufficient,
                sources=check.sources,
                domains=check.domains,
                reason=check.reason,
            )
            if check.sufficient:
                stop = f"enough evidence after round {round_no}"
                break
            if round_no == self.config.max_rounds:
                stop = f"reached the limit of {self.config.max_rounds} search rounds"
                break
            queries = self._refine(round_no, check)
        self.trace.record(
            "stop", f"stopped searching: {stop}", reason=stop, rounds=rounds_run, searches=len(self.tried)
        )
        return check

    def _plan(self) -> list[str]:
        messages = [
            {"role": "system", "content": prompts.PLANNER_SYSTEM},
            {"role": "user", "content": prompts.planner_user(self.question, self.config.max_queries)},
        ]
        try:
            proposed = parse_queries(completion_text(self.llm.complete(messages, max_tokens=300, json_mode=True)))
            error = "" if proposed else "planner returned no queries"
        except (LLMError, ValueError) as exc:
            proposed, error = [], f"planner failed: {exc}"
        if error:
            fallback = [self.question[:MAX_QUERY_CHARS]]
            self.trace.warn("plan", f"{error}; searching the question as written", queries=fallback, error=error)
            return fallback
        return self._select("plan", 1, proposed)

    def _refine(self, round_no: int, check: EvidenceCheck) -> list[str]:
        found = [(s.title, s.domain) for s in self.pool.sources()]
        user = prompts.refine_user(self.question, self.tried, found, check.reason, self.config.max_queries)
        messages = [{"role": "system", "content": prompts.REFINE_SYSTEM}, {"role": "user", "content": user}]
        try:
            proposed = parse_queries(completion_text(self.llm.complete(messages, max_tokens=300, json_mode=True)))
        except (LLMError, ValueError) as exc:
            self.trace.warn("refine", f"follow-up planning failed: {exc}", round=round_no + 1, queries=[], error=str(exc))
            return []
        return self._select("refine", round_no + 1, proposed)

    def _select(self, event: str, round_no: int, proposed: list[str]) -> list[str]:
        """Keep proposed queries that are new, up to the per-round limit; trace the rest with a reason."""
        tried = {q.casefold() for q in self.tried}
        selected, discarded = [], []
        for query in proposed:
            if query.casefold() in tried:
                discarded.append({"query": query, "reason": "already searched"})
            elif len(selected) >= self.config.max_queries:
                discarded.append({"query": query, "reason": f"over the {self.config.max_queries}-query limit per round"})
            else:
                selected.append(query)
        message = f"round {round_no} queries: " + ("; ".join(selected) if selected else "(none)")
        if discarded:
            message += f" (discarded {len(discarded)})"
        self.trace.record(event, message, round=round_no, queries=selected, discarded=discarded)
        return selected

    def _search_one(self, round_no: int, query: str) -> None:
        self.tried.append(query)
        try:
            body = self.search.search(query, max_results=self.config.results_per_query)
            results = parse_results(body)
        except SearchError as exc:
            self.failed_searches += 1
            self.trace.warn("search", f"search failed for {query!r}: {exc}", round=round_no, query=query, error=str(exc))
            return
        self.trace.record(
            "search",
            f"{query!r}: {len(results)} result{'s' if len(results) != 1 else ''}",
            round=round_no,
            query=query,
            request_id=body.get("request_id"),
            results=[_result_summary(r) for r in results],
        )
        for d in self.pool.add(query, results):
            verdict = "kept" if d.kept else "dropped"
            self.trace.record(
                "source",
                f"{verdict}: {d.url} ({d.reason})",
                round=round_no,
                query=query,
                url=d.url,
                title=d.title,
                score=d.score,
                kept=d.kept,
                reason=d.reason,
            )

    # -- writing ---------------------------------------------------------------

    def _write(self, sources: list[Source]) -> ResearchResult:
        n = len(sources)
        numbered = [(i, s.title, s.url, s.content) for i, s in enumerate(sources, 1)]
        messages: list[Message] = [
            {"role": "system", "content": prompts.WRITER_SYSTEM},
            {"role": "user", "content": prompts.writer_user(self.question, numbered)},
        ]
        try:
            draft = completion_text(self.llm.complete(messages, max_tokens=1500))
        except LLMError as exc:
            self.trace.warn("writer", f"writer failed: {exc}", attempt=1, error=str(exc))
            return self._finish(Status.FAILED, "", f"writer failed: {exc}", [])
        report = citations.check(draft, n)
        self._trace_draft(1, draft, report, n)

        if not _is_insufficient(draft) and not report.ok:
            messages += [
                {"role": "assistant", "content": draft},
                {"role": "user", "content": prompts.writer_fix(report.problems())},
            ]
            try:
                draft = completion_text(self.llm.complete(messages, max_tokens=1500))
                report = citations.check(draft, n)
                self._trace_draft(2, draft, report, n)
            except LLMError as exc:
                self.trace.warn("writer", f"citation repair failed: {exc}; using first draft", attempt=2, error=str(exc))

        if _is_insufficient(draft):
            return self._finish(Status.INSUFFICIENT_EVIDENCE, "", "writer found the sources insufficient", _numbered(sources))
        if not report.ok:
            draft, dropped = citations.drop_uncited(draft, n)
            for claim in dropped:
                self.trace.warn("claim_removed", f"removed uncited claim: {claim!r}", claim=claim)
        if not citations.key_claims(draft):
            return self._finish(
                Status.INSUFFICIENT_EVIDENCE, "", "no claim in the answer had a valid citation", _numbered(sources)
            )
        answer, order = citations.renumber(draft, n)
        cited = [Citation(new, sources[old - 1].title, sources[old - 1].url) for new, old in enumerate(order, 1)]
        not_cited = [s.url for i, s in enumerate(sources, 1) if i not in order]
        return self._finish(Status.ANSWERED, answer, "", cited, not_cited=not_cited)

    def _trace_draft(self, attempt: int, draft: str, report: citations.CitationReport, n: int) -> None:
        if _is_insufficient(draft):
            self.trace.record("writer", f"draft {attempt}: writer reported insufficient evidence", attempt=attempt)
            return
        claims = len(citations.key_claims(draft))
        message = f"draft {attempt}: {claims} key claims from {n} sources"
        if report.ok:
            self.trace.record("writer", f"{message}, all cited", attempt=attempt, claims=claims, problems=[])
        else:
            self.trace.warn("writer", f"{message}; {'; '.join(report.problems())}", attempt=attempt, claims=claims,
                            problems=report.problems())

    def _finish(
        self, status: Status, answer: str, reason: str, sources: list[Citation], not_cited: list[str] | None = None
    ) -> ResearchResult:
        fields = {"status": status.value, "reason": reason, "sources": [c.url for c in sources]}
        if not_cited is not None:
            fields["not_cited"] = not_cited
        self.trace.record("finished", f"{status.value}" + (f": {reason}" if reason else f" with {len(sources)} cited sources"),
                          **fields)
        return ResearchResult(self.question, status, answer, reason, sources, list(self.tried), self.trace)


def _result_summary(raw: object) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    return {"title": raw.get("title"), "url": raw.get("url"), "score": raw.get("score")}


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
    queries: list[str] = []
    for q in data:
        if isinstance(q, str) and q.strip() and q.strip().casefold() not in {x.casefold() for x in queries}:
            queries.append(q.strip()[:MAX_QUERY_CHARS])
    return queries


def _is_insufficient(text: str) -> bool:
    return text.strip().startswith(prompts.INSUFFICIENT_EVIDENCE)


def _numbered(sources: list[Source]) -> list[Citation]:
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
