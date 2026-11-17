"""Claim-level citation check: every key claim must be backed by the text of a source it cites.

One model call judges all claims against their cited source text. Then:
- supported claims keep only the citations whose source backs them;
- partly supported claims stay, marked "(Partly supported)";
- unsupported claims are removed;
- claims the verifier skipped stay, marked "(Unverified)".
A "supported" verdict is downgraded to partial when the claim has a number
(price, version, limit, date) that appears in none of its supporting sources.
If the check itself fails, the answer is kept unchanged and the failure is reported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from . import citations, prompts
from .evidence import Source
from .llm import LLMClient, LLMError, completion_text, parse_json

SUPPORTED = "supported"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"
UNVERIFIED = "unverified"
PARTIAL_MARK = "(Partly supported)"
UNVERIFIED_MARK = "(Unverified)"

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass(frozen=True)
class ClaimCheck:
    claim: str
    cited: list[int]
    verdict: str
    supported_by: list[int]
    reason: str


@dataclass
class Verification:
    text: str
    checks: list[ClaimCheck] = field(default_factory=list)
    error: str = ""

    def count(self, verdict: str) -> int:
        return sum(c.verdict == verdict for c in self.checks)


def verify_claims(text: str, sources: Sequence[Source], llm: LLMClient) -> Verification:
    """`text` cites sources by 1-based position in `sources`; every key claim already has a valid id."""
    claims = citations.key_claims(text)
    cited = [[i for i in dict.fromkeys(citations.cited_ids(c)) if 1 <= i <= len(sources)] for c in claims]
    used = sorted({i for ids in cited for i in ids})
    messages = [
        {"role": "system", "content": prompts.VERIFIER_SYSTEM},
        {
            "role": "user",
            "content": prompts.verifier_user(
                [(i, sources[i - 1].title, sources[i - 1].url, sources[i - 1].content) for i in used],
                [(n, ids, _plain(claim)) for n, (claim, ids) in enumerate(zip(claims, cited), 1)],
            ),
        },
    ]
    try:
        verdicts = parse_verdicts(completion_text(llm.complete(messages, max_tokens=2000, json_mode=True)))
    except (LLMError, ValueError) as exc:
        return Verification(text, error=str(exc))

    checks = [
        _judge(claim, ids, verdicts.get(n), sources) for n, (claim, ids) in enumerate(zip(claims, cited), 1)
    ]
    pending = iter(checks)

    def apply(sentence: str) -> str | None:
        check = next(pending)
        if check.verdict == UNSUPPORTED:
            return None
        if check.verdict == UNVERIFIED:
            return f"{UNVERIFIED_MARK} {sentence}"
        backed = citations.keep_ids(sentence, lambda i: i in check.supported_by)
        return f"{PARTIAL_MARK} {backed}" if check.verdict == PARTIAL else backed

    return Verification(citations.rewrite_claims(text, apply), checks)


def parse_verdicts(text: str) -> dict[int, tuple[str, list[int], str]]:
    """claim number -> (verdict, supporting ids, reason); raises ValueError if nothing usable."""
    data = parse_json(text)
    items = data.get("results") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("verifier output has no results list")
    verdicts: dict[int, tuple[str, list[int], str]] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("claim"), int):
            continue
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict not in (SUPPORTED, PARTIAL, UNSUPPORTED):
            continue
        ids = [i for i in item.get("sources") or [] if isinstance(i, int)]
        verdicts[item["claim"]] = (verdict, ids, str(item.get("reason", "")).strip())
    if not verdicts:
        raise ValueError("verifier output has no usable verdicts")
    return verdicts


def _judge(
    claim: str, cited: list[int], verdict: tuple[str, list[int], str] | None, sources: Sequence[Source]
) -> ClaimCheck:
    if verdict is None:
        return ClaimCheck(claim, cited, UNVERIFIED, [], "verifier returned no verdict for this claim")
    label, ids, reason = verdict
    backing = [i for i in dict.fromkeys(ids) if i in cited]
    if label == UNSUPPORTED or not backing:
        return ClaimCheck(claim, cited, UNSUPPORTED, [], reason or "no cited source supports it")
    if label == SUPPORTED:
        missing = missing_numbers(claim, [sources[i - 1].content for i in backing])
        if missing:
            return ClaimCheck(claim, cited, PARTIAL, backing, f"not in cited source text: {', '.join(missing)}")
    return ClaimCheck(claim, cited, label, backing, reason)


def missing_numbers(claim: str, texts: Sequence[str]) -> list[str]:
    """Numbers in the claim found in none of `texts`. "3" matches "3.0" so versions are not over-flagged."""
    have = {n.replace(",", "") for t in texts for n in _NUMBER.findall(t)}
    wanted = dict.fromkeys(n.replace(",", "") for n in _NUMBER.findall(_plain(claim)))
    return [n for n in wanted if n not in have and not any(h.startswith(n + ".") for h in have)]


def _plain(claim: str) -> str:
    text = re.sub(r"\s+", " ", citations.CITATION.sub("", claim))
    return re.sub(r" ([.,;:!?])", r"\1", text).strip()
