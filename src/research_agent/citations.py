"""Citation rules for the writer: every key claim carries a [n] that points at a real source.

A key claim is a sentence of four or more words that is not a heading and does
not end in a colon (a lead-in such as "Key differences:").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

_CITE = r"\[\d+(?:\s*,\s*\d+)*\]"
CITATION = re.compile(_CITE)
# A sentence ends at . ! or ? (plus any citations right after it) followed by whitespace or end of text,
# so decimals such as "3.5" and citations placed after the full stop stay with their sentence.
_BOUNDARY = re.compile(r"[.!?]+(?:\s*" + _CITE + r")*(?=\s|$)")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_MIN_CLAIM_WORDS = 4


def split_sentences(text: str) -> list[str]:
    sentences, start = [], 0
    for match in _BOUNDARY.finditer(text):
        sentences.append(text[start : match.end()].strip())
        start = match.end()
    sentences.append(text[start:].strip())
    return [s for s in sentences if s]


def cited_ids(sentence: str) -> list[int]:
    return [int(n) for group in CITATION.findall(sentence) for n in re.findall(r"\d+", group)]


def is_key_claim(sentence: str) -> bool:
    plain = re.sub(r"[*_`]", "", CITATION.sub("", sentence)).strip()
    return not plain.endswith(":") and len(plain.split()) >= _MIN_CLAIM_WORDS


def rewrite_claims(text: str, rewrite: Callable[[str], str | None]) -> str:
    """Pass every key claim, in order, through `rewrite`; None removes the claim.
    Headings, blank lines and short non-claim sentences are left as they are."""
    out = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            out.append(line)
            continue
        bullet = _BULLET.match(line)
        prefix = bullet.group(0) if bullet else ""
        kept = []
        for sentence in split_sentences(line[len(prefix) :]):
            if is_key_claim(sentence):
                sentence = rewrite(sentence)
                if sentence is None:
                    continue
            kept.append(sentence)
        if kept:
            out.append(prefix + " ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def key_claims(text: str) -> list[str]:
    claims: list[str] = []

    def collect(sentence: str) -> str:
        claims.append(sentence)
        return sentence

    rewrite_claims(text, collect)
    return claims


@dataclass
class CitationReport:
    uncited: list[str] = field(default_factory=list)
    invalid_ids: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.uncited and not self.invalid_ids

    def problems(self) -> list[str]:
        out = [f"No valid citation: \"{s}\"" for s in self.uncited]
        if self.invalid_ids:
            out.append(f"Cited source ids that do not exist: {', '.join(map(str, self.invalid_ids))}")
        return out


def check(text: str, n_sources: int) -> CitationReport:
    report = CitationReport()
    for claim in key_claims(text):
        ids = cited_ids(claim)
        bad = [i for i in ids if not 1 <= i <= n_sources]
        report.invalid_ids.extend(i for i in bad if i not in report.invalid_ids)
        if len(bad) == len(ids):
            report.uncited.append(claim)
    return report


def keep_ids(text: str, allowed: Callable[[int], bool]) -> str:
    """Drop citation ids that fail `allowed`, and any citation group left empty."""

    def repl(match: re.Match[str]) -> str:
        ids = [i for i in cited_ids(match.group(0)) if allowed(i)]
        return "".join(f"[{i}]" for i in dict.fromkeys(ids))

    return re.sub(r"[ \t]+([.,;:!?])", r"\1", CITATION.sub(repl, text))


def drop_uncited(text: str, n_sources: int) -> tuple[str, list[str]]:
    """Remove ids that point nowhere, then every key claim left without a citation."""
    dropped: list[str] = []

    def keep_if_cited(sentence: str) -> str | None:
        if cited_ids(sentence):
            return sentence
        dropped.append(sentence)
        return None

    return rewrite_claims(keep_ids(text, lambda i: 1 <= i <= n_sources), keep_if_cited), dropped


def renumber(text: str, n_sources: int) -> tuple[str, list[int]]:
    """Number cited sources 1..k in order of first citation; returns the new text and
    the original ids in their new order."""
    order: list[int] = []
    for i in cited_ids(text):
        if 1 <= i <= n_sources and i not in order:
            order.append(i)
    new_id = {old: new for new, old in enumerate(order, start=1)}

    def repl(match: re.Match[str]) -> str:
        ids = dict.fromkeys(new_id[i] for i in cited_ids(match.group(0)) if i in new_id)
        return "".join(f"[{i}]" for i in ids)

    return CITATION.sub(repl, text), order
