from __future__ import annotations

from helpers import page

from research_agent.evidence import EvidencePool, normalize_url


def test_same_page_variants_collapse_to_the_highest_scored_copy():
    pool = EvidencePool(min_score=0.3)
    decisions = pool.add("q", [
        page("https://docs.dagster.io/deployment/?utm_source=x#k8s", score=0.5, content="first copy " * 10),
        page("http://www.docs.dagster.io/deployment", score=0.9, content="second copy " * 10),
        page("https://docs.dagster.io/deployment/", score=0.7, content="third copy " * 10),
    ])

    assert [d.kept for d in decisions] == [True, True, False]
    assert decisions[1].reason.startswith("replaces lower-scored duplicate (same page)")
    assert decisions[2].reason == "duplicate (same page) of http://www.docs.dagster.io/deployment"
    assert [(s.url, s.score) for s in pool.sources()] == [("http://www.docs.dagster.io/deployment", 0.9)]


def test_query_params_that_select_content_keep_pages_distinct():
    assert normalize_url("https://x.io/docs?version=2&utm_medium=a") == "x.io/docs?version=2"
    assert normalize_url("https://x.io/docs?version=2") != normalize_url("https://x.io/docs?version=3")


def test_syndicated_copies_on_other_domains_are_duplicates():
    text = "Dagster Plus pricing starts with a free solo plan and a starter plan billed per credit for serverless runs."
    pool = EvidencePool(min_score=0.3)
    decisions = pool.add("q", [
        page("https://dagster.io/pricing", score=0.9, content=text),
        page("https://news.example.com/dagster-pricing", score=0.6, content=text + " Read more."),
    ])

    assert decisions[1].kept is False
    assert decisions[1].reason == "duplicate (same content) of https://dagster.io/pricing"


def test_short_snippets_are_not_matched_on_content():
    pool = EvidencePool(min_score=0.3)
    pool.add("q", [page("https://a.io/x", content="Pricing page."), page("https://b.io/y", content="Pricing page.")])
    assert len(pool.sources()) == 2


def test_weak_malformed_and_empty_results_are_dropped_with_reasons():
    pool = EvidencePool(min_score=0.3)
    decisions = pool.add("q", [
        page("https://a.io", score=0.29),
        page("https://b.io", content=""),
        {"title": "no url", "content": "text", "score": 0.9},
        {"url": "https://c.io", "content": "text", "score": None},
        "not a dict",
    ])

    assert [d.reason for d in decisions] == [
        "weak: score 0.29 below 0.30",
        "empty content",
        "no usable URL",
        "no relevance score",
        "no usable URL",
    ]
    assert pool.sources() == []


def test_sufficiency_needs_both_source_and_domain_counts():
    pool = EvidencePool(min_score=0.3)
    pool.add("q", [page("https://a.io/1"), page("https://a.io/2"), page("https://a.io/3")])
    assert not pool.check(min_sources=3, min_domains=2).sufficient

    pool.add("q", [page("https://b.io/1")])
    check = pool.check(min_sources=3, min_domains=2)
    assert check.sufficient
    assert check.reason == "4 sources from 2 domains; need at least 3 sources from 2 domains"
