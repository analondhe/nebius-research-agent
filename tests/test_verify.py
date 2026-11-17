from __future__ import annotations

import pytest
from helpers import page, plan, verdicts

from research_agent import prompts
from research_agent.llm import FakeLLMClient, LLMError
from research_agent.pipeline import Status, render, research
from research_agent.search import FakeSearchClient
from research_agent.verify import missing_numbers

Q = "Should we move from Airflow to Dagster?"
K8S = page(
    "https://docs.dagster.io/deployment/k8s",
    score=0.9,
    content="Dagster deploys to Kubernetes with the dagster-k8s Helm chart and also runs on Dagster+ Serverless.",
)
AIRFLOW = page(
    "https://airflow.apache.org/docs/release_notes",
    score=0.8,
    content="Airflow 3.0 introduced a new task execution interface and removed SubDAGs.",
)
PRICING = page(
    "https://dagster.io/pricing",
    score=0.7,
    content="Dagster+ Starter costs $100 per month and includes 30,000 credits.",
)


def run(answer: str, verifier_reply):
    llm = FakeLLMClient([plan("a"), answer, verifier_reply])
    return research(Q, llm, FakeSearchClient({"a": [K8S, AIRFLOW, PRICING]})), llm


def test_unsupported_claim_is_removed_and_citations_that_do_not_back_a_claim_are_dropped():
    result, _ = run(
        "Dagster runs on Kubernetes clusters [1][3]. Airflow 3 removed SubDAGs entirely [2]. "
        "Dagster is free forever for every team [3].",
        verdicts(("supported", [1]), ("supported", [2]), ("unsupported", [], "pricing page lists paid plans")),
    )

    assert result.status is Status.ANSWERED
    assert result.answer == "Dagster runs on Kubernetes clusters [1]. Airflow 3 removed SubDAGs entirely [2]."
    assert [c.url for c in result.sources] == [K8S["url"], AIRFLOW["url"]]
    assert result.notices == [
        "Claim check: 2 of 3 key claims fully supported by their cited sources; "
        "1 removed because their sources did not support them."
    ]
    removed = [e for e in result.trace.of("claim_check") if e["verdict"] == "unsupported"]
    assert removed[0]["claim"] == "Dagster is free forever for every team [3]."
    assert removed[0]["reason"] == "pricing page lists paid plans"
    assert removed[0]["level"] == "warning"


def test_partly_supported_claim_is_kept_and_marked_for_the_reader():
    result, _ = run(
        "Dagster deploys to Kubernetes and Nomad [1].",
        verdicts(("partial", [1], "Nomad is not mentioned")),
    )

    assert result.answer == "(Partly supported) Dagster deploys to Kubernetes and Nomad [1]."
    assert "1 marked (Partly supported)" in result.notices[0]
    assert render(result).endswith(result.notices[0])


def test_number_missing_from_the_cited_source_downgrades_support():
    result, _ = run(
        "Dagster+ Starter costs $120 per month [3]. The Starter plan includes 30,000 credits [3].",
        verdicts(("supported", [3]), ("supported", [3])),
    )

    assert result.answer == (
        "(Partly supported) Dagster+ Starter costs $120 per month [1]. The Starter plan includes 30,000 credits [1]."
    )
    checks = result.trace.of("claim_check")
    assert [c["verdict"] for c in checks] == ["partial", "supported"]
    assert checks[0]["reason"] == "not in cited source text: 120"


@pytest.mark.parametrize(
    ("claim", "source", "missing"),
    [
        ("Airflow 3 removed SubDAGs [2].", "Airflow 3.0 removed SubDAGs.", []),
        ("Starter costs $1,200 a year [1].", "Billed at 1200 USD per year.", []),
        ("Each deployment supports 50 workers [1].", "Each deployment supports 500 workers.", ["50"]),
        ("Version 2.10 added datasets [4].", "Version 2.1 added datasets.", ["2.10"]),
    ],
)
def test_missing_numbers_ignores_citation_ids_and_matches_whole_numbers(claim, source, missing):
    assert missing_numbers(claim, [source]) == missing


def test_claims_skipped_by_the_verifier_are_marked_unverified():
    result, _ = run(
        "Dagster runs on Kubernetes clusters [1]. Airflow 3 removed SubDAGs entirely [2].",
        verdicts(("supported", [1])),
    )

    assert result.answer == "Dagster runs on Kubernetes clusters [1]. (Unverified) Airflow 3 removed SubDAGs entirely [2]."
    assert "1 marked (Unverified)" in result.notices[0]


@pytest.mark.parametrize(
    "verifier_reply",
    [LLMError("Nebius Token Factory call failed: APITimeoutError"), "I think they all look fine."],
)
def test_failed_claim_check_keeps_the_answer_and_says_it_is_unverified(verifier_reply):
    answer = "Dagster runs on Kubernetes clusters [1]."
    result, _ = run(answer, verifier_reply)

    assert result.status is Status.ANSWERED
    assert result.answer == answer
    assert result.notices[0].startswith("Claim check: could not verify claims against their sources")
    assert any(w.startswith("claim check failed") for w in result.trace.warnings())


def test_no_supported_claim_means_not_enough_evidence():
    result, _ = run(
        "Dagster is free forever for every team [3]. Airflow is deprecated by Apache [2].",
        verdicts(("unsupported", []), ("unsupported", [])),
    )

    assert result.status is Status.INSUFFICIENT_EVIDENCE
    assert result.reason == "no claim was supported by its cited sources"
    assert result.answer == ""


def test_verifier_sees_only_cited_source_text_and_cannot_credit_uncited_sources():
    result, llm = run("Dagster runs on Kubernetes clusters [1].", verdicts(("supported", [2])))

    system, user = llm.calls[-1]["messages"]
    assert system["content"] == prompts.VERIFIER_SYSTEM
    assert "untrusted" in system["content"]
    assert K8S["content"] in user["content"]
    assert AIRFLOW["content"] not in user["content"] and PRICING["content"] not in user["content"]
    assert user["content"].endswith("Claim 1 (cites 1): Dagster runs on Kubernetes clusters.")
    # The verifier named source 2, which the claim never cited, so nothing supports it.
    assert result.status is Status.INSUFFICIENT_EVIDENCE
