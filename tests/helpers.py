from __future__ import annotations

import json
import random

from research_agent.search import make_result

_WORDS = (
    "airflow dagster scheduler sensor asset lineage partition backfill retry worker executor kubernetes "
    "celery docker pricing seat credit serverless hybrid deployment changelog release deprecation upgrade "
    "migration operator plugin python dbt snowflake bigquery spark warehouse observability alert metadata "
    "catalog freshness contract schema versioning branch environment secret webhook latency throughput "
    "queue concurrency quota limit tier enterprise support sla region compliance audit"
).split()


def page(url: str, *, score: float = 0.8, title: str | None = None, content: str | None = None) -> dict:
    """A Tavily result whose content is distinct per URL unless given."""
    if content is None:
        content = " ".join(random.Random(url).sample(_WORDS, 20)) + "."
    return make_result(title=title or f"Page {url}", url=url, content=content, score=score)


def plan(*queries: str) -> str:
    return json.dumps({"queries": list(queries)})


THREE_DOMAINS = [
    page("https://docs.dagster.io/deployment", score=0.9),
    page("https://airflow.apache.org/docs/stable/release_notes.html", score=0.8),
    page("https://dagster.io/pricing", score=0.7),
]
