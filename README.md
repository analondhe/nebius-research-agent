# <Project Name> — Research Agent

> ODSC × Nebius Hackathon submission · Challenge: *Build a research agent that brings its sources*

**One-line pitch:** <What question does your agent answer, and for whom?>

- **Demo video (2–3 min):** <link>
- **Team:** <names / GitHub handles>

## The problem
<Who is the user? What research task does the agent help them finish? Why does it matter?>

## How it works
```
question → planner (Nebius) → Tavily search → evidence check ──enough?──► writer (Nebius) → answer with [n] citations
                                   ▲                │ no
                                   └── follow-up queries (Nebius), up to max rounds ──► "not enough evidence"
```
Code lives in `src/research_agent/`; `src/agent.py` is the CLI.

- **Query planning** (`pipeline.py`, `prompts.py`): the model turns the question into 2–3 queries aimed at official docs, changelogs and pricing pages. If the evidence check fails, the model sees the tried queries, the sources found and the gap, and proposes new queries. Repeats are filtered out. The run stops when evidence is sufficient, when the round limit is reached (2 by default), or when no new queries come back.
- **Evidence handling** (`evidence.py`): drops results with no URL, no content or a Tavily score below 0.3. Deduplicates pages by normalized URL (scheme, `www.`, fragments, trailing slashes and tracking parameters are ignored), and syndicated copies by content overlap; the higher-scored copy wins. Evidence counts as sufficient with at least 3 sources from at least 2 domains.
- **Citations** (`citations.py`): every key claim (a sentence of 4+ words that isn't a heading or lead-in) must cite a source id that exists. If a draft breaks the rule, the writer gets one repair round. Claims still uncited after that are removed. If no cited claim survives, the answer becomes "not enough evidence". Citations are renumbered in order of first use, and only cited sources are listed.
- **Safety** (`prompts.py`): instructions live only in system messages. Web text goes into user messages inside `<source>` tags, marked as untrusted. Text that would open or close a tag is neutralized.
- **Failure handling**: when one search fails the run continues with the other queries. If every search fails, or the writer call fails, the run ends with a clear "Research failed" message instead of a traceback. A failed planner call falls back to searching the question as written.
- **Trade-offs:** <latency, cost, reliability>

## Required services
| Requirement | How we use it |
|---|---|
| Nebius Token Factory (inference) | Model: `meta-llama/Llama-3.3-70B-Instruct` by default (`NEBIUS_MODEL`). It plans queries, plans follow-up searches and writes the cited answer, through the OpenAI-compatible API at `https://api.tokenfactory.nebius.com/v1/` (`NEBIUS_BASE_URL`). |
| Tavily (real-time web search) | `search` with `search_depth=basic`, 5 results per query. Each result's title, URL, snippet and score become a numbered source, and the URL is printed next to its `[n]` in the answer. |

## Setup
```bash
git clone <repo-url>
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your keys — never commit them
python src/agent.py "Should we move from Airflow to Dagster?"
```
Exit codes: `0` answered, `3` not enough evidence, `1` research failed, `2` missing keys.

## Tests
Both services sit behind small interfaces (`LLMClient` in `llm.py`, `SearchClient` in `search.py`). Each has a real implementation and an offline fake that returns the same response bodies as the real APIs. The CLI always uses the real clients and reads the keys from `.env`. Fakes are used only when `RESEARCH_AGENT_USE_FAKES=1` is set.
```bash
pip install -r requirements-dev.txt
pytest
RESEARCH_AGENT_USE_FAKES=1 python src/agent.py "Should we move from Airflow to Dagster?"   # offline run
```

## Example
**Question:** <example question>
**Answer (excerpt):** <answer with [1], [2] citations>
**Sources:** [1] <url> · [2] <url>

## Failure / insufficient-evidence case
<Show what the agent does when search returns nothing useful or a tool fails.>

## Judging criteria map
| Criterion | Weight | Where to look |
|---|---|---|
| Sourced and verifiable answers | 30% | Example above, demo 0:00–… |
| Works end to end | 25% | Setup, demo |
| Usefulness | 20% | The problem |
| Agent design | 15% | How it works |
| Use of Token Factory and Tavily | 10% | Required services |

## License
MIT
