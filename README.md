# <Project Name> — Research Agent

> ODSC × Nebius Hackathon submission · Challenge: *Build a research agent that brings its sources*

**One-line pitch:** <What question does your agent answer, and for whom?>

- **Demo video (2–3 min):** <link>
- **Team:** <names / GitHub handles>

## The problem
**Who it's for:** data engineers and platform teams deciding whether to adopt, replace or upgrade a tool in their stack. For example, an orchestrator, a warehouse, an ingestion service or a table format.

**The research task:** answering a question like *"Should we move from Airflow to Dagster?"* or *"Is it worth switching from Fivetran to Airbyte for 200 connectors?"*. A good answer pulls from:
- official documentation, for what each tool actually supports and how it's deployed;
- changelogs and release notes, for what changed recently, what's deprecated and what breaks on upgrade;
- pricing pages, for how each tool bills and where the plan limits are.

The agent plans the searches, collects those sources, drops weak and duplicate ones, searches again when there are gaps, and writes a short answer in which every key claim links to the page it came from. When the sources don't support an answer, it says so instead of guessing.

**Why it matters:**
- **The facts change fast.** Features ship, defaults change and pricing tiers get reworked from one release to the next. Last year's blog post or a model's training data is often out of date, and the decision has to rest on the current docs.
- **The cost of getting it wrong is high.** A migration takes weeks of engineering time. One wrong claim about a missing feature, a breaking change or a price can send a team down the wrong path.
- **Decisions need receipts.** A recommendation goes to teammates, managers and procurement. Each claim needs a source someone can open and check, not a confident paragraph with no links.
- **Doing it by hand is slow.** Comparing two tools means reading dozens of docs pages, changelogs and pricing tables, and cross-checking them. This agent does that first pass for you and shows its work, through citations and the research trace, so a reviewer can verify it.

## How it works
```
question → planner (Nebius) → Tavily search → evidence check ──enough?──► writer (Nebius) → claim check (Nebius) → answer with [n] citations
                                   ▲                │ no
                                   └── follow-up queries (Nebius), up to max rounds ──► "not enough evidence"
```
Code lives in `src/research_agent/`; `src/agent.py` is the CLI.

- **Query planning** (`pipeline.py`, `prompts.py`): the model turns the question into 2–3 queries aimed at official docs, changelogs and pricing pages. If the evidence check fails, the model sees the tried queries, the sources found and the gap, and proposes new queries. Repeats are filtered out. The run stops when evidence is sufficient, when the round limit is reached (2 by default), or when no new queries come back.
- **Evidence handling** (`evidence.py`): drops results with no URL, no content or a Tavily score below 0.3. Deduplicates pages by normalized URL (scheme, `www.`, fragments, trailing slashes and tracking parameters are ignored), and syndicated copies by content overlap; the higher-scored copy wins. Evidence counts as sufficient with at least 3 sources from at least 2 domains.
- **Citations** (`citations.py`): every key claim (a sentence of 4+ words that isn't a heading or lead-in) must cite a source id that exists. If a draft breaks the rule, the writer gets one repair round. Claims still uncited after that are removed. If no cited claim survives, the answer becomes "not enough evidence". Citations are renumbered in order of first use, and only cited sources are listed.
- **Claim check** (`verify.py`): after writing, one more model call judges every key claim against the text of the sources it cites.
  - Supported claims keep only the citations whose source backs them.
  - Partly supported claims stay, marked `(Partly supported)`.
  - Unsupported claims are removed.
  - Claims the checker skipped are marked `(Unverified)`.
  - A "supported" verdict is downgraded to partly supported when the claim has a number (price, version, limit) that doesn't appear in any of its supporting sources.
  - If the check itself fails, the answer is kept and labelled unverified.
  - If no claim survives, the result is "not enough evidence".
  - A one-line summary under the sources says how many claims were fully supported, marked or removed.
- **Safety** (`prompts.py`): instructions live only in system messages. Web text goes into user messages inside `<source>` tags, marked as untrusted. Text that would open or close a tag is neutralized.
- **Failure handling**: when one search fails the run continues with the other queries. If every search fails, or the writer call fails, the run ends with a clear "Research failed" message instead of a traceback. A failed planner call falls back to searching the question as written.
- **Research trace** (`trace.py`): every run records each planned query (and any dropped as repeats or over the per-round limit), the results each search returned, whether each source was kept or dropped and why, the evidence check after each round, the stop reason, the writer's citation checks, and the final status. Warnings (failed calls, fallbacks, removed claims) always go to stderr. `--trace` prints the full trace and `--trace-file run.jsonl` saves it as JSON lines. API keys from the environment, and key-shaped strings (`tvly-…`, `Bearer …`, `api_key=…`), are redacted before anything is stored, even when they appear in an error message or in web text.
- **Trade-offs:** <latency, cost, reliability>

## Required services
| Requirement | How we use it |
|---|---|
| Nebius Token Factory (inference) | Model: `meta-llama/Llama-3.3-70B-Instruct` by default (`NEBIUS_MODEL`). It plans queries, plans follow-up searches, writes the cited answer and checks each claim against its sources, through the OpenAI-compatible API at `https://api.tokenfactory.nebius.com/v1/` (`NEBIUS_BASE_URL`). |
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
Exit codes: `0` answered, `3` not enough evidence, `1` research failed, `2` missing keys. Add `--trace` to see how the answer was researched, or `--trace-file run.jsonl` to save that trace.

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
