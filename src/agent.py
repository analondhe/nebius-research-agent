"""Command-line entry point: python src/agent.py "Should we move from X to Y?" """
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from research_agent.config import ConfigError, build_clients
from research_agent.pipeline import Status, render, research

EXIT_CODES = {Status.ANSWERED: 0, Status.FAILED: 1, Status.INSUFFICIENT_EVIDENCE: 3}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research a tooling question and answer with cited sources.")
    parser.add_argument("question", nargs="+", help="the research question")
    args = parser.parse_args(argv)

    load_dotenv()
    try:
        llm, search = build_clients(os.environ)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = research(" ".join(args.question), llm, search)
    print(render(result))
    for note in result.notes:
        print(f"note: {note}", file=sys.stderr)
    return EXIT_CODES[result.status]


if __name__ == "__main__":
    sys.exit(main())
