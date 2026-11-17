"""Command-line entry point: python src/agent.py "Should we move from X to Y?" """
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from research_agent.config import ConfigError, build_clients, secret_values
from research_agent.pipeline import Status, render, research
from research_agent.trace import Trace

EXIT_CODES = {Status.ANSWERED: 0, Status.FAILED: 1, Status.INSUFFICIENT_EVIDENCE: 3}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research a tooling question and answer with cited sources.")
    parser.add_argument("question", nargs="+", help="the research question")
    parser.add_argument("--trace", action="store_true", help="print the full research trace to stderr")
    parser.add_argument("--trace-file", type=Path, metavar="PATH", help="write the research trace as JSON lines")
    args = parser.parse_args(argv)

    load_dotenv()
    try:
        llm, search = build_clients(os.environ)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    trace = Trace(secrets=secret_values(os.environ))
    result = research(" ".join(args.question), llm, search, trace=trace)
    print(render(result))
    if args.trace:
        print("\nResearch trace:\n" + trace.render(), file=sys.stderr)
    else:
        for warning in trace.warnings():
            print(f"warning: {warning}", file=sys.stderr)
    if args.trace_file:
        args.trace_file.write_text(trace.to_jsonl(), encoding="utf-8")
    return EXIT_CODES[result.status]


if __name__ == "__main__":
    sys.exit(main())
