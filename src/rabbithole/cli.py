"""Market RabbitHole CLI entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rabbithole.market_agent import build_market_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Research a market question using local RAG and live web evidence")
    parser.add_argument("question")
    parser.add_argument("--corpus", default=str(Path(__file__).parents[2] / "sample_corpus"))
    args = parser.parse_args()
    result = build_market_agent(args.corpus).research(args.question)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
