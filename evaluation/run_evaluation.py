"""Small deterministic retrieval evaluation; no external APIs are used."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rabbithole.semantic_rag import LocalChunk, SemanticRAG  # noqa: E402


class EvalEncoder:
    vocabulary = ["copper", "mine", "supply", "inventory", "semiconductor", "valuation", "pipeline", "gas"]

    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = np.asarray([text.lower().count(term) for term in self.vocabulary], dtype=np.float32)
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.asarray(vectors)


DOCUMENTS = [
    LocalChunk("copper-supply", "Copper supply", "Mine disruptions constrain copper supply growth.", "fixture"),
    LocalChunk("copper-inventory", "Copper inventory", "Visible copper inventory provides a near-term buffer.", "fixture"),
    LocalChunk("chip-valuation", "Chip valuation", "Semiconductor valuation assumes rapid earnings growth.", "fixture"),
    LocalChunk("gas-pipeline", "Gas pipeline", "Pipeline outages reduce flexible gas supply.", "fixture"),
]

CASES = [
    ("copper mine supply disruption", "copper-supply"),
    ("near-term copper inventory buffer", "copper-inventory"),
    ("semiconductor valuation expectations", "chip-valuation"),
    ("gas pipeline outage", "gas-pipeline"),
]


def recall_at_one(index: SemanticRAG, keyword_weight: float) -> float:
    hits = sum(index.search(query, top_k=1, keyword_weight=keyword_weight)[0].chunk.id == expected for query, expected in CASES)
    return hits / len(CASES)


def main() -> None:
    index = SemanticRAG(DOCUMENTS, EvalEncoder())
    result = {
        "dataset": "deterministic synthetic retrieval fixture",
        "case_count": len(CASES),
        "metric": "Recall@1",
        "semantic_only": recall_at_one(index, 0.0),
        "hybrid_20_percent_keyword": recall_at_one(index, 0.2),
        "limitations": "This tiny fixture verifies behavior; it is not evidence of production-quality improvement.",
    }
    output = ROOT / "evaluation" / "results.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
