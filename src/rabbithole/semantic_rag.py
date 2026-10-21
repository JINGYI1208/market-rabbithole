"""Local retrieval for TXT, Markdown, and PDF sources."""

from __future__ import annotations

from io import BytesIO
from collections import Counter
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class LocalChunk:
    id: str
    title: str
    text: str
    path: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LocalSearchResult:
    chunk: LocalChunk
    score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings."""


class LocalRetriever(Protocol):
    chunks: list[LocalChunk]
    backend_name: str

    def search(self, query: str, top_k: int = 3, keyword_weight: float = 0.2) -> list[LocalSearchResult]:
        """Return ranked local chunks."""

    def with_chunks(self, chunks: list[LocalChunk]) -> "LocalRetriever":
        """Create an isolated request index over the supplied chunks."""


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install project dependencies before starting Market RabbitHole") from exc
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def load_local_chunks(root: str | Path, chunk_words: int = 140, overlap: int = 30) -> list[LocalChunk]:
    root = Path(root)
    chunks: list[LocalChunk] = []
    for path in sorted(root.iterdir() if root.exists() else []):
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        for section_index, (text, section) in enumerate(_read_sections(path)):
            chunks.extend(_overlapping_chunks(
                text,
                id_prefix=f"{path.stem}-{section_index}",
                title=path.stem.replace("_", " ").title(),
                path=str(path),
                metadata={"section": section},
                chunk_words=chunk_words,
                overlap=overlap,
            ))
    return chunks


def load_uploaded_chunks(files: list[tuple[str, bytes]], chunk_words: int = 140, overlap: int = 30) -> list[LocalChunk]:
    """Parse request-scoped uploads without writing them to persistent storage."""
    chunks: list[LocalChunk] = []
    for file_index, (filename, content) in enumerate(files):
        suffix = Path(filename).suffix.lower()
        if suffix not in {".txt", ".md", ".pdf"}:
            continue
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("pypdf is required for PDF ingestion") from exc
            sections = []
            for page_index, page in enumerate(PdfReader(BytesIO(content)).pages, 1):
                text = (page.extract_text() or "").strip()
                if text:
                    sections.append((text, f"page {page_index}"))
        else:
            sections = [(content.decode("utf-8", errors="replace").strip(), "document")]

        for section_index, (text, section) in enumerate(sections):
            chunks.extend(_overlapping_chunks(
                text,
                id_prefix=f"upload-{file_index}-{section_index}",
                title=Path(filename).stem.replace("_", " ").title(),
                path=filename,
                metadata={"section": section, "uploaded": "true"},
                chunk_words=chunk_words,
                overlap=overlap,
            ))
    return chunks


def _overlapping_chunks(
    text: str,
    *,
    id_prefix: str,
    title: str,
    path: str,
    metadata: dict[str, str],
    chunk_words: int,
    overlap: int,
) -> list[LocalChunk]:
    """Build bounded word buffers while carrying context into the next chunk."""
    if chunk_words < 1:
        raise ValueError("chunk_words must be positive")
    if overlap < 0 or overlap >= chunk_words:
        raise ValueError("overlap must be between zero and chunk_words - 1")

    completed: list[LocalChunk] = []
    buffer: list[str] = []
    words_added_after_emit = 0

    def emit() -> None:
        index = len(completed)
        completed.append(LocalChunk(
            id=f"{id_prefix}-{index}",
            title=title,
            text=" ".join(buffer),
            path=path,
            metadata={**metadata, "chunk": str(index)},
        ))

    for word in text.split():
        buffer.append(word)
        words_added_after_emit += 1
        if len(buffer) == chunk_words:
            emit()
            buffer = buffer[-overlap:] if overlap else []
            words_added_after_emit = 0

    if buffer and words_added_after_emit:
        emit()
    return completed


def _read_sections(path: Path) -> list[tuple[str, str]]:
    if path.suffix.lower() in {".txt", ".md"}:
        return [(path.read_text(encoding="utf-8").strip(), "document")]
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for PDF ingestion") from exc
    sections: list[tuple[str, str]] = []
    for index, page in enumerate(PdfReader(str(path)).pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append((text, f"page {index}"))
    return sections


class SemanticRAG:
    """In-memory hybrid index over dense embeddings and lexical overlap."""

    def __init__(self, chunks: list[LocalChunk], encoder: Encoder) -> None:
        self.chunks = chunks
        self.encoder = encoder
        self.backend_name = "sentence-transformers hybrid"
        self.embeddings = encoder.encode([chunk.text for chunk in chunks]) if chunks else np.empty((0, 0), dtype=np.float32)

    def with_chunks(self, chunks: list[LocalChunk]) -> "SemanticRAG":
        return SemanticRAG(chunks, self.encoder)

    def search(self, query: str, top_k: int = 3, keyword_weight: float = 0.2) -> list[LocalSearchResult]:
        if not self.chunks:
            return []
        if not 0.0 <= keyword_weight <= 1.0:
            raise ValueError("keyword_weight must be between 0 and 1")
        query_vector = self.encoder.encode([query])[0]
        semantic_scores = self.embeddings @ query_vector
        query_terms = _terms(query)
        keyword_scores = np.asarray([
            len(query_terms & _terms(chunk.text)) / max(1, len(query_terms))
            for chunk in self.chunks
        ], dtype=np.float32)
        scores = (1.0 - keyword_weight) * semantic_scores + keyword_weight * keyword_scores
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            LocalSearchResult(
                self.chunks[int(index)],
                float(scores[int(index)]),
                float(semantic_scores[int(index)]),
                float(keyword_scores[int(index)]),
            )
            for index in indices
            if scores[int(index)] > 0
        ]


class TfidfRetriever:
    """Sparse lexical TF-IDF retrieval without Transformers or PyTorch."""

    backend_name = "TF-IDF lexical"

    def __init__(self, chunks: list[LocalChunk]) -> None:
        self.chunks = chunks
        term_counts = [Counter(_term_list(chunk.text)) for chunk in chunks]
        document_frequency = Counter(term for counts in term_counts for term in counts)
        document_count = len(chunks)
        self.idf = {
            term: math.log((1 + document_count) / (1 + frequency)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self.vectors = [self._vector(counts) for counts in term_counts]

    def with_chunks(self, chunks: list[LocalChunk]) -> "TfidfRetriever":
        return TfidfRetriever(chunks)

    def search(self, query: str, top_k: int = 3, keyword_weight: float = 0.2) -> list[LocalSearchResult]:
        if not self.chunks or top_k < 1:
            return []
        query_vector = self._vector(Counter(_term_list(query)))
        if not query_vector:
            return []
        scores = [sum(weight * document.get(term, 0.0) for term, weight in query_vector.items()) for document in self.vectors]
        indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
        return [
            LocalSearchResult(self.chunks[index], scores[index], semantic_score=0.0, keyword_score=scores[index])
            for index in indices
            if scores[index] > 0
        ]

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        weighted = {
            term: (1.0 + math.log(count)) * self.idf[term]
            for term, count in counts.items()
            if term in self.idf and count > 0
        }
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        return {term: value / norm for term, value in weighted.items()} if norm else {}


def resolve_retrieval_backend(value: str | None = None) -> str:
    """Resolve an explicit backend or choose the low-memory backend on Render."""
    selected = (value or os.getenv("RABBITHOLE_RETRIEVAL_BACKEND", "auto")).strip().lower()
    aliases = {"semantic": "sentence-transformers", "dense": "sentence-transformers", "lexical": "tfidf"}
    selected = aliases.get(selected, selected)
    if selected == "auto":
        render_hosted = os.getenv("RENDER", "").strip().lower() in {"1", "true", "yes", "on"}
        return "tfidf" if render_hosted else "sentence-transformers"
    if selected not in {"tfidf", "sentence-transformers"}:
        raise ValueError("RABBITHOLE_RETRIEVAL_BACKEND must be auto, tfidf, or sentence-transformers")
    return selected


def _terms(text: str) -> set[str]:
    return set(_term_list(text))


def _term_list(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", text.lower()) if len(term) > 1]
