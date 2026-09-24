"""Lightweight semantic retrieval for local TXT, Markdown, and PDF sources."""

from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass, field
from pathlib import Path
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


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings."""


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
    """In-memory cosine index over normalized sentence-transformer embeddings."""

    def __init__(self, chunks: list[LocalChunk], encoder: Encoder) -> None:
        self.chunks = chunks
        self.encoder = encoder
        self.embeddings = encoder.encode([chunk.text for chunk in chunks]) if chunks else np.empty((0, 0), dtype=np.float32)

    def search(self, query: str, top_k: int = 3) -> list[LocalSearchResult]:
        if not self.chunks:
            return []
        query_vector = self.encoder.encode([query])[0]
        scores = self.embeddings @ query_vector
        indices = np.argsort(scores)[::-1][:top_k]
        return [LocalSearchResult(self.chunks[int(index)], float(scores[int(index)])) for index in indices if scores[int(index)] > 0]
