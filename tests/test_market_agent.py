import numpy as np

from rabbithole.market_agent import MarketRabbitHoleAgent
from rabbithole.providers import ResilientStructuredLLM, ResilientWebSearchProvider
from rabbithole.semantic_rag import LocalChunk, SemanticRAG, load_uploaded_chunks


class TinyEncoder:
    vocabulary = ["gas", "storage", "supply", "demand"]

    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = np.array([text.lower().count(word) for word in self.vocabulary], dtype=np.float32)
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.asarray(vectors)


class FakeWeb:
    def search(self, query, max_results=2):
        return [{"title": "Live flows", "url": "https://example.com/flows", "source": "example.com", "date": "2026-09-20", "content": "Pipeline supply fell while demand increased."}]


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def complete_json(self, system, user):
        self.calls += 1
        if self.calls == 1:
            return {"angles": [
                {"name": "Supply", "question": "Are flows falling?", "rationale": "Marginal supply matters."},
                {"name": "Demand", "question": "Is demand rising?", "rationale": "Demand tightens balance."},
                {"name": "Storage", "question": "How protective is storage?", "rationale": "Inventory offsets tightness."},
            ]}
        if self.calls == 2:
            import json
            evidence = json.loads(user.split("Evidence: ", 1)[1].split("\nReturn:", 1)[0])
            rows = []
            for index, item in enumerate(evidence):
                rows.append({"id": item["id"], "research_angle": "Supply", "stance": "SUPPORT" if index % 2 == 0 else "CONTRADICT", "key_claim": item["excerpt"][:70], "explanation": "This changes the market balance."})
            return {"evidence": rows}
        return {"current_view": "Tighter supply is outweighing the storage buffer.", "key_drivers": ["↑ Supply disruption", "↓ Storage risk"], "conflicting_signals": ["Inventories remain high."], "evidence_strength": "MEDIUM", "monitor_next": ["Pipeline flows", "Weather forecasts"]}


class FailingProvider:
    def complete_json(self, system, user):
        raise RuntimeError("secret-bearing upstream failure")

    def search(self, query, max_results=1):
        raise RuntimeError("secret-bearing upstream failure")


def test_market_workflow_combines_local_and_web_evidence():
    chunks = [LocalChunk("gas-1", "Gas note", "Gas storage is high but supply is falling.", "gas.md")]
    agent = MarketRabbitHoleAgent(SemanticRAG(chunks, TinyEncoder()), FakeLLM(), FakeWeb(), local_top_k=1, web_results=1)
    result = agent.research("Why are gas prices rising despite high storage?")

    assert len(result.angles) == 3
    assert {item.source_type for item in result.evidence} == {"LOCAL", "WEB"}
    assert result.brief.evidence_strength == "MEDIUM"
    assert result.brief.monitor_next == ["Pipeline flows", "Weather forecasts"]
    assert len(result.trace) == 5
    assert {item.citation_id for item in result.evidence} == {"L1", "W1"}


def test_semantic_rag_returns_most_relevant_chunk():
    chunks = [
        LocalChunk("supply", "Supply", "Gas pipeline supply is falling.", "supply.md"),
        LocalChunk("storage", "Storage", "Gas storage remains high.", "storage.md"),
    ]
    results = SemanticRAG(chunks, TinyEncoder()).search("pipeline supply", top_k=1)
    assert results[0].chunk.id == "supply"


def test_uploaded_text_enters_only_the_research_session():
    base = [LocalChunk("storage", "Storage", "Gas storage remains high.", "storage.md")]
    rag = SemanticRAG(base, TinyEncoder())
    uploaded = load_uploaded_chunks([("private_supply.md", b"Private pipeline supply data shows falling flows.")])
    agent = MarketRabbitHoleAgent(rag, FakeLLM(), FakeWeb(), local_top_k=3, web_results=1)

    result = agent.research("Why is gas supply falling?", uploaded)

    assert any(item.source == "private_supply.md" for item in result.evidence)
    assert all(chunk.path != "private_supply.md" for chunk in rag.chunks)


def test_uploaded_text_chunking_preserves_overlap_and_metadata():
    uploaded = load_uploaded_chunks(
        [("private_note.txt", b"one two three four five six seven")],
        chunk_words=4,
        overlap=2,
    )

    assert [chunk.text for chunk in uploaded] == [
        "one two three four",
        "three four five six",
        "five six seven",
    ]
    assert [chunk.id for chunk in uploaded] == [
        "upload-0-0-0",
        "upload-0-0-1",
        "upload-0-0-2",
    ]
    assert all(chunk.path == "private_note.txt" for chunk in uploaded)
    assert all(chunk.metadata["uploaded"] == "true" for chunk in uploaded)


def test_external_provider_failures_degrade_to_local_brief():
    chunks = [LocalChunk("gas-1", "Gas note", "Gas storage is high but supply is falling.", "gas.md")]
    agent = MarketRabbitHoleAgent(
        SemanticRAG(chunks, TinyEncoder()),
        ResilientStructuredLLM(FailingProvider()),
        ResilientWebSearchProvider(FailingProvider()),
        local_top_k=1,
        web_results=1,
    )

    result = agent.research("Why are gas prices rising despite high storage?")

    assert result.brief.evidence_strength == "LOW"
    assert {item.source_type for item in result.evidence} == {"LOCAL"}
    assert any("Live web research unavailable" in warning for warning in result.warnings)
    assert any("Zhipu unavailable" in warning for warning in result.warnings)
    assert all("secret" not in warning for warning in result.warnings)
