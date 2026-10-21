import json

import numpy as np
import pytest

from rabbithole.market_agent import AgentLimits, MarketRabbitHoleAgent, build_market_agent
from rabbithole.providers import ResilientStructuredLLM, ResilientWebSearchProvider
from rabbithole.semantic_rag import LocalChunk, SemanticRAG, TfidfRetriever, load_uploaded_chunks, resolve_retrieval_backend


class TinyEncoder:
    vocabulary = ["gas", "storage", "supply", "demand", "pipeline"]

    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = np.array([text.lower().count(word) for word in self.vocabulary], dtype=np.float32)
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.asarray(vectors)


class FakeWeb:
    def __init__(self):
        self.calls = []

    def search(self, query, max_results=2):
        self.calls.append(query)
        return [{"title": "Live flows", "url": "https://example.com/flows", "source": "example.com", "date": "2026-09-20", "content": "Pipeline supply fell while demand increased."}]


class AgenticFakeLLM:
    def complete_json(self, system, user):
        lowered = system.lower()
        if "initial plan" in lowered:
            return {"angles": [
                {"name": "Supply", "question": "Are flows falling?", "rationale": "Marginal supply matters."},
                {"name": "Demand", "question": "Is demand rising?", "rationale": "Demand tightens balance."},
                {"name": "Storage", "question": "How protective is storage?", "rationale": "Inventory offsets tightness."},
            ]}
        if "bounded market research loop" in lowered:
            state = json.loads(user.split("Research state: ", 1)[1].split("\nReturn:", 1)[0])
            evidence = state["evidence"]
            if not evidence:
                return {"action": "search_local", "query": "pipeline supply", "angle": "Supply", "rationale": "Start with private context.", "evidence_gap": "No private evidence."}
            if any(not item["claim"] for item in evidence):
                return {"action": "analyze_evidence", "query": "", "angle": "", "rationale": "Classify new evidence.", "evidence_gap": ""}
            if not any(item["source_type"] == "WEB" for item in evidence):
                return {"action": "search_web", "query": "latest gas pipeline supply", "angle": "Supply", "rationale": "Check current flows.", "evidence_gap": "Private evidence may be stale."}
            return {"action": "stop", "query": "", "angle": "", "rationale": "Evidence is sufficient and analyzed.", "evidence_gap": ""}
        if "analyze market evidence" in lowered:
            evidence = json.loads(user.split("Evidence: ", 1)[1].split("\nReturn:", 1)[0])
            return {"evidence": [{
                "id": item["id"],
                "research_angle": "Supply",
                "stance": "SUPPORT" if index % 2 == 0 else "CONTRADICT",
                "kind": "FACT" if item.get("date") else "OPINION",
                "key_claim": item["excerpt"][:70],
                "explanation": "This changes the market balance.",
                "time_period": item.get("date") or "current",
                "assumptions": ["Other drivers unchanged"],
            } for index, item in enumerate(evidence)]}
        evidence = json.loads(user.split("Evidence: ", 1)[1].split("\nDetected conflicts:", 1)[0])
        citations = [item["citation"] for item in evidence]
        return {
            "current_view": f"Supply is tighter [{citations[0]}].",
            "conclusions": [{"statement": "Supply is tighter.", "citations": citations[:2], "uncertainty": "Storage remains high."}],
            "key_drivers": [f"↑ Supply disruption [{citations[0]}]"],
            "conflicting_signals": ["Inventories remain high."],
            "evidence_strength": "MEDIUM",
            "monitor_next": ["Pipeline flows", "Weather forecasts"],
        }


class FailingProvider:
    def complete_json(self, system, user):
        raise RuntimeError("secret-bearing upstream failure")

    def search(self, query, max_results=1):
        raise RuntimeError("secret-bearing upstream failure")


def build_agent(llm=None, web=None, chunks=None, **limit_overrides):
    chunks = chunks or [LocalChunk("gas-1", "Gas note", "Gas storage is high but pipeline supply is falling.", "gas.md")]
    values = {"max_iterations": 8, "max_local_searches": 3, "max_web_searches": 3, "local_top_k": 1, "web_results": 1, "min_evidence": 2}
    values.update(limit_overrides)
    limits = AgentLimits(**values)
    return MarketRabbitHoleAgent(SemanticRAG(chunks, TinyEncoder()), llm or AgenticFakeLLM(), web or FakeWeb(), limits=limits)


def test_agent_loop_chooses_tools_from_state_and_stops():
    web = FakeWeb()
    result = build_agent(web=web).research("Why are gas prices rising despite high storage?")

    actions = [event.metadata.get("action") for event in result.trace if event.event_type == "agent_decision"]
    assert actions == ["search_local", "analyze_evidence", "search_web", "analyze_evidence", "stop"]
    assert result.stop_reason == "Evidence is sufficient and analyzed."
    assert {item.source_type for item in result.evidence} == {"LOCAL", "WEB"}
    assert len(result.conflicts) == 1
    assert result.brief.conclusions[0].citation_verified is True
    assert any(event.event_type == "citation_verification_completed" for event in result.trace)
    assert web.calls == ["latest gas pipeline supply"]


def test_agent_rejects_invalid_structured_action():
    class InvalidLLM(AgenticFakeLLM):
        def complete_json(self, system, user):
            if "bounded market research loop" in system.lower():
                return {"action": "run_shell", "rationale": "unsafe"}
            return super().complete_json(system, user)

    with pytest.raises(ValueError, match="Unsupported agent action"):
        build_agent(llm=InvalidLLM()).research("What drives gas prices?")


def test_premature_stop_is_rejected_until_iteration_bound():
    class PrematureStopLLM(AgenticFakeLLM):
        def complete_json(self, system, user):
            if "bounded market research loop" in system.lower():
                return {"action": "stop", "rationale": "Stop without evidence."}
            if "write a concise market intelligence brief" in system.lower():
                return {"current_view": "Insufficient evidence.", "conclusions": [], "key_drivers": [], "conflicting_signals": ["No evidence."], "evidence_strength": "LOW", "monitor_next": ["Collect evidence"]}
            return super().complete_json(system, user)

    result = build_agent(llm=PrematureStopLLM(), max_iterations=3).research("What drives gas prices?")
    assert result.iterations == 3
    assert result.stop_reason == "iteration limit reached"
    assert len([event for event in result.trace if event.event_type == "stop_rejected"]) == 3
    assert result.evidence == []
    assert "Research stopped at the iteration limit; unresolved evidence gaps may remain." in result.brief.conflicting_signals


def test_semantic_rag_hybrid_search_uses_keyword_signal():
    chunks = [
        LocalChunk("supply", "Supply", "Gas pipeline supply is falling.", "supply.md"),
        LocalChunk("storage", "Storage", "Gas storage remains high.", "storage.md"),
    ]
    results = SemanticRAG(chunks, TinyEncoder()).search("pipeline supply", top_k=1, keyword_weight=0.3)
    assert results[0].chunk.id == "supply"
    assert results[0].keyword_score > 0


def test_tfidf_retriever_is_lexical_and_handles_empty_corpus():
    chunks = [
        LocalChunk("copper", "Copper", "Copper mine supply disruptions tighten concentrate markets.", "copper.md"),
        LocalChunk("rates", "Rates", "Treasury yields respond to monetary policy.", "rates.md"),
    ]
    retriever = TfidfRetriever(chunks)

    assert retriever.backend_name == "TF-IDF lexical"
    assert retriever.search("copper supply", top_k=1)[0].chunk.id == "copper"
    assert retriever.search("unseen-vocabulary") == []
    assert TfidfRetriever([]).search("anything") == []


def test_retrieval_backend_auto_uses_tfidf_on_render(monkeypatch):
    monkeypatch.delenv("RABBITHOLE_RETRIEVAL_BACKEND", raising=False)
    monkeypatch.setenv("RENDER", "true")
    assert resolve_retrieval_backend() == "tfidf"


def test_build_agent_tfidf_mode_does_not_initialize_transformer(monkeypatch, tmp_path):
    (tmp_path / "note.md").write_text("Copper supply is constrained.", encoding="utf-8")
    monkeypatch.setenv("RABBITHOLE_RETRIEVAL_BACKEND", "tfidf")
    monkeypatch.setenv("ZHIPUAI_API_KEY", "")
    monkeypatch.setenv("ZHIPU_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("Sentence Transformers must not load in TF-IDF mode")

    monkeypatch.setattr("rabbithole.market_agent.SentenceTransformerEncoder", fail_if_loaded)
    agent = build_market_agent(tmp_path)

    assert isinstance(agent.rag, TfidfRetriever)


def test_uploaded_text_enters_only_the_research_session():
    base = [LocalChunk("storage", "Storage", "Gas storage remains high.", "storage.md")]
    uploaded = load_uploaded_chunks([("private_supply.md", b"Private pipeline supply data shows falling flows.")])
    agent = build_agent(chunks=base)
    result = agent.research("Why is gas supply falling?", uploaded)

    assert any(item.source == "private_supply.md" for item in result.evidence)
    assert all(chunk.path != "private_supply.md" for chunk in agent.rag.chunks)


def test_uploaded_text_is_retrievable_in_tfidf_agent_loop():
    base = [LocalChunk("storage", "Storage", "Gas storage remains high.", "storage.md")]
    uploaded = load_uploaded_chunks([("private_supply.txt", b"Private pipeline supply data shows falling flows.")])
    limits = AgentLimits(max_iterations=8, max_local_searches=3, max_web_searches=3, local_top_k=1, web_results=1, min_evidence=2)
    agent = MarketRabbitHoleAgent(TfidfRetriever(base), AgenticFakeLLM(), FakeWeb(), limits=limits)

    result = agent.research("Why is pipeline supply falling?", uploaded)

    assert any(item.source == "private_supply.txt" for item in result.evidence)
    assert all(chunk.path != "private_supply.txt" for chunk in agent.rag.chunks)


def test_uploaded_text_chunking_preserves_overlap_and_metadata():
    uploaded = load_uploaded_chunks([("private_note.txt", b"one two three four five six seven")], chunk_words=4, overlap=2)
    assert [chunk.text for chunk in uploaded] == ["one two three four", "three four five six", "five six seven"]
    assert [chunk.id for chunk in uploaded] == ["upload-0-0-0", "upload-0-0-1", "upload-0-0-2"]
    assert all(chunk.path == "private_note.txt" for chunk in uploaded)
    assert all(chunk.metadata["uploaded"] == "true" for chunk in uploaded)


def test_external_provider_failures_are_bounded_and_understandable():
    agent = build_agent(llm=ResilientStructuredLLM(FailingProvider()), web=ResilientWebSearchProvider(FailingProvider()))
    result = agent.research("Why are gas prices rising despite high storage?")

    assert result.iterations <= agent.limits.max_iterations
    assert result.brief.evidence_strength == "LOW"
    assert {item.source_type for item in result.evidence} == {"LOCAL"}
    assert any("Live web research unavailable" in warning for warning in result.warnings)
    assert any("Zhipu unavailable" in warning for warning in result.warnings)
    assert any("research planning (RuntimeError)" in warning for warning in result.warnings)
    assert all("secret" not in warning for warning in result.warnings)


def test_invalid_citation_is_flagged_as_unsupported():
    class BadCitationLLM(AgenticFakeLLM):
        def complete_json(self, system, user):
            if "write a concise market intelligence brief" in system.lower():
                return {"current_view": "Unsupported claim [Z9].", "conclusions": [{"statement": "Unsupported claim.", "citations": ["Z9"], "uncertainty": "Unknown"}], "key_drivers": [], "conflicting_signals": [], "evidence_strength": "LOW", "monitor_next": []}
            return super().complete_json(system, user)

    result = build_agent(llm=BadCitationLLM()).research("What drives gas prices?")
    assert result.brief.conclusions[0].citation_verified is False
    assert result.brief.unsupported_claims == ["Unsupported claim."]
