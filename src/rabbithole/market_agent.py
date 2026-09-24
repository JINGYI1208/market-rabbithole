"""Sequential Market RabbitHole research workflow."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rabbithole.providers import (
    ResilientStructuredLLM,
    ResilientWebSearchProvider,
    StructuredLLM,
    TavilySearchProvider,
    WebSearchProvider,
    ZhipuStructuredLLM,
)
from rabbithole.semantic_rag import LocalChunk, SemanticRAG, SentenceTransformerEncoder, load_local_chunks


@dataclass(slots=True)
class ResearchAngle:
    name: str
    question: str
    rationale: str


@dataclass(slots=True)
class MarketEvidence:
    id: str
    source_type: str
    title: str
    source: str
    locator: str
    date: str
    excerpt: str
    research_angle: str
    citation_id: str = ""
    stance: str = "NEUTRAL"
    key_claim: str = ""
    explanation: str = ""
    relevance: float | None = None


@dataclass(slots=True)
class MarketConflict:
    angle: str
    support: list[str]
    contradict: list[str]


@dataclass(slots=True)
class MarketBrief:
    current_view: str
    key_drivers: list[str]
    conflicting_signals: list[str]
    evidence_strength: str
    monitor_next: list[str]


@dataclass(slots=True)
class MarketTraceStep:
    step: str
    detail: str


@dataclass(slots=True)
class MarketResearch:
    question: str
    angles: list[ResearchAngle]
    evidence: list[MarketEvidence]
    conflicts: list[MarketConflict]
    brief: MarketBrief
    trace: list[MarketTraceStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MarketRabbitHoleAgent:
    def __init__(self, rag: SemanticRAG, llm: StructuredLLM, web: WebSearchProvider, local_top_k: int = 2, web_results: int = 1) -> None:
        self.rag = rag
        self.llm = llm
        self.web = web
        self.local_top_k = local_top_k
        self.web_results = web_results

    def research(self, question: str, uploaded_chunks: list[LocalChunk] | None = None) -> MarketResearch:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        trace: list[MarketTraceStep] = []

        angles = self._plan(question)
        trace.append(MarketTraceStep("Research plan", f"Generated {len(angles)} question-specific angles"))

        session_rag = self.rag
        if uploaded_chunks:
            session_rag = SemanticRAG([*self.rag.chunks, *uploaded_chunks], self.rag.encoder)
        evidence = self._collect(question, angles, session_rag)
        self._assign_citations(evidence)
        local_count = sum(item.source_type == "LOCAL" for item in evidence)
        trace.append(MarketTraceStep("Evidence search", f"Collected {local_count} local and {len(evidence) - local_count} live web items"))

        self._analyze(question, angles, evidence)
        trace.append(MarketTraceStep("Evidence analysis", f"Classified {len(evidence)} evidence items with Zhipu"))

        conflicts = self._detect_conflicts(evidence)
        trace.append(MarketTraceStep("Conflict radar", f"Detected {len(conflicts)} angles with competing signals"))

        brief = self._synthesize(question, angles, evidence, conflicts)
        trace.append(MarketTraceStep("Market brief", f"Synthesized a {brief.evidence_strength} evidence-strength view"))
        warnings = self._provider_warnings()
        return MarketResearch(question, angles, evidence, conflicts, brief, trace, warnings)

    def _plan(self, question: str) -> list[ResearchAngle]:
        payload = self.llm.complete_json(
            "You are a market research planner. Return JSON only. Create 3-4 distinct, question-specific research angles. Do not answer the question.",
            f'''Question: {question}\nReturn: {{"angles":[{{"name":"short label","question":"targeted research question","rationale":"why this matters"}}]}}''',
        )
        angles = [ResearchAngle(str(item["name"]), str(item["question"]), str(item.get("rationale", ""))) for item in payload.get("angles", [])[:4]]
        if len(angles) < 3:
            raise ValueError("Zhipu research plan must contain at least three angles")
        return angles

    def _collect(self, question: str, angles: list[ResearchAngle], rag: SemanticRAG) -> list[MarketEvidence]:
        evidence: list[MarketEvidence] = []
        seen: set[str] = set()
        for angle in angles:
            query = f"{question} {angle.question}"
            for result in rag.search(query, self.local_top_k):
                key = f"local:{result.chunk.id}"
                if key in seen:
                    continue
                seen.add(key)
                section = result.chunk.metadata.get("section", "")
                evidence.append(MarketEvidence(key, "LOCAL", result.chunk.title, Path(result.chunk.path).name, result.chunk.path, section, result.chunk.text, angle.name, relevance=round(result.score, 3)))
            for index, item in enumerate(self.web.search(query, self.web_results)):
                key = f"web:{item.get('url') or angle.name + str(index)}"
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(MarketEvidence(key, "WEB", item.get("title", "Untitled source"), item.get("source", "Web source"), item.get("url", ""), item.get("date", ""), item.get("content", ""), angle.name))
        return evidence

    def _analyze(self, question: str, angles: list[ResearchAngle], evidence: list[MarketEvidence]) -> None:
        compact = [{"id": item.id, "angle": item.research_angle, "title": item.title, "excerpt": item.excerpt[:1200]} for item in evidence]
        payload = self.llm.complete_json(
            "You analyze market evidence. Classify how each item bears on the market question. Return JSON only. Stance must be SUPPORT, CONTRADICT, or NEUTRAL. Never invent evidence.",
            f'''Question: {question}\nResearch angles: {json.dumps([asdict(angle) for angle in angles], ensure_ascii=False)}\nEvidence: {json.dumps(compact, ensure_ascii=False)}\nReturn: {{"evidence":[{{"id":"exact id","research_angle":"angle","stance":"SUPPORT|CONTRADICT|NEUTRAL","key_claim":"one grounded sentence","explanation":"short relation to question"}}]}}''',
        )
        by_id = {item.id: item for item in evidence}
        for analysis in payload.get("evidence", []):
            item = by_id.get(str(analysis.get("id")))
            if not item:
                continue
            stance = str(analysis.get("stance", "NEUTRAL")).upper()
            item.stance = stance if stance in {"SUPPORT", "CONTRADICT", "NEUTRAL"} else "NEUTRAL"
            item.research_angle = str(analysis.get("research_angle") or item.research_angle)
            item.key_claim = str(analysis.get("key_claim", ""))
            item.explanation = str(analysis.get("explanation", ""))

    @staticmethod
    def _detect_conflicts(evidence: list[MarketEvidence]) -> list[MarketConflict]:
        conflicts: list[MarketConflict] = []
        for angle in dict.fromkeys(item.research_angle for item in evidence):
            support = [item.key_claim for item in evidence if item.research_angle == angle and item.stance == "SUPPORT" and item.key_claim]
            contradict = [item.key_claim for item in evidence if item.research_angle == angle and item.stance == "CONTRADICT" and item.key_claim]
            if support and contradict:
                conflicts.append(MarketConflict(angle, support[:2], contradict[:2]))
        return conflicts

    @staticmethod
    def _assign_citations(evidence: list[MarketEvidence]) -> None:
        counts = {"LOCAL": 0, "WEB": 0}
        prefixes = {"LOCAL": "L", "WEB": "W"}
        for item in evidence:
            counts[item.source_type] += 1
            item.citation_id = f"{prefixes[item.source_type]}{counts[item.source_type]}"

    def _provider_warnings(self) -> list[str]:
        warnings: list[str] = []
        for provider in (self.web, self.llm):
            drain = getattr(provider, "drain_warnings", None)
            if drain:
                warnings.extend(drain())
        return warnings

    def _synthesize(self, question: str, angles: list[ResearchAngle], evidence: list[MarketEvidence], conflicts: list[MarketConflict]) -> MarketBrief:
        analyzed = [{"id": item.id, "citation": item.citation_id, "angle": item.research_angle, "stance": item.stance, "claim": item.key_claim, "source": item.title, "date": item.date} for item in evidence]
        payload = self.llm.complete_json(
            "You write a concise market intelligence brief using only supplied evidence. Return JSON only. Cite every substantive conclusion with the supplied citation IDs in square brackets, such as [L1] or [W2]. Evidence strength must be LOW, MEDIUM, or HIGH; consider amount, source diversity, recency, and conflicts without inventing precision.",
            f'''Question: {question}\nAngles: {json.dumps([asdict(angle) for angle in angles], ensure_ascii=False)}\nEvidence: {json.dumps(analyzed, ensure_ascii=False)}\nDetected conflicts: {json.dumps([asdict(conflict) for conflict in conflicts], ensure_ascii=False)}\nReturn: {{"current_view":"2-4 cited sentences","key_drivers":["3-5 cited directional findings using ↑/↓/↔"],"conflicting_signals":["cited risks or caveats"],"evidence_strength":"LOW|MEDIUM|HIGH","monitor_next":["2-3 concrete indicators or events"]}}''',
        )
        strength = str(payload.get("evidence_strength", "LOW")).upper()
        return MarketBrief(
            current_view=str(payload.get("current_view", "Insufficient evidence to form a market view.")),
            key_drivers=[str(item) for item in payload.get("key_drivers", [])][:5],
            conflicting_signals=[str(item) for item in payload.get("conflicting_signals", [])][:4],
            evidence_strength=strength if strength in {"LOW", "MEDIUM", "HIGH"} else "LOW",
            monitor_next=[str(item) for item in payload.get("monitor_next", [])][:3],
        )


def build_market_agent(corpus: str | Path) -> MarketRabbitHoleAgent:
    from dotenv import load_dotenv

    load_dotenv()
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    rag = SemanticRAG(load_local_chunks(corpus), SentenceTransformerEncoder(model_name))
    zhipu_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    llm = ResilientStructuredLLM(ZhipuStructuredLLM(api_key=zhipu_key) if zhipu_key else None)
    web = ResilientWebSearchProvider(TavilySearchProvider(api_key=tavily_key) if tavily_key else None)
    return MarketRabbitHoleAgent(rag, llm, web)
