"""Bounded single-agent market research workflow."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from rabbithole.providers import (
    ResilientStructuredLLM,
    ResilientWebSearchProvider,
    StructuredLLM,
    TavilySearchProvider,
    WebSearchProvider,
    ZhipuStructuredLLM,
)
from rabbithole.semantic_rag import (
    LocalChunk,
    LocalRetriever,
    SemanticRAG,
    SentenceTransformerEncoder,
    TfidfRetriever,
    load_local_chunks,
    resolve_retrieval_backend,
)


@dataclass(slots=True)
class AgentLimits:
    max_iterations: int = 8
    max_local_searches: int = 4
    max_web_searches: int = 4
    local_top_k: int = 2
    web_results: int = 1
    min_evidence: int = 3
    keyword_weight: float = 0.2

    def __post_init__(self) -> None:
        positive = ("max_iterations", "max_local_searches", "max_web_searches", "local_top_k", "web_results", "min_evidence")
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.keyword_weight <= 1.0:
            raise ValueError("keyword_weight must be between 0 and 1")


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
    evidence_kind: str = "UNCLASSIFIED"
    time_period: str = ""
    assumptions: list[str] = field(default_factory=list)
    relevance: float | None = None


@dataclass(slots=True)
class MarketConflict:
    angle: str
    support: list[str]
    contradict: list[str]
    support_citations: list[str] = field(default_factory=list)
    contradict_citations: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass(slots=True)
class MarketConclusion:
    statement: str
    citations: list[str]
    uncertainty: str = ""
    citation_verified: bool = False


@dataclass(slots=True)
class MarketBrief:
    current_view: str
    key_drivers: list[str]
    conflicting_signals: list[str]
    evidence_strength: str
    monitor_next: list[str]
    conclusions: list[MarketConclusion] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResearchEvent:
    event_type: str
    detail: str
    iteration: int = 0
    tool: str = ""
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step(self) -> str:
        return self.event_type.replace("_", " ").title()


@dataclass(slots=True)
class MarketResearch:
    question: str
    angles: list[ResearchAngle]
    evidence: list[MarketEvidence]
    conflicts: list[MarketConflict]
    brief: MarketBrief
    trace: list[ResearchEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stop_reason: str = ""
    iterations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentDecision:
    action: str
    rationale: str
    query: str = ""
    angle: str = ""
    evidence_gap: str = ""
    source: str = "llm"
    rejection_reasons: list[str] = field(default_factory=list)
    incomplete_stop: bool = False


EventCallback = Callable[[ResearchEvent], None]


class MarketRabbitHoleAgent:
    """One LLM-directed agent with validated tools and explicit stop bounds."""

    ACTIONS = {"search_local", "search_web", "analyze_evidence", "stop"}

    def __init__(
        self,
        rag: LocalRetriever,
        llm: StructuredLLM,
        web: WebSearchProvider,
        local_top_k: int | None = None,
        web_results: int | None = None,
        *,
        limits: AgentLimits | None = None,
    ) -> None:
        self.rag = rag
        self.llm = llm
        self.web = web
        self.limits = limits or AgentLimits(local_top_k=local_top_k or 2, web_results=web_results or 1)

    def research(
        self,
        question: str,
        uploaded_chunks: list[LocalChunk] | None = None,
        on_event: EventCallback | None = None,
    ) -> MarketResearch:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        events: list[ResearchEvent] = []

        def record(event_type: str, detail: str, iteration: int = 0, tool: str = "", status: str = "ok", **metadata: Any) -> None:
            event = ResearchEvent(event_type, detail, iteration, tool, status, metadata)
            events.append(event)
            if on_event:
                on_event(event)

        angles = self._plan(question)
        record("initial_research_plan", f"Created {len(angles)} question-specific research angles", angle_count=len(angles))
        session_rag = self.rag
        if uploaded_chunks:
            session_rag = self.rag.with_chunks([*self.rag.chunks, *uploaded_chunks])

        evidence: list[MarketEvidence] = []
        seen: set[str] = set()
        budgets = {"search_local": self.limits.max_local_searches, "search_web": self.limits.max_web_searches}
        stop_reason = "iteration limit reached"
        iterations = 0

        for iteration in range(1, self.limits.max_iterations + 1):
            iterations = iteration
            conflicts = self._detect_conflicts(evidence)
            decision = self._decide(question, angles, evidence, conflicts, budgets, iteration)
            for reason in decision.rejection_reasons:
                record("decision_rejected", reason, iteration, "decision_validator", "rejected")
            if decision.source == "deterministic_recovery":
                record("decision_recovered", decision.rationale, iteration, decision.action, source=decision.source)
            elif decision.source == "llm_retry":
                record("decision_retry_accepted", "Corrected model decision passed validation", iteration, decision.action)
            decision_event = "agent_decision" if decision.source.startswith("llm") else "fallback_decision"
            record(decision_event, decision.rationale, iteration, decision.action, action=decision.action, angle=decision.angle, source=decision.source)
            if decision.evidence_gap:
                record("evidence_gap_detected", decision.evidence_gap, iteration, decision.action, angle=decision.angle)

            if decision.action == "stop":
                if decision.incomplete_stop:
                    stop_reason = decision.rationale
                    record("research_stopped", stop_reason, iteration, "deterministic_recovery", incomplete=True)
                    break
                allowed, reason = self._can_stop(evidence)
                if allowed:
                    stop_reason = decision.rationale or "agent judged the evidence sufficient"
                    record("research_stopped", stop_reason, iteration, "stop")
                    break
                record("stop_rejected", reason, iteration, "stop", "rejected")
                continue

            if decision.action in budgets:
                if budgets[decision.action] <= 0:
                    record("tool_budget_exhausted", f"No {decision.action} budget remained", iteration, decision.action, "rejected")
                    continue
                budgets[decision.action] -= 1

            if decision.action == "search_local":
                record("tool_selected", "Agent selected private/local retrieval", iteration, "search_local", angle=decision.angle)
                results = session_rag.search(decision.query, self.limits.local_top_k, self.limits.keyword_weight)
                added = self._add_local_evidence(results, decision.angle, evidence, seen)
                self._assign_citations(evidence)
                record("search_performed", f"Local retrieval added {added} new evidence items", iteration, "search_local", result_count=added)
            elif decision.action == "search_web":
                record("tool_selected", "Agent selected live web research", iteration, "search_web", angle=decision.angle)
                results = self.web.search(decision.query, self.limits.web_results)
                added = self._add_web_evidence(results, decision.angle, evidence, seen)
                self._assign_citations(evidence)
                record("search_performed", f"Web research added {added} new evidence items", iteration, "search_web", result_count=added)
            elif decision.action == "analyze_evidence":
                if not evidence:
                    record("tool_skipped", "No evidence was available to analyze", iteration, "analyze_evidence", "rejected")
                    continue
                record("tool_selected", "Agent selected structured evidence analysis", iteration, "analyze_evidence")
                self._analyze(question, angles, evidence)
                conflicts = self._detect_conflicts(evidence)
                record("evidence_analyzed", f"Analyzed {len(evidence)} evidence items", iteration, "analyze_evidence", evidence_count=len(evidence))
                if conflicts:
                    record("contradiction_found", f"Found {len(conflicts)} research angles with opposing evidence", iteration, "analyze_evidence", conflict_count=len(conflicts))

        if evidence and any(not item.key_claim for item in evidence):
            self._analyze(question, angles, evidence)
            record("evidence_analyzed", f"Final analysis covered {len(evidence)} evidence items", iterations, "analyze_evidence", evidence_count=len(evidence))

        conflicts = self._detect_conflicts(evidence)
        brief = self._synthesize(question, angles, evidence, conflicts)
        if stop_reason == "iteration limit reached":
            caveat = "Research stopped at the iteration limit; unresolved evidence gaps may remain."
            if caveat not in brief.conflicting_signals:
                brief.conflicting_signals.append(caveat)
        verified, unsupported = self._verify_citations(brief, evidence)
        record("citation_verification_completed", f"Verified {verified} cited conclusions; {unsupported} lacked valid evidence links", iterations, "citation_verifier", verified=verified, unsupported=unsupported)
        if not any(event.event_type == "research_stopped" for event in events):
            record("research_stopped", stop_reason, iterations, "bounds")
        return MarketResearch(question, angles, evidence, conflicts, brief, events, self._provider_warnings(), stop_reason, iterations)

    def _plan(self, question: str) -> list[ResearchAngle]:
        payload = self.llm.complete_json(
            "You are a market research agent creating an initial plan. Return JSON only. Decompose the question into 3-4 distinct research angles. Do not answer it.",
            f'''Question: {question}\nReturn: {{"angles":[{{"name":"short label","question":"targeted research question","rationale":"why this matters"}}]}}''',
        )
        angles = [ResearchAngle(str(item["name"]), str(item["question"]), str(item.get("rationale", ""))) for item in payload.get("angles", [])[:4] if item.get("name") and item.get("question")]
        if len(angles) < 3:
            raise ValueError("Research plan must contain at least three valid angles")
        return angles

    def _decide(self, question: str, angles: list[ResearchAngle], evidence: list[MarketEvidence], conflicts: list[MarketConflict], budgets: dict[str, int], iteration: int) -> AgentDecision:
        state = {
            "iteration": iteration,
            "angles": [asdict(angle) for angle in angles],
            "evidence": [{"citation": item.citation_id, "source_type": item.source_type, "angle": item.research_angle, "claim": item.key_claim, "stance": item.stance, "kind": item.evidence_kind, "date": item.date, "excerpt": item.excerpt[:500]} for item in evidence],
            "conflicts": [asdict(conflict) for conflict in conflicts],
            "remaining_budget": budgets,
            "minimum_evidence": self.limits.min_evidence,
        }
        system = "You control a bounded market research loop. Choose exactly one next action based on actual state: search_local, search_web, analyze_evidence, or stop. Search when a named evidence gap remains; analyze after useful new evidence; investigate contradictions with a targeted search; stop only when evidence has been analyzed and is sufficient for a cautious brief. Return JSON only. Never claim a tool ran."
        user = f'''Question: {question}\nResearch state: {json.dumps(state, ensure_ascii=False)}\nReturn: {{"action":"search_local|search_web|analyze_evidence|stop","query":"tool query or empty","angle":"existing angle name or empty","rationale":"decision reason","evidence_gap":"specific missing evidence or empty"}}'''
        rejections: list[str] = []
        for attempt in range(2):
            correction = "" if attempt == 0 else f" Your previous decision was rejected: {rejections[-1]}. Return one corrected object matching the schema exactly."
            payload = self.llm.complete_json(system + correction, user)
            try:
                decision = self._validated_decision(payload, angles)
                decision.source = "llm" if attempt == 0 else "llm_retry"
                decision.rejection_reasons = rejections
                return decision
            except ValueError as exc:
                rejections.append(str(exc))
        return self._recover_decision(question, angles, evidence, budgets, rejections)

    def _validated_decision(self, payload: dict[str, Any], angles: list[ResearchAngle]) -> AgentDecision:
        if not isinstance(payload, dict):
            raise ValueError("Decision must be a JSON object")
        action = "_".join(str(payload.get("action", "")).strip().lower().replace("-", " ").split())
        if action not in self.ACTIONS:
            raise ValueError("Decision used an unsupported action")
        raw_angle = " ".join(str(payload.get("angle", "")).split())
        angle_lookup = {" ".join(item.name.split()).casefold(): item.name for item in angles}
        angle = angle_lookup.get(raw_angle.casefold(), "") if raw_angle else ""
        if raw_angle and not angle:
            raise ValueError("Decision referenced an unknown research angle")
        query = str(payload.get("query", "")).strip()
        if action.startswith("search_") and not query:
            raise ValueError("Search decision omitted its query")
        return AgentDecision(
            action,
            str(payload.get("rationale", "")).strip(),
            query,
            angle,
            str(payload.get("evidence_gap", "")).strip(),
        )

    def _recover_decision(self, question: str, angles: list[ResearchAngle], evidence: list[MarketEvidence], budgets: dict[str, int], rejections: list[str]) -> AgentDecision:
        angle = angles[0].name if angles else ""
        can_stop, _ = self._can_stop(evidence)
        if can_stop:
            return AgentDecision("stop", "Deterministic recovery stopped because the programmatic evidence threshold was satisfied.", source="deterministic_recovery", rejection_reasons=rejections)
        if evidence and any(not item.key_claim for item in evidence):
            return AgentDecision("analyze_evidence", "Deterministic recovery selected analysis for unclassified evidence.", source="deterministic_recovery", rejection_reasons=rejections)
        if not evidence and budgets.get("search_local", 0) > 0:
            return AgentDecision("search_local", "Deterministic recovery selected bounded local retrieval because no evidence exists.", question, angle, "No evidence has been retrieved.", "deterministic_recovery", rejections)
        if budgets.get("search_web", 0) > 0:
            return AgentDecision("search_web", "Deterministic recovery selected bounded web retrieval for the remaining evidence gap.", question, angle, "Current evidence is insufficient.", "deterministic_recovery", rejections)
        if budgets.get("search_local", 0) > 0:
            return AgentDecision("search_local", "Deterministic recovery selected the remaining bounded local retrieval.", question, angle, "Current evidence is insufficient.", "deterministic_recovery", rejections)
        return AgentDecision(
            "stop",
            "Research incomplete: model decisions remained invalid and no safe tool budget remained.",
            source="deterministic_recovery",
            rejection_reasons=rejections,
            incomplete_stop=True,
        )

    @staticmethod
    def _add_local_evidence(results: list[Any], angle: str, evidence: list[MarketEvidence], seen: set[str]) -> int:
        added = 0
        for result in results:
            key = f"local:{result.chunk.id}"
            if key in seen:
                continue
            seen.add(key)
            evidence.append(MarketEvidence(key, "LOCAL", result.chunk.title, Path(result.chunk.path).name, result.chunk.path, result.chunk.metadata.get("section", ""), result.chunk.text, angle, relevance=round(result.score, 3)))
            added += 1
        return added

    @staticmethod
    def _add_web_evidence(results: list[dict[str, str]], angle: str, evidence: list[MarketEvidence], seen: set[str]) -> int:
        added = 0
        for index, item in enumerate(results):
            key = f"web:{item.get('url') or angle + str(index)}"
            if key in seen:
                continue
            seen.add(key)
            evidence.append(MarketEvidence(key, "WEB", item.get("title", "Untitled source"), item.get("source", "Web source"), item.get("url", ""), item.get("date", ""), item.get("content", ""), angle))
            added += 1
        return added

    def _analyze(self, question: str, angles: list[ResearchAngle], evidence: list[MarketEvidence]) -> None:
        compact = [{"id": item.id, "angle": item.research_angle, "title": item.title, "date": item.date, "excerpt": item.excerpt[:1200]} for item in evidence]
        payload = self.llm.complete_json(
            "You analyze market evidence. For each item, extract one grounded claim; classify stance as SUPPORT, CONTRADICT, or NEUTRAL and kind as FACT, OPINION, or FORECAST; identify time period and key assumptions. Return JSON only. Do not infer facts absent from the passage.",
            f'''Question: {question}\nResearch angles: {json.dumps([asdict(angle) for angle in angles], ensure_ascii=False)}\nEvidence: {json.dumps(compact, ensure_ascii=False)}\nReturn: {{"evidence":[{{"id":"exact id","research_angle":"existing angle","stance":"SUPPORT|CONTRADICT|NEUTRAL","kind":"FACT|OPINION|FORECAST","key_claim":"grounded sentence","explanation":"relation to question","time_period":"stated period or unknown","assumptions":["explicit assumptions"]}}]}}''',
        )
        by_id = {item.id: item for item in evidence}
        valid_angles = {angle.name for angle in angles}
        for analysis in payload.get("evidence", []):
            item = by_id.get(str(analysis.get("id")))
            if not item:
                continue
            stance = str(analysis.get("stance", "NEUTRAL")).upper()
            kind = str(analysis.get("kind", "UNCLASSIFIED")).upper()
            proposed_angle = str(analysis.get("research_angle") or item.research_angle)
            item.stance = stance if stance in {"SUPPORT", "CONTRADICT", "NEUTRAL"} else "NEUTRAL"
            item.evidence_kind = kind if kind in {"FACT", "OPINION", "FORECAST"} else "UNCLASSIFIED"
            item.research_angle = proposed_angle if proposed_angle in valid_angles else item.research_angle
            item.key_claim = str(analysis.get("key_claim", ""))
            item.explanation = str(analysis.get("explanation", ""))
            item.time_period = str(analysis.get("time_period", ""))
            item.assumptions = [str(value) for value in analysis.get("assumptions", [])][:4]

    @staticmethod
    def _detect_conflicts(evidence: list[MarketEvidence]) -> list[MarketConflict]:
        conflicts: list[MarketConflict] = []
        for angle in dict.fromkeys(item.research_angle for item in evidence):
            supports = [item for item in evidence if item.research_angle == angle and item.stance == "SUPPORT" and item.key_claim]
            contradicts = [item for item in evidence if item.research_angle == angle and item.stance == "CONTRADICT" and item.key_claim]
            if supports and contradicts:
                periods = {item.time_period for item in [*supports, *contradicts] if item.time_period and item.time_period != "unknown"}
                explanation = "Opposing claims may reflect different assumptions"
                if len(periods) > 1:
                    explanation += " or time periods"
                conflicts.append(MarketConflict(angle, [item.key_claim for item in supports[:2]], [item.key_claim for item in contradicts[:2]], [item.citation_id for item in supports[:2]], [item.citation_id for item in contradicts[:2]], explanation + "."))
        return conflicts

    @staticmethod
    def _assign_citations(evidence: list[MarketEvidence]) -> None:
        counts = {"LOCAL": 0, "WEB": 0}
        prefixes = {"LOCAL": "L", "WEB": "W"}
        for item in evidence:
            if item.citation_id:
                counts[item.source_type] = max(counts[item.source_type], int(item.citation_id[1:]))
                continue
            counts[item.source_type] += 1
            item.citation_id = f"{prefixes[item.source_type]}{counts[item.source_type]}"

    def _can_stop(self, evidence: list[MarketEvidence]) -> tuple[bool, str]:
        if len(evidence) < self.limits.min_evidence:
            return False, f"At least {self.limits.min_evidence} evidence items are required"
        if any(not item.key_claim for item in evidence):
            return False, "New evidence still requires structured analysis"
        if not any(item.stance in {"SUPPORT", "CONTRADICT"} for item in evidence):
            return False, "Evidence has not established a directional claim"
        return True, "Evidence is analyzed and meets the minimum stopping threshold"

    def _synthesize(self, question: str, angles: list[ResearchAngle], evidence: list[MarketEvidence], conflicts: list[MarketConflict]) -> MarketBrief:
        analyzed = [{"citation": item.citation_id, "angle": item.research_angle, "stance": item.stance, "kind": item.evidence_kind, "claim": item.key_claim, "source": item.title, "date": item.date, "time_period": item.time_period, "assumptions": item.assumptions} for item in evidence]
        payload = self.llm.complete_json(
            "You write a concise market intelligence brief using only supplied evidence. Return JSON only. Every conclusion must list the citation IDs that support it. Separate unresolved contradictions and uncertainty. Evidence strength is LOW, MEDIUM, or HIGH and is qualitative, not a probability.",
            f'''Question: {question}\nAngles: {json.dumps([asdict(angle) for angle in angles], ensure_ascii=False)}\nEvidence: {json.dumps(analyzed, ensure_ascii=False)}\nDetected conflicts: {json.dumps([asdict(conflict) for conflict in conflicts], ensure_ascii=False)}\nReturn: {{"current_view":"2-4 cited sentences","conclusions":[{{"statement":"grounded conclusion","citations":["L1","W1"],"uncertainty":"specific caveat"}}],"key_drivers":["3-5 cited directional findings"],"conflicting_signals":["cited risks"],"evidence_strength":"LOW|MEDIUM|HIGH","monitor_next":["2-3 indicators"]}}''',
        )
        conclusions = [MarketConclusion(str(item.get("statement", "")), [str(citation) for citation in item.get("citations", [])], str(item.get("uncertainty", ""))) for item in payload.get("conclusions", [])[:5] if item.get("statement")]
        strength = str(payload.get("evidence_strength", "LOW")).upper()
        return MarketBrief(str(payload.get("current_view", "Insufficient evidence to form a market view.")), [str(item) for item in payload.get("key_drivers", [])][:5], [str(item) for item in payload.get("conflicting_signals", [])][:4], strength if strength in {"LOW", "MEDIUM", "HIGH"} else "LOW", [str(item) for item in payload.get("monitor_next", [])][:3], conclusions)

    @staticmethod
    def _verify_citations(brief: MarketBrief, evidence: list[MarketEvidence]) -> tuple[int, int]:
        valid = {item.citation_id for item in evidence}
        verified = 0
        unsupported: list[str] = []
        for conclusion in brief.conclusions:
            conclusion.citations = list(dict.fromkeys(citation for citation in conclusion.citations if citation in valid))
            conclusion.citation_verified = bool(conclusion.citations)
            if conclusion.citation_verified:
                verified += 1
            else:
                unsupported.append(conclusion.statement)
        brief.unsupported_claims = unsupported
        return verified, len(unsupported)

    def _provider_warnings(self) -> list[str]:
        warnings: list[str] = []
        for provider in (self.web, self.llm):
            drain = getattr(provider, "drain_warnings", None)
            if drain:
                warnings.extend(drain())
        return warnings


def build_market_agent(corpus: str | Path, *, model: str | None = None, limits: AgentLimits | None = None) -> MarketRabbitHoleAgent:
    from dotenv import load_dotenv

    load_dotenv()
    chunks = load_local_chunks(corpus)
    retrieval_backend = resolve_retrieval_backend()
    if retrieval_backend == "tfidf":
        rag: LocalRetriever = TfidfRetriever(chunks)
    else:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        rag = SemanticRAG(chunks, SentenceTransformerEncoder(model_name))
    zhipu_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    llm = ResilientStructuredLLM(ZhipuStructuredLLM(api_key=zhipu_key, model=model) if zhipu_key else None)
    web = ResilientWebSearchProvider(TavilySearchProvider(api_key=tavily_key) if tavily_key else None)
    return MarketRabbitHoleAgent(rag, llm, web, limits=limits)
