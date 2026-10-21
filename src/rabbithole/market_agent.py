"""Bounded single-agent market research workflow."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

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
    source_quality: str = "LOCAL_CONTEXT"


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
    citation_ids_valid: bool = False
    support_status: str = "UNVERIFIED"


@dataclass(slots=True)
class MarketBrief:
    current_view: str
    key_drivers: list[str]
    conflicting_signals: list[str]
    evidence_strength: str
    monitor_next: list[str]
    conclusions: list[MarketConclusion] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    evidence_strength_reason: str = ""
    partial_fallback: bool = False


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


@dataclass(slots=True)
class SearchAttempt:
    action: str
    query: str
    angle: str
    added: int
    rejected: int = 0


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
        historical_context: str = "",
    ) -> MarketResearch:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        research_question = question
        if historical_context.strip():
            research_question = f"{question}\n\n{historical_context.strip()}"
        events: list[ResearchEvent] = []

        def record(event_type: str, detail: str, iteration: int = 0, tool: str = "", status: str = "ok", **metadata: Any) -> None:
            event = ResearchEvent(event_type, detail, iteration, tool, status, metadata)
            events.append(event)
            if on_event:
                on_event(event)

        angles = self._plan(research_question)
        record("initial_research_plan", f"Created {len(angles)} question-specific research angles", angle_count=len(angles))
        session_rag = self.rag
        if uploaded_chunks:
            session_rag = self.rag.with_chunks([*self.rag.chunks, *uploaded_chunks])

        evidence: list[MarketEvidence] = []
        seen: set[str] = set()
        budgets = {"search_local": self.limits.max_local_searches, "search_web": self.limits.max_web_searches}
        search_history: list[SearchAttempt] = []
        stop_reason = "iteration limit reached"
        iterations = 0

        for iteration in range(1, self.limits.max_iterations + 1):
            iterations = iteration
            conflicts = self._detect_conflicts(evidence)
            decision = self._decide(research_question, angles, evidence, conflicts, budgets, iteration, search_history, self._requires_current_evidence(question))
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
                search_history.append(SearchAttempt("search_local", decision.query, decision.angle, added))
                self._assign_citations(evidence)
                record("search_performed", f"Local retrieval added {added} new evidence items", iteration, "search_local", result_count=added)
                if added == 0:
                    record("strategy_reconsidered", "Local retrieval produced no marginal evidence; repeating a similar local query will be rejected.", iteration, "search_local", "warning")
            elif decision.action == "search_web":
                record("tool_selected", "Agent selected live web research", iteration, "search_web", angle=decision.angle)
                candidate_limit = min(3, max(2, self.limits.web_results * 3))
                results = self.web.search(decision.query, candidate_limit)
                added, rejected = self._add_web_evidence(results, decision.query, decision.angle, evidence, seen, self.limits.web_results)
                search_history.append(SearchAttempt("search_web", decision.query, decision.angle, added, rejected))
                self._assign_citations(evidence)
                record("search_performed", f"Web research added {added} new evidence items; rejected {rejected} low-quality or irrelevant results", iteration, "search_web", result_count=added, rejected_count=rejected)
                if added == 0:
                    record("strategy_reconsidered", "Web retrieval produced no usable marginal evidence; the next search must target a different evidence gap.", iteration, "search_web", "warning")
            elif decision.action == "analyze_evidence":
                if not evidence:
                    record("tool_skipped", "No evidence was available to analyze", iteration, "analyze_evidence", "rejected")
                    continue
                record("tool_selected", "Agent selected structured evidence analysis", iteration, "analyze_evidence")
                self._analyze(research_question, angles, evidence)
                conflicts = self._detect_conflicts(evidence)
                record("evidence_analyzed", f"Analyzed {len(evidence)} evidence items", iteration, "analyze_evidence", evidence_count=len(evidence))
                if conflicts:
                    record("contradiction_found", f"Found {len(conflicts)} research angles with opposing evidence", iteration, "analyze_evidence", conflict_count=len(conflicts))

        if evidence and any(not item.key_claim for item in evidence):
            self._analyze(research_question, angles, evidence)
            record("evidence_analyzed", f"Final analysis covered {len(evidence)} evidence items", iterations, "analyze_evidence", evidence_count=len(evidence))

        conflicts = self._detect_conflicts(evidence)
        brief, synthesis_rejections = self._synthesize(research_question, angles, evidence, conflicts)
        for rejection in synthesis_rejections:
            record("synthesis_rejected", rejection, iterations, "brief_synthesis", "rejected")
        if brief.partial_fallback:
            record("synthesis_fallback_used", "Model synthesis remained invalid; emitted a labeled evidence-only partial report.", iterations, "brief_synthesis", "warning")
        if stop_reason == "iteration limit reached":
            caveat = "Research stopped at the iteration limit; unresolved evidence gaps may remain."
            if caveat not in brief.conflicting_signals:
                brief.conflicting_signals.append(caveat)
        supported, unsupported = self._verify_citations(brief, evidence)
        self._calibrate_evidence_strength(question, brief, evidence)
        record("citation_verification_completed", f"Found {supported} conclusions with passage support; {unsupported} were unsupported or weakly linked", iterations, "citation_verifier", supported=supported, unsupported=unsupported)
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

    def _decide(
        self,
        question: str,
        angles: list[ResearchAngle],
        evidence: list[MarketEvidence],
        conflicts: list[MarketConflict],
        budgets: dict[str, int],
        iteration: int,
        search_history: list[SearchAttempt] | None = None,
        current_evidence_required: bool | None = None,
    ) -> AgentDecision:
        search_history = search_history or []
        requires_current = self._requires_current_evidence(question) if current_evidence_required is None else current_evidence_required
        state = {
            "iteration": iteration,
            "angles": [asdict(angle) for angle in angles],
            "evidence": [{"citation": item.citation_id, "source_type": item.source_type, "angle": item.research_angle, "claim": item.key_claim, "stance": item.stance, "kind": item.evidence_kind, "date": item.date, "excerpt": item.excerpt[:500]} for item in evidence],
            "conflicts": [asdict(conflict) for conflict in conflicts],
            "remaining_budget": budgets,
            "minimum_evidence": self.limits.min_evidence,
            "requires_current_web_evidence": requires_current,
            "source_coverage": {
                "local": sum(item.source_type == "LOCAL" for item in evidence),
                "web": sum(item.source_type == "WEB" for item in evidence),
            },
            "search_history": [asdict(item) for item in search_history],
        }
        system = "You control a bounded market research loop. Choose exactly one action: search_local, search_web, analyze_evidence, or stop. Use the search history and marginal evidence counts: never repeat a failed or near-identical search. Prefer a targeted web query when current facts, dates, financial metrics, valuation, company disclosures, or opposing evidence are missing. Analyze after new evidence. Stop only when the evidence is analyzed and adequate for a cautious brief. Return JSON only; values must describe this research case, never schema instructions. Never claim a tool ran."
        user = f'''Question: {question}\nResearch state: {json.dumps(state, ensure_ascii=False)}\nReturn: one object with string fields action, query, angle, rationale, and evidence_gap. For non-search actions query must be empty. angle must be an existing angle name or empty.'''
        rejections: list[str] = []
        for attempt in range(2):
            correction = "" if attempt == 0 else f" Your previous decision was rejected: {rejections[-1]}. Return one corrected object matching the schema exactly."
            payload = self.llm.complete_json(system + correction, user)
            try:
                decision = self._validated_decision(payload, angles, search_history)
                if decision.action == "stop" and requires_current and not any(item.source_type == "WEB" for item in evidence) and budgets.get("search_web", 0) > 0:
                    raise ValueError("Stop decision omitted required current web evidence while web budget remained")
                decision.source = "llm" if attempt == 0 else "llm_retry"
                decision.rejection_reasons = rejections
                return decision
            except ValueError as exc:
                rejections.append(str(exc))
        return self._recover_decision(question, angles, evidence, budgets, rejections, search_history, requires_current)

    def _validated_decision(self, payload: dict[str, Any], angles: list[ResearchAngle], search_history: list[SearchAttempt] | None = None) -> AgentDecision:
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
        for attempt in search_history or []:
            if attempt.action == action and self._queries_similar(query, attempt.query):
                outcome = "zero new evidence" if attempt.added == 0 else "already-covered evidence"
                raise ValueError(f"Search decision repeated a near-duplicate query that produced {outcome}")
        return AgentDecision(
            action,
            str(payload.get("rationale", "")).strip(),
            query,
            angle,
            str(payload.get("evidence_gap", "")).strip(),
        )

    def _recover_decision(
        self,
        question: str,
        angles: list[ResearchAngle],
        evidence: list[MarketEvidence],
        budgets: dict[str, int],
        rejections: list[str],
        search_history: list[SearchAttempt] | None = None,
        current_evidence_required: bool | None = None,
    ) -> AgentDecision:
        search_history = search_history or []
        angle = angles[0].name if angles else ""
        requires_current = self._requires_current_evidence(question) if current_evidence_required is None else current_evidence_required
        needs_web = requires_current and not any(item.source_type == "WEB" for item in evidence)
        can_stop, _ = self._can_stop(evidence)
        if can_stop and not (needs_web and budgets.get("search_web", 0) > 0):
            return AgentDecision("stop", "Deterministic recovery stopped because the programmatic evidence threshold was satisfied.", source="deterministic_recovery", rejection_reasons=rejections)
        if evidence and any(not item.key_claim for item in evidence):
            return AgentDecision("analyze_evidence", "Deterministic recovery selected analysis for unclassified evidence.", source="deterministic_recovery", rejection_reasons=rejections)
        local_attempts = [item for item in search_history if item.action == "search_local"]
        web_attempts = [item for item in search_history if item.action == "search_web"]
        if needs_web and budgets.get("search_web", 0) > 0:
            return AgentDecision("search_web", "Deterministic recovery prioritized current public evidence required by the question.", self._targeted_recovery_query(question, angles, evidence, web_attempts), angle, "Current public evidence is missing.", "deterministic_recovery", rejections)
        if not evidence and budgets.get("search_local", 0) > 0 and not local_attempts:
            return AgentDecision("search_local", "Deterministic recovery selected bounded local retrieval because no evidence exists.", question, angle, "No evidence has been retrieved.", "deterministic_recovery", rejections)
        if budgets.get("search_web", 0) > 0:
            return AgentDecision("search_web", "Deterministic recovery selected a targeted web query for an unresolved evidence gap.", self._targeted_recovery_query(question, angles, evidence, web_attempts), angle, "Current evidence lacks an independent current or opposing source.", "deterministic_recovery", rejections)
        if budgets.get("search_local", 0) > 0 and not any(item.added == 0 for item in local_attempts):
            query = angles[min(len(local_attempts), len(angles) - 1)].question if angles else question
            return AgentDecision("search_local", "Deterministic recovery selected a distinct remaining local research angle.", query, angle, "A local research angle remains uncovered.", "deterministic_recovery", rejections)
        return AgentDecision(
            "stop",
            "Research incomplete: model decisions remained invalid and no safe tool budget remained.",
            source="deterministic_recovery",
            rejection_reasons=rejections,
            incomplete_stop=True,
        )

    @staticmethod
    def _requires_current_evidence(question: str) -> bool:
        lowered = question.casefold()
        return any(token in lowered for token in ("latest", "current", "recent", "today", "now", "最新", "目前", "当前", "近期", "市场情况"))

    @classmethod
    def _queries_similar(cls, left: str, right: str) -> bool:
        left_terms, right_terms = cls._content_terms(left), cls._content_terms(right)
        if not left_terms or not right_terms:
            return left.strip().casefold() == right.strip().casefold()
        return len(left_terms & right_terms) / len(left_terms | right_terms) >= 0.8

    @staticmethod
    def _targeted_recovery_query(question: str, angles: list[ResearchAngle], evidence: list[MarketEvidence], attempts: list[SearchAttempt]) -> str:
        covered = {item.angle for item in attempts}
        target = next((angle for angle in angles if angle.name not in covered), angles[min(len(attempts), len(angles) - 1)] if angles else None)
        focus = target.question if target else question
        missing = "official company disclosure earnings valuation risks opposing evidence"
        if any(item.source_type == "WEB" for item in evidence):
            missing = "counterargument downside risk conflicting evidence"
        return f"{focus} {missing}".strip()

    @staticmethod
    def _add_local_evidence(results: list[Any], angle: str, evidence: list[MarketEvidence], seen: set[str]) -> int:
        added = 0
        for result in results:
            key = f"local:{result.chunk.id}"
            if key in seen:
                continue
            seen.add(key)
            evidence.append(MarketEvidence(key, "LOCAL", result.chunk.title, Path(result.chunk.path).name, result.chunk.path, result.chunk.metadata.get("section", ""), result.chunk.text, angle, relevance=round(result.score, 3), source_quality="LOCAL_CONTEXT"))
            added += 1
        return added

    @classmethod
    def _add_web_evidence(
        cls,
        results: list[dict[str, str]],
        query: str,
        angle: str,
        evidence: list[MarketEvidence],
        seen: set[str],
        max_items: int,
    ) -> tuple[int, int]:
        added = 0
        rejected = 0
        ranked: list[tuple[int, dict[str, str], str, str]] = []
        for item in results:
            title = " ".join(str(item.get("title", "")).split())
            content = " ".join(str(item.get("content", "")).split())
            url = cls._normalize_url(str(item.get("url", "")))
            if not cls._usable_web_result(query, title, content, url):
                rejected += 1
                continue
            quality, score = cls._source_quality(url, title)
            ranked.append((score, item, quality, url))
        for _, item, quality, url in sorted(ranked, key=lambda row: row[0], reverse=True):
            if added >= max_items:
                break
            key = f"web:{url}"
            if key in seen:
                rejected += 1
                continue
            seen.add(key)
            evidence.append(MarketEvidence(key, "WEB", str(item["title"]).strip(), item.get("source", "Web source"), url, item.get("date", ""), str(item["content"]).strip(), angle, source_quality=quality))
            added += 1
        return added, rejected

    @classmethod
    def _usable_web_result(cls, query: str, title: str, content: str, url: str) -> bool:
        navigation_titles = {"menu", "home", "homepage", "navigation", "untitled source", "index", "login", "sign in"}
        if not url.startswith(("http://", "https://")) or not title or title.casefold() in navigation_titles:
            return False
        if len(content) < 30:
            return False
        query_terms = cls._content_terms(query)
        result_terms = cls._content_terms(f"{title} {content}")
        generic = {"latest", "public", "evidence", "market", "thesis", "official", "company", "disclosure", "risks", "opposing", "current", "recent", "research"}
        meaningful = {term for term in query_terms - generic if len(term) > 2 or any("\u4e00" <= char <= "\u9fff" for char in term)}
        return not meaningful or bool(meaningful & result_terms)

    @staticmethod
    def _normalize_url(url: str) -> str:
        parts = urlsplit(url.strip())
        if not parts.scheme or not parts.netloc:
            return ""
        path = parts.path.rstrip("/") or "/"
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))

    @staticmethod
    def _source_quality(url: str, title: str) -> tuple[str, int]:
        lowered = f"{url} {title}".casefold()
        if any(marker in lowered for marker in ("sec.gov", ".gov/", "investor", "investors", "earnings", "annual-report", "quarterly-results")):
            return "PRIMARY", 3
        if any(marker in lowered for marker in ("reuters.com", "bloomberg.com", "ft.com", "wsj.com")):
            return "REPUTABLE_REPORTING", 2
        return "OTHER_WEB", 1

    @staticmethod
    def _content_terms(value: str) -> set[str]:
        lowered = value.casefold()
        latin = set(re.findall(r"[a-z0-9][a-z0-9._%-]{1,}", lowered))
        cjk_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
        cjk = {run[index:index + 2] for run in cjk_runs for index in range(max(0, len(run) - 1))}
        return latin | cjk

    def _analyze(self, question: str, angles: list[ResearchAngle], evidence: list[MarketEvidence]) -> None:
        compact = [{"id": item.id, "angle": item.research_angle, "title": item.title, "date": item.date, "excerpt": item.excerpt[:1200]} for item in evidence]
        payload = self.llm.complete_json(
            "You analyze market evidence. For each item, extract one grounded claim; classify stance as SUPPORT, CONTRADICT, or NEUTRAL and kind as FACT, OPINION, or FORECAST; identify time period and key assumptions. Return JSON only. Do not infer facts absent from the passage.",
            f'''Question: {question}\nResearch angles: {json.dumps([asdict(angle) for angle in angles], ensure_ascii=False)}\nEvidence: {json.dumps(compact, ensure_ascii=False)}\nReturn: an object whose evidence field is a list. Each list item must contain: the exact evidence id; an existing research_angle; stance from SUPPORT, CONTRADICT, NEUTRAL; kind from FACT, OPINION, FORECAST; one passage-grounded key_claim; a concise explanation; the stated time_period or "unknown"; and an assumptions list. Do not copy these instructions into values.''',
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
            claim = " ".join(str(analysis.get("key_claim", "")).split())
            item.key_claim = "" if self._contains_placeholder(claim) else claim
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

    def _synthesize(self, question: str, angles: list[ResearchAngle], evidence: list[MarketEvidence], conflicts: list[MarketConflict]) -> tuple[MarketBrief, list[str]]:
        if not evidence:
            return self._partial_brief(evidence, conflicts), ["no evidence was available for model synthesis"]
        analyzed = [{"citation": item.citation_id, "angle": item.research_angle, "stance": item.stance, "kind": item.evidence_kind, "claim": item.key_claim, "source": item.title, "date": item.date, "time_period": item.time_period, "assumptions": item.assumptions} for item in evidence]
        system = "You write a concise market intelligence brief using only supplied evidence. Make it decision-useful: explain mechanisms, distinguish facts from forecasts, surface opposing evidence, and state what remains unknown. Every material conclusion must cite supporting evidence IDs. Never invent dates, figures, valuation metrics, or citations. Return JSON only. Values must contain case-specific analysis, never field instructions or template language. Evidence strength must be LOW, MEDIUM, or HIGH and is qualitative, not a probability."
        user = f'''Question: {question}\nAngles: {json.dumps([asdict(angle) for angle in angles], ensure_ascii=False)}\nEvidence: {json.dumps(analyzed, ensure_ascii=False)}\nDetected conflicts: {json.dumps([asdict(conflict) for conflict in conflicts], ensure_ascii=False)}\nReturn an object with: current_view as substantive case-specific prose; conclusions as a list of objects with statement, citations, and uncertainty; key_drivers as a list; conflicting_signals as a list; evidence_strength; and monitor_next as a list of concrete indicators. Use empty lists when evidence cannot support a section.'''
        rejections: list[str] = []
        for attempt in range(2):
            correction = "" if attempt == 0 else f" The prior output was rejected because {rejections[-1]}. Correct that defect and return the full object again."
            payload = self.llm.complete_json(system + correction, user)
            try:
                return self._parse_brief(payload, bool(evidence)), rejections
            except ValueError as exc:
                rejections.append(str(exc))
        return self._partial_brief(evidence, conflicts), rejections

    def _parse_brief(self, payload: dict[str, Any], has_evidence: bool) -> MarketBrief:
        if not isinstance(payload, dict):
            raise ValueError("the response was not a JSON object")
        current_view = " ".join(str(payload.get("current_view", "")).split())
        if not current_view or self._contains_placeholder(current_view):
            raise ValueError("current_view was empty or contained template text")
        conclusions: list[MarketConclusion] = []
        for item in payload.get("conclusions", [])[:5]:
            statement = " ".join(str(item.get("statement", "")).split())
            if statement and not self._contains_placeholder(statement):
                conclusions.append(MarketConclusion(statement, [str(value) for value in item.get("citations", [])], str(item.get("uncertainty", ""))))
        if has_evidence and not conclusions:
            raise ValueError("no substantive source-linked conclusion was returned")
        key_drivers = self._clean_list(payload.get("key_drivers", []), 5)
        monitor_next = self._clean_list(payload.get("monitor_next", []), 3)
        if has_evidence and (not key_drivers or not monitor_next):
            raise ValueError("key_drivers or monitor_next was empty or contained template text")
        strength = str(payload.get("evidence_strength", "LOW")).upper()
        return MarketBrief(
            current_view,
            key_drivers,
            self._clean_list(payload.get("conflicting_signals", []), 4),
            strength if strength in {"LOW", "MEDIUM", "HIGH"} else "LOW",
            monitor_next,
            conclusions,
        )

    def _partial_brief(self, evidence: list[MarketEvidence], conflicts: list[MarketConflict]) -> MarketBrief:
        usable = [item for item in evidence if item.key_claim and item.citation_id]
        if usable:
            summary = "Partial evidence report: " + " ".join(f"{item.key_claim} [{item.citation_id}]" for item in usable[:3])
            conclusions = [MarketConclusion(item.key_claim, [item.citation_id], "Model synthesis was invalid; this statement reproduces a retrieved, analyzed passage.") for item in usable[:3]]
            drivers = [f"{item.key_claim} [{item.citation_id}]" for item in usable[:5]]
        else:
            summary = "Partial evidence report: available sources were insufficient to form a grounded market view."
            conclusions, drivers = [], []
        caveats = [conflict.explanation for conflict in conflicts[:3]] or ["Model synthesis was invalid; only directly retrieved evidence is shown."]
        return MarketBrief(summary, drivers, caveats, "LOW", ["Collect current primary-source evidence for the unresolved research angles."], conclusions, partial_fallback=True)

    @classmethod
    def _clean_list(cls, values: Any, limit: int) -> list[str]:
        if not isinstance(values, list):
            return []
        return [cleaned for value in values[:limit] if (cleaned := " ".join(str(value).split())) and not cls._contains_placeholder(cleaned)]

    @staticmethod
    def _contains_placeholder(value: str) -> bool:
        lowered = value.casefold()
        markers = (
            "2-4 cited sentences", "grounded conclusion", "specific caveat",
            "3-5 cited directional findings", "cited risks", "2-3 indicators",
            "tool query or empty", "specific missing evidence or empty",
        )
        return any(marker in lowered for marker in markers)

    @classmethod
    def _verify_citations(cls, brief: MarketBrief, evidence: list[MarketEvidence]) -> tuple[int, int]:
        by_citation = {item.citation_id: item for item in evidence}
        supported = 0
        unsupported: list[str] = []
        for conclusion in brief.conclusions:
            conclusion.citations = list(dict.fromkeys(citation for citation in conclusion.citations if citation in by_citation))
            conclusion.citation_ids_valid = bool(conclusion.citations)
            cited = [by_citation[citation] for citation in conclusion.citations]
            conclusion.support_status = cls._claim_support(conclusion.statement, cited)
            conclusion.citation_verified = conclusion.support_status == "SUPPORTED"
            if conclusion.citation_verified:
                supported += 1
            else:
                unsupported.append(conclusion.statement)
        brief.unsupported_claims = unsupported
        return supported, len(unsupported)

    @classmethod
    def _claim_support(cls, statement: str, cited: list[MarketEvidence]) -> str:
        if not cited:
            return "UNSUPPORTED"
        claim_numbers = set(re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)?%?", statement))
        passages = " ".join(f"{item.key_claim} {item.excerpt}" for item in cited)
        passage_numbers = set(re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)?%?", passages))
        if claim_numbers - passage_numbers:
            return "UNSUPPORTED"
        stopwords = {"this", "that", "with", "from", "have", "will", "market", "evidence", "industry", "company", "目前", "市场", "行业", "投资"}
        claim_terms = cls._content_terms(statement) - stopwords
        passage_terms = cls._content_terms(passages) - stopwords
        overlap = len(claim_terms & passage_terms) / max(1, len(claim_terms))
        if overlap >= 0.25 or (claim_numbers and claim_numbers <= passage_numbers):
            return "SUPPORTED"
        return "WEAK"

    @classmethod
    def _calibrate_evidence_strength(cls, question: str, brief: MarketBrief, evidence: list[MarketEvidence]) -> None:
        usable = [item for item in evidence if item.key_claim]
        web = [item for item in usable if item.source_type == "WEB"]
        supported = [item for item in brief.conclusions if item.support_status == "SUPPORTED"]
        primary_or_reporting = [item for item in web if item.source_quality in {"PRIMARY", "REPUTABLE_REPORTING"}]
        reasons: list[str] = []
        cap = "HIGH"
        if len(usable) < 3 or not supported:
            cap = "LOW"
            reasons.append("fewer than three usable evidence items or no passage-supported conclusion")
        if cls._requires_current_evidence(question) and not web:
            cap = "LOW"
            reasons.append("the question requires current evidence but no usable live-web source was retained")
        elif cap != "LOW" and (not web or len({item.source_type for item in usable}) < 2):
            cap = "MEDIUM"
            reasons.append("evidence lacks local/web source diversity")
        elif cap != "LOW" and (len(usable) < 5 or len(primary_or_reporting) < 2):
            cap = "MEDIUM"
            reasons.append("the evidence set is limited or has fewer than two high-quality public sources")
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        model_strength = brief.evidence_strength if brief.evidence_strength in order else "LOW"
        brief.evidence_strength = min((model_strength, cap), key=order.get)
        brief.evidence_strength_reason = "; ".join(reasons) or "multiple usable sources and passage-supported conclusions"

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
