"""Small provider boundaries for Zhipu structured reasoning and Tavily search."""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
import urllib.request
from typing import Any, Protocol


class ProviderError(RuntimeError):
    """Sanitized provider failure safe to surface without leaking credentials."""


class StructuredLLM(Protocol):
    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Return a parsed JSON object."""


class WebSearchProvider(Protocol):
    def search(self, query: str, max_results: int = 2) -> list[dict[str, str]]:
        """Return compact web results with provenance."""


class ZhipuStructuredLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        api_key = api_key or os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ProviderError("Zhipu API key is not configured.")
        try:
            from zhipuai import ZhipuAI
        except ImportError as exc:
            raise RuntimeError("Install project dependencies before starting Market RabbitHole") from exc
        self.api_key = api_key
        self.base_url = os.getenv("ZHIPU_BASE_URL", "").strip()
        self.client = ZhipuAI(api_key=api_key, base_url=self.base_url) if self.base_url else ZhipuAI(api_key=api_key)
        self.model = model or os.getenv("ZHIPU_MODEL", "glm-4-flash")

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.1,
            )
            return _parse_json(response.choices[0].message.content)
        except Exception as exc:
            _development_log("Zhipu", exc, secret=self.api_key, context=f"{_llm_stage(system)} model={self.model}")
            raise ProviderError("Zhipu reasoning request failed.") from exc


class TavilySearchProvider:
    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ProviderError("Tavily API key is not configured.")

    def search(self, query: str, max_results: int = 2) -> list[dict[str, str]]:
        body = json.dumps({
            "api_key": self.api_key,
            "query": query,
            "topic": "news",
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ProviderError("Tavily live search failed.") from exc
        return [
            {
                "title": str(item.get("title", "Untitled source")),
                "url": str(item.get("url", "")),
                "source": _publisher(str(item.get("url", ""))),
                "date": str(item.get("published_date") or ""),
                "content": str(item.get("content", "")),
            }
            for item in payload.get("results", [])
        ]


class FallbackStructuredLLM:
    """Understandable local fallback used only when Zhipu is unavailable."""

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        lowered = system.lower()
        if "initial plan" in lowered or "planner" in lowered:
            question = _between(user, "Question: ", "\nReturn:")
            return {"angles": [
                {"name": "Core drivers", "question": f"What supply, demand, or competitive forces explain: {question}", "rationale": "Tests the main market mechanism."},
                {"name": "Counter-signals", "question": f"What evidence complicates or challenges: {question}", "rationale": "Prevents a one-sided interpretation."},
                {"name": "Market context", "question": f"What timing, policy, or macro context matters for: {question}", "rationale": "Places the evidence in its relevant horizon."},
            ]}
        if "bounded market research loop" in lowered:
            question = _between(user, "Question: ", "\nResearch state:")
            state = json.loads(_between(user, "Research state: ", "\nReturn:"))
            evidence = state.get("evidence", [])
            angle = state["angles"][0]["name"] if state.get("angles") else "Core drivers"
            budget = state.get("remaining_budget", {})
            if not evidence and budget.get("search_local", 0):
                return {"action": "search_local", "query": question, "angle": angle, "rationale": "Start with available private context.", "evidence_gap": "No evidence has been retrieved."}
            if any(not item.get("claim") for item in evidence):
                return {"action": "analyze_evidence", "query": "", "angle": "", "rationale": "New evidence requires classification.", "evidence_gap": ""}
            if len(evidence) < state.get("minimum_evidence", 3) and budget.get("search_web", 0):
                return {"action": "search_web", "query": question, "angle": angle, "rationale": "Add current public evidence.", "evidence_gap": "Current evidence is below the minimum threshold."}
            return {"action": "stop", "query": "", "angle": "", "rationale": "Available evidence is analyzed and bounded research can stop.", "evidence_gap": ""}
        if "analyze market evidence" in lowered:
            rows = json.loads(_between(user, "Evidence: ", "\nReturn:"))
            return {"evidence": [{
                "id": item["id"],
                "research_angle": item["angle"],
                "stance": "NEUTRAL",
                "kind": "FACT",
                "key_claim": _first_sentence(item.get("excerpt", "")),
                "explanation": "Retrieved evidence; GLM classification was unavailable.",
                "time_period": "unknown",
                "assumptions": [],
            } for item in rows]}
        rows = json.loads(_between(user, "Evidence: ", "\nDetected conflicts:"))
        claims = [f"{item.get('claim', '')} [{item.get('citation', '')}]".strip() for item in rows if item.get("claim")]
        summary = " ".join(claims[:3]) or "Local retrieval returned no usable evidence."
        return {
            "current_view": summary,
            "conclusions": [
                {"statement": item.get("claim", ""), "citations": [item.get("citation", "")], "uncertainty": "GLM synthesis was unavailable."}
                for item in rows[:3]
                if item.get("claim") and item.get("citation")
            ],
            "key_drivers": claims[:5],
            "conflicting_signals": ["GLM synthesis was unavailable; evidence has not been fully classified."],
            "evidence_strength": "LOW",
            "monitor_next": ["Restore Zhipu access and rerun structured synthesis."],
        }


class ResilientStructuredLLM:
    def __init__(self, primary: StructuredLLM | None, fallback: StructuredLLM | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or FallbackStructuredLLM()
        self._warnings: list[str] = []

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        if self.primary is not None:
            try:
                return self.primary.complete_json(system, user)
            except Exception as exc:
                stage = _llm_stage(system)
                underlying = exc.__cause__ or exc
                warning = (
                    "Zhipu unavailable during "
                    f"{stage} ({type(underlying).__name__}) — local deterministic fallback used."
                )
        else:
            warning = "Zhipu unavailable — local deterministic fallback used for reasoning."
        if warning not in self._warnings:
            self._warnings.append(warning)
        return self.fallback.complete_json(system, user)

    def drain_warnings(self) -> list[str]:
        warnings, self._warnings = self._warnings, []
        return warnings


class ResilientWebSearchProvider:
    def __init__(self, primary: WebSearchProvider | None) -> None:
        self.primary = primary
        self._warnings: list[str] = []

    def search(self, query: str, max_results: int = 1) -> list[dict[str, str]]:
        if self.primary is not None:
            try:
                return self.primary.search(query, max_results)
            except Exception as exc:
                log_safe_exception("Tavily", "live web search", exc)
        warning = "Live web research unavailable — brief uses local evidence only."
        if warning not in self._warnings:
            self._warnings.append(warning)
        return []

    def drain_warnings(self) -> list[str]:
        warnings, self._warnings = self._warnings, []
        return warnings


def _publisher(url: str) -> str:
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1) if match else "Web source"


def _parse_json(value: str) -> dict[str, Any]:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.S)
        if not match:
            raise ValueError("Zhipu response did not contain valid JSON")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from Zhipu")
    return parsed


def _between(value: str, start: str, end: str) -> str:
    return value.split(start, 1)[1].split(end, 1)[0]


def _first_sentence(value: str) -> str:
    value = value.strip()
    match = re.search(r"^.*?[.!?](?:\s|$)", value)
    return (match.group(0) if match else value[:240]).strip()


def _llm_stage(system: str) -> str:
    """Return a safe operation label without including prompt or document text."""
    lowered = system.lower()
    if "initial plan" in lowered or "planner" in lowered:
        return "research planning"
    if "bounded market research loop" in lowered:
        return "agent decision"
    if "analyze market evidence" in lowered:
        return "evidence analysis"
    if "market intelligence brief" in lowered:
        return "brief synthesis"
    return "structured reasoning"


def _development_log(provider: str, exc: Exception, *, secret: str = "", context: str = "") -> None:
    """Print safe provider diagnostics only in development mode."""
    if os.getenv("RABBITHOLE_DEBUG", "").lower() not in {"1", "true", "yes", "on"}:
        return
    log_safe_exception(provider, context or "provider request", exc, debug_label=True)


def log_safe_exception(component: str, stage: str, exc: Exception, *, debug_label: bool = False) -> None:
    """Log operational diagnostics without exception text, prompts, or request data."""
    root = exc
    while root.__cause__ is not None:
        root = root.__cause__
    status = getattr(root, "status_code", None) or getattr(root, "status", None) or getattr(root, "code", None)
    frames = traceback.extract_tb(root.__traceback__ or exc.__traceback__)
    locations = [f"{os.path.basename(frame.filename)}:{frame.lineno}:{frame.name}" for frame in frames[-3:]]
    label = "debug" if debug_label else "error"
    fields = [f"component={component}", f"stage={stage}", f"type={type(root).__name__}"]
    if status is not None:
        fields.append(f"status={status}")
    if locations:
        fields.append(f"location={' <- '.join(locations)}")
    if isinstance(exc, ProviderError):
        fields.append(f"message={str(exc)}")
    print(f"[MarketRabbitHole {label}] {' '.join(fields)}", file=sys.stderr, flush=True)
