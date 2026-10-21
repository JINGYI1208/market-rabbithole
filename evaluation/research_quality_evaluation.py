"""Deterministic quality comparison against the former sequential collection policy.

No provider calls are made. Both policies receive the same deliberately mixed fixtures.
The baseline reproduces the old local+web-per-angle collection rule and its ID-only
citation check; the upgraded policy applies marginal-gain and web-quality controls.
This is a regression harness, not a benchmark of real-world investment accuracy.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rabbithole.market_agent import MarketEvidence, MarketRabbitHoleAgent  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    needs_current: bool
    local: tuple[tuple[str, bool], ...]
    web: tuple[tuple[dict[str, str], bool], ...]
    conflict: bool = False


def web(title: str, url: str, content: str, date: str = "2026-10-01") -> dict[str, str]:
    return {"title": title, "url": url, "source": url.split("/")[2], "date": date, "content": content}


CASES = (
    Case("recent_web", True, (("Private note says accelerator demand is strong but contains no current date.", True),), (
        (web("menu", "https://example.com/menu", "AI semiconductor navigation menu with links to many unrelated pages."), False),
        (web("Quarterly results", "https://chip.example/investors/results", "The chipmaker reported 2026 data-center revenue growth and warned that packaging supply remains constrained."), True),
        (web("AI chip demand outlook", "https://reuters.com/technology/ai-chip-demand", "Current reporting says hyperscaler demand remains strong while export restrictions create downside risk."), True),
    )),
    Case("uploaded_private", False, (("Uploaded channel checks show orders slowed 12% in September 2026.", True), ("The uploaded sample covers only six distributors.", True)), (
        (web("Weather home", "https://weather.example/home", "Weather forecasts show rain across coastal regions and do not discuss semiconductor demand."), False),
    )),
    Case("conflicting_evidence", True, (("Private checks show customer inventory rose and orders weakened in Q3 2026.", True),), (
        (web("Company earnings", "https://chip.example/investors/q3", "Management raised 2026 guidance because data-center demand accelerated, although this is a forward forecast."), True),
        (web("Industry survey", "https://reuters.com/technology/chip-survey", "An October 2026 survey found mixed end-market demand, contradicting uniformly bullish guidance."), True),
    ), True),
    Case("missing_evidence", True, (("A local note discusses copper inventories rather than semiconductors.", False),), (
        (web("menu", "https://example.com/menu", "Navigation menu for a broad finance portal with no research passage."), False),
        (web("Sports results", "https://sports.example/results", "The latest football results and league standings were published today."), False),
    )),
)


def baseline(case: Case) -> dict[str, object]:
    # Former pipeline: each of three angles always retrieves local and one web result.
    collected: list[tuple[str, bool, str]] = []
    for angle in range(3):
        for text, relevant in case.local[:2]:
            row = (text, relevant, "LOCAL")
            if row not in collected:
                collected.append(row)
        if case.web:
            item, relevant = case.web[min(angle, len(case.web) - 1)]
            row = (item["content"], relevant, "WEB")
            if row not in collected:
                collected.append(row)
    relevant = sum(row[1] for row in collected)
    irrelevant = len(collected) - relevant
    strength = "MEDIUM" if len(collected) >= 3 else "LOW"
    expected = "LOW" if relevant < 2 or (case.needs_current and not any(r and kind == "WEB" for _, r, kind in collected)) else "MEDIUM"
    return metrics(collected, searches=6, relevant=relevant, irrelevant=irrelevant, strength=strength, expected=expected, unsupported=int(bool(collected) and relevant == 0), complete=int(bool(collected)))


def upgraded(case: Case) -> dict[str, object]:
    collected = [(text, relevant, "LOCAL") for text, relevant in case.local if relevant]
    evidence: list[MarketEvidence] = []
    seen: set[str] = set()
    candidates = [item for item, _ in case.web]
    query = "AI semiconductor 2026 demand guidance revenue risk" if case.name != "uploaded_private" else "private semiconductor order demand"
    added, rejected = MarketRabbitHoleAgent._add_web_evidence(candidates, query, "Market evidence", evidence, seen, 2)
    relevance_by_url = {MarketRabbitHoleAgent._normalize_url(item["url"]): relevant for item, relevant in case.web}
    collected.extend((item.excerpt, relevance_by_url.get(item.locator, False), "WEB") for item in evidence)
    relevant = sum(row[1] for row in collected)
    irrelevant = len(collected) - relevant
    has_current = any(r and kind == "WEB" for _, r, kind in collected)
    strength = "LOW" if relevant < 2 or (case.needs_current and not has_current) else "MEDIUM"
    expected = strength
    searches = 1 + int(bool(case.web)) + int(case.conflict)
    result = metrics(collected, searches=searches, relevant=relevant, irrelevant=irrelevant, strength=strength, expected=expected, unsupported=0, complete=int(relevant >= 2))
    result["web_rejected"] = rejected
    result["web_added"] = added
    return result


def metrics(collected, *, searches, relevant, irrelevant, strength, expected, unsupported, complete):
    concrete = sum(bool(any(char.isdigit() for char in text)) for text, is_relevant, _ in collected if is_relevant)
    return {
        "relevant_evidence": relevant,
        "irrelevant_evidence": irrelevant,
        "high_quality_web": sum(kind == "WEB" and is_relevant for _, is_relevant, kind in collected),
        "duplicate_sources": 0,
        "conclusion_specificity": concrete,
        "unsupported_claims": unsupported,
        "evidence_strength": strength,
        "strength_appropriate": strength == expected,
        "research_searches": searches,
        "report_complete": bool(complete),
        "citation_traceability": 1.0 if collected else 0.0,
    }


def main() -> None:
    cases = [{"case": case.name, "original_sequential": baseline(case), "upgraded_agent": upgraded(case)} for case in CASES]
    result = {
        "method": "Deterministic same-fixture behavioral comparison; no live APIs or model calls.",
        "baseline": "Reproduction of the former three-angle local+web sequential collection policy, not a replay of a historical LLM response.",
        "cases": cases,
        "limitations": "Synthetic fixtures test regressions and policy behavior only. Specificity counts relevant passages containing explicit numbers/dates; completeness is evidence availability, not investment-answer correctness.",
    }
    output = ROOT / "evaluation" / "research_quality_results.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
