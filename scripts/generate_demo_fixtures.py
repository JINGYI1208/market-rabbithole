"""Generate zero-API public demo recordings through the real agent loop."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rabbithole.market_agent import AgentLimits, MarketRabbitHoleAgent  # noqa: E402
from rabbithole.semantic_rag import LocalChunk, SemanticRAG  # noqa: E402


CASES = [
    {
        "slug": "ai_semiconductors",
        "title": "AI semiconductor fundamentals vs valuation expectations",
        "description": "Tests whether strong demand evidence is already reflected in valuation and expectations.",
        "question": "Are AI semiconductor fundamentals strong enough to justify current valuation expectations over the next 6–12 months?",
        "local": [
            "Accelerator demand and hyperscaler capital expenditure remain strong, supporting continued revenue growth for leading AI chip suppliers.",
            "However, premium valuation multiples assume sustained growth and leave equities exposed if deployment or monetization slows.",
            "Advanced packaging capacity is improving, but supply-chain concentration remains a material execution risk.",
        ],
        "web": [
            "Recent company guidance indicates continued data-center demand, while management describes the outlook rather than a guaranteed result.",
            "A valuation review notes that consensus expectations already embed rapid earnings growth and could reset on weaker forward guidance.",
        ],
    },
    {
        "slug": "copper_balance",
        "title": "Copper supply and demand balance",
        "description": "Examines structural electrification demand against near-term inventory and macro risks.",
        "question": "Is the copper market entering a durable deficit, and which evidence could invalidate that view?",
        "local": [
            "Mine disruptions and declining ore grades constrain medium-term copper supply growth.",
            "Grid investment and electrification forecasts support structural copper demand, subject to project execution.",
            "However, visible inventories and weak construction demand can soften the near-term balance.",
        ],
        "web": [
            "Current exchange inventory data show a larger buffer than the structural deficit narrative implies.",
            "New mine approvals could improve future supply, but commissioning schedules remain uncertain forecasts.",
        ],
    },
    {
        "slug": "private_public_conflict",
        "title": "Conflicting private and public market evidence",
        "description": "Shows how private channel checks can conflict with current public reporting without hiding the disagreement.",
        "question": "How should an analyst reconcile weakening private demand checks with optimistic public company guidance?",
        "local": [
            "Private distributor checks report slower order growth and elevated customer inventories during the current quarter.",
            "The channel sample is limited and may overrepresent smaller customers with tighter financing conditions.",
            "Internal base-case research expects a cautious demand recovery rather than immediate acceleration.",
        ],
        "web": [
            "Public company guidance forecasts stronger second-half demand and cites a growing pipeline of customer projects.",
            "A subsequent industry survey reports mixed end-market conditions, leaving the timing of recovery unresolved.",
        ],
    },
]


class FixtureEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = np.zeros(64, dtype=np.float32)
            for term in text.lower().split():
                bucket = int.from_bytes(hashlib.sha256(term.strip(".,?-").encode()).digest()[:2], "big") % len(vector)
                vector[bucket] += 1
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.asarray(vectors)


class FixtureWeb:
    def __init__(self, slug: str, results: list[str]):
        self.slug = slug
        self.results = results
        self.calls = 0

    def search(self, query, max_results=1):
        index = min(self.calls, len(self.results) - 1)
        self.calls += 1
        return [{
            "title": f"Fixture public source {self.calls}",
            "url": f"https://example.com/{self.slug}/{self.calls}",
            "source": "example.com",
            "date": "2026-10-01",
            "content": self.results[index],
        }]


class FixtureLLM:
    def complete_json(self, system, user):
        lowered = system.lower()
        if "initial plan" in lowered:
            return {"angles": [
                {"name": "Fundamentals vs expectations", "question": "What evidence supports or challenges the base case?", "rationale": "Separates operating evidence from expectations."},
                {"name": "Timing and assumptions", "question": "Which horizon and assumptions drive disagreement?", "rationale": "Prevents false contradictions."},
                {"name": "Invalidation risk", "question": "What would invalidate the current view?", "rationale": "Defines monitoring conditions."},
            ]}
        if "bounded market research loop" in lowered:
            state = json.loads(user.split("Research state: ", 1)[1].split("\nReturn:", 1)[0])
            evidence = state["evidence"]
            web_count = sum(item["source_type"] == "WEB" for item in evidence)
            if not evidence:
                return {"action": "search_local", "query": user.split("Question: ", 1)[1].split("\n", 1)[0], "angle": "Fundamentals vs expectations", "rationale": "Establish the private research baseline.", "evidence_gap": "No private evidence has been examined."}
            if any(not item["claim"] for item in evidence):
                return {"action": "analyze_evidence", "query": "", "angle": "", "rationale": "Classify newly retrieved evidence before deciding what is missing.", "evidence_gap": ""}
            if web_count == 0:
                return {"action": "search_web", "query": "latest public evidence for the market thesis", "angle": "Fundamentals vs expectations", "rationale": "Test private context against current public evidence.", "evidence_gap": "The private view lacks current public confirmation."}
            if state["conflicts"] and web_count < 2:
                return {"action": "search_web", "query": "evidence resolving timing and assumptions behind the disagreement", "angle": "Timing and assumptions", "rationale": "Investigate the detected contradiction before stopping.", "evidence_gap": "The conflict may reflect different periods or assumptions."}
            return {"action": "stop", "query": "", "angle": "", "rationale": "The evidence is analyzed, the contradiction is explicit, and the follow-up budget has been used selectively.", "evidence_gap": ""}
        if "analyze market evidence" in lowered:
            rows = json.loads(user.split("Evidence: ", 1)[1].split("\nReturn:", 1)[0])
            analyzed = []
            for row in rows:
                text = row["excerpt"]
                opposing = any(marker in text.lower() for marker in ("however", "weaker", "slower", "larger buffer", "premium valuation"))
                forecast = any(marker in text.lower() for marker in ("forecast", "guidance", "expects", "outlook"))
                analyzed.append({
                    "id": row["id"],
                    "research_angle": "Fundamentals vs expectations",
                    "stance": "CONTRADICT" if opposing else "SUPPORT",
                    "kind": "FORECAST" if forecast else "FACT",
                    "key_claim": text,
                    "explanation": "Fixture classification grounded in the recorded passage.",
                    "time_period": "current or forward",
                    "assumptions": ["Other market drivers remain broadly stable"] if forecast else [],
                })
            return {"evidence": analyzed}
        evidence = json.loads(user.split("Evidence: ", 1)[1].split("\nDetected conflicts:", 1)[0])
        supports = [item for item in evidence if item["stance"] == "SUPPORT"]
        opposes = [item for item in evidence if item["stance"] == "CONTRADICT"]
        conclusions = []
        if supports:
            conclusions.append({"statement": supports[0]["claim"], "citations": [supports[0]["citation"]], "uncertainty": "The conclusion depends on the stated horizon."})
        if opposes:
            conclusions.append({"statement": opposes[0]["claim"], "citations": [opposes[0]["citation"]], "uncertainty": "The opposing signal may use different assumptions."})
        cited = " ".join(f"{item['claim']} [{item['citation']}]" for item in evidence[:2])
        return {
            "current_view": cited,
            "conclusions": conclusions,
            "key_drivers": [f"↔ {item['claim']} [{item['citation']}]" for item in evidence[:3]],
            "conflicting_signals": ["Supporting and opposing evidence coexist; timing and assumptions remain unresolved."],
            "evidence_strength": "MEDIUM",
            "monitor_next": ["Updated company guidance", "Inventory or order data", "Evidence that resolves the stated assumptions"],
        }


def main() -> None:
    output = ROOT / "demo_cases"
    output.mkdir(exist_ok=True)
    for case in CASES:
        chunks = [LocalChunk(f"{case['slug']}-{index}", case["title"], text, f"fixture://{case['slug']}/private") for index, text in enumerate(case["local"])]
        agent = MarketRabbitHoleAgent(
            SemanticRAG(chunks, FixtureEncoder()),
            FixtureLLM(),
            FixtureWeb(case["slug"], case["web"]),
            limits=AgentLimits(max_iterations=9, max_local_searches=2, max_web_searches=2, local_top_k=3, web_results=1, min_evidence=3),
        )
        result = agent.research(case["question"])
        payload = {
            "title": case["title"],
            "description": case["description"],
            "fixture_based": True,
            "generated_by": "scripts/generate_demo_fixtures.py",
            "result": result.to_dict(),
        }
        (output / f"{case['slug']}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"generated {case['slug']}: {len(result.evidence)} evidence, {result.iterations} iterations")


if __name__ == "__main__":
    main()
