"""Streamlit portfolio UI for recorded demos and bounded live research."""

from __future__ import annotations

import hmac
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from rabbithole.config import load_app_config  # noqa: E402
from rabbithole.market_agent import AgentLimits, build_market_agent  # noqa: E402
from rabbithole.semantic_rag import load_uploaded_chunks  # noqa: E402


st.set_page_config(page_title="Market RabbitHole", page_icon="🐇", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem}
.hero {border-bottom: 3px solid #191714; padding-bottom: 1rem; margin-bottom: 1rem}
.hero h1 {font-size: 3.8rem; letter-spacing: -.06em; margin: 0}
.pill {display:inline-block; border:1px solid #191714; border-radius:99px; padding:.15rem .55rem; margin:.15rem; font-size:.8rem}
.event {border-left:4px solid #6579ff; padding:.5rem .8rem; margin:.4rem 0; background:#f7f7fb}
.conflict {border-left:5px solid #ef5b3f; padding:.7rem 1rem; background:#fff1ed}
</style>
<div class="hero"><div class="pill">SINGLE RESEARCH AGENT</div><div class="pill">HYBRID RAG</div><div class="pill">TRACEABLE EVIDENCE</div><h1>Market RabbitHole</h1><p>Evidence-grounded market research that plans, retrieves, challenges, and cites.</p></div>
""", unsafe_allow_html=True)


def _load_server_secrets() -> None:
    for name in ("ZHIPUAI_API_KEY", "TAVILY_API_KEY", "ZHIPU_MODEL", "RABBITHOLE_LIVE_ENABLED", "RABBITHOLE_LIVE_PASSWORD", "RABBITHOLE_RETRIEVAL_BACKEND"):
        try:
            value = st.secrets.get(name)
        except Exception:
            value = None
        if value is not None:
            os.environ.setdefault(name, str(value))


load_dotenv()
_load_server_secrets()
CONFIG = load_app_config()
RUN_GATE = threading.BoundedSemaphore(CONFIG.max_concurrency)


@st.cache_data
def load_demo_cases() -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((ROOT / "demo_cases").glob("*.json"))]


@st.cache_resource
def cached_agent(model: str, limits_json: str):
    limits = AgentLimits(**json.loads(limits_json))
    return build_market_agent(ROOT / "sample_corpus", model=model, limits=limits)


def render_research(result: dict[str, Any], *, fixture_label: str = "") -> None:
    if fixture_label:
        st.info(fixture_label)
    brief = result["brief"]
    evidence = result.get("evidence", [])
    local_count = sum(item["source_type"] == "LOCAL" for item in evidence)
    web_count = sum(item["source_type"] == "WEB" for item in evidence)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evidence strength", brief["evidence_strength"])
    c2.metric("Local evidence", local_count)
    c3.metric("Web evidence", web_count)
    c4.metric("Agent iterations", result.get("iterations", 0))

    st.subheader("Market view")
    st.write(brief["current_view"])
    if brief.get("conclusions"):
        st.markdown("#### Verified conclusions")
        for conclusion in brief["conclusions"]:
            mark = "✓" if conclusion.get("citation_verified") else "⚠"
            citations = " ".join(f"[{value}]" for value in conclusion.get("citations", []))
            st.markdown(f"- {mark} **{conclusion['statement']}** {citations}")
            if conclusion.get("uncertainty"):
                st.caption(f"Uncertainty: {conclusion['uncertainty']}")

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Evidence Explorer")
        for item in evidence:
            title = f"[{item['citation_id']}] {item['title']} · {item['stance']} · {item.get('evidence_kind', 'UNCLASSIFIED')}"
            with st.expander(title):
                st.write(item.get("key_claim") or item["excerpt"])
                st.caption(item.get("explanation", ""))
                st.code(item["excerpt"], language=None)
                if item["source_type"] == "WEB" and item.get("locator"):
                    st.link_button("Open source", item["locator"])
                else:
                    st.caption(f"Local source: {item['source']} · {item.get('date', '')}")
    with right:
        st.subheader("Conflict Radar")
        if not result.get("conflicts"):
            st.success("No structured contradiction was detected.")
        for conflict in result.get("conflicts", []):
            st.markdown(f"<div class='conflict'><b>{conflict['angle']}</b><br>{conflict.get('explanation','')}</div>", unsafe_allow_html=True)
            st.markdown("**Supporting**")
            for citation, claim in zip(conflict.get("support_citations", []), conflict["support"]):
                st.write(f"[{citation}] {claim}")
            st.markdown("**Opposing**")
            for citation, claim in zip(conflict.get("contradict_citations", []), conflict["contradict"]):
                st.write(f"[{citation}] {claim}")
        st.subheader("What to monitor")
        for item in brief.get("monitor_next", []):
            st.write(f"• {item}")

    st.subheader("Recorded agent execution")
    for event in result.get("trace", []):
        tool = f" · `{event['tool']}`" if event.get("tool") else ""
        st.markdown(f"<div class='event'><b>{event['event_type'].replace('_',' ').title()}</b>{tool}<br>{event['detail']}</div>", unsafe_allow_html=True)
    st.caption(f"Stop reason: {result.get('stop_reason', 'not recorded')}")
    for warning in result.get("warnings", []):
        st.warning(warning)


mode = st.sidebar.radio("Mode", ["Public Demo", "Live Agent"])

if mode == "Public Demo":
    st.subheader("Recorded research cases")
    st.write("Explore reproducible fixture-based runs produced by the real agent loop. No external API calls are made in this mode.")
    cases = load_demo_cases()
    if not cases:
        st.error("Demo fixtures are unavailable. Run scripts/generate_demo_fixtures.py.")
        st.stop()
    selected = st.selectbox("Research case", cases, format_func=lambda case: case["title"])
    st.caption(selected["description"])
    render_research(selected["result"], fixture_label="Fixture-based recorded run — generated by the bounded agent with deterministic test providers.")
    with st.expander("How the architecture works"):
        st.code("Question → LLM plan → bounded tool loop → hybrid local RAG / Tavily → evidence analysis → conflict detection → citation verification → brief", language=None)
        st.write("The controller chooses one validated action per iteration. Programmatic budgets, citation checks, upload isolation, and stop guards constrain execution.")
else:
    st.subheader("Live Agent")
    if not CONFIG.live_agent_enabled:
        st.warning("Live Agent is disabled on this deployment. Set RABBITHOLE_LIVE_ENABLED=1 server-side to enable it.")
        st.stop()
    if CONFIG.live_agent_password:
        supplied = st.text_input("Access password", type="password")
        if not supplied or not hmac.compare_digest(supplied, CONFIG.live_agent_password):
            st.info("Enter the configured server-side password to continue.")
            st.stop()
    if not os.getenv("ZHIPUAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        st.error("Live providers are not configured. Add ZHIPUAI_API_KEY and TAVILY_API_KEY as server-side secrets.")
        st.stop()

    model = st.selectbox("Configured model", CONFIG.models, index=CONFIG.models.index(CONFIG.default_model))
    if CONFIG.retrieval_backend == "tfidf":
        st.caption("Local retrieval: low-memory TF-IDF lexical ranking (hosted mode).")
    else:
        st.caption("Local retrieval: MiniLM dense embeddings with lexical score fusion.")
    question = st.text_area("Market question", height=120, placeholder="What has changed, what evidence conflicts, and what should be monitored next?")
    upload = st.file_uploader("Optional private source", type=["pdf", "md", "txt"], accept_multiple_files=False)
    if st.button("Start bounded research", type="primary", disabled=not question.strip()):
        payloads: list[tuple[str, bytes]] = []
        if upload:
            content = upload.getvalue()
            if len(content) > CONFIG.max_upload_mb * 1024 * 1024:
                st.error(f"Upload exceeds the {CONFIG.max_upload_mb} MB server limit.")
                st.stop()
            payloads.append((Path(upload.name).name, content))
        if not RUN_GATE.acquire(blocking=False):
            st.error("The live agent is busy. Please retry after the current bounded run finishes.")
            st.stop()
        status = st.status("Research agent running…", expanded=True)
        event_box = status.empty()
        observed: list[str] = []

        def show_event(event) -> None:
            observed.append(f"{event.step}: {event.detail}")
            event_box.markdown("\n\n".join(f"- {line}" for line in observed[-8:]))

        try:
            limits_json = json.dumps({name: getattr(CONFIG.limits, name) for name in CONFIG.limits.__dataclass_fields__})
            agent = cached_agent(model, limits_json)
            chunks = load_uploaded_chunks(payloads) if payloads else []
            result = agent.research(question, chunks, on_event=show_event)
            status.update(label="Research complete", state="complete", expanded=False)
            render_research(result.to_dict())
        except Exception as exc:
            status.update(label="Research failed safely", state="error")
            st.error(f"Live research could not complete: {type(exc).__name__}. Check provider configuration or rate limits.")
        finally:
            RUN_GATE.release()
