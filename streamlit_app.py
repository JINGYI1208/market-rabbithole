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
from rabbithole.browser_storage import mount_browser_storage  # noqa: E402
from rabbithole.market_agent import AgentLimits, build_market_agent  # noqa: E402
from rabbithole.providers import log_safe_exception  # noqa: E402
from rabbithole.research_history import (  # noqa: E402
    add_failed_record,
    add_record,
    continuation_context,
    export_json,
    export_markdown,
    filter_records,
    import_json,
    merge_histories,
    restore_browser_history,
    serialize_browser_history,
    delete_record,
)
from rabbithole.semantic_rag import load_uploaded_chunks  # noqa: E402


st.set_page_config(page_title="Market RabbitHole", page_icon="🐇", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem}
.hero {border-bottom: 3px solid #191714; padding-bottom: 1rem; margin-bottom: 1rem}
.hero h1 {font-size: 3.8rem; letter-spacing: -.06em; margin: 0}
.pill {display:inline-block; border:1px solid #191714; border-radius:99px; padding:.15rem .55rem; margin:.15rem; font-size:.8rem}
.event {border-left:4px solid #6579ff; padding:.5rem .8rem; margin:.4rem 0; background:#f7f7fb}
.event-decision {border-left-color:#6579ff}.event-tool {border-left-color:#1f9d77}
.event-recovery {border-left-color:#ef9b3f}.event-gap {border-left-color:#9a62d6}
.event-stop {border-left-color:#191714}
.conflict {border-left:5px solid #ef5b3f; padding:.7rem 1rem; background:#fff1ed}
.library-card {border:1px solid #ddd8cf; border-radius:10px; padding:1rem; margin:.75rem 0; background:#fff}
</style>
<div class="hero"><div class="pill">SINGLE RESEARCH AGENT</div><div class="pill">EVIDENCE-AWARE TOOL USE</div><div class="pill">TRACEABLE SYNTHESIS</div><h1>Market RabbitHole</h1><p>Ask a market question. The agent plans research, chooses between private context and live search, challenges conflicting evidence, and produces a cited brief.</p></div>
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
    if brief.get("evidence_strength_reason"):
        st.caption(f"Evidence-strength basis: {brief['evidence_strength_reason']}")

    st.subheader("Market view")
    st.write(brief["current_view"])
    if brief.get("conclusions"):
        st.markdown("#### Source-linked conclusions")
        for conclusion in brief["conclusions"]:
            status = conclusion.get("support_status", "UNVERIFIED")
            mark = "✓" if status == "SUPPORTED" else "⚠"
            citations = " ".join(f"[{value}]" for value in conclusion.get("citations", []))
            st.markdown(f"- {mark} **{conclusion['statement']}** {citations} · {status.title()} passage support")
            if conclusion.get("uncertainty"):
                st.caption(f"Uncertainty: {conclusion['uncertainty']}")

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Evidence Explorer")
        for item in evidence:
            title = f"[{item['citation_id']}] {item['title']} · {item['stance']} · {item.get('evidence_kind', 'UNCLASSIFIED')}"
            with st.expander(title):
                st.write(item.get("key_claim") or item.get("excerpt") or "Evidence text is not available in this restored browser record.")
                st.caption(item.get("explanation", ""))
                if item.get("excerpt"):
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

    st.subheader("Agent execution summary")
    st.caption("Operational decisions and tool outcomes only; hidden model reasoning is not displayed.")
    for event in result.get("trace", []):
        event_type = event["event_type"]
        if event_type in {"agent_decision", "decision_retry_accepted"}:
            category, css = "LLM decision", "decision"
        elif event_type in {"fallback_decision", "decision_recovered", "decision_rejected"}:
            category, css = "Deterministic recovery", "recovery"
        elif event_type == "evidence_gap_detected":
            category, css = "Evidence gap", "gap"
        elif event_type in {"research_stopped", "stop_rejected", "tool_budget_exhausted"}:
            category, css = "Research termination", "stop"
        else:
            category, css = "Tool execution", "tool"
        tool = f" · `{event['tool']}`" if event.get("tool") else ""
        st.markdown(f"<div class='event event-{css}'><b>{category}</b> · {event_type.replace('_',' ').title()}{tool}<br>{event['detail']}</div>", unsafe_allow_html=True)
    st.caption(f"Stop reason: {result.get('stop_reason', 'not recorded')}")
    for warning in result.get("warnings", []):
        st.warning(warning)


def private_access() -> bool:
    if not CONFIG.live_agent_enabled:
        st.warning("Live Research and Research Library are disabled on this deployment. Set RABBITHOLE_LIVE_ENABLED=1 server-side to enable them.")
        return False
    if not CONFIG.live_agent_password:
        return True
    if st.session_state.get("private_access_granted"):
        return True
    supplied = st.text_input("Access password", type="password", key="workspace_password")
    if supplied and hmac.compare_digest(supplied, CONFIG.live_agent_password):
        st.session_state.private_access_granted = True
        st.rerun()
    st.info("Enter the configured server-side password to access live research and this session's library.")
    return False


def render_demo_gallery() -> None:
    st.subheader("Demo Gallery")
    st.write("Browse fixture-based recorded runs produced by the real bounded agent loop. This page makes zero external API calls.")
    cases = load_demo_cases()
    if not cases:
        st.error("Demo fixtures are unavailable. Run scripts/generate_demo_fixtures.py.")
        return
    selected = st.selectbox("Research case", cases, format_func=lambda case: case["title"])
    st.caption(selected["description"])
    render_research(selected["result"], fixture_label="Fixture-based recorded run — generated by the bounded agent with deterministic test providers.")
    with st.expander("Why this is agentic"):
        st.code("Question → LLM plan → evidence-aware tool choice → local retrieval / Tavily → evidence analysis → conflict detection → citation support checks → brief", language=None)
        st.write("The model selects one action per iteration. Programmatic budgets, marginal-gain checks, source filters, recovery, and stopping guards constrain execution without turning it into a fixed pipeline.")


def render_research_workspace() -> None:
    st.subheader("Research Workspace")
    st.write("Run a bounded agent that selects tools from the evidence state, investigates gaps and conflicts, and produces a source-linked market brief.")
    if not private_access():
        return
    if not os.getenv("ZHIPUAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        st.error("Live providers are not configured. Add ZHIPUAI_API_KEY and TAVILY_API_KEY as server-side secrets.")
        return
    model = st.selectbox("Configured model", CONFIG.models, index=CONFIG.models.index(CONFIG.default_model))
    st.caption("Local retrieval: low-memory TF-IDF lexical ranking (hosted mode)." if CONFIG.retrieval_backend == "tfidf" else "Local retrieval: MiniLM dense embeddings with lexical score fusion.")
    seed = st.session_state.pop("continuation_seed", "")
    if seed:
        st.info("Continuing a prior report. Historical claims are unverified background; the agent must retrieve fresh evidence before relying on them.")
        st.session_state["research_question"] = seed
    question = st.text_area("Market question", height=120, key="research_question", placeholder="What has changed, what evidence conflicts, and what should be monitored next?")
    upload = st.file_uploader("Optional private source", type=["pdf", "md", "txt"], accept_multiple_files=False)
    if not st.button("Start bounded research", type="primary", disabled=not question.strip()):
        return
    payloads: list[tuple[str, bytes]] = []
    if upload:
        content = upload.getvalue()
        if len(content) > CONFIG.max_upload_mb * 1024 * 1024:
            st.error(f"Upload exceeds the {CONFIG.max_upload_mb} MB server limit.")
            return
        payloads.append((Path(upload.name).name, content))
    if not RUN_GATE.acquire(blocking=False):
        st.error("The live agent is busy. Please retry after the current bounded run finishes.")
        return
    status = st.status("Research agent running…", expanded=True)
    event_box = status.empty()
    observed: list[str] = []

    def show_event(event) -> None:
        observed.append(f"{event.step}: {event.detail}")
        event_box.markdown("\n\n".join(f"- {line}" for line in observed[-8:]))

    stage = "initialize retrieval and providers"
    try:
        limits_json = json.dumps({name: getattr(CONFIG.limits, name) for name in CONFIG.limits.__dataclass_fields__})
        agent = cached_agent(model, limits_json)
        stage = "parse request-scoped uploads"
        chunks = load_uploaded_chunks(payloads) if payloads else []
        stage = "run bounded research agent"
        result = agent.research(question, chunks, on_event=show_event, historical_context=st.session_state.pop("continuation_context", ""))
        result_dict = result.to_dict()
        parent_run_id = st.session_state.pop("continuation_parent_id", None)
        add_record(st.session_state.research_history, result_dict, parent_run_id=parent_run_id)
        stage = "render research result"
        status.update(label="Research complete — saved to this session's library", state="complete", expanded=False)
        render_research(result_dict)
    except Exception as exc:
        log_safe_exception("LiveAgent", stage, exc)
        parent_run_id = st.session_state.pop("continuation_parent_id", None)
        add_failed_record(st.session_state.research_history, question, type(exc).__name__, parent_run_id=parent_run_id)
        status.update(label="Research failed safely", state="error")
        st.error(f"Live research could not complete: {type(exc).__name__}. Check provider configuration or rate limits.")
    finally:
        RUN_GATE.release()


def begin_continuation(record: dict[str, Any]) -> None:
    st.session_state.continuation_context = continuation_context(record)
    st.session_state.continuation_seed = f"Follow up on: {record['question']}"
    st.session_state.continuation_parent_id = record["id"]
    st.session_state.navigation = "Research"


def render_library() -> None:
    st.subheader("Research Library")
    if not private_access():
        return
    st.caption("Saved in this browser's local storage so records can survive reconnects and server restarts. This is not encrypted secure storage; anyone using this browser profile may access it. Local/uploaded source text is excluded. Clear it below or through your browser site-data settings.")
    if st.session_state.get("browser_storage_error"):
        st.warning(f"Browser history is unavailable ({st.session_state['browser_storage_error']}). Export important reports manually.")
    with st.expander("Import an exported research record"):
        imported = st.file_uploader("Market RabbitHole JSON", type=["json"], key="history_import")
        if st.button("Import record", disabled=imported is None):
            try:
                st.session_state.research_history.insert(0, import_json(imported.getvalue()))
                st.success("Research record imported into this session.")
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                st.error("That file is not a valid Market RabbitHole research export.")
    search_col, status_col = st.columns([2, 1])
    query = search_col.text_input("Search history", placeholder="Question or summary")
    status_filter = status_col.selectbox("Run status", ["All", "Completed", "Partial", "Failed"])
    records = filter_records(st.session_state.research_history, query, status_filter)
    if not records:
        st.info("No matching research in this active session.")
        return
    for record in records:
        st.markdown(f"<div class='library-card'><b>{record['title']}</b><br><small>{record['created_at']} · {record['status']} · {record['evidence_strength']} evidence · {record['source_count']} sources</small><br>{record['summary']}</div>", unsafe_allow_html=True)
        open_col, continue_col, delete_col = st.columns(3)
        if open_col.button("Open report", key=f"open-{record['id']}"):
            st.session_state.open_record_id = record["id"]
        continue_col.button("Continue Research", key=f"continue-{record['id']}", on_click=begin_continuation, args=(record,))
        if delete_col.button("Delete record", key=f"delete-{record['id']}"):
            st.session_state.research_history = delete_record(st.session_state.research_history, record["id"])
            st.rerun()
        export_col, json_col = st.columns(2)
        export_col.download_button("Export Markdown", export_markdown(record), file_name=f"market-rabbithole-{record['id'][:8]}.md", mime="text/markdown", key=f"md-{record['id']}")
        json_col.download_button("Export JSON", export_json(record), file_name=f"market-rabbithole-{record['id'][:8]}.json", mime="application/json", key=f"json-{record['id']}")
        if st.session_state.get("open_record_id") == record["id"]:
            with st.container(border=True):
                render_research(record["result"])
    if st.button("Clear browser research history", type="secondary"):
        st.session_state.research_history = []
        st.session_state.open_record_id = ""
        st.session_state.browser_storage_clear = True
        st.rerun()


if "research_history" not in st.session_state:
    st.session_state.research_history = []
if "browser_history_loaded" not in st.session_state:
    st.session_state.browser_history_loaded = False

storage_authorized = CONFIG.live_agent_enabled and (
    not CONFIG.live_agent_password or st.session_state.get("private_access_granted", False)
)
if storage_authorized:
    browser_read = mount_browser_storage("read", key="browser-history-reader")
    if not st.session_state.browser_history_loaded and browser_read.ready:
        if browser_read.error:
            st.session_state.browser_storage_error = browser_read.error
        elif browser_read.payload:
            try:
                restored = restore_browser_history(browser_read.payload)
                st.session_state.research_history = merge_histories(st.session_state.research_history, restored)
            except ValueError as exc:
                st.session_state.browser_storage_error = str(exc)
                st.session_state.browser_storage_clear = True
        st.session_state.browser_history_loaded = True
    if st.session_state.get("browser_storage_error"):
        st.sidebar.warning("Browser research history could not be saved or restored. Open Research Library for details and export important reports manually.")

mode = st.sidebar.radio("Workspace", ["Research", "Research Library", "Demo Gallery"], key="navigation")
st.sidebar.caption("Demo Gallery is public and offline. Research and Library use the server-side access gate.")

if mode == "Demo Gallery":
    render_demo_gallery()
elif mode == "Research Library":
    render_library()
else:
    render_research_workspace()

if storage_authorized and st.session_state.browser_history_loaded:
    try:
        browser_payload, evicted = serialize_browser_history(st.session_state.research_history)
        browser_mode = "clear" if st.session_state.pop("browser_storage_clear", False) else "write"
        browser_write = mount_browser_storage(browser_mode, browser_payload, key="browser-history-writer")
        if browser_write.error:
            st.session_state.browser_storage_error = browser_write.error
        if evicted and st.session_state.get("last_eviction_count") != evicted:
            st.session_state.last_eviction_count = evicted
            st.warning(f"Browser history limit reached; {evicted} oldest record(s) were not retained locally.")
    except ValueError as exc:
        st.session_state.browser_storage_error = str(exc)
