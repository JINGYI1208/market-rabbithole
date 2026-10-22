"""Research records, privacy-safe browser serialization, and portable exports.

This module has no server persistence layer. The Streamlit app keeps full records in
session state and stores a redacted, bounded copy in the current browser.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


HISTORY_SCHEMA_VERSION = 2
BROWSER_HISTORY_MAX_RECORDS = 20
BROWSER_HISTORY_MAX_BYTES = 2_000_000


def run_status(result: dict[str, Any]) -> str:
    brief = result.get("brief", {})
    stop_reason = str(result.get("stop_reason", "")).casefold()
    if brief.get("partial_fallback") or "incomplete" in stop_reason or "iteration limit" in stop_reason:
        return "Partial"
    return "Completed"


def create_record(
    result: dict[str, Any],
    *,
    created_at: str | None = None,
    record_id: str | None = None,
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    question = " ".join(str(result.get("question", "Untitled research")).split())
    brief = result.get("brief", {})
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "id": record_id or uuid4().hex,
        "title": question[:100] or "Untitled research",
        "question": question,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": " ".join(str(brief.get("current_view", "No executive summary available.")).split())[:360],
        "evidence_strength": str(brief.get("evidence_strength", "LOW")),
        "source_count": len(result.get("evidence", [])),
        "status": run_status(result),
        "parent_run_id": parent_run_id,
        "result": result,
    }


def add_record(history: list[dict[str, Any]], result: dict[str, Any], *, parent_run_id: str | None = None) -> dict[str, Any]:
    record = create_record(result, parent_run_id=parent_run_id)
    history.insert(0, record)
    return record


def add_failed_record(history: list[dict[str, Any]], question: str, error_type: str, *, parent_run_id: str | None = None) -> dict[str, Any]:
    """Record a safe failure without exception text, prompts, or uploaded content."""
    result = {
        "question": " ".join(question.split()),
        "brief": {
            "current_view": "Research did not complete; no market conclusion was produced.",
            "evidence_strength": "LOW",
            "partial_fallback": True,
            "conclusions": [],
            "key_drivers": [],
            "conflicting_signals": ["The run failed before a grounded report could be completed."],
            "monitor_next": [],
        },
        "evidence": [],
        "conflicts": [],
        "trace": [{"event_type": "research_failed", "detail": f"Run ended safely with {error_type}.", "tool": "", "status": "error", "metadata": {}}],
        "warnings": [],
        "stop_reason": "Research failed safely.",
        "iterations": 0,
    }
    record = create_record(result, parent_run_id=parent_run_id)
    record["status"] = "Failed"
    history.insert(0, record)
    return record


def filter_records(history: list[dict[str, Any]], query: str = "", status: str = "All") -> list[dict[str, Any]]:
    needle = query.strip().casefold()
    return [
        record for record in history
        if (status == "All" or record.get("status") == status)
        and (not needle or needle in f"{record.get('title', '')} {record.get('summary', '')}".casefold())
    ]


def get_record(history: list[dict[str, Any]], record_id: str) -> dict[str, Any] | None:
    return next((record for record in history if record.get("id") == record_id), None)


def delete_record(history: list[dict[str, Any]], record_id: str) -> list[dict[str, Any]]:
    return [record for record in history if record.get("id") != record_id]


def export_json(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, indent=2)


def import_json(payload: str | bytes) -> dict[str, Any]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("result"), dict):
        raise ValueError("Export must contain one Market RabbitHole research record")
    result = parsed["result"]
    if not isinstance(result.get("brief"), dict) or not isinstance(result.get("evidence", []), list):
        raise ValueError("Exported research result is incomplete")
    record = create_record(
        result,
        created_at=str(parsed.get("created_at") or "") or None,
        record_id=str(parsed.get("id") or "") or None,
        parent_run_id=str(parsed.get("parent_run_id") or "") or None,
    )
    if parsed.get("status") in {"Completed", "Partial", "Failed"}:
        record["status"] = parsed["status"]
    return record


def browser_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    """Remove raw private/local passages before browser persistence."""
    safe = deepcopy(record)
    safe["schema_version"] = HISTORY_SCHEMA_VERSION
    for item in safe.get("result", {}).get("evidence", []):
        if item.get("source_type") == "LOCAL":
            item["excerpt"] = ""
            item["key_claim"] = ""
            item["explanation"] = "Local evidence text omitted from browser storage."
            item["browser_content_omitted"] = True
    return safe


def serialize_browser_history(
    history: list[dict[str, Any]],
    *,
    max_records: int = BROWSER_HISTORY_MAX_RECORDS,
    max_bytes: int = BROWSER_HISTORY_MAX_BYTES,
) -> tuple[str, int]:
    """Return a bounded versioned payload and the number of evicted records."""
    records = [browser_safe_record(record) for record in history[:max_records]]
    evicted = max(0, len(history) - len(records))
    while True:
        envelope = {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "saved_at": str(records[0].get("created_at", "")) if records else "",
            "records": records,
        }
        payload = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        if len(payload.encode("utf-8")) <= max_bytes:
            return payload, evicted
        if not records:
            raise ValueError("Browser history exceeds the configured storage limit")
        records.pop()
        evicted += 1


def restore_browser_history(payload: str | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    try:
        envelope = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Browser history is corrupted and could not be restored") from exc
    if not isinstance(envelope, dict) or envelope.get("schema_version") != HISTORY_SCHEMA_VERSION:
        raise ValueError("Browser history uses an unsupported schema version")
    records = envelope.get("records")
    if not isinstance(records, list):
        raise ValueError("Browser history records are malformed")
    restored: list[dict[str, Any]] = []
    for value in records[:BROWSER_HISTORY_MAX_RECORDS]:
        if not isinstance(value, dict):
            continue
        try:
            restored.append(import_json(json.dumps(value, ensure_ascii=False)))
        except ValueError:
            continue
    return restored


def merge_histories(session_records: list[dict[str, Any]], browser_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge by run ID with current-session records taking precedence."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in [*session_records, *browser_records]:
        record_id = str(record.get("id", ""))
        if not record_id or record_id in seen:
            continue
        seen.add(record_id)
        merged.append(record)
    return sorted(merged, key=lambda item: str(item.get("created_at", "")), reverse=True)[:BROWSER_HISTORY_MAX_RECORDS]


def export_markdown(record: dict[str, Any]) -> str:
    result = record["result"]
    brief = result.get("brief", {})
    lines = [
        f"# {record['title']}", "", f"**Created:** {record['created_at']}",
        f"**Status:** {record['status']}  ", f"**Evidence strength:** {record['evidence_strength']}",
        "", "## Market View", "", str(brief.get("current_view", "")), "", "## Source-linked Conclusions", "",
    ]
    for conclusion in brief.get("conclusions", []):
        citations = " ".join(f"[{item}]" for item in conclusion.get("citations", []))
        lines.append(f"- {conclusion.get('statement', '')} {citations}".rstrip())
        if conclusion.get("uncertainty"):
            lines.append(f"  - Uncertainty: {conclusion['uncertainty']}")
    lines.extend(["", "## Sources", ""])
    for item in result.get("evidence", []):
        locator = item.get("locator") or item.get("source", "")
        lines.append(f"- [{item.get('citation_id', '')}] {item.get('title', 'Untitled')} — {locator}")
    lines.extend(["", "## Research Stop", "", str(result.get("stop_reason", "Not recorded")), ""])
    return "\n".join(lines)


def continuation_context(record: dict[str, Any]) -> str:
    """Return compact background, not evidence, for a new independent run."""
    result = record["result"]
    brief = result.get("brief", {})
    conclusions = [
        {
            "statement": str(item.get("statement", ""))[:500],
            "uncertainty": str(item.get("uncertainty", ""))[:240],
            "support_status": item.get("support_status", "UNVERIFIED"),
        }
        for item in brief.get("conclusions", [])[:4]
    ]
    gaps = [
        str(event.get("detail", ""))[:300] for event in result.get("trace", [])
        if event.get("event_type") == "evidence_gap_detected"
    ][:4]
    sources = [
        {
            "citation": item.get("citation_id", ""),
            "title": str(item.get("title", ""))[:180],
            "source_type": item.get("source_type", ""),
            "locator": item.get("locator", ""),
        }
        for item in result.get("evidence", [])[:8]
    ]
    context = {
        "previous_question": str(record.get("question", ""))[:600],
        "prior_conclusions_and_uncertainties": conclusions,
        "previously_identified_evidence_gaps": gaps,
        "source_references": sources,
    }
    return "Historical research context (background only; claims are not current evidence and must be re-researched before reliance):\n" + json.dumps(context, ensure_ascii=False)
