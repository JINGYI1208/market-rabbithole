import json

import pytest

from rabbithole.research_history import (
    BROWSER_HISTORY_MAX_BYTES,
    BROWSER_HISTORY_MAX_RECORDS,
    add_failed_record,
    add_record,
    continuation_context,
    delete_record,
    export_json,
    export_markdown,
    filter_records,
    get_record,
    import_json,
    merge_histories,
    restore_browser_history,
    serialize_browser_history,
)


def result(partial=False):
    return {
        "question": "What changed in the chip market?",
        "brief": {
            "current_view": "Demand improved, but valuation remains uncertain [W1].",
            "evidence_strength": "MEDIUM",
            "partial_fallback": partial,
            "conclusions": [{"statement": "Demand improved.", "citations": ["W1"], "uncertainty": "Valuation", "support_status": "SUPPORTED"}],
        },
        "evidence": [{"citation_id": "W1", "title": "Quarterly results", "source_type": "WEB", "locator": "https://example.com/results", "excerpt": "private body must not enter continuation context"}],
        "trace": [{"event_type": "evidence_gap_detected", "detail": "Current valuation evidence is missing."}],
        "stop_reason": "agent judged evidence sufficient",
    }


def test_history_add_filter_open_and_status():
    first, second = result(), result(partial=True)
    second["question"] = "Private channel checks"
    history = []
    completed = add_record(history, first)
    partial = add_record(history, second)

    assert completed["status"] == "Completed"
    assert partial["status"] == "Partial"
    assert filter_records(history, "channel", "Partial") == [partial]
    assert filter_records(history, "missing") == []
    assert get_record(history, completed["id"])["result"] == first
    assert get_record(history, "missing") is None


def test_markdown_and_json_round_trip():
    record = add_record([], result())
    restored = import_json(export_json(record))
    markdown = export_markdown(restored)

    assert restored["result"] == record["result"]
    assert "## Market View" in markdown
    assert "[W1] Quarterly results" in markdown


def test_import_rejects_non_research_json():
    with pytest.raises(ValueError, match="research record"):
        import_json(json.dumps({"hello": "world"}))


def test_continuation_is_compact_and_marks_history_unverified():
    record = add_record([], result())
    context = continuation_context(record)

    assert "claims are not current evidence" in context
    assert "Current valuation evidence is missing" in context
    assert "Quarterly results" in context
    assert "private body must not enter continuation context" not in context


def test_history_lists_are_session_isolated_by_caller():
    session_a, session_b = [], []
    add_record(session_a, result())

    assert len(session_a) == 1
    assert session_b == []


def test_failed_record_contains_safe_status_without_exception_message():
    history = []
    record = add_failed_record(history, "Why did the market move?", "ProviderError")

    assert record["status"] == "Failed"
    assert record["source_count"] == 0
    assert "ProviderError" in record["result"]["trace"][0]["detail"]


def test_browser_save_and_reconnect_restore_without_private_local_text():
    source = result()
    source["evidence"].append({
        "citation_id": "L1", "title": "Uploaded note", "source_type": "LOCAL",
        "locator": "private.pdf", "excerpt": "confidential uploaded paragraph",
        "key_claim": "confidential extracted claim", "explanation": "private analysis",
    })
    history = []
    record = add_record(history, source, parent_run_id="parent-123")

    payload, evicted = serialize_browser_history(history)
    restored = restore_browser_history(payload)

    assert evicted == 0
    assert restored[0]["id"] == record["id"]
    assert restored[0]["parent_run_id"] == "parent-123"
    assert restored[0]["result"]["brief"]["conclusions"] == source["brief"]["conclusions"]
    local = next(item for item in restored[0]["result"]["evidence"] if item["citation_id"] == "L1")
    assert local["excerpt"] == ""
    assert local["key_claim"] == ""
    assert "confidential uploaded paragraph" not in payload


def test_browser_history_corruption_and_schema_are_rejected():
    with pytest.raises(ValueError, match="corrupted"):
        restore_browser_history("not-json")
    with pytest.raises(ValueError, match="schema"):
        restore_browser_history(json.dumps({"schema_version": 999, "records": []}))


def test_browser_history_enforces_count_and_size_limits():
    history = []
    for index in range(4):
        value = result()
        value["question"] = f"Question {index}"
        add_record(history, value)

    payload, evicted = serialize_browser_history(history, max_records=2, max_bytes=100_000)
    restored = restore_browser_history(payload)

    assert len(restored) == 2
    assert evicted == 2

    size_limited_payload, size_evicted = serialize_browser_history(history, max_records=20, max_bytes=1_200)
    assert len(size_limited_payload.encode("utf-8")) <= 1_200
    assert size_evicted > 0


def test_browser_history_default_limits_are_20_records_and_2_mb():
    history = []
    for index in range(BROWSER_HISTORY_MAX_RECORDS + 1):
        value = result()
        value["question"] = f"Question {index}"
        add_record(history, value)

    payload, evicted = serialize_browser_history(history)

    assert BROWSER_HISTORY_MAX_RECORDS == 20
    assert BROWSER_HISTORY_MAX_BYTES == 2_000_000
    assert len(restore_browser_history(payload)) == 20
    assert len(payload.encode("utf-8")) <= BROWSER_HISTORY_MAX_BYTES
    assert evicted == 1


def test_restored_records_support_open_delete_export_and_continue():
    history = []
    record = add_record(history, result())
    payload, _ = serialize_browser_history(history)
    restored = restore_browser_history(payload)
    merged = merge_histories([], restored)
    reopened = get_record(merged, record["id"])

    assert "Market View" in export_markdown(reopened)
    assert json.loads(export_json(reopened))["id"] == record["id"]
    assert "previous_question" in continuation_context(reopened)
    assert delete_record(merged, record["id"]) == []
