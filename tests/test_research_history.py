import json

import pytest

from rabbithole.research_history import (
    add_failed_record,
    add_record,
    continuation_context,
    export_json,
    export_markdown,
    filter_records,
    get_record,
    import_json,
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
