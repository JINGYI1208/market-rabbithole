"""Small Streamlit v2 component boundary for browser-local research history."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


STORAGE_KEY = "market_rabbithole_research_history_v2"

_BROWSER_STORAGE_JS = r"""
export default function(component) {
  const { data, setStateValue } = component;
  let payload = null;
  let error = null;
  try {
    if (data.mode === "write") {
      window.localStorage.setItem(data.storageKey, data.payload);
    } else if (data.mode === "clear") {
      window.localStorage.removeItem(data.storageKey);
    }
    payload = window.localStorage.getItem(data.storageKey);
  } catch (exception) {
    error = exception && exception.name ? exception.name : "BrowserStorageError";
  }
  setStateValue("payload", payload);
  setStateValue("error", error);
  setStateValue("ready", true);
}
"""

_browser_storage_component = st.components.v2.component(
    "market_rabbithole_browser_storage",
    js=_BROWSER_STORAGE_JS,
)


@dataclass(frozen=True, slots=True)
class BrowserStorageResult:
    payload: str | None
    error: str | None
    ready: bool


def mount_browser_storage(mode: str, payload: str = "", *, key: str) -> BrowserStorageResult:
    if mode not in {"read", "write", "clear"}:
        raise ValueError("Unsupported browser storage operation")
    result = _browser_storage_component(
        data={"mode": mode, "storageKey": STORAGE_KEY, "payload": payload},
        default={"payload": None, "error": None, "ready": False},
        key=key,
        height=0,
        on_payload_change=lambda: None,
        on_error_change=lambda: None,
        on_ready_change=lambda: None,
    )
    return BrowserStorageResult(result.payload, result.error, bool(result.ready))
