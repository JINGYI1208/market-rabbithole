"""Validated server-side configuration for local and hosted execution."""

from __future__ import annotations

import os
from dataclasses import dataclass

from rabbithole.market_agent import AgentLimits
from rabbithole.semantic_rag import resolve_retrieval_backend


def _positive_int(name: str, default: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class AppConfig:
    live_agent_enabled: bool
    live_agent_password: str
    models: tuple[str, ...]
    default_model: str
    max_upload_mb: int
    max_concurrency: int
    retrieval_backend: str
    limits: AgentLimits


def load_app_config() -> AppConfig:
    models = tuple(dict.fromkeys(value.strip() for value in os.getenv("RABBITHOLE_MODELS", "glm-4-flash").split(",") if value.strip()))
    if not models:
        raise ValueError("RABBITHOLE_MODELS must contain at least one model")
    default_model = os.getenv("ZHIPU_MODEL", models[0]).strip()
    if default_model not in models:
        default_model = models[0]
    limits = AgentLimits(
        max_iterations=_positive_int("RABBITHOLE_MAX_ITERATIONS", 8, 12),
        max_local_searches=_positive_int("RABBITHOLE_MAX_LOCAL_SEARCHES", 4, 8),
        max_web_searches=_positive_int("RABBITHOLE_MAX_WEB_SEARCHES", 4, 8),
        local_top_k=_positive_int("RABBITHOLE_LOCAL_TOP_K", 2, 5),
        web_results=_positive_int("RABBITHOLE_WEB_RESULTS", 1, 3),
        min_evidence=_positive_int("RABBITHOLE_MIN_EVIDENCE", 3, 10),
        keyword_weight=float(os.getenv("RABBITHOLE_KEYWORD_WEIGHT", "0.2")),
    )
    return AppConfig(
        live_agent_enabled=_flag("RABBITHOLE_LIVE_ENABLED"),
        live_agent_password=os.getenv("RABBITHOLE_LIVE_PASSWORD", ""),
        models=models,
        default_model=default_model,
        max_upload_mb=_positive_int("RABBITHOLE_MAX_UPLOAD_MB", 5, 15),
        max_concurrency=_positive_int("RABBITHOLE_MAX_CONCURRENCY", 1, 4),
        retrieval_backend=resolve_retrieval_backend(),
        limits=limits,
    )
