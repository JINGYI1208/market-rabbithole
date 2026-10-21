# Architecture Audit: Pre-V2

This audit records the implementation that existed before the agentic upgrade. It is based on source behavior, not README claims.

## What was implemented

The original market workflow was a fixed orchestration pipeline:

```text
question → GLM plan → local retrieval for every angle → Tavily for every angle
         → one evidence-analysis pass → deterministic conflict grouping
         → GLM brief → rendered result
```

- `app.py` provided a standard-library HTTP UI and in-memory PDF/MD/TXT uploads.
- `semantic_rag.py` extracted and chunked local documents, embedded them with MiniLM, and ranked them by dense cosine similarity.
- `providers.py` wrapped Tavily Search and Zhipu structured generation with graceful local fallbacks.
- `market_agent.py` planned research angles, but Python then executed the same local-and-web sequence for each angle.

## What was not agentic

The LLM did not choose tools, inspect intermediate state, decide whether new research was needed, formulate a follow-up based on an evidence gap, or decide when to stop. Research depth scaled with the number of planned angles rather than the evidence actually found. The trace described predetermined stages; it was not an execution-event log of autonomous decisions.

## Evidence and retrieval gaps

- Retrieval was dense-only; exact entities and market terms had no lexical signal.
- Stance classification existed, but evidence did not explicitly model fact/opinion/forecast, time period, or assumptions.
- Conflict detection grouped SUPPORT and CONTRADICT evidence, but did not expose both citation sets or explain temporal/assumption differences.
- Citations were assigned to evidence, but generated conclusion-to-citation references were not independently checked.
- There was no bounded stopping policy, per-tool budget, invalid-action validation, hosted demo mode, or reproducible evaluation fixture.

## Components worth preserving

The request-scoped upload model, lightweight in-memory retrieval, Tavily/GLM provider adapters, sanitized fallbacks, evidence-oriented UI, and deterministic conflict check were sound prototype choices. V2 extends those components instead of replacing the product with a framework-heavy multi-agent design.
