# Market RabbitHole

**A bounded market-research agent that decides what to investigate next, combines private documents with current web evidence, challenges contradictions, and produces a cited brief.**

## Demo

[▶ Watch the product demo](assets/market-rabbithole-demo.mov)

The Streamlit app has two deliberately separate modes:

- **Public Demo** replays three reproducible, fixture-based runs of the real agent loop with zero API usage.
- **Live Agent** is server-gated and runs bounded research with local RAG, Tavily, and Zhipu GLM.

## Why I built this

Generic LLM answers often obscure which evidence was used, whether recent information conflicts with prior research, and where uncertainty comes from. Market RabbitHole turns that opaque answer into an inspectable workflow:

**Plan → Decide → Retrieve → Analyze → Challenge → Synthesize → Verify citations**

Local RAG supplies private context, live search supplies recency, and structured evidence analysis makes disagreement and uncertainty visible.

## What makes it agentic

After creating an initial research plan, one controller chooses exactly one action on each iteration: search local sources, search the live web, analyze collected evidence, or stop. Its choice depends on the current evidence, detected conflicts, named evidence gaps, and remaining budget. Programmatic validation rejects unknown tools, invalid angles, premature stops, and out-of-budget calls.

This is not an unrestricted autonomous agent. It is intentionally bounded by iteration, search, result, upload, and concurrency limits.

```text
Question + optional PDF/MD/TXT
          ↓
Initial research plan
          ↓
┌── Bounded decision loop ─────────────────────────┐
│ decide → hybrid local RAG / Tavily / analyze     │
│          → inspect gaps & contradictions → stop  │
└───────────────────────────────────────────────────┘
          ↓
Evidence model: claim · stance · type · time · assumptions
          ↓
Conflict detection → cited brief → citation-ID verification
```

## Core capabilities

- **Dynamic research control** — the LLM chooses follow-up actions from actual state; a structured event stream shows decisions, tool calls, gaps, contradictions, verification, and stop reason without exposing chain-of-thought.
- **Configurable local retrieval** — local development can use normalized MiniLM similarity plus lexical overlap; memory-constrained Render instances use sparse TF-IDF lexical ranking. Both search bundled and request-scoped uploaded sources, while Tavily provides current web evidence.
- **Evidence reasoning** — each item becomes a grounded claim classified as fact, opinion, or forecast and SUPPORT, CONTRADICT, or NEUTRAL, with time period and assumptions.
- **Conflict detection** — deterministic logic surfaces opposing analyzed claims and their citation IDs, while distinguishing possible time-period or assumption differences.
- **Traceable synthesis** — conclusions carry evidence IDs; a programmatic verifier removes nonexistent citations and flags unsupported conclusions. This validates citation existence, not full semantic entailment.

## Example question

> How has the September 2026 FOMC meeting changed the US rates outlook compared with the prior policy path, and what are the implications for 2Y Treasuries, US tech stocks, and gold over the next 3–6 months? Identify the key risks that could invalidate the base case.

## Quick start

```bash
conda create -n market-rabbithole python=3.12 -y
conda activate market-rabbithole
python -m pip install -e '.[test,semantic]'
cp .env.example .env
streamlit run streamlit_app.py
```

Open `http://localhost:8501`. Public Demo works without credentials. For Live Agent, add your own keys to `.env` and set `RABBITHOLE_LIVE_ENABLED=1`:

```dotenv
ZHIPUAI_API_KEY=your_zhipu_api_key
TAVILY_API_KEY=your_tavily_api_key
ZHIPU_MODEL=glm-4-flash
RABBITHOLE_LIVE_ENABLED=1
RABBITHOLE_RETRIEVAL_BACKEND=auto
```

Never commit `.env` or `.streamlit/secrets.toml`. Uploaded files stay in memory for that request and are not added to the bundled index. Relevant excerpts are sent to the configured LLM, so users should only upload material they are permitted to share with that provider.

The original local HTTP interface remains available with `python app.py`. The CLI is:

```bash
rabbithole "What could change the copper market balance?" --max-iterations 8
```

## Technical stack

Python 3.12 · Streamlit · `sentence-transformers/all-MiniLM-L6-v2` · NumPy · Tavily Search · Zhipu GLM · PyPDF · Pytest

- Source documents: request-scoped PDF/Markdown/text plus a small bundled fallback corpus.
- Chunking: independently implemented, sentence-aware overlapping windows with source metadata.
- Retrieval: `auto` selects sparse TF-IDF lexical ranking on Render and MiniLM dense embeddings with lexical-score fusion locally. Set `RABBITHOLE_RETRIEVAL_BACKEND=tfidf` or `sentence-transformers` to choose explicitly.
- Providers: structured JSON calls with deterministic, sanitized fallback behavior.
- State: request-local Python objects; no accounts, database, or persistent upload storage.

## Validation

```bash
pytest -q
python evaluation/run_evaluation.py
python -m compileall -q app.py streamlit_app.py src tests
```

The deterministic evaluation fixture currently reports Recall@1 of **1.0 for both semantic-only and 20% keyword-fused retrieval** across four small cases. It verifies behavior but does not demonstrate a quality improvement or production performance. See [evaluation details](docs/EVALUATION.md).

## Documentation

- [Architecture audit](docs/ARCHITECTURE_AUDIT.md) — what the pre-upgrade system actually did
- [Current architecture](docs/ARCHITECTURE.md) — loop, tools, state, evidence, and safeguards
- [Evaluation](docs/EVALUATION.md) — test coverage, retrieval fixture, and limitations
- [Deployment](docs/DEPLOYMENT.md) — Streamlit Community Cloud configuration
- [Portfolio narrative](docs/PORTFOLIO.md) — one-minute explanation and résumé bullets

## Limitations

Market RabbitHole is a research-workflow prototype; production deployment would require hardened authentication, encrypted persistence, broader evaluation, observability, abuse controls, and provider-level data governance. Evidence strength is qualitative rather than statistically calibrated. Citation verification checks IDs, not entailment. PDF extraction has no OCR, Tavily coverage varies, and the in-memory index is intended for small corpora.

## Repository map

```text
streamlit_app.py                Public Demo + gated Live Agent UI
app.py                          Compatible local HTTP UI
src/rabbithole/market_agent.py  Agent state, bounded loop, evidence, verification
src/rabbithole/semantic_rag.py  Parsing, chunking, hybrid retrieval
src/rabbithole/providers.py     Tavily/GLM adapters and fallbacks
src/rabbithole/config.py        Validated server-side limits
demo_cases/                     Recorded fixture-based agent runs
evaluation/                     Deterministic retrieval evaluation
tests/                          Offline tests; no API credits consumed
```
