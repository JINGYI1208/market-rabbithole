# Market RabbitHole

**Market RabbitHole takes a market question, creates a research plan, searches both private/local documents and live web sources, analyzes conflicting evidence, and produces a cited market brief.**

## Demo

[▶ Watch the Market RabbitHole demo](assets/market-rabbithole-demo.mov)

## Why I built this

Generic LLM answers often hide what evidence was used, whether current information conflicts with existing research, and where uncertainty comes from. Market RabbitHole is therefore organized around a transparent research workflow:

**Plan → Retrieve → Analyze → Challenge → Synthesize → Cite**

Local RAG supplies private and contextual evidence, while live search adds recency. Evidence classification and Conflict Radar surface disagreement, and citations keep the final view traceable to its sources.

## How it works

```text
Market Question
      ↓
GLM Research Planning
      ↓
Local Semantic RAG  +  Tavily Live Web Search
      ↓
Unified Evidence Pool
      ↓
GLM Evidence Analysis
      ↓
Conflict Detection
      ↓
Cited Market Brief
```

Users can optionally upload private `.pdf`, `.md`, or `.txt` research before asking a question. Uploaded files are processed in memory for that request only and are never added to persistent storage.

## Key features

- **Dynamic Research Planning** — Zhipu GLM creates question-specific research angles and exposes the workflow through a Research Trace.
- **Hybrid Retrieval** — Local RAG searches bundled and optionally uploaded documents; conservative Tavily searches add current web evidence and source URLs.
- **Evidence Reasoning** — GLM extracts claims, classifies evidence as SUPPORT, CONTRADICT, or NEUTRAL, and assigns qualitative Evidence Strength without implying statistical confidence.
- **Conflict Detection** — Conflict Radar highlights research angles containing meaningful supporting and contradicting evidence.
- **Traceable Synthesis** — the market brief links local evidence through `[L#]` citations and web evidence through clickable `[W#]` sources, with understandable fallbacks if an external provider fails.

## Example research question

September 2026 FOMC portfolio case:

> **How has the September 2026 FOMC meeting changed the US rates outlook compared with the prior policy path, and what are the implications for 2Y Treasuries, US tech stocks, and gold over the next 3–6 months? Identify the key risks that could invalidate the base case.**

The resulting page separates:

- Executive Summary
- Key Findings
- Evidence from uploaded/local documents
- Live Web Evidence
- Risks / Caveats
- Conflict Radar
- Sources and citations
- What to Monitor Next

## Architecture

```text
app.py                         Server-rendered research UI
  ↓
MarketRabbitHoleAgent          Sequential orchestration
  ├── SemanticRAG              TXT/MD/PDF → chunks → embeddings → cosine top-k
  ├── TavilySearchProvider     Basic live search, one result per research angle
  ├── ZhipuStructuredLLM       Planning, evidence classification, synthesis
  └── Resilient fallbacks      Sanitized errors and local-only degradation
  ↓
MarketResearch                 In-memory evidence, conflicts, citations, brief
```

### Technical implementation

- Local documents are split into overlapping 140-word chunks.
- `sentence-transformers/all-MiniLM-L6-v2` creates normalized dense embeddings.
- NumPy cosine similarity provides lightweight in-memory top-k retrieval.
- Tavily contributes fresh snippets, dates, publishers, and clickable URLs.
- Zhipu GLM returns structured JSON for planning, evidence analysis, and final synthesis.
- Conflict detection is deterministic: an angle is flagged when analyzed evidence contains both SUPPORT and CONTRADICT signals.
- Tests use provider mocks and never consume real Tavily or Zhipu credits.

## Tech stack

- Python 3.10+
- `sentence-transformers`
- NumPy
- `pypdf`
- Tavily Search API
- Zhipu BigModel / GLM API
- Python standard-library HTTP server
- Pytest

## Local setup

```bash
git clone <your-repository-url>
cd market-rabbithole

conda create -n market-rabbithole python=3.12 -y
conda activate market-rabbithole
python -m pip install -e '.[test]'

cp .env.example .env
```

Add your own keys to `.env`:

```dotenv
ZHIPUAI_API_KEY=your_zhipu_api_key
TAVILY_API_KEY=your_tavily_api_key

# Optional overrides
ZHIPU_MODEL=glm-4-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RABBITHOLE_DEBUG=0
```

Never commit `.env`. It is ignored by Git.

Start the application:

```bash
python app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The embedding model downloads locally on first use.

## Testing

```bash
pytest -q
```

The suite covers semantic retrieval, request-scoped uploads, Tavily request configuration, GLM structured responses, citations, and provider failure fallbacks.

## Privacy behavior

- Uploaded PDF/Markdown/text files remain in memory for the current request.
- Uploads are not written to a database or document library.
- The bundled semantic index is not mutated by uploaded files.
- API keys are read from environment variables and sanitized from development logs.
- Relevant retrieved excerpts are sent to GLM for evidence analysis and synthesis; users should not upload material they are not permitted to send to the configured API provider.

## Limitations and future improvements

Market RabbitHole is designed as a research workflow prototype; production deployment would require stronger security, evaluation, persistence, and operational controls.

- No authentication, persistent sessions, scheduled monitoring, or database.
- Evidence Strength is qualitative rather than statistically calibrated.
- Search coverage depends on Tavily; reasoning quality depends on the configured GLM model.
- PDF extraction does not include OCR for scanned documents.
- NumPy retrieval is appropriate for a small corpus; FAISS or a persistent vector store would be the next step at larger scale.
- A production version should add a labeled evaluation set for retrieval recall, stance accuracy, citation faithfulness, and brief quality.

## Repository map

```text
assets/market-rabbithole-demo.mov  Portfolio demo
app.py                             UI and HTTP entry point
sample_corpus/                     Bundled fallback/demo knowledge
src/rabbithole/market_agent.py     Market research workflow and models
src/rabbithole/semantic_rag.py     Parsing, chunking, embeddings, retrieval
src/rabbithole/providers.py        Tavily, Zhipu, logging, fallbacks
tests/                              Offline unit and workflow tests
docs/INTERVIEW_PLAYBOOK.md         Interview narrative and trade-offs
```
