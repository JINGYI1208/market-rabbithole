# Current Architecture

## System map

```text
Streamlit / HTTP / CLI
        ↓
MarketRabbitHoleAgent
  ├─ initial planner (structured GLM)
  ├─ bounded controller (one validated action per iteration)
  │    ├─ search_local → hybrid in-memory RAG
  │    ├─ search_web   → Tavily basic search
  │    ├─ analyze_evidence → structured GLM extraction
  │    └─ stop → programmatic sufficiency guard
  ├─ deterministic conflict detector
  ├─ structured GLM synthesis
  └─ programmatic citation-ID verifier
        ↓
MarketResearch: plan, evidence, conflicts, brief, events, warnings, stop state
```

## Agent loop

The system uses a single controller rather than role-playing multiple agents. On each iteration it receives a compact state: research angles, collected excerpts and analysis, conflicts, remaining budgets, and the minimum evidence threshold. It returns one structured action. Python validates that the action is allowed, that any angle exists, that searches include a query, and that the relevant tool budget remains.

Stopping is shared responsibility: the LLM proposes `stop`, while code rejects a premature stop unless minimum evidence exists, all evidence has been analyzed, and at least one directional claim exists. A hard iteration limit always terminates the run. Defaults are eight iterations, four local searches, four web searches, two local results, and one web result per call.

If the hard iteration limit is reached, the stop reason remains explicit and the brief receives a deterministic caveat that unresolved evidence gaps may remain. Provider fallback warnings identify the safe operation stage and exception type, but never include prompts, document excerpts, or credentials.

## Retrieval and upload isolation

Documents become sentence-aware overlapping chunks with stable source metadata. MiniLM produces normalized dense vectors. Retrieval min-max normalizes the dense score and combines it with token overlap; the default keyword weight is 0.2. This is a pragmatic hybrid ranker, not BM25 and not a production vector database.

Bundled chunks form the reusable base index. Uploaded PDF/MD/TXT chunks are combined into a request-local `SemanticRAG` instance. They are not written to disk, placed in a database, or appended to the base index.

## Evidence and citations

Each evidence object stores source type, locator, excerpt, research angle, citation ID, extracted claim, stance, evidence kind, time period, assumptions, and optional retrieval relevance. Conflict detection is programmatic: for a research angle, at least one analyzed SUPPORT and one CONTRADICT item creates a conflict with both citation sets.

GLM synthesis must attach evidence IDs to each conclusion. The verifier removes IDs that are not present in the evidence pool and marks conclusions without any valid ID as unsupported. This proves reference integrity only. It does not prove that a cited passage semantically entails the conclusion; that requires a separate faithfulness evaluator.

## Observable events

`ResearchEvent` records typed events such as plan creation, decision, evidence gap, tool selection, search completion, evidence analysis, contradiction discovery, stop rejection, citation verification, and final stop. Events contain concise operational reasons and counts—not hidden chain-of-thought. Public examples are static fixture output; live events remain in the active user session and are not persisted by the application.

## Provider failure and safety

- Provider wrappers return deterministic fallbacks and sanitized warnings; keys are never placed in prompts or responses.
- Live mode is disabled by default and can be protected by a server-side password.
- Model names come from a server-side allowlist.
- Upload type, filename, count, and size are constrained.
- Iteration, search, results-per-call, concurrency, and evidence limits constrain cost and latency.
- Search actions are data passed to provider APIs, never shell commands.

## Trade-offs

A single bounded agent is easier to inspect and evaluate than a multi-agent conversation, but it concentrates planning quality in one model. In-memory NumPy retrieval keeps the demo deployable but will not scale to a large research library. Deterministic conflict and citation checks are explainable, but deliberately narrower than semantic fact checking.
