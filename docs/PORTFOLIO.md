# Portfolio Narrative

## One-minute explanation

Market RabbitHole is a bounded market-research agent, not a generic chatbot. A user asks a market question and may attach private research. The agent first creates a plan, then repeatedly decides whether the next best step is local retrieval, current web research, evidence analysis, or stopping. It models each source as a claim with stance, evidence type, time period, and assumptions; surfaces supporting-versus-opposing evidence; and produces conclusions tied to citation IDs. I kept the agent controllable with validated tools, search and iteration budgets, a stop guard, upload isolation, and programmatic citation checks. The public demo uses recorded deterministic runs, so recruiters can inspect the product without consuming APIs, while a gated live mode demonstrates the real Tavily and GLM integration.

## Résumé bullets

- Built a bounded single-agent market-intelligence workflow that dynamically selects local RAG, Tavily web search, evidence analysis, or stopping based on evidence gaps and remaining tool budgets.
- Implemented request-scoped PDF/Markdown/text retrieval using MiniLM embeddings plus lexical score fusion, with structured fact/opinion/forecast classification and deterministic contradiction detection.
- Designed traceable synthesis with claim-level citation IDs, programmatic reference validation, reproducible zero-API demo fixtures, mocked provider tests, and Streamlit deployment controls.

## Honest design trade-offs

- “Agentic” refers to state-dependent tool selection and stopping, not multiple personas or unrestricted autonomy.
- Citation verification confirms that referenced evidence exists; it does not claim full entailment checking.
- The checked-in retrieval fixture is a regression check, not proof of market-research quality.
- Qualitative evidence strength communicates uncertainty without implying statistical confidence.
