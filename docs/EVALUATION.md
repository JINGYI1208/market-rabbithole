# Evaluation

## Automated tests

The offline suite uses deterministic encoders and mocked providers, so it does not consume Tavily or Zhipu credits. It covers:

- LLM-directed local → analyze → web follow-up → analyze → stop sequencing
- evidence-gap-driven follow-up queries and conflict detection
- invalid tool rejection, premature-stop rejection, and hard iteration bounds
- citation verification and unsupported-conclusion reporting
- hybrid lexical/dense retrieval behavior
- PDF/MD/TXT chunk metadata, overlap, and request isolation
- Tavily and Zhipu request/response adapters and graceful provider failure
- application multipart parsing and hosted configuration bounds/model allowlist

Run with `pytest -q`.

## Retrieval check

`evaluation/run_evaluation.py` contains four deterministic synthetic cases with an expected source. It compares Recall@1 for semantic-only retrieval and the default 20% keyword fusion.

Current checked-in result (`evaluation/results.json`):

| Variant | Recall@1 | Cases |
|---|---:|---:|
| Semantic only | 1.0 | 4 |
| Hybrid, 20% keyword | 1.0 | 4 |

The result verifies that hybrid fusion preserves expected retrieval on this fixture. It does **not** show an improvement, statistical significance, production accuracy, stance accuracy, citation faithfulness, or investment usefulness.

## What a production evaluation needs

A credible next evaluation would use time-stamped analyst questions and independently labeled passages, measuring retrieval Recall@k, claim extraction accuracy, stance/type classification, contradiction precision/recall, citation entailment, unsupported-claim rate, latency, and provider cost. Human reviewers should compare briefs blind and score usefulness and calibration separately from writing quality.
