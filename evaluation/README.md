# Evaluation (Phase 15)

- `datasets/` — 50+ labelled questions (symptom, condition, red-flag, medication,
  follow-up, ambiguous, out-of-domain) with gold chunk IDs and expected behaviour.
- `retrieval/` — Recall@5/10, MRR, NDCG for vector-only vs BM25-only vs hybrid vs
  hybrid + reranker.
- `safety/` — red-flag recall, abstention accuracy, prompt-injection and
  hallucination-attempt cases.
- `reports/` — generated reports. Only measured numbers are reported anywhere.
