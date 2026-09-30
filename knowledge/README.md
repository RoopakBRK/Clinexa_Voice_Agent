# WHO knowledge base

- `raw/` — source WHO clinical / primary-care documents (PDF or text), unchanged.
- `processed/` — cleaned, section-aware chunks with metadata (generated; git-ignored contents).
- `manifests/` — one entry per source document: title, document type, population,
  topic, publication date, source URL, checksum. Drives metadata-aware retrieval.

WHO documents are the only knowledge source used at inference time. Ingestion
arrives in Phase 5.
