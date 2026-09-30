# Clinexa — Agentic Primary-Care Voice Assistant

A real-time phone assistant that listens to a caller, collects a structured symptom
history, retrieves evidence from WHO primary-care guidance, screens for red flags,
and recommends an appropriate next step — escalating to a human clinician when needed.

> **Safety scope.** Clinexa is a health *information* assistant. It does not diagnose,
> prescribe, change medication, or present itself as a clinician. Every call opens with
> that disclaimer and emergency guidance. All patient data in this project is synthetic.

**Stack:** Twilio Voice + Media Streams · Deepgram streaming STT · FastAPI (async) ·
LangGraph multi-agent orchestration · hybrid RAG (Qdrant + BM25 + RRF + cross-encoder) ·
PostgreSQL · Redis · Pydantic Logfire (OpenTelemetry) · Next.js dashboard

Full design with diagrams: **[docs/architecture.md](docs/architecture.md)**

---

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Twilio → WebSocket → Deepgram streaming STT | ✅ done |
| 5 | WHO document ingestion & chunking → `data/processed/chunks.jsonl` | ✅ done |
| 6 | Local embeddings (bge-small) + Qdrant dense retrieval with metadata filters | ✅ done (validated on the embedded index; Cloud pending) |
| 7–8 | BM25 + RRF hybrid · cross-encoder reranking | ⏳ next |
| 2 | STT → LLM → streaming TTS → Twilio | |
| 3 | Streaming, endpointing, barge-in | |
| 4 | LangGraph state machine | |
| 9–12 | Intake · red-flag safety · medication info · human escalation agents | |
| 13–14 | Redis + PostgreSQL memory · observability | |
| 15 | RAG + safety evaluation | |
| 16–17 | Next.js dashboard · Docker + deployment | |

Metrics on this page are only ever values measured by the evaluation and benchmark
scripts in this repo.

## Phase 1: what works today

```
Caller ──PSTN──▶ Twilio ──POST /twilio/voice──▶ FastAPI  (signature-verified; returns TwiML)
                   │                                      <Say> disclaimer
                   │                                      <Connect><Stream> + per-call HMAC token
                   └──WSS /twilio/media-stream──▶ CallSession ──▶ Deepgram nova-3 (streaming)
                          μ-law 8 kHz, 20 ms frames        │
                                                           ▼
                               caller utterances + STT latency (p50/p95/p99) per call
```

- **Twilio webhook** (`POST /twilio/voice`) validates `X-Twilio-Signature` against the
  public URL, speaks a safety disclaimer and opens a bidirectional Media Stream.
- **Media Stream endpoint** (`WS /twilio/media-stream`) parses every Twilio frame with
  typed Pydantic models, and only accepts streams carrying an HMAC token minted for that
  exact `CallSid` by the verified webhook.
- **Deepgram streaming STT** over the raw WebSocket API: interim results, endpointing,
  `UtteranceEnd` fallback, VAD `SpeechStarted` (the Phase 3 barge-in trigger), KeepAlive,
  graceful `CloseStream` flush, and reconnect-with-backoff without losing queued audio.
- **Turn assembly** joins finalized segments into caller utterances.
- **Latency measured, not estimated**: an `AudioClock` maps Deepgram's audio timestamps
  to the wall-clock time each frame was sent, giving real `stt_finalization_ms` and
  `stt_endpoint_ms` samples per call.
- **Call inspection API**: `GET /api/calls/active`, `/api/calls/recent`, `/api/calls/{call_sid}`.
- **Provider abstraction**: `STTProvider.transcribe_stream()` / `TTSProvider.synthesize_stream()`.
- **Domain contracts ready for later phases**: `VoiceClinicalState`, `ClinicalIntake`,
  `SafetyAssessment` (urgent ⇒ escalation enforced in the schema), `Evidence`,
  `EscalationSummary`, `DocumentMetadata`, retrieval score models.

## Knowledge base: WHO ingestion (Phase 5)

`make ingest` turns the PDFs in `data/` into section-aware chunks (no API keys needed):

```
data/*.pdf ─▶ extract ─▶ clean ─▶ section detection ─▶ structure-aware chunking ─▶ classify
              (layout model,       (per-doc strategy:      (token budget from the        (topic, population,
               cached per PDF)      toc / numbered /        embedding model's own          chunk type,
                                    markdown / running)     tokenizer, tables atomic)      retrievable)
```

| Output (`data/processed/`) | Contents |
|---|---|
| `chunks.jsonl` | One chunk per line: text, token count, deterministic `chunk_id`, full `DocumentMetadata` |
| `ingestion_report.md` | Chunks, section counts, token p50/p95/max and label distributions per document |
| `samples.md` | Random chunks per document for human spot-checking |

Design notes:

- **Layout-model extraction** (PyMuPDF layout via `pymupdf4llm`) classifies every page region
  (heading, text, list item, table, header/footer), so bullets, tables and running headers
  are handled by type instead of by guessing from markdown.
- **Per-document structure strategy** in `data/manifests/documents.yaml`, because these
  PDFs number their headings differently: PDF bookmarks (malaria), numbered headings
  (pocket books, BSI), heading size (adult primary care) or the running page header (mhGAP).
- **Chunks never cross section boundaries.** Small sibling sections are merged with their
  sub-headings kept inline; oversized tables split by row with the header repeated; lists stay
  with their introductory sentence; consecutive prose chunks share a short sentence overlap.
- **Rule-based labels** (topic, population, `text`/`table`/`recommendation`/`warning`) drive
  Qdrant metadata filters. References, contents pages, front matter and fragments under 8
  tokens are stored but flagged `retrievable=false` so they are never indexed.
- `python -m app.rag.ingestion --docs <doc_id> --force-extract` re-runs one document.

> `sa-ndoh-adult-primary-care` is the South African National Department of Health *Adult
> Primary Care* guide, not a WHO publication. It is kept as a supplementary adult
> source and labelled with its real publisher in the manifest.

## Vector index: embeddings + Qdrant (Phase 6)

```bash
make index                      # embed chunks.jsonl → Qdrant (Cloud if QDRANT_URL is set)
make index ARGS="--local"       # embedded local index in data/indexes/qdrant (no server)
make query Q="child with fast breathing and cough"          # ARGS="--local" if using the local index
uv run --project apps/api python -m app.rag.retrieval status
uv run --project apps/api python -m app.rag.retrieval query "fever" --population adult --topic infectious_disease
```

- **Embeddings run locally** (`BAAI/bge-small-en-v1.5`, 384-d, cosine): no network hop in the
  call path and no patient text sent to an embedding API. Each chunk is embedded with a
  contextual header — `Document > Section > Subsection` — so short passages carry their
  context. Queries use BGE's retrieval instruction; passages don't.
- **Incremental and idempotent.** Point IDs are `uuid5(chunk_id)` and each point stores a
  hash of (model, embedded text). Re-running embeds only what changed, deletes points for
  chunks that no longer exist, and refuses to write into a collection with a different
  vector size (`--recreate` rebuilds).
- **Metadata filters** (`RetrievalFilters`): population, topics, document type, document id and
  chunk type — each matches any listed value — and a mandatory `retrievable = true`, so
  references and front matter can never be returned. Payload indexes are created on Qdrant Cloud.
- **Backend selection:** `QDRANT_URL` set → Qdrant Cloud/server; unset (or `--local`) → the
  embedded local index. If the cluster is unreachable the CLI says so; it never falls back silently.

## Repository layout

```
.
├── apps/
│   ├── api/                      FastAPI service (Python 3.12, uv)
│   │   ├── app/
│   │   │   ├── main.py           app factory, /health
│   │   │   ├── core/             settings, structured logging
│   │   │   ├── voice/            Twilio webhook + media stream, session, turns, audio, security
│   │   │   │   └── providers/    STT/TTS interfaces, Deepgram implementation
│   │   │   ├── api/              call inspection endpoints
│   │   │   ├── graph/            LangGraph state (graph itself: Phase 4)
│   │   │   ├── schemas/          clinical domain contracts
│   │   │   ├── observability/    latency metrics (p50/p95/p99)
│   │   │   ├── agents/  rag/  tools/  memory/  database/    ← later phases
│   │   └── tests/
│   └── web/                      Next.js dashboard (Phase 16)
├── data/                     WHO source PDFs, manifests/documents.yaml, processed/ (chunks)
├── evaluation/{datasets,retrieval,safety,reports}
├── scripts/simulate_call.py      local Twilio call simulator
├── docs/architecture.md
├── .env.example
└── Makefile
```

## Local setup

**Prerequisites:** [uv](https://docs.astral.sh/uv/) (installs Python 3.12 automatically),
a [Deepgram](https://console.deepgram.com) API key; for real phone calls also a Twilio
account with a voice-capable number and [ngrok](https://ngrok.com).

```bash
cp .env.example .env          # add DEEPGRAM_API_KEY (and Twilio values for real calls)
make install                  # uv sync in apps/api
make test                     # unit + integration tests (no network needed)
make dev                      # API on http://localhost:8000
curl localhost:8000/health
```

### Try it without a phone

`scripts/simulate_call.py` behaves like Twilio: it calls the signed webhook, reads
the stream URL and token from the TwiML, and streams 20 ms μ-law frames in real time.
Your real Deepgram key does the transcription.

```bash
make dev                                              # terminal 1
make simulate TEXT="I've had a cough for five days"   # terminal 2 (macOS `say` voice)
# or: uv run --project apps/api python scripts/simulate_call.py --wav caller_8k.wav
```

The simulator prints the transcript and the measured STT latency percentiles.

### Real phone calls via Twilio

1. `make tunnel` (ngrok) and copy the `https://…` forwarding URL.
2. In `.env` set `PUBLIC_BASE_URL` to that URL and fill `TWILIO_ACCOUNT_SID` /
   `TWILIO_AUTH_TOKEN`. Restart `make dev`.
3. Twilio Console → Phone Numbers → your number → **Voice configuration** →
   "A call comes in": **Webhook**, `https://<ngrok-host>/twilio/voice`, **HTTP POST**.
4. Call the number. Speak after the disclaimer. Utterances appear in the API logs
   (`call.utterance`; set `LOG_TRANSCRIPTS=true` to see text) and at
   `GET /api/calls/recent`.

> `PUBLIC_BASE_URL` must exactly match the URL configured in Twilio. Signature
> validation is computed over it, and mismatches return `403`.

### Configuration

All settings are environment variables documented in [`.env.example`](.env.example).
The most important ones:

| Variable | Purpose |
|---|---|
| `PUBLIC_BASE_URL` | Public origin Twilio reaches; builds the `wss://` stream URL and validates signatures |
| `TWILIO_AUTH_TOKEN` | Webhook signature validation (fails closed if missing while validation is on) |
| `STREAM_TOKEN_SECRET` | HMAC secret for stream tokens; set explicitly when running >1 instance |
| `DEEPGRAM_API_KEY` | Streaming STT |
| `DEEPGRAM_ENDPOINTING_MS` / `DEEPGRAM_UTTERANCE_END_MS` | Turn-detection tuning |
| `LOG_TRANSCRIPTS` | Log utterance text (off by default: transcripts are health data) |

### Quality gates

```bash
make lint        # ruff
make typecheck   # mypy --strict
make test        # pytest
make check       # all of the above
```

## API reference (Phase 1)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness, STT provider status, active call count |
| `POST` | `/twilio/voice` | Twilio incoming-call webhook → TwiML |
| `WS` | `/twilio/media-stream` | Twilio bidirectional Media Stream |
| `GET` | `/api/calls/active` | Live calls |
| `GET` | `/api/calls/recent` | Last 50 finished calls |
| `GET` | `/api/calls/{call_sid}` | Transcript, status, latency percentiles |
