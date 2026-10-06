# Clinexa — Agentic Primary-Care Voice Assistant

A real-time phone assistant that listens to a caller, collects a structured symptom
history, retrieves evidence from WHO primary-care guidance, screens for red flags,
and recommends an appropriate next step — escalating to a human clinician when needed.

> **Safety scope.** Clinexa is a health *information* assistant. It does not diagnose,
> prescribe, change medication, or present itself as a clinician. Every call opens with
> that disclaimer and emergency guidance. All patient data in this project is synthetic.

**Stack:** Twilio Voice + Media Streams · Deepgram streaming STT + Aura TTS · Claude · FastAPI (async) ·
LangGraph multi-agent orchestration · hybrid RAG (Qdrant + BM25 + RRF + cross-encoder) ·
PostgreSQL · Redis · Pydantic Logfire (OpenTelemetry) · Next.js dashboard

Full design with diagrams: **[docs/architecture.md](docs/architecture.md)**

---

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Twilio → WebSocket → Deepgram streaming STT | ✅ done |
| 2 | STT → Claude → streaming TTS → Twilio (spoken replies) | ✅ built; telephony, STT and TTS verified on simulated live calls, the Claude call so far only against a fake |
| 5 | WHO document ingestion & chunking → `data/processed/chunks.jsonl` | ✅ done |
| 6 | Local embeddings (bge-small) + Qdrant dense retrieval with metadata filters | ✅ done (validated on the embedded index; Cloud pending) |
| 7 | BM25 + reciprocal rank fusion hybrid retrieval, clinical metadata filters | ✅ done |
| 8 | Cross-encoder reranking + retrieval evaluation harness | ✅ built; question set is a **draft awaiting human review** |
| 3 | Endpointing tuning, barge-in, persistent TTS connection | |
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

## Spoken replies (Phase 2)

```
caller utterance ─▶ Claude (streamed text) ─▶ sentence chunker ─▶ Deepgram Aura TTS ─▶ Twilio `media` frames ─▶ caller
   (end of turn)        runs while the TTS         first sentence is       μ-law 8 kHz, no        + a `mark` when the
                        socket is connecting       spoken on its own       transcoding            reply has been queued
```

- **One reply task per turn** (`CallSession._reply`). The LLM streams in its own task, so it is already
  generating while the TTS connection opens. Text is cut into sentences (`app/voice/sentences.py`) and
  the first sentence is synthesised while the rest is still being written.
- **Claude** (`app/agents/responder.py`): `ANTHROPIC_MODEL` (default `claude-opus-5-5`) at low effort,
  streamed, with server-side refusal fallback. The system prompt holds the safety scope: no diagnosis,
  no prescribing or dose changes, emergency signposting before anything else, short spoken sentences.
  This single call is the whole "agent" until the LangGraph graph (Phase 4) replaces it. It has no
  retrieval and no red-flag policy yet.
- **Deepgram Aura** over the raw `/v1/speak` WebSocket, returning μ-law 8 kHz directly, so audio goes
  to Twilio untouched.
- **A caller is never left in silence.** If the model errors, times out, declines or returns nothing, a
  fixed line is spoken instead ("…please contact a clinician or call *emergency number*"). A reply that
  fails midway is cut short rather than restarted.
- **The transcript records what was heard.** An assistant turn is stored only if its audio started, and
  is flagged `interrupted` if it was cut off (hang-up, TTS failure).
- **Turn-taking is sequential for now.** If the caller speaks during a reply, one further reply follows
  that covers everything said since. Cutting the assistant off mid-sentence (barge-in) is Phase 3.
- **Latency per call:** `llm_ttft_ms`, `llm_first_sentence_ms`, `llm_total_ms`, `tts_ttfa_ms`,
  `response_latency_ms` (end of turn detected → first reply audio sent) and `voice_to_voice_ms`
  (caller's last word → first reply audio sent).
- Without `ANTHROPIC_API_KEY` (or with no TTS provider) calls are transcribed only, as in Phase 1.

## Roopiee on the website: the browser channel

The Clinexsa website's "Onboarding for Patient" page talks to the same pipeline over a
WebSocket instead of a phone line. Roopiee asks the questions, hears the answers and fills
the page's form through tool calls. The Twilio path is unchanged.

```
POST /web/session {onboarding_id, language}  ->  {ws_url, token, expires_in}
WS   /web/onboarding-stream?token=...

browser -> server   binary  linear16 16 kHz mono mic frames (about 20 ms each)
                    JSON    start | pause | resume | stop | tool_result | form_edit
server -> browser   binary  linear16 24 kHz mono speech
                    JSON    transcript | status | clear_audio | tool_call | error
```

- `app/voice/transport.py`: `CallSession` now speaks through a `Transport`. `TwilioTransport`
  wraps the Media Streams frames; `WebTransport` sends raw PCM plus JSON events.
- `app/voice/web_stream.py`: the two endpoints. The token is an HMAC over the onboarding id
  with a 60 s expiry (`STREAM_TOKEN_SECRET`), usable once. Sessions are rate-limited per IP,
  CORS and the WebSocket `Origin` are limited to `WEB_ALLOWED_ORIGINS`, and a session with
  15 minutes of silence is closed.
- `app/agents/onboarding.py`: `OnboardingReplyGenerator`, one per session. Claude calls the
  ten form tools; each call goes to the page as `tool_call` and the page's `tool_result`
  (or `"timeout"` after 5 s) goes back to Claude. Only text is spoken. Tool arguments are
  health details and are never logged.
- Barge-in: when Deepgram reports `SpeechStarted` while a reply is still playing, the reply
  is cancelled and the page is told to `clear_audio`. If no words follow within a few
  seconds (a cough, a door), Roopiee repeats her question. Phone calls are not affected.
- `app/voice/languages.py`: Nova-3 transcribes all nine languages the page offers, but Aura
  has no Indian-language voices yet, so only English can be spoken. `POST /web/session`
  answers 422 `language_unavailable` for the others and `GET /web/languages` lists which
  can be spoken. Give a language a `tts_model` there to switch it on.

Try it without a browser (API running, real Deepgram and Claude keys):

```bash
make dev PORT=8010
uv run --project apps/api python scripts/simulate_web_onboarding.py --api http://localhost:8010 \
    --text "My name is Ramesh Kumar and I am sixty two years old"
```

## Hosting on Fly.io

The website can only reach Roopiee on a public https address. `Dockerfile` and
`fly.toml` put her on Fly.io in Mumbai (`bom`), on one always-on machine.

**Only a signed-in person can start a session.** `POST /web/session` asks Supabase
whether the access token the website sent is real (`app/voice/sign_in.py`). With
`ENVIRONMENT=production` and no `SUPABASE_PUBLISHABLE_KEY`, every request is
refused. On a laptop, without that key, the check is off.

One-time, from this folder:

```bash
brew install flyctl
fly auth login
fly launch --no-deploy --copy-config --name clinexsa-roopiee --region bom
fly secrets set \
  DEEPGRAM_API_KEY=... \
  ANTHROPIC_API_KEY=... \
  SUPABASE_PUBLISHABLE_KEY=... \
  STREAM_TOKEN_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
fly deploy
```

`SUPABASE_PUBLISHABLE_KEY` is the same value as the website's
`NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY`. The other settings are in `fly.toml`. If
the name `clinexsa-roopiee` is taken, pick another and use that address below.

Then tell the website where she is: on Vercel set
`NEXT_PUBLIC_ROOPIEE_API_URL=https://clinexsa-roopiee.fly.dev` and redeploy.

Check it without starting a conversation:

```bash
curl https://clinexsa-roopiee.fly.dev/health          # "status": "ok", "environment": "production"
curl https://clinexsa-roopiee.fly.dev/web/languages   # English has "speech": true
curl -X POST https://clinexsa-roopiee.fly.dev/web/session \
  -H 'content-type: application/json' -d '{"onboarding_id":"check-0001"}'   # 401, signed_out
```

Things to know:
- **One machine only.** Session tokens and the per-address rate limit live in the
  process's memory. Do not scale to two.
- **The image leaves out the knowledge-base libraries** (PyTorch and the NVIDIA runtime,
  several gigabytes), because nothing a call or a web session touches imports
  `app/rag`. The Dockerfile says which line to change when that is wired in.
- **The Anthropic key needs credit.** Without it Roopiee connects and listens, but
  every reply is the fallback line.
- Not checked yet: whether Fly's proxy closes a paused conversation that sends nothing
  for a minute. If it does, send a keep-alive from the page while paused.

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

## Hybrid retrieval (Phase 7)

```
query ─┬─▶ dense: embed + Qdrant ─ top 15 ─┐
       │                                    ├─▶ RRF (k=60) ─▶ top 20 candidates ─▶ (cross-encoder: Phase 8)
       └─▶ BM25 (stemmed, in-memory) ─ top 15 ┘
              same metadata filter on both legs; legs run concurrently
```

```bash
make query Q="artemether lumefantrine dose" ARGS="--local"      # hybrid (default)
uv run --project apps/api python -m app.rag.retrieval --local query "cough" --mode bm25
uv run --project apps/api python -m app.rag.retrieval --local query "cough" --mode dense --population adult
```

- **Why both legs:** dense search matches meaning ("tummy hurts" ≈ abdominal pain) but can blur exact
  clinical terms; BM25 matches those exactly (drug names, "G6PD", doses). Section titles are part of
  the BM25 text, so a query for "headache" finds the Headache section.
- **`reciprocal_rank_fusion`** is explicit and unit-tested against hand-computed scores. It fuses
  by rank, so cosine similarity and BM25 scores never need calibrating against each other. Every
  candidate keeps `dense_score/rank`, `bm25_score/rank` and `rrf_score` for observability.
- **Filters are identical on both legs** (`RetrievalFilters.matches` for BM25, `to_qdrant` for
  Qdrant; a test proves they select the same chunks). `filters_from_clinical` derives them from what
  the caller has said: age → population (`child`/`adult`, plus `pregnancy`; unknown age → no
  restriction), complaint words → topic hints.
- **Population is a safety constraint, topic is a hint.** If a strict filter leaves fewer than 5
  candidates, the topic filter is dropped and the result is flagged `filter_relaxed`, but population is
  never dropped, so an adult is never answered from paediatric dosing.
- **Per-stage latency** (`dense_ms`, `bm25_ms`, `rrf_ms`, `total_ms`) is returned and logged. On a
  laptop, dense ≈ 30 ms (including the query embedding), BM25 ≈ 1–2 ms, RRF ≈ 0.1 ms.
- The retriever accepts a separate `sparse_query`, so the query rewriter (Phase 9) can give BM25 a clean
  keyword form while the dense leg keeps the patient's natural wording.

## Reranking and retrieval evaluation (Phase 8)

```bash
python -m app.rag.evaluation validate                       # check every gold criterion against the corpus
python -m app.rag.evaluation retrieval --local \
    --rerankers cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-base   # writes evaluation/reports/
```

- **Cross-encoder reranker** (`app/rag/reranking`): re-scores the 20 fused candidates by reading query and
  passage together, keeps every earlier-stage score, adds `reranker_score`/`reranker_rank`. The model
  is a setting (`RERANKER_MODEL`); default `ms-marco-MiniLM-L-6-v2` for latency.
- **Context compression** keeps the query-relevant sentences of each chunk. Caution and referral
  sentences ("do not…", "refer urgently…") are always kept *on top of* the budget, so an irrelevant caution can
  never push out the sentence that answers the question. Implemented and tested; not yet wired into the
  pipeline until faithfulness evaluation (Phase 12) shows it is safe.
- **Evaluation harness** (`app/rag/evaluation`): gold evidence is defined by *criteria* (document +
  section/pages + key terms), not chunk ids, so it survives re-chunking. It compares vector-only, BM25-only,
  hybrid, vector + rerank and hybrid + rerank, and reports Hit@k, evidence recall, MRR, NDCG, per-category results,
  latency, paired bootstrap confidence intervals, a "does BM25 help?" ablation, a metadata-filter ablation,
  and an abstention-signal analysis.
- The 57-question set (`evaluation/datasets/retrieval_eval_v1.yaml`, 42 answerable) is a **draft**: reports
  say so until a human has reviewed it (`evaluation/datasets/REVIEW_NOTES.md`). Do not quote its numbers.

## Repository layout

```
.
├── apps/
│   ├── api/                      FastAPI service (Python 3.12, uv)
│   │   ├── app/
│   │   │   ├── main.py           app factory, /health
│   │   │   ├── core/             settings, structured logging
│   │   │   ├── voice/            Twilio webhook + media stream, session, turns, audio, security
│   │   │   │   └── providers/    STT/TTS interfaces, Deepgram implementations
│   │   │   ├── api/              call inspection endpoints
│   │   │   ├── agents/           reply generator (Claude); LangGraph agents arrive in Phase 4+
│   │   │   ├── graph/            LangGraph state (graph itself: Phase 4)
│   │   │   ├── schemas/          clinical domain contracts
│   │   │   ├── observability/    latency metrics (p50/p95/p99)
│   │   │   ├── rag/              ingestion, retrieval, reranking, evaluation
│   │   │   ├── tools/  memory/  database/    ← later phases
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
a [Deepgram](https://console.deepgram.com) API key (STT and TTS) and an
[Anthropic](https://console.anthropic.com) API key (replies); for real phone calls also a
Twilio account with a voice-capable number and [ngrok](https://ngrok.com).

```bash
cp .env.example .env          # add DEEPGRAM_API_KEY, ANTHROPIC_API_KEY (and Twilio values for real calls)
make install                  # uv sync in apps/api
make test                     # unit + integration tests (no network needed)
make dev                      # API on http://localhost:8000
curl localhost:8000/health
```

### Try it without a phone

`scripts/simulate_call.py` behaves like Twilio: it calls the signed webhook, reads
the stream URL and token from the TwiML, and streams 20 ms μ-law frames in real time.
Your real Deepgram key does the transcription and the speech; your Anthropic key writes the reply.
The simulator stays on the line until the reply has finished, like a caller listening.

```bash
make dev                                              # terminal 1
make simulate TEXT="I've had a cough for five days"   # terminal 2 (macOS `say` voice)
# or: uv run --project apps/api python scripts/simulate_call.py --wav caller_8k.wav
# add --save-reply reply.wav to keep the assistant's speech
```

The simulator prints the transcript (caller and assistant), how long after the caller stopped
speaking the first reply audio arrived, and the latency percentiles the API measured.

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
| `DEEPGRAM_API_KEY` | Streaming STT and TTS |
| `DEEPGRAM_TTS_MODEL` | Aura voice for replies |
| `ANTHROPIC_API_KEY` | Reply generation; without it calls are transcribed but not answered |
| `ANTHROPIC_MODEL` / `LLM_EFFORT` | Model and thinking depth for replies (`claude-opus-5-5`, `low`) |
| `DEEPGRAM_ENDPOINTING_MS` / `DEEPGRAM_UTTERANCE_END_MS` | Turn-detection tuning |
| `LOG_TRANSCRIPTS` | Log utterance text (off by default: transcripts are health data) |

### Quality gates

```bash
make lint        # ruff
make typecheck   # mypy --strict
make test        # pytest
make check       # all of the above
```

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness, STT / TTS / LLM provider status, active call count |
| `POST` | `/twilio/voice` | Twilio incoming-call webhook → TwiML |
| `WS` | `/twilio/media-stream` | Twilio bidirectional Media Stream (caller audio in, reply audio out) |
| `GET` | `/web/languages` | Languages the website offers, and which Roopiee can speak |
| `POST` | `/web/session` | Short-lived token and stream URL for one website onboarding |
| `WS` | `/web/onboarding-stream` | Browser channel: mic audio in, speech and form tool calls out |
| `GET` | `/api/calls/active` | Live calls |
| `GET` | `/api/calls/recent` | Last 50 finished calls |
| `GET` | `/api/calls/{call_sid}` | Transcript, status, latency percentiles |
