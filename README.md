# Clinexa — Agentic Primary-Care Voice Assistant

A real-time phone assistant that listens to a caller, tells them what a medicine is
called and what it contains from a catalogue of 252,553 Indian medicine names, answers
health questions from WHO primary-care guidance, and recommends an appropriate next
step — escalating to a human clinician when needed.

> **Safety scope.** Clinexa is a health *information* assistant. It does not diagnose,
> prescribe, change medication, or present itself as a clinician. Every call opens with
> that disclaimer and emergency guidance. All patient data in this project is synthetic.

**Stack:** Twilio Voice + Media Streams · Deepgram streaming STT + Aura TTS · Claude with tool use ·
FastAPI (async) · hybrid RAG (Qdrant + BM25 + RRF + cross-encoder) · medicine-name search
(Qdrant sparse vectors of spelling and sound) · LangGraph multi-agent orchestration ·
PostgreSQL · Redis · Pydantic Logfire (OpenTelemetry) · Next.js dashboard

Full design with diagrams: **[docs/architecture.md](docs/architecture.md)** ·
what the retrieval pipelines hold and how they measure: **[docs/rag.md](docs/rag.md)**

---

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Twilio → WebSocket → Deepgram streaming STT | ✅ done |
| 2 | STT → Claude → streaming TTS → Twilio (spoken replies) | ✅ built; telephony, STT and TTS verified on simulated live calls, the Claude call so far only against a fake |
| 5 | WHO document ingestion & chunking → `data/processed/chunks.jsonl` | ✅ done |
| 6 | Local embeddings (bge-small) + Qdrant dense retrieval with metadata filters | ✅ done (indexed in Qdrant Cloud and on the embedded index) |
| 7 | BM25 + reciprocal rank fusion hybrid retrieval, clinical metadata filters | ✅ done |
| 8 | Cross-encoder reranking + retrieval evaluation harness | ✅ built; question set is a **draft awaiting human review** |
| — | Lookups on a call: Claude tools for medicine names (Indian medicines catalogue) and WHO guidance | ✅ built; each tool run against real data, the Claude loop so far only against a scripted model |
| 3 | Endpointing tuning, barge-in, persistent TTS connection | |
| 4 | LangGraph state machine | |
| 9–12 | Intake · red-flag safety · human escalation agents (medicine and guidance lookups already work as tools) | |
| 14 | Tracing to Pydantic Logfire over OpenTelemetry | ✅ done: each call is one trace (replies, Claude requests, lookups), with no caller content in it |
| 13 | Redis + PostgreSQL memory | |
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
  This call, with its two lookup tools (next section), is the whole "agent" until the LangGraph graph
  (Phase 4) replaces it. It has no red-flag policy of its own yet.
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

## Tracing (Logfire)

Set `LOGFIRE_TOKEN` and every call becomes one trace in Pydantic Logfire, sent over
OpenTelemetry (`app/observability/tracing.py`). Without the token nothing is sent.

```
/twilio/media-stream            the call
  reply                         one spoken answer: sentences, lookups, fallback or not
    llm round                   one request to Claude: model, stop reason, token counts
    tool lookup_medicine        ok, and how it went (exact, several, close, unknown)
      medicines lookup
    tool search_guidelines      ok, and how many passages
      guidelines search         dense, BM25, RRF and cross-encoder timings
```

A span says what ran, how long it took and how it went. What a caller said, what Claude
asked a tool, medicine names, request headers and endpoint arguments are never put in one,
and the endpoints whose address carries a secret (`/exotel/...`, `/web/onboarding-stream`)
are not traced at all. `tests/test_tracing.py` checks each of these. `/health` reports
whether tracing is on.

## Lookups on a call: medicine names and WHO guidance

```
caller: "I take glycomate five hundred"
   │
   ▼
Claude ──tool_use lookup_medicine("glycomate 500")──▶ medicines catalogue (Qdrant, 252,553 names)
   │        caller hears "Let me check that medicine name."      found by spelling and sound
   │◀─tool_result: not in the catalogue as heard; "Glycomet 500" is, and sounds like it ──┘
   ▼
"Did you mean Glycomet 500?"                     …and for a health question:

Claude ──tool_use search_guidelines(query, age_years, pregnant)──▶ dense + BM25 → RRF → cross-encoder
   │◀─tool_result: 5 passages, each under its document, section and page ──┘   filtered by population
   ▼
"World Health Organization guidance says…"
```

Claude decides when to look something up (`app/tools/knowledge.py`). Two tools:

| Tool | What it searches | What Claude gets back |
|---|---|---|
| `lookup_medicine` | The Indian medicines catalogue: National List of Essential Medicines 2022, the Jan Aushadhi (PMBJP) list and the A to Z medicines dataset of India | The catalogue's spelling of the name, what the medicine contains, its strength and form where every product of that name agrees, or the products the name could mean |
| `search_guidelines` | The WHO knowledge base, with the full hybrid + rerank pipeline below | Up to five passages, each with its document, publisher, section and page |

- **A medicine is looked up before anything is said about it.** Medicine names are what speech
  recognition gets wrong most. The catalogue is searched by letters and sound, so "glycomate" finds
  Glycomet and "eco sprin" finds Ecosprin, with no model in the path.
- **Rules decide the match** (`app/medicines/lookup.py`): the name as said (`exact`), a brand with
  several products (`several`, and Claude asks which is on the strip), a name it could have been misheard
  from (`close`, said back to confirm, with nothing else claimed about it), or not there (`unknown`). A
  name that is merely like another is never swapped for it: the catalogue does not hold every medicine.
- **Guidance is searched for the person asked about.** Claude asks their age first and passes it, and
  pregnancy, with the search. Population is a hard filter: a child is never answered from adult guidance.
- **No doses over the phone.** Passages carry doses for health workers. Claude may name the medicine
  guidance recommends; the dose is left to a clinician or pharmacist.
- **A caller is not left in silence.** A lookup means a second round with Claude, so a short line is
  spoken while it runs. A lookup that fails, times out or has no index comes back to Claude as an error
  it reports to the caller; it never guesses instead.
- **What was looked up stays with the call.** `GET /api/calls/{call_sid}` returns `lookups` (tool,
  outcome, duration) and `evidence` (document, section, page, excerpt), and latency gains
  `lookup_medicine_ms` and `search_guidelines_ms`. Logs carry the tool and how it went, never what the
  caller asked.

Both stores live in the Qdrant named by `QDRANT_URL` and have to be filled once:

```bash
make index                            # WHO chunks → collection clinexa_who_primary_care
make medicines                        # medicines catalogue → collection clinexa_medicines (about 3 minutes to Qdrant Cloud)
make medicine NAME="glycomate 500"    # try one name against the catalogue
curl localhost:8000/health            # "knowledge" shows each store's status
```

Without `QDRANT_URL` the WHO search uses the embedded local index (`make index ARGS="--local"`) and
the medicine lookup is switched off: a quarter of a million names are too many for the embedded index.

## Medicines catalogue

| Source (`data/`) | Entries read | Names kept |
|---|---|---|
| `nlem2022.pdf`: National List of Essential Medicines 2022 | 1,348 | 1,155 |
| `jan_aushdi.pdf`: Jan Aushadhi (PMBJP) product list | 2,111 | 2,088 |
| `A_Z_medicines_dataset_of_India.csv` | 253,973 | 249,310 |
| **Total**, one entry for each name | 257,432 | **252,553** |

One Qdrant point for each name, no chunking. Each carries a sparse vector made of its words, numbers,
three-letter groups and a sound key for each word, weighted by Qdrant's IDF, plus the medicine as
payload (composition, strength, form, pack, manufacturer, source). A bi-encoder leg and a cross-encoder
tie-break exist behind `MEDICINES_ENCODERS=true`; they are off by default because, measured on the
whole catalogue, they did not change which medicine is found. The comparison is in
[docs/rag.md](docs/rag.md).

## Clinexsa on the website: the browser channel

The Clinexsa website's "Onboarding for Patient" page talks to the same pipeline over a
WebSocket instead of a phone line. Clinexsa asks the questions, hears the answers and fills
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
  seconds (a cough, a door), Clinexsa repeats the question. Phone calls are not affected.
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

The website can only reach Clinexsa on a public https address. `Dockerfile` and
`fly.toml` put it on Fly.io in Mumbai (`bom`), on one always-on machine.

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

Then tell the website where it is: on Vercel set
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
- **The image leaves out the model libraries** (PyTorch and the NVIDIA runtime, several
  gigabytes). The server starts without them, and the one thing that needs them, the WHO
  guidance search on phone calls, is switched off there. Web sessions and the medicines
  lookup are not affected. The Dockerfile says which line to change to put them in.
- **The Anthropic key needs credit.** Without it Clinexsa connects and listens, but
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
│   │   │   ├── agents/           reply generator (Claude + tool loop); LangGraph agents arrive in Phase 4+
│   │   │   ├── tools/            the lookups Claude can make on a call
│   │   │   ├── medicines/        Indian medicines catalogue: sources, Qdrant store, name lookup
│   │   │   ├── graph/            LangGraph state (graph itself: Phase 4)
│   │   │   ├── schemas/          clinical domain contracts
│   │   │   ├── observability/    latency metrics (p50/p95/p99)
│   │   │   ├── rag/              ingestion, retrieval, reranking, evaluation
│   │   │   ├── memory/  database/    ← later phases
│   │   └── tests/
│   └── web/                      Next.js dashboard (Phase 16)
├── data/                     WHO source PDFs, medicines sources (NLEM, Jan Aushadhi, A to Z),
│                             manifests/documents.yaml, processed/ (chunks)
├── evaluation/{datasets,retrieval,safety,reports}
├── scripts/simulate_call.py      local Twilio call simulator
├── docs/architecture.md, docs/rag.md
├── .env.example
└── Makefile
```

## Local setup

**Prerequisites:** [uv](https://docs.astral.sh/uv/) (installs Python 3.12 automatically),
a [Deepgram](https://console.deepgram.com) API key (STT and TTS) and an
[Anthropic](https://console.anthropic.com) API key (replies); for real phone calls also a
Twilio account with a voice-capable number and [ngrok](https://ngrok.com).

```bash
cp .env.example .env          # add DEEPGRAM_API_KEY, ANTHROPIC_API_KEY, QDRANT_URL (and Twilio values for real calls)
make install                  # uv sync in apps/api
make test                     # unit + integration tests (no network needed)
make index && make medicines  # once: fill both Qdrant collections
make dev                      # API on http://localhost:8000
curl localhost:8000/health    # stt / tts / llm / knowledge status
```

### Try it without a phone

`scripts/simulate_call.py` behaves like Twilio: it calls the signed webhook, reads
the stream URL and token from the TwiML, and streams 20 ms μ-law frames in real time.
Your real Deepgram key does the transcription and the speech; your Anthropic key writes the reply.
The simulator stays on the line until the reply has finished, like a caller listening.

```bash
make dev                                              # terminal 1
make simulate TEXT="I've had a cough for five days"   # terminal 2 (macOS `say` voice)
make simulate TEXT="What is in dolo six fifty?"       # a medicine name: looked up in the catalogue
# or: uv run --project apps/api python scripts/simulate_call.py --wav caller_8k.wav
# add --save-reply reply.wav to keep the assistant's speech
```

The simulator prints the transcript (caller and assistant), what was looked up and the guidance
passages used, how long after the caller stopped speaking the first reply audio arrived, and the
latency percentiles the API measured.

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
| `LLM_MAX_TOOL_ROUNDS` | Lookups allowed in one reply before Claude has to answer (3) |
| `QDRANT_URL` / `QDRANT_API_KEY` | The Qdrant server holding both collections; unset = embedded local index, WHO guidance only |
| `MEDICINES_LOOKUP` / `MEDICINES_COLLECTION` | Medicine-name lookup on calls, and the collection it reads (`clinexa_medicines`) |
| `GUIDELINES_RETRIEVAL` / `GUIDELINES_TIMEOUT_S` | WHO guidance search on calls, and how long a caller waits on one (4 s) |
| `DEEPGRAM_KEYTERMS` | Terms boosted in speech recognition; defaults to common Indian medicine names |
| `DEEPGRAM_ENDPOINTING_MS` / `DEEPGRAM_UTTERANCE_END_MS` | Turn-detection tuning |
| `LOG_TRANSCRIPTS` | Log utterance text (off by default: transcripts are health data) |
| `LOGFIRE_TOKEN` / `LOGFIRE_SERVICE_NAME` | Send traces to Pydantic Logfire; blank = no tracing |

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
| `GET` | `/health` | Liveness, STT / TTS / LLM provider status, medicines and guidance status, active call count |
| `POST` | `/twilio/voice` | Twilio incoming-call webhook → TwiML |
| `WS` | `/twilio/media-stream` | Twilio bidirectional Media Stream (caller audio in, reply audio out) |
| `GET` | `/web/languages` | Languages the website offers, and which Clinexsa can speak |
| `POST` | `/web/session` | Short-lived token and stream URL for one website onboarding |
| `WS` | `/web/onboarding-stream` | Browser channel: mic audio in, speech and form tool calls out |
| `GET` | `/api/calls/active` | Live calls |
| `GET` | `/api/calls/recent` | Last 50 finished calls |
| `GET` | `/api/calls/{call_sid}` | Transcript, lookups and guidance passages used, status, latency percentiles |
