# Clinexa — Architecture

Clinexa is a phone-based primary-care **information** assistant. It listens,
collects a structured history, retrieves WHO evidence, screens for red flags, and
routes the caller to the right next step — escalating to a clinician when needed.
It never diagnoses, prescribes, or presents itself as a doctor.

## 1. System overview (target)

```mermaid
flowchart LR
    caller((Caller)) -- PSTN --> twilio[Twilio Voice]
    twilio -- "POST /twilio/voice (signed)" --> api
    twilio <-- "WSS Media Stream<br/>μ-law 8 kHz, 20 ms frames" --> api

    subgraph api[FastAPI · apps/api]
        ws[Media Stream handler] --> session[CallSession]
        session <--> stt[STTProvider<br/>Deepgram streaming]
        session --> graph[LangGraph<br/>agent state machine]
        graph --> policy[Response policy]
        policy --> tts[TTSProvider<br/>streaming]
        tts --> ws
        graph --> tools[Typed tools]
    end

    tools --> rag[Hybrid RAG<br/>Qdrant + BM25 → RRF → cross-encoder]
    tools --> pg[(PostgreSQL)]
    session <--> redis[(Redis<br/>live call state)]
    rag --> kb[(WHO knowledge base)]
    api -. traces .-> obs[Logfire · OpenTelemetry]
    dash[Next.js dashboard] --> api
```

The LLM never touches PostgreSQL or Qdrant directly: every side effect goes
through a typed tool with a timeout, retry policy and structured output.

## 2. Phase 1 — telephony and streaming STT (implemented)

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant T as Twilio
    participant W as POST /twilio/voice
    participant M as WS /twilio/media-stream
    participant S as CallSession
    participant D as Deepgram /v1/listen

    C->>T: dials number
    T->>W: webhook (X-Twilio-Signature)
    W->>W: verify signature, sign stream token = HMAC(secret, CallSid)
    W-->>T: TwiML <Say> disclaimer + <Connect><Stream> + token
    T->>M: WebSocket connect
    T->>M: start {callSid, customParameters.stream_token}
    M->>M: verify token bound to callSid (else close 1008)
    M->>S: create session, start STT task
    S->>D: connect (nova-3, mulaw/8k, interim, endpointing, vad_events)
    loop every 20 ms
        T->>M: media (base64 μ-law)
        M->>S: feed_audio → bounded queue
        S->>D: binary audio (AudioClock records send time)
    end
    D-->>S: SpeechStarted / interim / final / speech_final / UtteranceEnd
    S->>S: UtteranceAssembler → caller turn + latency metrics
    T->>M: stop
    M->>S: close(): CloseStream, flush trailing results
    S-->>M: final transcript + p50/p95/p99 latency
```

### Design decisions

| Decision | Why |
|---|---|
| **Raw Deepgram WebSocket** instead of SDK | Full control over lifecycle, KeepAlive, CloseStream flush and the VAD events barge-in needs; no coupling to SDK major-version churn. |
| **`STTProvider` / `TTSProvider` ABCs** | The call pipeline depends only on `transcribe_stream(audio, fmt) -> AsyncIterator[TranscriptEvent]`; vendors are swappable. |
| **STT in its own task, fed by a bounded queue** | The WS read loop never blocks on the provider. The queue holds audio across STT reconnects and drops frames rather than growing without bound. |
| **Reconnect with backoff** | A transient Deepgram failure mid-call reconnects without losing queued audio; after N failures the call is marked `stt_unavailable`. |
| **Per-call HMAC stream token** | The WebSocket is only accepted if it carries a token minted by the signature-verified webhook for that exact `CallSid`. |
| **Shielded `close()`** | Client disconnects/shutdown cancel the handler, but finalization (STT flush, metrics, summary) still completes. |
| **`AudioClock`** | Deepgram timestamps are in audio time; mapping them to the wall-clock time each frame was sent yields *real* STT latency. |
| **Masked caller ID, transcript logging off by default** | Transcripts are health information; only masked numbers leave the webhook. |

### Latency metrics recorded per call

| Metric | Definition |
|---|---|
| `stt_finalization_ms` | Wall time from sending the last audio covered by a final result to receiving it. |
| `stt_endpoint_ms` | Wall time from sending the caller's last word to the end-of-turn signal (`speech_final` / `UtteranceEnd`). This includes the endpointing silence window and is the STT share of perceived response delay. |

Each is summarised as count / mean / p50 / p95 / p99 / max (`GET /api/calls/{call_sid}`).
Later phases add LLM TTFT, agent, retrieval, reranker, tool and TTS time-to-first-audio.

## 3. Agent graph (Phase 4+)

```mermaid
flowchart TD
    START([START]) --> init[Session init]
    init --> cm[Conversation Manager<br/>intent + stage]
    cm -->|MEDICATION_INFORMATION| med[Medication Info Agent]
    cm -->|EMERGENCY_CONCERN| safety
    cm --> intake[Clinical Intake<br/>structured extraction]
    intake --> safety[Red-Flag / Safety Agent]
    safety --> need{Need more info?}
    need -- yes --> ask[Ask 1–2 targeted questions] --> tts
    need -- no --> rag[Clinical RAG]
    med --> rag
    rag --> validate[Evidence validation]
    validate --> policy[Safety policy]
    policy --> planner[Response planner]
    planner --> esc{Escalate?}
    esc -- yes --> escalation[Clinician escalation summary] --> END([END])
    esc -- no --> tts[Streaming TTS] --> cm
```

The graph state is `app.graph.state.VoiceClinicalState`; the shared contracts
(`ClinicalIntake`, `SafetyAssessment`, `Evidence`, `EscalationSummary`, …) live in
`app.schemas.clinical`. `SafetyAssessment` enforces a fail-safe invariant in the
schema itself: `urgent`/`emergency` always sets `escalation_required`.

## 4. RAG pipeline (Phases 5–8)

```mermaid
flowchart LR
    q[Utterance + clinical state] --> rw[Query rewrite]
    rw --> dense[Qdrant dense · top 15]
    rw --> bm25[BM25 · top 15]
    meta[Metadata filters<br/>population · topic · doc type] --> dense & bm25
    dense & bm25 --> rrf[Reciprocal Rank Fusion k=60 · top 20]
    rrf --> ce[Cross-encoder · top 5]
    ce --> comp[Context compression]
    comp --> ev[Evidence set with scores]
```
