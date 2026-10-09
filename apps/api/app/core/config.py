"""Application configuration.

All settings are read from environment variables (or the repo-root ``.env``).
Secrets are wrapped in ``SecretStr`` so they never appear in logs or reprs.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

API_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = API_DIR.parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", API_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application -------------------------------------------------------
    app_name: str = "Clinexa"
    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"
    log_json: bool = True
    # Transcripts are health information; only log their text in dev with synthetic callers.
    log_transcripts: bool = False

    # Public HTTPS origin Twilio uses to reach this service (e.g. an ngrok URL).
    # Needed to build the Media Stream wss:// URL and to validate Twilio signatures.
    public_base_url: str | None = None

    # --- Twilio ------------------------------------------------------------
    twilio_account_sid: str | None = None
    twilio_auth_token: SecretStr | None = None
    twilio_validate_signatures: bool = True
    twilio_say_voice: str = "Polly.Joanna-Neural"

    # HMAC secret for the per-call token handed to the Media Stream via TwiML.
    # Must be set explicitly when running more than one API instance.
    stream_token_secret: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32))
    )

    # --- Deepgram (STT) ----------------------------------------------------
    deepgram_api_key: SecretStr | None = None
    deepgram_stt_url: str = "wss://api.deepgram.com/v1/listen"
    deepgram_stt_model: str = "nova-3"
    deepgram_language: str = "en-US"
    # Silence (ms) before Deepgram marks a result speech_final.
    deepgram_endpointing_ms: int = Field(default=300, ge=10, le=5000)
    # Word-gap (ms) fallback for end-of-utterance when endpointing misses it.
    deepgram_utterance_end_ms: int = Field(default=1000, ge=1000, le=5000)
    # Domain terms to boost recognition (Nova-3 keyterm prompting). Medicine names are
    # what speech recognition gets wrong most, so common Indian ones are on by default.
    deepgram_keyterms: list[str] = Field(
        default_factory=lambda: [
            "Paracetamol",
            "Metformin",
            "Glimepiride",
            "Amlodipine",
            "Telmisartan",
            "Losartan",
            "Atorvastatin",
            "Rosuvastatin",
            "Thyroxine",
            "Pantoprazole",
            "Aspirin",
            "Clopidogrel",
            "Amoxicillin",
            "Azithromycin",
            "Insulin",
            "Dolo 650",
            "Crocin",
            "Glycomet",
            "Telma",
            "Thyronorm",
            "Ecosprin",
            "Pan 40",
            "Shelcal",
        ]
    )

    # --- Deepgram (TTS) ----------------------------------------------------
    deepgram_tts_url: str = "wss://api.deepgram.com/v1/speak"
    deepgram_tts_model: str = "aura-2-thalia-en"

    # --- Tracing (app/observability/tracing.py) --------------------------------
    # Write token of a Pydantic Logfire project. Unset: no trace is sent anywhere.
    logfire_token: SecretStr | None = None
    logfire_service_name: str = "clinexa-api"

    # --- LLM (Anthropic Claude) ----------------------------------------------
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    # Thinking depth; low keeps time-to-first-token short for spoken replies.
    # Blank = not sent, for models that reject the parameter (e.g. claude-haiku-4-5).
    llm_effort: Literal["low", "medium", "high"] | None = "low"
    # Re-run a request declined by a safety classifier on Anthropic's recommended
    # fallback model, server-side. Turn off for models that do not support it.
    llm_refusal_fallback: bool = True
    # Includes thinking tokens; spoken replies themselves are a few sentences.
    llm_max_tokens: int = Field(default=2048, ge=256)
    llm_timeout_s: float = Field(default=20.0, gt=0)
    # How many times in one reply Claude may look something up (the medicines catalogue,
    # the guidelines) before it has to answer with what it has. A caller is waiting.
    llm_max_tool_rounds: int = Field(default=3, ge=0, le=8)

    # --- Web onboarding channel (Clinexsa website) ---------------------------
    # Browser origins allowed to open an onboarding session (CORS + WebSocket Origin).
    web_allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    # Lifetime of the single-use token handed out by POST /web/session.
    web_token_ttl_s: int = Field(default=60, ge=5, le=600)
    # A session with no speech and no form activity for this long is closed.
    web_idle_timeout_s: int = Field(default=900, ge=30)
    # Sessions one IP address may start per minute.
    web_sessions_per_minute: int = Field(default=10, ge=1)
    # How long the agent waits for the page to answer a tool call.
    web_tool_timeout_s: float = Field(default=5.0, gt=0)
    # Spoken on the web channel when a reply could not be generated.
    onboarding_reply_fallback: str = (
        "I'm sorry, I'm having trouble right now. "
        "You can keep filling in the form yourself. It's the same form."
    )
    # Medicine names boosted in speech recognition on this channel.
    onboarding_keyterms: list[str] = Field(
        default_factory=lambda: [
            "Metformin",
            "Glimepiride",
            "Amlodipine",
            "Telmisartan",
            "Losartan",
            "Atorvastatin",
            "Rosuvastatin",
            "Thyroxine",
            "Pantoprazole",
            "Aspirin",
            "Clopidogrel",
            "Insulin",
            "Dolo 650",
            "Crocin",
            "Glycomet",
            "Telma",
            "Thyronorm",
            "Ecosprin",
            "Pan 40",
            "Shelcal",
        ]
    )

    # --- Exotel (patient alerts: WhatsApp messages and voice calls) -----------
    # Kept apart from sign-in on purpose: nothing here knows about user sessions.
    # API credentials from https://my.exotel.com/apisettings/site#api-credentials
    exotel_api_key: SecretStr | None = None
    exotel_api_token: SecretStr | None = None
    exotel_account_sid: str | None = None
    # Mumbai cluster. Singapore is api.exotel.com.
    exotel_subdomain: str = "api.in.exotel.com"
    # The ExoPhone (virtual number) reminder calls come from.
    exotel_caller_id: str | None = None
    # The Exotel flow (app id) whose Voicebot applet points at /exotel/stream.
    exotel_voice_app_id: str | None = None
    # The WhatsApp Business number messages come from, with country code.
    exotel_whatsapp_from: str | None = None
    # Message templates, approved on Exotel before they can be sent.
    exotel_dose_template: str = "clinexsa_dose_reminder"
    exotel_low_stock_template: str = "clinexsa_low_stock"
    exotel_template_language: str = "en"
    exotel_call_time_limit_s: int = Field(default=180, ge=30, le=14400)
    exotel_ring_timeout_s: int = Field(default=30, ge=10, le=120)
    # What Exotel must present to us. The Voicebot applet URL carries the first two as
    # wss://<username>:<password>@host/exotel/stream; status callbacks carry ?key=<key>.
    exotel_stream_username: str | None = None
    exotel_stream_password: SecretStr | None = None
    exotel_callback_key: SecretStr | None = None

    # --- Supabase (server key, for the alerts worker only) ---------------------
    # The service-role key bypasses row-level security. It lives here and nowhere else:
    # never in the website, never in a browser.
    supabase_url: str | None = None
    supabase_service_role_key: SecretStr | None = None
    # The website's publishable key, the same one every browser already has. With
    # supabase_url it lets POST /web/session check that the caller is signed in to the
    # website. Required in production: without it nobody can start a session there.
    supabase_publishable_key: SecretStr | None = None

    # --- Alerts worker ---------------------------------------------------------
    # Off until Exotel and Supabase are configured and message templates are approved.
    alerts_enabled: bool = False
    # Bearer token the scheduler presents to POST /jobs/alerts/run.
    alerts_jobs_token: SecretStr | None = None
    alerts_timezone: str = "Asia/Kolkata"
    # Run the worker on a timer inside this server. Set false to drive
    # POST /jobs/alerts/run from an external cron instead.
    alerts_scheduler: bool = True
    # Hour of the day (in ALERTS_TIMEZONE) after which low-stock messages go out, once.
    alerts_low_stock_hour: int = Field(default=10, ge=0, le=23)
    # A reminder is due if its time fell within this many minutes before now.
    alerts_window_min: int = Field(default=5, ge=1, le=30)

    # --- Voice session -----------------------------------------------------
    stt_max_reconnect_attempts: int = Field(default=3, ge=0)
    audio_queue_max_frames: int = Field(default=1500, ge=50)  # ~30 s of 20 ms frames

    # --- Knowledge base / RAG ----------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Chunk budget in embedding-model tokens (bge-small max is 512, incl. context header).
    chunk_target_tokens: int = Field(default=350, ge=64, le=480)
    chunk_min_tokens: int = Field(default=60, ge=0)
    chunk_overlap_tokens: int = Field(default=50, ge=0)
    # Consecutive sibling sections smaller than this are merged into one chunk.
    chunk_section_merge_tokens: int = Field(default=150, ge=0)
    # BGE retrieval models embed short queries with this instruction; passages get none.
    embedding_query_instruction: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_device: str | None = None  # None = auto (mps / cuda / cpu)

    # --- Reranking -------------------------------------------------------------
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = Field(default=16, ge=1)
    reranker_top_k: int = Field(default=5, ge=1)

    # --- Qdrant --------------------------------------------------------------
    # QDRANT_URL set   -> Qdrant Cloud / server.  Unset -> embedded local index under
    # data/indexes/qdrant (same client API; one process at a time; fine for development).
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "clinexa_who_primary_care"
    qdrant_timeout_s: int = Field(default=30, ge=1)
    qdrant_local_path: Path = REPO_ROOT / "data" / "indexes" / "qdrant"

    # --- Guidelines in a call (app/rag/retrieval/service.py) --------------------
    # Let Claude search the WHO knowledge base while it is on a call: dense + BM25, RRF,
    # then the cross-encoder. Needs the chunks (make ingest) and the index (make index).
    guidelines_retrieval: bool = True
    # A caller is waiting while this runs. Past it, Claude is told nothing was found.
    guidelines_timeout_s: float = Field(default=4.0, gt=0)

    # --- Medicines catalogue (app/medicines) -------------------------------------
    # The Qdrant collection that holds the Indian medicines catalogue: Jan Aushadhi, the
    # National List of Essential Medicines 2022 and the A to Z medicines dataset of India.
    medicines_collection: str = "clinexa_medicines"
    # Look a medicine's name up in it when a caller says one. Needs QDRANT_URL: the
    # embedded local index is too slow for a quarter of a million names.
    medicines_lookup: bool = True
    # A caller is waiting while this runs. Past it, Claude is told the name was not found.
    medicines_lookup_timeout_s: float = Field(default=1.5, gt=0)
    # The bi-encoder and the cross-encoder of the catalogue (app/medicines/encoders.py).
    # Off by default: measured on the whole catalogue they did not change which medicine
    # is found, and cost 50 to 100 ms a lookup (docs/rag.md, section 3). With this false a
    # name is found by its spelling and sound alone, and `make medicines` stores no dense
    # vectors. Turning it on means indexing the catalogue again with --recreate.
    medicines_encoders: bool = False
    # Embeds every name when the catalogue is indexed, and each name that is looked up.
    # Change it and the catalogue has to be indexed again (make medicines ARGS="--recreate").
    medicines_bi_encoder: str = "BAAI/bge-small-en-v1.5"
    # Put before what was heard, not before the catalogue's names. BGE models ask for it.
    medicines_query_instruction: str = "Represent this sentence for searching relevant passages: "
    # Orders names the lookup's rules hold equal. Empty: no cross-encoder.
    medicines_cross_encoder: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many names the bi-encoder may add to the 64 the spelling-and-sound search finds.
    medicines_dense_candidates: int = Field(default=16, ge=0, le=64)

    # --- Conversation ------------------------------------------------------
    emergency_number_phrase: str = "your local emergency number"
    call_greeting: str = (
        "Hello, you've reached Clinexa, an automated health information assistant. "
        "I'm not a doctor and can't diagnose conditions. "
        "If this is an emergency, please hang up and call {emergency}. "
        "How can I help you today?"
    )
    # Spoken when the reply could not be generated (LLM error, timeout or refusal).
    reply_fallback: str = (
        "I'm sorry, I'm having trouble answering right now. "
        "If this is urgent, please contact a clinician or call {emergency}."
    )

    @field_validator(
        "qdrant_url",
        "qdrant_api_key",
        "embedding_device",
        "anthropic_api_key",
        "logfire_token",
        "llm_effort",
        "exotel_api_key",
        "exotel_api_token",
        "exotel_account_sid",
        "exotel_caller_id",
        "exotel_voice_app_id",
        "exotel_whatsapp_from",
        "exotel_stream_username",
        "exotel_stream_password",
        "exotel_callback_key",
        "supabase_url",
        "supabase_service_role_key",
        "supabase_publishable_key",
        "alerts_jobs_token",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("public_base_url", "supabase_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str | None) -> str | None:
        return v.rstrip("/") if v else v

    @property
    def greeting_text(self) -> str:
        return self.call_greeting.format(emergency=self.emergency_number_phrase)

    @property
    def reply_fallback_text(self) -> str:
        return self.reply_fallback.format(emergency=self.emergency_number_phrase)


@lru_cache
def get_settings() -> Settings:
    return Settings()
