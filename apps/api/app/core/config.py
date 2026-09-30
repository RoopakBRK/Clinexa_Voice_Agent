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
    app_name: str = "CareFlow AI"
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
    # Domain terms to boost recognition (Nova-3 keyterm prompting).
    deepgram_keyterms: list[str] = Field(default_factory=list)

    # --- Voice session -----------------------------------------------------
    stt_max_reconnect_attempts: int = Field(default=3, ge=0)
    audio_queue_max_frames: int = Field(default=1500, ge=50)  # ~30 s of 20 ms frames

    # --- Conversation ------------------------------------------------------
    emergency_number_phrase: str = "your local emergency number"
    call_greeting: str = (
        "Hello, you've reached CareFlow, an automated health information assistant. "
        "I'm not a doctor and can't diagnose conditions. "
        "If this is an emergency, please hang up and call {emergency}. "
        "How can I help you today?"
    )

    @field_validator("public_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str | None) -> str | None:
        return v.rstrip("/") if v else v

    @property
    def greeting_text(self) -> str:
        return self.call_greeting.format(emergency=self.emergency_number_phrase)


@lru_cache
def get_settings() -> Settings:
    return Settings()
