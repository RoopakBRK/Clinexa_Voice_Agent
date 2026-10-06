from __future__ import annotations

from app.core.config import Settings
from app.core.logging import get_logger
from app.voice.providers.base import STTProvider, TTSProvider
from app.voice.providers.deepgram_stt import DeepgramSTTProvider
from app.voice.providers.deepgram_tts import DeepgramTTSProvider

log = get_logger(__name__)


def build_stt_provider(
    settings: Settings, *, language: str | None = None, keyterms: list[str] | None = None
) -> STTProvider | None:
    """``language`` and ``keyterms`` override the settings for one session."""
    if settings.deepgram_api_key is None:
        log.warning("stt.not_configured", hint="set DEEPGRAM_API_KEY")
        return None
    return DeepgramSTTProvider(
        settings.deepgram_api_key.get_secret_value(),
        url=settings.deepgram_stt_url,
        model=settings.deepgram_stt_model,
        language=language or settings.deepgram_language,
        endpointing_ms=settings.deepgram_endpointing_ms,
        utterance_end_ms=settings.deepgram_utterance_end_ms,
        keyterms=settings.deepgram_keyterms if keyterms is None else keyterms,
    )


def build_tts_provider(settings: Settings, *, model: str | None = None) -> TTSProvider | None:
    """``model`` overrides the configured voice for one session."""
    if settings.deepgram_api_key is None:
        log.warning("tts.not_configured", hint="set DEEPGRAM_API_KEY")
        return None
    return DeepgramTTSProvider(
        settings.deepgram_api_key.get_secret_value(),
        url=settings.deepgram_tts_url,
        model=model or settings.deepgram_tts_model,
    )
