from __future__ import annotations

from app.core.config import Settings
from app.core.logging import get_logger
from app.voice.providers.base import STTProvider
from app.voice.providers.deepgram_stt import DeepgramSTTProvider

log = get_logger(__name__)


def build_stt_provider(settings: Settings) -> STTProvider | None:
    if settings.deepgram_api_key is None:
        log.warning("stt.not_configured", hint="set DEEPGRAM_API_KEY")
        return None
    return DeepgramSTTProvider(
        settings.deepgram_api_key.get_secret_value(),
        url=settings.deepgram_stt_url,
        model=settings.deepgram_stt_model,
        language=settings.deepgram_language,
        endpointing_ms=settings.deepgram_endpointing_ms,
        utterance_end_ms=settings.deepgram_utterance_end_ms,
        keyterms=settings.deepgram_keyterms,
    )
