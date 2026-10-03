"""Provider-agnostic STT/TTS interfaces.

The voice session depends only on these abstractions, so Deepgram can be
swapped for another vendor without touching the call pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class AudioFormat(BaseModel):
    encoding: Literal["mulaw", "linear16"] = "mulaw"
    sample_rate: int = 8000
    channels: int = 1

    @property
    def bytes_per_second(self) -> int:
        bytes_per_sample = 1 if self.encoding == "mulaw" else 2
        return self.sample_rate * self.channels * bytes_per_sample


TWILIO_AUDIO_FORMAT = AudioFormat(encoding="mulaw", sample_rate=8000, channels=1)


class TranscriptEventType(StrEnum):
    INTERIM = "interim"  # partial hypothesis; may still change
    FINAL = "final"  # finalized segment; speech_final marks end of the caller's turn
    SPEECH_STARTED = "speech_started"  # VAD detected caller speech (barge-in trigger)
    UTTERANCE_END = "utterance_end"  # word-gap based end of utterance
    METADATA = "metadata"


class TranscriptEvent(BaseModel):
    type: TranscriptEventType
    text: str = ""
    speech_final: bool = False
    confidence: float | None = None
    # Position in the provider's audio timeline (seconds since stream start).
    audio_start_s: float | None = None
    audio_end_s: float | None = None
    # Audio time at which the last recognised word ended.
    last_word_end_s: float | None = None
    # Wall-clock latency from sending the audio covered by this event to receiving it.
    finalization_latency_ms: float | None = None
    # Wall-clock latency from sending the caller's last word to this event.
    endpoint_latency_ms: float | None = None
    provider_request_id: str | None = None


class STTError(Exception):
    """Raised when the STT stream fails; the session may reconnect."""


class STTConnectionError(STTError):
    """Raised when a connection to the STT provider cannot be established."""


class STTProvider(ABC):
    name: str

    @abstractmethod
    def transcribe_stream(
        self, audio: AsyncIterator[bytes], audio_format: AudioFormat
    ) -> AsyncIterator[TranscriptEvent]:
        """Stream ``audio`` to the provider and yield transcript events.

        The stream ends after ``audio`` is exhausted and the provider has
        flushed its remaining results. Raises ``STTError`` on failure.
        """


class TTSError(Exception):
    """Raised when speech synthesis fails; the reply cannot be spoken."""


class TTSConnectionError(TTSError):
    """Raised when a connection to the TTS provider cannot be established."""


class TTSProvider(ABC):
    """Streaming text-to-speech."""

    name: str

    @abstractmethod
    def synthesize_stream(
        self, text: AsyncIterator[str], audio_format: AudioFormat
    ) -> AsyncIterator[bytes]:
        """Yield encoded audio chunks as soon as each text chunk is synthesised.

        ``text`` yields speakable chunks (sentences). The stream ends once ``text``
        is exhausted and all of its audio has been yielded. Raises ``TTSError``.
        """
