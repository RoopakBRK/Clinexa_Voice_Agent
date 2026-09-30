"""Assembles streaming transcript events into complete caller utterances.

Deepgram finalizes speech in segments (``is_final``). A caller's turn is
complete when a final segment has ``speech_final`` (silence-based endpointing)
or, as a fallback for noisy lines, when an ``UtteranceEnd`` event arrives
(word-gap based). See https://developers.deepgram.com/docs/understand-endpointing-interim-results
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.voice.providers.base import TranscriptEvent, TranscriptEventType


class Utterance(BaseModel):
    text: str
    confidence: float | None
    end_reason: Literal["speech_final", "utterance_end", "stream_closed"]
    endpoint_latency_ms: float | None = None


class UtteranceAssembler:
    def __init__(self) -> None:
        self._segments: list[str] = []
        self._confidences: list[float] = []

    @property
    def pending_text(self) -> str:
        return " ".join(self._segments)

    def process(self, event: TranscriptEvent) -> Utterance | None:
        if event.type is TranscriptEventType.FINAL:
            if event.text:
                self._segments.append(event.text)
                if event.confidence is not None:
                    self._confidences.append(event.confidence)
            if event.speech_final:
                return self._emit("speech_final", event.endpoint_latency_ms)
        elif event.type is TranscriptEventType.UTTERANCE_END:
            return self._emit("utterance_end", event.endpoint_latency_ms)
        return None

    def flush(self) -> Utterance | None:
        return self._emit("stream_closed", None)

    def _emit(
        self,
        reason: Literal["speech_final", "utterance_end", "stream_closed"],
        endpoint_latency_ms: float | None,
    ) -> Utterance | None:
        if not self._segments:
            return None
        confidence = (
            round(sum(self._confidences) / len(self._confidences), 4) if self._confidences else None
        )
        utterance = Utterance(
            text=self.pending_text,
            confidence=confidence,
            end_reason=reason,
            endpoint_latency_ms=endpoint_latency_ms,
        )
        self._segments.clear()
        self._confidences.clear()
        return utterance
