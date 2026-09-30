"""A single live phone call: Twilio audio in -> streaming STT -> caller utterances.

The WebSocket handler only parses Twilio frames and feeds audio here; STT runs
in its own task so a slow provider never blocks frame reads. Audio is buffered
in a bounded queue, which also preserves audio while STT reconnects.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel

from app.core.config import Settings
from app.core.logging import get_logger
from app.graph.state import AgentName, ConversationMessage, VoiceClinicalState
from app.observability.metrics import LatencyMetrics, LatencyStats
from app.schemas.clinical import Intent
from app.voice.providers.base import (
    TWILIO_AUDIO_FORMAT,
    AudioFormat,
    STTError,
    STTProvider,
    TranscriptEvent,
    TranscriptEventType,
)
from app.voice.turns import Utterance, UtteranceAssembler

log = get_logger(__name__)

UtteranceHandler = Callable[["CallSession", Utterance], Awaitable[None]]

_STT_DRAIN_TIMEOUT_S = 6.0


class CallStatus(StrEnum):
    ACTIVE = "active"
    STT_UNAVAILABLE = "stt_unavailable"
    ENDED = "ended"


class CallSnapshot(BaseModel):
    """Read-only view of a call for the API/dashboard."""

    call_sid: str
    stream_sid: str
    caller: str | None
    status: CallStatus
    started_at: datetime
    ended_at: datetime | None
    duration_s: float
    current_intent: Intent | None
    current_agent: AgentName | None
    escalation_required: bool
    live_transcript: str
    transcript: list[ConversationMessage]
    audio_frames_received: int
    audio_frames_dropped: int
    stt_reconnects: int
    latency: dict[str, LatencyStats]


class CallSession:
    def __init__(
        self,
        *,
        call_sid: str,
        stream_sid: str,
        stt: STTProvider,
        settings: Settings,
        caller: str | None = None,
        audio_format: AudioFormat = TWILIO_AUDIO_FORMAT,
        on_utterance: UtteranceHandler | None = None,
    ) -> None:
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.caller = caller
        self.audio_format = audio_format
        self.state = VoiceClinicalState(call_id=call_sid)
        self.metrics = LatencyMetrics()
        self.status = CallStatus.ACTIVE
        self.started_at = datetime.now(UTC)
        self.ended_at: datetime | None = None

        self._stt = stt
        self._settings = settings
        self._on_utterance = on_utterance
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=settings.audio_queue_max_frames
        )
        self._assembler = UtteranceAssembler()
        self._stt_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._audio_exhausted = False
        self._closed = False
        self._t0 = time.monotonic()

        self.frames_received = 0
        self.frames_dropped = 0
        self.stt_reconnects = 0

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._stt_task = asyncio.create_task(self._run_stt(), name=f"stt-{self.call_sid}")
        log.info("call.started", caller=self.caller, stt_provider=self._stt.name)

    def feed_audio(self, chunk: bytes) -> None:
        if self._closed or self.status is CallStatus.STT_UNAVAILABLE:
            return
        self.frames_received += 1
        try:
            self._audio_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            self.frames_dropped += 1
            if self.frames_dropped % 50 == 1:
                log.warning("call.audio_dropped", dropped=self.frames_dropped)

    async def close(self) -> None:
        """End of call: flush STT so trailing words are not lost, then finalize.

        Runs as a shielded task, so a cancelled caller (client disconnect, server
        shutdown) cannot leave the call half-closed; later callers await the same task.
        """
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._finalize(), name=f"close-{self.call_sid}")
        await asyncio.shield(self._close_task)

    async def _finalize(self) -> None:
        self._enqueue_end_of_audio()

        if self._stt_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._stt_task), _STT_DRAIN_TIMEOUT_S)
            except TimeoutError:
                log.warning("call.stt_drain_timeout")
                self._stt_task.cancel()
                await asyncio.gather(self._stt_task, return_exceptions=True)
            except Exception:
                log.exception("call.stt_task_failed")

        if (utterance := self._assembler.flush()) is not None:
            await self._handle_utterance(utterance)

        self.status = CallStatus.ENDED
        self.ended_at = datetime.now(UTC)
        log.info(
            "call.ended",
            duration_s=round(time.monotonic() - self._t0, 2),
            turns=len(self.state.conversation_history),
            frames_received=self.frames_received,
            frames_dropped=self.frames_dropped,
            latency={k: v.model_dump() for k, v in self.metrics.summary().items()},
        )

    def on_mark(self, name: str) -> None:
        # Phase 2/3: marks confirm when queued TTS audio finished playing.
        log.debug("call.mark", name=name)

    def snapshot(self) -> CallSnapshot:
        end = self.ended_at or datetime.now(UTC)
        return CallSnapshot(
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            caller=self.caller,
            status=self.status,
            started_at=self.started_at,
            ended_at=self.ended_at,
            duration_s=round((end - self.started_at).total_seconds(), 2),
            current_intent=self.state.intent,
            current_agent=self.state.current_agent,
            escalation_required=self.state.escalation_required,
            live_transcript=self.state.current_transcript,
            transcript=list(self.state.conversation_history),
            audio_frames_received=self.frames_received,
            audio_frames_dropped=self.frames_dropped,
            stt_reconnects=self.stt_reconnects,
            latency=self.metrics.summary(),
        )

    # --- STT ---------------------------------------------------------------

    def _enqueue_end_of_audio(self) -> None:
        while True:
            try:
                self._audio_queue.put_nowait(None)
                return
            except asyncio.QueueFull:
                self._audio_queue.get_nowait()
                self.frames_dropped += 1

    async def _audio_chunks(self) -> AsyncIterator[bytes]:
        # A fresh generator per STT connection: cancelling one on failure must
        # not close the iterator the reconnected stream reads from.
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                self._audio_exhausted = True
                return
            yield chunk

    async def _run_stt(self) -> None:
        failures = 0
        while True:
            try:
                async for event in self._stt.transcribe_stream(
                    self._audio_chunks(), self.audio_format
                ):
                    failures = 0
                    await self._on_transcript_event(event)
                return
            except STTError as exc:
                if self._audio_exhausted:
                    log.warning("stt.failed_during_flush", error=str(exc))
                    return
                failures += 1
                if failures > self._settings.stt_max_reconnect_attempts:
                    log.error("stt.unavailable", error=str(exc), attempts=failures)
                    self.status = CallStatus.STT_UNAVAILABLE
                    return
                self.stt_reconnects += 1
                backoff = min(0.25 * 2**failures, 2.0)
                log.warning("stt.reconnecting", error=str(exc), attempt=failures, backoff_s=backoff)
                await asyncio.sleep(backoff)

    async def _on_transcript_event(self, event: TranscriptEvent) -> None:
        match event.type:
            case TranscriptEventType.SPEECH_STARTED:
                # Phase 3: barge-in hook (stop TTS, clear Twilio buffer, cancel turn).
                log.debug("stt.speech_started", audio_s=event.audio_start_s)
            case TranscriptEventType.INTERIM:
                pending = self._assembler.pending_text
                self.state.current_transcript = f"{pending} {event.text}".strip()
            case TranscriptEventType.FINAL:
                if event.finalization_latency_ms is not None:
                    self.metrics.record("stt_finalization_ms", event.finalization_latency_ms)
            case TranscriptEventType.METADATA:
                log.info("stt.metadata", request_id=event.provider_request_id)

        if (utterance := self._assembler.process(event)) is not None:
            await self._handle_utterance(utterance)

    async def _handle_utterance(self, utterance: Utterance) -> None:
        if utterance.endpoint_latency_ms is not None:
            self.metrics.record("stt_endpoint_ms", utterance.endpoint_latency_ms)
        self.state.conversation_history.append(
            ConversationMessage(
                role="patient", content=utterance.text, stt_confidence=utterance.confidence
            )
        )
        self.state.current_transcript = ""
        log.info(
            "call.utterance",
            text=utterance.text if self._settings.log_transcripts else "<redacted>",
            chars=len(utterance.text),
            confidence=utterance.confidence,
            end_reason=utterance.end_reason,
            endpoint_latency_ms=utterance.endpoint_latency_ms,
        )
        if self._on_utterance is not None:
            await self._on_utterance(self, utterance)
