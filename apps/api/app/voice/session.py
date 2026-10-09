<<<<<<< HEAD
"""A single live conversation: audio in -> STT -> reply (LLM -> TTS) -> audio out.
=======
"""A single live phone call: Twilio audio in -> STT -> reply (LLM + lookups -> TTS) -> Twilio.
>>>>>>> 5ed78ec (added the rag query for medicine catalogue)

The WebSocket handler only parses frames and feeds audio here; STT runs in its
own task so a slow provider never blocks frame reads. Audio is buffered in a
bounded queue, which also preserves audio while STT reconnects.

Each caller utterance is answered in a separate reply task: reply text streams
from the LLM, is cut into sentences, and each sentence is synthesised and sent
<<<<<<< HEAD
through the session's transport (a Twilio phone call or a browser tab) while
the rest is still being written.
=======
to Twilio while the rest is still being written. What the LLM looked up to write
it (a medicine's name, WHO guidance) is kept with the call.
>>>>>>> 5ed78ec (added the rag query for medicine catalogue)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from app.agents.responder import ReplyError, ReplyGenerator, ReplyTrace
from app.core.config import Settings
from app.core.logging import get_logger
from app.graph.state import AgentName, ConversationMessage, Lookup, VoiceClinicalState
from app.observability.metrics import LatencyMetrics, LatencyStats
from app.schemas.clinical import Evidence, Intent
from app.voice.providers.base import (
    TWILIO_AUDIO_FORMAT,
    AudioFormat,
    STTError,
    STTProvider,
    TranscriptEvent,
    TranscriptEventType,
    TTSError,
    TTSProvider,
)
from app.voice.sentences import SentenceChunker
from app.voice.transport import FrameSender, Transport, TwilioTransport
from app.voice.turns import Utterance, UtteranceAssembler

log = get_logger(__name__)

UtteranceHandler = Callable[["CallSession", Utterance], Awaitable[None]]

_STT_DRAIN_TIMEOUT_S = 6.0
# After a reply is cut off by speech that turns out to be noise (no words follow),
# the assistant picks the conversation back up rather than leave a silence.
_BARGE_IN_RECOVERY_S = 6.0


@dataclass
class _ReplyTurn:
    """Progress of one assistant reply, shared by its LLM and TTS halves."""

    heard_at: float  # monotonic time the caller's utterance was complete
    endpoint_latency_ms: float | None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    sentences: list[str] = field(default_factory=list)
    first_sentence_at: float | None = None
    audio_started: bool = False
    fallback: bool = False
    trace: ReplyTrace = field(default_factory=ReplyTrace)


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
    # What the assistant looked up, and the guidance passages its replies drew on.
    lookups: list[Lookup]
    evidence: list[Evidence]
    audio_frames_received: int
    audio_frames_dropped: int
    stt_reconnects: int
    replies_spoken: int
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
        audio_format: AudioFormat | None = None,
        on_utterance: UtteranceHandler | None = None,
        responder: ReplyGenerator | None = None,
        tts: TTSProvider | None = None,
        send: FrameSender | None = None,
        transport: Transport | None = None,
        fallback_text: str | None = None,
    ) -> None:
        # A bare Twilio frame sender is still accepted and wrapped.
        if transport is None and send is not None:
            transport = TwilioTransport(stream_sid, send)
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.caller = caller
        self.audio_format = audio_format or (
            transport.input_format if transport is not None else TWILIO_AUDIO_FORMAT
        )
        self.state = VoiceClinicalState(call_id=call_sid)
        self.metrics = LatencyMetrics()
        self.status = CallStatus.ACTIVE
        self.started_at = datetime.now(UTC)
        self.ended_at: datetime | None = None

        self._stt = stt
        self._settings = settings
        self._on_utterance = on_utterance
        # Replies need all three; without them the call is transcribed only.
        self._responder = responder
        self._tts = tts
        self._transport = transport
        # Spoken when a reply cannot be generated.
        self._fallback_text = fallback_text or settings.reply_fallback_text
        self._reply_task: asyncio.Task[None] | None = None
        self._reply_due: tuple[float, float | None] | None = None
        self.replies_spoken = 0
        # Monotonic time until which audio already sent is still playing out.
        self._playback_until = 0.0
        self._recovery_task: asyncio.Task[None] | None = None
        self.utterances_heard = 0
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
        log.info(
            "call.started",
            caller=self.caller,
            stt_provider=self._stt.name,
            replies_enabled=self.replies_enabled,
        )

    @property
    def replies_enabled(self) -> bool:
        return self._responder is not None and self._tts is not None and self._transport is not None

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
        # The caller has gone: stop talking before flushing what they said last.
        if self._recovery_task is not None:
            self._recovery_task.cancel()
        if self._reply_task is not None:
            self._reply_task.cancel()
            await asyncio.gather(self._reply_task, return_exceptions=True)
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
        # Twilio echoes a reply's mark once its audio has finished playing.
        # Phase 3 uses this to know whether the assistant is still speaking.
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
            lookups=list(self.state.lookups),
            evidence=list(self.state.evidence),
            audio_frames_received=self.frames_received,
            audio_frames_dropped=self.frames_dropped,
            stt_reconnects=self.stt_reconnects,
            replies_spoken=self.replies_spoken,
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
                log.debug("stt.speech_started", audio_s=event.audio_start_s)
                await self._barge_in()
            case TranscriptEventType.INTERIM:
                pending = self._assembler.pending_text
                self.state.current_transcript = f"{pending} {event.text}".strip()
                await self._send_live_transcript()
            case TranscriptEventType.FINAL:
                if event.finalization_latency_ms is not None:
                    self.metrics.record("stt_finalization_ms", event.finalization_latency_ms)
                if not event.speech_final:
                    pending = self._assembler.pending_text
                    self.state.current_transcript = f"{pending} {event.text}".strip()
                    await self._send_live_transcript()
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
        self.utterances_heard += 1
        if self._transport is not None:
            await self._emit(self._transport.send_transcript("user", utterance.text, final=True))
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
        if self.replies_enabled and not self._closed:
            self._request_reply(utterance)

    # --- replies -----------------------------------------------------------

    def request_reply(self) -> None:
        """Ask for a reply with no new utterance: the opening turn, or after a pause."""
        if self.replies_enabled and not self._closed:
            self._schedule_reply(None)

    async def interrupt(self) -> None:
        """Stop the reply being spoken and drop audio that has not played yet."""
        task = self._reply_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._playback_until = 0.0
        if self._transport is not None:
            await self._emit(self._transport.clear_audio())
            await self._emit(self._transport.send_status("listening"))

    def _request_reply(self, utterance: Utterance) -> None:
        self._schedule_reply(utterance.endpoint_latency_ms)

    def _schedule_reply(self, endpoint_latency_ms: float | None) -> None:
        # Replies are spoken one at a time. If the caller says more while one is
        # being spoken, a single further reply covers everything said since.
        self._reply_due = (time.monotonic(), endpoint_latency_ms)
        if self._reply_task is None or self._reply_task.done():
            self._reply_task = asyncio.create_task(
                self._reply_loop(), name=f"reply-{self.call_sid}"
            )

    async def _reply_loop(self) -> None:
        while (due := self._reply_due) is not None:
            self._reply_due = None
            try:
                await self._reply(_ReplyTurn(heard_at=due[0], endpoint_latency_ms=due[1]))
            except Exception:
                log.exception("reply.failed")
        if self._transport is not None:
            await self._emit(self._transport.send_status("listening"))

    async def _barge_in(self) -> None:
        """The person started talking: cut off a reply they can still hear."""
        if self._transport is None or not self._transport.supports_barge_in:
            return
        if time.monotonic() >= self._playback_until:
            return  # nothing is playing (a reply still being written is left alone)
        log.info("call.barge_in")
        await self.interrupt()
        if self._recovery_task is not None:
            self._recovery_task.cancel()
        self._recovery_task = asyncio.create_task(
            self._recover_from_barge_in(self.utterances_heard), name=f"recover-{self.call_sid}"
        )

    async def _recover_from_barge_in(self, utterances_before: int) -> None:
        await asyncio.sleep(_BARGE_IN_RECOVERY_S)
        if self._closed or self.utterances_heard != utterances_before:
            return
        if self.state.current_transcript or self._assembler.pending_text:
            return  # they are mid-sentence; their utterance will get the reply
        if self._responder is not None:
            self._responder.add_note(
                "You were cut off by a noise but nobody spoke. Say your last question "
                "again, briefly."
            )
        self.request_reply()

    async def _send_live_transcript(self) -> None:
        if self._transport is not None and self.state.current_transcript:
            await self._emit(
                self._transport.send_transcript("user", self.state.current_transcript, final=False)
            )

    async def _emit(self, event: Coroutine[Any, Any, None]) -> None:
        # Transcript and status events are best-effort: a socket that has gone
        # away must not take the STT or reply task down with it.
        try:
            await event
        except Exception as exc:
            log.debug("transport.event_dropped", error=repr(exc))

    async def _reply(self, turn: _ReplyTurn) -> None:
        history = self.state.conversation_history
        position = len(history)  # the reply belongs here even if the caller speaks during it
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        if self._transport is not None:
            await self._emit(self._transport.send_status("thinking"))
        # The LLM runs in its own task so it is already generating while the TTS
        # connection is being opened.
        generator = asyncio.create_task(
            self._generate(list(history), sentences, turn), name=f"llm-{self.call_sid}"
        )
        completed = False
        try:
            await self._speak(sentences, turn)
            completed = True
        except TTSError as exc:
            log.error("reply.tts_failed", error=str(exc), audio_started=turn.audio_started)
        finally:
            generator.cancel()
            await asyncio.gather(generator, return_exceptions=True)
            self._record_lookups(turn.trace)
            # Only what the caller could actually hear goes into the transcript.
            if turn.audio_started and turn.sentences:
                text = " ".join(turn.sentences)
                history.insert(
                    position,
                    ConversationMessage(
                        role="assistant",
                        content=text,
                        timestamp=turn.started_at,
                        interrupted=not completed,
                    ),
                )
                self.state.response_text = text
                self.replies_spoken += 1
                log.info(
                    "call.reply",
                    text=text if self._settings.log_transcripts else "<redacted>",
                    chars=len(text),
                    sentences=len(turn.sentences),
                    fallback=turn.fallback,
                    interrupted=not completed,
                )

    async def _generate(
        self,
        history: list[ConversationMessage],
        sentences: asyncio.Queue[str | None],
        turn: _ReplyTurn,
    ) -> None:
        assert self._responder is not None
        chunker = SentenceChunker()
        started = time.monotonic()
        first_text = True

        def emit(sentence: str) -> None:
            if turn.first_sentence_at is None:
                turn.first_sentence_at = time.monotonic()
                if not turn.fallback:
                    self.metrics.record("llm_first_sentence_ms", _ms_since(started))
            turn.sentences.append(sentence)
            sentences.put_nowait(sentence)

        try:
            async for delta in self._responder.stream_reply(history, turn.trace):
                if first_text and delta:
                    first_text = False
                    self.metrics.record("llm_ttft_ms", _ms_since(started))
                for sentence in chunker.feed(delta):
                    emit(sentence)
            if (rest := chunker.flush()) is not None:
                emit(rest)
            self.metrics.record("llm_total_ms", _ms_since(started))
            if not turn.sentences:
                raise ReplyError("model returned no text")
        except ReplyError as exc:
            log.error("reply.llm_failed", error=str(exc), sentences_spoken=len(turn.sentences))
            # Never leave a caller in silence. A reply that already started is
            # simply cut short; otherwise say that something went wrong.
            if not turn.sentences:
                turn.fallback = True
                emit(self._fallback_text)
                if self._transport is not None:
                    await self._emit(
                        self._transport.send_error("reply_failed", "The reply could not be made.")
                    )
        finally:
            sentences.put_nowait(None)

    async def _speak(self, sentences: asyncio.Queue[str | None], turn: _ReplyTurn) -> None:
        assert self._tts is not None and self._transport is not None
        transport = self._transport
        bytes_per_second = transport.output_format.bytes_per_second

        async def text() -> AsyncIterator[str]:
            while (sentence := await sentences.get()) is not None:
                await self._emit(transport.send_transcript("assistant", sentence, final=True))
                yield sentence

        audio = self._tts.synthesize_stream(text(), transport.output_format)
        try:
            async for chunk in audio:
                if not turn.audio_started:
                    turn.audio_started = True
                    self._record_first_audio(turn)
                    await self._emit(transport.send_status("speaking"))
                await transport.send_audio(chunk)
                # Sent audio plays out in real time on the other side.
                self._playback_until = (
                    max(self._playback_until, time.monotonic()) + len(chunk) / bytes_per_second
                )
        finally:
            # Hang-up cancels this task between chunks; close the provider stream now
            # rather than whenever the generator is garbage-collected.
            if isinstance(audio, AsyncGenerator):
                await audio.aclose()
        if turn.audio_started:
            await transport.end_of_reply(f"reply-{self.replies_spoken + 1}")

    def _record_lookups(self, trace: ReplyTrace) -> None:
        # Kept even if the reply was never heard: the lookups still happened.
        self.state.lookups.extend(trace.lookups)
        self.state.evidence.extend(trace.evidence)
        for lookup in trace.lookups:
            self.metrics.record(f"{lookup.tool}_ms", lookup.duration_ms)

    def _record_first_audio(self, turn: _ReplyTurn) -> None:
        now = time.monotonic()
        response_ms = round((now - turn.heard_at) * 1000, 2)
        self.metrics.record("response_latency_ms", response_ms)
        if turn.first_sentence_at is not None:
            self.metrics.record("tts_ttfa_ms", round((now - turn.first_sentence_at) * 1000, 2))
        if turn.endpoint_latency_ms is not None:
            self.metrics.record(
                "voice_to_voice_ms", round(turn.endpoint_latency_ms + response_ms, 2)
            )


def _ms_since(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 2)
