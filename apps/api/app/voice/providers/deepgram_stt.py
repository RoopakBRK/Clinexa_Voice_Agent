"""Deepgram streaming STT over the raw WebSocket API.

Talking to ``/v1/listen`` directly (rather than through the SDK) keeps full
control over connection lifecycle, keep-alives and the VAD/endpointing events
the barge-in logic depends on.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

from app.core.logging import get_logger
from app.voice.audio import AudioClock
from app.voice.providers.base import (
    AudioFormat,
    STTConnectionError,
    STTError,
    STTProvider,
    TranscriptEvent,
    TranscriptEventType,
)

log = get_logger(__name__)

# Deepgram closes idle streams after ~10 s without audio.
_KEEPALIVE_INTERVAL_S = 4.0
_FLUSH_TIMEOUT_S = 5.0


class DeepgramSTTProvider(STTProvider):
    name = "deepgram"

    def __init__(
        self,
        api_key: str,
        *,
        url: str = "wss://api.deepgram.com/v1/listen",
        model: str = "nova-3",
        language: str = "en-US",
        endpointing_ms: int = 300,
        utterance_end_ms: int = 1000,
        keyterms: list[str] | None = None,
        connect_timeout_s: float = 5.0,
    ) -> None:
        self._api_key = api_key
        self._url = url
        self._model = model
        self._language = language
        self._endpointing_ms = endpointing_ms
        self._utterance_end_ms = utterance_end_ms
        self._keyterms = keyterms or []
        self._connect_timeout_s = connect_timeout_s

    def build_url(self, audio_format: AudioFormat) -> str:
        params: list[tuple[str, str | int]] = [
            ("model", self._model),
            ("language", self._language),
            ("encoding", audio_format.encoding),
            ("sample_rate", audio_format.sample_rate),
            ("channels", audio_format.channels),
            ("interim_results", "true"),
            ("endpointing", self._endpointing_ms),
            ("utterance_end_ms", self._utterance_end_ms),
            ("vad_events", "true"),
            ("smart_format", "true"),
            ("punctuate", "true"),
        ]
        params += [("keyterm", term) for term in self._keyterms]
        return f"{self._url}?{urlencode(params)}"

    async def transcribe_stream(
        self, audio: AsyncIterator[bytes], audio_format: AudioFormat
    ) -> AsyncIterator[TranscriptEvent]:
        clock = AudioClock(audio_format.bytes_per_second)
        try:
            ws = await connect(
                self.build_url(audio_format),
                additional_headers={"Authorization": f"Token {self._api_key}"},
                open_timeout=self._connect_timeout_s,
                max_size=2**20,
            )
        except InvalidStatus as exc:
            status = exc.response.status_code
            raise STTConnectionError(f"Deepgram rejected connection (HTTP {status})") from exc
        except (OSError, TimeoutError, InvalidHandshake) as exc:
            raise STTConnectionError(f"Could not connect to Deepgram: {exc!r}") from exc

        request_id = ws.response.headers.get("dg-request-id") if ws.response else None
        log.info("deepgram.connected", request_id=request_id, model=self._model)

        sender = asyncio.create_task(self._send_audio(ws, audio, clock), name="dg-send")
        keepalive = asyncio.create_task(self._keepalive(ws, clock), name="dg-keepalive")
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                event = self._parse_message(raw, clock)
                if event is not None:
                    yield event
        except ConnectionClosed as exc:
            raise STTError(f"Deepgram connection closed unexpectedly: {exc}") from exc
        finally:
            keepalive.cancel()
            sender.cancel()
            await asyncio.gather(keepalive, sender, return_exceptions=True)
            await ws.close()

        if sender.done() and not sender.cancelled() and (exc := sender.exception()):
            raise STTError(f"Deepgram audio sender failed: {exc!r}") from exc
        if ws.close_code not in (None, 1000):
            raise STTError(f"Deepgram closed stream: code={ws.close_code} {ws.close_reason!r}")

    async def _send_audio(
        self, ws: ClientConnection, audio: AsyncIterator[bytes], clock: AudioClock
    ) -> None:
        async for chunk in audio:
            if not chunk:
                continue
            await ws.send(chunk)
            clock.record(len(chunk))
        # Ask Deepgram to flush remaining results and close. If it doesn't
        # within the timeout, close from our side so the receive loop ends.
        await ws.send(json.dumps({"type": "CloseStream"}))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(ws.wait_closed(), _FLUSH_TIMEOUT_S)
        await ws.close()

    async def _keepalive(self, ws: ClientConnection, clock: AudioClock) -> None:
        last_seen = clock.seconds_sent
        while True:
            await asyncio.sleep(_KEEPALIVE_INTERVAL_S)
            if clock.seconds_sent == last_seen:
                with contextlib.suppress(ConnectionClosed):
                    await ws.send(json.dumps({"type": "KeepAlive"}))
            last_seen = clock.seconds_sent

    def _parse_message(self, raw: str, clock: AudioClock) -> TranscriptEvent | None:
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("deepgram.invalid_json", raw=raw[:200])
            return None

        now = time.monotonic()
        match data.get("type"):
            case "Results":
                return self._parse_results(data, clock, now)
            case "SpeechStarted":
                return TranscriptEvent(
                    type=TranscriptEventType.SPEECH_STARTED,
                    audio_start_s=data.get("timestamp"),
                )
            case "UtteranceEnd":
                last_word_end = data.get("last_word_end")
                return TranscriptEvent(
                    type=TranscriptEventType.UTTERANCE_END,
                    last_word_end_s=last_word_end,
                    endpoint_latency_ms=clock.latency_ms(last_word_end, now),
                )
            case "Metadata":
                return TranscriptEvent(
                    type=TranscriptEventType.METADATA,
                    provider_request_id=data.get("request_id"),
                )
            case "Error":
                log.error("deepgram.error", payload=data)
                return None
            case other:
                log.debug("deepgram.unhandled_message", type=other)
                return None

    @staticmethod
    def _parse_results(data: dict[str, Any], clock: AudioClock, now: float) -> TranscriptEvent:
        alternatives = data.get("channel", {}).get("alternatives") or [{}]
        best = alternatives[0]
        words = best.get("words") or []
        start = data.get("start")
        duration = data.get("duration")
        audio_end = start + duration if start is not None and duration is not None else None
        last_word_end = words[-1].get("end") if words else None
        is_final = bool(data.get("is_final"))
        speech_final = bool(data.get("speech_final"))

        return TranscriptEvent(
            type=TranscriptEventType.FINAL if is_final else TranscriptEventType.INTERIM,
            text=(best.get("transcript") or "").strip(),
            speech_final=speech_final,
            confidence=best.get("confidence"),
            audio_start_s=start,
            audio_end_s=audio_end,
            last_word_end_s=last_word_end,
            finalization_latency_ms=clock.latency_ms(audio_end, now) if is_final else None,
            endpoint_latency_ms=clock.latency_ms(last_word_end, now) if speech_final else None,
        )
