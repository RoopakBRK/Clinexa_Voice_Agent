"""Exotel Voicebot WebSocket endpoint: Roopiee on a medicine reminder call.

How a call reaches here:
1. The alerts worker asks Exotel to ring the patient and connect them to a flow.
2. That flow has a Voicebot applet whose URL is
   ``wss://<username>:<password>@<this server>/exotel/stream``.
3. Exotel opens this socket and streams the call audio both ways.

Only calls the worker placed are served: the call's sid must match a delivery in
the database. That is also where Roopiee learns which medicine the call is about.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hmac
from collections.abc import Callable
from typing import cast

import structlog
from fastapi import APIRouter, Depends, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from app.agents.reminder import REMINDER_FALLBACK, build_reminder_generator
from app.agents.responder import ReplyGenerator
from app.core.config import Settings
from app.core.logging import get_logger
from app.notify.exotel import mask
from app.notify.routes import get_alert_store
from app.notify.store import AlertStore, CallContext
from app.voice.deps import get_call_registry, get_settings
from app.voice.exotel_protocol import (
    FRAME_BYTES,
    MIN_CHUNK_BYTES,
    DtmfMessage,
    MarkMessage,
    MediaMessage,
    StartMessage,
    StopMessage,
    outbound_clear,
    outbound_mark,
    outbound_media,
    parse_inbound,
)
from app.voice.languages import LANGUAGES, Language, resolve_language
from app.voice.providers.base import AudioFormat, STTProvider, TTSProvider
from app.voice.registry import CallRegistry
from app.voice.session import CallSession
from app.voice.transport import FrameSender, Transport

log = get_logger(__name__)

router = APIRouter(prefix="/exotel", tags=["exotel"])

WS_POLICY_VIOLATION = 1008
WS_INTERNAL_ERROR = 1011

ReminderAgentFactory = Callable[[CallContext], ReplyGenerator | None]


class ExotelTransport(Transport):
    """A phone call over Exotel's Voicebot applet: raw 16-bit PCM both ways.

    Exotel wants audio in chunks of at least 3.2 kB whose size is a multiple of
    320 bytes, so speech is buffered and sent in even pieces. What is left at the
    end of a reply is padded with silence.
    """

    supports_barge_in = True

    def __init__(self, stream_sid: str, send: FrameSender, sample_rate: int = 8000) -> None:
        self.input_format = AudioFormat(encoding="linear16", sample_rate=sample_rate, channels=1)
        self.output_format = self.input_format
        self._stream_sid = stream_sid
        self._send = send
        self._buffer = bytearray()
        # Bigger pieces at higher sample rates, so each still holds the same length of speech.
        self._chunk_bytes = MIN_CHUNK_BYTES * max(1, sample_rate // 8000)
        self._sequence = 0
        self._chunk = 0
        # A reply and a barge-in can reach the socket from different tasks.
        self._lock = asyncio.Lock()

    async def _send_media(self, audio: bytes) -> None:
        self._sequence += 1
        self._chunk += 1
        await self._send(
            outbound_media(self._stream_sid, audio, sequence=self._sequence, chunk=self._chunk)
        )

    async def send_audio(self, chunk: bytes) -> None:
        async with self._lock:
            self._buffer.extend(chunk)
            while len(self._buffer) >= self._chunk_bytes:
                piece = bytes(self._buffer[: self._chunk_bytes])
                del self._buffer[: self._chunk_bytes]
                await self._send_media(piece)

    async def end_of_reply(self, label: str) -> None:
        async with self._lock:
            if self._buffer:
                padding = -len(self._buffer) % FRAME_BYTES
                piece = bytes(self._buffer) + b"\x00" * padding
                self._buffer.clear()
                await self._send_media(piece)
            self._sequence += 1
            await self._send(outbound_mark(self._stream_sid, label, sequence=self._sequence))

    async def clear_audio(self) -> None:
        async with self._lock:
            self._buffer.clear()
            await self._send(outbound_clear(self._stream_sid))


def authorised(websocket: WebSocket, settings: Settings) -> bool:
    """Exotel sends the username and password from the applet URL as HTTP Basic auth."""
    username = settings.exotel_stream_username
    password = settings.exotel_stream_password
    if not username or password is None:
        return False  # never serve calls with no credential configured
    scheme, _, encoded = websocket.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        given = base64.b64decode(encoded, validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return False
    return hmac.compare_digest(given, f"{username}:{password.get_secret_value()}")


@router.websocket("/stream")
async def exotel_stream(
    websocket: WebSocket,
    settings: Settings = Depends(get_settings),
    store: AlertStore | None = Depends(get_alert_store),
    registry: CallRegistry = Depends(get_call_registry),
) -> None:
    if not authorised(websocket, settings):
        log.warning("exotel_stream.rejected", reason="bad_credentials")
        await websocket.close(code=WS_POLICY_VIOLATION)
        return
    await websocket.accept()
    session: CallSession | None = None
    try:
        async for raw in websocket.iter_text():
            match parse_inbound(raw):
                case MediaMessage() as msg:
                    if session is not None:
                        session.feed_audio(msg.audio())
                case StartMessage() as msg if session is None:
                    session = await _start_session(websocket, msg, settings, store)
                    if session is None:
                        return
                    registry.add(session)
                    session.start()
                    # Roopiee speaks first: this is a call we placed.
                    session.request_reply()
                case MarkMessage() as msg if session is not None:
                    session.on_mark(msg.mark.name)
                case DtmfMessage() as msg:
                    log.info("exotel.dtmf", digit=str(msg.dtmf.digit))
                case StopMessage() as msg:
                    log.info("exotel.stream_stopped", reason=msg.stop.reason)
                    break
                case _:
                    pass
    finally:
        if session is not None:
            registry.finish(session.call_sid)
            await session.close()
        if (
            websocket.application_state is WebSocketState.CONNECTED
            and websocket.client_state is WebSocketState.CONNECTED
        ):
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await websocket.close()
        structlog.contextvars.unbind_contextvars("call_sid", "stream_sid")


async def _start_session(
    websocket: WebSocket, msg: StartMessage, settings: Settings, store: AlertStore | None
) -> CallSession | None:
    start = msg.start
    structlog.contextvars.bind_contextvars(call_sid=start.call_sid, stream_sid=msg.stream_sid)

    context = await store.call_context(start.call_sid) if store is not None else None
    if context is None:
        # Not a call the alerts worker placed.
        log.warning("exotel_stream.rejected", reason="unknown_call")
        await websocket.close(code=WS_POLICY_VIOLATION)
        return None

    # Only languages with a voice can be spoken; the rest fall back to English.
    language = resolve_language(context.language)
    if language is None or language.tts_model is None:
        language = LANGUAGES["en"]

    state = websocket.app.state
    stt = cast("Callable[[Language], STTProvider | None]", state.web_stt_factory)(language)
    tts = cast("Callable[[Language], TTSProvider | None]", state.web_tts_factory)(language)
    if stt is None:
        log.error("exotel_stream.rejected", reason="stt_not_configured")
        await websocket.close(code=WS_INTERNAL_ERROR)
        return None
    agent_factory = cast(
        "ReminderAgentFactory",
        getattr(state, "reminder_agent_factory", None)
        or (lambda ctx: build_reminder_generator(settings, ctx)),
    )

    return CallSession(
        call_sid=start.call_sid,
        stream_sid=msg.stream_sid,
        stt=stt,
        settings=settings,
        caller=mask(start.caller),
        responder=agent_factory(context),
        tts=tts,
        transport=ExotelTransport(
            msg.stream_sid, websocket.send_text, start.media_format.sample_rate
        ),
        fallback_text=REMINDER_FALLBACK,
    )
