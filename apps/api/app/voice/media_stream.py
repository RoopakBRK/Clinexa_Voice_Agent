"""Twilio Media Streams WebSocket endpoint (bidirectional ``<Connect><Stream>``)."""

from __future__ import annotations

import contextlib

import structlog
from fastapi import APIRouter, Depends, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from app.agents.responder import ReplyGenerator
from app.core.config import Settings
from app.core.logging import get_logger
from app.voice.deps import (
    get_call_registry,
    get_reply_generator,
    get_settings,
    get_stt_provider,
    get_tts_provider,
)
from app.voice.providers.base import TWILIO_AUDIO_FORMAT, STTProvider, TTSProvider
from app.voice.registry import CallRegistry
from app.voice.security import STREAM_TOKEN_PARAM, verify_stream_token
from app.voice.session import CallSession
from app.voice.twilio_protocol import (
    ConnectedMessage,
    DtmfMessage,
    MarkMessage,
    MediaMessage,
    StartMessage,
    StopMessage,
    parse_inbound,
)
from app.voice.twilio_webhook import CALLER_PARAM

log = get_logger(__name__)

router = APIRouter(prefix="/twilio", tags=["twilio"])

# WebSocket close codes (RFC 6455)
WS_POLICY_VIOLATION = 1008
WS_UNSUPPORTED_DATA = 1003
WS_INTERNAL_ERROR = 1011


@router.websocket("/media-stream")
async def media_stream(
    websocket: WebSocket,
    settings: Settings = Depends(get_settings),
    stt: STTProvider | None = Depends(get_stt_provider),
    tts: TTSProvider | None = Depends(get_tts_provider),
    responder: ReplyGenerator | None = Depends(get_reply_generator),
    registry: CallRegistry = Depends(get_call_registry),
) -> None:
    await websocket.accept()
    session: CallSession | None = None
    try:
        async for raw in websocket.iter_text():
            match parse_inbound(raw):
                case MediaMessage() as msg:
                    if session is not None and msg.media.track == "inbound":
                        session.feed_audio(msg.audio())
                case StartMessage() as msg if session is None:
                    session = await _start_session(websocket, msg, settings, stt, tts, responder)
                    if session is None:
                        return
                    registry.add(session)
                    session.start()
                case MarkMessage() as msg if session is not None:
                    session.on_mark(msg.mark.name)
                case DtmfMessage() as msg:
                    log.info("twilio.dtmf", digit=msg.dtmf.digit)
                case StopMessage():
                    log.info("twilio.stream_stopped")
                    break
                case ConnectedMessage() | None:
                    pass
                case other:
                    log.warning(
                        "twilio.unexpected_message", twilio_event=getattr(other, "event", None)
                    )
    finally:
        if session is not None:
            registry.finish(session.call_sid)
            await session.close()
        if (
            websocket.application_state is WebSocketState.CONNECTED
            and websocket.client_state is WebSocketState.CONNECTED
        ):
            # Twilio usually drops the socket right after `stop`, before we close it.
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await websocket.close()
        structlog.contextvars.unbind_contextvars("call_sid", "stream_sid")


async def _start_session(
    websocket: WebSocket,
    msg: StartMessage,
    settings: Settings,
    stt: STTProvider | None,
    tts: TTSProvider | None,
    responder: ReplyGenerator | None,
) -> CallSession | None:
    start = msg.start
    # Tasks created from here on inherit these, so all call logs are correlated.
    structlog.contextvars.bind_contextvars(call_sid=start.call_sid, stream_sid=msg.stream_sid)

    token = start.custom_parameters.get(STREAM_TOKEN_PARAM)
    if not verify_stream_token(
        settings.stream_token_secret.get_secret_value(), start.call_sid, token
    ):
        log.warning("media_stream.rejected", reason="invalid_stream_token")
        await websocket.close(code=WS_POLICY_VIOLATION)
        return None

    fmt = start.media_format
    if fmt.encoding != "audio/x-mulaw" or fmt.sample_rate != TWILIO_AUDIO_FORMAT.sample_rate:
        log.error("media_stream.rejected", reason="unsupported_format", format=fmt.model_dump())
        await websocket.close(code=WS_UNSUPPORTED_DATA)
        return None

    if stt is None:
        log.error("media_stream.rejected", reason="stt_not_configured")
        await websocket.close(code=WS_INTERNAL_ERROR)
        return None

    return CallSession(
        call_sid=start.call_sid,
        stream_sid=msg.stream_sid,
        stt=stt,
        settings=settings,
        caller=start.custom_parameters.get(CALLER_PARAM),
        responder=responder,
        tts=tts,
        send=websocket.send_text,
    )
