"""Typed models for Exotel's Voicebot applet WebSocket protocol.

Inbound:  connected, start, media, dtmf, mark, stop
Outbound: media, mark, clear
https://support.exotel.com/support/solutions/articles/3000108630-working-with-the-stream-and-voicebot-applet

It is close to Twilio Media Streams, with two differences that matter: field names
are snake_case, and the audio is raw 16-bit little-endian PCM ("slin"), not mu-law.
"""

from __future__ import annotations

import base64
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from app.core.logging import get_logger

log = get_logger(__name__)

# Audio sent to Exotel must be a multiple of this many bytes (20 ms at 8 kHz).
FRAME_BYTES = 320
# Exotel asks for chunks of at least 3.2 kB and at most 100 kB.
MIN_CHUNK_BYTES = 3200
MAX_CHUNK_BYTES = 100_000
DEFAULT_SAMPLE_RATE = 8000


class _ExotelModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class MediaFormat(_ExotelModel):
    encoding: str | None = None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    bit_rate: str | None = None


class StartPayload(_ExotelModel):
    stream_sid: str | None = None
    call_sid: str
    account_sid: str | None = None
    caller: str | None = Field(default=None, alias="from")
    called: str | None = Field(default=None, alias="to")
    custom_parameters: dict[str, str] = Field(default_factory=dict)
    media_format: MediaFormat = Field(default_factory=MediaFormat)


class MediaPayload(_ExotelModel):
    chunk: int | None = None
    timestamp: str | None = None
    payload: str


class MarkPayload(_ExotelModel):
    name: str


class DtmfPayload(_ExotelModel):
    digit: str | int
    duration: str | int | None = None


class StopPayload(_ExotelModel):
    call_sid: str | None = None
    account_sid: str | None = None
    reason: str | None = None


class ConnectedMessage(_ExotelModel):
    event: Literal["connected"]


class StartMessage(_ExotelModel):
    event: Literal["start"]
    stream_sid: str
    start: StartPayload


class MediaMessage(_ExotelModel):
    event: Literal["media"]
    stream_sid: str | None = None
    media: MediaPayload

    def audio(self) -> bytes:
        return base64.b64decode(self.media.payload)


class DtmfMessage(_ExotelModel):
    event: Literal["dtmf"]
    stream_sid: str | None = None
    dtmf: DtmfPayload


class MarkMessage(_ExotelModel):
    event: Literal["mark"]
    stream_sid: str | None = None
    mark: MarkPayload


class StopMessage(_ExotelModel):
    event: Literal["stop"]
    stream_sid: str | None = None
    stop: StopPayload = Field(default_factory=StopPayload)


ExotelInboundMessage = Annotated[
    ConnectedMessage | StartMessage | MediaMessage | DtmfMessage | MarkMessage | StopMessage,
    Field(discriminator="event"),
]
_inbound_adapter: TypeAdapter[ExotelInboundMessage] = TypeAdapter(ExotelInboundMessage)


def parse_inbound(raw: str) -> ExotelInboundMessage | None:
    """Parse an Exotel frame; returns None for unknown or malformed messages."""
    try:
        return _inbound_adapter.validate_json(raw)
    except ValidationError as exc:
        log.warning("exotel.unparseable_message", error=exc.errors()[0]["msg"], raw=raw[:200])
        return None


def outbound_media(stream_sid: str, audio: bytes, *, sequence: int, chunk: int) -> str:
    """Frame raw PCM for playback to the person on the call."""
    return json.dumps(
        {
            "event": "media",
            "sequence_number": sequence,
            "stream_sid": stream_sid,
            "media": {"chunk": chunk, "payload": base64.b64encode(audio).decode("ascii")},
        }
    )


def outbound_mark(stream_sid: str, name: str, *, sequence: int) -> str:
    """Exotel echoes a mark back once all audio queued before it has played."""
    return json.dumps(
        {
            "event": "mark",
            "sequence_number": sequence,
            "stream_sid": stream_sid,
            "mark": {"name": name},
        }
    )


def outbound_clear(stream_sid: str) -> str:
    """Discard audio Exotel has buffered but not yet played (used for barge-in)."""
    return json.dumps({"event": "clear", "stream_sid": stream_sid})
