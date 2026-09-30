"""Typed models for the Twilio Media Streams WebSocket protocol.

Inbound:  connected, start, media, mark, dtmf, stop
Outbound: media, mark, clear
https://www.twilio.com/docs/voice/media-streams/websocket-messages
"""

from __future__ import annotations

import base64
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from pydantic.alias_generators import to_camel

from app.core.logging import get_logger

log = get_logger(__name__)


class _TwilioModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


class MediaFormat(_TwilioModel):
    encoding: str
    sample_rate: int
    channels: int


class StartPayload(_TwilioModel):
    stream_sid: str
    account_sid: str
    call_sid: str
    tracks: list[str] = Field(default_factory=list)
    custom_parameters: dict[str, str] = Field(default_factory=dict)
    media_format: MediaFormat


class MediaPayload(_TwilioModel):
    track: str = "inbound"
    chunk: str | None = None
    timestamp: str | None = None
    payload: str


class MarkPayload(_TwilioModel):
    name: str


class DtmfPayload(_TwilioModel):
    track: str | None = None
    digit: str


class StopPayload(_TwilioModel):
    account_sid: str | None = None
    call_sid: str | None = None


class ConnectedMessage(_TwilioModel):
    event: Literal["connected"]
    protocol: str | None = None
    version: str | None = None


class StartMessage(_TwilioModel):
    event: Literal["start"]
    sequence_number: str | None = None
    stream_sid: str
    start: StartPayload


class MediaMessage(_TwilioModel):
    event: Literal["media"]
    sequence_number: str | None = None
    stream_sid: str
    media: MediaPayload

    def audio(self) -> bytes:
        return base64.b64decode(self.media.payload)


class MarkMessage(_TwilioModel):
    event: Literal["mark"]
    sequence_number: str | None = None
    stream_sid: str
    mark: MarkPayload


class DtmfMessage(_TwilioModel):
    event: Literal["dtmf"]
    sequence_number: str | None = None
    stream_sid: str
    dtmf: DtmfPayload


class StopMessage(_TwilioModel):
    event: Literal["stop"]
    sequence_number: str | None = None
    stream_sid: str
    stop: StopPayload = Field(default_factory=StopPayload)


TwilioInboundMessage = Annotated[
    ConnectedMessage | StartMessage | MediaMessage | MarkMessage | DtmfMessage | StopMessage,
    Field(discriminator="event"),
]
_inbound_adapter: TypeAdapter[TwilioInboundMessage] = TypeAdapter(TwilioInboundMessage)


def parse_inbound(raw: str) -> TwilioInboundMessage | None:
    """Parse a Twilio frame; returns None for unknown or malformed messages."""
    try:
        return _inbound_adapter.validate_json(raw)
    except ValidationError as exc:
        log.warning("twilio.unparseable_message", error=exc.errors()[0]["msg"], raw=raw[:200])
        return None


def outbound_media(stream_sid: str, audio: bytes) -> str:
    """Frame μ-law audio for playback to the caller."""
    payload = base64.b64encode(audio).decode("ascii")
    return json.dumps({"event": "media", "streamSid": stream_sid, "media": {"payload": payload}})


def outbound_mark(stream_sid: str, name: str) -> str:
    """Twilio echoes a mark back once all audio queued before it has played."""
    return json.dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": name}})


def outbound_clear(stream_sid: str) -> str:
    """Discard all audio buffered at Twilio (used for barge-in)."""
    return json.dumps({"event": "clear", "streamSid": stream_sid})
