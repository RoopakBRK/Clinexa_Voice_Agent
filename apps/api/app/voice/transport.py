"""How a session's audio and events reach the person on the other end.

``CallSession`` speaks to a ``Transport`` rather than to Twilio directly, so the
same pipeline serves a phone call (μ-law 8 kHz frames wrapped in Twilio JSON) and
a browser (raw linear16 frames plus JSON events for the page).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from starlette.websockets import WebSocket

from app.voice.providers.base import TWILIO_AUDIO_FORMAT, AudioFormat
from app.voice.twilio_protocol import outbound_mark, outbound_media

# Sends one Twilio Media Streams frame (JSON text) to the caller's WebSocket.
FrameSender = Callable[[str], Awaitable[None]]

SessionStatus = Literal["listening", "thinking", "speaking"]
TranscriptRole = Literal["user", "assistant"]

WEB_INPUT_FORMAT = AudioFormat(encoding="linear16", sample_rate=16000, channels=1)
WEB_OUTPUT_FORMAT = AudioFormat(encoding="linear16", sample_rate=24000, channels=1)


class Transport(ABC):
    # Audio the person sends us (fed to STT) and audio we send back (asked of TTS).
    input_format: AudioFormat
    output_format: AudioFormat
    # Whether a reply can be cut short when the person starts talking over it.
    supports_barge_in: bool = False

    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None:
        """Play one chunk of synthesised speech."""

    # The rest are optional: a transport that cannot do one simply ignores it.

    async def end_of_reply(self, label: str) -> None:  # noqa: B027
        """All audio of one reply has been sent."""

    async def clear_audio(self) -> None:  # noqa: B027
        """Drop audio that was sent but has not been played yet."""

    async def send_transcript(self, role: TranscriptRole, text: str, *, final: bool) -> None:  # noqa: B027
        """What was heard or said, for a live transcript."""

    async def send_status(self, value: SessionStatus) -> None:  # noqa: B027
        """What the assistant is doing right now."""

    async def send_error(self, code: str, message: str) -> None:  # noqa: B027
        """Something went wrong that the other side may want to show."""


class TwilioTransport(Transport):
    """A phone call over Twilio Media Streams. Transcript and status go nowhere."""

    input_format = TWILIO_AUDIO_FORMAT
    output_format = TWILIO_AUDIO_FORMAT

    def __init__(self, stream_sid: str, send: FrameSender) -> None:
        self._stream_sid = stream_sid
        self._send = send

    async def send_audio(self, chunk: bytes) -> None:
        await self._send(outbound_media(self._stream_sid, chunk))

    async def end_of_reply(self, label: str) -> None:
        await self._send(outbound_mark(self._stream_sid, label))


class WebTransport(Transport):
    """A browser tab: binary linear16 frames out, JSON events alongside them."""

    input_format = WEB_INPUT_FORMAT
    output_format = WEB_OUTPUT_FORMAT
    supports_barge_in = True

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        # Replies, tool calls and transcript events are sent from different tasks.
        self._lock = asyncio.Lock()

    async def send_audio(self, chunk: bytes) -> None:
        async with self._lock:
            await self._ws.send_bytes(chunk)

    async def send_json(self, message: dict[str, Any]) -> None:
        async with self._lock:
            await self._ws.send_json(message)

    async def clear_audio(self) -> None:
        await self.send_json({"type": "clear_audio"})

    async def send_transcript(self, role: TranscriptRole, text: str, *, final: bool) -> None:
        await self.send_json({"type": "transcript", "role": role, "text": text, "final": final})

    async def send_status(self, value: SessionStatus) -> None:
        await self.send_json({"type": "status", "value": value})

    async def send_error(self, code: str, message: str) -> None:
        await self.send_json({"type": "error", "code": code, "message": message})
