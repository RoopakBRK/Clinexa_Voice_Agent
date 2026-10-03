"""DeepgramTTSProvider against an in-process fake of Deepgram's /v1/speak socket."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from app.voice.providers.base import TWILIO_AUDIO_FORMAT, TTSConnectionError, TTSError
from app.voice.providers.deepgram_tts import DeepgramTTSProvider

API_KEY = "dg-test-key"


class FakeDeepgramSpeak:
    """Buffers Speak text and, like Deepgram, only returns audio for it on Flush."""

    def __init__(self, *, ack_flushes: bool = True, close_code: int | None = None) -> None:
        self.ack_flushes = ack_flushes
        self.close_code = close_code
        self.messages: list[dict[str, Any]] = []
        self.path: str = ""
        self.headers: Headers | None = None

    def process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        self.path = request.path
        self.headers = request.headers
        if request.headers.get("Authorization") != f"Token {API_KEY}":
            return connection.respond(HTTPStatus.UNAUTHORIZED, "invalid credentials\n")
        return None

    async def handler(self, ws: ServerConnection) -> None:
        await ws.send(json.dumps({"type": "Metadata", "request_id": "req-1"}))
        buffered: list[str] = []
        flushes = 0
        async for raw in ws:
            assert isinstance(raw, str)
            message = json.loads(raw)
            self.messages.append(message)
            match message["type"]:
                case "Speak":
                    buffered.append(message["text"])
                case "Flush":
                    if self.close_code is not None:
                        await ws.close(self.close_code, "boom")
                        return
                    for text in buffered:
                        await ws.send(text.strip().encode())
                    buffered.clear()
                    if self.ack_flushes:
                        await ws.send(json.dumps({"type": "Flushed", "sequence_id": flushes}))
                    flushes += 1
                case "Close":
                    return

    @property
    def types(self) -> list[str]:
        return [m["type"] for m in self.messages]


async def sentences(*items: str | Exception) -> AsyncIterator[str]:
    for item in items:
        if isinstance(item, Exception):
            raise item
        yield item


async def run(
    fake: FakeDeepgramSpeak, *text: str | Exception, api_key: str = API_KEY, **kwargs: Any
) -> list[bytes]:
    async with serve(fake.handler, "127.0.0.1", 0, process_request=fake.process_request) as server:
        port = server.sockets[0].getsockname()[1]
        provider = DeepgramTTSProvider(api_key, url=f"ws://127.0.0.1:{port}/v1/speak", **kwargs)
        return [c async for c in provider.synthesize_stream(sentences(*text), TWILIO_AUDIO_FORMAT)]


async def test_first_sentence_is_flushed_alone_and_the_rest_together() -> None:
    fake = FakeDeepgramSpeak()
    audio = await run(fake, "I'm sorry to hear that.", "How long?", "Is it worse?")

    assert audio == [b"I'm sorry to hear that.", b"How long?", b"Is it worse?"]
    # One Flush for the first sentence (speech starts early), one for everything after.
    assert fake.types == ["Speak", "Flush", "Speak", "Speak", "Flush", "Close"]
    assert fake.messages[0]["text"] == "I'm sorry to hear that. "


async def test_single_sentence_needs_one_flush() -> None:
    fake = FakeDeepgramSpeak()
    assert await run(fake, "Please call back.", "  ") == [b"Please call back."]
    assert fake.types == ["Speak", "Flush", "Close"]


async def test_no_text_returns_no_audio() -> None:
    fake = FakeDeepgramSpeak()
    assert await run(fake) == []
    assert "Speak" not in fake.types and "Flush" not in fake.types


async def test_connection_url_and_auth() -> None:
    fake = FakeDeepgramSpeak()
    await run(fake, "Hello there.")

    assert fake.headers is not None and fake.headers["Authorization"] == f"Token {API_KEY}"
    parts = urlsplit(fake.path)
    assert parts.path == "/v1/speak"
    # Telephony format straight from Deepgram: no transcoding before Twilio.
    assert parse_qs(parts.query) == {
        "model": ["aura-2-thalia-en"],
        "encoding": ["mulaw"],
        "sample_rate": ["8000"],
    }


async def test_rejected_credentials_raise_connection_error() -> None:
    with pytest.raises(TTSConnectionError, match="401"):
        await run(FakeDeepgramSpeak(), "Hello there.", api_key="wrong")


async def test_unreachable_server_raises_connection_error() -> None:
    provider = DeepgramTTSProvider(API_KEY, url="ws://127.0.0.1:9/v1/speak", connect_timeout_s=1)
    with pytest.raises(TTSConnectionError):
        async for _ in provider.synthesize_stream(sentences("Hello."), TWILIO_AUDIO_FORMAT):
            pass


async def test_abnormal_close_raises_tts_error() -> None:
    with pytest.raises(TTSError):
        await run(FakeDeepgramSpeak(close_code=1011), "Hello there.")


async def test_missing_flush_acknowledgement_times_out() -> None:
    fake = FakeDeepgramSpeak(ack_flushes=False)
    with pytest.raises(TTSError, match="before all audio"):
        await run(fake, "Hello there.", flush_timeout_s=0.2)


async def test_failing_text_source_raises_tts_error() -> None:
    with pytest.raises(TTSError, match="text sender failed"):
        await run(FakeDeepgramSpeak(), "Hello there.", RuntimeError("llm died"))


async def test_consumer_stopping_early_closes_the_connection() -> None:
    fake = FakeDeepgramSpeak()
    async with serve(fake.handler, "127.0.0.1", 0, process_request=fake.process_request) as server:
        port = server.sockets[0].getsockname()[1]
        provider = DeepgramTTSProvider(API_KEY, url=f"ws://127.0.0.1:{port}/v1/speak")
        stream = provider.synthesize_stream(
            sentences("First sentence here.", "Second one."), TWILIO_AUDIO_FORMAT
        )
        assert await anext(stream) == b"First sentence here."
        await stream.aclose()  # type: ignore[attr-defined]
    assert fake.types[-1] == "Close"
