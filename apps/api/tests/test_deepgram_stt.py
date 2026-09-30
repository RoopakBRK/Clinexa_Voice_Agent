"""DeepgramSTTProvider against an in-process fake of Deepgram's /v1/listen socket."""

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

from app.voice.providers.base import (
    TWILIO_AUDIO_FORMAT,
    STTConnectionError,
    STTError,
    TranscriptEvent,
    TranscriptEventType,
)
from app.voice.providers.deepgram_stt import DeepgramSTTProvider

API_KEY = "dg-test-key"


def results(
    text: str, *, start: float, duration: float, is_final: bool, speech_final: bool = False
) -> dict[str, Any]:
    words = [{"word": w, "start": start, "end": start + duration, "confidence": 0.95} for w in text.split()]
    return {
        "type": "Results",
        "start": start,
        "duration": duration,
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": text, "confidence": 0.93, "words": words}]},
    }


class FakeDeepgram:
    """Records what the client sends; replies with a script once CloseStream arrives."""

    def __init__(self, script: list[dict[str, Any]], close_code: int = 1000) -> None:
        self.script = script
        self.close_code = close_code
        self.audio = bytearray()
        self.control: list[str] = []
        self.path: str = ""
        self.headers: Headers | None = None

    def process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        self.path = request.path
        self.headers = request.headers
        if request.headers.get("Authorization") != f"Token {API_KEY}":
            return connection.respond(HTTPStatus.UNAUTHORIZED, "invalid credentials\n")
        return None

    async def handler(self, ws: ServerConnection) -> None:
        async for message in ws:
            if isinstance(message, bytes):
                self.audio.extend(message)
                continue
            self.control.append(json.loads(message)["type"])
            if self.control[-1] == "CloseStream":
                for payload in self.script:
                    await ws.send(json.dumps(payload))
                await ws.send(json.dumps({"type": "Metadata", "request_id": "req-1"}))
                await ws.close(self.close_code, "done" if self.close_code == 1000 else "boom")
                return


async def audio_frames(n: int, size: int = 160) -> AsyncIterator[bytes]:
    for _ in range(n):
        yield b"\xff" * size


async def run(fake: FakeDeepgram, api_key: str = API_KEY, frames: int = 5) -> list[TranscriptEvent]:
    async with serve(fake.handler, "127.0.0.1", 0, process_request=fake.process_request) as server:
        port = server.sockets[0].getsockname()[1]
        provider = DeepgramSTTProvider(
            api_key,
            url=f"ws://127.0.0.1:{port}/v1/listen",
            keyterms=["paracetamol", "shortness of breath"],
        )
        return [e async for e in provider.transcribe_stream(audio_frames(frames), TWILIO_AUDIO_FORMAT)]


async def test_streams_audio_and_parses_events() -> None:
    fake = FakeDeepgram(
        [
            {"type": "SpeechStarted", "timestamp": 0.02},
            results("I have", start=0.0, duration=0.04, is_final=False),
            results("I have a cough.", start=0.0, duration=0.08, is_final=True, speech_final=True),
            {"type": "UtteranceEnd", "last_word_end": 0.08},
        ]
    )
    events = await run(fake)

    assert bytes(fake.audio) == b"\xff" * 800
    assert fake.control == ["CloseStream"]
    assert [e.type for e in events] == [
        TranscriptEventType.SPEECH_STARTED,
        TranscriptEventType.INTERIM,
        TranscriptEventType.FINAL,
        TranscriptEventType.UTTERANCE_END,
        TranscriptEventType.METADATA,
    ]
    final = events[2]
    assert final.text == "I have a cough."
    assert final.speech_final is True
    assert final.confidence == pytest.approx(0.93)
    assert final.audio_end_s == pytest.approx(0.08)
    # 0.08 s of audio was sent, so latency is measurable and non-negative.
    assert final.finalization_latency_ms is not None and final.finalization_latency_ms >= 0
    assert final.endpoint_latency_ms is not None and final.endpoint_latency_ms >= 0
    assert events[1].finalization_latency_ms is None  # interims are not final
    assert events[3].endpoint_latency_ms is not None
    assert events[4].provider_request_id == "req-1"


async def test_connection_url_and_auth() -> None:
    fake = FakeDeepgram([])
    await run(fake, frames=1)

    assert fake.headers is not None and fake.headers["Authorization"] == f"Token {API_KEY}"
    parts = urlsplit(fake.path)
    assert parts.path == "/v1/listen"
    query = parse_qs(parts.query)
    assert query["model"] == ["nova-3"]
    assert query["encoding"] == ["mulaw"]
    assert query["sample_rate"] == ["8000"]
    assert query["interim_results"] == ["true"]
    assert query["vad_events"] == ["true"]
    assert query["endpointing"] == ["300"]
    assert query["utterance_end_ms"] == ["1000"]
    assert query["keyterm"] == ["paracetamol", "shortness of breath"]


async def test_rejected_credentials_raise_connection_error() -> None:
    with pytest.raises(STTConnectionError, match="401"):
        await run(FakeDeepgram([]), api_key="wrong")


async def test_unreachable_server_raises_connection_error() -> None:
    provider = DeepgramSTTProvider(API_KEY, url="ws://127.0.0.1:9/v1/listen", connect_timeout_s=1)
    with pytest.raises(STTConnectionError):
        async for _ in provider.transcribe_stream(audio_frames(1), TWILIO_AUDIO_FORMAT):
            pass


async def test_abnormal_close_raises_stt_error() -> None:
    fake = FakeDeepgram([results("hello", start=0, duration=0.02, is_final=True)], close_code=1011)
    with pytest.raises(STTError):
        await run(fake)
