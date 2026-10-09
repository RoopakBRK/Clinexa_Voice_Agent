from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Iterator, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agents.responder import ReplyGenerator, ReplyTrace
from app.core.config import Settings
from app.graph.state import ConversationMessage
from app.main import create_app
from app.voice.providers.base import (
    AudioFormat,
    STTError,
    STTProvider,
    TranscriptEvent,
    TranscriptEventType,
    TTSError,
    TTSProvider,
)
from app.voice.security import sign_stream_token

AUTH_TOKEN = "test-twilio-auth-token"
STREAM_SECRET = "test-stream-secret"
PUBLIC_BASE_URL = "https://clinexa.example.com"


class FakeSTTProvider(STTProvider):
    """Consumes all audio, then emits a scripted list of transcript events.

    ``fail_first`` makes the first N streams raise STTError after consuming
    one chunk, to exercise reconnection.
    """

    name = "fake"

    def __init__(self, script: list[TranscriptEvent], fail_first: int = 0) -> None:
        self.script = script
        self.fail_first = fail_first
        self.received = bytearray()
        self.streams_opened = 0

    async def transcribe_stream(
        self, audio: AsyncIterator[bytes], audio_format: AudioFormat
    ) -> AsyncIterator[TranscriptEvent]:
        self.streams_opened += 1
        if self.streams_opened <= self.fail_first:
            self.received.extend(await anext(audio))
            raise STTError("simulated provider failure")
        async for chunk in audio:
            self.received.extend(chunk)
        for event in self.script:
            yield event


class LiveFakeSTTProvider(STTProvider):
    """A caller mid-call: emits one scripted event per audio chunk received.

    Unlike ``FakeSTTProvider`` the events arrive while the call is still open,
    which is when replies are generated.
    """

    name = "fake-live"

    def __init__(self, script: list[TranscriptEvent]) -> None:
        self.script = script

    async def transcribe_stream(
        self, audio: AsyncIterator[bytes], audio_format: AudioFormat
    ) -> AsyncIterator[TranscriptEvent]:
        for event in self.script:
            try:
                await anext(audio)
            except StopAsyncIteration:
                return
            yield event
        async for _ in audio:
            pass


class FakeReplyGenerator(ReplyGenerator):
    """Streams scripted text deltas; one script per reply, the last one repeating."""

    name = "fake"

    def __init__(self, *replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.histories: list[list[ConversationMessage]] = []
        # Tests can hold a reply open by clearing this before the turn starts.
        self.gate = asyncio.Event()
        self.gate.set()

    async def stream_reply(
        self, history: Sequence[ConversationMessage], trace: ReplyTrace | None = None
    ) -> AsyncIterator[str]:
        self.histories.append(list(history))
        script = self.replies[min(len(self.histories), len(self.replies)) - 1]
        for item in script:
            if isinstance(item, Exception):
                raise item
            yield item
            await self.gate.wait()


class FakeTTSProvider(TTSProvider):
    """Returns one audio chunk per sentence: the sentence's own bytes."""

    name = "fake"

    def __init__(self, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.spoken: list[str] = []
        self.streams_closed = 0

    async def synthesize_stream(
        self, text: AsyncIterator[str], audio_format: AudioFormat
    ) -> AsyncIterator[bytes]:
        try:
            async for sentence in text:
                if self.fail_after is not None and len(self.spoken) >= self.fail_after:
                    raise TTSError("simulated synthesis failure")
                self.spoken.append(sentence)
                yield sentence.encode()
        finally:
            self.streams_closed += 1


def final(text: str, *, speech_final: bool = False, confidence: float = 0.9) -> TranscriptEvent:
    return TranscriptEvent(
        type=TranscriptEventType.FINAL,
        text=text,
        speech_final=speech_final,
        confidence=confidence,
        endpoint_latency_ms=120.0 if speech_final else None,
        finalization_latency_ms=80.0,
    )


def twilio_start(call_sid: str, *, token: str | None, stream_sid: str = "MZtest") -> str:
    params = {"caller": "***4567"}
    if token is not None:
        params["stream_token"] = token
    return json.dumps(
        {
            "event": "start",
            "sequenceNumber": "1",
            "streamSid": stream_sid,
            "start": {
                "streamSid": stream_sid,
                "accountSid": "ACtest",
                "callSid": call_sid,
                "tracks": ["inbound"],
                "customParameters": params,
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
            },
        }
    )


def twilio_media(audio: bytes, *, stream_sid: str = "MZtest", track: str = "inbound") -> str:
    return json.dumps(
        {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"track": track, "payload": base64.b64encode(audio).decode()},
        }
    )


def twilio_stop(call_sid: str, *, stream_sid: str = "MZtest") -> str:
    return json.dumps(
        {
            "event": "stop",
            "streamSid": stream_sid,
            "stop": {"accountSid": "ACtest", "callSid": call_sid},
        }
    )


def stream_token(call_sid: str) -> str:
    return sign_stream_token(STREAM_SECRET, call_sid)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,  # never read the developer's real .env in tests
        environment="test",
        log_json=False,
        log_transcripts=True,
        public_base_url=PUBLIC_BASE_URL,
        twilio_auth_token=AUTH_TOKEN,
        stream_token_secret=STREAM_SECRET,
        deepgram_api_key=None,
        anthropic_api_key=None,
    )


@pytest.fixture
def fake_stt() -> FakeSTTProvider:
    return FakeSTTProvider(
        [
            final("I've had a headache"),
            final("for three days.", speech_final=True),
            final("It's getting worse.", confidence=0.8),
            TranscriptEvent(type=TranscriptEventType.UTTERANCE_END, endpoint_latency_ms=900.0),
        ]
    )


@pytest.fixture
def app(settings: Settings, fake_stt: FakeSTTProvider) -> FastAPI:
    return create_app(settings, stt_provider=fake_stt)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
