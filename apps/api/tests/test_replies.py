"""Phase 2: caller utterance -> LLM reply -> sentences -> TTS audio -> Twilio frames."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from app.agents.responder import (
    SYSTEM_PROMPT,
    ClaudeReplyGenerator,
    ReplyError,
    ReplyRefused,
    to_messages,
)
from app.core.config import Settings
from app.graph.state import ConversationMessage
from app.main import create_app
from app.voice.sentences import SentenceChunker
from app.voice.session import CallSession
from tests.conftest import (
    FakeReplyGenerator,
    FakeSTTProvider,
    FakeTTSProvider,
    LiveFakeSTTProvider,
    final,
    stream_token,
    twilio_media,
    twilio_start,
    twilio_stop,
)

FRAME = b"\xff" * 160
REPLY = ["I'm sorry to hear ", "that. How long has ", "it been going on?"]
SENTENCES = ["I'm sorry to hear that.", "How long has it been going on?"]


# --- sentence chunking ---------------------------------------------------


def chunk(*deltas: str) -> list[str]:
    chunker = SentenceChunker()
    out = [s for d in deltas for s in chunker.feed(d)]
    if (rest := chunker.flush()) is not None:
        out.append(rest)
    return out


def test_sentence_is_emitted_as_soon_as_the_next_one_starts() -> None:
    chunker = SentenceChunker()
    assert chunker.feed("I'm sorry to hear that.") == []  # could still be "that.5" or "that..."
    assert chunker.feed(" How long") == ["I'm sorry to hear that."]
    assert chunker.feed(" has it lasted?") == []
    assert chunker.flush() == "How long has it lasted?"
    assert chunker.flush() is None


def test_sentences_split_across_arbitrary_deltas() -> None:
    assert chunk(*REPLY) == SENTENCES
    assert chunk(*"Is it getting worse? Please tell me more!") == [
        "Is it getting worse?",
        "Please tell me more!",
    ]


def test_decimals_and_abbreviations_do_not_split() -> None:
    assert chunk("A fever is 38.5 degrees or more. Dr. Lee e.g. would check it. Okay?") == [
        "A fever is 38.5 degrees or more.",
        "Dr. Lee e.g. would check it.",
        "Okay?",
    ]


def test_short_fragments_are_spoken_with_the_next_sentence() -> None:
    assert chunk("Okay. I understand. How long has it lasted?") == [
        "Okay. I understand.",
        "How long has it lasted?",
    ]


def test_quoted_and_whitespace_only_text() -> None:
    assert chunk('She said "call now." Then she left.') == [
        'She said "call now."',
        "Then she left.",
    ]
    assert chunk("  ", "\n") == []


# --- conversation history -> API messages ----------------------------------


def msg(role: str, content: str, **kwargs: Any) -> ConversationMessage:
    return ConversationMessage(role=role, content=content, **kwargs)  # type: ignore[arg-type]


def test_history_maps_to_alternating_api_roles() -> None:
    history = [
        msg("assistant", "greeting kept out: the first message must be the caller's"),
        msg("patient", "I have a headache."),
        msg("system", "internal note"),
        msg("assistant", "How long?"),
        msg("patient", "  "),
        msg("patient", "Three days."),
    ]
    assert to_messages(history) == [
        {"role": "user", "content": "I have a headache."},
        {"role": "assistant", "content": "How long?"},
        {"role": "user", "content": "Three days."},
    ]


def test_system_prompt_states_the_safety_scope() -> None:
    prompt = SYSTEM_PROMPT.format(emergency="911", greeting="Hello.")
    for required in ("not a doctor", "do not diagnose", "do not prescribe", "call 911"):
        assert required in prompt


# --- Claude reply generator (SDK surface faked) ------------------------------


class FakeStream:
    def __init__(self, deltas: list[str], stop_reason: str, error: Exception | None) -> None:
        self._deltas, self._stop_reason, self._error = deltas, stop_reason, error

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    async def text_stream(self) -> AsyncIterator[str]:
        for delta in self._deltas:
            yield delta
        if self._error is not None:
            raise self._error

    async def get_final_message(self) -> Any:
        return SimpleNamespace(
            stop_reason=self._stop_reason,
            stop_details=SimpleNamespace(category="bio")
            if self._stop_reason == "refusal"
            else None,
            model="claude-opus-5-5",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0),
        )


class FakeAnthropic:
    def __init__(
        self,
        deltas: list[str],
        *,
        stop_reason: str = "end_turn",
        error: Exception | None = None,
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self._stream = FakeStream(deltas, stop_reason, error)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._open))

    def _open(self, **kwargs: Any) -> FakeStream:
        self.requests.append(kwargs)
        return self._stream


def generator(client: FakeAnthropic, **kwargs: Any) -> ClaudeReplyGenerator:
    return ClaudeReplyGenerator("unused", system_prompt="SYSTEM", client=client, **kwargs)  # type: ignore[arg-type]


HISTORY = [msg("patient", "I have a headache.")]


async def test_claude_request_shape_and_streamed_text() -> None:
    client = FakeAnthropic(REPLY)
    text = [t async for t in generator(client).stream_reply(HISTORY)]

    assert text == REPLY
    (request,) = client.requests
    assert request["model"] == "claude-opus-5-5"
    assert request["system"] == "SYSTEM"
    assert request["messages"] == [{"role": "user", "content": "I have a headache."}]
    assert request["output_config"] == {"effort": "low"}
    assert request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["cache_control"] == {"type": "ephemeral"}
    # Opus 5.5 rejects sampling parameters and explicit thinking budgets.
    assert not {"temperature", "top_p", "thinking"} & request.keys()


async def test_effort_and_fallback_can_be_omitted_for_models_without_them() -> None:
    client = FakeAnthropic(["Hello there."])
    gen = generator(client, model="claude-haiku-4-5", effort=None, refusal_fallback=False)
    assert [t async for t in gen.stream_reply(HISTORY)] == ["Hello there."]

    (request,) = client.requests
    for name in ("output_config", "fallbacks", "betas"):
        assert request[name] is anthropic.omit


async def test_refusal_raises_after_any_streamed_text() -> None:
    gen = generator(FakeAnthropic(["I can't"], stop_reason="refusal"))
    with pytest.raises(ReplyRefused, match="bio"):
        async for _ in gen.stream_reply(HISTORY):
            pass


def api_errors() -> list[Exception]:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def status(cls: type[anthropic.APIStatusError], code: int) -> Exception:
        return cls("boom", response=httpx2.Response(code, request=request), body=None)

    return [
        anthropic.APITimeoutError(request=request),
        anthropic.APIConnectionError(request=request),
        status(anthropic.RateLimitError, 429),
        status(anthropic.BadRequestError, 400),
        status(anthropic.InternalServerError, 500),
    ]


@pytest.mark.parametrize("error", api_errors(), ids=lambda e: type(e).__name__)
async def test_api_failures_become_reply_errors(error: Exception) -> None:
    gen = generator(FakeAnthropic([], error=error))
    with pytest.raises(ReplyError):
        async for _ in gen.stream_reply(HISTORY):
            pass


async def test_nothing_to_reply_to_is_an_error_not_a_request() -> None:
    client = FakeAnthropic(REPLY)
    with pytest.raises(ReplyError, match="no caller turn"):
        async for _ in generator(client).stream_reply([msg("assistant", "Hello?")]):
            pass
    assert client.requests == []


# --- reply pipeline in a call ------------------------------------------------


class Call:
    """A CallSession wired to fakes, capturing every frame sent to Twilio."""

    def __init__(
        self,
        settings: Settings,
        responder: FakeReplyGenerator,
        *,
        tts: FakeTTSProvider | None = None,
        utterances: list[str] | None = None,
        on_frame: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self.frames: list[dict[str, Any]] = []
        self.tts = tts or FakeTTSProvider()
        self._on_frame = on_frame
        script = [final(u, speech_final=True) for u in utterances or ["I have a headache."]]
        self.session = CallSession(
            call_sid="CA1",
            stream_sid="MZ1",
            stt=LiveFakeSTTProvider(script),
            settings=settings,
            responder=responder,
            tts=self.tts,
            send=self._send,
        )
        self.session.start()

    async def _send(self, raw: str) -> None:
        frame = json.loads(raw)
        self.frames.append(frame)
        if self._on_frame is not None:
            await self._on_frame(frame)

    async def caller_speaks(self) -> None:
        """Deliver the next scripted utterance."""
        turns = len(self.transcript("patient"))
        self.session.feed_audio(FRAME)
        await wait_until(lambda: len(self.transcript("patient")) > turns)

    async def reply_finished(self) -> None:
        task = self.session._reply_task
        assert task is not None
        await asyncio.wait_for(task, 2)

    def audio(self) -> list[str]:
        return [
            base64.b64decode(f["media"]["payload"]).decode()
            for f in self.frames
            if f["event"] == "media"
        ]

    def transcript(self, role: str | None = None) -> list[str]:
        history = self.session.state.conversation_history
        return [m.content for m in history if role is None or m.role == role]


async def wait_until(condition: Callable[[], bool], timeout_s: float = 2.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not condition():  # noqa: ASYNC110 - polling plain session state
            await asyncio.sleep(0)


async def test_reply_is_spoken_sentence_by_sentence(settings: Settings) -> None:
    responder = FakeReplyGenerator(REPLY)
    call = Call(settings, responder)
    await call.caller_speaks()
    await call.reply_finished()

    assert call.tts.spoken == SENTENCES
    assert call.audio() == SENTENCES
    assert [f["event"] for f in call.frames] == ["media", "media", "mark"]
    assert {f["streamSid"] for f in call.frames} == {"MZ1"}
    assert call.frames[-1]["mark"] == {"name": "reply-1"}

    assert [m.content for m in responder.histories[0]] == ["I have a headache."]
    assert call.transcript() == ["I have a headache.", " ".join(SENTENCES)]
    reply = call.session.state.conversation_history[1]
    assert (reply.role, reply.interrupted) == ("assistant", False)
    await call.session.close()

    snapshot = call.session.snapshot()
    assert snapshot.replies_spoken == 1
    for metric in (
        "llm_ttft_ms",
        "llm_first_sentence_ms",
        "llm_total_ms",
        "tts_ttfa_ms",
        "response_latency_ms",
        "voice_to_voice_ms",
    ):
        assert snapshot.latency[metric].count == 1, metric
    # voice-to-voice = STT endpointing (120 ms in the fake) + time to first audio.
    assert snapshot.latency["voice_to_voice_ms"].max_ms == pytest.approx(
        120.0 + snapshot.latency["response_latency_ms"].max_ms
    )


async def test_first_sentence_is_spoken_before_the_reply_is_complete(settings: Settings) -> None:
    responder = FakeReplyGenerator(["Thanks for telling me. ", "How long has it lasted?"])
    responder.gate.clear()  # the LLM stalls after its second delta
    call = Call(settings, responder)
    await call.caller_speaks()

    await wait_until(lambda: call.audio() == ["Thanks for telling me."])
    assert call.transcript("assistant") == []  # still talking

    responder.gate.set()
    await call.reply_finished()
    assert call.audio() == ["Thanks for telling me.", "How long has it lasted?"]
    await call.session.close()


@pytest.mark.parametrize(
    "script", [[ReplyError("boom")], [], ["   "]], ids=["error", "empty", "blank"]
)
async def test_caller_hears_fallback_when_no_reply_is_generated(
    settings: Settings, script: list[str | Exception]
) -> None:
    call = Call(settings, FakeReplyGenerator(script))
    await call.caller_speaks()
    await call.reply_finished()

    assert call.audio() == [settings.reply_fallback_text]
    assert "your local emergency number" in call.audio()[0]
    assert call.transcript("assistant") == [settings.reply_fallback_text]
    # The canned line is not model output, so it must not count as LLM latency.
    assert "llm_first_sentence_ms" not in call.session.metrics.summary()
    assert call.session.metrics.summary()["response_latency_ms"].count == 1
    await call.session.close()


async def test_reply_that_fails_midway_is_cut_short_without_fallback(settings: Settings) -> None:
    call = Call(settings, FakeReplyGenerator(["I'm sorry to hear that. ", ReplyError("boom")]))
    await call.caller_speaks()
    await call.reply_finished()

    assert call.audio() == ["I'm sorry to hear that."]
    assert call.transcript("assistant") == ["I'm sorry to hear that."]
    await call.session.close()


async def test_tts_failure_is_survived_and_only_heard_text_is_recorded(settings: Settings) -> None:
    call = Call(
        settings,
        FakeReplyGenerator(REPLY),
        tts=FakeTTSProvider(fail_after=0),
        utterances=["I have a headache.", "Hello?"],
    )
    await call.caller_speaks()
    await call.reply_finished()
    assert call.frames == []
    assert call.transcript("assistant") == []

    call.tts.fail_after = 1  # second reply: one sentence is heard, then synthesis fails
    await call.caller_speaks()
    await call.reply_finished()
    assert call.audio() == [SENTENCES[0]]
    assert [f["event"] for f in call.frames] == ["media"]  # no mark: playback never completed
    reply = call.session.state.conversation_history[-1]
    assert (reply.role, reply.interrupted) == ("assistant", True)
    await call.session.close()


async def test_caller_speaking_during_a_reply_gets_one_further_reply(settings: Settings) -> None:
    responder = FakeReplyGenerator(["How long has it lasted? "], ["Thank you, that helps. "])
    responder.gate.clear()
    call = Call(
        settings, responder, utterances=["I have a headache.", "Three days.", "And a fever."]
    )
    await call.caller_speaks()
    await wait_until(lambda: len(responder.histories) == 1)
    await call.caller_speaks()
    await call.caller_speaks()  # two utterances while the first reply is in progress

    responder.gate.set()
    await call.reply_finished()

    assert len(responder.histories) == 2
    # The first reply sits where it was said, so the model sees the turns in order
    # and the conversation still ends on the caller's words.
    assert [(m.role, m.content) for m in responder.histories[1]] == [
        ("patient", "I have a headache."),
        ("assistant", "How long has it lasted?"),
        ("patient", "Three days."),
        ("patient", "And a fever."),
    ]
    assert call.transcript("assistant") == ["How long has it lasted?", "Thank you, that helps."]
    assert [f["mark"]["name"] for f in call.frames if f["event"] == "mark"] == [
        "reply-1",
        "reply-2",
    ]
    await call.session.close()


async def test_hang_up_mid_reply_stops_speaking(settings: Settings) -> None:
    responder = FakeReplyGenerator(["Thanks for telling me. ", "How long has it lasted?"])
    responder.gate.clear()
    call = Call(settings, responder)
    await call.caller_speaks()
    await wait_until(lambda: call.audio() == ["Thanks for telling me."])

    await call.session.close()

    assert call.audio() == ["Thanks for telling me."]
    assert call.tts.streams_closed == 1
    reply = call.session.state.conversation_history[-1]
    assert (reply.content, reply.interrupted) == ("Thanks for telling me.", True)


async def test_send_failure_does_not_break_the_call(settings: Settings) -> None:
    async def twilio_gone(frame: dict[str, Any]) -> None:
        raise RuntimeError("websocket closed")

    call = Call(settings, FakeReplyGenerator(REPLY), on_frame=twilio_gone)
    await call.caller_speaks()
    await call.reply_finished()
    assert call.tts.streams_closed == 1
    await call.session.close()
    assert call.transcript("patient") == ["I have a headache."]


async def test_words_flushed_at_hang_up_are_not_answered(settings: Settings) -> None:
    responder = FakeReplyGenerator(REPLY)
    session = CallSession(
        call_sid="CA1",
        stream_sid="MZ1",
        stt=FakeSTTProvider([final("Bye.", speech_final=True)]),  # arrives during close()
        settings=settings,
        responder=responder,
        tts=FakeTTSProvider(),
        send=lambda raw: asyncio.sleep(0),
    )
    session.start()
    session.feed_audio(FRAME)
    await session.close()

    assert [m.content for m in session.state.conversation_history] == ["Bye."]
    assert responder.histories == []


def test_session_without_reply_providers_only_transcribes(settings: Settings) -> None:
    session = CallSession(
        call_sid="CA1", stream_sid="MZ1", stt=FakeSTTProvider([]), settings=settings
    )
    assert session.replies_enabled is False


# --- end to end through the Media Stream endpoint ----------------------------


def test_media_stream_sends_reply_audio_back_to_twilio(settings: Settings) -> None:
    app = create_app(
        settings,
        stt_provider=LiveFakeSTTProvider([final("I have a headache.", speech_final=True)]),
        tts_provider=FakeTTSProvider(),
        reply_generator=FakeReplyGenerator(REPLY),
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["tts"] == {"provider": "fake", "configured": True}
        assert health["llm"] == {"provider": "fake", "configured": True}

        with client.websocket_connect("/twilio/media-stream") as ws:
            ws.send_text(twilio_start("CA900", token=stream_token("CA900")))
            ws.send_text(twilio_media(FRAME))
            frames = [json.loads(ws.receive_text()) for _ in range(3)]
            ws.send_text(twilio_stop("CA900"))

        assert [f["event"] for f in frames] == ["media", "media", "mark"]
        assert [base64.b64decode(f["media"]["payload"]).decode() for f in frames[:2]] == SENTENCES
        assert all(f["streamSid"] == "MZtest" for f in frames)

        call = client.get("/api/calls/CA900").json()
        assert [(m["role"], m["content"]) for m in call["transcript"]] == [
            ("patient", "I have a headache."),
            ("assistant", " ".join(SENTENCES)),
        ]
        assert call["replies_spoken"] == 1
        assert call["latency"]["response_latency_ms"]["count"] == 1
