"""Browser channel: token -> WebSocket -> speech in, speech and form tool calls out."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from app.agents.onboarding import OnboardingReplyGenerator, ToolExecutor
from app.agents.responder import ReplyGenerator
from app.core.config import Settings
from app.main import create_app
from app.voice.languages import resolve_language
from app.voice.providers.base import (
    AudioFormat,
    TranscriptEvent,
    TranscriptEventType,
    TTSProvider,
)
from app.voice.security import sign_web_token, verify_web_token
from app.voice.session import CallSession
from app.voice.sign_in import SignInCheck, SignInUnavailableError
from app.voice.transport import (
    WEB_INPUT_FORMAT,
    WEB_OUTPUT_FORMAT,
    SessionStatus,
    TranscriptRole,
    Transport,
)
from app.voice.web_protocol import (
    FormEditMessage,
    PauseMessage,
    StartMessage,
    ToolResultMessage,
    parse_web_inbound,
)
from app.voice.web_stream import RateLimiter, UsedTokens
from tests.conftest import (
    STREAM_SECRET,
    FakeReplyGenerator,
    FakeTTSProvider,
    LiveFakeSTTProvider,
    final,
)
from tests.test_onboarding_agent import FakeClaude, text_block, tool_use

ONBOARDING_ID = "onb-1234567890"
MIC_FRAME = b"\x00" * 640  # 20 ms of linear16 at 16 kHz


# --- tokens -------------------------------------------------------------------


def test_web_token_is_bound_to_the_onboarding_id_and_expires() -> None:
    now = int(time.time())
    token = sign_web_token(STREAM_SECRET, ONBOARDING_ID, now + 60)

    assert verify_web_token(STREAM_SECRET, ONBOARDING_ID, token)
    assert not verify_web_token(STREAM_SECRET, "onb-someone-else", token)
    assert not verify_web_token("another-secret", ONBOARDING_ID, token)
    assert not verify_web_token(STREAM_SECRET, ONBOARDING_ID, token, now=now + 61)
    assert not verify_web_token(STREAM_SECRET, ONBOARDING_ID, None)
    assert not verify_web_token(STREAM_SECRET, ONBOARDING_ID, "not-a-token")
    # Pushing the expiry out changes the signature it must match.
    forged = f"{now + 9999}.{token.partition('.')[2]}"
    assert not verify_web_token(STREAM_SECRET, ONBOARDING_ID, forged)


def test_token_can_be_claimed_once_and_rate_limit_slides() -> None:
    used = UsedTokens()
    token = sign_web_token(STREAM_SECRET, ONBOARDING_ID, int(time.time()) + 60)
    assert used.claim(token)
    assert not used.claim(token)

    limiter = RateLimiter(per_minute=2)
    assert limiter.allow("1.2.3.4", now=0.0)
    assert limiter.allow("1.2.3.4", now=1.0)
    assert not limiter.allow("1.2.3.4", now=2.0)
    assert limiter.allow("5.6.7.8", now=2.0)  # another visitor is unaffected
    assert limiter.allow("1.2.3.4", now=61.5)  # the first hit has aged out


# --- message parsing ------------------------------------------------------------


def test_inbound_messages_parse_by_type() -> None:
    start = parse_web_inbound(
        json.dumps({"type": "start", "onboarding_id": ONBOARDING_ID, "language": "hi"})
    )
    assert isinstance(start, StartMessage) and start.language == "hi" and start.form_state is None

    result = parse_web_inbound('{"type": "tool_result", "id": "tu_1", "content": "ok"}')
    assert isinstance(result, ToolResultMessage) and result.content == "ok"

    edit = parse_web_inbound('{"type": "form_edit", "field": "age", "value": 62}')
    assert isinstance(edit, FormEditMessage) and edit.value == 62

    assert isinstance(parse_web_inbound('{"type": "pause"}'), PauseMessage)
    assert parse_web_inbound('{"type": "launch_missiles"}') is None
    assert parse_web_inbound("not json") is None
    assert parse_web_inbound('{"type": "start", "onboarding_id": "x"}') is None  # id too short


def test_languages_without_a_voice_are_known_but_cannot_speak() -> None:
    english, hindi = resolve_language("en"), resolve_language("Hindi")
    assert english is not None and english.tts_model is not None
    assert hindi is not None and hindi.stt_language == "multi" and hindi.tts_model is None
    assert resolve_language("hi-IN") == hindi
    assert resolve_language("klingon") is None


# --- HTTP ----------------------------------------------------------------------


def make_app(
    settings: Settings,
    claude: FakeClaude,
    stt_script: list[TranscriptEvent] | None = None,
    sign_in: SignInCheck | None = None,
) -> FastAPI:
    def onboarding(
        execute_tool: ToolExecutor, *, language: str, first_name: str | None, resumed: bool
    ) -> ReplyGenerator:
        return OnboardingReplyGenerator(
            "sk-test",
            system_prompt="system",
            execute_tool=execute_tool,
            resumed=resumed,
            client=claude,  # type: ignore[arg-type]
        )

    return create_app(
        settings,
        stt_provider=LiveFakeSTTProvider(stt_script or []),
        tts_provider=FakeTTSProvider(),
        reply_generator=None,
        onboarding_factory=onboarding,
        sign_in=sign_in,
    )


def new_session(client: TestClient, language: str = "en") -> dict[str, Any]:
    response = client.post(
        "/web/session", json={"onboarding_id": ONBOARDING_ID, "language": language}
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_session_endpoint_returns_a_stream_url_and_token(settings: Settings) -> None:
    with TestClient(make_app(settings, FakeClaude())) as client:
        body = new_session(client)

    assert body["ws_url"] == "ws://testserver/web/onboarding-stream"
    assert body["expires_in"] == settings.web_token_ttl_s
    assert verify_web_token(STREAM_SECRET, ONBOARDING_ID, body["token"])


def test_session_endpoint_reports_languages_roopiee_cannot_speak(settings: Settings) -> None:
    with TestClient(make_app(settings, FakeClaude())) as client:
        response = client.post(
            "/web/session", json={"onboarding_id": ONBOARDING_ID, "language": "kn"}
        )
        listed = {lang["code"]: lang["speech"] for lang in client.get("/web/languages").json()}

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "language_unavailable"
    assert listed["en"] is True and listed["kn"] is False and len(listed) == 9


def test_session_endpoint_is_rate_limited_per_address(settings: Settings) -> None:
    limited = settings.model_copy(update={"web_sessions_per_minute": 2})
    with TestClient(make_app(limited, FakeClaude())) as client:
        codes = [
            client.post("/web/session", json={"onboarding_id": ONBOARDING_ID}).status_code
            for _ in range(3)
        ]
    assert codes == [200, 200, 429]


async def only_the_good_token(access_token: str) -> bool:
    return access_token == "good-token"


async def supabase_is_down(access_token: str) -> bool:
    raise SignInUnavailableError


def test_session_endpoint_is_only_for_someone_signed_in_to_the_website(settings: Settings) -> None:
    body = {"onboarding_id": ONBOARDING_ID}
    with TestClient(make_app(settings, FakeClaude(), sign_in=only_the_good_token)) as client:
        nobody = client.post("/web/session", json=body)
        stranger = client.post(
            "/web/session", json=body, headers={"Authorization": "Bearer made-up"}
        )
        wrong_kind = client.post(
            "/web/session", json=body, headers={"Authorization": "Basic good-token"}
        )
        signed_in = client.post(
            "/web/session", json=body, headers={"Authorization": "Bearer good-token"}
        )

    assert [nobody.status_code, stranger.status_code, wrong_kind.status_code] == [401, 401, 401]
    assert nobody.json()["detail"]["code"] == "signed_out"
    assert signed_in.status_code == 200
    assert verify_web_token(STREAM_SECRET, ONBOARDING_ID, signed_in.json()["token"])


def test_session_endpoint_stays_shut_when_sign_in_cannot_be_checked(settings: Settings) -> None:
    body = {"onboarding_id": ONBOARDING_ID}
    headers = {"Authorization": "Bearer good-token"}
    with TestClient(make_app(settings, FakeClaude(), sign_in=supabase_is_down)) as client:
        unreachable = client.post("/web/session", json=body, headers=headers)
    # A public server with no check configured lets nobody in. A laptop does (other tests).
    production = settings.model_copy(update={"environment": "production"})
    with TestClient(make_app(production, FakeClaude())) as client:
        unconfigured = client.post("/web/session", json=body, headers=headers)

    assert unreachable.status_code == 503
    assert unconfigured.status_code == 503
    assert unconfigured.json()["detail"]["code"] == "unavailable"


def test_the_website_may_send_the_sign_in_header(settings: Settings) -> None:
    with TestClient(make_app(settings, FakeClaude())) as client:
        preflight = client.options(
            "/web/session",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
    assert preflight.status_code == 200
    assert "authorization" in preflight.headers["access-control-allow-headers"].lower()


def test_only_the_website_origins_pass_cors(settings: Settings) -> None:
    preflight = {"Access-Control-Request-Method": "POST"}
    with TestClient(make_app(settings, FakeClaude())) as client:
        ours = client.options(
            "/web/session", headers={"Origin": "http://localhost:3000", **preflight}
        )
        theirs = client.options(
            "/web/session", headers={"Origin": "https://evil.example", **preflight}
        )

    assert ours.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-origin" not in theirs.headers


# --- WebSocket ------------------------------------------------------------------


def read_until(
    ws: WebSocketTestSession, stop: Callable[[dict[str, Any]], bool], limit: int = 60
) -> list[dict[str, Any] | bytes]:
    """Collect frames (JSON as dicts, audio as bytes) up to the first one ``stop`` accepts."""
    frames: list[dict[str, Any] | bytes] = []
    for _ in range(limit):
        raw = ws.receive()
        if raw.get("bytes") is not None:
            frames.append(raw["bytes"])
            continue
        message: dict[str, Any] = json.loads(raw["text"])
        frames.append(message)
        if stop(message):
            return frames
    raise AssertionError(f"expected frame never arrived; got {frames}")


def is_type(kind: str, **fields: Any) -> Callable[[dict[str, Any]], bool]:
    return lambda m: m.get("type") == kind and all(m.get(k) == v for k, v in fields.items())


def start(ws: WebSocketTestSession, **extra: Any) -> None:
    ws.send_json({"type": "start", "onboarding_id": ONBOARDING_ID, "language": "en", **extra})


def said(frames: list[dict[str, Any] | bytes], role: str) -> str:
    return " ".join(
        f["text"]
        for f in frames
        if isinstance(f, dict)
        and f.get("type") == "transcript"
        and f["role"] == role
        and f["final"]
    )


def test_roopiee_speaks_first_then_fills_the_form_from_speech(settings: Settings) -> None:
    claude = FakeClaude(
        ([tool_use("tu_1", "get_form_state")], "tool_use"),
        ([text_block("Hello, I am Roopiee. What is your name?")], "end_turn"),
        (
            [tool_use("tu_2", "set_field", field="full_name", value="Ramesh", confidence=0.9)],
            "tool_use",
        ),
        ([text_block("Thank you, Ramesh. How old are you?")], "end_turn"),
    )
    app = make_app(settings, claude, [final("My name is Ramesh.", speech_final=True)])

    with TestClient(app) as client:
        token = new_session(client)["token"]
        with client.websocket_connect(f"/web/onboarding-stream?token={token}") as ws:
            start(ws)

            # Opening turn: the agent reads the form before it says anything.
            call = read_until(ws, is_type("tool_call"))[-1]
            assert isinstance(call, dict)
            assert (call["id"], call["name"], call["arguments"]) == ("tu_1", "get_form_state", {})
            ws.send_json(
                {"type": "tool_result", "id": "tu_1", "content": '{"empty": ["full_name"]}'}
            )

            opening = read_until(ws, is_type("status", value="listening"))
            assert said(opening, "assistant") == "Hello, I am Roopiee. What is your name?"
            assert any(isinstance(f, bytes) for f in opening)  # the reply came as audio too
            assert {"type": "status", "value": "speaking"} in opening

            # The person answers out loud.
            ws.send_bytes(MIC_FRAME)
            heard = read_until(ws, is_type("tool_call"))
            assert said(heard, "user") == "My name is Ramesh."
            fill = heard[-1]
            assert isinstance(fill, dict)
            assert fill["name"] == "set_field"
            assert fill["arguments"] == {"field": "full_name", "value": "Ramesh", "confidence": 0.9}
            ws.send_json({"type": "tool_result", "id": "tu_2", "content": "ok"})

            reply = read_until(ws, is_type("status", value="listening"))
            assert said(reply, "assistant") == "Thank you, Ramesh. How old are you?"
            ws.send_json({"type": "stop"})

    # Claude saw the page's answers to both tool calls.
    assert claude.requests[1]["messages"][2]["content"][0]["content"] == '{"empty": ["full_name"]}'
    assert claude.requests[3]["messages"][-1]["content"][0]["content"] == "ok"
    assert app.state.call_registry.sessions() == []


def test_unanswered_tool_call_times_out_and_the_agent_carries_on(settings: Settings) -> None:
    quick = settings.model_copy(update={"web_tool_timeout_s": 0.05})
    claude = FakeClaude(
        ([tool_use("tu_1", "get_form_state")], "tool_use"),
        ([text_block("Hello there.")], "end_turn"),
    )
    with TestClient(make_app(quick, claude)) as client:
        token = new_session(client)["token"]
        with client.websocket_connect(f"/web/onboarding-stream?token={token}") as ws:
            start(ws)
            frames = read_until(ws, is_type("status", value="listening"))  # no tool_result sent

    assert said(frames, "assistant") == "Hello there."
    assert claude.requests[1]["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "timeout"}
    ]


def test_typed_edits_reach_the_agent_as_notes(settings: Settings) -> None:
    claude = FakeClaude(
        ([text_block("Hello. What is your name?")], "end_turn"),
        ([text_block("Thanks.")], "end_turn"),
    )
    app = make_app(settings, claude, [final("That is done.", speech_final=True)])
    with TestClient(app) as client:
        token = new_session(client)["token"]
        with client.websocket_connect(f"/web/onboarding-stream?token={token}") as ws:
            start(ws, form_state={"resumed": True})
            read_until(ws, is_type("status", value="listening"))
            ws.send_json({"type": "form_edit", "field": "full_name", "value": "Ramesh Kumar"})
            ws.send_bytes(MIC_FRAME)
            read_until(ws, is_type("status", value="listening"))

    assert "reconnected" in claude.requests[0]["messages"][0]["content"]
    latest = claude.requests[1]["messages"][-1]["content"]
    assert 'typed full_name on the form themselves: "Ramesh Kumar"' in latest
    assert latest.endswith("That is done.")


def test_stream_rejects_a_bad_or_reused_token(settings: Settings) -> None:
    claude = FakeClaude(([text_block("Hello.")], "end_turn"))
    with TestClient(make_app(settings, claude)) as client:
        token = new_session(client)["token"]
        with client.websocket_connect(f"/web/onboarding-stream?token={token}") as ws:
            start(ws)
            read_until(ws, is_type("status", value="listening"))

        for bad in (token, "123.deadbeef"):  # used once already / never valid
            with client.websocket_connect(f"/web/onboarding-stream?token={bad}") as ws:
                start(ws)
                error = ws.receive_json()
                assert (error["type"], error["code"]) == ("error", "unauthorized")


def test_stream_rejects_other_origins(settings: Settings) -> None:
    with TestClient(make_app(settings, FakeClaude())) as client:
        token = new_session(client)["token"]
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect(
                f"/web/onboarding-stream?token={token}", headers={"origin": "https://evil.example"}
            ),
        ):
            pass


# --- barge-in --------------------------------------------------------------------


class RecordingTransport(Transport):
    input_format = WEB_INPUT_FORMAT
    output_format = WEB_OUTPUT_FORMAT
    supports_barge_in = True

    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.cleared = 0
        self.statuses: list[SessionStatus] = []
        self.transcripts: list[tuple[TranscriptRole, str, bool]] = []

    async def send_audio(self, chunk: bytes) -> None:
        self.audio.append(chunk)

    async def clear_audio(self) -> None:
        self.cleared += 1

    async def send_transcript(self, role: TranscriptRole, text: str, *, final: bool) -> None:
        self.transcripts.append((role, text, final))

    async def send_status(self, value: SessionStatus) -> None:
        self.statuses.append(value)


class OneSecondTTS(TTSProvider):
    """One second of audio per sentence, so playback outlasts the test's next step."""

    name = "fake-slow"

    async def synthesize_stream(
        self, text: AsyncIterator[str], audio_format: AudioFormat
    ) -> AsyncIterator[bytes]:
        async for _ in text:
            yield b"\x00" * audio_format.bytes_per_second


async def eventually(condition: Callable[[], bool], timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "condition never became true"
        await asyncio.sleep(0.01)


async def test_speech_over_a_reply_cancels_it_and_clears_queued_audio(settings: Settings) -> None:
    stt = LiveFakeSTTProvider(
        [
            final("Hello there.", speech_final=True),
            TranscriptEvent(type=TranscriptEventType.SPEECH_STARTED),
        ]
    )
    replies = FakeReplyGenerator(["One moment please. Let", " me check that for you."])
    replies.gate.clear()  # hold the reply open after its first sentence
    transport = RecordingTransport()
    session = CallSession(
        call_sid="web-test",
        stream_sid="web-test",
        stt=stt,
        settings=settings,
        responder=replies,
        tts=OneSecondTTS(),
        transport=transport,
    )
    session.start()

    session.feed_audio(MIC_FRAME)
    await eventually(lambda: len(transport.audio) == 1)  # first sentence is playing
    assert transport.statuses[-1] == "speaking"

    session.feed_audio(MIC_FRAME)  # the person starts talking
    await eventually(lambda: transport.cleared == 1)
    assert transport.statuses[-1] == "listening"

    reply = session.state.conversation_history[-1]
    assert (reply.role, reply.content, reply.interrupted) == (
        "assistant",
        "One moment please.",
        True,
    )
    await session.close()


async def test_phone_calls_are_never_cut_off_by_speech_events(settings: Settings) -> None:
    stt = LiveFakeSTTProvider(
        [
            final("Hello there.", speech_final=True),
            TranscriptEvent(type=TranscriptEventType.SPEECH_STARTED),
        ]
    )
    sent: list[str] = []

    async def send(frame: str) -> None:
        sent.append(frame)

    session = CallSession(
        call_sid="CA1",
        stream_sid="MZ1",
        stt=stt,
        settings=settings,
        responder=FakeReplyGenerator(["I am sorry to hear that."]),
        tts=FakeTTSProvider(),
        send=send,
    )
    session.start()
    session.feed_audio(b"\xff" * 160)
    await eventually(lambda: session.replies_spoken == 1)
    session.feed_audio(b"\xff" * 160)
    await session.close()

    assert [json.loads(f)["event"] for f in sent] == ["media", "mark"]  # no "clear"
    assert session.state.conversation_history[-1].interrupted is False
