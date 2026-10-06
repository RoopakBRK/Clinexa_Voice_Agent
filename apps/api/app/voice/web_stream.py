"""Browser channel for the Clinexsa website's "Onboarding for Patient" page.

``POST /web/session`` hands the page a short-lived token; the page then opens
``/web/onboarding-stream`` and streams microphone audio (binary linear16 16 kHz)
while Roopiee speaks back (binary linear16 24 kHz) and fills the page's form
through ``tool_call`` / ``tool_result`` messages.

No provider keys ever reach the browser.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, status
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect, WebSocketState

from app.agents.onboarding import ToolExecutor
from app.agents.responder import ReplyGenerator
from app.core.config import Settings
from app.core.logging import get_logger
from app.voice.deps import get_call_registry, get_settings
from app.voice.languages import LANGUAGES, Language, resolve_language
from app.voice.providers.base import STTProvider, TTSProvider
from app.voice.registry import CallRegistry
from app.voice.security import sign_web_token, verify_web_token
from app.voice.session import CallSession
from app.voice.sign_in import SignInCheck, SignInUnavailableError
from app.voice.transport import WebTransport
from app.voice.turns import Utterance
from app.voice.web_protocol import (
    FormEditMessage,
    PauseMessage,
    ResumeMessage,
    StartMessage,
    StopMessage,
    ToolResultMessage,
    parse_web_inbound,
)

log = get_logger(__name__)

router = APIRouter(prefix="/web", tags=["web"])

WS_POLICY_VIOLATION = 1008
WS_INTERNAL_ERROR = 1011
_START_TIMEOUT_S = 10.0

STTFactory = Callable[[Language], STTProvider | None]
TTSFactory = Callable[[Language], TTSProvider | None]


class AgentFactory(Protocol):
    def __call__(
        self,
        execute_tool: ToolExecutor,
        *,
        language: str,
        first_name: str | None,
        resumed: bool,
    ) -> ReplyGenerator | None: ...


class RateLimiter:
    """Sliding one-minute window per key. In-process, like the call registry."""

    def __init__(self, per_minute: int) -> None:
        self._per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str, *, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] > 60.0:
            hits.popleft()
        if len(hits) >= self._per_minute:
            return False
        hits.append(now)
        return True


class UsedTokens:
    """Remembers tokens until they expire, so each opens exactly one stream."""

    def __init__(self) -> None:
        self._expiry: dict[str, int] = {}

    def claim(self, token: str, *, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        self._expiry = {t: exp for t, exp in self._expiry.items() if exp >= now}
        if token in self._expiry:
            return False
        expiry, _, _ = token.partition(".")
        self._expiry[token] = int(expiry) if expiry.isdigit() else int(now)
        return True


# --- HTTP ---------------------------------------------------------------------


class WebSessionRequest(BaseModel):
    onboarding_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    language: str = "en"


class WebSessionResponse(BaseModel):
    ws_url: str
    token: str
    expires_in: int


class LanguageInfo(BaseModel):
    code: str
    name: str
    # False until a voice exists for the language; the page shows "Coming soon".
    speech: bool


@router.get("/languages")
async def languages() -> list[LanguageInfo]:
    return [
        LanguageInfo(code=lang.code, name=lang.name, speech=lang.tts_model is not None)
        for lang in LANGUAGES.values()
    ]


@router.post("/session")
async def create_web_session(
    body: WebSessionRequest, request: Request, settings: Settings = Depends(get_settings)
) -> WebSessionResponse:
    limiter: RateLimiter = request.app.state.web_rate_limiter
    client_ip = request.client.host if request.client else "unknown"
    if not limiter.allow(client_ip):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "Too many sessions. Try again in a minute."
        )

    await _require_sign_in(request, settings)

    language = resolve_language(body.language)
    if language is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            {"code": "unknown_language", "message": "That language isn't offered."},
        )
    if language.tts_model is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            {
                "code": "language_unavailable",
                "message": f"Roopiee can't speak {language.name} yet. Coming soon.",
            },
        )

    ttl = settings.web_token_ttl_s
    token = sign_web_token(
        settings.stream_token_secret.get_secret_value(), body.onboarding_id, int(time.time()) + ttl
    )
    return WebSessionResponse(ws_url=_stream_url(request), token=token, expires_in=ttl)


async def _require_sign_in(request: Request, settings: Settings) -> None:
    """Lets the request through only for someone signed in to the website."""
    check: SignInCheck | None = request.app.state.web_sign_in
    if check is None:
        # A laptop and the tests run without the check. A public server must not.
        if settings.environment == "production":
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                {"code": "unavailable", "message": "Roopiee's sign-in check isn't set up."},
            )
        return

    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    try:
        signed_in = scheme.lower() == "bearer" and bool(token) and await check(token.strip())
    except SignInUnavailableError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {"code": "unavailable", "message": "Roopiee couldn't check your sign-in just now."},
        ) from None
    if not signed_in:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            {"code": "signed_out", "message": "Sign in to talk to Roopiee."},
        )


def _stream_url(request: Request) -> str:
    # Built from the address the browser reached us on, so it works the same on
    # localhost, behind ngrok and behind a load balancer.
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "localhost")
    return f"{'wss' if secure else 'ws'}://{host}{router.prefix}/onboarding-stream"


# --- WebSocket ----------------------------------------------------------------


@router.websocket("/onboarding-stream")
async def onboarding_stream(
    websocket: WebSocket,
    settings: Settings = Depends(get_settings),
    registry: CallRegistry = Depends(get_call_registry),
) -> None:
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in settings.web_allowed_origins:
        log.warning("web.rejected", reason="origin_not_allowed")
        await websocket.close(code=WS_POLICY_VIOLATION)
        return
    await websocket.accept()
    connection = _OnboardingConnection(websocket, settings, registry)
    try:
        await connection.run()
    finally:
        await connection.close()
        structlog.contextvars.unbind_contextvars("call_sid")


class _OnboardingConnection:
    def __init__(self, websocket: WebSocket, settings: Settings, registry: CallRegistry) -> None:
        self._ws = websocket
        self._settings = settings
        self._registry = registry
        self._transport = WebTransport(websocket)
        self._session: CallSession | None = None
        self._agent: ReplyGenerator | None = None
        # Tool calls sent to the page and not answered yet, by tool_use id.
        self._pending: dict[str, asyncio.Future[str]] = {}
        self._paused = False
        self._last_activity = time.monotonic()

    async def run(self) -> None:
        idle_timeout = self._settings.web_idle_timeout_s
        while True:
            timeout = _START_TIMEOUT_S if self._session is None else idle_timeout
            try:
                frame = await asyncio.wait_for(self._ws.receive(), timeout)
            except TimeoutError:
                await self._fail("idle", "The session was quiet for too long, so it was closed.")
                return
            if frame["type"] == "websocket.disconnect":
                return

            if (audio := frame.get("bytes")) is not None:
                if self._session is not None and not self._paused:
                    self._session.feed_audio(audio)
                # Mic frames keep arriving through silence, so idleness is judged
                # by speech and form activity, not by frames.
                if time.monotonic() - self._last_activity > idle_timeout:
                    await self._fail("idle", "The session was quiet for too long.")
                    return
                continue

            match parse_web_inbound(frame.get("text") or ""):
                case StartMessage() as msg if self._session is None:
                    if not await self._start(msg):
                        return
                case ToolResultMessage() as msg:
                    self._touch()
                    future = self._pending.get(msg.id)
                    if future is not None and not future.done():
                        future.set_result(msg.content)
                case FormEditMessage() as msg if self._agent is not None:
                    self._touch()
                    value = json.dumps(msg.value, ensure_ascii=False)
                    self._agent.add_note(
                        f"The person typed {msg.field} on the form themselves: {value}. "
                        "Don't ask for it again."
                    )
                case PauseMessage() if self._session is not None:
                    self._paused = True
                    await self._session.interrupt()
                case ResumeMessage() if self._session is not None and self._paused:
                    self._paused = False
                    self._touch()
                    if self._agent is not None:
                        self._agent.add_note(
                            "The person paused and is back now. Carry on from where you stopped."
                        )
                    self._session.request_reply()
                case StopMessage():
                    return
                case _:
                    pass

    async def close(self) -> None:
        for future in self._pending.values():
            future.cancel()
        if self._session is not None:
            self._registry.finish(self._session.call_sid)
            await self._session.close()
        if (
            self._ws.application_state is WebSocketState.CONNECTED
            and self._ws.client_state is WebSocketState.CONNECTED
        ):
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await self._ws.close()

    async def _start(self, msg: StartMessage) -> bool:
        settings = self._settings
        state = self._ws.app.state
        token = self._ws.query_params.get("token")
        secret = settings.stream_token_secret.get_secret_value()
        used: UsedTokens = state.web_used_tokens
        if not verify_web_token(secret, msg.onboarding_id, token) or not used.claim(token or ""):
            log.warning("web.rejected", reason="invalid_token")
            await self._fail("unauthorized", "This session link has expired.", WS_POLICY_VIOLATION)
            return False

        language = resolve_language(msg.language)
        if language is None or language.tts_model is None:
            name = language.name if language else "that language"
            await self._fail("language_unavailable", f"Roopiee can't speak {name} yet.")
            return False

        call_sid = f"web-{msg.onboarding_id}-{secrets.token_hex(3)}"
        # Tasks created from here on inherit this, so all session logs are correlated.
        structlog.contextvars.bind_contextvars(call_sid=call_sid)

        stt_factory: STTFactory = state.web_stt_factory
        tts_factory: TTSFactory = state.web_tts_factory
        agent_factory: AgentFactory = state.web_agent_factory
        stt, tts = stt_factory(language), tts_factory(language)
        resumed = bool(msg.form_state and msg.form_state.get("resumed"))
        agent = agent_factory(
            self._execute_tool, language=language.name, first_name=msg.first_name, resumed=resumed
        )
        if stt is None or tts is None or agent is None:
            log.error("web.rejected", reason="providers_not_configured")
            await self._fail(
                "unavailable", "Roopiee isn't set up on this server.", WS_INTERNAL_ERROR
            )
            return False

        self._agent = agent
        self._session = CallSession(
            call_sid=call_sid,
            stream_sid=call_sid,
            stt=stt,
            settings=settings,
            on_utterance=self._on_utterance,
            responder=agent,
            tts=tts,
            transport=self._transport,
            fallback_text=settings.onboarding_reply_fallback,
        )
        self._registry.add(self._session)
        self._session.start()
        log.info("web.session_started", language=language.code, resumed=resumed)
        # Roopiee speaks first.
        self._session.request_reply()
        return True

    async def _execute_tool(self, call_id: str, name: str, arguments: dict[str, Any]) -> str:
        """Run one of the agent's tools on the page and wait for its answer."""
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._pending[call_id] = future
        try:
            await self._transport.send_json(
                {"type": "tool_call", "id": call_id, "name": name, "arguments": arguments}
            )
            return await asyncio.wait_for(future, self._settings.web_tool_timeout_s)
        except TimeoutError:
            log.warning("web.tool_timeout", tool=name)
            return "timeout"
        finally:
            self._pending.pop(call_id, None)

    async def _on_utterance(self, session: CallSession, utterance: Utterance) -> None:
        self._touch()

    def _touch(self) -> None:
        self._last_activity = time.monotonic()

    async def _fail(self, code: str, message: str, close_code: int = 1000) -> None:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await self._transport.send_json({"type": "error", "code": code, "message": message})
            await self._ws.close(code=close_code)
