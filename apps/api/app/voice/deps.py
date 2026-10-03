"""FastAPI dependencies for HTTP and WebSocket routes (resolved from app.state)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi.requests import HTTPConnection

from app.core.config import Settings

if TYPE_CHECKING:
    from app.agents.responder import ReplyGenerator
    from app.voice.providers.base import STTProvider, TTSProvider
    from app.voice.registry import CallRegistry


def get_settings(conn: HTTPConnection) -> Settings:
    return cast(Settings, conn.app.state.settings)


def get_stt_provider(conn: HTTPConnection) -> STTProvider | None:
    return cast("STTProvider | None", conn.app.state.stt_provider)


def get_tts_provider(conn: HTTPConnection) -> TTSProvider | None:
    return cast("TTSProvider | None", conn.app.state.tts_provider)


def get_reply_generator(conn: HTTPConnection) -> ReplyGenerator | None:
    return cast("ReplyGenerator | None", conn.app.state.reply_generator)


def get_call_registry(conn: HTTPConnection) -> CallRegistry:
    return cast("CallRegistry", conn.app.state.call_registry)
