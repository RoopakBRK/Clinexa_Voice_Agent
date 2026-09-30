"""FastAPI dependencies for HTTP and WebSocket routes (resolved from app.state)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi.requests import HTTPConnection

from app.core.config import Settings

if TYPE_CHECKING:
    from app.voice.providers.base import STTProvider
    from app.voice.registry import CallRegistry


def get_settings(conn: HTTPConnection) -> Settings:
    return cast(Settings, conn.app.state.settings)


def get_stt_provider(conn: HTTPConnection) -> STTProvider | None:
    return cast("STTProvider | None", conn.app.state.stt_provider)


def get_call_registry(conn: HTTPConnection) -> CallRegistry:
    return cast("CallRegistry", conn.app.state.call_registry)
