"""FastAPI application factory.

Run with:  uvicorn app.main:create_app --factory
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app import __version__
from app.agents.responder import ReplyGenerator, build_reply_generator
from app.api import calls
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.voice import media_stream, twilio_webhook
from app.voice.providers.base import STTProvider, TTSProvider
from app.voice.providers.factory import build_stt_provider, build_tts_provider
from app.voice.registry import CallRegistry

log = get_logger(__name__)

_UNSET: Any = object()


def create_app(
    settings: Settings | None = None,
    stt_provider: STTProvider | None = _UNSET,
    tts_provider: TTSProvider | None = _UNSET,
    reply_generator: ReplyGenerator | None = _UNSET,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    registry = CallRegistry()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info(
            "app.startup",
            environment=settings.environment,
            stt_provider=getattr(app.state.stt_provider, "name", None),
            tts_provider=getattr(app.state.tts_provider, "name", None),
            llm=getattr(app.state.reply_generator, "name", None),
            public_base_url=settings.public_base_url,
        )
        yield
        for session in registry.sessions():
            await session.close()
            registry.finish(session.call_sid)
        log.info("app.shutdown")

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.stt_provider = (
        build_stt_provider(settings) if stt_provider is _UNSET else stt_provider
    )
    app.state.tts_provider = (
        build_tts_provider(settings) if tts_provider is _UNSET else tts_provider
    )
    app.state.reply_generator = (
        build_reply_generator(settings) if reply_generator is _UNSET else reply_generator
    )
    app.state.call_registry = registry

    app.include_router(twilio_webhook.router)
    app.include_router(media_stream.router)
    app.include_router(calls.router)

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, Any]:
        stt: STTProvider | None = app.state.stt_provider
        tts: TTSProvider | None = app.state.tts_provider
        llm: ReplyGenerator | None = app.state.reply_generator
        return {
            "status": "ok",
            "version": __version__,
            "environment": settings.environment,
            "stt": {"provider": stt.name if stt else None, "configured": stt is not None},
            "tts": {"provider": tts.name if tts else None, "configured": tts is not None},
            "llm": {"provider": llm.name if llm else None, "configured": llm is not None},
            "active_calls": len(registry.sessions()),
        }

    return app
