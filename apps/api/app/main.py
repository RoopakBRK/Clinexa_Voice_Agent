"""FastAPI application factory.

Run with:  uvicorn app.main:create_app --factory
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.agents.onboarding import ToolExecutor, build_onboarding_generator
from app.agents.responder import ReplyGenerator, build_reply_generator
from app.api import calls
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.medicines.lookup import MedicineLookup, build_medicine_lookup
from app.notify import routes as alert_routes
from app.notify.exotel import ExotelClient, build_exotel_client
from app.notify.scheduler import AlertScheduler
from app.notify.store import AlertStore, build_alert_store
from app.observability.tracing import (
    configure_tracing,
    flush_tracing,
    instrument_app,
    tracing_enabled,
)
from app.rag.retrieval.service import GuidelineSearch, build_guideline_search
from app.tools.knowledge import KnowledgeTools
from app.voice import exotel_stream, media_stream, twilio_webhook, web_stream
from app.voice.languages import DEFAULT_VOICE, Language
from app.voice.providers.base import STTProvider, TTSProvider
from app.voice.providers.factory import build_stt_provider, build_tts_provider
from app.voice.registry import CallRegistry
from app.voice.sign_in import SignInCheck, build_sign_in_check

log = get_logger(__name__)

_UNSET: Any = object()


def create_app(
    settings: Settings | None = None,
    stt_provider: STTProvider | None = _UNSET,
    tts_provider: TTSProvider | None = _UNSET,
    reply_generator: ReplyGenerator | None = _UNSET,
    onboarding_factory: web_stream.AgentFactory = _UNSET,
    exotel: ExotelClient | None = _UNSET,
    alert_store: AlertStore | None = _UNSET,
    sign_in: SignInCheck | None = _UNSET,
    medicine_lookup: MedicineLookup | None = _UNSET,
    guideline_search: GuidelineSearch | None = _UNSET,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    configure_tracing(settings)
    registry = CallRegistry()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info(
            "app.startup",
            environment=settings.environment,
            stt_provider=getattr(app.state.stt_provider, "name", None),
            tts_provider=getattr(app.state.tts_provider, "name", None),
            llm=getattr(app.state.reply_generator, "name", None),
            medicines=app.state.medicine_lookup is not None,
            guidelines=app.state.guideline_search is not None,
            public_base_url=settings.public_base_url,
        )
        # The alerts worker runs here on a timer, once everything it needs is in place.
        scheduler: AlertScheduler | None = None
        if (
            settings.alerts_enabled
            and settings.alerts_scheduler
            and app.state.exotel is not None
            and app.state.alert_store is not None
        ):
            scheduler = AlertScheduler(app.state.alert_store, app.state.exotel, settings)
            scheduler.start()
        # Models and indexes take seconds to load, so they load in the background. Until
        # they are in, a lookup answers that it could not be made, and the call goes on.
        knowledge = (app.state.medicine_lookup, app.state.guideline_search)
        for resource in knowledge:
            if resource is not None:
                resource.start()
        yield
        if scheduler is not None:
            await scheduler.stop()
        for session in registry.sessions():
            await session.close()
            registry.finish(session.call_sid)
        for resource in (app.state.exotel, app.state.alert_store, app.state.web_sign_in):
            close = getattr(resource, "aclose", None)
            if close is not None:
                await close()
        for resource in knowledge:
            if resource is not None:
                await resource.aclose()
        flush_tracing()
        log.info("app.shutdown")

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.stt_provider = (
        build_stt_provider(settings) if stt_provider is _UNSET else stt_provider
    )
    app.state.tts_provider = (
        build_tts_provider(settings) if tts_provider is _UNSET else tts_provider
    )

    # What Claude can look up during a call (app/tools/knowledge.py): the Indian medicines
    # catalogue and the WHO guidance. Only built where there is a Claude to use them, and
    # each is None where it has nothing to search: replies then go without that lookup.
    wanted = reply_generator is _UNSET and settings.anthropic_api_key is not None
    if medicine_lookup is _UNSET:
        medicine_lookup = build_medicine_lookup(settings) if wanted else None
    if guideline_search is _UNSET:
        guideline_search = build_guideline_search(settings) if wanted else None
    app.state.medicine_lookup = medicine_lookup
    app.state.guideline_search = guideline_search
    tools = KnowledgeTools(medicines=medicine_lookup, guidelines=guideline_search)

    app.state.reply_generator = (
        build_reply_generator(settings, tools) if reply_generator is _UNSET else reply_generator
    )
    app.state.call_registry = registry
    # Patient alerts (WhatsApp and voice calls through Exotel). Separate from sign-in:
    # nothing below reads a user session. Both stay None until they are configured.
    app.state.exotel = build_exotel_client(settings) if exotel is _UNSET else exotel
    app.state.alert_store = build_alert_store(settings) if alert_store is _UNSET else alert_store

    # Web onboarding sessions pick their own language and voice, so providers are
    # built per session. Providers injected above (tests) are used as they are.
    def web_stt(language: Language) -> STTProvider | None:
        if stt_provider is not _UNSET:
            return stt_provider
        return build_stt_provider(
            settings, language=language.stt_language, keyterms=settings.onboarding_keyterms
        )

    def web_tts(language: Language) -> TTSProvider | None:
        if tts_provider is not _UNSET:
            return tts_provider
        voice = None if language.tts_model == DEFAULT_VOICE else language.tts_model
        return build_tts_provider(settings, model=voice)

    def web_agent(
        execute_tool: ToolExecutor, *, language: str, first_name: str | None, resumed: bool
    ) -> ReplyGenerator | None:
        return build_onboarding_generator(
            settings, execute_tool, language=language, first_name=first_name, resumed=resumed
        )

    app.state.web_stt_factory = web_stt
    app.state.web_tts_factory = web_tts
    app.state.web_agent_factory = web_agent if onboarding_factory is _UNSET else onboarding_factory
    # Only someone signed in to the website may start a session (app/voice/sign_in.py).
    app.state.web_sign_in = build_sign_in_check(settings) if sign_in is _UNSET else sign_in
    app.state.web_rate_limiter = web_stream.RateLimiter(settings.web_sessions_per_minute)
    app.state.web_used_tokens = web_stream.UsedTokens()

    # Only the website's origins may call POST /web/session from a browser.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.web_allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "authorization"],
    )

    app.include_router(twilio_webhook.router)
    app.include_router(media_stream.router)
    app.include_router(web_stream.router)
    app.include_router(calls.router)
    app.include_router(alert_routes.router)
    app.include_router(exotel_stream.router)
    # Last, so that it wraps everything above. A no-op unless LOGFIRE_TOKEN is set.
    instrument_app(app)

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, Any]:
        stt: STTProvider | None = app.state.stt_provider
        tts: TTSProvider | None = app.state.tts_provider
        llm: ReplyGenerator | None = app.state.reply_generator
        medicines: MedicineLookup | None = app.state.medicine_lookup
        guidelines: GuidelineSearch | None = app.state.guideline_search
        return {
            "status": "ok",
            "version": __version__,
            "environment": settings.environment,
            "stt": {"provider": stt.name if stt else None, "configured": stt is not None},
            "tts": {"provider": tts.name if tts else None, "configured": tts is not None},
            "llm": {"provider": llm.name if llm else None, "configured": llm is not None},
            "knowledge": {
                "medicines": {
                    "configured": medicines is not None,
                    "collection": settings.medicines_collection,
                    "status": medicines.status if medicines else None,
                },
                "guidelines": {
                    "configured": guidelines is not None,
                    "collection": settings.qdrant_collection,
                    "status": guidelines.status if guidelines else None,
                },
            },
            "tracing": {
                "provider": "logfire" if tracing_enabled() else None,
                "configured": tracing_enabled(),
            },
            "active_calls": len(registry.sessions()),
        }

    return app
