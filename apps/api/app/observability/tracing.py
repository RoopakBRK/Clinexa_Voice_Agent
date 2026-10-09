"""Traces, sent to Pydantic Logfire over OpenTelemetry.

Off unless LOGFIRE_TOKEN is set: without it every span below is a no-op and nothing
leaves the machine. With it, a phone call becomes one trace:

    /twilio/media-stream               the call, from FastAPI's instrumentation
      reply                            one spoken answer (app/voice/session.py)
        llm round                      one request to Claude (app/agents/responder.py)
        tool lookup_medicine           a lookup Claude asked for (app/tools/knowledge.py)
          medicines lookup             the catalogue search and the rules
        tool search_guidelines
          guidelines search            dense + BM25, RRF, cross-encoder, with stage timings

A span says what ran, how long it took and how it went. It never carries what a caller
said or asked for. Transcripts, tool arguments, medicine names and search questions are
health information and Logfire is somebody else's server, so they stay out of spans just
as they stay out of logs. Request headers, bodies and endpoint arguments are not recorded
either, and endpoints whose address carries a secret are not traced at all.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.util.types import AttributeValue

from app import __version__
from app.core.config import Settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

log = get_logger(__name__)

# The code is instrumented with the OpenTelemetry API, which does nothing until a
# provider is set up. Logfire is that provider.
tracer = trace.get_tracer("clinexa")

# Not traced: the address of these carries a secret in its query string (the Exotel
# callback key, the single-use token of a web session).
_NOT_TRACED = "/exotel/,/web/onboarding-stream"

_enabled = False


def tracing_enabled() -> bool:
    return _enabled


def configure_tracing(settings: Settings) -> bool:
    """Start sending traces to Logfire, if a token is set. Returns whether tracing is on."""
    global _enabled
    if settings.logfire_token is None:
        return _enabled
    # Imported only where it is used: it brings the OpenTelemetry SDK and exporters with it.
    import logfire

    logfire.configure(
        token=settings.logfire_token.get_secret_value(),
        service_name=settings.logfire_service_name,
        service_version=__version__,
        environment=settings.environment,
        console=False,
        inspect_arguments=False,
    )
    _enabled = True
    log.info("tracing.enabled", provider="logfire", service=settings.logfire_service_name)
    return True


def instrument_app(app: FastAPI) -> None:
    """Trace the app's requests and calls. Does nothing while tracing is off."""
    if not _enabled:
        return
    import logfire

    logfire.instrument_fastapi(
        app,
        capture_headers=False,
        excluded_urls=_NOT_TRACED,
        # Endpoint arguments hold phone numbers and form answers: none of them is recorded.
        request_attributes_mapper=lambda request, attributes: None,
    )


def flush_tracing() -> None:
    """Send what is still waiting. Called as the server shuts down."""
    if _enabled:
        import logfire

        logfire.force_flush(timeout_millis=3000)


@contextmanager
def detached_span(name: str, **attributes: AttributeValue) -> Iterator[Span]:
    """A span that is never made the current one, for code that yields while it is open.

    An async generator can be closed from another task than the one that started it, and a
    span it had made current could then not be put back. This one leaves the context alone:
    it is a child of whatever span is current when it starts, and nothing more.
    """
    span = tracer.start_span(name, attributes=attributes)
    try:
        yield span
    except BaseException as exc:
        span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
        raise
    finally:
        span.end()
