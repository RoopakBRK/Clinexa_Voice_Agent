"""Tracing: what a call's trace looks like, and what must never be in one."""

from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from logfire.testing import CaptureLogfire
from opentelemetry.trace import StatusCode
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient

from app.agents.responder import ReplyError
from app.core.config import Settings
from app.main import create_app
from app.medicines.lookup import MedicineLookup
from app.medicines.store import MedicineStore
from app.observability import tracing
from app.observability.tracing import configure_tracing, detached_span, tracer, tracing_enabled
from app.rag.retrieval.service import GuidelineSearch
from app.tools.knowledge import LOOKUP_MEDICINE, SEARCH_GUIDELINES, KnowledgeTools
from app.voice.session import CallSession
from tests.conftest import FakeReplyGenerator, FakeTTSProvider, LiveFakeSTTProvider, final
from tests.test_knowledge import (
    FakeGuidelines,
    FakeMedicines,
    Index,
    Round,
    ScriptedClaude,
    caller,
    claude,
    ready,
    spoken,
    text,
    tool_use,
)
from tests.test_medicines import CATALOG

FRAME = b"\xff" * 160


def spans(capfire: CaptureLogfire) -> list[dict[str, Any]]:
    return capfire.exporter.exported_spans_as_dict()


def named(capfire: CaptureLogfire, name: str) -> list[dict[str, Any]]:
    return [span for span in spans(capfire) if span["name"] == name]


def finished(capfire: CaptureLogfire, name: str) -> list[Any]:
    """The spans of a name as they ended, status and all (Logfire also sends one as each starts)."""
    return [
        span
        for span in capfire.exporter.exported_spans
        if span.name == name and (span.attributes or {}).get("logfire.span_type") != "pending_span"
    ]


def everything_sent(capfire: CaptureLogfire) -> str:
    """Every span as it would leave the machine, in one string to search."""
    return json.dumps(spans(capfire), default=str).lower()


# --- off unless asked for ------------------------------------------------------------------


def test_nothing_is_traced_without_a_token(settings: Settings) -> None:
    assert settings.logfire_token is None
    assert configure_tracing(settings) is False and tracing_enabled() is False
    # A blank token in .env is no token.
    assert Settings(_env_file=None, logfire_token="  ").logfire_token is None  # type: ignore[arg-type]

    with TestClient(create_app(settings, stt_provider=None)) as http:
        assert http.get("/health").json()["tracing"] == {"provider": None, "configured": False}
    # Spans are still opened by the code, and go nowhere.
    with tracer.start_as_current_span("reply") as span, detached_span("llm round") as inner:
        assert not span.is_recording() and not inner.is_recording()


# --- a reply is one trace -------------------------------------------------------------------


async def test_a_reply_is_traced_with_its_requests_and_lookups_and_none_of_their_content(
    capfire: CaptureLogfire,
) -> None:
    client = ScriptedClaude(
        Round(
            tool_use("tu_1", LOOKUP_MEDICINE, name="dolo 650"),
            tool_use("tu_2", SEARCH_GUIDELINES, query="fever in a child", age_years=3),
        ),
        Round(text("Dolo 650 contains paracetamol.")),
    )
    tools = KnowledgeTools(medicines=FakeMedicines(), guidelines=FakeGuidelines())
    with tracer.start_as_current_span("reply", attributes={"call_sid": "CA1"}):
        await spoken(claude(client, tools).stream_reply(caller("My son has a fever, dolo 650?")))

    (reply,) = named(capfire, "reply")
    rounds = sorted(named(capfire, "llm round"), key=lambda span: span["attributes"]["round"])
    assert [span["attributes"]["round"] for span in rounds] == [0, 1]
    assert rounds[0]["attributes"] | {"logfire.span_type": "", "logfire.msg": ""} == {
        "model": "claude-opus-5-5",
        "round": 0,
        "stop_reason": "tool_use",
        "input_tokens": 1,
        "output_tokens": 1,
        "cache_read_tokens": 0,
        "logfire.span_type": "",
        "logfire.msg": "",
    }
    assert rounds[1]["attributes"]["stop_reason"] == "end_turn"

    (medicine,) = named(capfire, f"tool {LOOKUP_MEDICINE}")
    (guidance,) = named(capfire, f"tool {SEARCH_GUIDELINES}")
    assert (medicine["attributes"]["ok"], medicine["attributes"]["detail"]) == (True, "exact")
    assert (guidance["attributes"]["ok"], guidance["attributes"]["detail"]) == (True, "2 passages")
    # Everything a reply does hangs off the reply.
    for span in (*rounds, medicine, guidance):
        assert span["parent"]["span_id"] == reply["context"]["span_id"]
        assert span["context"]["trace_id"] == reply["context"]["trace_id"]

    # What the caller said, what Claude asked the tools, and what came back: none of it.
    sent = everything_sent(capfire)
    for private in ("dolo", "fever", "paracetamol", "my son", "pocket book", "danger signs"):
        assert private not in sent, private


async def test_a_request_to_claude_that_fails_is_marked_and_its_span_still_ends(
    capfire: CaptureLogfire,
) -> None:
    broken = ScriptedClaude(Round(error=ValueError("Unable to parse tool input")))
    with pytest.raises(ReplyError):
        await spoken(claude(broken, None).stream_reply(caller("hello")))
    (failed,) = finished(capfire, "llm round")
    assert failed.status.status_code is StatusCode.ERROR and failed.end_time is not None
    assert failed.status.description == "ReplyError"

    # A caller who hangs up closes the reply part-way: the span ends there, and says so.
    stream = claude(ScriptedClaude(Round(text("One. "), text("Two."))), None).stream_reply(
        caller("hello")
    )
    assert await anext(stream) == "One. "
    await stream.aclose()
    cut_off = finished(capfire, "llm round")[-1]
    assert cut_off.status.description == "GeneratorExit"


@pytest.fixture
async def catalogue() -> AsyncIterator[MedicineStore]:
    client = AsyncQdrantClient(":memory:")
    store = MedicineStore(client, "clinexa_medicines_tracing_test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        await store.ensure_collection()
    await store.upsert(CATALOG)
    yield store
    await client.close()


async def test_the_lookups_themselves_say_how_they_went_and_not_what_was_asked(
    capfire: CaptureLogfire, catalogue: MedicineStore
) -> None:
    lookup = MedicineLookup(catalogue)
    assert await lookup.find("dolo 650") is not None
    assert await lookup.find("glycomate 500") is not None
    assert await lookup.find("   ") is None
    assert [span["attributes"]["status"] for span in named(capfire, "medicines lookup")] == [
        "exact",
        "close",
        "unavailable",
    ]

    search = await ready(GuidelineSearch(Index().load, top_k=2))
    assert await search.search("child with cough and fast breathing", age=3) is not None
    await search.aclose()
    (searched,) = named(capfire, "guidelines search")
    attributes = searched["attributes"]
    assert (attributes["searched"], attributes["passages"]) == (True, 2)
    assert list(attributes["population"]) == ["child", "all"]
    assert attributes["filter_relaxed"] is False
    assert {"dense_ms", "bm25_ms", "rrf_ms", "rerank_ms", "total_ms"} <= attributes.keys()

    # Before the models are in, a search is traced as one that could not be made.
    assert await GuidelineSearch(Index().load).search("cough") is None
    assert named(capfire, "guidelines search")[-1]["attributes"]["searched"] is False

    sent = everything_sent(capfire)
    for private in ("dolo", "glycom", "cough", "breathing", "pneumonia"):
        assert private not in sent, private


async def test_a_call_traces_each_reply_it_speaks(
    capfire: CaptureLogfire, settings: Settings
) -> None:
    async def send(raw: str) -> None:
        return None

    session = CallSession(
        call_sid="CA77",
        stream_sid="MZ1",
        stt=LiveFakeSTTProvider([final("I have a headache.", speech_final=True)]),
        settings=settings,
        responder=FakeReplyGenerator(["I'm sorry to hear that. ", "How long has it lasted?"]),
        tts=FakeTTSProvider(),
        send=send,
    )
    session.start()
    session.feed_audio(FRAME)
    async with asyncio.timeout(2):
        while session.replies_spoken == 0:  # noqa: ASYNC110 - polling plain session state
            await asyncio.sleep(0)
    await session.close()

    (reply,) = named(capfire, "reply")
    assert reply["attributes"] | {"logfire.span_type": "", "logfire.msg": ""} == {
        "call_sid": "CA77",
        "sentences": 2,
        "lookups": 0,
        "fallback": False,
        "audio_started": True,
        "logfire.span_type": "",
        "logfire.msg": "",
    }
    sent = everything_sent(capfire)
    assert "headache" not in sent and "sorry to hear" not in sent


# --- the server's own requests ---------------------------------------------------------------


def test_requests_are_traced_without_their_arguments_and_secret_addresses_not_at_all(
    capfire: CaptureLogfire, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As after configure_tracing() with a token. Here the spans go to the test's exporter.
    monkeypatch.setattr(tracing, "_enabled", True)
    open_webhook = settings.model_copy(
        update={
            "twilio_validate_signatures": False,
            "exotel_callback_key": SecretStr("callback-sekret"),
        }
    )
    with TestClient(create_app(open_webhook, stt_provider=None)) as http:
        assert http.get("/health").json()["tracing"] == {"provider": "logfire", "configured": True}
        form = {"CallSid": "CA123", "From": "+15551234567", "To": "+15557654321"}
        assert http.post("/twilio/voice", data=form).status_code == 200
        # The callback key is in the address, so this endpoint is left out altogether.
        callback = http.post("/exotel/call-status?key=callback-sekret", data={"CallSid": "x"})
        assert callback.status_code != 401

    names = {span["name"] for span in spans(capfire)}
    assert any(name.startswith("GET /health") for name in names), names
    assert any(name.startswith("POST /twilio/voice") for name in names), names
    assert not any("exotel" in name for name in names), names

    sent = everything_sent(capfire)
    # The caller's number and the callback key never leave, nor does any request header.
    for private in ("15551234567", "15557654321", "callback-sekret", "user-agent"):
        assert private not in sent, private
