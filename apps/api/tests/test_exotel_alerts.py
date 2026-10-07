"""Exotel alerts: the REST client, the worker, the callbacks and the reminder call."""

from __future__ import annotations

import base64
import json
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.agents.reminder import ReminderReplyGenerator, opening_line
from app.core.config import Settings
from app.graph.state import ConversationMessage
from app.main import create_app
from app.notify.exotel import ExotelClient, ExotelConfig, ExotelError, to_e164_india
from app.notify.scheduler import AlertScheduler
from app.notify.store import (
    CallContext,
    DueReminder,
    LowStock,
    Medicine,
    SupabaseAlertStore,
    in_quiet_hours,
)
from app.notify.worker import run_dose_reminders, run_low_stock
from app.voice.exotel_protocol import StartMessage, parse_inbound
from app.voice.exotel_stream import ExotelTransport
from app.voice.providers.base import TranscriptEvent
from tests.conftest import FakeSTTProvider, FakeTTSProvider

IST = ZoneInfo("Asia/Kolkata")
METFORMIN = Medicine(name="Metformin", strength="500 mg", dose=1, unit="tablets")
CONFIG = ExotelConfig(
    api_key="key",
    api_token="token",
    account_sid="clinexsa1",
    subdomain="api.in.exotel.com",
    caller_id="08012345678",
    voice_app_id="4242",
    whatsapp_from="+918012345678",
    template_language="en",
    call_time_limit_s=180,
    ring_timeout_s=30,
)


def exotel_with(handler: Any) -> ExotelClient:
    transport = httpx.MockTransport(handler)
    return ExotelClient(
        CONFIG, client=httpx.AsyncClient(base_url=CONFIG.base_url, transport=transport)
    )


class FakeExotel:
    can_message = True
    can_call = True

    def __init__(self, fail: bool = False) -> None:
        self.messages: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.fail = fail

    async def send_whatsapp_template(self, to: str, **kwargs: Any) -> str:
        if self.fail:
            raise ExotelError("Exotel returned 500")
        self.messages.append({"to": to, **kwargs})
        return f"WA{len(self.messages)}"

    async def place_call(self, to: str, **kwargs: Any) -> str:
        self.calls.append({"to": to, **kwargs})
        return f"CALL{len(self.calls)}"


class FakeStore:
    def __init__(
        self,
        reminders: list[DueReminder] | None = None,
        low: list[LowStock] | None = None,
        contexts: dict[str, CallContext] | None = None,
    ) -> None:
        self.reminders = reminders or []
        self.low = low or []
        self.contexts = contexts or {}
        self.claimed: set[str] = set()
        self.marks: list[tuple[str, str, str | None]] = []
        self.window: tuple[date, time, time] | None = None

    async def due_reminders(self, today: date, start: time, end: time) -> list[DueReminder]:
        self.window = (today, start, end)
        return self.reminders

    async def low_stock(self) -> list[LowStock]:
        return self.low

    async def claim(self, *, dedupe_key: str, **_: Any) -> str | None:
        if dedupe_key in self.claimed:
            return None
        self.claimed.add(dedupe_key)
        return f"delivery-{len(self.claimed)}"

    async def mark(
        self,
        delivery_id: str,
        status: str,
        *,
        provider_sid: str | None = None,
        detail: str | None = None,
    ) -> None:
        self.marks.append((delivery_id, status, provider_sid))

    async def mark_by_sid(self, provider_sid: str, status: str, detail: str | None = None) -> None:
        self.marks.append((provider_sid, status, detail))

    async def call_context(self, call_sid: str) -> CallContext | None:
        return self.contexts.get(call_sid)


def reminder(**changes: Any) -> DueReminder:
    values: dict[str, Any] = {
        "alert_id": "alert-1",
        "client_id": "client-1",
        "medication_id": "med-1",
        "first_name": "Lakshmi",
        "language": "en",
        "phone": "9845011223",
        "whatsapp_number": "9845011223",
        "medicine": METFORMIN,
        "at": time(8, 0),
        "channels": ("app", "whatsapp", "call"),
    }
    return DueReminder(**{**values, **changes})


# --- Phone numbers ---------------------------------------------------------------


def test_only_real_indian_mobile_numbers_are_dialled() -> None:
    assert to_e164_india("98450 11223") == "+919845011223"
    assert to_e164_india("+91 9845011223") == "+919845011223"
    # The placeholder on test records must never be called or messaged.
    assert to_e164_india("0000000000") is None
    assert to_e164_india("12345") is None
    assert to_e164_india(None) is None


def test_quiet_hours_can_run_over_midnight() -> None:
    assert in_quiet_hours(time(23, 0), time(22, 0), time(6, 0))
    assert in_quiet_hours(time(5, 59), time(22, 0), time(6, 0))
    assert not in_quiet_hours(time(8, 0), time(22, 0), time(6, 0))
    assert not in_quiet_hours(time(8, 0), None, None)


# --- REST client -----------------------------------------------------------------


async def test_whatsapp_template_request_matches_exotels_api() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            202, json={"response": {"whatsapp": {"messages": [{"data": {"sid": "WA123"}}]}}}
        )

    sid = await exotel_with(handler).send_whatsapp_template(
        "+919845011223",
        template="clinexsa_dose_reminder",
        body_params=["Lakshmi", "Metformin 500 mg", "8:00 AM"],
        custom_data="delivery-1",
        status_callback="https://example.com/exotel/whatsapp-status?key=k",
    )

    assert sid == "WA123"
    assert seen["path"] == "/v2/accounts/clinexsa1/messages"
    message = seen["body"]["whatsapp"]["messages"][0]
    assert message["from"] == "+918012345678" and message["to"] == "+919845011223"
    template = message["content"]["template"]
    assert template["name"] == "clinexsa_dose_reminder"
    assert [p["text"] for p in template["components"][0]["parameters"]] == [
        "Lakshmi",
        "Metformin 500 mg",
        "8:00 AM",
    ]
    assert seen["body"]["custom_data"] == "delivery-1"


async def test_call_request_connects_the_patient_to_the_voice_flow() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["form"] = dict(httpx.QueryParams(request.content.decode()))
        return httpx.Response(200, json={"Call": {"Sid": "CALL9", "Status": "in-progress"}})

    sid = await exotel_with(handler).place_call("+919845011223", custom_field="delivery-2")

    assert sid == "CALL9"
    assert seen["path"] == "/v1/Accounts/clinexsa1/Calls/connect.json"
    assert seen["form"]["From"] == "+919845011223"
    assert seen["form"]["CallerId"] == "08012345678"
    assert seen["form"]["Url"] == "http://my.exotel.com/clinexsa1/exoml/start_voice/4242"
    assert seen["form"]["CallType"] == "trans"
    assert seen["form"]["CustomField"] == "delivery-2"


async def test_an_exotel_error_is_raised_without_the_response_body() -> None:
    client = exotel_with(lambda request: httpx.Response(403, text="number +919845011223 blocked"))
    with pytest.raises(ExotelError) as raised:
        await client.place_call("+919845011223", custom_field="x")
    assert "9845011223" not in str(raised.value)


async def test_an_xml_reply_to_a_call_request_is_understood() -> None:
    xml = "<TwilioResponse><Call><Sid>CALLXML</Sid><Status>in-progress</Status></Call></TwilioResponse>"
    client = exotel_with(lambda request: httpx.Response(200, text=xml))
    assert await client.place_call("+919845011223", custom_field="x") == "CALLXML"


async def test_listing_exophones_reads_only() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"], seen["path"] = request.method, request.url.path
        return httpx.Response(
            200, json={"incoming_phone_numbers": [{"phone_number": "08012345678"}]}
        )

    assert await exotel_with(handler).list_exophones() == ["08012345678"]
    assert seen == {"method": "GET", "path": "/v2_beta/Accounts/clinexsa1/IncomingPhoneNumbers"}


# --- Reading the database ----------------------------------------------------------


async def test_the_two_switches_in_the_dashboard_decide_how_a_reminder_goes_out() -> None:
    def alarm(alert_id: str, client_id: str) -> dict[str, Any]:
        return {
            "id": alert_id,
            "client_id": client_id,
            "medication_id": "m1",
            "time": "08:00:00",
            "medications": {
                "name": "Metformin",
                "strength": "500 mg",
                "dose": 1,
                "unit": "tablets",
            },
            "clients": {"full_name": "Lakshmi Iyer", "language": "en"},
        }

    tables: dict[str, list[dict[str, Any]]] = {
        "medication_alerts": [alarm("a1", "c1"), alarm("a2", "c2"), alarm("a3", "c3")],
        "alert_settings": [
            # WhatsApp on. A call number is on file, but calls are switched off.
            {
                "client_id": "c1",
                "whatsapp_enabled": True,
                "whatsapp_number": "9845011223",
                "calls_enabled": False,
                "call_number": "9845011223",
            },
            {"client_id": "c2", "calls_enabled": True, "call_number": "9845099887"},
            # c3 has switched nothing on.
        ],
        # All three are inside their trial or a paid month.
        "clients_with_access": ["c1", "c2", "c3"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=tables[request.url.path.rsplit("/", 1)[-1]])

    store = SupabaseAlertStore(
        "https://example.supabase.co",
        "server-key",
        client=httpx.AsyncClient(
            base_url="https://example.supabase.co/rest/v1", transport=httpx.MockTransport(handler)
        ),
    )
    first, second, third = await store.due_reminders(date(2026, 10, 6), time(7, 57), time(8, 2))

    assert first.channels == ("whatsapp",) and first.phone is None
    assert second.channels == ("call",) and second.whatsapp_number is None
    assert third.channels == ()
    assert first.first_name == "Lakshmi" and first.medicine.label == "Metformin 500 mg"


async def test_only_patients_on_a_trial_or_a_paid_month_are_reminded() -> None:
    asked: list[Any] = []
    tables: dict[str, list[Any]] = {
        "medication_alerts": [
            {"id": "a1", "client_id": "paid", "medication_id": "m1", "time": "08:00:00"},
            {"id": "a2", "client_id": "lapsed", "medication_id": "m2", "time": "08:00:00"},
        ],
        "medications": [
            {"id": "m1", "client_id": "paid", "name": "Metformin", "quantity": 2},
            {"id": "m2", "client_id": "lapsed", "name": "Calcium", "quantity": 2},
        ],
        "alert_settings": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        if name == "clients_with_access":
            # The database decides. Here it says the trial of "lapsed" has run out.
            assert request.method == "POST" and request.headers["content-profile"] == "public"
            asked.append(json.loads(request.content)["p_client_ids"])
            return httpx.Response(200, json=["paid"])
        return httpx.Response(200, json=tables[name])

    store = SupabaseAlertStore(
        "https://example.supabase.co",
        "server-key",
        client=httpx.AsyncClient(
            base_url="https://example.supabase.co/rest/v1", transport=httpx.MockTransport(handler)
        ),
    )
    due = await store.due_reminders(date(2026, 10, 6), time(7, 57), time(8, 2))
    low = await store.low_stock()

    assert [item.client_id for item in due] == ["paid"]
    assert [item.client_id for item in low] == ["paid"]
    assert asked == [["lapsed", "paid"], ["lapsed", "paid"]]


async def test_nothing_is_sent_if_the_plan_cannot_be_checked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/clients_with_access"):
            return httpx.Response(503)
        return httpx.Response(
            200, json=[{"id": "a1", "client_id": "c1", "medication_id": "m1", "time": "08:00:00"}]
        )

    store = SupabaseAlertStore(
        "https://example.supabase.co",
        "server-key",
        client=httpx.AsyncClient(
            base_url="https://example.supabase.co/rest/v1", transport=httpx.MockTransport(handler)
        ),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await store.due_reminders(date(2026, 10, 6), time(7, 57), time(8, 2))


def store_reading(handler: Any) -> SupabaseAlertStore:
    return SupabaseAlertStore(
        "https://example.supabase.co",
        "server-key",
        client=httpx.AsyncClient(
            base_url="https://example.supabase.co/rest/v1", transport=httpx.MockTransport(handler)
        ),
    )


def alarm_row(alert_id: str, at: str, name: str, strength: str | None = None) -> dict[str, Any]:
    return {
        "id": alert_id,
        "client_id": "c1",
        "medication_id": f"m-{alert_id}",
        "time": at,
        "medications": {"name": name, "strength": strength},
        "clients": {"full_name": "Lakshmi Iyer", "language": "en"},
    }


async def test_medicines_due_at_the_same_time_go_out_as_one_alert(settings: Settings) -> None:
    alarms = [
        alarm_row("a1", "08:00:00", "Metformin", "500 mg"),
        alarm_row("a2", "08:00:00", "Glimepiride", "1 mg"),
    ]
    tables: dict[str, list[Any]] = {
        "medication_alerts": alarms,
        "alert_settings": [
            {
                "client_id": "c1",
                "whatsapp_enabled": True,
                "whatsapp_number": "9845011223",
                "calls_enabled": True,
                "call_number": "9845011223",
            }
        ],
        "clients_with_access": ["c1"],
    }
    store = store_reading(
        lambda request: httpx.Response(200, json=tables[request.url.path.rsplit("/", 1)[-1]])
    )

    due = await store.due_reminders(date(2026, 10, 6), time(7, 57), time(8, 2))

    assert len(due) == 1
    assert due[0].medicine.label == "Glimepiride 1 mg and Metformin 500 mg"

    # And the worker sends that one alert once on each channel, not once a medicine.
    fake, exotel = FakeStore(due), FakeExotel()
    now = datetime(2026, 10, 6, 8, 1, tzinfo=IST)
    await run_dose_reminders(fake, exotel, settings, now)  # type: ignore[arg-type]
    await run_dose_reminders(fake, exotel, settings, now)  # type: ignore[arg-type]
    assert len(exotel.messages) == 1 and len(exotel.calls) == 1
    assert exotel.messages[0]["body_params"][1] == "Glimepiride 1 mg and Metformin 500 mg"


async def test_a_person_gets_at_most_three_medicine_alerts_a_day() -> None:
    # Four alert times are switched on. Only the three earliest count.
    every = [
        alarm_row("a1", "06:00:00", "Thyroxine"),
        alarm_row("a2", "08:00:00", "Metformin"),
        alarm_row("a3", "14:00:00", "Calcium"),
        alarm_row("a4", "21:00:00", "Atorvastatin"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        if name == "clients_with_access":
            return httpx.Response(200, json=["c1"])
        if name != "medication_alerts":
            return httpx.Response(200, json=[])
        window = request.url.params.get("and")
        if window is None:
            return httpx.Response(200, json=every)  # the question "which times does c1 have?"
        start, end = (part.split(".", 2)[2] for part in window.strip("()").split(","))
        return httpx.Response(200, json=[row for row in every if start <= row["time"] <= end])

    store = store_reading(handler)
    day = date(2026, 10, 6)
    sent = [
        [item.medicine.name for item in await store.due_reminders(day, start, end)]
        for start, end in [
            (time(5, 57), time(6, 2)),
            (time(7, 57), time(8, 2)),
            (time(13, 57), time(14, 2)),
            (time(20, 57), time(21, 2)),
        ]
    ]

    assert sent == [["Thyroxine"], ["Metformin"], ["Calcium"], []]


async def test_running_low_is_one_message_a_person_however_many_medicines() -> None:
    tables: dict[str, list[Any]] = {
        "medications": [
            {
                "id": "m1",
                "client_id": "c1",
                "name": "Metformin",
                "strength": "500 mg",
                "quantity": 6,
            },
            {"id": "m2", "client_id": "c1", "name": "Calcium", "quantity": 2},
            {"id": "m3", "client_id": "c1", "name": "Thyroxine", "quantity": 90},
        ],
        "alert_settings": [
            {"client_id": "c1", "whatsapp_enabled": True, "whatsapp_number": "9845011223"}
        ],
        "clients_with_access": ["c1"],
    }
    store = store_reading(
        lambda request: httpx.Response(200, json=tables[request.url.path.rsplit("/", 1)[-1]])
    )

    low = await store.low_stock()

    assert len(low) == 1
    assert low[0].medicine.label == "Calcium and Metformin 500 mg"
    assert low[0].days_left == 2


async def test_one_call_names_every_medicine_taken_at_that_time() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/alert_deliveries"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "d1",
                        "client_id": "c1",
                        "medication_alerts": {"time": "08:00:00"},
                        "clients": {"full_name": "Lakshmi Iyer", "language": "en"},
                        "medications": {"name": "Metformin", "strength": "500 mg"},
                    }
                ],
            )
        return httpx.Response(
            200,
            json=[
                {"medications": {"name": "Metformin", "strength": "500 mg"}},
                {"medications": {"name": "Glimepiride", "strength": "1 mg"}},
            ],
        )

    context = await store_reading(handler).call_context("call-sid")

    assert context is not None and context.count == 2
    line = opening_line(context)
    assert "It is time for Glimepiride 1 mg and Metformin 500 mg." in line
    assert line.endswith("Have you taken them?")


# --- Worker ----------------------------------------------------------------------


async def test_a_due_reminder_goes_out_once_on_each_chosen_channel(settings: Settings) -> None:
    store, exotel = FakeStore([reminder()]), FakeExotel()
    now = datetime(2026, 10, 6, 8, 2, tzinfo=IST)

    first = await run_dose_reminders(store, exotel, settings, now)  # type: ignore[arg-type]
    second = await run_dose_reminders(store, exotel, settings, now)  # type: ignore[arg-type]

    assert (first.whatsapp, first.calls) == (1, 1)
    assert (second.whatsapp, second.calls) == (0, 0)  # the second run sends nothing again
    assert store.window == (date(2026, 10, 6), time(7, 57), time(8, 2))
    assert exotel.messages[0]["body_params"] == ["Lakshmi", "Metformin 500 mg", "8:00 AM"]
    assert exotel.calls[0]["to"] == "+919845011223"
    assert [status for _, status, _ in store.marks] == ["sent", "sent"]


async def test_unverified_whatsapp_quiet_hours_and_placeholder_numbers_are_skipped(
    settings: Settings,
) -> None:
    store = FakeStore(
        [
            reminder(alert_id="a", whatsapp_number=None, channels=("whatsapp",)),
            reminder(alert_id="b", quiet=True, channels=("call",)),
            reminder(alert_id="c", phone="0000000000", channels=("call",)),
        ]
    )
    exotel = FakeExotel()

    result = await run_dose_reminders(
        store,
        exotel,
        settings,
        datetime(2026, 10, 6, 8, 2, tzinfo=IST),  # type: ignore[arg-type]
    )

    assert result.skipped == 3
    assert exotel.messages == [] and exotel.calls == []


async def test_a_failed_send_is_recorded_as_failed(settings: Settings) -> None:
    store, exotel = FakeStore([reminder(channels=("whatsapp",))]), FakeExotel(fail=True)

    result = await run_dose_reminders(
        store,
        exotel,
        settings,
        datetime(2026, 10, 6, 8, 2, tzinfo=IST),  # type: ignore[arg-type]
    )

    assert result.failed == 1
    assert store.marks[0][1] == "failed"


async def test_low_stock_sends_one_whatsapp_message_a_day(settings: Settings) -> None:
    item = LowStock(
        client_id="client-1",
        medication_id="med-1",
        first_name="Lakshmi",
        whatsapp_number="9845011223",
        medicine=METFORMIN,
        days_left=3,
    )
    store, exotel = FakeStore(low=[item]), FakeExotel()
    now = datetime(2026, 10, 6, 10, 0, tzinfo=IST)

    await run_low_stock(store, exotel, settings, now)  # type: ignore[arg-type]
    await run_low_stock(store, exotel, settings, now)  # type: ignore[arg-type]

    assert len(exotel.messages) == 1
    assert exotel.messages[0]["body_params"] == ["Lakshmi", "Metformin 500 mg", "3"]


async def test_the_scheduler_sends_reminders_each_tick_and_low_stock_once_a_day(
    settings: Settings,
) -> None:
    item = LowStock("client-1", "med-1", "Lakshmi", "9845011223", METFORMIN, 3)
    store, exotel = FakeStore([reminder(channels=("whatsapp",))], low=[item]), FakeExotel()
    scheduler = AlertScheduler(store, exotel, settings)  # type: ignore[arg-type]

    await scheduler.tick(datetime(2026, 10, 6, 8, 2, tzinfo=IST))  # before 10:00: reminders only
    assert len(exotel.messages) == 1
    await scheduler.tick(datetime(2026, 10, 6, 10, 1, tzinfo=IST))
    await scheduler.tick(datetime(2026, 10, 6, 10, 2, tzinfo=IST))

    templates = [message["template"] for message in exotel.messages]
    assert templates.count("clinexsa_low_stock") == 1


# --- Endpoints -------------------------------------------------------------------


def alerts_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "alerts_enabled": True,
            "alerts_jobs_token": "jobs-token",
            "exotel_callback_key": "callback-key",
            "exotel_stream_username": "exotel",
            "exotel_stream_password": "stream-pass",
        }
    )


def make_client(settings: Settings, store: FakeStore, **providers: Any) -> TestClient:
    app = create_app(
        Settings.model_validate(alerts_settings(settings).model_dump()),
        stt_provider=providers.get("stt", FakeSTTProvider([])),
        tts_provider=providers.get("tts"),
        reply_generator=None,
        exotel=FakeExotel(),  # type: ignore[arg-type]
        alert_store=store,  # type: ignore[arg-type]
    )
    return TestClient(app)


def test_the_jobs_endpoint_needs_its_token(settings: Settings) -> None:
    with make_client(settings, FakeStore([reminder(channels=("whatsapp",))])) as client:
        assert client.post("/jobs/alerts/run").status_code == 401
        wrong = client.post("/jobs/alerts/run", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        ok = client.post("/jobs/alerts/run", headers={"Authorization": "Bearer jobs-token"})
        assert ok.status_code == 200 and ok.json()["due"] == 1


def test_status_callbacks_need_the_key_and_update_the_delivery(settings: Settings) -> None:
    store = FakeStore()
    with make_client(settings, store) as client:
        assert client.post("/exotel/call-status", json={"CallSid": "C1"}).status_code == 401
        client.post(
            "/exotel/call-status?key=callback-key", json={"CallSid": "C1", "Status": "completed"}
        )
        client.post(
            "/exotel/whatsapp-status?key=callback-key",
            json={
                "whatsapp": {
                    "messages": [{"callback_type": "dlr", "sid": "W1", "exo_status_code": 30002}]
                }
            },
        )
    assert store.marks == [("C1", "answered", None), ("W1", "delivered", None)]


# --- The reminder call -----------------------------------------------------------


def exotel_start(call_sid: str) -> str:
    return json.dumps(
        {
            "event": "start",
            "sequence_number": 1,
            "stream_sid": "stream-1",
            "start": {
                "stream_sid": "stream-1",
                "call_sid": call_sid,
                "account_sid": "clinexsa1",
                "from": "09845011223",
                "to": "08012345678",
                "media_format": {"encoding": "slin", "sample_rate": "8000", "bit_rate": "128kbps"},
            },
        }
    )


def test_exotel_start_message_is_parsed_with_string_numbers() -> None:
    message = parse_inbound(exotel_start("CALL1"))
    assert isinstance(message, StartMessage)
    assert message.start.call_sid == "CALL1"
    assert message.start.media_format.sample_rate == 8000
    assert message.start.caller == "09845011223"


async def test_audio_goes_to_exotel_in_even_chunks() -> None:
    sent: list[dict[str, Any]] = []

    async def send(frame: str) -> None:
        sent.append(json.loads(frame))

    transport = ExotelTransport("stream-1", send)
    await transport.send_audio(b"\x01" * 5000)
    await transport.end_of_reply("reply-1")

    sizes = [len(base64.b64decode(f["media"]["payload"])) for f in sent if f["event"] == "media"]
    assert sizes == [3200, 1920]  # 1800 bytes left, padded up to a multiple of 320
    assert sent[-1] == {
        "event": "mark",
        "sequence_number": 3,
        "stream_sid": "stream-1",
        "mark": {"name": "reply-1"},
    }


async def test_the_opening_line_is_scripted_and_names_the_medicine() -> None:
    context = CallContext("d1", "Lakshmi", "en", METFORMIN, time(8, 0))
    generator = ReminderReplyGenerator(context, inner=None)

    opening = [text async for text in generator.stream_reply([])]
    after = [
        text
        async for text in generator.stream_reply(
            [ConversationMessage(role="patient", content="Yes, I took it.")]
        )
    ]

    assert opening == [opening_line(context)]
    assert "Metformin 500 mg" in opening[0] and "8:00 AM" in opening[0]
    assert after == ["Thank you. Please take care. Goodbye."]


def basic(username: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_the_call_stream_refuses_wrong_credentials_and_unknown_calls(settings: Settings) -> None:
    with make_client(settings, FakeStore()) as client:
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/exotel/stream", headers=basic("exotel", "wrong")),
        ):
            pass
        with client.websocket_connect(
            "/exotel/stream", headers=basic("exotel", "stream-pass")
        ) as ws:
            ws.send_text(exotel_start("NOT-OURS"))
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()


def test_roopiee_opens_a_reminder_call_with_the_medicine(settings: Settings) -> None:
    context = CallContext("d1", "Lakshmi", "en", METFORMIN, time(8, 0))
    tts = FakeTTSProvider()
    stt = FakeSTTProvider([TranscriptEvent(type="metadata")])  # type: ignore[arg-type]
    with (
        make_client(settings, FakeStore(contexts={"CALL1": context}), stt=stt, tts=tts) as client,
        client.websocket_connect("/exotel/stream", headers=basic("exotel", "stream-pass")) as ws,
    ):
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(exotel_start("CALL1"))
        frames = [json.loads(ws.receive_text()) for _ in range(2)]
        ws.send_text(json.dumps({"event": "stop", "stream_sid": "stream-1", "stop": {}}))

    # The opening is spoken sentence by sentence.
    assert " ".join(tts.spoken) == opening_line(context)
    assert [frame["event"] for frame in frames] == ["media", "mark"]
    assert frames[0]["stream_sid"] == "stream-1"
