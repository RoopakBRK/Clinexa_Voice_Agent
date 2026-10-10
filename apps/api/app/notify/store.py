"""What the alerts worker reads from and writes to the Clinexsa database.

The worker talks to Supabase's REST API with the service-role key, which bypasses
row-level security. That key stays on this server. Everything the worker may touch
is listed here, so it is easy to see how little it reads: alarms, the medicine each
one is for, and the number to reach.

Reminders are part of the plan. Before anything is sent, the database is asked which
of the patients are inside their free trial or a paid month (clients_with_access).
Nobody else is reminded.

A person gets at most three medicine alerts a day (MAX_ALERTS_A_DAY). Medicines taken
at the same time go out as one alert that names them all, and only a person's three
earliest alert times count. Running-low news is one message a day, not one a medicine.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time
from typing import Any, Protocol

import httpx

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

SCHEMA = "clinexsa"

# The most medicine alerts one person is sent in a day, on each way they have switched
# on. The website holds the same number (lib/dashboard/alertLimit.ts).
MAX_ALERTS_A_DAY = 3


@dataclass(frozen=True)
class Medicine:
    name: str
    strength: str | None = None
    dose: float | None = None
    unit: str | None = None

    @property
    def label(self) -> str:
        """As it is said and written: "Metformin 500 mg"."""
        return f"{self.name} {self.strength}".strip() if self.strength else self.name


@dataclass(frozen=True)
class DueReminder:
    alert_id: str
    client_id: str
    medication_id: str
    first_name: str
    language: str
    # Each number is set only while the person has that switch on in their dashboard
    # (clinexsa.alert_settings). Off means None, and nothing is sent that way.
    phone: str | None
    whatsapp_number: str | None
    medicine: Medicine
    at: time
    channels: tuple[str, ...]
    # The alarm falls inside the person's quiet hours: no phone call.
    quiet: bool = False


@dataclass(frozen=True)
class LowStock:
    client_id: str
    medication_id: str
    first_name: str
    whatsapp_number: str | None
    medicine: Medicine
    days_left: int


@dataclass(frozen=True)
class CallContext:
    """What Clinexsa needs to make one reminder call."""

    delivery_id: str
    first_name: str
    language: str
    medicine: Medicine
    at: time | None
    # How many medicines this one call is for. They are all named in `medicine`.
    count: int = 1


class AlertStore(Protocol):
    async def due_reminders(self, today: date, start: time, end: time) -> list[DueReminder]: ...

    async def low_stock(self) -> list[LowStock]: ...

    async def claim(
        self,
        *,
        dedupe_key: str,
        client_id: str,
        kind: str,
        channel: str,
        alert_id: str | None = None,
        medication_id: str | None = None,
    ) -> str | None:
        """Record that this is about to be sent. None means it was already claimed."""
        ...

    async def mark(
        self,
        delivery_id: str,
        status: str,
        *,
        provider_sid: str | None = None,
        detail: str | None = None,
    ) -> None: ...

    async def mark_by_sid(
        self, provider_sid: str, status: str, detail: str | None = None
    ) -> None: ...

    async def call_context(self, call_sid: str) -> CallContext | None: ...


def in_quiet_hours(at: time, start: time | None, end: time | None) -> bool:
    if start is None or end is None or start == end:
        return False
    if start < end:
        return start <= at < end
    return at >= start or at < end  # the quiet period runs over midnight


def _first_name(full_name: str | None) -> str:
    parts = (full_name or "").split()
    return parts[0] if parts else "there"


def _time(value: Any) -> time | None:
    try:
        return time.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def _medicine(row: dict[str, Any] | None) -> Medicine:
    row = row or {}
    dose = row.get("dose")
    return Medicine(
        name=str(row.get("name") or "your medicine"),
        strength=row.get("strength") or None,
        dose=float(dose) if dose is not None else None,
        unit=row.get("unit") or None,
    )


def _together(labels: list[str]) -> Medicine:
    """Several medicines as one, to be said or written: "Metformin 500 mg and Calcium"."""
    names = sorted(set(labels))
    said = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
    return Medicine(name=said)


def _in(ids: set[str]) -> str:
    return f"in.({','.join(sorted(ids))})"


class SupabaseAlertStore:
    def __init__(self, url: str, service_key: str, *, client: httpx.AsyncClient | None = None):
        self._client = client or httpx.AsyncClient(
            base_url=f"{url}/rest/v1",
            headers={
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
                "Accept-Profile": SCHEMA,
                "Content-Profile": SCHEMA,
            },
            timeout=15.0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        response = await self._client.get(f"/{table}", params=params)
        response.raise_for_status()
        rows = response.json()
        return rows if isinstance(rows, list) else []

    async def _with_access(self, ids: set[str]) -> set[str]:
        """The patients among these who may be reminded: on their free trial, or paid up.

        The rule lives in the database, the same one the website uses. If the question
        cannot be asked, this raises and nothing is sent.
        """
        if not ids:
            return set()
        response = await self._client.post(
            "/rpc/clients_with_access",
            headers={"Accept-Profile": "public", "Content-Profile": "public"},
            json={"p_client_ids": sorted(ids)},
        )
        response.raise_for_status()
        rows = response.json()
        return {str(row) for row in rows} if isinstance(rows, list) else set()

    async def _alert_times(self, ids: set[str]) -> dict[str, set[time]]:
        """For each of these patients, the alert times that count: their three earliest."""
        if not ids:
            return {}
        rows = await self._get(
            "medication_alerts",
            {"select": "client_id,time", "enabled": "eq.true", "client_id": _in(ids)},
        )
        found: dict[str, set[time]] = {}
        for row in rows:
            at = _time(row.get("time"))
            if at is not None:
                found.setdefault(str(row["client_id"]), set()).add(at)
        return {client: set(sorted(times)[:MAX_ALERTS_A_DAY]) for client, times in found.items()}

    async def _by_client(self, table: str, select: str, ids: set[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        rows = await self._get(table, {"select": select, "client_id": _in(ids)})
        return {str(row["client_id"]): row for row in rows}

    async def due_reminders(self, today: date, start: time, end: time) -> list[DueReminder]:
        rows = await self._get(
            "medication_alerts",
            {
                "select": (
                    "id,client_id,medication_id,time,"
                    "medications(name,strength,dose,unit),"
                    "clients(full_name,language)"
                ),
                "enabled": "eq.true",
                "and": f"(time.gte.{start.isoformat()},time.lte.{end.isoformat()})",
            },
        )
        ids = await self._with_access({str(row["client_id"]) for row in rows})
        rows = [row for row in rows if str(row["client_id"]) in ids]
        settings = await self._by_client(
            "alert_settings",
            "client_id,reminders,quiet_start,quiet_end,paused_until,"
            "whatsapp_enabled,whatsapp_number,calls_enabled,call_number",
            ids,
        )
        counted = await self._alert_times(ids)

        # One alert for each person and time, however many medicines are due then.
        due: dict[tuple[str, time], DueReminder] = {}
        labels: dict[tuple[str, time], list[str]] = {}
        for row in rows:
            client_id = str(row["client_id"])
            at = _time(row.get("time"))
            if at is None or at not in counted.get(client_id, set()):
                continue  # no time, or past the person's three alerts a day
            setting = settings.get(client_id, {})
            if setting.get("reminders") is False:
                continue  # the person has switched reminders off
            paused = setting.get("paused_until")
            if paused and date.fromisoformat(str(paused)) >= today:
                continue
            client = row.get("clients") or {}
            # The two switches in the dashboard decide how a reminder goes out.
            whatsapp = setting.get("whatsapp_number") if setting.get("whatsapp_enabled") else None
            call = setting.get("call_number") if setting.get("calls_enabled") else None
            labels.setdefault((client_id, at), []).append(_medicine(row.get("medications")).label)
            due.setdefault(
                (client_id, at),
                DueReminder(
                    alert_id=str(row["id"]),
                    client_id=client_id,
                    medication_id=str(row["medication_id"]),
                    first_name=_first_name(client.get("full_name")),
                    language=str(client.get("language") or "en"),
                    phone=call,
                    whatsapp_number=whatsapp,
                    medicine=_medicine(row.get("medications")),
                    at=at,
                    channels=tuple(
                        name for name, number in (("whatsapp", whatsapp), ("call", call)) if number
                    ),
                    quiet=in_quiet_hours(
                        at, _time(setting.get("quiet_start")), _time(setting.get("quiet_end"))
                    ),
                ),
            )
        return [
            replace(reminder, medicine=_together(labels[key])) if len(labels[key]) > 1 else reminder
            for key, reminder in due.items()
        ]

    async def low_stock(self) -> list[LowStock]:
        rows = await self._get(
            "medications",
            {
                "select": (
                    "id,client_id,name,strength,dose,unit,quantity,times_per_day,clients(full_name)"
                ),
                "quantity": "not.is.null",
            },
        )
        ids = await self._with_access({str(row["client_id"]) for row in rows})
        rows = [row for row in rows if str(row["client_id"]) in ids]
        settings = await self._by_client(
            "alert_settings",
            "client_id,low_stock_alerts,low_stock_days,paused_until,whatsapp_enabled,whatsapp_number",
            ids,
        )

        low: list[LowStock] = []
        for row in rows:
            client_id = str(row["client_id"])
            if settings.get(client_id, {}).get("low_stock_alerts") is False:
                continue
            per_day = float(row.get("dose") or 1) * int(row.get("times_per_day") or 1)
            if per_day <= 0:
                continue
            days_left = int(float(row["quantity"]) // per_day)
            if days_left > int(settings.get(client_id, {}).get("low_stock_days") or 7):
                continue
            client = row.get("clients") or {}
            setting = settings.get(client_id, {})
            low.append(
                LowStock(
                    client_id=client_id,
                    medication_id=str(row["id"]),
                    first_name=_first_name(client.get("full_name")),
                    whatsapp_number=(
                        setting.get("whatsapp_number") if setting.get("whatsapp_enabled") else None
                    ),
                    medicine=_medicine(row),
                    days_left=days_left,
                )
            )
        # One message a person a day, naming every medicine that is running out, with
        # the fewest days any of them has left.
        by_client: dict[str, list[LowStock]] = {}
        for item in low:
            by_client.setdefault(item.client_id, []).append(item)
        return [
            items[0]
            if len(items) == 1
            else replace(
                items[0],
                medicine=_together([item.medicine.label for item in items]),
                days_left=min(item.days_left for item in items),
            )
            for items in by_client.values()
        ]

    async def claim(
        self,
        *,
        dedupe_key: str,
        client_id: str,
        kind: str,
        channel: str,
        alert_id: str | None = None,
        medication_id: str | None = None,
    ) -> str | None:
        response = await self._client.post(
            "/alert_deliveries",
            params={"on_conflict": "dedupe_key"},
            headers={"Prefer": "resolution=ignore-duplicates,return=representation"},
            json={
                "dedupe_key": dedupe_key,
                "client_id": client_id,
                "kind": kind,
                "channel": channel,
                "alert_id": alert_id,
                "medication_id": medication_id,
            },
        )
        response.raise_for_status()
        rows = response.json()
        return str(rows[0]["id"]) if isinstance(rows, list) and rows else None

    async def _patch(self, params: dict[str, str], values: dict[str, Any]) -> None:
        values = {key: value for key, value in values.items() if value is not None}
        values["updated_at"] = datetime.now(UTC).isoformat()
        response = await self._client.patch("/alert_deliveries", params=params, json=values)
        response.raise_for_status()

    async def mark(
        self,
        delivery_id: str,
        status: str,
        *,
        provider_sid: str | None = None,
        detail: str | None = None,
    ) -> None:
        await self._patch(
            {"id": f"eq.{delivery_id}"},
            {"status": status, "provider_sid": provider_sid, "detail": detail},
        )

    async def mark_by_sid(self, provider_sid: str, status: str, detail: str | None = None) -> None:
        await self._patch(
            {"provider_sid": f"eq.{provider_sid}"}, {"status": status, "detail": detail}
        )

    async def call_context(self, call_sid: str) -> CallContext | None:
        rows = await self._get(
            "alert_deliveries",
            {
                "select": (
                    "id,client_id,medication_alerts(time),clients(full_name,language),"
                    "medications(name,strength,dose,unit)"
                ),
                "provider_sid": f"eq.{call_sid}",
                "channel": "eq.call",
                "limit": "1",
            },
        )
        if not rows:
            return None
        row = rows[0]
        client = row.get("clients") or {}
        at = _time((row.get("medication_alerts") or {}).get("time"))
        medicine, count = _medicine(row.get("medications")), 1
        if at is not None and row.get("client_id"):
            # The one call is for every medicine the person takes at this time.
            same_time = await self._get(
                "medication_alerts",
                {
                    "select": "medications(name,strength,dose,unit)",
                    "client_id": f"eq.{row['client_id']}",
                    "enabled": "eq.true",
                    "time": f"eq.{at.isoformat()}",
                },
            )
            names = {
                _medicine(r.get("medications")).label for r in same_time if r.get("medications")
            }
            if len(names) > 1:
                medicine, count = _together(list(names)), len(names)
        return CallContext(
            delivery_id=str(row["id"]),
            first_name=_first_name(client.get("full_name")),
            language=str(client.get("language") or "en"),
            medicine=medicine,
            at=at,
            count=count,
        )


def build_alert_store(settings: Settings) -> SupabaseAlertStore | None:
    if settings.supabase_url is None or settings.supabase_service_role_key is None:
        log.warning(
            "alerts.store_not_configured", hint="set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"
        )
        return None
    return SupabaseAlertStore(
        settings.supabase_url, settings.supabase_service_role_key.get_secret_value()
    )
