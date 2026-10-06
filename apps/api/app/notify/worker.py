"""The alerts worker: finds what is due and sends it through Exotel.

A scheduler calls ``POST /jobs/alerts/run`` every minute or so, and
``POST /jobs/low-stock/run`` once a day. Each thing to send is claimed in the
database first (one row per alarm, day and channel), so a second run, or a second
server, never sends it again.

What it will not do:
* message or phone anyone who has not switched that on in their dashboard, with a number;
* phone anyone during their quiet hours;
* call or message a number that is not a valid Indian mobile number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo

from app.core.config import Settings
from app.core.logging import get_logger
from app.notify.exotel import ExotelClient, ExotelError, to_e164_india
from app.notify.store import AlertStore

log = get_logger(__name__)

CALL = "call"
WHATSAPP = "whatsapp"


@dataclass
class RunResult:
    due: int = 0
    whatsapp: int = 0
    calls: int = 0
    skipped: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def clock(at: time) -> str:
    """08:00 -> "8:00 AM"."""
    return f"{at.hour % 12 or 12}:{at.minute:02d} {'AM' if at.hour < 12 else 'PM'}"


def callback_url(settings: Settings, path: str) -> str | None:
    """Where Exotel reports back. None until a public address and a key are set."""
    if settings.public_base_url is None or settings.exotel_callback_key is None:
        return None
    key = quote(settings.exotel_callback_key.get_secret_value(), safe="")
    return f"{settings.public_base_url}{path}?key={key}"


async def run_dose_reminders(
    store: AlertStore, exotel: ExotelClient, settings: Settings, now: datetime | None = None
) -> RunResult:
    local = (now or datetime.now(ZoneInfo(settings.alerts_timezone))).astimezone(
        ZoneInfo(settings.alerts_timezone)
    )
    today = local.date()
    end = local.time().replace(microsecond=0)
    # Just after midnight the window is cut short rather than reaching into yesterday.
    earliest = local - timedelta(minutes=settings.alerts_window_min)
    start = earliest.time().replace(microsecond=0) if earliest.date() == today else time(0, 0)

    result = RunResult()
    for reminder in await store.due_reminders(today, start, end):
        result.due += 1
        for channel in reminder.channels:
            if channel == WHATSAPP:
                to = to_e164_india(reminder.whatsapp_number)
                usable = exotel.can_message
            elif channel == CALL:
                to = None if reminder.quiet else to_e164_india(reminder.phone)
                usable = exotel.can_call
            else:
                continue  # "app" alarms ring in the dashboard, not here
            if to is None or not usable:
                result.skipped += 1
                continue

            delivery_id = await store.claim(
                dedupe_key=f"dose:{reminder.alert_id}:{today.isoformat()}:{reminder.at.isoformat()}:{channel}",
                client_id=reminder.client_id,
                kind="dose_reminder",
                channel=channel,
                alert_id=reminder.alert_id,
                medication_id=reminder.medication_id,
            )
            if delivery_id is None:
                continue  # already sent by an earlier run

            try:
                if channel == WHATSAPP:
                    sid = await exotel.send_whatsapp_template(
                        to,
                        template=settings.exotel_dose_template,
                        body_params=[
                            reminder.first_name,
                            reminder.medicine.label,
                            clock(reminder.at),
                        ],
                        custom_data=delivery_id,
                        status_callback=callback_url(settings, "/exotel/whatsapp-status"),
                    )
                    result.whatsapp += 1
                else:
                    sid = await exotel.place_call(
                        to,
                        custom_field=delivery_id,
                        status_callback=callback_url(settings, "/exotel/call-status"),
                    )
                    result.calls += 1
                await store.mark(delivery_id, "sent", provider_sid=sid)
            except ExotelError as exc:
                result.failed += 1
                log.error("alerts.send_failed", channel=channel, error=str(exc))
                await store.mark(delivery_id, "failed", detail=str(exc)[:200])
    log.info("alerts.dose_run", **result.as_dict())
    return result


async def run_low_stock(
    store: AlertStore, exotel: ExotelClient, settings: Settings, now: datetime | None = None
) -> RunResult:
    """One WhatsApp message a day for each medicine that is running out. No calls."""
    today = (
        (now or datetime.now(ZoneInfo(settings.alerts_timezone)))
        .astimezone(ZoneInfo(settings.alerts_timezone))
        .date()
    )
    result = RunResult()
    for item in await store.low_stock():
        result.due += 1
        to = to_e164_india(item.whatsapp_number)
        if to is None or not exotel.can_message:
            result.skipped += 1
            continue
        delivery_id = await store.claim(
            dedupe_key=f"low:{item.medication_id}:{today.isoformat()}:{WHATSAPP}",
            client_id=item.client_id,
            kind="low_stock",
            channel=WHATSAPP,
            medication_id=item.medication_id,
        )
        if delivery_id is None:
            continue
        try:
            sid = await exotel.send_whatsapp_template(
                to,
                template=settings.exotel_low_stock_template,
                body_params=[item.first_name, item.medicine.label, str(item.days_left)],
                custom_data=delivery_id,
                status_callback=callback_url(settings, "/exotel/whatsapp-status"),
            )
            result.whatsapp += 1
            await store.mark(delivery_id, "sent", provider_sid=sid)
        except ExotelError as exc:
            result.failed += 1
            log.error("alerts.send_failed", channel=WHATSAPP, error=str(exc))
            await store.mark(delivery_id, "failed", detail=str(exc)[:200])
    log.info("alerts.low_stock_run", **result.as_dict())
    return result
