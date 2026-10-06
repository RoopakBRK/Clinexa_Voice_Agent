"""HTTP endpoints for the alerts worker and Exotel's status callbacks.

* ``POST /jobs/alerts/run`` and ``POST /jobs/low-stock/run`` are called by a
  scheduler, with ``Authorization: Bearer <ALERTS_JOBS_TOKEN>``.
* ``POST /exotel/call-status`` and ``POST /exotel/whatsapp-status`` are called by
  Exotel. Exotel does not sign its callbacks, so the URL we hand it carries a
  secret ``key`` that is checked here.
"""

from __future__ import annotations

import hmac
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.requests import HTTPConnection

from app.core.config import Settings
from app.core.logging import get_logger
from app.notify.exotel import CALL_STATUS, WHATSAPP_STATUS, ExotelClient
from app.notify.store import AlertStore
from app.notify.worker import run_dose_reminders, run_low_stock
from app.voice.deps import get_settings

log = get_logger(__name__)

router = APIRouter(tags=["alerts"])


def get_exotel(conn: HTTPConnection) -> ExotelClient | None:
    return cast("ExotelClient | None", conn.app.state.exotel)


def get_alert_store(conn: HTTPConnection) -> AlertStore | None:
    return cast("AlertStore | None", conn.app.state.alert_store)


def _same(expected: str | None, given: str | None) -> bool:
    return bool(expected) and bool(given) and hmac.compare_digest(str(expected), str(given))


def require_jobs_token(request: Request, settings: Settings = Depends(get_settings)) -> None:
    expected = settings.alerts_jobs_token.get_secret_value() if settings.alerts_jobs_token else None
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not _same(expected, token):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not allowed")


def require_callback_key(request: Request, settings: Settings = Depends(get_settings)) -> None:
    expected = (
        settings.exotel_callback_key.get_secret_value() if settings.exotel_callback_key else None
    )
    if not _same(expected, request.query_params.get("key")):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not allowed")


def _ready(
    settings: Settings, store: AlertStore | None, exotel: ExotelClient | None
) -> tuple[AlertStore, ExotelClient]:
    if not settings.alerts_enabled:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Alerts are switched off")
    if store is None or exotel is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Alerts are not configured")
    return store, exotel


@router.post("/jobs/alerts/run", dependencies=[Depends(require_jobs_token)])
async def run_alerts(
    settings: Settings = Depends(get_settings),
    store: AlertStore | None = Depends(get_alert_store),
    exotel: ExotelClient | None = Depends(get_exotel),
) -> dict[str, int]:
    ready_store, ready_exotel = _ready(settings, store, exotel)
    return (await run_dose_reminders(ready_store, ready_exotel, settings)).as_dict()


@router.post("/jobs/low-stock/run", dependencies=[Depends(require_jobs_token)])
async def run_low_stock_alerts(
    settings: Settings = Depends(get_settings),
    store: AlertStore | None = Depends(get_alert_store),
    exotel: ExotelClient | None = Depends(get_exotel),
) -> dict[str, int]:
    ready_store, ready_exotel = _ready(settings, store, exotel)
    return (await run_low_stock(ready_store, ready_exotel, settings)).as_dict()


async def _payload(request: Request) -> dict[str, Any]:
    """Exotel posts JSON when asked to, and a form otherwise."""
    if "json" in request.headers.get("content-type", ""):
        try:
            body = await request.json()
        except ValueError:
            return {}
        return body if isinstance(body, dict) else {}
    form = await request.form()
    return {key: value for key, value in form.items() if isinstance(value, str)}


@router.post("/exotel/call-status", dependencies=[Depends(require_callback_key)])
async def call_status(
    request: Request, store: AlertStore | None = Depends(get_alert_store)
) -> dict[str, bool]:
    body = await _payload(request)
    sid = body.get("CallSid")
    outcome = CALL_STATUS.get(str(body.get("Status", "")).lower())
    if store is not None and isinstance(sid, str) and sid and outcome:
        await store.mark_by_sid(sid, outcome)
        log.info("exotel.call_status", sid=sid, status=outcome)
    return {"ok": True}


@router.post("/exotel/whatsapp-status", dependencies=[Depends(require_callback_key)])
async def whatsapp_status(
    request: Request, store: AlertStore | None = Depends(get_alert_store)
) -> dict[str, bool]:
    body = await _payload(request)
    whatsapp = body.get("whatsapp")
    messages = whatsapp.get("messages") if isinstance(whatsapp, dict) else None
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict) or message.get("callback_type") != "dlr":
            continue  # "icm" is a message the person sent us; replies are not handled yet
        sid = message.get("sid")
        try:
            code = int(message.get("exo_status_code") or 0)
        except (TypeError, ValueError):
            code = 0
        if store is None or not isinstance(sid, str) or not sid:
            continue
        outcome = WHATSAPP_STATUS.get(code, "failed")
        detail = None if code in WHATSAPP_STATUS else str(message.get("exo_detailed_status"))[:200]
        await store.mark_by_sid(sid, outcome, detail)
        log.info("exotel.whatsapp_status", sid=sid, status=outcome, code=code)
    return {"ok": True}
