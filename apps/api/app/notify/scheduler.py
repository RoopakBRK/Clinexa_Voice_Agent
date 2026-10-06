"""Runs the alerts worker on a timer, inside the API process.

With ``ALERTS_SCHEDULER=true`` (the default) nothing outside this server is needed:
dose reminders are checked every minute and low-stock messages go out once a day.
Each send is claimed in the database first, so two servers running this at once
still send each reminder once.

Set ``ALERTS_SCHEDULER=false`` to drive ``POST /jobs/alerts/run`` from an external
cron instead.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from zoneinfo import ZoneInfo

from app.core.config import Settings
from app.core.logging import get_logger
from app.notify.exotel import ExotelClient
from app.notify.store import AlertStore
from app.notify.worker import run_dose_reminders, run_low_stock

log = get_logger(__name__)

TICK_S = 60.0


class AlertScheduler:
    def __init__(self, store: AlertStore, exotel: ExotelClient, settings: Settings) -> None:
        self._store = store
        self._exotel = exotel
        self._settings = settings
        self._low_stock_done: date | None = None
        self._task: asyncio.Task[None] | None = None

    async def tick(self, now: datetime | None = None) -> None:
        """One pass. A failure is logged and the next pass goes ahead."""
        zone = ZoneInfo(self._settings.alerts_timezone)
        local = (now or datetime.now(zone)).astimezone(zone)
        try:
            await run_dose_reminders(self._store, self._exotel, self._settings, local)
            # Low-stock messages go out once a day, after the chosen hour.
            if local.hour >= self._settings.alerts_low_stock_hour and (
                self._low_stock_done != local.date()
            ):
                await run_low_stock(self._store, self._exotel, self._settings, local)
                self._low_stock_done = local.date()
        except Exception:
            log.exception("alerts.scheduler_tick_failed")

    async def _loop(self) -> None:
        while True:
            await self.tick()
            await asyncio.sleep(TICK_S)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="alerts-scheduler")
            log.info("alerts.scheduler_started", every_s=TICK_S)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
