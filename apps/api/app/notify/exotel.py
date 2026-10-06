"""Exotel REST client: WhatsApp messages and outbound voice calls.

Written from Exotel's API documentation:
* WhatsApp:  POST https://<key>:<token>@<subdomain>/v2/accounts/<sid>/messages
  https://developer.exotel.com/api/whatsapp
* Call a number and connect it to a flow:
  POST https://<key>:<token>@<subdomain>/v1/Accounts/<sid>/Calls/connect
  https://developer.exotel.com/api/make-a-call-api

Not yet run against a live Exotel account. Check the response shapes on the first
real call and message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

_INDIAN_MOBILE = re.compile(r"[6-9]\d{9}")
# WhatsApp status codes Exotel reports on the status callback.
WHATSAPP_STATUS = {30001: "sent", 30002: "delivered", 30003: "read"}
# Terminal call statuses, mapped to what the dashboard shows.
CALL_STATUS = {
    "completed": "answered",
    "busy": "no_answer",
    "no-answer": "no_answer",
    "failed": "failed",
    "canceled": "failed",
}


_XML_SID = re.compile(r"<Sid>\s*([^<\s]+)\s*</Sid>")


def _call_sid(response: httpx.Response) -> str | None:
    """The call sid from Exotel's reply. The v1 API answers in JSON or, on some accounts, XML."""
    try:
        data = response.json()
    except ValueError:
        found = _XML_SID.search(response.text)
        return found.group(1) if found else None
    call = data.get("Call") if isinstance(data, dict) else None
    sid = call.get("Sid") if isinstance(call, dict) else None
    return str(sid) if sid else None


class ExotelError(Exception):
    """Exotel refused the request or could not be reached."""


def to_e164_india(number: str | None) -> str | None:
    """A 10-digit Indian mobile number as +91XXXXXXXXXX, or None if it is not one.

    Placeholder numbers (all zeros, test data) fail this check, so they are never
    called or messaged.
    """
    digits = re.sub(r"\D", "", number or "")
    last10 = digits[-10:]
    if len(digits) < 10 or not _INDIAN_MOBILE.fullmatch(last10):
        return None
    return f"+91{last10}"


def mask(number: str | None) -> str:
    """Phone numbers are personal data: only the last four digits go into logs."""
    digits = re.sub(r"\D", "", number or "")
    return f"***{digits[-4:]}" if len(digits) >= 4 else "***"


@dataclass(frozen=True)
class ExotelConfig:
    api_key: str
    api_token: str
    account_sid: str
    subdomain: str
    caller_id: str | None
    voice_app_id: str | None
    whatsapp_from: str | None
    template_language: str
    call_time_limit_s: int
    ring_timeout_s: int

    @property
    def base_url(self) -> str:
        return f"https://{self.subdomain}"

    @property
    def voice_flow_url(self) -> str:
        return f"http://my.exotel.com/{self.account_sid}/exoml/start_voice/{self.voice_app_id}"


class ExotelClient:
    def __init__(self, config: ExotelConfig, *, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url,
            auth=(config.api_key, config.api_token),
            timeout=15.0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def can_message(self) -> bool:
        return self._config.whatsapp_from is not None

    @property
    def can_call(self) -> bool:
        return self._config.caller_id is not None and self._config.voice_app_id is not None

    async def send_whatsapp_template(
        self,
        to: str,
        *,
        template: str,
        body_params: list[str],
        custom_data: str,
        status_callback: str | None = None,
    ) -> str:
        """Send an approved template message. Returns Exotel's message sid.

        Outside the 24-hour window after the person last wrote to us, WhatsApp only
        allows approved templates, so reminders always go as templates.
        """
        if self._config.whatsapp_from is None:
            raise ExotelError("EXOTEL_WHATSAPP_FROM is not set")
        body: dict[str, Any] = {
            "custom_data": custom_data,
            "whatsapp": {
                "messages": [
                    {
                        "from": self._config.whatsapp_from,
                        "to": to,
                        "content": {
                            "type": "template",
                            "template": {
                                "name": template,
                                "language": {
                                    "policy": "deterministic",
                                    "code": self._config.template_language,
                                },
                                "components": [
                                    {
                                        "type": "body",
                                        "parameters": [
                                            {"type": "text", "text": value} for value in body_params
                                        ],
                                    }
                                ],
                            },
                        },
                    }
                ]
            },
        }
        if status_callback:
            body["status_callback"] = status_callback
        data = await self._post(f"/v2/accounts/{self._config.account_sid}/messages", json=body)
        try:
            message = data["response"]["whatsapp"]["messages"][0]
            sid = message["data"]["sid"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ExotelError("WhatsApp message was not accepted") from exc
        log.info("exotel.whatsapp_sent", to=mask(to), template=template, sid=sid)
        return str(sid)

    async def place_call(
        self, to: str, *, custom_field: str, status_callback: str | None = None
    ) -> str:
        """Ring ``to`` and connect them to the voice flow. Returns Exotel's call sid."""
        if not self.can_call:
            raise ExotelError("EXOTEL_CALLER_ID and EXOTEL_VOICE_APP_ID must be set")
        form: dict[str, str] = {
            "From": to,
            "CallerId": self._config.caller_id or "",
            "Url": self._config.voice_flow_url,
            # Transactional: a reminder the person asked for, not marketing.
            "CallType": "trans",
            "TimeLimit": str(self._config.call_time_limit_s),
            "TimeOut": str(self._config.ring_timeout_s),
            "CustomField": custom_field,
        }
        if status_callback:
            form["StatusCallback"] = status_callback
            form["StatusCallbackEvents[0]"] = "terminal"
            form["StatusCallbackContentType"] = "application/json"
        response = await self._send(
            "POST", f"/v1/Accounts/{self._config.account_sid}/Calls/connect.json", data=form
        )
        sid = _call_sid(response)
        if sid is None:
            raise ExotelError("the call was not accepted")
        log.info("exotel.call_placed", to=mask(to), sid=sid)
        return sid

    async def list_exophones(self) -> list[str]:
        """The account's ExoPhones. Read-only: used to check that the keys work."""
        response = await self._send(
            "GET", f"/v2_beta/Accounts/{self._config.account_sid}/IncomingPhoneNumbers"
        )
        try:
            data = response.json()
        except ValueError:
            return []
        rows = data.get("incoming_phone_numbers") if isinstance(data, dict) else data
        return [
            str(row.get("phone_number"))
            for row in (rows if isinstance(rows, list) else [])
            if isinstance(row, dict) and row.get("phone_number")
        ]

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ExotelError(f"could not reach Exotel: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            # The body can echo phone numbers, so only the status is reported.
            raise ExotelError(f"Exotel returned {response.status_code}")
        return response

    async def _post(self, path: str, **kwargs: Any) -> dict[str, Any]:
        response = await self._send("POST", path, **kwargs)
        try:
            data = response.json()
        except ValueError as exc:
            raise ExotelError("Exotel returned something that is not JSON") from exc
        if not isinstance(data, dict):
            raise ExotelError("Exotel returned an unexpected response")
        return data


def build_exotel_client(settings: Settings) -> ExotelClient | None:
    if (
        settings.exotel_api_key is None
        or settings.exotel_api_token is None
        or settings.exotel_account_sid is None
    ):
        log.warning("exotel.not_configured", hint="set EXOTEL_API_KEY, _TOKEN and _ACCOUNT_SID")
        return None
    return ExotelClient(
        ExotelConfig(
            api_key=settings.exotel_api_key.get_secret_value(),
            api_token=settings.exotel_api_token.get_secret_value(),
            account_sid=settings.exotel_account_sid,
            subdomain=settings.exotel_subdomain,
            caller_id=settings.exotel_caller_id,
            voice_app_id=settings.exotel_voice_app_id,
            whatsapp_from=settings.exotel_whatsapp_from,
            template_language=settings.exotel_template_language,
            call_time_limit_s=settings.exotel_call_time_limit_s,
            ring_timeout_s=settings.exotel_ring_timeout_s,
        )
    )
