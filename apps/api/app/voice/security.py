"""Request authentication for the Twilio voice endpoints.

* HTTP webhooks are verified with Twilio's ``X-Twilio-Signature``.
* The Media Stream WebSocket is bound to the verified webhook with a per-call
  HMAC token passed through TwiML ``<Parameter>`` and checked on ``start``.
"""

from __future__ import annotations

import hashlib
import hmac

from fastapi import Depends, HTTPException, Request, status
from twilio.request_validator import RequestValidator

from app.core.config import Settings
from app.core.logging import get_logger
from app.voice.deps import get_settings

log = get_logger(__name__)

STREAM_TOKEN_PARAM = "stream_token"


def sign_stream_token(secret: str, call_sid: str) -> str:
    return hmac.new(secret.encode(), call_sid.encode(), hashlib.sha256).hexdigest()


def verify_stream_token(secret: str, call_sid: str, token: str | None) -> bool:
    if not token:
        return False
    return hmac.compare_digest(sign_stream_token(secret, call_sid), token)


def public_url_for(request: Request, settings: Settings) -> str:
    """The URL Twilio actually requested, as seen from the public internet.

    Behind ngrok or a load balancer ``request.url`` is the internal URL, which
    would not match Twilio's signature, so the configured public origin wins.
    """
    if settings.public_base_url:
        query = f"?{request.url.query}" if request.url.query else ""
        return f"{settings.public_base_url}{request.url.path}{query}"
    return str(request.url)


async def verify_twilio_signature(
    request: Request, settings: Settings = Depends(get_settings)
) -> None:
    if not settings.twilio_validate_signatures:
        return
    if settings.twilio_auth_token is None:
        log.error("twilio.signature_validation_misconfigured")
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Signature validation not configured")

    form = await request.form()
    params = {k: v for k, v in form.items() if isinstance(v, str)}
    signature = request.headers.get("X-Twilio-Signature", "")
    validator = RequestValidator(settings.twilio_auth_token.get_secret_value())
    if not validator.validate(public_url_for(request, settings), params, signature):
        log.warning("twilio.invalid_signature", path=request.url.path)
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid Twilio signature")
