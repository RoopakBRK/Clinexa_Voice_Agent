"""Twilio Programmable Voice webhook: answers the call and opens a Media Stream."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from twilio.twiml.voice_response import Connect, VoiceResponse

from app.core.config import Settings
from app.core.logging import get_logger
from app.voice.deps import get_settings
from app.voice.security import STREAM_TOKEN_PARAM, sign_stream_token, verify_twilio_signature

log = get_logger(__name__)

router = APIRouter(prefix="/twilio", tags=["twilio"])

MEDIA_STREAM_PATH = "/twilio/media-stream"
CALLER_PARAM = "caller"


def mask_phone_number(number: str | None) -> str | None:
    if not number:
        return None
    digits = "".join(c for c in number if c.isdigit())
    return f"***{digits[-4:]}" if len(digits) >= 4 else "***"


def media_stream_url(request: Request, settings: Settings) -> str:
    base = settings.public_base_url or str(request.base_url).rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base.removeprefix("https://")
    elif base.startswith("http://"):
        base = "ws://" + base.removeprefix("http://")
    return f"{base}{MEDIA_STREAM_PATH}"


def build_connect_twiml(
    *,
    greeting: str,
    voice: str,
    stream_url: str,
    stream_token: str,
    caller: str | None,
) -> str:
    response = VoiceResponse()
    response.say(greeting, voice=voice)
    connect = Connect()
    stream = connect.stream(url=stream_url)
    stream.parameter(name=STREAM_TOKEN_PARAM, value=stream_token)
    if caller:
        stream.parameter(name=CALLER_PARAM, value=caller)
    response.append(connect)
    return str(response)


@router.post("/voice", dependencies=[Depends(verify_twilio_signature)])
async def incoming_call(request: Request, settings: Settings = Depends(get_settings)) -> Response:
    form = await request.form()
    call_sid = form.get("CallSid")
    if not isinstance(call_sid, str) or not call_sid:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing CallSid")
    caller_raw = form.get("From")
    caller = mask_phone_number(caller_raw if isinstance(caller_raw, str) else None)

    twiml = build_connect_twiml(
        greeting=settings.greeting_text,
        voice=settings.twilio_say_voice,
        stream_url=media_stream_url(request, settings),
        stream_token=sign_stream_token(settings.stream_token_secret.get_secret_value(), call_sid),
        caller=caller,
    )
    log.info("twilio.incoming_call", call_sid=call_sid, caller=caller)
    return Response(content=twiml, media_type="application/xml")
