from __future__ import annotations

import xml.etree.ElementTree as ET

from fastapi.testclient import TestClient
from twilio.request_validator import RequestValidator

from app.core.config import Settings
from app.main import create_app
from tests.conftest import AUTH_TOKEN, PUBLIC_BASE_URL, stream_token

FORM = {"CallSid": "CA123", "From": "+15551234567", "To": "+15557654321", "AccountSid": "ACtest"}


def signed_headers(params: dict[str, str], path: str = "/twilio/voice") -> dict[str, str]:
    signature = RequestValidator(AUTH_TOKEN).compute_signature(f"{PUBLIC_BASE_URL}{path}", params)
    return {"X-Twilio-Signature": signature}


def test_health_reports_stt_provider(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["stt"] == {"provider": "fake", "configured": True}


def test_incoming_call_returns_stream_twiml(client: TestClient) -> None:
    resp = client.post("/twilio/voice", data=FORM, headers=signed_headers(FORM))

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/xml")
    root = ET.fromstring(resp.text)
    say = root.find("Say")
    assert say is not None and "not a doctor" in (say.text or "")
    stream = root.find("Connect/Stream")
    assert stream is not None
    assert stream.get("url") == "wss://clinexa.example.com/twilio/media-stream"
    params = {p.get("name"): p.get("value") for p in stream.findall("Parameter")}
    assert params["stream_token"] == stream_token("CA123")
    # The full caller number never leaves the webhook.
    assert params["caller"] == "***4567"


def test_invalid_signature_is_rejected(client: TestClient) -> None:
    resp = client.post("/twilio/voice", data=FORM, headers={"X-Twilio-Signature": "forged"})
    assert resp.status_code == 403


def test_missing_signature_is_rejected(client: TestClient) -> None:
    assert client.post("/twilio/voice", data=FORM).status_code == 403


def test_signature_over_tampered_params_is_rejected(client: TestClient) -> None:
    headers = signed_headers(FORM)
    tampered = {**FORM, "CallSid": "CAother"}
    assert client.post("/twilio/voice", data=tampered, headers=headers).status_code == 403


def test_validation_fails_closed_without_auth_token(settings: Settings) -> None:
    settings = settings.model_copy(update={"twilio_auth_token": None})
    with TestClient(create_app(settings, stt_provider=None)) as client:
        assert client.post("/twilio/voice", data=FORM).status_code == 403


def test_missing_call_sid_is_bad_request(settings: Settings) -> None:
    settings = settings.model_copy(update={"twilio_validate_signatures": False})
    with TestClient(create_app(settings, stt_provider=None)) as client:
        assert client.post("/twilio/voice", data={"From": "+15551234567"}).status_code == 400


def test_stream_url_falls_back_to_request_host(settings: Settings) -> None:
    settings = settings.model_copy(
        update={"public_base_url": None, "twilio_validate_signatures": False}
    )
    with TestClient(create_app(settings, stt_provider=None)) as client:
        resp = client.post("/twilio/voice", data=FORM)
    stream = ET.fromstring(resp.text).find("Connect/Stream")
    assert stream is not None and stream.get("url") == "ws://testserver/twilio/media-stream"
