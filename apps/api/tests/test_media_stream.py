from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.config import Settings
from app.main import create_app
from tests.conftest import (
    FakeSTTProvider,
    stream_token,
    twilio_media,
    twilio_start,
    twilio_stop,
)

FRAME = b"\x7f" * 160  # one 20 ms μ-law frame


def run_call(client: TestClient, call_sid: str, frames: int = 5) -> None:
    with client.websocket_connect("/twilio/media-stream") as ws:
        ws.send_text(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        ws.send_text(twilio_start(call_sid, token=stream_token(call_sid)))
        for _ in range(frames):
            ws.send_text(twilio_media(FRAME))
        ws.send_text(twilio_media(b"\x00" * 160, track="outbound"))  # ignored
        ws.send_text(twilio_stop(call_sid))


def test_call_audio_is_transcribed_into_utterances(
    client: TestClient, fake_stt: FakeSTTProvider
) -> None:
    run_call(client, "CA100")

    assert bytes(fake_stt.received) == FRAME * 5
    call = client.get("/api/calls/CA100").json()
    assert call["status"] == "ended"
    assert call["caller"] == "***4567"
    assert [m["content"] for m in call["transcript"]] == [
        "I've had a headache for three days.",
        "It's getting worse.",
    ]
    assert call["transcript"][0]["role"] == "patient"
    assert call["audio_frames_received"] == 5
    assert call["latency"]["stt_endpoint_ms"]["count"] == 2
    assert call["latency"]["stt_finalization_ms"]["p50_ms"] == 80.0


def test_recent_calls_lists_finished_call(client: TestClient) -> None:
    run_call(client, "CA101")
    assert client.get("/api/calls/active").json() == []
    assert [c["call_sid"] for c in client.get("/api/calls/recent").json()] == ["CA101"]


def test_unknown_call_is_404(client: TestClient) -> None:
    assert client.get("/api/calls/CAnope").status_code == 404


@pytest.mark.parametrize("token", [None, "forged", stream_token("CAother")])
def test_stream_without_valid_token_is_closed(
    client: TestClient, fake_stt: FakeSTTProvider, token: str | None
) -> None:
    with client.websocket_connect("/twilio/media-stream") as ws:
        ws.send_text(twilio_start("CA200", token=token))
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1008
    assert fake_stt.streams_opened == 0


def test_stream_closed_when_stt_not_configured(settings: Settings) -> None:
    with (
        TestClient(create_app(settings, stt_provider=None)) as client,
        client.websocket_connect("/twilio/media-stream") as ws,
    ):
        ws.send_text(twilio_start("CA300", token=stream_token("CA300")))
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == 1011


def test_malformed_frames_are_ignored(client: TestClient, fake_stt: FakeSTTProvider) -> None:
    with client.websocket_connect("/twilio/media-stream") as ws:
        ws.send_text("not json")
        ws.send_text(json.dumps({"event": "mystery"}))
        ws.send_text(twilio_media(FRAME))  # media before start is dropped
        ws.send_text(twilio_start("CA400", token=stream_token("CA400")))
        ws.send_text(twilio_media(FRAME))
        ws.send_text(twilio_stop("CA400"))
    assert bytes(fake_stt.received) == FRAME


def test_client_disconnect_without_stop_still_finalizes_call(client: TestClient) -> None:
    with client.websocket_connect("/twilio/media-stream") as ws:
        ws.send_text(twilio_start("CA500", token=stream_token("CA500")))
        ws.send_text(twilio_media(FRAME))
    # Disconnect cancels the handler; the shielded close must still complete.
    deadline = time.monotonic() + 2
    while (call := client.get("/api/calls/CA500").json())["status"] != "ended":
        assert time.monotonic() < deadline, "call was never finalized"
        time.sleep(0.01)
    assert len(call["transcript"]) == 2
