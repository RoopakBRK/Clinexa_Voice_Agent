"""Unit tests for pure components: protocol, audio, turns, metrics, schemas."""

from __future__ import annotations

import base64
import json
import warnings

import pytest

from app.graph.state import VoiceClinicalState
from app.observability.metrics import LatencyMetrics, percentile
from app.schemas.clinical import ClinicalIntake, SafetyAssessment, Urgency
from app.voice.audio import AudioClock, mulaw_to_pcm16, pcm16_to_mulaw
from app.voice.providers.base import TranscriptEvent, TranscriptEventType
from app.voice.security import sign_stream_token, verify_stream_token
from app.voice.turns import UtteranceAssembler
from app.voice.twilio_protocol import (
    MediaMessage,
    StartMessage,
    StopMessage,
    outbound_clear,
    outbound_mark,
    outbound_media,
    parse_inbound,
)
from app.voice.twilio_webhook import mask_phone_number
from tests.conftest import final, twilio_start

# --- Twilio protocol ------------------------------------------------------


def test_parse_start_message() -> None:
    msg = parse_inbound(twilio_start("CA1", token="tok"))
    assert isinstance(msg, StartMessage)
    assert msg.start.call_sid == "CA1"
    assert msg.start.custom_parameters["stream_token"] == "tok"
    assert msg.start.media_format.sample_rate == 8000


def test_parse_media_message_decodes_audio() -> None:
    raw = json.dumps(
        {
            "event": "media",
            "sequenceNumber": "3",
            "streamSid": "MZ1",
            "media": {
                "track": "inbound",
                "chunk": "1",
                "timestamp": "5",
                "payload": base64.b64encode(b"\x01\x02").decode(),
            },
        }
    )
    msg = parse_inbound(raw)
    assert isinstance(msg, MediaMessage)
    assert msg.audio() == b"\x01\x02"


def test_parse_stop_and_unknown() -> None:
    assert isinstance(parse_inbound('{"event": "stop", "streamSid": "MZ1"}'), StopMessage)
    assert parse_inbound('{"event": "mystery"}') is None
    assert parse_inbound("{not json") is None


def test_outbound_messages() -> None:
    media = json.loads(outbound_media("MZ1", b"\xff\xfe"))
    assert media == {"event": "media", "streamSid": "MZ1", "media": {"payload": "//4="}}
    assert json.loads(outbound_mark("MZ1", "turn-1"))["mark"] == {"name": "turn-1"}
    assert json.loads(outbound_clear("MZ1")) == {"event": "clear", "streamSid": "MZ1"}


# --- security -------------------------------------------------------------


def test_stream_token_is_bound_to_call() -> None:
    token = sign_stream_token("s3cret", "CA1")
    assert verify_stream_token("s3cret", "CA1", token)
    assert not verify_stream_token("s3cret", "CA2", token)
    assert not verify_stream_token("other", "CA1", token)
    assert not verify_stream_token("s3cret", "CA1", None)


@pytest.mark.parametrize(
    ("raw", "masked"), [("+15551234567", "***4567"), ("12", "***"), (None, None), ("", None)]
)
def test_mask_phone_number(raw: str | None, masked: str | None) -> None:
    assert mask_phone_number(raw) == masked


# --- audio ----------------------------------------------------------------


def test_mulaw_known_values() -> None:
    assert pcm16_to_mulaw((0).to_bytes(2, "little", signed=True)) == b"\xff"
    assert mulaw_to_pcm16(b"\xff") == b"\x00\x00"


def test_mulaw_matches_reference_codec() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        audioop = pytest.importorskip("audioop")
    pcm = b"".join(s.to_bytes(2, "little", signed=True) for s in range(-32768, 32768, 7))
    assert pcm16_to_mulaw(pcm) == audioop.lin2ulaw(pcm, 2)
    ulaw = bytes(range(256))
    assert mulaw_to_pcm16(ulaw) == audioop.ulaw2lin(ulaw, 2)


def test_audio_clock_maps_audio_time_to_send_time() -> None:
    clock = AudioClock(bytes_per_second=8000)
    clock.record(160, now=10.0)  # audio 0.00-0.02 s
    clock.record(160, now=10.5)  # audio 0.02-0.04 s
    assert clock.seconds_sent == pytest.approx(0.04)
    assert clock.sent_at(0.01) == 10.0
    assert clock.sent_at(0.02) == 10.0
    assert clock.sent_at(0.03) == 10.5
    assert clock.sent_at(0.05) is None
    assert clock.latency_ms(0.04, now=10.75) == 250.0
    assert clock.latency_ms(None) is None


# --- turns ----------------------------------------------------------------


def test_assembler_joins_finals_until_speech_final() -> None:
    a = UtteranceAssembler()
    assert a.process(TranscriptEvent(type=TranscriptEventType.INTERIM, text="I've")) is None
    assert a.process(final("I've had a cough", confidence=0.8)) is None
    utterance = a.process(final("for five days.", speech_final=True, confidence=1.0))
    assert utterance is not None
    assert utterance.text == "I've had a cough for five days."
    assert utterance.confidence == 0.9
    assert utterance.end_reason == "speech_final"
    assert a.pending_text == ""


def test_assembler_uses_utterance_end_as_fallback() -> None:
    a = UtteranceAssembler()
    a.process(final("Chest pain"))
    utterance = a.process(TranscriptEvent(type=TranscriptEventType.UTTERANCE_END))
    assert utterance is not None and utterance.end_reason == "utterance_end"
    # UtteranceEnd right after speech_final must not emit an empty turn.
    assert a.process(TranscriptEvent(type=TranscriptEventType.UTTERANCE_END)) is None


def test_assembler_ignores_empty_speech_final() -> None:
    assert UtteranceAssembler().process(final("", speech_final=True)) is None


# --- metrics --------------------------------------------------------------


def test_percentile_matches_linear_interpolation() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == pytest.approx(50.5)
    assert percentile(values, 95) == pytest.approx(95.05)
    assert percentile(values, 99) == pytest.approx(99.01)
    assert percentile([42.0], 99) == 42.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_latency_metrics_summary() -> None:
    m = LatencyMetrics()
    for v in (100, 200, 300):
        m.record("stt_endpoint_ms", v)
    stats = m.summary()["stt_endpoint_ms"]
    assert (stats.count, stats.mean_ms, stats.p50_ms, stats.max_ms) == (3, 200.0, 200.0, 300.0)


# --- schemas --------------------------------------------------------------


@pytest.mark.parametrize("urgency", [Urgency.URGENT, Urgency.EMERGENCY])
def test_urgent_safety_assessment_always_escalates(urgency: Urgency) -> None:
    assessment = SafetyAssessment(urgency=urgency, confidence=0.9, escalation_required=False)
    assert assessment.escalation_required is True


def test_safety_confidence_is_bounded() -> None:
    with pytest.raises(ValueError):
        SafetyAssessment(confidence=1.5)


def test_intake_reports_missing_core_fields() -> None:
    intake = ClinicalIntake(chief_complaint="headache", duration="3 days")
    assert intake.missing() == ["severity"]


def test_state_round_trips_through_json() -> None:
    state = VoiceClinicalState(call_id="CA1", clinical_state=ClinicalIntake(symptoms=["cough"]))
    restored = VoiceClinicalState.model_validate_json(state.model_dump_json())
    assert restored == state
