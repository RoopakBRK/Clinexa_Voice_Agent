from __future__ import annotations

from app.core.config import Settings
from app.voice.session import CallSession, CallStatus
from app.voice.turns import Utterance
from tests.conftest import FakeSTTProvider, final


def make_session(settings: Settings, stt: FakeSTTProvider, **kwargs: object) -> CallSession:
    return CallSession(call_sid="CA1", stream_sid="MZ1", stt=stt, settings=settings, **kwargs)  # type: ignore[arg-type]


async def test_utterance_handler_receives_each_turn(settings: Settings) -> None:
    stt = FakeSTTProvider([final("Hello.", speech_final=True), final("Still there?")])
    seen: list[str] = []

    async def handler(session: CallSession, utterance: Utterance) -> None:
        seen.append(utterance.text)

    session = make_session(settings, stt, on_utterance=handler)
    session.start()
    session.feed_audio(b"\xff" * 160)
    await session.close()

    # The second turn had no endpoint before hang-up; it is flushed on close.
    assert seen == ["Hello.", "Still there?"]
    assert session.status is CallStatus.ENDED


async def test_stt_reconnects_without_losing_audio(settings: Settings) -> None:
    stt = FakeSTTProvider([final("Recovered.", speech_final=True)], fail_first=1)
    session = make_session(settings, stt)
    session.start()
    for i in range(3):
        session.feed_audio(bytes([i]) * 160)
    await session.close()

    assert stt.streams_opened == 2
    assert session.stt_reconnects == 1
    assert bytes(stt.received) == b"".join(bytes([i]) * 160 for i in range(3))
    assert session.state.conversation_history[0].content == "Recovered."


async def test_stt_marked_unavailable_after_max_retries(settings: Settings) -> None:
    settings = settings.model_copy(update={"stt_max_reconnect_attempts": 1})
    stt = FakeSTTProvider([], fail_first=10)
    session = make_session(settings, stt)
    session.start()
    for _ in range(3):
        session.feed_audio(b"\xff" * 160)
    assert session._stt_task is not None
    await session._stt_task

    assert session.status is CallStatus.STT_UNAVAILABLE
    session.feed_audio(b"\xff" * 160)  # ignored, does not grow the queue
    assert session.frames_received == 3
    await session.close()


async def test_full_audio_queue_drops_frames_instead_of_blocking(settings: Settings) -> None:
    settings = settings.model_copy(update={"audio_queue_max_frames": 50})
    session = make_session(settings, FakeSTTProvider([]))  # STT never started
    for _ in range(60):
        session.feed_audio(b"\xff" * 160)
    assert session.frames_dropped == 10
    await session.close()
