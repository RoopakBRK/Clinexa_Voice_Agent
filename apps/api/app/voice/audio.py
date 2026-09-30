"""Audio utilities: G.711 μ-law codec and an audio-timeline clock.

``audioop`` was removed in Python 3.13, so μ-law is implemented here following
the Sun/ITU G.711 reference (bit-exact with ``audioop``), with lookup tables
(encode: 64 Ki entries, decode: 256 entries).
"""

from __future__ import annotations

import array
import sys
import time
from bisect import bisect_left
from functools import cache

MULAW_SILENCE = b"\xff"
_BIAS = 0x84
_ENC_BIAS = _BIAS >> 2  # encoder works on 14-bit magnitudes
_ENC_CLIP = 8159
_SEG_END = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)


def _encode_sample(sample: int) -> int:
    sample >>= 2
    if sample < 0:
        sample, mask = -sample, 0x7F
    else:
        mask = 0xFF
    sample = min(sample, _ENC_CLIP) + _ENC_BIAS
    segment = bisect_left(_SEG_END, sample)
    if segment >= len(_SEG_END):
        return 0x7F ^ mask
    return ((segment << 4) | ((sample >> (segment + 1)) & 0x0F)) ^ mask


def _decode_sample(byte: int) -> int:
    byte = ~byte & 0xFF
    t = (((byte & 0x0F) << 3) + _BIAS) << ((byte >> 4) & 0x07)
    return _BIAS - t if byte & 0x80 else t - _BIAS


@cache
def _encode_table() -> bytes:
    # Indexed by the unsigned 16-bit view of each signed sample.
    return bytes(_encode_sample(i if i < 0x8000 else i - 0x10000) for i in range(0x10000))


@cache
def _decode_table() -> tuple[int, ...]:
    return tuple(_decode_sample(b) for b in range(256))


def pcm16_to_mulaw(pcm: bytes) -> bytes:
    """Encode little-endian signed 16-bit PCM to μ-law."""
    samples = array.array("H", pcm)
    if sys.byteorder == "big":
        samples.byteswap()
    table = _encode_table()
    return bytes(table[s] for s in samples)


def mulaw_to_pcm16(data: bytes) -> bytes:
    """Decode μ-law to little-endian signed 16-bit PCM."""
    table = _decode_table()
    samples = array.array("h", (table[b] for b in data))
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


class AudioClock:
    """Maps positions in a sent audio stream to when they were sent.

    STT providers timestamp results in audio time (seconds since stream start).
    Recording when each chunk left this process lets us turn those timestamps
    into real wall-clock latency.
    """

    _MAX_MARKS = 6000  # ~2 minutes of 20 ms frames; older marks are trimmed

    def __init__(self, bytes_per_second: int) -> None:
        self._bytes_per_second = bytes_per_second
        self._sent_bytes = 0
        self._audio_ends: list[float] = []
        self._sent_times: list[float] = []

    @property
    def seconds_sent(self) -> float:
        return self._sent_bytes / self._bytes_per_second

    def record(self, n_bytes: int, now: float | None = None) -> None:
        self._sent_bytes += n_bytes
        self._audio_ends.append(self.seconds_sent)
        self._sent_times.append(time.monotonic() if now is None else now)
        if len(self._audio_ends) > self._MAX_MARKS:
            half = self._MAX_MARKS // 2
            del self._audio_ends[:half]
            del self._sent_times[:half]

    def sent_at(self, audio_s: float) -> float | None:
        """Monotonic time the chunk containing ``audio_s`` was sent, if known."""
        i = bisect_left(self._audio_ends, audio_s - 1e-6)
        if i >= len(self._audio_ends):
            return None
        return self._sent_times[i]

    def latency_ms(self, audio_s: float | None, now: float | None = None) -> float | None:
        if audio_s is None:
            return None
        sent = self.sent_at(audio_s)
        if sent is None:
            return None
        return round(((time.monotonic() if now is None else now) - sent) * 1000, 2)
