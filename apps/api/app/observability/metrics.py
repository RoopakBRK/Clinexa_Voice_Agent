"""Per-call latency metrics with percentile summaries."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel


def percentile(values: Sequence[float], p: float) -> float:
    """Percentile with linear interpolation between closest ranks (NumPy's default)."""
    if not values:
        raise ValueError("percentile() of empty sequence")
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


class LatencyStats(BaseModel):
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


class LatencyMetrics:
    """Collects latency samples (in milliseconds) keyed by metric name.

    Metric names used by the pipeline:
      stt_finalization_ms     audio sent -> final transcript received
      stt_endpoint_ms         caller's last word -> end-of-utterance detected
      llm_ttft_ms             reply requested -> first text from the LLM
      llm_first_sentence_ms   reply requested -> first complete sentence
      llm_total_ms            reply requested -> full reply text
      tts_ttfa_ms             first sentence ready -> first audio from TTS
      response_latency_ms     end-of-utterance detected -> first reply audio sent to Twilio
      voice_to_voice_ms       caller's last word -> first reply audio sent to Twilio
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    def record(self, name: str, value_ms: float) -> None:
        self._samples[name].append(value_ms)

    def samples(self, name: str) -> list[float]:
        return list(self._samples.get(name, []))

    def summary(self) -> dict[str, LatencyStats]:
        return {
            name: LatencyStats(
                count=len(vals),
                mean_ms=round(sum(vals) / len(vals), 2),
                p50_ms=round(percentile(vals, 50), 2),
                p95_ms=round(percentile(vals, 95), 2),
                p99_ms=round(percentile(vals, 99), 2),
                max_ms=round(max(vals), 2),
            )
            for name, vals in self._samples.items()
            if vals
        }
