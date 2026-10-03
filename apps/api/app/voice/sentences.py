"""Splits streamed LLM text into sentences that can be sent to TTS one at a time.

The first sentence of a reply is spoken while the rest is still being generated,
so the split has to be made on partial text: a sentence is only emitted once the
whitespace after its closing punctuation has arrived ("3.5" and "e.g." never split).
"""

from __future__ import annotations

import re

_BOUNDARY = re.compile(r"([.!?]+[\"')\]]*)\s+")
_ABBREVIATIONS = frozenset(
    {"dr", "mr", "mrs", "ms", "prof", "st", "vs", "etc", "e.g", "i.e", "approx"}
)


def _ends_with_abbreviation(sentence: str) -> bool:
    if not sentence.endswith(".") or sentence.endswith(".."):
        return False
    words = sentence[:-1].split()
    return bool(words) and words[-1].lower().lstrip("(\"'") in _ABBREVIATIONS


class SentenceChunker:
    def __init__(self, min_chars: int = 12) -> None:
        # Very short fragments ("Okay.") are held and spoken with the next sentence:
        # a separate synthesis request for one word costs more than it saves.
        self._min_chars = min_chars
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Add streamed text; returns the sentences it completed, in order."""
        self._buffer += text
        sentences: list[str] = []
        pos = 0
        while (match := _BOUNDARY.search(self._buffer, pos)) is not None:
            candidate = self._buffer[: match.end(1)].strip()
            if len(candidate) < self._min_chars or _ends_with_abbreviation(candidate):
                pos = match.end()
                continue
            sentences.append(candidate)
            self._buffer = self._buffer[match.end() :]
            pos = 0
        return sentences

    def flush(self) -> str | None:
        """End of text: whatever is left, if anything."""
        rest, self._buffer = self._buffer.strip(), ""
        return rest or None
