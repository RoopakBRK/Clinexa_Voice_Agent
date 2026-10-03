"""Extractive context compression: keep the sentences that answer the question.

Each retained chunk is cut down to its query-relevant sentences/bullets before it
reaches the LLM, which shortens the prompt (faster first token on a phone call).

Clinical text is dangerous to shorten carelessly, so compression is conservative:
  * a chunk already within the token budget is returned unchanged;
  * the budget is spent on the most query-relevant sentences; the items below are kept on top;
  * the opening sentence (usually the framing) is always kept;
  * sentences carrying a caution or referral cue ("do not", "refer urgently",
    "contraindicated", "danger sign"...) are ALWAYS kept, whatever their relevance;
  * original order is preserved, and omissions are marked with "…".
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel

from app.rag.embeddings import Embedder
from app.schemas.clinical import RetrievedChunk

TokenCounter = Callable[[str], int]

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'•-])")
_SAFETY_CUE = re.compile(
    r"\b(?:do not|don't|never|must not|should not|avoid|contraindicat\w*|not recommended|"
    r"danger signs?|red flags?|refer(?:ral)?\b|urgent(?:ly)?|emergency|immediately|life[- ]threatening|"
    r"warning|caution|toxic\w*|only if|unless)\b",
    re.I,
)
OMISSION = "…"


class CompressedEvidence(BaseModel):
    chunk_id: str
    text: str
    original_tokens: int
    kept_tokens: int
    kept_units: int
    total_units: int

    @property
    def compressed(self) -> bool:
        return self.kept_units < self.total_units


def split_units(text: str) -> list[str]:
    """Sentences and bullet/table lines, in order. A table is kept as rows."""
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(("-", "|")):
            units.append(line)
        else:
            units.extend(s.strip() for s in _SENTENCE_END.split(line) if s.strip())
    return units


def _words(text: str) -> int:
    return len(text.split())


def compress_chunk(
    query_vector: Sequence[float],
    chunk: RetrievedChunk,
    embedder: Embedder,
    *,
    token_budget: int = 200,
    count: TokenCounter = _words,
) -> CompressedEvidence:
    units = split_units(chunk.text)
    original = count(chunk.text)
    whole = CompressedEvidence(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        original_tokens=original,
        kept_tokens=original,
        kept_units=len(units),
        total_units=len(units),
    )
    if original <= token_budget or len(units) <= 2:
        return whole

    # Table header rows ("|a|b|" + "|---|---|") must travel with any kept table row.
    header = [u for u in units[:2] if u.startswith("|")][:2] if units[0].startswith("|") else []
    vectors = np.array(embedder.embed_documents(units))
    sims = vectors @ np.array(query_vector)  # embeddings are L2-normalised: cosine similarity

    # Forced units (framing, table header, caution/referral cues) are kept IN ADDITION to the
    # budget, which is spent only on relevance-ranked units. Otherwise irrelevant cautions
    # could crowd out the very sentence that answers the question.
    forced = {0, *range(len(header))}
    forced |= {i for i, u in enumerate(units) if _SAFETY_CUE.search(u)}
    kept = set(forced)
    used = 0
    for i in np.argsort(-sims):
        i = int(i)
        if i in kept:
            continue
        cost = count(units[i])
        if used + cost > token_budget:
            continue
        kept.add(i)
        used += cost

    if len(kept) == len(units):
        return whole
    parts: list[str] = []
    previous = -1
    for i in sorted(kept):
        if previous != -1 and i != previous + 1:
            parts.append(OMISSION)
        parts.append(units[i])
        previous = i
    if previous != len(units) - 1:
        parts.append(OMISSION)
    text = "\n".join(parts)
    return CompressedEvidence(
        chunk_id=chunk.chunk_id,
        text=text,
        original_tokens=original,
        kept_tokens=count(text),
        kept_units=len(kept),
        total_units=len(units),
    )


def compress_chunks(
    query: str,
    chunks: Sequence[RetrievedChunk],
    embedder: Embedder,
    *,
    token_budget: int = 200,
    count: TokenCounter = _words,
) -> list[CompressedEvidence]:
    query_vector = embedder.embed_query(query)
    return [
        compress_chunk(query_vector, c, embedder, token_budget=token_budget, count=count)
        for c in chunks
    ]
