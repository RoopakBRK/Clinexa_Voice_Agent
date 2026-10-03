"""BM25 sparse retrieval over the chunk corpus.

Dense embeddings capture meaning but can blur exact clinical terms (drug names,
doses, "G6PD", "artemether-lumefantrine"); BM25 matches those exactly. The index
holds ~4k chunks and builds in about a second, so it is built in memory at start-up
from ``chunks.jsonl`` rather than persisted, which keeps it always consistent with
the chunk file.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Sequence

import bm25s
import numpy as np
import Stemmer

from app.core.logging import get_logger
from app.rag.models import Chunk
from app.rag.retrieval.filters import RetrievalFilters
from app.schemas.clinical import RetrievalScores, RetrievedChunk

log = get_logger(__name__)

_MASK_CACHE_SIZE = 128


def indexed_text(chunk: Chunk) -> str:
    """What BM25 sees: the section path (strong signal: "Headache", "Dosing of ACTs")
    plus the chunk text. The document title is omitted: it repeats in every chunk."""
    return f"{' '.join(chunk.metadata.heading_path)}\n{chunk.text}"


class SparseRetriever:
    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.2, b: float = 0.75) -> None:
        self._chunks = [c for c in chunks if c.metadata.retrievable]
        if not self._chunks:
            raise ValueError("SparseRetriever needs at least one retrievable chunk")
        self._stemmer = Stemmer.Stemmer("english")
        corpus = bm25s.tokenize(
            [indexed_text(c) for c in self._chunks],
            stopwords="en",
            stemmer=self._stemmer,
            show_progress=False,
        )
        self._bm25 = bm25s.BM25(k1=k1, b=b)
        self._bm25.index(corpus, show_progress=False)
        self._masks: OrderedDict[str, np.ndarray] = OrderedDict()
        log.info("bm25.built", chunks=len(self._chunks), k1=k1, b=b)

    def __len__(self) -> int:
        return len(self._chunks)

    def _tokens(self, query: str) -> list[str]:
        return bm25s.tokenize(  # type: ignore[no-any-return]
            query, stopwords="en", stemmer=self._stemmer, return_ids=False, show_progress=False
        )[0]

    def _mask(self, filters: RetrievalFilters | None) -> np.ndarray | None:
        if filters is None or filters.is_empty:
            return None  # every indexed chunk is retrievable already
        key = filters.model_dump_json()
        if (cached := self._masks.get(key)) is not None:
            self._masks.move_to_end(key)
            return cached
        mask = np.array([filters.matches(c.metadata) for c in self._chunks], dtype=np.float32)
        self._masks[key] = mask
        if len(self._masks) > _MASK_CACHE_SIZE:
            self._masks.popitem(last=False)
        return mask

    def search(
        self, query: str, *, filters: RetrievalFilters | None = None, k: int = 15
    ) -> list[RetrievedChunk]:
        tokens = self._tokens(query)
        if not tokens or k <= 0:
            return []
        scores = self._bm25.get_scores(tokens, weight_mask=self._mask(filters))
        top = min(k, len(scores))
        candidates = np.argpartition(-scores, top - 1)[:top]
        # Highest score first; ties broken by corpus order so results are deterministic.
        ordered = sorted(candidates.tolist(), key=lambda i: (-float(scores[i]), i))
        return [
            RetrievedChunk(
                chunk_id=self._chunks[i].chunk_id,
                text=self._chunks[i].text,
                metadata=self._chunks[i].metadata,
                scores=RetrievalScores(bm25_score=float(scores[i]), bm25_rank=rank),
            )
            for rank, i in enumerate(ordered, start=1)
            if scores[i] > 0
        ]

    async def asearch(
        self, query: str, *, filters: RetrievalFilters | None = None, k: int = 15
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self.search, query, filters=filters, k=k)
