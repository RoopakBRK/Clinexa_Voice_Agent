"""Cross-encoder reranking.

A bi-encoder (the embedding model) scores query and passage independently, so it is
fast but coarse. A cross-encoder reads query and passage *together* and judges
relevance directly, which is much more precise but too slow to run over the whole
corpus. The standard pattern is what we do: cheap hybrid recall of ~20 candidates,
then the cross-encoder re-orders them and keeps the best few.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from app.core.logging import get_logger
from app.rag.embeddings import detect_device
from app.schemas.clinical import RetrievedChunk

log = get_logger(__name__)


class Reranker(Protocol):
    @property
    def model_name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


def passage_for(chunk: RetrievedChunk) -> str:
    """What the cross-encoder reads: section path (context) + chunk text."""
    path = " > ".join(chunk.metadata.heading_path)
    return f"{path}\n{chunk.text}" if path else chunk.text


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
        model_factory: Callable[[str, str, int], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = detect_device(device) if model_factory is None else (device or "cpu")
        self._model_factory = model_factory or self._default_factory
        self._model: Any | None = None

    @staticmethod
    def _default_factory(model_name: str, device: str, max_length: int) -> Any:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(model_name, device=device, max_length=max_length)

    @property
    def model(self) -> Any:
        if self._model is None:
            log.info("reranker.loading", model=self.model_name, device=self._device)
            self._model = self._model_factory(self.model_name, self._device, self._max_length)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Raw relevance scores, higher = more relevant (comparable within one query;
        calibration across queries differs by model)."""
        if not passages:
            return []
        scores = self.model.predict(
            [(query, p) for p in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        return rerank_with(self, query, candidates, top_k)


def rerank_with(
    reranker: Reranker, query: str, candidates: Sequence[RetrievedChunk], top_k: int | None = None
) -> list[RetrievedChunk]:
    """Re-order ``candidates`` by cross-encoder score (best first), keeping all earlier
    stage scores and adding ``reranker_score`` / ``reranker_rank``. Ties keep the
    incoming (fused) order, so the result is deterministic."""
    if not candidates:
        return []
    started = time.perf_counter()
    scores = reranker.score(query, [passage_for(c) for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    kept = order if top_k is None else order[:top_k]
    reranked = [
        candidates[i].model_copy(
            update={
                "scores": candidates[i].scores.model_copy(
                    update={"reranker_score": scores[i], "reranker_rank": rank}
                )
            }
        )
        for rank, i in enumerate(kept, start=1)
    ]
    log.debug(
        "rerank.done",
        candidates=len(candidates),
        kept=len(reranked),
        ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return reranked


async def arerank(
    reranker: Reranker,
    query: str,
    candidates: Sequence[RetrievedChunk],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Run the (GPU/CPU-bound) scoring off the event loop."""
    return await asyncio.to_thread(rerank_with, reranker, query, candidates, top_k)
