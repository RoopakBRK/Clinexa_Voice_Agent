"""Hybrid retrieval: dense + BM25 in parallel, fused with RRF.

    query ──┬─▶ dense (embed + Qdrant) ── top 15 ─┐
            │                                      ├─▶ RRF (k=60) ─▶ top 20 candidates
            └─▶ BM25                   ── top 15 ─┘          (cross-encoder reranks next)

Both legs use the same metadata filter. If a strict filter leaves too few
candidates, it is relaxed (topic dropped, population kept) and the result says so.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.filters import RetrievalFilters
from app.rag.retrieval.fusion import reciprocal_rank_fusion
from app.rag.retrieval.sparse import SparseRetriever
from app.schemas.clinical import RetrievedChunk

log = get_logger(__name__)


class HybridResult(BaseModel):
    candidates: list[RetrievedChunk]
    filters: RetrievalFilters  # the filter that produced ``candidates``
    filter_relaxed: bool = False
    dense_count: int = 0
    sparse_count: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)


async def _timed[T](coro: Awaitable[T]) -> tuple[T, float]:
    start = time.perf_counter()
    result = await coro
    return result, (time.perf_counter() - start) * 1000


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        *,
        k_dense: int = 15,
        k_sparse: int = 15,
        fused_k: int = 20,
        rrf_k: int = 60,
        min_results: int = 5,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._k_dense = k_dense
        self._k_sparse = k_sparse
        self._fused_k = fused_k
        self._rrf_k = rrf_k
        self._min_results = min_results

    async def search(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        sparse_query: str | None = None,
        relax_filters: bool = True,
    ) -> HybridResult:
        """``sparse_query`` lets the query rewriter give BM25 an expanded keyword
        form while the dense leg keeps the natural-language question."""
        started = time.perf_counter()
        active = filters or RetrievalFilters()
        result = await self._run(query, sparse_query or query, active)

        if (
            relax_filters
            and len(result.candidates) < self._min_results
            and not active.is_empty
            and active.relaxed() != active
        ):
            log.info(
                "retrieval.relaxing_filter",
                found=len(result.candidates),
                strict=active.model_dump(exclude_none=True),
            )
            result = await self._run(query, sparse_query or query, active.relaxed())
            result.filter_relaxed = True

        result.timings_ms["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        log.info(
            "retrieval.hybrid",
            candidates=len(result.candidates),
            dense=result.dense_count,
            sparse=result.sparse_count,
            relaxed=result.filter_relaxed,
            **result.timings_ms,
        )
        return result

    async def _run(self, query: str, sparse_query: str, filters: RetrievalFilters) -> HybridResult:
        (dense_hits, dense_ms), (sparse_hits, sparse_ms) = await asyncio.gather(
            _timed(self._dense.search(query, filters=filters, k=self._k_dense)),
            _timed(self._sparse.asearch(sparse_query, filters=filters, k=self._k_sparse)),
        )
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=self._rrf_k)[: self._fused_k]
        rrf_ms = (time.perf_counter() - t0) * 1000
        return HybridResult(
            candidates=fused,
            filters=filters,
            dense_count=len(dense_hits),
            sparse_count=len(sparse_hits),
            timings_ms={
                "dense_ms": round(dense_ms, 2),
                "bm25_ms": round(sparse_ms, 2),
                "rrf_ms": round(rrf_ms, 3),
            },
        )
