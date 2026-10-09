"""WHO guidance for a caller's question, while they are on the line.

    question ─▶ hybrid (dense + BM25, RRF) ─▶ 20 candidates ─▶ cross-encoder ─▶ top 5

The same pipeline the evaluation measures (app/rag/evaluation), behind one call that
never holds a conversation up. The models and the BM25 index take seconds to load, so
they load in the background when the server starts; until they are in, and whenever a
search fails or runs past its deadline, the answer is None and the caller is told the
guidance could not be searched, not kept waiting.
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient

from app.core.config import Settings
from app.core.logging import get_logger
from app.rag.embeddings import SentenceTransformerEmbedder
from app.rag.reranking.cross_encoder import CrossEncoderReranker, Reranker, arerank
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.filters import RetrievalFilters, population_for
from app.rag.retrieval.hybrid import HybridResult, HybridRetriever
from app.rag.retrieval.qdrant_store import QdrantChunkStore, build_client
from app.rag.retrieval.sparse import SparseRetriever
from app.schemas.clinical import RetrievedChunk

log = get_logger(__name__)

Status = Literal["not_started", "loading", "ready", "unavailable"]


class GuidelinesNotIndexedError(RuntimeError):
    """The Qdrant collection the guidance is searched in does not exist yet."""


class GuidelineResult(BaseModel):
    passages: list[RetrievedChunk]
    # The population the search was held to: None when the caller's age is not known.
    population: list[str] | None = None
    filter_relaxed: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)


class Candidates(Protocol):
    """The hybrid retriever, as the search needs it."""

    async def search(
        self, query: str, *, filters: RetrievalFilters | None = None
    ) -> HybridResult: ...


class Pipeline(NamedTuple):
    hybrid: Candidates
    reranker: Reranker | None
    client: AsyncQdrantClient | None = None


Loader = Callable[[], Awaitable[Pipeline]]


class GuidelineSearch:
    def __init__(self, loader: Loader, *, top_k: int = 5, timeout_s: float = 4.0) -> None:
        self._loader = loader
        self._top_k = top_k
        self._timeout_s = timeout_s
        self._pipeline: Pipeline | None = None
        self._loading: asyncio.Task[None] | None = None
        self.status: Status = "not_started"

    def start(self) -> None:
        """Load the index and the models in the background. Safe to call twice."""
        if self._loading is None:
            self.status = "loading"
            self._loading = asyncio.create_task(self._load(), name="guidelines-load")

    async def _load(self) -> None:
        started = time.perf_counter()
        try:
            self._pipeline = await self._loader()
        except Exception as exc:
            self.status = "unavailable"
            log.error("guidelines.unavailable", error=type(exc).__name__, detail=str(exc))
            return
        self.status = "ready"
        log.info("guidelines.ready", load_s=round(time.perf_counter() - started, 1))

    async def search(
        self, query: str, *, age: int | None = None, pregnant: bool | None = None
    ) -> GuidelineResult | None:
        """The passages that best answer ``query``, or None if they could not be searched.

        ``age`` and ``pregnant`` hold the search to guidance written for that person: a
        child is never answered from adult guidance, nor an adult from a child's.
        """
        pipeline = self._pipeline
        if pipeline is None:
            return None
        population = population_for(age, "pregnant" if pregnant else None)
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout_s):
                found = await pipeline.hybrid.search(
                    query, filters=RetrievalFilters(population=population)
                )
                reranked_at = time.perf_counter()
                if pipeline.reranker is not None:
                    passages = await arerank(
                        pipeline.reranker, query, found.candidates, self._top_k
                    )
                else:
                    passages = found.candidates[: self._top_k]
        except Exception as exc:
            # The question is health information: only the kind of failure is logged.
            log.warning("guidelines.search_failed", error=type(exc).__name__)
            return None
        ended = time.perf_counter()
        timings = {
            **found.timings_ms,
            "rerank_ms": round((ended - reranked_at) * 1000, 2),
            "total_ms": round((ended - started) * 1000, 2),
        }
        log.info(
            "guidelines.search",
            passages=len(passages),
            population=population,
            relaxed=found.filter_relaxed,
            **timings,
        )
        return GuidelineResult(
            passages=passages,
            population=population,
            filter_relaxed=found.filter_relaxed,
            timings_ms=timings,
        )

    async def aclose(self) -> None:
        if self._loading is not None and not self._loading.done():
            self._loading.cancel()
            await asyncio.gather(self._loading, return_exceptions=True)
        if self._pipeline is not None and self._pipeline.client is not None:
            await self._pipeline.client.close()


async def load_pipeline(settings: Settings) -> Pipeline:
    """Open the index and load both models, running each once so the first search is warm."""
    client = build_client(settings)
    try:
        if not await client.collection_exists(settings.qdrant_collection):
            raise GuidelinesNotIndexedError(
                f"There is no '{settings.qdrant_collection}' collection. Run: make index"
            )

        def models() -> tuple[SparseRetriever, SentenceTransformerEmbedder, CrossEncoderReranker]:
            # Imported here: it brings the PDF extraction stack with it, which a server
            # that only reads the finished chunks should not pay for at start-up.
            from app.rag.ingestion.pipeline import load_chunks

            sparse = SparseRetriever(load_chunks(_chunks_path(settings)))
            embedder = SentenceTransformerEmbedder(
                settings.embedding_model,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
                query_instruction=settings.embedding_query_instruction,
            )
            embedder.embed_query("fever")
            reranker = CrossEncoderReranker(
                settings.reranker_model,
                device=settings.embedding_device,
                batch_size=settings.reranker_batch_size,
            )
            reranker.score("fever", ["Fever in a child under five"])
            return sparse, embedder, reranker

        sparse, embedder, reranker = await asyncio.to_thread(models)
    except BaseException:
        await client.close()
        raise
    store = QdrantChunkStore(client, settings.qdrant_collection, settings.embedding_model)
    return Pipeline(HybridRetriever(DenseRetriever(store, embedder), sparse), reranker, client)


def _chunks_path(settings: Settings) -> Path:
    return settings.data_dir / "processed" / "chunks.jsonl"


def build_guideline_search(settings: Settings) -> GuidelineSearch | None:
    """The guidance search for this server, or None where there is nothing to search with."""
    if not settings.guidelines_retrieval:
        return None
    if not _chunks_path(settings).exists():
        log.warning("guidelines.not_configured", hint="run `make ingest`, then `make index`")
        return None
    if importlib.util.find_spec("sentence_transformers") is None:
        log.warning("guidelines.not_configured", hint="sentence-transformers is not installed")
        return None
    return GuidelineSearch(
        lambda: load_pipeline(settings),
        top_k=settings.reranker_top_k,
        timeout_s=settings.guidelines_timeout_s,
    )
