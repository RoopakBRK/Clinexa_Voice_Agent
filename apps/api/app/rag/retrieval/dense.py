"""Dense (vector) retrieval: embed the query, search Qdrant with metadata filters."""

from __future__ import annotations

from app.rag.embeddings import Embedder, aembed_query
from app.rag.retrieval.filters import RetrievalFilters
from app.rag.retrieval.qdrant_store import QdrantChunkStore
from app.schemas.clinical import RetrievedChunk


class DenseRetriever:
    def __init__(self, store: QdrantChunkStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    async def search(
        self, query: str, *, filters: RetrievalFilters | None = None, k: int = 15
    ) -> list[RetrievedChunk]:
        vector = await aembed_query(self._embedder, query)
        return await self._store.search(vector, filters=filters, limit=k)
