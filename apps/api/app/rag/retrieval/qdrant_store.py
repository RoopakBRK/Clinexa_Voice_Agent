"""Qdrant-backed chunk store: collection lifecycle, upserts, dense search.

Only this module talks to Qdrant, so agents reach the knowledge base through
tools/retrieval services rather than the database directly.
"""

from __future__ import annotations

import hashlib
import uuid
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.core.config import Settings
from app.core.logging import get_logger
from app.rag.models import Chunk
from app.rag.retrieval.filters import RetrievalFilters
from app.schemas.clinical import DocumentMetadata, RetrievalScores, RetrievedChunk

log = get_logger(__name__)

# Stable namespace: a chunk_id always maps to the same point id, so re-indexing is idempotent.
_POINT_NAMESPACE = uuid.UUID("6f0d3c1e-5b7a-4c1e-9a55-0c11e7a00001")

# Payload fields used in filters, and how Qdrant should index them.
_PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "population": models.PayloadSchemaType.KEYWORD,
    "topics": models.PayloadSchemaType.KEYWORD,
    "document_type": models.PayloadSchemaType.KEYWORD,
    "document_id": models.PayloadSchemaType.KEYWORD,
    "chunk_type": models.PayloadSchemaType.KEYWORD,
    "retrievable": models.PayloadSchemaType.BOOL,
    "page_number": models.PayloadSchemaType.INTEGER,
}

_UPSERT_BATCH = 64
_SCROLL_PAGE = 512


class CollectionMismatchError(RuntimeError):
    """The existing collection is incompatible with the embedder (e.g. vector size)."""


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def content_hash(embedding_model: str, embedding_text: str) -> str:
    """Changes whenever the model or the exact embedded text changes."""
    return hashlib.sha1(f"{embedding_model}\n{embedding_text}".encode()).hexdigest()


def build_client(settings: Settings, *, force_local: bool = False) -> AsyncQdrantClient:
    if settings.qdrant_url and not force_local:
        return AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            timeout=settings.qdrant_timeout_s,
        )
    settings.qdrant_local_path.mkdir(parents=True, exist_ok=True)
    return AsyncQdrantClient(path=str(settings.qdrant_local_path))


class QdrantChunkStore:
    def __init__(self, client: AsyncQdrantClient, collection: str, embedding_model: str) -> None:
        self.client = client
        self.collection = collection
        self.embedding_model = embedding_model

    # --- collection lifecycle ------------------------------------------------

    async def ensure_collection(self, dimension: int, *, recreate: bool = False) -> None:
        exists = await self.client.collection_exists(self.collection)
        if exists and recreate:
            await self.client.delete_collection(self.collection)
            exists = False
        if exists:
            info = await self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            size = vectors.size if isinstance(vectors, models.VectorParams) else None
            if size != dimension:
                raise CollectionMismatchError(
                    f"Collection '{self.collection}' has vector size {size}, embedder produces "
                    f"{dimension}. Re-run with --recreate (this rebuilds the whole index)."
                )
        else:
            await self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
            log.info("qdrant.collection_created", collection=self.collection, dimension=dimension)
        with warnings.catch_warnings():
            # The embedded local index has no payload indexes (filters still work, unindexed).
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            for field, schema in _PAYLOAD_INDEXES.items():
                await self.client.create_payload_index(self.collection, field, field_schema=schema)

    async def count(self) -> int:
        return (await self.client.count(self.collection, exact=True)).count

    # --- writes ---------------------------------------------------------------

    async def existing_hashes(self) -> dict[str, str]:
        """point id -> content hash, for every point currently in the collection."""
        found: dict[str, str] = {}
        offset: Any = None
        while True:
            points, offset = await self.client.scroll(
                self.collection,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=["content_hash"],
                with_vectors=False,
            )
            for p in points:
                found[str(p.id)] = str((p.payload or {}).get("content_hash", ""))
            if offset is None:
                return found

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        points = [
            models.PointStruct(
                id=point_id(chunk.chunk_id),
                vector=list(vector),
                payload={
                    **chunk.metadata.model_dump(mode="json"),
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "content_hash": content_hash(self.embedding_model, chunk.embedding_text()),
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), _UPSERT_BATCH):
            await self.client.upsert(self.collection, points=points[start : start + _UPSERT_BATCH])

    async def delete(self, point_ids: Iterable[str]) -> None:
        ids = list(point_ids)
        for start in range(0, len(ids), _UPSERT_BATCH):
            await self.client.delete(
                self.collection,
                points_selector=models.PointIdsList(points=ids[start : start + _UPSERT_BATCH]),
            )

    # --- reads ------------------------------------------------------------------

    async def search(
        self,
        vector: Sequence[float],
        *,
        filters: RetrievalFilters | None = None,
        limit: int = 15,
    ) -> list[RetrievedChunk]:
        response = await self.client.query_points(
            self.collection,
            query=list(vector),
            query_filter=(filters or RetrievalFilters()).to_qdrant(),
            limit=limit,
            with_payload=True,
        )
        return [_to_retrieved(p) for p in response.points]


def _to_retrieved(point: models.ScoredPoint) -> RetrievedChunk:
    payload = point.payload or {}
    return RetrievedChunk(
        chunk_id=str(payload["chunk_id"]),
        text=str(payload["text"]),
        metadata=DocumentMetadata.model_validate(payload),
        scores=RetrievalScores(dense_score=float(point.score)),
    )
