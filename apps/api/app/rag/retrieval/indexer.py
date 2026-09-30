"""Chunks → embeddings → Qdrant, incrementally.

Re-running only embeds chunks whose text (or the embedding model) changed, and
removes points for chunks that no longer exist, so the index always mirrors
``chunks.jsonl`` without re-embedding thousands of unchanged passages.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from pydantic import BaseModel

from app.core.logging import get_logger
from app.rag.embeddings import Embedder, aembed_documents
from app.rag.models import Chunk
from app.rag.retrieval.qdrant_store import QdrantChunkStore, content_hash, point_id

log = get_logger(__name__)

ProgressFn = Callable[[int, int], None]


class IndexResult(BaseModel):
    indexable: int  # retrievable chunks that should be in the index
    embedded: int  # newly embedded and upserted this run
    unchanged: int  # already present with identical content
    deleted: int  # stale points removed
    total_in_collection: int
    duration_s: float


async def index_chunks(
    store: QdrantChunkStore,
    embedder: Embedder,
    chunks: Sequence[Chunk],
    *,
    recreate: bool = False,
    embed_batch: int = 128,
    progress: ProgressFn | None = None,
) -> IndexResult:
    started = time.perf_counter()
    wanted = [c for c in chunks if c.metadata.retrievable]
    await store.ensure_collection(embedder.dimension, recreate=recreate)

    existing = await store.existing_hashes()
    wanted_ids = {point_id(c.chunk_id) for c in wanted}
    stale = [pid for pid in existing if pid not in wanted_ids]
    todo = [
        c
        for c in wanted
        if existing.get(point_id(c.chunk_id))
        != content_hash(store.embedding_model, c.embedding_text())
    ]

    for start in range(0, len(todo), embed_batch):
        batch = todo[start : start + embed_batch]
        vectors = await aembed_documents(embedder, [c.embedding_text() for c in batch])
        await store.upsert(batch, vectors)
        if progress:
            progress(min(start + embed_batch, len(todo)), len(todo))
    if stale:
        await store.delete(stale)

    result = IndexResult(
        indexable=len(wanted),
        embedded=len(todo),
        unchanged=len(wanted) - len(todo),
        deleted=len(stale),
        total_in_collection=await store.count(),
        duration_s=round(time.perf_counter() - started, 1),
    )
    log.info("index.done", **result.model_dump())
    return result
