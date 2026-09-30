"""Embedder, filters, Qdrant store and indexer (in-memory Qdrant, fake embedder)."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient, models

from app.rag.embeddings import SentenceTransformerEmbedder
from app.rag.models import Chunk
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.filters import RetrievalFilters
from app.rag.retrieval.indexer import index_chunks
from app.rag.retrieval.qdrant_store import (
    CollectionMismatchError,
    QdrantChunkStore,
    content_hash,
    point_id,
)
from app.schemas.clinical import ChunkType, DocumentMetadata

DIM = 64


class FakeEmbedder:
    """Hashed bag-of-words vectors: texts sharing words are closer (cosine)."""

    model_name = "fake-bow"
    dimension = DIM

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for word in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    population: str = "child",
    topics: list[str] | None = None,
    chunk_type: ChunkType = ChunkType.TEXT,
    retrievable: bool = True,
    doc: str = "doc-a",
) -> Chunk:
    topics = topics or ["respiratory"]
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        token_count=len(text.split()),
        metadata=DocumentMetadata(
            source=f"{doc}.pdf",
            document_id=doc,
            document_title="Test Book",
            document_type="pocket_book",
            section="1 Section",
            heading_path=["1 Section"],
            page_number=7,
            page_end=7,
            topic=topics[0],
            topics=topics,
            population=population,
            chunk_type=chunk_type,
            retrievable=retrievable,
        ),
    )


CHUNKS = [
    make_chunk("a:1", "fast breathing and cough in a child with pneumonia", topics=["respiratory"]),
    make_chunk(
        "a:2",
        "give oral rehydration solution for diarrhoea and dehydration",
        topics=["gastrointestinal"],
    ),
    make_chunk(
        "a:3",
        "adult with cough and fever needs urgent referral",
        population="adult",
        topics=["respiratory", "infectious_disease"],
        chunk_type=ChunkType.WARNING,
        doc="doc-b",
    ),
    make_chunk("a:4", "references and acknowledgements list", retrievable=False),
]


@pytest.fixture
async def store() -> AsyncIterator[QdrantChunkStore]:
    client = AsyncQdrantClient(":memory:")
    yield QdrantChunkStore(client, "test", "fake-bow")
    await client.close()


# --- embedder -------------------------------------------------------------


class StubModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, sentences: Any, **kwargs: Any) -> Any:
        self.calls.append({"input": sentences, **kwargs})
        if isinstance(sentences, str):
            return np.array([1.0, 0.0, 0.0])
        return np.array([[0.0, 1.0, 0.0]] * len(sentences))


def test_embedder_applies_query_instruction_to_queries_only() -> None:
    model = StubModel()
    embedder = SentenceTransformerEmbedder(
        "bge", batch_size=8, query_instruction="Q: ", model_factory=lambda _name, _device: model
    )
    assert embedder.dimension == 3
    assert embedder.embed_query("cough") == [1.0, 0.0, 0.0]
    assert embedder.embed_documents(["a passage", "another"]) == [[0.0, 1.0, 0.0]] * 2
    assert embedder.embed_documents([]) == []

    query_call, doc_call = model.calls
    assert query_call["input"] == "Q: cough"  # instruction on queries
    assert doc_call["input"] == ["a passage", "another"]  # none on passages
    assert query_call["normalize_embeddings"] and doc_call["normalize_embeddings"]
    assert doc_call["batch_size"] == 8


# --- filters ----------------------------------------------------------------


def test_filters_always_require_retrievable() -> None:
    f = RetrievalFilters().to_qdrant()
    assert isinstance(f.must, list) and len(f.must) == 1
    assert f.must[0].key == "retrievable"  # type: ignore[union-attr]


def test_filters_translate_each_field_to_match_any() -> None:
    f = RetrievalFilters(
        population=["adult", "all"], topics=["respiratory"], chunk_types=[ChunkType.WARNING]
    ).to_qdrant()
    conditions = {c.key: c.match for c in f.must}  # type: ignore[union-attr]
    assert conditions["population"].any == ["adult", "all"]  # type: ignore[union-attr]
    assert conditions["topics"].any == ["respiratory"]  # type: ignore[union-attr]
    assert conditions["chunk_type"].any == ["warning"]  # type: ignore[union-attr]


def test_relaxed_filter_keeps_only_population() -> None:
    strict = RetrievalFilters(population=["adult"], topics=["skin"], document_ids=["x"])
    relaxed = strict.relaxed()
    assert (
        relaxed.population == ["adult"] and relaxed.topics is None and relaxed.document_ids is None
    )
    assert RetrievalFilters().is_empty and not strict.is_empty


# --- ids and hashes -----------------------------------------------------------


def test_point_ids_are_deterministic_uuids() -> None:
    assert point_id("who:p0001:abc") == point_id("who:p0001:abc")
    assert point_id("who:p0001:abc") != point_id("who:p0001:abd")
    assert re.fullmatch(r"[0-9a-f-]{36}", point_id("x"))


def test_content_hash_depends_on_model_and_text() -> None:
    base = content_hash("m1", "text")
    assert base == content_hash("m1", "text")
    assert base != content_hash("m2", "text") and base != content_hash("m1", "text!")


# --- store + indexer ------------------------------------------------------------


async def test_indexer_skips_non_retrievable_and_is_idempotent(store: QdrantChunkStore) -> None:
    embedder = FakeEmbedder()
    first = await index_chunks(store, embedder, CHUNKS)
    assert (first.indexable, first.embedded, first.unchanged, first.total_in_collection) == (
        3,
        3,
        0,
        3,
    )

    embedder.embedded.clear()
    second = await index_chunks(store, embedder, CHUNKS)
    assert (second.embedded, second.unchanged, second.deleted) == (0, 3, 0)
    assert embedder.embedded == []  # nothing re-embedded


async def test_indexer_reembeds_changed_and_removes_stale(store: QdrantChunkStore) -> None:
    embedder = FakeEmbedder()
    await index_chunks(store, embedder, CHUNKS)

    changed = [make_chunk("a:1", "completely new text about asthma"), *CHUNKS[2:]]  # a:2 dropped
    embedder.embedded.clear()
    result = await index_chunks(store, embedder, changed)
    assert (result.embedded, result.deleted, result.total_in_collection) == (1, 1, 2)
    assert len(embedder.embedded) == 1 and "asthma" in embedder.embedded[0]


async def test_recreate_rebuilds_everything(store: QdrantChunkStore) -> None:
    embedder = FakeEmbedder()
    await index_chunks(store, embedder, CHUNKS)
    result = await index_chunks(store, embedder, CHUNKS, recreate=True)
    assert result.embedded == 3


async def test_vector_size_mismatch_is_refused(store: QdrantChunkStore) -> None:
    await store.ensure_collection(DIM)
    with pytest.raises(CollectionMismatchError, match="--recreate"):
        await store.ensure_collection(384)


async def test_dense_search_ranks_and_round_trips_metadata(store: QdrantChunkStore) -> None:
    embedder = FakeEmbedder()
    await index_chunks(store, embedder, CHUNKS)
    hits = await DenseRetriever(store, embedder).search("child with cough and fast breathing", k=3)

    assert hits[0].chunk_id == "a:1"
    assert hits[0].scores.dense_score is not None
    assert hits[0].scores.dense_score >= hits[1].scores.dense_score
    top = hits[0].metadata
    assert (top.document_id, top.population, top.page_number, top.heading_path) == (
        "doc-a",
        "child",
        7,
        ["1 Section"],
    )
    assert top.chunk_type is ChunkType.TEXT and top.topics == ["respiratory"]
    assert hits[0].text.startswith("fast breathing")


async def test_search_applies_metadata_filters(store: QdrantChunkStore) -> None:
    embedder = FakeEmbedder()
    await index_chunks(store, embedder, CHUNKS)
    retriever = DenseRetriever(store, embedder)

    adult = await retriever.search("cough", filters=RetrievalFilters(population=["adult"]))
    assert [h.chunk_id for h in adult] == ["a:3"]

    warnings = await retriever.search(
        "cough", filters=RetrievalFilters(chunk_types=[ChunkType.WARNING])
    )
    assert [h.chunk_id for h in warnings] == ["a:3"]

    gi = await retriever.search("cough", filters=RetrievalFilters(topics=["gastrointestinal"]))
    assert [h.chunk_id for h in gi] == ["a:2"]  # topic filter beats semantic similarity

    doc_b = await retriever.search("cough", filters=RetrievalFilters(document_ids=["doc-b"]))
    assert [h.chunk_id for h in doc_b] == ["a:3"]

    none = await retriever.search("cough", filters=RetrievalFilters(population=["pregnancy"]))
    assert none == []


async def test_non_retrievable_points_are_never_returned(store: QdrantChunkStore) -> None:
    embedder = FakeEmbedder()
    await store.ensure_collection(DIM)
    hidden = make_chunk("h:1", "references and acknowledgements", retrievable=False)
    await store.upsert([hidden], embedder.embed_documents([hidden.embedding_text()]))
    assert await store.count() == 1

    hits = await DenseRetriever(store, embedder).search("references and acknowledgements")
    assert hits == []  # present in the collection, excluded by the mandatory filter


async def test_upsert_length_mismatch_raises(store: QdrantChunkStore) -> None:
    await store.ensure_collection(DIM)
    with pytest.raises(ValueError, match="same length"):
        await store.upsert(CHUNKS[:2], [[0.0] * DIM])


def test_payload_index_schema_covers_filter_fields() -> None:
    from app.rag.retrieval.qdrant_store import _PAYLOAD_INDEXES

    used = {
        c.key
        for c in RetrievalFilters(  # type: ignore[union-attr]
            population=["a"],
            topics=["b"],
            document_types=["c"],
            document_ids=["d"],
            chunk_types=[ChunkType.TEXT],
        )
        .to_qdrant()
        .must
    }
    assert used <= set(_PAYLOAD_INDEXES)
    assert _PAYLOAD_INDEXES["retrievable"] is models.PayloadSchemaType.BOOL
