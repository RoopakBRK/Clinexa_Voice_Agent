"""BM25, reciprocal rank fusion, clinical filters and the hybrid retriever."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.filters import (
    RetrievalFilters,
    filters_from_clinical,
    is_pregnant,
    population_for,
)
from app.rag.retrieval.fusion import reciprocal_rank_fusion
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.qdrant_store import QdrantChunkStore
from app.rag.retrieval.sparse import SparseRetriever
from app.schemas.clinical import (
    ChunkType,
    ClinicalIntake,
    DocumentMetadata,
    RetrievalScores,
    RetrievedChunk,
)
from tests.test_retrieval_store import FakeEmbedder, make_chunk

# --- helpers ----------------------------------------------------------------


def hit(cid: str, *, dense: float | None = None, bm25: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        text=f"text {cid}",
        metadata=DocumentMetadata(source="s", document_title="t", document_type="d"),
        scores=RetrievalScores(dense_score=dense, bm25_score=bm25),
    )


CORPUS = [
    make_chunk(
        "c1", "child with cough and fast breathing suggests pneumonia", topics=["respiratory"]
    ),
    make_chunk(
        "c2", "oral rehydration solution for diarrhoea and dehydration", topics=["gastrointestinal"]
    ),
    make_chunk(
        "c3",
        "dose of artemether lumefantrine for uncomplicated malaria",
        population="all",
        topics=["malaria"],
        doc="who-malaria",
    ),
    make_chunk(
        "c4",
        "adult with coughing for two weeks should be tested for tuberculosis",
        population="adult",
        topics=["respiratory", "infectious_disease"],
        chunk_type=ChunkType.WARNING,
        doc="apc",
    ),
    make_chunk(
        "c5", "rash and itching of the skin", population="adult", topics=["skin"], doc="apc"
    ),
    make_chunk("c6", "cough references acknowledgements", retrievable=False),
]


# --- RRF ------------------------------------------------------------------------


def test_rrf_matches_hand_computed_scores() -> None:
    dense = [hit("A", dense=0.9), hit("B", dense=0.8), hit("C", dense=0.7)]
    sparse = [hit("B", bm25=9), hit("D", bm25=7), hit("A", bm25=5)]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)

    expected = {
        "A": 1 / 61 + 1 / 63,  # dense #1, bm25 #3
        "B": 1 / 62 + 1 / 61,  # dense #2, bm25 #1
        "C": 1 / 63,
        "D": 1 / 62,
    }
    assert {h.chunk_id: h.scores.rrf_score for h in fused} == pytest.approx(expected)
    assert [h.chunk_id for h in fused] == ["B", "A", "D", "C"]


def test_rrf_keeps_stage_scores_and_ranks() -> None:
    fused = {
        h.chunk_id: h
        for h in reciprocal_rank_fusion(
            [hit("A", dense=0.9), hit("B", dense=0.8)], [hit("B", bm25=12.5)], k=60
        )
    }
    both, dense_only = fused["B"].scores, fused["A"].scores
    assert (both.dense_score, both.dense_rank, both.bm25_score, both.bm25_rank) == (0.8, 2, 12.5, 1)
    assert (dense_only.dense_rank, dense_only.bm25_score, dense_only.bm25_rank) == (1, None, None)


def test_rrf_rewards_consensus_over_a_single_top_rank() -> None:
    # B is #2 in both lists; A is #1 in only one. Agreement should win.
    fused = reciprocal_rank_fusion([hit("A"), hit("B")], [hit("X"), hit("B")], k=60)
    assert fused[0].chunk_id == "B"


def test_rrf_weights_shift_the_balance() -> None:
    dense, sparse = [hit("D1")], [hit("S1")]
    assert reciprocal_rank_fusion(dense, sparse, sparse_weight=2.0)[0].chunk_id == "S1"
    assert reciprocal_rank_fusion(dense, sparse, dense_weight=2.0)[0].chunk_id == "D1"


def test_rrf_is_deterministic_on_ties_and_handles_empty_inputs() -> None:
    # Equal scores and ranks: ordered by chunk id.
    assert [h.chunk_id for h in reciprocal_rank_fusion([hit("b")], [hit("a")])] == ["a", "b"]
    assert reciprocal_rank_fusion([], []) == []
    assert [h.chunk_id for h in reciprocal_rank_fusion([hit("only")], [])] == ["only"]
    with pytest.raises(ValueError, match="k must be"):
        reciprocal_rank_fusion([], [], k=-1)


# --- BM25 -----------------------------------------------------------------------


@pytest.fixture
def sparse() -> SparseRetriever:
    return SparseRetriever(CORPUS)


def test_bm25_exact_clinical_term_ranks_first(sparse: SparseRetriever) -> None:
    hits = sparse.search("artemether lumefantrine dose")
    assert hits[0].chunk_id == "c3"
    assert (
        hits[0].scores.bm25_rank == 1
        and hits[0].scores.bm25_score
        and hits[0].scores.bm25_score > 0
    )


def test_bm25_stems_and_drops_stopwords(sparse: SparseRetriever) -> None:
    # "coughing"/"cough" share a stem; "the", "of" are ignored.
    ids = [h.chunk_id for h in sparse.search("the coughing of a child")]
    assert {"c1", "c4"} <= set(ids)
    assert sparse.search("the and of") == []
    assert sparse.search("zzzzunknownterm") == []
    assert sparse.search("cough", k=0) == []


def test_bm25_never_returns_non_retrievable_chunks(sparse: SparseRetriever) -> None:
    assert "c6" not in [h.chunk_id for h in sparse.search("cough references acknowledgements")]
    assert len(sparse) == 5


def test_bm25_section_path_is_searchable() -> None:
    chunk = make_chunk("h1", "give paracetamol and rest", topics=["neurological"])
    chunk.metadata.heading_path = ["Headache", "Management"]
    other = make_chunk("h2", "completely unrelated text about ships")
    hits = SparseRetriever([chunk, other]).search("headache")
    assert [h.chunk_id for h in hits] == ["h1"]  # matched via the heading, not the body


def test_bm25_filters_restrict_results(sparse: SparseRetriever) -> None:
    adult = sparse.search("cough", filters=RetrievalFilters(population=["adult"]))
    assert [h.chunk_id for h in adult] == ["c4"]
    warn = sparse.search("cough", filters=RetrievalFilters(chunk_types=[ChunkType.WARNING]))
    assert [h.chunk_id for h in warn] == ["c4"]
    assert sparse.search("cough", filters=RetrievalFilters(population=["pregnancy"])) == []
    # ranks restart at 1 within the filtered result list
    assert adult[0].scores.bm25_rank == 1


def test_bm25_results_are_ordered_and_deterministic(sparse: SparseRetriever) -> None:
    first = sparse.search("cough")
    assert first == sparse.search("cough")
    scores = [h.scores.bm25_score or 0 for h in first]
    assert scores == sorted(scores, reverse=True)
    assert [h.scores.bm25_rank for h in first] == list(range(1, len(first) + 1))
    assert len(sparse.search("cough", k=1)) == 1


def test_bm25_filter_masks_are_cached(sparse: SparseRetriever) -> None:
    f = RetrievalFilters(population=["adult"])
    sparse.search("cough", filters=f)
    sparse.search("rash", filters=f)
    assert len(sparse._masks) == 1


def test_bm25_requires_a_retrievable_corpus() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SparseRetriever([make_chunk("x", "text", retrievable=False)])


# --- the Python filter and the Qdrant filter must select identical chunks ------------


FILTER_CASES = [
    RetrievalFilters(),
    RetrievalFilters(population=["adult"]),
    RetrievalFilters(population=["child", "all"]),
    RetrievalFilters(topics=["respiratory"]),
    RetrievalFilters(topics=["respiratory", "skin"], population=["adult"]),
    RetrievalFilters(chunk_types=[ChunkType.WARNING]),
    RetrievalFilters(document_ids=["apc"], topics=["infectious_disease"]),
    RetrievalFilters(document_types=["pocket_book"], population=["all"]),
    RetrievalFilters(population=["pregnancy"]),
]


@pytest.fixture
async def indexed() -> AsyncIterator[tuple[QdrantChunkStore, FakeEmbedder]]:
    client = AsyncQdrantClient(":memory:")
    store = QdrantChunkStore(client, "hybrid-test", "fake-bow")
    embedder = FakeEmbedder()
    # Index the non-retrievable chunk too, to prove the mandatory filter excludes it.
    await store.ensure_collection(embedder.dimension)
    await store.upsert(CORPUS, embedder.embed_documents([c.embedding_text() for c in CORPUS]))
    yield store, embedder
    await client.close()


@pytest.mark.parametrize("filters", FILTER_CASES)
async def test_python_and_qdrant_filters_agree(
    indexed: tuple[QdrantChunkStore, FakeEmbedder], filters: RetrievalFilters
) -> None:
    store, _ = indexed
    points, _ = await store.client.scroll(
        "hybrid-test", scroll_filter=filters.to_qdrant(), limit=100, with_payload=["chunk_id"]
    )
    via_qdrant = {str(p.payload["chunk_id"]) for p in points if p.payload}
    via_python = {c.chunk_id for c in CORPUS if filters.matches(c.metadata)}
    assert via_qdrant == via_python


# --- clinical filter builder ------------------------------------------------------------


@pytest.mark.parametrize(
    ("age", "pregnancy", "expected"),
    [
        (None, None, None),
        (3, None, ["child", "all"]),
        (17, None, ["child", "all"]),
        (18, None, ["adult", "all"]),
        (45, "no", ["adult", "all"]),
        (30, "pregnant, 20 weeks", ["adult", "all", "pregnancy"]),
        (30, "not pregnant", ["adult", "all"]),
    ],
)
def test_population_for(age: int | None, pregnancy: str | None, expected: list[str] | None) -> None:
    assert population_for(age, pregnancy) == expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("pregnant", True),
        ("Yes", True),
        ("13 weeks pregnant", True),
        ("no", False),
        ("not pregnant", False),
        ("isn't pregnant", False),
        (None, False),
        ("", False),
    ],
)
def test_is_pregnant(status: str | None, expected: bool) -> None:
    assert is_pregnant(status) is expected


def test_filters_from_clinical_state() -> None:
    intake = ClinicalIntake(
        chief_complaint="cough", symptoms=["wheezing"], associated_symptoms=["fever"], age=34
    )
    f = filters_from_clinical(intake)
    assert f.population == ["adult", "all"]
    assert f.topics and f.topics[0] == "respiratory"

    assert filters_from_clinical(intake, include_topics=False).topics is None
    # Nothing known yet: no filter at all (and no invented restriction).
    assert filters_from_clinical(ClinicalIntake()).is_empty


# --- hybrid ---------------------------------------------------------------------------


@pytest.fixture
async def hybrid(
    indexed: tuple[QdrantChunkStore, FakeEmbedder],
) -> HybridRetriever:
    store, embedder = indexed
    return HybridRetriever(DenseRetriever(store, embedder), SparseRetriever(CORPUS), min_results=2)


async def test_hybrid_fuses_both_legs_with_all_scores(hybrid: HybridRetriever) -> None:
    result = await hybrid.search("child cough with fast breathing")
    top = result.candidates[0]
    assert top.chunk_id == "c1"
    s = top.scores
    assert s.dense_score and s.bm25_score and s.rrf_score
    assert s.dense_rank == 1 and s.bm25_rank == 1
    assert set(result.timings_ms) == {"dense_ms", "bm25_ms", "rrf_ms", "total_ms"}
    assert result.dense_count > 0 and result.sparse_count > 0 and not result.filter_relaxed
    rrf = [c.scores.rrf_score or 0 for c in result.candidates]
    assert rrf == sorted(rrf, reverse=True)


async def test_hybrid_surfaces_bm25_only_hits(hybrid: HybridRetriever) -> None:
    result = await hybrid.search("artemether lumefantrine")
    by_id = {c.chunk_id: c for c in result.candidates}
    assert result.candidates[0].chunk_id == "c3"
    assert by_id["c3"].scores.bm25_rank == 1


async def test_hybrid_applies_filters_to_both_legs(hybrid: HybridRetriever) -> None:
    result = await hybrid.search("cough", filters=RetrievalFilters(population=["adult"]))
    assert {c.chunk_id for c in result.candidates} <= {"c4", "c5"}
    assert all(c.metadata.population == "adult" for c in result.candidates)


async def test_hybrid_relaxes_topic_but_keeps_population(hybrid: HybridRetriever) -> None:
    strict = RetrievalFilters(population=["adult"], topics=["malaria"])  # matches nothing
    result = await hybrid.search("cough", filters=strict)
    assert result.filter_relaxed
    assert result.filters == RetrievalFilters(
        population=["adult"]
    )  # topic dropped, population kept
    assert result.candidates and all(c.metadata.population == "adult" for c in result.candidates)


async def test_hybrid_does_not_relax_when_disabled_or_unnecessary(hybrid: HybridRetriever) -> None:
    strict = RetrievalFilters(population=["adult"], topics=["malaria"])
    off = await hybrid.search("cough", filters=strict, relax_filters=False)
    assert not off.filter_relaxed and off.candidates == []

    enough = await hybrid.search("cough", filters=RetrievalFilters(topics=["respiratory"]))
    assert not enough.filter_relaxed  # c1 and c4 satisfy the strict filter

    population_only = await hybrid.search(
        "cough", filters=RetrievalFilters(population=["pregnancy"])
    )
    assert (
        not population_only.filter_relaxed and population_only.candidates == []
    )  # nothing to relax


async def test_hybrid_can_use_a_different_query_for_bm25(hybrid: HybridRetriever) -> None:
    result = await hybrid.search("tummy trouble", sparse_query="diarrhoea dehydration")
    bm25_hits = [c for c in result.candidates if c.scores.bm25_rank]
    assert [c.chunk_id for c in bm25_hits] == ["c2"]  # BM25 saw the rewritten query


async def test_hybrid_respects_fused_k(indexed: tuple[QdrantChunkStore, FakeEmbedder]) -> None:
    store, embedder = indexed
    small = HybridRetriever(
        DenseRetriever(store, embedder), SparseRetriever(CORPUS), fused_k=2, k_dense=5, k_sparse=5
    )
    assert len((await small.search("cough")).candidates) == 2
