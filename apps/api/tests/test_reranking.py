"""Cross-encoder reranking and extractive context compression."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from app.rag.reranking.compression import (
    OMISSION,
    compress_chunk,
    compress_chunks,
    split_units,
)
from app.rag.reranking.cross_encoder import (
    CrossEncoderReranker,
    arerank,
    passage_for,
    rerank_with,
)
from app.schemas.clinical import DocumentMetadata, RetrievalScores, RetrievedChunk
from tests.test_retrieval_store import FakeEmbedder


def chunk(cid: str, text: str, path: list[str] | None = None, **scores: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        text=text,
        metadata=DocumentMetadata(
            source="s", document_title="t", document_type="d", heading_path=path or []
        ),
        scores=RetrievalScores(**scores),
    )


class OverlapReranker:
    """Scores a passage by how many of the query's words it contains."""

    model_name = "overlap"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        words = {w for w in query.lower().split() if len(w) > 2}
        return [float(sum(w in p.lower() for w in words)) for p in passages]


# --- reranking ------------------------------------------------------------------


def test_rerank_orders_by_score_and_keeps_earlier_stage_scores() -> None:
    cands = [
        chunk("a", "unrelated text", rrf_score=0.03, dense_score=0.9),
        chunk("b", "cough and wheeze in children", rrf_score=0.02, bm25_score=4.0),
        chunk("c", "cough only", rrf_score=0.01),
    ]
    out = rerank_with(OverlapReranker(), "cough wheeze children", cands)
    assert [c.chunk_id for c in out] == ["b", "c", "a"]
    assert [c.scores.reranker_rank for c in out] == [1, 2, 3]
    assert out[0].scores.reranker_score == 3.0
    # Earlier-stage scores survive for observability.
    assert (out[0].scores.rrf_score, out[0].scores.bm25_score) == (0.02, 4.0)
    assert out[2].scores.dense_score == 0.9
    # Inputs are not mutated.
    assert cands[0].scores.reranker_score is None


def test_rerank_top_k_ties_and_empty() -> None:
    cands = [chunk("x", "same"), chunk("y", "same"), chunk("z", "same")]
    out = rerank_with(OverlapReranker(), "unrelated", cands, top_k=2)
    assert [c.chunk_id for c in out] == ["x", "y"]  # tie -> incoming (fused) order
    assert rerank_with(OverlapReranker(), "q", []) == []


def test_reranker_reads_section_path_with_the_chunk() -> None:
    c = chunk("a", "Give paracetamol.", path=["Headache", "Management"])
    assert passage_for(c) == "Headache > Management\nGive paracetamol."
    assert passage_for(chunk("b", "plain")) == "plain"
    reranker = OverlapReranker()
    rerank_with(reranker, "headache", [c])
    assert "Headache > Management" in reranker.calls[0][1][0]


async def test_arerank_runs_off_the_event_loop() -> None:
    out = await arerank(
        OverlapReranker(), "cough", [chunk("a", "cough"), chunk("b", "nothing")], top_k=1
    )
    assert [c.chunk_id for c in out] == ["a"]


class StubCrossEncoder:
    def __init__(self) -> None:
        self.inputs: list[Any] = []
        self.kwargs: dict[str, Any] = {}

    def predict(self, pairs: Any, **kwargs: Any) -> Any:
        self.inputs, self.kwargs = pairs, kwargs
        return np.array([0.1, 0.9][: len(pairs)])


def test_cross_encoder_wrapper_builds_query_passage_pairs() -> None:
    stub = StubCrossEncoder()
    seen: dict[str, Any] = {}

    def factory(name: str, device: str, max_length: int) -> Any:
        seen.update(name=name, device=device, max_length=max_length)
        return stub

    reranker = CrossEncoderReranker(
        "m", device="cpu", batch_size=4, max_length=256, model_factory=factory
    )
    assert reranker.score("q", ["p1", "p2"]) == pytest.approx([0.1, 0.9])
    assert stub.inputs == [("q", "p1"), ("q", "p2")]
    assert stub.kwargs["batch_size"] == 4
    assert seen == {"name": "m", "device": "cpu", "max_length": 256}
    assert reranker.score("q", []) == []
    out = reranker.rerank("q", [chunk("a", "p1"), chunk("b", "p2")])
    assert [c.chunk_id for c in out] == ["b", "a"]


# --- compression -------------------------------------------------------------------


LONG = (
    "Cough is common in children. "
    "Fast breathing suggests pneumonia and needs antibiotics. "
    "Zinc supplements help diarrhoea recovery. "
    "Keep the child warm and dry at all times. "
    "Do not give cough syrup to children under two years. "
    "Offer extra fluids and continue feeding. "
    "Refer urgently if the child cannot drink or is very drowsy. "
    "Vitamin A is given in measles campaigns."
)


def test_split_units_sentences_bullets_and_table_rows() -> None:
    text = "Intro sentence. Second one.\n- bullet a\n- bullet b\n|h1|h2|\n|---|---|\n|x|y|"
    assert split_units(text) == [
        "Intro sentence.",
        "Second one.",
        "- bullet a",
        "- bullet b",
        "|h1|h2|",
        "|---|---|",
        "|x|y|",
    ]


def test_short_chunks_are_returned_unchanged() -> None:
    embedder = FakeEmbedder()
    c = chunk("a", "A short chunk. Two sentences.")
    out = compress_chunk(embedder.embed_query("cough"), c, embedder, token_budget=200)
    assert out.text == c.text and not out.compressed


def test_compression_keeps_relevant_and_safety_sentences_in_order() -> None:
    embedder = FakeEmbedder()
    c = chunk("a", LONG)
    out = compress_chunk(
        embedder.embed_query("fast breathing pneumonia antibiotics"), c, embedder, token_budget=30
    )
    assert out.compressed and out.kept_tokens < out.original_tokens
    assert "Fast breathing suggests pneumonia" in out.text  # relevant
    assert out.text.startswith("Cough is common in children.")  # opening framing always kept
    # Safety cues survive even though they are irrelevant to this query.
    assert "Do not give cough syrup" in out.text
    assert "Refer urgently if the child cannot drink" in out.text
    assert "Vitamin A" not in out.text and OMISSION in out.text  # irrelevant, dropped and marked
    # Original order preserved.
    assert (
        out.text.index("Fast breathing")
        < out.text.index("Do not give")
        < out.text.index("Refer urgently")
    )


def test_compression_keeps_table_header_with_rows() -> None:
    names = [
        "amoxicillin",
        "benzylpenicillin",
        "chloramphenicol",
        "doxycycline",
        "erythromycin",
        "flucloxacillin",
        "gentamicin",
        "hydrocortisone",
        "ibuprofen",
        "lincomycin",
    ]
    rows = "\n".join(f"|{n}|usual dose of {n} given twice daily for infection|" for n in names)
    c = chunk("t", f"|Drug|Dose|\n|---|---|\n{rows}")
    embedder = FakeEmbedder()
    out = compress_chunk(embedder.embed_query("chloramphenicol"), c, embedder, token_budget=24)
    assert out.compressed and out.text.startswith(
        "|Drug|Dose|\n|---|---|"
    )  # header travels with rows
    assert "|chloramphenicol|" in out.text and "|lincomycin|" not in out.text


def test_safety_sentences_do_not_crowd_out_the_answer() -> None:
    # Many caution sentences, one relevant answer, tiny budget: the answer must still be kept.
    cautions = " ".join(f"Do not give drug{chr(97 + i)} to anyone." for i in range(6))
    c = chunk("a", f"Intro sentence here. The dose of zinc is twenty milligrams daily. {cautions}")
    embedder = FakeEmbedder()
    out = compress_chunk(embedder.embed_query("zinc dose milligrams"), c, embedder, token_budget=10)
    assert "dose of zinc is twenty milligrams" in out.text
    assert out.text.count("Do not give") == 6  # every caution is retained as well


def test_compress_chunks_embeds_the_query_once_per_call() -> None:
    embedder = FakeEmbedder()
    out = compress_chunks(
        "pneumonia", [chunk("a", LONG), chunk("b", "tiny")], embedder, token_budget=30
    )
    assert [e.chunk_id for e in out] == ["a", "b"]
    assert out[0].compressed and not out[1].compressed
