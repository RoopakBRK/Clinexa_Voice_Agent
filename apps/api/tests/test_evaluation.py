"""Evaluation metrics, gold-criteria matching, runner and report."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient

from app.rag.evaluation.dataset import (
    Caller,
    Category,
    EvalDataset,
    EvalQuestion,
    GoldUnit,
    validate_gold,
)
from app.rag.evaluation.metrics import (
    auroc,
    evidence_recall_at_k,
    hit_at_k,
    ndcg_at_k,
    percentile,
    reciprocal_rank,
)
from app.rag.evaluation.report import (
    bootstrap_ci,
    build_meta,
    paired_delta,
    render_markdown,
)
from app.rag.evaluation.runner import (
    QuestionRun,
    RetrievalEvaluator,
    abstention_signal,
    aggregate,
)
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.qdrant_store import QdrantChunkStore
from app.rag.retrieval.sparse import SparseRetriever
from app.schemas.clinical import DocumentMetadata
from tests.test_reranking import OverlapReranker
from tests.test_retrieval_hybrid import CORPUS
from tests.test_retrieval_store import FakeEmbedder

# --- metrics -----------------------------------------------------------------------


def test_hit_and_reciprocal_rank() -> None:
    rels = [False, False, True, True]
    assert (hit_at_k(rels, 2), hit_at_k(rels, 3)) == (0.0, 1.0)
    assert reciprocal_rank(rels) == pytest.approx(1 / 3)
    assert reciprocal_rank(rels, k=2) == 0.0
    assert reciprocal_rank([]) == 0.0


def test_ndcg_matches_hand_computation() -> None:
    # relevant at ranks 2 and 4; two relevant chunks exist in the corpus
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([False, True, False, True], total_relevant=2, k=4) == pytest.approx(
        dcg / ideal
    )
    assert ndcg_at_k([True, True], total_relevant=2, k=2) == pytest.approx(1.0)
    assert ndcg_at_k([False, False], total_relevant=2, k=2) == 0.0
    # Ideal is capped at k: five relevant chunks exist but only k=2 slots are scored.
    assert ndcg_at_k([True, True], total_relevant=5, k=2) == pytest.approx(1.0)
    assert ndcg_at_k([True], total_relevant=0, k=5) == 0.0


def test_evidence_recall() -> None:
    first = {"0": 2, "1": 7, "2": None}
    assert evidence_recall_at_k(first, 5) == pytest.approx(1 / 3)
    assert evidence_recall_at_k(first, 10) == pytest.approx(2 / 3)
    assert evidence_recall_at_k({}, 5) == 0.0


def test_auroc_and_percentile() -> None:
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1], [0.9]) == 0.0
    assert auroc([0.5], [0.5]) == 0.5
    assert auroc([3, 1], [2]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        auroc([], [1.0])
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([10.0], 95) == 10.0


# --- gold criteria ---------------------------------------------------------------------


def meta(
    doc: str = "d", pages: tuple[int, int] = (10, 11), path: list[str] | None = None
) -> DocumentMetadata:
    return DocumentMetadata(
        source="s",
        document_title="t",
        document_type="x",
        document_id=doc,
        page_number=pages[0],
        page_end=pages[1],
        heading_path=path or ["5 Diarrhoea", "5.2 Acute diarrhoea"],
    )


def test_gold_unit_matching_rules() -> None:
    text = "Give oral rehydration solution (ORS) for some dehydration."
    assert GoldUnit(document_id="d", must_contain=["ors", "dehydration"]).matches(meta(), text)
    assert not GoldUnit(document_id="other", must_contain=["ors"]).matches(meta(), text)
    assert not GoldUnit(document_id="d", must_contain=["ors", "zinc"]).matches(
        meta(), text
    )  # ALL required
    assert GoldUnit(document_id="d", any_of=["zinc", "ors"]).matches(meta(), text)  # ANY of
    assert GoldUnit(document_id="d", section_contains="acute diarrhoea").matches(meta(), text)
    assert not GoldUnit(document_id="d", section_contains="malaria").matches(meta(), text)
    # page range overlaps with the chunk's [page, page_end]
    assert GoldUnit(document_id="d", pages=(11, 20)).matches(meta(pages=(10, 11)), text)
    assert GoldUnit(document_id="d", pages=(1, 10)).matches(meta(pages=(10, 11)), text)
    assert not GoldUnit(document_id="d", pages=(12, 20)).matches(meta(pages=(10, 11)), text)


def test_gold_unit_accepts_alternative_documents_and_sections() -> None:
    text = "Fast breathing suggests pneumonia."
    unit = GoldUnit(
        document_id=["hos", "eur"], section_contains=["Pneumonia", "Cough or difficulty"]
    )
    assert unit.matches(
        meta("eur", path=["6 Complaints", "6.1 Cough or difficulty in breathing"]), text
    )
    assert unit.matches(meta("hos", path=["4 Cough", "4.2 Pneumonia"]), text)
    assert not unit.matches(meta("apc", path=["Pneumonia"]), text)  # document not accepted
    assert not unit.matches(
        meta("hos", path=["5 Diarrhoea"]), text
    )  # no section alternative matches


def test_gold_unit_needs_a_constraint_beyond_the_document() -> None:
    with pytest.raises(ValidationError, match="needs pages"):
        GoldUnit(document_id="d")


def test_question_gold_must_match_expected_behaviour() -> None:
    unit = GoldUnit(document_id="d", must_contain=["x"])
    EvalQuestion(id="a", category=Category.SYMPTOM, question="q", gold=[unit])
    EvalQuestion(id="b", category=Category.OUT_OF_DOMAIN, question="q", expected="abstain")
    with pytest.raises(ValidationError, match="needs gold"):
        EvalQuestion(id="c", category=Category.SYMPTOM, question="q")
    with pytest.raises(ValidationError, match="must not have gold"):
        EvalQuestion(
            id="d", category=Category.AMBIGUOUS, question="q", expected="clarify", gold=[unit]
        )


def test_dataset_ids_unique_and_loadable(tmp_path: Path) -> None:
    q = {
        "id": "a",
        "category": "symptom",
        "question": "q",
        "gold": [{"document_id": "d", "must_contain": ["x"]}],
    }
    path = tmp_path / "ds.yaml"
    import yaml

    path.write_text(yaml.safe_dump({"name": "t", "version": "1", "questions": [q]}))
    assert EvalDataset.load(path).questions[0].gold[0].must_contain == ["x"]
    with pytest.raises(ValidationError, match="unique"):
        EvalDataset(name="t", version="1", questions=[EvalQuestion(**q), EvalQuestion(**q)])


def test_validate_gold_flags_unmatched_and_overbroad_units() -> None:
    corpus = [(meta(), "ors for dehydration")] * 3
    ds = EvalDataset(
        name="t",
        version="1",
        questions=[
            EvalQuestion(
                id="ok",
                category=Category.SYMPTOM,
                question="q",
                gold=[GoldUnit(document_id="d", must_contain=["ors"])],
            ),
            EvalQuestion(
                id="bad",
                category=Category.SYMPTOM,
                question="q",
                gold=[GoldUnit(document_id="d", must_contain=["nonexistent"])],
            ),
        ],
    )
    problems = validate_gold(ds, corpus, max_matches=2)
    assert {(p.question_id, p.severity) for p in problems} == {("ok", "warning"), ("bad", "error")}


# --- runner (tiny corpus, fake embedder and reranker) ------------------------------------


def question(
    qid: str,
    text: str,
    *,
    gold: list[GoldUnit] | None = None,
    category: Category = Category.SYMPTOM,
    expected: str = "answer",
    caller: Caller | None = None,
) -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        category=category,
        question=text,
        gold=gold or [],
        expected=expected,  # type: ignore[arg-type]
        caller=caller,
    )


DATASET = EvalDataset(
    name="tiny",
    version="0",
    questions=[
        question(
            "q-pneu",
            "child with cough and fast breathing",
            gold=[GoldUnit(document_id="doc-a", must_contain=["pneumonia"])],
            caller=Caller(age=4),
        ),
        question(
            "q-mal",
            "dose of artemether lumefantrine",
            gold=[
                GoldUnit(document_id="who-malaria", must_contain=["artemether"]),
                GoldUnit(document_id="doc-a", must_contain=["nonexistent-term"]),
            ],
        ),
        question(
            "q-ood", "best pizza toppings", category=Category.OUT_OF_DOMAIN, expected="abstain"
        ),
    ],
)


@pytest.fixture
async def evaluator() -> AsyncIterator[RetrievalEvaluator]:
    client = AsyncQdrantClient(":memory:")
    store = QdrantChunkStore(client, "eval", "fake-bow")
    embedder = FakeEmbedder()
    await store.ensure_collection(embedder.dimension)
    await store.upsert(CORPUS, embedder.embed_documents([c.embedding_text() for c in CORPUS]))
    dense, sparse = DenseRetriever(store, embedder), SparseRetriever(CORPUS)
    yield RetrievalEvaluator(
        dense,
        sparse,
        HybridRetriever(dense, sparse),
        {"overlap": OverlapReranker()},
        [(c.metadata, c.text) for c in CORPUS],
    )
    await client.close()


async def test_runner_scores_every_system(evaluator: RetrievalEvaluator) -> None:
    results = await evaluator.evaluate(DATASET)
    assert set(results) == {
        "vector",
        "bm25",
        "hybrid",
        "vector+rerank[overlap]",
        "hybrid+rerank[overlap]",
        "hybrid+rerank[overlap]+pop",
        "hybrid+rerank[overlap]+pop+topic",
    }
    # Filtered systems only run on questions that have caller context.
    assert [r.question_id for r in results["hybrid+rerank[overlap]+pop"]] == ["q-pneu"]
    assert len(results["hybrid"]) == 3
    # vector+rerank reorders the dense-only pool with the same cross-encoder.
    assert results["vector+rerank[overlap]"][0].stage_ms.keys() >= {"dense_ms", "rerank_ms"}
    assert "bm25_ms" not in results["vector+rerank[overlap]"][0].stage_ms

    by_q = {r.question_id: r for r in results["hybrid+rerank[overlap]"]}
    pneu = by_q["q-pneu"]
    assert pneu.relevant[0] and pneu.first_relevant_rank == 1 and pneu.chunk_ids[0] == "c1"
    assert pneu.total_relevant_in_corpus == 1 and pneu.top_score is not None
    assert pneu.stage_ms.keys() >= {"dense_ms", "bm25_ms", "rrf_ms", "rerank_ms"}
    # Two gold units; one can never be satisfied -> evidence recall is 0.5.
    mal = by_q["q-mal"]
    assert mal.unit_first_rank == {"0": 1, "1": None}
    ood = by_q["q-ood"]
    assert ood.expected == "abstain" and not any(ood.relevant)


async def test_aggregate_uses_answerable_questions_only(evaluator: RetrievalEvaluator) -> None:
    results = await evaluator.evaluate(DATASET, systems=["hybrid+rerank[overlap]"])
    m = aggregate(results["hybrid+rerank[overlap]"])
    assert m is not None and m.n == 2  # the out-of-domain question is excluded
    assert m.hit[5] == 1.0 and m.mrr == 1.0
    assert m.evidence_recall[5] == pytest.approx((1.0 + 0.5) / 2)
    assert (
        aggregate([r for r in results["hybrid+rerank[overlap]"] if r.expected != "answer"]) is None
    )


def run(qid: str, expected: str, top: float | None, relevant: list[bool]) -> QuestionRun:
    return QuestionRun(
        question_id=qid,
        category=Category.SYMPTOM,
        expected=expected,
        system="s",
        chunk_ids=[str(i) for i in range(len(relevant))],
        relevant=relevant,
        top_score=top,
        unit_first_rank={"0": 1 if any(relevant) else None},
        total_relevant_in_corpus=1,
    )


def test_abstention_signal() -> None:
    runs = [
        run("a", "answer", 0.9, [True]),
        run("b", "answer", 0.8, [True]),
        run("c", "abstain", 0.1, [False]),
        run("d", "clarify", 0.85, [False]),
    ]
    sig = abstention_signal(runs)
    assert sig["auroc_answer_vs_out_of_domain"] == 1.0
    assert sig["auroc_answer_vs_abstain_or_clarify"] == pytest.approx(
        0.75
    )  # pairs: 0.9>0.1, 0.9>0.85, 0.8>0.1, but 0.8<0.85
    assert (sig["n_answer"], sig["n_abstain"], sig["n_clarify"]) == (2, 1, 1)


# --- statistics and report ---------------------------------------------------------------


def test_bootstrap_ci_is_deterministic_and_sane() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    lo, hi = bootstrap_ci(values)
    assert (lo, hi) == bootstrap_ci(values)  # seeded
    assert lo <= sum(values) / len(values) <= hi
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)


def test_paired_delta_compares_same_questions() -> None:
    base = [
        run("a", "answer", 0.5, [False, False, False, False, False, True]),
        run("b", "answer", 0.5, [True]),
        run("c", "answer", 0.5, [False]),
    ]
    better = [
        run("a", "answer", 0.5, [True]),
        run("b", "answer", 0.5, [True]),
        run("c", "answer", 0.5, [False]),
    ]
    mean, lo, hi = paired_delta(base, better, "hit5") or (0, 0, 0)
    assert mean == pytest.approx(1 / 3) and lo <= mean <= hi
    assert paired_delta(base[:1], better[:1], "hit5") is None  # too few paired questions


async def test_markdown_report_flags_unreviewed_datasets(
    evaluator: RetrievalEvaluator, tmp_path: Path
) -> None:
    results = await evaluator.evaluate(DATASET)
    path = tmp_path / "ds.yaml"
    path.write_text("name: tiny")
    texts = {q.id: q.question for q in DATASET.questions}
    for reviewed in (False, True):
        meta_ = build_meta(
            DATASET,
            path,
            corpus_chunks=5,
            embedding_model="fake-bow",
            reranker_models=["overlap"],
            pipeline={"k_dense": 15, "k_sparse": 15, "fused_k": 20, "rrf_k": 60},
            reviewed=reviewed,
        )
        md = render_markdown(meta_, DATASET, results, questions_text=texts)
        assert ("DRAFT" in md) is (not reviewed)
        for heading in (
            "Systems compared",
            "Reranker improvement",
            "By question category",
            "Effect of metadata filters",
            "Can retrieval tell",
            "Misses",
        ):
            assert heading in md
        assert "`hybrid+rerank[overlap]`" in md
