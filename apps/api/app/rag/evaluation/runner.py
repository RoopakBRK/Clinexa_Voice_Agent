"""Run retrieval systems over the evaluation dataset and score them.

Systems compared (each returns a ranked list of up to DEPTH chunks):
    vector        dense retrieval only
    bm25          BM25 only
    hybrid        dense + BM25 fused with RRF
    hybrid+rerank hybrid candidates re-ordered by a cross-encoder
and, for questions with caller context, hybrid+rerank with metadata filters.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.rag.evaluation.dataset import Category, EvalDataset, EvalQuestion
from app.rag.evaluation.metrics import (
    auroc,
    evidence_recall_at_k,
    hit_at_k,
    ndcg_at_k,
    percentile,
    reciprocal_rank,
)
from app.rag.reranking.cross_encoder import Reranker, arerank
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.filters import RetrievalFilters, filters_from_clinical
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.sparse import SparseRetriever
from app.schemas.clinical import ClinicalIntake, DocumentMetadata, RetrievedChunk

log = get_logger(__name__)

DEPTH = 20
KS = (1, 3, 5, 10, 20)

System = Callable[[EvalQuestion], Awaitable[tuple[list[RetrievedChunk], dict[str, float]]]]


class QuestionRun(BaseModel):
    question_id: str
    category: Category
    expected: str
    system: str
    chunk_ids: list[str]
    relevant: list[bool]
    unit_first_rank: dict[str, int | None] = Field(default_factory=dict)
    total_relevant_in_corpus: int = 0
    top_score: float | None = None  # system's own top-1 score (abstention signal)
    latency_ms: float = 0.0
    stage_ms: dict[str, float] = Field(default_factory=dict)

    @property
    def first_relevant_rank(self) -> int | None:
        return next((i for i, r in enumerate(self.relevant, start=1) if r), None)


class Metrics(BaseModel):
    n: int
    hit: dict[int, float]
    evidence_recall: dict[int, float]
    mrr: float
    ndcg: dict[int, float]


def aggregate(runs: Sequence[QuestionRun]) -> Metrics | None:
    """Macro-average over answerable questions (None if there are none)."""
    answerable = [r for r in runs if r.expected == "answer"]
    if not answerable:
        return None
    n = len(answerable)
    return Metrics(
        n=n,
        hit={k: sum(hit_at_k(r.relevant, k) for r in answerable) / n for k in KS},
        evidence_recall={
            k: sum(evidence_recall_at_k(r.unit_first_rank, k) for r in answerable) / n for k in KS
        },
        mrr=sum(reciprocal_rank(r.relevant) for r in answerable) / n,
        ndcg={
            k: sum(ndcg_at_k(r.relevant, r.total_relevant_in_corpus, k) for r in answerable) / n
            for k in (5, 10)
        },
    )


def latency_summary(runs: Sequence[QuestionRun]) -> dict[str, float]:
    values = [r.latency_ms for r in runs]
    return {
        "p50_ms": round(percentile(values, 50), 1),
        "p95_ms": round(percentile(values, 95), 1),
        "max_ms": round(max(values), 1),
    }


def abstention_signal(runs: Sequence[QuestionRun]) -> dict[str, float | int]:
    """Can the system's top score tell answerable questions from unanswerable ones?
    (AUROC 0.5 = no signal, 1.0 = perfect separation.)"""
    scored = [(r.expected, r.top_score) for r in runs if r.top_score is not None]
    answer = [s for e, s in scored if e == "answer"]
    abstain = [s for e, s in scored if e == "abstain"]
    clarify = [s for e, s in scored if e == "clarify"]
    out: dict[str, float | int] = {
        "n_answer": len(answer),
        "n_abstain": len(abstain),
        "n_clarify": len(clarify),
    }
    if answer and abstain:
        out["auroc_answer_vs_out_of_domain"] = round(auroc(answer, abstain), 3)
    if answer and (abstain or clarify):
        out["auroc_answer_vs_abstain_or_clarify"] = round(auroc(answer, abstain + clarify), 3)
    return out


class RetrievalEvaluator:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        hybrid: HybridRetriever,
        rerankers: Mapping[str, Reranker],
        corpus: Sequence[tuple[DocumentMetadata, str]],
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._hybrid = hybrid
        self._rerankers = rerankers
        self._corpus = [(m, t) for m, t in corpus if m.retrievable]

    # --- systems ---------------------------------------------------------------

    def systems(self) -> dict[str, System]:
        systems: dict[str, System] = {
            "vector": self._vector,
            "bm25": self._bm25,
            "hybrid": self._hybrid_only,
        }
        for name, reranker in self._rerankers.items():
            # Same cross-encoder over the dense-only top 20: isolates what BM25 adds.
            systems[f"vector+rerank[{name}]"] = self._vector_rerank_system(reranker)
            systems[f"hybrid+rerank[{name}]"] = self._rerank_system(reranker, filters=None)
            systems[f"hybrid+rerank[{name}]+pop"] = self._rerank_system(
                reranker, filters="population"
            )
            systems[f"hybrid+rerank[{name}]+pop+topic"] = self._rerank_system(
                reranker, filters="population+topic"
            )
        return systems

    async def _vector(self, q: EvalQuestion) -> tuple[list[RetrievedChunk], dict[str, float]]:
        t = time.perf_counter()
        hits = await self._dense.search(q.query, k=DEPTH)
        return hits, {"dense_ms": (time.perf_counter() - t) * 1000}

    async def _bm25(self, q: EvalQuestion) -> tuple[list[RetrievedChunk], dict[str, float]]:
        t = time.perf_counter()
        hits = await self._sparse.asearch(q.query, k=DEPTH)
        return hits, {"bm25_ms": (time.perf_counter() - t) * 1000}

    async def _hybrid_only(self, q: EvalQuestion) -> tuple[list[RetrievedChunk], dict[str, float]]:
        result = await self._hybrid.search(q.query)
        return result.candidates, result.timings_ms

    def _vector_rerank_system(self, reranker: Reranker) -> System:
        async def run(q: EvalQuestion) -> tuple[list[RetrievedChunk], dict[str, float]]:
            t = time.perf_counter()
            hits = await self._dense.search(q.query, k=DEPTH)
            dense_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            reranked = await arerank(reranker, q.query, hits, top_k=None)
            return reranked, {"dense_ms": dense_ms, "rerank_ms": (time.perf_counter() - t) * 1000}

        return run

    def _rerank_system(self, reranker: Reranker, *, filters: str | None) -> System:
        async def run(q: EvalQuestion) -> tuple[list[RetrievedChunk], dict[str, float]]:
            f: RetrievalFilters | None = None
            if filters:
                f = _filters_for(q, include_topics=filters == "population+topic")
            result = await self._hybrid.search(q.query, filters=f)
            t = time.perf_counter()
            reranked = await arerank(reranker, q.query, result.candidates, top_k=None)
            timings = {**result.timings_ms, "rerank_ms": (time.perf_counter() - t) * 1000}
            return reranked, timings

        return run

    # --- scoring ------------------------------------------------------------------

    def _score(
        self,
        q: EvalQuestion,
        system: str,
        hits: list[RetrievedChunk],
        stage: dict[str, float],
        latency_ms: float,
    ) -> QuestionRun:
        relevant = [any(u.matches(h.metadata, h.text) for u in q.gold) for h in hits]
        unit_first_rank: dict[str, int | None] = {}
        for i, unit in enumerate(q.gold):
            unit_first_rank[str(i)] = next(
                (rank for rank, h in enumerate(hits, start=1) if unit.matches(h.metadata, h.text)),
                None,
            )
        total_relevant = (
            sum(any(u.matches(meta, text) for u in q.gold) for meta, text in self._corpus)
            if q.gold
            else 0
        )
        top = hits[0].scores if hits else None
        top_score = None
        if top is not None:
            top_score = next(
                (
                    v
                    for v in (top.reranker_score, top.dense_score, top.bm25_score, top.rrf_score)
                    if v is not None
                ),
                None,
            )
        return QuestionRun(
            question_id=q.id,
            category=q.category,
            expected=q.expected,
            system=system,
            chunk_ids=[h.chunk_id for h in hits],
            relevant=relevant,
            unit_first_rank=unit_first_rank,
            total_relevant_in_corpus=total_relevant,
            top_score=top_score,
            latency_ms=round(latency_ms, 2),
            stage_ms={k: round(v, 2) for k, v in stage.items()},
        )

    async def evaluate(
        self,
        dataset: EvalDataset,
        *,
        systems: Sequence[str] | None = None,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, list[QuestionRun]]:
        available = self.systems()
        names = list(systems) if systems else list(available)
        results: dict[str, list[QuestionRun]] = {}
        for name in names:
            run_fn = available[name]
            runs: list[QuestionRun] = []
            questions = [
                q
                for q in dataset.questions
                if not name.endswith(("+pop", "+pop+topic"))
                or (q.caller and q.caller.age is not None)
            ]
            for i, q in enumerate(questions, start=1):
                t = time.perf_counter()
                hits, stage = await run_fn(q)
                runs.append(self._score(q, name, hits, stage, (time.perf_counter() - t) * 1000))
                if progress:
                    progress(name, i, len(questions))
            results[name] = runs
            log.info("eval.system_done", system=name, questions=len(runs))
        return results


def _filters_for(q: EvalQuestion, *, include_topics: bool) -> RetrievalFilters | None:
    if not q.caller:
        return None
    intake = ClinicalIntake(
        chief_complaint=q.query, age=q.caller.age, pregnancy_status=q.caller.pregnancy_status
    )
    return filters_from_clinical(intake, include_topics=include_topics)
