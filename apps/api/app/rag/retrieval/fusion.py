"""Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009).

    score(d) = Σ_lists  weight / (k + rank_list(d))        rank is 1-based

RRF fuses by *rank*, so it needs no calibration between cosine similarities and
BM25 scores (which live on unrelated scales). ``k`` damps the influence of the
very top ranks; 60 is the value from the paper and the usual default.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.schemas.clinical import RetrievalScores, RetrievedChunk


def reciprocal_rank_fusion(
    dense_results: Sequence[RetrievedChunk],
    sparse_results: Sequence[RetrievedChunk],
    k: int = 60,
    *,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[RetrievedChunk]:
    """Merge two ranked lists into one, best first.

    A chunk present in both lists keeps both stage scores and ranks; one present in
    only one list simply gets that list's contribution. Ties are broken by best
    individual rank, then chunk id, so the order is deterministic.
    """
    if k < 0:
        raise ValueError("k must be >= 0")

    merged: dict[str, RetrievedChunk] = {}
    rrf: dict[str, float] = {}
    best_rank: dict[str, int] = {}

    for results, weight, is_dense in (
        (dense_results, dense_weight, True),
        (sparse_results, sparse_weight, False),
    ):
        for rank, hit in enumerate(results, start=1):
            cid = hit.chunk_id
            rrf[cid] = rrf.get(cid, 0.0) + weight / (k + rank)
            best_rank[cid] = min(best_rank.get(cid, rank), rank)
            existing = merged.get(cid)
            scores = existing.scores if existing else RetrievalScores()
            if is_dense:
                scores = scores.model_copy(
                    update={"dense_score": hit.scores.dense_score, "dense_rank": rank}
                )
            else:
                scores = scores.model_copy(
                    update={"bm25_score": hit.scores.bm25_score, "bm25_rank": rank}
                )
            merged[cid] = hit.model_copy(update={"scores": scores})

    for cid, chunk in merged.items():
        merged[cid] = chunk.model_copy(
            update={"scores": chunk.scores.model_copy(update={"rrf_score": rrf[cid]})}
        )
    order = sorted(merged, key=lambda cid: (-rrf[cid], best_rank[cid], cid))
    return [merged[cid] for cid in order]
