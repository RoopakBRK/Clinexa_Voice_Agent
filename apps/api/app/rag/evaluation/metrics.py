"""Retrieval metrics. Pure functions over ranked relevance judgments.

``rels`` is the ranked list of per-result relevance flags (True = the result matches
at least one gold evidence item); rank 1 is index 0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def hit_at_k(rels: Sequence[bool], k: int) -> float:
    """1.0 if any of the top-k results is relevant ("the answer was retrievable")."""
    return float(any(rels[:k]))


def reciprocal_rank(rels: Sequence[bool], k: int | None = None) -> float:
    """1 / rank of the first relevant result (0 if none within ``k``)."""
    for rank, rel in enumerate(rels[:k], start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def dcg(rels: Sequence[bool], k: int) -> float:
    return sum(1.0 / math.log2(rank + 1) for rank, rel in enumerate(rels[:k], start=1) if rel)


def ndcg_at_k(rels: Sequence[bool], total_relevant: int, k: int) -> float:
    """Binary-relevance NDCG@k. ``total_relevant`` is how many relevant chunks exist in
    the whole corpus, so the ideal ranking is min(k, total_relevant) relevant results first."""
    ideal = dcg([True] * min(k, total_relevant), k)
    return dcg(rels, k) / ideal if ideal > 0 else 0.0


def evidence_recall_at_k(unit_first_rank: Mapping[str, int | None], k: int) -> float:
    """Share of gold evidence items found within the top k. ``unit_first_rank`` maps each
    gold item to the rank of the first result satisfying it (None = never retrieved)."""
    if not unit_first_rank:
        return 0.0
    found = sum(1 for rank in unit_first_rank.values() if rank is not None and rank <= k)
    return found / len(unit_first_rank)


def auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """P(random positive scores higher than random negative); ties count half.
    Used to ask: does the top retrieval score separate answerable from unanswerable questions?"""
    if not positive_scores or not negative_scores:
        raise ValueError("auroc needs at least one positive and one negative score")
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positive_scores for n in negative_scores
    )
    return wins / (len(positive_scores) * len(negative_scores))


def percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)
