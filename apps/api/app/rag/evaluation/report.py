"""Evaluation report: Markdown for people, JSON for tooling."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from app.rag.evaluation.dataset import Category, EvalDataset
from app.rag.evaluation.metrics import ndcg_at_k, reciprocal_rank
from app.rag.evaluation.runner import (
    Metrics,
    QuestionRun,
    abstention_signal,
    aggregate,
    latency_summary,
)

_CATEGORY_ORDER = list(Category)


class ReportMeta(BaseModel):
    dataset_name: str
    dataset_version: str
    dataset_sha256: str
    n_questions: int
    n_answerable: int
    corpus_chunks: int
    embedding_model: str
    reranker_models: list[str]
    pipeline: dict[str, int]
    generated_at: str
    dataset_reviewed: bool


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# --- statistics --------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (resampling questions)."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def _per_question(runs: Sequence[QuestionRun], metric: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in runs:
        if r.expected != "answer":
            continue
        if metric == "hit5":
            out[r.question_id] = float(any(r.relevant[:5]))
        elif metric == "hit10":
            out[r.question_id] = float(any(r.relevant[:10]))
        elif metric == "mrr":
            out[r.question_id] = reciprocal_rank(r.relevant)
        elif metric == "ndcg10":
            out[r.question_id] = ndcg_at_k(r.relevant, r.total_relevant_in_corpus, 10)
    return out


def paired_delta(
    base: Sequence[QuestionRun], other: Sequence[QuestionRun], metric: str
) -> tuple[float, float, float] | None:
    """(mean difference other-base, CI low, CI high) over questions both systems answered."""
    a, b = _per_question(base, metric), _per_question(other, metric)
    ids = sorted(set(a) & set(b))
    if len(ids) < 2:
        return None
    diffs = [b[i] - a[i] for i in ids]
    lo, hi = bootstrap_ci(diffs)
    return sum(diffs) / len(diffs), lo, hi


# --- rendering -----------------------------------------------------------------------


def _f(x: float) -> str:
    return f"{x:.3f}"


def _row(system: str, m: Metrics, runs: Sequence[QuestionRun]) -> str:
    lat = latency_summary(runs)
    return (
        f"| `{system}` | {m.n} | {_f(m.hit[5])} | {_f(m.hit[10])} | {_f(m.evidence_recall[5])} | "
        f"{_f(m.evidence_recall[10])} | {_f(m.mrr)} | {_f(m.ndcg[5])} | {_f(m.ndcg[10])} | "
        f"{lat['p50_ms']:.0f} / {lat['p95_ms']:.0f} |"
    )


_HEADER = (
    "| System | n | Hit@5 | Hit@10 | EvRecall@5 | EvRecall@10 | MRR | NDCG@5 | NDCG@10 | "
    "latency p50 / p95 ms |\n|---|---|---|---|---|---|---|---|---|---|"
)


def render_markdown(
    meta: ReportMeta,
    dataset: EvalDataset,
    results: Mapping[str, Sequence[QuestionRun]],
    *,
    questions_text: Mapping[str, str],
) -> str:
    main = [s for s in results if not s.endswith(("+pop", "+pop+topic"))]
    filtered = [s for s in results if s.endswith(("+pop", "+pop+topic"))]
    L: list[str] = []

    status = (
        ""
        if meta.dataset_reviewed
        else "> ⚠️ **DRAFT**: the question set has not yet been reviewed by a human. Treat every number "
        "below as preliminary; do not quote it.\n"
    )
    L += [
        f"# Retrieval evaluation: {meta.dataset_name} v{meta.dataset_version}",
        "",
        status,
        f"Generated {meta.generated_at} · dataset `{meta.dataset_sha256}` · "
        f"{meta.n_questions} questions ({meta.n_answerable} answerable) · "
        f"{meta.corpus_chunks} indexed chunks",
        "",
        f"Embedding model `{meta.embedding_model}` · rerankers {', '.join(f'`{m}`' for m in meta.reranker_models)} · "
        f"candidate pool: dense top {meta.pipeline['k_dense']} + BM25 top {meta.pipeline['k_sparse']} → "
        f"RRF k={meta.pipeline['rrf_k']} → top {meta.pipeline['fused_k']}",
        "",
    ]

    L += [
        "## How to read this",
        "",
        '* **Hit@k**: an answer-bearing chunk is in the top k ("was the answer retrievable?").',
        "* **EvRecall@k** (evidence recall): share of the question's gold evidence items found in the top k.",
        "* **MRR**: mean of 1/rank of the first relevant chunk. **NDCG@k**: rank-aware, binary relevance.",
        "* Gold evidence is defined by criteria (document + pages/section + key terms), not chunk ids. "
        "Metrics average over answerable questions only.",
        "",
    ]

    L += ["## Systems compared (all answerable questions, no metadata filters)", "", _HEADER]
    for s in main:
        if (m := aggregate(results[s])) is not None:
            L.append(_row(s, m, results[s]))
    L.append("")

    # Reranker improvement with paired bootstrap CIs
    rerank_systems = [s for s in main if s.startswith("hybrid+rerank")]
    if "hybrid" in results and rerank_systems:
        L += [
            "## Reranker improvement over plain hybrid (paired, 95% bootstrap CI)",
            "",
            "| Reranker | ΔHit@5 | ΔHit@10 | ΔMRR | ΔNDCG@10 |",
            "|---|---|---|---|---|",
        ]
        for s in rerank_systems:
            cells = []
            for metric in ("hit5", "hit10", "mrr", "ndcg10"):
                d = paired_delta(results["hybrid"], results[s], metric)
                cells.append("n/a" if d is None else f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
            L.append(f"| `{s}` | " + " | ".join(cells) + " |")
        L += [
            "",
            "An interval that excludes 0 means the difference is unlikely to be sampling noise.",
            "",
        ]

    # Does BM25 earn its place? Same reranker, hybrid pool vs dense-only pool.
    pairs = [(s, s.replace("hybrid+rerank", "vector+rerank")) for s in rerank_systems]
    pairs = [(h, v) for h, v in pairs if v in results]
    if pairs:
        L += [
            "## Does BM25 help? (hybrid pool vs dense-only pool, same reranker; paired 95% CI)",
            "",
            "| Reranker | ΔHit@5 | ΔHit@10 | ΔMRR | ΔNDCG@10 |",
            "|---|---|---|---|---|",
        ]
        for hybrid_sys, vector_sys in pairs:
            cells = []
            for metric in ("hit5", "hit10", "mrr", "ndcg10"):
                d = paired_delta(results[vector_sys], results[hybrid_sys], metric)
                cells.append("n/a" if d is None else f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
            L.append(f"| `{hybrid_sys.split('[', 1)[1][:-1]}` | " + " | ".join(cells) + " |")
        L += [
            "",
            "Positive = adding BM25 to the candidate pool improved the reranked result. "
            "An interval spanning 0 means BM25's contribution is not distinguishable from noise "
            "on this question set.",
            "",
        ]

    # Absolute CIs for the headline systems
    L += [
        "## Confidence intervals (95% bootstrap over questions)",
        "",
        "| System | Hit@5 | MRR |",
        "|---|---|---|",
    ]
    for s in main:
        h, r = _per_question(results[s], "hit5"), _per_question(results[s], "mrr")
        if h:
            hl, hh = bootstrap_ci(list(h.values()))
            rl, rh = bootstrap_ci(list(r.values()))
            L.append(
                f"| `{s}` | {sum(h.values()) / len(h):.3f} [{hl:.3f}, {hh:.3f}] | "
                f"{sum(r.values()) / len(r):.3f} [{rl:.3f}, {rh:.3f}] |"
            )
    L.append("")

    # Per-category
    cats = [
        c
        for c in _CATEGORY_ORDER
        if any(q.category == c and q.expected == "answer" for q in dataset.questions)
    ]
    best = rerank_systems[0] if rerank_systems else None
    show = [s for s in ("vector", "bm25", "hybrid", best) if s and s in results]
    L += [
        "## By question category (Hit@5 / MRR)",
        "",
        "| Category | n | " + " | ".join(f"`{s}`" for s in show) + " |",
        "|---|---|" + "---|" * len(show),
    ]
    for c in cats:
        cells, n = [], 0
        for s in show:
            sub = [r for r in results[s] if r.category == c]
            m = aggregate(sub)
            n = m.n if m else n
            cells.append("–" if m is None else f"{m.hit[5]:.2f} / {m.mrr:.2f}")
        L.append(f"| {c.value} | {n} | " + " | ".join(cells) + " |")
    L.append("")

    # Filters ablation on the subset with caller context
    if filtered and best:
        subset_ids = {r.question_id for r in results[filtered[0]]}
        base_sub = [r for r in results[best] if r.question_id in subset_ids]
        L += [
            f"## Effect of metadata filters (the {len(subset_ids)} questions with caller context)",
            "",
            _HEADER,
        ]
        for name, runs in [(f"{best} (no filter)", base_sub), *[(s, results[s]) for s in filtered]]:
            if (m := aggregate(runs)) is not None:
                L.append(_row(name, m, runs))
        L += [
            "",
            "Population is a safety constraint (child vs adult dosing); topic is only a relevance "
            "hint and relies on rule-based topic labels, so it is the first thing to drop if it hurts.",
            "",
        ]

    # Abstention signal
    if best:
        sig = abstention_signal(results[best])
        vec = abstention_signal(results["vector"]) if "vector" in results else {}
        L += [
            "## Can retrieval tell when it has nothing relevant?",
            "",
            f"Top-1 score as a detector of unanswerable questions "
            f"({sig['n_answer']} answerable, {sig['n_abstain']} out-of-domain, {sig['n_clarify']} ambiguous). "
            "AUROC: 0.5 = no signal, 1.0 = perfect.",
            "",
            "| Score | AUROC answerable vs out-of-domain | AUROC answerable vs (out-of-domain + ambiguous) |",
            "|---|---|---|",
            f"| reranker score (`{best}`) | {sig.get('auroc_answer_vs_out_of_domain', 'n/a')} | "
            f"{sig.get('auroc_answer_vs_abstain_or_clarify', 'n/a')} |",
        ]
        if vec:
            L.append(
                f"| dense cosine (`vector`) | {vec.get('auroc_answer_vs_out_of_domain', 'n/a')} | "
                f"{vec.get('auroc_answer_vs_abstain_or_clarify', 'n/a')} |"
            )
        L += [
            "",
            "Final abstention accuracy is measured end-to-end once the agents exist (Phase 12); "
            "this only shows whether the retrieval score is a usable ingredient.",
            "",
        ]

    # Failures
    if best:
        misses = [r for r in results[best] if r.expected == "answer" and not any(r.relevant[:5])]
        L += [f"## Misses: answerable questions with no relevant chunk in the top 5 (`{best}`)", ""]
        if not misses:
            L.append("None.")
        for miss in misses:
            rank = miss.first_relevant_rank
            L.append(
                f'* **{miss.question_id}** ({miss.category.value}): "{questions_text[miss.question_id]}": '
                f"first relevant at rank {rank if rank else 'not in top 20'}"
            )
        L.append("")
    return "\n".join(L)


def build_meta(
    dataset: EvalDataset,
    dataset_path: Path,
    *,
    corpus_chunks: int,
    embedding_model: str,
    reranker_models: list[str],
    pipeline: dict[str, int],
    reviewed: bool,
) -> ReportMeta:
    return ReportMeta(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_sha256=file_sha256(dataset_path),
        n_questions=len(dataset.questions),
        n_answerable=sum(q.expected == "answer" for q in dataset.questions),
        corpus_chunks=corpus_chunks,
        embedding_model=embedding_model,
        reranker_models=reranker_models,
        pipeline=pipeline,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        dataset_reviewed=reviewed,
    )
