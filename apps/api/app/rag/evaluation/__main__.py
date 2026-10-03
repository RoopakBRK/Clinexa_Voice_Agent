"""CLI: run the retrieval evaluation.

    python -m app.rag.evaluation validate  [--dataset PATH]
    python -m app.rag.evaluation retrieval [--dataset PATH] [--rerankers MODEL ...] [--local] [--reviewed]

``validate`` checks every gold criterion against the real corpus (no models needed).
``retrieval`` compares vector-only, BM25-only, hybrid and hybrid+reranker and writes
evaluation/reports/retrieval_<date>[_DRAFT].{md,json}. Reports are marked DRAFT until
the dataset is reviewed and ``--reviewed`` is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime

from app.core.config import REPO_ROOT, get_settings
from app.core.logging import configure_logging
from app.rag.embeddings import SentenceTransformerEmbedder
from app.rag.evaluation.dataset import EvalDataset, validate_gold
from app.rag.evaluation.report import build_meta, render_markdown
from app.rag.evaluation.runner import RetrievalEvaluator
from app.rag.ingestion.pipeline import load_chunks
from app.rag.reranking.cross_encoder import CrossEncoderReranker
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.qdrant_store import QdrantChunkStore, build_client
from app.rag.retrieval.sparse import SparseRetriever

DEFAULT_DATASET = REPO_ROOT / "evaluation" / "datasets" / "retrieval_eval_v1.yaml"


def _short(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def cmd_validate(args: argparse.Namespace) -> int:
    settings = get_settings()
    dataset = EvalDataset.load(args.dataset)
    chunks = load_chunks(settings.data_dir / "processed" / "chunks.jsonl")
    problems = validate_gold(dataset, [(c.metadata, c.text) for c in chunks])
    by_cat: dict[str, int] = {}
    for q in dataset.questions:
        by_cat[q.category.value] = by_cat.get(q.category.value, 0) + 1
    print(f"{dataset.name} v{dataset.version}: {len(dataset.questions)} questions {by_cat}")
    for p in problems:
        print(f"  [{p.severity}] {p.question_id} unit {p.unit_index}: {p.message}")
    errors = sum(p.severity == "error" for p in problems)
    print(f"{errors} error(s), {len(problems) - errors} warning(s)")
    return 1 if errors else 0


async def cmd_retrieval(args: argparse.Namespace) -> int:
    settings = get_settings()
    dataset = EvalDataset.load(args.dataset)
    chunks = load_chunks(settings.data_dir / "processed" / "chunks.jsonl")
    corpus = [(c.metadata, c.text) for c in chunks]
    problems = [p for p in validate_gold(dataset, corpus) if p.severity == "error"]
    if problems:
        print(
            "Dataset has gold criteria that match nothing; run `validate` first.", file=sys.stderr
        )
        return 1

    client = build_client(settings, force_local=args.local)
    store = QdrantChunkStore(client, settings.qdrant_collection, settings.embedding_model)
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        query_instruction=settings.embedding_query_instruction,
    )
    dense, sparse = DenseRetriever(store, embedder), SparseRetriever(chunks)
    hybrid = HybridRetriever(dense, sparse)
    models = args.rerankers or [settings.reranker_model]
    rerankers = {
        _short(m): CrossEncoderReranker(
            m, device=settings.embedding_device, batch_size=settings.reranker_batch_size
        )
        for m in models
    }
    evaluator = RetrievalEvaluator(dense, sparse, hybrid, rerankers, corpus)

    def progress(system: str, i: int, n: int) -> None:
        print(f"\r  {system:40} {i}/{n}", end="", flush=True)

    results = await evaluator.evaluate(dataset, progress=progress)
    print()
    await client.close()

    meta = build_meta(
        dataset,
        args.dataset,
        corpus_chunks=sum(c.metadata.retrievable for c in chunks),
        embedding_model=settings.embedding_model,
        reranker_models=models,
        pipeline={"k_dense": 15, "k_sparse": 15, "fused_k": 20, "rrf_k": 60},
        reviewed=args.reviewed,
    )
    markdown = render_markdown(
        meta, dataset, results, questions_text={q.id: q.question for q in dataset.questions}
    )
    stem = f"retrieval_{datetime.now(UTC):%Y-%m-%d}{'' if args.reviewed else '_DRAFT'}"
    out_dir = REPO_ROOT / "evaluation" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "meta": meta.model_dump(),
                "runs": {s: [r.model_dump() for r in runs] for s, runs in results.items()},
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(markdown)
    print(f"\nWrote {out_dir / (stem + '.md')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.rag.evaluation",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "retrieval"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", type=type(DEFAULT_DATASET), default=DEFAULT_DATASET)
        if name == "retrieval":
            p.add_argument("--rerankers", nargs="*", help="cross-encoder models to compare")
            p.add_argument(
                "--local", action="store_true", help="use the embedded local Qdrant index"
            )
            p.add_argument(
                "--reviewed", action="store_true", help="dataset has been human-reviewed"
            )
    args = parser.parse_args()
    configure_logging("WARNING", json_logs=False)
    if args.command == "validate":
        return cmd_validate(args)
    return asyncio.run(cmd_retrieval(args))


if __name__ == "__main__":
    sys.exit(main())
