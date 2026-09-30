"""CLI for the vector index.

    python -m app.rag.retrieval status
    python -m app.rag.retrieval index [--recreate] [--local]
    python -m app.rag.retrieval query "cough for five days" [--population child] [--topic respiratory] [-k 5]

Backend: Qdrant Cloud/server when QDRANT_URL is set, otherwise an embedded local
index in data/indexes/qdrant. ``--local`` forces the embedded index.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.rag.embeddings import SentenceTransformerEmbedder
from app.rag.ingestion.pipeline import load_chunks
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.filters import RetrievalFilters
from app.rag.retrieval.indexer import index_chunks
from app.rag.retrieval.qdrant_store import QdrantChunkStore, build_client
from app.schemas.clinical import ChunkType


def _embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        query_instruction=settings.embedding_query_instruction,
    )


def _describe_backend(settings: Settings, force_local: bool) -> str:
    if settings.qdrant_url and not force_local:
        return f"Qdrant server ({settings.qdrant_url.split('//')[-1].split('.')[0]}…)"
    return f"embedded local index ({settings.qdrant_local_path})"


def _connection_help(settings: Settings, error: Exception) -> str:
    host = (settings.qdrant_url or "").split("//")[-1]
    return (
        f"Could not reach Qdrant at {host or '(no url)'}: {type(error).__name__}.\n"
        "  • Qdrant Cloud: open cloud.qdrant.io → Clusters. A free cluster that is suspended,\n"
        "    or deleted, resets connections. Check it is Running and copy its URL and API key.\n"
        "  • Or run without a server:  add --local (embedded index in data/indexes/qdrant).\n"
    )


async def cmd_status(settings: Settings, args: argparse.Namespace) -> int:
    client = build_client(settings, force_local=args.local)
    store = QdrantChunkStore(client, settings.qdrant_collection, settings.embedding_model)
    print(f"Backend:    {_describe_backend(settings, args.local)}")
    print(f"Collection: {settings.qdrant_collection}")
    if not await client.collection_exists(settings.qdrant_collection):
        print("Status:     not created yet (run: make index)")
    else:
        print(f"Points:     {await store.count()}")
    await client.close()
    return 0


async def cmd_index(settings: Settings, args: argparse.Namespace) -> int:
    chunks_path = settings.data_dir / "processed" / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"{chunks_path} not found — run `make ingest` first.", file=sys.stderr)
        return 1
    chunks = load_chunks(chunks_path)
    client = build_client(settings, force_local=args.local)
    store = QdrantChunkStore(client, settings.qdrant_collection, settings.embedding_model)
    embedder = _embedder(settings)

    print(f"Backend:   {_describe_backend(settings, args.local)}")
    print(
        f"Embedding: {settings.embedding_model}  ({sum(c.metadata.retrievable for c in chunks)} retrievable chunks)"
    )
    t0 = time.perf_counter()

    def progress(done: int, total: int) -> None:
        rate = done / max(time.perf_counter() - t0, 1e-6)
        print(f"\r  embedded {done}/{total}  ({rate:.0f} chunks/s)", end="", flush=True)

    result = await index_chunks(store, embedder, chunks, recreate=args.recreate, progress=progress)
    print(
        f"\nDone in {result.duration_s}s: {result.embedded} embedded, {result.unchanged} unchanged, "
        f"{result.deleted} removed → {result.total_in_collection} points in '{settings.qdrant_collection}'"
    )
    await client.close()
    return 0


async def cmd_query(settings: Settings, args: argparse.Namespace) -> int:
    client = build_client(settings, force_local=args.local)
    store = QdrantChunkStore(client, settings.qdrant_collection, settings.embedding_model)
    retriever = DenseRetriever(store, _embedder(settings))
    filters = RetrievalFilters(
        population=args.population or None,
        topics=args.topic or None,
        document_types=args.document_type or None,
        chunk_types=[ChunkType(c) for c in args.chunk_type] or None,
    )
    t0 = time.perf_counter()
    hits = await retriever.search(args.query, filters=filters, k=args.k)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(
        f"{len(hits)} results in {elapsed_ms:.0f} ms  ({_describe_backend(settings, args.local)})\n"
    )
    for rank, hit in enumerate(hits, start=1):
        m = hit.metadata
        print(
            f"{rank}. score={hit.scores.dense_score:.3f}  {m.document_id} p{m.page_number}  "
            f"[{m.chunk_type.value}/{m.population}/{m.topic}]"
        )
        print(f"   {' > '.join(m.heading_path)[-110:]}")
        print(f"   {hit.text[: args.chars].replace(chr(10), ' ')}\n")
    await client.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.rag.retrieval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="use the embedded local index even if QDRANT_URL is set",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show backend and index size")
    p_index = sub.add_parser(
        "index", help="embed data/processed/chunks.jsonl and upsert into Qdrant"
    )
    p_index.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    p_query = sub.add_parser("query", help="dense search with optional metadata filters")
    p_query.add_argument("query")
    p_query.add_argument("-k", type=int, default=5)
    p_query.add_argument(
        "--population", action="append", default=[], help="repeatable: adult|child|pregnancy|all"
    )
    p_query.add_argument(
        "--topic", action="append", default=[], help="repeatable, e.g. respiratory"
    )
    p_query.add_argument("--document-type", action="append", default=[])
    p_query.add_argument(
        "--chunk-type", action="append", default=[], choices=[c.value for c in ChunkType]
    )
    p_query.add_argument("--chars", type=int, default=260, help="excerpt length to print")
    args = parser.parse_args()

    # subcommand-level --local (after the subcommand) is also accepted
    settings = get_settings()
    configure_logging("WARNING", json_logs=False)
    handlers = {"status": cmd_status, "index": cmd_index, "query": cmd_query}
    try:
        return asyncio.run(handlers[args.command](settings, args))
    except (ResponseHandlingException, UnexpectedResponse, OSError) as exc:
        print(_connection_help(settings, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
