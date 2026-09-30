"""Ingestion pipeline: WHO PDFs → cleaned, structured, labelled chunks (JSONL).

data/<file>.pdf ──extract──▶ data/processed/extracted/<doc_id>.jsonl (cache)
                ──structure + chunk + classify──▶ data/processed/chunks.jsonl
                ──report──▶ data/processed/ingestion_report.{md,json}, samples.md
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from app.core.config import Settings
from app.core.logging import get_logger
from app.observability.metrics import percentile
from app.rag.ingestion.chunker import TokenCounter, chunk_document
from app.rag.ingestion.extract import extract_documents
from app.rag.ingestion.manifest import Manifest
from app.rag.ingestion.structure import structure_units
from app.rag.models import Chunk

log = get_logger(__name__)


class DocumentReport(BaseModel):
    doc_id: str
    title: str
    pages: int
    sections: int
    chunks: int
    retrievable_chunks: int
    tokens_p50: float
    tokens_p95: float
    tokens_max: int
    chunk_types: dict[str, int]
    populations: dict[str, int]
    top_topics: dict[str, int]


class IngestionReport(BaseModel):
    documents: list[DocumentReport]
    total_chunks: int
    total_retrievable: int
    target_tokens: int
    embedding_tokenizer: str
    duration_s: float


def build_token_counter(model_name: str) -> TokenCounter:
    """Count tokens with the embedding model's tokenizer (excluding special tokens)."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(model_name)
    tokenizer.no_truncation()

    @lru_cache(maxsize=200_000)
    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count


def _report(doc_id: str, title: str, pages: int, chunks: list[Chunk]) -> DocumentReport:
    tokens = [c.token_count for c in chunks] or [0]
    return DocumentReport(
        doc_id=doc_id,
        title=title,
        pages=pages,
        sections=len({tuple(c.metadata.heading_path) for c in chunks}),
        chunks=len(chunks),
        retrievable_chunks=sum(c.metadata.retrievable for c in chunks),
        tokens_p50=round(percentile(tokens, 50), 1),
        tokens_p95=round(percentile(tokens, 95), 1),
        tokens_max=max(tokens),
        chunk_types=dict(Counter(c.metadata.chunk_type.value for c in chunks)),
        populations=dict(Counter(c.metadata.population or "unknown" for c in chunks)),
        top_topics=dict(Counter(c.metadata.topic or "none" for c in chunks).most_common(5)),
    )


def _write_markdown_report(report: IngestionReport, path: Path) -> None:
    lines = [
        "# Ingestion report",
        "",
        f"Tokenizer: `{report.embedding_tokenizer}` · target {report.target_tokens} tokens/chunk · "
        f"{report.total_chunks} chunks ({report.total_retrievable} retrievable) · "
        f"{report.duration_s:.0f}s",
        "",
        "| Document | Pages | Sections | Chunks | Retrievable | Tokens p50 / p95 / max | Types |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in report.documents:
        types = ", ".join(f"{k} {v}" for k, v in sorted(d.chunk_types.items()))
        lines.append(
            f"| `{d.doc_id}` | {d.pages} | {d.sections} | {d.chunks} | {d.retrievable_chunks} | "
            f"{d.tokens_p50:.0f} / {d.tokens_p95:.0f} / {d.tokens_max} | {types} |"
        )
    lines += ["", "## Topics and populations", ""]
    for d in report.documents:
        lines.append(f"- **{d.doc_id}** — topics {d.top_topics}; populations {d.populations}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_samples(chunks_by_doc: dict[str, list[Chunk]], path: Path, per_doc: int = 4) -> None:
    """Random retrievable chunks per document, for human spot-checking."""
    rng = random.Random(7)
    lines = ["# Sample chunks (for review)", ""]
    for doc_id, chunks in chunks_by_doc.items():
        pool = [c for c in chunks if c.metadata.retrievable]
        lines += [f"## {doc_id}", ""]
        for chunk in rng.sample(pool, min(per_doc, len(pool))):
            m = chunk.metadata
            lines += [
                f"### `{chunk.chunk_id}`",
                f"*{' > '.join(m.heading_path) or '(no section)'}* · pages {m.page_number}–{m.page_end} · "
                f"{m.chunk_type.value} · {m.population} · topics {m.topics} · {chunk.token_count} tokens",
                "",
                "```text",
                chunk.text,
                "```",
                "",
            ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ingestion(
    settings: Settings,
    *,
    doc_ids: list[str] | None = None,
    workers: int | None = None,
    force_extract: bool = False,
    token_counter: TokenCounter | None = None,
) -> IngestionReport:
    started = time.perf_counter()
    data_dir = settings.data_dir
    out_dir = data_dir / "processed"
    manifest = Manifest.load(data_dir / "manifests" / "documents.yaml")
    docs = [d for d in manifest.documents if not doc_ids or d.doc_id in doc_ids]
    if doc_ids and len(docs) != len(doc_ids):
        unknown = set(doc_ids) - {d.doc_id for d in docs}
        raise ValueError(f"Unknown doc_id(s): {sorted(unknown)}")

    extracted = extract_documents(
        docs, data_dir, out_dir / "extracted", workers=workers, force=force_extract
    )
    count = token_counter or build_token_counter(settings.embedding_model)

    chunks_by_doc: dict[str, list[Chunk]] = {}
    reports: list[DocumentReport] = []
    for doc in docs:
        pages = [p for p in extracted[doc.doc_id] if not doc.skips(p.page_number)]
        units = structure_units(pages, doc.structure, doc.section_aliases)
        chunks = chunk_document(
            doc,
            units,
            count,
            target_tokens=settings.chunk_target_tokens,
            min_tokens=settings.chunk_min_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            section_merge_tokens=settings.chunk_section_merge_tokens,
        )
        chunks_by_doc[doc.doc_id] = chunks
        reports.append(_report(doc.doc_id, doc.title, len(pages), chunks))
        log.info("ingest.chunked", doc_id=doc.doc_id, units=len(units), chunks=len(chunks))

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunks in chunks_by_doc.values():
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")

    all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]
    report = IngestionReport(
        documents=reports,
        total_chunks=len(all_chunks),
        total_retrievable=sum(c.metadata.retrievable for c in all_chunks),
        target_tokens=settings.chunk_target_tokens,
        embedding_tokenizer=settings.embedding_model,
        duration_s=round(time.perf_counter() - started, 1),
    )
    (out_dir / "ingestion_report.json").write_text(report.model_dump_json(indent=2))
    _write_markdown_report(report, out_dir / "ingestion_report.md")
    _write_samples(chunks_by_doc, out_dir / "samples.md")
    log.info("ingest.done", chunks=report.total_chunks, retrievable=report.total_retrievable)
    return report


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk.model_validate(json.loads(line)) for line in f]
