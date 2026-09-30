"""PDF → typed page blocks, using PyMuPDF's layout model (via pymupdf4llm).

The layout model classifies every region of a page (section-header, text,
list-item, table, page-header, page-footer, picture…) and reports where each
region's markdown sits in the page text. Consuming those typed blocks is far
more reliable than re-parsing markdown: e.g. a bullet rendered as "# ■ shock"
is still a list item, and running headers can be separated from content.

Extraction is CPU-bound, so page batches run in a process pool, and results are
cached per document (keyed by the PDF's SHA-256) so chunking can be iterated on
without re-running the layout model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pymupdf

from app.core.logging import get_logger
from app.rag.ingestion.manifest import DocumentManifest, StructureStrategy
from app.rag.models import Block, BlockKind, PageExtraction

log = get_logger(__name__)

EXTRACTOR_VERSION = "1"  # bump to invalidate caches when extraction logic changes
_PAGES_PER_TASK = 40

_KIND_BY_CLASS: dict[str, BlockKind] = {
    "title": BlockKind.HEADING,
    "section-header": BlockKind.HEADING,
    "text": BlockKind.TEXT,
    "caption": BlockKind.TEXT,
    "footnote": BlockKind.TEXT,
    "list-item": BlockKind.LIST_ITEM,
    "table": BlockKind.TABLE,
    "page-header": BlockKind.PAGE_HEADER,
    "page-footer": BlockKind.PAGE_FOOTER,
    # "picture" / "formula" carry no usable text here and are dropped.
}
_MD_HEADING = re.compile(r"^(#{1,6})\s")


def _extract_batch(pdf_path: str, page_indexes: list[int]) -> list[PageExtraction]:
    """Worker: extract a batch of 0-based page indexes into typed blocks."""
    import pymupdf4llm  # imported in the worker; loads the layout model per process

    page_chunks = pymupdf4llm.to_markdown(
        pdf_path,
        pages=page_indexes,
        page_chunks=True,
        use_ocr=False,  # every source PDF has a text layer; OCR only adds noise
        show_progress=False,
    )
    pages: list[PageExtraction] = []
    for chunk in page_chunks:
        text: str = chunk["text"]
        blocks: list[Block] = []
        for box in chunk["page_boxes"]:
            kind = _KIND_BY_CLASS.get(box["class"])
            if kind is None:
                continue
            start, end = box["pos"]
            raw = text[start:end].strip()
            if not raw:
                continue
            md = _MD_HEADING.match(raw)
            blocks.append(Block(kind=kind, text=raw, md_level=len(md.group(1)) if md else 0))
        pages.append(PageExtraction(page_number=chunk["metadata"]["page_number"], blocks=blocks))
    return pages


def toc_paths(pdf_path: Path) -> dict[int, list[str]]:
    """Active bookmark path for every page (1-based), from the PDF outline."""
    doc = pymupdf.open(pdf_path)  # type: ignore[no-untyped-call]
    entries = [
        (level, re.sub(r"\s+", " ", title).strip(), page)
        for level, title, page in doc.get_toc()
        if title.strip() and not title.startswith("_Hlk")  # Word bookmark noise
    ]
    paths: dict[int, list[str]] = {}
    stack: list[str] = []
    i = 0
    for page_number in range(1, doc.page_count + 1):
        while i < len(entries) and entries[i][2] <= page_number:
            level, title, _ = entries[i]
            stack = [*stack[: level - 1], title]
            i += 1
        paths[page_number] = list(stack)
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_paths(cache_dir: Path, doc_id: str) -> tuple[Path, Path]:
    return cache_dir / f"{doc_id}.jsonl", cache_dir / f"{doc_id}.meta.json"


def _load_cache(cache_dir: Path, doc: DocumentManifest, sha: str) -> list[PageExtraction] | None:
    pages_path, meta_path = _cache_paths(cache_dir, doc.doc_id)
    if not (pages_path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    if meta != {"sha256": sha, "extractor_version": EXTRACTOR_VERSION}:
        return None
    with pages_path.open(encoding="utf-8") as f:
        return [PageExtraction.model_validate_json(line) for line in f]


def _save_cache(
    cache_dir: Path, doc: DocumentManifest, sha: str, pages: list[PageExtraction]
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pages_path, meta_path = _cache_paths(cache_dir, doc.doc_id)
    with pages_path.open("w", encoding="utf-8") as f:
        for page in pages:
            f.write(page.model_dump_json() + "\n")
    meta_path.write_text(json.dumps({"sha256": sha, "extractor_version": EXTRACTOR_VERSION}))


def extract_documents(
    docs: list[DocumentManifest],
    data_dir: Path,
    cache_dir: Path,
    *,
    workers: int | None = None,
    force: bool = False,
) -> dict[str, list[PageExtraction]]:
    """Extract all documents (cached), returning pages in order per doc_id."""
    results: dict[str, list[PageExtraction]] = {}
    pending: list[tuple[DocumentManifest, Path, str]] = []
    for doc in docs:
        pdf = data_dir / doc.file
        sha = _sha256(pdf)
        cached = None if force else _load_cache(cache_dir, doc, sha)
        if cached is not None:
            log.info("ingest.extract_cached", doc_id=doc.doc_id, pages=len(cached))
            results[doc.doc_id] = cached
        else:
            pending.append((doc, pdf, sha))

    if pending:
        workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for doc, pdf, _ in pending:
                page_count = pymupdf.open(pdf).page_count  # type: ignore[no-untyped-call]
                indexes = list(range(page_count))  # skip_pages is applied at chunk time
                for start in range(0, len(indexes), _PAGES_PER_TASK):
                    batch = indexes[start : start + _PAGES_PER_TASK]
                    futures[pool.submit(_extract_batch, str(pdf), batch)] = doc.doc_id
            collected: dict[str, list[PageExtraction]] = {doc.doc_id: [] for doc, _, _ in pending}
            for n, future in enumerate(as_completed(futures), start=1):
                collected[futures[future]].extend(future.result())
                if n % 10 == 0 or n == len(futures):
                    log.info("ingest.extract_progress", batches_done=n, batches_total=len(futures))

        for doc, pdf, sha in pending:
            pages = sorted(collected[doc.doc_id], key=lambda p: p.page_number)
            if doc.structure is StructureStrategy.TOC:
                paths = toc_paths(pdf)
                for page in pages:
                    page.toc_path = paths.get(page.page_number, [])
            _save_cache(cache_dir, doc, sha, pages)
            results[doc.doc_id] = pages
            log.info("ingest.extracted", doc_id=doc.doc_id, pages=len(pages))
    return results
