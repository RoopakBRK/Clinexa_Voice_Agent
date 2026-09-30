"""Curated per-document metadata (``data/manifests/documents.yaml``)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class StructureStrategy(StrEnum):
    """How section hierarchy is recovered for a document.

    toc       PDF bookmarks give the path; body headings add one level below it.
    numbered  Numbered headings ("1.9.2 Scorpion sting") set depth by their numbering;
              unnumbered headings nest one level below the current numbered heading.
    markdown  Heading depth follows the layout model's heading size (# / ##).
    running   The running page header/footer names the top-level section
              (e.g. mhGAP modules); body headings nest below it.
    caps_topics
              ALL-CAPS headings are page-level topics ("FEVER", "DIABETES") whatever
              their markdown depth; every other heading nests one level below the topic.
    """

    TOC = "toc"
    NUMBERED = "numbered"
    MARKDOWN = "markdown"
    RUNNING = "running"
    CAPS_TOPICS = "caps_topics"


class DocumentManifest(BaseModel):
    doc_id: str
    file: str
    title: str
    publisher: str
    document_type: str
    population: str  # adult | child | all
    publication_date: str | None = None
    source_url: str | None = None
    isbn: str | None = None
    topics: list[str] = Field(default_factory=list)
    structure: StructureStrategy
    # 1-based page ranges excluded entirely (covers, credits, cover-art algorithms, indexes).
    skip_pages: list[tuple[int, int]] = Field(default_factory=list)
    # Running-header text -> canonical section name, for headers the PDF renders inconsistently.
    section_aliases: dict[str, str] = Field(default_factory=dict)

    def skips(self, page_number: int) -> bool:
        return any(start <= page_number <= end for start, end in self.skip_pages)


class Manifest(BaseModel):
    documents: list[DocumentManifest]

    @classmethod
    def load(cls, path: Path) -> Manifest:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def get(self, doc_id: str) -> DocumentManifest:
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        raise KeyError(doc_id)
