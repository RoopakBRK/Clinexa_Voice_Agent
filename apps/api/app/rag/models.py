"""Data models shared by ingestion, indexing and retrieval."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.clinical import DocumentMetadata


class BlockKind(StrEnum):
    HEADING = "heading"
    TEXT = "text"
    LIST_ITEM = "list_item"
    TABLE = "table"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"


class Block(BaseModel):
    """A typed layout region of a page, as classified by the layout model."""

    kind: BlockKind
    text: str
    # Markdown heading depth (#=1, ##=2) for HEADING blocks; 0 otherwise.
    md_level: int = 0


class PageExtraction(BaseModel):
    page_number: int  # 1-based, as printed by PDF viewers
    blocks: list[Block] = Field(default_factory=list)
    # PDF bookmark path active on this page (only for documents with a usable ToC).
    toc_path: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    text: str
    token_count: int
    metadata: DocumentMetadata

    def embedding_text(self) -> str:
        """Text sent to the embedding model: a contextual header, then the chunk.

        The header situates short chunks ("Give ORS 75 ml/kg…") in their document
        and section, which the raw excerpt alone does not carry.
        """
        context = " > ".join([self.metadata.document_title, *self.metadata.heading_path])
        return f"{context}\n{self.text}"
