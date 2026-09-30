"""Metadata filters for retrieval.

One backend-neutral model, translated to a Qdrant filter here and applied to the
BM25 index in Phase 7, so dense and sparse retrieval always see the same subset.
Only ``retrievable`` chunks (no references, front matter, contents) are ever returned.
"""

from __future__ import annotations

from pydantic import BaseModel
from qdrant_client import models

from app.schemas.clinical import ChunkType


class RetrievalFilters(BaseModel):
    """Each field, when set, restricts results to chunks matching ANY listed value."""

    population: list[str] | None = None
    topics: list[str] | None = None
    document_types: list[str] | None = None
    document_ids: list[str] | None = None
    chunk_types: list[ChunkType] | None = None

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.population, self.topics, self.document_types, self.document_ids, self.chunk_types)
        )

    def relaxed(self) -> RetrievalFilters:
        """A looser filter for when the strict one returns too little: keep only
        the constraints that define *who* the answer is for (population)."""
        return RetrievalFilters(population=self.population)

    def to_qdrant(self) -> models.Filter:
        must: list[models.Condition] = [
            models.FieldCondition(key="retrievable", match=models.MatchValue(value=True))
        ]
        for key, values in (
            ("population", self.population),
            ("topics", self.topics),
            ("document_type", self.document_types),
            ("document_id", self.document_ids),
            ("chunk_type", [c.value for c in self.chunk_types] if self.chunk_types else None),
        ):
            if values:
                must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(values))))
        return models.Filter(must=must)
