"""Metadata filters for retrieval.

One backend-neutral model, translated to a Qdrant filter for dense search and
evaluated in Python (``matches``) for BM25, so both retrievers always see exactly
the same subset of the knowledge base. Only ``retrievable`` chunks (no references,
front matter, contents) are ever returned.

Population is a *safety* constraint (child and adult doses differ), topic is a
*relevance* hint. ``relaxed()`` therefore drops topic but never population.
"""

from __future__ import annotations

from pydantic import BaseModel
from qdrant_client import models

from app.rag.ingestion.classify import infer_topics
from app.schemas.clinical import ChunkType, ClinicalIntake, DocumentMetadata

ADULT_AGE_FROM = 18  # the pocket books cover newborns through adolescents (to 17)


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
        the constraint that defines *who* the answer is for (population)."""
        return RetrievalFilters(population=self.population)

    def matches(self, meta: DocumentMetadata) -> bool:
        """Python equivalent of ``to_qdrant`` (used by the BM25 retriever)."""
        if not meta.retrievable:
            return False
        if self.population and meta.population not in self.population:
            return False
        if self.topics and not set(self.topics) & set(meta.topics):
            return False
        if self.document_types and meta.document_type not in self.document_types:
            return False
        if self.document_ids and meta.document_id not in self.document_ids:
            return False
        return not (self.chunk_types and meta.chunk_type not in self.chunk_types)

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


def is_pregnant(status: str | None) -> bool:
    if not status:
        return False
    s = status.lower()
    if s.startswith(("no", "not ")) or " not " in s or "n't" in s:
        return False
    return "pregnan" in s or s in {"yes", "true", "y"}


def population_for(age: int | None, pregnancy_status: str | None = None) -> list[str] | None:
    """Corpus populations relevant to a caller. Unknown age -> no population restriction
    (the intake agent should ask for age early, since the corpus is ~45% paediatric)."""
    if age is None:
        return None
    populations = ["child", "all"] if age < ADULT_AGE_FROM else ["adult", "all"]
    if is_pregnant(pregnancy_status):
        populations.append("pregnancy")
    return populations


def filters_from_clinical(
    intake: ClinicalIntake, *, include_topics: bool = True
) -> RetrievalFilters:
    """Retrieval filters implied by what the caller has told us so far."""
    topics: list[str] | None = None
    if include_topics:
        words = " ".join(
            [intake.chief_complaint or "", *intake.symptoms, *intake.associated_symptoms]
        )
        topics = infer_topics(words) or None
    return RetrievalFilters(
        population=population_for(intake.age, intake.pregnancy_status), topics=topics
    )
