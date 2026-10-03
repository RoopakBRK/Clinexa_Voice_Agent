"""Retrieval evaluation dataset: questions + criteria-based gold evidence.

Gold is a *criterion*, not a chunk id ("malaria guideline, pages 175-176, mentions
artemether and lumefantrine"). Chunk ids change whenever chunking changes; criteria
stay valid, and a reviewer can read and check them without looking at the index.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, Field, model_validator

from app.schemas.clinical import DocumentMetadata


class Category(StrEnum):
    SYMPTOM = "symptom"
    CONDITION = "condition"
    RED_FLAG = "red_flag"
    MEDICATION = "medication"
    FOLLOW_UP = "follow_up"
    AMBIGUOUS = "ambiguous"
    OUT_OF_DOMAIN = "out_of_domain"


class GoldUnit(BaseModel):
    """One piece of evidence that a good answer must be grounded in.

    A chunk satisfies the unit when ALL of the given constraints hold:
    an accepted document, page overlap, a section-path substring, every ``must_contain``
    string, and at least one ``any_of`` string. (All string matching is case-insensitive.)

    ``document_id`` and ``section_contains`` may each be a list of alternatives (ANY
    matches), because one fact is often covered equally well by two sources, e.g. a
    child's cough appears in both paediatric pocket books under different headings.
    """

    document_id: str | list[str]
    pages: tuple[int, int] | None = None  # inclusive PDF page range
    section_contains: str | list[str] | None = None
    must_contain: list[str] = Field(default_factory=list)
    any_of: list[str] = Field(default_factory=list)
    note: str | None = None  # why this is the evidence (for the human reviewer)

    @model_validator(mode="after")
    def _has_a_constraint_beyond_document(self) -> Self:
        if not (self.pages or self.section_contains or self.must_contain or self.any_of):
            raise ValueError("a gold unit needs pages, section_contains, must_contain or any_of")
        return self

    @staticmethod
    def _as_list(value: str | list[str] | None) -> list[str]:
        return [] if value is None else [value] if isinstance(value, str) else value

    def matches(self, meta: DocumentMetadata, text: str) -> bool:
        if meta.document_id not in self._as_list(self.document_id):
            return False
        if self.pages is not None:
            start = meta.page_number or 0
            end = meta.page_end or start
            if start > self.pages[1] or end < self.pages[0]:
                return False
        sections = self._as_list(self.section_contains)
        if sections:
            path = " > ".join(meta.heading_path).casefold()
            if not any(sec.casefold() in path for sec in sections):
                return False
        lowered = text.casefold()
        if not all(term.casefold() in lowered for term in self.must_contain):
            return False
        return not (self.any_of and not any(term.casefold() in lowered for term in self.any_of))


class Caller(BaseModel):
    """What the system would already know about the caller (drives metadata filters)."""

    age: int | None = None
    pregnancy_status: str | None = None


class EvalQuestion(BaseModel):
    id: str
    category: Category
    question: str  # exactly as a caller would say it
    # Self-contained form of a follow-up ("and how long do I give it?" -> "How long to give
    # artemether-lumefantrine?"), i.e. what the query rewriter should produce.
    retrieval_query: str | None = None
    history: list[str] = Field(default_factory=list)  # earlier turns, for follow-ups
    caller: Caller | None = None
    expected: Literal["answer", "clarify", "abstain"] = "answer"
    gold: list[GoldUnit] = Field(default_factory=list)
    rationale: str | None = None

    @model_validator(mode="after")
    def _gold_matches_expected_behaviour(self) -> Self:
        if self.expected == "answer" and not self.gold:
            raise ValueError(f"{self.id}: an answerable question needs gold evidence")
        if self.expected != "answer" and self.gold:
            raise ValueError(f"{self.id}: '{self.expected}' questions must not have gold evidence")
        return self

    @property
    def query(self) -> str:
        return self.retrieval_query or self.question


class EvalDataset(BaseModel):
    name: str
    version: str
    questions: list[EvalQuestion]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique")
        return self

    @classmethod
    def load(cls, path: Path) -> EvalDataset:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def by_category(self) -> dict[Category, list[EvalQuestion]]:
        grouped: dict[Category, list[EvalQuestion]] = {c: [] for c in Category}
        for q in self.questions:
            grouped[q.category].append(q)
        return grouped


class GoldProblem(BaseModel):
    question_id: str
    unit_index: int
    severity: Literal["error", "warning"]
    message: str


def validate_gold(
    dataset: EvalDataset,
    corpus: Sequence[tuple[DocumentMetadata, str]],
    *,
    max_matches: int = 40,
) -> list[GoldProblem]:
    """Check every gold unit against the real corpus: a unit that matches nothing can
    never be satisfied (dataset bug); one that matches a huge share of a document is
    too vague to be meaningful evidence."""
    problems: list[GoldProblem] = []
    for q in dataset.questions:
        for i, unit in enumerate(q.gold):
            n = sum(unit.matches(meta, text) for meta, text in corpus if meta.retrievable)
            if n == 0:
                problems.append(
                    GoldProblem(
                        question_id=q.id,
                        unit_index=i,
                        severity="error",
                        message="matches no chunk in the corpus",
                    )
                )
            elif n > max_matches:
                problems.append(
                    GoldProblem(
                        question_id=q.id,
                        unit_index=i,
                        severity="warning",
                        message=f"matches {n} chunks (> {max_matches}): too broad?",
                    )
                )
    return problems
