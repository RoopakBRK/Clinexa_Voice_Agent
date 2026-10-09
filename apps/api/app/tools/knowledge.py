"""What Clinexa can look up while it is on a call.

    lookup_medicine     a medicine's name, in the Indian medicines catalogue (app/medicines)
    search_guidelines   WHO primary-care guidance (app/rag/retrieval/service.py)

Claude decides when to call them. This module is the only way from the reply generator
to either store: it checks what Claude asked for, runs the lookup, and words the answer
for a model that will say it aloud. A lookup that fails comes back as an answer that
says so, never as an exception: a caller is waiting on the other end.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from anthropic.types.beta import BetaToolParam
from pydantic import BaseModel, Field, ValidationError

from app.core.logging import get_logger
from app.medicines.lookup import Match
from app.observability.tracing import tracer
from app.rag.retrieval.service import GuidelineResult
from app.schemas.clinical import Evidence, RetrievedChunk

log = get_logger(__name__)

LOOKUP_MEDICINE = "lookup_medicine"
SEARCH_GUIDELINES = "search_guidelines"

# Spoken while a lookup runs, if the reply has not said anything yet.
_FILLERS = {
    LOOKUP_MEDICINE: "Let me check that medicine name.",
    SEARCH_GUIDELINES: "Let me check the guidance on that.",
}
_FILLER = "Let me check that for you."

# How much of a passage is kept for the call record (the model is given all of it).
_EXCERPT_CHARS = 280


class MedicineFinder(Protocol):
    """The medicines catalogue, as the tools need it (app/medicines/lookup.py)."""

    async def find(self, heard: str) -> Match | None: ...


class GuidelineFinder(Protocol):
    """The guidance search, as the tools need it (app/rag/retrieval/service.py)."""

    async def search(
        self, query: str, *, age: int | None = None, pregnant: bool | None = None
    ) -> GuidelineResult | None: ...


@dataclass(frozen=True)
class ToolOutcome:
    content: str  # what Claude is told
    is_error: bool = False
    # A word or two for logs and the call record. Never what the caller said.
    detail: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    duration_ms: float = 0.0


class _MedicineInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class _GuidelineInput(BaseModel):
    query: str = Field(min_length=2, max_length=400)
    age_years: int | None = Field(default=None, ge=0, le=130)
    pregnant: bool | None = None


_MEDICINE_TOOL: BetaToolParam = {
    "name": LOOKUP_MEDICINE,
    "description": (
        "Look a medicine's name up in the Indian medicines catalogue: the National List of "
        "Essential Medicines 2022, the Jan Aushadhi product list and the A to Z medicines "
        "of India, about 250,000 names. Call it whenever the caller names a medicine, brand "
        "or generic, before you say anything about that medicine: what you hear is an "
        "automatic transcript, and medicine names are what it gets wrong most. It returns "
        "the catalogue's spelling of the name, what the medicine contains, and its strength "
        "and form where the catalogue is sure, or the products the name could mean. You "
        "can also call it with a generic name that guidance mentions, to tell the caller "
        "how that medicine is listed in India."
    ),
    # The client checks the input itself (_MedicineInput), so it can stream as written.
    "eager_input_streaming": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "The medicine name as you heard it, with numbers as digits, for example "
                    '"Dolo 650" or "glycomet 500". One medicine in each call.'
                ),
            }
        },
        "required": ["name"],
    },
}

_GUIDELINE_TOOL: BetaToolParam = {
    "name": SEARCH_GUIDELINES,
    "description": (
        "Search the health guidance Clinexa holds: World Health Organization guidelines "
        "and pocket books for primary care, covering common illnesses in children, "
        "adolescents and adults, malaria, mental health and infection. Call it before you "
        "give health information about a symptom, an illness, care at home, danger signs, "
        "or which medicines guidance recommends. It returns up to five passages, each "
        "with the document and section it is from. Do not call it when the caller may be "
        "describing an emergency: tell them to get emergency help first."
    ),
    "eager_input_streaming": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to look for, as a short question or phrase, for example "
                    '"fever in a child under five, when to seek care".'
                ),
            },
            "age_years": {
                "type": "integer",
                "description": (
                    "The age in years of the person the question is about, if the caller "
                    "has said it. Guidance for children and for adults differs, so "
                    "passages written for the other group are left out."
                ),
            },
            "pregnant": {
                "type": "boolean",
                "description": "True if the person is pregnant, where the caller has said so.",
            },
        },
        "required": ["query"],
    },
}


def describe_match(match: Match) -> str:
    """What Claude is told about a medicine name, to say back to the caller."""
    facts = _facts(match, each=match.status == "several" and len(match.choices) > 1)
    if match.status == "exact":
        return (
            f'The catalogue has this medicine as "{match.name}".{facts} '
            "Say the name back so the caller can confirm it is the one they mean."
        )
    if match.status == "several" and len(match.choices) == 1:
        return (
            f'The catalogue does not have "{match.name}" just as said. The closest product '
            f'is "{match.choices[0]}".{facts} Ask the caller whether that is what is written '
            "on their strip or bottle."
        )
    if match.status == "several":
        return (
            f'The catalogue has more than one product for "{match.name}": '
            f"{'; '.join(match.choices)}.{facts} If it matters which one the caller has, "
            "ask what is written on their strip or bottle. Do not choose for them."
        )
    if match.status == "close":
        others = f" It could also be: {'; '.join(match.choices)}." if match.choices else ""
        return (
            f'"{match.heard}" is not in the catalogue. "{match.name}" is, and sounds like '
            f"it.{others} This is a guess. Say the name back and ask the caller whether it "
            "is their medicine. Say nothing else about it until they confirm, then look "
            "that name up."
        )
    near = (
        f" Names in the catalogue a little like it: {'; '.join(match.choices)}. Mention one "
        "only if the caller says that is their medicine."
        if match.choices
        else ""
    )
    return (
        f'"{match.heard}" is not in the catalogue.{near} The catalogue does not hold every '
        "medicine, so the name may still be right. Say you could not find it, and ask the "
        "caller to spell it or read it from the strip or bottle. Do not suggest another "
        "medicine in its place."
    )


def _facts(match: Match, *, each: bool = False) -> str:
    """What the catalogue settles about the medicine, as sentences. Empty if nothing.

    Where the name could mean several products, only what they all share is known.
    """
    it = "Each" if each else "It"
    facts = []
    if match.composition and match.composition != match.name:
        facts.append(f"{it} contains: {match.composition}.")
    if match.strength:
        facts.append(f"Strength: {match.strength}.")
    if match.unit:
        facts.append(f"{it} comes as: {match.unit}.")
    if match.source:
        facts.append(f"Listed in: {match.source}.")
    return " " + " ".join(facts) if facts else ""


def _source(chunk: RetrievedChunk) -> str:
    meta = chunk.metadata
    published = [part for part in (meta.publisher, (meta.publication_date or "")[:4]) if part]
    where = [
        f"{meta.document_title} ({', '.join(published)})" if published else meta.document_title
    ]
    if meta.heading_path:
        where.append(" > ".join(meta.heading_path))
    if meta.page_number is not None:
        where.append(f"page {meta.page_number}")
    if meta.population and meta.population != "all":
        where.append(f"written for: {meta.population}")
    return " | ".join(where)


def describe_passages(result: GuidelineResult) -> str:
    """The passages a search found, each under the source it is from."""
    if not result.passages:
        return (
            "Nothing in the guidance matched. Tell the caller you do not have guidance on "
            "this, and suggest they ask a clinician or pharmacist."
        )
    if result.population is None:
        who = (
            "The person's age was not given, so these may be written for children or for "
            "adults: each passage says which. If the answer depends on age, ask before "
            "you use them."
        )
    else:
        who = f"Limited to guidance written for: {', '.join(result.population)}."
    passages = "\n\n".join(
        f"[{number}] {_source(chunk)}\n{chunk.text}"
        for number, chunk in enumerate(result.passages, start=1)
    )
    return (
        f"Passages from the guidance, most relevant first. {who} Use only what they say. "
        "If they do not answer the caller's question, say you do not have guidance on it.\n\n"
        f"{passages}"
    )


def _evidence(chunk: RetrievedChunk) -> Evidence:
    meta, scores = chunk.metadata, chunk.scores
    score = scores.reranker_score if scores.reranker_score is not None else scores.rrf_score
    return Evidence(
        document_id=meta.document_id or meta.source,
        document_title=meta.document_title,
        section=" > ".join(meta.heading_path) or (meta.section or ""),
        page_number=meta.page_number,
        excerpt=chunk.text[:_EXCERPT_CHARS],
        relevance_score=score or 0.0,
    )


class KnowledgeTools:
    def __init__(
        self,
        *,
        medicines: MedicineFinder | None = None,
        guidelines: GuidelineFinder | None = None,
    ) -> None:
        self._medicines = medicines
        self._guidelines = guidelines
        self.definitions: list[BetaToolParam] = [
            tool
            for tool, backend in ((_MEDICINE_TOOL, medicines), (_GUIDELINE_TOOL, guidelines))
            if backend is not None
        ]

    @property
    def names(self) -> list[str]:
        return [tool["name"] for tool in self.definitions]

    def filler(self, tools: Sequence[str]) -> str:
        """A few words for the caller to hear while these lookups run."""
        return _FILLERS.get(tools[0], _FILLER) if len(set(tools)) == 1 else _FILLER

    async def run(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
        # The span carries the tool and how it went. Never the arguments.
        with tracer.start_as_current_span(f"tool {name}", attributes={"tool": name}) as span:
            outcome = await self._run(name, arguments)
            span.set_attributes({"ok": not outcome.is_error, "detail": outcome.detail})
            return outcome

    async def _run(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
        started = time.perf_counter()
        try:
            if name == LOOKUP_MEDICINE and self._medicines is not None:
                outcome = await self._lookup_medicine(_MedicineInput.model_validate(arguments))
            elif name == SEARCH_GUIDELINES and self._guidelines is not None:
                outcome = await self._search_guidelines(_GuidelineInput.model_validate(arguments))
            else:
                outcome = ToolOutcome(f"There is no tool called {name}.", True, "unknown_tool")
        except ValidationError as exc:
            # What Claude sent did not fit the tool. Say what is wrong so it can ask again.
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
            outcome = ToolOutcome(f"Invalid input. {problems}", True, "invalid_input")
        took = round((time.perf_counter() - started) * 1000, 2)
        # Arguments are health details: log the tool and how it went, nothing more.
        log.info("tool.call", tool=name, detail=outcome.detail, ok=not outcome.is_error, ms=took)
        return ToolOutcome(
            outcome.content, outcome.is_error, outcome.detail, outcome.evidence, took
        )

    async def _lookup_medicine(self, asked: _MedicineInput) -> ToolOutcome:
        assert self._medicines is not None
        match = await self._medicines.find(asked.name)
        if match is None:
            return ToolOutcome(
                "The medicines catalogue could not be reached just now. Tell the caller you "
                "cannot check the name at the moment, and do not guess what the medicine is.",
                True,
                "unavailable",
            )
        return ToolOutcome(describe_match(match), detail=match.status)

    async def _search_guidelines(self, asked: _GuidelineInput) -> ToolOutcome:
        assert self._guidelines is not None
        result = await self._guidelines.search(
            asked.query, age=asked.age_years, pregnant=asked.pregnant
        )
        if result is None:
            return ToolOutcome(
                "The guidance could not be searched just now. Tell the caller you cannot "
                "look this up at the moment, and suggest they speak to a clinician or "
                "pharmacist. Do not answer from memory.",
                True,
                "unavailable",
            )
        return ToolOutcome(
            describe_passages(result),
            detail=f"{len(result.passages)} passages",
            evidence=[_evidence(chunk) for chunk in result.passages],
        )
