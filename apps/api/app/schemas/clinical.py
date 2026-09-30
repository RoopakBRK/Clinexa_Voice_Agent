"""Clinical domain schemas shared by agents, RAG, safety and persistence.

These are contracts, not behaviour: agents produce/consume them as structured
LLM outputs, and the policy layer enforces the safety invariants encoded here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, model_validator


class Intent(StrEnum):
    SYMPTOM_INTAKE = "SYMPTOM_INTAKE"
    GENERAL_HEALTH_INFORMATION = "GENERAL_HEALTH_INFORMATION"
    MEDICATION_INFORMATION = "MEDICATION_INFORMATION"
    FOLLOW_UP = "FOLLOW_UP"
    APPOINTMENT_REQUEST = "APPOINTMENT_REQUEST"
    EMERGENCY_CONCERN = "EMERGENCY_CONCERN"
    UNKNOWN = "UNKNOWN"


class Urgency(StrEnum):
    ROUTINE = "routine"
    SOON = "soon"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class ResponseAction(StrEnum):
    """Outcome chosen by the response policy layer (never by the LLM alone)."""

    CONTINUE_INTAKE = "CONTINUE_INTAKE"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    PROVIDE_INFORMATION = "PROVIDE_INFORMATION"
    RECOMMEND_CLINICIAN = "RECOMMEND_CLINICIAN"
    URGENT_ESCALATION = "URGENT_ESCALATION"
    EMERGENCY_ESCALATION = "EMERGENCY_ESCALATION"


class ClinicalIntake(BaseModel):
    """Structured intake collected by the Clinical Intake Agent. Not a diagnosis."""

    CORE_FIELDS: ClassVar[tuple[str, ...]] = ("chief_complaint", "duration", "severity")

    chief_complaint: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    duration: str | None = None
    onset: str | None = None
    severity: str | None = None
    location: str | None = None
    frequency: str | None = None
    associated_symptoms: list[str] = Field(default_factory=list)
    aggravating_factors: list[str] = Field(default_factory=list)
    relieving_factors: list[str] = Field(default_factory=list)
    existing_conditions: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    age: int | None = Field(default=None, ge=0, le=130)
    sex: str | None = None
    pregnancy_status: str | None = None
    relevant_history: list[str] = Field(default_factory=list)

    def missing(self, fields: tuple[str, ...] = CORE_FIELDS) -> list[str]:
        """Names of the given fields that are still unanswered (None or empty)."""
        return [f for f in fields if getattr(self, f) in (None, "", [])]


class ChunkType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    RECOMMENDATION = "recommendation"
    WARNING = "warning"  # danger signs / urgent-referral content


class DocumentMetadata(BaseModel):
    """Per-chunk metadata; also the Qdrant payload used for filtering."""

    source: str
    document_title: str
    document_type: str
    section: str | None = None
    subsection: str | None = None
    page_number: int | None = None
    topic: str | None = None
    population: str | None = None
    publication_date: str | None = None
    source_url: str | None = None
    document_id: str | None = None
    publisher: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    page_end: int | None = None
    chunk_type: ChunkType = ChunkType.TEXT
    topics: list[str] = Field(default_factory=list)
    # False for front matter, references, contents pages etc. (kept for audit, not indexed)
    retrievable: bool = True


class RetrievalScores(BaseModel):
    """Scores from each retrieval stage, kept for observability and evaluation."""

    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: DocumentMetadata
    scores: RetrievalScores = Field(default_factory=RetrievalScores)


class Evidence(BaseModel):
    """A piece of WHO evidence selected to ground a response."""

    document_id: str
    document_title: str
    section: str
    page_number: int | None = None
    excerpt: str
    relevance_score: float


class SafetyAssessment(BaseModel):
    red_flags_detected: list[str] = Field(default_factory=list)
    urgency: Urgency = Urgency.ROUTINE
    evidence_chunks: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    escalation_required: bool = False

    @model_validator(mode="after")
    def _urgent_implies_escalation(self) -> SafetyAssessment:
        # Fail-safe invariant: an urgent/emergency assessment always escalates,
        # regardless of what a model returned for escalation_required.
        if self.urgency in (Urgency.URGENT, Urgency.EMERGENCY):
            self.escalation_required = True
        return self


class EscalationReason(StrEnum):
    USER_REQUESTED_HUMAN = "user_requested_human"
    RED_FLAG = "red_flag"
    EMERGENCY_CONCERN = "emergency_concern"
    LOW_CONFIDENCE = "low_confidence"
    REPEATED_MISUNDERSTANDING = "repeated_misunderstanding"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SENSITIVE_SITUATION = "sensitive_situation"
    MEDICATION_CHANGE_REQUEST = "medication_change_request"


class EscalationSummary(BaseModel):
    reason: EscalationReason
    chief_complaint: str
    symptoms: list[str] = Field(default_factory=list)
    duration: str | None = None
    relevant_history: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    conversation_summary: str
    recommended_next_step: str


class CallSummary(BaseModel):
    call_id: str
    chief_complaint: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    duration: str | None = None
    relevant_history: list[str] = Field(default_factory=list)
    questions_asked: list[str] = Field(default_factory=list)
    information_provided: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    escalated: bool = False
    recommended_next_step: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


Role = Literal["patient", "assistant", "system"]
