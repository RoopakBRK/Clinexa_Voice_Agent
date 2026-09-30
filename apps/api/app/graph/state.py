"""LangGraph state for a voice clinical conversation.

Phase 1 only populates the transcript; the graph nodes arrive in Phase 4.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.clinical import (
    ClinicalIntake,
    Evidence,
    Intent,
    ResponseAction,
    RetrievedChunk,
    Role,
    SafetyAssessment,
)


class AgentName(StrEnum):
    SESSION_INIT = "session_init"
    CONVERSATION_MANAGER = "conversation_manager"
    CLINICAL_INTAKE = "clinical_intake"
    CLINICAL_RAG = "clinical_rag"
    SAFETY = "safety"
    MEDICATION = "medication"
    RESPONSE_PLANNER = "response_planner"
    ESCALATION = "escalation"


class ConversationMessage(BaseModel):
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stt_confidence: float | None = None
    # Set on assistant turns cut off by caller barge-in (Phase 3).
    interrupted: bool = False


class VoiceClinicalState(BaseModel):
    call_id: str
    patient_id: str | None = None

    conversation_history: list[ConversationMessage] = Field(default_factory=list)
    current_transcript: str = ""

    intent: Intent | None = None
    clinical_state: ClinicalIntake = Field(default_factory=ClinicalIntake)
    missing_information: list[str] = Field(default_factory=list)

    retrieved_documents: list[RetrievedChunk] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    safety_assessment: SafetyAssessment | None = None

    current_agent: AgentName | None = None
    next_action: ResponseAction | None = None
    response_text: str | None = None

    interruption_detected: bool = False
    escalation_required: bool = False

    latency_metrics: dict[str, float] = Field(default_factory=dict)
