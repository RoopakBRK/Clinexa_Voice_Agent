"""Messages exchanged with the Clinexsa website's onboarding page.

Text frames are JSON (the models below); binary frames are raw audio.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from app.core.logging import get_logger

log = get_logger(__name__)


class StartMessage(BaseModel):
    type: Literal["start"]
    onboarding_id: str = Field(min_length=8, max_length=64)
    language: str = "en"
    first_name: str | None = Field(default=None, max_length=80)
    # The page's current form. Only used to tell a fresh start from a reconnect.
    form_state: dict[str, Any] | None = None


class PauseMessage(BaseModel):
    type: Literal["pause"]


class ResumeMessage(BaseModel):
    type: Literal["resume"]


class StopMessage(BaseModel):
    type: Literal["stop"]


class ToolResultMessage(BaseModel):
    type: Literal["tool_result"]
    id: str
    content: str = "ok"


class FormEditMessage(BaseModel):
    type: Literal["form_edit"]
    field: str = Field(max_length=80)
    value: Any = None


InboundMessage = Annotated[
    StartMessage | PauseMessage | ResumeMessage | StopMessage | ToolResultMessage | FormEditMessage,
    Field(discriminator="type"),
]

_adapter: TypeAdapter[InboundMessage] = TypeAdapter(InboundMessage)


def parse_web_inbound(raw: str) -> InboundMessage | None:
    """Parse one JSON text frame; unknown or malformed messages are dropped."""
    try:
        return _adapter.validate_python(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        # The payload can hold health details, so only its size is logged.
        log.warning("web.invalid_message", chars=len(raw))
        return None
