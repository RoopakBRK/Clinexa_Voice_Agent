"""Roopiee on a medicine reminder call.

The opening line is written here, not by the model: it names the medicine and the
time exactly as they are in the patient's record, with no chance of a slip. After
the person answers, Claude carries the short conversation that follows.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from app.agents.responder import ClaudeReplyGenerator, ReplyGenerator, ReplyTrace
from app.core.config import Settings
from app.graph.state import ConversationMessage
from app.notify.store import CallContext
from app.notify.worker import clock

REMINDER_PROMPT = """\
You are Roopiee, the reminder voice for Clinexsa. You are on a phone call with
{first_name}. You called to remind them about one medicine: {medicine}{when}.
You have already opened the call by saying: "{opening}"

How to talk:
- Everything you write is read aloud. Use plain spoken sentences. No lists, symbols or emoji.
- Warm and patient. One or two short sentences at a time. Speak {language}.
- If they say they have taken it, thank them, wish them well and say goodbye.
- If they have not taken it yet, ask them kindly to take it now, then say goodbye.
- If they say they will take it later, say that is fine and say goodbye.
- Only talk about this reminder. You do not know anything else about their health.

What you must never do:
- Never give medical advice. Never say whether a medicine, a dose or a timing is right.
- Never tell them to skip, stop, double or change a medicine.
- If they ask anything medical, say: "That's a good one for your doctor."
- If they mention chest pain, a fall, trouble breathing or feeling very unwell, say:
  "Please call 112 now, or ask someone near you to help." Do not go back to the reminder.
"""

REMINDER_FALLBACK = "Thank you. Please take care. Goodbye."

_LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
    "ml": "Malayalam",
    "gu": "Gujarati",
}


def opening_line(context: CallContext) -> str:
    when = f" It is your {clock(context.at)} medicine." if context.at else ""
    return (
        f"Hello {context.first_name}, this is Roopiee from Clinexsa with your medicine reminder."
        f"{when} It is time for {context.medicine.label}."
        f" Have you taken {'them' if context.count > 1 else 'it'}?"
    )


class ReminderReplyGenerator(ReplyGenerator):
    """Says the scripted opening first, then hands the conversation to Claude."""

    name = "reminder"

    def __init__(self, context: CallContext, inner: ReplyGenerator | None) -> None:
        self._opening = opening_line(context)
        self._inner = inner

    async def stream_reply(
        self, history: Sequence[ConversationMessage], trace: ReplyTrace | None = None
    ) -> AsyncIterator[str]:
        if not any(message.role == "patient" for message in history):
            yield self._opening
            return
        if self._inner is None:
            # No model is configured: still end the call politely.
            yield REMINDER_FALLBACK
            return
        async for text in self._inner.stream_reply(history, trace):
            yield text


def build_reminder_generator(settings: Settings, context: CallContext) -> ReplyGenerator:
    inner: ReplyGenerator | None = None
    if settings.anthropic_api_key is not None:
        inner = ClaudeReplyGenerator(
            settings.anthropic_api_key.get_secret_value(),
            system_prompt=REMINDER_PROMPT.format(
                first_name=context.first_name,
                medicine=context.medicine.label,
                when=f", due at {clock(context.at)}" if context.at else "",
                opening=opening_line(context),
                language=_LANGUAGE_NAMES.get(context.language.split("-")[0], "English"),
            ),
            model=settings.anthropic_model,
            effort=settings.llm_effort,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.llm_timeout_s,
            refusal_fallback=settings.llm_refusal_fallback,
        )
    return ReminderReplyGenerator(context, inner)
