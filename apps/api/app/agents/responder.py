"""Reply generation for a caller turn: one streamed Claude call (Phase 2).

This is the whole "agent" until the LangGraph state machine (Phase 4) takes over:
the conversation so far goes in, the next spoken reply streams out. There is no
retrieval, intake extraction or red-flag policy here yet, so the system prompt
keeps the assistant to listening, clarifying and conservative signposting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Literal

import anthropic
from anthropic.types.beta import BetaMessageParam, BetaOutputConfigParam

from app.core.config import Settings
from app.core.logging import get_logger
from app.graph.state import ConversationMessage

log = get_logger(__name__)

# `fallbacks="default"` needs this exact beta; the array form uses a different one.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """\
You are Clinexa, an automated health information assistant talking with a caller on the phone.

Your job is to listen to the caller's health concern, ask about it, give general health \
information, and help them work out a sensible next step: looking after themselves at home, \
seeing a clinician, or getting emergency care.

Limits that always apply:
- You are not a doctor. You do not diagnose, you do not prescribe, and you do not tell a caller \
to start, stop or change the dose of a medicine. If they ask for any of that, say plainly that a \
clinician or pharmacist needs to decide it, and offer what you can do instead.
- If anything the caller describes could be an emergency, such as chest pain, trouble breathing, \
signs of a stroke, heavy bleeding, a seizure, someone who is unconscious or very hard to wake, a \
very unwell baby, or thoughts of suicide or self-harm, tell them straight away to call \
{emergency} or go to the nearest emergency department. Say this before you ask anything else.
- When you are not sure, say so and recommend seeing a clinician. A wrong answer on a health \
line can hurt someone, so never guess.

How to speak: everything you write is read aloud by a text-to-speech voice. Use plain spoken \
sentences with no lists, headings, symbols, abbreviations or emoji. Keep each turn to one to \
three short sentences and ask only one question at a time, because a caller cannot scroll back. \
What you hear is an automatic transcript, so if it does not make sense, ask the caller to say \
it again rather than guessing what they meant.

The caller has already heard this greeting, so do not repeat it or introduce yourself again:
"{greeting}"
"""


class ReplyError(Exception):
    """The reply could not be generated (or was cut short); the caller hears a fallback."""


class ReplyRefused(ReplyError):
    """The model declined to answer."""


class ReplyGenerator(ABC):
    name: str

    @abstractmethod
    def stream_reply(self, history: Sequence[ConversationMessage]) -> AsyncIterator[str]:
        """Yield the assistant's next reply as text deltas. Raises ``ReplyError``."""

    def add_note(self, note: str) -> None:  # noqa: B027 - optional hook, not abstract
        """Something happened that the next reply should take into account.

        Ignored by generators that only answer what the caller said.
        """


def to_messages(history: Sequence[ConversationMessage]) -> list[BetaMessageParam]:
    """Conversation history as API messages, starting with the caller's first turn."""
    messages: list[BetaMessageParam] = []
    for message in history:
        if message.role == "system" or not message.content.strip():
            continue
        if message.role == "assistant" and not messages:
            continue  # the API requires the first message to be the user's
        role: Literal["user", "assistant"] = "user" if message.role == "patient" else "assistant"
        messages.append({"role": role, "content": message.content})
    return messages


class ClaudeReplyGenerator(ReplyGenerator):
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        system_prompt: str,
        model: str = "claude-opus-5-5",
        effort: Literal["low", "medium", "high"] | None = "low",
        max_tokens: int = 2048,
        timeout_s: float = 20.0,
        refusal_fallback: bool = True,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        # One retry at most: a caller is waiting in silence while this runs.
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout_s, max_retries=1
        )
        self._system = system_prompt
        self._model = model
        self._output_config: BetaOutputConfigParam | anthropic.Omit = (
            {"effort": effort} if effort else anthropic.omit
        )
        self._max_tokens = max_tokens
        self._refusal_fallback = refusal_fallback

    async def stream_reply(self, history: Sequence[ConversationMessage]) -> AsyncIterator[str]:
        messages = to_messages(history)
        if not messages or messages[-1]["role"] != "user":
            raise ReplyError("no caller turn to reply to")

        try:
            async with self._client.beta.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system,
                messages=messages,
                # Caches the growing conversation prefix once it is long enough to qualify.
                cache_control={"type": "ephemeral"},
                output_config=self._output_config,
                betas=[_FALLBACK_BETA] if self._refusal_fallback else anthropic.omit,
                fallbacks="default" if self._refusal_fallback else anthropic.omit,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        except anthropic.APITimeoutError as exc:
            raise ReplyError("Claude request timed out") from exc
        except anthropic.APIConnectionError as exc:
            raise ReplyError(f"could not reach the Claude API: {exc!r}") from exc
        except anthropic.RateLimitError as exc:
            raise ReplyError("Claude API rate limit reached") from exc
        except anthropic.APIStatusError as exc:
            raise ReplyError(f"Claude API error {exc.status_code}: {exc.message}") from exc

        if final.stop_reason == "refusal":
            category = final.stop_details.category if final.stop_details else None
            raise ReplyRefused(f"model declined to answer (category={category})")
        log.debug(
            "llm.reply",
            model=final.model,
            stop_reason=final.stop_reason,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            cache_read_tokens=final.usage.cache_read_input_tokens,
        )


def build_reply_generator(settings: Settings) -> ReplyGenerator | None:
    if settings.anthropic_api_key is None:
        log.warning("llm.not_configured", hint="set ANTHROPIC_API_KEY")
        return None
    return ClaudeReplyGenerator(
        settings.anthropic_api_key.get_secret_value(),
        system_prompt=SYSTEM_PROMPT.format(
            emergency=settings.emergency_number_phrase, greeting=settings.greeting_text
        ),
        model=settings.anthropic_model,
        effort=settings.llm_effort,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
        refusal_fallback=settings.llm_refusal_fallback,
    )
