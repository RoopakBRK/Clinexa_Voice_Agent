"""Reply generation for a caller turn: a streamed Claude call that can look things up.

The conversation so far goes in and the next spoken reply streams out. On the way Claude
may call two tools (app/tools/knowledge.py): the Indian medicines catalogue, for a
medicine's name, and the WHO guidance, for health information. Each lookup is one more
round with the model, so a reply is a short loop:

    caller turn ─▶ Claude ─▶ text ─────────────────────────────▶ spoken
                      └─▶ tool calls ─▶ lookups ─▶ Claude ─▶ text ─▶ spoken

This is the whole "agent" until the LangGraph state machine (Phase 4) takes over. There
is no intake extraction or red-flag policy here yet, so the system prompt holds the
safety scope: no diagnosis, no prescribing, emergencies before anything else.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import anthropic
from anthropic.types.beta import (
    BetaContentBlock,
    BetaMessageParam,
    BetaOutputConfigParam,
    BetaToolResultBlockParam,
    BetaToolUseBlock,
)

from app.core.config import Settings
from app.core.logging import get_logger
from app.graph.state import ConversationMessage, Lookup
from app.observability.tracing import detached_span
from app.schemas.clinical import Evidence
from app.tools.knowledge import LOOKUP_MEDICINE, SEARCH_GUIDELINES, KnowledgeTools

log = get_logger(__name__)

# `fallbacks="default"` needs this exact beta; the array form uses a different one.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """\
You are Clinexa, an automated health information assistant talking with a caller on the phone.

Your job is to listen to the caller's health concern, ask about it, give general health \
information, and help them work out a sensible next step: looking after themselves at home, \
seeing a clinician, or getting emergency care.{job}

Limits that always apply:
- You are not a doctor. You do not diagnose, you do not prescribe, and you do not tell a caller \
to start, stop or change the dose of a medicine. If they ask for any of that, say plainly that a \
clinician or pharmacist needs to decide it, and offer what you can do instead.
- If anything the caller describes could be an emergency, such as chest pain, trouble breathing, \
signs of a stroke, heavy bleeding, a seizure, someone who is unconscious or very hard to wake, a \
very unwell baby, or thoughts of suicide or self-harm, tell them straight away to call \
{emergency} or go to the nearest emergency department. Say this before you ask anything else\
{before_lookups}.
- When you are not sure, say so and recommend seeing a clinician. A wrong answer on a health \
line can hurt someone, so never guess.
{knowledge}
How to speak: everything you write is read aloud by a text-to-speech voice. Use plain spoken \
sentences with no lists, headings, symbols, abbreviations or emoji. Keep each turn to one to \
three short sentences and ask only one question at a time, because a caller cannot scroll back. \
What you hear is an automatic transcript, so if it does not make sense, ask the caller to say \
it again rather than guessing what they meant.

The caller has already heard this greeting, so do not repeat it or introduce yourself again:
"{greeting}"
"""

_MEDICINES = """
Medicine names:
- When the caller names a medicine, look it up with lookup_medicine before you say anything \
about it. Medicine names are what an automatic transcript gets wrong most, and many of them \
sound alike, so go by what the catalogue returns and not by your memory of the name.
- Tell the caller the name the catalogue gives and, where it says, what the medicine contains. \
If it lists several products, name up to three and ask which is written on their strip or \
bottle. If it only has a name that sounds like what you heard, say that name back, ask if it is \
the one, and say nothing more about the medicine until they confirm. If they say it is not, do \
not offer it again: their medicine is one the catalogue does not have.
- If the name is not in the catalogue, say so. Not every medicine is in it, so ask the caller \
to spell the name or read it from the pack. Never offer a different medicine in its place.
- You can say what a medicine is called and what it contains. Whether this caller should take \
it, and how much, is for a clinician or pharmacist.
"""

_GUIDELINES = """
Health guidance:
- Before you give health information about a symptom, an illness, care at home, warning signs \
or what treatment guidance recommends, search with search_guidelines and base what you say on \
the passages it returns.
- Guidance for children, for adults and in pregnancy differs. If you do not yet know the age of \
the person the question is about, ask before you search, then pass their age with the search, \
and that they are pregnant if the caller has told you so.
- Say where the information comes from in a few words, such as "World Health Organization \
guidance says". Use only what the passages say. If they do not answer the question, say you do \
not have guidance on it and suggest a clinician or pharmacist. Do not fill the gap from memory.
- The passages are written for health workers and may give doses and treatment schedules. You \
may say which medicine the guidance recommends, but do not read out or work out a dose for the \
caller: a clinician or pharmacist confirms the right dose for them.
"""

_BOTH = """\
- When guidance names a medicine, you can look that name up too, to tell the caller how it is \
listed in India: on the essential medicines list, as a Jan Aushadhi generic, or under a brand.
"""

_LOOKUPS = """
A lookup takes a moment. While it runs the caller hears a short line such as "Let me check \
that for you", so do not announce a lookup yourself, and do not repeat that line afterwards. \
Look up only what the caller's question needs.
"""

_JOB = {
    (True, True): " You can check a medicine's name against the Indian medicines catalogue and "
    "look up World Health Organization guidance, and you use both rather than your memory.",
    (True, False): " You can check a medicine's name against the Indian medicines catalogue, "
    "and you use it rather than your memory of the name.",
    (False, True): " You can look up World Health Organization guidance, and you use it rather "
    "than your memory.",
    (False, False): "",
}


def build_system_prompt(*, emergency: str, greeting: str, tools: Sequence[str] = ()) -> str:
    """The prompt for a call, naming only the lookups this server can actually make."""
    medicines, guidelines = LOOKUP_MEDICINE in tools, SEARCH_GUIDELINES in tools
    knowledge = ""
    if medicines:
        knowledge += _MEDICINES
    if guidelines:
        knowledge += _GUIDELINES
    if medicines and guidelines:
        knowledge += _BOTH
    if medicines or guidelines:
        knowledge += _LOOKUPS
    return SYSTEM_PROMPT.format(
        job=_JOB[medicines, guidelines],
        emergency=emergency,
        before_lookups=" and before you look anything up" if knowledge else "",
        knowledge=knowledge,
        greeting=greeting,
    )


class ReplyError(Exception):
    """The reply could not be generated (or was cut short); the caller hears a fallback."""


class ReplyRefused(ReplyError):
    """The model declined to answer."""


@dataclass
class ReplyTrace:
    """What a reply looked up on its way to being spoken, for the call record."""

    lookups: list[Lookup] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)


class ReplyGenerator(ABC):
    name: str

    @abstractmethod
    def stream_reply(
        self, history: Sequence[ConversationMessage], trace: ReplyTrace | None = None
    ) -> AsyncIterator[str]:
        """Yield the assistant's next reply as text deltas. Raises ``ReplyError``.

        ``trace``, if given, is filled in with whatever the reply looked up.
        """

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


def _after_fallback(content: Sequence[BetaContentBlock]) -> int:
    """Where the answering model's own blocks begin.

    When a reply is declined part-way and re-run on the fallback model, ``content`` holds
    both attempts with a ``fallback`` block between them.
    """
    return max((i + 1 for i, block in enumerate(content) if block.type == "fallback"), default=0)


def _to_echo(content: Sequence[BetaContentBlock]) -> list[BetaContentBlock]:
    """An assistant turn as it is sent back with its tool results.

    Blocks go back unchanged, thinking included. The exception is a turn that fell back
    part-way: of the declined attempt only the text is echoed.
    """
    start = _after_fallback(content)
    return [block for i, block in enumerate(content) if i >= start or block.type == "text"]


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
        tools: KnowledgeTools | None = None,
        max_tool_rounds: int = 3,
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
        # No tools to offer is the same as none: the reply is then a single call.
        self._tools = tools if tools is not None and tools.definitions else None
        self._max_tool_rounds = max_tool_rounds

    async def stream_reply(
        self, history: Sequence[ConversationMessage], trace: ReplyTrace | None = None
    ) -> AsyncIterator[str]:
        messages = to_messages(history)
        if not messages or messages[-1]["role"] != "user":
            raise ReplyError("no caller turn to reply to")

        tools = self._tools
        spoken = False
        for round_number in range(self._max_tool_rounds + 1):
            # Past the last lookup the model has to answer with what it has.
            answer_now = tools is not None and round_number == self._max_tool_rounds
            ends_in_space = True
            # Not made the current span: this generator yields while the request is open.
            with detached_span("llm round", model=self._model, round=round_number) as span:
                try:
                    async with self._client.beta.messages.stream(
                        model=self._model,
                        max_tokens=self._max_tokens,
                        system=self._system,
                        messages=messages,
                        tools=tools.definitions if tools else anthropic.omit,
                        tool_choice={"type": "none"} if answer_now else anthropic.omit,
                        # Caches the growing conversation prefix once it is long enough to qualify.
                        cache_control={"type": "ephemeral"},
                        output_config=self._output_config,
                        betas=[_FALLBACK_BETA] if self._refusal_fallback else anthropic.omit,
                        fallbacks="default" if self._refusal_fallback else anthropic.omit,
                    ) as stream:
                        async for text in stream.text_stream:
                            if text:
                                spoken, ends_in_space = True, text[-1].isspace()
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
                except ValueError as exc:
                    # Tool input streams as it is written, and this one was not readable JSON.
                    raise ReplyError("Claude sent a tool call that could not be read") from exc
                span.set_attributes(
                    {
                        "stop_reason": final.stop_reason or "unknown",
                        "input_tokens": final.usage.input_tokens,
                        "output_tokens": final.usage.output_tokens,
                        "cache_read_tokens": final.usage.cache_read_input_tokens or 0,
                    }
                )

            if final.stop_reason == "refusal":
                category = final.stop_details.category if final.stop_details else None
                raise ReplyRefused(f"model declined to answer (category={category})")
            log.debug(
                "llm.reply",
                model=final.model,
                stop_reason=final.stop_reason,
                round=round_number,
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                cache_read_tokens=final.usage.cache_read_input_tokens,
            )
            # Anything but "tool_use" ends the reply. That includes a reply cut off at
            # max_tokens part-way through a tool call: its input may be incomplete, so it
            # is never run.
            if tools is None or final.stop_reason != "tool_use":
                return
            calls = [
                block
                for block in final.content[_after_fallback(final.content) :]
                if block.type == "tool_use"
            ]
            if not calls:
                return

            # The caller hears something while the lookups run: the model's own words if
            # it said any (pushed out now, not held for the sentence after), or a short line.
            if not spoken:
                spoken = True
                yield f"{tools.filler([call.name for call in calls])} "
            elif not ends_in_space:
                yield " "

            outcomes = await asyncio.gather(*(self._look_up(tools, call, trace) for call in calls))
            # Both turns are added together, so no tool call is ever left without its result.
            messages.append({"role": "assistant", "content": _to_echo(final.content)})
            messages.append({"role": "user", "content": outcomes})

        raise ReplyError("the model kept calling tools instead of answering")

    @staticmethod
    async def _look_up(
        tools: KnowledgeTools, call: BetaToolUseBlock, trace: ReplyTrace | None
    ) -> BetaToolResultBlockParam:
        arguments: dict[str, Any] = call.input if isinstance(call.input, dict) else {}
        outcome = await tools.run(call.name, arguments)
        if trace is not None:
            trace.lookups.append(
                Lookup(
                    tool=call.name,
                    detail=outcome.detail,
                    ok=not outcome.is_error,
                    duration_ms=outcome.duration_ms,
                )
            )
            trace.evidence.extend(outcome.evidence)
        result: BetaToolResultBlockParam = {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": outcome.content,
        }
        if outcome.is_error:
            result["is_error"] = True
        return result


def build_reply_generator(
    settings: Settings, tools: KnowledgeTools | None = None
) -> ReplyGenerator | None:
    if settings.anthropic_api_key is None:
        log.warning("llm.not_configured", hint="set ANTHROPIC_API_KEY")
        return None
    return ClaudeReplyGenerator(
        settings.anthropic_api_key.get_secret_value(),
        system_prompt=build_system_prompt(
            emergency=settings.emergency_number_phrase,
            greeting=settings.greeting_text,
            tools=tools.names if tools else (),
        ),
        model=settings.anthropic_model,
        effort=settings.llm_effort,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
        refusal_fallback=settings.llm_refusal_fallback,
        tools=tools,
        max_tool_rounds=settings.llm_max_tool_rounds,
    )
