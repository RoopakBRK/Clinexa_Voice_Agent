"""Clinexsa as the onboarding voice on the website.

Unlike the phone responder, this agent works a form: Claude calls tools, each
call is forwarded to the browser (which owns the form) and the browser's answer
is fed back, until Claude has something to say. Only text is spoken.

One generator serves one session and keeps the whole exchange, tool calls
included, so the model always sees what it already filled in.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any, Literal

import anthropic
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaMessageParam,
    BetaOutputConfigParam,
    BetaToolParam,
    BetaToolResultBlockParam,
)

from app.agents.responder import ReplyError, ReplyGenerator, ReplyRefused, ReplyTrace
from app.core.config import Settings
from app.core.logging import get_logger
from app.graph.state import ConversationMessage

log = get_logger(__name__)

# `fallbacks="default"` needs this exact beta; the array form uses a different one.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# (tool_use id, tool name, arguments) -> the browser's short answer, e.g. "ok".
ToolExecutor = Callable[[str, str, dict[str, Any]], Awaitable[str]]

FIELDS = [
    "full_name",
    "date_of_birth",
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "phone",
    "language",
    "conditions",
    "on_medication",
    "prescribed_by_doctor",
    "doctor_name",
    # DISABLED (blood report upload): "has_blood_report", "report_date",
    # DISABLED (food recommendations): "food_type", "food_avoid",
]
# Permissions and family are set up later, on the patient's dashboard.
# DISABLED (blood report upload): was ["about", "health", "medicines", "report", "food", "done"]
# DISABLED (food recommendations): "food" was between "medicines" and "done".
STEPS = ["about", "health", "medicines", "done"]
UNITS = ["tablets", "capsules", "strips", "bottles", "ml", "injections", "sachets"]
TIMINGS = ["morning", "afternoon", "night", "before_food", "after_food"]

_CONFIDENCE = {
    "type": "number",
    "minimum": 0,
    "maximum": 1,
    "description": "How sure you are of what you heard. Below 0.7 the page asks the person "
    "to check it.",
}
_MEDICATION_PROPERTIES: dict[str, Any] = {
    "name": {"type": "string", "description": "Medicine name as the person said it."},
    "quantity": {
        "type": "integer",
        "minimum": 0,
        "maximum": 101,
        "description": "How many they have left. Use 101 for more than 100.",
    },
    "unit": {"type": "string", "enum": UNITS},
    "strength": {"type": "string", "description": "For example 500 mg."},
    "timing": {"type": "array", "items": {"type": "string", "enum": TIMINGS}},
    "confidence": _CONFIDENCE,
}

TOOLS: list[BetaToolParam] = [
    {
        "name": "set_field",
        "description": (
            "Fill one field of the form. Value by field: full_name, doctor_name "
            "are text. date_of_birth is YYYY-MM-DD (the page works the age out from it). age, height_cm, weight_kg are numbers (convert feet and inches to "
            "centimetres). phone is the ten-digit Indian mobile number as digits. gender is "
            "female, male, other or prefer_not_to_say. language is a code: en, hi, kn, ta, "
            "te, mr, bn, ml or gu. conditions is a list of condition names. on_medication is "
            "yes or no. prescribed_by_doctor is yes, no or some. "
            # DISABLED (blood report upload): "has_blood_report is yes, no or not_sure. "
            # "report_date is YYYY-MM-DD. "
            # DISABLED (food recommendations): "food_type is vegetarian, non_vegetarian or eggetarian. "
            # "food_avoid is text."
            ""
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "enum": FIELDS},
                "value": {"description": "The value, in the form described for this field."},
                "confidence": _CONFIDENCE,
            },
            "required": ["field", "value", "confidence"],
        },
    },
    {
        "name": "set_medication_count",
        "description": "Set how many medicines the person takes. The page adds or removes "
        "empty rows to match.",
        "input_schema": {
            "type": "object",
            "properties": {"count": {"type": "integer", "minimum": 0, "maximum": 20}},
            "required": ["count"],
        },
    },
    {
        "name": "add_medication",
        "description": "Add one medicine. It fills the first empty row, or adds a row.",
        "input_schema": {
            "type": "object",
            "properties": _MEDICATION_PROPERTIES,
            "required": ["name", "quantity", "unit", "confidence"],
        },
    },
    {
        "name": "update_medication",
        "description": "Change a medicine that is already in the list. index counts from 0.",
        "input_schema": {
            "type": "object",
            "properties": {"index": {"type": "integer", "minimum": 0}, **_MEDICATION_PROPERTIES},
            "required": ["index"],
        },
    },
    {
        "name": "go_to_step",
        "description": "Scroll the page to a step and highlight it.",
        "input_schema": {
            "type": "object",
            "properties": {"step": {"type": "string", "enum": STEPS}},
            "required": ["step"],
        },
    },
    {
        "name": "request_upload",
        "description": "Draw the person's eye to an upload box. They can upload now or later.",
        "input_schema": {
            "type": "object",
            # DISABLED (blood report upload): "enum": ["blood_report", "prescription"]
            "properties": {"kind": {"type": "string", "enum": ["prescription"]}},
            "required": ["kind"],
        },
    },
    {
        "name": "get_form_state",
        "description": "Read the form as it is now, including which fields are still empty "
        "and which the person typed themselves.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "finish_onboarding",
        "description": "Open the review summary. The person taps Finish themselves.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# DISABLED (blood report upload): this step was between "Medicines" and "Food" in the prompt below.
#   5. Blood report: ask if they have a recent one; call request_upload
#      "blood_report" and tell them they can drop the PDF in the box now or later.
# Put it back and renumber the steps after it when blood reports are switched on again.
SYSTEM_PROMPT = """\
You are Clinexsa, the onboarding voice. You are talking to a patient,
or a family member setting things up for them, on the Clinexsa website. A form is
on the screen next to you. Fill it using your tools as you hear answers.

How to talk:
- Everything you write is read aloud, so use plain spoken sentences with no lists,
  symbols or emoji. Tool calls are silent; only your text is spoken.
- Warm, patient, short sentences. One question at a time. Speak the language the
  person chose; switch if they switch.
- Repeat back names of medicines and numbers to confirm ("That's Metformin 500,
  and you have 30 tablets left, right?").
- If you're not sure what you heard, still fill it but with confidence below 0.7,
  and say "I've put that in, please check the spelling on the screen."
- Never give medical advice, never comment on whether a medicine or dose is right,
  never interpret report values. If asked, say: "That's a good one for your doctor."
- If the person mentions chest pain, a fall, trouble breathing, or feeling very
  unwell, stop the onboarding and say: "Please call 112 now, or ask someone near
  you to help." Don't continue until they say they're safe.

Order:
1. Call get_form_state first (it also tells you the person's chosen language). Skip anything already filled.
2. About you: name, date of birth (or just their age if they don't know the date),
   gender, height, weight. (go_to_step "about")
3. Health: conditions, whether they take medicines now, whether a doctor
   prescribed them. Offer the prescription upload. (go_to_step "health")
4. Medicines: how many, then each one's name and how many they have left.
   Call set_medication_count, then add_medication for each. (go_to_step "medicines")
5. Do not ask about food, reminders, ordering or family members. Those are set up
   afterwards on the dashboard. If the person asks, tell them so.
6. Call get_form_state again, ask about anything still empty, then
   finish_onboarding and say: "Everything's on the screen. Have a quick look,
   change anything that's wrong, and tap Finish. You'll get your Client username
   right after."

If a tool returns "field locked by user", don't try again; the person typed it
themselves.
"""

# Session facts are appended after the prompt above, so the prompt itself stays
# byte-identical between sessions.
_CONTEXT = """
About this session:
- The person chose to speak {language}.{name_line}
- What you hear is an automatic transcript. If it makes no sense, ask them to say it again.
- Lines in square brackets are notes from the page, not something the person said.
  Never read them aloud.
"""

_OPENING_NOTE = "[The session has just started. Greet the person and begin.]"
_RESUMED_NOTE = (
    "[The person has reconnected and the form already has some answers. Call "
    "get_form_state, welcome them back in one short sentence and carry on from the "
    "first thing that is still empty.]"
)


def build_system_prompt(language: str, first_name: str | None) -> str:
    name_line = f"\n- Their first name is {first_name}." if first_name else ""
    return SYSTEM_PROMPT + _CONTEXT.format(language=language, name_line=name_line)


class OnboardingReplyGenerator(ReplyGenerator):
    name = "anthropic-onboarding"

    def __init__(
        self,
        api_key: str,
        *,
        system_prompt: str,
        execute_tool: ToolExecutor,
        model: str = "claude-opus-5-5",
        effort: Literal["low", "medium", "high"] | None = "low",
        max_tokens: int = 2048,
        timeout_s: float = 20.0,
        refusal_fallback: bool = True,
        resumed: bool = False,
        max_tool_rounds: int = 8,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout_s, max_retries=1
        )
        self._system = system_prompt
        self._execute_tool = execute_tool
        self._model = model
        self._output_config: BetaOutputConfigParam | anthropic.Omit = (
            {"effort": effort} if effort else anthropic.omit
        )
        self._max_tokens = max_tokens
        self._refusal_fallback = refusal_fallback
        self._max_tool_rounds = max_tool_rounds

        # The exchange with Claude, tool calls included. Append-only: earlier turns
        # carry thinking blocks that are only valid if nothing before them changes.
        self._messages: list[BetaMessageParam] = []
        self._patient_turns_seen = 0
        self._notes: list[str] = [_RESUMED_NOTE if resumed else _OPENING_NOTE]

    @property
    def messages(self) -> list[BetaMessageParam]:
        return list(self._messages)

    def add_note(self, note: str) -> None:
        """Tell the model about something that happened on the page, on its next turn."""
        self._notes.append(f"[{note}]")

    async def stream_reply(
        self, history: Sequence[ConversationMessage], trace: ReplyTrace | None = None
    ) -> AsyncIterator[str]:
        # ``trace`` stays empty: this agent's tools fill the page's form, and look nothing up.
        self._append_user_text(self._next_user_text(history))

        for _ in range(self._max_tool_rounds + 1):
            spoken: list[str] = []
            try:
                async with self._client.beta.messages.stream(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=self._system,
                    messages=self._messages,
                    tools=TOOLS,
                    # Caches the growing conversation prefix once it is long enough.
                    cache_control={"type": "ephemeral"},
                    output_config=self._output_config,
                    betas=[_FALLBACK_BETA] if self._refusal_fallback else anthropic.omit,
                    fallbacks="default" if self._refusal_fallback else anthropic.omit,
                ) as stream:
                    async for text in stream.text_stream:
                        spoken.append(text)
                        yield text
                    final = await stream.get_final_message()
            except (asyncio.CancelledError, GeneratorExit):
                # The person talked over the reply. Keep what they heard so the
                # model knows where it was cut off.
                if heard := "".join(spoken).strip():
                    self._messages.append({"role": "assistant", "content": heard})
                raise
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

            calls = [block for block in final.content if block.type == "tool_use"]
            if final.stop_reason != "tool_use" or not calls:
                if final.stop_reason == "max_tokens" and calls:
                    # The last tool input may be cut off mid-way; never run it.
                    raise ReplyError("reply hit max_tokens during a tool call")
                self._messages.append({"role": "assistant", "content": final.content})
                return

            # Run in order: the page applies them one by one (count, then each row).
            results: list[BetaContentBlockParam] = []
            for call in calls:
                arguments = call.input if isinstance(call.input, dict) else {}
                # Tool arguments are health details: log the name only.
                log.info("onboarding.tool_call", tool=call.name)
                result: BetaToolResultBlockParam = {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": await self._execute_tool(call.id, call.name, arguments),
                }
                results.append(result)
            # Added together, so an interrupted round never leaves a tool call
            # without its result.
            self._messages.append({"role": "assistant", "content": final.content})
            self._messages.append({"role": "user", "content": results})

        raise ReplyError("too many tool rounds in one reply")

    def _next_user_text(self, history: Sequence[ConversationMessage]) -> str:
        patient_turns = [m.content for m in history if m.role == "patient" and m.content.strip()]
        new_turns = patient_turns[self._patient_turns_seen :]
        self._patient_turns_seen = len(patient_turns)
        parts = [*self._notes, *new_turns]
        self._notes = []
        if not parts:
            raise ReplyError("nothing new to reply to")
        return "\n".join(parts)

    def _append_user_text(self, text: str) -> None:
        last = self._messages[-1] if self._messages else None
        if last is None or last["role"] != "user":
            self._messages.append({"role": "user", "content": text})
            return
        # The previous reply was cut off before the model answered (or ended on
        # tool results): add to that user turn rather than start a second one.
        content = last["content"]
        blocks: list[BetaContentBlockParam] = (
            [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
        )
        blocks.append({"type": "text", "text": text})
        self._messages[-1] = {"role": "user", "content": blocks}


def build_onboarding_generator(
    settings: Settings,
    execute_tool: ToolExecutor,
    *,
    language: str,
    first_name: str | None = None,
    resumed: bool = False,
) -> OnboardingReplyGenerator | None:
    if settings.anthropic_api_key is None:
        log.warning("llm.not_configured", hint="set ANTHROPIC_API_KEY")
        return None
    return OnboardingReplyGenerator(
        settings.anthropic_api_key.get_secret_value(),
        system_prompt=build_system_prompt(language, first_name),
        execute_tool=execute_tool,
        model=settings.anthropic_model,
        effort=settings.llm_effort,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
        refusal_fallback=settings.llm_refusal_fallback,
        resumed=resumed,
    )
