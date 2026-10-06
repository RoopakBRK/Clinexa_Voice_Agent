"""Roopiee's onboarding agent: Claude tool calls -> the page -> tool results -> spoken text."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from app.agents.onboarding import (
    SYSTEM_PROMPT,
    TOOLS,
    OnboardingReplyGenerator,
    build_system_prompt,
)
from app.agents.responder import ReplyError, ReplyRefused
from app.graph.state import ConversationMessage


def tool_use(call_id: str, tool: str, /, **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=tool, input=arguments)


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


class FakeToolStream:
    def __init__(self, content: list[SimpleNamespace], stop_reason: str) -> None:
        self._content = content
        self._stop_reason = stop_reason

    async def __aenter__(self) -> FakeToolStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    async def text_stream(self) -> AsyncIterator[str]:
        for block in self._content:
            if block.type == "text":
                # Two deltas per text block, as a real stream would split it.
                middle = len(block.text) // 2
                yield block.text[:middle]
                yield block.text[middle:]

    async def get_final_message(self) -> Any:
        return SimpleNamespace(
            content=self._content,
            stop_reason=self._stop_reason,
            stop_details=SimpleNamespace(category="bio")
            if self._stop_reason == "refusal"
            else None,
        )


class FakeClaude:
    """Returns one scripted response per request; the last one repeats."""

    def __init__(self, *responses: tuple[list[SimpleNamespace], str]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._open))

    def _open(self, **kwargs: Any) -> FakeToolStream:
        # The generator keeps appending to its list, so snapshot it per request.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        content, stop_reason = self._responses[min(len(self.requests), len(self._responses)) - 1]
        return FakeToolStream(content, stop_reason)


class RecordingPage:
    """Stands in for the browser: answers every tool call with a canned result."""

    def __init__(self, result: str = "ok") -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, call_id: str, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return self.result


def patient(text: str) -> ConversationMessage:
    return ConversationMessage(role="patient", content=text)


def agent(client: FakeClaude, page: RecordingPage, **kwargs: Any) -> OnboardingReplyGenerator:
    return OnboardingReplyGenerator(
        "sk-test",
        system_prompt="system",
        execute_tool=page,
        client=client,  # type: ignore[arg-type]
        **kwargs,
    )


async def collect(stream: AsyncIterator[str]) -> str:
    return "".join([delta async for delta in stream])


def test_prompt_and_tools_cover_the_form() -> None:
    names = {tool["name"] for tool in TOOLS}
    assert names == {
        "set_field",
        "set_medication_count",
        "add_medication",
        "update_medication",
        "go_to_step",
        "request_upload",
        "get_form_state",
        "finish_onboarding",
    }
    assert "Never give medical advice" in SYSTEM_PROMPT
    assert "Please call 112 now" in SYSTEM_PROMPT
    assert "Client username" in SYSTEM_PROMPT
    # Reminders and family moved to the dashboard, and the food step is switched off,
    # so the agent must not ask.
    assert "Do not ask about food, reminders, ordering or family members" in SYSTEM_PROMPT
    prompt = build_system_prompt("Hindi", "Ramesh")
    assert prompt.startswith(SYSTEM_PROMPT)  # the fixed part stays first, so it caches
    assert "Hindi" in prompt and "Ramesh" in prompt


async def test_tool_call_goes_to_the_page_and_its_result_back_to_claude() -> None:
    client = FakeClaude(
        ([tool_use("tu_1", "set_field", field="age", value=62, confidence=0.9)], "tool_use"),
        ([text_block("Sixty two. And your height?")], "end_turn"),
    )
    page = RecordingPage()
    gen = agent(client, page)

    spoken = await collect(gen.stream_reply([patient("I am sixty two.")]))

    assert spoken == "Sixty two. And your height?"
    assert page.calls == [("set_field", {"field": "age", "value": 62, "confidence": 0.9})]
    assert client.requests[0]["tools"] == TOOLS
    # The second request carries the tool call and the page's answer.
    second = client.requests[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[2]["content"] == [{"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}]
    # Tool calls stay in the session's history, followed by what was said.
    assert [m["role"] for m in gen.messages] == ["user", "assistant", "user", "assistant"]


async def test_tools_in_one_turn_run_in_order_and_return_in_one_message() -> None:
    client = FakeClaude(
        (
            [
                tool_use("tu_1", "set_medication_count", count=1),
                tool_use("tu_2", "add_medication", name="Metformin", quantity=30, unit="tablets"),
            ],
            "tool_use",
        ),
        ([text_block("Metformin, thirty tablets. Right?")], "end_turn"),
    )
    page = RecordingPage()
    await collect(agent(client, page).stream_reply([patient("Just metformin, thirty left.")]))

    assert [name for name, _ in page.calls] == ["set_medication_count", "add_medication"]
    results = client.requests[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_1", "tu_2"]


async def test_locked_field_result_reaches_claude_verbatim() -> None:
    client = FakeClaude(
        (
            [tool_use("tu_1", "set_field", field="full_name", value="Ramesh", confidence=0.9)],
            "tool_use",
        ),
        ([text_block("I see you typed your name.")], "end_turn"),
    )
    await collect(
        agent(client, RecordingPage("field locked by user")).stream_reply([patient("Ramesh")])
    )
    assert client.requests[1]["messages"][2]["content"][0]["content"] == "field locked by user"


async def test_opening_turn_and_page_notes_are_passed_as_bracketed_notes() -> None:
    client = FakeClaude(([text_block("Hello, I'm Roopiee.")], "end_turn"))
    gen = agent(client, RecordingPage())

    await collect(gen.stream_reply([]))  # nobody has spoken yet
    opening = client.requests[0]["messages"][0]["content"]
    assert opening.startswith("[") and "just started" in opening

    gen.add_note("The person typed age on the form themselves: 62.")
    await collect(gen.stream_reply([patient("My name is Ramesh.")]))
    latest = client.requests[1]["messages"][-1]["content"]
    assert latest == "[The person typed age on the form themselves: 62.]\nMy name is Ramesh."


async def test_only_new_patient_turns_are_sent_each_time() -> None:
    client = FakeClaude(([text_block("Okay.")], "end_turn"))
    gen = agent(client, RecordingPage(), resumed=True)
    history = [patient("One.")]
    await collect(gen.stream_reply(history))
    history += [ConversationMessage(role="assistant", content="Okay."), patient("Two.")]
    await collect(gen.stream_reply(history))

    assert "reconnected" in client.requests[0]["messages"][0]["content"]
    assert client.requests[1]["messages"][-1] == {"role": "user", "content": "Two."}


async def test_interrupted_reply_keeps_what_was_heard() -> None:
    client = FakeClaude(([text_block("What is your height in centimetres?")], "end_turn"))
    gen = agent(client, RecordingPage())

    stream = gen.stream_reply([patient("I am sixty two.")])
    heard = await anext(stream)
    await stream.aclose()  # the person talked over the reply

    assert gen.messages[-1] == {"role": "assistant", "content": heard.strip()}
    # The next turn continues after it rather than rewriting earlier messages.
    await collect(gen.stream_reply([patient("I am sixty two."), patient("Sorry, go on.")]))
    assert client.requests[1]["messages"][-1] == {"role": "user", "content": "Sorry, go on."}


async def test_interrupted_before_any_text_merges_into_one_user_turn() -> None:
    client = FakeClaude(([text_block("Okay.")], "end_turn"))
    gen = agent(client, RecordingPage())

    stream = gen.stream_reply([patient("First.")])
    await stream.aclose()  # cancelled before the model produced anything
    # Nothing was requested, so the user turn was never appended either.
    await collect(gen.stream_reply([patient("First."), patient("Second.")]))
    assert [m["role"] for m in client.requests[0]["messages"]] == ["user"]


async def test_refusal_and_runaway_tool_loops_raise_reply_errors() -> None:
    refusing = FakeClaude(([], "refusal"))
    with pytest.raises(ReplyRefused):
        await collect(agent(refusing, RecordingPage()).stream_reply([patient("Hi")]))

    looping = FakeClaude(([tool_use("tu_1", "get_form_state")], "tool_use"))
    page = RecordingPage()
    with pytest.raises(ReplyError, match="too many tool rounds"):
        await collect(agent(looping, page, max_tool_rounds=2).stream_reply([patient("Hi")]))
    assert len(page.calls) == 3
