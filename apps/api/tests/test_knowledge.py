"""Lookups during a call: the two tools, the guidance search, and Claude's tool loop."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest
from anthropic.types.beta import BetaTextBlock, BetaThinkingBlock, BetaToolUseBlock
from fastapi.testclient import TestClient
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient

from app.agents.responder import (
    ClaudeReplyGenerator,
    ReplyError,
    ReplyGenerator,
    ReplyTrace,
    build_reply_generator,
    build_system_prompt,
)
from app.core.config import Settings
from app.graph.state import ConversationMessage, Lookup
from app.main import create_app
from app.medicines.lookup import Match
from app.rag.retrieval import service as service_module
from app.rag.retrieval.dense import DenseRetriever
from app.rag.retrieval.filters import RetrievalFilters
from app.rag.retrieval.hybrid import HybridResult, HybridRetriever
from app.rag.retrieval.qdrant_store import QdrantChunkStore
from app.rag.retrieval.service import (
    GuidelineResult,
    GuidelineSearch,
    GuidelinesNotIndexedError,
    Pipeline,
    build_guideline_search,
    load_pipeline,
)
from app.rag.retrieval.sparse import SparseRetriever
from app.schemas.clinical import DocumentMetadata, Evidence, RetrievalScores, RetrievedChunk
from app.tools.knowledge import (
    LOOKUP_MEDICINE,
    SEARCH_GUIDELINES,
    KnowledgeTools,
    describe_match,
    describe_passages,
)
from app.voice.session import CallSession
from tests.conftest import (
    FakeTTSProvider,
    LiveFakeSTTProvider,
    final,
    stream_token,
    twilio_media,
    twilio_start,
    twilio_stop,
)
from tests.test_reranking import OverlapReranker
from tests.test_retrieval_hybrid import CORPUS
from tests.test_retrieval_store import FakeEmbedder

FRAME = b"\xff" * 160

DOLO = Match(
    status="exact",
    heard="dolo 650",
    name="Dolo 650 Tablet",
    strength="650 mg",
    unit="tablets",
    composition="Paracetamol 650 mg",
    source="A to Z medicines dataset of India",
)


def passage(cid: str, text: str, *, population: str = "child", page: int = 7) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        text=text,
        metadata=DocumentMetadata(
            source="book.pdf",
            document_id="who-pocket-book",
            document_title="Pocket book of hospital care for children",
            document_type="pocket_book",
            publisher="World Health Organization",
            publication_date="2013",
            heading_path=["Fever", "Treatment"],
            page_number=page,
            population=population,
        ),
        scores=RetrievalScores(rrf_score=0.03, reranker_score=4.2),
    )


FEVER = GuidelineResult(
    passages=[
        passage("p1", "Give paracetamol if the child is distressed by the fever."),
        passage("p2", "Danger signs: unable to drink, convulsions, lethargy.", page=9),
    ],
    population=["child", "all"],
)


class FakeMedicines:
    def __init__(self, *answers: Match | None) -> None:
        self.answers = list(answers) or [DOLO]
        self.asked: list[str] = []

    async def find(self, heard: str) -> Match | None:
        self.asked.append(heard)
        return self.answers[min(len(self.asked), len(self.answers)) - 1]


class FakeGuidelines:
    def __init__(self, result: GuidelineResult | None = FEVER) -> None:
        self.result = result
        self.asked: list[tuple[str, int | None, bool | None]] = []

    async def search(
        self, query: str, *, age: int | None = None, pregnant: bool | None = None
    ) -> GuidelineResult | None:
        self.asked.append((query, age, pregnant))
        return self.result


# --- the guidance search --------------------------------------------------------------


class Index:
    """The retrieval test corpus, indexed in memory, behind a loader the search can call."""

    def __init__(self) -> None:
        self.client = AsyncQdrantClient(":memory:")
        self.reranker = OverlapReranker()
        self.loaded = 0

    async def load(self) -> Pipeline:
        self.loaded += 1
        embedder = FakeEmbedder()
        store = QdrantChunkStore(self.client, "guidelines_test", embedder.model_name)
        await store.ensure_collection(embedder.dimension)
        indexed = [chunk for chunk in CORPUS if chunk.metadata.retrievable]
        await store.upsert(indexed, embedder.embed_documents([c.embedding_text() for c in indexed]))
        hybrid = HybridRetriever(DenseRetriever(store, embedder), SparseRetriever(CORPUS))
        return Pipeline(hybrid, self.reranker, self.client)


async def ready(search: GuidelineSearch) -> GuidelineSearch:
    search.start()
    for _ in range(400):
        if search.status != "loading":
            break
        await asyncio.sleep(0.005)
    return search


@pytest.fixture
async def guidance() -> AsyncIterator[GuidelineSearch]:
    index = Index()
    search = GuidelineSearch(index.load, top_k=2)
    yield search
    await search.aclose()


async def test_guidance_cannot_be_searched_until_it_has_loaded(guidance: GuidelineSearch) -> None:
    assert guidance.status == "not_started"
    assert await guidance.search("cough") is None  # nobody waits for the models

    await ready(guidance)
    assert guidance.status == "ready"
    found = await guidance.search("child with cough and fast breathing")
    assert found is not None and found.passages[0].chunk_id == "c1"


async def test_a_search_keeps_the_best_few_passages_and_says_how_long_it_took(
    guidance: GuidelineSearch,
) -> None:
    found = await (await ready(guidance)).search("cough")
    assert found is not None
    assert len(found.passages) == 2  # top_k, of the three that mention a cough
    assert [p.scores.reranker_rank for p in found.passages] == [1, 2]
    assert found.population is None and not found.filter_relaxed
    assert {"dense_ms", "bm25_ms", "rrf_ms", "rerank_ms", "total_ms"} <= found.timings_ms.keys()


async def test_a_search_is_held_to_guidance_written_for_that_person(
    guidance: GuidelineSearch,
) -> None:
    await ready(guidance)
    child = await guidance.search("cough", age=3)
    assert child is not None and child.population == ["child", "all"]
    assert {p.metadata.population for p in child.passages} <= {"child", "all"}
    assert "c4" not in [p.chunk_id for p in child.passages]  # the adult tuberculosis advice

    adult = await guidance.search("cough", age=40)
    assert adult is not None and adult.population == ["adult", "all"]
    assert adult.passages[0].chunk_id == "c4"
    assert {p.metadata.population for p in adult.passages} <= {"adult", "all"}

    expecting = await guidance.search("cough", age=30, pregnant=True)
    assert expecting is not None and expecting.population == ["adult", "all", "pregnancy"]


async def test_the_index_is_loaded_once_and_closed_with_the_search() -> None:
    index = Index()
    search = GuidelineSearch(index.load)
    search.start()
    search.start()
    await ready(search)
    assert index.loaded == 1

    closed: list[bool] = []
    close = index.client.close

    async def closing() -> None:
        closed.append(True)
        await close()

    index.client.close = closing  # type: ignore[method-assign]
    await search.aclose()
    assert closed == [True]


async def test_guidance_that_will_not_load_is_unavailable_and_says_nothing() -> None:
    async def broken() -> Pipeline:
        raise GuidelinesNotIndexedError("There is no collection. Run: make index")

    search = await ready(GuidelineSearch(broken))
    assert search.status == "unavailable"
    assert await search.search("cough") is None
    await search.aclose()


class Stalling:
    """A retriever that fails, or takes longer than a caller will wait."""

    def __init__(self, *, takes_s: float = 0.0, fails: bool = False) -> None:
        self.takes_s, self.fails = takes_s, fails

    async def search(self, query: str, *, filters: RetrievalFilters | None = None) -> HybridResult:
        await asyncio.sleep(self.takes_s)
        if self.fails:
            raise ConnectionError("qdrant is down")
        return HybridResult(
            candidates=[passage("p1", "text")], filters=filters or RetrievalFilters()
        )


async def test_a_search_that_fails_or_runs_late_costs_the_caller_nothing() -> None:
    for retriever in (Stalling(fails=True), Stalling(takes_s=2.0)):

        async def loader(retriever: Stalling = retriever) -> Pipeline:
            return Pipeline(retriever, None)

        search = await ready(GuidelineSearch(loader, timeout_s=0.05))
        started = time.perf_counter()
        assert await search.search("cough") is None
        assert time.perf_counter() - started < 0.5

    # Without a cross-encoder the fused order stands.
    async def plain() -> Pipeline:
        return Pipeline(Stalling(), None)

    found = await (await ready(GuidelineSearch(plain))).search("cough")
    assert found is not None and [p.chunk_id for p in found.passages] == ["p1"]


def test_guidance_is_only_searched_where_there_is_something_to_search(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert build_guideline_search(settings.model_copy(update={"data_dir": tmp_path})) is None

    chunks = tmp_path / "processed" / "chunks.jsonl"
    chunks.parent.mkdir()
    chunks.write_text("")
    here = settings.model_copy(update={"data_dir": tmp_path})
    built = build_guideline_search(here)
    # Built, and nothing opened or loaded until the server starts it.
    assert isinstance(built, GuidelineSearch) and built.status == "not_started"

    assert build_guideline_search(here.model_copy(update={"guidelines_retrieval": False})) is None
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *more: None if name == "sentence_transformers" else real(name, *more),
    )
    assert build_guideline_search(here) is None


async def test_an_index_that_was_never_built_is_named_before_any_model_is_loaded(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []

    def in_memory(settings: Settings) -> AsyncQdrantClient:
        client = AsyncQdrantClient(":memory:")
        close = client.close

        async def closing() -> None:
            closed.append(True)
            await close()

        client.close = closing  # type: ignore[method-assign]
        return client

    monkeypatch.setattr(service_module, "build_client", in_memory)
    with pytest.raises(GuidelinesNotIndexedError, match="make index"):
        await load_pipeline(settings.model_copy(update={"data_dir": tmp_path}))
    assert closed == [True]


# --- what Claude is told ----------------------------------------------------------------


def test_only_the_lookups_this_server_can_make_are_offered() -> None:
    both = KnowledgeTools(medicines=FakeMedicines(), guidelines=FakeGuidelines())
    assert both.names == [LOOKUP_MEDICINE, SEARCH_GUIDELINES]
    for tool in both.definitions:
        assert tool["eager_input_streaming"] is True
        assert tool["input_schema"]["type"] == "object" and tool["input_schema"]["required"]  # type: ignore[index]
    assert KnowledgeTools(guidelines=FakeGuidelines()).names == [SEARCH_GUIDELINES]
    assert KnowledgeTools().definitions == []

    assert both.filler([LOOKUP_MEDICINE]) == "Let me check that medicine name."
    assert both.filler([SEARCH_GUIDELINES, SEARCH_GUIDELINES]) == (
        "Let me check the guidance on that."
    )
    assert both.filler([LOOKUP_MEDICINE, SEARCH_GUIDELINES]) == "Let me check that for you."


async def test_a_medicine_lookup_says_what_the_catalogue_settles_and_no_more() -> None:
    medicines = FakeMedicines()
    outcome = await KnowledgeTools(medicines=medicines).run(LOOKUP_MEDICINE, {"name": "dolo 650"})
    assert medicines.asked == ["dolo 650"]
    assert (outcome.is_error, outcome.detail, outcome.evidence) == (False, "exact", [])
    assert outcome.duration_ms >= 0
    assert outcome.content == (
        'The catalogue has this medicine as "Dolo 650 Tablet". It contains: Paracetamol 650 mg. '
        "Strength: 650 mg. It comes as: tablets. Listed in: A to Z medicines dataset of India. "
        "Say the name back so the caller can confirm it is the one they mean."
    )
    # A generic is its own composition: it is not said twice.
    generic = Match(status="exact", heard="metformin", name="Metformin", composition="Metformin")
    assert "contains" not in describe_match(generic)
    # A guess carries nothing but the name.
    guess = Match(status="close", heard="dollo 650", name="Dolo 650 Tablet")
    assert "Strength" not in describe_match(guess) and "This is a guess." in describe_match(guess)


async def test_a_lookup_that_cannot_be_made_is_an_answer_that_says_so() -> None:
    tools = KnowledgeTools(medicines=FakeMedicines(None), guidelines=FakeGuidelines(None))
    down = await tools.run(LOOKUP_MEDICINE, {"name": "dolo 650"})
    assert (down.is_error, down.detail) == (True, "unavailable")
    assert "do not guess what the medicine is" in down.content
    silent = await tools.run(SEARCH_GUIDELINES, {"query": "fever in a child"})
    assert (silent.is_error, silent.detail) == (True, "unavailable")
    assert "Do not answer from memory." in silent.content

    # A tool this server does not have, even if the model asks for it.
    only_medicines = KnowledgeTools(medicines=FakeMedicines())
    for name in (SEARCH_GUIDELINES, "prescribe"):
        missing = await only_medicines.run(name, {"query": "fever"})
        assert (missing.is_error, missing.detail) == (True, "unknown_tool")


async def test_what_claude_asks_for_is_checked_before_anything_is_looked_up() -> None:
    medicines, guidelines = FakeMedicines(), FakeGuidelines()
    tools = KnowledgeTools(medicines=medicines, guidelines=guidelines)
    for name, arguments in (
        (LOOKUP_MEDICINE, {}),
        (LOOKUP_MEDICINE, {"name": ""}),
        (LOOKUP_MEDICINE, {"name": ["dolo"]}),
        (SEARCH_GUIDELINES, {"query": "fever", "age_years": 400}),
        (SEARCH_GUIDELINES, {"query": "fever", "age_years": "three"}),
        (SEARCH_GUIDELINES, {"age_years": 3}),
    ):
        outcome = await tools.run(name, arguments)
        assert (outcome.is_error, outcome.detail) == (True, "invalid_input"), arguments
        assert outcome.content.startswith("Invalid input. ")
    assert medicines.asked == [] and guidelines.asked == []


async def test_a_guidance_search_returns_each_passage_under_its_source() -> None:
    guidelines = FakeGuidelines()
    outcome = await KnowledgeTools(guidelines=guidelines).run(
        SEARCH_GUIDELINES, {"query": "fever in a child", "age_years": 3}
    )
    assert guidelines.asked == [("fever in a child", 3, None)]
    assert (outcome.is_error, outcome.detail) == (False, "2 passages")
    assert "Limited to guidance written for: child, all." in outcome.content
    assert (
        "[1] Pocket book of hospital care for children (World Health Organization, 2013) | "
        "Fever > Treatment | page 7 | written for: child\n"
        "Give paracetamol if the child is distressed by the fever."
    ) in outcome.content
    assert "[2] " in outcome.content and "page 9" in outcome.content
    assert outcome.evidence == [
        Evidence(
            document_id="who-pocket-book",
            document_title="Pocket book of hospital care for children",
            section="Fever > Treatment",
            page_number=page,
            excerpt=text,
            relevance_score=4.2,
        )
        for page, text in (
            (7, "Give paracetamol if the child is distressed by the fever."),
            (9, "Danger signs: unable to drink, convulsions, lethargy."),
        )
    ]


def test_passages_say_who_they_were_written_for_when_the_age_is_not_known() -> None:
    unknown = describe_passages(FEVER.model_copy(update={"population": None}))
    assert "The person's age was not given" in unknown and "written for: child" in unknown
    nothing = describe_passages(GuidelineResult(passages=[]))
    assert nothing.startswith("Nothing in the guidance matched.")
    # Guidance for everyone is not labelled, and a source may give no publisher or page.
    bare = passage("p3", "Wash hands.", population="all")
    bare.metadata.publisher = bare.metadata.publication_date = bare.metadata.page_number = None
    assert "[1] Pocket book of hospital care for children | Fever > Treatment\nWash hands." in (
        describe_passages(GuidelineResult(passages=[bare], population=["adult", "all"]))
    )


# --- the system prompt ------------------------------------------------------------------


def test_the_prompt_names_only_the_lookups_there_are() -> None:
    plain = build_system_prompt(emergency="911", greeting="Hello.")
    assert "lookup_medicine" not in plain and "search_guidelines" not in plain
    assert "Say this before you ask anything else.\n" in plain
    assert '"Hello."' in plain and "{" not in plain

    both = build_system_prompt(
        emergency="911", greeting="Hello.", tools=[LOOKUP_MEDICINE, SEARCH_GUIDELINES]
    )
    for said in (
        "not a doctor",
        "do not diagnose",
        "call 911",
        "look it up with lookup_medicine before you say anything",
        "search with search_guidelines",
        "ask before you search",
        "do not read out or work out a dose",
        "how it is listed in India",
        "before you ask anything else and before you look anything up",
        "do not announce a lookup yourself",
    ):
        assert said in both, said
    assert "{" not in both

    medicines = build_system_prompt(emergency="911", greeting="Hello.", tools=[LOOKUP_MEDICINE])
    assert "lookup_medicine" in medicines and "search_guidelines" not in medicines
    assert "World Health Organization" not in medicines


def test_the_reply_generator_is_built_with_the_tools_it_was_given(settings: Settings) -> None:
    keyed = settings.model_copy(
        update={"anthropic_api_key": SecretStr("sk-test"), "llm_max_tool_rounds": 2}
    )
    tools = KnowledgeTools(medicines=FakeMedicines())
    built = build_reply_generator(keyed, tools)
    assert isinstance(built, ClaudeReplyGenerator)
    assert built._tools is tools and built._max_tool_rounds == 2
    assert "lookup_medicine" in built._system and "search_guidelines" not in built._system

    # Nothing to look things up in: no tools are sent, and the prompt mentions none.
    bare = build_reply_generator(keyed, KnowledgeTools())
    assert isinstance(bare, ClaudeReplyGenerator) and bare._tools is None
    assert "lookup_medicine" not in bare._system
    assert build_reply_generator(settings, tools) is None  # no key


# --- Claude's tool loop (SDK surface faked) ---------------------------------------------


def text(value: str) -> BetaTextBlock:
    return BetaTextBlock(type="text", text=value)


def thinking(signature: str = "sig") -> BetaThinkingBlock:
    return BetaThinkingBlock(type="thinking", thinking="", signature=signature)


def tool_use(call_id: str, tool: str, /, **arguments: Any) -> BetaToolUseBlock:
    return BetaToolUseBlock(type="tool_use", id=call_id, name=tool, input=arguments)


class Round:
    """One response from Claude: the text it streams, then the finished message."""

    def __init__(
        self,
        *content: Any,
        stop_reason: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self.content = list(content)
        calls = any(getattr(block, "type", None) == "tool_use" for block in content)
        self.stop_reason = stop_reason or ("tool_use" if calls else "end_turn")
        self.error = error

    async def __aenter__(self) -> Round:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    async def text_stream(self) -> AsyncIterator[str]:
        for block in self.content:
            if getattr(block, "type", None) == "text":
                yield block.text
        if self.error is not None:
            raise self.error

    async def get_final_message(self) -> Any:
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=None,
            content=self.content,
            model="claude-opus-5-5",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0),
        )


class ScriptedClaude:
    """Answers each request with the next scripted round (the last one repeating)."""

    def __init__(self, *rounds: Round) -> None:
        self.rounds = list(rounds)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._open))

    def _open(self, **kwargs: Any) -> Round:
        # The generator adds to its messages after each round: keep them as they were sent.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.rounds[min(len(self.requests), len(self.rounds)) - 1]


def claude(
    client: ScriptedClaude, tools: KnowledgeTools | None, **more: Any
) -> ClaudeReplyGenerator:
    return ClaudeReplyGenerator(
        "unused",
        system_prompt="SYSTEM",
        client=client,
        tools=tools,
        **more,  # type: ignore[arg-type]
    )


def caller(said: str) -> list[ConversationMessage]:
    return [ConversationMessage(role="patient", content=said)]


async def spoken(reply: AsyncIterator[str]) -> list[str]:
    return [delta async for delta in reply]


def results(request: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool results a request carried back to Claude."""
    return list(request["messages"][-1]["content"])


async def test_a_medicine_is_looked_up_before_anything_is_said_about_it() -> None:
    call = tool_use("tu_1", LOOKUP_MEDICINE, name="dolo 650")
    client = ScriptedClaude(
        Round(thinking(), call), Round(text("That is Dolo 650, which contains paracetamol."))
    )
    medicines = FakeMedicines()
    tools = KnowledgeTools(medicines=medicines, guidelines=FakeGuidelines())
    trace = ReplyTrace()

    said = await spoken(claude(client, tools).stream_reply(caller("I take dolo 650"), trace))

    # The caller hears a line while the catalogue is asked, then the answer.
    assert said == [
        "Let me check that medicine name. ",
        "That is Dolo 650, which contains paracetamol.",
    ]
    assert medicines.asked == ["dolo 650"]
    first, second = client.requests
    assert [tool["name"] for tool in first["tools"]] == [LOOKUP_MEDICINE, SEARCH_GUIDELINES]
    assert first["tool_choice"] is anthropic.omit and second["tool_choice"] is anthropic.omit
    assert first["messages"] == [{"role": "user", "content": "I take dolo 650"}]
    # Claude's own turn goes back as it came, thinking included, with the answer after it.
    assert second["messages"][1] == {"role": "assistant", "content": [thinking(), call]}
    (result,) = results(second)
    assert result["type"] == "tool_result" and result["tool_use_id"] == "tu_1"
    assert result["content"].startswith('The catalogue has this medicine as "Dolo 650 Tablet".')
    assert "is_error" not in result
    assert second["system"] == first["system"] and second["tools"] == first["tools"]

    (lookup,) = trace.lookups
    assert (lookup.tool, lookup.detail, lookup.ok) == (LOOKUP_MEDICINE, "exact", True)
    assert trace.evidence == []


async def test_guidance_is_searched_for_the_person_asked_about_and_its_sources_kept() -> None:
    client = ScriptedClaude(
        Round(tool_use("tu_1", SEARCH_GUIDELINES, query="fever in a child", age_years=3)),
        Round(text("World Health Organization guidance says paracetamol can help.")),
    )
    guidelines = FakeGuidelines()
    trace = ReplyTrace()
    said = await spoken(
        claude(client, KnowledgeTools(guidelines=guidelines)).stream_reply(
            caller("My three year old has a fever"), trace
        )
    )

    assert said[0] == "Let me check the guidance on that. "
    assert guidelines.asked == [("fever in a child", 3, None)]
    (result,) = results(client.requests[1])
    assert "[1] Pocket book of hospital care for children" in result["content"]
    assert [(e.section, e.page_number) for e in trace.evidence] == [
        ("Fever > Treatment", 7),
        ("Fever > Treatment", 9),
    ]
    assert [(look.tool, look.detail) for look in trace.lookups] == [
        (SEARCH_GUIDELINES, "2 passages")
    ]


async def test_what_claude_said_before_a_lookup_is_spoken_at_once_with_no_line_added() -> None:
    client = ScriptedClaude(
        Round(text("One moment."), tool_use("tu_1", LOOKUP_MEDICINE, name="dolo 650")),
        Round(text("It is Dolo 650.")),
    )
    said = await spoken(
        claude(client, KnowledgeTools(medicines=FakeMedicines())).stream_reply(caller("dolo"))
    )
    # The space ends the sentence for the chunker, so it is spoken while the lookup runs.
    assert said == ["One moment.", " ", "It is Dolo 650."]


async def test_lookups_asked_for_together_are_answered_together_in_one_turn() -> None:
    client = ScriptedClaude(
        Round(
            tool_use("tu_1", SEARCH_GUIDELINES, query="fever", age_years=30, pregnant=True),
            tool_use("tu_2", LOOKUP_MEDICINE, name="paracetamol"),
        ),
        Round(text("Here is what I found.")),
    )
    medicines, guidelines = FakeMedicines(), FakeGuidelines()
    trace = ReplyTrace()
    tools = KnowledgeTools(medicines=medicines, guidelines=guidelines)
    said = await spoken(claude(client, tools).stream_reply(caller("fever"), trace))

    assert said[0] == "Let me check that for you. "
    assert guidelines.asked == [("fever", 30, True)] and medicines.asked == ["paracetamol"]
    assert len(client.requests) == 2
    # One user turn, a result for every call, in the order they were asked.
    assert [m["role"] for m in client.requests[1]["messages"]] == ["user", "assistant", "user"]
    assert [r["tool_use_id"] for r in results(client.requests[1])] == ["tu_1", "tu_2"]
    assert {look.tool for look in trace.lookups} == {SEARCH_GUIDELINES, LOOKUP_MEDICINE}


async def test_a_lookup_that_fails_goes_back_to_claude_as_an_error_not_an_exception() -> None:
    client = ScriptedClaude(
        Round(
            tool_use("tu_1", LOOKUP_MEDICINE, name="dolo 650"),
            tool_use("tu_2", LOOKUP_MEDICINE),  # no name
        ),
        Round(text("I can't check that right now.")),
    )
    medicines = FakeMedicines(None)
    trace = ReplyTrace()
    said = await spoken(
        claude(client, KnowledgeTools(medicines=medicines)).stream_reply(caller("dolo"), trace)
    )

    assert said[-1] == "I can't check that right now."
    down, invalid = results(client.requests[1])
    assert down["is_error"] is True and "could not be reached" in down["content"]
    assert invalid["is_error"] is True and invalid["content"].startswith("Invalid input. name")
    assert medicines.asked == ["dolo 650"]
    assert [(look.detail, look.ok) for look in trace.lookups] == [
        ("unavailable", False),
        ("invalid_input", False),
    ]


async def test_after_the_last_lookup_claude_has_to_answer_with_what_it_has() -> None:
    again = Round(tool_use("tu_1", LOOKUP_MEDICINE, name="dolo"))
    client = ScriptedClaude(again, again, Round(text("It may be Dolo.")))
    tools = KnowledgeTools(medicines=FakeMedicines())
    said = await spoken(claude(client, tools, max_tool_rounds=2).stream_reply(caller("dolo")))

    assert said == ["Let me check that medicine name. ", "It may be Dolo."]  # one line, not two
    assert [request["tool_choice"] for request in client.requests] == [
        anthropic.omit,
        anthropic.omit,
        {"type": "none"},
    ]
    assert client.requests[2]["tools"] == tools.definitions  # still declared: the turns use them

    # A model that calls a tool all the same is stopped, not followed.
    stuck = ScriptedClaude(again)
    with pytest.raises(ReplyError, match="kept calling tools"):
        await spoken(claude(stuck, tools, max_tool_rounds=1).stream_reply(caller("dolo")))
    assert len(stuck.requests) == 2

    # No lookups allowed at all: a single request that cannot call one.
    none = ScriptedClaude(Round(text("Hello.")))
    await spoken(claude(none, tools, max_tool_rounds=0).stream_reply(caller("hi")))
    assert [request["tool_choice"] for request in none.requests] == [{"type": "none"}]


async def test_a_tool_call_cut_off_part_way_is_never_run() -> None:
    client = ScriptedClaude(
        Round(tool_use("tu_1", LOOKUP_MEDICINE, name="do"), stop_reason="max_tokens")
    )
    medicines = FakeMedicines()
    said = await spoken(
        claude(client, KnowledgeTools(medicines=medicines)).stream_reply(caller("dolo"))
    )
    assert said == [] and medicines.asked == [] and len(client.requests) == 1

    # Nor is one that could not be read at all: the caller hears the fallback line.
    unreadable = ScriptedClaude(Round(error=ValueError("Unable to parse tool input")))
    with pytest.raises(ReplyError, match="could not be read"):
        await spoken(
            claude(unreadable, KnowledgeTools(medicines=medicines)).stream_reply(caller("dolo"))
        )


async def test_a_reply_that_fell_back_part_way_runs_only_the_answering_models_calls() -> None:
    declined = tool_use("tu_old", LOOKUP_MEDICINE, name="first attempt")
    answered = tool_use("tu_new", LOOKUP_MEDICINE, name="dolo 650")
    marker = SimpleNamespace(type="fallback")
    client = ScriptedClaude(
        Round(thinking("a"), text("Let me see."), declined, marker, thinking("b"), answered),
        Round(text("It is Dolo 650.")),
    )
    medicines = FakeMedicines()
    await spoken(claude(client, KnowledgeTools(medicines=medicines)).stream_reply(caller("dolo")))

    assert medicines.asked == ["dolo 650"]
    # Of the declined attempt only its text is echoed. The rest is the answering model's.
    assert client.requests[1]["messages"][1]["content"] == [
        text("Let me see."),
        thinking("b"),
        answered,
    ]
    assert [r["tool_use_id"] for r in results(client.requests[1])] == ["tu_new"]


async def test_with_nothing_to_look_up_a_reply_is_one_call_without_tools() -> None:
    for tools in (None, KnowledgeTools()):
        client = ScriptedClaude(Round(text("Hello there.")))
        assert await spoken(claude(client, tools).stream_reply(caller("hi"))) == ["Hello there."]
        (request,) = client.requests
        assert request["tools"] is anthropic.omit and request["tool_choice"] is anthropic.omit


# --- in a call ----------------------------------------------------------------------------


class LookingUp(ReplyGenerator):
    """Replies after one lookup, the way the Claude generator reports it."""

    name = "looking-up"

    async def stream_reply(
        self, history: Sequence[ConversationMessage], trace: ReplyTrace | None = None
    ) -> AsyncIterator[str]:
        assert trace is not None
        trace.lookups.append(Lookup(tool=SEARCH_GUIDELINES, detail="2 passages", duration_ms=42.0))
        trace.evidence.extend(
            Evidence(
                document_id="who-pocket-book",
                document_title="Pocket book",
                section="Fever",
                page_number=7,
                excerpt="Give paracetamol.",
                relevance_score=4.2,
            )
            for _ in range(2)
        )
        yield "Guidance says paracetamol can help. "


async def test_a_call_keeps_what_was_looked_up_and_how_long_it_took(settings: Settings) -> None:
    frames: list[str] = []

    async def send(raw: str) -> None:
        frames.append(raw)

    session = CallSession(
        call_sid="CA1",
        stream_sid="MZ1",
        stt=LiveFakeSTTProvider([final("My child has a fever.", speech_final=True)]),
        settings=settings,
        responder=LookingUp(),
        tts=FakeTTSProvider(),
        send=send,
    )
    session.start()
    session.feed_audio(FRAME)
    async with asyncio.timeout(2):
        while session.replies_spoken == 0:  # noqa: ASYNC110 - polling plain session state
            await asyncio.sleep(0)
    await session.close()

    snapshot = session.snapshot()
    assert [(look.tool, look.detail, look.ok) for look in snapshot.lookups] == [
        (SEARCH_GUIDELINES, "2 passages", True)
    ]
    assert [e.page_number for e in snapshot.evidence] == [7, 7]
    assert snapshot.latency["search_guidelines_ms"].max_ms == 42.0
    assert snapshot.transcript[-1].content == "Guidance says paracetamol can help."


def test_a_caller_who_names_a_medicine_hears_the_catalogues_answer(settings: Settings) -> None:
    client = ScriptedClaude(
        Round(tool_use("tu_1", LOOKUP_MEDICINE, name="dolo 650")),
        Round(text("That is Dolo 650 Tablet. It contains paracetamol.")),
    )
    medicines = FakeMedicines()
    app = create_app(
        settings,
        stt_provider=LiveFakeSTTProvider([final("I take dolo six fifty.", speech_final=True)]),
        tts_provider=FakeTTSProvider(),
        reply_generator=claude(client, KnowledgeTools(medicines=medicines)),
    )
    with TestClient(app) as http:
        with http.websocket_connect("/twilio/media-stream") as ws:
            ws.send_text(twilio_start("CA901", token=stream_token("CA901")))
            ws.send_text(twilio_media(FRAME))
            frames = [json.loads(ws.receive_text()) for _ in range(4)]
            ws.send_text(twilio_stop("CA901"))

        heard = [base64.b64decode(f["media"]["payload"]).decode() for f in frames[:3]]
        assert heard == [
            "Let me check that medicine name.",
            "That is Dolo 650 Tablet.",
            "It contains paracetamol.",
        ]
        assert frames[3]["event"] == "mark"
        assert medicines.asked == ["dolo 650"]

        call = http.get("/api/calls/CA901").json()
        assert [(look["tool"], look["detail"]) for look in call["lookups"]] == [
            (LOOKUP_MEDICINE, "exact")
        ]
        assert call["latency"]["lookup_medicine_ms"]["count"] == 1
        assert call["transcript"][-1]["content"] == " ".join(heard)


# --- the server ---------------------------------------------------------------------------


class Resource:
    """Stands in for the catalogue or the guidance search: started and closed with the app."""

    def __init__(self) -> None:
        self.started = self.closed = 0
        self.status = "ready"

    def start(self) -> None:
        self.started += 1

    async def aclose(self) -> None:
        self.closed += 1

    async def find(self, heard: str) -> Match | None:
        return DOLO

    async def search(self, query: str, **who: Any) -> GuidelineResult | None:
        return FEVER


def test_the_server_gives_claude_the_lookups_it_has_and_reports_them(settings: Settings) -> None:
    keyed = settings.model_copy(update={"anthropic_api_key": SecretStr("sk-test")})
    medicines, guidelines = Resource(), Resource()
    app = create_app(
        keyed,
        stt_provider=None,
        medicine_lookup=medicines,  # type: ignore[arg-type]
        guideline_search=guidelines,  # type: ignore[arg-type]
    )
    generator = app.state.reply_generator
    assert isinstance(generator, ClaudeReplyGenerator)
    assert generator._tools is not None
    assert generator._tools.names == [LOOKUP_MEDICINE, SEARCH_GUIDELINES]

    with TestClient(app) as http:
        assert (medicines.started, guidelines.started) == (1, 1)
        assert http.get("/health").json()["knowledge"] == {
            "medicines": {"configured": True, "collection": "clinexa_medicines", "status": "ready"},
            "guidelines": {
                "configured": True,
                "collection": "clinexa_who_primary_care",
                "status": "ready",
            },
        }
    assert (medicines.closed, guidelines.closed) == (1, 1)


def test_without_a_claude_nothing_is_opened_to_look_things_up_in(settings: Settings) -> None:
    # No key, so no replies: no Qdrant is connected to and no model is loaded.
    app = create_app(settings, stt_provider=None)
    assert app.state.medicine_lookup is None and app.state.guideline_search is None
    with TestClient(app) as http:
        knowledge = http.get("/health").json()["knowledge"]
    assert knowledge["medicines"] == {
        "configured": False,
        "collection": "clinexa_medicines",
        "status": None,
    }
    assert knowledge["guidelines"]["configured"] is False

    # A reply generator handed in brings its own tools: the server adds none.
    injected = create_app(
        settings.model_copy(update={"anthropic_api_key": SecretStr("sk-test")}),
        stt_provider=None,
        reply_generator=LookingUp(),
    )
    assert injected.state.medicine_lookup is None and injected.state.guideline_search is None
