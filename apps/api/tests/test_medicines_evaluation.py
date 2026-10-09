"""The medicine lookup evaluation: how its queries are made, and how an answer is scored."""

from __future__ import annotations

import random
from typing import Any

import pytest

from app.medicines.catalog import Medicine
from app.medicines.evaluation import (
    KINDS,
    Query,
    Result,
    _edits,
    by_ear,
    interval,
    make_queries,
    one_edit,
    outcome,
    render,
    run,
    said,
    split_or_joined,
    tally,
)
from app.medicines.lookup import Match


def medicine(name: str, number: int = 0) -> Medicine:
    return Medicine(id=f"az:{number or name}", name=name, source="az")


# --- making a query ----------------------------------------------------------------------


def test_what_a_person_says_is_the_leading_words_and_the_first_number() -> None:
    assert said("Glimihex M 2mg/500mg Tablet") == (["glimihex", "m"], "2")
    assert said("Crocin Advance Tablet") == (["crocin", "advance"], None)
    assert said("Paracetamol Tablets IP 500 mg") == (["paracetamol"], "500")
    # Three words at most, and nothing for a name that is only a strength or a form.
    assert said("Aquaray Plus Eye Care Drop") == (["aquaray", "plus", "eye"], None)
    assert said("500 mg Tablet") is None and said("Tablet") is None


def test_one_letter_out_is_always_exactly_one_letter() -> None:
    rng = random.Random(1)
    for _ in range(300):
        changed = one_edit("paracetamol", rng)
        assert _edits("paracetamol", changed) == 1 and changed != "paracetamol"


def test_a_name_respelt_by_ear_is_at_least_two_letters_away() -> None:
    rng = random.Random(2)
    for word in ("roximet", "crocin", "glycomet", "telmisartan"):
        for _ in range(100):
            changed = by_ear(word, rng)
            assert changed is not None and _edits(word, changed) >= 2
    # Nothing in it that a listener is known to respell.
    assert by_ear("jjjjj", rng) is None


def test_a_word_is_split_in_two_or_two_are_joined() -> None:
    rng = random.Random(3)
    seen: set[tuple[str, ...]] = set()
    for _ in range(200):
        heard = split_or_joined(["braxclav"], rng)
        assert heard is not None and len(heard) == 2 and "".join(heard) == "braxclav"
        assert min(len(part) for part in heard) >= 3
        seen.add(tuple(heard))
    assert len(seen) == 3  # brax|clav, bra|xclav, braxc|lav: no piece under three letters

    # Two short words can only be joined. A long one beside them is split half the time.
    assert split_or_joined(["pan", "d"], rng) == ["pand"]
    both = {tuple(split_or_joined(["tomrab", "l"], rng) or ()) for _ in range(60)}
    assert ("tomrabl",) in both and ("tom", "rab", "l") in both
    assert split_or_joined(["pan"], rng) is None


def test_the_same_seed_makes_the_same_queries_each_from_another_medicine() -> None:
    catalog = [medicine(f"Brand{letter}{other}ol {n} Tablet", n) for n, (letter, other) in
               enumerate(((a, b) for a in "abcdefghij" for b in "klmnopqrst"), start=1)]  # fmt: skip
    first = make_queries(catalog, 20, seed=7)
    assert first == make_queries(catalog, 20, seed=7)
    assert first != make_queries(catalog, 20, seed=8)
    assert [q.kind for q in first] == [kind for kind in KINDS for _ in range(20)]
    assert len({q.medicine_id for q in first}) == 80

    for query in first:
        if query.kind == "said":
            assert query.heard == query.meant
        else:
            assert query.heard != query.meant
        assert query.meant.split()[-1] == query.heard.split()[-1]  # the number is never changed
    with pytest.raises(ValueError, match="too small"):
        make_queries(catalog, 26, seed=7)


# --- scoring an answer -------------------------------------------------------------------

DOLO = Query(
    kind="one_edit",
    medicine_id="az:1",
    medicine="Dolo 650 Tablet",
    meant="dolo 650",
    heard="dollo 650",
)
PRODUCTS = {"Dolo 650 Tablet", "Dolo 650 SR Tablet", "Dolopar Tablet", "Dolo Drops"}


def match(status: str, name: str | None = None, *choices: str) -> Match:
    return Match(status=status, heard="dollo 650", name=name, choices=list(choices))  # type: ignore[arg-type]


def test_the_right_medicine_named_or_offered_is_resolved() -> None:
    assert outcome(DOLO, match("exact", "Dolo 650 Tablet"), PRODUCTS) == "named"
    # A misheard name gets the catalogue's spelling of what was meant, to be confirmed.
    assert outcome(DOLO, match("close", "Dolo 650"), PRODUCTS) == "named"
    assert outcome(DOLO, match("close", "Dolo 650 Tablet"), PRODUCTS) == "named"
    assert outcome(DOLO, match("several", "Dolo", "Dolo Drops", "Dolo 650 Tablet"), PRODUCTS) == (
        "offered"
    )
    # The first guess is another spelling, and the right one is offered beside it.
    assert outcome(DOLO, match("close", "Dollo 650", "Dolo 650"), PRODUCTS) == "offered"


def test_another_product_called_just_what_was_meant_is_counted_apart() -> None:
    longer = DOLO.model_copy(update={"medicine": "Dolo 650 SR Tablet"})
    # "dolo 650" is the whole name of another product. Right by the rules, not the one sampled.
    assert outcome(longer, match("exact", "Dolo 650 Tablet"), PRODUCTS) == "as_said"
    assert outcome(longer, match("close", "Dolo 650 Tablet"), PRODUCTS) == "as_said"
    # The brand was recognised, and the sampled product is past the choices listed.
    assert outcome(longer, match("several", "Dolo 650", "Dolo 650 Tablet"), PRODUCTS) == (
        "beyond_listed"
    )


def test_anything_else_is_a_miss_and_another_medicine_is_the_worst_of_them() -> None:
    assert outcome(DOLO, match("unknown"), PRODUCTS) == "not_found"
    assert outcome(DOLO, match("unknown", None, "Dolopar Tablet"), PRODUCTS) == "not_found"
    assert outcome(DOLO, match("exact", "Dolopar Tablet"), PRODUCTS) == "wrong"
    assert outcome(DOLO, match("close", "Dolopar"), PRODUCTS) == "wrong"
    assert outcome(DOLO, match("close", "Dolopar", "Dolo Drops"), PRODUCTS) == "wrong"
    assert outcome(DOLO, match("several", "Dolopar", "Dolopar Tablet"), PRODUCTS) == "wrong"


# --- a run ---------------------------------------------------------------------------------


class FakeStore:
    """Returns a fixed list of names, failing the first ``fails`` times it is asked."""

    def __init__(self, found: list[Medicine], fails: int = 0) -> None:
        self.found, self.fails = found, fails
        self.asked: list[tuple[str, dict[str, Any]]] = []

    async def search(self, heard: str, **how: Any) -> list[Medicine]:
        self.asked.append((heard, how))
        if len(self.asked) <= self.fails:
            raise ConnectionError("dropped")
        return self.found


async def test_a_run_looks_each_query_up_and_scores_the_search_and_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_wait(seconds: float) -> None:
        return None

    monkeypatch.setattr("app.medicines.evaluation.asyncio.sleep", no_wait)
    dolo = Medicine(id="az:1", name="Dolo 650 Tablet", source="az", strength="650 mg")
    exact = DOLO.model_copy(update={"kind": "said", "heard": "dolo 650"})
    # Changed so far that it reads as another product, and the search no longer returns its own.
    absent = Query(
        kind="by_ear",
        medicine_id="az:99",
        medicine="Dolopar Tablet",
        meant="dolopar",
        heard="dolo 650",
    )
    drops = Medicine(id="az:2", name="Dolo Drops", source="az")
    store = FakeStore([drops, dolo], fails=2)  # two dropped requests are tried again
    ticks: list[tuple[int, int]] = []

    results = await run(
        store,  # type: ignore[arg-type]
        [exact, DOLO, absent],
        concurrency=1,
        progress=lambda done, total: ticks.append((done, total)),
    )

    assert [(r.rank, r.status, r.outcome) for r in results] == [
        (2, "exact", "named"),  # second in the search's list, and still the one named
        (2, "close", "named"),  # "dollo 650": the catalogue's spelling, to be confirmed
        (None, "exact", "wrong"),  # another medicine, stated as fact
    ]
    assert results[0].answer == "Dolo 650 Tablet"
    assert ticks == [(1, 3), (2, 3), (3, 3)]
    assert all(how == {"limit": 64} for _, how in store.asked) and len(store.asked) == 5

    # A store that stays down ends the run: a report with holes in it would mislead.
    with pytest.raises(ConnectionError):
        await run(FakeStore([dolo], fails=99), [DOLO])  # type: ignore[arg-type]


def result(kind: str, outcome_: str, *, rank: int | None = 1, heard: str = "dollo 650") -> Result:
    query = DOLO.model_copy(update={"kind": kind, "heard": heard})
    status = {"named": "close", "offered": "several", "not_found": "unknown", "wrong": "exact"}
    return Result(
        query=query,
        rank=rank,
        status=status[outcome_],
        outcome=outcome_,  # type: ignore[arg-type]
        answer="Dolopar Tablet" if outcome_ == "wrong" else None,
    )


def test_the_report_counts_what_was_resolved_and_says_how_wrong_answers_arrive() -> None:
    results = [
        *(result("said", "named") for _ in range(3)),
        result("said", "offered"),
        *(result("one_edit", "named") for _ in range(2)),
        result("one_edit", "wrong", rank=7, heard="rolo 650"),  # the first letter was changed
        result("one_edit", "not_found", rank=None, heard="polo 650"),
        result("by_ear", "named", rank=40),
        result("split", "named"),
    ]
    counted = tally(results)
    assert (counted.queries, counted.resolved) == (10, 8)
    # Found at ranks 1 (seven times), 7, 40, and once not at all.
    assert counted.within == {5: 7, 10: 8, 64: 9}
    assert counted.outcomes["wrong"] == 1 and counted.outcomes["as_said"] == 0

    report = render(results, seed=7, collection="clinexa_medicines", names=252553, day="2026-10-09")
    assert "| Kind | Queries | Recall@5 | Recall@10 | Recall@64 | Resolved |" in report
    assert "| Said correctly | 4 | 100.0% | 100.0% | 100.0% | 100.0% |" in report
    assert "| One letter out | 4 | 50.0% | 75.0% | 75.0% | 50.0% |" in report
    assert "| **All** | 10 | 70.0% | 80.0% | 90.0% | 80.0% |" in report
    assert (
        "| One letter out, first letter changed | 2 | 0.0% | 50.0% | 50.0% | 0.0% | 50.0% | 50.0% |"
        in report
    )
    assert "Of 1 wrong answers, 1 were stated as that medicine (10.0% of all queries)" in report
    assert "`rolo 650` (meant `dolo 650`, from Dolo 650 Tablet): gave Dolopar Tablet" in report
    assert "python -m app.medicines evaluate --per-kind 4 --seed 7" in report
    assert "These are not recordings of speech." in report


def test_an_interval_narrows_with_more_queries_and_stays_inside_zero_and_one() -> None:
    low, high = interval(960, 1000)
    assert 0.945 < low < 0.96 < high < 0.971
    wide_low, wide_high = interval(24, 25)
    assert wide_low < low and wide_high > high
    assert interval(0, 0) == (0.0, 0.0)
    assert interval(10, 10)[1] <= 1.0 and interval(0, 10)[0] >= 0.0
