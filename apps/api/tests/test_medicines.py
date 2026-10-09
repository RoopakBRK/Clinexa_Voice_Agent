"""The medicines catalogue: reading the sources, finding a name, and asking it on a call."""

from __future__ import annotations

import asyncio
import importlib.util
import time
import warnings
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

from app.core.config import Settings
from app.main import create_app
from app.medicines import lookup as lookup_module
from app.medicines.catalog import (
    Medicine,
    az_medicine,
    jan_aushadhi_medicine,
    merged,
    nlem_entries,
    strengths_in,
    unit_for,
)
from app.medicines.encoders import NameEncoders, build_encoders, passage
from app.medicines.lookup import Match, MedicineLookup, build_medicine_lookup, decide
from app.medicines.store import MedicineStore, NoDenseVectorsError, point_id
from app.medicines.text import alike, content_tokens, features, near, sound
from app.tools.knowledge import describe_match

# --- names as text -----------------------------------------------------------------


def test_a_name_is_its_words_without_the_form_and_the_units() -> None:
    assert content_tokens("Glycomet-GP 0.5mg Tablet PR") == ["glycomet", "gp", "0.5", "pr"]
    assert content_tokens("Paracetamol Tablets IP 650 mg") == ["paracetamol", "650"]
    assert content_tokens("Vitamin D3") == ["vitamin", "d", "3"]


def test_a_single_letter_is_part_of_the_name_unless_it_is_a_unit() -> None:
    # S-Amlong is another medicine than Amlong, and vitamin A than vitamin D.
    assert content_tokens("S-Amlong 5 Tablet") == ["s", "amlong", "5"]
    assert content_tokens("Vitamin A Capsules") == ["vitamin", "a"]
    assert content_tokens("G-Clav 625") == ["g", "clav", "625"]
    assert content_tokens("Diclofenac Gel IP 1.16%w/w 30 g") == ["diclofenac", "1.16", "30"]
    assert content_tokens("Paracetamol Tablets 10's") == ["paracetamol", "10"]
    assert features("S-Amlong 5 Tablet") != features("Amlong 5 Tablet")


def test_a_name_may_be_misheard_by_a_letter_or_by_its_vowels_and_no_further() -> None:
    assert sound("glycomate") == sound("glycomet")
    assert sound("thelma") == sound("telma")
    assert sound("atorwa") == sound("atorva")  # v and w are heard as one
    for heard, written in [
        ("dollo", "dolo"),
        ("thelma", "telma"),
        ("atorwa", "atorva"),
        ("glycomate", "glycomet"),
        ("krosin", "crocin"),
        ("parasitamol", "paracetamol"),
        ("500", "500.0"),
    ]:
        assert near(heard, written), (heard, written)
    # Different medicines, however close they look.
    for heard, written in [
        ("shelcal", "selca"),  # two letters out, and it does not sound the same
        ("atorwa", "atetor"),
        ("zzyzx", "suzox"),  # the same consonants, and hardly a letter in common
        ("telma", "telmikind"),
        ("glycomet", "glyciphage"),
        ("pan", "pen"),  # short names must be exact
        ("pandy", "andy"),  # a different first sound
        ("500", "650"),
    ]:
        assert not near(heard, written), (heard, written)
    assert alike("pan", "pen") == 0.0 and alike("pandy", "andy") == 0.0
    assert not near("zzyzxforte", "suzoxforte", by_sound=False)


def test_a_name_makes_the_same_vector_every_time() -> None:
    first, again = features("Glycomet 500 SR Tablet"), features("glycomet  500 sr tablet")
    assert first == again and len(first) > 8
    assert all(0 <= index < 2**32 for index in first)
    # What was misheard still shares most of it with the real name.
    shared = set(features("glycomate 500")) & set(first)
    assert len(shared) >= 6
    assert features("mg ml") == {}


# --- reading the sources -------------------------------------------------------------


def test_an_a_to_z_row_becomes_a_medicine_with_its_composition() -> None:
    row = {
        "id": "1",
        "name": "Augmentin 625 Duo Tablet",
        "Is_discontinued": "FALSE",
        "manufacturer_name": "Glaxo SmithKline Pharmaceuticals Ltd",
        "pack_size_label": "strip of 10 tablets",
        "short_composition1": "Amoxycillin  (500mg) ",
        "short_composition2": "  Clavulanic Acid (125mg)",
    }
    medicine = az_medicine(row)
    assert medicine is not None
    assert medicine.id == "az:1" and medicine.name == "Augmentin 625 Duo Tablet"
    assert medicine.composition == "Amoxycillin 500 mg + Clavulanic Acid 125 mg"
    assert medicine.strength == "500 mg + 125 mg" and medicine.unit == "tablets"
    assert az_medicine({"id": "2", "name": "  "}) is None

    syrup = az_medicine(
        {
            "id": "3",
            "name": "Ascoril LS Syrup",
            "Is_discontinued": "TRUE",
            "pack_size_label": "bottle of 100 ml Syrup",
            "short_composition1": "Ambroxol (30mg/5ml) ",
            "short_composition2": "",
        }
    )
    assert syrup is not None and syrup.unit == "ml" and syrup.discontinued
    assert syrup.strength == "30 mg/5 ml"


def test_a_jan_aushadhi_row_is_read_and_the_table_headings_are_not() -> None:
    medicine = jan_aushadhi_medicine(["34", "36", "Promethazine Syrup IP 5mg per 5ml", "100 ml"])
    assert medicine is not None and medicine.id == "ja:36"
    assert medicine.strength == "5 mg/5 ml" and medicine.unit == "ml" and medicine.pack == "100 ml"
    assert jan_aushadhi_medicine(["S. No.", "Drug", "Generic Name of Item", "Unit Size"]) is None
    assert jan_aushadhi_medicine(["1", "1", "Aceclofenac Tablets IP 100 mg"]) is None


def test_strengths_and_units_come_only_from_what_is_written() -> None:
    assert strengths_in("Aceclofenac 100mg and Paracetamol 325mg Tablets") == "100 mg + 325 mg"
    assert strengths_in("Diclofenac Sodium Injection IP 25mg per ml") == "25 mg/ml"
    assert strengths_in("Diclofenac Gel IP 1.16%w/w") == "1.16 %w/w"
    assert strengths_in("Liquid for inhalation") is None
    assert unit_for("strip of 10 capsule sr") == "capsules"
    assert unit_for("Oral Suspension") == "ml"  # "suspension" is not a "pen"
    assert unit_for("vial of 1 Injection") == "injections"
    assert unit_for("tube of 20 gm Cream") is None


def test_the_essential_medicines_list_is_read_line_by_line() -> None:
    page = """
        Section 2
        Analgesics
        2.3-Medicines used to treat Gout
        Medicine
        Level of
        Healthcare
        Dosage form(s) and strength(s)
        2.2.2
        Morphine*
        P,S,T
        Tablet 10 mg
        Injection 10 mg/mL
        2.2.3 Benzathine
        benzylpenicillin
        S,T
        Powder for Injection 1000 mg
        (A) + 125 mg (B)
        S,T
        Capsule 50 mg
        * Morphine formulations are also listed in Section 1.3.4
        7
        """.splitlines()
    assert list(nlem_entries(page)) == [
        ("2.2.2", "Morphine", ["Tablet 10 mg", "Injection 10 mg/mL"]),
        (
            "2.2.3",
            "Benzathine benzylpenicillin",
            ["Powder for Injection 1000 mg (A) + 125 mg (B)", "Capsule 50 mg"],
        ),
    ]


def test_one_entry_for_each_name_and_no_strength_where_two_disagree() -> None:
    def named(name: str, strength: str | None, **more: Any) -> Medicine:
        return Medicine(
            id=f"az:{name}:{strength}", name=name, source="az", strength=strength, **more
        )

    kept = merged(
        [
            named("Telma 40 Tablet", "40 mg", discontinued=True),
            named("telma 40 tablet", "40 mg"),
            named("NS Infusion", "0.9 %"),
            named("NS Infusion", "0.45 %"),
        ]
    )
    assert [(m.name, m.strength, m.discontinued) for m in kept] == [
        ("Telma 40 Tablet", "40 mg", False),
        ("NS Infusion", None, False),
    ]


# --- finding a name ------------------------------------------------------------------


def medicine(
    name: str, strength: str | None = None, unit: str | None = "tablets", **more: Any
) -> Medicine:
    return Medicine(
        id=more.pop("id", f"az:{name}"), name=name, source=more.pop("source", "az"),
        strength=strength, unit=unit, **more,
    )  # fmt: skip


CATALOG = [
    medicine("Dolo 650 Tablet", "650 mg", composition="Paracetamol 650 mg"),
    medicine("Dolo 500 Tablet", "500 mg"),
    medicine("Dolo Drops", "100 mg/ml", "ml"),
    medicine("Dolopar Tablet", "500 mg + 25 mg"),
    medicine("Glycomet Tablet", "500 mg"),
    medicine("Glycomet 250 Tablet", "250 mg"),
    medicine("Glycomet 500 SR Tablet", "500 mg"),
    medicine("Glycomet GP 1 Tablet PR", "500 mg + 1 mg"),
    medicine("Glyciphage 500mg Tablet", "500 mg"),
    medicine("Ecosprin 75 Tablet", "75 mg"),
    medicine("Ecosprin AV 75 Capsule", "75 mg + 10 mg", "capsules"),
    medicine("Pan-D Capsule PR", "30 mg + 40 mg", "capsules"),
    medicine("Pan 40 Tablet", "40 mg"),
    medicine("A Pan 40mg Tablet", "40 mg"),
    medicine("Andy 50mg Injection", "50 mg", "injections"),
    medicine("Amlong Tablet", "5 mg"),
    medicine("S-Amlong 5 Tablet", "5 mg", composition="S-Amlodipine 5 mg"),
    medicine("Selca 500 Tablet", "500 mg"),
    medicine("Suzox Forte Tablet", "100 mg + 500 mg"),
    medicine("Atorva Tablet", "10 mg"),
    medicine("Atorva 20 Tablet", "20 mg"),
    medicine("Atorvan 10mg Tablet", "10 mg"),
    medicine("Ascoril LS Syrup", "30 mg/5 ml", "ml"),
    medicine("Ascoril LS Drops", "7.5 mg/ml", "ml"),
    medicine("Ascodil LS Syrup", "30 mg/5 ml", "ml"),
    medicine("Crocin Advance Tablet", "500 mg"),
    medicine("Metformin", None, None, id="nlem:18.5.3", source="nlem"),
    medicine("Metformin Tablet 500 mg", "500 mg", id="nlem:18.5.3:1", source="nlem"),
    medicine("Metformin Tablet 1000 mg", "1000 mg", id="nlem:18.5.3:2", source="nlem"),
    medicine(
        "Metformin Hydrochloride Tablets IP 500mg", "500 mg", id="ja:412", source="jan_aushadhi"
    ),
]


@pytest.fixture
async def store() -> AsyncIterator[MedicineStore]:
    client = AsyncQdrantClient(":memory:")
    found = MedicineStore(client, "clinexa_medicines_test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the embedded index has no payload indexes
        await found.ensure_collection()
    await found.upsert(CATALOG)
    yield found
    await client.close()


async def find(store: MedicineStore, heard: str) -> Match:
    return decide(heard, await store.search(heard))


async def test_the_catalogue_is_kept_once_however_often_it_is_indexed(store: MedicineStore) -> None:
    assert await store.count() == len(CATALOG)
    await store.upsert(CATALOG)
    assert await store.count() == len(CATALOG)
    assert point_id("az:1") == point_id("az:1") != point_id("az:2")
    assert (await store.search("dolo 650"))[0].name == "Dolo 650 Tablet"
    assert await store.search("mg") == []


async def test_a_name_in_the_catalogue_just_as_said_is_written_with_its_strength(
    store: MedicineStore,
) -> None:
    match = await find(store, "Dolo 650")
    assert match.status == "exact" and match.name == "Dolo 650 Tablet"
    assert (match.strength, match.unit, match.composition) == (
        "650 mg",
        "tablets",
        "Paracetamol 650 mg",
    )
    assert match.source == "A to Z medicines dataset of India" and match.choices == []
    # No number, and nothing else in the catalogue is called this.
    crocin = await find(store, "crocin advance")
    assert (crocin.status, crocin.name, crocin.strength) == (
        "exact",
        "Crocin Advance Tablet",
        "500 mg",
    )
    # A generic name, with its strength, from the essential medicines list.
    generic = await find(store, "metformin 500")
    assert (generic.status, generic.name, generic.strength) == (
        "exact",
        "Metformin Tablet 500 mg",
        "500 mg",
    )


async def test_a_brand_with_several_products_is_named_and_none_is_picked(
    store: MedicineStore,
) -> None:
    match = await find(store, "glycomet")
    assert match.status == "several" and match.name == "Glycomet"
    assert match.strength is None  # 250 mg and 500 mg both carry the name
    assert match.unit == "tablets"  # they are all tablets
    assert set(match.choices) == {
        "Glycomet Tablet", "Glycomet 250 Tablet", "Glycomet 500 SR Tablet", "Glycomet GP 1 Tablet PR",
    }  # fmt: skip
    told = describe_match(match)
    assert "more than one product" in told and "Do not choose for them." in told
    assert "Each comes as: tablets." in told and "Strength" not in told

    dolo = await find(store, "dolo")
    assert (dolo.status, dolo.name, dolo.strength, dolo.unit) == ("several", "Dolo", None, None)
    assert "Dolopar Tablet" not in dolo.choices  # another medicine


async def test_a_word_the_person_did_not_say_is_not_added_for_them(store: MedicineStore) -> None:
    # The catalogue's only Glycomet with 500 in its name is the SR one. They did not say SR.
    match = await find(store, "glycomet 500")
    assert (match.status, match.name) == ("several", "Glycomet 500")
    assert match.strength == "500 mg" and match.choices == ["Glycomet 500 SR Tablet"]
    assert 'The closest product is "Glycomet 500 SR Tablet".' in describe_match(match)


async def test_a_number_the_catalogue_has_no_product_for_is_kept_as_said(
    store: MedicineStore,
) -> None:
    # "Atorva 10" is sold as "Atorva Tablet". It must not become Atorvan, which has a 10.
    match = await find(store, "atorva 10")
    assert (match.status, match.name) == ("several", "Atorva 10")
    assert match.strength is None
    assert set(match.choices) == {"Atorva Tablet", "Atorva 20 Tablet"}


async def test_a_letter_at_the_front_makes_it_another_medicine(store: MedicineStore) -> None:
    # The catalogue's only "Amlong 5" is S-Amlong 5, which is not the same medicine.
    match = await find(store, "amlong 5")
    assert (match.status, match.name) == ("several", "Amlong 5")
    assert match.choices == ["S-Amlong 5 Tablet"] and match.composition == "S-Amlodipine 5 mg"
    named = await find(store, "s amlong 5")
    assert (named.status, named.name) == ("exact", "S-Amlong 5 Tablet")
    # "Pan 40" is there just as said. "A Pan 40" does not make it a choice.
    pan = await find(store, "pan 40")
    assert (pan.status, pan.name, pan.strength) == ("exact", "Pan 40 Tablet", "40 mg")


async def test_a_misheard_name_gets_the_nearest_spelling_and_nothing_else_is_filled_in(
    store: MedicineStore,
) -> None:
    match = await find(store, "glycomate 500")
    assert (match.status, match.name) == ("close", "Glycomet 500")
    # The name is a guess. A strength that followed from it would be a guess too.
    assert (match.strength, match.unit, match.composition) == (None, None, None)
    told = describe_match(match)
    assert '"glycomate 500" is not in the catalogue. "Glycomet 500" is, and sounds like it.' in told
    assert "Say nothing else about it until they confirm" in told
    dolo = await find(store, "dollo 650")
    assert (dolo.status, dolo.name, dolo.strength) == ("close", "Dolo 650 Tablet", None)
    # Glyciphage is another brand, not a mishearing of Glycomet.
    assert (await find(store, "glycomate")).choices == []


def test_two_names_it_could_have_been_are_both_offered() -> None:
    match = decide(
        "glycomate 500",
        [medicine("Glycomet 500 SR Tablet", "500 mg"), medicine("Glucomate 500mg Tablet XR", "500 mg")],
    )  # fmt: skip
    assert (match.status, match.name, match.choices) == ("close", "Glucomate 500", ["Glycomet 500"])
    assert match.strength is None
    assert "It could also be: Glycomet 500." in describe_match(match)


async def test_words_that_speech_recognition_split_or_joined_still_find_the_name(
    store: MedicineStore,
) -> None:
    split = await find(store, "eco sprin 75")
    assert (split.status, split.name, split.strength) == ("exact", "Ecosprin 75 Tablet", "75 mg")
    joined = await find(store, "pandy")
    # Not "Andy": a name keeps its first sound.
    assert (joined.status, joined.name, joined.choices) == ("close", "Pan D", [])


async def test_the_form_the_person_said_narrows_it_and_a_near_brand_does_not_join_in(
    store: MedicineStore,
) -> None:
    match = await find(store, "ascoril ls syrup")
    assert (match.status, match.name, match.unit) == ("exact", "Ascoril LS Syrup", "ml")
    either = await find(store, "ascoril ls")
    assert either.status == "several" and set(either.choices) == {
        "Ascoril LS Syrup",
        "Ascoril LS Drops",
    }


async def test_a_name_that_is_not_there_is_left_as_heard(store: MedicineStore) -> None:
    match = await find(store, "limcee 500")
    assert (match.status, match.name, match.strength, match.choices) == ("unknown", None, None, [])
    told = describe_match(match)
    assert '"limcee 500" is not in the catalogue.' in told and "Names in the catalogue" not in told
    assert "Do not suggest another medicine in its place." in told
    assert (await find(store, "500 mg")).status == "unknown"  # no name at all
    # Sharing a number or a common word does not make a near miss.
    assert (await find(store, "zzyzx advance")).choices == []


async def test_a_name_that_is_only_like_another_is_not_changed_to_it(store: MedicineStore) -> None:
    # Shelcal is a real medicine the catalogue does not have. Selca is another one.
    match = await find(store, "shelcal 500")
    assert (match.status, match.name, match.strength) == ("unknown", None, None)
    assert match.choices == ["Selca 500 Tablet"]  # mentioned to Claude, never offered as theirs
    assert "Mention one only if the caller says that is their medicine." in describe_match(match)
    # The same consonants are not enough when hardly a letter agrees.
    made_up = await find(store, "zzyzx forte")
    assert (made_up.status, made_up.name, made_up.choices) == ("unknown", None, [])


# --- asking the catalogue during a conversation ----------------------------------------


class BrokenStore:
    collection = "broken"

    def __init__(self) -> None:
        self.asked = 0

    async def exists(self) -> bool:
        raise ConnectionError("the catalogue is down")

    async def search(self, heard: str, **how: Any) -> list[Medicine]:
        self.asked += 1
        raise ConnectionError("the catalogue is down")


async def test_a_catalogue_that_is_down_is_not_asked_again_for_a_while() -> None:
    broken = BrokenStore()
    lookup = MedicineLookup(broken, pause_s=60.0)  # type: ignore[arg-type]
    assert await lookup.find("Dolo 650") is None
    assert await lookup.find("Dolo 650") is None
    # Two failures in a row: the next medicines do not wait on it at all.
    assert await lookup.find("Telma 40") is None
    assert await lookup.find("Pan D") is None
    assert broken.asked == 2


async def test_a_lookup_answers_from_the_store_and_ignores_an_empty_name(
    store: MedicineStore,
) -> None:
    lookup = MedicineLookup(store)
    found = await lookup.find("  Dolo   650 ")
    assert found is not None and found.name == "Dolo 650 Tablet"
    assert await lookup.find("   ") is None


def test_the_catalogue_is_only_asked_where_there_is_a_qdrant(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No network in the tests: whichever Qdrant is asked for, this one is handed over.
    monkeypatch.setattr(
        lookup_module, "build_client", lambda settings: AsyncQdrantClient(":memory:")
    )
    assert build_medicine_lookup(settings) is None  # no QDRANT_URL in the tests
    switched_off = settings.model_copy(
        update={"qdrant_url": "https://example.invalid:6333", "medicines_lookup": False}
    )
    assert build_medicine_lookup(switched_off) is None
    configured = settings.model_copy(update={"qdrant_url": "https://example.invalid:6333"})
    assert isinstance(build_medicine_lookup(configured), MedicineLookup)


# --- the bi-encoder and the cross-encoder ------------------------------------------------


class FakeBiEncoder:
    """Three numbers for a name: how much it is about dolo, about glycomet, about anything else."""

    model_name = "fake-bi-encoder"
    dimension = 3

    def __init__(self, fails: bool = False) -> None:
        self.asked: list[str] = []
        self.fails = fails

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float("dolo" in lowered), float("glyco" in lowered), 0.1]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.asked.append(text)
        if self.fails:
            raise RuntimeError("the model would not run")
        return self._embed(text)


class FakeCrossEncoder:
    """Scores a name by a list of favourites, best first. Anything else scores nothing."""

    model_name = "fake-cross-encoder"

    def __init__(self, *favourites: str, fails: bool = False, takes_s: float = 0.0) -> None:
        self.favourites = favourites
        self.fails = fails
        self.takes_s = takes_s
        self.read: list[list[str]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.read.append(list(passages))
        time.sleep(self.takes_s)
        if self.fails:
            raise RuntimeError("the model would not run")
        return [
            next(
                (
                    float(len(self.favourites) - place)
                    for place, name in enumerate(self.favourites)
                    if text.startswith(name)
                ),
                0.0,
            )
            for text in passages
        ]


def encoders(
    bi: FakeBiEncoder | None = None, cross: FakeCrossEncoder | None = None
) -> NameEncoders:
    made = NameEncoders(bi or FakeBiEncoder(), cross)
    made.warm()
    return made


@pytest.fixture
async def dense_store() -> AsyncIterator[MedicineStore]:
    """The same catalogue, with a bi-encoder's vector beside each name."""
    client = AsyncQdrantClient(":memory:")
    found = MedicineStore(client, "clinexa_medicines_dense_test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        await found.ensure_collection(dimension=FakeBiEncoder.dimension)
    await found.upsert(CATALOG, dense=FakeBiEncoder().embed_documents([m.name for m in CATALOG]))
    yield found
    await client.close()


async def test_a_name_carries_a_dense_vector_only_if_the_catalogue_was_indexed_with_one(
    store: MedicineStore, dense_store: MedicineStore
) -> None:
    assert await dense_store.has_dense() and not await store.has_dense()
    assert await dense_store.count() == len(CATALOG)
    # Dense vectors cannot be added to a collection made without them.
    with pytest.raises(NoDenseVectorsError, match="--recreate"):
        await store.ensure_collection(dimension=3)
    with pytest.raises(ValueError, match="same length"):
        await dense_store.upsert(CATALOG, dense=[[1.0, 0.0, 0.0]])


async def test_the_dense_leg_adds_names_after_the_ones_found_by_spelling(
    dense_store: MedicineStore,
) -> None:
    # No name is spelt or sounds like "kidney". Only a bi-encoder, which here takes it to
    # mean Glycomet, finds any.
    assert await dense_store.search("kidney") == []
    by_meaning = await dense_store.search("kidney", dense=[0.0, 1.0, 0.1], dense_limit=3)
    assert len(by_meaning) == 3 and all("Glyco" in m.name for m in by_meaning)

    # Spelling finds Dolo. The dense leg may only add to that, after it, and nothing twice.
    alone = await dense_store.search("dolo 650", limit=4)
    both = await dense_store.search("dolo 650", limit=4, dense=[0.0, 1.0, 0.1], dense_limit=3)
    assert [m.name for m in both[:4]] == [m.name for m in alone]
    assert len(both) == 7 and len({m.name for m in both}) == 7
    assert all("Glyco" in m.name for m in both[4:])
    assert (
        await dense_store.search("dolo 650", limit=4, dense=[0.0, 1.0, 0.1], dense_limit=0) == alone
    )


def test_the_cross_encoder_reads_the_name_and_what_is_in_it() -> None:
    dolo, generic = CATALOG[0], next(m for m in CATALOG if m.name == "Metformin")
    assert passage(dolo) == "Dolo 650 Tablet (Paracetamol 650 mg)"
    assert passage(generic) == "Metformin"

    cross = FakeCrossEncoder("Dolo Drops", "Dolo 500")
    ordered = encoders(cross=cross).order("dolo", CATALOG[:4])
    # Its favourites first, and the rest in the order they came.
    assert [m.name for m in ordered] == ["Dolo Drops", "Dolo 500 Tablet", "Dolo 650 Tablet", "Dolopar Tablet"]  # fmt: skip
    assert cross.read[-1][0] == "Dolo 650 Tablet (Paracetamol 650 mg)"
    assert encoders().order("dolo", CATALOG[:4]) == CATALOG[:4]  # no cross-encoder, no change


async def test_the_cross_encoder_orders_equals_and_never_overrules_the_rules(
    dense_store: MedicineStore,
) -> None:
    plain = await MedicineLookup(dense_store).find("dolo")
    assert plain is not None and plain.status == "several"

    # Dolo 500 and Dolo 650 each have one word more than was said. Which is listed first
    # is the cross-encoder's to say.
    for favourite, other in (("Dolo 500", "Dolo 650"), ("Dolo 650", "Dolo 500")):
        lookup = MedicineLookup(dense_store, encoders=encoders(cross=FakeCrossEncoder(favourite)))
        match = await lookup.find("dolo")
        assert match is not None and set(match.choices) == set(plain.choices)
        assert match.choices.index(f"{favourite} Tablet") < match.choices.index(f"{other} Tablet")
        assert (match.status, match.name, match.strength) == ("several", "Dolo", None)

    # Its favourite is another medicine. What was said is still what is written.
    swayed = MedicineLookup(
        dense_store, encoders=encoders(cross=FakeCrossEncoder("Dolopar", "Glycomet"))
    )
    exact = await swayed.find("dolo 650")
    assert exact is not None
    assert (exact.status, exact.name, exact.strength) == ("exact", "Dolo 650 Tablet", "650 mg")
    # "Glycomet Tablet" has nothing more to it than was said, so it stays first whatever
    # the cross-encoder makes of the others.
    several = await MedicineLookup(
        dense_store, encoders=encoders(cross=FakeCrossEncoder("Glycomet GP 1"))
    ).find("glycomet")
    assert several is not None and several.choices[0] == "Glycomet Tablet"


class SpyStore:
    """Answers with Dolo 650 and records how it was asked."""

    collection = "spy"

    def __init__(self, dense: bool = True) -> None:
        self.dense = dense
        self.asked: list[dict[str, Any]] = []

    async def exists(self) -> bool:
        return True

    async def count(self) -> int:
        return len(CATALOG)

    async def has_dense(self) -> bool:
        return self.dense

    async def search(self, heard: str, **how: Any) -> list[Medicine]:
        self.asked.append(how)
        return CATALOG[:2]


async def test_what_was_heard_is_embedded_only_when_the_models_are_in_and_the_names_have_vectors() -> (
    None
):
    bi = FakeBiEncoder()
    spy = SpyStore()
    await MedicineLookup(spy, encoders=encoders(bi), dense_candidates=9).find("dolo 650")  # type: ignore[arg-type]
    assert bi.asked[-1] == "dolo 650"
    assert spy.asked == [{"limit": 64, "dense": [1.0, 0.0, 0.1], "dense_limit": 9}]

    # Still loading: the lookup does not wait for them.
    loading = NameEncoders(bi := FakeBiEncoder(), cross := FakeCrossEncoder("Dolo 500"))
    spy = SpyStore()
    match = await MedicineLookup(spy, encoders=loading).find("dolo 650")  # type: ignore[arg-type]
    assert match is not None and match.name == "Dolo 650 Tablet"
    assert bi.asked == [] and cross.read == [] and spy.asked[0]["dense"] is None

    # A catalogue indexed without a bi-encoder, and a lookup told to leave the dense leg out.
    for spy, how in ((SpyStore(dense=False), {}), (SpyStore(), {"dense_candidates": 0})):
        bi = FakeBiEncoder()
        await MedicineLookup(spy, encoders=encoders(bi), **how).find("dolo 650")  # type: ignore[arg-type]
        assert bi.asked == ["paracetamol"]  # the warm-up, and nothing since
        assert spy.asked[0]["dense"] is None


async def test_a_model_that_fails_or_is_slow_costs_the_lookup_nothing(
    dense_store: MedicineStore,
) -> None:
    # A cross-encoder that breaks, then one that takes longer than the person will wait.
    for fails, takes_s in ((True, 0.0), (False, 0.6)):
        cross = FakeCrossEncoder("Dolo 500")
        made = encoders(FakeBiEncoder(), cross)
        cross.fails, cross.takes_s = fails, takes_s
        started = time.perf_counter()
        match = await MedicineLookup(dense_store, encoders=made, timeout_s=0.25).find("dolo 650")
        assert match is not None and (match.status, match.name) == ("exact", "Dolo 650 Tablet")
        assert time.perf_counter() - started < 0.5

    # A bi-encoder that will not embed: the name is found by its spelling as before.
    bi = FakeBiEncoder()
    made = encoders(bi)
    bi.fails = True
    match = await MedicineLookup(dense_store, encoders=made).find("dolo 650")
    assert match is not None and match.name == "Dolo 650 Tablet"


async def test_the_models_load_in_the_background_and_are_dropped_if_they_will_not() -> None:
    lookup = MedicineLookup(SpyStore(), encoders=NameEncoders(FakeBiEncoder(), FakeCrossEncoder()))  # type: ignore[arg-type]
    assert lookup.encoders is not None and not lookup.encoders.ready
    lookup.start()
    lookup.start()  # asked twice, loaded once
    for _ in range(100):
        if lookup.encoders.ready:
            break
        await asyncio.sleep(0.01)
    assert lookup.encoders.ready

    failing = MedicineLookup(SpyStore(), encoders=NameEncoders(FakeBiEncoder(fails=True)))  # type: ignore[arg-type]
    failing.start()
    for _ in range(100):
        if failing.encoders is None:
            break
        await asyncio.sleep(0.01)
    assert failing.encoders is None
    found = await failing.find("dolo 650")
    assert found is not None and found.name == "Dolo 650 Tablet"


def test_the_models_are_built_only_where_they_can_run(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert build_encoders(settings) is None  # switched off unless asked for
    # "cpu": no need to look for a GPU here.
    here = settings.model_copy(update={"medicines_encoders": True, "embedding_device": "cpu"})
    made = build_encoders(here)
    assert made is not None and not made.ready  # named, not loaded
    assert made.bi_encoder == "BAAI/bge-small-en-v1.5"
    assert made.cross_encoder == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    without = build_encoders(here.model_copy(update={"medicines_cross_encoder": ""}))
    assert without is not None and without.cross_encoder is None

    assert build_encoders(here.model_copy(update={"medicines_encoders": False})) is None
    # The production image has no sentence-transformers.
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *more: None if name == "sentence_transformers" else real(name, *more),
    )
    assert build_encoders(here) is None


def test_the_server_starts_the_catalogue_and_closes_it(settings: Settings) -> None:
    class Catalogue:
        started = closed = 0
        status = "ready"

        def start(self) -> None:
            self.started += 1

        async def aclose(self) -> None:
            self.closed += 1

    catalogue = Catalogue()
    app = create_app(settings, stt_provider=None, medicine_lookup=catalogue)  # type: ignore[arg-type]
    with TestClient(app) as client:
        assert (catalogue.started, catalogue.closed) == (1, 0)
        assert client.get("/health").json()["knowledge"]["medicines"] == {
            "configured": True,
            "collection": "clinexa_medicines",
            "status": "ready",
        }
    assert (catalogue.started, catalogue.closed) == (1, 1)


# --- is the catalogue there at all -------------------------------------------------------


async def settled(lookup: MedicineLookup) -> str:
    for _ in range(200):
        if lookup.status != "not_started":
            break
        await asyncio.sleep(0.005)
    return lookup.status


async def test_a_catalogue_that_was_never_indexed_is_reported_and_not_asked() -> None:
    client = AsyncQdrantClient(":memory:")
    try:
        # No collection at all, then one with nothing in it: `make medicines` was not run.
        missing = MedicineLookup(MedicineStore(client, "clinexa_medicines_missing"))
        assert missing.status == "not_started"
        missing.start()
        missing.start()  # checked once
        assert await settled(missing) == "not_indexed"
        assert await missing.find("dolo 650") is None

        empty = MedicineStore(client, "clinexa_medicines_empty")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await empty.ensure_collection()
        lookup = MedicineLookup(empty)
        lookup.start()
        assert await settled(lookup) == "not_indexed"
    finally:
        await client.close()


async def test_a_catalogue_that_is_there_is_ready_and_one_that_is_down_is_still_tried(
    store: MedicineStore,
) -> None:
    ready = MedicineLookup(store)
    ready.start()
    assert await settled(ready) == "ready"
    found = await ready.find("dolo 650")
    assert found is not None and found.name == "Dolo 650 Tablet"

    # Down when the server started. It may come back, so a lookup is still attempted.
    broken = BrokenStore()
    down = MedicineLookup(broken)  # type: ignore[arg-type]
    down.start()
    assert await settled(down) == "unreachable"
    assert await down.find("dolo 650") is None and broken.asked == 1
