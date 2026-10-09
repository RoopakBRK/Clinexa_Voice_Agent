"""Reads the three source files in ``data/`` into one list of medicines.

    A_Z_medicines_dataset_of_India.csv   branded products, with pack and composition
    jan_aushdi.pdf                       the Jan Aushadhi (PMBJP) list of generic medicines
    nlem2022.pdf                         National List of Essential Medicines 2022

Nothing here is guessed. A strength is taken only where the source states it, and is left
empty where two products of the same name disagree.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.core.logging import get_logger

log = get_logger(__name__)

AZ_FILE = "A_Z_medicines_dataset_of_India.csv"
JAN_AUSHADHI_FILE = "jan_aushdi.pdf"
NLEM_FILE = "nlem2022.pdf"

Source = Literal["nlem", "jan_aushadhi", "az"]
SOURCE_NAMES: dict[Source, str] = {
    "nlem": "National List of Essential Medicines 2022",
    "jan_aushadhi": "Jan Aushadhi (PMBJP) product list",
    "az": "A to Z medicines dataset of India",
}


class Medicine(BaseModel):
    id: str  # "az:4821", "ja:34", "nlem:2.2.2"
    name: str  # as the source writes it
    source: Source
    # "Amoxycillin 500 mg + Clavulanic Acid 125 mg"
    composition: str | None = None
    # "500 mg + 125 mg". Empty when the source gives none, or gives more than one.
    strength: str | None = None
    # What it comes as, where the name or the pack says: tablets, capsules, ml, injections, sachets.
    unit: str | None = None
    # "strip of 10 tablets", "10's"
    pack: str | None = None
    manufacturer: str | None = None
    discontinued: bool = False


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _spaced(strength: str) -> str:
    """ "500mg" -> "500 mg", "30mg/5ml" -> "30 mg/5 ml"."""
    return _clean(re.sub(r"(\d)([A-Za-z%])", r"\1 \2", strength))


_UNITS: list[tuple[str, re.Pattern[str]]] = [
    ("tablets", re.compile(r"\btab(let)?s?\b")),
    ("capsules", re.compile(r"\bcap(sule)?s?\b")),
    (
        "injections",
        re.compile(r"\b(injection|injectable|vial|ampoule|infusion|syringe|penfill|cartridge)s?\b"),
    ),
    ("sachets", re.compile(r"\bsachets?\b")),
    (
        "ml",
        re.compile(r"\b(syrup|suspension|solution|liquid|drops?|linctus|expectorant|elixir)\b"),
    ),
]


def unit_for(*texts: str | None) -> str | None:
    """What a product comes as, from its name or pack: tablets, capsules, ml..."""
    text = " ".join(t.lower() for t in texts if t)
    return next((unit for unit, pattern in _UNITS if pattern.search(text)), None)


_STRENGTH = re.compile(
    r"(\d+(?:\.\d+)?\s?(?:mcg|mg|gm|g|iu|%)(?:\s?w/[wv])?)"
    r"(?:\s?(?:per|/)\s?(\d+(?:\.\d+)?\s?ml|ml))?",
    re.IGNORECASE,
)


def strengths_in(text: str) -> str | None:
    """Every strength written in a generic name: "100mg and 325mg Tablets" -> "100 mg + 325 mg"."""
    found = [
        _spaced(amount) + (f"/{_spaced(volume)}" if volume else "")
        for amount, volume in _STRENGTH.findall(text)
    ]
    return " + ".join(found) or None


# --- A to Z medicines dataset of India ---------------------------------------------------

_INGREDIENT = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")


def _ingredient(text: str | None) -> tuple[str, str | None] | None:
    """ "Amoxycillin  (500mg) " -> ("Amoxycillin", "500 mg")."""
    text = _clean(text)
    if not text:
        return None
    if (match := _INGREDIENT.match(text)) is None:
        return text, None
    return _clean(match.group(1)), _spaced(match.group(2)) or None


def az_medicine(row: dict[str, str]) -> Medicine | None:
    name = _clean(row.get("name"))
    if not name:
        return None
    parts = [
        p
        for key in ("short_composition1", "short_composition2")
        if (p := _ingredient(row.get(key)))
    ]
    strengths = [strength for _, strength in parts]
    pack = _clean(row.get("pack_size_label")) or None
    return Medicine(
        id=f"az:{_clean(row.get('id'))}",
        name=name,
        source="az",
        composition=" + ".join(f"{n} {s}".strip() if s else n for n, s in parts) or None,
        strength=" + ".join(s for s in strengths if s) if parts and all(strengths) else None,
        unit=unit_for(name, pack),
        pack=pack,
        manufacturer=_clean(row.get("manufacturer_name")) or None,
        discontinued=_clean(row.get("Is_discontinued")).upper() == "TRUE",
    )


def read_az(path: Path) -> Iterator[Medicine]:
    with path.open(newline="", encoding="utf-8", errors="replace") as file:
        for row in csv.DictReader(file):
            if (medicine := az_medicine(row)) is not None:
                yield medicine


# --- Jan Aushadhi --------------------------------------------------------------------------


def jan_aushadhi_medicine(cells: list[str]) -> Medicine | None:
    """One row of the list: serial number, drug code, generic name, unit size."""
    if len(cells) < 4 or not cells[0].isdigit() or not cells[1].isdigit():
        return None
    name, pack = cells[2], cells[3]
    return Medicine(
        id=f"ja:{cells[1]}",
        name=name,
        source="jan_aushadhi",
        composition=name,
        strength=strengths_in(name),
        unit=unit_for(name, pack),
        pack=pack,
    )


def read_jan_aushadhi(path: Path) -> Iterator[Medicine]:
    import pymupdf  # only needed when the catalogue is built

    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        for page in document:
            for table in page.find_tables().tables:
                for row in table.extract():
                    cells = [_clean(cell) for cell in row if cell and cell.strip()]
                    if (medicine := jan_aushadhi_medicine(cells)) is not None:
                        yield medicine


# --- National List of Essential Medicines 2022 ---------------------------------------------

_NLEM_NUMBER = re.compile(r"^(\d+(?:\.\d+){1,4})\b[\s\-–]*(.*)$")
_NLEM_LEVEL = re.compile(r"^[PST](\s*,\s*[PST])*$")
_NLEM_HEADER = re.compile(r"^(medicine|level of|healthcare|dosage form|section\b)", re.IGNORECASE)
_NLEM_END = "alphabetical list of medicines"


def _nlem_name(lines: list[str]) -> str:
    name = _clean(" ".join(lines))
    name = re.sub(r"\*+", "", name)  # footnote marks
    return _clean(re.sub(r"\s*\([AB]\)", "", name))  # "(A) + ... (B)" labels of a combination


def nlem_entries(lines: Iterable[str]) -> Iterator[tuple[str, str, list[str]]]:
    """(number, medicine, [form and strength lines]) for each medicine in the page text.

    A medicine is a numbered line, then its name, then the level of healthcare (P, S, T),
    then one line for each form. A numbered line with no level after it is a heading.
    """
    number = ""
    name: list[str] = []
    forms: list[str] = []
    in_forms = False

    def finished() -> Iterator[tuple[str, str, list[str]]]:
        if number and in_forms and name:
            yield number, _nlem_name(name), forms

    for raw in lines:
        line = _clean(raw)
        if not line or line.isdigit():  # a blank, or a page number
            continue
        starts = _NLEM_NUMBER.match(line)
        if starts or line.startswith("*") or _NLEM_HEADER.match(line):
            yield from finished()
            number, name, forms, in_forms = "", [], [], False
            if starts:
                number, name = starts.group(1), [starts.group(2)] if starts.group(2) else []
        elif not number:
            continue
        elif not in_forms and _NLEM_LEVEL.match(line):
            in_forms = True
        elif not in_forms:
            name.append(line)
        elif _NLEM_LEVEL.match(line):
            continue  # some forms carry a level of their own
        elif line[0].isupper() or not forms:
            forms.append(line)
        else:
            forms[-1] = f"{forms[-1]} {line}"  # a form that ran on to the next line
    yield from finished()


def read_nlem(path: Path) -> Iterator[Medicine]:
    import pymupdf  # only needed when the catalogue is built

    lines: list[str] = []
    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        for page in document:
            text = page.get_text("text")
            # The lists end where the index begins. (The contents page names it too, lower down.)
            if text.strip().lower().startswith(_NLEM_END):
                break
            lines.extend(text.splitlines())
            lines.append("*")  # nothing carries over a page

    for number, name, forms in nlem_entries(lines):
        if not re.search(r"[A-Za-z]{3}", name):
            continue
        yield Medicine(id=f"nlem:{number}", name=name, source="nlem", composition=name)
        for index, form in enumerate(forms, start=1):
            if not (unit_for(form) or re.search(r"\d", form)):
                continue  # "As licensed": not a form anybody names
            yield Medicine(
                id=f"nlem:{number}:{index}",
                name=f"{name} {form}",
                source="nlem",
                composition=name,
                strength=strengths_in(form),
                unit=unit_for(form),
                pack=form,
            )


# --- All three ------------------------------------------------------------------------------


def merged(medicines: Iterable[Medicine]) -> list[Medicine]:
    """One entry for each name. The first one read is kept.

    Where two products share a name but not a strength (the same brand from two makers,
    say), the strength is dropped: there is no telling which one a person means.
    """
    by_name: dict[str, Medicine] = {}
    for medicine in medicines:
        key = medicine.name.lower()
        kept = by_name.get(key)
        if kept is None:
            by_name[key] = medicine
        elif kept.strength != medicine.strength and kept.strength is not None:
            by_name[key] = kept.model_copy(update={"strength": None})
        if kept is not None and kept.discontinued and not medicine.discontinued:
            by_name[key] = by_name[key].model_copy(update={"discontinued": False})
    return list(by_name.values())


def load_catalog(data_dir: Path) -> list[Medicine]:
    """Every medicine in the three files, essential and generic names first."""
    readers = [(NLEM_FILE, read_nlem), (JAN_AUSHADHI_FILE, read_jan_aushadhi), (AZ_FILE, read_az)]
    found: list[Medicine] = []
    for file_name, read in readers:
        path = data_dir / file_name
        if not path.exists():
            log.warning("medicines.source_missing", file=file_name)
            continue
        before = len(found)
        found.extend(read(path))
        log.info("medicines.source_read", file=file_name, medicines=len(found) - before)
    return merged(found)
