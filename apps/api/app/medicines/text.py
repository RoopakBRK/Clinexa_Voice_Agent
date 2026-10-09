"""How a medicine name is broken up, compared and turned into a sparse vector.

Everything here is plain string work: no model, no network. The same functions are
used when the catalogue is indexed and when a name someone said is looked up, so the
two always agree.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence
from difflib import SequenceMatcher
from functools import lru_cache

_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")

# Words that say what kind of thing it is, not which medicine. They help choose between
# a syrup and a tablet of the same name, and are left out of the name itself.
FORM_WORDS = frozenset(
    [
        "tablet",
        "tablets",
        "tab",
        "tabs",
        "capsule",
        "capsules",
        "cap",
        "caps",
        "syrup",
        "suspension",
        "injection",
        "inj",
        "cream",
        "gel",
        "ointment",
        "drops",
        "drop",
        "solution",
        "lotion",
        "powder",
        "sachet",
        "sachets",
        "spray",
        "inhaler",
        "respules",
        "rotacaps",
        "infusion",
        "liquid",
        "oral",
        "soap",
        "shampoo",
        "granules",
        "lozenges",
        "patch",
        "pessary",
        "suppository",
        "vial",
        "ampoule",
        "injectable",
        "emulsion",
        "paste",
        "rinse",
        "wash",
        "linctus",
        "expectorant",
        "elixir",
    ]
)
# Units and pharmacopoeia marks: on almost every label, and never what a person says to
# tell two medicines apart.
NOISE_WORDS = frozenset(
    [
        "mg",
        "mcg",
        "ml",
        "gm",
        "kg",
        "iu",
        "ip",
        "bp",
        "usp",
        "and",
        "with",
        "per",
        "in",
        "of",
        "for",
        "the",
    ]
)
# Single letters that are units after a number ("5 g", "10's", "1 % w/w") and part of the
# name anywhere else: "S-Amlong" is not "Amlong".
_UNIT_LETTERS = frozenset("gswv")


def tokens(text: str) -> list[str]:
    """Lower-case words and numbers: "Glycomet-GP 0.5mg" -> glycomet, gp, 0.5, mg."""
    return _TOKEN.findall(text.lower())


def is_number(token: str) -> bool:
    return token[0].isdigit()


def is_noise(words: Sequence[str], index: int) -> bool:
    """Whether the lower-case word at ``index`` is a unit or a mark, not part of the name."""
    word = words[index]
    if word in _UNIT_LETTERS:
        before = words[index - 1] if index else ""
        return bool(before) and (
            is_number(before) or before in NOISE_WORDS or before in _UNIT_LETTERS
        )
    return word in NOISE_WORDS


def name_tokens(text: str) -> list[str]:
    """Lower-case words and numbers, without the units and marks."""
    found = tokens(text)
    return [token for index, token in enumerate(found) if not is_noise(found, index)]


def content_tokens(text: str) -> list[str]:
    """The words that say which medicine it is: no form words, no units."""
    return [token for token in name_tokens(text) if token not in FORM_WORDS]


def form_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if t in FORM_WORDS}


def brand_phrase(text: str) -> str:
    """The leading words of a name run together: "Pan D 40 Tablet" -> "pand".

    Speech recognition splits and joins words freely ("eco sprin", "pandy"), so the
    start of a name is also compared as one run of letters.
    """
    letters: list[str] = []
    for token in name_tokens(text):
        if is_number(token) or token in FORM_WORDS or len(letters) == 3:
            break
        letters.append(token)
    return "".join(letters)


@lru_cache(maxsize=50_000)
def sound(word: str) -> str:
    """A rough key for how a word sounds: its first letter and its consonants.

    Vowels are what speech recognition gets wrong most ("glycomate" for "Glycomet"), so
    they are dropped after the first letter, and v and w are heard as one ("Atorwa"). Not a
    full phonetic algorithm: just enough to bring near-misses of a name together.
    """
    word = word.replace("ph", "f").replace("ck", "k").replace("sh", "s").replace("th", "t")
    out: list[str] = []
    for index, letter in enumerate(word):
        following = word[index + 1] if index + 1 < len(word) else ""
        if letter == "c":
            letter = "s" if following in "eiy" else "k"
        elif letter == "q":
            letter = "k"
        elif letter == "z":
            letter = "s"
        elif letter == "x":
            letter = "ks"
        elif letter == "w":
            letter = "v"
        if index > 0 and letter in "aeiouyh":
            continue
        if not out or out[-1] != letter:
            out.append(letter)
    return "".join(out)


def alike(a: str, b: str) -> float:
    """How alike two words are, from 0 to 1. Numbers are alike only if they are equal."""
    if a == b:
        return 1.0
    if is_number(a) or is_number(b):
        return 1.0 if is_number(a) and is_number(b) and float(a) == float(b) else 0.0
    if min(len(a), len(b)) <= 3:
        return 0.0  # "pan" is not "pen"
    if sound(a)[0] != sound(b)[0]:
        return 0.0  # nor is "pandy" "Andy": a name is not misheard into one that starts differently
    return SequenceMatcher(None, a, b).ratio()


def _edits(a: str, b: str) -> int:
    """How many letters must be added, dropped or changed to turn one word into the other."""
    before = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        row = [i]
        for j, y in enumerate(b, start=1):
            row.append(min(before[j] + 1, row[j - 1] + 1, before[j - 1] + (x != y)))
        before = row
    return before[-1]


def near(a: str, b: str, *, by_sound: bool = True) -> bool:
    """Whether one word could be the other, misheard.

    One letter out ("Dollo", "Atorwa"), or the same consonants with most of the letters
    ("Glycomate", "Krosin"). Not merely alike: "Shelcal" is not "Selca", and the
    catalogue does not hold every medicine, so the difference has to be kept.
    """
    score = alike(a, b)
    if score in (0.0, 1.0):
        return score == 1.0
    return _edits(a, b) <= 1 or (by_sound and score >= 0.6 and sound(a) == sound(b))


def _feature(kind: str, value: str) -> int:
    # CRC32, not hash(): the same on every machine and in every process.
    return zlib.crc32(f"{kind}:{value}".encode())


def features(name: str) -> dict[int, float]:
    """A name as a sparse vector: whole words, numbers, letter groups and sound keys.

    The index of each feature is a 32-bit hash of it. Qdrant weighs each one by how rare
    it is across the catalogue (IDF), so "tablet" counts for little and "glycomet" for a
    lot.
    """
    found: set[int] = set()
    for token in name_tokens(name):
        if is_number(token):
            found.add(_feature("n", str(float(token))))
            continue
        found.add(_feature("w", token))
        if token in FORM_WORDS:
            continue
        marked = f"^{token}$"
        found.update(_feature("g", marked[i : i + 3]) for i in range(len(marked) - 2))
        if len(key := sound(token)) >= 2:
            found.add(_feature("p", key))
    run = brand_phrase(name)
    if len(run) >= 3:
        found.update(_feature("c", run[i : i + 3]) for i in range(len(run) - 2))
        found.add(_feature("pc", sound(run)))
    return dict.fromkeys(sorted(found), 1.0)
