"""Text normalisation for extracted PDF blocks."""

from __future__ import annotations

import re
import string

_LIGATURES = str.maketrans(
    {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "\xa0": " ", "­": ""}
)
_PICTURE = re.compile(r"\*\*==>.*?<==\*\*", re.S)
# Reference markers such as "_[34]_", "_(1, 12)_", "(1)" after punctuation-free text.
_CITATION = re.compile(r"\s*_(?:\[\d+(?:\s*[,–-]\s*\d+)*\]|\(\d+(?:\s*[,–-]\s*\d+)*\))_")
# Footnote markers ("adrenaline<sup>1</sup>"), but not unit exponents (m², cm²).
_SUPERSCRIPT_REF = re.compile(r"(?<!\bm)(?<!\bcm)<sup>\s*[\d,\s*†‡a-z]{1,6}\s*</sup>")
# Remaining inline tags from the layout model keep their contents ("m<sup>2</sup>" -> "m2").
_INLINE_TAG = re.compile(r"</?(?:sup|sub|mark|b|i|u|em|strong)>", re.I)
_HTML_BREAK = re.compile(r"<br\s*/?>")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_ITALIC = re.compile(r"(?<![\w*])_(?!_)(.+?)(?<!_)_(?!\w)", re.S)
_MD_HEADING_MARK = re.compile(r"^#{1,6}\s*")
# PDF text layers often split words at ligatures ("diffi culty", "infl ating").
_SPLIT_LIGATURE = re.compile(r"(\w)(ffi|ffl|fi|fl|ff) (?=[a-z])")
_LEADING_SPLIT_LIGATURE = re.compile(r"(?<![A-Za-z])(fi|fl|ff) (?=[a-z]{2,})")  # "fi rst", "fl ow"
# Drop-cap headings extracted as "7 S evere" / "1 T riage" (not "A"/"I": real words).
_DROP_CAP = re.compile(r"^((?:\d+(?:\.\d+)*\s+)?)([B-HJ-Z]) (?=[a-z]{3,})")
# Line-break hyphenation that survived as "sub- stance".
_HYPHEN_BREAK = re.compile(r"(\w)- (?=[a-z])")
_SPACES = re.compile(r"[ \t]+")
# Symbol-font bullets (Wingdings "u" = ▶) extracted as a literal letter: "- u Stay calm".
_GLYPH_BULLET = re.compile(r"(?m)^((?:[-*•■]\s+)?)u\s+(?=[A-Z])")
_BULLET_PREFIX = re.compile(r"^\s*(?:[-*•●■▪◦‣–]|\d+[.)])\s+")

ACRONYMS = frozenset(
    {
        "HIV",
        "AIDS",
        "TB",
        "COPD",
        "ART",
        "ARV",
        "ITN",
        "ITNS",
        "IRS",
        "SMC",
        "PMC",
        "MDA",
        "CVC",
        "CVCS",
        "BSI",
        "BSIS",
        "ORS",
        "IMCI",
        "MNS",
        "PTSD",
        "ADHD",
        "ECG",
        "BP",
        "IV",
        "IM",
        "WHO",
        "STI",
        "STIS",
        "UTI",
        "ENT",
        "CPR",
        "HPV",
        "BCG",
        "DTP",
        "MMR",
        "IPTP",
        "PDMC",
        "ACT",
        "ACTS",
        "RDT",
        "RDTS",
        "G6PD",
        "APC",
        "TIA",
        "NCD",
        "NCDS",
        "DKA",
        "ECP",
        "MSE",
        "MC",
        "GDG",
        "GPS",
        "CLABSI",
        "MMIS",
        "ICU",
        "HAI",
        "HAIS",
    }
)


def normalise(text: str) -> str:
    """Clean inline markdown and PDF artefacts, keeping line structure."""
    text = text.translate(_LIGATURES)
    text = _PICTURE.sub("", text)
    text = _CITATION.sub("", text)
    text = _SUPERSCRIPT_REF.sub("", text)
    text = _INLINE_TAG.sub("", text)
    text = _HTML_BREAK.sub(" ", text)
    text = _BOLD.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    text = _SPLIT_LIGATURE.sub(r"\1\2", text)
    text = _LEADING_SPLIT_LIGATURE.sub(r"\1", text)
    text = _HYPHEN_BREAK.sub(r"\1", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    return _GLYPH_BULLET.sub(r"\1", "\n".join(line for line in lines if line))


def clean_paragraph(text: str) -> str:
    return " ".join(normalise(text).split())


def clean_list_item(text: str) -> str:
    return "- " + _BULLET_PREFIX.sub("", clean_paragraph(text))


def clean_heading(text: str) -> str:
    heading = clean_paragraph(_MD_HEADING_MARK.sub("", text)).strip(" .:")
    return _DROP_CAP.sub(r"\1\2", heading)


def clean_table(text: str) -> str:
    rows = [row.strip() for row in normalise(text).splitlines()]
    return "\n".join(row for row in rows if row.strip("| "))


def starts_with_bullet(text: str) -> bool:
    return bool(re.match(r"^\s*[-*•●■▪◦‣]", _MD_HEADING_MARK.sub("", text)))


_SMALL_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "and/or",
        "as",
        "at",
        "by",
        "for",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


def display_case(heading: str) -> str:
    """ALL-CAPS headings ("FACE SYMPTOMS") → "Face Symptoms", keeping acronyms."""
    letters = [c for c in heading if c.isalpha()]
    if not letters or not all(c.isupper() for c in letters):
        return heading
    words = []
    for i, word in enumerate(heading.split(" ")):
        core = word.strip(string.punctuation)
        if i and word.lower() in _SMALL_WORDS:
            words.append(word.lower())
            continue
        # Known acronyms, and short vowel-less tokens ("GDG", "HCW"), stay upper-case.
        is_acronym = core.upper() in ACRONYMS or (
            1 < len(core) <= 4 and not any(c in "AEIOU" for c in core.upper())
        )
        words.append(word if is_acronym else string.capwords(word.lower()))
    return " ".join(words)
