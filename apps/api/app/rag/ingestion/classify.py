"""Rule-based chunk labelling: topic, population, chunk type, retrievability.

Rules (not an LLM) keep ingestion deterministic, fast and auditable; the labels
drive metadata filters, so their precision matters more than their coverage.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from app.schemas.clinical import ChunkType

# Whole words (plural "s"/"es" allowed) unless marked as a stem with a trailing "*".
# Word boundaries matter: "ear" must not match "near", nor "ors" match "errors".
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "respiratory": (
        "cough",
        "pneumonia",
        "asthma",
        "wheez*",
        "bronchi*",
        "copd",
        "respiratory",
        "breathing difficult*",
        "difficult breathing",
        "difficulty breathing",
        "shortness of breath",
        "croup",
        "stridor",
    ),
    "cardiovascular": (
        "hypertension",
        "blood pressure",
        "heart failure",
        "cardiac",
        "chest pain",
        "stroke",
        "cholesterol",
        "palpitation*",
        "cardiovascular",
    ),
    "malaria": (
        "malaria",
        "plasmodium",
        "antimalarial*",
        "artemisinin",
        "mosquito*",
        "vector control",
    ),
    "infectious_disease": (
        "fever",
        "sepsis",
        "infection*",
        "antibiotic*",
        "meningitis",
        "measles",
        "dengue",
        "typhoid",
        "hiv",
        "tuberculosis",
        "tb",
    ),
    "infection_prevention": (
        "catheter*",
        "hand hygiene",
        "aseptic",
        "bloodstream infection*",
        "chlorhexidine",
        "infection prevention",
        "sterile",
    ),
    "gastrointestinal": (
        "diarrhoea",
        "diarrhea",
        "vomiting",
        "dehydration",
        "abdominal pain",
        "constipation",
        "ors",
        "oral rehydration",
        "dysentery",
        "worms",
    ),
    "mental_health": (
        "depression",
        "anxiety",
        "psychosis",
        "suicide",
        "self-harm",
        "mental health",
        "bipolar",
        "stress*",
        "emotional",
        "behavioural disorder*",
    ),
    "neurological": (
        "seizure*",
        "epilepsy",
        "convulsion*",
        "headache*",
        "dementia",
        "neurolog*",
        "unconscious*",
        "coma",
    ),
    "substance_use": ("alcohol", "drug use", "opioid*", "substance use", "tobacco", "smoking"),
    "maternal_reproductive": (
        "pregnan*",
        "antenatal",
        "postpartum",
        "contracept*",
        "breastfe*",
        "menstrua*",
        "vaginal",
        "labour",
    ),
    "neonatal": ("newborn*", "neonat*", "preterm", "low birth weight", "umbilical", "jaundice"),
    "nutrition": (
        "malnutrition",
        "wasting",
        "stunting",
        "feeding",
        "vitamin*",
        "anaemia",
        "micronutrient*",
        "growth",
    ),
    "endocrine": ("diabetes", "glucose", "insulin", "thyroid", "hypoglycaemia"),
    "musculoskeletal": (
        "joint pain",
        "back pain",
        "arthritis",
        "fracture*",
        "musculoskeletal",
        "gout",
    ),
    "skin": ("rash*", "skin", "eczema", "itch*", "wound*", "burn*", "ulcer*"),
    "eye_ent": ("eye*", "vision", "ear", "hearing", "otitis", "throat", "tonsil*"),
    "urinary_renal": ("urine", "urinary", "kidney*", "renal", "dysuria"),
    "emergency": (
        "emergency",
        "resuscitation",
        "shock",
        "airway*",
        "anaphylaxis",
        "poisoning",
        "envenom*",
        "snake bite",
        "trauma",
        "triage",
    ),
    "immunization": ("vaccin*", "immuniz*", "immunis*"),
    "medication": (
        "dose*",
        "dosage*",
        "mg/kg",
        "tablet*",
        "side-effect*",
        "side effect*",
        "contraindicat*",
        "drug interaction*",
    ),
}


def _keyword_pattern(keywords: Sequence[str]) -> re.Pattern[str]:
    parts = []
    for kw in keywords:
        if kw.endswith("*"):
            parts.append(re.escape(kw[:-1]))
        else:
            parts.append(re.escape(kw) + r"(?:s|es)?\b")
    return re.compile(r"\b(?:" + "|".join(parts) + ")")


_TOPIC_PATTERNS = {topic: _keyword_pattern(kws) for topic, kws in TOPIC_KEYWORDS.items()}

_WARNING = re.compile(
    r"danger signs?|red flags?|emergency signs?|refer (?:urgently|immediately|same day)|"
    r"urgent(?:ly)? refer|needing urgent attention|urgent attention|immediate referral|"
    r"life[- ]threatening|seek (?:urgent|immediate) (?:medical )?care|call for help",
    re.I,
)
_RECOMMENDATION = re.compile(
    r"\b(?:strong|conditional|weak) recommendation\b|\bgood practice statement\b|"
    r"\brecommendation\s*\d|^recommendations?\b",
    re.I | re.M,
)
_CHILD = re.compile(
    r"\b(?:child|children|infant|newborn|neonat\w*|paediatric|pediatric|adolescen\w*)\b", re.I
)
_ADULT = re.compile(r"\badults?\b", re.I)
_PREGNANCY = re.compile(r"\bpregnan\w*", re.I)

_NON_RETRIEVABLE_SECTION = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(?:references?|bibliography|acknowledge?ments?|contributors|"
    r"declarations? of interests?|abbreviations(?: and acronyms)?|acronyms|contents|"
    r"table of contents|index|foreword|preface|sponsors/funding|web annex|"
    r"list of (?:tables|figures|boxes)|search strateg\w*|annex \d+\.? (?:pico questions|methods)|"
    r"embase|medline|cochrane database|cinahl)\b",
    re.I,
)
_FRONT_MATTER = re.compile(
    r"\bISBN\b|all rights reserved|creative commons|cataloguing-in-publication|"
    r"suggested citation|third-party materials",
    re.I,
)
_TOC_ROW = re.compile(r"\|\s*\d{1,4}\s*\|\s*$")


def classify_topics(path: Sequence[str], text: str, fallback: Sequence[str]) -> list[str]:
    """Topics ranked by keyword hits; section headings weigh 3x body text."""
    heading = " ".join(path).lower()
    body = f" {text.lower()} "
    scores: Counter[str] = Counter()
    for topic, pattern in _TOPIC_PATTERNS.items():
        score = 3 * len(pattern.findall(heading)) + len(pattern.findall(body))
        if score >= 2:
            scores[topic] = score
    ranked = [topic for topic, _ in scores.most_common(3)]
    return ranked or list(fallback[:1])


def classify_population(doc_population: str, path: Sequence[str]) -> str:
    """Document population, narrowed by section headings for mixed documents."""
    if doc_population != "all":
        return doc_population
    heading = " ".join(path)
    if _PREGNANCY.search(heading):
        return "pregnancy"
    if _CHILD.search(heading):
        return "child"
    if _ADULT.search(heading):
        return "adult"
    return "all"


def classify_chunk_type(path: Sequence[str], text: str, table_share: float) -> ChunkType:
    heading = " ".join(path)
    if _WARNING.search(heading) or _WARNING.search(text):
        return ChunkType.WARNING
    if _RECOMMENDATION.search(heading) or _RECOMMENDATION.search(text):
        return ChunkType.RECOMMENDATION
    if table_share >= 0.5:
        return ChunkType.TABLE
    return ChunkType.TEXT


def is_retrievable(path: Sequence[str], text: str) -> bool:
    """False for front matter, references, contents pages and bare page-number tables."""
    if any(_NON_RETRIEVABLE_SECTION.match(part) for part in path):
        return False
    if _FRONT_MATTER.search(text):
        return False
    rows = [row for row in text.splitlines() if row.startswith("|")]
    return not (len(rows) >= 4 and sum(bool(_TOC_ROW.search(r)) for r in rows) / len(rows) >= 0.6)
