"""Languages the web onboarding channel offers.

Deepgram Nova-3 transcribes all of them. Aura has no Indian-language voices yet
(checked against ``GET /v1/models`` on 2026-10-04), so only English can be
spoken; the others are reported to the page as unavailable until a voice exists.
Give a language a ``tts_model`` here to switch it on.
"""

from __future__ import annotations

from pydantic import BaseModel


class Language(BaseModel):
    code: str
    name: str
    # Deepgram ``language`` for speech-to-text.
    stt_language: str
    # Deepgram Aura voice, or None when there is no voice for this language.
    # ``DEFAULT_VOICE`` means "use DEEPGRAM_TTS_MODEL from settings".
    tts_model: str | None = None


DEFAULT_VOICE = "default"

_LANGUAGES = [
    Language(code="en", name="English", stt_language="en-IN", tts_model=DEFAULT_VOICE),
    # "multi" follows people who mix Hindi and English in one sentence.
    Language(code="hi", name="Hindi", stt_language="multi"),
    Language(code="kn", name="Kannada", stt_language="kn"),
    Language(code="ta", name="Tamil", stt_language="ta"),
    Language(code="te", name="Telugu", stt_language="te"),
    Language(code="mr", name="Marathi", stt_language="mr"),
    Language(code="bn", name="Bengali", stt_language="bn"),
    Language(code="ml", name="Malayalam", stt_language="ml"),
    Language(code="gu", name="Gujarati", stt_language="gu"),
]

LANGUAGES: dict[str, Language] = {lang.code: lang for lang in _LANGUAGES}
_BY_NAME = {lang.name.lower(): lang for lang in _LANGUAGES}


def resolve_language(value: str) -> Language | None:
    """Look a language up by code ("hi", "hi-IN") or English name ("Hindi")."""
    key = value.strip().lower()
    return LANGUAGES.get(key.split("-")[0]) or _BY_NAME.get(key)
