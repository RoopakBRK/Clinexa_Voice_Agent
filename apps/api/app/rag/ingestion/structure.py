"""Recover each block's section path (heading hierarchy) across pages."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from app.rag.ingestion.clean import (
    clean_heading,
    clean_list_item,
    clean_paragraph,
    clean_table,
    display_case,
    starts_with_bullet,
)
from app.rag.ingestion.manifest import StructureStrategy
from app.rag.models import BlockKind, PageExtraction

_NUMBERED = re.compile(r"^(?:chapter\s+|section\s+)?(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+(\S.*)$", re.I)
_BARE_NUMBER = re.compile(r"^(?:chapter\s+)?(\d{1,2})\.?$", re.I)
# Headings that start a new top-level part regardless of the current numbering.
_TOP_LEVEL = re.compile(
    r"^(?:annex(?:es)?|appendix|appendices|references|bibliography|glossary|acknowledge?ments|"
    r"abbreviations|acronyms|foreword|preface|executive summary|contents|table of contents|index)\b",
    re.I,
)
_PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?$", re.I)
_MAX_HEADING_CHARS = 100
# A running header/footer seen on more than this share of pages is document
# boilerplate (e.g. the malaria title line), not a section name.
_BOILERPLATE_SHARE = 0.3
_BOILERPLATE_MIN_PAGES = 5
_BOILERPLATE_MIN_CHARS = 40
_BOILERPLATE_TEXT_PAGES = 6
_BOILERPLATE_TEXT_SHARE = 0.04


@dataclass(frozen=True, slots=True)
class Unit:
    """A cleaned content block with its position in the document structure."""

    kind: BlockKind  # TEXT, LIST_ITEM or TABLE
    text: str
    page: int
    path: tuple[str, ...]


# Module/section codes in running headers ("DEP", "DEP 1", "MC 2"), not names.
_CODE_LIKE = re.compile(r"^[A-Za-z]{1,4}(?:\s?\d+)?$")


def _is_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 3 and all(c.isupper() for c in letters)


def _pick_running_title(texts: list[str]) -> str | None:
    """The section name among a page's running header/footer texts: prefer an
    ALL-CAPS name over mixed-case captions, ignore codes, then take the longest."""
    names = [_leading_caps_name(t) for t in texts if not _CODE_LIKE.match(t)]
    caps = [t for t in names if _is_all_caps(t)]
    pool = caps or names
    return max(pool, key=len) if pool else None


def _leading_caps_name(text: str) -> str:
    """ "DEPRESSION Assessment" -> "DEPRESSION": a module name followed by a caption."""
    words = text.split()
    lead: list[str] = []
    for word in words:
        letters = [c for c in word if c.isalpha()]
        if word != "&" and not (letters and all(c.isupper() for c in letters)):
            break  # first mixed-/lower-case word starts the caption
        lead.append(word)
    name = " ".join(lead)
    return name if lead and len(lead) < len(words) and _is_all_caps(name) else text


def _is_heading(text: str) -> bool:
    if len(text) < 3 or len(text) > _MAX_HEADING_CHARS:
        return False
    digits = sum(c.isdigit() for c in text)
    return digits / len(text) < 0.3 and any(c.isalpha() for c in text)


def running_titles(pages: list[PageExtraction]) -> dict[int, str | None]:
    """Per page, the running header/footer text that names a section (if any)."""
    candidates: dict[int, list[str]] = {}
    for page in pages:
        texts = []
        for block in page.blocks:
            if block.kind in (BlockKind.PAGE_HEADER, BlockKind.PAGE_FOOTER):
                text = clean_heading(block.text)
                if text and not _PAGE_NUMBER.match(text) and _is_heading(text):
                    texts.append(text)
        candidates[page.page_number] = texts
    counts = Counter(t for texts in candidates.values() for t in set(texts))
    threshold = max(_BOILERPLATE_SHARE * len(pages), _BOILERPLATE_MIN_PAGES)
    boilerplate = {t for t, n in counts.items() if n > threshold}
    titles: dict[int, str | None] = {}
    for page_number, texts in candidates.items():
        picked = _pick_running_title([t for t in texts if t not in boilerplate])
        titles[page_number] = display_case(picked) if picked else None
    return titles


def repeated_boilerplate(pages: list[PageExtraction]) -> set[str]:
    """Long paragraphs repeated on many pages (e.g. the document title in a page
    header that the layout model mislabelled as body text)."""
    seen: Counter[str] = Counter()
    for page in pages:
        for text in {clean_paragraph(b.text) for b in page.blocks if b.kind is BlockKind.TEXT}:
            if len(text) >= _BOILERPLATE_MIN_CHARS:
                seen[text] += 1
    threshold = max(_BOILERPLATE_TEXT_PAGES, _BOILERPLATE_TEXT_SHARE * len(pages))
    return {text for text, n in seen.items() if n >= threshold}


class SectionTracker:
    """Stateful walk over a document's headings, producing section paths."""

    def __init__(self, strategy: StructureStrategy) -> None:
        self.strategy = strategy
        self._base: list[str] = []  # from ToC or running header
        # (section number, title), e.g. ((3, 2, 1), "3.2.1 Chlorhexidine body wash")
        self._numbered: list[tuple[tuple[int, ...], str]] = []
        self._levels: list[str] = []  # markdown-depth stack / unnumbered sub-headings
        self._pending_chapter: str | None = None  # a bare "4" awaiting its title

    @property
    def path(self) -> tuple[str, ...]:
        if self.strategy is StructureStrategy.NUMBERED:
            parts = [*(title for _, title in self._numbered), *self._levels]
        else:
            parts = [*self._base, *self._levels]
        deduped: list[str] = []
        for part in parts:
            if not deduped or deduped[-1].casefold() != part.casefold():
                deduped.append(part)
        return tuple(deduped)

    def start_page(self, page: PageExtraction, running: str | None) -> None:
        if self.strategy is StructureStrategy.TOC and page.toc_path != self._base:
            self._base = list(page.toc_path)
            self._levels = []
        elif self.strategy is StructureStrategy.RUNNING and running and [running] != self._base:
            self._base = [running]
            self._levels = []

    def chapter_number(self, number: str) -> None:
        """A heading that is only a number: the chapter title follows separately."""
        self._pending_chapter = number

    def heading(self, text: str, md_level: int) -> None:
        match self.strategy:
            case StructureStrategy.NUMBERED:
                if self._pending_chapter and not _NUMBERED.match(text):
                    text = f"{self._pending_chapter} {text}"
                self._pending_chapter = None
                m = _NUMBERED.match(text)
                if _TOP_LEVEL.match(text):
                    self._numbered = [((), display_case(text))]
                    self._levels = []
                elif m:
                    number = tuple(int(n) for n in m.group(1).split("."))
                    title = f"{m.group(1)} {display_case(m.group(2))}"
                    # Keep only true ancestors (numeric prefixes): "3.1.2" must not
                    # nest under "3.1.1", and "4.1" must leave chapter 3.
                    ancestors = [
                        (num, t)
                        for num, t in self._numbered
                        if num and len(num) < len(number) and number[: len(num)] == num
                    ]
                    self._numbered = [*ancestors, (number, title)]
                    self._levels = []
                else:
                    self._levels = [display_case(text)]
            case StructureStrategy.MARKDOWN:
                depth = min(max(md_level, 1), 3)
                self._levels = [*self._levels[: depth - 1], display_case(text)]
            case StructureStrategy.CAPS_TOPICS:
                if _is_all_caps(text):
                    self._base, self._levels = [display_case(text)], []
                else:
                    self._levels = [display_case(text)]
            case StructureStrategy.TOC | StructureStrategy.RUNNING:
                self._levels = [display_case(text)]


def structure_units(
    pages: list[PageExtraction],
    strategy: StructureStrategy,
    section_aliases: dict[str, str] | None = None,
) -> list[Unit]:
    """Clean blocks and assign each content block its section path.

    ``section_aliases`` maps (case-insensitively) a running-header text to its
    canonical section name, to repair headers the PDF itself renders inconsistently.
    """
    tracker = SectionTracker(strategy)
    aliases = {k.casefold(): v for k, v in (section_aliases or {}).items()}
    running = {
        page: aliases.get(title.casefold(), title) if title else None
        for page, title in running_titles(pages).items()
    }
    boilerplate = repeated_boilerplate(pages)
    units: list[Unit] = []
    for page in pages:
        tracker.start_page(page, running.get(page.page_number))
        for block in page.blocks:
            match block.kind:
                case BlockKind.HEADING:
                    text = clean_heading(block.text)
                    # Check the cleaned text: raw "**Title**" would look like a "*" bullet.
                    if starts_with_bullet(text):
                        # e.g. "# ■ shock": a list item the layout model styled as a heading
                        units.append(
                            Unit(
                                BlockKind.LIST_ITEM,
                                clean_list_item(text),
                                page.page_number,
                                tracker.path,
                            )
                        )
                    elif m := _BARE_NUMBER.match(text):
                        tracker.chapter_number(m.group(1))
                    elif _is_heading(text):
                        tracker.heading(text, block.md_level)
                    elif text:
                        units.append(
                            Unit(
                                BlockKind.TEXT,
                                clean_paragraph(text),
                                page.page_number,
                                tracker.path,
                            )
                        )
                case BlockKind.TEXT:
                    if (text := clean_paragraph(block.text)) and text not in boilerplate:
                        units.append(Unit(BlockKind.TEXT, text, page.page_number, tracker.path))
                case BlockKind.LIST_ITEM:
                    if clean_paragraph(block.text):
                        units.append(
                            Unit(
                                BlockKind.LIST_ITEM,
                                clean_list_item(block.text),
                                page.page_number,
                                tracker.path,
                            )
                        )
                case BlockKind.TABLE:
                    if text := clean_table(block.text):
                        units.append(Unit(BlockKind.TABLE, text, page.page_number, tracker.path))
                case BlockKind.PAGE_HEADER | BlockKind.PAGE_FOOTER:
                    pass  # used only as section hints (running_titles)
    return units
