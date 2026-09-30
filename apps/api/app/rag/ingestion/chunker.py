"""Structure-aware chunking.

Chunks never cross a section boundary. Within a section, content units are packed
up to a token budget measured with the embedding model's own tokenizer:

* paragraphs split at sentence boundaries only when they exceed the budget;
* list items stay together with the sentence introducing them ("They include:");
* tables are atomic, or split by rows with the header row repeated;
* consecutive text chunks share a short sentence overlap for context;
* a too-small tail is merged into the previous chunk of the same section.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import groupby

from app.rag.ingestion.classify import (
    classify_chunk_type,
    classify_population,
    classify_topics,
    is_retrievable,
)
from app.rag.ingestion.manifest import DocumentManifest
from app.rag.ingestion.structure import Unit
from app.rag.models import BlockKind, Chunk
from app.schemas.clinical import DocumentMetadata

TokenCounter = Callable[[str], int]

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'•-])")
_HARD_CAP_MARGIN = 1.35  # a merged tail may push a chunk up to 135% of target
_MIN_RETRIEVABLE_TOKENS = 8  # shorter fragments (cover titles, stray labels) are not indexed


@dataclass(slots=True)
class Piece:
    kind: BlockKind
    text: str
    page: int
    tokens: int


@dataclass(slots=True)
class _Draft:
    pieces: list[Piece] = field(default_factory=list)
    carried: int = 0  # leading pieces repeated from the previous chunk as overlap

    @property
    def tokens(self) -> int:
        return sum(p.tokens for p in self.pieces)

    @property
    def new_pieces(self) -> list[Piece]:
        return self.pieces[self.carried :]


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s.strip()]


def _split_by_words(text: str, budget: int, count: TokenCounter) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for word in text.split():
        if current and count(" ".join([*current, word])) > budget:
            parts.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts


def _split_oversized(unit: Unit, budget: int, count: TokenCounter) -> list[Piece]:
    """Split one unit into pieces that each fit the budget."""
    if count(unit.text) <= budget:
        return [Piece(unit.kind, unit.text, unit.page, count(unit.text))]

    if unit.kind is BlockKind.TABLE:
        lines = unit.text.splitlines()
        has_header = len(lines) > 1 and set(lines[1].replace("|", "").strip()) <= set("-: ")
        header = lines[:2] if has_header else lines[:1]
        rows = lines[len(header) :]
        segments, current = [], list(header)
        for row in rows:
            if count("\n".join([*header, row])) > budget:
                # One row alone is too big (evidence tables with paragraph cells):
                # flush, then split the row as prose, prefixed with its label cell.
                if len(current) > len(header):
                    segments.append("\n".join(current))
                    current = list(header)
                cells = [c.strip() for c in row.strip("|").split("|") if c.strip()]
                label, body = (
                    (cells[0], " ".join(cells[1:])) if len(cells) > 1 else ("", " ".join(cells))
                )
                prose = Unit(BlockKind.TEXT, body, unit.page, unit.path)
                label_budget = budget - count(label) - 1
                segments.extend(
                    f"{label}: {p.text}" if label else p.text
                    for p in _split_oversized(prose, label_budget, count)
                )
                continue
            if len(current) > len(header) and count("\n".join([*current, row])) > budget:
                segments.append("\n".join(current))
                current = list(header)
            current.append(row)
        if len(current) > len(header):
            segments.append("\n".join(current))
    else:
        separator = "\n" if unit.kind is BlockKind.LIST_ITEM else " "
        pieces_text = (
            unit.text.split("\n") if unit.kind is BlockKind.LIST_ITEM else _sentences(unit.text)
        )
        segments, current = [], []
        for sentence in pieces_text:
            if count(sentence) > budget:
                if current:
                    segments.append(separator.join(current))
                    current = []
                segments.extend(_split_by_words(sentence, budget, count))
                continue
            if current and count(separator.join([*current, sentence])) > budget:
                segments.append(separator.join(current))
                current = []
            current.append(sentence)
        if current:
            segments.append(separator.join(current))
    return [Piece(unit.kind, s, unit.page, count(s)) for s in segments if s.strip()]


def _merge_lists(units: list[Unit]) -> list[Unit]:
    """Join consecutive list items, and attach an introducing "...:" sentence."""
    merged: list[Unit] = []
    for unit in units:
        prev = merged[-1] if merged else None
        continues_list = (
            prev is not None
            and unit.kind is BlockKind.LIST_ITEM
            and (
                prev.kind is BlockKind.LIST_ITEM
                or (prev.kind is BlockKind.TEXT and prev.text.rstrip().endswith(":"))
            )
        )
        if prev is not None and continues_list:
            merged[-1] = Unit(
                BlockKind.LIST_ITEM, f"{prev.text}\n{unit.text}", prev.page, prev.path
            )
            continue
        merged.append(unit)
    return merged


def _overlap(draft: _Draft, budget: int, count: TokenCounter) -> list[Piece]:
    """Trailing sentences of the last prose piece, up to ``budget`` tokens."""
    if budget <= 0 or not draft.pieces or draft.pieces[-1].kind is not BlockKind.TEXT:
        return []
    last = draft.pieces[-1]
    tail: list[str] = []
    for sentence in reversed(_sentences(last.text)):
        if count(" ".join([sentence, *tail])) > budget:
            break
        tail.insert(0, sentence)
    if not tail or len(tail) == len(_sentences(last.text)):
        return []  # nothing to carry, or it would duplicate the whole piece
    text = " ".join(tail)
    return [Piece(BlockKind.TEXT, text, last.page, count(text))]


def _pack_section(
    units: list[Unit], count: TokenCounter, target: int, minimum: int, overlap: int
) -> list[_Draft]:
    pieces = [p for unit in _merge_lists(units) for p in _split_oversized(unit, target, count)]
    drafts: list[_Draft] = []
    current = _Draft()
    for piece in pieces:
        if current.new_pieces and current.tokens + piece.tokens > target:
            drafts.append(current)
            carry = _overlap(current, overlap, count) if piece.kind is not BlockKind.TABLE else []
            # Drop the overlap if it would push this piece over budget on its own.
            if sum(p.tokens for p in carry) + piece.tokens > target:
                carry = []
            current = _Draft(list(carry), carried=len(carry))
        current.pieces.append(piece)
    if current.new_pieces:
        drafts.append(current)

    # Merge an undersized tail into its predecessor when it still fits the hard cap.
    if len(drafts) >= 2 and sum(p.tokens for p in drafts[-1].new_pieces) < minimum:
        tail = drafts[-1].new_pieces
        if drafts[-2].tokens + sum(p.tokens for p in tail) <= target * _HARD_CAP_MARGIN:
            drafts[-2].pieces.extend(tail)
            drafts.pop()
    return drafts


def _common_prefix(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def _coalesce_sections(
    units: list[Unit], count: TokenCounter, target: int, small: int
) -> list[tuple[tuple[str, ...], list[Unit]]]:
    """Merge runs of small sibling/child sections into one section.

    Pocket books have a heading every few lines ("History", "What to expect by
    1 month"); one chunk per heading yields fragments too small to retrieve well.
    Merged sections keep their sub-headings inline as "Heading:" lines and take
    the common ancestor as their path.
    """
    groups = [(path, list(g)) for path, g in groupby(units, key=lambda u: u.path)]
    runs: list[list[tuple[tuple[str, ...], list[Unit], int]]] = []
    for path, group in groups:
        tokens = sum(count(u.text) for u in group)
        if runs:
            run = runs[-1]
            run_path = run[0][0]
            for p, _, _ in run[1:]:
                run_path = _common_prefix(run_path, p)
            shared = _common_prefix(run_path, path)
            run_tokens = sum(t for _, _, t in run)
            close_kin = len(shared) >= 1 and len(shared) >= min(len(run_path), len(path)) - 1
            if (
                close_kin
                and (run_tokens < small or tokens < small)
                and run_tokens + tokens <= target
            ):
                run.append((path, group, tokens))
                continue
        runs.append([(path, group, tokens)])

    sections: list[tuple[tuple[str, ...], list[Unit]]] = []
    for run in runs:
        if len(run) == 1:
            sections.append((run[0][0], run[0][1]))
            continue
        merged_path = run[0][0]
        for p, _, _ in run[1:]:
            merged_path = _common_prefix(merged_path, p)
        merged: list[Unit] = []
        for path, group, _ in run:
            if label := " > ".join(path[len(merged_path) :]).rstrip(":"):
                merged.append(Unit(BlockKind.TEXT, f"{label}:", group[0].page, merged_path))
            merged.extend(Unit(u.kind, u.text, u.page, merged_path) for u in group)
        sections.append((merged_path, merged))
    return sections


def _chunk_id(doc_id: str, page: int, path: tuple[str, ...], text: str) -> str:
    digest = hashlib.sha1(f"{doc_id}|{'|'.join(path)}|{text}".encode()).hexdigest()[:12]
    return f"{doc_id}:p{page:04d}:{digest}"


def chunk_document(
    doc: DocumentManifest,
    units: list[Unit],
    count: TokenCounter,
    *,
    target_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
    section_merge_tokens: int = 0,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen_ids: set[str] = set()
    sections = _coalesce_sections(units, count, target_tokens, small=section_merge_tokens)
    for path, group in sections:
        for draft in _pack_section(group, count, target_tokens, min_tokens, overlap_tokens):
            text = "\n\n".join(p.text for p in draft.pieces)
            pages = [p.page for p in draft.pieces]
            table_tokens = sum(p.tokens for p in draft.pieces if p.kind is BlockKind.TABLE)
            topics = classify_topics(path, text, doc.topics)
            token_count = count(text)
            chunk_id = _chunk_id(doc.doc_id, min(pages), path, text)
            if chunk_id in seen_ids:  # identical repeated content (e.g. repeated boxes)
                continue
            seen_ids.add(chunk_id)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    token_count=token_count,
                    metadata=DocumentMetadata(
                        source=doc.file,
                        document_id=doc.doc_id,
                        document_title=doc.title,
                        document_type=doc.document_type,
                        publisher=doc.publisher,
                        section=path[0] if path else None,
                        subsection=path[-1] if len(path) > 1 else None,
                        heading_path=list(path),
                        page_number=min(pages),
                        page_end=max(pages),
                        topic=topics[0] if topics else None,
                        topics=topics,
                        population=classify_population(doc.population, path),
                        publication_date=doc.publication_date,
                        source_url=doc.source_url,
                        chunk_type=classify_chunk_type(
                            path, text, table_tokens / max(draft.tokens, 1)
                        ),
                        retrievable=token_count >= _MIN_RETRIEVABLE_TOKENS
                        and is_retrievable(path, text),
                    ),
                )
            )
    return chunks
