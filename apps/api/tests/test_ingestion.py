"""Ingestion: cleaning, section recovery, chunking and labelling (no PDFs needed)."""

from __future__ import annotations

import pytest

from app.rag.ingestion.chunker import chunk_document
from app.rag.ingestion.classify import (
    classify_chunk_type,
    classify_population,
    classify_topics,
    is_retrievable,
)
from app.rag.ingestion.clean import (
    clean_heading,
    clean_list_item,
    clean_paragraph,
    clean_table,
    display_case,
)
from app.rag.ingestion.manifest import DocumentManifest, StructureStrategy
from app.rag.ingestion.structure import Unit, running_titles, structure_units
from app.rag.models import Block, BlockKind, PageExtraction
from app.schemas.clinical import ChunkType


def words(text: str) -> int:
    return len(text.split())


def page(
    n: int,
    *blocks: tuple[BlockKind, str] | tuple[BlockKind, str, int],
    toc: list[str] | None = None,
) -> PageExtraction:
    return PageExtraction(
        page_number=n,
        blocks=[Block(kind=b[0], text=b[1], md_level=b[2] if len(b) > 2 else 0) for b in blocks],  # type: ignore[misc]
        toc_path=toc or [],
    )


def manifest(
    structure: StructureStrategy = StructureStrategy.NUMBERED, population: str = "child"
) -> DocumentManifest:
    return DocumentManifest(
        doc_id="doc",
        file="doc.pdf",
        title="Test Pocket Book",
        publisher="WHO",
        document_type="pocket_book",
        population=population,
        topics=["child_health"],
        structure=structure,
    )


H, T, L, TB = BlockKind.HEADING, BlockKind.TEXT, BlockKind.LIST_ITEM, BlockKind.TABLE
HDR, FTR = BlockKind.PAGE_HEADER, BlockKind.PAGE_FOOTER


# --- cleaning -------------------------------------------------------------


def test_clean_paragraph_fixes_pdf_artefacts() -> None:
    raw = (
        "**Breathing** diffi culty and infl ating the bag; sub- stance use _[34]_ in ﬁeld_(1, 2)_."
    )
    assert (
        clean_paragraph(raw)
        == "Breathing difficulty and inflating the bag; substance use in field."
    )


def test_clean_list_item_normalises_bullets() -> None:
    assert clean_list_item("■ shock") == "- shock"
    assert clean_list_item("- **high** or low blood pressure") == "- high or low blood pressure"


def test_clean_table_keeps_rows_and_drops_breaks() -> None:
    table = "|**Drug**|Dose<br>per kg|\n|---|---|\n|ORS|75 ml|"
    assert clean_table(table) == "|Drug|Dose per kg|\n|---|---|\n|ORS|75 ml|"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("FACE SYMPTOMS", "Face Symptoms"),
        ("HIV AND TB", "HIV and TB"),
        ("COUGH AND/OR DIFFICULT BREATHING", "Cough and/or Difficult Breathing"),
        ("COMMON PRESENTATIONS OF DEPRESSION", "Common Presentations of Depression"),
        ("WHO GDG", "WHO GDG"),
        ("Acute diarrhoea", "Acute diarrhoea"),
    ],
)
def test_display_case(raw: str, expected: str) -> None:
    assert display_case(raw) == expected


# --- structure ------------------------------------------------------------


def test_numbered_headings_nest_by_number() -> None:
    pages = [
        page(
            1,
            (H, "# **1**", 1),
            (H, "## **Triage and emergency conditions**", 2),
            (T, "Chapter intro."),
        ),
        page(
            2,
            (H, "# 1.9 Envenoming"),
            (H, "## 1.9.2 Scorpion sting"),
            (H, "# **Diagnosis**"),
            (T, "Signs develop within minutes."),
            (H, "# ■ shock"),
            (L, "- fast pulse"),
        ),
        page(
            3,
            (H, "## 1.9.3 Other sources"),
            (T, "Fish stings."),
            (H, "# 2.1 Diagnostic approaches"),
            (T, "Take a history."),
        ),
    ]
    units = structure_units(pages, StructureStrategy.NUMBERED)
    by_text = {u.text: u.path for u in units}
    assert by_text["Chapter intro."] == ("1 Triage and emergency conditions",)
    assert by_text["Signs develop within minutes."] == (
        "1 Triage and emergency conditions",
        "1.9 Envenoming",
        "1.9.2 Scorpion sting",
        "Diagnosis",
    )
    # "# ■ shock" is a list item the layout model styled as a heading.
    assert by_text["- shock"][-1] == "Diagnosis"
    # A sibling replaces 1.9.2 rather than nesting under it.
    assert by_text["Fish stings."] == (
        "1 Triage and emergency conditions",
        "1.9 Envenoming",
        "1.9.3 Other sources",
    )
    # A new chapter number leaves chapter 1 entirely.
    assert by_text["Take a history."] == ("2.1 Diagnostic approaches",)


def test_skipped_level_does_not_nest_under_sibling() -> None:
    pages = [
        page(
            1,
            (H, "3 Recommendations"),
            (H, "3.1.1 Education"),
            (T, "a."),
            (H, "3.1.2 Hand hygiene"),
            (T, "b."),
        )
    ]
    by_text = {u.text: u.path for u in structure_units(pages, StructureStrategy.NUMBERED)}
    assert by_text["b."] == ("3 Recommendations", "3.1.2 Hand hygiene")


def test_back_matter_resets_to_top_level() -> None:
    pages = [
        page(
            1,
            (H, "7.4 Implementation"),
            (T, "a."),
            (H, "References"),
            (T, "1. Smith J."),
            (H, "Annex 2. Drug doses"),
            (T, "b."),
        )
    ]
    by_text = {u.text: u.path for u in structure_units(pages, StructureStrategy.NUMBERED)}
    assert by_text["1. Smith J."] == ("References",)
    assert by_text["b."] == ("Annex 2. Drug doses",)


def test_markdown_strategy_uses_heading_depth() -> None:
    pages = [
        page(
            1,
            (H, "# **FACE SYMPTOMS**", 1),
            (H, "## Recognise the patient needing urgent attention:", 2),
            (L, "- Sudden facial weakness: stroke likely"),
            (H, "## Approach to the patient", 2),
            (T, "Ask about pain."),
        )
    ]
    by_text = {u.text: u.path for u in structure_units(pages, StructureStrategy.MARKDOWN)}
    assert by_text["- Sudden facial weakness: stroke likely"] == (
        "Face Symptoms",
        "Recognise the patient needing urgent attention",
    )
    assert by_text["Ask about pain."] == ("Face Symptoms", "Approach to the patient")


def test_running_strategy_uses_running_footer_as_section() -> None:
    pages = [
        page(1, (T, "Assess mood."), (FTR, "**23**"), (FTR, "**DEPRESSION**")),
        page(
            2,
            (H, "**CLINICAL TIP**"),
            (T, "Ask about sleep."),
            (FTR, "**24**"),
            (FTR, "**DEPRESSION**"),
        ),
        page(3, (T, "Assess for psychosis."), (FTR, "**PSYCHOSES**")),
    ]
    by_text = {u.text: u.path for u in structure_units(pages, StructureStrategy.RUNNING)}
    assert by_text["Assess mood."] == ("Depression",)
    assert by_text["Ask about sleep."] == ("Depression", "Clinical Tip")
    assert by_text["Assess for psychosis."] == ("Psychoses",)


def test_running_title_prefers_module_name_over_codes_and_captions() -> None:
    pages = [
        page(28, (T, "a."), (HDR, "20"), (HDR, "DEP"), (HDR, "DEPRESSION")),
        page(30, (T, "b."), (HDR, "22"), (HDR, "DEP 1"), (HDR, "DEPRESSION Assessment")),
        page(32, (T, "c."), (FTR, "24"), (FTR, "DEP 1")),  # only a code: no section change
    ]
    by_text = {u.text: u.path for u in structure_units(pages, StructureStrategy.RUNNING)}
    assert by_text["a."] == by_text["b."] == ("Depression",)
    assert by_text["c."] == ("Depression",)  # keeps the current module


def test_running_title_keeps_short_words_and_applies_aliases() -> None:
    pages = [
        page(
            114,
            (T, "a."),
            (HDR, "**106**"),
            (HDR, "SUB"),
            (HDR, "**DISORDERS DUE TO SUBSTANCE USE**"),
        ),
        page(
            16, (T, "b."), (HDR, "8"), (HDR, "ECP"), (HDR, "ESSENTIAL CARE AND PRACTICE& PRACTI E")
        ),
    ]
    aliases = {"Essential Care and Practice& Practi E": "Essential Care and Practice"}
    by_text = {u.text: u.path for u in structure_units(pages, StructureStrategy.RUNNING, aliases)}
    assert by_text["a."] == ("Disorders Due to Substance Use",)
    assert by_text["b."] == ("Essential Care and Practice",)


def test_caps_topics_strategy_ignores_markdown_depth() -> None:
    pages = [
        page(13, (H, "# **CHRONIC RESPIRATORY DISEASE**", 1), (T, "Asthma text.")),
        # Symptom pages are level 2 but are topics, not children of the chapter above.
        page(
            14,
            (H, "## **FEVER**", 2),
            (H, "#### **Recognise the patient with fever**", 4),
            (L, "- confusion"),
        ),
        page(
            19, (H, "## **HEADACHE**", 2), (H, "#### **Management**", 4), (T, "Give paracetamol.")
        ),
    ]
    by_text = {u.text: u.path for u in structure_units(pages, StructureStrategy.CAPS_TOPICS)}
    assert by_text["Asthma text."] == ("Chronic Respiratory Disease",)
    assert by_text["- confusion"] == ("Fever", "Recognise the patient with fever")
    assert by_text["Give paracetamol."] == ("Headache", "Management")


def test_toc_strategy_nests_body_headings_under_bookmarks() -> None:
    pages = [
        page(
            40,
            (H, "# Repellents in general"),
            (T, "Repellents reduce bites."),
            toc=["4 Prevention", "4.1 Vector control"],
        )
    ]
    by_text = {u.text: u.path for u in structure_units(pages, StructureStrategy.TOC)}
    assert by_text["Repellents reduce bites."] == (
        "4 Prevention",
        "4.1 Vector control",
        "Repellents in general",
    )


def test_repeated_running_header_is_boilerplate() -> None:
    pages = [page(i, (HDR, "WHO guidelines for malaria"), (FTR, f"{i} of 9")) for i in range(1, 10)]
    assert set(running_titles(pages).values()) == {None}


# --- chunking -------------------------------------------------------------


def unit(
    text: str, kind: BlockKind = T, page_no: int = 1, path: tuple[str, ...] = ("1 Section",)
) -> Unit:
    return Unit(kind, text, page_no, path)


def chunk(
    units: list[Unit],
    target: int = 20,
    minimum: int = 4,
    overlap: int = 6,
    population: str = "child",
):  # type: ignore[no-untyped-def]
    return chunk_document(
        manifest(population=population),
        units,
        words,
        target_tokens=target,
        min_tokens=minimum,
        overlap_tokens=overlap,
    )


def test_chunks_never_cross_sections() -> None:
    chunks = chunk([unit("Alpha one two.", path=("1 A",)), unit("Beta one two.", path=("2 B",))])
    assert [c.metadata.heading_path for c in chunks] == [["1 A"], ["2 B"]]


def test_chunks_respect_budget_and_overlap() -> None:
    sentences = [f"Sentence number {i} has six words." for i in range(8)]  # 6 words each
    chunks = chunk(
        [unit(" ".join(sentences[:4])), unit(" ".join(sentences[4:]))], target=20, overlap=6
    )
    assert all(c.token_count <= 20 + 6 for c in chunks)
    # The second chunk starts with the last sentence of the first (overlap).
    assert chunks[1].text.startswith(chunks[0].text.split("\n\n")[-1].split(". ")[-1].rstrip("."))


def test_list_stays_with_its_introduction() -> None:
    chunks = chunk(
        [unit("Signs of envenoming include:"), unit("- shock", L), unit("- fast pulse", L)],
        target=50,
    )
    assert chunks[0].text == "Signs of envenoming include:\n- shock\n- fast pulse"


def test_oversized_table_is_split_with_repeated_header() -> None:
    rows = "\n".join(f"|drug {i}|{i} mg|" for i in range(12))
    table = f"|Drug|Dose|\n|---|---|\n{rows}"
    chunks = chunk([unit(table, TB)], target=12)
    assert len(chunks) > 1
    assert all(c.text.startswith("|Drug|Dose|\n|---|---|") for c in chunks)
    assert all(c.metadata.chunk_type is ChunkType.TABLE for c in chunks)


def test_small_tail_is_merged_into_previous_chunk() -> None:
    chunks = chunk(
        [
            unit(
                "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen."
            ),
            unit("Tail."),
        ],
        target=18,
        minimum=4,
        overlap=0,
    )
    assert len(chunks) == 1 and chunks[0].text.endswith("Tail.")


def test_small_sibling_sections_are_merged_with_inline_headings() -> None:
    parent = ("3 Well-child visits", "3.3 Visit: 1 week")
    units = [
        unit("Look for jaundice.", path=parent),
        unit("- Feeding difficulties", L, path=(*parent, "History")),
        unit("- Weight and length", L, path=(*parent, "Examination")),
        unit("A long unrelated section " + "word " * 30, path=("4 Nutrition",)),
    ]
    chunks = chunk_document(
        manifest(),
        units,
        words,
        target_tokens=40,
        min_tokens=4,
        overlap_tokens=0,
        section_merge_tokens=15,
    )
    assert chunks[0].metadata.heading_path == list(parent)
    assert (
        chunks[0].text
        == "Look for jaundice.\n\nHistory:\n- Feeding difficulties\n\nExamination:\n- Weight and length"
    )
    # Different top-level section: never merged.
    assert chunks[1].metadata.heading_path == ["4 Nutrition"]


def test_oversized_table_row_is_split_as_prose_with_its_label() -> None:
    row = "|Benefits and harms|" + " ".join(f"Finding {i} was observed." for i in range(10)) + "|"
    chunks = chunk([unit(f"|Domain|Detail|\n|---|---|\n{row}", TB)], target=15)
    assert len(chunks) > 1
    assert all(c.text.startswith("Benefits and harms: ") for c in chunks)
    assert all(c.token_count <= 15 for c in chunks)


def test_tiny_fragments_are_not_retrievable() -> None:
    assert chunk([unit("POCKET BOOK OF")])[0].metadata.retrievable is False


def test_inline_tags_and_split_ligatures_are_cleaned() -> None:
    assert clean_paragraph(
        "assess (<sup>SUI)</sup> within the fi rst 6 h; oxygen fl ow; m<sup>2</sup>"
    ) == ("assess (SUI) within the first 6 h; oxygen flow; m2")
    assert clean_paragraph("<mark>Dr</mark> A doctor must confirm") == "Dr A doctor must confirm"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("## 7 S evere acute malnutrition", "7 Severe acute malnutrition"),
        ("# 1 T riage", "1 Triage"),
        ("# 5 A child with fever", "5 A child with fever"),
        ("# **ECP**", "ECP"),
        (
            "## Recognise the patient needing urgent attention:",
            "Recognise the patient needing urgent attention",
        ),
    ],
)
def test_drop_cap_headings(raw: str, expected: str) -> None:
    assert clean_heading(raw) == expected


def test_repeated_page_title_is_dropped_from_body() -> None:
    title = "Guidelines for the prevention of bloodstream infections associated with catheters"
    pages = [
        page(i, (H, "3 Recs"), (T, f"Unique finding number {i} is described here."), (T, title))
        for i in range(1, 12)
    ]
    texts = [u.text for u in structure_units(pages, StructureStrategy.NUMBERED)]
    assert title not in texts and len(texts) == 11


def test_glyph_bullets_are_removed() -> None:
    assert clean_list_item("- u Stay calm. Call for help.") == "- Stay calm. Call for help."
    assert clean_paragraph("u Wash your hands") == "Wash your hands"


def test_chunk_metadata_and_stable_ids() -> None:
    units = [
        unit(
            "Give ORS 75 ml/kg over 4 hours for some dehydration.",
            path=("5 Diarrhoea", "5.2 Acute diarrhoea"),
            page_no=151,
        )
    ]
    first, second = chunk(units)[0], chunk(units)[0]
    assert first.chunk_id == second.chunk_id and first.chunk_id.startswith("doc:p0151:")
    m = first.metadata
    assert (m.section, m.subsection, m.page_number) == ("5 Diarrhoea", "5.2 Acute diarrhoea", 151)
    assert m.topic == "gastrointestinal"
    assert first.embedding_text().startswith(
        "Test Pocket Book > 5 Diarrhoea > 5.2 Acute diarrhoea\n"
    )


# --- classification -------------------------------------------------------


def test_chunk_type_rules() -> None:
    assert (
        classify_chunk_type(["Face symptoms needing urgent attention"], "stroke likely", 0)
        is ChunkType.WARNING
    )
    assert (
        classify_chunk_type(["3.2.1 Body wash"], "Conditional recommendation, low certainty", 0)
        is ChunkType.RECOMMENDATION
    )
    assert classify_chunk_type(["Doses"], "|a|b|", 0.9) is ChunkType.TABLE
    assert classify_chunk_type(["Background"], "Some text.", 0) is ChunkType.TEXT


def test_population_narrowing_for_mixed_documents() -> None:
    assert classify_population("child", ["Adults"]) == "child"  # document population wins
    assert classify_population("all", ["Treatment", "Children under 5"]) == "child"
    assert classify_population("all", ["Malaria in pregnancy"]) == "pregnancy"
    assert classify_population("all", ["Vector control"]) == "all"


def test_topic_keywords_match_whole_words_only() -> None:
    # "ear " used to match inside "near"/"clear"/"year", tagging intro text as ENT.
    assert classify_topics(
        ["Intro"], "In the near future, a clear year of ears and errors.", ["x"]
    ) == ["x"]
    assert classify_topics(["Ears"], "Check both ears for hearing loss.", ["x"])[0] == "eye_ent"


def test_topics_weight_headings() -> None:
    assert (
        classify_topics(["Cough or difficult breathing"], "Count the breaths.", ["x"])[0]
        == "respiratory"
    )
    assert classify_topics(["Misc"], "nothing relevant here", ["child_health"]) == ["child_health"]


@pytest.mark.parametrize(
    ("path", "text", "expected"),
    [
        (["References"], "1. Smith J.", False),
        (["Annex 2. PICO questions"], "Population: adults", False),
        (["Front"], "ISBN 978 92 4 154837 3. All rights reserved.", False),
        (["Contents"], "|Diarrhoea|275|", False),
        (["6.4 Diarrhoea"], "|a|275|\n|b|278|\n|c|280|\n|d|282|", False),  # mini table of contents
        (["6.4 Diarrhoea"], "Diarrhoea is the passage of 3 or more loose stools.", True),
        (["Annex 2. Drug doses"], "|Amoxicillin|40 mg/kg|", True),
    ],
)
def test_retrievability(path: list[str], text: str, expected: bool) -> None:
    assert is_retrievable(path, text) is expected
