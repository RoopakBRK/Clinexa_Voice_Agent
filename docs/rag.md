# RAG report: Clinexa Voice Agent

The retrieval pipelines behind Clinexa's two lookups: what they hold, how they search,
and what was measured.

**Where the numbers come from.** Unless a line says otherwise, every number was measured
on 8 October 2026 by running the code in `apps/api/app/rag` and `apps/api/app/medicines`,
in the repository those modules were developed in. They were moved here on 9 October
unchanged (the medicines modules only reworded). Lines marked *9 October* were measured
in this repository. The retrieval-quality table in section 2.7 is from the draft
evaluation of 1 October 2026 and is marked as such.

## 1. Summary

Clinexa has two knowledge stores, both kept in Qdrant, and they work differently.

| | Medicines catalogue | WHO knowledge base |
|---|---|---|
| What it holds | Indian medicine names: Jan Aushadhi, NLEM 2022, A to Z medicines of India | WHO primary-care guidance, 6 PDFs |
| Items stored | **252,553** names, one point each, with a sparse and a dense vector | **4,794** chunks, of which **4,201** are indexed |
| Chunking | None: one medicine name is one point | Section-aware, 350-token target, 50-token overlap |
| Sparse retrieval | Hashed words, numbers, letter groups and sound keys, IDF-weighted by Qdrant. The main search: 64 names | BM25, in memory |
| Bi-encoder | `BAAI/bge-small-en-v1.5`, 384 dimensions. A second leg: adds up to 16 names | `BAAI/bge-small-en-v1.5`, 384 dimensions |
| Fusion | None. Fusing the two legs by rank was measured and made results worse | Reciprocal rank fusion, k = 60 |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2`. Breaks ties only: rules in code decide which medicine it is | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 20 to top 5 |
| Ingestion, total | **2 min 54 s** with the bi-encoder, **23.5 s** without | **4 min 17 s** from the PDFs, **42 s** when extraction is cached |
| Retrieval, total (p50 / p95) | With both encoders **57 ms / 74 ms** on GPU, **103 ms / 168 ms** on CPU. Without them **4 ms / 8 ms** | **90 ms / 108 ms** on GPU, **251 ms / 288 ms** on CPU |
| Used on a call | Yes: Claude's `lookup_medicine` tool | Yes: Claude's `search_guidelines` tool |

**The medicines encoders are switched off** (`MEDICINES_ENCODERS=false`, the default).
The catalogue runs by spelling and sound alone: the "without" figures below are what it
does today. The bi-encoder and cross-encoder columns describe
what is built and measured, and what `MEDICINES_ENCODERS=true` switches back on.

Five things to know before reading the numbers:

1. **Both stores have to be indexed into the Qdrant in `.env` before a call can use
   them.** `make index` uploads the WHO chunks and `make medicines` the catalogue. Until
   then the server says so at start-up (`guidelines.unavailable`, `medicines.not_indexed`),
   `/health` reports it, and Claude is told a lookup could not be made.
   *9 October:* both were indexed into the Qdrant Cloud cluster named in `.env`:
   `clinexa_who_primary_care` holds 4,201 points and `clinexa_medicines` holds 252,553
   (spelling and sound only, no dense vectors).
2. **The two medicines encoders do not change which medicine is found, which is why they
   are off.** They are wired in and they run when switched on. But on 4,000 test lookups
   the right medicine was named or offered 3,848 times without them and 3,850 times with
   them, for 53 ms more on a GPU and 99 ms more on a CPU. Section 3.3 shows why: a
   misheard brand name has to be matched by its letters and its sound, and that is not
   what these models were trained to do.
3. **The WHO search needs its two models on the server.** The bi-encoder and the
   cross-encoder run locally through sentence-transformers (PyTorch). They load in the
   background at start-up; a search made before they are in answers "not available".
   The medicines lookup needs no model.
4. **Both pipelines are wired into the phone assistant as tools, and a whole call with
   the real model has not been run.** The tool loop is tested against a scripted model,
   and each tool was run against real data on 9 October (sections 2.5, 3.6 and 3.7). The
   Anthropic key in `.env` was rejected on 8 October. It was replaced on 9 October and
   is accepted, but the account had no credit, so the API refused every request.
5. **All timings are from one laptop** (Apple M3 Pro, 11 cores, 18 GB, macOS 26.6), with
   Qdrant 1.19.2 running on the same machine or embedded in the process, unless a line
   says Qdrant Cloud. Against a cloud cluster every search adds a network round trip.

## 2. WHO knowledge base

Code: `apps/api/app/rag/`. Data: `data/*.pdf`, `data/manifests/documents.yaml`,
`data/processed/chunks.jsonl`.

### 2.1 Sources and chunk counts

| Document | Pages (kept) | Structure strategy | Sections | Chunks | Indexed | Tokens p50 / p95 / max |
|---|---|---|---|---|---|---|
| WHO guidelines for malaria | 494 (492) | `toc` | 494 | 1,777 | 1,666 | 274 / 346 / 400 |
| mhGAP Intervention Guide v2.0 | 174 (169) | `running` | 95 | 233 | 229 | 269 / 354 / 389 |
| Pocket book of hospital care for children | 438 (438) | `numbered` | 358 | 764 | 684 | 275 / 350 / 401 |
| Pocket book of primary health care for children and adolescents (WHO Europe) | 940 (940) | `numbered` | 509 | 1,285 | 1,110 | 273 / 355 / 405 |
| Bloodstream infections: central venous catheters | 152 (152) | `numbered` | 96 | 405 | 182 | 292 / 348 / 407 |
| Adult Primary Care 2016/17 (South Africa NDoH, not a WHO publication) | 118 (111) | `caps_topics` | 203 | 330 | 330 | 255 / 350 / 392 |
| **Total** | **2,316 (2,302)** | | **1,755** | **4,794** | **4,201** | 275 / 350 / 407 |

The 593 chunks that are not indexed are reference lists, contents pages, front matter and
fragments under 8 tokens. They stay in `chunks.jsonl`, flagged `retrievable: false`.

The 4,201 indexed chunks:

| Measure | Value |
|---|---|
| Total tokens | 1,045,595 |
| Tokens per chunk | mean 249, median 269, p95 350, max 407 |
| Tokens per chunk as embedded (with its context header) | mean 289, p95 397, max 484. None reaches the model's 512-token limit |
| Chunks under 60 tokens / over the 350 target | 93 / 155 |
| Chunks that span more than one page | 826 |
| By type | text 2,767 · table 991 · warning 321 · recommendation 122 |
| By population | all 1,933 · child 1,862 · adult 330 · pregnancy 76 |

Tokens are counted with the embedding model's own tokenizer. Chunking is deterministic: a
fresh run on 8 October, including a full re-extraction of the PDFs, produced the same
4,794 chunk ids as the `chunks.jsonl` in the repository.

### 2.2 Chunking strategy

Structure-aware chunking: a chunk never crosses a section boundary, and its size is a
token budget, not a character count. In order:

1. **Extract typed blocks.** PyMuPDF's layout model (`pymupdf4llm`) labels every region of
   a page as heading, text, list item, table, page header or page footer. No OCR: every
   PDF has a text layer. Results are cached per document, keyed by the PDF's SHA-256.
2. **Clean.** Ligatures, split words, hyphenation, citation markers, symbol-font bullets,
   and running headers that repeat on more than 30% of pages are repaired or removed.
3. **Recover the section path.** Each PDF numbers its headings differently, so each has a
   strategy in the manifest: PDF bookmarks (`toc`), numbered headings (`numbered`), the
   running page header (`running`), or ALL-CAPS topic headings (`caps_topics`). Every
   block ends up with a path such as `Fever > Child under 5 > Treatment`.
4. **Merge small sibling sections.** Runs of neighbouring sections under 150 tokens are
   joined, up to the target, with their sub-headings kept inline. Pocket books have a
   heading every few lines; one chunk per heading would be too small to retrieve.
5. **Pack each section to 350 tokens.**
   - A paragraph is split only when it exceeds the budget, and only at sentence ends.
   - List items stay together, and stay with the sentence that introduces them.
   - A table is kept whole. An oversized table is split by rows with the header repeated.
   - Consecutive prose chunks share an overlap of up to 50 tokens of whole sentences.
   - A tail under 60 tokens is merged into the previous chunk, up to 135% of the target.
6. **Label by rule, not by model.** Topic, population, chunk type (text, table,
   recommendation, warning) and the `retrievable` flag come from keyword rules, so
   ingestion is deterministic and the labels can be audited.
7. **Add a context header for embedding.** What is embedded is
   `Document title > Section > Subsection`, a new line, then the chunk, so that a short
   passage carries where it came from.

Settings: `CHUNK_TARGET_TOKENS=350`, `CHUNK_MIN_TOKENS=60`, `CHUNK_OVERLAP_TOKENS=50`,
`CHUNK_SECTION_MERGE_TOKENS=150` (`app/core/config.py`).

### 2.3 Encoders and retrieval stages

A bi-encoder turns the question and each passage into vectors separately, so passages can
be embedded once in advance and searched quickly. A cross-encoder reads the question and
one passage together, which is more accurate and far slower, so it only sees a short list.

| Stage | What is used | Details |
|---|---|---|
| Bi-encoder (dense) | `BAAI/bge-small-en-v1.5` | 33.4 M parameters, 384 dimensions, cosine similarity on normalised vectors, 512-token limit, batch 32. Questions get BGE's retrieval instruction as a prefix; passages do not. Runs locally through sentence-transformers: no text leaves the machine |
| Vector store | Qdrant, collection `clinexa_who_primary_care` | 384-d cosine. Seven payload indexes for filters (population, topics, document type, document id, chunk type, retrievable, page). At 4,201 points Qdrant had built no HNSW graph, so the search is an exact scan |
| Sparse | BM25 (`bm25s`) | k1 = 1.2, b = 0.75, English stop words, Snowball stemmer. Indexes the section path plus the chunk text. Built in memory at start-up from `chunks.jsonl` |
| Candidates | Dense top 15 and BM25 top 15, in parallel | Both use the same metadata filter. If a strict filter leaves fewer than 5, every filter but population is dropped |
| Fusion | Reciprocal rank fusion, k = 60 | Fuses by rank, so cosine and BM25 scores need no calibration. Keeps the top 20 |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22.7 M parameters, 512-token limit, batch 16. Reads the section path plus the chunk. Re-orders the 20 and keeps the top 5 |
| Compression | Extractive, by the bi-encoder | Built and tested, **not wired in**. Cuts each kept chunk to its most relevant sentences within a budget (200 by default), always keeping sentences with a caution or referral cue |

Two other cross-encoders were evaluated and are not the default: `BAAI/bge-reranker-base`
(278 M parameters) and `ncbi/MedCPT-Cross-Encoder` (109 M parameters).

No language model is part of retrieval. Clinexa's replies come from Claude
(`claude-opus-5-5`), which asks for a search with its `search_guidelines` tool and is
given the five passages, each under its document, section and page
(`app/tools/knowledge.py`). The search is held to the caller's population: the age and
pregnancy Claude passes become the `population` filter, which is never relaxed.

### 2.4 Ingestion latency

| Step | Time | Notes |
|---|---|---|
| Extract 2,316 pages with the layout model | 215.0 s | 8 worker processes, about 10.8 pages a second. Skipped when cached: reading the cache takes 0.09 s |
| Load the tokenizer | 0.5 s | |
| Recover section structure | 1.3 s | |
| Chunk and label | 7.7 s | 4,794 chunks |
| Embed 4,201 chunks | 29.7 s | 142 chunks a second on the laptop's GPU (MPS) |
| Write to Qdrant, with payload indexes | 2.8 s | local Qdrant server |
| Build the BM25 index | 0.28 s | done at every start-up, not stored, not in the totals |
| **Total, from the PDFs** | **257 s (4 min 17 s)** | |
| **Total, extraction cached** | **42 s** | what `make ingest` then `make index` cost on a normal day |
| Re-index with nothing changed | 0.2 s | chunks are hashed; only changed ones are embedded again |

Loading the embedding model adds about 10 s once per process. Extraction is 84% of a cold
run, which is why it is cached.

### 2.5 Retrieval latency

Timed over the 57 questions of the evaluation set, 5 passes each (285 runs), after a
warm-up. Questions average 13 tokens. The cross-encoder reads 20 passages averaging 277
tokens. Times are milliseconds, p50 / p95.

| Stage | Qdrant server, GPU | Qdrant server, CPU | Embedded index, GPU |
|---|---|---|---|
| 1. Embed the question (bi-encoder) | 7.9 / 10.4 | 6.3 / 7.0 | 7.3 / 8.2 |
| 2. Dense search in Qdrant, top 15 | 2.6 / 4.7 | 2.4 / 2.6 | 18.6 / 20.7 |
| 3. BM25, top 15 | 0.3 / 0.8 | 0.2 / 0.3 | 0.3 / 0.4 |
| 4. Fuse with RRF, keep 20 | 0.2 / 0.2 | 0.2 / 0.2 | 0.2 / 0.2 |
| 5. Cross-encoder, 20 to 5 | 79.5 / 92.8 | 242.3 / 281.7 | 78.0 / 89.3 |
| **Total, question in to 5 passages out** | **90.5 / 107.7** | **250.9 / 288.5** | **104.1 / 114.3** |
| Total, with a population filter | 89.5 / 103.8 | 246.1 / 285.4 | 118.4 / 128.7 |
| Hybrid only, no cross-encoder | 11.5 / 16.3 | 9.8 / 10.5 | 26.7 / 29.2 |
| Dense only | 10.8 / 14.1 | 9.0 / 9.9 | 25.8 / 28.2 |
| Compression of the top 5 (not wired in) | 78.6 / 129.2 | 156.2 / 255.9 | 72.1 / 108.4 |

p99 for the total: 171 ms (server, GPU), 311 ms (server, CPU), 131 ms (embedded, GPU).
The slowest single run was 878 ms, once, on the server with GPU.

What the table shows:

- **The cross-encoder is the cost.** It is 88% of the total on GPU and 97% on CPU. The
  whole of hybrid retrieval before it takes about 11 ms.
- **BM25 and fusion are free** at this size: under a millisecond together.
- **The embedded index is seven times slower to search** than a Qdrant server (18.6 ms
  against 2.6 ms). It is the development default and should not be the production one.
- **"CPU" is this laptop's CPU.** A small shared cloud CPU will be considerably slower than the
  251 ms shown, and that has not been measured.

*9 October, in this repository:* the same search run through the `search_guidelines`
tool (`GuidelineSearch`, embedded index, GPU), five questions with an age or pregnancy
filter, five passes each: **115 ms p50, 121 ms p95, 123 ms max** over 25 searches. That
is the "Embedded index, GPU" column above, plus the tool's own work. Loading the index
and both models in the background took 21.5 s.

*9 October, against Qdrant Cloud from this laptop:* the same 25 searches through the tool
took **291 ms p50, 301 ms p95, 314 ms max**. The dense leg is one request to the cluster,
so the difference from the embedded index is again the network round trip.

The cross-encoder stage alone, for the three models compared (20 passages, p50 / p95):

| Cross-encoder | GPU | CPU |
|---|---|---|
| `ms-marco-MiniLM-L-6-v2` (default) | 79.5 / 92.8 ms | 242.3 / 281.7 ms |
| `ncbi/MedCPT-Cross-Encoder` | 392.8 / 432.5 ms | 1,207 / 1,343 ms |
| `BAAI/bge-reranker-base` | 479.6 / 518.3 ms | 1,436 / 1,639 ms |

Cold start, once per process: reading the chunks 70 ms, building BM25 250 ms, loading the
bi-encoder and answering a first question about 9 s (this includes importing PyTorch),
loading the cross-encoder and scoring a first pair about 9 s. A fresh process is therefore
about 18 s from its first answer.

### 2.6 Total latency in a conversation

Not measured end to end: no call has been run with the real model (point 4 above). What
the numbers above say: a search adds about 90 ms on a GPU and about 250 ms on a fast CPU.
The larger cost is not the search. A lookup is a tool call, so Claude is called a second
time with the passages, and that second round is what a caller waits for. While it runs
the caller hears a short spoken line ("Let me check the guidance on that"). Each search
is timed on the call as `search_guidelines_ms` (`GET /api/calls/{call_sid}`).

### 2.7 Retrieval quality (draft, do not quote)

From `evaluation/reports/retrieval_2026-10-01_DRAFT.md`: 57 questions, 42 of them
answerable, 4,201 indexed chunks. **The question set has not been reviewed by a person**,
so these are preliminary.

| System | Hit@5 | Hit@10 | MRR |
|---|---|---|---|
| Dense only (bi-encoder) | 0.714 | 0.881 | 0.595 |
| BM25 only | 0.476 | 0.500 | 0.402 |
| Hybrid (dense + BM25, RRF) | 0.690 | 0.762 | 0.564 |
| Hybrid + cross-encoder (MiniLM, the default) | 0.810 | 0.857 | 0.578 |
| Dense + cross-encoder (`bge-reranker-base`) | 0.833 | 0.905 | 0.628 |
| Hybrid + cross-encoder + population and topic filters (30 questions) | 0.900 | 0.900 | 0.692 |

Two findings from that report worth carrying forward once the set is reviewed. The
cross-encoder raised Hit@5 by 0.12 over plain hybrid, and the interval excluded zero. BM25
did not measurably help on this set: with the same cross-encoder, the hybrid pool was no
better than the dense-only pool. Four of the eight questions it missed were red-flag
questions phrased as a caller would say them ("My newborn is floppy and won't feed").

## 3. Medicines catalogue

Code: `apps/api/app/medicines/`. Data: three files in `data/`.

### 3.1 Sources and counts

| File | Size | Entries read | Names kept | Time to read |
|---|---|---|---|---|
| `nlem2022.pdf`, National List of Essential Medicines 2022 | 4.7 MB | 1,348 | 1,155 | 0.24 s |
| `jan_aushdi.pdf`, Jan Aushadhi (PMBJP) product list | 1.7 MB | 2,111 | 2,088 | 8.94 s |
| `A_Z_medicines_dataset_of_India.csv` | 32.1 MB | 253,973 | 249,310 | 2.12 s |
| **Total** | | **257,432** | **252,553** | **11.3 s** |

A name found in more than one place is kept once, from the first file in the order above;
4,879 entries were merged this way. The Jan Aushadhi PDF is the slow one because its
tables have to be detected page by page. An NLEM entry is a medicine, plus one entry for
each form and strength it is listed in.

Of the 252,553 names, 251,662 carry a strength, 239,785 carry one of the catalogue's units
(tablets, capsules, ml and so on), and 7,388 are marked discontinued.

### 3.2 What is stored for each name

There is no chunking. A medicine name averages 22 characters (p95 32), so one name is one
Qdrant point. Its payload is the medicine: name, source, composition, strength, unit, pack,
manufacturer, discontinued. Each point has a sparse vector, and a dense one too when the
catalogue is indexed with `MEDICINES_ENCODERS=true`.

**The sparse vector, `name`: spelling and sound.** No model makes it. It is made of:

| Feature | Example for "Glycomet 500 SR Tablet" |
|---|---|
| Whole words | `glycomet`, `sr`, `tablet` |
| Numbers | `500.0` |
| Three-letter groups of each word | `^gl`, `gly`, `lyc`, ... `et$` |
| A sound key for each word: first letter and consonants | `glkmt` |
| Three-letter groups and a sound key for the leading words run together | catches "eco sprin" and "pandy" |

Each feature is hashed with CRC32 to an index and given the value 1. Qdrant weighs each by
how rare it is across the catalogue (its IDF modifier), so "tablet" counts for little and
"glycomet" for a lot. A name has 20 features on average (p95 30), and the catalogue has
291,998 distinct features.

**The dense vector, `dense`: the bi-encoder's.** `BAAI/bge-small-en-v1.5` embeds the name
alone, into 384 numbers compared by cosine. The full vectors are kept on disk and a
one-byte copy of each number is searched in memory (scalar quantization), which brings a
quarter of a million vectors from 388 MB to 97 MB of memory. What was heard is embedded
with BGE's retrieval instruction in front; the catalogue's names are not.

### 3.3 Choosing the bi-encoder and the cross-encoder

Three bi-encoders and three cross-encoders were tried on the whole catalogue before any
was wired in. The test is synthetic: it takes names from the catalogue, changes them the
way a listener might, and asks whether the medicine each came from is found again. It is
not real speech. 1,000 names of each kind:

| Kind | From the catalogue | What is looked up |
|---|---|---|
| Said correctly: the brand and its number | Glimihex M 2mg/500mg Tablet | `glimihex m 2` |
| One letter out | Paracetamol Tablet 500 mg | `piracetamol 500` |
| Respelt by ear, two or three changes | Roximet 150mg Tablet | `roksimet 150` |
| Split or joined | Braxclav 500mg/125mg Tablet | `brax clav 500` |

**Bi-encoders.** The share of lookups whose medicine is among the 64 nearest of 252,553:

| Retrieval | Said correctly | One letter out | Respelt by ear | Split or joined |
|---|---|---|---|---|
| Spelling and sound (the sparse vector) | 1.000 | 0.998 | 0.967 | 0.998 |
| `BAAI/bge-small-en-v1.5`, general | 1.000 | 0.870 | 0.580 | 0.758 |
| The same, with BGE's query instruction | 1.000 | 0.881 | 0.594 | 0.817 |
| `abhinand/MedEmbed-small-v0.1`, medical | 1.000 | 0.862 | 0.542 | 0.752 |
| `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, biomedical names | 0.796 | 0.495 | 0.206 | 0.378 |
| Sparse and bge-small fused by rank (RRF) | 1.000 | 0.998 | 0.943 | 1.000 |
| **Sparse 64, then bge-small's 16 added (what is used)** | **1.000** | **0.998** | **0.973** | **1.000** |

| Bi-encoder | Parameters | Dimensions | Time to embed all names | Vectors |
|---|---|---|---|---|
| bge-small-en-v1.5 | 33.4 M | 384 | 57 s (4,434 names a second) | 388 MB |
| MedEmbed-small-v0.1 | 33.4 M | 384 | 55 s | 388 MB |
| SapBERT | 109.5 M | 768 | 183 s | 776 MB |

What this shows:

- **A bi-encoder finds a name said correctly and loses a misheard one.** It embeds
  meaning, and "roksimet" has none. Spelling and sound finds 97% of names respelt by ear;
  the best bi-encoder finds 59%.
- **The medical models are not better at this.** MedEmbed is level with the general model
  it was tuned from. SapBERT, which is trained to match biomedical terms, is far worse:
  Indian brand names are not terms it has seen.
- **Fusing the two legs by rank hurts.** RRF gives the weak leg the same say as the strong
  one. The right medicine was ranked first 55% of the time by spelling and sound on names
  respelt by ear, and 46% after fusion. So the legs are not fused: the 64 names found by
  spelling and sound come first, and the bi-encoder adds up to 16 it alone found.
- **The gain is small.** The second leg lifts the hardest kind from 0.967 to 0.973.

`BAAI/bge-small-en-v1.5` was chosen: the best of the three here, the smallest, and already
the WHO store's bi-encoder, so one model serves both.

**Cross-encoders.** Each re-ordered the same candidates (about 75 names) for 250 lookups
of each kind. The share where the right medicine comes first, and where it is in the
first six, which is how many Clinexa is given to offer:

| Ordered by | Said correctly | One letter out | Respelt by ear | Split or joined | Time, p50 |
|---|---|---|---|---|---|
| The search's own order, no cross-encoder | 0.892 / 1.000 | 0.836 / 0.980 | 0.556 / 0.776 | 0.920 / 0.992 | none |
| `ms-marco-MiniLM-L-6-v2`, reading the name | 0.880 / 1.000 | 0.548 / 0.760 | 0.288 / 0.504 | 0.424 / 0.696 | 37 ms |
| **`ms-marco-MiniLM-L-6-v2`, reading the name and composition** | **0.912 / 1.000** | 0.552 / 0.764 | 0.264 / 0.504 | 0.436 / 0.684 | 46 ms |
| `ncbi/MedCPT-Cross-Encoder`, medical | 0.756 / 0.968 | 0.364 / 0.672 | 0.164 / 0.412 | 0.268 / 0.564 | 115 ms |
| `BAAI/bge-reranker-base` | 0.596 / 0.956 | 0.588 / 0.932 | 0.584 / 0.876 | 0.592 / 0.916 | 127 ms |

What this shows:

- **No cross-encoder should be in charge of the order.** Each one ranks misheard names
  worse than the search already does, most of them much worse.
- **The medical cross-encoder is the worst of the three**, for the same reason as SapBERT.
- **`ms-marco-MiniLM-L-6-v2`, given the composition to read, is the one that helps
  anywhere at low cost:** on names said correctly it puts the right product first 91% of
  the time against 89%. That is the case of a brand with several products, where
  something has to decide which are listed first.
- **`bge-reranker-base` is the best on names respelt by ear** (in the first six 88% of the
  time against 78%), and worse everywhere else, at three times the cost and twelve times
  the size. It is not used. It would be the one to try if Clinexa were later given
  suggestions for names the rules cannot place.

So the cross-encoder is `cross-encoder/ms-marco-MiniLM-L-6-v2`, reading "name
(composition)", and its job is to break ties: among names the rules already hold equal,
its order decides which come first.

### 3.4 How a lookup runs

1. **Embed what was heard** with the bi-encoder. Skipped if the models are not loaded.
2. **Ask Qdrant twice in one request:** the 64 nearest names by spelling and sound, and
   the 16 nearest by the dense vector. The first list is kept in its order and the second
   adds whatever it alone found. About 76 names come back.
3. **The cross-encoder scores each of them** against what was heard, and they are put
   best first.
4. **Rules decide which medicine it is** (`app/medicines/lookup.py`): the name exactly as
   said (`exact`), a name with several products to choose between (`several`), a name it
   could have been misheard from (`close`), or nothing near (`unknown`). The order from
   step 3 only settles ties.
5. **Claude is told the result in words** (`app/tools/knowledge.py`): the catalogue's
   spelling, what the medicine contains, and what to ask the caller next.

The rules have the last word because telling a caller about the wrong medicine is worse
than saying it was not found. A strength, a form or a composition is given only when
every product the name could mean has the same one, never from a guessed name. A name is
only changed to one it could have been misheard from: one letter out, or the same
consonants.

Steps 1 and 3 are extras, and with `MEDICINES_ENCODERS=false` they are left out: the
lookup is then steps 2 and 4, by spelling and sound only. When they are on, the models
load in the background when the server starts
(18 s); lookups in that time skip them. If a model fails, or would take the lookup past
its 1.5 s limit, the lookup goes on without it.

### 3.5 Ingestion latency

| Step | With the bi-encoder | Without |
|---|---|---|
| Read the three files and merge duplicate names | 15 s | 11.3 s |
| Start up and load the bi-encoder | 13 s | none |
| Embed 252,553 names | 79 s (3,200 names a second, GPU) | none |
| Write 252,553 points to Qdrant | 67 s | 12.1 s (20,900 points a second) |
| **Total** | **174 s (2 min 54 s)** | **23.5 s** |

Both columns are one run of `make medicines` against a Qdrant server on the same laptop.
To the Qdrant Cloud cluster, on 9 October, the right-hand column took 12 s to read the
files and 171 s to write the 252,553 points.
With the encoders off, as now, it is the right-hand column, and no dense vectors are
stored.
Writing is five times slower with dense vectors because each point then carries a few
kilobytes more. To a cloud cluster the write will be slower again, by however long
roughly a gigabyte takes to upload. That was not measured.

Size, with dense vectors: 800 MB on disk, against 278 MB without. The Qdrant server's
resident memory was 143 MB after indexing and 105 MB after the test runs. That should fit
a free Qdrant Cloud cluster (1 GB of memory, 4 GB of disk) beside the WHO collection.

### 3.6 Lookup latency

The time for one lookup as Clinexa calls it, in milliseconds, p50 / p95:

| Set-up | GPU (4,000 lookups) | CPU (1,000 lookups) |
|---|---|---|
| Spelling and sound only | 3.8 / 8.3 | 3.2 / 6.7 |
| With the bi-encoder | 14.8 / 23.0 | 11.3 / 16.3 |
| **With the bi-encoder and the cross-encoder** | **57.2 / 73.9** | **102.6 / 167.7** |

The full set-up, one stage at a time (1,000 lookups, p50 / p95):

| Stage | GPU | CPU |
|---|---|---|
| 1. Embed what was heard (bi-encoder) | 7.7 / 9.8 | 6.5 / 8.9 |
| 2. Qdrant, both legs in one request | 3.1 / 4.6 | 3.2 / 5.3 |
| 3. Cross-encoder, 76 names on average | 43.0 / 54.9 | 86.1 / 136.8 |
| 4. Rules | 2.2 / 6.2 | 2.2 / 6.1 |

p99 for the whole lookup: 102 ms on GPU and 283 ms on CPU. The slowest single lookup was
683 ms. The limit is 1.5 s (`MEDICINES_LOOKUP_TIMEOUT_S`), after which Claude is told the
catalogue could not be reached, and not to guess.

The cross-encoder is three quarters of the time on GPU and more on CPU. "CPU" is this
laptop's CPU; a small shared cloud CPU would be slower, and was not measured.

Loading both models takes 15 s in a script and 18 s in the server, where it happens in the
background. A lookup made in that time took 22 ms.

By spelling and sound alone, with lookups sent together (measured earlier the same day):
10 at once took 23 ms (p50) and 32 ms (p95); 50 at once took 130 ms and 290 ms.

On a call a lookup is Claude's `lookup_medicine` tool call, timed as `lookup_medicine_ms`.
As with the WHO search, the second round with Claude that follows costs far more than
the lookup.

*9 October, against Qdrant Cloud from this laptop, in this repository:* 60 lookups through
the `lookup_medicine` tool, one at a time on a warm connection, took **191 ms p50, 224 ms
p95, 230 ms max**. Against the 3.8 ms of a server on the same machine, nearly all of that
is the network round trip to the cluster.

### 3.7 What a lookup returns

The same 4,000 synthetic lookups, through the whole lookup, for the three set-ups. The
count is of lookups where the right medicine was the one named, or was among the choices
offered for the caller to pick from:

| Set-up | Said correctly | One letter out | Respelt by ear | Split or joined | All 4,000 |
|---|---|---|---|---|---|
| Spelling and sound only | 992 | 974 | 925 | 957 | 3,848 |
| With the bi-encoder | 992 | 974 | 927 | 957 | 3,850 |
| With the bi-encoder and the cross-encoder | 992 | 974 | 927 | 957 | 3,850 |

The bi-encoder's second leg rescued two names of 4,000. The cross-encoder changed none of
these totals: it moved a few names between "named" and "offered" (2,970 named without
it, 2,968 with it), which is what breaking ties does.

The full set-up in detail:

| Outcome | Said correctly | One letter out | Respelt by ear | Split or joined |
|---|---|---|---|---|
| The right medicine named | 618 | 933 | 813 | 604 |
| The right medicine among the choices offered | 374 | 41 | 114 | 353 |
| A product named exactly as said (the sample came from a longer name) | 6 | 0 | 0 | 8 |
| Offered, but the right one beyond the six listed | 2 | 0 | 0 | 0 |
| Not found: reported as not in the catalogue | 0 | 6 | 26 | 5 |
| Another name given or offered, the right one not among them | 0 | 20 | 47 | 30 |

A name found from a misheard one (`close`) is always said back to the caller to confirm,
and nothing else is said about the medicine until they do. The last row is the one to
watch: 97 of 3,000 changed names ended on another medicine's name. For 33 of them the
changed spelling is itself a name in the catalogue. For the other 64 the rules found a
different name it could have been misheard from, and not the right one. With a quarter of
a million names that is common: in an earlier run, 29 of 100 made-up names landed near
something in the catalogue. It is why a guessed name never brings a strength or a
composition with it, and why the system prompt has Claude drop a guess the caller says
is wrong and not offer it again.

*9 October, in this repository:* thirteen names through the `lookup_medicine` tool, on the
real catalogue (24,602 of its names, indexed in memory). `dolo 650`, `crocin advance`,
`eco sprin 75`, `telma 40`, `thyronorm 50` and `metformin 500` came back `exact` with
their composition; `glycomet`, `azithral 500`, `augmentin 625` and `paracetamol` came back
`several`; `glycomate 500` came back `close` to "Glycomet 500"; `shelcal 500` and a made-up
name came back `unknown`. A spot check, not a measurement.

## 4. Totals at a glance

| | Ingestion | Retrieval, p50 | Retrieval, p95 |
|---|---|---|---|
| Medicines, both encoders, GPU | 2 min 54 s | 57.2 ms | 73.9 ms |
| Medicines, both encoders, CPU | not measured | 102.6 ms | 167.7 ms |
| Medicines, spelling and sound only | 23.5 s | 3.8 ms | 8.3 ms |
| WHO, GPU, Qdrant server | 4 min 17 s cold, 42 s cached | 90.5 ms | 107.7 ms |
| WHO, CPU, Qdrant server | not measured | 250.9 ms | 288.5 ms |
| WHO, GPU, embedded index | as above | 104.1 ms | 114.3 ms |
| WHO, hybrid only, no cross-encoder | as above | 11.5 ms | 16.3 ms |
| Medicines, spelling and sound only, **Qdrant Cloud** (9 October, through the tool) | 171 s to upload | 191 ms | 224 ms |
| WHO, GPU, **Qdrant Cloud** (9 October, through the tool) | 82 s to embed and upload | 291 ms | 301 ms |

## 5. What has not been measured

- Anything on a deployed server, or under load. The Qdrant Cloud figures are from one
  laptop to the cluster, one request at a time.
- The encoders on real speech-recognition output. Every medicines number is from names
  changed by rule, which is a stand-in for mishearing and not a sample of it.
- The latency of a whole voice turn with a lookup in it: speech recognition, Claude, the
  lookup, Claude again, speech.
- A call with the real model: on 9 October the Anthropic account behind the key in
  `.env` had no credit, so no request was answered. How often Claude calls each tool,
  what it says with the result, and whether the API accepts the request exactly as it is
  built, are untested outside a scripted model.
- WHO retrieval quality on a reviewed question set.
- Embedding the WHO chunks, or the medicine names, on CPU.

## 6. How to reproduce

```bash
make ingest                                   # chunk the PDFs; prints counts and time
uv run --project apps/api python -m app.rag.ingestion --force-extract   # also re-run extraction
make index ARGS="--local"                     # embed and index; prints chunks per second
make query Q="child with fast breathing and cough" ARGS="--local"       # prints stage timings
uv run --project apps/api python -m app.rag.evaluation retrieval --local # quality and latency per system
make medicines ARGS="--recreate"              # read and index the three files; prints the time
make medicine NAME="glycomate 500"            # one lookup, with its time
MEDICINES_ENCODERS=true make medicines ARGS="--recreate"     # the same, embedding each name too
MEDICINES_ENCODERS=true make medicine NAME="glycomate 500"   # one lookup with both encoders
```

The stage-by-stage timings and the encoder comparison came from one-off scripts that call
the same functions as those commands. They are not in the repository.
