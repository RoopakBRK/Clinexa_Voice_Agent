# Evaluations

Every evaluation in this repository: what it asks, how to run it, where its results are
kept, and what state it is in.

**Where the numbers come from.** Each number below is copied from the report or document
named beside it, with the date it was measured. Nothing here was measured for this page
except the two lines marked *10 October*. The reports are the source: if one is run again
and differs from this page, the report is right.

## At a glance

| # | Evaluation | What it asks | State |
|---|---|---|---|
| 1 | [Medicine lookup on names said wrongly](#1-medicine-lookup-on-names-said-wrongly) | Is the right medicine still found when its name is misheard? | Measured 9 October 2026. Seeded, so it can be run again |
| 2 | [WHO guidance retrieval](#2-who-guidance-retrieval) | Does the search return the passage that answers the question? | **Draft.** The question set has not been reviewed by a person. Do not quote |
| 3 | [Question-set check](#3-question-set-check) | Does every gold criterion match real chunks? | 0 errors, 0 warnings (10 October 2026) |
| 4 | [Medicines encoders compared](#4-medicines-encoders-compared) | Do a bi-encoder and a cross-encoder find more medicines? | Measured 8 October 2026 by scripts that are not in the repository. Cannot be reproduced |
| 5 | [Latency benchmarks](#5-latency-benchmarks) | How long do ingestion and each retrieval stage take? | Measured 8 and 9 October 2026 on one laptop. Stage timings came from scripts that are not in the repository |
| 6 | [Ingestion report](#6-ingestion-report) | What did chunking produce, document by document? | Rewritten by every `make ingest` |
| 7 | [Latency of each call](#7-latency-of-each-call) | How long did each stage of this call take? | Recorded on every call |
| 8 | [Tests of the evaluation code](#8-tests-of-the-evaluation-code) | Are the metrics and scoring rules themselves right? | 27 tests, run in CI |
| 9 | [Not evaluated yet](#9-not-evaluated-yet) | Safety, faithfulness, real speech, a whole call | Not built |

| # | Code | Data | Results |
|---|---|---|---|
| 1 | `apps/api/app/medicines/evaluation.py` | Made by rule from the catalogue, seed 20261009 | `evaluation/reports/medicines_lookup_2026-10-09.md`, and every query in `.jsonl` beside it |
| 2 | `apps/api/app/rag/evaluation/` | `evaluation/datasets/retrieval_eval_v1.yaml` | `evaluation/reports/retrieval_2026-10-01_DRAFT.md`, and every run in `.json` beside it |
| 3 | `apps/api/app/rag/evaluation/dataset.py` | The same question set, against `data/processed/chunks.jsonl` | Printed |
| 4 | Not in the repository | Synthetic, 4,000 lookups | `docs/rag.md`, sections 3.3 and 3.7 |
| 5 | Not in the repository (stage timings) | 57 questions, 5 passes each; 4,000 lookups | `docs/rag.md`, sections 2.4, 2.5, 3.5, 3.6 and 4 |
| 6 | `apps/api/app/rag/ingestion/` | `data/*.pdf` | `data/processed/ingestion_report.md`, `data/processed/samples.md` |
| 7 | `apps/api/app/observability/metrics.py` | Live calls | `GET /api/calls/{call_sid}` |
| 8 | `apps/api/tests/test_evaluation.py`, `test_medicines_evaluation.py` | Fakes | `make test` |

## 1. Medicine lookup on names said wrongly

```bash
make medicines-eval                                   # 1,000 names of each kind, seed 20261009
make medicines-eval ARGS="--per-kind 200 --seed 7"    # a smaller run, other names
```

It needs `QDRANT_URL` and the whole catalogue indexed (`make medicines`). It stops if the
collection holds a different number of names from the files.

There are no recordings of real callers, so the queries are made here: names drawn from
the catalogue and changed by rule. Four kinds, each from a different medicine:

| Kind | What is changed | Example |
|---|---|---|
| Said correctly | Nothing: the brand and its number, as a person says it | `glimihex m 2` |
| One letter out | One letter added, dropped or changed, anywhere | `piracetamol 500` |
| Respelt by ear | Two or three changes of the kind a listener makes | `roksimet 150` |
| Split or joined | One word heard as two, or two heard as one | `brax clav 500` |

Two things are scored for each query, against the one medicine it was made from:

- **Recall@5, @10 and @64**: where the medicine comes among the 64 names the search
  returns, in the search's own order, before any rule is applied.
- **Outcome**: what the lookup's rules then make of those 64. *Named* (its name was
  given), *offered* (it is among the six choices at most), *as said* (another product is
  named exactly what was meant), *beyond listed* (the brand was recognised, this product
  is past the six), *not found*, or *wrong* (another medicine was named or offered and
  this one was not). *Resolved* is named plus offered.

**Results, 9 October 2026.** 4,000 queries against the 252,553 names in Qdrant Cloud, by
spelling and sound, without the encoders. Run twice that day with the same figures.

| Kind | Queries | Recall@5 | Recall@10 | Recall@64 | Resolved | 95% interval | Not found | Wrong |
|---|---|---|---|---|---|---|---|---|
| Said correctly | 1,000 | 100.0% | 100.0% | 100.0% | 99.3% | 98.6% to 99.7% | 0.0% | 0.0% |
| One letter out | 1,000 | 87.2% | 91.3% | 97.4% | 82.3% | 79.8% to 84.5% | 8.5% | 8.5% |
| Respelt by ear | 1,000 | 49.1% | 55.7% | 71.1% | 29.9% | 27.1% to 32.8% | 45.0% | 24.7% |
| Split or joined | 1,000 | 98.4% | 99.2% | 99.7% | 87.2% | 85.0% to 89.1% | 1.2% | 10.5% |
| The three misheard kinds | 3,000 | 78.2% | 82.1% | 89.4% | 66.5% | 64.8% to 68.1% | 18.2% | 14.6% |
| **All** | **4,000** | **83.7%** | **86.6%** | **92.0%** | **74.7%** | 73.3% to 76.0% | 13.7% | 10.9% |

The report also cuts the same results three more ways:

| Cut | Finding |
|---|---|
| The first letter | One letter out with the first letter kept: 92.9% resolved (884 queries). With it changed: 1.7% (116 queries). Respelt by ear: 36.5% kept (709), 13.7% changed (291) |
| How far the name was changed | Respelt by ear, two letters different: 39.3% resolved. Three: 18.6%. Four or more: 4.0% |
| How a wrong answer reaches the caller | Of 437 wrong answers, 5 (0.1% of all queries) were stated as that medicine. The other 432 were put to the caller as a guess to confirm or as choices to pick from |

**What it does not show.** These are not recordings of speech. They are catalogue names
changed by rule, which stands in for mishearing and is not a sample of it. The figure a
real caller would get needs recordings to measure.

## 2. WHO guidance retrieval

```bash
cd apps/api
uv run python -m app.rag.evaluation retrieval --local \
    --rerankers cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-base
```

It writes `evaluation/reports/retrieval_<date>_DRAFT.md` and `.json`. The report stays
marked DRAFT until the question set is reviewed and `--reviewed` is passed.

> **Draft.** The question set was drafted by Claude and has not been reviewed by a person
> (`evaluation/datasets/REVIEW_NOTES.md`). Every number in this section is preliminary.
> Do not quote it.

**The question set**, `retrieval_eval_v1.yaml`: 57 questions.

| Category | Questions | Expected |
|---|---|---|
| Symptom | 11 | answer |
| Condition | 10 | answer |
| Red flag | 8 | answer |
| Medication | 8 | answer |
| Follow-up | 5 | answer, from a hand-written standalone query |
| Ambiguous | 5 | clarify: ask a question first |
| Out of domain | 10 | abstain: the knowledge base does not cover it |

42 are answerable, and 30 carry what the system would know about the caller (age,
pregnancy), which drives the metadata filters. Gold evidence is a criterion (document,
pages or section, key terms), not a chunk id, so it survives re-chunking.

**Systems compared**, each returning up to 20 chunks: dense only (`vector`), BM25 only,
hybrid (dense and BM25 fused by rank), dense with a cross-encoder, hybrid with a
cross-encoder, and hybrid with a cross-encoder and metadata filters (population, then
population and topic). Three cross-encoders, so 15 systems in all.

**Metrics**: Hit@k, evidence recall@k, MRR and NDCG@5 and @10, averaged over answerable
questions; latency p50 and p95; 95% bootstrap intervals over questions (2,000 resamples);
paired differences between systems on the same questions; and AUROC of the top score as a
signal that a question cannot be answered.

**Results, 1 October 2026 (draft).** 42 answerable questions, 4,201 indexed chunks, no
metadata filters:

| System | Hit@5 | Hit@10 | MRR | NDCG@10 | Latency p50 / p95 |
|---|---|---|---|---|---|
| `vector` | 0.714 | 0.881 | 0.595 | 0.449 | 29 / 46 ms |
| `bm25` | 0.476 | 0.500 | 0.402 | 0.239 | 0 / 0 ms |
| `hybrid` | 0.690 | 0.762 | 0.564 | 0.402 | 30 / 31 ms |
| `vector+rerank[ms-marco-MiniLM-L-6-v2]` | 0.810 | 0.881 | 0.606 | 0.477 | 110 / 131 ms |
| `hybrid+rerank[ms-marco-MiniLM-L-6-v2]` (the default) | 0.810 | 0.857 | 0.578 | 0.445 | 106 / 122 ms |
| `vector+rerank[bge-reranker-base]` | 0.833 | 0.905 | 0.628 | 0.500 | 520 / 580 ms |
| `hybrid+rerank[bge-reranker-base]` | 0.810 | 0.857 | 0.579 | 0.448 | 515 / 563 ms |
| `vector+rerank[MedCPT-Cross-Encoder]` | 0.786 | 0.881 | 0.595 | 0.461 | 442 / 532 ms |
| `hybrid+rerank[MedCPT-Cross-Encoder]` | 0.643 | 0.833 | 0.535 | 0.423 | 428 / 486 ms |

The report's other tables, each one an evaluation of its own:

| Table | Finding (draft) |
|---|---|
| Cross-encoder against plain hybrid, paired | MiniLM and `bge-reranker-base` each raised Hit@5 by 0.119, interval +0.024 to +0.214. MedCPT did not: -0.048, interval -0.167 to +0.071 |
| Does BM25 help? Hybrid pool against dense-only pool, same cross-encoder | No measurable gain on this set. With MiniLM, Hit@5 changed by 0.000 (interval -0.071 to +0.071) and NDCG@10 by -0.032 (-0.062 to -0.004) |
| By category, default system, Hit@5 | Follow-up 1.00, condition 0.90, medication 0.88, symptom 0.82, red flag 0.50 |
| Metadata filters, the 30 questions with caller context, default system | Hit@5 0.800 with no filter, 0.833 with population, 0.900 with population and topic |
| Can retrieval tell when it has nothing relevant? | AUROC answerable against out of domain: 0.976 by the cross-encoder's score, 0.986 by dense cosine |
| Misses | 8 answerable questions had no relevant chunk in the top 5. Four are red-flag questions |

**What the review notes say** (`evaluation/datasets/REVIEW_NOTES.md`):

- Three of the eight misses may be gold that is too strict, not retrieval that failed:
  `con-09-catheter-infection`, `red-08-malaria-drowsy-fitting`, `med-08-paracetamol-child`.
- Five are real misses: `red-02-newborn-danger-signs`, `red-06-child-seizure`,
  `sym-10-adult-burning-urine`, `sym-02-baby-fever-rash`, `red-05-suicidal-thoughts`.
- With 42 answerable questions the intervals are wide, about 0.1 either side on Hit@5.
- The wording is mostly conversational, which favours dense search over BM25.
- The filter results are optimistic: the questions with a caller age were written to have
  gold in the matching population.
- Do not tune retrieval on this set and then report on it. Hold out a test split first.

## 3. Question-set check

```bash
cd apps/api && uv run python -m app.rag.evaluation validate
```

Checks every gold criterion in the question set against the real corpus. A criterion that
matches no chunk can never be satisfied (an error). One that matches more than 40 chunks
is too vague to be evidence (a warning). No models and no network are needed.

*10 October 2026:* 57 questions, **0 errors, 0 warnings**.

A criterion can match chunks and still be the wrong evidence. This check does not replace
the review in section 2.

## 4. Medicines encoders compared

Run on 8 October 2026 by one-off scripts that are not in the repository, so how the names
were changed cannot be checked and the figures cannot be reproduced. Evaluation 1 is the
test that can be run again, and it gives lower figures. These still show what they were
run for: that the encoders change almost nothing. Full tables: `docs/rag.md`, sections
3.3 and 3.7.

**Bi-encoders.** The share of lookups whose medicine is among the 64 nearest of 252,553,
1,000 names of each kind:

| Retrieval | Said correctly | One letter out | Respelt by ear | Split or joined |
|---|---|---|---|---|
| Spelling and sound (the sparse vector) | 1.000 | 0.998 | 0.967 | 0.998 |
| `BAAI/bge-small-en-v1.5` | 1.000 | 0.870 | 0.580 | 0.758 |
| The same, with BGE's query instruction | 1.000 | 0.881 | 0.594 | 0.817 |
| `abhinand/MedEmbed-small-v0.1` | 1.000 | 0.862 | 0.542 | 0.752 |
| `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | 0.796 | 0.495 | 0.206 | 0.378 |
| Sparse and bge-small fused by rank (RRF) | 1.000 | 0.998 | 0.943 | 1.000 |
| Sparse 64, then bge-small's 16 added | 1.000 | 0.998 | 0.973 | 1.000 |

**Cross-encoders.** The share where the right medicine comes first, and where it is in
the first six, 250 lookups of each kind:

| Ordered by | Said correctly | One letter out | Respelt by ear | Split or joined | Time, p50 |
|---|---|---|---|---|---|
| The search's own order | 0.892 / 1.000 | 0.836 / 0.980 | 0.556 / 0.776 | 0.920 / 0.992 | none |
| `ms-marco-MiniLM-L-6-v2`, reading the name | 0.880 / 1.000 | 0.548 / 0.760 | 0.288 / 0.504 | 0.424 / 0.696 | 37 ms |
| `ms-marco-MiniLM-L-6-v2`, reading the name and composition | 0.912 / 1.000 | 0.552 / 0.764 | 0.264 / 0.504 | 0.436 / 0.684 | 46 ms |
| `ncbi/MedCPT-Cross-Encoder` | 0.756 / 0.968 | 0.364 / 0.672 | 0.164 / 0.412 | 0.268 / 0.564 | 115 ms |
| `BAAI/bge-reranker-base` | 0.596 / 0.956 | 0.588 / 0.932 | 0.584 / 0.876 | 0.592 / 0.916 | 127 ms |

**The three set-ups, through the whole lookup.** Lookups where the right medicine was
named or offered, of 4,000:

| Set-up | Said correctly | One letter out | Respelt by ear | Split or joined | All 4,000 |
|---|---|---|---|---|---|
| Spelling and sound only | 992 | 974 | 925 | 957 | 3,848 |
| With the bi-encoder | 992 | 974 | 927 | 957 | 3,850 |
| With the bi-encoder and the cross-encoder | 992 | 974 | 927 | 957 | 3,850 |

This is why `MEDICINES_ENCODERS` is off by default.

## 5. Latency benchmarks

All from one laptop (Apple M3 Pro), 8 October 2026 unless marked. The stage-by-stage
timings came from one-off scripts that call the same functions as the `make` targets and
are not in the repository. Full tables: `docs/rag.md`, sections 2.4, 2.5, 3.5, 3.6 and 4.

| | Ingestion | Retrieval, p50 | Retrieval, p95 |
|---|---|---|---|
| Medicines, both encoders, GPU | 2 min 54 s | 57.2 ms | 73.9 ms |
| Medicines, both encoders, CPU | not measured | 102.6 ms | 167.7 ms |
| Medicines, spelling and sound only | 23.5 s | 3.8 ms | 8.3 ms |
| WHO, GPU, Qdrant server | 4 min 17 s cold, 42 s cached | 90.5 ms | 107.7 ms |
| WHO, CPU, Qdrant server | not measured | 250.9 ms | 288.5 ms |
| WHO, GPU, embedded index | as above | 104.1 ms | 114.3 ms |
| WHO, hybrid only, no cross-encoder | as above | 11.5 ms | 16.3 ms |
| Medicines, spelling and sound only, Qdrant Cloud (9 October, through the tool) | 171 s to upload | 191 ms | 224 ms |
| WHO, GPU, Qdrant Cloud (9 October, through the tool) | 82 s to embed and upload | 291 ms | 301 ms |

The WHO figures are over the 57 questions of the question set, 5 passes each (285 runs).
The medicines figures are over 4,000 lookups on GPU and 1,000 on CPU. The cross-encoder
is the cost in both: 88% of a WHO search on GPU and 97% on CPU.

The commands that print timings themselves:

```bash
make ingest                                   # counts and time
make index ARGS="--local"                     # chunks per second
make query Q="child with fast breathing and cough" ARGS="--local"   # stage timings
make medicines ARGS="--recreate"              # time to read and index
make medicine NAME="glycomate 500"            # one lookup, with its time
```

## 6. Ingestion report

```bash
make ingest
```

Writes `data/processed/ingestion_report.md` (chunks, sections, token p50, p95 and max, and
label counts for each document) and `data/processed/samples.md` (random chunks from each
document, to read by eye). As committed: 4,794 chunks, of which 4,201 are retrievable,
from 6 documents, with a 350-token target.

Chunking is deterministic. A fresh run on 8 October 2026, including a full re-extraction
of the PDFs, produced the same 4,794 chunk ids as the `chunks.jsonl` in the repository
(`docs/rag.md`, section 2.1).

## 7. Latency of each call

Every call records its own timings, returned by `GET /api/calls/{call_sid}` as p50, p95
and p99, and printed by `make simulate`:

| Metric | From, to |
|---|---|
| `stt_finalization_ms` | Audio sent, to the final transcript received |
| `stt_endpoint_ms` | The caller's last word, to the end of the turn detected |
| `llm_ttft_ms`, `llm_first_sentence_ms`, `llm_total_ms` | Reply requested, to the first text from Claude, the first whole sentence and the full reply |
| `tts_ttfa_ms` | First sentence ready, to the first audio from the speech voice |
| `response_latency_ms` | End of the caller's turn detected, to the first reply audio sent |
| `voice_to_voice_ms` | The caller's last word, to the first reply audio sent |
| `lookup_medicine_ms`, `search_guidelines_ms` | One tool call, start to finish |

These are measurements of single calls, not an evaluation over a set. No figures from them
are reported anywhere yet.

## 8. Tests of the evaluation code

```bash
make test
```

| File | Tests | What they hold to |
|---|---|---|
| `apps/api/tests/test_evaluation.py` | 16 | Hit, MRR, NDCG and evidence recall against hand-computed values; AUROC and percentiles; the gold matching rules; that an answerable question needs gold and the others must have none; that the runner scores every system and averages over answerable questions only; that the bootstrap is deterministic; that a report on an unreviewed set says so |
| `apps/api/tests/test_medicines_evaluation.py` | 11 | That "one letter out" is exactly one letter and "respelt by ear" at least two; that the same seed makes the same queries, each from another medicine; how each outcome is decided; that the interval stays between 0 and 1 |

They run in CI with the rest of the suite (356 tests), with no keys and no network.

## 9. Not evaluated yet

- **WHO retrieval on a reviewed question set.** Evaluation 2 is a draft until a person has
  gone through `REVIEW_NOTES.md`.
- **Medicine names from real speech.** Every medicines number is from names changed by
  rule.
- **A whole call with the real model.** How often Claude calls each tool, what it says
  with the result, and the latency of a whole voice turn with a lookup in it. The tool
  loop is tested against a scripted model only. *10 October 2026:* one simulated call
  (`make simulate TEXT="What is in dolo six fifty?"`) was transcribed correctly and
  answered with the fallback line, because the Anthropic API refused the request: the
  account behind the key in `.env` has no credit.
- **Safety** (Phase 15): red-flag recall, abstention accuracy, prompt-injection and
  hallucination-attempt cases. `evaluation/README.md` plans `evaluation/safety/` and
  `evaluation/retrieval/` for these; neither folder exists yet.
- **Faithfulness** (Phase 12): whether replies say only what the passages say. Context
  compression stays unwired until this shows it is safe.
- **The query rewriter** (Phase 9). Follow-up questions are evaluated with a hand-written
  standalone query for now.
- **Anything on a deployed server, or under load.** The Qdrant Cloud figures are from one
  laptop to the cluster, one request at a time.
- **Embedding on CPU**, for the WHO chunks or the medicine names.
