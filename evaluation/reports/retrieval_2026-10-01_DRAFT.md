# Retrieval evaluation: clinexa-retrieval-eval v1-draft

> ⚠️ **DRAFT**: the question set has not yet been reviewed by a human. Treat every number below as preliminary; do not quote it.

Generated 2026-10-01 20:29 UTC · dataset `9e95b5c6ee695585` · 57 questions (42 answerable) · 4201 indexed chunks

Embedding model `BAAI/bge-small-en-v1.5` · rerankers `cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-base`, `ncbi/MedCPT-Cross-Encoder` · candidate pool: dense top 15 + BM25 top 15 → RRF k=60 → top 20

## How to read this

* **Hit@k**: an answer-bearing chunk is in the top k ("was the answer retrievable?").
* **EvRecall@k** (evidence recall): share of the question's gold evidence items found in the top k.
* **MRR**: mean of 1/rank of the first relevant chunk. **NDCG@k**: rank-aware, binary relevance.
* Gold evidence is defined by criteria (document + pages/section + key terms), not chunk ids. Metrics average over answerable questions only.

## Systems compared (all answerable questions, no metadata filters)

| System | n | Hit@5 | Hit@10 | EvRecall@5 | EvRecall@10 | MRR | NDCG@5 | NDCG@10 | latency p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| `vector` | 42 | 0.714 | 0.881 | 0.714 | 0.881 | 0.595 | 0.422 | 0.449 | 29 / 46 |
| `bm25` | 42 | 0.476 | 0.500 | 0.476 | 0.500 | 0.402 | 0.237 | 0.239 | 0 / 0 |
| `hybrid` | 42 | 0.690 | 0.762 | 0.690 | 0.762 | 0.564 | 0.381 | 0.402 | 30 / 31 |
| `vector+rerank[ms-marco-MiniLM-L-6-v2]` | 42 | 0.810 | 0.881 | 0.810 | 0.881 | 0.606 | 0.442 | 0.477 | 110 / 131 |
| `hybrid+rerank[ms-marco-MiniLM-L-6-v2]` | 42 | 0.810 | 0.857 | 0.810 | 0.857 | 0.578 | 0.419 | 0.445 | 106 / 122 |
| `vector+rerank[bge-reranker-base]` | 42 | 0.833 | 0.905 | 0.833 | 0.905 | 0.628 | 0.489 | 0.500 | 520 / 580 |
| `hybrid+rerank[bge-reranker-base]` | 42 | 0.810 | 0.857 | 0.810 | 0.857 | 0.579 | 0.439 | 0.448 | 515 / 563 |
| `vector+rerank[MedCPT-Cross-Encoder]` | 42 | 0.786 | 0.881 | 0.786 | 0.881 | 0.595 | 0.431 | 0.461 | 442 / 532 |
| `hybrid+rerank[MedCPT-Cross-Encoder]` | 42 | 0.643 | 0.833 | 0.643 | 0.833 | 0.535 | 0.388 | 0.423 | 428 / 486 |

## Reranker improvement over plain hybrid (paired, 95% bootstrap CI)

| Reranker | ΔHit@5 | ΔHit@10 | ΔMRR | ΔNDCG@10 |
|---|---|---|---|---|
| `hybrid+rerank[ms-marco-MiniLM-L-6-v2]` | +0.119 [+0.024, +0.214] | +0.095 [+0.024, +0.190] | +0.015 [-0.074, +0.099] | +0.043 [-0.010, +0.096] |
| `hybrid+rerank[bge-reranker-base]` | +0.119 [+0.024, +0.214] | +0.095 [+0.024, +0.190] | +0.015 [-0.097, +0.126] | +0.046 [-0.024, +0.109] |
| `hybrid+rerank[MedCPT-Cross-Encoder]` | -0.048 [-0.167, +0.071] | +0.071 [+0.000, +0.167] | -0.028 [-0.120, +0.067] | +0.021 [-0.035, +0.081] |

An interval that excludes 0 means the difference is unlikely to be sampling noise.

## Does BM25 help? (hybrid pool vs dense-only pool, same reranker; paired 95% CI)

| Reranker | ΔHit@5 | ΔHit@10 | ΔMRR | ΔNDCG@10 |
|---|---|---|---|---|
| `ms-marco-MiniLM-L-6-v2` | +0.000 [-0.071, +0.071] | -0.024 [-0.071, +0.000] | -0.028 [-0.088, +0.023] | -0.032 [-0.062, -0.004] |
| `bge-reranker-base` | -0.024 [-0.119, +0.048] | -0.048 [-0.119, +0.000] | -0.049 [-0.116, +0.005] | -0.053 [-0.092, -0.019] |
| `MedCPT-Cross-Encoder` | -0.143 [-0.262, -0.048] | -0.048 [-0.119, +0.000] | -0.059 [-0.129, +0.009] | -0.038 [-0.069, -0.006] |

Positive = adding BM25 to the candidate pool improved the reranked result. An interval spanning 0 means BM25's contribution is not distinguishable from noise on this question set.

## Confidence intervals (95% bootstrap over questions)

| System | Hit@5 | MRR |
|---|---|---|
| `vector` | 0.714 [0.571, 0.833] | 0.595 [0.470, 0.716] |
| `bm25` | 0.476 [0.333, 0.619] | 0.402 [0.263, 0.536] |
| `hybrid` | 0.690 [0.548, 0.833] | 0.564 [0.443, 0.694] |
| `vector+rerank[ms-marco-MiniLM-L-6-v2]` | 0.810 [0.690, 0.929] | 0.606 [0.489, 0.714] |
| `hybrid+rerank[ms-marco-MiniLM-L-6-v2]` | 0.810 [0.690, 0.929] | 0.578 [0.462, 0.691] |
| `vector+rerank[bge-reranker-base]` | 0.833 [0.714, 0.929] | 0.628 [0.515, 0.741] |
| `hybrid+rerank[bge-reranker-base]` | 0.810 [0.690, 0.929] | 0.579 [0.461, 0.695] |
| `vector+rerank[MedCPT-Cross-Encoder]` | 0.786 [0.667, 0.905] | 0.595 [0.469, 0.713] |
| `hybrid+rerank[MedCPT-Cross-Encoder]` | 0.643 [0.500, 0.786] | 0.535 [0.411, 0.654] |

## By question category (Hit@5 / MRR)

| Category | n | `vector` | `bm25` | `hybrid` | `hybrid+rerank[ms-marco-MiniLM-L-6-v2]` |
|---|---|---|---|---|---|
| symptom | 11 | 0.82 / 0.56 | 0.36 / 0.21 | 0.73 / 0.52 | 0.82 / 0.65 |
| condition | 10 | 0.60 / 0.64 | 0.60 / 0.48 | 0.80 / 0.69 | 0.90 / 0.62 |
| red_flag | 8 | 0.38 / 0.37 | 0.25 / 0.26 | 0.25 / 0.29 | 0.50 / 0.38 |
| medication | 8 | 0.88 / 0.65 | 0.38 / 0.33 | 0.75 / 0.52 | 0.88 / 0.45 |
| follow_up | 5 | 1.00 / 0.87 | 1.00 / 1.00 | 1.00 / 0.90 | 1.00 / 0.87 |

## Effect of metadata filters (the 30 questions with caller context)

| System | n | Hit@5 | Hit@10 | EvRecall@5 | EvRecall@10 | MRR | NDCG@5 | NDCG@10 | latency p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| `hybrid+rerank[ms-marco-MiniLM-L-6-v2] (no filter)` | 30 | 0.800 | 0.833 | 0.800 | 0.833 | 0.545 | 0.371 | 0.404 | 105 / 118 |
| `hybrid+rerank[ms-marco-MiniLM-L-6-v2]+pop` | 30 | 0.833 | 0.867 | 0.833 | 0.867 | 0.643 | 0.475 | 0.493 | 121 / 133 |
| `hybrid+rerank[ms-marco-MiniLM-L-6-v2]+pop+topic` | 30 | 0.900 | 0.900 | 0.900 | 0.900 | 0.692 | 0.500 | 0.523 | 135 / 201 |
| `hybrid+rerank[bge-reranker-base]+pop` | 30 | 0.833 | 0.867 | 0.833 | 0.867 | 0.649 | 0.508 | 0.497 | 526 / 578 |
| `hybrid+rerank[bge-reranker-base]+pop+topic` | 30 | 0.833 | 0.900 | 0.833 | 0.900 | 0.671 | 0.538 | 0.532 | 596 / 681 |
| `hybrid+rerank[MedCPT-Cross-Encoder]+pop` | 30 | 0.700 | 0.833 | 0.700 | 0.833 | 0.623 | 0.449 | 0.483 | 440 / 492 |
| `hybrid+rerank[MedCPT-Cross-Encoder]+pop+topic` | 30 | 0.733 | 0.867 | 0.733 | 0.867 | 0.627 | 0.473 | 0.507 | 447 / 492 |

Population is a safety constraint (child vs adult dosing); topic is only a relevance hint and relies on rule-based topic labels, so it is the first thing to drop if it hurts.

## Can retrieval tell when it has nothing relevant?

Top-1 score as a detector of unanswerable questions (42 answerable, 10 out-of-domain, 5 ambiguous). AUROC: 0.5 = no signal, 1.0 = perfect.

| Score | AUROC answerable vs out-of-domain | AUROC answerable vs (out-of-domain + ambiguous) |
|---|---|---|
| reranker score (`hybrid+rerank[ms-marco-MiniLM-L-6-v2]`) | 0.976 | 0.962 |
| dense cosine (`vector`) | 0.986 | 0.978 |

Final abstention accuracy is measured end-to-end once the agents exist (Phase 12); this only shows whether the retrieval score is a usable ingredient.

## Misses: answerable questions with no relevant chunk in the top 5 (`hybrid+rerank[ms-marco-MiniLM-L-6-v2]`)

* **sym-02-baby-fever-rash** (symptom): "My baby has a high temperature and some spots on her skin.": first relevant at rank not in top 20
* **sym-10-adult-burning-urine** (symptom): "It burns when I pee and I need to go all the time.": first relevant at rank 11
* **con-09-catheter-infection** (condition): "How can bloodstream infections from central venous catheters be prevented?": first relevant at rank not in top 20
* **red-02-newborn-danger-signs** (red_flag): "My newborn is floppy and won't feed.": first relevant at rank not in top 20
* **red-05-suicidal-thoughts** (red_flag): "I've been thinking about ending my life.": first relevant at rank 9
* **red-06-child-seizure** (red_flag): "My child is having a fit and it's not stopping.": first relevant at rank not in top 20
* **red-08-malaria-drowsy-fitting** (red_flag): "Someone with malaria has become very drowsy and had a convulsion.": first relevant at rank 9
* **med-08-paracetamol-child** (medication): "How much paracetamol can I give my child for a fever?": first relevant at rank not in top 20
