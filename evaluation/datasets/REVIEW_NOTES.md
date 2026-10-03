# Reviewing `retrieval_eval_v1.yaml`

The question set was drafted by Claude from the knowledge base's own headings and term
searches, not from retriever output. **No number from it should be quoted (resume, README)
until a human has reviewed it.** Every question and gold criterion needs your eyes; the notes
below only point at where I already see problems.

## What to check on every question
1. Is the wording what a real caller would say?
2. Is each `gold` unit a fact a good answer *needs*, and does the listed document/section
   really contain it? (A chunk satisfies a unit if it matches ALL of: an accepted document, a page
   range, ONE of the section substrings, every `must_contain` term, and at least one `any_of` term.)
3. Is `expected` right: `answer`, `clarify` (too vague; ask a question first) or `abstain` (not covered)?

`python -m app.rag.evaluation validate` confirms every criterion matches real chunks (0 errors,
0 warnings today), but a criterion can match chunks and still be the *wrong* evidence.

## Gold I suspect is too strict (retrieved evidence looked valid but was not counted)
Found by reading what the best system retrieved for the questions it "missed". Change only if you agree.

| Question | Issue | Suggested fix |
|---|---|---|
| `con-09-catheter-infection` | Retrieved the Executive Summary and its recommendations list, which answer the broad question. Gold demands the word *chlorhexidine*. | Accept the Executive Summary / Recommendations sections without the key term. |
| `red-08-malaria-drowsy-fitting` | Top results were the paediatric pocket book's *Severe malaria* sections (relevant, arguably better). Gold only accepts the WHO malaria guideline. | Add `who-pocket-book-hospital-care-children` section "Severe malaria" as an accepted document. |
| `med-08-paracetamol-child` | Gold requires the drug-dosage *annex* tables (which extract poorly). The fever-management sections that were retrieved also tell parents to give paracetamol, and may include the dose. | Decide whether fever-management text counts; if it states a dose, accept it. |

## Real retrieval misses (do NOT change the gold)
| Question | What happened |
|---|---|
| `red-02-newborn-danger-signs` | "floppy and won't feed" never reached the *Danger signs in newborns* sections. Safety-relevant. |
| `red-06-child-seizure` | Without an age filter, adult seizure guidance outranks the paediatric convulsion sections. |
| `sym-10-adult-burning-urine` | "burns" matched the *Burns* sections and the child *Dysuria* section; adult *Urinary Symptoms* ranked 11th. |
| `sym-02-baby-fever-rash` | "baby/spots" matched newborn skin conditions; the paediatric fever-with-rash sections were not in the top 20. |
| `red-05-suicidal-thoughts` | The mhGAP *Glossary* definition of "suicidal thoughts" outranked the actionable self-harm guidance. |

## Known limitations of this set (please address when extending it)
* Only 42 answerable questions: confidence intervals are wide (about ±0.1 on Hit@5).
* Mostly conversational wording. Exact-term questions (drug names, doses, codes) are a minority,
  which favours dense search over BM25. Add more of them before judging BM25.
* Population filters look better than they should: questions with a caller age were written to
  have gold in the matching population, and topic filters reuse the same keyword rules that label
  the chunks. Treat the filter results as optimistic.
* Follow-up questions are evaluated with a hand-written standalone `retrieval_query`; the query
  rewriter that must produce it is evaluated separately (Phase 9).
* Do not tune retrieval parameters on this set and then report on it. Hold out a test split first.
