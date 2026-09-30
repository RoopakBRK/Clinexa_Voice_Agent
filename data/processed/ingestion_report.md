# Ingestion report

Tokenizer: `BAAI/bge-small-en-v1.5` · target 350 tokens/chunk · 4821 chunks (4217 retrievable) · 10s

| Document | Pages | Sections | Chunks | Retrievable | Tokens p50 / p95 / max | Types |
|---|---|---|---|---|---|---|
| `who-malaria-guidelines` | 492 | 494 | 1777 | 1666 | 274 / 346 / 400 | recommendation 67, table 450, text 1252, warning 8 |
| `who-mhgap-ig-v2` | 169 | 95 | 233 | 229 | 269 / 354 / 389 | recommendation 4, table 40, text 176, warning 13 |
| `who-pocket-book-hospital-care-children` | 438 | 358 | 764 | 684 | 275 / 350 / 401 | table 211, text 497, warning 56 |
| `who-euro-pocket-book-phc-children` | 940 | 509 | 1285 | 1110 | 273 / 355 / 405 | table 287, text 823, warning 175 |
| `who-bsi-cvc-guidelines` | 152 | 96 | 405 | 182 | 292 / 348 / 407 | recommendation 55, table 40, text 310 |
| `sa-ndoh-adult-primary-care` | 117 | 225 | 357 | 346 | 247 / 350 / 392 | table 87, text 177, warning 93 |

## Topics and populations

- **who-malaria-guidelines** — topics {'malaria': 1445, 'immunization': 193, 'medication': 53, 'infectious_disease': 45, 'maternal_reproductive': 17}; populations {'all': 1667, 'pregnancy': 72, 'child': 38}
- **who-mhgap-ig-v2** — topics {'mental_health': 120, 'substance_use': 47, 'neurological': 29, 'medication': 16, 'maternal_reproductive': 8}; populations {'all': 199, 'pregnancy': 4, 'child': 30}
- **who-pocket-book-hospital-care-children** — topics {'infectious_disease': 168, 'child_health': 132, 'emergency': 80, 'respiratory': 77, 'medication': 69}; populations {'child': 764}
- **who-euro-pocket-book-phc-children** — topics {'child_health': 170, 'infectious_disease': 159, 'neonatal': 103, 'nutrition': 100, 'respiratory': 85}; populations {'child': 1285}
- **who-bsi-cvc-guidelines** — topics {'infection_prevention': 334, 'infectious_disease': 63, 'skin': 4, 'neonatal': 3, 'substance_use': 1}; populations {'all': 405}
- **sa-ndoh-adult-primary-care** — topics {'infectious_disease': 85, 'maternal_reproductive': 48, 'adult_primary_care': 38, 'cardiovascular': 27, 'mental_health': 26}; populations {'adult': 357}
