# Medicine lookup on names said wrongly (2026-10-09)

4,000 queries made by rule from names in the catalogue, 1,000 of each kind, each from a different medicine. Looked up in `clinexa_medicines` (252,553 names) by spelling and sound, without the encoders. Seed 20261009.

Run it again with: `python -m app.medicines evaluate --per-kind 1000 --seed 20261009`

**These are not recordings of speech.** They are catalogue names changed by rule, which stands in for mishearing and is not a sample of it.

| Kind | Queries | Recall@5 | Recall@10 | Recall@64 | Resolved | 95% interval | Named | Offered | As said | Beyond listed | Not found | Wrong |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Said correctly | 1000 | 100.0% | 100.0% | 100.0% | 99.3% | 98.6% to 99.7% | 60.1% | 39.2% | 0.7% | 0.0% | 0.0% | 0.0% |
| One letter out | 1000 | 87.2% | 91.3% | 97.4% | 82.3% | 79.8% to 84.5% | 81.4% | 0.9% | 0.7% | 0.0% | 8.5% | 8.5% |
| Respelt by ear | 1000 | 49.1% | 55.7% | 71.1% | 29.9% | 27.1% to 32.8% | 28.6% | 1.3% | 0.4% | 0.0% | 45.0% | 24.7% |
| Split or joined | 1000 | 98.4% | 99.2% | 99.7% | 87.2% | 85.0% to 89.1% | 55.3% | 31.9% | 0.8% | 0.3% | 1.2% | 10.5% |
| **The three misheard kinds** | 3000 | 78.2% | 82.1% | 89.4% | 66.5% | 64.8% to 68.1% | 55.1% | 11.4% | 0.6% | 0.1% | 18.2% | 14.6% |
| **All** | 4000 | 83.7% | 86.6% | 92.0% | 74.7% | 73.3% to 76.0% | 56.4% | 18.3% | 0.7% | 0.1% | 13.7% | 10.9% |

- **Recall@k**: the medicine the query was made from is among the first k names the search returns, in the search's own order, before any rule is applied. The lookup reads the first 64.
- **Resolved**: named, or offered. The lookup's rules got the caller to it.
- **Named**: the lookup gave its name: the product, or the catalogue's spelling of what was meant, which the caller is asked to confirm.
- **Offered**: it is among the choices, six at most, the caller is asked to pick from.
- **As said**: another product is named exactly what was meant. Right by the rules, and not the one sampled.
- **Beyond listed**: the brand was recognised and this product is past the six listed.
- **Not found**: reported as not in the catalogue. The caller is asked to spell it.
- **Wrong**: another medicine was named or offered and this one was not. The one to watch.

## What a query of each kind looks like

| Kind | Catalogue name | Meant | Looked up | Outcome |
|---|---|---|---|---|
| Said correctly | Arpit 15mg Tablet | `arpit 15` | `arpit 15` | named |
| Said correctly | Hipress 50 Tablet | `hipress 50` | `hipress 50` | named |
| Said correctly | Safenix 200mg Tablet DT | `safenix 200` | `safenix 200` | offered |
| One letter out | Coptop-OF Tablet DT | `coptop dt` | `cotop dt` | named |
| One letter out | Upclock 0.5mg Tablet | `upclock 0.5` | `uplock 0.5` | named |
| One letter out | Allstate 125mg Injection | `allstate 125` | `allstte 125` | named |
| Respelt by ear | Esolin Injection | `esolin` | `esulim` | wrong |
| Respelt by ear | Etornext-P Tablet | `etornext p` | `aturnekst p` | not_found |
| Respelt by ear | Histagon L 5mg/10mg Tablet | `histagon l 5` | `heztagon l 5` | named |
| Split or joined | Tomrab L 75mg/20mg Tablet SR | `tomrab l 75` | `tomrabl 75` | wrong |
| Split or joined | Devbone D 70mg/5600IU Tablet | `devbone d 70` | `devboned 70` | wrong |
| Split or joined | ND Cad 25mg Injection | `nd cad 25` | `ndcad 25` | named |

## Where the misses come from

**The first letter.** The lookup never changes a name to one that starts with another sound, so a name whose first letter was changed is reported as not found, or taken for another medicine.

| | Queries | Recall@5 | Recall@10 | Recall@64 | Resolved | Not found | Wrong |
|---|---|---|---|---|---|---|---|
| One letter out, first letter kept | 884 | 86.4% | 90.4% | 97.1% | 92.9% | 1.1% | 5.2% |
| One letter out, first letter changed | 116 | 93.1% | 98.3% | 100.0% | 1.7% | 64.7% | 33.6% |
| Respelt by ear, first letter kept | 709 | 51.1% | 58.0% | 74.5% | 36.5% | 40.1% | 23.0% |
| Respelt by ear, first letter changed | 291 | 44.3% | 50.2% | 62.9% | 13.7% | 57.0% | 28.9% |

**How far the name was changed.** Names respelt by ear, by how many letters ended up different from what was meant.

| | Queries | Recall@5 | Recall@10 | Recall@64 | Resolved | Not found | Wrong |
|---|---|---|---|---|---|---|---|
| 2 letters | 580 | 57.4% | 64.0% | 78.3% | 39.3% | 35.2% | 25.2% |
| 3 letters | 370 | 37.3% | 43.8% | 61.4% | 18.6% | 55.7% | 25.7% |
| 4 or more letters | 50 | 40.0% | 48.0% | 60.0% | 4.0% | 80.0% | 12.0% |

**How a wrong answer reaches the caller.** Of 437 wrong answers, 5 were stated as that medicine (0.1% of all queries): the changed spelling was itself another product's name. The other 432 were put to the caller as a guess to confirm or as choices to pick from, with nothing else said about the medicine.

- `torcin 500` (meant `toracin 500`, from Toracin 500mg Injection): gave Torcin 500 Injection
- `ofil 400` (meant `ofpil 400`, from Ofpil 400mg Tablet ER): gave Ofil 400mg Tablet
- `rabia 20` (meant `rabica 20`, from Rabica 20mg Tablet): gave Rabia 20 Tablet
- `in cef 250` (meant `incef 250`, from Incef 250mg Tablet): gave Cef 250mg Tablet
- `tab rox 150` (meant `tabrox 150`, from Tabrox 150mg Tablet): gave Rox 150mg Tablet

## Misses, a few of each

**wrong** (437 in all)

- `orat` (meant `zorat`, from Zorat Syrup): gave Orat
- `cvlnay 5` (meant `cilnay 5`, from Cilnay 5 Tablet): gave Cvilin 5
- `ivivocet 50` (meant `vivocet 50`, from Vivocet 50 Tablet): gave Ivacet 50
- `adoxin plus` (meant `nadoxin plus`, from Nadoxin Plus Cream): gave Aloxin Plus
- `zacetroin 10` (meant `acetroin 10`, from Acetroin 10mg Tablet): gave Cetron 10
- `rertap 16` (meant `vertap 16`, from Vertap 16mg Tablet): gave Retop 16
- `metok 25` (meant `metzok 25`, from Metzok 25 Tablet PR): gave Meto 25
- `qpade 10` (meant `spade 10`, from Spade 10mg Tablet): gave Capad 10

**not_found** (547 in all)

- `zeuprolide acetate 11.25` (meant `leuprolide acetate 11.25`, from Leuprolide acetate Powder for injection 11.25 mg)
- `ivasag er 500` (meant `divasag er 500`, from Divasag ER 500 Tablet)
- `doxfit cv kid` (meant `moxfit cv kid`, from Moxfit CV Kid Tablet DT)
- `hcelast 100` (meant `acelast 100`, from Acelast 100mg Capsule)
- `zbetapro 80` (meant `betapro 80`, from Betapro 80mg Capsule SR)
- `veutin 8` (meant `vertin 8`, from Vertin 8mg Tablet)
- `yepresol 25` (meant `nepresol 25`, from Nepresol 25mg Tablet)
- `ldriblastina` (meant `adriblastina`, from Adriblastina Injection)

