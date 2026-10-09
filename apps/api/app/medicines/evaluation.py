"""How often the lookup finds the right medicine from a name that was said or heard wrongly.

There are no recordings of real callers to test with, so the queries are made here: names
drawn from the catalogue and changed by rule. Everything is seeded, so a run can be
repeated exactly:

    python -m app.medicines evaluate              # 1,000 names of each kind

Four kinds of query, each made from a different catalogue name:

    said       the brand and its number, as a person says it       "glimihex m 2"
    one_edit   one letter added, dropped or changed, anywhere      "piracetamol 500"
    by_ear     two or three changes of the kind a listener makes   "roksimet 150"
    split      one word heard as two, or two heard as one          "brax clav 500"

Two things are scored for each query, against the one medicine it was made from:

    rank       where the medicine comes among the 64 names the search returns, which
               gives recall@5, @10 and @64
    outcome    what the lookup's rules then make of those 64 (see ``outcome``)

This is a stand-in for mishearing, not a sample of it. The changes are ones speech
recognition is known to make, chosen without looking at what the lookup can undo: a third
of the ``by_ear`` changes swap consonants (b and p, d and t, m and n), which the lookup's
sound key does not treat as alike.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from typing import Literal, get_args

from pydantic import BaseModel

from app.medicines.catalog import Medicine
from app.medicines.lookup import Match, decide
from app.medicines.store import MedicineStore
from app.medicines.text import content_tokens, is_number

Kind = Literal["said", "one_edit", "by_ear", "split"]
KINDS: tuple[Kind, ...] = get_args(Kind)
KIND_NAMES: dict[Kind, str] = {
    "said": "Said correctly",
    "one_edit": "One letter out",
    "by_ear": "Respelt by ear",
    "split": "Split or joined",
}

# named:         the lookup gave this medicine's name: the product itself, or the
#                catalogue's spelling of what was meant, to be confirmed by the caller.
# offered:       the medicine is among the choices (six at most) the caller is asked to
#                pick from.
# as_said:       another product is named exactly what was meant ("Dolo 650" when the
#                sample came from "Dolo 650 SR"). Right by the rules, not the sampled one.
# beyond_listed: the brand was recognised, and this product is past the six listed.
# not_found:     reported as not in the catalogue.
# wrong:         another medicine was named or offered, and this one was not.
Outcome = Literal["named", "offered", "as_said", "beyond_listed", "not_found", "wrong"]
OUTCOMES: tuple[Outcome, ...] = get_args(Outcome)

_LETTERS = "abcdefghijklmnopqrstuvwxyz"
# A word shorter than this is left alone: "pan" with a letter changed is another name.
_MIN_WORD = 5
_LEADING_WORDS = 3

# How a listener respells what they hear, in three groups. One group is picked, then one
# change from it, so no group crowds the others out.
_BY_EAR: tuple[tuple[tuple[str, str], ...], ...] = (
    # The same sound, spelt another way.
    (("ph", "f"), ("f", "ph"), ("ck", "k"), ("c", "k"), ("k", "c"), ("x", "ks"), ("z", "s"),
     ("s", "z"), ("qu", "kw"), ("y", "i"), ("i", "y"), ("ee", "i"), ("oo", "u")),
    # Vowels taken for one another.
    (("a", "e"), ("e", "a"), ("e", "i"), ("i", "e"), ("o", "u"), ("u", "o"), ("a", "o"),
     ("o", "a")),
    # Consonants taken for one another.
    (("b", "p"), ("p", "b"), ("d", "t"), ("t", "d"), ("g", "k"), ("m", "n"), ("n", "m"),
     ("v", "w"), ("w", "v"), ("th", "t"), ("t", "th")),
)  # fmt: skip


class Query(BaseModel):
    kind: Kind
    medicine_id: str
    medicine: str  # the catalogue name it was made from
    meant: str  # what the person meant to say
    heard: str  # what is looked up


# How far down the search's list the medicine may come and still count, for recall@k.
CUTOFFS = (5, 10, 64)


class Result(BaseModel):
    query: Query
    # Where the medicine came in the search's list, from 1. None if it was not in it.
    rank: int | None
    status: str
    outcome: Outcome
    answer: str | None = None  # the name the lookup gave, for reading failures


def _edits(a: str, b: str) -> int:
    """Letters to add, drop or change to turn one word into the other."""
    before = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        row = [i]
        for j, y in enumerate(b, start=1):
            row.append(min(before[j] + 1, row[j - 1] + 1, before[j - 1] + (x != y)))
        before = row
    return before[-1]


def said(name: str) -> tuple[list[str], str | None] | None:
    """The words a person says for a catalogue name, and its first number if it has one.

    "Glimihex M 2mg/500mg Tablet" -> (["glimihex", "m"], "2"). None if it has no words.
    """
    words: list[str] = []
    for token in content_tokens(name):
        if is_number(token):
            return (words, token) if words else None
        if len(words) == _LEADING_WORDS:
            break
        words.append(token)
    return (words, None) if words else None


def _longest(words: Sequence[str]) -> int | None:
    """The word a mishearing lands on: the longest, if any is long enough."""
    at = max(range(len(words)), key=lambda i: len(words[i]))
    return at if len(words[at]) >= _MIN_WORD and words[at].isalpha() else None


def one_edit(word: str, rng: random.Random) -> str:
    """One letter added, dropped or changed, at any place in the word."""
    while True:
        how = rng.choice(("change", "drop", "add"))
        at = rng.randrange(len(word) + (how == "add"))
        letter = rng.choice(_LETTERS)
        changed = {
            "change": word[:at] + letter + word[at + 1 :],
            "drop": word[:at] + word[at + 1 :],
            "add": word[:at] + letter + word[at:],
        }[how]
        if changed != word:
            return changed


def by_ear(word: str, rng: random.Random) -> str | None:
    """Two or three changes of the kind a listener makes. None if the word allows none."""
    for _ in range(20):
        changed = word
        for _ in range(rng.choice((2, 3))):
            groups = [[(old, new) for old, new in group if old in changed] for group in _BY_EAR]
            usable = [group for group in groups if group]
            if not usable:
                break
            old, new = rng.choice(rng.choice(usable))
            places = [i for i in range(len(changed)) if changed.startswith(old, i)]
            at = rng.choice(places)
            changed = changed[:at] + new + changed[at + len(old) :]
        # Two changes that undo each other, or come to one letter, are another kind.
        if _edits(word, changed) >= 2:
            return changed
    return None


def split_or_joined(words: Sequence[str], rng: random.Random) -> list[str] | None:
    """One word heard as two ("brax clav"), or the first two heard as one ("pand")."""
    at = _longest(words)
    can_join = len(words) >= 2 and len(words[0] + words[1]) >= _MIN_WORD - 1
    if at is None and not can_join:
        return None
    if can_join and (at is None or rng.random() < 0.5):
        return [words[0] + words[1], *words[2:]]
    assert at is not None
    word = words[at]
    part = 3 if len(word) >= 6 else 2  # no piece shorter than this
    cut = rng.randrange(part, len(word) - part + 1)
    return [*words[:at], word[:cut], word[cut:], *words[at + 1 :]]


def _heard(kind: Kind, words: list[str], rng: random.Random) -> list[str] | None:
    """The words as they are heard for one kind of query. None if this name cannot make one."""
    if kind == "said":
        return words
    if kind == "split":
        return split_or_joined(words, rng)
    at = _longest(words)
    if at is None:
        return None
    changed = one_edit(words[at], rng) if kind == "one_edit" else by_ear(words[at], rng)
    return None if changed is None else [*words[:at], changed, *words[at + 1 :]]


def make_queries(catalog: Sequence[Medicine], per_kind: int, seed: int) -> list[Query]:
    """``per_kind`` queries of each kind, each from a different medicine. Same seed, same queries."""
    rng = random.Random(seed)
    order = list(range(len(catalog)))
    rng.shuffle(order)
    remaining = iter(order)
    queries: list[Query] = []
    for kind in KINDS:
        made = 0
        while made < per_kind:
            index = next(remaining, None)
            if index is None:
                raise ValueError(f"the catalogue is too small for {per_kind} '{kind}' queries")
            medicine = catalog[index]
            if (spoken := said(medicine.name)) is None:
                continue
            words, number = spoken
            if (heard := _heard(kind, words, rng)) is None:
                continue
            tail = [number] if number else []
            queries.append(
                Query(
                    kind=kind,
                    medicine_id=medicine.id,
                    medicine=medicine.name,
                    meant=" ".join([*words, *tail]),
                    heard=" ".join([*heard, *tail]),
                )
            )
            made += 1
    return queries


def outcome(query: Query, match: Match, products: Collection[str]) -> Outcome:
    """What the lookup's answer comes to, for the medicine the query was made from.

    ``products`` are the names of the catalogue entries the search returned. They tell a
    whole product the lookup named ("Dolo 650 Tablet") from a spelling it put together
    from the words heard ("Dolo 650").
    """
    meant = content_tokens(query.meant)

    def is_meant(name: str | None) -> bool:
        return name is not None and content_tokens(name) == meant

    if match.status == "unknown":
        return "not_found"
    if match.status == "several":
        if query.medicine in match.choices:
            return "offered"
        return "beyond_listed" if is_meant(match.name) else "wrong"
    # exact, or close: one name was given.
    if match.name == query.medicine:
        return "named"
    if match.name in products:
        # Another product, named outright. Fair only if it is called just what was meant.
        return "as_said" if is_meant(match.name) else "wrong"
    if is_meant(match.name):
        return "named"  # the catalogue's spelling of what was meant, to be confirmed
    offered = any(choice == query.medicine or is_meant(choice) for choice in match.choices)
    return "offered" if offered else "wrong"


async def run(
    store: MedicineStore,
    queries: Sequence[Query],
    *,
    candidates: int = 64,
    concurrency: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> list[Result]:
    """Look every query up as the server does: the search, then the rules. No encoders."""
    gate = asyncio.Semaphore(concurrency)
    done = 0

    async def one(query: Query) -> Result:
        nonlocal done
        async with gate:
            for attempt in range(4):
                try:
                    found = await store.search(query.heard, limit=candidates)
                    break
                except Exception:
                    # One dropped request must not end a run of thousands.
                    if attempt == 3:
                        raise
                    await asyncio.sleep(0.5 * (attempt + 1))
        match = decide(query.heard, found)
        done += 1
        if progress:
            progress(done, len(queries))
        ranks = (
            at for at, medicine in enumerate(found, start=1) if medicine.id == query.medicine_id
        )
        return Result(
            query=query,
            rank=next(ranks, None),
            status=match.status,
            outcome=outcome(query, match, {medicine.name for medicine in found}),
            answer=match.name,
        )

    return list(await asyncio.gather(*(one(query) for query in queries)))


class Tally(BaseModel):
    queries: int
    # For each cutoff k: how many had the medicine among the search's first k names.
    within: dict[int, int]
    outcomes: dict[Outcome, int]

    @property
    def resolved(self) -> int:
        """The right medicine named, or among the choices the caller is offered."""
        return self.outcomes["named"] + self.outcomes["offered"]


def tally(results: Sequence[Result]) -> Tally:
    counted = Counter(result.outcome for result in results)
    return Tally(
        queries=len(results),
        within={
            k: sum(result.rank is not None and result.rank <= k for result in results)
            for k in CUTOFFS
        },
        outcomes={name: counted[name] for name in OUTCOMES},
    )


def interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval for a share: the range the true rate plausibly lies in."""
    if total == 0:
        return 0.0, 0.0
    share = hits / total
    middle = share + z * z / (2 * total)
    spread = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total))
    scale = 1 + z * z / total
    # Rounding can leave it a hair outside the range a share can take.
    return max(0.0, (middle - spread) / scale), min(1.0, (middle + spread) / scale)


def _share(hits: int, total: int) -> str:
    return f"{hits / total:.1%}" if total else "n/a"


def _first_letter_kept(result: Result) -> bool:
    return result.query.heard[:1] == result.query.meant[:1]


def _where_misses_come_from(results: Sequence[Result]) -> list[str]:
    """The same results cut two more ways, and how many wrong answers were stated as fact."""

    def row(label: str, group: Sequence[Result]) -> str:
        counted = tally(group)
        total = counted.queries
        cells = [
            label,
            str(total),
            *(_share(counted.within[k], total) for k in CUTOFFS),
            _share(counted.resolved, total),
            _share(counted.outcomes["not_found"], total),
            _share(counted.outcomes["wrong"], total),
        ]
        return "| " + " | ".join(cells) + " |"

    recall = " | ".join(f"Recall@{k}" for k in CUTOFFS)
    head = [
        f"| | Queries | {recall} | Resolved | Not found | Wrong |",
        "|---|---|" + "---|" * (len(CUTOFFS) + 3),
    ]
    lines = [
        "## Where the misses come from",
        "",
        "**The first letter.** The lookup never changes a name to one that starts with "
        "another sound, so a name whose first letter was changed is reported as not found, "
        "or taken for another medicine.",
        "",
        *head,
    ]
    for kind in ("one_edit", "by_ear"):
        of_kind = [r for r in results if r.query.kind == kind]
        for kept in (True, False):
            group = [r for r in of_kind if _first_letter_kept(r) == kept]
            label = f"{KIND_NAMES[kind]}, first letter {'kept' if kept else 'changed'}"
            lines.append(row(label, group))
    lines += [
        "",
        "**How far the name was changed.** Names respelt by ear, by how many letters ended "
        "up different from what was meant.",
        "",
        *head,
    ]
    by_ear_results = [r for r in results if r.query.kind == "by_ear"]
    for letters in (2, 3, 4):
        group = [
            r for r in by_ear_results if min(_edits(r.query.meant, r.query.heard), 4) == letters
        ]
        if group:
            lines.append(row(f"{letters}{' or more' if letters == 4 else ''} letters", group))
    wrong = [r for r in results if r.outcome == "wrong"]
    stated = [r for r in wrong if r.status == "exact"]
    lines += [
        "",
        f"**How a wrong answer reaches the caller.** Of {len(wrong):,} wrong answers, "
        f"{len(stated):,} were stated as that medicine ({_share(len(stated), len(results))} of all "
        "queries): the changed spelling was itself another product's name. The other "
        f"{len(wrong) - len(stated):,} were put to the caller as a guess to confirm or as "
        "choices to pick from, with nothing else said about the medicine.",
        "",
    ]
    lines += [
        f"- `{r.query.heard}` (meant `{r.query.meant}`, from {r.query.medicine}): gave {r.answer}"
        for r in stated[:10]
    ]
    lines.append("")
    return lines


def render(results: Sequence[Result], *, seed: int, collection: str, names: int, day: str) -> str:
    """The report, as Markdown."""
    by_kind = {kind: tally([r for r in results if r.query.kind == kind]) for kind in KINDS}
    misheard = tally([r for r in results if r.query.kind != "said"])
    everything = tally(results)
    per_kind = by_kind["said"].queries

    def row(label: str, counted: Tally) -> str:
        low, high = interval(counted.resolved, counted.queries)
        cells = [
            label,
            str(counted.queries),
            *(_share(counted.within[k], counted.queries) for k in CUTOFFS),
            _share(counted.resolved, counted.queries),
            f"{low:.1%} to {high:.1%}",
            *(_share(counted.outcomes[name], counted.queries) for name in OUTCOMES),
        ]
        return "| " + " | ".join(cells) + " |"

    lines = [
        f"# Medicine lookup on names said wrongly ({day})",
        "",
        f"{len(results):,} queries made by rule from names in the catalogue, {per_kind:,} of "
        f"each kind, each from a different medicine. Looked up in `{collection}` "
        f"({names:,} names) by spelling and sound, without the encoders. Seed {seed}.",
        "",
        "Run it again with: `python -m app.medicines evaluate "
        f"--per-kind {per_kind} --seed {seed}`",
        "",
        "**These are not recordings of speech.** They are catalogue names changed by rule, "
        "which stands in for mishearing and is not a sample of it.",
        "",
        "| Kind | Queries | "
        + " | ".join(f"Recall@{k}" for k in CUTOFFS)
        + " | Resolved | 95% interval | Named | Offered | As said | Beyond listed | Not found | "
        "Wrong |",
        "|---|---|" + "---|" * (len(CUTOFFS) + 8),
        *(row(KIND_NAMES[kind], by_kind[kind]) for kind in KINDS),
        row("**The three misheard kinds**", misheard),
        row("**All**", everything),
        "",
        "- **Recall@k**: the medicine the query was made from is among the first k names the "
        "search returns, in the search's own order, before any rule is applied. The lookup "
        "reads the first 64.",
        "- **Resolved**: named, or offered. The lookup's rules got the caller to it.",
        "- **Named**: the lookup gave its name: the product, or the catalogue's spelling of "
        "what was meant, which the caller is asked to confirm.",
        "- **Offered**: it is among the choices, six at most, the caller is asked to pick from.",
        "- **As said**: another product is named exactly what was meant. Right by the "
        "rules, and not the one sampled.",
        "- **Beyond listed**: the brand was recognised and this product is past the six listed.",
        "- **Not found**: reported as not in the catalogue. The caller is asked to spell it.",
        "- **Wrong**: another medicine was named or offered and this one was not. The one "
        "to watch.",
        "",
        "## What a query of each kind looks like",
        "",
        "| Kind | Catalogue name | Meant | Looked up | Outcome |",
        "|---|---|---|---|---|",
    ]
    for kind in KINDS:
        for result in [r for r in results if r.query.kind == kind][:3]:
            query = result.query
            lines.append(
                f"| {KIND_NAMES[kind]} | {query.medicine} | `{query.meant}` | "
                f"`{query.heard}` | {result.outcome} |"
            )
    lines += ["", *_where_misses_come_from(results), "## Misses, a few of each", ""]
    for name in ("wrong", "not_found"):
        missed = [r for r in results if r.outcome == name and r.query.kind != "said"][:8]
        lines += [f"**{name}** ({everything.outcomes[name]:,} in all)", ""]
        lines += [
            f"- `{r.query.heard}` (meant `{r.query.meant}`, from {r.query.medicine})"
            + (f": gave {r.answer}" if r.answer else "")
            for r in missed
        ]
        lines.append("")
    return "\n".join(lines)
